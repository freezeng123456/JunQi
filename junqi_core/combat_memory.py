"""CombatMemory v6 — DARK-mode high-order combat memory.

Per docs/COMBAT_MEMORY_DESIGN.md (v4) + ADR-129 (v5 layer-3) + v6 reverse
projection (this file).

DARK-mode information boundary
-------------------------------
* Each observer can see only their own seat's piece types.
* Movements (src/dst), event kinds (EAT/KILLED/BOMB), and piece_ids are
  all public.
* Path-only-GONGB moves are public (every observer can see the geometry).

State per (observer × piece_id), shape (4, 120)
-----------------------------------------------
Direct memory (only when ``victim_seat == observer``):
    direct_ate_my_pid_lo / direct_ate_my_pid_hi : uint64 — pid bitmap
    direct_ate_my_type_mask                     : uint16 — 12 type bits
    last_direct_step                            : int16

Direct counter for non-observer victims (DARK: type unknown):
    direct_other_count                          : int16 — count only

Chain memory (all observers; chain ops act on public piece_ids):
    chain_pid_lo / chain_pid_hi                 : uint64 — full 120-pid bitmap
    chain_ate_my_type_mask                      : uint16 — only my-pid types
    last_chain_step                             : int16

Reverse memory (v6, victim-anchored; only writes for observer's own pids):
    eaten_by_pid_lo / eaten_by_pid_hi           : uint64 — bitmap of pids
        (any seat) that have directly OR chain-eaten ``mpid`` from
        observer's perspective.  DARK-safe because we only write when
        ``mpid`` is observer's own piece (V_seat == observer for the
        triggering EAT/KILLED, or mpid was previously listed in V's
        direct_my/chain bitmap restricted to observer's own pid range).

Ordinary rank floor:
    rank_floor                                  : int8 ∈ [0, 9]; 0 = unknown
    rank_floor_step                             : int16

GONGB flags:
    is_gongb                                    : bool — set on
        (A1) ate observer's DILEI (Event.EAT, victim_seat == observer), or
        (A2) walked a GONGB-only path (any observer)
    not_gongb                                   : bool — set on
        (B1) ate observer's non-DILEI piece (Event.EAT, victim_seat == observer), or
        (B2) chain-killed an ordinary-rank piece (rank_floor lift implies
             non-GONGB)

Defender history (for runtime dilei_candidate):
    attacked_by_known_gongb                     : bool — set on Event.KILLED
        where the attacker (now dead) was known to be GONGB at time of
        attack (own seat + actual type, or previously is_gongb).

DILEI candidate (runtime, NOT stored):
    dilei_candidate[obs][pid] = (
        alive[pid] AND
        zero_pos[pid] in seat-of-pid's back-two-rows AND
        move_count_arr[pid] == 0 AND
        NOT attacked_by_known_gongb[obs][pid]
    )

Channel layout (now 110 ch in observation.py, projecting state to canvas)
-------------------------------------------------------------------------
Layer 1 (45 ch, projected to enemy alive pieces in observer view):
    cm_kill_mine_type      [12]  multi-hot of victim types I lost to this enemy
    cm_kill_mine_ge        [3]   ≥1, ≥2, ≥3 (counted via popcount of my-pid bitmap)
    cm_kill_other_ge       [3]   ≥1, ≥2, ≥3 (direct_other_count thresholds)
    cm_chain_type          [12]  chain my-piece types
    cm_chain_ge            [3]   ≥1, ≥2, ≥3 (chain_pid bitmap popcount)
    cm_floor_ge            [9]   ≥GONGB ... ≥SILING cumulative
    cm_is_gongb            [1]
    cm_not_gongb           [1]
    cm_dilei_candidate     [1]   runtime computed

Layer 2 (5 ch, theory-of-mind, projected to my own alive pieces):
    cm_my_kill_count_ge    [3]   any single opponent's view of my kill count
    cm_my_is_gongb         [1]   AND of two opponents' is_gongb (= path-revealed only)
    cm_my_dilei_candidate  [1]   based on AND of two opponents' attacked_by_known_gongb

Layer 3 (46 ch, ADR-129 v5; projected to enemy alive pieces):
    cm_kill_mine_count     [12]  per-type COUNT (popcount of direct ∩ type mask)/3
    cm_kill_mine_slot      [30]  observer slot-i bit at killer cell
    cm_recency             [4]   sigmoid(elapsed/τ) signals

Layer 4 (60 ch, v6; projected to observer's own pieces — alive cell or zero_pos):
    cm_eaten_by_pid        [60]  for observer's mpid at (x,y), set bit k iff
                                 enemy pid (left_opp's slot k or right_opp's
                                 slot k-30) has directly OR chain-eaten mpid.
                                 Channels 0..29 = (observer+1)%4 (left enemy);
                                 channels 30..59 = (observer+3)%4 (right enemy).
                                 DARK-safe: only writes for observer's own
                                 mpid (V_seat == observer chain-restricted).

OBS_CHANNELS: 256 + 50 + 46 + 60 = 412.

This module is the CPU reference; ``BatchedGameState`` and CUDA kernels
must replicate the same updates byte-identically (CUDA layer-4 kernel
implementation pending — see src/env/cuda/src/combat_memory.cu TODOs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .rules import PieceType, Seat

# ===========================================================================
# Vocabulary
# ===========================================================================

# 12 tracked piece types in fixed order (must match observation.TRACKED_TYPES).
TRACKED_TYPES: Final[tuple[PieceType, ...]] = (
    PieceType.JUNQI,    # 0
    PieceType.DILEI,    # 1
    PieceType.ZHADAN,   # 2
    PieceType.SILING,   # 3
    PieceType.JUNZH,    # 4
    PieceType.SHIZH,    # 5
    PieceType.LVZH,     # 6
    PieceType.TUANZH,   # 7
    PieceType.YINGZH,   # 8
    PieceType.LIANZH,   # 9
    PieceType.PAIZH,    # 10
    PieceType.GONGB,    # 11
)
NUM_TRACKED_TYPES: Final[int] = 12

_PIECETYPE_TO_TRACKED_IDX: Final[np.ndarray] = (
    lambda: (
        a := np.full(max(p.value for p in PieceType) + 1, -1, dtype=np.int8),
        [a.__setitem__(t.value, i) for i, t in enumerate(TRACKED_TYPES)],
        a,
    )[2]
)()

# Ordinary rank ladder (weakest → strongest), used for floor lifts.
ORDINARY_RANKS: Final[tuple[PieceType, ...]] = (
    PieceType.GONGB,    # rank 1
    PieceType.PAIZH,    # 2
    PieceType.LIANZH,   # 3
    PieceType.YINGZH,   # 4
    PieceType.TUANZH,   # 5
    PieceType.LVZH,     # 6
    PieceType.SHIZH,    # 7
    PieceType.JUNZH,    # 8
    PieceType.SILING,   # 9
)
NUM_ORDINARY_RANKS: Final[int] = 9
RANK_FLOOR_UNKNOWN: Final[int] = 0

# PieceType.value -> ordinary rank in [1, 9]; 0 if not ordinary.
_RANK_OF_TYPE: Final[np.ndarray] = (
    lambda: (
        a := np.zeros(max(p.value for p in PieceType) + 1, dtype=np.int8),
        [a.__setitem__(t.value, i + 1) for i, t in enumerate(ORDINARY_RANKS)],
        a,
    )[2]
)()


def next_floor_after_eat(victim_type: PieceType) -> int:
    """Lower bound on the killer's ordinary rank floor after EAT-ing this victim.

    Specials (JUNQI / DILEI / ZHADAN) return 0 (no ordinary-floor inference).
    SILING victim returns 9 (capped — ZHADAN-vs-SILING goes through
    Event.BOMB which is a separate path).
    """
    rank = int(_RANK_OF_TYPE[victim_type.value])
    if rank == 0:
        return RANK_FLOOR_UNKNOWN
    return min(rank + 1, NUM_ORDINARY_RANKS)


# ===========================================================================
# Geometry helpers
# ===========================================================================


def is_in_seat_back_two_rows(pos_flat: int, owner_seat: int) -> bool:
    """Is ``pos_flat`` (world-frame) in ``owner_seat``'s own back two rows?

    Back two rows house the 5 slots that may legally hold a DILEI (per
    setup constraints C1-C5). For SOUTH that's world y ∈ {15, 16};
    WEST x ∈ {0, 1}; NORTH y ∈ {0, 1}; EAST x ∈ {15, 16}.
    """
    y = pos_flat // 17
    x = pos_flat % 17
    if owner_seat == Seat.SOUTH.value:
        return y >= 15
    if owner_seat == Seat.NORTH.value:
        return y <= 1
    if owner_seat == Seat.WEST.value:
        return x <= 1
    if owner_seat == Seat.EAST.value:
        return x >= 15
    return False


# ===========================================================================
# Bitmap helpers (pid ∈ [0, 120) packed into uint64 lo/hi)
# ===========================================================================

PID_LO_LIMIT: Final[int] = 64


def _pid_bit(pid: int) -> tuple[bool, np.uint64]:
    """Return (is_high_half, mask) for setting/testing pid bit."""
    if pid < PID_LO_LIMIT:
        return False, np.uint64(1) << np.uint64(pid)
    return True, np.uint64(1) << np.uint64(pid - PID_LO_LIMIT)


def _set_pid_bit(lo_arr: np.ndarray, hi_arr: np.ndarray, pid: int) -> None:
    """Set bit ``pid`` in the (lo_arr, hi_arr) uint64 pair (scalar slot)."""
    is_hi, mask = _pid_bit(pid)
    if is_hi:
        hi_arr |= mask  # may not work for 0-d; caller passes 0-d slots
    else:
        lo_arr |= mask


# Per-observer 30-pid bitmasks for "observer's own pids" (used by reverse
# projection in v6).  Pre-computed once.  Layout: pids [obs*30, obs*30+30).
def _build_obs_pid_masks() -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(4, dtype=np.uint64)
    hi = np.zeros(4, dtype=np.uint64)
    for obs in range(4):
        for s in range(30):
            gp = obs * 30 + s
            if gp < PID_LO_LIMIT:
                lo[obs] |= np.uint64(1) << np.uint64(gp)
            else:
                hi[obs] |= np.uint64(1) << np.uint64(gp - PID_LO_LIMIT)
    return lo, hi


_OBS_PID_MASKS: Final[tuple[np.ndarray, np.ndarray]] = _build_obs_pid_masks()
_OBS_PID_MASK_LO, _OBS_PID_MASK_HI = _OBS_PID_MASKS


# ===========================================================================
# CombatMemoryState dataclass
# ===========================================================================

NUM_OBSERVERS: Final[int] = 4
NUM_PIDS: Final[int] = 120


@dataclass(slots=True)
class CombatMemoryState:
    """Per-(observer × piece_id) DARK-mode combat memory.

    All arrays have shape ``(4, 120)``.  Observer axis 0 is by Seat.value
    order (SOUTH=0, WEST=1, NORTH=2, EAST=3).
    """

    # --- direct memory (victim_seat == observer) ---
    direct_ate_my_pid_lo:    np.ndarray   # uint64
    direct_ate_my_pid_hi:    np.ndarray   # uint64
    direct_ate_my_type_mask: np.ndarray   # uint16
    last_direct_step:        np.ndarray   # int16

    # --- direct counter for "non-my" victims (DARK: type unknown) ---
    direct_other_count:      np.ndarray   # int16

    # --- chain memory (all observers; full 120-pid bitmap) ---
    chain_pid_lo:            np.ndarray   # uint64
    chain_pid_hi:            np.ndarray   # uint64
    chain_ate_my_type_mask:  np.ndarray   # uint16  (my-pid types only)
    last_chain_step:         np.ndarray   # int16

    # --- reverse projection (v6, victim-anchored; DARK-safe) ---
    # eaten_by_pid[obs, mpid] = bitmap of pids (any seat) that have
    # directly OR chain-eaten ``mpid`` from observer's perspective.
    # Writes are only made for observer's own mpid; cells for other
    # seats stay zero.
    eaten_by_pid_lo:         np.ndarray   # uint64
    eaten_by_pid_hi:         np.ndarray   # uint64

    # --- ordinary rank floor ---
    rank_floor:              np.ndarray   # int8 ∈ [0, 9]
    rank_floor_step:         np.ndarray   # int16

    # --- GONGB / DILEI deduction flags ---
    is_gongb:                np.ndarray   # bool
    not_gongb:               np.ndarray   # bool
    attacked_by_known_gongb: np.ndarray   # bool

    @classmethod
    def zeros(cls) -> CombatMemoryState:
        sh = (NUM_OBSERVERS, NUM_PIDS)
        return cls(
            direct_ate_my_pid_lo    = np.zeros(sh, dtype=np.uint64),
            direct_ate_my_pid_hi    = np.zeros(sh, dtype=np.uint64),
            direct_ate_my_type_mask = np.zeros(sh, dtype=np.uint16),
            last_direct_step        = np.full(sh, -1, dtype=np.int16),
            direct_other_count      = np.zeros(sh, dtype=np.int16),
            chain_pid_lo            = np.zeros(sh, dtype=np.uint64),
            chain_pid_hi            = np.zeros(sh, dtype=np.uint64),
            chain_ate_my_type_mask  = np.zeros(sh, dtype=np.uint16),
            last_chain_step         = np.full(sh, -1, dtype=np.int16),
            eaten_by_pid_lo         = np.zeros(sh, dtype=np.uint64),
            eaten_by_pid_hi         = np.zeros(sh, dtype=np.uint64),
            rank_floor              = np.zeros(sh, dtype=np.int8),
            rank_floor_step         = np.full(sh, -1, dtype=np.int16),
            is_gongb                = np.zeros(sh, dtype=bool),
            not_gongb               = np.zeros(sh, dtype=bool),
            attacked_by_known_gongb = np.zeros(sh, dtype=bool),
        )

    def clone(self) -> CombatMemoryState:
        return CombatMemoryState(
            direct_ate_my_pid_lo    = self.direct_ate_my_pid_lo.copy(),
            direct_ate_my_pid_hi    = self.direct_ate_my_pid_hi.copy(),
            direct_ate_my_type_mask = self.direct_ate_my_type_mask.copy(),
            last_direct_step        = self.last_direct_step.copy(),
            direct_other_count      = self.direct_other_count.copy(),
            chain_pid_lo            = self.chain_pid_lo.copy(),
            chain_pid_hi            = self.chain_pid_hi.copy(),
            chain_ate_my_type_mask  = self.chain_ate_my_type_mask.copy(),
            last_chain_step         = self.last_chain_step.copy(),
            eaten_by_pid_lo         = self.eaten_by_pid_lo.copy(),
            eaten_by_pid_hi         = self.eaten_by_pid_hi.copy(),
            rank_floor              = self.rank_floor.copy(),
            rank_floor_step         = self.rank_floor_step.copy(),
            is_gongb                = self.is_gongb.copy(),
            not_gongb               = self.not_gongb.copy(),
            attacked_by_known_gongb = self.attacked_by_known_gongb.copy(),
        )


# ===========================================================================
# Update API
# ===========================================================================


def apply_path_revealed_gongb(cm: CombatMemoryState, pid: int) -> None:
    """Mark ``pid`` as a publicly-revealed GONGB.

    Triggered by the engine when a piece walks a path that ONLY a GONGB
    can legally take (curve rail / multi-hop rail BFS / etc.).  Update is
    written to ALL four observers because the geometry is public.
    """
    cm.is_gongb[:, pid] = True


def apply_combat_event(
    cm: CombatMemoryState,
    *,
    event_is_eat: bool,         # True for EAT (defender dies); False for KILLED (attacker dies)
    attacker_pid: int,
    defender_pid: int,
    attacker_seat: int,
    defender_seat: int,
    attacker_type: PieceType,
    defender_type: PieceType,
    defender_pos_flat: int,
    death_step: int,
) -> None:
    """Apply one EAT-or-KILLED combat event to ``cm`` in place.

    BOMB events are NOT routed here (both pieces die — no live target to
    project onto, no chain to propagate).  The caller in
    ``state.py::step_inplace`` skips BOMB.

    Normalised viewpoint:
      * K = the survivor (winner)
      * V = the dead piece
      For EAT:    K = attacker, V = defender (defender dies)
      For KILLED: K = defender, V = attacker (attacker dies)

    Per-observer dispatch on whether V is visible to obs (DARK: only the
    obs == V_seat case).  Plus chain propagation (all observers) and
    ``attacked_by_known_gongb`` flag for KILLED events.
    """
    # Dispatch K / V
    if event_is_eat:
        K, V = attacker_pid, defender_pid
        V_seat = defender_seat
        V_type = defender_type
    else:
        # KILLED
        K, V = defender_pid, attacker_pid
        V_seat = attacker_seat
        V_type = attacker_type

    if K < 0 or V < 0:
        return

    v_idx = int(_PIECETYPE_TO_TRACKED_IDX[V_type.value])  # may be -1

    # --- KILLED-event preflight: was the attacker (V) known GONGB? ---
    # We compute this BEFORE the chain/direct updates so it doesn't
    # depend on V's just-cleared chain state.  For each observer:
    #   v_known_gongb[obs] = (V_seat == obs AND V_type == GONGB)
    #                        OR cm.is_gongb[obs, V] (from earlier path-reveal)
    if not event_is_eat:  # KILLED
        v_known_gongb = cm.is_gongb[:, V].copy()
        if attacker_type is PieceType.GONGB:
            # Observer == V_seat (attacker.seat) knows their own type
            v_known_gongb[V_seat] = True
        # Set defender (K)'s flag for any observer who saw a known-GONGB attack.
        cm.attacked_by_known_gongb[v_known_gongb, K] = True

    # --- Chain propagation (all observers) ---
    # chain[K] |= direct_my[V] | chain[V] | {V}
    cm.chain_pid_lo[:, K] |= cm.direct_ate_my_pid_lo[:, V]
    cm.chain_pid_hi[:, K] |= cm.direct_ate_my_pid_hi[:, V]
    cm.chain_pid_lo[:, K] |= cm.chain_pid_lo[:, V]
    cm.chain_pid_hi[:, K] |= cm.chain_pid_hi[:, V]
    # Add V itself to chain bitmap.
    is_hi_v, mask_v = _pid_bit(V)
    if is_hi_v:
        cm.chain_pid_hi[:, K] |= mask_v
    else:
        cm.chain_pid_lo[:, K] |= mask_v
    # chain_type: include V's existing direct_my and chain types
    cm.chain_ate_my_type_mask[:, K] |= cm.direct_ate_my_type_mask[:, V]
    cm.chain_ate_my_type_mask[:, K] |= cm.chain_ate_my_type_mask[:, V]
    cm.last_chain_step[:, K] = death_step

    # Chain rank-floor propagation: K's floor ≥ max(K, V.floor + 1).
    v_floor_per_obs = cm.rank_floor[:, V]
    proposed = np.minimum(v_floor_per_obs + 1, NUM_ORDINARY_RANKS)
    has_floor = v_floor_per_obs > RANK_FLOOR_UNKNOWN
    update_mask = has_floor & (proposed > cm.rank_floor[:, K])
    cm.rank_floor[update_mask, K] = proposed[update_mask]
    cm.rank_floor_step[update_mask, K] = death_step
    # Chain killer of an ordinary-rank piece can't be GONGB.
    cm.not_gongb[has_floor, K] = True

    # --- v6 reverse projection: eaten_by_pid (DARK-safe) ---
    # For each observer, identify which of observer's own mpids have
    # just become "eaten by K" — either because:
    #   (a) V itself is observer's own piece (V_seat == obs), or
    #   (b) V had previously direct-eaten or chain-bridged some of
    #       observer's own pids (V's chain or direct_my masks intersected
    #       with observer's own pid range).
    # We OR K's pid bit into eaten_by_pid[obs, mpid] for each such mpid.
    #
    # DARK boundary: by masking with _OBS_PID_MASK_*, we never write
    # for non-observer-own mpids.  Since direct_ate_my_pid is only ever
    # populated for V_seat == observer (see per-observer dispatch
    # below), and chain_pid bits are public piece_ids, the AND with
    # observer's own range gives exactly the visible "I lost mpid via
    # K's chain/direct" set.
    is_hi_k, k_mask = _pid_bit(K)
    for obs in range(NUM_OBSERVERS):
        # V's "ate observer's pids" set, computed BEFORE this event's
        # K-side chain update (we read direct_ate_my[V] and chain[V],
        # neither of which has been mutated yet for V).
        v_my_lo = (
            cm.direct_ate_my_pid_lo[obs, V] | cm.chain_pid_lo[obs, V]
        ) & _OBS_PID_MASK_LO[obs]
        v_my_hi = (
            cm.direct_ate_my_pid_hi[obs, V] | cm.chain_pid_hi[obs, V]
        ) & _OBS_PID_MASK_HI[obs]
        # Plus V itself if V is observer's own pid (DARK rule: only
        # observer can know V's seat == obs, here V_seat is public so
        # this is equivalent to V_seat == obs).
        if V_seat == obs:
            is_hi_v, mask_v = _pid_bit(V)
            if is_hi_v:
                v_my_hi |= mask_v
            else:
                v_my_lo |= mask_v
        # Iterate set bits (observer's pid range is at most 30 bits).
        bits_lo = int(v_my_lo)
        while bits_lo:
            b = bits_lo & -bits_lo
            target_pid = b.bit_length() - 1
            if is_hi_k:
                cm.eaten_by_pid_hi[obs, target_pid] |= k_mask
            else:
                cm.eaten_by_pid_lo[obs, target_pid] |= k_mask
            bits_lo ^= b
        bits_hi = int(v_my_hi)
        while bits_hi:
            b = bits_hi & -bits_hi
            target_pid = (b.bit_length() - 1) + PID_LO_LIMIT
            if is_hi_k:
                cm.eaten_by_pid_hi[obs, target_pid] |= k_mask
            else:
                cm.eaten_by_pid_lo[obs, target_pid] |= k_mask
            bits_hi ^= b

    # --- Per-observer dispatch on V visibility (DARK rule) ---
    for obs in range(NUM_OBSERVERS):
        v_visible = (V_seat == obs)
        if not v_visible:
            # Type unknown: count only.
            cm.direct_other_count[obs, K] = min(
                int(cm.direct_other_count[obs, K]) + 1, 32767
            )
            cm.last_direct_step[obs, K] = death_step
            # If V was observer's own piece's chain ancestor, chain_type
            # already updated above — no further work.
            continue

        # V is observer's own piece — full type-aware update.
        # 1) Direct bitmap and type-mask
        is_hi, mask = _pid_bit(V)
        if is_hi:
            cm.direct_ate_my_pid_hi[obs, K] |= mask
        else:
            cm.direct_ate_my_pid_lo[obs, K] |= mask
        if v_idx >= 0:
            cm.direct_ate_my_type_mask[obs, K] |= np.uint16(1 << v_idx)
            # Also propagate V's type into K's chain_type (V is a "my piece"
            # link in K's chain).
            cm.chain_ate_my_type_mask[obs, K] |= np.uint16(1 << v_idx)
        cm.last_direct_step[obs, K] = death_step

        # 2) GONGB / non-GONGB rules — V's type drives this.
        # Critical DARK rules:
        #   - EAT, V == DILEI (my mine eaten alive) → K is GONGB
        #   - EAT, V != DILEI → K is NOT GONGB (engineers only beat mines)
        #   - KILLED, V's type known → K could be anything that beats V's
        #     type; ordinary-rank floor lifts handle this. K cannot be
        #     GONGB iff V is ordinary-rank (handled by chain block above
        #     when V's floor matters; but for direct, we set explicitly).
        if event_is_eat:
            if V_type is PieceType.DILEI:
                cm.is_gongb[obs, K] = True
            else:
                # GONGB cannot win EAT vs non-DILEI.
                cm.not_gongb[obs, K] = True
        else:
            # KILLED: K is defender; V is attacker (now dead).
            # V_type is attacker's type, visible to obs since V_seat == obs.
            # The defender K (still alive) beat V.  Floor lift on K:
            if V_type.is_ranked_combatant:
                # K beat V → K's rank > V's rank → floor = next_floor(V)
                promoted = next_floor_after_eat(V_type)
                if promoted > int(cm.rank_floor[obs, K]):
                    cm.rank_floor[obs, K] = promoted
                    cm.rank_floor_step[obs, K] = death_step
                # K beat an ordinary attacker non-trivially → K is not GONGB
                # UNLESS attacker was GONGB (then defender could be DILEI ...
                # but DILEI is immobile, can't be defender attacking).
                # In KILLED, defender survives — defender could be DILEI
                # if attacker was a non-GONGB ordinary.  For non-GONGB
                # attacker → defender survives because either DILEI or
                # higher rank.  We can't pin K to "not GONGB" here because
                # K could be DILEI in this case.  Skip not_gongb assertion.
                pass
            elif V_type is PieceType.ZHADAN:
                # ZHADAN attacker → KILLED would require defender to be...
                # actually ZHADAN-vs-anything is BOMB, not KILLED.  This
                # branch shouldn't fire.
                pass
            elif V_type is PieceType.JUNQI:
                # JUNQI attacking is impossible (immobile).  Defensive.
                pass
            # else (NONE / DARK): unreachable

        # 3) EAT-direct floor lift on K.
        if event_is_eat and V_type.is_ranked_combatant:
            promoted = next_floor_after_eat(V_type)
            if promoted > int(cm.rank_floor[obs, K]):
                cm.rank_floor[obs, K] = promoted
                cm.rank_floor_step[obs, K] = death_step


# ===========================================================================
# Self-test
# ===========================================================================


if __name__ == "__main__":
    cm = CombatMemoryState.zeros()
    assert cm.direct_ate_my_pid_lo.shape == (4, 120)
    assert cm.is_gongb.dtype == bool
    assert next_floor_after_eat(PieceType.PAIZH) == 3       # LIANZH+
    assert next_floor_after_eat(PieceType.SHIZH) == 8       # JUNZH+
    assert next_floor_after_eat(PieceType.SILING) == 9      # capped
    assert next_floor_after_eat(PieceType.JUNQI) == 0
    assert next_floor_after_eat(PieceType.DILEI) == 0
    assert next_floor_after_eat(PieceType.ZHADAN) == 0
    # Geometry
    assert is_in_seat_back_two_rows(15 * 17 + 8, Seat.SOUTH.value)
    assert not is_in_seat_back_two_rows(8 * 17 + 8, Seat.SOUTH.value)
    assert is_in_seat_back_two_rows(0, Seat.NORTH.value)
    assert is_in_seat_back_two_rows(8 * 17 + 0, Seat.WEST.value)
    assert is_in_seat_back_two_rows(8 * 17 + 16, Seat.EAST.value)

    # Roundtrip: SOUTH's PAIZH eaten by an enemy (Event.EAT victim=PAIZH).
    apply_combat_event(
        cm,
        event_is_eat=True,
        attacker_pid=37, defender_pid=12,
        attacker_seat=Seat.WEST.value, defender_seat=Seat.SOUTH.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=10 * 17 + 6, death_step=42,
    )
    south = Seat.SOUTH.value
    # Direct bitmap for SOUTH: bit 12 set.
    assert cm.direct_ate_my_pid_lo[south, 37] == np.uint64(1) << np.uint64(12)
    paizh_idx = int(_PIECETYPE_TO_TRACKED_IDX[PieceType.PAIZH.value])
    assert cm.direct_ate_my_type_mask[south, 37] == np.uint16(1 << paizh_idx)
    assert cm.rank_floor[south, 37] == 3                # LIANZH+
    assert cm.not_gongb[south, 37]                       # ate non-mine
    assert not cm.is_gongb[south, 37]
    # Other observers: only count
    assert cm.direct_other_count[Seat.WEST.value, 37] == 1
    assert cm.direct_other_count[Seat.NORTH.value, 37] == 1
    assert cm.direct_other_count[Seat.EAST.value, 37] == 1
    assert cm.direct_ate_my_type_mask[Seat.WEST.value, 37] == 0
    # v6 reverse projection: SOUTH should now know that mpid=12 was eaten
    # by enemy pid 37; all other pids should be unset.
    is_hi_37, mask_37 = _pid_bit(37)
    if is_hi_37:
        assert cm.eaten_by_pid_hi[south, 12] & mask_37 == mask_37
    else:
        assert cm.eaten_by_pid_lo[south, 12] & mask_37 == mask_37
    # Other observers must NOT know what types SOUTH lost (DARK boundary):
    # but they can know V was eaten by K (since the event itself is public);
    # however, reverse projection is observer-anchored on observer's own
    # pids, and 12 is SOUTH's pid not WEST/NORTH/EAST's, so:
    assert cm.eaten_by_pid_lo[Seat.WEST.value, 12] == 0
    assert cm.eaten_by_pid_hi[Seat.WEST.value, 12] == 0
    assert cm.eaten_by_pid_lo[Seat.NORTH.value, 12] == 0
    assert cm.eaten_by_pid_hi[Seat.NORTH.value, 12] == 0
    assert cm.eaten_by_pid_lo[Seat.EAST.value, 12] == 0
    assert cm.eaten_by_pid_hi[Seat.EAST.value, 12] == 0
    print("junqi_core.combat_memory v6 self-test: OK")
