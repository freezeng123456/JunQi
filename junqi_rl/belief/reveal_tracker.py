"""junqi_rl.belief.reveal_tracker — Emit piece-reveal labels on terminal states.

Bridges :meth:`junqi_rl.gpu_rollout.GpuRollout`'s ``on_termination`` callback
to :class:`junqi_rl.belief.buffer.BeliefBuffer`. When a game ends, we snapshot
the observer's view plus the ground-truth lineup of the opposing team and
push a ``(obs_spatial, seat_idx, true_type_idx, enemy_mask)`` tuple to the
buffer for belief training.

Scope (P1)
----------
**Terminal reveals only.** The existing ``on_termination`` callback fires
*after* combat resolution and *before* :meth:`reset_terminated_device`,
giving us exactly the final-state observation. Mid-game reveals (piece
X died on turn 42) would need a separate CUDA hook over combat events
and are deferred to P2.

Per Ataraxos (Appendix D.5), this is sufficient: belief loss is trained
against the "ground-truth opponent hidden pieces" at the final game state,
i.e. on a stream of *positions* each labelled with the final types of
the opponent's surviving pieces. Pieces that died earlier in the game
are not labelled in P1 — those positions would contribute via the
mid-game hook once that lands.

Geometry cheatsheet
-------------------
* Board is 17×17 = 289 cells; world coords ``(x, y) ∈ [0, 17) × [0, 17)``.
* Cell flat index ``c = y * 17 + x``.
* Seats: SOUTH=0, WEST=1, NORTH=2, EAST=3.
* Teams: {SOUTH, NORTH} vs {WEST, EAST}.
* Each seat owns 30 local slots; mapping is :func:`junqi_core.board.index_to_pos`.
* 5 of the 30 slots are *camps* (:data:`junqi_core.rules.CAMP_INDICES` — no
  piece ever sits on a camp; we emit label ``-1`` for those cells).

Vocab alignment
---------------
Arrangements are stored in the *arrangement-net vocab* (13 entries: NONE
at idx 0, JUNQI..GONGB at idx 1..12). BeliefNet outputs the *tracked-types
vocab* (12 entries: JUNQI..GONGB at idx 0..11). The mapping is simply
``belief_idx = arr_vocab_idx - 1`` when ``arr_vocab_idx > 0``, and ``-1``
sentinel otherwise.

Threading
---------
The callback is called synchronously from the GPU collector loop. We do
one ``.cpu().numpy()`` sync per terminated step (unavoidable — labels
have to reach the CPU-resident BeliefBuffer). Typical terminations
are 1-8 envs per step, so this is cheap (<0.1 ms).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Callable, TYPE_CHECKING

from junqi_core.board import index_to_pos
from junqi_core.rules import ALL_SEATS, CAMP_INDICES, SLOTS_PER_SEAT, Seat
from junqi_rl.networks.arrangement_net import (
    NONE_IDX as _ARR_NONE_IDX,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
)
from junqi_rl.networks.belief_net import N_BELIEF_TYPES
from junqi_core.info_model import TRACKED_TYPES

if TYPE_CHECKING:
    from junqi_rl.belief.buffer import BeliefBuffer
    from junqi_rl.gpu_rollout import GpuRollout


BOARD_SIZE: int = 17
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 289

# Teams: seats {0, 2} vs {1, 3}.
_TEAMMATE_SEAT = {0: 2, 2: 0, 1: 3, 3: 1}


def _build_slot_to_cell_lut() -> np.ndarray:
    """Return ``(4, 30) -> int64`` LUT of flat board cell per (seat, slot).

    For slots that are camps, returns ``-1`` (no piece ever sits there).
    """
    lut = np.full((4, SLOTS_PER_SEAT), -1, dtype=np.int64)
    for seat in ALL_SEATS:
        for i in range(SLOTS_PER_SEAT):
            if i in CAMP_INDICES:
                continue
            x, y = index_to_pos(seat, i)
            lut[int(seat), i] = y * BOARD_SIZE + x
    return lut


_SLOT_TO_CELL: np.ndarray = _build_slot_to_cell_lut()   # (4, 30) int64


def _build_arr_vocab_to_belief_idx_lut() -> np.ndarray:
    """Return ``(13,) int64`` LUT: arrangement-vocab → belief-vocab.

    Maps ``arr_vocab_idx ∈ [0, 13)`` → ``belief_idx ∈ [-1, 12)``:
        arr_vocab_idx=0 (NONE)  → -1  (sentinel)
        arr_vocab_idx=1 (JUNQI) →  0
        ...
        arr_vocab_idx=12 (GONGB)→ 11

    We don't hard-code the offset (``belief_idx = arr_vocab - 1``) in case
    the tracked-types order ever drifts from the arrangement-vocab order.
    """
    lut = np.full(13, -1, dtype=np.int64)
    belief_type_to_idx = {t: i for i, t in enumerate(TRACKED_TYPES)}
    for arr_vocab_idx in range(13):
        pt_value = VOCAB_IDX_TO_PIECE_TYPE_VALUE[arr_vocab_idx]
        from junqi_core.rules import PieceType
        pt = PieceType(pt_value)
        if pt in belief_type_to_idx:
            lut[arr_vocab_idx] = belief_type_to_idx[pt]
    return lut


_ARR_VOCAB_TO_BELIEF: np.ndarray = _build_arr_vocab_to_belief_idx_lut()   # (13,) int64


def _build_seat_cells_lut() -> np.ndarray:
    """Return ``(4, 25) int64`` of the board-cell flat indices each seat *owns*.

    "Owns" means the 25 non-camp slots belonging to that seat's territory.
    Used to construct enemy masks quickly.
    """
    out = np.zeros((4, 25), dtype=np.int64)
    for seat in ALL_SEATS:
        cells = []
        for i in range(SLOTS_PER_SEAT):
            if i in CAMP_INDICES:
                continue
            x, y = index_to_pos(seat, i)
            cells.append(y * BOARD_SIZE + x)
        assert len(cells) == 25
        out[int(seat)] = np.array(cells, dtype=np.int64)
    return out


_SEAT_CELLS: np.ndarray = _build_seat_cells_lut()   # (4, 25) int64


def _lineup_to_labels(
    arr_vocab_lineup: np.ndarray,        # (30,) int64 — arrangement-vocab
    seat_idx: int,
) -> np.ndarray:
    """Convert one seat's 30-slot lineup to a ``(289,) int64`` belief label array.

    Only cells that belong to ``seat_idx`` get a non-``-1`` value. All other
    cells are ``-1`` (unknown).

    Non-camp slots with a non-NONE piece type → ``belief_idx`` of that type.
    Camp slots (or NONE pieces, which shouldn't occur outside camps) → ``-1``.
    """
    labels = np.full(NUM_CELLS, -1, dtype=np.int64)
    for slot in range(SLOTS_PER_SEAT):
        cell = _SLOT_TO_CELL[seat_idx, slot]
        if cell < 0:
            continue
        arr_vocab = int(arr_vocab_lineup[slot])
        belief_idx = _ARR_VOCAB_TO_BELIEF[arr_vocab]
        if belief_idx >= 0:
            labels[cell] = belief_idx
    return labels


def _build_enemy_mask(observer_seat: int) -> np.ndarray:
    """Return ``(289,) bool`` mask: True where an enemy seat owns a cell.

    Enemy seats are the two not on ``observer_seat``'s team. The "owns"
    relation is structural (derived from seat geometry) and doesn't depend
    on the current game state — any enemy piece that was ever at that cell
    will have been an "enemy" for the purposes of belief training.
    """
    mask = np.zeros(NUM_CELLS, dtype=bool)
    teammate = _TEAMMATE_SEAT[observer_seat]
    enemies = [s for s in range(4) if s != observer_seat and s != teammate]
    for enemy_seat in enemies:
        mask[_SEAT_CELLS[enemy_seat]] = True
    return mask


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------


class RevealTracker:
    """Stateful bridge from ``on_termination`` callback → :class:`BeliefBuffer`.

    One instance per training run. Reusable across rollouts — the caller
    obtains a fresh :meth:`make_callback` each rollout, passing the just-
    snapshotted ``env_arr_snapshot`` (arrangements for all envs, all seats).

    Parameters
    ----------
    belief_buffer
        Target ring buffer for ``(obs, seat, label, enemy_mask)`` tuples.
        Pass ``None`` to disable insertion (callback becomes a no-op;
        useful for ablation experiments or warmup phases).
    """

    def __init__(self, belief_buffer: "BeliefBuffer | None") -> None:
        self.belief_buffer = belief_buffer
        self.n_events: int = 0      # terminal events seen (not env count)
        self.n_inserted: int = 0    # samples pushed into buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_callback(
        self,
        env_arr_snapshot: np.ndarray,
    ) -> Callable[..., None]:
        """Return a fresh callback bound to ``env_arr_snapshot``.

        Parameters
        ----------
        env_arr_snapshot
            ``(num_envs, 4, 30)`` int64 array of arrangement-vocab indices
            per (env, seat, slot). Cached once per rollout from
            :func:`junqi_rl.arrangement.pool_upload.read_env_arrangements_from_state`.

        Returns
        -------
        callable
            Signature: ``(*, fired_t, rewards_t, acting_t, rollout_world,
            **_) -> None``, matching the existing ``on_termination`` API.
        """
        if env_arr_snapshot.ndim != 3 or env_arr_snapshot.shape[1:] != (4, 30):
            raise ValueError(
                f"env_arr_snapshot must be (N, 4, 30); got {env_arr_snapshot.shape}"
            )

        def callback(
            *,
            fired_t: "torch.Tensor",
            rewards_t: "torch.Tensor",
            acting_t: "torch.Tensor",
            rollout_world: "GpuRollout",
            **_: object,
        ) -> None:
            self._emit(fired_t, acting_t, rollout_world, env_arr_snapshot)

        return callback

    # ------------------------------------------------------------------
    # Core emission logic
    # ------------------------------------------------------------------

    def _emit(
        self,
        fired_t: "torch.Tensor",
        acting_t: "torch.Tensor",
        rollout_world: "GpuRollout",
        env_arr_snapshot: np.ndarray,
    ) -> None:
        """Fetch obs for terminated envs, assemble labels, push to buffer.

        For each terminated env we emit ONE tuple from the perspective of
        ``acting_t[env_id]`` (the seat that just acted). A future revision
        could emit one tuple per observer per terminated env (4×), but
        that 4× memory cost has not yet been demonstrated to improve
        belief accuracy — keep 1× for P1.

        BUG-N fix (2026-05-11): labels now use the CURRENT piece positions
        from the GPU SoA, not the initial slot-cell mapping derived from
        ``env_arr_snapshot``. The legacy code wrote
        ``labels[_SLOT_TO_CELL[seat, slot]] = lineup_type[slot]`` which
        is correct only for the *initial* board state; after ~50 random
        moves only ~66% of labels were aligned with the obs's enemy pieces,
        and after a full game (~1000 moves) only ~22% were aligned. The
        rest were either: (a) labels at empty cells (piece moved away or
        died), or (b) cells holding a different enemy piece than the
        label claimed. BeliefNet was trained on ~78% noise.

        Now: read the GPU SoA at terminate-time (``piece_seat_arr``,
        ``piece_type_arr``, ``alive``, ``pos_x``, ``pos_y``) and label
        each enemy piece at its CURRENT cell with its TRUE type.
        ``env_arr_snapshot`` is no longer needed for label construction
        (kept in the signature for backwards-compat — callers may pass
        an empty array).
        """
        if self.belief_buffer is None:
            return

        fired_np = fired_t.detach().cpu().numpy().astype(bool)
        n_term = int(fired_np.sum())
        if n_term == 0:
            return

        self.n_events += n_term

        # Build obs for *all* envs whose seat is acting_t (GPU kernel); we
        # then slice out the terminated-env rows on the host.
        # ``build_acting_seat_observation_torch`` is idempotent / cheap to
        # re-run — the gpu_collector already called it once for this step,
        # but it's not cached anywhere the callback can reach. Calling it
        # again is ~0.5 ms for 128 envs; negligible vs the belief forward
        # we'll eventually do on these samples.
        obs_sp_t, _ = rollout_world.build_acting_seat_observation_torch(acting_t)

        fired_idx = np.where(fired_np)[0]
        # Pull obs for only the terminated envs (subselect on GPU → copy).
        fired_idx_t = torch.from_numpy(fired_idx).to(
            device=obs_sp_t.device, dtype=torch.long,
        )
        obs_sp_term = obs_sp_t.index_select(0, fired_idx_t)       # (K, C, 17, 17)
        obs_np = obs_sp_term.detach().cpu().numpy().astype(np.float32)

        acting_np = acting_t.detach().cpu().numpy().astype(np.int64)

        # BUG-N fix: pull the GPU SoA so we know each piece's *current*
        # location and type, not the initial-slot mapping. One D2H per
        # rollout-step that has any termination; typically 0-8 envs fire,
        # so this is cheap (a few KB per array).
        # Layout per copy_to_host: dict of flat (N*120,) or (N*K,) arrays.
        soa = rollout_world.state.copy_to_host()
        N_total = rollout_world.num_envs
        piece_seat = np.asarray(soa["piece_seat_arr"]).reshape(N_total, 120)
        piece_type = np.asarray(soa["piece_type_arr"]).reshape(N_total, 120)
        pos_x      = np.asarray(soa["pos_x"]).reshape(N_total, 120)
        pos_y      = np.asarray(soa["pos_y"]).reshape(N_total, 120)
        alive      = np.asarray(soa["alive"]).reshape(N_total, 120).astype(bool)

        # Vocab translation: PieceType.value (engine domain) → belief idx.
        # PieceType.value range: 0=NONE, 2=JUNQI, 3=DILEI, 4=ZHADAN, 5..13=SILING..GONGB.
        # belief idx 0..11 covers the 12 TRACKED_TYPES (JUNQI..GONGB).
        # Build a small LUT once and reuse:
        if not hasattr(self, "_pt_value_to_belief"):
            from junqi_core.rules import PieceType
            from junqi_core.info_model import TRACKED_TYPES
            lut = np.full(16, -1, dtype=np.int64)
            for i, tracked_pt in enumerate(TRACKED_TYPES):
                lut[int(tracked_pt.value)] = i
            self._pt_value_to_belief = lut

        # Assemble per-env labels.
        K = len(fired_idx)
        seat_out = np.zeros(K, dtype=np.int64)
        labels_out = np.full((K, NUM_CELLS), -1, dtype=np.int64)
        enemy_out = np.zeros((K, NUM_CELLS), dtype=bool)

        for out_i, env_id in enumerate(fired_idx):
            observer_seat = int(acting_np[env_id])
            seat_out[out_i] = observer_seat
            # BUG-N (continued): enemy_mask MUST track current piece cells,
            # not the static seat-geometry mask. An enemy piece that has
            # walked into observer's own territory is still an enemy whose
            # type we want to learn; an enemy initial slot whose piece has
            # since moved away is no longer an enemy cell. We rebuild the
            # mask from the GPU SoA below alongside the labels.
            obs_team = observer_seat & 1
            # Walk all 120 pieces; label each alive enemy piece at its
            # current cell with its true type. (Camp slots cannot host
            # pieces, so we don't need a separate camp mask.)
            ps_e = piece_seat[env_id]   # (120,)
            pt_e = piece_type[env_id]
            px_e = pos_x[env_id]
            py_e = pos_y[env_id]
            al_e = alive[env_id]
            for pid in range(120):
                if not al_e[pid]:
                    continue
                seat = int(ps_e[pid])
                if seat < 0 or (seat & 1) == obs_team:
                    continue   # observer's own team or invalid
                x = int(px_e[pid])
                y = int(py_e[pid])
                if x < 0 or y < 0:
                    continue
                cell = y * BOARD_SIZE + x
                # Mark cell as enemy-occupied (BUG-N fix).
                enemy_out[out_i, cell] = True
                belief_idx = int(self._pt_value_to_belief[int(pt_e[pid])])
                if belief_idx >= 0:
                    labels_out[out_i, cell] = belief_idx

        self.belief_buffer.add(
            obs_spatial=obs_np,
            seat_idx=seat_out,
            true_type_idx=labels_out,
            enemy_mask=enemy_out,
        )
        self.n_inserted += K

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return logging scalars."""
        return {
            "reveal_tracker/n_events": float(self.n_events),
            "reveal_tracker/n_inserted": float(self.n_inserted),
        }


__all__ = ["RevealTracker", "NUM_CELLS", "BOARD_SIZE"]
