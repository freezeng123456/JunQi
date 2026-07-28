"""End-to-end DARK-mode visibility invariant tests.

Asserts the canonical 四暗 (four-dark) information contract:

* observer's ``piece_own`` channels reveal type ONLY for own pieces
  (no teammate / no enemy type bleed-through).
* under ``ShowMode.DARK``: teammate type is NOT one-hot in observer's
  belief — it stays under the per-slot prior (or the network-inferred
  posterior).
* enemy type is NEVER one-hot in observer's belief, with these
  rule-allowed exceptions:
    - ``BeliefTensor`` may set the *flag's* position to one-hot JUNQI on
      stronghold-EAT deduction (R6 in ``info_model.py``).
    - ``BeliefTensor`` may set an attacker to one-hot GONGB after eating
      a defending DILEI (R5/R7).
    - The seat-level ``flag_revealed[seat]`` planes turn on after a
      SILING death (Q7).
  In all other cases enemy belief is a non-degenerate distribution.
* ``MoveResult.{src_type_revealed, dst_type_revealed}`` is ``None`` under
  DARK / HALF_DARK (Q2 — combat info broadcast strips piece types).
* SILING-death events DO toggle ``state.seat_flag_revealed_arr[seat]``
  for each side whose SILING died — the only public type leak.

These checks run against the CPU reference engine (``junqi_core``) so
they are torch-free and fast.  They sit between the rule golden tests
(``test_combat_rules.py``) and the GPU parity tests
(``test_gpu_obs_parity.py``) which assert the CUDA kernel mirrors the
CPU reference byte-for-byte.

If any of these tests fail the model would be receiving illegal
information during training (``cheating''), which would invalidate the
entire 四暗 RL goal — so they are precisely the tests that need to be
green to claim "the visibility model is correct".
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor, _is_one_hot
from junqi_core.observation import (
    CHANNEL_LAYOUT,
    NUM_TRACKED_TYPES,
    ObservationBuilder,
    TRACKED_TYPES,
)
from junqi_core.rules import ALL_SEATS, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState

PIECETYPE_TO_TRACKED_IDX = {pt: i for i, pt in enumerate(TRACKED_TYPES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_game(seed: int = 0, *, show_mode: ShowMode = ShowMode.DARK) -> GameState:
    rng = random.Random(seed)
    setups = generate_random_setup(rng)
    return GameState.new_game(setups, show_mode=show_mode)


def _build_obs(state: GameState, observer: Seat):
    belief = BeliefTensor.initial(state, observer, show_mode=state.show_mode)
    builder = ObservationBuilder()
    obs = builder.build(state, belief, observer)
    return obs, belief


# ---------------------------------------------------------------------------
# 1.  piece_own contains only the observer's own pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_piece_own_contains_only_own_pieces(observer: Seat):
    """The 12-plane piece_own block must light up at exactly the observer's
    own piece cells, with the correct type, and nowhere else."""
    state = _new_game(seed=42, show_mode=ShowMode.DARK)
    obs, _ = _build_obs(state, observer)

    own_block = obs.spatial[CHANNEL_LAYOUT["piece_own"]]
    assert own_block.shape == (12, 17, 17)

    # In CANONICAL frame the observer is rotated to SOUTH; figure out the
    # actual world cells of the observer's pieces and rotate them.
    own_cell_count_truth = 0
    from junqi_core.rotation import world_to_canonical
    for pos, piece in state.pieces.items():
        if not piece.alive:
            continue
        if piece.seat is not observer:
            continue
        own_cell_count_truth += 1
        type_idx = PIECETYPE_TO_TRACKED_IDX[piece.piece_type]
        cx, cy = world_to_canonical(pos[0], pos[1], observer)
        v = own_block[type_idx, cy, cx]
        assert v == 1.0, (
            f"observer={observer}, piece={piece.piece_type} at world {pos} / "
            f"canonical {(cx, cy)}: own_block[{type_idx}] should be 1, got {v}"
        )

    # The block's L1 sum must equal the count of observer's own alive pieces:
    # any *extra* lit cell would indicate type-leak from another seat.
    assert int(own_block.sum()) == own_cell_count_truth, (
        f"piece_own L1 sum {own_block.sum()} != own piece count "
        f"{own_cell_count_truth} — possible type leak"
    )


# ---------------------------------------------------------------------------
# 2.  Under DARK, teammate is NOT one-hot in belief
# ---------------------------------------------------------------------------


def test_dark_teammate_belief_is_not_one_hot():
    """In DARK mode, observer's belief on teammate cells must NOT be one-hot
    (since teammate types are hidden to observer, just like enemies).

    By contrast in HALF_DARK (Q11 default training mode), teammate cells
    SHOULD be one-hot (teammate setup is shared at game start).
    """
    state_dark = _new_game(seed=7, show_mode=ShowMode.DARK)
    state_half = _new_game(seed=7, show_mode=ShowMode.HALF_DARK)
    observer = Seat.SOUTH
    teammate = observer.teammate

    bel_dark = BeliefTensor.initial(state_dark, observer, show_mode=ShowMode.DARK)
    bel_half = BeliefTensor.initial(state_half, observer, show_mode=ShowMode.HALF_DARK)

    teammate_pieces = [
        (pos, p) for pos, p in state_dark.pieces.items()
        if p.alive and p.seat is teammate
    ]
    assert teammate_pieces, "fixture must have teammate pieces"

    n_one_hot_dark = 0
    n_one_hot_half = 0
    for pos, _piece in teammate_pieces:
        d = bel_dark.probs.get(pos)
        h = bel_half.probs.get(pos)
        assert d is not None and h is not None
        if _is_one_hot(d):
            n_one_hot_dark += 1
        if _is_one_hot(h):
            n_one_hot_half += 1

    # Some teammate cells *might* be one-hot in DARK because of
    # constraint-prior degeneracy (e.g. a stronghold cell can only host
    # JUNQI in some priors; back-row cells might force DILEI). The
    # invariant is that NOT ALL teammate cells are one-hot — i.e. the
    # observer must be uncertain about at least some teammate types.
    assert n_one_hot_dark < len(teammate_pieces), (
        f"DARK teammate belief is FULLY one-hot ({n_one_hot_dark}/{len(teammate_pieces)})"
        " — teammate types are leaking through"
    )
    # In HALF_DARK every teammate cell MUST be one-hot.
    assert n_one_hot_half == len(teammate_pieces), (
        f"HALF_DARK teammate belief is NOT fully one-hot "
        f"({n_one_hot_half}/{len(teammate_pieces)}) — Q11 training-mode contract violated"
    )


# ---------------------------------------------------------------------------
# 3.  Under DARK, enemy belief is generally a distribution (not truth)
# ---------------------------------------------------------------------------


def test_dark_enemy_belief_does_not_equal_truth():
    """Observer's belief must NOT just mirror the true enemy types.

    Strict version: at least one enemy cell should have a probability
    distribution that disagrees with the truth (i.e. not a single argmax
    on the true type).
    """
    state = _new_game(seed=11, show_mode=ShowMode.DARK)
    observer = Seat.SOUTH
    bel = BeliefTensor.initial(state, observer, show_mode=ShowMode.DARK)

    n_disagreeing = 0
    n_total_enemy = 0
    for pos, piece in state.pieces.items():
        if not piece.alive:
            continue
        if piece.seat is observer or piece.seat is observer.teammate:
            continue
        n_total_enemy += 1
        v = bel.probs.get(pos)
        assert v is not None
        true_idx = PIECETYPE_TO_TRACKED_IDX[piece.piece_type]
        # The argmax may coincidentally equal the truth, but the entropy
        # should be > 0 (the belief should not be one-hot on the truth).
        if not _is_one_hot(v) or float(v[true_idx]) < 1.0 - 1e-6:
            n_disagreeing += 1

    assert n_total_enemy > 0
    assert n_disagreeing == n_total_enemy, (
        f"{n_total_enemy - n_disagreeing}/{n_total_enemy} enemy cells "
        "have one-hot-on-truth belief — observer is cheating"
    )


# ---------------------------------------------------------------------------
# 4.  MoveResult never leaks piece types in DARK / HALF_DARK
# ---------------------------------------------------------------------------


def test_move_result_strips_types_in_dark():
    """Q2 contract: MoveResult must not expose attacker/defender types
    under DARK / HALF_DARK."""
    state = _new_game(seed=13, show_mode=ShowMode.DARK)

    # Step until we hit at least one combat (EAT/KILLED/BOMB).
    saw_combat = False
    for _ in range(400):
        if state.terminated:
            break
        actions = state.legal_actions(state.turn)
        if not actions:
            # Q12 dead-seat handling is internal to step_inplace; if we
            # get here with no actions for the live turn, something is
            # wrong — bail out gracefully.
            break
        action = actions[0]
        result = state.step_inplace(action)
        # Strict invariant — under DARK these MUST be None always.
        assert result.src_type_revealed is None, (
            f"src_type_revealed leaked under DARK: {result.src_type_revealed}"
        )
        assert result.dst_type_revealed is None, (
            f"dst_type_revealed leaked under DARK: {result.dst_type_revealed}"
        )
        if result.event.value > 1:  # not MOVE
            saw_combat = True

    # We don't strictly require combat to have occurred (random play might
    # not produce one in 400 steps), but assert the no-type-leak rule
    # ran on every step including any combats that did happen.
    _ = saw_combat


def test_move_result_strips_types_in_half_dark():
    """Same contract under HALF_DARK (Q11 training mode)."""
    state = _new_game(seed=13, show_mode=ShowMode.HALF_DARK)
    for _ in range(200):
        if state.terminated:
            break
        actions = state.legal_actions(state.turn)
        if not actions:
            break
        result = state.step_inplace(actions[0])
        assert result.src_type_revealed is None
        assert result.dst_type_revealed is None


# ---------------------------------------------------------------------------
# 5.  SILING-death triggers per-seat flag_revealed; nothing else does
# ---------------------------------------------------------------------------


def test_only_siling_death_or_surrender_reveals_flag():
    """``seat_flag_revealed_arr[seat]`` must flip True iff (a) seat's SILING
    died (Q7), or (b) seat surrendered (flag captured / all-dead).
    """
    state = _new_game(seed=21, show_mode=ShowMode.DARK)

    # Snapshot which seats had SILINGs and where:
    siling_pid_of: dict[Seat, int] = {}
    for piece in state.pieces.values():
        if piece.piece_type is PieceType.SILING:
            siling_pid_of[piece.seat] = piece.piece_id

    # Take the simulation up to N steps, tracking each seat's first
    # flag_revealed transition and asserting it coincides with that seat's
    # SILING death OR a flag capture / surrender event.
    flag_revealed_step: dict[Seat, int] = {}
    siling_died_step: dict[Seat, int] = {}
    surrender_step: dict[Seat, int] = {}

    for step in range(2000):
        if state.terminated:
            break
        prev_flag = state.seat_flag_revealed_arr.copy()
        prev_siling_alive = {
            s: bool(state.alive[siling_pid_of[s]]) if s in siling_pid_of else False
            for s in ALL_SEATS
        }
        actions = state.legal_actions(state.turn)
        if not actions:
            break
        result = state.step_inplace(actions[0])
        # Detect SILING death this step
        for s in ALL_SEATS:
            if s not in siling_pid_of:
                continue
            now_alive = bool(state.alive[siling_pid_of[s]])
            if prev_siling_alive[s] and not now_alive:
                siling_died_step.setdefault(s, step)
        # Detect surrender (flag captured) this step
        if result.flag_captured:
            for s in ALL_SEATS:
                if state.info[s].dead and s not in surrender_step:
                    surrender_step.setdefault(s, step)
        # Detect new flag reveals
        for s in ALL_SEATS:
            if state.seat_flag_revealed_arr[s.value] and not prev_flag[s.value]:
                flag_revealed_step.setdefault(s, step)

    # Every flag_revealed transition must be explained by a SILING death
    # at the same step, or by a surrender (flag capture) at the same step.
    for s, fstep in flag_revealed_step.items():
        sd = siling_died_step.get(s, None)
        sr = surrender_step.get(s, None)
        ok = (sd == fstep) or (sr == fstep)
        assert ok, (
            f"seat {s} flag_revealed at step {fstep} but no SILING death "
            f"({sd}) and no surrender ({sr}) at that step — illegal type leak"
        )


# ---------------------------------------------------------------------------
# 6.  Belief one-hots on enemy pieces are bounded by hard rules
# ---------------------------------------------------------------------------


def test_enemy_one_hot_only_via_rule_deductions():
    """Whenever an enemy belief becomes one-hot mid-game, the upgrade must
    be attributable to a hard rule deduction (R5/R7 GONGB-eats-DILEI,
    R6 stronghold-EAT, or SILING flag reveal).

    We don't try to enumerate every legal trigger — instead we assert the
    weaker contract: at game-START, NO enemy belief is one-hot on the
    truth.  (Mid-game one-hots get exercised by the golden inference
    cases in tests/golden/inference/.)
    """
    state = _new_game(seed=31, show_mode=ShowMode.DARK)
    observer = Seat.SOUTH
    bel = BeliefTensor.initial(state, observer, show_mode=ShowMode.DARK)

    bad: list[str] = []
    for pos, piece in state.pieces.items():
        if not piece.alive:
            continue
        if piece.seat is observer or piece.seat is observer.teammate:
            continue
        v = bel.probs.get(pos)
        assert v is not None
        if _is_one_hot(v):
            true_idx = PIECETYPE_TO_TRACKED_IDX[piece.piece_type]
            argmax = int(np.argmax(v))
            # Constraint priors do produce one-hots in some cells (e.g.
            # back-row cells can be FULLY constrained to a small subset).
            # But that one-hot must NOT match the truth UNLESS the prior
            # itself yielded that probability ≥ 1 deterministically.
            # We are conservative: any one-hot at game-start that matches
            # the truth is suspect — flag it.
            if argmax == true_idx:
                bad.append(
                    f"enemy {piece.piece_type} at {pos}: belief one-hot on truth"
                )

    assert not bad, "game-start belief contains enemy one-hots on truth: " + str(bad)


# ---------------------------------------------------------------------------
# 7.  global remaining-inventory channels exclude observer's own / teammate
# ---------------------------------------------------------------------------


def test_global_remaining_only_for_left_right_enemies():
    """The 24-dim ``remaining_left_side`` + ``remaining_right_side`` block
    in obs_global must reflect only the two enemy seats (not own / not
    teammate). It is a counts vector, not a type leak — but we verify
    the counts exactly match the truth for the enemy seats and that no
    own/teammate count appears in those slots.
    """
    state = _new_game(seed=41, show_mode=ShowMode.DARK)
    observer = Seat.SOUTH
    obs, bel = _build_obs(state, observer)

    from junqi_core.observation import GLOBAL_LAYOUT

    rem_left  = obs.global_[GLOBAL_LAYOUT["remaining_left_side"]]
    rem_right = obs.global_[GLOBAL_LAYOUT["remaining_right_side"]]
    assert rem_left.shape  == (NUM_TRACKED_TYPES,)
    assert rem_right.shape == (NUM_TRACKED_TYPES,)

    # Compute truth counts for left/right enemies.
    left_seat = observer.left_side_enemy
    right_seat = observer.right_side_enemy
    truth_left = np.zeros(NUM_TRACKED_TYPES, dtype=np.float32)
    truth_right = np.zeros(NUM_TRACKED_TYPES, dtype=np.float32)
    for piece in state.pieces.values():
        if not piece.alive:
            continue
        idx = PIECETYPE_TO_TRACKED_IDX.get(piece.piece_type, -1)
        if idx < 0:
            continue
        if piece.seat is left_seat:
            truth_left[idx] += 1
        elif piece.seat is right_seat:
            truth_right[idx] += 1

    # The observation channel stores the *believed* remaining inventory
    # which at game-start equals the truth (each enemy has 25 pieces with
    # the canonical type-count distribution). We assert equality.
    assert np.allclose(rem_left,  truth_left), (
        f"remaining_left_side {rem_left} != truth {truth_left}"
    )
    assert np.allclose(rem_right, truth_right), (
        f"remaining_right_side {rem_right} != truth {truth_right}"
    )

    # Sanity: observer's own & teammate count are NOT placed anywhere
    # in obs_global (we don't have channels for them; the rule is
    # observer-vs-enemy only).  This is just structural — there IS no
    # remaining-own / remaining-teammate channel — so the test simply
    # asserts the schema.
    assert "remaining_own" not in GLOBAL_LAYOUT
    assert "remaining_teammate" not in GLOBAL_LAYOUT


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
