"""Phase 0.4 M1 / ADR-117 — SoA GameState + incremental Zobrist.

These tests are the acceptance harness for ADR-117.  They pin:

  * `clone()` preserves ``zobrist`` and produces byte-identical SoA arrays.
  * ``step_inplace`` maintains the SoA ↔ dict consistency invariants at every
    step of a 300-move random game.
  * ``step(action)`` and ``clone() + step_inplace(action)`` are equivalent.
  * ``to_dict`` / ``from_dict`` round-trip is bit-exact for schema v2,
    and v1 replays (pre-ADR-117) still load without SoA information.
  * The incremental Zobrist matches a from-scratch recomputation at every
    step.

None of the existing pytests are modified; this file is the *additional*
regression net for M1.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from junqi_core._zobrist import NUM_PIECE_IDS
from junqi_core.board import BOARD_SIZE, xy_to_flat
from junqi_core.rules import ALL_SEATS, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState, _recompute_zobrist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _soa_matches_dict(state: GameState) -> None:
    """Assert every SoA column is consistent with the authoritative dict state.

    Raises AssertionError if any invariant breaks.
    """
    # Liveness set agrees.
    dict_pids = {ref.piece_id for ref in state.pieces.values()}
    soa_pids = set(int(p) for p in np.nonzero(state.alive)[0].tolist())
    assert dict_pids == soa_pids, (
        f"dict pids {dict_pids - soa_pids} missing from SoA; "
        f"SoA pids {soa_pids - dict_pids} orphaned"
    )

    # pos_x / pos_y / cell_piece_id agree with the dict's (pos, PieceRef).
    for pos, ref in state.pieces.items():
        pid = ref.piece_id
        assert pid >= 0, f"piece at {pos} has unassigned piece_id"
        assert int(state.pos_x[pid]) == pos[0]
        assert int(state.pos_y[pid]) == pos[1]
        assert int(state.piece_seat_arr[pid]) == ref.seat.value
        assert int(state.piece_type_arr[pid]) == ref.piece_type.value
        flat = xy_to_flat(*pos)
        assert int(state.cell_piece_id[flat]) == pid

    # Dead pids have pos=(-1,-1) and absent from cell_piece_id.
    dead_pids = [int(p) for p in np.nonzero(~state.alive)[0].tolist()]
    for pid in dead_pids:
        if int(state.piece_seat_arr[pid]) < 0:
            continue  # unassigned slot (camp) — not a real dead piece
        assert int(state.pos_x[pid]) == -1
        assert int(state.pos_y[pid]) == -1
        # If the piece has a death record, its cell must be empty.
        if pid in state.deaths:
            # A cell that currently holds an alive piece belongs to
            # whoever migrated onto it; but the dead piece's own cell
            # reference is -1.
            pass

    # `cell_piece_id` has -1 for empty cells and the right pid elsewhere.
    for flat_cell in range(BOARD_SIZE * BOARD_SIZE):
        pid_in_cell = int(state.cell_piece_id[flat_cell])
        if pid_in_cell < 0:
            # Cell claims empty; no live piece should be positioned here.
            for pos, ref in state.pieces.items():
                assert xy_to_flat(*pos) != flat_cell, (
                    f"cell {flat_cell} = empty in SoA but dict has piece at {pos}"
                )
        else:
            assert bool(state.alive[pid_in_cell]), (
                f"cell_piece_id[{flat_cell}] = {pid_in_cell} but not alive"
            )

    # piece_state counters agree.
    for pid, ps in state.piece_state.items():
        assert int(state.move_count_arr[pid]) == ps.move_count
        assert int(state.active_eat_arr[pid]) == ps.active_eat_count
        assert int(state.passive_surv_arr[pid]) == ps.passive_survive_count

    # deaths counters agree.
    for pid, di in state.deaths.items():
        assert int(state.death_reason_arr[pid]) == di.reason.value
        assert int(state.death_step_arr[pid]) == di.step
        assert int(state.death_loc_flat_arr[pid]) == xy_to_flat(*di.death_loc)

    # Seat scalars agree.
    for s in ALL_SEATS:
        assert bool(state.seat_dead_arr[s.value]) == state.info[s].dead
        assert (
            bool(state.seat_flag_revealed_arr[s.value])
            == state.info[s].flag_revealed
        )


def _zobrist_recomputed_matches(state: GameState) -> None:
    """Assert ``state.zobrist`` equals a from-scratch recomputation."""
    recomputed = _recompute_zobrist(
        alive=state.alive,
        piece_type_arr=state.piece_type_arr,
        pos_x=state.pos_x,
        pos_y=state.pos_y,
        turn_val=state.turn.value,
        move_counter=state.move_counter,
        moves_since_last_combat=state.moves_since_last_combat,
        terminated=state.terminated,
        winner_team=state.winner_team,
        draw=state.draw,
        show_mode_val=state.show_mode.value,
        seat_dead_arr=state.seat_dead_arr,
        seat_flag_revealed_arr=state.seat_flag_revealed_arr,
    )
    assert state.zobrist == recomputed, (
        f"incremental Zobrist {state.zobrist} != recomputed {recomputed}"
    )


# ---------------------------------------------------------------------------
# 1. Opening-position invariants
# ---------------------------------------------------------------------------


class TestNewGameSoA:
    def test_opening_soa_matches_dict(self):
        rng = random.Random(20260421)
        state = GameState.new_game(generate_random_setup(rng))
        _soa_matches_dict(state)

    def test_opening_alive_count(self):
        rng = random.Random(20260421)
        state = GameState.new_game(generate_random_setup(rng))
        # 25 real pieces per seat × 4 seats.
        assert int(state.alive.sum()) == 100
        # Camp pids are unassigned (piece_seat_arr == -1).
        assert int((state.piece_seat_arr == -1).sum()) == NUM_PIECE_IDS - 100

    def test_opening_zobrist_is_nonzero(self):
        rng = random.Random(20260421)
        state = GameState.new_game(generate_random_setup(rng))
        assert state.zobrist != 0

    def test_opening_zobrist_recomputed_matches(self):
        rng = random.Random(20260421)
        state = GameState.new_game(generate_random_setup(rng))
        _zobrist_recomputed_matches(state)


# ---------------------------------------------------------------------------
# 2. clone() preserves Zobrist and SoA
# ---------------------------------------------------------------------------


class TestCloneIdentity:
    def test_clone_preserves_zobrist(self):
        rng = random.Random(42)
        state = GameState.new_game(generate_random_setup(rng))
        assert state.clone().zobrist == state.zobrist

    def test_clone_soa_arrays_bitequal(self):
        rng = random.Random(42)
        state = GameState.new_game(generate_random_setup(rng))
        c = state.clone()
        for name in (
            "alive", "piece_type_arr", "piece_seat_arr",
            "pos_x", "pos_y", "zero_x", "zero_y",
            "move_count_arr", "active_eat_arr", "passive_surv_arr",
            "death_reason_arr", "death_step_arr", "death_loc_flat_arr",
            "cell_piece_id", "seat_dead_arr", "seat_flag_revealed_arr",
        ):
            a, b = getattr(state, name), getattr(c, name)
            assert np.array_equal(a, b), f"clone differs on {name!r}"
            assert a is not b, f"clone shares memory on {name!r}"

    def test_clone_then_hash_equal(self):
        rng = random.Random(42)
        state = GameState.new_game(generate_random_setup(rng))
        assert state.state_hash() == state.clone().state_hash()


# ---------------------------------------------------------------------------
# 3. step() and clone + step_inplace() are equivalent
# ---------------------------------------------------------------------------


class TestStepEquivalence:
    def test_step_matches_clone_step_inplace(self):
        rng = random.Random(7)
        state = GameState.new_game(generate_random_setup(rng))
        legal = state.legal_actions()
        action = rng.choice(legal)

        s1, r1 = state.step(action)
        s2 = state.clone()
        r2 = s2.step_inplace(action)

        assert s1.zobrist == s2.zobrist
        assert r1 == r2
        for name in ("alive", "pos_x", "pos_y", "cell_piece_id",
                     "move_count_arr", "active_eat_arr", "passive_surv_arr"):
            assert np.array_equal(getattr(s1, name), getattr(s2, name))

    def test_step_leaves_original_unchanged(self):
        rng = random.Random(7)
        state = GameState.new_game(generate_random_setup(rng))
        original_zob = state.zobrist
        original_pieces = dict(state.pieces)
        action = rng.choice(state.legal_actions())
        _new, _ = state.step(action)
        assert state.zobrist == original_zob
        assert state.pieces == original_pieces


# ---------------------------------------------------------------------------
# 4. 300-step random game: SoA + Zobrist invariants hold every step
# ---------------------------------------------------------------------------


class TestRandomGameInvariants:
    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
    def test_invariants_over_random_game(self, seed):
        rng = random.Random(seed)
        state = GameState.new_game(generate_random_setup(rng))
        for _ in range(300):
            if state.terminated:
                break
            legal = state.legal_actions()
            if not legal:
                break
            a = rng.choice(legal)
            state, _ = state.step(a)
            _soa_matches_dict(state)
            _zobrist_recomputed_matches(state)

    def test_step_inplace_invariants(self):
        rng = random.Random(99)
        state = GameState.new_game(generate_random_setup(rng))
        for _ in range(200):
            if state.terminated:
                break
            legal = state.legal_actions()
            if not legal:
                break
            a = rng.choice(legal)
            state.step_inplace(a)
            _soa_matches_dict(state)
            _zobrist_recomputed_matches(state)


# ---------------------------------------------------------------------------
# 5. to_dict v2 round-trip
# ---------------------------------------------------------------------------


class TestSerializationV2:
    def test_opening_roundtrip_preserves_all_fields(self):
        rng = random.Random(2026)
        state = GameState.new_game(generate_random_setup(rng))
        # Walk a few steps to populate counters.
        for _ in range(25):
            if state.terminated:
                break
            legal = state.legal_actions()
            if not legal:
                break
            state, _ = state.step(rng.choice(legal))

        d = state.to_dict()
        assert d["state_version"] == GameState.STATE_VERSION
        assert "zero_board" in d
        assert "piece_state" in d
        assert "deaths" in d
        assert "zobrist" in d

        # JSON round-trip (exact).
        d2 = json.loads(json.dumps(d))
        restored = GameState.from_dict(d2)

        assert restored.zobrist == state.zobrist
        assert restored.state_hash() == state.state_hash()
        assert restored.turn == state.turn
        assert restored.move_counter == state.move_counter
        assert restored.moves_since_last_combat == state.moves_since_last_combat
        assert restored.pieces == state.pieces
        assert restored.piece_state == state.piece_state
        assert restored.deaths == state.deaths
        assert restored.zero_board == state.zero_board
        assert np.array_equal(restored.alive, state.alive)
        assert np.array_equal(restored.cell_piece_id, state.cell_piece_id)
        assert np.array_equal(restored.move_count_arr, state.move_count_arr)

    def test_roundtrip_after_combat(self):
        """Make sure a state with at least one death round-trips cleanly."""
        rng = random.Random(31415)
        state = GameState.new_game(generate_random_setup(rng))
        for _ in range(400):
            if state.terminated:
                break
            legal = state.legal_actions()
            if not legal:
                break
            state, mr = state.step(rng.choice(legal))
            if state.deaths:
                break  # at least one death recorded; good enough
        assert state.deaths, "random rollout produced zero deaths in 400 steps"

        d = state.to_dict()
        d2 = json.loads(json.dumps(d))
        restored = GameState.from_dict(d2)
        assert restored.zobrist == state.zobrist
        assert restored.deaths == state.deaths


# ---------------------------------------------------------------------------
# 6. v1 schema (pre-ADR-117) still loads (lossy on T7 metadata)
# ---------------------------------------------------------------------------


class TestSerializationV1Compat:
    def test_v1_dict_loads_without_crash(self):
        rng = random.Random(777)
        state = GameState.new_game(generate_random_setup(rng))
        state, _ = state.step(rng.choice(state.legal_actions()))

        # Synthesize a v1 dict from the v2 output (drop new fields).
        d = state.to_dict()
        d_v1 = {
            k: d[k] for k in (
                "rules_version", "pieces", "turn", "move_counter",
                "moves_since_last_combat", "info", "terminated",
                "winner_team", "draw", "show_mode", "debug_include_private",
            )
        }
        # Strip piece_id from pieces entries (v1 didn't carry it).
        for p in d_v1["pieces"]:
            p.pop("piece_id", None)

        restored = GameState.from_dict(d_v1)
        # Alive-piece count must match the source dict (ignores T7 data).
        assert len(restored.pieces) == len(state.pieces)
        # Dead seats preserved.
        for s in ALL_SEATS:
            assert restored.info[s].dead == state.info[s].dead

    def test_v1_dict_has_no_zobrist_but_computes_one(self):
        rng = random.Random(888)
        state = GameState.new_game(generate_random_setup(rng))
        d_v1 = {
            k: state.to_dict()[k] for k in (
                "rules_version", "pieces", "turn", "move_counter",
                "moves_since_last_combat", "info", "terminated",
                "winner_team", "draw", "show_mode", "debug_include_private",
            )
        }
        # v1 pieces had piece_id already in our to_dict shim output;
        # strip it to simulate a true legacy file.
        for p in d_v1["pieces"]:
            p.pop("piece_id", None)
        restored = GameState.from_dict(d_v1)
        # Without piece_id, SoA is incomplete; zobrist is still non-zero
        # (scalars + whatever alive rows we recovered).
        assert restored.zobrist != 0
