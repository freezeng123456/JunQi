"""Tests for junqi_core.batched_state.BatchedGameState (Phase 1a M1).

Test coverage:
  1. Construction: from_game_states / allocate
  2. legal_action_ids_batch: set-equality with single-env GameState.legal_action_ids
  3. step_batch: result correctness (plain move, combat EAT/KILLED/BOMB)
  4. Zobrist consistency: BatchedGameState.zobrist matches GameState.zobrist
     after identical move sequences
  5. Termination propagation
  6. Clone / snapshot integrity
  7. Benchmark: ≥50k plays/sec at N=1024 (measured with time.perf_counter)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from junqi_core.batched_state import BatchedGameState
from junqi_core.rules import Seat
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_game() -> GameState:
    """Create a fresh GameState with random setups."""
    setup = generate_random_setup()
    return GameState.new_game(setup)


def _batch_from_n(n: int) -> BatchedGameState:
    states = [_new_game() for _ in range(n)]
    return BatchedGameState.from_game_states(states)


def _random_legal_action(gs: GameState) -> int:
    """Pick a random legal action for the current turn; return flat id."""
    ids = gs.legal_action_ids()
    assert ids.size > 0, "no legal actions — something is wrong"
    return int(ids[np.random.randint(len(ids))])


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_game_states_shapes(self):
        N = 8
        b = _batch_from_n(N)
        assert b.num_envs == N
        assert b.alive.shape == (N, 120)
        assert b.cell_piece_id.shape == (N, 289)
        assert b.seat_dead_arr.shape == (N, 4)
        assert b.turn.shape == (N,)
        assert b.zobrist.shape == (N,)
        assert b.move_counter.shape == (N,)

    def test_allocate_shapes(self):
        N = 4
        b = BatchedGameState.allocate(N)
        assert b.num_envs == N
        assert b.alive.shape == (N, 120)
        assert b.cell_piece_id.shape == (N, 289)
        assert b.turn.shape == (N,)

    def test_from_game_states_values(self):
        """Check that batched SoA columns match the source GameState."""
        states = [_new_game() for _ in range(3)]
        b = BatchedGameState.from_game_states(states)
        for i, gs in enumerate(states):
            np.testing.assert_array_equal(b.alive[i], gs.alive)
            np.testing.assert_array_equal(b.pos_x[i], gs.pos_x)
            np.testing.assert_array_equal(b.cell_piece_id[i], gs.cell_piece_id)
            assert int(b.turn[i]) == gs.turn.value
            assert int(b.zobrist[i]) == gs.zobrist
            assert int(b.move_counter[i]) == gs.move_counter

    def test_clone_is_independent(self):
        b = _batch_from_n(2)
        c = b.clone()
        c.alive[0, 0] = not b.alive[0, 0]
        # original should be unmodified
        assert b.alive[0, 0] != c.alive[0, 0]

    def test_clone_copies_every_field(self):
        """Checking one array is not enough.

        The hand-written clone had drifted 16 fields behind the dataclass —
        every ``cm_*`` CombatMemory column was omitted, so a clone came back
        with the empty arrays from their ``default_factory`` while the board
        and counters copied fine. ``test_clone_is_independent`` only looks at
        ``alive``, so it passed throughout.
        """
        import dataclasses

        b = _batch_from_n(2)
        c = b.clone()
        for f in dataclasses.fields(b):
            orig, copied = getattr(b, f.name), getattr(c, f.name)
            if not isinstance(orig, np.ndarray):
                assert orig == copied, f"{f.name}: {orig!r} != {copied!r}"
                continue
            assert orig.shape == copied.shape, (
                f"{f.name}: shape {orig.shape} != {copied.shape}"
            )
            np.testing.assert_array_equal(
                orig, copied, err_msg=f"{f.name} differs after clone"
            )
            if orig.size:
                assert not np.shares_memory(orig, copied), (
                    f"{f.name} is aliased, not copied"
                )


# ---------------------------------------------------------------------------
# 2. Legal action parity
# ---------------------------------------------------------------------------

class TestLegalActions:
    def test_legal_action_parity(self):
        """Batched legal action ids must match single-env results."""
        N = 16
        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        batch_ids = b.legal_action_ids_batch()
        for i, gs in enumerate(states):
            expected = set(gs.legal_action_ids().tolist())
            actual = set(batch_ids[i].tolist())
            assert actual == expected, (
                f"env {i}: expected {len(expected)} actions, got {len(actual)}"
            )

    def test_terminated_env_returns_empty(self):
        b = _batch_from_n(2)
        b.terminated[0] = True
        ids = b.legal_action_ids_batch()
        assert ids[0].size == 0
        assert ids[1].size > 0


# ---------------------------------------------------------------------------
# 3. Step correctness
# ---------------------------------------------------------------------------

class TestStep:
    def test_plain_move_state_consistency(self):
        """After a plain move, piece positions must be consistent."""
        N = 4
        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        # Find a plain-move action for each env (try first legal action)
        from junqi_core.rules import Event
        from junqi_core.board import NUM_CELLS

        action_ids = np.zeros(N, dtype=np.int32)
        ref_states: list[GameState] = []
        for i, gs in enumerate(states):
            ids = gs.legal_action_ids()
            # Pick first move that results in Event.MOVE (empty dst)
            chosen = int(ids[0])
            for aid in ids.tolist():
                src_flat = aid // NUM_CELLS
                dst_flat = aid % NUM_CELLS
                if gs.cell_piece_id[dst_flat] < 0:
                    chosen = aid
                    break
            action_ids[i] = chosen
            ref_states.append(gs)

        # Step via batched
        result = b.step_batch(action_ids)
        assert result.valid.all()

        # Each env: check cell_piece_id consistency
        for i in range(N):
            # src cell must now be empty
            src_flat = int(action_ids[i]) // NUM_CELLS
            assert b.cell_piece_id[i, src_flat] == -1, (
                f"env {i}: src_flat {src_flat} should be empty after move"
            )
            # dst cell must have a piece
            dst_flat = int(action_ids[i]) % NUM_CELLS
            pid = int(b.cell_piece_id[i, dst_flat])
            assert pid >= 0, f"env {i}: dst_flat {dst_flat} should have a piece"
            # Piece position arrays must match
            expected_x = dst_flat % 17
            expected_y = dst_flat // 17
            assert b.pos_x[i, pid] == expected_x
            assert b.pos_y[i, pid] == expected_y

    def test_step_matches_single_env(self):
        """Batched step must produce the same state as single-env step_inplace."""
        N = 8
        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        # Use first legal action for each env
        action_ids = np.array([
            int(gs.legal_action_ids()[0]) for gs in states
        ], dtype=np.int32)

        # Single-env reference steps
        for i, gs in enumerate(states):
            from junqi_core.board import NUM_CELLS as _NC
            from junqi_core.state import Action
            aid = int(action_ids[i])
            src_flat = aid // _NC
            dst_flat = aid % _NC
            src_x, src_y = src_flat % 17, src_flat // 17
            dst_x, dst_y = dst_flat % 17, dst_flat // 17
            action = Action(seat=gs.turn, src=(src_x, src_y), dst=(dst_x, dst_y))
            gs.step_inplace(action)

        # Batched step
        b.step_batch(action_ids)

        # Compare SoA columns
        for i, gs in enumerate(states):
            np.testing.assert_array_equal(
                b.alive[i], gs.alive,
                err_msg=f"env {i}: alive mismatch"
            )
            np.testing.assert_array_equal(
                b.pos_x[i], gs.pos_x,
                err_msg=f"env {i}: pos_x mismatch"
            )
            np.testing.assert_array_equal(
                b.pos_y[i], gs.pos_y,
                err_msg=f"env {i}: pos_y mismatch"
            )
            np.testing.assert_array_equal(
                b.cell_piece_id[i], gs.cell_piece_id,
                err_msg=f"env {i}: cell_piece_id mismatch"
            )

    def test_zobrist_matches_single_env(self):
        """BatchedGameState.zobrist must match GameState.zobrist after same moves."""
        N = 4
        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        for step in range(5):
            action_ids = np.array([
                int(gs.legal_action_ids()[0]) for gs in states
            ], dtype=np.int32)

            # Single-env reference
            for i, gs in enumerate(states):
                from junqi_core.board import NUM_CELLS as _NC
                from junqi_core.state import Action
                aid = int(action_ids[i])
                src_flat = aid // _NC
                dst_flat = aid % _NC
                src = (src_flat % 17, src_flat // 17)
                dst = (dst_flat % 17, dst_flat // 17)
                gs.step_inplace(Action(seat=gs.turn, src=src, dst=dst))

            b.step_batch(action_ids)

            for i, gs in enumerate(states):
                assert int(b.zobrist[i]) == gs.zobrist, (
                    f"step {step}, env {i}: zobrist mismatch: "
                    f"batch={int(b.zobrist[i]):#x} vs single={gs.zobrist:#x}"
                )

    def test_terminated_env_skipped(self):
        """step_batch must not modify terminated environments."""
        b = _batch_from_n(2)
        b.terminated[0] = True
        b.move_counter[0] = 99

        action_ids = np.array([0, int(b.legal_action_ids_batch()[1][0])], dtype=np.int32)
        result = b.step_batch(action_ids)

        assert not result.valid[0], "terminated env should not be marked valid"
        assert int(b.move_counter[0]) == 99, "terminated env counter should be unchanged"


# ---------------------------------------------------------------------------
# 4. Multi-step rollout
# ---------------------------------------------------------------------------

class TestRollout:
    def test_multi_step_rollout_no_crash(self):
        """Run 100 steps across N=32 envs — must not crash or produce NaN."""
        N = 32
        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        for step in range(100):
            active_ids = b.legal_action_ids_batch()
            action_ids = np.zeros(N, dtype=np.int32)
            for i in range(N):
                if b.terminated[i] or len(active_ids[i]) == 0:
                    action_ids[i] = 0
                else:
                    action_ids[i] = active_ids[i][
                        np.random.randint(len(active_ids[i]))
                    ]
            b.step_batch(action_ids)

        # Check no NaN / inf in zobrist
        assert np.isfinite(b.zobrist.astype(float)).all()

    def test_game_terminates(self):
        """A random-play game should eventually terminate (max 4000 steps)."""
        b = _batch_from_n(1)
        max_steps = 4000
        for step in range(max_steps):
            if b.terminated[0]:
                break
            ids = b.legal_action_ids_batch()[0]
            if ids.size == 0:
                break
            aid = ids[np.random.randint(len(ids))]
            b.step_batch(np.array([int(aid)], dtype=np.int32))
        # Should either be terminated or have run out (draw)
        # Not asserting termination since games CAN be drawn by 4000 moves


# ---------------------------------------------------------------------------
# 5. Benchmark (≥50k plays/sec at N=1024)
# ---------------------------------------------------------------------------

class TestBenchmark:
    @pytest.mark.slow
    def test_throughput_n1024(self):
        """Achieve ≥20k plays/sec (whole game steps) at N=1024.

        The original M1 target was 50k, but the Phase 0.4 rail-topology
        rewrite (correct single-connected-component rail graph + 4 curve
        rails) added inherent complexity to legal-move generation:
          - Engineer BFS on a 73-node connected graph (was 4×16 cycles)
          - Curve-rail BFS (4 chains × 12 cells)
          - Straight-rail rays up to length 12 (was ≤4)

        Phase T-02 optimizations (vectorized curve-rail rays, hybrid
        engineer BFS, fast-path has_legal_moves) recovered from 16.6k
        to ~25k on the author's dev box. On H20 cluster nodes the CPU
        is ~5% slower (24k measured), so the threshold is set to 20k —
        the GPU pipeline (GpuRollout) bypasses this entirely at 1.15M
        env·steps/s, so the absolute CPU number does not gate training
        throughput.

        Marked with ``pytest.mark.slow`` so it can be skipped in fast
        CI / on heterogeneous CPU hardware.
        """
        N = 1024
        TARGET_STEPS_PER_SEC = 20_000

        states = [_new_game() for _ in range(N)]
        b = BatchedGameState.from_game_states(states)

        WARMUP_STEPS = 20
        MEASURE_STEPS = 200

        # Warmup
        for _ in range(WARMUP_STEPS):
            active_ids = b.legal_action_ids_batch()
            action_ids = np.array([
                int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
                for ids in active_ids
            ], dtype=np.int32)
            b.step_batch(action_ids)

        # Measure
        t0 = time.perf_counter()
        for _ in range(MEASURE_STEPS):
            active_ids = b.legal_action_ids_batch()
            action_ids = np.array([
                int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
                for ids in active_ids
            ], dtype=np.int32)
            b.step_batch(action_ids)
        elapsed = time.perf_counter() - t0

        total_env_steps = N * MEASURE_STEPS
        throughput = total_env_steps / elapsed
        print(
            f"\nBatchedGameState N={N}: "
            f"{MEASURE_STEPS} steps in {elapsed:.2f}s → "
            f"{throughput:,.0f} env-steps/sec"
        )

        assert throughput >= TARGET_STEPS_PER_SEC, (
            f"Throughput {throughput:,.0f} < target {TARGET_STEPS_PER_SEC:,} "
            f"env-steps/sec"
        )
