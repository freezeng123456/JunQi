"""Phase 0.4 M7 — Trajectory replay determinism + on-disk round-trip."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from junqi_core.replay import Trajectory, record_trajectory
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


# ---------------------------------------------------------------------------
# 1. Round-trip: save → load → replay → hash match
# ---------------------------------------------------------------------------


class TestTrajectoryRoundTrip:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_save_load_replay_hash(self, seed: int, tmp_path: Path) -> None:
        traj = record_trajectory(rng_seed=seed, max_steps=200)
        path = tmp_path / f"replay_{seed}.npz"
        traj.save(path)
        assert path.exists()
        loaded = Trajectory.load(path)
        # Structural equivalence.
        assert loaded.num_steps == traj.num_steps
        assert loaded.rng_seed == traj.rng_seed
        assert loaded.rules_version == traj.rules_version
        assert loaded.state_version == traj.state_version
        assert loaded.first_seat is traj.first_seat
        assert loaded.show_mode is traj.show_mode
        np.testing.assert_array_equal(loaded.actions, traj.actions)
        # Setup equivalence (tuple-of-tuples deep compare).
        for s_new, s_old in zip(loaded.setups, traj.setups):
            assert tuple(s_new) == tuple(s_old)
        # Hash must match after replay from disk.
        loaded.validate()  # raises on determinism break

    def test_determinism_is_bit_exact(self, tmp_path: Path) -> None:
        """Same seed → same action sequence → same final hash."""
        t1 = record_trajectory(rng_seed=123, max_steps=300)
        t2 = record_trajectory(rng_seed=123, max_steps=300)
        assert t1.num_steps == t2.num_steps
        np.testing.assert_array_equal(t1.actions, t2.actions)
        assert t1.final_state_hash == t2.final_state_hash

    def test_load_rejects_incompatible_rules_version(
        self, tmp_path: Path,
    ) -> None:
        traj = record_trajectory(rng_seed=0, max_steps=10)
        path = tmp_path / "bad.npz"
        traj.save(path)
        # Manually corrupt rules_version.
        with np.load(path, allow_pickle=True) as data:
            fields = {k: data[k] for k in data.files}
        fields["rules_version"] = np.array("2.0.0")
        np.savez(str(path), **fields)
        with pytest.raises(ValueError, match="rules_version"):
            Trajectory.load(path)


# ---------------------------------------------------------------------------
# 2. Replay iterator yields MoveResults equivalent to fresh step()
# ---------------------------------------------------------------------------


class TestReplayIter:
    def test_iter_step_count_and_final_hash(self) -> None:
        traj = record_trajectory(rng_seed=5, max_steps=250)
        steps = list(traj.replay_iter())
        assert len(steps) == traj.num_steps
        # Last state's hash matches recorded.
        last_state, _action, _result = steps[-1]
        assert last_state.state_hash() == traj.final_state_hash

    def test_replay_with_results_equivalent_to_iter(self) -> None:
        traj = record_trajectory(rng_seed=9, max_steps=200)
        final_a, results_a = traj.replay(return_move_results=True)
        results_b = [r for _s, _a, r in traj.replay_iter()]
        # Both paths produce the same MoveResult sequence.
        assert len(results_a) == len(results_b)
        for ra, rb in zip(results_a, results_b):
            # MoveResult is a frozen dataclass; equality is structural.
            assert ra == rb
        assert final_a.state_hash() == traj.final_state_hash


# ---------------------------------------------------------------------------
# 3. 100-game smoke: every replay round-trips deterministically
# ---------------------------------------------------------------------------


class TestHundredGameReplay:
    def test_100_random_games_roundtrip(self, tmp_path: Path) -> None:
        """For a batch of 100 short games, save → load → validate."""
        N_GAMES = 100
        STEPS   = 80
        hashes = []
        for g in range(N_GAMES):
            traj = record_trajectory(rng_seed=g, max_steps=STEPS)
            p = tmp_path / f"g{g:03d}.npz"
            traj.save(p)
            loaded = Trajectory.load(p)
            loaded.validate()
            hashes.append(loaded.final_state_hash)
        # Seeds yield distinct trajectories (not a hard requirement, but
        # hash collisions would be a major red flag).
        assert len(set(hashes)) > N_GAMES // 2


# ---------------------------------------------------------------------------
# 4. Custom policy capture
# ---------------------------------------------------------------------------


class TestCustomPolicyCapture:
    def test_policy_is_called_per_step(self) -> None:
        calls = {"n": 0}

        def greedy_first_legal(state: GameState) -> int:
            calls["n"] += 1
            ids = state.legal_action_ids()
            return int(ids[0]) if ids.size else -1

        traj = record_trajectory(
            rng_seed=0, policy=greedy_first_legal, max_steps=50,
        )
        assert traj.num_steps <= 50
        assert calls["n"] == traj.num_steps or calls["n"] == traj.num_steps + 1

    def test_explicit_setups_override_rng(self) -> None:
        """Passing setups= pins the starting position independent of seed."""
        rng = random.Random(123)
        setups = generate_random_setup(rng)
        t1 = record_trajectory(setups=setups, rng_seed=0, max_steps=30)
        t2 = record_trajectory(setups=setups, rng_seed=0, max_steps=30)
        # Same seed AND same setups → identical trajectory.
        np.testing.assert_array_equal(t1.actions, t2.actions)
        assert t1.final_state_hash == t2.final_state_hash
