"""Regression tests for the ed360f7 audit; real NumPy/Torch math on CPU.

In a full checkout: pytest -q tests/test_audit_20260907_training.py
The offline package also has an isolated runner with explicitly documented
observation/network import shims. That runner is NOT an engine integration test.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from junqi_core.rules import Seat
from junqi_rl.training import checkpoint, collector
from junqi_rl.training._checkpoint_io import atomic_write
from junqi_rl.training.config import TrainConfig
from junqi_rl.training.rollout import (
    BOARD_SIZE, FLAT_ACTION_DIM, OBS_CHANNELS, OBS_GLOBAL_DIMS, RolloutBuffer,
)
from junqi_rl.training.rollout_gpu import (
    RolloutBufferGPU, _csr_to_dense_selected, _dense_mask_to_csr,
)


def _returns_buffer(backend, rewards, values, dones, seats, gae, td, gamma=1.0):
    cls = RolloutBuffer if backend == "numpy" else RolloutBufferGPU
    b = cls.__new__(cls)
    b.steps_per_env, b.num_envs = rewards.shape
    b.gamma, b.gae_lambda, b.td_lambda = gamma, gae, td
    b.device = torch.device("cpu")
    b.random_opponent = False
    b.train_value_on_random_seats = False
    for name, array in dict(rewards=rewards, values=values, dones=dones, seats=seats).items():
        setattr(b, name, array.copy() if backend == "numpy" else torch.from_numpy(array.copy()))
    b.advantages_ = np.zeros_like(values) if backend == "numpy" else torch.zeros_like(b.values)
    b.returns_ = np.zeros_like(values) if backend == "numpy" else torch.zeros_like(b.values)
    return b


def _compute(b, last_values, last_seats):
    # The pre-fix CPU API had no last_seats. This lets baseline comparison
    # fail on its actual numerical result, not just on a signature TypeError.
    kwargs = {"last_seats": last_seats} if "last_seats" in inspect.signature(b.compute_returns).parameters else {}
    b.compute_returns(last_values, **kwargs)
    def numpy(x):
        return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    return numpy(b.advantages_), numpy(b.returns_)


def _nstep_lambda_oracle(rewards, values, dones, seats, last_values, last_seats, gamma, lam):
    """Independent forward n-step mixture, NOT the implementation's trace loop."""
    T, N = rewards.shape
    out = np.zeros((T, N), dtype=np.float64)
    for t in range(T):
        for e in range(N):
            own_team = int(seats[t, e]) & 1
            reward_sum = 0.0
            targets = []
            for j in range(t, T):
                n = j - t + 1
                sign = 1.0 if (int(seats[j, e]) & 1) == own_team else -1.0
                reward_sum += (gamma ** (n - 1)) * sign * float(rewards[j, e])
                target = reward_sum
                if not dones[j, e]:
                    v = float(values[j + 1, e]) if j + 1 < T else float(last_values[e])
                    next_seat = int(seats[j + 1, e]) if j + 1 < T else int(last_seats[e])
                    sign_next = 1.0 if (next_seat & 1) == own_team else -1.0
                    target += gamma**n * sign_next * v
                targets.append(target)
                if dones[j, e]:
                    break
            H = len(targets)
            out[t, e] = sum(
                ((1.0 - lam) * lam**k if k < H - 1 else lam**k) * target
                for k, target in enumerate(targets)
            )
    return out


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_team_perspective_terminal_loss(backend):
    r = np.array([[0], [0], [0], [-1]], np.float32)
    b = _returns_buffer(backend, r, np.zeros_like(r), np.array([[0], [0], [0], [1]], bool),
                        np.arange(4, dtype=np.int8)[:, None], 1.0, 1.0)
    adv, ret = _compute(b, np.zeros(1, np.float32), np.array([0], np.int8))
    np.testing.assert_allclose(adv[:, 0], [1, -1, 1, -1])
    np.testing.assert_allclose(ret[:, 0], [1, -1, 1, -1])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_td_lambda_is_independent(backend):
    r = np.array([[0], [0], [1]], np.float32)
    b = _returns_buffer(backend, r, np.zeros_like(r), np.array([[0], [0], [1]], bool),
                        np.zeros_like(r, np.int8), 0.0, 1.0)
    adv, ret = _compute(b, np.zeros(1, np.float32), np.array([0], np.int8))
    np.testing.assert_allclose(adv[:, 0], [0, 0, 1])
    np.testing.assert_allclose(ret[:, 0], [1, 1, 1])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("lambdas", [(0.0, 1.0), (0.5, 0.8), (1.0, 0.0)])
def test_returns_match_forward_nstep_oracle(backend, seed, lambdas):
    rng = np.random.default_rng(seed)
    T, N = 1 + seed % 7, 1 + seed % 3
    rewards = rng.uniform(-1, 1, size=(T, N)).astype(np.float32)
    values = rng.uniform(-1, 1, size=(T, N)).astype(np.float32)
    dones = rng.random((T, N)) < 0.3
    seats = rng.integers(0, 4, (T, N), dtype=np.int8)
    last_values = rng.uniform(-1, 1, N).astype(np.float32)
    last_seats = rng.integers(0, 4, N, dtype=np.int8)
    gae, td = lambdas
    gamma = 0.97
    b = _returns_buffer(backend, rewards, values, dones, seats, gae, td, gamma)
    adv, ret = _compute(b, last_values, last_seats)
    oracle_adv = _nstep_lambda_oracle(rewards, values, dones, seats, last_values, last_seats, gamma, gae) - values
    oracle_ret = _nstep_lambda_oracle(rewards, values, dones, seats, last_values, last_seats, gamma, td)
    np.testing.assert_allclose(adv, oracle_adv, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(ret, oracle_ret, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_teammate_bootstrap_after_elimination(backend):
    r = np.zeros((1, 1), np.float32)
    b = _returns_buffer(backend, r, r, np.zeros_like(r, bool), np.array([[0]], np.int8), 0.5, 0.8)
    adv, ret = _compute(b, np.array([0.75], np.float32), np.array([2], np.int8))
    np.testing.assert_allclose(adv, [[0.75]])
    np.testing.assert_allclose(ret, [[0.75]])


@pytest.mark.parametrize("value_only", [False, True])
def test_random_opponent_policy_gate_preserved(value_only):
    r = np.array([[0], [0], [0], [-1]], np.float32)
    b = _returns_buffer("torch", r, np.zeros_like(r), np.array([[0], [0], [0], [1]], bool),
                        np.arange(4, dtype=np.int8)[:, None], 1, 1)
    b.random_opponent = True
    b.train_value_on_random_seats = value_only
    adv, ret = _compute(b, np.zeros(1, np.float32), np.array([0], np.int8))
    np.testing.assert_array_equal(adv[:, 0], [1, 0, 1, 0])
    np.testing.assert_array_equal(ret[:, 0], [1, -1, 1, -1] if value_only else [1, 0, 1, 0])


def _minibatch_buffer(advantages, seats=None, random_opponent=False, grouping="global"):
    b = RolloutBufferGPU.__new__(RolloutBufferGPU)
    b.device = torch.device("cpu")
    b.steps_per_env, b.num_envs = 1, len(advantages)
    shape = (1, b.num_envs)
    b.uses_compact_history = False
    b.csr_legal_mask = False
    b.random_opponent = random_opponent
    b.train_value_on_random_seats = True
    b.adv_filter_scope = "timestep"
    b.value_sample_scope = "policy"
    b.minibatch_group = grouping
    b.adv_filt_thresh = 0.0
    b.adv_filt_rate = 1.0
    b.obs_spatial = torch.zeros((*shape, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    b.obs_global = torch.zeros((*shape, OBS_GLOBAL_DIMS))
    b.legal_mask = torch.ones((*shape, FLAT_ACTION_DIM), dtype=torch.bool)
    b.actions = torch.zeros(shape, dtype=torch.int32)
    b.log_probs = torch.zeros(shape)
    b.advantages_ = torch.tensor(advantages, dtype=torch.float32).reshape(shape)
    b.returns_ = torch.zeros(shape)
    b.values = torch.zeros(shape)
    b.rewards = torch.zeros(shape)
    b.seats = torch.tensor(seats or [0] * b.num_envs, dtype=torch.int8).reshape(shape)
    return b


@pytest.mark.parametrize("grouping", ["global", "timestep"])
def test_singleton_minibatch_is_finite_and_kept(grouping):
    b = _minibatch_buffer([3.0], grouping=grouping)
    batches = list(b.minibatches(1, shuffle=False))
    assert len(batches) == 1
    assert torch.isfinite(batches[0].advantages).all()
    assert batches[0].advantages.item() == 0.0
    assert b.num_valid_transitions() == 1
    assert np.isfinite(b.stats()["rollout/std_advantage"])


def test_single_policy_seat_with_random_opponent_is_finite():
    b = _minibatch_buffer([3, 0], seats=[0, 1], random_opponent=True)
    batches = list(b.minibatches(4, shuffle=False))
    assert len(batches) == 1
    assert torch.isfinite(batches[0].advantages).all()
    assert batches[0].value_only_mask.tolist() == [False, True]


def test_torch_advantage_normalization_matches_numpy_population_std():
    a = np.array([1, 2, 4, 7], np.float32)
    b = _minibatch_buffer(a.tolist())
    batches = list(b.minibatches(4, shuffle=False))
    np.testing.assert_allclose(batches[0].advantages.numpy(), (a - a.mean()) / (a.std() + 1e-8), atol=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_csr_roundtrip_complete_support(seed):
    rng = np.random.default_rng(seed)
    mask_np = np.zeros((4, 64), bool)
    for row in mask_np:
        row[rng.choice(64, size=int(rng.integers(0, 17)), replace=False)] = True
    mask = torch.from_numpy(mask_np)
    ids = torch.full((4, 16), -1, dtype=torch.int32)
    counts = torch.full((4,), -1, dtype=torch.int32)
    _dense_mask_to_csr(mask, ids, counts)
    selected = torch.tensor([2, 0, 2, 3, 1])
    restored = _csr_to_dense_selected(ids, counts, selected, 64)
    assert torch.equal(restored, mask[selected])
    assert torch.equal(counts, mask.sum(1).to(torch.int32))


@pytest.mark.parametrize("capacity", [2, 256])
def test_csr_overflow_raises_before_mutating_storage(capacity):
    mask = torch.ones((2, capacity + 1), dtype=torch.bool)
    ids = torch.full((2, capacity), 123, dtype=torch.int32)
    counts = torch.full((2,), 7, dtype=torch.int32)
    with pytest.raises(ValueError, match="overflow"):
        _dense_mask_to_csr(mask, ids, counts)
    assert (ids == 123).all() and (counts == 7).all()


def test_csr_exact_capacity_and_empty_batch():
    mask = torch.ones((1, 8), dtype=torch.bool)
    ids, counts = torch.empty((1, 8), dtype=torch.int32), torch.empty((1,), dtype=torch.int32)
    _dense_mask_to_csr(mask, ids, counts)
    assert counts.item() == 8
    assert torch.equal(_csr_to_dense_selected(ids, counts, torch.tensor([0]), 8), mask)
    _dense_mask_to_csr(torch.empty((0, 8), dtype=torch.bool), torch.empty((0, 8), dtype=torch.int32), torch.empty((0,), dtype=torch.int32))


class _Trainer:
    def state_dict(self):
        return {"policy": {"weight": torch.tensor([1.25])}, "num_rollout": 3}


def test_checkpoint_roundtrip_and_alias(tmp_path):
    saved = checkpoint.save_checkpoint(_Trainer(), TrainConfig(), 3, str(tmp_path))
    latest = tmp_path / "ckpt_latest.pt"
    assert latest.read_bytes() == Path(saved).read_bytes()
    state = torch.load(latest, weights_only=False)
    assert isinstance(state["train_cfg"], dict)
    assert state["policy"]["weight"].item() == 1.25


def test_checkpoint_failure_preserves_old_file_and_alias(tmp_path, monkeypatch):
    path = tmp_path / "ckpt_000003.pt"
    latest = tmp_path / "ckpt_latest.pt"
    path.write_bytes(b"previous valid checkpoint")
    latest.write_bytes(b"previous alias content")
    def interrupted_save(state, target):
        if hasattr(target, "write"):
            target.write(b"partial")
        else:
            Path(target).write_bytes(b"partial")
        raise OSError("simulated disk full")
    monkeypatch.setattr(checkpoint.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="disk full"):
        checkpoint.save_checkpoint(_Trainer(), TrainConfig(), 3, str(tmp_path))
    assert path.read_bytes() == b"previous valid checkpoint"
    assert latest.read_bytes() == b"previous alias content"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["ckpt_000003.pt", "ckpt_latest.pt"]


def test_alias_failure_preserves_previous_alias(tmp_path, monkeypatch):
    source, alias = tmp_path / "source.pt", tmp_path / "latest.pt"
    source.write_bytes(b"new")
    alias.write_bytes(b"old")
    def forbidden(*args, **kwargs):
        raise OSError("symlink not permitted")
    monkeypatch.setattr(os, "symlink", forbidden)
    with pytest.raises(OSError, match="not permitted"):
        checkpoint.update_checkpoint_alias(str(source), str(alias))
    assert alias.read_bytes() == b"old"


def test_checkpoint_uses_copy_fallback_without_symlink_permission(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise OSError("symlink not permitted")
    monkeypatch.setattr(os, "symlink", forbidden)
    path = checkpoint.save_checkpoint(_Trainer(), TrainConfig(), 3, str(tmp_path))
    assert (tmp_path / "ckpt_latest.pt").read_bytes() == Path(path).read_bytes()


def test_alias_rejects_self_without_deleting_source(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"valid")
    with pytest.raises(ValueError, match="different"):
        checkpoint.update_checkpoint_alias(str(source), str(source))
    assert source.read_bytes() == b"valid"


def test_alias_missing_source_does_not_replace_existing_alias(tmp_path):
    alias = tmp_path / "latest.pt"
    alias.write_bytes(b"old")
    with pytest.raises(FileNotFoundError):
        checkpoint.update_checkpoint_alias(str(tmp_path / "missing.pt"), str(alias))
    assert alias.read_bytes() == b"old"


def test_alias_relative_target_survives_directory_move(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    source, alias = old / "source.pt", old / "latest.pt"
    source.write_bytes(b"valid")
    try:
        checkpoint.update_checkpoint_alias(str(source), str(alias))
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this platform: {exc}")
    old.rename(tmp_path / "moved")
    assert (tmp_path / "moved" / "latest.pt").read_bytes() == b"valid"


def test_atomic_helper_write_failure_preserves_and_cleans(tmp_path):
    dest = tmp_path / "state.pt"
    dest.write_bytes(b"old")
    def failing(handle):
        handle.write(b"partial")
        raise RuntimeError("serialization failed")
    with pytest.raises(RuntimeError):
        atomic_write(dest, failing)
    assert dest.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [dest]


class _Single:
    def __init__(self):
        self._state = None
        self.moves = 0
        self.seeds = []
    def reset(self, *, seed=None):
        self._state = object()
        self.moves = 0
        self.seeds.append(seed)


class _Vector:
    """Deterministic lifecycle double; intentionally not a JunQi rule engine."""
    def __init__(self, N=1, horizon=1000, seats=(0, 1, 2, 3)):
        self.num_envs, self.horizon, self.seat_cycle = N, horizon, seats
        self._envs = [_Single() for _ in range(N)]
        self._done = np.zeros(N, bool)
        self.obs_spatial = np.zeros((N, 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), np.float32)
        self.obs_global = np.zeros((N, 4, OBS_GLOBAL_DIMS), np.float32)
    @property
    def done(self):
        return self._done
    def _fill_all_obs(self):
        for i, env in enumerate(self._envs):
            self.obs_spatial[i].fill(env.moves)
    def reset(self, seed_base=None):
        for i, env in enumerate(self._envs):
            env.reset(seed=None if seed_base is None else seed_base + i)
        self._done.fill(False)
        self._fill_all_obs()
        return self.obs_spatial, self.obs_global
    def current_seats(self):
        assert all(env._state is not None for env in self._envs)
        return [Seat(self.seat_cycle[env.moves % len(self.seat_cycle)]) for env in self._envs]
    def step(self, actions):
        assert not self.done.any(), "stepped a terminal environment"
        rewards = np.zeros((self.num_envs, 4), np.float32)
        acting = self.current_seats()
        for i, env in enumerate(self._envs):
            env.moves += 1
            if env.moves >= self.horizon:
                self._done[i] = True
                rewards[i, acting[i].value] = 1
        self._fill_all_obs()
        return self.obs_spatial, self.obs_global, rewards, self._done, []


class _Policy:
    def __init__(self):
        self.calls = []
    def eval(self):
        return self
    def act(self, spatial, global_, legal):
        assert legal.any(dim=1).all(), "policy called on an all-illegal terminal mask"
        self.calls.append(spatial[:, 0, 0, 0].tolist())
        B = spatial.shape[0]
        return torch.zeros(B, dtype=torch.int64), torch.zeros(B), torch.full((B, 1), 0.25)


@pytest.fixture
def lifecycle(monkeypatch):
    def masks(env, seats):
        result = np.zeros((env.num_envs, FLAT_ACTION_DIM), bool)
        result[~env.done, 0] = True
        return result
    monkeypatch.setattr(collector, "build_legal_mask_batch", masks)
    monkeypatch.setattr(collector, "UNROTATE_LUT", np.tile(np.arange(FLAT_ACTION_DIM), (4, 1)))
    def create(horizon=1000, T=2, N=1, seats=(0, 1, 2, 3)):
        env = _Vector(N, horizon, seats)
        policy = _Policy()
        buf = RolloutBuffer(num_envs=N, steps_per_env=T, adv_filt_thresh=0, adv_filt_rate=1)
        return env, policy, buf
    return create


def test_collection_continues_live_episode_across_rollouts(lifecycle):
    env, policy, buf = lifecycle()
    collector.collect_rollout(env, policy, buf, seed_base=17)
    collector.collect_rollout(env, policy, buf, seed_base=18)
    assert env._envs[0].moves == 4
    assert env._envs[0].seeds == [17]
    assert buf.obs_spatial[0, 0, 0, 0, 0] == 2


def test_collection_preserves_already_initialized_environment(lifecycle):
    env, policy, buf = lifecycle()
    env.reset(seed_base=7)
    env._envs[0].moves = 11
    env._fill_all_obs()
    collector.collect_rollout(env, policy, buf, seed_base=19)
    assert env._envs[0].moves == 13
    assert env._envs[0].seeds == [7]


def test_collection_reaches_terminal_beyond_one_rollout(lifecycle):
    env, policy, buf = lifecycle(horizon=3)
    collector.collect_rollout(env, policy, buf, seed_base=7)
    collector.collect_rollout(env, policy, buf, seed_base=8)
    assert buf.dones[:, 0].tolist() == [True, False]
    assert buf.rewards[0, 0] == 1
    assert env._envs[0].moves == 1


def test_collection_reset_seed_stream_reproducible_not_constant(lifecycle):
    streams = []
    for _ in range(2):
        env, policy, buf = lifecycle(horizon=1, T=3)
        collector.collect_rollout(env, policy, buf, seed_base=17)
        streams.append(env._envs[0].seeds)
    assert streams[0] == streams[1]
    assert len(set(streams[0])) == 4


def test_collection_terminal_last_step_skips_invalid_bootstrap(lifecycle):
    env, policy, buf = lifecycle(horizon=2, T=2)
    collector.collect_rollout(env, policy, buf, seed_base=1, auto_reset_done=False)
    assert buf.is_ready
    assert len(policy.calls) == 2
    assert buf.dones[-1, 0]


def test_collection_terminal_before_full_rollout_fails_clearly(lifecycle):
    env, policy, buf = lifecycle(horizon=1, T=2)
    with pytest.raises(RuntimeError, match="auto_reset_done"):
        collector.collect_rollout(env, policy, buf, seed_base=1, auto_reset_done=False)
    assert not buf.is_ready


def test_collection_passes_actual_bootstrap_seat(lifecycle, monkeypatch):
    env, policy, buf = lifecycle(T=1, seats=(0, 2))
    captured = {}
    actual_compute = buf.compute_returns
    def capture(values, **kwargs):
        captured.update(kwargs)
        return actual_compute(values, **kwargs)
    monkeypatch.setattr(buf, "compute_returns", capture)
    collector.collect_rollout(env, policy, buf, seed_base=1)
    assert captured["last_seats"].tolist() == [2]


def test_collection_rejects_terminal_start_without_auto_reset(lifecycle):
    env, policy, buf = lifecycle()
    env.reset(seed_base=1)
    env._done[0] = True
    with pytest.raises(RuntimeError, match="auto_reset_done"):
        collector.collect_rollout(env, policy, buf, auto_reset_done=False)
    assert env._done[0]
