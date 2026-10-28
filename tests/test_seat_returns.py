"""Analytic and independent episode/seat oracle checks for deferred returns."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from junqi_rl.training.seat_returns import same_seat_returns


def _run(values, seats, *, rewards=None, dones=None, valid=None, gamma=1.0,
         gae_lambda=0.5, td_lambda=0.8):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim == 1:
        values = values[:, None]
    seats = torch.as_tensor(seats, dtype=torch.int8).reshape(values.shape)
    rewards = torch.zeros_like(values) if rewards is None else torch.as_tensor(
        rewards, dtype=torch.float32).reshape(values.shape)
    dones = torch.zeros_like(values, dtype=torch.bool) if dones is None else torch.as_tensor(
        dones, dtype=torch.bool).reshape(values.shape)
    valid = torch.ones_like(dones) if valid is None else torch.as_tensor(
        valid, dtype=torch.bool).reshape(values.shape)
    return same_seat_returns(values, rewards, dones, seats, valid, gamma=gamma,
                             gae_lambda=gae_lambda, td_lambda=td_lambda)


def _oracle(values, rewards, dones, seats, valid, gamma, gae_lambda, td_lambda):
    """Group complete episodes first, then independently visit each seat."""
    time, num_envs = values.shape
    adv = np.zeros_like(values)
    ret = values.copy()
    ready = np.zeros_like(valid)
    pending = np.full((num_envs, 4), -1, dtype=np.int64)
    for env in range(num_envs):
        start = 0
        endpoints = [t for t in range(time) if dones[t, env]] + [time]
        for end in endpoints:
            terminal = end < time
            for seat in range(4):
                rows = [t for t in range(start, end + int(terminal))
                        if valid[t, env] and seats[t, env] == seat]
                next_adv = next_td = 0.0
                if terminal:
                    next_value = rewards[end, env] * (
                        1 if seat % 2 == seats[end, env] % 2 else -1)
                    closed = True
                else:
                    closed = False
                    next_value = 0.0
                    if rows:
                        pending[env, seat] = rows[-1]
                terminal_successor = terminal
                for t in reversed(rows):
                    if closed:
                        discount = 1.0 if terminal_successor else gamma
                        delta = discount * next_value - values[t, env]
                        cont = 0.0 if terminal_successor else gamma
                        next_adv = delta + cont * gae_lambda * next_adv
                        next_td = delta + cont * td_lambda * next_td
                        adv[t, env] = next_adv
                        ret[t, env] = values[t, env] + next_td
                        ready[t, env] = True
                    next_value = values[t, env]
                    terminal_successor = False
                    closed = True
            start = end + 1
    return adv, ret, ready, pending


def test_distinct_lambdas_follow_only_own_decisions():
    adv, ret, ready, pending = _run([0.1, 900, 0.4, -800, 0.7], [0, 1, 0, 2, 0])
    torch.testing.assert_close(adv[[0, 2, 4], 0], torch.tensor([0.45, 0.3, 0.0]))
    torch.testing.assert_close(ret[[0, 2, 4], 0], torch.tensor([0.64, 0.7, 0.7]))
    assert ready[:, 0].tolist() == [True, False, True, False, False]
    assert pending.tolist() == [[4, 1, 3, -1]]


def test_teammate_value_does_not_become_same_seat_bootstrap():
    first = _run([0.1, 0.6, 0.3, 0.9], [0, 2, 0, 2])
    changed = _run([0.1, -990, 0.3, 990], [0, 2, 0, 2])
    torch.testing.assert_close(first[0][[0, 2]], changed[0][[0, 2]])
    torch.testing.assert_close(first[1][[0, 2]], changed[1][[0, 2]])


@pytest.mark.parametrize("terminal_reward", [-1.0, 0.0, 1.0])
def test_terminal_pays_all_seats_including_eliminated_player(terminal_reward):
    # Seat 0 disappears after its first move; teammate 2 ends the game.
    seats = [0, 1, 2, 3, 1, 2, 3, 2]
    adv, ret, ready, pending = _run(
        [0.0] * 8, seats, rewards=[0] * 7 + [terminal_reward],
        dones=[False] * 7 + [True], gae_lambda=1.0, td_lambda=1.0,
    )
    expected = torch.tensor([terminal_reward if s % 2 == 0 else -terminal_reward
                             for s in seats])
    torch.testing.assert_close(adv[:, 0], expected)
    torch.testing.assert_close(ret[:, 0], expected)
    assert ready.all() and (pending == -1).all()


def test_terminal_actor_loss_credits_earlier_opponent_win():
    adv, ret, ready, pending = _run([0.2, -0.3], [0, 1], rewards=[0, -1],
                                   dones=[False, True])
    torch.testing.assert_close(ret[:, 0], torch.tensor([1.0, -1.0]))
    assert ready.all() and (pending == -1).all()


def test_auto_reset_blocks_credit_and_only_final_episode_is_pending():
    adv, ret, ready, pending = _run(
        [0.1, 0.2, 0.3, 99.0, 100.0], [0, 1, 0, 0, 1],
        rewards=[0, -1, 0, 0, 0], dones=[False, True, False, False, False],
    )
    torch.testing.assert_close(ret[:2, 0], torch.tensor([1.0, -1.0]))
    assert ready[:, 0].tolist() == [True, True, True, False, False]
    assert pending.tolist() == [[3, 4, -1, -1]]
    torch.testing.assert_close(ret[2, 0], torch.tensor(99.0))


def test_carried_action_closes_once_in_next_rollout():
    # Before the boundary, only seat 0's first action is trainable.
    first = _run([0.1, 0.4, 0.2], [0, 1, 0])
    assert first[2][:, 0].tolist() == [True, False, False]
    assert first[3].tolist() == [[2, 1, -1, -1]]
    # Prefix rows retain the last action of each seat, not a duplicated
    # copy of every previous action. Terminal is from seat 2's perspective.
    second = _run([0.2, 0.4, 0, 0, 0.6], [0, 1, 2, 3, 2],
                  valid=[True, True, False, False, True],
                  rewards=[0, 0, 0, 0, 1], dones=[False] * 4 + [True])
    assert second[2][:, 0].tolist() == [True, True, False, False, True]
    torch.testing.assert_close(second[1][[0, 1, 4], 0], torch.tensor([1.0, -1.0, 1.0]))
    assert (second[3] == -1).all()


def test_eliminated_tail_survives_multiple_rollouts_without_false_loss():
    # Seat 0 is already gone from the board; its pending decision remains.
    middle = _run([0.4, 0, 0, 0, 0.2, -0.1], [0, 1, 2, 3, 1, 2],
                  valid=[True, False, False, False, True, True])
    assert not middle[2][0, 0] and middle[3][0, 0] == 0
    torch.testing.assert_close(middle[1][0, 0], torch.tensor(0.4))
    end = _run([0.4, 0.2, -0.1, 0, 0.5], [0, 1, 2, 3, 2],
               valid=[True, True, True, False, True], rewards=[0, 0, 0, 0, 1],
               dones=[False] * 4 + [True])
    assert end[2][0, 0] and end[1][0, 0] == 1.0
    assert (end[3] == -1).all()


def test_discount_counts_own_decisions_and_terminal_is_immediate():
    result = _run([0.1, 0.3], [0, 0], gamma=0.9, gae_lambda=1,
                  td_lambda=1, rewards=[0, 1], dones=[False, True])
    torch.testing.assert_close(result[1][:, 0], torch.tensor([0.9, 1.0]))


def test_partitioned_rollouts_learn_every_action_once_with_same_td_target():
    rng = np.random.default_rng(31)
    time, num_envs = 29, 3
    values = rng.uniform(-1, 1, (time, num_envs)).astype(np.float32)
    seats = rng.integers(0, 4, (time, num_envs), dtype=np.int8)
    dones = rng.random((time, num_envs)) < 0.12
    dones[-1] = True  # Finish every environment, including all deferred tails.
    rewards = rng.choice([-1.0, 0.0, 1.0], (time, num_envs)).astype(np.float32) * dones
    whole = _run(values, seats, rewards=rewards, dones=dones, gae_lambda=0, td_lambda=0)
    seen = []
    reconstructed = np.full_like(values, np.nan)
    prefix_v = np.zeros((4, num_envs), dtype=np.float32)
    prefix_s = np.broadcast_to(np.arange(4, dtype=np.int8)[:, None], prefix_v.shape).copy()
    prefix_valid = np.zeros_like(prefix_v, dtype=bool)
    prefix_ids = np.full(prefix_v.shape, -1, dtype=np.int64)
    all_ids = np.arange(time * num_envs).reshape(time, num_envs)
    for start in range(0, time, 7):
        end = min(start + 7, time)
        block_v = np.concatenate([prefix_v, values[start:end]])
        block_s = np.concatenate([prefix_s, seats[start:end]])
        block_valid = np.concatenate([prefix_valid, np.ones_like(dones[start:end])])
        block_ids = np.concatenate([prefix_ids, all_ids[start:end]])
        block_r = np.concatenate([np.zeros_like(prefix_v), rewards[start:end]])
        block_d = np.concatenate([np.zeros_like(prefix_valid), dones[start:end]])
        _, ret, ready, pending = _run(
            block_v, block_s, rewards=block_r, dones=block_d, valid=block_valid,
            gae_lambda=0, td_lambda=0,
        )
        selected = ready.numpy()
        learned_ids = block_ids[selected]
        assert (learned_ids >= 0).all()
        seen.extend(learned_ids.tolist())
        reconstructed.reshape(-1)[learned_ids] = ret.numpy()[selected]
        prefix_valid.fill(False)
        prefix_ids.fill(-1)
        for env in range(num_envs):
            for seat in range(4):
                idx = int(pending[env, seat])
                if idx >= 0:
                    prefix_v[seat, env] = block_v[idx, env]
                    prefix_ids[seat, env] = block_ids[idx, env]
                    prefix_valid[seat, env] = True
    assert sorted(seen) == list(range(time * num_envs))
    assert not prefix_valid.any()
    np.testing.assert_allclose(reconstructed, whole[1].numpy(), atol=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_scan_matches_independent_episode_oracle(seed):
    rng = np.random.default_rng(seed)
    shape = (37, 9)
    values = rng.uniform(-1, 1, shape).astype(np.float32)
    seats = rng.integers(0, 4, shape, dtype=np.int8)
    valid = rng.random(shape) > 0.2
    dones = (rng.random(shape) < 0.12) & valid
    rewards = rng.choice([-1.0, 0.0, 1.0], shape).astype(np.float32) * dones
    kwargs = dict(gamma=0.97, gae_lambda=0.5, td_lambda=0.8)
    expected = _oracle(values, rewards, dones, seats, valid, **kwargs)
    actual = _run(values, seats, rewards=rewards, dones=dones, valid=valid, **kwargs)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, torch.as_tensor(want), rtol=1e-5, atol=1e-6)


def test_empty_invalid_prefix_is_safe():
    result = _run([0.0] * 4, [-1] * 4, valid=[False] * 4)
    assert not result[2].any() and (result[3] == -1).all()
    assert (result[0] == 0).all()


def test_dense_shaping_reward_cannot_silently_enter_terminal_only_estimator():
    with pytest.raises(ValueError, match="terminal-only"):
        _run([0.0], [0], rewards=[0.05])


def test_invalid_terminal_row_is_rejected():
    with pytest.raises(ValueError, match="valid action"):
        _run([0.0], [0], valid=[False], dones=[True])
