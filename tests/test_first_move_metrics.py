import copy
from types import SimpleNamespace

import pytest
import torch

from junqi_rl.training.first_move_metrics import POLICY_FIELDS, policy_statistics, summarize_games
from junqi_rl.training.game_progress import GameProgress


def test_conditional_policy_entropy_is_not_mixed_board_entropy():
    # Opposite deterministic decisions can look diverse after aggregation.
    p = torch.tensor([[1., 0, 0, 0], [0., 0, 0, 1.]])
    legal = torch.ones_like(p, dtype=torch.bool)
    before = torch.get_rng_state()
    stats = policy_statistics(p, legal, torch.zeros(2))
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(stats[:, 5], torch.zeros(2))
    assert torch.equal(stats[:, 6], torch.ones(2))
    uniform = policy_statistics(torch.full((2, 4), .25), legal,
                                torch.full((2,), 4.).log())
    torch.testing.assert_close(uniform[:, 1], torch.ones(2))
    torch.testing.assert_close(uniform[:, 5], torch.ones(2))
    torch.testing.assert_close(uniform[:, 6], torch.full((2,), .5))


def test_first_policy_survives_game_spanning_rollouts_and_resume():
    setups = torch.arange(240).reshape(2, 120).numpy()
    world = SimpleNamespace(setup_pool_revision=1, setup_pool_hashes=lambda: ['a', 'b'],
        setup_pool_snapshot=lambda: setups,
        move_counter_torch=lambda: torch.tensor([1, 2]),
        moves_since_last_combat_torch=lambda: torch.zeros(2),
        state=SimpleNamespace(max_num_moves=4000))
    g = GameProgress(2, 'cpu', track_arrangements=True, track_first_policy=True)
    g.begin(world, 1, reset_seed=10)
    first = torch.arange(20).reshape(2, 10).float()
    g.record_action(torch.tensor([129, 258]), torch.ones(2, dtype=torch.bool), policy_stats=first)
    state = copy.deepcopy(g.state_dict())
    resumed = GameProgress(2, 'cpu', track_arrangements=True, track_first_policy=True)
    resumed.load_state_dict(state)
    assert resumed.pool_revision == 1 and resumed.pool_hashes == ['a', 'b']
    resumed.begin(world, 1, reset_seed=999)
    resumed.record_action(torch.tensor([777, 888]), torch.ones(2, dtype=torch.bool),
                          policy_stats=first + 100)
    fired = torch.ones(2, dtype=torch.bool)
    resumed.record(world, dict(winner_team=torch.zeros(2), draw=torch.zeros(2, dtype=torch.bool)), fired, 0)
    resumed.after_reset(world, fired, reset_seed=11)
    rows = resumed.finish(1)
    assert rows[0]['first_action'] == 129
    assert rows[1]['first_policy'] == dict(zip(POLICY_FIELDS, first[1].tolist()))
    assert (resumed.first_policy_stats == 0).all()
    # v5 state must retain arrangement history as well.
    state.update(version=5, track_first_policy=False)
    old = GameProgress(2, 'cpu', track_arrangements=True)
    old.load_state_dict(state)
    assert old.pool_hashes == ['a', 'b']


def test_source_counts_sum_multiple_actions_and_missing_metrics_are_visible():
    records = [dict(first_action=a) for a in [129, 130, 258, -1]]
    result = summarize_games(records)
    assert result['sources']['counts'] == {1: 2, 2: 1}
    assert result['missing_first_actions'] == 1
    assert result['missing_first_policy'] == 3
    assert result['sources']['top1'] == pytest.approx(2/3)


def test_chunked_act_preserves_statistics_order():
    from junqi_rl.training.gpu_collector import _policy_act
    class Policy:
        track_first_policy = True
        def act(self, sp, gl, lm):
            self._last_entropy_t = sp[:, 0]
            self._last_first_policy_stats = sp.repeat(1, 10)
            return sp[:, 0].int(), sp[:, 0], sp[:, 0]
    p = Policy()
    sp = torch.arange(7.).reshape(-1, 1)
    _policy_act(p, sp, sp, sp.bool(), chunk_size=3)
    torch.testing.assert_close(p._last_first_policy_stats, sp.repeat(1, 10))


def test_diagnostics_do_not_change_network_actions_or_rng():
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
    net = JunqiNet(JunqiNetConfig(embed_dim=32, depth=1, n_head=4,
        cnn_channels=16, use_global_summary=False, identity_encoding='known_only')).eval()
    sp = torch.zeros(2, 317, 17, 17)
    gl = torch.empty(2, 0)
    mask = torch.zeros(2, 129*129, dtype=torch.bool)
    mask[:, [129, 130, 258, 259]] = True
    rng = torch.get_rng_state()
    with torch.no_grad():
        before = net.act(sp, gl, mask)
        end = torch.get_rng_state()
        torch.set_rng_state(rng)
        net.track_first_policy = True
        after = net.act(sp, gl, mask)
    assert torch.equal(end, torch.get_rng_state())
    for expected, actual in zip(before, after):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert net._last_first_policy_stats.shape == (2, 10)
    assert net._last_first_policy_stats.isfinite().all()
