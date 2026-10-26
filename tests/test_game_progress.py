import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi_rl.training.game_progress import GameProgress, capture_rng, restore_rng, setup_pool_index
from junqi_rl.analysis.protocol import paired_score_interval


def test_terminal_accounting_survives_reset_and_resume():
    hashes = [str(i) for i in range(10000)]
    moves = torch.tensor([40, 200, 4000])
    since = torch.tensor([1, 200, 20])
    world = SimpleNamespace(state=SimpleNamespace(max_num_moves=4000),
        setup_pool_hashes=lambda: hashes, move_counter_torch=lambda: moves,
        moves_since_last_combat_torch=lambda: since)
    games = GameProgress(3, 'cpu', lineage_id='fixture')
    games.begin(world, 2, reset_seed=3000)
    result = {'draw': torch.tensor([False, True, True]), 'winner_team': torch.tensor([1, -1, -1])}
    fired = torch.tensor([True, True, False])
    games.record_action(torch.tensor([11, 22, 33]), torch.ones(3, dtype=torch.bool))
    games.record(world, result, fired, 0)
    games.after_reset(world, fired, reset_seed=3001)
    moves[:] = torch.tensor([1, 1, 4000])
    fired = torch.tensor([False, False, True])
    games.record_action(torch.tensor([44, 55, 66]), torch.ones(3, dtype=torch.bool))
    games.record(world, result, fired, 1)
    games.after_reset(world, fired, reset_seed=3002)
    records = games.finish(1)
    assert [r['moves'] for r in records] == [40, 200, 4000]
    assert [r['termination_reason'] for r in records] == ['team_victory','no_combat_limit','total_move_limit']
    assert [r['first_action'] for r in records] == [11, 22, 33]
    assert [r['first_source'] for r in records] == [0, 0, 0]
    assert games.first_action_counts == {11: 1, 22: 1, 33: 1}
    assert [r['setup_reset_seed'] for r in records] == [3000, 3000, 3000]
    assert [r['initial_setup_sha256'] for r in records] == [hashes[setup_pool_index(i, 3000, len(hashes))] for i in range(3)]
    assert games.completed_games == 3 and games.environment_steps == 6
    saved = copy.deepcopy(games.state_dict())
    resumed = GameProgress(3, 'cpu')
    resumed.load_state_dict(saved)
    resumed.begin(world, 1, reset_seed=9999)
    moves[:] = torch.tensor([2, 2, 1])
    resumed.record(world, {'draw':torch.zeros(3,dtype=torch.bool),'winner_team':torch.zeros(3,dtype=torch.int8)},
                   torch.tensor([True, False, False]), 0)
    rows = resumed.finish(2)
    assert rows[0]['setup_reset_seed'] == 3001
    assert rows[0]['game_id'] == 3 and rows[0]['lineage_id'] == 'fixture'
    assert resumed.completed_games == 4 and resumed.environment_steps == 9
    assert resumed.first_action_counts[11] == 1


def test_rng_restores_all_host_streams():
    mb = np.random.default_rng(512)
    state = capture_rng(mb)
    def sample():
        return random.random(), np.random.random(), mb.permutation(20), torch.rand(15)
    expected = sample()
    restore_rng(state, mb)
    actual = sample()
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_paired_interval_preserves_layout_dependence():
    rows = [dict(setup_sha256=str(i), first_team=t, outcome='win' if t == 0 else 'loss')
            for i in range(50) for t in (0,1)]
    result = paired_score_interval(rows)
    assert result['layout_pairs'] == 50
    assert result['score_ci95_low'] == result['score_ci95_high'] == 0.5
    with pytest.raises(ValueError, match='both team'):
        paired_score_interval(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        paired_score_interval(rows + [rows[0]])
