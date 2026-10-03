"""Statistical unit is a layout pair, never a truncated match."""
import pytest
from scripts.compare_facts_joint import paired_statistics, bind_protocol, validated_games, write_json, setup_sha256


def game(pair, team, outcome):
    return dict(pair=pair, candidate_team=team, outcome=outcome, setup_sha256=str(pair).zfill(64))


def test_pair_bootstrap_preserves_dependence():
    games = [game(p, t, 'win' if t == 0 else 'loss') for p in range(5) for t in (0, 1)]
    result = paired_statistics(games)
    assert result['score'] == .5
    assert result['score_ci95'] == [.5, .5]
    assert result['win_rate_ci95'] == [.5, .5]


def test_unpaired_games_do_not_bias_team_balance():
    result = paired_statistics([game(0, 0, 'win'), game(0, 1, 'draw'), game(1, 0, 'loss')])
    assert result['completed_games'] == 3
    assert result['unpaired_completed_games'] == 1
    assert result['score'] == .75
    assert result['win_rate'] == .5
    assert result['score_ci95'] is None


def test_ongoing_and_duplicate_games_rejected():
    with pytest.raises(ValueError, match='completed'):
        paired_statistics([game(0, 0, 'ongoing')])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_statistics([game(0, 0, 'win'), game(0, 0, 'loss')])


def test_mixed_layout_pair_rejected():
    wrong = {**game(0, 1, 'loss'), 'setup_sha256': 'f' * 64}
    with pytest.raises(ValueError, match='same initial setup'):
        paired_statistics([game(0, 0, 'win'), wrong])


def test_worker_protocol_and_completed_game_provenance(tmp_path):
    protocol = dict(pairs=3, seed=42, joint_sha256='a' * 64)
    digest = bind_protocol(tmp_path, protocol)
    assert bind_protocol(tmp_path, protocol) == digest
    with pytest.raises(ValueError, match='another protocol'):
        bind_protocol(tmp_path, {**protocol, 'joint_sha256': 'b' * 64})
    record = {**game(0, 0, 'win'), 'setup_seed': 42, 'protocol_sha256': digest,
              'setup_sha256': setup_sha256(42)}
    path = tmp_path / 'game-000000-0.json'
    write_json(path, record)
    assert validated_games(tmp_path, protocol, digest) == [record]
    write_json(path, {**record, 'protocol_sha256': 'c' * 64})
    with pytest.raises(ValueError, match='provenance'):
        validated_games(tmp_path, protocol, digest)


def test_invalid_team_cannot_disappear_as_unpaired():
    with pytest.raises(ValueError, match='team'):
        paired_statistics([game(0, 2, 'win')])


def test_prior_protocol_excludes_all_planned_layouts_and_binds_content(tmp_path):
    from scripts.compare_facts_joint import excluded_layouts
    manifest = {'raw_games': [{'initial_setup_sha256': setup_sha256(31)}]}
    old = tmp_path / 'old'
    old.mkdir()
    write_json(old / 'protocol.json', {'schema': 'junqi.facts-joint-comparison.v1', 'seed': 42, 'pairs': 3})
    forbidden, evidence = excluded_layouts(manifest, [old])
    assert forbidden == {setup_sha256(s) for s in (31, 42, 43, 44)}
    assert evidence[0]['layout_count'] == 3 and len(evidence[0]['sha256']) == 64


def test_both_search_sides_require_exact_config_coverage_and_complete_moves():
    from copy import deepcopy
    import hashlib
    import json
    from scripts.compare_facts_joint import validate_side_metrics
    config = {'particles': 2, 'depth': 1}
    protocol = {'sides': {side: {'kind': 'belief_search', 'checkpoint_sha256': side}
                         for side in ('candidate', 'opponent')}, 'search_config': config}
    search = dict(decisions=3, successful_decisions=3, sample_budget_failures=0,
                  constraint_budget_failures=0, candidate_actions=6, legal_actions=6, config=config)
    record = {'sides_sha256': hashlib.sha256(json.dumps(protocol['sides'], sort_keys=True).encode()).hexdigest(),
              'side_metrics': {'candidate': deepcopy(search), 'opponent': deepcopy(search)}, 'plies': 6}
    validate_side_metrics(record, protocol)
    wrong = deepcopy(record)
    wrong['side_metrics']['opponent']['candidate_actions'] = 5
    with pytest.raises(ValueError, match='coverage'):
        validate_side_metrics(wrong, protocol)
    wrong = deepcopy(record)
    wrong['side_metrics']['opponent']['config']['particles'] = 1
    with pytest.raises(ValueError, match='coverage'):
        validate_side_metrics(wrong, protocol)
    wrong = deepcopy(record)
    wrong['plies'] = 7
    with pytest.raises(ValueError, match='plies'):
        validate_side_metrics(wrong, protocol)
    wrong = deepcopy(protocol)
    wrong['sides']['opponent']['checkpoint_sha256'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        validate_side_metrics(record, wrong)
