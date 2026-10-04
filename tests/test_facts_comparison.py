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


@pytest.mark.parametrize('kinds,flags,backend,overlap', [
    (('joint', 'marginal'), [], 'scan', (False, False)),
    (('joint', 'marginal'), ['--inventory-backend', 'incremental', '--overlap-constraints'],
     'incremental', (True, False)),
    (('marginal', 'joint'), ['--inventory-backend', 'incremental', '--overlap-constraints'],
     'incremental', (False, True)),
])
def test_comparison_runtime_reaches_both_sides_and_binds_resume(tmp_path, monkeypatch,
                                                             kinds, flags, backend, overlap):
    """Exercise the real CLI, protocol binding, match construction and side validation."""
    from copy import deepcopy
    from dataclasses import asdict
    import hashlib
    import json
    from types import SimpleNamespace
    import numpy as np
    import scripts.compare_facts_joint as comparison
    import junqi_core.state as state_module
    import junqi_rl.analysis.facts_joint_search as search_module
    import junqi_rl.belief.facts_joint_checkpoint as belief_module
    import junqi_rl.env as env_module
    import junqi_rl.training.checkpoint as checkpoint_module

    layout = np.zeros(120, dtype=np.int64)
    layout_hash = hashlib.sha256(layout.tobytes()).hexdigest()
    monkeypatch.setattr(comparison, 'setup_sha256', lambda seed: layout_hash)
    monkeypatch.setattr(comparison, 'sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(comparison, 'inference_source_sha256', lambda: 'b' * 64)
    monkeypatch.setattr(comparison, 'selected_belief_provenance',
                        lambda path, net, digest: dict(kind='belief_search', belief_kind=net.belief_kind))
    monkeypatch.setattr(checkpoint_module, 'load_evaluation_policy', lambda *a, **kw: object())
    models = dict(zip(('candidate.pt', 'opponent.pt'), kinds))
    monkeypatch.setattr(belief_module, 'load_facts_belief',
                        lambda path, **kw: SimpleNamespace(belief_kind=models[path]))
    monkeypatch.setattr(search_module, 'FactsBlueprint', lambda *a: object())

    constructed = []
    class Search:
        def __init__(self, blueprint, model, config, *, seed, **runtime):
            constructed.append((model.belief_kind, runtime))
            self.config, self.runtime, self.decisions = asdict(config), runtime, 0

        def action(self, env, history):
            self.decisions += 1
            return 0

        def report(self):
            return dict(decisions=self.decisions, successful_decisions=self.decisions,
                        sample_budget_failures=0, constraint_budget_failures=0,
                        candidate_actions=self.decisions, legal_actions=self.decisions,
                        config=self.config, inventory_backend=self.runtime['inventory_backend'],
                        joint_overlap_constraints=self.runtime['overlap_constraints'])

    class Env:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(piece_type_arr=layout, turn=0, terminated=False,
                                         draw=False, winner_team=0)
        def reset(self, seed):
            pass

    def step(env, action):
        env.state.turn += 1
        env.state.terminated = env.state.turn == 2
        return env.state.turn

    monkeypatch.setattr(search_module, 'FactsJointSearch', Search)
    monkeypatch.setattr(search_module, 'step_world', step)
    monkeypatch.setattr(search_module, 'public_result', lambda result: result)
    monkeypatch.setattr(state_module, 'classify_termination', lambda state, result: 'test terminal')
    monkeypatch.setattr(env_module, 'JunqiEnv', Env)
    dataset, output = tmp_path / 'dataset', tmp_path / 'matches'
    dataset.mkdir()
    write_json(dataset / 'dataset-manifest.json',
               dict(schema='junqi.facts-belief-dataset.v1', status='complete',
                    raw_games=[{'initial_setup_sha256': 'f' * 64}]))
    argv = ['--policy', 'pi0.pt', '--belief', 'candidate.pt', '--opponent-belief', 'opponent.pt',
            '--dataset', str(dataset), '--output', str(output), '--device', 'cpu', '--pairs', '1']
    comparison.main(argv + flags)
    expected = {role: dict(inventory_backend=backend, overlap_constraints=enabled)
                for role, enabled in zip(('candidate', 'opponent'), overlap)}
    assert [runtime for kind, runtime in constructed] == list(expected.values()) * 2
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['inference_runtime'] == expected
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    records = validated_games(output, protocol, digest)
    assert len(records) == 2
    wrong = deepcopy(records[0])
    wrong['side_metrics']['opponent']['inventory_backend'] = (
        'incremental' if backend == 'scan' else 'scan')
    with pytest.raises(ValueError, match='inference runtime'):
        comparison.validate_side_metrics(wrong, protocol)
    # A completed output may resume only with the same effective per-side choices.
    comparison.main(argv + flags + ['--resume'])
    assert len(constructed) == 4
    changed_flags = [] if flags else ['--inventory-backend', 'incremental']
    with pytest.raises(ValueError, match='another protocol'):
        comparison.main(argv + changed_flags + ['--resume'])
