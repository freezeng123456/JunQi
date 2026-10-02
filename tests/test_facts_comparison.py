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
