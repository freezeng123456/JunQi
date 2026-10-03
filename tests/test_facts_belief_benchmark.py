"""Free sample summaries are separate from inference and use no teacher masks."""
import numpy as np
import pytest

from scripts.benchmark_facts_belief import sample_quality
from scripts.aggregate_facts_comparison import side_cost_summary


def test_sample_quality_excludes_publicly_known_identities():
    samples = np.zeros((3, 75), dtype=np.int64)
    samples[:, :2] = [[0, 1], [0, 2], [1, 1]]
    truth = np.full(75, 11)
    truth[:2] = [0, 1]
    unknown = np.zeros(75, dtype=bool)
    unknown[:2] = True
    result = sample_quality(samples, truth, unknown)
    assert result['uncertain_identities'] == 2
    assert result['sampled_true_label_frequency'] == pytest.approx(2 / 3)
    assert result['empirical_mode_accuracy'] == 1
    assert result['unique_complete_assignments'] == 3
    assert result['mean_pairwise_unknown_hamming'] == pytest.approx(2 / 3)
    changed = truth.copy()
    changed[2:] = 7
    assert sample_quality(samples, changed, unknown) == result
    with pytest.raises(ValueError, match='uncertain'):
        sample_quality(samples, truth, np.zeros(75, dtype=bool))


def test_side_cost_summary_never_credits_opponent_search_to_candidate():
    games = [{'side_metrics': {
        'candidate': {'decisions': 4, 'decision_seconds_total': 8, 'legal_actions': 20, 'candidate_actions': 20,
                      'blueprint_queries': {'value_positions': 40}},
        'opponent': {'decisions': 3, 'decision_seconds_total': 15, 'legal_actions': 15, 'candidate_actions': 15,
                     'blueprint_queries': {'value_positions': 30}}}}]
    cost = side_cost_summary(games)
    assert cost['candidate']['mean_decision_seconds'] == 2
    assert cost['opponent']['mean_decision_seconds'] == 5
    assert cost['candidate']['blueprint_queries']['value_positions'] == 40
    assert cost['opponent']['blueprint_queries']['value_positions'] == 30
