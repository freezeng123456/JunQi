"""Common sampler scores use causal supports, and uncertainty respects assignments."""
import numpy as np
import pytest
import torch

from junqi_rl.belief.constraints import ConstraintBudgetExceeded, compile_constraints
from junqi_rl.networks.facts_marginal_belief_net import FactsMarginalBeliefNet
from scripts.score_facts_belief_samples import (
    aggregate_scores, empirical_brier, open_diagnostic_output, paired_score_differences, proposal_chain_score,
)
from tests.test_facts_joint_belief import fixture, tiny_model


@pytest.mark.parametrize('kind', ['marginal', 'joint'])
def test_chain_score_matches_actual_causal_normalization(kind):
    _, inputs, targets, obs = fixture()
    net = tiny_model()
    if kind == 'marginal':
        net = FactsMarginalBeliefNet(net.cfg).eval()
    masks = torch.from_numpy(compile_constraints(inputs).teacher_masks(targets[0].numpy()))[None]
    with torch.no_grad():
        if kind == 'marginal':
            # Public logits do not receive teacher labels. Prefix masks are
            # applied only by this independent offline reference scorer.
            logits = net.marginal_logits(obs, inputs=inputs).masked_fill(~masks, -torch.inf)
        else:
            logits = net.teacher_logits(obs, inputs=inputs, targets=targets, teacher_masks=masks)
        expected = logits.log_softmax(-1).gather(-1, targets[..., None])[0, :, 0].numpy()
    score = proposal_chain_score(net, obs, inputs, targets[0].numpy())
    np.testing.assert_allclose(score['token_log_probabilities'], expected, rtol=2e-5, atol=2e-5)
    assert score['proposal_chain_log_likelihood'] == pytest.approx(float(expected.sum()), abs=1e-4)
    assert score['truth_in_support'] is True
    assert score['accepted_distribution_exact_likelihood'] is False
    for index, allowed in enumerate(score['causal_allowed_labels']):
        assert allowed == torch.where(masks[0, index])[0].tolist()


def test_marginal_scoring_computes_neural_predictions_once_without_prefix(monkeypatch):
    _, inputs, targets, obs = fixture()
    net = FactsMarginalBeliefNet(tiny_model().cfg).eval()
    encoded, decoded = [], []
    original_encode, original_decode = net.encode_facts, net.decode

    def encode(public_obs, public_inputs):
        assert public_inputs is inputs
        encoded.append(1)
        return original_encode(public_obs, public_inputs)

    def decode(memory, context):
        decoded.append(1)
        return original_decode(memory, context)

    monkeypatch.setattr(net, 'encode_facts', encode)
    monkeypatch.setattr(net, 'decode', decode)
    proposal_chain_score(net, obs, inputs, targets[0].numpy())
    assert encoded == decoded == [1]


def test_chain_invalid_truth_and_exhausted_budget_fail_without_probability_repair(monkeypatch):
    _, inputs, targets, obs = fixture()
    wrong = np.zeros(75, dtype=np.int64)
    with pytest.raises(ValueError, match='inventory|domain'):
        proposal_chain_score(tiny_model(), obs, inputs, wrong)

    class Exhausted:
        stats = {}

        def validate(self, targets):
            pass

        def allowed_next(self, prefix):
            raise ConstraintBudgetExceeded('scoring bounded exact solver')

    monkeypatch.setattr('scripts.score_facts_belief_samples.compile_constraints',
                        lambda *args, **kwargs: Exhausted())
    with pytest.raises(ConstraintBudgetExceeded, match='bounded exact solver'):
        proposal_chain_score(tiny_model(), obs, inputs, targets[0].numpy())


def test_brier_known_identity_exclusion_bias_and_assignment_mc():
    labels = np.zeros((2, 75), dtype=np.int64)
    labels[1, :2] = 1
    truth = np.zeros(75, dtype=np.int64)
    one_unknown = np.zeros(75, dtype=bool)
    one_unknown[0] = True
    two_unknown = one_unknown.copy()
    two_unknown[1] = True
    one = empirical_brier(labels, truth, one_unknown, bootstrap_repeats=100, seed=7)
    both = empirical_brier(labels, truth, two_unknown, bootstrap_repeats=100, seed=7)
    assert both['empirical_brier'] == .5
    assert both['plugin_bias_estimate'] == .5
    assert both['unbiased_brier'] == 0
    # Perfectly correlated identities must not halve the Monte Carlo variance.
    assert both['mc_standard_error'] == one['mc_standard_error']
    changed = truth.copy()
    changed[2:] = 11
    assert empirical_brier(labels, changed, two_unknown, bootstrap_repeats=100, seed=7) == both
    wrong = np.ones_like(labels)
    assert empirical_brier(wrong, truth, one_unknown)['empirical_brier'] == 2
    assert empirical_brier(np.zeros_like(labels), truth, one_unknown)['empirical_brier'] == 0
    with pytest.raises(ValueError, match='at least two'):
        empirical_brier(labels[:1], truth, two_unknown)


def test_no_uncertain_identities_is_explicitly_unscored():
    labels = np.zeros((2, 75), dtype=np.int64)
    result = empirical_brier(labels, labels[0], np.zeros(75, dtype=bool))
    assert result['unknown_identities'] == 0
    assert result['empirical_brier'] is None
    assert result['mc_standard_error'] is None


def test_aggregate_keeps_games_and_training_seeds_separate():
    def model(digest, seed, score):
        return dict(checkpoint_sha256=digest, training_seed=seed, belief_kind='joint',
                    proposal_score={'proposal_chain_nll_per_unknown_identity': score},
                    free_sample_score={'empirical_brier': score, 'unbiased_brier': score})
    records = [dict(game_id=0, stage='early', models=[model('a', 1, 0), model('b', 2, .9)]),
               dict(game_id=1, stage='late', models=[model('a', 1, 1), model('b', 2, .8)])]
    summary = aggregate_scores(records, repeats=100)
    assert summary[0]['stages']['all']['empirical_brier']['games'] == 2
    assert summary[0]['stages']['all']['empirical_brier']['mean'] == .5
    assert summary[1]['training_seed'] == 2
    assert summary[0]['stages']['middle']['empirical_brier']['mean'] is None
    with pytest.raises(ValueError, match='one independently selected'):
        aggregate_scores(records + records[:1])


def test_progress_cannot_resume_another_state_selection(tmp_path):
    import json
    path = tmp_path / 'score.json'
    protocol = dict(states=2, selected_rows=[{'game_id': 1}, {'game_id': 2}])
    assert open_diagnostic_output(path, protocol)['states'] == []
    progress = path.with_suffix('.progress.json')
    progress.write_text(json.dumps(dict(protocol=protocol, complete=False, states=[{'game_id': 1}])))
    assert len(open_diagnostic_output(path, protocol)['states']) == 1
    progress.write_text(json.dumps(dict(protocol=protocol, complete=False, states=[{'game_id': 2}])))
    with pytest.raises(ValueError, match='frozen state prefix'):
        open_diagnostic_output(path, protocol)


def test_matched_differences_resample_paired_games_in_each_fixed_training_seed():
    records = []
    for game_id in range(3):
        models = []
        for training_seed in (10, 20):
            for kind, offset in (('marginal', 0), ('joint', -.2)):
                models.append(dict(training_seed=training_seed, belief_kind=kind,
                    proposal_score={'proposal_chain_nll_per_unknown_identity': game_id + offset},
                    free_sample_score={'empirical_brier': game_id + offset, 'unbiased_brier': game_id + offset}))
        records.append(dict(game_id=game_id, stage='early', models=models))
    paired = paired_score_differences(records, repeats=30)
    assert len(paired) == 2
    for row in paired:
        metric = row['stages']['all']['empirical_brier']
        assert metric['games'] == 3
        assert metric['mean'] == pytest.approx(-.2)
        assert metric['paired_game_bootstrap_ci95'] == pytest.approx([-.2, -.2])
