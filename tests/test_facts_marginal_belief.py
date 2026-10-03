"""Matched marginal control: public-only predictions, strict lineage, exact resume."""
import copy

import numpy as np
import pytest
import torch

from junqi_rl.belief.constraints import compile_constraints
from junqi_rl.belief.facts_dataset import concatenate_shards, file_sha256
from junqi_rl.belief.facts_joint_checkpoint import (
    facts_belief_state, facts_joint_state, load_facts_belief,
    load_facts_joint_belief, load_facts_marginal_belief,
)
from junqi_rl.networks.facts_marginal_belief_net import FactsMarginalBeliefNet
from scripts.train_facts_joint_belief import (
    _batch, _supervised_logits, evaluate, train, unknown_targets,
)
from tests.test_facts_joint_belief import checkpoint_fixture, fixture, tiny_model
from tests.test_train_facts_joint_belief import arguments, dataset  # noqa: F401


def marginal_model():
    return FactsMarginalBeliefNet(tiny_model().cfg).eval()


def test_predictions_cannot_depend_on_any_hidden_labels_or_prefix_masks():
    _, inputs, targets, obs = fixture()
    net = marginal_model()
    with torch.no_grad():
        reference = _supervised_logits(net, obs, inputs=inputs, targets=targets, masks=None)
        changed = _supervised_logits(net, obs, inputs=inputs,
                                     targets=(targets + 5) % 12, masks=None)
    torch.testing.assert_close(reference, changed, rtol=0, atol=0)
    np.testing.assert_array_equal(torch.isfinite(reference).numpy(), inputs.type_domains)
    with pytest.raises(ValueError, match='never receive teacher-prefix masks'):
        _supervised_logits(net, obs, inputs=inputs, targets=targets,
                           masks=torch.ones(1, 75, 12, dtype=torch.bool))
    with pytest.raises(TypeError):
        net(obs, inputs=inputs, targets=targets)


def test_marginal_has_no_prefix_parameters_or_unused_trainable_modules():
    _, inputs, targets, obs = fixture()
    joint = tiny_model()
    net = FactsMarginalBeliefNet(joint.cfg).train()
    assert set(joint.state_dict()) - set(net.state_dict()) == {'type_embedding.weight'}
    assert net.num_parameters() == joint.num_parameters() - 13 * joint.cfg.embed_dim
    assert not hasattr(net, 'type_embedding')
    logits = net.marginal_logits(obs, inputs=inputs)
    loss = -logits.log_softmax(-1).gather(-1, targets[..., None]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in net.parameters())


def test_matched_seed_initializes_every_shared_parameter_identically():
    from junqi_rl.networks.facts_joint_belief_net import FactsJointBeliefNet
    cfg = tiny_model().cfg
    torch.manual_seed(1203)
    joint = FactsJointBeliefNet(cfg)
    torch.manual_seed(1203)
    marginal = FactsMarginalBeliefNet(cfg)
    for key, value in marginal.state_dict().items():
        torch.testing.assert_close(joint.state_dict()[key], value, rtol=0, atol=0)


def test_static_scores_computed_once_but_exact_support_updates_every_token(monkeypatch):
    _, inputs, _, obs = fixture()
    net = marginal_model()
    solver = compile_constraints(inputs)
    encoded, decoded, lengths = [], [], []
    original_encode, original_decode, original_allowed = net.encode_facts, net.decode, solver.allowed_next

    def encode(*args):
        encoded.append(1)
        return original_encode(*args)

    def decode(*args):
        decoded.append(1)
        return original_decode(*args)

    def allowed(prefix):
        lengths.append(len(prefix))
        return original_allowed(prefix)

    monkeypatch.setattr(net, 'encode_facts', encode)
    monkeypatch.setattr(net, 'decode', decode)
    monkeypatch.setattr(solver, 'allowed_next', allowed)
    samples = net.sample(obs, inputs=inputs, n_samples=2, constraints=[solver],
                         generator=torch.Generator().manual_seed(31))
    assert encoded == decoded == [1]
    assert lengths == [index for index in range(75) for _ in range(2)]
    assert samples.shape == (1, 2, 75)
    for labels in samples[0].numpy():
        solver.validate(labels)
        for owner in (1, 2, 3):
            np.testing.assert_array_equal(np.bincount(labels[inputs.owners[0] == owner], minlength=12),
                                          inputs.initial_inventory[0, owner])
    assert 'not a globally conditioned independent product' in net.distribution_semantics


def test_marginal_solver_budget_failure_propagates():
    from junqi_rl.belief.constraints import ConstraintBudgetExceeded
    _, inputs, _, obs = fixture()

    class Failure:
        def allowed_next(self, prefix):
            raise ConstraintBudgetExceeded('marginal test budget')

    with pytest.raises(ConstraintBudgetExceeded, match='marginal test budget'):
        marginal_model().sample(obs, inputs=inputs, n_samples=1, constraints=[Failure()])


def test_generic_checkpoint_dispatch_and_strict_kind_boundary(tmp_path):
    joint, joint_state, policy = checkpoint_fixture(tmp_path)
    marginal = FactsMarginalBeliefNet(joint.cfg)
    saved = facts_belief_state(marginal, parent_policy_sha256=file_sha256(policy))
    path = tmp_path / 'marginal.pt'
    torch.save(saved, path)
    loaded = load_facts_belief(path, policy_path=policy)
    assert type(loaded) is FactsMarginalBeliefNet
    for key, value in marginal.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='architecture'):
        load_facts_joint_belief(path, policy_path=policy)
    with pytest.raises(ValueError, match='joint model'):
        facts_joint_state(marginal, parent_policy_sha256=file_sha256(policy))
    joint_path = tmp_path / 'joint.pt'
    torch.save(joint_state, joint_path)
    assert type(load_facts_belief(joint_path, policy_path=policy)) is type(joint)
    with pytest.raises(ValueError, match='architecture'):
        load_facts_marginal_belief(joint_path, policy_path=policy)
    for mutation in ('support', 'semantics', 'config', 'kind'):
        changed = copy.deepcopy(saved)
        if mutation == 'support':
            changed['input_spec']['training_support'] = 'teacher-prefix masks'
        elif mutation == 'semantics':
            del changed['input_spec']['sampling_distribution']
        elif mutation == 'config':
            changed['config']['n_head'] = True
        else:
            changed['architecture_id'] = joint.architecture_id
        torch.save(changed, path)
        with pytest.raises(ValueError, match='contract|config'):
            load_facts_belief(path, policy_path=policy)


def test_marginal_training_batches_never_read_cached_teacher_masks(dataset):
    shard = concatenate_shards(dataset[0], 'train')

    class NoTeacherMasks(dict):
        def __getitem__(self, name):
            if name == 'teacher_masks':
                raise AssertionError('Marginal path accessed hidden-label prefix masks')
            return super().__getitem__(name)

    guarded = NoTeacherMasks(shard)
    inputs, obs, targets, masks, unknown = _batch(guarded, np.arange(len(shard['targets'])),
                                               'cpu', belief_kind='marginal')
    assert masks is None
    assert int(unknown.sum()) == int(unknown_targets(shard).sum())
    metrics = evaluate(marginal_model(), guarded, batch_size=2, device='cpu')
    assert metrics['unknown_targets'] == int(unknown.sum())
    assert 'marginal_nll_per_unknown_piece' in metrics
    assert 'conditional_nll_per_unknown_piece' not in metrics


def test_marginal_training_exact_resume_and_frozen_test(dataset, tmp_path):
    from scripts.accept_facts_belief_training import assert_exact
    from scripts.evaluate_facts_joint_belief import main as audit
    full, partial, resumed = (tmp_path / name for name in ('full', 'partial', 'resumed'))
    train(arguments(dataset, full, '--belief-kind', 'marginal'))
    train(arguments(dataset, partial, '--belief-kind', 'marginal', '--epochs', '1'))
    report = train(arguments(dataset, resumed, '--belief-kind', 'marginal',
                             '--resume', str(partial / 'last.pt')))
    assert report['schema'] == 'junqi.facts-marginal-training.v1'
    expected = torch.load(full / 'last.pt', weights_only=False)
    actual = torch.load(resumed / 'last.pt', weights_only=False)
    for key in ('net', 'optimizer', 'best_model_state', 'rng', 'training_contract',
                'updates', 'epochs_completed', 'best_epoch', 'best_validation_nll'):
        assert_exact(expected[key], actual[key], key)
    selected = resumed / 'marginal-belief.pt'
    assert type(load_facts_belief(selected, policy_path=dataset[1])) is FactsMarginalBeliefNet
    assert not (resumed / 'joint-belief.pt').exists()
    before_hash = file_sha256(selected)
    args = ['--belief', str(selected), '--policy', str(dataset[1]), '--dataset', str(dataset[0]),
            '--output', str(resumed / 'test-audit.json'), '--device', 'cpu', '--num-threads', '1']
    audited = audit(args)
    assert audited['protocol']['schema'] == 'junqi.facts-marginal-final-test.v1'
    assert audited['protocol']['selection_performed_on_test'] is False
    assert 'marginal_nll_per_unknown_piece' in audited['test']
    assert file_sha256(selected) == before_hash
    assert audit(args) == audited
    with pytest.raises(ValueError, match='architecture'):
        train(arguments(dataset, tmp_path / 'wrong-kind', '--resume', str(partial / 'last.pt')))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires a CUDA device')
def test_cuda_marginal_parity_gradients_and_exact_feasible_sampling():
    _, inputs, targets, obs = fixture()
    cpu = marginal_model()
    gpu = marginal_model().to('cuda')
    gpu.load_state_dict(cpu.state_dict())
    with torch.no_grad():
        expected = cpu.marginal_logits(obs, inputs=inputs)
        actual = gpu.marginal_logits(obs.cuda(), inputs=inputs)
    torch.testing.assert_close(expected, actual.cpu(), atol=2e-5, rtol=2e-5)
    gpu.train()
    logits = gpu.marginal_logits(obs.cuda(), inputs=inputs)
    loss = -logits.log_softmax(-1).gather(-1, targets.cuda()[..., None]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in gpu.parameters())
    solver = compile_constraints(inputs)
    samples = gpu.eval().sample(obs.cuda(), inputs=inputs, n_samples=2, constraints=[solver],
        generator=torch.Generator(device='cuda').manual_seed(31))
    for labels in samples[0].cpu().numpy():
        solver.validate(labels)
