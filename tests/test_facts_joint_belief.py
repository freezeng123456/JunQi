"""Facts/history belief causality, valid joint sampling and strict lineage."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest
import torch

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import PieceType, Seat, ShowMode
from junqi_rl.belief.facts_joint_checkpoint import (
    facts_joint_state, load_facts_joint_belief,
)
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.env import JunqiEnv
from junqi_rl.networks.facts_joint_belief_net import (
    FactsJointBeliefNet, FactsJointBeliefNetConfig, history_features,
    prepare_teacher_masks,
)
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.checkpoint import export_evaluation_policy


def tiny_model():
    torch.manual_seed(41)
    return FactsJointBeliefNet(FactsJointBeliefNetConfig(n_encoder_layer=1,
        n_decoder_layer=1, embed_dim=16, n_head=2, ff_factor=2,
        cnn_channels=8, cnn_layers=1)).eval()


def fixture():
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                   identity_encoding='known_only', record_public_history=True)
    observations = env.reset(seed=7)
    inputs = build_joint_input([env.knowledge_snapshot(Seat.SOUTH)])
    # The training label reader is deliberately outside the input builder.
    labels = torch.tensor([[TYPE_TO_INDEX[PieceType(int(env.state.piece_type_arr[pid]))]
                            for pid in inputs.piece_ids[0]]], dtype=torch.long)
    obs = torch.from_numpy(observations[Seat.SOUTH].spatial[None].copy())
    return env, inputs, labels, obs


def test_full_army_training_uses_facts_and_has_gradients():
    _, inputs, targets, obs = fixture()
    net = tiny_model().train()
    masks = prepare_teacher_masks(inputs, targets)
    logits = net.teacher_logits(obs, inputs=inputs, targets=targets, teacher_masks=masks)
    loss = -logits.log_softmax(-1).gather(-1, targets[..., None]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert net.type_embedding.weight.grad.norm() > 0
    assert net.domain_projection.weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in net.cnn.parameters())


def test_sampling_obeys_complete_inventory_and_encodes_once(monkeypatch):
    from junqi_rl.belief.constraints import compile_constraints
    _, inputs, _, obs = fixture()
    net = tiny_model()
    original = net.encode_facts
    calls = []

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(net, 'encode_facts', count)
    constraints = [compile_constraints(inputs)]
    samples = net.sample(obs, inputs=inputs, n_samples=2, constraints=constraints,
                         generator=torch.Generator().manual_seed(9))
    assert samples.shape == (1, 2, 75)
    assert len(calls) == 1
    for labels in samples[0].numpy():
        constraints[0].validate(labels)
        for owner in (1, 2, 3):
            np.testing.assert_array_equal(np.bincount(labels[inputs.owners[0] == owner], minlength=12),
                                          inputs.initial_inventory[0, owner])
    assert net.last_sample_metrics['token_host_syncs'] == 75


def test_future_labels_cannot_change_earlier_neural_logits():
    _, inputs, targets, obs = fixture()
    net = tiny_model()
    with torch.no_grad():
        memory, context = net.encode_facts(obs, inputs)
        first = net.decode(memory, context, targets)
        changed = targets.clone()
        changed[:, 10:] = (changed[:, 10:] + 1) % 12
        second = net.decode(memory, context, changed)
    torch.testing.assert_close(first[:, :11], second[:, :11])


def test_dead_piece_anchors_and_public_history_are_encoded():
    env, inputs, _, obs = fixture()
    net = tiny_model()
    dead_cells = inputs.cells.copy()
    dead_cells[:, 0] = -1
    alive = inputs.alive.copy()
    alive[:, 0] = False
    dead_input = replace(inputs, cells=dead_cells, alive=alive)
    _, before = net.encode_facts(obs, inputs)
    _, after = net.encode_facts(obs, dead_input)
    assert not torch.equal(before[:, 0], after[:, 0])
    assert np.count_nonzero(history_features(inputs)) == 0
    env.step(env.legal_action_ids()[0])
    moved = build_joint_input([env.knowledge_snapshot(Seat.SOUTH)])
    encoded = history_features(moved)
    assert encoded.shape == (1, 64, 24)
    assert encoded[0, -1, -1] == 1
    assert not encoded[0, :-1].any()


def test_constraint_failures_propagate_without_policy_fallback():
    from junqi_rl.belief.constraints import ConstraintBudgetExceeded
    _, inputs, _, obs = fixture()

    class Failure:
        def allowed_next(self, prefix):
            raise ConstraintBudgetExceeded('test budget')

    with pytest.raises(ConstraintBudgetExceeded, match='test budget'):
        tiny_model().sample(obs, inputs=inputs, n_samples=1, constraints=[Failure()])


def checkpoint_fixture(tmp_path):
    policy = JunqiNet(JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1,
        embed_dim=16, n_head=2, ff_factor=2, action_key_dim=8,
        identity_encoding='known_only', use_global_summary=False))
    policy_path = tmp_path / 'policy.pt'
    export_evaluation_policy(policy, policy_path)
    net = tiny_model()
    saved = facts_joint_state(net, parent_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                              updates=0)
    return net, saved, policy_path


def test_checkpoint_roundtrip_and_parent_hash(tmp_path):
    net, saved, policy = checkpoint_fixture(tmp_path)
    checkpoint = tmp_path / 'facts-belief.pt'
    torch.save(saved, checkpoint)
    loaded = load_facts_joint_belief(checkpoint, policy_path=policy)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    policy.write_bytes(b'changed parent')
    with pytest.raises(ValueError, match='parent policy mismatch'):
        load_facts_joint_belief(checkpoint, policy_path=policy)


@pytest.mark.parametrize('mutation', ['legacy', 'missing-history', 'bool-version', 'partial-config', 'bool-heads'])
def test_strict_checkpoint_rejects_semantic_and_config_drift(tmp_path, mutation):
    _, saved, policy = checkpoint_fixture(tmp_path)
    if mutation == 'legacy':
        saved['architecture_id'] = 'junqi.belief317.autoregressive.v1'
    elif mutation == 'missing-history':
        del saved['input_spec']['full_history_constraints']
    elif mutation == 'bool-version':
        saved['input_spec']['format_version'] = True
    elif mutation == 'partial-config':
        del saved['config']['history_window']
    elif mutation == 'bool-heads':
        saved['config']['n_head'] = True
    checkpoint = tmp_path / 'invalid.pt'
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='architecture|contract|config'):
        load_facts_joint_belief(checkpoint, policy_path=policy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires a CUDA device')
def test_cuda_teacher_parity_gradients_and_feasible_sampling():
    from junqi_rl.belief.constraints import compile_constraints
    _, inputs, targets, obs = fixture()
    cpu = tiny_model()
    gpu = tiny_model().to('cuda')
    gpu.load_state_dict(cpu.state_dict())
    constraints = [compile_constraints(inputs)]
    masks = prepare_teacher_masks(inputs, targets, constraints=constraints)
    with torch.no_grad():
        reference = cpu.teacher_logits(obs, inputs=inputs, targets=targets, teacher_masks=masks)
        logits = gpu.teacher_logits(obs.cuda(), inputs=inputs, targets=targets.cuda(), teacher_masks=masks.cuda())
    torch.testing.assert_close(reference, logits.detach().cpu(), atol=2e-5, rtol=2e-5)
    # Both parity models use inference BatchNorm statistics. Exercise training
    # and backward separately; training BN is a different mathematical map.
    gpu.train()
    logits = gpu.teacher_logits(obs.cuda(), inputs=inputs, targets=targets.cuda(), teacher_masks=masks.cuda())
    loss = -logits.log_softmax(-1).gather(-1, targets.cuda()[..., None]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in gpu.parameters()
               if parameter.grad is not None)
    samples = gpu.eval().sample(obs.cuda(), inputs=inputs, n_samples=2, constraints=constraints,
                                generator=torch.Generator(device='cuda').manual_seed(9))
    for labels in samples[0].cpu().numpy():
        constraints[0].validate(labels)
