"""Preserve AdamW's scalar-counter placement across portable checkpoints."""
import copy

import pytest
import torch

from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import PPOConfig, PPOTrainer


def trainer(device="cpu"):
    config = JunqiNetConfig(embed_dim=16, depth=1, n_head=2, cnn_channels=8,
                           identity_encoding="known_only", use_global_summary=False)
    policy = JunqiNet(config).to(device)
    return PPOTrainer(policy, PPOConfig(net=config, dtype="float32"), device=device)


def optimizer_step(learner):
    learner.optimizer.zero_grad(set_to_none=True)
    next(learner.policy.parameters()).grad = torch.ones_like(next(learner.policy.parameters()))
    learner.optimizer.step()


def test_trainer_file_resume_deserializes_on_cpu_and_preserves_optimizer_progress(tmp_path, monkeypatch):
    original = trainer()
    optimizer_step(original)
    path = str(tmp_path / "checkpoint.pt")
    original.save(path)
    torch_load = torch.load
    observed = []

    def checked_load(*args, **kwargs):
        observed.append(kwargs["map_location"])
        return torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    restored = trainer()
    restored.load(path)
    assert observed == ["cpu"]
    optimizer_step(original)
    optimizer_step(restored)
    for left, right in zip(original.policy.parameters(), restored.policy.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    state = next(iter(restored.optimizer.state.values()))
    assert state["step"].item() == 2
    assert state["step"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA counter placement requires a real GPU")
@pytest.mark.parametrize("mode", ["ordinary", "capturable", "fused"])
def test_cuda_resume_repairs_historical_steps_but_preserves_capturable_and_fused(mode):
    original = trainer("cuda")
    optimizer_step(original)
    saved = copy.deepcopy(original.state_dict())
    group = saved["optimizer"]["param_groups"][0]
    group["capturable"] = mode == "capturable"
    group["fused"] = mode == "fused"
    group["foreach"] = mode != "fused"
    for state in saved["optimizer"]["state"].values():
        state["step"] = state["step"].cuda()  # old map_location='cuda' behavior
    restored = trainer("cuda")
    restored.load_state_dict(saved)
    for state in restored.optimizer.state.values():
        assert state["step"].device.type == ("cpu" if mode == "ordinary" else "cuda")
        assert state["step"].item() == 1
        assert state["exp_avg"].device.type == state["exp_avg_sq"].device.type == "cuda"
    optimizer_step(restored)
    assert next(iter(restored.optimizer.state.values()))["step"].item() == 2
    assert all(bool(torch.isfinite(parameter).all()) for parameter in restored.policy.parameters())
