"""Fault injection must leave PPO usable for the next healthy minibatch."""
import datetime

import pytest

pytest.importorskip("torch")
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBatch


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    nc = JunqiNetConfig(
        cnn_channels=8,
        cnn_layers=1,
        depth=1,
        embed_dim=32,
        n_head=2,
        ff_factor=2,
        action_key_dim=8,
        use_cat_vf=True,
    )
    t = PPOTrainer(
        JunqiNet(nc), PPOConfig(net=nc, dtype="float32", num_epochs_per_rollout=1), device="cpu"
    )
    sp = torch.zeros(2, OBS_CHANNELS, 17, 17)
    gl = torch.zeros(2, OBS_GLOBAL_DIMS)
    legal = torch.zeros(2, 129 * 129, dtype=torch.bool)
    legal[:, [3, 7, 11, 19]] = True
    acts = torch.tensor([11, 7])
    t.policy.eval()
    with torch.no_grad():
        old = t.policy(sp, gl, legal, actions=acts)["action_log_prob"].detach().clone()
    b = RolloutBatch(
        obs_spatial=sp,
        obs_global=gl,
        legal_mask=legal,
        actions=acts,
        old_log_probs=old,
        advantages=torch.tensor([0.25, -0.25]),
        returns=torch.tensor([0.5, -0.5]),
        values=torch.zeros(2),
        adv_mask=torch.ones(2, dtype=torch.bool),
        value_only_mask=torch.zeros(2, dtype=torch.bool),
    )
    return t, b


class Buffer:
    uses_compact_history = False

    def __init__(self, b):
        self.b = b

    def minibatches(self, *a, **k):
        yield self.b
        yield self.b

    def stats(self):
        return {}


@pytest.mark.parametrize("split_value", [False, True])
@pytest.mark.parametrize("bad_step", [1, 2])
def test_gradient_skip_aggregation_and_recovery(bad_step, split_value):
    t, b = setup()
    call = [0]
    if split_value:
        b.value_obs_spatial = b.obs_spatial
        b.value_obs_global = b.obs_global
        b.value_returns = b.returns
        b.policy_value_indices = torch.arange(2)

    def hook(g):
        call[0] += 1
        return torch.full_like(g, float("nan")) if call[0] == bad_step else g

    before = [p.detach().clone() for p in t.policy.parameters()]
    handle = next(t._policy_unwrapped.parameters()).register_hook(hook)
    result = t.train_epoch(Buffer(b))
    handle.remove()
    assert result["train/grad_skip"] == 0.5
    assert result["train/grad_skip_total"] == 1
    assert result["train/num_updates"] == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, t.policy.parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in t.policy.parameters())


def worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        t, b = setup()
        before = [p.detach().clone() for p in t.policy.parameters()]
        handle = None
        if rank == 0:

            def corrupt(module, args, out):
                return {**out, "value": out["value"] * float("nan")}

            handle = t._policy_unwrapped.register_forward_hook(corrupt)
        t._update_step(b)
        if handle is not None:
            handle.remove()
        assert t._nan_skip_count == 1
        assert all(torch.equal(a, b) for a, b in zip(before, t.policy.parameters(), strict=True))
        t._update_step(b)
        assert any(not torch.equal(a, b) for a, b in zip(before, t.policy.parameters(), strict=True))
        for p in t.policy.parameters():
            assert torch.isfinite(p).all()
            other = p.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(p, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_one_rank_bad_forward_then_healthy_update(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
