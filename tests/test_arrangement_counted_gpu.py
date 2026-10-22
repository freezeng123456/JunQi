"""Real CUDA regression: setup refresh, terminal credits and exact continuation."""
import copy
import numpy as np
import pytest
import torch

pytest.importorskip("junqi_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.networks.arrangement_net import ArrangementNet, ArrangementNetConfig
from junqi_rl.arrangement.buffer import ArrangementBuffer
from junqi_rl.arrangement.sampling import generate_arrangements
from junqi_rl.arrangement.credits import credit_completed_games
from junqi_rl.training.arr_ppo import ArrangementPPOTrainer, ArrangementPPOConfig
from junqi_rl.training.ppo import PPOTrainer, PPOConfig
from junqi_rl.training.rollout_gpu import RolloutBufferGPU
from junqi_rl.training.game_progress import GameProgress, capture_rng, restore_rng
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2


def components():
    net_cfg = JunqiNetConfig(embed_dim=32, depth=1, n_head=4, cnn_channels=16,
        use_global_summary=False, identity_encoding="known_only")
    trainer = PPOTrainer(JunqiNet(net_cfg).cuda(), PPOConfig(net=net_cfg, dtype="float32",
        minibatch_size=16, value_minibatch_size=16, adv_filt_rate=1., adv_filt_thresh=0.), device="cuda")
    world = GpuRollout(4, include_global=False, identity_encoding="known_only", max_num_moves=17)
    rollout = RolloutBufferGPU(num_envs=4, steps_per_env=5, device="cuda", obs_global_dims=0,
        random_opponent=False, storage_mode="compact_history", history=world.create_rollout_history(5))
    progress = GameProgress(4, "cuda", lineage_id="dynamic-resume", track_arrangements=True)
    arr = ArrangementPPOTrainer(ArrangementNet(ArrangementNetConfig(depth=1, embed_dim=32,
        n_head=4)).cuda(), ArrangementPPOConfig(batch_size=32, num_epoch_per_train=1))
    return world, trainer, rollout, progress, np.random.default_rng(81), arr, ArrangementBuffer(storage_duration=6, device="cuda")


def advance(parts, index, reset=False):
    world, trainer, rollout, progress, rng, arr, buffer = parts
    if index % 2 == 0:
        buffer.filter(index)
        gen = generate_arrangements(32, arr.net, max_resample=0)
        assert not gen.fallback_mask.any()
        world.refresh_setup_pool_from_arrangements(gen.samples, gen.seat_idx, pool_size=64, seed=88+index)
        buffer.add_arrangements(gen.samples, gen.values, gen.ent_pred, gen.log_probs, gen.seat_idx, index)
    collect_rollout_gpu_v2(world, trainer.policy, rollout, reset_at_start=reset,
        seed_base=3000+index, reward_shaping=False, use_compile=False,
        autocast_dtype=torch.float32, game_progress=progress)
    actions = rollout.actions.clone()
    trainer.train_epoch(rollout, rng=rng)
    rows = progress.finish(index+1)
    assert credit_completed_games(buffer, progress.completed_setups, rows) == 4 * len(rows)
    buffer.process_data()
    metrics = arr.train_epoch(buffer)
    if metrics.get("arr_train/n_batches", 0):
        assert metrics["arr_train/grad_skip_total"] == 0
        buffer.ready_flags.zero_()
    return actions, rows


def test_dynamic_layout_resume_matches_uninterrupted_cuda_training(tmp_path):
    torch.manual_seed(910)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a = components()
    advance(a, 0, reset=True)
    advance(a, 1)
    world, trainer, _, progress, rng, arr, buffer = a
    snapshot = copy.deepcopy(dict(world=world.state_dict(), trainer=trainer.state_dict(),
        games=progress.state_dict(), rng=capture_rng(rng), arr=arr.state_dict(), buffer=buffer.state_dict()))
    torch.save(snapshot, tmp_path / "runtime.pt")
    snapshot = torch.load(tmp_path / "runtime.pt", map_location="cpu", weights_only=False)
    expected = [advance(a, i) for i in (2, 3, 4)]
    assert progress.completed_games >= 4
    assert arr._total_batches > 0
    end = copy.deepcopy(world.state_dict())
    b = components()
    b[0].load_state_dict(snapshot["world"])
    b[1].load_state_dict(snapshot["trainer"])
    b[3].load_state_dict(snapshot["games"])
    b[5].load_state_dict(snapshot["arr"])
    b[6].load_state_dict(snapshot["buffer"])
    restore_rng(snapshot["rng"], b[4])
    for i, (actions, rows) in zip((2, 3, 4), expected):
        actual_actions, actual_rows = advance(b, i)
        torch.testing.assert_close(actual_actions, actions, rtol=0, atol=0)
        assert actual_rows == rows
    for key, value in end["native"]["arrays"].items():
        np.testing.assert_array_equal(value, b[0].state_dict()["native"]["arrays"][key], err_msg=key)
    for model, restored in ((trainer.policy, b[1].policy), (arr.net, b[5].net)):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    for name in buffer._sample_fields + buffer._reward_fields:
        torch.testing.assert_close(getattr(buffer, name), getattr(b[6], name), rtol=0, atol=0)
