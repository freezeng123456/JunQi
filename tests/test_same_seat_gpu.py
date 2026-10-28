"""Real CUDA collection, pending-action training and exact restart acceptance."""
import copy
import numpy as np
import pytest
import torch

pytest.importorskip("junqi_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import PPOTrainer, PPOConfig
from junqi_rl.training.rollout_gpu import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
from junqi_rl.training.game_progress import GameProgress, capture_runtime, restore_runtime
from junqi_rl.training.config import TrainConfig, validate_config


def components(storage):
    net = JunqiNetConfig(embed_dim=32, depth=1, n_head=4, cnn_channels=16,
                        use_global_summary=False, identity_encoding="known_only")
    cfg = TrainConfig(net=net, target_completed_games=100, eval_enabled=False,
                      random_opponent=False, reward_shaping=False)
    cfg.ppo = PPOConfig(net=net, dtype="float32", minibatch_size=16,
        adv_filt_rate=1., adv_filt_thresh=0., minibatch_group="timestep",
        adv_filter_scope="rollout", advantage_normalization="raw",
        return_perspective="same_seat", torch_compile=False)
    cfg.env.num_envs = 4
    cfg.env.steps_per_env = 9
    cfg.env.max_num_moves = 17
    cfg.env.use_gpu_rollout = True
    cfg.belief.enabled = False
    cfg.track_first_policy = True
    cfg.rollout.storage_mode = storage
    cfg.device = "cuda"
    validate_config(cfg)
    trainer = PPOTrainer(JunqiNet(net).cuda(), cfg.ppo, device="cuda")
    trainer.policy.track_first_policy = True
    world = GpuRollout(4, include_global=False, identity_encoding="known_only", max_num_moves=17)
    buf = RolloutBufferGPU(num_envs=4, steps_per_env=9, obs_global_dims=0,
        random_opponent=False, storage_mode=storage,
        history=world.create_rollout_history(9) if storage == "compact_history" else None,
        adv_filt_rate=1., adv_filt_thresh=0., minibatch_group="timestep",
        adv_filter_scope="rollout", advantage_normalization="raw", return_perspective="same_seat")
    progress = GameProgress(4, "cuda", lineage_id="same-seat-restart", track_first_policy=True)
    return world, trainer, buf, progress, np.random.default_rng(89), cfg


def advance(parts, index, reset=False):
    world, trainer, buf, progress, rng, cfg = parts
    old_pending = buf.pending.begin_rollout()
    collect_rollout_gpu_v2(world, trainer.policy, buf, reset_at_start=reset,
        seed_base=3000+index, reward_shaping=False, use_compile=False,
        autocast_dtype=torch.float32, game_progress=progress)
    assert torch.isfinite(buf.advantages_).all() and torch.isfinite(buf.returns_).all()
    assert not (buf.ready_mask & ~buf.input_valid).any()
    assert int((buf.input_valid & ~buf.ready_mask).sum()) == int(buf.pending.valid.sum())
    if not reset:
        torch.testing.assert_close(buf._carried["values"], old_pending["values"], rtol=0, atol=0)
        torch.testing.assert_close(buf._carried["behavior_log_probs"], old_pending["behavior_log_probs"], rtol=0, atol=0)
    expected_counts = buf.ready_mask.sum(1).cpu().tolist()
    emitted = []
    for batch in buf.minibatches(16, shuffle=False):
        emitted.append(len(batch.actions))
        assert batch.legal_mask.gather(1, batch.actions[:, None]).all()
        if batch.behavior_log_probs is not None:
            assert torch.equal(batch.behavior_log_probs.gather(1, batch.actions[:, None]).squeeze(1), batch.old_log_probs)
    assert emitted == [n for n in expected_counts if n]
    actions = buf.actions[4:].clone()
    before = trainer.num_train_step
    metrics = trainer.train_epoch(buf, rng=rng)
    assert trainer.num_train_step > before
    assert trainer._nan_skip_count == 0 and trainer._grad_nan_skip_count == 0
    assert all(torch.isfinite(t).all() for t in trainer.policy.state_dict().values())
    rows = progress.finish(index+1)
    return actions, rows


@pytest.mark.parametrize("storage", ["full_obs", "compact_history"])
def test_same_seat_collection_and_pending_resume_are_exact(tmp_path, storage):
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parts = components(storage)
    advance(parts, 0, reset=True)
    world, trainer, buf, progress, rng, cfg = parts
    assert progress.completed_games == 0 and buf.pending.valid.any()
    snapshot = copy.deepcopy(dict(trainer=trainer.state_dict(), runtime=capture_runtime(
        world, progress, rng, cfg, 0, rollout_buffer=buf)))
    torch.save(snapshot, tmp_path / "runtime.pt")
    snapshot = torch.load(tmp_path / "runtime.pt", map_location="cuda", weights_only=False)
    expected = [advance(parts, i) for i in (1, 2, 3)]
    final_world = copy.deepcopy(world.state_dict())
    resumed = components(storage)
    w2, t2, b2, p2, r2, c2 = resumed
    t2.load_state_dict(snapshot["trainer"])
    restore_runtime(snapshot["runtime"], w2, p2, r2, c2, rollout_buffer=b2)
    for i, (actions, rows) in zip((1, 2, 3), expected):
        actual_actions, actual_rows = advance(resumed, i)
        torch.testing.assert_close(actual_actions, actions, rtol=0, atol=0)
        assert actual_rows == rows
    for key, value in final_world["native"]["arrays"].items():
        np.testing.assert_array_equal(value, w2.state_dict()["native"]["arrays"][key])
    assert progress.summary() == p2.summary() and progress.completed_games >= 4
    for key, value in trainer.policy.state_dict().items():
        torch.testing.assert_close(value, t2.policy.state_dict()[key], rtol=0, atol=0)
    for key in buf.pending.TENSOR_FIELDS:
        torch.testing.assert_close(getattr(buf.pending, key), getattr(b2.pending, key), rtol=0, atol=0)
