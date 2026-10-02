"""Compare interrupted/resumed GPU self-play and optimizer updates to continuous execution."""
import copy
import hashlib
import numpy as np
import pytest
import torch
cuda = pytest.importorskip('junqi_cuda')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import PPOTrainer, PPOConfig
from junqi_rl.training.rollout_gpu import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
from junqi_rl.training.game_progress import GameProgress, capture_rng, restore_rng, setup_pool_index


def components():
    net_cfg = JunqiNetConfig(embed_dim=32, depth=1, n_head=4, cnn_channels=16,
        use_global_summary=False, identity_encoding='known_only')
    policy = JunqiNet(net_cfg).cuda()
    cfg = PPOConfig(net=net_cfg, dtype='float32', minibatch_size=16,
        value_minibatch_size=16, adv_filt_rate=1.0, adv_filt_thresh=0.0)
    trainer = PPOTrainer(policy, cfg, device='cuda')
    world = GpuRollout(4, include_global=False, identity_encoding='known_only', max_num_moves=17)
    buf = RolloutBufferGPU(num_envs=4, steps_per_env=9, device='cuda', obs_global_dims=0, random_opponent=False,
        storage_mode='compact_history', history=world.create_rollout_history(9))
    return world, trainer, buf, GameProgress(4, 'cuda', lineage_id='resume-test'), np.random.default_rng(89)


def advance(world, trainer, buf, progress, rng, index, reset=False):
    collect_rollout_gpu_v2(world, trainer.policy, buf, reset_at_start=reset,
        seed_base=3000+index, reward_shaping=False, use_compile=False,
        autocast_dtype=torch.float32, game_progress=progress)
    actions = buf.actions.clone()
    trainer.train_epoch(buf, rng=rng)
    rows = progress.finish(index+1)
    return actions, rows


def test_complete_gpu_save_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a = components()
    advance(*a, 0, reset=True)
    world, trainer, buf, progress, rng = a
    initial_types = world.state.copy_to_host()['piece_type_arr'].reshape(4, 120)
    initial_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in initial_types]
    assert initial_hashes == [progress.pool_hashes[setup_pool_index(i, 3000, len(progress.pool_hashes))] for i in range(4)]
    snapshot = copy.deepcopy(dict(world=world.state_dict(), trainer=trainer.state_dict(),
        games=progress.state_dict(), rng=capture_rng(rng)))
    torch.save(snapshot, tmp_path / 'runtime.pt')
    snapshot = torch.load(tmp_path / 'runtime.pt', map_location='cuda', weights_only=False)
    assert progress.completed_games == 0  # 9 of 17 moves; all games must survive resume.
    expected = [advance(*a, i) for i in (1,2)]
    assert [row['initial_setup_sha256'] for row in expected[0][1]] == initial_hashes
    current_types = world.state.copy_to_host()['piece_type_arr'].reshape(4, 120)
    current_seeds = progress.initial_seed.cpu().tolist()
    assert [hashlib.sha256(row.tobytes()).hexdigest() for row in current_types] == [
        progress.pool_hashes[setup_pool_index(i, current_seeds[i], len(progress.pool_hashes))] for i in range(4)]
    end = copy.deepcopy(world.state_dict())
    b = components()
    b[0].load_state_dict(snapshot['world']); b[1].load_state_dict(snapshot['trainer'])
    b[3].load_state_dict(snapshot['games']); restore_rng(snapshot['rng'], b[4])
    assert not b[0].state.has_belief_buffers
    for i, (actions, rows) in zip((1,2), expected):
        actual_actions, actual_rows = advance(*b, i)
        torch.testing.assert_close(actual_actions, actions, rtol=0, atol=0)
        assert actual_rows == rows
    for key, value in end['native']['arrays'].items():
        np.testing.assert_array_equal(value, b[0].state_dict()['native']['arrays'][key], err_msg=key)
    assert progress.summary() == b[3].summary()
    assert progress.completed_games == 4
    for key,value in trainer.policy.state_dict().items():
        torch.testing.assert_close(value, b[1].policy.state_dict()[key], rtol=0, atol=0)


def test_uniform_evaluation_preserves_seeded_actions_with_strict_cuda_determinism():
    from junqi_rl.analysis.random_eval import _game_uniform, _sample_uniform
    torch.use_deterministic_algorithms(True)
    legal = torch.rand(512, 37, generator=torch.Generator().manual_seed(82)) > .4
    legal[:, 5] = True
    seeds = torch.arange(512) + 6100200
    for ply in (0, 31, 3999):
        # Freeze the historical CPU float32 uniform reference independently
        # of the intentionally versioned nonuniform weighted sampler.
        cumulative = legal.float().cumsum(-1)
        threshold = _game_uniform(seeds, ply)[:, None] * cumulative[:, -1:]
        expected = (cumulative > threshold).to(torch.int32).argmax(-1).to(torch.int32)
        actual = _sample_uniform(legal.cuda(), seeds.cuda(), ply).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonuniform_sampling_is_strict_cuda_deterministic_and_batch_independent():
    from junqi_rl.analysis.random_eval import _sample_weights
    torch.use_deterministic_algorithms(True)
    rng = torch.Generator().manual_seed(991)
    weights = torch.rand(32, 16641, generator=rng).square()
    legal = torch.rand(32, 16641, generator=rng) > .8
    legal[:, 13] = True
    weights[:, 13] = 1.0
    weights[:, 17] = torch.finfo(torch.float32).tiny
    legal[:, 17] = True
    seeds = torch.arange(32) + 6200300
    for ply in (0, 317, 3999):
        expected = _sample_weights(weights, legal, seeds, ply, stream=19)
        actual = _sample_weights(weights.cuda(), legal.cuda(), seeds.cuda(), ply, stream=19)
        split = torch.cat([_sample_weights(weights[a:b].cuda(), legal[a:b].cuda(),
                          seeds[a:b].cuda(), ply, stream=19) for a, b in ((0, 7), (7, 23), (23, 32))])
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, split, rtol=0, atol=0)
        assert legal.gather(1, actual.cpu().long()[:, None]).all()
