"""The new workflow runs even when every legacy belief import is forbidden."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


def run_fresh(source):
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'OMP_NUM_THREADS': '1'},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_facts_metadata_import_is_independent_of_torch_cuda_and_legacy():
    run_fresh('''
        import importlib.abc, sys
        class ForbidDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if (fullname in ('torch', 'junqi_cuda', 'junqi_core.info_model')
                        or fullname.startswith('junqi_rl.belief.inference')
                        or fullname == 'junqi_rl.belief.legacy_joint'):
                    raise AssertionError('Unwanted import: ' + fullname)
        sys.meta_path.insert(0, ForbidDependencies())
        from junqi_rl.belief.joint import build_joint_input
        from junqi_rl.env import JunqiEnv
        from junqi_core.rules import Seat, ShowMode
        env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                       identity_encoding='known_only', record_public_history=True)
        env.reset(seed=7)
        inputs = build_joint_input([env.knowledge_snapshot(Seat.SOUTH)])
        assert inputs.type_domains.shape == (1, 75, 12)
        assert 'junqi_rl.gpu_rollout' not in sys.modules
    ''')


def test_training_checkpoint_and_replay_work_with_all_rule_imports_forbidden():
    run_fresh('''
        import importlib.abc, sys
        class ForbidRules(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if (fullname == 'junqi_core.info_model' or fullname in (
                        'junqi_rl.belief.inference', 'junqi_rl.belief.sampling',
                        'junqi_rl.belief.reveal_tracker', 'junqi_rl.belief.legacy_joint')):
                    raise AssertionError('Rule dependency: ' + fullname)
        sys.meta_path.insert(0, ForbidRules())
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from junqi_core.rules import ShowMode
        from junqi_rl.env import VectorJunqiEnv
        from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
        from junqi_rl.training.collector import collect_rollout
        from junqi_rl.training.rollout import RolloutBuffer
        from junqi_rl.training.ppo import PPOConfig, PPOTrainer
        from junqi_rl.training.checkpoint import export_evaluation_policy, load_evaluation_policy
        from junqi_rl.analysis.record import record_game_with_policy
        torch.manual_seed(11)
        cfg = JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1, embed_dim=16,
                             n_head=2, ff_factor=2, action_key_dim=8,
                             use_global_summary=False, identity_encoding='known_only')
        policy = JunqiNet(cfg)
        env = VectorJunqiEnv(2, show_mode=ShowMode.DARK, include_global=False,
                             identity_encoding='known_only')
        rollout = RolloutBuffer(num_envs=2, steps_per_env=4, obs_global_dims=0,
                                adv_filt_rate=1, adv_filt_thresh=0)
        before = policy.cls_token.detach().clone()
        collect_rollout(env, policy, rollout, device='cpu', seed_base=7)
        trainer = PPOTrainer(policy, PPOConfig(net=cfg, dtype='float32',
            minibatch_size=4, num_epochs_per_rollout=1,
            adv_filt_rate=1, adv_filt_thresh=0), device='cpu')
        trainer.train_epoch(rollout, rng=np.random.default_rng(7))
        assert trainer.num_train_step > 0
        assert not torch.equal(before, policy.cls_token)
        with tempfile.TemporaryDirectory() as directory:
            restored = load_evaluation_policy(export_evaluation_policy(
                policy, Path(directory) / 'policy.pt'))
            for key, value in policy.state_dict().items():
                assert torch.equal(value, restored.state_dict()[key])
            traj = record_game_with_policy(restored, rng_seed=7, device='cpu',
                                          show_mode=ShowMode.DARK, max_steps=8, top_k=4)
            assert traj.num_steps == 8
            assert traj.meta['identity_encoding'] == 'known_only'
            assert traj.meta['global_dims'] == 0 and traj.beliefs is None
            assert len(list(traj.replay_with_records())) == 8
        assert 'junqi_core.info_model' not in sys.modules
    ''')


def test_legacy_collectors_and_publishers_reject_facts_before_access():
    from junqi_rl.belief.inference import (
        refresh_beliefs_neural, guarded_refresh_beliefs_neural,
    )
    from junqi_rl.belief.sampling import MidgameBeliefSampler
    from junqi_rl.analysis.legacy_joint_search import BatchedBlueprint
    from scripts.legacy_train_joint_belief import collect

    class FactsOnly:
        identity_encoding = 'known_only'

        def __getattr__(self, name):
            raise AssertionError('Reached old machinery: ' + name)

    facts = FactsOnly()
    sampler = MidgameBeliefSampler(SimpleNamespace())
    calls = [lambda: refresh_beliefs_neural(facts, facts),
             lambda: guarded_refresh_beliefs_neural(facts, facts, facts),
             lambda: sampler(rollout_world=facts),
             lambda: BatchedBlueprint(facts, 'cpu'),
             lambda: collect(facts, rows=1, seed=1, device='cpu')]
    for call in calls:
        with pytest.raises(ValueError, match='known_only|facts-only'):
            call()
    assert sampler.steps == 0


def test_legacy_public_imports_remain_compatible():
    from junqi_rl import GpuRollout, GpuRolloutHistory, GpuWorld, VectorJunqiEnvGPU
    from junqi_rl.belief import BeliefBuffer, RevealTracker, refresh_beliefs_neural
    from junqi_rl.belief.buffer import BeliefBuffer as OriginalBuffer
    from junqi_rl.belief.reveal_tracker import RevealTracker as OriginalTracker
    assert BeliefBuffer is OriginalBuffer and RevealTracker is OriginalTracker
    assert callable(refresh_beliefs_neural)
    assert all(isinstance(cls, type) for cls in (
        GpuRollout, GpuRolloutHistory, GpuWorld, VectorJunqiEnvGPU))
