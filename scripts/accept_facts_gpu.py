#!/usr/bin/env python3
"""Bounded target-GPU acceptance of a real facts checkpoint; never skips CUDA.

This writes an audit plus a disposable training snapshot beside it. The source
checkpoint is read-only. Smoke training warm-starts its weights with a fresh
optimizer and a 17-move limit, so its game counts are acceptance data only.
Run the broader native parity pytest gates separately before a long experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "junqi_rl"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# Must precede the first CUDA context, including contexts created by imports.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_cuda(device_name_contains: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU acceptance requires real CUDA; CPU fallback/skip is forbidden")
    cuda = importlib.import_module("junqi_cuda")
    if cuda.get_gpu_count() < 1:
        raise RuntimeError("junqi_cuda reports no GPU")
    if not getattr(cuda, "OPTIONAL_GLOBAL_OBSERVATIONS", False):
        raise RuntimeError("rebuild junqi_cuda: zero-global observations are required")
    name = torch.cuda.get_device_name(0)
    if device_name_contains and device_name_contains.lower() not in name.lower():
        raise RuntimeError(f"expected device containing {device_name_contains!r}, found {name!r}")
    torch.cuda.set_device(0)
    cuda.init_tables()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return cuda, dict(name=name, capability=list(torch.cuda.get_device_capability(0)),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      native_sha256=_sha256(Path(cuda.__file__)))


def _validate_facts_policy(policy) -> dict:
    import torch
    from junqi_rl.checkpoint_compat import current_checkpoint_metadata

    metadata = current_checkpoint_metadata(policy)
    required = dict(identity_encoding="known_only", observation_channels=317,
                    observation_global_dims=0, global_summary_enabled=False,
                    observation_semantics_version=5,
                    architecture_id="junqi.graph317.facts.v1", policy_features={})
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"frozen facts baseline requires {key}={expected!r}")
    if not all(bool(torch.isfinite(p).all()) for p in policy.parameters()):
        raise ValueError("source checkpoint contains non-finite parameters")
    return metadata


def _inference_audit(checkpoint: Path, cuda):
    import numpy as np
    import torch
    from junqi_core.board import COMPACT_ACTION_DIM
    from junqi_core.observation import ObservationBuilder
    from junqi_core.rules import Seat, ShowMode
    from junqi_rl.env import JunqiEnv, rotate_to_compact_action_id
    from junqi_rl.env_gpu import _pack_state_arrays
    from junqi_rl.training.checkpoint import load_evaluation_policy

    cpu_policy = load_evaluation_policy(checkpoint, device="cpu")
    metadata = _validate_facts_policy(cpu_policy)
    gpu_policy = copy.deepcopy(cpu_policy).cuda().eval()
    metrics = []
    # Fresh games allow every channel, including the empty move-history ring,
    # to be compared. Advanced-state/history coverage lives in test_facts_gpu.
    for mode in ShowMode:
        envs = [JunqiEnv(show_mode=mode, include_global=False,
                        identity_encoding="known_only") for _ in range(4)]
        for index, env in enumerate(envs):
            env.reset(seed=2026100300 + index)
        builder = ObservationBuilder(include_global=False, identity_encoding="known_only")
        expected = np.stack([np.stack([builder.build(env.state, None, seat).spatial.copy()
                                      for seat in Seat]) for env in envs])
        state = cuda.DeviceGameStateBatch(len(envs))
        state.copy_from_host(_pack_state_arrays(envs))
        cuda.upload_state_observer_seats(state, np.tile(np.arange(4, dtype=np.int8), (len(envs), 1)))
        observations = cuda.DeviceObservationBatch(len(envs), False)
        cuda.build_observation_batch_resident(state, observations, np.int8(mode.value), True)
        actual, globals_ = observations.copy_to_host()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        if globals_.shape != (len(envs), 4, 0) or state.has_belief_buffers:
            raise AssertionError("facts observation allocated globals or rule belief buffers")
        legal = np.zeros((len(envs) * 4, COMPACT_ACTION_DIM), dtype=bool)
        for index, env in enumerate(envs):
            for seat in Seat:
                ids = [rotate_to_compact_action_id(int(a), seat) for a in env.legal_action_ids(seat)]
                legal[index * 4 + seat.value, ids] = True
        if not legal.any(-1).all():
            raise AssertionError("initial parity fixture has an empty action set")
        mask = torch.from_numpy(legal)
        spatial = torch.from_numpy(expected.reshape(-1, 317, 17, 17))
        globals_cpu = torch.empty(len(legal), 0)
        actions = mask.to(torch.int8).argmax(-1)
        with torch.inference_mode():
            cpu_out = cpu_policy(spatial, globals_cpu, mask, actions=actions)
            gpu_out = gpu_policy(spatial.cuda(), globals_cpu.cuda(), mask.cuda(), actions=actions.cuda())
        differences = {}
        for key in ("log_probs", "action_log_prob", "value"):
            left, right = cpu_out[key], gpu_out[key].cpu()
            if key == "log_probs":
                left, right = left[mask], right[mask]
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                raise AssertionError(f"non-finite {key} in checkpoint inference")
            torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-4)
            differences[key] = float((left - right).abs().max())
        if getattr(cpu_policy, "_nan_fwd_count", 0) or getattr(gpu_policy, "_nan_fwd_count", 0):
            raise AssertionError("policy invoked a non-finite fallback")
        metrics.append(dict(show_mode=mode.name, positions=len(legal), spatial_channels_compared=317,
                            observation_max_abs=float(np.max(np.abs(expected - actual))),
                            output_max_abs=differences))
    return cpu_policy, dict(checkpoint_meta=metadata,
                            parameters=sum(p.numel() for p in cpu_policy.parameters()),
                            strict_load=True, parameters_finite=True, cases=metrics,
                            observation_atol=1e-6, output_atol=3e-4, output_rtol=3e-4)


def _smoke_audit(cpu_policy, snapshot_path: Path) -> dict:
    import numpy as np
    import torch
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.training.game_progress import GameProgress, capture_rng, restore_rng
    from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
    from junqi_rl.training.ppo import PPOConfig, PPOTrainer
    from junqi_rl.training.rollout_gpu import RolloutBufferGPU

    random.seed(20261003)
    np.random.seed(20261003)
    torch.manual_seed(20261003)

    def components():
        policy = copy.deepcopy(cpu_policy).cuda()
        config = PPOConfig(net=policy.cfg, dtype="float32", minibatch_size=16,
                           value_minibatch_size=16, num_epochs_per_rollout=1,
                           adv_filt_rate=1.0, adv_filt_thresh=0.0)
        trainer = PPOTrainer(policy, config, device="cuda")
        world = GpuRollout(4, include_global=False, identity_encoding="known_only", max_num_moves=17)
        buffer = RolloutBufferGPU(num_envs=4, steps_per_env=9, device="cuda", obs_global_dims=0,
                                  random_opponent=False, adv_filt_rate=1.0, adv_filt_thresh=0.0,
                                  storage_mode="compact_history", history=world.create_rollout_history(9))
        progress = GameProgress(4, "cuda", lineage_id="facts-gpu-acceptance")
        return world, trainer, buffer, progress, np.random.default_rng(89)

    def advance(parts, index, *, reset=False):
        world, trainer, buffer, progress, rng = parts
        collect_rollout_gpu_v2(world, trainer.policy, buffer, reset_at_start=reset,
                              seed_base=3000 + index, reward_shaping=False,
                              use_compile=False, autocast_dtype=torch.float32, game_progress=progress)
        actions = buffer.actions.clone()
        trainer.train_epoch(buffer, rng=rng)
        rows = progress.finish(index + 1)
        if world.state.has_belief_buffers:
            raise AssertionError("training allocated rule belief buffers")
        if trainer._nan_skip_count or trainer._grad_nan_skip_count:
            raise AssertionError("smoke training skipped a non-finite update")
        if not all(bool(torch.isfinite(p).all()) for p in trainer.policy.parameters()):
            raise AssertionError("smoke training produced non-finite parameters")
        return actions, rows

    original = components()
    advance(original, 0, reset=True)
    world, trainer, _, progress, rng = original
    first_updates = trainer.num_train_step
    if first_updates <= 0 or not trainer.optimizer.state:
        raise AssertionError("optimizer made no progress before saving")
    first_delta = max(float((value.detach().cpu() - cpu_policy.state_dict()[name]).abs().max())
                      for name, value in trainer.policy.state_dict().items() if value.is_floating_point())
    if first_delta <= 0:
        raise AssertionError("no policy parameter changed")
    if progress.completed_games != 0:
        raise AssertionError("fixture must save four unfinished games at move 9 of 17")
    snapshot = copy.deepcopy(dict(world=world.state_dict(), trainer=trainer.state_dict(),
                                  games=progress.state_dict(), rng=capture_rng(rng)))
    torch.save(snapshot, snapshot_path)
    loaded = torch.load(snapshot_path, map_location="cuda", weights_only=False)
    expected = [advance(original, index) for index in (1, 2)]
    restored = components()
    restored[0].load_state_dict(loaded["world"])
    restored[1].load_state_dict(loaded["trainer"])
    restored[3].load_state_dict(loaded["games"])
    restore_rng(loaded["rng"], restored[4])
    if restored[1].num_train_step != first_updates or restored[3].completed_games != 0:
        raise AssertionError("saved optimizer/game counters were not restored")
    for index, (actions, rows) in zip((1, 2), expected):
        actual_actions, actual_rows = advance(restored, index)
        torch.testing.assert_close(actual_actions, actions, rtol=0, atol=0)
        if actual_rows != rows:
            raise AssertionError("resumed complete-game records differ from uninterrupted execution")
    for key, value in trainer.policy.state_dict().items():
        torch.testing.assert_close(value, restored[1].policy.state_dict()[key], rtol=0, atol=0)
    original_opt, restored_opt = trainer.optimizer.state_dict(), restored[1].optimizer.state_dict()
    for key, values in original_opt["state"].items():
        for field, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, restored_opt["state"][key][field], rtol=0, atol=0)
            elif value != restored_opt["state"][key][field]:
                raise AssertionError("resumed optimizer state differs")
    for key, value in world.state_dict()["native"]["arrays"].items():
        np.testing.assert_array_equal(value, restored[0].state_dict()["native"]["arrays"][key])
    if progress.summary() != restored[3].summary() or progress.completed_games != 4:
        raise AssertionError("completed-game continuation is inconsistent")
    if restored[1].num_train_step <= first_updates:
        raise AssertionError("optimizer did not advance after resume")
    torch.cuda.synchronize()
    return dict(initialization="frozen policy weights; fresh optimizer",
                acceptance_only=True, num_envs=4, steps_per_rollout=9, max_game_moves=17,
                unique_rollouts=3, executed_rollouts=5, updates_before_save=first_updates,
                updates_after_resume=restored[1].num_train_step, first_parameter_max_abs_change=first_delta,
                parameters_and_optimizer_match_uninterrupted=True, actions_match_uninterrupted=True,
                completed_records_match_uninterrupted=True, unfinished_games_at_save=4,
                progress=progress.summary(), snapshot=snapshot_path.name,
                snapshot_sha256=_sha256(snapshot_path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON audit path")
    parser.add_argument("--device-name-contains", default="V100")
    args = parser.parse_args(argv)
    snapshot = args.output.with_name(args.output.stem + "-smoke-runtime.pt")
    if snapshot.resolve() == args.checkpoint.resolve() or args.output.resolve() == args.checkpoint.resolve():
        parser.error("output paths must not overwrite the source checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, status="failed", checkpoint=args.checkpoint.name)
    started = time.monotonic()
    phase = "cuda_preflight"
    try:
        cuda, report["hardware"] = _require_cuda(args.device_name_contains)
        phase = "checkpoint_inference"
        report["checkpoint_sha256"] = _sha256(args.checkpoint)
        policy, report["inference"] = _inference_audit(args.checkpoint, cuda)
        phase = "training_resume"
        report["training_resume"] = _smoke_audit(policy, snapshot)
        if _sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise AssertionError("source checkpoint changed during acceptance")
        report["status"] = "passed"
    except Exception as error:
        report["failed_phase"] = phase
        report["error"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], audit=str(args.output),
                          failed_phase=report.get("failed_phase")), ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
