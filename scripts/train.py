#!/usr/bin/env python3
"""scripts/train.py — PPO training entry point for 四国军棋.

Usage
-----
# Basic training with defaults:
python scripts/train.py

# Override fields via YAML config:
python scripts/train.py --config configs/small.yaml

# Override individual fields on CLI:
python scripts/train.py --num_envs 64 --steps_per_env 256 --save_dir exps/run01

# Resume from checkpoint:
python scripts/train.py --resume exps/run01/ckpt_0010.pt --save_dir exps/run01

Config hierarchy (later sources override earlier):
    dataclass defaults → inherited YAML parent(s) → YAML file → CLI flags
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

# Ensure the project root is on sys.path (needed when invoked as
# ``python3 scripts/train.py`` — Python adds ``scripts/`` not the CWD).
# Also ensure ``junqi_rl/`` itself is on sys.path so the compiled
# ``junqi_cuda`` extension (installed by build_cuda.py to junqi_rl/junqi_cuda*.so)
# is importable as a top-level module — the convention used throughout the
# codebase. Without this, GPU-rollout paths silently fall back to "junqi_cuda
# import failed" and ``GpuRollout`` raises at construction time.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JUNQI_RL_PKG = os.path.join(_PROJECT_ROOT, "junqi_rl")
for _p in (_PROJECT_ROOT, _JUNQI_RL_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ---------------------------------------------------------------------------
# DDP helpers (no-ops in single-process mode)
# ---------------------------------------------------------------------------


def _ddp_world() -> tuple[int, int, int]:
    """Return (world_size, global_rank, local_rank) consulting torchrun env vars.

    Single-process when ``WORLD_SIZE`` env var is missing or 1; in that case
    returns ``(1, 0, 0)`` and we never touch ``torch.distributed``.
    """
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 1, 0, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return ws, rank, local_rank


def _init_distributed(cfg: "TrainConfig") -> tuple[int, int, int]:
    """Init NCCL process group and bind this rank to its local CUDA device.

    Mutates ``cfg`` so downstream code (env seed, save_dir, device) sees
    rank-aware values:

    * ``cfg.device`` is set to ``cuda:<local_rank>``
    * ``cfg.env.seed`` is offset by ``global_rank * 10_000`` so each rank
      generates distinct rollout trajectories (data-parallel diversity)
    * ``cfg.seed`` similarly offset for per-rank RNG reproducibility

    No-op for single-process runs.
    """
    world_size, global_rank, local_rank = _ddp_world()
    if world_size <= 1:
        return 1, 0, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
    else:
        cfg.device = "cpu"
    cfg.env.seed = cfg.env.seed + global_rank * 10_000
    cfg.seed = cfg.seed + global_rank * 10_000
    return world_size, global_rank, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


from junqi_rl.analysis.random_eval import (
    evaluate_paired_vs_random,
    evaluate_paired_head_to_head,
    evaluate_vs_random_cpu as evaluate_vs_random,
    evaluate_vs_random_gpu,
)
from junqi_rl.analysis.protocol import merge_evaluations
from junqi_rl.env import VectorJunqiEnv
from junqi_rl.networks.arrangement_net import (
    ArrangementNet, ArrangementNetConfig,
)
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.checkpoint_compat import validate_policy_checkpoint
from junqi_rl.arrangement.buffer import ArrangementBuffer
from junqi_rl.arrangement.sampling import generate_arrangements
from junqi_rl.training import (
    EMAPolicy,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    RolloutBufferGPU,
    collect_rollout,
    collect_rollout_gpu,
    collect_rollout_gpu_v2,
)
from junqi_rl.training.arr_ppo import (
    ArrangementPPOConfig, ArrangementPPOTrainer,
)
from junqi_rl.training.checkpoint import (
    save_checkpoint,
    update_checkpoint_alias,
)
from junqi_rl.training.config import (
    ArrangementTrainConfig,
    BeliefTrainConfig,
    EnvConfig,
    RolloutTrainConfig,
    TrainConfig,
    _dataclass_to_dict,
    _dict_to_dataclass,
    _nested_update,
    load_config,
)
from junqi_rl.training.logger import MultiCounter, TrainingLogger
from junqi_rl.training.rollout_gpu import observation_storage_dtype

__all__ = [
    "ArrangementNetConfig",
    "ArrangementPPOConfig",
    "ArrangementTrainConfig",
    "BeliefTrainConfig",
    "EnvConfig",
    "JunqiNetConfig",
    "PPOConfig",
    "RolloutTrainConfig",
    "TrainConfig",
    "_dataclass_to_dict",
    "_dict_to_dataclass",
    "_nested_update",
    "build_parser",
    "evaluate_vs_random",
    "evaluate_vs_random_gpu",
    "load_config",
    "save_checkpoint",
    "train",
]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 四国军棋 AI with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Path to YAML config file")
    parser.add_argument(
        "--resume",
        "--resume_cli",
        dest="resume_cli",
        default="",
        help="Checkpoint to resume from",
    )
    # Flat overrides — common fields
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps_per_env", type=int, default=None)
    parser.add_argument("--use_gpu_rollout", action="store_true", default=None,
                        help="Use GpuRollout (all-GPU game logic) instead of VectorJunqiEnv")
    parser.add_argument("--total_rollouts", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument(
        "--set",
        "--extra",
        dest="extra",
        nargs="*",
        metavar="KEY=VALUE",
        help="Nested overrides, e.g. ppo__clip_range=0.2",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the resolved config without training",
    )
    return parser


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    """Full PPO training loop."""
    # ---- Distributed init (no-op if WORLD_SIZE<=1) ----
    # Done first so cfg.device / cfg.seed / cfg.env.seed are rank-aware before
    # anything else uses them (RNG, logger, model construction, env seeding).
    world_size, global_rank, _local_rank = _init_distributed(cfg)
    is_rank0 = global_rank == 0
    is_distributed = world_size > 1
    if is_distributed and cfg.rollout.storage_mode == "compact_history":
        raise ValueError(
            "rollout.storage_mode=compact_history currently supports "
            "single-GPU training only"
        )

    # ---- Reproducibility ----
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- Directories ----
    # All ranks may need to read save_dir (e.g. for resume), but only rank 0
    # writes — so create on rank 0 and barrier the others.
    if is_rank0:
        os.makedirs(cfg.save_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()
    log_dir = os.path.join(cfg.save_dir, "logs")
    if is_rank0:
        from junqi_rl.analysis.league import LeaguePool

        league_pool: LeaguePool | None = LeaguePool(
            os.path.join(cfg.save_dir, "league.json"),
            max_entries=cfg.league_max_checkpoints,
        )
    else:
        league_pool = None

    # ---- Tee stdout/stderr to train.log so user can always tail the file ----
    # Under DDP each rank gets its own log file (train.log on rank 0,
    # train_rank{N}.log on others) so tracebacks never get scrambled and
    # rank 0's output stays clean for monitoring.
    _log_basename = "train.log" if is_rank0 else f"train_rank{global_rank}.log"
    _log_path = os.path.join(cfg.save_dir, _log_basename)
    _log_file = open(_log_path, "a", buffering=1)  # line-buffered

    class _Tee:
        """Write to both the original stream and the log file."""
        def __init__(self, stream):
            self._stream = stream
            self._file = _log_file
        def write(self, msg):
            self._stream.write(msg)
            self._file.write(msg)
        def flush(self):
            self._stream.flush()
            self._file.flush()

    sys.stdout = _Tee(sys.stdout)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr)  # type: ignore[assignment]

    # ---- Dump config (rank 0 only) ----
    # Write the runtime-resolved cfg to a SEPARATE file (not the source
    # cfg.yaml) so human-authored comments and structure in the source
    # are preserved across re-runs. ``cfg.yaml`` is the input; the
    # post-merge runtime view goes into ``cfg.runtime.yaml`` for audit.
    cfg_path = os.path.join(cfg.save_dir, "cfg.runtime.yaml")
    if is_rank0:
        with open(cfg_path, "w") as f:
            yaml.safe_dump(_dataclass_to_dict(cfg), f, default_flow_style=False)
        print(f"[train] Runtime config saved to {cfg_path}")
        if is_distributed:
            print(f"[train] DDP world_size={world_size} backend="
                  f"{dist.get_backend() if dist.is_initialized() else 'n/a'}")
        print(f"[train] device={cfg.device}  num_envs={cfg.env.num_envs}  "
              f"steps_per_env={cfg.env.steps_per_env}")
    else:
        print(f"[train rank{global_rank}] device={cfg.device} "
              f"seed={cfg.seed} env.seed={cfg.env.seed}")

    # ---- Device ----
    device = torch.device(cfg.device)
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_idx)
        if is_rank0:
            print(f"[train] GPU: {torch.cuda.get_device_name(gpu_idx)}")

    # ---- Model ----
    policy = JunqiNet(cfg.net)
    n_params = sum(p.numel() for p in policy.parameters())
    if is_rank0:
        print(f"[train] JunqiNet parameters: {n_params:,}")

    # ---- Trainer (handles EMA, optimiser, AMP) ----
    trainer = PPOTrainer(policy, cfg.ppo, device=device)

    # F-3: print the lr schedule trajectory at start so any 'lr touches floor
    # at rollout K of N' bug is visible from the very first second of training.
    # See exps/v35_ddp_combat_memory/launch.log.v35_buggy: there lr hit floor
    # at rollout 124 of 1500, silently freezing 92% of the training plan.
    if is_rank0:
        from junqi_rl.training.ppo import power_schedule as _pwsched
        unit = cfg.ppo.lr_schedule_unit
        # Probe a few representative steps depending on the unit.
        if unit == "rollout":
            probes = [0, 10, 50, 100, 500, 1000, 1500]
            unit_name = "R"
        else:
            # Per-grad-step: the actual count per rollout depends on
            # adv_filt_rate, num_envs, minibatch_size, world_size — we can't
            # know it exactly here; assume a typical 50-200 mb/rollout.
            probes = [0, 100, 1000, 5000, 10_000, 50_000, 200_000]
            unit_name = "step"
        lrs = [
            _pwsched(cfg.ppo.lr_coef, s, cfg.ppo.lr_decay,
                     cfg.ppo.lr_ceil, cfg.ppo.lr_floor)
            for s in probes
        ]
        floor_hit = next(
            (s for s, lr in zip(probes, lrs) if lr <= cfg.ppo.lr_floor + 1e-12),
            None,
        )
        sched = "  ".join(f"{u}={s}:{lr:.2e}" for u, s, lr in zip([unit_name] * len(probes), probes, lrs))
        print(f"[train] lr schedule (unit={unit}, decay={cfg.ppo.lr_decay}, "
              f"ceil={cfg.ppo.lr_ceil:.1e}, floor={cfg.ppo.lr_floor:.1e}):")
        print(f"[train]   {sched}")
        from junqi_rl.training.ppo import magnet_alpha as _magalpha
        mag_unit = cfg.ppo.temperature_schedule_unit
        mag_probes = [1, 2, 25, 100, 900, 3000]
        if mag_unit == "rollout":
            mag_vals = [
                _magalpha(
                    cfg.ppo.temperature_coef, t, cfg.ppo.temperature_decay,
                    cfg.ppo.temperature_floor,
                )
                for t in mag_probes
            ]
            mag_line = "  ".join(
                f"t={t}:{a:.4f}" for t, a in zip(mag_probes, mag_vals)
            )
            print(
                f"[train] magnet α (unit={mag_unit}, "
                f"shape={cfg.ppo.magnet_shape}, "
                f"coef={cfg.ppo.temperature_coef}, "
                f"decay={cfg.ppo.temperature_decay}, "
                f"floor={cfg.ppo.temperature_floor}):"
            )
            print(f"[train]   {mag_line}")
        else:
            print(
                f"[train] magnet α unit={mag_unit} "
                f"(legacy power_schedule with floor/ceil)"
            )
        print(
            f"[train] minibatch_group={cfg.ppo.minibatch_group}  "
            f"adv_filter_scope={cfg.ppo.adv_filter_scope}  "
            f"steps_per_env={cfg.env.steps_per_env}  "
            f"adv_filt_rate={cfg.ppo.adv_filt_rate}"
        )
        if floor_hit is not None and floor_hit < cfg.total_rollouts:
            print(f"[train]   ⚠️  lr will hit floor at {unit_name}={floor_hit} "
                  f"(< total_rollouts={cfg.total_rollouts}). "
                  f"Most of the planned training will run with frozen lr. "
                  f"Consider lowering ppo.lr_decay or raising ppo.lr_floor.")

    # ---- Environments ----
    if cfg.env.use_gpu_rollout:
        from junqi_rl.gpu_rollout import GpuRollout
        # Each rank pins its GpuRollout to its own GPU. Without this, every
        # rank would default to device_id=0 and we'd see "tensors on cuda:0
        # and cuda:1" runtime errors during the policy.act() call inside
        # collect_rollout_gpu_v2.
        env_device_id = (device.index if device.index is not None else 0) if device.type == "cuda" else 0
        canonical_styles = (
            tuple(cfg.fixed_setup_styles)
            if cfg.fixed_setup_styles
            else None
        )
        mixed_setup = cfg.mixed_setup
        mixed_own_team_styles = tuple(cfg.mixed_own_team_styles)
        env = GpuRollout(
            num_envs=cfg.env.num_envs,
            device_id=env_device_id,
            canonical_setup_styles=canonical_styles,
            mixed_setup=mixed_setup,
            mixed_own_team_styles=mixed_own_team_styles,
        )
        if is_rank0:
            if mixed_setup:
                mode_str = (
                    f"MIXED setup (own team styles={mixed_own_team_styles}, "
                    f"enemy team uniform-random)"
                )
            elif canonical_styles:
                mode_str = f"canonical_setup_styles={canonical_styles}"
            else:
                mode_str = "uniform-random setups"
            print(f"[train] GpuRollout: {cfg.env.num_envs} envs (all on-device, {mode_str})")
    else:
        env = VectorJunqiEnv(
            num_envs=cfg.env.num_envs,
            max_num_moves=cfg.env.max_num_moves,
        )
        if is_rank0:
            print(f"[train] VectorJunqiEnv: {cfg.env.num_envs} envs")

    # ---- Rollout buffer ----
    # The ``random_opponent`` flag mirrors cfg.random_opponent and gates two
    # behaviours inside RolloutBufferGPU:
    #   * vs-random (True): seats 1, 3 ran a uniform-random policy, so their
    #     advantages are zeroed after GAE (no learnable signal to extract).
    #   * self-play (False): seats 1, 3 ran the trainable policy, so we keep
    #     ALL transitions and let GAE flip V_{t+1}'s sign on cross-team steps.
    # This is the v33c fix for the "half-self-play" bug observed in v33b.
    rollout_random_opponent = cfg.random_opponent
    if cfg.env.use_gpu_rollout and device.type == "cuda":
        rollout_obs_dtype = observation_storage_dtype(cfg.ppo.get_dtype())
        rollout_history = (
            env.create_rollout_history(cfg.env.steps_per_env)
            if cfg.rollout.storage_mode == "compact_history"
            else None
        )
        rollout = RolloutBufferGPU(
            num_envs=cfg.env.num_envs,
            steps_per_env=cfg.env.steps_per_env,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            td_lambda=cfg.ppo.td_lambda,
            adv_filt_thresh=cfg.ppo.adv_filt_thresh,
            adv_filt_rate=cfg.ppo.adv_filt_rate,
            adv_filter_scope=cfg.ppo.adv_filter_scope,
            minibatch_group=cfg.ppo.minibatch_group,
            device=device,
            csr_legal_mask=cfg.rollout.csr_legal_mask,
            csr_k_max=cfg.rollout.csr_k_max,
            random_opponent=rollout_random_opponent,
            train_value_on_random_seats=cfg.train_value_on_random_seats,
            obs_storage_dtype=rollout_obs_dtype,
            storage_mode=cfg.rollout.storage_mode,
            history=rollout_history,
        )
        if is_rank0:
            mode = "vs-random" if rollout_random_opponent else "self-play"
            v_only = cfg.train_value_on_random_seats
            v_tag = "value-also-on-random-seats" if v_only else "legacy-no-value-on-random"
            shaping = cfg.reward_shaping
            shape_tag = "reward-shaping=ON" if shaping else "reward-shaping=OFF (terminal-only)"
            print(
                "[train] RolloutBufferGPU "
                f"(zero-CPU collect path, storage={cfg.rollout.storage_mode}, "
                f"obs={rollout_obs_dtype}, "
                f"mode={mode}, {v_tag}, {shape_tag})"
            )
            print(
                f"[train] Rollout storage: "
                f"{rollout.storage_bytes() / 1024**3:.2f} GiB"
            )
    else:
        rollout = RolloutBuffer(
            num_envs=cfg.env.num_envs,
            steps_per_env=cfg.env.steps_per_env,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            td_lambda=cfg.ppo.td_lambda,
            adv_filt_thresh=cfg.ppo.adv_filt_thresh,
            adv_filt_rate=cfg.ppo.adv_filt_rate,
            adv_filter_scope=cfg.ppo.adv_filter_scope,
            minibatch_group=cfg.ppo.minibatch_group,
            device=device,
        )

    # ---- Logger ----
    # Pass rank so non-rank-0 logger calls become no-ops; this avoids
    # 8 duplicate tensorboard event streams writing to the same dir.
    logger = TrainingLogger(
        log_dir=log_dir,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name or None,
        config=_dataclass_to_dict(cfg),
        rank=global_rank,
    )

    # ---- ArrangementNet (P0) ----
    # Only enabled when the GPU rollout is active: the CUDA setup-pool
    # refresh path requires the junqi_cuda extension.
    arr_enabled = bool(cfg.arr.enabled and cfg.env.use_gpu_rollout and device.type == "cuda")
    if cfg.arr.enabled and not arr_enabled:
        print("[train] arr.enabled=True ignored: requires use_gpu_rollout=True and CUDA.")
    arr_net: ArrangementNet | None = None
    arr_trainer: ArrangementPPOTrainer | None = None
    arr_buffer: ArrangementBuffer | None = None
    arr_ema: EMAPolicy | None = None
    if arr_enabled:
        if cfg.arr.n_arr % 4 != 0:
            raise ValueError(
                f"arr.n_arr must be a multiple of 4 (one sample per seat per pool entry), "
                f"got {cfg.arr.n_arr}"
            )
        arr_net = ArrangementNet(cfg.arr.net).to(device)
        n_arr_params = sum(p.numel() for p in arr_net.parameters())
        if is_rank0:
            print(f"[train] ArrangementNet parameters: {n_arr_params:,}")
        arr_trainer = ArrangementPPOTrainer(arr_net, cfg.arr.ppo)
        arr_ema = EMAPolicy(arr_net, decay=cfg.arr.ema_decay)
        arr_buffer = ArrangementBuffer(
            storage_duration=cfg.arr.storage_duration,
            device=device,
            use_cat_vf=cfg.arr.net.use_cat_vf,
            n_vf_cat=cfg.arr.net.n_vf_cat,
        )
        if is_rank0:
            print(f"[train] Arrangement training enabled: n_arr={cfg.arr.n_arr}, "
                  f"pool_size={cfg.arr.pool_size}, "
                  f"refresh_every={cfg.arr.refresh_every}, "
                  f"storage_duration={cfg.arr.storage_duration}")

    # ---- Belief net (P1) setup ------------------------------------------
    belief_enabled = bool(cfg.belief.enabled and cfg.env.use_gpu_rollout and device.type == "cuda")
    if cfg.belief.enabled and not belief_enabled:
        print("[train] belief.enabled=True ignored: requires use_gpu_rollout=True and CUDA.")
    belief_net = None
    belief_trainer = None
    belief_buffer = None
    reveal_tracker = None
    if belief_enabled:
        from junqi_rl.belief.buffer import BeliefBuffer
        from junqi_rl.belief.reveal_tracker import RevealTracker
        from junqi_rl.networks.belief_net import (
            BeliefNet,
            BeliefNetConfig,
            belief_net_config_from_dict,
        )
        from junqi_rl.training.belief_ppo import (
            BeliefPPOConfig,
            BeliefPPOTrainer,
        )

        belief_net_cfg = (
            belief_net_config_from_dict(cfg.belief.net)
            if isinstance(cfg.belief.net, dict)
            else (cfg.belief.net or BeliefNetConfig())
        )
        belief_ppo_cfg = (
            BeliefPPOConfig(**cfg.belief.ppo)
            if isinstance(cfg.belief.ppo, dict)
            else (cfg.belief.ppo or BeliefPPOConfig())
        )
        belief_net = BeliefNet(belief_net_cfg).to(device)
        n_belief_params = sum(p.numel() for p in belief_net.parameters())
        if is_rank0:
            print(f"[train] BeliefNet parameters: {n_belief_params:,}")
        belief_trainer = BeliefPPOTrainer(
            belief_net, belief_ppo_cfg, device=device,
        )
        belief_buffer = BeliefBuffer(
            capacity=cfg.belief.buffer_capacity,
            seed=cfg.seed,
        )
        reveal_tracker = RevealTracker(belief_buffer=belief_buffer)
        if is_rank0:
            print(f"[train] Belief training enabled: "
                  f"buffer_capacity={cfg.belief.buffer_capacity}, "
                  f"refresh_every={cfg.belief.refresh_every}, "
                  f"warmup_rollouts={cfg.belief.warmup_rollouts}")

    # ---- Resume ----
    start_rollout = 0
    if cfg.resume:
        if is_rank0:
            print(f"[train] Resuming from {cfg.resume}")
        resume_sd = torch.load(cfg.resume, map_location=device, weights_only=False)
        trainer.load_state_dict(resume_sd)
        if arr_trainer is not None and "arrangement" in resume_sd:
            arr_sd = resume_sd["arrangement"]
            arr_trainer.load_state_dict(arr_sd["trainer"])
            if arr_ema is not None and arr_sd.get("ema") is not None:
                arr_ema.load_state_dict(arr_sd["ema"])
            if is_rank0:
                print("[train] Resumed ArrangementNet state")
        elif arr_trainer is not None and is_rank0:
            print("[train] Resume checkpoint has no ArrangementNet state; starting arr fresh")
        if belief_trainer is not None and "belief" in resume_sd:
            belief_trainer.load_state_dict(resume_sd["belief"])
            if is_rank0:
                print("[train] Resumed BeliefNet state")
        elif belief_trainer is not None and is_rank0:
            print("[train] Resume checkpoint has no BeliefNet state; starting belief fresh")
        start_rollout = trainer.num_rollout
        if is_rank0:
            print(f"[train] Resumed: rollout={start_rollout}, "
                  f"train_step={trainer.num_train_step}")

    # ---- Signal handling (graceful stop on SIGINT/SIGTERM) ----
    _stop_requested = [False]

    def _handle_signal(sig, frame):
        print(f"\n[train] Signal {sig} received — will stop after current rollout.")
        _stop_requested[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- RNG for minibatch shuffling ----
    rng = np.random.default_rng(cfg.seed)

    # ---- Training loop ----
    mc = MultiCounter(step=start_rollout)
    t_rollout_start = time.time()

    if is_rank0:
        print(f"[train] Starting training loop: rollouts {start_rollout} → {cfg.total_rollouts}")
        if is_distributed:
            global_steps = cfg.env.num_envs * cfg.env.steps_per_env * world_size
            print(f"[train] Per-rank env steps per rollout: "
                  f"{cfg.env.num_envs * cfg.env.steps_per_env:,} "
                  f"(global: {global_steps:,} across {world_size} ranks)")
            # F-2: num_envs is interpreted PER-RANK, not global. v33 used
            # num_envs=512/rank * 6 ranks = 3072 global; v35 uses 384/rank *
            # 2 ranks = 768 global. If you want a fixed *global* env budget,
            # divide by world_size before writing cfg.env.num_envs.
            print(f"[train] (Note: cfg.env.num_envs={cfg.env.num_envs} is "
                  f"PER-RANK; cluster-wide envs = {cfg.env.num_envs}×{world_size} = "
                  f"{cfg.env.num_envs * world_size}.)")
        else:
            print(f"[train] Total env steps per rollout: "
                  f"{cfg.env.num_envs * cfg.env.steps_per_env:,}")

    # Early-stop state: number of *consecutive* evals below threshold.
    # Only consulted when cfg.early_stop_win_rate > 0 and rollout >= min.
    _early_stop_low_streak = 0
    _best_eval_win_rate = -1.0
    _eval_baseline_policy = None
    if is_rank0 and cfg.eval_baseline_ckpt:
        ckpt_path = cfg.eval_baseline_ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"eval_baseline_ckpt not found: {ckpt_path}")
        baseline_state = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        baseline_net_cfg = baseline_state["cfg"].net
        _eval_baseline_policy = JunqiNet(baseline_net_cfg).to(device)
        validate_policy_checkpoint(
            _eval_baseline_policy,
            baseline_state,
            source=f"eval baseline {ckpt_path}",
        )
        _eval_baseline_policy.load_state_dict(baseline_state["policy"])
        _eval_baseline_policy.eval()
        print(
            f"[train] Frozen baseline for h2h: {ckpt_path} "
            f"(embed={baseline_net_cfg.embed_dim}, depth={baseline_net_cfg.depth})"
        )

    for rollout_idx in range(start_rollout, cfg.total_rollouts):
        mc.step = rollout_idx

        # ---- P0 arrangement-net refresh (before collect) ----
        if arr_enabled and (rollout_idx - start_rollout) % cfg.arr.refresh_every == 0:
            t_arr0 = time.time()
            # Ensure old rows are dropped so new ones can be tracked.
            arr_buffer.filter(current_step=rollout_idx)
            # Use EMA weights for generation (matches Ataraxos convention).
            arr_gen_net = arr_ema.model if arr_ema is not None else arr_net
            arr_gen = generate_arrangements(
                n_sample=cfg.arr.n_arr,
                model=arr_gen_net,
                rng_seed=cfg.env.seed + rollout_idx * 17,
            )
            # Push the pool to CUDA so reset_terminated_envs picks from it.
            pool_size = env.refresh_setup_pool_from_arrangements(
                arr_gen.samples,
                arr_gen.seat_idx,
                pool_size=cfg.arr.pool_size,
                seed=cfg.env.seed + rollout_idx * 17,
            )
            # Record the generation-time info so process_data has targets.
            arr_buffer.add_arrangements(
                arrangements=arr_gen.samples,
                values=arr_gen.values,
                ents=arr_gen.ent_pred,
                log_probs=arr_gen.log_probs,
                seat_idx=arr_gen.seat_idx,
                step=rollout_idx,
            )
            mc.inc("arr/pool_size", float(pool_size))
            mc.inc(
                "arr/n_unique_pool",
                float(getattr(env, "last_setup_pool_unique", pool_size)),
            )
            mc.inc("arr/retry_rate", float(arr_gen.stats["retry_rate"]))
            mc.inc("arr/fallback_rate", float(arr_gen.stats["fallback_rate"]))
            mc.inc("time/arr_refresh_s", time.time() - t_arr0)

        # ---- Collect rollout ----
        t0 = time.time()
        # Per-rollout cache of env arrangement snapshots (for the on_termination
        # callback). Grabbed once here and re-sliced as envs terminate.
        env_arr_snapshot = None
        if arr_enabled:
            # (N, 4, 30) int64 vocab-idx
            env_arr_snapshot = env.snapshot_env_arrangements()

        def _arr_on_termination(*, fired_t, rewards_t, acting_t, rollout_world, **_):
            """Credit terminal reward to ArrangementBuffer rows.

            BUG-K fix (2026-05-11): the previous implementation credited ONLY
            the acting seat's lineup, which had two issues in vs-random mode:

              K1. Teammate's lineup is co-responsible for the win/loss but
                  got no signal — ~50% of useful reward signal was discarded.
                  Empirical (audit_arr_buffer_ready, T=4096 N=32): only
                  11.5% of 1024 buffer rows ever became ``ready``.

              K2. When acting_seat ∈ {1, 3} (random opponent), the reward
                  was caused by random walking, not by the lineup quality.
                  Crediting ArrNet for these is noise. Empirical: ~45% of
                  ready rows came from random-seat episodes — i.e. ArrNet
                  was being trained on the strong correlation between
                  "random opponent's lineup" and "random walking outcome",
                  which is a useless feature.

            Fix: in vs-random mode (random_opponent=True), credit BOTH
            policy-controlled seats (0 and 2) on every terminal:
              - Both share the same team's reward (+1 or -1 from the team's
                perspective).
              - Skip enemy seats entirely (their reward is uninformative
                because their walks were random).
            In self-play mode (random_opponent=False), credit all 4 seats
            with team-perspective rewards.
            """
            if arr_buffer is None or env_arr_snapshot is None:
                return
            fired_np = fired_t.detach().cpu().numpy().astype(bool)
            if not fired_np.any():
                return
            rew_np = rewards_t.detach().cpu().numpy().astype(np.float32)
            seats_np = acting_t.detach().cpu().numpy().astype(np.int64)
            N = env_arr_snapshot.shape[0]

            # Compute the team-perspective reward for team 0 (SOUTH+NORTH).
            #   reward_team0[e] = +1 if winner == team 0
            #                   = -1 if winner == team 1
            #                   = 0  on draw (rew_np[e] = 0)
            # acting_team[e] = seats_np[e] & 1
            # acting_team is the winner iff rew_np[e] == +1 (per reward kernel).
            acting_team = seats_np & 1
            # winner == team 0  iff  (acting_team == 0 and rew == +1) or (acting_team == 1 and rew == -1)
            #                  iff  (acting_team == 0) XOR (rew == -1)   when rew != 0
            sign = np.sign(rew_np).astype(np.float32)  # -1, 0, or +1
            # team 0 reward = sign if acting_team == 0 else -sign
            reward_team0 = np.where(acting_team == 0, sign, -sign).astype(np.float32)
            reward_team1 = -reward_team0

            # Decide which seats to credit.
            # vs-random mode: only seats 0 and 2 are policy-controlled and use
            # the ArrNet's lineup with the policy actually playing them. Skip
            # 1 and 3 (random walks).
            # self-play mode: all 4 seats are policy-controlled.
            random_opp = cfg.random_opponent
            if random_opp:
                seats_to_credit = (0, 2)
            else:
                seats_to_credit = (0, 1, 2, 3)

            for seat in seats_to_credit:
                team = seat & 1
                seat_reward = reward_team0 if team == 0 else reward_team1
                seat_lineup = env_arr_snapshot[np.arange(N), seat]  # (N, 30)
                arr_buffer.add_rewards(
                    env_arrangements=torch.from_numpy(seat_lineup),
                    is_newly_terminal=torch.from_numpy(fired_np),
                    rewards=torch.from_numpy(seat_reward),
                )

        def _refresh_arr_snapshot_after_reset(*, fired_t, rollout_world, **_):
            """Refresh cached lineup rows for envs reset inside this rollout."""
            if env_arr_snapshot is None:
                return
            fired_np = fired_t.detach().cpu().numpy().astype(bool)
            if not fired_np.any():
                return
            refreshed = rollout_world.snapshot_env_arrangements()
            env_arr_snapshot[fired_np] = refreshed[fired_np]

        if cfg.env.use_gpu_rollout:
            # Use v2 (zero-CPU hot path) when GPU buffer is available
            _collect = collect_rollout_gpu_v2 if isinstance(rollout, RolloutBufferGPU) else collect_rollout_gpu
            # The behaviour policy must be the learner itself in eval mode.
            # EMA is intentionally kept out of the PPO collection path: its
            # parameter lag and separate BatchNorm buffers otherwise make the
            # stored old log-probs off-policy before the first update starts.
            behavior_policy = trainer.policy
            behavior_policy.eval()
            kwargs = dict(
                rollout_world=env,
                policy=behavior_policy,
                buffer=rollout,
                device=device,
                seed_base=cfg.env.seed + rollout_idx,
                reset_at_start=(rollout_idx == start_rollout),
                act_chunk_size=int(getattr(cfg.ppo, "act_chunk_size", 0) or 0),
            )
            # Pass random_opponent from config to v2 collector. When
            # cfg.random_opponent=False, the collector will use the
            # current behaviour policy for the non-training team (self-play).
            if _collect is collect_rollout_gpu_v2:
                kwargs["random_opponent"] = cfg.random_opponent
                # Plumb the cfg.ppo.torch_compile flag through. Without this
                # the collector ALWAYS torch.compile's policy.act, which on
                # torch 2.1 + DDP + BatchNorm trips dynamo's decomposition
                # path and crashes. Respecting the cfg flag makes
                # ``torch_compile: false`` actually mean "no compile".
                kwargs["use_compile"] = cfg.ppo.torch_compile
                # Match the collect autocast dtype to cfg.ppo.dtype. v33c
                # was hardcoded to fp16 in collect while the trainer used
                # bf16; the fp16 path produced NaN values once self-play
                # made the policy confident (logits/value-head outputs
                # overflowed fp16's ±65504 range). bf16 has fp32 dynamic
                # range so the same forward stays finite.
                kwargs["autocast_dtype"] = cfg.ppo.get_dtype()
                # F-6: reward shaping toggle (cfg.reward_shaping).
                kwargs["reward_shaping"] = cfg.reward_shaping
            # Pass on_termination to the v2 collector when arr training is on.
            # If belief training is also on, compose both callbacks.
            callbacks = []
            needs_arr_snapshot_refresh = False
            if arr_enabled and _collect is collect_rollout_gpu_v2:
                callbacks.append(_arr_on_termination)
                needs_arr_snapshot_refresh = True
            if belief_enabled and _collect is collect_rollout_gpu_v2:
                # Reveal tracker needs the same env_arr_snapshot the arr
                # callback reads. `env_arr_snapshot` is built just above
                # inside the ``if arr_enabled:`` branch; if arr is off we
                # build it here on demand.
                if env_arr_snapshot is None:
                    env_arr_snapshot = env.snapshot_env_arrangements()
                callbacks.append(
                    reveal_tracker.make_callback(env_arr_snapshot),
                )
                needs_arr_snapshot_refresh = True
            if callbacks:
                def _combined_on_termination(**kw):
                    for cb in callbacks:
                        cb(**kw)
                kwargs["on_termination"] = _combined_on_termination
            if needs_arr_snapshot_refresh:
                kwargs["on_reset"] = _refresh_arr_snapshot_after_reset
            _collect(**kwargs)
        else:
            behavior_policy = trainer.policy
            behavior_policy.eval()
            collect_rollout(
                env=env,
                policy=behavior_policy,
                rollout=rollout,
                device=device,
                seed_base=cfg.env.seed + rollout_idx,
            )
        t_collect = time.time() - t0

        # ---- PPO update ----
        t0 = time.time()
        metrics = trainer.train_epoch(rollout, rng=rng)
        t_train = time.time() - t0

        # ---- ArrangementNet PPO update ----
        # Under DDP: must call into arr_trainer.train_epoch on EVERY rank so
        # the collective all_reduce-MIN inside it sees a contribution from
        # each rank. The trainer itself returns {} early if any rank has 0
        # ready rows. Per-rank gating here would cause "rank A in arr_train,
        # rank B in belief_train" deadlocks (which we observed in v33-smoke
        # at rollout 0 — different ranks terminate different game counts).
        if arr_enabled:
            t_arr_train0 = time.time()
            # Ataraxos-aligned α schedule:
            #   α(t) = max(reg_temp_floor, reg_temp_init / (iter+1)^reg_temp_decay)
            # (paper Table 20: α = 0.1 / t^0.3). iter=rollout_idx+1 so iter=1 at
            # first rollout gives α=reg_temp_init.
            arr_iter = float(rollout_idx + 1)
            scheduled_reg_temp = max(
                cfg.arr.reg_temp_floor,
                cfg.arr.reg_temp_init / (arr_iter ** cfg.arr.reg_temp_decay),
            )
            mc.update({"arr_train/reg_temp": scheduled_reg_temp})
            # BUG-L (2026-05-11): process_data() returns a stats dict including
            # ``arr_buf/n_ready`` and several quantile diagnostics; the caller
            # used to discard it, so the per-rollout log line always printed
            # ``arr_ready=0`` even when the buffer had hundreds of ready rows.
            # Mirror the train_epoch() pattern: capture and mc.update.
            #
            # ``process_data`` is per-rank work over local buffer; safe
            # outside the DDP collective even when local ready_flags is
            # empty (no-op returns {}). DDP: every rank computes its own
            # stats — mc.update averages across ranks naturally.
            if bool(arr_buffer.ready_flags.any()):
                arr_proc_stats = arr_buffer.process_data(
                    td_lambda=1.0, gae_lambda=1.0,
                    reg_temp=scheduled_reg_temp, reg_norm=cfg.arr.reg_norm,
                )
                if arr_proc_stats:
                    mc.update(arr_proc_stats)

            # train_epoch internally syncs across ranks; it's safe (and
            # necessary) to call on every rank even if local buffer is empty.
            if cfg.disable_arr_train:
                arr_metrics = {}
            else:
                arr_metrics = arr_trainer.train_epoch(arr_buffer)
            if arr_metrics:
                mc.update(arr_metrics)
                # EMA update only after a real gradient step happened.
                arr_ema.update(arr_net)
            mc.inc("time/arr_train_s", time.time() - t_arr_train0)

        # ---- BeliefNet (P1) train + refresh ----
        if belief_enabled:
            t_belief0 = time.time()
            if cfg.disable_belief_train:
                belief_metrics = {}
            else:
                belief_metrics = belief_trainer.train_epoch(belief_buffer)
            if belief_metrics:
                mc.update(belief_metrics)
            # Expose the RevealTracker insertion counters too.
            mc.update(reveal_tracker.stats())
            mc.update(belief_buffer.stats())
            mc.inc("time/belief_train_s", time.time() - t_belief0)

            # Refresh the device-resident belief tensor from the EMA net
            # once past warmup and on the configured cadence.
            if (
                not cfg.disable_belief_train
                and rollout_idx >= cfg.belief.warmup_rollouts
                and (rollout_idx - start_rollout) % cfg.belief.refresh_every == 0
                and len(belief_buffer) > 0
            ):
                t_refresh0 = time.time()
                from junqi_rl.belief.inference import refresh_beliefs_neural
                refresh_metrics = refresh_beliefs_neural(
                    rollout=env,
                    belief_net=belief_trainer.ema.model,
                    chunk_size=cfg.belief.infer_chunk_size,
                )
                mc.update(refresh_metrics)
                mc.inc("time/belief_refresh_s", time.time() - t_refresh0)

        # ---- Accumulate metrics ----
        mc.update(metrics)
        mc.inc("time/collect_s", t_collect)
        mc.inc("time/train_s", t_train)
        mc.inc("count/rollout", rollout_idx)
        mc.inc("count/env_steps",
               float(cfg.env.num_envs * cfg.env.steps_per_env * (rollout_idx + 1)))
        mc.inc("count/train_steps", float(trainer.num_train_step))

        # ---- Periodic logging ----
        if (rollout_idx + 1) % cfg.log_every == 0:
            summary = mc.summary()
            summary["time/elapsed_s"] = logger.elapsed()
            logger.log(summary, step=rollout_idx)

            if is_rank0:
                # Console summary (rank 0 only — non-rank-0 logs go to
                # train_rank{N}.log without the formatted summary, which is
                # fine because their metrics are essentially identical to
                # rank 0's after gradient all-reduce).
                policy_loss = summary.get("train/policy_loss", float("nan"))
                value_loss = summary.get("train/value_loss", float("nan"))
                mean_ret = summary.get("rollout/mean_return", float("nan"))
                lr = summary.get("train/lr", float("nan"))
                elapsed = summary.get("time/elapsed_s", 0.0)
                kl_loss = summary.get("train/kl_loss", float("nan"))
                approx_kl = summary.get("train/approx_kl", float("nan"))
                kl_log_ratio_max = summary.get(
                    "train/kl_log_ratio_abs_max", float("nan")
                )
                alpha = summary.get("train/temperature", float("nan"))
                entropy = summary.get("train/entropy", float("nan"))
                collect_entropy = summary.get("collect/entropy", float("nan"))
                magnet_kl = summary.get("train/entropy_loss", float("nan"))
                n_upd = summary.get("train/num_updates", float("nan"))
                nan_skips = summary.get("train/nan_skip_total", 0.0)
                grad_skips = summary.get("train/grad_skip_total", 0.0)
                policy_kept = summary.get("rollout/n_policy_kept", 0.0)
                kept_mean = summary.get("rollout/kept_mean", float("nan"))
                kept_min = summary.get("rollout/kept_min", float("nan"))
                # In DDP mode this prints PER-RANK fps; cluster-wide
                # throughput is approximately (fps × world_size).
                fps = (cfg.env.num_envs * cfg.env.steps_per_env * cfg.log_every
                       / max(1e-6, time.time() - t_rollout_start))
                arr_batches = summary.get("arr_train/n_batches", 0.0)
                arr_suffix = ""
                if arr_batches > 0:
                    arr_suffix = (
                        f"  arr_loss={summary.get('arr_train/total_loss', float('nan')):.4f}"
                        f"  arr_ready={summary.get('arr_buf/n_ready', 0.0):.0f}"
                        f"  arr_batches={arr_batches:.0f}"
                        f"  arr_gn={summary.get('arr_train/g_norm', float('nan')):.3f}"
                    )
                elif "arr/pool_size" in summary:
                    arr_suffix = (
                        f"  arr_pool={summary.get('arr/pool_size', 0.0):.0f}"
                        f"  arr_uniq={summary.get('arr/n_unique_pool', 0.0):.0f}"
                        f"  arr_retry={summary.get('arr/retry_rate', 0.0):.3f}"
                        f"  arr_fb={summary.get('arr/fallback_rate', 0.0):.3f}"
                    )
                print(
                    f"[{rollout_idx:6d}] "
                    f"loss_p={policy_loss:+.4f}  loss_v={value_loss:.4f}  "
                    f"ret={mean_ret:+.4f}  lr={lr:.2e}  "
                    f"fps={fps:.0f}  elapsed={elapsed:.0f}s  "
                    f"kl_loss={kl_loss:+.4f}  approx_kl={approx_kl:+.4f}  "
                    f"kl_max={kl_log_ratio_max:.3f}  "
                    f"alpha={alpha:.4f}  H={entropy:.3f}  "
                    f"Hc={collect_entropy:.3f}  "
                    f"mkl={magnet_kl:+.4f}  n_upd={n_upd:.0f}  "
                    f"kept={policy_kept:.0f}  kept_mean={kept_mean:.0f}  "
                    f"kept_min={kept_min:.0f}  "
                    f"nan_skip={nan_skips:.0f}  grad_skip={grad_skips:.0f}"
                    f"{arr_suffix}"
                )
            t_rollout_start = time.time()
            mc.reset()

        # ---- Periodic checkpoint (rank-0 only; others wait at barrier) ----
        if (rollout_idx + 1) % cfg.save_every == 0:
            if is_rank0:
                ckpt_path = save_checkpoint(
                    trainer, cfg, rollout_idx + 1, cfg.save_dir,
                    arr_trainer=arr_trainer,
                    arr_ema=arr_ema,
                    belief_trainer=belief_trainer,
                )
                print(f"[train] Checkpoint saved: {ckpt_path}")
                assert league_pool is not None
                league_pool.register(
                    ckpt_path,
                    rollout=rollout_idx + 1,
                    tags={"kind": "periodic"},
                )
                # Log config text once on first save
                if rollout_idx + 1 == cfg.save_every:
                    with open(cfg_path) as f:
                        logger.log_text("config/yaml", f.read(), step=rollout_idx)
            if is_distributed:
                dist.barrier()

        # ---- Periodic evaluation (rank-0 only; others wait) ----
        # Evaluate the actual learner policy used for collection. EMA remains
        # checkpoint metadata rather than a second, lagging definition of what
        # the main win-rate means.
        if (rollout_idx + 1) % cfg.eval_every == 0:
            eval_should_stop = torch.zeros(1, dtype=torch.long,
                                           device=device if device.type == "cuda" else "cpu")
            if is_rank0:
                print(f"[train] Evaluating (rollout {rollout_idx + 1})…")
                pool_size = 0
                try:
                    eval_seed = cfg.env.seed + rollout_idx + 1_000_000
                    if (
                        cfg.eval_fixed_setup_pool
                        and cfg.env.use_gpu_rollout
                        and hasattr(env, "upload_fixed_evaluation_setup_pool")
                    ):
                        pool_size = env.upload_fixed_evaluation_setup_pool(
                            seed=cfg.eval_setup_seed,
                        )
                    else:
                        pool_size = 0
                    eval_policy = trainer.policy
                    eval_policy.eval()
                    eval_metrics = evaluate_paired_vs_random(
                        eval_policy,
                        num_games=cfg.eval_num_games,
                        num_envs=cfg.env.num_envs,
                        use_gpu=cfg.env.use_gpu_rollout,
                        device=device,
                        seed=eval_seed,
                        max_moves=cfg.env.max_num_moves,
                        autocast_dtype=cfg.ppo.get_dtype(),
                        greedy=True,
                    )
                    if pool_size:
                        eval_metrics["eval/fixed_setup_pool_size"] = float(pool_size)
                        eval_metrics["eval/fixed_setup_seed"] = float(cfg.eval_setup_seed)
                    logger.log(eval_metrics, step=rollout_idx)
                    win_rate = eval_metrics.get("eval/win_rate", 0.0)
                    loss_rate = eval_metrics.get("eval/loss_rate", 0.0)
                    draw_rate = eval_metrics.get("eval/draw_rate", 0.0)
                    ongoing_rate = eval_metrics.get("eval/ongoing_rate", 0.0)
                    avg_len = eval_metrics.get("eval/avg_game_len", 0.0)
                    done_games = int(eval_metrics.get("eval/num_games", 0.0))
                    requested_games = int(eval_metrics.get("eval/requested_games", 0.0))
                    print(f"[eval]  win={win_rate:.3f}  loss={loss_rate:.3f}  "
                          f"draw={draw_rate:.3f}  ongoing={ongoing_rate:.3f}  "
                          f"avg_len={avg_len:.0f}  done={done_games}/{requested_games}  "
                          f"ci95=[{eval_metrics.get('eval/win_rate_ci95_low', 0.0):.3f}, "
                          f"{eval_metrics.get('eval/win_rate_ci95_high', 1.0):.3f}]")

                    if cfg.eval_baseline_ckpt:
                        baseline_games = cfg.eval_baseline_games or cfg.eval_num_games
                        h2h = evaluate_paired_head_to_head(
                            eval_policy,
                            _eval_baseline_policy,
                            num_games=baseline_games,
                            num_envs=cfg.env.num_envs,
                            device=device,
                            seed=eval_seed + 17,
                            max_moves=cfg.env.max_num_moves,
                            autocast_dtype=cfg.ppo.get_dtype(),
                            greedy=False,
                        )
                        logger.log(h2h, step=rollout_idx)
                        print(
                            f"[h2h]   vs_baseline  win={h2h.get('h2h/win_rate', 0.0):.3f}  "
                            f"loss={h2h.get('h2h/loss_rate', 0.0):.3f}  "
                            f"draw={h2h.get('h2h/draw_rate', 0.0):.3f}  "
                            f"avg_len={h2h.get('h2h/avg_game_len', 0.0):.0f}  "
                            f"done={int(h2h.get('h2h/num_games', 0.0))}/"
                            f"{int(h2h.get('h2h/requested_games', 0.0))}  "
                            f"ci95=[{h2h.get('h2h/win_rate_ci95_low', 0.0):.3f}, "
                            f"{h2h.get('h2h/win_rate_ci95_high', 1.0):.3f}]"
                        )

                    if cfg.eval_record_games > 0:
                        # A replay is useful evidence, but it must not prevent
                        # league evaluation, best-checkpoint selection, or
                        # early-stop decisions. Keep this optional artifact
                        # path isolated from the main evaluation transaction.
                        try:
                            from junqi_rl.analysis.record import record_game_with_policy

                            replay_dir = os.path.join(cfg.save_dir, "replays")
                            os.makedirs(replay_dir, exist_ok=True)
                            for replay_idx in range(cfg.eval_record_games):
                                policy_team = replay_idx & 1
                                replay_seed = eval_seed + replay_idx
                                trajectory = record_game_with_policy(
                                    eval_policy,
                                    rng_seed=replay_seed,
                                    device=device,
                                    max_steps=cfg.env.max_num_moves,
                                    greedy=True,
                                    random_opponent=True,
                                    policy_team=policy_team,
                                    record_beliefs=cfg.eval_record_beliefs,
                                    meta={
                                        "rollout": rollout_idx + 1,
                                        "policy_team": policy_team,
                                        "checkpoint_kind": "policy",
                                        **eval_metrics,
                                    },
                                )
                                replay_path = os.path.join(
                                    replay_dir,
                                    f"eval_{rollout_idx + 1:06d}_"
                                    f"{replay_idx:02d}_team{policy_team}.npz",
                                )
                                trajectory.save(replay_path)
                                print(f"[eval] Replay saved: {replay_path}")
                        except Exception:
                            print("[eval] Replay recording failed; continuing evaluation:")
                            traceback.print_exc()

                    if cfg.league_eval_games > 0:
                        from junqi_rl.analysis.evaluate import eval_head_to_head

                        assert league_pool is not None
                        current_path = save_checkpoint(
                            trainer,
                            cfg,
                            rollout_idx + 1,
                            cfg.save_dir,
                            arr_trainer=arr_trainer,
                            arr_ema=arr_ema,
                            belief_trainer=belief_trainer,
                        )
                        current_entry = league_pool.register(
                            current_path,
                            rollout=rollout_idx + 1,
                            tags={"kind": "evaluation", "win_rate": win_rate},
                        )
                        opponent_entry = league_pool.sample(
                            seed=eval_seed,
                            exclude_rollout=rollout_idx + 1,
                        )
                        if opponent_entry is not None:
                            opponent = JunqiNet(cfg.net).to(device)
                            opponent_state = torch.load(
                                opponent_entry.checkpoint,
                                map_location=device,
                                weights_only=False,
                            )
                            # League comparisons use the same raw learner
                            # definition as the primary score and collection.
                            # Keep EMA only as a backwards-compatible fallback
                            # for a checkpoint that lacks raw policy weights.
                            weights = opponent_state.get("policy")
                            if weights is None:
                                ema_state = opponent_state.get("ema", {})
                                weights = (
                                    ema_state.get("shadow")
                                    if isinstance(ema_state, dict)
                                    else None
                                )
                            if weights is None:
                                raise KeyError(
                                    "league checkpoint has neither policy nor EMA weights"
                                )
                            opponent.load_state_dict(weights)
                            opponent.eval()

                            league_team0_games = max(1, cfg.league_eval_games // 2)
                            league_team1_games = (
                                cfg.league_eval_games - league_team0_games
                            )
                            league_shards = [
                                eval_head_to_head(
                                    eval_policy,
                                    opponent,
                                    num_games=league_team0_games,
                                    first_team=0,
                                    max_steps=cfg.env.max_num_moves,
                                    device=device,
                                    seed_base=eval_seed,
                                    greedy=True,
                                )
                            ]
                            if league_team1_games > 0:
                                league_shards.append(
                                    eval_head_to_head(
                                        eval_policy,
                                        opponent,
                                        num_games=league_team1_games,
                                        first_team=1,
                                        max_steps=cfg.env.max_num_moves,
                                        device=device,
                                        seed_base=eval_seed,
                                        greedy=True,
                                    )
                                )
                            league_metrics = merge_evaluations(
                                *league_shards,
                                prefix="league",
                            )
                            league_metrics["league/opponent_rollout"] = float(
                                opponent_entry.rollout
                            )
                            logger.log(league_metrics, step=rollout_idx)
                            league_pool.record_match(
                                current_entry.sha256,
                                opponent_entry.sha256,
                                first_score=league_metrics[
                                    "league/score_completed"
                                ],
                                games=int(
                                    league_metrics["league/num_games"]
                                ),
                            )
                            print(
                                "[league] "
                                f"vs rollout {opponent_entry.rollout}: "
                                f"score={league_metrics['league/score_completed']:.3f} "
                                f"win={league_metrics['league/win_rate']:.3f} "
                                f"games={int(league_metrics['league/num_games'])}"
                            )
                            del opponent

                    if (
                        requested_games > 0
                        and done_games == requested_games
                        and ongoing_rate == 0.0
                        and win_rate > _best_eval_win_rate
                    ):
                        _best_eval_win_rate = win_rate
                        best_path = save_checkpoint(
                            trainer, cfg, rollout_idx + 1, cfg.save_dir,
                            arr_trainer=arr_trainer,
                            arr_ema=arr_ema,
                            belief_trainer=belief_trainer,
                        )
                        assert league_pool is not None
                        league_pool.register(
                            best_path,
                            rollout=rollout_idx + 1,
                            tags={"kind": "best", "win_rate": win_rate},
                        )
                        best_link = os.path.join(cfg.save_dir, "ckpt_best.pt")
                        update_checkpoint_alias(
                            best_path,
                            best_link,
                            copy_fallback=True,
                        )
                        print(f"[train] New best eval win={win_rate:.3f}; saved {best_link}")

                    # Early-stop logic stays on rank 0 only; the verdict is
                    # broadcast to all ranks below so every rank exits in sync.
                    if (
                        cfg.early_stop_win_rate > 0.0
                        and rollout_idx + 1 >= cfg.early_stop_min_rollout
                    ):
                        if win_rate < cfg.early_stop_win_rate:
                            _early_stop_low_streak += 1
                            print(f"[train] early-stop streak: "
                                  f"{_early_stop_low_streak}/{cfg.early_stop_patience} "
                                  f"(win_rate={win_rate:.3f} < "
                                  f"{cfg.early_stop_win_rate:.3f})")
                            if _early_stop_low_streak >= cfg.early_stop_patience:
                                print(f"[train] EARLY STOP: win_rate below "
                                      f"{cfg.early_stop_win_rate:.3f} for "
                                      f"{cfg.early_stop_patience} consecutive evals. "
                                      f"Saving final ckpt and exiting.")
                                save_checkpoint(
                                    trainer, cfg, rollout_idx + 1, cfg.save_dir,
                                    arr_trainer=arr_trainer,
                                    arr_ema=arr_ema,
                                    belief_trainer=belief_trainer,
                                )
                                eval_should_stop.fill_(1)
                        else:
                            _early_stop_low_streak = 0
                except Exception:
                    print("[train] Evaluation failed:")
                    traceback.print_exc()
                finally:
                    if pool_size and hasattr(env, "restore_training_setup_pool"):
                        env.restore_training_setup_pool()
            # Broadcast the should-stop flag to every rank so they all exit
            # the loop together. Without this, only rank 0 would see the
            # verdict and the others would deadlock at the next all-reduce.
            if is_distributed:
                dist.broadcast(eval_should_stop, src=0)
            if int(eval_should_stop.item()) == 1:
                _stop_requested[0] = True

        # ---- Stop if requested ----
        if _stop_requested[0]:
            if is_rank0:
                print("[train] Graceful stop: saving final checkpoint.")
                save_checkpoint(
                    trainer, cfg, rollout_idx + 1, cfg.save_dir,
                    arr_trainer=arr_trainer,
                    arr_ema=arr_ema,
                    belief_trainer=belief_trainer,
                )
            if is_distributed:
                dist.barrier()
            break

    # ---- Final checkpoint ----
    if not _stop_requested[0]:
        if is_rank0:
            final_path = save_checkpoint(
                trainer, cfg, cfg.total_rollouts, cfg.save_dir,
                arr_trainer=arr_trainer,
                arr_ema=arr_ema,
                belief_trainer=belief_trainer,
            )
            print(f"[train] Training complete. Final checkpoint: {final_path}")
        if is_distributed:
            dist.barrier()

    logger.close()
    if is_rank0:
        print(f"[train] Done. Total elapsed: {logger.elapsed():.1f}s")
    _cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args)
    if args.validate_only:
        print(yaml.safe_dump(_dataclass_to_dict(cfg), sort_keys=False))
        return
    train(cfg)


if __name__ == "__main__":
    main()
