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
    dataclass defaults → YAML file → CLI flags
"""

from __future__ import annotations

import argparse
import dataclasses
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
import yaml
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.belief_net import BeliefNetConfig
    from junqi_rl.training.belief_ppo import BeliefPPOConfig


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


def _is_rank0() -> bool:
    return _ddp_world()[1] == 0

from junqi_rl.env import VectorJunqiEnv
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.networks.arrangement_net import (
    ArrangementNet, ArrangementNetConfig,
)
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
from junqi_rl.training.logger import MultiCounter, TrainingLogger


# ---------------------------------------------------------------------------
# Top-level training configuration
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Environment parameters."""

    num_envs: int = 32
    """Number of parallel environments."""

    steps_per_env: int = 128
    """Steps collected per environment per rollout."""

    max_num_moves: int = 4000
    """Maximum moves per episode before forced draw."""

    seed: int = 42
    """Base seed for environments."""

    use_gpu_rollout: bool = False
    """If True, use junqi_rl.training.collect_rollout_gpu with a
    :class:`GpuRollout` (all game logic on-device) instead of the CPU
    :class:`VectorJunqiEnv`.  Requires the junqi_cuda extension."""


@dataclass
class ArrangementTrainConfig:
    """ArrangementNet (P0) training parameters.

    When ``enabled=False`` (default), training runs the legacy fixed pool —
    bit-for-bit identical to pre-P0 behaviour. When True AND
    ``env.use_gpu_rollout=True`` we jointly train the arrangement network
    alongside the move network:

      * Every ``refresh_every`` rollouts, generate ``n_arr`` fresh lineups
        and re-upload the CUDA setup pool so subsequent auto-resets pick
        from the current policy's distribution.
      * Terminal rewards from ``collect_rollout_gpu_v2`` are credited to
        the buffer via an ``on_termination`` callback.
      * After the main PPO update, run ``arr_trainer.train_epoch`` to update
        the arrangement net.
    """

    enabled: bool = False
    """Enable joint arrangement-net training."""

    refresh_every: int = 1
    """Refresh the setup pool from a new sample every N rollouts."""

    n_arr: int = 1024
    """Number of arrangements to sample per refresh. Must be a multiple of 4
    (one arrangement per seat per pool entry → pool_size = n_arr/4)."""

    storage_duration: int = 4
    """Buffer TTL in **rollouts** (NOT env-steps; the buffer's filter() uses
    ``current_step=rollout_idx``). A value of 4 keeps the current + last 3
    rollouts' arrangements, which is ~the longest possible delay between
    creation of an arrangement and the terminal reward arriving from a game
    that started from it (JunQi max_num_moves=4000 ≈ 0.06 rollouts when
    num_envs=128, steps_per_env=512). Values much larger than this cause
    memory blow-up without improving learning (the reward credit is
    already settled).

    Legacy default in v14-v17 was 2048, which effectively disabled the
    filter over any realistic training horizon and caused the v17 OOM
    crash at rollout 374. Do not raise this above ~20 without a clear
    reason."""

    # Network config
    net: ArrangementNetConfig = field(default_factory=ArrangementNetConfig)

    # PPO config
    ppo: ArrangementPPOConfig = field(default_factory=ArrangementPPOConfig)

    # EMA
    ema_decay: float = 0.999
    """EMA decay for the arrangement net (used for generation, not training)."""

    # ---- Ataraxos-aligned regularization schedule ----
    reg_temp_init: float = 0.1
    """Initial regularization temperature α for setup entropy maximisation.
    Ataraxos (Appendix D.3, Table 20) uses 0.1 at training-iter=1.
    (Legacy v16 used 0.02 and never annealed.)"""

    reg_temp_decay: float = 0.3
    """Decay exponent for the power schedule α(t) = reg_temp_init / (t+1)^decay.
    Ataraxos uses 0.3."""

    reg_temp_floor: float = 0.0
    """Minimum α. Ataraxos does not clamp — we default to 0 (pure MLE in the limit)."""

    reg_norm: float = 10.0
    """Normalizing constant for conditional entropy prediction (paper: 1/10)."""


@dataclass
class BeliefTrainConfig:
    """Neural belief-network (P1) training parameters.

    When ``enabled=False`` (default), the device-resident belief buffer is
    driven only by the hand-coded deductive rules in ``belief.cu``. When
    True AND ``env.use_gpu_rollout=True``:

      * A BeliefBuffer accumulates (obs, seat, label, enemy_mask) tuples
        emitted by a RevealTracker on every game termination.
      * After the main PPO update, ``belief_trainer.train_epoch`` runs
        a few supervised CE steps on the buffer.
      * Every ``refresh_every`` rollouts, :func:`refresh_beliefs_neural`
        runs the EMA belief net over the current obs and replaces the
        device belief tensor. Subsequent deductive updates start from
        the neural prior rather than the hand-coded prior table.

    See docs/P1_BELIEF_NET_PLAN.md for the full design.
    """

    enabled: bool = False
    """Enable joint belief-net training + inference."""

    refresh_every: int = 1
    """Recompute + upload neural beliefs every N rollouts. 1 = every rollout
    (highest signal, highest cost). Higher values let beliefs go stale but
    save throughput."""

    buffer_capacity: int = 50_000
    """Max CPU replay-buffer size. 50k × 74 KB/sample ≈ 3.7 GB RAM (fp16
    obs storage). At 128 envs × ~10 terminal reveals/rollout we fill in
    ~40 rollouts of training."""

    warmup_rollouts: int = 20
    """Defer refresh_beliefs_neural until this many rollouts have passed,
    so the belief net has seen enough training data to not emit garbage
    that poisons the deductive prior. During warmup the deductive rules
    use the fixed prior table as before."""

    # Network config
    net: "BeliefNetConfig | None" = None
    """Belief net hyperparams. None → use library defaults."""

    # Trainer config
    ppo: "BeliefPPOConfig | None" = None
    """BeliefPPO (really CE) trainer hyperparams. None → use library defaults."""

    # EMA
    ema_decay: float = 0.999
    """EMA decay for the belief net (inference uses the EMA copy)."""

    # Inference memory control
    infer_chunk_size: int = 128
    """Per-forward batch size for BeliefNet inference (``refresh_beliefs_neural``).
    Peak FFN activation scales linearly with this value; default 128 leaves
    ~4× headroom on T4 with ``num_envs=128``. Set higher on bigger GPUs to
    recover throughput. See :func:`junqi_rl.belief.inference.refresh_beliefs_neural`
    for the memory math — v23 OOMed at rollout 320 with an unchunked 512-row
    forward, v24+ uses chunk_size=128."""


@dataclass
class TrainConfig:
    """Master configuration: combines env, network, PPO, and run parameters."""

    # Sub-configs
    env: EnvConfig = field(default_factory=EnvConfig)
    net: JunqiNetConfig = field(default_factory=JunqiNetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    arr: ArrangementTrainConfig = field(default_factory=ArrangementTrainConfig)
    belief: BeliefTrainConfig = field(default_factory=BeliefTrainConfig)

    # Run logistics
    save_dir: str = "exps/default"
    """Directory for checkpoints, logs, and configs."""

    total_rollouts: int = 10_000
    """Total number of rollout epochs to train for."""

    save_every: int = 100
    """Save a checkpoint every N rollouts."""

    eval_every: int = 50
    """Run evaluation every N rollouts."""

    early_stop_win_rate: float = 0.0
    """If > 0, training stops when ``eval/win_rate`` stays below this value
    for ``early_stop_patience`` consecutive evals *after* rollout
    ``early_stop_min_rollout``. Set 0 to disable (legacy behaviour).

    Useful for failed-run detection: e.g. ``early_stop_win_rate=0.4`` +
    ``early_stop_patience=3`` + ``early_stop_min_rollout=100`` aborts a
    run whose policy has collapsed post-warmup, saving ~1-2h of wasted
    compute. The final ckpt is still saved so you can diagnose."""

    early_stop_patience: int = 3
    """How many *consecutive* sub-threshold evals trigger early stop.
    Only consulted when ``early_stop_win_rate > 0``."""

    early_stop_min_rollout: int = 100
    """Earliest rollout at which early-stop can fire. Should be >=
    ``belief.warmup_rollouts`` so the belief net has had a chance to
    influence the policy before we decide."""

    resume: str = ""
    """Path to checkpoint to resume from (empty = start fresh)."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Training device."""

    # Logging
    use_wandb: bool = False
    """Enable Weights & Biases logging."""

    wandb_project: str = "junqi-rl"
    """W&B project name."""

    wandb_run_name: str = ""
    """W&B run name (empty = auto-generated)."""

    log_every: int = 10
    """Log metrics every N rollouts."""

    # Misc
    seed: int = 42
    """Global random seed."""

    torch_deterministic: bool = False
    """If True, force deterministic CuDNN ops (slower)."""

    random_opponent: bool = True
    """If True (default), the non-training team plays uniform-random during
    rollout collection. If False, BOTH teams use the current EMA policy
    (self-play). Self-play removes the 'defensive stall' exploit where the
    policy learns to avoid combat against a random opponent, because a
    self-play opponent also won't allow arbitrarily long games. Only
    respected by ``collect_rollout_gpu_v2``."""

    train_value_on_random_seats: bool = False
    """In vs-random mode (random_opponent=True), control whether the value
    head is trained on enemy-seat (random) transitions.

    Default ``False`` (legacy v17–v35 behaviour, validated by the
    100-rollout A/B in 2026-05-11 to be empirically better than
    ``True``).

    History
    -------
    The 2026-05-10 F-4 patch initially defaulted this to ``True`` based
    on a theoretical argument: the legacy path zeroed seat 1/3 returns
    (so value_loss=0 there), and we hypothesised that the GAE bootstrap
    term ``γ·flip·V(s_{t+1})`` was reading a noisy / untrained V-head
    on those states, degenerating the advantage signal.

    Empirical 100-rollout A/B (single H20, num_envs=192, T=512,
    100 rollouts; ``v35_fixed_100R`` vs ``v35_legacy_100R``) showed:

      * legacy (False)  : last-5-eval mean win=0.636, |ret| mean=0.0054
      * F-4 ON (True)   : last-5-eval mean win=0.581, |ret| mean=0.0010

    F-4 ON was *worse* on every measured dimension. Likely cause: in
    vs-random the seat-1/3 GAE return is "expected return after a
    uniform-random opponent move", which is a different distribution
    from "expected return after our policy moves". Forcing a single
    V-head to fit both distributions costs capacity and hurts the
    seat-0/2 V estimate that GAE bootstrap actually uses for policy
    transitions. The legacy "zero out enemy returns" acts as an implicit
    capacity-allocation regulariser.

    Set ``True`` only for diagnostic A/Bs; the production default is
    ``False``. The F-4 plumbing (RolloutBatch.value_only_mask, PPO
    weight_per arg) is retained so the experiment is reproducible
    without code revert.
    """

    reward_shaping: bool = True
    """If True (legacy v17–v35 default), the GPU collector adds piece-loss
    penalties to non-terminal rewards: -0.05 when the acting piece died into
    a stronger defender (event=4), -0.02 on mutual BOMB (event=3).

    Set False to use *pure terminal* rewards (±1 / 0) — recommended when
    debugging F-4 (the value-head training fix) since shaping interacts
    nonlinearly with bootstrap noise. See
    ``junqi_rl/training/gpu_collector.py:786-791`` for the shaping logic
    and the long comment at line 768-781 documenting why iteration 1's EAT
    bonus broke training."""

    fixed_setup_styles: tuple[str, ...] | list[str] | None = None
    """If set (e.g. ``["T", "D"]``), the GPU rollout pool is built from
    fixed canonical lineups in ``junqi_core.setup_canonical`` instead of
    uniform-random.  All 4 seats share the SAME lineup per game (Plan A).

    .. warning::
       Plan A leaks information under DARK rule: piece_id channels
       deterministically encode each piece's true type when the layout
       is the same every game.  Use ``mixed_setup`` (Plan B) instead for
       genuine 暗棋 training.
    """

    mixed_setup: bool = False
    """Plan B (Recommended): own team (SOUTH+NORTH) uses canonical lineup,
    enemy team (WEST+EAST) draws fresh uniform-random lineups every game,
    independently per seat.

    Why this is the correct setup for vs-random training:

    * Evaluation runs against random opponents → training distribution
      should match.
    * Independent enemy lineups force the policy to handle 2 distinct
      enemy formations per game, exposing the model to the full prior
      space of opponents.
    * Own team's lineup is stable so the move-policy network can
      specialise on a familiar self-side prior; this is the curriculum
      benefit Plan A also gave, without the 4-seat-mirror information
      leak.

    When ``mixed_setup=True``, ``fixed_setup_styles`` is ignored.
    See ``mixed_own_team_styles`` for picking which canonical lineup the
    own team uses.
    """

    mixed_own_team_styles: tuple[str, ...] | list[str] = ("T",)
    """Canonical-lineup pool for the OWN team in mixed_setup mode.
    Default ``("T",)`` (single style — fastest to converge).  Pass a
    larger tuple (e.g. ``("T", "D")``) to add own-side diversity once
    the basic lineup is mastered."""

    disable_arr_train: bool = False
    """If True, the ArrangementNet trainer is constructed but ``train_epoch``
    is skipped each rollout.  Used together with ``fixed_setup_styles`` to
    fix the opponent's setup distribution and let the move-policy network
    learn first.  Cheaper than rebuilding a separate train loop without
    arr at all."""

    disable_belief_train: bool = False
    """If True, the BeliefNet trainer is constructed but ``train_epoch`` /
    ``refresh_beliefs_neural`` are skipped each rollout.  Side-steps the
    DDP NaN-skip inconsistency observed in v40 long runs (rank A skips on
    NaN ce_loss while rank B all-reduces, leading to SIGFPE under NCCL).
    Use with ``fixed_setup_styles`` for curriculum baseline."""


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _nested_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (in place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _nested_update(base[k], v)
        else:
            base[k] = v
    return base


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert nested dataclasses to plain dicts (for YAML serialisation)."""
    if dataclasses.is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_dataclass_to_dict(x) for x in obj)
    else:
        return obj


def _dict_to_dataclass(cls: type, d: Any) -> Any:
    """Recursively instantiate nested dataclasses from a plain dict."""
    if not dataclasses.is_dataclass(cls):
        return d
    if not isinstance(d, dict):
        return d
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for name, f in field_map.items():
        if name not in d:
            continue
        val = d[name]
        ftype = f.type
        # Resolve string annotations
        if isinstance(ftype, str):
            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # type: ignore[arg-type]
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = _dict_to_dataclass(ftype, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(args: argparse.Namespace) -> TrainConfig:
    """Build :class:`TrainConfig` from defaults → YAML → CLI overrides."""
    # Start from defaults
    cfg = TrainConfig()
    cfg_dict = _dataclass_to_dict(cfg)

    # Overlay YAML
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            yaml_dict = yaml.safe_load(f) or {}
        _nested_update(cfg_dict, yaml_dict)
        print(f"[config] Loaded YAML: {args.config}")

    # Overlay CLI flags
    # Flat CLI keys (no "__") default to the root of cfg_dict, but a few
    # common env fields live under cfg.env.* — route those to the right place.
    _FLAT_TO_ENV_KEYS = {
        "num_envs", "steps_per_env", "use_gpu_rollout",
    }
    cli_overrides: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k in ("config", "resume_cli") or v is None:
            continue
        # Map flat CLI keys into nested dict
        if "__" in k:
            # e.g. ppo__clip_range → ppo.clip_range
            parts = k.split("__")
            sub = cli_overrides
            for part in parts[:-1]:
                sub = sub.setdefault(part, {})
            sub[parts[-1]] = v
        elif k in _FLAT_TO_ENV_KEYS:
            env_sub = cli_overrides.setdefault("env", {})
            env_sub[k] = v
        else:
            cli_overrides[k] = v
    _nested_update(cfg_dict, cli_overrides)

    # CLI resume shortcut (--resume maps to cfg.resume)
    if getattr(args, "resume_cli", None):
        cfg_dict["resume"] = args.resume_cli

    cfg = _dict_to_dataclass(TrainConfig, cfg_dict)

    # PPO sub-config must also carry net config
    # (PPOTrainer takes policy separately; PPOConfig.net is advisory)
    cfg.ppo.net = cfg.net

    return cfg


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 四国军棋 AI with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Path to YAML config file")
    parser.add_argument("--resume_cli", dest="resume_cli", default="", help="Checkpoint to resume from")
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
    # Allow arbitrary --ppo__clip_range=0.3 style overrides
    parser.add_argument("--extra", nargs="*", metavar="KEY=VALUE",
                        help="Arbitrary nested overrides, e.g. ppo__clip_range=0.2")
    return parser


# ---------------------------------------------------------------------------
# Evaluation (GPU-accelerated)
# ---------------------------------------------------------------------------


_eval_rollout_cache: dict[int, "GpuRollout"] = {}  # keyed by num_envs


def evaluate_vs_random_gpu(
    policy: JunqiNet,
    num_envs: int = 64,
    num_games: int = 128,
    device: str = "cuda",
    seed: int = 0,
    max_moves: int = 4000,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """GPU-accelerated evaluation: policy team (SOUTH+NORTH) vs random (WEST+EAST).

    Uses GpuRollout for ~100x faster eval than the CPU version.
    Runs until ``num_games`` episodes complete across ``num_envs`` parallel envs,
    or until the safety step cap is hit. Unfinished games are counted as
    ongoing in the denominator, so a single completed game can no longer report
    a misleading 0%/100% win rate.

    The GpuRollout is cached across calls to avoid repeated GPU memory
    allocation/deallocation which causes OOM after many eval rounds.
    """
    from junqi_rl.gpu_rollout import GpuRollout

    policy.eval()
    N = min(num_envs, num_games)
    # Reuse cached rollout to avoid GPU memory fragmentation
    if N not in _eval_rollout_cache:
        _eval_rollout_cache[N] = GpuRollout(num_envs=N)
    rollout = _eval_rollout_cache[N]
    rollout.reset(seed_base=seed)

    wins = 0
    losses = 0
    draws = 0
    total_games = 0
    total_steps = 0

    while total_games < num_games:
        turn_t = rollout.turn_torch()           # (N,) int8 CUDA
        acting_t = turn_t.clone()

        # Build obs + legal mask
        obs_sp, obs_gl = rollout.build_acting_seat_observation_torch(acting_t)
        lm_t = rollout.legal_mask_canonical_torch_device(acting_t)

        # Policy inference for ALL seats
        # Use the original (non-compiled) act to avoid CUDA graph shape mismatch
        # when eval N differs from collect N. Match the training/collect dtype;
        # v33d relies on bf16 to avoid fp16 overflow in confident policies.
        _act_fn = getattr(policy, '_orig_act', policy.act)
        eval_dtype = autocast_dtype or torch.bfloat16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=eval_dtype):
            actions, _, _ = _act_fn(obs_sp, obs_gl, lm_t)
        actions = actions.to(torch.int32)

        # Replace enemy seats (1, 3) with random legal actions
        is_enemy = (acting_t == 1) | (acting_t == 3)
        if is_enemy.any():
            uniform = torch.where(lm_t, 0.0, float('-inf'))
            u = torch.rand_like(uniform.float()).clamp_(1e-10, 1.0)
            gumbel = -torch.log(-torch.log(u))
            random_acts = (uniform + gumbel).argmax(dim=-1).to(torch.int32)
            actions = torch.where(is_enemy, random_acts, actions)

        # Step
        result = rollout.step_device_torch(actions, acting_t)
        term_t = result["terminated"]
        total_steps += N

        # Count terminations
        if term_t.any():
            winner_t = result["winner_team"]   # int8: -1/0/1
            draw_t = result["draw"]

            term_np = term_t.cpu().numpy()
            winner_np = winner_t.cpu().numpy()
            draw_np = draw_t.cpu().numpy()

            for i in range(N):
                if term_np[i] and total_games < num_games:
                    total_games += 1
                    if draw_np[i]:
                        draws += 1
                    elif winner_np[i] == 0:  # team 0 (SOUTH+NORTH) wins
                        wins += 1
                    else:
                        losses += 1

            # Reset terminated envs
            rollout.reset_terminated_device(seed=seed + total_games)

        # Safety: prevent infinite loops
        if total_steps > num_games * max_moves:
            break

    completed = min(total_games, num_games)
    ongoing = max(0, num_games - completed)
    denom = max(1, num_games)
    completed_denom = max(1, completed)
    return {
        # Rates are over the requested evaluation size, not just completed games.
        # This prevents 1/1 finished games from being reported as 100% while
        # the other 127 games are still ongoing/stalled at the safety cap.
        "eval/win_rate": wins / denom,
        "eval/loss_rate": losses / denom,
        "eval/draw_rate": draws / denom,
        "eval/ongoing_rate": ongoing / denom,
        "eval/avg_game_len": total_steps / completed_denom,
        "eval/avg_game_len_all": total_steps / denom,
        "eval/num_games": float(completed),
        "eval/requested_games": float(num_games),
    }


def evaluate_vs_random(
    policy: JunqiNet,
    num_envs: int = 16,
    num_games: int = 32,
    device: str = "cpu",
    seed: int = 0,
    max_moves: int = 4000,
) -> dict[str, float]:
    """Run ``num_games`` episodes: EMA policy (SOUTH team) vs random (NORTH team).

    Returns metrics dict with win/loss/draw rates and average game length.
    """
    from junqi_core.rules import Seat

    policy.eval()

    n = min(num_envs, num_games)
    env = VectorJunqiEnv(num_envs=n, max_num_moves=max_moves)

    wins = 0
    losses = 0
    draws = 0
    total_games = 0
    total_moves = 0

    obs_sp, obs_gl = env.reset(seed_base=seed)

    game_moves = np.zeros(n, dtype=np.int32)

    while total_games < num_games:
        current_seats = env.current_seats()
        acting_idx = np.array([s.value for s in current_seats], dtype=np.int64)

        act_obs_sp = obs_sp[np.arange(n), acting_idx]
        act_obs_gl = obs_gl[np.arange(n), acting_idx]

        # Legal masks
        from junqi_rl.training.collector import _build_legal_mask
        legal_mask_np = _build_legal_mask(env, current_seats)

        # Decide action: EMA policy for SOUTH/NORTH team (seats 0,2),
        # random for WEST/EAST (seats 1,3)
        actions_world = np.zeros(n, dtype=np.int32)
        for i, seat in enumerate(current_seats):
            if env.done[i]:
                continue
            wids = env.envs[i].legal_action_ids(seat)
            if len(wids) == 0:
                continue
            if seat in (Seat.SOUTH, Seat.NORTH):  # policy team
                # Use policy for this env
                sp_t = torch.from_numpy(act_obs_sp[i : i + 1]).to(device)
                gl_t = torch.from_numpy(act_obs_gl[i : i + 1]).to(device)
                lm_t = torch.from_numpy(legal_mask_np[i : i + 1]).to(device)
                with torch.no_grad():
                    acts, _, _ = policy.act(sp_t, gl_t, lm_t)
                from junqi_rl.env import unrotate_compact_action_id
                actions_world[i] = unrotate_compact_action_id(int(acts[0].cpu()), seat)
            else:
                # Random legal action
                actions_world[i] = int(np.random.choice(wids))

        obs_sp, obs_gl, rewards_np, done_np, infos = env.step(actions_world)
        game_moves += 1

        for i, done_i in enumerate(done_np):
            if done_i:
                # rewards_np[i] shape: (4,) — reward per seat
                south_reward = float(rewards_np[i, Seat.SOUTH.value])
                if south_reward > 0:
                    wins += 1
                elif south_reward < 0:
                    losses += 1
                else:
                    draws += 1
                total_games += 1
                total_moves += int(game_moves[i])
                game_moves[i] = 0

                if total_games >= num_games:
                    break

                # Reset this env
                env.envs[i].reset(seed=seed + total_games)
                env._done[i] = False
                env._fill_all_obs()
                obs_sp = env.obs_spatial
                obs_gl = env.obs_global

    env_games = total_games if total_games > 0 else 1
    return {
        "eval/win_rate": wins / env_games,
        "eval/loss_rate": losses / env_games,
        "eval/draw_rate": draws / env_games,
        "eval/ongoing_rate": 0.0,
        "eval/avg_game_len": total_moves / env_games,
        "eval/avg_game_len_all": total_moves / env_games,
        "eval/num_games": float(total_games),
        "eval/requested_games": float(num_games),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    trainer: PPOTrainer,
    cfg: TrainConfig,
    rollout_idx: int,
    save_dir: str,
    *,
    arr_trainer: ArrangementPPOTrainer | None = None,
    arr_ema: EMAPolicy | None = None,
    belief_trainer: Any | None = None,
) -> str:
    """Save trainer checkpoint and return path.

    Besides the main move-policy PPO state, also persists optional auxiliary
    networks whose training state matters for resume and deployment:
    ArrangementNet (+ EMA/optimizer) and BeliefNet (+ EMA/optimizer).
    Large replay buffers are intentionally not stored here.
    """
    ckpt_path = os.path.join(save_dir, f"ckpt_{rollout_idx:06d}.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    state = trainer.state_dict()
    state["train_cfg"] = _dataclass_to_dict(cfg)
    if arr_trainer is not None:
        state["arrangement"] = {
            "trainer": arr_trainer.state_dict(),
            "ema": arr_ema.state_dict() if arr_ema is not None else None,
        }
    if belief_trainer is not None:
        state["belief"] = belief_trainer.state_dict()
    torch.save(state, ckpt_path)
    # Also keep a "latest" symlink for easy resuming
    latest = os.path.join(save_dir, "ckpt_latest.pt")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.abspath(ckpt_path), latest)
    except OSError:
        pass
    return ckpt_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    """Full PPO training loop."""
    # ---- Distributed init (no-op if WORLD_SIZE<=1) ----
    # Done first so cfg.device / cfg.seed / cfg.env.seed are rank-aware before
    # anything else uses them (RNG, logger, model construction, env seeding).
    world_size, global_rank, local_rank = _init_distributed(cfg)
    is_rank0 = global_rank == 0
    is_distributed = world_size > 1

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
            yaml.dump(_dataclass_to_dict(cfg), f, default_flow_style=False)
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
        unit = getattr(cfg.ppo, "lr_schedule_unit", "grad_step")
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
            if getattr(cfg, "fixed_setup_styles", None)
            else None
        )
        mixed_setup = bool(getattr(cfg, "mixed_setup", False))
        mixed_own_team_styles = tuple(
            getattr(cfg, "mixed_own_team_styles", ("T",))
        ) or ("T",)
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
    rollout_random_opponent = bool(getattr(cfg, "random_opponent", True))
    if cfg.env.use_gpu_rollout and device.type == "cuda":
        rollout = RolloutBufferGPU(
            num_envs=cfg.env.num_envs,
            steps_per_env=cfg.env.steps_per_env,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            td_lambda=cfg.ppo.td_lambda,
            adv_filt_thresh=cfg.ppo.adv_filt_thresh,
            adv_filt_rate=cfg.ppo.adv_filt_rate,
            device=device,
            random_opponent=rollout_random_opponent,
            train_value_on_random_seats=getattr(
                cfg, "train_value_on_random_seats", True
            ),
        )
        if is_rank0:
            mode = "vs-random" if rollout_random_opponent else "self-play"
            v_only = getattr(cfg, "train_value_on_random_seats", True)
            v_tag = "value-also-on-random-seats" if v_only else "legacy-no-value-on-random"
            shaping = bool(getattr(cfg, "reward_shaping", True))
            shape_tag = "reward-shaping=ON" if shaping else "reward-shaping=OFF (terminal-only)"
            print(f"[train] RolloutBufferGPU (zero-CPU collect path, mode={mode}, {v_tag}, {shape_tag})")
    else:
        rollout = RolloutBuffer(
            num_envs=cfg.env.num_envs,
            steps_per_env=cfg.env.steps_per_env,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            td_lambda=cfg.ppo.td_lambda,
            adv_filt_thresh=cfg.ppo.adv_filt_thresh,
            adv_filt_rate=cfg.ppo.adv_filt_rate,
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
    _env_arr_cache: list[np.ndarray] = []  # cached snapshot of env arrangements per rollout
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
                arr_gen.samples, arr_gen.seat_idx,
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
            mc.inc("arr/retry_rate", float(arr_gen.stats["retry_rate"]))
            mc.inc("arr/fallback_rate", float(arr_gen.stats["fallback_rate"]))
            mc.inc("time/arr_refresh_s", time.time() - t_arr0)

        # ---- Collect rollout ----
        t0 = time.time()
        # Per-rollout cache of env arrangement snapshots (for the on_termination
        # callback). Grabbed once here and re-sliced as envs terminate.
        env_arr_snapshot = None
        env_seat_snapshot = None
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
            random_opp = bool(getattr(cfg, "random_opponent", True))
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
            kwargs = dict(
                rollout_world=env,
                policy=trainer.ema.model,
                buffer=rollout,
                device=device,
                seed_base=cfg.env.seed + rollout_idx,
                reset_at_start=(rollout_idx == start_rollout),
            )
            # Pass random_opponent from config to v2 collector. When
            # cfg.random_opponent=False, the collector will use the
            # current EMA policy for the non-training team (self-play).
            if _collect is collect_rollout_gpu_v2:
                kwargs['random_opponent'] = getattr(cfg, 'random_opponent', True)
                # Plumb the cfg.ppo.torch_compile flag through. Without this
                # the collector ALWAYS torch.compile's policy.act, which on
                # torch 2.1 + DDP + BatchNorm trips dynamo's decomposition
                # path and crashes. Respecting the cfg flag makes
                # ``torch_compile: false`` actually mean "no compile".
                kwargs['use_compile'] = bool(getattr(cfg.ppo, "torch_compile", True))
                # Match the collect autocast dtype to cfg.ppo.dtype. v33c
                # was hardcoded to fp16 in collect while the trainer used
                # bf16; the fp16 path produced NaN values once self-play
                # made the policy confident (logits/value-head outputs
                # overflowed fp16's ±65504 range). bf16 has fp32 dynamic
                # range so the same forward stays finite.
                kwargs['autocast_dtype'] = cfg.ppo.get_dtype()
                # F-6: reward shaping toggle (cfg.reward_shaping).
                kwargs['reward_shaping'] = bool(getattr(cfg, 'reward_shaping', True))
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
                kwargs['on_termination'] = _combined_on_termination
            if needs_arr_snapshot_refresh:
                kwargs['on_reset'] = _refresh_arr_snapshot_after_reset
            _collect(**kwargs)
        else:
            collect_rollout(
                env=env,
                policy=trainer.ema.model,   # use EMA policy for data collection
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
            if getattr(cfg, "disable_arr_train", False):
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
            if getattr(cfg, "disable_belief_train", False):
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
                not getattr(cfg, "disable_belief_train", False)
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
                        f"  arr_retry={summary.get('arr/retry_rate', 0.0):.3f}"
                        f"  arr_fb={summary.get('arr/fallback_rate', 0.0):.3f}"
                    )
                print(
                    f"[{rollout_idx:6d}] "
                    f"loss_p={policy_loss:+.4f}  loss_v={value_loss:.4f}  "
                    f"ret={mean_ret:+.4f}  lr={lr:.2e}  "
                    f"fps={fps:.0f}  elapsed={elapsed:.0f}s"
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
                # Log config text once on first save
                if rollout_idx + 1 == cfg.save_every:
                    with open(cfg_path) as f:
                        logger.log_text("config/yaml", f.read(), step=rollout_idx)
            if is_distributed:
                dist.barrier()

        # ---- Periodic evaluation (rank-0 only; others wait) ----
        # Eval is single-process by design (uses trainer.ema.model, which is
        # identical on every rank because gradients are all-reduced and EMA
        # is applied identically per rank). Doing it once on rank 0 and
        # broadcasting the early-stop verdict is far cheaper than running
        # 8 redundant eval loops in parallel.
        if (rollout_idx + 1) % cfg.eval_every == 0:
            eval_should_stop = torch.zeros(1, dtype=torch.long,
                                           device=device if device.type == "cuda" else "cpu")
            if is_rank0:
                print(f"[train] Evaluating (rollout {rollout_idx + 1})…")
                try:
                    if cfg.env.use_gpu_rollout:
                        eval_metrics = evaluate_vs_random_gpu(
                            policy=trainer.ema.model,
                            num_envs=64,
                            num_games=128,
                            device=str(device),
                            seed=cfg.env.seed + rollout_idx + 1000000,
                            max_moves=cfg.env.max_num_moves,
                            autocast_dtype=cfg.ppo.get_dtype(),
                        )
                    else:
                        eval_metrics = evaluate_vs_random(
                            policy=trainer.ema.model,
                            num_envs=min(cfg.env.num_envs, 16),
                            num_games=max(32, cfg.env.num_envs // 2),
                            device=str(device),
                            seed=cfg.env.seed + rollout_idx,
                            max_moves=cfg.env.max_num_moves,
                        )
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
                          f"avg_len={avg_len:.0f}  done={done_games}/{requested_games}")

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
                        best_link = os.path.join(cfg.save_dir, "ckpt_best.pt")
                        if os.path.islink(best_link) or os.path.exists(best_link):
                            os.remove(best_link)
                        try:
                            os.symlink(os.path.abspath(best_path), best_link)
                        except OSError:
                            import shutil
                            shutil.copy2(best_path, best_link)
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

    # Handle extra KEY=VALUE overrides
    extra_dict: dict[str, Any] = {}
    for item in (args.extra or []):
        if "=" not in item:
            print(f"[config] Ignoring malformed extra override: {item!r}", file=sys.stderr)
            continue
        key, _, val_str = item.partition("=")
        # Attempt type coercion
        try:
            val: Any = yaml.safe_load(val_str)
        except yaml.YAMLError:
            val = val_str
        key_parts = key.split("__")
        sub = extra_dict
        for part in key_parts[:-1]:
            sub = sub.setdefault(part, {})
        sub[key_parts[-1]] = val

    # Inject extra into args namespace so load_config sees them
    for k, v in extra_dict.items():
        setattr(args, k, v)

    cfg = load_config(args)
    train(cfg)


if __name__ == "__main__":
    main()
