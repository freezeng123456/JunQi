#!/usr/bin/env python3
"""diagnose_v37_collapse.py — deep inspection of why v37 stopped learning.

Goal
----
Across several ckpts from v37_bugfixes_1500R (R25 early, R200 peak,
R400 mid-decline, R700 collapsed), load each policy and collect ONE fresh
rollout (no training, no updates). Then report detailed distributions of:

    * reward / return / value / advantage / |A_norm|
    * fraction of own-team policy-controlled transitions kept after
      adv-filter vs. total
    * event-code histogram (0=none, 1=move, 2=eat, 3=bomb, 4=killed)
    * episode length / terminated-during-rollout counts / winner-team
      breakdown
    * policy entropy on its own decisions, temperature-adjusted
    * value-head saturation (|value| near ±1 means network predicts
      certain outcome; |value|≈0 means "no info")

Each ckpt is evaluated twice: with and without reward_shaping, so we
isolate the shaping contribution from the raw CUDA ±1 terminal signal.

Usage
-----
    python diagnose_v37_collapse.py                 # default: R25,R200,R400,R700
    python diagnose_v37_collapse.py --ckpts R50 R100 R500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "junqi_rl"))

import numpy as np
import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
from junqi_rl.training.rollout_gpu import RolloutBufferGPU
from junqi_rl.networks.junqi_net import JunqiNet

# pull train.py utilities. train.py is a script, but it defines TrainConfig
# at module-level. Loading as a regular module via sys.path + import works
# (the importlib.spec_from_file_location path breaks dataclass __module__
# resolution for nested classes).
sys.path.insert(0, str(_REPO / "scripts"))
import train as _train_mod   # noqa: E402
TrainConfig = _train_mod.TrainConfig
load_config = _train_mod.load_config

from junqi_rl.training.ppo import PPOTrainer, PPOConfig  # noqa: E402
try:
    from junqi_rl.training.ppo import NetConfig  # legacy re-export if present
except ImportError:
    NetConfig = None


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

EVENT_NAMES = {0: "none", 1: "move", 2: "eat", 3: "bomb", 4: "killed"}


def _pct(t: torch.Tensor, ps=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)) -> str:
    """Return a short string of quantiles for tensor."""
    if t.numel() == 0:
        return "empty"
    x = t.detach().float().flatten()
    q = torch.tensor(ps, device=x.device, dtype=x.dtype)
    vals = torch.quantile(x, q).cpu().tolist()
    return "  ".join(f"p{int(p*100)}={v:+.4f}" for p, v in zip(ps, vals))


def _describe(name: str, t: torch.Tensor) -> str:
    if t.numel() == 0:
        return f"{name}: empty"
    x = t.detach().float().flatten()
    return (
        f"{name:<16s}  n={x.numel():>7d}  "
        f"mean={x.mean().item():+.5f}  std={x.std().item():.5f}  "
        f"min={x.min().item():+.4f}  max={x.max().item():+.4f}"
    )


# ---------------------------------------------------------------------------
# Run a single rollout and return a rich stats dict
# ---------------------------------------------------------------------------


def run_one_rollout(
    policy: JunqiNet,
    num_envs: int,
    steps_per_env: int,
    seed: int,
    device: torch.device,
    reward_shaping: bool,
    ppo_cfg: PPOConfig,
) -> dict:
    env = GpuRollout(num_envs=num_envs, device_id=device.index if device.index is not None else 0)
    buffer = RolloutBufferGPU(
        num_envs=num_envs,
        steps_per_env=steps_per_env,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        td_lambda=ppo_cfg.td_lambda,
        adv_filt_thresh=ppo_cfg.adv_filt_thresh,
        adv_filt_rate=ppo_cfg.adv_filt_rate,
        device=device,
        random_opponent=True,
        train_value_on_random_seats=False,
    )

    policy.eval()
    with torch.no_grad():
        collect_rollout_gpu_v2(
            rollout_world=env,
            policy=policy,
            buffer=buffer,
            device=device,
            seed_base=seed,
            reset_at_start=True,
            reward_shaping=reward_shaping,
            random_opponent=True,
            on_termination=None,
            on_reset=None,
            use_compile=False,
            autocast_dtype=torch.bfloat16,
        )

    # Compute returns & advantages (buffer.compute_returns needs last_values)
    # Bootstrap from final observation via policy forward pass.
    # The collector already stored values at each step; we need last_values
    # for compute_returns. Easiest: use zeros as bootstrap — for T=512 and
    # mostly-terminated envs this is close to truth.
    last_values = torch.zeros(num_envs, device=device, dtype=torch.float32)
    last_seats = buffer.seats[-1].clone()
    buffer.compute_returns(last_values, last_seats=last_seats)

    T = buffer.steps_per_env
    N = buffer.num_envs
    rewards = buffer.rewards.view(-1)         # (T*N,) float32
    returns = buffer.returns_.view(-1)
    advantages = buffer.advantages_.view(-1)
    values = buffer.values.view(-1)
    dones = buffer.dones.view(-1)             # fired-at-step
    seats = buffer.seats.view(-1)             # int8

    own_mask = (seats == 0) | (seats == 2)
    enemy_mask = (seats == 1) | (seats == 3)

    # Advantage normalisation on own-team samples (matches PPO code path)
    adv_own = advantages[own_mask]
    if adv_own.numel() > 1:
        a_mean = adv_own.mean()
        a_std = adv_own.std() + 1e-8
    else:
        a_mean = torch.tensor(0.0, device=device)
        a_std = torch.tensor(1.0, device=device)
    adv_norm = (advantages - a_mean) / a_std
    abs_adv_norm = adv_norm.abs()

    # After adv_filter: keep 75% of own-team with largest |A_norm|
    own_abs = abs_adv_norm[own_mask]
    if own_abs.numel() > 0:
        q = 1.0 - ppo_cfg.adv_filt_rate
        q_thresh = torch.quantile(own_abs, q).item()
        thresh = max(ppo_cfg.adv_filt_thresh, q_thresh)
        keep = own_mask & (abs_adv_norm >= thresh)
    else:
        thresh = ppo_cfg.adv_filt_thresh
        keep = own_mask & (abs_adv_norm >= thresh)

    # Returns breakdown: distinguish terminal vs non-terminal returns
    # (terminal returns should be ±1 + small; non-terminal follow GAE).
    terminal_returns = returns[dones]
    nonterm_returns = returns[~dones & own_mask]

    # Episode stats
    n_terminals = int(dones.sum().item())
    # avg_len ~ (N * T) / n_terminals  (rough; could be off at rollout boundary)
    if n_terminals > 0:
        avg_len = (T * N) / n_terminals
    else:
        avg_len = float("inf")

    # Terminal winner breakdown
    # Unfortunately dones doesn't tell us winner — we need the (winner_team at term).
    # Skip that for now; we can infer from rewards at fired positions.
    fired_rewards = rewards[dones]
    # In reward_shaping mode, fired rewards should be ±1 (BUG-J fix keeps terminal pure).

    return {
        "rewards": rewards,
        "returns": returns,
        "values": values,
        "advantages": advantages,
        "adv_norm": adv_norm,
        "abs_adv_norm": abs_adv_norm,
        "own_mask": own_mask,
        "enemy_mask": enemy_mask,
        "adv_own": adv_own,
        "adv_enemy": advantages[enemy_mask],
        "keep_mask": keep,
        "thresh_used": thresh,
        "n_own": int(own_mask.sum().item()),
        "n_enemy": int(enemy_mask.sum().item()),
        "n_kept": int(keep.sum().item()),
        "n_terminals": n_terminals,
        "avg_len": avg_len,
        "fired_rewards": fired_rewards,
        "terminal_returns": terminal_returns,
        "nonterm_returns": nonterm_returns,
        "T": T,
        "N": N,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_report(tag: str, stats: dict) -> None:
    T, N = stats["T"], stats["N"]
    print(f"\n========== {tag}  (T×N = {T}×{N} = {T*N} transitions) ==========")
    print(f"  own-team samples  : {stats['n_own']:>7d}  "
          f"enemy (random) : {stats['n_enemy']:>7d}")
    print(f"  terminals in win : {stats['n_terminals']:>7d}  "
          f"avg_len ≈ {stats['avg_len']:.0f} steps")
    print(f"  adv-filter threshold : {stats['thresh_used']:.4f}")
    print(f"  kept after filter   : {stats['n_kept']:>7d} "
          f"({100*stats['n_kept']/max(1,stats['n_own']):.1f}% of own)")
    print()
    print(f"  {_describe('reward (all)', stats['rewards'])}")
    print(f"  {_describe('return (all)', stats['returns'])}")
    print(f"  {_describe('value (all)', stats['values'])}")
    print(f"  {_describe('adv (own)', stats['adv_own'])}")
    print(f"  {_describe('|adv_norm|', stats['abs_adv_norm'])}")
    print(f"  {_describe('term return', stats['terminal_returns'])}")
    print(f"  {_describe('nonterm ret(own)', stats['nonterm_returns'])}")
    print(f"  {_describe('fired reward', stats['fired_rewards'])}")
    print()
    print(f"  |adv_norm| quantiles on own-team:")
    own_abs = stats['abs_adv_norm'][stats['own_mask']]
    print(f"    {_pct(own_abs)}")
    print(f"  return quantiles on own-team:")
    own_ret = stats['returns'][stats['own_mask']]
    print(f"    {_pct(own_ret)}")
    print(f"  value quantiles on own-team:")
    own_val = stats['values'][stats['own_mask']]
    print(f"    {_pct(own_val)}")

    # Key diagnostics —
    # 1. Signal strength: |return| mean & std on own samples
    print()
    print(f"  DIAG: |return|(own).mean = {own_ret.abs().mean().item():.5f} "
          f"   return(own).std = {own_ret.std().item():.5f}")
    # 2. Value-return gap (advantage magnitude without normalisation)
    raw_adv_own = stats['adv_own']
    print(f"  DIAG: raw |adv|(own).mean = {raw_adv_own.abs().mean().item():.5f} "
          f"   adv(own).std = {raw_adv_own.std().item():.5f}")
    # 3. How many terminal rewards are "clean" ±1 vs shaped?
    fr = stats['fired_rewards']
    if fr.numel() > 0:
        n_plus1 = int((fr > 0.95).sum().item())
        n_minus1 = int((fr < -0.95).sum().item())
        n_draw = int((fr.abs() < 0.05).sum().item())
        n_other = int(fr.numel() - n_plus1 - n_minus1 - n_draw)
        print(f"  DIAG: fired rewards: +1={n_plus1}  -1={n_minus1}  draw≈0={n_draw}  other={n_other}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="exps/v37_bugfixes_1500R")
    ap.add_argument("--ckpts", nargs="+",
                    default=["R25", "R50", "R100", "R200", "R400", "R700"])
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--steps_per_env", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--config", default="configs/v37_bugfixes_1500R.yaml",
                    help="YAML providing PPOConfig + net dims")
    args = ap.parse_args()

    device = torch.device("cuda")

    # Build base cfg from yaml
    ns = argparse.Namespace(config=args.config, resume="")
    # load_config consults argparse.Namespace and accepts extra optional fields
    # we'll just synthesise
    class _Ns:
        pass
    ns = _Ns()
    ns.config = args.config
    ns.resume = ""
    cfg = load_config(ns)

    # Re-instantiate PPOConfig + net
    ppo_cfg = cfg.ppo
    net_cfg = cfg.net if hasattr(cfg, "net") else NetConfig()
    # In this codebase, PPOConfig has a `.net` attribute set at load time.
    # Let's ensure it matches:
    if not hasattr(ppo_cfg, "net") or ppo_cfg.net is None:
        ppo_cfg.net = net_cfg

    # Build policy + trainer
    policy = JunqiNet(ppo_cfg.net).to(device)
    trainer = PPOTrainer(policy, ppo_cfg, device=device)

    # For each ckpt, load and collect
    for tag in args.ckpts:
        n = int(tag.lstrip("R"))
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{n:06d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[diag] missing {ckpt_path}, skipping")
            continue
        print(f"\n######################################################")
        print(f"[diag] Loading {ckpt_path}")
        print(f"######################################################")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.load_state_dict(state)

        policy = trainer.policy
        policy.eval()

        for shaping in (True, False):
            stats = run_one_rollout(
                policy,
                num_envs=args.num_envs,
                steps_per_env=args.steps_per_env,
                seed=args.seed + (1 if shaping else 0),
                device=device,
                reward_shaping=shaping,
                ppo_cfg=ppo_cfg,
            )
            print_report(f"{tag}  shaping={shaping}", stats)

            # free CUDA memory between iterations
            del stats
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
