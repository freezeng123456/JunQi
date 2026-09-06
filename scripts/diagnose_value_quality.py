#!/usr/bin/env python3
"""diagnose_value_quality.py — Does V(s) predict the outcome at all?

Load a ckpt, collect an EMA-policy rollout, then measure:
  * Correlation between V(s_t) and actual (bootstrapped) return_t, on
    own-team rows only. If V is useless we'd see r ~ 0.
  * Explained variance: 1 - Var(ret - V) / Var(ret).
  * Correlation between V(s) and the FINAL outcome of the episode this
    row belongs to. (This needs per-row "episode result" which we can
    compute by walking forward until we hit a terminal.)

If V predicts nothing, shaping noise dominates advantage → policy is
training on pure noise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "junqi_rl"))
sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np
import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
from junqi_rl.training.rollout_gpu import RolloutBufferGPU
from junqi_rl.networks.junqi_net import JunqiNet

import train as _train_mod
from junqi_rl.training.ppo import PPOTrainer


def episode_final_outcome(dones, seats, rewards):
    """For each (t, n), find the next terminal step and return the
    signed outcome for the player at seat (t, n).
    Returns tensor (T, N) with values in {-1, 0, +1} (+1 = our seat's
    team won the episode this row belongs to; 0 if no terminal in rollout).
    """
    T, N = dones.shape
    outcome = torch.zeros((T, N), dtype=torch.float32, device=dones.device)
    # For each env, propagate the next terminal's reward-from-POV back.
    # Walk from t=T-1 to 0: outcome[t, n] = outcome[t+1, n] flipped if team differs,
    # reset to reward[t, n] at done rows.
    for t in reversed(range(T)):
        if t == T - 1:
            prev = torch.zeros(N, dtype=torch.float32, device=dones.device)
            prev_team = ((seats[t].to(torch.int64) & 1) ^ 1)  # bootstrap: assume opp
        # done → outcome = reward at this step (always from acting seat POV)
        d = dones[t]
        r = rewards[t]
        cur_team = (seats[t].to(torch.int64) & 1)
        # if not done, outcome[t] = flip * outcome[t+1] (flip if team changes)
        if t == T - 1:
            # at rollout end, no "next". Leave outcome at 0 for non-done rows.
            outcome[t] = torch.where(d, r, torch.zeros_like(r))
            prev = outcome[t]
            prev_team = cur_team
        else:
            flip = (cur_team == prev_team).to(torch.float32) * 2.0 - 1.0
            # non-done: propagate
            non_done = ~d
            outcome[t] = torch.where(
                d, r,                     # terminal: take reward directly
                flip * prev,               # non-terminal: flipped next outcome
            )
            prev = outcome[t]
            prev_team = cur_team
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["R25", "R100", "R200", "R400", "R700"])
    ap.add_argument("--ckpt_dir", default="exps/v37_bugfixes_1500R")
    ap.add_argument("--config", default="configs/v37_bugfixes_1500R.yaml")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps_per_env", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=31415)
    args = ap.parse_args()

    device = torch.device("cuda")

    class _Ns:
        pass
    ns = _Ns()
    ns.config = args.config
    ns.resume = ""
    cfg = _train_mod.load_config(ns)

    policy = JunqiNet(cfg.ppo.net).to(device)
    trainer = PPOTrainer(policy, cfg.ppo, device=device)

    env = GpuRollout(num_envs=args.num_envs, device_id=0)

    # Pre-allocate a single buffer and reuse across ckpts (avoid OOM)
    buf = RolloutBufferGPU(
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        td_lambda=cfg.ppo.td_lambda,
        adv_filt_thresh=cfg.ppo.adv_filt_thresh,
        adv_filt_rate=cfg.ppo.adv_filt_rate,
        device=device,
        random_opponent=True,
        train_value_on_random_seats=False,
    )

    print(f"{'tag':<8s}  {'n_own':>8s}  {'term':>6s}  "
          f"{'V.std':>8s}  {'V.mean':>9s}  "
          f"{'ret.std':>8s}  "
          f"{'corr(V,ret)':>12s}  {'corr(V,final)':>14s}  "
          f"{'explVar(ret)':>13s}  {'explVar(final)':>14s}")

    for tag in args.ckpts:
        n = int(tag.lstrip("R"))
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{n:06d}.pt")
        if not os.path.exists(ckpt_path):
            continue
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.load_state_dict(state)

        # Reset the buffer's write pointer so collect fills from scratch.
        buf._ptr = 0

        with torch.no_grad():
            collect_rollout_gpu_v2(
                rollout_world=env,
                policy=trainer.policy,
                buffer=buf,
                device=device,
                seed_base=args.seed + n,
                reset_at_start=True,
                reward_shaping=True,
                random_opponent=True,
                use_compile=False,
                autocast_dtype=torch.bfloat16,
            )

        buf.compute_returns(
            torch.zeros(args.num_envs, device=device),
            last_seats=buf.seats[-1].clone(),
        )

        seats = buf.seats
        own_mask = (seats == 0) | (seats == 2)
        dones = buf.dones
        rewards = buf.rewards
        values = buf.values
        returns = buf.returns_

        # Compute final-episode outcome tensor
        outcome = episode_final_outcome(dones, seats, rewards)

        V = values[own_mask].float()
        R_ = returns[own_mask].float()
        O = outcome[own_mask].float()

        # Spearman would be better but pearson is fine here
        def corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            denom = (a.std() * b.std()).clamp_min(1e-12)
            return (a * b).mean() / denom

        c_ret = corr(V, R_).item()
        c_final = corr(V, O).item()
        # Explained variance: 1 - Var(R - V) / Var(R)
        ev_ret = (1.0 - (R_ - V).var() / R_.var().clamp_min(1e-12)).item()
        ev_final = (1.0 - (O - V).var() / O.var().clamp_min(1e-12)).item()

        n_own = V.numel()
        n_term = int(dones.sum().item())
        print(f"{tag:<8s}  {n_own:>8d}  {n_term:>6d}  "
              f"{V.std().item():>8.5f}  {V.mean().item():>+9.5f}  "
              f"{R_.std().item():>8.5f}  "
              f"{c_ret:>12.4f}  {c_final:>14.4f}  "
              f"{ev_ret:>13.4f}  {ev_final:>14.4f}")


if __name__ == "__main__":
    main()
