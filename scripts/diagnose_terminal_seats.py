#!/usr/bin/env python3
"""diagnose_terminal_seats.py — Check whether terminal-step seats carry -1 reward.

Hypothesis under test
---------------------
In vs-random mode, the game ALWAYS ends on a winner's move (the player who
eats the flag / causes opponent Q12 is by construction the winner). If
team 1 (random) made the winning move, then:
  * ``fired_reward`` (written by CUDA kernel from acting-seat viewpoint) = +1
  * BUT the acting seat is 1 or 3 → buffered row is on enemy_mask
  * compute_returns zeros the advantage AND (legacy) overwrites return
    with value, erasing the signal that "team 0 LOST" at that moment
  * The *preceding* team-0 rows only see this via GAE flip through V(s_{t+1}),
    but V is trained ≈ 0 (no shaping), so there's no signal to flip.

Under this hypothesis, the v37 policy never sees a -1 reward anywhere
reachable by own-team policy gradient, even when it actually loses many
games.

This script dumps the seat distribution on fired-step rows to confirm.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exps/v37_bugfixes_1500R/ckpt_000200.pt")
    ap.add_argument("--config", default="configs/v37_bugfixes_1500R.yaml")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps_per_env", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=9999)
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
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    trainer.load_state_dict(state)

    env = GpuRollout(num_envs=args.num_envs, device_id=0)
    buffer = RolloutBufferGPU(
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

    with torch.no_grad():
        collect_rollout_gpu_v2(
            rollout_world=env,
            policy=trainer.ema.model,
            buffer=buffer,
            device=device,
            seed_base=args.seed,
            reset_at_start=True,
            reward_shaping=False,  # no confusion from shaping
            random_opponent=True,
            use_compile=False,
            autocast_dtype=torch.bfloat16,
        )

    rewards = buffer.rewards  # (T, N)
    dones = buffer.dones      # (T, N)
    seats = buffer.seats      # (T, N) int8

    # Flatten
    R = rewards.view(-1)
    D = dones.view(-1)
    S = seats.view(-1)

    fired = D.bool()
    n_fired = int(fired.sum().item())
    print(f"\n[diag] ckpt = {args.ckpt}")
    print(f"[diag] rollout: T={args.steps_per_env}  N={args.num_envs}  total={args.steps_per_env*args.num_envs}")
    print(f"[diag] total terminal events: {n_fired}")

    if n_fired == 0:
        print("[diag] no terminals!")
        return

    fr = R[fired]
    fs = S[fired]

    print(f"\n[diag] terminal-step reward distribution:")
    n_plus = int((fr > 0.5).sum().item())
    n_minus = int((fr < -0.5).sum().item())
    n_zero = int((fr.abs() < 0.1).sum().item())
    print(f"  +1 (own team won from acting-seat POV): {n_plus}")
    print(f"  -1 (own team lost from acting-seat POV): {n_minus}")
    print(f"   0 (draw):                                {n_zero}")

    print(f"\n[diag] terminal-step SEAT distribution:")
    for s in [0, 1, 2, 3]:
        n = int((fs == s).sum().item())
        team = "team0 (ours)" if s in (0, 2) else "team1 (random)"
        n_win = int(((fs == s) & (R[fired] > 0.5)).sum().item())
        n_loss = int(((fs == s) & (R[fired] < -0.5)).sum().item())
        print(f"  seat={s} {team:<16s}: total={n:>4d}   won={n_win:>4d}  lost={n_loss:>4d}")

    # Now the critical question: cross-reference with game outcome.
    # fr at seat∈{0,2} (own) should be +1 (we won) or -1 (we lost)
    # fr at seat∈{1,3} (random) should be +1 (random won) or -1 (random lost)
    # In acting-seat POV, fired rewards are ALWAYS +1 (winner makes the last move).
    # So fired rows at seat∈{0,2} with reward=+1 ⇒ team-0 won
    #    fired rows at seat∈{1,3} with reward=+1 ⇒ team-1 won ⇒ team-0 LOST
    own_seat_mask = (fs == 0) | (fs == 2)
    enemy_seat_mask = (fs == 1) | (fs == 3)
    team0_wins = int(((R[fired] > 0.5) & own_seat_mask).sum().item())
    team0_losses = int(((R[fired] > 0.5) & enemy_seat_mask).sum().item())
    team0_losses_explicit = int(((R[fired] < -0.5) & own_seat_mask).sum().item())
    # Verify: terminal rows at own seat with -1 means we made a suicidal move
    # that ended the game (rare in 4-seat junqi)
    print(f"\n[diag] INFERRED GAME OUTCOME (terminal row seat + sign):")
    print(f"  team-0 WINS  (own seat ended, reward=+1) : {team0_wins}")
    print(f"  team-0 LOSES (enemy seat ended, reward=+1): {team0_losses}")
    print(f"  team-0 LOSES (own seat ended, reward=-1) : {team0_losses_explicit}")
    print(f"  true win_rate ≈ {team0_wins / max(1, team0_wins + team0_losses + team0_losses_explicit):.3f}")

    # Now let's check what buffer.returns_ looks like after compute_returns.
    # For enemy-seat terminal rows, returns are overwritten to values
    # (legacy behaviour), which destroys the loss signal.
    last_values = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    last_seats = buffer.seats[-1].clone()
    buffer.compute_returns(last_values, last_seats=last_seats)

    returns = buffer.returns_.view(-1)
    advantages = buffer.advantages_.view(-1)
    # On fired rows at enemy seat: returns should equal values now (overwrite)
    enemy_fired = fired & ((S == 1) | (S == 3))
    own_fired = fired & ((S == 0) | (S == 2))
    if int(enemy_fired.sum().item()) > 0:
        enemy_fired_returns = returns[enemy_fired]
        enemy_fired_rewards = R[enemy_fired]
        print(f"\n[diag] Enemy-seat terminal rows: {int(enemy_fired.sum().item())}")
        print(f"  reward  mean={enemy_fired_rewards.mean().item():+.4f} "
              f"std={enemy_fired_rewards.std().item():.4f}")
        print(f"  return  mean={enemy_fired_returns.mean().item():+.4f} "
              f"std={enemy_fired_returns.std().item():.4f}  "
              "(should be ≈ value, NOT ±1)")

    if int(own_fired.sum().item()) > 0:
        own_fired_returns = returns[own_fired]
        own_fired_rewards = R[own_fired]
        print(f"\n[diag] Own-seat terminal rows: {int(own_fired.sum().item())}")
        print(f"  reward  mean={own_fired_rewards.mean().item():+.4f} "
              f"std={own_fired_rewards.std().item():.4f}")
        print(f"  return  mean={own_fired_returns.mean().item():+.4f} "
              f"std={own_fired_returns.std().item():.4f}")

    # The critical test: how does the GAE flip propagate the LOSS signal
    # back to prior own-team rows when the loss happened on an enemy seat?
    # It does so by computing delta_t = r_t + gamma * flip * V_{t+1} * (1-done_t)
    # At t=T_end (enemy seat with reward=+1), r_t is overwritten... wait
    # actually r_t is NOT overwritten. Only returns_[enemy] is overwritten.
    # Let me re-check:
    #   compute_returns BACKWARD LOOP uses self.rewards[t], which is the
    #   raw reward (still +1 for enemy terminal rows). Good.
    #   But the LOOP also computes gae starting from last_val=0.
    #   At t=T_end: mask=0 (done), so delta = r_t + 0 - V(s_t) = +1 - V.
    #   Then gae = delta (mask=0 kills recursion).
    #   advantage[T_end] = +1 - V (large positive from enemy POV!)
    #   return[T_end]    = advantage + V = +1.
    #   At t=T_end-1 (if own seat, opposite team to T_end):
    #     flip = -1 (different teams)
    #     next_val = V(s_{T_end}) (from enemy POV)
    #     delta = r[t=T_end-1] + gamma * (-1) * V(s_{T_end}) * mask - V(s_{T_end-1})
    #     The (-1) * V(s_{T_end}) flip IS the mechanism. V(s_{T_end}) ≈ 0.003
    #     so even if it correctly predicted the win, the signal at t-1 is only
    #     -0.003. That's why |adv| is tiny!
    #
    # Then self.advantages[enemy_mask] = 0 (correct for policy grad gating)
    # AND self.returns[enemy_mask] = values[enemy_mask] (overwrite).
    # So after that, the fired-row returns at enemy seat are CLOBBERED.
    # But the gae flow INTO the own-team row at T_end-1 was already computed
    # using the RAW reward (+1 at enemy seat), so it should propagate.
    # Unless... let me re-read the code carefully.
    # Yes, the flow goes: loop fills advantages_ first, THEN enemy_mask
    # overwrite. So own-team rows at t=T_end-1 DO get the flipped V_{t+1}
    # bootstrap... but V_{t+1} is approx 0, so the signal is ~0.
    #
    # WAIT - that's the bug path. V(s) is trained by returns, and returns
    # at enemy seat are set to values (= value predicts itself = zero loss).
    # So V never learns to predict the outcome at enemy states! Without a
    # meaningful V at enemy states, the GAE flip through them is useless.

    # Let's check how much signal actually lands on own-team rows PRECEDING
    # an enemy-seat terminal (those are the "we lost" transitions).
    # For each enemy_fired row at (t_end, n), find max t' < t_end with
    # own-team seat, and inspect advantage there.
    T = buffer.steps_per_env
    N = buffer.num_envs
    dones_2d = buffer.dones            # (T, N) bool
    seats_2d = buffer.seats            # (T, N) int8
    adv_2d = buffer.advantages_        # (T, N)
    ret_2d = buffer.returns_

    own_team_mask_2d = (seats_2d == 0) | (seats_2d == 2)

    # For each env n, find its terminal step t_end and the terminal seat.
    # Assumes at most one terminal per env during collection (env resets
    # kick in right after fire, so any further terminals in the same
    # rollout would be a NEW episode).
    # Actually env DOES reset and can terminate multiple times.
    # Let's aggregate: for each (t, n) with dones_2d[t, n]=True, look
    # back k=1..10 steps and see if there's an own-team row; report
    # its advantage & return.

    # Flatten via (t, n) indexing
    term_positions = dones_2d.nonzero(as_tuple=False)  # (K, 2)
    preceding_own_adv = []
    preceding_own_ret = []
    for row in term_positions.cpu().tolist():
        t_end, n = row
        seat_end = int(seats_2d[t_end, n].item())
        # Look back up to 4 steps for an own-team row in the SAME episode
        for dt in range(1, 5):
            t = t_end - dt
            if t < 0:
                break
            if dones_2d[t, n].item():
                # hit a prior terminal — different episode
                break
            s = int(seats_2d[t, n].item())
            if s in (0, 2):
                preceding_own_adv.append((seat_end, float(adv_2d[t, n].item()),
                                          float(ret_2d[t, n].item())))
                break

    # Separate by whether terminal was on enemy seat (we lost) or own seat (we won)
    from collections import defaultdict
    groups = defaultdict(list)  # key: "own_won" / "enemy_won"
    for seat_end, a, r in preceding_own_adv:
        key = "own_won (team-0 winner)" if seat_end in (0, 2) else "enemy_won (team-0 LOST)"
        groups[key].append((a, r))

    print(f"\n[diag] === KEY FINDING: signal landing on own-team rows "
          f"preceding each terminal ===")
    for key, vals in groups.items():
        if not vals:
            continue
        advs = np.array([v[0] for v in vals])
        rets = np.array([v[1] for v in vals])
        print(f"  {key:<32s}  n={len(vals):>4d}  "
              f"adv: mean={advs.mean():+.5f} std={advs.std():.5f} "
              f"|adv|.mean={np.abs(advs).mean():.5f}")
        print(f"  {' '*34}           "
              f"ret: mean={rets.mean():+.5f} std={rets.std():.5f}")


if __name__ == "__main__":
    main()
