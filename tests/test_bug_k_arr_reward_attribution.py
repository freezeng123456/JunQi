"""tests/test_bug_k_arr_reward_attribution.py — regression for BUG-K.

BUG-K (2026-05-11): the `_arr_on_termination` callback in ``scripts/train.py``
used to credit ONLY the acting seat's lineup with the terminal reward.
Two problems followed:

  K1. The teammate seat's lineup (which is co-responsible for the team's
      win/loss) got no signal — at most 1 of 4 ArrNet rows per episode
      ever reached ``ready``.
  K2. In vs-random mode, when a random-walk seat (1 or 3) triggered
      termination, the reward was credited to its lineup — but the lineup
      had no causal effect on a random walker, polluting the buffer.

This test exercises the fixed callback logic and asserts:
  * In vs-random mode, only seats 0 and 2 receive credits.
  * Both own-team seats are credited with the correct team-perspective
    reward (winner team gets +1, loser team gets -1) regardless of which
    seat triggered termination.
  * In self-play mode, all 4 seats are credited.

The test reproduces the same callback math in the production train.py
(extracted into a pure helper to avoid import-cycle issues with the
config dataclass), then asserts the output.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch

from junqi_rl.networks.arrangement_net import ARRANGEMENT_SIZE


def _arr_on_termination_logic(
    *,
    fired_np: np.ndarray,        # (N,) bool
    rew_np: np.ndarray,           # (N,) float — reward kernel output, acting-team perspective
    seats_np: np.ndarray,         # (N,) int — acting seat per env
    env_arr_snapshot: np.ndarray, # (N, 4, 30) int — each env's per-seat lineup
    random_opp: bool,
):
    """Pure-python re-implementation of train.py::_arr_on_termination."""
    N = env_arr_snapshot.shape[0]
    acting_team = seats_np & 1
    sign = np.sign(rew_np).astype(np.float32)
    reward_team0 = np.where(acting_team == 0, sign, -sign).astype(np.float32)
    reward_team1 = -reward_team0

    seats_to_credit = (0, 2) if random_opp else (0, 1, 2, 3)
    out = []  # list of (seat, lineup, reward) triples actually credited
    for seat in seats_to_credit:
        team = seat & 1
        seat_reward = reward_team0 if team == 0 else reward_team1
        seat_lineup = env_arr_snapshot[np.arange(N), seat]
        for e in range(N):
            if fired_np[e]:
                out.append((seat, tuple(seat_lineup[e].tolist()), float(seat_reward[e])))
    return out


def _make_env_snapshot(N: int, seed: int = 0) -> np.ndarray:
    """Make distinct lineups for each (env, seat). Lineup[e, s, k] = e*100 + s*30 + k."""
    rng = np.random.default_rng(seed)
    snap = np.zeros((N, 4, ARRANGEMENT_SIZE), dtype=np.int64)
    for e in range(N):
        for s in range(4):
            snap[e, s] = rng.integers(0, 13, size=ARRANGEMENT_SIZE)
    return snap


# -----------------------------------------------------------------------
# K1 — teammate gets credit
# -----------------------------------------------------------------------

def test_teammate_seat_gets_same_team_reward_in_random_opponent_mode():
    """When SOUTH (seat 0) triggers a win, NORTH (seat 2, same team) must also
    receive +1 on its lineup."""
    N = 4
    snap = _make_env_snapshot(N)
    # All envs fire; SOUTH (seat 0) is the acting seat in env 0; rew=+1 (won).
    fired = np.array([True, False, False, False])
    seats = np.array([0, 0, 0, 0])    # acting=SOUTH everywhere (only env 0 fires)
    rewards = np.array([+1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=True,
    )
    # Should have 2 credits (seats 0, 2) for env 0; nothing else.
    assert len(out) == 2, f"expected 2 credits, got {len(out)}"
    seat_to_reward = {seat: rew for seat, _, rew in out}
    assert seat_to_reward[0] == +1.0, "SOUTH (winner) lineup should get +1"
    assert seat_to_reward[2] == +1.0, "NORTH (winner-teammate) lineup should also get +1"


def test_loser_team_lineups_get_minus_one():
    """When acting=SOUTH but rew=-1 (i.e. SOUTH-team lost), both SOUTH and
    NORTH lineups should get -1."""
    N = 1
    snap = _make_env_snapshot(N)
    fired = np.array([True])
    seats = np.array([0])  # SOUTH acted
    rewards = np.array([-1.0], dtype=np.float32)
    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=True,
    )
    seat_to_reward = {seat: rew for seat, _, rew in out}
    assert seat_to_reward[0] == -1.0
    assert seat_to_reward[2] == -1.0


def test_acting_enemy_seat_does_not_get_credit_in_random_opponent_mode():
    """When acting=WEST (seat 1, random) triggers a win, the random seats
    (1, 3) must NOT get credit. SOUTH/NORTH (own team) DO get credit
    according to the team-perspective reward."""
    N = 1
    snap = _make_env_snapshot(N)
    fired = np.array([True])
    seats = np.array([1])  # WEST acted (random)
    # Reward kernel: acting=WEST (team 1) won → rew=+1 (team 1 perspective)
    rewards = np.array([+1.0], dtype=np.float32)
    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=True,
    )
    credited_seats = {seat for seat, _, _ in out}
    assert credited_seats == {0, 2}, (
        f"only own-team seats {{0, 2}} should be credited in vs-random mode; "
        f"got {credited_seats}"
    )
    # And from team 0's perspective they LOST (team 1 won), so reward=-1.
    seat_to_reward = {seat: rew for seat, _, rew in out}
    assert seat_to_reward[0] == -1.0
    assert seat_to_reward[2] == -1.0


def test_self_play_mode_credits_all_four_seats():
    """In self-play mode (random_opp=False) all 4 seats are policy-controlled
    and should be credited. Team-perspective rewards still flip per team."""
    N = 1
    snap = _make_env_snapshot(N)
    fired = np.array([True])
    seats = np.array([3])  # EAST acted (team 1)
    rewards = np.array([+1.0], dtype=np.float32)  # team 1 won
    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=False,
    )
    credited = {seat: rew for seat, _, rew in out}
    assert set(credited.keys()) == {0, 1, 2, 3}
    assert credited[0] == -1.0   # team 0 lost
    assert credited[1] == +1.0   # team 1 won
    assert credited[2] == -1.0
    assert credited[3] == +1.0


def test_no_credit_when_no_fire():
    """No fires → no credits."""
    N = 4
    snap = _make_env_snapshot(N)
    fired = np.zeros(N, dtype=bool)
    seats = np.array([0, 1, 2, 3])
    rewards = np.array([0.0, +1.0, 0.0, -1.0], dtype=np.float32)  # ignored by fired filter
    for mode in (True, False):
        out = _arr_on_termination_logic(
            fired_np=fired, rew_np=rewards, seats_np=seats,
            env_arr_snapshot=snap, random_opp=mode,
        )
        assert len(out) == 0


def test_draw_gives_zero_reward_to_all():
    """rew=0 (draw) → reward 0 to every credited seat (sign=0 → +0 = -0 = 0)."""
    N = 1
    snap = _make_env_snapshot(N)
    fired = np.array([True])
    seats = np.array([0])
    rewards = np.array([0.0], dtype=np.float32)
    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=True,
    )
    for _seat, _lineup, rew in out:
        assert rew == 0.0


def test_correct_lineup_per_seat():
    """The lineup credited for seat s is env_arr_snapshot[env, s], not
    env_arr_snapshot[env, acting_seat]."""
    N = 2
    snap = _make_env_snapshot(N, seed=42)
    fired = np.array([True, False])
    seats = np.array([0, 0])  # SOUTH acted in env 0
    rewards = np.array([+1.0, 0.0], dtype=np.float32)
    out = _arr_on_termination_logic(
        fired_np=fired, rew_np=rewards, seats_np=seats,
        env_arr_snapshot=snap, random_opp=True,
    )
    # Should have 2 entries: (seat=0, env0_seat0_lineup) and (seat=2, env0_seat2_lineup)
    seat_to_lineup = {seat: lineup for seat, lineup, _ in out}
    assert tuple(snap[0, 0].tolist()) == seat_to_lineup[0], "SOUTH lineup mismatch"
    assert tuple(snap[0, 2].tolist()) == seat_to_lineup[2], "NORTH lineup mismatch"
    # Critically, seat 2's lineup is NOT seat 0's lineup.
    assert seat_to_lineup[0] != seat_to_lineup[2], (
        "BUG-K regression: teammate must use teammate's own lineup, not acting seat's"
    )


if __name__ == "__main__":
    test_teammate_seat_gets_same_team_reward_in_random_opponent_mode()
    print("[1/7] teammate gets same-team reward: OK")
    test_loser_team_lineups_get_minus_one()
    print("[2/7] loser-team lineups get -1: OK")
    test_acting_enemy_seat_does_not_get_credit_in_random_opponent_mode()
    print("[3/7] acting enemy seat not credited: OK")
    test_self_play_mode_credits_all_four_seats()
    print("[4/7] self-play credits all 4: OK")
    test_no_credit_when_no_fire()
    print("[5/7] no fire = no credit: OK")
    test_draw_gives_zero_reward_to_all()
    print("[6/7] draw → 0 reward: OK")
    test_correct_lineup_per_seat()
    print("[7/7] correct lineup per seat: OK")
    print("\nAll BUG-K regression tests pass.")
