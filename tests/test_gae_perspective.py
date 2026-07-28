"""tests/test_gae_perspective.py — GAE math correctness for multi-seat games.

These tests verify the cross-team perspective flip + random-opponent gating
introduced in v33c after diagnosing the "half-self-play" failure mode in
v33a/b.

The mathematical invariant we check:

    For a single environment with 4 seats taking turns S->W->N->E->...,
    canonical-frame value V_t is "expected return for seat S_t".
    Reward at terminal step is from the acting seat's perspective.

    Correct GAE delta (with cross-team flip):
        flip = +1 if same_team(S_t, S_{t+1}) else -1
        delta_t = r_t + gamma * flip * V_{t+1} * (1 - done_t) - V_t

    Without the flip, every cross-team boundary contributes a sign
    error of magnitude 2 * gamma * V_{t+1} to the advantage —
    catastrophic in 4-seat alternating play.

We construct synthetic rollouts with known returns and check the buffer
produces the analytically-correct advantages.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Make sure the local junqi_rl is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "junqi_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from junqi_rl.training.rollout_gpu import RolloutBufferGPU


def _make_buffer(
    *,
    T: int,
    N: int = 1,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,  # MC backup makes hand-checking trivial
    random_opponent: bool = False,
    device: str = "cuda",
):
    return RolloutBufferGPU(
        num_envs=N,
        steps_per_env=T,
        gamma=gamma,
        gae_lambda=gae_lambda,
        td_lambda=1.0,
        adv_filt_thresh=0.0,  # no filter: keep everything
        adv_filt_rate=1.0,
        device=device,
        csr_legal_mask=False,
        random_opponent=random_opponent,
    )


def _fill_minimal(buf: RolloutBufferGPU, *, seats, rewards, values, dones=None):
    """Fill buffer with given (T,N) seats/rewards/values; everything else
    is irrelevant for compute_returns. dones default to all-False."""
    T, N = buf.steps_per_env, buf.num_envs
    if dones is None:
        dones = torch.zeros((T, N), dtype=torch.bool, device=buf.device)
    buf.seats = torch.as_tensor(seats, dtype=torch.int8, device=buf.device).reshape(T, N)
    buf.rewards = torch.as_tensor(rewards, dtype=torch.float32, device=buf.device).reshape(T, N)
    buf.values = torch.as_tensor(values, dtype=torch.float32, device=buf.device).reshape(T, N)
    buf.dones = torch.as_tensor(dones, dtype=torch.bool, device=buf.device).reshape(T, N)
    # advantages_/returns_ are the outputs we'll check.
    buf.advantages_ = torch.zeros((T, N), dtype=torch.float32, device=buf.device)
    buf.returns_ = torch.zeros((T, N), dtype=torch.float32, device=buf.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_alternating_seats_4player_self_play_terminal_win() -> None:
    """4-seat rotation, terminal +1 to the LAST acting seat (winner).

    Seats are S0,W1,N2,E3 alternating. Suppose S=N (team 0) wins; the
    LAST acting seat is N (seat 2, team 0), which gets reward +1. From
    seats 1, 3 (team 1, losing) perspective the *true* value of any
    intermediate state is -1; from seats 0, 2 (winning) it's +1.

    We give zero values everywhere (V_t=0) so the bootstrap doesn't
    distort, and check that GAE advantages match the +/- pattern.
    """
    T = 4
    seats = [[0], [1], [2], [3]]   # T=4 steps in one env, 4 unique seats
    rewards = [[0.0], [0.0], [0.0], [+1.0]]  # team 0 wins on last step (acted by seat 2 not 3 here — but we put it at t=3 as if seat 3 tipped the scales while losing)
    # Wait — rephrase the scenario:
    # Let's instead say *seat 3* (team 1) loses at the last step, which
    # means the env reports r=-1 (acting team != winner team).
    # Net: seats 0,2 see "I won"; seats 1,3 see "I lost".
    rewards = [[0.0], [0.0], [0.0], [-1.0]]   # seat 3 acted, lost
    values = [[0.0], [0.0], [0.0], [0.0]]
    buf = _make_buffer(T=T, N=1, gamma=1.0, gae_lambda=1.0, random_opponent=False)
    _fill_minimal(buf, seats=seats, rewards=rewards, values=values)

    # Bootstrap: T-th seat would be seat 0 (cycle). same-team-as-3? no -> flip.
    last_seats = torch.tensor([0], dtype=torch.int64, device=buf.device)
    buf.compute_returns(torch.zeros(1, device=buf.device), last_seats=last_seats)

    # Hand-compute with cross-team flip, gamma=1, lambda=1 (pure MC):
    # next_val = 0, next_team = 0 (seat 0)
    # t=3: cur_team=1 (seat 3), flip=-1 (1!=0)
    #      delta = -1 + 1*(-1)*0*1 - 0 = -1
    #      gae   = -1 + 1*1*1*(-1)*0 = -1
    # t=2: cur_team=0 (seat 2), flip=-1 (0!=1)
    #      delta = 0 + 1*(-1)*0*1 - 0 = 0
    #      gae   = 0 + 1*1*1*(-1)*(-1) = +1
    # t=1: cur_team=1 (seat 1), flip=-1 (1!=0)
    #      delta = 0 + 1*(-1)*0*1 - 0 = 0
    #      gae   = 0 + 1*1*1*(-1)*(+1) = -1
    # t=0: cur_team=0 (seat 0), flip=-1 (0!=1)
    #      delta = 0 + 1*(-1)*0*1 - 0 = 0
    #      gae   = 0 + 1*1*1*(-1)*(-1) = +1
    expected = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=buf.device).reshape(T, 1)
    got = buf.advantages_
    assert torch.allclose(got, expected, atol=1e-6), (
        f"\nExpected (+1, -1, +1, -1) for alternating-seat terminal-loss:\n"
        f"got {got.flatten().tolist()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_random_opponent_zeros_seats_1_3() -> None:
    """In vs-random mode, seats 1 and 3 advantages must be zero
    (legacy v32 behaviour, gated on random_opponent=True)."""
    T = 4
    seats = [[0], [1], [2], [3]]
    rewards = [[0.0], [0.0], [0.0], [-1.0]]
    values = [[0.0], [0.0], [0.0], [0.0]]
    buf = _make_buffer(T=T, N=1, gamma=1.0, gae_lambda=1.0, random_opponent=True)
    _fill_minimal(buf, seats=seats, rewards=rewards, values=values)

    last_seats = torch.tensor([0], dtype=torch.int64, device=buf.device)
    buf.compute_returns(torch.zeros(1, device=buf.device), last_seats=last_seats)

    # Seats 1, 3 should be zeroed.
    assert buf.advantages_[1, 0].item() == 0.0
    assert buf.advantages_[3, 0].item() == 0.0
    # Seats 0, 2 should be non-zero (their advantage was non-zero
    # before the post-GAE zeroing pass).
    assert buf.advantages_[0, 0].item() != 0.0
    assert buf.advantages_[2, 0].item() != 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_play_keeps_all_seats_nonzero() -> None:
    """In self-play mode, every seat's advantage must remain non-zero
    (advantages reflect each seat's own learning signal)."""
    T = 4
    seats = [[0], [1], [2], [3]]
    rewards = [[0.0], [0.0], [0.0], [-1.0]]
    values = [[0.0], [0.0], [0.0], [0.0]]
    buf = _make_buffer(T=T, N=1, gamma=1.0, gae_lambda=1.0, random_opponent=False)
    _fill_minimal(buf, seats=seats, rewards=rewards, values=values)

    last_seats = torch.tensor([0], dtype=torch.int64, device=buf.device)
    buf.compute_returns(torch.zeros(1, device=buf.device), last_seats=last_seats)

    # Every seat must have non-zero advantage (the cross-team flipped GAE
    # trickles the terminal reward to every step in the right sign).
    for t in range(T):
        assert buf.advantages_[t, 0].item() != 0.0, (
            f"seat {seats[t][0]} at t={t} got zero advantage in self-play "
            f"-- bug 1 may have re-emerged"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_same_team_consecutive_no_flip() -> None:
    """Q12 case: when one team has lost a seat, consecutive actors may
    belong to the SAME team and no flip should happen.

    Setup: seats 0, 2 (team 0, both alive) alternate after seat 1 died
    via Q12 cascade. Terminal +1 to last seat (still team 0) means
    every team-0 step should get *positive* advantage.
    """
    T = 4
    # Seat-1 (team 1) died early; from then on only seats 0 and 2 act.
    seats = [[0], [2], [0], [2]]
    rewards = [[0.0], [0.0], [0.0], [+1.0]]   # seat 2 wins
    values = [[0.0], [0.0], [0.0], [0.0]]
    buf = _make_buffer(T=T, N=1, gamma=1.0, gae_lambda=1.0, random_opponent=False)
    _fill_minimal(buf, seats=seats, rewards=rewards, values=values)

    # Bootstrap: t=4 would be seat 0 (cycle), same team as seat 2 → flip=+1
    last_seats = torch.tensor([0], dtype=torch.int64, device=buf.device)
    buf.compute_returns(torch.zeros(1, device=buf.device), last_seats=last_seats)

    # All seats are team 0; gamma=lambda=1 with same-team flip=+1 throughout.
    # GAE recurrence: gae_t = delta_t + 1*1*1*(+1)*gae_{t+1} (same team)
    # delta_4 (terminal) = +1 + 1*(+1)*0*1 - 0 = +1
    # delta_3 = 0 + 1*(+1)*0*1 - 0 = 0; gae_3 = 0 + 1*(+1)*1 = +1
    # ... all stages should see a positive advantage of +1.
    expected = torch.full((T, 1), 1.0, device=buf.device)
    got = buf.advantages_
    assert torch.allclose(got, expected, atol=1e-6), (
        f"\nSame-team consecutive (Q12 case) — expected all +1:\n"
        f"got {got.flatten().tolist()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_perspective_bug_old_behaviour_was_wrong() -> None:
    """Regression sentinel: confirm the new code does NOT reproduce the
    OLD buggy behaviour (which used +V_{t+1} unconditionally, mixing
    perspectives). Old code on the alternating-seat scenario would have
    produced GAE all positive (+1 propagated forward without sign flip);
    new code produces alternating +1/-1.

    This test would FAIL on pre-fix code, PASS on fixed code — exactly
    what we want for catching accidental reverts.
    """
    T = 4
    seats = [[0], [1], [2], [3]]
    rewards = [[0.0], [0.0], [0.0], [+1.0]]   # last actor (seat 3) wins
    values = [[0.0], [0.0], [0.0], [0.0]]
    buf = _make_buffer(T=T, N=1, gamma=1.0, gae_lambda=1.0, random_opponent=False)
    _fill_minimal(buf, seats=seats, rewards=rewards, values=values)

    last_seats = torch.tensor([0], dtype=torch.int64, device=buf.device)
    buf.compute_returns(torch.zeros(1, device=buf.device), last_seats=last_seats)

    # Old (buggy) code would have given advantages [+1, +1, +1, +1] (no flip).
    # New code: seat 3 won (+1); from seat 2's perspective (team 0, opposite
    # of seat 3 team) it's a LOSS, so seat 2's advantage is -1; etc.
    expected = torch.tensor([-1.0, +1.0, -1.0, +1.0], device=buf.device).reshape(T, 1)
    got = buf.advantages_
    assert torch.allclose(got, expected, atol=1e-6), (
        f"\nAlternating-seat terminal-WIN by team 1: expected (-1,+1,-1,+1) "
        f"(perspective-flipped); got {got.flatten().tolist()}.\n"
        f"If this fails to be alternating, the cross-team flip is broken."
    )


if __name__ == "__main__":
    # Allow standalone runs.
    test_alternating_seats_4player_self_play_terminal_win()
    print("[1/5] alternating-seat terminal loss: OK")
    test_random_opponent_zeros_seats_1_3()
    print("[2/5] random_opponent zeros seats 1,3: OK")
    test_self_play_keeps_all_seats_nonzero()
    print("[3/5] self-play keeps all seats: OK")
    test_same_team_consecutive_no_flip()
    print("[4/5] same-team Q12 no flip: OK")
    test_no_perspective_bug_old_behaviour_was_wrong()
    print("[5/5] perspective regression sentinel: OK")
    print("ALL GAE PERSPECTIVE TESTS PASSED")
