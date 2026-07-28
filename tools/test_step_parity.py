"""Strict parity: step_device_torch vs old step + numpy rewards."""
import sys
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import _per_seat_terminal_rewards, _UNROTATE_LUT_STACK

N = 128
NUM_STEPS = 50
np.random.seed(7)

# Two independent rollouts starting from same state
r_new = GpuRollout(num_envs=N)
r_new.reset(seed_base=42)
r_old = GpuRollout(num_envs=N)
r_old.reset(seed_base=42)

done_old = np.zeros(N, dtype=bool)
mismatches = 0

for step_i in range(NUM_STEPS):
    # Get turn
    turn_np = np.asarray(r_old.state.copy_turn_to_host(), dtype=np.int8).reshape(N)

    # Pick canonical actions from the new rollout's legal mask
    lm = r_new.legal_mask_canonical_torch(turn_np)
    actions_can_t = torch.zeros(N, dtype=torch.int32, device='cuda')
    for i in range(N):
        legal = lm[i].nonzero(as_tuple=False).squeeze(-1)
        if len(legal) > 0:
            idx = int(np.random.randint(len(legal)))
            actions_can_t[i] = legal[idx]

    actions_can_np = actions_can_t.cpu().numpy()

    # === OLD PATH ===
    seat_idx = turn_np.astype(np.int64)
    actions_world_np = _UNROTATE_LUT_STACK[seat_idx, actions_can_np.astype(np.int64)].astype(np.int32)
    result_old = r_old.step(actions_world_np)
    new_term_old = result_old["terminated"].astype(bool)
    new_win_old = result_old["winner_team"].astype(np.int8)
    new_draw_old = result_old["draw"].astype(bool)
    fired_old = new_term_old & ~done_old
    rewards_old = _per_seat_terminal_rewards(fired_old, new_win_old, new_draw_old, turn_np)
    # Reward shaping
    ev_old = result_old["event"].astype(np.int8)
    live_old = ~done_old
    rewards_old += np.where(live_old & (ev_old == 2), 0.05, 0.0).astype(np.float32)
    rewards_old += np.where(live_old & (ev_old == 4), -0.05, 0.0).astype(np.float32)
    rewards_old += np.where(live_old & (ev_old == 3), -0.02, 0.0).astype(np.float32)
    done_old = new_term_old.copy()

    # === NEW PATH ===
    seats_t = torch.from_numpy(turn_np).to('cuda')
    result_new = r_new.step_device_torch(actions_can_t, seats_t)
    rewards_new = result_new["rewards"].cpu().numpy()
    term_new = result_new["terminated"].cpu().numpy()
    event_new = result_new["event"].cpu().numpy()

    # Compare
    reward_match = np.allclose(rewards_old, rewards_new, atol=1e-6)
    term_match = np.array_equal(new_term_old, term_new)
    event_match = np.array_equal(ev_old, event_new)

    if not (reward_match and term_match and event_match):
        mismatches += 1
        if mismatches <= 3:
            print(f"Step {step_i}: MISMATCH")
            if not reward_match:
                diff = np.where(np.abs(rewards_old - rewards_new) > 1e-6)[0]
                print(f"  reward diff at envs {diff[:5]}: old={rewards_old[diff[:5]]} new={rewards_new[diff[:5]]}")
            if not term_match:
                diff = np.where(new_term_old != term_new)[0]
                print(f"  terminated diff at envs {diff[:5]}")
            if not event_match:
                diff = np.where(ev_old != event_new)[0]
                print(f"  event diff at envs {diff[:5]}: old={ev_old[diff[:5]]} new={event_new[diff[:5]]}")

if mismatches == 0:
    print(f"PARITY CHECK PASSED: {NUM_STEPS} steps × {N} envs, all match!")
else:
    print(f"PARITY CHECK FAILED: {mismatches}/{NUM_STEPS} steps had mismatches")
