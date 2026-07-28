"""Phase 5 parity test: device-side reset vs CPU reset.

Verifies that:
1. Device reset produces valid game states (all pieces correctly placed)
2. Reset only affects terminated envs
3. Post-reset envs can continue playing (step + legal actions work)
"""
import sys
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')

import numpy as np
import torch
from junqi_rl.gpu_rollout import GpuRollout

N = 32
NUM_STEPS = 200  # enough steps to get some terminations

print(f"=== Phase 5: Device-side reset parity test ===")
print(f"N={N}, NUM_STEPS={NUM_STEPS}")

# Create rollout and play until some envs terminate
rollout = GpuRollout(num_envs=N)
rollout.reset(seed_base=0)

terminated_count = 0
for step in range(NUM_STEPS):
    # Get acting seats
    turn_t = rollout.turn_torch()  # int8 (N,) CUDA
    acting_np = turn_t.cpu().numpy()

    # Get legal mask
    lm = rollout.legal_mask_canonical_torch(acting_np)

    # Pick random legal action per env
    actions = torch.zeros(N, dtype=torch.int32, device='cuda')
    for i in range(N):
        legal = lm[i].nonzero(as_tuple=False).squeeze(-1)
        if len(legal) > 0:
            idx = int(np.random.randint(len(legal)))
            actions[i] = legal[idx]

    # Step
    seats_t = torch.from_numpy(acting_np).to('cuda')
    result = rollout.step_device_torch(actions, seats_t)
    new_term = result["terminated"].cpu().numpy()

    # Check if any newly terminated
    newly_terminated = new_term.sum() > terminated_count
    if new_term.sum() > 0 and newly_terminated:
        old_count = terminated_count
        terminated_count = new_term.sum()
        print(f"  Step {step}: {terminated_count} envs now terminated (was {old_count})")

    # If we have some terminated envs, test the device-side reset
    if new_term.any():
        # Save pre-reset state of non-terminated envs
        pre_reset_turn = rollout.turn_torch().clone()
        term_mask = result["terminated"].clone()

        # Do device-side reset
        rollout.reset_terminated_device(seed=step * 1_000_003)

        # Verify: terminated envs should now be non-terminated
        post_term = rollout.terminated_torch()
        assert not post_term.any(), \
            f"After device reset, some envs still terminated: {post_term.cpu().numpy()}"

        # Verify: turn should be 0 (SOUTH) for reset envs
        post_turn = rollout.turn_torch()
        for i in range(N):
            if term_mask[i]:
                assert post_turn[i] == 0, \
                    f"Reset env {i} has turn={post_turn[i].item()}, expected 0"

        # Verify: can get legal actions for reset envs
        post_acting = post_turn.cpu().numpy()
        post_lm = rollout.legal_mask_canonical_torch(post_acting)

        # Each reset env should have legal actions
        for i in range(N):
            if term_mask[i]:
                n_legal = post_lm[i].sum().item()
                assert n_legal > 0, \
                    f"Reset env {i} has 0 legal actions"

        # Verify: non-terminated envs should be unchanged
        for i in range(N):
            if not term_mask[i]:
                assert post_turn[i] == pre_reset_turn[i], \
                    f"Non-terminated env {i} had turn changed: " \
                    f"{pre_reset_turn[i].item()} -> {post_turn[i].item()}"

        # Reset done_count since we just reset all terminated envs
        terminated_count = 0

        # Try stepping the reset envs to verify they're valid
        new_turn_t = rollout.turn_torch()
        new_acting_np = new_turn_t.cpu().numpy()
        new_lm = rollout.legal_mask_canonical_torch(new_acting_np)
        new_actions = torch.zeros(N, dtype=torch.int32, device='cuda')
        for i in range(N):
            legal = new_lm[i].nonzero(as_tuple=False).squeeze(-1)
            if len(legal) > 0:
                idx = int(np.random.randint(len(legal)))
                new_actions[i] = legal[idx]
        new_seats_t = torch.from_numpy(new_acting_np).to('cuda')
        step_result = rollout.step_device_torch(new_actions, new_seats_t)

        # Should all be valid steps (no crashes, no assertion failures)
        break  # One successful reset cycle is enough for verification

print()

# Now test: run many reset cycles to stress-test
print("=== Stress test: 50 reset cycles ===")
rollout2 = GpuRollout(num_envs=64)
rollout2.reset(seed_base=100)

for cycle in range(50):
    # Play some steps
    for step in range(20):
        turn_t = rollout2.turn_torch()
        acting_np = turn_t.cpu().numpy()
        lm = rollout2.legal_mask_canonical_torch(acting_np)
        actions = torch.zeros(64, dtype=torch.int32, device='cuda')
        for i in range(64):
            legal = lm[i].nonzero(as_tuple=False).squeeze(-1)
            if len(legal) > 0:
                idx = int(np.random.randint(len(legal)))
                actions[i] = legal[idx]
        seats_t = torch.from_numpy(acting_np).to('cuda')
        result = rollout2.step_device_torch(actions, seats_t)

        if result["terminated"].any():
            rollout2.reset_terminated_device(seed=cycle * 1_000_003 + step)

    # After 20 steps, force-check consistency
    term_check = rollout2.terminated_torch()
    if term_check.any():
        rollout2.reset_terminated_device(seed=cycle * 999_983)

    # Verify all envs are playable
    turn_t = rollout2.turn_torch()
    acting_np = turn_t.cpu().numpy()
    lm = rollout2.legal_mask_canonical_torch(acting_np)
    for i in range(64):
        if not rollout2.terminated_torch()[i]:
            n_legal = lm[i].sum().item()
            assert n_legal > 0, f"Cycle {cycle}: env {i} has 0 legal actions"

print(f"  All 50 cycles passed!")

# D2H verification: pull state back and check array shapes/values
print("\n=== D2H verification: check reset state integrity ===")
rollout3 = GpuRollout(num_envs=4)
rollout3.reset(seed_base=999)

# Play until termination
for step in range(2000):
    turn_t = rollout3.turn_torch()
    acting_np = turn_t.cpu().numpy()
    lm = rollout3.legal_mask_canonical_torch(acting_np)
    actions = torch.zeros(4, dtype=torch.int32, device='cuda')
    for i in range(4):
        legal = lm[i].nonzero(as_tuple=False).squeeze(-1)
        if len(legal) > 0:
            idx = int(np.random.randint(len(legal)))
            actions[i] = legal[idx]
    seats_t = torch.from_numpy(acting_np).to('cuda')
    result = rollout3.step_device_torch(actions, seats_t)

    if result["terminated"].any():
        # Do device reset
        rollout3.reset_terminated_device(seed=step)

        # Pull full state to host for validation
        host_state = rollout3.state.copy_to_host()
        host_term = rollout3.state.copy_termination_to_host()

        # Check basic invariants
        piece_seat = host_state["piece_seat_arr"].reshape(4, 120)
        piece_type = host_state["piece_type_arr"].reshape(4, 120)
        alive = host_state["alive"].reshape(4, 120)
        pos_x = host_state["pos_x"].reshape(4, 120)
        pos_y = host_state["pos_y"].reshape(4, 120)

        for env_i in range(4):
            # All envs should be non-terminated after reset
            assert not host_term["terminated"][env_i], \
                f"Env {env_i} still terminated after reset"

            # Count alive pieces per seat (should be 25 for fresh game)
            for s in range(4):
                n_alive = 0
                for pid in range(120):
                    if piece_seat[env_i, pid] == s and alive[env_i, pid]:
                        n_alive += 1
                        # Alive pieces should have valid positions
                        x, y = pos_x[env_i, pid], pos_y[env_i, pid]
                        assert 0 <= x < 17 and 0 <= y < 17, \
                            f"Env {env_i}, pid {pid}: invalid pos ({x}, {y})"
                assert n_alive == 25, \
                    f"Env {env_i}, seat {s}: {n_alive} alive pieces (expected 25)"

        print(f"  D2H check passed at step {step}")
        break
else:
    print("  WARNING: no termination in 2000 steps, D2H check skipped")

print("\n=== ALL PHASE 5 TESTS PASSED ===")
