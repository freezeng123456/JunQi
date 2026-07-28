"""junqi_rl.analysis.record — Record one game with a trained policy,
capturing per-step policy distributions for later viewing.

Produces a :class:`~junqi_core.replay_with_policy.TrajectoryWithPolicy`
file.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from junqi_core.board import COMPACT_TO_FLAT, FLAT_TO_COMPACT, NUM_ON_BOARD_CELLS
from junqi_core.replay_with_policy import (
    NUM_CELLS,
    NUM_TRACKED_TYPES,
    TrajectoryWithPolicy,
    probs_to_top_k,
)
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState
from junqi_rl.action_lut import ROTATE_LUT, UNROTATE_LUT
from junqi_rl.training.rollout import FLAT_ACTION_DIM

if TYPE_CHECKING:
    from junqi_rl.networks.junqi_net import JunqiNet


@torch.no_grad()
def record_game_with_policy(
    policy: JunqiNet,
    *,
    setups=None,
    rng_seed: int | None = None,
    first_seat: Seat = Seat.SOUTH,
    show_mode: ShowMode = ShowMode.HALF_DARK,
    device: str | torch.device = "cuda",
    top_k: int = 16,
    max_steps: int = 1000,
    greedy: bool = False,
    record_beliefs: bool = False,
    meta: dict[str, Any] | None = None,
) -> TrajectoryWithPolicy:
    """Play one game with ``policy`` and capture per-step data.

    Uses the **CPU** engine (:class:`GameState`) for simplicity — one game
    at a time, no batching.  Observations are built per step via
    :func:`build_observation_for_seat` and rotated to canonical frame so
    the network sees the correct orientation.

    Parameters
    ----------
    policy
        A trained :class:`JunqiNet` already ``.to(device)`` and ``.eval()``.
    greedy
        If True, take argmax of policy each step; otherwise sample.
    record_beliefs
        If True, store the per-seat rule-based belief tensor at each step.
    meta
        Optional JSON-serialisable training/checkpoint metadata (for example
        ``iteration``, ``checkpoint``, ``win_rate``).  It is persisted with
        the replay so progress dashboards can group games by run.
    """
    from junqi_core.observation import (
        build_observation,
    )

    rng = random.Random(rng_seed)
    recorded_seed = -1 if rng_seed is None else int(rng_seed)

    if setups is None:
        setups = generate_random_setup(rng)

    state = GameState.new_game(
        setups, show_mode=show_mode, first_seat=first_seat,
    )
    dev = torch.device(device)

    # Build initial belief tensors for every seat; updated after each move.
    from junqi_core.info_model import BeliefTensor
    beliefs = {
        s: BeliefTensor.initial(state, s, show_mode=show_mode)
        for s in ALL_SEATS
    }

    # Accumulators
    actions_rows: list[tuple[int, int, int, int, int]] = []
    top_ids_buf: list[np.ndarray] = []
    top_probs_buf: list[np.ndarray] = []
    values_buf: list[float] = []
    acting_buf: list[int] = []
    beliefs_buf: list[np.ndarray] = []  # only used if record_beliefs

    policy.eval()

    for _ in range(max_steps):
        if state.terminated:
            break
        seat = state.turn
        ids_world = state.legal_action_ids(seat)
        if ids_world.size == 0:
            break  # Engine will normally handle this via Q12 on next call.

        # Build canonical-frame obs for this seat.
        obs = build_observation(state, beliefs[seat], observer=seat)
        sp_np = obs.spatial.astype(np.float32, copy=False)
        gl_np = obs.global_.astype(np.float32, copy=False)

        sp_t = torch.from_numpy(sp_np).unsqueeze(0).to(dev)
        gl_t = torch.from_numpy(gl_np).unsqueeze(0).to(dev)

        # Rotate legal world ids → compact canonical, build mask.
        # (ROTATE_LUT / UNROTATE_LUT are indexed in the compact 129x129 frame
        # since commit 1826873; we must compact the world ids first.)
        src_w_arr = ids_world // NUM_CELLS
        dst_w_arr = ids_world %  NUM_CELLS
        src_c_arr = np.asarray(FLAT_TO_COMPACT, dtype=np.int64)[src_w_arr]
        dst_c_arr = np.asarray(FLAT_TO_COMPACT, dtype=np.int64)[dst_w_arr]
        both_on_board = (src_c_arr >= 0) & (dst_c_arr >= 0)
        compact_world = (np.clip(src_c_arr, 0, None) * NUM_ON_BOARD_CELLS
                          + np.clip(dst_c_arr, 0, None))[both_on_board]
        can_ids = ROTATE_LUT[seat.value][compact_world]
        can_ids = can_ids[can_ids >= 0]
        mask_np = np.zeros((FLAT_ACTION_DIM,), dtype=bool)
        mask_np[can_ids] = True
        lm_t = torch.from_numpy(mask_np).unsqueeze(0).to(dev)

        # Forward: we want the full log-prob distribution, not just the
        # sampled action.  Use .forward() which exposes log_probs.
        out = policy.forward(sp_t, gl_t, lm_t)
        log_probs = out["log_probs"].squeeze(0)      # (FLAT,)
        probs = log_probs.exp().cpu().numpy().astype(np.float32)

        chosen_can = int(probs.argmax()) if greedy else int(out["action"].item())

        # Convert compact canonical → compact world → world-full, decode src/dst.
        compact_world_chosen = int(UNROTATE_LUT[seat.value][chosen_can])
        src_c = compact_world_chosen // NUM_ON_BOARD_CELLS
        dst_c = compact_world_chosen %  NUM_ON_BOARD_CELLS
        src_flat = int(COMPACT_TO_FLAT[src_c])
        dst_flat = int(COMPACT_TO_FLAT[dst_c])
        sy, sx = divmod(src_flat, 17)
        dy, dx = divmod(dst_flat, 17)

        # Apply action.
        action = Action(seat=seat, src=(sx, sy), dst=(dx, dy))
        prev_state = state.clone()
        result = state.step_inplace(action)

        # Propagate beliefs for all seats.
        for s in ALL_SEATS:
            beliefs[s].update(prev_state, state, result)

        # Record (top-K in canonical frame first — un-rotate to world for
        # visualisation so users see moves in world coords).
        top_can_ids, top_can_probs = probs_to_top_k(probs, mask_np, k=top_k)
        top_world_ids = UNROTATE_LUT[seat.value][top_can_ids]
        top_ids_buf.append(top_world_ids.astype(np.int32))
        top_probs_buf.append(top_can_probs.astype(np.float32))

        value = float(out["value"].item())
        values_buf.append(value)
        acting_buf.append(int(seat.value))
        actions_rows.append((int(seat.value), sx, sy, dx, dy))

        if record_beliefs:
            # Persist the actual deterministic rule-based belief state.  The
            # previous placeholder wrote all-zero tensors, which made a
            # replay look as if the belief model had collapsed even when the
            # policy was healthy.
            per_seat = []
            for s in ALL_SEATS:
                world = beliefs[s].to_world_tensor()  # (17, 17, 12)
                per_seat.append(
                    np.transpose(world, (2, 0, 1)).reshape(NUM_TRACKED_TYPES, NUM_CELLS)
                )
            beliefs_buf.append(np.stack(per_seat, axis=0).astype(np.float16, copy=False))

    T = len(actions_rows)
    actions_arr = np.asarray(actions_rows, dtype=np.int16) if T else np.zeros((0, 5), dtype=np.int16)
    beliefs_arr = np.stack(beliefs_buf, axis=0) if (record_beliefs and T) else None

    replay_meta: dict[str, Any] = dict(meta or {})
    replay_meta.update({
        "source": "junqi_rl.analysis.record_game_with_policy",
        "policy_type": type(policy).__name__,
        "device": str(dev),
        "greedy": bool(greedy),
        "top_k": int(top_k),
        "record_beliefs": bool(record_beliefs),
    })

    traj = TrajectoryWithPolicy(
        setups=setups,
        actions=actions_arr,
        rng_seed=recorded_seed,
        final_state_hash=int(state.zobrist),
        first_seat=first_seat,
        show_mode=show_mode,
        top_action_ids=(
            np.stack(top_ids_buf, axis=0) if T else np.zeros((0, top_k), dtype=np.int32)
        ),
        top_probs=(
            np.stack(top_probs_buf, axis=0) if T else np.zeros((0, top_k), dtype=np.float32)
        ),
        values=np.asarray(values_buf, dtype=np.float32),
        acting_seats=np.asarray(acting_buf, dtype=np.int8),
        beliefs=beliefs_arr,
        meta=replay_meta,
    )
    return traj


__all__ = ["record_game_with_policy"]
