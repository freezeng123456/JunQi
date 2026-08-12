"""junqi_rl.training.gpu_collector — GPU-native PPO rollout collector.

Feeds :class:`~junqi_rl.training.rollout.RolloutBuffer` from a
:class:`~junqi_rl.gpu_rollout.GpuRollout` in place of the CPU
:class:`~junqi_rl.env.VectorJunqiEnv`.

Design
------
PPO sees one transition per env-step (the acting seat's view).  The GPU
rollout:

  * mutates state on-device via ``GpuWorld.step_batch``
  * fetches dense legal-action ids (world frame) from the GPU kernel
  * builds per-seat observations on-device (a tensor of shape
    ``(N, 4, 101, 17, 17)``) and copies them to the host

Per step we only *use* the acting seat slice (``N × 101 × 17 × 17`` float32,
≈118 MB at N = 1024).  When torch is available we expose this to the policy
as a CUDA tensor via the existing host round-trip; future work can swap that
for ``torch.from_dlpack`` using :attr:`DeviceObservationBatch.d_spatial_ptr`.

Action-frame conventions
~~~~~~~~~~~~~~~~~~~~~~~~
* Policy → canonical-frame ``(0, 83520)`` flat ids.
* GPU step() consumes **world-frame** ids; we un-rotate via
  :data:`junqi_rl.action_lut.UNROTATE_LUT`.
* GPU ``legal_action_ids_batch`` returns **world-frame** ids; we rotate the
  per-env slice with :data:`junqi_rl.action_lut.ROTATE_LUT` to build the
  canonical-frame dense mask the policy expects.

Reward
~~~~~~
Reward is zero on non-terminal steps and ``team_rewards[acting_seat]``
(+1/0/-1) at termination — same contract as
:func:`junqi_rl.training.collector.collect_rollout`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from junqi_rl.action_lut import FLAT_ACTION_DIM, ROTATE_LUT, UNROTATE_LUT
from junqi_rl.gpu_rollout import GpuRollout

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from junqi_rl.networks.junqi_net import JunqiNet
    from junqi_rl.training.rollout import RolloutBuffer

try:
    import junqi_cuda as _cuda  # type: ignore[import]
except ImportError:
    _cuda = None  # type: ignore[assignment]


# Pre-stacked LUTs for vectorised per-env rotation (one 2-level fancy-index
# replaces a Python for-loop over envs).  Shape: (4, FLAT_ACTION_DIM) int32.
_ROTATE_LUT_STACK: np.ndarray = np.stack(ROTATE_LUT, axis=0).astype(np.int32)
_UNROTATE_LUT_STACK: np.ndarray = np.stack(UNROTATE_LUT, axis=0).astype(np.int32)

# Lazy torch mirror of _ROTATE_LUT_STACK — populated on first use for the
# GPU-resident legal-mask builder.  Stored as long (int64) so the 2-level
# fancy-index below works straight from torch.
_ROTATE_LUT_STACK_T = None  # type: ignore[assignment]


def _rotate_lut_stack_torch(device):
    global _ROTATE_LUT_STACK_T
    if _ROTATE_LUT_STACK_T is None or _ROTATE_LUT_STACK_T.device != device:
        _ROTATE_LUT_STACK_T = torch.from_numpy(
            _ROTATE_LUT_STACK.astype(np.int64)
        ).to(device)
    return _ROTATE_LUT_STACK_T


def build_legal_mask_batch_gpu_torch(
    rollout: GpuRollout,
    acting_seats: np.ndarray,  # (N,) int8
    terminated: np.ndarray,    # (N,) bool
    device,
):
    """GPU-resident legal mask construction (compact frame).

    The CUDA ``legal_action_ids_batch`` kernel returns WORLD-frame action ids
    (``src_flat*289 + dst_flat``).  We:
      1. Decompose each id into (src_flat, dst_flat).
      2. Map each via ``FLAT_TO_COMPACT`` to (src_compact, dst_compact).
      3. Recompose into a compact-world id = src_compact*129 + dst_compact.
      4. Rotate via the (compact) ROTATE_LUT to the canonical compact id.
      5. Scatter into the ``(N, FLAT_ACTION_DIM=16641)`` mask.

    Used by ``eval_vs_random`` (the training hot path goes through
    ``GpuRollout.legal_mask_canonical_torch_device`` instead, which runs
    entirely in CUDA).
    """
    from junqi_core.board import FLAT_TO_COMPACT, NUM_ON_BOARD_CELLS

    ids_np, counts_np = rollout.legal_actions_dense(acting_seats)
    # ids_np shape: (N, K) int32, K=512; values are world-frame src*289+dst.
    N, K = ids_np.shape

    ids_t       = torch.from_numpy(ids_np).to(device, dtype=torch.long)
    counts_t    = torch.from_numpy(np.asarray(counts_np, dtype=np.int64)).to(device)
    term_t      = torch.from_numpy(terminated).to(device)
    seats_t     = torch.from_numpy(acting_seats.astype(np.int64))\
                      .to(device)                                 # (N,)

    col_idx  = torch.arange(K, device=device, dtype=torch.long)   # (K,)
    valid    = (col_idx[None, :] < counts_t[:, None]) & \
               (~term_t)[:, None]                                 # (N, K)

    # Clamp invalid slots to 0 so subsequent indexing is in-bounds.
    safe_ids = torch.where(valid, ids_t, torch.zeros_like(ids_t))

    # World → compact: decompose into src/dst, map through FLAT_TO_COMPACT.
    flat_to_compact_t = torch.from_numpy(
        np.asarray(FLAT_TO_COMPACT, dtype=np.int64)
    ).to(device)                                                   # (289,)
    src_w = safe_ids // 289
    dst_w = safe_ids % 289
    src_c = flat_to_compact_t[src_w]
    dst_c = flat_to_compact_t[dst_w]
    # If either side is off-board (-1), mark invalid — shouldn't happen for
    # real legal actions but guards against garbage in padded slots.
    both_on_board = (src_c >= 0) & (dst_c >= 0)
    valid = valid & both_on_board
    compact_world = src_c.clamp(min=0) * NUM_ON_BOARD_CELLS + dst_c.clamp(min=0)
    # Clamp to LUT bounds for safety (invalid rows will be filtered below).
    compact_world = compact_world.clamp_(min=0, max=FLAT_ACTION_DIM - 1)

    lut_t    = _rotate_lut_stack_torch(device)                     # (4, 16641)
    can_ids  = lut_t[seats_t[:, None], compact_world]              # (N, K)
    # -1 in the LUT indicates no canonical mapping (shouldn't occur for
    # on-board actions); drop those.
    valid = valid & (can_ids >= 0)
    can_ids = can_ids.clamp_(min=0, max=FLAT_ACTION_DIM - 1)

    mask = torch.zeros((N, FLAT_ACTION_DIM), dtype=torch.bool, device=device)
    # Scatter via flat 1-D indices (per-row offset).
    row_offsets = torch.arange(N, device=device, dtype=torch.long) * FLAT_ACTION_DIM
    flat_idx = (row_offsets[:, None] + can_ids).reshape(-1)        # (N*K,)
    flat_valid = valid.reshape(-1)
    if flat_valid.any():
        mask.view(-1)[flat_idx[flat_valid]] = True
    return mask


# ---------------------------------------------------------------------------
# Team/seat reward helper
# ---------------------------------------------------------------------------

# Seat → team mapping: SOUTH(0)=team0, WEST(1)=team1, NORTH(2)=team0, EAST(3)=team1
_SEAT_TEAM = np.array([0, 1, 0, 1], dtype=np.int8)


def _per_seat_terminal_rewards(
    terminated: np.ndarray,   # (N,) bool
    winner_team: np.ndarray,  # (N,) int8 ∈ {-1, 0, 1}
    draw: np.ndarray,         # (N,) bool
    acting_seat: np.ndarray,  # (N,) int8 ∈ [0, 3]
) -> np.ndarray:
    """Return acting-seat reward: +1/-1 on terminal wins/losses, else 0."""
    N = terminated.shape[0]
    out = np.zeros(N, dtype=np.float32)
    fired = terminated & ~draw & (winner_team >= 0)
    if fired.any():
        seat_team = _SEAT_TEAM[acting_seat]
        won = fired & (seat_team == winner_team)
        lost = fired & (seat_team != winner_team)
        out[won] = 1.0
        out[lost] = -1.0
    return out


def _categorical_value_to_scalar(values):
    """Convert categorical log-prob values to their scalar expectation."""

    if values.dim() <= 1:
        return values
    probs = values.exp()
    if values.size(-1) == 3:
        # The fixed Junqi categorical value bins are [-1, 0, +1].
        # Avoid constructing and multiplying a bins tensor on every env step.
        return probs[..., 2] - probs[..., 0]
    bins = torch.linspace(
        -1.0,
        1.0,
        values.size(-1),
        device=values.device,
        dtype=probs.dtype,
    )
    return (probs * bins).sum(dim=-1)


# ---------------------------------------------------------------------------
# Legal-mask construction for GpuRollout
# ---------------------------------------------------------------------------

def build_legal_mask_batch_gpu(
    rollout: GpuRollout,
    acting_seats: np.ndarray,  # (N,) int8 (world-seat index, 0-3)
    terminated: np.ndarray,    # (N,) bool
) -> np.ndarray:
    """Build dense canonical-frame legal mask from the GPU kernel.

    The GPU returns WORLD-frame action ids (``src_flat*289 + dst_flat``) in
    an ``(N, 512)`` dense layout with a companion count array.  We convert
    each id to the compact 129×129 frame, rotate to canonical, and scatter
    into a ``(N, FLAT_ACTION_DIM=16641)`` bool mask.

    NOTE: The training hot path uses
    ``GpuRollout.legal_mask_canonical_torch_device`` instead, which runs
    entirely in CUDA.  This NumPy path is retained for ``eval_vs_random``
    and benchmarks.

    Returns
    -------
    mask : (N, FLAT_ACTION_DIM) bool
    """
    from junqi_core.board import FLAT_TO_COMPACT, NUM_ON_BOARD_CELLS

    ids, counts = rollout.legal_actions_dense(acting_seats)
    N, K = ids.shape                             # K = 512

    col_idx = np.arange(K, dtype=np.int32)[None, :]     # (1, K)
    valid = (col_idx < counts[:, None]) & (~terminated)[:, None]  # (N, K)

    # Clamp invalid entries' world_id to 0 so the lookup stays in range.
    safe_ids = np.where(valid, ids, 0).astype(np.int64, copy=False)

    # World → compact: decompose, map through FLAT_TO_COMPACT (shape (289,)).
    src_w = safe_ids // 289
    dst_w = safe_ids %  289
    src_c = np.asarray(FLAT_TO_COMPACT, dtype=np.int64)[src_w]
    dst_c = np.asarray(FLAT_TO_COMPACT, dtype=np.int64)[dst_w]
    both_on_board = (src_c >= 0) & (dst_c >= 0)
    valid = valid & both_on_board
    compact_world = np.clip(src_c, 0, None) * NUM_ON_BOARD_CELLS + np.clip(dst_c, 0, None)
    compact_world = np.clip(compact_world, 0, FLAT_ACTION_DIM - 1)

    # Rotate via the compact ROTATE_LUT.
    lut_stack = _ROTATE_LUT_STACK          # (4, 16641) int32
    seat_idx = acting_seats.astype(np.int64, copy=False)[:, None]   # (N, 1)
    can_ids = lut_stack[seat_idx, compact_world]      # (N, K) int32
    valid = valid & (can_ids >= 0)
    can_ids = np.clip(can_ids, 0, FLAT_ACTION_DIM - 1)

    # Scatter True into the dense mask.
    mask = np.zeros((N, FLAT_ACTION_DIM), dtype=bool)
    row_idx = np.repeat(np.arange(N, dtype=np.int32), K)  # (N*K,)
    flat_valid = valid.ravel()
    if flat_valid.any():
        mask[row_idx[flat_valid], can_ids.ravel()[flat_valid]] = True
    return mask


# ---------------------------------------------------------------------------
# collect_rollout_gpu
# ---------------------------------------------------------------------------

def collect_rollout_gpu(
    rollout_world: GpuRollout,
    policy: "JunqiNet",
    buffer: "RolloutBuffer",
    *,
    device: "str | torch.device" = "cpu",
    seed_base: int = 0,
    reset_at_start: bool = True,
    reward_shaping: bool = False,
) -> None:
    """Collect ``buffer.steps_per_env`` transitions using ``rollout_world``.

    Parameters
    ----------
    rollout_world
        Pre-constructed :class:`~junqi_rl.gpu_rollout.GpuRollout`.  Its
        ``num_envs`` must equal ``buffer.num_envs``.
    policy
        A :class:`~junqi_rl.networks.junqi_net.JunqiNet` in eval mode.  Its
        ``act`` method is called with
        ``(sp: FloatTensor[N,C,17,17], gl: FloatTensor[N,G], lm: BoolTensor[N,FLAT])``
        and is expected to return
        ``(actions_can: Int[N], log_probs: Float[N], values: Float[N])``.
    buffer
        Target :class:`RolloutBuffer`.  Populated in order, then
        ``compute_returns`` is called with zero-bootstrap for terminated
        envs and policy-value bootstrap otherwise.
    device
        Torch device for policy input tensors.
    seed_base
        Initial reset seed; after each terminated env, the next reset uses
        ``seed_base + step_counter * 1_000_003 + env_idx`` to diversify.
    reset_at_start
        If True, call ``rollout_world.reset(seed_base)`` before collecting.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for collect_rollout_gpu")
    if rollout_world.num_envs != buffer.num_envs:
        raise ValueError(
            f"num_envs mismatch: GpuRollout={rollout_world.num_envs} "
            f"buffer={buffer.num_envs}"
        )

    N = rollout_world.num_envs
    T = buffer.steps_per_env
    _device = torch.device(device)
    _zero_copy = _device.type == "cuda"
    # Detect RolloutBufferGPU (fully device-resident) — skip numpy downloads
    # of obs/mask/actions entirely when the buffer accepts tensors.
    _gpu_buffer = _zero_copy and hasattr(buffer, "obs_spatial") and \
                  isinstance(getattr(buffer, "obs_spatial", None), torch.Tensor)
    policy.eval()
    buffer.reset()

    if reset_at_start:
        rollout_world.reset(seed_base=seed_base)

    # Pre-allocate per-step index vector
    _env_range = np.arange(N, dtype=np.intp)

    # Pull current termination status (updated every step).
    term = rollout_world.read_termination()
    done_flags = term["terminated"].copy()

    step_counter = 0

    for t in range(T):
        # ---- Determine acting seat per env (fast D2H of only turn) ------
        turns = rollout_world.state.copy_turn_to_host()
        turns = np.asarray(turns, dtype=np.int8).reshape(N)
        # For terminated envs the turn is meaningless; clamp to 0 for safety.
        acting_seats = np.where(done_flags, np.int8(0), turns).astype(np.int8)

        # ---- Build obs and slice acting seat ----------------------------
        if _zero_copy:
            # Zero-copy: view device obs as torch CUDA tensor, slice on GPU.
            sp_full_t, gl_full_t = rollout_world.build_all_seat_observations_torch()
            env_idx_t = torch.arange(N, device=_device)
            acting_t = torch.from_numpy(acting_seats).to(
                _device, dtype=torch.long
            )
            act_sp_t = sp_full_t[env_idx_t, acting_t].contiguous()  # (N, C, 17, 17)
            act_gl_t = gl_full_t[env_idx_t, acting_t].contiguous()  # (N, G)
            if _gpu_buffer:
                # No D2H — the buffer accepts tensors directly.
                act_sp = act_gl = None
            else:
                act_sp = act_sp_t.detach().cpu().numpy()
                act_gl = act_gl_t.detach().cpu().numpy()
        else:
            spatial_full, global_full = rollout_world.build_all_seat_observations()
            act_sp = spatial_full[_env_range, acting_seats]
            act_gl = global_full[_env_range, acting_seats]

        # ---- Legal mask in canonical frame -----------------------------
        if _zero_copy:
            # Build the mask directly on the CUDA device — avoids a ~20 ms
            # host-side scatter and the subsequent H2D of an 83 MB tensor.
            lm_t = build_legal_mask_batch_gpu_torch(
                rollout_world, acting_seats, done_flags, _device,
            )
            if _gpu_buffer:
                legal_mask_np = None
            else:
                legal_mask_np = lm_t.cpu().numpy()
        else:
            legal_mask_np = build_legal_mask_batch_gpu(
                rollout_world, acting_seats, done_flags
            )
            lm_t = None

        # ---- Policy inference ------------------------------------------
        with torch.no_grad():
            if _zero_copy:
                sp_t = act_sp_t
                gl_t = act_gl_t
                # lm_t already on device
            else:
                sp_t = torch.from_numpy(act_sp).to(_device)
                gl_t = torch.from_numpy(act_gl).to(_device)
                lm_t = torch.from_numpy(legal_mask_np).to(_device)
            actions_can, log_probs, values = policy.act(sp_t, gl_t, lm_t)
        if _gpu_buffer:
            # Keep tensors resident; we only need CPU copies of actions to
            # feed the GPU step kernel.
            actions_can_np = actions_can.detach().cpu().numpy().astype(np.int32)
            log_probs_t = log_probs.detach().to(torch.float32)
            values_t = values.detach().to(torch.float32)
            if values_t.dim() > 1:
                values_t = values_t[:, 0]
        else:
            actions_can_np = actions_can.detach().cpu().numpy().astype(np.int32)
            log_probs_np = log_probs.detach().cpu().numpy().astype(np.float32)
            values_np = values.detach().cpu().numpy().astype(np.float32)
            if values_np.ndim > 1:
                values_np = values_np[:, 0]

        # ---- canonical → world for GPU step (vectorised) ---------------
        #   world[i] = UNROTATE_LUT[acting_seats[i]][actions_can[i]]
        seat_idx = acting_seats.astype(np.int64, copy=False)
        actions_world = _UNROTATE_LUT_STACK[
            seat_idx, actions_can_np.astype(np.int64, copy=False)
        ].astype(np.int32, copy=False)
        # Zero out terminated envs (their action_ids are garbage anyway).
        if done_flags.any():
            actions_world = np.where(done_flags, np.int32(0), actions_world)

        # ---- Store BEFORE step (reward/done patched afterwards) ---------
        if _gpu_buffer:
            buffer.add(
                obs_spatial=act_sp_t,
                obs_global=act_gl_t,
                legal_mask=lm_t,
                actions=actions_can.detach().to(torch.int32),
                log_probs=log_probs_t,
                values=values_t,
                rewards=torch.zeros(N, dtype=torch.float32, device=_device),
                dones=torch.from_numpy(done_flags).to(_device),
                seats=torch.from_numpy(acting_seats).to(_device, dtype=torch.int8),
            )
        else:
            buffer.add(
                obs_spatial=act_sp,
                obs_global=act_gl,
                legal_mask=legal_mask_np,
                actions=actions_can_np,
                log_probs=log_probs_np,
                values=values_np,
                rewards=np.zeros(N, dtype=np.float32),
                dones=done_flags.copy(),
                seats=acting_seats,
            )

        # ---- Step -------------------------------------------------------
        result = rollout_world.step(actions_world)
        # Newly-terminated envs fire their terminal reward *this* step.
        new_term = result["terminated"].astype(bool, copy=False)
        new_win = result["winner_team"].astype(np.int8, copy=False)
        new_draw = result["draw"].astype(bool, copy=False)

        # Transitions that crossed the terminal boundary this step.
        fired_this_step = new_term & ~done_flags
        rewards = _per_seat_terminal_rewards(
            terminated=fired_this_step,
            winner_team=new_win,
            draw=new_draw,
            acting_seat=acting_seats,
        )
        # --- Reward shaping: small per-step signal to densify credit ----
        # Event codes: 0=none, 1=move, 2=eat, 3=bomb (mutual kill),
        # 4=killed (attacker died into stronger defender).
        # Flag capture is already the terminal win — the +1 covers it.
        if reward_shaping:
            ev = result["event"].astype(np.int8, copy=False)
            flagc = result["flag_captured"].astype(bool, copy=False)
            # Live envs only (not terminated before the step and not dead here).
            live = ~done_flags
            # +0.05 for a successful eat (attacker killed defender cleanly)
            rewards = rewards + np.where(live & (ev == 2), 0.05, 0.0).astype(np.float32)
            # +0.02 for a move (very small — encourages progress over stalling)
            rewards = rewards + np.where(live & (ev == 1), 0.0, 0.0).astype(np.float32)  # off
            # -0.05 if *we* died (EV==4 = attacker died, so acting seat lost a piece)
            rewards = rewards + np.where(live & (ev == 4), -0.05, 0.0).astype(np.float32)
            # -0.02 for mutual kill (EV==3) — neutral-ish but acting seat still lost a piece
            rewards = rewards + np.where(live & (ev == 3), -0.02, 0.0).astype(np.float32)
        if _gpu_buffer:
            buffer.rewards[t] = torch.from_numpy(rewards).to(_device)
            buffer.dones[t] = torch.from_numpy(fired_this_step).to(_device)
        else:
            buffer.rewards[t] = rewards
            buffer.dones[t] = fired_this_step

        # ---- Auto-reset freshly-terminated envs so the next step is live
        if fired_this_step.any():
            seed_stride = step_counter * 1_000_003
            _reset_envs_inplace(
                rollout_world,
                reset_mask=fired_this_step,
                seed_base=seed_base + seed_stride,
            )
            # After reset the termination flags for those envs are cleared.
            term = rollout_world.read_termination()
            done_flags = term["terminated"].copy()
        else:
            done_flags = new_term.copy()

        step_counter += 1

    # ---- Bootstrap values for the last observed state -------------------
    turns = rollout_world.state.copy_turn_to_host()
    turns = np.asarray(turns, dtype=np.int8).reshape(N)
    acting_seats = np.where(done_flags, np.int8(0), turns).astype(np.int8)

    if _zero_copy:
        sp_full_t, gl_full_t = rollout_world.build_all_seat_observations_torch()
        env_idx_t = torch.arange(N, device=_device)
        acting_t = torch.from_numpy(acting_seats).to(_device, dtype=torch.long)
        sp_t = sp_full_t[env_idx_t, acting_t].contiguous()
        gl_t = gl_full_t[env_idx_t, acting_t].contiguous()
        lm_t = build_legal_mask_batch_gpu_torch(
            rollout_world, acting_seats, done_flags, _device,
        )
    else:
        spatial_full, global_full = rollout_world.build_all_seat_observations()
        act_sp = spatial_full[_env_range, acting_seats]
        act_gl = global_full[_env_range, acting_seats]
        sp_t = torch.from_numpy(act_sp).to(_device)
        gl_t = torch.from_numpy(act_gl).to(_device)
        legal_mask_np = build_legal_mask_batch_gpu(
            rollout_world, acting_seats, done_flags
        )
        lm_t = torch.from_numpy(legal_mask_np).to(_device)

    with torch.no_grad():
        _, _, last_values = policy.act(sp_t, gl_t, lm_t)

    last_values_np = last_values.detach().cpu().numpy().astype(np.float32)
    if last_values_np.ndim > 1:
        last_values_np = last_values_np[:, 0]
    # Terminated envs contribute zero bootstrap.
    last_values_np = last_values_np * (~done_flags).astype(np.float32)

    buffer.compute_returns(last_values_np)


# ---------------------------------------------------------------------------
# Reset helper (GPU)
# ---------------------------------------------------------------------------

def _reset_envs_inplace(
    rollout_world: GpuRollout,
    reset_mask: np.ndarray,   # (N,) bool
    seed_base: int,
) -> None:
    """Re-seed the envs selected by ``reset_mask`` by rebuilding the packed
    SoA for those slots and overwriting them on device.

    We build a fresh :class:`BatchedGameState` from ``reset_mask.sum()``
    CPU games, pack to the device SoA format, and overwrite only the
    affected rows in the device tensors via a full re-upload of the batch.

    (A per-slot device kernel would avoid the extra H2D; deferred to a
    later optimisation pass.)
    """
    from junqi_core.batched_state import BatchedGameState
    from junqi_core.setup import generate_random_setup
    from junqi_core.state import GameState
    from junqi_rl.gpu_rollout import _pack_from_batched

    N = rollout_world.num_envs
    # Pull current batch back to host.
    host = rollout_world.state.copy_to_host()
    term = rollout_world.state.copy_termination_to_host()

    # Build N fresh game states to reuse for the selected slots; cheap
    # because we only keep the ones we need.
    idxs = np.where(reset_mask)[0]
    fresh_states = [
        GameState.new_game(generate_random_setup(random.Random(int(seed_base) + int(i))))
        for i in idxs
    ]
    fresh_batch = BatchedGameState.from_game_states(fresh_states)
    fresh_dict = _pack_from_batched(fresh_batch)

    # Field schemas: (key, per-env stride, dtype).  We rewrite only
    # arrays that depend on the per-env game state.
    per_piece_120 = (
        "cell_piece_id_per_piece", "piece_seat_arr", "piece_type_arr",
        "alive", "pos_x", "pos_y", "zero_x", "zero_y",
        "move_count_arr", "active_eat_arr", "passive_surv_arr",
        "death_reason_arr", "death_step_arr", "death_loc_flat_arr",
    )
    per_cell_289 = ("cell_piece_id",)
    per_seat_4 = ("seat_dead_arr", "seat_flag_revealed_arr")
    per_env_1 = ("turn", "zobrist", "move_counter", "moves_since_last_combat")

    def _overwrite(dst: np.ndarray, src: np.ndarray, stride: int) -> None:
        # dst flat layout: (N * stride,) or (N,) — we rewrite idxs' rows.
        dst_view = dst.reshape(N, stride)
        src_view = src.reshape(len(idxs), stride)
        dst_view[idxs] = src_view

    for k in per_piece_120:
        _overwrite(host[k], fresh_dict[k], 120)
    for k in per_cell_289:
        _overwrite(host[k], fresh_dict[k], 289)
    for k in per_seat_4:
        _overwrite(host[k], fresh_dict[k], 4)
    for k in per_env_1:
        host_arr = host[k].reshape(N)
        src_arr = fresh_dict[k].reshape(len(idxs))
        host_arr[idxs] = src_arr

    rollout_world.state.copy_from_host(host)

    # Termination — fresh envs are non-terminal.
    term["terminated"][idxs] = False
    term["winner_team"][idxs] = -1
    term["draw"][idxs] = False
    rollout_world.state.copy_termination_from_host(
        term["terminated"].astype(bool),
        term["winner_team"].astype(np.int8),
        term["draw"].astype(bool),
    )


__all__ = [
    "build_legal_mask_batch_gpu",
    "collect_rollout_gpu",
    "collect_rollout_gpu_v2",
]


# ===========================================================================
# Phase 6: Zero-CPU collect loop
#
# Uses Phase 1-4 device-resident APIs.  The hot loop performs ZERO
# host-device transfers per step.  Only the episode-reset path (Phase 5
# pending) still falls back to CPU; all other operations stay on GPU.
# ===========================================================================


def collect_rollout_gpu_v2(
    rollout_world,
    policy,
    buffer,
    device="cuda",
    seed_base: int = 0,
    reset_at_start: bool = True,
    reward_shaping: bool = True,
    random_opponent: bool = False,
    on_termination=None,
    on_reset=None,
    use_compile: bool = True,
    autocast_dtype=None,  # torch.dtype; defaults to bfloat16 below
):
    """Zero-CPU hot-path rollout collector.

    Same semantics as :func:`collect_rollout_gpu` but uses device-resident
    legal mask, action rotation, step, and reward kernels (Phase 1-4).

    Parameters
    ----------
    random_opponent : bool
        If True, seats 1 (WEST) and 3 (EAST) play random legal actions
        instead of using the policy.  This matches the eval setup (policy
        controls team 0 = SOUTH+NORTH vs random team 1 = WEST+EAST) and
        prevents self-play overfitting.
    on_termination : callable or None
        Optional callback invoked once per step with signature
        ``fn(*, fired_t, rewards_t, acting_t, rollout_world)``. Used by the
        arrangement trainer to credit terminal rewards to the per-env
        arrangement in :class:`ArrangementBuffer`. Invoked BEFORE
        ``reset_terminated_device`` so the callback can snapshot the
        just-ended games before they're overwritten.
    on_reset : callable or None
        Optional callback invoked after ``reset_terminated_device`` with
        signature ``fn(*, fired_t, rollout_world)``. Used to refresh any
        host-side cache rows for envs that just received a new arrangement.

    Requirements:
      - buffer must be a RolloutBufferGPU (GPU-resident)
      - device must be "cuda"
    """
    import torch

    N = rollout_world.num_envs
    T = buffer.steps_per_env
    _device = torch.device(device)

    # Resolve autocast dtype. fp16 has only ±65504 dynamic range and CAN
    # produce NaN intermediates when self-play makes the policy confident
    # (extreme logits + value-head outputs overflow). v33c crashed at R107
    # exactly this way: fp16 collect produced NaN values/log_probs ->
    # buffer poisoned -> compute_returns spread NaN to all advantages ->
    # every PPO minibatch hit the grad-NaN guard -> training froze. bf16
    # has fp32-equivalent dynamic range (±3.4e38) so the same forward is
    # numerically robust. Default to bf16; caller may override (e.g.
    # fp16 if explicitly desired for older hardware without bf16 support).
    if autocast_dtype is None:
        autocast_dtype = torch.bfloat16

    policy.eval()
    buffer.reset()

    # O-1: torch.compile the full act() method for maximum cross-op fusion.
    # Gumbel-max sampling (rand + log + argmax + logsumexp) fuses well with
    # the preceding encode + policy_logits. Skipped when ``use_compile=False``
    # (e.g. torch 2.1's dynamo trips on BatchNorm decomposition under DDP +
    # bf16 autocast — see PROGRESS.md "torch.compile + fp16 autocast not
    # bit-reproducible") or when the caller's cfg.ppo.torch_compile is off.
    if use_compile and not getattr(policy, "_compiled_for_collect", False):
        try:
            # Save original act for eval (different batch size breaks CUDA graph)
            policy._orig_act = policy.act
            policy.act = torch.compile(policy.act, mode="reduce-overhead")
            policy._compiled_for_collect = True
        except Exception:
            # torch.compile not available or fails — proceed without
            policy._compiled_for_collect = True
    elif not use_compile:
        # Mark as "decided" so we don't keep retrying on every rollout.
        policy._compiled_for_collect = True

    if reset_at_start:
        rollout_world.reset(seed_base=seed_base)

    # Track done flags on GPU
    done_t = rollout_world.terminated_torch().clone()
    callbacks_enabled = on_termination is not None or on_reset is not None

    step_counter = 0

    for t in range(T):
        # ---- Turn (zero-copy GPU view) -----------------------------------
        turn_t = rollout_world.turn_torch()           # int8 (N,) CUDA
        # Clamp terminated envs' turn to 0
        acting_t = torch.where(done_t, torch.zeros_like(turn_t), turn_t)

        # ---- Observations (single-seat GPU kernel, 4x faster) ----------------
        obs_sp_t, obs_gl_t = rollout_world.build_acting_seat_observation_torch(acting_t)

        # ---- Legal mask (GPU kernel, canonical frame, zero CPU) -------------
        lm_t = rollout_world.legal_mask_canonical_torch_device(acting_t)  # (N, 83521) bool

        # ---- Policy inference (GPU autocast for Tensor Core speedup) --------
        # dtype is configurable (defaults to bf16; see resolution at function
        # entry). DO NOT hard-code fp16 here — see v33c R107 NaN root cause.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            actions_can, log_probs, values = policy.act(obs_sp_t, obs_gl_t, lm_t)
        log_probs_t = log_probs.detach().to(torch.float32)
        values_t = values.detach().to(torch.float32)
        values_t = _categorical_value_to_scalar(values_t)

        # ---- Random opponent: replace actions for enemy seats (1, 3) ---------
        if random_opponent:
            # Identify enemy-seat envs (team 1 = WEST, EAST)
            is_enemy = (acting_t == 1) | (acting_t == 3)
            if is_enemy.any():
                # Sample random legal actions for enemy envs via Gumbel-max
                # on uniform logits (all legal actions equally likely)
                uniform_logits = torch.where(lm_t, 0.0, float('-inf'))
                u = torch.rand_like(uniform_logits.float()).clamp_(1e-10, 1.0)
                gumbel = -torch.log(-torch.log(u))
                random_actions = (uniform_logits + gumbel).argmax(dim=-1).to(torch.int32)
                # Replace only enemy seats
                actions_can = torch.where(is_enemy, random_actions, actions_can.to(torch.int32))
                #
                # Audit note (2026-05-11, refreshed 2026-05-17 after F-4):
                #   For enemy seats we now have an ``(action, log_prob)``
                #   mismatch in the buffer — ``actions`` is the random
                #   action, but ``log_probs`` is still the policy's
                #   log-prob for the action it ORIGINALLY sampled
                #   (computed at line 718). PPO ratio = exp(new_lp - old_lp)
                #   would be wrong on these rows.
                #
                # Why this is still harmless under F-4
                # (``train_value_on_random_seats=True``):
                #   1. ``compute_returns`` zeros advantages on seats 1/3
                #      (in ``RolloutBufferGPU.compute_returns``), so
                #      ``own_mask = (seats==0)|(seats==2)`` selects only
                #      policy-active rows for the |adv|-quantile filter.
                #   2. F-4 *additionally* admits enemy-seat rows back into
                #      the minibatch as ``value_only_mask=True`` so that
                #      ``_value_loss`` can train V(s) on those states. But
                #      ``_policy_loss`` / ``_entropy_loss`` / ``kl_loss``
                #      all multiply per-sample contributions by
                #      ``policy_weight_per = (~value_only_mask).float()``
                #      ⇒ the stale ``old_log_prob`` on those rows multiplies
                #      a weight of 0 and never reaches a gradient.
                #   3. ``_value_loss`` reads ``values`` and ``returns``
                #      (not ``log_probs``), so the mismatch never bleeds
                #      into V either.
                #
                # Tested: ``tests/test_random_opponent_invariants.py``
                # pins down (1) compute_returns zeros enemy advantages,
                # (2) value_only_mask selects only enemy rows,
                # (3) ``_policy_loss(weight_per=~value_only_mask)`` is
                # invariant under arbitrary perturbations of value-only
                # rows, (4) self-play keeps every seat's signal alive.
                #
                # Future-change checklist: if you ever
                #   * read ``actions`` or ``old_log_probs`` from the buffer
                #     on a path that does NOT respect ``policy_weight_per``,
                #   * remove the ``policy_weight_per`` argument from
                #     ``_policy_loss`` / ``_entropy_loss`` / ``kl_loss``,
                #   * change ``own_mask`` to include enemy rows for the
                #     policy gradient,
                # then YOU MUST recompute the random action's log-prob
                # here (re-run ``policy.act`` or compute log_softmax over
                # the legal mask after the random sample). Otherwise the
                # PPO ratio on enemy rows becomes a non-stationary noise
                # source biased by the policy's confidence in actions it
                # never actually took.

        # ---- Store pre-step data in GPU buffer ----------------------------
        # Note (2026-05-11 audit): we considered zeroing values/log_probs for
        # envs that were ALREADY terminated at start of step t ("stale rows"),
        # on the theory that step_device_torch skips them and the obs we
        # computed is from a post-game state. Empirical instrumentation
        # (tools/_archive/audit/debug_collector_logic — N=16 T=2048 rollout)
        # showed stale row count is exactly 0: ``reset_terminated_device``
        # at the END of every step flips d_terminated back to False before
        # the next iteration reads it, so ``done_t`` is never True at the
        # top of the loop. The defensive zeroing is therefore a no-op and
        # was removed to keep the hot path simple. ``buffer.dones[t]`` is
        # set to ``fired_t`` (just-terminated this step) below, which is
        # exactly what GAE needs to truncate bootstrap.
        buffer.add(
            obs_spatial=obs_sp_t,
            obs_global=obs_gl_t,
            legal_mask=lm_t,
            actions=actions_can.detach().to(torch.int32),
            log_probs=log_probs_t,
            values=values_t,
            # Both fields are patched from the step result immediately below;
            # skip a redundant allocation/copy on every environment step.
            rewards=None,
            dones=None,
            seats=acting_t,
        )

        # ---- Step (GPU: rotate + step + reward, zero CPU) -----------------
        result = rollout_world.step_device_torch(
            actions_can.to(torch.int32),
            acting_t,
        )
        new_term_t = result["terminated"]
        rewards_t = result["rewards"]

        # Detect newly-terminated envs
        fired_t = new_term_t & ~done_t

        # ---- Reward shaping (densify credit for vs-random training) ------
        # The CUDA reward kernel emits PURE terminal ±1 (mirrors Ataraxos).
        # That is correct for self-play, where opponents punish stalling
        # naturally, but in vs-random it produces the "stall = +ret" trap
        # documented in PROGRESS.md (v30 R240 avg_len=1700).
        #
        # Iteration 1 (commit ?) added +0.05 / EAT bonus to encourage
        # combat. Empirically (200R run on T4) this caused classic reward
        # hacking: ret rose 0.07 -> 0.20 while win_rate stayed flat at
        # 0.36-0.55 because "kill more pieces" is decoupled from "win the
        # game" against a non-strategic opponent.
        #
        # Iteration 2 (current) keeps only piece-loss penalties — these
        # nudge the policy away from suicidal trades without inventing a
        # new optimisation target. Pieces are a means; victory is the end.
        #   -0.05  attacker died (event=4)  — into stronger defender
        #   -0.02  mutual BOMB   (event=3)  — neutral but lost a piece
        # Live-only: skip envs that were already terminated before the step
        # (their rewards are not added to the buffer anyway).
        if reward_shaping:
            ev_t = result["event"]                # int8 (N,) on cuda
            # BUG-J fix (2026-05-11): shaping must NOT stack on terminal
            # rewards. Old code used ``live_t = ~done_t`` which is the
            # "was live BEFORE this step" mask, but the step's terminal
            # ±1 reward and the shaping penalty would then BOTH fire on
            # the fired step (an env that was live, just acted, and
            # terminated this step). Empirically (T=4096 N=32 vs-random
            # rollout, 2026-05-11): 6/118 fired-step rewards were
            # ``+0.98 = +1.0 - 0.02`` — terminal win penalised by the
            # mutual-BOMB shaper. This cross-talks the two reward
            # channels (terminal: who won/lost; shaping: avoid suicidal
            # piece trades) and biases the policy to gold-protect its
            # last piece in the winning move.
            #
            # Correct mask: shape only on steps that do NOT terminate
            # the game and were live to begin with.
            #   live_after_step = ~new_term_t  (env still alive after step)
            #   live_t          = ~done_t      (env was alive before step)
            #   apply_shaping   = live_after_step & live_t
            # Equivalently: ``~(done_t | new_term_t)``.
            apply_shaping = (~done_t) & (~new_term_t)
            killed_pen = (apply_shaping & (ev_t == 4)).to(torch.float32) * (-0.05)
            mutual_pen = (apply_shaping & (ev_t == 3)).to(torch.float32) * (-0.02)
            rewards_t = rewards_t + killed_pen + mutual_pen

        # Patch rewards into the buffer
        buffer.rewards[t] = rewards_t
        buffer.dones[t] = fired_t

        # ---- Arrangement-net termination hook ----------------------------
        # Fires BEFORE reset_terminated_device so the callback can snapshot
        # the just-ended games' arrangements.
        # ``bool(cuda_tensor.any())`` synchronises the stream. The common
        # move-policy-only path has no callbacks, so avoid that sync entirely.
        fired_any = bool(fired_t.any()) if callbacks_enabled else False
        if on_termination is not None and fired_any:
            on_termination(
                fired_t=fired_t,
                rewards_t=rewards_t,
                acting_t=acting_t,
                rollout_world=rollout_world,
            )

        # ---- Update beliefs on-device based on step outcome ----------------
        # Must happen BEFORE reset_terminated_device: step_device_torch captured
        # the pre-step belief snapshot, while reset_terminated_device captures a
        # different pre-reset snapshot and overwrites reset envs with new games.
        # Updating after reset would apply the old move result to the new game.
        rollout_world.update_beliefs_device(result, acting_t)

        # ---- Auto-reset terminated envs (Phase 5: fully on GPU) ----------
        # The kernel is a no-op for non-terminated envs, so we call it
        # unconditionally. If any env reset, let the caller refresh host-side
        # caches (e.g. arrangement snapshots) after the new setup is installed.
        seed_stride = step_counter * 1_000_003
        rollout_world.reset_terminated_device(seed=seed_base + seed_stride)
        done_t = rollout_world.terminated_torch().clone()
        if on_reset is not None and fired_any:
            on_reset(fired_t=fired_t, rollout_world=rollout_world)

        step_counter += 1

    # ---- Bootstrap values for last state ----------------------------------
    turn_t = rollout_world.turn_torch()
    acting_t = torch.where(done_t, torch.zeros_like(turn_t), turn_t)
    obs_sp_t, obs_gl_t = rollout_world.build_acting_seat_observation_torch(acting_t)
    lm_t = rollout_world.legal_mask_canonical_torch_device(acting_t)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
        _, _, last_values = policy.act(obs_sp_t, obs_gl_t, lm_t)

    # Keep everything on-device; compute_returns accepts torch tensors.
    lv = last_values.detach().to(torch.float32)
    lv = _categorical_value_to_scalar(lv)
    lv = lv * (~done_t).to(torch.float32)

    # Pass the seat that would act next (the "T" boundary's seat) so the
    # buffer can correctly flip V_{T} when computing GAE for the last
    # stored transition. Without this the bootstrap value's perspective
    # is unknown and the very last delta can be biased.
    buffer.compute_returns(lv, last_seats=acting_t.to(torch.int64))
