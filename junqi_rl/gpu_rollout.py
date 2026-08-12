"""junqi_rl/gpu_rollout.py — GPU-native batched Junqi rollout for PPO.

Phase 1b milestone #2 deliverable (ADR-112 PPO path).

Unlike :class:`VectorJunqiEnvGPU` (which still runs CPU game logic), this
class keeps EVERYTHING on-device after the initial seeding:

  * state mutates via ``GpuWorld.step_batch``
  * legal actions via ``legal_action_ids_batch`` (GPU kernel, host D2H kept
    small by the CSR variant)
  * observations via ``build_observation_batch_resident`` (beliefs live in
    persistent scratch)

PPO loop shape
--------------
::

    world = GpuRollout(num_envs=1024)
    world.reset(seed_base=0)
    for step in range(T):
        # Net forward on last observation (torch tensor, or numpy).
        action_ids = policy(world.obs_acting_spatial, world.obs_acting_global,
                            world.legal_mask)
        world.step(action_ids)                        # mutates state on-GPU
        buffer.append(world.result_dict())            # tiny D2H (per-env scalars)
    # Only at learner-update time do we pull observation blocks back for PPO loss.

Belief inputs
-------------
For the PPO warmup stage beliefs are assumed all-zero; for later milestones
the belief network's output will be uploaded via
:meth:`upload_beliefs` (does NOT touch the rollout's persistent scratch).

Observation coverage
--------------------
For speed we build observations ONLY for the acting seat of each env (PPO
makes its action choice from that seat's POV).  Users who need full per-seat
observations (e.g. for league-of-opponents cross-play) can call
:meth:`build_all_seat_observations`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _CUDA_AVAILABLE = True
except ImportError as _e:
    import sys as _sys
    print(f"[gpu_rollout] WARNING: junqi_cuda import failed: {_e}", file=_sys.stderr)
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

from junqi_core.batched_state import BatchedGameState
from junqi_core.board import (
    BOARD_SIZE,
    COMPACT_TO_FLAT,
    FLAT_TO_COMPACT,
    NUM_ON_BOARD_CELLS,
)
from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_core.rules import ShowMode, CAMP_INDICES, PieceType, SLOTS_PER_SEAT
from junqi_core.setup import generate_random_setup, generate_random_lineup
from junqi_core.setup_canonical import (
    CANONICAL_LINEUPS,
    generate_canonical_setup,
)
from junqi_core.state import GameState

from .gpu_world import _upload_zobrist_tables


NUM_TRACKED_TYPES = 12
FLAT_ACTION_DIM = 129 * 129  # 16641 — compact on-board action space

# Default setup pool size for device-side reset.
DEFAULT_POOL_SIZE = 10_000


class _CudaArrayInterfaceView:
    """Minimal CAI wrapper so :func:`torch.as_tensor` can claim a raw device
    pointer as a zero-copy tensor.

    We intentionally do NOT implement ``__dlpack__`` / ``__dlpack_device__``
    — torch's preferred path ``torch.from_dlpack`` requires a DLPack capsule
    which would need a matching C helper.  ``__cuda_array_interface__`` is
    universally supported and costs nothing.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape":   tuple(shape),
            "typestr": typestr,
            "data":    (int(ptr), False),  # read-write
            "version": 2,
        }


class GpuRolloutHistory:
    """Compact device history with train-time observation reconstruction."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        show_mode: ShowMode,
    ) -> None:
        if not hasattr(_cuda, "DeviceRolloutHistory"):
            raise RuntimeError(
                "junqi_cuda was built without compact rollout history support; "
                "rebuild it with `python3 build_cuda.py`"
            )
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.show_mode = show_mode
        self._impl = _cuda.DeviceRolloutHistory(
            self.num_steps,
            self.num_envs,
        )

    @property
    def history_bytes(self) -> int:
        return int(self._impl.history_bytes)

    def snapshot(
        self,
        state,
        acting_seats: "torch.Tensor",
        step: int,
    ) -> None:
        import torch

        if acting_seats.device.type != "cuda":
            raise ValueError("acting_seats must be a CUDA tensor")
        if acting_seats.dtype != torch.int8:
            acting_seats = acting_seats.to(torch.int8)
        if not acting_seats.is_contiguous():
            acting_seats = acting_seats.contiguous()
        if acting_seats.shape != (self.num_envs,):
            raise ValueError(
                f"acting_seats must have shape ({self.num_envs},), "
                f"got {tuple(acting_seats.shape)}"
            )
        self._impl.snapshot(
            state,
            acting_seats.data_ptr(),
            int(step),
        )

    def reconstruct(
        self,
        flat_indices: "torch.Tensor",
        acting_seats: "torch.Tensor",
        *,
        dtype: "torch.dtype",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Rebuild one PPO minibatch entirely from device history."""

        import torch

        if flat_indices.device.type != "cuda" or acting_seats.device.type != "cuda":
            raise ValueError("history indices and seats must be CUDA tensors")
        if flat_indices.dtype != torch.int64:
            flat_indices = flat_indices.to(torch.int64)
        if acting_seats.dtype != torch.int8:
            acting_seats = acting_seats.to(torch.int8)
        flat_indices = flat_indices.contiguous()
        acting_seats = acting_seats.contiguous()
        if flat_indices.ndim != 1 or acting_seats.shape != flat_indices.shape:
            raise ValueError("history indices and seats must be matching 1-D tensors")

        batch_size = int(flat_indices.numel())
        if batch_size <= 0:
            raise ValueError("cannot reconstruct an empty history minibatch")
        pointers = self._impl.reconstruct(
            flat_indices.data_ptr(),
            acting_seats.data_ptr(),
            batch_size,
            np.int8(self.show_mode.value),
        )
        if int(pointers["batch_size"]) != batch_size:
            raise RuntimeError("CUDA history returned an unexpected batch size")

        spatial = torch.as_tensor(
            _CudaArrayInterfaceView(
                pointers["d_spatial_ptr"],
                (batch_size, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                "<f4",
            ),
            device=flat_indices.device,
        ).to(dtype=dtype)
        global_ = torch.as_tensor(
            _CudaArrayInterfaceView(
                pointers["d_global_ptr"],
                (batch_size, OBS_GLOBAL_DIMS),
                "<f4",
            ),
            device=flat_indices.device,
        ).to(dtype=dtype)
        legal_mask = torch.as_tensor(
            _CudaArrayInterfaceView(
                pointers["d_legal_mask_ptr"],
                (batch_size, FLAT_ACTION_DIM),
                "|b1",
            ),
            device=flat_indices.device,
        )
        return spatial, global_, legal_mask


def _build_reset_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the constant tables for device-side reset.

    Returns
    -------
    pos_x     : (120,) int8 — world x for each piece_id at fresh state
    pos_y     : (120,) int8 — world y for each piece_id at fresh state
    piece_seat: (120,) int8 — seat assignment for each piece_id
    camp_slot : (30,)  bool — True for camp slot indices
    """
    from junqi_core.board import index_to_pos
    from junqi_core.rules import Seat

    pos_x = np.zeros(120, dtype=np.int8)
    pos_y = np.zeros(120, dtype=np.int8)
    piece_seat = np.zeros(120, dtype=np.int8)

    for seat_val in range(4):
        seat = Seat(seat_val)
        for slot in range(SLOTS_PER_SEAT):
            pid = seat_val * SLOTS_PER_SEAT + slot
            piece_seat[pid] = seat_val
            if slot in CAMP_INDICES:
                pos_x[pid] = -1
                pos_y[pid] = -1
            else:
                x, y = index_to_pos(seat, slot)
                pos_x[pid] = x
                pos_y[pid] = y

    camp_slot = np.array([i in CAMP_INDICES for i in range(SLOTS_PER_SEAT)],
                         dtype=bool)
    return pos_x, pos_y, piece_seat, camp_slot


def _build_setup_pool(pool_size: int, seed: int = 42) -> np.ndarray:
    """Pre-generate a pool of valid random setups.

    Returns
    -------
    piece_types : (pool_size, 120) int8
        For each pool entry, the piece type at each of the 120 piece slots
        (4 seats × 30 slots).  Camp slots carry PieceType.NONE (0).
    """
    rng = random.Random(seed)
    pool = np.zeros((pool_size, 120), dtype=np.int8)

    for i in range(pool_size):
        setup = generate_random_setup(rng)
        for seat_val in range(4):
            lineup = setup[seat_val]
            for slot in range(SLOTS_PER_SEAT):
                pid = seat_val * SLOTS_PER_SEAT + slot
                pool[i, pid] = int(lineup[slot])

    return pool


def _build_canonical_pool(
    pool_size: int,
    styles: tuple[str, ...] = ("T", "D", "G", "F"),
    seed: int = 42,
) -> np.ndarray:
    """Pre-generate a pool of CANONICAL (curriculum) setups.

    Each pool entry independently picks a style from ``styles`` and uses
    the same lineup at all 4 seats.  This is the starter pool for the
    fixed-setup curriculum (see ``junqi_core.setup_canonical``).

    Returns shape ``(pool_size, 120) int8``, identical layout to
    :func:`_build_setup_pool` so the existing GPU reset path needs no
    additional changes — just call ``upload_setup_pool`` on the result.
    """
    rng = random.Random(seed)
    pool = np.zeros((pool_size, 120), dtype=np.int8)
    style_list = list(styles)
    if not style_list:
        raise ValueError("styles must be non-empty")
    for i in range(pool_size):
        style = rng.choice(style_list)
        setup = generate_canonical_setup(style, same_for_all_seats=True)
        for seat_val in range(4):
            lineup = setup[seat_val]
            for slot in range(SLOTS_PER_SEAT):
                pid = seat_val * SLOTS_PER_SEAT + slot
                pool[i, pid] = int(lineup[slot])
    return pool


def _build_mixed_pool(
    pool_size: int,
    *,
    own_team_styles: tuple[str, ...] = ("T",),
    enemy_team_random: bool = True,
    enemy_team_styles: tuple[str, ...] = (),
    own_team_seats: tuple[int, ...] = (0, 2),  # SOUTH + NORTH = team 0
    seed: int = 42,
) -> np.ndarray:
    """Plan-B mixed pool: 我方固定布阵 + 敌方随机布阵.

    For every pool entry:

    * Our team's seats (``own_team_seats``, default {SOUTH=0, NORTH=2})
      get the SAME randomly-chosen canonical lineup from ``own_team_styles``.
      Both teammates share one lineup so the assistant network can rely
      on a stable prior for "where my team's pieces start".
    * The enemy team's two seats get **independent** lineups:
      - if ``enemy_team_random=True`` (default), each enemy seat gets a
        fresh uniform-random lineup via :func:`generate_random_lineup`;
      - else each enemy seat picks independently from
        ``enemy_team_styles``.
      "Independent" means seat WEST and seat EAST are sampled separately,
      so the model is forced to handle 2 distinct enemy formations per
      game — exactly the situation it'll face at evaluation against a
      uniform-random opponent.

    Why split this way (per the user's Plan B):

    1. Evaluation runs against UNIFORM RANDOM opponents, so training
       distribution should match.
    2. Same-布阵-on-all-4-seats (Plan A / v41) leaks information: under
       DARK rule, each piece's true type is "encoded" by its piece_id
       channel and the deterministic mapping piece_id → type is
       constant across games.  Network learns to look up rather than
       infer — exactly what we don't want for a 暗棋 task.

    Returns shape ``(pool_size, 120) int8`` (same layout as the other
    pool builders).
    """
    rng = random.Random(seed)
    pool = np.zeros((pool_size, 120), dtype=np.int8)
    own_set = set(own_team_seats)
    own_styles_list = list(own_team_styles)
    enemy_styles_list = list(enemy_team_styles)
    if not own_styles_list:
        raise ValueError("own_team_styles must be non-empty")
    if not enemy_team_random and not enemy_styles_list:
        raise ValueError(
            "enemy_team_styles must be non-empty when enemy_team_random=False"
        )

    for i in range(pool_size):
        # Pick one canonical lineup for OUR team (same for all own seats).
        own_style = rng.choice(own_styles_list)
        own_setup = generate_canonical_setup(own_style, same_for_all_seats=True)
        own_lineup = own_setup[0]   # all 4 seats identical → just take seat-0 entry
        for seat_val in range(4):
            if seat_val in own_set:
                # Our team — copy the shared canonical lineup.
                lineup = own_lineup
            else:
                # Enemy seat — independent draw.
                if enemy_team_random:
                    lineup = generate_random_lineup(rng)
                else:
                    enemy_style = rng.choice(enemy_styles_list)
                    enemy_setup = generate_canonical_setup(
                        enemy_style, same_for_all_seats=True,
                    )
                    lineup = enemy_setup[0]
            for slot in range(SLOTS_PER_SEAT):
                pid = seat_val * SLOTS_PER_SEAT + slot
                pool[i, pid] = int(lineup[slot])
    return pool


def _build_belief_prior_table() -> np.ndarray:
    """Build the 30×12 per-slot prior table for HALF_DARK enemy belief init.

    Returns shape (30, 12) float32.  Camp slots are all-zero.
    Matches :func:`junqi_core.info_model._per_slot_prior_vector`.
    """
    from junqi_core.rules import (
        PIECE_COUNTS, STRONGHOLD_INDICES, FRONT_ROW_INDICES,
        BACK_TWO_ROWS_INDICES, CAMP_INDICES,
    )
    from junqi_core.info_model import TRACKED_TYPES

    table = np.zeros((30, 12), dtype=np.float32)
    for slot in range(30):
        if slot in CAMP_INDICES:
            continue
        vec = np.zeros(12, dtype=np.float32)
        for ti, t in enumerate(TRACKED_TYPES):
            if slot in STRONGHOLD_INDICES:
                vec[ti] = float(PIECE_COUNTS[t])
            else:
                if t is PieceType.JUNQI:
                    continue
                if t is PieceType.DILEI and slot not in BACK_TWO_ROWS_INDICES:
                    continue
                if t is PieceType.ZHADAN and slot in FRONT_ROW_INDICES:
                    continue
                vec[ti] = float(PIECE_COUNTS[t])
        s = vec.sum()
        if s > 0:
            table[slot] = vec / s
    return table


def _build_seat_strongholds() -> np.ndarray:
    """Build the 4×2 stronghold world-frame flat positions.

    Returns shape (8,) int16: [seat0_sh0, seat0_sh1, seat1_sh0, ...].
    """
    from junqi_core.board import index_to_pos
    from junqi_core.rules import Seat, STRONGHOLD_INDICES

    sh_indices = sorted(STRONGHOLD_INDICES)  # [26, 28]
    result = np.zeros(8, dtype=np.int16)
    for s_val in range(4):
        seat = Seat(s_val)
        for k, sh_idx in enumerate(sh_indices):
            x, y = index_to_pos(seat, sh_idx)
            result[s_val * 2 + k] = np.int16(y * BOARD_SIZE + x)
    return result


# Module-level flag: have we uploaded the reset pool?
_reset_pool_uploaded = False


def _pack_from_batched(b: BatchedGameState) -> dict[str, np.ndarray]:
    """Flatten a BatchedGameState into the dict accepted by
    ``DeviceGameStateBatch.copy_from_host``."""
    N = b.num_envs
    flat = lambda a: a.reshape(-1)
    alv = flat(b.alive)
    px = flat(b.pos_x)
    py = flat(b.pos_y)
    cpip = np.where(
        alv,
        py.astype(np.int16) * np.int16(BOARD_SIZE) + px.astype(np.int16),
        np.int16(-1),
    ).astype(np.int16, copy=False)
    return {
        "cell_piece_id_per_piece":  cpip,
        "piece_seat_arr":           flat(b.piece_seat_arr).astype(np.int8),
        "piece_type_arr":           flat(b.piece_type_arr).astype(np.int8),
        "alive":                    alv.astype(bool),
        "pos_x":                    px.astype(np.int8),
        "pos_y":                    py.astype(np.int8),
        "zero_x":                   flat(b.zero_x).astype(np.int8),
        "zero_y":                   flat(b.zero_y).astype(np.int8),
        "move_count_arr":           flat(b.move_count_arr).astype(np.int16),
        "active_eat_arr":           flat(b.active_eat_arr).astype(np.int16),
        "passive_surv_arr":         flat(b.passive_surv_arr).astype(np.int16),
        "death_reason_arr":         flat(b.death_reason_arr).astype(np.int8),
        "death_step_arr":           flat(b.death_step_arr).astype(np.int16),
        "death_loc_flat_arr":       flat(b.death_loc_flat_arr).astype(np.int16),
        "cell_piece_id":            flat(b.cell_piece_id).astype(np.int16),
        "seat_dead_arr":            flat(b.seat_dead_arr).astype(bool),
        "seat_flag_revealed_arr":   flat(b.seat_flag_revealed_arr).astype(bool),
        "turn":                     b.turn.astype(np.int8),
        "zobrist":                  b.zobrist.astype(np.int64),
        "move_counter":             b.move_counter.astype(np.int32),
        "moves_since_last_combat":  b.moves_since_last_combat.astype(np.int32),
    }


class GpuRollout:
    """GPU-native rollout for PPO (Phase 1b).

    All game state lives on the device.  The only per-step host transfers are:

      * upload: ``action_ids`` (N int32)        — a few KB
      * download: ``MoveResultBatch`` scalars   — a few KB

    Observation arrays are pulled to host only when the learner asks (via
    :meth:`read_observations`); intermediate steps leave them device-resident.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        show_mode: ShowMode = ShowMode.DARK,
        device_id: int = 0,
        canonical_setup_styles: tuple[str, ...] | None = None,
        mixed_setup: bool = False,
        mixed_own_team_styles: tuple[str, ...] = ("T",),
    ) -> None:
        if not _CUDA_AVAILABLE:
            raise ImportError(
                "junqi_cuda extension is not available. "
                "Build it with: python3 build_cuda.py"
            )
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if _cuda.get_gpu_count() == 0:
            raise RuntimeError("No CUDA-capable GPU found")

        _cuda.set_device(device_id)
        _cuda.init_tables()
        _upload_zobrist_tables()

        self.num_envs = num_envs
        self.show_mode = show_mode
        self._device_id = device_id
        self._canonical_setup_styles = canonical_setup_styles
        self._mixed_setup = bool(mixed_setup)
        self._mixed_own_team_styles = tuple(mixed_own_team_styles)
        if self._mixed_setup and not self._mixed_own_team_styles:
            raise ValueError(
                "mixed_setup=True requires mixed_own_team_styles to be non-empty"
            )

        self.state = _cuda.DeviceGameStateBatch(num_envs)
        self.obs   = _cuda.DeviceObservationBatch(num_envs)
        # Single-seat obs buffer for acting-seat-only builds (4x less memory)
        self.obs_single = _cuda.DeviceObservationSingleBatch(num_envs)

        # Observer mapping: slot k observes the k-th seat (identity); users
        # override via :meth:`set_observer_seats` if they need a different
        # per-env mapping (e.g. league play).
        self._observer_seats = np.tile(
            np.arange(4, dtype=np.int8), (num_envs, 1)
        )
        _cuda.upload_observer_seats(num_envs, self._observer_seats)

        # Default beliefs are all zero — the belief network drops in later.
        # Phase 1: beliefs will be initialised to proper priors after the first
        # reset via init_beliefs_for_reset_envs().
        self._beliefs = np.zeros(
            (num_envs, 4, NUM_TRACKED_TYPES, BOARD_SIZE * BOARD_SIZE),
            dtype=np.float32,
        )
        _cuda.upload_beliefs(num_envs, self._beliefs)
        self._beliefs_enabled = hasattr(_cuda, "init_beliefs_for_reset_envs")
        self._last_step_ptrs: dict = {}  # raw device pointers from last step_device

        # Last step's MoveResult (for reward shaping, termination, etc).
        self._last_result: dict[str, np.ndarray] | None = None

        # Phase 5: upload device-side reset pool (once per process).
        # Three modes (mutually exclusive):
        #   1. mixed_setup=True   → Plan B: own team uses canonical lineup,
        #                           enemy team uses uniform-random.
        #   2. canonical_setup_styles → all 4 seats use the same canonical
        #                           lineup (Plan A / v41).
        #   3. neither            → uniform-random for all (legacy default).
        # All three pools have shape (pool_size, 120) int8 so the GPU
        # side is unaffected.
        self._ensure_reset_pool(
            canonical_styles=canonical_setup_styles,
            mixed_setup=self._mixed_setup,
            mixed_own_team_styles=self._mixed_own_team_styles,
        )

    def create_rollout_history(self, num_steps: int) -> GpuRolloutHistory:
        """Allocate compact pre-action state history for one PPO rollout."""

        return GpuRolloutHistory(
            num_steps=num_steps,
            num_envs=self.num_envs,
            show_mode=self.show_mode,
        )

    @staticmethod
    def _ensure_reset_pool(
        pool_size: int = DEFAULT_POOL_SIZE,
        *,
        canonical_styles: tuple[str, ...] | None = None,
        mixed_setup: bool = False,
        mixed_own_team_styles: tuple[str, ...] = ("T",),
    ) -> None:
        """Upload the device-side setup pool (once per process).

        Three mutually-exclusive modes:

        1. ``mixed_setup=True`` — Plan B mixed pool (our team canonical,
           enemy team uniform-random; see :func:`_build_mixed_pool`).
        2. ``canonical_styles=(..)`` — Plan A pool (4 seats share one
           canonical lineup per game; see :func:`_build_canonical_pool`).
        3. neither — legacy uniform-random for all seats.
        """
        global _reset_pool_uploaded
        if _reset_pool_uploaded:
            return

        # Upload reset tables (constant; same for all three modes).
        pos_x, pos_y, piece_seat, camp_slot = _build_reset_tables()
        _cuda.upload_reset_tables(pos_x, pos_y, piece_seat, camp_slot)
        # Upload compact cell mapping tables (for compact action space)
        _cuda.upload_compact_cell_maps(FLAT_TO_COMPACT, COMPACT_TO_FLAT)

        # Build setup pool — pick mode by precedence: mixed > canonical > random.
        if mixed_setup:
            unknown = [
                s for s in mixed_own_team_styles if s not in CANONICAL_LINEUPS
            ]
            if unknown:
                raise ValueError(
                    f"unknown mixed_own_team_styles {unknown}; valid: "
                    f"{sorted(CANONICAL_LINEUPS)}"
                )
            pool = _build_mixed_pool(
                pool_size,
                own_team_styles=tuple(mixed_own_team_styles),
                enemy_team_random=True,
                seed=42,
            )
        elif canonical_styles:
            unknown = [s for s in canonical_styles if s not in CANONICAL_LINEUPS]
            if unknown:
                raise ValueError(
                    f"unknown canonical setup styles {unknown}; valid: "
                    f"{sorted(CANONICAL_LINEUPS)}"
                )
            pool = _build_canonical_pool(
                pool_size, styles=tuple(canonical_styles), seed=42,
            )
        else:
            pool = _build_setup_pool(pool_size, seed=42)
        _cuda.upload_setup_pool(pool)
        # Phase 1 belief: upload prior table and stronghold positions
        if hasattr(_cuda, "upload_belief_prior_table"):
            prior_table = _build_belief_prior_table()
            _cuda.upload_belief_prior_table(prior_table.ravel())
            strongholds = _build_seat_strongholds()
            _cuda.upload_seat_strongholds(strongholds)
        _reset_pool_uploaded = True

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed_base: int = 0) -> None:
        """Build N fresh games, upload to device, zero the result buffer.

        Three setup modes (precedence: mixed > canonical > random):

        * ``mixed_setup=True`` — own team uses canonical lineup,
          enemy seats independent uniform-random.
        * ``canonical_setup_styles`` set — all 4 seats share one
          canonical lineup.
        * Otherwise — uniform-random for every seat.
        """
        if self._mixed_setup:
            from junqi_core.rules import Seat
            from junqi_core.setup import generate_random_lineup
            own_seats = {Seat.SOUTH.value, Seat.NORTH.value}
            own_styles = list(self._mixed_own_team_styles)
            states = []
            for i in range(self.num_envs):
                rng = random.Random(seed_base + i)
                style = rng.choice(own_styles)
                own_setup = generate_canonical_setup(
                    style, same_for_all_seats=True,
                )
                own_lineup = own_setup[0]
                lineups = [None] * 4
                for seat_val in range(4):
                    if seat_val in own_seats:
                        lineups[seat_val] = own_lineup
                    else:
                        lineups[seat_val] = generate_random_lineup(rng)
                states.append(GameState.new_game(tuple(lineups)))
        elif self._canonical_setup_styles:
            styles = list(self._canonical_setup_styles)
            states = []
            for i in range(self.num_envs):
                rng = random.Random(seed_base + i)
                style = rng.choice(styles)
                setup = generate_canonical_setup(style, same_for_all_seats=True)
                states.append(GameState.new_game(setup))
        else:
            states = [
                GameState.new_game(generate_random_setup(random.Random(seed_base + i)))
                for i in range(self.num_envs)
            ]
        b = BatchedGameState.from_game_states(states)
        self.state.copy_from_host(_pack_from_batched(b))
        self.state.copy_termination_from_host(
            b.terminated.astype(bool),
            b.winner_team.astype(np.int8),
            b.draw.astype(bool),
        )
        self._last_result = None
        # Phase 1: init beliefs to proper priors after state upload
        if self._beliefs_enabled:
            _cuda.init_all_beliefs(self.state)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Advance every env by one action.  Returns the MoveResultBatch dict.

        Mutates GPU state in place.  Terminated envs are skipped.
        """
        if action_ids.shape != (self.num_envs,):
            raise ValueError(
                f"action_ids shape mismatch: expected ({self.num_envs},), "
                f"got {action_ids.shape}"
            )
        if action_ids.dtype != np.int32:
            action_ids = action_ids.astype(np.int32, copy=False)
        self._last_result = self.state.step_batch(action_ids)
        return self._last_result

    def step_device_torch(
        self,
        canonical_actions: "torch.Tensor",  # (N,) int32 CUDA
        acting_seats: "torch.Tensor",       # (N,) int8 CUDA
    ) -> dict[str, "torch.Tensor"]:
        """Device-resident step: rotate + step + reward, zero CPU involvement.

        Parameters
        ----------
        canonical_actions : torch.cuda.IntTensor (N,)
            Actions in canonical frame (src_can * 289 + dst_can).
        acting_seats : torch.cuda.ByteTensor or int8 (N,)
            Per-env acting seat value (0-3).

        Returns
        -------
        dict with keys:
            terminated : torch.cuda.BoolTensor (N,)
            rewards    : torch.cuda.FloatTensor (N,)
            event      : torch.cuda.IntTensor (N,) — int8
            draw       : torch.cuda.BoolTensor (N,)
            winner_team: torch.cuda.IntTensor (N,) — int8
        All tensors are device-resident (zero D2H).
        """
        import torch
        # Phase 1 belief: snapshot pre-step flags for change detection
        if self._beliefs_enabled:
            _cuda.snapshot_pre_step_flags(self.state)

        # Get raw device pointers from torch tensors
        d_acts_ptr = canonical_actions.data_ptr()
        d_seats_ptr = acting_seats.data_ptr()

        result_ptrs = _cuda.step_device(self.state, d_acts_ptr, d_seats_ptr)
        N = result_ptrs["N"]

        # Record world-frame actions into the move history ring buffer.
        _cuda.record_move_history(
            self.state,
            result_ptrs["d_world_actions_ptr"],
            result_ptrs["d_valid_ptr"],
        )

        # Save raw pointer dict for belief update (before wrapping as tensors)
        self._last_step_ptrs = result_ptrs

        # Wrap device pointers as torch tensors (zero-copy)
        return {
            "terminated": torch.as_tensor(
                _CudaArrayInterfaceView(result_ptrs["d_terminated_ptr"], (N,), "|b1"),
                device="cuda",
            ),
            "rewards": torch.as_tensor(
                _CudaArrayInterfaceView(result_ptrs["d_rewards_ptr"], (N,), "<f4"),
                device="cuda",
            ),
            "event": torch.as_tensor(
                _CudaArrayInterfaceView(result_ptrs["d_event_ptr"], (N,), "|i1"),
                device="cuda",
            ),
            "draw": torch.as_tensor(
                _CudaArrayInterfaceView(result_ptrs["d_draw_ptr"], (N,), "|b1"),
                device="cuda",
            ),
            "winner_team": torch.as_tensor(
                _CudaArrayInterfaceView(result_ptrs["d_winner_team_ptr"], (N,), "|i1"),
                device="cuda",
            ),
        }

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def legal_actions_dense(
        self, acting_seats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (action_ids, counts) with shape ``(N, 512)`` and ``(N,)``."""
        if acting_seats.dtype != np.int8:
            acting_seats = acting_seats.astype(np.int8, copy=False)
        return _cuda.legal_action_ids_batch(self.state, acting_seats)

    def legal_actions_mask(self, acting_seats: np.ndarray) -> np.ndarray:
        """Return per-piece slot mask ``(N, 120, 80)`` bool.

        Multiply this directly into the policy network's action logits.
        """
        if acting_seats.dtype != np.int8:
            acting_seats = acting_seats.astype(np.int8, copy=False)
        return _cuda.legal_action_mask_batch(self.state, acting_seats)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def build_all_seat_observations(self) -> tuple[np.ndarray, np.ndarray]:
        """Build observations for all 4 observer seats; copy to host.

        Returns
        -------
        spatial : ndarray (N, 4, OBS_CHANNELS, 17, 17) float32
        global_ : ndarray (N, 4, 28)          float32
        """
        _cuda.build_observation_batch_resident(
            self.state, self.obs, np.int8(self.show_mode.value),
        )
        return self.obs.copy_to_host()

    def build_all_seat_observations_torch(self):
        """Build observations on-device and expose them as torch CUDA tensors.

        **Zero-copy**: the returned tensors wrap the device buffers owned by
        :attr:`obs`.  They are invalidated the next time this method (or
        :meth:`build_all_seat_observations`) is called, because the kernel
        overwrites the same buffers.  Callers who need a stable copy should
        ``.clone()`` or keep the tensors out of harm's way.

        Returns
        -------
        spatial : torch.cuda.FloatTensor (N, 4, OBS_CHANNELS, 17, 17)
        global_ : torch.cuda.FloatTensor (N, 4, 28)

        Raises
        ------
        RuntimeError
            If torch is not importable.
        """
        try:
            import torch  # type: ignore[import]
        except ImportError as e:  # pragma: no cover - torch missing
            raise RuntimeError("torch is required for to_torch()") from e

        _cuda.build_observation_batch_resident(
            self.state, self.obs, np.int8(self.show_mode.value),
        )
        N = self.num_envs
        sp_view = _CudaArrayInterfaceView(
            self.obs.d_spatial_ptr, (N, 4, OBS_CHANNELS, 17, 17), "<f4"
        )
        gl_view = _CudaArrayInterfaceView(
            self.obs.d_global_ptr, (N, 4, 28), "<f4"
        )
        return (
            torch.as_tensor(sp_view, device="cuda"),
            torch.as_tensor(gl_view, device="cuda"),
        )

    def build_acting_seat_observation_torch(
        self, acting_seats_t: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Build observation for ONE seat per env (the acting seat).

        **4x faster** than :meth:`build_all_seat_observations_torch` + slicing
        because the CUDA kernel only runs for 1 seat instead of 4, and the
        output is already contiguous (N, 256, 17, 17) — no ``.contiguous()``
        copy needed.

        Parameters
        ----------
        acting_seats_t : torch.cuda.CharTensor (N,)
            Per-env acting seat value (0-3), device-resident int8.

        Returns
        -------
        spatial : torch.cuda.FloatTensor (N, OBS_CHANNELS, 17, 17)
        global_ : torch.cuda.FloatTensor (N, 28)
        """
        import torch

        _cuda.build_observation_single_seat(
            self.state, self.obs_single,
            acting_seats_t.data_ptr(),
            np.int8(self.show_mode.value),
        )
        N = self.num_envs
        sp_view = _CudaArrayInterfaceView(
            self.obs_single.d_spatial_ptr, (N, OBS_CHANNELS, 17, 17), "<f4"
        )
        gl_view = _CudaArrayInterfaceView(
            self.obs_single.d_global_ptr, (N, 28), "<f4"
        )
        return (
            torch.as_tensor(sp_view, device="cuda"),
            torch.as_tensor(gl_view, device="cuda"),
        )

    def upload_beliefs(self, beliefs: np.ndarray) -> None:
        """Replace the device-resident belief buffer.

        beliefs shape: ``(N, 4, 12, 289)`` float32.  One-shot H2D into
        persistent GpuScratch; subsequent ``build_all_seat_observations``
        calls will read the new values.
        """
        if beliefs.shape != (self.num_envs, 4, NUM_TRACKED_TYPES, BOARD_SIZE * BOARD_SIZE):
            raise ValueError(
                f"beliefs shape must be ({self.num_envs}, 4, 12, 289), "
                f"got {beliefs.shape}"
            )
        if beliefs.dtype != np.float32:
            beliefs = beliefs.astype(np.float32, copy=False)
        self._beliefs = beliefs
        _cuda.upload_beliefs(self.num_envs, beliefs)

    def set_observer_seats(self, observer_seats: np.ndarray) -> None:
        """Reassign the per-env observer mapping ``(N, 4)`` int8.

        Useful for league-play where each slot corresponds to a different
        trained agent.  Uploads to persistent GpuScratch (one-shot H2D).
        """
        if observer_seats.shape != (self.num_envs, 4):
            raise ValueError(
                f"observer_seats shape must be ({self.num_envs}, 4), "
                f"got {observer_seats.shape}"
            )
        if observer_seats.dtype != np.int8:
            observer_seats = observer_seats.astype(np.int8, copy=False)
        self._observer_seats = observer_seats
        _cuda.upload_observer_seats(self.num_envs, observer_seats)

    # ------------------------------------------------------------------
    # Termination / bookkeeping
    # ------------------------------------------------------------------

    def read_termination(self) -> dict[str, np.ndarray]:
        """Pull terminated / winner_team / draw arrays from device."""
        return self.state.copy_termination_to_host()

    def reset_terminated_device(self, seed: int) -> None:
        """Reset all terminated envs on GPU using the pre-computed setup pool.

        Zero CPU involvement — the CUDA kernel picks random setups from
        the pre-uploaded pool and overwrites the SoA arrays in-place.

        Parameters
        ----------
        seed : int
            Combined with env index inside the kernel for hash-based pool
            entry selection.  Different seeds yield different reset patterns.
        """
        # Phase 1 belief: snapshot which envs are terminated (about to reset)
        # so we can re-init their beliefs after the reset kernel.
        if self._beliefs_enabled:
            _cuda.snapshot_pre_step_flags(self.state)  # saves seat flags + terminated
        _cuda.reset_terminated_envs(self.state, seed)
        # Phase 1: re-init beliefs for envs that were just reset.
        # The pre-step snapshot captured d_terminated=true for envs about to reset.
        # init_beliefs_for_reset_envs uses that saved mask.
        if self._beliefs_enabled:
            _cuda.init_beliefs_for_reset_envs(self.state)

    # ------------------------------------------------------------------
    # P0.4 — ArrangementNet setup pool management
    # ------------------------------------------------------------------

    def refresh_setup_pool_from_arrangements(
        self,
        samples: "torch.Tensor",   # (n_arr, 30, 13) one-hot
        seat_idx: "torch.Tensor",  # (n_arr,) int64
    ) -> int:
        """Re-upload the GPU setup pool from arrangement-net samples.

        Subsequent ``reset_terminated_device`` calls will seed fresh games
        from the new distribution (hash-based pool selection in the CUDA
        kernel still applies — each env picks one of the new pool entries
        uniformly at random across ``reset_terminated_envs`` invocations).

        Parameters
        ----------
        samples
            ``(n_arr, 30, 13)`` one-hot from
            :func:`junqi_rl.arrangement.sampling.generate_arrangements`.
        seat_idx
            ``(n_arr,)`` int64 seat index for each sample. ``n_arr`` must
            have at least 1 sample per seat; pool_size = min across seats.

        Returns
        -------
        pool_size : int
            Number of combined-setup entries in the new pool.
        """
        from junqi_rl.arrangement.pool_upload import (
            arrangements_to_pool, refresh_gpu_setup_pool,
        )
        pool = arrangements_to_pool(samples, seat_idx)
        refresh_gpu_setup_pool(pool)
        return int(pool.shape[0])

    def snapshot_env_arrangements(self) -> "np.ndarray":
        """Read back each env's current (4 seats × 30 slots) piece-type array.

        Returns
        -------
        per_env_per_seat : ``(N, 4, 30)`` int64 — vocab indices (0..12).
            Suitable for passing into ``ArrangementBuffer.add_rewards``
            after indexing the acting seat.

        Notes
        -----
        This is a D2H copy. Cache across multiple ``add_rewards`` calls
        within one rollout loop iteration.
        """
        from junqi_rl.arrangement.pool_upload import (
            read_env_arrangements_from_state,
        )
        return read_env_arrangements_from_state(self)


    def update_beliefs_device(
        self,
        step_result: dict,
        acting_seats: "torch.Tensor",
    ) -> None:
        """Update device-resident beliefs based on step outcome.

        Called after step_device_torch() and reset_terminated_device(), but before
        the next observation build. This allows beliefs to be updated incrementally
        on-device without CPU intervention.

        Parameters
        ----------
        step_result : dict[str, torch.Tensor]
            Output from step_device_torch() containing device pointer dicts.
            Must include "event", "terminated", "d_flag_captured_ptr",
            "d_world_actions_ptr".
        acting_seats : torch.Tensor
            (N,) int8 CUDA tensor with the acting seat for each env.

        Notes
        -----
        Applies deductive inference rules (R1, R4, R5/R7, R6, R9, I5)
        entirely on GPU.  Zero CPU/H2D involved.
        """
        if not self._beliefs_enabled:
            return

        # Get raw device pointers from step_result.
        # step_result stores torch tensors that wrap persistent device buffers.
        d_events_ptr = step_result["event"].data_ptr()

        # d_flag_captured_ptr and d_world_actions_ptr are stored as raw ints
        # in the _last_step_ptrs dict set by step_device_torch.
        d_fc_ptr = self._last_step_ptrs.get("d_flag_captured_ptr", 0)
        d_wa_ptr = self._last_step_ptrs.get("d_world_actions_ptr", 0)

        if d_fc_ptr == 0 or d_wa_ptr == 0:
            return  # Pointers not available — skip

        try:
            _cuda.update_beliefs_after_step(
                self.state,
                d_events_ptr,
                d_fc_ptr,
                d_wa_ptr,
            )
        except Exception as e:
            # Log but don't crash — belief failure shouldn't stop training
            import warnings
            warnings.warn(f"update_beliefs_after_step failed: {e}", stacklevel=2)

    # ------------------------------------------------------------------
    # Phase 2: Zero-copy device tensor views (no D2H)
    # ------------------------------------------------------------------

    def turn_torch(self) -> "torch.Tensor":
        """Zero-copy int8 (N,) CUDA tensor of the ``turn`` array.

        The returned tensor shares device memory with the engine; it is
        invalidated by any call that mutates game state (``step``, ``reset``).
        """
        import torch
        view = _CudaArrayInterfaceView(
            self.state.d_turn_ptr, (self.num_envs,), "|i1"
        )
        return torch.as_tensor(view, device="cuda")

    def terminated_torch(self) -> "torch.Tensor":
        """Zero-copy bool (N,) CUDA tensor of terminated flags."""
        import torch
        view = _CudaArrayInterfaceView(
            self.state.d_terminated_ptr, (self.num_envs,), "|b1"
        )
        return torch.as_tensor(view, device="cuda")

    def winner_team_torch(self) -> "torch.Tensor":
        """Zero-copy int8 (N,) CUDA tensor of winner_team."""
        import torch
        view = _CudaArrayInterfaceView(
            self.state.d_winner_team_ptr, (self.num_envs,), "|i1"
        )
        return torch.as_tensor(view, device="cuda")

    def draw_torch(self) -> "torch.Tensor":
        """Zero-copy bool (N,) CUDA tensor of draw flags."""
        import torch
        view = _CudaArrayInterfaceView(
            self.state.d_draw_ptr, (self.num_envs,), "|b1"
        )
        return torch.as_tensor(view, device="cuda")

    def legal_mask_canonical_torch(
        self, acting_seats: np.ndarray | None = None
    ) -> "torch.Tensor":
        """Dense (N, FLAT_ACTION_DIM) bool CUDA tensor of legal actions in
        canonical frame.  **Zero D2H/H2D in the hot path** — the CUDA kernel
        writes directly to device memory and we wrap it with torch.

        This replaces the old path:
          legal_actions_dense → D2H → scatter → H2D → torch.
        """
        import torch
        if acting_seats is None:
            # Use device turn directly
            acting_seats = np.asarray(
                self.state.copy_turn_to_host(), dtype=np.int8
            ).reshape(self.num_envs)
        if acting_seats.dtype != np.int8:
            acting_seats = acting_seats.astype(np.int8, copy=False)

        d_mask_ptr = _cuda.legal_mask_canonical_batch(self.state, acting_seats)
        view = _CudaArrayInterfaceView(
            d_mask_ptr, (self.num_envs, FLAT_ACTION_DIM), "|b1"
        )
        return torch.as_tensor(view, device="cuda")

    def legal_mask_canonical_torch_device(
        self, acting_seats_t: "torch.Tensor"
    ) -> "torch.Tensor":
        """Like :meth:`legal_mask_canonical_torch` but takes a CUDA int8 tensor
        for acting_seats — **zero H2D** in the hot path.

        Parameters
        ----------
        acting_seats_t : torch.cuda.CharTensor (N,)
            Per-env acting seat value (0-3), device-resident.
        """
        import torch
        d_mask_ptr = _cuda.legal_mask_canonical_batch_from_device(
            self.state, acting_seats_t.data_ptr()
        )
        view = _CudaArrayInterfaceView(
            d_mask_ptr, (self.num_envs, FLAT_ACTION_DIM), "|b1"
        )
        return torch.as_tensor(view, device="cuda")

    @property
    def last_result(self) -> dict[str, np.ndarray] | None:
        """Most recent ``step()`` MoveResult, or None if no step has run."""
        return self._last_result
