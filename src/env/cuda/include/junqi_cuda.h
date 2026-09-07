/*
 * junqi_cuda.h
 * Public C++ API for the JunQi CUDA backend.
 */

#pragma once

#include <cstdint>
#include <tuple>

namespace junqi_cuda {

constexpr int NUM_ENVS_MAX = 4096;
constexpr int NUM_PIECES = 120;
constexpr int NUM_CELLS = 289;
constexpr int BOARD_SIZE = 17;
constexpr int NUM_SEATS = 4;
constexpr int NUM_OBS_CHANNELS = 317;  // 161 base + 50 v4 tail + 46 v5 layer-3 + 60 v6 layer-4
constexpr int NUM_GLOBAL_DIMS = 28;
// Compact on-board action space: only 129 on-board cells (of 289 total)
constexpr int NUM_ON_BOARD = 129;
constexpr int FLAT_ACTION_SPACE = NUM_ON_BOARD * NUM_ON_BOARD;  // 16,641
// Legacy full action space (289×289) used by step_batch_kernel internally
constexpr int FLAT_ACTION_SPACE_FULL = NUM_CELLS * NUM_CELLS;   // 83,521
// 12 tracked piece types: JUNQI..GONGB (same as Python TRACKED_TYPES)
constexpr int NUM_TRACKED_TYPES = 12;
// Move history ring buffer depth (src_dst_planes)
constexpr int MOVE_HISTORY_LEN = 32;
// Belief tensor per-env layout: [NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS]
constexpr int BELIEF_STRIDE = NUM_SEATS * NUM_TRACKED_TYPES * NUM_CELLS;

// CombatMemory constants (must match Python junqi_core.combat_memory).
constexpr int CM_NUM_OBSERVERS = 4;
constexpr int CM_NUM_PIDS = NUM_PIECES;          // 120
constexpr int CM_NUM_SPECIAL_HINTS = 5;
constexpr int CM_RECENCY_BUCKET_T0 = 32;          // ≤32 steps
constexpr int CM_RECENCY_BUCKET_T1 = 128;         // ≤128 steps
// 9-rank ordinary ladder: GONGB=1 ... SILING=9 (0=unknown).
constexpr int CM_NUM_RANK_FLOORS = 9;

enum class Seat : int8_t { SOUTH = 0, WEST = 1, NORTH = 2, EAST = 3 };
enum class PieceType : int8_t {
  NONE = 0, DARK = 1, JUNQI = 2, DILEI = 3, ZHADAN = 4,
  SILING = 5, JUNZH = 6, SHIZH = 7, LVZH = 8, TUANZH = 9,
  YINGZH = 10, LIANZH = 11, PAIZH = 12, GONGB = 13
};

struct DeviceGameStateBatch {
  int num_envs = 0;
  const int max_num_moves;
  int16_t* d_cell_piece_id_per_piece = nullptr;
  int8_t*  d_piece_seat_arr = nullptr;
  int8_t*  d_piece_type_arr = nullptr;
  bool*    d_alive = nullptr;
  int8_t*  d_pos_x = nullptr;
  int8_t*  d_pos_y = nullptr;
  int8_t*  d_zero_x = nullptr;
  int8_t*  d_zero_y = nullptr;
  int16_t* d_move_count_arr = nullptr;
  int16_t* d_active_eat_arr = nullptr;
  int16_t* d_passive_surv_arr = nullptr;
  int8_t*  d_death_reason_arr = nullptr;
  int16_t* d_death_step_arr = nullptr;
  int16_t* d_death_loc_flat_arr = nullptr;
  int16_t* d_cell_piece_id = nullptr;
  bool* d_seat_dead_arr = nullptr;
  bool* d_seat_flag_revealed_arr = nullptr;
  int8_t*  d_turn = nullptr;
  int64_t* d_zobrist = nullptr;
  int32_t* d_move_counter = nullptr;
  int32_t* d_moves_since_last_combat = nullptr;

  // Phase 1b: per-env termination state (needed for step_batch).
  // Kept GPU-resident so step_batch can skip already-terminated envs and
  // produce a complete MoveResultBatch without a pre-pass H2D copy.
  bool*    d_terminated   = nullptr;   // (N,)
  int8_t*  d_winner_team  = nullptr;   // (N,) -1/0/1
  bool*    d_draw         = nullptr;   // (N,)

  // Move history ring buffer for src_dst_planes observation channels.
  // d_move_history: (N, MOVE_HISTORY_LEN, 2) int16 — [step][0]=src_flat, [step][1]=dst_flat
  // d_history_write_idx: (N,) int32 — next write position (mod MOVE_HISTORY_LEN)
  // d_history_count: (N,) int32 — valid entries (0..MOVE_HISTORY_LEN)
  int16_t* d_move_history      = nullptr;
  int32_t* d_history_write_idx = nullptr;
  int32_t* d_history_count     = nullptr;

  // ----------------------------------------------------------------
  // CombatMemory v4 state (per-observer × per-piece-id, see
  // junqi_core/combat_memory.py).
  //
  // All arrays have shape (N, 4 observers, 120 pids) flattened row-major:
  //   index = ((env * 4 + observer) * 120) + pid.
  // ----------------------------------------------------------------
  uint64_t* d_cm_direct_lo                 = nullptr;  // direct pid bitmap low
  uint64_t* d_cm_direct_hi                 = nullptr;  // direct pid bitmap high
  uint16_t* d_cm_direct_type               = nullptr;  // 12-bit type multi-hot
  int16_t*  d_cm_last_direct_step          = nullptr;
  int16_t*  d_cm_direct_other_count        = nullptr;  // count of non-my victims
  uint64_t* d_cm_chain_lo                  = nullptr;  // full chain pid bitmap low
  uint64_t* d_cm_chain_hi                  = nullptr;  // full chain pid bitmap high
  uint16_t* d_cm_chain_type                = nullptr;  // chain type multi-hot
  int16_t*  d_cm_last_chain_step           = nullptr;
  uint64_t* d_cm_eaten_by_pid_lo           = nullptr;  // reverse index low
  uint64_t* d_cm_eaten_by_pid_hi           = nullptr;  // reverse index high
  int8_t*   d_cm_rank_floor                = nullptr;  // 0..9
  int16_t*  d_cm_rank_floor_step           = nullptr;
  bool*     d_cm_is_gongb                  = nullptr;
  bool*     d_cm_not_gongb                 = nullptr;
  bool*     d_cm_attacked_by_known_gongb   = nullptr;

  DeviceGameStateBatch(int num_envs, int max_num_moves = 4000);
  ~DeviceGameStateBatch();
  DeviceGameStateBatch(const DeviceGameStateBatch&) = delete;
  DeviceGameStateBatch& operator=(const DeviceGameStateBatch&) = delete;

  void copy_from_host(
    const int16_t* h_cell_piece_id_per_piece, const int8_t* h_piece_seat_arr,
    const int8_t* h_piece_type_arr, const bool* h_alive, const int8_t* h_pos_x,
    const int8_t* h_pos_y, const int8_t* h_zero_x, const int8_t* h_zero_y,
    const int16_t* h_move_count_arr, const int16_t* h_active_eat_arr,
    const int16_t* h_passive_surv_arr, const int8_t* h_death_reason_arr,
    const int16_t* h_death_step_arr, const int16_t* h_death_loc_flat_arr,
    const int16_t* h_cell_piece_id, const bool* h_seat_dead_arr,
    const bool* h_seat_flag_revealed_arr, const int8_t* h_turn,
    const int64_t* h_zobrist, const int32_t* h_move_counter,
    const int32_t* h_moves_since_last_combat, int stream_id = 0
  );

  // Minimal H2D path for legal_action_ids_batch.  Uploads only the 6 fields
  // the M2 kernel actually reads, and fuses them into a single contiguous
  // staging buffer + one cudaMemcpy.  Leaves the other SoA arrays on the
  // device untouched — callers that need them (step / observation) must call
  // copy_from_host instead.
  void copy_from_host_legal_lite(
    const int8_t*  h_piece_seat_arr,   // N*120
    const int8_t*  h_piece_type_arr,   // N*120
    const bool*    h_alive,            // N*120
    const int8_t*  h_pos_x,            // N*120
    const int8_t*  h_pos_y,            // N*120
    const int16_t* h_cell_piece_id,    // N*289
    int stream_id = 0
  );

  void copy_to_host(
    int16_t* h_cell_piece_id_per_piece, int8_t* h_piece_seat_arr,
    int8_t* h_piece_type_arr, bool* h_alive, int8_t* h_pos_x,
    int8_t* h_pos_y, int8_t* h_zero_x, int8_t* h_zero_y,
    int16_t* h_move_count_arr, int16_t* h_active_eat_arr,
    int16_t* h_passive_surv_arr, int8_t* h_death_reason_arr,
    int16_t* h_death_step_arr, int16_t* h_death_loc_flat_arr,
    int16_t* h_cell_piece_id, bool* h_seat_dead_arr,
    bool* h_seat_flag_revealed_arr, int8_t* h_turn,
    int64_t* h_zobrist, int32_t* h_move_counter,
    int32_t* h_moves_since_last_combat, int stream_id = 0
  ) const;
};

struct DeviceObservationBatch {
  int num_envs = 0;
  float* d_spatial = nullptr;
  float* d_global = nullptr;

  DeviceObservationBatch(int num_envs);
  ~DeviceObservationBatch();
  DeviceObservationBatch(const DeviceObservationBatch&) = delete;
  DeviceObservationBatch& operator=(const DeviceObservationBatch&) = delete;

  void copy_to_host(float* h_spatial, float* h_global, int stream_id = 0) const;
};

// Single-seat observation batch: (N, NUM_OBS_CHANNELS, 17, 17).
// Used in the PPO collect hot-path where only the acting seat's observation is needed.
struct DeviceObservationSingleBatch {
  int num_envs = 0;
  float* d_spatial = nullptr;  // (N, NUM_OBS_CHANNELS, 17, 17)
  float* d_global = nullptr;   // (N, 28)

  DeviceObservationSingleBatch(int num_envs);
  ~DeviceObservationSingleBatch();
  DeviceObservationSingleBatch(const DeviceObservationSingleBatch&) = delete;
  DeviceObservationSingleBatch& operator=(const DeviceObservationSingleBatch&) = delete;
};

// ---------------------------------------------------------------------------
// DeviceRolloutHistory — compact GPU-resident training history.
//
// Stores only state required to reconstruct the acting seat's observation and
// legal-action mask. Belief and CombatMemory are saved for the acting observer
// only (rather than all four observers), reducing history storage by 4x for
// those dominant fields. A gathered minibatch is restored into temporary
// DeviceGameStateBatch/DeviceObservationSingleBatch buffers on device.
// ---------------------------------------------------------------------------
struct RolloutHistoryReconstruction {
  const float* d_spatial = nullptr;  // (B, NUM_OBS_CHANNELS, 17, 17)
  const float* d_global = nullptr;   // (B, NUM_GLOBAL_DIMS)
  const bool* d_legal_mask = nullptr;  // (B, FLAT_ACTION_SPACE)
  int batch_size = 0;
};

struct DeviceRolloutHistory {
  int num_steps = 0;
  int num_envs = 0;
  int replay_capacity = 0;
  uint64_t history_bytes = 0;

  // State fields read by observation/legal kernels. Layout is
  // (num_steps, num_envs, stride), flattened with env as the inner row.
  int8_t* d_piece_seat_arr = nullptr;
  int8_t* d_piece_type_arr = nullptr;
  bool* d_alive = nullptr;
  int8_t* d_pos_x = nullptr;
  int8_t* d_pos_y = nullptr;
  int8_t* d_zero_x = nullptr;
  int8_t* d_zero_y = nullptr;
  int16_t* d_move_count_arr = nullptr;
  int16_t* d_active_eat_arr = nullptr;
  int16_t* d_passive_surv_arr = nullptr;
  int8_t* d_death_reason_arr = nullptr;
  int16_t* d_death_loc_flat_arr = nullptr;
  int16_t* d_cell_piece_id = nullptr;
  bool* d_seat_dead_arr = nullptr;
  bool* d_seat_flag_revealed_arr = nullptr;
  int8_t* d_turn = nullptr;
  int32_t* d_move_counter = nullptr;
  int32_t* d_moves_since_last_combat = nullptr;
  int16_t* d_move_history = nullptr;
  int32_t* d_history_write_idx = nullptr;
  int32_t* d_history_count = nullptr;

  // Acting-observer slice only: (T, N, 12, 289).
  float* d_observer_belief = nullptr;

  // Acting-observer CombatMemory slice only: (T, N, 120).
  uint64_t* d_cm_direct_lo = nullptr;
  uint64_t* d_cm_direct_hi = nullptr;
  uint16_t* d_cm_direct_type = nullptr;
  int16_t* d_cm_last_direct_step = nullptr;
  int16_t* d_cm_direct_other_count = nullptr;
  uint64_t* d_cm_chain_lo = nullptr;
  uint64_t* d_cm_chain_hi = nullptr;
  uint16_t* d_cm_chain_type = nullptr;
  int16_t* d_cm_last_chain_step = nullptr;
  uint64_t* d_cm_eaten_by_pid_lo = nullptr;
  uint64_t* d_cm_eaten_by_pid_hi = nullptr;
  int8_t* d_cm_rank_floor = nullptr;
  int16_t* d_cm_rank_floor_step = nullptr;
  bool* d_cm_is_gongb = nullptr;
  bool* d_cm_not_gongb = nullptr;
  bool* d_cm_attacked_by_known_gongb = nullptr;

  // Reusable reconstruction scratch, grown to the largest PPO minibatch.
  DeviceGameStateBatch* replay_state = nullptr;
  DeviceObservationSingleBatch* replay_obs = nullptr;
  float* d_replay_belief = nullptr;

  DeviceRolloutHistory(int num_steps, int num_envs);
  ~DeviceRolloutHistory();
  DeviceRolloutHistory(const DeviceRolloutHistory&) = delete;
  DeviceRolloutHistory& operator=(const DeviceRolloutHistory&) = delete;

  void snapshot(
    const DeviceGameStateBatch& state,
    const float* d_belief,
    const int8_t* d_acting_seats,
    int step,
    int stream_id = 0
  );

  RolloutHistoryReconstruction reconstruct(
    const int64_t* d_flat_indices,
    const int8_t* d_acting_seats,
    int batch_size,
    int8_t show_mode = 2,
    int stream_id = 0
  );
};

// ---------------------------------------------------------------------------
// GpuScratch — per-process singleton holding all transient device / pinned
// buffers that previously went through cudaMalloc/cudaFree on every call.
//
// Lifetime: allocated once, grows on demand, lives until process exit.
// Not thread-safe across distinct devices (single-GPU assumption for now).
// ---------------------------------------------------------------------------
struct GpuScratch {
  // Device-resident scratch buffers (grow-only).
  int8_t*  d_acting_seats    = nullptr;  // cap: acting_seats_cap envs
  float*   d_belief          = nullptr;  // cap: belief_cap envs
  int8_t*  d_observer_seats  = nullptr;  // cap: observer_seats_cap envs
  int32_t  acting_seats_cap  = 0;
  int32_t  belief_cap        = 0;
  int32_t  observer_seats_cap = 0;

  // Legal-action output buffers (grow-only, used by legal_action_ids_batch).
  int32_t* d_action_ids      = nullptr;  // cap: action_ids_cap envs
  int32_t* d_action_counts   = nullptr;  // cap: action_ids_cap envs
  int32_t  action_ids_cap    = 0;

  // CSR legal-action output buffers.
  int32_t* d_csr_offsets     = nullptr;  // cap: action_ids_cap + 1
  int32_t* d_csr_values      = nullptr;  // cap: csr_values_cap
  int32_t  csr_values_cap    = 0;

  // Per-piece legal mask (32 slots per piece).  cap: mask_cap envs × 120 × 32 bits.
  bool*    d_piece_slot_mask = nullptr;
  int32_t  mask_cap          = 0;

  // CUDA streams (ping-pong) for overlapped H2D and kernel execution.
  void* stream_a = nullptr;   // cudaStream_t, opaque to public headers
  void* stream_b = nullptr;

  static GpuScratch& instance();

  void ensure_acting_seats(int n);
  void ensure_belief(int n);
  void ensure_observer_seats(int n);
  void ensure_action_ids(int n);
  void ensure_csr_values(int n);
  void ensure_mask(int n);
  void ensure_streams();

  // Test-only: releases everything (subsequent calls reallocate).
  void reset();

  GpuScratch() = default;
  ~GpuScratch();
  GpuScratch(const GpuScratch&) = delete;
  GpuScratch& operator=(const GpuScratch&) = delete;
};

std::pair<int32_t*, int32_t*> legal_action_ids_batch(
  const DeviceGameStateBatch& d_state, const int8_t* d_acting_seats, int stream_id = 0
);

// CSR variant: returns pointers into persistent GpuScratch buffers.
// offsets[0..N]  — prefix sum of per-env legal-action counts (int32_t, N+1 entries)
// values[0..T)  — concatenated action IDs   where T = offsets[N]
// counts[0..N)  — per-env count (same as offsets[i+1]-offsets[i]); convenience
struct LegalActionCsrResult {
  const int32_t* d_offsets;   // size N+1
  const int32_t* d_values;    // size offsets[N]
  const int32_t* d_counts;    // size N
  int N;
  int total_actions;          // == offsets[N]
};

LegalActionCsrResult legal_action_ids_batch_csr(
  const DeviceGameStateBatch& d_state, const int8_t* d_acting_seats, int stream_id = 0
);

// Per-piece slot legal mask variant.
// mask[env, pid, slot] ∈ {0,1}; shape flat (N, 120, 32).  Returns the owning
// pointer (part of GpuScratch).
struct LegalActionMaskResult {
  const bool* d_mask;       // size N * 120 * 32
  int N;
};

LegalActionMaskResult legal_action_mask_batch(
  const DeviceGameStateBatch& d_state, const int8_t* d_acting_seats, int stream_id = 0
);

// Dense canonical-frame legal mask.
// Returns a device pointer to (N, FLAT_ACTION_SPACE) bool mask where each
// action id = src_canonical * 289 + dst_canonical is in the observer's
// canonical coordinate frame.  The pointer is owned by a module-level
// persistent buffer (grows on demand, never freed until process exit).
bool* legal_mask_canonical_batch(
  const DeviceGameStateBatch& d_state, const int8_t* d_acting_seats, int stream_id = 0
);

// Device-resident step result — all pointers are persistent device buffers
// owned by a module-level singleton (never freed until process exit).
struct StepDeviceResult {
  const bool*    d_valid;
  const int8_t*  d_event;
  const bool*    d_terminated;
  const int8_t*  d_winner_team;
  const bool*    d_draw;
  const bool*    d_flag_captured;
  const float*   d_rewards;       // per-acting-seat reward (terminal + shaping)
  const int32_t* d_world_actions; // world-frame actions (src*289+dst) for belief update
  int N;
};

// Full device-resident step pipeline: canonical action rotation + step + reward.
// No host transfers in the hot path.
StepDeviceResult step_device(
  DeviceGameStateBatch& d_state,
  const int32_t* d_canonical_actions,  // device pointer
  const int8_t*  d_acting_seats,       // device pointer
  int stream_id = 0
);

// Record world-frame actions into the move history ring buffer.
// Called after step_device() — only records for non-terminated envs.
void record_move_history(
  DeviceGameStateBatch& d_state,
  const int32_t* d_world_actions,  // (N,) src_flat*289+dst_flat
  const bool*    d_valid,          // (N,) from step result
  int stream_id = 0
);

// Slots per piece in the compact legal-mask output.
// Expanded to 80 slots for the legacy-correct rail topology (ADR-126):
//   slots 0-3   : ortho 1-step                (ADJACENT_CELLS[0..3])
//   slots 4-7   : diag 1-step via camp        (ADJACENT_CELLS[4..7])
//   slots 8-55  : non-engineer straight rail  (4 dirs × 12 cells, k=1..12)
//   slots 56-67 : non-engineer curve-rail BFS (up to 12 cells / curve)
//   slots 8-79  : engineer BFS-reachable rail cells (up to 72 in traversal
//                 order)  — engineer uses slots 8..79 as a single ragged
//                 packed list; non-engineer leaves 68..79 zero.
// Total = 4 + 4 + 4*12 + 12 + 12 = 80.
constexpr int SLOTS_PER_PIECE = 80;

// ---------------------------------------------------------------------------
// step_batch — advance every environment by one action.
//
// Inputs:
//   d_state       : in-place mutated device state (GPU-resident)
//   d_action_ids  : (N,) int32 — one flat action id per env (src*289+dst).
//                   Terminated envs are skipped; the action at those slots
//                   is ignored.
//
// Output buffers (all pre-allocated by caller, device-resident, shape (N,)):
//   d_valid           : bool    — false iff action was invalid (no src piece)
//                                 or env was already terminated.
//   d_event           : int8    — Event.value: 0 invalid, 1 MOVE, 2 EAT,
//                                 3 BOMB, 4 KILLED.
//   d_terminated      : bool    — new terminated flag (mirrors in-state).
//   d_winner_team     : int8    — -1 (None), 0, or 1.
//   d_draw            : bool    — draw flag.
//   d_flag_captured   : bool    — whether this step ended with a flag capture.
//
// The kernel keeps all state GPU-resident (no H2D / D2H inside).  Callers
// that need host-side MoveResultBatch must issue a single D2H copy of the
// six output arrays afterwards.
// ---------------------------------------------------------------------------
void step_batch(
  DeviceGameStateBatch& d_state, const int32_t* d_action_ids,
  bool* d_valid_out, int8_t* d_event_out,
  bool* d_terminated_out, int8_t* d_winner_out,
  bool* d_draw_out, bool* d_flag_captured_out,
  int stream_id = 0
);

void build_observation_batch(
  const DeviceGameStateBatch& d_state, const float* d_beliefs,
  const int8_t* d_observer_seats, DeviceObservationBatch& d_obs_out,
  int8_t show_mode = 2, int stream_id = 0
);

// Single-seat variant: builds observation for ONE seat per env.
// d_acting_seats: device pointer (N,) int8 — per-env seat to observe.
// Output: d_obs_out.d_spatial (N, 412, 17, 17), d_obs_out.d_global (N, 28).
void build_observation_single_seat(
  const DeviceGameStateBatch& d_state, const float* d_beliefs,
  const int8_t* d_acting_seats, DeviceObservationSingleBatch& d_obs_out,
  int8_t show_mode = 2, int stream_id = 0
);

void init_tables();
void cleanup_tables();
int get_gpu_count();
void set_device(int device_id);

// Phase 5: Device-side episode reset via pre-computed setup pool.
// Upload constant tables (position/seat lookup) — call once at init.
void upload_reset_tables(
    const int8_t* h_pos_x,         // (120,) int8
    const int8_t* h_pos_y,         // (120,) int8
    const int8_t* h_piece_seat,    // (120,) int8
    const bool*   h_camp_slot);    // (30,)  bool

// Upload pool of valid initial setups — call once at init.
// h_piece_types: (count, 120) int8 — piece types for each pool entry.
void upload_setup_pool(const int8_t* h_piece_types, int count);

// Reset all terminated envs from the setup pool.  Zero CPU involvement.
// seed: combined with env index for hash-based pool selection.
void reset_terminated_envs(DeviceGameStateBatch& d_state, int64_t seed);

// Upload compact cell index mapping tables (call once at init).
void upload_compact_cell_maps(const int16_t* h_flat_to_compact,
                               const int16_t* h_compact_to_flat);

// Upload CPU-seeded zobrist tables so GPU and CPU hashes match bit-for-bit.
// Sizes must match junqi_core/_zobrist.py exactly; see tables.cuh for layout.
void upload_zobrist_tables_from_host(
    const int64_t* h_piece,
    const int64_t* h_turn,
    const int64_t* h_move_counter,
    const int64_t* h_moves_since_combat,
    const int64_t* h_winner,
    int64_t        h_terminated,
    int64_t        h_draw,
    const int64_t* h_seat_dead,
    const int64_t* h_seat_flag_revealed
);

// ---------------------------------------------------------------------------
// Phase 1 Belief Update — GPU-side deductive belief inference.
//
// Belief tensor layout: [N, 4, 12, 289] float32 (C-contiguous).
// Stored in GpuScratch::d_belief.
// ---------------------------------------------------------------------------

// Upload the per-slot prior table (30 slots × 12 types, float32).
// Call once at process start, before any belief init.
void upload_belief_prior_table(const float* h_table);

// Upload the precomputed stronghold positions (4 seats × 2 strongholds, int16).
// Call once at process start.
void upload_seat_strongholds(const int16_t* h_strongholds);

// Initialise beliefs for all envs (called after initial reset or bulk reset).
// Sets own/teammate beliefs to one-hot, enemy beliefs to per-slot priors.
void init_beliefs_for_reset_envs(
    const DeviceGameStateBatch& d_state,
    float* d_belief,
    int stream_id = 0
);

// Initialise beliefs for ALL envs unconditionally (initial bulk reset).
void init_all_beliefs(
    const DeviceGameStateBatch& d_state,
    float* d_belief,
    int stream_id = 0
);

// Save a snapshot of seat_flag_revealed and seat_dead BEFORE step_device().
// Used to detect new flag reveals and seat deaths in the belief update.
void snapshot_pre_step_flags(
    const DeviceGameStateBatch& d_state,
    int stream_id = 0
);

// Incremental belief update after step_device() completes.
// Applies deductive rules R1 (migration), R4 (flag capture),
// R5/R7 (engineer signature), R6 (stronghold deduction),
// R9 (seat death), I5 (consistency sweep).
void update_beliefs_after_step(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_event,
    const bool* d_flag_captured,
    const int32_t* d_world_actions,
    float* d_belief,
    int stream_id = 0
);

}  // namespace junqi_cuda
