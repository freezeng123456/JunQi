/*
 * tables.cuh
 * Precomputed static tables for move generation and zobrist hashing.
 *
 * AUTHORITATIVE TOPOLOGY SOURCE:
 *
 * Rail data (RAIL_FLAT, NINE_GRID_FLAT, ENG_RAIL_*, STRAIGHT_RAIL_RAYS,
 * CURVE_RAIL_OF) mirrors ``junqi_core/rail_topology.py`` which is derived
 * directly from the legacy C engine (``legacy_engine/src/junqi.c``).
 * Notable features of this topology:
 *
 *   * NineGrid cells ARE railways (``InitNineGrid`` sets ``isRailway=1``).
 *   * The rail graph includes:
 *       - orthogonal rail↔rail edges,
 *       - NineGrid ↔ NineGrid 2-step orthogonal jumps,
 *       - 4 ``AddSpcNode`` diagonal corner edges.
 *   * 73 rail cells in ONE connected component (not 4 × 16-cycles).
 *   * Node degree up to 4 at curve corners and NineGrid hubs.
 *   * Curve rails (legacy ``InitCurveRail``): 4 L-shaped rails of 12 cells.
 *
 * The widths below (ENG_NUM_RAIL, ENG_ADJ_WIDTH, STRAIGHT_RAY_LEN) are
 * upper bounds measured from the legacy topology; DO NOT increase them
 * without also bumping the matching Python-side constants in
 * ``_movegen_tables.py``.
 */

#pragma once

#include <cstdint>

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// Zobrist tables (device global pointers to heap allocations)
//
// Layout mirrors ``junqi_core/_zobrist.py`` EXACTLY — the tables must be
// uploaded from the host via ``upload_zobrist_tables_from_host`` with the
// CPU-side numpy arrays (seed "JUNQI_ZO") so that GPU and CPU produce
// bit-identical zobrist values at every step.  If no host upload happens,
// ``init_tables()`` seeds the tables with a deterministic MT19937 stream
// (see tables.cu) — this is ONLY for standalone C++ tests and will NOT
// match CPU.
// ---------------------------------------------------------------------------
extern __device__ int64_t* d_zobrist_piece;          // [120 * 14 * 289]
extern __device__ int64_t* d_zobrist_turn;           // [4]
extern __device__ int64_t* d_zobrist_move_counter;   // [4096] low-12-bits sketch
extern __device__ int64_t* d_zobrist_moves_since_combat; // [512] low-9-bits
extern __device__ int64_t* d_zobrist_winner;         // [3]    idx = team+1
extern __device__ int64_t  d_zobrist_terminated;     // scalar
extern __device__ int64_t  d_zobrist_draw;           // scalar
extern __device__ int64_t* d_zobrist_seat_dead;      // [4]
extern __device__ int64_t* d_zobrist_seat_flag_revealed; // [4]

// Masks for counter indexing (keep in sync with _zobrist.py).
constexpr int ZOB_MOVE_COUNTER_MASK = 0xFFF;
constexpr int ZOB_MOVES_SINCE_COMBAT_MASK = 0x1FF;

// Move generation static tables in constant memory.
// ADJACENT_CELLS[flat * 8 + k]: k=0..3 ortho neighbors (+x,-x,+y,-y),
// k=4..7 diag-camp neighbors.  -1 = off-board / no entry.
extern __constant__ int16_t ADJACENT_CELLS[289 * 8];

// Boolean property masks for each of the 289 flat cells.
extern __constant__ bool CAMP_FLAT[289];
extern __constant__ bool STRONGHOLD_FLAT[289];
extern __constant__ bool RAIL_FLAT[289];
extern __constant__ bool ON_BOARD_FLAT[289];
extern __constant__ bool NINE_GRID_FLAT[289];

// ---------------------------------------------------------------------------
// Straight-line rail rays, graph-constrained to same row/column
// (replicates junqi_core._movegen_tables.STRAIGHT_RAIL_RAYS_PAD).
//
// STRAIGHT_RAIL_RAYS[flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN + k]
// -1 = end of ray (and any further k's for this dir).
// The longest ray is 12 cells (e.g. column x=6 from (6,1) to (6,15) via
// NineGrid 2-step jumps).
// ---------------------------------------------------------------------------
constexpr int STRAIGHT_RAY_LEN = 12;
extern __constant__ int16_t STRAIGHT_RAIL_RAYS[289 * 4 * STRAIGHT_RAY_LEN];

// ---------------------------------------------------------------------------
// Engineer BFS rail graph (generalized — supports junctions of degree 4).
//
// ENG_RAIL_CELLS[rail_idx] = flat cell id.  ENG_NUM_RAIL = 73.
// ENG_RAIL_TO_IDX[flat] = rail_idx, -1 for non-rail cells.
// ENG_RAIL_ADJ[rail_idx * ENG_ADJ_WIDTH + k] = rail-idx of k-th neighbour,
//   -1 = padding slot.  Width 4 because the max degree in the legacy rail
//   graph is 4 (curve corners + NineGrid hubs).  Includes NineGrid 2-step
//   jumps and the 4 AddSpcNode diagonal corner edges.
// ---------------------------------------------------------------------------
constexpr int ENG_NUM_RAIL  = 73;
constexpr int ENG_ADJ_WIDTH = 4;
extern __constant__ int16_t ENG_RAIL_CELLS[ENG_NUM_RAIL];
extern __constant__ int8_t  ENG_RAIL_TO_IDX[289];
extern __constant__ int8_t  ENG_RAIL_ADJ[ENG_NUM_RAIL * ENG_ADJ_WIDTH];

// ---------------------------------------------------------------------------
// Curve rails (legacy InitCurveRail).
//
// CURVE_RAIL_OF[flat]         = curve-rail id in {0, 1, 2, 3, 4}; 0 = not
//                               on any curve rail.
// Each of the 4 curve rails holds 12 cells (cross-seat "L" shape around
// inner corners).  On the rail graph these are connected via the
// AddSpcNode diagonal edges, so curve-rail BFS is just "rail BFS where
// every step must stay on the same curve".
// NOTE: only 10 of each 12 are railway — the legacy InitCurveRail loop also
// tags two headquarters cells per curve.  Always pair this with RAIL_FLAT.
// ---------------------------------------------------------------------------
extern __constant__ int8_t CURVE_RAIL_OF[289];

// ---------------------------------------------------------------------------
// CURVE_ARC_FLAT[flat]        = true for the 8 cells that adjoin an
//                               AddSpcNode diagonal (arc) link, two per board
//                               corner.  The arc belongs to the edge, not the
//                               cell: these are the only points where a
//                               non-engineer may leave a straight rail run.
//                               Mirrors junqi_core.rail_topology.CURVE_ARC_CELLS.
// ---------------------------------------------------------------------------
extern __constant__ bool CURVE_ARC_FLAT[289];

constexpr int SLOTS_PER_SEAT_DEV = 30;

// ---------------------------------------------------------------------------
// SLOT_TO_CHANNEL[slot]       = plane index within the observation's
//                               piece_slot group, or -1 for the five camp
//                               slots, which no legal setup ever occupies.
//                               Mirrors junqi_core.observation._SLOT_TO_CHANNEL.
// ---------------------------------------------------------------------------
extern __constant__ int8_t SLOT_TO_CHANNEL[SLOTS_PER_SEAT_DEV];

// Host functions
void init_tables();
void cleanup_tables();

// ---------------------------------------------------------------------------
// Upload CPU-seeded zobrist tables so GPU / CPU hashes match bit-for-bit.
//
// Sizes MUST match _zobrist.py exactly:
//   piece:                    120 * 14 * 289 = 485,640 int64
//   turn:                     4
//   move_counter:             4096
//   moves_since_combat:       512
//   winner:                   3
//   terminated:               scalar int64
//   draw:                     scalar int64
//   seat_dead:                4
//   seat_flag_revealed:       4
// ---------------------------------------------------------------------------
void upload_zobrist_tables_from_host(
    const int64_t* h_piece,              // 485,640
    const int64_t* h_turn,               // 4
    const int64_t* h_move_counter,       // 4096
    const int64_t* h_moves_since_combat, // 512
    const int64_t* h_winner,             // 3
    int64_t        h_terminated,
    int64_t        h_draw,
    const int64_t* h_seat_dead,          // 4
    const int64_t* h_seat_flag_revealed  // 4
);

}  // namespace junqi_cuda
