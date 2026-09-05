/*
 * tables.cu
 * Static lookup table initialization for the JunQi CUDA backend.
 *
 * Builds board topology and Zobrist hashing tables at runtime using the
 * same geometry formulas as junqi_core/rail_topology.py.  Everything
 * uploaded here lives in constant or device memory for the lifetime of
 * the process; call cleanup_tables() to free heap tables.
 *
 * LEGACY-CORRECT RAIL TOPOLOGY (mirrors junqi_core.rail_topology):
 *   * NineGrid cells are marked as railways (InitNineGrid).
 *   * Rail adjacency graph adds NineGrid 2-step ortho jumps AND the 4
 *     AddSpcNode diagonal corner edges.
 *   * 73 rail cells, single connected component, max degree 4.
 */

#include "tables.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// Device-global Zobrist pointer storage
// ---------------------------------------------------------------------------
__device__ int64_t* d_zobrist_piece = nullptr;
__device__ int64_t* d_zobrist_turn  = nullptr;
__device__ int64_t* d_zobrist_move_counter = nullptr;
__device__ int64_t* d_zobrist_moves_since_combat = nullptr;
__device__ int64_t* d_zobrist_winner = nullptr;
__device__ int64_t  d_zobrist_terminated = 0;
__device__ int64_t  d_zobrist_draw = 0;
__device__ int64_t* d_zobrist_seat_dead = nullptr;
__device__ int64_t* d_zobrist_seat_flag_revealed = nullptr;

// Host-side copies so cleanup_tables() can call cudaFree.
static int64_t* h_d_zobrist_piece = nullptr;
static int64_t* h_d_zobrist_turn  = nullptr;
static int64_t* h_d_zobrist_move_counter = nullptr;
static int64_t* h_d_zobrist_moves_since_combat = nullptr;
static int64_t* h_d_zobrist_winner = nullptr;
static int64_t* h_d_zobrist_seat_dead = nullptr;
static int64_t* h_d_zobrist_seat_flag_revealed = nullptr;

// ---------------------------------------------------------------------------
// Constant-memory board topology tables (declared in tables.cuh)
// ---------------------------------------------------------------------------
__constant__ int16_t ADJACENT_CELLS[289 * 8];
__constant__ bool    CAMP_FLAT[289];
__constant__ bool    STRONGHOLD_FLAT[289];
__constant__ bool    RAIL_FLAT[289];
__constant__ bool    ON_BOARD_FLAT[289];
__constant__ bool    NINE_GRID_FLAT[289];
__constant__ int16_t STRAIGHT_RAIL_RAYS[289 * 4 * STRAIGHT_RAY_LEN];
__constant__ int16_t ENG_RAIL_CELLS[ENG_NUM_RAIL];
__constant__ int8_t  ENG_RAIL_TO_IDX[289];
__constant__ int8_t  ENG_RAIL_ADJ[ENG_NUM_RAIL * ENG_ADJ_WIDTH];
__constant__ int8_t  CURVE_RAIL_OF[289];
__constant__ int8_t  SLOT_TO_CHANNEL[SLOTS_PER_SEAT_DEV];
__constant__ bool    CURVE_ARC_FLAT[289];

// ---------------------------------------------------------------------------
// Board geometry (replicated from junqi_core.rail_topology)
// ---------------------------------------------------------------------------

static constexpr int BS = 17;   // BOARD_SIZE
static constexpr int NC = 289;  // NUM_CELLS

static inline int flat(int x, int y) { return y * BS + x; }
static inline int flat_x(int f)      { return f % BS; }
static inline int flat_y(int f)      { return f / BS; }

// Camp / stronghold indices within a seat's 30 slots.
static const int CAMP_IDX[5]       = { 6, 8, 12, 16, 18 };
static const int STRONGHOLD_IDX[2] = { 26, 28 };

static void seat_pos(int seat, int i, int& x, int& y) {
    int col = i % 5, row = i / 5;
    switch (seat) {
        case 0: x = 10 - col; y = 11 + row; break;   // SOUTH / HOME
        case 1: x = 5 - row;  y = 10 - col; break;   // WEST  / RIGHT
        case 2: x = 6 + col;  y = 5 - row;  break;   // NORTH / OPPS
        case 3: x = 11 + row; y = 6 + col;  break;   // EAST  / LEFT
        default: x = y = -1; break;
    }
}

// Nine-grid: (x,y) = (10 - (i%3)*2, 6 + (i//3)*2)
static void nine_grid_pos(int i, int& x, int& y) {
    x = 10 - (i % 3) * 2;
    y = 6 + (i / 3) * 2;
}

// ---------------------------------------------------------------------------
// Per-cell boolean properties (same definitions as rail_topology.py).
// ---------------------------------------------------------------------------

struct CellProps {
    bool on_board[NC];
    bool is_camp[NC];
    bool is_stronghold[NC];
    bool is_railway[NC];
    bool is_nine_grid[NC];
    int8_t curve_rail[NC];   // 0 = no curve; 1..4 = curve id
};

static CellProps build_cell_props() {
    CellProps p{};

    // Seat zones (4 × 30 = 120 cells).  Mirrors SetBoardRailway and
    // friends; NineGrid cells are handled separately below.
    for (int seat = 0; seat < 4; ++seat) {
        for (int i = 0; i < 30; ++i) {
            int x, y;
            seat_pos(seat, i, x, y);
            int f = flat(x, y);
            p.on_board[f] = true;
            for (int ci : CAMP_IDX)       if (i == ci) { p.is_camp[f]       = true; break; }
            for (int si : STRONGHOLD_IDX) if (i == si) { p.is_stronghold[f] = true; break; }
            int col = i % 5, row = i / 5;
            if (i < 25 && (row == 0 || row == 4 || col == 0 || col == 4))
                p.is_railway[f] = true;
        }
    }

    // NineGrid (9 central cells).  Legacy InitNineGrid sets isRailway=1.
    for (int i = 0; i < 9; ++i) {
        int x, y;
        nine_grid_pos(i, x, y);
        int f = flat(x, y);
        p.on_board[f] = true;
        p.is_nine_grid[f] = true;
        p.is_railway[f] = true;
    }

    // Curve rails (legacy InitCurveRail):
    //   for cid in 1..4:
    //     seat_a = cid - 1, seat_b = cid % 4
    //     for slot j in 0..29 with j%5 == 4:
    //       ChessPos[seat_a][j].eCurveRail     = cid
    //       ChessPos[seat_b][j-4].eCurveRail   = cid
    for (int cid = 1; cid <= 4; ++cid) {
        int seat_a = cid - 1;
        int seat_b = cid % 4;
        for (int j = 0; j < 30; ++j) {
            if (j % 5 != 4) continue;
            int xa, ya; seat_pos(seat_a, j, xa, ya);
            int xb, yb; seat_pos(seat_b, j - 4, xb, yb);
            p.curve_rail[flat(xa, ya)] = (int8_t)cid;
            p.curve_rail[flat(xb, yb)] = (int8_t)cid;
        }
    }

    return p;
}

static CellProps g_props{};
static bool g_props_built = false;

static void ensure_props() {
    if (!g_props_built) {
        g_props = build_cell_props();
        g_props_built = true;
    }
}

// ---------------------------------------------------------------------------
// Build ADJACENT_CELLS[289 * 8]:
//   slots 0..3: orthogonal (+x,-x,+y,-y), -1 if off-board
//   slots 4..7: diagonal neighbors where src OR dst is a camp, -1 otherwise
// ---------------------------------------------------------------------------
static void build_adjacent_cells(int16_t out[289 * 8]) {
    ensure_props();
    for (int i = 0; i < 289 * 8; ++i) out[i] = -1;

    const int dx_orth[4] = {1, -1, 0, 0};
    const int dy_orth[4] = {0, 0, 1, -1};
    const int dx_diag[4] = {1, 1, -1, -1};
    const int dy_diag[4] = {1, -1, 1, -1};

    for (int y = 0; y < BS; ++y) {
        for (int x = 0; x < BS; ++x) {
            int f = flat(x, y);
            if (!g_props.on_board[f]) continue;
            for (int k = 0; k < 4; ++k) {
                int nx = x + dx_orth[k], ny = y + dy_orth[k];
                if (nx >= 0 && nx < BS && ny >= 0 && ny < BS &&
                    g_props.on_board[flat(nx, ny)])
                    out[f * 8 + k] = (int16_t)flat(nx, ny);
            }
            bool src_camp = g_props.is_camp[f];
            int slot = 4;
            for (int k = 0; k < 4 && slot < 8; ++k) {
                int nx = x + dx_diag[k], ny = y + dy_diag[k];
                if (nx < 0 || nx >= BS || ny < 0 || ny >= BS) continue;
                int nf = flat(nx, ny);
                if (!g_props.on_board[nf]) continue;
                if (src_camp || g_props.is_camp[nf])
                    out[f * 8 + (slot++)] = (int16_t)nf;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build the 73-cell rail adjacency list, legacy-correct:
//   (a) orthogonal rail↔rail edges
//   (b) NineGrid ↔ NineGrid 2-step orthogonal jumps
//   (c) AddSpcNode diagonal corner edges (4 hard-coded pairs)
// Returns a vector indexed by rail_idx; each entry is a sorted vector of
// neighbour rail_indices.
// ---------------------------------------------------------------------------
struct RailGraph {
    std::vector<int16_t>               cells;            // flat cell id of each rail idx
    std::array<int8_t, NC>             flat_to_idx{};    // -1 for non-rail cells
    std::vector<std::vector<int8_t>>   adj;              // adjacency list
};

// Hard-coded AddSpcNode corner edges (legacy junqi.c:204-236).
static constexpr std::array<std::pair<std::pair<int,int>, std::pair<int,int>>, 4>
SPC_EDGES = {{
    {{10, 11}, {11, 10}},   // South-East inner corner
    {{ 6, 11}, { 5, 10}},   // South-West inner corner
    {{ 6,  5}, { 5,  6}},   // West-North inner corner
    {{11,  6}, {10,  5}},   // North-East inner corner
}};

static RailGraph build_rail_graph() {
    ensure_props();
    RailGraph g;
    g.flat_to_idx.fill(-1);

    // (0) Collect rail cells in flat order; assign rail_idx.
    for (int f = 0; f < NC; ++f) {
        if (g_props.is_railway[f]) {
            g.flat_to_idx[f] = (int8_t)g.cells.size();
            g.cells.push_back((int16_t)f);
        }
    }
    g.adj.resize(g.cells.size());

    auto add_edge = [&](int a_flat, int b_flat) {
        int8_t ai = g.flat_to_idx[a_flat];
        int8_t bi = g.flat_to_idx[b_flat];
        if (ai < 0 || bi < 0) return;
        // Dedup: skip if already present.
        for (int8_t n : g.adj[ai]) if (n == bi) return;
        g.adj[ai].push_back(bi);
        g.adj[bi].push_back(ai);
    };

    // (a) ortho rail-rail edges
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    for (int ri = 0; ri < (int)g.cells.size(); ++ri) {
        int f = g.cells[ri];
        int x = flat_x(f), y = flat_y(f);
        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx < 0 || nx >= BS || ny < 0 || ny >= BS) continue;
            int nf = flat(nx, ny);
            if (g_props.is_railway[nf]) add_edge(f, nf);
        }
    }

    // (b) NineGrid 2-step ortho jumps
    for (int ri = 0; ri < (int)g.cells.size(); ++ri) {
        int f = g.cells[ri];
        if (!g_props.is_nine_grid[f]) continue;
        int x = flat_x(f), y = flat_y(f);
        const int dx2[4] = {2, -2, 0, 0};
        const int dy2[4] = {0, 0, 2, -2};
        for (int k = 0; k < 4; ++k) {
            int nx = x + dx2[k], ny = y + dy2[k];
            if (nx < 0 || nx >= BS || ny < 0 || ny >= BS) continue;
            int nf = flat(nx, ny);
            if (g_props.is_nine_grid[nf]) add_edge(f, nf);
        }
    }

    // (c) AddSpcNode corner edges
    for (const auto& e : SPC_EDGES) {
        add_edge(flat(e.first.first,  e.first.second),
                 flat(e.second.first, e.second.second));
    }

    // Sort each adjacency list for determinism.
    for (auto& nbrs : g.adj) std::sort(nbrs.begin(), nbrs.end());
    return g;
}

// ---------------------------------------------------------------------------
// Build STRAIGHT_RAIL_RAYS[289 * 4 * STRAIGHT_RAY_LEN]
//
// Each ray walks the rail graph in a chosen direction (0: +x, 1: -x,
// 2: +y, 3: -y) while staying on the same row (for ±x) or same column
// (for ±y).  -1 terminates a ray.  The walk is a simple chain: for any
// (axis, signed direction) every rail node has at most one rail-graph
// neighbour that satisfies the axis + monotonic-step constraints.
// ---------------------------------------------------------------------------
static void build_straight_rail_rays(int16_t out[289 * 4 * STRAIGHT_RAY_LEN]) {
    ensure_props();
    for (int i = 0; i < 289 * 4 * STRAIGHT_RAY_LEN; ++i) out[i] = -1;

    RailGraph g = build_rail_graph();

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    for (int ri = 0; ri < (int)g.cells.size(); ++ri) {
        int src_flat = g.cells[ri];
        int sx = flat_x(src_flat), sy = flat_y(src_flat);
        for (int dir = 0; dir < 4; ++dir) {
            int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
            int k = 0;
            int cur = src_flat;
            int cur_x = sx, cur_y = sy;
            int prev = -1;
            while (k < STRAIGHT_RAY_LEN) {
                int next_flat = -1;
                int8_t cur_ri = g.flat_to_idx[cur];
                if (cur_ri < 0) break;
                for (int8_t nbi : g.adj[cur_ri]) {
                    int nb_flat = g.cells[nbi];
                    if (nb_flat == prev) continue;
                    int nx = flat_x(nb_flat), ny = flat_y(nb_flat);
                    if (dx[dir] != 0) {
                        if (ny != sy) continue;
                        if ((nx - cur_x) * dx[dir] <= 0) continue;
                    } else {
                        if (nx != sx) continue;
                        if ((ny - cur_y) * dy[dir] <= 0) continue;
                    }
                    next_flat = nb_flat;
                    break;  // chain is simple; first match wins
                }
                if (next_flat < 0) break;
                out[ray_base + k] = (int16_t)next_flat;
                ++k;
                prev = cur;
                cur = next_flat;
                cur_x = flat_x(cur);
                cur_y = flat_y(cur);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build engineer rail adjacency tables (ENG_RAIL_CELLS, ENG_RAIL_TO_IDX,
// ENG_RAIL_ADJ).  Kept in rail-idx space with -1 padding; the BFS kernel
// iterates over ENG_ADJ_WIDTH entries per node.
// ---------------------------------------------------------------------------
static void build_engineer_tables(
    int16_t rail_cells_out[ENG_NUM_RAIL],
    int8_t  rail_to_idx_out[289],
    int8_t  adj_out[ENG_NUM_RAIL * ENG_ADJ_WIDTH])
{
    RailGraph g = build_rail_graph();
    if ((int)g.cells.size() != ENG_NUM_RAIL) {
        // Invariant violation — the header's constant must be kept in sync.
        std::fprintf(stderr,
                     "tables.cu: expected %d rail cells, got %zu\n",
                     (int)ENG_NUM_RAIL, g.cells.size());
        std::abort();
    }

    std::memcpy(rail_to_idx_out, g.flat_to_idx.data(), NC);
    for (int ri = 0; ri < ENG_NUM_RAIL; ++ri)
        rail_cells_out[ri] = g.cells[ri];

    std::fill(adj_out, adj_out + ENG_NUM_RAIL * ENG_ADJ_WIDTH, (int8_t)-1);
    for (int ri = 0; ri < ENG_NUM_RAIL; ++ri) {
        const auto& nbrs = g.adj[ri];
        if ((int)nbrs.size() > ENG_ADJ_WIDTH) {
            std::fprintf(stderr,
                         "tables.cu: rail node %d has degree %zu > %d\n",
                         ri, nbrs.size(), ENG_ADJ_WIDTH);
            std::abort();
        }
        for (int k = 0; k < (int)nbrs.size(); ++k)
            adj_out[ri * ENG_ADJ_WIDTH + k] = nbrs[k];
    }
}

// ---------------------------------------------------------------------------
// init_tables(): build all tables on host, upload to GPU
// ---------------------------------------------------------------------------
void init_tables() {
    ensure_props();

    // --- Constant memory: board topology ---
    {
        int16_t adj[289 * 8];
        build_adjacent_cells(adj);
        CUDA_CHECK(cudaMemcpyToSymbol(ADJACENT_CELLS, adj, sizeof(adj)));
    }
    {
        CUDA_CHECK(cudaMemcpyToSymbol(CAMP_FLAT,       g_props.is_camp,       289));
        CUDA_CHECK(cudaMemcpyToSymbol(STRONGHOLD_FLAT, g_props.is_stronghold, 289));
        CUDA_CHECK(cudaMemcpyToSymbol(RAIL_FLAT,       g_props.is_railway,    289));
        CUDA_CHECK(cudaMemcpyToSymbol(ON_BOARD_FLAT,   g_props.on_board,      289));
        CUDA_CHECK(cudaMemcpyToSymbol(NINE_GRID_FLAT,  g_props.is_nine_grid,  289));
        CUDA_CHECK(cudaMemcpyToSymbol(CURVE_RAIL_OF,   g_props.curve_rail,    289));
    }
    {
        // piece_slot plane index per seat slot; camp slots get -1. Derived
        // from the same CAMP_IDX the board properties are built from.
        int8_t slot_ch[SLOTS_PER_SEAT_DEV];
        int8_t next_ch = 0;
        for (int i = 0; i < SLOTS_PER_SEAT_DEV; ++i) {
            bool is_camp_slot = false;
            for (int ci : CAMP_IDX) if (i == ci) { is_camp_slot = true; break; }
            slot_ch[i] = is_camp_slot ? (int8_t)-1 : next_ch++;
        }
        CUDA_CHECK(cudaMemcpyToSymbol(SLOT_TO_CHANNEL, slot_ch, sizeof(slot_ch)));
    }
    {
        // The 8 cells adjoining an arc link, derived from the same SPC_EDGES
        // the rail graph is built from.
        bool arc[289] = {false};
        for (const auto& e : SPC_EDGES) {
            arc[flat(e.first.first,  e.first.second)]  = true;
            arc[flat(e.second.first, e.second.second)] = true;
        }
        CUDA_CHECK(cudaMemcpyToSymbol(CURVE_ARC_FLAT, arc, sizeof(arc)));
    }
    {
        static int16_t rays[289 * 4 * STRAIGHT_RAY_LEN];
        build_straight_rail_rays(rays);
        CUDA_CHECK(cudaMemcpyToSymbol(STRAIGHT_RAIL_RAYS, rays, sizeof(rays)));
    }
    {
        int16_t rail_cells[ENG_NUM_RAIL];
        int8_t  rail_to_idx[289];
        int8_t  adj[ENG_NUM_RAIL * ENG_ADJ_WIDTH];
        build_engineer_tables(rail_cells, rail_to_idx, adj);
        CUDA_CHECK(cudaMemcpyToSymbol(ENG_RAIL_CELLS,  rail_cells,  sizeof(rail_cells)));
        CUDA_CHECK(cudaMemcpyToSymbol(ENG_RAIL_TO_IDX, rail_to_idx, sizeof(rail_to_idx)));
        CUDA_CHECK(cudaMemcpyToSymbol(ENG_RAIL_ADJ,    adj,         sizeof(adj)));
    }

    // --- Device heap: Zobrist tables ---
    //
    // Sizes must match junqi_core/_zobrist.py exactly.  The tables are
    // initialized with an MT19937_64 stream so standalone C++ tests have
    // *some* deterministic values, but for GPU/CPU parity callers MUST
    // invoke ``upload_zobrist_tables_from_host`` after ``init_tables``
    // with the CPU-seeded numpy arrays (seed "JUNQI_ZO").  Python's
    // ``_cuda.init_tables()`` wrapper does this automatically.
    constexpr size_t Z_PIECE               = 120 * 14 * 289;   // 485,640
    constexpr size_t Z_TURN                = 4;
    constexpr size_t Z_MOVE_COUNTER        = ZOB_MOVE_COUNTER_MASK + 1;         // 4096
    constexpr size_t Z_MOVES_SINCE_COMBAT  = ZOB_MOVES_SINCE_COMBAT_MASK + 1;   // 512
    constexpr size_t Z_WINNER              = 3;
    constexpr size_t Z_SEAT_DEAD           = 4;
    constexpr size_t Z_SEAT_FLAG_REVEALED  = 4;

    int64_t* dp_piece        = nullptr;
    int64_t* dp_turn         = nullptr;
    int64_t* dp_mc           = nullptr;
    int64_t* dp_mslc         = nullptr;
    int64_t* dp_winner       = nullptr;
    int64_t* dp_seat_dead    = nullptr;
    int64_t* dp_seat_flag_rev = nullptr;

    CUDA_CHECK(cudaMalloc(&dp_piece,       Z_PIECE               * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_turn,        Z_TURN                * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_mc,          Z_MOVE_COUNTER        * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_mslc,        Z_MOVES_SINCE_COMBAT  * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_winner,      Z_WINNER              * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_seat_dead,   Z_SEAT_DEAD           * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dp_seat_flag_rev, Z_SEAT_FLAG_REVEALED * sizeof(int64_t)));

    {
        std::mt19937_64 rng(0xDEADBEEFCAFE1234ULL);
        auto fill = [&](int64_t* dev_ptr, size_t n) {
            int64_t* buf = new int64_t[n];
            for (size_t i = 0; i < n; ++i) buf[i] = (int64_t)rng();
            CUDA_CHECK(cudaMemcpy(dev_ptr, buf, n * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
            delete[] buf;
        };
        fill(dp_piece,         Z_PIECE);
        fill(dp_turn,          Z_TURN);
        fill(dp_mc,            Z_MOVE_COUNTER);
        fill(dp_mslc,          Z_MOVES_SINCE_COMBAT);
        fill(dp_winner,        Z_WINNER);
        fill(dp_seat_dead,     Z_SEAT_DEAD);
        fill(dp_seat_flag_rev, Z_SEAT_FLAG_REVEALED);

        // scalars
        int64_t scalar_terminated = (int64_t)rng();
        int64_t scalar_draw       = (int64_t)rng();
        CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_terminated,
                                      &scalar_terminated, sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_draw,
                                      &scalar_draw,       sizeof(int64_t)));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_piece,              &dp_piece,        sizeof(dp_piece)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_turn,               &dp_turn,         sizeof(dp_turn)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_move_counter,       &dp_mc,           sizeof(dp_mc)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_moves_since_combat, &dp_mslc,         sizeof(dp_mslc)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_winner,             &dp_winner,       sizeof(dp_winner)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_seat_dead,          &dp_seat_dead,    sizeof(dp_seat_dead)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_seat_flag_revealed, &dp_seat_flag_rev, sizeof(dp_seat_flag_rev)));

    h_d_zobrist_piece              = dp_piece;
    h_d_zobrist_turn               = dp_turn;
    h_d_zobrist_move_counter       = dp_mc;
    h_d_zobrist_moves_since_combat = dp_mslc;
    h_d_zobrist_winner             = dp_winner;
    h_d_zobrist_seat_dead          = dp_seat_dead;
    h_d_zobrist_seat_flag_revealed = dp_seat_flag_rev;
}

void cleanup_tables() {
    auto free_if = [](int64_t*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_if(h_d_zobrist_piece);
    free_if(h_d_zobrist_turn);
    free_if(h_d_zobrist_move_counter);
    free_if(h_d_zobrist_moves_since_combat);
    free_if(h_d_zobrist_winner);
    free_if(h_d_zobrist_seat_dead);
    free_if(h_d_zobrist_seat_flag_revealed);
}

// ---------------------------------------------------------------------------
// upload_zobrist_tables_from_host
//
// Overwrites the device-side zobrist tables with the CPU-seeded arrays from
// _zobrist.py.  ``init_tables`` must be called first (so the device
// allocations exist).  After this returns the device tables are byte-
// identical to the Python-side arrays and GPU/CPU hashes will match.
// ---------------------------------------------------------------------------
void upload_zobrist_tables_from_host(
    const int64_t* h_piece,
    const int64_t* h_turn,
    const int64_t* h_move_counter,
    const int64_t* h_moves_since_combat,
    const int64_t* h_winner,
    int64_t        h_terminated,
    int64_t        h_draw,
    const int64_t* h_seat_dead,
    const int64_t* h_seat_flag_revealed)
{
    if (!h_d_zobrist_piece) {
        std::fprintf(stderr,
                     "upload_zobrist_tables_from_host: init_tables() must be "
                     "called first\n");
        std::abort();
    }
    constexpr size_t Z_PIECE              = 120 * 14 * 289;
    constexpr size_t Z_TURN               = 4;
    constexpr size_t Z_MOVE_COUNTER       = ZOB_MOVE_COUNTER_MASK + 1;
    constexpr size_t Z_MOVES_SINCE_COMBAT = ZOB_MOVES_SINCE_COMBAT_MASK + 1;
    constexpr size_t Z_WINNER             = 3;
    constexpr size_t Z_SEAT_DEAD          = 4;
    constexpr size_t Z_SEAT_FLAG          = 4;

    CUDA_CHECK(cudaMemcpy(h_d_zobrist_piece,              h_piece,
                          Z_PIECE * sizeof(int64_t),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_turn,               h_turn,
                          Z_TURN  * sizeof(int64_t),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_move_counter,       h_move_counter,
                          Z_MOVE_COUNTER * sizeof(int64_t),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_moves_since_combat, h_moves_since_combat,
                          Z_MOVES_SINCE_COMBAT * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_winner,             h_winner,
                          Z_WINNER * sizeof(int64_t),             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_seat_dead,          h_seat_dead,
                          Z_SEAT_DEAD * sizeof(int64_t),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_d_zobrist_seat_flag_revealed, h_seat_flag_revealed,
                          Z_SEAT_FLAG * sizeof(int64_t),          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_terminated,
                                  &h_terminated, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zobrist_draw,
                                  &h_draw,       sizeof(int64_t)));
}

}  // namespace junqi_cuda
