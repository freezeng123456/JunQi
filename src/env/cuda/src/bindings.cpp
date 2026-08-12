/*
 * bindings.cpp
 * PyBind11 Python bindings for the JunQi CUDA backend.
 *
 * Exposed API:
 *   junqi_cuda.init_tables()
 *   junqi_cuda.cleanup_tables()
 *   junqi_cuda.get_gpu_count() -> int
 *   junqi_cuda.set_device(device_id: int)
 *
 *   class DeviceGameStateBatch:
 *     __init__(num_envs: int)
 *     num_envs: int  (read-only)
 *     copy_from_host(arrays: dict[str, np.ndarray])
 *     copy_to_host() -> dict[str, np.ndarray]
 *
 *   class DeviceObservationBatch:
 *     __init__(num_envs: int)
 *     num_envs: int  (read-only)
 *     copy_to_host() -> tuple[np.ndarray, np.ndarray]  (spatial, global)
 *
 *   build_observation_batch(state, beliefs, observer_seats, obs_out, show_mode=2)
 *   legal_action_ids_batch(state, acting_seats) -> (action_ids, counts)
 *
 * The bindings use numpy arrays as the host-side exchange format.
 * All arrays are expected to be C-contiguous and have the correct dtype.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "junqi_cuda.h"

namespace py = pybind11;
using namespace junqi_cuda;

// Local CUDA error guard that raises Python exceptions.
#define CUDA_CHECK_PY(expr) do {                                               \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
        throw std::runtime_error(std::string("CUDA error: ")                   \
                                 + cudaGetErrorString(_err));                  \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// Helper: require C-contiguous numpy array of given dtype and size.
// ---------------------------------------------------------------------------
template<typename T>
static const T* require_array(const py::array_t<T>& arr, size_t expected_size, const char* name) {
    if (!(arr.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    if ((size_t)arr.size() != expected_size)
        throw std::runtime_error(std::string(name) + ": expected " +
            std::to_string(expected_size) + " elements, got " + std::to_string(arr.size()));
    return arr.data();
}

template<typename T>
static T* require_mutable_array(py::array_t<T>& arr, size_t expected_size, const char* name) {
    if (!(arr.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    if ((size_t)arr.size() != expected_size)
        throw std::runtime_error(std::string(name) + ": expected " +
            std::to_string(expected_size) + " elements, got " + std::to_string(arr.size()));
    return arr.mutable_data();
}

// ---------------------------------------------------------------------------
// copy_from_host: accept a dict of named numpy arrays
// ---------------------------------------------------------------------------
static void py_copy_from_host(DeviceGameStateBatch& self, py::dict arrays) {
    const int n = self.num_envs;
    auto get16 = [&](const char* k) -> const int16_t* {
        return require_array<int16_t>(arrays[k].cast<py::array_t<int16_t>>(), (size_t)n * 120, k);
    };
    auto get8 = [&](const char* k, size_t count) -> const int8_t* {
        return require_array<int8_t>(arrays[k].cast<py::array_t<int8_t>>(), (size_t)n * count, k);
    };
    auto getbool = [&](const char* k, size_t count) -> const bool* {
        return require_array<bool>(arrays[k].cast<py::array_t<bool>>(), (size_t)n * count, k);
    };

    self.copy_from_host(
        get16("cell_piece_id_per_piece"),
        get8("piece_seat_arr", 120),
        get8("piece_type_arr", 120),
        getbool("alive", 120),
        get8("pos_x", 120),
        get8("pos_y", 120),
        get8("zero_x", 120),
        get8("zero_y", 120),
        get16("move_count_arr"),
        get16("active_eat_arr"),
        get16("passive_surv_arr"),
        get8("death_reason_arr", 120),
        get16("death_step_arr"),
        get16("death_loc_flat_arr"),
        require_array<int16_t>(arrays["cell_piece_id"].cast<py::array_t<int16_t>>(),
                               (size_t)n * 289, "cell_piece_id"),
        getbool("seat_dead_arr", 4),
        getbool("seat_flag_revealed_arr", 4),
        get8("turn", 1),
        require_array<int64_t>(arrays["zobrist"].cast<py::array_t<int64_t>>(),
                               (size_t)n, "zobrist"),
        require_array<int32_t>(arrays["move_counter"].cast<py::array_t<int32_t>>(),
                               (size_t)n, "move_counter"),
        require_array<int32_t>(arrays["moves_since_last_combat"].cast<py::array_t<int32_t>>(),
                               (size_t)n, "moves_since_last_combat")
    );
}

// Fast-path H2D for legal_action_ids_batch: only the 6 arrays the M2 kernel reads.
// Uses one fused H2D transfer + device scatter kernel.
static void py_copy_from_host_legal_lite(
    DeviceGameStateBatch& self,
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> piece_seat_arr,
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> piece_type_arr,
    py::array_t<bool,    py::array::c_style | py::array::forcecast> alive,
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> pos_x,
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> pos_y,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> cell_piece_id)
{
    const int n = self.num_envs;
    self.copy_from_host_legal_lite(
        require_array<int8_t>(piece_seat_arr, (size_t)n * 120, "piece_seat_arr"),
        require_array<int8_t>(piece_type_arr, (size_t)n * 120, "piece_type_arr"),
        require_array<bool>  (alive,          (size_t)n * 120, "alive"),
        require_array<int8_t>(pos_x,          (size_t)n * 120, "pos_x"),
        require_array<int8_t>(pos_y,          (size_t)n * 120, "pos_y"),
        require_array<int16_t>(cell_piece_id, (size_t)n * 289, "cell_piece_id")
    );
}

// ---------------------------------------------------------------------------
// copy_to_host: return a dict of named numpy arrays
// ---------------------------------------------------------------------------
static py::dict py_copy_to_host(const DeviceGameStateBatch& self) {
    const int n = self.num_envs;

    // Allocate output arrays.
    auto mk16p = [&](size_t count) { return py::array_t<int16_t>((size_t)n * count); };
    auto mk8p  = [&](size_t count) { return py::array_t<int8_t>((size_t)n * count); };
    auto mkb   = [&](size_t count) { return py::array_t<bool>((size_t)n * count); };
    auto mk64  = [&](size_t count) { return py::array_t<int64_t>((size_t)n * count); };
    auto mk32  = [&](size_t count) { return py::array_t<int32_t>((size_t)n * count); };

    auto a_cpip = mk16p(120); auto a_psa  = mk8p(120);
    auto a_pta  = mk8p(120);  auto a_alv  = mkb(120);
    auto a_px   = mk8p(120);  auto a_py   = mk8p(120);
    auto a_zx   = mk8p(120);  auto a_zy   = mk8p(120);
    auto a_mca  = mk16p(120); auto a_aea  = mk16p(120);
    auto a_psa2 = mk16p(120); auto a_dra  = mk8p(120);
    auto a_dsa  = mk16p(120); auto a_dlfa = mk16p(120);
    auto a_cpi  = mk16p(289); auto a_sda  = mkb(4);
    auto a_sfra = mkb(4);     auto a_turn = mk8p(1);
    auto a_zob  = mk64(1);    auto a_mc   = mk32(1);
    auto a_mslc = mk32(1);

    self.copy_to_host(
        a_cpip.mutable_data(), a_psa.mutable_data(),
        a_pta.mutable_data(),  a_alv.mutable_data(),
        a_px.mutable_data(),   a_py.mutable_data(),
        a_zx.mutable_data(),   a_zy.mutable_data(),
        a_mca.mutable_data(),  a_aea.mutable_data(),
        a_psa2.mutable_data(), a_dra.mutable_data(),
        a_dsa.mutable_data(),  a_dlfa.mutable_data(),
        a_cpi.mutable_data(),  a_sda.mutable_data(),
        a_sfra.mutable_data(), a_turn.mutable_data(),
        a_zob.mutable_data(),  a_mc.mutable_data(),
        a_mslc.mutable_data()
    );

    py::dict d;
    d["cell_piece_id_per_piece"]   = a_cpip;
    d["piece_seat_arr"]            = a_psa;
    d["piece_type_arr"]            = a_pta;
    d["alive"]                     = a_alv;
    d["pos_x"]                     = a_px;
    d["pos_y"]                     = a_py;
    d["zero_x"]                    = a_zx;
    d["zero_y"]                    = a_zy;
    d["move_count_arr"]            = a_mca;
    d["active_eat_arr"]            = a_aea;
    d["passive_surv_arr"]          = a_psa2;
    d["death_reason_arr"]          = a_dra;
    d["death_step_arr"]            = a_dsa;
    d["death_loc_flat_arr"]        = a_dlfa;
    d["cell_piece_id"]             = a_cpi;
    d["seat_dead_arr"]             = a_sda;
    d["seat_flag_revealed_arr"]    = a_sfra;
    d["turn"]                      = a_turn;
    d["zobrist"]                   = a_zob;
    d["move_counter"]              = a_mc;
    d["moves_since_last_combat"]   = a_mslc;
    return d;
}

// Fast-path: copy only the per-env acting-turn vector (N int8).  Useful for
// PPO-style loops where only the current seat is needed to slice obs and
// build the legal mask.
static py::array_t<int8_t> py_copy_turn_to_host(const DeviceGameStateBatch& self) {
    const int n = self.num_envs;
    auto turn = py::array_t<int8_t>((size_t)n);
    CUDA_CHECK_PY(cudaMemcpy(turn.mutable_data(), self.d_turn,
                             n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    return turn;
}

// ---------------------------------------------------------------------------
// Termination-state upload/download helpers (Phase 1b).  These fields are
// GPU-resident once initialized and mutate only via step_batch; callers push
// their initial values (all zeros for fresh games) and optionally pull them
// back to verify parity.
// ---------------------------------------------------------------------------
static void py_copy_termination_from_host(
    DeviceGameStateBatch& self,
    py::array_t<bool,   py::array::c_style | py::array::forcecast> terminated,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> winner_team,
    py::array_t<bool,   py::array::c_style | py::array::forcecast> draw)
{
    const int n = self.num_envs;
    if (terminated.size()  != (ssize_t)n) throw std::runtime_error("terminated size mismatch");
    if (winner_team.size() != (ssize_t)n) throw std::runtime_error("winner_team size mismatch");
    if (draw.size()        != (ssize_t)n) throw std::runtime_error("draw size mismatch");
    CUDA_CHECK_PY(cudaMemcpy(self.d_terminated,  terminated.data(),  n * sizeof(bool),   cudaMemcpyHostToDevice));
    CUDA_CHECK_PY(cudaMemcpy(self.d_winner_team, winner_team.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_PY(cudaMemcpy(self.d_draw,        draw.data(),        n * sizeof(bool),   cudaMemcpyHostToDevice));
}

static py::dict py_copy_termination_to_host(const DeviceGameStateBatch& self) {
    const int n = self.num_envs;
    auto terminated  = py::array_t<bool>((size_t)n);
    auto winner_team = py::array_t<int8_t>((size_t)n);
    auto draw        = py::array_t<bool>((size_t)n);
    CUDA_CHECK_PY(cudaMemcpy(terminated.mutable_data(),  self.d_terminated,  n * sizeof(bool),   cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(winner_team.mutable_data(), self.d_winner_team, n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(draw.mutable_data(),        self.d_draw,        n * sizeof(bool),   cudaMemcpyDeviceToHost));
    py::dict d;
    d["terminated"]  = terminated;
    d["winner_team"] = winner_team;
    d["draw"]        = draw;
    return d;
}

// ---------------------------------------------------------------------------
// CombatMemory v6 host↔device parity helpers.
//
// IMPORTANT: these functions are for PARITY TESTING only.  In the
// production training hot path, CombatMemory state lives entirely on the
// device — step_batch_kernel mutates it, observation_kernel reads it,
// and no per-step host transfer ever happens.  The two functions below
// are invoked from tests/test_gpu_combat_memory_parity.py to verify
// that the GPU update matches the CPU reference bit-for-bit.
// ---------------------------------------------------------------------------
static void py_cm_copy_from_host(
    DeviceGameStateBatch& self,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> direct_lo,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> direct_hi,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> direct_type,
    py::array_t<int16_t,  py::array::c_style | py::array::forcecast> last_direct_step,
    py::array_t<int16_t,  py::array::c_style | py::array::forcecast> direct_other_count,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> chain_lo,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> chain_hi,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> chain_type,
    py::array_t<int16_t,  py::array::c_style | py::array::forcecast> last_chain_step,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> eaten_by_pid_lo,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> eaten_by_pid_hi,
    py::array_t<int8_t,   py::array::c_style | py::array::forcecast> rank_floor,
    py::array_t<int16_t,  py::array::c_style | py::array::forcecast> rank_floor_step,
    py::array_t<bool,     py::array::c_style | py::array::forcecast> is_gongb,
    py::array_t<bool,     py::array::c_style | py::array::forcecast> not_gongb,
    py::array_t<bool,     py::array::c_style | py::array::forcecast> attacked_by_known_gongb)
{
    const size_t n = (size_t)self.num_envs;
    const size_t cm_n = n * 4 * 120;
    auto check = [&](ssize_t got, const char* name) {
        if ((size_t)got != cm_n)
            throw std::runtime_error(std::string("cm size mismatch: ") + name);
    };
    check(direct_lo.size(),               "direct_lo");
    check(direct_hi.size(),               "direct_hi");
    check(direct_type.size(),             "direct_type");
    check(last_direct_step.size(),        "last_direct_step");
    check(direct_other_count.size(),      "direct_other_count");
    check(chain_lo.size(),                "chain_lo");
    check(chain_hi.size(),                "chain_hi");
    check(chain_type.size(),              "chain_type");
    check(last_chain_step.size(),         "last_chain_step");
    check(eaten_by_pid_lo.size(),         "eaten_by_pid_lo");
    check(eaten_by_pid_hi.size(),         "eaten_by_pid_hi");
    check(rank_floor.size(),              "rank_floor");
    check(rank_floor_step.size(),         "rank_floor_step");
    check(is_gongb.size(),                "is_gongb");
    check(not_gongb.size(),               "not_gongb");
    check(attacked_by_known_gongb.size(), "attacked_by_known_gongb");

#define CM_H2D(devptr, host, T) \
    CUDA_CHECK_PY(cudaMemcpy(self.devptr, host.data(), cm_n * sizeof(T), \
                             cudaMemcpyHostToDevice))
    CM_H2D(d_cm_direct_lo,                 direct_lo,               uint64_t);
    CM_H2D(d_cm_direct_hi,                 direct_hi,               uint64_t);
    CM_H2D(d_cm_direct_type,               direct_type,             uint16_t);
    CM_H2D(d_cm_last_direct_step,          last_direct_step,        int16_t);
    CM_H2D(d_cm_direct_other_count,        direct_other_count,      int16_t);
    CM_H2D(d_cm_chain_lo,                  chain_lo,                uint64_t);
    CM_H2D(d_cm_chain_hi,                  chain_hi,                uint64_t);
    CM_H2D(d_cm_chain_type,                chain_type,              uint16_t);
    CM_H2D(d_cm_last_chain_step,           last_chain_step,         int16_t);
    CM_H2D(d_cm_eaten_by_pid_lo,           eaten_by_pid_lo,         uint64_t);
    CM_H2D(d_cm_eaten_by_pid_hi,           eaten_by_pid_hi,         uint64_t);
    CM_H2D(d_cm_rank_floor,                rank_floor,              int8_t);
    CM_H2D(d_cm_rank_floor_step,           rank_floor_step,         int16_t);
    CM_H2D(d_cm_is_gongb,                  is_gongb,                bool);
    CM_H2D(d_cm_not_gongb,                 not_gongb,               bool);
    CM_H2D(d_cm_attacked_by_known_gongb,   attacked_by_known_gongb, bool);
#undef CM_H2D
}

static py::dict py_cm_copy_to_host(const DeviceGameStateBatch& self) {
    const size_t n = (size_t)self.num_envs;
    const size_t cm_n = n * 4 * 120;
    auto a_dlo  = py::array_t<uint64_t>(cm_n);
    auto a_dhi  = py::array_t<uint64_t>(cm_n);
    auto a_dty  = py::array_t<uint16_t>(cm_n);
    auto a_lds  = py::array_t<int16_t>(cm_n);
    auto a_doc  = py::array_t<int16_t>(cm_n);
    auto a_clo  = py::array_t<uint64_t>(cm_n);
    auto a_chi  = py::array_t<uint64_t>(cm_n);
    auto a_cty  = py::array_t<uint16_t>(cm_n);
    auto a_lcs  = py::array_t<int16_t>(cm_n);
    auto a_eblo = py::array_t<uint64_t>(cm_n);
    auto a_ebhi = py::array_t<uint64_t>(cm_n);
    auto a_rf   = py::array_t<int8_t>(cm_n);
    auto a_rfs  = py::array_t<int16_t>(cm_n);
    auto a_isg  = py::array_t<bool>(cm_n);
    auto a_nog  = py::array_t<bool>(cm_n);
    auto a_atk  = py::array_t<bool>(cm_n);

#define CM_D2H(host, devptr, T) \
    CUDA_CHECK_PY(cudaMemcpy(host.mutable_data(), self.devptr, cm_n * sizeof(T), \
                             cudaMemcpyDeviceToHost))
    CM_D2H(a_dlo,  d_cm_direct_lo,                 uint64_t);
    CM_D2H(a_dhi,  d_cm_direct_hi,                 uint64_t);
    CM_D2H(a_dty,  d_cm_direct_type,               uint16_t);
    CM_D2H(a_lds,  d_cm_last_direct_step,          int16_t);
    CM_D2H(a_doc,  d_cm_direct_other_count,        int16_t);
    CM_D2H(a_clo,  d_cm_chain_lo,                  uint64_t);
    CM_D2H(a_chi,  d_cm_chain_hi,                  uint64_t);
    CM_D2H(a_cty,  d_cm_chain_type,                uint16_t);
    CM_D2H(a_lcs,  d_cm_last_chain_step,           int16_t);
    CM_D2H(a_eblo, d_cm_eaten_by_pid_lo,            uint64_t);
    CM_D2H(a_ebhi, d_cm_eaten_by_pid_hi,            uint64_t);
    CM_D2H(a_rf,   d_cm_rank_floor,                int8_t);
    CM_D2H(a_rfs,  d_cm_rank_floor_step,           int16_t);
    CM_D2H(a_isg,  d_cm_is_gongb,                  bool);
    CM_D2H(a_nog,  d_cm_not_gongb,                 bool);
    CM_D2H(a_atk,  d_cm_attacked_by_known_gongb,   bool);
#undef CM_D2H

    py::dict d;
    d["direct_lo"]                 = a_dlo;
    d["direct_hi"]                 = a_dhi;
    d["direct_type"]               = a_dty;
    d["last_direct_step"]          = a_lds;
    d["direct_other_count"]        = a_doc;
    d["chain_lo"]                  = a_clo;
    d["chain_hi"]                  = a_chi;
    d["chain_type"]                = a_cty;
    d["last_chain_step"]           = a_lcs;
    d["eaten_by_pid_lo"]           = a_eblo;
    d["eaten_by_pid_hi"]           = a_ebhi;
    d["rank_floor"]                = a_rf;
    d["rank_floor_step"]           = a_rfs;
    d["is_gongb"]                  = a_isg;
    d["not_gongb"]                 = a_nog;
    d["attacked_by_known_gongb"]   = a_atk;
    return d;
}

// ---------------------------------------------------------------------------
// step_batch — launch kernel and return MoveResultBatch as dict of numpy arrays.
// ---------------------------------------------------------------------------
static py::dict py_step_batch(
    DeviceGameStateBatch& self,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> action_ids)
{
    const int n = self.num_envs;
    if (action_ids.size() != (ssize_t)n)
        throw std::runtime_error("action_ids size must equal num_envs");

    // Upload actions.
    int32_t* d_action_ids = nullptr;
    CUDA_CHECK_PY(cudaMalloc(&d_action_ids, n * sizeof(int32_t)));
    CUDA_CHECK_PY(cudaMemcpy(d_action_ids, action_ids.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Allocate output device buffers.
    bool*   d_valid  = nullptr;  int8_t* d_event  = nullptr;
    bool*   d_term   = nullptr;  int8_t* d_win    = nullptr;
    bool*   d_draw_o = nullptr;  bool*   d_flagc  = nullptr;
    CUDA_CHECK_PY(cudaMalloc(&d_valid,  n * sizeof(bool)));
    CUDA_CHECK_PY(cudaMalloc(&d_event,  n * sizeof(int8_t)));
    CUDA_CHECK_PY(cudaMalloc(&d_term,   n * sizeof(bool)));
    CUDA_CHECK_PY(cudaMalloc(&d_win,    n * sizeof(int8_t)));
    CUDA_CHECK_PY(cudaMalloc(&d_draw_o, n * sizeof(bool)));
    CUDA_CHECK_PY(cudaMalloc(&d_flagc,  n * sizeof(bool)));

    step_batch(self, d_action_ids,
               d_valid, d_event, d_term, d_win, d_draw_o, d_flagc, /*stream_id=*/0);

    // D2H.
    auto h_valid = py::array_t<bool>((size_t)n);
    auto h_event = py::array_t<int8_t>((size_t)n);
    auto h_term  = py::array_t<bool>((size_t)n);
    auto h_win   = py::array_t<int8_t>((size_t)n);
    auto h_draw  = py::array_t<bool>((size_t)n);
    auto h_flagc = py::array_t<bool>((size_t)n);
    CUDA_CHECK_PY(cudaMemcpy(h_valid.mutable_data(), d_valid, n * sizeof(bool),   cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(h_event.mutable_data(), d_event, n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(h_term.mutable_data(),  d_term,  n * sizeof(bool),   cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(h_win.mutable_data(),   d_win,   n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(h_draw.mutable_data(),  d_draw_o,n * sizeof(bool),   cudaMemcpyDeviceToHost));
    CUDA_CHECK_PY(cudaMemcpy(h_flagc.mutable_data(), d_flagc, n * sizeof(bool),   cudaMemcpyDeviceToHost));

    cudaFree(d_action_ids);
    cudaFree(d_valid); cudaFree(d_event); cudaFree(d_term);
    cudaFree(d_win);   cudaFree(d_draw_o); cudaFree(d_flagc);

    py::dict d;
    d["valid"]          = h_valid;
    d["event"]          = h_event;
    d["terminated"]     = h_term;
    d["winner_team"]    = h_win;
    d["draw"]           = h_draw;
    d["flag_captured"]  = h_flagc;
    return d;
}


// ---------------------------------------------------------------------------
// DeviceObservationBatch: copy_to_host → (spatial, global) numpy arrays
// ---------------------------------------------------------------------------
static py::tuple py_obs_copy_to_host(const DeviceObservationBatch& self) {
    const int n = self.num_envs;
    auto spatial = py::array_t<float>({
        (ssize_t)n, (ssize_t)NUM_SEATS,
        (ssize_t)NUM_OBS_CHANNELS, (ssize_t)BOARD_SIZE, (ssize_t)BOARD_SIZE});
    auto global_ = py::array_t<float>({
        (ssize_t)n, (ssize_t)NUM_SEATS, (ssize_t)NUM_GLOBAL_DIMS});
    self.copy_to_host(spatial.mutable_data(), global_.mutable_data());
    return py::make_tuple(spatial, global_);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(junqi_cuda, m) {
    m.doc() = "JunQi GPU-accelerated game engine backend";

    // --- Constants ---
    m.attr("NUM_ENVS_MAX")       = NUM_ENVS_MAX;
    m.attr("NUM_PIECES")         = NUM_PIECES;
    m.attr("NUM_CELLS")          = NUM_CELLS;
    m.attr("BOARD_SIZE")         = BOARD_SIZE;
    m.attr("NUM_SEATS")          = NUM_SEATS;
    m.attr("NUM_OBS_CHANNELS")   = NUM_OBS_CHANNELS;
    m.attr("NUM_GLOBAL_DIMS")    = NUM_GLOBAL_DIMS;
    m.attr("FLAT_ACTION_SPACE")  = FLAT_ACTION_SPACE;
    m.attr("NUM_TRACKED_TYPES")  = NUM_TRACKED_TYPES;
    m.attr("SLOTS_PER_PIECE")    = SLOTS_PER_PIECE;
    m.attr("COMPACT_ACTION_SPACE") = NUM_PIECES * SLOTS_PER_PIECE;  // 120 × 32 = 3840

    // --- Global functions ---
    m.def("init_tables",    &init_tables,
          "Upload board topology and Zobrist tables to GPU constant/device memory.");
    m.def("cleanup_tables", &cleanup_tables,
          "Free GPU-resident Zobrist heap allocations.");
    m.def("get_gpu_count",  &get_gpu_count,
          "Return the number of available CUDA-capable GPUs.");
    m.def("set_device",     &set_device, py::arg("device_id"),
          "Select the active CUDA device.");

    // Upload CPU-seeded zobrist tables so GPU and CPU hashes match bit-for-bit.
    m.def("upload_zobrist_tables",
        [](py::array_t<int64_t, py::array::c_style | py::array::forcecast> piece,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> turn,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> move_counter,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> moves_since_combat,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> winner,
           int64_t terminated,
           int64_t draw,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> seat_dead,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> seat_flag_revealed)
        {
            // Shape sanity (assertions on size, layout is C-contiguous from forcecast).
            if (piece.size()              != 120 * 14 * 289) throw std::runtime_error("zobrist piece size mismatch");
            if (turn.size()               != 4)              throw std::runtime_error("zobrist turn size mismatch");
            if (move_counter.size()       != 4096)           throw std::runtime_error("zobrist move_counter size mismatch");
            if (moves_since_combat.size() != 512)            throw std::runtime_error("zobrist moves_since_combat size mismatch");
            if (winner.size()             != 3)              throw std::runtime_error("zobrist winner size mismatch");
            if (seat_dead.size()          != 4)              throw std::runtime_error("zobrist seat_dead size mismatch");
            if (seat_flag_revealed.size() != 4)              throw std::runtime_error("zobrist seat_flag_revealed size mismatch");
            upload_zobrist_tables_from_host(
                piece.data(), turn.data(), move_counter.data(),
                moves_since_combat.data(), winner.data(),
                terminated, draw,
                seat_dead.data(), seat_flag_revealed.data());
        },
        py::arg("piece"), py::arg("turn"), py::arg("move_counter"),
        py::arg("moves_since_combat"), py::arg("winner"),
        py::arg("terminated"), py::arg("draw"),
        py::arg("seat_dead"), py::arg("seat_flag_revealed"),
        "Upload CPU-seeded zobrist tables (ZOB_*) from _zobrist.py. "
        "Must be called after init_tables() for GPU/CPU hash parity.");


    // --- DeviceGameStateBatch ---
    py::class_<DeviceGameStateBatch>(m, "DeviceGameStateBatch",
        "GPU-resident batch of game states (SoA layout).")
        .def(py::init<int>(), py::arg("num_envs"),
             "Allocate device memory for num_envs game states.")
        .def_readonly("num_envs", &DeviceGameStateBatch::num_envs)
        // --- Phase 2: device pointer accessors for zero-copy torch views ---
        .def_property_readonly("d_turn_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_turn);
            }, "Raw device pointer to d_turn (int8, N). For zero-copy torch view.")
        .def_property_readonly("d_terminated_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_terminated);
            }, "Raw device pointer to d_terminated (bool, N).")
        .def_property_readonly("d_winner_team_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_winner_team);
            }, "Raw device pointer to d_winner_team (int8, N).")
        .def_property_readonly("d_draw_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_draw);
            }, "Raw device pointer to d_draw (bool, N).")
        .def_property_readonly("d_piece_seat_arr_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_piece_seat_arr);
            }, "Raw device pointer to d_piece_seat_arr (int8, N*120).")
        .def_property_readonly("d_piece_type_arr_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_piece_type_arr);
            }, "Raw device pointer to d_piece_type_arr (int8, N*120).")
        .def_property_readonly("d_alive_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_alive);
            }, "Raw device pointer to d_alive (bool, N*120).")
        .def_property_readonly("d_pos_x_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_pos_x);
            }, "Raw device pointer to d_pos_x (int8, N*120).")
        .def_property_readonly("d_pos_y_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_pos_y);
            }, "Raw device pointer to d_pos_y (int8, N*120).")
        .def_property_readonly("d_cell_piece_id_ptr",
            [](const DeviceGameStateBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_cell_piece_id);
            }, "Raw device pointer to d_cell_piece_id (int16, N*289).")
        .def("copy_from_host", &py_copy_from_host,
             py::arg("arrays"),
             R"doc(
Copy host numpy arrays into device memory.

Parameters
----------
arrays : dict[str, np.ndarray]
    Keys: cell_piece_id_per_piece (int16, N×120), piece_seat_arr (int8, N×120),
    piece_type_arr (int8, N×120), alive (bool, N×120), pos_x (int8, N×120),
    pos_y (int8, N×120), zero_x (int8, N×120), zero_y (int8, N×120),
    move_count_arr (int16, N×120), active_eat_arr (int16, N×120),
    passive_surv_arr (int16, N×120), death_reason_arr (int8, N×120),
    death_step_arr (int16, N×120), death_loc_flat_arr (int16, N×120),
    cell_piece_id (int16, N×289), seat_dead_arr (bool, N×4),
    seat_flag_revealed_arr (bool, N×4), turn (int8, N),
    zobrist (int64, N), move_counter (int32, N),
    moves_since_last_combat (int32, N).
All arrays must be C-contiguous.
)doc")
        .def("copy_to_host", &py_copy_to_host,
             "Copy device arrays to host. Returns dict[str, np.ndarray].")
        .def("copy_turn_to_host", &py_copy_turn_to_host,
             "Fast-path D2H of just the (N,) int8 ``turn`` vector.  Used by\n"
             "PPO-style loops that only need the acting seat to slice obs\n"
             "and build the legal-action mask.")
        .def("copy_termination_from_host", &py_copy_termination_from_host,
             py::arg("terminated"), py::arg("winner_team"), py::arg("draw"),
             "Upload per-env termination state (bool (N,), int8 (N,), bool (N,)).")
        .def("copy_termination_to_host", &py_copy_termination_to_host,
             "Return dict with terminated/winner_team/draw numpy arrays.")
        .def("step_batch", &py_step_batch, py::arg("action_ids"),
             R"doc(
Advance every env by one action.  Returns a dict with keys:
  valid         : bool   (N,)
  event         : int8   (N,)  Event.value ∈ {0,1,2,3,4}
  terminated    : bool   (N,)
  winner_team   : int8   (N,)  -1/0/1
  draw          : bool   (N,)
  flag_captured : bool   (N,)

The device-side state is mutated in place.  Terminated envs are skipped.
``upload_zobrist_tables`` MUST have been called first for bit-identical
zobrist parity with CPU.
)doc")
        .def("copy_from_host_legal_lite", &py_copy_from_host_legal_lite,
             py::arg("piece_seat_arr"),
             py::arg("piece_type_arr"),
             py::arg("alive"),
             py::arg("pos_x"),
             py::arg("pos_y"),
             py::arg("cell_piece_id"),
             R"doc(
Fast-path H2D uploader for the 6 fields ``legal_action_ids_batch`` reads.

This is a surgical optimisation: ``legal_action_kernel`` only consumes
``piece_seat_arr``, ``piece_type_arr``, ``alive``, ``pos_x``, ``pos_y`` and
``cell_piece_id``.  Uploading all 20 SoA arrays on every step wastes ~70% of
the PCIe budget.  Using this method instead of :py:meth:`copy_from_host` cuts
the end-to-end cost of ``legal_action_ids_batch`` roughly in half.

Internally the fields are packed into a single pinned staging buffer and
uploaded with one ``cudaMemcpy``, then a device scatter kernel splits the blob
into the 6 per-field SoA arrays already owned by this
``DeviceGameStateBatch``.  Other SoA arrays on the device are left unchanged;
callers that need them for ``step_batch`` / ``build_observation_batch`` must
still call :py:meth:`copy_from_host`.

Parameters
----------
piece_seat_arr  : int8,  shape (N * 120)
piece_type_arr  : int8,  shape (N * 120)
alive           : bool,  shape (N * 120)
pos_x           : int8,  shape (N * 120)
pos_y           : int8,  shape (N * 120)
cell_piece_id   : int16, shape (N * 289)
)doc")
        .def("cm_copy_from_host", &py_cm_copy_from_host,
             py::arg("direct_lo"),
             py::arg("direct_hi"),
             py::arg("direct_type"),
             py::arg("last_direct_step"),
             py::arg("direct_other_count"),
             py::arg("chain_lo"),
             py::arg("chain_hi"),
             py::arg("chain_type"),
             py::arg("last_chain_step"),
             py::arg("eaten_by_pid_lo"),
             py::arg("eaten_by_pid_hi"),
             py::arg("rank_floor"),
             py::arg("rank_floor_step"),
             py::arg("is_gongb"),
             py::arg("not_gongb"),
             py::arg("attacked_by_known_gongb"),
             R"doc(
PARITY-TEST ONLY.  Upload CombatMemory v6 state from host arrays.

Each array has shape ``(N * 4 * 120,)`` flattened row-major as
``(env, observer, pid)``.  In production training there is no host-side
CombatMemory state — step_batch_kernel and observation_kernel manage it
entirely on device.  Use this helper from
``tests/test_gpu_combat_memory_parity.py`` to seed the GPU state with a
CPU-built reference and verify GPU updates match bit-for-bit.
)doc")
        .def("cm_copy_to_host", &py_cm_copy_to_host,
             R"doc(
PARITY-TEST ONLY.  Download CombatMemory v6 state into a numpy dict.

Returned arrays are flattened ``(N * 4 * 120,)`` mirrors of the device
SoA fields.  Compare against
``junqi_core.batched_state.BatchedGameState.cm_*`` after running the
same sequence of actions through both backends.
)doc");

    // --- DeviceObservationBatch ---
    py::class_<DeviceObservationBatch>(m, "DeviceObservationBatch",
        "GPU-resident observation tensors for all (env, seat) pairs.")
        .def(py::init<int>(), py::arg("num_envs"),
             "Allocate device memory for num_envs × NUM_SEATS observation tensors.")
        .def_readonly("num_envs", &DeviceObservationBatch::num_envs)
        .def_property_readonly("d_spatial_ptr",
            [](const DeviceObservationBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_spatial);
            },
            "Raw device pointer (int) to the spatial tensor. Useful for\n"
            "zero-copy torch integration via ``torch.from_dlpack`` or a\n"
            "manually-constructed cuda tensor wrapper.  Shape is\n"
            "(N, 4, 412, 17, 17) float32, row-major.")
        .def_property_readonly("d_global_ptr",
            [](const DeviceObservationBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_global);
            },
            "Raw device pointer (int) to the global tensor.  Shape is\n"
            "(N, 4, 28) float32, row-major.")
        .def("copy_to_host", &py_obs_copy_to_host,
             R"doc(
Copy observation tensors to host.

Returns
-------
spatial : np.ndarray  shape (N, 4, 101, 17, 17) float32
global_ : np.ndarray  shape (N, 4, 28)          float32
)doc");

    // --- DeviceObservationSingleBatch ---
    // Single-seat variant: (N, 101, 17, 17) instead of (N, 4, 101, 17, 17).
    py::class_<DeviceObservationSingleBatch>(m, "DeviceObservationSingleBatch",
        "GPU-resident observation tensors for ONE seat per env (acting seat only).")
        .def(py::init<int>(), py::arg("num_envs"),
             "Allocate device memory for num_envs single-seat observation tensors.")
        .def_readonly("num_envs", &DeviceObservationSingleBatch::num_envs)
        .def_property_readonly("d_spatial_ptr",
            [](const DeviceObservationSingleBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_spatial);
            },
            "Raw device pointer. Shape (N, 101, 17, 17) float32, row-major.")
        .def_property_readonly("d_global_ptr",
            [](const DeviceObservationSingleBatch& self) {
                return reinterpret_cast<uintptr_t>(self.d_global);
            },
            "Raw device pointer. Shape (N, 28) float32, row-major.");

    // --- DeviceRolloutHistory ---
    py::class_<DeviceRolloutHistory>(m, "DeviceRolloutHistory",
        "Compact GPU-resident history that reconstructs PPO minibatches on device.")
        .def(
            py::init<int, int>(),
            py::arg("num_steps"),
            py::arg("num_envs"),
            "Allocate compact state history for (num_steps, num_envs).")
        .def_readonly("num_steps", &DeviceRolloutHistory::num_steps)
        .def_readonly("num_envs", &DeviceRolloutHistory::num_envs)
        .def_readonly("history_bytes", &DeviceRolloutHistory::history_bytes)
        .def_readonly("replay_capacity", &DeviceRolloutHistory::replay_capacity)
        .def(
            "snapshot",
            [](DeviceRolloutHistory& history,
               const DeviceGameStateBatch& state,
               uintptr_t d_acting_seats_ptr,
               int step) {
                GpuScratch& scratch = GpuScratch::instance();
                if (scratch.d_belief == nullptr ||
                    scratch.belief_cap < state.num_envs) {
                    throw std::runtime_error(
                        "DeviceRolloutHistory.snapshot requires resident beliefs");
                }
                const int8_t* d_acting =
                    reinterpret_cast<const int8_t*>(d_acting_seats_ptr);
                history.snapshot(
                    state,
                    scratch.d_belief,
                    d_acting,
                    step);
            },
            py::arg("state"),
            py::arg("d_acting_seats_ptr"),
            py::arg("step"),
            R"doc(
Snapshot the current pre-action state, acting observer's belief, and
CombatMemory slice into one history step. All copies remain device-to-device.
)doc")
        .def(
            "reconstruct",
            [](DeviceRolloutHistory& history,
               uintptr_t d_flat_indices_ptr,
               uintptr_t d_acting_seats_ptr,
               int batch_size,
               int8_t show_mode) {
                const int64_t* d_indices =
                    reinterpret_cast<const int64_t*>(d_flat_indices_ptr);
                const int8_t* d_acting =
                    reinterpret_cast<const int8_t*>(d_acting_seats_ptr);
                RolloutHistoryReconstruction result = history.reconstruct(
                    d_indices,
                    d_acting,
                    batch_size,
                    show_mode);
                py::dict output;
                output["d_spatial_ptr"] =
                    reinterpret_cast<uintptr_t>(result.d_spatial);
                output["d_global_ptr"] =
                    reinterpret_cast<uintptr_t>(result.d_global);
                output["d_legal_mask_ptr"] =
                    reinterpret_cast<uintptr_t>(result.d_legal_mask);
                output["batch_size"] = result.batch_size;
                return output;
            },
            py::arg("d_flat_indices_ptr"),
            py::arg("d_acting_seats_ptr"),
            py::arg("batch_size"),
            py::arg("show_mode") = (int8_t)2,
            R"doc(
Gather selected compact states, rebuild acting-seat observations and legal
masks on device, and return raw device pointers to reusable output buffers.
)doc");

    // --- build_observation_single_seat ---
    m.def("build_observation_single_seat",
        [](const DeviceGameStateBatch& state,
           DeviceObservationSingleBatch& obs_out,
           uintptr_t d_acting_seats_ptr,
           int8_t show_mode) {
            GpuScratch& scratch = GpuScratch::instance();
            if (scratch.d_belief == nullptr) {
                throw std::runtime_error(
                    "build_observation_single_seat: call upload_beliefs first.");
            }
            const int8_t* d_acting = reinterpret_cast<const int8_t*>(d_acting_seats_ptr);
            build_observation_single_seat(
                state, scratch.d_belief, d_acting, obs_out, show_mode);
        },
        py::arg("state"), py::arg("obs_out"),
        py::arg("d_acting_seats_ptr"), py::arg("show_mode") = (int8_t)2,
        R"doc(
Build observation for ONE seat per env (the acting seat).
d_acting_seats_ptr: device pointer to (N,) int8 acting seat values.
Output: obs_out.d_spatial (N, 101, 17, 17), obs_out.d_global (N, 28).
4x faster than build_observation_batch + slicing.
)doc");

    // --- legal_action_ids_batch ---
    // acting_seats arrives as a host numpy array (int8, shape (N,)).
    // We upload it to device memory, call the kernel (which returns pointers to
    // persistent device buffers), then download action_ids and action_counts to host.
    // MAX_ACTIONS_PER_ENV = 512 (must match game_state.cu).
    m.def("legal_action_ids_batch",
        [](const DeviceGameStateBatch& state,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> acting_seats) {
            const int N = state.num_envs;
            constexpr int MAX_ACTIONS_PER_ENV = 512;

            // Validate host array.
            const int8_t* h_acting = require_array<int8_t>(acting_seats, (size_t)N, "acting_seats");

            // Use persistent scratch buffer for d_acting_seats (no malloc/free per call).
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_acting_seats(N);
            cudaError_t err = cudaMemcpy(scratch.d_acting_seats, h_acting,
                                          (size_t)N * sizeof(int8_t),
                                          cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy acting_seats H2D: ") + cudaGetErrorString(err));

            // Call CUDA launcher — returns pointers into persistent device buffers.
            // After this returns the kernel is done (KERNEL_CHECK called inside).
            auto [d_action_ids, d_action_counts] = legal_action_ids_batch(state, scratch.d_acting_seats);

            // Download results to host numpy arrays.
            auto h_ids    = py::array_t<int32_t>({(ssize_t)N, (ssize_t)MAX_ACTIONS_PER_ENV});
            auto h_counts = py::array_t<int32_t>((ssize_t)N);

            err = cudaMemcpy(h_ids.mutable_data(), d_action_ids,
                             (size_t)N * MAX_ACTIONS_PER_ENV * sizeof(int32_t),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy action_ids D2H: ") + cudaGetErrorString(err));

            err = cudaMemcpy(h_counts.mutable_data(), d_action_counts,
                             (size_t)N * sizeof(int32_t),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy action_counts D2H: ") + cudaGetErrorString(err));

            return py::make_tuple(h_ids, h_counts);
        },
        py::arg("state"), py::arg("acting_seats"),
        R"doc(
Generate legal action IDs for a batch of game states on GPU.

Parameters
----------
state : DeviceGameStateBatch
    GPU-resident game state batch.
acting_seats : np.ndarray int8, shape (N,)
    The seat (0=SOUTH,1=WEST,2=NORTH,3=EAST) that is to move in each env.

Returns
-------
action_ids : np.ndarray int32, shape (N, 512)
    Flat action IDs (src_flat * 289 + dst_flat). Valid entries are in
    action_ids[i, :counts[i]]. Remaining slots are uninitialized.
counts : np.ndarray int32, shape (N,)
    Number of legal actions for each env.
)doc");

    // --- legal_action_ids_batch_csr (CSR-format output) ---
    m.def("legal_action_ids_batch_csr",
        [](const DeviceGameStateBatch& state,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> acting_seats) {
            const int N = state.num_envs;

            const int8_t* h_acting = require_array<int8_t>(acting_seats, (size_t)N, "acting_seats");
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_acting_seats(N);
            cudaError_t err = cudaMemcpy(scratch.d_acting_seats, h_acting,
                                          (size_t)N * sizeof(int8_t),
                                          cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy acting_seats H2D: ") + cudaGetErrorString(err));

            LegalActionCsrResult r = legal_action_ids_batch_csr(state, scratch.d_acting_seats);

            // Download offsets (N+1) and values (total_actions).
            auto h_offsets = py::array_t<int32_t>((ssize_t)(N + 1));
            auto h_values  = py::array_t<int32_t>((ssize_t)r.total_actions);

            err = cudaMemcpy(h_offsets.mutable_data(), r.d_offsets,
                             (size_t)(N + 1) * sizeof(int32_t),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy offsets D2H: ") + cudaGetErrorString(err));

            if (r.total_actions > 0) {
                err = cudaMemcpy(h_values.mutable_data(), r.d_values,
                                 (size_t)r.total_actions * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                    throw std::runtime_error(std::string("cudaMemcpy values D2H: ") + cudaGetErrorString(err));
            }

            return py::make_tuple(h_offsets, h_values);
        },
        py::arg("state"), py::arg("acting_seats"),
        R"doc(
Legal actions in CSR (compressed-sparse-row) format.

Parameters
----------
state : DeviceGameStateBatch
acting_seats : int8 (N,)

Returns
-------
offsets : int32 (N+1,)
    Prefix sum of per-env legal-action counts.
    env i's actions live at values[offsets[i]:offsets[i+1]].
values : int32 (sum_of_counts,)
    Concatenated flat action IDs (src_flat * 289 + dst_flat) for all envs,
    in ascending env order.

Compared to the dense ``legal_action_ids_batch`` this avoids the D2H transfer
of the empty slots in the (N, 512) buffer.  At N=1024 and avg 27 actions/env
this is a 19× bandwidth reduction.
)doc");

    // --- legal_action_mask_batch (per-piece 32-slot mask) ---
    m.def("legal_action_mask_batch",
        [](const DeviceGameStateBatch& state,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> acting_seats) {
            const int N = state.num_envs;

            const int8_t* h_acting = require_array<int8_t>(acting_seats, (size_t)N, "acting_seats");
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_acting_seats(N);
            cudaError_t err = cudaMemcpy(scratch.d_acting_seats, h_acting,
                                          (size_t)N * sizeof(int8_t),
                                          cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy acting_seats H2D: ") + cudaGetErrorString(err));

            LegalActionMaskResult r = legal_action_mask_batch(state, scratch.d_acting_seats);

            // Download mask (N, 120, 32) bool.
            auto h_mask = py::array_t<bool>({(ssize_t)N, (ssize_t)NUM_PIECES, (ssize_t)SLOTS_PER_PIECE});
            err = cudaMemcpy(h_mask.mutable_data(), r.d_mask,
                             (size_t)N * NUM_PIECES * SLOTS_PER_PIECE * sizeof(bool),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy mask D2H: ") + cudaGetErrorString(err));

            return h_mask;
        },
        py::arg("state"), py::arg("acting_seats"),
        R"doc(
Per-piece legal-action mask on GPU.

Returns a bool array of shape (N, 120, 32) — ``mask[env, pid, slot]`` is
``True`` iff piece ``pid`` in env ``env`` can legally execute slot ``slot``.

Slot layout (constant across piece types):

    slot 0-3   : orthogonal 1-step (E/W/S/N in ADJACENT_CELLS order)
    slot 4-7   : diagonal 1-step via camp
    slot 8-11  : straight rail ray dir 0, distance k=1..4 (non-engineer)
    slot 12-15 : straight rail ray dir 1, distance k=1..4
    slot 16-19 : straight rail ray dir 2, distance k=1..4
    slot 20-23 : straight rail ray dir 3, distance k=1..4

    Engineers use slot 8..23 for BFS-reachable rail cells (in BFS-branch
    traversal order), up to 16 cells maximum.  Unused slots are 0.

    slot 24-31 : reserved (always 0).

This output is network-friendly: the action space is fixed at 120 × 32 = 3840,
27× smaller than the dense 289 × 289 = 83521 flat space.
)doc");

    // --- legal_mask_canonical_batch (dense flat mask in canonical frame) ---
    m.def("legal_mask_canonical_batch",
        [](const DeviceGameStateBatch& state,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> acting_seats) {
            const int N = state.num_envs;
            const int8_t* h_acting = require_array<int8_t>(acting_seats, (size_t)N, "acting_seats");
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_acting_seats(N);
            cudaError_t err = cudaMemcpy(scratch.d_acting_seats, h_acting,
                                          (size_t)N * sizeof(int8_t),
                                          cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy acting_seats H2D: ") + cudaGetErrorString(err));

            bool* d_mask = legal_mask_canonical_batch(state, scratch.d_acting_seats);
            return reinterpret_cast<uintptr_t>(d_mask);
        },
        py::arg("state"), py::arg("acting_seats"),
        R"doc(
Dense canonical-frame legal mask on GPU.  Returns raw device pointer (int)
to a bool array of shape (N, 83521) on the CUDA device.  Use
``_CudaArrayInterfaceView`` + ``torch.as_tensor`` for zero-copy torch access.
No D2H transfer occurs.
)doc");

    // --- legal_mask_canonical_batch_from_device (seats already on GPU) ---
    m.def("legal_mask_canonical_batch_from_device",
        [](const DeviceGameStateBatch& state, uintptr_t d_acting_seats_ptr) {
            const int8_t* d_acting = reinterpret_cast<const int8_t*>(d_acting_seats_ptr);
            bool* d_mask = legal_mask_canonical_batch(state, d_acting);
            return reinterpret_cast<uintptr_t>(d_mask);
        },
        py::arg("state"), py::arg("d_acting_seats_ptr"),
        R"doc(
Same as legal_mask_canonical_batch but takes a device pointer for acting_seats
(avoids H2D copy when seats are already on GPU as a torch tensor).
Returns device pointer to (N, 83521) bool mask.
)doc");


    // --- step_device (Phase 3+4: zero-CPU step pipeline) ---
    m.def("step_device",
        [](DeviceGameStateBatch& state,
           uintptr_t d_canonical_actions_ptr,
           uintptr_t d_acting_seats_ptr) {
            const int32_t* d_acts = reinterpret_cast<const int32_t*>(d_canonical_actions_ptr);
            const int8_t*  d_seats = reinterpret_cast<const int8_t*>(d_acting_seats_ptr);

            StepDeviceResult r = step_device(state, d_acts, d_seats, /*stream_id=*/0);

            // Return device pointers as dict of ints — caller wraps with torch.
            py::dict d;
            d["d_valid_ptr"]         = reinterpret_cast<uintptr_t>(r.d_valid);
            d["d_event_ptr"]         = reinterpret_cast<uintptr_t>(r.d_event);
            d["d_terminated_ptr"]    = reinterpret_cast<uintptr_t>(r.d_terminated);
            d["d_winner_team_ptr"]   = reinterpret_cast<uintptr_t>(r.d_winner_team);
            d["d_draw_ptr"]          = reinterpret_cast<uintptr_t>(r.d_draw);
            d["d_flag_captured_ptr"] = reinterpret_cast<uintptr_t>(r.d_flag_captured);
            d["d_rewards_ptr"]       = reinterpret_cast<uintptr_t>(r.d_rewards);
            d["d_world_actions_ptr"] = reinterpret_cast<uintptr_t>(r.d_world_actions);
            d["N"]                   = r.N;
            return d;
        },
        py::arg("state"), py::arg("d_canonical_actions_ptr"), py::arg("d_acting_seats_ptr"),
        R"doc(
Device-resident step: canonical actions (device int32 ptr) + acting seats
(device int8 ptr) → step + reward, all on GPU.

Returns dict of raw device pointers (int) for torch zero-copy access.
No host transfers occur.
)doc");

    // --- record_move_history ---
    m.def("record_move_history",
        [](DeviceGameStateBatch& state,
           uintptr_t d_world_actions_ptr,
           uintptr_t d_valid_ptr) {
            const int32_t* d_acts = reinterpret_cast<const int32_t*>(d_world_actions_ptr);
            const bool*    d_valid = reinterpret_cast<const bool*>(d_valid_ptr);
            record_move_history(state, d_acts, d_valid, /*stream_id=*/0);
        },
        py::arg("state"), py::arg("d_world_actions_ptr"), py::arg("d_valid_ptr"),
        "Record world-frame actions into the move history ring buffer.");

    // --- build_observation_batch ---
    // beliefs and observer_seats arrive as host numpy arrays.
    // We upload them to persistent GpuScratch buffers (one-shot cudaMalloc
    // that lives for the process, not per call).
    m.def("build_observation_batch",
        [](const DeviceGameStateBatch& state,
           py::array_t<float, py::array::c_style | py::array::forcecast> beliefs,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> observer_seats,
           DeviceObservationBatch& obs_out,
           int8_t show_mode) {
            const size_t bel_sz  = (size_t)state.num_envs * NUM_SEATS * NUM_TRACKED_TYPES * NUM_CELLS;
            const size_t obs_sz  = (size_t)state.num_envs * NUM_SEATS;

            const float*  h_bel = require_array<float>(beliefs, bel_sz, "beliefs");
            const int8_t* h_obs = require_array<int8_t>(observer_seats, obs_sz, "observer_seats");

            // Use persistent scratch buffers.
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_belief(state.num_envs);
            scratch.ensure_observer_seats(state.num_envs);

            cudaError_t err;
            err = cudaMemcpy(scratch.d_belief, h_bel, bel_sz * sizeof(float),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy beliefs H2D: ") + cudaGetErrorString(err));

            err = cudaMemcpy(scratch.d_observer_seats, h_obs, obs_sz * sizeof(int8_t),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaMemcpy observer_seats H2D: ") + cudaGetErrorString(err));

            // Launch kernel (persistent device pointers from scratch).
            build_observation_batch(state, scratch.d_belief, scratch.d_observer_seats,
                                    obs_out, show_mode);
        },
        py::arg("state"), py::arg("beliefs"),
        py::arg("observer_seats"), py::arg("obs_out"),
        py::arg("show_mode") = (int8_t)2,
        "Build observation tensors on GPU for all (env, seat) pairs.\n\n"
        "beliefs: float32 array of shape (N, 4, 12, 289) — host memory, copied to device.\n"
        "observer_seats: int8 array of shape (N, 4) — host memory, copied to device.\n"
        "show_mode: int8, 0=BRIGHT, 1=DARK, 2=HALF_DARK (default). Controls dark_teammate channel.");

    // Upload beliefs into the persistent GpuScratch buffer.
    // Call once at episode start (or whenever beliefs change) to avoid the
    // ~227 MB H2D per-step cost at N=4096.  Subsequent
    // ``build_observation_batch_resident`` calls reuse the device-resident
    // buffer without any H2D traffic.
    m.def("upload_beliefs",
        [](int num_envs,
           py::array_t<float, py::array::c_style | py::array::forcecast> beliefs) {
            const size_t bel_sz = (size_t)num_envs * NUM_SEATS * NUM_TRACKED_TYPES * NUM_CELLS;
            const float* h_bel = require_array<float>(beliefs, bel_sz, "beliefs");
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_belief(num_envs);
            CUDA_CHECK_PY(cudaMemcpy(scratch.d_belief, h_bel, bel_sz * sizeof(float),
                                     cudaMemcpyHostToDevice));
        },
        py::arg("num_envs"), py::arg("beliefs"),
        "One-shot H2D of beliefs into the persistent GpuScratch buffer.\n"
        "Pairs with ``build_observation_batch_resident`` for zero-H2D observation\n"
        "builds in hot RL loops.  beliefs shape: (N, 4, 12, 289) float32.");

    m.def("upload_observer_seats",
        [](int num_envs,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> observer_seats) {
            const size_t obs_sz = (size_t)num_envs * NUM_SEATS;
            const int8_t* h_obs = require_array<int8_t>(observer_seats, obs_sz, "observer_seats");
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_observer_seats(num_envs);
            CUDA_CHECK_PY(cudaMemcpy(scratch.d_observer_seats, h_obs,
                                     obs_sz * sizeof(int8_t), cudaMemcpyHostToDevice));
        },
        py::arg("num_envs"), py::arg("observer_seats"),
        "One-shot H2D of observer_seats (N, 4) into the persistent GpuScratch buffer.");

    m.def("build_observation_batch_resident",
        [](const DeviceGameStateBatch& state,
           DeviceObservationBatch& obs_out,
           int8_t show_mode) {
            GpuScratch& scratch = GpuScratch::instance();
            if (scratch.d_belief == nullptr || scratch.d_observer_seats == nullptr) {
                throw std::runtime_error(
                    "build_observation_batch_resident: call upload_beliefs and "
                    "upload_observer_seats first to populate the scratch buffers.");
            }
            build_observation_batch(state, scratch.d_belief, scratch.d_observer_seats,
                                    obs_out, show_mode);
        },
        py::arg("state"), py::arg("obs_out"), py::arg("show_mode") = (int8_t)2,
        "Variant of ``build_observation_batch`` that uses beliefs and\n"
        "observer_seats already uploaded via ``upload_beliefs`` and\n"
        "``upload_observer_seats``.  Does ZERO H2D per call; intended for\n"
        "hot-path RL loops where beliefs are relatively static between steps.");

    // --- Compact cell mapping tables ---
    m.def("upload_compact_cell_maps",
        [](py::array_t<int16_t, py::array::c_style | py::array::forcecast> flat_to_compact,
           py::array_t<int16_t, py::array::c_style | py::array::forcecast> compact_to_flat) {
            const int16_t* h_f2c = require_array<int16_t>(flat_to_compact, 289, "flat_to_compact");
            const int16_t* h_c2f = require_array<int16_t>(compact_to_flat, 129, "compact_to_flat");
            upload_compact_cell_maps(h_f2c, h_c2f);
        },
        py::arg("flat_to_compact"), py::arg("compact_to_flat"),
        "Upload compact cell index mapping tables (call once at init).");

    // --- Phase 5: Device-side episode reset ---
    m.def("upload_reset_tables",
        [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> pos_x,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> pos_y,
           py::array_t<int8_t, py::array::c_style | py::array::forcecast> piece_seat,
           py::array_t<bool,   py::array::c_style | py::array::forcecast> camp_slot) {
            const int8_t* h_px = require_array<int8_t>(pos_x, 120, "pos_x");
            const int8_t* h_py = require_array<int8_t>(pos_y, 120, "pos_y");
            const int8_t* h_ps = require_array<int8_t>(piece_seat, 120, "piece_seat");
            const bool*   h_cs = require_array<bool>(camp_slot, 30, "camp_slot");
            upload_reset_tables(h_px, h_py, h_ps, h_cs);
        },
        py::arg("pos_x"), py::arg("pos_y"), py::arg("piece_seat"), py::arg("camp_slot"),
        "Upload constant tables for device-side reset (call once at init).");

    m.def("upload_setup_pool",
        [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> piece_types) {
            auto buf = piece_types.request();
            if (buf.ndim != 2 || buf.shape[1] != 120) {
                throw std::runtime_error(
                    "upload_setup_pool: piece_types must be (pool_size, 120) int8");
            }
            int count = (int)buf.shape[0];
            const int8_t* h_data = static_cast<const int8_t*>(buf.ptr);
            upload_setup_pool(h_data, count);
        },
        py::arg("piece_types"),
        "Upload pool of valid initial setups (pool_size, 120) int8.\n"
        "Called once at init; the pool is used for device-side resets.");

    m.def("reset_terminated_envs",
        [](DeviceGameStateBatch& state, int64_t seed) {
            reset_terminated_envs(state, seed);
        },
        py::arg("state"), py::arg("seed"),
        "Reset all terminated envs from the pre-uploaded setup pool.\n"
        "Zero CPU involvement — all state reconstruction on GPU.");

    // --- Phase 1: Belief update GPU kernels ---

    m.def("upload_belief_prior_table",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> table) {
            const float* h_table = require_array<float>(table, 30 * 12, "belief_prior_table");
            upload_belief_prior_table(h_table);
        },
        py::arg("table"),
        "Upload 30×12 per-slot belief prior table to GPU constant memory.\n"
        "Call once at process start.");

    m.def("upload_seat_strongholds",
        [](py::array_t<int16_t, py::array::c_style | py::array::forcecast> strongholds) {
            const int16_t* h_sh = require_array<int16_t>(strongholds, 4 * 2, "seat_strongholds");
            upload_seat_strongholds(h_sh);
        },
        py::arg("strongholds"),
        "Upload 4×2 stronghold world-frame flat positions to GPU constant memory.\n"
        "Call once at process start.");

    m.def("init_beliefs_for_reset_envs",
        [](DeviceGameStateBatch& state) {
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_belief(state.num_envs);
            init_beliefs_for_reset_envs(state, scratch.d_belief);
        },
        py::arg("state"),
        "Initialise beliefs for envs marked by the pre-reset terminated snapshot.\n"
        "Must be called after reset and upload of prior table.");

    m.def("init_all_beliefs",
        [](DeviceGameStateBatch& state) {
            GpuScratch& scratch = GpuScratch::instance();
            scratch.ensure_belief(state.num_envs);
            init_all_beliefs(state, scratch.d_belief);
        },
        py::arg("state"),
        "Initialise beliefs for ALL envs unconditionally.\n"
        "Used for initial bulk reset.");

    m.def("snapshot_pre_step_flags",
        [](const DeviceGameStateBatch& state) {
            snapshot_pre_step_flags(state);
        },
        py::arg("state"),
        "Snapshot seat_flag_revealed and seat_dead before step_device().\n"
        "Used to detect changes in the subsequent belief update.");

    m.def("update_beliefs_after_step",
        [](DeviceGameStateBatch& state,
           uintptr_t d_event_ptr,
           uintptr_t d_flag_captured_ptr,
           uintptr_t d_world_actions_ptr) {
            GpuScratch& scratch = GpuScratch::instance();
            if (!scratch.d_belief) {
                throw std::runtime_error(
                    "update_beliefs_after_step: call init_beliefs_for_reset_envs first.");
            }
            const int8_t*  d_ev   = reinterpret_cast<const int8_t*>(d_event_ptr);
            const bool*    d_fc   = reinterpret_cast<const bool*>(d_flag_captured_ptr);
            const int32_t* d_acts = reinterpret_cast<const int32_t*>(d_world_actions_ptr);
            update_beliefs_after_step(state, d_ev, d_fc, d_acts, scratch.d_belief);
        },
        py::arg("state"),
        py::arg("d_event_ptr"),
        py::arg("d_flag_captured_ptr"),
        py::arg("d_world_actions_ptr"),
        "Incremental belief update after step_device().\n"
        "Applies deductive rules R1, R4, R5/R7, R6, R9, I5.\n"
        "All pointers must be device pointers from step_device result.");

    // --- GpuScratch management (test-only) ---
    m.def("gpu_scratch_reset", []() {
        GpuScratch::instance().reset();
    }, "Release all persistent GPU scratch buffers. Primarily for testing.");

    m.def("device_synchronize", []() {
        CUDA_CHECK_PY(cudaDeviceSynchronize());
    }, "Wait for all pending CUDA work to complete. Primarily for benchmarking.");
}
