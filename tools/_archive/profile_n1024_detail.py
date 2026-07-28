"""Line-level profile of generate_legal_action_ids_n at N=1024."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState
import junqi_core._movegen_tables as _T
from junqi_core.move_gen import (
    _compute_occupancy_masks_n, _split_by_env,
    _ADJ_STRAIGHT_PAD, _ADJ_DIAG_PAD, _SRAYS_PAD, _SRAYS_L,
    _EMPTY_INT32, NUM_CELLS, BOARD_SIZE, _engineer_dests_soa,
)

N = 1024
states = [GameState.new_game(generate_random_setup()) for _ in range(N)]
b = BatchedGameState.from_game_states(states)
for _ in range(20):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Manual execution with timing
seat_per_env = b.turn
skip = b.terminated | b.seat_dead_arr[np.arange(N), seat_per_env.astype(np.intp)]

REPS = 5
timings = {}

for trial in range(REPS):
    t0 = time.perf_counter()
    ai = np.nonzero(~skip)[0]
    A = len(ai)
    timings.setdefault('00_active', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    cpid = b.cell_piece_id[ai]
    psa = b.piece_seat_arr[ai]
    pta = b.piece_type_arr[ai]
    alv = b.alive[ai]
    px = b.pos_x[ai]
    py = b.pos_y[ai]
    spe = seat_per_env[ai]
    cta = b.cell_team_arr[ai]
    timings.setdefault('01_subbatch_views', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(cpid, spe, cta)
    enemy_att = landable_pad[:, :NUM_CELLS] & ~empty
    timings.setdefault('02_occupancy', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    seat_match = alv & (psa == spe[:, np.newaxis])
    pt_vals_full = pta.astype(np.intp, copy=False)
    mobile_full = ~_T.IS_IMMOBILE_TYPE[pt_vals_full]
    on_board_full = (px >= 0) & (py >= 0)
    sx_full = px.astype(np.int32, copy=False)
    sy_full = py.astype(np.int32, copy=False)
    sf_full = sy_full * BOARD_SIZE + sx_full
    sf_clip = np.clip(sf_full, 0, NUM_CELLS - 1).astype(np.intp)
    not_sh_full = ~_T.IS_STRONGHOLD_FLAT[sf_clip]
    cand = seat_match & mobile_full & on_board_full & not_sh_full
    timings.setdefault('03_cand_mask', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    c_a, c_p = np.nonzero(cand)
    C = c_a.shape[0]
    c_env = ai[c_a].astype(np.int32)
    c_sf = sf_clip[c_a, c_p]
    c_pt = pt_vals_full[c_a, c_p]
    is_rail_c = _T.IS_RAIL_FLAT[c_sf]
    is_eng_c = _T.IS_ENGINEER_TYPE[c_pt]
    src_act = c_sf.astype(np.int32) * NUM_CELLS
    lp_flat = landable_pad.ravel()
    ep_flat = empty_pad.ravel()
    row_stride = NUM_CELLS + 1
    timings.setdefault('04_sparse_setup', []).append(time.perf_counter() - t0)

    env_parts, act_parts = [], []

    t0 = time.perf_counter()
    adj = _ADJ_STRAIGHT_PAD[c_sf]
    fi_a = c_a[:, np.newaxis] * row_stride + adj
    aok = lp_flat[fi_a.ravel()].reshape(C, 4)
    if aok.any():
        ci, dk = np.nonzero(aok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + adj[ci, dk])
    timings.setdefault('05_ortho', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    diag = _ADJ_DIAG_PAD[c_sf]
    fi_d = c_a[:, np.newaxis] * row_stride + diag
    dok = lp_flat[fi_d.ravel()].reshape(C, 4)
    if dok.any():
        ci, dk = np.nonzero(dok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + diag[ci, dk])
    timings.setdefault('06_diag', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    rail_ne = is_rail_c & ~is_eng_c
    if rail_ne.any():
        r_sf = c_sf[rail_ne]
        r_a = c_a[rail_ne]
        r_env = c_env[rail_ne]
        r_sc = src_act[rail_ne]
        rays = _SRAYS_PAD[r_sf]
        Cr = r_sf.shape[0]
        L = _SRAYS_L
        fi_r = (r_a[:, np.newaxis, np.newaxis] * row_stride + rays).ravel()
        re = ep_flat[fi_r].reshape(Cr, 4, L)
        rl = lp_flat[fi_r].reshape(Cr, 4, L)
        pref = np.empty((Cr, 4, L), dtype=bool)
        pref[..., 0] = True
        if L > 1:
            pref[..., 1] = re[..., 0]
            running = re[..., 0]
            for ll in range(2, L):
                running = running & re[..., ll - 1]
                pref[..., ll] = running
        ok = pref & rl
        if ok.any():
            ci, di, li = np.nonzero(ok)
            env_parts.append(r_env[ci])
            act_parts.append(r_sc[ci] + rays[ci, di, li])
    timings.setdefault('07_straight_rail', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    eng_on_rail = is_rail_c & is_eng_c
    if eng_on_rail.any():
        eng_idx = np.nonzero(eng_on_rail)[0]
        eng_envs, eng_acts = [], []
        for kidx in eng_idx.tolist():
            sf = int(c_sf[kidx])
            a_idx = int(c_a[kidx])
            dests = _engineer_dests_soa(sf, empty[a_idx], enemy_att[a_idx])
            if dests:
                d_arr = np.asarray(dests, dtype=np.int32)
                eng_envs.append(np.full(len(dests), int(c_env[kidx]), dtype=np.int32))
                eng_acts.append(sf * NUM_CELLS + d_arr)
        if eng_envs:
            env_parts.append(np.concatenate(eng_envs))
            act_parts.append(np.concatenate(eng_acts))
    timings.setdefault('08_engineer', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    # curve rail
    curve_cand = is_rail_c & ~is_eng_c
    if curve_cand.any() and _T.CURVE_CELLS:
        cids = _T.CURVE_ID_OF[c_sf]
        curve_mask = curve_cand & (cids > 0) & _T.IS_CURVE_ACTIVE[c_sf]
        if curve_mask.any():
            cr_sf = c_sf[curve_mask]
            cr_a = c_a[curve_mask]
            cr_env = c_env[curve_mask]
            cr_sc = src_act[curve_mask]
            rays_c = _T.CURVE_CHAIN_RAYS_PAD[cr_sf]
            Cv = cr_sf.shape[0]
            CL = rays_c.shape[2]
            fi_cr = (cr_a[:, np.newaxis, np.newaxis] * row_stride + np.where(
                rays_c >= 0, rays_c, NUM_CELLS).astype(np.intp)).ravel()
            cr_empty = ep_flat[fi_cr].reshape(Cv, 2, CL)
            cr_land = lp_flat[fi_cr].reshape(Cv, 2, CL)
            sh = np.empty_like(cr_empty)
            sh[..., 0] = True
            if CL > 1:
                sh[..., 1:] = cr_empty[..., :-1]
            pref_c = np.cumprod(sh, axis=-1).astype(bool, copy=False)
            ok = pref_c & cr_land
            if ok.any():
                ci, di, li = np.nonzero(ok)
                env_parts.append(cr_env[ci])
                act_parts.append(cr_sc[ci] + rays_c[ci, di, li].astype(np.int32))
    timings.setdefault('09_curve', []).append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    if env_parts:
        env_col_all = np.concatenate(env_parts)
        act_col_all = np.concatenate(act_parts)
        result = _split_by_env(env_col_all, act_col_all, N)
    timings.setdefault('10_assembly', []).append(time.perf_counter() - t0)

print(f"N={N}, A={A}, C={C}")
print(f"Rail non-eng: {int(rail_ne.sum())}, Eng: {int(eng_on_rail.sum())}")
total = 0
for key in sorted(timings.keys()):
    avg_ms = np.mean(timings[key]) * 1000
    total += avg_ms
    print(f"  {key:25s}: {avg_ms:8.3f} ms")
print(f"  {'TOTAL':25s}: {total:8.3f} ms")
print(f"  Target combined (legal+step+select) ≤ 20.5ms")
