#!/usr/bin/env python3
"""Collect complete facts-policy games on CUDA and preprocess them on job CPUs.

Run inside a Slurm allocation. Games are collected in bounded waves; active
games are never reset. Device action/event arrays cross to CPU once per block,
and only completed games are persisted. A rerun with the identical protocol
skips already durable game IDs and supervised shards.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "junqi_rl"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from junqi_rl.belief.facts_dataset import (
    DATASET_SCHEMA, TRACE_SCHEMA, file_sha256, persist_trace, prepare_protocol,
    preprocess_game, protocol_sha256, rebuild_manifest,
)


def _contiguous_waves(ids, width):
    for _, group in itertools.groupby(enumerate(ids), lambda pair: pair[1] - pair[0]):
        run = [item[1] for item in group]
        for offset in range(0, len(run), width):
            yield run[offset:offset + width]


def _collect_wave(policy, ids, protocol, block_steps):
    import hashlib
    import numpy as np
    import torch
    from junqi_rl.analysis.random_eval import _policy_actions
    from junqi_rl.gpu_rollout import GpuRollout

    count = len(ids)
    world = GpuRollout(count, include_global=False, identity_encoding="known_only",
                      max_num_moves=protocol["max_moves"])
    world.reset(seed_base=protocol["setup_seed"] + ids[0])
    # One initial truth download per wave. It is stored solely for replay and
    # supervised targets and is never supplied to policy or BeliefNet inputs.
    initial = np.asarray(world.state.copy_to_host()["piece_type_arr"], dtype=np.int8).reshape(count, 120)
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in initial]
    seeds = torch.arange(protocol["action_seed"] + ids[0],
                         protocol["action_seed"] + ids[0] + count, device="cuda")
    active = torch.ones(count, dtype=torch.bool, device="cuda")
    chunks = []
    with torch.inference_mode():
        for start in range(0, protocol["max_moves"], block_steps):
            block_length = min(block_steps, protocol["max_moves"] - start)
            block = torch.empty((block_length, count, 6), dtype=torch.int32, device="cuda")
            for offset in range(block_length):
                acting = torch.where(active, world.turn_torch(), 0).to(torch.int8)
                spatial, global_ = world.build_acting_seat_observation_torch(acting)
                legal = world.legal_mask_canonical_torch_device(acting).clone()
                legal[~active, 0] = True
                actions = _policy_actions(policy, spatial, global_, legal, seeds,
                                          start + offset, False, stream=19)
                result = world.step_device_torch(actions, acting)
                block[offset, :, 0] = actions
                block[offset, :, 1] = acting
                for column, name in enumerate(("event", "terminated", "draw", "winner_team"), start=2):
                    block[offset, :, column] = result[name]
                active &= ~result["terminated"]
            # Transfer compact traces once per block, not full state per move.
            chunks.append(block.cpu().numpy())
            if not bool(active.any()):
                break
    if bool(active.any()):
        raise RuntimeError("collection hit a step cap with unfinished games; none may count as complete")
    if world.state.has_belief_buffers:
        raise RuntimeError("facts collector unexpectedly allocated rule probabilities")
    packed = np.concatenate(chunks, axis=0)
    for lane, game_id in enumerate(ids):
        terminal = np.flatnonzero(packed[:, lane, 3])
        if not len(terminal):
            raise RuntimeError("missing terminal event in supposedly complete lane")
        moves = int(terminal[0]) + 1
        rows = packed[:moves, lane]
        trace = dict(initial_piece_types=initial[lane].copy(), actions=rows[:, 0].astype(np.int32),
                     seats=rows[:, 1].astype(np.int8), events=rows[:, 2].astype(np.int8),
                     terminated=rows[:, 3].astype(bool), draw=rows[:, 4].astype(bool),
                     winner_team=rows[:, 5].astype(np.int8))
        draw, winner = bool(rows[-1, 4]), int(rows[-1, 5])
        reason = ("total_move_limit" if draw and moves >= protocol["max_moves"] else
                  "no_combat_limit" if draw else "team_victory")
        metadata = dict(schema=TRACE_SCHEMA, complete=True, game_id=game_id, moves=moves,
                        max_moves=protocol["max_moves"], setup_seed=protocol["setup_seed"] + game_id,
                        action_seed=protocol["action_seed"] + game_id,
                        initial_setup_sha256=hashes[lane], draw=draw, winner_team=winner,
                        termination_reason=reason, collection_backend="native_cuda",
                        action_frame="observer_south_compact129", show_mode="DARK")
        yield trace, metadata


def _load_gpu_policy(checkpoint):
    import importlib
    import torch
    from junqi_rl.checkpoint_compat import current_checkpoint_metadata
    from junqi_rl.training.checkpoint import load_evaluation_policy

    if not torch.cuda.is_available():
        raise RuntimeError("complete-game collection requires CUDA; CPU fallback is forbidden")
    native = importlib.import_module("junqi_cuda")
    native.init_tables()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = load_evaluation_policy(checkpoint, device="cuda")
    metadata = current_checkpoint_metadata(policy)
    if (metadata["identity_encoding"] != "known_only" or metadata["observation_channels"] != 317
            or metadata["observation_global_dims"] != 0 or metadata["policy_features"]):
        raise ValueError("collection parent must be the unmodified facts317/0 baseline")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
        raise ValueError("non-finite collection parent weights")
    print(json.dumps(dict(phase="cuda_ready", device=torch.cuda.get_device_name(),
                          parent_checkpoint_sha256=file_sha256(checkpoint))), flush=True)
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--positions-per-game", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--block-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=4000)
    parser.add_argument("--setup-seed", type=int, default=5200300)
    parser.add_argument("--action-seed", type=int, default=6200300)
    parser.add_argument("--position-seed", type=int, default=20261003)
    parser.add_argument("--phase", choices=("all", "collect", "preprocess"), default="all")
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this command in a Slurm compute allocation, never on a login node")
    if any(getattr(args, name) < 1 for name in
           ("games", "positions_per_game", "num_envs", "block_steps", "workers", "max_moves")):
        parser.error("game, sample, environment, block, worker, and move budgets must be positive")
    if min(args.setup_seed, args.action_seed, args.position_seed) < 0:
        parser.error("seeds must be nonnegative")
    protocol = dict(schema=DATASET_SCHEMA, parent_checkpoint_sha256=file_sha256(args.checkpoint),
                    identity_input_schema="junqi.hidden-identity-facts.v1", identity_encoding="known_only",
                    observation_channels=317, observation_global_dims=0, observation_semantics_version=5,
                    games=args.games, positions_per_game=args.positions_per_game, max_moves=args.max_moves,
                    setup_seed=args.setup_seed, action_seed=args.action_seed, position_seed=args.position_seed,
                    show_mode="DARK", action_sampling="seeded_stochastic_stream19",
                    split="initial_setup_sha256_mod10_train0to7_validation8_test9",
                    position_sampling="one_random_ply_per_stratum_rotating_observer",
                    teacher_masks="exact_public_constraints_causal_prefix",
                    acceptance_only=args.max_moves != 4000)
    prepare_protocol(args.output, protocol)
    # Verifying existing hashes up front makes collection resume fail closed.
    manifest = rebuild_manifest(args.output)
    durable = {row["game_id"] for row in manifest["raw_games"]}
    processed = {row["game_id"] for row in manifest["shards"]}
    missing = sorted(set(range(args.games)) - durable)
    if args.phase == "preprocess" and missing:
        raise ValueError("preprocess-only requires all requested complete raw games to be durable")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
            if args.phase != "collect" else None)
    pending = {}
    try:
        if pool:
            pending = {pool.submit(preprocess_game, str(args.output), game_id): game_id
                       for game_id in sorted(durable - processed)}
        if args.phase != "preprocess" and missing:
            policy = _load_gpu_policy(args.checkpoint)
            for ids in _contiguous_waves(missing, args.num_envs):
                for trace, metadata in _collect_wave(policy, ids, protocol, args.block_steps):
                    stored = persist_trace(args.output, trace, metadata, protocol)
                    durable.add(stored["game_id"])
                    if pool:
                        future = pool.submit(preprocess_game, str(args.output), stored["game_id"])
                        pending[future] = stored["game_id"]
                for future in [future for future in pending if future.done()]:
                    future.result()
                    processed.add(pending.pop(future))
                print(json.dumps(dict(phase="collection", completed_games=len(durable),
                                      target_games=args.games, preprocessed_games=len(processed))), flush=True)
        for future in as_completed(pending):
            future.result()
            processed.add(pending[future])
            if len(processed) % 25 == 0 or len(processed) == args.games:
                print(json.dumps(dict(phase="preprocessing", completed_games=len(durable),
                                      preprocessed_games=len(processed), target_games=args.games)), flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        manifest = rebuild_manifest(args.output)
    if file_sha256(args.checkpoint) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("source policy checkpoint changed during dataset collection")
    if len(durable) != args.games or (args.phase != "collect" and manifest["status"] != "complete"):
        raise RuntimeError("dataset did not reach its frozen complete-game budget")
    print(json.dumps(dict(phase="finished", status=manifest["status"],
                          completed_games=manifest["completed_games"],
                          observation_rows=manifest["observation_rows"],
                          protocol_sha256=protocol_sha256(protocol),
                          manifest_sha256=file_sha256(args.output / "dataset-manifest.json"))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
