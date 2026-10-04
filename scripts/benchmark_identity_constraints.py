#!/usr/bin/env python3
"""Compare cold exact-constraint solving with an explicitly supplied old source.

Each repetition uses separate fresh solvers to compute all 75 causal teacher
masks and generate a complete assignment with a fixed random stream. The previous
source is loaded as a sibling module, so its full original algorithm is used.
Timing includes compilation and both workloads but excludes dataset I/O.
This is a CPU microbenchmark, not a GPU or end-to-end search speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from junqi_rl.belief import constraints as current
from junqi_rl.belief.facts_dataset import batch_inputs, file_sha256, read_npz, write_json


def load_reference(path):
    name = "junqi_rl.belief._constraint_benchmark_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.CONSTRAINT_SCHEMA != current.CONSTRAINT_SCHEMA:
        raise ValueError("reference and candidate must implement identical constraint semantics")
    return module


def workload(module, inputs, targets, *, seed, max_search_nodes, compiler_options=None):
    start = time.perf_counter()
    solver = module.compile_constraints(inputs, max_search_nodes=max_search_nodes,
                                        **(compiler_options or {}))
    teacher = solver.teacher_masks(targets)
    teacher_elapsed = time.perf_counter() - start
    teacher_witnesses = sorted((list(prefix), witness) for prefix, witness in solver._cache.items())
    teacher_stats = solver.stats.copy()
    cache_stats = {"teacher": getattr(solver, "matching_cache_stats", None)}
    # Generation must start cold, as it does on a new search observer state.
    start = time.perf_counter()
    solver = module.compile_constraints(inputs, max_search_nodes=max_search_nodes,
                                        **(compiler_options or {}))
    rng = np.random.default_rng(seed)
    generated, generation_masks = [], []
    for _ in range(solver.size):
        mask = solver.allowed_next(generated)
        generation_masks.append(mask)
        generated.append(int(rng.choice(np.flatnonzero(mask))))
    solver.validate(generated)
    generation_elapsed = time.perf_counter() - start
    cache_stats["generation"] = getattr(solver, "matching_cache_stats", None)
    witnesses = sorted((list(prefix), witness) for prefix, witness in solver._cache.items())
    result = dict(
        teacher_masks_sha256=hashlib.sha256(teacher.tobytes()).hexdigest(),
        generation_masks_sha256=hashlib.sha256(np.stack(generation_masks).tobytes()).hexdigest(),
        teacher_witnesses_sha256=hashlib.sha256(json.dumps(teacher_witnesses).encode()).hexdigest(),
        witnesses_sha256=hashlib.sha256(json.dumps(witnesses).encode()).hexdigest(),
        generated_labels=generated, teacher_stats=teacher_stats, generation_stats=solver.stats,
    )
    times = dict(teacher=teacher_elapsed, generation=generation_elapsed,
                 total=teacher_elapsed + generation_elapsed)
    return times, result, teacher, cache_stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--max-search-nodes", type=int, default=100_000)
    parser.add_argument("--inventory-backend", choices=["scan", "incremental"], default="scan",
                        help="Candidate only; reference source keeps its original backend")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_search_nodes < 1:
        parser.error("repeats and max-search-nodes must be positive")
    if min(args.games + args.rows + [args.seed]) < 0:
        parser.error("game IDs, rows, and seed must be nonnegative")
    reference = load_reference(args.reference_source)
    report = dict(
        schema="junqi.constraint-cold-benchmark.v1", status="running",
        constraint_schema=current.CONSTRAINT_SCHEMA,
        implementation=(current.INCREMENTAL_IMPLEMENTATION if args.inventory_backend == "incremental"
                        else current.CONSTRAINT_IMPLEMENTATION),
        inventory_backend=args.inventory_backend,
        python=platform.python_version(), platform=platform.platform(),
        reference_source_sha256=file_sha256(args.reference_source),
        candidate_source_sha256=file_sha256(current.__file__),
        dataset_manifest_sha256=file_sha256(args.dataset / "dataset-manifest.json"),
        repeats=args.repeats, seed=args.seed, max_search_nodes=args.max_search_nodes,
        scope="two fresh CPU solvers per state: teacher masks and generated assignment",
        timing_excludes="dataset I/O, neural network, materialization, policy search, GPU",
        states=[],
    )
    for game in args.games:
        path = args.dataset / "shards" / f"game-{game:07d}.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError("dataset shard hash mismatch")
        shard = read_npz(path)
        for row in args.rows:
            inputs = batch_inputs(shard, [row])
            events = inputs.public_history[0]
            entry = dict(game_id=game, row=row, ply=int(shard["ply"][row]),
                         shard_sha256=metadata["sha256"], public_events=len(events),
                         public_combats=sum(e.has_combat for e in events),
                         public_deaths=sum(len(e.seats_died) for e in events),
                         reference_seconds=[], candidate_seconds=[],
                         reference_teacher_seconds=[], candidate_teacher_seconds=[],
                         reference_generation_seconds=[], candidate_generation_seconds=[])
            expected = None
            for repetition in range(args.repeats):
                order = [("reference", reference), ("candidate", current)]
                if repetition % 2:
                    order.reverse()
                for name, module in order:
                    times, result, teacher, cache_stats = workload(
                        module, inputs, shard["targets"][row],
                        seed=args.seed + game * 100 + row,
                        max_search_nodes=args.max_search_nodes,
                        compiler_options=({"inventory_backend": args.inventory_backend}
                                          if name == "candidate" else None))
                    if expected is None:
                        expected = result
                    if result != expected:
                        raise AssertionError(f"masks, witnesses, samples, or budget differ: {game}/{row}")
                    np.testing.assert_array_equal(teacher, shard["teacher_masks"][row])
                    entry[name + "_seconds"].append(times["total"])
                    for phase in ("teacher", "generation"):
                        entry[name + "_" + phase + "_seconds"].append(times[phase])
                    if name == "candidate":
                        entry["matching_cache_stats"] = cache_stats
            entry["parity"] = expected
            entry["speedup_median"] = (statistics.median(entry["reference_seconds"]) /
                                       statistics.median(entry["candidate_seconds"]))
            for phase in ("teacher", "generation"):
                entry[phase + "_speedup_median"] = (
                    statistics.median(entry["reference_" + phase + "_seconds"]) /
                    statistics.median(entry["candidate_" + phase + "_seconds"]))
            report["states"].append(entry)
            print(json.dumps(dict(game_id=game, row=row, ply=entry["ply"],
                                  speedup=entry["speedup_median"], parity="passed")), flush=True)
    report["status"] = "passed"
    report["aggregate_speedup_median_totals"] = (
        sum(statistics.median(s["reference_seconds"]) for s in report["states"]) /
        sum(statistics.median(s["candidate_seconds"]) for s in report["states"]))
    for phase in ("teacher", "generation"):
        report[phase + "_aggregate_speedup_median_totals"] = (
            sum(statistics.median(s["reference_" + phase + "_seconds"]) for s in report["states"]) /
            sum(statistics.median(s["candidate_" + phase + "_seconds"]) for s in report["states"]))
    report["all_cached_teacher_masks_match"] = True
    report["all_generated_masks_witnesses_samples_and_node_counts_match"] = True
    write_json(args.output, report)
    print(json.dumps(dict(status=report["status"], states=len(report["states"]),
                          aggregate_speedup=report["aggregate_speedup_median_totals"],
                          output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
