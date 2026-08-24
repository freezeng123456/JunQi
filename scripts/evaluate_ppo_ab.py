#!/usr/bin/env python3
"""Compare two one-step PPO continuations and evaluate both against a baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "policy" not in state:
        raise ValueError(f"not a trainer checkpoint: {path}")
    return state


def compare_policy_updates(
    base_state: dict,
    reference_state: dict,
    candidate_state: dict,
) -> dict[str, float | int]:
    base = base_state["policy"]
    reference = reference_state["policy"]
    candidate = candidate_state["policy"]
    if set(base) != set(reference) or set(base) != set(candidate):
        raise ValueError("policy state dictionaries do not have identical keys")

    dot = reference_norm2 = candidate_norm2 = difference_norm2 = 0.0
    max_abs_difference = 0.0
    changed_coordinates = total_coordinates = 0
    for key in base:
        if not base[key].is_floating_point():
            if not torch.equal(base[key], reference[key]) or not torch.equal(
                base[key], candidate[key]
            ):
                raise ValueError(f"non-floating buffer changed: {key}")
            continue
        reference_delta = (reference[key] - base[key]).double().reshape(-1)
        candidate_delta = (candidate[key] - base[key]).double().reshape(-1)
        difference = reference_delta - candidate_delta
        dot += torch.dot(reference_delta, candidate_delta).item()
        reference_norm2 += torch.dot(reference_delta, reference_delta).item()
        candidate_norm2 += torch.dot(candidate_delta, candidate_delta).item()
        difference_norm2 += torch.dot(difference, difference).item()
        if difference.numel():
            max_abs_difference = max(
                max_abs_difference,
                difference.abs().max().item(),
            )
            changed_coordinates += int(torch.count_nonzero(difference).item())
            total_coordinates += difference.numel()

    reference_norm = math.sqrt(reference_norm2)
    candidate_norm = math.sqrt(candidate_norm2)
    difference_norm = math.sqrt(difference_norm2)
    return {
        "reference_delta_l2": reference_norm,
        "candidate_delta_l2": candidate_norm,
        "delta_cosine_similarity": dot / max(reference_norm * candidate_norm, 1e-30),
        "delta_difference_l2": difference_norm,
        "delta_difference_relative_to_reference": difference_norm
        / max(reference_norm, 1e-30),
        "max_abs_delta_difference": max_abs_difference,
        "different_update_coordinates": changed_coordinates,
        "floating_coordinates": total_coordinates,
    }


def load_policy(state: dict, device: torch.device):
    from junqi_rl.networks.junqi_net import JunqiNet

    policy = JunqiNet(state["cfg"].net).to(device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    return policy


def evaluate_candidate(
    candidate_path: Path,
    baseline_state: dict,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head
    from junqi_rl.gpu_rollout import GpuRollout

    autocast_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    state = load_state(candidate_path)
    candidate = load_policy(state, device)
    baseline = load_policy(baseline_state, device)

    action_seed = args.seed + 100_000
    torch.manual_seed(action_seed)
    torch.cuda.manual_seed_all(action_seed)
    bootstrap = GpuRollout(
        num_envs=min(args.num_envs, (args.games + 1) // 2),
        device_id=device.index or 0,
    )
    pool_size = bootstrap.upload_fixed_evaluation_setup_pool(seed=args.setup_seed)
    metrics = evaluate_paired_head_to_head(
        candidate,
        baseline,
        num_games=args.games,
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        max_moves=args.max_moves,
        autocast_dtype=autocast_dtype,
        greedy=args.greedy,
    )
    result = {
        "path": str(candidate_path),
        "sha256": file_sha256(candidate_path),
        "rollout": state.get("num_rollout"),
        "setup_pool_size": pool_size,
        "setup_seed": args.setup_seed,
        "environment_seed": args.seed,
        "action_seed": action_seed,
        "greedy": args.greedy,
        "dtype": args.dtype,
        "metrics": metrics,
    }
    del state, candidate, baseline, bootstrap
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1_000_587)
    parser.add_argument("--setup-seed", type=int, default=20_260_817)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="autocast dtype used by H2H policy inference",
    )
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    paths = [args.base, args.reference, args.candidate, args.baseline]
    paths = [path.expanduser().resolve() for path in paths]
    args.base, args.reference, args.candidate, args.baseline = paths

    print("[ab] loading checkpoints and comparing policy deltas", flush=True)
    base_state = load_state(args.base)
    reference_state = load_state(args.reference)
    candidate_state = load_state(args.candidate)
    comparison = compare_policy_updates(
        base_state,
        reference_state,
        candidate_state,
    )
    del base_state, reference_state, candidate_state
    gc.collect()
    print(json.dumps({"policy_update_comparison": comparison}, indent=2), flush=True)

    baseline_state = load_state(args.baseline)
    evaluations = {}
    for label, path in (
        ("reference", args.reference),
        ("candidate", args.candidate),
    ):
        print(f"[ab] evaluating {label} against frozen baseline", flush=True)
        evaluations[label] = evaluate_candidate(
            path,
            baseline_state,
            args=args,
            device=device,
        )
        print(json.dumps({label: evaluations[label]}, indent=2), flush=True)

    output = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "base": {"path": str(args.base), "sha256": file_sha256(args.base)},
        "baseline": {
            "path": str(args.baseline),
            "sha256": file_sha256(args.baseline),
        },
        "policy_update_comparison": comparison,
        "evaluations": evaluations,
    }
    print("[ab] final_result", flush=True)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
