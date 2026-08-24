#!/usr/bin/env python3
"""Benchmark PPO engineering fast paths with synthetic or checkpoint weights.

The inputs are synthetic, but ``--checkpoint`` loads the exact policy, EMA and
optimizer continuation state.  This measures compute/memory engineering only;
policy quality must be evaluated separately with fixed checkpoints, seeds and
H2H games.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM, JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBatch


def production_net_config() -> JunqiNetConfig:
    return JunqiNetConfig(
        cnn_channels=128,
        cnn_layers=3,
        depth=8,
        embed_dim=320,
        n_head=8,
        ff_factor=4,
        dropout=0.0,
        use_cat_vf=True,
        action_key_dim=64,
        pos_emb_std=0.02,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_inputs(batch_size: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    spatial = torch.randn(
        batch_size,
        OBS_CHANNELS,
        17,
        17,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    global_ = torch.randn(
        batch_size,
        OBS_GLOBAL_DIMS,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    legal_ids = torch.randint(
        FLAT_ACTION_DIM,
        (batch_size, 64),
        device=device,
        generator=generator,
    )
    legal = torch.zeros(
        batch_size,
        FLAT_ACTION_DIM,
        device=device,
        dtype=torch.bool,
    )
    legal.scatter_(1, legal_ids, True)
    actions = legal_ids[:, 0].to(torch.int64)
    return spatial, global_, legal, actions


def timed_cuda(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def benchmark_forward(
    cfg: JunqiNetConfig,
    initial_state: dict[str, torch.Tensor],
    *,
    batch_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    model = JunqiNet(cfg).to(device).eval()
    model.load_state_dict(initial_state)
    spatial, global_, legal, actions = make_inputs(batch_size, device, seed=1001)

    def sampled():
        with torch.inference_mode():
            model(spatial, global_, legal)

    def evaluated():
        with torch.inference_mode():
            model(spatial, global_, legal, actions=actions)

    sampled_ms = timed_cuda(sampled, warmup=warmup, iterations=iterations)
    evaluated_ms = timed_cuda(evaluated, warmup=warmup, iterations=iterations)
    result = {
        "sampled_ms": sampled_ms,
        "evaluated_ms": evaluated_ms,
        "evaluate_speedup": sampled_ms / evaluated_ms,
    }
    # Drop closure references before emptying the CUDA allocator cache. Using
    # assignment instead of ``del`` also keeps Ruff's closure analysis from
    # treating the already-executed benchmark callbacks as undefined names.
    sampled = evaluated = None
    model = spatial = global_ = legal = actions = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_update_mode(
    mode: str,
    cfg: JunqiNetConfig,
    initial_state: dict[str, torch.Tensor],
    trainer_state: dict | None,
    *,
    batch_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(2002)
    torch.cuda.manual_seed_all(2002)
    model = JunqiNet(cfg).to(device)
    model.load_state_dict(initial_state)
    trainer = PPOTrainer(
        model,
        PPOConfig(
            net=cfg,
            dtype="float32",
            kl_mode=mode,
            torch_compile=False,
            num_epochs_per_rollout=1,
            minibatch_size=batch_size,
        ),
        device=device,
    )
    if trainer_state is not None:
        trainer.load_state_dict(trainer_state)
    trainer._sync_collect_policy()
    spatial, global_, legal, actions = make_inputs(batch_size, device, seed=3003)
    with torch.inference_mode():
        old_log_probs = model(
            spatial,
            global_,
            legal,
            actions=actions,
        )["action_log_prob"].float()
    generator = torch.Generator(device=device).manual_seed(4004)
    batch = RolloutBatch(
        obs_spatial=spatial,
        obs_global=global_,
        legal_mask=legal,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.randn(batch_size, device=device, generator=generator),
        returns=torch.rand(batch_size, device=device, generator=generator) * 2 - 1,
        values=torch.zeros(batch_size, device=device),
        adv_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
        value_only_mask=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )

    torch.cuda.reset_peak_memory_stats(device)
    update_ms = timed_cuda(
        lambda: trainer._update_step(batch),
        warmup=warmup,
        iterations=iterations,
    )
    result = {
        "update_ms": update_ms,
        "updates_per_second": 1000.0 / update_ms,
        "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "has_collection_policy": float(trainer._collect_policy is not None),
        "nan_skip_count": float(trainer._nan_skip_count),
        "grad_skip_count": float(trainer._grad_nan_skip_count),
    }
    trainer = model = batch = None
    spatial = global_ = legal = actions = old_log_probs = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional trainer checkpoint; loads policy, EMA and optimizer state",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device("cuda:0")
    trainer_state = None
    checkpoint_sha256 = None
    checkpoint_rollout = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        trainer_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(trainer_state, dict) or "policy" not in trainer_state:
            raise ValueError(f"not a trainer checkpoint: {checkpoint_path}")
        cfg = trainer_state["cfg"].net
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer_state["policy"].items()
        }
        checkpoint_sha256 = file_sha256(checkpoint_path)
        checkpoint_rollout = trainer_state.get("num_rollout")
        strict_probe = JunqiNet(cfg)
        strict_probe.load_state_dict(initial_state, strict=True)
        del strict_probe
    else:
        cfg = production_net_config()
        torch.manual_seed(42)
        initial_model = JunqiNet(cfg)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in initial_model.state_dict().items()
        }
        del initial_model
    state_key_sha256 = hashlib.sha256(
        "\n".join(initial_state).encode("utf-8")
    ).hexdigest()
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "parameters": sum(tensor.numel() for tensor in initial_state.values()),
        "state_key_sha256": state_key_sha256,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_rollout": checkpoint_rollout,
        "forward": benchmark_forward(
            cfg,
            initial_state,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        ),
    }
    full = benchmark_update_mode(
        "reverse_full",
        cfg,
        initial_state,
        trainer_state,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    sampled = benchmark_update_mode(
        "sampled_proxy",
        cfg,
        initial_state,
        trainer_state,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    result["ppo_update"] = {
        "reverse_full": full,
        "sampled_proxy": sampled,
        "speedup": full["update_ms"] / sampled["update_ms"],
        "peak_memory_saved_gib": (
            full["peak_allocated_gib"] - sampled["peak_allocated_gib"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
