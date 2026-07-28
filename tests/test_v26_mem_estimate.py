"""Memory-estimate for a scaled-up JunqiNet config. Estimates peak
activation & parameter memory so we can pick v26 dims without running
out of GPU at rollout-time.

Run under pytest so it works in auto mode.
"""
from __future__ import annotations

import pytest
import torch


def _estimate_junqi_mem(
    depth: int,
    embed_dim: int,
    n_head: int,
    ff_factor: int,
    cnn_channels: int,
    cnn_layers: int,
    num_envs: int,
    dtype_bytes: int = 2,   # fp16
    obs_channels: int = 256,
):
    """Return an approximate peak-memory estimate in GB.

    JunqiNet forward peaks at the transformer FFN:
      (B, 289, embed_dim * ff_factor) × dtype_bytes
    plus the CNN stem:
      (B, cnn_channels, 17, 17) × dtype_bytes × cnn_layers
    We also account for obs input (N*4, obs_channels, 17, 17).

    B here is the PPO minibatch size (512) for the forward pass during
    train. For rollout collect B = num_envs (acting seat only).
    """
    cells = 17 * 17
    forward_B = 512         # PPO minibatch
    rollout_B = num_envs    # collect

    # Parameters
    # Transformer block ~ 4 * embed^2 for QKVO + 2 * embed^2 * ff_factor for FFN
    params_per_block = 4 * embed_dim**2 + 2 * embed_dim**2 * ff_factor
    params_total = depth * params_per_block
    # Plus CNN + head (rough, ignore small layers)
    params_total += cnn_channels * obs_channels * 3 * 3   # first conv
    params_total += (cnn_layers - 1) * cnn_channels**2 * 3 * 3
    params_total += embed_dim * cells * 2   # action head + value head

    # Param memory (fp32 optimizer states + fp16 weights + grad scaler)
    # 1 × fp32 weight = 4 bytes, 1× fp16 copy = 2 bytes, 2× fp32 Adam = 8 bytes
    # So ~14 bytes per param
    param_gb = params_total * 14 / 1e9

    # Activation peaks
    # FFN inner activation: (B, 289, embed*ff_factor)
    ffn_gb_forward = forward_B * cells * embed_dim * ff_factor * dtype_bytes / 1e9
    ffn_gb_rollout = rollout_B * cells * embed_dim * ff_factor * dtype_bytes / 1e9

    # Attention scores: (B, n_head, 289, 289)
    attn_gb_forward = forward_B * n_head * cells * cells * dtype_bytes / 1e9
    attn_gb_rollout = rollout_B * n_head * cells * cells * dtype_bytes / 1e9

    # Obs + backbone
    obs_gb = rollout_B * 4 * obs_channels * cells * 4 / 1e9   # fp32
    cnn_gb = forward_B * cnn_channels * cells * dtype_bytes * cnn_layers / 1e9

    return {
        "params": params_total,
        "param_gb": param_gb,
        "ffn_forward_gb": ffn_gb_forward,
        "ffn_rollout_gb": ffn_gb_rollout,
        "attn_forward_gb": attn_gb_forward,
        "attn_rollout_gb": attn_gb_rollout,
        "obs_gb": obs_gb,
        "cnn_gb": cnn_gb,
        "peak_estimate_gb": (
            param_gb + ffn_gb_forward + attn_gb_forward + obs_gb + cnn_gb
        ),
    }


def _pretty_print(name, cfg):
    print(f"\n=== {name} ===")
    print(f"  depth={cfg['depth']}  embed={cfg['embed_dim']}  "
          f"head={cfg['n_head']}  ff={cfg['ff_factor']}  "
          f"cnn={cfg['cnn_channels']}×{cfg['cnn_layers']}")
    est = _estimate_junqi_mem(**{k: v for k, v in cfg.items() if k != "label"})
    print(f"  params: {est['params']/1e6:.2f}M  "
          f"(Adam+grad mem ~{est['param_gb']:.2f} GB)")
    print(f"  FFN forward peak (B=512):   {est['ffn_forward_gb']:.3f} GB")
    print(f"  Attn forward peak (B=512):  {est['attn_forward_gb']:.3f} GB")
    print(f"  Obs buffer (N×4):           {est['obs_gb']:.3f} GB")
    print(f"  CNN stem:                   {est['cnn_gb']:.3f} GB")
    print(f"  *** PEAK ESTIMATE ***       {est['peak_estimate_gb']:.2f} GB "
          f"(T4 budget after arr+belief: ~7 GB main PPO)")


def test_display_v26_candidates():
    """Print memory estimates for candidate v26 net sizes.

    T4 total: 15 GB. Known usage at v25 steady state:
      - arr net + buffer + optimizer: ~5 GB
      - belief net + buffer: ~1 GB
      - obs + CUDA rollout: ~1 GB
      - main PPO (v25 sized): ~2-3 GB
      - padding + allocator frag: ~1 GB
    Target for main PPO in v26: ≤ 6 GB peak to stay safely under budget.
    """
    configs = [
        # (label, depth, embed, head, ff, cnn_ch, cnn_layers, num_envs)
        dict(label="v25 baseline",   depth=4, embed_dim=128, n_head=4,
             ff_factor=4, cnn_channels=64, cnn_layers=2, num_envs=128),
        dict(label="v26-A (conservative)", depth=6, embed_dim=192, n_head=8,
             ff_factor=4, cnn_channels=96, cnn_layers=2, num_envs=128),
        dict(label="v26-B (mid)",    depth=6, embed_dim=256, n_head=8,
             ff_factor=4, cnn_channels=128, cnn_layers=2, num_envs=128),
        dict(label="v26-C (aggressive)", depth=8, embed_dim=256, n_head=8,
             ff_factor=4, cnn_channels=128, cnn_layers=3, num_envs=128),
        dict(label="v26-D (match paper)", depth=8, embed_dim=384, n_head=8,
             ff_factor=4, cnn_channels=128, cnn_layers=3, num_envs=128),
    ]
    for cfg in configs:
        label = cfg.pop("label")
        _pretty_print(label, cfg)
    print()
