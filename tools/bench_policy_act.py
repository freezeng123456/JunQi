"""Isolate policy.act cost with full sample/value stack."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def bench(cfg, N, label):
    device = torch.device("cuda")
    policy = JunqiNet(cfg).to(device).eval()
    sp = torch.randn(N, 101, 17, 17, device=device)
    gl = torch.randn(N, 28, device=device)
    lm = torch.ones(N, 83521, dtype=torch.bool, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            policy.act(sp, gl, lm)
    torch.cuda.synchronize()

    STEPS = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(STEPS):
            policy.act(sp, gl, lm)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / STEPS
    print(f"{label:10s}  N={N:5d}  params={sum(p.numel() for p in policy.parameters())/1e6:5.2f}M "
          f" policy.act {t*1e3:7.2f} ms/call  → {N/t:,.0f} env·fwd/s")


def main():
    tiny = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                          embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    big = JunqiNetConfig()  # defaults = 128/3/6/256/8/4

    for N in (256, 1024, 4096):
        bench(tiny, N, "tiny")
    for N in (256, 1024):
        bench(big, N, "big")


if __name__ == "__main__":
    main()
