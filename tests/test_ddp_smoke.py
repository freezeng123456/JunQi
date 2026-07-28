"""tests/test_ddp_smoke.py — Smoke tests for the DDP wrapping pattern used by
``junqi_rl.training.{ppo,arr_ppo,belief_ppo}.*Trainer`` and ``scripts/train.py``.

These tests are deliberately **standalone** — they do NOT import junqi_core
or junqi_rl. They validate the *pattern* I added (compile → DDP wrap, save
unwrapped state_dict, all_reduce(MIN) for minibatch counts, broadcast for
early-stop verdicts, cross-world-size resume) on a tiny dummy model.

The full integration test (real PPOTrainer with JunqiNet) requires Python
3.10+ (junqi_core uses ``dataclass(slots=True)``) and is run separately
on the target H20 box; see ``tests/test_ddp_integration.py``.

Run with::

    python -m pytest tests/test_ddp_smoke.py -v

Each test spawns 2 child processes via ``torch.multiprocessing.spawn`` to
exercise a real torch.distributed group on the gloo CPU backend (no GPU
required). Total runtime ~10 s.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


# ---------------------------------------------------------------------------
# Tiny model + the same DDP wrap pattern PPOTrainer uses
# ---------------------------------------------------------------------------


class TinyNet(nn.Module):
    """Three-layer MLP, no shared params, no batch norm — keeps DDP semantics
    pristine so we can isolate the wrapping/all-reduce behaviour."""

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _make_trainer(device: torch.device) -> tuple[TinyNet, nn.Module, torch.optim.Optimizer]:
    """Mirrors PPOTrainer.__init__: keep an unwrapped reference, wrap with
    DDP only when a process group is initialised, build optimiser over the
    unwrapped params."""
    net = TinyNet().to(device)
    ddp_kwargs: dict[str, Any] = {"find_unused_parameters": False}
    if device.type == "cuda" and device.index is not None:
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    if dist.is_initialized():
        wrapped = DistributedDataParallel(net, **ddp_kwargs)
    else:
        wrapped = net
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    return net, wrapped, opt


# ---------------------------------------------------------------------------
# Worker entry points
# ---------------------------------------------------------------------------


def _setup_pg(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)


def _teardown_pg() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker_grad_sync(
    rank: int,
    world: int,
    port: int,
    out: list,
) -> None:
    """Verify that DDP all-reduces gradients: every rank sees the same
    parameter values after one optimiser step regardless of which rank's
    local data was used."""
    _setup_pg(rank, world, port)
    try:
        torch.manual_seed(123)  # SAME seed → same init across ranks
        device = torch.device("cpu")
        net, wrapped, opt = _make_trainer(device)

        # Each rank uses DIFFERENT input data (this is the whole point of DP).
        torch.manual_seed(rank * 10_000 + 1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)

        # One forward+backward+step; gradients all-reduce inside backward().
        wrapped.train()
        opt.zero_grad()
        out_t = wrapped(x)
        loss = ((out_t - y) ** 2).mean()
        loss.backward()
        opt.step()

        # Snapshot parameters after step. Under correct DDP they MUST match
        # bit-for-bit across ranks (same init + averaged gradients).
        with torch.no_grad():
            flat = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
        out.append((rank, flat.tolist()))
    finally:
        _teardown_pg()


def _worker_minibatch_min(
    rank: int,
    world: int,
    port: int,
    local_n: list[int],
    out: list,
) -> None:
    """Verify the all_reduce(MIN) pattern used in ``train_epoch`` for
    aligning minibatch counts across ranks."""
    _setup_pg(rank, world, port)
    try:
        n = torch.tensor(local_n[rank], dtype=torch.long)
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
        out.append((rank, int(n.item())))
    finally:
        _teardown_pg()


def _worker_broadcast_stop(
    rank: int,
    world: int,
    port: int,
    out: list,
) -> None:
    """Verify that the early-stop broadcast pattern from train.py works:
    rank 0 decides, every rank sees the same verdict."""
    _setup_pg(rank, world, port)
    try:
        device = torch.device("cpu")
        # Match train.py: pre-allocate a 1-element long tensor on every rank.
        flag = torch.zeros(1, dtype=torch.long, device=device)
        if rank == 0:
            flag.fill_(1)  # rank 0 votes "stop"
        dist.broadcast(flag, src=0)
        out.append((rank, int(flag.item())))
    finally:
        _teardown_pg()


def _worker_save_unwrapped(
    rank: int,
    world: int,
    port: int,
    save_dir: str,
    out: list,
) -> None:
    """Verify that the state_dict saved is the UNWRAPPED model's keys (no
    leading 'module.' prefix), so it loads cleanly into a different
    world_size or single-process resume.

    Reproduces ``PPOTrainer.state_dict / load_state_dict``: save references
    ``self._policy_unwrapped.state_dict()``, NOT the DDP-wrapped one."""
    _setup_pg(rank, world, port)
    try:
        torch.manual_seed(7 + rank)
        device = torch.device("cpu")
        net, wrapped, opt = _make_trainer(device)

        # Train one step so weights diverge from init.
        x = torch.randn(2, 8)
        y = torch.randn(2, 4)
        wrapped.train()
        opt.zero_grad()
        ((wrapped(x) - y) ** 2).mean().backward()
        opt.step()

        if rank == 0:
            sd = {
                # PPOTrainer / Arr / Belief all save the unwrapped state.
                "policy": net.state_dict(),
                "world_size_at_save": world,
            }
            torch.save(sd, os.path.join(save_dir, "ckpt.pt"))
            keys = list(sd["policy"].keys())
            out.append((rank, keys))
        else:
            out.append((rank, None))
    finally:
        _teardown_pg()


def _worker_grad_nan_guard(
    rank: int,
    world: int,
    port: int,
    nan_rank: int,
    out: list,
) -> None:
    """Verify the gradient-NaN guard: if ANY rank has a NaN gradient, EVERY
    rank must skip optimizer.step() and the parameters must remain unchanged
    (no single rank gets to update while others freeze).

    This is the exact pattern PPOTrainer / ArrangementPPOTrainer use after
    the v33a R107 NaN crash. See ppo.py "Gradient NaN/Inf guard (DDP-safe)".
    """
    _setup_pg(rank, world, port)
    try:
        torch.manual_seed(7)
        device = torch.device("cpu")
        net, wrapped, opt = _make_trainer(device)

        # Snapshot params before step.
        before = torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])

        # Forward+backward as usual.
        torch.manual_seed(rank * 1000 + 1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        wrapped.train()
        opt.zero_grad()
        loss = ((wrapped(x) - y) ** 2).mean()
        loss.backward()
        # Inject a NaN into one rank's gradients.
        if rank == nan_rank:
            with torch.no_grad():
                next(net.parameters()).grad.fill_(float("nan"))
        # Mirror PPOTrainer's guard: per-rank flag, all_reduce(MAX), skip
        # optimizer.step() if anyone has NaN.
        any_nan = any(
            (p.grad is not None and not torch.isfinite(p.grad).all())
            for p in net.parameters()
        )
        flag = torch.tensor(1 if any_nan else 0, dtype=torch.long)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if int(flag.item()) != 0:
            opt.zero_grad(set_to_none=True)
            # SKIP step
        else:
            opt.step()

        # Snapshot params after.
        after = torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])
        # Critical: every rank's params must be UNCHANGED.
        unchanged = torch.allclose(before, after, atol=0.0, rtol=0.0)
        out.append((rank, unchanged, int(flag.item())))
    finally:
        _teardown_pg()


def _worker_load_into_single_process(
    rank: int,
    world: int,
    port: int,
    save_dir: str,
    out: list,
) -> None:
    """Build a fresh single-process trainer and load the ckpt saved by the
    DDP-2 worker above. Pass iff load_state_dict succeeds with no
    'unexpected key' or 'missing key' errors."""
    # NOTE: world=1 here — we explicitly DO NOT init a process group, to
    # mirror "single-process resume from DDP-trained ckpt" usage.
    device = torch.device("cpu")
    net = TinyNet().to(device)

    sd = torch.load(os.path.join(save_dir, "ckpt.pt"), map_location=device)
    keys_in_ckpt = list(sd["policy"].keys())
    # Critical assertion: no 'module.' prefix (which would mean we accidentally
    # saved the DDP-wrapped state).
    assert not any(k.startswith("module.") for k in keys_in_ckpt), (
        f"Checkpoint keys must NOT have DDP 'module.' prefix; got {keys_in_ckpt}"
    )
    # Strict load → fails loudly on any key mismatch.
    net.load_state_dict(sd["policy"], strict=True)
    out.append((rank, keys_in_ckpt))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3])
def test_ddp_grad_sync_keeps_params_consistent(world: int) -> None:
    """All ranks must end up with bit-identical parameters after one DDP step."""
    mgr = mp.get_context("spawn").Manager()
    out = mgr.list()
    port = 29500 + world * 7  # avoid collision when running tests in parallel
    mp.spawn(
        _worker_grad_sync, args=(world, port, out), nprocs=world, join=True,
    )
    assert len(out) == world, f"expected {world} reports, got {len(out)}"
    by_rank = {r: torch.tensor(p) for r, p in out}
    ref = by_rank[0]
    for r, p in by_rank.items():
        assert torch.allclose(ref, p, atol=0.0, rtol=0.0), (
            f"rank {r} params drifted from rank 0 — DDP all-reduce broken"
        )


def test_minibatch_count_min_alignment() -> None:
    """all_reduce(MIN) over per-rank batch counts gives every rank the
    smallest count, mirroring how ``PPOTrainer.train_epoch`` truncates."""
    world = 4
    local_n = [12, 8, 15, 10]  # min = 8
    mgr = mp.get_context("spawn").Manager()
    out = mgr.list()
    port = 29550
    mp.spawn(
        _worker_minibatch_min, args=(world, port, local_n, out),
        nprocs=world, join=True,
    )
    assert len(out) == world
    for _, v in out:
        assert v == 8, f"expected MIN=8 on every rank, got {v}"


def test_broadcast_early_stop_signal() -> None:
    """rank 0 sets the early-stop flag; broadcast must propagate the
    decision to every rank exactly as ``train.py`` relies on."""
    world = 3
    mgr = mp.get_context("spawn").Manager()
    out = mgr.list()
    port = 29570
    mp.spawn(
        _worker_broadcast_stop, args=(world, port, out),
        nprocs=world, join=True,
    )
    assert len(out) == world
    for _, v in out:
        assert v == 1, f"every rank must see flag=1; got {v}"


@pytest.mark.parametrize("nan_rank", [0, 1, 2])
def test_grad_nan_guard_freezes_all_ranks(nan_rank: int) -> None:
    """If ANY rank emits a NaN gradient, every rank must skip optimizer.step.

    This is the exact bug v33a hit at R107 — without an all-rank skip, some
    ranks step into NaN params and DDP's parameter replicas diverge, producing
    permanent EMA contamination and the win_rate=0.5 collapse that followed.
    """
    world = 3
    mgr = mp.get_context("spawn").Manager()
    out = mgr.list()
    port = 29610 + nan_rank
    mp.spawn(
        _worker_grad_nan_guard, args=(world, port, nan_rank, out),
        nprocs=world, join=True,
    )
    assert len(out) == world
    # Every rank must have detected the all-reduced "bad" flag.
    for r, unchanged, flag in out:
        assert flag == 1, f"rank {r} did not see all-reduce flag (got {flag})"
        assert unchanged, (
            f"rank {r} parameters changed despite NaN-on-rank-{nan_rank} — "
            f"DDP grad-NaN guard not propagating skip decision"
        )


def test_unwrapped_state_dict_loads_cross_world_size() -> None:
    """A 2-rank DDP run saves an unwrapped state_dict; a single-process
    process must be able to ``load_state_dict(strict=True)`` it.

    This is the critical invariant that makes B3 (8-card scale-up of a
    2-card B2 winner) actually work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: 2-rank DDP train + save
        save_dir = tmpdir
        mgr = mp.get_context("spawn").Manager()
        out_save = mgr.list()
        mp.spawn(
            _worker_save_unwrapped,
            args=(2, 29590, save_dir, out_save),
            nprocs=2, join=True,
        )
        rank0_keys = next((k for r, k in out_save if r == 0 and k is not None), None)
        assert rank0_keys is not None and len(rank0_keys) > 0
        # Phase 2: single-process load
        out_load: list = []
        # Run synchronously in the parent (no spawn needed for world=1).
        _worker_load_into_single_process(0, 1, 29591, save_dir, out_load)
        assert len(out_load) == 1
        loaded_keys = out_load[0][1]
        assert loaded_keys == rank0_keys, (
            f"key sets must be identical; saved={rank0_keys} loaded={loaded_keys}"
        )


if __name__ == "__main__":
    # Quick standalone runner (without pytest):
    test_ddp_grad_sync_keeps_params_consistent(2)
    print("[1/4] gradient sync OK")
    test_minibatch_count_min_alignment()
    print("[2/4] all_reduce(MIN) OK")
    test_broadcast_early_stop_signal()
    print("[3/4] broadcast(early_stop) OK")
    test_unwrapped_state_dict_loads_cross_world_size()
    print("[4/4] unwrapped state_dict cross-world-size OK")
    print("ALL DDP SMOKE TESTS PASSED")
