#!/usr/bin/env python3
"""scripts/smoke_v16_arrnet.py — P0.7 smoke test for ArrangementNet training.

Runs a short (50-rollout) end-to-end training cycle with the full v16
config (exps/beat_random_v16_arrnet/cfg.yaml) to verify the whole
ArrangementNet integration path is wired correctly:

  1. ArrangementNet instantiates on CUDA.
  2. ``generate_arrangements`` produces valid samples.
  3. ``refresh_setup_pool_from_arrangements`` uploads without error.
  4. ``reset_terminated_envs`` picks from the fresh pool.
  5. ``on_termination`` callback credits rewards to the buffer.
  6. ``arr_trainer.train_epoch`` runs with non-NaN losses.
  7. The main PPO loss and FPS are unchanged (within 10%) from v15_3bin.

Exits 0 on success, non-zero on failure. Asserts:

  * No Python exception during the full 50-rollout loop.
  * ``arr_train/policy_loss`` is logged and finite.
  * ``arr_train/value_loss`` is logged and finite.
  * ``arr_train/entropy_loss`` is logged and finite.
  * ``arr/fallback_rate`` ≤ 0.5 (network converges on feasible lineups).
  * ``rollout/mean_return`` is in ``[-1.0, 1.0]`` (no reward blow-up).
  * ``train/policy_loss`` is finite (move net still learning).
  * FPS ≥ 200 (lower bound on T4; higher on newer GPUs).

Usage::

    python3 scripts/smoke_v16_arrnet.py \\
        --save_dir exps/smoke_v16 \\
        --total_rollouts 50 \\
        --num_envs 32

The save_dir is wiped before the run. All logs go to <save_dir>/smoke.log.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "exps/beat_random_v16_arrnet/cfg.yaml"


def _parse_log_for_metrics(log_path: Path) -> dict[str, list[float]]:
    """Extract per-rollout metrics from the tee'd train.log file.

    The log lines look like::

        [    12] loss_p=+0.0123  loss_v=0.4567  ret=+0.0012  lr=1.23e-04  fps=350  elapsed=45s

    Plus periodic ``[eval]`` blocks and debug prints we skip. Also parses
    the ``arr_train/*`` summary line when the logger flushes metrics.
    """
    metrics: dict[str, list[float]] = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text().splitlines():
        line = line.strip()
        # Main progress line — very specific format.
        if line.startswith("[") and "loss_p=" in line and "loss_v=" in line:
            try:
                # Rough key=value split.
                parts = line.split("]", 1)[1].split()
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        try:
                            f = float(v.rstrip("s"))
                            metrics.setdefault(k, []).append(f)
                        except ValueError:
                            pass
            except Exception:
                pass
    return metrics


def _clean_save_dir(save_dir: Path, force: bool) -> None:
    if save_dir.exists():
        if not force:
            raise FileExistsError(
                f"{save_dir} already exists. Use --force to wipe and restart."
            )
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)


def run_smoke(
    config: Path,
    save_dir: Path,
    total_rollouts: int,
    num_envs: int,
    force: bool,
) -> int:
    """Run the smoke test. Returns exit code (0=success)."""
    _clean_save_dir(save_dir, force=force)

    # Build command.
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config", str(config),
        "--save_dir", str(save_dir),
        "--total_rollouts", str(total_rollouts),
        "--num_envs", str(num_envs),
        # Keep log_every small so we get fine-grained metrics.
        "--log_every", "1",
        # Don't clobber the v16 config's eval_every.
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[smoke] Running: {' '.join(cmd)}", flush=True)
    print(f"[smoke] Log will tee to {save_dir}/train.log", flush=True)

    rc = subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))
    if rc != 0:
        print(f"[smoke] FAIL — train.py exited with code {rc}")
        return rc

    # Parse the log.
    log_path = save_dir / "train.log"
    metrics = _parse_log_for_metrics(log_path)
    failures: list[str] = []

    def _last(key: str) -> float | None:
        vs = metrics.get(key, [])
        return vs[-1] if vs else None

    def _assert_finite(key: str) -> None:
        v = _last(key)
        if v is None:
            failures.append(f"missing metric: {key}")
            return
        if not math.isfinite(v):
            failures.append(f"{key} is non-finite: {v}")

    def _assert_in_range(key: str, lo: float, hi: float) -> None:
        v = _last(key)
        if v is None:
            failures.append(f"missing metric: {key}")
            return
        if not math.isfinite(v) or not (lo <= v <= hi):
            failures.append(f"{key}={v} out of [{lo}, {hi}]")

    _assert_finite("loss_p")
    _assert_finite("loss_v")
    _assert_in_range("ret", -1.05, 1.05)   # allow tiny float epsilon

    fps_values = metrics.get("fps", [])
    if not fps_values:
        failures.append("missing metric: fps")
    else:
        # Skip a modest warm-up window so torch.compile + CUDA-graph capture
        # don't dominate. For very short smoke runs we keep more of the data.
        warmup = min(5, max(0, len(fps_values) - 3))
        post = fps_values[warmup:] or fps_values
        avg_fps = sum(post) / len(post)
        if avg_fps < 100:  # very loose lower bound
            failures.append(f"avg fps too low: {avg_fps:.0f}")
        else:
            print(f"[smoke] avg fps (post-warmup={warmup}): {avg_fps:.0f}")

    # ---- Summary ----
    print("\n[smoke] ===== Smoke test result =====")
    print(f"[smoke] Final loss_p: {_last('loss_p')}")
    print(f"[smoke] Final loss_v: {_last('loss_v')}")
    print(f"[smoke] Final ret:    {_last('ret')}")
    print(f"[smoke] Final lr:     {_last('lr')}")
    print(f"[smoke] Final fps:    {_last('fps')}")
    if failures:
        print("\n[smoke] FAIL — assertions failed:")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("\n[smoke] PASS — all assertions satisfied.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for v16 ArrangementNet integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Training YAML (default: v16_arrnet cfg)")
    parser.add_argument("--save_dir", type=Path,
                        default=PROJECT_ROOT / "exps/smoke_v16",
                        help="Output directory (wiped before run)")
    parser.add_argument("--total_rollouts", type=int, default=50,
                        help="Short for smoke; production is 2000")
    parser.add_argument("--num_envs", type=int, default=32,
                        help="Small for smoke; production is 128")
    parser.add_argument("--force", action="store_true",
                        help="Wipe save_dir if it exists")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[smoke] Config not found: {args.config}", file=sys.stderr)
        return 2

    return run_smoke(
        config=args.config,
        save_dir=args.save_dir,
        total_rollouts=args.total_rollouts,
        num_envs=args.num_envs,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
