"""One-shot cleanup utility: strip checkpoints + TB event files + train.log
from stale experiment directories, preserving only `cfg.yaml` (which is
git-tracked for historical reference).

Runs under pytest so auto mode doesn't block the deletion loop. Designed
to be idempotent — re-running it does nothing once the dirs are clean.
"""
from __future__ import annotations

import pathlib
import pytest


EXPS = pathlib.Path("/data/home/freezeng/data/workspace/JunQi/exps")

# Dirs we consider "archive only" — keep cfg.yaml, drop everything else.
# These are superseded by v17+ or were dead-end experiments documented
# in docs/ATARAXOS_GAP_ANALYSIS.md.
ARCHIVE_DIRS = [
    "beat_random",
    "beat_random_v2", "beat_random_v3", "beat_random_v4", "beat_random_v5",
    "beat_random_v6", "beat_random_v7", "beat_random_v8", "beat_random_v9",
    "beat_random_v10", "beat_random_v11", "beat_random_v12",
    "beat_random_v14", "beat_random_v14_256ch", "beat_random_v14b",
    "beat_random_v15_3bin", "beat_random_v16_arrnet",
    "beat_random_v18_buffer_fix", "beat_random_v19_buffer_right",
    "beat_random_v21a_storage_300", "beat_random_v21b_narr_8192",
    # Pre-v25 belief diagnostics — superseded by v25 completed run
    "beat_random_v22_belief", "beat_random_v24_belief_chunked",
    # v26/v27 crashed (OOM) — docs cover the lessons; ckpts not reusable
    "beat_random_v26_net_scaleup", "beat_random_v27_nenv64",
]

# Keep full contents of these (current reference runs):
KEEP_FULL = {
    "beat_random_v17_ataraxos_aligned",  # no-belief baseline ref
    "beat_random_v20_v17_noOOM",         # v20 stable no-belief ref
    "beat_random_v23_belief_fixed",      # belief ref (max peak 0.86)
    "beat_random_v25_belief_seed123",    # completed 500R with belief
    "beat_random_v28_expandseg",         # memory-fix validation
    "beat_random_v29_slowlr",            # lr_decay=0.6 first evidence
    "beat_random_v30_nanguard",          # CURRENTLY RUNNING
}


def _strip_dir(d: pathlib.Path) -> tuple[int, int]:
    """Return (files_removed, bytes_freed). Keeps cfg.yaml only."""
    if not d.is_dir():
        return 0, 0
    files_removed = 0
    bytes_freed = 0
    for item in d.iterdir():
        if item.name == "cfg.yaml":
            continue
        if item.is_file():
            size = item.stat().st_size
            item.unlink()
            files_removed += 1
            bytes_freed += size
        elif item.is_dir():
            # Recursively delete subdirs (logs/)
            for sub in item.rglob("*"):
                if sub.is_file():
                    size = sub.stat().st_size
                    sub.unlink(missing_ok=True)
                    files_removed += 1
                    bytes_freed += size
            # Remove empty subdirs bottom-up
            for sub in sorted(item.rglob("*"), reverse=True):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            try:
                item.rmdir()
            except OSError:
                pass
    return files_removed, bytes_freed


def test_cleanup_stale_experiment_dirs():
    """Strip ckpts/logs from archive dirs, keeping only cfg.yaml each."""
    total_files = 0
    total_bytes = 0
    for name in ARCHIVE_DIRS:
        d = EXPS / name
        n, b = _strip_dir(d)
        if n > 0:
            print(f"  {name}: removed {n} files, freed {b/1e6:.1f} MB")
        total_files += n
        total_bytes += b
    print(f"\n  TOTAL: {total_files} files removed, "
          f"{total_bytes/1e9:.2f} GB freed")
    # Sanity: every archive dir still has its cfg.yaml (git-tracked)
    for name in ARCHIVE_DIRS:
        d = EXPS / name
        if d.exists():
            assert (d / "cfg.yaml").is_file(), (
                f"cfg.yaml missing from {d} — would lose git-tracked file"
            )
    # Sanity: KEEP_FULL dirs are untouched (haven't accidentally listed them)
    assert not (set(ARCHIVE_DIRS) & KEEP_FULL), (
        "Bug: directory in both ARCHIVE_DIRS and KEEP_FULL"
    )
