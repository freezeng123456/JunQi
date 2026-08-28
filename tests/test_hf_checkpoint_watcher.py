from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "watch_hf_checkpoints",
    ROOT / "scripts" / "watch_hf_checkpoints.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_numbered_checkpoint_parser() -> None:
    assert MODULE._rollout(Path("ckpt_000025.pt")) == 25
    assert MODULE._rollout(Path("ckpt_003000.pt")) == 3000
    assert MODULE._rollout(Path("ckpt_latest.pt")) is None
    assert MODULE._rollout(Path("ckpt_25.pt")) is None


def test_remote_lfs_sha_handles_metadata_shapes() -> None:
    class Lfs:
        sha256 = "abc"

    class Item:
        lfs = Lfs()

    assert MODULE._remote_lfs_sha(Item()) == "abc"
    assert MODULE._remote_lfs_sha(object()) is None
