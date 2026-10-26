from pathlib import Path

from junqi_rl.training.checkpoint import prune_numbered_checkpoints


def test_prune_numbered_checkpoints_preserves_aliases(tmp_path: Path) -> None:
    for name in ("ckpt_000001.pt", "ckpt_000002.pt", "ckpt_000003.pt", "ckpt_000004.pt"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "ckpt_latest.pt").write_bytes(b"alias")
    removed = prune_numbered_checkpoints(tmp_path, 2)
    assert [Path(p).name for p in removed] == ["ckpt_000001.pt", "ckpt_000002.pt"]
    assert not (tmp_path / "ckpt_000001.pt").exists()
    assert (tmp_path / "ckpt_000003.pt").exists()
    assert (tmp_path / "ckpt_000004.pt").exists()
    assert (tmp_path / "ckpt_latest.pt").exists()
