"""Failure-safe checkpoint publication (no Torch dependency).

Temporary files are created on the destination filesystem, flushed and fsynced
before os.replace. This protects an existing file from serialization failures;
it does not make a checkpoint plus its alias a multi-file transaction or
provide coordination between concurrent writers.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable


def atomic_write(path: str | os.PathLike[str], writer: Callable[[BinaryIO], None]) -> None:
    """Publish a completely serialized file, preserving the old one on error."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        # os.replace consumes the temporary pathname on success.
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_alias(
    checkpoint_path: str | os.PathLike[str],
    alias_path: str | os.PathLike[str],
    *,
    copy_fallback: bool = False,
) -> None:
    """Atomically replace an alias; never delete it before its successor exists.

    Relative symlinks remain usable when a checkpoint directory is moved.
    On platforms without symlink permission, an explicitly requested copy
    fallback is also written completely before publication. Without fallback,
    symlink failure is propagated to the caller instead of silently ignored.
    """
    source = Path(checkpoint_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"checkpoint source is not a file: {source}")
    requested = Path(alias_path).absolute()
    # Resolve the parent but NOT an existing alias symlink's final component.
    destination = requested.parent.resolve() / requested.name
    if source == destination:
        raise ValueError("checkpoint and alias must have different paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent,
    ) as staging:
        candidate = Path(staging) / "alias"
        try:
            # The relative target is for the final location, not staging.
            os.symlink(os.path.relpath(source, destination.parent), candidate)
        except OSError:
            if not copy_fallback:
                raise
            with source.open("rb") as src, candidate.open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
        os.replace(candidate, destination)
