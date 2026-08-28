#!/usr/bin/env python3
"""Watch a training directory and upload stable numbered checkpoints to HF.

The Hugging Face token is read once from ``JUNQI_HF_UPLOAD_TOKEN`` and removed
from the process environment immediately. It is never written to the status
file or logs. Every uploaded checkpoint is verified against the remote LFS
SHA256 before it is marked complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_ENV = "JUNQI_HF_UPLOAD_TOKEN"
CKPT_RE = re.compile(r"^ckpt_(\d{6})\.pt$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rollout(path: Path) -> int | None:
    match = CKPT_RE.match(path.name)
    return int(match.group(1)) if match else None


def _remote_lfs_sha(item: Any) -> str | None:
    lfs = getattr(item, "lfs", None)
    return getattr(lfs, "sha256", None) if lfs is not None else None


def _remote_files(api: Any, repo_id: str, revision: str) -> dict[str, Any]:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    return {item.rfilename: item for item in (info.siblings or [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stable-seconds", type=float, default=30.0)
    parser.add_argument("--stop-rollout", type=int, default=0)
    args = parser.parse_args()

    token = os.environ.pop(TOKEN_ENV, None)
    if not token:
        raise RuntimeError(f"missing required environment variable {TOKEN_ENV}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    state: dict[str, Any] = {
        "stage": "WATCHING",
        "started_at": _now(),
        "updated_at": _now(),
        "run_dir": str(args.run_dir),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "files": {},
    }
    _atomic_json(args.status, state)

    # path -> (size, mtime_ns, first time this exact pair was observed)
    observed: dict[Path, tuple[int, int, float]] = {}
    while True:
        try:
            remote = _remote_files(api, args.repo_id, args.revision)
            numbered = sorted(
                (r, path)
                for path in args.run_dir.glob("ckpt_*.pt")
                if path.is_file() and not path.is_symlink()
                if (r := _rollout(path)) is not None
            )
            for rollout, path in numbered:
                entry = state["files"].setdefault(path.name, {"rollout": rollout})
                if entry.get("state") == "VERIFIED":
                    continue

                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                previous = observed.get(path)
                if previous is None or previous[:2] != signature:
                    observed[path] = (*signature, time.monotonic())
                    entry.update({"state": "STABILIZING", "size": stat.st_size})
                    continue
                if time.monotonic() - previous[2] < args.stable_seconds:
                    continue

                entry.update({"state": "HASHING", "updated_at": _now()})
                _atomic_json(args.status, state)
                local_sha = _sha256(path)
                existing = remote.get(path.name)
                if (
                    existing is not None
                    and getattr(existing, "size", None) == stat.st_size
                    and _remote_lfs_sha(existing) == local_sha
                ):
                    entry.update({
                        "state": "VERIFIED",
                        "sha256": local_sha,
                        "verified_at": _now(),
                        "skipped_existing": True,
                    })
                    continue

                entry.update({"state": "UPLOADING", "updated_at": _now()})
                _atomic_json(args.status, state)
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=args.repo_id,
                    repo_type="model",
                    revision=args.revision,
                    commit_message=f"checkpoints: add {path.name}",
                )
                remote = _remote_files(api, args.repo_id, args.revision)
                uploaded = remote.get(path.name)
                if (
                    uploaded is None
                    or getattr(uploaded, "size", None) != stat.st_size
                    or _remote_lfs_sha(uploaded) != local_sha
                ):
                    raise RuntimeError(f"remote verification failed for {path.name}")
                entry.update({
                    "state": "VERIFIED",
                    "size": stat.st_size,
                    "sha256": local_sha,
                    "verified_at": _now(),
                })
                print(f"[hf] verified {path.name}", flush=True)

            state["updated_at"] = _now()
            state["verified_count"] = sum(
                item.get("state") == "VERIFIED"
                for item in state["files"].values()
            )
            state.pop("last_error", None)
            if args.stop_rollout > 0:
                final_name = f"ckpt_{args.stop_rollout:06d}.pt"
                if state["files"].get(final_name, {}).get("state") == "VERIFIED":
                    state.update({"stage": "COMPLETE", "finished_at": _now()})
                    _atomic_json(args.status, state)
                    return 0
            _atomic_json(args.status, state)
        except Exception as exc:
            state.update({
                "stage": "RETRYING",
                "last_error": f"{type(exc).__name__}: {exc}",
                "updated_at": _now(),
            })
            _atomic_json(args.status, state)
            print(f"[hf] retry after error: {state['last_error']}", flush=True)
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
