"""Persistent historical-checkpoint pool for cross-play evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LeagueEntry:
    checkpoint: str
    rollout: int
    created_at: str
    sha256: str
    rating: float = 1000.0
    games: int = 0
    tags: dict[str, Any] | None = None


class LeaguePool:
    """JSON-backed checkpoint registry with deterministic sampling and Elo."""

    def __init__(self, path: str | os.PathLike[str], *, max_entries: int = 12):
        if max_entries < 2:
            raise ValueError("max_entries must be at least 2")
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self.entries: list[LeagueEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
            raise ValueError(f"invalid league registry: {self.path}")
        self.entries = [LeagueEntry(**item) for item in raw["entries"]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "max_entries": self.max_entries,
            "entries": [asdict(item) for item in self.entries],
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def register(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        rollout: int,
        tags: dict[str, Any] | None = None,
    ) -> LeagueEntry:
        ckpt = Path(checkpoint).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        digest = _sha256_file(ckpt)
        for entry in self.entries:
            if entry.sha256 == digest:
                entry.checkpoint = str(ckpt)
                entry.rollout = int(rollout)
                if tags:
                    entry.tags = dict(tags)
                self._save()
                return entry
        entry = LeagueEntry(
            checkpoint=str(ckpt),
            rollout=int(rollout),
            created_at=datetime.now(timezone.utc).isoformat(),
            sha256=digest,
            tags=dict(tags) if tags else None,
        )
        self.entries.append(entry)
        self.entries.sort(key=lambda item: item.rollout)
        self._prune()
        self._save()
        return entry

    def _prune(self) -> None:
        """Keep endpoints plus evenly distributed historical checkpoints."""

        while len(self.entries) > self.max_entries:
            interior = self.entries[1:-1]
            remove = min(
                interior,
                key=lambda item: (
                    item.games,
                    abs(item.rating - 1000.0),
                    item.rollout,
                ),
            )
            self.entries.remove(remove)

    def sample(
        self,
        *,
        seed: int,
        exclude_rollout: int | None = None,
    ) -> LeagueEntry | None:
        candidates = [
            item
            for item in self.entries
            if item.rollout != exclude_rollout and Path(item.checkpoint).is_file()
        ]
        if not candidates:
            return None
        # Prefer similarly rated opponents, but keep every entry selectable.
        latest_rating = self.entries[-1].rating if self.entries else 1000.0
        weights = [1.0 / (1.0 + abs(item.rating - latest_rating) / 200.0) for item in candidates]
        return random.Random(seed).choices(candidates, weights=weights, k=1)[0]

    def record_match(
        self,
        first_sha256: str,
        second_sha256: str,
        *,
        first_score: float,
        games: int,
        k_factor: float = 24.0,
    ) -> None:
        if not 0.0 <= first_score <= 1.0:
            raise ValueError("first_score must be in [0, 1]")
        if games <= 0:
            raise ValueError("games must be positive")
        first = self._by_sha(first_sha256)
        second = self._by_sha(second_sha256)
        expected = 1.0 / (1.0 + 10.0 ** ((second.rating - first.rating) / 400.0))
        delta = k_factor * (first_score - expected)
        first.rating += delta
        second.rating -= delta
        first.games += games
        second.games += games
        self._save()

    def _by_sha(self, digest: str) -> LeagueEntry:
        for entry in self.entries:
            if entry.sha256 == digest:
                return entry
        raise KeyError(digest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["LeagueEntry", "LeaguePool"]
