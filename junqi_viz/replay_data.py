"""Framework-neutral serialization for the local replay web client."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from junqi_core.board import CELL_TABLE
from junqi_core.replay_viewer import ReplayViewer, frame_summary, load_replay
from junqi_core.replay_with_policy import TrajectoryWithPolicy

PIECE_LABELS = {
    "JUNQI": "军旗",
    "DILEI": "地雷",
    "ZHADAN": "炸弹",
    "SILING": "司令",
    "JUNZH": "军长",
    "SHIZH": "师长",
    "LVZH": "旅长",
    "TUANZH": "团长",
    "YINGZH": "营长",
    "LIANZH": "连长",
    "PAIZH": "排长",
    "GONGB": "工兵",
}


class ReplayData:
    """Thread-safe, seekable replay data source for HTTP and tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.source = load_replay(str(self.path))
        self.source.validate()
        self._viewer = ReplayViewer(self.source)
        self._lock = threading.RLock()

    @property
    def length(self) -> int:
        return self._viewer.length

    def metadata(self) -> dict[str, Any]:
        is_policy = isinstance(self.source, TrajectoryWithPolicy)
        cells = [
            {
                "x": cell.x,
                "y": cell.y,
                "owner": cell.owner.name if cell.owner is not None else None,
                "camp": cell.is_camp,
                "stronghold": cell.is_stronghold,
                "railway": cell.is_railway,
                "nine_grid": cell.is_nine_grid,
            }
            for cell in CELL_TABLE
            if cell.is_on_board
        ]
        return {
            "filename": self.path.name,
            "length": self.length,
            "kind": "policy" if is_policy else "manual",
            "rules_version": self.source.rules_version,
            "state_version": self.source.state_version,
            "top_k": self.source.top_k if is_policy else 0,
            "meta": dict(self.source.meta),
            "cells": cells,
        }

    def frame(self, step: int) -> dict[str, Any]:
        if not 0 <= step <= self.length:
            raise IndexError(f"replay step {step} outside [0, {self.length}]")
        with self._lock:
            frame = self._viewer.seek(step)
            payload = frame_summary(frame)
            payload["length"] = self.length
            payload["pieces"] = [
                {
                    "piece_id": int(piece.piece_id),
                    "seat": piece.seat.name,
                    "type": piece.piece_type.name,
                    "label": PIECE_LABELS.get(piece.piece_type.name, piece.piece_type.name),
                    "x": int(x),
                    "y": int(y),
                }
                for (x, y), piece in sorted(
                    frame.state.pieces.items(),
                    key=lambda item: item[1].piece_id,
                )
            ]
            payload["dead_pieces"] = len(frame.state.deaths)
            if frame.policy is not None:
                policy = payload["policy"]
                policy["top_actions"] = [
                    {
                        "action_id": int(action_id),
                        "probability": float(probability),
                        "src": [
                            int((action_id // 289) % 17),
                            int((action_id // 289) // 17),
                        ],
                        "dst": [
                            int((action_id % 289) % 17),
                            int((action_id % 289) // 17),
                        ],
                    }
                    for action_id, probability in zip(
                        frame.policy.top_action_ids,
                        frame.policy.top_probs,
                        strict=False,
                    )
                    if float(probability) > 0.0
                ]
            return payload


__all__ = ["PIECE_LABELS", "ReplayData"]
