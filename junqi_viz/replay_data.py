"""Framework-neutral serialization for the local replay web client."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from junqi_core.board import CELL_TABLE
from junqi_core.replay_viewer import ReplayViewer, frame_summary, load_replay
from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_core.rules import ALL_SEATS

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

SEAT_LABELS = {
    "SOUTH": "南",
    "WEST": "西",
    "NORTH": "北",
    "EAST": "东",
}

EVENT_LABELS = {
    "MOVE": "移动",
    "EAT": "吃子",
    "KILLED": "被击杀",
    "BOMB": "炸弹同归",
}

SOURCE_LABELS = {
    "policy_sample": "策略采样",
    "policy_greedy": "策略贪心",
    "random_opponent": "随机对手",
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
        self._event_index = self._build_event_index()

    @property
    def length(self) -> int:
        return self._viewer.length

    @staticmethod
    def _piece_payload(piece: Any | None) -> dict[str, Any] | None:
        if piece is None:
            return None
        seat = piece.seat.name
        piece_type = piece.piece_type.name
        return {
            "piece_id": int(piece.piece_id),
            "seat": seat,
            "seat_label": SEAT_LABELS.get(seat, seat),
            "team": int(piece.seat.team),
            "type": piece_type,
            "label": PIECE_LABELS.get(piece_type, piece_type),
        }

    @staticmethod
    def _seat_payload(state: Any) -> dict[str, dict[str, Any]]:
        return {
            seat.name: {
                "seat": seat.name,
                "label": SEAT_LABELS.get(seat.name, seat.name),
                "team": int(seat.team),
                "alive_pieces": sum(
                    1 for piece in state.pieces.values() if piece.seat is seat
                ),
                "dead": bool(state.info[seat].dead),
                "flag_revealed": bool(state.info[seat].flag_revealed),
            }
            for seat in ALL_SEATS
        }

    def _build_event_index(self) -> list[dict[str, Any]]:
        """Build jump targets for combat and Q12 seat-death moments."""

        events: list[dict[str, Any]] = []
        for step in range(1, self.length + 1):
            frame = self._viewer.seek(step)
            result = frame.result
            if result is None:
                continue
            is_key_event = (
                result.event.name != "MOVE"
                or bool(result.flag_captured)
                or bool(result.flag_reveal_src)
                or bool(result.flag_reveal_dst)
                or bool(result.seats_died_this_step)
            )
            if not is_key_event:
                continue
            action = frame.action
            policy = frame.policy
            events.append(
                {
                    "step": int(step),
                    "seat": action.seat.name if action is not None else None,
                    "seat_label": (
                        SEAT_LABELS.get(action.seat.name, action.seat.name)
                        if action is not None
                        else None
                    ),
                    "team": int(action.seat.team) if action is not None else None,
                    "src": list(action.src) if action is not None else None,
                    "dst": list(action.dst) if action is not None else None,
                    "event": result.event.name,
                    "event_label": EVENT_LABELS.get(result.event.name, result.event.name),
                    "source": policy.action_source_name if policy is not None else None,
                    "source_label": (
                        SOURCE_LABELS.get(policy.action_source_name, policy.action_source_name)
                        if policy is not None
                        else None
                    ),
                    "value": float(policy.value) if policy is not None else None,
                    "flag_captured": bool(result.flag_captured),
                    "seats_died": [seat.name for seat in result.seats_died_this_step],
                    "seats_died_labels": [
                        SEAT_LABELS.get(seat.name, seat.name)
                        for seat in result.seats_died_this_step
                    ],
                    "terminated_after": bool(result.terminated_after),
                }
            )
        self._viewer.reset()
        return events

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
            "has_beliefs": bool(
                isinstance(self.source, TrajectoryWithPolicy)
                and self.source.beliefs is not None
            ),
            "belief_shape": (
                list(self.source.beliefs.shape)
                if isinstance(self.source, TrajectoryWithPolicy)
                and self.source.beliefs is not None
                else None
            ),
            "meta": dict(self.source.meta),
            "cells": cells,
            "key_events": list(self._event_index),
        }

    def frame(self, step: int) -> dict[str, Any]:
        if not 0 <= step <= self.length:
            raise IndexError(f"replay step {step} outside [0, {self.length}]")
        with self._lock:
            frame = self._viewer.seek(step)
            previous = None
            if step > 0:
                previous = self._viewer.seek(step - 1)
                # Keep the cursor aligned with the frame returned to callers.
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
            payload["seat_info"] = self._seat_payload(frame.state)
            payload["state"] = {
                "move_counter": int(frame.state.move_counter),
                "moves_since_last_combat": int(frame.state.moves_since_last_combat),
                "terminated": bool(frame.state.terminated),
                "winner_team": frame.state.winner_team,
                "draw": bool(frame.state.draw),
            }
            alive = payload["alive_pieces"]
            payload["alive_teams"] = {
                "team0": int(alive["SOUTH"] + alive["NORTH"]),
                "team1": int(alive["WEST"] + alive["EAST"]),
            }
            if payload.get("action") is not None:
                action = frame.action
                assert action is not None
                payload["action"]["seat_label"] = SEAT_LABELS.get(
                    action.seat.name, action.seat.name
                )
                payload["action"]["team"] = int(action.seat.team)
                before_state = previous.state if previous is not None else None
                payload["move_detail"] = {
                    "seat": action.seat.name,
                    "seat_label": SEAT_LABELS.get(action.seat.name, action.seat.name),
                    "team": int(action.seat.team),
                    "src": list(action.src),
                    "dst": list(action.dst),
                    "src_piece": self._piece_payload(
                        before_state.pieces.get(action.src) if before_state is not None else None
                    ),
                    "dst_piece": self._piece_payload(
                        before_state.pieces.get(action.dst) if before_state is not None else None
                    ),
                }
            else:
                payload["move_detail"] = None
            if payload.get("result") is not None:
                result = frame.result
                assert result is not None
                payload["result"]["event_label"] = EVENT_LABELS.get(
                    result.event.name, result.event.name
                )
                payload["result"]["seats_died_labels"] = [
                    SEAT_LABELS.get(seat.name, seat.name)
                    for seat in result.seats_died_this_step
                ]
                payload["result"]["winner_team_after"] = result.winner_team_after
            if frame.policy is not None:
                policy = payload["policy"]
                policy["source_label"] = SOURCE_LABELS.get(
                    frame.policy.action_source_name, frame.policy.action_source_name
                )
                policy["chosen_probability"] = None
                policy["chosen_rank"] = None
                top_actions = []
                for rank, (action_id, probability) in enumerate(
                    zip(
                        frame.policy.top_action_ids,
                        frame.policy.top_probs,
                        strict=False,
                    ),
                    start=1,
                ):
                    probability = float(probability)
                    action_id = int(action_id)
                    if probability <= 0.0:
                        continue
                    src_flat = action_id // 289
                    dst_flat = action_id % 289
                    chosen = action_id == int(frame.policy.chosen_action_id)
                    top_actions.append(
                        {
                            "rank": rank,
                            "action_id": action_id,
                            "probability": probability,
                            "chosen": chosen,
                            "src": [int(src_flat % 17), int(src_flat // 17)],
                            "dst": [int(dst_flat % 17), int(dst_flat // 17)],
                        }
                    )
                    if chosen:
                        policy["chosen_probability"] = probability
                        policy["chosen_rank"] = rank
                policy["top_actions"] = top_actions
                policy["has_beliefs"] = bool(
                    self.source.beliefs is not None
                    if isinstance(self.source, TrajectoryWithPolicy)
                    else False
                )
                policy["belief_step"] = step - 1 if policy["has_beliefs"] and step > 0 else None
            else:
                payload["policy"] = None
            return payload


__all__ = ["PIECE_LABELS", "ReplayData"]
