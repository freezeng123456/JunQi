"""Visualization and replay services for JunQi.

The package intentionally imports no web framework at module import time so
the core engine remains usable with only the base dependencies installed.
"""

from .replay_data import ReplayData


def create_app(replay_path: str):
    """Lazily import FastAPI and create a local replay application."""

    from .app import create_app as implementation

    return implementation(replay_path)


__all__ = ["ReplayData", "create_app"]
