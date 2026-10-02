"""Compatibility alias for the historical probability-input search prototype."""
from . import legacy_joint_search as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
