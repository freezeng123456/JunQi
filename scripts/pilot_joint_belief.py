"""Historical command alias. Use scripts/legacy_pilot_joint_belief.py."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import legacy_pilot_joint_belief as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def main():
    warnings.warn("Historical probability-input command; use scripts/legacy_pilot_joint_belief.py",
                  FutureWarning, stacklevel=2)
    return _implementation.main()


if __name__ == '__main__':
    main()
