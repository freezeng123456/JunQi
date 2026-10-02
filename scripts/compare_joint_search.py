"""Current facts-only joint-model command. Historical runs use explicit legacy_* scripts."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_facts_joint import main


if __name__ == "__main__":
    main()
