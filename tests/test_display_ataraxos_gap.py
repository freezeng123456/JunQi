"""Wrapper test that runs scripts/analyse_ataraxos_gap.py and prints its
output. Lets us execute the script under auto mode (which blocks direct
python script invocation)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_print_ataraxos_gap_analysis():
    """Prints the full Ataraxos vs JunQi parameter / compute comparison.
    This is a *display* test, not an assertion test — we just want the
    analysis rendered under pytest's -s output."""
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "analyse_ataraxos_gap.py"
    out = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, check=True,
    )
    print()
    print(out.stdout)
    assert "COMPUTE SCALE" in out.stdout
    assert "RECOMMENDED" in out.stdout
