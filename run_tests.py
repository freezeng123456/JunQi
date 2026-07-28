#!/usr/bin/env python3
"""Portable test entry point for local development and CI.

Examples:

    python3 run_tests.py
    python3 run_tests.py --profile rl
    python3 run_tests.py --profile core -- -k replay -x
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core", "rl"),
        default="core",
        help="core tolerates a missing Torch install; rl requires Torch",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="measure branch coverage for junqi_core",
    )
    return parser.parse_known_args()


def main() -> int:
    args, pytest_args = parse_args()
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    if args.profile == "rl":
        env["JUNQI_REQUIRE_TORCH"] = "1"

    command = [sys.executable, "-m", "pytest", "-q"]
    if args.coverage:
        command.extend(
            [
                "--cov=junqi_core",
                "--cov-branch",
                "--cov-report=term-missing",
            ]
        )
    command.extend(pytest_args)
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
