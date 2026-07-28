"""Top-level pytest conftest.

Adds ``junqi_rl/`` to ``sys.path`` so the compiled ``junqi_cuda`` extension
(installed by ``build_cuda.py`` as ``junqi_rl/junqi_cuda.cpython-*.so``)
becomes importable as ``import junqi_cuda``. This matches the import
convention used throughout the codebase
(``junqi_rl/gpu_rollout.py`` line 50, ``junqi_rl/env_gpu.py``,
``junqi_rl/gpu_world.py``, ``tests/test_gpu_*.py`` etc.).

Without this, those modules fall back to the import-time warning
"junqi_cuda import failed" and ``GpuRollout`` constructors raise
``RuntimeError: junqi_cuda extension not available``.
"""

from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_JUNQI_RL = os.path.join(_REPO_ROOT, "junqi_rl")

# Repo root must precede site-packages so local edits win over any installed
# version. junqi_rl/ is added so the bare ``import junqi_cuda`` works.
for _p in (_REPO_ROOT, _JUNQI_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_TORCH_IMPORT_SIGNALS = (
    "import torch",
    "from torch",
    "junqi_rl.networks",
    "junqi_rl.training",
)
_TORCH_REQUIRED_TEST_MODULES = {
    # Dynamically imports scripts/train.py, whose runtime is intentionally
    # Torch-backed even though several tests only inspect its config classes.
    "test_train_integration.py",
}


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Exclude Torch-only modules from the dependency-light core test run.

    Pytest imports a test module before it can apply a module-level marker.
    Direct ``import torch`` statements therefore used to abort collection on
    machines that intentionally installed only ``.[dev]``.  The RL CI job
    installs Torch and consequently collects the exact same files.

    Set ``JUNQI_REQUIRE_TORCH=1`` when a missing Torch installation should be
    treated as a configuration error instead of selecting the core suite.
    """
    if _TORCH_AVAILABLE or collection_path.suffix != ".py":
        return None
    if collection_path.parent.name != "tests":
        return None
    if collection_path.name in _TORCH_REQUIRED_TEST_MODULES:
        return True
    try:
        source = collection_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return any(signal in source for signal in _TORCH_IMPORT_SIGNALS)


def pytest_sessionstart(session: pytest.Session) -> None:
    if not _TORCH_AVAILABLE and os.environ.get("JUNQI_REQUIRE_TORCH") == "1":
        raise pytest.UsageError(
            "Torch is required for this test profile. "
            "Install the RL development dependencies with: pip install -e '.[rl,dev]'"
        )


def pytest_report_header(config: pytest.Config) -> str | None:
    if not _TORCH_AVAILABLE:
        return "JunQi test profile: core (Torch-dependent modules excluded)"
    return "JunQi test profile: core + RL"
