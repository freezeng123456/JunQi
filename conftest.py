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
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_JUNQI_RL = os.path.join(_REPO_ROOT, "junqi_rl")

# Repo root must precede site-packages so local edits win over any installed
# version. junqi_rl/ is added so the bare ``import junqi_cuda`` works.
for _p in (_REPO_ROOT, _JUNQI_RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
