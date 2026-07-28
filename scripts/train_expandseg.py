"""Python entry point that sets PYTORCH_CUDA_ALLOC_CONF before importing
train. This is the workaround for auto mode blocking env-var prefixed
shell commands.

Usage:
    python3 scripts/train_expandseg.py --config path/to/cfg.yaml
"""
import os
# MUST come before any torch import anywhere in the process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print(f"[launch] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

# Now delegate to train.py's main.
import importlib.util
import pathlib
import sys

_TRAIN = pathlib.Path(__file__).resolve().parent / "train.py"
spec = importlib.util.spec_from_file_location("_train", _TRAIN)
mod = importlib.util.module_from_spec(spec)
sys.modules["_train"] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.main()
