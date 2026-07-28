#!/bin/bash
# Launch train.py with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# Workaround for Claude auto-mode blocking inline env-var prefixed commands.
# Used by v28 to eliminate the allocator fragmentation that OOMed v26/v27.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[launch] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
exec python3 scripts/train.py "$@"
