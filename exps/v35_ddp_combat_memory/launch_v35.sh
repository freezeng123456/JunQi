#!/bin/bash
# launch_v35.sh — start v35 DDP training on 2× H20.
#
# F-7 (2026-05-10): removed the previously-hardcoded
# ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. That env var was
# documented in docs/PROGRESS.md as "perturbs fp16 numerics enough to
# change cuDNN kernel dispatch and wreck training trajectories", and was
# only kept on T4 to avoid OOM at end of v28. v35 on H20 has 96 GB HBM
# and does not need it; bf16 has not been verified against this allocator
# tweak. Set it explicitly in your shell if you really want it.
set -e
export CUDA_HOME=/jizhicfs/yuyechen/miniconda3/envs/cl
export LD_LIBRARY_PATH=/jizhicfs/yuyechen/miniconda3/envs/cl/lib
export PATH=/jizhicfs/denryli/venv/bin:/usr/bin:/usr/local/bin
export CUDA_VISIBLE_DEVICES=0,1
cd /root/JunQi
exec bash scripts/launch_h20_ddp.sh exps/v35_ddp_combat_memory/cfg.yaml 2
