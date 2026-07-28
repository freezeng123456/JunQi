#!/bin/bash
set -e
export CUDA_HOME=/jizhicfs/yuyechen/miniconda3/envs/cl
export PATH=$CUDA_HOME/bin:/jizhicfs/denryli/venv/bin:/usr/bin:/usr/local/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib
export CUDA_ARCH=90
cd /root/JunQi
# Patch the build script in-place inside the repo so `__file__` resolves
# to /root/JunQi/... and CMake finds CMakeLists.txt at src/env/cuda/.
sed -e "s|/usr/local/cuda/bin/nvcc|$CUDA_HOME/bin/nvcc|g" build_cuda.py > /root/JunQi/build_cuda_h20.py
exec /jizhicfs/denryli/venv/bin/python /root/JunQi/build_cuda_h20.py
