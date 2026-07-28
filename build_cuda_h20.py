#!/usr/bin/env python3
"""
build_cuda.py — Build the junqi_cuda CUDA extension.

Run from the repo root:
    python3 build_cuda.py

Requires:
    pybind11     (pip install pybind11)
    cmake        (pip install cmake)
    nvcc         (/jizhicfs/yuyechen/miniconda3/envs/cl/bin/nvcc)
"""
import os
import subprocess
import sys
import sysconfig

# ---- Paths ----------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
CUDA_SRC   = os.path.join(REPO_ROOT, "src", "env", "cuda")
BUILD_DIR  = os.path.join(CUDA_SRC, "build")
INSTALL_TO = os.path.join(REPO_ROOT, "junqi_rl")   # so 'import junqi_cuda' works

# ---- Locate cmake ---------------------------------------------------------
import cmake as _cmake_pkg
CMAKE_BIN = os.path.join(os.path.dirname(_cmake_pkg.__file__), "data", "bin", "cmake")

# ---- pybind11 cmake dir ---------------------------------------------------
import pybind11
PYBIND11_DIR = pybind11.get_cmake_dir()

# ---- Python info ----------------------------------------------------------
PYTHON_EXEC = sys.executable
EXT_SUFFIX  = sysconfig.get_config_var("EXT_SUFFIX")

# ---- CUDA arch: auto-detect or default ------------------------------------
CUDA_ARCH = os.environ.get("CUDA_ARCH", "75;80;86;89;90")

print("=" * 60)
print(f"cmake   : {CMAKE_BIN}")
print(f"pybind11: {PYBIND11_DIR}")
print(f"Python  : {PYTHON_EXEC}")
print(f"CUDA src: {CUDA_SRC}")
print(f"Build   : {BUILD_DIR}")
print(f"Install : {INSTALL_TO}")
print(f"Arch    : {CUDA_ARCH}")
print("=" * 60)

os.makedirs(BUILD_DIR, exist_ok=True)

# ---- cmake configure ------------------------------------------------------
configure_cmd = [
    CMAKE_BIN,
    CUDA_SRC,
    "-B", BUILD_DIR,
    f"-Dpybind11_DIR={PYBIND11_DIR}",
    f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",
    f"-DCMAKE_CUDA_COMPILER=/jizhicfs/yuyechen/miniconda3/envs/cl/bin/nvcc",
    f"-DCMAKE_INSTALL_PREFIX={INSTALL_TO}",
    "-DCMAKE_BUILD_TYPE=Release",
]
print("\n[1/3] Configuring ...")
print(" ".join(configure_cmd))
ret = subprocess.run(configure_cmd, cwd=CUDA_SRC)
if ret.returncode != 0:
    print("CMake configure FAILED.")
    sys.exit(ret.returncode)

# ---- cmake build ----------------------------------------------------------
build_cmd = [CMAKE_BIN, "--build", BUILD_DIR, "--", "-j8"]
print("\n[2/3] Building ...")
ret = subprocess.run(build_cmd)
if ret.returncode != 0:
    print("CMake build FAILED.")
    sys.exit(ret.returncode)

# ---- cmake install --------------------------------------------------------
install_cmd = [CMAKE_BIN, "--install", BUILD_DIR]
print("\n[3/3] Installing ...")
ret = subprocess.run(install_cmd)
if ret.returncode != 0:
    print("CMake install FAILED.")
    sys.exit(ret.returncode)

# ---- Verify ---------------------------------------------------------------
so_files = [f for f in os.listdir(INSTALL_TO) if f.startswith("junqi_cuda") and f.endswith(".so")]
if so_files:
    print(f"\nSuccess! Built: {so_files[0]}")
else:
    print("\nWarning: .so not found in junqi_rl/ after install.")
    sys.exit(1)
