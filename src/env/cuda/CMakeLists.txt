# CMakeLists.txt for JunQi CUDA Backend
# 
# Usage:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80;86;90
#   cmake --build build -j8
#
# The compiled module `junqi_cuda` is installed to:
#   junqi_rl/junqi_cuda.so (via setup.py install)

cmake_minimum_required(VERSION 3.18)
project(junqi_cuda LANGUAGES CXX CUDA)

# ==============================================================================
# Configuration
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architectures (Ampere, Ada, Hopper)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()

# Release build by default with -O3
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Disable LTO: pybind11 3.x enables -flto=auto by default, but LTO interacts
# badly with CUDA separable compilation — the device link object gets lost.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

# ==============================================================================
# Dependencies
# ==============================================================================

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# ==============================================================================
# Source Files
# ==============================================================================

set(JUNQI_CUDA_KERNELS
  src/tables.cu
  src/game_state.cu
  src/observation.cu
  src/belief.cu
  src/combat_memory.cu
  src/rollout_history.cu
)

set(JUNQI_CUDA_HEADERS
  include/junqi_cuda.h
  src/common.cuh
  src/tables.cuh
  src/game_state.cuh
  src/observation.cuh
  src/belief.cuh
  src/combat_memory.cuh
)

# ==============================================================================
# pybind11 Python module — all CUDA sources + bindings in one target.
#
# The standard pattern for separable CUDA compilation in a pybind11 module:
#   1. Add all .cu sources to pybind11_add_module (not a separate library).
#   2. Set CUDA_SEPARABLE_COMPILATION ON so each .cu is compiled with -dc.
#   3. Set CUDA_RESOLVE_DEVICE_SYMBOLS ON so CMake runs nvlink on this target.
#
# This is the only target, so nvlink sees all device code and resolves fatbinData.
# ==============================================================================

pybind11_add_module(junqi_cuda
  ${JUNQI_CUDA_KERNELS}
  src/bindings.cpp
)

# Include directories
target_include_directories(junqi_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(junqi_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(junqi_cuda PRIVATE ${Python3_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(junqi_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)

# Position-independent code (required for shared library)
set_property(TARGET junqi_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

# Separable compilation: each .cu compiled with -dc, then nvlink resolves device symbols
set_target_properties(junqi_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Generate GPU code for the specified architectures
# CMAKE_CUDA_ARCHITECTURES is handled automatically by CMake 3.18+

# ==============================================================================
# Installation
# ==============================================================================

# Install the shared library directly into CMAKE_INSTALL_PREFIX.
# build_cuda.py sets CMAKE_INSTALL_PREFIX=.../junqi_rl, so DESTINATION "."
# puts the .so at .../junqi_rl/junqi_cuda*.so as expected by `import junqi_cuda`.
install(TARGETS junqi_cuda
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)

# ==============================================================================
# Testing (optional, if ENABLE_TESTS is set)
# ==============================================================================

if(ENABLE_TESTS)
  enable_testing()
  
  # Create test executable (link against CUDA runtime)
  add_executable(test_legal_actions tests/test_legal_actions.cu)
  target_include_directories(test_legal_actions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_legal_actions PRIVATE CUDA::cudart)
  
  add_test(NAME LegalActionsKernelTest COMMAND test_legal_actions)
endif()

# ==============================================================================
# Print Configuration
# ==============================================================================

message(STATUS "JunQi CUDA Backend Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIR}")
message(STATUS "  PyBind11: ${pybind11_INCLUDE_DIR}")

