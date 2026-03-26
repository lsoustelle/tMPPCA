cmake_minimum_required(VERSION 3.18)
project(tmppca_cpp LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ─────────────────────────────────────────────────────────────────
# Prefer a version installed via pip (scikit-build-core / setup.py will have
# already placed it on the path); fall back to a system install.
find_package(pybind11 CONFIG REQUIRED)

# ── Eigen ────────────────────────────────────────────────────────────────────
# 1. Try an Eigen3 CMake config (system install or conda).
# 2. Fall back to the header-only copy bundled under extern/eigen if present.
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 3.4 QUIET MODULE)
endif()
if(NOT Eigen3_FOUND)
    set(_EIGEN_FALLBACK "${CMAKE_SOURCE_DIR}/extern/eigen")
    if(EXISTS "${_EIGEN_FALLBACK}/Eigen/Dense")
        message(STATUS "Eigen3 not found via CMake — using bundled copy at ${_EIGEN_FALLBACK}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${_EIGEN_FALLBACK}")
    else()
        message(FATAL_ERROR
            "Eigen3 >= 3.4 not found.\n"
            "Install it with 'conda install -c conda-forge eigen' (preferred).")
    endif()
endif()

# ── OpenMP (optional) ────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Extension target ─────────────────────────────────────────────────────────
pybind11_add_module(tmppca_cpp
    src/bindings.cpp
)

target_include_directories(tmppca_cpp PRIVATE
    src        # core_tmppca.hpp, denoise_tmppca.hpp
)

install(TARGETS tmppca_cpp
        LIBRARY DESTINATION tmppca)

target_link_libraries(tmppca_cpp PRIVATE
    Eigen3::Eigen
)

# Release-mode optimisations (scikit-build-core sets these automatically,
# but keep them explicit so a plain cmake build also gets them).
target_compile_options(tmppca_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tmppca_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tmppca_cpp PRIVATE _OPENMP)
    message(STATUS "OpenMP found — multi-threading enabled")
else()
    message(STATUS "OpenMP not found — single-threaded build")
endif()
