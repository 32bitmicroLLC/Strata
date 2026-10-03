# poc/sycl/CMakeLists.txt -- standalone SYCL Phase A PoC build (see
# plans/sycl-phase-a.md). The main tree (root CMakeLists.txt, src/) is never
# touched; this project builds on its own under poc/sycl/build.
#
# Compiler: the oneAPI DPC++ driver, icpx, with -fsycl (T0 finding: icx is
# the C driver and does not link the SYCL runtime). run.sh selects it via
# -DCMAKE_CXX_COMPILER; we assert here so a manual configure fails loudly.
#
# If CMake ever fights the clang-based DPC++ driver, the documented fallback
# is to compile targets directly with icpx in run.sh (CMake is convenience,
# not the deliverable).
#
# Target convention for T2-T5 (one kernel = three small units):
#   k_<name>        STATIC library, one source  kernels/<name>.cpp
#                   (exposes strata::kernels::submit_<name>(queue, ...))
#   k_<name>_parity executable, one source drivers/<name>_parity.cpp
#   add_test(NAME k_<name>_parity COMMAND k_<name>_parity)   (TIMEOUT 120)

if(NOT DEFINED CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "pass -DCMAKE_CXX_COMPILER=$(command -v icpx) -- see poc/sycl/run.sh")
endif()
get_filename_component(_compiler_name "${CMAKE_CXX_COMPILER}" NAME)
if(NOT _compiler_name MATCHES "ic(px|x)$")
  message(FATAL_ERROR "SYCL PoC needs the oneAPI DPC++ compiler (icpx), got: ${CMAKE_CXX_COMPILER}")
endif()

cmake_minimum_required(VERSION 3.24)
project(strata_sycl_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fsycl)

# Binaries must resolve their runtime libraries without a sourced shell
# (T0 only ever ran inside a sourced setvars.sh environment). Two dirs are
# required -- verified by env -u LD_LIBRARY_PATH runs:
#   <compiler>/../lib   libsycl, libur_loader, the UR adapters
#   $ONEAPI_ROOT/umf/1.1/lib   Unified Memory Framework -- the Level Zero
#   adapter needs it for device init; without it the runtime reports
#   "No device of requested type available" even though every library loads.
# CMake's CMAKE_BUILD_RPATH emits DT_RUNPATH, which dlopen from the adapter
# does NOT consult; -Wl,--disable-new-dtags forces traditional DT_RPATH, which
# is consulted process-wide.
get_filename_component(_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(_rpath_dirs "${_compiler_dir}/../lib")
if(DEFINED ENV{ONEAPI_ROOT})
  list(APPEND _rpath_dirs "$ENV{ONEAPI_ROOT}/umf/1.1/lib")
endif()
set(_rpath_opts "-Wl,--disable-new-dtags")
foreach(_d IN LISTS _rpath_dirs)
  list(APPEND _rpath_opts "-Wl,-rpath,${_d}")
endforeach()

# The repo's include/ is on the path so the PoC can reuse the real API headers
# (SForm/Codebook) and the host dequantizer chain (dequant.hpp) unmodified.
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

enable_testing()

# T0 standing regressions + the T1 smoke driver.
# ctest order is registration order: smoke first, the bandwidth gate last, so
# a red gate is never hidden behind driver noise.
add_executable(smoke_parity   drivers/smoke_parity.cpp)

# T2: the first real kernel (plans/sycl-phase-a-t2.md).
# The static lib compiles with the global -fsycl; the consumer executable
# already carries the -fsycl link flag + rpath + UMF via the foreach below.
add_library(k_dequant_s2 STATIC kernels/dequant_s2.cpp)
add_executable(k_dequant_s2_parity drivers/dequant_s2_parity.cpp)
target_link_libraries(k_dequant_s2_parity PRIVATE k_dequant_s2)

# T3: the probe that gates the router_top10 port design (found: icpx 2026.1 has
# no subgroup shuffle/reduce API -> handler local accessors + local-memory trees).
add_executable(t3_sg_probe t3/sg_probe.cpp)

# T3: the first shared-memory kernel (plans/sycl-phase-a-t3.md).
add_library(k_router_top10 STATIC kernels/router_top10.cpp)
add_executable(k_router_top10_parity drivers/router_top10_parity.cpp)
target_link_libraries(k_router_top10_parity PRIVATE k_router_top10)

# T4: the flagship S-family GEMV (plans/sycl-phase-a-t4.md).
add_executable(t4_fp16_probe t4/fp16_probe.cpp)

# T5: the sampler (plans/sycl-phase-a-t5.md).  Three risk probes first
# (atomic_ref on local memory, device double math vs host, the 1024-thread
# reduction tree), then the kernel and the parity driver.
add_executable(t5_atomic_probe t5/atomic_probe.cpp)
add_executable(t5_math_probe   t5/math_probe.cpp)
add_executable(t5_tree_probe   t5/tree_probe.cpp)
# P2b: the fixture-argument re-probe that decides the double-math fallback
# (plans/sycl-phase-a-t5-step-1.md §1.3.1).
add_executable(t5_math_fixture_probe t5/math_fixture_probe.cpp)
# T5: the sampler kernels (plans/sycl-phase-a-t5-step-2.md) and the parity
# driver (plans/sycl-phase-a-t5-step-4.md).  ctest runs the driver once per
# sampled path with the env pinned exactly as the dispatcher reads it - one
# process per path, because sampled_path() caches its env read per process.
add_library(k_sampler STATIC kernels/sampler.cpp)
add_executable(k_sampler_smoke t5/k_paths_smoke.cpp)
target_link_libraries(k_sampler_smoke PRIVATE k_sampler)
add_executable(k_sampler_parity drivers/sampler_parity.cpp)
target_link_libraries(k_sampler_parity PRIVATE k_sampler)
add_library(k_s_gemv STATIC kernels/s_gemv.cpp)
add_executable(k_s_gemv_parity drivers/s_gemv_parity.cpp)
target_link_libraries(k_s_gemv_parity PRIVATE k_s_gemv)

add_executable(t0_dev0        t0/dev0.cpp)
add_executable(t0_bw_probe    t0/bw_probe.cpp)
add_executable(t0_bw_spike    t0/bw_spike.cpp)

add_test(NAME smoke_parity COMMAND smoke_parity)
add_test(NAME k_dequant_s2_parity COMMAND k_dequant_s2_parity --selftest)
add_test(NAME t3_sg_probe   COMMAND t3_sg_probe)
add_test(NAME k_router_top10_parity COMMAND k_router_top10_parity --selftest)
add_test(NAME t4_fp16_probe   COMMAND t4_fp16_probe)
add_test(NAME k_s_gemv_parity COMMAND k_s_gemv_parity --selftest)
add_test(NAME k_s_gemv_bench  COMMAND k_s_gemv_parity --bench --iters 50)
add_test(NAME t5_atomic_probe   COMMAND t5_atomic_probe)
add_test(NAME t5_math_probe     COMMAND t5_math_probe)
add_test(NAME t5_tree_probe     COMMAND t5_tree_probe)
add_test(NAME t5_math_fixture_probe COMMAND t5_math_fixture_probe)
add_test(NAME k_sampler_smoke   COMMAND k_sampler_smoke)
add_test(NAME k_sampler_parity          COMMAND k_sampler_parity --selftest)
add_test(NAME k_sampler_parity_one_block COMMAND k_sampler_parity --selftest)
add_test(NAME k_sampler_parity_old      COMMAND k_sampler_parity --selftest)
add_test(NAME k_sampler_bench           COMMAND k_sampler_parity --bench)
set_tests_properties(k_sampler_parity PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 300)
set_tests_properties(k_sampler_parity_one_block PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1" TIMEOUT 300)
set_tests_properties(k_sampler_parity_old PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 300)
set_tests_properties(k_sampler_bench PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 300)
add_test(NAME t0_dev0      COMMAND t0_dev0)
add_test(NAME t0_bw_probe  COMMAND t0_bw_probe)
add_test(NAME t0_bw_gate   COMMAND t0_bw_spike)   # bw_spike exits 0 iff its gate passes

# The DPC++ driver only adds the SYCL runtime libraries when -fsycl is on the
# LINK line as well (compile-only -fsycl leaves libsycl undefined). Verified:
# CMAKE_CXX_LINK_OPTIONS does not reach the link line on CMake 4.2, per-target
# LINK_OPTIONS does.
foreach(t smoke_parity k_dequant_s2_parity t3_sg_probe k_router_top10_parity t4_fp16_probe k_s_gemv_parity
                    t5_atomic_probe t5_math_probe t5_tree_probe t5_math_fixture_probe k_sampler_smoke
                    k_sampler_parity t0_dev0 t0_bw_probe t0_bw_spike)
  target_link_options(${t} PRIVATE -fsycl ${_rpath_opts})
  # UMF must be a direct dependency: the Level Zero adapter dlopens it, and
  # dlopen from a shared library does NOT consult the executable's RPATH.
  # As a NEEDED entry the executable's own RPATH maps it at startup, and the
  # adapter's later dlopen then hits the already-loaded object.
  if(DEFINED ENV{ONEAPI_ROOT})
    target_link_libraries(${t} PRIVATE "$ENV{ONEAPI_ROOT}/umf/1.1/lib/libumf.so.1")
  endif()
endforeach()

set_tests_properties(smoke_parity k_dequant_s2_parity t3_sg_probe k_router_top10_parity t4_fp16_probe
                      k_s_gemv_parity t5_atomic_probe t5_math_probe t5_tree_probe
                      t0_dev0 t0_bw_probe t0_bw_gate PROPERTIES TIMEOUT 120)
set_tests_properties(k_s_gemv_bench t5_math_fixture_probe k_sampler_smoke PROPERTIES TIMEOUT 300)
