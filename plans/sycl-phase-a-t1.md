# T1 — detailed plan: the skeleton

Parent: `plans/sycl-phase-a.md` (Phase A plan, task T1). T0 is done (PASS, see
`plans/sycl-phase-a-report.md` §T0); T1 builds the empty pipeline that T2–T5
drive: standalone CMake build, the thin `sycl_compat` host layer, the
mirror/fixture convention, and one smoke driver proving the whole chain.

## Definition of done

From a clean state (`rm -rf poc/sycl/build`), `./poc/sycl/run.sh`:

1. Configures and builds with `icpx -fsycl` (CXX picked from the sourced
   oneAPI env; never a system g++).
2. Runs the ctest suite green:
   - `smoke_parity` — the first (only) driver: trivial GPU copy, compared
     against a CPU reference, bitwise exact.
   - `t0_bw_gate` — re-runs `t0/bw_spike` (exit 0 ⇒ gate still passes; the T0
     result stays a standing regression, not a one-off transcript).
3. The gate holds under `env -u LD_LIBRARY_PATH` — built binaries carry an
   rpath into `$ONEAPI_ROOT/compiler/2026.1/lib`, verified with `ldd`.
4. `poc/sycl/check_mirrors.sh` exits 0 (the fixture-drift checker, see below).
5. `git status` shows additions under `poc/sycl/` and `plans/` only — the main
   tree (`src/`, `CMakeLists.txt`, `include/`) is untouched, per the Phase A
   non-goals.
6. Review rule, not just a test: `include/sycl_compat/` contains no
   kernel-specific code — only device/queue/alloc/error/launch plumbing.

## Layout (final form after T1)

```
poc/sycl/
  CMakeLists.txt           standalone project; never touches the root build
  run.sh                   source setvars -> cmake -> ctest -> gate
  check_mirrors.sh         fixture-drift guard (see convention below)
  include/sycl_compat/
    test_ctx.hpp           device selection, context/queue, USM alloc, error check
    launch.hpp             the one submit convention for ported kernels
  kernels/                 (empty at T1; one .cpp per ported kernel from T2 on)
  drivers/
    smoke_parity.cpp       the pipeline proof driver
  bench/                   (empty; T6)
  t0/                      dev0.cpp, bw_spike.cpp, bw_probe.cpp (existing, T0)
  build/                   (git-ignored by convention; regenerated)
```

Two naming decisions made explicit here (they bind T2–T5):

- **Kernel files are `.cpp`, not `.sycl`.** T0 proved lambda-form SYCL kernels
  compile as plain C++ under `-fsycl`; no separate kernel file type is needed.
  The Phase A layout's `<name>.sycl` is superseded.
- **One kernel = one submit function.** Every ported kernel module exposes
  exactly one function of this shape; drivers never touch SYCL launch
  mechanics:

  ```cpp
  namespace strata::kernels {
  sycl::event submit_dequant_s2(sycl::queue& q, /* device pointers, shape, params */);
  }
  ```

  Driver pattern (fixed for all of T2–T5):

  ```cpp
  strata::sycl_compat::ctx c;
  auto* x = c.device_alloc<float>(N);
  /* ... */
  strata::kernels::submit_foo(c.q, /* ... */).wait();
  /* compare against the mirrored host reference, print, return 0/1 */
  ```

## Components

### 1. `CMakeLists.txt`

Spec (the code itself goes in the file; this is the contract it must meet):

- `cmake_minimum_required(VERSION 3.24)`, `project(strata_sycl_poc LANGUAGES CXX)`.
- C++20 required; compile and link options get `-fsycl` (T0's exact flag
  spelling — single dash). **Implemented reality:** the global
  `CMAKE_CXX_LINK_OPTIONS` never reached the link line on CMake 4.2
  (reproduced in a minimal project), so `-fsycl` goes on per-target
  `target_link_options(t PRIVATE -fsycl ...)`.
- **Compiler must be `icpx`** (T0 finding: `icx` is the C driver and warns
  that it won't link the C++/SYCL runtime). `run.sh` passes
  `-DCMAKE_CXX_COMPILER="$(command -v icpx)"` explicitly rather than trusting
  CMake's env autodetect; CMakeLists asserts the chosen compiler name
  contains `icx` and fails configure otherwise.
- **rpath** (the one real packaging gap T0 left): built binaries must
  resolve `libsycl`/UR without a sourced shell, verified by done criterion 3,
  not assumed. **Implemented reality:** `CMAKE_BUILD_RPATH` emits DT_RUNPATH,
  which dlopen *from the UR adapter* does not consult; the working form is
  traditional DT_RPATH via `-Wl,--disable-new-dtags -Wl,-rpath,<compiler>/../lib`
  plus a direct link to `$ONEAPI_ROOT/umf/1.1/lib/libumf.so.1` — the Level
  Zero adapter dlopens UMF, and without it device enumeration fails with
  "No device of requested type available" (full story: report §T1).
- Include path for the project: `poc/sycl` (so `#include
  "sycl_compat/test_ctx.hpp"` works; SYCL itself comes from `icpx`'s built-in
  paths — T0 needed no extra `-isystem`).
- Targets:
  - `t0_dev0`, `t0_bw_spike`, `t0_bw_probe` — the three existing T0 programs
    (single source each).
  - `smoke_parity` — `drivers/smoke_parity.cpp`.
  - Convention for T2–T5, documented in the CMakeLists header comment:
    kernel module `k_<name>` (STATIC library, one source
    `kernels/<name>.cpp`) + driver `k_<name>_parity` (executable, one source
    `drivers/<name>_parity.cpp`, links `k_<name>`). No other target kinds.
- `enable_testing()`; every driver registered via `add_test` with
  `TIMEOUT 120`; `t0_bw_gate` registered as the gate test (bw_spike exits 0
  iff its gate passes).
- The ctest set is intentionally ordered: smoke first, gate last — a red gate
  must not be hidden behind driver noise.

### 2. `include/sycl_compat/test_ctx.hpp`

Host-side mirror of the CUDA parity tests' idioms (`check()`, alloc/free,
queue). Content, reflecting every T0 API finding:

- `pick_device()`: device-selector **lambda returning `int`** (T0 finding),
  predicate = `is_gpu() && get_backend() == sycl::backend::ext_oneapi_level_zero
  && name contains <substr>`, where `<substr>` defaults to `"e223"` and can be
  overridden with `STRATA_SYCL_DEVICE_SUBSTR` (keeps the harness portable
  without a `--device` flag).
- `ctx` class: owns `sycl::device` (via the selector), `sycl::context`, and
  an **in-order queue built with `enable_profiling()`** (T0 finding — without
  it `get_profiling_info` throws). Members:
  - `template<class T> T* device_alloc(size_t)` / `host_alloc(size_t)` — USM
    via `sycl::malloc_device<T>(n, q)` / `sycl::malloc_host<T>(n, ctx)`
    (T0 finding — `malloc_host` needs the context);
  - `void free(T*)` — `sycl::free(p, q)` / `sycl::free(p, ctx)`;
  - `void wait_and_throw()`.
- `[[noreturn]] void fail(const char* what, const std::string& detail = "")` —
  prints and `exit(1)`, the shape of the CUDA tests' `check()`.
- **Nothing else.** No kernel math, no fixture code, no per-kernel types.

### 3. `include/sycl_compat/launch.hpp`

- `strata_launch(queue, items, work_group, kernel_lambda)` — wraps
  `q.parallel_for(sycl::nd_range<1>(items, work_group), ...)`; one place to
  keep the nd_range idiom.
- Documented in the header: kernels needing shared memory do **not** use
  this; they define a kernel struct with a `sycl::local_accessor` member and
  `parallel_for` it directly (that pattern first appears in T3's router port;
  it is deliberately not abstracted here).
- Constants for the testbed (from T0's `dev0`): max work-group 1024,
  sub-groups 16/32, 128 KiB shared — referenced by comments in later ports.

### 4. Fixture-mirror convention + `check_mirrors.sh`

The Phase A rule is that drivers **reuse the CUDA parity tests' fixtures and
host references, copy or factored — never re-derived**, since touching
`src/` is out of scope. The mechanism:

- Copied code sits between explicit markers:

  ```cpp
  // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:57:79 @ <short commit sha>
  int reference_pick(const std::vector<float>& l, const SamplerParams& p, bool temp_first) {
  /* ... verbatim copy ... */
  }
  // SYCL-MIRROR-END
  ```

- `check_mirrors.sh` scans `poc/sycl/**` for `SYCL-MIRROR-BEGIN
  <path>:<from>:<to> @ <sha>`; for each block it extracts lines
  `<from>..<to>` of the original file and diffs them against the mirrored
  lines (markers excluded). Any drift → non-zero exit with the offending
  block. At T1 there are zero mirrors except the smoke's trivial one; the
  script still must run and pass (it's the guard for T2–T5).
- Rationale recorded in the script header: a drifted fixture means the SYCL
  driver is testing something the CUDA test no longer tests — exactly the
  silent-weakness the Phase A plan forbids.

### 5. `drivers/smoke_parity.cpp`

The smallest driver that exercises the whole chain; spec:

- 64 KiB of floats, pattern-init on the host, USM device copy.
- Kernel: `y[i] = a[i]` (identity copy, one work-item per 4 floats — already
  the T0 access pattern, so it also proves the 128-bit-load shape).
- Reference: host `std::memcmp` — **bitwise exact**, no tolerance (a copy has
  no rounding; a tolerance would be hiding something).
- Prints `smoke_parity: PASS` and returns 0; any mismatch or SYCL failure →
  `fail()` and exit 1.
- Registered in ctest as `smoke_parity`, timeout 120 s.

### 6. `run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Only source when the env is absent: a subshell that inherited a sourced
# env makes setvars.sh refuse to re-run (exit 3) -- the implemented guard
# (the plan's original unconditional source failed in exactly that case).
[ -n "${ONEAPI_ROOT:-}" ] || source "$HOME/intel/oneapi/setvars.sh"
cd "$ROOT"
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
bash check_mirrors.sh
```

(The gate is inside ctest as `t0_bw_gate`; the final `check_mirrors.sh` step
keeps fixture drift visible even when ctest is skipped by hand.)

## Task order within T1

1. `CMakeLists.txt` + wire the three T0 programs as targets; `run.sh`; get a
   clean `./run.sh` green with zero parity drivers. *(the real first risk:
   compiler detection + rpath)*
2. `sycl_compat/{test_ctx,launch}.hpp` + `smoke_parity`; ctest entry; green.
3. `check_mirrors.sh` + one marker block in the smoke driver (its init pattern
   can mirror `bw_spike`'s, keeping the convention exercised from day one);
   green, then delete the marker block? — **no**: keep it; it is the permanent
   worked example of the convention.
4. Final pass: done criteria 1–6, `env -u LD_LIBRARY_PATH ./build/t0_bw_spike`,
   `git status` check.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CMake mis-detects `icpx` (clang-based DPC++ can trip ABI probes) | `run.sh` passes `-DCMAKE_CXX_COMPILER` explicitly; CMakeLists asserts the name. Last resort if CMake fights us: `run.sh` compiles directly with `icpx` (CMake is convenience, not the deliverable) — note that fallback in the CMakeLists header so it's a decision, not a flail |
| Binaries die outside the sourced shell (libsycl not found) | `CMAKE_BUILD_RPATH`; verified by done criterion 3 (`env -u LD_LIBRARY_PATH`) and `ldd` |
| `setvars.sh` env leaking into other builds | sourced only inside `run.sh`, scoped to that shell |
| `sycl_compat` accretes kernel-specific helpers | done criterion 6 (review rule) + the rule that per-kernel code lives in `kernels/` |
| Fixture copies silently drift from the CUDA tests | `check_mirrors.sh` in every `run.sh` invocation |
| Testbed-specific device id (`e223`) hardwired everywhere | one place: `test_ctx::pick_device()`, env-overridable |

## Effort

- Step 1 (build plumbing + T0 targets): ~2–4 h — the compiler/rpath risk lives here.
- Step 2 (shim + smoke): ~2 h.
- Step 3 (mirror guard): ~1 h.
- Step 4 (final pass + notes): ~1 h.
- **Total ≈ 1 day**, matching the Phase A table ("T0–T1: ~1 day").

## Handoff to T2

T2 (first real kernel port: `dequant_s2`) starts from a green `run.sh` and
adds exactly three things: `kernels/dequant_s2.cpp` (with `submit_dequant_s2`),
`drivers/dequant_s2_parity.cpp` (mirroring `src/kernels/dequant_s2_parity.cpp`'s
fixture + reference behind markers), and two CMake targets + one ctest entry.
If that addition takes longer than the plan's 1–2 days, the bottleneck is the
kernel, not the harness — which is the signal the skeleton did its job.
