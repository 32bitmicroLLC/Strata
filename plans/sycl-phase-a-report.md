# Phase A — execution report

## T0 — environment verification + sanity spike — **PASS** (2026-07-xx)

### T0.1 — environment verification

| # | Check | Result |
|---|---|---|
| 1 | `source ~/intel/oneapi/setvars.sh && icx --version` | `Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)` ✓ |
| 2 | `sycl-ls` GPU on Level Zero | `[level_zero:gpu][level_zero:0] ... over Level-Zero V2, Intel(R) Graphics [0xe223] 20.2.0 [1.14.37020]` ✓ |
| 3 | Driver/runtime coherence | kernel module `xe`; `libze_intel_gpu.so.1.14.37020` + `libze_loader.so.1.28.2`; `sycl-ls` reports the same `1.14.37020` ✓ (no mismatch to chase) |
| 4 | Toolchain targets the GPU | `dev0` runs, prints the device (transcript below) ✓ |
| 5 | SKU + rated bandwidth | PCI `8086:e223` "Battlemage G31"; SYCL `global_mem_size` = **32.5 GB** and `max_compute_units` = **256** → **Arc B770 class (32 GB)**, not the B580 (16 GB) I assumed when writing the plan. Official rated bandwidth could not be confirmed offline; my 512 GB/s assumption is for the B580 class. See anomaly note below — measured numbers imply the SKU is ≥ ~640 GB/s class, and the gate passes under any plausible rating. |
| 6 | GPU health | `intel_gpu_top` cannot enumerate this XE card in this environment (`-L` lists it as "Intel Battlemage (Gen20)" but no filter accepts it). Substituted: three consecutive benchmark runs are repeatable to ±1% — no throttling signature. |
| 7 | Isolation | No other GPU workload observed on the Arc (NVIDIA GM206 also present but unused). |

`dev0` transcript (compile: `icpx -fsycl -O2 -std=c++20 dev0.cpp -o dev0`):

```
name            = Intel(R) Graphics [0xe223]
vendor          = Intel(R) Corporation
driver version  = 1.14.37020
backend version = 1.14.37020
max_compute_units       = 256
max_work_group_size     = 1024
max_num_sub_groups      = 64
sub_group_sizes         = 16 32
preferred_vec_width_f32 = 1
local_mem_size (per WG) = 131072 bytes
global_mem_size         = 32530182144 bytes
max_clock_frequency     = 2800 MHz
```

**Numbers Phase B needs (recorded, per plan):**

- Subgroup width: **16** (max 32) — confirms the plan's assumption that Battlemage is 16-wide; the 32-lane warp assumption in Strata kernels must be handled per the T3 mapping decision.
- Shared memory per work-group: **128 KiB** — comfortably above the kernels' 64 KiB requirement; the 64 KiB LDS assumption does not bind.
- Max work-group size 1024; 256 EU; core clock up to 2.8 GHz.
- `preferred_vector_width_float = 1` — do not rely on the compiler to vectorize; use explicit 128-bit loads (the spike does).

### T0.2 — bandwidth spike

`poc/sycl/t0/bw_spike.cpp` (256 MiB arrays, USM device, event-profiling
timing, 1 warmup + 5 timed runs per benchmark, median reported).

```
device = Intel(R) Graphics [0xe223] (ext_oneapi_level_zero)
arrays = 3 x 256 MiB USM device, pattern-init
gate   = triad >= 307.2 GB/s (60% of 512 GB/s rated)

H2D   = 13.8 GB/s   (runs: 13.8 13.8 13.8 13.8 13.8)
D2H   = 14.3 GB/s   (runs: 14.3 14.3 14.2 14.3 14.3)
D2D   = 1580.5 GB/s (runs: 1581.5 1579.5 1580.5 1577.6 1582.4)
triad = 935.5 GB/s  (work-group 64)
triad = 931.2 GB/s  (work-group 256)

GATE: triad best = 935.5 GB/s vs 307.2 GB/s threshold -> PASS
```

**Anomaly found and resolved** — the effective triad (935 GB/s) exceeded the
plan's assumed 512 GB/s rated figure, which is physically impossible if the
timing were wrong. Cross-checked with independent host wall-clock timing on
1 GiB arrays (`poc/sycl/t0/bw_probe.cpp`):

```
D2D  1GiB = 1144.6 GB/s   (wall 1.876 ms)
triad 1GiB=  935.8 GB/s   (wall 3.442 ms)
read  1GiB =  590.5 GB/s   (wall 1.818 ms)
```

Both methods agree to < 0.1%. The timing is trustworthy; the **512 GB/s
assumption was simply wrong for this SKU** (B770 class, 32 GB). Single-stream
read at 590 GB/s is consistent with a ~640 GB/s class part. Gate verdict is
robust: PASS at 935.5 vs 307.2 (512 assumed), and still PASS at 0.6 × 640 =
384 even if the true rating is higher.

**H2D/D2H at ~14 GB/s** ≈ PCIe Gen3 x16 effective — this testbed's link is
older-generation (sysfs even reports a stale 2.5 GT/s x1 status; the box is a
KVM host with passthrough). For Strata's CPU-expert ↔ GPU design this matters:
on this machine the PCIe tier carries only ~14 GB/s, which shifts the
optimal `--pcie-frac` behaviour. Desktop Gen4/5 links (25–50 GB/s) are the
real deployment target; note for Phase D calibration.

### Toolchain gotchas (for T1 and later)

- The DPC++ driver for C++ is **`icpx -fsycl`** (single dash), not
  `icx --fsycl` — `icx` warns it is the C compiler and does not link the
  SYCL runtime; the flag is `-fsycl`, not `--fsycl`.
- Include is **`<sycl/sycl.hpp>`** (the umbrella `sycl.hpp` is not on the
  default search path of `icpx`).
- SYCL 2020 API deltas vs. common examples: device selector lambdas must
  return `int`; the backend is `enum class sycl::backend` with
  `ext_oneapi_level_zero` (no `.is<>()`, no `.string()`); `sycl::malloc_host`
  requires a `sycl::context`; event profiling uses
  `e.get_profiling_info<sycl::info::event_profiling::command_start|command_end>()`
  (ns, uint64) and the queue must be built with
  `sycl::property::queue::enable_profiling()`.
- Device info of note: no bus-width/mem-clock params (OpenCL-era names do not
  exist); subgroup width comes from `info::device::sub_group_sizes` (Intel) or
  `info::kernel_device_specific::max_sub_group_size`.

### T0 verdict

**PASS.** GATE line: triad best = 935.5 GB/s ≥ 307.2 GB/s threshold.

What Phase B may assume:

- The Level Zero runtime path works end-to-end (alloc, copy, kernel, events)
  and the card sustains well beyond the gate under streaming loads.
- 16-wide subgroups, 128 KiB shared per work-group, work-groups up to 1024.
- `poc/sycl/t0/{dev0,bw_spike,bw_probe}.cpp` are committed and re-runnable —
  T1's `run.sh` should include them as the standing regression.

Next: T1 (`poc/sycl/CMakeLists.txt`, `sycl_compat` host shim, `run.sh`).

## T1 — skeleton — **PASS**

`run.sh` from a clean `build/` is green: 4/4 ctest (`smoke_parity`,
`t0_dev0`, `t0_bw_probe`, `t0_bw_gate`), `check_mirrors.sh` OK. Binaries run
with `env -u LD_LIBRARY_PATH` (smoke PASS, gate 934.7 GB/s). `git status`
shows additions under `poc/sycl/` and `plans/` only.

Deviations from the T1 plan, all forced by reality, all now documented in
`poc/sycl/CMakeLists.txt` comments:

- **`setvars.sh` guard in run.sh.** Sourcing it in a subshell that already
  inherited a sourced env makes it print usage and exit 3 ("setvars.sh has
  already been run"). run.sh now sources only when `ONEAPI_ROOT` is unset.
- **Link flags: CMake's `CMAKE_CXX_LINK_OPTIONS` silently drops.** The DPC++
  driver adds the SYCL runtime libraries only when `-fsycl` is on the *link*
  line. `CMAKE_CXX_LINK_OPTIONS` never reached the link line (reproduced in
  a minimal CMake 4.2 project); per-target `target_link_options(t PRIVATE
  -fsycl)` does.
- **UMF must be a direct NEEDED entry, not just rpath.** Without
  `$ONEAPI_ROOT/umf/1.1/lib/libumf.so.1` reachable, device enumeration fails
  with "No device of requested type available" even though every library
  loads (LD_DEBUG: all UR adapters resolve; the failing load is UMF, pulled
  in by the Level Zero adapter). Bisected the sourced `LD_LIBRARY_PATH` to
  the single `umf/1.1/lib` entry. Executable DT_RPATH is *not* consulted for
  dlopen from a shared library (and CMake emits DT_RUNPATH anyway), so each
  target links `libumf.so.1` directly by path; the executable's own rpath
  then maps it at startup and the adapter's dlopen hits the loaded object.
  `-Wl,--disable-new-dtags` forces traditional DT_RPATH for the compiler lib
  dir.
- **`sycl::queue::wait_and_throw` is not const** — `ctx::wait_and_throw()`
  dropped its `const`.
- **Mirror-marker format** as planned (`<path>:<from>:<to> @ <sha>`); the
  checker had two bugs of its own (sha-strip pattern `%% *@` never matched —
  the `@` is not the last character; `sed 1d;$d` stripped content lines that
  were never marker lines). Fixed; negative test: flipping `1e-10f` to
  `1e-9f` in the mirrored line fails the checker with a one-line diff,
  restoring passes.

## T2 — `dequant_s2` port — **PASS**

First real math through the pipeline. Kernel + parity driver per
`plans/sycl-phase-a-t2.md`; sources `poc/sycl/kernels/dequant_s2.cpp` and
`poc/sycl/drivers/dequant_s2_parity.cpp` (mirrored from
`src/kernels/cuda/dequant_s2.cu` / `src/kernels/dequant_s2_parity.cpp` /
`include/strata/artifact/dequant.hpp` @ 6bd3e58).

### Result

- ctest `k_dequant_s2_parity` (200 000 blocks, `--selftest`):
  `dequant_s2: 200000 blocks, 12800000 elements, 0 mismatched` — **bit-exact**
  against the CPU reference chain (scalar `dequantize_q2_0` proved equal to
  ggml by `bench/micro/dequant_xcheck`). Non-vacuity line prints
  `4 of 4 codes present`.
- Edge runs green under `env -u LD_LIBRARY_PATH`: `--blocks 1` (single
  work-item) and `--blocks 12345` (partial final work-group).
- **Mutation test (non-vacuity of the harness):** temporarily breaking the
  bit offset (`* 2` → `* 2 + 1`) makes the driver fail with
  `first mismatch at element 0 (block 0, offset 0): cpu -0 gpu -0.5`;
  restoring it goes green. The driver demonstrably catches a wrong kernel.
- `check_mirrors.sh`: all 10 new blocks OK (plus the T1 smoke block), 11 total.

### Findings

1. **The Intel Level Zero backend rejects non-uniform work-groups.**
   200 000 items / 256 = 781.25 work-groups →
   `sycl::exception: Non-uniform work-groups are not supported by the target
   device` at submit. CUDA runs the partial last block fine, so this did not
   exist in the ported code. The T1 smoke test happened to use an exact
   multiple (16 384 / 256), which is why it did not surface earlier. Fix,
   centralised in `sycl_compat::strata_launch`: round the item count **up** to
   a multiple of the work-group; padded work-items fall through every
   kernel's bounds check. Consequence: **every PoC kernel must bounds-check
   its global id against the logical item count** — now stated in
   `launch.hpp`. (T3's shared-memory kernels must keep this too: the padded
   work-groups still participate in the launch.)
2. **The mirror discipline caught a real drift on first use.** The checker
   flagged block `dequant_s2_parity.cpp:40-51`: line 51 of the original is a
   blank line the first copy omitted. One-line diff, fixed, green. This is the
   mechanism working as designed — the "fixture drift" risk row in the parent
   plan is now enforced, not merely reviewed.
3. **Bit-exactness held, as predicted.** No rounding is possible in
   `(code - 1) * d` (scaling by ±1/2), so the `memcmp` comparison passes
   without any tolerance machinery — confirmed, not assumed. The mutation
   test's `cpu -0 gpu -0.5` output also shows the comparison is bitwise at the
   sign-bit level (the element-0 CPU value is −0).

### Plan deviations (implemented reality)

- Mirror line ranges in the plan were off by one in two places: the kernel
  constants are `dequant_s2.cu:21:22` (not 22:23), the kernel body
  `27:34` (not 27:35 — line 35 is the function's closing brace, which the
  SYCL lambda supplies itself), and the driver's `QK` constant is
  `dequant_s2_parity.cpp:22` (not line 20, which is `namespace {`).
- 10 new mirror blocks, not 9: the plan's table undercounted (three
  `dequant.hpp` blocks + five parity-test blocks + two kernel-file blocks).
- Everything else matched the plan: CMake targets, ctest registration
  position (after smoke, before the t0 regressions), 120 s timeout, ~half-day
  effort (well inside the 1–2 day envelope; the dominant time was the one
  real bug, finding 1).

### Standing regressions (ctest)

| Test | What it guards |
|---|---|
| `smoke_parity` | build → USM → kernel → compare chain |
| `k_dequant_s2_parity` | the S2 decode, bit-exact vs the ggml-proven CPU chain |
| `t0_dev0` / `t0_bw_probe` / `t0_bw_gate` | environment + bandwidth gate |

## T3 — `router_top10` port — **PASS**

### Result

7/7 ctest. The parity driver (`k_router_top10_parity`, 64 tokens × 512
experts, k=10, seed 2024):

```
  random normal              ids exact (0 bad)   weights worst rel 1.378e-07 (0 over tol)   |sum-1| 2.2e-08
  all equal (ties)           ids exact (0 bad)   weights worst rel 0.000e+00 (0 over tol)   |sum-1| 1.5e-08
  12-way exact tie           ids exact (0 bad)   weights worst rel 1.347e-07 (0 over tol)   |sum-1| 2.2e-08
  dominant expert            ids exact (0 bad)   weights worst rel 5.960e-08 (0 over tol)   |sum-1| 1.9e-08
  clamp reachability         top-10 sum >= k/n = 0.01953, clamp is 6.104e-05 -> margin 320x  (CLAMP IS UNREACHABLE, asserted)
```

- **IDs exact on every distribution** — the tie-break survived the port; the
  all-equal case's weights are **bit-identical** to the glibc reference
  (worst rel 0.000e+00), so the plan's Risk 1 (double `exp` on software-FP64
  Arc) materialised as a non-issue for this geometry.
- **Mutation test:** inverting the pair tie-break (`oi < bi` → `oi > bi`)
  made the all-equal case report `ids *** WRONG *** (640 bad)` and the driver
  exit 1, exactly as the plan required; restoring went green. The 12-way and
  dominant cases also failed (576/37 bad) — the harness is not vacuous on the
  near-tie paths either.
- Probe (`t3_sg_probe`): checks 2 and 3 pass; check 1 prints the sub-group
  width (512 — the runtime reports the full work-group, so T0's {16, 32}
  question is moot for this design).
- 38/38 mirror blocks OK under `check_mirrors.sh`; `env -u LD_LIBRARY_PATH`
  runs both binaries clean.

### Findings

1. **The plan's Option A/B premise is unbuildable on icpx 2026.1.**
   Verified by compile, not assumed:
   - there is **no subgroup shuffle or reduce API**: no `sycl::shfl*`,
     `sub_group::reduce`/`shfl_down` do not exist (with or without a mask),
     and no `sycl::property` requests a sub-group width (`-sycl-std` accepts
     only `2020`, so there is no SYCL-2023 mode either). The SYCL-2020-spec
     subgroup algorithms simply are not in this toolchain;
   - the standard **kernel-struct-with-`local_accessor`-members pattern does
     not compile**: `local_accessor` has no host-side constructor from a
     range (only `(range, handler&)`).
2. **The working pattern (now the PoC standard for shared-memory kernels):**
   construct `sycl::local_accessor`s inside the `q.submit` handler and capture
   them **by value** into the kernel lambda; replace every CUDA warp-shuffle
   tree + shared 2nd stage with a **barrier-based tree over local memory**.
   Bit-exactness argument: `fmaxf` is exact and order-independent, and the
   `(value, index)` pair compare is a total order, so ANY reduction tree
   returns the same winner as CUDA's two-stage shuffle reduction — the
   shuffle was an implementation detail, not a semantic one. Cost: ~104
   `it.barrier()`s per work-group (9 per tree × (1 max + 10 selection) +
   stage barriers) — a Phase B performance item, not a parity one.
3. **`nd_range` mapping trap.** `nd_range(n_tokens, threads)` means
   *n_tokens items in groups of threads* — non-uniform, rejected by Level
   Zero. One-group-per-token needs global range `n_tokens * threads` and
   `token = global_id / group_size`. The T2 non-uniformity finding was a
   device constraint; this one is a mapping bug — same crash, different cause.
4. **New host checks (documented deviations):** the local-memory trees need a
   **power-of-two work-group** (CUDA's warp shuffles tolerated any multiple
   of 32) → explicit throw; and the 120 KiB Arc local-memory cap (CUDA's cap
   is 32 768 experts ≈ 400 KiB smem). Engine geometry (n_expert = 512 →
   512 threads, ~11 KiB) sits comfortably inside both.
5. The `sub_group` object that *does* exist (`it.get_sub_group()`) offers only
   `barrier` (deprecated), `leader()`, and id/range accessors.

### Plan deviations (implemented reality)

- **Design:** Option C (handler accessors + local-memory trees) replaces the
  plan's Option A/B — see Findings 1–2. Everything the probe was meant to
  decide (shuffle semantics, subgroup width) became moot; the probe now
  validates the actual pattern instead (checks 2–3) and keeps the width print
  as information.
- **Mirror blocks:** 14 in the kernel (plan said ~7 — finer grain after seeing
  which lines survive verbatim), 13 in the driver (plan said 11 — the
  `run_case` return line and the reference comment line joined in). 38 total
  in the tree.
- **Line ranges:** the sum block is `router_top10.cu:109:111` (the plan's
  guess 108:110 starts on the stage comment); the threads block keeps both
  lines 203–204; two drifts caught by the checker on first run and fixed.
- **Effort:** well under the 2–3 day estimate; the dominant time was Finding 1
  (toolchain API discovery), which the plan's probe step was designed to
  absorb.

### Standing regressions (ctest)

| Test | What it guards |
|---|---|
| `smoke_parity` | build → USM → kernel → compare chain |
| `k_dequant_s2_parity` | the S2 decode, bit-exact vs the ggml-proven CPU chain |
| `t3_sg_probe` | handler local accessors + local-memory tree pattern |
| `k_router_top10_parity` | the MoE router: ids exact, weights ≤ 1e-5, clamp asserted unreachable |
| `t0_dev0` / `t0_bw_probe` / `t0_bw_gate` | environment + bandwidth gate |

---

## T4 — `s_gemv` port + driver — **PASS**

### Parity result (`k_s_gemv_parity --selftest`, Arc Battlemage G31)

| Case | rows | over tol (1e-4) | worst rel | notes |
|---|---|---|---|---|
| S2/Q2_0 | 128 | 0 | 0.000e+00 | bit-exact (bias -1, fp16 scales) |
| S4/Q4_0 | 128 | 0 | 6.634e-07 | |
| S4/IQ4_NL | 128 | 0 | 9.278e-06 | table codebook, fp16 scales |
| S8/Q8_0 | 128 | 0 | 9.649e-06 | |
| S4/Q4_K | 128 | 0 | 2.814e-07 rel-to-terms | rel-to-result 1.350e-06; cond up to 2.775e+07 |

**No form deferred** — the naive kernel's single template over {2,4,8}-bit widths
plus the runtime `has_offset` flag and the 16-entry `kIq4Nl` local-memory table
covers all four fixture forms and the Q4_K case; nothing in scope needed
machinery the S2 path lacked. The Q4_K worst error matches the CUDA test's own
recorded figure (2.8e-07 rel-to-terms) essentially exactly.

The mirrored self-consistency diagnostic prints
`CPU planes vs CPU raw block: max |diff| = 4.883e-04` (the two CPU reference
decode paths differ by a few fp32 ulps). It is **host-only** code, verbatim-
mirrored, cannot depend on the SYCL port, and does not gate the verdict — the
CUDA test prints the same thing. Recorded here so nobody reads it as a port
defect.

### Bench (naive kernel only — Phase B variants deferred)

```
s2_gemv bench  [2560 x 640]  200 iters
  ms per gemv: min 0.3592  median 0.3614  mean 0.4167
  min -> 4.6 G weights/s, 1.3 GiB/s of S2 streamed
s2_gemv bench  [640 x 2560]  200 iters
  ms per gemv: min 0.0969  median 0.0981  mean 0.0983
  min -> 16.9 G weights/s, 4.9 GiB/s of S2 streamed
```

First Arc numbers for the engine's GEMV path. The one-thread-per-row naive
kernel streams 1.3–4.9 GiB/s of S2 — well under a percent of the measured
D2D read bandwidth (590 GB/s, T0), as expected: `[2560 x 640]` gives only
5 work-groups of 128 threads (640 rows / 128), and each thread accumulates
its 2560 elements serially. This is exactly the gap the Phase B
`s_gemv_split` / quads / fast variants exist to close (the CUDA test's own
sweep targets ~184 GB/s-class throughput on the split kernel); the naive
kernel is a parity instrument here, not a performance claim. No CUDA
comparison number exists in this checkout (no LEDGER.md, no
docs/benchmarks/) — these are the baseline Arc figures Phase B will
optimise against.

### Mutation tests (both caught, both by design)

| Mutation | Expected | Observed |
|---|---|---|
| drop `+ b` (the weight offset) | Q4_K fails, 4 no-offset forms green | Q4_K 128/128 over tol (1.100 rel-to-terms), other 4 green, exit 1 |
| invert bias sign in `decode_tbl` | S2/S4/Q4_0/S8 fail, IQ4_NL green | 384 rows over tol in the 3 bias forms, IQ4_NL green, Q4_K green (its offset comes from the `off` array, bias = 0), exit 1 |

Restored → full suite green again.

### Findings

1. **icpx 2026.1 has no `sycl::fp16`.** The draft-stage name is absent; the
   SYCL 2020 16-bit IEEE binary16 type is `sycl::half` in this toolchain.
   `t4_fp16_probe` validates `sycl::half → float` widening against the host
   `strata::fp16_to_fp32` on the driver's 15 kScales patterns plus 9 edge
   patterns (subnormals, max, zero): **24/24 bit-exact**. The kernel uses
   `sycl::half`.
2. **`sycl::local_accessor` has no `.data()`** in DPC++ 2026.1 (no member
   found; `operator[]` yes). `&s_iq4nl[0]` is the working pointer idiom.
3. **The `s_iq4nl` table stays on the barrier-before-return pattern** (T2/T3
   finding): padded work-items past `n_out` fill the table and reach the
   barrier; an early return before it would strand peers.

### Plan deviations (implemented reality)

- **Block count:** 10 kernel blocks + 33 driver + 1 probe = 44 new blocks;
  82 total in the tree, all checker-green.
- **Bench sweep:** the CUDA test's split/quads/fast sweep (lines 356–464) is
  Phase B — not mirrored; the timing harness, fixed data, and print/projection
  lines are verbatim.
- **`test_q4k` takes `ctx&`** (SYCL queues are explicit; the CUDA test's
  implicit context drops) and the vacuous-`exit(1)` is kept (it is a test,
  not a library).
- **Effort:** ~1 day against the 1–2 week envelope — the T2/T3 pipeline
  (mirror discipline, checker, launch padding, ctx) was already built.

### Standing regressions (ctest)

| Test | What it guards |
|---|---|
| `t4_fp16_probe` | `sycl::half → float` bit-exact vs the host fp16 widening |
| `k_s_gemv_parity` | all 4 S-forms + Q4_K at 1e-4 (incl. the sum|term|-conditioned metric) |
| `k_s_gemv_bench` | smoke-times the two real expert shapes at 50 iters |
| (all T0–T3 entries above still run) | |

## T5 — `sampler` port — **done**

Plan: `plans/sycl-phase-a-t5.md`; step plan: `plans/sycl-phase-a-t5-step-1.md`.

### T5 step 1 — toolchain probes (`poc/sycl/t5/`)

**P1 — `atomic_ref` on local memory — PASS.**
`sycl::atomic_ref<unsigned, relaxed, memory_scope_work_group>::fetch_or` over
a local accessor, 1024 threads: 0 of 256 words differ from the serial OR.
**Consequence:** the penalty bitmap keeps the CUDA parallel-build structure;
no thread-0 serial fallback. (DPC++ 2026.1 requires the explicit
three-template-argument spelling — see the probe.)

**P2 — device double math on a structured grid — documented finding.**
Device `exp(double)` vs host `std::exp(double)`: 111,570 of 320,894 grid
pairs differ by 1 ulp. Device `logf` vs host `(float) std::log`: 6,843 of
100,000 grid values differ by 1 ulp. Bit-exactness on arbitrary arguments
does not hold; the grid is not the decision basis (see P2b).

**P2b — fixture-argument probe (`t5_math_fixture_probe`) — GATE PASS.**
Rebuilds every sampled-chain fixture of `src/kernels/sampler_parity.cpp`
data-for-data (verbatim, drift-checked), runs each row through the host
transcription of the kernel tail, and measures the exact tail arguments and
cut margins: 10,670 rows (1,340 void — fixture 16's NaN/±inf rows),
130,589 double-exp arguments. Device `exp` differs from host on 14,443 of
the 130,589 (1 ulp); device `logf` is **bit-identical on all four fixture
min_p values** {0.05, 0.3, 0.5, 0.9}. Minimum cut margins across all
fixtures: top_p 214,748,367 double ulps, draw 180,290,208 double ulps,
min_p 8,950 fp32 ulps — ~6 orders of magnitude beyond the gate thresholds
(1024 double / 8 fp32 ulps), which bound the worst perturbation a 1-ulp
exp term or thresh shift can cause.
**Consequence (fallback (a) of the parent plan):** the ported tail keeps the
plain `exp` / `logf` and the mirrored host references keep `std::exp` /
`std::log`; the affected set above is the documented gap. No hand-rolled
double `exp` (1.3.2), no fp32 `log` rework (1.3.3), no knife-edge finding
(1.3.4).

**P3 — 1024-wide local-memory reduction tree — PASS.**
1024 threads, 10 halving barriers, `(value, index)` total order: 0 of 64
trials differ from the serial host scan, including deliberate ties, NaNs,
and −inf sentinels. **Consequence:** the §4.2 block-argmax replacement is
approved at Arc's max work-group.

All four probes are ctest entries; `ctest -R t5_` is 4/4 green under
`env -u LD_LIBRARY_PATH` (~2 s total).

### Carry-over into step 2

- `poc/sycl/kernels/sampler.cpp` (1,020 lines) already exists in the tree
  from an earlier pass. It is **not** wired into `poc/sycl/CMakeLists.txt`
  yet, and its mirror blocks are currently **red** in `check_mirrors.sh`
  (indentation drift against the CUDA source; one misspelled `sampler.cuda`
  marker). Step 2 must make those blocks green before `k_sampler` /
  `k_sampler_parity` join ctest.
- Tail design inputs are fixed: local-memory tree per P3, `atomic_ref`
  bitmap per P1, plain `exp`/`logf` per P2b.

### T5 step 2 — repair, wire, compile, smoke (`poc/sycl/kernels/sampler.cpp`)

**2.1 Mirror repair — done.** All 75 blocks re-flowed to the source's
original indentation (router_top10 pattern), the `sampler.cuda` marker
fixed, all 76 SHA annotations updated `@ 6cabad2` → `@ bb7e783`.
`check_mirrors.sh` exits 0: 184 OK blocks tree-wide (75 in
`kernels/sampler.cpp`, 3 in `t5/k_paths_smoke.cpp` — the smoke's mirrored
penalty/philox blocks are drift-checked too).

**2.2 Header documentation — done.** The P2 bullet now states the P2b gate
numbers (14,443/130,589 one-ulp exp args; cut margins ≥ 2.1e8 double ulps /
≥ 1.8e8 double ulps / ≥ 8,950 fp32 ulps; fallback (a) adopted) instead of
the premature "verified by the fixtures running".

**2.3 CMake — done.** `k_sampler` is a `STATIC` library
(`kernels/sampler.cpp`), the three `k_sampler_parity*` `add_test` entries
moved to Step 4, `k_sampler_smoke` added (TIMEOUT 120, in the
`-fsycl`/rpath/UMF foreach).

**2.4 Compile green — done.** Whole TU compiles under `icpx -fsycl`
(exit 0). Every fix was in glue, never in a mirrored line:

- launch ranges: the draft's undefined `strata_launch_items(...)` replaced
  by plain `(size_t) n_tokens * N` item counts in all three per-row
  kernels; `n_tokens` parameter added to `submit_sample_greedy` /
  `submit_sample_one_block`;
- `submit_split_part`: the merge sink's undefined `dst_row` replaced by the
  direct `cand[(size_t) t * n_blocks * k + b * k + r]` write;
  `warp_first`/`warp_merge_lists`/`fold_block` now take raw pointers
  (`.get_pointer()`) so the four 128-thread regions of the split kernel use
  disjoint 32-slot butterfly regions;
- mirror `526:538` extended to `526:539` (the mirrored range had dropped
  CUDA's `for (m…)` loop-closing brace); one redundant glue brace removed
  in `submit_split_part`, one in the dispatcher;
- mirror `412:422` split into `412:421` + `423:423` with a glue comment:
  CUDA's `sampled_tail_warp` is a template parameter there and the SYCL
  port is not, so that one line is glue-excluded by documentation;
- three unguarded local-memory broadcast writes in `submit_split_merge`
  (`s_sum[0]`, `s_cut[0]`) given `if (lane == 0)` guards (data-race fix,
  glue);
- dispatcher: `sycl::event` hoisted out of the first if-branch; the empty
  return became an empty `q.submit`.

Known and accepted: 14 `get_pointer()` deprecation warnings (DPC++
2026.1); the accessor-based spellings that replace them (`get_multi_ptr`)
would fight the raw-pointer butterfly design, so the warnings stand.

**The `warp_first` butterfly bug — the step's substantive finding.**
The shared-memory butterfly transcribed from CUDA's `__shfl_xor` chain
wrote each lane's `(value, id)` slot once, at entry, then read the
partner's slot every stage:

```
s_v[lane] = bv;  s_i[lane] = bi;   it.barrier();
for (int off = 16; off > 0; off >>= 1) {
    const float ov = s_v[lane ^ off];  const int oi = s_i[lane ^ off];
    take_first(bv, bi, ov, oi);
    it.barrier();                       // <- missing: no write-back
}
```

A shuffle butterfly carries the running max in registers, so each stage's
partner read sees the previous stage's result; the shared-memory form must
write `s_v[lane] = bv; s_i[lane] = bi;` before each stage's barrier or the
butterfly degenerates to a per-lane *quadrant* max. The per-lane scans
were correct — the corruption was invisible in any single-lane dump and
only showed up as the split path's merged top-k list containing
duplicated, non-maximum entries (e.g. row 0 of the smoke: `[385, 136, 136,
356, 356, 160, 240, 33]` where the true top-8 is
`[511, 364, 239, 52, 385, 344, 133, 73]`). Detection path: the 2.5 smoke
failed (split 61/64 rows off, old/one_block exact against a host serial
reference) → per-path standalone runs showed the split family alone was
wrong → a scratch-dump of `cand` plus a host replica of stage 1 proved the
corruption was *in* stage 1 → a pre-butterfly per-lane dump in a debug
build showed correct per-lane chain winners, localising the bug to the
butterfly → a minimal standalone kernel (128 threads, per-thread `s[32]`,
2 rounds) reproduced it deterministically. Fix: the two glue write-back
lines. Note for Step 3 review: `fold_block` (greedy/old/one_block) is the
*tree* fold with correct per-level write-back — that is why those three
paths passed while split failed.

**2.5 Smoke — green.** `poc/sycl/t5/k_paths_smoke.cpp`: 64 rows × 512
vocab (64 == `kSplitMaxRows`, so the default split path is exercised),
16-token history per row with heavy repetition (penalties on: repeat
1.25, freq 0.25, present 0.5, `penalty_last_n = 16`), `top_k = 8`,
`top_p = 0.9`, `temperature = 0.8`, seed 42. The parent runs split +
greedy in-process; old and one_block run in forked children with fresh
SYCL runtimes (their env read is cached function-static in
`sampled_path()`, so one process can take only one sampled path).
Assertions: every pick is a valid index; greedy equals the host serial
argmax after penalties (**0/64 rows differ**); each of the three sampled
paths equals a host serial reference — k rounds of the penalised
strict-argmax (the kernels' (value, lower-index) total order) plus the
mirrored tail (278:315) with host math — (**0/64 rows differ on all
three**). `ctest -R "t5_|k_sampler_smoke"` green; full suite **15/15**
under `env -u LD_LIBRARY_PATH` (~7 s). The smoke doubles as the
barrier-deadlock detector the steps file asked for.

Environmental note: mid-step the Arc GPU dropped out of the driver once
(`UR_RESULT_ERROR_DEVICE_LOST`, `card1` vanished from `/sys/class/drm`
for a few minutes, no root available to reset it); it recovered on its
own and all subsequent runs were clean. Recorded because a future step
hitting the same mid-build is not a kernel fault.

### T5 step 3 — one-block + split stages + dispatcher verification

All of the Step-3 code had already been written and compiled in Step 2
(the draft was complete), so this step verified it against the shapes
the Step 2 smoke never exercised, fixed two glue gaps found in audit,
and closed the local-memory budget check. No mirrored line changed;
`check_mirrors.sh` stays exit 0.

**Dispatcher glue fixes (3.4).**
(a) `sycl::malloc_device` *throws* on failure (never returns null), so the
draft's `scratch == nullptr` → one-block fallback was unreachable for a
failed allocation: the exception would have propagated out of
`submit_sample_tokens`. The alloc is now wrapped in try/catch; a failure
leaves `scratch == nullptr` and takes the same one-block fallback CUDA's
failed `cudaMalloc` takes (verified by inspection — the failure cannot be
forced on this machine).
(b) Bad-argument behaviour was inconsistent: missing history with
penalties threw, but `n_tokens <= 0 || n_vocab <= 0` returned an empty
event while the header says bad arguments throw. Both cases now throw
`std::runtime_error`, matching the header; scenario E asserts both.

**Launch-range ratification (3.2).** All five submitters use exact
`nd_range<1>(n_tokens * …, G)` item counts — always a work-group
multiple — so padded items cannot exist and the CUDA `t >= n_tokens`
guards are safely omitted. The header's "1-D strata_launch-shaped
launch" wording was corrected to say so (doc-only).

**Local-memory budget (3.5).** Computed from the shipped accessors:

| kernel | work-group | worst-shape local memory (vocab 248,320) |
|---|---|---|
| greedy | 1024 | 31,040 (7,760-word bitmap) + 8,192 (rv/ri) = 39,232 B |
| old | 1024 | + 512 (sel_ids/sel_logit) = **39,744 B** |
| one_block | 1024 | **39,744 B** |
| split_part | 128 | 512 + 2,048 + 1,024 = 3,584 B (vocab-independent) |
| split_merge | 32 | 32,768 (lists) + 1,296 = 34,060 B (vocab-independent) |

Max consumer 39,744 B ≈ 38.8 KiB (old/one_block at vocab 248,320) — 31 %
of Arc's 128 KiB work-group cap. The steps file's "~34 KiB" was the
split_merge consumer; the true max is 39,744 B. **The parent plan's
deferral trigger does not fire; no local-memory opt-in machinery is
needed.** Proven at runtime, not just arithmetic: scenario F launches
the 39-KiB shapes on Arc at vocab 248,320 (a reject would throw at
enqueue) and its picks match the serial reference.

**Scenario matrix (3.6).** `k_paths_smoke` refactored into six scenarios;
every path is judged against the host serial reference (k rounds of the
penalised strict argmax + mirrored tail, host math) — same arbiter as
Step 2, now parameterised over vocab/history. The parent runs each
scenario's default path (plus greedy for A, B, F); the forced old /
one-block paths run in two forked children (old: A, B; one_block: A, C,
D, F) with fresh runtimes, as in Step 2.

| scenario | shape | what it proved | result |
|---|---|---|---|
| A | 64 × 512 | all four paths at the `kSplitMaxRows` boundary (Step 2 shape, unchanged data) | 0/64 differ on every path |
| B | 16 × 12,288 | multi-block split: `n_blocks = 3`, `(t, b)` item mapping, per-block bitmaps, the multi-list merge in `split_merge` (whose 64-entry input lists come from 3 distinct block lists, so a broken merge — the class of bug that killed `warp_first` — could not hide) | 0/16 differ (default, greedy, old) |
| C | 100 × 512 | `n_tokens > kSplitMaxRows` → dispatcher one-block fallback | 0/100 differ; default == forced one_block row for row |
| D | 4 × 262,200 | `n_blocks = 65 > kSplitMaxBlocks` → one-block fallback | 0/4 differ; same double check |
| E | host-side | `n_tokens = 0` throws; missing history with penalties throws; `temperature = 0` dispatches to the greedy kernel (mirrored 847); degenerate 1 × 4 shape (the reference uses the kernel's `sampled_k` clamp, `k = min(top_k, 64, nv) = 4`) | all four assertions hold |
| F | 4 × 248,320 | worst local-memory shapes launch on Arc; default = split with `n_blocks = 61`, near the merge's capacity | 0/4 differ (default, greedy, forced one_block) |

Two honesty notes, recorded rather than smoothed over:
- The C/D "default == forced one_block" cross-check is a *consistency*
  check, not a branch identifier — both fallback targets are correct, so
  a broken fallback *condition* would still produce reference-matching
  picks. The condition's coverage is structural instead: it is mirrored
  verbatim (862:864) and checked byte-for-byte, and both branches of the
  surrounding glue are behaviourally exercised (split by A/B/F, fallback
  by C/D).
- This plan's table row "F: greedy + one_block (parent)" was wrong as
  written: one_block cannot be run in-process without the env pin, so F
  runs default + greedy in the parent and forced one_block in the forked
  child (which also makes F the scenario that proves the 39,744 B shape
  at runtime, since the parent's greedy is only 39,232 B).

Full suite: **15/15** under `env -u LD_LIBRARY_PATH` (~8 s); the smoke
alone is ~0.5 s (the shapes are small; Step 2's "~15 s" figure was from
the mid-debug era). No GPU incident this step.

### T5 step 4 — parity driver (`poc/sycl/drivers/sampler_parity.cpp`)

The CUDA driver `src/kernels/sampler_parity.cpp` @ 47d6894 ported per the
block table: the entire host half (reference_pick, reference_cut,
sampled_reference, sampled_cut, mirror_select/mirror_pick, the host
Philox, window_of, sampled_k, the summary/exit contract) and every
fixture's data generation, reference calls, observability asserts, and
verdict prints are mirrored verbatim — **72 driver blocks** (the plan's
"~45" undercounted; interleaving host and CUDA lines inside the fixtures
splits blocks more than the fixture count suggests). Tree total: 259 OK
blocks, `check_mirrors.sh` exit 0.

Glue decisions, all documented in the driver header:
- `check(cudaError_t, …)` not mirrored (SYCL throws); alloc/copy/free →
  `ctx` calls; `cudaMemset(ptr, 0xFF)` → host-fill `-1` + copy (the
  unwritten-row-unmatchable prefill, comment mirrored too).
- **Streams:** `cudaStream_t` is a `void*` typedef, so the original's
  stream declarations and fixture 17's `{nullptr, cs}` loop stay
  verbatim; both "streams" are the one in-order `ctx.q` — *more*
  sequential than CUDA's legacy-stream implicit sync, so no assertion
  weakens. No fixture assertion depends on stream identity (17 checks
  picks, not stream behaviour).
- `DeviceRows` keeps its member shape and `sample()` (which still takes
  the `stream` argument, ignored); `bench_sampled` keeps its data,
  parameters, iteration counts, and printf format, with CUDA events
  replaced by a `chrono` wall clock around submit + `wait_and_throw()`.
- **Fixture 18a (graph capture, CUDA 1240:1260) excluded in writing**
  (Phase A has no stream capture; the `stream_capturing` dispatcher
  clause is deferred). 18's compare/make lambdas, the pre-capture host
  setup, and 18b's row cap (64 rows split / 70 one-block) port in full;
  the verdict label stays verbatim.

Result — all three path-pinned ctests **green**, not merely running
(each test's log prints its own path line, verified via `ctest -V`):

| ctest entry | env pin | runtime |
|---|---|---|
| `k_sampler_parity` | `STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0` (split, default) | 2.3 s |
| `k_sampler_parity_one_block` | `STRATA_SAMPLER_ONE_BLOCK=1` | 2.7 s |
| `k_sampler_parity_old` | `STRATA_OLD_SAMPLER=1` | 5.0 s |

Every fixture matches on every path; the observability asserts all fired
(the fixtures can see their features, so the matches are not vacuous):
"order is observable: yes", "one penalties stage (#53): yes" (draw share
within 4-sigma), "top_p before min_p: yes", "top_k list under ties: 0 of
1,340 positions differ; 280 sentinel positions", "sampled draws under
ties: 0 of 1,632 draws differ; 1,057 picks tie with another kept
token", "fallbacks: graph capture, row cap: 0 of 134 draws differ",
counter segmentation PASS.

Bench (provisional — Step 6 finalizes the §8 table), split path, Arc:

| n_vocab | rows | top_k | us per call |
|---|---|---|---|
| 248,320 | 1 | 20 | 148.9 |
| 248,320 | 1 | 64 | 420.2 |
| 248,320 | 4 | 20 | 138.4 |
| 248,320 | 4 | 64 | 404.0 |
| 248,320 | 8 | 20 | 216.8 |
| 248,320 | 8 | 64 | 609.5 |

Findings:
- **Checker-green ≠ compiles.** The first assembly was mirror-clean
  (the checker verifies against the source, which is CUDA) but failed to
  compile: two of my block ranges had swallowed CUDA lines (f3's two
  `check(cudaMemcpy…)` inside 523:534, and f17's `DeviceRows` ctor inside
  1132:1193). The compile is part of drift-detection for host-mirrored
  drivers, not just a build step; both ranges were split and re-verified.
- The old path is ~2× slower than split in the parity driver (5.0 s vs
  2.3 s), consistent with the 1,024-thread per-row design vs the block-
  parallel split — recorded, no action (Phase B decides the default).

Full suite: **19/19** under `env -u LD_LIBRARY_PATH` (~18 s). No GPU
incident this step.

### T5 step 5 — mutation tests M1–M3

Three kernel mutations, each reverted before the next (`git checkout` on
the kernel, which was clean at HEAD); the host references and `src/`
untouched throughout. Every mutation reded its designed fixture on **all
three** path-pinned ctest entries, with `check_mirrors.sh` red exactly on
the touched blocks while the mutation was in (expected, per T2–T4
doctrine):

| # | mutation (as run) | designed victim | observed red | checker red |
|---|---|---|---|---|
| M1 | every tie comparison flipped to break ties to the **highest** id — **six** sites: `take_first` (338:338), `fold_block` (148:148), greedy per-thread (142:143), old per-thread (246:250), one_block per-thread (476:477), split_part chains (621:632) | fixtures 16/17/18b | all three entries: 16 = 1,325/1,340 list positions + 1,349/1,632 draws, 18b = 119/134 draws (#13/#14: 2,793 failures; #15: 2,493 — its 16 count is 1,025/1,340); no other fixture reds | the six blocks, exactly |
| M2 | min_p before top_p (pre-#53 order): inserted `n_pre` pre-cut + the top_p cut run over `n_pre` in **both** serial tails (278:315 in old and one_block) | fixture 10 | all three entries: 91/256 differ, kernel never draws token 2; 91 = the shared-Philox-u arithmetic exactly (differ bands u ∈ [0.444, 0.571) ∪ [0.777, 1)); incidental: fixture 17 reds 2/1,632 on the two forced entries (its min_p = 0.05 configs fire the pre-cut on the serial tails) — the default entry's split tail was not mutated and stays clean | the two 278:315 blocks, exactly |
| M3 | `apply_penalties`: divide unconditionally (the pre-sign-fix reading) | fixture 4A | all three entries: 2/2 differ ("want 1 got 0"), no incidental reds | the 74:78 block, exactly |

Findings:
- **M1's site list was wrong in the plan, and the mutation caught it.**
  The plan (following the parent table's "`take_first`: `oi < bi` → `oi >
  bi`") assumed one shared helper. Run literally, it reded only the split
  path — the forced one_block/old entries stayed green — proving the
  1,024-thread kernels carry independent tie logic in `fold_block` and
  their per-thread argmax loops (each its own mirror block). Extending
  the flip to all six tie sites reded every entry on exactly the tie
  fixtures and nothing else. The parent table's "+ greedy ties" remains
  fixture-less (no fixture feeds tied logits to the greedy path); the
  greedy kernel's tie branch is covered structurally, not by a fixture —
  recorded, not fixed.
- **M2's red count matches the shared-u arithmetic, not the naive one.**
  Both sides draw with the same Philox `u`, so the mismatch probability is
  the measure of the bands where the two CDFs disagree (35 %, ~90 of 256
  observed 91), not "the reference draws token 2 ~22 % of the time". The
  later min_p block inside the mutated tails is a proven no-op under the
  mutation (`n_keep <= n_pre`), as the plan's analysis predicted — no
  fixture contradicted it.
- **M3 reds only its designed victim, and that is the right coverage.**
  The plan allowed incidental reds on other penalty fixtures; none
  occurred because 4A is the only fixture that puts a *negative history
  logit inside the candidate set* — 17's negative random history tokens
  cannot reach a top-20/64 list of N(0,3) logits, and the other penalty
  fixtures punish positive or unlisted tokens. Written here rather than
  read as "M3 barely does anything": it does exactly what the multiply-rule
  observability was built for.
- One transient incident: the first post-revert ctest run timed out test
  13 at 300 s; the direct rerun was 2.3 s green and the full suite green.
  Transient device flakiness (Steps 2/3 precedent), not a code change.

Restored state: kernel `git diff` empty, **19/19** under `env -u
LD_LIBRARY_PATH` (18.25 s), `check_mirrors.sh` exit 0 (259 OK).

### T5 step 6 — bench finalisation + full regression

The bench machinery (mirrored `bench_sampled`, `--bench`, the
`k_sampler_bench` ctest entry) landed in Step 4; this step finalised the
numbers and re-proved the suite.

- **Determinism fix:** `k_sampler_bench` pinned only `TIMEOUT 300` — the
  dispatcher's `sampled_path()` would measure whichever path the caller's
  shell happened to select. It now pins `ENVIRONMENT
  "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0"`; all three recorded
  runs print `sampled path: split top_k (default)` as their first line.

Final §8 table — split path, Arc, 248,320 vocab, N(0,3) logits, top_p
0.95, temperature 0.7, 3 warm-ups + 50 timed calls, chrono wall-clock
around submit + wait:

| rows | top_k | us per call (median of 3) | spread |
|---|---|---|---|
| 1 | 20 | 165.8 | 165.8–171.2 |
| 1 | 64 | 477.0 | 469.1–478.7 |
| 4 | 20 | 101.5 | 101.5–102.0 |
| 4 | 64 | 311.0 | 310.9–311.2 |
| 8 | 20 | 174.0 | 173.7–174.8 |
| 8 | 64 | 540.8 | 540.7–541.3 |

The three runs agree to ≤ 5 %. The Step 4 single-run provisional numbers
(148.9/420.2/138.4/404.0/216.8/609.5) sit within ~15 % at T = 1 and
20–27 % at T = 4/8 — wall-clock noise between sessions, no code change
in between.

Caveats (parent §8): **first Arc sampler figures**; no CUDA comparison
exists in this checkout — the numbers join T6's ledger. Performance
information, not a correctness gate; the exact-pick parity contract is
what Steps 4/5 proved.

Full regression: **19/19** under `env -u LD_LIBRARY_PATH`,
`check_mirrors.sh` exit 0 (259 OK).

### T5 deviations (parent §11 list)

Parent §11 requires the report to carry the deviations list; one
auditable place here (verified against `poc/sycl/kernels/sampler.cpp` in
step 7, not transcribed on faith):

1. **All-thread one-block tail.** CUDA's `submit_one_block` runs the
   shuffle-based `sampled_tail_warp` on warp 0 only (`if (warp != 0)
   return;`, 502:503). A SYCL barrier is work-group scope — threads that
   returned early would strand their peers (the T2/T3 stranded-peer rule,
   in reverse) — so all 1,024 threads run the tail in the Old kernel's
   serial shape (the mirrored block is 278:315, *not* the warp tail's
   368–420 lines, which appear only in the 32-thread `split_merge` tail).
   Identical `pick`; thread 0 writes. FP64 cost ×32 vs CUDA's one-warp
   tail — Phase B perf note, not a parity one (kernel glue ~629:634).
2. **Per-call scratch.** CUDA's per-(device, stream) `split_scratch()`
   slot cache (789:832, retired-buffer bookkeeping) becomes a per-call
   `sycl::malloc_device((n_tokens > 16 ? n_tokens : 16) × n_blocks ×
   kSelMax)` freed in order on the queue after both stages; the
   allocation *throws* on failure (never returns null) → the same
   one-block fallback CUDA's failed `cudaMalloc` (null scratch) takes.
   One documented in-order sync vs the slot cache (dispatcher glue
   ~1019:1043).
3. **Fixture 18a (graph capture) excluded in writing** — Phase A has no
   stream capture; the `stream_capturing` dispatcher clause is deferred to
   Phase B. Step 4 section.
4. **`atomic_ref` outcome (P1)** — PASS: work-group-scope
   `fetch_or`-based local-memory bitmaps, 0/256 words differ. Step 1
   section.
5. **Double-math outcome (P2/P2b)** — documented finding (1-ulp device-vs-
   host `exp` grid gaps, P2 informational) plus the P2b gate PASS with
   margins ≥ 6 orders of magnitude; no hand-rolled double `exp` needed.
   Step 1 section.

Phase B deferrals, with reasons (parent §2/§3): the coupled-draft kernels
(`coupled_stage_kernel`, `coupled_penalize_kernel`, `coupled_merge_kernel`,
`coupled_draft_sample` — CUDA 670–751) are not needed for the five-kernel
scope; the `stream_capturing` dispatcher clause has no graph capture in
Phase A (fixture 18a); the per-(device, stream) scratch slot cache becomes
the per-call allocation above.
