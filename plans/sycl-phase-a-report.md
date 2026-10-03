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

## T5 — `sampler` port — **in progress** (step 2 of 7 done)

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
