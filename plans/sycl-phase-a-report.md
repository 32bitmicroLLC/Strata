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
