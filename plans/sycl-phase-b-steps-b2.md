# B2 step — GEMV / expert families (`plans/sycl-phase-b-steps-b2.md`)

Parent: `plans/sycl-phase-b.md` (Phase B), `plans/sycl-phase-b-steps.md` (step 3 of 9).
Predecessor: B1 (steps 2.4–2.10 must be **closed first**): `shared_expert.cu`'s native
path calls `bf16_gemv_fp32_mmvf`/`_multi`, which live in `native_bf16.cu` (B1 step 2.8,
not yet ported), and its legacy path calls `quantize_q8_0`/`quantize_q8_K` (B1 2.3,
done) plus the `s_gemv` remainder this batch absorbs (F9). Nothing in B2 may start
3.8 until `poc/sycl/kernels/native_bf16.cpp` exists and its driver is green.

Scope: the parent's 8 files / 3,175 lines / 38 kernels, **plus the unowned remainder of
`s_gemv.cu`** (F9, ~500 lines / 3 kernels): 8 files + 1 remainder ≈ **3,675 lines /
41 kernels**. All kernel work lands under `poc/sycl/`; the main tree is never touched.

| # | file (src/kernels/cuda) | lines / kernels | CUDA-driver coverage | SYCL driver (poc/sycl/drivers) |
|---|---|---|---|---|
| 3.1 | `s2_gemv.cu` | 63 / 1 | `s2_gemv_parity.cpp` | `s2_gemv_parity.cpp` (new mirror) |
| 3.2 | `s2_gemv_q8.cu` | 96 / 1 (1 dp4a-free; dp4a is F-note) | `s2_gemv_q8_parity.cpp` | `s2_gemv_q8_parity.cpp` (new mirror) |
| 3.3 | `s2_gemv_quads.cu` | 93 / 1 | `s_gemv_parity.cpp` bench sweep (**parked by Phase A**) | sweep un-parked (3.6) |
| 3.4 | `s2_gemv_fast.cu` | 153 / 1 | `s_gemv_parity.cpp` bench sweep (parked) | sweep un-parked (3.6) |
| 3.5 | `s_gemv.cu` remainder (F9) | ~500 / 3 | `s_gemv_q8k_parity.cpp` (+ sweep uses `s_gemv_split`) | `s_gemv_q8k_parity.cpp` (new mirror) |
| 3.6 | — (driver step) | — | `s_gemv_parity.cpp` lines 356–464 | **un-park** the sweep in the existing `poc/sycl/drivers/s_gemv_parity.cpp` |
| 3.7 | `bf16_gemv.cu` | 141 / 3 | `bf16_gemv_parity.cpp` | `bf16_gemv_parity.cpp` (new mirror) |
| 3.8 | `shared_expert.cu` | 346 / 9 | `shared_expert_parity.cpp` | `shared_expert_parity.cpp` (new mirror; F12) |
| 3.9 | `native_mmvq.cu` | 1,493 / 9 | `mmvq_multi_parity.cpp` | `mmvq_multi_parity.cpp` (new mirror; F11) |
| 3.10 | `s2_expert_grouped.cu` | 790 / 13 | **none** (F10) | **new** `s2_expert_grouped_parity.cpp` |
| 3.11 | — | — | — | batch close: suite, checker, report, parked-block amendments |

Status (updated as each step lands): 3.1–3.11 pending.

Inherited conventions (unchanged from Phase A/B0/B1): mirror blocks byte-verbatim with
original indentation; glue carries lambda indentation; no DPC++ fix may touch a
mirrored line; `check(cudaError_t, …)` / `launch_check()` / `check()` status helpers are
not mirrored (SYCL throws; their *call sites* stay verbatim over a glue no-op);
every CUDA stream maps to `ctx.q`; launches are exact padded work-groups with
in-kernel bounds checks; non-uniform work-groups are rejected by Arc; local memory
≤ 128 KiB per work-group; a knife-edge is a written finding, never a silent
tolerance; mutation tests for order-sensitive logic; `std::mt19937` fixtures;
rerun transients before recording; host reference code in drivers is mirrored
(pinned cross-file blocks are legal — Phase A's `dequant_s2_parity` pattern),
device plumbing and call sites are glue.

## Audit results (executed at plan-writing time)

Intrinsic/hazard map (B0 header coverage in parentheses):

| file | intrinsics | smem / barriers | launch shape | host-side hazards |
|---|---|---|---|---|
| `s2_gemv.cu` | `__half2float`/`__ushort_as_half` | none | 1-D, block 128, one thread per row | `exit(1)` validation; unconditional `cudaDeviceSynchronize` |
| `s2_gemv_q8.cu` | `__half2float`/`__ushort_as_half` (act fp16 scale decode) | `extern __shared__ float partial[tpr]`, tree + `__syncthreads` ×2 | 1-D, grid `n_out`, block `tpr` (driver sweep 32/64/128), **dynamic smem** | `exit(1)`; `GetLastError` + conditional sync |
| `s2_gemv_quads.cu` | `__low2float`/`__high2float`, `uint2` + `__half2` reinterpret loads | `extern __shared__ float partial[tpr]`, tree | 1-D, grid `n_out`, block `tpr`, **dynamic smem** | `exit(1)`; sync |
| `s2_gemv_fast.cu` | `float4` broadcast load from `__constant__` LUT (F13), `__low2float`/`__high2float`, `__half` staging | `extern __shared__ float smem[]` (staged `x` up to 2,048 halves + `tpr` partials), `__syncthreads` ×3 | 1-D, grid `n_out`, block `tpr`, `template<bool STAGE_X>`, **dynamic smem ≤ ~9.6 KB** | `ensure_lut()` (`cudaGetDevice`/`cudaMemcpyToSymbol`/per-device flag), `exit(1)`, sync |
| `s_gemv.cu` remainder (F9) | `__half22float2` ×2 (split kernel; **not in intrinsics.hpp — G13**), `__half2float`/`__ushort_as_half` (`q8k_at`/`q8_0_at`), `__shfl_down_sync` (q8_split), `s_iq4nl[16]` staging (Phase-A pattern) | `__shared__ signed char s_iq4nl[16]` static ×2; `extern __shared__ float partial[]` (split) | 1-D; q8k block 128; q8_split block 64 (4 warps per row, verify at port); split block `tpr` dynamic | `finish()` helper (GetLastError + sync); 6 host wrappers incl. 2 `_async` (stream-capture-legal — the reason they exist) |
| `bf16_gemv.cu` | `__shfl_down_sync` warp tree, `f32_from_bf16` | `extern __shared__ float scratch[tpr]` (split), tree | naive/warp: 1-D block 256; split: grid `n_out`, block `tpr` (power of two, driver-driven), **dynamic smem** | `finish()`; power-of-two `exit(1)` validation; dispatch threshold `n_out >= 64` |
| `shared_expert.cu` | `exp` **double** (legacy swiglu + scalar gate), `__expf` ×2, `__fdividef` ×2 (native path; B0 gaps 64/2 ulp), `__shfl_down_sync`/`__shfl_sync` on **double** (`warp_sum_d` — B0 `shfl32` covers 8-byte values, verified in `intrinsics.hpp`), `f16_from_f32` | `__shared__ double scratch[8]` (scalar_gate, static) | 1-D block 128 elementwise; `<<<1, 256>>>` scalar gate; **1-D block 1 and 1..8** native sigmoids (Arc sub-16 work-group probe, stop condition 1); 2-D grid `dim3(g_embd, n_tok)` scale_rows | scratch layout math; `static const bool batch = [] {getenv…}()` host cache; calls into `s2_gemv_q8` (3.2), `s_gemv_q8k_split`/`s_gemv_q8_0_split` (3.5), `native_mmvq`/`native_quantize_q8_1` (3.9), `bf16_gemv_fp32_mmvf(_multi)` (B1); driver's CUDA-graph section (F12); 3× `exit(1)` |
| `native_mmvq.cu` | `__byte_perm` ×13, `__vsubss4` ×2, `__half22float2` ×4 (G13), `__shfl_xor_sync`/`__shfl_down_sync` (`warp_sum`/`warp_max`), `make_int2`/`make_uint2` (**not in intrinsics.hpp — G13**), `STRATA_DP4A` (quant kernel), `__device__ __align__(4) int8_t iq4nl_values[16]` device constant (G11-b) | 4-D `__shared__ float partial[NW-1][NCOLS][ROWS][WARP]` in the multi kernel (**G10**, max 3×8×2×32×4 B = 6 KB); no smem in the per-row kernels | **2-D grid** `dim3(WARP=32, WARPS=4)` (G1); per-row and packed variants switched host-side; `__launch_bounds__` ×2 | F11 iq dispatch (types 16,17,18,21,22,29 → `iq_mmvq`/`iq_row_bytes` in `iq_kernels.cu`, B3); `g_multi_exact` host global + setter; validation helpers already throw-style (mirror cleanly); `launch_check()` glue |
| `s2_expert_grouped.cu` | `STRATA_DP4A` (dot4 + chunk_dot), `__ballot_sync` ×2, `__popc`, `__fmaf_rn`/`__fmul_rn`/`__fadd_rn`/`__fsub_rn`/`__fdiv_rn` (cpu_order family — B0 exact), `__shfl_down_sync` (`warp_sum`, `row_dot_cpu_order`'s 4,1,2-order reduce), `expf` (cpu_order swiglu), `__half2float`/`__ushort_as_half` (`f16_at`), `__expf` (grouped swiglu) | 2-D static arrays `xs_q[GMAX][H/4]`, `xs_d[GMAX][H/32]`, `hs_q[GMAX][FF/4]`, `hs_d[GMAX][FF/32]` (**G10**, gu staging 20 KB + 2.5 KB, down 5 KB + 0.64 KB); 5× 1-D arrays in `group_resident_kernel` (~2.6 KB); `warp_count[4]` | 1-D block 256 (THREADS); **2-D grids** `dim3(2*FF/GU_ROWS=40, cap_groups)` and `dim3(H/D_ROWS=40, cap_groups)` (G1); `<<<1, 32>>>`/`<<<1, 128>>>` selectors; `add_hits` 2-D `dim3(min(ceil(n_embd/256), 8), cap)` | `check()` glue; `kVerifyMaxT` from `verify_kernels.hpp` (B7 file — header-only `inline constexpr`, includable, no code dependency); `cudaMemcpyAsync` device→device in the cpu_order path → `ctx.q.memcpy` glue (in-order queue preserves sequencing); calls `quantize_q8_0`/`quantize_q8_0_scaled` (B1, done); 5× `exit(1)` |

Findings from the audit (numbering continues from B1's F1–F8):

- **F9 — the parent plan counts `s_gemv.cu` as "ported" (Phase A), but only the
  naive `s_gemv_kernel` + `s_gemv()` host wrapper is.** The unowned remainder is
  ~500 lines / 3 kernels: `s_gemv_q8k_kernel` (lines 91–142),
  `s_gemv_q8_split_kernel` (143–300), `s_gemv_split_kernel` (401–532), the device
  helpers `q8k_at`/`q8_0_at` (60–90), and host wrappers `s_gemv_split`,
  `s_gemv_split_async`, `s_gemv_q8k`, `s_gemv_q8k_split`, `s_gemv_q8_0_split`,
  `finish` (533–694). B2 **absorbs it** because two in-batch consumers need it:
  `shared_expert.cu` calls `s_gemv_q8_0_split`/`s_gemv_q8k_split`, and the
  un-parked `s_gemv_parity` bench sweep calls `s_gemv_split`. The remainder extends
  the existing `poc/sycl/kernels/s_gemv.cpp` TU (same CUDA source file → same port
  file). Adds ~1–1.5 d to the batch.
- **F10 — `s2_expert_grouped` has no CUDA parity driver** (parent plan already
  budgets ~1 d). The `bench/micro/moe_hit_parity.cu` its comments reference is
  **not in the repo**, so the new SYCL driver carries its own contract: it
  synthesises expert blobs in the documented Q2_0 geometry (H=2560, FF=640, QK=64,
  `ROW_GU=640 B`, `ROW_D=160 B`, fp16 scales — the literals the .cu itself restates),
  and its host reference is a double-precision scalar transcription of
  `cpu/expert.cpp`'s arithmetic (code `→ code-1` in the integer domain, fp16 group
  scale, fp16/fp32 activation scale, Q8_0 quantisation of the intermediate per
  `ggml_mul_mat`'s contract, `dst_index` row placement, gate-major layout). Covers
  all 10 public entries under `--selftest`: `moe_hit_grouped_s2` (both
  `x_scales` variants), `moe_hit_select`, `moe_hit_grouped_s2_dev`, `moe_hit_add`,
  `moe_hit_select_multi`, `moe_hit_grouped_s2_multi`, `moe_group_resident` +
  `moe_grouped_s2`, and `moe_hit_grouped_s2_cpu_order` (its reference is the
  8-lane CPU-order loop the kernel transcribes). Stop condition 6 caps the
  reference at ~400 lines.
- **F11 — `native_mmvq()` dispatches types 16, 17, 18, 21, 22, 29 (the six
  non-IQ4 iq formats) to `iq_mmvq`/`iq_row_bytes` in `iq_kernels.cu` (B3).**
  The B2 driver (`mmvq_multi_parity`) exercises only types 2, 8, 12, 13, 14, 23
  (its `CASES`), and `shared_expert`'s native weights use in-file types, so the
  iq branch is never reached in B2. The port replaces those six switch cases with a
  **glue throw** (`"native MMVQ iq branch (B3) not linked yet"`), leaving a
  written note; `strata/kernels/iq_kernels.hpp` is a pure declaration header and is
  included for the other symbols. **B3's `iq_kernels.cpp` step restores the mirrored
  dispatcher** when `iq_mmvq` lands (parked-block amendment to B3's step file).
  This is a loud, documented deviation — not a silent tolerance.
- **F12 — `shared_expert_parity.cpp`'s native-shared section uses CUDA stream
  capture** (`cudaStreamBeginCapture`/`EndCapture`/`GraphInstantiate`/`GraphLaunch`
  + the byte-identical-replay check). SYCL 1.21 has no stream-capture equivalent;
  the block is **excluded as glue** with a written note (G12) — it tests the CUDA
  runtime, not the kernels. The kernel-bearing checks survive: the native path's
  `rel < 2e-6` against the host sigmoid and the `refused_missing` rejection.
  The native path flows through `__fdividef`/`__expf` (B0-measured 2-ulp / 64-ulp
  gaps); if `2e-6` fails it is a P2b margin analysis (the gate sigmoid bounds the
  damage to `[0,1]`), never a silent widening.
- **F13 — `s2_gemv_fast.cu`'s `__constant__ float c_codes[256][4]` is
  runtime-filled** (`ensure_lut()` via `cudaMemcpyToSymbol`), so G4's
  `constexpr`-macro convention does not apply. Glue (G11): a USM device array
  behind a wrapper struct whose rows are `alignas(16)`, so the mirrored
  `*reinterpret_cast<const float4*>(&c_codes[byte][0])` compiles verbatim **and**
  is 16-byte aligned (the CUDA constant memory bank is; a plain USM `float` is
  only 4-byte aligned, which would make the `float4` reinterpret a hazard).
  `ensure_lut()` becomes glue taking `sycl::queue&` (G8 host management;
  single-device, filled once per process — CUDA's per-device `g_lut_ready[64]`
  collapses to one flag, documented).
- **F14 — B2 adds four dynamic-smem sites that B0.4's table does not cover**
  (B0.4's five sites are all B4/B5 files). B2's: `s2_gemv_fast` staged variant
  (max `(4096 halves * 2) + (128 * 4)` = **5,120 B**), `s2_gemv_quads`
  (`tpr * 4`), `s2_gemv_q8` (`tpr * 4`), `bf16_gemv_split` (`tpr * 4`). All far
  under the 128 KiB cap; rows land in report §B2's local-memory budget table.
  Mechanically these launches need G9 (runtime local range), not
  `strata_launch`'s fixed-128 pattern.
- **F15 — `intrinsics.hpp` gaps for B2** (glue-layer additions, B0-class):
  `__half22float2` (native_mmvq q4/q5 dots ×4, `s_gemv_split_kernel` ×2) and
  `make_int2`/`make_uint2`/`make_float2` constructors (iq4 table lookup, grouped
  down staging). Both are trivial; validated in situ by the bit-exact
  `mmvq_multi_parity` and `k_s_gemv_bench` sweeps, no separate probe.

New glue conventions introduced by B2 (all glue; none touch mirrored lines):

- **G9 — runtime local range (dynamic smem).** Where CUDA launches with
  `<<<grid, blockDim, smem_bytes>>>`, the SYCL submit passes an explicit
  `nd_range<1>(padded_items, blockDim)` and the `local_accessor` is sized by the
  same expression CUDA uses for `smem_bytes`. `strata_launch` (fixed local 128)
  does not apply. Padded items fall through the mirrored `o >= n_out`
  bounds checks; `tpr` values come from the mirrored driver sweeps (32/64/128/256
  — all Arc-compatible multiples of 16). One work-group per output row, so no
  grid padding is needed (`grid = n_out` is exact).
- **G10 — multi-dimensional static `__shared__`.** 1-D arrays use plain
  `local_accessor` (G3, the mirrored `name[i]` compiles as-is). 2-D/4-D arrays
  (`s2_expert_grouped` staging, `native_mmvq` multi partials) get a
  `local_accessor` plus a **chained-index proxy** wrapper, templated on the
  trailing dimensions, so mirrored lines like
  `partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i]` compile verbatim:
  `template<int A,int B,int C> struct acc4 { float* p; … nested operator[] … };`
  with a glue line `acc4<…> partial{smem_part.get_pointer()};` at the lambda
  scope. The 1-D `group_resident_kernel`/`hit_select_multi` arrays need no
  wrapper.
- **G11 — device-resident constants (runtime LUT + `__device__` arrays).**
  (a) Runtime-filled `__constant__` LUTs (F13) behind an `alignas(16)` wrapper
  struct with `operator[]` returning a row pointer. (b) `__device__`-scoped
  constant arrays (`iq4nl_values`) become a namespace-scope device pointer +
  a glue line in the lambda that shadows the name with the device pointer, so
  the mirrored `reinterpret_cast<const uint32_t*>(iq4nl_values)` reads device
  memory. Both are documented per use in the report.
- **G12 — CUDA graph-capture driver blocks are excluded as glue** (F12). The
  exclusion sits in a comment block naming the CUDA line range and the reason
  (no SYCL stream capture); the surviving checks keep their mirror blocks.
- **G13 — intrinsics.hpp additions** (F15): `__half22float2` and the
  `make_*2`/`make_*4` constructors.

Per-file parity expectations (the CUDA driver's own comparison logic is the spec;
device-vs-host math gaps are measured and recorded per P2):

- `s2_gemv`: decode is exact integer-to-float; the driver compares against the
  `dequantize_q2_0` chain at the driver's own relative tolerance (measured,
  printed). Expected green; the number goes in the report.
- `s2_gemv_q8`: same tolerance vs the host reference that quantises exactly as
  `ref/quant.py::q8_0`; **check 2 (the FP16-vs-Q8_0 gap must be ~1 % and
  observable)** exercises the activation contract — both paths use B1's
  `quantize_q8_0`, so F8 does not touch it (no fp32 division in Q8_0).
- `s2_gemv_quads` / `s2_gemv_fast` / `s_gemv_split`: checked against the naive
  reference inside the bench sweep at `rel <= 1e-4` per configuration (the
  driver's existing gate). These change summation order by design — the sweep's
  per-tpr numbers are the perf ledger's headline vs Phase A's T4 baseline
  (4.6–16.9 G weights/s).
- `s_gemv_q8k(_split)`: host reference dequantises the **same device Q8_K bytes**
  in double, so the F8-affected `d` is shared by both sides and the driver
  stays green (unlike `quantize_act_parity`, which compared bytes against
  ggml-exact host bytes).
- `bf16_gemv`: reference reads the same bf16 bits → tolerance is summation-order
  only; naive-vs-split agreement is the driver's own check; the activation
  contract's rival reading (fp16 activations) must differ measurably — the
  driver asserts it.
- `shared_expert`: the CUDA driver sets its tolerance from the fp16 intermediate
  conversion (~1e-2 asserted, measured ~2e-7 in the CUDA run — both sides
  perform the same quantisation, so it cancels); the structural traps
  (silu-on-up, elementwise gate, unweighted routed sum, weighted shared) each
  must stay >50 % / >5 % apart. Legacy path uses **double** `exp`/division
  (P2b: IEEE-exact on Arc) → expect green. Native path: F12's margin analysis.
- `native_mmvq`: the driver is **bitwise** (same Q8_1 bytes, multi-column vs
  T single-column calls) — `byte_perm`/`vsubss4`/`dp4a` are all B0-validated
  bit-exact primitives, so green is expected; the **negative control** (T > 4
  must differ when `multi_exact` is off) must also fire, or the test has no
  power. `multi_exact` on keeps every column bit-equal to a single-column call —
  the 2-D-grid G1 decomposition must preserve the CUDA warp/block indexing
  exactly.
- `s2_expert_grouped`: the CUDA contract (per the header comment) is
  "6.279e-08" against the CPU pool at the per-entry kernel; the new driver
  holds the same per-entry comparisons to the driver's mirrored/own tolerance
  and asserts the grouped path is bitwise the per-entry path (that is the
  file's central claim). The gate-major vs gate-interleaved trap (the
  documented first-version bug, worst rel 2.2e+03) is the mutation fixture
  below.

Mutation-test candidates (T4/T5 protocol: mutate → red on a designed fixture →
revert; one per file where the logic is order/semantics-sensitive):

| file | mutation | fixture that catches it |
|---|---|---|
| `s2_gemv` | swap the nibble bit order (MSB-first instead of LSB-first) | `s2_gemv_parity` rel check |
| `s2_gemv_q8` | decode the act scale big-endian (`blk[1]` low) | `s2_gemv_q8_parity` check 1 |
| `s2_gemv_quads` | swap `w2`/`w3` (bytes bits 4–7 onto lanes 0–1) | bench sweep rel ≤ 1e-4 |
| `s2_gemv_fast` | read the LUT before `ensure_lut` fills it (zeros) | bench sweep: all rows 0 vs reference |
| `s_gemv` remainder | drop the inter-warp smem combine in `s_gemv_q8_split_kernel` | `s_gemv_q8k_parity` rel |
| `bf16_gemv` | reverse the shfl offset order (1,2,4,8,16 instead of 16,…,1) | driver's naive-vs-split agreement |
| `shared_expert` | silu on UP instead of gate (the documented rival reading) | driver's observable-trap + kernel comparison |
| `native_mmvq` | swap the two `__byte_perm` selections in `q2_q8_dot` (0x5140 ↔ 0x7362) | `mmvq_multi_parity` Q4_0 case goes red |
| `s2_expert_grouped` | decode slot `i` from row-slot `i/2` (the documented first-version interleaving bug) | new driver: worst rel ~2.2e+03 |

## Stop conditions (stop and ask)

1. Arc/Level-Zero rejects a work-group size the CUDA code uses — the
   `native_scalar_sigmoid` kernels launch with blocks of **1 and 1..8** (sub-16).
   Probe sizes {1, 2, 4, 8} on Arc *before* wiring 3.8. If rejected, the padded
   launch (local 16) plus a glue bounds-check line before the mirrored store is
   the fallback — it keeps every mirrored line verbatim, but it is a written
   deviation, decided here, not silently at port time.
2. Any mirrored driver's own comparison fails on a math gap not covered by a
   written margin analysis (P2b) — including F12's native-shared `2e-6` gate.
3. A mirrored block cannot be made to compile without touching a mirrored line
   (icpx bug class) — isolate the kernel/file, record in the blocklist, continue
   the rest.
4. `shfl32` in the warp-reduction kernels (`bf16_gemv_warp`, `s_gemv_q8_split`'s
   4,1,2-ordered reduce, `native_mmvq`'s `warp_sum`/`warp_max`,
   `s2_expert_grouped`'s `row_dot_cpu_order`) behaves differently from the B0
   probes — re-probe before proceeding with that file. Note `warp_sum_d` is the
   first **double-valued** shuffle use in a port; the B0 primitive covers 8-byte
   values (two u32 words per lane) — verify it in 3.8 and record the check.
5. The G11 LUT alignment (F13) or any `reinterpret_cast` to a wider-aligned type
   (`float4`, `uint2`, `__half2`) is rejected by icpx on Arc — re-probe that
   specific access before any workaround.
6. The F10 driver's reference implementation (blob synthesis + double-precision
   transcription) grows past ~400 lines — re-cut the driver's entry coverage
   (the group family and the cpu_order path are the first to drop) with a written
   note, rather than silently shrinking the reference.
7. `U_RESULT_ERROR_DEVICE_LOST` / ctest timeout — rerun before recording
   anything.
8. The F11 glue throw is ever reached by a B2 driver run (it should be
   unreachable per `CASES`) — that would mean the driver's fixture changed shape;
   stop and re-examine B3 ordering rather than linking iq work early.

## Effort

Parent plan: 5–6 d + 1 d driver. F9 absorbs ~500 lines / 3 kernels of `s_gemv.cu`
(+1–1.5 d), and the F10 driver's self-contained reference sits inside the
parent's existing +1 d. **Revised: ~7–9 d.**

## Done when (B2 complete)

- 9 kernel TUs under `poc/sycl/kernels/`: `s2_gemv.cpp`, `s2_gemv_fast.cpp`,
  `s2_gemv_q8.cpp`, `s2_gemv_quads.cpp`, `s_gemv.cpp` (extended, F9),
  `bf16_gemv.cpp`, `shared_expert.cpp`, `native_mmvq.cpp`,
  `s2_expert_grouped.cpp` — all compiled with icpx `-fsycl -O2 -std=c++20`.
- New ctest entries green under `env -u LD_LIBRARY_PATH` (TIMEOUT 120 each):
  `k_s2_gemv_parity`, `k_s2_gemv_q8_parity`, `k_s_gemv_q8k_parity`,
  `k_bf16_gemv_parity`, `k_shared_expert_parity`, `k_mmvq_multi_parity`,
  `k_s2_expert_grouped_parity` — 7 new; the existing `k_s_gemv_parity`
  (selftest) and `k_s_gemv_bench` are re-armed by the 3.6 un-park (the bench now
  runs the split/quads/fast sweep). Suite: 35 (post-B1) + 7 = **42 tests**.
- `poc/sycl/check_mirrors.sh` exit 0 tree-wide, including the un-parked
  `s_gemv_parity` sweep blocks and the F11 exclusion comment staying outside any
  mirror block.
- Mutation tests: every candidate above run, red-verified on its fixture,
  reverted; any skipped candidate carries a written reason.
- `plans/sycl-phase-b-report.md` §B2 written: per-file section (what mapped 1:1,
  the glue lines, G9–G13 uses, local-memory budget rows — F14's four dynamic
  sites + G10's static arrays + `scalar_gate`'s `scratch[8]` — parity numbers,
  F9–F15, and the **perf ledger**: naive/split/quads/fast at the two real expert
  shapes (2560×640, 640×2560) vs Phase A's T4 baseline 4.6–16.9 G weights/s),
  plus the parked-block amendments — B3 (restore the `native_mmvq` iq dispatcher
  once `iq_kernels.cpp` lands), and the Phase A `s_gemv_parity` un-park note.
- Findings F9–F15 acknowledged in the parent plan's B2 row and the steps
  file's Step 3 section (effort re-based to ~7–9 d).
