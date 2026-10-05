# B1 step — elementwise / streaming batch (`plans/sycl-phase-b-steps-b1.md`)

Parent: `plans/sycl-phase-b.md` (Phase B), `plans/sycl-phase-b-steps.md` (step 2 of 9).
Predecessor: B0 (`plans/sycl-phase-b-steps-b0.md`) — committed `3df7f5f`. The intrinsic
layer `poc/sycl/include/sycl_compat/intrinsics.hpp`, the shuffle primitive `shfl32`,
the launch idiom `strata_launch` (padded 1-D `nd_range` + per-item bounds check), and
the mirror-marker convention (`// SYCL-MIRROR-BEGIN src/kernels/cuda/<file>.cu:<a>:<b> @ <hash>` …
`// SYCL-MIRROR-END`, checked by `poc/sycl/check_mirrors.sh`) all already exist.

Scope: 9 `.cu` files / 1,718 lines / 30 kernels, plus the iq-dequant subset of
`iq_kernels.cu` (~230 lines, see finding F4) that `dequant_bf16.cu` hard-references.
All kernel work lands under `poc/sycl/`; the main tree is never touched.

| # | file (src/kernels/cuda) | lines / kernels | CUDA-driver coverage | SYCL driver (poc/sycl/drivers) |
|---|---|---|---|---|
| 2.1 | `rope.cu` | 136 / 1 | `rope_parity.cpp` | `rope_parity.cpp` (shared with 2.2) |
| 2.2 | `native_rope.cu` | 102 / 1 | `rope_parity.cpp` (check 5) | `rope_parity.cpp` (shared with 2.1) |
| 2.3 | `quantize_act.cu` | 313 / 5 | `quantize_act_parity.cpp` | `quantize_act_parity.cpp` |
| 2.4 | `native_gr_postops.cu` | 115 / 3 | **none direct** (F1) | **new** `native_gr_postops_parity.cpp` |
| 2.5 | `dequant_bf16.cu` (+ iq subset) | 248 / 1 (+8) | `dequant_bf16_test.cpp` (shard tool), type-8 slice in `elementwise_parity.cpp` | **new-mode** `dequant_bf16_parity.cpp` (F5) |
| 2.6 | `elementwise.cu` | 325 / 14 | `elementwise_parity.cpp` (7 of 14 entries) | `elementwise_parity.cpp` |
| 2.7 | `kv_q8.cu` | 127 / 2 | `kv_q8_parity.cpp` (q8 half; FP16 half needs B5, F3) | `kv_q8_parity.cpp` (q8 subset) |
| 2.8 | `native_bf16.cu` | 186 / 2 | **none direct** (F1) | **new** `native_bf16_parity.cpp` |
| 2.9 | `cvec.cu` | 166 / 1 | `cvec_parity.cpp` (fused-gr section needs B4, F2) | `cvec_parity.cpp` (subset) |
| 2.10 | — | — | — | batch close: suite, checker, report, parked-block amendments |

Status (updated as each step lands): 2.1 ✓, 2.2 ✓, 2.3 ✓ (Q8_0 byte-
exact; Q8_K red by construction under F8 → `k_quantize_act_parity` runs as a
documented ctest `WILL_FAIL`; mutation tested, report §B1.2),
2.4 ✓ (`native_gr_postops`; new F1 driver; Arc `fmaf` signed-zero gap fixed in
`intrinsics.hpp` `fmaf_rn`; ULP gates documented, mutation tested, report
§B1.3), 2.5 ✓ (`dequant_bf16` + `iq_dequant` F4 closure; new F5 driver with
--selftest; all 16 types bit-exact; `-Wno-c++11-narrowing` documented
compiler exception; iq-only types lack a chain reference, report §B1.4),
2.6 ✓ (`elementwise`; 14 kernels + F6 execution smoke; two host-reference
findings pinned and measured (icpx host constant-multiply fold; host
`std::fma` is two-rounding → FMA3 reference); `rsqrtf` glue; all seven
covered entries pass the mirrored gates, report §B1.5),
2.7 ✓ (`kv_q8`; both kernels, q8 driver subset -- FP16 cross-check parks in
B5 per F3; the F8 division gap is the step's one knife edge (variable
`x / sf`); constant-divisor division probed IEEE-exact on the device;
checks 1+2 bit-exact, mutation tested, report §B1.6),
2.8 ✓ (`native_bf16`; new F1 dedicated driver -- no CUDA-side parity
coverage exists for these entries; all 8 adaptive block sizes bit-exact on
single + multi; the host reference runs entirely in pure-C integer arithmetic
because icpx host -O2 flushes f32/double subnormals in plain float ops and
miscompiles inline FMA3 asm (both measured); device fmaf + glue verified on
all 125 degenerate zero combinations; float2 glue-size bug found and fixed,
mutation tested, report §B1.7),
2.9 ✓ (`cvec`; the shfl32 emulation's join moved from the work-group
barrier to the 32-wide spin barrier because the mirrored stage-2 shuffle
runs inside a warp-0-only branch (measured: block-wide dot came out
warp-local); project within 4.9e-7 of the double reference, add / off /
untouched bit-exact, write-path checks park in B4 per F2, report §B1.8),
2.10 ✓ (batch closeout; suite 35/35, checker green, mutation tests all
red-verified, parked-block amendments recorded, report §B1.9).

Inherited conventions (unchanged from Phase A/B0): mirror blocks byte-verbatim with
original indentation; glue carries lambda indentation; no DPC++ fix may touch a
mirrored line; `check(cudaError_t, …)` is not mirrored (SYCL throws); every CUDA
stream maps to `ctx.q`; launches are exact padded work-groups with in-kernel bounds
checks; non-uniform work-groups are rejected by Arc; local memory ≤ 128 KiB per
work-group; a knife-edge is a written finding, never a silent tolerance; mutation
tests for order-sensitive logic; `std::mt19937` fixtures; rerun transients before
recording.

## Audit results (executed at plan-writing time)

Intrinsic/hazard map of the nine files (B0 header coverage in parentheses):

| file | intrinsics | smem / barriers | launch shape | host-side hazards |
|---|---|---|---|---|
| `rope.cu` | none in kernel; host float64 `pow/cos/sin` table build | none | 1-D, block 128, one thread per row | `mrope_table()` (defined in `native_rope.cu`), `rope_scaling_set/()` host state |
| `native_rope.cu` | `powf`, `rope_scaled_angle` (host-device header helper) | none | **2-D grid** `(ceil(width/2/128), rows)`, block 128 | `mrope_tab[64]` atomics + `cudaGetDevice`; `mrope_pos` is `#if __CUDACC__`-guarded in `mrope.hpp` → glue must define it (F7) |
| `quantize_act.cu` | `__fmul_rn`, `f16_from_f32`, device **double division + `rint`** | none | 1-D, blocks 128/64, one 32- or 256-element block per thread | `nearest_int_dev` magic-number tie rule (transcribed, not rewritable) |
| `native_gr_postops.cu` | `expf`, `__fmaf_rn`, `__fmul_rn`, `__fadd_rn` | none | 1-D, block 256 | signed-zero care is explicit (`scale_zero_bias`); `pre_gated<true/false>` template |
| `dequant_bf16.cu` | `__half2float`, `__ushort_as_half`, `__float_as_uint`, `__half_as_ushort`, `__float2half_rn`, `__constant__` lookup table | none | 1-D, block 256, one 32-element group per thread | **`iq_only` dispatch into `iq_kernels.cu` (F4)**; 10-type template dispatch |
| `elementwise.cu` | `__fmul_rn`, `__fadd_rn`, `expf`, `log1pf`, device **double `exp`** (silu), `rsqrtf`, `__shfl_down_sync`, `__syncthreads`, `__threadfence_system` ×5, `strata_spin_pause` (from `dp4a.hpp`) | `__shared__ int hit` in `copy_rows_from_mapped_kernel` | 1-D, blocks 256/128; doorbell kernels 1×1 and 1×1024 | 5 kernels (doorbell ×3, mapped-copy ×3) have **no CUDA driver coverage** (F6); `dp4a.hpp` include must be dropped (B0 macro conflict) |
| `kv_q8.cu` | `__ldg`, `__shfl_xor_sync` (16-wide, 2 warps), `__float2int_rn`, `f16_from_f32`, **`char4`/`ushort4` vector types** | `__shared__ float warp_max[2]` | **3-D grid** `(n_head_kv, head_dim/64, 2)`, block 64; gather 1-D block 256 | `KvHostPools` (mapped pinned host pools — Phase C; driver passes `nullptr`) |
| `native_bf16.cu` | `__shfl_xor_sync`, `__fmaf_rn`, `f32_from_bf16`, `float2` | `__shared__ float partials[32]` / `partials[NT][32]` | 1-D, **block sizes 32…256 in steps of 32** (96/160/192/224 included) | 2×8 template instantiations via macro dispatch |
| `cvec.cu` | `__expf` (inside `sigmoidf_`), `__shfl_xor_sync` ×2 (two-stage 16-wide), `fmaf` | `__shared__ float part[8]` | **2-D grid** `(hc, T)`, block 256 | multi-device table plumbing (`cudaGetDevice/SetDevice/Malloc/Copy/Free` over 64 devices); `__expf` 64-ulp gap (B0-measured) touches only the `write=true` path |

Findings from the audit (amend the parent plan's "all drivers exist, mirror-only"
claim for B1):

- **F1 — two files have no direct parity driver in the CUDA tree either.**
  `native_gr_postops.cu`'s three entries are called only from `gr.cu` (B4);
  `native_bf16.cu`'s two entry points are called only from `gr.cu`, `shared_expert.cu`
  (B2) and `ple.cu` (B6). Their SYCL coverage therefore cannot ride on B4/B2/B6
  driver mirrors. **B1 gets two small dedicated drivers** (new-driver contract header
  each: "what this test can and cannot see"). ~1 d total.
- **F2 — `cvec_parity.cpp`'s write-path section depends on `fused_gr.cu` (B4).**
  Lines ~148–190 run `k::fused_gr_read(a, …)` and compare cvec's folded write against
  it. B1's driver mirrors everything else and replaces that section with a
  host-transcription check of the documented arithmetic `R += bo * 2·sigmoid(inj/hc)`.
  The fused-vs-cvec cross-check's mirror blocks are **parked in B4's fused_gr driver**.
- **F3 — `kv_q8_parity.cpp`'s FP16-pool cross-check depends on `qsa.cu` (B5)**
  (`kv_append_step`/`kv_gather_step` live there, not in `kv_stream.cu`). B1's driver
  mirrors the q8 path only (append + gather vs the host reference, `host = nullptr`);
  the FP16 section's mirror blocks are **parked in B5's qsa driver**.
- **F4 — `dequant_bf16.cu`'s `iq_only` dispatch hard-references `iq_kernels.cu` (B3).**
  `dequant_f16/f32` call `iq_dequant_f16/f32` + `iq_row_bytes`; the B1 driver cannot
  link without them. The **iq dequant surface** of `iq_kernels.cu` —
  `dequant_flat_kernel`, `dq_dispatch`, the 10 `dq_*` device functions, `cvt`, `is_iq`,
  `iq_row_bytes`, `iq_supported`, `iq_dequant_f16/f32` (and whatever helpers the
  transitive closure needs, e.g. `get_int_b2/b4`, `unpack_ksigns`,
  `get_int_from_table_16`) — ports in B1 as `poc/sycl/kernels/iq_dequant.cpp` with
  mirror blocks pinned to `src/kernels/cuda/iq_kernels.cu` line ranges (cross-file
  pins are checker-legal; Phase A used them). **B3's `iq_kernels.cpp` mirrors the
  remainder and stops covering the pinned ranges.** B3 links `k_iq_dequant` for the
  shared host functions. (The `vec_dot_*`/`mmvq`/`quantize_q8_1`/`embed_rows`/`gu`
  parts stay in B3.)
- **F5 — `dequant_bf16_test.cpp` needs a model shard; the repo has none.**
  The CUDA test is a manual tool (not a ctest entry). The SYCL mirror
  `poc/sycl/drivers/dequant_bf16_parity.cpp` keeps the shard mode but adds a
  `--selftest` mode that synthesises block bytes per type (random + tie/edge fixtures)
  and compares against the validated CPU chain `strata/artifact/dequant.hpp`
  (same reference as `dequant_s2_parity` — "the chain, not a second opinion").
  ctest runs `--selftest`.
- **F6 — 6 of `elementwise.cu`'s 14 entries have no parity coverage, on CUDA either**
  (`doorbell_ring/wait/publish`, `copy_from_mapped`, `copy_rows_from_mapped`,
  `copy_i32_from_mapped` — and `add_inplace`, which only the engine exercises).
  They port and compile in B1; no parity gate until Phase C's host flow exists.
  Recorded as a known coverage gap, not hidden.
- **F7 — `mrope_pos` is absent from `mrope.hpp` under SYCL** (its body is
  `#if defined(__CUDACC__) || defined(__HIPCC__)`-guarded). Both `rope.cu` and
  `native_rope.cu` call it from device code. Glue defines
  `strata::kernels::mrope_pos` (same body, `__ldg` via the B0 macro) in each port TU.
  `rope_neox_pair`/`rope_scaled_angle` are plain `inline` under the non-CUDA branch of
  their `STRATA_*_HD` macros → host-device by default, callable from kernels (verify
  at first compile).

- **F8 — device float division on Arc is not IEEE-correctly-rounded**
  (measured during step 2.3; evidence: `b0/b0_divprobe.cpp`, `b0/b0_divprobe2.cpp`,
  `b0/b0_q8k_audit.cpp`). The host `1.0f/x` in the same compiler (icpx 2026.1,
  -O0 through -O2) is IEEE-exact: **0 of 4,194,304** operands off vs the exact
  double-quotient rounded RNE. The device `1.0f/x` in a SYCL kernel differs by
  **exactly 1 ulp in 14 %** of the same operands (200-exponent log-uniform sweep,
  identical result at -O0/-O1/-O2 and under `-fp-model precise`), and the Q8_K
  scale chain (`iscale = -127.0f/max; *d = 1.0f/iscale;`) compounds to **up to ~4
  ulp in d**: the `b0_q8k_audit` replay of the driver's own `ka` fixture shows
  **711/2048 blocks (34.7 %) with a differing f32 `d`** (586 by 1 ulp, 125 by 2–4
  ulp), while **every `qs`, `bsum`, and all Q8_0 bytes are exact** — the kernel
  logic is fully faithful; the residual is solely the non-IEEE division. Consequence:
  the mirrored `quantize_act_parity` driver's Q8_K byte comparison **cannot pass on
  this toolchain**; `k_quantize_act_parity` runs with ctest `WILL_FAIL TRUE`
  (a loud, documented expected failure, not a tolerance) until Phase C re-examines
  the gate — fixed icpx, or a driver update with a documented 1-ulp band on `d` and
  iscale-derived `qs`. The Q8_0 half of the same driver passes byte-exact today
  (its fp16 scale rounding masks the fp32 1 ulp; its quants use the IEEE-exact
  device double division).

New glue conventions introduced by B1 (all glue, none touch mirrored lines):

- **G1 — multi-dim grids.** `native_rope` (2-D), `cvec` (2-D), `kv_q8` append (3-D).
  The launch stays `strata_launch(q, gridDim.x*gridDim.y*gridDim.z, blockDim.x, …)`;
  the lambda scope gets glue `struct dim3 { int x, y, z; };` plus `gridDim`,
  `blockDim`, `threadIdx` (from `local_id`), and `blockIdx` decomposed from the flat
  global id in CUDA's order (x fastest). `blockIdx.z` distinguishes K/V in
  `kv_append_q8_kernel`. Padded items hit the existing per-kernel bounds checks
  (`row >= rows || pair >= …`, `i >= total`, etc.); where a kernel has none, the
  launcher passes the exact logical count and the grid is already a work-group
  multiple of logical units — verify per kernel at port time and add the check as a
  bounds-check glue line if missing (never inside a mirror block).
- **G2 — `__syncthreads()`.** `#define __syncthreads() it.barrier()` (the `nd_item`
  reference is named `it` inside the submit handler; smem kernels use the raw
  `q.submit` + by-value-captured accessor pattern from `launch.hpp`, not
  `strata_launch`).
- **G3 — `__shared__ T name[n];`.** Not macroable (type-position + following type).
  The line becomes a **glue line between two split mirror blocks**:
  `sycl::local_accessor<T, 1> name(n, it.get_group());` at the lambda indentation.
  Note the semantic delta: CUDA shared arrays are uninitialized, `local_accessor`
  is zero-initialized — every B1 use site writes before the barrier-protected read,
  so the delta is unobservable; note it in the report per site. All B1 smem arrays
  are ≤ 128 B (`warp_max[2]`, `hit`, `part[8]`, `partials[32]`, `partials[8][32]`).
- **G4 — qualifiers.** `__global__`, `__device__`, `__forceinline__`,
  `__launch_bounds__(…)`, `__host__` drop at mirror-block boundaries (T5 pattern: the
  qualifier line sits outside the mirrored block). `__constant__` → `constexpr`
  macro (first user: `dequant_bf16`'s `kv_iq4nl` table; B2's `s2_gemv_fast`/`s_gemv`
  reuse it).
- **G5 — header additions to `sycl_compat/intrinsics.hpp`** (glue layer, allowed):
  `char4` and `ushort4` structs (CUDA layouts: 4× `char`; 4× `unsigned short`),
  needed by `kv_gather_q8_kernel`. Validated by the `kv_q8` parity driver itself
  (bit-exact vs the host scalar reference), so no separate probe.
- **G6 — `strata_spin_pause()`.** `elementwise.cu` includes `dp4a.hpp` for it; the
  B1 port drops that include (B0: `STRATA_DP4A` macro conflict) and defines
  `inline void strata_spin_pause() {}` as glue (a `__builtin_ia32_pause()` body is an
  acceptable refinement; measure only if Phase C cares). Only `doorbell_wait_kernel`
  uses it, which is not parity-gated in B1 (F6).
- **G7 — `__threadfence_system()`.** B0 header already maps it to `threadfence()`
  (host atomic fence). With USM host-shared pointers the ordering is coherent in
  practice; the real doorbell protocol is exercised in Phase C. Recorded, not gated.
- **G8 — host device-management lines.** `cudaGetDevice/cudaSetDevice/cudaMalloc/
  cudaMemcpy/cudaFree/cudaDeviceSynchronize` inside mirrored host functions become
  glue (single device: `cur_device() == 0`; `sycl::malloc_device` throws;
  `ctx.q.memcpy`). These lines are not mirrorable (not compilable host C++) and are
  listed per file in the report, T5-style.

Per-file parity expectations (the CUDA driver's own comparison logic is the spec;
device-vs-host math gaps are measured and recorded per P2):

- `rope`: table build is host float64 on both sides → bit-exact; kernel rotation is
  float32 with the same table → bit-exact (the driver's checks 1–2). Native path
  (check 5): device `powf` + float32 trig vs the table path — held to the driver's
  own tolerance; measure the gap and record it.
- `native_rope`: same as above plus `mrope_pos` glue (F7); overlap checks and
  `std::invalid_argument`/`std::runtime_error` host paths mirror as-is.
- `quantize_act`: device **double division + `rint`** in `quantize_q8_0_kernel` is the
  knife-edge — MEASURED IEEE-exact on Arc (Q8_0 byte-exact, round trip bit-exact,
  all 5 distributions; `b0_q8k_audit`). `nearest_int_dev`'s magic-number tie rule and
  the `__fmul_rn` no-FMA-contract pin are bit-exact (macros hold). **BUT F8**: the
  Q8_K float divisions (`-127.0f/max`, `1.0f/iscale`) are 1 ulp off on ~35 % of blocks
  (up to ~4 ulp through the chain) → the mirrored driver's Q8_K byte check is red by
  construction; `k_quantize_act_parity` is `WILL_FAIL` with the full margin analysis in
  the report, pending the Phase C gate decision.
- `native_gr_postops`: `expf` device-vs-host ulp measured (P2); the signed-zero
  fixtures (`scale_zero_bias`) are bit-exact under IEEE `fma`.
- `dequant_bf16` (+ iq subset): pure bit ops + one float multiply per element;
  `f2bf`/`h2f` intrinsics B0-validated; bit-exact expected including the NaN-scale
  path (the `elementwise_parity` type-8 slice already guards the NaN-kept case).
- `elementwise`: `gdn_gate` (`expf`+`log1pf`), `silu` (double `exp` — T5 precedent:
  1 ulp on ~11 % of args, gate margins ≫ that; mirror the driver's tolerance, don't
  invent one), `rms_norm_weighted` (`rsqrtf` — verify DPC++ provides it; if not, glue
  macro `1.0f / sqrtf(x)` and measure), `embedding_gather`, `scale/add`,
  `f32→f16/bf16` bulk → per-driver comparison as mirrored. Doorbell/mapped kernels:
  compile-only in B1 (F6).
- `kv_q8`: q8 append/gather bit-exact vs the mirrored host reference
  (quantize-against-the-stored-scale and the host-pool skip both exercised;
  `host = nullptr` as in the CUDA driver). `char4`/`ushort4` validated in situ (G5).
- `native_bf16`: the driver reference transcribes the exact order (strided pairs, two
  ordered `__fmaf_rn`, XOR butterfly, block partials) → bit-exact expected. The
  kernel's documented property — multi-row output bit-identical to per-row single-row
  launches — is its own check. **Probe the unusual local sizes 96/160/192/224 on Arc
  before writing the dispatcher** (if Level-Zero rejects them, the finding is written
  before any workaround).
- `cvec`: steering (`write=false`) path is bit-exact (no `__expf` involved);
  `write=true` path flows through `sigmoidf_` (`__expf`, 64-ulp B0 gap) → hold it to a
  measured-ulp tolerance, recorded. F2's host-transcription section replaces the
  fused-gr cross-check for B1.

Mutation-test candidates (T4/T5 protocol: mutate → red on a designed fixture →
revert; one per file where the logic is order/semantics-sensitive):

| file | mutation | fixture that catches it |
|---|---|---|
| `rope` ✓ | delete the `n_rot..head_dim` tail copy | any row: tail must equal input (36,864 over 1e-6 → green after revert) |
| `native_rope` ✓ | swap the two rotation outputs (`orow[pair]` ↔ `orow[pair+half]`) | rotation fixture (24,452/24,460 over 3e-3 → green after revert) |
| `quantize_act` ✓ | divide by the stored fp16 `d16` instead of `d32` (subtlety 1) | Q8_0 lines only (Q8_K red per F8 anyway): 136,515 failures → 0 after revert |
| `native_gr_postops` | replace `scale_zero_bias` with plain multiply | signed-zero fixture (`-0.0f` in/out) |
| `dequant_bf16` | break `f2bf`'s RNE (drop the `+0x7fff` add's low-bit term) | RNE tie fixture (x.000…5 in bf16 ulps) |
| `elementwise` | drop the `/ cols` in `rms_norm_weighted` (mean→sum) | driver's rel comparison, off by √cols |
| `kv_q8` | quantize against `amax/127` directly instead of the stored scale | scale-tie fixture |
| `native_bf16` | swap the two ordered `__fmaf_rn` per pair | bit-exact driver turns red |
| `cvec` | flip the update sign (`-dot` → `+dot`) in mode 0 | driver's wrong-way assertions |

## Stop conditions (stop and ask)

1. Arc/Level-Zero rejects a work-group size the CUDA code uses (mmvf 96/160/192/224,
   any of the 64/128/256) — a re-tile is a batch decision, not a silent edit.
2. Any mirrored driver's own comparison fails on a math gap that is not covered by a
   written margin analysis (P2b treatment) — no silent tolerance widening.
3. A mirrored block cannot be made to compile without touching a mirrored line (icpx
   bug class) — isolate the kernel/file, record in the blocklist, continue the rest.
4. `shfl32` composed with shared memory (cvec two-stage, native_bf16 partials)
   behaves differently from the B0 probes — re-probe before proceeding with that file.
5. `U_RESULT_ERROR_DEVICE_LOST` / ctest timeout — rerun before recording anything.
6. The F4 transitive closure of the iq-dequant surface grows past ~300 lines
   (it would mean B1 is pulling in B3's real work) — re-cut the boundary or move the
   dequant iq dispatch to B3.

## Effort

Parent plan: 3–4 d. The audit findings add: 2 new mini drivers (F1, ~1 d), iq-dequant
subset port + selftest dequant driver (F4/F5, ~1 d), the new G1–G8 conventions land in
the first two files. **Revised: ~4–5 d.**

## Done when (B1 complete)

- 10 new PoC TUs under `poc/sycl/kernels/`: `rope.cpp`, `native_rope.cpp`,
  `quantize_act.cpp`, `native_gr_postops.cpp`, `dequant_bf16.cpp`,
  `iq_dequant.cpp` (F4), `elementwise.cpp`, `kv_q8.cpp`, `native_bf16.cpp`,
  `cvec.cpp` — all compiled with icpx `-fsycl -O2 -std=c++20`.
- 8 ctest entries green under `env -u LD_LIBRARY_PATH` (TIMEOUT 120 each):
  `k_rope_parity` (covers 2.1+2.2), `k_quantize_act_parity`
  (**`WILL_FAIL TRUE` per F8** — expected failure, loudly, until the Phase C gate
  decision), `k_native_gr_postops_parity`, `k_dequant_bf16_parity --selftest`,
  `k_elementwise_parity`, `k_kv_q8_parity`, `k_native_bf16_parity`,
  `k_cvec_parity`. Suite total: 27 + 8 = 35 tests.
- `poc/sycl/check_mirrors.sh` exit 0 tree-wide (all B1 blocks green; no pre-existing
  block regressed).
- Mutation tests: every candidate above run, red-verified on its fixture, reverted;
  any skipped candidate carries a written reason.
- `plans/sycl-phase-b-report.md` §B1 written: per-file section (what mapped 1:1, the
  glue lines, G-convention uses, local-memory budget row, parity numbers, math-gap
  measurements, F1–F7), plus the parked-block amendments — B3 step file (iq-dequant
  ranges now owned by B1), B4 (cvec fused section), B5 (kv_q8 FP16 section).
- Findings F1–F8 acknowledged in the parent plan's B1 row.
