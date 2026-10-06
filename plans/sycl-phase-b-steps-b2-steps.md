# B2 per-step execution plan (`plans/sycl-phase-b-steps-b2-steps.md`)

Parent: `plans/sycl-phase-b-steps-b2.md` — the batch audit (scope table, findings
F9–F15, glue conventions G9–G13, stop conditions, effort, Done-when). This file
is the **step-by-step execution plan**: what each step ports, what it mirrors,
what the driver checks, and what can bite. Line numbers are measured against the
current tree (commit `1a22ff3`) and are approximate at block level — exact
`SYCL-MIRROR-BEGIN …:A:B @ sha` ranges are fixed at port time by
`check_mirrors.sh`, as in B1.

Status (updated as each step lands): 3.1 done (report §B2.1); 3.2–3.11 pending.

## Ordering and dependencies

```
3.1 s2_gemv ──────────────────────────────┐
3.2 s2_gemv_q8 ───────┐                    │ (3.2's driver check 2 uses 3.1's kernel, F16)
3.3 s2_gemv_quads ────┤                    │
3.4 s2_gemv_fast ─────┼──────────┐         │
3.5 s_gemv remainder ─┤          │         │
3.6 sweep un-park ────┘ (needs 3.3+3.4+3.5)│
3.7 bf16_gemv ──────────────────────────────┤  (independent; B1 native_bf16 already done)
3.8 shared_expert ── (needs 3.2, 3.5, 3.7's header; legacy driver path needs no 3.9)
3.9 native_mmvq ────────────────────────────┤  (driver self-contained, F17)
3.10 s2_expert_grouped ─────────────────────┤  (needs only B1 quantize_q8_0/_scaled)
3.11 batch close
```

No step may start before B1 steps 2.4–2.10 are closed (done — gate open). 3.8's
*driver* needs only 3.2 + 3.5 + B1's `bf16_gemv_fp32_mmvf(_multi)`; the *port*
references 3.9's `native_mmvq`/`native_quantize_q8_1` symbols at link time, so
3.9's TU must at least exist before 3.8 links — **sequence 3.8 after 3.9's
kernels land, or link 3.8's test last inside the 3.9 step** (decided at port
time; both keep the numbered step boundaries).

## New findings from the close read (numbering continues from F15)

- **F16 — `s2_gemv_q8_parity`'s check 2 calls `s_gemv_split` (step 3.5).**  The
  CUDA driver drives the same weights by an fp16 activation through
  `s_gemv_split` (tpr 32) to measure the activation-contract gap.  The SYCL
  driver instead drives the fp16 comparison through **3.1's `s2_gemv`** — the
  fp16-activation GEMV the check actually wants — removing the forward
  dependency on 3.5.  Documented deviation; the check is an observability gate
  (gap must exceed 1e-4), and both fp16 paths give the same gap to summation
  order.
- **F17 — `mmvq_multi_parity` uses `quantize_q8_1_rows`, which lives in
  `iq_kernels.cu` (B3), not `native_mmvq.cu`.**  The two Q8_1 quantizers are
  the same algorithm (32-block warp amax/sum, `d = amax / 127.0f`,
  `q = roundf(xi / d)`, `ds = make_half2(d, sum)`; `native_mmvq.cu:140-153` vs
  `iq_kernels.cu:407-422`), so the SYCL driver substitutes
  `native_quantize_q8_1` (step 3.9).  Documented deviation; the driver's
  comparison is self-contained (multi vs single over the *same* bytes), so
  identical bytes on both sides are all the contract needs.  **B3 amendment**:
  when `iq_kernels.cpp` lands, re-point (or keep the substitution — the
  substitution is the safer default since it removes a B3 symbol from a B2
  test's link).
- **F18 — `shared_expert_multi` has no CUDA-driver coverage.**
  `shared_expert_parity.cpp` exercises only `shared_expert` (legacy path, plus
  the native *gate* toggle, which needs B1's `bf16_gemv_fp32_mmvf` and nothing
  from 3.9 — the driver always passes `native == nullptr`).  `shared_expert_multi`
  (the only caller of `native_scalar_sigmoid_multi_kernel`, `<<<1, 1..8>>>`) is
  therefore **compile-only in B2 + a trivial-oracle execution smoke**, F6-style.
- **F19 — `to_f16_kernel` (`shared_expert.cu:71-75`) is dead code** — defined,
  never launched from the TU.  Mirrored for completeness, no driver coverage.
- **F20 — the audit's `s_gemv` remainder line is wrong about the split blocks.**
  `s_gemv_q8k_split` / `s_gemv_q8_0_split` launch `<<<blocks, 256>>>` — **8
  warps per block, one warp per row** (not "block 64 / 4 warps per row").
  The ragged-row driver section (`n_out ∈ {1,7,9,61}`) is ragged by *8* for the
  same reason.  Block 256 is Arc-compatible; no padding needed beyond the
  mirrored in-kernel warp-return.
- **F21 — one more B0-class intrinsic gap for B2: `make_half2`**
  (`native_mmvq.cu:151`, inside `native_quantize_q8_1_kernel`).  Glue: a
  `make_half2` free function mapping to `floats2half2_rn` (both halves
  RN-rounded, matching CUDA).  Added to F15/G13's list.
- **F22 — `s2_expert_grouped.cu` includes `strata/kernels/dp4a.hpp`.**  Drop the
  include in the port (the B0 macro conflict that forced the same drop for
  `elementwise` in B1); `STRATA_DP4A` / `__vsubss4` come from
  `sycl_compat/intrinsics.hpp`.
- **F23 — F14's staged-smem figure for `s2_gemv_fast` is understated.**  The
  formula is `stage_x ? MAX_SHARED_HALVES * sizeof(__half) : 0 + tpr * 4`
  = 8192 + 512 = **8,704 B** max (the audit's "5,120 B" does not follow from
  `4096 halves * 2`).  Still far under the 128 KiB cap; the corrected number
  goes in report §B2's local-memory budget table.
- **F24 — `shared_expert.cu`'s `static const bool batch = []{ getenv…}()`
  host cache** (line ~180) is host-side `getenv` captured in a lambda — plain
  host code, mirrors verbatim; no device involvement.  `shared_expert_set_native_bf16`
  flips a TU-global read by `shared_expert`/`shared_expert_multi`; host state,
  no kernel capture, mirrors cleanly.

## Step 3.1 — `s2_gemv.cu` (63 lines, 1 kernel) — DETAILED PLAN

Status: done (report §B2.1; suite 36/36, checker green, mutation red-verified).  The simplest kernel in B2: no shared memory, no barriers,
no shuffles, no LUTs — it exercises only the launch/glue/idiom machinery, so
it doubles as the dry run that shakes out any leftover B0 convention before
the harder steps.

### Scope

- **Kernel TU (new)**: `poc/sycl/kernels/s2_gemv.cpp` — CUDA source
  `src/kernels/cuda/s2_gemv.cu` @ **`c1ff5cf`** (63 lines: 1 kernel, 1 host entry).
- **Driver (new)**: `poc/sycl/drivers/s2_gemv_parity.cpp` — CUDA source
  `src/kernels/s2_gemv_parity.cpp` @ **`ab69379`** (173 lines).
- **CMake**: `k_s2_gemv` (STATIC) + `k_s2_gemv_parity` (exe + test), per the
  delta below.
- No changes anywhere under `src/`; no new intrinsics needed (B0's
  `sycl::half`-as-fp16 and the launch idiom are all that is required —
  F13/F15 do not touch this step).
- **Public entry** (3.2's driver, F16, will call it):
  `strata::kernels::submit_s2_gemv(sycl::queue& q, const uint16_t* x,
  const uint8_t* codes, const float* scales, float* y, int64_t n_in,
  int64_t n_out) -> sycl::event` — returns the event, throws
  `std::runtime_error` instead of `exit(1)` (the standing B1 wrapper
  convention).

### Kernel TU plan (`poc/sycl/kernels/s2_gemv.cpp`)

Mirror blocks (marker form `SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:A:B @ c1ff5cf`),
checked by `poc/sycl/check_mirrors.sh`:

| # | CUDA lines | content |
|---|---|---|
| 1 | 17:17 | `constexpr int QK = 64;` |
| 2 | 23:23 | `if (o >= n_out) return;` (the padded-item exit) |
| 3 | 25:27 | `nb`, `c`, `s` plane pointers |
| 4 | 29:37 | `float acc = 0.0f;`, both loop headers, `d`, `cb`, `xb`, the two in-loop comment lines, and the `code` nibble extraction |
| 5 | 41:41 | `y[o] = acc;` |
| 6 | 49:49 | `if (n_in % QK != 0) {` (host check) |
| 7 | 53:54 | `const int threads = 128;` / `const long long blocks = (n_out + threads - 1) / threads;` |

Glue (each commented with the CUDA line it replaces, the `s_gemv.cpp` T4
pattern):

1. **Signature 19–21** → lambda; the five data pointers + `n_in`/`n_out`
   captured by value.
2. **L22** `o = blockIdx.x * blockDim.x + threadIdx.x` →
   `const long long o = (long long) it.get_global_id(0);`.  Launch shape: 1-D
   `nd_range<1>(items, 128)` with `items = ceil(n_out / 128) * 128` (the T2
   Level-Zero non-uniform-work-group padding; padded items die at block 2).
3. **L38** `acc += (float) (code - 1) * d * __half2float(__ushort_as_half(xb[j]));`
   → `acc += (float) (code - 1) * d * (float) xp[(size_t) b * QK + j];` with
   `const sycl::half* xp = reinterpret_cast<const sycl::half*>(x);` above the
   mirror (T4 `t4_fp16_probe` precedent: `sycl::half` is the bit-exact
   binary16 widening, the exact replacement for `__half2float(__ushort_as_half(…))`).
   `acc += a * b * c` stays verbatim inside/around block 4/5 — icpx may
   contract the last multiply-add into an FMA; device FMA is IEEE-exact
   (B1), and the driver's 1e-5 gate bounds any contraction difference, same
   class as the CUDA file's own "differ only by floating-point contraction"
   comment.
4. **L48** `if (n_in <= 0 || n_out <= 0) return;` →
   `return q.submit([](sycl::handler&) {});` (empty-case convention from
   `s_gemv.cpp`).
5. **L50–51** (`fprintf` + `exit(1)`) → `throw std::runtime_error(…)` with the
   same message (`"s2_gemv: n_in … is not a multiple of 64"`).
6. **L55** `s2_gemv_kernel<<<(unsigned) blocks, threads>>>(…)` → `q.submit` +
   `h.parallel_for(nd_range)`.  `blocks` stays in the mirrored line 53:54;
   mark it used or `(void) blocks;` with a comment (the padded item count is
   what the launch actually uses), as in `s_gemv.cpp`.
7. **L56–60** `cudaDeviceSynchronize` + error check → glue comment: the event
   return IS the synchronisation point; failed submits throw from `q.submit`.
8. Includes: `strata/kernels/s2_gemv.hpp` (the engine header, as in CUDA) +
   `<sycl/sycl.hpp>` + `<cstddef>` + `<stdexcept>`; drops
   `<cuda_fp16.h>` / `<cuda_runtime.h>` / `<cstdlib>`.

### Driver plan (`poc/sycl/drivers/s2_gemv_parity.cpp`)

Header comment follows the B1 driver template: CUDA source + SHA, what is
mirrored vs glue, contract differences (submit/event/throw), the one
adaptation note (below).  Includes: `sycl_compat/test_ctx.hpp`,
`strata/artifact/dequant.hpp` (header-inline `dequantize_q2_0` /
`fp16_to_fp32` — the B1 drivers include the real header rather than
re-mirroring the chain; the T2 `dequant_s2_parity.cpp` re-mirror is the
older precedent and not followed), `<random>`, `<vector>`, `<cmath>`,
`<cstdio>`, `<cstring>`.  Forward-declares `submit_s2_gemv` in
`namespace strata::kernels` (no include/ header exists for the SYCL entry yet,
the standing B1 note).

Mirror blocks (source @ `ab69379`):

| # | CUDA lines | content |
|---|---|---|
| 1 | 26:26 | `constexpr int QK = 64;` |
| 2 | 30:45 | the anti-vacuity comment + the 25-pattern `kScales` fp16 table (keep VERBATIM — the full-mantissa entries are what make the tolerance load-bearing) |
| 3 | 54:68 | arg parsing (`--selftest`, `--n-in`, `--n-out`, `--tol`, usage, `n_in % QK` guard, `nb`) |
| 4 | 71:84 | fixture: `mt19937 rng(999)`, the fp16 activation `x` drawn from `kScales` (finite/exact — the NaN argument comment comes with it), the S2 `codes`/`scales` planes for [n_in, n_out] |
| 5 | 89:112 | the CPU reference: per row, rebuild the 18-byte RAW Q2_0 block, re-narrow `d` back to its fp16 pattern (the widening-reversal lambda), `strata::dequantize_q2_0`, and accumulate `acc += dec[j] * fp16_to_fp32(x[…])` in the kernel's order |
| 6 | 132:141 | verdict loop: `bad`/`first_bad`/`worst` with the NaN-safe `!(rel <= worst)` / `!(rel <= tol)` comparisons |
| 7 | 145:166 | NON-VACUITY comment + `lo`/`hi`/`spread` (spread ≤ 0 ⇒ `VACUOUS` + return 1), the printf lines, first-bad-row print, the "agrees with the scalar decode…" line, and the `--selftest` OK line |

Glue (the CUDA memory/sync machinery, per the B1 idiom —
`check(cudaError_t)` is NOT mirrored, SYCL throws):

- **L115–125** `cudaMalloc`/`cudaMemcpy` ×4 → `c.device_alloc<…>` +
  `c.q.memcpy` (the 3-arg byte-count overload).
- **L127** `strata::kernels::s2_gemv(d_x, …)` →
  `strata::kernels::submit_s2_gemv(c.q, d_x, d_codes, d_scales, d_y, n_in,
  n_out); c.wait_and_throw();`
- **L129–131** copy back + `check` → `c.q.memcpy(got.data(), d_y, …);
  c.wait_and_throw();`
- **L168–171** `cudaFree` ×4 → `c.free_device(…)` ×4.

**Reference host-arithmetic note (written margin, required by B1's host-f32
rule)**: the mirrored reference loop (block 5) does plain host f32
`acc += dec[j] * fp16_to_fp32(x[…])`.  B1's findings forbid host f32
arithmetic only where subnormal magnitudes are reachable (icpx host -O2
flushes them; `fmaf` asm is miscompiled).  Here the minimum nonzero
magnitude of any term is 2^-10 × 2^-10 = 2^-20 ≈ 1e-6 (every `kScales` entry
has |value| ≥ 2^-10, the smallest pattern 0x1400), and the running sums stay
far from the f32 subnormal band (1.4e-45) — the flush cannot fire, and any
icpx host contraction is bounded by the same 1e-5 gate that bounds the device
kernel's own contraction.  If a future fixture change makes small magnitudes
reachable, the reference moves to the pure-C integer-exact FMA — do not
widen the tolerance first (stop condition 2).

### CMake delta (`poc/sycl/CMakeLists.txt`)

```cmake
# B2 step 3.1: the P2.S2 GEMV (plans/sycl-phase-b-steps-b2-steps.md)
add_library(k_s2_gemv STATIC kernels/s2_gemv.cpp)
add_executable(k_s2_gemv_parity drivers/s2_gemv_parity.cpp)
target_link_libraries(k_s2_gemv_parity PRIVATE k_s2_gemv)
add_test(NAME k_s2_gemv_parity COMMAND k_s2_gemv_parity --selftest)
```

plus `k_s2_gemv_parity` in the `foreach(t …)` rpath list (~line 240) and in
the `set_tests_properties(… PROPERTIES TIMEOUT 120)` list (~line 259).
Link is self-contained: the driver's reference chain is header-only
(`strata/artifact/dequant.hpp`), so `k_s2_gemv_parity` links only
`k_s2_gemv`.  Note for ctest: the regex `k_s_gemv` does NOT match
`k_s2_gemv_parity` (the `2` breaks it) — run this test by exact name or
`-R "k_s2_gemv"`.

### Mutation test

In `poc/sycl/kernels/s2_gemv.cpp`, change the mirrored `code` extraction
(block 4, L37) from LSB-first to MSB-first —
`(cb[j >> 2] >> ((3 - (j & 3)) * 2)) & 0x03` — rebuild, run
`k_s2_gemv_parity --selftest`: expect a large `bad` count (nibble order is
wrong for 3 of every 4 codes in each byte), a printed first-bad row with
ref/got differing by O(1) relative, exit 1.  Record the red output in the
report, revert, rebuild, re-run green.  (This mutation is red *even if*
the fixture were unlucky — 640 rows × 40 blocks, so it is not dependent on
fixture spread.)

### Verification (in order)

1. `cd poc/sycl/build && cmake --build . -j` — clean (warnings noted, none
   expected: no narrowing, no unused variables).
2. `poc/sycl/check_mirrors.sh` — exit 0; the output lists all 7 TU blocks and
   all 7 driver blocks.
3. `env -u LD_LIBRARY_PATH ctest -R "k_s2_gemv" --output-on-failure` — green;
   the driver prints the worst-rel line (expect ≪ 1e-5; record the measured
   number in report §B2.1) and the reference spread line (expect a wide
   range, the full-mantissa scales make it so).
4. Full suite `env -u LD_LIBRARY_PATH ctest --output-on-failure` — **36/36**
   (35 + 1 new).  A transient failure (the standing `t0_bw_gate`
   bandwidth-threshold example) is rerun serially before recording.
5. `git status --porcelain src/` — empty.

### Stop conditions that can fire in this step

- **2** — measured worst-rel anywhere near the 1e-5 gate (or above it):
  investigate contraction/fixture issues; never widen the tolerance to close
  the gap.  If the *host reference itself* were the wrong side (e.g. an
  icpx host contraction beyond expectation), the fix is the pure-C
  integer-exact reference, documented as a finding — not a tolerance change.
- **3** — any mirrored line uncompilable under icpx without touching it:
  isolate, record in the blocklist, continue (not expected: this kernel uses
  no exotic intrinsic).
- **7** — timeout / `DEVICE_LOST`: rerun before recording.
- Stop condition 5 (wide-type reinterprets) does NOT apply: the only
  reinterpret is the 2-byte `sycl::half*` cast, proven in T4.

### Done when (3.1)

1. `poc/sycl/kernels/s2_gemv.cpp` and `poc/sycl/drivers/s2_gemv_parity.cpp`
   exist; `check_mirrors.sh` exit 0 with all 14 blocks listed (7 + 7).
2. ctest `k_s2_gemv_parity --selftest` green; full suite 36/36 under
   `env -u LD_LIBRARY_PATH`.
3. Mutation test red-verified and reverted; the red output and the measured
   worst-rel + spread numbers are in `plans/sycl-phase-b-report.md` §B2.1.
4. `src/` untouched; `k_s2_gemv_parity` TIMEOUT 120 in CMake.
5. The steps file's status line updated: "3.1 done".

### Effort

~0.5 d (port + driver + mutation + closeout).

## Step 3.2 — `s2_gemv_q8.cu` (96 lines, 1 kernel)

Detailed execution plan: `plans/sycl-phase-b-steps-b2-step-2.md` (written;
status pending).

- **Files**: `poc/sycl/kernels/s2_gemv_q8.cpp`; driver mirrors
  `src/kernels/s2_gemv_q8_parity.cpp`.
- **Mirror**: constants 30–31, `s2_gemv_q8_kernel` 33–69, wrapper 77–96.
  One block per output row, block `tpr`, **dynamic smem `tpr * 4`** → G9
  (runtime local range; `grid = n_out` exact, no grid padding).  `__syncthreads`
  ×2 → `it.barrier()`.
- **Driver checks** (mirrored, n_in 2560, n_out 64, tpr 32):
  1. Q8_0-activation GEMV vs the host reference that decodes the *same device*
     Q8_0 bytes (produced by B1's `quantize_q8_0`); rel gate **1e-4** (measured
     1.479e-5 in the CUDA run — FMA contraction over 2560 terms).  The host
     `f32_to_f16` / fp16-decode helpers are pure C (mirror verbatim — they are
     integer/bit manipulation, not f32 arithmetic).
  2. **F16**: the fp16 rival path through 3.1's `s2_gemv` (not
     `s_gemv_split`); the fp16-vs-q8 gap must exceed 1e-4 relative of the
     reference magnitude (the CUDA run measures ~1 %).
  3. Measured gap printed (the state-file number must come from the test).
- **Mutation**: decode the act scale big-endian (`blk[1]` low) → check 1 red.
- **ctest**: `k_s2_gemv_q8_parity --selftest`, TIMEOUT 120.
- **Effort**: ~0.5 d.

## Step 3.3 — `s2_gemv_quads.cu` (93 lines, 1 kernel)

- **File**: `poc/sycl/kernels/s2_gemv_quads.cpp`.  No dedicated parity driver —
  the 3.6 sweep is its coverage (rel ≤ 1e-4 vs the naive reference, per
  configuration).
- **Mirror**: constants 30–31, kernel 33–70, wrapper 76–93.  Same dynamic-smem
  G9 shape as 3.2.  **Alignment**: `*reinterpret_cast<const uint2*>(x + q*4)`
  needs 8-byte alignment of `x`; USM allocations are ≥ 64-byte aligned, so the
  CUDA comment's 256-byte alignment is satisfied a fortiori (note in the
  report).  `__low2float`/`__high2float` + `__half2` reinterpret (B0 glue).
- **Mutation**: swap `w2`/`w3` (bytes bits 4–7 onto lanes 0–1) → sweep rel red.
- **Effort**: ~0.25 d.

## Step 3.4 — `s2_gemv_fast.cu` (153 lines, 1 kernel template × 2)

- **File**: `poc/sycl/kernels/s2_gemv_fast.cpp`.  Sweep-covered like 3.3.
- **Mirror**: `MAX_SHARED_HALVES` (L41), kernel 69–116 (both `STAGE_X`
  instantiations), wrapper 124–153.  Dynamic smem per F23: staged 8,704 B max,
  unstaged `tpr * 4`; `__syncthreads` ×3 (staged copy, tree).
- **Glue (F13/G11a)**: `__constant__ float c_codes[256][4]` is runtime-filled by
  `ensure_lut()` (`cudaMemcpyToSymbol`) → an `alignas(16)` USM array behind a
  wrapper struct with row-pointer `operator[]`, so the mirrored
  `*reinterpret_cast<const float4*>(&c_codes[byte][0])` compiles verbatim and
  stays 16-byte aligned.  `ensure_lut()` becomes glue taking `sycl::queue&`;
  the per-device `g_lut_ready[64]` collapses to one process-wide flag
  (single-device port, documented).
- **Mutation**: read the LUT before `ensure_lut` fills it (zeros) → all rows 0
  vs reference, sweep red.
- **Effort**: ~0.5 d (the LUT alignment is the risk — stop condition 5 if icpx
  rejects the `float4` reinterpret on Arc).

## Step 3.5 — `s_gemv.cu` remainder (F9: ~500 lines, 3 kernels)

- **File**: extend `poc/sycl/kernels/s_gemv.cpp` (same CUDA source → same port
  file).  Driver mirrors `src/kernels/s_gemv_q8k_parity.cpp`.
- **Already ported (Phase A)**: `kIq4Nl` constant (mirror block 33:33 — reuse,
  it is a compile-time constant under the G4 convention), `decode_tbl`
  (45:46), the naive `s_gemv_kernel` + `s_gemv` host entry.
- **Mirror now**: `q8k_at`/`q8_0_at` (80–94; `__ldg` is B0 glue),
  `s_gemv_q8k_kernel` 96–142 (block 128, static `__shared__ signed char
  s_iq4nl[16]` loaded **before** the early return — barrier-safe as mirrored),
  `s_gemv_q8_split_kernel` 143–300 (block 256 = 8 warps, one warp per row, F20;
  `__shfl_down_sync` 16…1 tree over 16 accumulators; static `s_iq4nl`),
  `s_gemv_split_kernel` (~383–455; block `tpr`, dynamic smem, `__half22float2`
  ×2 → **G13/F15** glue), host: `s_gemv_split_impl` + `s_gemv_split` +
  `s_gemv_split_async` (the `_async` is stream-capture-legal — maps to a
  `void* stream` argument over `ctx.q`, no sync), `q8k_form_ok`, `finish`,
  `s_gemv_q8k`, `s_gemv_q8k_split`, `s_gemv_q8_0_split`.
- **Driver checks** (mirrored; forms from the real manifest: IQ4_XS S4 g32
  no-offset, Q4_K S4 g32 offset, Q5_K S8 g32 offset):
  1. naive + warp kernels vs the double host reference over the **same device**
     Q8_K bytes (from B1's `quantize_q8_K`) — gate **1e-5** (f32-vs-f64 floor,
     not structural).  This is why the F8 division gap does not fire: both
     sides dequantise the same bytes, so the shared `d` cancels.
  2b. the Q8_0 activation loader (`s_gemv_q8_0_split`) over the same weight
     planes — gate 1e-5; this is what exists for `ffn_down_shexp` (n_in 640,
     structurally not Q8_K).
  3. Q8_K-vs-FP16 activation gap **must exceed 0.1 %** (the contract decision's
     K-quant case; the fp16 side uses Phase A's `s_gemv`).
  4. **Ragged rows**: `n_out ∈ {1, 7, 9, 61}`, a 64-word `0x7FC0DEAD` guard band
     beyond `n_out` must come back untouched, rel ≤ 1e-5, zero guard words
     written — the direct check that surplus warps neither hang the block nor
     write.  This exercises the barrier-before-return ordering that the B1
     cvec finding (shfl join = 32-wide spin barrier) made safe for the
     warp-shuffle tail.
- **Mutation**: drop the inter-lane shuffle combine in
  `s_gemv_q8_split_kernel` → the warp-vs-reference rel goes red.
- **ctest**: `k_s_gemv_q8k_parity --selftest`, TIMEOUT 120.
- **Effort**: ~1.5 d.

## Step 3.6 — un-park the sweep in `poc/sycl/drivers/s_gemv_parity.cpp`

- **Scope**: port the parked CUDA block (`src/kernels/s_gemv_parity.cpp`
  356–464) into the existing SYCL driver: after the naive reference, for
  `tpr ∈ {32, 64, 128, 256}` — time `s_gemv_split` and check it against the
  naive reference (rel ≤ 1e-4); for S2 forms additionally time + check
  `s2_gemv_quads` (`t2 ∈ {32, 64, 128}`) and `s2_gemv_fast` (staged and global,
  `t3 ∈ {32, 64, 128}`) against the same reference.  The existing timing
  harness, fixtures, and both bench shapes (2560×640 and 640×2560) are already
  there; only the sweep loop and the `split_bad` accumulation come back.
- **Glue note**: replace the driver's "parked" comment blocks (currently the
  glue at SYCL-driver lines ~336 and ~582) with mirror blocks over CUDA lines
  356–464; the `check_mirrors.sh` entry for the unparked blocks must go green.
- **No new ctest entry** — `k_s_gemv_parity` (selftest) and `k_s_gemv_bench`
  (TIMEOUT 300, already registered) are re-armed; the bench now produces the
  **perf ledger** numbers: naive/split/quads/fast per tpr at both expert
  shapes vs Phase A's T4 baseline (4.6–16.9 G weights/s).  Timing is host
  `steady_clock` per iteration, as in CUDA.
- **Dependency**: 3.3 + 3.4 + 3.5.
- **Effort**: ~0.5 d.

## Step 3.7 — `bf16_gemv.cu` (141 lines, 3 kernels)

- **Files**: `poc/sycl/kernels/bf16_gemv.cpp`; driver mirrors
  `src/kernels/bf16_gemv_parity.cpp`.
- **Mirror**: `THREADS` (18), `bf16_gemv_naive_kernel` 19–30,
  `bf16_gemv_warp_kernel` 32–46 (`__shfl_down_sync` 16…1 tree — all 32 lanes
  participate; the warp-uniform return sits before it),
  `bf16_gemv_split_kernel` 48–76 (dynamic smem `tpr * 4`, G9), `finish`
  78–91, `bf16_gemv` 93–115 (threshold `n_out >= 64` → warp, else naive),
  `bf16_gemv_split` 117–141 (`tpr == 32` → warp path; power-of-two validation).
- **Driver checks** (mirrored; real shapes 2560×48, 2560×128, 2560×512,
  2560×2560; activations and weights rounded to bf16 first — the contract):
  1. naive / warp / split-256 each vs the double host reference over the same
     bf16 bits — gate **1e-5** (products are exact; only summation order
     differs).
  2. naive-vs-split agreement is implicit in check 1.
  3. **Activation contract observable**: the rival fp16-rounded activation must
     differ by > 1e-4 relative (the pre-contract engine reading).
  Conversions come from the host headers `bf16_bits.hpp` / `f16_bits.hpp`
  (pure headers, includable) — no private driver copies.
- **Mutation**: reverse the shfl offset order (1,2,4,8,16 instead of 16,…,1) →
  the naive-vs-split agreement breaks.
- **ctest**: `k_bf16_gemv_parity --selftest`, TIMEOUT 120.
- **Effort**: ~0.5 d.

## Step 3.8 — `shared_expert.cu` (346 lines, 9 kernels)

- **Files**: `poc/sycl/kernels/shared_expert.cpp`; driver mirrors
  `src/kernels/shared_expert_parity.cpp` (482 lines).
- **Mirror (kernels)**: `swiglu_kernel` 54–60 (**double** `exp` — IEEE-exact on
  Arc per the P2b precedent), `native_swiglu_kernel` 62–69 (`__fdividef` +
  `__expf` ×2 — B0-measured 2-ulp/64-ulp gaps, write-path only),
  `to_f16_kernel` 71–75 (dead, F19), `warp_sum_d` 104–107 (the first
  **double-valued** `shfl32` use — B0's 8-byte path; **stop condition 4**: run
  the driver's native-gate section first and record the check),
  `scalar_gate_kernel` 109–126 (`<<<1, 256>>>`, static `__shared__ double
  scratch[8]`, double accumulate), `scale_kernel` 128–131,
  `native_scalar_sigmoid_kernel` 133–136 (**`<<<1, 1>>>`** — sub-16 work-group),
  `native_scalar_sigmoid_multi_kernel` 138–141 (**`<<<1, 1..8>>>`**),
  `moe_combine_kernel` 143–157 (double accumulation, block 128),
  `scale_rows_kernel` 162–167 (**2-D grid** `dim3(ceil(n_embd/128), n_tok)` →
  G1 flatten).
- **Mirror (host)**: `shared_expert_set_native_bf16` 159, `shared_expert_multi`
  169–200 (F18 — compile-only + oracle smoke; the only caller of the
  1..8-thread sigmoid), `shared_expert_scratch_bytes` 201–207,
  `shared_expert` 209–324 (scratch carved from the caller's buffer — no token
  allocations; the `gemv` lambda dispatches to 3.2's `s2_gemv_q8` for
  `code_bits == 2` and to 3.5's `s_gemv_q8k_split`/`s_gemv_q8_0_split` by
  `act_kind`; the `static const bool batch` getenv cache mirrors verbatim,
  F24), `moe_combine` 325–346.  Cross-TU calls resolve at link time:
  `k_shared_expert` links `k_s2_gemv_q8`, `k_s_gemv`, `k_native_bf16` (B1),
  `k_quantize_act` (B1), and — for the native-path symbols — `k_native_mmvq`
  (3.9; see the ordering note at the top).
- **Stop condition 1 must be probed before wiring this step**: Arc/Level-Zero
  acceptance of work-group sizes {1, 2, 4, 8} (the two native sigmoid kernels).
  If rejected: padded launch (local 16) + a glue bounds-check line *before*
  the mirrored store — a written deviation decided here, not at port time.
- **Driver checks** (mirrored; n_embd 256, n_ff 64, all-S2 Q8_0 activations):
  - silu-on-gate vs silu-on-up must differ by **> 50 %** (the documented trap);
  - structure vs host reference worst rel ≤ **1e-2** (measured 2.177e-7 in the
    CUDA run — both sides quantise the same intermediate, so it cancels);
  - the fixture's SwiGLU intermediate stays inside fp16 range (< 65504,
    non-finite) — the anti-NaN-vacuity guard, with `!(rel <= worst)`-style NaN
    comparisons kept verbatim;
  - native-gate isolation: zero gate weight → exactly half output; basis gate →
    native-vs-legacy separation > 1e-5 and rel < **2e-6** vs the host sigmoid
    (F12 margin analysis if it fails — the gate sigmoid bounds damage to
    [0,1], never a silent widening); `refused_missing` (native on, no `x_f32`)
    throws; restored default byte-identical;
  - **the CUDA stream-capture/graph-replay block is excluded as glue (G12/F12)**
    with a comment naming the CUDA line range and reason;
  - `moe_combine`: both rival readings (unweighted routed; weighted shared)
    observable > 5 %; vs reference rel ≤ 1e-6; a null `shared` observable > 5 %.
- **Mutation**: silu on UP instead of gate → the trap check + comparison red.
- **ctest**: `k_shared_expert_parity --selftest`, TIMEOUT 120.
- **Effort**: ~1 d (probe + margin analysis included).

## Step 3.9 — `native_mmvq.cu` (1,493 lines, 9 kernel templates)

- **Files**: `poc/sycl/kernels/native_mmvq.cpp`; driver mirrors
  `src/kernels/mmvq_multi_parity.cpp`.  The largest B2 file — plan the mirror
  blocks top-down: block structs + `static_assert`s (27–122),
  `warp_sum`/`warp_max` 124–138, `native_quantize_q8_1_kernel` 140–153
  (`make_half2` → F21 glue), the six dot-impl + per-type kernel pairs
  (155–781: q5, q2, q3, iq4_xs, q4, q6 — each `template<bool SmallK>` with
  **static** 3-D `__shared__ float partial[WARPS-1][ROWS][WARP]` ≤ 3×4×32×4 B →
  G10 chained-index proxy), the iq4 device constant + table lookup
  (415–434; **G11b**: namespace-scope device pointer + lambda shadow so the
  mirrored `reinterpret_cast<const uint32_t*>(iq4nl_values)` reads device
  memory), `native_small_mmvq_kernel` 738–781, `native_mmvq_multi_kernel`
  999–1039 (**4-D static** `partial[NW-1][NCOLS][ROWS][WARP]`, max
  3×8×2×32×4 = 6,144 B → G10; **2-D block** `dim3(WARP, NW)` → 1-D padded
  work-group of `WARP * NW` with glue `threadIdx.y = id/32`,
  `threadIdx.x = id%32`), host `launch_multi_n`/`launch_multi` 1045–1083,
  `validate_*`/`launch_check` 1086–1108 (throw-style — mirror cleanly;
  `launch_check` becomes a glue no-op since SYCL throws at submit),
  per-type `_mmvq`/`_f32` wrappers 1166–1425, `native_mmvq_set_multi_exact` /
  `native_mmvq_multi_exact` (host global `g_multi_exact` — host state, no
  kernel capture), `native_q8_1_bytes`, `native_quantize_q8_1`,
  `native_mmvq_supported`, `native_mmvq_weight_bytes`, `native_mmvq`.
- **F11 glue throw (loud, documented)**: the six iq cases of `native_mmvq`
  (types 16, 17, 18, 21, 22, 29 → `iq_mmvq`) and the iq case-range of
  `native_mmvq_weight_bytes` (→ `iq_row_bytes`) are replaced with a glue throw
  `"native MMVQ iq branch (B3) not linked yet"` — **excluded from the mirror
  blocks** (the checker requires the exclusion comment to sit outside any
  mirror block).  `iq_kernels.hpp` is a pure declaration header — include it for
  the other symbols.  **B3 amendment** (already in the parent's parked list):
  B3's `iq_kernels.cpp` step restores the mirrored dispatcher.  Stop condition
  8: if any B2 driver ever reaches the throw, the fixture changed shape — stop
  and re-examine B3 ordering.
- **Driver checks** (mirrored): CASES {Q4_0 t2 2048×512, Q8_0 t8 2048×512,
  Q4_K t12, Q5_K t13, Q6_K t14 2048×512, IQ4_XS t23 **4096**×512 (the
  negative-control row length), Q5_K-wide t13 4096×2048}; WIDTHS T ∈
  {1,2,3,4,5,6,8}; weights random bytes with the block fp16 scales rewritten to
  sane halves in ±[2^-10, 2^-5] (keep verbatim — NaN-canonicalisation
  argument); one shared quantization per case (**F17**:
  `native_quantize_q8_1`, not `quantize_q8_1_rows`);
  - `multi_exact` **on**: every T-column output **bitwise equal** to its
    single-column call (all outputs, all widths) — the 2-D-grid G1
    decomposition must preserve the CUDA warp/block indexing exactly;
  - `multi_exact` **off** (negative control): at T > 4 (the first width where
    `NW = 2 ≠ 4` differs from the exact layout) the control must find finite
    differences in **every** case (`powerless_cases == 0`), or the test has no
    power and fails; at T ≤ 4 the layout coincidence is reported, not
    skipped;
  - non-finite outputs count as failures (finite scales + activations ⇒
    finite outputs, by the header's precondition).
  Driver runtime: 7 cases × 7 widths incl. 4096×2048 — if a TIMEOUT 120 ctest
  entry is tight, register it at 300 (decide from the first local run).
- **Mutation**: swap the two `__byte_perm` selections in `q2_q8_dot`
  (0x5140 ↔ 0x7362) → the Q4_0 case goes red (Q4_0 is the small-K t2 path; the
  swap targets the q2/q4 byte-selection logic shared by the small kernels —
  pick the case the swap actually disturbs at port time and record it).
- **ctest**: `k_mmvq_multi_parity --selftest` (add the flag; the CUDA driver
  takes no args, so the SYCL mirror adds `--selftest` for the ctest
  convention), TIMEOUT 120 (or 300).
- **Effort**: ~2.5 d.

## Step 3.10 — `s2_expert_grouped.cu` (790 lines, 13 kernels) + **new** driver (F10)

- **File**: `poc/sycl/kernels/s2_expert_grouped.cpp`; driver
  `poc/sycl/drivers/s2_expert_grouped_parity.cpp` (**no CUDA parity driver
  exists** — F10; the `bench/micro/moe_hit_parity.cu` the .cu references is not
  in the repo).
- **Include note (F22)**: drop `#include "strata/kernels/dp4a.hpp"` (B0 macro
  conflict); `verify_kernels.hpp` stays (header-only `kVerifyMaxT = 8`, used by
  the `static_assert(GMAX >= kVerifyMaxT)`); `quantize_act.hpp` stays (B1).
- **Mirror (device)**: geometry constants 35–44 (restated Q2_0 blob geometry:
  H 2560, FF 640, QK 64, ROW_GU 640 B, ROW_D 160 B, SC_GU 40, SC_D 10, offsets
  O_*), `f16_at` 46, `row_dot_s2_q8` 62–102 (`STRATA_DP4A` chunk dots +
  fp16 group scales), `warp_sum` 103, kernels: `gu_kernel` 119,
  `swiglu_kernel` 164 (fp32 `expf`), `down_kernel` 176,
  `activation_correction_kernel` 200, `dot4` 209, `row_dot_cpu_order` 218
  (**4,1,2-ordered** shuffle reduce — stop condition 4, re-probe before this
  file if anything looks off), `cpu_order_projection_kernel<DOWN>` 243–265
  (×2; the in-file `cudaMemcpyAsync` device→device in the *host* flow maps to
  `ctx.q.memcpy` — in-order queue preserves sequencing),
  `cpu_order_swiglu_kernel` 267, `cpu_order_quantize_kernel` 277,
  `gu_grouped_kernel` 543 (**static** `__shared__ int xs_q[8][640]` (20,480 B) +
  `float xs_d[8][80]` (2,560 B) → G10 proxy; `__ballot_sync`/`__popc` B0),
  `down_grouped_kernel` 605 (`hs_q[8][160]` + `hs_d[8][20]`, 5,760 B),
  `group_resident_kernel` 657 (five 1-D arrays ~2.6 KB; `<<<1, 128>>>`),
  `hit_select_kernel` 377 (`<<<1, 32>>>`), `hit_select_multi_kernel` 397
  (`<<<1, 128>>>`, `__shared__ int warp_count[4]`), `add_hits_kernel` 420
  (2-D grid `dim3(min(ceil(n_embd/256), 8), cap)` → G1).
- **Mirror (host)**: `check()` 304–312 (glue: throw, keep call sites; no
  non-null-stream sync — keep), `moe_hit_grouped_scratch_bytes` 321, the nine
  public entries (329, 430, 437, 470, 477, 700, 709, 738, 746) — including the
  `quantize_q8_0` / `quantize_q8_0_scaled` branches (B1, linked), the 2-D-grid
  launches in `moe_grouped_s2` (`dim3(2*FF/GU_ROWS=40, cap_groups)` and
  `dim3(H/D_ROWS=40, cap_groups)`), and the `gate_up_trace` device→device copy
  glue in `moe_hit_grouped_s2_cpu_order`.
- **New driver (F10, self-contained contract)**: synthesises expert blobs in
  the documented Q2_0 geometry (the literals above), with
  `std::mt19937` fixtures; the host reference is a double-precision scalar
  transcription of `cpu/expert.cpp`'s arithmetic (code `→` code-1 in the
  integer domain, fp16 group scale, fp16/fp32 activation scale, Q8_0
  quantisation of the intermediate per `ggml_mul_mat`'s contract, `dst_index`
  row placement, gate-major layout).  Covers all nine public entries
  (10 counting both `x_scales` variants of `moe_hit_grouped_s2`):
  `moe_hit_grouped_s2` (both variants), `moe_hit_select`,
  `moe_hit_grouped_s2_dev`, `moe_hit_add`, `moe_hit_select_multi`,
  `moe_hit_grouped_s2_multi`, `moe_group_resident` + `moe_grouped_s2`, and
  `moe_hit_grouped_s2_cpu_order` (its reference is the 8-lane CPU-order loop
  the kernel transcribes).  Assertions: per-entry kernel-vs-reference at the
  documented ~6.279e-08-class tolerance (mirrored/measured, printed); the
  **grouped path is bitwise the per-entry path** (the file's central claim);
  **stop condition 6**: if the reference grows past ~400 lines, re-cut entry
  coverage (group family and cpu_order drop first) with a written note — never
  shrink the reference silently.  **Reference note**: the transcription does
  integer-exact chunk dots and double accumulation; any f32 step must use the
  pure-C integer-exact helpers (B1's icpx-host flush findings).
- **Mutation**: decode slot `i` from row-slot `i/2` (the documented
  first-version gate-interleaving bug) → worst rel ~2.2e+03, observable.
- **ctest**: `k_s2_expert_grouped_parity --selftest`, TIMEOUT 120.
- **Effort**: ~1.5 d port + ~1 d driver (inside the parent's +1 d driver
  budget — the 400-line cap is the binding constraint).

## Step 3.11 — batch closeout

Per the parent's *Done when* (suite, checker, report, parked-block
amendments).  Concrete checklist, verified in order:

1. `poc/sycl/check_mirrors.sh` exit 0 tree-wide, including the unparked
   `s_gemv_parity` sweep blocks (3.6) and the F11 exclusion comment sitting
   outside any mirror block (3.9).
2. Full suite under `env -u LD_LIBRARY_PATH`: 35 (post-B1) + 7 new
   (`k_s2_gemv_parity`, `k_s2_gemv_q8_parity`, `k_s_gemv_q8k_parity`,
   `k_bf16_gemv_parity`, `k_shared_expert_parity`, `k_mmvq_multi_parity`,
   `k_s2_expert_grouped_parity`) = **42 tests**, each TIMEOUT 120 (300 where
   noted: `k_s_gemv_bench`, possibly `k_mmvq_multi_parity`); rerun transients
   before recording (B1's `t0_bw_gate` is the standing example).
3. Mutation tests: every candidate above (3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8,
   3.9, 3.10) run, red-verified on its fixture, reverted; any skipped
   candidate carries a written reason.
4. `plans/sycl-phase-b-report.md` §B2: per-file section (1:1 mapping, glue
   lines, G9–G13 uses, local-memory budget rows — F14's four dynamic sites
   with F23's corrected staged number, G10's static arrays,
   `scalar_gate`'s `scratch[8]` — parity numbers, F9–F15 + **F16–F24**, and
   the **perf ledger**: naive/split/quads/fast at both expert shapes (2560×640,
   640×2560) vs Phase A's T4 baseline 4.6–16.9 G weights/s, from the 3.6
   un-park).
5. Parked-block amendments recorded: **B3** — restore the `native_mmvq` iq
   dispatcher (and decide the `quantize_q8_1_rows` ownership, F17) once
   `iq_kernels.cpp` lands; **Phase A** — the `s_gemv_parity` sweep un-park
   note (3.6); the parent plan's B2 row updated to complete.
6. Findings F9–F15 (existing) + F16–F24 (this file) acknowledged in the
   parent plan's B2 row and the step file's effort line, re-based to ~7.5–10 d
   (parent's 7–9 d + the F16/F17/F18 handling and the F10 driver's 400-line
   cap risk).

## Standing stop conditions (from the parent, applied at each step)

1. Sub-16 work-groups (3.8's native sigmoids) — probe {1, 2, 4, 8} on Arc
   *before* wiring 3.8; fallback is the padded launch + glue bounds-check, a
   written deviation.
2. Any mirrored driver comparison failing on a math gap not covered by a
   written P2b margin analysis — including F12's native-shared 2e-6 gate.
3. A mirrored block uncompilable without touching a mirrored line (icpx bug
   class) — isolate, record in the blocklist, continue.
4. `shfl32` behaving differently from the B0 probes in the warp-reduction
   kernels (`bf16_gemv_warp`, `s_gemv_q8_split`, `native_mmvq`'s
   `warp_sum`/`warp_max`, `s2_expert_grouped`'s `row_dot_cpu_order`, and
   3.8's first **double** `warp_sum_d`) — re-probe before proceeding with
   that file.
5. The G11 LUT alignment (F13) or any `reinterpret_cast` to a wider-aligned
   type (`float4`, `uint2`, `__half2`) rejected by icpx on Arc — re-probe the
   specific access before any workaround.
6. The F10 reference grows past ~400 lines — re-cut coverage with a written
   note.
7. `U_RESULT_ERROR_DEVICE_LOST` / ctest timeout — rerun before recording.
8. The F11 glue throw reached by any B2 driver run — stop and re-examine B3
   ordering.
