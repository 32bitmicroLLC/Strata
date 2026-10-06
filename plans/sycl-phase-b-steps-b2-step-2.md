# B2 step 3.2 — detailed execution plan (`plans/sycl-phase-b-steps-b2-step-2.md`)

Parent: `plans/sycl-phase-b-steps-b2-steps.md` (Step 3.2 section) and the batch
audit `plans/sycl-phase-b-steps-b2.md`.  Predecessor: **3.1 done** (its
`submit_s2_gemv` entry is what F16's driver substitution calls).

Status: pending.

## Scope

- **Kernel TU (new)**: `poc/sycl/kernels/s2_gemv_q8.cpp` — CUDA source
  `src/kernels/cuda/s2_gemv_q8.cu` @ **`386f882`** (96 lines: 1 kernel, 1 host
  entry).  The Q8_0-activation S2 GEMV (the round-186 activation contract:
  ggml quantizes the activation to Q8_0 before the dot; FP16 activations give
  different numbers for different experts of the same token — ~1 % relative,
  against the 1e-3 FP16-path tolerance).
- **Driver (new)**: `poc/sycl/drivers/s2_gemv_q8_parity.cpp` — CUDA source
  `src/kernels/s2_gemv_q8_parity.cpp` @ **`833e751`** (181 lines, three
  checks).
- **CMake**: `k_s2_gemv_q8` (STATIC) + `k_s2_gemv_q8_parity` (exe + test).
- No new B0 intrinsics: `__half2float`/`__ushort_as_half` are the existing
  intrinsics.hpp macros (lines 536/540), and `__syncthreads` is its macro
  (line 511, `it.barrier()`).  The only B1 dependency is
  `submit_quantize_q8_0` (the driver quantizes its fixture on-device).
- **Public entry** (3.8's `shared_expert` host path will call it):
  `strata::kernels::submit_s2_gemv_q8(sycl::queue& q, const uint8_t* act,
  const uint8_t* codes, const float* scales, float* y, int64_t n_in,
  int64_t n_out, int threads_per_row) -> sycl::event`.

## Kernel TU plan (`poc/sycl/kernels/s2_gemv_q8.cpp`)

Mirror blocks (source @ `386f882`):

| # | CUDA lines | content |
|---|---|---|
| 1 | 30:31 | `constexpr int QK_S2 = 64;` / `constexpr int QK8_0 = 32;` |
| 2 | 38:38 | `if (o >= n_out) return;` (unreachable with the exact grid — kept as mirrored) |
| 3 | 41:44 | `n_quads`, `c` (one code byte per quad, trailing comment included), `s` |
| 4 | 45:65 | the whole accumulation loop: four accumulators, the per-quad body (`byte`, `d` with its shift comment, the activation quad read, `dbits`, `dx` through `__half2float(__ushort_as_half(…))`, `xq`, `off`), the four `w_k` weight decodes, the four `a_k += w_k * (xq[off+k] * dx)` terms |
| 5 | 66:72 | the block reduction: `partial[tid] = (a0 + a1) + (a2 + a3);`, `__syncthreads();`, the halving tree loop with its barrier, `if (tid == 0) y[o] = partial[0];` |
| 6 | 80:80 | `if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {` (host check) |
| 7 | 84:84 | `const size_t smem = (size_t) threads_per_row * sizeof(float);` |

Glue (each commented with the CUDA line it replaces):

1. **Signature 33–35** → plain function `s2_gemv_q8_kernel_body(const uint8_t*
   act, const uint8_t* codes, const float* scales, float* y, long long n_in,
   long long n_out, int threads_per_row, sycl::nd_item<1> it)` (or inline in
   the submit lambda — the B1 cvec/quantize_act pattern; the kernel body
   stays a named function so the host entry mirrors cleanly).
2. **L36** `extern __shared__ float partial[];` → handler-constructed
   `sycl::local_accessor<float, 1> partial(sycl::range<1>(threads_per_row),
   h);` — **named `partial`** so the mirrored references read it unchanged
   (the cvec `part` precedent).
3. **L37** `o = blockIdx.x` → `const long long o = (long long)
   it.get_global_id(0);` (one work-group per row — no `threadIdx` component).
4. **L39** `tid = threadIdx.x` → `const int tid = (int) it.get_local_id(0);`.
5. **L67/70** `__syncthreads();` stay IN the mirror — the intrinsics.hpp
   macro expands them to `it.barrier()` (the nd_item is named `it`, per
   convention).
6. **L53** stays IN the mirror — `__half2float(__ushort_as_half(dbits))`
   expands through the intrinsics.hpp macros to the B0-probed
   `sycl::half` widening (bit-exact).
7. **L48** `s[q >> 4]` stays verbatim (the comment documents it as
   `(q*4) >> 6`).
8. **L79** `if (n_in <= 0 || n_out <= 0) return;` → empty-case submit
   (`return q.submit([](sycl::handler&) {});`, T4/B1 convention).
9. **L81–82** (`fprintf` + `exit(1)`) → `throw std::runtime_error(...)` with
   the same message (`"s2_gemv_q8: n_in … must be a multiple of 64"` — the
   mirrored check tests both moduli; 64 subsumes 32, and the CUDA message
   prints QK_S2).
10. **L85–87** `<<<n_out, tpr, smem, stream>>>` → `q.submit` handler with
    `h.parallel_for(sycl::nd_range<1>((size_t) n_out * tpr, (size_t) tpr), …)`
    — exact grid, no padding needed; the local accessor built in the handler
    (the T3 finding: accessors must be constructed in the handler, not via
    `strata_launch`).
11. **L88–93** `cudaGetLastError` + conditional `cudaDeviceSynchronize` → glue
    comment: the returned event IS the synchronisation point; a failed
    submit throws from `q.submit`.  (The CUDA wrapper syncs only when
    `stream == nullptr`; the SYCL entry never syncs — caller-owned waits,
    documented in the header comment like 3.1's.)
12. **Work-group strictness (written contract difference)**: the CUDA
    wrapper has no `threads_per_row` validation, but the mirrored tree
    reduction is correct only for power-of-two `tpr`; icpx's `submit`
    throws for a non-power-of-two local size.  The SYCL entry is therefore
    STRICTER than CUDA (it rejects where CUDA would miscompute silently);
    the engine always passes 32.  Note in the header comment.
13. Includes: `strata/kernels/s2_gemv_q8.hpp`, `sycl_compat/intrinsics.hpp`,
    `<sycl/sycl.hpp>`, `<cstddef>`, `<stdexcept>`, `<cstdint>`.  Drops the
    CUDA runtime/fp16 headers.  No G4 empty-qualifier macros needed
    (`__restrict__` appears only in the glue signature line).

## Driver plan (`poc/sycl/drivers/s2_gemv_q8_parity.cpp`)

Header comment (B1 template): CUDA source + SHA, the three checks, contract
differences, and the two adaptation notes below.  Includes:
`sycl_compat/test_ctx.hpp`, `<cmath>`, `<cstdint>`, `<cstdio>`, `<cstdlib>`,
`<cstring>`, `<random>`, `<string>`, `<vector>` (`std::string` is used at
mirrored line 39 — the CUDA driver got it transitively; name it).  No
strata kernel header is needed: both cross-TU entries are forward-declared
(B1 convention, as in `quantize_act_parity.cpp`):

```cpp
namespace strata::kernels {
sycl::event submit_s2_gemv_q8(sycl::queue& q, const uint8_t* act, const uint8_t* codes,
                             const float* scales, float* y, int64_t n_in, int64_t n_out,
                             int threads_per_row);
sycl::event submit_s2_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* codes,
                           const float* scales, float* y, int64_t n_in, int64_t n_out);
sycl::event submit_quantize_q8_0(sycl::queue& q, const float* x, uint8_t* blocks, int64_t n);
}
```

Mirror blocks (source @ `833e751`):

| # | CUDA lines | content |
|---|---|---|
| 1 | 37:41 | arg parsing (`--selftest` only) |
| 2 | 43:53 | the fixture: `n_in 2560, n_out 64, tpr 32`, `mt19937 rng(31)`, `normal_distribution<float> gauss(0,1)`, the fp32 activation `x`, the S2 `codes` plane, the fp32 weight scales in [0.001, 0.011) |
| 3 | 73:73 | `std::vector<uint8_t> h_act((size_t) (n_in / 32) * 34);` |
| 4 | 75:101 | `act_val` (the pure-integer fp16→f32 decode of a Q8_0 block element, the no-CUDA-header comment and all) + the host reference loop accumulating `acc += (code-1) * d * act_val(i)` in the kernel's order |
| 5 | 103:103 | `std::vector<float> got((size_t) n_out);` |
| 6 | 106:123 | check-1 verdict: `bad`/`worst`/`sum_abs`, the denominator comment (rounds 169/189), the 1e-4 gate with its measured-1.479e-05 justification comment, the result printf |
| 7 | 125:154 | check-2 setup: the "gap is real" comment, the real `f32_to_f16` lambda (integer-exact, with the 2770 %-fixture post-mortem comment), the fp16-rounded activation `hx` fill |
| 8 | 160:160 | `std::vector<float> y16((size_t) n_out);` |
| 9 | 163:179 | the gap metric: `diff` over rows, `rel_gap` against `sum_abs`, the printf, the `rel_gap < 1e-4` failure branch (the "cannot see the difference it exists to measure" stderr), the summary printf, `--selftest` OK |

Glue:

- **`check(cudaError_t)` (27–32)**: not mirrored (SYCL throws).
- **L55–68** allocations/copies → `c.device_alloc<…>` + `c.q.memcpy`
  (byte-count overload) for `d_x`, `d_act` (n_in/32 × 34), `d_codes`,
  `d_scales`, `d_y_q8`.
- **L70** `quantize_q8_0(d_x, d_act, n_in, nullptr)` →
  `strata::kernels::submit_quantize_q8_0(c.q, d_x, d_act, n_in);
  c.wait_and_throw();` (B1 entry drops the stream arg).
- **L74** copy back `h_act` → `c.q.memcpy(h_act.data(), d_act, h_act.size());
  c.wait_and_throw();`
- **L102** `s2_gemv_q8(d_act, …, tpr, nullptr)` →
  `strata::kernels::submit_s2_gemv_q8(c.q, d_act, d_codes, d_scales, d_y_q8,
  n_in, n_out, tpr); c.wait_and_throw();`
- **L104** copy back `got` → memcpy + wait.
- **L155–159 — the F16 substitution (the one documented deviation from the
  CUDA driver)**: lines 158–159 are
  `SForm form{2, -1, 64, Affine, false};` +
  `s_gemv_split(d_hx, d_codes, d_scales, nullptr, d_y_fp16, n_in, n_out,
  form, tpr)` — `s_gemv_split` is **step 3.5's** kernel, not yet ported.
  The SYCL driver drives the fp16 comparison through **3.1's
  `submit_s2_gemv`** (the fp16-activation S2 GEMV, same weight planes),
  replacing lines 158–159 with:
  `strata::kernels::submit_s2_gemv(c.q, d_hx, d_codes, d_scales, d_y_fp16,
  n_in, n_out); c.wait_and_throw();` and a comment naming CUDA lines
  158–159 and the reason (both are fp16-activation GEMVs over the same S2
  bytes; the activation-format gap dwarfs the summation-order difference).
  The `SForm` line is dropped (no longer referenced).  Lines 155–157
  (`d_hx` alloc/copy) are plain glue.
- **L161** copy back `y16` → memcpy + wait.
- **L180** `return 0;` + the four frees → `c.free_device(…)` ×4 then
  `return 0`.

**Reference host-arithmetic note (written margin, required by B1's host-f32
rule)**: the mirrored reference (block 4) does plain host f32
`acc += (code-1) * d * act_val(i)`.  The weight scales lie in [0.001,
0.011) and the activation values are `xq · dx` with `dx` the fp16
block-scale of a 32-sample Gaussian block — the smallest nonzero term is
O(1e-8) in the degenerate case, ~37 orders above the f32 subnormal band
(1.4e-45), and no running sum (magnitude O(1..10)) can cancel into it.
  icpx host contraction of the triple product is bounded by the driver's
  OWN gate of 1e-4 (the mirrored comment: measured worst 1.479e-05, gate is
  ten times looser).  If a future fixture change makes small magnitudes
  reachable, move the reference to the pure-C integer-exact FMA — do not
  widen the gate first (stop condition 2).

## CMake delta (`poc/sycl/CMakeLists.txt`)

```cmake
# B2 step 3.2 (plans/sycl-phase-b-steps-b2-step-2.md): the Q8_0-activation
# S2 GEMV (s2_gemv_q8.cu @ 386f882).  The driver links k_s2_gemv for the F16
# fp16 comparison path and k_quantize_act for the on-device Q8_0 quantizer.
add_library(k_s2_gemv_q8 STATIC kernels/s2_gemv_q8.cpp)
add_executable(k_s2_gemv_q8_parity drivers/s2_gemv_q8_parity.cpp)
target_link_libraries(k_s2_gemv_q8_parity PRIVATE k_s2_gemv_q8 k_s2_gemv k_quantize_act)
add_test(NAME k_s2_gemv_q8_parity COMMAND k_s2_gemv_q8_parity --selftest)
```

plus `k_s2_gemv_q8_parity` in the `foreach(t …)` rpath/`-fsycl` list and a
`set_tests_properties(k_s2_gemv_q8_parity PROPERTIES TIMEOUT 120)` line.
Suite goes to **37/37**.  (ctest regex note: `-R "k_s2_gemv"` also matches
3.1's test — run `-R "k_s2_gemv_q8"` for this one.)

## The three checks (what green means)

1. **Q8_0-activation GEMV vs host reference**: 64 rows, gate rel ≤ 1e-4
   (measured 1.479e-05 in the CUDA run — FMA contraction over 2560 terms,
   not structural); the `mean |ref|` print guards the scale.
2. **The gap is real**: same S2 weights, fp16-rounded activation (via 3.1's
   `submit_s2_gemv`) must differ from the Q8_0 path by rel_gap **> 1e-4** of
   the reference magnitude (~1 % in the CUDA run); below that the test
   increments `bad` with the "cannot see the difference" stderr — the guard
   against a q8 path that silently used fp16.
3. **Measured gap printed** — the report quotes the number the test
   produced, not one from the CUDA state file.

## Mutation test

Big-endian activation-scale decode in the kernel TU (mirrored block 4, CUDA
line 52): `dbits = (uint16_t) (blk[1] | (blk[0] << 8))` instead of
`(blk[0] | (blk[1] << 8))`.  Expected red: check 1's `bad > 0` (every block
scale is mis-decoded — every one of the 800 blocks differs, so this is not
fixture-luck), the worst rel print, and `s2_gemv_q8: N failures` with exit 1.
Record the red output, revert, rebuild, re-run green.  (The checker also
goes red while the mutation is in — expected; the revert restores both.)

## Verification (in order)

1. `cd poc/sycl/build && cmake --build . -j` — clean.
2. `poc/sycl/check_mirrors.sh` — exit 0; 7 kernel blocks + 9 driver blocks.
3. `env -u LD_LIBRARY_PATH ctest -R "k_s2_gemv_q8" --output-on-failure` —
   green; record the measured worst rel, the mean |ref|, and the printed
   gap % for report §B2.2.
4. Full suite `env -u LD_LIBRARY_PATH ctest --output-on-failure` — **37/37**;
   rerun transients serially before recording.
5. `git status --porcelain src/` — empty.

## Stop conditions that can fire

- **2** — measured worst rel near/above the 1e-4 gate, or the gap below
  1e-4: investigate before touching any number; no tolerance widening.
- **3** — a mirrored line uncompilable under icpx: isolate, blocklist,
  continue (not expected — the block uses only the proven B0 macro glue).
- **7** — timeout / `DEVICE_LOST`: rerun before recording.
- Stop condition 5 does not apply (no wide-type reinterpret; the
  `int8_t*` cast at CUDA line 54 is a byte-pointer identity, and the
  half widening is the B0-probed macro path).

## Done when (3.2)

1. `poc/sycl/kernels/s2_gemv_q8.cpp` + `poc/sycl/drivers/s2_gemv_q8_parity.cpp`
   exist; `check_mirrors.sh` exit 0 with all 16 blocks listed (7 + 9).
2. ctest `k_s2_gemv_q8_parity --selftest` green; full suite 37/37.
3. Mutation red-verified and reverted; measured numbers (worst rel, gap %)
   in `plans/sycl-phase-b-report.md` §B2.2.
4. `src/` untouched; `k_s2_gemv_q8_parity` TIMEOUT 120 in CMake; the F16
   substitution comment present in the driver.
5. Status lines updated in `plans/sycl-phase-b-steps-b2.md` and
   `plans/sycl-phase-b-steps-b2-steps.md` ("3.2 done").

## Effort

~0.5 d (port + driver + mutation + closeout).
