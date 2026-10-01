# T2 — detailed plan: first real kernel port (`dequant_s2`)

Parent: `plans/sycl-phase-a.md` (task T2). T0 and T1 are done (PASS —
`plans/sycl-phase-a-report.md`); the pipeline exists and is green. T2 is the
first *math* through it: the S2 decode, chosen first because it is
self-contained (no shared memory, no shuffles, no BLAS) and is "the single
hottest decode in the engine" (31.64 GiB of the artifact is Q2_0).

## What is being ported

**CUDA kernel** — `src/kernels/cuda/dequant_s2.cu` (51 lines), HEAD `6bd3e58`:

- One CUDA thread per S2 block (64 elements). Per thread: 1 scale float +
  16 code bytes in, 64 floats out. Codes are 2 bits each, 4 per byte
  LSB-first (element `j` at byte `j/4`, bits `(j%4)*2`); decode is
  `y[j] = (float)(code - 1) * d` — the `-1` in the **integer** domain, then
  one multiply. That is what makes it bit-exact (`docs/pack-format.md` §3.1).
- Grid/block: `blockDim 256`, `gridDim = ceil(n_blocks / 256)`, index bounds
  check. Naive on purpose — no vectors, no shared, no unroll (the CUDA file's
  own Phase 3 will optimize it; the parity test is the guard).
- Host wrapper `dequant_s2(...)` **synchronises before returning** and
  `exit(1)`s on a driver error.

**Parity test** — `src/kernels/dequant_s2_parity.cpp` (143 lines). Its
reference is a chain, not a second opinion: raw 18-byte Q2_0 blocks
`{ fp16 d; uint8 qs[16] }` (scale from a 14-pattern fp16 table, `qs` from
`mt19937(12345)`) → CPU reference via `strata::dequantize_q2_0`
(`include/strata/artifact/dequant.hpp`, a scalar transcription that
`bench/micro/dequant_xcheck` proved equal to ggml) → canonical planes →
kernel → per-element `std::memcmp` (**bit comparison, no tolerance**) →
**non-vacuity check** (all 4 codes must occur; ≥ 4 distinct output values) →
`0/1` exit.

## Definition of done

1. `./poc/sycl/run.sh` from a clean `build/` is green, including the new
   ctest entry `k_dequant_s2_parity` (default 200 000 blocks, `--selftest`).
   ctest output shows `dequant_s2: 200000 blocks, 12800000 elements, 0
   mismatched` and the `bit-exact ...` line.
2. `check_mirrors.sh` passes with **all ten new mirror blocks** (list below;
   the implemented file has exactly 10 — the plan's "nine" undercounted) —
   every fixture, the CPU reference, the kernel body, and the verdict logic
   are verbatim from the main tree, so the SYCL driver provably tests what
   the CUDA test tests.
3. **Mutation test (non-vacuity of the harness itself):** with the kernel
   temporarily broken (e.g. bit offset `* 2` → `* 2 + 1`), the driver must
   fail with a first-mismatch report; restoring it must go green. A parity
   driver that can't catch a deliberately wrong kernel is worse than none.
4. Edge runs pass by hand: `--blocks 1` (single work-item) and
   `--blocks 12345` (not a multiple of the 256 work-group — exercises the
   bounds check).
5. `env -u LD_LIBRARY_PATH ./build/k_dequant_s2_parity` works (the T1 rpath +
   UMF plumbing covers it — verify, don't assume).
6. `git status`: additions under `poc/sycl/` and `plans/` only.

## Files

```
poc/sycl/
  kernels/dequant_s2.cpp          NEW  (T1 convention: one kernel, one submit)
  drivers/dequant_s2_parity.cpp   NEW
  CMakeLists.txt                  EDIT (two new targets + ctest entry)
```

### `kernels/dequant_s2.cpp`

```cpp
namespace strata::kernels {
// Contract note (differs from the CUDA wrapper on purpose): the CUDA
// dequant_s2() synchronises before returning; the SYCL submit_ returns the
// event and leaves synchronisation to the caller. Phase B integration wraps
// this; Phase A drivers call .wait() / ctx.wait_and_throw().
sycl::event submit_dequant_s2(sycl::queue& q, const uint8_t* codes,
                             const float* scales, float* out, int64_t n_blocks);
}
```

Body:

- `constexpr int QK = 64; constexpr int CODES_PER_BYTE = 4;` — mirrored from
  `src/kernels/cuda/dequant_s2.cu:22:23`.
- `n_blocks <= 0` → return an already-complete event (`q.submit([](sycl::handler&){})`),
  mirroring the CUDA early return without a sync call.
- Launch via `strata_launch(q, (size_t) n_blocks, 256, lambda)` — same
  work-group size as the CUDA `blockDim`. **Implemented finding:** the Intel
  Level Zero backend throws on non-uniform work-groups (`items % wg != 0`),
  so `strata_launch` was changed in T2 to round the item count up to a
  work-group multiple — padded work-items must fall through each kernel's
  bounds check (every PoC kernel now must have one; stated in `launch.hpp`).
  Inside the lambda:
  `const long long b = (long long) it.get_global_id(0);` (SYCL-specific, not
  mirrored), then the kernel body **mirrored verbatim** from
  `src/kernels/cuda/dequant_s2.cu:27:35` (`if (b >= n_blocks) return;` through
  the end of the 64-iteration loop, original indentation kept).
- Pointers are USM raw pointers captured by value; no accessors, no local
  memory — this is the cheapest possible port on purpose.

**Bit-exactness argument (why a tolerance-free comparison can pass):** the
output of each element is one of `{−d, 0, +d, 2d}` where `d` is an exact
fp16→fp32 widening. `code−1 ∈ {−1,0,1,2}` is exactly representable, and
scaling by ±1/2 is exact in binary32 — **the multiply cannot round**, so no
compiler FMA/contraction choice can change the bits. Any observed difference
is a real fault. (This is also why the CUDA test uses `memcmp`.)

### `drivers/dequant_s2_parity.cpp`

The CUDA parity test with its CUDA calls replaced by `sycl_compat`
equivalents; everything else behind markers. Mirror blocks (all `@ 6bd3e58`):

| # | Marker | Source range | Lands in |
|---|--------|--------------|----------|
| 1 | `SYCL-MIRROR-BEGIN include/strata/artifact/dequant.hpp:33:55` | `fp16_to_fp32` | `namespace strata { ... }` |
| 2 | `... dequant.hpp:64:66` | `read_u16` | `namespace strata { ... }` |
| 3 | `... dequant.hpp:69:78` | `dequantize_q2_0` (the CPU reference) | `namespace strata { ... }` |
| 4 | `SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:22:22` | `constexpr int QK = 64;` (line 20 is `namespace {` — plan typo, corrected) | anonymous namespace |
| 5 | `... dequant_s2_parity.cpp:31:35` | `kScales` table + comment | anonymous namespace |
| 6 | `... dequant_s2_parity.cpp:40:51` | CLI parsing (`--blocks`, `--selftest`) | `main` |
| 7 | `... dequant_s2_parity.cpp:52:79` | fixture: raw blocks, CPU reference loop, canonical planes | `main` |
| 8 | `... dequant_s2_parity.cpp:95:137` | compare loop, non-vacuity, verdict printing | `main` |
| 9 | (kernel file) `SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_s2.cu:21:22` and `:27:34` | constants + kernel body (line 35 is the function's closing brace — the SYCL lambda supplies its own; plan said 22:23/27:35, off by one) | `kernels/dequant_s2.cpp` |

**Two whitespace/namespace traps the mirror discipline catches — plan for
them now:**

- Blocks 1–3 sit in `namespace strata` in the original (0-column
  indentation); they land inside `namespace strata { ... }` in the driver and
  must keep their original columns — verbatim means verbatim, and
  `check_mirrors.sh` will fail any re-indentation.
- Blocks 4–8 reference `strata::dequantize_q2_0` / `strata::fp16_to_fp32`
  with qualification (blocks 6–8) — so the mirrored functions *must* live in
  namespace `strata`, not the driver's anonymous namespace.

**Not mirrored (the SYCL glue, with comments):** `ctx c;`, the three
`c.device_alloc` calls (for the CUDA `cudaMalloc`s of lines 83–85),
`c.q.memcpy` + `c.wait_and_throw()` (lines 86–88, 92–93), the kernel call —
`strata::kernels::submit_dequant_s2(c.q, d_codes, d_scales, d_out, n_blocks).wait();`
(replacing line 90's `strata::kernels::dequant_s2(...)`), and the final
`c.free_device/free_host` (lines 139–141). The original's
`check(cudaError_t, ...)` helper (lines 23–28) is replaced by
`sycl_compat::fail` — deliberately not mirrored.

### `CMakeLists.txt` edits

- `add_library(k_dequant_s2 STATIC kernels/dequant_s2.cpp)` — compile picks
  up `-fsycl` from the global `add_compile_options`; no link options needed
  on the static lib (the consumer exe already carries `-fsycl` + rpath + UMF
  from the T1 `foreach`).
- `add_executable(k_dequant_s2_parity drivers/dequant_s2_parity.cpp)` and
  `target_link_libraries(k_dequant_s2_parity PRIVATE k_dequant_s2)`.
- Add `k_dequant_s2_parity` to the T1 `foreach` so it gets `-fsycl` + rpath +
  `libumf` like every other target.
- `add_test(NAME k_dequant_s2_parity COMMAND k_dequant_s2_parity --selftest)`,
  `TIMEOUT 120`, registered **after `smoke_parity`, before the `t0_*` tests**
  (registration order is ctest order: smoke first, bandwidth gate last —
  keep that).

## Task order

1. **Kernel first** (`kernels/dequant_s2.cpp` + `k_dequant_s2` target).
   Build-only check: it compiles, no test yet. *(~20 min — the kernel is 15
   lines of logic)*
2. **Driver** with all mirror blocks + glue; `add_test`; run the single test.
   *(~1–2 h — most of it is pasting verbatim and keeping columns; the
   realistic debugging is CMake wiring and marker typos, not SYCL)*
3. **Edge runs** (`--blocks 1`, `--blocks 12345`). *(~10 min)*
4. **Mutation test**: break the bit offset, expect a failing first-mismatch
   report, restore, expect green. *(~20 min — this step is not optional; it is
   the repo's non-vacuity doctrine applied to the harness)*
5. **Final pass**: done criteria 1–6, including `env -u LD_LIBRARY_PATH` and
   `git status`. *(~20 min)*

## Risks

| Risk | Mitigation |
|---|---|
| Re-indenting mirrored code to look nicer | `check_mirrors.sh` fails it; keep original columns (noted per block above) |
| Mirrored `strata::` functions placed in the wrong namespace | block table above says `namespace strata`; the driver won't compile otherwise, so this fails loudly, not silently |
| Someone "improves" the kernel (vector loads, unroll) in T2 | Out of scope by plan: the port must be naive like the CUDA original; the mirrored kernel body enforces it, and optimization belongs to the CUDA file's own Phase 3, validated by the CUDA parity test first |
| `n_blocks` overflow in `(size_t) n_blocks` grid | `int64_t` in, `size_t` out on a 64-bit host — fine for any real artifact size; noted in the header comment |
| ctest timeout if the 200k-block CPU reference loop is slower than expected | measured in step 2; 120 s is generous (the loop is ~13 M trivial elements); lower it to the measured value × 4 if needed |
| Driver silently tests a *weaker* fixture than the CUDA test (e.g. dropped non-vacuity block) | block 8 includes the non-vacuity verdict; done criterion 1 requires the printed `non-vacuity: 4 of 4 codes present` line |

## Effort

- Steps 1–2: ~2–4 h. Step 3: ~0.25 h. Step 4: ~0.5 h. Step 5: ~0.5 h.
- **Total ≈ half a day, and actual ≈ half a day** — inside the Phase A
  table's "T2: 1–2 days". The single real bug was the non-uniform
  work-group rejection (a T1 latent gap the smoke test's exact multiple
  happened not to trip); everything else was bookkeeping, and the mirror
  checker caught one trailing-blank-line drift on first use — the mechanism
  working as designed. Full result in `plans/sycl-phase-a-report.md` §T2.

## Handoff to T3

After T2, the pipeline has proven real math end-to-end, and the one thing
still unproven is the **shared-memory kernel-struct form** (the pattern
`launch.hpp` explicitly defers): `router_top10` (226 CUDA lines) is the first
kernel needing a `sycl::local_accessor` and intra-group reduction — T3 will
establish that pattern, plus the first real use of the 16-wide subgroup
finding from T0 (its CUDA code assumes 32-lane warps). `s_gemv` (the flagship)
and `sampler` (heaviest, deferrable) follow at T4/T5 per the parent plan.
