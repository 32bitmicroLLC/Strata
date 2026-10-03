# T5 Step 4 — Driver: `poc/sycl/drivers/sampler_parity.cpp`

Parents: `plans/sycl-phase-a-t5.md` §5.2 (block table), §6 (parity
contract), §7 (mutations, for Step 5), §9 (CMake);
`plans/sycl-phase-a-t5-steps.md` Step 4 (est. 1.5 d). Steps 2–3 are done:
the `k_sampler` library (five kernels + dispatcher) compiles, all 75
kernel mirror blocks are green, and `k_sampler_smoke` proves every
dispatcher branch and path against a host serial reference.

This step ports the CUDA parity driver
`src/kernels/sampler_parity.cpp` (1,285 lines, last touched at `47d6894`)
to `poc/sycl/drivers/sampler_parity.cpp`. The port's big win is that the
entire host-reference half — `reference_pick`, `sampled_reference`,
`mirror_select`/`mirror_pick`, the Philox host step — is **pure C++ and
mirrors verbatim**: this driver provably tests what the CUDA test tests,
with no re-derivation.

## Driver inventory (CUDA original)

| region | lines | SYCL treatment |
|---|---|---|
| header comment + `#include <cuda_runtime.h>` | 1:32 | rewritten for SYCL; `check(cudaError_t, …)` helper **not mirrored** (T3 precedent: SYCL throws, so the glue needs no status checking) |
| `reference_pick` | 34:69 | mirrored verbatim |
| `reference_cut` | 71:102 | mirrored verbatim |
| `run` compare + verdict printf | 123:127 | mirrored verbatim; the alloc/copy lines 108:122 and frees 128:131 are glue (`ctx` allocs, `cudaMemset(d_o, 0xFF)` → host-fill `-1` + copy, final `wait_and_throw`) |
| `PhiloxRound` / `host_philox_uniform` | 137:150 / 152:160 | mirrored verbatim (host transcription of the kernel's Philox) |
| `sampled_reference` | 164:225 | mirrored verbatim — **the** full sampled-chain host reference |
| `sampled_cut` | 227:272 | mirrored verbatim |
| `SelList` | 274:277 | mirrored verbatim |
| `mirror_select` / `mirror_pick` | 280:307 / 309:344 | mirrored verbatim — the kernel's selection/tail semantics, host-side |
| `sampled_k` (host) | 346:346 | mirrored verbatim |
| `DeviceRows` | 350:381 | glue shell: keeps its member shape and `sample()`'s body, alloc/copy/free become `ctx` calls, the `stream` parameter drops |
| `window_of` | 383:389 | mirrored verbatim |
| `bench_sampled` | 391:434 | glue shell keeping its printf/loop bodies; CUDA events → `chrono` wall-clock around submit + `wait_and_throw` (Step 6 runs and records it) |
| `main` arg parse + "sampled path" printf | 436:452 | mirrored verbatim (it reads the same env vars the dispatcher reads; one process per ctest path, so no forks) |
| fixtures 1–18 (from 457) | 457:1279 | mirrored block-per-block at section boundaries; CUDA device/stream lines inside each are glue |
| summary + exit contract | 1280:1284 | mirrored verbatim (`bad` → exit 1; `--selftest` prints `sampler_parity OK`) |

≈ 45 mirror blocks expected; the tree currently holds 187 OK blocks, so
the checker will state the post-Step-4 truth (the parent plan's "~155
total" predates the kernel's 75).

Mirror-block convention: lines that only touch CUDA device memory or
streams are glue; host data generation, reference calls, comparisons,
observability asserts, and verdict prints are mirrored. All annotations
read `@ 47d6894` (last commit of `src/kernels/sampler_parity.cpp`).

## Work items

### 4.1 Reference/compare machinery

Port the block table above, top to bottom. Glue decisions (all documented
in the driver header, mirroring the T3 driver's header style):

- **`check(cudaError_t, …)` is gone.** The original's four uses in
  `run` and the per-fixture `check` calls become plain throwing `ctx`
  calls plus a `c.wait_and_throw()` where the original implicitly
  synchronised (the `cudaMemcpy` back). No fixture logic changes.
- **`cudaMemset(d_o, 0xFF, …)` → host-fill `-1` + `c.q.memcpy`.** The
  original's comment (a row the kernel leaves unwritten can never match)
  is kept as glue text; `-1` is `0xFFFFFFFF` for `int`.
- **Every stream maps to `ctx.q`.** `run`'s `nullptr`, fixture 17's
  `{nullptr, cs}` loop, and `DeviceRows::sample`'s `stream` parameter all
  become one in-order queue. Fixture 17's stream loop becomes a two-
  iteration glue loop (`for (int pass : {0, 1})`) over the verbatim
  compare body — the in-order queue is *more* sequential than CUDA's
  legacy-stream implicit sync, so the assertion (picks match) cannot
  weaken. Documented: **no fixture assertion depends on stream identity**
  (17 checks picks, not stream behaviour).
- **`DeviceRows`** keeps its member layout and `sample()` shape (fill `-1`,
  submit, copy back, return `got`) with `ctx` internals; its destructor's
  frees become `ctx`-owned.
- **`submit_sample_tokens` forward-declared** in the driver against
  `k_sampler` (T3 pattern; the smoke did the same). No new header —
  Phase B moves declarations to `include/`.
- **`bench_sampled`** ported as a glue shell: fixed N(0,3) logits at
  248,320 vocab, T ∈ {1,4,8}, k ∈ {20,64}, top_p 0.95, temperature 0.7,
  3 warm-ups + 50 timed calls, `chrono` wall-clock around submit +
  `wait_and_throw`, the same `us per call` printf format. It measures
  whichever path the env selects (default: split). **Step 6 runs it and
  records the numbers**; this step only needs it to run.

Done when the machinery compiles and `check_mirrors.sh` is green on the
blocks ported so far.

### 4.2 The 18 fixtures (main, from line 457)

Mirrored block-per-block at their section boundaries (landmarks per
§5.2: f1 457, f2 468, f3 506, f4 539, f5 615, f6 646, f7 681, f8 721,
f9 755, f10 782, f11 806, f12 832, f13 896, f14 940, f15 962,
counter-segmentation ~1001, f16 1028, f17 1122, f18 1200). Data
generation, reference calls, observability asserts, and verdict prints
verbatim; inside each fixture only the device/stream lines are glue.
Specific call-outs:

- **Fixture 18a (graph capture, 1240:1260) is excluded** — written
  exclusion per the parent's §2 (no stream capture in Phase A; the
  `stream_capturing` dispatcher clause is deferred). Ported in full:
  18's `compare`/`make` lambdas, the capture-*fallback-decision* part
  that doesn't itself capture, and **18b's row cap** (`T ∈ {64, 70}`:
  64 rows stay split, 70 fall back to one block). The verdict label
  "fallbacks: graph capture, row cap" stays verbatim; the header
  documents that the graph half is out.
- **Fixture 16's NaN tail:** every row holds a `+inf` logit, so the
  tail's arithmetic is NaN (`inf - inf`); no cut fires, no draw lands,
  and the chain returns its last kept entry `sel_ids[k-1]`. The host
  `mirror_pick` does the same NaN dance in doubles, so parity is exact —
  and the P2 1-ulp `exp` gap cannot flip it (no draw or cut comparison
  is evaluated). This is the fixture that would hide a tail-structure
  bug that leaves the pick "valid".
- **Fixture 17** runs `T = 17` rows (the split's scratch is first sized
  for 16 — the per-call alloc regrows it) at vocab 248,320 and 512,
  through 24 configs (2 penalties × 3 top_k × 2 top_p × 2 min_p) × 2
  "streams" = 48 `DeviceRows::sample` calls per vocab. The host `mirror_select` at 248,320 vocab × 17 rows is the
  driver's CPU hotspot (O(k·nv) per row per penalties state); the
  per-path 300 s envelope is generous (the parent notes the whole T4
  suite ran in 0.4 s).
- **The counter-segmentation block** (~1001–1027) uses raw device
  `input`/`output` pointers → `ctx.device_alloc`; one 32-row batch
  submit, 32 single-row submits into `output + i`, and a repeat batch —
  asserting `batch == singles == repeated` and that the draws vary.
  With the counter at `(1 << 32) + 7` this pins the Philox draw
  exactly; any RNG drift in the kernel's counter plumbing reds it.
- **The order-observable fixtures (2, 9, 10, 16, 17)** keep their
  observability asserts intact: fixture 2 requires the two tail orders to
  pick *different* tokens before it checks the kernel (without that half
  the test would pass against either order); 10 requires token 2 to be
  drawn; 16/17 require tied picks to exist; 9 exercises the single
  penalties stage. A wrong sampler order still returns a valid token —
  "it produced a token" proves nothing.
- **Fixture 1** uses `p.greedy = true` (argmax); `top_k = 0` is
  irrelevant on that path but kept verbatim (it would clamp to 64 via
  `sampled_k` if it ever reached the sampled paths).
- P2 note: the fixture cuts were sized in Step 1's `math_fixture_probe`
  gate — 14,443 of 130,589 tail `exp` args differ from host `std::exp` by
  1 ulp, but every measured cut margin is ≥ 2.1e8 double ulps (top_p/draw
  chains) / ≥ 8,950 fp32 ulps (min_p), ~6 orders of magnitude beyond what
  a 1-ulp term can perturb. The fixtures are designed to pass despite the
  documented gap; a knife-edge red here is a finding, not a tolerance.

### 4.3 Parity contract checks (parent §6)

Verified in code, not just claimed:

- picks are **integers compared exactly** — no tolerance anywhere in the
  compare paths (the CUDA test has none; a wrong order is not "close");
- every fixture's **observability assert survives verbatim** (a fixture
  that can't see its feature fails the run even if every pick matches);
- the **Philox draw is pinned exactly** (counter-based: (seed, counter +
  row)); fixtures 3, 9, and the counter block would catch any drift;
- ctest runs the binary **three times** with the env pinned exactly as
  the main CMake does (see 4.4); the dispatcher's env logic is mirrored,
  so the same `ENVIRONMENT` properties apply.

### 4.4 CMake wiring (parent §9)

In `poc/sycl/CMakeLists.txt`, following the existing convention (static
lib + consumer exe; the global `-fsycl`, rpath, and UMF come from the
foreach):

```cmake
add_executable(k_sampler_parity drivers/sampler_parity.cpp)
target_link_libraries(k_sampler_parity PRIVATE k_sampler)

add_test(NAME k_sampler_parity COMMAND k_sampler_parity --selftest)
set_tests_properties(k_sampler_parity PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0"
    TIMEOUT 300)
add_test(NAME k_sampler_parity_one_block COMMAND k_sampler_parity --selftest)
set_tests_properties(k_sampler_parity_one_block PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1"
    TIMEOUT 300)
add_test(NAME k_sampler_parity_old COMMAND k_sampler_parity --selftest)
set_tests_properties(k_sampler_parity_old PROPERTIES
    ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0"
    TIMEOUT 300)
add_test(NAME k_sampler_bench COMMAND k_sampler_parity --bench)
set_tests_properties(k_sampler_bench PROPERTIES TIMEOUT 300)
```

`k_sampler_parity` joins the `-fsycl`/rpath/UMF foreach; the two
placeholder comments ("k_sampler_parity joins this list once the kernel
source lands" / "k_sampler_bench joins this list with its Step 4 ctest
registration") are retired. The `--selftest` flag prints `sampler_parity
OK` on success; the exit contract is mirrored (any `bad` → exit 1).

## Risks / open questions

- **CPU time of the host references** (f16/f17 at 248,320 vocab):
  estimated seconds, well inside 300 s per path; if a path surprises, the
  envelope is a property, not a constant.
- **A red fixture at Step 4 is a real bug, not "greenness is Step 5".**
  The steps file's done-when says the three ctests *run* and defers the
  *detection-power* proof to Step 5's mutations; but a parity red here
  means the kernel disagrees with the mirrored reference and must be
  fixed (or, if it is the documented P2 gap, written up as a finding per
  the parent's §12 contingency) before Step 4 closes.
- **No new public header:** `submit_sample_tokens` is forward-declared in
  the driver (T3 pattern). If Step 6 or a later task wants it in
  `include/`, that is a small follow-up, not a Step-4 blocker.
- **Fixture 18's verdict label** mentions graph capture although 18a is
  excluded — kept verbatim by the mirror rule; the header documents it.
- **`run()` allocates per call** (as in CUDA). Hundreds of small
  `ctx.device_alloc` calls per path are fine on the UMF pool; no
  pooling needed.

## Run

```bash
cd poc/sycl/build && cmake . && cmake --build . -j
bash ../check_mirrors.sh                                  # must stay exit 0
env -u LD_LIBRARY_PATH ctest -R k_sampler_parity --output-on-failure   # 3 paths
env -u LD_LIBRARY_PATH ctest                              # full suite (now 19 tests)
```

(`check_mirrors.sh` lives in `poc/sycl/`, not the repo root — a path
footgun hit and recorded in Step 3.)

## Done when (Step 4 complete)

- [x] `poc/sycl/drivers/sampler_parity.cpp` exists, compiles under
      `icpx -fsycl`, and mirrors the CUDA driver per the block table —
      all host references verbatim, every fixture's data generation,
      reference calls, observability asserts, and verdict prints
      verbatim; 18a excluded in writing.
- [x] `check_mirrors.sh` exits 0 — 72 driver blocks (the plan's ~45
      undercounted; the checker states the truth: 259 OK blocks in the
      tree, was 187).
- [x] All three path ctests — `k_sampler_parity` (split, 2.3 s),
      `k_sampler_parity_one_block` (2.7 s), `k_sampler_parity_old`
      (5.0 s) — run to completion under their `ENVIRONMENT` pins and are
      green (each prints its own "sampled path:" line; verified via
      `ctest -V`); `k_sampler_bench` registered and runs (numbers
      recorded in Step 6).
- [x] Parity contract holds in code: exact integer comparisons, no
      tolerance, observability asserts intact (1,057 tied picks,
      280 sentinel positions, the 0.105 draw share, token-2 draws all
      observed), Philox pinned exactly (counter segmentation PASS).
- [x] Full-suite `ctest` green under `env -u LD_LIBRARY_PATH` —
      19/19 in ~18 s.
- [x] Steps file + report §T5 updated (per-path parity table rows land in
      Step 7's report writing).

## Estimate

1.5 d per the steps file: the machinery (~13 mirrored host blocks) is
mechanical; the day goes to the 18 fixtures' per-block splicing (stream
and device lines split out as glue), the 18a exclusion, the CMake
wiring, and driving the three paths to green.
