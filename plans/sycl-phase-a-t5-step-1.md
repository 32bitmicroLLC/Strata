# T5 Step 1 — toolchain probes (`poc/sycl/t5/`)

Parents: `plans/sycl-phase-a-t5.md` §3 (probe spec + gates) and
`plans/sycl-phase-a-t5-steps.md` Step 1. Three small gate programs decide the
penalty-bitmap build strategy, the double-math strategy, and the 1024-wide
reduction pattern for the `sampler` port (Steps 2–3).

All three use `sycl_compat/test_ctx.hpp` (`sc::ctx`), are registered in
`poc/sycl/CMakeLists.txt` as standalone exes + ctest entries (following the
T3 `t3_sg_probe` pattern), and are run under `env -u LD_LIBRARY_PATH`.

## Current status (all gates resolved — Step 1 complete)

| probe | source | gate | result |
|---|---|---|---|
| P1 | `t5/atomic_probe.cpp` | `atomic_ref` fetch_or on a local accessor, 1024 threads | **PASS** — 0 of 256 words differ |
| P2 | `t5/math_probe.cpp` | device double `exp` / `logf` vs host on a structured grid | **documented finding** (1-ulp gaps; not the decision basis — see below) |
| P2b | `t5/math_fixture_probe.cpp` | the exact fixture tail arguments + cut margins | **GATE PASS** — fallback (a) adopted |
| P3 | `t5/tree_probe.cpp` | 1024-wide 10-level local-memory tree vs host serial scan (64 trials, ties/NaN/−inf included) | **PASS** — 0 of 64 trials differ |

Consequences locked in (carried to report §T5):

- **P1 → the bitmap keeps the CUDA parallel-build structure**
  (`sycl::atomic_ref<unsigned, relaxed, memory_scope_work_group>::fetch_or`
  over the local accessor). No thread-0 serial fallback needed. Note the
  DPC++ 2026.1 spelling `atomic_ref<T, DefaultOrder, DefaultScope>` — the
  port must use the same three-template-argument form.
- **P3 → the §4.2 block-argmax replacement is approved at scale**: 1024
  threads, 10 halving barriers, the `(value, index)` total order survives the
  tree against the serial host scan, including deliberate ties, NaNs, and
  −inf sentinels.
- **P2/P2b → the ported tail uses plain `exp` / `logf`** (fallback (a)); the
  affected set is documented in report §T5. No hand-rolled double `exp`
  (1.3.2) and no fp32 `log` rework (1.3.3) is required.

All four probes are ctest targets (`t5_atomic_probe`, `t5_math_probe`,
`t5_tree_probe`, `t5_math_fixture_probe`); ctest `-R t5_` is 4/4 green under
`env -u LD_LIBRARY_PATH`, and `check_mirrors.sh` passes on every
`math_fixture_probe` mirror block.

## P2 resolution — how it went

P2 as written FAILed its grid checks:

```
t5_math_probe: device exp(double) vs host std::exp(double): *** FAIL *** (111570 of 320894 differ)
t5_math_probe: device logf vs host (float) std::log: *** FAIL *** (6843 of 100000 + 4 fixture values differ)
```

(That "+ 4 fixture values differ" in the old print is a static format string;
the fixture-value loop actually found **0 of 4** differing — see P2b below.)

The tail only *consumes* these in two ways (sampler.cu 283–316, 396):
ordered double `sum`/`cum` chains compared against `p.top_p` / the Philox
`u`, and the fp32 threshold `thresh = sel_logit[0] + logf(p.min_p)` compared
against `sel_logit` entries. A 1-ulp difference flips a fixture only if some
fixture argument sits within a few ulps of its cut.

### 1.3.1 Fixture-argument re-probe — DONE

`t5/math_fixture_probe.cpp` (P2b) rebuilds every sampled-chain fixture of
`src/kernels/sampler_parity.cpp` data-for-data (same seeds; the generation
blocks and the `mirror_select`/Philox/`sampled_k`/`window_of` machinery are
verbatim mirrors, drift-checked by `check_mirrors.sh`), runs each row through
the host transcription of the kernel tail, and measures exactly:

- 10,670 rows (1,340 void — fixture 16's NaN/±inf rows, where no cut fires),
  130,589 collected double-exp arguments, 4 distinct min_p values;
- device `exp` vs host `std::exp` on exactly those arguments:
  **14,443 of 130,589 differ** (all 1 ulp);
- device `logf` vs host `(float) std::log` on the fixture min_p values:
  **0 of 4 differ** (bit-identical);
- minimum cut margins per fixture, worst across all fixtures:
  top_p **214,748,367 double ulps**, draw **180,290,208 double ulps**,
  min_p **8,950 fp32 ulps** (fixture 17 is the closest to a cut in every
  column).

Gate (thresholds in the probe header): a 1-ulp exp term shifts each cum by
≤ ~128 double ulps (≤ 64 terms), and a 1-ulp `thresh` shift flips a survivor
only within 1 fp32 ulp — the observed margins exceed the gates (1024 double
/ 8 fp32 ulps) by ~6 orders of magnitude.

**Verdict: GATE PASS — fallback (a) adopted.** The ported tail keeps the
plain `exp` / `logf` (and the mirrored host references keep `std::exp` /
`std::log`), and the affected set above is the documented finding in report
§T5. `t5_math_probe` keeps its grid numbers as an informational regression
record (exit 0); the gate lives in `t5_math_fixture_probe`.

### 1.3.2 / 1.3.3 — not triggered

- No hand-rolled double `exp` needed (no fixture argument near a cut).
- No fp32 `log` rework needed (all four fixture min_p values bit-identical).
- No knife-edge fixture → no written finding of the 1.3.4 class.

## CMake / run

Wired in `poc/sycl/CMakeLists.txt`: `t5_atomic_probe`, `t5_math_probe`,
`t5_tree_probe`, `t5_math_fixture_probe` as exes + ctest, in the
`-fsycl`/rpath/UMF foreach; TIMEOUT 120 (the fixture probe runs ~1.5 s).

```
cd poc/sycl/build && cmake --build . -j
env -u LD_LIBRARY_PATH ctest -R t5_   # 4/4 green
```

The `--bench`/3-path `k_sampler_parity` ctest entries in CMake reference
targets that land in Steps 2–4; ctest on `t5_` only is the Step 1 surface.

**Carry-over note for Step 2:** `poc/sycl/kernels/sampler.cpp` (1,020 lines)
already exists in the tree from an earlier pass but is **not** wired into
CMake, and its mirror blocks are currently **red** in `check_mirrors.sh`
(indentation drift from the CUDA source, plus one misspelled `.cuda`
marker). Step 2 must make those blocks green before the `k_sampler` /
`k_sampler_parity` targets join ctest.

## Done when (Step 1 complete)

- [x] P1, P3 green and their outcomes documented here + carried into
      report §T5 (bitmap: `atomic_ref` parallel build; reduction: 1024-wide
      local-memory tree).
- [x] 1.3.1 run: fixture-argument agreement + cut margins recorded
      (14,443/130,589 exp args 1-ulp; margins ≥ 2.1e8 double / 8,950 fp32 ulps).
- [x] Double-math strategy decided and recorded: plain `exp`/`logf`,
      affected set documented (fallback a).
- [x] `min_p` threshold strategy decided: bit-identical on all four fixture
      values; mirrored tail keeps `logf(p.min_p)` verbatim.
- [x] `t5_math_probe` exit 0 (its FAIL replaced by the documented finding in
      report §T5).
- [x] Step 2/3 design inputs fixed: §4.2 tree shape, §4.3 bitmap build
      (atomic), tail math functions (plain `exp`/`logf`).
