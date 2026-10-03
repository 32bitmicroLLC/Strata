# T5 step 6 — bench + full regression

Close the measurement half of T5, per the steps file's Step 6 and parent
plan §8. The bench machinery already landed in Step 4 (`--bench` mirrors
`bench_sampled`, driver 391:434, minus CUDA events; `k_sampler_bench`
registered with TIMEOUT 300), and provisional numbers were recorded in
the Step 4 report section. This step finalises the numbers and re-proves
the suite.

## Audit of the current state (what is already done)

- `k_sampler_parity --bench` runs and prints the `us per call` table
  (Step 4 provisional numbers: 138–610 us over T ∈ {1,4,8} × k ∈ {20,64},
  split path, 248,320 vocab).
- The full suite was green at the close of Step 5 (19/19, ~18 s, mirrors
  exit 0, 259 OK blocks).
- **One gap found:** the `k_sampler_bench` ctest entry pins only
  `TIMEOUT 300`, no `ENVIRONMENT`. The dispatcher's `sampled_path()` reads
  the caller's environment, so the bench entry measures whichever path
  the inherited shell happens to select — nondeterministic. The other
  three entries are pinned exactly for this reason.

## 6.1 — finalise the bench numbers

1. **Pin the bench entry to the default path** in
   `poc/sycl/CMakeLists.txt` (per §8 / the steps file: "This measures
   whichever path the env selects (default: split)" — pin it so it is
   *known* to be split):

   ```cmake
   set_tests_properties(k_sampler_bench PROPERTIES
       ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0" TIMEOUT 300)
   ```

   and re-run `cmake .` in the build directory (properties only — no
   rebuild needed).

2. **Run the bench three times** (`env -u LD_LIBRARY_PATH ctest -R
   "^k_sampler_bench$"` or the binary directly). Each run must print
   `sampled path: split top_k (default)` as its first line — that line is
   part of the record, not decoration. Wall-clock numbers vary with
   machine load and GPU clocks; three runs give a median and a spread
   instead of one possibly-lucky number.

3. **Record in report §T5** (finalising the Step 4 provisional table):
   one row per (T, k) cell — median us per call across the three runs
   plus the observed min–max spread — and the standing caveats:

   - chrono wall-clock around submit + `ctx.wait_and_throw()`, 3
     warm-ups + 50 timed calls, fixed N(0,3) logits, top_p 0.95,
     temperature 0.7, seed 1 — the mirrored `bench_sampled` spec.
   - **First Arc sampler figures**; no CUDA comparison exists in this
     checkout (parent §8: the numbers join T6's ledger).
   - These are performance information, not a correctness gate — the
     parity contract (exact picks, Step 4/5) is what the suite proves.

## 6.2 — full regression

1. `env -u LD_LIBRARY_PATH ctest` over the whole suite: T0 ×3, T1–T4
   parity/bench entries, the four T5 probes, `k_sampler_smoke`,
   `k_sampler_parity` ×3 (env-pinned paths), `k_sampler_bench` — all
   green.
2. `bash poc/sycl/check_mirrors.sh` exit 0 (259 OK blocks).
3. The env pin added to `k_sampler_bench` changes no other test; the
   re-proof is cheap and is the step's done-when, not ceremony.

## Risks

- **Bench noise** (load, GPU clocks): mitigated by the 3-run median +
  spread; a wildly off outlier run is redone, never averaged in quietly.
- **Transient ctest timeout / device flake** (Steps 3 and 5 precedent):
  rerun the suite before recording; do not record numbers from a run
  whose log shows an incident.
- No compilation is expected in this step (CMake properties only); the
  icpx exit-3 workaround does not apply unless the CMake change is made
  wrong enough to break the build.

## Done when (Step 6 complete)

- [x] `k_sampler_bench` pins `ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0"`;
      every bench run's log shows `sampled path: split top_k (default)`.
- [x] Bench numbers (median of 3 + spread, per (T, k) cell) recorded in
      report §T5 with the caveats above. (The Step 4 provisional table
      stays in place as the historical record — the final table is in the
      step-6 section, since that is where the numbers are finalised.)
- [x] Full suite green under `env -u LD_LIBRARY_PATH` (19/19);
      `check_mirrors.sh` exit 0 (259 OK).
- [x] Steps file marked done, this plan's boxes ticked.

Est: **part of 1 d** (the steps file's figure); realistically under an
hour — the bench exists, the suite was green at Step 5's close, and the
audit found one gap.
