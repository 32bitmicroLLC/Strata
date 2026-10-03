# T5 Step 1 — toolchain probes (`poc/sycl/t5/`)

Parents: `plans/sycl-phase-a-t5.md` §3 (probe spec + gates) and
`plans/sycl-phase-a-t5-steps.md` Step 1. Three small gate programs decide the
penalty-bitmap build strategy, the double-math strategy, and the 1024-wide
reduction pattern for the `sampler` port (Steps 2–3).

All three use `sycl_compat/test_ctx.hpp` (`sc::ctx`), are registered in
`poc/sycl/CMakeLists.txt` as standalone exes + ctest entries (following the
T3 `t3_sg_probe` pattern), and are run under `env -u LD_LIBRARY_PATH`.

## Current status (verified on this machine)

| probe | source | gate | result |
|---|---|---|---|
| P1 | `t5/atomic_probe.cpp` | `atomic_ref` fetch_or on a local accessor, 1024 threads | **PASS** — 0 of 256 words differ |
| P2 | `t5/math_probe.cpp` | device double `exp` / `logf` bit-identical to host | **FAIL** — see below |
| P3 | `t5/tree_probe.cpp` | 1024-wide 10-level local-memory tree vs host serial scan (64 trials, ties/NaN/−inf included) | **PASS** — 0 of 64 trials differ |

Consequences already locked in:

- **P1 → the bitmap keeps the CUDA parallel-build structure**
  (`sycl::atomic_ref<unsigned, relaxed, memory_scope_work_group>::fetch_or`
  over the local accessor). No thread-0 serial fallback needed. Note the
  DPC++ 2026.1 spelling `atomic_ref<T, DefaultOrder, DefaultScope>` — the
  port must use the same three-template-argument form.
- **P3 → the §4.2 block-argmax replacement is approved at scale**: 1024
  threads, 10 halving barriers, the `(value, index)` total order survives the
  tree against the serial host scan, including deliberate ties, NaNs, and
  −inf sentinels.

## Remaining work: P2 (double math)

P2 as written FAILs both checks:

```
t5_math_probe: device exp(double) vs host std::exp(double): *** FAIL *** (111570 of 320894 differ)
t5_math_probe: device logf vs host (float) std::log: *** FAIL *** (6843 of 100000 + 4 fixture values differ)
```

Sample: `x -744 mx -744`: device `0.99975588917489733` vs host
`0.99975588917489722` — last-ulp differences in the double `exp`, on the
structured grid; and device `logf` differs from host `(float) std::log` on
~6.8% of the log-uniform grid **and on all four fixture min_p values**
{0.05, 0.3, 0.5, 0.9}.

The tail only *consumes* these in two ways (sampler.cu 283–316, 396):
ordered double `sum`/`cum` chains compared against `p.top_p`, and the fp32
threshold `thresh = sel_logit[0] + logf(p.min_p)` compared against `sel_logit`
entries. A 1-ulp difference flips a fixture only if some fixture argument is
within ~1 ulp of its cut. So the resolution order is:

### 1.3.1 Fixture-argument re-probe (do this first)

Extend `t5/math_probe.cpp` (or add a fourth small probe) to stop measuring the
structured grid as the decision basis and instead measure **the exact
arguments the fixtures actually compute**:

- Run each fixture's logits through the host-side sampled chain (the
  `sampled_reference` logic that Step 4 will mirror verbatim — re-derive it
  minimally here, a few serial lines, not the full driver) to collect the
  actual per-round argument pairs `(x, mx)` feeding `exp((double) x -
  (double) mx)`, the fp32 `thresh` values, and the distances from every
  `sel_logit[i]` to its `thresh` and from every `cum` to `top_p`.
- Compare device vs host bit-for-bit on exactly those pairs, and print the
  minimum cut margin per fixture (in ulps of the fp32 logit / fp64 cum).
- Gate: device == host on every fixture argument, or margins ≫ 1 ulp →
  adopt fallback (a): document the exact affected set (grid counts above)
  in the report and proceed to Step 2 with the plain `exp`/`logf`.

### 1.3.2 Hand-rolled double `exp` (only if 1.3.1 hits a fixture argument)

Per parent plan §3 fallback (b): a correctly-rounded double `exp` in device
code for fp32-representable arguments in [−745, 0] — range-reduce by
`ln 2` into a small interval, double Taylor with enough guard bits — ~100
lines, first proven inside the probe against the *full* grid before it goes
into `kernels/sampler.cpp`. Re-run 1.3.1 afterwards; the kernel's tail uses
this `exp` verbatim-mirrored wherever the CUDA source spells `exp`.

### 1.3.3 The `logf`/min_p threshold

Same decision logic, fp32 side: 1 ulp of `thresh` only matters if a fixture
has a `sel_logit[i]` within ~1 ulp of it (1.3.1 measures exactly this).
- Margin-safe → document the 1-ulp `logf` divergence (device libm vs host
  glibc) with the affected set in the report; the mirrored tail keeps
  `logf(p.min_p)` verbatim.
- Knife-edge → hand-roll the fp32 log for the four fixture min_p values
  (table/extended-precision for the fixed constants is far cheaper than a
  general `log`), or, if no clean fix exists, a **written report finding on
  that fixture** — never a silent tolerance, per the parent plan.

### 1.3.4 Stop rule

If after 1.3.2/1.3.3 a fixture still sits on a knife edge that no device-side
change can clear, the honest output is a written finding (parent plan §3
fallback (c) / risk R1) — that finding goes in report §T5 and does not block
the other three path runs.

## CMake / run

Already wired (verified in `poc/sycl/CMakeLists.txt`): `t5_atomic_probe`,
`t5_math_probe`, `t5_tree_probe` as exes + ctest, in the `-fsycl`/rpath/UMF
foreach, TIMEOUT 120.

```
cd poc/sycl/build && cmake --build . -j
env -u LD_LIBRARY_PATH ctest -R t5_   # P1, P3 green now; P2 until 1.3.x lands
```

The `--bench`/3-path `k_sampler_parity` ctest entries in CMake reference
targets that land in Steps 2–4; ctest on `t5_` only is the Step 1 surface.

## Done when (Step 1 complete)

- [ ] P1, P3 green (done) and their outcomes documented here + carried into
      report §T5 (bitmap: `atomic_ref` parallel build; reduction: 1024-wide
      local-memory tree).
- [ ] 1.3.1 run: fixture-argument agreement + cut margins recorded.
- [ ] Double-math strategy decided and recorded: plain `exp`/`logf` with the
      documented affected set (fallback a), or hand-rolled double `exp`
      (fallback b) proven green on the full grid and fixture arguments.
- [ ] `min_p` threshold strategy decided per 1.3.3 (documented 1-ulp set,
      hand-rolled fp32 log, or written finding).
- [ ] `t5_math_probe` exit 0 (or its FAIL replaced by the documented finding,
      which is stated in report §T5 before Step 2 starts).
- [ ] Step 2/3 design inputs fixed: §4.2 tree shape, §4.3 bitmap build
      (atomic), tail math functions named for the port.
