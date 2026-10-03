# T5 Step 2 — kernel part 1: shared helpers + `submit_greedy` + `submit_old` — **done**

Parents: `plans/sycl-phase-a-t5.md` §4 (port design) and
`plans/sycl-phase-a-t5-steps.md` Step 2 (est. 1 day). Step 1
(`plans/sycl-phase-a-t5-step-1.md`) is complete: bitmap via `atomic_ref`
fetch_or (P1), 1024-wide local-memory tournament tree (P3), plain
`exp`/`logf` in the tail (P2/P2b fallback (a)). All three decisions are
consumed by this step.

Scope of this step, per the steps file: the shared device helpers,
`submit_greedy`, and `submit_old` — i.e. everything the non-split sampled
path and the greedy path need — compiled green inside
`poc/sycl/kernels/sampler.cpp`, registered as the `k_sampler` static
library, with every mirror block green in `check_mirrors.sh`.

Main-tree discipline is unchanged: `src/` is never touched; all work is in
`poc/sycl/`; the parity driver and its ctest entries land in Step 4.

## Starting state — a full draft already exists

`poc/sycl/kernels/sampler.cpp` (1,020 lines) was written in an earlier
pass and contains **all five** `submit_*` functions plus the host
dispatcher (`submit_sample_tokens`) — i.e. the code of Steps 2 *and* 3 in
one translation unit, with 75 `SYCL-MIRROR` blocks against
`src/kernels/cuda/sampler.cu`. It has **never been compiled**, is **not
wired into CMake**, and its mirror blocks are **red**. Known defects:

| # | defect | evidence |
|---|---|---|
| D1 | ~74 mirror blocks are re-indented to match the enclosing `q.submit` lambda body, violating the verbatim rule | `check_mirrors.sh` DRIFT on every in-lambda block, e.g. `sampler.cu:148:148` (mirror line 283 has 12 leading spaces; the source line has 4) |
| D2 | one marker misspells the source as `src/kernels/cuda/sampler.cuda` → "source missing" | mirror marker at draft line 980 (`...sampler.cuda:857:857`) |
| D3 | every SHA annotation reads `@ 6cabad2`, but the source file's last commit is `1643965` and the tree HEAD (clean) is `bb7e783`; line ranges 1–885 are still valid against the current file (the coupled-draft commit only appended 891–933) | `git log -- src/kernels/cuda/sampler.cu`; checker output |
| D4 | the header's P2 note claims the 1-ulp tail math is "verified by the fixtures running" — the fixtures have never run against the SYCL kernel | draft lines 35–41 |
| D5 | never compiled with `icpx -fsycl`; DPC++ errors unknown | no build target references the file |

The Step 3 code (split + one-block + dispatcher) is inside the same TU
and therefore *compiles* in this step, but its *correctness* is Step 3's
verification target; the Step 4 ctest entries pin the paths functionally.

Note the scope consequence: the steps file split Step 2/3 as "write part 1,
verify those blocks" vs "write part 2, verify the rest (~30 blocks)". The
draft being complete means all 75 blocks already exist and are red, and
`check_mirrors.sh` gates the whole tree — so the mechanical re-flow (2.1)
spans both steps' blocks and is done here in one pass; Step 3 then has no
mirror work left and is pure split/one-block/dispatcher correctness plus the
local-memory budget check (3.5).

### The indentation fix (D1)

`poc/sycl/kernels/router_top10.cpp` (T3, green) already fixed this exact
problem with an established pattern: **mirror blocks keep the CUDA source's
original indentation even inside the lambda**; the glue lines around them
carry the lambda indentation. The checker diffs lines byte-for-byte and
does not understand nesting, so a mirror block indented to its lambda will
never pass. The fix for all ~74 blocks is mechanical: restore each block's
lines to the original indentation, re-flow the surrounding glue to the
lambda indentation, keep the marker lines where they are. No mirrored line
may be edited otherwise — that would be new drift.

## Work items

### 2.1 Repair the mirror blocks (D1, D2, D3)

1. Re-flow every in-lambda mirror block to the source indentation
   (router_top10 pattern), block by block.
2. Fix the `sampler.cuda` marker (line 980) to `sampler.cu`.
3. Update all 75 SHA annotations from `@ 6cabad2` to `@ bb7e783`
   (current HEAD; tree clean).
4. Gate: `bash poc/sycl/check_mirrors.sh` reports `OK` for all 75 blocks in
   `kernels/sampler.cpp` and exits 0 for the whole PoC tree.

### 2.2 Fix the header documentation (D4)

Rewrite the P2 note (lines 35–41) to state the actual evidence: device
`exp` differs from host on 14,443 of the 130,589 fixture tail arguments
(1 ulp), the four fixture min_p values are bit-identical under `logf`, and
the minimum cut margins (≥ 2.1e8 double ulps, ≥ 1.8e8 double ulps,
≥ 8,950 fp32 ulps; report §T5) keep every cut off the knife edge — i.e.
fallback (a) was adopted by the `t5_math_fixture_probe` gate, and the
parity fixtures will confirm it when they run in Step 4. The other
"DOCUMENTED DEVIATIONS" entries are checked against the Step 1 record
while this is happening.

### 2.3 CMake wiring

In `poc/sycl/CMakeLists.txt`:

1. `add_library(k_sampler STATIC kernels/sampler.cpp)` next to the other
   kernel libs — the global `-fsycl` compile flag applies, as for
   `k_router_top10`/`k_s_gemv`.
2. **Move the three pre-registered `add_test(k_sampler_parity*)` entries
   down to Step 4** (they reference an executable that does not exist yet
   and would red the full-suite ctest; keep the `TIMEOUT 300` property on
   `k_sampler_bench`). The `# k_sampler / k_sampler_parity are added here
   once the sources land` comment is replaced by the library line.
3. No foreach change for the library itself; the consumer executable joins
   the `-fsycl` link/rpath/UMF foreach in Step 4.

### 2.4 Compile green (D5)

`icpx -fsycl` must accept the whole TU. Expected risk areas, from the
probe findings and the draft's glue:

- **`sycl::atomic_ref` spelling** — the three-template-argument form
  (`<unsigned, memory_order_relaxed, memory_scope_work_group>`) per P1;
  the draft's bitmap build must use exactly it.
- **Barriers in diverged control flow** — a `group.barrier()` reached by
  only some threads deadlocks. The greedy/old tournament loop and the
  split's four lockstep region merges (draft header note, lines 44–47)
  must be audited so every thread in the work-group reaches each barrier.
  The one-shot smoke (2.5) is the deadlock detector.
- **Local accessors** — multiple handler-constructed accessors per kernel
  (bitmap words, `(value, index)` scratch, sel list, philox state) must all
  be value-captured; check sizes against Arc's 128 KiB work-group budget
  (T0) for the worst fixture shape (vocab 248,320 greedy: ~31 KiB bitmap +
  tournament scratch).
- **`int2` glue layout** — `{int, int}` matching CUDA's `int2` (y = float
  bits), and its use as a `sycl::malloc_device<int2>` element type for the
  split scratch (8-byte alignment; fine).
- **Kernel argument counts / value-capture of `SamplerParams`** — struct
  capture into the kernel lambda; keep it value-captured like the other
  PoC kernels.
- **`exp`/`logf` availability on device** — plain libm spellings per P2b.

Any DPC++ fix that changes a mirrored line is a **no-go**: only glue may
change. If a mirrored construction cannot compile as written, stop and
write a finding (report §T5) before bending the mirror.

### 2.5 One-shot path smoke — `poc/sycl/t5/k_paths_smoke.cpp`

Small ctest exe (not a fixture parity run — that is Step 4) that proves the
library links and the four paths launch without crashing or deadlocking:

- fixed input: 64 rows × 512 vocab, one token history row, penalties on
  (`penalty_last_n = 16`, repeat/freq/present non-zero);
- run `submit_sample_tokens` four times on the same data: greedy
  (`p.greedy = true`), old (`STRATA_OLD_SAMPLER=1`), one-block
  (`STRATA_SAMPLER_ONE_BLOCK=1`), default split — each on a fresh queue,
  `wait_and_throw()`'d;
- assert: every `out[r]` is a valid token index; the greedy row picks equal
  the host serial argmax after penalties (the parity driver's penalty
  arithmetic mirrored as glue in the smoke, drift-checked); and the three
  sampled paths agree with each other row for row (they share the tail
  math, so any disagreement is a real bug);
- registered as ctest `k_sampler_smoke` (TIMEOUT 120), in the foreach link
  list with `-fsycl` + rpath + UMF.

### 2.6 Record

- `plans/sycl-phase-a-t5-steps.md`: mark Step 2 done, note the one-shot
  smoke as the verification added beyond the steps-file minimum.
- `plans/sycl-phase-a-report.md` §T5: add the step-2 outcome (library
  target, mirror repair count, any DPC++ fixes and what they touched,
  smoke results).

## CMake / run

```bash
cd poc/sycl/build && cmake . && cmake --build . -j
bash ../../check_mirrors.sh                 # all blocks OK
env -u LD_LIBRARY_PATH ctest -R "t5_|k_sampler_smoke" --output-on-failure
```

Full-suite `ctest` stays green: the `k_sampler_parity*` entries are not
registered until Step 4.

## Done when (Step 2 complete)

- [x] `k_sampler` static library target exists and compiles green with
      `icpx -fsycl` (whole TU, including the Step 3 kernels).
- [x] `check_mirrors.sh` exits 0; all 75 mirror blocks in
      `kernels/sampler.cpp` report `OK`; markers read
      `sampler.cu … @ bb7e783`.
- [x] Header deviations match the Step 1 record (P2b gate numbers, not
      "verified by the fixtures running").
- [x] `k_sampler_smoke` ctest green: all four paths launch, greedy picks
      match the host serial argmax, the three sampled paths agree (each
      equals the host serial reference: 0/64 rows differ).
- [x] CTest suite green under `env -u LD_LIBRARY_PATH` (15/15: t5 probes +
      `k_sampler_smoke` + all T0–T4 entries).
- [x] Steps file + report §T5 updated.

## Outcome

All six work items done. The one substantive finding of the step is the
**`warp_first` butterfly bug** (glue, not mirror drift): the
shared-memory butterfly transcribed from CUDA's `__shfl_xor` chain wrote
each lane's slot **once** at entry. A shuffle butterfly updates the
registers in place at every stage; the shared-memory form must write the
running max back to the lane's own slot at every stage, or each lane ends
with its own quadrant max instead of the warp max. Symptom: the split path
returned duplicated, non-max top-k lists (old/one_block were correct); the
2.5 smoke caught it (61/64 rows off), and it was isolated via a host
serial reference, per-path standalone runs, per-lane dumps, and a minimal
standalone repro kernel. Fix: two glue lines (`s_v[lane] = bv; s_i[lane] =
bi;` before each stage's barrier). Full narrative in report §T5.

The smoke's final assertion is stronger than the steps-file minimum:
each sampled path is compared against a host serial reference (k rounds of
the penalised argmax + the mirrored tail with host math) rather than only
pairwise agreement, so a bug shared by all three paths could not hide.

## Out of scope (later steps)

- Split/one-block *fixture* correctness — Step 3.
- `k_sampler_parity` driver, its three ctest path entries, and
  `k_sampler_bench` — Step 4 (the `--bench`/3-path wiring already sketched
  in CMake returns here).
- `stream_capturing`, the per-stream scratch cache, fixture 18a's
  captured-graph half, and the coupled draft kernels (670:751, 891:933) —
  Phase B, as documented in the draft header.

## Estimate

1 day per the steps file: the draft is ~90% written; the day goes to the
mirror re-flow (mechanical but 74 blocks), unknown DPC++ compile fixes,
the smoke driver, and the record.
