# T5 step 7 — report §T5 final + commit

Close out T5: finish the report §T5, sync the parent plan, and land the
final commit `phase a t5`. Steps 1–6 are already committed incrementally
(`bb7e783`, `5176131`, `3433666`, `578b1c3`, `047e457`, `1faea0e`); the
working tree is clean, so this step's commit is small and consists of
report/plan edits plus this plan file. No code changes are expected — any
that appear is a stop-and-ask.

## Audit of what §11 requires vs what the report has

Parent §11 ("Done when") for the report: *parity table per path, bench
numbers, deviations list (all-thread one-block tail, per-call scratch,
18a exclusion, `atomic_ref` outcome, double-math outcome).* Status of the
current §T5 (sections for steps 1–6 under the heading "in progress (step 6
of 7 done)"):

| §11 item | in report? |
|---|---|
| parity table per path | yes — step 4 section (per-path ctest table, all fixtures green per path) |
| bench numbers | yes — step 6 section (final median-of-3 table; step 4 provisional kept as history) |
| 18a exclusion | yes — step 4 section (fixture 18a excluded in writing, Phase B) |
| `atomic_ref` outcome | yes — step 1 section (P1 PASS) |
| double-math outcome | yes — step 1 section (P2 documented finding + P2b gate) |
| **all-thread one-block tail** | **no** — parent §4 documents the deviation (CUDA runs `sampled_tail_warp` on warp 0 only; SYCL barriers are work-group scope, so all 1024 threads run the tail — the Old kernel's shape, identical `pick`, thread 0 writes; FP64 cost ×32 vs CUDA, Phase B perf note). The report never states it. |
| **per-call scratch** | **no** — parent §4.4 documents it (CUDA's per-(device, stream) slot cache, 789–832, replaced by per-call `sycl::malloc_device` + `sycl::free` after the queue wait — one documented sync vs the slot cache; failure → one-block fallback, same as CUDA's `scratch == nullptr`). The report covers only the throw-vs-null fallback, not the caching deviation. |

## 7.1 — report §T5 final pass

1. **Add a compact `### T5 deviations (parent §11 list)` subsection at the
   end of §T5** (after the step-6 section), listing all five deviations in
   one place — the two missing ones written from the parent §4/§4.4 text
   and **verified against `poc/sycl/kernels/sampler.cpp` before writing**
   (the all-1024-threads tail in `submit_sample_one_block`; the
   per-call `sycl::malloc_device`/`sycl::free` in the dispatcher), the
   three already-covered ones as one-line pointers to their step sections.
   This satisfies §11's "deviations list" as a single auditable place
   instead of scattered mentions.
2. **Finalise the §T5 heading**: `**in progress** (step 6 of 7 done)` →
   `**done**`.

Stop-and-ask: if either of the two missing deviations does not match the
kernel as parent §4 describes it (i.e. the code differs from both the
parent text and my summary), stop and record what is actually there — do
not write the parent text into the report on faith.

## 7.2 — parent plan sync

- `plans/sycl-phase-a-t5.md` §11: all five boxes → `[x]` (probes, kernel +
  driver + checker, ctest ×3 + bench + regressions, mutations M1–M3, report
  + commit — the commit box is ticked as part of 7.4 in the same step).
- `plans/sycl-phase-a-t5-steps.md`: Step 7 heading marked **done** with a
  short result block (commit hash, final suite state).

## 7.3 — commit gate proof (before committing)

1. `env -u LD_LIBRARY_PATH ctest` → **19/19**.
2. `bash poc/sycl/check_mirrors.sh` → exit 0 (259 OK).
3. `git status --short` → exactly the step-7 file set below and nothing
   else (no stray build output, no `src/` touch).

No commit while the proof is not green.

## 7.4 — commit

Stage exactly (no `git add -A` — the tree is clean, so a targeted add is
both safe and the audit):

```
plans/sycl-phase-a-report.md         (deviations subsection, heading final)
plans/sycl-phase-a-t5.md             (§11 boxes)
plans/sycl-phase-a-t5-steps.md       (step 7 done + result)
plans/sycl-phase-a-t5-step-7.md      (this plan)
```

Message: `phase a t5` (parent §11: "Committed as `phase a t5`"). After the
commit: `git status --short` empty, `git show --stat HEAD` shows the four
files, `git log --oneline -2` shows `phase a t5` on top of `t5 step 6
complete`.

## Risks / stop-and-ask

- **Report accuracy**: the deviations subsection describes code that
  already exists; any mismatch between the parent §4 text and the kernel
  source is a finding to record, not a reason to make the report match the
  parent text (Step 5's M1 lesson: the parent table was wrong once).
- **Unexpected dirty files**: if `git status` shows anything outside the
  four files (e.g. an accidental `src/` edit), stop and investigate —
  the main tree is never touched, and nothing else should be modified.
- **Flaky suite**: the known transient timeout/device-flake class
  (Steps 3/5 precedent) — rerun before recording; do not commit on a
  flake's red.

## Done when (Step 7 complete)

- [x] Deviations subsection added, both missing items verified against the
      kernel (one-block glue ~629:634, dispatcher glue ~1019:1043); §T5
      heading `**done**`. The Phase B deferral note (coupled-draft kernels,
      `stream_capturing`, slot cache) was also missing and is now written.
- [x] Parent §11 boxes all `[x]`; steps file step 7 marked done with
      result. (Commit hash recorded in `git log` rather than the steps file,
      to keep the tree clean after the commit.)
- [x] Gate proof green (19/19, checker exit 0) and clean `git status`.
- [x] Committed as `phase a t5` with exactly the four files; tree clean
      after.

Est: **part of 1 d** (the steps file's figure); realistically ~30 min —
no code work, two report additions, one small commit.
