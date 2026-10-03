# T5 step 5 — mutation tests M1–M3

Mutation-proof the Step 4 parity driver, per the steps file's Step 5 and
parent plan §7. Three kernel mutations, each targeting a fixture designed
to see it; revert each, restore green. The doctrine (T2–T4): a mutation
that no fixture catches is a hole in the suite; `check_mirrors.sh` going
red while a mutation is in is **expected** — the edited lines are mirrored
kernel lines.

## Ground rules

- **Only `poc/sycl/kernels/sampler.cpp` is mutated.** The driver
  (`poc/sycl/drivers/sampler_parity.cpp`), the host references, and
  `src/` stay untouched — the host reference is what makes a red
  meaningful.
- **One mutation in at a time.** Apply → build → run the three path
  ctests → record the red → `git checkout -- poc/sycl/kernels/sampler.cpp`
  → build. Overlapping mutations would muddy attribution.
- The kernel file is clean at HEAD before starting (verified: it is not in
  `git status`), so `git checkout` reverts exactly the mutation and
  nothing else. After the step, `git diff` on the kernel must be empty.
- `icpx` intermittently exits 3 — retry the build. Transient
  `UR_RESULT_ERROR_DEVICE_LOST` — retry the run (Steps 2/3 precedent).

## The mutations (all sites pinned in the current kernel)

| # | site (SYCL file) | edit | designed victim | checker while in |
|---|---|---|---|---|
| M1 | **all six tie sites** (the plan's single `take_first` site proved insufficient — see the M1 section): `take_first` 188 (mirror 338:338), `fold_block` 295 (148:148), greedy per-thread (142:143), old per-thread (246:250), one_block per-thread (476:477), split_part chains (621:632) | every tie comparison flipped to break ties to the highest id | fixtures 16/17/18b (tie lists, tied draws) | red on the six tie-site blocks |
| M2 | the two 1,024-thread tails — mirror 278:315 in `submit_sample_old` (~493) and `submit_sample_one_block` (~640) | min_p cut inserted before the top_p cut; the top_p cut edited to run over the pre-cut count | fixture 10 (token 2 must be drawn) | red on both 278:315 blocks |
| M3 | `apply_penalties`, lines 158:159 (mirror 74:78) | multiply/divide branch → divide unconditionally | fixture 4A (multiply-rule observable) | red on that one block |

### M1 — `take_first` tie flip, extended to all six tie sites

The plan's first draft assumed `take_first` (line 188) was the single
shared tournament helper every selection goes through. The first run
disproved it: with only `take_first` flipped, **only the split path reded**
(`k_sampler_parity`), while the forced one_block/old entries stayed green.
The 1,024-thread kernels fold through `fold_block` (line 295, its own
mirror block 148:148), and each kernel's per-thread argmax carries its own
inlined tie logic (greedy 142:143, old 246:250, one_block 476:477,
split_part's two chains 621:632). The semantic mutation "break ties to the
highest id instead of the lowest" therefore required flipping **all six
sites**, which is what the step ran:

- **take_first (188):** `oi < bi` → `oi > bi`
- **fold_block (295):** `oi < best` → `oi > best`
- **greedy per-thread (142:143):** `if (s > bv)` →
  `if (s > bv || (s == bv && v > best))`
- **old per-thread (246:250):** the same flip as greedy
- **one_block per-thread (476:477):** eligibility `v > prev_i` →
  `v < prev_i` (the previous pick consumed the *highest* tied id, not the
  lowest) plus the same `s > bv` flip
- **split_part chains (621:632):** eligibility `v0 > prev_i` →
  `v0 < prev_i` and `x0 > b0` → `x0 > b0 || (x0 == b0 && v0 > i0)`; the
  two chain winners are then ordered by the flipped `take_first`

Why it reds: the host `mirror_select` breaks ties to the **lowest** id
(ascending scan, strict `>`); the flipped kernel keeps the **highest** id
among equals at every site, so every tied group's list order reverses
against the host list and a draw among tied candidates lands on a
different token. The tie fixtures are exactly 16 (0.5-quantised
1,240/1,260-vocab rows, 1,632 draws / 1,057 tied picks), 17 (512-vocab
`floor(v/2)` rows), 18b (512-vocab 0.5-quantised rows, `T ∈ {64, 70}` →
split and one-block fallback). All three appear in every ctest entry: 16/
17 take the split path under `k_sampler_parity` and the 1,024-thread
kernels under the forced entries; 18b is split at `T = 64` and one-block
at `T = 70`.

**Observed:** exit 1 on all three entries, red on exactly those fixtures
and nothing else — #13/#14: 1,325/1,340 list positions, 1,349/1,632 draws,
119/134 fallback draws (2,793 failures each); #15: 1,025/1,340 positions,
same draw counts (2,493 failures). Checker red on exactly the six blocks
above. (Intermediate run, before the extension: #13 alone red, 2,793
failures — the evidence that forced the extension.)

Coverage note (recorded, no action): the parent table's "+ greedy ties"
has **no dedicated fixture** — no fixture feeds tied logits to the greedy
path. M1's designed victims are 16/17/18b; the greedy kernel's tie branch
is the same `take_first`, so it is covered structurally, not by a fixture.

### M2 — min_p before top_p (the pre-#53 order)

Both serial (1,024-thread) tail instances are mutated — the historical bug
lived in the serial `sampler_kernel`, and fixture 10 (256 rows) runs on a
serial kernel on **every** path: one-block under `k_sampler_parity` and
`k_sampler_parity_one_block`, the old kernel under `k_sampler_parity_old`.
Mutating only one tail would leave the old entry green and the mutation
half-unproven.

For each of the two instances (identical text at ~493 and ~640):

1. **Insert glue** after the `const float inv_t = …` line, before the
   `// SYCL-MIRROR-BEGIN … 278:315` block:

   ```cpp
            // M2 mutation (pre-#53 order): min_p before top_p.  n_pre is the
            // min_p prefix of the top_k list; the top_p cut below runs over n_pre.
            int n_pre = k;
            if (p.min_p > 0.0f) {
                const float thresh = sel_logit[0] + logf(p.min_p);
                for (int i = 0; i < n_pre; ++i)
                    if (sel_logit[i] < thresh) { n_pre = i; break; }
            }
   ```

2. **Edit the mirror block** (3 lines, both instances):
   - `        int cut = k;` → `        int cut = n_pre;`
   - the sum loop `        for (int i = 0; i < k; ++i) sum += …` → `i < n_pre`
   - the cum loop `        for (int i = 0; i < k; ++i) { cum += …` → `i < n_pre`

The **later** min_p block (inside the same mirror) stays untouched and is a
proven no-op under the mutation: it keeps the descending prefix with
`sel_logit[i] >= sel_logit[0] + logf(min_p)`, and the pre-cut already
guaranteed that for every index the top_p cut can reach (`n_keep <= n_pre`).

Why it reds: fixture 10's row is p = 0.4/0.3/0.2/0.1, top_p 0.75, min_p
0.3. Correct order: top_p keeps three (0.4 + 0.3 < 0.75 <= 0.9), min_p
(keeps p >= 0.12) keeps all three → the reference draws token 2 ~22 % of
the time (`twos > 0`, host-side). Mutated order: min_p drops the 0.1 token
first, renormalises to 0.444/0.333/0.222, top_p then stops at two → the
kernel draws only {0, 1}.

**Observed:** "sampled chain: top_p then min_p *** WRONG *** (91 of 256
differ)" on all three entries; the kernel never drew token 2. 91 matches
the shared-Philox-u arithmetic exactly: both sides draw with the *same*
u, and the picks differ on u ∈ [0.444, 0.571) (kernel 0 vs host 1) and
u ∈ [0.777, 1) (kernel 1 vs host 2) — 35 % of rows, ~90 of 256 (the
draft's "~57" had forgotten the shared u). (The parent table's "red:
`twos == 0`" is shorthand: `twos` is counted on the *host* reference,
which the mutation does not touch — the red is the `run` mismatch, while
the observability print still says "yes".)

**Incidental (predicted in this plan, confirmed):** fixture 17 reds
2/1,632 draws on the two forced entries — its `min_p = 0.05` configs fire
the pre-cut on the mutated serial tails. The default entry stays clean on
17 because its split tail was not mutated.

Coverage note (recorded, no action): the **split_merge** 32-lane tail
(~864:905) implements the same order but is **not** mutated. No designed
fixture reaches it with the cut firing: fixture 10's 256 rows never take
the split path, and fixture 17's `min_p = 0.05` configs may or may not
fire the cut on a 64-entry list of N(0,3) logits. The split tail's order
is behaviourally verified (fixture 17 matches the host reference, which
applies top_p → min_p), which is weaker than a mutation test — written
here rather than smoothed over.

### M3 — divide unconditionally (the pre-sign-fix reading)

- **Before (158:159):**
  ```cpp
      if (logit <= 0.0f) logit *= p.penalty_repeat;
      else               logit /= p.penalty_repeat;
  ```
- **After:**
  ```cpp
      logit /= p.penalty_repeat;   // M3: divide unconditionally
  ```

The `if (count <= 0) return logit;` guard and the freq/present line stay:
tokens without history are unaffected, tokens with history lose the
negative-multiply rule. `apply_penalties` is shared by every kernel, so one
edit covers all three ctest paths (fixture 4A is greedy, and greedy
dispatches to the greedy kernel under every path).

Why it reds: fixture 4A — all-negative row, token 0 at -1.0 (in the
history), token 1 at -1.2, repeat 2.0. Correct rule: -1.0 **× 2.0** =
-2.0 → argmax token 1 (`want_a = 1`). Mutation: -1.0 / 2.0 = -0.5 beats
-1.2 → kernel picks 0 on both rows → `run` reds 2/2, exit 1 on all three
entries.

**Observed:** "penalties: repeat on negatives *** WRONG *** (2 of 2
differ)  first: want 1 got 0" on all three entries, 2 failures each. The
`A_visible` print is host-side and unchanged ("yes"); the red is the
comparison.

Incidental reds (expected, corroborating, do not stop the step): other
penalty fixtures with negative history logits *would* sample the same
rule — but **none occurred**, and the reason is worth keeping: only 4A
puts a negative history logit inside the candidate set. Fixture 17's
`pen = 1` configs carry negative random history tokens, but a negative
logit cannot reach a top-20/top-64 list of 248,320 or 512 N(0,3) logits;
the window fixtures (6/7/8) and the per-row fixtures (12/13) punish
positive or unlisted tokens. Fixture 4B (presence, all-positive history
logits) is unaffected — divide == divide there.

## Per-mutation protocol

1. `git status --short poc/sycl/kernels/sampler.cpp` → empty.
2. Apply the edit(s).
3. `cmake --build . -j` (retry on icpx exit 3).
4. `env -u LD_LIBRARY_PATH ctest -R "^k_sampler_parity"` — all three
   entries expected **Failed**; capture the red fixture lines (first
   mismatch detail line per fixture is enough for the report).
5. `bash check_mirrors.sh` — expected to fail, red **exactly** on the
   blocks the mutation touched (M1: six; M2: two; M3: one). Any *other*
   red block means the edit drifted beyond the recipe — stop and fix.
6. `git checkout -- poc/sycl/kernels/sampler.cpp`; rebuild.
7. After M3's revert only: full `env -u LD_LIBRARY_PATH ctest` (19/19) +
   `check_mirrors.sh` exit 0 (259 OK) — the restored-suite proof.

Stop-and-ask conditions: M1 reds a fixture outside {16, 17, 18b} (a tie
path that does not go through `take_first`); M2 does not red fixture 10 on
the old entry (the dispatcher routed it elsewhere); and the standing rule
from Step 4 applies: a red here is a caught bug, never a pass to be
waived. (The first condition *fired in practice*, but in the opposite
direction: M1 taken literally — `take_first` only — left the forced
entries green, exposing that the 1,024-thread kernels carry independent
tie sites. Resolved within the step by extending M1 to all six sites;
recorded in the M1 section and the report.)

## Risks

- **Blade-width of M1:** it is the shared helper, so the blast radius is the
  whole kernel. The stop-and-ask condition above is the guard.
- **M2's no-op claim** (the later min_p block) rests on `n_keep <= n_pre`
  after the mutated top_p — if a fixture ever shows the later block still
  cutting, the analysis above is wrong and the edit recipe needs
  revisiting (the red pattern would show it).
- **Build/run flakiness** (icpx exit 3, transient device-lost): retry; do
  not attribute a red to the mutation before the build is confirmed clean
  and the suite ran.

## Done when (Step 5 complete)

- [x] M1 in (all six tie sites): fixtures 16/17/18b red on all three path
      ctest entries, no other fixture reds; checker red exactly on the six
      tie-site blocks; reverted.
- [x] M2 in: fixture 10 red (91/256) on all three path ctest entries;
      checker red exactly on the two 278:315 blocks; reverted.
- [x] M3 in: fixture 4A red (2/2 differ) on all three path ctest entries,
      no incidental reds; checker red exactly on the 74:78 block;
      reverted.
- [x] Restored: kernel `git diff` empty, full suite 19/19 green (18.25 s),
      `check_mirrors.sh` exit 0 (259 OK). One transient incident: the first
      post-revert ctest run timed out test 13 at 300 s; the direct rerun
      was 2.3 s green and the full suite green — recorded as transient
      device flakiness (Steps 2/3 precedent), not a code change.
- [x] Records: report §T5 step-5 section (per-mutation table: site,
      designed victim, observed red lines, incidental reds, checker state),
      steps file marked done, this plan's boxes ticked.

Est: **0.5 d** (the steps file's "part of 1 d").
