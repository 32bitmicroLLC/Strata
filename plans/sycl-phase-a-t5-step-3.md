# T5 Step 3 — Kernel part 2: one-block + split stages + dispatcher — verification

Parents: `plans/sycl-phase-a-t5.md` §4 (port design) and
`plans/sycl-phase-a-t5-steps.md` Step 3 (est. 1 day). Step 2
(`plans/sycl-phase-a-t5-step-2.md`) is complete: all five `submit_*`
functions plus the host dispatcher exist in `poc/sycl/kernels/sampler.cpp`,
compile green in the `k_sampler` static library, all 75 mirror blocks are
green, and `k_sampler_smoke` passes (all four paths vs a host serial
reference on 64 rows × 512 vocab).

Because the Step 2 draft was complete (all of Steps 2 *and* 3 in one TU),
this step is **not** a writing step: it verifies the Step-3-shaped code —
`submit_one_block`, `submit_split_part`, `submit_split_merge`, and the
dispatcher — against the cases the smoke has not yet exercised, fixes the
one glue gap found in audit (scratch-alloc failure), and closes the
local-memory budget check (3.5). No mirrored line may change; every kernel
edit in this step is glue, and the expectation is *very* few of them.

## Starting state — what is verified and what is not

Verified by the Step 2 smoke (`k_sampler_smoke`): all four paths launch
and produce correct picks on 64 rows × 512 vocab with penalties on —
greedy 0/64 vs host serial argmax; split, old, one_block each 0/64 vs the
host serial reference. That single shape covers `n_blocks == 1`,
`n_tokens == kSplitMaxRows` exactly, and the default (split) path. The
`warp_first` butterfly write-back bug (report §T5 step 2) was caught there.

Not yet exercised, found by auditing the shipped code:

| # | gap | evidence |
|---|---|---|
| G1 | **Multi-block split untested.** Every smoke row has `n_blocks == 1`; the split path's interesting machinery — the `(t, b)` item mapping over a 2-D block grid, the per-(row, block) bitmap, and `split_merge` merging *several* block lists into one row list — has never run. | smoke uses 512 vocab; `n_blocks = ceil(512/4096) = 1` |
| G2 | **Neither dispatcher fallback has been taken.** The conditions `n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows` (dispatcher, mirrored 862:864 + glue) have only ever been true, and `STRATA_*` env fallback only via explicit env. | no smoke shape exceeds 64 rows or 262,144 vocab |
| G3 | **Scratch-alloc failure does not fall back to one-block.** `sycl::malloc_device` *throws* on failure (it never returns null), so a failed alloc propagates out of `submit_sample_tokens` instead of taking the mirrored `scratch == nullptr` → one-block branch that CUDA's failed `cudaMalloc` takes. | dispatcher glue, `scratch = sycl::malloc_device<int2>(...)` with no try/catch |
| G4 | **Bad-argument behaviour is inconsistent.** Missing history with penalties throws (correct, T3 rule), but `n_tokens <= 0 \|\| n_vocab <= 0` returns an empty event while the header says bad arguments throw. Either the code or the header must change. | dispatcher, first glue block |
| G5 | **`temperature <= 0` greedy dispatch untested** (mirrored 847: `p.greedy \|\| p.temperature <= 0.0f` → greedy kernel). | no smoke scenario sets `temperature = 0` |
| G6 | **Local-memory budget is unproven at the worst shape.** The budget claim (steps 3.5) has never been run at vocab 248,320 (max bitmap), only reasoned about. | 3.5 below |
| G7 | **Degenerate shapes untested** (tiny vocab, single row). | — |

Launch-range audit result (steps 3.2's "T2 bounds check"): all five
submitters use exact `nd_range<1>(n_tokens * …, G)` launches — the item
count is always a work-group multiple, so padded items cannot exist and
the kernels' omitted `t >= n_tokens` guards are safe. The header's
"struct 1-D strata_launch-shaped launch" wording should be corrected to
say *exact* launches (doc-only change); no runtime guards are added.

## Work items

### 3.1 `submit_one_block` — ratify, no code change

The shipped kernel matches the steps file except in the documented,
already-headered ways: all 1,024 threads run the tail (CUDA ran it on
warp 0; an early return would strand peers across the work-group
barriers) and the `__shfl_sync` broadcasts are lane-0-writes / local
reads. The header's STRUCTURE paragraph already states this. Action:
confirm the header wording matches the code line-for-line intent, and let
the scenario matrix (3.6) carry the behavioural proof — one_block is the
fallback target of scenarios C/D and is also run explicitly.

### 3.2 `submit_split_part` — ratify exact launch; prove multi-block

No code change expected. The exact-launch / no-guard decision (gap
audit above) gets the header wording fixed ("1-D strata_launch-shaped
launch" → "1-D launches with exact `nd_range` item counts — always a
work-group multiple, no padded items, so the CUDA `t >= n_tokens` guards
are omitted"). Behavioural proof is scenario B (3.6): 12,288 vocab →
`n_blocks = 3`, 16 rows — exercises the `(t, b)` mapping, three per-block
bitmaps, and three warp lists per row feeding the merge.

### 3.3 `submit_split_merge` — ratify; prove the multi-list merge

No code change expected. The 32-thread row work-group copies the row's
`n_blocks` lists (always the full 64×64 `lists` accessor, 32 KiB — the
budget counts the worst case, 3.5), merges them (tid owns lists `tid`
and `tid+32`), then runs the tail on all 32 threads. Scenario B is the
behavioural proof: with `n_blocks = 3` the merge consumes three distinct
block lists whose top-8 union is *not* any single block's list, so a
broken merge (the class of bug that killed `warp_first`) would show as
wrong picks even if each block's list were right. Scenario D runs the
one-block fallback *instead* of this kernel, so its `n_blocks <= 64`
boundary is covered from both sides by B (3 blocks) and D (65 blocks →
not this kernel).

### 3.4 Host dispatcher — two glue fixes, then verify

1. **Scratch-alloc failure → one-block fallback (G3).** Wrap the
   `sycl::malloc_device<int2>` call in `try { … } catch (const
   std::exception&) { scratch = nullptr; }` so a failed allocation takes
   the mirrored `scratch == nullptr` branch — the same behaviour CUDA's
   failed `cudaMalloc` produces. (No test can force the failure on this
   machine; the fix is verified by inspection and the fact that the
   success path is unchanged and covered by the smoke.)
2. **Make the bad-argument behaviour uniform (G4).** Both
   `n_tokens <= 0 || n_vocab <= 0` and the missing-history case throw
   `std::runtime_error`, matching the header's contract ("throws
   `std::runtime_error` instead of exiting"). The empty-event return is
   deleted. Scenarios E (3.6) assert both throws.
3. Everything else in the dispatcher is already in place and is ratified,
   not rewritten: `sampled_path()` env cache (function-static — the
   reason the smoke forks), the `temperature <= 0` greedy clause
   (scenario E), the fallback condition (scenarios C/D), per-call scratch
   sized `max(n_tokens, 16) × n_blocks × kSelMax` int2 and freed in-order
   on the same queue (one documented sync vs CUDA's per-stream slot
   cache), and the deferred `stream_capturing` clause (Phase B).

### 3.5 Local-memory budget check

Computed from the shipped accessors (bytes, no alignment assumed; DPC++
may pad each accessor to 16 B — immaterial at this scale):

| kernel | work-group | local memory | size at worst fixture shape (vocab 248,320) |
|---|---|---|---|
| `submit_sample_greedy` | 1024 | `penal_bits` 7,760 words; `rv`/`ri` 1,024 each | 31,040 + 8,192 = **39,232 B** |
| `submit_sample_old` | 1024 | as greedy + `sel_ids`/`sel_logit` 64 each | 31,040 + 8,192 + 512 = **39,744 B** |
| `submit_sample_one_block` | 1024 | same layout as old | **39,744 B** |
| `submit_split_part` | 128 | per-block bitmap (128 words), `wl` 4×64 int2, `s_v`/`s_i` 128 each | 512 + 2,048 + 1,024 = **3,584 B** (vocab-independent) |
| `submit_split_merge` | 32 | `lists` 64×64 int2 (always full), `sel_ids`/`sel_logit`/`ex`, `s_sum`/`s_cut`, `s_v`/`s_i` | 32,768 + 256 + 256 + 512 + 12 + 256 = **34,060 B** (vocab-independent) |

Max consumer: **39,744 B ≈ 38.8 KiB** (old/one_block at vocab 248,320)
vs Arc's 128 KiB work-group cap — 31 %. The split kernels are bounded
independently of vocab by design (per-block bitmap; fixed list table).
**Conclusion: the parent plan's deferral trigger does not fire; no
opt-in/local-memory machinery is needed.** (The steps file's "~34 KiB"
figure was the split_merge consumer; the true max is 39,744 B.)

Runtime proof (scenario F): actually launch greedy and one_block at
vocab 248,320 and check the picks. If Arc rejected the work-group for
local-memory size the submit would throw — this is the check that turns
the table from arithmetic into evidence.

### 3.6 Extend `poc/sycl/t5/k_paths_smoke.cpp` into a scenario matrix

Refactor the existing smoke body into a scenario loop; every scenario's
sampled paths are judged against the same host serial reference already
in the file (k rounds of the penalised strict-argmax + the mirrored tail
with host math), and the forked children (needed because
`sampled_path()` caches its env read) are spawned per scenario that
requires them.

| scenario | shape | what it proves | paths run |
|---|---|---|---|
| A (existing) | 64 rows × 512 vocab | all four paths; default = split at the `kSplitMaxRows` boundary | split + greedy (parent), old + one_block (forks) |
| B | 16 rows × 12,288 vocab | **G1** — multi-block split: `n_blocks = 3`, `(t, b)` mapping, per-block bitmaps, multi-list merge in `split_merge` | split + greedy (parent), old (fork) |
| C | 100 rows × 512 vocab | **G2** — `n_tokens > kSplitMaxRows` fallback: default path must equal both the serial reference *and* the explicitly-forced one-block picks (behavioural proof the fallback was taken) | default (parent), one_block (fork) |
| D | 16 rows × 262,200 vocab | **G2** — `n_blocks = 65 > kSplitMaxBlocks` fallback (262,200 > 262,144): same double check | default (parent), one_block (fork) |
| E | host-side, no GPU work | **G4/G5** — missing history with `penalty_last_n > 0` throws; `n_tokens = 0` throws; `temperature = 0` dispatches to greedy (picks equal the argmax reference, not the sampled one); degenerate 1 row × 4 vocab returns token 0 on all paths | throws in-process; greedy + degenerate via the normal submits |
| F | 4 rows × 248,320 vocab | **G6** — the 39 KiB local-memory shapes actually launch on Arc (worst bitmap, 7,760 words, at 1,024 threads) | greedy + one_block (parent) |

Data generation stays the same style (fixed-seed `mt19937`,
`uniform_real_distribution<float>(-2, 2)`, repeated-token history windows
so penalties are exercised); scenario D's 1.05 MB of logits is trivial to
allocate and the host serial reference at that vocab is a fraction of a
second. The smoke keeps its PASS/FAIL single-exit-code contract; any
scenario failing is a red ctest.

CMake: `k_sampler_smoke` TIMEOUT rises 120 → 300 (five scenarios, five
forked SYCL runtimes — A×2, B, C, D — plus the in-process paths); no
other CMake change.

### 3.7 Record

- `plans/sycl-phase-a-t5-steps.md`: mark Step 3 done.
- `plans/sycl-phase-a-t5-report.md` §T5: step-3 outcome — the budget
  table with the 39,744 B max, the two dispatcher glue fixes (malloc
  fallback, uniform throws), the launch-range ratification, the scenario
  matrix results, and any surprise.
- This file: check the "Done when" boxes.

## CMake / run

```bash
cd poc/sycl/build && cmake . && cmake --build . -j
bash ../../check_mirrors.sh                 # must stay exit 0 (glue-only changes)
env -u LD_LIBRARY_PATH ctest -R "k_sampler_smoke" --output-on-failure
env -u LD_LIBRARY_PATH ctest                # full suite stays green
```

## Done when (Step 3 complete)

- [ ] Dispatcher glue fixed: scratch-alloc failure falls back to
      one-block; `n_tokens <= 0 || n_vocab <= 0` throws like the
      missing-history case (header and code agree).
- [ ] Header wording corrected: exact `nd_range` launches (no padded
      items), not "strata_launch-shaped".
- [ ] Local-memory budget table computed (max 39,744 B ≈ 38.8 KiB vs
      128 KiB) and proven at runtime by a greedy + one_block launch at
      vocab 248,320.
- [ ] Scenario matrix green in `k_sampler_smoke`: A (4 paths × 64 rows),
      B (multi-block split, `n_blocks = 3`), C (`n_tokens > 64` fallback),
      D (`n_blocks > 64` fallback), E (throws + `temperature = 0` +
      degenerate), F (worst-shape local memory).
- [ ] `check_mirrors.sh` still exits 0 (no mirrored line changed).
- [ ] Full-suite `ctest` green under `env -u LD_LIBRARY_PATH`.
- [ ] Steps file + report §T5 updated.

## Out of scope (later steps)

- Fixture-level parity of every path, including the tie/NaN/±inf fixtures
  (16/17) and the mutation-sensitive fixtures — Step 4's
  `k_sampler_parity` driver and its three path-pinned ctest entries, plus
  `k_sampler_bench`.
- `stream_capturing` detection, the per-stream scratch cache, fixture
  18a's captured-graph half, and the coupled draft kernels (670:751,
  891:933) — Phase B, as documented in the kernel header.
- The Step-5 mutation tests (M1–M3) — they run against the Step-4 driver.

## Estimate

1 day per the steps file: the code is already written and compiling; the
day goes to the two glue fixes, the scenario-matrix extension of the
smoke (the multi-block and fallback shapes are the new surface), the
budget proof, and the record.
