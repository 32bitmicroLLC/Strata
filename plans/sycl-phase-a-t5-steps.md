# T5 — `sampler` port: execution steps

Parent plan: `plans/sycl-phase-a-t5.md`. This file breaks that plan into
concrete, ordered steps. Each step states what to do and what "done" means.
The deferral contingency from the parent plan is expected **not** to fire.

## Step 1 — Toolchain probes (`poc/sycl/t5/`) (est 0.5 d)

Three small gate programs, built into `poc/sycl/CMakeLists.txt` as
standalone exes + ctest entries (following the T3 `t3_sg_probe` pattern).
All use `strata_launch` and the T3 handler-`local_accessor` pattern.

- **1.1 P1 — `t5/atomic_probe.cpp`: `atomic_ref` on local memory.**
  `sycl::atomic_ref<unsigned>` with `sycl::memory_scope_work_group`,
  `fetch_or` over a local accessor from many threads; compare to the
  serial OR of the same bits (validates the penalty-bitmap parallel build
  at sampler.cu 129/219/455/593).
  - Gate: works → bitmap keeps CUDA parallel-build structure.
  - Fallback: thread 0 ORs all `hlen ≤ 1024` history entries serially after
    the zero-fill barrier (correctness-identical, µs-scale). Document which
    one is used.
- **1.2 P2 — `t5/math_probe.cpp`: device double `exp`/`logf` vs host.**
  Grid of all fp32 values in [−745, 0] (plus fixture-16 NaN/±inf specials)
  through device `exp((double)x)` vs host `std::exp((double)x)`; device
  `logf` vs `(float) std::log` on min_p values {0.05, 0.3, 0.5, 0.9}.
  - Gate: bit-identical → document bit-exactness.
  - Fallback ladder: (a) a handful of non-fixture-arg differences →
    document the exact affected set; (b) else hand-roll a
    correctly-rounded double `exp` for fp32-representable args
    (range-reduce + double Taylor, ~100 lines); (c) a surviving fixture
    knife-edge is a written report finding, not a tolerance.
- **1.3 P3 — `t5/tree_probe.cpp`: 1024-wide local-memory reduction.**
  1024 threads each hold a (value, index) candidate; local memory +
  barrier + 10 halving levels in the §4.2 selection order; compare winner
  to a host serial scan, including a deliberate tie pair.
  - Gate: winner matches serial scan (lowest-index tie rule) → pattern
    approved at Arc's max work-group.

Done when: all three probes green (or fallbacks adopted and documented),
and the P1/P2 outcomes are noted — they decide bitmap build and double-math
strategy for Steps 2–3.

## Step 2 — Kernel part 1: shared helpers + greedy + old (est 1 d) — **done**

Outcome: `poc/sycl/kernels/sampler.cpp` all five `submit_*` + dispatcher
compiling green in the `k_sampler` static lib; all 75 mirror blocks green;
`k_sampler_smoke` ctest green (all four paths vs host serial reference);
full suite 15/15. One real glue bug found and fixed on the way: the
`warp_first` shared-memory butterfly was missing its per-stage write-back
(CUDA's `__shfl_xor` updates registers in place) — it corrupted only the
split path's top-k lists; see report §T5 step 2.

Create `poc/sycl/kernels/sampler.cpp` (one `submit_*` per CUDA kernel,
`sycl::event`-returning, no internal sync — the driver owns the wait).
Mirror blocks use the §5.1 format with the `tid`/`lane`/`warp` glue
convention.

- **2.1 Shared device helpers (§4.1):** `philox4x32_round` (verbatim except
  the two `__umulhi` lines → `(uint32_t)(((uint64_t) a * b) >> 32)`),
  `philox_uniform`, `history_count`, `apply_penalties` (verbatim — the
  semantic core), `sampled_k`, `take_first`; constants
  `kSelMax`/`kFullMask`. `warp_first` and the two-stage block reductions
  are replaced per §4.2.
- **2.2 `submit_greedy` (CUDA 100–167, work-group 1024):** §4.2 block
  argmax replacement (per-thread strided scan → local pair array → barrier
  → 10 halving levels with the mirrored compare → thread-0
  sentinel→0 write), §4.3 penalty bitmap (local accessor, ≤32 KiB; P1's
  `atomic_ref` or thread-0 serial build), scan loop with the
  `n_vocab`-sentinel (verbatim), T2 bounds check for padded items.
- **2.3 `submit_old` (187–317, work-group 1024):** per-round two reduction
  stages; `taken` sweep (246–248 verbatim); the all-thread double tail
  (278–316) fully verbatim (pure per-thread FP64, no warp intrinsics);
  `sel_ids`/`sel_logit` as handler-built local accessors; thread-0 `out[t]`
  write.
- **2.4 Register the `k_sampler` static lib in CMake (`kernels/sampler.cpp`).**

Done when: file compiles under `icpx -fsycl`, mirror blocks added,
`check_mirrors.sh` green for every block so far.

## Step 3 — Kernel part 2: one-block + split stages + dispatcher (est 1 d) — **done**

- **3.1 `submit_one_block` (433–504, work-group 1024):** rounds with the
  prev-threshold candidate test (477 verbatim); **documented deviation** —
  all 1024 threads run the tail (CUDA ran it on warp 0 only; early return
  would strand peers across the work-group barrier). Tail: `ex[]` scratch,
  `__syncwarp` → group barriers, `__shfl_sync` broadcasts →
  lane-0-writes/local-reads; mirrored arithmetic verbatim.
- **3.2 `submit_split_part` (568–647, work-group 128 = 4 × 32 regions):**
  per-region register loads of 1,024 penalised logits (600–606 verbatim
  modulo the `__int_as_float(-inf)` glue line), `k` rounds of two-chain
  in-register argmax + per-region 32-entry local-memory butterfly
  replacing `warp_first` (region-local 32-pair slots; group barriers keep
  the 4 regions in lockstep); region 0 merges the 4 lists via
  local-memory `warp_merge_lists` (§4.6) into the block `cand` list;
  `wl` + block bitmap as handler accessors. Launch padded 1D:
  `strata_launch(q, n_blocks × n_tokens × 128, 128, …)` with
  `b = (item/128) % n_blocks`, `t = (item/128) / n_blocks`, T2 bounds
  check on `t`.
- **3.3 `submit_split_merge` (651–668, work-group 32):** copy the row's
  ≤64 lists into a local accessor (≤32 KiB + ~1.3 KiB tail scratch), run
  `warp_merge_lists` (tid owns lists `tid` and `tid+32`), then the tail —
  all 32 threads participate (is CUDA's warp-only shape). Use
  `struct I2 { int x; int y; }` instead of `int2`.
- **3.4 Host dispatcher `submit_sample_tokens` (mirrors 836–897):**
  `SampledPath`/`env_flag`/`sampled_path` logic verbatim (753–767);
  fallback conditions `n_blocks <= kSplitMaxBlocks && n_tokens <=
  kSplitMaxRows` verbatim except the `stream_capturing` clause (Phase B).
  Documented deviations: no `stream_capturing` check; scratch =
  per-call `sycl::malloc_device(n_tokens × n_blocks × kSelMax ×
  sizeof(I2))`, freed after a queue wait (one documented sync vs CUDA's
  slot cache); alloc failure → one-block fallback; missing-history-with-
  penalties `exit(1)` → throw (T3 library rule).
- **3.5 Local-memory budget check (§4.5):** verify the table still holds
  (max consumer ~34 KiB vs 128 KiB cap) → no opt-in machinery; parent
  plan's deferral trigger does not fire.

Done when: full kernel file compiles, `check_mirrors.sh` green on all §5.1
blocks (~30).

## Step 4 — Driver (est 1.5 d) — **done**

Create `poc/sycl/drivers/sampler_parity.cpp` mirroring
`src/kernels/sampler_parity.cpp` per §5.2. Host references are pure C++ —
mirror them verbatim, never re-derive:

- **4.1 Reference/compare machinery:** `reference_pick`, `reference_cut`,
  the `run` compare, `PhiloxRound`, `host_philox_uniform`,
  `sampled_reference`, `sampled_cut`, `SelList`, `mirror_select`,
  `mirror_pick`, `sampled_k`, `window_of`, summary/exit contract.
  `cudaMemset(d_o, 0xFF)` → host-fill `-1` + copy; all CUDA alloc/copy/
  sync/free → `ctx` glue; `DeviceRows` and `bench_sampled` are glue shells
  keeping their printf/loop bodies.
- **4.2 The 18 fixtures** (main, from line 436), mirrored block-per-block
  at their section boundaries (landmarks per §5.2): data generation,
  reference calls, observability asserts, verdict prints verbatim.
  - **Excluded:** fixture 18a (graph-capture fallback, ~1240–1260) —
    written exclusion per §2; 18b (row-cap fallback: 64 rows split,
    70 rows one-block) ports in full.
  - **Streams:** `run`'s `nullptr` and fixture 17's `{nullptr, cs}` both
    map to `ctx.q`; document in the driver header that no fixture
    assertion depends on stream identity.
  - The order-observable fixtures (2, 9, 10, 16, 17) must survive with
    their observability asserts intact — a wrong sampler order returns a
    valid token, so "it produced a token" proves nothing.
- **4.3 Parity contract checks (§6):** picks are integers with exact
  equality (no tolerance); Philox draw pinned exactly (counter-based);
  every fixture keeps its observability assert verbatim.
- **4.4 CMake wiring (§9):** `k_sampler_parity` exe linking `k_sampler`;
  three ctest path runs — `k_sampler_parity` (default/split),
  `k_sampler_parity_one_block` (`STRATA_SAMPLER_ONE_BLOCK=1`),
  `k_sampler_parity_old` (`STRATA_OLD_SAMPLER=1`) — each with the
  ENVIRONMENT property and TIMEOUT 300; `k_sampler_bench`
  (`--bench`, TIMEOUT 300).

Done when: `check_mirrors.sh` green on the ~45 driver blocks (~155 total in
the tree after T5), and all three path ctests run (greenness is Step 5).

Result (implemented reality): the driver landed with 72 mirror blocks
(the plan's ~45 undercounted the per-split host lines; the tree now holds
259 OK blocks). All three path-pinned ctests are green, not merely
running — default/split 2.3 s, one_block 2.7 s, old 5.0 s — and
`k_sampler_bench` runs (numbers recorded in Step 6). Two recipe bugs
caught by the compiler, not the checker (both mirrored ranges had
swallowed a CUDA line: f3's two `check(cudaMemcpy…)` and f17's
`DeviceRows` ctor) — fixed before first green run. Records: report §T5
step-4 section.

## Step 5 — Mutation tests (est part of 1 d) — **done**

Three mutations, each targeting a fixture designed to see it;
`check_mirrors.sh` going red while a mutation is in is expected (as in
T2–T4). Revert each mutation and re-run the full 3-path suite before
commit.

| # | mutation (kernel) | designed victim | expected |
|---|---|---|---|
| M1 | `take_first` tie: `oi < bi` → `oi > bi` | fixtures 16/17 + greedy ties | red on the tie fixtures |
| M2 | tail order: min_p before top_p (pre-#53 bug) | fixture 10 (token 2 must be drawn) | red: `twos == 0` |
| M3 | `apply_penalties`: divide unconditionally | fixture 4A (multiply-rule observable) | red: `A_visible` pick wrong |

Done when: each mutation reds its designed fixture; restore → all green.

Result (implemented reality): all three mutations red their designed
fixtures on **all three** path-pinned ctest entries, each with the
checker red exactly on the touched blocks, and the restored suite is
19/19 with 259 mirror blocks OK. M1 needed more sites than the table
states: `take_first` alone reds only the split path — the 1,024-thread
kernels carry independent tie logic in `fold_block` and their per-thread
argmax loops, so the flip was extended to all six tie sites (recorded
in `plans/sycl-phase-a-t5-step-5.md` and report §T5).

## Step 6 — Bench + full regression (est part of 1 d) — **done**

- **6.1 Bench (§8):** `--bench` mirrors `bench_sampled` (driver 391–434)
  minus CUDA events: fixed N(0,3) logits at 248,320 vocab, T ∈ {1,4,8},
  k ∈ {20,64}, top_p 0.95, temperature 0.7, 3 warm-ups + 50 timed calls,
  chrono wall-clock around `sample_tokens` + `ctx.wait_and_throw()`, same
  `us per call` printf format. Default path (split). No CUDA comparison
  exists in this checkout — these are the first Arc sampler figures.
- **6.2 Full regression:** ctest over T5 probes + `k_sampler_parity` ×3
  paths + `k_sampler_bench` + all T0–T4 tests green, running under
  `env -u LD_LIBRARY_PATH`.

Done when: the full suite is green and the bench numbers are recorded.

Result: the audit found one real gap — the `k_sampler_bench` ctest entry
had no `ENVIRONMENT` pin, so it measured whichever path the caller's shell
selected; it now pins the default (split) path, and every recorded run
prints `sampled path: split top_k (default)`. Final §8 table (median of
three runs, ≤ 5 % spread): 165.8/477.0/101.5/311.0/174.0/540.8 us per call
over (rows, top_k) = (1,20)/(1,64)/(4,20)/(4,64)/(8,20)/(8,64). Full suite
19/19, `check_mirrors.sh` exit 0 (259 OK). Details in the report §T5
step-6 section.

## Step 7 — Report §T5 + commit (est part of 1 d) — **done**

- **7.1 Report** (`plans/sycl-phase-a-report.md` §T5): parity table per
  path, bench numbers, and the deviations list:
  - all-thread one-block tail (vs CUDA's warp-0-only),
  - per-call scratch allocation (vs the slot cache),
  - fixture 18a exclusion (graph capture),
  - P1 `atomic_ref` outcome,
  - P2 double-math outcome,
  - plus: coupled-draft kernels / `stream_capturing` / slot cache deferred
    to Phase B with reasons (§2).
- **7.2 Verify §11 done-when checklist** in the parent plan, all boxes ticked.
- **7.3 Commit** as `phase a t5`.

Done when: report updated, checklist satisfied, commit made.

Result: report §T5 gained one auditable deviations subsection — the two
items missing from it (all-thread one-block tail, per-call scratch)
verified against `poc/sycl/kernels/sampler.cpp` before writing, plus the
Phase B deferral note (coupled-draft kernels, `stream_capturing`, slot
cache). §T5 heading `**done**`; all six parent §11 boxes ticked. Gate
proof: **19/19** under `env -u LD_LIBRARY_PATH`, `check_mirrors.sh` exit 0
(259 OK), clean tree. Committed as `phase a t5` (the hash is in
`git log`, on top of `t5 step 6 complete`).

## Stop conditions (deferral branch)

If any of these fires, stop and write the deferral per the parent plan
(exact missing pieces, so Phase B is not surprised) — gates 1–4 must stay
satisfiable with the other three kernels:

- P2 leaves a fixture knife-edge even after the fallback ladder (report
  finding on that fixture — never a silent tolerance).
- Any consumer needs local memory beyond the 128 KiB cap (none predicted;
  §4.5 shows 4× margin).
- Any in-scope kernel requires machinery outside Phase A (graph capture,
  mapped memory, subgroup API — none predicted).

## Total

~5 d (Steps 1–7), inside the parent plan's ~1 week envelope for T5.
