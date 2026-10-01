# T5 — `sampler` port + driver (the heaviest kernel)

Parent: `plans/sycl-phase-a.md` (task T5). T0–T4 are done (PASS —
`plans/sycl-phase-a-report.md`). T5 is the largest port in Phase A: **933
lines of CUDA kernel** (`src/kernels/cuda/sampler.cu`) and a **1285-line
parity driver** (`src/kernels/sampler_parity.cpp`), grounded at HEAD
`6cabad2`. It is also the first kernel where the *order of operations is the
correctness content* (issue #53) and where double-precision device math must
match the host reference bit-for-bit.

Source inventory:

- kernel: `src/kernels/cuda/sampler.cu` (933 lines) — 5 kernels + 3 host
  dispatchers, in llama.cpp's chain order: penalties → top_k → top_p → min_p
  → temperature → pick
- parity driver: `src/kernels/sampler_parity.cpp` (1285 lines) — 18 fixtures +
  a host Philox transcription + host reference chains + `--bench`
- API header (reused, not re-derived): `include/strata/kernels/sampler.hpp`
  (`SamplerParams`, `sample_tokens`, `penalty_rows` — `penalty_rows` is an
  inline host function, used verbatim by fixtures 12 and 15)
- `include/strata/core/coupled_draft.hpp` — **not needed** (see scope)

## 1. Goal

Port the sampler chain — the kernel that decides which token the engine
emits — to SYCL, validated by a parity driver that reuses the CUDA test's
fixtures, references, and observability assertions verbatim. The order
observable fixtures (2, 9, 10, 16, 17) must survive: a wrong sampler order
still returns a valid token, so "it produced a token" would prove nothing.

Done when: all three sampled paths (split / one-block / old) green through
their ctest runs, mutations caught, or — per the parent plan — a written
deferral with the exact missing pieces. **My analysis below predicts green**
(local memory fits, no subgroup API needed, double-math risk is probed in
Task 0), so the deferral branch is a contingency, not the expectation.

## 2. Scope decision (read before the rest)

**Ported (in `poc/sycl/kernels/sampler.cpp`):**

| CUDA kernel | lines | what it does |
|---|---|---|
| `sampler_greedy_kernel` | 100–167 | one block per token, penalty bitmap + block argmax over the vocab |
| `sampler_kernel` (Old) | 187–317 | `k` argmax rounds with the `taken` sweep + all-thread double tail |
| `sampler_one_block_kernel` | 433–504 | `k` rounds with the prev-threshold (no sweep) + warp-0 tail |
| `sampler_split_part_kernel` | 568–647 | split stage 1: warp-1024-logit register top_k, 4-list block merge |
| `sampler_split_merge_kernel` | 651–668 | split stage 2: 64-list merge + tail, one warp per row |
| host: `SampledPath`/`env_flag`/`sampled_path` | 753–767 | env-selected path, mirrored so ctest can run all 3 paths |
| host: `sample_tokens` | 836–897 | dispatch + fallback conditions (scratch simplified, §4.4) |

Plus the shared device helpers they all use: `philox4x32_round`,
`philox_uniform`, `history_count`, `apply_penalties`, `sampled_k`,
`take_first`, `warp_first`, `sampled_tail_warp`, `warp_merge_lists`,
constants `kSelMax`/`kFullMask`/`kSplit*`.

**Deferred to Phase B, with reasons:**

- **Coupled draft kernels** (`coupled_stage_kernel`, `coupled_penalize_kernel`,
  `coupled_merge_kernel`, `coupled_draft_sample`; kernels 670–751, host
  functions 891–933). They exist only to be captured into the MTP drafter's round/step
  **graphs** (their inputs are read from device/mapped memory precisely so a
  capture can hold them). No graph capture in Phase A → nothing to call them
  from. They reuse the same primitives ported here (`apply_penalties`,
  `history_count`, `sampled_k`, `sampled_tail_warp<kProb=true>`,
  `warp_merge_lists`), so Phase B's port is assembly, not invention.
- **`stream_capturing()`** (769–779): graph-capture query, Phase B.
- **`split_scratch()` slot cache** (789–832): per-(device, stream) growing
  scratch with retired-buffer bookkeeping, sized for multi-thread engine
  streams. The PoC dispatcher allocates scratch per call and frees it after
  the driver's wait (§4.5).
- **Fixture 18a** (graph-capture fallback test, driver lines ~1240–1260):
  exercises exactly the machinery above. Excluded from the ported driver with
  a documented note; **18b** (row-cap fallback: 64 rows split, 70 rows
  one-block) ports in full.

## 3. Task 0 — probes (`poc/sycl/t5/`)

Three small programs, each a gate for the design decisions below. All use
`strata_launch` / the T3 handler-`local_accessor` pattern.

**P1 — `atomic_ref` on local memory** (`t5/atomic_probe.cpp`).
The penalty bitmap is built with `atomicOr` on shared memory (kernel lines
129, 219, 455, 593; coupled, deferred, needs the *return value* of the
atomic — not ported). Probe: `sycl::atomic_ref<unsigned>` with
`sycl::memory_scope_work_group` doing `fetch_or` over a local accessor, from
many threads, result compared to the serial OR of the same bits.
**Gate:** works → bitmap keeps the CUDA parallel-build structure.
**Fallback (correctness-identical):** thread 0 alone ORs all `hlen` history
entries into the bitmap after the zero-fill barrier (`hlen ≤ 1024` serial
local reads — µs-scale). The bitmap is fully published before any candidate
scan, so serial vs parallel construction changes no sampled value.

**P2 — device double `exp` / `logf` vs host** (`t5/math_probe.cpp`).
The tail (old/one-block/split) computes `exp((double) sel_logit[i] - (double) mx)`
in **double** and compares cumulative doubles against `p.top_p`; the host
references (`sampled_reference`, `sampled_cut`, `mirror_pick`) do the same in
host `std::exp`. The parity is only as good as the last ulp of the device
double math. Note the arguments are special: `sel_logit[i]` and `mx` are
fp32 (exact in double), so every argument is a **fp32-representable double in
[−745, 0]** (plus the NaN/±inf cases of fixture 16). Probe: a grid of all
fp32 values in [−745, 0] (and the special values) through device
`exp((double)x)` vs host `std::exp((double)x)`, plus device `logf` vs
`(float) std::log` on the fixtures' min_p values {0.05, 0.3, 0.5, 0.9}.
**Gate:** bit-identical → documented bit-exactness.
**Fallback (in order of preference):** (a) if only a handful of values
differ and none is a fixture argument, document the gap with the exact
affected set; (b) otherwise hand-roll a correctly-rounded double `exp` in
device code for fp32-representable arguments (range-reduce + double Taylor —
bounded work, ~100 lines); (c) if that still leaves a fixture knife-edge,
that is a *finding for the report*, not something to paper over.

**P3 — 1024-wide local-memory reduction** (`t5/tree_probe.cpp`).
T3 validated the barrier-based local-memory tree at 320 threads; the greedy
/ old / one-block kernels need it at **1024** (Arc's max work-group).
Probe: 1024 threads each hold a (value, index) candidate; write to local
memory, barrier, 10 halving levels, each level comparing pairs in the
selection order (§4.2); compare the winner against a host serial scan over
the same candidates, including a deliberate tie pair.
**Gate:** winner matches the serial scan (lowest-index tie rule) → pattern
approved at scale.

## 4. Kernel design — `poc/sycl/kernels/sampler.cpp`

One file, one `submit_*` per CUDA kernel, `sycl::event`-returning (no
internal sync — the driver owns the wait, per T2/T3/T4). All shared-memory
kernels use the **T3 pattern**: `local_accessor`s constructed in the
`q.submit` handler, captured by value; every `__shfl_*`/`__syncwarp`
replaced by local-memory + `it.barrier()`.

### 4.1 Shared device helpers

- `philox4x32_round` (`sampler.cu` 39–51): verbatim body except the two
  `__umulhi` lines (41–42) → `(uint32_t)(((uint64_t) a * b) >> 32)` glue
  (identical by construction; the host driver's `PhiloxRound` already spells
  it this way).
- `philox_uniform` (53–61), `history_count` (67–71), `apply_penalties`
  (73–79): verbatim bodies, `__device__ __forceinline__`/`__restrict__`
  dropped. **These three are the semantic core** — the penalty
  multiply/divide rule and the presence-as-boolean live in
  `apply_penalties`, verbatim.
- `sampled_k` (331–334), `take_first` (337–339): verbatim.
- `warp_first` (344–351) and the two-stage block reductions: **glue**,
  replaced as below. The verbatim `take_first` compare survives inside
  every replacement.

### 4.2 The block argmax replacement (greedy / old / one-block rounds)

CUDA: per-thread strided scan → 32-lane `__shfl_down` tree → warp winners to
`sv[]/si[]` → warp 0 second `__shfl_down` tree over ≤32 warp winners.
SYCL (one stage, no subgroups): per-thread strided scan (glue: `tid`/
`threads` for `threadIdx.x`/`blockDim.x`) → each of the 1024 threads writes
`(bv, best)` into a local pair array → barrier → 10 halving levels: level
`w` (1024→1), thread `t < w/2` folds pair slots `2t`, `2t+1` with the
mirrored compare `if (ov > bv || (ov == bv && oi < best)) { bv = ov; best =
oi; }` and writes the survivor back; barrier. Thread 0 reads the final pair
and does the mirrored write `out[t] = (best < n_vocab) ? best : 0;`
(`sampler.cu` 165, verbatim modulo variable name).
Bit-exactness: `take_first` is a total order (value desc, index asc) and
the scan keeps the first maximum in ascending-id order — identical to the
serial semantics the header proves; the tree shape cannot change the
winner. Cost vs CUDA: 10 barrier levels instead of 2 shuffle stages — a
Phase B perf note, not a parity one (P3 validates it at 1024).

### 4.3 The penalty bitmap

Local accessor `unsigned penal_bits[bits_words]`, `bits_words =
(n_vocab + 31)/32` (line 120, verbatim) — **at most 32 KiB** for the
262,145-vocab fixture, inside Arc's 128 KiB per work-group (T0). Build:
zero-fill (strided) → barrier → OR pass (P1's `atomic_ref` or thread-0
serial fallback) → barrier. `use_bits` gate (123, verbatim) and the
`hit_count` lambda (132–135, **verbatim** — `penal_bits[v >> 5]` indexes the
accessor identically) unchanged. The out-of-vocab id guard (128, verbatim)
is what fixture 13 pins.

### 4.4 Per-kernel notes

- **`submit_greedy`** (CUDA 100–167, work-group 1024): §4.2 reduction, §4.3
  bitmap, scan loop with the `n_vocab`-as-no-candidate sentinel (137–140,
  verbatim). Padded items past `n_tokens` hit the bounds check (T2 rule).
- **`submit_old`** (187–317, work-group 1024): the `taken` sweep (246–248,
  verbatim) and **the all-thread tail (278–316) is fully verbatim** — it is
  pure per-thread double arithmetic with no warp intrinsics (that is
  precisely what made it slow on CUDA: 1024 threads redundantly running the
  FP64 chain). `sel_ids/sel_logit` are handler-built local accessors; the
  tail indexes them as before. Only glue: the two reduction stages per round
  and `if (threadIdx.x == 0) out[t] = pick;` (316).
- **`submit_one_block`** (433–504, work-group 1024): rounds use the
  prev-threshold candidate test (477, verbatim) instead of the sweep.
  **Documented deviation:** CUDA runs `sampled_tail_warp` on warp 0 only
  (`if (warp != 0) return;`, 502). A SYCL barrier is work-group scope —
  threads that returned early would strand their peers (the T2/T3
  stranded-peer rule, in reverse). So **all 1024 threads run the tail**
  (exactly the Old kernel's shape): identical `pick`, thread 0 writes.
  The tail's `ex[]` scratch, `__syncwarp`s → group barriers, `__shfl_sync(…,
  0)` broadcasts → lane-0-writes/local-reads, all as §4.2-style glue with
  the mirrored arithmetic lines (368–370, 384–387, 395–398, 401–402,
  413–420 verbatim). FP64 cost ×32 vs CUDA's one-warp tail: Phase B perf
  note.
- **`submit_split_part`** (568–647, work-group 128 = 4 "warp regions" of 32):
  each region loads its 1,024 penalised logits into 32 registers (600–606,
  verbatim modulo the `__int_as_float(-inf)` line), runs `k` rounds of the
  two-chain in-register argmax (623–631, verbatim) plus a **per-region
  32-entry local-memory butterfly** replacing `warp_first` (region `r` uses
  its own 32-pair local slot block; group barriers sync all 4 regions in
  lockstep — they are always at the same round). Region 0 then merges the 4
  region lists with the local-memory `warp_merge_lists` (§4.6) into the
  block's `cand` list. `wl` (616) and the block bitmap (585–596) are handler
  accessors. Padded 1D, as T3 did for (rows × threads): `strata_launch(q,
  n_blocks × n_tokens × 128, 128, …)` with `b = (item / 128) % n_blocks`,
  `t = (item / 128) / n_blocks` and the T2 bounds check on `t` (padded
  items land in valid (b, t) pairs — the global range is a multiple of
  128 by construction).
- **`submit_split_merge`** (651–668, work-group 32 — exactly one CUDA warp):
  copy the row's ≤64 lists (≤32 KiB) into a local accessor (656), run
  `warp_merge_lists` (tid owns lists `tid` and `tid + 32`, as in CUDA),
  then the tail — all 32 threads participate, which *is* CUDA's warp-only
  shape. Local memory here: 32 KiB lists + 1.3 KiB tail scratch ≈ 34 KiB,
  inside the 128 KiB cap. **No `int2`**: the PoC uses `struct I2 { int x;
  int y; }` (glue; `c.x`/`c.y`, `make_int2` lines become glue, the `lists`
  arithmetic lines stay verbatim).
- **`submit_sample_tokens` dispatcher** (host, mirrors 836–897): same
  `SampledPath`/`env_flag`/`sampled_path` logic (753–767, verbatim — the
  ctest ENVIRONMENT trick works unchanged); same fallback conditions
  `n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows` (865–866,
  verbatim except the `stream_capturing` clause, Phase B). **Deviations:**
  (1) no `stream_capturing` check — Phase B; (2) scratch: when the split
  path applies, `sycl::malloc_device(n_tokens × n_blocks × kSelMax × sizeof(I2))`
  per call, freed via `sycl::free` after a queue wait (the driver always
  waits before reusing buffers; one documented sync vs CUDA's slot cache);
  allocation failure → one-block fallback, same as CUDA's `scratch == nullptr`.
  (3) `sample_tokens`' `exit(1)` on a missing-history-with-penalties call
  (839–842) → throw, per the T3 library rule; the driver never hits it.

### 4.5 Local-memory budget check (the parent plan's "dynamic shared memory opt-in" question)

| consumer | CUDA dynamic smem | SYCL local memory | Arc cap |
|---|---|---|---|
| greedy/old/one-block bitmap @ 262,145 vocab | 32 768 B | 32 768 B | 131 072 B |
| old/one-block `sel_*`+`sv`+`si`+`ex` | 1 280 B | 1 280 B | ✓ |
| split part `wl` + block bitmap | 2 048 + 512 B | same | ✓ |
| split merge `lists` + tail | 32 768 + 1 280 B | same | ✓ |

Nothing approaches the 128 KiB cap → **no opt-in machinery exists or is
needed**. The parent plan's deferral trigger does not fire.

## 5. Mirror blocks

Format as before: `// SYCL-MIRROR-BEGIN <path>:<from>:<to> @ 6cabad2` …
verbatim … `// SYCL-MIRROR-END`, checked by `check_mirrors.sh`. Ranges below
are grounded against the numbered sources at `6cabad2`; the checker is the
final arbiter (T3/T4 both caught drift on first run — expect the same).
**Naming convention:** the SYCL side defines `tid` / `lane` / `warp` as glue
(`(int) it.get_local_id(0)` etc.) so that mirrored lines using those names
stay verbatim.

### 5.1 Kernel (`poc/sycl/kernels/sampler.cpp`), source `src/kernels/cuda/sampler.cu`

| block | lines | content |
|---|---|---|
| philox body | 43:50 | round math (`__umulhi` lines 41–42 are glue) |
| philox_uniform | 54:60 | counter setup + 10 rounds + 24-bit result |
| history_count | 68:70 | the count scan |
| apply_penalties | 74:78 | **the** multiply/divide + boolean-presence rule |
| greedy prologue | 104:112 | row pointer + tail-window clamp |
| bitmap gate | 120:123 | `bits_words`, `use_bits` |
| hit_count | 132:135 | membership test + count-on-hit |
| sentinel init | 137:138, 140:140 | no-candidate convention (line 139 `__int_as_float(-inf)` is glue) |
| scan body | 142:143 | penalised logit + strict-`>` keep |
| take compare | 148:148 | the total-order compare (reused in every reduction) |
| greedy write | 165:165 | sentinel→0 rule |
| old prologue | 192:196 | `inv_t` + window clamp |
| old k clamp | 229:231 | top_k 1..64 / 0 / wider |
| taken sweep | 246:248 | the Old-only O(k) sweep |
| old round scan | 249:250 | + the prev-threshold twin at 477:477 (one-block) |
| **old tail** | 278:315 | top_p (double) → min_p → temperature → Philox draw, all verbatim |
| constants | 327:328, 332:333, 338:338 | `kSelMax`, `kFullMask`, `sampled_k`, `take_first` |
| tail shared blocks | 368:370, 384:387, 395:398, 401:402, 413:420 | `inv_t`/mx, the two ordered double sums, min_p thresh, smx, pick loop (`warp_first`/`__shfl_sync`/`__syncwarp` lines between are glue) |
| one-block prev init | 471:471, 499:500 | `int prev_i = -1` round-0 seed, prev update (line 470 `__int_as_float(+inf)` is glue) |
| split constants | 511:516 | per-lane/warp/block spans, caps |
| merge init | 526:527, 529:531 | two-owned-lists setup |
| merge round | 540:543, 548:558 | take_first + advance (int2 lines glue) |
| split part prologue | 576:582 | tail-window clamp |
| part bitmap guard | 591:592 | in-block id test (atomicOr line 593 glue) |
| register load | 604:605 | lane+32j load (−inf line glue) |
| part penalty | 611:612 | hits-only penalise |
| part round | 618:619, 625:631, 632:634 | prev init, two chains, take_first, empty break (line 617 `__int_as_float(+inf)` is glue) |
| merge prologue | 654:654, 660:662 | row id + copy loop (line 655 `threadIdx.x` and the int2 lines glue; the SYCL side keeps a `lane` variable so the mirrored lines read verbatim) |

≈ **30 blocks** (vs T4's 10) — the sampler's semantic content is spread
across five kernels.

### 5.2 Driver (`poc/sycl/drivers/sampler_parity.cpp`), source `src/kernels/sampler_parity.cpp`

The host references are pure C++ — the biggest single win of this port:

| block | lines | content |
|---|---|---|
| `reference_pick` | 34:69 | the specified-order host reference (full function) |
| `reference_cut` | 71:102 | survivors-after-top_p |
| `run` compare | 123:127 | bad-count + verdict printf (alloc/copy lines 108–122 and frees 128–131 are glue; `cudaMemset(d_o, 0xFF)` → host-fill `-1` + copy) |
| `PhiloxRound` | 137:150 | host Philox step |
| `host_philox_uniform` | 152:160 | host draw |
| `sampled_reference` | 164:225 | **the** full sampled-chain host reference |
| `sampled_cut` | 227:272 | survivors after min_p + top_p |
| `SelList` | 274:277 | list-of-id/logit type |
| `mirror_select` | 280:307 | the kernel's selection semantics, host-side |
| `mirror_pick` | 309:344 | the kernel's tail semantics, host-side |
| `sampled_k` (host) | 346:346 | k clamp |
| `window_of` | 383:389 | counted-window extraction |
| summary | 1280:1284 | failures printf + exit contract |

The 18 fixtures (main, from line 436) mirror block-per-block at their
section boundaries (landmarks: f1 457, f2 468, f3 506, f4 539, f5 615, f6
646, f7 681, f8 721, f9 755, f10 782, f11 806, f12 832, f13 896, f14 940,
f15 962, counter-segmentation ~1001, f16 1028, f17 1122, f18 1200): data
generation, reference calls, observability asserts, and verdict prints are
verbatim; only the CUDA alloc/copy/sync/free lines (and f3's raw
`cudaMalloc`/`sample_tokens` calls) become `ctx` glue. **Excluded:**
fixture 18a's graph-capture half (§2) — the rest of 18 (capture *fallback
decision* minus capture itself, and 18b's row cap) ports. `DeviceRows`
(350–381) and `bench_sampled` (391–434) are glue shells keeping their
printf/loop bodies. The `stream` parameters (`run`'s `nullptr`, fixture 17's
`{nullptr, cs}`) all map to `ctx.q` — a single in-order queue cannot express
two streams; documented in the driver header, and no fixture's *assertion*
depends on stream identity (17 checks picks, not stream behaviour).

≈ **45 blocks** → **~155 total in the tree** after T5 (82 now + ~75; the
checker will state the truth).

## 6. Parity contract (mirrored, not re-derived)

- Picks are **integers** — exact equality, no tolerance (the CUDA test has
  none; a wrong order is not "close").
- Every fixture keeps its **observability assert** verbatim (cut_spec ≠
  cut_alt, token-0-share within 4σ of 0.1050, token-2-drawn, sentinel
  positions reached, tied picks present, …) — a fixture that can't see its
  feature fails the run even if every pick matches.
- The Philox draw is pinned **exactly** (counter-based: (seed, counter + row));
  fixtures 3 (greedy ignores seed), 9 (one penalties stage), and the
  counter-segmentation block would catch any RNG drift.
- ctest runs the binary **three times**, exactly as the main CMake does
  (lines 460–466): default (split), `STRATA_SAMPLER_ONE_BLOCK=1`,
  `STRATA_OLD_SAMPLER=1`. The dispatcher's env logic is mirrored, so the
  same ENVIRONMENT properties apply.

## 7. Mutation tests (mandatory, per T2–T4 doctrine)

Three mutations, each targeting a fixture designed to see it; restore → green:

| # | mutation (kernel) | designed victim | expected |
|---|---|---|---|
| M1 | `take_first` tie: `oi < bi` → `oi > bi` | fixtures 16/17 (tie lists, sentinel rows) + greedy ties | red on the tie fixtures |
| M2 | tail order: run min_p **before** top_p (the pre-#53 bug) | fixture 10 (token 2 must be drawn; old order never draws it) | red: `twos == 0` |
| M3 | `apply_penalties`: divide unconditionally (the source-paper reading) | fixture 4A (multiply-rule observable) | red: `A_visible` pick wrong |

(Each mutation is reverted and the full 3-path suite re-run before commit;
`check_mirrors.sh` goes red while a mutation is in — that is expected and is
how the T2/T3/T4 runs documented them.)

## 8. Bench spec (`--bench`, ctest `k_sampler_bench`)

Mirror `bench_sampled` (driver 391–434) minus the CUDA events: fixed
N(0,3) logits at the engine's 248,320 vocab, T ∈ {1,4,8}, k ∈ {20,64},
top_p 0.95, temperature 0.7, 3 warm-ups + 50 timed calls, chrono wall-clock
around `sample_tokens` + `ctx.wait_and_throw()`, same printf format
(`us per call`). This measures whichever path the env selects (default:
split). The numbers join T6's ledger as the first Arc sampler figures; no
CUDA comparison exists in this checkout.

## 9. CMake (`poc/sycl/CMakeLists.txt`)

Following the existing convention (static lib + consumer exe; the global
`-fsycl`, rpath, and UMF come from the foreach at the bottom of the file):

```cmake
# T5: the sampler chain (plans/sycl-phase-a-t5.md).
add_executable(t5_atomic_probe t5/atomic_probe.cpp)
add_executable(t5_math_probe   t5/math_probe.cpp)
add_executable(t5_tree_probe   t5/tree_probe.cpp)
add_library(k_sampler STATIC kernels/sampler.cpp)
add_executable(k_sampler_parity drivers/sampler_parity.cpp)
target_link_libraries(k_sampler_parity PRIVATE k_sampler)

add_test(NAME t5_atomic_probe COMMAND t5_atomic_probe)
add_test(NAME t5_math_probe   COMMAND t5_math_probe)
add_test(NAME t5_tree_probe   COMMAND t5_tree_probe)
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

Plus the T5 probes as standalone exes/ctests (`t5_atomic_probe`,
`t5_math_probe`, `t5_tree_probe`) following the T3 `t3_sg_probe` pattern.
Fixture 16 alone is 5 vocabs × 67 top_k launches; 300 s per path is a
generous envelope (T4's whole suite ran in 0.4 s; even 100× slower
fits).

## 10. Risks

| # | risk | mitigation / verdict |
|---|---|---|
| R1 | device double `exp`/`logf` ≠ host on tail arguments (P2) | probed before any port; fallback ladder in §3 (hand-rolled correctly-rounded exp for fp32-arg inputs; else a written finding). The fixtures were built with margins off the knife edges, but fixture 16's NaN/±inf rows exercise the specials, which IEEE pins |
| R2 | `sycl::atomic_ref` work-group scope missing (P1) | thread-0 serial bitmap build — correctness-identical, µs cost |
| R3 | barrier count: one-block rounds = k × (scan + 10-level tree + 2 barriers) + the tail's ~6 barriers, at 1024 threads | perf-only. It is the same local-memory pattern T3 used at 320 threads, scaled to Arc's 1024 max work-group; Phase B can revisit the cost (wider splits, vectorised loads). Parity unaffected |
| R4 | 32–34 KiB local memory per work-group | fits the 128 KiB cap with 4× margin (§4.5 table) — no opt-in needed |
| R5 | `int2`, `make_int2`, `__float_as_int`, `blockIdx`, `threadIdx.x`, `dim3`, `__launch_bounds__` have no SYCL spelling | all glue, listed per-block in §5.1; the semantic lines around them mirror verbatim |
| R6 | early-return-before-barrier hazards (`if (t >= n_tokens) return;`, `if (warp != 0) return;`) | T2/T3 stranded-peer rule: bounds-checks before any barrier (padded items included); warp-0-only tail → all-thread tail (§4.4, documented) |
| R7 | fixture 18a unportable (graph capture) | written exclusion (§2); 18b's row-cap fallback still tested; Phase B owns capture |
| R8 | driver's two-stream fixtures lose stream identity on one queue | documented (§5.2); no assertion depends on it |
| R9 | Philox on device: 64-bit multiply chain in device code | pure integer, no intrinsic after the §4.1 glue; pinned by fixtures 3/9/counter-seg |

## 11. Done when

- [ ] P1/P2/P3 probes green (or their fallbacks adopted and documented).
- [ ] `k_sampler.cpp` (5 kernels + dispatcher + helpers) and
  `k_sampler_parity.cpp` written; `check_mirrors.sh` all OK.
- [ ] ctest: `k_sampler_parity` **×3 paths** + `k_sampler_bench` + probes +
  all T0–T4 regressions green; `env -u LD_LIBRARY_PATH` clean.
- [ ] Mutations M1–M3 each red on their designed fixture, restore → green.
- [ ] Report §T5: parity table per path, bench numbers, deviations list
  (all-thread one-block tail, per-call scratch, 18a exclusion,
  `atomic_ref` outcome, double-math outcome).
- [ ] Committed as `phase a t5`.

## 12. Effort

| task | work | est |
|---|---|---|
| T5.0 probes P1–P3 | 3 small programs + gates | 0.5 d |
| T5.1 helpers + greedy + old | the verbatim-heavy half of the kernel | 1 d |
| T5.2 one-block + split stages + dispatcher | the structural half (barrier trees, I2, scratch) | 1 d |
| T5.3 driver | 18 fixtures, mostly verbatim host code + glue | 1.5 d |
| T5.4 mutations + bench + report + commit | per §7/§8 | 1 d |
| **total** | | **~5 d** (parent plan: ~1 week — inside) |

The parent plan's deferral contingency (§2 of the parent) is expected **not**
to fire: no local-memory opt-in, no subgroup API, no mapped memory, no graph
capture is required for the in-scope surface. If P2 forces the hand-rolled
double `exp` and that still leaves a fixture knife-edge, the honest output is
a written finding on that fixture — not a silent tolerance.
