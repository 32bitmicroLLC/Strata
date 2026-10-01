# T3 — detailed plan: `router_top10` port (first shared-memory kernel)

Parent: `plans/sycl-phase-a.md` (task T3). T0–T2 are done (PASS —
`plans/sycl-phase-a-report.md`). T3 is the first kernel that cannot be
ported as a flat lambda: it needs **shared (local) memory** and
**intra-group shuffles**, i.e. the kernel-struct pattern `launch.hpp`
deliberately deferred, and the first real consumer of the T0 finding that
Arc sub-groups are 16-wide by default.

## What is being ported

**CUDA kernel** — `src/kernels/cuda/router_top10.cu` (226 lines), HEAD
`49da5df`. MoE router: per token, softmax over ALL 512 experts, stable
top-10 (ties break by **ascending index**), gather, renormalise with
ggml's 2^-14 lower clamp. Semantics from `ref/moe.py::router`.

Structure (one block per token, 512 threads for n_expert=512):

| Stage | CUDA mechanism | Portability |
|---|---|---|
| max over experts | grid-stride scan + **32-lane `__shfl_down_sync` tree** + 2nd stage over per-warp results in shared `s_red` | shuffle → `sycl::shfl_down`; tree is exact (`fmaxf`, order-independent) |
| 512 exponentials | parallel, `exp((double) l[e] - (double) mx)` into shared `s_ex` | double `exp` on software-FP64 Arc — see Risk 1 |
| double sum | **one thread, ascending order** — deliberately NOT parallelised (keeps last bits of the reference-faithful serial accumulation) | portable verbatim; double add/div are IEEE-exact |
| p[] = (float)(s_ex × inv) | computed **once** into shared `s_p` (262 144 double muls hoisted out in the CUDA file's history) | portable verbatim |
| selection | **k=10 passes of block-wide argmax**: per-warp shuffle-tree pair `(value, index)` with tie-break `(ov > bv) \|\| (ov == bv && oi < bi)`, 2nd stage in shared `s_red`/`s_rid`, thread 0 writes `ids[t,i]`, `weights[t,i]`, sets `s_taken` mask | shuffle → `sycl::shfl_down`; the tie-break is the bit of logic the parity test's all-equal case exists to catch |
| renormalise | one thread, double, `fmax(s, 2**-14)` clamp | portable verbatim; the test proves the clamp is unreachable for this geometry (k/n = 10/512 ≈ 0.0195, 320× above the clamp) |

Shared memory (n_expert=512): `s_taken` 512 B (16-aligned) + `s_ex` 4096 B
(double) + `s_p` 2048 B (float, **aliased onto `s_ex`'s tail** in CUDA) +
static `s_red[16]`/`s_rid[16]`/`s_sum` ≈ **6.8 KiB total** — well inside
Arc's 128 KiB per work-group (T0).

**Parity test** — `src/kernels/router_top10_parity.cpp` (167 lines). Host
reference (`reference()`) is a C++ transcription of the *spec* (double max,
double `std::exp`, double sum, `std::stable_sort` with strict `>`,
gather, clamp) — independently of the kernel. 4 distributions × 64 tokens ×
512 experts, k=10: random normal, **all-equal (pure ties)**, 12-way exact
tie, dominant expert. Verdict: **IDs exact, weights ≤ 1e-5 relative**,
per-token weight sum ≈ 1, plus the clamp-reachability assertion.
(The float-vs-double p[] difference between kernel and reference is why
weights carry a tolerance while IDs do not — same design as on CUDA.)

## The one architectural decision: subgroup width

Every reduction in this kernel is a 32-lane warp tree + shared 2nd stage.
T0 found Arc reports sub-group sizes **{16, 32}** — 16 by default.

- **Option A (preferred — minimal drift):** request 32-wide sub-groups via
  `sycl::property::nd_range::sub_group_size(32)` on the `parallel_for`
  call. Then CUDA warp ≡ Arc sub-group 1:1, and the port is near-line-for-
  line:
  - `__shfl_down_sync(0xffffffffu, v, off)` → `sycl::shfl_down(v, off, sg)`
    (SYCL 2020 subgroup shuffle; the all-lanes mask is implicit — every lane
    in the group calls it, including lanes that took `continue` in the
    argmax scan, exactly as on CUDA)
  - `(tid & 31) == 0` / `tid >> 5` → `sg.get_id() == 0` / `sg.get_id()`
  - `__syncthreads()` → `it.barrier()`
  - `blockDim.x` / `threadIdx.x` / `blockIdx.x` → local range / local id /
    global id
- **Option B (fallback, only if the probe shows 32 is not honoured):**
  16-wide sub-groups; 512 threads = 32 sub-groups. Per-group tree becomes
  16-lane (`off = 8,4,2,1`); `s_red`/`s_rid` grow to 32 entries; the 2nd
  stage reduces the 32 per-group pairs **serially on one thread** (32
  fmaxf/pair-compares per token — negligible) instead of in a 32-lane
  shuffle. More drift from the CUDA structure, more surface for a tie-break
  bug — hence only if A fails.

**The probe decides before porting** (task 1 below) — this is the only way
to find out whether the runtime honours the request, and it also verifies
`shfl_down` semantics and the local-accessor/barrier pattern, which are
both first uses.

## Definition of done

1. ctest `k_router_top10_parity` green: all 4 distributions, **ids exact**,
   weights worst-rel ≤ 1e-5, clamp line prints `(CLAMP IS UNREACHABLE,
   asserted)`, selftest line present.
2. ctest `t3_sg_probe` green, and its output (sub-group width under default
   and `sub_group_size(32)` requests, shuffle-tree correctness, barrier +
   local-memory reduction) is recorded in the report — the A/B decision is
   documented, not assumed.
3. `check_mirrors.sh` green on all new blocks (list below).
4. **Mutation test:** with the shuffle-pair tie-break temporarily inverted
   (`oi < bi` → `oi > bi`), the **all-equal (ties)** case must report
   `*** WRONG ***` ids (it is the case designed to catch exactly this);
   restoring goes green.
5. `env -u LD_LIBRARY_PATH ./build/k_router_top10_parity` works.
6. `git status`: `poc/sycl/` + `plans/` only.

## Files

```
poc/sycl/
  t3/sg_probe.cpp               NEW  (sub-group + shuffle + barrier probe)
  kernels/router_top10.cpp      NEW   (kernel struct + submit_router_top10)
  drivers/router_top10_parity.cpp NEW
  CMakeLists.txt                EDIT  (k_router_top10 STATIC, k_router_top10_parity,
                                        t3_sg_probe, two ctest entries)
```

### `t3/sg_probe.cpp`

One small program, three checks, prints a PASS/FAIL per check:

1. **Sub-group width.** Launch 512 items in a 512 work-group, print
   `it.get_local_range().get(0)` — once with no property, once with
   `sycl::property::nd_range::sub_group_size(32)`. (Device advertises
   {16, 32}, so 32 must be accepted; this confirms the runtime honours it.)
2. **Shuffle tree.** Per sub-group, reduce the lane values `local_id+1`
   with the exact 5-step `shfl_down` tree the port will use; compare the
   tree result against the known sum for the group. Catches both a broken
   `shfl_down` and a width mismatch (a 16-lane tree run as a 32-lane tree
   would sum wrong).
3. **Barrier + local memory.** 512 items scatter known values into a
   `sycl::local_accessor<float,1>`, `it.barrier()`, thread 0 sums, compare.
   (The pattern every router stage relies on, in isolation.)

Registered in ctest as `t3_sg_probe`, before the kernel tests.

### `kernels/router_top10.cpp`

```cpp
namespace strata::kernels {
sycl::event submit_router_top10(sycl::queue& q, const float* logits, int n_tokens,
                               int n_expert, int k, int* ids, float* weights);
}
```

- **Host checks mirror the CUDA wrapper** (`k <= 64`, `n_expert > 0`,
  `n_tokens > 0`, `threads = max(n_expert,32)` rounded to 32, capped 512) —
  mirrored line-for-line from `router_top10.cu` — but on violation they
  **throw `std::runtime_error`** instead of `exit(1)` (documented interface
  deviation: a PoC library should not kill the process; drivers never
  trigger these).
- **Arc local-memory cap check** (new, documented): compute the same
  `smem` size as the CUDA host code; if it exceeds 120 KiB, throw. CUDA's
  cap is `RT_MAX_THREADS * 64 = 32 768` experts ≈ 400 KiB smem, which cannot
  exist on Arc (128 KiB/work-group, T0). The engine runs n_expert=512
  (6.8 KiB) — comfortably in.
- **Kernel struct** (this is the pattern T1's `launch.hpp` said would first
  appear here; it launches directly, not via `strata_launch`, because it
  needs a group/size `nd_range`, a sub-group property, and per-call local
  sizes):

  ```cpp
  struct rt10 {
      const float* logits; int n_tokens, n_expert, k;
      int* ids; float* weights;
      sycl::local_accessor<unsigned char, 1> s_taken;  // align16(n_expert)
      sycl::local_accessor<double, 1> s_ex;            // n_expert
      sycl::local_accessor<float, 1>  s_p;             // n_expert
      sycl::local_accessor<float, 1>  s_red;           // 16 (= RT_MAX_THREADS/32)
      sycl::local_accessor<int, 1>    s_rid;           // 16
      sycl::local_accessor<double, 1> s_sum;           // 1
      operator()(sycl::nd_item<1> it) const { /* stages, per table above */ }
  };
  ```
  - **One deviation from the CUDA layout, on purpose:** CUDA aliases
    `s_p` onto the tail of `s_ex` (`float* s_p = (float*)(s_ex + n_expert)`)
    to save an offset parameter; SYCL gets a separate `s_p` accessor — same
    semantics, +0 B complexity, honest layout. Comment carries the note.
  - Option A launch:
    `q.parallel_for(sycl::nd_range<1>((size_t) n_tokens, (size_t) threads),
                    sycl::property_list{sycl::property::nd_range::sub_group_size(32)},
                    kernel_obj)`.
  - `fmaxf` → `std::fmaxf` (exact, order-independent — the max stages stay
    bit-identical to the CUDA tree). Double `exp` stays `exp((double) ...)`
    (Risk 1). `INFINITY` sentinel stays (IEEE).

**Mirror discipline for the kernel (finer grain than T2, by necessity).**
The shuffle/barrier/shared lines have no verbatim SYCL form, so the kernel
file does **not** carry one big body block. Instead, the semantically
load-bearing computations are mirrored in small blocks (ranges pinned at
HEAD `49da5df` at implementation time), and each SYCL-glue replacement
carries a comment naming the CUDA line it replaces:

| Block (source `router_top10.cu`) | Content |
|---|---|
| host wrapper | `RT_MAX_THREADS`, k/n_expert validation, `threads` computation, `smem` computation |
| max stage | grid-stride scan loop `mx = fmaxf(mx, l[e])`, warp-2nd-stage body `v = (tid < nw) ? s_red[tid] : -INFINITY` + `s_red[0] = v` |
| exp stage | `s_ex[e] = exp((double) l[e] - (double) mx);` |
| sum stage | the one-thread ascending `sum += s_ex[e]; s_sum = sum;` + `inv = (float)(1.0 / s_sum)` |
| p stage | `s_p[e] = (float) (s_ex[e] * inv);` |
| selection | scan loop body `if (s_taken[e]) continue; ... if (pe > bv) { bv = pe; bi = e; }`, the pair compare `if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }`, the write-out `ids[...] = ix; weights[...] = v; s_taken[ix] = 1;` and the `s_taken` initialisation |
| renorm stage | the one-thread double sum, `sc = fmax(s, 6.103515625e-05)`, the `weights[...] = (float)((double) weights[...] / sc)` loop |

The shuffle-loop *structure* (`for (off = 16; off > 0; off >>= 1)` with
`shfl_down` inside) is mirrored only in the parts that are verbatim (the
`fmaxf(mx, shfl)` lines); the `__shfl_down_sync(...)` calls themselves are
glue, one-for-one with `sycl::shfl_down`.

### `drivers/router_top10_parity.cpp`

The CUDA parity test with alloc/copy/launch replaced by `sycl_compat`
equivalents; everything else verbatim. Mirror blocks (ranges pinned at
implementation time, all `@ 49da5df`):

| # | Source | Content |
|---|--------|---------|
| 1 | parity `:33` | `RENORM_CLAMP` constant |
| 2 | parity `:36:61` | `reference()` — the host spec implementation, in full (no CUDA in it) |
| 3 | parity run_case head | the four `h_ids/h_w/r_ids/r_w` vectors + the per-token `reference()` loop (the `run_case` signature line itself is glue: gains a `ctx&` parameter) |
| 4 | parity run_case compare | `id_bad`/`w_bad`/`worst` loop, the sum-to-1 loop, the per-case `printf` — the whole verdict logic |
| 5 | parity main | `NE/K/NT` constants, `rng(2024)`, `gauss`, `bad = 0` |
| 6–9 | parity main | each of the 4 distribution fixtures (vector + fill) as its own block; the four `bad += run_case(...)` call lines are glue (gain the `ctx` argument) |
| 10 | parity main | the clamp-reachability comment + assertion block, verbatim |
| 11 | parity main | the final summary `printf` + `selftest` line |

SYCL glue (commented, replacing CUDA lines): `ctx c;`, `c.device_alloc`
×3 + `c.q.memcpy` + `submit_router_top10(c.q, ...)` + `c.wait_and_throw()`,
copy-back + wait, `c.free_device` ×3. The original's `check(cudaError_t, ...)`
helper is replaced by `sycl_compat::fail` — deliberately not mirrored.

### `CMakeLists.txt` edits

- `add_library(k_router_top10 STATIC kernels/router_top10.cpp)`
- `add_executable(k_router_top10_parity drivers/router_top10_parity.cpp)` +
  `target_link_libraries(... PRIVATE k_router_top10)`
- `add_executable(t3_sg_probe t3/sg_probe.cpp)`
- add `k_router_top10_parity` and `t3_sg_probe` to the `-fsycl`/rpath/UMF
  `foreach`
- `add_test(NAME t3_sg_probe COMMAND t3_sg_probe)` and
  `add_test(NAME k_router_top10_parity COMMAND k_router_top10_parity --selftest)`,
  both registered **after `k_dequant_s2_parity`, before the `t0_*`
  regressions** (gate stays last)
- `TIMEOUT 120` for both (the parity run is 64 tokens × 10 argmax passes ×
  512 — expected well under a second on Arc; the CPU `reference()` side is
  64 × 512 double exps, also trivial)

## Task order

1. **`t3/sg_probe.cpp` + ctest.** The A/B decision and the shuffle/barrier
   pattern, proven in isolation. *(~2 h — small, but the result gates the
   whole kernel design; if 32 is not honoured, stop and document before
   porting)*
2. **`kernels/router_top10.cpp`** per Option A or B, with the mirror
   blocks. Build-only. *(~1–2 days — the largest port yet: three reduction
   sites, the tie-break, the local-memory layout; expect most of the time
   here)*
3. **`drivers/router_top10_parity.cpp`** + ctest entry. *(~0.5 day — mostly
   verbatim copying; the `run_case` signature gaining `ctx&` is the only
   structural change)*
4. **Mutation test** (invert the tie-break; all-equal case must fail;
   restore; green). *(~1 h)*
5. **Final pass**: done criteria 1–6; record probe output + A/B decision +
   the worst-rel number in the report. *(~1 h)*

## Risks

| Risk | Mitigation |
|---|---|
| **Double `exp` on Arc (software FP64) differs from glibc's in the last ulp** | Not gated: the test's contract is *IDs exact + weights ≤ 1e-5 rel*, and a double-ulp difference (~1e-16) is two orders of magnitude inside that tolerance after the float cast. The all-equal case (where p values are bit-identical on any correct device) is the real tie-order guard. Optional diagnostic: an exp bit-probe dumping device vs host `exp` on the fixture values, reported as ulp distance — information for the report, not a gate |
| `sub_group_size(32)` not honoured / `shfl_down` misbehaves | `t3_sg_probe` runs before the port and answers both; Option B is fully specified above as the fallback |
| Tie-break bug in the port (the highest-value bug class — wrong expert selection is a silent quality loss, not a crash) | The all-equal distribution decides IDs purely on the index rule; the mutation test proves the driver catches an inverted tie-break; `s_taken` masking is mirrored verbatim |
| Arc local-memory overrun for large `n_expert` (CUDA allows 32 768) | explicit throw above 120 KiB smem; engine uses 512 (6.8 KiB); deviation documented in the kernel header |
| `exit(1)`-style host checks killing the test process on a bad fixture | checks throw instead (documented deviation); fixtures are the mirrored, known-good ones, so they never trigger |
| Divergent `continue` in the argmax scan vs `shfl_down` participation | every work-item reaches the shuffle on every path (the `continue` skips the scan only) — same participation guarantee as the CUDA warp; the probe's check 2 uses a loop with a `continue` to prove it |
| 12 mirror blocks in the driver + 7 in the kernel → drift bookkeeping | same machinery as T2; the checker already caught a real drift in T2, so this is enforced, not reviewed |

## Effort

- Step 1: ~2 h. Step 2: ~1–2 days. Step 3: ~0.5 day. Step 4: ~1 h.
  Step 5: ~1 h.
- **Total ≈ 2–3 days**, inside the parent plan's "T3: ~2–4 days". The
  schedule risk is concentrated in step 2 *and* step 1's outcome: if the
  probe forces Option B, add ~1 day for the restructured reductions and a
  second mutation pass.

## Handoff to T4

After T3, the shared-memory mechanisms are proven (handler local accessors,
barriers, per-call local sizing), and the subgroup question is settled — so
`s_gemv` (T4, the flagship: S2 GEMV with vector loads and, if needed, `int8`
dot via the T1-deferred intrinsics fallback) starts from a known-good
substrate. `s_gemv` is the kernel that will decide Phase A's performance
verdict, which is why T6's timing bench must measure it at the engine's real
shapes.

## Implemented reality (done 2026-09-30, PASS)

**Option A/B was not buildable — see `plans/sycl-phase-a-report.md` §T3.**
icpx 2026.1 exposes no subgroup shuffle/reduce API (no `sycl::shfl*`, no
`sub_group::reduce`/`shfl_down`, no property to request a sub-group width) and
the kernel-struct-with-`local_accessor`-members pattern does not compile (no
host-side range constructor). The port therefore uses **Option C**:

- `sycl::local_accessor`s constructed in the `q.submit` handler, captured
  **by value** into the kernel lambda;
- every CUDA warp-shuffle tree + shared 2nd stage → a **barrier-based tree
  over local memory**. Bit-exact: `fmaxf` is exact/order-independent and the
  `(value, index)` pair compare is a total order, so any tree returns the
  same winner as CUDA's two-stage shuffle reduction.

New host checks beyond the plan: power-of-two work-group requirement for the
local-memory trees (explicit throw), and the 120 KiB Arc cap. Both documented
in the kernel file header.

Result: parity green — ids exact on all 4 distributions (worst rel 1.378e-07
vs 1e-5 tolerance; all-equal weights bit-identical), mutation test caught by
the all-equal case, 38 mirror blocks OK, probe checks 2–3 pass. Effort: well
under the 2–3 day estimate; the dominant time was the toolchain API
 discovery the probe step was designed to absorb.
