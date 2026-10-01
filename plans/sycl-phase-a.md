# Phase A — SYCL proof of concept: implementation plan

Parent: `plans/sycl-backend.md` (feasibility research). This plan is Phase A only:
prove that Strata's kernels can be ported to SYCL (oneAPI DPC++, Intel Arc) and
validate them with the existing parity harness — **before** any work on the host
overlap engine, prefill GEMM, or CMake backend wiring.

## Definition of done (gate criteria)

The phase is complete when all of these hold on one Intel Arc GPU (A770-class or
better, Linux):

1. Four representative Strata kernels, ported to SYCL, run on the GPU.
2. Each passes a parity driver that reuses the **existing host references and
   fixtures** from the CUDA parity tests (same tolerances, same "what this test
   can and cannot see" guarantees).
3. A timing run shows the memory-bound kernels reach a sane share of the card's
   rated bandwidth (target: ≥ 60% on the dequant streaming kernel) — enough to
   confirm the design's premise, not to benchmark the backend.
4. A short porting report (`plans/sycl-phase-a-report.md`) records, per kernel:
   what mapped 1:1, what needed rework (barriers, shuffles, packed bytes,
   dynamic shared memory), and what the `sycl_compat` layer therefore needs.
   This report feeds Phase B.

Non-goals (explicitly out of scope for Phase A):

- The doorbell / graph-capture overlap engine (`src/core/graph.cpp`) — untouched.
- Prefill GEMM, cuBLAS replacement, GGML MMQ path — untouched.
- Mapped host memory, pinned staging, multi-GPU — untouched.
- Any change to the main `CMakeLists.txt` or to `src/` files (ports live in a
  separate tree; the originals stay the source of truth).
- Windows. Linux first (Level Zero is mature there; Windows support decides
  packaging in Phase D, not the PoC).

## Environment (T0)

**All present on this machine — nothing to procure:**

- oneAPI Base Toolkit **2026.1** installed at `~/intel/oneapi/` (DPC++ 2026.1.1,
  `icx`/`icpx`/`dpcpp`). Activate with `source ~/intel/oneapi/setvars.sh` before
  any build; `poc/sycl/CMakeLists.txt` and `run.sh` must do this (or document it
  as the first line).
- Target GPU: **Intel Arc Battlemage G31** (0xe223, the B570/B580 class),
  confirmed visible to the runtime:
  `sycl-ls` → `[level_zero:gpu][level_zero:0] ... over Level-Zero V2, Intel(R) Graphics [0xe223]`.
  This machine is the dev machine; the old "no GPU = phase cannot start" risk is gone.
- The machine also exposes an OpenCL CPU (Xeon Gold 6148) and an OpenCL GPU NEO
  device — useful as CPU reference targets, not the backend under test.

- Sanity spike (still required, still the gate): 100-line SYCL program — USM
  device alloc, one kernel, event timing, device bandwidth number printed.
  Gate: it runs on `level_zero:0` and reports ≥ 60% of rated bandwidth, else stop
  and reassess. Listing the device in `sycl-ls` proves the driver is present, not
  that the runtime path works end-to-end — the spike is that proof.

## Deliverable layout

Everything new lives under `poc/sycl/` so the main build is untouched:

```
poc/sycl/
  CMakeLists.txt            standalone build: DPC++ (icx --sycl), no main-tree targets
  include/sycl_compat/      host-side shims + launch helpers (mirrors hip_compat/, but thinner)
  kernels/                  SYCL ports: <name>.sycl  (one per kernel, side-by-side with src/kernels/cuda/<name>.cu)
  drivers/                  parity drivers: <name>_parity.cpp  (adapted from src/kernels/<name>_parity.cpp)
  bench/                    tiny timing harness (dequant streaming bandwidth, GEMV tok/s-ish)
  run.sh                    one command: build + run all parity drivers + bench, print summary
```

Key rule: each driver must include/reuse the **same fixture and reference code**
as the CUDA parity test it mirrors (copy or factor the `reference_*` functions
out, never re-derive them), so a green light means the same thing in both
backends.

## sycl_compat (what it must contain, nothing more)

A HIP-style `#define cudaX hipX` shim does **not** apply to device code — SYCL
kernels are lambdas/kernels, not `__global__` functions. So Phase A keeps the
shim deliberately small:

- Host: `sycl::queue`/`device` wrappers exposing the ~10 runtime calls the parity
  drivers use (alloc/free/copy/sync/error-string), so driver code reads like the
  CUDA tests.
- Launch helper: `strata_launch(queue, nd_range, fn)` wrapping the fixed
  work-group size each kernel uses.
- Device intrinsics header (mirror of `hip_compat/intrinsics.hpp`), populated
  **on demand** as kernels reveal what they need: packed-byte ops
  (`__byte_perm`-class), `__ldg` (drops to plain loads, verify perf), `__expf`,
  rounding intrinsics, shuffle replacements (see below).

## Kernel selection and order

Chosen from the research doc, ordered by risk (cheapest first, so the pipeline —
toolchain, shim, harness, runner — is de-risked before the hard kernels). All
four are self-contained (no doorbells, no graphs, no BLAS).

| # | Kernel | Size | Why it's in the set | Main SYCL risk |
|---|---|---|---|---|
| 1 | `dequant_s2.cu` | 51 lines | Memory-bound streaming; zero shuffles/shared; proves USM + launch + timing end-to-end | packed-byte byte extraction |
| 2 | `router_top10.cu` | 226 | MoE router; host reference is spec-based (`ref/moe.py`); exercises block reduction | reduction barriers at 16-wide EU |
| 3 | `s_gemv.cu` | 694 | **The flagship**: the actual per-token expert math (S2/Q2_0 path first, then Q8_0); float4 vector loads, fp16 scales | vectorized loads, fp16 math, 12 barrier/shuffle sites |
| 4 | `sampler.cu` | 933 | Largest; ~60 barrier/shuffle sites; exercises everything the others don't | heaviest rework; last so its cost is known, not guessed |

Parity sources (do not rewrite these):

- `src/kernels/dequant_s2_parity.cpp` — reference = artifact scalar dequantizers
  cross-checked against ggml.
- `src/kernels/router_top10_parity.cpp` — reference = host impl of `ref/moe.py`
  router spec; exact IDs, clamped/renormalised weights, tie cases.
- `src/kernels/s_gemv_parity.cpp` — reference = CPU dequant chain; covers S2/S4/
  S8 forms, codebook vs affine, the documented `has_offset` gap stays documented.
- `src/kernels/sampler_parity.cpp` — reference = host sampler in specified order;
  keeps the order-observable fixture (wrong order must FAIL, not pass).

## Tasks

**T0 — Toolchain + sanity spike — DONE (PASS).**
Done when `run.sh` of a hello-world prints device, bandwidth, one timed kernel.
Executed: `poc/sycl/t0/{dev0,bw_spike,bw_probe}.cpp` built with
`icpx -fsycl -O2 -std=c++20`; triad 935.5 GB/s ≥ 307.2 GB/s gate (full
transcript, SKU findings, and the API gotchas that shaped the code in
`plans/sycl-phase-a-report.md` §T0).

**T1 — Skeleton — DONE (PASS).**
Done when an empty driver compiles under `icpx -fsycl` and `run.sh` runs the test suite green.
Executed: standalone CMake build, `sycl_compat/{test_ctx,launch}.hpp`, `smoke_parity`
driver, `check_mirrors.sh` drift guard (verified to fail on a one-char drift),
`run.sh` green from a clean state, and the binaries run with `env -u LD_LIBRARY_PATH`
(oneAPI rpath plus a direct link to `libumf` — the Level Zero adapter dlopens UMF and
that dlopen does not consult the executable's rpath; full story in
`plans/sycl-phase-a-report.md` §T1).

**T2 — `dequant_s2` port + driver.**  **DONE (PASS).**
Port kernel 1:1 (same loop mapping, same output layout); driver reuses
`dequant_s2_parity` fixtures/reference.
Done when: green vs CPU reference at the same tolerances; streaming bandwidth
number in the report.
Executed: `kernels/dequant_s2.cpp` (mirrored kernel body) +
`drivers/dequant_s2_parity.cpp` (mirrored fixture/reference/verdict, 10 new
mirror blocks), ctest green at 200 000 blocks — **0 mismatched, bit-exact**,
edge runs (`--blocks 1`, `--blocks 12345`) green, mutation test (broken bit
offset) fails the driver as required, all under `env -u LD_LIBRARY_PATH`.
One device finding forced a fix in `strata_launch`: the Intel Level Zero
backend rejects non-uniform work-groups, so the helper now pads the item
count to a work-group multiple and every kernel must bounds-check (full
story in `plans/sycl-phase-a-report.md` §T2).

**T3 — `router_top10` port + driver.**
Block reduction → shared (local_accessor) reduction; verify the 32-lane
assumption on 16-wide EUs (work-group mapping chosen so reduction stays within
one EU group, or split — document whichever is used and its cost).
Done when: exact IDs + weight tolerance green, including tie and near-tie cases.
Implemented: the plan's subgroup premise fell through — icpx 2026.1 exposes no
subgroup shuffle/reduce API and no kernel-struct local_accessor pattern, so the
port uses handler-constructed local accessors captured by value plus
barrier-based local-memory trees (bit-exact: fmaxf and the (value, index) pair
total order are both order-independent). Parity green: ids exact on all 4
distributions including pure ties, worst rel 1.4e-07 vs the 1e-5 tolerance,
all-equal weights bit-identical; the tie-break mutation is caught by the
all-equal case as designed. The toolchain API story is the section to read in
`plans/sycl-phase-a-report.md` §T3. **DONE (PASS)**

**T4 — `s_gemv` port + driver (flagship).**
S2 (Q2_0) instantiation first, then Q8_0/S8; keep the four-form fixture but it is
acceptable for Phase A to mark S4/codebook forms as "ported in Phase B" **only
if** they need machinery (e.g. iq packed layouts) the S2 path doesn't — state
this in the report, per the repo's honest-gap convention.
Done when: green for every form marked in-scope; GEMV throughput number in the
report.
Implemented: no form deferred — the single template over {2,4,8}-bit widths plus
the runtime `has_offset` flag and the 16-entry local-memory `kIq4Nl` table
covers all four fixture forms and Q4_K. Parity green at 1e-4: worst rel
0.000e+00 (S2, bit-exact) to 9.649e-06 (S8); Q4_K 2.814e-07 rel-to-terms
matching the CUDA test's own recorded figure. Both mutations caught
(offset-drop → Q4_K red, bias-invert → 3 bias forms red, IQ4_NL green). Bench:
naive kernel 4.6–16.9 G weights/s on the two real expert shapes (1.3–4.9 GiB/s
of S2) — the underparallelised one-thread-per-row baseline Phase B's split /
quads / fast variants will beat; first Arc numbers for the path. Toolchain
findings: no `sycl::fp16` (use `sycl::half` — validated bit-exact by
`t4_fp16_probe`), `local_accessor` has no `.data()` (`&acc[0]` idiom). See
`plans/sycl-phase-a-report.md` §T4. **DONE (PASS)**

**T5 — `sampler` port + driver.**
Heaviest: barriers/shuffles → local-memory equivalents; keep the
order-observable fixture so a wrong sampler order cannot pass silently.
Done when: green, or — if it turns out to need machinery clearly outside Phase A
(e.g. dynamic shared memory opt-in at scale) — a written deferral with the exact
missing pieces, so Phase B is not surprised. A deferred kernel must still leave
gates 1–4 satisfiable with the other three.

**T6 — Timing bench + report** (`bench/`, `plans/sycl-phase-a-report.md`).
One timed run per kernel, fixed shapes, printed summary; report per the
definition of done. This report is the input to the Phase B go/no-go.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Broken/slow Level Zero runtime on the Battlemage G31 (device is listed by `sycl-ls` but the runtime path may still be misbehaving) | T0 spike gates on measured bandwidth before any porting starts; oneAPI 2026.1 + driver are already installed, so failures are config, not procurement |
| 16-wide EU vs 32-lane warp assumptions (barriers, shuffles, vector widths) | T3 forces the issue early on a small kernel; mapping decisions documented per kernel in the report |
| `__shfl_*` has no standard SYCL equivalent | Shared-memory broadcast within the work-group; measure cost; vendor extension only if the cost is unacceptable |
| Missing packed-byte / int8-dot intrinsics (`__byte_perm`, dp4a-family) | Bit-level fallbacks in `sycl_compat` (pattern from `hip_compat/intrinsics.hpp`, which does exactly this for AMD); perf noted in report |
| `icx`/DPC++ compiler bugs on edge-case C++ | Keep ports as close to the CUDA source structure as legal; isolate per-kernel files so one broken kernel never blocks the rest |
| Fixture drift: SYCL driver silently tests weaker cases | Hard rule: drivers include the CUDA test's fixture/reference code, copy or factored — never re-derived; diff-check against the original in review |
| Scope creep toward Phase C (host engine) | Everything stays in `poc/sycl/`; any need to touch `src/core` is, by definition, Phase B/C and gets written down instead of done |

## Effort

- T0–T1: done (~1 day). T2: done (~0.5 day, vs the 1–2 day envelope).
- T3: done (~1 day, vs the 2–4 day envelope; toolchain API discovery dominated).
- T4: done (~1 day, vs the 1–2 week envelope; the T2/T3 pipeline paid off).
- T5: ~1 week, or deferral write-up.
- T6: ~1–2 days.
- **Total: ~3–4 weeks for one developer with the GPU available.**

## Go / no-go after Phase A

Proceed to Phase B if: all in-scope parity drivers are green, the bandwidth
gate passes, and the report shows no Phase A kernel required machinery
qualitatively worse than what Phase B's remaining ~16k lines would face.
Stop/rethink if: measured GPU bandwidth is far below rated (platform problem),
or the 16-wide-EU mapping costs the GEMV kernel its whole reason to exist.
