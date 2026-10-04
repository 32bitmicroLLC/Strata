# Phase B — SYCL kernel port: plan

Parent: `plans/sycl-backend.md` (feasibility research, Phase B = "kernel port
~16k lines"). Prerequisite: **Phase A — complete** (`plans/sycl-phase-a.md`,
report `plans/sycl-phase-a-report.md`): toolchain gate, mirror discipline,
parity pipeline, and four representative kernels proven on Arc (dequant_s2,
router_top10, s_gemv, sampler). Phase A's pipeline is this phase's working
method; this plan does not re-debate it.

## Scope

**In (device side only):**

- All remaining `src/kernels/cuda/*.cu` ports: **40 files / 12,402 lines /
  ~182 `__global__` kernels** (measured: the 44 files total 14,306 lines,
  191 `__global__`, 186 `__syncthreads`, 98 `__shfl_*`, 36 `cuda_dp4a`
  mentions; Phase A already ported 4 files / 1,904 lines —
  `dequant_s2`, `router_top10`, `s_gemv`, `sampler`). B0's audit
  (`plans/sycl-phase-b-report.md` §B0.1) corrected the parent's original
  38/42 counts and assigned two previously unassigned files
  (`native_flash_attn` → B4, `native_router` → B6).
- `sycl_compat/intrinsics.hpp` — the device intrinsic layer, populated from
  the inventory below (same pattern as `hip_compat/intrinsics.hpp`).
- Parity drivers for every ported kernel, including **new** drivers where a
  CUDA one does not exist (coverage audit in B0).
- Per-kernel records: local-memory budget table, perf ledger, report
  (`plans/sycl-phase-b-report.md`) feeding the Phase C decision.

**Out — Phase C** (`src/core`, the host engine):

- Graph-capture / doorbell overlap redesign (`graph.cpp`, 39 graph launches,
  the `stream_capturing` dispatcher clause — already deferred by the T5
  port).
- Mapped pinned-host staging (`cudaHostAllocMapped` users: `remote_experts.cpp`,
  `layer.cpp`, `native_head.cpp`, `verify.cpp`) — includes the single
  `cudaLaunchHostFunc` site (`src/core/verify.cpp:1164`, verified: it is a
  host-engine call, not a kernel task).
- cuBLAS prefill GEMM replacement, hipBLASLt tuning, GGML MMQ path.

**Out — Phase D:** main `CMakeLists.txt` backend option, `setup.py` oneAPI
installer, docs, multi-backend detection in `src/core/device.cu`, Windows.

**Kernel/host boundary rule:** kernels whose *host users* need Phase C
machinery are still ported in Phase B (kernel + parity driver); end-to-end
use waits on C. `verify_kernels.cu` and the `native_*` engine users are the
named cases.

## Standing conventions (inherited from Phase A — binding, not re-debated)

1. **Mirror discipline:** every ported file carries `SYCL-MIRROR-BEGIN/END`
   blocks pinned to the CUDA file's SHA; `poc/sycl/check_mirrors.sh` must
   stay exit 0. Mirror blocks keep the CUDA source's original indentation;
   DPC++ fixes go in glue only; a mirror block that cannot compile as
   written is a **written finding**, never a bent mirror. Checker-green ≠
   compiles (Step 4 finding) — the compile is part of drift-detection.
2. **Parity culture:** every driver reuses the CUDA test's fixture and
   reference code verbatim (host-serial reference), with the "what this test
   can and cannot see" observability asserts. Drivers never re-derive
   references. New drivers (no CUDA original) must state their own
   observability contract in their header.
3. **Launch shape:** fixed 1,024 work-groups, padded item counts, per-item
   bounds checks (T2 finding: the Level Zero backend rejects non-uniform
   work-groups); all streams map to the in-order `ctx.q`; scratch via
   pooled UMF device alloc, per call.
4. **No subgroup API** (T3: icpx 2026.1 exposes no subgroup shuffle/reduce);
   no kernel-struct local_accessor pattern — handler-constructed local
   accessors captured by value, barrier trees for reductions (T3/T5).
5. **SYCL throws** — no CUDA status-check mirroring (T3 precedent);
   `sycl::malloc_device` throws on failure, never returns null.
6. **Order-sensitive logic gets mutation tests** (T4/T5 precedent): a
   kernel whose output depends on selection order, tie-break, or stage
   order carries at least one designed mutation that must red its designed
   fixture.
7. **Knife-edges are written findings, never silent tolerances.**
8. **No new public headers** — drivers forward-declare their kernel
   functions (T3/T5 pattern); Phase D moves declarations to `include/`.
9. **Local-memory budget per kernel**, tabled (T5 §3.5 precedent); Arc cap
   is 128 KiB per work-group; a mirrored shape that does not fit is a
   finding with the exact missing margin, not a silent re-tile.
10. Build/run: `poc/sycl/build`, `cmake --build . -j` (retry on icpx exit
    3), tests under `env -u LD_LIBRARY_PATH`; known transient
    device-lost/timeout flakes are rerun, never recorded (Steps 3/5/6).

## B0 — audit + intrinsic layer (est 2–3 d)

The infrastructure every batch draws on, built and probed once.

- **B0.1 Coverage audit** — table every CUDA kernel file against its parity
  driver(s). Confirmed present: `bf16_gemv`, `cvec`, `dequant_bf16`
  (`dequant_bf16_test`), `dequant_s2` (ported), `elementwise`, `gdn`, `gr`,
  `iq_kernels`, `kv_q4`, `kv_q8`, `kv_stream`, `kv_hybrid`, `native_mmvq`
  (`mmvq_multi_parity`), `ple`, `qsa`/`qsa_prompt_attn` (`qsa_parity`,
  `qsa_prompt_attn_parity`, `ngram.cpp`, `qsa_select_bench`),
  `quantize_act`, `rope`, `router_top10` (ported), `s2_gemv` family
  (`s2_gemv_parity`, `s2_gemv_q8_parity`), `s_gemv` (ported, plus
  `s_gemv_q8k_parity`), `sampler` (ported), `shared_expert`,
  `native_expert` (`native_expert_parity`). **To verify at B0** (driver may
  or may not cover the file): `fused_gdn`, `fused_gr`, `native_flash_attn`,
  `native_gdn`, `native_gdn_preprocess`, `native_gr_norm`,
  `native_gr_postops`, `native_ple_postops`, `native_qsa`,
  `native_qsa_indexer`, `native_qsa_score`, `qsa_decode_attn`,
  `native_rope`, `native_router`, `s2_expert_grouped`, `s2_gemv_fast`,
  `s2_gemv_quads`. **Known gaps needing new drivers:** 13 files (the
  parent originally assumed 1–2) — `fused_gdn`, `native_flash_attn`,
  `native_gdn`, `native_gdn_preprocess`, `native_gr_norm`, `native_moe`,
  `native_ple_postops`, `native_qsa`, `native_qsa_score`, `native_router`,
  `s2_expert_grouped`, `qsa_select` (its `qsa_select_bench` is a bench, not
  a parity), `verify_kernels` (23 kernels, no parity file at all). B0's
  deliverable is the resolved table (`plans/sycl-phase-b-report.md`
  §B0.1) — it fixes each batch's driver work; ~10–12 dev-days of new
  driver work land in the batch steps (the B1 audit, findings F1/F5,
  adds ~1.5 d of mini-drivers on top — see the B1 row below).
- **B0.2 `sycl_compat/intrinsics.hpp`**, populated from the measured
  inventory:
  - Packed bytes: `__byte_perm`, `__vsub4`, `__vcmpne4`, `__vadd4`
    (28 uses in `iq_kernels`, 12 in `native_mmvq`) — u32 bit-level
    implementations, parity-probed.
  - `cuda_dp4a` (25 uses, `iq_kernels` hub) — **probe the SPIR-V/oneAPI
    int8-dot mapping first**; if unavailable, bit-level fallback with the
    perf noted (the `hip_compat` precedent did exactly this for AMD).
  - `__ldg` (drop to plain loads, perf-note), `__expf` (Phase A's P2
    finding: 1-ulp device-vs-host `exp` gaps exist on this toolchain —
    parity harnesses catch real damage; kernels with knife-edge
    normalisation get the double-math treatment per P2b, not a silent
    tolerance), `__fadd_rn`/`__fmaf_rn` (standard ops), `__popc`,
    `__trap`, `__threadfence_system` (queue-scope fence — verify the
    semantics match where it is used), `__int_as_float` (bit cast),
    `__half` → `sycl::half` (T4: no `sycl::fp16`, use `sycl::half` —
    validated bit-exact). B0 finding: the CUDA `__nv_bfloat16` type is
    absent from all 40 unported kernels (they use `uint16_t` + the repo's
    portable `bf16_bits.hpp`); the `__nv_bfloat16` typedef is kept as
    Phase-C future-proofing only.
  - **TF32 conversion** (`qsa_select.cu:174`, `asm("cvt.rna.tf32.f32 …")`):
    the *other* of the two asm sites — not the tensor-core `mma`. Needs a
    bit-exact portable emulation (round mantissa to 10 bits, ties-away)
    against the CUDA conversion; probe in B0, used in B5.
- **B0.3 32-wide shuffle primitives.** The main per-kernel risk (98
  `__shfl_*` sites, warp-32 assumed; Arc EUs are 16-wide and the toolchain
  has no subgroup API): implement `shfl_xor/shfl_down/shfl_up/shfl` 32-wide
  over local memory inside the 1,024 work-group, generalising Phase A's
  barrier-tree fold (T3/T5). One implementation in the intrinsics layer,
  probed once on a representative attention-shaped workload. **Stop-and-ask
  if the measured cost is unacceptable** → Intel vendor-extension opt-in,
  decision written (Phase A risk-table rule: vendor extension only if the
  cost is unacceptable).
- **B0.4 Dynamic shared memory** — the 5 `cudaFuncSetAttribute` sites
  (measured: `fused_gr` ×1, `qsa` ×1, `qsa_prompt_attn` ×2, `qsa_select`
  ×1; the parent's "6" counted a comment line in `fused_gr.cu`, and the
  research doc's "~7" is approximate) become call-site-sized
  `local_accessor`s (no kernel-struct pattern, T3). The affected kernels
  sit in B4/B5; B0 sized and documented each
  (`plans/sycl-phase-b-report.md` §B0.4).
- **B0.5 Tree-layout decision** — confirm `poc/sycl/` stays the working
  tree through Phase B (recommendation: yes — the main-tree wiring is
  Phase D, and Phase A's "any need to touch `src/core` is by definition
  Phase B/C" rule keeps this phase auditable); Phase D moves it alongside
  `cmake/hip_backend.cmake`-style wiring.

## Porting batches (in risk order, mirroring Phase A's selection logic)

Per-file protocol (the Phase A step pattern, per file): read CUDA source +
its parity driver (or write a new one per B0.1) → write
`poc/sycl/kernels/<name>.cpp` with mirror blocks + glue → write/mirror
`poc/sycl/drivers/<name>_parity.cpp` (host-serial reference, observability
asserts) → CMake (static lib + exe + ctest, TIMEOUT, env pins where the
kernel has multiple paths) → `check_mirrors.sh` green → compile green →
ctest green → mutation tests where order-sensitive → report section. A
batch is done when all its files are green and recorded.

| batch | files (lines, `__global__`) | hazards | driver coverage | est |
|---|---|---|---|---|
| **B1 — elementwise / streaming** | `dequant_bf16` (248,1) + iq-dequant surface of `iq_kernels.cu` (~230 lines, F4), `rope` (136,1), `native_rope` (102,1), `elementwise` (325,14), `quantize_act` (313,5), `cvec` (166,1), `native_bf16` (186,2), `native_gr_postops` (115,3), `kv_q8` (127,2) — 1,718 lines, 30 kernels | low: light shuffles (2 in cvec/elementwise), no packed bytes, small smem | audit-refined (findings F1–F8 in `plans/sycl-phase-b-steps-b1.md`): `rope`/`native_rope` ✓ (rope_parity), `quantize_act` ✓ (Q8_K byte check red by design under F8 → `WILL_FAIL`), `elementwise` ✓ (7 of 14 entries; F6), `dequant_bf16` → selftest driver (F5), `kv_q8` ✓ q8-only (FP16 section parks in B5, F3), `cvec` ✓ minus fused-gr section (parks in B4, F2); **`native_bf16` and `native_gr_postops` have no direct CUDA driver — two new mini drivers in B1** (F1) | **4–5 d** |
| **B2 — GEMV / expert families** | `s2_gemv` (63,1), `s2_gemv_fast` (153,1), `s2_gemv_q8` (96,1, 1 dp4a), `s2_gemv_quads` (93,1), `bf16_gemv` (141,3), `shared_expert` (346,9), `s2_expert_grouped` (790,13), `native_mmvq` (1,493,9, 12 packed bytes) — 3,175 lines, 38 kernels | `native_mmvq` is the largest file and the packed-byte second hub; `s2_expert_grouped` is 13 kernels incl. the grouped GEMV; Phase A's T4 naive GEMV (4.6–16.9 G weights/s) is the baseline the `fast`/`quads` variants must beat — the perf headline of the ledger | `s2_gemv` ✓ (+ `s2_gemv_q8` ✓, `s_gemv_q8k` ✓ for the family), `bf16_gemv` ✓, `shared_expert` ✓, `native_mmvq` ✓ (mmvq_multi_parity); `s2_expert_grouped` → **new driver (~1 d)** per the B0 audit | 5–6 d (+1 d driver) |
| **B3 — KV + quantised weight access** | `kv_q4` (231,4), `kv_stream` (266,4), `iq_kernels` (748,8, 25 dp4a, 28 packed bytes) — 1,245 lines, 16 kernels | **the packed-byte/dp4a risk hub** — B0.2's payoff lands here; heavy quant-format bit twiddling | `kv_q4` ✓, `kv_stream` ✓, `kv_hybrid` ✓, `iq` ✓ | 2–3 d |
| **B4 — GDN / GR attention family** | `gdn` (248,5), `fused_gdn` (166,3), `native_gdn` (122,1), `native_gdn_preprocess` (199,5), `native_flash_attn` (212,3, **assigned by B0**), `gr` (432,6), `fused_gr` (405,5, 2 dynamic-smem sites), `native_gr_norm` (102,1, moved from B1 by the B0 audit) — 1,886 lines, 29 kernels | shuffle-heavy recurrent attention (the research doc's "attention/GDN" risk); `fused_gr` carries dynamic smem (B0.4); B0.3 primitives exercised at scale | `gdn` ✓, `gr` ✓; `fused_gdn`, `native_gdn`, `native_gdn_preprocess` (1.5–2 d), `native_flash_attn` (1 d), `native_gr_norm` (0.5 d) → **new drivers** per the B0 audit | 4–5 d (+3–3.5 d drivers) |
| **B5 — QSA family (hardest after T5)** | `qsa` (819,8, 1 dynamic-smem), `qsa_decode_attn` (298,2), `qsa_prompt_attn` (717,2, 9 shfl, 2 dynamic-smem), `qsa_select` (581,6, 30 smem sites, TF32-asm), `native_qsa_indexer` (263,4), `native_qsa_score` (213,2, tensor-core `mma` + **existing FP32-FMA fallback** — the fallback path is what ports; tensor cores dropped cleanly, research doc), `native_qsa` (127,2, moved from B1 by the B0 audit) — 3,018 lines, 26 kernels | densest shuffles + smem of the port; `qsa_prompt_attn` is prefill-shaped attention (kernel in scope; its host flow may touch Phase C); local-memory budget per kernel is the likely finding source (convention 9) | `qsa` ✓ (`qsa_parity`, `ngram.cpp`), `qsa_prompt_attn` ✓, `native_qsa_indexer` ✓ (qsa_parity); `qsa_select` (1 d), `native_qsa` + `native_qsa_score` (1–1.5 d) → **new drivers** per the B0 audit | 6–8 d (+2–2.5 d drivers) |
| **B6 — MoE + PLE** | `native_moe` (86,1), `ple` (367,8), `native_ple_postops` (221,7), `native_router` (127,4, **assigned by B0**) — 801 lines, 20 kernels | low-medium; `ple` has the `ple_oracle_vectors.inc` oracle to reuse; `native_router` adds 4 light shuffles | `ple` ✓; `native_moe` + `native_ple_postops` (1–1.5 d), `native_router` (0.5 d) → **new drivers** per the B0 audit | 1–2 d (+1.5–2 d drivers) |
| **B7 — verify kernels** | `verify_kernels` (559,23) — 23 kernels | **no parity driver exists** (B0.1): new driver, new observability contract (host users are Phase C's `verify.cpp` — kernel port stands alone); verify-window host staging explicitly out of scope | **new driver (~2 d)** per the B0 audit | 2 d (+2 d driver) |

Batch order rationale (same as Phase A's): B1 keeps the pipeline warm on
low-risk files; B2 lands the perf headline early; B3 isolates the
intrinsic-layer risk where it is concentrated; B4/B5 escalate shuffle
pressure gradually; B6/B7 are tail items. A batch may slip earlier if it
unblocks a perf question; a kernel that stalls for toolchain reasons is
isolated (per-file files keep one broken kernel from blocking the rest —
Phase A risk table) and recorded, not silently deferred.

## Validation + records

- **Per batch:** all files' ctests green under `env -u LD_LIBRARY_PATH`,
  `check_mirrors.sh` exit 0 (tree-wide — one red block anywhere fails the
  step), report section per file (what mapped 1:1, what needed glue,
  local-memory budget row, parity numbers, mutation results if any).
- **Per file with order-sensitive logic:** mutation test(s) designed, run,
  red-verified on the designed fixture, reverted — the T4/T5 protocol.
- **Final (B8, est 2 d):** full-suite ctest 100 %; local-memory budget
  table complete (all 40 files); **perf ledger** — one timed run per
  kernel family at fixed shapes (streaming bandwidth for memory-bound,
  G weights/s for GEMV, us/call where Phase A set a number),
  Arc-vs-rated-bandwidth shares, joined to Phase A's ledger;
  `plans/sycl-phase-b-report.md` written — the go/no-go input to Phase C,
  per the research doc's "deciding risk" framing (what B learned about
  shuffle cost, dp4a availability, and smem budgets is exactly what sizes
  C's doorbell/GEMM redesign).

## Risks and stop-and-ask

| risk | handling |
|---|---|
| 32-wide shuffle emulation cost on attention (B0.3) | probed once; if unacceptable → vendor-extension opt-in, decision written. Never a silent re-architecture of a mirrored kernel. |
| `cuda_dp4a` has no oneAPI mapping | bit-level fallback (hip_compat precedent), perf noted per use in the report. |
| Local-memory budget overflow in QSA/attention (convention 9) | finding with the exact missing margin; re-tiling a mirrored shape is a Phase B/C decision, not a silent edit. |
| `__expf` 1-ulp gaps (P2) | parity harnesses catch real damage; knife-edge normalisation gets the P2b treatment or a written finding. |
| TF32 `cvt.rna` emulation not bit-exact | B0 probe decides; if not bit-exact, `qsa_select`'s tolerance path is examined and the finding written before B5 proceeds. |
| New drivers (B7, possibly B5) testing weaker cases than intended | the "what this test can and cannot see" contract is written in each new driver's header (repo convention), and its blind spots go in the report. |
| icpx bugs on edge-case C++ | per-file isolation (Phase A rule); a broken kernel is recorded, the batch continues around it, and the blocklist is in the report. |
| Transient device-lost / ctest timeouts (Steps 3/5/6) | rerun before recording; no numbers from an incident run. |

## Done when (Phase B complete)

- [ ] All 40 remaining files ported (or a written deferral per the repo's
      honest-gap convention stating the exact missing machinery — expected
      none, since Phase C machinery is host-side and the kernels are
      self-contained).
- [ ] Every kernel has a green parity ctest under `env -u LD_LIBRARY_PATH`
      (existing drivers mirrored; new ones written per B0.1, observability
      contract in the header).
- [ ] `check_mirrors.sh` exit 0 across the whole tree.
- [ ] Order-sensitive kernels carry red-verified mutation tests.
- [ ] Local-memory budget table + perf ledger complete in
      `plans/sycl-phase-b-report.md`.
- [ ] `sycl_compat/intrinsics.hpp` complete for the in-scope surface;
      every fallback (dp4a, TF32, shuffle) perf-noted.
- [ ] Phase C handoff: the report states what B learned about the doorbell
      redesign, GEMM replacement, and mapped-staging decisions (shuffle
      cost, smem budgets, dp4a availability).

**Effort:** B0 (2–3 d) + B1–B7 (23–30 d porting **+ ~11.5–13.5 d of
new-driver work** — the B0 audit found 13 driver-less files, not 1–2, and
the B1 audit added two mini-drivers, F1) + B8 (2 d) ≈
**~38–50 dev-days (9–10 weeks)** for one developer with the Arc machine —
inside the research doc's "several-month project" envelope once Phase C is
added, consistent with its "mostly mechanical" read of this phase now that
Phase A's pipeline exists.
