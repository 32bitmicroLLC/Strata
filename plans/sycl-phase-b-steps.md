# Phase B — execution steps

Parent plan: `plans/sycl-phase-b.md`. This file breaks that plan into
ordered, concrete steps. Each step states what to do and what "done"
means. The parent's **standing conventions** (mirror discipline +
`check_mirrors.sh`, parity culture with host-serial references, 1,024
padded work-groups with bounds checks, no subgroup API, throws-not-status,
mutation tests for order-sensitive logic, knife-edges as written findings,
no new public headers, per-kernel local-memory budget, the
build/run/`env -u LD_LIBRARY_PATH` protocol) bind on every step and are
not repeated here.

The per-file protocol is the parent's "Porting batches" paragraph, applied
file by file: read CUDA source + driver → write `poc/sycl/kernels/<name>.cpp`
(mirror blocks + glue) → write/mirror the parity driver → wire CMake
(static lib + exe + ctest, TIMEOUT 300, env pins where a kernel has
multiple paths, per the Phase A pattern) → checker green → compile green →
ctest green → mutation test(s) where the file's output depends on
selection/tie-break/stage order → report section. A step is done when all
its files are green **and** recorded in `plans/sycl-phase-b-report.md`.

Sequencing is the parent's risk order; batches may slip earlier if they
unblock a perf question. A kernel stuck on a toolchain bug is isolated,
recorded, and worked around — never silently deferred (expected deferrals:
none, since Phase C machinery is host-side; if a kernel genuinely needs it,
the deferral states the exact missing pieces per the repo's honest-gap
convention).

## Step 1 — B0: audit + intrinsic layer (est 2–3 d)

Infrastructure every later step draws on, built and probed once.

- **1.1 Coverage audit (parent B0.1).** Resolve the parent's
  confirmed / to-verify / new-driver table into a fixed file → parity
  driver mapping; deliverable: the resolved table (in this file's result
  block and the report), which fixes every later step's driver work.

  **RESULT (done):** the resolved table is in
  `plans/sycl-phase-b-report.md` §B0.1. Corrections to the parent: 44
  `.cu` files total (not 42) → **40 remaining / 12,402 lines**; two
  previously unassigned files placed (`native_flash_attn` → Step 5/B4,
  `native_router` → Step 7/B6); **13 files need new drivers** (not 1–2):
  `fused_gdn`, `native_flash_attn`, `native_gdn`, `native_gr_norm` (Step
  5); `native_moe`, `native_ple_postops`, `native_router` (Step 7);
  `native_qsa`, `native_qsa_score`, `qsa_select` (Step 6); `s2_expert_grouped`
  (Step 3); `verify_kernels` (Step 8); plus `native_gdn_preprocess`
  (Step 5). ~10–12 dev-days of that driver work lands in Steps 3–8.
  `native_gr_norm` and `native_qsa` move out of Step 2 (B1) per the audit.
- **1.2 `sycl_compat/intrinsics.hpp` + probes (parent B0.2).**
  - Packed bytes `__byte_perm`/`__vsub4`/`__vcmpne4`/`__vadd4` — u32
    bit-level implementations, each parity-probed against the CUDA
    builtins.
  - `cuda_dp4a` — probe the SPIR-V/oneAPI int8-dot mapping **first**;
    absent → bit-level fallback, perf noted (the `hip_compat` precedent).
  - **TF32 `cvt.rna` emulation** (`qsa_select.cu:174`) — bit-exactness
    probe. If not bit-exact: examine `qsa_select`'s tolerance path and
    write the finding **before Step 6 proceeds** — this is a standing
    stop condition, not something Step 6 discovers late.
  - The rest in one pass: `__ldg` (plain loads, perf-note), `__expf`
    (Phase A's P2 finding governs: harnesses catch real damage;
    knife-edge normalisation gets the P2b double-math treatment or a
    written finding), `__fadd_rn`/`__fmaf_rn` (standard), `__popc`,
    `__trap`, `__threadfence_system` (verify queue-scope semantics at its
    use sites), `__int_as_float` (bit cast), `__nv_bfloat16` →
    `sycl::ext::oneapi::bfloat16`, `__half` → `sycl::half` (T4: no
    `sycl::fp16` — validated bit-exact).
- **1.3 32-wide shuffle primitives (parent B0.3).** `shfl_xor/shfl_down/
  shfl_up/shfl` over local memory inside the 1,024 work-group,
  generalising the T3/T5 barrier-tree folds; one implementation, probed
  once on a representative attention-shaped workload. **Stop-and-ask if
  the measured cost is unacceptable** → Intel vendor-extension opt-in,
  decision written. The decision lands **before** Steps 5 and 6
  (the shuffle-heavy batches), not inside them.
- **1.4 Dynamic-smem sizing (parent B0.4).** The 5 `cudaFuncSetAttribute`
  sites (`fused_gr` ×1, `qsa` ×1, `qsa_prompt_attn` ×2, `qsa_select` ×1;
  the parent's "6" counted a comment line in `fused_gr.cu`) →
  call-site-sized `local_accessor`s (no kernel-struct pattern — T3);
  each sized and tabled in `plans/sycl-phase-b-report.md` §B0.4, with its
  128 KiB budget position.
- **1.5 Tree-layout decision (parent B0.5).** Confirm `poc/sycl/` stays
  the working tree through Phase B (recommendation: yes); Phase D wires
  the main build.

Done when: coverage table resolved; the probe suite (packed bytes, dp4a,
TF32, shuffle, smem sizes) green or fallbacks adopted and documented;
decisions written in the report. The Step 1 outputs gate Steps 2–8.

**RESULT (done):** all eight B0 probes green (ctest 27/27 incl. Phase A's
19); `check_mirrors.sh` exit 0 (259 OK blocks); dp4a fallback stands
(no oneAPI int8 dot); TF32 emulation **bit-exact** (the Step 6 gate is
satisfied); shuffle primitives accepted at ~275 ns/shuffle (decision in
report §B0.3); `__trap` silent-no-op finding recorded (report §B0.2);
smem table complete (5 sites); tree layout documented.

## Step 2 — B1: elementwise / streaming batch (est 3–4 d)

9 files / 1,718 lines / 30 kernels: `dequant_bf16` (248), `rope` (136),
`native_rope` (102), `elementwise` (325, 14 kernels), `quantize_act` (313),
`cvec` (166, 2 shfl), `native_bf16` (186), `native_gr_postops` (115),
`kv_q8` (127). (`native_gr_norm` moved to Step 5 and `native_qsa` to
Step 6 by the B0 audit.)

Low-risk by the parent's hazard table: light shuffle use (2 sites each in
`cvec`/`elementwise`), no packed bytes, small smem. Driver coverage is
fully resolved by the Step 1 table — every file in this batch has an
existing driver (`dequant_bf16` ✓, `rope` ✓, `cvec` ✓, `elementwise` ✓,
`quantize_act` ✓, `kv_q8` ✓, `native_rope` ✓, `native_bf16` ✓,
`native_gr_postops` ✓) — so this step is mirror-only, no new drivers.

Done when: all 9 files mirror-green, ctest-green, recorded.

## Step 3 — B2: GEMV / expert families (est 5–6 d + 1 d driver)

8 files / 3,175 lines / 38 kernels: `s2_gemv` (63), `s2_gemv_fast` (153),
`s2_gemv_q8` (96, 1 dp4a), `s2_gemv_quads` (93), `bf16_gemv` (141, 3
kernels), `shared_expert` (346, 9), `s2_expert_grouped` (790, 13),
`native_mmvq` (1,493, 9, 12 packed-byte sites).

- The **perf headline** of the ledger: Phase A's T4 naive `s_gemv`
  measured 4.6–16.9 G weights/s on the two real expert shapes — the
  `fast`/`quads`/`grouped` variants are the expected beaters; time all of
  them at the same fixed shapes and table the family.
- `native_mmvq` is the largest file of the port and the second
  packed-byte hub after `iq_kernels` — Step 1.2's packed-byte layer is
  exercised here first at scale.
- `s2_gemv_q8` carries the batch's single `cuda_dp4a` — the Step 1.2
  dp4a mapping (or fallback) is validated here.
- Drivers: `s2_gemv` ✓ (+ `s2_gemv_q8` ✓, `s_gemv_q8k` ✓ for the family),
  `bf16_gemv` ✓, `shared_expert` ✓, `native_mmvq` ✓ (mmvq_multi_parity);
  `s2_expert_grouped` → **new driver (~1 d)** per the Step 1 audit table,
  with its observability contract in the header.

Done when: all 8 files green and recorded; the GEMV-family perf table in
the report with the T4 baseline quoted.

## Step 4 — B3: KV + quantised weight access (est 2–3 d)

3 files / 1,245 lines / 16 kernels: `kv_q4` (231, 4), `kv_stream` (266, 4),
`iq_kernels` (748, 8, **25 `cuda_dp4a` + 28 packed-byte sites — the risk
hub**).

- `iq_kernels` is where Step 1.2's layer is most tested; if the dp4a
  fallback (not the hardware mapping) is in use, this file's perf is the
  number that justifies — or challenges — that decision; table it.
- Drivers all exist: `kv_q4` ✓, `kv_stream` ✓, `kv_hybrid` ✓, `iq` ✓ —
  mirror, don't write.

Done when: all 3 files green and recorded; dp4a availability outcome
(hardware mapping vs fallback + perf) settled in the report — this is the
last chance before B4/B5 rely on the intrinsic layer.

## Step 5 — B4: GDN / GR attention family (est 4–5 d + 3–3.5 d drivers)

8 files / 1,886 lines / 29 kernels: `gdn` (248, 5), `fused_gdn` (166, 3),
`native_gdn` (122, 1), `native_gdn_preprocess` (199, 5), `native_flash_attn`
(212, 3 — assigned here by the B0 audit), `gr` (432, 6), `fused_gr`
(405, 5), `native_gr_norm` (102, 1 — moved here from B1 by the B0 audit).

- Recurrent-attention kernels — the first real scale test of the Step 1.3
  shuffle primitives and of barrier-tree cost at 1,024 threads; the
  measured numbers go in the report (they size the B5 plan, not a guess).
- `fused_gr` carries 2 of the 6 dynamic-smem sites (Step 1.4 sizing must
  have covered them); its 21 smem sites are the batch's budget row.
- Drivers: `gdn` ✓, `gr` ✓, `fused_gr` ✓ (cvec_parity, gr_parity); new
  drivers per the Step 1 table: `fused_gdn` + `native_gdn` +
  `native_gdn_preprocess` (1.5–2 d), `native_flash_attn` (1 d),
  `native_gr_norm` (0.5 d) — each with its "what this test can and cannot
  see" contract in the header. The fused kernels' host flow may touch
  Phase C machinery, but the kernel + parity port stands alone per the
  parent's boundary rule.

Done when: all 6 files green and recorded; shuffle-cost numbers at this
scale written.

## Step 6 — B5: QSA family (est 6–8 d + 2–2.5 d drivers)

The hardest batch after Phase A's T5. 7 files / 3,018 lines / 26 kernels:
`qsa` (819, 8, 1 dynamic-smem site), `qsa_decode_attn` (298, 2),
`qsa_prompt_attn` (717, 2, **9 shfl + 2 dynamic-smem sites**),
`qsa_select` (581, 6, 30 smem sites, the TF32 `cvt.rna` asm),
`native_qsa_indexer` (263, 4), `native_qsa_score` (213, 2, tensor-core
`mma` — port the **existing FP32-FMA fallback path**, tensor cores dropped
cleanly), `native_qsa` (127, 2 — moved here from B1 by the B0 audit).

- **Prerequisite (hard):** the Step 1.2 TF32 probe outcome is bit-exact
  or the finding is written and the tolerance path decided — no
  `qsa_select` work on an unresolved asm emulation. **RESOLVED (Step 1
  done):** the emulation is bit-exact (0/1,056,744 differ), so this gate
  passes with no tolerance-path examination.
- `qsa_prompt_attn` is prefill-shaped attention: the kernel port is in
  scope; its host flow may lean on Phase C machinery — boundary rule
  applies, note it where touched.
- **Local-memory budget per kernel is the likely finding source** in this
  batch (parent convention 9): the 30 smem sites in `qsa_select` and the
  tile shapes in `qsa_prompt_attn` get budget rows; an overflow is a
  finding with the exact missing margin — re-tiling a mirrored shape is a
  decision, not a silent edit.
- Selection/tie/stage order in `qsa_select` and the score/indexer
  kernels → **mutation tests** per the standing convention.
- Drivers: `qsa` ✓ (`qsa_parity`, `ngram.cpp`), `qsa_prompt_attn` ✓,
  `qsa_decode_attn` ✓ (qsa_prompt_attn_parity, kv_hybrid, kv_stream),
  `native_qsa_indexer` ✓ (qsa_parity); new drivers per the Step 1 table:
  `qsa_select` (1 d — reuses `qsa_select_bench` data generation; its
  bench is not a parity), `native_qsa` + `native_qsa_score` (1–1.5 d) —
  each with its own observability contract in the header.

Done when: all 6 files green and recorded (or the recorded findings say
otherwise); mutation results in the report; the QSA smem-budget table
complete.

## Step 7 — B6: MoE + PLE (est 1–2 d + 1.5–2 d drivers)

4 files / 801 lines / 20 kernels: `native_moe` (86, 1), `ple` (367, 8),
`native_ple_postops` (221, 7), `native_router` (127, 4 — assigned here by
the B0 audit; 4 light shuffles).

Low-medium hazard; `ple` reuses the `ple_oracle_vectors.inc` oracle.
Drivers: `ple` ✓; new drivers per the Step 1 table: `native_moe` +
`native_ple_postops` (1–1.5 d — `native_expert_parity` covers the
native-expert pair, confirm the `native_moe` coverage against it first),
`native_router` (0.5 d) — each with its observability contract in the
header.

Done when: all 3 files green and recorded.

## Step 8 — B7: verify kernels (est 2 d)

1 file / 559 lines / 23 kernels: `verify_kernels.cu`.

- **No parity driver exists** — this step writes a new one. Its header
  carries the "what this test can and cannot see" contract (repo
  convention), and the contract's blind spots go in the report.
- The host users (`src/core/verify.cpp` — the verify window, the
  `cudaLaunchHostFunc` flag-raise) are Phase C: the kernel port + parity
  stand alone, per the parent's boundary rule. State exactly what the
  driver cannot exercise (the window's async flag protocol) so Phase C is
  not surprised.

Done when: kernel green, new driver green, contract + blind-spot notes in
the report.

## Step 9 — B8: full regression + ledger + report + handoff (est 2 d)

- Full-suite ctest under `env -u LD_LIBRARY_PATH` — every parity entry
  green; `check_mirrors.sh` exit 0 across the whole tree (one red block
  anywhere fails the step).
- **Local-memory budget table** complete (all 40 files, worst-shape rows,
  128 KiB cap position) — the Phase A §T5 table format.
- **Perf ledger**: one timed run per kernel family at fixed shapes
  (streaming bandwidth for memory-bound, G weights/s for GEMV, us/call
  where Phase A set a number), Arc share of rated bandwidth; joins Phase
  A's ledger in the report.
- **`plans/sycl-phase-b-report.md`** written — the go/no-go input to
  Phase C: what B learned about shuffle cost, dp4a availability, smem
  budgets, and the fallback perf costs is exactly what sizes C's
  doorbell/GEMM/mapped-staging redesign.

Done when: suite 100 %, checker exit 0, budget table + ledger complete,
report written with the Phase C handoff section.

## Stop conditions (phase-level)

- **TF32 emulation not bit-exact** (Step 1.2) → `qsa_select` work (Step 6)
  does not start until the tolerance path is examined and the finding
  written.
- **Shuffle emulation cost unacceptable** (Step 1.3) → vendor-extension
  decision written before Steps 5/6; if rejected, the batch plan changes
  (finding), it is not absorbed silently.
- **Smem budget overflow in a mirrored shape** → finding with the exact
  missing margin; re-tiling is a Phase B/C decision (convention 9).
- **Kernel stuck on a toolchain bug** → isolated, recorded, worked around;
  the blocklist lives in the report. Per-file isolation means one broken
  kernel never blocks the rest (Phase A risk-table rule).
- **Transient device-lost / ctest timeouts** → rerun before recording; no
  numbers from an incident run (Phase A Steps 3/5/6 precedent).
- **Any kernel found to need Phase C machinery** → written deferral with
  the exact missing pieces (expected: none).

## Total

Step 1 (2–3 d) + Steps 2–8 (23–30 d porting **+ 10–12 d of new-driver
work** — the B0 audit found 13 driver-less files, not 1–2; the per-step
driver estimates are annotated above) + Step 9 (2 d) ≈ **~37–48 dev-days
(8–10 weeks)** — the parent's original 27–36 d envelope re-based by the
B0 audit, still inside the research doc's "several-month project" once
Phase C is added.
