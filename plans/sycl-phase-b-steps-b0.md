# B0 step — audit + intrinsic layer (`plans/sycl-phase-b-steps-b0.md`)

Parent: `plans/sycl-phase-b.md` (B0 = parent §"B0 — audit + intrinsic layer"),
step definition in `plans/sycl-phase-b-steps.md` Step 1. This plan pins every
B0 item against the actual tree. The coverage audit was executed while
writing this plan — its results below **correct the parent plan** (file
counts, two unassigned files, and the new-driver count), and closing those
corrections is part of B0's done-when.

Environment reminder (Phase A §Environment): Linux dev machine, Arc
Battlemage G31, oneAPI 2026.1, **no CUDA device** — every intrinsic's
parity reference is a host C++ implementation of the documented CUDA
semantics (the `hip_compat/intrinsics.hpp` validation pattern), not a
CUDA cross-check.

## Audit results (executed at plan-writing time)

1. **File counts**: `src/kernels/cuda/` holds **44** `.cu` files / 14,306
   lines (the parent plan's "42" undercounted). Phase A ported 4
   (`dequant_s2`, `router_top10`, `s_gemv`, `sampler` — 1,904 lines).
   **Remaining: 40 files / 12,402 lines.**
2. **Two files were never assigned to a parent batch**: `native_flash_attn`
   (212 lines, 3 shfl) → **B4** (attention family); `native_router` (127
   lines, 4 shfl) → **B1** (low-risk batch).
3. **13 files have no parity driver** (the parent assumed 1–2):
   `fused_gdn`, `native_flash_attn`, `native_gdn`, `native_gdn_preprocess`,
   `native_gr_norm`, `native_moe`, `native_ple_postops`, `native_qsa`,
   `native_qsa_score`, `native_router`, `s2_expert_grouped`, `qsa_select`
   (its `qsa_select_bench` is a bench, not a parity), `verify_kernels`.
   Driver work estimated per family below — **~10–12 dev-days**, added to
   the phase total.
4. **`dp4a.hpp` already exists in the repo** (`include/strata/kernels/`):
   `STRATA_DP4A(a, b, c)` selects `__dp4a` on sm ≥ 61 and a **bit-exact
   fallback** below (signed-byte 4-dot, int32 accumulation — llama.cpp's
   own). The SYCL wrapper mirrors this pattern, and the repo's fallback
   semantics *are* the parity reference.
5. **`mrope.hpp` is not a kernel file**: it is host/device table plumbing
   (no `mrope.cu` exists); the rope-scaling logic it names is implemented
   inside `rope.cu`/`native_rope.cu` (both already covered by
   `rope_parity`). No port action; noted so B0's table isn't read as
   missing it.

## 1.1 — resolved coverage audit (deliverable)

Method: each kernel's exported functions (per `include/strata/kernels/*.hpp`)
grepped across all `src/kernels/*parity*.cpp`, `*test*.cpp`, `ngram.cpp`.
Substring false-positives checked (e.g. `native_qsa` hits were all
`native_qsa_indexer`; `qsa_select` hits were all `qsa_selection_width`).

| status | files |
|---|---|
| **Ported (Phase A)** | `dequant_s2`, `router_top10`, `s_gemv`, `sampler` |
| **Covered — mirror the existing driver** (31) | `bf16_gemv` (bf16_gemv_parity), `cvec`, `dequant_bf16` (dequant_bf16_test, elementwise_parity), `elementwise`, `fused_gr` (cvec_parity, gr_parity), `gdn`, `gr`, `iq_kernels` (iq_parity, mmvq_multi_parity, native_expert_parity), `kv_q4`, `kv_q8`, `kv_stream` (all three + kv_hybrid_parity), `native_bf16` (ple_parity, shared_expert_parity), `native_gr_postops` (gr_parity), `native_mmvq` (mmvq_multi_parity, iq_parity, ple_parity), `native_qsa_indexer` (qsa_parity), `native_rope` (rope_parity), `ple`, `qsa` (qsa_parity, qsa_prompt_attn_parity, kv_*), `qsa_decode_attn` (qsa_prompt_attn_parity, kv_hybrid, kv_stream), `qsa_prompt_attn` (qsa_prompt_attn_parity, kv_hybrid), `quantize_act` (s2_gemv_q8, s_gemv_q8k, shared_expert parities), `rope` (rope_parity, qsa_parity), `s2_gemv` (+fast, +quads via s2_gemv_parity/s_gemv_parity), `s2_gemv_q8`, `shared_expert` |
| **No driver — write new** (13) | the audit's list above; batch assignments: GDN family of 3 + `native_flash_attn` → **B4**; `native_gr_norm` → **B4**; `native_moe`, `native_ple_postops`, `native_router` → **B6** (router could sit in B1 — B6 keeps the MoE/PLE pair together); `native_qsa`, `native_qsa_score` → **B5**; `s2_expert_grouped` → **B2**; `qsa_select` → **B5** (reuses `qsa_select_bench` data generation); `verify_kernels` → **B7** |

New-driver estimate by family (each driver carries its own observability
contract in the header, repo convention): GDN family 3 files 1.5–2 d,
`native_flash_attn` 1 d, `native_gr_norm` 0.5 d, `native_moe` +
`native_ple_postops` 1–1.5 d, `native_qsa` + `native_qsa_score` 1–1.5 d,
`native_router` 0.5 d, `s2_expert_grouped` 1 d, `qsa_select` 1 d,
`verify_kernels` (23 kernels) 2 d → **~10–12 d total**, landed inside the
batch steps, not B0.

**Deliverable:** this table copied into the report, and the parent plan +
steps file corrected (counts 44/40/12,402; the two batch assignments; the
13-driver list in the affected batch entries; effort re-based to
**~37–48 dev-days (8–10 weeks)**: parent 27–36 + driver work 10–12).

## 1.2 — `sycl_compat/intrinsics.hpp` + probes

New header `poc/sycl/include/sycl_compat/intrinsics.hpp` (pattern:
`hip_compat/intrinsics.hpp`). Probes under `poc/sycl/b0/`, each a
standalone exe + ctest entry wired into `poc/sycl/CMakeLists.txt` per the
Phase A `t5/` probe pattern; all run under `env -u LD_LIBRARY_PATH`.

| probe | what it settles | reference (host C++) | gate |
|---|---|---|---|
| `b0_dp4a_probe.cpp` | whether oneAPI 2026.1 exposes a usable int8 dot, else the fallback stands | the repo's `STRATA_DP4A` fallback semantics (signed bytes, int32 acc) | bit-exact over a random int32-pair grid + corner values; decision (mapping vs fallback) written |
| `b0_byteperm_probe.cpp` | `__byte_perm` LUT semantics | CUDA-documented 32-bit byte-perm table re-implemented in host u32 | bit-exact, random (a, b, sel) |
| `b0_packed4_probe.cpp` | `__vsub4`, `__vcmpne4`, `__vadd4` (sbyte4 ops) | host sbyte4 loops, same overflow/compare semantics | bit-exact, random u32 pairs |
| `b0_tf32_probe.cpp` | bit-exactness of the `cvt.rna.tf32.f32` emulation (`qsa_select.cu:174`) | host round-to-nearest-ties-away to 10-bit mantissa per PTX ISA, NaN/±inf preserved | bit-exact over a 2^24 sample grid **plus targeted tie cases** (dropped mantissa == 0x1000) |
| `b0_bf16_probe.cpp` | `__nv_bfloat16` → `sycl::ext::oneapi::bfloat16` bit layout | mirror of the repo's `bf16_bits_test` semantics | bit-exact |
| `b0_misc_probe.cpp` | `__popc`, `__int_as_float`, `__fadd_rn`/`__fmaf_rn` mappings | host builtins/ops | bit-exact |

Documented-only (no probe, semantics stated in the header): `__ldg` (plain
load — perf-note when first used at scale), `__expf` (Phase A P2 finding
governs: 1-ulp device gaps documented, harnesses catch real damage,
knife-edges get the P2b treatment or a written finding), `__trap` (abort
semantics — verified at its use site in B4/B5, not probed in B0),
`__threadfence_system` (queue-scope; semantics verified at use sites).

**Hard stop condition:** `b0_tf32` not bit-exact → the finding is written
and `qsa_select`'s tolerance path examined **before Step 6 (B5) starts** —
this gate is checked at the Step 6 boundary, not discovered mid-step.

## 1.3 — 32-wide shuffle primitives

Implementation in `intrinsics.hpp`, generalising the T3/T5 barrier-tree
folds. "Warp" = 32 consecutive tids; group `g` owns local-memory slots
`[32g, 32g+32)`:

- `shfl_xor_32(v, mask)` / `shfl_down_32(v, delta)` / `shfl_up_32(v,
  delta)` / `shfl_broadcast_32(v, lane)`: write `v` to own slot, barrier,
  read slot `(lane ⊕ mask)` / `(lane + delta) mod 32` / `(lane − delta)
  mod 32` / slot `lane`. (CUDA's `__shfl_*` are in-register; this is the
  local-memory substitute — same *values*, more barriers.)
- **Correctness probe** `b0_shfl_probe.cpp`: all four variants, random
  values, all 1,024 threads, vs host shfl semantics (xor wraparound,
  up/down edge behaviour at lane boundaries — CUDA leaves edges unchanged,
  the port must match).
- **Perf probe** `b0_shfl_bench.cpp`: an attention-shaped use (32-lane
  butterfly rounds via `shfl_xor` + a broadcast-heavy reduce) timed,
  reported in ns/shuffle and ns/round, alongside the local-tree baseline
  the T3/T5 ports use where it was cheaper.
- **Decision (written, before Steps 5/6):** acceptable → the primitives are
  the standard and the tree folds stay for bulk reductions; unacceptable →
  the Intel vendor-extension opt-in decision is written (Phase A
  risk-table rule: vendor extension *only if* the cost is unacceptable).
  The number, not the hope, is what goes in the report.

## 1.4 — dynamic-smem sizing (the 5 opt-in sites)

Measured from source (the parent's "6 sites" counted a comment line in
`fused_gr.cu`):

| site | kernel | CUDA dynamic size | SYCL port |
|---|---|---|---|
| `fused_gr.cu:337` | `gr_down_multi_kernel` | `kFusedGrMaxT * TILE * 4`, source comment: **80 KB** at full shape | 80 KB = ~61 % of the 128 KiB cap — fits, headroom ~46 KB; tabled row |
| `qsa.cu:682` | `qsa_attend_kernel` | call-site `smem` (expression read at port time) | sized at the call site; budget row in B5 |
| `qsa_prompt_attn.cu:622` | `prompt_attn_i8_kernel` | call-site `bytes` (i8 variant) | sized at the call site; budget row in B5 |
| `qsa_prompt_attn.cu:653` | `prompt_attn_kernel<KV_MODE>` | call-site `bytes` (templated on KV mode) | sized per instantiation; budget row in B5 |
| `qsa_select.cu:532` | `block_scores_tc_kernel` | `(TC_QT*TC_QS + TC_NB*TC_KS) * 4` | sized at the call site; budget row in B5 |

Mechanics: CUDA's opt-in exists because >48 KB dynamic smem needs the
attribute; SYCL has no such split — the `local_accessor` is simply sized at
the call site (no kernel-struct pattern, T3 finding), so "sizing" means
carrying the same size expression into the handler and recording its
128 KiB position. No behaviour change; the table is the record.

## 1.5 — tree-layout decision

`poc/sycl/` **stays the working tree through Phase B** (confirmed
recommendation): per-file `poc/sycl/kernels/<name>.cpp` mirroring
`src/kernels/cuda/<name>.cu`, per-kernel static libs (`k_<name>`, the Phase
A pattern), drivers in `poc/sycl/drivers/`, probes in `poc/sycl/b0/`
(this step) and per-batch probe dirs if needed; `check_mirrors.sh` covers
the whole `poc/sycl/` tree. Phase D moves it into the main build with
`cmake/hip_backend.cmake`-style wiring. Nothing in B0 writes under
`src/`.

## Stop conditions

- **TF32 not bit-exact** → written finding + `qsa_select` tolerance decision
  before Step 6 (hard gate, §1.2).
- **Shuffle cost unacceptable** → vendor-extension decision written before
  Steps 5/6 (§1.3); the batches do not start on an open question.
- **dp4a mapping absent** → fallback stands (the repo's bit-exact
  semantics), perf noted — not a stop.
- **Any probe red that is not an intrinsic-semantics error** → the probe
  or the harness is wrong; fix and re-run, do not "adopt the fallback"
  around a harness bug (Phase A Step 4 lesson: checker-green ≠ compiles,
  green ≠ right).

## Done when (B0 complete)

- [ ] Resolved coverage table recorded in `plans/sycl-phase-b-report.md`
      (new §B0); parent plan + steps file corrected (44/40 files, 12,402
      lines, the two batch assignments, the 13-driver list, effort re-based
      to ~37–48 dev-days).
- [ ] `poc/sycl/include/sycl_compat/intrinsics.hpp` complete for the §1.2
      set; all six probes wired as ctests and green (or fallbacks/decisions
      written); TF32 outcome recorded.
- [ ] Shuffle primitives implemented; correctness probe green; perf number
      + decision written (the Step 5/6 gate).
- [ ] Dynamic-smem sizing table (§1.4) complete in the report.
- [ ] Tree-layout decision documented (§1.5).
- [ ] Full existing suite still green under `env -u LD_LIBRARY_PATH`
      (Phase A's 19/19 + the new B0 probe entries); `check_mirrors.sh`
      exit 0.

Est: **2–3 d** (the audit is already executed here; ~1.5–2 d of probe
building, the rest documentation and the parent-file corrections). The
10–12 d of new-driver work is *not* B0 — it lands in Steps 2–8, which is
why the phase total re-bases.
