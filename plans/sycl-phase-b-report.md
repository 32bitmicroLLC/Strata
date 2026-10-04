# Phase B — SYCL kernel port: report

Parent plan: `plans/sycl-phase-b.md`; step decomposition:
`plans/sycl-phase-b-steps.md`. This report is the running record of what
each batch/file ported, what needed glue, the local-memory budget table,
the perf ledger, and — for Phase C — what B learned about shuffle cost,
dp4a availability, and smem budgets. Written batch by batch; §B0 below
closes the infrastructure step.

Status: **B0 done.** Batches B1–B8 pending.

---

## B0 — audit + intrinsic layer (done)

Environment (Phase A §Environment, restated because it governs every
intrinsic reference below): Linux dev machine, Intel Arc Battlemage G31,
oneAPI 2026.1 (icpx `-fsycl -O2 -std=c++20`), **no CUDA device**. Every
intrinsic's parity reference is a host C++ implementation of the documented
CUDA semantics (the `include/strata/hip_compat/intrinsics.hpp` validation
pattern), not a CUDA cross-check. Build/run: `poc/sycl/build`,
`cmake --build . -j`, tests under `env -u LD_LIBRARY_PATH`.

### B0.1 — resolved coverage audit (corrects the parent's counts)

**File counts (corrected):** `src/kernels/cuda/` holds **44** `.cu` files /
14,306 lines (the parent plan's "42" undercounted). Phase A ported 4
(`dequant_s2`, `router_top10`, `s_gemv`, `sampler` — 1,904 lines).
**Remaining: 40 files / 12,402 lines.** The parent's Scope and batch table
are corrected to these numbers.

**Two files were never assigned to a parent batch** (now assigned):
- `native_flash_attn` (212 lines, 3 `__shfl_*`) → **B4** (attention family).
- `native_router` (127 lines, 4 `__shfl_*`) → **B6** (keeps the MoE/PLE pair
  together; could equally sit in B1).

**13 files have no parity driver** (the parent assumed 1–2). This is the
bulk of the phase's hidden driver work (~10–12 dev-days, landed in the
batch steps, not B0):

| new-driver file | batch | est |
|---|---|---|
| `fused_gdn` | B4 | (GDN family) |
| `native_gdn` | B4 | 1.5–2 d for the 3 |
| `native_gdn_preprocess` | B4 |  GDN files |
| `native_flash_attn` | B4 | 1 d |
| `native_gr_norm` | B4 | 0.5 d |
| `native_moe` | B6 | (MoE/PLE pair) |
| `native_ple_postops` | B6 | 1–1.5 d |
| `native_router` | B6 | 0.5 d |
| `native_qsa` | B5 | (QSA pair) |
| `native_qsa_score` | B5 | 1–1.5 d |
| `s2_expert_grouped` | B2 | 1 d |
| `qsa_select` | B5 | 1 d (reuses `qsa_select_bench` data gen) |
| `verify_kernels` | B7 | 2 d (23 kernels, no parity at all) |

Method: each kernel's exported functions (per `include/strata/kernels/*.hpp`)
were grepped across all `src/kernels/*parity*.cpp`, `*test*.cpp`, and
`ngram.cpp`; substring false-positives were checked (e.g. `native_qsa`
hits were all `native_qsa_indexer`; `qsa_select` hits were all
`qsa_selection_width`).

**Resolved file → driver table** (the parent's "to-verify" list resolved):

| status | files |
|---|---|
| **Ported (Phase A)** | `dequant_s2`, `router_top10`, `s_gemv`, `sampler` |
| **Covered — mirror the existing driver** (31) | `bf16_gemv` (bf16_gemv_parity), `cvec`, `dequant_bf16` (dequant_bf16_test, elementwise_parity), `elementwise`, `fused_gr` (cvec_parity, gr_parity), `gdn`, `gr`, `iq_kernels` (iq_parity, mmvq_multi_parity, native_expert_parity), `kv_q4`, `kv_q8`, `kv_stream` (all three + kv_hybrid_parity), `native_bf16` (ple_parity, shared_expert_parity), `native_gr_postops` (gr_parity), `native_mmvq` (mmvq_multi_parity, iq_parity, ple_parity), `native_qsa_indexer` (qsa_parity), `native_rope` (rope_parity), `ple`, `qsa` (qsa_parity, qsa_prompt_attn_parity, kv_*), `qsa_decode_attn` (qsa_prompt_attn_parity, kv_hybrid, kv_stream), `qsa_prompt_attn` (qsa_prompt_attn_parity, kv_hybrid), `quantize_act` (s2_gemv_q8, s_gemv_q8k, shared_expert parities), `rope` (rope_parity, qsa_parity), `s2_gemv` (+fast, +quads via s2_gemv_parity/s_gemv_parity), `s2_gemv_q8`, `shared_expert` |
| **No driver — write new** (13) | the table above; batch assignments in the new-driver table |

Two audit corrections that change no batch but close a parent footnote:
- **`dp4a.hpp` already exists** in the repo (`include/strata/kernels/`):
  `STRATA_DP4A(a,b,c)` selects `__dp4a` on sm ≥ 61 and a **bit-exact
  fallback** below (signed-byte 4-dot, int32 accumulation — llama.cpp's
  own). The SYCL wrapper mirrors this pattern, and the repo's fallback
  semantics *are* the parity reference (no CUDA cross-check available).
- **`mrope.hpp` is not a kernel file**: host/device table plumbing (no
  `mrope.cu` exists); the rope-scaling logic it names lives inside
  `rope.cu`/`native_rope.cu` (both covered by `rope_parity`). No port
  action; noted so the table is not read as missing it.

### B0.2 — `sycl_compat/intrinsics.hpp` + probes (all green)

New header `poc/sycl/include/sycl_compat/intrinsics.hpp` (pattern:
`hip_compat/intrinsics.hpp`). Every `__` name is a free global so mirrored
CUDA bodies compile verbatim; the macro block at the bottom wires the CUDA
spellings to the implementations. Probes under `poc/sycl/b0/`, each a
standalone exe + ctest entry (Phase A `t5/` probe pattern), run under
`env -u LD_LIBRARY_PATH`.

**ARCH GATE (documented, binds every port):** each kernel `.cpp` defines
`__CUDA_ARCH__` as 750 in glue before its mirror blocks. That selects the
repo's own non-sm80 (Turing) paths inside the mirrored `#if` ladders: the
fp16/tf32 `mma` helpers compile to `__trap()` (their kernels are
runtime-rejected; the FMA fallback kernels run instead), `tf32_hi`
degenerates to `__float_as_uint`, the `cp.async` helpers become
`__trap()`, and `STRATA_DP4A` takes the native `__dp4a` branch of
`include/strata/kernels/dp4a.hpp` (mapped in the header). The `asm` lines
inside `#if STRATA_*_SM80` branches therefore **never compile** in the
SYCL tree. This is why the tensor-core `mma` in `native_qsa_score` and the
sm80 GEMM paths drop cleanly — the fallback path is what ports.

Probe results (all exit 0; full-suite ctest 27/27):

| probe | settles | result |
|---|---|---|
| `b0_dp4a_probe` | oneAPI int8 dot vs fallback | **PASS, 0/65536 differ.** oneAPI 2026.1 exposes no usable int8 dot → the repo's bit-exact `STRATA_DP4A` **fallback stands** (signed bytes, int32 acc mod 2^32). Perf cost noted per use at B3 (`iq_kernels`) / B2 (`native_mmvq`). Not a stop (hip_compat precedent). |
| `b0_byteperm_probe` | `__byte_perm` LUT semantics | **PASS, 0/65536 differ.** Bit-exact vs a host u32 re-implementation of the CUDA 32-bit byte-perm table. |
| `b0_packed4_probe` | `__vsub4`/`__vadd4`/`__vcmpne4`/`__vsubss4` | **PASS, 0/262144 values differ.** Bit-exact vs host byte-lane loops (unsigned-byte lanes, mod-256 wrap; `vsubss4` saturates to [−128,127]). |
| `b0_tf32_probe` | `cvt.rna.tf32.f32` bit-exactness (`qsa_select.cu:174`) | **PASS, 0/1056744 differ — bit-exact.** The add-`0x1000`-then-mask emulation is exactly round-to-nearest **ties-away** to a 10-bit mantissa, NaN/±inf preserved, carry rolls into the exponent. **The Step 6 (B5) TF32 gate is satisfied**; no tolerance-path examination needed before `qsa_select` work. |
| `b0_half_probe` | `__half`/`__half2` family (the fp16 the 10 unported kernels use) | **PASS, 0/1064518 differ.** `__float2half_rn` bit-exact RNE (incl. subnormal, tie-at-2^-14, 65504↔inf corners); f→h→f→h round trip identity; `__hsub2`/`__floats2half2_rn`/`__halves2half2` vs host refs. |
| `b0_misc_probe` | `__popc`, `__umulhi`, `__float2int_rn/rz`, `__*_as_*` bit casts, round-to-nearest scalars, `__threadfence` | **PASS, 0 mismatches.** `__threadfence()` compiles (2026.1 has no `sycl::fence(fence_access, domain)`; the 2023-style `sycl::atomic_fence(seq_cst, scope_device)` is what the runtime ships) and a fenced atomic round trip is correct. See the `__expf`/`__fdividef` measured gaps and the `__trap` finding below. |
| `b0_shfl_probe` | 32-wide shuffle correctness (B0.3) | **PASS, 0/18432 checks differ.** All four `shfl_*` variants + `ballot` + `syncwarp`, random values, all 1,024 threads, vs host shfl semantics (xor wraparound; up/down edges leave the edge lane unchanged — matched). |
| `b0_shfl_bench` | 32-wide shuffle cost (B0.3 decision input) | **PASS; numbers below.** |

**Deviations from the B0.2 plan table (documented, not silent):**
- The plan listed a `b0_bf16_probe` for `__nv_bfloat16`. **The CUDA
  `__nv_bfloat16` type is absent from all 40 unported kernels** (verified
  by grep): the BF16 kernels (`bf16_gemv`, `dequant_bf16`, `elementwise`,
  `fused_gr`, `gr`, `native_bf16`, `ple`, `shared_expert`) use `uint16_t`
  plus the repo's portable `include/strata/kernels/bf16_bits.hpp`
  (`bf16_from_f32`/`f32_from_bf16`, bit-exact, already validated in the CUDA
  tree and SYCL-compilable — `STRATA_BF16_HD` is empty without
  `__CUDACC__`/`__HIPCC__`). So no bf16 probe is needed; the `__nv_bfloat16`
  typedef in the header is kept only as Phase-C future-proofing.
  `b0_half_probe` covers the `__half`/fp16 family the 10 unported kernels
  actually use (`iq_kernels`, `native_flash_attn`, `native_mmvq`,
  `native_ple_postops`, `native_qsa_indexer`, `qsa`, `qsa_decode_attn`,
  `qsa_prompt_attn`, `s2_expert_grouped`, `dequant_bf16`).
- `__expf` is mapped to device `std::expf`. Measured max ulp gap vs host
  `expf` over a 2^20 grid: **64 ulp** (GPU libm ≠ CPU libm). Recorded, not
  gated — Phase A's P2 precedent governs: parity harnesses catch real
  damage; any knife-edge normalisation in a B-kernel gets the P2b
  double-math treatment or a written finding.
- `__fdividef` is mapped to correctly-rounded `/`. Measured max ulp gap vs
  host `a/b`: **2 ulp**. Recorded; each use site re-checked at port time,
  any dependence on the CUDA approximation is a written finding.
- **`__trap` finding (the one real B0 finding):** on the target Arc device
  with the standard in-order queue, `__builtin_trap()` is a **silent
  no-op** — the kernel reports completed, **no `sycl::exception` is
  delivered** at event completion, the host process stays alive. (A bare
  default-queue kernel segfaulted instead; the behaviour is
  queue/device-dependent and is never a clean host-observable abort.) The
  CUDA abort semantics have no kernel-side equivalent that DPC++ 2026.1
  accepts (no in-kernel exceptions, no fence-based abort). In the ports
  `__trap()` only ever *compiles* on the runtime-rejected sm80 paths
  (mma/cp.async helpers), which are never launched, so the no-op is dead
  code in practice — but a port that actually reached one would silently
  continue on Arc, which is a written finding, not a silent tolerance, and
  such a site must be re-examined at port time.

### B0.3 — 32-wide shuffle primitives (correctness green; perf + decision written)

Implementation in `intrinsics.hpp`: a `shfl32` object over work-group local
memory, generalising the T3/T5 barrier-tree folds. A "warp" is 32
consecutive work-group items; group `g` owns local-memory slots
`[32g, 32g+32)`; storage is 2×u32 per lane (8 KiB) + one u32 per group for
the `syncwarp` arrival barrier. The kernel set only ever passes the full
mask, mirroring the HIP header's `require_full_wave_mask`; a non-full mask
`__trap()`s (dead code, see B0.2 finding).

Measured cost (`b0_shfl_bench`, 1,024 threads, Arc e223, median):

| quantity | measured | note |
|---|---|---|
| one `shfl_xor` (store + 1024-wide `it.barrier()` + load) | **~275 ns/shuffle** | dominated by the 43 ns `it.barrier()` plus local-mem store/load |
| `__syncwarp` (32-wide spin barrier) | **~2,278 ns/sync** | 53× a single `it.barrier()`; see decision |
| 1024-wide `it.barrier()` (baseline) | ~43 ns/barrier | the hardware barrier |

**Decision (written; gates Steps 5/6):** **acceptable — the local-memory
primitives are the standard; no vendor extension.** Rationale:
- Correctness is exact (0/18432), and the ~275 ns/shuffle cost is a
  constant multiplier on the number of `__shfl_*` sites in a kernel, not a
  re-architecture. The shuffle-heavy files (`qsa_prompt_attn` 9 shfl,
  `native_flash_attn` 3, `native_router` 4) still carry their mirrored
  logic; the perf ledger (B8) records the real cost per kernel, and any
  kernel whose shuffle count makes 275 ns/shuffle material at its shape
  gets a written perf finding — not a silent vendor-extension swap.
- The Phase A risk-table rule ("vendor extension *only if* the cost is
  unacceptable") is not triggered: 275 ns is measured and bounded, and the
  correctness ports do not depend on beating it.
- **`__syncwarp` note (perf, not a stop):** the 32-wide spin barrier is
  ~2,278 ns — 53× the 43 ns hardware `it.barrier()`. It was kept because it
  is semantically narrower (joins only the warp's 32 lanes, never
  deadlocks on divergent per-group call counts), which matters for
  correctness; a kernel that turns out to call `__syncwarp` in a tight loop
  is a perf finding for B4/B5, and the port may then map `__syncwarp` to
  `it.barrier()` at that site (a written decision), not a silent edit.

### B0.4 — dynamic-smem sizing (5 opt-in sites, tabled)

CUDA's `cudaFuncSetAttribute` opt-in exists because >48 KB dynamic smem
needs the attribute; SYCL has no such split — the `local_accessor` is sized
at the call site (no kernel-struct pattern, T3 finding). "Sizing" means
carrying the same size expression into the handler and recording its
position against the 128 KiB per-work-group cap. No behaviour change; this
table is the record. The parent's "6 sites" counted a comment line in
`fused_gr.cu`; the measured count is **5**:

| site | kernel | CUDA dynamic size | SYCL budget (128 KiB cap) |
|---|---|---|---|
| `fused_gr.cu:337` | `gr_down_multi_kernel` | `kFusedGrMaxT * TILE * 4`, source comment: **80 KB** at full shape | 80 KB ≈ **61 %** of the cap; headroom ~46 KB — fits, tabled |
| `qsa.cu:682` | `qsa_attend_kernel` | call-site `smem` (expression read at port time) | sized at the call site; budget row lands in B5 |
| `qsa_prompt_attn.cu:622` | `prompt_attn_i8_kernel` | call-site `bytes` (i8 variant) | sized at the call site; budget row lands in B5 |
| `qsa_prompt_attn.cu:653` | `prompt_attn_kernel<KV_MODE>` | call-site `bytes` (templated on KV mode) | sized per instantiation; budget row lands in B5 |
| `qsa_select.cu:532` | `block_scores_tc_kernel` | `(TC_QT*TC_QS + TC_NB*TC_KS) * 4` | sized at the call site; budget row lands in B5 |

An overflow in a mirrored shape is a finding with the exact missing margin
(convention 9); re-tiling is a B/C decision, not a silent edit.

### B0.5 — tree-layout decision (documented)

`poc/sycl/` **stays the working tree through Phase B** (the confirmed
recommendation). Per-file `poc/sycl/kernels/<name>.cpp` mirrors
`src/kernels/cuda/<name>.cu`; per-kernel static libs (`k_<name>`, the Phase
A pattern); drivers in `poc/sycl/drivers/`; probes in `poc/sycl/b0/` (this
step) and per-batch probe dirs if needed; `check_mirrors.sh` covers the
whole `poc/sycl/` tree. Nothing in B0 (or any B step) writes under `src/`.
Phase D moves the tree into the main build with
`cmake/hip_backend.cmake`-style wiring and moves the kernel declarations to
`include/` (convention 8: no new public headers in B).

### B0 done-when checklist

- [x] Resolved coverage table recorded here (§B0.1); parent plan + steps
  file corrected (44/40 files, 12,402 lines, the two batch assignments,
  the 13-driver list, effort re-based to ~37–48 dev-days).
- [x] `intrinsics.hpp` complete for the §B0.2 set; all probes wired as
  ctests and green; TF32 outcome recorded (bit-exact, gate satisfied).
- [x] Shuffle primitives implemented; correctness probe green; perf number
  + accept decision written (the Step 5/6 gate).
- [x] Dynamic-smem sizing table (§B0.4) complete.
- [x] Tree-layout decision documented (§B0.5).
- [x] Full existing suite green under `env -u LD_LIBRARY_PATH` — ctest
  **27/27** (Phase A's 19 + the 8 new B0 probes); `check_mirrors.sh`
  **exit 0** (259 OK blocks, whole tree).

---

*(Batches B1–B8 append their sections here as they complete.)*
