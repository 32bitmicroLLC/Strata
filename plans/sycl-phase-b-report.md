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

---

## B1 — elementwise / streaming batch (in progress)

Plan: `plans/sycl-phase-b-steps-b1.md`. Steps 2.1–2.3 complete as of this
writing; 2.4–2.10 pending.

### B1.1 — `rope.cpp` + `native_rope.cpp` + `rope_parity` (steps 2.1–2.2)

- **Mirrored 1:1**: all five rope table-build/validation blocks, the rope
  rotation kernel body, the native_rope `enabled`/`overlaps` host checks and
  kernel body, `mrope_tab` init. Driver: all five parity sections (table,
  rotation, pairing, scaled tables 4a–4g, native-vs-table 5) mirrored verbatim.
- **Glue**: G1 2-D grid decomposition in `native_rope` (`blk = gid/threads;`\
  `blockIdx.x = blk % gridX; blockIdx.y = blk / gridX`); G7 `mrope_pos`
  defined in both kernel TUs (F7); G8 device management + the `cs5` non-null
  stream marker for `submit_native_rope`'s mirrored null-stream check; `q.memcpy`
  instead of `cudaMemcpy` (the `q.copy` USM path segfaults in driver
  1.14.37020 — recorded, T5-known class).
- **Parity**: all 12 checks green — table bit-exact; rotation agrees (worst
  1.089e-07, driver's own tolerance); pairing NEOX; scaled tables bit-exact;
  yarn structure confirmed; native-vs-table (yarn, factor 2) worst
  8.631e-05; (none) worst 5.619e-05.
- **Mutations** (T4/T5 protocol, both reverted green after):
  1. rope: deleted the `n_rot..head_dim` tail copy → **36,864 over 1e-6**
     (exactly the tail channels) → detected.
  2. native_rope: swapped the two rotation outputs → **24,452/24,460 over
     3e-3** (worst 1.953/1.826) → detected.

### B1.2 — `quantize_act.cpp` + `quantize_act_parity` (step 2.3)

- **Mirrored 1:1**: all five kernel bodies (q8_0, q8_0_scaled, dequant_q8_0,
  q8_K, dequant_q8_K), `nearest_int_dev`, the launch-size computations, the
  `QK8_0`/`QK_K`/`Q8K_BYTES` constants, and in the driver both reference
  transcriptions, both run_cases, the tie fixture, the -127 trap, the fp16
  overflow/subnormal regression table, and `main`.
- **Glue**: G1 dim3 in each body; G8 validation throws (`std::runtime_error`)
  and empty-case no-op submissions; `using std::min`/`std::max` for the
  unqualified `min(127, v)` (CUDA device builtin);
  **intrinsic-layer fix**: `fmul_rn` in `intrinsics.hpp` now uses a `volatile`
  store so icpx cannot contract the multiply into the following
  `+ 12582912.0f` (the CUDA `__fmul_rn` semantics the q8_K tie fixture is
  built to catch) — a glue-layer change, no mirrored line touched.
- **Parity, Q8_0: exact.** All 5 distributions × 131,072 elements: blocks
  byte-exact, round trip bit-exact. This closes the plan's stated knife-edge
  (device **double** division, line 72): IEEE-exact on Arc, P2b-consistent.
  The fp16 overflow/subnormal regression block and the tie-discrimination
  fixture (522,240 ties, 49.8 % discriminating) pass; the -127-vs--128 trap is
  observable (0.915 % of magnitude, 172,016 bytes differ).
- **Parity, Q8_K: red by construction — finding F8** (full analysis in the
  B1 step file; evidence in `b0/b0_divprobe.cpp`, `b0/b0_divprobe2.cpp`,
  `b0/b0_q8k_audit.cpp`):
  - host `1.0f/x` (icpx x86, -O0…-O2): **0/4,194,304** off vs exact RNE;
  - device `1.0f/x` (Arc, same compiler): **1 ulp on 14 %** of operands,
    identical at -O0/-O1/-O2 and under `-fp-model precise` → a codegen
    property (reciprocal-style lowering), not an optimization artifact;
  - Q8_K chain `-127.0f/max; 1.0f/iscale`: **711/2048 blocks (34.7 %) with a
    differing f32 `d`** (586 ×1 ulp, 125 ×2–4 ulp); **every `qs`, `bsum`, and
    all Q8_0 bytes exact** — kernel logic fully faithful, residual solely the
    non-IEEE division.
  - **Disposition**: `k_quantize_act_parity` runs with ctest `WILL_FAIL TRUE`
    — a loud, documented expected failure, NOT a silent tolerance. The Q8_0
    half already passes; the Q8_K byte comparison waits on the Phase C gate
    decision (fixed icpx, or a driver update with a documented 1-ulp band on
    `d`/iscale-derived `qs`).
- **Mutation test** (plan candidate: divide by the fp16-rounded scale instead
  of `d32`, Q8_0 line 72): applied to the kernel, rebuilt, driver Q8_0 half
  shows **136,515 failures** (random normal 949 bad bytes; small magnitude
  129,011 — the tiny `d32` rounds to a subnormal/zero fp16 and every quant
  flips; exact-.5 boundaries 2,293; one-extreme-per-block 0, as expected); Q8_K
  half red as in the unmutated baseline (F8, unchanged signature). Reverted via
  backup copy: Q8_0 back to **0 failures**, Q8_K signature unchanged. Green
  confirmed.

### B1.3 — `native_gr_postops.cpp` + `native_gr_postops_parity` (step 2.4)

- **No CUDA parity driver exists for this file (finding F1)**: the dedicated
  driver was written in new mode with a documented contract (host C++
  references transcribed from the kernel math; fixtures: random + signed-zero
  + aliased/non-aliased buffer cases). Mirrored: all three kernel bodies
  (`down_silu`, `pre_gated<Fused>` ×2, `post`), `sigmoid`, `scale_zero_bias`,
  `DSV4`/`DSV4_HC_POST` constants, the launch geometry, and the host
  validation.
- **Intrinsic-layer fix (G5)**: the Arc device `fmaf` does NOT implement
  IEEE signed-zero on a zero addend — `fmaf(s, -0.0f, +0.0f)` yields `-0.0f`
  (the sign of the product) where IEEE requires `+0.0f`; and icpx -O2 folds
  `x + 0.0f → x` and `std::fma(a, b, 0.0f) → a*b` even through volatile reads,
  so the host reference cannot use either spelling. `fmaf_rn` in
  `intrinsics.hpp` now corrects the zero-addend case by bit inspection, and
  the driver's reference uses a bit-inspecting `fma32_ref()` instead of
  performing the add. The mirrored kernel line (`__fmaf_rn(scale, x, 0.0f)`)
  is untouched.
- **Parity: within documented ULP gates** (no division on these paths, but the
  64-ulp device-vs-host `expf` gap from the B0 misc probe accumulates over the
  `hc`-term sums): down_silu ≤128 ulp (measured 2), gate ≤128 ulp (measured
  8), mixed ≤8192 ulp (measured 160/743), post ≤16384 ulp (measured 5218/7179).
  Signed-zero fixtures bit-exact after the G5 fix.
- **Mutation test** (plan candidate: replace `scale_zero_bias` with a plain
  `__fmul_rn`): both signed-zero fixtures RED (f16 sign-bit flips), revert →
  GREEN.

### B1.4 — `dequant_bf16.cpp` + `iq_dequant.cpp` + `dequant_bf16_parity` (step 2.5)

- **F4 closure**: `dequant_bf16.cu` hard-references the iq-dequant surface of
  `iq_kernels.cu`; that closure (cvt, ten `dq_*` device functions,
  `dq_dispatch`, `dequant_flat_kernel`, `is_iq`/`iq_supported`/
  `iq_row_bytes`, the two launchers, plus host-decode hooks for the driver)
  is ported in `iq_dequant.cpp`. The block structs and codebook grids come
  from `third_party/ggml/ggml-common.h` under its `GGML_COMMON_*_SYCL` macros
  — plain host `static const` tables, probed reachable from Arc kernels
  (the CUDA file uses the `*_CUDA` variants; documented glue difference).
- **Compiler exception (documented, not a code change)**: the CUDA source's
  `const int8_t s[N] = {(int8_t) …}` brace initializers use C-style casts,
  a narrowing conversion clang rejects and nvcc accepts (all values fit).
  `k_dequant_bf16` compiles with `-Wno-c++11-narrowing`; mirrored lines stay
  byte-verbatim.
- **Parity: all 16 types bit-exact** (4×1024 synthetic values each; f32, f16
  and bf16 paths; zero-scale and NaN-scale superblocks in every type):
  types 2, 6, 8, 11, 12, 13, 14, 20, 23, 42 against the `dequant.hpp` chain
  (independent; worst relative error 0.00 — no division on these paths, so the
  F8 gap cannot bite), types 16, 17, 18, 21, 22, 29 against
  `host_decode_iq_*` (the same mirrored `dq_dispatch` code run on the host).
- **Reference-coverage gap (recorded, not papered over)**: the chain has no
  iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_m dequantizers, so the six iq-only
  types check device launch/memory/indexing against their own transcription,
  not an independent reference. True parity for them waits on a shard or a
  chain extension.
- **NaN-payload platform difference (mutation-test finding)**: f2bf's NaN
  guard is load-bearing on NVIDIA (a small-payload float NaN rounds to +inf
  without it) but NOT on Arc: the device's half→float conversion emits a
  large-payload NaN (0x7FC02000 where the host yields 0x7F802000 for fp16
  0x7C01) that survives the rounding add as a quiet NaN. Removing the guard
  is therefore undetectable by any fixture on this machine; the guard stays
  mirrored verbatim (defensive parity).
- **Mutation test** (plan candidate: drop Q5_0's high-bit term `xh1`): type 6
  RED (955 f32 / 965 f16 / 955 bf16 off), revert → GREEN. (The NaN-guard
  mutation above is the second candidate; it is uncatchable on Arc, hence the
  bit-level Q5_0 bit instead.)
- **Port bug caught by the selftest** (before closeout): `dequant_flat_body`
  initially used the raw global id as `blockIdx.x` (one work-group per
  superblock); the driver's iq types failed 3814/4096 — fixed, all green.

### B1.5 — `elementwise.cpp` + `elementwise_parity` (step 2.6)

- **Scope**: all 14 kernels of `elementwise.cu` @ 22d022e (embedding gather,
  gdn gate, scale/add, f16/bf16 bridges, silu, rms_norm_weighted, three
  doorbell kernels, three mapped-copy kernels) plus the host dispatch, in
  `poc/sycl/kernels/elementwise.cpp`; the driver mirrors the CUDA driver
  `elementwise_parity.cpp` @ 9a8fc3f (its seven covered entries, comparison
  logic verbatim) and adds an F6 execution smoke (below).
- **Kernel-side findings (measured, all inside the mirrored gates)**:
  - `rsqrtf` is NOT available to DPC++ device code (host-only libm macro;
    probed: "SYCL kernel cannot call an undefined function"). Per the plan it
    is a glue macro `1.0f / sqrtf(x)`; rms_norm_weighted measures
    rel 4.05e-08 / worst 2.26e-07 against the f64 oracle, far under the CUDA
    driver's own 1e-6 gate, and both rival readings stay 70–89 % apart.
  - Device DOUBLE `exp` (silu) and device `log1pf`/`expf` (gdn_gate) compile
    and run on Arc: silu rel 0.000e+00 (bit-exact vs the double reference),
    gdn_gate rel 2.48e-08 ≤ 1e-6.
  - `rms_norm_weighted`'s warp shuffles run on the B0 `shfl32` local-memory
    emulation; the row guard (the QSA bug the comment describes) is exercised
    by the padded 4-warp grid exactly as in CUDA.
  - Work-groups of 1 (doorbell ring/wait) and 1024 (doorbell publish) are both
    accepted on Arc; `__threadfence_system` maps to the B0 `threadfence()`
    (G7 — the protocol itself is Phase C's gate), `strata_spin_pause` is an
    empty spin (no device `__nanosleep`; dp4a.hpp deliberately not included,
    B0 macro clash), `__shared__ int hit` becomes a handler-built
    local_accessor with glue `it.barrier()` lines.
- **Host-reference-side findings (driver-side, both measured, both pinned —
  the mirrored kernel lines are untouched)**:
  - icpx host -O2 transforms the scale oracle's single f32 multiply by the
    compile-time constant `1.0f/sqrt(128.0f)`: 466 of the 1024 products came
    1 ulp away from a true single f32 multiply (the device kernel's multiply
    is the true one — a volatile-pinned reference matches it 1024/1024). The
    oracle pins the multiply volatile, as the driver's own embedding comment
    requires.
  - icpx host `std::fma(f32)` is a TWO-rounding implementation (probed:
    514,084 of 531,441 samples differ from the `vfmadd231ss` result), so it
    cannot stand in for the fused product the embedding fixture's
    `fma_diff > 0` non-vacuity check asserts. The reference uses the FMA3
    instruction directly; with it the fixture shows 8141 FMA differences
    (the check is non-vacuous) while the device output stays 0/108 cases off
    (the kernel's `__fmul_rn`/`__fadd_rn` glue keeps the separate roundings).
- **Parity: all seven covered entries pass under the CUDA driver's own
  gates, verbatim** (gdn 2.48e-08 ≤ 1e-6; silu 0.000e+00 ≤ 1e-7; scale and
  f32→f16 bit-exact; rms 4.05e-08 < 1e-6 with both rival readings observable;
  embedding gather 108 row cases, 0 bit mismatches, 0 guard failures;
  f32→bf16 NaN-kept: bulk 0/4096 vs the ggml rule, dequant type-8 0/64 —
  linking k_dequant_bf16 for that slice).
- **F6 (the plan's six uncovered entries)**: per the plan they are
  compile-only in B1 — the mapped-pinned handoff protocol is gated in Phase C.
  The driver still runs each once with a trivial oracle (identity copies,
  the +0.0 hit row, seq increments, the immediate-exit wait): all pass. The
  live spin path of `doorbell_wait` is NOT exercised: a single in-order queue
  cannot hold a concurrent writer to device memory, and a second-queue race is
  not a deterministic ctest. Recorded, not hidden.
- **The CUDA driver's graph-capture block** is replaced by two ordered
  in-order-queue launches (gather then scale): the same contract (no hidden
  synchronization, caller's queue honored) without stream capture, which SYCL
  does not have.
- **Mutation test** (plan candidate: drop `/ cols` in rms_norm_weighted):
  rel 9.375e-01, gate RED, revert → GREEN (mirror checker exit 0).
- **Suite**: ctest 32/32 green (31 real passes + `k_quantize_act_parity`
  inverted by its documented F8 `WILL_FAIL`), `check_mirrors.sh` exit 0.

### B1.6 — `kv_q8.cpp` + `kv_q8_parity` (step 2.7)

- **Scope**: both kernels of `kv_q8.cu` @ fb2ccec (INT8 KV append with the
  64-value group scale, and the gather that dequantizes selected cells into
  the FP16 scratch) plus host dispatch; the driver mirrors the q8 half of
  `kv_q8_parity.cpp` @ 733d7c7 (checks 1 and 2). Per F3, the FP16 pool calls
  (`kv_append_step`/`kv_gather_step`, CUDA driver lines 44, 66, 101, 111,
  116–117) and check #3 (INT8-vs-FP16 error ≤ 0.624 quantization steps,
  lines 129–131, 138–139) park in B5's qsa driver.
- **Port notes**: the 3-D append grid (n_head_kv, head_dim/64, 2) × 64 is
  flattened to a 1-D launch of 2048 items (16 exact 64-wide work-groups); the
  body resolves blockIdx.x/y/z from the global id (each CUDA block maps to
  exactly one (h, g, is_v)). The block's two CUDA warps are the B0 `shfl32`
  emulation's groups 0 and 1, and the `warp_max` shared float-2 is a
  handler-built local_accessor with glue `it.barrier()` for the
  `__syncthreads` (T3/router_top10/elementwise precedent). `char4`/`ushort4`
  are glue PODs (DPC++'s `sycl::char4`/`ushort4` classes are different types;
  the mirrored `reinterpret_cast` pattern needs trivially-copyable layouts).
  `KvHostPools` is never populated by the driver (the mapped-pinned host pools
  are Phase C's protocol, per the audit): the host-copy branch compiles but
  is not exercised.
- **F8 probe (new, measured)**: the kernel's two divisions behave
  differently on the Arc device — constant-divisor division is IEEE-exact
  (probed: 0/200,000 off), while variable-divisor division is the documented
  F8 gap (53,062/200,000 off, 1 ulp). So `amax / 127.0f` (the scale) is safe,
  and the variable-divisor `x / sf` inside `__float2int_rn` is this step's
  single knife edge: a code byte can flip only when the division lands within
  one f32 ulp of a half-integer tie — the same family as the Q8_K finding.
  The bit-exact gate is kept (not widened): the margin analysis is written
  here, and a future flip on a different fixture is a documented F8 instance,
  not a port bug.
- **Driver-side API note**: 2026.1's `queue::fill` takes
  `fill<T>(ptr, value, count)` with an explicit template argument (the byte-
  count spelling I first tried does not compile); the driver's `dalloc`
  glue uses it with a zero value for the CUDA `cudaMemset`.
- **Parity: checks 1 and 2 bit-exact** (487 filled cells × 2 KV heads ×
  4 groups × 64 values × K,V: 497,664 code bytes and 7,808 scales, all
  identical to the host reference; all 20 gather trials bit-identical to the
  host dequantization of the stored codes) — 0 F8 flips on this fixture.
- **Mutation test** (the order-sensitive logic: the two-warp reduction and
  its fixed-order shared combine): narrowing the butterfly from `o = 16`
  to `o = 8` breaks the warp max — 222,124 codes/scales off, gate RED;
  revert → GREEN (mirror checker exit 0).
- **Suite**: ctest 33/33 green (32 real passes + `k_quantize_act_parity`
  inverted by its documented F8 `WILL_FAIL`), `check_mirrors.sh` exit 0.

### B1.7 — `native_bf16.cpp` + `native_bf16_parity` (step 2.8)

- **Scope**: `src/kernels/cuda/native_bf16.cu` (186 lines, 2 kernels ×
  instantiations via the 2×8 template dispatch macro). Finding F1 from the
  plan: **no CUDA-side parity driver exists** for
  `bf16_gemv_fp32_mmvf` / `bf16_gemv_fp32_mmvf_multi` (they are exercised
  only through gr.cu / shared_expert.cu / ple.cu in B4/B2/B6), so this step
  carries a NEW dedicated driver with its own transcribed reference and its
  own contract.
- **Contract pinned (F1)**: `y[o] = sum_i x[i] * w[o*n_in+i]` — ONE shared
  activation row for every output row; the multi kernel runs one workgroup
  per output column over `n_tok` activation rows and must be
  bit-identical to each row's own single-row launch. The driver fixture
  holds N_OUT+1 rows (1 shared + 8 multi); fixture values are kept in
  |v| ≤ 2^9 so no sum overflows to inf/NaN (payload-bit NaN comparison
  would be a platform artifact, out of scope for a bit-exact gate).
- **Host-reference platform findings (new, all measured)**:
  1. icpx host -O2 **flushes f32 subnormal intermediates to zero in plain
     float addition** (`0x00000002 + 0.0f → 0x00000000`; gcc -O2 and icpx -O0
     are exact). It also flushes double-subnormal intermediates, which is
     why a double-precision FMA ground truth is unreliable under icpx -O2.
  2. icpx host -O2 **miscompiles inline `vfmadd231ss` asm** — a subnormal
     FMA result (e.g. 0x681) comes back as +0; the same bits under gcc -O2
     -mfma and icpx -O0 are exact. A host FMA3-asm reference is therefore
     unusable under icpx -O2.
  3. Consequence: the host reference uses a **pure-C bit-exact f32 FMA**
     (`fmaf_rn_ref`, 49-bit `__uint128_t` product, exact aligned integer sum,
     RNE in subnormal-ulp units, the `__fmaf_rn` glue's signed-zero
     correction mirrored). It is verified **200,000/200,000 bit-exact
     against a gcc -mfma FMA3 build** over full-range finite f32 triples
     (subnormals, zeros, mixed signs included). The butterfly additions are
     the exact integer add `fmaf_rn_ref(a, 1.0f, b)` (the ×1.0 product is
     exact in the integer domain), so **no host f32 arithmetic appears
     anywhere in the references**.
  4. The reference itself needed two fixes found during bring-up (a
     zero-sign branch that encoded `-0` as `S = -1` and thus returned
     `-1.0` for both-zero FMA shapes, and a term-alignment direction bug in
     the sum) — both caught by fixed-case isolation and cross-checked
     before the 200k gate; documented here rather than hidden.
- **Device side**: `std::fmaf` on the Arc device is IEEE-exact on the 200k
  fixture; its only signed-zero gap (exact-zero product with a +0 addend
  returns the product's sign) is the documented B1 glue correction in
  `intrinsics.hpp`. The NEW `b0/b1_fma_zero_probe.cpp` exhaustively asserts
  **device fmaf + glue correction vs the IEEE-exact integer reference on all
  125 degenerate zero combinations** (a, b, c ∈ {+0, -0, +1, -1, 2^-100}):
  **0/125 differ**.
- **float2 glue bug (found under this step, fixed)**: `intrinsics.hpp` had
  declared `struct float2 { float x, y, z, w; }` (16 bytes) instead of CUDA's
  8-byte `{x, y}` — every `float2*`-indexed kernel step read at 2× the right
  offset, which is why the single-row port initially failed on every output.
  Fixed in the glue with a comment; the B0 half probe's initializer was
  updated to match. All other vector glue structs were audited and are
  correct.
- **Port notes**: block sizes 32…256 in steps of 32 launch as padded 1-D
  work-groups with per-item bounds checks (Arc rejects non-uniform
  work-groups); the `__shared__` partials arrays are handler-built local
  accessors with glue `it.barrier()`; `__shfl_xor_sync` butterflies use the
  B0 `shfl32` emulation; the 5-round butterfly and the two-stage shared
  reduction are transcribed verbatim, so the accumulation order is a
  mirrored property, not a tolerance. The kernel's adaptive block-size
  selector is mirrored into the driver (`mmvf_block_size`) and proven to
  reach all eight sizes (reps 2→32, 66→64, 130→96, 194→128, 258→160,
  322→192, 386→224, 450→256).
- **Parity**: single-row bit-exact on all eight block sizes (8 output rows
  each); multi bit-exact vs the host reference for n_tok ∈ {2, 4, 8} and
  bit-identical to each row's own single-row launch (the header contract);
  the n_tok == 1 fast path matches the single-row kernel; the signed-zero
  fixture (w = -0.0, x = 0, acc = +0) passes for the no-shared (32) and
  shared (64) block sizes; 11 mirrored validation throws, 1 legal call
  clean.
- **Mutation test** (the order-sensitive logic): an all-ones fixture makes
  every output the exact integer n_in; flipping one weight bit (1.0f →
  2.0f) moves exactly row 0 from n_in to n_in+1 and no other row — the
  gate is provably not blind.
- **Suite**: ctest **34/34** green (33 prior passes + `k_native_bf16_parity`),
  `check_mirrors.sh` exit 0.

### B1.8 — `cvec.cpp` + `cvec_parity` (step 2.9)

**Scope.** One kernel (`cvec.cu`, 166 lines, source `74d73e7`): per-stream
representation steering — a block-wide dot `s = h·v` (two-stage 256-thread
reduction through `__shared__ float part[8]` and two 16-round `__shfl_xor_sync`
butterflies) followed by the per-element update `h' = h - s·(h·v)·v` (steer,
mode 0), `h' = h + v` (add, mode 1), plus an optional pending fused-GR write
(`write=true`). The fused-GR cross-check and the `write=true` path both
exercise `__expf` (64-ulp gap, B0-measured) and depend on B4's `fused_gr.cu`
→ parked per F2; the driver covers the steering (mode 0, s = 2 and s = 1),
untouched layers, the switch-off gate, the add path, and the mirrored
validation throws. `__shared__` → handler-built `part` local accessor (same
name, so the mirrored lines are unchanged); 2-D `(hc, T)` grid flattened to a
padded 1-D launch; the 64-device table plumbing collapses to a single
`DevTables` slot (the mirrored `kDevices`/`cur_device` stay verbatim, glue
returns 0); SYCL's no-globals-in-kernel rule is met by reading the mirrored
globals into locals before `submit`.

**Finding — the shfl32 join is now the 32-wide spin barrier.**  The B0
`shfl32` emulation joined each shuffle round with the work-group barrier.
That is correct only when the shuffle runs uniformly in every warp.  cvec's
mirrored second reduction stage runs inside `if (threadIdx.x < 32) { … }` —
in CUDA that is a warp-local instruction, but in the emulation it became
five work-group barriers reached by 32 of 256 threads, which the other seven
warps raced past: the block-wide dot came out warp-local (measured 8× low;
fixture: all-ones `h`, unit `v` — 2240 of 2560 elements wrong).  The fix is in
the glue, not the mirror: every `shfl*_sync`/`ballot_sync` now joins with
`syncwarp()` (the 32-wide generation-counter barrier from `b0_shfl_probe`),
which is exactly the participation set a CUDA warp shuffle requires.  All
shuffle-using ports (kv_q8, elementwise, native_bf16, …) were re-run green
after the change.

**Parity.** Project: max |err| **4.9e-7** vs the double-precision reference
(mirrored accumulation order; gate 1e-4 for s = 2, 1e-3 for s = 1), orthogonality
max |h'·v + h·v| 1.6e-6; s = 1 projects the direction out within the same
gate. Bit-exact: an un-steered layer leaves `R` untouched; with the switch
off and no pending write, `R` is bit-unchanged; the add path is `h + d` in
every stream bitwise (the host reference uses the pure-C integer-exact
`fmaf_rn_ref`, because icpx host -O2 flushes f32 subnormals — §B1.7). The
`cvec_enabled()` gate follows the switch; `n_embd` / table-size validation
throws are mirrored.  **Mutation** (measured): flipping the update sign
(`-dot` → `+dot`) fails the project gate by O(1) — max |err| 2.73,
orthogonality max |h'·v + h·v| 35 vs the 1e-4/1e-3 gates.

**Suite.** ctest **35/35** green (34 prior passes + `k_cvec_parity`),
`check_mirrors.sh` exit 0.

*(Batches B1–B8 append their sections here as they complete.)*
