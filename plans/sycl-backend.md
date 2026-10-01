# SYCL backend — feasibility research

Status: research / not started. Date: 2026-07 (engine 0.1.30 baseline).

## Question

Can Strata gain a SYCL backend (Intel Arc / Data Center Max GPUs via oneAPI
DPC++), alongside the existing CUDA and HIP backends?

## Short answer

Feasible in principle, but **not** the way the HIP backend was added.

The HIP backend is cheap because HIP is API-compatible with CUDA: it is ~254
lines of shim headers (`include/strata/hip_compat/` — `cuda_runtime.h` maps
`cudaX` → `hipX`, `intrinsics.hpp` supplies missing device builtins), force-
included, and the same `.cu` sources are retagged as HIP in CMake
(`cmake/hip_backend.cmake`).

SYCL is **not** CUDA-API-compatible. A shim of that kind can only cover the
host runtime calls. The ~16.5k lines of device code need real porting, and one
core architectural mechanism — CUDA graph capture — has no SYCL equivalent at
all.

## What the engine actually uses

### Host/runtime surface (`src/core`, `src/prefill`)

| Used by Strata | Volume (approx.) | SYCL mapping |
|---|---|---|
| Streams, events, async memcpy | ~900 call sites | queues / events / USM copies — fine |
| Pinned host memory (`cudaMallocHost`, `cudaHostAlloc`) | ~120 | `usm_host_alloc` — fine |
| `cudaFuncSetAttribute` (dynamic smem opt-in) | 7 | `local_accessor` sized at kernel build — rework, doable |
| `cudaLaunchHostFunc` (host callbacks) | 1 | `sycl::event` callbacks — doable |
| **Mapped pinned memory** (`cudaHostAllocMapped` + `cudaHostGetDevicePointer`) | ~15 | **No standard SYCL zero-copy mapped-host memory.** Used for GPU↔CPU doorbells, expert staging (`remote_experts.cpp`, `layer.cpp`, `native_head.cpp`, `verify.cpp`) |
| **Stream-capture → graph instantiate/replay** (`GraphRegistry`, `src/core/graph.cpp`) | 39 graph launches; per-token-count captured graphs | **No SYCL equivalent.** Backbone of the CPU/GPU overlap engine; correctness depends on capture ordering (documented in `graph.cpp` comments) |
| cuBLAS prefill GEMM + hipBLASLt solution-table tuning (`src/prefill/gemm.cu`) | — | No cuBLAS analog; hand-written bf16 GEMM or oneDNN integration |
| GGML MMQ prefill (via `third_party/ggml` CUDA backend; `moe_mmq.cu`, `ggml_cuda_host.cu`) | optional path (`STRATA_PREFILL_MMQ`, HIP-only today) | ggml has no SYCL backend |

### Device kernel surface (`src/kernels`, ~40 files, ~16.5k lines of `.cu`)

- 229 `__global__` kernels, 210 `__syncthreads`, ~100+ warp shuffles
  (`__shfl_xor_sync`, `__shfl_down_sync`, `__shfl_sync`, `__shfl_up_sync`, ...).
- Kernels assume **warp = 32 lanes**. The HIP port targets wave32 RDNA only
  (`cmake/hip_backend.cmake` refuses other wave widths). Intel GPUs are
  16-wide SIMD (EU vector width 16), so shuffle-heavy attention/GDN kernels
  need shared-memory portable shuffles or vendor extensions — the main
  per-kernel risk.
- CUDA-specific builtins:
  - `cuda_dp4a` (int8 dot, 25 uses). Xe exposes an int8 dot in SPIR-V; a
    mapping likely exists but must be verified (cf. `hip_compat/intrinsics.hpp`
    `dp4a` → `__builtin_amdgcn_sudot4`).
  - Packed-byte builtins `__byte_perm`, `__vsub4`, `__vcmpne4`, `__vadd4`.
  - `__ldg`, `__expf`, `__fadd_rn`/`__fmaf_rn`, `__threadfence_system`,
    `__syncwarp`, `__popc`, `__trap`.
  - `__nv_bfloat16` / `__half` → `sycl::ext::oneapi::bfloat16` + half —
    mechanical.
  - All of the above would go into a `sycl_compat/intrinsics` header, same
    pattern as `hip_compat/intrinsics.hpp`.
- Inline assembly: only **2** sites. The tf32 tensor-core `mma` in
  `native_qsa_score.cu` **already has an FP32-FMA fallback** (the sm_75
  Turing path; HIP compiles it out too). Tensor cores can therefore be dropped
  cleanly.

### Hardware side (works in our favor)

- SYCL via oneAPI DPC++ (icx, Level Zero offload) runs on Intel Arc A/B and
  Data Center Max.
- Arc B580 (12–16 GB) and A770 (16 GB) sit in Strata's 12–24 GB class, with
  ~500+ GB/s memory bandwidth.
- The RAM/SSD tiering design is platform-agnostic.
- Caveat: Level Zero/SYCL on Windows is recent (oneAPI 2025+); Linux is the
  mature path. Strata is Windows-first, so this matters for packaging.

## Proposed phases

1. **Phase A — proof of concept (gate for everything).**
   - A `sycl_compat` shim for host calls (mirror of `hip_compat/cuda_runtime.h`).
   - Port 3–5 representative kernels (dequant, one GEMV, router, sampler).
   - Drive them through the existing `*_parity` harness. The repo's parity-test
     culture (GPU kernel vs CPU reference vs pinned llama.cpp oracle) is the
     asset that makes this tractable — each kernel is validated against the
     same oracles the CUDA/HIP ports used.
   - Gate criteria: parity green on an Intel GPU, measured bandwidth in the
     range expected for the design.
2. **Phase B — kernel port (~16k lines).**
   - Mostly mechanical; time goes into warp-width / shuffle and packed-byte
     builtins.
3. **Phase C — the hard part (months-scale, deciding risk).**
   - Redesign the doorbell/overlap protocol without graph capture: explicit
     launch sequences or Level-Zero command-list replay, with correctness
     re-proven.
   - Replace cuBLAS prefill GEMM (hand-written bf16 GEMM or oneDNN).
   - Replace mapped-host staging for doorbells / remote experts / native head.
4. **Phase D — packaging.**
   - DPC++ in CMake: a third mutually-exclusive backend option next to
     `STRATA_ENABLE_CUDA` / `STRATA_ENABLE_HIP`.
   - oneAPI installer steps in `setup.py` (mirroring `./setup.sh --backend
     hip`).
   - Calibration (`--calibrate`), docs (cf. `docs/AMD_HIP.md`), multi-backend
     runtime detection in `src/core/device.cu`.

## Verdict

- Realistic as a **several-month project** with a well-defined spike first:
  Phase A gates everything.
- Not a repeat of the HIP port: the shim trick covers maybe the host layer;
  Phase C's graph-capture and mapped-memory redesign is the deciding risk.
- If the goal is specifically "run Strata on Intel Arc", the cheapest first
  investment is the parity-harness proof of concept (Phase A).
- If the goal is "a third API backend on existing NVIDIA/AMD hardware", SYCL
  adds nothing over what CUDA/HIP already provide — do not pursue.

## Key files to start with

- `cmake/hip_backend.cmake` — the template for wiring a second backend in.
- `include/strata/hip_compat/` — the template for the SYCL shim headers.
- `src/core/graph.cpp` — graph-capture overlap design to redesign.
- `src/core/layer.cpp`, `src/core/remote_experts.cpp`, `src/core/native_head.cpp`,
  `src/core/verify.cpp` — mapped-host-memory users.
- `src/kernels/cuda/native_qsa_score.cu` — the only asm/tensor-core kernel
  (fallback already exists).
- `src/prefill/gemm.cu`, `src/prefill/hipblaslt_tuning.hpp` — BLAS replacement.
- `src/kernels/*_parity*.cpp` — the validation harness to reuse.
