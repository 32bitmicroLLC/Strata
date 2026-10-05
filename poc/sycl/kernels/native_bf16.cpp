// poc/sycl/kernels/native_bf16.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.8)
// SYCL port of src/kernels/cuda/native_bf16.cu @ 1390a6a: the BF16-weight,
// FP32-activation MMVF (single-row and the up-to-8-row multi kernel) plus the
// adaptive block-size selection and both public entry points.
//
// The mirrored blocks below are VERBATIM (checked by check_mirrors.sh).
// Glue:
//  - G4: empty device qualifiers; `dim3` glue resolved per work-item (one
//    work-group per output row, exactly as in CUDA; all eight block sizes
//    32..256 step 32 are accepted by Arc as uniform work-groups -- probed).
//  - `__shfl_xor_sync` expands onto the B0 `shfl32` object (local-memory
//    32-wide warp emulation, B0 §1.3) built in each body; the CUDA
//    `mmvf_warp_sum` helper takes that object as an extra parameter so its
//    mirrored body stays verbatim (the call sites gain the argument).
//  - `__shared__ float partials[32]` / `partials[NT][32]` are handler-built
//    local_accessors (T3 finding; router_top10/elementwise precedent); the
//    multi kernel's 2-D array is flattened to `[k * 32 + lane]`, and every
//    `partials[...]` reference plus every `__syncthreads` is a glue line
//    (documented at the site).
//  - `__fmaf_rn` is the intrinsics.hpp device fma (IEEE on Arc apart from the
//    documented signed-zero correction built into the glue).
//  - Host dispatch: the CUDA launch macros are redefined as glue over
//    q.submit launchers, and the mirrored `switch` lines keep their CUDA
//    spelling; `cudaGetLastError` checks are dropped (SYCL throws);
//    `void* stream` becomes `sycl::queue&`; the single-row fast path's bare
//    `return;` becomes a forwarded `return`.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

// glue: the arch gate (B0 header) -- selects the repo's non-sm80 paths.
#define __CUDA_ARCH__ 750

namespace strata::kernels {
namespace {

// glue (G4): empty device qualifiers so the mirrored lines keep their CUDA spelling.
#define __device__
#define __forceinline__
#define __global__

// glue (G1): CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// glue: CUDA lines 36-41 are a `__device__` helper; its shuffle lines expand
// onto the `shfl32` warp-emulation object, which is passed in so the mirrored
// body stays verbatim (the call sites below gain the object argument).
float mmvf_warp_sum(strata::sycl_compat::shfl32& shfl32, float value) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:37:40 @ 1390a6a
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
    // SYCL-MIRROR-END
}

// The `bf16_f32_mmvf_kernel` `__global__` signature (CUDA lines 44-45) is glue;
// the body (46-70) is mirrored except the shared-memory lines:
//   line 50 `__shared__ float partials[32];`      -> the handler-built `parts`;
//   line 52 `partials[t] = 0.0f;`                 -> glue `parts[t] = 0.0f;`
//   line 53 `__syncthreads();`                    -> glue `it.barrier();`
//   line 63 `mmvf_warp_sum(acc)`                  -> glue with the shfl32 object;
//   line 66 `partials[t / 32] = acc;`             -> glue `parts[t / 32] = acc;`
//   line 67 `__syncthreads();`                    -> glue `it.barrier();`
//   line 68 `mmvf_warp_sum(partials[t])`          -> glue with the shfl32 object.
template <int BLOCK_SIZE>
void bf16_f32_mmvf_body(const float* x, const uint16_t* w, float* y, int n_in, sycl::nd_item<1> it,
                        const sycl::local_accessor<float, 1>& parts,
                        const sycl::local_accessor<uint32_t, 1>& sfl, const sycl::local_accessor<uint32_t, 1>& spin) {
    const dim3 blockDim{BLOCK_SIZE, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / BLOCK_SIZE), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:46:49 @ 1390a6a
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    // SYCL-MIRROR-END
    // glue: CUDA line 50 `__shared__ float partials[32];` is the handler-built `parts`
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:51:51 @ 1390a6a
    if constexpr (BLOCK_SIZE > 32) {
    // SYCL-MIRROR-END
        if (t < 32) parts[t] = 0.0f;   // glue: CUDA line 52 (partials[t] = 0.0f;)
        it.barrier();                  // glue: CUDA line 53 (__syncthreads)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:54:62 @ 1390a6a
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    // SYCL-MIRROR-END
    acc = mmvf_warp_sum(shfl32, acc);  // glue: CUDA line 63 (mmvf_warp_sum(acc))
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:64:64 @ 1390a6a
    if constexpr (BLOCK_SIZE > 32) {
    // SYCL-MIRROR-END
        // (glue: CUDA line 65 -- "All lanes have the same reduced value; one store
        // avoids a same-value shared-memory race.")
        if ((t & 31) == 0) parts[t / 32] = acc;                  // glue: CUDA line 66
        it.barrier();                                            // glue: CUDA line 67
        if (t < 32) acc = mmvf_warp_sum(shfl32, parts[t]);       // glue: CUDA line 68
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:69:70 @ 1390a6a
    }
    if (t == 0) y[blockIdx.x] = acc;
    // SYCL-MIRROR-END
}

// The `bf16_f32_mmvf_multi_kernel` `__global__` signature (CUDA lines 77-78) is glue;
// the body (79-118) is mirrored except the shared-memory lines:
//   line 82 `__shared__ float partials[NT][32];`   -> the flat handler-built `parts`;
//   lines 84-86 `partials[k][t] = 0.0f;`           -> glue over `parts[k * 32 + t]`;
//   line 87 `__syncthreads();`                     -> glue `it.barrier();`
//   lines 104-105 `acc[k] = mmvf_warp_sum(acc[k])` -> glue with the shfl32 object;
//   lines 107-109 `partials[k][t / 32] = acc[k]`   -> glue over `parts[k * 32 + t / 32]`;
//   line 110 `__syncthreads();`                    -> glue `it.barrier();`
//   lines 111-113 `acc[k] = mmvf_warp_sum(partials[k][t])` -> glue with the object.
template <int BLOCK_SIZE, int NT>
void bf16_f32_mmvf_multi_body(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in,
                              int n_tok, sycl::nd_item<1> it, const sycl::local_accessor<float, 1>& parts,
                              const sycl::local_accessor<uint32_t, 1>& sfl, const sycl::local_accessor<uint32_t, 1>& spin) {
    const dim3 blockDim{BLOCK_SIZE, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / BLOCK_SIZE), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:79:81 @ 1390a6a
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    // SYCL-MIRROR-END
    // glue: CUDA line 82 `__shared__ float partials[NT][32];` is the flat handler-built `parts`
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:83:83 @ 1390a6a
    if constexpr (BLOCK_SIZE > 32) {
    // SYCL-MIRROR-END
        if (t < 32) {
            for (int k = 0; k < NT; ++k) parts[k * 32 + t] = 0.0f;  // glue: CUDA lines 84-86
        }
        it.barrier();                                               // glue: CUDA line 87
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:88:103 @ 1390a6a
    }
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
    // SYCL-MIRROR-END
    for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(shfl32, acc[k]);  // glue: CUDA lines 104-105
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:106:106 @ 1390a6a
    if constexpr (BLOCK_SIZE > 32) {
    // SYCL-MIRROR-END
        if ((t & 31) == 0) {
            for (int k = 0; k < NT; ++k) parts[k * 32 + t / 32] = acc[k];  // glue: CUDA lines 107-109
        }
        it.barrier();                                                     // glue: CUDA line 110
        if (t < 32) {
            for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(shfl32, parts[k * 32 + t]);  // glue: CUDA lines 111-113
        }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:114:118 @ 1390a6a
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:121:132 @ 1390a6a
int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}
// SYCL-MIRROR-END

// glue: the CUDA launch macros become q.submit launchers (the local accessors
// must be handler-built, T3); the mirrored `switch` lines below keep their
// CUDA spelling and expand through these macros.
template <int N>
sycl::event launch_single(sycl::queue& q, const float* x, const uint16_t* w, float* y, int n_in, int64_t n_out) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> parts(sycl::range<1>(32), h);
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * (size_t) N, (size_t) N),
                       [=](sycl::nd_item<1> it) { bf16_f32_mmvf_body<N>(x, w, y, n_in, it, parts, sfl, spin); });
    });
}

template <int N, int NT>
sycl::event launch_multi(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                         int n_in, int n_tok, int64_t n_out) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> parts(sycl::range<1>((size_t) NT * 32), h);
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * (size_t) N, (size_t) N),
                       [=](sycl::nd_item<1> it) {
                           bf16_f32_mmvf_multi_body<N, NT>(x, ldx, w, y, ldy, n_in, n_tok, it, parts, sfl, spin);
                       });
    });
}

}  // namespace

// glue (CUDA lines 136-154): the single-row fast path forwards; the CUDA
// `cudaGetLastError` check is dropped (SYCL throws).
// forward declaration: the SYCL overload of bf16_gemv_fp32_mmvf (defined below;
// the header declares only the CUDA-signature overload)
sycl::event bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                                sycl::queue& q);
sycl::event bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                      int64_t n_in, int64_t n_out, int n_tok, sycl::queue& q) {
    if (n_tok == 1 && ldy >= n_out) return bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, q);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:139:141 @ 1390a6a
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    // SYCL-MIRROR-END
#define STRATA_MMVF_M(N) case N: \
    if (n_tok <= 4) return launch_multi<N, 4>(q, x, ldx, w, y, ldy, n_in, n_tok, n_out); \
    else return launch_multi<N, 8>(q, x, ldx, w, y, ldy, n_in, n_tok, n_out)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:146:149 @ 1390a6a
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
    // SYCL-MIRROR-END
#undef STRATA_MMVF_M
    return {};
}

// glue (CUDA lines 156-183): the CUDA `cudaGetLastError` check is dropped
// (SYCL throws).
sycl::event bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                               sycl::queue& q) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:158:165 @ 1390a6a
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    // SYCL-MIRROR-END
#define STRATA_MMVF_CASE(N) case N: return launch_single<N>(q, x, w, y, n_in, n_out)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:169:178 @ 1390a6a
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
    // SYCL-MIRROR-END
#undef STRATA_MMVF_CASE
    return {};
}

}  // namespace strata::kernels
