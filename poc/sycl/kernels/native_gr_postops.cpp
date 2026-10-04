// poc/sycl/kernels/native_gr_postops.cu's SYCL port -- B1
// (plans/sycl-phase-b-steps-b1.md, step 2.4): the pinned llama.cpp
// single-token GR post-operations (down SiLU, pre-gated, post residual).
// CUDA source: src/kernels/cuda/native_gr_postops.cu @ 8616e33.
//
// The three kernel bodies and the host helpers are MIRRORED VERBATIM
// (marker blocks, checked by check_mirrors.sh).  Contract differences, on
// purpose (T4/T5 pattern):
//  - each CUDA wrapper's cudaStream_t becomes an sycl::queue; the wrappers
//    return an sycl::event instead of checking cudaGetLastError() (SYCL
//    throws; check_launch's body is dropped, the driver calls
//    wait_and_throw()).
//  - `__fmaf_rn`/`__fmul_rn`/`__fadd_rn` are the B0 intrinsic-layer macros
//    (fmaf/fmul-with-forced-rounding; the pinned-rounding contract is kept).
//  - `expf` is the plain C libm device function on both sides (the CUDA
//    source uses expf, NOT __expf).  b0_misc_probe measured the device
//    expf-vs-host gap at up to 64 f32 ulp on Arc with icpx 2026.1; the
//    parity driver's gate absorbs it (written margin, not a tolerance:
//    logic errors are orders of magnitude larger).
#include "strata/kernels/native_gr_postops.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {

// glue (G4): TU-local empty qualifier macros so the one-line
// `__device__ __forceinline__` helpers below mirror verbatim (the CUDA file
// has no standalone qualifier lines for them).
#define __device__
#define __forceinline__

namespace {

// SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:35:41 @ 8616e33
constexpr int THREADS = 256;
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
__device__ __forceinline__ float scale_zero_bias(float x, float scale) {
    return __fmaf_rn(scale, x, 0.0f);
}
// SYCL-MIRROR-END

// G1 glue: CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// The `down_silu` signature (CUDA line 42) is glue - the submit wrapper
// passes the same arguments; the body (43-46) is mirrored.
void down_silu_body(float* lo, int count, float scale, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:43:46 @ 8616e33
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count)) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + expf(-x));
    // SYCL-MIRROR-END
}

// The `pre_gated` signature (CUDA lines 48-50) is glue; the template
// parameter survives in the body (the Fused branch is part of the mirrored
// text, lines 51-63).
template<bool Fused>
void pre_gated_body(const float* xn, float* gate, float* mixed, int n_embd, int hc, float scale,
                    sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:51:63 @ 8616e33
    const std::size_t d = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= std::size_t(n_embd)) return;
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const std::size_t i = std::size_t(c) * n_embd + d;
        const float x = xn[i], w = sigmoid(gate[i]);
        const float product = __fmul_rn(x, w);
        gate[i] = product;
        if constexpr (Fused) sum = __fmaf_rn(x, w, sum);
        else sum = c == 0 ? product : __fadd_rn(sum, product);
    }
    if constexpr (Fused) mixed[d] = scale * sum;
    else mixed[d] = scale_zero_bias(sum, scale);
    // SYCL-MIRROR-END
}

// The `post` signature (CUDA lines 65-67) is glue; the body (68-73) is
// mirrored.  The exact residual/output alias stays supported: each thread
// reads its own residual[i] and then writes its own output[i].
void post_body(const float* residual, const float* block_out, const float* inject, float* output,
               int n_embd, int hc, float scale, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:68:73 @ 8616e33
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_embd) * hc) return;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = __fmaf_rn(block_out[d], weight, residual[i]);
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:75:83 @ 8616e33
void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }
// SYCL-MIRROR-END
// (CUDA lines 84-88, check_launch: cudaGetLastError check - dropped, SYCL
// throws; the driver calls wait_and_throw.)
}  // namespace

sycl::event submit_native_gr_down_silu(sycl::queue& q, float* lo, int hc_lr, int hc) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:92:93 @ 8616e33
    check_shape(hc_lr, hc);
    check_pointer(lo);
    // SYCL-MIRROR-END
    // glue (CUDA line 94): the launch; the host `1.0f / float(hc)` is
    // IEEE-exact; check_launch dropped
    return strata::sycl_compat::strata_launch(q, (size_t) blocks(hc_lr) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) {
        down_silu_body(lo, hc_lr, 1.0f / float(hc), it, THREADS);
    });
}

sycl::event submit_native_gr_pre_gated(sycl::queue& q, const float* xn, float* gate, float* mixed,
                                       int n_embd, int hc, bool fused_layer) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:99:100 @ 8616e33
    check_shape(n_embd, hc);
    check_pointer(xn); check_pointer(gate); check_pointer(mixed);
    // SYCL-MIRROR-END
    // glue (CUDA lines 101-104): the fused/non-fused launches; check_launch dropped
    if (fused_layer)
        return strata::sycl_compat::strata_launch(q, (size_t) blocks(n_embd) * THREADS, THREADS,
                                                  [=](sycl::nd_item<1> it) {
            pre_gated_body<true>(xn, gate, mixed, n_embd, hc, 1.0f / float(hc), it, THREADS);
        });
    return strata::sycl_compat::strata_launch(q, (size_t) blocks(n_embd) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) {
        pre_gated_body<false>(xn, gate, mixed, n_embd, hc, 1.0f / float(hc), it, THREADS);
    });
}

sycl::event submit_native_gr_post(sycl::queue& q, const float* residual, const float* block_out,
                                  const float* inject, float* output, int n_embd, int hc) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_gr_postops.cu:109:110 @ 8616e33
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    // SYCL-MIRROR-END
    // glue (CUDA lines 111-112): the launch; check_launch dropped
    return strata::sycl_compat::strata_launch(q, (size_t) blocks(std::size_t(n_embd) * hc) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) {
        post_body(residual, block_out, inject, output, n_embd, hc, 1.0f / float(hc), it, THREADS);
    });
}

}  // namespace strata::kernels
