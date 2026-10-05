// poc/sycl/kernels/s2_gemv.cpp -- B2 step 3.1 (plans/sycl-phase-b-steps-b2-steps.md):
// the SYCL port of src/kernels/cuda/s2_gemv.cu @ c1ff5cf (the P2.S2 GEMV, one
// thread per output row).
//
// Naive per the phase rule, exactly as the CUDA file: dequantize on the fly,
// FP32 accumulation inside the row, no shared memory, no vector loads, no
// cache hints.  The parity test (poc/sycl/drivers/s2_gemv_parity.cpp) is what
// says whether a change is still right.
//
// The arithmetic lines are MIRRORED VERBATIM from the CUDA file (marker
// blocks, checked by check_mirrors.sh); the CUDA-specific forms (blockIdx /
// threadIdx, __half2float/__ushort_as_half, <<< >>>, cudaDeviceSynchronize)
// are glue, each commented with the line it replaces.
//
// Contract differences from the CUDA wrapper, on purpose and for Phase B
// (same as T4/B1): the CUDA s2_gemv() synchronises before returning and
// exit(1)s on a bad argument; submit_s2_gemv() returns the event, leaves
// synchronisation to the caller, and throws std::runtime_error instead of
// exiting.  The Phase B engine wrapper absorbs both.
#include "strata/kernels/s2_gemv.hpp"
#include <sycl/sycl.hpp>

#include <cstddef>
#include <stdexcept>

namespace strata::kernels {
namespace {

// SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:17:17 @ c1ff5cf
constexpr int QK = 64;
// SYCL-MIRROR-END

sycl::event submit_impl(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                        float* y, long long n_in, long long n_out) {
    // T2 device fact: Level Zero rejects non-uniform work-groups, so the
    // item count is padded up to the work-group multiple; padded work-items
    // fall through the mirrored `o >= n_out` bounds check (CUDA line 23).
    const int threads = 128;
    const size_t items = (size_t) ((n_out + threads - 1) / threads) * (size_t) threads;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(items, (size_t) threads),
                       [=](sycl::nd_item<1> it) {
    // glue (CUDA line 22): blockIdx.x * blockDim.x + threadIdx.x
    const long long o = (long long) it.get_global_id(0);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:23:23 @ c1ff5cf
    if (o >= n_out) return;
    // SYCL-MIRROR-END

    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:25:27 @ c1ff5cf
    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;
    // SYCL-MIRROR-END

    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:29:37 @ c1ff5cf
    float acc = 0.0f;
    for (long long b = 0; b < nb; ++b) {
        const float d = s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;
        for (int j = 0; j < QK; ++j) {
            // (code - 1) in the INTEGER domain, then the group scale, then the activation; the CPU reference
            // below is written in the same order so the two differ only by floating-point contraction.
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
    // SYCL-MIRROR-END
            // glue (CUDA line 38): __half2float(__ushort_as_half(xb[j])) -- sycl::half IS the
            // binary16 type (T4 t4_fp16_probe), so the widening reads the same two bytes.
            acc += (float) (code - 1) * d * (float) *reinterpret_cast<const sycl::half*>(xb + j);
        }
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:41:41 @ c1ff5cf
    y[o] = acc;
    // SYCL-MIRROR-END
                       });
    });
}

}  // namespace

// One work-group of 128 covers 128 output rows, mirroring the CUDA wrapper's
// `threads = 128` (line 53).  The host checks mirror the CUDA wrapper
// (48-52) with throws instead of exit(1); the event return replaces the
// wrapper's cudaDeviceSynchronize (lines 56-60) -- a failed submit throws
// from q.submit.
sycl::event submit_s2_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                          float* y, int64_t n_in, int64_t n_out) {
    // glue (CUDA line 48): the CUDA wrapper returns void; the empty case
    // submits an empty kernel (T4/B1 convention).
    if (n_in <= 0 || n_out <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:49:49 @ c1ff5cf
    if (n_in % QK != 0) {
    // SYCL-MIRROR-END
        // glue (CUDA lines 50-51): same message, throw instead of exit(1)
        throw std::runtime_error("s2_gemv: n_in " + std::to_string(n_in) + " is not a multiple of " +
                                 std::to_string(QK));
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s2_gemv.cu:53:54 @ c1ff5cf
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    // SYCL-MIRROR-END
    (void) blocks;   // kept for the mirror; the padded item count is what the launch uses
    return submit_impl(q, x, codes, scales, y, n_in, n_out);
}

}  // namespace strata::kernels
