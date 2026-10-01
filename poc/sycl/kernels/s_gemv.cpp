// poc/sycl/kernels/s_gemv.cpp -- T4: the flagship S-family GEMV port
// (plans/sycl-phase-a-t4.md).  CUDA source: src/kernels/cuda/s_gemv.cu
// @ f69bc66, the naive `s_gemv_kernel` (one thread per output row).
//
// One kernel covers all three code widths {2,4,8}, both codebooks, and the
// offset branch, because the CUDA kernel is templated only on CODE_BITS and
// takes the rest per-tensor at runtime - the S2/S4/S8/Q4_K parity cases all
// run through this one entry point.
//
// Contract differences from the CUDA wrapper, on purpose and for Phase B
// (same as T2/T3): the CUDA s_gemv() synchronises before returning and
// exit(1)s on a bad argument; submit_s_gemv() returns the event, leaves
// synchronisation to the caller, and throws std::runtime_error instead of
// exiting.  The Phase B engine wrapper absorbs both.
//
// The arithmetic lines are MIRRORED VERBATIM from the CUDA file (marker
// blocks, checked by check_mirrors.sh); the CUDA-specific load/store forms
// (__ldg, __half2float/__ushort_as_half, blockIdx/threadIdx) are glue, each
// commented with the line it replaces.
#include "strata/kernels/s_gemv.hpp"
#include <sycl/sycl.hpp>

#include <cstddef>
#include <stdexcept>

namespace strata::kernels {
namespace {

// CUDA's `__constant__` codebook (lines 32-33): the values are mirrored
// verbatim below; only the storage class changes (there is no SYCL constant
// memory here, and the kernel copies the 16 bytes into local memory per
// work-group exactly as the CUDA kernel does - see the note on kIq4Nl in the
// CUDA file for why a per-element constant read would have been the wrong
// access pattern).
static const signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
// SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:33:33 @ f69bc66
                                       1,   13,   25,  38,  53,  69,  89,  113};
// SYCL-MIRROR-END

// The CUDA decode_tbl (lines 43-47).  The CODE_BITS template parameter is
// unused in the body, so it drops; everything else is mirrored.
static inline float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:45:46 @ f69bc66
    if (codebook == (int) Codebook::Iq4Nl) return (float) tbl[code & 0x0F];
    return (float) (code + bias);
    // SYCL-MIRROR-END
}

template <int CODE_BITS>
sycl::event submit_impl(sycl::queue& q, const uint16_t* x, const uint8_t* codes,
                        const float* scales, const float* offset, float* y, long long n_in,
                        long long n_out, int bias, int codebook, int group_elems, int has_offset) {
    // T2 device fact: Level Zero rejects non-uniform work-groups, so the item
    // count is padded up to the work-group multiple; padded work-items fall
    // through the mirrored `o >= n_out` bounds check.
    const int threads = 128;
    const size_t items = (size_t) ((n_out + threads - 1) / threads) * (size_t) threads;
    return q.submit([&](sycl::handler& h) {
        // The CUDA __shared__ s_iq4nl[16] (line 310) is a handler-constructed
        // local accessor captured by value (the T3 pattern).
        sycl::local_accessor<signed char, 1> s_iq4nl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(items, (size_t) threads),
                       [=](sycl::nd_item<1> it) {
    // glue (CUDA lines 310-311): the table fill + barrier stay BEFORE the
    // early return, as in CUDA - a work-item that returned first would
    // strand its peers at a barrier.  Padded work-items all reach it.
    const int tid = (int) it.get_local_id(0);
    if (tid < 16) s_iq4nl[tid] = kIq4Nl[tid];
    it.barrier();
    const long long o = (long long) it.get_global_id(0);   // glue: CUDA line 312 (blockIdx.x * blockDim.x + threadIdx.x)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:313:313 @ f69bc66
    if (o >= n_out) return;
    // SYCL-MIRROR-END

    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:315:320 @ f69bc66
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;
    // SYCL-MIRROR-END

    // glue: CUDA line 336 loads the activation through
    // __half2float(__ushort_as_half(x[i])); sycl::half does the same
    // bit-exact widening (validated by t4/fp16_probe.cpp; icpx 2026.1 has no
    // sycl::fp16 - sycl::half IS the SYCL 2020 binary16 type).  __ldg's read-only
    // cache hint has no USM-pointer equivalent and drops.
    const sycl::half* xp = reinterpret_cast<const sycl::half*>(x);

    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:322:334 @ f69bc66
    float acc = 0.0f;
    for (long long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const long long base = g * (long long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            // The offset belongs to the WEIGHT, so it is applied to the decoded value BEFORE the activation
            // multiply: `w = code*scale + offset; acc += w * x`.  Writing `acc += code*scale*x + offset*x`
            // computes a mathematically equal expression that ROUNDS DIFFERENTLY, because it performs two
            // multiplications and an addition where this performs one of each.  The no-offset types cannot
            // tell the difference - which is why this was wrong until the Q4_K case was added.
    // SYCL-MIRROR-END
            const float w = decode_tbl(code, bias, codebook, &s_iq4nl[0]) * d + b;   // glue: CUDA line 335
            acc += w * (float) xp[i];                                                   // glue: CUDA line 336
        }
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:339:339 @ f69bc66
    y[o] = acc;
    // SYCL-MIRROR-END
                       });
    });
}

}  // namespace

// One work-group of 128 covers 128 output rows, mirroring the CUDA wrapper's
// `threads = 128` (line 356).  The host checks mirror the CUDA wrapper
// (346-355) with throws instead of exit(1).
sycl::event submit_s_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                         const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form) {
    // glue: CUDA line 346, adapted - the CUDA wrapper returns void, this one
    // returns an event, so the empty case submits an empty kernel (T2/T3
    // convention).
    if (n_in <= 0 || n_out <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:347:347 @ f69bc66
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
    // SYCL-MIRROR-END
        // glue (CUDA lines 348-350): same message, throw instead of exit(1)
        throw std::runtime_error("s_gemv: n_in " + std::to_string(n_in) + " is not a multiple of group_elems " +
                                 std::to_string(form.group_elems));
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:352:352 @ f69bc66
    if (form.has_offset && offset == nullptr) {
    // SYCL-MIRROR-END
        // glue (CUDA lines 353-354): same message, throw instead of exit(1)
        throw std::runtime_error("s_gemv: form says has_offset but offset is null");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/s_gemv.cu:356:358 @ f69bc66
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    const int cb = (int) form.codebook;
    // SYCL-MIRROR-END
    (void) blocks;   // kept for the mirror; the padded item count is derived in submit_impl
    switch (form.code_bits) {
        case 2:
            return submit_impl<2>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                  form.group_elems, form.has_offset ? 1 : 0);
        case 4:
            return submit_impl<4>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                  form.group_elems, form.has_offset ? 1 : 0);
        case 8:
            return submit_impl<8>(q, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                  form.group_elems, form.has_offset ? 1 : 0);
        default:
            // glue: CUDA lines 376-379, throw instead of exit(1)
            throw std::runtime_error("s_gemv: unsupported code_bits " + std::to_string(form.code_bits));
    }
}

}  // namespace strata::kernels
