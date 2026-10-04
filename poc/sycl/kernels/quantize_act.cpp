// poc/sycl/kernels/quantize_act.cu's SYCL port -- B1
// (plans/sycl-phase-b-steps-b1.md, step 2.3): the Q8_0 / Q8_K activation
// quantizers.  CUDA source: src/kernels/cuda/quantize_act.cu @ f2a08d4.
//
// The five kernel bodies are MIRRORED VERBATIM (marker blocks, checked by
// check_mirrors.sh).  Contract differences, on purpose (T4/T5 pattern):
//  - each CUDA wrapper's exit(1) validation becomes a thrown
//    std::runtime_error, and each returns an sycl::event instead of
//    synchronising; the cudaGetLastError() blocks are dropped (SYCL throws).
//  - `nearest_int_dev`'s CUDA `__device__ __forceinline__` spelling is glue;
//    its body is mirrored.  It is a plain inline the device lambda inlines.
//  - `__fmul_rn` is the B0 intrinsic-layer macro; `fmul_rn` there uses a
//    volatile store so icpx cannot contract the multiply into the following
//    `+ 12582912.0f` (the CUDA comment documents exactly that failure mode).
//  - unqualified `min` (a CUDA device builtin) resolves via a `using std::min`
//    glue in this namespace.
//
// Knife-edge per the plan: `quantize_q8_0_kernel` divides in DOUBLE on device
// (line 72).  P2b measured device double math on Arc as IEEE-exact vs host
// (6+ orders of margin); the driver compares BLOCK BYTES, so any residual is
// red, not tolerated.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
// glue: CUDA's device-code `min`/`max` builtins; resolve to std::min/std::max
// for host-device code (the mirrored line 200 uses unqualified `min`).
using std::min;
using std::max;

namespace {

// SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:36:36 @ f2a08d4
constexpr int QK8_0 = 32;
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:145:146 @ f2a08d4
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;
// SYCL-MIRROR-END

// G1 glue: CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// The __global__ signatures (CUDA lines 47-48, 100-101, 129-130, 161-162,
// 210-211) are glue - the submit_ wrappers below pass the same arguments;
// the bodies are mirrored.
void quantize_q8_0_kernel_body(const float* x, uint8_t* blocks, long long n_blocks,
                               sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:49:77 @ f2a08d4
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;                 // { fp16 d ; int8 qs[32] }

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    if (amax == 0.0f) {
        // ggml leaves the block zeroed: d = 0 and every q = 0.  Writing the fp16 zero explicitly rather than
        // skipping keeps the block layout deterministic for the byte comparison.
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);

    // double division, matching the reference exactly (subtlety 2)
    for (int i = 0; i < QK8_0; ++i) {
        double q = rint((double) xb[i] / (double) d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t) (int8_t) q;
    }
}
    // SYCL-MIRROR-END

void quantize_q8_0_scaled_kernel_body(const float* x, uint8_t* blocks, float* scales, long long n_blocks,
                                      sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:102:127 @ f2a08d4
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    // VERBATIM from `cpu/expert.cpp:144-145`, including the `amax > 0` guard, so the fp32 value written here
    // is bit-identical to the `s` the CPU path used.
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    scales[b] = s;

    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < QK8_0; ++i) {
        // VERBATIM from `cpu/expert.cpp:159-162`: reciprocal multiply, then `t + copysign(0.5, t)` truncated
        // toward zero, which is `lround`'s rule - round half away from zero.
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t) (int8_t) v;
    }
}
    // SYCL-MIRROR-END

void dequant_q8_0_kernel_body(const uint8_t* blocks, float* x, long long n_blocks,
                              sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:131:138 @ f2a08d4
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}
    // SYCL-MIRROR-END

// glue (CUDA line 154): the __device__ __forceinline__ spelling
inline int nearest_int_dev(float fval) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:155:159 @ f2a08d4
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}
    // SYCL-MIRROR-END

void quantize_q8_K_kernel_body(const float* x, uint8_t* blocks, long long n_blocks,
                               sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:163:208 @ f2a08d4
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK_K;
    uint8_t* out = blocks + b * Q8K_BYTES;
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bsums = (int16_t*) (out + 4 + QK_K);

    // `max` is the SIGNED value at the largest magnitude position, and the comparison is STRICTLY greater,
    // so a tie keeps the FIRST maximum - which is what `np.argmax` does in the reference transcription too.
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        // ggml writes d = 0 and zeroes qs, then `continue`s - which LEAVES bsums UNWRITTEN.  A zeroed block
        // and an untouched one are indistinguishable to the dot product, but not to a byte comparison, so
        // this zeroes bsums as well and the parity test's reference does the same.  Recorded because it is a
        // deliberate divergence from the letter of the source.
        *d = 0.0f;
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;          // -127, NOT -128; see the header
    for (int j = 0; j < QK_K; ++j) {
        // `__fmul_rn`, NOT `iscale * xb[j]`.  ggml computes the product, ROUNDS IT TO F32, and then adds
        // 12582912.0f inside `nearest_int`.  Written as a plain expression, nvcc CONTRACTS the multiply into
        // the add as an FMA - which is a more accurate product but not the same one, and it flips the result
        // wherever the true product sits just off a .5 boundary.  The first version of this kernel differed
        // from the reference in 1 element of 524,288 for exactly this reason, and `__fmul_rn` pins the
        // rounding step the source actually performs.
        const int v = nearest_int_dev(__fmul_rn(iscale, xb[j]));
        qs[j] = (int8_t) min(127, v);            // MIN only - the source has no lower clamp
    }
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = 1.0f / iscale;
}
    // SYCL-MIRROR-END

void dequant_q8_K_kernel_body(const uint8_t* blocks, float* x, long long n_blocks,
                              sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:212:220 @ f2a08d4
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*) (blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
}
    // SYCL-MIRROR-END

}  // namespace

sycl::event submit_quantize_q8_0(sycl::queue& q, const float* x, uint8_t* blocks, int64_t n) {
    // glue (CUDA line 225): the empty case submits nothing (T4 convention)
    if (n <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:226:227 @ f2a08d4
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
    // SYCL-MIRROR-END
        // glue (CUDA line 228): throw instead of exit(1); line 229's close-brace follows
        throw std::runtime_error("quantize_q8_0: n is not a multiple of 32");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:230:232 @ f2a08d4
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    // SYCL-MIRROR-END
    // glue (CUDA lines 233-238): the launch; the cudaGetLastError block is dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        quantize_q8_0_kernel_body(x, blocks, nb, it, threads);
    });
}

sycl::event submit_quantize_q8_0_scaled(sycl::queue& q, const float* x, uint8_t* blocks, float* scales, int64_t n) {
    // glue (CUDA line 246): the empty case submits nothing (T4 convention)
    if (n <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:247:248 @ f2a08d4
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
    // SYCL-MIRROR-END
        // glue (CUDA line 249): throw instead of exit(1); line 250's close-brace follows
        throw std::runtime_error("quantize_q8_0_scaled: n is not a multiple of 32");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:251:252 @ f2a08d4
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
    // SYCL-MIRROR-END
        // glue (CUDA line 253): throw instead of exit(1); line 254's close-brace follows
        throw std::runtime_error("quantize_q8_0_scaled: scales is null");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:255:257 @ f2a08d4
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    // SYCL-MIRROR-END
    // glue (CUDA lines 258-263): the launch; the cudaGetLastError block is dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        quantize_q8_0_scaled_kernel_body(x, blocks, scales, nb, it, threads);
    });
}

sycl::event submit_dequant_q8_0(sycl::queue& q, const uint8_t* blocks, float* x, int64_t n) {
    // glue (CUDA line 268): the empty case submits nothing (T4 convention)
    if (n <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:269:271 @ f2a08d4
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    // SYCL-MIRROR-END
    // glue (CUDA lines 272-277): the launch; the cudaGetLastError block is dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        dequant_q8_0_kernel_body(blocks, x, nb, it, threads);
    });
}

sycl::event submit_quantize_q8_K(sycl::queue& q, const float* x, uint8_t* blocks, int64_t n) {
    // glue (CUDA line 282): the empty case submits nothing (T4 convention)
    if (n <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:283:284 @ f2a08d4
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
    // SYCL-MIRROR-END
        // glue (CUDA line 285): throw instead of exit(1); line 286's close-brace follows
        throw std::runtime_error("quantize_q8_K: n is not a multiple of 256");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:287:289 @ f2a08d4
    const long long nb = n / QK_K;
    const int threads = 64;                       // one block per thread, and a block is 256 elements
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    // SYCL-MIRROR-END
    // glue (CUDA lines 290-295): the launch; the cudaGetLastError block is dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        quantize_q8_K_kernel_body(x, blocks, nb, it, threads);
    });
}

sycl::event submit_dequant_q8_K(sycl::queue& q, const uint8_t* blocks, float* x, int64_t n) {
    // glue (CUDA line 300): the empty case submits nothing (T4 convention)
    if (n <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/quantize_act.cu:301:303 @ f2a08d4
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    // SYCL-MIRROR-END
    // glue (CUDA lines 304-309): the launch; the cudaGetLastError block is dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        dequant_q8_K_kernel_body(blocks, x, nb, it, threads);
    });
}

}  // namespace strata::kernels
