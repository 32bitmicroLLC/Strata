// poc/sycl/kernels/elementwise.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.6)
// SYCL port of src/kernels/cuda/elementwise.cu @ 22d022e: the 14 glue kernels of
// the layer graph (embedding gather, gdn gate, scale/add, f16/bf16 bridges,
// silu, rms_norm_weighted, the three doorbell kernels, the three mapped-copy
// kernels) plus their host dispatch.
//
// The mirrored blocks below are VERBATIM (checked by check_mirrors.sh).
// Glue:
//  - G4: `__device__`/`__forceinline__`/`__global__` are empty macros; `dim3`
//    and `gridDim` are glue structs resolved per work-item.
//  - `__shfl_down_sync`/`__shfl_sync` are the B0 intrinsics.hpp macros; the
//    rms body builds its `shfl32` warp-emulation object in glue (a local-
//    memory 32-wide warp, B0 §1.3).
//  - `__syncthreads` becomes a glue `it.barrier()` line (not mirrored;
//    router_top10 precedent).  `__shared__ int hit` becomes a handler-built
//    local_accessor; its two references are glue lines.
//  - `rsqrtf` is NOT available to DPC++ device code (host-only libm macro;
//    probed: "SYCL kernel cannot call an undefined function").  Per the plan
//    it is a glue macro `1.0f / sqrtf(x)`; the driver's rms gate (rel 1e-6,
//    the CUDA driver's own) is far above the sqrt/div ulp gap and the
//    measured result is recorded in the step report.
//  - `strata_spin_pause` (included from dp4a.hpp in CUDA) is an empty spin
//    here: the device has no `__nanosleep`, and dp4a.hpp is deliberately NOT
//    included (its STRATA_DP4A macro would clash with intrinsics.hpp --
//    the plan's F6/B0 note).  The pause is a backoff hint, not a contract.
//  - `__threadfence_system` is the intrinsics.hpp `threadfence()` (G7: the
//    doorbell protocol itself is Phase C; ordering under USM shared pointers
//    is coherent in practice and recorded, not gated).
//  - Host dispatch: `<<<>>>` becomes `strata::sycl_compat::strata_launch` or a
//    `q.submit` (the two shared-memory kernels); `check_launch`/
//    `sync_if_needed` are dropped (SYCL throws; the driver waits); the
//    validation `exit(1)` lines become thrown errors except where the CUDA
//    source keeps `fprintf` and `exit(1)` on one line (doorbell_publish, kept
//    verbatim as a loud host-side failure); `void* stream` becomes
//    `sycl::queue&`.  The CUDA one-line `if (n <= 0) return;` guards become
//    glue `if (n <= 0) return {};` (they do not compile in a function that
//    returns an event).
//  - `f16_from_f32`/`bf16_from_f32` come from the shared host-device headers
//    (f16_bits.hpp / bf16_bits.hpp, included directly, as in CUDA).
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

// glue (G4): empty device qualifiers so the mirrored lines keep their CUDA spelling.
#define __device__
#define __forceinline__
#define __global__

// glue (G1): CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// glue: CUDA's `rsqrtf` has no DPC++ device form (host-only libm macro --
// probed); `1.0f / sqrtf(x)` is the portable equivalent (the plan's fallback,
// measured in the driver's rms gate).
#define rsqrtf(x) (1.0f / sqrtf(x))

// glue: CUDA's `strata_spin_pause` is `__nanosleep(100)` on sm_70+; the device
// has no equivalent, so the pause is empty.  It is a backoff hint, not a
// contract (dp4a.hpp's own comment says so).
inline void strata_spin_pause() {}

// glue: CUDA's `float4` builtin becomes sycl::float4 for the mirrored lines, and
// CUDA's `make_float4` constructor becomes sycl::make_float4.
using sycl::float4;
inline sycl::float4 make_float4(float x, float y, float z, float w) { return sycl::float4(x, y, z, w); }

// SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:17:17 @ 22d022e
constexpr int THREADS = 256;
// SYCL-MIRROR-END

// The `embedding_gather_kernel` `__global__` signature (CUDA lines 19-23) is glue;
// the body (24-31) is mirrored verbatim.
void embedding_gather_body(const uint8_t* codes, const float* scales, const float* offsets,
                           int64_t n, int code_bits, int code_bias, int group_elems, float* out,
                           sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:24:31 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), scales[group]);
    out[i] = __fadd_rn(product, offsets ? offsets[group] : 0.0f);
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:38:38 @ 22d022e
__device__ __forceinline__ float softplus_dev(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
// SYCL-MIRROR-END

// The `gdn_gate_kernel` `__global__` signature (CUDA lines 40-41) is glue; the body
// (42-47) is mirrored verbatim.
void gdn_gate_body(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t h_v,
                   sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:42:47 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
    // SYCL-MIRROR-END
}

// The `scale_kernel` `__global__` signature (CUDA line 50) is glue; the body (51-52)
// is mirrored verbatim.
void scale_body(float* x, int64_t n, float s, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:51:52 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
    // SYCL-MIRROR-END
}

// The `add_kernel` `__global__` signature (CUDA line 57) is glue; the body (58-59)
// is mirrored verbatim.
void add_body(float* dst, const float* src, long long n, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:58:59 @ 22d022e
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
    // SYCL-MIRROR-END
}

// The `to_f16_kernel` `__global__` signature (CUDA line 62) is glue; the body (63-64)
// is mirrored verbatim.
void to_f16_body(const float* x, uint16_t* y, int64_t n, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:63:64 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_from_f32(x[i]);
    // SYCL-MIRROR-END
}

// The `to_bf16_kernel` `__global__` signature (CUDA line 67) is glue; the body (68-69)
// is mirrored verbatim.
void to_bf16_body(const float* x, uint16_t* y, int64_t n, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:68:69 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_from_f32(x[i]);
    // SYCL-MIRROR-END
}

// The `silu_kernel` `__global__` signature (CUDA line 72) is glue; the body (73-79)
// is mirrored verbatim.  The device DOUBLE `exp` compiles and runs on Arc
// (probed); the driver's rel gate (1e-7, the CUDA driver's own) is held verbatim.
void silu_body(float* x, int64_t n, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:73:79 @ 22d022e
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float) (v / (1.0 + exp(-v)));
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:83:83 @ 22d022e
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }
// SYCL-MIRROR-END

// The `rms_norm_weighted_kernel` `__global__` signature (CUDA lines 88-89) is glue.
// The body (91-110) is mirrored verbatim; the two shuffle lines expand through the
// B0 intrinsics.hpp macros onto the `shfl32` object built in glue here (local-
// memory 32-wide warp emulation, B0 §1.3), and `rsqrtf` is the glue macro above.
void rms_body(float* x, const float* w, int64_t rows, int64_t cols, float eps, sycl::nd_item<1> it,
              const sycl::local_accessor<uint32_t, 1>& sfl, const sycl::local_accessor<uint32_t, 1>& spin) {
    const dim3 blockDim{128, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / 128), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:91:110 @ 22d022e
    const int lane = threadIdx.x & 31;
    const int64_t row = ((int64_t) blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = rsqrtf(acc / (float) cols + eps);
    inv = __shfl_sync(0xFFFFFFFFu, inv, 0);
    for (int64_t c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
    // SYCL-MIRROR-END
}

// The `doorbell_ring_kernel` `__global__` signature (CUDA line 209) is glue; the body
// (210-211) is mirrored verbatim (`__threadfence_system` -> intrinsics.hpp).
void doorbell_ring_body(uint32_t* seq) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:210:211 @ 22d022e
    __threadfence_system();
    *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
    // SYCL-MIRROR-END
}

// The `doorbell_wait_kernel` `__global__` signature (CUDA line 214) is glue; the body
// (215-217) is mirrored verbatim (`strata_spin_pause` is the empty-spin glue above).
void doorbell_wait_body(const volatile uint32_t* flag, const volatile uint32_t* seq) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:215:217 @ 22d022e
    const uint32_t want = *seq;
    while (*flag != want) strata_spin_pause();
    __threadfence_system();
    // SYCL-MIRROR-END
}

// The `copy_from_mapped_kernel` `__global__` signature (CUDA line 226) is glue; the
// body (227-230) is mirrored verbatim.  `gridDim` is glue (the exact block count,
// as in CUDA's `<<<blocks, 256>>>`).
void copy_from_mapped_body(sycl::float4* dst, const volatile sycl::float4* src, int64_t n4, int blocks,
                           sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    const dim3 gridDim{blocks, 1, 1};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:227:230 @ 22d022e
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = const_cast<const float4*>(src)[i];
        dst[i] = v;
    }
    // SYCL-MIRROR-END
}

// The `copy_rows_from_mapped_kernel` `__global__` signature (CUDA lines 235-236) is glue.
// The body is mirrored except three lines that need shared-memory plumbing:
//   line 238 `__shared__ int hit;`   -> the handler-built `s_hit` local_accessor;
//   line 243 `hit = h;`              -> glue `s_hit[0] = h;`
//   line 245 `__syncthreads();`      -> glue `it.barrier();`
//   line 247 `if (hit)`              -> glue `if (s_hit[0])`
// (the same treatment as router_top10.cpp's shared lines).
void copy_rows_body(sycl::float4* dst, const volatile sycl::float4* src, int64_t row4, const int32_t* hit_rows,
                    const int32_t* count, sycl::nd_item<1> it, const sycl::local_accessor<int, 1>& s_hit) {
    const dim3 blockDim{128, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / 128), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:237:237 @ 22d022e
    const int row = blockIdx.x;
    // SYCL-MIRROR-END
    // (glue: CUDA line 238 `__shared__ int hit;` is the handler-built `s_hit`)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:239:242 @ 22d022e
    if (threadIdx.x == 0) {
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
    // SYCL-MIRROR-END
        s_hit[0] = h;                                  // glue: CUDA line 243 (hit = h;)
    }
    it.barrier();                                      // glue: CUDA line 245 (__syncthreads)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:246:246 @ 22d022e
    float4* d = dst + (int64_t) row * row4;
    // SYCL-MIRROR-END
    if (s_hit[0]) {                                    // glue: CUDA line 247 (if (hit))
        // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:248:248 @ 22d022e
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // SYCL-MIRROR-END
    } else {
        // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:250:251 @ 22d022e
        const volatile float4* sr = src + (int64_t) row * row4;
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(sr)[i];
        // SYCL-MIRROR-END
    }
}

// The `doorbell_publish_kernel` `__global__` signature (CUDA lines 276-278) is glue; the
// body is mirrored except line 282 `__syncthreads();`, which is a glue `it.barrier()`.
// The 1024-wide launch is exact (Arc's max work-group size is 1024 -- launch.hpp/T0).
void doorbell_publish_body(const float* x, const int32_t* ids, const float* w, int n, int k, float* x_out,
                           int32_t* ids_out, float* w_out, uint32_t* seq, sycl::nd_item<1> it) {
    const dim3 blockDim{1024, 1, 1};
    const dim3 blockIdx{0, 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:279:281 @ 22d022e
    for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
    if ((int) threadIdx.x < k) { ids_out[threadIdx.x] = ids[threadIdx.x]; w_out[threadIdx.x] = w[threadIdx.x]; }
    __threadfence_system();
    // SYCL-MIRROR-END
    it.barrier();                                      // glue: CUDA line 282 (__syncthreads)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:283:286 @ 22d022e
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
    }
    // SYCL-MIRROR-END
}

// The `copy_i32_from_mapped_kernel` `__global__` signature (CUDA line 297) is glue; the
// body (298) is mirrored verbatim.
void copy_i32_body(int32_t* dst, const volatile int32_t* src, int n, sycl::nd_item<1> it, int threads) {
    const dim3 blockDim{threads, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:298:298 @ 22d022e
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    // SYCL-MIRROR-END
}

}  // namespace

// ---------------------------------------------------------------------------
// Host dispatch: the CUDA wrappers' validation lines are mirrored where they
// compile; `<<<>>>` becomes strata_launch/q.submit; check_launch and
// sync_if_needed are dropped (SYCL throws; the driver waits on the event);
// `exit(1)` validation becomes a thrown error (doorbell_publish's one-line
// fprintf+exit is kept verbatim, as a loud host-side failure).
// ---------------------------------------------------------------------------

sycl::event submit_embedding_gather(sycl::queue& q, const uint8_t* codes, const float* scales, const float* offsets,
                                    int64_t n, int code_bits, int code_bias, int group_elems, float* out) {
    if (n <= 0) return {};                             // glue: CUDA line 136 (return; -> return {};)
    // glue (CUDA lines 137-139): the launch; check_launch dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) {
        embedding_gather_body(codes, scales, offsets, n, code_bits, code_bias, group_elems, out, it, THREADS);
    });
}

sycl::event submit_gdn_gate(sycl::queue& q, const float* alpha, const float* dt, const float* ssm_a, float* gate,
                            int64_t n_tokens, int64_t h_v) {
    if (n_tokens <= 0 || h_v <= 0) return {};          // glue: CUDA line 144
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:145:145 @ 22d022e
    const int64_t n = n_tokens * h_v;
    // SYCL-MIRROR-END
    // glue (CUDA lines 146-148): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { gdn_gate_body(alpha, dt, ssm_a, gate, h_v, it, THREADS); });
}

sycl::event submit_scale_inplace(sycl::queue& q, float* x, int64_t n, float s) {
    if (n <= 0) return {};                             // glue: CUDA line 152
    // glue (CUDA lines 153-155): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { scale_body(x, n, s, it, THREADS); });
}

sycl::event submit_add_inplace(sycl::queue& q, float* dst, const float* src, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 158
    // glue (CUDA lines 159-162): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { add_body(dst, src, (long long) n, it, THREADS); });
}

sycl::event submit_f32_to_f16_bulk(sycl::queue& q, const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 166
    // glue (CUDA lines 167-169): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { to_f16_body(x, y, n, it, THREADS); });
}

sycl::event submit_f32_to_bf16_bulk(sycl::queue& q, const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 173
    // glue (CUDA lines 174-176): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { to_bf16_body(x, y, n, it, THREADS); });
}

sycl::event submit_silu_inplace(sycl::queue& q, float* x, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 180
    // glue (CUDA lines 181-183): the launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, (size_t) grid_for(n) * THREADS, THREADS,
                                              [=](sycl::nd_item<1> it) { silu_body(x, n, it, THREADS); });
}

sycl::event submit_doorbell_ring(sycl::queue& q, uint32_t* d_seq) {
    if (d_seq == nullptr) return {};                   // glue: CUDA line 308
    // glue (CUDA lines 309-311): the 1x1 launch; check_launch/sync_if_needed dropped
    return strata::sycl_compat::strata_launch(q, 1, 1, [=](sycl::nd_item<1> it) {
        (void) it;
        doorbell_ring_body(d_seq);
    });
}

sycl::event submit_doorbell_wait(sycl::queue& q, const uint32_t* d_flag, const uint32_t* d_seq) {
    if (d_flag == nullptr || d_seq == nullptr) return {};  // glue: CUDA line 221 (`return;` becomes `return {};`)    // glue (CUDA lines 222-223): the 1x1 launch; check_launch dropped
    return strata::sycl_compat::strata_launch(q, 1, 1, [=](sycl::nd_item<1> it) {
        (void) it;
        doorbell_wait_body((const volatile uint32_t*) d_flag, (const volatile uint32_t*) d_seq);
    });
}

sycl::event submit_copy_from_mapped(sycl::queue& q, float* dst, const float* src, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 265
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:266:267 @ 22d022e
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
    // SYCL-MIRROR-END
        // glue: the CUDA `std::exit(1)` becomes a thrown error.
        throw std::runtime_error("copy_from_mapped: misaligned or unaligned-4 n");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:270:271 @ 22d022e
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    // SYCL-MIRROR-END
    // glue (CUDA line 272): the launch; check_launch dropped
    return strata::sycl_compat::strata_launch(q, (size_t) blocks * 256, 256, [=](sycl::nd_item<1> it) {
        copy_from_mapped_body((sycl::float4*) dst, (const volatile sycl::float4*) src, n4, blocks, it, 256);
    });
}

sycl::event submit_copy_rows_from_mapped(sycl::queue& q, float* dst, const float* src, int64_t rows, int64_t width,
                                         const int32_t* hit_rows, const int32_t* count) {
    if (rows <= 0) return {};                          // glue: CUDA line 256
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:257:258 @ 22d022e
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
    // SYCL-MIRROR-END
        // glue: the CUDA `std::exit(1)` becomes a thrown error.
        throw std::runtime_error("copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned");
    }
    // glue (CUDA lines 261-262): the rows x 128 launch with the shared-memory accessor; check_launch dropped
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> s_hit(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * 128, 128),
                       [=](sycl::nd_item<1> it) {
            copy_rows_body((sycl::float4*) dst, (const volatile sycl::float4*) src, width / 4, hit_rows, count, it,
                           s_hit);
        });
    });
}

sycl::event submit_doorbell_publish(sycl::queue& q, const float* x, const int32_t* ids, const float* weights, int64_t n,
                                    int64_t k, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:291:291 @ 22d022e
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    // SYCL-MIRROR-END
    // glue (CUDA lines 292-294): the 1x1024 launch (Arc's max work-group size is
    // 1024); check_launch dropped
    return strata::sycl_compat::strata_launch(q, 1024, 1024, [=](sycl::nd_item<1> it) {
        doorbell_publish_body(x, ids, weights, (int) n, (int) k, x_out, ids_out, weights_out, d_seq, it);
    });
}

sycl::event submit_copy_i32_from_mapped(sycl::queue& q, int32_t* dst, const int32_t* src, int64_t n) {
    if (n <= 0) return {};                             // glue: CUDA line 302
    // glue (CUDA lines 303-304): the 1x128 launch; check_launch dropped
    return strata::sycl_compat::strata_launch(q, 128, 128, [=](sycl::nd_item<1> it) {
        copy_i32_body(dst, (const volatile int32_t*) src, (int) n, it, 128);
    });
}

sycl::event submit_rms_norm_weighted(sycl::queue& q, float* x, const float* w, int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return {};  // glue: CUDA line 315 (return; -> return {})
    // SYCL-MIRROR-BEGIN src/kernels/cuda/elementwise.cu:316:319 @ 22d022e
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    // SYCL-MIRROR-END
    // glue (CUDA line 320): the grid x 128 launch with the shfl32 local-memory
    // storage; check_launch/sync_if_needed dropped
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) grid * 128, 128),
                       [=](sycl::nd_item<1> it) { rms_body(x, w, rows, cols, eps, it, sfl, spin); });
    });
}

}  // namespace strata::kernels
