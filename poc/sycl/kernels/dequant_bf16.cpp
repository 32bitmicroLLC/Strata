// poc/sycl/kernels/dequant_bf16.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.5)
// SYCL port of src/kernels/cuda/dequant_bf16.cu @ faac2ab.  The kernel and the
// host dispatch are the CUDA source verbatim; only the qualifiers, the launch
// spelling and the stream type change.
//
// The mirrored blocks below are VERBATIM (checked by check_mirrors.sh).
// Glue:
//  - `__device__`/`__forceinline__`/`__global__` become empty macros (G4), so
//    the mirrored one-line helpers keep their CUDA spelling.
//  - `kv_iq4nl` is declared `constexpr` (the CUDA `__constant__` table; a
//    host static constant is readable from kernels -- probed), declared
//    outside the mirror so the mirrored table line keeps its spelling.
//  - `dim3` is a glue struct; CUDA's `blockIdx.x`/`threadIdx.x` are resolved
//    per work-item.
//  - `dequant_kernel`'s `__global__` signature is glue (its params keep
//    `__restrict__` = empty); its body is mirrored verbatim.
//  - The launch `<<<>>>` becomes `strata::sycl_compat::strata_launch` (one
//    padded 1-D range, 256-wide work-groups); `cudaGetLastError` is dropped
//    (SYCL throws), and the validation `exit(1)` becomes a thrown
//    std::runtime_error.
//  - The iq-only branch calls the iq-dequant surface ported in iq_dequant.cpp
//    (the CUDA file hard-references iq_kernels.cu's launchers).
#include "strata/kernels/dequant_bf16.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cstdint>
#include <cstdio>
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

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:17:19 @ faac2ab
__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:20:27 @ faac2ab
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:28:31 @ faac2ab
__device__ __forceinline__ void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__device__ __forceinline__ void put(H16* o, int i, float v) { o[i].v = __half_as_ushort(__float2half_rn(v)); }
__device__ __forceinline__ void put(float* o, int i, float v) { o[i] = v; }
// SYCL-MIRROR-END

// glue: the CUDA `__constant__` table, declared outside the mirror as a
// host static constant (readable from kernels -- probed).
static constexpr int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:35:38 @ faac2ab
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// SYCL-MIRROR-END

// The `group32` signature (CUDA lines 41-42) is glue; its body is mirrored.
template <int TYPE, typename T>
void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:43:154 @ faac2ab
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
    // SYCL-MIRROR-END
}

// The `dequant_kernel` `__global__` signature (CUDA lines 157-159) is glue;
// the body is mirrored.  One work-item per 32-element group (CUDA block of
// 256); the padded tail is skipped exactly as the CUDA bounds check does.
template <int TYPE, typename T>
void dequant_body(const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows, int64_t groups_per_row, T* out,
                  sycl::nd_item<1> it, int threads) {
    const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
    const dim3 threadIdx{(int) (it.get_global_id(0) % threads), 0, 0};
    const dim3 blockDim{threads, 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:160:163 @ faac2ab
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, out + r * groups_per_row * 32 + gi * 32);
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:166:180 @ faac2ab
bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}
// SYCL-MIRROR-END

// The CUDA launch (lines 183-210): the validation, geometry and dispatch
// are mirrored; `<<<>>>` becomes strata_launch and the trailing
// cudaGetLastError check is dropped (SYCL throws).  The dispatch macro is
// re-implemented as a glue switch -- same ten arms, same order.
template <typename T>
sycl::event launch(sycl::queue& q, int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:184:187 @ faac2ab
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
    // SYCL-MIRROR-END
        // glue: the CUDA `exit(1)` becomes a thrown error.
        throw std::runtime_error("dequant: unsupported type or shape");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:190:192 @ faac2ab
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    // SYCL-MIRROR-END
#define STRATA_DQ(TY) return strata::sycl_compat::strata_launch(q, (size_t) grid * 256, 256, [=](sycl::nd_item<1> it) { dequant_body<TY, T>(p, row_bytes, row0, rows, gpr, out, it, 256); })
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    throw std::runtime_error("dequant: unreachable type");
}

}  // namespace

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:214:217 @ faac2ab
bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:219:222 @ faac2ab
namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace
// SYCL-MIRROR-END

// forward decls: the iq-dequant surface (iq_dequant.cpp, B1 step 2.5 F4)
sycl::event submit_iq_dequant_f16(sycl::queue& q, int t, const void* src, int64_t n, uint16_t* dst);
sycl::event submit_iq_dequant_f32(sycl::queue& q, int t, const void* src, int64_t n, float* dst);
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

sycl::event submit_dequant_bf16(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               uint16_t* out) {
    return launch<uint16_t>(q, ggml_type, blocks, row0, rows, cols, out);
}

sycl::event submit_dequant_f16(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               uint16_t* out) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:231:231 @ faac2ab
    if (iq_only(ggml_type)) {
    // SYCL-MIRROR-END
        // glue: the CUDA `iq_dequant_f16(..., stream)` becomes the submit_ form.
        return submit_iq_dequant_f16(q, ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols),
                                     rows * cols, out);
    }
    // glue: `launch<H16>(...)` keeps its body; `reinterpret_cast` spelling from CUDA.
    return launch<H16>(q, ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out));
}

sycl::event submit_dequant_f32(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               float* out) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_bf16.cu:240:240 @ faac2ab
    if (iq_only(ggml_type)) {
    // SYCL-MIRROR-END
        // glue: the CUDA `iq_dequant_f32(..., stream)` becomes the submit_ form.
        return submit_iq_dequant_f32(q, ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols),
                                     rows * cols, out);
    }
    return launch<float>(q, ggml_type, blocks, row0, rows, cols, out);
}

}  // namespace strata::kernels
