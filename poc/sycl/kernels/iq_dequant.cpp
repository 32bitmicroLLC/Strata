// poc/sycl/kernels/iq_dequant.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.5,
// finding F4): the iq-dequant surface of src/kernels/cuda/iq_kernels.cu @ 705ff06
// that dequant_bf16.cu hard-references.  The CUDA `dequant_f16/f32` call
// `iq_dequant_f16/f32` + `iq_row_bytes`, so this TU is the linkable closure:
//   cvt, the ten dq_* device functions, dq_dispatch, dequant_flat_kernel,
//   is_iq, iq_supported, iq_row_bytes, and the two host launchers.
// The vec_dot_*/mmvq/quantize_q8_1/embed_rows/gu parts of iq_kernels.cu stay
// in B3 (its iq_kernels.cpp mirrors the remainder and stops covering the
// ranges pinned here).
//
// The mirrored blocks are VERBATIM (checked by check_mirrors.sh).  Glue:
//  - the block structs and codebook grids come from third_party/ggml/
//    ggml-common.h, included with the GGML_COMMON_*_SYCL variants (the CUDA
//    file uses the *_CUDA macros; the SYCL variants are the header's own
//    plain-`static const` host tables, reachable from kernels -- probed).
//  - `__device__`/`__forceinline__` are TU-local empty macros (G4); the
//    dequant_flat_kernel `__global__` signature is glue, its body mirrored.
//  - the launcher validation keeps `std::exit(1)` VERBATIM: unlike the T4
//    files, the exit shares its line with the stderr message, so a split
//    mirror block is impossible; the loud failure matches CUDA.  The
//    cudaGetLastError `check` is dropped (SYCL throws); cudaStream_t becomes
//    sycl::queue.
#include "strata/kernels/iq_kernels.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace strata::kernels {
namespace {

// glue (G4): empty device qualifiers so the mirrored one-line and
// signature lines keep their CUDA spelling.
#define __device__
#define __forceinline__

// G1 glue: CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:425:427 @ 705ff06
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:429:441 @ 705ff06
template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:442:452 @ 705ff06
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:453:462 @ 705ff06
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:463:479 @ 705ff06
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:480:494 @ 705ff06
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:495:512 @ 705ff06
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:513:524 @ 705ff06
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:525:546 @ 705ff06
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:547:558 @ 705ff06
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:559:570 @ 705ff06
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:572:587 @ 705ff06
template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}
// SYCL-MIRROR-END

// The `dequant_flat_kernel` signature (CUDA lines 590-591) is glue; the
// body (592-593) is mirrored.  One work-group per superblock (CUDA grid =
// n/256 blocks of 32), so no padding is needed and no bounds check exists in
// the kernel: the launch passes the exact logical count.
template<typename dst_t>
void dequant_flat_body(int ty, const void* vx, dst_t* y, sycl::nd_item<1> it) {
    const dim3 blockIdx{(int) (it.get_global_id(0) / 32), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:592:593 @ 705ff06
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:604:604 @ 705ff06
bool is_iq(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11; }
// SYCL-MIRROR-END

}  // namespace

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:608:608 @ 705ff06
bool iq_supported(int t) noexcept { return is_iq(t); }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:610:624 @ 705ff06
size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}
// SYCL-MIRROR-END

sycl::event submit_iq_dequant_f16(sycl::queue& q, int t, const void* src, int64_t n, uint16_t* dst) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:655:655 @ 705ff06
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    // SYCL-MIRROR-END
    // glue (CUDA line 656): the launch; `check` dropped (SYCL throws)
    return strata::sycl_compat::strata_launch(q, (size_t) (n / 256) * 32, 32,
                                              [=](sycl::nd_item<1> it) {
        dequant_flat_body<__half>(t, src, (__half*) dst, it);
    });
}

sycl::event submit_iq_dequant_f32(sycl::queue& q, int t, const void* src, int64_t n, float* dst) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/iq_kernels.cu:680:680 @ 705ff06
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    // SYCL-MIRROR-END
    // glue (CUDA line 681): the launch; `check` dropped (SYCL throws)
    return strata::sycl_compat::strata_launch(q, (size_t) (n / 256) * 32, 32,
                                              [=](sycl::nd_item<1> it) {
        dequant_flat_body<float>(t, src, dst, it);
    });
}

// glue: host-side decode hooks for the parity driver.  They run the SAME
// mirrored dq_dispatch code on the host -- this verifies the device launch,
// memory and indexing, but is NOT an independent reference: strata/
// artifact/dequant.hpp has no dequantize_iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/
// iq1_m, so the iq-only types cannot be checked against the validated chain
// until that chain is extended (recorded in the step report).
void host_decode_iq_f32(int t, const void* src, int64_t n, float* dst) {
    if (n % 256 != 0 || !is_iq(t)) throw std::runtime_error("host_decode_iq_f32: bad arguments");
    for (int64_t i = 0; i < n / 256; ++i)
        for (int tid = 0; tid < 32; ++tid) dq_dispatch<float>(t, src, i, dst + i * QK_K, tid);
}
void host_decode_iq_f16(int t, const void* src, int64_t n, uint16_t* dst) {
    if (n % 256 != 0 || !is_iq(t)) throw std::runtime_error("host_decode_iq_f16: bad arguments");
    for (int64_t i = 0; i < n / 256; ++i)
        for (int tid = 0; tid < 32; ++tid) dq_dispatch<__half>(t, src, i, (__half*) dst + i * QK_K, tid);
}

}  // namespace strata::kernels
