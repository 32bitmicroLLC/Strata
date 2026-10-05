// poc/sycl/kernels/kv_q8.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.7)
// SYCL port of src/kernels/cuda/kv_q8.cu @ fb2ccec: the INT8 KV append/gather
// pair of the QSA layers (one FP16 scale per 64 values).
//
// The mirrored blocks below are VERBATIM (checked by check_mirrors.sh).
// Glue:
//  - G4: empty device qualifiers; `dim3` glue resolved per work-item; the
//    3-D append grid (n_head_kv, head_dim/64, 2) x 64 is flattened to a 1-D
//    launch and the body resolves blockIdx.x/y/z from the global id
//    (semantics unchanged; each CUDA block maps to exactly one (h, g, is_v)).
//  - `__ldg`/`__float2int_rn` are the intrinsics.hpp layer; `__shfl_xor_sync`
//    expands onto the B0 `shfl32` object built in body glue (local-memory
//    32-wide warp emulation, B0 §1.3): the 64-thread block's two CUDA warps
//    are shfl32 groups 0 and 1, and the butterfly `o = 16..1` stays inside
//    each 32-lane group exactly as inside each CUDA warp.
//  - `__shared__ float warp_max[2]` is a handler-built local_accessor (T3
//    finding; router_top10/elementwise precedent); its two references and the
//    `__syncthreads` are glue lines (documented at the site).
//  - `char4`/`ushort4` are local glue PODs: the mirrored spellings must
//    resolve unqualified inside namespace strata::kernels, and the
//    reinterpret_cast pattern needs trivially-copyable types (DPC++'s
//    sycl::char4/ushort4 classes are different types).
//  - `KvHostPools`: the mapped pinned host pools are Phase C's protocol (the
//    audit's note); the driver passes nullptr, so the host-copy branch
//    compiles but is not exercised.
//  - Host dispatch: `<<<>>>` becomes `q.submit` (append: needs the local
//    accessors) / `strata_launch` (gather: the CUDA `blocks` formula is the
//    same padding); `check()` is dropped (SYCL throws); `validate`'s
//    `exit(1)` becomes a throw; `void* stream` becomes `sycl::queue&`;
//    `if (max_ids <= 0) return;` becomes `return {};`.
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

// glue: the CUDA vector builtin spellings the gather body uses.
struct char4 { int8_t x, y, z, w; };
struct ushort4 { uint16_t x, y, z, w; };

// CUDA line 21 signature is glue; the body (22-25) is mirrored except the
// `std::exit(1)`, which throws (SYCL host-side convention).
void validate(const QsaShapes& s, const char* what) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/kv_q8.cu:22:24 @ fb2ccec
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
    // SYCL-MIRROR-END
        throw std::runtime_error("kv_q8: head_dim must be a multiple of KV_Q8_GROUP and both counts positive");
    }
}

// The `kv_append_q8_kernel` `__global__` signature (CUDA lines 30-34) is glue;
// the body (35-66) is mirrored except the shared-memory lines:
//   line 43 `__shared__ float warp_max[2];`    -> the handler-built `s_wm`;
//   line 44 `warp_max[t >> 5] = a;`            -> glue `s_wm[t >> 5] = a;`
//   line 45 `__syncthreads();`                 -> glue `it.barrier();`
//   line 46 `fmaxf(warp_max[0], warp_max[1])`  -> glue `fmaxf(s_wm[0], s_wm[1])`
void kv_append_q8_body(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                       const int32_t* table, const int32_t* step, const float* kcur, const float* vcur,
                       int kv_heads, int head_dim, int page_size, KvHostPools host, sycl::nd_item<1> it,
                       const sycl::local_accessor<float, 1>& s_wm, const sycl::local_accessor<uint32_t, 1>& sfl,
                       const sycl::local_accessor<uint32_t, 1>& spin) {
    const int gr = head_dim / KV_Q8_GROUP;  // glue: for the flattened block index below
    const dim3 blockDim{64, 1, 1};
    // glue (G1): the CUDA 3-D grid flattened to 1-D; blockIdx.x = h, .y = g, .z = is_v.
    const int bid = (int) (it.get_global_id(0) / 64);
    const dim3 blockIdx{bid / (2 * gr), (bid / 2) % gr, bid % 2};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/kv_q8.cu:35:42 @ fb2ccec
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const bool is_v = blockIdx.z == 1;
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    // SYCL-MIRROR-END
    if ((t & 31) == 0) s_wm[t >> 5] = a;                 // glue: CUDA line 44 (warp_max[t >> 5] = a)
    it.barrier();                                        // glue: CUDA line 45 (__syncthreads)
    const float amax = fmaxf(s_wm[0], s_wm[1]);          // glue: CUDA line 46 (fmaxf(warp_max[0], warp_max[1]))
    // SYCL-MIRROR-BEGIN src/kernels/cuda/kv_q8.cu:47:65 @ fb2ccec
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(x / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
    // SYCL-MIRROR-END
}

// The `kv_gather_q8_kernel` `__global__` signature (CUDA lines 69-73) is glue;
// the body (74-98) is mirrored verbatim.
void kv_gather_q8_body(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* table, const int32_t* ids, const int32_t* step, int kv_heads, int head_dim,
                       int page_size, uint16_t* k_scratch, uint16_t* v_scratch, sycl::nd_item<1> it) {
    const dim3 blockDim{256, 1, 1};
    const dim3 blockIdx{(int) (it.get_global_id(0) / 256), 0, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/kv_q8.cu:74:98 @ fb2ccec
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const char4 kc = reinterpret_cast<const char4*>(k_q + row * head_dim)[q4];
    const char4 vc = reinterpret_cast<const char4*>(v_q + row * head_dim)[q4];
    ushort4 ko, vo;
    ko.x = f16_from_f32((float) kc.x * ks); ko.y = f16_from_f32((float) kc.y * ks);
    ko.z = f16_from_f32((float) kc.z * ks); ko.w = f16_from_f32((float) kc.w * ks);
    vo.x = f16_from_f32((float) vc.x * vs); vo.y = f16_from_f32((float) vc.y * vs);
    vo.z = f16_from_f32((float) vc.z * vs); vo.w = f16_from_f32((float) vc.w * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<ushort4*>(k_scratch)[dst] = ko;
    reinterpret_cast<ushort4*>(v_scratch)[dst] = vo;
    // SYCL-MIRROR-END
}

}  // namespace

// glue (CUDA lines 103-112): the 3-D grid (n_head_kv, head_dim/64, 2) with
// 64 threads per block becomes a flattened 1-D launch of an exact multiple
// of 64 work-items; the local accessors are handler-built (T3); `check()`
// is dropped (SYCL throws).
sycl::event kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                             const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                             const QsaShapes& s, sycl::queue& q, const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const size_t items = (size_t) s.n_head_kv * (size_t) (s.head_dim / KV_Q8_GROUP) * 2 * (size_t) KV_Q8_GROUP;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> s_wm(sycl::range<1>(2), h);
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(items, (size_t) KV_Q8_GROUP),
                       [=](sycl::nd_item<1> it) {
                           kv_append_q8_body(k_q, v_q, k_scale, v_scale, page_table, step, kcur, vcur,
                                              (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
                                              host ? *host : KvHostPools{}, it, s_wm, sfl, spin);
                       });
    });
}

// glue (CUDA lines 114-125): the 256-wide 1-D launch; the CUDA `blocks`
// formula is the same padding strata_launch applies; the in-kernel
// `i >= total` guard covers the pad; `check()` is dropped (SYCL throws).
sycl::event kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                             const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                             const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, sycl::queue& q) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return {};
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    return strata::sycl_compat::strata_launch(q, (size_t) total, 256, [=](sycl::nd_item<1> it) {
        kv_gather_q8_body(k_q, v_q, k_scale, v_scale, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
                          (int) s.page_size, k_scratch, v_scratch, it);
    });
}

}  // namespace strata::kernels
