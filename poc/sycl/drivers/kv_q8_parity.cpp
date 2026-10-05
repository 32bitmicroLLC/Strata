// poc/sycl/drivers/kv_q8_parity.cpp -- B1 (plans/sycl-phase-b-steps-b1.md,
// step 2.7, finding F3): the q8 half of src/kernels/kv_q8_parity.cpp @ 733d7c7.
//
// The CUDA driver appends/gathers random K/V cells through a non-identity
// page table into BOTH the INT8 pools (kv_append_q8_step) and the FP16 pools
// (kv_append_step) and runs three checks.  The FP16 pool entry points live in
// qsa.cu (a B5 file, F3), so this driver mirrors and runs the q8 path only:
//   1. INT8 codes and scales are BITWISE equal to the host reference (same
//      f16_from_f32 rounding);
//   2. the INT8 gather is BITWISE equal to the host dequantization of those
//      codes.
// CHECK #3 (INT8 vs FP16 error <= 0.624 quantization steps) and every
// `kv_append_step`/`kv_gather_step` call (CUDA lines 44, 66, 101, 111,
// 116-117, 129-131, 138-139) are parked in B5's qsa driver.
//
// The mirrored blocks are verbatim (checked by check_mirrors.sh); everything
// else is glue: `ctx`/queue for the CUDA stream and syncs, `dalloc` with a
// zero fill for the CUDA malloc+memset, `KvHostPools` left null (the mapped
// pinned host pools are Phase C's protocol; the driver never populated them),
// and the `--selftest` argument is the ctest contract spelling (the CUDA
// driver is single-mode and the argument is accepted, not branched on).
//
// PLATFORM NOTE (measured, see the step report): the F8 gap (icpx 2026.1
// emits non-IEEE f32 division on Arc) applies to the append kernel's
// variable-divisor `x / sf`; the constant-divisor `amax / 127.0f` is
// IEEE-exact on the device (probed: 0/200,000 off).  The bit-exact gate
// below is therefore the F8 knife edge exactly as documented for Q8_K:
// a code byte can flip only when the division lands within one f32 ulp of
// a half-integer tie of `__float2int_rn`.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

// forward declarations: the port defines these in poc/sycl/kernels/kv_q8.cpp
// (`void* stream` -> `sycl::queue&`, per the Phase B convention; the header's
// CUDA overloads remain in scope and are never viable here, since the
// stream position takes a queue, not a pointer).
namespace strata::kernels {
sycl::event kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                             const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                             const QsaShapes& s, sycl::queue& q, const KvHostPools* host = nullptr);
sycl::event kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                             const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                             const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, sycl::queue& q);
}  // namespace strata::kernels

namespace {
int g_fail = 0;
// glue for the CUDA driver's ck/dalloc (lines 26-29): device malloc + zero
// fill, with the queue standing in for cudaMemcpy/DeviceSynchronize.
template <typename T> T* dalloc(strata::sycl_compat::ctx& c, size_t n) {
    T* p = c.device_alloc<T>(n);
    // glue: a device fill stands in for the CUDA driver's cudaMemset.
    c.q.fill<T>(p, T{}, n);
    return p;
}
}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc == 1 || (argc == 2 && std::strcmp(argv[1], "--selftest") == 0);
    if (!selftest) {
        std::fprintf(stderr, "usage: kv_q8_parity [--selftest]\n");
        return 2;
    }
    strata::sycl_compat::ctx ctxq;
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:33:39 @ 733d7c7
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;                                                              // several pages in a small pool
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / k::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;
    std::mt19937 rng(17);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 5 + 3) % pages;            // a non-identity permutation
    // SYCL-MIRROR-END
    int32_t* d_table = dalloc<int32_t>(ctxq, (size_t) pages);
    ctxq.q.memcpy(d_table, table.data(), (size_t) pages * 4);
    int8_t *kq = dalloc<int8_t>(ctxq, (size_t) cells * H * D), *vq = dalloc<int8_t>(ctxq, (size_t) cells * H * D);
    uint16_t *ks = dalloc<uint16_t>(ctxq, (size_t) cells * H * G), *vs = dalloc<uint16_t>(ctxq, (size_t) cells * H * G);
    // (F3: CUDA line 44's FP16 pools kp/vp are parked in B5's qsa driver)
    float *kcur = dalloc<float>(ctxq, (size_t) H * D), *vcur = dalloc<float>(ctxq, (size_t) H * D);
    int32_t* step = dalloc<int32_t>(ctxq, (size_t) k::kStepCount);
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:47:53 @ 733d7c7
    std::vector<std::vector<float>> hk(cells), hv(cells);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    const int n_fill = cells - 37;                                                  // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
    // SYCL-MIRROR-END
        // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:54:61 @ 733d7c7
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = (n % 7 == 0) ? 1e-3f : (n % 11 == 0 ? 40.f : 1.f);    // small and large groups
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (n % 13 == 0) std::fill(kv.begin(), kv.begin() + 64, 0.f);              // an all-zero group
        hk[pos] = kv; hv[pos] = vv;
        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        // SYCL-MIRROR-END
        ctxq.q.memcpy(step, hstep, sizeof hstep);
        ctxq.q.memcpy(kcur, kv.data(), kv.size() * 4);
        ctxq.q.memcpy(vcur, vv.data(), vv.size() * 4);
        k::kv_append_q8_step(kq, vq, ks, vs, d_table, step, kcur, vcur, s, ctxq.q);
        // (F3: CUDA line 66 `k::kv_append_step(kp, vp, ...)` is parked in B5's qsa driver)
        ctxq.wait_and_throw();
    }
    // 1. codes and scales bitwise vs the host reference
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:70:71 @ 733d7c7
    std::vector<int8_t> hkq((size_t) cells * H * D), hvq(hkq.size());
    std::vector<uint16_t> hks((size_t) cells * H * G), hvs(hks.size());
    // SYCL-MIRROR-END
    ctxq.q.memcpy(hkq.data(), kq, hkq.size());
    ctxq.q.memcpy(hvq.data(), vq, hvq.size());
    ctxq.q.memcpy(hks.data(), ks, hks.size() * 2);
    ctxq.q.memcpy(hvs.data(), vs, hvs.size() * 2);
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:76:96 @ 733d7c7
    long bad_codes = 0;
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int kvsel = 0; kvsel < 2; ++kvsel)
            for (int h = 0; h < H; ++h)
                for (int g = 0; g < G; ++g) {
                    const float* x = (kvsel ? hv[pos] : hk[pos]).data() + h * D + g * 64;
                    float amax = 0.f;
                    for (int t = 0; t < 64; ++t) amax = std::fmax(amax, std::fabs(x[t]));
                    const uint16_t sb = k::f16_from_f32(amax / 127.0f);
                    const float sf = k::f32_from_f16(sb);
                    const long long row = ((long long) table[pos / P] * H + h) * P + pos % P;
                    if ((kvsel ? hvs : hks)[row * G + g] != sb) ++bad_codes;
                    for (int t = 0; t < 64; ++t) {
                        int q = sf > 0.f ? (int) std::nearbyint(x[t] / sf) : 0;
                        q = std::clamp(q, -127, 127);
                        if ((kvsel ? hvq : hkq)[row * D + g * 64 + t] != (int8_t) q) ++bad_codes;
                    }
                }
    }
    if (bad_codes) { std::fprintf(stderr, "FAIL: %ld codes/scales differ from the host reference\n", bad_codes); ++g_fail; }
    // SYCL-MIRROR-END
    // 2. gather random selections from the q8 path
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:98:98 @ 733d7c7
    const int max_ids = 200;
    // SYCL-MIRROR-END
    int32_t* d_ids = dalloc<int32_t>(ctxq, (size_t) max_ids);
    uint16_t *k8 = dalloc<uint16_t>(ctxq, (size_t) max_ids * H * D), *v8 = dalloc<uint16_t>(ctxq, (size_t) max_ids * H * D);
    // (F3: CUDA line 101's FP16 scratch k16/v16, line 102's `worst`, and the
    // check-#3 error accumulation are parked in B5's qsa driver)
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:103:106 @ 733d7c7
    for (int trial = 0; trial < 20; ++trial) {
        const int n_ids = 1 + (int) (rng() % max_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& id : ids) id = positions[rng() % n_fill];
        // SYCL-MIRROR-END
        ctxq.q.memcpy(d_ids, ids.data(), (size_t) n_ids * 4);
        // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:108:108 @ 733d7c7
        int32_t hstep[k::kStepCount] = {0, 0, 0, n_ids};
        // SYCL-MIRROR-END
        ctxq.q.memcpy(step, hstep, sizeof hstep);
        k::kv_gather_q8_step(kq, vq, ks, vs, d_table, d_ids, step, max_ids, s, k8, v8, ctxq.q);
        // (F3: CUDA line 111 `k::kv_gather_step(...)` is parked in B5's qsa driver)
        ctxq.wait_and_throw();
        std::vector<uint16_t> a8((size_t) n_ids * H * D), b8(a8.size());
        // (F3: CUDA line 113's a16/b16 and lines 116-117's downloads are parked)
        ctxq.q.memcpy(a8.data(), k8, a8.size() * 2);
        ctxq.q.memcpy(b8.data(), v8, b8.size() * 2);
        // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:118:128 @ 733d7c7
        for (int j = 0; j < n_ids; ++j)
            for (int h = 0; h < H; ++h) {
                const long long row = ((long long) table[ids[j] / P] * H + h) * P + ids[j] % P;
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) j * H + h) * D + d;
                    for (int kvsel = 0; kvsel < 2; ++kvsel) {
                        const float sf = k::f32_from_f16((kvsel ? hvs : hks)[row * G + d / 64]);
                        const int8_t q = (kvsel ? hvq : hkq)[row * D + d];
                        const uint16_t want = k::f16_from_f32((float) q * sf);
                        const uint16_t got = (kvsel ? b8 : a8)[o];
                        if (got != want) { if (g_fail < 5) std::fprintf(stderr, "FAIL: gather id %d h %d d %d\n", j, h, d); ++g_fail; }
    // SYCL-MIRROR-END
    // (F3: CUDA lines 129-131 -- the FP16 reference and the error accumulation
    // -- are parked in B5's qsa driver)
        // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:132:135 @ 733d7c7
                    }
                }
            }
    }
    // SYCL-MIRROR-END
    // (F3: CUDA lines 136-139 -- the INT8-vs-FP16 step-error gate and its
    // printf -- are parked in B5's qsa driver with the FP16 section)
    std::printf("kv_q8_parity: %s (q8 subset; the FP16 cross-check parks in B5, F3)\n", g_fail ? "FAILED" : "OK");
    // SYCL-MIRROR-BEGIN src/kernels/kv_q8_parity.cpp:140:140 @ 733d7c7
    return g_fail ? 1 : 0;
    // SYCL-MIRROR-END
}
