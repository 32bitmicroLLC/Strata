// poc/sycl/b0/b0_dp4a_probe.cpp -- B0 probe: the cuda_dp4a mapping.
//
// Decision under test: oneAPI 2026.1 exposes no public int8 dot intrinsic in
// its SYCL API (search of the installed headers for dot4/dp4a-style APIs
// came up empty; the repo's own STRATA_DP4A header exists precisely to
// provide a portable path). This probe therefore VALIDATES THE FALLBACK:
// ::strata::sycl_compat::dp4a (and the STRATA_DP4A macro that the mirrored
// kernel bodies use) vs an independently-written host reference of the
// documented semantics -- four SIGNED byte products of a and b, accumulated
// into c modulo 2^32 (the repo's include/strata/kernels/dp4a.hpp fallback,
// which is also the parity reference every port uses).
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
// Host reference, written independently of the header's implementation:
// plain int8_t extracts and products, int32 wraparound.
inline int ref_dp4a(int a, int b, int c) {
    int acc = c;
    const int8_t* pa = (const int8_t*)&a;
    const int8_t* pb = (const int8_t*)&b;
    for (int i = 0; i < 4; ++i) acc += (int)(pa[i] * pb[i]);
    return acc;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const size_t N = 1u << 16;
    std::vector<uint32_t> a(N), b(N), acc(N);
    {
        std::mt19937 rng(20260101u);
        for (size_t i = 0; i < N; ++i) {
            a[i] = rng();
            b[i] = rng();
            acc[i] = rng();
        }
        // Corner bytes in every lane: 0, +/-1, +/-127, -128, 0x7f, 0x80, 0xff.
        const uint32_t corners[16] = {0x00000000, 0x01010101, 0x7f7f7f7f, 0x80808080,
                                      0xff00ff00, 0x807f807f, 0x7f807f80, 0x01800180,
                                      0x80008000, 0xff7fff7f, 0x7fff7fff, 0x80ff80ff,
                                      0x00ffff00, 0x80808081, 0xfffffe7f, 0x01010180};
        for (size_t i = 0; i < N; i += 1024)
            for (int k = 0; k < 16; ++k)
                if (i + (size_t) k * 4 < N) {
                    a[i + (size_t) k * 4] = corners[(size_t) k];
                    b[i + (size_t) k * 4 + 1] = corners[(size_t) k + 8];
                }
    }

    uint32_t* da = c.device_alloc<uint32_t>(N);
    uint32_t* db = c.device_alloc<uint32_t>(N);
    uint32_t* dc = c.device_alloc<uint32_t>(N);
    uint32_t* out = c.device_alloc<uint32_t>(N);
    c.q.memcpy(da, a.data(), N * 4);
    c.q.memcpy(db, b.data(), N * 4);
    c.q.memcpy(dc, acc.data(), N * 4);
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) out[i] = (uint32_t) STRATA_DP4A((int) da[i], (int) db[i], (int) dc[i]);
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got(N);
    c.q.memcpy(got.data(), out, N * 4);
    c.wait_and_throw();

    size_t bad = 0, first = 0;
    for (size_t i = 0; i < N; ++i) {
        const int want = ref_dp4a((int) a[i], (int) b[i], (int) acc[i]);
        if ((uint32_t) want != got[i]) {
            if (bad < 8) {
                std::printf("    i %zu: a 0x%08x b 0x%08x c 0x%08x: want %d got %d\n", i, a[i],
                            b[i], acc[i], want, (int) got[i]);
                first = i;
            }
            ++bad;
        }
    }
    std::printf("b0_dp4a_probe: %s (%zu of %zu differ%s)\n", bad ? "*** FAIL ***" : "PASS", bad, N,
                bad ? "" : " -- fallback bit-exact; oneAPI int8 dot absent, documented in the report");
    (void) first;
    c.free_device(da);
    c.free_device(db);
    c.free_device(dc);
    c.free_device(out);
    return bad ? 1 : 0;
}
