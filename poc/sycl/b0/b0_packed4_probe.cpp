// poc/sycl/b0/b0_packed4_probe.cpp -- B0 probe: the packed byte-lane ops
// __vsub4 / __vadd4 / __vsubss4 / __vcmpne4 (the iq_kernels hazard set).
// Device implementations vs independently-written host references of the
// documented CUDA lane semantics (unsigned-byte wrap for vsub4/vadd4,
// signed-byte saturating subtract for vsubss4, 0xff/0x00 per-lane compare).
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
inline uint32_t ref_vsub4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= ((((a >> (8 * i)) & 0xffu) - ((b >> (8 * i)) & 0xffu)) & 0xffu) << (8 * i);
    return r;
}
inline uint32_t ref_vadd4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= ((((a >> (8 * i)) & 0xffu) + ((b >> (8 * i)) & 0xffu)) & 0xffu) << (8 * i);
    return r;
}
inline uint32_t ref_vsubss4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        int v = (int8_t)(a >> (8 * i)) - (int8_t)(b >> (8 * i));
        v = v < -128 ? -128 : (v > 127 ? 127 : v);
        r |= ((uint32_t) v & 0xffu) << (8 * i);
    }
    return r;
}
inline uint32_t ref_vcmpne4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        if (((a >> (8 * i)) & 0xffu) != ((b >> (8 * i)) & 0xffu)) r |= 0xffu << (8 * i);
    return r;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const size_t N = 1u << 16;
    std::vector<uint32_t> a(N), b(N);
    {
        std::mt19937 rng(7712u);
        for (size_t i = 0; i < N; ++i) {
            a[i] = rng();
            b[i] = rng();
        }
        // Corner bytes: saturation edges (127-(-128), -128-127, 0-1, 255-0)
        // and signed-boundary bytes in each lane.
        const uint32_t pairs[16] = {0x7f807f80, 0x807f807f, 0x00ff00ff, 0xff00ff00,
                                    0x7fff0000, 0x80007fff, 0x01000100, 0x00010001,
                                    0x7f008000, 0x80007f00, 0xffff7fff, 0x7ffffffe,
                                    0x80808080, 0x7f7f7f7f, 0x00000000, 0xffffffff};
        for (size_t i = 0; i < N; i += 512)
            for (int k = 0; k < 16; ++k) {
                if (i + (size_t) k * 2 < N) {
                    a[i + (size_t) k * 2] = pairs[(size_t) k];
                    b[i + (size_t) k * 2 + 1] = pairs[(size_t) (k + 8)];
                }
            }
    }

    uint32_t* da = c.device_alloc<uint32_t>(N);
    uint32_t* db = c.device_alloc<uint32_t>(N);
    uint32_t* out = c.device_alloc<uint32_t>(4 * N);
    c.q.memcpy(da, a.data(), N * 4);
    c.q.memcpy(db, b.data(), N * 4);
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) {
                out[i] = (uint32_t) __vsub4((int) da[i], (int) db[i]);
                out[N + i] = (uint32_t) __vadd4((int) da[i], (int) db[i]);
                out[2 * N + i] = (uint32_t) __vsubss4((int) da[i], (int) db[i]);
                out[3 * N + i] = (uint32_t) __vcmpne4((int) da[i], (int) db[i]);
            }
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got(4 * N);
    c.q.memcpy(got.data(), out, 4 * N * 4);
    c.wait_and_throw();

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) {
        const uint32_t want[4] = {ref_vsub4(a[i], b[i]), ref_vadd4(a[i], b[i]),
                                  ref_vsubss4(a[i], b[i]), ref_vcmpne4(a[i], b[i])};
        for (int k = 0; k < 4; ++k)
            if (got[k * N + i] != want[k]) {
                if (bad < 8)
                    std::printf("    op %d i %zu: a 0x%08x b 0x%08x: want 0x%08x got 0x%08x\n", k, i,
                                a[i], b[i], want[k], got[k * N + i]);
                ++bad;
            }
    }
    std::printf("b0_packed4_probe: %s (%zu of %zu values differ)\n", bad ? "*** FAIL ***" : "PASS",
                bad, 4 * N);
    c.free_device(da);
    c.free_device(db);
    c.free_device(out);
    return bad ? 1 : 0;
}
