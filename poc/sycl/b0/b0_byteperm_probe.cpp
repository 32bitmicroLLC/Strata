// poc/sycl/b0/b0_byteperm_probe.cpp -- B0 probe: __byte_perm.
//
// Device ::strata::sycl_compat::byte_perm (shift-based) vs a host reference
// written as a LUT over the 8-byte operand pair (a's bytes 0..3, then b's
// bytes 0..3, numbered from the LSB) -- the documented CUDA semantics,
// coded independently of the implementation under test.
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
inline uint32_t ref_byte_perm(int a, int b, int sel) {
    const uint8_t tab[8] = {(uint8_t)(a & 0xff), (uint8_t)((a >> 8) & 0xff),
                            (uint8_t)((a >> 16) & 0xff), (uint8_t)((a >> 24) & 0xff),
                            (uint8_t)(b & 0xff), (uint8_t)((b >> 8) & 0xff),
                            (uint8_t)((b >> 16) & 0xff), (uint8_t)((b >> 24) & 0xff)};
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (uint32_t) tab[(sel >> (8 * i)) & 7] << (8 * i);
    return r;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const size_t N = 1u << 16;
    std::vector<uint32_t> a(N), b(N), sel(N);
    {
        std::mt19937 rng(41501u);
        for (size_t i = 0; i < N; ++i) {
            a[i] = rng();
            b[i] = rng();
            sel[i] = rng();
        }
        // Every 8-byte selector pattern on two fixed distinct operands.
        for (int s = 0; s < 256; ++s) {
            a[(size_t) s] = 0x11223344u;
            b[(size_t) s] = 0x55667788u;
            sel[(size_t) s] = (uint32_t) s * 0x01010101u;  // same sel byte in every lane
        }
        for (int s = 256; s < 512; ++s) {
            a[s] = 0x00ff00ffu;
            b[s] = 0xff00ff00u;
            sel[s] = (uint32_t) s;  // distinct per-lane picks over the low 24 bits
        }
    }

    uint32_t* da = c.device_alloc<uint32_t>(N);
    uint32_t* db = c.device_alloc<uint32_t>(N);
    uint32_t* ds = c.device_alloc<uint32_t>(N);
    uint32_t* out = c.device_alloc<uint32_t>(N);
    c.q.memcpy(da, a.data(), N * 4);
    c.q.memcpy(db, b.data(), N * 4);
    c.q.memcpy(ds, sel.data(), N * 4);
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) out[i] = __byte_perm((int) da[i], (int) db[i], (int) ds[i]);
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got(N);
    c.q.memcpy(got.data(), out, N * 4);
    c.wait_and_throw();

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i)
        if (got[i] != ref_byte_perm((int) a[i], (int) b[i], (int) sel[i])) {
            if (bad < 8)
                std::printf("    i %zu: a 0x%08x b 0x%08x sel 0x%08x: want 0x%08x got 0x%08x\n", i,
                            a[i], b[i], sel[i], ref_byte_perm((int) a[i], (int) b[i], (int) sel[i]), got[i]);
            ++bad;
        }
    std::printf("b0_byteperm_probe: %s (%zu of %zu differ)\n", bad ? "*** FAIL ***" : "PASS", bad, N);
    c.free_device(da);
    c.free_device(db);
    c.free_device(ds);
    c.free_device(out);
    return bad ? 1 : 0;
}
