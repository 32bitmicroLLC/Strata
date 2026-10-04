// scratch audit for the B1 F8 finding: is EVERY Q8_K deviation exactly the
// non-IEEE device division (1 ulp in iscale/d), with all logic otherwise exact?
#include "sycl_compat/test_ctx.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace strata::kernels {
sycl::event submit_quantize_q8_K(sycl::queue& q, const float* x, uint8_t* blocks, int64_t n);
}
static int nearest_int_host(float fval) {
    const float val = fval + 12582912.0f;
    int i; std::memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}
int main() {
    // replay the driver's ka exactly (same rng sequence as the mirrored driver's main)
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const long long N = 32 * 4096;
    for (long long i = 0; i < N; ++i) (void) gauss(rng);            // case a
    for (long long i = 0; i < N; ++i) (void) gauss(rng);            // case b (c, d, e draw nothing)
    const long long NK = 256 * 2048;
    std::vector<float> ka((size_t) NK);
    for (auto& v : ka) v = gauss(rng);

    strata::sycl_compat::ctx ctx;
    float* d_x = ctx.device_alloc<float>((size_t) NK);
    uint8_t* d_b = ctx.device_alloc<uint8_t>((size_t) NK / 256 * 292);
    ctx.q.memcpy(d_x, ka.data(), (size_t) NK * 4);
    strata::kernels::submit_quantize_q8_K(ctx.q, d_x, d_b, NK);
    ctx.wait_and_throw();
    std::vector<uint8_t> gb((size_t) NK / 256 * 292);
    ctx.q.memcpy(gb.data(), d_b, gb.size());
    // recover the DEVICE's own iscale per block with the same division the kernel does
    float* d_isc = ctx.device_alloc<float>(NK / 256);
    { float* a = d_x; float* r = d_isc; long long nn = NK / 256;
      ctx.q.submit([&](sycl::handler& hdl) {
        hdl.parallel_for(sycl::range<1>((size_t) nn), [=](sycl::id<1> i) {
            const float* xb = a + (size_t) i * 256;
            float mx = 0.0f, amx = 0.0f;
            for (int j = 0; j < 256; ++j) {
                const float ax = std::fabs(xb[j]);
                if (ax > amx) { amx = ax; mx = xb[j]; }
            }
            r[(size_t) i] = amx == 0.0f ? 0.0f : -127.0f / mx;
        });
      }); }
    ctx.wait_and_throw();
    std::vector<float> isc_dev(NK / 256);
    ctx.q.memcpy(isc_dev.data(), d_isc, NK / 256 * 4);

    int d_off = 0, d_1ulp = 0, d_other = 0;
    int q_1ulp_isc = 0, q_other = 0, bsum_bad = 0;
    for (long long b = 0; b < NK / 256; ++b) {
        const float* xb = ka.data() + b * 256;
        const uint8_t* blk = gb.data() + b * 292;
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax == 0.0f) continue;
        const float isc_ieee = -127.0f / max;
        const float d_ieee = 1.0f / isc_ieee;
        float d_got; std::memcpy(&d_got, blk, 4);
        if (*(int*) &d_got != *(int*) &d_ieee) {
            ++d_off;
            uint32_t lo = (uint32_t) *(int*)&d_got < (uint32_t) *(int*)&d_ieee ? (uint32_t) *(int*)&d_got : (uint32_t) *(int*)&d_ieee;
            uint32_t hi = (uint32_t) *(int*)&d_got < (uint32_t) *(int*)&d_ieee ? (uint32_t) *(int*)&d_ieee : (uint32_t) *(int*)&d_got;
            if (hi - lo == 1) ++d_1ulp; else { ++d_other; if (d_other <= 2) printf("d other: blk %lld ie %a got %a\n", b, (double) d_ieee, (double) d_got); }
        }
        // recompute qs with the DEVICE's own iscale (recovered above)
        const float isc_d = isc_dev[(size_t) b];
        const int8_t* qs = (const int8_t*) (blk + 4);
        const int16_t* bs = (const int16_t*) (blk + 4 + 256);
        for (int j = 0; j < 256; ++j) {
            const int v_ieee = nearest_int_host(isc_ieee * xb[j]);
            const int8_t q_ieee = (int8_t) std::min(127, v_ieee);
            const int v_dev = nearest_int_host(isc_d * xb[j]);
            const int8_t q_dev = (int8_t) std::min(127, v_dev);
            if (qs[j] != q_ieee) {
                if (qs[j] == q_dev) ++q_1ulp_isc; else { ++q_other; if (q_other <= 2) printf("q other: blk %lld j %d q_ieee %d q_dev %d got %d\n", b, j, q_ieee, q_dev, qs[j]); }
            }
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            if (bs[j] != (int16_t) sum) ++bsum_bad;
        }
    }
    printf("blocks 2048: d off %d (1ulp %d, other %d)\n", d_off, d_1ulp, d_other);
    printf("qs: %d explainable by the 1-ulp iscale, %d unexplained, bsums bad %d\n", q_1ulp_isc, q_other, bsum_bad);
}
