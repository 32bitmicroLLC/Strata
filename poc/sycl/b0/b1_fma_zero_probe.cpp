// poc/sycl/b0/b1_fma_zero_probe.cpp -- assert that the Arc device fmaf plus
// the intrinsics.hpp __fmaf_rn glue correction is bit-exact against the
// IEEE-exact integer reference on ALL 125 degenerate zero input combinations
// (a, b, c in {+0, -0, +1, -1, 2^-100}).  A nonzero exit means the glue's
// signed-zero coverage has a hole.
#include "sycl_compat/test_ctx.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float fmaf_rn_ieee(float a, float b, float c) {
    uint32_t ua, ub, uc;
    std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4); std::memcpy(&uc, &c, 4);
    // value = m * 2^shift with m in [2^23, 2^24) (normal, shift = E-150)
    // or m < 2^23, shift = -149 (subnormal)
    auto parts = [](uint32_t u, int& shiftout) -> uint64_t {
        if (u == 0) return 0;
        uint32_t E = (u >> 23) & 0xFFu;
        uint32_t m = u & 0x7FFFFFu;
        if (E == 0) { shiftout = -149; return m; }
        shiftout = (int) E - 150;
        return (uint64_t) m | 0x800000u;
    };
    int ea, eb, ec;
    uint64_t ma = parts(ua, ea), mb = parts(ub, eb), mc = parts(uc, ec);
    // exact product: 49-bit significand, sign included in sp
    __uint128_t prod = 0; int ep = 0; int sp = 1;
    if (ma && mb) {
        prod = ma * mb;
        ep = ea + eb;
        sp = ((ua >> 31) ^ (ub >> 31)) ? -1 : 1;
    }
    // exact sum S * 2^es, S signed, aligned at the smaller exponent
    __int128 S = 0; int es = 0;
    bool zero_neg = false;
    int sc = (uc >> 31) ? -1 : 1;
    if (prod == 0 && mc == 0) {
        // IEEE zero sign: -0 only when both addends are -0
        zero_neg = (uc >> 31) && ((ua >> 31) ^ (ub >> 31));
    } else if (prod == 0) { S = (__int128) mc * sc; es = ec; }
    else if (mc == 0) { S = (__int128) prod * sp; es = ep; }
    else if (ep > ec) {
        int d = ep - ec;
        if (d > 77) { S = (__int128) prod * sp; es = ep; }  // c < p*2^-77: absorbed, exact for RNE
        else { S = (((__int128) prod * sp) << d) + ((__int128) mc * sc); es = ec; }
    } else {
        int d = ec - ep;
        if (d > 77) { S = (__int128) mc * sc; es = ec; }    // p < c*2^-77: absorbed, exact for RNE
        else { S = ((__int128) prod * sp) + (((__int128) mc * sc) << d); es = ep; }
    }
    // Round S * 2^es to f32.
    bool sgn = S < 0;
    __int128 M = sgn ? -S : S;
    float r;
    if (M == 0) {
        r = (sgn || zero_neg) ? -0.0f : 0.0f;
    } else {
        int k = 0; { __int128 t = M; while (t >> 1) { t >>= 1; ++k; } }  // k = floor(log2 M)
        if (es + k >= -126 && es + k <= 127) {
            // normal: 24-bit RNE of M at bit k
            uint32_t m24;
            int eRes;
            if (k >= 23) {
                m24 = (uint32_t) (M >> (k - 23));
                __uint128_t drop = (unsigned __int128) (M & (((__int128) 1 << (k - 23)) - 1));
                __uint128_t half = (__uint128_t) 1 << (k - 24);
                if (drop > half || (drop == half && (m24 & 1))) ++m24;
                eRes = es + k + (m24 == 0x1000000u ? 1 : 0);
                m24 &= 0x7FFFFFu;
            } else {
                m24 = (uint32_t) (M << (23 - k));  // exact, no dropped bits
                eRes = es + k;
            }
            r = __builtin_bit_cast(float, (uint32_t) (((uint32_t) (eRes + 127) << 23) | (m24 & 0x7FFFFFu) | (sgn ? 0x80000000u : 0u)));
        } else if (es + k > 127) {
            r = sgn ? -__builtin_inff() : __builtin_inff();
        } else {
            // subnormal: U = RNE(M * 2^(es+149)); es+k < -126 -> es+149 <= 22-k+... use fraction when negative
            int e149 = es + 149;              // > -149+... and <= (k-1-126)+149 = k+22
            __int128 U = 0;
            if (e149 >= 0) {
                U = M << e149;                 // exact (k + e149 <= k + k + 22 <= 2*126+22 < 128? -- guard below)
            } else if (-e149 <= 127) {
                __int128 q = (__int128) 1 << (-e149);
                __int128 quo = M / q, rem = M % q;
                if (2 * rem > q) ++quo;
                else if (2 * rem == q && (quo & 1)) ++quo;
                U = quo;
            } // -e149 > 127: M < 2^127 so M*2^e149 < 1/2 -> 0
            r = (U == 0) ? (sgn ? -0.0f : 0.0f)
                        : __builtin_bit_cast(float, (uint32_t) U);  // U < 2^23 here
        }
    }
    return r;
}

int main() {
    strata::sycl_compat::ctx ctxq;
    // a, b, c each in {+0, -0, +1, -1, +2^-100} -- covers every degenerate-zero FMA shape
    std::vector<float> vals = {0.0f, -0.0f, 1.0f, -1.0f, std::ldexp(1.0f, -100)};
    const int n = (int) vals.size();
    const int total = n * n * n;          // 125 cases
    const int pad = 128;                  // Arc rejects non-uniform work-groups: pad to one full group
    std::vector<float> a3(pad), b3(pad), c3(pad);
    int idx = 0;
    for (float ca : vals) for (float cb : vals) for (float cc : vals) {
        a3[idx] = ca; b3[idx] = cb; c3[idx] = cc; ++idx;
    }
    float* da = ctxq.device_alloc<float>(pad);
    float* db = ctxq.device_alloc<float>(pad);
    float* dc = ctxq.device_alloc<float>(pad);
    float* dr = ctxq.device_alloc<float>(pad);
    ctxq.q.memcpy(da, a3.data(), pad * 4);
    ctxq.q.memcpy(db, b3.data(), pad * 4);
    ctxq.q.memcpy(dc, c3.data(), pad * 4);
    sycl::event e = ctxq.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>((size_t) pad, 128), [=](sycl::nd_item<1> it) {
            size_t g = it.get_global_id(0);
            if (g >= total) return;
            // the device std::fmaf plus the EXACT intrinsics.hpp glue correction,
            // asserted against the IEEE-exact integer reference below
            float r = std::fmaf(da[g], db[g], dc[g]);
            uint32_t ur, uc;
            std::memcpy(&ur, &r, 4);
            std::memcpy(&uc, &dc[g], 4);
            if (uc == 0x00000000u && ur == 0x80000000u && (da[g] == 0.0f || db[g] == 0.0f)) r = 0.0f;
            dr[g] = r;
        });
    });
    e.wait();
    std::vector<float> got(pad);
    ctxq.q.memcpy(got.data(), dr, pad * 4);
    ctxq.wait_and_throw();
    int diff = 0;
    for (int i = 0; i < total; ++i) {
        float want = fmaf_rn_ieee(a3[i], b3[i], c3[i]);
        uint32_t ug, uw;
        std::memcpy(&ug, &got[i], 4); std::memcpy(&uw, &want, 4);
        if (ug != uw) {
            if (diff < 40) {
                std::printf("a=%+g b=%+g c=%+g: device 0x%08x ieee 0x%08x\n", a3[i], b3[i], c3[i], ug, uw);
            }
            ++diff;
        }
    }
    std::printf("fmaf zero-table: %d of %d differ from IEEE-exact\n", diff, total);
    return diff != 0;
}
