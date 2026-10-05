// poc/sycl/b0/b0_half_probe.cpp -- B0 probe: the CUDA __half mapping to
// sycl::half (the tree's half precision is __half = IEEE binary16; the
// audit found NO __nv_bfloat16 use in the 40 files, so bfloat16 stays a
// typedef-only convenience and this probe covers the __half family).
//
// Checked: __float2half_rn (RNE float->binary16: every fp16 normal and the
// tie between each consecutive pair, the subnormal/flush region, and
// saturation to +-inf), __half2float (with the round-trip identity), the
// __half2 ops (__hsub2, the __floats2half2_rn CROSS-pack {a.y, b.x},
// __halves2half2) and the bit casts (__half_as_ushort / __ushort_as_half).
// References are host C++ written independently of the header (bit-level
// RNE to binary16).
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {
// Independent RNE float -> binary16 (bits).
//   e' = e - 112 is the fp16 exponent field (e' in 1..30 for normals,
//   31 reserved for inf/NaN); e < 16 means below the fp16 normal range and
//   rounds through exact double arithmetic; e >= 143 saturates to inf.
inline uint16_t ref_f2h(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000);
    const uint32_t e = (u >> 23) & 0xff;
    const uint32_t f = u & 0x7fffff;
    if (e == 0xff) return sign | (f ? 0x7c01u : 0x7c00u);  // NaN / +-inf
    if (e >= 143) return sign | 0x7c00u;  // |x| >= 2^16: saturate to inf
    if (e < 113) {  // below 2^-14: fp16 subnormal or zero; exact in double
        double d = x >= 0 ? (double) x : -(double) x;
        double m = d * (double) (1 << 24);  // fp16 subnormal units
        long l = (long) std::floor(m);
        const double fr = m - (double) l;
        if (fr > 0.5 || (fr == 0.5 && (l & 1))) ++l;  // RNE
        if (l >= 1024) return sign | 0x0400u;  // 1024 * 2^-24 = 2^-14, smallest normal
        return sign | (uint16_t) l;
    }
    uint32_t e2 = e;  // mutable: a mantissa carry can roll into the exponent
    uint32_t dropped = f & 0x1fff;
    uint32_t kept = f >> 13;  // the 10 stored fraction bits (bits 22..13)
    if (dropped > 0x1000 || (dropped == 0x1000 && (kept & 1))) {
        ++kept;
        if (kept > 0x3ff) {  // mantissa carry rolls into the exponent
            kept = 0;
            ++e2;
            if (e2 >= 143) return sign | 0x7c00u;  // only 65504.x rounds to inf
        }
    }
    return sign | (uint16_t) (((e2 - 112) << 10) | kept);
}

inline float ref_h2f(uint16_t h) {
    const uint32_t s = h & 0x8000;
    const uint32_t e = (h >> 10) & 0x1f;
    const uint32_t f = h & 0x3ff;
    double d;
    if (e == 0)
        d = (double) f * std::ldexp(1.0, -24);
    else if (e == 0x1f)
        d = std::numeric_limits<double>::infinity();
    else
        d = (1.0 + (double) f * std::ldexp(1.0, -10)) * std::ldexp(1.0, (int) e - 15);
    const float r = (float) d;  // exact fp16 values are exactly representable as float
    return s ? -r : r;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    std::vector<uint32_t> bits;
    auto push_float = [&](double d) {
        if (std::isfinite(d) && std::fabs(d) < 1e30) {
            const float f = (float) d;
            uint32_t u;
            std::memcpy(&u, &f, 4);
            bits.push_back(u);
        }
    };
    // Every fp16 normal (exact as float) plus the exact midpoint tie between
    // each consecutive pair -- RNE must pick the even mantissa.
    for (uint16_t h = 1; h <= 0x7be0; h += 4) {  // sampled: 2056 normals
        push_float((double) ref_h2f(h));
        push_float(((double) ref_h2f(h) + (double) ref_h2f(h + 1)) * 0.5);
    }
    for (uint16_t h = 0x7be0; h <= 0x7bff; ++h) {  // the whole top-end densely
        push_float((double) ref_h2f(h));
        push_float(((double) ref_h2f(h) + (double) ref_h2f(h + 1)) * 0.5);
    }
    push_float(0.0);
    push_float(-0.0);
    push_float(std::numeric_limits<float>::denorm_min());
    push_float(2.0 * std::numeric_limits<float>::denorm_min());  // flush tie region
    push_float(4.9e-32);  // just above half of the first subnormal step
    push_float(0x1p-14);       // smallest fp16 normal
    push_float(0x1p-15);       // tie: 0 <-> 2^-14
    push_float(0x1p15);        // 32768, middle of the exponent range
    push_float(65504.0);       // largest fp16
    push_float(65520.0);       // 65504 <-> inf midpoint (16 to each)
    push_float(65535.0);       // must round up to inf
    push_float(65536.0);       // must saturate to inf
    push_float(-65536.0);
    push_float(1e-10);
    push_float(-1e-10);
    // Random across ranges.
    {
        std::mt19937 rng(5231u);
        for (int i = 0; i < (1 << 20); ++i) {
            double m = std::ldexp((double) ((rng() % 1048576) - 524288), -(int) (rng() % 40));
            push_float((rng() & 1) ? m : -m);
        }
    }

    const size_t N = bits.size();
    uint32_t* dbits = c.device_alloc<uint32_t>(N);
    uint16_t* dout = c.device_alloc<uint16_t>(N);
    uint16_t* back = c.device_alloc<uint16_t>(N);  // round trip: f(h(f(x)))
    c.q.memcpy(dbits, bits.data(), N * 4);
    const size_t G = (N + 255) / 256 * 256;  // Arc: uniform work-groups only
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(G, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) {
                float x;
                std::memcpy(&x, &dbits[i], 4);
                const __half hf = __float2half_rn(x);
                dout[i] = __half_as_ushort(hf);
                back[i] = __half_as_ushort(__float2half_rn(__half2float(hf)));
            }
        });
    });
    e.wait_and_throw();

    std::vector<uint16_t> got(N), rt(N);
    c.q.memcpy(got.data(), dout, N * 2);
    c.q.memcpy(rt.data(), back, N * 2);
    c.wait_and_throw();

    size_t bad = 0, bad_rt = 0;
    for (size_t i = 0; i < N; ++i) {
        float x;
        std::memcpy(&x, &bits[i], 4);
        const uint16_t want = ref_f2h(x);
        if (got[i] != want &&
            !(std::isnan(x) && (got[i] & 0x7c00) == 0x7c00)) {  // NaN payload: impl-defined
            if (bad < 8)
                std::printf("    i %zu: bits 0x%08x (%g): want 0x%04x got 0x%04x\n", i, bits[i],
                            (double) x, want, got[i]);
            ++bad;
        }
        // An exact fp16 value is exactly representable as float, so the
        // f->h->f->h round trip must be the identity.
        if (rt[i] != got[i]) {
            if (bad_rt < 4) std::printf("    round-trip i %zu: 0x%04x -> 0x%04x\n", i, got[i], rt[i]);
            ++bad_rt;
        }
    }

    // The __half2 ops on the device, vs host references.
    std::vector<uint32_t> pa(8), pb(8);
    for (int i = 0; i < 8; ++i) {
        pa[i] = (uint32_t) ref_f2h((float) i) | ((uint32_t) ref_f2h(-0.5f * (float) i) << 16);
        pb[i] = (uint32_t) ref_f2h(0.25f * (float) i) | ((uint32_t) ref_f2h(-(float) i) << 16);
    }
    uint32_t* dpa = c.device_alloc<uint32_t>(8);
    uint32_t* dpb = c.device_alloc<uint32_t>(8);
    uint32_t* dres = c.device_alloc<uint32_t>(24);
    c.q.memcpy(dpa, pa.data(), 32);
    c.q.memcpy(dpb, pb.data(), 32);
    e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(8, 8), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_global_id(0);
            __half2 a{__ushort_as_half((uint16_t) (dpa[i] & 0xffff)),
                      __ushort_as_half((uint16_t) (dpa[i] >> 16))};
            __half2 b{__ushort_as_half((uint16_t) (dpb[i] & 0xffff)),
                      __ushort_as_half((uint16_t) (dpb[i] >> 16))};
            const __half2 s = __hsub2(a, b);
            float2 fa{(float) a.x, (float) a.y}, fb{(float) b.x, (float) b.y};
            const __half2 cross = __floats2half2_rn(fa, fb);
            const __half2 hh = __halves2half2(1.5f, -2.5f);
            dres[3 * i] = (uint32_t) __half_as_ushort(s.x) | ((uint32_t) __half_as_ushort(s.y) << 16);
            dres[3 * i + 1] = (uint32_t) __half_as_ushort(cross.x) | ((uint32_t) __half_as_ushort(cross.y) << 16);
            dres[3 * i + 2] = (uint32_t) __half_as_ushort(hh.x) | ((uint32_t) __half_as_ushort(hh.y) << 16);
        });
    });
    e.wait_and_throw();
    std::vector<uint32_t> res(24);
    c.q.memcpy(res.data(), dres, 96);
    c.wait_and_throw();

    size_t bad2 = 0;
    for (int i = 0; i < 8; ++i) {
        const float ax = (float) ref_h2f((uint16_t) (pa[i] & 0xffff));
        const float ay = (float) ref_h2f((uint16_t) (pa[i] >> 16));
        const float bx = (float) ref_h2f((uint16_t) (pb[i] & 0xffff));
        const float by = (float) ref_h2f((uint16_t) (pb[i] >> 16));
        const uint32_t want_s = (uint32_t) ref_f2h(ax - bx) | ((uint32_t) ref_f2h(ay - by) << 16);
        const uint32_t want_c = (uint32_t) ref_f2h(ay) | ((uint32_t) ref_f2h(bx) << 16);  // {a.y, b.x}
        const uint32_t want_h = (uint32_t) ref_f2h(1.5f) | ((uint32_t) ref_f2h(-2.5f) << 16);
        if (res[3 * i] != want_s || res[3 * i + 1] != want_c || res[3 * i + 2] != want_h) {
            if (bad2 < 8)
                std::printf("    i %d: got %08x %08x %08x want %08x %08x %08x\n", i, res[3 * i],
                            res[3 * i + 1], res[3 * i + 2], want_s, want_c, want_h);
            ++bad2;
        }
    }

    const size_t total = bad + bad_rt + bad2;
    std::printf("b0_half_probe: %s (%zu of %zu differ: %zu convert, %zu round-trip, %zu half2 ops)\n",
                total ? "*** FAIL ***" : "PASS", total, N + 8, bad, bad_rt, bad2);
    c.free_device(dbits);
    c.free_device(dout);
    c.free_device(back);
    c.free_device(dpa);
    c.free_device(dpb);
    c.free_device(dres);
    return total ? 1 : 0;
}
