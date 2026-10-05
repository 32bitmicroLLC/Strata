// poc/sycl/drivers/cvec_parity.cpp -- B1 (plans/sycl-phase-b-steps-b1.md,
// step 2.9): the control vector kernel (strata/kernels/cvec.hpp) against a
// host reference, mirroring src/kernels/cvec_parity.cpp's gates.
//
// SCOPE (finding F2): the CUDA driver's section 3 cross-checks the kernel's
// folded write against `fused_gr_read(apply)` -- that reference is B4's
// fused_gr port and PARKS THERE.  Everything else (project s=2/s=1, the
// un-steered layer, switched-off no-write, add) runs here.
//
// GATES: project checks are tolerance gates vs the double-precision
// reference (the kernel's f32 fixed-order dot cannot be exact against a
// double sum -- the CUDA driver gates the same way: 1e-4 / 1e-3); the
// add check and the unchanged-R checks are bit-exact (add: the update is a
// plain per-element h + d, exact in f32; unchanged: the kernel returns
// early).
//
// PLATFORM NOTE: the add reference uses the pure-C exact f32 add below
// (no host f32 arithmetic: icpx host -O2 flushes f32 subnormal intermediates
// to zero in plain float +, measured in step 2.8; the fixture's O(1) values
// make a subnormal result practically impossible, but the exact add costs
// nothing and removes the argument).
#include "strata/kernels/cvec.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

// forward declarations: the port defines these in poc/sycl/kernels/cvec.cpp
// with a leading `sycl::queue&` (glue deviation; the cvec.hpp spellings are
// the CUDA ones).
namespace strata::kernels {
bool cvec_upload(sycl::queue& q, const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err);
bool cvec_replicate(sycl::queue& q, std::string& err);
void cvec_set_enabled(sycl::queue& q, bool on);
sycl::event cvec_apply(sycl::queue& q, float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
                       const float* inj, int64_t inj_ld, bool write);
}  // namespace strata::kernels

namespace {

int g_fail = 0;

// one IEEE f32 fma in pure C (bit-exact for all finite f32; verified
// 200,000/200,000 against gcc -mfma FMA3 in step 2.8).  The exact f32 add is
// fmaf(a, 1.0f, b): the x1.0 product is exact in the integer domain.
float fmaf_rn_ref(float a, float b, float c) {
    uint32_t ua, ub, uc;
    std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4); std::memcpy(&uc, &c, 4);
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
    __uint128_t prod = 0; int ep = 0; int sp = 1;
    if (ma && mb) {
        prod = ma * mb;
        ep = ea + eb;
        sp = ((ua >> 31) ^ (ub >> 31)) ? -1 : 1;
    }
    __int128 S = 0; int es = 0;
    bool zero_neg = false;
    int sc = (uc >> 31) ? -1 : 1;
    if (prod == 0 && mc == 0) {
        zero_neg = (uc >> 31) && ((ua >> 31) ^ (ub >> 31));
    } else if (prod == 0) { S = (__int128) mc * sc; es = ec; }
    else if (mc == 0) { S = (__int128) prod * sp; es = ep; }
    else if (ep > ec) {
        int d = ep - ec;
        if (d > 77) { S = (__int128) prod * sp; es = ep; }
        else { S = (((__int128) prod * sp) << d) + ((__int128) mc * sc); es = ec; }
    } else {
        int d = ec - ep;
        if (d > 77) { S = (__int128) mc * sc; es = ec; }
        else { S = ((__int128) prod * sp) + (((__int128) mc * sc) << d); es = ep; }
    }
    bool sgn = S < 0;
    __int128 M = sgn ? -S : S;
    float r;
    if (M == 0) {
        r = (sgn || zero_neg) ? -0.0f : 0.0f;
    } else {
        int kk = 0; { __int128 t = M; while (t >> 1) { t >>= 1; ++kk; } }
        if (es + kk >= -126 && es + kk <= 127) {
            uint32_t m24;
            int eRes;
            if (kk >= 23) {
                m24 = (uint32_t) (M >> (kk - 23));
                __uint128_t drop = (unsigned __int128) (M & (((__int128) 1 << (kk - 23)) - 1));
                __uint128_t half = (__uint128_t) 1 << (kk - 24);
                if (drop > half || (drop == half && (m24 & 1))) ++m24;
                eRes = es + kk + (m24 == 0x1000000u ? 1 : 0);
                m24 &= 0x7FFFFFu;
            } else {
                m24 = (uint32_t) (M << (23 - kk));
                eRes = es + kk;
            }
            r = __builtin_bit_cast(float, (uint32_t) (((uint32_t) (eRes + 127) << 23) | (m24 & 0x7FFFFFu) | (sgn ? 0x80000000u : 0u)));
        } else if (es + kk > 127) {
            r = sgn ? -__builtin_inff() : __builtin_inff();
        } else {
            int e149 = es + 149;
            __int128 U = 0;
            if (e149 >= 0) U = M << e149;
            else if (-e149 <= 127) {
                __int128 q = (__int128) 1 << (-e149);
                __int128 quo = M / q, rem = M % q;
                if (2 * rem > q) ++quo;
                else if (2 * rem == q && (quo & 1)) ++quo;
                U = quo;
            }
            r = (U == 0) ? (sgn ? -0.0f : 0.0f) : __builtin_bit_cast(float, (uint32_t) U);
        }
    }
    return r;
}
float fadd_rn_ref(float a, float b) { return fmaf_rn_ref(a, 1.0f, b); }

template <typename T>
T* dalloc(strata::sycl_compat::ctx& c, size_t n) { return c.device_alloc<T>(n); }
template <typename T>
void up(strata::sycl_compat::ctx& c, T* d, const std::vector<T>& h) { c.q.memcpy(d, h.data(), h.size() * sizeof(T)); }
template <typename T>
std::vector<T> down(strata::sycl_compat::ctx& c, const T* d, size_t n) {
    std::vector<T> h(n);
    c.q.memcpy(h.data(), d, n * sizeof(T));
    c.wait_and_throw();
    return h;
}
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc == 1 || (argc == 2 && std::strcmp(argv[1], "--selftest") == 0);
    if (!selftest) {
        std::fprintf(stderr, "usage: cvec_parity [--selftest]\n");
        return 2;
    }
    strata::sycl_compat::ctx ctxq;

    // the CUDA driver's fixture (its section 3, the fused-gr cross-check,
    // parks in B4 per F2)
    constexpr int64_t L = 48, N = 2560, HC = 4, LR = 320, T = 5, D = HC * N;
    constexpr int64_t kLayer = 7, kOff = 3;   // steered / not steered
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // a unit direction at kLayer with a norm-2 scaled one's s = 2; nothing at kOff
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    {
        double nrm = 0.0;
        std::vector<float> v((size_t) N);
        for (auto& x : v) { x = nd(rng); nrm += (double) x * x; }
        nrm = std::sqrt(nrm);
        for (int64_t j = 0; j < N; ++j) dir[(size_t) (kLayer * N + j)] = (float) (v[(size_t) j] / nrm);
        s[(size_t) kLayer] = 2.0f;
    }
    std::string err;
    if (!k::cvec_upload(ctxq.q, dir, s, /*project*/ 0, 4, 44, N, HC, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    check(k::cvec().covers(kLayer) && !k::cvec().covers(kOff) && !k::cvec().covers(45), "covers(): steered layers only");

    std::vector<float> R((size_t) (T * D)), bo((size_t) (T * N)), inj((size_t) (T * HC));
    for (auto& x : R) x = nd(rng) * 3.0f;
    for (auto& x : bo) x = nd(rng);
    for (auto& x : inj) x = nd(rng);
    float* dR = dalloc<float>(ctxq, R.size());
    float* dbo = dalloc<float>(ctxq, bo.size());
    float* dinj = dalloc<float>(ctxq, inj.size());
    up(ctxq, dbo, bo);
    up(ctxq, dinj, inj);

    // ---- 1. project, no pending write
    std::printf("project\n");
    up(ctxq, dR, R);
    k::cvec_apply(ctxq.q, dR, kLayer, T, D, nullptr, 0, nullptr, 0, false);
    ctxq.wait_and_throw();
    {
        const auto got = down(ctxq, dR, R.size());
        double worst = 0.0, worst_dot = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c) {
                const float* h = R.data() + t * D + c * N;
                const float* v = dir.data() + kLayer * N;
                double dot = 0.0;
                for (int64_t j = 0; j < N; ++j) dot += (double) h[j] * v[j];
                double after = 0.0;
                for (int64_t j = 0; j < N; ++j) {
                    const double want = h[j] - 2.0 * dot * v[j];
                    const double g = got[(size_t) (t * D + c * N + j)];
                    worst = std::fmax(worst, std::fabs(g - want));
                    after += g * v[j];
                }
                // s = 2 reflects the component: h.v -> -h.v
                worst_dot = std::fmax(worst_dot, std::fabs(after + dot));
            }
        std::printf("    max |err| %.3g, max |h'.v + h.v| %.3g\n", worst, worst_dot);
        check(worst < 1e-4, "project matches h - s (h.v) v (s = 2)");
        check(worst_dot < 1e-3, "the component along v is reflected (s = 2)");
    }
    // s = 1 removes it
    s[(size_t) kLayer] = 1.0f;
    if (!k::cvec_upload(ctxq.q, dir, s, 0, 4, 44, N, HC, err)) return 2;
    up(ctxq, dR, R);
    k::cvec_apply(ctxq.q, dR, kLayer, T, D, nullptr, 0, nullptr, 0, false);
    ctxq.wait_and_throw();
    {
        const auto got = down(ctxq, dR, R.size());
        double worst = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c) {
                double after = 0.0;
                for (int64_t j = 0; j < N; ++j) after += (double) got[(size_t) (t * D + c * N + j)] * dir[(size_t) (kLayer * N + j)];
                worst = std::fmax(worst, std::fabs(after));
            }
        std::printf("    max |h'.v| %.3g\n", worst);
        check(worst < 1e-3, "s = 1 projects the direction out");
    }

    // ---- 4. a layer without a direction
    up(ctxq, dR, R);
    k::cvec_apply(ctxq.q, dR, kOff, T, D, nullptr, 0, nullptr, 0, false);
    ctxq.wait_and_throw();
    check(std::memcmp(down(ctxq, dR, R.size()).data(), R.data(), R.size() * 4) == 0, "a layer without a direction is untouched");

    // ---- 3 (subset). switched off, no pending write: R unchanged (the
    // pending-write half of the CUDA driver's section 3 needs fused_gr_read,
    // which parks in B4 per F2)
    std::printf("switched off\n");
    k::cvec_set_enabled(ctxq.q, false);
    check(!k::cvec_enabled(), "cvec_enabled() follows the switch");
    up(ctxq, dR, R);
    k::cvec_apply(ctxq.q, dR, kLayer, T, D, nullptr, 0, nullptr, 0, false);
    ctxq.wait_and_throw();
    check(std::memcmp(down(ctxq, dR, R.size()).data(), R.data(), R.size() * 4) == 0, "off, no pending write: R unchanged");
    k::cvec_set_enabled(ctxq.q, true);

    // ---- 2. add
    std::printf("add\n");
    for (int64_t j = 0; j < N; ++j) dir[(size_t) (kLayer * N + j)] *= 0.1f;   // d = 0.1 v, s = 1
    s[(size_t) kLayer] = 1.0f;
    if (!k::cvec_upload(ctxq.q, dir, s, /*add*/ 1, 4, 44, N, HC, err)) return 2;
    up(ctxq, dR, R);
    k::cvec_apply(ctxq.q, dR, kLayer, T, D, nullptr, 0, nullptr, 0, false);
    ctxq.wait_and_throw();
    {
        const auto got = down(ctxq, dR, R.size());
        bool same = true;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c)
                for (int64_t j = 0; j < N; ++j) {
                    const size_t i = (size_t) (t * D + c * N + j);
                    const float want = fadd_rn_ref(R[i], dir[(size_t) (kLayer * N + j)]);
                    same = same && std::memcmp(&got[i], &want, 4) == 0;
                }
        check(same, "add is h + d in every stream (bitwise)");
    }

    // ---- the upload validation (mirrored host checks)
    std::printf("upload validation\n");
    {
        std::string e2;
        check(!k::cvec_upload(ctxq.q, dir, s, 1, 4, 44, 0, HC, e2) && !e2.empty(), "n_embd = 0 rejected");
        check(!k::cvec_upload(ctxq.q, dir, s, 1, 4, 44, 4097, HC, e2) && !e2.empty(), "n_embd > 4096 rejected");
        check(!k::cvec_upload(ctxq.q, dir, std::vector<float>(1), 1, 4, 44, N, HC, e2) && !e2.empty(), "bad table sizes rejected");
    }

    std::printf(g_fail ? "cvec_parity: %d FAILED\n" : "cvec_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
