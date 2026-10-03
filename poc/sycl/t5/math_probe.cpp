// poc/sycl/t5/math_probe.cpp -- T5 probe P2 (plans/sycl-phase-a-t5.md §3):
// the sampler tail compares DEVICE double math against the HOST reference
// (std::exp / std::log in the mirrored parity driver).  This probe measures
// whether the two agree bit for bit on the arguments the tail actually
// generates.
//
// Argument structure: `sel_logit[i]` and `mx` are fp32, so the kernel's
// double argument is `(double) x - (double) mx` for fp32 x, mx - always
// <= 0.  We cover: (a) a structured grid of mx (half steps, -744..0)
// against a distance ladder, (b) dense ulp-spaced near pairs (the
// widest-mantissa differences), (c) 300k pseudo-random fp32 pairs.  For
// min_p's threshold, device logf(v) vs host (float) std::log(v) on a
// 100k log-uniform grid in (0, 1] plus the fixtures' exact values.
#include <sycl_compat/test_ctx.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    // ---- the exp argument pairs (x, mx), all fp32, x <= mx ----
    std::vector<float> xs, mxs;
    for (double mx = -744.0; mx <= 0.0 + 1e-9; mx += 0.5) {
        const double d0 = mx;
        const double ds[] = {0.0, 9.5367431640625e-7, 0.000244140625, 0.00390625, 0.015625, 0.0625, 0.25, 1.0,
                             8.0, 64.0, 512.0, 4096.0, 32768.0, 100000.0};
        for (double d : ds) {
            mxs.push_back((float) d0);
            xs.push_back((float) (d0 - d));
        }
    }
    // ulp-spaced near pairs: the differences with the widest mantissas
    const double dense_mx[] = {-0.5, -1.5, -3.25, -7.75, -16.25, -64.125};
    for (double d0 : dense_mx)
        for (int j = 1; j <= 8; ++j) {
            mxs.push_back((float) d0);
            xs.push_back((float) (d0 - j * 2.384185791015625e-7));
        }
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.0f, 10.0f);
    for (int i = 0; i < 300000; ++i) {
        float a = g(rng), b = g(rng);
        mxs.push_back(a < b ? a : b);
        xs.push_back(a < b ? b : a);
    }
    const int NE = (int) xs.size();

    float* dx = c.device_alloc<float>((size_t) NE);
    float* dm = c.device_alloc<float>((size_t) NE);
    double* de = c.device_alloc<double>((size_t) NE);
    c.q.memcpy(dx, xs.data(), (size_t) NE * sizeof(float));
    c.q.memcpy(dm, mxs.data(), (size_t) NE * sizeof(float));
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) NE), [dx, dm, de](sycl::id<1> id) {
            de[id[0]] = exp((double) dx[id[0]] - (double) dm[id[0]]);
        });
    });
    e.wait_and_throw();

    std::vector<double> got((size_t) NE);
    c.q.memcpy(got.data(), de, (size_t) NE * sizeof(double));
    c.wait_and_throw();

    int bad = 0;
    for (int i = 0; i < NE; ++i) {
        const double want = std::exp((double) xs[(size_t) i] - (double) mxs[(size_t) i]);
        if (got[(size_t) i] != want) {
            if (bad < 8)
                std::printf("    i %d: x %g mx %g: device %.17g host %.17g\n", i, xs[(size_t) i], mxs[(size_t) i],
                            got[(size_t) i], want);
            ++bad;
        }
    }
    const bool exp_ok = bad == 0;
    std::printf("t5_math_probe: device exp(double) vs host std::exp(double): %s (%d of %d differ)\n",
                exp_ok ? "PASS" : "*** FAIL ***", bad, NE);
    c.free_device(dx);
    c.free_device(dm);
    c.free_device(de);

    // ---- logf: device vs host (float) std::log on a log-uniform (0,1] grid ----
    const int NL = 100000;
    std::vector<float> vs((size_t) NL);
    for (int j = 0; j < NL; ++j) vs[(size_t) j] = std::exp2f(-j * 30.0 / (double) NL);
    float* dv = c.device_alloc<float>((size_t) NL);
    float* dl = c.device_alloc<float>((size_t) NL);
    c.q.memcpy(dv, vs.data(), (size_t) NL * sizeof(float));
    e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) NL), [dv, dl](sycl::id<1> id) { dl[id[0]] = logf(dv[id[0]]); });
    });
    e.wait_and_throw();
    std::vector<float> lg((size_t) NL);
    c.q.memcpy(lg.data(), dl, (size_t) NL * sizeof(float));
    c.wait_and_throw();
    int lbad = 0;
    for (int j = 0; j < NL; ++j) {
        const float want = (float) std::log(vs[(size_t) j]);
        if (lg[(size_t) j] != want) {
            if (lbad < 8) std::printf("    v %g: device %a host %a\n", vs[(size_t) j], lg[(size_t) j], want);
            ++lbad;
        }
    }
    // the fixtures' exact min_p values
    for (float v : {0.05f, 0.3f, 0.5f, 0.9f}) {
        const float want = (float) std::log(v);
        float gotv;
        float* d1 = c.device_alloc<float>(1);
        c.q.memcpy(d1, &v, sizeof(float));
        auto e2 = c.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [d1](sycl::id<1>) { d1[0] = logf(d1[0]); });
        });
        e2.wait_and_throw();
        c.q.memcpy(&gotv, d1, sizeof(float));
        c.wait_and_throw();
        c.free_device(d1);
        if (gotv != want) {
            std::printf("    min_p %g: device %a host %a\n", (double) v, gotv, want);
            ++lbad;
        }
    }
    const bool log_ok = lbad == 0;
    std::printf("t5_math_probe: device logf vs host (float) std::log: %s (%d of %d + 4 fixture values differ)\n",
                log_ok ? "PASS" : "*** FAIL ***", lbad, NL);
    c.free_device(dv);
    c.free_device(dl);

    return exp_ok && log_ok ? 0 : 1;
}
