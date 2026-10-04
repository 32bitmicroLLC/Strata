// scratch: wide-range float division host-vs-device (disposable)
#include "sycl_compat/test_ctx.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
    (void) argc; (void) argv;
    strata::sycl_compat::ctx ctx;
    sycl::queue& q = ctx.q;
    const size_t N = 1u << 22;
    // span 2^-100 .. 2^100, log-uniform, plus the exact failing operand
    std::vector<float> h(N);
    for (size_t i = 0; i < N; ++i) {
        double e = (double) (i % 201) - 100.0;
        double m = 1.0 + (double) ((i * 2654435761u) >> 40) / 16777216.0;
        h[(size_t) i] = (float) (m * std::ldexp(1.0, (int) e));
    }
    h[0] = -42.94570541f;   // the exact failing iscale from block 2
    float* din = ctx.device_alloc<float>(N);
    float* d = ctx.device_alloc<float>(N);
    q.memcpy(din, h.data(), N * 4);
    { float* a = din; float* b = d; size_t nn = N;
      q.submit([&](sycl::handler& hdl) {
        hdl.parallel_for(sycl::range<1>(nn), [=](sycl::id<1> i) {
            b[(size_t) i] = 1.0f / a[i];
        });
      }); }
    ctx.wait_and_throw();
    std::vector<float> rb(N);
    q.memcpy(rb.data(), d, N * 4);
    int bad = 0; int ulp1 = 0, ulp2 = 0; double worst = 0;
    for (size_t i = 0; i < N; ++i) {
        float hostd = 1.0f / h[i];
        if (*(int*) &hostd != *(int*) &rb[i]) {
            ++bad;
            double rel = (double) std::fabs(hostd - rb[i]) / (std::fabs((double) hostd) + 1e-30);
            if (rel > worst) worst = rel;
            uint32_t a1 = *(uint32_t*) &hostd, a2 = *(uint32_t*) &rb[i];
            uint32_t lo = a1 < a2 ? a1 : a2, hi = a1 < a2 ? a2 : a1;
            if (hi - lo == 1) ++ulp1; else ++ulp2;
            if (bad <= 3) std::printf("i=%zu host %a got %a\n", i, (double) hostd, (double) rb[i]);
        }
    }
    std::printf("1.0f/x over 2^-100..2^100: %d of %zu differ (1ulp %d, >1ulp %d, worst rel %.3e)\n",
                bad, N, ulp1, ulp2, worst);
}
