// scratch probe: host-vs-device float division agreement (disposable, not a ctest)
#include "sycl_compat/test_ctx.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    strata::sycl_compat::ctx ctx;
    sycl::queue& q = ctx.q;
    std::vector<float> h(1 << 20);
    for (int i = 0; i < (1 << 20); ++i) h[(size_t) i] = 1.0f + 1e-4f * ((i % 100000) - 50000) * 0.001f;
    h[0] = 127.0f;
    float* din = ctx.device_alloc<float>(h.size());
    float* d = ctx.device_alloc<float>(h.size());
    q.memcpy(din, h.data(), h.size() * 4);
    { float* a = din; float* b = d; size_t nn = h.size();
      q.submit([&](sycl::handler& hdl) {
        hdl.parallel_for(sycl::range<1>(nn), [=](sycl::id<1> i) {
            b[(size_t) i] = 1.0f / a[i];
        });
      }); }
    ctx.wait_and_throw();
    std::vector<float> rb(h.size());
    q.memcpy(rb.data(), d, h.size() * 4);
    int bad = 0; double worst = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        float hostd = 1.0f / h[i];
        if (*(int*) &hostd != *(int*) &rb[i]) {
            ++bad;
            double rel = (double) std::fabs(hostd - rb[i]) / (std::fabs((double) hostd) + 1e-30);
            if (rel > worst) worst = rel;
            if (bad <= 3) std::printf("i=%zu host %a got %a\n", i, (double) hostd, (double) rb[i]);
        }
    }
    std::printf("float div host-vs-device: %d of %zu differ, worst rel %.3e\n", bad, h.size(), worst);
    // also check the reciprocal-of-reciprocal pattern 1.0f / iscale with negative iscale
    std::vector<float> r2(h.size());
    { float* a = din; float* b = d; size_t nn = h.size();
      q.submit([&](sycl::handler& hdl) {
        hdl.parallel_for(sycl::range<1>(nn), [=](sycl::id<1> i) {
            const float isc = -127.0f / a[i];
            b[(size_t) i] = 1.0f / isc;
        });
      }); }
    ctx.wait_and_throw();
    q.memcpy(r2.data(), d, h.size() * 4);
    int bad2 = 0; double worst2 = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        const float isc = -127.0f / h[i];
        const float ref = 1.0f / isc;
        if (*(int*) &ref != *(int*) &r2[i]) {
            ++bad2;
            double rel = (double) std::fabs(ref - r2[i]) / (std::fabs((double) ref) + 1e-30);
            if (rel > worst2) worst2 = rel;
            if (bad2 <= 3) std::printf("iscase i=%zu host %a got %a\n", i, (double) ref, (double) r2[i]);
        }
    }
    std::printf("isc/1-over-isc host-vs-device: %d of %zu differ, worst rel %.3e\n", bad2, h.size(), worst2);
}
