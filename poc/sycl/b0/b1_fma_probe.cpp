// poc/sycl/b0/b1_fma_probe.cpp -- B1 scratch probe (not a ctest): does icpx
// 2026.1 contract std::fmaf(a, b, 0.0f) into a multiply on the Arc device,
// flipping signed zero (fmaf(s, -0.0f, 0.0f) must be +0.0f under IEEE)?
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>

int main() {
    sycl::queue q;
    float* d = sycl::malloc_device<float>(8, q);
    const float h[8] = {0.125f, -0.0f, 0.0f, 0.0f, 0.125f, -0.0f, 1.0f, 2.0f};
    q.memcpy(d, h, 8 * sizeof(float));
    q.submit([&](sycl::handler& hh) {
        hh.parallel_for(8, [=](sycl::id<1> i) {
            const size_t k = (size_t) i.get(0);
            if (k == 0) d[k] = std::fmaf(d[0], d[1], 0.0f);          // unguarded fma
            else if (k == 1) { volatile float t = std::fmaf(d[0], d[1], 0.0f); d[k] = t; }  // volatile-pinned
            else if (k == 2) d[k] = d[4] * d[5];                     // plain multiply
            else if (k == 3) d[k] = std::fmaf(d[6], d[7], d[1]);     // nonzero c control
        });
    }).wait();
    float r[8];
    q.memcpy(r, d, 8 * sizeof(float));
    const char* names[8] = {"dev fmaf(0.125,-0,0)", "dev pinned fmaf", "dev 0.125*-0", "dev fmaf(1,2,-0)"};
    for (int i = 0; i < 4; ++i) {
        uint32_t u;
        std::memcpy(&u, &r[i], 4);
        std::printf("%s = %08x\n", names[i], u);
    }
    float h0 = std::fmaf(0.125f, -0.0f, 0.0f);
    float h2 = 0.125f * -0.0f;
    float h3 = std::fmaf(1.0f, 2.0f, -0.0f);
    uint32_t u0, u2, u3;
    std::memcpy(&u0, &h0, 4);
    std::memcpy(&u2, &h2, 4);
    std::memcpy(&u3, &h3, 4);
    std::printf("host fmaf(0.125,-0,0) = %08x\n", u0);
    std::printf("host 0.125*-0          = %08x\n", u2);
    std::printf("host fmaf(1,2,-0)      = %08x\n", u3);
    return 0;
}
