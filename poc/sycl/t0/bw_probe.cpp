// poc/sycl/t0/bw_probe.cpp - cross-check for bw_spike.cpp: host wall-clock timing
// on 1 GiB arrays (4x larger than the spike, far beyond any cache), to verify whether
// the event-profiling numbers are trustworthy or under-reported.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <string>

using clk = std::chrono::high_resolution_clock;

int main() {
    constexpr size_t BYTES = (size_t) 1024 * 1024 * 1024;   // 1 GiB per array
    constexpr size_t ARR = BYTES / sizeof(float);

    sycl::device dev([](const sycl::device& d) {
        return (int)(d.is_gpu()
            && d.get_backend() == sycl::backend::ext_oneapi_level_zero
            && d.get_info<sycl::info::device::name>().find("e223") != std::string::npos);
    });
    sycl::context ctx(dev);
    sycl::queue q(dev, sycl::property::queue::in_order());

    float* h_a = sycl::malloc_host<float>(ARR, ctx);
    float* d_a = sycl::malloc_device<float>(ARR, q);
    float* d_b = sycl::malloc_device<float>(ARR, q);
    float* d_c = sycl::malloc_device<float>(ARR, q);
    for (size_t i = 0; i < ARR; ++i) h_a[i] = (float) i * 1e-10f + 1.0f;
    q.wait_and_throw();

    auto wall_ms = [&](auto&& submit) {
        auto t0 = clk::now();
        sycl::event e = submit(q);
        e.wait();
        auto t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();   // ms
    };

    // warmup
    (void) wall_ms([&](sycl::queue& qq) { return qq.memcpy(d_b, d_a, BYTES); });

    const double d2d_ms = wall_ms([&](sycl::queue& qq) { return qq.memcpy(d_b, d_a, BYTES); });
    std::printf("D2D  1GiB = %.1f GB/s   (wall %.3f ms)\n", 2.0 * BYTES / d2d_ms / 1e6, d2d_ms);

    auto triad = [&](sycl::queue& qq) {
        return qq.parallel_for(sycl::nd_range<1>(ARR / 4, (size_t) 256),
                               [=](sycl::nd_item<1> it) {
                                   const size_t base = it.get_global_id(0) * 4;
                                   const float* a = d_a + base;
                                   const float* b = d_b + base;
                                   float* c = d_c + base;
                                   for (int j = 0; j < 4; ++j) c[j] = a[j] + a[j + 1] + b[j];
                               });
    };
    (void) wall_ms(triad);
    const double tri_ms = wall_ms(triad);
    std::printf("triad 1GiB= %.1f GB/s   (wall %.3f ms)\n", 3.0 * BYTES / tri_ms / 1e6, tri_ms);

    // single-stream read: checksum of d_a
    float* sum_dev = sycl::malloc_device<float>(ARR / 4, q);
    auto readk = [&](sycl::queue& qq) {
        return qq.parallel_for(sycl::nd_range<1>(ARR / 4, (size_t) 256),
                               [=](sycl::nd_item<1> it) {
                                   const size_t base = it.get_global_id(0) * 4;
                                   float s = 0;
                                   for (int j = 0; j < 4; ++j) s += d_a[base + j];
                                   sum_dev[it.get_global_id(0)] = s;
                               });
    };
    (void) wall_ms(readk);
    const double rd_ms = wall_ms(readk);
    std::printf("read  1GiB= %.1f GB/s   (wall %.3f ms)\n", (double) BYTES / rd_ms / 1e6, rd_ms);
    q.wait_and_throw();
    return 0;
}
