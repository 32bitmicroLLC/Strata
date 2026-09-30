// poc/sycl/t0/bw_spike.cpp - T0.2: the Phase A gate.
//
// Measures on the Intel Arc GPU (selected by name+backend, never by index):
//   1. H2D  - host -> device (PCIe; reference only, never gated)
//   2. D2H  - device -> host  (PCIe; reference only)
//   3. D2D  - device -> device copy (1 read + 1 write stream; sanity floor)
//   4. TRIAD - c[i] = a[i] + a[i+1] + b[i] (2 reads + 1 write; THE gated number)
//
// Effective GB/s = total bytes read + written / kernel time.
// Gate: triad >= 60% of the rated bandwidth (default 512 GB/s for the
// Battlemage G31 class; pass the confirmed SKU number as argv[1]).
//
// Timing: SYCL events (command_execution_time, device-side), 1 warmup + 5
// timed runs, median reported. Exit 0 iff the gate passes.

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t BYTES = (size_t) 256 * 1024 * 1024;   // 256 MiB per array
constexpr size_t ARR = BYTES / sizeof(float);         // 67,108,864 floats

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double run_timed(sycl::queue& q, size_t total_bytes, auto&& submit, int warmup = 1, int runs = 5) {
    for (int i = 0; i < warmup; ++i) {
        sycl::event e = submit(q);
        e.wait();
    }
    std::vector<double> ms;
    ms.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        sycl::event e = submit(q);
        e.wait();
        const uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ms.push_back((end - start) / 1e6);   // ns -> ms
    }
    q.wait_and_throw();
    const double med = median(ms);
    return total_bytes / (med * 1e-3) / 1e9;   // effective GB/s
}

void print_run(const char* name, const std::vector<double>& gbps) {
    std::printf("%-5s = %.1f GB/s   (runs:", name, median(gbps));
    for (double g : gbps) std::printf(" %.1f", g);
    std::printf(")\n");
}

}  // namespace

int main(int argc, char** argv) {
    const double rated = argc > 1 ? std::atof(argv[1]) : 512.0;
    const double threshold = 0.6 * rated;

    // Name+backend selection: this box also has an OpenCL CPU and an OpenCL NEO
    // GPU; the default selector is not safe.
    sycl::device dev([](const sycl::device& d) {
        return (int)(d.is_gpu()
            && d.get_backend() == sycl::backend::ext_oneapi_level_zero
            && d.get_info<sycl::info::device::name>().find("e223") != std::string::npos);
    });
    std::ostringstream bss;
    bss << dev.get_backend();
    std::printf("device = %s (%s)\n", dev.get_info<sycl::info::device::name>().c_str(),
                bss.str().c_str());
    std::printf("arrays = 3 x %.0f MiB USM device, pattern-init\n", BYTES / 1048576.0);
    std::printf("gate   = triad >= %.1f GB/s (60%% of %.0f GB/s rated)\n\n", threshold, rated);

    sycl::context ctx(dev);
    sycl::queue q(dev, sycl::property_list{sycl::property::queue::in_order(),
                                          sycl::property::queue::enable_profiling()});

    float* h_a = sycl::malloc_host<float>(ARR, ctx);
    float* d_a = sycl::malloc_device<float>(ARR, q);
    float* d_b = sycl::malloc_device<float>(ARR, q);
    float* d_c = sycl::malloc_device<float>(ARR, q);
    for (size_t i = 0; i < ARR; ++i) h_a[i] = (float) i * 1e-10f + 1.0f;
    q.wait_and_throw();

    // 1+2. PCIe paths (reference only). run_timed includes its own warmup.
    {
        std::vector<double> h2d, d2h;
        auto h = [&](sycl::queue& qq) { return qq.memcpy(d_a, h_a, BYTES); };
        auto g = [&](sycl::queue& qq) { return qq.memcpy(h_a, d_a, BYTES); };
        for (int i = 0; i < 5; ++i) h2d.push_back(run_timed(q, BYTES, h));
        print_run("H2D", h2d);
        for (int i = 0; i < 5; ++i) d2h.push_back(run_timed(q, BYTES, g));
        print_run("D2H", d2h);
    }

    // 3. D2D copy (2 streams; sanity floor).
    {
        std::vector<double> v;
        auto k = [&](sycl::queue& qq) { return qq.memcpy(d_b, d_a, BYTES); };
        for (int i = 0; i < 5; ++i) v.push_back(run_timed(q, 2 * BYTES, k));
        print_run("D2D", v);
    }

    // 4. Triad, one work-item per 4 floats (128-bit accesses), work-group 64 and 256.
    double best = 0.0;
    for (int wg : {64, 256}) {
        std::vector<double> v;
        auto k = [&](sycl::queue& qq) {
            return qq.parallel_for(sycl::nd_range<1>(ARR / 4, (size_t) wg), [=](sycl::nd_item<1> it) {
                const size_t base = it.get_global_id(0) * 4;
                const float* a = d_a + base;
                const float* b = d_b + base;
                float* c = d_c + base;
                for (int j = 0; j < 4; ++j) c[j] = a[j] + a[j + 1] + b[j];
            });
        };
        for (int i = 0; i < 5; ++i) v.push_back(run_timed(q, 3 * BYTES, k));
        print_run("triad", v);
        std::printf("         (work-group %d)\n", wg);
        best = std::max(best, median(v));
    }

    const bool pass = best >= threshold;
    std::printf("\nGATE: triad best = %.1f GB/s vs %.1f GB/s threshold -> %s\n",
                best, threshold, pass ? "PASS" : "FAIL");
    (void) h_a;
    return pass ? 0 : 1;
}
