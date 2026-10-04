// poc/sycl/b0/b0_shfl_bench.cpp -- B0 probe: the MEASURED cost of the
// local-memory shuffle primitives (B0 §1.3). The accept/vendor-extension
// decision must rest on numbers, not hope: this times three workloads on
// the 1,024 work-group
//
//   A: an attention-shaped shuffle burst -- one 5-round butterfly reduce
//      (shfl_xor masks 1,2,4,8,16) + a 5-round broadcast (shfl_sync),
//      10 shuffles per unit, x 64 units
//   B: __syncwarp (the 32-wide spin barrier), 1000 units
//   C: it.barrier() (the 1,024-wide work-group barrier), 1000 units
//
// 5 warmup + 20 timed iterations, median reported. B vs C answers "is the
// spin barrier cheaper than the full barrier it replaces"; A/10 gives
// ns/shuffle.
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
long median(std::vector<long> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const int N = 1024, UNITS = 64, BUNITS = 1000;
    float* dres = c.device_alloc<float>(N);
    auto run = [&](int which) -> sycl::event {
        return c.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
            sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
            sycl::local_accessor<float, 1> acc(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
                const int tid = (int) it.get_local_id(0);
                strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
                acc[tid] = (float) tid;
                if (which == 0) {
                    for (int u = 0; u < UNITS; ++u) {
                        for (int m = 1; m < 32; m <<= 1)
                            acc[tid] += shfl32.shfl_xor_sync(0xffffffffu, acc[tid], m);
                        for (int m = 16; m >= 1; m >>= 1)
                            acc[tid] += shfl32.shfl_sync(0xffffffffu, acc[tid], m);
                    }
                } else if (which == 1) {
                    for (int u = 0; u < BUNITS; ++u) shfl32.syncwarp();
                    acc[tid] += (float) BUNITS;
                } else {
                    for (int u = 0; u < BUNITS; ++u) it.barrier();
                    acc[tid] += (float) BUNITS;
                }
                if (tid == 0) dres[0] = acc[tid];  // keep the compiler honest
            });
        });
    };
    for (int w = 0; w < 3; ++w) {
        for (int i = 0; i < 5; ++i) run(w).wait_and_throw();  // warmup
        std::vector<long> ns;
        for (int i = 0; i < 20; ++i) {
            sycl::event e = run(w);
            e.wait_and_throw();
            // 2026.1 dropped the SYCL-2020 command_exec_time; the 2023-style
            // event_profiling keys are what the runtime ships (T0 pattern).
            const uint64_t s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            ns.push_back((long) (t - s));
        }
        const long med = median(ns);
        if (w == 0)
            std::printf("b0_shfl_bench: A (10 shfl x %d units, %d threads): median %ld us  ->  %.1f ns/shuffle\n",
                        UNITS, N, med / 1000, (double) med / (double) (UNITS * 10));
        else if (w == 1)
            std::printf("b0_shfl_bench: B (__syncwarp spin barrier, %d units): median %ld us  ->  %.2f ns/sync\n",
                        BUNITS, med / 1000, (double) med / (double) BUNITS);
        else
            std::printf("b0_shfl_bench: C (it.barrier, 1024-wide, %d units): median %ld us  ->  %.2f ns/barrier\n",
                        BUNITS, med / 1000, (double) med / (double) BUNITS);
    }
    std::printf("b0_shfl_bench: PASS (numbers above are the decision input; the accept /\n"
                "                vendor-extension call is written in the report against them)\n");
    c.free_device(dres);
    return 0;
}
