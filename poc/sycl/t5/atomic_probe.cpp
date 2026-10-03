// poc/sycl/t5/atomic_probe.cpp -- T5 probe P1 (plans/sycl-phase-a-t5.md §3):
// does sycl::atomic_ref with sycl::memory_scope_work_group work on
// local-accessor elements?  The sampler's penalty bitmap is built with
// atomicOr on shared memory in CUDA; this decides whether the SYCL port
// keeps the parallel build or falls back to a thread-0 serial OR
// (correctness-identical: the bitmap is fully published before any scan).
//
// 1024 threads (the sampler's work-group) OR deterministic bits into 256
// local words; threads 0..63 collide on word 0's bit 5.  The result is
// compared to the serial OR of the same operations on the host.
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const int N = 1024, WORDS = 256;
    std::vector<uint32_t> expect(WORDS, 0);
    for (int i = 0; i < N; ++i) {
        expect[(size_t) (i * 13) % WORDS] |= 1u << ((i * 7) % 32);
        if (i < 64) expect[0] |= 1u << 5;
    }

    uint32_t* dev = c.device_alloc<uint32_t>(WORDS);
    auto e = c.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> bits(sycl::range<1>(WORDS), h);
        h.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            if (i < WORDS) bits[i] = 0u;
            it.barrier();
            // DPC++ 2026.1 atomic_ref template: <T, DefaultOrder, DefaultScope>
            sycl::atomic_ref<uint32_t, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                bits[(size_t) (i * 13) % WORDS])
                .fetch_or(1u << ((i * 7) % 32));
            if (i < 64)
                sycl::atomic_ref<uint32_t, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                    bits[0])
                    .fetch_or(1u << 5);
            it.barrier();
            if (i < WORDS) dev[i] = bits[i];
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got(WORDS);
    c.q.memcpy(got.data(), dev, (size_t) WORDS * sizeof(uint32_t));
    c.wait_and_throw();
    c.free_device(dev);

    int bad = 0;
    for (int w = 0; w < WORDS; ++w)
        if (got[(size_t) w] != expect[(size_t) w]) {
            if (bad < 8)
                std::printf("    word %d: expect 0x%08x got 0x%08x\n", w, expect[(size_t) w], got[(size_t) w]);
            ++bad;
        }
    std::printf("t5_atomic_probe: %s (%d of %d words differ)\n", bad ? "*** FAIL ***" : "PASS", bad, WORDS);
    return bad ? 1 : 0;
}
