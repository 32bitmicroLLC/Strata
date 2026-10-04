// poc/sycl/b0/b0_shfl_probe.cpp -- B0 probe: the 32-wide shuffle
// primitives (B0 §1.3). Correctness of shfl_xor/down/up/sync, ballot and
// the __syncwarp spin barrier vs host references of the documented CUDA
// lane semantics, over all 1,024 work-group items (32 groups of 32 lanes),
// full-width and width-16 sub-warps.
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
inline uint32_t laneval(int tid) {  // deterministic per-lane value
    return (uint32_t) tid * 2654435761u ^ 0x9e3779b9u;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const int N = 1024;
    // One output word per item per phase: 6 xor phases + 3 down + 3 up +
    // 5 sync/ballot + 1 syncwarp marker.
    constexpr int PHASES = 18;
    std::vector<uint32_t> in(N);
    for (int i = 0; i < N; ++i) in[i] = laneval(i);
    uint32_t* din = c.device_alloc<uint32_t>(N);
    uint32_t* dout = c.device_alloc<uint32_t>((size_t) N * PHASES);
    c.q.memcpy(din, in.data(), N * 4);
    auto e = c.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        sycl::local_accessor<uint32_t, 1> val(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
            const uint32_t v = din[tid];
            const int o = tid * PHASES;
            // xor, all masks, width 32
            dout[o + 0] = __shfl_xor_sync(0xffffffffu, v, 1);
            dout[o + 1] = __shfl_xor_sync(0xffffffffu, v, 2);
            dout[o + 2] = __shfl_xor_sync(0xffffffffu, v, 4);
            dout[o + 3] = __shfl_xor_sync(0xffffffffu, v, 8);
            dout[o + 4] = __shfl_xor_sync(0xffffffffu, v, 16);
            dout[o + 5] = __shfl_down_sync(0xffffffffu, v, 1);
            dout[o + 6] = __shfl_down_sync(0xffffffffu, v, 2);
            dout[o + 7] = __shfl_down_sync(0xffffffffu, v, 5);
            dout[o + 8] = __shfl_up_sync(0xffffffffu, v, 1);
            dout[o + 9] = __shfl_up_sync(0xffffffffu, v, 2);
            dout[o + 10] = __shfl_up_sync(0xffffffffu, v, 5);
            dout[o + 11] = __shfl_sync(0xffffffffu, v, 0);
            dout[o + 12] = __shfl_sync(0xffffffffu, v, 5);
            dout[o + 13] = __shfl_sync(0xffffffffu, v, 31);
            dout[o + 14] = __shfl_xor_sync(0xffffffffu, v, 8, 16);  // width-16 sub-warps
            dout[o + 15] = __shfl_down_sync(0xffffffffu, v, 1, 16);
            // ballot: (tid % 7 == 0) | (lane == 3)
            dout[o + 16] = __ballot_sync(0xffffffffu, (tid % 7 == 0) || ((tid & 31) == 3));
            // syncwarp stress: write round-tagged values, sync, lane 0 audits
            const int round = 1;
            val[tid] = (uint32_t) tid + (uint32_t) round * N;
            __syncwarp();
            uint32_t audit = 0;
            if ((tid & 31) == 0) {
                const int base = tid & ~31;
                bool ok = true;
                for (int l = 0; l < 32; ++l)
                    if (val[base + l] != (uint32_t) (base + l) + (uint32_t) round * N) ok = false;
                audit = ok ? 1u : 0u;
            }
            dout[o + 17] = audit;
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got((size_t) N * PHASES);
    c.q.memcpy(got.data(), dout, (size_t) N * PHASES * 4);
    c.wait_and_throw();

    size_t bad = 0;
    auto chk = [&](int tid, int p, uint32_t want, const char* what) {
        if (got[(size_t) tid * PHASES + p] != want) {
            if (bad < 12)
                std::printf("    %s tid %d: want 0x%08x got 0x%08x\n", what, tid, want,
                            got[(size_t) tid * PHASES + p]);
            ++bad;
        }
    };
    for (int tid = 0; tid < N; ++tid) {
        const int base = tid & ~31, lane = tid & 31;
        for (int m : {1, 2, 4, 8, 16})
            chk(tid, m == 1 ? 0 : m == 2 ? 1 : m == 4 ? 2 : m == 8 ? 3 : 4, laneval(base + (lane ^ m)), "xor");
        for (int d : {1, 2, 5}) {
            const int i = d == 1 ? 5 : d == 2 ? 6 : 7;
            chk(tid, i, lane + d < 32 ? laneval(base + lane + d) : laneval(tid), "down");
        }
        for (int d : {1, 2, 5}) {
            const int i = d == 1 ? 8 : d == 2 ? 9 : 10;
            chk(tid, i, lane - d >= 0 ? laneval(base + lane - d) : laneval(tid), "up");
        }
        chk(tid, 11, laneval(base), "sync0");
        chk(tid, 12, laneval(base + 5), "sync5");
        chk(tid, 13, laneval(base + 31), "sync31");
        // width 16: base16 = tid & ~15
        const int base16 = tid & ~15, lane16 = tid & 15;
        chk(tid, 14, laneval(base16 + (lane16 ^ 8)), "xor16");
        chk(tid, 15, lane16 + 1 < 16 ? laneval(base16 + lane16 + 1) : laneval(tid), "down16");
        // ballot reference
        uint32_t ref = 0;
        for (int l = 0; l < 32; ++l)
            if ((base + l) % 7 == 0 || l == 3) ref |= 1u << l;
        chk(tid, 16, ref, "ballot");
        chk(tid, 17, (lane == 0) ? 1u : 0u, "syncwarp");
    }
    std::printf("b0_shfl_probe: %s (%zu of %zu checks differ)\n", bad ? "*** FAIL ***" : "PASS", bad,
                (size_t) N * PHASES);
    c.free_device(din);
    c.free_device(dout);
    return bad ? 1 : 0;
}
