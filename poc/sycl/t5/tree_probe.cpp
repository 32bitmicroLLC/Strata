// poc/sycl/t5/tree_probe.cpp -- T5 probe P3 (plans/sycl-phase-a-t5.md §3):
// the (value, index) reduction the sampler rounds run on CUDA's 32-wide warp
// shuffles becomes a barrier-based tree over one work-group of up to 1024
// threads.  `take_first` is a total order (value, then lower index wins),
// so folding the 1024 per-thread partials in ANY order must give the same
// winner as the serial host scan - this verifies that on 1024 threads at
// 10 levels, with ties, NaNs, and -inf sentinels in the data.
#include <sycl_compat/test_ctx.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    const int N = 1024, TRIALS = 64;
    float* dv = c.device_alloc<float>((size_t) N);
    float* d_bv = c.device_alloc<float>(1);
    int* d_bi = c.device_alloc<int>(1);

    std::mt19937 rng(11);
    std::uniform_int_distribution<int> pick(0, 5);
    const float pool[] = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, 0.5f};   // small pool: lots of ties
    int bad_trials = 0;
    for (int tr = 0; tr < TRIALS; ++tr) {
        std::vector<float> v((size_t) N);
        for (int i = 0; i < N; ++i) v[(size_t) i] = pool[(size_t) pick(rng)];
        v[(size_t) (tr * 37) % N] = -INFINITY;                          // sentinels
        v[(size_t) (tr * 53 + 11) % N] = std::numeric_limits<float>::quiet_NaN();   // NaNs
        v[(size_t) (tr * 29 + 5) % N] = std::numeric_limits<float>::quiet_NaN();
        c.q.memcpy(dv, v.data(), (size_t) N * sizeof(float));

        auto e = c.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> rv(sycl::range<1>(N), h);
            sycl::local_accessor<int, 1> ri(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(N, N), [dv, d_bv, d_bi, rv, ri](sycl::nd_item<1> it) {
                const int tid = (int) it.get_local_id(0);
                // Step A (the CUDA per-thread scan): strict > from -inf, so NaN
                // never enters a pair - the fold below only sees real values or
                // the (-inf, N) sentinel.
                float bv = -INFINITY;
                int best = N;
                const float s = dv[tid];
                if (s > bv) {
                    bv = s;
                    best = tid;
                }
                rv[tid] = bv;
                ri[tid] = best;
                it.barrier();
                // Step B: ascending-window decimation tournament.  Before level
                // w, position m*w holds the winner of [m*w, (m+1)*w); the level
                // folds each pair of adjacent windows into position m*2w (done
                // by the thread at that position).
                for (int w = 1; w <= N / 2; w <<= 1) {
                    if ((tid & (2 * w - 1)) == 0) {
                        const float ov = rv[tid + w];
                        const int oi = ri[tid + w];
                        if (ov > rv[tid] || (ov == rv[tid] && oi < ri[tid])) {
                            rv[tid] = ov;
                            ri[tid] = oi;
                        }
                    }
                    it.barrier();
                }
                if (tid == 0) {
                    d_bv[0] = rv[0];
                    d_bi[0] = ri[0];
                }
            });
        });
        e.wait_and_throw();
        float bv;
        int bi;
        c.q.memcpy(&bv, d_bv, sizeof(float));
        c.q.memcpy(&bi, d_bi, sizeof(int));
        c.wait_and_throw();

        // serial scan, ascending index, strict > (keeps the first of the max value)
        float sv = -INFINITY;
        int si = N;
        for (int i = 0; i < N; ++i) {
            const float s = v[(size_t) i];
            if (s > sv) {
                sv = s;
                si = i;
            }
        }
        if (bv != sv || bi != si) {
            if (bad_trials < 8)
                std::printf("    trial %d: tree (%g, %d) serial (%g, %d)\n", tr, (double) bv, bi, (double) sv, si);
            ++bad_trials;
        }
    }
    c.free_device(dv);
    c.free_device(d_bv);
    c.free_device(d_bi);
    std::printf("t5_tree_probe: %s (%d of %d trials differ)\n", bad_trials ? "*** FAIL ***" : "PASS", bad_trials,
                TRIALS);
    return bad_trials ? 1 : 0;
}
