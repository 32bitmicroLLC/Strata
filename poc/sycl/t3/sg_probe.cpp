// poc/sycl/t3/sg_probe.cpp -- T3 step 1 (plans/sycl-phase-a-t3.md): the probe that
// gates the router_top10 port design.
//
// The plan's Option A (request 32-wide sub-groups + sycl::shfl_down warp trees)
// turned out to be unbuildable on icpx 2026.1 -- the toolchain has NO subgroup
// shuffle or reduce API (verified: no sycl::shfl*, no sub_group::reduce/shfl_down,
// no sycl::property to request a sub-group width, -sycl-std accepts only 2020),
// and the kernel-struct-with-local_accessor-members pattern does not compile
// (local_accessor has no host-side constructor from a range).  The port
// therefore uses:
//
//   * sycl::local_accessor constructed in the submit handler and captured BY
//     VALUE into the kernel lambda;
//   * barrier-based trees over local memory in place of the CUDA warp-shuffle
//     trees -- bit-exact, because fmaxf is exact/order-independent and the
//     (value, index) pair compare is a total order, so ANY tree returns the
//     same winner as CUDA's two-stage shuffle reduction.
//
// This program verifies those pieces in isolation:
//   check 1  the default sub-group width (informational now -- no design choice
//            depends on it, which is the point of the fallback design);
//   check 2  the exact local-memory pair-tree reduction the port uses: a scan
//            with a divergent `continue` (the router's `if (s_taken[e])
//            continue;`) plus the router's tie-break compare, known answer;
//   check 3  barrier + local_accessor sum.
//
// Exit 0 iff checks 2 and 3 pass.
#include "sycl_compat/test_ctx.hpp"

#include <cmath>
#include <cstdio>

namespace {

using strata::sycl_compat::ctx;

constexpr int N = 512;  // one work-group, the same shape the router runs (n_expert = 512)

// check 1: the sub-group width the runtime gives for a default launch.
int probe_width(sycl::queue& q) {
    int* d = sycl::malloc_device<int>(1, q);
    q.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
        if (it.get_local_id(0) == 0) d[0] = (int) it.get_group().get_local_range().get(0);
    });
    q.wait_and_throw();
    int h = -1;
    q.memcpy(&h, d, sizeof(int));
    q.wait_and_throw();
    sycl::free(d, q);
    return h;
}

}  // namespace

int main() {
    ctx c;
    int bad = 0;

    // -- check 1: the width, printed (the A/B decision no longer depends on it)
    const int w = probe_width(c.q);
    std::printf("  check 1: sub-group width  default %d  (informational: icpx 2026.1 exposes no "
                "subgroup shuffle/reduce API, so the port does not depend on it)\n", w);

    // -- check 2: the port's exact reduction.  One candidate per lane; odd lanes
    // skip the scan (the router's taken-skip) but still take every tree step.
    // Winner over all 512 lanes: the largest even e, i.e. 510.
    float* d_val = sycl::malloc_device<float>(1, c.q);
    int* d_idx = sycl::malloc_device<int>(1, c.q);
    c.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> s_red_f(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> s_red_i(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int nt = (int) it.get_local_range(0);
            float bv = -INFINITY;
            int bi = N;  // a sentinel that loses to every real index
            for (int e = tid; e < N; e += nt) {
                if (e & 1) continue;
                const float pe = (float) e;
                if (pe > bv) { bv = pe; bi = e; }
            }
            s_red_f[tid] = bv;
            s_red_i[tid] = bi;
            it.barrier();
            for (int stride = nt >> 1; stride > 0; stride >>= 1) {
                const float ov = (tid < stride) ? s_red_f[tid + stride] : -INFINITY;
                const int oi = (tid < stride) ? s_red_i[tid + stride] : N;
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                s_red_f[tid] = bv;
                s_red_i[tid] = bi;
                it.barrier();
            }
            if (tid == 0) {
                d_val[0] = bv;
                d_idx[0] = bi;
            }
        });
    });
    c.wait_and_throw();
    float h_val = 0.0f;
    int h_idx = -1;
    c.q.memcpy(&h_val, d_val, sizeof(float));
    c.q.memcpy(&h_idx, d_idx, sizeof(int));
    c.wait_and_throw();
    const bool ok2 = h_val == 510.0f && h_idx == 510;
    std::printf("  check 2: local-memory pair tree  %s  (got (%.0f, %d), want (510, 510))\n", ok2 ? "PASS"
                                                                                                 : "*** FAIL ***",
                h_val, h_idx);
    if (!ok2) ++bad;

    // -- check 3: barrier + local_accessor sum: 1 + ... + 512 = 131328 (exact in double)
    double* d_s = sycl::malloc_device<double>(1, c.q);
    c.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(N, N), [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            buf[lid] = (float) (lid + 1);
            it.barrier();
            if (lid == 0) {
                double s = 0.0;
                for (int i = 0; i < N; ++i) s += buf[i];
                d_s[0] = s;
            }
        });
    });
    c.wait_and_throw();
    double h_s = -1.0;
    c.q.memcpy(&h_s, d_s, sizeof(double));
    c.wait_and_throw();
    const double want_s = (double) N * (N + 1) / 2;
    const bool ok3 = h_s == want_s;
    std::printf("  check 3: barrier + local_accessor sum  %s  (got %.1f, want %.1f)\n",
                ok3 ? "PASS" : "*** FAIL ***", h_s, want_s);
    if (!ok3) ++bad;

    sycl::free(d_val, c.q);
    sycl::free(d_idx, c.q);
    sycl::free(d_s, c.q);

    std::printf("t3_sg_probe: %d failure(s)\n", bad);
    return bad ? 1 : 0;
}
