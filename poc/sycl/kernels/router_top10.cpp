// poc/sycl/kernels/router_top10.cpp -- T3: the first shared-memory kernel port
// (see plans/sycl-phase-a-t3.md).  CUDA source: src/kernels/cuda/router_top10.cu
// @ 49da5df.
//
// Contract difference from the CUDA wrapper, on purpose and for Phase B:
// CUDA's router_top10() synchronises before returning and exit(1)s on a bad
// argument or a launch error; submit_router_top10() returns the event, leaves
// synchronisation to the caller (drivers call wait_and_throw()), and throws
// std::runtime_error instead of exiting -- a PoC library should not kill the
// process.  The Phase B engine wrapper absorbs both differences.
//
// DESIGN NOTE -- the plan's Option A was found unbuildable on icpx 2026.1 by
// the t3/sg_probe.cpp step (see plans/sycl-phase-a-report.md):
//   * the toolchain has NO subgroup shuffle or reduce API (no sycl::shfl*,
//     no sub_group::reduce/shfl_down, no property to request a sub-group
//     width);
//   * the kernel-struct-with-local_accessor-members pattern does not compile
//     (local_accessor has no host-side constructor from a range).
// So the port uses handler-constructed sycl::local_accessor captured BY VALUE
// into the kernel lambda, and replaces every CUDA warp-shuffle tree + shared
// 2nd stage with a barrier-based tree over local memory.  That is bit-exact:
// fmaxf is exact and order-independent, and the (value, index) pair compare
// is a total order, so ANY reduction tree returns the same winner as CUDA's
// two-stage shuffle reduction.
//
// The arithmetic lines are MIRRORED VERBATIM from the CUDA file (marker
// blocks, checked by check_mirrors.sh); the shuffle/barrier/shared lines are
// glue, each commented with the CUDA line it replaces.  The kernel body keeps
// the CUDA file's 4-space indentation (deeper where the mirror block demands
// it) so the mirrored blocks stay column-identical.
#include "sycl_compat/launch.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace strata::kernels {
namespace {

// SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:65:65 @ 49da5df
constexpr int RT_MAX_THREADS = 512;
// SYCL-MIRROR-END

}  // namespace

// Route `n_tokens` tokens: `logits` is (n_tokens, n_expert) f32; `ids`
// (n_tokens, k) int32 and `weights` (n_tokens, k) f32, all USM device
// pointers.  One work-group of `threads` per token, mirroring the CUDA
// grid/block shape (n_tokens is 1 in decode; the batch case works without a
// second code path, as on CUDA).
sycl::event submit_router_top10(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k,
                               int* ids, float* weights) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0)
        return q.submit([](sycl::handler&) {});  // mirror of the CUDA early return (line 194)
    if (k > 64)
        throw std::runtime_error("router_top10: k " + std::to_string(k) + " exceeds the kernel's 64");
    if (n_expert > RT_MAX_THREADS * 64)
        throw std::runtime_error("router_top10: n_expert " + std::to_string(n_expert) + " is past the kernel's " +
                                 std::to_string(RT_MAX_THREADS * 64));
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:203:204 @ 49da5df
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions
    // SYCL-MIRROR-END
    // New (not in the CUDA file): the local-memory trees need a power-of-two
    // work-group (the CUDA warp shuffles tolerate any multiple of 32).  The
    // engine's n_expert = 512 gives 512; the parity test uses the same.
    if (threads & (threads - 1))
        throw std::runtime_error("router_top10: work-group size " + std::to_string(threads) +
                                 " is not a power of two (the SYCL local-memory trees require one)");
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:205:209 @ 49da5df
    // the selection's taken-mask, then n_expert doubles for the exponentials, then n_expert floats for the
    // probabilities.  The mask is first so the two aligned arrays need no offset parameter.
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem =
        taken_bytes + (size_t) n_expert * sizeof(double) + (size_t) n_expert * sizeof(float);
    // SYCL-MIRROR-END
    // New (not in the CUDA file): Arc's shared memory is 128 KiB per work-group
    // (T0), so the CUDA 32 768-expert cap cannot be honoured here; the engine's
    // n_expert = 512 uses ~11 KiB with the tree scratch below.
    const size_t smem_total = smem + (size_t) threads * (sizeof(float) + sizeof(int)) + sizeof(double);
    if (smem_total > 120 * 1024)
        throw std::runtime_error("router_top10: local memory " + std::to_string(smem_total) +
                                 " B exceeds the Arc work-group budget");
    return q.submit([&](sycl::handler& h) {
        // The CUDA extern __shared__ s_raw[] + __shared__ s_red/s_rid/s_sum
        // (lines 73-79) become handler-constructed local accessors captured by
        // value.  CUDA aliases s_p onto s_ex's tail (line 126); SYCL gets a
        // separate accessor -- same semantics, no pointer arithmetic.
        sycl::local_accessor<unsigned char, 1> s_taken(sycl::range<1>(taken_bytes), h);
        sycl::local_accessor<double, 1> s_ex(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_p(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_red_f(sycl::range<1>((size_t) threads), h);
        sycl::local_accessor<int, 1> s_red_i(sycl::range<1>((size_t) threads), h);
        sycl::local_accessor<double, 1> s_sum(sycl::range<1>(1), h);
        // One work-group per token, mirroring the CUDA grid: the global range is
        // n_tokens * threads items (Level Zero rejects non-uniform splits, T2).
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * (size_t) threads, (size_t) threads),
                       [=](sycl::nd_item<1> it) {
    const int t = (int) (it.get_global_id(0) / it.get_local_range(0));  // glue: CUDA line 81 (blockIdx.x = token)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:82:82 @ 49da5df
    if (t >= n_tokens) return;
    // SYCL-MIRROR-END
    const int tid = (int) it.get_local_id(0);       // glue: CUDA line 83 (threadIdx.x)
    const int nt = (int) it.get_local_range(0);     // glue: CUDA line 84 (blockDim.x)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:85:85 @ 49da5df
    const float* l = logits + (size_t) t * n_expert;
    // SYCL-MIRROR-END

    // ---- softmax over ALL experts, for stability: the max.  A tree of `fmaxf`
    // is EXACT and order-independent, so this is bit-identical to the serial
    // scan.  (CUDA lines 87-88)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:89:90 @ 49da5df
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += nt) mx = fmaxf(mx, l[e]);
    // SYCL-MIRROR-END
    // glue (CUDA lines 91-100): the warp shuffle tree + shared 2nd stage become
    // a barrier tree over all nt partials in local memory -- fmaxf is exact and
    // order-independent, so the result is bit-identical to the CUDA two-stage one.
    s_red_f[tid] = mx;
    it.barrier();
    for (int stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red_f[tid] = fmaxf(s_red_f[tid], s_red_f[tid + stride]);
        it.barrier();
    }
    mx = s_red_f[0];                                // glue: CUDA line 101 (s_red[0] -> s_red_f[0])

    // ---- THE EXPONENTIALS, ONCE EACH AND IN PARALLEL.  (CUDA lines 103-104)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:105:105 @ 49da5df
    for (int e = tid; e < n_expert; e += nt) s_ex[e] = exp((double) l[e] - (double) mx);
    // SYCL-MIRROR-END
    it.barrier();                                   // glue: CUDA line 106 (__syncthreads)

    // ---- the sum, ascending, on one thread: see the CUDA file on why this is
    // NOT parallelised.  (CUDA line 107)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:109:111 @ 49da5df
    if (tid == 0) {
        double sum = 0.0;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
    // SYCL-MIRROR-END
        s_sum[0] = sum;                             // glue: CUDA line 112 (s_sum is a local_accessor)
    }
    it.barrier();                                   // glue: CUDA line 114 (__syncthreads)
    const float inv = (float) (1.0 / s_sum[0]);     // glue: CUDA line 115

    // ---- p[] ONCE.  (CUDA note lines 116-125: hoisted out of the selection so
    // the comparisons see exactly the same floats.)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:127:127 @ 49da5df
    for (int e = tid; e < n_expert; e += nt) s_p[e] = (float) (s_ex[e] * inv);
    // SYCL-MIRROR-END
    it.barrier();                                   // glue: CUDA line 128 (__syncthreads)

    // ================================ THE SELECTION, IN k PASSES ================================
    // k passes of a group-wide argmax: scanning e ascending with a strict >
    // keeps the LOWEST index on a tie -- the same order the CUDA kernel produces
    // (CUDA note lines 131-142).  The pair compare is a TOTAL ORDER, so the
    // local-memory tree returns the bit-identical winner to the CUDA two-stage
    // shuffle one.
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:142:142 @ 49da5df
    for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
    // SYCL-MIRROR-END
    it.barrier();                                   // glue: CUDA line 143 (__syncthreads)

    for (int i = 0; i < k; ++i) {
        // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:146:152 @ 49da5df
        float bv = -INFINITY;
        int bi = n_expert;               // a sentinel that loses to every real index
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        // SYCL-MIRROR-END
        // glue (CUDA lines 153-158): the per-warp shuffle tree + s_red/s_rid
        // write become a scatter to local memory + a barrier tree.
        s_red_f[tid] = bv;
        s_red_i[tid] = bi;
        it.barrier();
        for (int stride = nt >> 1; stride > 0; stride >>= 1) {
            const float ov = (tid < stride) ? s_red_f[tid + stride] : -INFINITY;
            const int oi = (tid < stride) ? s_red_i[tid + stride] : n_expert;
            // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:156:156 @ 49da5df
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            // SYCL-MIRROR-END
            s_red_f[tid] = bv;
            s_red_i[tid] = bi;
            it.barrier();
        }
        // glue (CUDA lines 159-168): the warp-leader 2nd stage collapses into
        // the tree above; lane 0 holds the group winner, exactly where CUDA's
        // 2nd stage left it.  Renamed to the CUDA 2nd-stage result names so
        // the write-out mirrors verbatim.
        const float v = bv;
        const int ix = bi;
            // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:169:173 @ 49da5df
            if (tid == 0 && ix < n_expert) {
                ids[(size_t) t * k + i] = ix;
                weights[(size_t) t * k + i] = v;
                s_taken[ix] = 1;
            }
            // SYCL-MIRROR-END
        it.barrier();                                   // glue: CUDA line 175 (__syncthreads)
    }
    it.barrier();                                   // glue: CUDA line 177 (__syncthreads)

    // ---- renormalise, with ggml's lower clamp.  Order preserved.  (CUDA line 179)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/router_top10.cu:180:186 @ 49da5df
    if (tid == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
        const double sc = fmax(s, 6.103515625e-05);       // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
    // SYCL-MIRROR-END
                       });
    });
}

}  // namespace strata::kernels
