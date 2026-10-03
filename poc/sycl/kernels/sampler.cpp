// poc/sycl/kernels/sampler.cpp -- T5: the sampler chain port (see
// plans/sycl-phase-a-t5.md).  CUDA source: src/kernels/cuda/sampler.cu @ bb7e783.
//
// Contract differences from the CUDA wrapper, on purpose and for Phase B:
// CUDA's sample_tokens() synchronises when stream == nullptr and exit(1)s on a bad
// argument or a launch error; submit_sample_tokens() returns the event, leaves
// synchronisation to the caller (drivers call wait_and_throw()), and throws
// std::runtime_error instead of exiting.  The Phase B engine wrapper absorbs
// both differences.
//
// STRUCTURE.  Five CUDA kernels become five submit_* functions, each a 1-D
// strata_launch-shaped launch:
//   - submit_sample_greedy   work-group 1024, one per token row
//   - submit_sample_old      work-group 1024, one per token row (the reference kernel)
//   - submit_sample_one_block work-group 1024, one per token row
//   - submit_split_part      work-group 128, one per (row, 4096-logit block)
//   - submit_split_merge     work-group 32, one per row
// The (value, index) reduction CUDA does with warp shuffles + shared second
// stages becomes a barrier-based decimation tournament over local memory:
// every thread scans its strided subset into a (value, index) pair (the strict
// `s > bv` scan, mirrored verbatim, which never admits NaN), then the 1024
// pairs fold in an ascending-window tournament.  The pair order (value, then
// lower index) is a strict total order, so the tournament winner equals
// CUDA's two-stage shuffle winner bit for bit.  t5/tree_probe.cpp verified the
// tournament shape at 1024 threads with ties, NaN and -inf in the data.
//
// The tail of the sampled path runs on ALL 1024 threads of the old and
// one-block kernels (exactly `sampler_kernel`'s shape, mirrored from lines
// 278:315); the split merge kernel runs it on its 32-thread row work-group in
// the warp shape of `sampled_tail_warp` (lines 367:421), with the lane-strided
// ex fills (race-free at exactly 32 threads) and the __shfl_sync broadcasts
// replaced by local-memory broadcasts.
//
// DOCUMENTED DEVIATIONS (probes in poc/sycl/t5/):
//   * P2/P2b: the device double `exp` differs from the host `std::exp` by 1
//     ulp on 14,443 of the 130,589 arguments the tail actually generates
//     (t5_math_fixture_probe), and device `logf` is bit-identical on all four
//     fixture min_p values.  The measured cut margins - at least ~2.1e8 double
//     ulps on the top_p/draw cum chains, at least ~9e3 fp32 ulps on the min_p
//     threshold - are ~6 orders of magnitude beyond what a 1-ulp exp term or
//     thresh shift can perturb (t5_math_fixture_probe gate, report §T5), so
//     the parity reference is compared on the PICKED INTEGER and the 1 ulp
//     cannot flip a cut; the fixtures confirm it when they run (Step 4).  A
//     stricter Phase B could move the host reference onto the device math.
//   * P1: the penalty bitmaps build with sycl::atomic_ref fetch_or on local
//     memory (works, verified) - the parallel build CUDA uses, kept.
//   * The split part's four regions run the warp_merge_lists merge of ALL four
//     warp lists each (4x redundant): the group barrier forces every region
//     through the merge in lockstep, and the total order makes the results
//     identical.  Only region 0 publishes cand.  Phase B can restructure.
//   * icpx 2026.1 has no subgroup shuffle/reduce API (T3 finding), so
//     warp_first's 32-lane XOR butterfly becomes a 32-entry local-memory
//     butterfly with two barriers per level.
//   * Phase A has no stream capture: stream_capturing (CUDA 769:779) is
//     deferred, the split's per-stream scratch cache (789:832) becomes a
//     per-call sycl::malloc_device freed in-order on the same queue, and
//     fixture 18a's captured-graph half is out of scope (18b's row-cap
//     fallback is ported).
//   * The coupled draft kernels (670:751, 891:933) are deferred to Phase B.
#include "sycl_compat/launch.hpp"

#include <strata/kernels/sampler.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

// glue (adaptation): CUDA's device intrinsics, as plain C++ so the mirrored
// kernel lines stay verbatim.  int2 keeps CUDA's layout ({int, int}); the
// sampler's lists store a float's bits in y.
struct int2 { int x; int y; };
static inline int2 make_int2(int x, int y) { return {x, y}; }
static inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, sizeof f); return f; }
static inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, sizeof i); return i; }
static inline uint32_t __umulhi(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * b) >> 32); }

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:1:20 @ bb7e783
// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`
// at 3cf03257) - ONE penalties stage, first, and TEMPERATURE AFTER THE TRUNCATION FILTERS.  (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
//
// `sampler_greedy_kernel` is the plain argmax, one block per token over the vocabulary.  The sampled chain has
// three implementations that pick the same token, bit for bit:
//   - the SPLIT top_k (default): `sampler_split_part_kernel` cuts each row into 4,096-logit blocks over the whole
//     GPU, each keeps its own top_k, and `sampler_split_merge_kernel` merges those lists and runs the tail;
//   - `sampler_one_block_kernel` (`STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split cannot run): one
//     block per token, `top_k` block-argmax rounds, each over the logits after the previous pick;
//   - `sampler_kernel` (`STRATA_OLD_SAMPLER=1`), the kernel of engine 0.1.20, kept as the reference.
// The two new ones share `sampled_tail_warp` (top_p / min_p / temperature / draw on one warp).
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:36:38 @ bb7e783
// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
// SYCL-MIRROR-END
// glue (CUDA 39:40): the __device__ qualifier drops off
static inline uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                        uint32_t k0, uint32_t k1) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:41:50 @ bb7e783
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
    // SYCL-MIRROR-END
}

// glue (CUDA 53): the __device__ qualifier drops off
static inline float philox_uniform(uint64_t seed, uint64_t counter) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:54:60 @ bb7e783
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:63:66 @ bb7e783
// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
// SYCL-MIRROR-END
static inline int history_count(const int* h, int n, int v) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:68:70 @ bb7e783
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
    // SYCL-MIRROR-END
}

static inline float apply_penalties(float logit, int count, const SamplerParams& p) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:74:78 @ bb7e783
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:327:328 @ bb7e783
constexpr int kSelMax = 64;               // the widest top_k list, `sampler_kernel`'s KMAX
constexpr unsigned kFullMask = 0xFFFFFFFFu;
// SYCL-MIRROR-END
// glue: kFullMask named for the mirrored CUDA code; the SYCL butterflies count their levels
// explicitly, so the constant itself is unused here
static_assert(kFullMask == 0xFFFFFFFFu);

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:330:330 @ bb7e783
// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
// SYCL-MIRROR-END
static inline int sampled_k(int top_k, int n_vocab) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:332:333 @ bb7e783
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
    // SYCL-MIRROR-END
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:336:336 @ bb7e783
// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
// SYCL-MIRROR-END
static inline void take_first(float& bv, int& bi, float ov, int oi) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:338:338 @ bb7e783
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    // SYCL-MIRROR-END
}

// glue (replaces CUDA 341:351, `warp_first`): the 32-lane XOR butterfly over the warp's
// candidates becomes a 32-entry local-memory butterfly.  The CUDA version exchanges each lane's
// running pair with lane ^ off and keeps it in registers; here every level publishes all 32 pairs
// (barrier), reads the neighbour's PRE-LEVEL pair (the second barrier keeps it from being
// overwritten mid-level), and folds with take_first.  Same total-order argument: every lane ends
// with the same pair.
// The scratch is passed as raw pointers (accessor .get_pointer()) so the
// callers can offset each 32-slot butterfly region; the [0..31] indexing in
// the body is lane-local.
static inline void warp_first(sycl::nd_item<1> it, float* s_v, int* s_i, int lane, float& bv, int& bi) {
    s_v[lane] = bv;
    s_i[lane] = bi;
    it.barrier();
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = s_v[lane ^ off];
        const int oi = s_i[lane ^ off];
        take_first(bv, bi, ov, oi);
        it.barrier();
    }
}

// glue (CUDA 522:524 signature): the merge needs the item (for the group barriers inside
// warp_first) and this warp's 32-pair scratch; `lane` comes from the caller.
// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:518:521 @ bb7e783
/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order and
/// padded with sentinels - into their first `k`: `sink(i, value, id)` runs in every lane for i = 0..k-1 with the
/// same pair.  Lane owns lists `lane` and `lane + 32`; a round takes the first of all heads and advances the list
/// it came from.  An id is in one list at most (the lists cover disjoint logits), so exactly one head matches.
// SYCL-MIRROR-END
template <typename Sink>
static inline void warp_merge_lists(sycl::nd_item<1> it, const int2* lists, int nl, int stride, int k, int n_vocab,
                                    int lane, float* s_v, int* s_i, Sink&& sink) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:526:539 @ bb7e783
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = __int_as_float(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const int2 c = lists[(size_t) L * stride];
            hi[m] = c.x;
            hv[m] = __int_as_float(c.y);
        }
    }
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:540:543 @ bb7e783
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        // SYCL-MIRROR-END
        // glue (CUDA 544): the XOR butterfly over the warp's 32 heads
        warp_first(it, s_v, s_i, lane, bv, bi);
        // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:545:559 @ bb7e783
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const int2 c = lists[(size_t) (lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x;
                    hv[m] = __int_as_float(c.y);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = __int_as_float(0xff800000);
                }
            }
        }
        // SYCL-MIRROR-END
    }
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:511:516 @ bb7e783
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)
// SYCL-MIRROR-END

// The decimation tournament fold: the per-thread scanned pairs sit in (rv, ri); level w folds each
// pair of adjacent windows into position m*2w, done by the thread at that position.  Invariant
// before level w: (rv[m*w], ri[m*w]) is the winner of [m*w, (m+1)*w).  Mirrors the comparison of
// CUDA's shuffle trees (line 148/255/482); the shuffle/sharing lines around it are glue.
static inline void fold_block(sycl::nd_item<1> it, const int N, float* rv, int* ri) {
    for (int w = 1; w <= N / 2; w <<= 1) {
        const int tid = (int) it.get_local_id(0);
        if ((tid & (2 * w - 1)) == 0) {
            float bv = rv[tid];
            int best = ri[tid];
            const float ov = rv[tid + w];
            const int oi = ri[tid + w];
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:148:148 @ bb7e783
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
            // SYCL-MIRROR-END
            rv[tid] = bv;
            ri[tid] = best;
        }
        it.barrier();
    }
}

// ---------------------------------------------------------------------------
// THE GREEDY ARGMAX
// ---------------------------------------------------------------------------

sycl::event submit_sample_greedy(sycl::queue& q, const float* logits, int n_tokens, int n_vocab, const int* history,
                                int history_len, const SamplerParams& p, int pmin, int plen, int* out) {
    const int N = 1024;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> penal_bits(sycl::range<1>(bits_words > 0 ? bits_words : 1), h);
        sycl::local_accessor<float, 1> rv(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> ri(sycl::range<1>(N), h);
        // glue: CUDA's <<<n_tokens, 1024>>> grid; exactly n_tokens groups of N, no padding
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * N, N), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int t = (int) (it.get_global_id(0) / N);
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:104:104 @ bb7e783
    const float* l = logits + (size_t) t * n_vocab;
            // SYCL-MIRROR-END
            // glue (CUDA 105): the unused pmin argument, kept for parity with the CUDA launch
            (void) pmin;
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:106:112 @ bb7e783
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
            // SYCL-MIRROR-END
            // glue (CUDA 119): the extern shared bitmap is a handler-built accessor
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:120:123 @ bb7e783
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
            // SYCL-MIRROR-END
            if (use_bits) {
                // glue (CUDA 125): strided zero over the work-group
                for (int w = tid; w < bits_words; w += N) penal_bits[w] = 0u;
                it.barrier();                        // glue: CUDA 126 __syncthreads
                for (int i = tid; i < hlen; i += N) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:128:128 @ bb7e783
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                    // SYCL-MIRROR-END
                        // glue (CUDA 129): atomicOr -> atomic_ref fetch_or (t5/atomic_probe.cpp: PASS)
                        sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                            penal_bits[hrow[i] >> 5])
                            .fetch_or(1u << (hrow[i] & 31));
                }
                it.barrier();                        // glue: CUDA 130 __syncthreads
            }
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:132:135 @ bb7e783
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };
            // SYCL-MIRROR-END
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:137:140 @ bb7e783
    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
            // SYCL-MIRROR-END
            for (int v = tid; v < n_vocab; v += N) {
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:142:143 @ bb7e783
        const float s = apply_penalties(l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
                // SYCL-MIRROR-END
            }
            // glue (replaces CUDA 145:154): the 32-wide warp-shuffle tree + the shared
            // second stage fold into the 1,024-pair tournament below
            rv[tid] = bv;
            ri[tid] = best;
            it.barrier();
            fold_block(it, N, rv.get_pointer(), ri.get_pointer());
            // glue (replaces CUDA 155:166): the tournament winner sits at local position 0
            // for every thread, so the write needs no warp-0 stage; the `lane == 0` gate
            // becomes tid == 0, and the sentinel id maps to 0 exactly as CUDA's line 164
            // comment explains (a tie between two -inf candidates leaves the sentinel).
            if (tid == 0) out[t] = (ri[0] < n_vocab) ? ri[0] : 0;
        });
    });
    return e;
}

// ---------------------------------------------------------------------------
// THE SAMPLED PATH, ONE BLOCK PER TOKEN (the reference kernel, STRATA_OLD_SAMPLER)
// ---------------------------------------------------------------------------

sycl::event submit_sample_old(sycl::queue& q, const float* logits, int n_vocab, int n_tokens, const int* history,
                             int history_len, const SamplerParams& p, int* out) {
    const int N = 1024;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // glue: the CUDA kernel guards `t >= n_tokens`; the SYCL launch has exactly n_tokens groups
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> penal_bits(sycl::range<1>(bits_words > 0 ? bits_words : 1), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> rv(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> ri(sycl::range<1>(N), h);
        // glue: CUDA's <<<n_tokens, 1024>>> grid; exactly n_tokens groups of N, no padding
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * N, N), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int t = (int) (it.get_global_id(0) / N);
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:192:196 @ bb7e783
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
            // SYCL-MIRROR-END
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:200:206 @ bb7e783
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
            // SYCL-MIRROR-END
            // glue (CUDA 211): the extern shared bitmap is a handler-built accessor
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:212:213 @ bb7e783
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
            // SYCL-MIRROR-END
            if (use_bits) {
                // glue (CUDA 215): strided zero over the work-group
                for (int w = tid; w < bits_words; w += N) penal_bits[w] = 0u;
                it.barrier();                        // glue: CUDA 216 __syncthreads
                for (int i = tid; i < hlen; i += N) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:218:218 @ bb7e783
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                    // SYCL-MIRROR-END
                        // glue (CUDA 219): atomicOr -> atomic_ref fetch_or (t5/atomic_probe.cpp: PASS)
                        sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                            penal_bits[hrow[i] >> 5])
                            .fetch_or(1u << (hrow[i] & 31));
                }
                it.barrier();                        // glue: CUDA 220 __syncthreads
            }
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:222:225 @ bb7e783
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };
            // SYCL-MIRROR-END
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:227:231 @ bb7e783
    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist the kernel
    // keeps, 64.  Every row writes out[t]: a verify window reads all of them.
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;
            // SYCL-MIRROR-END
            // ---- top_k: k rounds of a block argmax over the not-yet-taken.  `sel_*` holds the kept ids and their
            // raw logits in selection order: descending by value, ties to the lower index, which is the order the
            // top_p cut below is defined over.  (CUDA 233:235, abridged here: the shared sel_* / sv / si arrays are
            // handler-built accessors, below.)
            for (int i = 0; i < k; ++i) {
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:243:244 @ bb7e783
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
                // SYCL-MIRROR-END
                for (int v = tid; v < n_vocab; v += N) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:246:250 @ bb7e783
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v), p);
            if (s > bv) { bv = s; best = v; }
                    // SYCL-MIRROR-END
                }
                // glue (replaces CUDA 252:256, 258, 260:270): the shuffle tree + shared
                // second stage fold into the tournament; the winner at local position 0
                // is written by thread 0 (CUDA wrote it from warp 0's lane 0)
                rv[tid] = bv;
                ri[tid] = best;
                it.barrier();
                fold_block(it, N, rv.get_pointer(), ri.get_pointer());
                if (tid == 0) {
                    sel_ids[i] = (ri[0] < n_vocab) ? ri[0] : 0;
                    sel_logit[i] = rv[0];
                }
                it.barrier();                        // glue: CUDA 271 __syncthreads
            }
            // ---- top_p over the top_k list (penalised logits, descending as the selection produced them), then min_p,
            // then temperature and one Philox draw - llama.cpp's order (issue #53).  Every thread computes the same chain
            // redundantly over `sel_*` - the arithmetic is the serial kernel's, instruction for instruction - so they
            // agree on `pick` and thread 0 writes it.  (CUDA 274:277)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:278:315 @ bb7e783
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53: they were applied a
    // second time here, after the temperature scaling - llama.cpp's chain has one penalties stage)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
            // SYCL-MIRROR-END
            // glue (CUDA 316): the `threadIdx.x == 0` gate
            if (tid == 0) out[t] = pick;
        });
    });
    return e;
}

// ---------------------------------------------------------------------------
// THE ONE-BLOCK SAMPLED PATH (STRATA_SAMPLER_ONE_BLOCK, and the split's fallback)
// ---------------------------------------------------------------------------

sycl::event submit_sample_one_block(sycl::queue& q, const float* logits, int n_tokens, int n_vocab, const int* history,
                                   int history_len, const SamplerParams& p, int* out) {
    const int N = 1024;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> penal_bits(sycl::range<1>(bits_words > 0 ? bits_words : 1), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> rv(sycl::range<1>(N), h);
        sycl::local_accessor<int, 1> ri(sycl::range<1>(N), h);
        // glue: CUDA's <<<n_tokens, 1024>>> grid; exactly n_tokens groups of N, no padding
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * N, N), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int t = (int) (it.get_global_id(0) / N);
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:437:437 @ bb7e783
    const float* l = logits + (size_t) t * n_vocab;
            // SYCL-MIRROR-END
            // the penalty window and its membership bitmap, exactly as in `sampler_kernel` (CUDA 439)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:440:446 @ bb7e783
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
            // SYCL-MIRROR-END
            // glue (CUDA 447): the extern shared bitmap is a handler-built accessor
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:448:449 @ bb7e783
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
            // SYCL-MIRROR-END
            if (use_bits) {
                // glue (CUDA 451): strided zero over the work-group
                for (int w = tid; w < bits_words; w += N) penal_bits[w] = 0u;
                it.barrier();                        // glue: CUDA 452 __syncthreads
                for (int i = tid; i < hlen; i += N) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:454:454 @ bb7e783
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                    // SYCL-MIRROR-END
                        // glue (CUDA 455): atomicOr -> atomic_ref fetch_or (t5/atomic_probe.cpp: PASS)
                        sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                            penal_bits[hrow[i] >> 5])
                            .fetch_or(1u << (hrow[i] & 31));
                }
                it.barrier();                        // glue: CUDA 456 __syncthreads
            }
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:458:461 @ bb7e783
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };
            // SYCL-MIRROR-END
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:463:463 @ bb7e783
    const int k = sampled_k(p.top_k, n_vocab);
            // SYCL-MIRROR-END
            // glue (replaces CUDA 464:469): the shared sel_* / ex / sv / si arrays are
            // handler-built accessors; the tail below runs on all threads (the
            // `sampler_kernel` shape), so no ex scratch or warp split is needed
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:470:471 @ bb7e783
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
            // SYCL-MIRROR-END
            for (int i = 0; i < k; ++i) {
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:473:474 @ bb7e783
        float bv = __int_as_float(0xff800000);   // -inf
        int best = n_vocab;
                // SYCL-MIRROR-END
                for (int v = tid; v < n_vocab; v += N) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:476:477 @ bb7e783
            const float s = apply_penalties(l[v], hit_count(v), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
                    // SYCL-MIRROR-END
                }
                // glue (replaces CUDA 479:485, 487:497): the shuffle tree + shared second
                // stage fold into the tournament; thread 0 writes the pick (CUDA: warp 0,
                // lane 0).  An empty round leaves (-inf, 0) exactly as CUDA's line 498
                // comment says, and every later round is empty in both versions.
                rv[tid] = bv;
                ri[tid] = best;
                it.barrier();
                fold_block(it, N, rv.get_pointer(), ri.get_pointer());
                if (tid == 0) {
                    sel_ids[i] = (ri[0] < n_vocab) ? ri[0] : 0;
                    sel_logit[i] = rv[0];
                }
                it.barrier();                        // glue: CUDA 497 __syncthreads
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:499:500 @ bb7e783
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
                // SYCL-MIRROR-END
            }
            // glue (replaces CUDA 502:503): the warp-0-only `sampled_tail_warp` call
            // becomes the all-thread tail below (the `sampler_kernel` shape, lines
            // 278:315); inv_t is the same value that kernel computed at line 196
            const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:278:315 @ bb7e783
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53: they were applied a
    // second time here, after the temperature scaling - llama.cpp's chain has one penalties stage)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
            // SYCL-MIRROR-END
            // glue (CUDA 316): the `threadIdx.x == 0` gate
            if (tid == 0) out[t] = pick;
        });
    });
    return e;
}

// ---------------------------------------------------------------------------
// THE SPLIT top_k, stage 1: the top_k of each 4,096-logit block
// ---------------------------------------------------------------------------

sycl::event submit_split_part(sycl::queue& q, const float* logits, int n_vocab, const int* history, int history_len,
                             const SamplerParams& p, int k, int n_blocks, int2* cand, int n_tokens) {
    const int G = kSplitWarps * 32;   // one work-group per (row, block): four CUDA warps
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> bits(sycl::range<1>(kSplitBlockSpan / 32), h);
        sycl::local_accessor<int2, 1> wl(sycl::range<1>(kSplitWarps * kSelMax), h);
        sycl::local_accessor<float, 1> s_v(sycl::range<1>(kSplitWarps * 32), h);
        sycl::local_accessor<int, 1> s_i(sycl::range<1>(kSplitWarps * 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * n_blocks * G, G), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int item = (int) (it.get_global_id(0) / G);
            const int b = item % n_blocks;                    // glue: CUDA blockIdx.x
            const int t = item / n_blocks;                    // glue: CUDA blockIdx.y
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:572:572 @ bb7e783
    const float* l = logits + (size_t) t * n_vocab;
            // SYCL-MIRROR-END
            // glue (CUDA 573): the 128-thread work-group is four CUDA warps
            const int warp = (int) (tid >> 5), lane = (int) (tid & 31);
            const int blo = b * kSplitBlockSpan;              // glue (CUDA 574)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:576:582 @ bb7e783
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
            // SYCL-MIRROR-END
            // The membership bitmap of THIS BLOCK'S 4,096 logits (512 bytes, not the vocabulary's 31 KB): the same
            // test, and a hit pays the same exact count over the whole window.  (CUDA 583:584; the extern shared
            // array is a handler-built accessor)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:586:587 @ bb7e783
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
            // SYCL-MIRROR-END
                // glue (CUDA 588): strided zero over the work-group
                for (int w = tid; w < kSplitBlockSpan / 32; w += G) bits[w] = 0u;
                it.barrier();                        // glue: CUDA 589 __syncthreads
                for (int i = tid; i < hlen; i += G) {
                    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:591:592 @ bb7e783
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                    // SYCL-MIRROR-END
                        // glue (CUDA 593): atomicOr -> atomic_ref fetch_or (t5/atomic_probe.cpp: PASS)
                        sycl::atomic_ref<unsigned, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
                            bits[(h - blo) >> 5])
                            .fetch_or(1u << ((h - blo) & 31));
                }
                it.barrier();                        // glue: CUDA 595 __syncthreads
            }                                        // glue: CUDA 596
            // This warp's logits, penalised: `apply_penalties` with a zero count returns the logit unchanged, so
            // only the bitmap's hits go through it.  Past the vocabulary: -inf, which no round picks.  (CUDA 598:599)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:600:606 @ bb7e783
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? l[v] : __int_as_float(0xff800000);
    }
            // SYCL-MIRROR-END
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:607:614 @ bb7e783
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }
            // SYCL-MIRROR-END
            // glue (replaces CUDA 616): the extern shared `wl` is a handler-built
            // accessor, row-major warp x kSelMax
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:617:619 @ bb7e783
    float prev_v = __int_as_float(0x7f800000);   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
            // SYCL-MIRROR-END
            for (; i < k; ++i) {
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:621:632 @ bb7e783
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first in
        // the order; `take_first` then orders the two
        float b0 = __int_as_float(0xff800000), b1 = __int_as_float(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
                // SYCL-MIRROR-END
                // glue (CUDA 633): the warp shuffle butterfly over this region's 32 lanes
                warp_first(it, s_v.get_pointer() + warp * 32, s_i.get_pointer() + warp * 32, lane, b0, i0);
                // glue (replaces CUDA 634: CUDA's `break` becomes the sentinel write
                // below - every region must stay in lockstep through the group
                // barriers of the butterfly, so a dead region keeps running rounds
                // that pick nothing (nothing beats -inf, and a dead round's
                // i0 >= n_vocab keeps it dead).  The stored value is exactly what
                // CUDA's break + sentinel fill (639) produced.
                const bool dead = i0 >= n_vocab;
                // glue (CUDA 635): the lane-0 write, sentinel when the region is dead
                if (lane == 0)
                    wl[warp * kSelMax + i] = dead ? make_int2(n_vocab, __float_as_int(-INFINITY))
                                                 : make_int2(i0, __float_as_int(b0));
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:636:638 @ bb7e783
        prev_v = b0;
        prev_i = i0;
    }
                // SYCL-MIRROR-END
            it.barrier();                        // glue: CUDA 640 __syncthreads
            // glue (replaces CUDA 641:646): CUDA's warp 0 merges the four warp lists;
            // here EVERY region runs the merge (the group barriers keep all four in
            // lockstep through warp_first), and only region 0 publishes cand - the
            // total order makes the four merges identical.
            warp_merge_lists(it, wl.get_pointer(), kSplitWarps, kSelMax, k, n_vocab, lane,
                             s_v.get_pointer() + warp * 32, s_i.get_pointer() + warp * 32,
                             [&](int r, float v, int id) {
                                 // glue (CUDA 642:645): dst = cand + ((size_t) t * n_blocks + b) * k; only
                                 // region 0's lane 0 writes (CUDA: warp 0's lane 0).  The per-warp s_v/s_i
                                 // offsets keep the four regions' butterflies in disjoint slots (CUDA's
                                 // per-warp shuffles never shared state).
                                 if (warp == 0 && lane == 0)
                                     cand[(size_t) t * n_blocks * k + b * k + r] = make_int2(id, __float_as_int(v));
                             });
        });
    });
    return e;
}

// ---------------------------------------------------------------------------
// THE SPLIT top_k, stage 2: one warp per row merges the block lists, then the tail
// ---------------------------------------------------------------------------

sycl::event submit_split_merge(sycl::queue& q, const int2* cand, int n_blocks, int n_vocab, const SamplerParams& p,
                              int k, int* out, int n_tokens) {
    const int G = 32;
    auto e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int2, 1> lists(sycl::range<1>(kSplitMaxBlocks * kSelMax), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<double, 1> ex(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<double, 1> s_sum(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> s_cut(sycl::range<1>(1), h);
        sycl::local_accessor<float, 1> s_v(sycl::range<1>(G), h);
        sycl::local_accessor<int, 1> s_i(sycl::range<1>(G), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * G, G), [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            const int t = (int) (it.get_global_id(0) / G);
            const int lane = tid;                 // glue (CUDA 655): one work-group of 32 per row
            // glue (CUDA 656:659): the shared arrays are handler-built accessors
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:660:661 @ bb7e783
    const int2* src = cand + (size_t) t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
            // SYCL-MIRROR-END
            it.barrier();                         // glue: CUDA 662 __syncwarp
            // glue (replaces CUDA 663:665): the merge's sink writes the row's list; the
            // total order makes every lane compute the same merge
            warp_merge_lists(it, lists.get_pointer(), n_blocks, k, k, n_vocab, lane, s_v.get_pointer(),
                             s_i.get_pointer(),
                             [&](int i, float v, int id) {
                                 // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:664:664 @ bb7e783
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
                                 // SYCL-MIRROR-END
                             });
            it.barrier();                         // glue: CUDA 666 __syncwarp
            // glue (replaces CUDA 667: `sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex)`
            // inlined below).  The lane-strided ex fills are race-free because this
            // work-group is exactly one CUDA warp; the __shfl_sync broadcasts become
            // local-memory broadcasts; the __syncwarps become work-group barriers.
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:367:372 @ bb7e783
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32) ex[i] = exp((double) sel_logit[i] - (double) mx);
            // SYCL-MIRROR-END
                it.barrier();                     // glue: CUDA 373 __syncwarp
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:374:376 @ bb7e783
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
                // SYCL-MIRROR-END
                // glue (CUDA 377:378): the sum shuffle broadcast, via local memory
                // (lane 0 writes, all lanes read: the unguarded write would race)
                if (lane == 0) s_sum[0] = sum;
                it.barrier();
                sum = s_sum[0];
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:379:379 @ bb7e783
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
                // SYCL-MIRROR-END
                it.barrier();                     // glue: CUDA 380 __syncwarp
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:381:388 @ bb7e783
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
                // SYCL-MIRROR-END
                // glue (CUDA 389): the cut shuffle broadcast (lane 0 writes)
                if (lane == 0) s_cut[0] = cut;
                it.barrier();
                cut = s_cut[0];
                // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:390:391 @ bb7e783
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
                // SYCL-MIRROR-END
                it.barrier();                     // glue: CUDA 392 __syncwarp (`ex` is written again below)
            }
            // min_p on top_p's survivors, as in `sampler_kernel` (every lane, the same float arithmetic) (CUDA 394)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:395:399 @ bb7e783
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
            // SYCL-MIRROR-END
            // temperature, then one Philox draw (CUDA 400)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:401:403 @ bb7e783
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32) ex[i] = exp((double) (sel_logit[i] * inv_t) - (double) smx);
            // SYCL-MIRROR-END
            it.barrier();                         // glue: CUDA 404 __syncwarp
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:405:407 @ bb7e783
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
            // SYCL-MIRROR-END
            // glue (CUDA 408:409): the sum shuffle broadcast (lane 0 writes)
            if (lane == 0) s_sum[0] = sum;
            it.barrier();
            sum = s_sum[0];
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:410:410 @ bb7e783
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
            // SYCL-MIRROR-END
            it.barrier();                         // glue: CUDA 411 __syncwarp
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:412:421 @ bb7e783
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        double cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
            // SYCL-MIRROR-END
            // glue (CUDA 422): the `if constexpr (kProb) *prob_out` line belongs to the coupled
            // draft (kProb = true); the sampler's own calls take kProb = false, where the line
            // is compiled out - deferred with the coupled draft, Phase B
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:423:423 @ bb7e783
    }
            // SYCL-MIRROR-END
        });
    });
    return e;
}

}  // namespace

// ---------------------------------------------------------------------------
// The host dispatcher
// ---------------------------------------------------------------------------

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:753:755 @ bb7e783
// Which sampled path runs, read once: `STRATA_OLD_SAMPLER=1` is `sampler_kernel` (engine 0.1.20),
// `STRATA_SAMPLER_ONE_BLOCK=1` the one-block kernel; by default the split top_k wherever it applies.
enum class SampledPath { Split, OneBlock, Old };
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:757:760 @ bb7e783
bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:762:767 @ bb7e783
SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}
// SYCL-MIRROR-END

sycl::event submit_sample_tokens(sycl::queue& q, const float* logits, int n_tokens, int n_vocab, const int* history,
                                int history_len, const SamplerParams& p, int* out) {
    // glue (replaces CUDA 838:843): the same validation, but a bad argument throws
    // instead of fprintf + exit(1) - a PoC library should not kill the process
    if (n_tokens <= 0 || n_vocab <= 0) return q.submit([](sycl::handler&) {});
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0))
        throw std::runtime_error("submit_sample_tokens: penalty_last_n " + std::to_string(p.penalty_last_n) +
                                 " needs a history");
    // glue (CUDA 844:846): the CUDA dynamic shared size is each submit's local accessor
    sycl::event e;   // glue: the CUDA wrapper launches inline; the SYCL submitters return events
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:847:847 @ bb7e783
    if (p.greedy || p.temperature <= 0.0f) {
    // SYCL-MIRROR-END
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`. (CUDA 848)
        // glue (replaces CUDA 849:851): the <<<>>> launch becomes the SYCL submit
        e = submit_sample_greedy(q, logits, n_tokens, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n,
                                 out);
        // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:852:852 @ bb7e783
    } else if (sampled_path() == SampledPath::Old) {
        // SYCL-MIRROR-END
            // glue (replaces CUDA 853:856): the same block-per-token shape
            e = submit_sample_old(q, logits, n_vocab, n_tokens, history, history_len, p, out);
        // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:857:857 @ bb7e783
    } else {
        // SYCL-MIRROR-END
            // The split top_k by default: stage 1 over (blocks x rows), stage 2 one warp per row.  The one-block
            // kernel when asked for, or when the split cannot run: a wider vocabulary than the merge holds, or more
            // than `kSplitMaxRows` rows.  (CUDA 858:861)
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:862:864 @ bb7e783
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        int2* scratch = nullptr;
            // SYCL-MIRROR-END
            // glue (replaces CUDA 865:868): Phase A has no stream capture to detect (the
            // deferred stream_capturing), and the per-stream scratch cache becomes a
            // per-call sycl::malloc_device (freed in-order on this queue, below) at the
            // same size - sized for 16 rows and 64 entries at least, so a verify window
            // or a wider top_k does not regrow it
            if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows)
                scratch = sycl::malloc_device<int2>((size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax, q);
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:869:869 @ bb7e783
        if (scratch != nullptr) {
            // SYCL-MIRROR-END
                // glue (replaces CUDA 870:874): the two-stage launch, in order on this queue
                auto e = submit_split_part(q, logits, n_vocab, history, history_len, p, k, n_blocks, scratch, n_tokens);
                e = submit_split_merge(q, scratch, n_blocks, n_vocab, p, k, out, n_tokens);
                // glue (CUDA 880:885): a failed submit throws at enqueue; the caller owns
                // the wait (CUDA synced when stream == nullptr)
                sycl::free(scratch, q);
                return e;
            // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:875:875 @ bb7e783
        } else {
            // SYCL-MIRROR-END
                // glue (replaces CUDA 876:877): the one-block fallback
                auto e = submit_sample_one_block(q, logits, n_tokens, n_vocab, history, history_len, p, out);
                return e;
            }
        }
        return e;
    }
}  // namespace strata::kernels
