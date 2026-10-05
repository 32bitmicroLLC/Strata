// poc/sycl/drivers/elementwise_parity.cpp -- B1 (plans/sycl-phase-b-steps-b1.md,
// step 2.6): the SYCL parity driver for src/kernels/cuda/elementwise.cu,
// mirroring the CUDA driver src/kernels/elementwise_parity.cpp @ 9a8fc3f (its
// comparison logic is the spec; mirrored blocks are checked by check_mirrors.sh).
//
// SYCL adaptations (contract differences, on purpose):
//  - Memory and synchronization are the ctx/USM idiom (G8): cudaMalloc ->
//    ctx.device_alloc, cudaMemcpy -> ctx.q.memcpy, the null-stream synchronous
//    CUDA wrappers become submit_* + ctx.wait_and_throw.  `check(cudaError_t)`
//    is not mirrored (SYCL throws).
//  - The embedding section's CUDA stream-capture/graph replay is not available
//    in SYCL; the same contract it checked (the gather has no hidden
//    synchronization, and a following scale runs on the caller's queue) is
//    checked with two ordered launches on the in-order queue.
//  - F6 (the plan): `doorbell_ring/wait/publish`, `copy_from_mapped`,
//    `copy_rows_from_mapped`, `copy_i32_from_mapped` and `add_inplace` have no
//    CUDA parity coverage; per the plan they are compile-only in B1 and the
//    mapped-pinned handoff protocol is gated in Phase C.  This driver still
//    runs them once with TRIVIAL oracles (identity copies, an increment) as an
//    un-gated execution smoke -- a value failure there is a real bug, but the
//    Phase C protocol is what remains untested.
//  - `rsqrtf` in `rms_norm_weighted` is the port's glue macro `1.0f / sqrtf(x)`
//    (DPC++ has no device rsqrtf); the gate below is the CUDA driver's own
//    (rel 1e-6), far above the sqrt/div ulp gap -- the measured number is
//    recorded in the step report.
//  - All other comparisons, fixtures, tolerances and the rival-reading
//    observability checks are mirrored verbatim.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
// Forward decls; the definitions are in poc/sycl/kernels/elementwise.cpp (the
// port) and poc/sycl/kernels/dequant_bf16.cpp (the type-8 slice below).
sycl::event submit_embedding_gather(sycl::queue& q, const uint8_t* codes, const float* scales, const float* offsets,
                                    int64_t n, int code_bits, int code_bias, int group_elems, float* out);
sycl::event submit_gdn_gate(sycl::queue& q, const float* alpha, const float* dt, const float* ssm_a, float* gate,
                            int64_t n_tokens, int64_t h_v);
sycl::event submit_scale_inplace(sycl::queue& q, float* x, int64_t n, float s);
sycl::event submit_add_inplace(sycl::queue& q, float* dst, const float* src, int64_t n);
sycl::event submit_f32_to_f16_bulk(sycl::queue& q, const float* x, uint16_t* y, int64_t n);
sycl::event submit_f32_to_bf16_bulk(sycl::queue& q, const float* x, uint16_t* y, int64_t n);
sycl::event submit_silu_inplace(sycl::queue& q, float* x, int64_t n);
sycl::event submit_rms_norm_weighted(sycl::queue& q, float* x, const float* w, int64_t rows, int64_t cols, float eps);
sycl::event submit_doorbell_ring(sycl::queue& q, uint32_t* d_seq);
sycl::event submit_doorbell_wait(sycl::queue& q, const uint32_t* d_flag, const uint32_t* d_seq);
sycl::event submit_doorbell_publish(sycl::queue& q, const float* x, const int32_t* ids, const float* weights, int64_t n,
                                    int64_t k, float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq);
sycl::event submit_copy_from_mapped(sycl::queue& q, float* dst, const float* src, int64_t n);
sycl::event submit_copy_rows_from_mapped(sycl::queue& q, float* dst, const float* src, int64_t rows, int64_t width,
                                         const int32_t* hit_rows, const int32_t* count);
sycl::event submit_copy_i32_from_mapped(sycl::queue& q, int32_t* dst, const int32_t* src, int64_t n);
sycl::event submit_dequant_bf16(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                                uint16_t* out);
}  // namespace strata::kernels

namespace {

// SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:38:45 @ 9a8fc3f
double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}
// SYCL-MIRROR-END

}  // namespace

int main(int argc, char** argv) {
    // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:50:56 @ 9a8fc3f
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: elementwise_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    const int64_t H_V = 48;
    // SYCL-MIRROR-END

    strata::sycl_compat::ctx ctx;

    // ---- 1. gdn_gate, with the reference's own softplus
    {
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:60:99 @ 9a8fc3f
        const int n = (int) H_V;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> alpha(n), dt(n), a(n), want(n);
        for (int i = 0; i < n; ++i) {
            // A MIXTURE THAT ACTUALLY CROSSES THE BRANCH.  The first version drew alpha from a normal with
            // sigma 3, so the largest value was about 9 and `softplus` never took its `x > 20` path - the
            // "branch is observable" check then reported **0.00% apart, 0 non-finite**, which is the fixture
            // saying it cannot see the thing it was written to see.  Every third head is now large.
            const bool large = (i % 3) == 0;
            // Up to ~120, so the fixture spans WELL PAST f32's `exp` overflow at 88.  A fixture that stopped
            // at 88 would show the branch as unobservable and would be right: `log1pf(expf(x))` equals `x` to
            // f32 precision for the whole range 20..88, so ggml's threshold of 20 is CONSERVATIVE and the
            // behaviour only actually changes where `expf` overflows.
            alpha[i] = large ? (22.0f + 6.0f * (float) (i % 17)) : g(rng) * 3.0f;
            dt[i] = g(rng) * 0.5f;
            // NEGATIVE, as the artifact's `ssm_a = -exp(A_log)` is.  Checked below as a property.
            a[i] = -(std::fabs(g(rng)) + 0.1f);
            const double x = (double) alpha[i] + (double) dt[i];
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[(size_t) i] = (float) (sp * (double) a[i]);
        }
        // the fixture must EXERCISE the branch, or check 1 below measures nothing
        {
            int over = 0, past_overflow = 0;
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                if (x > 20.0f) ++over;
                if (x > 88.0f) ++past_overflow;
            }
            std::printf("  %-44s %s (%d above 20, %d above 88)\n", "the fixture crosses the branch",
                        over ? "yes" : "*** NO ***", over, past_overflow);
            if (!over) ++bad;
        }
        // the sign property, asserted rather than assumed
        int positive = 0;
        for (int i = 0; i < n; ++i) if (a[(size_t) i] >= 0.0f) ++positive;
        std::printf("  %-44s %s (%d of %d non-negative)\n", "the fixture's ssm_a is negative",
                    positive ? "*** NO ***" : "yes", positive, n);
        if (positive) ++bad;
        // SYCL-MIRROR-END

        // glue: cudaMalloc/cudaMemcpy -> ctx USM + queue copies
        float* d_a = ctx.device_alloc<float>(n);
        float* d_dt = ctx.device_alloc<float>(n);
        float* d_sa = ctx.device_alloc<float>(n);
        float* d_g = ctx.device_alloc<float>(n);
        ctx.q.memcpy(d_a, alpha.data(), n * 4);
        ctx.q.memcpy(d_dt, dt.data(), n * 4);
        ctx.q.memcpy(d_sa, a.data(), n * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_gdn_gate(ctx.q, d_a, d_dt, d_sa, d_g, 1, H_V);
        std::vector<float> got((size_t) n);
        ctx.q.memcpy(got.data(), d_g, n * 4);
        ctx.wait_and_throw();

        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:113:116 @ 9a8fc3f
        const double rel = rel_l1(want, got);
        std::printf("  %-44s rel %.3e\n", "gdn_gate vs the reference", rel);
        // double softplus and double multiply on both sides, then one cast
        if (!(rel <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++bad; }
        // SYCL-MIRROR-END

        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:118:146 @ 9a8fc3f
        // TRAP: no large-x branch.  `log1p(exp(x))` in f32, with no `x > 20` path.
        //
        // AND THE MEASUREMENT SAYS WHERE THE BRANCH ACTUALLY MATTERS.  The first version of this fixture
        // spanned 20..88 and reported **0 heads differ**: `log1pf(expf(x))` agrees with `x` to f32 precision
        // over that whole range, so ggml's threshold of 20 is CONSERVATIVE - the branch changes the answer only
        // where `expf` OVERFLOWS, at about 88.  The check reports the smallest x at which the two readings
        // part company, so the number is in the output rather than in this comment.
        {
            int wrong = 0, differing = 0;
            float first_differ = -1.0f;
            std::vector<float> rival((size_t) n);
            for (int i = 0; i < n; ++i) {
                const float x = alpha[(size_t) i] + dt[(size_t) i];
                const float sp = std::log1p(std::exp(x));       // no branch, f32
                rival[(size_t) i] = sp * a[(size_t) i];
                if (!std::isfinite(sp)) ++wrong;
                const float branched = x > 20.0f ? x : std::log1pf(std::exp(x));
                if (sp != branched) {
                    ++differing;
                    if (first_differ < 0.0f || x < first_differ) first_differ = x;
                }
            }
            const double r = rel_l1(want, rival);
            const bool visible = r > 0.05 || wrong > 0;
            std::printf("  %-44s %s (%.2f%% apart, %d non-finite, %d heads differ, first at x = %.1f)\n",
                        "the softplus large-x branch is observable", visible ? "yes" : "*** NO ***",
                        r * 100, wrong, differing, (double) first_differ);
            if (!visible) ++bad;
        }
        // SYCL-MIRROR-END
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:147:154 @ 9a8fc3f
        // PROPERTY: exp(gate) < 1 for every element, which is what makes the state decay
        {
            int over = 0;
            for (float v : got) if (!(std::exp(v) < 1.0f)) ++over;
            std::printf("  %-44s %s (%d of %d not < 1)\n", "exp(gate) < 1 for every head",
                        over ? "*** NO ***" : "yes", over, n);
            if (over) ++bad;
        }
        // SYCL-MIRROR-END
        ctx.free_device(d_a);
        ctx.free_device(d_dt);
        ctx.free_device(d_sa);
        ctx.free_device(d_g);
    }

    // ---- 2. silu, in double then cast
    {
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:160:168 @ 9a8fc3f
        const int n = 4096;
        std::mt19937 rng(22);
        std::normal_distribution<float> g(0.0f, 3.0f);
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) {
            x[(size_t) i] = g(rng);
            const double v = (double) x[(size_t) i];
            want[(size_t) i] = (float) (v / (1.0 + std::exp(-v)));
        }
        // SYCL-MIRROR-END
        float* d_x = ctx.device_alloc<float>(n);
        ctx.q.memcpy(d_x, x.data(), n * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_silu_inplace(ctx.q, d_x, n);
        std::vector<float> got((size_t) n);
        ctx.q.memcpy(got.data(), d_x, n * 4);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:175:177 @ 9a8fc3f
        const double rel = rel_l1(want, got);
        std::printf("\n  %-44s rel %.3e\n", "silu (double) vs the reference", rel);
        if (!(rel <= 1e-7)) { std::printf("    *** over 1e-7 ***\n"); ++bad; }
        // SYCL-MIRROR-END
        ctx.free_device(d_x);
    }

    // ---- 3. scale and the f32->f16 bridge
    {
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:183:185 @ 9a8fc3f
        const int n = 1024;
        std::vector<float> x((size_t) n), want((size_t) n);
        for (int i = 0; i < n; ++i) x[(size_t) i] = (float) (i - n / 2) * 0.013f;
        // SYCL-MIRROR-END
        // glue for CUDA lines 186-187: the oracle's ONE f32 multiply, pinned volatile.
        // icpx host -O2 transforms `x[i] * s` for the compile-time constant
        // s = 1.0f/sqrt(128.0f): 466 of the 1024 products came 1 ulp away from a
        // true single f32 multiply (probed; the device kernel's multiply is the
        // true one -- a volatile-pinned reference matches it 1024/1024).  The
        // embedding section's own comment requires the same pin.
        const float s = 1.0f / std::sqrt(128.0f);
        for (int i = 0; i < n; ++i) {
            volatile float xv = x[(size_t) i], sv = s;
            want[(size_t) i] = xv * sv;
        }
        float* d_x = ctx.device_alloc<float>(n);
        uint16_t* d_h = ctx.device_alloc<uint16_t>(n);
        ctx.q.memcpy(d_x, x.data(), n * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_scale_inplace(ctx.q, d_x, n, s);
        std::vector<float> got((size_t) n);
        ctx.q.memcpy(got.data(), d_x, n * 4);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:197:200 @ 9a8fc3f
        int diff = 0;
        for (int i = 0; i < n; ++i) if (got[(size_t) i] != want[(size_t) i]) ++diff;
        std::printf("  %-44s %d of %d differ\n", "scale_inplace is exact", diff, n);
        if (diff) ++bad;
        // SYCL-MIRROR-END
        strata::kernels::submit_f32_to_f16_bulk(ctx.q, d_x, d_h, n);
        std::vector<uint16_t> h((size_t) n);
        ctx.q.memcpy(h.data(), d_h, n * 2);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:205:209 @ 9a8fc3f
        int hbad = 0;
        for (int i = 0; i < n; ++i)
            if (h[(size_t) i] != strata::kernels::f16_from_f32(got[(size_t) i])) ++hbad;
        std::printf("  %-44s %d of %d differ\n", "f32_to_f16_bulk uses the shared conversion", hbad, n);
        if (hbad) ++bad;
        // SYCL-MIRROR-END
        ctx.free_device(d_x);
        ctx.free_device(d_h);
    }

    // ---- 4. rms_norm_weighted, and the TWO RIVAL READINGS it exists to be told apart from.
    //
    // This kernel's whole reason for being a separate entry point from `gdn_l2_norm` is that the two differ by
    // one `/ cols`, and that a swap produces a well-scaled, plausible tensor.  So the test computes BOTH wrong
    // readings and requires each to differ from the right one by more than the tolerance before it judges the
    // kernel - the same discipline the shared-expert test uses for silu-on-gate.
    {
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:220:264 @ 9a8fc3f
        const int64_t rows = 24, cols = 256;          // the real `attn_q`-norm shape
        const float eps = 1e-6f;
        std::vector<float> x((size_t) (rows * cols));
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) std::sin((double) i * 0.017) * 1.7f + 0.4f;
        std::vector<float> w((size_t) cols);
        for (int64_t c = 0; c < cols; ++c) w[(size_t) c] = 0.5f + 0.002f * (float) (c % 97);

        // the oracle: f64, mean, times w
        auto ref = [&](bool use_sum, bool use_w) {
            std::vector<double> y((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r) {
                double ss = 0;
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    ss += v * v;
                }
                const double den = std::sqrt((use_sum ? ss : ss / (double) cols) + (double) eps);
                for (int64_t c = 0; c < cols; ++c) {
                    const double v = x[(size_t) (r * cols + c)];
                    y[(size_t) (r * cols + c)] = (v / den) * (use_w ? (double) w[(size_t) c] : 1.0);
                }
            }
            std::vector<float> out((size_t) (rows * cols));
            for (size_t i = 0; i < out.size(); ++i) out[i] = (float) y[i];
            return out;
        };
        const std::vector<float> want = ref(false, true);
        const std::vector<float> rival_sum = ref(true, false);    // what gdn_l2_norm computes
        const std::vector<float> rival_now = ref(false, false);   // the (1+w)-vs-w / no-weight confusion

        auto rel = [&](const std::vector<float>& a, const std::vector<float>& b) {
            double d = 0, m = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                d += std::fabs((double) a[i] - (double) b[i]);
                m += std::fabs((double) a[i]);
            }
            return d / (m > 1e-30 ? m : 1e-30);
        };
        const double r_sum = rel(want, rival_sum), r_now = rel(want, rival_now);
        std::printf("  %-44s %.2f%% apart\n", "sum-vs-mean is observable", r_sum * 100);
        std::printf("  %-44s %.2f%% apart\n", "with-w-vs-without-w is observable", r_now * 100);
        // `sqrt(cols)` = 16 for the first and the weight's own spread for the second; a fixture that cannot
        // separate them is not testing the kernel that was written.
        if (!(r_sum > 0.5)) { std::printf("  *** the sum reading is NOT observable ***\n"); ++bad; }
        if (!(r_now > 0.5)) { std::printf("  *** the no-weight reading is NOT observable ***\n"); ++bad; }
        // SYCL-MIRROR-END

        float* d_x = ctx.device_alloc<float>(x.size());
        float* d_w = ctx.device_alloc<float>(w.size());
        ctx.q.memcpy(d_x, x.data(), x.size() * 4);
        ctx.q.memcpy(d_w, w.data(), w.size() * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_rms_norm_weighted(ctx.q, d_x, d_w, rows, cols, eps);
        std::vector<float> got((size_t) (rows * cols));
        ctx.q.memcpy(got.data(), d_x, got.size() * 4);
        ctx.wait_and_throw();

        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:276:290 @ 9a8fc3f
        const double r_got = rel(want, got);
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double m = std::fabs((double) want[i]);
            const double e = std::fabs((double) got[i] - (double) want[i]) / (m > 1e-30 ? m : 1e-30);
            if (e > worst) worst = e;
        }
        int nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        std::printf("  %-44s rel %.3e  worst %.3e  nonfinite %d\n", "rms_norm_weighted vs f64 oracle", r_got,
                    worst, nonfinite);
        if (nonfinite) ++bad;
        // The kernel accumulates in f32 over 256 terms; 1e-6 is loose enough for that and far tighter than the
        // 16x the rival readings are off by.
        if (!(r_got < 1e-6)) { std::printf("  *** rms_norm_weighted is WRONG ***\n"); ++bad; }
        // SYCL-MIRROR-END
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:291:298 @ 9a8fc3f
        // and a null `w` must be legal, because `ref/qsa.py` allows it.
        //
        // THE INPUT HAS TO BE RE-UPLOADED FIRST.  The kernel is IN PLACE, so `d_x` currently holds the
        // NORMALISED tensor from the call above; running it again normalises a second time and every element
        // differs.  That is exactly what the first version of this check reported - "6144 of 6144 differ" -
        // and it is a fixture bug rather than a kernel bug: the oracle reads the ORIGINAL `x`, so the two
        // sides were never looking at the same input.
        const std::vector<float> want_null = ref(false, false);
        // SYCL-MIRROR-END
        ctx.q.memcpy(d_x, x.data(), x.size() * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_rms_norm_weighted(ctx.q, d_x, nullptr, rows, cols, eps);
        ctx.q.memcpy(got.data(), d_x, got.size() * 4);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:302:306 @ 9a8fc3f
        int null_bad = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (std::fabs((double) got[i] - (double) want_null[i]) > 1e-6) ++null_bad;
        std::printf("  %-44s %d of %zu differ\n", "a null weight is legal", null_bad, got.size());
        if (null_bad) ++bad;
        // SYCL-MIRROR-END
        ctx.free_device(d_x);
        ctx.free_device(d_w);
    }

    // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:311:314 @ 9a8fc3f
    // Packed embedding rows: every code width, scale-group size, optional offset,
    // row boundary and partial CUDA block. Compare bits with a scalar CPU decoder.
    // Volatile materializes the multiply so this oracle cannot silently use FMA.
    {
        // glue: the CUDA side's non-blocking stream becomes the in-order ctx.q
        // (the graph-capture check below becomes an ordered-launch check).
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:317:340 @ 9a8fc3f
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> values(-2.0f, 2.0f);
        int mismatches = 0, guards = 0, fma_diff = 0, cases = 0;
        for (const int bits : {2, 4, 8}) {
            const int bias = bits == 2 ? -1 : bits == 4 ? -7 : -16;
            const int per_byte = 8 / bits;
            const unsigned mask = (1u << bits) - 1u;
            for (const int group : {16, 32, 64}) {
                for (const int n : {320, 2560}) {
                    const int row_bytes = n / per_byte, row_groups = n / group;
                    std::vector<uint8_t> codes((size_t) 3 * row_bytes, 0);
                    std::vector<float> scales((size_t) 3 * row_groups), offsets(scales.size());
                    std::vector<int> unpacked((size_t) 3 * n);
                    for (size_t i = 0; i < unpacked.size(); ++i) {
                        const unsigned code = (unsigned) (i * 13 + i / n * 7) & mask;
                        unpacked[i] = (int) code;
                        codes[i / per_byte] |= (uint8_t) (code << ((i % per_byte) * bits));
                    }
                    for (size_t i = 0; i < scales.size(); ++i) {
                        scales[i] = values(rng);
                        offsets[i] = values(rng);
                    }
                    scales[0] = -0.0f;
                    offsets[0] = 0.0f;
                    // SYCL-MIRROR-END
                    uint8_t* dc = ctx.device_alloc<uint8_t>(codes.size());
                    float* ds = ctx.device_alloc<float>(scales.size());
                    float* dof = ctx.device_alloc<float>(offsets.size());
                    float* out = ctx.device_alloc<float>((size_t) n + 2);
                    ctx.q.memcpy(dc, codes.data(), codes.size());
                    ctx.q.memcpy(ds, scales.data(), scales.size() * sizeof(float));
                    ctx.q.memcpy(dof, offsets.data(), offsets.size() * sizeof(float));
                    ctx.wait_and_throw();

                    for (const bool with_offset : {false, true}) {
                        for (int row = 0; row < 3; ++row) {
                            std::vector<float> want((size_t) n), got((size_t) n + 2, -12345.0f);
                            for (int i = 0; i < n; ++i) {
                                const size_t gi = (size_t) row * row_groups + i / group;
                                const float code = (float) (unpacked[(size_t) row * n + i] + bias);
                                volatile float product = code * scales[gi];
                                const float offset = with_offset ? offsets[gi] : 0.0f;
                                want[(size_t) i] = product + offset;
                                // glue for CUDA line 361: the fused-product reference, via the FMA3
                                // instruction.  icpx host -O2's `std::fma` is a TWO-rounding
                                // implementation (probed: 514,084 of 531,441 samples differ from
                                // vfmadd231ss), so it cannot stand in for the fused product the
                                // `fma_diff > 0` non-vacuity check asserts.
                                float fused_r = offset;
                                asm("vfmadd231ss %1, %2, %0" : "+x"(fused_r) : "x"(code), "x"(scales[gi]));
                                const float fused = fused_r;
                                if (std::memcmp(&fused, &want[(size_t) i], sizeof(float)) != 0) ++fma_diff;
                            }
                            ctx.q.memcpy(out, got.data(), got.size() * sizeof(float));
                            strata::kernels::submit_embedding_gather(ctx.q, dc + (size_t) row * row_bytes,
                                                                   ds + (size_t) row * row_groups,
                                                                   with_offset ? dof + (size_t) row * row_groups
                                                                              : nullptr,
                                                                   n, bits, bias, group, out + 1);
                            ctx.q.memcpy(got.data(), out, got.size() * sizeof(float));
                            ctx.wait_and_throw();
                            for (int i = 0; i < n; ++i) {
                                if (std::memcmp(&want[(size_t) i], &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                            }
                            if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                            ++cases;

                            // The CUDA driver captured this gather+scale in a graph to prove the gather has no
                            // hidden synchronization and runs on the caller's stream.  SYCL has no stream
                            // capture; the same contract is checked with two ordered launches on the in-order
                            // queue: if the gather synchronised or escaped the queue, the following scale would
                            // not observe its result in order.
                            if (row == 1) {
                                strata::kernels::submit_embedding_gather(ctx.q, dc + (size_t) row * row_bytes,
                                                                        ds + (size_t) row * row_groups,
                                                                        with_offset ? dof + (size_t) row * row_groups
                                                                                    : nullptr,
                                                                        n, bits, bias, group, out + 1);
                                strata::kernels::submit_scale_inplace(ctx.q, out + 1, n, 2.0f);
                                ctx.q.memcpy(got.data(), out, got.size() * sizeof(float));
                                ctx.wait_and_throw();
                                // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:393:397 @ 9a8fc3f
                                for (int i = 0; i < n; ++i) {
                                    const float expected = want[(size_t) i] * 2.0f;
                                    if (std::memcmp(&expected, &got[(size_t) i + 1], sizeof(float)) != 0) ++mismatches;
                                }
                                if (got.front() != -12345.0f || got.back() != -12345.0f) ++guards;
                                // SYCL-MIRROR-END
                            }
                        }
                    }
                    ctx.free_device(dc);
                    ctx.free_device(ds);
                    ctx.free_device(dof);
                    ctx.free_device(out);
                }
            }
        }
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:408:410 @ 9a8fc3f
        std::printf("  embedding gather: %d row cases, %d bit mismatches, %d guard failures, %d FMA differences\n",
                    cases, mismatches, guards, fma_diff);
        if (mismatches || guards || fma_diff == 0) ++bad;
        // SYCL-MIRROR-END
    }

    // ---- THE f32 -> bf16 CONVERSIONS KEEP A NaN A NaN.  `f32_to_bf16_bulk` (the header's `bf16_from_f32`
    // on the device) and the prompt path's dequantizer (its own `f2bf`) against ggml_compute_fp32_to_bf16's rule;
    // before the fix 0x7FFFFFFF came back as -0 and 0x7F800001 as +inf.  `bf16_bits_test` covers every f32 on
    // the host; this is the same header compiled by nvcc, plus the second copy in dequant_bf16.cu.
    {
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:418:427 @ 9a8fc3f
        auto ggml_bf16 = [](uint32_t u) -> uint16_t {
            if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
            return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
        };
        std::vector<uint32_t> bits = {0x7FFFFFFFu, 0xFFFFFFFFu, 0x7F800001u, 0xFF800001u, 0x7FC00000u, 0x7FBFFFFFu,
                                      0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0x00000000u, 0x80000000u, 0x3F808000u,
                                      0x3F818000u, 0x00008000u, 0x7F7F8000u};
        std::mt19937 rng(13);
        while (bits.size() < 4096) bits.push_back((uint32_t) rng());
        const size_t n = bits.size();
        // SYCL-MIRROR-END
        float* dx = ctx.device_alloc<float>(n);
        uint16_t* dy = ctx.device_alloc<uint16_t>(n);
        ctx.q.memcpy(dx, bits.data(), n * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_f32_to_bf16_bulk(ctx.q, dx, dy, (int64_t) n);
        std::vector<uint16_t> got(n);
        ctx.q.memcpy(got.data(), dy, n * 2);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:436:437 @ 9a8fc3f
        int wrong = 0;
        for (size_t i = 0; i < n; ++i) wrong += got[i] != ggml_bf16(bits[i]);
        // SYCL-MIRROR-END

        // A Q8_0 block (type 8) whose fp16 scale is a NaN: every dequantized value is NaN * q, a NaN, and it must
        // still be one in BF16.  A second block with an ordinary scale checks the finite path did not move.
        std::vector<uint8_t> blk(2 * 34, 0);
        blk[0] = 0x00; blk[1] = 0x7E;                        // fp16 quiet NaN
        blk[34] = 0x00; blk[35] = 0x3C;                      // fp16 1.0
        for (int j = 0; j < 32; ++j) {
            blk[(size_t) (2 + j)] = (uint8_t) (int8_t) (j - 16);
            blk[(size_t) (36 + j)] = (uint8_t) (int8_t) (3 * j - 50);
        }
        uint8_t* db = ctx.device_alloc<uint8_t>(blk.size());
        uint16_t* dq = ctx.device_alloc<uint16_t>(64);
        ctx.q.memcpy(db, blk.data(), blk.size());
        ctx.wait_and_throw();
        strata::kernels::submit_dequant_bf16(ctx.q, 8, db, 0, 2, 32, dq);
        std::vector<uint16_t> q(64);
        ctx.q.memcpy(q.data(), dq, 64 * 2);
        ctx.wait_and_throw();
        // SYCL-MIRROR-BEGIN src/kernels/elementwise_parity.cpp:457:467 @ 9a8fc3f
        int dq_wrong = 0;
        for (int j = 0; j < 32; ++j) {
            if (!((q[(size_t) j] & 0x7FFFu) > 0x7F80u)) ++dq_wrong;           // row 0: NaN scale -> NaN
            const float v = (float) (3 * j - 50);                                // row 1: d = 1.0 -> the integer
            uint32_t vb;
            std::memcpy(&vb, &v, 4);
            if (q[(size_t) (32 + j)] != ggml_bf16(vb)) ++dq_wrong;
        }
        std::printf("  f32 -> bf16 (NaN kept): bulk %d of %zu differ from ggml, dequant_bf16 %d of 64 wrong\n",
                    wrong, n, dq_wrong);
        if (wrong || dq_wrong) ++bad;
        // SYCL-MIRROR-END
        ctx.free_device(dx);
        ctx.free_device(dy);
        ctx.free_device(db);
        ctx.free_device(dq);
    }

    // ---- F6 (the plan): the six entries without CUDA parity coverage, run once
    // with TRIVIAL oracles.  Not the parity gate (the mapped-pinned handoff
    // protocol is Phase C's); an execution or value failure here is a real bug.
    {
        // add_inplace: dst += src, bit-exact for these values (a single rounded add)
        const int n = 1000;
        std::vector<float> h_dst(n), h_src(n);
        for (int i = 0; i < n; ++i) { h_dst[i] = (float) (i % 13) - 6.0f; h_src[i] = 0.25f * (float) ((i % 7) - 3); }
        float* d_dst = ctx.device_alloc<float>(n);
        float* d_src = ctx.device_alloc<float>(n);
        ctx.q.memcpy(d_dst, h_dst.data(), n * 4);
        ctx.q.memcpy(d_src, h_src.data(), n * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_add_inplace(ctx.q, d_dst, d_src, n);
        std::vector<float> got((size_t) n);
        ctx.q.memcpy(got.data(), d_dst, n * 4);
        ctx.wait_and_throw();
        int bad_add = 0;
        for (int i = 0; i < n; ++i) if (got[(size_t) i] != h_dst[(size_t) i] + h_src[(size_t) i]) ++bad_add;
        std::printf("  %-44s %d of %d differ\n", "add_inplace (F6 smoke) is exact", bad_add, n);
        if (bad_add) ++bad;
        ctx.free_device(d_dst);
        ctx.free_device(d_src);

        // copy_from_mapped / copy_i32_from_mapped: identity copies on device memory
        const int nf = 1000;      // multiple of 4, aligned USM pointers
        const int ni = 300;
        std::vector<float> h_f(nf);
        std::vector<int32_t> h_i(ni);
        for (int i = 0; i < nf; ++i) h_f[i] = 0.5f * (float) (i - nf / 2);
        for (int i = 0; i < ni; ++i) h_i[i] = (int32_t) (i * 97 + 5);
        float* d_fsrc = ctx.device_alloc<float>(nf);
        float* d_fdst = ctx.device_alloc<float>(nf);
        int32_t* d_isrc = ctx.device_alloc<int32_t>(ni);
        int32_t* idst = ctx.device_alloc<int32_t>(ni);
        ctx.q.memcpy(d_fsrc, h_f.data(), nf * 4);
        ctx.q.memcpy(d_isrc, h_i.data(), ni * 4);
        ctx.wait_and_throw();
        strata::kernels::submit_copy_from_mapped(ctx.q, d_fdst, d_fsrc, nf);
        strata::kernels::submit_copy_i32_from_mapped(ctx.q, idst, d_isrc, ni);
        std::vector<float> gf((size_t) nf);
        std::vector<int32_t> gi((size_t) ni);
        ctx.q.memcpy(gf.data(), d_fdst, nf * 4);
        ctx.q.memcpy(gi.data(), idst, ni * 4);
        ctx.wait_and_throw();
        int bad_f = 0, bad_i = 0;
        for (int i = 0; i < nf; ++i) if (gf[(size_t) i] != h_f[(size_t) i]) ++bad_f;
        for (int i = 0; i < ni; ++i) if (gi[(size_t) i] != h_i[(size_t) i]) ++bad_i;
        std::printf("  %-44s %d/%d floats, %d/%d ints\n", "mapped copies (F6 smoke) are identity", bad_f, nf, bad_i, ni);
        if (bad_f || bad_i) ++bad;
        ctx.free_device(d_fsrc);
        ctx.free_device(d_fdst);
        ctx.free_device(d_isrc);
        ctx.free_device(idst);

        // copy_rows_from_mapped: the hit row gets +0.0, the others copy
        {
            const int64_t rows = 6, width = 128;
            std::vector<float> h_src((size_t) rows * width), h_dst((size_t) rows * width, -1.0f);
            for (size_t i = 0; i < h_src.size(); ++i) h_src[i] = 0.01f * (float) i;
            const int32_t hit_row = 2, hit_count = 1;
            int32_t* d_hr = ctx.device_alloc<int32_t>(1);
            int32_t* d_cnt = ctx.device_alloc<int32_t>(1);
            float* d_src = ctx.device_alloc<float>(h_src.size());
            float* d_dst = ctx.device_alloc<float>(h_dst.size());
            ctx.q.memcpy(d_hr, &hit_row, 4);
            ctx.q.memcpy(d_cnt, &hit_count, 4);
            ctx.q.memcpy(d_src, h_src.data(), h_src.size() * 4);
            ctx.wait_and_throw();
            strata::kernels::submit_copy_rows_from_mapped(ctx.q, d_dst, d_src, rows, width, d_hr, d_cnt);
            ctx.q.memcpy(h_dst.data(), d_dst, h_dst.size() * 4);
            ctx.wait_and_throw();
            int bad_rows = 0;
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < width; ++c) {
                    const float expect = (r == hit_row) ? 0.0f : h_src[(size_t) (r * width + c)];
                    if (h_dst[(size_t) (r * width + c)] != expect) ++bad_rows;
                }
            std::printf("  %-44s %d of %zu elements\n", "copy_rows_from_mapped (F6 smoke): the hit row is +0.0",
                        bad_rows, h_dst.size());
            if (bad_rows) ++bad;
            ctx.free_device(d_hr);
            ctx.free_device(d_cnt);
            ctx.free_device(d_src);
            ctx.free_device(d_dst);
        }

        // doorbell_ring: *seq += 1 in memory (an increment, not a stored value)
        {
            const uint32_t start = 7;
            uint32_t* d_seq = ctx.device_alloc<uint32_t>(1);
            ctx.q.memcpy(d_seq, &start, 4);
            ctx.wait_and_throw();
            strata::kernels::submit_doorbell_ring(ctx.q, d_seq);
            uint32_t seq = 0;
            ctx.q.memcpy(&seq, d_seq, 4);
            ctx.wait_and_throw();
            const int bad_ring = (seq != start + 1u) ? 1 : 0;
            std::printf("  %-44s seq %u -> %u\n", "doorbell_ring (F6 smoke) increments", start, seq);
            if (bad_ring) ++bad;
            ctx.free_device(d_seq);
        }

        // doorbell_wait: the immediate-exit path (flag == seq at launch).  The live
        // spin path needs a concurrent writer to device memory, which a single
        // in-order queue cannot provide; the mapped-pinned handoff is Phase C's gate.
        {
            const uint32_t v = 9;
            uint32_t* d_flag = ctx.device_alloc<uint32_t>(1);
            uint32_t* d_seq = ctx.device_alloc<uint32_t>(1);
            ctx.q.memcpy(d_flag, &v, 4);
            ctx.q.memcpy(d_seq, &v, 4);
            ctx.wait_and_throw();
            sycl::event ev = strata::kernels::submit_doorbell_wait(ctx.q, d_flag, d_seq);
            ev.wait();
            try {
                ctx.wait_and_throw();
                std::printf("  %-44s %s\n", "doorbell_wait (F6 smoke) exits when flag == seq", "yes");
            } catch (const std::exception& e) {
                std::printf("  %-44s *** NO ***: %s\n", "doorbell_wait (F6 smoke) exits when flag == seq", e.what());
                ++bad;
            }
            ctx.free_device(d_flag);
            ctx.free_device(d_seq);
        }

        // doorbell_publish: payload copy + ring in one 1x1024 launch
        {
            const int n = 5, k = 3;
            std::vector<float> hx(n), hw(k), hxo(n, -1.0f), hwo(k, -1.0f);
            std::vector<int32_t> hi(k), hxi(k, -7);
            for (int i = 0; i < n; ++i) hx[i] = 1.5f * (float) i;
            for (int i = 0; i < k; ++i) { hi[i] = 100 + i; hw[i] = 0.5f * (float) i; }
            const uint32_t start = 21;
            float* d_x = ctx.device_alloc<float>(n);
            float* d_w = ctx.device_alloc<float>(k);
            int32_t* d_i = ctx.device_alloc<int32_t>(k);
            float* d_xo = ctx.device_alloc<float>(n);
            float* d_wo = ctx.device_alloc<float>(k);
            int32_t* d_xi = ctx.device_alloc<int32_t>(k);
            uint32_t* d_seq = ctx.device_alloc<uint32_t>(1);
            ctx.q.memcpy(d_x, hx.data(), n * 4);
            ctx.q.memcpy(d_w, hw.data(), k * 4);
            ctx.q.memcpy(d_i, hi.data(), k * 4);
            ctx.q.memcpy(d_seq, &start, 4);
            ctx.wait_and_throw();
            strata::kernels::submit_doorbell_publish(ctx.q, d_x, d_i, d_w, n, k, d_xo, d_xi, d_wo, d_seq);
            ctx.q.memcpy(hxo.data(), d_xo, n * 4);
            ctx.q.memcpy(hxi.data(), d_xi, k * 4);
            ctx.q.memcpy(hwo.data(), d_wo, k * 4);
            uint32_t seq = 0;
            ctx.q.memcpy(&seq, d_seq, 4);
            ctx.wait_and_throw();
            int bad_pub = 0;
            for (int i = 0; i < n; ++i) if (hxo[(size_t) i] != hx[(size_t) i]) ++bad_pub;
            for (int i = 0; i < k; ++i) {
                if (hxi[(size_t) i] != hi[(size_t) i]) ++bad_pub;
                if (hwo[(size_t) i] != hw[(size_t) i]) ++bad_pub;
            }
            if (seq != start + 1u) ++bad_pub;
            std::printf("  %-44s %d mismatches, seq %u -> %u\n", "doorbell_publish (F6 smoke) copies and rings",
                        bad_pub, start, seq);
            if (bad_pub) ++bad;
            ctx.free_device(d_x);
            ctx.free_device(d_w);
            ctx.free_device(d_i);
            ctx.free_device(d_xo);
            ctx.free_device(d_wo);
            ctx.free_device(d_xi);
            ctx.free_device(d_seq);
        }
    }

    std::printf("\nelementwise: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("elementwise_parity OK\n");
    return 0;
}
