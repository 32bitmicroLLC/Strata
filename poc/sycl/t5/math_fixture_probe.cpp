// poc/sycl/t5/math_fixture_probe.cpp -- T5 probe P2b (plans/sycl-phase-a-t5-step-1.md §1.3.1).
//
// `t5/math_probe.cpp` (P2) measured device-vs-host double math on a STRUCTURED grid and failed both
// checks (1-ulp exp differences over ~35% of grid pairs; logf differences including all four fixture
// min_p values).  That grid is not the decision basis: the fixture tail only CONSUMES the double math
// in ordered sum/cum chains compared against `top_p` / the Philox `u`, and in the fp32 min_p threshold
// `sel_logit[0] + logf(min_p)`.  A 1-ulp term difference flips a fixture only if some fixture argument
// sits within a few ulps of its cut.
//
// This probe therefore (1) rebuilds every sampled-chain fixture of `src/kernels/sampler_parity.cpp`
// (same seeds, same data, mirrored verbatim and drift-checked), (2) runs each row through the host
// transcription of the kernel tail (`mirror_select` + the tail arithmetic of `sampler.cu` 282-315 /
// `sampled_tail_warp` 362-420) to collect the EXACT `(x, mx)` arguments feeding
// `exp((double) x - (double) mx)`, the fp32 `thresh` values, and the minimum distance of every `cum`
// to `top_p` and every `u` to its `cum`, and every `sel_logit[i]` to `thresh` (in ulps), (3) computes
// the device `exp` / `logf` of exactly those arguments and compares bit for bit with the host.
//
// GATE (exit 0 = fallback (a) of the parent plan may be adopted - document the affected set and
// proceed to the port; exit 1 = fallback (b), the hand-rolled double exp, is required, or a fixture
// knife-edge exists and is a written report finding):
//   * device exp == host std::exp bit-for-bit on every collected fixture argument; OR
//   * min |cum - top_p| >= 1024 double ulps AND min |u - cum| >= 1024 double ulps (a 1-ulp exp term
//     shifts each cum by <= ~k * 2^-52 relative, k <= 64 terms - 1024 ulps is a wide safety factor); AND
//   * device logf == host (float) std::log on every fixture min_p value; OR
//   * min |sel_logit[i] - thresh| >= 8 fp32 ulps (a 1-ulp thresh shift flips a survivor only within
//     1 ulp of it).
//
// Fixture 18a (graph capture) is out of scope (parent plan §2); 18b's row cap ports.  Fixture 2-5 are
// greedy (no tail) and are not exercised here.
#include "strata/kernels/sampler.hpp"
#include <sycl_compat/test_ctx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace {

struct Pair {
    float x, mx;      // one double-exp argument: exp((double) x - (double) mx)
};

std::vector<Pair> exp_args;
std::set<float> min_p_vals;

double min_top_p_ulps = INFINITY;  // |cum - top_p|, double ulps, over all fixture rows (n/a: never exercised)
double min_draw_ulps = INFINITY;   // |u - cum|, double ulps
double min_min_p_ulps = INFINITY;  // |sel_logit[i] - thresh|, fp32 ulps
int n_rows_probed = 0;
int n_rows_void = 0;              // NaN/inf rows where no cut fires (fixture 16): margins void there

// ---- the kernel's own semantics, mirrored for the fixtures (drift-checked) ----

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:455:455 @ fc36072
    const int NV = 512, NT = 4;
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:137:150 @ fc36072
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:152:158 @ fc36072
float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:274:277 @ fc36072
struct SelList {
    std::vector<int> ids;
    std::vector<float> logit;
};
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:280:306 @ fc36072
SelList mirror_select(const float* l, int nv, const std::vector<int>& window, const strata::kernels::SamplerParams& p,
                      int k) {
    std::vector<float> s((size_t) nv);
    for (int v = 0; v < nv; ++v) {
        int c = 0;
        for (int h : window) c += h == v;
        float x = l[v];
        if (c > 0) {
            if (x <= 0.0f) x *= p.penalty_repeat; else x /= p.penalty_repeat;
            x -= (float) c * p.penalty_freq + (c > 0 ? 1.0f : 0.0f) * p.penalty_present;
        }
        s[(size_t) v] = x;
    }
    SelList out;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = nv;
        float bv = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < nv; ++v)
            if (!taken[(size_t) v] && s[(size_t) v] > bv) { bv = s[(size_t) v]; best = v; }
        const int id = best < nv ? best : 0;
        taken[(size_t) id] = 1;
        out.ids.push_back(id);
        out.logit.push_back(bv);
    }
    return out;
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:346:346 @ fc36072
int sampled_k(int top_k, int nv) { return std::min(nv, (top_k > 0 && top_k < 64) ? top_k : 64); }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:383:388 @ fc36072
std::vector<int> window_of(const std::vector<int>& hist, int hist_len, int t, int last_n) {
    if (hist_len <= 0 || last_n <= 0) return {};
    const int h = std::min(last_n, hist_len);
    const int* row = hist.data() + (size_t) t * hist_len;
    return std::vector<int>(row + (hist_len - h), row + hist_len);
}
// SYCL-MIRROR-END

// ---- ulp metrics ----

static uint64_t dbits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}
static uint32_t fbits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// |a - b| in ulps of the larger-magnitude operand; -1 when either is NaN (a void margin).
double dbl_ulps(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return -1.0;
    const double d = std::fabs(a - b);
    if (d == 0.0) return 0.0;
    const uint64_t mb = dbits(std::max(std::fabs(a), std::fabs(b)));
    const int e = (int) ((mb >> 52) & 0x7FFu);
    if (e == 0) return d;                    // subnormal: no meaningful ulp scale
    return d / std::ldexp(1.0, e - 1075);
}
double f32_ulps(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return -1.0;
    const float d = std::fabs(a - b);
    if (d == 0.0f) return 0.0f;
    const uint32_t mb = fbits(std::max(std::fabs(a), std::fabs(b)));
    const int e = (int) ((mb >> 23) & 0xFFFu);
    if (e == 0) return d;
    return d / std::ldexp(1.0f, e - 150);
}

// ---- the instrumented tail ----

// The kernel tail's host arithmetic (sampler.cu 282-315, instruction for instruction; the warp form
// 362-420 computes the same doubles), with the pick itself dropped: the measurement is the argument
// set and the cut margins.  Margins update the current fixture's stats and the globals.
struct Fix {
    const char* name;
    double top, draw, minp;      // min margins within this fixture (INFINITY: not exercised)
    int rows;
    size_t base_args;
    int args = 0;
};
Fix cur;
std::vector<Fix> fixes;

void start_fixture(const char* name) {
    cur = {name, INFINITY, INFINITY, INFINITY, 0, exp_args.size()};
}
void end_fixture() {
    cur.args = (int) exp_args.size() - (int) cur.base_args;
    fixes.push_back(cur);
}

void tail_probe(const SelList& sel, int k, const strata::kernels::SamplerParams& p, int row) {
    ++cur.rows;
    ++n_rows_probed;
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel.logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmax(mx, sel.logit[(size_t) i]);
    if (p.top_p < 1.0f) {
        for (int i = 0; i < k; ++i) exp_args.push_back({sel.logit[(size_t) i], mx});
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel.logit[(size_t) i] - (double) mx);
        double cum = 0.0;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel.logit[(size_t) i] - (double) mx) / sum;
            const double m = dbl_ulps(cum, (double) p.top_p);
            if (m >= 0.0) { cur.top = std::min(cur.top, m); min_top_p_ulps = std::min(min_top_p_ulps, m); }
            if (cum >= (double) p.top_p) { n_keep = i + 1; break; }
        }
        if (n_keep < p.min_keep) n_keep = p.min_keep < k ? p.min_keep : k;
    }
    if (p.min_p > 0.0f) {
        min_p_vals.insert(p.min_p);
        const float thresh = sel.logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i) {
            const double m = f32_ulps(sel.logit[(size_t) i], thresh);
            if (m >= 0.0) { cur.minp = std::min(cur.minp, m); min_min_p_ulps = std::min(min_min_p_ulps, m); }
            if (sel.logit[(size_t) i] < thresh) { n_keep = i; break; }
        }
    }
    float smx = sel.logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, sel.logit[(size_t) i] * inv_t);
    for (int i = 0; i < n_keep; ++i) exp_args.push_back({sel.logit[(size_t) i] * inv_t, smx});
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) (sel.logit[(size_t) i] * inv_t) - (double) smx);
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    bool void_row = false;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) (sel.logit[(size_t) i] * inv_t) - (double) smx) / sum;
        const double m = dbl_ulps((double) u, cum);
        if (m >= 0.0) {
            cur.draw = std::min(cur.draw, m);
            min_draw_ulps = std::min(min_draw_ulps, m);
        } else
            void_row = true;
    }
    if (void_row) ++n_rows_void;
}

// Convenience: one row through selection + tail.
void probe_row(const float* l, int nv, const std::vector<int>& window, const strata::kernels::SamplerParams& p,
               int row) {
    const SelList sel = mirror_select(l, nv, window, p, sampled_k(64, nv));
    tail_probe(sel, sampled_k(p.top_k, nv), p, row);
}

}      // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    // =====================================================================
    // The sampled-chain fixtures of src/kernels/sampler_parity.cpp, mirrored
    // data-for-data (same seeds, same generation code, drift-checked), each
    // run through the instrumented host tail instead of the kernel.
    // =====================================================================

    // ---- fixture 6: PENALTIES IN THE SAMPLED CHAIN (sampled part) ----
    start_fixture("f6  sampled penalties");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:650:662 @ fc36072
        const int NV2 = 8, NT2 = 2;
        strata::kernels::SamplerParams p;
        p.top_k = 5; p.top_p = 0.9f; p.temperature = 0.8f; p.seed = 9; p.counter = 0;
        p.penalty_last_n = 4; p.penalty_repeat = 3.0f; p.penalty_freq = 0.2f; p.penalty_present = 0.6f;

        std::vector<float> l((size_t) NV2 * NT2, -8.0f);
        std::vector<int> hist((size_t) NT2 * 4, -1);
        for (int t = 0; t < NT2; ++t) {
            float* row = l.data() + (size_t) t * NV2;
            row[0] = 9.0f; row[1] = 4.5f; row[2] = 4.4f; row[3] = 4.3f;   // token 0 leads clean (9 vs 4.5)
            hist[(size_t) t * 4 + 0] = 0;                                  // and falls to 2.2/2.0 penalised
            hist[(size_t) t * 4 + 1] = t == 1 ? 0 : -1;                    // (repeat 3, freq, presence)
        }
    // SYCL-MIRROR-END
        for (int t = 0; t < NT2; ++t) {
            const std::vector<float> row(l.begin() + (size_t) t * NV2, l.begin() + (size_t) (t + 1) * NV2);
            const std::vector<int> h(hist.begin() + (size_t) t * 4, hist.begin() + (size_t) (t + 1) * 4);
            probe_row(row.data(), NV2, h, p, t);
        }
    }
    end_fixture();

    // ---- fixture 7: MIN_P ----
    start_fixture("f7  min_p");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:685:693 @ fc36072
        const int NV3 = 8, NT3 = 2;
        std::vector<float> l((size_t) NV3 * NT3, -8.0f);
        for (int t = 0; t < NT3; ++t) {
            float* row = l.data() + (size_t) t * NV3;
            row[0] = 4.0f; row[1] = 3.5f; row[2] = 3.2f; row[3] = 3.1f;   // gaps keep the cut off the
            row[4] = 2.0f;                                                // logf/rounding knife edge
        }
        strata::kernels::SamplerParams base;
        base.top_k = 6; base.top_p = 1.0f; base.temperature = 0.9f; base.seed = 77;
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:708:709 @ fc36072
        for (float mp : {0.0f, 0.5f, 0.9f}) {
            strata::kernels::SamplerParams q = base; q.min_p = mp;
    // SYCL-MIRROR-END
            for (int t = 0; t < NT3; ++t)
                probe_row(l.data() + (size_t) t * NV3, NV3, {}, q, t);
        }
    }
    end_fixture();

    // ---- fixture 9: ONE PENALTIES STAGE (#53) ----
    start_fixture("f9  one penalties stage");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:762:767 @ fc36072
        const int NT9 = 8000;
        strata::kernels::SamplerParams p;
        p.top_k = 2; p.top_p = 1.0f; p.min_p = 0.0f; p.temperature = 0.7f; p.seed = 53; p.counter = 0;
        p.penalty_last_n = 1; p.penalty_repeat = 1.0f; p.penalty_freq = 0.0f; p.penalty_present = 1.5f;
        std::vector<float> l((size_t) 2 * NT9, 0.0f);
        std::vector<int> hist((size_t) NT9, 0);
    // SYCL-MIRROR-END
        for (int t = 0; t < NT9; ++t)
            probe_row(l.data() + (size_t) t * 2, 2, window_of(hist, 1, t, 1), p, t);
    }
    end_fixture();

    // ---- fixture 10: TOP_P BEFORE MIN_P ----
    start_fixture("f10  top_p then min_p");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:787:793 @ fc36072
        const int NT10 = 256;
        strata::kernels::SamplerParams p;
        p.top_k = 4; p.top_p = 0.75f; p.min_p = 0.3f; p.temperature = 1.0f; p.seed = 10; p.counter = 0;
        const float lp[4] = {std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};
        std::vector<float> row = {lp[0], lp[1], lp[2], lp[3], -30.0f, -30.0f, -30.0f, -30.0f};
        std::vector<float> l;
        for (int t = 0; t < NT10; ++t) l.insert(l.end(), row.begin(), row.end());
    // SYCL-MIRROR-END
        const SelList sel = mirror_select(l.data(), 8, {}, p, sampled_k(64, 8));
        for (int t = 0; t < NT10; ++t) tail_probe(sel, sampled_k(p.top_k, 8), p, t);
    }
    end_fixture();

    // ---- fixture 11: STALE HISTORY WITH last_n = 0 (sampled part) ----
    start_fixture("f11  stale history (sampled)");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:810:817 @ fc36072
        std::mt19937 rng(11); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * 8, -1);
        for (int t = 0; t < NT; ++t) hist[(size_t) t * 8] = 3;

        strata::kernels::SamplerParams p;
        p.top_k = 20; p.top_p = 0.95f; p.temperature = 0.8f; p.seed = 5;
    // SYCL-MIRROR-END
        for (int t = 0; t < NT; ++t)
            probe_row(l.data() + (size_t) t * NV, NV, {}, p, t);
    }
    end_fixture();

    // ---- fixture 12: ONE HISTORY PER ROW (sampled part) ----
    start_fixture("f12  per-row histories (sampled)");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:853:872 @ fc36072
        std::mt19937 rng(12); std::normal_distribution<float> g(0.0f, 1.0f);
        const int TAIL = 5000;                           // longer than the widest window below
        std::vector<int32_t> tail((size_t) TAIL);
        for (auto& v : tail) v = (int32_t) (rng() % NV);
        int observable = 0, rows_checked = 0;
        for (int T : {1, 2, 4, 8}) {
            for (int H : {1, 64, 1024, 4096}) {
                std::vector<int32_t> window((size_t) T);
                for (int t = 0; t < T; ++t) window[(size_t) t] = (int32_t) (100 + 37 * t);   // distinct, in range
                std::vector<int32_t> rows((size_t) T * H);
                strata::kernels::penalty_rows(tail.data(), TAIL, window.data(), T, H, rows.data());
                std::vector<float> l((size_t) T * NV);
                for (auto& v : l) v = g(rng);
                for (int t = 0; t < T; ++t) l[(size_t) t * NV + window[(size_t) t]] = 5.0f;
                strata::kernels::SamplerParams gp;
                gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
                gp.penalty_last_n = H; gp.penalty_present = 4.0f;
                strata::kernels::SamplerParams sp2;
                sp2.top_k = 20; sp2.top_p = 0.9f; sp2.temperature = 0.7f; sp2.seed = 1000 + (uint64_t) H;
                sp2.counter = 77; sp2.penalty_last_n = H; sp2.penalty_present = 4.0f; sp2.penalty_repeat = 1.1f;
    // SYCL-MIRROR-END
                (void) observable; (void) rows_checked;
                for (int t = 0; t < T; ++t) {
                    const std::vector<int> own(rows.begin() + (size_t) t * H, rows.begin() + (size_t) (t + 1) * H);
                    probe_row(l.data() + (size_t) t * NV, NV, own, sp2, t);
                }
            }
        }
    }
    end_fixture();

    // ---- fixture 13: HISTORY IDS OUTSIDE THE VOCABULARY (sampled part) ----
    start_fixture("f13  out-of-vocab history ids (sampled)");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:900:919 @ fc36072
        std::mt19937 rng(13); std::normal_distribution<float> g(0.0f, 1.0f);
        const int H = 16;
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng);
        std::vector<int> hist((size_t) NT * H, -1), valid_only((size_t) NT * H, -1);
        const int junk[] = {NV, NV + 1000, 0x7fffffff, -5, 1 << 20};
        for (int t = 0; t < NT; ++t) {
            for (int j = 0; j < H; ++j) {
                const bool bogus = j % 3 == 0;
                const int v = bogus ? junk[(size_t) (j / 3) % 5] : (int) (rng() % NV);
                hist[(size_t) t * H + j] = v;
                if (!bogus) valid_only[(size_t) t * H + j] = v;
            }
            l[(size_t) t * NV + hist[(size_t) t * H + 1]] = 4.0f;     // a penalised favourite, so penalties matter
        }
        strata::kernels::SamplerParams gp;
        gp.greedy = true; gp.temperature = 0.0f; gp.top_k = 0; gp.top_p = 1.0f;
        gp.penalty_last_n = H; gp.penalty_present = 3.0f;
        strata::kernels::SamplerParams sp2 = gp;
        sp2.greedy = false; sp2.temperature = 0.8f; sp2.top_k = 20; sp2.top_p = 0.95f; sp2.seed = 13;
    // SYCL-MIRROR-END
        for (int t = 0; t < NT; ++t) {
            const std::vector<int> ok(valid_only.begin() + (size_t) t * H, valid_only.begin() + (size_t) (t + 1) * H);
            probe_row(l.data() + (size_t) t * NV, NV, ok, sp2, t);
        }
    }
    end_fixture();

    // ---- fixture 14: THE top_k CONTRACT ----
    start_fixture("f14  top_k contract");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:944:948 @ fc36072
        std::mt19937 rng(14); std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> l((size_t) NV * NT);
        for (auto& v : l) v = g(rng) * 0.3f;             // flat: the 64-wide list matters to the draw
        strata::kernels::SamplerParams p64;
        p64.top_k = 64; p64.top_p = 1.0f; p64.temperature = 1.5f; p64.seed = 14;
    // SYCL-MIRROR-END
        for (int t = 0; t < NT; ++t) {
            const float* row = l.data() + (size_t) t * NV;
            for (int top_k : {64, 0, 100, -3}) {         // the four launches of the fixture (0/100/-3 mean 64)
                strata::kernels::SamplerParams p = p64;
                p.top_k = top_k;
                probe_row(row, NV, {}, p, t);
            }
        }
    }
    end_fixture();

    // ---- counter segmentation (the batch vs singles draws) ----
    start_fixture("counter  batch == singles");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1001:1002 @ fc36072
        constexpr int count = 32, vocab = 16;
        std::vector<float> uniform(count * vocab, 0.0f);
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1008:1009 @ fc36072
        strata::kernels::SamplerParams p;
        p.top_k = vocab; p.top_p = 1.0f; p.seed = 123; p.counter = (uint64_t(1) << 32) + 7;
    // SYCL-MIRROR-END
        const SelList sel = mirror_select(uniform.data(), vocab, {}, p, sampled_k(vocab, vocab));
        for (int i = 0; i < count; ++i) {
            strata::kernels::SamplerParams one = p;
            one.counter += (uint64_t) i;                  // the singles launches' per-row counter
            tail_probe(sel, sampled_k(vocab, vocab), one, i);
        }
    }
    end_fixture();

    // ---- fixture 16: THE WHOLE top_k LIST, POSITION BY POSITION, UNDER TIES ----
    start_fixture("f16  top_k list under ties");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1038:1094 @ fc36072
        const float inf = std::numeric_limits<float>::infinity();
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        int wrong = 0, probes = 0, sentinels = 0;
        for (int nv : {248320, 100003, 262144, 262145, 1000}) {
            const int T = 4, H = 64;
            std::mt19937 rng((unsigned) (1600 + nv));
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> l((size_t) nv * T);
            for (auto& v : l) v = std::floor(g(rng) * 4.0f) / 4.0f;          // quarter steps: ties everywhere
            auto spread = [&](int j) { return (int) (((int64_t) j * 7919 + 13) % nv); };
            // row 0: 20 logits at 6.25 and 300 at 6.0 over the whole row, two +inf
            float* r0 = l.data();
            for (int j = 0; j < 320; ++j) r0[spread(j)] = j < 20 ? 6.25f : 6.0f;
            r0[nv / 2] = inf;
            r0[nv - 1] = inf;
            // row 1: nothing above 0, every zero signed by its id's parity (-0 at even ids), one +inf
            float* r1 = l.data() + (size_t) nv;
            for (int v = 0; v < nv; ++v) {
                r1[v] = -std::fabs(r1[v]);
                if (r1[v] == 0.0f) r1[v] = (v & 1) ? 0.0f : -0.0f;
            }
            r1[3] = inf;
            // row 2: -inf everywhere but ten finite logits (two values), two NaN and a +inf: 11 candidates in all
            float* r2 = l.data() + (size_t) 2 * nv;
            for (int v = 0; v < nv; ++v) r2[v] = -inf;
            for (int j = 0; j < 10; ++j) r2[spread(j + 400)] = j < 5 ? 1.0f : 0.5f;
            r2[spread(500)] = qnan;
            r2[spread(501)] = qnan;
            r2[spread(502)] = inf;
            // row 3: penalties.  Window ids on block and warp edges, repeats, and ids outside the vocabulary; the
            // penalised tokens sit at 9.0, which repeat 2 / freq 0.25 / present 0.5 turns into 4.0 - 0.25 x count
            // (3.75, 3.5, ...), values the quarter-step logits share.
            float* r3 = l.data() + (size_t) 3 * nv;
            std::vector<int> hist((size_t) T * H, -1);
            int* h3 = hist.data() + (size_t) 3 * H;
            const int edges[] = {0, 1023, 1024, 4095, 4096, 8191, 8192, 12345, nv / 2 + 1, nv - 2};
            int hn = 0;
            for (int e : edges)
                if (e < nv && e != nv / 2) {
                    h3[hn++] = e;
                    if (hn % 3 == 0) h3[hn++] = e;                              // counted twice
                    r3[e] = 9.0f;
                }
            h3[hn++] = nv;                                                      // ignored: outside
            h3[hn++] = nv + 77;
            h3[hn++] = -5;
            for (int j = 0; j < 30; ++j) r3[spread(j + 600)] = j < 15 ? 3.75f : 3.5f;
            r3[nv / 2] = inf;                                                   // not in the window

            strata::kernels::SamplerParams base;
            base.top_p = 1.0f; base.min_p = 0.0f; base.temperature = 0.8f; base.seed = 16;
            base.penalty_last_n = H; base.penalty_repeat = 2.0f; base.penalty_freq = 0.25f;
            base.penalty_present = 0.5f;
            const int kmax = sampled_k(64, nv);
            std::vector<SelList> lists;
            for (int t = 0; t < T; ++t)
                lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, H), base, kmax));
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1096:1101 @ fc36072
            for (int top_k = 0; top_k <= 66; ++top_k) {
                strata::kernels::SamplerParams p = base;
                p.top_k = top_k == 65 ? 100 : top_k == 66 ? -3 : top_k;       // 0, 100 and -3 mean 64
                p.top_p = (top_k & 1) ? 0.5f : 1.0f;                            // both tail branches (NaN: no cut)
                p.counter = (uint64_t) top_k;
                const int k = sampled_k(p.top_k, nv);
    // SYCL-MIRROR-END
                (void) wrong; (void) probes; (void) sentinels;
                for (int t = 0; t < T; ++t) tail_probe(lists[(size_t) t], k, p, t);
            }
        }
    }
    end_fixture();

    // ---- fixture 17: SAMPLED DRAWS UNDER TIES (one pass per config; the fixture's two-stream
    // loop adds no new tail arguments - stream identity never enters the tail arithmetic) ----
    start_fixture("f17  sampled draws under ties");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1129:1129 @ fc36072
        const int T = 17, H = 64;
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1133:1150 @ fc36072
        for (int nv : {248320, 512}) {
            std::mt19937 rng((unsigned) (1700 + nv));
            std::normal_distribution<float> g(0.0f, 1.5f);
            std::vector<float> l((size_t) nv * T);
            for (auto& v : l) v = std::floor(g(rng) * 2.0f) / 2.0f;
            if (nv == 512)
                for (auto& v : l) v = std::floor(v / 2.0f);                    // whole steps: a few big tie groups
            std::vector<int> hist((size_t) T * H);
            for (int t = 0; t < T; ++t) {
                const float* row = l.data() + (size_t) t * nv;
                const float mx = *std::max_element(row, row + nv);
                std::vector<int> head;
                for (int v = 0; v < nv; ++v)
                    if (row[v] >= mx - 1.0f) head.push_back(v);
                for (int j = 0; j < H; ++j)
                    hist[(size_t) t * H + j] = (j & 1) ? (int) (rng() % (unsigned) nv)
                                                       : head[(size_t) (rng() % (unsigned) head.size())];
            }
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1152:1162 @ fc36072
            int config = 0;
            for (int pen = 0; pen < 2; ++pen) {
                strata::kernels::SamplerParams base;
                base.temperature = 2.5f; base.seed = 17;
                base.penalty_last_n = pen ? H : 0;
                base.penalty_repeat = 1.25f; base.penalty_freq = 0.25f; base.penalty_present = 0.5f;
                const int kmax = sampled_k(64, nv);
                std::vector<SelList> lists;
                for (int t = 0; t < T; ++t)
                    lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, window_of(hist, H, t, base.penalty_last_n),
                                                  base, kmax));
    // SYCL-MIRROR-END
                for (int top_k : {1, 20, 64})
                    for (float top_p : {0.9f, 1.0f})
                        for (float min_p : {0.0f, 0.05f}) {
                            strata::kernels::SamplerParams p = base;       // from 1167:1169, stream loop elided
                            p.top_k = top_k; p.top_p = top_p; p.min_p = min_p;
                            p.counter = (uint64_t) (1000 * ++config);
                            const int k = sampled_k(top_k, nv);
                            for (int t = 0; t < T; ++t) tail_probe(lists[(size_t) t], k, p, t);
                        }
            }
        }
    }
    end_fixture();

    // ---- fixture 18b: THE ROW-CAP FALLBACK (18a's graph capture is out of scope) ----
    start_fixture("f18b  row cap fallback");
    {
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1207:1207 @ fc36072
        const int H = 64;
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1219:1226 @ fc36072
        auto make = [&](int nv, int T, unsigned seed, std::vector<float>& l, std::vector<int>& hist) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> g(0.0f, 1.5f);
            l.assign((size_t) nv * T, 0.0f);
            for (auto& v : l) v = std::floor(g(rng) * 2.0f) / 2.0f;
            hist.assign((size_t) T * H, 0);
            for (auto& h : hist) h = (int) (rng() % (unsigned) nv);
        };
    // SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/sampler_parity.cpp:1262:1271 @ fc36072
        for (int T : {64, 70}) {
            const int nv = 512;
            std::vector<float> l;
            std::vector<int> hist;
            make(nv, T, 1810u + (unsigned) T, l, hist);
            strata::kernels::SamplerParams p;
            p.top_k = 64; p.top_p = 1.0f; p.temperature = 2.5f; p.seed = 18; p.counter = (uint64_t) T;
            const int k = sampled_k(p.top_k, nv);
            std::vector<SelList> lists;
            for (int t = 0; t < T; ++t) lists.push_back(mirror_select(l.data() + (size_t) t * nv, nv, {}, p, k));
    // SYCL-MIRROR-END
            for (int t = 0; t < T; ++t) tail_probe(lists[(size_t) t], k, p, t);
        }
    }
    end_fixture();

    // =====================================================================
    // Device vs host, on exactly the collected fixture arguments.
    // =====================================================================
    std::printf("t5_math_fixture_probe: %d rows probed (%d void), %zu double-exp arguments, %zu distinct min_p values\n",
                n_rows_probed, n_rows_void, exp_args.size(), min_p_vals.size());

    const size_t N = exp_args.size();
    float* dx = c.device_alloc<float>(N);
    float* dm = c.device_alloc<float>(N);
    double* de = c.device_alloc<double>(N);
    std::vector<float> xs(N), mxs(N);
    for (size_t i = 0; i < N; ++i) { xs[i] = exp_args[i].x; mxs[i] = exp_args[i].mx; }
    c.q.memcpy(dx, xs.data(), N * sizeof(float));
    c.q.memcpy(dm, mxs.data(), N * sizeof(float));
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [dx, dm, de](sycl::id<1> id) {
            de[id[0]] = exp((double) dx[id[0]] - (double) dm[id[0]]);
        });
    });
    e.wait_and_throw();
    std::vector<double> got(N);
    c.q.memcpy(got.data(), de, N * sizeof(double));
    c.wait_and_throw();

    int bad = 0;
    for (size_t i = 0; i < N; ++i) {
        const double want = std::exp((double) xs[i] - (double) mxs[i]);
        if (dbits(got[i]) != dbits(want)) {
            if (bad < 8)
                std::printf("    x %g mx %g: device %.17g host %.17g\n", xs[i], mxs[i], got[i], want);
            ++bad;
        }
    }
    const bool exp_exact = bad == 0;
    std::printf("  fixture-arg exp: device vs host std::exp %s (%d of %zu differ)\n",
                exp_exact ? "bit-identical" : "differ", bad, N);
    c.free_device(dx);
    c.free_device(dm);
    c.free_device(de);

    int lbad = 0;
    for (float v : min_p_vals) {
        const float want = (float) std::log(v);
        float gotv;
        float* d1 = c.device_alloc<float>(1);
        c.q.memcpy(d1, &v, sizeof(float));
        auto e2 = c.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [d1](sycl::id<1>) { d1[0] = logf(d1[0]); });
        });
        e2.wait_and_throw();
        c.q.memcpy(&gotv, d1, sizeof(float));
        c.wait_and_throw();
        c.free_device(d1);
        if (fbits(gotv) != fbits(want)) {
            std::printf("    min_p %g: device %a host %a\n", (double) v, gotv, want);
            ++lbad;
        }
    }
    const bool log_exact = lbad == 0;
    std::printf("  fixture min_p logf: device vs host (float) std::log %s (%d of %zu differ)\n",
                log_exact ? "bit-identical" : "differ", lbad, min_p_vals.size());

    auto fmt_margin = [](double m) -> std::string { return m == INFINITY ? "n/a" : std::to_string(m); };
    std::printf("  per-fixture cut margins (ulps; n/a = not exercised):\n");
    for (const Fix& f : fixes)
        std::printf("    %-34s %5d rows %6d args  top_p %-12s  draw %-12s  min_p %-8s\n", f.name, f.rows,
                    f.args, fmt_margin(f.top).c_str(), fmt_margin(f.draw).c_str(), fmt_margin(f.minp).c_str());

    // The gate from the file header: bit-exact, or margins far beyond a 1-ulp term difference.
    // A stage no fixture row exercises (INFINITY margin) cannot be flipped by any device difference.
    const double gate_dbl = 1024.0, gate_f32 = 8.0;
    const bool exp_gate = exp_exact || (min_top_p_ulps >= gate_dbl && min_draw_ulps >= gate_dbl);
    const bool log_gate = log_exact || min_p_vals.empty() || min_min_p_ulps >= gate_f32;
    const bool ok = exp_gate && log_gate;
    std::printf("  margins: top_p >= %s double ulps, draw >= %s double ulps, min_p >= %s fp32 ulps "
                "(gates: exp %s, log %s)\n", fmt_margin(min_top_p_ulps).c_str(), fmt_margin(min_draw_ulps).c_str(),
                fmt_margin(min_min_p_ulps).c_str(), exp_gate ? "PASS" : "FAIL", log_gate ? "PASS" : "FAIL");
    std::printf("t5_math_fixture_probe: %s\n",
                ok ? "GATE PASS - plain exp/logf are safe for the ported fixtures (fallback (a): document the affected set)"
                   : "GATE FAIL - fallback (b): hand-rolled double exp (and/or a written fixture knife-edge finding)");
    return ok ? 0 : 1;
}
