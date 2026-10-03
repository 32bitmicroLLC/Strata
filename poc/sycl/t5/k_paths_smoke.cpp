// poc/sycl/t5/k_paths_smoke.cpp -- T5 step 3 scenario-matrix smoke
// (plans/sycl-phase-a-t5-step-3.md §3.6).  NOT fixture parity (that is Step 4's
// k_sampler_parity): this proves every dispatcher branch and every path the
// Step 2 smoke left unexercised, each against a host serial reference - the
// k top_k rounds of the penalised strict argmax (the kernels' (value,
// lower-index) total order) plus the mirrored tail (278:315) with host math.
//
// Scenario matrix (nv = vocab, nt = rows):
//   A  64 x   512   all four paths; default = split at the kSplitMaxRows
//                  boundary (the Step 2 shape, unchanged)
//   B  16 x 12288   multi-block split: n_blocks = 3, (t,b) item mapping,
//                  per-block bitmaps, multi-list merge in split_merge
//   C 100 x   512   n_tokens > kSplitMaxRows -> dispatcher one-block fallback
//   D   4 x 262200  n_blocks = 65 > kSplitMaxBlocks -> one-block fallback
//   F   4 x 248320  worst local-memory shapes (7,760-word bitmap at 1,024
//                  threads) actually launch on Arc; default = split with
//                  n_blocks = 61, near the merge's capacity
//   E     host     bad-argument throws (n_tokens = 0, missing history with
//                  penalties), temperature = 0 dispatching to the greedy
//                  kernel, degenerate 1 x 4 shape
//
// The parent runs each scenario's default path (and greedy where noted)
// in-process; the forced old / one-block paths run in forked children with a
// fresh SYCL runtime each (sampled_path() caches its env read in a
// function-static, so one process can take only one sampled path).  The
// forks happen before the parent touches the SYCL runtime:
//   STRATA_OLD_SAMPLER child:      scenarios A, B
//   STRATA_SAMPLER_ONE_BLOCK child: scenarios A, C, D, F
#include "sycl_compat/test_ctx.hpp"

#include <strata/kernels/sampler.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace sc = strata::sycl_compat;
using sc::ctx;
using strata::kernels::SamplerParams;

// The SYCL submit has no header yet (Step 4 adds one with the parity driver);
// declare it here against the definition in k_sampler.
namespace strata::kernels {
sycl::event submit_sample_tokens(sycl::queue& q, const float* logits, int n_tokens, int n_vocab,
                                const int* history, int history_len, const SamplerParams& p, int* out);
}

namespace {

constexpr int HL = 16;    // history length per row, every scenario
constexpr int PICK_CAP = 128;  // >= the largest scenario's row count

struct Scenario {
    const char* name;
    int nt;
    int nv;
    int hl;
    int seed;
};

struct Pack {
    int ok;
    int nt;
    int picks[PICK_CAP];
};

std::vector<Scenario> scenarios() {
    return {
        {"A", 64, 512, HL, 12345},     // the Step 2 shape (same data as before)
        {"B", 16, 12288, HL, 23456},   // n_blocks = 3
        {"C", 100, 512, HL, 34567},    // nt > kSplitMaxRows
        {"D", 4, 262200, HL, 45678},   // n_blocks = 65 > kSplitMaxBlocks
        {"F", 4, 248320, HL, 56789},   // worst local-memory shape; n_blocks = 61
    };
}

SamplerParams base_params() {
    SamplerParams p;
    p.top_k = 8;
    p.top_p = 0.9f;
    p.min_p = 0.0f;
    p.temperature = 0.8f;
    p.penalty_last_n = HL;
    p.penalty_repeat = 1.25f;
    p.penalty_freq = 0.25f;
    p.penalty_present = 0.5f;
    p.seed = 42;
    p.counter = 0;
    return p;
}

void scenario_data(const Scenario& s, std::vector<float>& l, std::vector<int>& h) {
    std::mt19937 rng((uint32_t) s.seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    l.resize((size_t) s.nt * s.nv);
    for (auto& x : l) x = dist(rng);
    const int mod = s.nv < 32 ? s.nv : 32;  // heavy repetition in a small window
    h.resize((size_t) s.nt * s.hl);
    for (int t = 0; t < s.nt; ++t)
        for (int i = 0; i < s.hl; ++i) h[(size_t) t * s.hl + i] = (t * 5 + i) % mod;
}

int host_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

// glue: the same 64x32->hi32 helper the kernel file defines, so the mirrored
// philox body stays verbatim
static inline uint32_t __umulhi(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * b) >> 32); }

float host_penalize(float logit, int count, const SamplerParams& p) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/sampler.cu:74:78 @ bb7e783
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
    // SYCL-MIRROR-END
}

uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
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

float philox_uniform_h(uint64_t seed, uint64_t counter) {
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

// The serial sampled reference: k rounds of the penalised strict argmax (the
// same (value, lower-index) total order the kernels' tournaments implement),
// then the mirrored tail (278:315) with HOST math.  The kernel-side double
// exp differs from the host by 1 ulp on some arguments (t5_math_fixture_probe);
// the smoke's random data keeps every cut far off the knife edge, so a row
// disagreeing here is a real kernel bug, not the documented P2 gap.
int host_sampled(const float* l, const int* hrow, const SamplerParams& p, int t, int nv, int hl) {
    int k = (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64;
    if (k > nv) k = nv;
    std::vector<int> sel_ids((size_t) k);
    std::vector<float> sel_logit((size_t) k);
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int best = nv;
        for (int v = 0; v < nv; ++v) {
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[(size_t) j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = host_penalize(l[v], host_count(hrow, hl, v), p);
            if (s > bv) { bv = s; best = v; }
        }
        sel_ids[(size_t) i] = (best < nv) ? best : 0;
        sel_logit[(size_t) i] = bv;
    }
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = std::fmaxf(mx, sel_logit[(size_t) i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::logf(p.min_p);
        for (int i = 0; i < n_keep; ++i) if (sel_logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform_h(p.seed, (uint64_t) p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[(size_t) n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}

// The serial greedy reference: the penalised argmax, ties to the lower index -
// the same strict-> plus lower-index tie order the kernel's tournament implements.
int host_greedy(const float* l, const int* hrow, const SamplerParams& p, int nv, int hl) {
    float bv = -INFINITY;
    int best = nv;
    for (int v = 0; v < nv; ++v) {
        const float s = host_penalize(l[v], host_count(hrow, hl, v), p);
        if (s > bv) { bv = s; best = v; }
    }
    return best;
}

int run_default(ctx& c, const std::vector<float>& l, const std::vector<int>& h, const Scenario& s, std::vector<int>& picks,
                const SamplerParams& p) {
    float* dl = c.device_alloc<float>(l.size());
    int* dh = c.device_alloc<int>(h.size());
    int* dout = c.device_alloc<int>((size_t) s.nt);
    c.q.memcpy(dl, l.data(), l.size() * sizeof(float));
    c.q.memcpy(dh, h.data(), h.size() * sizeof(int));
    submit_sample_tokens(c.q, dl, s.nt, s.nv, dh, s.hl, p, dout).wait_and_throw();
    picks.resize((size_t) s.nt);
    c.q.memcpy(picks.data(), dout, (size_t) s.nt * sizeof(int));
    c.wait_and_throw();
    return 0;
}

int run_greedy(ctx& c, const std::vector<float>& l, const std::vector<int>& h, const Scenario& s, std::vector<int>& picks) {
    float* dl = c.device_alloc<float>(l.size());
    int* dh = c.device_alloc<int>(h.size());
    int* dout = c.device_alloc<int>((size_t) s.nt);
    c.q.memcpy(dl, l.data(), l.size() * sizeof(float));
    c.q.memcpy(dh, h.data(), h.size() * sizeof(int));
    SamplerParams p = base_params();
    p.greedy = true;
    submit_sample_tokens(c.q, dl, s.nt, s.nv, dh, s.hl, p, dout).wait_and_throw();
    picks.resize((size_t) s.nt);
    c.q.memcpy(picks.data(), dout, (size_t) s.nt * sizeof(int));
    c.wait_and_throw();
    return 0;
}

// One forced sampled path (old or one_block, via env) over several scenarios.
void child(const char* env, int write_fd, const std::vector<Scenario>& mine) {
    ::close(write_fd ^ 1);
    ::setenv(env, "1", 1);
    try {
        ctx c;
        const SamplerParams p = base_params();
        for (const Scenario& s : mine) {
            Pack pack{};
            pack.nt = s.nt;
            try {
                std::vector<float> l;
                std::vector<int> h;
                scenario_data(s, l, h);
                float* dl = c.device_alloc<float>(l.size());
                int* dh = c.device_alloc<int>(h.size());
                int* dout = c.device_alloc<int>((size_t) s.nt);
                c.q.memcpy(dl, l.data(), l.size() * sizeof(float));
                c.q.memcpy(dh, h.data(), h.size() * sizeof(int));
                submit_sample_tokens(c.q, dl, s.nt, s.nv, dh, s.hl, p, dout).wait_and_throw();
                c.q.memcpy(pack.picks, dout, (size_t) s.nt * sizeof(int));
                c.wait_and_throw();
                pack.ok = 1;
                std::printf("k_sampler_smoke %s child (%s): %d rows ok\n", s.name, env, s.nt);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "k_sampler_smoke %s child (%s): %s\n", s.name, env, e.what());
            }
            const ssize_t n = ::write(write_fd, &pack, sizeof pack);
            (void) n;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "k_sampler_smoke %s child: %s\n", env, e.what());
    }
    ::close(write_fd);
    std::fflush(stdout);  // std::_Exit skips the stdio flush
    std::_Exit(0);
}

bool read_pack(int fd, Pack& pk) {
    size_t got = 0;
    while (got < sizeof pk) {
        const ssize_t n = ::read(fd, (char*) &pk + got, sizeof pk - got);
        if (n <= 0) return false;
        got += (size_t) n;
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<Scenario> sc = scenarios();
    std::vector<std::vector<float>> L(sc.size());
    std::vector<std::vector<int>> H(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) scenario_data(sc[i], L[i], H[i]);

    int p_old[2], p_ob[2];
    if (::pipe(p_old) != 0 || ::pipe(p_ob) != 0) { std::perror("pipe"); return 1; }
    std::vector<Scenario> ob_mine = {sc[0], sc[2], sc[3], sc[4]};  // A, C, D, F
    const pid_t c_old = ::fork();
    if (c_old < 0) { std::perror("fork"); return 1; }
    if (c_old == 0) child("STRATA_OLD_SAMPLER", p_old[1], {sc[0], sc[1]});  // A, B
    const pid_t c_ob = ::fork();
    if (c_ob < 0) { std::perror("fork"); return 1; }
    if (c_ob == 0) child("STRATA_SAMPLER_ONE_BLOCK", p_ob[1], ob_mine);
    ::close(p_old[1]);
    ::close(p_ob[1]);

    int fails = 0;
    try {
        ctx c;

        // E: host-side dispatcher behaviour - the throws happen before any
        // device work, so they need no data of their own
        {
            SamplerParams p = base_params();
            float* dl = c.device_alloc<float>(1);
            int* dh = c.device_alloc<int>(1);
            int* dout = c.device_alloc<int>(1);
            bool t1 = false, t2 = false;
            try {
                submit_sample_tokens(c.q, dl, 0, 512, dh, HL, p, dout);
            } catch (const std::exception&) {
                t1 = true;
            }
            try {
                submit_sample_tokens(c.q, dl, 4, 512, nullptr, 0, p, dout);
            } catch (const std::exception&) {
                t2 = true;
            }
            if (!t1) { std::fprintf(stderr, "k_sampler_smoke E: n_tokens = 0 did not throw\n"); ++fails; }
            if (!t2) { std::fprintf(stderr, "k_sampler_smoke E: missing history with penalties did not throw\n"); ++fails; }

            // temperature = 0 dispatches to the greedy kernel (mirrored 847)
            Scenario t0{"E3", 16, 512, HL, 67890};
            std::vector<float> l;
            std::vector<int> h;
            scenario_data(t0, l, h);
            SamplerParams p0 = base_params();
            p0.temperature = 0.0f;
            std::vector<int> picks;
            run_default(c, l, h, t0, picks, p0);
            int bad = 0;
            for (int t = 0; t < t0.nt; ++t)
                if (picks[(size_t) t] != host_greedy(l.data() + (size_t) t * t0.nv, h.data() + (size_t) t * t0.hl, base_params(), t0.nv, t0.hl))
                    ++bad;
            if (bad) { std::fprintf(stderr, "k_sampler_smoke E: temperature = 0 (greedy dispatch): %d of %d rows differ\n", bad, t0.nt); ++fails; }

            // degenerate shape: 1 row x 4 vocab (all paths; the kernel's
            // sampled_k clamps top_k to the vocab, so the reference must use
            // the same clamp: k = min(8, 4) = 4)
            Scenario dg{"E4", 1, 4, HL, 78901};
            std::vector<float> ld;
            std::vector<int> hd;
            scenario_data(dg, ld, hd);
            std::vector<int> pd;
            run_default(c, ld, hd, dg, pd, base_params());
            std::vector<int> gd;
            run_greedy(c, ld, hd, dg, gd);
            const int want_s = host_sampled(ld.data(), hd.data(), base_params(), 0, dg.nv, dg.hl);
            const int want_g = host_greedy(ld.data(), hd.data(), base_params(), dg.nv, dg.hl);
            if (pd[0] != want_s || pd[0] < 0 || pd[0] >= dg.nv) { std::fprintf(stderr, "k_sampler_smoke E: degenerate sampled pick %d != %d\n", pd[0], want_s); ++fails; }
            if (gd[0] != want_g) { std::fprintf(stderr, "k_sampler_smoke E: degenerate greedy pick %d != %d\n", gd[0], want_g); ++fails; }
        }

        // Scenarios A, B, C, D, F: parent default (+ greedy for A, B, F),
        // children forced old (A, B) / one_block (A, C, D, F)
        std::vector<Pack> old_pk(2), ob_pk(4);
        std::vector<int> dflt, grdy;
        for (size_t i = 0; i < sc.size(); ++i) {
            const Scenario& s = sc[i];
            const std::vector<float>& l = L[i];
            const std::vector<int>& h = H[i];
            const bool greedy = (s.name[0] == 'A' || s.name[0] == 'B' || s.name[0] == 'F');

            run_default(c, l, h, s, dflt, base_params());
            std::vector<int> greedy_picks;
            if (greedy) run_greedy(c, l, h, s, greedy_picks);

            // the serial sampled reference, row by row (host math)
            std::vector<int> ref((size_t) s.nt);
            const SamplerParams p = base_params();
            for (int t = 0; t < s.nt; ++t)
                ref[(size_t) t] = host_sampled(l.data() + (size_t) t * s.nv, h.data() + (size_t) t * s.hl, p, t, s.nv, s.hl);

            auto valid = [&s](const std::vector<int>& v) {
                for (int x : v) if (x < 0 || x >= s.nv) return false;
                return true;
            };
            auto neq = [](const std::vector<int>& a, const std::vector<int>& b) {
                int n = 0;
                for (size_t t = 0; t < a.size(); ++t) if (a[t] != b[t]) ++n;
                return n;
            };

            int bad = 0;
            if (!valid(dflt)) { std::fprintf(stderr, "k_sampler_smoke %s: default produced an out-of-range pick\n", s.name); ++fails; }
            bad += neq(dflt, ref);
            if (bad) { std::fprintf(stderr, "k_sampler_smoke %s: default vs serial ref: %d of %d rows differ\n", s.name, bad, s.nt); ++fails; }
            if (greedy) {
                if (!valid(greedy_picks)) { std::fprintf(stderr, "k_sampler_smoke %s: greedy produced an out-of-range pick\n", s.name); ++fails; }
                int gbad = 0;
                for (int t = 0; t < s.nt; ++t)
                    if (greedy_picks[(size_t) t] != host_greedy(l.data() + (size_t) t * s.nv, h.data() + (size_t) t * s.hl, base_params(), s.nv, s.hl))
                        ++gbad;
                if (gbad) { std::fprintf(stderr, "k_sampler_smoke %s: greedy vs host serial: %d of %d rows differ\n", s.name, gbad, s.nt); ++fails; }
            }
            // child packs, in scenario order: old child A,B; one-block child A,C,D,F
            if (s.name[0] == 'A' || s.name[0] == 'B') {
                Pack& pk = old_pk[s.name[0] == 'A' ? 0 : 1];
                if (!read_pack(p_old[0], pk)) { std::fprintf(stderr, "k_sampler_smoke %s: no result from the old-path child\n", s.name); ++fails; continue; }
                if (!pk.ok) { std::fprintf(stderr, "k_sampler_smoke %s: old-path child failed\n", s.name); ++fails; }
                std::vector<int> oldv(pk.picks, pk.picks + pk.nt);
                if (!valid(oldv)) { std::fprintf(stderr, "k_sampler_smoke %s: old produced an out-of-range pick\n", s.name); ++fails; }
                int obad = neq(oldv, ref);
                if (obad) { std::fprintf(stderr, "k_sampler_smoke %s: old vs serial ref: %d of %d rows differ\n", s.name, obad, s.nt); ++fails; }
            }
            if (s.name[0] == 'A' || s.name[0] == 'C' || s.name[0] == 'D' || s.name[0] == 'F') {
                static int ob_i = 0;  // A, C, D, F arrival order
                Pack& pk = ob_pk[ob_i++];
                if (!read_pack(p_ob[0], pk)) { std::fprintf(stderr, "k_sampler_smoke %s: no result from the one-block child\n", s.name); ++fails; continue; }
                if (!pk.ok) { std::fprintf(stderr, "k_sampler_smoke %s: one-block child failed\n", s.name); ++fails; }
                std::vector<int> obv(pk.picks, pk.picks + pk.nt);
                if (!valid(obv)) { std::fprintf(stderr, "k_sampler_smoke %s: one_block produced an out-of-range pick\n", s.name); ++fails; }
                int bad2 = neq(obv, ref);
                if (bad2) { std::fprintf(stderr, "k_sampler_smoke %s: one_block vs serial ref: %d of %d rows differ\n", s.name, bad2, s.nt); ++fails; }
                // the fallback scenarios also cross-check default against the
                // forced one-block picks (consistency, see the step-3 plan: both
                // branches are correct, so this is not a branch identifier)
                if (s.name[0] == 'C' || s.name[0] == 'D') {
                    int cd = neq(dflt, obv);
                    if (cd) { std::fprintf(stderr, "k_sampler_smoke %s: default vs forced one_block: %d of %d rows differ\n", s.name, cd, s.nt); ++fails; }
                }
            }
            auto pr8 = [](const std::vector<int>& v, const char* n, const char* sn) {
                const int m = (int) v.size() < 8 ? (int) v.size() : 8;
                std::printf("k_sampler_smoke %s %s: ", sn, n);
                for (int i = 0; i < m; ++i) std::printf(i < m - 1 ? "%d " : "%d\n", v[(size_t) i]);
            };
            pr8(dflt, "default", s.name);
            if (greedy) pr8(greedy_picks, "greedy", s.name);
        }
        ::waitpid(c_old, nullptr, 0);
        ::waitpid(c_ob, nullptr, 0);

        std::printf("k_sampler_smoke: %s\n", fails == 0 ? "PASS" : "*** FAIL ***");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "k_sampler_smoke: %s\n", e.what());
        fails = 1;
    }
    return fails;
}
