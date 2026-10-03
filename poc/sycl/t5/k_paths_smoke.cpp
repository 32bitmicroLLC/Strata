// poc/sycl/t5/k_paths_smoke.cpp -- T5 step 2 one-shot path smoke
// (plans/sycl-phase-a-t5-step-2.md §2.5).  NOT fixture parity (that is Step 4's
// k_sampler_parity): this proves the k_sampler library links, all four paths
// launch without crashing or deadlocking, the greedy picks equal the host
// serial argmax with penalties, and the three sampled paths agree with each
// other row for row (they share the tail math, so a disagreement is a bug).
//
// Fixed data, no fixtures: 64 rows x 512 vocab (64 rows == kSplitMaxRows, so
// the default split path is actually taken), one 16-token history window per
// row with heavy repetition to exercise the penalties.
//
// The other two sampled paths run in forked children (setenv + a fresh SYCL
// runtime each): sampled_path() caches its env read in a function-static, so
// one process can only take one sampled path.  The fork happens before either
// parent or child touches the SYCL runtime.
#include "sycl_compat/test_ctx.hpp"

#include <strata/kernels/sampler.hpp>

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

constexpr int NT = 64;
constexpr int NV = 512;
constexpr int HL = 16;

struct Pack {
    int ok;
    int picks[NT];
};

SamplerParams smoke_params() {
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

void smoke_data(std::vector<float>& l, std::vector<int>& h) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    l.resize((size_t) NT * NV);
    for (auto& x : l) x = dist(rng);
    h.resize((size_t) NT * HL);
    for (int t = 0; t < NT; ++t)
        for (int i = 0; i < HL; ++i) h[(size_t) t * HL + i] = (t * 5 + i) % 32;
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

uint32_t philox4x32_round_h(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
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
        philox4x32_round_h(c0, c1, c2, c3, (uint32_t) i, 0u);
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
int host_sampled(const float* l, const int* hrow, const SamplerParams& p, int t) {
    int k = (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64;
    if (k > NV) k = NV;
    std::vector<int> sel_ids((size_t) k);
    std::vector<float> sel_logit((size_t) k);
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int best = NV;
        for (int v = 0; v < NV; ++v) {
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[(size_t) j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = host_penalize(l[v], host_count(hrow, HL, v), p);
            if (s > bv) { bv = s; best = v; }
        }
        sel_ids[(size_t) i] = (best < NV) ? best : 0;
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
int host_greedy(const float* l, const int* hrow, const SamplerParams& p) {
    float bv = -INFINITY;
    int best = NV;
    for (int v = 0; v < NV; ++v) {
        const float s = host_penalize(l[v], host_count(hrow, HL, v), p);
        if (s > bv) { bv = s; best = v; }
    }
    return best;
}

void child(const char* env, int write_fd) {
    ::close(write_fd ^ 1);
    ::setenv(env, "1", 1);
    Pack pack{};
    try {
        ctx c;
        std::vector<float> lh;
        std::vector<int> hh;
        smoke_data(lh, hh);
        const SamplerParams p = smoke_params();
        float* dl = c.device_alloc<float>(lh.size());
        int* dh = c.device_alloc<int>(hh.size());
        int* dout = c.device_alloc<int>(NT);
        c.q.memcpy(dl, lh.data(), lh.size() * sizeof(float));
        c.q.memcpy(dh, hh.data(), hh.size() * sizeof(int));
        submit_sample_tokens(c.q, dl, NT, NV, dh, HL, p, dout).wait_and_throw();
        c.q.memcpy(pack.picks, dout, (size_t) NT * sizeof(int));
        c.wait_and_throw();
        pack.ok = 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s child: %s\n", env, e.what());
    }
    const ssize_t n = ::write(write_fd, &pack, sizeof pack);
    (void) n;
    ::close(write_fd);
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
    std::vector<float> lh;
    std::vector<int> hh;
    smoke_data(lh, hh);
    const SamplerParams p = smoke_params();

    int p_old[2], p_ob[2];
    if (::pipe(p_old) != 0 || ::pipe(p_ob) != 0) { std::perror("pipe"); return 1; }
    const pid_t c_old = ::fork();
    if (c_old < 0) { std::perror("fork"); return 1; }
    if (c_old == 0) child("STRATA_OLD_SAMPLER", p_old[1]);
    const pid_t c_ob = ::fork();
    if (c_ob < 0) { std::perror("fork"); return 1; }
    if (c_ob == 0) child("STRATA_SAMPLER_ONE_BLOCK", p_ob[1]);
    ::close(p_old[1]);
    ::close(p_ob[1]);

    int fails = 0;
    try {
        ctx c;
        float* dl = c.device_alloc<float>(lh.size());
        int* dh = c.device_alloc<int>(hh.size());
        int* dout = c.device_alloc<int>(NT);
        c.q.memcpy(dl, lh.data(), lh.size() * sizeof(float));
        c.q.memcpy(dh, hh.data(), hh.size() * sizeof(int));

        // The default path: the split top_k (n_blocks == 1, NT == kSplitMaxRows).
        submit_sample_tokens(c.q, dl, NT, NV, dh, HL, p, dout).wait_and_throw();
        std::vector<int> split((size_t) NT);
        c.q.memcpy(split.data(), dout, (size_t) NT * sizeof(int));
        c.wait_and_throw();

        // The greedy path (p.greedy short-circuits the sampled paths).
        SamplerParams pg = p;
        pg.greedy = true;
        submit_sample_tokens(c.q, dl, NT, NV, dh, HL, pg, dout).wait_and_throw();
        std::vector<int> greedy((size_t) NT);
        c.q.memcpy(greedy.data(), dout, (size_t) NT * sizeof(int));
        c.wait_and_throw();

        Pack old_pk{}, ob_pk{};
        if (!read_pack(p_old[0], old_pk)) std::fprintf(stderr, "k_sampler_smoke: no result from the old-path child\n");
        if (!read_pack(p_ob[0], ob_pk)) std::fprintf(stderr, "k_sampler_smoke: no result from the one-block child\n");

        // The serial sampled reference, row by row (host math).
        std::vector<int> ref((size_t) NT);
        for (int t = 0; t < NT; ++t)
            ref[(size_t) t] = host_sampled(lh.data() + (size_t) t * NV, hh.data() + (size_t) t * HL, p, t);
        int st = 0;
        ::waitpid(c_old, &st, 0);
        ::waitpid(c_ob, &st, 0);

        // 1) every pick is a valid token index
        auto valid = [](const std::vector<int>& v) {
            for (int x : v) if (x < 0 || x >= NV) return false;
            return true;
        };
        // 2) greedy == the host serial argmax with penalties
        int gbad = 0;
        for (int t = 0; t < NT; ++t) {
            const int want = host_greedy(lh.data() + (size_t) t * NV, hh.data() + (size_t) t * HL, p);
            if (greedy[(size_t) t] != want) ++gbad;
        }
        // 3) each sampled path matches the serial reference row for row
        int sbad = 0, obad = 0, oab = 0;
        for (int t = 0; t < NT; ++t) {
            if (split[t] != ref[(size_t) t]) ++sbad;
            if (old_pk.picks[t] != ref[(size_t) t]) ++obad;
            if (ob_pk.picks[t] != ref[(size_t) t]) ++oab;
        }

        const bool ok = old_pk.ok && ob_pk.ok && valid(split) && valid(greedy) && gbad == 0 && sbad == 0 && obad == 0 &&
                        oab == 0;
        auto pr8 = [](const std::vector<int>& v, const char* n) {
            std::printf("k_sampler_smoke: %s: ", n);
            for (int i = 0; i < 8; ++i) std::printf(i < 7 ? "%d " : "%d\n", v[(size_t) i]);
        };
        pr8(split, "split");
        pr8(greedy, "greedy");
        std::printf("k_sampler_smoke: old: ");
        for (int i = 0; i < 8; ++i) std::printf(i < 7 ? "%d " : "%d\n", old_pk.picks[i]);
        std::printf("k_sampler_smoke: oneblock: ");
        for (int i = 0; i < 8; ++i) std::printf(i < 7 ? "%d " : "%d\n", ob_pk.picks[i]);
        std::printf("k_sampler_smoke: greedy vs host serial: %d of %d differ; vs serial ref - split: %d, old: %d, one_block: %d\n",
                    gbad, NT, sbad, obad, oab);
        std::printf("k_sampler_smoke: %s\n", ok ? "PASS" : "*** FAIL ***");
        fails = ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "k_sampler_smoke: %s\n", e.what());
        fails = 1;
    }
    return fails;
}
