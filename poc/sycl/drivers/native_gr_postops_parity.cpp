// poc/sycl/drivers/native_gr_postops_parity.cpp -- B1 (plans/sycl-phase-b-steps-b1.md,
// step 2.4, finding F1): a NEW dedicated parity driver for
// src/kernels/cuda/native_gr_postops.cu.  There is no CUDA-side parity driver
// for these entries (they are exercised only through gr.cu, a B4 file), so
// this driver carries its own reference and its own contract.
//
// CONTRACT -- what this test can and cannot see:
//   CAN: the arithmetic of all three public entries (native_gr_down_silu,
//        native_gr_pre_gated in BOTH fused_layer variants, native_gr_post),
//        the exact residual/output alias of post, the validation throws, and
//        the signed-zero behavior pinned by scale_zero_bias.
//   CANNOT: stream ordering / multi-launch consumer wiring (gr.cu owns it,
//        B4), or any model-scale performance property.
//
// Reference: host transcriptions of each kernel's own operation order, done
// in float64 (fma for __fmaf_rn, correctly-rounded product for __fmul_rn,
// std::exp for expf; the f32 kernel results are compared against the
// double-precision transcriptions in f32 ULPs).
//
// GATE (P2b margin analysis, written): every expf-dependent output is
// compared in f32 ULPs with the measured max reported.  The 64-ulp
// device-vs-host expf gap (b0_misc_probe, Arc/icpx 2026.1) is the dominant
// error source: it enters once per sigmoid, and the pre_gated mixed
// accumulates it over the hc=16 products, so the gates scale with the
// accumulation depth --
//   down_silu:  128 ulp  (one sigmoid per element)
//   gate[]:     128 ulp  (one sigmoid per element)
//   mixed[]:    8192 ulp (16 x 64 ulp x 8 accumulation margin)
//   output[]:   16384 ulp (one sigmoid per channel; the fma with the
//               residual can cancel, so ulp counts inflate near zero)
// A logic error (wrong formula, dropped /hc, fused vs non-fused mixup,
// transposed indices) moves results by relative amounts orders of magnitude
// above these gates.  The signed-zero fixtures are gated BIT-EXACT (IEEE
// fma/exp of zero is exact on both sides; the Arc device fma's signed-zero
// gap is corrected in the fmaf_rn glue, b1_fma_probe) -- those are the
// mutation fixtures for the scale_zero_bias-vs-plain-multiply change
// (fmaf(s, -0.0f, 0.0f) = +0.0f, s * -0.0f = -0.0f).
#include "strata/kernels/native_gr_postops.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// forward declarations: the port defines these in its own TU (no new public
// headers, per the Phase B convention)
namespace strata::kernels {
sycl::event submit_native_gr_down_silu(sycl::queue& q, float* lo, int hc_lr, int hc);
sycl::event submit_native_gr_pre_gated(sycl::queue& q, const float* xn, float* gate, float* mixed,
                                       int n_embd, int hc, bool fused_layer);
sycl::event submit_native_gr_post(sycl::queue& q, const float* residual, const float* block_out,
                                  const float* inject, float* output, int n_embd, int hc);
}  // namespace strata::kernels

namespace {

// ordered-integer distance between two f32 values (finite only -- the
// fixtures are finite and the kernels cannot produce NaN/inf on them).
long long ulp_diff(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    auto ord = [](uint32_t u) -> long long {
        return (long long) (u & 0x80000000u ? 0x80000000u | ~u : u);
    };
    const long long da = ord(ua), db = ord(ub);
    return da > db ? da - db : db - da;
}

// ---- reference transcriptions (the kernel's exact operation order) ----

// One IEEE f32 fma, transcribed EXACTLY: the f32 product has at most 48
// significant bits (fits f64 exactly), so adding the f32 addend and rounding
// once to f32 matches the hardware fma bit-for-bit -- including signed zero
// (RN of -0.0 * s + 0.0 is +0.0 under IEEE).
//
// TOOLCHAIN NOTE (measured in this driver's signed-zero fixture, -O2):
// icpx 2026.1 folds `x + 0.0` to `x` and `std::fma(a, b, 0.0)` to `a*b` in
// constant-fold contexts, BOTH of which flip IEEE's signed-zero result even
// through volatile reads.  The zero-addend cases are therefore resolved by
// inspection below instead of performing the add; only non-zero addends are
// actually added (no zero-sign case exists there).  This is a reference-side
// workaround, documented; the DEVICE kernels are unaffected (the device fma
// is a single instruction, and data-driven values cannot constant-fold).
inline float fma32_ref(float a, float b, float c) {
    const double p = (double) a * (double) b;    // exact f32 product in f64
    uint32_t uc;
    std::memcpy(&uc, &c, 4);
    if (uc == 0x00000000u) {                      // +0.0 addend: RN(+/-0.0 + 0.0) = +0.0
        if (p == 0.0) return 0.0f;
        return (float) p;                         // adding +0.0 to a finite non-zero is exact
    }
    if (uc == 0x80000000u) {                      // -0.0 addend: sign of the zero product is kept
        return (float) p;                         // exact for all finite p (zero or not)
    }
    return (float) (p + (double) c);             // non-zero addend: single f32 rounding
}

// down_silu: x = scale_zero_bias(lo[i], scale); lo[i] = x / (1 + expf(-x))
void ref_down_silu(const float* lo, float* out, long long n, int hc) {
    const float scale = 1.0f / (float) hc;            // same f32 value the kernel computes
    for (long long i = 0; i < n; ++i) {
        const float x = fma32_ref(scale, lo[(size_t) i], 0.0f);
        out[(size_t) i] = (float) ((double) x / (1.0 + std::exp(-(double) x)));
    }
}

// pre_gated: gate[i] = xn[i]*sigmoid(gate_in[i]);
// fused:     sum = ordered __fmaf_rn chain, mixed = scale * sum;
// non-fused: sum = ordered adds of the rounded products starting at zero,
//            mixed = scale_zero_bias(sum, scale).
void ref_pre_gated(const float* xn, const float* gate_in, float* gate, float* mixed,
                   int n_embd, int hc, bool fused) {
    const float scale = 1.0f / (float) hc;            // same f32 value the kernel computes
    for (int d = 0; d < n_embd; ++d) {
        float sum = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const size_t i = (size_t) c * n_embd + (size_t) d;
            // one f32 rounding of the exact sigmoid (the device expf gap vs
            // this is the measured 64-ulp term, absorbed by the gate)
            const float w = (float) (1.0 / (1.0 + std::exp(-(double) gate_in[i])));
            gate[i] = xn[i] * w;                       // one correctly-rounded f32 product
            if (fused)
                sum = fma32_ref(xn[i], w, sum);
            else
                sum = c == 0 ? gate[i] : sum + gate[i];
        }
        mixed[d] = fused ? scale * sum : fma32_ref(scale, sum, 0.0f);
    }
}

// post: weight = 2*sigmoid(scale*inject[c]); output[i] = fma(block_out[d], weight, residual[i])
void ref_post(const float* residual, const float* block_out, const float* inject, float* out,
              int n_embd, int hc) {
    const float scale = 1.0f / (float) hc;
    for (int c = 0; c < hc; ++c) {
        const float inner = fma32_ref(scale, inject[c], 0.0f);
        const float w = (float) (1.0 / (1.0 + std::exp(-(double) inner)));
        const float weight = fma32_ref(2.0f, w, 0.0f);
        for (int d = 0; d < n_embd; ++d) {
            const size_t i = (size_t) c * n_embd + (size_t) d;
            out[i] = fma32_ref(block_out[d], weight, residual[i]);
        }
    }
}

// ---- comparison (the 128-ulp gate, measured max reported) ----

int compare_ulp(const char* what, const float* got, const float* want, long long n,
                long long* max_ulp, long long gate = 128) {
    long long worst = 0, bad = 0, first = -1;
    for (long long i = 0; i < n; ++i) {
        const long long u = ulp_diff(got[(size_t) i], want[(size_t) i]);
        if (u > worst) worst = u;
        if (u > gate) { if (first < 0) first = i; ++bad; }
    }
    *max_ulp = worst;
    std::printf("  %-40s %s (max %lld ulp%s)\n", what, bad ? "*** WRONG ***" : "ok", worst,
                bad ? ", first bad idx " : "");
    if (first >= 0) std::printf("      first bad idx %lld: got %.9g want %.9g\n", first,
                               (double) got[(size_t) first], (double) want[(size_t) first]);
    return (int) bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_gr_postops_parity [--selftest]\n"); return 2; }
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ---- down_silu: random activations ----
    {
        const int hc = 8, n = 3 * 256 + 7;          // odd: exercises the padded tail
        std::vector<float> lo((size_t) n), want((size_t) n), got((size_t) n);
        for (auto& v : lo) v = gauss(rng);
        std::vector<float> hlo(lo);
        strata::sycl_compat::ctx ctx;
        float* d_lo = ctx.device_alloc<float>((size_t) n);
        ctx.q.memcpy(d_lo, hlo.data(), (size_t) n * sizeof(float));
        strata::kernels::submit_native_gr_down_silu(ctx.q, d_lo, n, hc);
        ctx.wait_and_throw();
        ctx.q.memcpy(got.data(), d_lo, (size_t) n * sizeof(float));
        ref_down_silu(hlo.data(), want.data(), n, hc);
        long long worst = 0;
        bad += compare_ulp("down_silu random", got.data(), want.data(), n, &worst);
        ctx.free_device(d_lo);
    }

    // ---- down_silu: signed-zero fixture (bit-exact; mutation target) ----
    // scale_zero_bias(-0.0f, s) = fmaf(s, -0.0f, 0.0f) = -0.0 + 0.0 = +0.0, so the
    // kernel must write +0.0f everywhere; a plain multiply would leave -0.0f.
    {
        const int hc = 8, n = 256;
        std::vector<float> lo((size_t) n);
        for (auto& v : lo) v = -0.0f;
        strata::sycl_compat::ctx ctx;
        float* d_lo = ctx.device_alloc<float>((size_t) n);
        ctx.q.memcpy(d_lo, lo.data(), (size_t) n * sizeof(float));
        strata::kernels::submit_native_gr_down_silu(ctx.q, d_lo, n, hc);
        ctx.wait_and_throw();
        std::vector<float> got((size_t) n);
        ctx.q.memcpy(got.data(), d_lo, (size_t) n * sizeof(float));
        const float plus_zero = 0.0f;
        long long zeros_ok = 0;
        for (long long i = 0; i < n; ++i) if (std::memcmp(&got[(size_t) i], &plus_zero, 4) == 0) ++zeros_ok;
        const bool ok = zeros_ok == n;
        std::printf("  %-40s %s (%lld of %lld exactly +0.0f)\n", "down_silu signed-zero",
                    ok ? "bit-exact" : "*** WRONG ***", zeros_ok, (long long) n);
        if (!ok) ++bad;
        ctx.free_device(d_lo);
    }

    // ---- pre_gated, both variants ----
    for (bool fused : {false, true}) {
        const int n_embd = 1024, hc = 16;
        const long long m = (long long) n_embd * hc;
        std::vector<float> xn((size_t) m), gate_in((size_t) m);
        for (auto& v : xn) v = gauss(rng);
        for (auto& v : gate_in) v = gauss(rng) * 3.0f;
        std::vector<float> want_gate((size_t) m), want_mixed(n_embd), got_gate((size_t) m), got_mixed(n_embd);
        strata::sycl_compat::ctx ctx;
        float* d_xn = ctx.device_alloc<float>((size_t) m);
        float* d_gate = ctx.device_alloc<float>((size_t) m);
        float* d_mixed = ctx.device_alloc<float>(n_embd);
        ctx.q.memcpy(d_xn, xn.data(), (size_t) m * sizeof(float));
        ctx.q.memcpy(d_gate, gate_in.data(), (size_t) m * sizeof(float));
        strata::kernels::submit_native_gr_pre_gated(ctx.q, d_xn, d_gate, d_mixed, n_embd, hc, fused);
        ctx.wait_and_throw();
        ctx.q.memcpy(got_gate.data(), d_gate, (size_t) m * sizeof(float));
        ctx.q.memcpy(got_mixed.data(), d_mixed, n_embd * sizeof(float));
        ref_pre_gated(xn.data(), gate_in.data(), want_gate.data(), want_mixed.data(), n_embd, hc, fused);
        long long worst = 0;
        bad += compare_ulp(fused ? "pre_gated fused: gate" : "pre_gated head: gate",
                          got_gate.data(), want_gate.data(), m, &worst);
        bad += compare_ulp(fused ? "pre_gated fused: mixed" : "pre_gated head: mixed",
                          got_mixed.data(), want_mixed.data(), n_embd, &worst, 8192);
        ctx.free_device(d_xn);
        ctx.free_device(d_gate);
        ctx.free_device(d_mixed);
    }

    // ---- pre_gated: signed-zero fixture (bit-exact; exercises the fmaf glue) ----
    // xn = -0.0f, gate_in = 1.0f:  gate[i] = fmul_rn(-0.0f, w) = -0.0f (plain
    // multiply keeps the sign), mixed = scale_zero_bias(sum, scale) with
    // sum = -0.0f -> fmaf(s, -0.0f, +0.0f) = +0.0f under IEEE (the Arc device
    // fma's sign-of-product behavior is corrected in the fmaf_rn glue).
    {
        const int n_embd = 256, hc = 4;
        const long long m = (long long) n_embd * hc;
        std::vector<float> xn((size_t) m, -0.0f), gate_in((size_t) m, 1.0f);
        std::vector<float> got_gate((size_t) m), got_mixed(n_embd), want_gate((size_t) m), want_mixed(n_embd);
        strata::sycl_compat::ctx ctx;
        float* d_xn = ctx.device_alloc<float>((size_t) m);
        float* d_gate = ctx.device_alloc<float>((size_t) m);
        float* d_mixed = ctx.device_alloc<float>(n_embd);
        ctx.q.memcpy(d_xn, xn.data(), (size_t) m * sizeof(float));
        ctx.q.memcpy(d_gate, gate_in.data(), (size_t) m * sizeof(float));
        strata::kernels::submit_native_gr_pre_gated(ctx.q, d_xn, d_gate, d_mixed, n_embd, hc, false);
        ctx.wait_and_throw();
        ctx.q.memcpy(got_gate.data(), d_gate, (size_t) m * sizeof(float));
        ctx.q.memcpy(got_mixed.data(), d_mixed, n_embd * sizeof(float));
        ref_pre_gated(xn.data(), gate_in.data(), want_gate.data(), want_mixed.data(), n_embd, hc, false);
        long long bad_gate = 0, bad_mixed = 0;
        for (long long i = 0; i < m; ++i) if (std::memcmp(&got_gate[(size_t) i], &want_gate[(size_t) i], 4) != 0) ++bad_gate;
        for (int d = 0; d < n_embd; ++d) if (std::memcmp(&got_mixed[d], &want_mixed[d], 4) != 0) ++bad_mixed;
        // the reference itself: gate must be -0.0f, mixed must be +0.0f
        const float neg0 = -0.0f, pos0 = 0.0f;
        const bool ref_ok = std::memcmp(&want_gate[0], &neg0, 4) == 0 && std::memcmp(&want_mixed[0], &pos0, 4) == 0;
        std::printf("  %-40s %s (%lld gate, %lld mixed differ%s)\n",
                    "pre_gated signed-zero (xn=-0, gate=1)",
                    (bad_gate || bad_mixed) ? "*** WRONG ***" : "bit-exact", bad_gate, bad_mixed,
                    ref_ok ? "" : ", reference itself wrong");
        if (bad_gate || bad_mixed || !ref_ok) ++bad;
        ctx.free_device(d_xn);
        ctx.free_device(d_gate);
        ctx.free_device(d_mixed);
    }

    // ---- post: non-aliased, then the exact residual/output alias ----
    for (bool alias : {false, true}) {
        const int n_embd = 1024, hc = 16;
        const long long m = (long long) n_embd * hc;
        std::vector<float> residual((size_t) m), block_out(n_embd), inject(hc);
        for (auto& v : residual) v = gauss(rng);
        for (auto& v : block_out) v = gauss(rng) * 0.5f;
        for (auto& v : inject) v = gauss(rng) * 2.0f;
        std::vector<float> want((size_t) m), got((size_t) m);
        strata::sycl_compat::ctx ctx;
        float* d_res = ctx.device_alloc<float>((size_t) m);
        float* d_bo = ctx.device_alloc<float>(n_embd);
        float* d_inj = ctx.device_alloc<float>(hc);
        float* d_out = ctx.device_alloc<float>((size_t) m);
        ctx.q.memcpy(d_res, residual.data(), (size_t) m * sizeof(float));
        ctx.q.memcpy(d_bo, block_out.data(), (size_t) n_embd * sizeof(float));
        ctx.q.memcpy(d_inj, inject.data(), (size_t) hc * sizeof(float));
        // alias mode: output == residual (the documented exact alias)
        float* d_target = alias ? d_res : d_out;
        strata::kernels::submit_native_gr_post(ctx.q, d_res, d_bo, d_inj, d_target, n_embd, hc);
        ctx.wait_and_throw();
        ctx.q.memcpy(got.data(), d_target, (size_t) m * sizeof(float));
        ref_post(residual.data(), block_out.data(), inject.data(), want.data(), n_embd, hc);
        long long worst = 0;
        bad += compare_ulp(alias ? "post in-place (output == residual)" : "post non-aliased",
                          got.data(), want.data(), m, &worst, 16384);
        ctx.free_device(d_res);
        ctx.free_device(d_out);
        ctx.free_device(d_bo);
        ctx.free_device(d_inj);
    }

    // ---- validation: every reject must throw std::invalid_argument ----
    {
        strata::sycl_compat::ctx ctx;
        auto throws_inv = [](auto&& f) {
            try { f(); return false; }
            catch (const std::invalid_argument&) { return true; }
            catch (...) { return false; }
        };
        float* p = ctx.device_alloc<float>(8);
        const bool ok =
            throws_inv([&] { strata::kernels::submit_native_gr_down_silu(ctx.q, nullptr, 8, 8); }) &&
            throws_inv([&] { strata::kernels::submit_native_gr_down_silu(ctx.q, p, 0, 8); }) &&
            throws_inv([&] { strata::kernels::submit_native_gr_pre_gated(ctx.q, nullptr, p, p, 8, 8, false); }) &&
            throws_inv([&] { strata::kernels::submit_native_gr_post(ctx.q, p, p, p, nullptr, 8, 8); });
        std::printf("  %-40s %s\n", "validation throws (null / zero dims)", ok ? "ok" : "*** WRONG ***");
        if (!ok) ++bad;
        ctx.free_device(p);
    }

    if (bad) { std::printf("native_gr_postops_parity: %d failures\n", bad); return 1; }
    if (selftest) std::printf("native_gr_postops_parity OK\n");
    return 0;
}
