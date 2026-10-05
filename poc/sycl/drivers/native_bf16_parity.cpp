// poc/sycl/drivers/native_bf16_parity.cpp -- B1 (plans/sycl-phase-b-steps-b1.md,
// step 2.8, finding F1): a NEW dedicated parity driver for
// src/kernels/cuda/native_bf16.cu (bf16_gemv_fp32_mmvf /
// bf16_gemv_fp32_mmvf_multi).  There is no CUDA-side parity driver for these
// entries (they are exercised only through gr.cu, shared_expert.cu and
// ple.cu, B4/B2/B6), so this driver carries its own reference and its own
// contract.
//
// CONTRACT -- what this test can and cannot see:
//   CAN: the exact accumulation order of both entries (per-thread pair loop,
//        the two ordered FMAs per pair in ggml_cuda_mad order, the 5-round
//        XOR butterfly, the two-stage shared reduction for BLOCK_SIZE > 32),
//        the multi kernel's "bit-identical to its own single-row call"
//        contract, the adaptive block-size selection over ALL eight sizes
//        (32..256 step 32), the validation throws, and the fmaf_rn
//        signed-zero path.
//   CANNOT: model-scale performance, or any consumer wiring (gr.cu, B4;
//        shared_expert.cu, B2; ple.cu, B6 own it).
//
// GATE: bit-exact (memcmp).  Both operands are bf16-valued, so every product
// is exact in f32 and only the summation order is free; the order is
// transcribed below, so a logic error is a hard failure, not a tolerance.
//
// PLATFORM NOTE (measured, see the step report): the DEVICE std::fmaf is
// IEEE-exact (200,000/200,000 vs a double-precision ground truth), but
// icpx's HOST std::fmaf(f32) is TWO-rounding (53,073/200,000 off, exactly
// matching a two-rounding mul+add) AND icpx host -O2 miscompiles inline
// `vfmadd231ss` asm, flushing subnormal FMA results to +0 (measured with
// gcc -mfma returning the exact subnormal for the same bits).  The host
// reference therefore uses a pure-C bit-exact f32 FMA (integer arithmetic,
// compiler-agnostic), verified 200,000/200,000 bit-exact against a
// gcc -mfma FMA3 build over full-range finite f32 triples, plus the same
// signed-zero correction the __fmaf_rn glue applies.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;

// forward declarations: the port defines these in poc/sycl/kernels/native_bf16.cpp
namespace strata::kernels {
sycl::event bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                               sycl::queue& q);
sycl::event bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                      int64_t n_in, int64_t n_out, int n_tok, sycl::queue& q);
}  // namespace strata::kernels

namespace {

int g_fail = 0;
using k::f32_from_bf16;  // the host-side bf16->f32 (bf16_bits.hpp, unqualified in the references below)

// one IEEE f32 fma with the device glue's signed-zero correction, computed
// in pure C (no FMA instruction, no inline asm) so the result is immune to
// the host codegen gaps documented in the header.  Bit-exact for all finite
// f32 a, b, c (verified vs gcc -mfma FMA3, 200,000/200,000).
float fmaf_rn_ref(float a, float b, float c) {
    uint32_t ua, ub, uc;
    std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4); std::memcpy(&uc, &c, 4);
    // value = m * 2^shift with m in [2^23, 2^24) (normal, shift = E-150)
    // or m < 2^23, shift = -149 (subnormal)
    auto parts = [](uint32_t u, int& shiftout) -> uint64_t {
        if (u == 0) return 0;
        uint32_t E = (u >> 23) & 0xFFu;
        uint32_t m = u & 0x7FFFFFu;
        if (E == 0) { shiftout = -149; return m; }
        shiftout = (int) E - 150;
        return (uint64_t) m | 0x800000u;
    };
    int ea, eb, ec;
    uint64_t ma = parts(ua, ea), mb = parts(ub, eb), mc = parts(uc, ec);
    // exact product: 49-bit significand, sign included in sp
    __uint128_t prod = 0; int ep = 0; int sp = 1;
    if (ma && mb) {
        prod = ma * mb;
        ep = ea + eb;
        sp = ((ua >> 31) ^ (ub >> 31)) ? -1 : 1;
    }
    // exact sum S * 2^es, S signed, aligned at the smaller exponent
    __int128 S = 0; int es = 0;
    bool zero_neg = false;
    int sc = (uc >> 31) ? -1 : 1;
    if (prod == 0 && mc == 0) {
        // IEEE zero sign: -0 only when both addends are -0
        zero_neg = (uc >> 31) && ((ua >> 31) ^ (ub >> 31));
    } else if (prod == 0) { S = (__int128) mc * sc; es = ec; }
    else if (mc == 0) { S = (__int128) prod * sp; es = ep; }
    else if (ep > ec) {
        int d = ep - ec;
        if (d > 77) { S = (__int128) prod * sp; es = ep; }  // c < p*2^-77: absorbed, exact for RNE
        else { S = (((__int128) prod * sp) << d) + ((__int128) mc * sc); es = ec; }
    } else {
        int d = ec - ep;
        if (d > 77) { S = (__int128) mc * sc; es = ec; }    // p < c*2^-77: absorbed, exact for RNE
        else { S = ((__int128) prod * sp) + (((__int128) mc * sc) << d); es = ep; }
    }
    // Round S * 2^es to f32.
    bool sgn = S < 0;
    __int128 M = sgn ? -S : S;
    float r;
    if (M == 0) {
        r = (sgn || zero_neg) ? -0.0f : 0.0f;
    } else {
        int k = 0; { __int128 t = M; while (t >> 1) { t >>= 1; ++k; } }  // k = floor(log2 M)
        if (es + k >= -126 && es + k <= 127) {
            // normal: 24-bit RNE of M at bit k
            uint32_t m24;
            int eRes;
            if (k >= 23) {
                m24 = (uint32_t) (M >> (k - 23));
                __uint128_t drop = (unsigned __int128) (M & (((__int128) 1 << (k - 23)) - 1));
                __uint128_t half = (__uint128_t) 1 << (k - 24);
                if (drop > half || (drop == half && (m24 & 1))) ++m24;
                eRes = es + k + (m24 == 0x1000000u ? 1 : 0);
                m24 &= 0x7FFFFFu;
            } else {
                m24 = (uint32_t) (M << (23 - k));  // exact, no dropped bits
                eRes = es + k;
            }
            r = __builtin_bit_cast(float, (uint32_t) (((uint32_t) (eRes + 127) << 23) | (m24 & 0x7FFFFFu) | (sgn ? 0x80000000u : 0u)));
        } else if (es + k > 127) {
            r = sgn ? -__builtin_inff() : __builtin_inff();  // overflow (unreachable in these fixtures)
        } else {
            // subnormal: U = RNE(M * 2^(es+149)) in subnormal-ulp units
            int e149 = es + 149;              // bounded: k + e149 <= 22 in this branch
            __int128 U = 0;
            if (e149 >= 0) {
                U = M << e149;
            } else if (-e149 <= 127) {
                __int128 q = (__int128) 1 << (-e149);
                __int128 quo = M / q, rem = M % q;
                if (2 * rem > q) ++quo;
                else if (2 * rem == q && (quo & 1)) ++quo;
                U = quo;
            } // -e149 > 127: M < 2^127 so M*2^e149 < 1/2 -> 0
            r = (U == 0) ? (sgn ? -0.0f : 0.0f)
                        : __builtin_bit_cast(float, (uint32_t) U);  // U < 2^23 here
        }
    }
    uint32_t outb;
    std::memcpy(&outb, &r, 4);
    // the __fmaf_rn glue signed-zero correction (mirrors intrinsics.hpp exactly):
    if (uc == 0x00000000u && outb == 0x80000000u && (a == 0.0f || b == 0.0f)) r = 0.0f;
    return r;
}

// the 5-round XOR butterfly over one 32-lane warp (in parallel per round).
// The add is the exact integer f32 add (via fmaf_rn_ref(a, 1, b): the x1.0
// product is exact in the integer domain), because icpx host -O2 flushes f32
// subnormal intermediates to zero in plain float + (measured: 0x00000002 +
// 0.0f -> 0x00000000; gcc and icpx -O0 are exact).  No host f32 arithmetic
// may appear anywhere in the references.
void butterfly32(float* acc) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        std::vector<float> cur(acc, acc + 32);
        for (int lane = 0; lane < 32; ++lane) acc[lane] = fmaf_rn_ref(cur[lane], 1.0f, cur[lane ^ offset]);
    }
}

// reference: the single-row kernel's exact order for one output row.
float mmvf_ref_single(const float* x, const uint16_t* w_row, int n_in, int block_size) {
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(w_row);
    std::vector<float> acc(block_size, 0.0f);
    for (int t = 0; t < block_size; ++t)
        for (int pair = t; pair < n_in / 2; pair += block_size) {
            const uint32_t weight = w2[pair];
            acc[t] = fmaf_rn_ref(f32_from_bf16((uint16_t) weight), x[2 * pair], acc[t]);
            acc[t] = fmaf_rn_ref(f32_from_bf16((uint16_t) (weight >> 16)), x[2 * pair + 1], acc[t]);
        }
    for (int w0 = 0; w0 < block_size / 32; ++w0) butterfly32(&acc[(size_t) w0 * 32]);
    if (block_size > 32) {
        // the kernel's second stage: lanes 0-31 butterfly the 32-entry
        // partials (unwritten entries are zero) and every lane ends on the sum
        std::vector<float> p(32, 0.0f);
        for (int w0 = 0; w0 < block_size / 32; ++w0) p[(size_t) w0] = acc[(size_t) w0 * 32];
        butterfly32(p.data());
        return p[0];
    }
    return acc[0];
}

// reference: the multi kernel's exact order for the WHOLE n_tok x n_out output
// matrix (one workgroup per output column o, n_tok activation rows inside it).
std::vector<std::vector<float>> mmvf_ref_multi(const float* x, int64_t ldx, const uint16_t* w, int64_t n_out,
                                               int n_in, int n_tok, int block_size) {
    std::vector<std::vector<std::vector<float>>> acc(n_tok, std::vector<std::vector<float>>(n_out,
                                                                                        std::vector<float>(block_size, 0.0f)));
    for (int o = 0; o < n_out; ++o)
        for (int t = 0; t < block_size; ++t)
            for (int pair = t; pair < n_in / 2; pair += block_size) {
                const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) o * n_in)[pair];
                const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
                for (int k = 0; k < n_tok; ++k) {
                    const float ix = x[(size_t) k * ldx + 2 * pair], iy = x[(size_t) k * ldx + 2 * pair + 1];
                    acc[k][(size_t) o][(size_t) t] = fmaf_rn_ref(w0, ix, acc[k][(size_t) o][(size_t) t]);
                    acc[k][(size_t) o][(size_t) t] = fmaf_rn_ref(w1, iy, acc[k][(size_t) o][(size_t) t]);
                }
            }
    std::vector<std::vector<float>> out(n_tok, std::vector<float>(n_out));
    for (int k = 0; k < n_tok; ++k)
        for (int o = 0; o < n_out; ++o) {
            float* a = acc[k][(size_t) o].data();
            for (int w0 = 0; w0 < block_size / 32; ++w0) butterfly32(a + (size_t) w0 * 32);
            if (block_size > 32) {
                std::vector<float> p(32, 0.0f);
                for (int w0 = 0; w0 < block_size / 32; ++w0) p[(size_t) w0] = a[(size_t) w0 * 32];
                butterfly32(p.data());
                out[k][(size_t) o] = p[0];
            } else {
                out[k][(size_t) o] = a[0];
            }
        }
    return out;
}

// the kernel's block-size selection, mirrored into the driver to know which
// block size a launch used (and to prove all eight are reachable).
// SYCL-MIRROR-BEGIN src/kernels/cuda/native_bf16.cu:121:132 @ 1390a6a
int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}
// SYCL-MIRROR-END

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc == 1 || (argc == 2 && std::strcmp(argv[1], "--selftest") == 0);
    if (!selftest) {
        std::fprintf(stderr, "usage: native_bf16_parity [--selftest]\n");
        return 2;
    }
    strata::sycl_compat::ctx ctxq;
    std::mt19937 rng(91);

    // ---- 0. block-size coverage: one representative n_in per selected block size
    std::vector<int> reps;  // first even n_in found for each of the eight sizes
    {
        std::vector<int> seen(257, 0);
        for (int64_t n = 2; n <= 16384; n += 2) {
            const int bs = mmvf_block_size(n);
            if (!seen[bs]) {
                seen[bs] = 1;
                reps.push_back((int) n);
            }
        }
        std::vector<int> want = {32, 64, 96, 128, 160, 192, 224, 256};
        for (int bs : want)
            if (!seen[bs]) { std::fprintf(stderr, "FAIL: block size %d is unreachable by mmvf_block_size\n", bs); ++g_fail; }
        if (reps.size() != 8) ++g_fail;
        std::printf("  block-size selection covers all 8 sizes (reps: ");
        for (int n : reps) std::printf("%d->%d ", n, mmvf_block_size(n));
        std::printf(")\n");
    }

    const int N_OUT = 8;  // output rows per representative
    for (int n_in : reps) {
        // ---- 1. single-row kernel vs the host reference (bit-exact).
        // The header contract: y[o] = sum_i x[i] * w[o*n_in+i] -- ONE shared activation row
        // x[0..n_in) for every output row.  The multi tests below use the N_OUT
        // activation rows stored after it, so dxm holds [shared row][row 1]...[row N_OUT-1].
        const int bs = mmvf_block_size(n_in);
        // The fixture values are kept in |v| <= 2^9 so no product/sum can overflow to
        // +/-inf or NaN (a NaN comparison would hit payload-bit platform
        // differences, the same class as the f2bf NaN guard in step 2.5, and is
        // out of scope for a bit-exact gate).  Finite small/subnormal values are
        // still free to flow through the FMA and reduction.
        auto draw = [&]() {
            uint16_t v;
            do { v = (uint16_t) (rng() & 0xFFFFu); } while ((v >> 7) > 0x88u);  // no NaN/inf, |v| <= 2^9
            return v;
        };
        std::vector<uint16_t> w((size_t) N_OUT * n_in);
        for (auto& b : w) b = draw();
        // N_OUT+1 rows: row 0 is the shared single-row activation, rows 1..N_OUT
        // are the multi-test activation rows (n_tok <= N_OUT).
        std::vector<uint16_t> xb((size_t) (N_OUT + 1) * n_in);
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = (size_t) i % 97 == 0 ? 0x0000u : draw();  // sprinkle exact zeros (and -0 below)
        for (int r = 0; r < N_OUT + 1; ++r) xb[(size_t) r * n_in] = 0x8000u;  // a -0.0 activation in each row
        // x must be bf16-valued f32 (the header contract): convert the bits
        std::vector<float> xf(xb.size());
        for (size_t i = 0; i < xf.size(); ++i) xf[i] = k::f32_from_bf16(xb[i]);
        uint16_t* dw = ctxq.device_alloc<uint16_t>(w.size());
        float* dxm = ctxq.device_alloc<float>(xf.size());
        float* dy = ctxq.device_alloc<float>(N_OUT);
        ctxq.q.memcpy(dw, w.data(), w.size() * 2);
        ctxq.q.memcpy(dxm, xf.data(), xf.size() * 4);
        // all N_OUT outputs share activation row 0 (x[0..n_in))
        k::bf16_gemv_fp32_mmvf(dxm, dw, dy, n_in, N_OUT, ctxq.q);
        ctxq.wait_and_throw();
        std::vector<float> got(N_OUT);
        ctxq.q.memcpy(got.data(), dy, N_OUT * 4);
        ctxq.wait_and_throw();
        int bad = 0;
        for (int o = 0; o < N_OUT; ++o) {
            const float want_ = mmvf_ref_single(xf.data(), &w[(size_t) o * n_in], n_in, bs);
            if (std::memcmp(&got[o], &want_, 4) != 0) {
                if (g_fail < 5) std::fprintf(stderr, "FAIL: single row %d (n_in %d, block %d): got %a want %a\n", o, n_in, bs, got[o], want_);
                ++bad;
            }
        }
        if (bad) { std::fprintf(stderr, "FAIL: %d of %d single-row outputs differ (n_in %d, block %d)\n", bad, N_OUT, n_in, bs); ++g_fail; }
        // ---- 2. multi kernel: reference (bit-exact) + bit-identical to single-row launches.
        // n_tok activation rows are rows 1..n_tok of dxm, ldx = n_in, ldy = N_OUT.
        for (int n_tok : {2, 4, 8}) {
            std::vector<float> gotm((size_t) n_tok * N_OUT);
            float* dym = ctxq.device_alloc<float>((size_t) n_tok * N_OUT);
            k::bf16_gemv_fp32_mmvf_multi(&dxm[n_in], n_in, dw, dym, N_OUT, n_in, N_OUT, n_tok, ctxq.q);
            ctxq.wait_and_throw();
            ctxq.q.memcpy(gotm.data(), dym, gotm.size() * 4);
            ctxq.wait_and_throw();
            ctxq.free_device(dym);
            std::vector<std::vector<float>> ref = mmvf_ref_multi(&xf[n_in], n_in, w.data(), N_OUT, n_in, n_tok, bs);
            int badm = 0;
            for (int krow = 0; krow < n_tok; ++krow)
                for (int o = 0; o < N_OUT; ++o)
                    if (std::memcmp(&gotm[(size_t) krow * N_OUT + o], &ref[krow][(size_t) o], 4) != 0) {
                        uint32_t gb, wb;
                        std::memcpy(&gb, &gotm[(size_t) krow * N_OUT + o], 4);
                        std::memcpy(&wb, &ref[krow][(size_t) o], 4);
                        if (g_fail < 12) std::fprintf(stderr, "FAIL: multi row %d out %d (n_in %d, tok %d, block %d): got 0x%08x want 0x%08x\n", krow, o, n_in, n_tok, bs, gb, wb);
                        ++badm;
                    }
            if (badm) { std::fprintf(stderr, "FAIL: %d multi outputs differ from the reference (n_tok %d, n_in %d)\n", badm, n_tok, n_in); ++g_fail; }
            // the header contract: every multi output is bit-identical to its OWN single-row launch
            int bads = 0;
            for (int krow = 0; krow < n_tok; ++krow) {
                float* d1 = ctxq.device_alloc<float>(N_OUT);
                k::bf16_gemv_fp32_mmvf(&dxm[(size_t) (1 + krow) * n_in], dw, d1, n_in, N_OUT, ctxq.q);
                ctxq.wait_and_throw();
                std::vector<float> g1(N_OUT);
                ctxq.q.memcpy(g1.data(), d1, N_OUT * 4);
                ctxq.wait_and_throw();
                ctxq.free_device(d1);
                for (int o = 0; o < N_OUT; ++o)
                    if (std::memcmp(&gotm[(size_t) krow * N_OUT + o], &g1[o], 4) != 0) {
                        if (g_fail < 5) std::fprintf(stderr, "FAIL: multi row %d != its own single-row call, out %d (n_in %d)\n", krow, o, n_in);
                        ++bads;
                    }
            }
            if (bads) { std::fprintf(stderr, "FAIL: %d multi outputs differ from their own single-row calls (n_tok %d)\n", bads, n_tok); ++g_fail; }
        }
        // ---- 3. the n_tok == 1 fast path forwards to the single-row kernel
        {
            float* d1 = ctxq.device_alloc<float>(N_OUT);
            k::bf16_gemv_fp32_mmvf_multi(&dxm[n_in], n_in, dw, d1, N_OUT, n_in, N_OUT, 1, ctxq.q);
            ctxq.wait_and_throw();
            std::vector<float> g1(N_OUT);
            ctxq.q.memcpy(g1.data(), d1, N_OUT * 4);
            ctxq.wait_and_throw();
            ctxq.free_device(d1);
            float* d2 = ctxq.device_alloc<float>(N_OUT);
            k::bf16_gemv_fp32_mmvf(&dxm[n_in], dw, d2, n_in, N_OUT, ctxq.q);
            ctxq.wait_and_throw();
            std::vector<float> g2(N_OUT);
            ctxq.q.memcpy(g2.data(), d2, N_OUT * 4);
            ctxq.wait_and_throw();
            ctxq.free_device(d2);
            if (!std::equal(g1.begin(), g1.end(), g2.begin())) { std::fprintf(stderr, "FAIL: n_tok==1 fast path diverges from the single-row kernel (n_in %d)\n", n_in); ++g_fail; }
        }
        ctxq.free_device(dw);
        ctxq.free_device(dxm);
        ctxq.free_device(dy);
    }

    // ---- 4. signed-zero through the fmaf_rn correction: w = -0.0f, x = 0, acc starts at +0.0
    // the DEVICE fma without the correction would emit -0.0 (the Arc gap, b1_fma_probe); the
    // correction yields +0.0, and the reference applies the same rule, so this is bit-exact.
    {
        for (int n_in : {64, 128}) {  // the no-shared (32) and shared (64) block sizes
            std::vector<uint16_t> w((size_t) n_in, 0x0000u);
            w[0] = 0x8000u;  // -0.0f
            std::vector<float> x((size_t) n_in, 0.0f);
            uint16_t* dw = ctxq.device_alloc<uint16_t>(n_in);
            float* dx = ctxq.device_alloc<float>(n_in);
            float* dy = ctxq.device_alloc<float>(1);
            ctxq.q.memcpy(dw, w.data(), n_in * 2);
            ctxq.q.memcpy(dx, x.data(), n_in * 4);
            k::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in, 1, ctxq.q);
            ctxq.wait_and_throw();
            float got;
            ctxq.q.memcpy(&got, dy, 4);
            ctxq.wait_and_throw();
            const float want = mmvf_ref_single(x.data(), w.data(), n_in, mmvf_block_size(n_in));
            uint32_t ug, uw;
            std::memcpy(&ug, &got, 4);
            std::memcpy(&uw, &want, 4);
            if (ug != uw) { std::fprintf(stderr, "FAIL: signed-zero fixture (n_in %d): got 0x%08x want 0x%08x\n", n_in, ug, uw); ++g_fail; }
            ctxq.free_device(dw);
            ctxq.free_device(dx);
            ctxq.free_device(dy);
        }
    }

    // ---- 5. mutation: the gate must not be blind.  All-ones fixture: every y[o]
    // is the exact integer n_in (no rounding anywhere), then one weight bit changes
    // row 0's sum from n_in to n_in+1 -- a parity driver that cannot see a single
    // changed weight bit would be worthless.
    {
        const int n_in = 64;
        std::vector<float> x((size_t) n_in, 1.0f);
        std::vector<uint16_t> w((size_t) N_OUT * n_in, 0x3F80u);  // 1.0f
        uint16_t* dw = ctxq.device_alloc<uint16_t>(w.size());
        float* dx = ctxq.device_alloc<float>(x.size());
        float* dy = ctxq.device_alloc<float>(N_OUT);
        ctxq.q.memcpy(dw, w.data(), w.size() * 2);
        ctxq.q.memcpy(dx, x.data(), x.size() * 4);
        k::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in, N_OUT, ctxq.q);
        ctxq.wait_and_throw();
        std::vector<float> base(N_OUT);
        ctxq.q.memcpy(base.data(), dy, N_OUT * 4);
        ctxq.wait_and_throw();
        const float all_ones = (float) n_in;
        for (int o = 0; o < N_OUT; ++o)
            if (std::memcmp(&base[o], &all_ones, 4) != 0) {
                uint32_t g, wv;
                std::memcpy(&g, &base[o], 4); std::memcpy(&wv, &all_ones, 4);
                std::fprintf(stderr, "FAIL: mutation baseline row %d = 0x%08x, want 0x%08x (exact integer n_in)\n", o, g, wv);
                ++g_fail;
            }
        w[0 * n_in + 2] = 0x4000u;  // 2.0f: row 0's sum becomes n_in + 1
        ctxq.q.memcpy(dw, w.data(), w.size() * 2);
        ctxq.wait_and_throw();
        k::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in, N_OUT, ctxq.q);
        ctxq.wait_and_throw();
        std::vector<float> mut(N_OUT);
        ctxq.q.memcpy(mut.data(), dy, N_OUT * 4);
        ctxq.wait_and_throw();
        const float mutated_row0 = (float) (n_in + 1);
        if (std::memcmp(&mut[0], &mutated_row0, 4) != 0) {
            uint32_t g, wv;
            std::memcpy(&g, &mut[0], 4); std::memcpy(&wv, &mutated_row0, 4);
            std::fprintf(stderr, "FAIL: mutation not visible: row 0 = 0x%08x, want 0x%08x\n", g, wv);
            ++g_fail;
        }
        for (int o = 1; o < N_OUT; ++o)
            if (std::memcmp(&mut[o], &base[o], 4) != 0) {
                std::fprintf(stderr, "FAIL: mutation leaked to unrelated row %d\n", o);
                ++g_fail;
            }
        ctxq.free_device(dw);
        ctxq.free_device(dx);
        ctxq.free_device(dy);
        std::printf("  mutation: one flipped weight bit flips exactly one output row (n_in -> n_in+1)\n");
    }

    // ---- 6. the validation throws (mirrored conditions, host-side std::invalid_argument)
    {
        const int n_in = 128;
        float* dx = ctxq.device_alloc<float>((size_t) n_in * 2);
        uint16_t* dw = ctxq.device_alloc<uint16_t>((size_t) n_in * 2);
        float* dy = ctxq.device_alloc<float>(2);
        int thrown = 0, ok = 0;
        auto expect_throw = [&](auto&& f, const char* what) {
            try { f(); std::fprintf(stderr, "FAIL: no throw: %s\n", what); ++g_fail; }
            catch (const std::exception&) { ++thrown; }
        };
        auto expect_ok = [&](auto&& f, const char* what) {
            try { f(); ++ok; }
            catch (const std::exception& e) { std::fprintf(stderr, "FAIL: unexpected throw: %s: %s\n", what, e.what()); ++g_fail; }
        };
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, 2, n_in, 2, 0, ctxq.q); }, "n_tok = 0");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, 2, n_in, 2, 9, ctxq.q); }, "n_tok = 9");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in + 1, dw, dy, 2, n_in + 1, 2, 2, ctxq.q); }, "odd n_in");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, 2, dw, dy, 2, 0, 2, 2, ctxq.q); }, "n_in = 0");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in + 1, dw, dy, 2, n_in, 2, 2, ctxq.q); }, "odd ldx");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx + 1, n_in, dw, dy, 2, n_in, 2, 2, ctxq.q); }, "misaligned x");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in, nullptr, dy, 2, n_in, 2, 2, ctxq.q); }, "null w");
        expect_ok([&] { k::bf16_gemv_fp32_mmvf_multi(dx, n_in, dw, dy, 2, n_in, 2, 2, ctxq.q); ctxq.wait_and_throw(); }, "legal multi");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf(dx, dw, dy, n_in + 1, 2, ctxq.q); }, "single: odd n_in");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf(dx, dw, dy, 0, 2, ctxq.q); }, "single: n_in = 0");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf(nullptr, dw, dy, n_in, 2, ctxq.q); }, "single: null x");
        expect_throw([&] { k::bf16_gemv_fp32_mmvf(dx, dw + 1, dy, n_in, 2, ctxq.q); }, "single: misaligned w");
        std::printf("  validation: %d throws as expected, %d legal call(s) clean\n", thrown, ok);
        if (thrown != 11) { std::fprintf(stderr, "FAIL: expected 11 throws, got %d\n", thrown); ++g_fail; }
        ctxq.free_device(dx);
        ctxq.free_device(dw);
        ctxq.free_device(dy);
    }

    std::printf("native_bf16_parity: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
