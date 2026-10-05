// poc/sycl/drivers/s2_gemv_parity.cpp -- B2 step 3.1: the SYCL parity driver
// for the P2.S2 GEMV.  CUDA source: src/kernels/s2_gemv_parity.cpp @ ab69379;
// the kernel under test is poc/sycl/kernels/s2_gemv.cpp (port of
// src/kernels/cuda/s2_gemv.cu @ c1ff5cf).
//
// The reference is the same CHAIN as the CUDA test (its header says so, and
// this driver keeps it): `strata::dequantize_q2_0` is a scalar transcription
// that `bench/micro/dequant_xcheck` checks against ggml's own dequantizer, so
// comparing the SYCL GEMV against a dot product built on that scalar decode
// compares against ggml through one proven hop.  The reference chain is
// INLINED from include/strata/artifact/dequant.hpp (header-only, as in the
// B1 drivers) rather than re-mirrored here.
//
// WHY A TOLERANCE AND NOT BIT EQUALITY (the CUDA test's own words, kept): a
// dot product is not bit-stable -- the compiler is free to contract `a*b +
// acc` into an FMA, and icpx does, on the device and on the host alike.  So
// this asserts a RELATIVE error, measures what it actually is, and prints
// the number.
//
// SYCL adaptations (contract differences, on purpose): memory and
// synchronisation are the ctx/USM idiom -- cudaMalloc/cudaMemcpy become
// ctx.device_alloc/ctx.q.memcpy, the wrapper's own cudaDeviceSynchronize is
// submit + ctx.wait_and_throw, and `check(cudaError_t)` is not mirrored
// (SYCL throws).  The host reference loop keeps plain f32 arithmetic: the
// smallest nonzero term magnitude is 2^-10 * 2^-10 = 2^-20 (every kScales
// entry has |value| >= 2^-10), so no intermediate can reach the f32
// subnormal band that icpx's host -O2 flushes (B1 finding) -- the margin is
// written, not assumed.
#include "strata/artifact/dequant.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Forward declaration of the kernel entry point; the definition is in
// poc/sycl/kernels/s2_gemv.cpp.  (Phase B moves these to an include/ header.)
namespace strata::kernels {
sycl::event submit_s2_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* codes, const float* scales,
                          float* y, int64_t n_in, int64_t n_out);
}  // namespace strata::kernels

namespace {

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:26:26 @ ab69379
constexpr int QK = 64;
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:35:49 @ ab69379
// FP16 patterns, and they are deliberately NOT all powers of two.
//
// The first version of this table was {1.0, 0.5, 0.25, ...} - every value a power of two.  The test then
// reported a worst relative error of EXACTLY zero, which is not a property of the kernel: (code-1) is in
// {-1,0,1,2} and a power-of-two times an fp16 is exact, so every product was exact and no rounding could
// occur in either implementation.  A test whose arithmetic cannot round cannot exercise its own tolerance, and
// it would have hidden an accumulation-order or contraction bug for ever.
//
// The added patterns have full 11-bit mantissas, so products round and the two implementations'
// rounding/contraction behaviour actually matters.
const uint16_t kScales[] = {
    0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,          // 1, 0.5, 0.25, ... powers of two
    0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400,
    0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B,          // 1.75, 0.333.., 1.0009765625, ...
    0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB};
// SYCL-MIRROR-END

}  // namespace

int main(int argc, char** argv) {
// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:54:69 @ ab69379
    long long n_in = 2560;          // the real n_embd, so the row length is the engine's
    long long n_out = 640;          // the real expert intermediate width
    double tol = 1e-5;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--n-in") == 0 && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--n-out") == 0 && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: s2_gemv_parity [--selftest] [--n-in N] [--n-out N] [--tol T]\n");
            return 2;
        }
    }
    if (n_in % QK != 0) { std::fprintf(stderr, "--n-in must be a multiple of %d\n", QK); return 2; }
    const long long nb = n_in / QK;
// SYCL-MIRROR-END

    strata::sycl_compat::ctx c;

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:71:85 @ ab69379
    std::mt19937 rng(999);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));

    // FP16 activations.  Patterns are drawn from the same finite/exact table as the decode test, because a
    // random 16-bit pattern can be NaN and a NaN makes the comparison fail for the wrong reason.
    std::vector<uint16_t> x((size_t) n_in);
    for (long long i = 0; i < n_in; ++i) x[(size_t) i] = kScales[rng() % n_scales];

    // S2 weight planes for an [n_in, n_out] matrix: row o occupies a contiguous run of nb blocks.
    std::vector<uint8_t> codes((size_t) n_out * nb * 16);
    std::vector<float> scales((size_t) n_out * nb);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t) (rng() & 0xFF);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = strata::fp16_to_fp32(kScales[rng() % n_scales]);
    }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:87:113 @ ab69379
    // CPU reference: rebuild each row's RAW Q2_0 blocks from the planes, decode with the scalar path that
    // dequant_xcheck proved equal to ggml, and accumulate in the same order the kernel does.
    std::vector<float> ref((size_t) n_out, 0.0f);
    std::vector<uint8_t> raw(18);
    std::vector<float> dec(QK);
    for (long long o = 0; o < n_out; ++o) {
        float acc = 0.0f;
        for (long long b = 0; b < nb; ++b) {
            const float d = scales[(size_t) (o * nb + b)];
            // write `d` back as the fp16 pattern the plane came from: the plane stores a WIDENED fp16, so
            // re-narrowing is exact and the scalar decoder sees the same value the kernel does
            const uint16_t d_bits = (uint16_t) [&] {
                for (uint16_t c : kScales) {
                    if (strata::fp16_to_fp32(c) == d) return c;
                }
                return (uint16_t) 0x3C00;
            }();
            raw[0] = (uint8_t) (d_bits & 0xFF);
            raw[1] = (uint8_t) (d_bits >> 8);
            std::memcpy(&raw[2], &codes[(size_t) ((o * nb + b) * 16)], 16);
            strata::dequantize_q2_0(raw.data(), dec.data());
            for (int j = 0; j < QK; ++j) {
                acc += dec[(size_t) j] * strata::fp16_to_fp32(x[(size_t) (b * QK + j)]);
            }
        }
        ref[(size_t) o] = acc;
    }
// SYCL-MIRROR-END

    // SYCL glue (replaces the CUDA test's cudaMalloc/cudaMemcpy of lines 115-125).
    uint16_t* d_x = c.device_alloc<uint16_t>(x.size());
    uint8_t* d_codes = c.device_alloc<uint8_t>(codes.size());
    float* d_scales = c.device_alloc<float>(scales.size());
    float* d_y = c.device_alloc<float>((size_t) n_out);
    c.q.memcpy(d_x, x.data(), x.size() * sizeof(uint16_t));
    c.q.memcpy(d_codes, codes.data(), codes.size());
    c.q.memcpy(d_scales, scales.data(), scales.size() * sizeof(float));

    // The kernel call (replaces line 127; the SYCL submit returns the event,
    // the driver owns the wait -- see the kernel file's contract note).
    strata::kernels::submit_s2_gemv(c.q, d_x, d_codes, d_scales, d_y, n_in, n_out);
    c.wait_and_throw();

    // Copy back (replaces lines 129-130).
    std::vector<float> got((size_t) n_out);
    c.q.memcpy(got.data(), d_y, got.size() * sizeof(float));
    c.wait_and_throw();

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:132:143 @ ab69379
    long long bad = 0, first_bad = -1;
    double worst = 0.0;
    for (long long o = 0; o < n_out; ++o) {
        const double a = ref[(size_t) o], b = got[(size_t) o];
        const double denom = std::fabs(a) > 1e-30 ? std::fabs(a) : 1e-30;
        const double rel = std::fabs(a - b) / denom;
        if (!(rel <= worst)) worst = rel;
        if (!(rel <= tol)) {
            if (first_bad < 0) first_bad = o;
            ++bad;
        }
    }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/s2_gemv_parity.cpp:145:166 @ ab69379
    // NON-VACUITY: a zero matrix and a zero activation vector agree at zero.  The reference must actually vary
    // and its magnitudes must be non-trivial, or the comparison cannot distinguish anything.
    double lo = ref[0], hi = ref[0];
    for (float v : ref) {
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
    }
    const double spread = hi - lo;
    std::printf("s2_gemv: n_in %lld, n_out %lld, %lld rows over tolerance, worst rel %.3e (tol %.1e)\n", n_in,
                n_out, bad, worst, tol);
    std::printf("  reference spread [%.4g, %.4g]\n", lo, hi);
    if (spread <= 0.0) {
        std::fprintf(stderr, "VACUOUS: the reference is constant, so agreement proves nothing\n");
        return 1;
    }
    if (bad) {
        std::printf("  first bad row %lld: ref %.9g got %.9g\n", first_bad, (double) ref[(size_t) first_bad],
                    (double) got[(size_t) first_bad]);
        return 1;
    }
    std::printf("  agrees with the scalar decode that dequant_xcheck proved equal to ggml\n");
    if (selftest) std::printf("s2_gemv_parity OK\n");
// SYCL-MIRROR-END

    // Frees (replace the CUDA test's cudaFrees of lines 168-171).
    c.free_device(d_x);
    c.free_device(d_codes);
    c.free_device(d_scales);
    c.free_device(d_y);
    return 0;
}
