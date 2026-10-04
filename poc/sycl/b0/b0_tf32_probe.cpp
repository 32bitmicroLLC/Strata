// poc/sycl/b0/b0_tf32_probe.cpp -- B0 probe: the cvt.rna.tf32.f32 emulation.
//
// Hard gate for Step 6 (B5 / qsa_select): is ::strata::sycl_compat::
// tf32_rna_bits bit-exact against an INDEPENDENT host reference of the PTX
// semantics (round-to-nearest, ties AWAY, to a 10-bit fraction; NaN/inf
// pass through)? Context: the qsa_select port compiles with the repo's
// non-sm80 path (STRATA_SEL_SM80 = 0 under the __CUDA_ARCH__ 750 gate), so
// tf32_hi degenerates to __float_as_uint and the warp (FMA) kernel runs --
// the emulation is never executed on the ported path. This probe exists so
// the gate is a MEASURED fact, and so a future tc-path port on Arc has a
// proven bit-exact conversion available.
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
// Independent reference: truncate the low 13 bits; when the dropped part is
// at least the half-ULP (0x1000 in dropped-bit space) add one KEPT-ULP
// (0x2000 -- the kept fraction starts at bit 13), exact ties rounding away.
// A full-mantissa carry rolls into the exponent through the mask.
inline uint32_t ref_tf32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) >= 0x7f800000u) return u;
    const uint32_t dropped = u & 0x1fffu;
    const uint32_t trunc = u & 0xffffe000u;
    return dropped >= 0x1000u ? (trunc + 0x2000u) & 0xffffe000u : trunc;
}
}  // namespace

int main() {
    namespace sc = strata::sycl_compat;
    sc::ctx c;

    std::vector<uint32_t> bits;
    // Targeted set: tie and near-tie mantissas across every exponent.
    const uint32_t mantissas[16] = {0x0000, 0x0fff, 0x1000, 0x1fff, 0x2fff, 0x3000,
                                    0x3001, 0x3fff, 0x4000, 0x8000, 0xc000, 0xdfff,
                                    0xe000, 0xefff, 0xf000, 0xffff};
    for (uint32_t exp = 0; exp < 0x100; ++exp)
        for (uint32_t m : mantissas)
            for (int sgn = 0; sgn < 2; ++sgn) {
                uint32_t u = (sgn ? 0x80000000u : 0u) | (exp << 23) | m;
                if ((u & 0x7fffffffu) >= 0x7f800000u) continue;  // skip nan/inf here
                bits.push_back(u);
            }
    // Special values: zero, +-denormal edge, +-inf, two NaN payloads.
    bits.push_back(0);
    bits.push_back(0x80000000u);
    bits.push_back(0x00000001u);
    bits.push_back(0x80000001u);
    bits.push_back(0x7f800000u);
    bits.push_back(0xff800000u);
    bits.push_back(0x7fc00001u);
    bits.push_back(0xffc0abcd);
    // Random normals over the full exponent range.
    {
        std::mt19937 rng(99173u);
        for (int i = 0; i < (1 << 20); ++i) {
            uint32_t u = ((rng() % 0xfe) << 23) | (rng() & 0x7fffff);
            if ((u & 0x7fffffffu) >= 0x7f800000u) continue;
            bits.push_back(u);
        }
    }

    const size_t N = bits.size();
    uint32_t* dbits = c.device_alloc<uint32_t>(N);
    uint32_t* out = c.device_alloc<uint32_t>(N);
    c.q.memcpy(dbits, bits.data(), N * 4);
    // Arc rejects non-uniform work-groups: pad the global range to a multiple
    // of the local size (the Phase A launcher's standing rule).
    const size_t G = (N + 255) / 256 * 256;
    auto e = c.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(G, 256), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) {
                float x;
                std::memcpy(&x, &dbits[i], 4);
                out[i] = strata::sycl_compat::tf32_rna_bits(x);
            }
        });
    });
    e.wait_and_throw();

    std::vector<uint32_t> got(N);
    c.q.memcpy(got.data(), out, N * 4);
    c.wait_and_throw();

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) {
        float x;
        std::memcpy(&x, &bits[i], 4);
        if (got[i] != ref_tf32(x)) {
            if (bad < 8)
                std::printf("    i %zu: bits 0x%08x: want 0x%08x got 0x%08x\n", i, bits[i],
                            ref_tf32(x), got[i]);
            ++bad;
        }
    }
    std::printf("b0_tf32_probe: %s (%zu of %zu differ)\n", bad ? "*** FAIL ***" : "PASS", bad, N);
    if (!bad)
        std::printf("    -> emulation bit-exact; gate for B5 qsa_select satisfied (the ported\n"
                    "       non-sm80 path does not execute it -- tf32_hi degenerates to __float_as_uint)\n");
    c.free_device(dbits);
    c.free_device(out);
    return bad ? 1 : 0;
}
