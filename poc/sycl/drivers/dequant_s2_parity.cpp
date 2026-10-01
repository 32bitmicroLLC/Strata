// poc/sycl/drivers/dequant_s2_parity.cpp -- T2 parity driver: the CUDA test
// src/kernels/dequant_s2_parity.cpp @ 6bd3e58 with its CUDA calls replaced by
// sycl_compat equivalents. Every fixture, the CPU reference chain, and the
// verdict logic are MIRRORED VERBATIM (marker blocks, checked by
// check_mirrors.sh) -- this driver provably tests what the CUDA test tests.
//
// The reference is the chain, not a second opinion (CUDA test's own words):
// raw Q2_0 blocks -> strata::dequantize_q2_0 (a scalar transcription that
// bench/micro/dequant_xcheck proved equal to ggml) -> canonical planes ->
// SYCL kernel -> per-element memcmp (bit comparison, no tolerance) ->
// non-vacuity check. Comparison is against a CPU decode already proven equal
// to ggml, never against a re-derivation.
#include "sycl_compat/test_ctx.hpp"

#include <cstdint>
#include <random>
#include <vector>

// CPU reference chain, mirrored from include/strata/artifact/dequant.hpp.
// These live in namespace strata (as in the original header) because the
// mirrored test code below calls them as strata::....
namespace strata {
// SYCL-MIRROR-BEGIN include/strata/artifact/dequant.hpp:33:55 @ 6bd3e58
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0)
            f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31)
        f = (sign << 31) | 0x7F800000u | (man << 13);
    else
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
// SYCL-MIRROR-END
// SYCL-MIRROR-BEGIN include/strata/artifact/dequant.hpp:64:66 @ 6bd3e58
inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
// SYCL-MIRROR-END
// SYCL-MIRROR-BEGIN include/strata/artifact/dequant.hpp:69:78 @ 6bd3e58
inline void dequantize_q2_0(const uint8_t* block, float* out) {
    const float d = fp16_to_fp32(read_u16(block));
    const uint8_t* qs = block + 2;
    for (int j = 0; j < 64; ++j) {
        const int byte_index = j / 4;
        const int bit_offset = (j % 4) * 2;
        const int code = (qs[byte_index] >> bit_offset) & 0x03;
        out[j] = (float)(code - 1) * d; // code {0,1,2,3} -> symbol {-1,0,+1,+2}
    }
}
// SYCL-MIRROR-END
}  // namespace strata

// Forward declaration of the kernel entry point; the definition is in
// poc/sycl/kernels/dequant_s2.cpp. (Phase B moves these to an include/ header.)
namespace strata::kernels {
sycl::event submit_dequant_s2(sycl::queue& q, const uint8_t* codes, const float* scales,
                             float* out, int64_t n_blocks);
}  // namespace strata::kernels

namespace {
// SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:22:22 @ 6bd3e58
constexpr int QK = 64;
// SYCL-MIRROR-END
// SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:31:35 @ 6bd3e58
// fp16 patterns that are finite, normal and exactly representable.  Random 16-bit patterns would include NaN
// and infinity, and a NaN makes `!=` true for a reason that has nothing to do with the kernel - the test
// would fail for the wrong cause, or worse, a real fault would hide behind a NaN it caused itself.
const uint16_t kScales[] = {0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00, 0x2800, 0x2000,
                            0xBC00, 0xB800, 0xB400, 0xB000, 0xAC00, 0x1800, 0x1400};
// SYCL-MIRROR-END
}  // namespace

int main(int argc, char** argv) {
// SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:40:51 @ 6bd3e58
    long long n_blocks = 200000;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) n_blocks = std::atoll(argv[++i]);
        else {
            std::fprintf(stderr, "usage: dequant_s2_parity [--selftest] [--blocks N]\n");
            return 2;
        }
    }
    if (n_blocks <= 0) { std::fprintf(stderr, "--blocks must be positive\n"); return 2; }

// SYCL-MIRROR-END

    strata::sycl_compat::ctx c;

// SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:52:79 @ 6bd3e58
    std::mt19937 rng(12345);
    const int n_scales = (int) (sizeof(kScales) / sizeof(kScales[0]));

    // raw Q2_0 blocks: { fp16 d ; uint8 qs[16] } - what ggml produced and what the artifact stores
    std::vector<uint8_t> raw((size_t) n_blocks * 18);
    for (long long b = 0; b < n_blocks; ++b) {
        const uint16_t d = kScales[rng() % n_scales];
        raw[(size_t) b * 18 + 0] = (uint8_t) (d & 0xFF);
        raw[(size_t) b * 18 + 1] = (uint8_t) (d >> 8);
        for (int t = 0; t < 16; ++t) raw[(size_t) b * 18 + 2 + t] = (uint8_t) (rng() & 0xFF);
    }

    // the CPU reference, over the RAW blocks
    std::vector<float> cpu((size_t) n_blocks * QK);
    for (long long b = 0; b < n_blocks; ++b) {
        strata::dequantize_q2_0(&raw[(size_t) b * 18], &cpu[(size_t) b * QK]);
    }

    // canonical planes: S2 packs 4 codes per byte exactly as Q2_0 stores them, so the codes plane is the raw
    // qs bytes; the scales plane is the fp16 `d` widened.  Copied into planes of their own rather than
    // aliased, because that is what the kernel will read in the engine.
    std::vector<uint8_t> codes((size_t) n_blocks * 16);
    std::vector<float> scales((size_t) n_blocks);
    for (long long b = 0; b < n_blocks; ++b) {
        std::memcpy(&codes[(size_t) b * 16], &raw[(size_t) b * 18 + 2], 16);
        const uint16_t d = (uint16_t) (raw[(size_t) b * 18] | (raw[(size_t) b * 18 + 1] << 8));
        scales[(size_t) b] = strata::fp16_to_fp32(d);
    }
// SYCL-MIRROR-END

    // SYCL glue (replaces the CUDA test's cudaMalloc/cudaMemcpy of lines 81-88).
    uint8_t* d_codes = c.device_alloc<uint8_t>(codes.size());
    float* d_scales = c.device_alloc<float>(scales.size());
    float* d_out = c.device_alloc<float>(cpu.size());
    c.q.memcpy(d_codes, codes.data(), codes.size());
    c.q.memcpy(d_scales, scales.data(), scales.size() * sizeof(float));

    // The kernel call (replaces line 90; the SYCL submit returns the event,
    // the driver owns the wait -- see the kernel file's contract note).
    strata::kernels::submit_dequant_s2(c.q, d_codes, d_scales, d_out, n_blocks);
    c.wait_and_throw();

    // Copy back (replaces the CUDA test's line 92-93).
    std::vector<float> gpu(cpu.size());
    c.q.memcpy(gpu.data(), d_out, gpu.size() * sizeof(float));
    c.wait_and_throw();

// SYCL-MIRROR-BEGIN src/kernels/dequant_s2_parity.cpp:95:137 @ 6bd3e58
    long long bad = 0, first_bad = -1;
    for (size_t i = 0; i < cpu.size(); ++i) {
        // bit comparison, not a tolerance: S2's decode is exact integer-to-float arithmetic, so ANY
        // difference is a bug and a tolerance would only hide which one
        if (std::memcmp(&cpu[i], &gpu[i], sizeof(float)) != 0) {
            if (first_bad < 0) first_bad = (long long) i;
            ++bad;
        }
    }

    // NON-VACUITY.  Two buffers of zeros compare equal, and so do two buffers a broken test filled with the
    // same constant - an agreement check that cannot distinguish "both correct" from "both empty" is the
    // failure this project keeps finding.  So the data must actually vary, and the four codes must all occur:
    // an S2 decode that only ever saw one code would agree while being unable to tell -1 from +2.
    bool seen[4] = {false, false, false, false};
    for (size_t i = 0; i < codes.size(); ++i) {
        for (int t = 0; t < 4; ++t) seen[(codes[i] >> (2 * t)) & 3] = true;
    }
    int distinct = 0;
    for (bool s : seen) distinct += s ? 1 : 0;
    long long distinct_vals = 0;
    for (size_t i = 0; i < cpu.size() && distinct_vals < 8; ++i) {
        bool dup = false;
        for (size_t j = 0; j < i && !dup; ++j) dup = (cpu[j] == cpu[i]);
        if (!dup) ++distinct_vals;
    }
    std::printf("  non-vacuity: %d of 4 codes present, %lld distinct values in the first elements\n", distinct,
                distinct_vals);
    if (distinct != 4 || distinct_vals < 4) {
        std::fprintf(stderr, "VACUOUS: the comparison could not have distinguished a wrong decode\n");
        return 1;
    }

    std::printf("dequant_s2: %lld blocks, %lld elements, %lld mismatched\n", n_blocks, (long long) cpu.size(),
                bad);
    if (bad) {
        std::printf("  first mismatch at element %lld (block %lld, offset %d): cpu %.9g gpu %.9g\n",
                    (long long) first_bad, (long long) (first_bad / QK), (int) (first_bad % QK),
                    (double) cpu[(size_t) first_bad], (double) gpu[(size_t) first_bad]);
        return 1;
    }
    std::printf("  bit-exact against the scalar dequantizer that dequant_xcheck proved equal to ggml\n");
    if (selftest) std::printf("dequant_s2_parity OK\n");
// SYCL-MIRROR-END

    // Frees (replace the CUDA test's cudaFrees of lines 139-141).
    c.free_device(d_codes);
    c.free_device(d_scales);
    c.free_device(d_out);
    return 0;
}
