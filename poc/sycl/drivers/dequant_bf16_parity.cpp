// poc/sycl/drivers/dequant_bf16_parity.cpp -- B1 step 2.5 (F5): the SYCL dequantizer
// parity driver, mirroring the CUDA test tool src/kernels/dequant_bf16_test.cpp
// @ 80d3d53 (its shard mode is this driver's default) and adding a --selftest
// mode that synthesizes block bytes per type (random + zero/NaN-scale fixtures)
// because this machine has no model shards.
//
// Contract (mirrors the CUDA tool's header): for each type, the device
// dequantizers are checked against the artifact's validated CPU dequantizers
// (strata/artifact/dequant.hpp -- the chain, not a second opinion): FP32 path
// with a relative tolerance of 1e-6 (the products are the reference's, reordered
// at most), BF16 path within half a BF16 ulp of the CPU value (native types
// only -- the CUDA dequant_bf16 wrapper's geometry table rejects the iq-only
// types).  The FP16 path is checked bit-for-bit against a host RNE-f16
// reference.
//
// Reference coverage, by type:
//   2,6,8,11,12,13,14,20,23,42  -> the dequant.hpp chain (independent).
//   16,17,18,21,22,29          -> host_decode_iq_f16/f32 (iq_dequant.cpp): the
//      SAME mirrored dq_dispatch code run on the host.  NOT independent: the
//      chain has no iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_m dequantizers, so
//      these six types check device launch/memory/indexing against their own
//      transcription, and their true parity waits on a shard or a chain
//      extension (recorded in the step report).
//
// Non-finite handling (driver addition, not in the CUDA tool): the selftest
// fixtures include a NaN-scale and a zero-scale block, and random bytes can
// produce inf scales (0x7C00).  The comparison treats NaN-vs-NaN and
// same-signed-inf as agreement; any other non-finite pairing is a failure.
//
// F8 note (plans/sycl-phase-b-report.md): this kernel's arithmetic is integer
// bit-manipulation plus fp32 multiply and fp16/bf16 RNE conversions -- no
// division -- so no F8-style 1-ulp gap is expected; the margins above are the
// CUDA tool's own, kept verbatim.
//
// NaN-payload note (mutation-test finding, step 2.5): f2bf's NaN guard IS
// load-bearing on NVIDIA (a small-payload float NaN, e.g. host-side
// 0x7F802000 from fp16 0x7C01, rounds to +inf without it), but NOT on this
// Arc: the device's half->float conversion emits a large-payload NaN
// (0x7FC02000 for the same fp16) that survives the rounding add as a quiet
// NaN, so removing the guard is not caught by any fixture here.  The guard
// stays mirrored verbatim (defensive parity); the platform difference is
// recorded in the step report.
#include "sycl_compat/test_ctx.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
// Forward decls; the definitions are in poc/sycl/kernels/dequant_bf16.cpp and
// poc/sycl/kernels/iq_dequant.cpp (the iq hooks are glue there, documented).
sycl::event submit_dequant_bf16(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               uint16_t* out);
sycl::event submit_dequant_f16(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               uint16_t* out);
sycl::event submit_dequant_f32(sycl::queue& q, int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols,
                               float* out);
bool dequant_bf16_supported(int ggml_type) noexcept;
void host_decode_iq_f32(int t, const void* src, int64_t n, float* dst);
void host_decode_iq_f16(int t, const void* src, int64_t n, uint16_t* dst);
}  // namespace strata::kernels

namespace {

// SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:23:38 @ 80d3d53
bool cpu_block(int type, const uint8_t* b, float* out) {
    using namespace strata;
    switch (type) {
    case 2: dequantize_q4_0(b, out); return true;
    case 6: dequantize_q5_0(b, out); return true;
    case 8: dequantize_q8_0(b, out); return true;
    case 11: dequantize_q3_K(b, out); return true;
    case 12: dequantize_q4_K(b, out); return true;
    case 13: dequantize_q5_K(b, out); return true;
    case 14: dequantize_q6_K(b, out); return true;
    case 20: dequantize_iq4_nl(b, out); return true;
    case 23: dequantize_iq4_xs(b, out); return true;
    case 42: dequantize_q2_0(b, out); return true;
    default: return false;
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:40:45 @ 80d3d53
float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// SYCL-MIRROR-END

// The ten types with a dequant.hpp chain reference (see the header).
bool has_chain_ref(int type) {
    return type == 2 || type == 6 || type == 8 || type == 11 || type == 12 || type == 13 || type == 14 || type == 20 ||
           type == 23 || type == 42;
}

// Byte offset of the fp16 scale inside one superblock, for the selftest's
// zero/NaN-scale fixtures.
int scale_off(int type) {
    switch (type) {
    case 11: return 108;  // Q3_K: d at the end of the 110-byte block
    case 14: return 208;  // Q6_K: d at the end of the 210-byte block
    case 29: return 12;   // IQ1_M: the iq1m_scale_t union after scales[12]
    default: return 0;    // everything else stores d first
    }
}

// Driver addition (not in the CUDA tool): the comparison with non-finite
// awareness.  `got`/`ref` are fp32 views of the device and reference values;
// the bf16 check reads the device's 16-bit output as BF16 (CUDA tool's
// tolerance, mirrored above in compare logic terms).
struct cmp_stats {
    int64_t bad = 0;
    double worst = 0.0;
};
cmp_stats compare_f32(const float* got, const float* ref, size_t n) {
    cmp_stats s;
    for (size_t i = 0; i < n; ++i) {
        const double g = (double) got[i], r = (double) ref[i];
        if (std::isfinite(r)) {
            if (!std::isfinite(g) || std::fabs(g - r) > 1e-6 * std::fabs(r) + 1e-30) ++s.bad;
            if (std::isfinite(g)) s.worst = (std::max)(s.worst, std::fabs(g - r) / (std::fabs(r) + 1e-30));
        } else if (std::isnan(r)) {
            if (!std::isnan(g)) ++s.bad;
        } else if (g != r) ++s.bad;  // signed infinity must match exactly
    }
    return s;
}
int64_t compare_bf16(const uint16_t* got, const float* ref, size_t n) {
    int64_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double g = (double) bf16_to_f32(got[i]), r = (double) ref[i];
        if (std::isfinite(r)) {
            if (!std::isfinite(g) || std::fabs(g - r) > std::ldexp(std::fabs(r), -8) + 1e-30) ++bad;
        } else if (std::isnan(r)) {
            if (!std::isnan(g)) ++bad;
        } else if (g != r) ++bad;
    }
    return bad;
}

// F16 output reference, host side: the device's f16 path is __float2half_rn
// per element (native types) or the mirrored cvt<__half> (iq types), both
// RNE; the host sycl::half conversion is the same RNE.
void ref_f16(const float* ref, uint16_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        sycl::half h(ref[i]);
        std::memcpy(&out[i], &h, 2);
    }
}

// Bit check of the f16 device output against a reference, skipping non-finite
// refs (their 16-bit NaN payloads are not compared bit-for-bit; the fp32
// comparisons above already cover them).
int64_t cmp_f16_bits(const uint16_t* got, const uint16_t* ref, const float* ref32, size_t n) {
    int64_t bad = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite((float) ref32[i]) && got[i] != ref[i]) ++bad;
    return bad;
}

// One selftest type: `rows` x `cols` of synthetic bytes, random + a zero-scale
// and a NaN-scale superblock in row 0.  Returns 0 iff the f32/bf16/f16 paths
// all agree with `ref` and the data is non-vacuous.
int run_type(strata::sycl_compat::ctx& c, int type, int64_t rows, int64_t cols) {
    int be = 0, bb = 0;
    strata::block_geometry((uint32_t) type, be, bb);
    const int64_t row_bytes = cols / be * bb;
    const int64_t n = rows * cols;
    const bool iq_only = !has_chain_ref(type);

    // random bytes, then the fixtures: superblock 0 of row 0 gets a +0 scale,
    // superblock 1 a NaN scale (0x7C01 little-endian -- the minimal-payload
    // NaN whose rounding add, without f2bf's guard, carries it to +inf)
    std::mt19937 rng(12345 + (unsigned) type);
    std::vector<uint8_t> raw((size_t) rows * row_bytes, 0);
    for (auto& b : raw) b = (uint8_t) (rng() & 0xFF);
    const int64_t sb_bytes = (int64_t) (256 / be) * bb;  // superblock bytes (256 values)
    const int so = scale_off(type);
    raw[so] = 0x00;
    raw[so + 1] = 0x00;
    raw[(size_t) sb_bytes + so] = 0x01;
    raw[(size_t) sb_bytes + so + 1] = 0x7C;

    // reference: the chain for the ten native types, the host decode for the
    // six iq-only ones (see the header's coverage note)
    std::vector<float> ref((size_t) n);
    if (has_chain_ref(type)) {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t b = 0; b < cols / be; ++b)
                cpu_block(type, raw.data() + r * row_bytes + b * bb, ref.data() + r * cols + b * be);
    } else
        strata::kernels::host_decode_iq_f32(type, raw.data(), n, ref.data());

    uint8_t* d_blocks = c.device_alloc<uint8_t>((size_t) rows * row_bytes);
    float* d_f = c.device_alloc<float>((size_t) n);
    uint16_t* d_h = c.device_alloc<uint16_t>((size_t) n);
    c.q.memcpy(d_blocks, raw.data(), (size_t) rows * row_bytes);
    strata::kernels::submit_dequant_f32(c.q, type, d_blocks, 0, rows, cols, d_f);
    strata::kernels::submit_dequant_f16(c.q, type, d_blocks, 0, rows, cols, d_h);
    std::vector<float> gf((size_t) n);
    std::vector<uint16_t> gh((size_t) n);
    c.q.memcpy(gf.data(), d_f, gf.size() * 4);
    c.q.memcpy(gh.data(), d_h, gh.size() * 2);
    c.wait_and_throw();

    // f32: the device path vs the reference (the CUDA tool's 1e-6 rel band)
    cmp_stats s = compare_f32(gf.data(), ref.data(), (size_t) n);
    // f16: the device path vs a host RNE-f16 reference
    //   native types: the chain value converted host-side
    //   iq types:     the mirrored dq_dispatch code run on the host
    std::vector<uint16_t> ref16((size_t) n);
    if (iq_only)
        strata::kernels::host_decode_iq_f16(type, raw.data(), n, ref16.data());
    else
        ref_f16(ref.data(), ref16.data(), (size_t) n);
    int64_t bad16 = cmp_f16_bits(gh.data(), ref16.data(), ref.data(), (size_t) n);
    // bf16: native types only -- the CUDA dequant_bf16 wrapper rejects the
    // iq-only types (its geometry table covers the ten native types)
    int64_t bad_bf = 0;
    if (!iq_only) {
        uint16_t* d_bf = c.device_alloc<uint16_t>((size_t) n);
        strata::kernels::submit_dequant_bf16(c.q, type, d_blocks, 0, rows, cols, d_bf);
        std::vector<uint16_t> gb((size_t) n);
        c.q.memcpy(gb.data(), d_bf, gb.size() * 2);
        c.wait_and_throw();
        c.free_device(d_bf);
        bad_bf = compare_bf16(gb.data(), ref.data(), (size_t) n);
    }

    // non-vacuity: the reference must actually vary (see dequant_s2_parity)
    long long distinct = 0;
    for (size_t i = 0; i < ref.size() && distinct < 4; ++i) {
        bool dup = false;
        for (size_t j = 0; j < i && !dup; ++j) dup = (ref[j] == ref[i]) || (std::isnan(ref[j]) && std::isnan(ref[i]));
        if (!dup) ++distinct;
    }
    bool ok = s.bad == 0 && bad16 == 0 && bad_bf == 0 && distinct >= 4;
    std::printf("type %2d  %-8s %lld x %lld: %s (f32 %lld off, worst rel %.2e, f16 %lld off, bf16 %lld off, %lld distinct)\n",
                type, iq_only ? "iq" : "chain", (long long) rows, (long long) cols, ok ? "ok" : "FAIL", (long long) s.bad, s.worst,
                (long long) bad16, (long long) bad_bf, distinct);
    c.free_device(d_blocks);
    c.free_device(d_f);
    c.free_device(d_h);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::vector<char*> files;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else files.push_back(argv[i]);
    }
    strata::sycl_compat::ctx c;
    if (selftest) {
        // the ten native types (chain reference) and the six iq-only types
        // (host-decode reference) -- 4 rows of 1024 values each
        int fails = 0;
        const int types[] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};
        for (int t : types) fails += run_type(c, t, 4, 1024);
        std::printf("k_dequant_bf16_parity selftest: %s\n", fails ? "FAILED" : "OK");
        return fails ? 1 : 0;
    }
    // SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:50:55 @ 80d3d53
    if (argc < 2) {
        std::fprintf(stderr, "usage: dequant_bf16_test SHARD.gguf [SHARD2.gguf]\n");
        return 2;
    }
    std::map<int, bool> done;
    int fails = 0, checked = 0;
    // SYCL-MIRROR-END
    for (int a = 0; a < (int) files.size(); ++a) {
        // SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:57:66 @ 80d3d53
        strata::GgufFile gguf(argv[a]);
        for (const auto& t : gguf.tensors()) {
            const int type = (int) t.type;
            if (done.count(type) || !strata::kernels::dequant_bf16_supported(type) || t.shape.size() != 2) continue;
            int be = 0, bb = 0;
            if (!strata::block_geometry(t.type, be, bb)) continue;
            const int64_t cols = (int64_t) t.shape[0];
            const int64_t rows = (std::min<int64_t>)(4, (int64_t) t.shape[1]);
            const int64_t row_bytes = cols / be * bb;
            const uint8_t* host = (const uint8_t*) gguf.tensor_data(t);
        // SYCL-MIRROR-END
            // SYCL glue (CUDA lines 67-82): alloc/copy in, kernel calls, copy out.
            void* d_blocks = c.device_alloc<uint8_t>((size_t) (rows * row_bytes));
            float* d_f = c.device_alloc<float>((size_t) (rows * cols));
            uint16_t* d_h = c.device_alloc<uint16_t>((size_t) (rows * cols));
            c.q.memcpy(d_blocks, host, (size_t) (rows * row_bytes));
            strata::kernels::submit_dequant_f32(c.q, type, d_blocks, 0, rows, cols, d_f);
            strata::kernels::submit_dequant_bf16(c.q, type, d_blocks, 0, rows, cols, d_h);
            std::vector<float> gf((size_t) (rows * cols));
            std::vector<uint16_t> gh((size_t) (rows * cols));
            c.q.memcpy(gf.data(), d_f, gf.size() * 4);
            c.q.memcpy(gh.data(), d_h, gh.size() * 2);
            c.wait_and_throw();
            // SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:83:103 @ 80d3d53
            std::vector<float> ref((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t b = 0; b < cols / be; ++b)
                    cpu_block(type, host + r * row_bytes + b * bb, ref.data() + r * cols + b * be);
            int64_t bad_f = 0, bad_h = 0;
            double worst = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double e = std::fabs((double) gf[i] - ref[i]);
                const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;
                if (e > tol) ++bad_f;
                worst = (std::max)(worst, e / (std::fabs((double) ref[i]) + 1e-30));
                const double eh = std::fabs((double) bf16_to_f32(gh[i]) - ref[i]);
                if (eh > std::ldexp(std::fabs((double) ref[i]), -8) + 1e-30) ++bad_h;
            }
            ++checked;
            done[type] = true;
            const bool ok = bad_f == 0 && bad_h == 0;
            if (!ok) ++fails;
            std::printf("type %2d  %-40s %lld x %lld: f32 %s (%lld off, worst rel %.2e), bf16 %s (%lld off)\n", type,
                        t.name.c_str(), (long long) rows, (long long) cols, bad_f ? "FAIL" : "ok", (long long) bad_f,
                        worst, bad_h ? "FAIL" : "ok", (long long) bad_h);
        // SYCL-MIRROR-END
            c.free_device(d_blocks);
            c.free_device(d_f);
            c.free_device(d_h);
        }
    }
    // SYCL-MIRROR-BEGIN src/kernels/dequant_bf16_test.cpp:106:107 @ 80d3d53
    std::printf("dequant_bf16_test: %d types checked, %s\n", checked, fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
    // SYCL-MIRROR-END
}
