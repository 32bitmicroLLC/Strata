// poc/sycl/t4/fp16_probe.cpp -- T4 task 0 (plans/sycl-phase-a-t4.md section 5):
// verify, before the s_gemv kernel is written, that
//   (a) the SYCL 16-bit float type converts in device code on icpx 2026.1, and
//   (b) the 16-bit -> float widening is bit-exact against the host
//       strata::fp16_to_fp32 (include/strata/artifact/dequant.hpp) on the
//       driver's kScales patterns (mirrored) plus edge patterns.
// FINDING: icpx 2026.1 has no sycl::fp16 (a draft-stage name); the SYCL 2020
// 16-bit IEEE binary16 type is sycl::half in this toolchain.  The probe (and
// the kernel) use sycl::half.
// fp16 -> fp32 needs no rounding (every fp16 value, subnormal included, is
// exactly representable in fp32), so bit-exactness is expected; a failure
// here means the hand-rolled fallback must be used in the kernel instead.
#include "sycl_compat/launch.hpp"
#include "sycl_compat/test_ctx.hpp"
#include "strata/artifact/dequant.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// SYCL-MIRROR-BEGIN src/kernels/s_gemv_parity.cpp:47:48 @ f69bc66
const uint16_t kScales[] = {0x3E00, 0x3555, 0x3C01, 0x4248, 0x4123, 0x2AAA, 0x4A2B, 0x3800,
                            0xBE00, 0xB555, 0xC248, 0x2AAB, 0x4A2C, 0x2AAB, 0xB800};
// SYCL-MIRROR-END

// edge patterns: zero, subnormals (positive and negative), smallest largest
// normals, max (positive and negative), NaN-ish exponent
const uint16_t kEdges[] = {0x0000, 0x0001, 0x8001, 0x0200, 0x01FF, 0x3C00, 0x7C00, 0xFC00, 0x7BFF};

}  // namespace

int main() {
    std::vector<uint16_t> bits;
    for (uint16_t v : kScales) bits.push_back(v);
    for (uint16_t v : kEdges) bits.push_back(v);
    const size_t n = bits.size();

    strata::sycl_compat::ctx ctx;
    uint16_t* d_bits = ctx.device_alloc<uint16_t>(n);
    float* d_out = ctx.device_alloc<float>(n);
    ctx.q.memcpy(d_bits, bits.data(), n * sizeof(uint16_t));
    sycl::event ev = strata::sycl_compat::strata_launch(ctx.q, n, 128, [=](sycl::nd_item<1> it) {
        const size_t g = it.get_global_id(0);
        if (g >= n) return;   // padded work-items (T2: Level Zero rejects non-uniform groups)
        const uint16_t b = d_bits[g];
        sycl::half h16;
        std::memcpy(&h16, &b, sizeof(h16));       // bit-construct, then widen exactly as the kernel will
        d_out[g] = (float) h16;
    });
    ev.wait();
    std::vector<float> got(n);
    ctx.q.memcpy(got.data(), d_out, n * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const float want = strata::fp16_to_fp32(bits[i]);
        bool same = std::memcmp(&got[i], &want, sizeof(float)) == 0;
        if (!same) ++bad;
        std::printf("  0x%04X  sycl %08X  host %08X  %s\n", bits[i],
                    *(const uint32_t*) &got[i], *(const uint32_t*) &want, same ? "exact" : "*** DIFF ***");
    }
    std::printf("t4_fp16_probe: %d of %zu patterns %s\n", bad, n, bad ? "*** WRONG ***" : "bit-exact");
    return bad ? 1 : 0;
}
