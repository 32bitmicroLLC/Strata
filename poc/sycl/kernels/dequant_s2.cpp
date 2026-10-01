// poc/sycl/kernels/dequant_s2.cpp -- T2: the first real kernel port (see
// plans/sycl-phase-a-t2.md). The S2 decode, CUDA source
// src/kernels/cuda/dequant_s2.cu @ 6bd3e58.
//
// Contract difference from the CUDA wrapper, on purpose and for Phase B:
// CUDA's dequant_s2() synchronises before returning and exit(1)s on a driver
// error; submit_dequant_s2() returns the event and leaves synchronisation to
// the caller (drivers call wait_and_throw()). The Phase B engine wrapper
// absorbs this.
//
// The kernel body below is MIRRORED VERBATIM from the CUDA file (marker block,
// checked by check_mirrors.sh). It stays naive on purpose -- one work-item per
// S2 block, scalar loads, 64-iteration loop. Optimisation belongs to the
// CUDA file's own Phase 3, guarded by the CUDA parity test, and is out of
// scope for Phase A.
//
// Bit-exactness: each output is one of {-d, 0, +d, 2d} where d is an exact
// fp16->fp32 widening and (code - 1) in {-1,0,1,2} -- scaling by -1/0/1/2
// cannot round, so the result is bit-identical on any IEEE-754 binary32
// device. The parity driver therefore compares with memcmp, no tolerance.
#include "sycl_compat/launch.hpp"

#include <cstdint>

namespace strata::kernels {
namespace {
// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_s2.cu:21:22 @ 6bd3e58
constexpr int QK = 64;             // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;
// SYCL-MIRROR-END
}  // namespace

// Decode `n_blocks` S2 blocks. `codes` is n_blocks*16 bytes, `scales`
// n_blocks floats, `out` n_blocks*64 floats, all USM device pointers.
// Work-group size 256 mirrors the CUDA blockDim; grid = n_blocks with the
// CUDA index bounds check kept inside the kernel.
sycl::event submit_dequant_s2(sycl::queue& q, const uint8_t* codes, const float* scales,
                             float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return q.submit([](sycl::handler&) {});  // mirror of the CUDA early return
    return strata::sycl_compat::strata_launch(q, (size_t) n_blocks, 256, [=](sycl::nd_item<1> it) {
        const long long b = (long long) it.get_global_id(0);
// SYCL-MIRROR-BEGIN src/kernels/cuda/dequant_s2.cu:27:34 @ 6bd3e58
    if (b >= n_blocks) return;
    const float d = scales[b];
    const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
    float* y = out + b * QK;
    for (int j = 0; j < QK; ++j) {
        const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
        y[j] = (float)(code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
    }
// SYCL-MIRROR-END
    });
}

}  // namespace strata::kernels
