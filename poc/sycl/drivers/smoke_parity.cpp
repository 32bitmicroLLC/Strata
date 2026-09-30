// smoke_parity -- T1 pipeline proof (plans/sycl-phase-a-t1.md).
//
// Proves the whole chain before any real kernel port: CMake target ->
// sycl_compat context -> USM alloc -> lambda kernel -> bitwise parity check
// -> ctest. 64 KiB of floats, copied on the GPU by an identity kernel
// (one work-item per 4 floats -- the T0 128-bit access pattern), compared
// against a host reference with std::memcmp. No tolerance: a copy has no
// rounding, so any tolerance here would be hiding a real difference.
//
// The init pattern is mirrored verbatim from the T0 spike behind
// SYCL-MIRROR markers -- the permanent worked example of the fixture-drift
// convention (enforced by check_mirrors.sh on every run.sh).
#include "sycl_compat/launch.hpp"
#include "sycl_compat/test_ctx.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>

using namespace strata::sycl_compat;

int main() {
    constexpr size_t ARR = (size_t) 64 * 1024 / sizeof(float);   // 64 KiB of floats
    ctx c;

    float* h_a = c.host_alloc<float>(ARR);
    // SYCL-MIRROR-BEGIN poc/sycl/t0/bw_spike.cpp:89:89 @ e5e4e3d
    for (size_t i = 0; i < ARR; ++i) h_a[i] = (float) i * 1e-10f + 1.0f;
    // SYCL-MIRROR-END
    float* h_y = c.host_alloc<float>(ARR);

    float* d_a = c.device_alloc<float>(ARR);
    float* d_y = c.device_alloc<float>(ARR);
    c.q.memcpy(d_a, h_a, ARR * sizeof(float));
    c.wait_and_throw();

    strata_launch(c.q, ARR / 4, 256, [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0) * 4;
        d_y[i]     = d_a[i];
        d_y[i + 1] = d_a[i + 1];
        d_y[i + 2] = d_a[i + 2];
        d_y[i + 3] = d_a[i + 3];
    }).wait();

    c.q.memcpy(h_y, d_y, ARR * sizeof(float));
    c.wait_and_throw();

    if (std::memcmp(h_a, h_y, ARR * sizeof(float)) != 0) {
        fail("smoke_parity: GPU copy diverges from host reference");
    }
    std::printf("smoke_parity: PASS (device %s)\n",
                c.device().get_info<sycl::info::device::name>().c_str());
    return 0;
}
