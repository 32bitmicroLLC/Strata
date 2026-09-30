#pragma once
// sycl_compat/test_ctx.hpp -- host-side context for the SYCL parity drivers.
//
// Mirrors the idioms of the CUDA parity tests (check(), alloc/free, queue) in
// SYCL, encoding every T0 API finding:
//   * the device-selector lambda must return int (SYCL 2020)
//   * backend is `enum class sycl::backend` -- the Level Zero value is
//     `ext_oneapi_level_zero` (no .is<>() / .string() methods)
//   * the queue is built in_order + enable_profiling(); without the
//     profiling property, get_profiling_info throws
//   * sycl::malloc_host requires a sycl::context
//
// RULE: no kernel-specific code in this header. Per-kernel code lives in
// poc/sycl/kernels/.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::sycl_compat {

// Select the testbed Arc GPU by backend + name substring -- never by index:
// this machine also exposes an OpenCL Xeon CPU and another NEO GPU (T0.1).
// Override the substring with STRATA_SYCL_DEVICE_SUBSTR for other machines.
inline sycl::device pick_device() {
    const char* env = std::getenv("STRATA_SYCL_DEVICE_SUBSTR");
    const std::string needle = env ? env : "e223";
    return sycl::device([needle](const sycl::device& d) {
        return (int)(d.is_gpu()
            && d.get_backend() == sycl::backend::ext_oneapi_level_zero
            && d.get_info<sycl::info::device::name>().find(needle) != std::string::npos);
    });
}

class ctx {
    sycl::device dev_ = pick_device();

public:
    sycl::context context;
    sycl::queue q;

    ctx() : context(dev_),
            q(dev_, sycl::property_list{sycl::property::queue::in_order(),
                                        sycl::property::queue::enable_profiling()}) {}

    sycl::device device() const { return dev_; }

    template <class T> T* device_alloc(size_t n) const { return sycl::malloc_device<T>(n, q); }
    template <class T> T* host_alloc(size_t n) const { return sycl::malloc_host<T>(n, context); }

    template <class T> void free_device(T* p) const { sycl::free(p, q); }
    template <class T> void free_host(T* p) const { sycl::free(p, context); }

    // sycl::queue::wait_and_throw is not const, so neither is this.
    void wait_and_throw() { q.wait_and_throw(); }
};

// Mirror of the CUDA tests' check(cudaError_t, what): print, exit(1).
inline void fail(const char* what, const std::string& detail = "") {
    std::fprintf(stderr, "FAIL %s%s%s\n", what, detail.empty() ? "" : ": ", detail.c_str());
    std::exit(1);
}

}  // namespace strata::sycl_compat
