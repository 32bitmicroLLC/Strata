#pragma once
// sycl_compat/launch.hpp -- the one launch idiom for the SYCL PoC.
//
// Testbed device facts (T0 dev0, Arc Battlemage [0xe223], 32 GB):
//   256 EU, max work-group 1024, sub-group sizes 16/32, 128 KiB shared
//   memory per work-group. Pick work-group sizes as powers of two <= 1024.
//
// Kernels that need shared memory do NOT use this helper: they define a
// kernel struct with a `sycl::local_accessor<T, 1>` member and launch it
// directly -- `q.parallel_for(sycl::nd_range<1>(items, work_group), kernel_obj)`.
// (That pattern first appears in the T3 router port; it is deliberately not
// abstracted here.)
#include <sycl/sycl.hpp>

#include <utility>

namespace strata::sycl_compat {

template <class Kernel>
inline sycl::event strata_launch(sycl::queue& q, size_t items, size_t work_group, Kernel&& kernel) {
    return q.parallel_for(sycl::nd_range<1>(items, work_group), std::forward<Kernel>(kernel));
}

}  // namespace strata::sycl_compat
