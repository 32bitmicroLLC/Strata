// poc/sycl/t0/dev0.cpp - T0.1 check 4: prove the SYCL toolchain can target the Arc GPU
// and report the numbers Phase B needs (subgroup widths, shared memory per work-group).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    // Select explicitly by name + backend so the OpenCL CPU (Xeon) or the
    // OpenCL NEO GPU cannot be picked: this box has three devices.
    sycl::device dev([](const sycl::device& d) {
        return (int)(d.is_gpu()
            && d.get_backend() == sycl::backend::ext_oneapi_level_zero
            && d.get_info<sycl::info::device::name>().find("e223") != std::string::npos);
    });
    std::printf("name            = %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("vendor          = %s\n", dev.get_info<sycl::info::device::vendor>().c_str());
    std::printf("driver version  = %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("backend version = %s\n", dev.get_info<sycl::info::device::backend_version>().c_str());
    std::printf("max_compute_units       = %u\n", dev.get_info<sycl::info::device::max_compute_units>());
    std::printf("max_work_group_size     = %zu\n", dev.get_info<sycl::info::device::max_work_group_size>());
    std::printf("max_num_sub_groups      = %u\n", dev.get_info<sycl::info::device::max_num_sub_groups>());
    const std::vector<size_t> sg = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::printf("sub_group_sizes         =");
    for (size_t s : sg) std::printf(" %zu", s);
    std::printf("\n");
    std::printf("preferred_vec_width_f32 = %u\n", dev.get_info<sycl::info::device::preferred_vector_width_float>());
    std::printf("local_mem_size (per WG) = %zu bytes\n", dev.get_info<sycl::info::device::local_mem_size>());
    std::printf("global_mem_size         = %zu bytes\n", dev.get_info<sycl::info::device::global_mem_size>());
    std::printf("max_clock_frequency     = %u MHz\n", dev.get_info<sycl::info::device::max_clock_frequency>());
    return 0;
}
