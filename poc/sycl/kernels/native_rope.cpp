// poc/sycl/kernels/native_rope.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.2):
// the native (device-side) RoPE path.  CUDA source: src/kernels/cuda/native_rope.cu @ 71f031f.
//
// The `apply` kernel body is MIRRORED VERBATIM (marker blocks, checked by
// check_mirrors.sh); its 2D grid (dim3(gridX, rows), block 128) becomes a
// padded 1-D launch with the G1 id-decomposition glue.  `rope_scaled_angle`
// is a portable header function; `mrope_pos` is glue (F7: mrope.hpp guards
// it under __CUDACC__).
//
// Contract differences, on purpose (T4/T5 pattern):
//  - CUDA's native_rope_apply() takes a cudaStream_t; submit_native_rope()
//    takes the sycl::queue plus the same trailing `void* stream` parameter so
//    the mirrored validation block (which rejects a null stream) compiles
//    verbatim.  Every CUDA stream maps to ctx.q (G8).
//  - cudaGetDevice() has no multi-device meaning here: mrope_dev() glue
//    returns 0, so the 64-slot table is a single-slot table in practice (G8).
//  - The cudaGetLastError() check after the launch is dropped (SYCL throws;
//    the driver calls wait_and_throw()).
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "sycl_compat/launch.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {

// SYCL-MIRROR-BEGIN src/kernels/cuda/native_rope.cu:35:40 @ 71f031f
namespace {
std::atomic<bool> enabled{false};
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
// SYCL-MIRROR-END
}  // namespace (the CUDA anonymous namespace closes at line 62, after the kernel)

namespace {

// G1 glue: CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// F7 glue: mrope.hpp only defines mrope_pos under __CUDACC__/__HIPCC__; same
// body, plain load.
inline int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? tab[(size_t) pos * 3 + pair % 3] : pos;
}

// The `apply` kernel: CUDA lines 41-43 (the __global__ signature) are glue -
// the launch below passes the same arguments; the body (44-60) is mirrored.
void apply_kernel(const float* x, float* out, int rows, int width,
                  int n_rot, float theta_scale, float freq_scale, float corr_low, float corr_high,
                  float ext_factor, float mscale, const int* positions, const int32_t* mtab,
                  sycl::nd_item<1> it, int gridX, int threads) {
    // G1 glue: CUDA dim3(gridX, rows) x 128; CUDA linearizes blockIdx.x fastest,
    // then blockIdx.y, then threadIdx.x: gid = (y*gx + x)*128 + tx
    const dim3 blockDim{threads, 1, 1};
    const size_t blk = (size_t) it.get_global_id(0) / (size_t) threads;
    const dim3 blockIdx{(int) (blk % (size_t) gridX), (int) (blk / (size_t) gridX), 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_rope.cu:44:60 @ 71f031f
    const int row = blockIdx.y;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || pair >= width / 2) return;
    const size_t start = size_t(row) * width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    const float theta_extrap = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, float(pair));
    float c, s;
    rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
    // SYCL-MIRROR-END
}

}  // namespace

// SYCL-MIRROR-BEGIN src/kernels/cuda/native_rope.cu:63:66 @ 71f031f
// one table per device (a layer split runs the rope kernels on several): set and read for the current device
namespace {
constexpr int kMropeDevices = 64;
std::atomic<const int32_t*> mrope_tab[kMropeDevices] = {};
// SYCL-MIRROR-END
int mrope_dev() {
    // glue (CUDA lines 67-71): single device; cudaGetDevice maps to device 0 (G8)
    return 0;
}
}  // namespace
// SYCL-MIRROR-BEGIN src/kernels/cuda/native_rope.cu:73:76 @ 71f031f
void mrope_table_set(const int32_t* device_table) { mrope_tab[mrope_dev()].store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return mrope_tab[mrope_dev()].load(std::memory_order_relaxed); }
void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }
// SYCL-MIRROR-END

sycl::event submit_native_rope(sycl::queue& q, const float* x, float* out, int rows, int head_dim,
                              int n_rot, const RopeScaling& scaling, const int* positions, void* stream) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/native_rope.cu:79:94 @ 71f031f
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base/scaling and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match pinned host-side float powf before device fast powf/trigonometry.
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);   // none: the identity constants
    // SYCL-MIRROR-END
    // glue (CUDA lines 95-100): dim3(gridX, rows) x 128 launch; the
    // cudaGetLastError() check is dropped (SYCL throws, driver waits)
    const int threads = 128;
    const int gridX = (head_dim / 2 + threads - 1) / threads;
    const int32_t* mtab = mrope_table();
    const float freq_scale = k.freq_scale, corr_low = k.corr_low, corr_high = k.corr_high;
    const float ext_factor = k.ext_factor, mscale = k.attn_factor;
    return strata::sycl_compat::strata_launch(q, (size_t) gridX * rows * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        apply_kernel(x, out, rows, head_dim, n_rot, theta_scale, freq_scale, corr_low, corr_high,
                     ext_factor, mscale, positions, mtab, it, gridX, threads);
    });
}

}  // namespace strata::kernels
