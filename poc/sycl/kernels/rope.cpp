// poc/sycl/kernels/rope.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.1):
// the NEOX partial RoPE table + rotation.  CUDA source: src/kernels/cuda/rope.cu
// @ 249e27a.
//
// Contract differences from the CUDA wrapper, on purpose (T4/T5 pattern):
// CUDA's rope_neox_apply() exit(1)s on a bad n_rot and synchronises when
// stream == nullptr; submit_rope_neox() returns the event, leaves
// synchronisation to the caller, and throws std::runtime_error instead of
// exiting.  The host state (rope_scaling_set/rope_scaling, build_rope_table,
// mrope_table) keeps its CUDA signatures - those functions are plain host
// code and the driver mirrors them as-is.
//
// The kernel body is MIRRORED VERBATIM (marker blocks, checked by
// check_mirrors.sh); blockIdx/blockDim/threadIdx are glue resolved from the
// flat work-group ids (G1).  mrope_pos is glue because mrope.hpp guards it
// under __CUDACC__ (finding F7): same body, plain load.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "sycl_compat/launch.hpp"

#include <cstdio>
#include <stdexcept>

namespace strata::kernels {

// SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:39:43 @ 249e27a
namespace {
// The process's rope config (rope_scaling.hpp).  One writer - the engine's startup thread, before
// session_init builds any table or captures any graph - and readers after it.
RopeScaling g_rope_scaling;
}  // namespace
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:45:46 @ 249e27a
void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:48:59 @ 249e27a
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}
// SYCL-MIRROR-END

// SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:61:86 @ 249e27a
void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}
// SYCL-MIRROR-END

namespace {

// G1 glue: CUDA's grid/block/thread spelling, resolved per work-item.
struct dim3 { int x, y, z; };

// F7 glue: mrope.hpp only defines mrope_pos under __CUDACC__/__HIPCC__; same
// body, plain load (__ldg's read-only-cache hint has no USM equivalent).
inline int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? tab[(size_t) pos * 3 + pair % 3] : pos;
}

}  // namespace

// One work-group of 128 covers 128 rows, mirroring the CUDA wrapper's
// `threads = 128`; `rows` rows pad to a work-group multiple and the mirrored
// `r >= rows` guard retires the surplus items.
sycl::event submit_rope_neox(sycl::queue& q, const float* x, float* out, long long rows,
                             int head_dim, int n_rot, const float* cos_tab, const float* sin_tab,
                             const int* pos) {
    // glue (CUDA line 119): the empty case submits nothing (T4 convention)
    if (rows <= 0 || n_rot <= 0) return q.submit([](sycl::handler&) {});
    // SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:120:120 @ 249e27a
    if (n_rot % 2 != 0 || n_rot > head_dim) {
    // SYCL-MIRROR-END
        // glue (CUDA lines 121-122): throw instead of exit(1)
        throw std::runtime_error("rope_neox_apply: n_rot must be even and <= head_dim");
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:124:125 @ 249e27a
    const int threads = 128;
    const unsigned grid = (unsigned) ((rows + threads - 1) / threads);
    // SYCL-MIRROR-END
    (void) grid;
    // glue (CUDA lines 126-127): the launch; mrope_table() is mirrored host state above
    const int32_t* mtab = mrope_table();
    return strata::sycl_compat::strata_launch(q, (size_t) grid * threads, threads,
                                              [=](sycl::nd_item<1> it) {
        // G1 glue: CUDA blockIdx/blockDim/threadIdx from the flat ids
        const dim3 blockDim{threads, 1, 1};
        const dim3 blockIdx{(int) (it.get_global_id(0) / threads), 0, 0};
        const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
        // SYCL-MIRROR-BEGIN src/kernels/cuda/rope.cu:101:112 @ 249e27a
    const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int half = n_rot / 2;
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
    for (int i = 0; i < half; ++i) {
        const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;   // the image path's per-pair position
        // THE PAIRING AND THE SIGNS COME FROM `rope_neox_pair`, shared with the indexer's pooling kernel.
        rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
    }
        // SYCL-MIRROR-END
    });
}

}  // namespace strata::kernels
