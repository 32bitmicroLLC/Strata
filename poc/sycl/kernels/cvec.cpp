// poc/sycl/kernels/cvec.cpp -- B1 (plans/sycl-phase-b-steps-b1.md, step 2.9)
// SYCL port of src/kernels/cuda/cvec.cu @ 74d73e7: the control vector on the
// residual stream (project: h <- h - s (h . v) v; add: h <- h + d), one
// block per (stream, token), with the pending FFN write folded in when the
// layer's writes are folded into the next read (fused_gr.cu's arithmetic).
//
// The mirrored blocks below are VERBATIM (checked by check_mirrors.sh).
// Glue:
//  - G4: empty device qualifiers; `dim3` glue resolved per work-item; the
//    2-D grid (hc, T) x 256 is flattened to a 1-D launch and the body
//    resolves blockIdx.x = c (stream), blockIdx.y = t (token) from the
//    global id (each CUDA block maps to exactly one (c, t)).
//  - `__shared__ float part[THREADS / 32]` (CUDA line 72) is the
//    handler-built local accessor `part` (same name, so the mirrored
//    references read it unchanged); `__syncthreads` (76, 83) is the
//    intrinsics.hpp macro -> `it.barrier()`.
//  - `__shfl_xor_sync` expands onto the B0 `shfl32` object built in body
//    glue (local-memory 32-wide warp emulation, B0 §1.3): the 256-thread
//    block's eight CUDA warps are shfl32 groups 0-7, and both butterflies
//    (`o = 16..1`) stay inside their 32-lane group exactly as inside a CUDA
//    warp. The second stage runs in lanes 0-31 (shfl32 group 0) over the
//    eight partials, as the CUDA code runs it in warp 0.
//  - bare `fmaf` (66, 68, 91) routes through the `__fmaf_rn` glue
//    correction (intrinsics.hpp; the Arc signed-zero gap, B1 finding);
//    `__expf` is the B0 expf_fast layer (its 64-ulp gap is B0-measured and
//    touches only the write=true path, whose fused-gr cross-check parks in
//    B4 per finding F2).
//  - Host dispatch: the CUDA file's 64-device table plumbing
//    (cudaGetDevice/SetDevice/Malloc/Copy/Free loops) is single-device here
//    (one Arc): `cur_device()` returns 0, the `g_dev[kDevices]` array keeps
//    its mirrored shape but only slot 0 is ever populated, and every
//    function that touched the device takes `sycl::queue& q`. The
//    `<<<>>>` launch becomes a `q.submit` with the handler-built accessors;
//    `cudaPeekAtLastError` is dropped (SYCL submit throws).
#include "strata/kernels/cvec.hpp"
#include "sycl_compat/intrinsics.hpp"
#include "sycl_compat/launch.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

// glue: the arch gate (B0 header).
#define __CUDA_ARCH__ 750

// glue (G4): empty device qualifiers so the mirrored lines keep their CUDA spelling.
#define __device__
#define __forceinline__
#define __global__

namespace strata::kernels {
namespace {

// SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:11:20 @ 74d73e7
constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
// SYCL-MIRROR-END

// glue (G1): CUDA's grid/block/thread spelling, resolved per work-item (the
// constructor form lets the mirrored `dim3 grid(x, y)` launch line keep its
// CUDA spelling).
struct dim3 { int x, y, z; dim3(int ax, int ay, int az = 0) : x(ax), y(ay), z(az) {} };

int cur_device() { return 0; }  // glue: one Arc device (the cudaGetDevice clamp is a no-op)

// The `upload_here` body: lines 27-29 and 35-37 are mirrored; the
// cudaMalloc/cudaMemcpy chain (CUDA lines 30-34) is the queue's
// device-heap + memcpy, on the same device (the uploading device's).
bool upload_here(sycl::queue& q, std::string& err) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:27:29 @ 74d73e7
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    // SYCL-MIRROR-END
    try {
        t.dir = (float*) sycl::malloc_device<float>(g_dir_host.size(), q);
        t.s = (float*) sycl::malloc_device<float>(g_s_host.size(), q);
        t.on = (int*) sycl::malloc_device<int>(1, q);
        q.memcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float));
        q.memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float));
        q.memcpy(t.on, &flag, sizeof(int));
    } catch (const std::exception&) {
        // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:35:37 @ 74d73e7
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
        // SYCL-MIRROR-END
    }
    return true;
}

// SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:42:43 @ 74d73e7
// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
// SYCL-MIRROR-END

// The `cvec_kernel` `__global__` signature (CUDA lines 46-49) is glue; the
// body (50-94) is mirrored except line 72 (`__shared__ float part[...]`,
// the handler-built accessor of the same name).
void cvec_body(float* R, const float* dir, const float* s_l, const int* on, int mode, int64_t layer, int n, int hc,
               int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj, int64_t inj_ld, int write,
               sycl::nd_item<1> it, const sycl::local_accessor<float, 1>& part,
               const sycl::local_accessor<uint32_t, 1>& sfl, const sycl::local_accessor<uint32_t, 1>& spin) {
    // glue (G1): the CUDA 2-D grid (hc, T) flattened to 1-D; blockIdx.x = c, blockIdx.y = t.
    const int bid = (int) (it.get_global_id(0) / THREADS);
    const dim3 blockIdx{bid % hc, bid / hc, 0};
    const dim3 threadIdx{(int) it.get_local_id(0), 0, 0};
    strata::sycl_compat::shfl32 shfl32(it, sfl, spin);
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:50:71 @ 74d73e7
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
// SYCL-MIRROR-END
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:73:94 @ 74d73e7
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
// SYCL-MIRROR-END
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

// The `cvec_upload` signature gains `sycl::queue& q` (glue; CUDA line 101-102).
bool cvec_upload(sycl::queue& q, const std::vector<float>& dir, const std::vector<float>& s, int mode, int first,
                 int last, int64_t n_embd, int64_t hc, std::string& err) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:103:104 @ 74d73e7
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    // SYCL-MIRROR-END
    // glue: the 64-device free/switch loop (CUDA lines 105-116) on one device.
    DevTables& t0 = g_dev[0];
    if (t0.dir != nullptr) {
        q.wait();
        sycl::free(t0.dir, q);
        sycl::free(t0.s, q);
        sycl::free(t0.on, q);
        t0 = DevTables{};
    }
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:117:119 @ 74d73e7
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    // SYCL-MIRROR-END
    if (!upload_here(q, err)) return false;  // glue: the queue is passed through
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:121:131 @ 74d73e7
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    // SYCL-MIRROR-END
    return true;
}

bool cvec_replicate(sycl::queue& q, std::string& err) {
    return !g_cvec.loaded() || upload_here(q, err);  // glue: single device (CUDA line 135 takes no queue)
}

void cvec_set_enabled(sycl::queue& q, bool on) {
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:138:138 @ 74d73e7
    if (!g_cvec.loaded() || on == g_on_host) return;
    // SYCL-MIRROR-END
    // glue: the 64-device flag loop (CUDA lines 139-148) on one device; the
    // cudaDeviceSynchronize ("nothing in flight may still read the flag") is
    // the queue wait.
    q.wait();
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:141:141 @ 74d73e7
    const int v = on ? 1 : 0;
    // SYCL-MIRROR-END
    if (g_dev[0].on != nullptr) q.memcpy(g_dev[0].on, &v, sizeof(int));
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:149:149 @ 74d73e7
    g_on_host = on;
    // SYCL-MIRROR-END
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }  // CUDA line 152

// The `cvec_apply` launch (CUDA lines 154-164) becomes a `q.submit` with the
// handler-built accessors; the signature gains `sycl::queue& q` and returns
// the event (the `void* stream` is dropped: the queue IS the stream).
sycl::event cvec_apply(sycl::queue& q, float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo,
                       int64_t bo_ld, const float* inj, int64_t inj_ld, bool write) {
    if (!g_cvec.loaded() || T < 1) return {};  // glue: CUDA line 156's `return;` (this returns the event)
    // SYCL-MIRROR-BEGIN src/kernels/cuda/cvec.cu:157:159 @ 74d73e7
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    // SYCL-MIRROR-END
    // glue: the `<<<>>>` (CUDA lines 160-162); `cudaPeekAtLastError` (163) is
    // dropped (a bad launch throws at submit).  grid.x * grid.y blocks of 256.
    const size_t items = (size_t) grid.x * (size_t) grid.y * (size_t) THREADS;
    // glue: the kernel lambda captures plain values (SYCL forbids globals in
    // the kernel), read once from the mirrored globals above.
    const float* d_dir = t.dir;
    const float* d_s = t.s;
    const int* d_on = t.on;
    const int d_mode = g_cvec.mode;
    const int d_n = (int) g_cvec.n_embd;
    const int d_hc = (int) g_cvec.hc;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(THREADS / 32), h);
        sycl::local_accessor<uint32_t, 1> sfl(sycl::range<1>(2048), h);
        sycl::local_accessor<uint32_t, 1> spin(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(items, (size_t) THREADS),
                       [=](sycl::nd_item<1> it) {
                           cvec_body(R, d_dir, d_s, d_on, d_mode, layer, d_n, d_hc, r_ld, bo, bo_ld, inj, inj_ld,
                                     write ? 1 : 0, it, part, sfl, spin);
                       });
    });
}

}  // namespace strata::kernels
