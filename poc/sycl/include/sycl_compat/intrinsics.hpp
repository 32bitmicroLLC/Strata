#pragma once
// sycl_compat/intrinsics.hpp -- the CUDA intrinsic layer for the SYCL kernel
// ports (Phase B B0, plans/sycl-phase-b-steps-b0.md §1.2).
//
// Pattern: include/strata/hip_compat/intrinsics.hpp (the HIP build's version
// of the same job). Every `__` name below is a free global so that mirrored
// CUDA bodies compile VERBATIM (the checker diffs the mirror blocks); the
// macros at the bottom wire the CUDA spellings to these implementations.
// Everything here is a portable implementation (u32 bit ops, local memory),
// validated by the probes in poc/sycl/b0/. There is no CUDA device on this
// machine: the parity references are host C++ implementations of the
// documented CUDA semantics (the hip_compat validation pattern).
//
// ARCH GATE for the ports: each kernel .cpp defines `__CUDA_ARCH__` as 750 in
// glue BEFORE its mirror blocks. That selects the repo's own non-sm80
// (Turing) paths inside the mirrored #if ladders: fp16/tf32 mma helpers
// compile to `__trap()` (their kernels are runtime-rejected, the FMA
// fallback kernels run instead), `tf32_hi` degenerates to `__float_as_uint`,
// cp.async helpers become `__trap()`, and `STRATA_DP4A` takes the native
// `__dp4a` branch of include/strata/kernels/dp4a.hpp (mapped below). The
// asm lines inside `#if STRATA_*_SM80` branches therefore never compile in
// the SYCL tree. Documented per kernel in plans/sycl-phase-b-report.md.
//
// SCOPE: exactly the intrinsics the 40 unported .cu files use (audited by
// grep, plans/sycl-phase-b-steps-b0.md §1.2), plus `__vadd4` carried by the
// plan and `__nv_bfloat16` as a typedef for Phase C. `__byte_perm`/`__v*4`
// follow CUDA's unsigned-byte-lane semantics; `__dp4a` follows the repo's
// bit-exact STRATA_DP4A fallback semantics (signed bytes, int32
// accumulation mod 2^32 -- include/strata/kernels/dp4a.hpp).

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <sycl/sycl.hpp>

// CUDA vector types (float2/float4/int2/uint2/uint4 audited in use across
// the 40 files) and the half-precision spellings. GLOBAL scope: the
// mirrored CUDA bodies name them unqualified from inside
// namespace strata::kernels, which cannot see strata::sycl_compat members.
struct float2 {
    float x, y, z, w;
};
struct float4 {
    float x, y, z, w;
};
struct int2 {
    int x, y;
};
struct uint2 {
    unsigned x, y;
};
struct uint4 {
    unsigned x, y, z, w;
};

// CUDA __half = IEEE binary16 = sycl::half; __half2 the two-element struct.
using __half = sycl::half;
struct __half2 {
    __half x, y;
};
using half = sycl::half;  // the un-underscored spelling the kernels use
struct half2 {
    half x, y;
};

namespace strata::sycl_compat {

// CUDA __trap: abort the program. DPC++ 2026.1 forbids exceptions in kernel
// code, so the substitute is __builtin_trap().
//
// FINDING (b0_misc_probe, target Arc e223): on this device, with the
// standard in_order + profiling queue, __builtin_trap() is a SILENT NO-OP --
// the kernel reports as completed, no sycl::exception is delivered at event
// completion, and the host process stays alive. (A bare default-queue kernel
// segfaulted instead; the behaviour is queue/device-dependent and never a
// clean host-observable abort.) So the CUDA abort semantics cannot be
// reproduced on Arc with any kernel-side construct that DPC++ 2026.1 accepts
// (no in-kernel exceptions, no sycl::fence-based abort).
//
// Consequence for the ports: __trap() only ever COMPILES on the
// runtime-rejected sm80 paths (the mma / cp.async helpers); those kernels
// fall back to the FMA/naive kernels and are never launched, so the no-op is
// dead code in practice. A port that actually reached a __trap() would
// silently continue on Arc -- that is a written finding, not a silent
// tolerance, and any such site must be re-examined at port time.
// Declared early: the shfl32 full-mask guard below calls it.
inline void trap() { __builtin_trap(); }

// ============================================================================
// 32-wide warp emulation over work-group local memory (B0 §1.3).
//
// A "warp" is 32 consecutive work-group items: group g owns tids [32g,
// 32g+32), lane = tid % 32. CUDA's __shfl_* move values between lanes in
// registers; here each lane parks its value in its own local-memory slot
// and reads the source lane's slot after a work-group barrier (same values,
// more barriers -- the cost is measured by b0_shfl_bench, and the
// full-mask requirement mirrors the HIP header's require_full_wave_mask:
// the kernel set only ever passes 0xffffffff).
//
// Storage: 2 x u32 words per lane (covers the largest value any kernel
// shuttles through a shuffle, 8 bytes); 32 groups x 32 lanes x 2 = 2048
// words = 8 KiB. `spin` is one u32 per group for the __syncwarp arrival
// barrier (a 32-wide generation counter, cheaper than the 1024-wide
// work-group barrier it replaces -- probed in b0_shfl_probe).
//
// Construction is GLUE inside each kernel lambda:
//     sycl::local_accessor<uint32_t,1> sfl(sycl::range<1>(2048), h);
//     sycl::local_accessor<uint32_t,1> spin(sycl::range<1>(32), h);
//     sycl_compat::shfl32 shfl32(it, sfl, spin);
// and the mirrored lines then call the macro names below.
struct shfl32 {
    sycl::nd_item<1> it;
    // DPC++ exposes the local_accessor OBJECT as const inside the kernel;
    // the elements are still writable through it (both const and non-const
    // operator[] return a writable reference), so const refs are the idiom.
    const sycl::local_accessor<uint32_t, 1>& la;   // >= 2048 words
    const sycl::local_accessor<uint32_t, 1>& spin; // one per group
    const uint32_t lane;
    const uint32_t group;

    shfl32(sycl::nd_item<1> it, const sycl::local_accessor<uint32_t, 1>& la,
           const sycl::local_accessor<uint32_t, 1>& spin)
        : it(it), la(la), spin(spin), lane(it.get_local_id(0) & 31u), group(it.get_local_id(0) >> 5) {}

    static void require_full_mask(uint32_t mask) {
        if (mask != 0xffffffffu)
            ::strata::sycl_compat::trap();
    }

    static constexpr uint32_t slot_of(uint32_t group, uint32_t lane) { return (group * 32u + lane) * 2u; }

    template <class T> void store(uint32_t slot, const T& v) {
        static_assert(sizeof(T) <= 8, "shfl32 carries <= 8-byte values");
        if constexpr (sizeof(T) <= 4) {
            uint32_t u;
            std::memcpy(&u, &v, sizeof(T));
            la[slot] = u;
        } else {
            uint32_t u[2];
            std::memcpy(u, &v, 8);
            la[slot] = u[0];
            la[slot + 1] = u[1];
        }
    }

    template <class T> T load(uint32_t slot) const {
        static_assert(sizeof(T) <= 8, "shfl32 carries <= 8-byte values");
        if constexpr (sizeof(T) <= 4) {
            uint32_t u = la[slot];
            T v;
            std::memcpy(&v, &u, sizeof(T));
            return v;
        } else {
            T v;
            uint32_t u[2] = {la[slot], la[slot + 1]};
            std::memcpy(&v, u, 8);
            return v;
        }
    }

    // CUDA: value from lane (base + ((lane - base) ^ lane_mask)), width a
    // power of two, base = lane & ~(width-1).
    template <class T> T shfl_xor_sync(uint32_t mask, T v, int lane_mask, int width = 32) {
        require_full_mask(mask);
        const uint32_t base = lane & ~(uint32_t)(width - 1);
        const uint32_t src = base + ((lane - base) ^ (uint32_t) lane_mask);
        store(slot_of(group, lane), v);
        it.barrier();
        return load<T>(slot_of(group, src));
    }

    // CUDA: value from lane + delta if it stays in the sub-warp, else v.
    template <class T> T shfl_down_sync(uint32_t mask, T v, unsigned delta, int width = 32) {
        require_full_mask(mask);
        const uint32_t base = lane & ~(uint32_t)(width - 1);
        const uint32_t src = lane + (uint32_t) delta;
        store(slot_of(group, lane), v);
        it.barrier();
        return src >= base + (uint32_t) width ? v : load<T>(slot_of(group, src));
    }

    // CUDA: value from lane - delta if it stays in the sub-warp, else v.
    // (Guard before the subtract: lane - delta wraps in u32 when delta > lane.)
    template <class T> T shfl_up_sync(uint32_t mask, T v, unsigned delta, int width = 32) {
        require_full_mask(mask);
        const uint32_t base = lane & ~(uint32_t)(width - 1);
        const uint32_t src = lane < (uint32_t) delta ? lane : lane - (uint32_t) delta;
        store(slot_of(group, lane), v);
        it.barrier();
        return lane < (uint32_t) delta ? v : load<T>(slot_of(group, src));
    }

    // CUDA: value from source lane of the sub-warp (srcLane relative to base).
    template <class T> T shfl_sync(uint32_t mask, T v, int src_lane, int width = 32) {
        require_full_mask(mask);
        const uint32_t base = lane & ~(uint32_t)(width - 1);
        const uint32_t src = base + ((uint32_t) src_lane & ((uint32_t) width - 1u));
        store(slot_of(group, lane), v);
        it.barrier();
        return load<T>(slot_of(group, src));
    }

    // CUDA: bit i = lane i's predicate.
    uint32_t ballot_sync(uint32_t mask, int pred) {
        require_full_mask(mask);
        la[slot_of(group, lane)] = pred ? 1u : 0u;
        it.barrier();
        uint32_t r = 0;
        for (int i = 0; i < 32; ++i)
            r |= la[slot_of(group, (uint32_t) i)] << (uint32_t) i;
        return r;
    }

    // CUDA: __syncwarp(mask) -- order + join the participating lanes of the
    // warp. The full-mask requirement means all 32 lanes of the group
    // participate, so a 32-wide arrival barrier suffices (no 1024-wide
    // barrier): each lane increments its group's generation counter and
    // waits until the counter reaches the end of its generation.
    void syncwarp(uint32_t mask = 0xffffffffu) {
        require_full_mask(mask);
        auto c = sycl::atomic_ref<uint32_t, sycl::memory_order_relaxed, sycl::memory_scope_work_group>(
            spin[group]);
        const uint32_t r = c.fetch_add(1u);
        const uint32_t target = (r / 32u + 1u) * 32u;
        while (c.load() < target) {}
    }
};

// ============================================================================
// Scalar intrinsics. Portable implementations; the probes in poc/sycl/b0/
// compare each against an independently-written host reference of the
// documented CUDA semantics.

// Per-lane unsigned-byte accessor (the CUDA __v*4 lanes are 8-bit cells of
// the 32-bit word, numbered from the LSB).
inline uint32_t ua_byte(uint32_t w, int lane) { return (w >> (8 * (uint32_t) lane)) & 0xffu; }

// CUDA __dp4a: four SIGNED byte products of a and b, accumulated into c
// modulo 2^32 (the repo's STRATA_DP4A fallback semantics -- the parity
// reference for every port, not merely the sm<61 fallback).
inline int dp4a(int a, int b, int c) {
    const uint32_t ua = (uint32_t) a, ub = (uint32_t) b;
    uint32_t sum = (uint32_t) c;
    for (int i = 0; i < 4; ++i) {
        const int8_t x = (int8_t) (ua >> (8 * (uint32_t) i));
        const int8_t y = (int8_t) (ub >> (8 * (uint32_t) i));
        sum += (uint32_t) (int)(x * y);
    }
    return (int) sum;
}

// CUDA __vsub4: per unsigned-byte-lane subtraction, wrapping mod 256
// (== signed-byte subtraction mod 256 bit-for-bit).
inline int vsub4(int a, int b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= (((ua_byte((uint32_t) a, i)) - (ua_byte((uint32_t) b, i))) & 0xffu) << (8 * (uint32_t) i);
    return (int) r;
}

// CUDA __vadd4: per unsigned-byte-lane addition, wrapping mod 256.
inline int vadd4(int a, int b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= (((ua_byte((uint32_t) a, i)) + (ua_byte((uint32_t) b, i))) & 0xffu) << (8 * (uint32_t) i);
    return (int) r;
}

// CUDA __vsubss4: per SIGNED-byte-lane subtraction, saturating to [-128,127].
inline int vsubss4(int a, int b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const int8_t x = (int8_t) (ua_byte((uint32_t) a, i));
        const int8_t y = (int8_t) (ua_byte((uint32_t) b, i));
        int v = (int) x - (int) y;
        v = v < -128 ? -128 : (v > 127 ? 127 : v);
        r |= ((uint32_t) v & 0xffu) << (8 * (uint32_t) i);
    }
    return (int) r;
}

// CUDA __vcmpne4: 0xff in each lane where the bytes differ, 0x00 where equal.
inline int vcmpne4(int a, int b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        if (ua_byte((uint32_t) a, i) != ua_byte((uint32_t) b, i))
            r |= 0xffu << (8 * (uint32_t) i);
    return (int) r;
}

// CUDA __byte_perm(a, b, sel): each result byte i is byte ((sel >> 8i) & 7)
// of the 64-bit pair {a in the low 32, b in the high 32}, bytes numbered
// from the LSB (a byte 0 = LSB of a).
inline uint32_t byte_perm(int a, int b, int sel) {
    const uint64_t combined = (uint64_t) (uint32_t) a | ((uint64_t) (uint32_t) b << 32);
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= ((combined >> (8 * (uint32_t) ((sel >> (8 * i)) & 7))) & 0xffu) << (8 * (uint32_t) i);
    return r;
}

// CUDA PTX cvt.rna.tf32.f32: round-to-nearest, ties AWAY, to a 10-bit
// mantissa (bits 23..13 dropped, keeping 10 fraction bits), sign+exponent
// preserved; NaN/inf pass through. Returns the u32 bits, like the asm.
inline uint32_t tf32_rna_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) >= 0x7f800000u) return u;  // NaN / inf: unchanged
    // Dropped 13 bits, half-ULP = 0x1000: adding 0x1000 and truncating is
    // exactly round-to-nearest, ties AWAY (exact tie 0x1000 carries, below
    // 0x1000 does not), and a full-mantissa carry rolls into the exponent.
    return (u + 0x1000u) & 0xffffe000u;
}

// CUDA __ldg: read-only data cache load. Plain load on Arc (no separate
// read cache); perf noted where it matters at scale.
template <class T> inline T ldg(const T* p) { return *p; }

// Round-to-nearest IEEE scalars: the standard operators are already
// round-to-nearest, so these are identities (kept for the mirrored spellings).
inline float fadd_rn(float a, float b) { return a + b; }
inline float fsub_rn(float a, float b) { return a - b; }
// CUDA __fmul_rn: ONE rounded multiply.  A plain `a*b` inline can be contracted
// into a following `+ c` on device as an FMA, which changes the product's
// rounding exactly where it matters (the quantize_q8_K kernel comment documents
// the first version differing in 1 of 524,288 elements for this reason; the
// tie fixture in the driver is designed to catch it).  The volatile store
// forces the multiply to round to f32 before the value is used.
inline float fmul_rn(float a, float b) { volatile float m = a * b; return m; }
inline float fdiv_rn(float a, float b) { return a / b; }
// CUDA __fmaf_rn: one IEEE-correctly-rounded fma.  GLUE CORRECTION (B1
// finding, probed in b0/b1_fma_probe.cpp): the Arc device fma with a +0.0
// addend returns the SIGN OF THE PRODUCT for an exact zero product -
// fmaf(s, -0.0f, 0.0f) yields -0.0, where IEEE 754 requires the RN of the
// exact sum, +0.0 (and -0.0 + 0.0 rounds to +0.0).  The correction below
// covers exactly that case: with a == 0 or b == 0 the exact product IS
// zero, so no underflowing sum can masquerade as it.  Host std::fmaf is
// IEEE-exact (probe), so only the device result needs the fix.  The CUDA
// sources rely on this signed-zero behavior on purpose (the
// scale_zero_bias comment in native_gr_postops.cu documents it), so this is
// a written glue correction, not a tolerated gap.
inline float fmaf_rn(float a, float b, float c) {
    float r = std::fmaf(a, b, c);
    uint32_t ur, uc;
    std::memcpy(&ur, &r, 4);
    std::memcpy(&uc, &c, 4);
    if (uc == 0x00000000u && ur == 0x80000000u && (a == 0.0f || b == 0.0f))
        r = 0.0f;
    return r;
}
inline double dadd_rn(double a, double b) { return a + b; }
inline double dmul_rn(double a, double b) { return a * b; }
inline double ddiv_rn(double a, double b) { return a / b; }
inline double dsqrt_rn(double a) { return std::sqrt(a); }

// CUDA __expf: fast device exp. Maps to the device expf; the b0 probe
// records the max ulp gap vs the host libm expf (Phase A P2 precedent:
// 1-ulp device gaps are documented, real damage is caught by harnesses).
inline float expf_fast(float x) { return std::expf(x); }

// CUDA __fdividef: approximate-division hint on some ISAs. The port uses
// correctly-rounded division; each use site is checked at port time and any
// dependence on the approximation is a written finding (knife-edge rule).
inline float fdivide_approx(float a, float b) { return a / b; }

inline bool isnanf(float x) { return std::isnan(x); }

inline int popc(unsigned x) { return (int) __builtin_popcount(x); }

// CUDA __umulhi: top 32 bits of the 64-bit product of two unsigned ints.
inline uint32_t umulhi(uint32_t a, uint32_t b) { return (uint64_t) a * (uint64_t) b >> 32; }

// Bit casts (CUDA __int_as_float & co): same bits, other type.
inline float int_as_float(int x) {
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
inline float uint_as_float(uint32_t x) {
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
inline int float_as_int(float f) {
    int i;
    std::memcpy(&i, &f, 4);
    return i;
}
inline uint32_t float_as_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// CUDA __float2int_rn: round-to-nearest-even to int, saturating to
// [INT_MIN, INT_MAX] on overflow (per the CUDA math API notes).
inline int float2int_rn(float x) {
    double d = (double) x;
    if (d >= 2147483648.0) return 2147483647;
    if (d <= -2147483649.0) return -2147483648;
    long l = (long) std::floor(d);
    const double frac = d - (double) l;
    if (frac > 0.5 || (frac == 0.5 && (l & 1))) ++l;
    return (int) l;
}
// CUDA __float2int_rz: truncate toward zero, saturating.
inline int float2int_rz(float x) {
    double d = (double) x;
    if (d >= 2147483648.0) return 2147483647;
    if (d <= -2147483649.0) return -2147483648;
    return (int) d;  // truncation toward zero
}

// CUDA __threadfence / __threadfence_system: order this thread's memory
// operations across the device. (The system-wide reach of the CUDA version
// does not exist on a single-GPU SYCL host; queue in-order semantics plus
// event completion already order host visibility -- noted at each use site.)
// 2026.1 has no SYCL-2020 sycl::fence(fence_access, domain); the 2023-style
// sycl::atomic_fence(memory_order, memory_scope) is what the runtime ships.
inline void threadfence() { sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_device); }

// ============================================================================
// Half-precision (CUDA __half = IEEE binary16 = sycl::half; the type
// aliases are at global scope, above the namespace). The
// __float2half(_rn) constructors round to nearest, ties to even, which is
// what sycl::half does; b0_half_probe checks the tie corner cases.

inline float half2float(const __half& h) { return (float) h; }
inline __half float2half_rn(float f) { return sycl::half(f); }
// CUDA legacy name; the CUDA header maps it to the round-to-nearest form.
inline __half float2half(float f) { return sycl::half(f); }
inline uint16_t half_as_ushort(const __half& h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
inline __half ushort_as_half(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return h;
}
inline float low2float(const __half2& h) { return (float) h.x; }
inline float high2float(const __half2& h) { return (float) h.y; }
// CUDA __float2half2_rn(a, b) = {a, b}.
inline __half2 floats2half2_rn(float a, float b) { return __half2{float2half_rn(a), float2half_rn(b)}; }
// CUDA __floats2half2_rn(a, b) CROSS-packs: {a.y, b.x} (per the CUDA header,
// matching how fp16x2 register pairs feed mma layouts).
inline __half2 floats2half2_rn(const float2& a, const float2& b) {
    return __half2{float2half_rn(a.y), float2half_rn(b.x)};
}
inline __half2 halves2half2(float a, float b) { return __half2{float2half_rn(a), float2half_rn(b)}; }
// CUDA __hsub2(a, b) = {a.x - b.x, a.y - b.y}.
inline __half2 hsub2(const __half2& a, const __half2& b) {
    return __half2{a.x - b.x, a.y - b.y};
}

}  // namespace strata::sycl_compat

// Phase C / future-proofing (audited absent from the 40 files): CUDA
// __nv_bfloat16 is the same 16-bit layout as sycl::ext::oneapi::bfloat16.
using __nv_bfloat16 = sycl::ext::oneapi::bfloat16;

// ============================================================================
// The mirrored CUDA spellings. These macros are glue: the mirrored lines
// keep their CUDA text verbatim (checker-green) and expand to the portable
// implementations. shfl/ballot/syncwarp expand to the `shfl32` object the
// kernel lambda constructs in glue (see the shfl32 comment above).
//
// NOTE: ports do NOT include include/strata/kernels/dp4a.hpp (its
// STRATA_DP4A would clash with this one and it exists to select CUDA
// arches); the .cu's include of it is glue, replaced here.
#define __restrict__
#define __dp4a(a, b, c) (::strata::sycl_compat::dp4a((a), (b), (c)))
#define STRATA_DP4A(a, b, c) ::strata::sycl_compat::dp4a((a), (b), (c))
#define __vsub4(a, b) (::strata::sycl_compat::vsub4((a), (b)))
#define __vadd4(a, b) (::strata::sycl_compat::vadd4((a), (b)))
#define __vsubss4(a, b) (::strata::sycl_compat::vsubss4((a), (b)))
#define __vcmpne4(a, b) (::strata::sycl_compat::vcmpne4((a), (b)))
#define __byte_perm(a, b, s) (::strata::sycl_compat::byte_perm((a), (b), (s)))
#define __ldg(p) (::strata::sycl_compat::ldg((p)))
#define __fadd_rn(a, b) (::strata::sycl_compat::fadd_rn((a), (b)))
#define __fsub_rn(a, b) (::strata::sycl_compat::fsub_rn((a), (b)))
#define __fmul_rn(a, b) (::strata::sycl_compat::fmul_rn((a), (b)))
#define __fdiv_rn(a, b) (::strata::sycl_compat::fdiv_rn((a), (b)))
#define __fmaf_rn(a, b, c) (::strata::sycl_compat::fmaf_rn((a), (b), (c)))
#define __dadd_rn(a, b) (::strata::sycl_compat::dadd_rn((a), (b)))
#define __dmul_rn(a, b) (::strata::sycl_compat::dmul_rn((a), (b)))
#define __ddiv_rn(a, b) (::strata::sycl_compat::ddiv_rn((a), (b)))
#define __dsqrt_rn(a) (::strata::sycl_compat::dsqrt_rn((a)))
#define __expf(x) (::strata::sycl_compat::expf_fast(x))
#define __fdividef(a, b) (::strata::sycl_compat::fdivide_approx((a), (b)))
#define __isnanf(x) (::strata::sycl_compat::isnanf((x)))
#define __popc(x) (::strata::sycl_compat::popc((x)))
#define __umulhi(a, b) (::strata::sycl_compat::umulhi((a), (b)))
#define __int_as_float(x) (::strata::sycl_compat::int_as_float((x)))
#define __uint_as_float(x) (::strata::sycl_compat::uint_as_float((x)))
#define __float_as_int(x) (::strata::sycl_compat::float_as_int((x)))
#define __float_as_uint(x) (::strata::sycl_compat::float_as_uint((x)))
#define __float2int_rn(x) (::strata::sycl_compat::float2int_rn((x)))
#define __float2int_rz(x) (::strata::sycl_compat::float2int_rz((x)))
#define __threadfence() ::strata::sycl_compat::threadfence()
#define __threadfence_system() ::strata::sycl_compat::threadfence()
#define __trap() ::strata::sycl_compat::trap()
#define __shfl_xor_sync(...) (shfl32.shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (shfl32.shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (shfl32.shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (shfl32.shfl_sync(__VA_ARGS__))
#define __ballot_sync(mask, p) (shfl32.ballot_sync((mask), (p)))
#define __syncwarp(...) shfl32.syncwarp(__VA_ARGS__)
#define __half2float(h) (::strata::sycl_compat::half2float((h)))
#define __float2half_rn(f) (::strata::sycl_compat::float2half_rn((f)))
#define __float2half(f) (::strata::sycl_compat::float2half((f)))
#define __half_as_ushort(h) (::strata::sycl_compat::half_as_ushort((h)))
#define __ushort_as_half(u) (::strata::sycl_compat::ushort_as_half((u)))
#define __low2float(h) (::strata::sycl_compat::low2float((h)))
#define __high2float(h) (::strata::sycl_compat::high2float((h)))
#define __float2half2_rn(a, b) (::strata::sycl_compat::floats2half2_rn((a), (b)))
#define __floats2half2_rn(a, b) (::strata::sycl_compat::floats2half2_rn((a), (b)))
#define __halves2half2(a, b) (::strata::sycl_compat::halves2half2((a), (b)))
#define __hsub2(a, b) (::strata::sycl_compat::hsub2((a), (b)))
