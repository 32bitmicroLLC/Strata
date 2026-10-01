# T4 — `s_gemv` port + driver (flagship)

Parent plan: `plans/sycl-phase-a.md` (T4 block). CUDA source grounded at HEAD
`f69bc66`:

- kernel: `src/kernels/cuda/s_gemv.cu` (694 lines)
- parity driver: `src/kernels/s_gemv_parity.cpp` (634 lines)
- API/fixture headers (reused, not re-derived): `include/strata/kernels/s_gemv.hpp`
  (`SForm`, `Codebook`), `include/strata/artifact/dequant.hpp` (scalar
  dequantizers the CPU reference chain calls, `fp16_to_fp32`)

## 1. Goal

Port the S-family GEMV — the kernel that carries most of the model's weight
bytes — to SYCL, validated by a parity driver that reuses the CUDA test's
reference chain, fixtures, verdict logic, and (in `--bench` mode) its timing
harness. Parent's done-criteria: **green for every form marked in-scope, and a
GEMV throughput number in the report.**

## 2. Scope decision (read before the rest)

The CUDA file contains **four kernel families**:

| CUDA family | Lines | Used by | T4 decision |
|---|---|---|---|
| `s_gemv_kernel<CW>` (naive, 1 thread/row) | 301–340 | `s_gemv` (the parity test's reference kernel) | **PORT — all three widths {2,4,8}, both codebooks, offset branch** |
| `s_gemv_split_kernel<CW>` (block/row + shared reduction) | 400–477 | `s_gemv_split[_async]` (overlap path) | **Phase B** — performance variant; its output is checked *against* the naive kernel, so validating the naive kernel on Arc validates the family's substrate |
| `s2_gemv_quads` / `s2_gemv_fast` | (separate file, `s2_gemv` family) | `bench_s2_gemv` only | **Phase B** — instruction-count experiments, not in the default parity run |
| `s_gemv_q8k*` / `s_gemv_q8_0_split` (Q8_K/Q8_0 *activation* blocks) | 90–139, 400–477, 580–694 | `layer.cpp` expert matvec | **Phase B** — a different *activation* contract (292 B / 34 B block loaders), machinery the fp16-activation path doesn't need |

Consequence: **one SYCL kernel covers all four fixture forms plus the Q4_K
offset branch.** The naive kernel is a single template on code width; S4/IQ4_NL
needs only the 16-byte codebook table (T3 already proved the local-accessor
pattern), and the offset branch is one runtime `int has_offset`. Nothing in the
in-scope set requires machinery the S2 path lacks — so **no form is deferred**,
which is the honest answer to the parent plan's deferral clause (it would have
applied only if S4 needed e.g. iq-packed layouts; it does not).

What the driver keeps from the CUDA test:
- **default run** — all 4 forms (`S2/Q2_0`, `S4/Q4_0`, `S4/IQ4_NL`, `S8/Q8_0`),
  the Q4_K offset case, the fp16 scale patterns, the raw-block reconstruction +
  scalar-dequant reference chain, the 1e-4 tolerance, the vacuity checks,
  selftest.
- **`--bench`** — the naive kernel timed at the engine's two real shapes
  (`[2560 x 640]`, `[640 x 2560]`) with the CUDA bench's fixed data and the
  same print/projection format. The split/quads/fast sweep (lines 356–465)
  does NOT port (lines 356–464): those kernels are Phase B, and benchmarking an
  unported kernel is impossible by definition. The bench output states this
  (mirroring the repo's NOTE convention).

## 3. Source grounding

### 3.1 The naive kernel (`s_gemv_kernel`, `s_gemv.cu` 301–340)

Structure, with file line numbers:

- 302–306 — signature: `x` (fp16 bits as `uint16_t*`), `codes` (packed,
  `n_out * n_in/PER_BYTE` bytes), `scales`/`offset` (f32, per group), `y`,
  `n_in`, `n_out`, `bias`, `codebook`, `group_elems`, `has_offset`
- 307–309 — comment: codebook table in shared, load+barrier BEFORE early
  return (a thread returning before the barrier would strand its peers)
- 310–311 — `__shared__ signed char s_iq4nl[16]` filled by first 16 threads,
  `__syncthreads()`
- 312 — `o = blockIdx.x * blockDim.x + threadIdx.x` (one row per thread)
- 313 — `if (o >= n_out) return;`
- 315–320 — `PER_BYTE` (constexpr from `CODE_BITS`), `n_groups = n_in /
  group_elems` (runtime 64-bit division, once), `codes_per_row`, row pointers
- 322–338 — the double loop: per group `d = s[g]`, `b = off ? off[g] : 0.0f`,
  per element `i` in group: extract code bit-pack (line 329), then
  `w = decode_tbl(code, bias, codebook, s_iq4nl) * d + b` (line 335 — offset
  applied to the WEIGHT, before the activation multiply: the comment at
  330–334 documents that the algebraically-equal other order rounds
  differently and only Q4_K could tell), then `acc += w * __half2float(...)`
  (line 336)
- 339 — `y[o] = acc;`

**No warp shuffles, no reduction, no atomics** — the only inter-thread coupling
is the 16-byte table fill + one barrier. The kernel is embarrassingly parallel
(one row per thread); the hard parts are the decode arithmetic, the bit-pack
extraction, the fp16 load, and the offset-branch rounding contract — which is
exactly why the Q4_K case exists.

`decode_tbl` (44–47): `codebook == Iq4Nl ? (float) tbl[code & 0x0F]
: (float)(code + bias)`. The codebook is carried as a runtime `int` because the
canonical form carries it per tensor; `CODE_BITS` stays a template parameter
(a compile-time property of the unpacking).

### 3.2 Host wrapper (`s_gemv`, `s_gemv.cu` 344–383)

- 346 — `n_in <= 0 || n_out <= 0` → return
- 347–351 — `group_elems` divides `n_in`, else print + `exit(1)`
- 352–355 — `has_offset` with null `offset`, else print + `exit(1)`
- 356–357 — `threads = 128`, `blocks = ceil(n_out / 128)`
- 358–382 — `switch (form.code_bits)` dispatching `s_gemv_kernel<2|4|8>` +
  `cudaDeviceSynchronize` + error exit

### 3.3 The parity driver (`s_gemv_parity.cpp`)

- 47–48 — `kScales`: 15 finite fp16 patterns **with full mantissas** (comment
  at 44–46: powers of two would make every product exact and the comparison
  could never round — the documented reason)
- 50–58 — `struct Case` { name, form, raw_bytes, to_raw, dequant }
- 60–121 — the four `to_raw`/`deq` pairs: rebuild the raw GGUF block for one
  group from the canonical planes (S2: packed-4-per-byte re-pack; S4: low/high
  nibble re-pack; IQ4_NL: same packing, table decode; S8: **sign-bit flip**
  `^ 0x80`, not a straight copy — lines 110–114 document the round-154 bug this
  guards) then decode with `strata::dequantize_*` (the ggml-checked scalar
  dequantizers from `dequant.hpp`)
- 130–291 — `test_q4k`: the ONLY canonical form with an offset. Generates d,
  dmin and eight (sc, m) pairs, derives the canonical planes exactly as the
  packer does, writes raw 144-byte blocks from the SAME parameters, and keeps
  **three references**: (a) raw block through the scalar dequantizer, (b)
  canonical planes through `code*scale + offset`, (c) the GPU. The metric is
  relative to `sum|term|` (the dot product's natural scale, not `|result|` —
  a 2560-term signed dot cancels; comment at 252–257), and it prints whether
  the test is **self-consistent** ((a) vs (b)) before any GPU result can be
  believed
- 302–472 — `bench_s2_gemv`: fixed data (codes 0xA5, scales 0.001, x = 1.0),
  warm-up discarded, min/median/mean ms, G weights/s + GiB/s of S2 streamed,
  the per-token projection (48 layers x 10 experts x 3 roles, VRAM-resident
  only — the NOTE at 353–354), then the split/quads/fast sweep (356–465,
  Phase B), each variant timed AND compared against the naive reference
- 473–634 — `main`: CLI (`--selftest --bench --iters --n-in --n-out`), defaults
  `n_in 2560 / n_out 128 / tol 1e-4` (comment 476–478: measured worst is ~1e-5,
  so 1e-5 would pass by 4%), the four `Case` fixtures (499–502), the
  one-code-per-element → packed-plane generation (519–536, comment documents
  the first-version bug: a random byte array used as both happened to agree
  for S2 only), CPU reference loop (544–564), GPU section (565–579), comparison
  + vacuity check (581–606), summary + `test_q4k` call (608–620), bench block
  (621–630), exit contract (631–633)

## 4. Kernel design — `poc/sycl/kernels/s_gemv.cpp`

One file, one public entry, same convention as T2/T3:

```cpp
namespace strata::kernels {
sycl::event submit_s_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* codes,
                         const float* scales, const float* offset, float* y,
                         long long n_in, long long n_out, const SForm& form);
}
```

Internals:

- `template <int CODE_BITS> sycl::event submit_impl(q, ..., int bias, int
  codebook, int group_elems, int has_offset)` — the kernel body; the public
  function runs the mirrored host checks and `switch (form.code_bits)` →
  `submit_impl<2|4|8>` (mirrors the CUDA wrapper's dispatch; the
  `default:` case becomes a throw, as in T3).
- **Launch geometry**: work-group size 128 (CUDA's `threads`), item count
  `ceil(n_out/128) * 128` — T2 padding, always uniform. `o = global_id`,
  `if (o >= n_out) return;` (mirrored line 313). All in-scope shapes divide 128
  (128, 640, 2560), so padding is dead but mandatory for arbitrary `--n-out`.
- **The table**: `sycl::local_accessor<signed char,1> s_iq4nl(range<1>(16), h)`
  built in the `q.submit` handler, captured by value (T3 pattern). Inside the
  kernel: fill by `tid < 16`, `it.barrier()`, **then** the early return — the
  CUDA ordering discipline (lines 307–309) is preserved, and the padded
  work-groups make it safe by construction.
- **fp16 activation**: `reinterpret_cast<const sycl::fp16*>(x)`, `acc += w *
  (float) xp[i];`. Task 0 below verifies `sycl::fp16` on this toolchain and
  its bit-exactness against `strata::fp16_to_fp32` before the kernel is
  written; fallback is a hand-rolled binary16→f32 (4 lines, bit-exact by
  construction).
- `__ldg` drops (read-only cache hint; no SYCL equivalent on raw USM pointers)
  — documented glue.
- **Host checks**: throw `std::runtime_error` instead of `exit(1)` (T3
  convention — a PoC library doesn't kill the process); same conditions,
  same message text. `submit_s_gemv` returns the event; the driver owns
  `wait_and_throw()` (Phase B's engine wrapper absorbs the difference, as with
  T2/T3).

The kernel body lives at 4-space indent (T3 convention) so the mirrored blocks
from the CUDA kernel stay column-identical.

## 5. Task 0 — fp16 API probe (`poc/sycl/t4/fp16_probe.cpp`)

~30 min, before the kernel. Compile + run a device lambda that converts 15
`kScales` patterns through `sycl::fp16 → float` and compares bit-exactly
against the host `strata::fp16_to_fp32` (from `dequant.hpp`). Records in the
report: (a) whether `sycl::fp16` exists and converts in device code on icpx
2026.1, (b) whether the widening is bit-exact (it should be — fp16→fp32 needs
no rounding), (c) the fallback decision. Registered as ctest `t4_fp16_probe`.

## 6. Mirror blocks

All reference `@ f69bc66`. The checker is the arbiter of exact ranges (T2/T3:
off-by-ones surface on first run and are fixed); the table below is the design
intent.

### 6.1 Kernel (`poc/sycl/kernels/s_gemv.cpp`), source `src/kernels/cuda/s_gemv.cu`

| Range | Content |
|---|---|
| `:33:33` | `kIq4Nl` value continuation line (line 32 is glue: `__constant__` → `static const`; the values must match) |
| `:45:46` | `decode_tbl` body (2 lines; template parameter dropped — it is unused in the body) |
| `:313:313` | `if (o >= n_out) return;` |
| `:315:320` | `PER_BYTE` / `n_groups` / `codes_per_row` / row pointer setup |
| `:322:334` | `acc` init, group loop, `d`/`b`/`base`, element loop, code extraction, the 5-line offset-order comment |
| `:339:339` | `y[o] = acc;` |
| `:346:346` | empty-input early return |
| `:347:347` | `group_elems` guard condition (body is glue: throw with mirrored message) |
| `:352:352` | `has_offset`/null guard condition (same treatment) |
| `:356:358` | `threads = 128`, `blocks` computation, `cb` |

10 blocks. The `w` line (335, `.data()` on the accessor + dropped template
arg) and the accumulation line (336, fp16 load) are commented glue, each
naming the CUDA line it replaces.

### 6.2 Driver (`poc/sycl/drivers/s_gemv_parity.cpp`), source `src/kernels/s_gemv_parity.cpp`

| Range | Content |
|---|---|
| `:47:48` | `kScales` table |
| `:50:58` | `struct Case` |
| `:62:74`, `:75:78` | `q2_0_to_raw` / `q2_0_deq` |
| `:82:88`, `:89:92` | `q4_0_to_raw` / `q4_0_deq` |
| `:95:97`, `:98:101` | `iq4_to_raw` / `iq4_deq` |
| `:106:117`, `:118:121` | `q8_0_to_raw` / `q8_0_deq` |
| `:131:133`, `:135:138`, `:140:141`, `:143:150` | `test_q4k` setup: form, rng, x, codes, planes |
| `:152:169`, `:171:179`, `:180:232` | `test_q4k` `struct Row` + row fill, reference vectors, the three-decode reference loop |

| `:252:285` | `test_q4k` comparison (rel-to-terms metric), prints, vacuity, `total_bad` |
| `:474:494` | `main` defaults + CLI loop |
| `:496:506` | usings, four `Case` fixtures, rng/`total_bad` |
| `:508:513` | per-`Case` loop head + derived constants |
| `:515:517`, `:519:542` | fixture generation: activations, codes (per-element → packed), scales |
| `:544:563` | CPU reference loop |
| `:581:606` | comparison loop, spread print, vacuity check, `total_bad` |
| `:608:620` | summary print, the Q4_K-gate history comment, `test_q4k` call |
| `:621:630` | bench block (adapted: no split sweep — see §2; `split_bad` stays 0 with a printed note) |
| `:631:633` | exit contract |

~25 blocks. Glue boundaries: the GPU sections (driver 234–251 in `test_q4k`,
565–579 in `main`; bench 313–333) become `ctx.device_alloc` + `memcpy` +
`submit_s_gemv` + `wait_and_throw()` + copy-back; the `check(cudaError_t, ...)`
helper is NOT mirrored (SYCL throws — same decision as T3); `cudaFree` →
`ctx.free_device`; the bench's `strata::kernels::s_gemv(...)` calls →
`submit + wait` inside the timing region (the CUDA call synced internally, so
per-iteration wait preserves the measurement's meaning). The vacuity
`std::exit(1)` in `test_q4k` (284) **is** mirrored — the driver is a test, not
a library, and exiting on a vacuous reference is the CUDA test's own contract.

## 7. Parity contract (mirrored, not re-derived)

- **Default run**: 4 forms × `n_in 2560` × `n_out 128`, seed 4242, `kScales`
  patterns, plus `test_q4k` (offset branch, self-consistency check).
- **Tolerance 1e-4** relative to `|ref|` for the four forms (CUDA comment:
  measured worst ~1e-5 from FMA contraction — the SYCL kernel's own contraction
  sits inside the same margin; no bit-exactness claim is made, and none is
  needed: the reference is the CPU chain, not the CUDA output).
- **Q4_K**: relative to `sum|term|`, 1e-4, plus the (a)-vs-(b) self-consistency
  print — a kernel result is only interpretable if the test's own planes agree
  with its own raw blocks.
- **Vacuity**: `hi - lo <= 0` per case → fail (mirrored).
- Verdict line format: mirrored verbatim (the report quotes it).

## 8. Mutation test (mandatory, per T2/T3 doctrine)

Two cheap mutations, each with a required failure signature:

1. **Drop the offset** in the kernel: line 335's `+ b` → removed (or `b` zeroed
   on the host). Expected: **`S4/Q4_K` fails** (offset terms are O(scale
   magnitudes) of the `sum|term|` — 128/128 rows over tol is plausible; any
   failure count > 0 with a large worst rel suffices), the four fixture cases
   stay green (they have no offset). This is the branch whose existence the
   kernel's own comment justifies — the mutation must be caught by it.
2. **Invert the bias** in `decode_tbl`: `code + bias` → `code - bias`.
   Expected: **S2, S4/Q4_0, S8 fail** (±2/±16/±256 value errors), `S4/IQ4_NL`
   stays green (table codebook doesn't use the bias).

Each: mutate → build → run (expect the named cases red, driver exit 1) →
restore → `check_mirrors.sh` green → full ctest green.

## 9. Bench spec (`--bench`, ctest `k_s_gemv_bench` with `--iters 50`)

Mirrors `bench_s2_gemv` (302–472) minus the variant sweep:

- data: codes 0xA5, scales 0.001, x = 1.0 (fp16 `0x3C00`) — mirrored;
- warm-up 3, min/median/mean ms — mirrored (`submit + wait` per iteration);
- prints: `G weights/s`, `GiB/s of S2 streamed`, per-token projection
  (48×10×3 roles, the NOTE block 353–354 verbatim) — mirrored;
- new honest-gap line: `split/quads/fast variants not ported (Phase B) —
  naive kernel only`;
- shapes: `[2560 x 640]` and `[640 x 2560]` (the engine's expert shapes).

The throughput number goes to `plans/sycl-phase-a-report.md` §T4 and is what
the parent plan's done-criterion ("GEMV throughput number in the report")
requires. T6 re-measures with more iterations. Checked: neither `Memory/LEDGER.md`
(noted in the kernel comments but not present in this checkout) nor
`docs/benchmarks/` carries a CUDA `s_gemv` number, so T4's Arc figure is the
first measurement and the report records it as such, with the CUDA
comparison target marked "not available in this checkout".

## 10. CMake (`poc/sycl/CMakeLists.txt`)

```cmake
# T4: the flagship S-family GEMV (plans/sycl-phase-a-t4.md)
add_library(k_s_gemv STATIC kernels/s_gemv.cpp)
add_executable(k_s_gemv_parity drivers/s_gemv_parity.cpp)
target_link_libraries(k_s_gemv_parity PRIVATE k_s_gemv)
add_executable(t4_fp16_probe t4/fp16_probe.cpp)

add_test(NAME t4_fp16_probe   COMMAND t4_fp16_probe)
add_test(NAME k_s_gemv_parity COMMAND k_s_gemv_parity --selftest)
add_test(NAME k_s_gemv_bench  COMMAND k_s_gemv_parity --bench --iters 50)
```

Order: after `k_router_top10_parity`, before the `t0_*` regressions; all four
targets get the existing `foreach` rpath/UMF treatment and `TIMEOUT 120`
(`k_s_gemv_bench` gets 300 — two shapes × 50 iterations of a serial-2560-element
row kernel, plus the default run, may exceed 120 s on this testbed).

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `sycl::fp16` missing/behaves differently on icpx 2026.1 | low (SYCL 2020 type) | Task 0 probe before the kernel; hand-rolled binary16→f32 fallback is bit-exact by construction |
| FMA contraction divergence pushes a case over 1e-4 | low (CUDA's own worst is ~1e-5 at this n_in) | tolerance is mirrored, not re-derived; if it fires at exactly one borderline row, the report records the number rather than loosening the mirrored threshold silently |
| Q4_K self-consistency ((a) vs (b)) fails on Arc's host CPU | very low (pure host C++, same as CUDA test) | the print exists precisely to separate "test built wrong planes" from "kernel wrong"; a failure here blocks belief in the GPU number, per the mirrored verdict |
| 64-bit division cost (`n_in / group_elems` per row) | n/a for parity; perf only | same expression as CUDA; T6 measures, no Phase A action |
| Bench timing skewed by testbed (PCIe Gen3, shared machine) | known | min (not mean) reported, mirroring the CUDA bench's own convention; report states the testbed caveat (T0 finding) |
| Driver mirror volume (25 blocks) → drift during assembly | medium | assemble block-by-block with `check_mirrors.sh` runs between, as in T2/T3 |

## 12. Done when

1. `k_s_gemv_parity` (default run) green: all 4 forms + Q4_K, ids of the
   verdict as printed (0 rows over tol, self-consistent Q4_K reference),
   selftest line present.
2. `t4_fp16_probe` green; `k_s_gemv_bench` green with throughput numbers
   recorded in the report.
3. `check_mirrors.sh` green (≈35 new blocks; tree total ≈73).
4. Both mutations caught with the required failure signatures; restore →
   full ctest green.
5. `env -u LD_LIBRARY_PATH ./build/k_s_gemv_parity` clean; `git status` clean
   under `poc/sycl/` + `plans/`.
6. `plans/sycl-phase-a.md` T4 marked DONE, `plans/sycl-phase-a-report.md`
   §T4 appended, this file updated with implemented reality, committed.

## 13. Effort

- Task 0 fp16 probe: ~1 h
- Kernel (10 mirror blocks, one barrier, no reductions): ~0.5–1 day
- Driver (~25 blocks, the largest port so far — 2 of the 634 lines' GPU
  sections become glue, the rest is verbatim): ~1–1.5 days
- Mutations + final pass + report: ~0.5 day

**~2.5–3.5 days total.** Well inside the parent plan's envelope if the fp16
probe passes clean; the dominant unknown is driver assembly time, not kernel
risk — this is the simplest kernel in the set to port (no shuffles, no
atomics), and the flagship status comes from what it validates, not how hard
it is to write.

---

## Implemented reality (done 2026-09-30, PASS)

Files: `t4/fp16_probe.cpp`, `kernels/s_gemv.cpp` (10 mirror blocks),
`drivers/s_gemv_parity.cpp` (33 mirror blocks), CMake targets
`t4_fp16_probe` / `k_s_gemv` / `k_s_gemv_parity` / `k_s_gemv_bench`.
82 mirror blocks green in the tree (was 38).

Parity at 1e-4, Arc Battlemage G31:

| Case | worst rel |
|---|---|
| S2/Q2_0 | 0.000e+00 (bit-exact) |
| S4/Q4_0 | 6.634e-07 |
| S4/IQ4_NL | 9.278e-06 |
| S8/Q8_0 | 9.649e-06 |
| S4/Q4_K | 2.814e-07 rel-to-terms (1.350e-06 rel-to-result, cond 2.775e+07) |

No form deferred — the {2,4,8}-bit template + runtime `has_offset` + the
local-memory `kIq4Nl` table cover everything. Q4_K matches the CUDA test's own
recorded 2.8e-07 figure. Mutations: offset-drop → Q4_K red (128/128, exit 1);
bias-invert → S2/S4/S8 red (384 rows), IQ4_NL green (table path), exit 1;
restore → 10/10 ctest.

Toolchain findings (new to the PoC): **no `sycl::fp16` in icpx 2026.1** — the
SYCL 2020 binary16 is `sycl::half` here; validated bit-exact against the host
`fp16_to_fp32` on 24 patterns by `t4_fp16_probe`. **`local_accessor` has no
`.data()`** — `&acc[0]` is the pointer idiom.

Bench (naive kernel, first Arc numbers): `[2560 x 640]` 0.3592 ms →
4.6 G weights/s (1.3 GiB/s S2); `[640 x 2560]` 0.0969 ms → 16.9 G weights/s
(4.9 GiB/s S2). 5 work-groups at the first shape — the underparallelised
baseline Phase B's split/quads/fast variants will beat; the CUDA sweep
(~184 GB/s-class) is the target to chase, not yet the comparison.

Full detail in `plans/sycl-phase-a-report.md` §T4.
