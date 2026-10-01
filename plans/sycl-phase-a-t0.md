# T0 — detailed plan: environment verification + SYCL sanity spike

Parent: `plans/sycl-phase-a.md` (Phase A plan, task T0). This document is the
executable specification for T0: nothing is procured, everything is already on
this machine. T0 ends either **green** (spike passes the gate → start T1) or
**stopped** (a written failure analysis, per the Phase A stop rule).

## Gate (unchanged from the Phase A plan)

A 100-line-class SYCL program runs the target GPU and reports ≥ 60% of rated
memory bandwidth in a streaming kernel. Listing the device in `sycl-ls` proves
the driver is present; the spike proves the runtime path works end-to-end.
These are different claims.

Reference hardware (verify in T0.1, don't assume):

- GPU: Intel Arc **Battlemage G31** (PCIe id 0xe223; the B570/B580 class die).
  Rated memory bandwidth for this class: **512 GB/s** (256-bit GDDR6, 16 Gbps —
  confirm the exact SKU's spec in T0.1 from the driver/Intel spec and record it;
  the gate number below is derived from it).
- Pass target: effective bandwidth ≥ **300 GB/s** (60% of 512 GB/s), where
  effective = total bytes read + written ÷ kernel time.

## Task T0.1 — environment verification (≈½ day)

Run in order; record each command's actual output in the report. Each row is a
gate: any failure goes to the diagnostic tree at the end, not to a workaround.

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | Toolkit activates | `source ~/intel/oneapi/setvars.sh && icx --version` | `Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1` |
| 2 | GPU visible on Level Zero | `sycl-ls` | `[level_zero:gpu][level_zero:0] ... [0xe223]` (already confirmed once; re-confirm in the same shell session that will build) |
| 3 | Driver/driver-runtime match | compare the Level Zero driver version printed by `sycl-ls` against `ls /usr/lib/x86_64-linux-gnu/` `ze_*.so` (or `lsmod \| grep intel`) | one coherent driver generation; record both versions |
| 4 | SYCL toolchain can target the GPU | compile+run a 10-line "print device name" program (below) with `icx --fsycl` | prints `Intel(R) Graphics [0xe223]` on backend `level_zero` |
| 5 | SKU + rated bandwidth | `lspci -vvv -d 8086:e223` / device queries (`max_mem_size`, mem bus width) + Intel spec for the confirmed SKU | record SKU name, VRAM, **rated GB/s** — this number sets the gate |
| 6 | GPU health/power | `intel_gpu_top -i 0 -s 2` (or `intel_metrics`) during a busy moment; check for throttling | clocks sustained, no "power/thermal limited" |
| 7 | Isolation | confirm nothing else is using the GPU (`nvidia-smi` irrelevant here; check `intel_gpu_top` for other workloads; the box is a Xeon Gold 6148 server, so watch for VMs) | Arc GPU idle before measurement |

Check 4 program (compile once with `icx --fsycl -O2 dev0.cpp -o dev0`):

```cpp
#include <sycl.hpp>
#include <cstdio>
int main() {
    // Select explicitly by name so the OpenCL CPU (Xeon) or the OpenCL NEO GPU
    // can't be picked: this box has three devices.
    sycl::device dev([](const sycl::device& d) {
        return d.is_gpu()
            && d.get_backend().is<sycl::backend::level_zero>()
            && d.get_info<sycl::info::device::name>().find("e223") != std::string::npos;
    });
    std::printf("%s | backend=%s | subgroup max=%u | shared mem max=%zu\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_backend().string().c_str(),
                dev.get_info<sycl::info::device::max_sub_group_size>(),
                dev.get_info<sycl::info::device::local_mem_size>());
}
```

Record from it: **`max_sub_group_size`** and **`local_mem_size`** (max shared
memory per work-group). These two numbers are inputs to Phase B sizing (the
kernels assume 32-wide subgroups and up to 64 KiB shared; Battlemage is 16-wide
EU SIMD — expect `max_sub_group_size = 16` — and T3's work-group mapping
decision depends on what the device actually reports).

## Task T0.2 — the bandwidth spike (≈½ day)

One file, no dependencies beyond SYCL: `poc/sycl/t0/bw_spike.cpp`.

Design rules (keep them; they make the number mean something):

- **Device selection** by name (as above), never by index — the box also has an
  OpenCL CPU device; `sycl::default_selector` is not safe here.
- **Buffer**: 256 MiB per array, USM device (`usm_device_alloc`); big enough to
  beat any cache effects, small enough to stay in VRAM.
- **Measure** four things, each timed with **SYCL events** (device-side timing,
  not host wall clock), 1 warmup + 5 timed runs, report the median:
  1. **H2D** — `queue.memcpy` host→device (PCIe path; reference only, not gated).
  2. **D2H** — device→host (same).
  3. **D2D copy** — `queue.memcpy` device→device (1 read + 1 write stream;
     gates as a sanity floor: ≥ 60% rated).
  4. **Triad kernel** — a `par_kernel`/ND-range kernel doing `c[i] = a[i] +
     a[i+1] + b[i]` over float arrays (2 reads + 1 write = 3 streams). This is
     the number the Phase A gate quotes. Coalesced 128-bit accesses, one
     work-item per 4 floats, work-group size a power of two ≤ 256 (try 64 and
     256, keep the better, record both — the spread is a Phase B data point).
- **Print** for every run: device name, array size, access pattern, median
  GB/s effective, per-run values. Exit code 0 iff triad ≥ 300 GB/s
  (parameterize the threshold so T0.1's confirmed SKU number feeds it).

The triad kernel is deliberately the *shape* of Strata's memory-bound work
(streaming dequant/expert loads), so a pass here predicts that
`dequant_s2` will be viable; it is not a claim about compute kernels.

Build & run (this exact command pair becomes `run.sh`'s T0 section in T1):

```sh
source ~/intel/oneapi/setvars.sh
mkdir -p poc/sycl/t0 && cd poc/sycl/t0
icx --fsycl -O2 -std=c++20 bw_spike.cpp -o bw_spike
./bw_spike
```

Expected first-run artifacts: the binary, and a captured run transcript
(pasted into the report).

## Task T0.3 — record & decide (≈2 h)

Write `plans/sycl-phase-a-report.md` section **T0** containing:

1. T0.1 table with real outputs (versions, SKU, rated GB/s, subgroup/shared-mem
   numbers, health observation).
2. T0.2 full transcript + the gated decision line: **PASS / FAIL with the
   measured number vs the threshold**.
3. If PASS: one paragraph on what Phase B can assume (e.g. "16-wide subgroups
   confirmed; X KiB shared per work-group; triad at Y GB/s").
4. If FAIL: the diagnostic tree below, what was tried, and a stop
   recommendation (Phase A does not continue on a red T0).

## Diagnostic tree (only if the gate fails)

Ordered by likelihood; stop at the first that explains the number:

1. **Wrong device picked** — did it run on the OpenCL CPU (Xeon) or the OpenCL
   NEO GPU? Check the printed device name. Fix: name-based selector (already
   mandated). A CPU run of 20–60 GB/s looks alarming but is a selector bug, not
   a runtime bug.
2. **PCIe, not device, path** — D2D is fine but H2D is ~24 GB/s: that is a 5.0
   x16 link and is *expected*; H2D is reference-only and never gated. Do not
   "fix" this.
3. **Driver/runtime mismatch** — Level Zero V2 runtime vs older driver (or vice
   versa): check `sycl-ls` driver version vs `ze_loader` version; install the
   matching generation of the `intel-level-zero-gpu` driver package.
4. **Throttling** — `intel_gpu_top` shows low clocks under load: check
   `intel_metrics` power/thermal limits, GPU P-states, and whether the server
   power budget is starving the card. Record sustained clocks.
5. **Kernel too naive** — if D2D copy is fast but triad is slow: the triad is
   likely compute- or layout-limited. Before concluding anything, (a) bump
   vector width to 32-bit×4, (b) try work-group 64 vs 256, (c) check the
   compiler used `-O2` and no fallback device. If triad ≥ 60% after these, the
   earlier number was a kernel artifact, re-record and pass.
6. **Genuinely slow/unsupported path** — none of the above; then T0 is a
   STOP: write the failure analysis with the measured numbers and stop Phase A
   per the parent plan's stop rule.

## Deliverables

- `poc/sycl/t0/bw_spike.cpp` (+ `dev0.cpp`) — committed as-is; T1's `run.sh`
  re-runs them so a green T0 stays re-verifiable.
- `plans/sycl-phase-a-report.md` §T0.
- Updated `plans/sycl-phase-a.md`: tick T0 done (or note the stop).

## Time

- T0.1: ≤ ½ day (all local; the long pole is finding the SKU spec and health
  check).
- T0.2: ≤ ½ day (the code is ~100 lines; debugging, not typing, dominates).
- T0.3: ≤ 2 h.
- **Total: ≤ 1 day**, consistent with the Phase A effort table; the only
  realistic overruns are the diagnostic-tree branches, which are themselves
  bounded by the stop rule.
