# LLK profiling harness (P0)

Status: built and validated on the CPU only. Nothing here has run on a card, on TT-Sim or in an image yet.
Numbers marked *(e)* are estimates; *(m)* measured; *(u)* unverified.

This is the P0 measurement harness of the LLK programme (task sheet L0-03, L0-04, L0-05, L0-06, L0-09,
L0-10, L0-11, L0-12). It answers one question per C2 hotspot kernel and per thread (unpack, math, pack,
and the two data-movement RISCs): where does the time go, and is the kernel bound by the LLK threads or by
dataflow? Every hardware part is a single set of gate arms (`llk-*`) that runs in the combined M+A window,
after every judged run.

## What it is made of

| Module | Role |
|---|---|
| `scripts/ci/llk_zones.py` | Exact, reversible source transforms. Makes zone-instrumented *copies*; `remove()` gives back the original bytes, and `instrument()` refuses to return a copy for which that does not hold |
| `scripts/ci/llk_kernels.py` | The hotspot registry: which kernels, their stage anchors, their pins, and how each copy reaches the JIT |
| `scripts/ci/llk_zone_override.py` | In the serving worker, when `QWEN_LLK_ZONES` is set: instruments the K5-A build after qualification |
| `scripts/ci/llk_profile_report.py` | Device-log parser and the per-kernel, per-thread LLK-vs-dataflow report |
| `scripts/ci/llk_profile_plan.py` | The gate's `llk-*` plans, the per-arm prepare/hand-back/finish, the no-card preflight |
| `scripts/ci/c2_serving_gate.py`, `c2_serving_job.py` | Wiring: the plans, the `llkcheck` action, the ordering rule |
| `.github/workflows/qwen-c2-serving.yml` | The `llkcheck` preflight step (no card) and an `always()` hand-back step |

### Instrumentation (llk_zones)

Three kinds of insertion, each an exact string tagged `qwen-llk-zone`:

- **Envelope.** The kernel entry point's whole body in one block zone `QWEN_LLK_<KEY>`. In a compute kernel one
  zone yields three per-thread intervals (the source is compiled once per TRISC). The kernel's identity is in the
  zone name, so the report needs no op table.
- **Region.** `[begin anchor, end anchor)` in one block zone `QWEN_LLK_<KEY>_<STAGE>`. Both anchors are exact,
  unique, line-initial strings. The region must be brace-balanced, declare nothing at its own depth, carry no
  case label, keep `#if`/`#endif` balanced and name no `hash` or `zone` (the profiler macro declares both).
- **Sync sums.** Every blocking synchronisation call statement is wrapped in a sum zone
  (`DeviceZoneScopedSumN1/N2`: no marker per execution, one `ZONE_TOTAL` row per program):

  | Thread | `QWEN_LLK_WAIT_IN` (upstream) | `QWEN_LLK_WAIT_OUT` (downstream) |
  |---|---|---|
  | TRISC_0 unpack | `cb_wait_front`: starved by the reader | – |
  | TRISC_1 math | – | `tile_regs_acquire`: dest still held by the packer |
  | TRISC_2 pack | `tile_regs_wait`: waiting on math | `cb_reserve_back`: writer back-pressure |
  | BRISC / NCRISC | read barriers, producer rings, semaphores | write barriers, flushes, ring space |

  Math waiting on unpack is a hardware stall (srcA/srcB valid) that no software zone can see; it comes from the
  counter pass (`WAITING_FOR_SRCA_VALID` / `SRCB_VALID`) or is reported undetermined.

Levels: `tag` (envelope only; names the kernel a counter row belongs to) and `stages` (envelope, regions, sums).
The marker lint allows at most 250 − 32 optional markers per RISC per program (v0.77.0
`PROFILER_L1_OPTIONAL_MARKER_COUNT` is 250); a zone costs two per execution. The instrumented K5-A compute uses
168, the K64j SDPA decode compute 52.

### Delivery: no stock or pinned source is edited

- **Generated (K5-A).** `gdn_seq_block` builds its kernels in-process and hands them to `generic_op` as source
  text, qualified by a sha256 triple. `llk_zone_override` wraps `gdn_seq_block.served_kernels` and instruments the
  qualified text in memory. The compile argument `SRC_TAG` is the instrumented text's sha256 prefix, so the JIT
  compiles the copy under its own cache key and never reuses or overwrites a served binary. `serving_runtime`
  imports the override only when `QWEN_LLK_ZONES` is set, and the override refuses to run without
  `TT_METAL_DEVICE_PROFILER=1`.
- **File (compiled ops).** The gate reads the image's own kernel bytes in a throwaway container (no devices, no
  network), instruments them on the host and bind-mounts each copy read-only over its path, on `llk-*` arms only.
  `TT_METAL_KERNEL_PATH` would not work: tt-metal resolves a relative kernel path against the working directory
  first, and the image's working directory is `/opt/tt-metal`. A file mount also keeps relative `#include`s working.
  Only kernel sources under `ttnn/cpp/ttnn/operations/**/kernels/` may be mounted; model files, grafts, pinned
  recipe files and binaries are refused.
- **Discover.** As file, but found in the image by pattern (PR-stack ops whose kernel paths this repo does not pin).

A pinned file whose image bytes differ from the pin gets the envelope and sums only (no anchors needed), and the
manifest says so.

### The registry (first version)

| Key | Kernel | Route | Stages | Pin |
|---|---|---|---|---|
| `K5A` | GDN recurrence K5-A compute | generated | PRO, T1, T23, T45, T6, T7, EPI (16 tokens) | the qualified build |
| `K5A_RD`, `K5A_WR` | K5-A reader and writer | generated | – | the qualified build |
| `SDPA_DEC` | K64j SDPA decode compute | file | KLOOP, TREE, FINAL (per head, at most 8 *(u)*) | K64j `K64J_COMPUTE_QWEN` |
| `SDPA_DEC_RD` | K64j SDPA decode reader | file | – | K64j `K64J_READER_QWEN` |
| `SDPA_COMMON` | SDPA compute helpers header | file | sums only | none |
| `ATTN_PREP` | AttnPrep compute | file | – | repo copy; image unverified *(u)* |
| `MM` | stock matmul compute (prefill only) | file | – | none |
| `SDPA_PF` | SDPA prefill compute | file | – | none |
| `CONV_GATES` | GdnConvGates compute | discover | – | path unpinned *(u)* |
| `AGMM` | prefill gate/up all-gather matmul compute | discover | – | path unpinned *(u)* |

Required kernels, per phase: `K5A` and `SDPA_DEC` in decode, `SDPA_PF` in prefill; each on TRISC_0/1/2 of both
chips. An arm with no required kernel would measure nothing, so the report refuses an empty list. The stock
matmul is instrumented in prefill only: in decode it runs on most cores every round, and its markers would crowd
the profiler buffer that the K5-A stage zones need.

### The report (llk_profile_report)

Reads only `QWEN_LLK_*` rows and counter rows (timer id 9090) of `profile_log_device.csv`, streaming. Pairs
START/END per execution key (chip, core, RISC, run host id, trace id, trace id counter); unmatched markers are
counted as drops, never hidden. Wait sums belong to the envelope of the same key.

Each arm is read over one window. A decode arm reads complete trace replay sessions only (`traced`): a session
with a dropped marker, or with fewer envelopes of any kernel than the fullest session of its trace on that chip,
is left out and counted, and every required kernel needs at least two complete sessions per chip. The profiler
buffer truncates (once it is full, every later marker is lost until the next read-back), so the sessions before
an overflow are whole. A prefill arm reads the untraced rows, and any drop there makes it incomplete.

A wait slot that a thread waits in (TRISC_0 `WAIT_IN`, TRISC_1 `WAIT_OUT`, TRISC_2 both) must have its sum row on
every invocation of a kernel whose sums were on. A missing row is undetermined, never zero: the define may not
have reached the worker, the zones may have compiled out, a zone-id collision may have renamed it, or the row may
have been dropped. Reading it as zero would class the kernel as LLK-bound. Such a kernel is classified
`undetermined`, and a zones arm with one is `NOT_EXERCISED`. Per kernel and thread:
envelope statistics, WAIT_IN, WAIT_OUT, busy; per stage: instances, per-thread statistics, the unpack/math
lockstep evidence, static reconfiguration counts times instances. Classification:
`dataflow-in`, `dataflow-out`, `llk-pack`, `llk-unpack`, `llk-math`, `llk-math-sfpu`, `llk-unpack-or-math`
(no counters), `llk-balanced`, data-movement classes, or `undetermined`; thresholds are stated in every report.
Ranked by the kernel's summed critical duration. A perturbation line compares with the unprofiled twin.

## Traps this harness is built around

- **The qualified recipe writes no raw device log.** `lever_n_m3native_run_arm.sh` passes tracy
  `--disable-device-data-dump-to-files`, which in v0.77.0 makes `DeviceProfiler::writeDeviceResultsToFiles`
  return before `profile_log_device.csv` is written (*(m)* from the source; no local copy of any past run holds
  that file). The `llk-*` arms never pass it, and `check_tracy` refuses it.
- **Op-support count.** 20000 only; 200000 segfaulted the dispatch thread (v129, v131, v135). Refused above 20000.
- **Root-owned output.** The profiler writes as the container's root. Each arm's tree is handed back in a
  `finally` (a SIGTERM included), and the workflow's `always()` step does it again before the upload and prunes
  any raw log the gate did not reach. Still: never cancel a job with `llk-*` plans while an arm runs.
- **Output location.** `<arm>/llk-profile` is bind-mounted at tt-metal's default artifacts directory, so the
  output is outside the checkout and out of the agent shape's 1 GB `/opt/tt-metal/generated` tmpfs. tracy's `-o`
  is that same directory, and so is `TT_METAL_PROFILER_DIR`. tracy hands its `-o` to the child process (v138's
  logs landed under `-o` with no `-e` set), so a different `-o` would make the arrival check fail every profiled
  arm. `check_tracy` refuses any other `-o`. The arrival check accepts the directory, or one inside it, as tracy
  writes it back.
- **The profiler buffer overflows.** Each RISC's DRAM buffer holds about 20000 programs' guaranteed markers at
  op-support 20000, and nothing leaves it until a read-back. v138 (the same 4 × 32768 shape, no zones, no drain)
  logged about 1,100 "Profiler DRAM buffers were full, markers were dropped!" lines at its round-4 read-back: the
  four prefills alone are about 19,600 programs per core *(e)*. So every profiled arm:
  - drains the prefill (`QWEN_PREFILL_PROFILE_FLUSH=1`: a read-back every 16 layers), and a missing drain
    marker makes the arm `NOT_EXERCISED`;
  - reads back again after packed round 2 (`QWEN_FAST_PROFILE_DUMP_ROUND=2`), the only read-back that is
    certain, because a clean close is not;
  - carries no matmul zones in decode;
  - has the markers between two drains budgeted before any container starts (`marker_budget`: at most 80% of
    the buffer). The decode zones arm is at about 62% *(e)*, the counters arm at about 48% *(e)*, and the prefill
    arms at about 1%.

  The drain every 16 layers has not run at op-support 20000 on hardware. Its one earlier run (v131) segfaulted at
  the first drain at op-support 200000, and 200000 alone segfaulted without it (v135). About 256 drains per four
  prefills add minutes, not hours *(e)*.
- **The raw log can fill the disk.** A decode arm's `profile_log_device.csv` is tens of GB *(e)*. Before a
  profiled arm, the results disk must stay under 80% used with 48 GB (decode) or 10 GB (prefill) written; if not,
  the arm is `NOT_EXERCISED` without a container. While it runs, a guard stops its container if the profile tree
  passes 64 GB (decode) or 16 GB (prefill), or the disk passes 85% used. The raw log is pruned as soon as it has
  been exported and hashed once.
- **Every profiled arm compiles cold.** Its JIT cache is a per-container tmpfs, and the C2 cold-compile time has
  never been measured. Each arm's docker limit is the gate's 1800 s readiness plus its stream timeout plus 300 s.
  If one profiled arm's server is not ready in time, the later profiled arms are skipped rather than each
  waiting out the same allowance. A server that crashes before readiness fails fast and skips nothing.
- **Kernel cache.** Profiled arms set `TT_METAL_CACHE` to the agent shape's per-container 8 GB tmpfs: profiler
  and instrumented compiles never enter the image's shared cache that production reads. Whether a full cold JIT
  fits in 8 GB is *(u)*.
- **Graft mounted is not graft executed.** The override logs a `[LLK] record {...}` line per part; the gate reads
  them back into the manifest, and coverage requires the zones in the device log itself.
- **Sums or counters versus tracy's C++ post-processing.** tracy disables its C++ post-processing when sums or
  counters are asked for; the arms therefore ask through tracy's own flags (`--enable-sum-profiling`,
  `--profiler-capture-perf-counters`) and read only the raw device log.

## Hardware arms for the combined run

One gate job (`status reset gate`), profile `c2-packed`, after every judged job of the window:

```
C2_ACTIONS=status reset gate
C2_PROFILE=c2-packed
C2_GATE_PLAN=llk-decode-twin,llk-prefill-twin,llk-decode-zones,llk-prefill-zones,llk-decode-counters,llk-prefill-counters
```

| Arm | Shape | Adds | Docker limit | Estimate |
|---|---|---|---|---|
| `llk-decode-twin` | 4 × 32768 real text, 48 out | nothing (served bytes) | 65 min | 10-12 min *(e)* |
| `llk-prefill-twin` | 1 × 32768, 1 out | nothing | 50 min | 6-8 min *(e)* |
| `llk-decode-zones` | as the decode twin | profiler, prefill drain, round-2 read-back, sums, `QWEN_LLK_ZONES=stages`, file overlays, scratch cache | 65 min | 30-45 min *(e)*, including a cold compile *(u)* |
| `llk-prefill-zones` | as the prefill twin | as above | 50 min | 20-30 min *(e)* |
| `llk-decode-counters` | as the decode twin | as the zones arm, but counters FPU+PACK+UNPACK+L1_0+INSTRN (47) and `QWEN_LLK_ZONES=tag` | 65 min | 25-40 min *(e)* |
| `llk-prefill-counters` | as the prefill twin | as above | 50 min | 15-25 min *(e)* |

Worst case, every arm to its limit plus overheads: 5.95 h, inside the gate step's 380 minutes. Expected:
2-3 h *(e)*, most of it cold compiles, drains and the exports of the raw logs.

Before the window, in the no-card build job: `C2_ACTIONS=status build llkcheck` runs the preflight, which makes
every instrumented copy the arms will mount from the image's own bytes, instruments the image's K5-A build
inside the image, and checks every arm's marker budget. An anchor that drifted, a profiler capability the zones
arms need, or a window past the budget fails there. The step is `continue-on-error`: it never keeps the image
from its push, so read its outcome and `llk-preflight/llk-preflight.json` before the window, and drop J3 if it
failed.

Artefacts per profiled arm: `llk-manifest.json` (capabilities, every record, every mount with its sha256),
`llk-export.csv.gz` (the filtered rows, with the full log's size and sha256), `llk-report.json`,
`llk-pruned.json`, the argv and the server log. The raw device log leaves the results tree after export.

Judged on:
- **Text.** Each stream's generated text (the harness report's `streams[i].text`) is identical to the twin's.
  The copies are byte-reversible, so a difference is a finding (`FAIL`). A stream without text is a `FAIL`, and a
  twin without text leaves the arm `NOT_EXERCISED`: absent values are never compared as equal.
- **Coverage.** Every required kernel has zones on every compute thread of both chips; in decode, at least two
  complete replay sessions per chip; on a zones arm, every wait sum; on a counters arm, counter rows on both
  chips.
- **Drops.** None in the analysed window.
- **The drain.** The prefill drain's marker was seen.

Otherwise the arm is `NOT_EXERCISED` with the reason. `UNSUPPORTED` counters skip the other phase's counter arm
without starting a container. No `llk-*` result is a timing result, and none blocks a placement.

## Not done yet (before the image freeze)

- Compile every instrumented copy once on TT-Sim (task L0-03's exit criterion). The CPU checks prove the copies
  are byte-reversible and lint-clean, not that they compile, nor that TRISC code size stays inside its limit.
- Confirm on the image (the preflight does this): tracy's options, sum-zone macros, counter readout, the marker
  budget, and the real paths of `CONV_GATES` and `AGMM`; re-pin `ATTN_PREP` against the image's bytes.
- The K5-A clock-page arm (`llk-k5a-clock`, task L0-08) is designed, not built; the job parser refuses it by name.
- The buffer model (`ROUND_PROGRAMS`, `LAYER_PROGRAMS`, the invocations per round) is estimated from v138's
  per-op report *(e)*. The traced window is what tolerates it being wrong: an overflow after round 2 costs
  sessions, not the arm.
- Zone ids are 16-bit hashes of the zone name, file and line, and each wrapped wait call is its own site. A
  collision is not detected; it shows up as a missing sum and so as `undetermined`, never as a zero wait.
