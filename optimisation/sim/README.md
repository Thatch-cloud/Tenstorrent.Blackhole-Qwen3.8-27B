# Simulator-first device-loop GDN gate

## Retained history ownership

`--retain-histories --conv --norm-gate --batch-conv --dma-windows
--packed-checkpoints --window-prefix` exercises early projection/temporary
release while preserving exactly five commit-history buffers and caller-owned
output. T2/seed1 run `20260906T155420Z-326` also used `--continuation` and passed
all3 restored continuations plus a stale-state control. T32/seed2 run
`20260906T155706Z-301` passed all32 outputs, recurrent/convolution prefixes and
prefix-copy checks with the stricter both-chip ownership guard. Both closed
cleanly. These gates do not certify multi-bucket hardware capture or performance.

## T32 static-verifier prerequisites

The wider experiment fills a complete 32-row tile; it does not change serving
or request-level proposal limits. Recurrence/norm run `20260906T134145Z-324`
(seed0) passed every output and recurrent prefix against serial T1 of the same
generated kernel, including rows 16-31. This is not an independent native oracle.

Run `20260906T134609Z-603` (seed1, T32) passed the batched convolution/DMA-window
chain, all packed recurrent/convolution histories, all 32 prefix-copy checks,
all 33 restored two-token continuations and one stale-state negative control.
Immutable entry/projected inputs, final convolution state and clean mesh close
also passed. These are functional simulator results, not timing evidence.

Ordered shared-page cache run `20260906T140728Z-1111` passed T32/seed2/start16383
with 1024 page-table columns: complete native BF8 cache equality on both chips,
unchanged input, omitted-write detection and repeat-after-restore. Hardware now
runs independent native full-layer correctness before full-model exactness,
rollback and the static cost curve. Host suites: 223 CI, 55 simulator-support,
40 speculative tests passed; shell syntax and whitespace checks passed.

## Reusable verifier input staging

T32 publication run `20260906T143759Z-322` passed all33 prefix selections for
two synthetic layers (seed2), with exact native/checkpoint contents, unchanged
inactive slots and immutable sources. The C++ DMA kernel is unchanged; this
widens only the audited shape guard. Clean close/exit0; no hardware timing claim.

`run-verifier-inputs.sh --rows 16` checks in-place host token, packed/singleton
position and RoPE updates at31/4095/16383, without changing captured buffer
addresses or page ownership. The RoPE output must exactly match the native
`rot_mats_decode` helper on both simulated chips.

T2 (`20260906T131724Z-315`) and T16 (`20260906T131732Z-436`) passed all three
updates, native RoPE comparisons, unchanged pages and clean close/exit0. The first
fixture failed on a Torch UInt32-versus-Int comparison, not on device data; its
expected token dtype was corrected to native UInt32 before retry. This is input
staging certification only, not full-model trace reuse or throughput.

T32 input staging also passed `20260906T150111Z-317`, including all32 singleton
positions and native RoPE at each of31/4095/16383. This is the prerequisite for
the wider two-block hardware replay test, not its result.

## Ordered shared-page cache writes

`run-ordered-cache.sh --rows 16 --start 31 --seed 1` compares a single ordered
device launch against native serial B1 paged-cache updates. It keeps native BF8
untilize/tilize and RMW ordering; only input staging and launch composition change.
There is no hardware fallback and no simulator performance claim.

Passed complete physical cache equality on both chips, immutable packed input,
omitted-write detection, repeat-after-restore, clean mesh close and exit0:

| Simulator fixture | T | Start | Seed |
| --- | ---: | ---: | ---: |
| 20260906T115610Z-325 | 2 | 63 | 0 |
| 20260906T115708Z-319 | 16 | 31 | 1 |
| 20260906T115722Z-460 | 8 | 63 | 2 |
| 20260906T115904Z-308 | 4 | 15 | 0 |
| 20260906T115916Z-538 | 1 | 65 | 2 |

Wide metadata: add `--page-columns 1024`. T16 at16383/seed1
(`20260906T124725Z-316`) and T2 at4095/seed2 (`20260906T124834Z-318`)
passed the same exact checks with1024 physical blocks. The first wide-table
fixture had only8 physical blocks and was rejected by the native capacity guard
before candidate dispatch; it was corrected, not treated as a kernel failure.

The hardware `attention-timing` suite now gates the ordered writer with real
weights and exact B1 SDPA. Its paired control uses the same batched attention
projections and B1 SDPA but retains serial cache-write dispatch, isolating the
writer change. No full-model or serving adoption follows from this layer gate.

## Verified first pass, 2026-09-06

Fused GDN publication has a separate `run-gdn-commit.sh --rows 16 --seed 1`
entry point. T2 seed0 (`20260906T120611Z-321`) and T16 seed1
(`20260906T120814Z-316`) passed every prefix for two synthetic layers on both
chips, including native inactive-slot canaries, external checkpoints, immutable
entry/history buffers, clean close and exit0. Runtime publication uses two
workers per layer;48 real layers would use96 workers per chip, which these
two-layer simulator fixtures do not certify. No commit timing claim is made.

- Norm/gate T2 (`20260906T064618Z-338`): both serial T1 calls completed; T2
  readback stalled until timeout. Generated hashes match the failing hardware case.
- Recurrence-only T2 (`20260906T064957Z-301`): exact output and every prefix state
  on both simulated chips, initial state unchanged, clean close; passed.
- Counter-handoff fix T2 (`20260906T070713Z-307`): passed all output/prefix-state
  comparisons and clean close. The packer's private CB5 received count must include
  the reader's initial full-ring push; otherwise feedback republishes16 rather
  than32 and the next token waits. All15 fixtures in the three-seed T1/2/4/8/16
  matrix subsequently passed exact outputs/prefix states, unchanged initial states,
  clean close and exit0. See the [execution ledger](../../docs/experiment-execution.md)
  for run IDs. Hardware follow-up34019033933 (`36b9eed`) then passed30 exact
  eager/trace cases,216 restored-prefix continuations and15 negative controls
  against the native implementation. No hardware speedup or full-model result
  follows from these synthetic functional gates.

## Running

### Convolution integration first pass

Add `--norm-gate --conv` and set `KERNEL_TIMEOUT=600` for the projected-input
composition fixture. It runs native serial convolution/gates, gathers their outputs,
then calls the device-loop recurrence/norm kernel once. Synthetic projected rows,
taps and state are compared against serial T1 calls of the same helper on both
simulated chips: every output, recurrent prefix and all four convolution prefixes.
This is not a convolution token-loop fusion, independent native recurrence oracle,
real-weight test, fast-dispatch trace test or performance result. The separate
`gdn-multitoken-conv` CI suite supplies real-weight native projections and compares
the composed path against native GDN before output projection, eager and traced.
That hardware gate passed all30 cases in run34019928513 (`8e75a95`).

Add `--continuation` with `--conv --norm-gate` to test restoration at every accepted
prefix (zero through all), then compare two new projected rows with an independent
serial-prefix snapshot control. It also requires a stale-final-state negative
control to differ. This remains synthetic functional evidence, not native-oracle
certification. New continuation results must pass before the extended hardware
suite is dispatched; use `KERNEL_TIMEOUT=900` for the added checks.
Continuation first pass: T2/seed0 `20260906T080711Z-332` passed; final guarded
adapter T1/seed1 `20260906T081239Z-328` and T4/seed1 `20260906T081508Z-676`
also passed every accepted prefix, detected stale-state controls and exited0.
See the ledger for the exact adapter hash and separate native hardware status.

`--output-projection --conv --norm-gate` also compares synthetic 3072x128 local
output projection with serial T1 projection on both chips. T2/seed0
`20260906T083558Z-338` passed. The simulator substitutes an identity collective;
real-weight 5120-wide projection and fabric reduction remain hardware checks.

`--model-adapter --conv --norm-gate` runs the full-model adapter's active/compact
state transfers with synthetic projected inputs. T2/seed0 `20260906T084858Z-363`
passed checkpoints0/1/2, final active state and preservation of all seven inactive
slots on both chips. This does not simulate the64-layer model or attention.

Add `--compact-prologue` to that adapter gate to test selected/final convolution
checkpoints and hoisted projected-row layout. T2/seed0 `20260906T092329Z-324`
passed all three checkpoint choices, final state and inactive-slot preservation.
Recurrent prefixes are still all materialized; no kernel math changed.
Full-model hardware follow-up34024642720 passed correctness and improved T8/T16
paired timings, but not T2. The experimental full-model selector retains the
previous path below T8; standalone simulator fixtures can still exercise T2.

Run new kernel changes here before dispatching hardware experiments. This entry
point requires the simulator library and slow dispatch, refuses hardware-allocation
flags, and has no hardware fallback. It uses the existing D-drive WSL distribution
`TT-Sim`, source-built TTNN and dual-Blackhole simulator under `/opt/ttsim`.

From PowerShell:

```powershell
wsl -d TT-Sim -u root --exec bash /mnt/d/Programming/Tenstorrent.Qwen-Runner-CI/optimisation/sim/run-gdn-multitoken.sh --source-root /mnt/d/Programming/Tenstorrent.Qwen-Runner-CI/hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source --rows 2 --norm-gate
```

`--source-root` names the read-only GDN source audit downloaded from hardware run
34009341359. All three native kernel SHA256 values must match the pinned hardware
sources; the simulator runs the same generated source strings and program builder
as hardware, without installing native custom-op registrations. Do not substitute
an upstream kernel with a different hash. Reports record generated hashes and the
local TTNN extension hash; runtime builds may differ from the hardware image.
Norm/gate also verifies the two runtime headers used by the CB5 counter handoff.
Use `--seed 0`, `--seed 1` or `--seed 2` to vary the deterministic fixture.

Each process builds a synthetic sequence and compares all token outputs and prefix
states on both simulated chips against serial T1 calls of the same kernel. This
is a liveness/functional first pass, not an independent native-oracle certification.
The hardware gate still requires native serial exactness, corrected continuation,
traces and paired measurements after simulator success. Start with T1/T2, then
T4/T8/T16. Omit `--norm-gate` to run the recurrence-only control.

JSON, logs and 30-second Python stack dumps reside in `/opt/ttsim/results` inside
the D-drive virtual disk. Default process timeout is 180 seconds plus 10-second
kill grace; `KERNEL_TIMEOUT` can bound a longer simulation. Simulator wall time is
not kernel latency or tokens/s. Slow dispatch, disabled SFPLOADMACRO, topology and
harvesting differences mean simulation does not certify production fast dispatch,
resource fit or fabric timing. No real-card reset or hardware job is launched here.
# Parallel causal convolution windows

Add `--window-prefix` to test selection from packed post-convolution windows;
add `--packed-checkpoints` to retain those windows instead of separate prefix
snapshots. Both require a multirow batched-convolution fixture; packed checkpoints
also require `--dma-windows`. `--continuation` validates the new restore path,
including prefix0 and stale-state detection. Standalone selection passed T4/seed2
and T16/seed1 (`20260906T103225Z-321`, `20260906T103340Z-466`); packed integration
passed T2/seed0 and T16/seed1 (`20260906T103845Z-544`, `20260906T104119Z-721`).

Add `--conv --norm-gate --batch-conv` to `run-gdn-multitoken.sh` to compare one
native convolution/gates call over causal token windows against serial T1 calls.
Use `--continuation` for every accepted-prefix restore and stale-state control,
or `--output-projection` for the synthetic local projection check. Add
`--dma-windows` to test aligned full-tile staging and local causal-window assembly
instead of the layout/concat/slice chain. `--model-adapter --compact-prologue`
checks active-slot publication with sparse checkpoints. T2/seed0, T16/seed1 and T4/seed2 passed
in fixtures `20260906T100224Z-326`, `20260906T100449Z-630` and
`20260906T100806Z-838` without DMA. Corrected DMA T2/seed0 and T16/seed1 passed
in `20260906T101726Z-332` and `20260906T101946Z-516`. These are functional
checks, not hardware timings. The discarded32-byte direct-read prototype failed
the simulator NOC-alignment check in `20260906T101509Z-321`; never run it on cards.
