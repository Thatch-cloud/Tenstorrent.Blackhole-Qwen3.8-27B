# Simulator-first device-loop GDN gate

## Verified first pass, 2026-09-06

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
