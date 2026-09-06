# Simulator-first device-loop GDN gate

## Verified first pass, 2026-09-06

- Norm/gate T2 (`20260906T064618Z-338`): both serial T1 calls completed; T2
  readback stalled until timeout. Generated hashes match the failing hardware case.
- Recurrence-only T2 (`20260906T064957Z-301`): exact output and every prefix state
  on both simulated chips, initial state unchanged, clean close; passed.
- The norm/gate bug is reproducible locally; its internal cause is not yet proved.
  Do not dispatch another hardware kernel run until the simulator gate passes.

## Running

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
