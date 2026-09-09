# Smaller activation tiles: simulator gate

**Not qualified; no hardware timing or throughput claim.** The current request
profile measures32.050 ms of summed matrix operations per chip. This experiment
tests16-row activation tiles for T8, retaining native BF4 gate/up, BF8 down,
LoFi/FP32 destination accumulation, packer L1 accumulation and the native1D grid.
Weights stay unchanged. Input conversion and return to32-row tiles are included.

The opt-in simulator probe uses deterministic synthetic matrices at the exact
local target MLP shapes, not learned target weights. It requires exact BF16
equality against native32-row projections, changed-input replay with both traces
live, immutable input storage and stale-input negative controls on both chips.
Only the gate projection has been attempted; up/down remain untested.

| Simulator attempt | Outcome |
| --- | --- |
| 20260909T013208Z-418 | Native config import required pytest; installed pytest9.1.1 in the TT-Sim venv |
| 20260909T013318Z-423 | Original packer hits known unsupported `Disable_pack_zero_flags`; not a numerical failure or pass |
| 20260909T013548Z-409 | Audited packer compatibility graft retains accumulation; output retile then exceeds L1 before comparison |

The last run allocates2,061,184 bytes of static circular buffers on one core,
exceeding1,572,864 bytes of L1. It reaches the16-row matmul and fails converting
its8,704-wide result back to32-row tiles. This is a concrete conversion-resource
failure, not evidence that smaller matmul tiles are numerically correct or fast.
The interleaved retile factory stages an entire width; a bounded-memory boundary
conversion is needed before repeating the gate. Do not bypass conversion costs
or disable packer accumulation to make the test pass.

The existing audited patch is `optimisation/sim/blackhole-packer-zero-flags.patch`.
The isolated graft header hash is
`8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7`.
After the terminal simulator failure, the original header is restored and checked:
`87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181`.
Logs/reports/status files are preserved under `hardware-evidence.local/tiny-tile-sim/`.
864 CI and60 harness tests pass; these are host tests, not the device gate.
