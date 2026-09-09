# Smaller activation tiles: simulator gate

**Combined simulator gate passed; no hardware timing or throughput claim.** The current request
profile measures32.050 ms of summed matrix operations per chip. This experiment
tests16-row activation tiles for T8, retaining native BF4 gate/up, BF8 down,
LoFi/FP32 destination accumulation, packer L1 accumulation and the native1D grid.
Weights stay unchanged. Input conversion and return to32-row tiles are included.

The opt-in simulator probe uses deterministic synthetic matrices at the exact
local target MLP shapes, not learned target weights. It requires exact BF16
equality against native32-row projections, changed-input replay with both traces
live, immutable input storage and stale-input negative controls on both chips.
All three projections and the combined MLP now pass on both simulated chips.
These are synthetic-weight correctness checks, not learned-model quality tests.

| Simulator attempt | Outcome |
| --- | --- |
| 20260909T013208Z-418 | Native config import required pytest; installed pytest9.1.1 in the TT-Sim venv |
| 20260909T013318Z-423 | Original packer hits known unsupported `Disable_pack_zero_flags`; not a numerical failure or pass |
| 20260909T013548Z-409 | Audited packer compatibility graft retains accumulation; output retile then exceeds L1 before comparison |
| 20260909T015215Z-393 | Gate plus bounded DMA conversion: exact eager and changed-input trace pass |
| 20260909T015534Z-398 | Up plus bounded DMA conversion: exact eager and changed-input trace pass |
| 20260909T015742Z-541 | Down plus bounded DMA conversion: exact eager and changed-input trace pass |
| 20260909T020632Z-421 | Combined MLP fails: stock multiplication does not preserve small tiles |
| 20260909T022227Z-403 | Small-tile FPU multiplication rejected by exact hidden-output comparison |
| 20260909T022751Z-385 | Native SFPU multiplication semantics: full MLP passes 16 eager and 32 traced bitwise comparisons |

The original generic return conversion allocates2,061,184 bytes of static circular buffers on one core,
exceeding1,572,864 bytes of L1. It reaches the16-row matmul and fails converting
its8,704-wide result back to32-row tiles. This is a concrete conversion-resource
failure, not evidence that smaller matmul tiles are numerically correct or fast.
The replacement copies BF16 faces with 2 KiB scratch per worker, without arithmetic.
The combined MLP performs only two DMA conversions: input32-to16 and output16-to32.
Gate/up, their product, and down remain in16-row tiles between those boundaries.

The pinned default `ttnn.mul` uses SFPU multiplication, not FPU multiplication.
Its stock binary factory defaults to32-row output pages. The custom product uses
explicit16-row circular buffers and the same SFPU operation and destination modes.
No relaxed tolerance, changed weights or disabled accumulation is used.

The composed gate checks every gate/up/product/down output bit, changed inputs
with both traces live, persistent addresses, input immutability, and two stale-input
negative controls. It closes both devices cleanly. Simulation duration is not a
hardware performance measurement; no collective is included in this simulator gate.

Next hardware gate: `tiny-mlp`, actual layer0 weights, native `Qwen36MLP.forward`
control, three changed inputs and nine ABBA blocks. Both arms include input staging
and the same four-link native reduce-scatter; candidate timings include both DMA
boundaries. Require bitwise native equality, then a greater-than2% win in every
block before a full-model test. A single-layer win is not a PP/CTX/TG result.
Hardware refuses incomplete or source-mismatched simulator evidence.
Hardware run [34303979499](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34303979499)
is in progress on `6457ec8`; no result is available yet. The original hardware
packer remains unchanged; simulator compatibility is not hardware qualification.

The existing audited patch is `optimisation/sim/blackhole-packer-zero-flags.patch`.
The isolated graft header hash is
`8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7`.
After the successful simulator gate, the original header is restored and checked:
`87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181`.
Logs/reports/status files are preserved under `hardware-evidence.local/tiny-tile-sim/`.
The original combined report SHA256 is
`06dba0d7163b17154f30e22541b957e071bd38198d755ee9f93cd1ba713833f3`.
The checked-in JSON has an added terminal newline, with SHA256
`135bf57430646e0e4a3fabe9cbd03fb7d5670673eb2eca0a680473347c850f51`.
All nine experiment-source hashes are revalidated before hardware use.
875 CI host tests and55 simulator-harness tests pass; neither count is hardware proof.
