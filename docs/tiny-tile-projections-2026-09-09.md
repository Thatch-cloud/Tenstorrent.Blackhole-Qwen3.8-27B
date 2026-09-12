# Smaller activation tiles: correctness passes, hardware regresses

**Rejected for speed: 4.70% slower on real target weights.** The current request
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

Hardware gate: `tiny-mlp`, actual layer0 weights, native `Qwen36MLP.forward`
control, three changed inputs and nine ABBA blocks. Both arms include input staging
and the same four-link native reduce-scatter; candidate timings include both DMA
boundaries. Require bitwise native equality, then a greater-than2% win in every
block before a full-model test. A single-layer win is not a PP/CTX/TG result.
Hardware refuses incomplete or source-mismatched simulator evidence.
Hardware run [34303979499](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34303979499)
on `6457ec8` is **not a valid MLP pass**. CI is green, but the shell exits after
the CCL prerequisite; no `tiny-mlp.json` exists and the MLP script never runs.
The corrected route continues to the real test. Executed shell-routing tests
cover that transition and failed build/link/MLP cases. The host wrapper now
requires the MLP artifact and independently validates sources, six eager checks,
12 traced checks, all nine ABBA blocks and every summary against raw samples.
The original hardware packer remains unchanged; simulator compatibility is not
hardware qualification. These failed attempts produce no MLP timing; the completed
result below is from the corrected route and audited native-source gate.
The routing/required-artifact fix passes883 CI host tests and60 speculative-harness tests.

The routed retry34304793770 reaches the MLP source gate but rejects `tp_common.py`
before opening the model mesh. Fast preflight34305440450 exports the difference
without another runtime rebuild: the hardware image changes only
`_mmrs_prefill_placement` and `matmul_reduce_scatter_prefill`. Removing those two
prefill functions from each AST leaves the entire module identical, including
the decode config and weight helpers. Unchanged AST SHA256:
`1d3374593f1caf77b453ef22f442779f9bd4da15b4b78d86d7a2178ed5216762`.

The decode-only gate accepts exactly this audited source pair:
simulator `bb43f0cde336c3f84725d47a64ed2b506b5287bdd0e910cd24b13feed0a0826a`,
hardware `5419361f26071b388fd58768f003b11c704b40d508524b968aab78362843aa66`.
All other source checks remain exact. The exception is recorded in the result;
it does not qualify prefill or relax numerical comparisons. Future source changes
still fail closed. No hardware MLP timing is available from either rejected run.

## Completed hardware result

[Run34305753974](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34305753974),
code `87d4cab`, passes correctness and independent artifact validation.

| Check | Result |
| --- | --- |
| Scope | Layer0, T8, one stream, two P150A; unchanged BF4 gate/up and BF8 down |
| Eager / changed-input trace checks | 6 / 12 exact comparisons |
| Timing | 9 ABBA blocks, 3 seeds, 50 replays per sample |
| Native MLP | 0.334811 ms |
| 16-row MLP with both DMA conversions | 0.350553 ms |
| Change | +15.74 microseconds / +4.70% latency; every block slower |
| Full-model promotion | Rejected; `eligible_for_full_model_gate=false` |

Both arms include replicated DRAM input staging and the same explicit four-link
native reduce-scatter. Timed outputs, immutable inputs and stable bindings are
checked. The original hardware packer is unchanged. Do not multiply this layer
timing by64 and call it measured full-model performance or PP/CTX/TG.

Artifact: `hardware-evidence.local/34305753974/artifacts/qwen-hardware-inventory-34305753974/tiny-mlp.json`.
SHA256: `e03eb5d368dc036bbccda4b97a22023de86b6030e36aef2a61eb42ea858d979f`.
The result rules out this complete small-tile composition as a speed improvement;
it does not by itself prove DRAM saturation or rule out different weight layouts.

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
