# DRAM-sharded projection: partial-reload investigation

Status: simulator experiment only. No hardware speed result or serving change.

The native DRAM-sharded matmul factory uses Float32 intermediate partials when
FP32 destination accumulation is enabled, but does not set the CB5 FP32 unpack
mode that the native 1D matmul factory sets. The candidate changes that descriptor
field only; it does not change weights, activation precision or arithmetic fidelity.
Gate, up and down now match the native control exactly for two input patterns
on both simulated chips. This is eager projection evidence, not full-model or
hardware qualification.

| Simulator attempt | Result | Next action |
| --- | --- | --- |
| `20260910T092238Z-405` | Wrapper omitted the optional bias slot; factory raised `IndexError` before candidate execution | Supply `[None]` on both mesh and per-chip inputs |
| `20260910T092808Z-436` | Descriptor format getter returned an unbound native enum; raised `TypeError` before candidate execution | Compare the exposed raw format value against a constructed Float32 descriptor |
| `20260910T093131Z-684` | UP projection passes all four eager comparisons exactly; clean closure and unchanged source hashes | Screen gate/down, then replay and integrity checks |
| `20260910T093507Z-403` | Gate projection also passes all four eager comparisons exactly, including fused SiLU; clean closure and unchanged source hashes | Screen down, then replay and integrity checks |
| `20260910T093831Z-405` | Down matches the first pattern, then retained intermediates clash with native circular buffers | Release completed eager intermediates between patterns |
| `20260910T094223Z-404` | Down passes all four eager comparisons exactly; clean closure and unchanged source hashes | Changed-input trace qualification |
| `20260910T094621Z-402` | Down replay qualification launched | Require poisoned-output replacement, all physical rows, stable bindings and packed-weight integrity |

`scripts/ci/dram-projection-binding-check.py` exercises the installed native
bindings without opening devices. It checks that the exact 64-entry unpack
policy survives descriptor mutation and that the explicit no-bias slot survives
the binding. This passed locally, alongside five host regression tests.

The binding check is not a numerical simulator pass. Even a successful eager
screen still needs replay, input/weight integrity, complete-MLP and hardware
request measurements before promotion toward the 200 committed TG target.
