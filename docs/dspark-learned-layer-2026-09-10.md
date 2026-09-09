# Complete learned DSpark layer: simulator bring-up

**Layer-zero port is implemented; device execution is not qualified yet.**
This connects the previously qualified, observed feature-projection output to
every operation in a learned DSpark layer. It does not use a neuron slice or
replace device intermediates with CPU-golden values.

| Part | Complete scope |
| --- | --- |
| Parameters | All 11 pinned layer-zero tensors, BF16 |
| Attention | 32 global query / 8 KV heads, split 16 / 4 per chip |
| History | 32 actual feature rows plus all seven proposal keys; 64 padded keys |
| Q/K preparation | Learned projection, direct RMS weights, absolute YaRN tables |
| Attention policy | Previously qualified precise exponential and 64-key chunks |
| MLP | All 17,408 channels; 8,704 per chip, full gate/up/SiLU/product/down |
| Reductions | FP32 partial sum, explicit BF16 cast, BF16 residual addition |
| Cases | Two frozen input patterns; absolute starts 0 and 8,190 |

The CPU layer reproduces the existing upstream-checked attention residual and
layer output **bitwise at all six checkpoints**. The whole checkpoint is
reverified on close. The native port declares its FP32 normalization, rotary
and SiLU intermediates explicitly; final comparisons retain the existing
0.01 relative plus 0.01 absolute threshold.

## Execution boundary

Three phases have separate changing-input traces: attention/output projection,
MLP, and final residual. The simulator host stages the **observed** FP32 partials
between chips at the two reduction boundaries. Each sum and residual addition
then runs on device. The initial context is the hash-pinned observed output from
the successful complete FC/norm run, not the CPU reference substituted for it.

This is not a fabric test, a complete captured pipeline, all five layers, a
target verifier or a serving result. Those eligibility flags remain false.
The first eager-only diagnostic skips replay to find integration errors sooner;
it cannot qualify the complete matrix, even if its comparisons pass.

## Current run

- Eager-only simulator `20260909T173511Z-440` is active from code `7d892df`.
- All 1,154 host tests and 59 simulator-harness tests pass, plus wrapper syntax.
- The previously qualified precise-attention and packer-compatibility grafts are
  held under an explicit owned lock; both native binaries remain unchanged.
- This diagnostic exercises all three arithmetic phases and saves their observed
  tensors. It has no replay or hardware eligibility; the original native files
  must be restored after its process reaches a terminal state.

| Full-matrix requirement | Checks |
| --- | ---: |
| Frozen CPU layer checkpoints | 6 |
| Complete learned eager stages and final references | 192 |
| Exact changing-input replays and stable bindings | 208 |
| Borrowed inputs | 196 |
| All learned parameters before/after | 44 |
| Stale-input controls | 6 |
| Inactive output rows remain zero | 12 |
| **Total** | **664** |

## Next gates

1. Reconcile the full learned eager chain; retain failures and captured tensors.
2. Qualify all 664 checks, restore the original native runtime, then independently
   reconcile the source-bound report.
3. Connect all five learned layers and selector, then the real TP collective and
   target-verification path. Measure coding acceptance and committed PP / CTX / TG
   through CI before any serving change.

The measured single-stream hardware result remains **PP 3,324.52 / CTX 4,096 /
TG 74.27**. No layer-component pass changes that result or establishes 200 TG.
