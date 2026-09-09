# Complete learned DSpark layer: simulator bring-up

**Complete layer executes, but its numerical gate fails. It is not qualified.**
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

## First device result

- Eager-only simulator `20260909T173511Z-440`, code `7d892df`, completes all
  attention, MLP and residual phases. It fails 102 of 192 eager comparisons.
- All 1,154 host tests and 59 simulator-harness tests pass, plus wrapper syntax.
- All 84 input, 44 parameter and 12 padding checks pass, as do six CPU checks.
  Mesh and checkpoint close cleanly; outer exit is 1. There is no replay result.
- All three native grafted files are restored byte-identically and owned locks
  released. Both native binaries remain unchanged. Independent qualification
  rejects the failed report, as required.

Failure report SHA256:
`f81fbb9aaab4eaae9ddadaddb4bbd59f8be0ebab6769eb67f038f192232f965c`.
The observed tensors are retained locally, SHA256
`e65c254e3197d28a888ddc5be194fdccc321a9afc29dd9a08d1d2f11bff31c2d`.

## What the failure isolates

The offline attribution uses each operation's **actual device inputs** rather
than attributing every downstream error to that operation. For case 0 / chip 0:

| Comparison against its own input reference | Result |
| --- | --- |
| Q/K rotary | Bitwise exact |
| Q/K and post-attention normalization | Pass the unchanged numerical threshold |
| Attention | 15 out-of-threshold elements; not qualified |
| Attention output projection | Pass the unchanged numerical threshold |
| MLP gate/up projections | Pass the unchanged numerical threshold |
| MLP down projection | 202 out-of-threshold elements; not qualified |
| Both BF16 residual additions | Within numerical tolerance, but fail required exact bits |

The full layer output has 1.324% relative L2 error against the frozen CPU
backbone in that case; this is diagnostic, not a coding-quality score or an
alternative passing threshold. All six cases/chips remain in
`scripts/ci/dspark-layer-attribution.json`, SHA256
`ca3d46e27f300421516f4255aba6dbd3a7ff2a19eb9b80b2c5609d86429f5ec8`.

A short residual diagnostic reproduces all 12 native controls exactly. Merely
requesting an FP32 output before a BF16 cast still fails all 12 candidate exact
checks, with clean exit 1. The next probe must widen the operands too; no
full-weight reload is needed to isolate this arithmetic.

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
