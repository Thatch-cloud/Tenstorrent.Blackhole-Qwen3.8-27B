# DSpark draft-length audit

Source review, 2026-09-17. No new performance result or serving change.

## What is already present

Our pinned checkpoint SHA matches the published DSpark v2 object. The upstream
[configuration](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json)
enables a confidence head and specifies seven serving proposals, separately from
the sixteen-position training width. The
[model code](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/dspark.py)
defines a linear acceptance predictor; this alone does not establish the serving
policy, its calibration, or the correct inference-time feature alignment.

## Gaps in this port

| Source | Observed behavior | Consequence |
| --- | --- | --- |
| `dspark_intake.py` | Inventories confidence weight `[1, 5376]` and bias `[1]` | Presence in the checkpoint is not execution evidence |
| `dspark_prepared_proposal.execute` | Backbone, normalized hidden, logits, Markov feedback, token packing; no confidence output | Current captured path does not use the learned acceptance predictor |
| `PreparedDSparkProposal.propose` | Replays the fixed trace, reads all tokens, then slices `[:count]` | Smaller returned count does not save backbone or Markov work |
| `full_request.measure_request` | Width selected by remaining tokens, replay plan and configured maximum | No confidence-driven verifier-width policy here |
| `FusedT16Arm.forward` | Fuses its exact configured token shape, otherwise falls back | T8 cannot be called the same optimized recipe merely by reducing a count |

## Ordered experiment

Upstream inspection at SGLang commit
`2733afe54e4efe142cfdd01efb672eabf603c9a4` establishes these pieces:

- `dspark_planner.build_markov_embed_stack` uses the anchor followed by all but
  the last draft token, not the current token at each position.
- `DSparkConfidenceHead` concatenates hidden features with predecessor Markov
  embeddings cast to the hidden dtype, then casts to projection-weight dtype.
  Calibration applies sigmoid to FP32 logits divided by scalar or per-position
  temperatures (default one).
- The host budget planner forms cumulative confidence products. Its budget
  objective uses predicted accepted tokens and a measured step-cost table,
  rather than a universal confidence cutoff. It also handles lagged confidence
  and request-generation identities; these are not implemented in our oracle.

`scripts/ci/dspark_confidence_reference.py` now provides a CPU-only arithmetic
reference for those head and prefix-survival calculations. Six local tests
cover predecessor alignment, token zero, calibration, single-position tails
and invalid inputs, including mixed-dtype rounding and arithmetic overflow.
A separate executed comparison against the upstream `DSparkConfidenceHead`
class passes exactly for logits, probabilities and prefix survival in sixteen
synthetic cases: batch two, hidden width 5120, Markov rank 256, proposal counts
1/7/15/31, FP32/BF16 hidden inputs and scalar/per-position calibration. Only
that inspected class was extracted; the SGLang package was not installed.
Upstream model-source SHA256:
`d70ebbbbb81b93c0ad9e5baf1cec73d8fea5ab90851d14ca9128afa23fb1d75a`.
This is not a calibrated policy or device qualification. Actual hidden-tensor selection in our port,
checkpoint-weight parity, EOS behavior and hardware cost tables remain open.
This inspected main revision is not asserted to be the model card's v0.5.17.

The bounded confidence fixture now fetches only the pinned 6640-byte header and
10754 bytes of learned head weights/bias. Eight additional upstream comparisons
pass exactly with those learned weights and synthetic hidden/Markov features.
It rejects corrupted payloads; nine offline unit tests cover the reference and
fixture together. No complete checkpoint, target weights or hardware were loaded.
The fixture pins both tensor hashes independently of the header.

Reproduce the bounded CPU parity check with:

```powershell
python -B scripts/ci/dspark_confidence_parity.py
```

This now covers sixteen learned-head cases including scalar and per-position
calibration. It hashes the reviewed upstream source before extracting just the
confidence class, and emits source/tensor identities with the result. Ten local
unit tests pass, including refusal to parse unpinned upstream code. The emitted
report explicitly rejects hardware and performance qualification. No CI run or
multi-gigabyte download is needed.

Upstream `DFlashDraftModel.forward` applies its final norm before returning the
hidden states consumed by the sampler. Our prepared proposal exposes the final
`normalized` tensor, making it the source-level candidate for the same tap.
Actual learned hidden-feature parity and token-row alignment still need device
validation; matching the head arithmetic alone does not establish them.

1. Inspect the pinned serving implementation for confidence inputs, predecessor
   alignment, normalization and the actual length-selection rule. Do not infer
   these from the head constructor or substitute an arbitrary threshold.
2. Establish a CPU reference against that implementation, including token zero,
   EOS, short tails, nonfinite confidence and low-confidence first positions.
3. Qualify the confidence projection and changing-input replay on the simulator
   with bounded tensors, not another full-model weight-loading run.
4. Prepare explicit T8 and T16 captures. Record which fusion, recurrence,
   attention and publication paths execute for each width. Qualify differing
   routes rather than claiming shape-only equivalence.
5. Compare fixed T8, fixed T16 and confidence-selected widths in complete coding
   requests. Separate policy benefits from missing T8 kernel optimizations.

Report draft/verify/commit time, proposed/accepted/committed tokens, selected
width distribution, PP/CTX/TG and setup-inclusive latency. Use multiple coding
tasks: the high acceptance of the retained short fixture is not a general
acceptance estimate. Preserve exact target decisions and state publication;
confidence chooses speculative work, never permits unverified output.

Confidence computed after a full T16 draft can avoid some verification work,
but cannot retroactively save that draft. Choosing a cheaper next-cycle capture
is a separate policy requiring its own evaluation. No speedup is assumed.

Hardware and simulator CI remain idle while the user waits for host contention
to subside. The unchanged T16 recipe remains the control.
