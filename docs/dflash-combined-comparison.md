# DFlash2 versus the promoted combined runtime

Status: matched combined comparison passed in run 35287866254; DFlash2 is not promoted.
Control is the user-promoted T16/DSpark stack, not the older DFlash2 recipe.

| Drafter | CTX | Streams | PP tok/s | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| DSpark | 4096 | 1 | 3333.91 | 131.05 |
| DFlash2 | 4096 | 1 | 3360.78 | 97.58 |

Source: run 35287866254 artifact `cumulative-validation.json`, two timed
requests per drafter after correctness audits. This is a coding-fixture pilot,
not held-out quality certification or serving acceptance. DFlash2 is 25.54%
slower here; target optimizations are shared but drafter optimizations are not.

## DFlash2 optimization sequence

Hardware is paused at the user's request. Do not dispatch hardware jobs or
reset cards until explicitly released. Continue CPU and bounded simulator work.

1. Qualify a separate T16 native proposal-attention candidate. Preserve the
   existing T8 gates and pinned shared sources. Check all 16 live queries,
   finite padding, masked-key isolation, changed-input replay and stable bindings.
2. Integrate only after simulator evidence, then compare against composed
   attention in the same combined runtime when hardware becomes available.
   Native proposal arithmetic is not assumed identical: exact target output
   and state, acceptance, and complete-request TG remain mandatory gates.
3. Reduce DFlash2 history-publication and selection overhead, measuring each
   change in combination rather than substituting kernel timing for TG.

`dflash_t16_native_attention.py` is an opt-in experimental adapter only; no
runtime calls it yet. CPU tests cover mask rejection, second-half live-row
diagnostics, dispatch configuration and reference masked-key isolation. They
do not certify device numerics, trace replay, performance or coding quality.
The next simulator probe must use T16 operands, not relabel T8 learned fixtures.

Simulator run 35289299758 failed during checkout, before any probe execution:
the shared workspace root was not a Git repository during checkout cleanup.
The simulator workflow now uses a run-specific child checkout, as the other
staged experiments do, rather than cleaning the shared root. This failure
provides no evidence about native attention.

### T16 simulator result

Run 35289482417 passed in 2m51s on CPU TTsim, using no model weights or cards.
Both 31-row and 2048-row histories passed finite-output, masked-key isolation,
borrowed-input preservation, stable-binding and changed-input replay checks on
both simulated chips. The downloaded reports also passed independent local
validation against current source hashes.

| Draft history | Maximum absolute difference from FP32 reference | All values within legacy tolerance? |
| --- | ---: | --- |
| 31 | 0.030413 | No |
| 2048 | 0.004695 | Yes |

This qualifies mask/replay behavior only. Native proposal arithmetic is **not
numerically equivalent** to the reference at short history; tolerances were not
relaxed. Do not replace target attention or declare coding-quality acceptance.
Next is an explicitly approximate proposal-only combined candidate, requiring
fresh exact target-output/state audits and measured acceptance/TG when hardware
is released. The current combined runtime remains unchanged.

The combined request API now has an opt-in `native_attention_evidence` argument.
It requires both pinned simulator reports and matching source/native hashes,
then enables the T16-specific mask validator and native proposal operation.
The admission is rechecked on exit and recorded with each request; summary
validation rejects missing admission. All existing target optimization scopes
remain active. The default comparison driver does not select this candidate;
candidate CI staging and combined acceptance are still pending. No hardware
dispatch is authorized while the pause remains in force.

The separate `dflash_native_comparison_experiment.py` driver prepares two fresh
target audits followed by composed/native/native/composed complete requests in
one loaded session. It stops before timings if either audit fails. Its report
validator checks target components, exact committed output, source/cache policy,
per-arm proposal repeatability, acceptance accounting and stage costs. Different
draft proposals between arms are allowed; different committed output is not.
This driver is CPU-tested. The existing combined workflow selects it only for
an explicit `experiment/cumulative-t16-full-v*-dflash-native` tag. That route
downloads the pinned simulator artifact, validates it before and after staging,
and selects the independent native-comparison report validator. No such tag
has been published; hardware remains paused until the user releases it.

## Comparison contract

- Start at CTX4096, one stream, the same coding fixture and output budget.
- Keep target weights, precision, four-link policy and T16 verification matched.
- Keep direct windows, wider MLP down, scatter normalization and register
  epilogue active in both arms, with per-request execution evidence.
- DSpark retains compact Markov selection. DFlash2 uses its own selector;
  absence of DSpark selection is intentional, not a missing target optimization.
- Preserve each drafter's checkpoint and native history policy. DFlash2's 2K
  draft history is not truncation of the target's 4K KV context. Report this
  difference rather than calling the comparison history-matched.
- Two fresh correctness audits followed by ABBA complete requests. Compare
  target outputs and state across arms, not draft proposals or acceptance paths.
- Report PP/CTX/committed TG, setup, draft, verification, publication and
  accepted tokens per block. Keep timing stalls, source hashes and raw reports.

## Implementation sequence

1. Separate drafter-specific selection from shared target scopes. Implemented
   with host tests for identity, route engagement and exception cleanup.
2. Prepare T16 masks, absolute rotary inputs and global vocabulary candidates.
   Host utilities now support 16 rows; old eight/32-row behavior is retained.
3. Device/cache execution now supports explicit T16 composed attention, with
   cached history and captured proposals. Capture is deferred until verifier
   persistent allocations exist. Native proposal attention and live-query QK
   remain T8-only; their gates are not widened. The DFlash2 request owns the
   same four target scopes directly, without claiming DSpark history/selector
   hooks executed. Full target-output, state, feature, cached-versus-uncached
   proposal and changed-input replay checks remain mandatory.
4. Qualify changed-input proposal replay and full-request target state on the
   combined hardware path before timed ABBA. Use tiny simulator cases only if
   kernel changes require them; do not load model weights in the simulator.
5. Compare DFlash2 T8 separately after matched T16; retain width/history labels.

The comparison driver runs both drafters in the same loaded target session.
The `experiment/cumulative-t16-full-v*-dflash` tag family selects the adapter
in `qwen-cumulative-t16.yml`; other cumulative tags retain the existing driver.
It uses cached pinned DFlash2 fixtures only (revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`), fails if missing, and verifies tensors
with the existing fixture loaders. It does not start a multi-hour download.
The independent report validator checks both target component identities,
within-drafter proposal repeatability and exact shared target outputs, while
allowing different proposals between drafters.

Host tests are preparation evidence, not device numerical or performance
acceptance. Serving defaults and the immutable promoted DSpark recipe remain
unchanged. Source-qualified artifacts must be regenerated/reconciled where
the new host-source hashes differ; old admissions are not silently reused.

## First hardware launch: preflight failure

Run 35285659419 attempt 3 passed the disk gate but stopped before device
execution: the shared simulator-source gate detected modified `draft_attention.py`.
The integration had widened a host mask argument check in a file shared with
DSpark. This was a staging regression, not a kernel crash or numerical failure.
The follow-on Apport `FileNotFoundError` was only error-reporting noise.

The fix restores that file exactly to its simulator-pinned bytes and puts the
T16 host mask specialization in `dflash_attention_mask.py`, used only by DFlash2.
The comparison stage now runs the real shared-source preflight before and after
its overlay. Independent CPU visibility-formula tests cover T16 masks; the
original source-pin regression and existing widths remain checked. No source
hash, numerical threshold or simulator admission was bypassed.

Run 35287122216 subsequently reached both audit generation paths, but failed
at the DFlash2 summary before timed requests. The new driver omitted
`ended_with_eos`, `sampler_num_links` and `fabric_sources`, previously populated
by `full-prefix.py`. The retained DSpark audit also lacked these fields. This
is an integration-reporting failure, not an accepted benchmark or kernel-speed
result. DFlash2's request was not retained because validation preceded append.

The driver now derives completion from emitted/EOS IDs, reads the active sampler
link count, and checks fabric-source hashes against the runtime audit before
attaching them. It records completed requests before summary validation so a
future rejection retains its evidence. Source identities are rechecked after
the comparison; validation thresholds and exact-state gates are unchanged.
