# Cumulative candidate review

Objective: 200 committed TG for one coding stream on two P150A cards, with
correctness preserved. Standalone gains below 2% are not automatically excluded
from cumulative testing. Serving defaults remain unchanged.

## Coverage and evidence

This is the first review pass, not a completed audit of every historical branch
or failed attempt. Outcomes below are read from retained experiment documents;
they are not new measurements. Before execution, recover each exact source,
simulator report and hardware artifact, and reconcile them with the current
recipe. A historical pass does not qualify a new composition.

The retained baseline already includes fused MLP, shared Q/K, norm prefetch,
incremental history/publication and tail-first drafting. Do not count these
again as independent additions.

## Cumulative queue

| Order | Candidate | Historical evidence | Integration requirement |
| --- | --- | --- | --- |
| 1 | Direct convolution windows + compact draft selection | Window verifier about 3.5 ms faster; compact draft about 0.4 ms faster | Separate request-owned hooks; test both active together, not sum TG percentages |
| 2 | Direct-scatter Q/K normalization | Verifier 66.60 to 65.75 ms; TG effectively unchanged | Alternative to current normalization path; retain shared Q/K and qualify composed sources |
| 3 | Wider MLP-down grid | Verifier savings 0.405 / 0.198 ms; TG -0.40% | Preserve fused gate/up; check integrated L1 and allocation effects |
| 4 | Register-resident MLP epilogue | TG +1.81%, paired +3.56% / +0.11% | Rebuild qualified fused compute; do not infer a verified kernel gain from TG |
| 5 | Down-only DRAM projection | Older verifier saving 0.740 ms; TG -1.27% | Rebase against fused MLP and control resident weight footprint; alternative to wider native down |
| 6 | Two-inflight MLP weights | TG +1.69%, mixed pairs; verifier worsened | Combine only after producer/consumer and CB compatibility checks |
| 7 | Q/K double-buffer ring | TG +1.71%, but verifier worsened | Shares recurrence loader with other GDN candidates; composite source qualification required |
| 8 | Bank-staggered MLP reads | TG +0.016%; mixed pairs | Interacts with weight pipeline; preserve packet order and validate composite reader |

Sources: [direct windows](gdn-direct-window.md),
[compact selection](compact-draft-score-selection.md),
[normalization](shared-qk-norm-scatter.md), [down grid](mlp-down-grid.md),
[epilogue](mlp-register-epilogue.md),
[down-only projection](dram-projection-reload-2026-09-10.md),
[weight pipeline](mlp-weight-pipeline.md), [Q/K ring](gdn-qk-double-buffer.md),
[read order](mlp-weight-read-order.md).

## Regressions and incompatible alternatives remain in the queue

These are not silently dropped. They need a distinct compatible arm or a repaired
composition; they must not be advertised as established incremental gains.

| Candidate | Recorded result | How to revisit |
| --- | --- | --- |
| [Overlapped window writes](gdn-window-write-overlap.md) | TG -0.089%; verifier about 0.25 ms faster | Alternative to direct windows, whose path removes this builder; enabling both would not exercise both |
| [K32 MLP blocks](mlp-k-block.md) | TG -1.26%; verifier +0.51 ms | Alternative block geometry; rederive CB sizes for any combined reader/epilogue |
| [Four-block MLP buffers](frozen-mlp-buffer-trial.md) | 32K TG -0.69% | Test interactions with pipeline only after combined L1 admission |
| [Full-K activation prefetch](frozen-mlp-input-prefetch.md) | 32K TG -2.75% | Alternative activation producer, not an independent flag on the streaming reader |
| [GDN dual state copy](gdn-dual-state-copy.md) | TG -4.44%; one pair broadly slower | Compose recurrence loader explicitly and measure phase effects again |
| [GDN outer-add fusion](gdn-outer-add-experiment.md) | Corrected two-tile variant TG -0.29%; trace +0.312 ms | Only corrected exact variant eligible; early numerically failing variants require repair |
| [Native GDN state slot](gdn-native-slot-experiment.md) | Older TG -0.312% | Recheck snapshot, ownership and rollback against current publication before composition |
| [Pooled streamed MLP](tensix-pooled-mlp-2026-09-09.md) | T8 layer 2.22x native latency | Replacement implementation, not additive to fused MLP; T16 simulator requalification first |
| [Sixteen producers](tensix-sixteen-producers-2026-09-09.md) | T8 layer +36.17% latency | Alternative to eight-producer pool, not a second simultaneous implementation |
| [Fixed-packet reads](weight-read-packets-2026-09-10.md) | T8 layer +26.92% latency | Extension of sixteen-producer design; charge full conversion/collective boundary |
| [Small activation tiles](tiny-tile-projections-2026-09-09.md) | T8 layer +4.70% latency | Requires layout-compatible producer/consumer composition; standalone conversion penalty is included |
| [T32 runtime](t32-score-reuse.md) | Older 4K 74.68 TG | Alternative verifier width; port T16 optimizations with new simulator coverage, not simultaneous T16/T32 hooks |

## Source-level conflicts already found

- `gdn_dual_state_scope` and `gdn_qk_double_buffer_scope` both replace
  `gdn_shared_qk_pipeline.load_kernels`; the latter also replaces `cb_plan`.
  Nesting wrappers is not proof that both source transformations survive.
- `shared_qk_norm_comparison_scope` patches the request measurement function and
  validation identities itself. Extract its component scope before combining;
  do not nest complete independent ABBA experiment drivers.
- `mlp_down_grid_scope` wraps `FusedT16Arm.forward`, whereas gate/up reader and
  epilogue experiments modify the underlying projection. Both engagement and
  precise numerical/weight identities must be demonstrated in the same request.
- Window overlap patches `gdn_conv_windows.build_windows`. Direct windows bypass
  that builder on T16. Its hit count must not falsely imply an additive T16 gain.

## Execution protocol

1. Maintain an immutable retained baseline and an explicit cumulative manifest.
2. For disjoint, already-qualified kernels, test scope composition, restoration
   and engagement locally; retain both simulator admissions. Requalify in the
   simulator whenever generated kernels, geometry, buffers or arithmetic change.
3. Use one weight load for native/cumulative correctness audits and ABBA complete
   requests. Preserve tokens, states, features, proposal acceptance and tails.
4. Record all samples, including stalls. Require repeatability before promotion;
   never silently remove a slow control. Attribute timings separately from TG.
5. Add one compatible group to the cumulative candidate, then perform leave-one-out
   tests on the resulting useful stack. Marginal components can remain when the
   combination wins; do not demand each independently exceeds 2%.
6. Test held-out coding prompts and the context ladder on a repeatable stack.
   Preserve cold PP versus cached PP and single-stream versus concurrency labels.

## Remaining historical reconciliation

- Inventory earlier commits and other experiment branches, not just current docs.
- Reconcile DFlash/MTP/drafter precision, banked history, score fusion, split-K,
  Markov bias caching and KV precision with the current DSpark path. Separate
  already-incorporated fixes from rejected alternatives and never-run hypotheses.
- Reconcile Ethernet dispatch/core reclaim, fabric loading, native argmax,
  prefill/disaggregation and batching: different objectives or runtime topology
  cannot be counted as single-stream decode gains without combined measurements.
- Read remaining numerical investigation histories through their final corrected
  outcome. A launch failure, timeout, wrong prompt or missing artifact is not a
  performance rejection of the kernel.

Implementation started: `cumulative_t16_scope.py` composes direct windows and
compact selection for one request. Three host tests cover simultaneous scope
activation, component entry failure, request failure and both-route engagement.
The cumulative request driver now runs both components in the same candidate
requests, retaining native controls, two audits and ABBA timing. It verifies
both route hit counts and all source fingerprints. A separate report validator
requires both component identities, complete request correctness and repeatability.
The staging workflow retains both simulator admissions and loads weights once.
There is no cumulative hardware performance result yet.
