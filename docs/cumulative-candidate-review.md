# Cumulative candidate review

Objective: 200 committed TG for one coding stream on two P150A cards, with
correctness preserved. Standalone gains below 2% are not automatically excluded
from cumulative testing. Serving defaults remain unchanged.

User decision: preserve current quantization and precision. Further weight/cache
quantization and lower-fidelity alternatives are deferred pending discussion,
not automatically enabled by this historical review.

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
| [GDN input overlap](gdn-input-overlap.md) | TG -0.92%; verifier 66.476 to 66.635 ms | Same recurrence-reader family as Q/K rings; compose transformed sources rather than nesting unverified hooks |
| [Drafter HiFi2](dspark-projection-precision.md) | TG -1.56%; no repeatable draft gain | Deferred under unchanged-precision policy; retained for historical completeness |
| [Bank-bound proposal traces](dspark-banked-runtime-admission.md) | 4K TG +0.036%; commit faster but draft slower | Alternative to copied-history proposal ownership; integrate compact selection into both graphs and audit both bank transitions |
| [Markov bias cache](markov-bias-cache-experiment.md) | 8K TG -6.93% | Interacts directly with compact selection and sequential feedback; retain reset epochs and count full-vocabulary cache traffic |

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

- A bounded `git log --all` pass now reconciles rejection records for HiFi2
  (`5aa1eb6`), input overlap (`d48057d`), bias cache (`478a7a9`), banked history
  (`ab021c3`) and two-tile outer-add (`34699539483` hardware record). The four
  newly identified families are included above, not omitted because they lost.
- Local refs inspected include `experiment/t16-matched`, `experiment/t32-score-reuse`,
  `experiment/concurrent-runtime`, `ci/qwen-hardware-correctness`, integration/main
  and retained PR refs. This is not yet a commit-by-commit or deleted-ref audit;
  those remaining histories must still be reconciled.
- Reconcile DFlash/MTP/drafter precision, banked history, score fusion, split-K,
  Markov bias caching and KV precision with the current DSpark path. Separate
  already-incorporated fixes from rejected alternatives and never-run hypotheses.
- Reconcile Ethernet dispatch/core reclaim, fabric loading, native argmax,
  prefill/disaggregation and batching: different objectives or runtime topology
  cannot be counted as single-stream decode gains without combined measurements.
- Read remaining numerical investigation histories through their final corrected
  outcome. A launch failure, timeout, wrong prompt or missing artifact is not a
  performance rejection of the kernel.

### Normalization composition finding

The frozen runtime wraps each measurement with its own norm-prefetch `build`
and admission overrides. A naive outer scatter scope would be overwritten by
that wrapper. Changing only the loader could also falsely label scatter as the
prefetch-qualified implementation. Integration must select the actual per-request
builder and its corresponding admission together, and update independent report
validation. Until that is implemented, do not claim scatter is present in the
cumulative recipe. The wider-down candidate uses a different forward hook and
is a simpler next independent addition while this identity boundary is repaired.

`cumulative_norm_runtime.py` now provides a separate opt-in normalization wrapper
for integration, leaving the frozen default wrapper unchanged. Selection is
request-owned: prefetch is the control, and scatter changes both the actual
builder and the checked simulator admission together. Scatter is explicitly
reported as **not** executing prefetch. Its validator cross-checks the shared-Q/K
admission, report identity, full-layer build count and restored bindings.
Host tests cover prefetch/scatter/prefetch ordering, wrong admissions, false
prefetch labels, nested selection and exceptions. This resolves the local
selection mechanism, not hardware integration: staging must replace the imported
`frozen_draft_tail_scope.norm_scope` binding, and cumulative report validators must
understand the distinct norm identity before scatter can be enabled. The current
two/three-component hardware recipes do not enable it.

Independent component report validators now accept an explicitly selected scatter
identity while retaining prefetch as their default. They reject mismatched
admissions, false prefetch labels, incomplete build coverage and changed
incremental-history identity. The retained compact-score (35268945497), wider-down
(35255552263) and direct-window attempt-two (35273229698) hardware reports still
pass independent validation. This is compatibility evidence, not a new speed
measurement. The cumulative driver and staging still need the scatter binding;
no current hardware recipe enables it. Precision remains unchanged, and further
quantization is deferred unless needed and discussed with the user.

The cumulative driver now supports an optional fourth component, `norm_scatter`,
with `QWEN_CUMULATIVE_NORM=1`. Candidate requests select scatter inside the same
direct-window/compact-score/wider-down scope; controls retain prefetch. The driver
refuses the flag unless the normalization wrapper is already installed. Staging
accepts `--norm-report` alongside pinned `--native-root`, checks both normalization
admissions and changes the draft-tail import to the explicit runtime selector.
The independent cumulative report validator checks each request's selected policy.
Host tests cover selection, restoration, missing execution and optional staging.
This integration is not hardware-qualified: the existing CI tag families still
leave scatter disabled, and full real-source staging needs the retained GDN native
kernel/API inventory (the model-only inventory is insufficient).

Real-source four-component staging subsequently passed in
`D:\qwen-cumulative-four-stage-20260918`. All seven native normalization dependencies
match the retained simulator report; missing compute headers were retrieved from
the pinned tt-metal revision and checked byte-for-byte by SHA-256. Both norm
admissions, compact selection and wider-down admissions passed against staged
files. `cumulative_native_sources.py` makes that small source restoration repeatable
in CI, without loading weights or opening cards. The opt-in
`experiment/cumulative-t16-stack-v*` tag family now stages all four components and
runs independent combined-report validation. No tag has been dispatched; this
remains preparation, not new hardware correctness or throughput evidence.

Next reintegration: `cumulative_register_scope.py` isolates the already-qualified
register epilogue without replacing `FusedT16Arm.forward`, so it does not displace
the wider-down hook. It selects the projection and its simulator admission
together, counts all 64 constructions and projection calls, and restores both
bindings on failure. Host tests check width rejection, nested-scope rejection,
restoration and missing layer execution. The four-component CI stack does not
enable this fifth candidate yet: cumulative runtime/report fusion identities must
first support the distinct epilogue admission. Its previous +1.81% result had
mixed paired gains and remains unpromoted; this preparation is not a new speedup.

Independent report composition now accepts an explicitly declared register
epilogue only on candidate requests, with matching per-request and route-audit
identities. All 256 packed-weight checks, 64 layer hit counts, restored bindings
and zero extra weight allocations remain mandatory. Baseline validation rejects
undeclared epilogue execution. The retained compact, wider-down and direct-window
reports still validate unchanged. Runtime selection and staging of the fifth
component remain to be connected; the prepared four-component CI path is unchanged.

The driver now has an opt-in `QWEN_CUMULATIVE_REGISTER=1` path. Its wrapper
records the epilogue identity after projection bindings restore, but before the
existing full-request comparison validates the request. Baseline controls remain
on their original fusion admission; candidate validation temporarily selects only
the matching fusion admission and retains the other route checks. Host tests cover
control/candidate/control ordering, failed requests, missing or duplicate
measurements, and composition with the wider-down request scope. Fifth-component
candidate payload/evidence staging and CI activation remain pending. No hardware
or precision change was made while the user requested the hardware pause.

Five-component staging now passes against real retained evidence in
`D:\qwen-cumulative-five-stage-20260918`. It rebuilds the exact admitted epilogue
payload, verifies its simulator report (35238822290), and checks both physical
rounding headers against the pinned runtime. All five admissions pass together;
this is source/staging qualification, not combined hardware acceptance.

| CI tag family | Candidate components |
| --- | --- |
| `experiment/cumulative-t16-v*` | Direct windows + compact drafting |
| `experiment/cumulative-t16-down-v*` | Above + wider MLP down |
| `experiment/cumulative-t16-stack-v*` | Above + norm scatter |
| `experiment/cumulative-t16-full-v*` | Above + register epilogue |

The new five-component lane is prepared but not dispatched. All lanes retain
unchanged controls, complete-request audits/ABBA timing, the I/O gate, exclusive
card ownership and independent report validation. No inference precision or
serving default changes. The host suite has 36 cumulative tests; neither that
count nor successful source staging establishes a throughput improvement.

The five-component host composition test now enters the actual normalization and
register wrappers together around the cumulative driver. Across the six-request
schedule it checks 288 prefetch and 288 scatter builder calls, matching fusion and
normalization admissions before route validation, unchanged controls and cleanup
after a failed candidate request. Device work and component kernels are mocked:
this catches wrapper/selection conflicts, not device numerical errors. All 38
cumulative host tests pass locally.

**Measurement checkpoint:** keep this candidate fixed until the user lifts the
hardware pause. Do not add more speculative components to this stack before its
complete-request correctness and timing are measured. On resumption, require
green CPU CI at the exact commit, grant only the immutable workflow/tag, and retain
the disk-pressure gate. Compare full-cycle committed TG and each phase against
the same-run controls; do not add historic percentages or promote a noisy pair.
The 200-TG objective and held-out coding-quality acceptance remain open.

CI coverage correction: inspection found that the existing green integration
workflow used an explicit test list and did not include `test_cumulative*.py`.
Earlier cumulative counts were local results, not CI coverage. The CPU workflow
now discovers cumulative, direct-window and compact-score tests explicitly;
require a green run containing these commands before hardware dispatch. The
measurement candidate's runtime sources are unchanged by this correction.

Hardware launch 35281141795 failed during source restoration, before the I/O gate
or weight loading: the helper attempted to download runner-exported GDN kernels
from upstream and received HTTP 404. Local staging had already populated those
files, hiding the fresh-CI path. The workflow now downloads retained inventory
35092212895 and explicitly supplies it to the restoration helper. Each file still
must match the simulator report hash; corrupt retained files fail rather than
falling back. A fresh empty-destination replay restored and verified all nine
native files, and 40 cumulative host tests pass. No kernel arithmetic changed.

Implementation started: `cumulative_t16_scope.py` composes direct windows and
compact selection for one request. Three host tests cover simultaneous scope
activation, component entry failure, request failure and both-route engagement.
The cumulative request driver now runs both components in the same candidate
requests, retaining native controls, two audits and ABBA timing. It verifies
both route hit counts and all source fingerprints. A separate report validator
requires both component identities, complete request correctness and repeatability.
The staging workflow retains both simulator admissions and loads weights once.
There is no cumulative hardware performance result yet.

Optional third component: `QWEN_CUMULATIVE_MLP_DOWN=1` adds the simulator-qualified
wider native down grid without changing weights, arithmetic or gate/up fusion.
Its admission pins run 35254182130 and the runtime `tp_common.py` source; current
local files and retained native inventory match. Reports require all 64 layer
hit counts to match fused MLP execution, as well as restoration of all three
scopes. Fifteen local tests cover two/three-component staging, missing execution,
failure unwinding and report identity. Three-component performance is unmeasured.

The workflow selects this stack only for `experiment/cumulative-t16-down-v*`;
the original tag family remains the two-component control experiment. No new
hardware job is dispatched while the user's other CI is creating I/O pressure.
First cumulative run 35275657339 stopped at the pre-load disk gate: it provides
staging evidence, not combined kernel correctness or a TG result.
