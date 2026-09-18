# Fast-path vLLM integration for Thatch

Decision: integrate the measured combined runtime first. **Do not publish the
historical baseline as the fast serving image.** Existing service stays unchanged.

## Current boundary

The combined T16 request loop is an offline experiment. `full_request.py` generates
a complete native reference before measuring the candidate. That reference pass
must remain in qualification, not run before every production request.

The pinned image recipe uses TT plugin `bf77cd6`. Its `TTModelRunner` builds
`ModelRunnerOutput` by reshaping sampled IDs to one token per request. The plugin's
input-state update likewise assumes one sampled token. Passing a 16-token block
to the existing methods is not a valid integration.

The runtime image ID in `runtime-reproduction.md` is local Docker identity, not a
registry manifest digest. No portable fast-serving image is qualified yet.

## Delivery order

| Gate | Work | Acceptance |
|---|---|---|
| Request transaction | Reuse GreedySession, feature drafter and verifier without native reference generation | Tokens exposed only after target and draft publication; EOS, budget, cancellation and cleanup tests |
| Scheduler contract | Explicit speculative token/page reservation and variable committed output; update computed/output token counts exactly once | Rejection, all-accepted, tail, EOS, cancellation, preemption and slot reuse against pinned vLLM tests |
| Persistent runtime | Own one model/weight pool; retain admitted kernel sources and request-local traces/state | Real single-request HTTP streaming equals target reference; no stale state between requests |
| Two-card image | Package exact TT runtime, plugin adapter, kernel/admission evidence and model revisions | Build/import checks, registry push, manifest digest and clean-pull verification |
| Thatch canary | Route through the existing Compute runtime lifecycle, with exclusive two-card allocation and four-link topology | Readiness, streaming, disconnect, shutdown/release, rollback and PP/CTX/TG through the platform |
| Multi-user | Add isolated slots/pages and qualified batched execution | Per-user correctness/latency and aggregate throughput, separate from the single-stream target |

## Initial supported serving scope

Configuration compatibility is explicit: the offline verifier requires native
GDN B8 state, while initial serving admission is one request. The pinned plugin's
`get_tt_max_batch_size` currently returns scheduler `max_num_seqs`, including in
the model loader. `serving_fast_policy.py` defines an opt-in internal capacity of
eight without increasing scheduler concurrency; the upstream function is not yet
patched. Its initial qualification profile is greedy DFlash T16, 64-token pages,
4K input/256 output, synchronous execution and no unsupported sampling features.
Configuration unit tests do not qualify loading or serving this layout.

Greedy generation, one active request, explicit T16 fast mode, target and drafter
revisions pinned. Begin at the measured 4K context. Larger contexts, sampling,
logprobs, structured-output constraints, prefix caching and multi-user concurrency
must be explicitly rejected or routed through an independently identified baseline
until qualified; never silently claim the fast recipe for them.

`serving_fast_request.py` is the first **unselected integration component**. It
steps an already prepared GreedySession/verifier/feature runtime, exposes only
committed IDs, and aborts a verified block before publication when cancelled.
Its CPU tests exercise the existing GreedySession and DFlashRequestRuntime with
a fake device boundary. It does not yet hook TTModelRunner, perform HTTP serving,
or establish device/coding-quality/performance acceptance.

`serving_vllm_contract.py` adds a two-phase handoff: prepare the real draft IDs,
advertise them as vLLM `DraftTokenIds`, then require the scheduler's exact request,
computed frontier, token reservation and proposal IDs before verification. It
constructs variable-length `ModelRunnerOutput` only from committed IDs. Repeated,
partial, stale, preempted and mismatched schedules fail before device verification.
Tests currently use doubles for vLLM output types, not an installed engine.
Production runner hooks, request-state/token-count updates, page-binding admission
and real pinned-vLLM scheduler tests remain required before enabling this path.

The pinned TT platform also explicitly rejects any `speculative_config`. Enabling
the fast path therefore requires a narrow opt-in platform gate plus worker/runner
hooks; removing that assertion alone is not a solution. The dedicated
`qwen-fast-vllm-cpu.yml` workflow installs vLLM 0.25.1 with the upstream plugin's
empty-target recipe on a GitHub-hosted CPU runner. It tests real `SchedulerOutput`,
`DraftTokenIds` and `ModelRunnerOutput` objects with the request bridge; device
execution and the complete scheduler/engine loop are still outside that test.

`serving_vllm_state.py` implements the bounded committed-block token update for
the pinned TT runner's existing fields. It checks captured request identity,
output-list aliasing, pre-step computed frontier, vocabulary and storage bounds;
then appends the accepted block once and assigns both computed-token counters to
the committed frontier. Tests cover variable acceptance, duplicate application,
slot reuse, cancellation and failure-before-mutation. This is not yet installed
as a TTModelRunner hook. Capacity/page admission must happen before device commit,
not just while updating host state afterward.

Source review used TT plugin full revision
`bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c` (`model_runner.py`, `input_batch.py`)
and vLLM `v0.25.1` scheduler/output definitions. In the TT input batch,
`req_output_token_ids[row]` aliases `CachedRequestState.output_token_ids`; updating
both independently would duplicate every emitted block. The new helper requires
that identity and extends it only once.

`serving_runner_bridge.py` now composes scheduler admission, pre-verification host
capacity checks, page refresh, the existing device transaction, host token-state
updates and vLLM output construction. It is an **uninstalled** bridge for a prepared
request; the image's runner does not invoke it yet. CPU tests use fake device and
vLLM boundaries, including upload failure and preempted-request rejection.

`serving_page_binding.py` inventories primary, singleton and replay-reader page
metadata across captured verifier buckets. New physical blocks are appended in
place, preserving captured device addresses; existing page remaps are rejected.
Unused table entries repeat a page owned by the same request rather than assuming
physical block zero belongs to it. Partial uploads poison the verifier. The added
`ModelBatch.singleton_pages` attribute exposes an already-owned tensor and adds
no device operation. Actual trace replay with changing scheduler pages still
requires simulator and hardware qualification before this binder is enabled.

The next adapter change must negotiate scheduler allocation before consuming
multiple target positions. Buffering extra tokens behind a one-token runner API
does not fix computed-token counts, KV allocation or preemption and is not an
acceptable shortcut. vLLM remains the serving engine; do not create a parallel
OpenAI proxy and describe it as vLLM integration.

## Image and platform requirements

`FastWorkerHook` routes a prepared single-request bridge through the worker's
existing `execute_model` delegation and exposes actual draft IDs through
`take_draft_token_ids`. It returns committed blocks directly, never queues the
baseline single-token sampler, rejects a busy sampler queue or second owner, and
restores the original methods only after trace/request cleanup succeeds. This
request-scoped hook still needs the production prefill/request factory to attach
it; no startup registration or platform speculative gate has been enabled.

Installed-vLLM CPU contract run **35325017284 passed** against vLLM 0.25.1.
This validates real scheduler/output data types, not the complete engine loop or
device execution. `serving_plugin_patch.py` stages the native batch-capacity hook
against the pinned plugin: one scheduler request retains eight internal GDN slots.
Default batch sizing is unchanged. CI checks the actual upstream function and
rejects a changed or already-patched contract. The platform's speculative-decoding
rejection remains in place until the request factory and runner hooks are wired;
this patch alone deliberately cannot enable an incomplete serving path.

- Reuse the platform registry/authentication and runtime ownership; no copied
  secrets or hard-coded deployment credentials.
- Use a registry digest, explicit model snapshot revisions and source admission
  manifest, not a mutable image tag or local Docker ID.
- Allocate the complete physical P150 pair through Compute; retain exclusive
  ownership while traces and caches exist. Do not enable parallel CI card use.
- Preserve the P150 mesh descriptor, four configured fabric links and existing
  precision. No P300 descriptor substitution.
- Expose readiness only after model initialization and a bounded request smoke;
  measure startup separately from warmed serving throughput.
- Report serving PP/CTX/TG and inter-token latency. The 121.12-TG DMA result is
  offline evidence, not a promised API rate or a held-out coding-quality result.
