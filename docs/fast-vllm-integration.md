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

The next adapter change must negotiate scheduler allocation before consuming
multiple target positions. Buffering extra tokens behind a one-token runner API
does not fix computed-token counts, KV allocation or preemption and is not an
acceptable shortcut. vLLM remains the serving engine; do not create a parallel
OpenAI proxy and describe it as vLLM integration.

## Image and platform requirements

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
