# Fast-path vLLM integration for Thatch

Decision: integrate the measured combined runtime first. **Do not publish the
historical baseline as the fast serving image.** Existing service stays unchanged.

## Current acceptance (2026-09-18)

The opt-in T16/DFlash worker path is now packaged in a built image. **Hardware
HTTP serving is not qualified; nothing has been published or deployed.**

| Gate | Evidence | Status |
|---|---|---|
| Pinned vLLM contracts | CPU 35327950138 | Real data types and pinned worker delegation pass; no complete engine loop |
| Source regression | CPU 35329738162 | Passed |
| Serving image | Build 35334618239 | Passed; includes EOS admission and runtime path fixes; superseding DFlash2 registration build pending |
| Startup dependencies | Image 35330608987 | Real imports, request components and topology source fingerprints pass |
| Real scheduler and allocator | CPU image 35332493127 | 4K prompt, 256 output budget, mixed acceptance, tail buckets, append-only page growth, completion, replacement and abort pass |
| Page uploads and captured cache writes | Simulator 35333075458 | 40 exact checks across two chips, stable page-buffer addresses and clean close |
| Page-updated T16 attention | Simulator 35333588113 | Exact native-serial equality on both chips at positions 4096 and 4160; stale-output controls and clean close pass |
| DFlash2 metadata | CPU image 35335959408 | Real speculative configuration passes without weights; separate guard test had an API-signature error, fixed in c3e60dc and awaiting image checks |
| Two-card HTTP lifecycle | Hardware 35335212616 | Failed before weight loading: missing DFlash2 registry entry; no throughput result |
| Registry / Thatch deployment | Pending | Existing serving unchanged |

Build source: `555e0cfd841fb90a94889876774c44067df5b15f`.
Local runner tag: `qwen-fast-serving:ci-555e0cfd841fb90a94889876774c44067df5b15f`.
Local Docker ID:
`sha256:42390dc83917d0c95cafbf43c1bc4c2367eef7feb3465c8b19711042004eacdd`.
This ID is not a registry manifest digest or proof of a clean pull elsewhere.
The build accessed neither cards nor weights.

The DFlash2 registry fix is metadata-only: it preserves `DFlash2DraftModel`
through vLLM configuration and rejects standard model construction. Actual draft
execution remains in the explicit combined TT worker, not an aliased upstream
DFlash implementation. Image checks must pass before the next hardware canary.

The image restores hash-verified native sources and the cached binary, stages the
pinned plugin patch, and connects startup, prefill feature capture, request
construction, scheduler reservations, committed-block output and cleanup. It does
not run offline reference generation before each serving request. Fast mode still
requires explicit configuration, admitted recipe paths and evidence; it is off by
default. Initial qualification is 4K input, up to 256 output, greedy T16, one
scheduler request, internal GDN B8, BF8 target KV and the four-link P150 pair.

The scheduler check runs real vLLM scheduling and KV allocation against synthetic
token outputs and a local metadata-only model configuration. It does not execute
Qwen, the worker, device traces or HTTP. `qwen-serving-scheduler.yml` reuses the
already-built image with no device or weight mounts, two CPUs and a 90-second
test cap, avoiding source-bundle downloads and image rebuilds for each check.

The page simulator uses the real serving page binder and native captured cache
writer, comparing the complete BF8 cache and all three metadata tables after
append-only page changes. Report SHA256:
`89678a253a11d30972670374faa7fb20b6af91bec01441dbc30f06647569305c`.
It does not qualify attention-reader execution, the full model or hardware speed.

The separate attention replay test now covers the actual T16 grouped reader after
the scheduler allocation grows from 65 to 66 pages. It compares both chips against
native serial attention, poisons the output before replay, rejects stale results,
and checks that KV remains unchanged. Report SHA256:
`7f3e006f1957b2e44890d1fa075bd6faa585d9cceb401aac52a5d3571ec30a7c`.
This remains weight-free component evidence, not full-model serving acceptance.

Next gates: complete engine/worker validation; repeated HTTP
requests on both cards with reference equality, EOS, cancellation and clean release;
then API PP/CTX/TG and streaming latency. Registry publication and Thatch rollout
follow acceptance, not merely a successful build. Worker-side rejection is not yet
a production HTTP admission layer.

The measured offline DMA result remains **121.12 TG at 4K**, not API throughput.
At 11 committed tokens per block, 200 TG requires a 55 ms total cycle; the roughly
60 ms target verifier alone exceeds that budget. Image packaging is not a speedup.

## Implementation history

The following records successive implementation stages. Statements that helpers
are unregistered or uninstalled describe their state at that stage, not the built
image above. Hardware and complete engine-loop acceptance remain outstanding.

### Original boundary

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

Bundle **35328870330 passed**. The canary Dockerfile consumes that source/binary
bundle and the inventoried cache manifest. `serving_native_install.py` recreates
the recorded SDPA factory, scratch patch, slice and lazy-link sources, checks their
cache provenance, then installs the matching binary. It does not compile kernels
or pretend that cards have been allocated. The image build runs serving unit tests
in the base image's actual Python/PyTorch environment. This lane builds locally on
the runner; it does not publish, deploy or claim HTTP/hardware acceptance.

Inventory **35328435640 passed**: the retained 52,480,392-byte runtime binary
matches `4b7299c1c9233b25aad310bc9a9d751a0631c6af0602cb1a151f934b4bfa07ea`,
and the staged direct-DMA kernel matches its qualified source. Docker reports
`zot.thatch.local:5000/tt-vllm@sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`
as a RepoDigest; a clean registry pull is still unverified. The image contains
PyTorch 2.11.0+cpu and vLLM 0.25.1+empty, distinct from the hosted CPU test environment.

`serving_bundle.py` checks the inventoried staged files before packaging them.
It overlays only serving helpers, the shared runtime wrapper and the singleton
page-table exposure; it does not overwrite the frozen verifier/geometry with
branch copies. The bundle job exports the hash-checked cached binary without
devices or model mounts. Native runtime source restoration and a built serving
image remain required; the bundle itself is not deployment acceptance.

The source-staging patch now wires explicit `qwen_fast_t16` opt-in through the
pinned platform guard and worker warmup entrypoint, and closes fast resources
before the worker deletes its model. Unselected configurations retain the original
warmup and speculative rejection. Staging verifies the exact plugin revision and
validates all three edits before writing. This is source integration, not a built
image or hardware/API acceptance; recipe directories and the complete admitted
runtime must still be packaged before launch.

The explicit startup entrypoint is `serving_startup.start(worker)`. It requires
absolute recipe paths, verifies the pinned four-link descriptor/sampling sources,
loads DFlash fixtures once, owns the 64-layer serial weight-stream pool, and
attaches the combined runtime. Startup currently remains unregistered in the
plugin; the old platform rejection therefore still prevents accidental enabling.
The new sampler retains the four-link override for the attachment lifetime.
CPU attachment testing initially failed because a mocked module registry removed
PyTorch after first import; the fixture now imports it before registry patching.

`serving_runtime.attach_combined_runtime` composes the loaded worker, shared
admitted recipe, prefill feature capture, request factory, scheduler page binder
and worker lifecycle. It requires the native-draft, serial weight-stream and KV
publication evidence explicitly; it does not register itself or publish an image.
`ServingCacheOwner` requires the runner, model and sixteen attention layers to
reference the same BF8 cache objects and records both chips' buffer addresses.
Ownership is rechecked after prefill and before target verification. This follows
the pinned model adapter's actual contract: its `kv_cache` forward parameter is
unused, while attention reads model-bound paged caches. Device ownership checks
still need to run against the real serving image.

The benchmark now enters `dflash_combined_request.combined_runtime`, a reusable
context manager containing the same source-admitted target, draft-attention,
weight-streaming and optional KV-publication scopes. Serving can hold these scopes
across its request lifecycle without invoking benchmark generation. Tests cover
both the original complete-request wrapper and direct runtime entry/cleanup on
failure. This refactor does not create new hardware performance evidence.

`FastServingLifecycle` provides the worker-level sequence: capture the native
prefill, return its seed once, construct/attach the fast bridge, publish drafts,
return committed decode blocks, and release traces before processing a finished
request's scheduler cleanup. Terminal prefill skips verifier allocation. Invalid
or partial prefill and setup failures poison the opt-in lifecycle rather than
silently falling back to slow decode. CPU tests compose this sequence with fake
device boundaries. Runtime-scope construction and startup registration remain
unconnected; no serving image has been published from these helpers.

`serving_request_factory.from_prefill` constructs the T16/DFlash request from
captured serving features and the already-emitted target seed. It does not invoke
the offline benchmark or perform reference generation. It retains the combined
flags (native proposal attention, cached draft history, fused convolution,
commit-only GDN, target T16 attention), releases prefill features before verifier
capture, and captures the proposal only after verifier persistent allocation.
CPU tests cover ownership, seed accounting and setup-failure cleanup. Integration
still requires the admitted combined-runtime scopes, actual plugin prefill capture,
page binding and request-lifecycle attachment; this is not a serving acceptance.

Capture warmup performs real KV writes: at the 4096-token frontier, a T16 verifier
needs 65 scheduler-owned 64-token pages, not just the 64 prompt pages. The factory
now rejects insufficient lookahead allocation before creating device resources.
It also checks that the captured table matches the scheduler's block IDs and that
unused entries reference only owned pages. Padding alone is not ownership and
must never redirect warmup writes into the prompt. Future decode page allocations
still require the existing append-only refresh before verification.

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
