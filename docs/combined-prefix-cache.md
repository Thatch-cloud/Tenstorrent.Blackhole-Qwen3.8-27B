# Combined T16/DSpark prefix caching

User-approved order: combined runtime first, serving afterward.
**4K offline combined screen passes; serving remains disabled.** Do not flip
the vLLM prefix-cache flag: this is a separate hybrid-state implementation.

## Verified 4K hardware result

[Run 35198371713](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35198371713)
at `14db1e57e94bf626fd3d4d5451eea76969c97b35` finishes in 4m33s on both P150A
cards. One changed-suffix feature audit and two timed combined requests pass.

| Measurement | Result |
| --- | ---: |
| Actual context / reused prefix / new suffix | 4,096 / 2,048 / 2,048 tokens |
| Mean cold-control prefill, including cache population | 1,239.17 ms |
| Mean cached prefill, including restore and native setup | 683.94 ms |
| Same-run prefill speedup | 1.81x |
| Effective cached PP / new-suffix request PP | 5,988.86 / 2,994.43 tok/s |
| Complete-cycle committed TG, one stream | **123.93 tok/s** |
| Timed committed tokens / accepted proposals | 242 / 224 of 300 |

This is not a cold-PP result or a matched decode-speedup claim. All three
requests match native tokens, target state and inactive slots. The audit has
100 feature checks and 11 proposal checks, including reuse after priming with
a different suffix. Exit status is zero and cleanup succeeds. Independently
recomputed summaries match the artifact; all 869 script, 1,520 native and 12
cache-source fingerprints are unchanged, and the cache sources match the
staging manifest. Report SHA-256:
`d95a7f07b2152c711491a3e683669053fde14a2b0f6547abd17a47724cec0d09`.

Mean timed block: draft **25.42 ms**, verification/readback **66.39 ms**,
selection/commit **5.00 ms**, whole cycle **97.60 ms**. At 12.1 committed tokens
per block, 200 TG needs a 60.5 ms whole cycle. Prefix caching does not remove
that verifier bottleneck. DSpark feature setup still averages 959.92 ms and
verifier setup 2,689.68 ms per request; neither is hidden in the cached-PP figure.
Longer-context caching, concurrent users, serving ownership and held-out coding
quality remain unqualified. The implementation notes below record the earlier
host-only stages and do not supersede this bounded hardware result.

## Required reusable state

| Component | Requirement |
| --- | --- |
| Attention KV | Exact prefix pages with explicit ownership and mapping; never overwrite another slot |
| GDN | All 48 recurrence tensors, 192 convolution states and 48 convolution carries |
| DSpark | Complete prefix feature history at the five target taps, with correct absolute positions; retain or reconstruct projected history |
| Native prefill | Last prefix hidden output when required by the chunk loop; preserve scratch bindings |
| Identity | Exact token prefix, model/drafter/runtime/precision identity and owning session; changed prefixes miss |

Read-only inventory **35189721030** exports the actual pinned model sources.
The native chunk path returns hidden activations, not just KV. Its B=1 GDN
scratch is distinct from batched decode state and is reset between requests.
The existing native GDN snapshot helper saves recurrence and convolution states
but omits `conv_carry`; the prefill path uses that carry between chunks.
Reusing that helper alone is therefore insufficient for prefix continuation.

`prefill_gdn_checkpoint.py` implements the GDN checkpoint component against
preallocated buffers. It includes carry, preserves both-chip bindings and
precision, synchronizes at checkpoint boundaries and rejects reuse after a
failed copy. It does not allocate device storage during restore. Four host tests
cover restoration, carry, aliasing, wrong boundaries, changed bindings and
partial failure. These tests are not hardware acceptance or a working cache.

`prefill_prefix_resume.py` adds the explicit eager suffix loop matching the
exported native chunk boundary. It restores first, does not reset GDN, skips
completed 2,048-token prefix chunks and retains absolute suffix positions.
It requires at least one uncached token: tail-only hits do not need a retained
prefix hidden tensor, while full suffix chunks supply their own final hidden.
This is text-only and not installed into the runtime yet. Its caller must supply
the complete KV/GDN/feature restore and validate identity and page ownership;
passing only the GDN checkpoint is not sufficient. Six host tests cover skipped
chunks, tail-only hits, final logits, restore failure and hidden-buffer cleanup.

`prefill_prefix_features.py` supplies a scoped feature-history borrower for that
suffix loop. It seeds the existing full-history capture at the cached absolute
boundary, retains the original five-tap prefix chunks without cloning them, and
owns only newly captured suffix chunks. The owner cannot close while borrowed;
suffix failures release only suffix storage. The combined caller must retain
this scope through all feature consumers. Four host tests exercise full-history
coverage, lifetime protection, failure cleanup and invalid owners. Runtime
integration and actual cold-versus-hit feature comparisons remain outstanding.

`prefill_prefix_lookup.py` bounds metadata to one entry, matches exact prefix
tokens and model/drafter/recipe fingerprints, and ties reuse to a session plus
allocation generation. Prefix-page remapping invalidates the entry; duplicate
pages or overlap with another slot are rejected. The caller supplies only valid
physical pages, not padded table entries. This is **not** a device page lease:
the runtime must still reserve these pages, invalidate before any cold overwrite,
and publish metadata only after all checkpoint components succeed. Six host
tests cover changed suffixes, misses, identity changes and ownership conflicts.

`prefill_prefix_boundary.py` hooks the completed native cold-prefill chunk to
capture preallocated GDN state **before** the suffix advances recurrence and
convolution carry. Missing, skipped or crossed boundaries cannot be published;
copy failure releases the otherwise orphaned hidden output. Four host tests
check callback ordering and failure restoration. Publication must require the
whole scope to complete, not merely observe the intermediate `captured` flag.

The host integration test now composes the real boundary, GDN checkpoint,
lookup, feature capture and suffix-loop helpers around a synthetic two-chip
state model. A 6,144-token request caches 4,096 tokens, changes the suffix,
deliberately corrupts current recurrent state, then restores and processes only
the final chunk. Output, all 288 state tensors, KV fixture and five-tap feature
history match a fresh cold request. This checks orchestration and ownership;
it does **not** test TT-NN numerics, DSpark projection or actual page retention.

The suffix helper now exposes `native_resume_scope`, an explicit one-request
override of `_prefill_chunked_eager_tp`. This leaves the native outer request
RoPE setup, prefill-scratch binding and decode-slot publication intact. It rejects
traced-prefill mode, vision input, changed tokens/page tables and repeated calls.
The host integration test uses this native-method boundary rather than calling
the suffix loop directly. No global patch, environment default or serving route
enables it; the combined cache controller still needs to invoke it explicitly.

`prefill_prefix_controller.py` now joins cold checkpoint capture, metadata
publication, feature ownership and native cache-hit routing. After a successful
cold request it retains only prefix feature chunks. Hits restore GDN and execute
the suffix; changed-prefix misses invalidate before cold overwrite, and lost
residency or request failures discard the entry. It requires a caller-provided
live page-residency validator and exclusive page ownership: no serving allocator
or hardware lease has been connected yet. The synthetic integration test covers
cold → changed-suffix hit → changed-prefix miss → lost lease, comparing hit state,
features and output with a fresh reference. Device integration remains unqualified.

`prefill_prefix_residency.py` checks the actual 16 native K/V pairs on both chips:
addresses, geometry, dtype, layout, memory placement, session identity and valid
page ranges. It rejects aliases, remapping and another slot claiming any reserved
page, even one unused by the current prompt. Failure revokes the validator.
The controller host test now uses this implementation rather than a stub. It
validates an existing exclusive reservation; it does not create one or detect
out-of-band writes to unchanged addresses. Integration must invalidate before
such writes. GDN checkpoint tests now explicitly preserve native mixed FP32
recurrence and BF16 convolution/carry types.

`prefill_checkpoint_allocation.py` allocates the 288 checkpoint buffers while
native B=1 scratch is bound, then restores decode bindings. It rejects existing
native decode, bucket or chunk traces, preserves each tensor's precision and
memory placement, and owns explicit teardown. The experiment must call it before
any additional verifier/proposal trace capture. Three host tests cover mixed
types, pre-existing traces and partial allocation failure; device allocation and
copy behavior still need qualification in the combined cache experiment.

The combined request harness now accepts an explicit `cached_prefill_factory`.
It requires a **cold native control followed by a cached candidate**, retains
the existing feature/state/token audits, records each prefill's cache outcome,
and unwinds cache ownership when the request fails. The controller's matching
factory invalidates before the control and retains the checkpoint for the
candidate. Host harness tests exercise both the audited path and rejection of
an accidentally warmed reference. This optional hook is not yet staged into a
hardware experiment or enabled in serving; the current 262K run remains unchanged.

Cached summaries require explicit `prefix_cached=True`, two complete prefill
records per request and restored capture/resume scopes. They report `pp: null`,
`effective_cached_pp` (whole prompt divided by cached-request prefill time) and
`suffix_request_pp` (new tokens divided by that same complete time, including
restore/setup). Cold PP cannot silently include reused tokens. Committed TG
continues to use the complete decode cycle. Seven focused accounting/integration
tests pass. Older unrelated preflight/workflow assertions in the broader request
test module still fail against historical source pins; they were not relaxed.

## Integration history and remaining gates

`prefill_prefix_session.py` now composes checkpoint allocation, native KV
residency checks and the combined request factory into one explicit offline
lifetime. It rejects nested sessions and overlapping reservations, releases
prefix features before checkpoint buffers, and revokes reuse on exit. Three
host tests cover cold/hit execution and cleanup on failure. The caller must
still hold an exclusive page lease; this helper does not create a serving lease
or enable hardware caching by itself.

Priority: combined T16/DSpark first, then serving. The initial host-only gate
had 65 passing focused tests; the accepted hardware scope is recorded above.

The opt-in `prefill_prefix_stage.py` now applies only the cache hook and cached
accounting to a freshly staged frozen T16 ladder. It does not copy the current
T32 request implementation. `prefill_prefix_experiment.py` allocates the session
before native warm-up, then runs the existing full feature audit and two timed
requests with a 2,048-token prefix at 4K. It retains the ladder's acceptance and
proposal checks, fingerprints its added sources, and rejects incomplete runs.
The feature-audited request now primes the cache with a different suffix before
reusing the prefix for the original prompt. Its tokens, state and features must
still match the original cold control. This extra cold priming is confined to
the untimed audit; both timed requests use ordinary cache hits. Changed-prefix
and failure paths remain host-tested, not device-qualified.
The experiment requires explicit `QWEN_PREFIX_CACHE_EXPERIMENT=1` and a
`experiment/combined-ladder-prefix-*` tag. That CI route selects only 4K and keeps
the cold ladder unchanged. The 4K hardware acceptance above covers this route.

First hardware attempt `35197029809` stopped before loading weights: the added
261,888 context option changed the historical geometry fingerprint even for 4K.
Ordinary ladder staging now restores the exact original geometry bytes; only
the explicit full-window route retains the narrowly admitted extension. All
42 component source checks pass locally against the retained real evidence
with the original geometry restored. No cache performance was measured in that
failed attempt.

Retry `35197650746` passed preflight and loaded the model, then stopped before
checkpoint allocation because the frozen report lacks the newer
`target_capacity` field. The adapter now derives capacity from all 16 allocated
K/V pairs and the actual page table, rejecting inconsistent shapes or conflicting
metadata. There was no cached-prefill or TG measurement in this attempt.

1. Extend the same combined changed-suffix screen to longer prefixes/contexts;
   retain the cold control, exact state/feature checks and complete-cycle TG.
2. Add device checks for changed-prefix misses, eviction and failure cleanup;
   those paths are currently host-tested only.
3. Separate restore cost from suffix execution and outer native setup. Current
   cached-prefill timing includes all three; do not relabel it kernel-only PP.
4. Connect serving's allocator, ownership and request lifecycle only after its
   separate correctness gate. Serving prefix caching remains disabled.
