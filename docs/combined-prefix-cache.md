# Combined T16/DSpark prefix caching

User-approved order: combined runtime first, serving afterward.
**Not enabled or hardware-qualified yet.** Do not flip the vLLM prefix-cache
flag: this work is a separate hybrid-state implementation.

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

## Remaining integration gates

1. Connect owned KV-prefix storage and captured feature chunks at the native
   completed-chunk boundary, with buffers allocated before trace capture.
2. Implement bounded single-stream lookup/invalidation and restore all components
   atomically before processing a changed suffix. Never reuse generated-state
   checkpoints as prompt-prefix state.
3. Compare cold and cached changed-suffix requests: exact emitted tokens, GDN,
   valid KV, inactive slots and DSpark features; exercise misses and failures.
4. Measure hit tokens, suffix tokens, restore cost, suffix PP, setup and complete
   TG independently. Keep the cold context ladder unchanged. Only then enable
   the combined path and start serving integration.
