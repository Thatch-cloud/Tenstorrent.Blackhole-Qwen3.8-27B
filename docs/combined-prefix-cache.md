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
