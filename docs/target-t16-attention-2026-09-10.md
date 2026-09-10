# T16 target attention replay

## Corrected hardware result

[34454698201](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34454698201)
passes with the candidate actually enabled. Independent re-analysis reproduces
the saved summary; source/runtime fingerprints match before/after. All six
requests preserve exact native outputs, state, inactive slots and cross-arm
proposals/acceptance. All timed samples are retained.

| Target attention | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native serial | 1 | 3330.87 | 4096 | 83.85 |
| Folded T16 | 1 | 3264.04 | 4096 | **89.89** |

Matched TG gain: **7.20%**, not yet repeat-confirmed. Each arm commits 234 timed
tokens, accepting 214/330 proposals. Verification/readback averages 76.17 versus
69.17 ms/block; whole cycles average 126.79 versus 118.29 ms. Setup-inclusive
request time is worse: 6309.29 versus 6588.83 ms. No serving promotion or
held-out quality claim follows; the 200 committed TG target remains unmet.

Artifact: `runner-evidence.local/34454698201/qwen-hardware-inventory-34454698201/dspark-target-attention-request-hardware.json`.
SHA256: `e0129e13e8877eb867ddca1ca3deadb9583bc7c1cff8153f68b9c0114ea94dc1`.

## Development and simulator evidence

The isolated simulator comparison and corrected hardware request comparison
pass. Repeatability and broader contexts remain open; serving defaults are unchanged.

Hardware comparison dispatched as
[34453904831](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34453904831),
source `583c1fd`. Both arms use native DSpark drafting, captured proposals,
commit-only GDN and the unchanged norm reader. Only target attention differs.
Two audited requests precede four timed A/B/B/A requests; all samples count.
1,399 host tests pass before dispatch.

That first hardware attempt failed final route validation: the candidate policy
omitted `target_attention_t16=True`, so all six completed requests used native
attention. Their exact outputs/state do not qualify the parallel candidate, and
their timings must not be reported as its performance. The report records clean
closure. No card reset was performed.

Fix `7a1e6bd` explicitly enables the candidate and validates its declared route
immediately after each request. Regression tests now inspect the actual policy,
not just fabricated result records; 20 targeted tests pass. Corrected retry:
[34454698201](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34454698201).

Both arms use a 256-token generation cap so every prepared T16 attention bucket
stays within the simulator-tested 4096..4352 family. This does not extend the
qualification to arbitrary contexts. The repository-derived coding prompt can
change when verifier source changes: compare within this run, not against an
older absolute TG value as though prompts were matched.

The existing folded four-query groups process T16 at capacity 4352. Native B1
attention supplies exact references at positions 4096, 4113, 4336 and rollback
to 4096. Query patterns change between replays; the page table is reversed.

| Check | Result |
| --- | --- |
| Output replay | 8/8 exact across both chips |
| Complete masks | 16/16 exact |
| Read-only KV | 4/4 unchanged |
| Stale controls | 2 detected |
| Mask-tail poison | 8 refreshes |
| Output poison | Before every replay; overwritten exactly |
| Source fingerprints | Before/after/current match |
| Cleanup | Exit 0, closed mesh, original packer restored |

Run: `20260910T080022Z-403`, about 302 simulator seconds, not hardware latency.
Report: `scripts/ci/target-t16-attention-simulator.json`.
SHA256: `3044483375ba689f0243b2bb479a67a7ce842b731fd221898ce50c31069afef6`.

Two preceding failures are retained. The diagnostic showed unpoisoned replay
was exact, but poisoning the entire mask corrupted its immutable prefix.
Long-context refresh intentionally writes only the last 256 positions. The
corrected test preserves the invariant prefix, poisons the refreshed tail and
checks the entire mask. No kernel arithmetic or tolerance changed.

This single-family, replicated-input simulator gate does not certify learned
full-model logits, feature taps, GDN state, request acceptance or throughput.
The next hardware request comparison must check all of those against the
unchanged native serial-attention arm before any promotion.
