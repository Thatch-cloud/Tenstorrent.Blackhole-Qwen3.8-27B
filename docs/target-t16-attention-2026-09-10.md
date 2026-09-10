# T16 target attention replay

Status: isolated simulator comparison passed; full-request integration and
hardware performance remain unqualified. Serving defaults are unchanged.

Hardware comparison dispatched as
[34453904831](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34453904831),
source `583c1fd`. Both arms use native DSpark drafting, captured proposals,
commit-only GDN and the unchanged norm reader. Only target attention differs.
Two audited requests precede four timed A/B/B/A requests; all samples count.
1,399 host tests pass before dispatch.

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
