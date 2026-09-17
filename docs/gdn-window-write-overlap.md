# Overlapping convolution-window writes

Status: simulator correctness passes. Combined hardware performance is untested.

Logical-width qualification **35262735913**, `2310a2f`, passes in **48 seconds**
at width **8240**. It repeats the 64 exact-window and 72 immutable-input matrix
with unchanged reader/builder hashes. Admission now requires explicit width 8240
and report SHA-256
`92b90c94fefaf8171fc82acec8714e53ba97d709e797775b71574dd6bd377428`.
The older width-8256 evidence remains recorded below; it is not the retry's
admission report. This does not by itself prove the prior hardware failure's
cause; the retained route diagnostics must confirm execution on the next run.

Combined attempt **35261517322**, `c38a701`, failed in 5m14s at the route-coverage
gate after the candidate audit, before timed ABBA requests. It was not a timeout.
The control recorded 96 shared-Q/K builds. The adapter failed to retain the
candidate's hit/fallback counts before raising, so the artifact cannot distinguish
an unselected shape from an incorrect call-count assumption. No candidate TG is
qualified. Runtime closure succeeded.

The adapter now retains per-request route diagnostics even on failure and rejects
unqualified T16 shapes immediately rather than silently falling back through a
whole audit. Native projections permit logical widths 8240 and 8256; only 8256
was simulated here. Qualify the other geometry before admitting it; do not simply
remove the route gate or claim this failed run measured overlapping writes.

The original builder uses one output scratch tile and waits after each of four
window writes. The candidate uses four separate output scratch tiles, issues
the writes, then waits once before reusing scratch for the next page. Arithmetic,
history indexing, zero padding and output placement are unchanged. Extra L1
scratch is 6 KiB on each of 48 workers; no persistent model weights are added.

[Simulator run 35260444433](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35260444433)
at `65fd4de5d227299d8bb7bb1a062a23deb5a2b24d` passes in **48 seconds**.
Independent source-bound validation confirms 64 exact eager/replay window checks
against both the frozen implementation and CPU indexing, plus 72 immutable-input
checks. All eight source hashes match; process exit and container cleanup succeed.
Report SHA-256:
`fc63f7f84a1b2291ef6d3790e77bb83e2f8ee932e8e1cb0784f762e544794b83`.

Scope is T16, two simulated chips, synthetic operands. The nine host input
tensors include four unused B8 history fixtures inherited from the probe; both
implementations consume only the same four compact B1 history tensors. Their
immutability checks are not native-B8 runtime qualification.

Next: install only around T16 combined verifier requests, preserve native tails,
audit actual invocation counts and restored bindings, and compare complete 4K
ABBA requests. No isolated timing or simulator result is a throughput claim.
