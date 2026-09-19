# Overlapping convolution-window writes

Status: simulator and combined hardware correctness pass; no full-cycle speedup.
**Not promoted. Keep the original window builder.**

## Combined result

[Run 35263303708](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35263303708)
at `e54e974655f1151e94b0ed8baf6e52fd248e69e1` completes in **6m30s**.
Independent report validation confirms two native audits and complete ABBA timing.

| 4K context, one stream | PP tok/s | Committed TG tok/s |
| --- | ---: | ---: |
| Unchanged recipe | 3,358.90 | 124.71 |
| Overlapped window writes | 3,311.84 | 124.60 |

Both arms commit 242 timed tokens and accept 224 of 300 proposals. Aggregate TG
changes **-0.089%**; paired changes are **-0.842% and +0.676%**, failing the
two-pair improvement screen. Mean T16 verification/readback improves only
**66.47 to 66.22 ms**; this does not produce a complete-request speedup.

Route diagnostics show 96 admitted T16 builds and 288 native T2/T4/T8 fallback
builds per candidate request, with restored bindings. The observed logical width
is **8240**, consistent with why the earlier 8256-only scope missed the route.
All 865 script, 1,520 native and eight adapter-source fingerprints are unchanged;
closure succeeds and process exit is zero. Report SHA-256:
`d022f44d84b4bca73db967a6aa88e0de4c48b99b4bb30a730ff10c98300fcc90`.
Do not rerun this unchanged candidate or present the verifier-only reduction as TG.

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

The combined experiment installs only around T16 verifier requests, preserves
native tails, and audits actual invocation counts and restored bindings.
No isolated timing or simulator result is a throughput claim.
