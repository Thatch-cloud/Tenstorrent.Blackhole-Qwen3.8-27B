# Captured drafter operation attribution

Run `34474912529`, immutable commit `224800e`, is an attribution-only request
on the two P150A cards: CTX 4096, one stream, fifteen draft queries, folded T16
target attention and the existing precise-native DSpark proposal backend.

The latest uninstrumented comparisons place drafting around 37 ms per block.
That is a stage measurement, not proof of which device operations dominate.
The profile is intended to select larger improvements than another small MLP
layout change.

## What is measured

- Every actual prepared proposal trace, including its capture warmup.
- Both chips, per-operation durations, core counts and operand metadata.
- Per-chip kernel envelopes, covered intervals and overlapping operation sums.
- Exact eager/replay proposals, target tokens/state and feature publication.

Reversible Python hooks observe the existing trace calls; qualified proposal
sources and kernel arithmetic are unchanged. Hooks restore even on failure.
The validator rejects missing/duplicate events, changed sources, incomplete
audits or a throughput claim. Forty-one relevant host tests pass.

## Boundaries

This profile excludes host-to-device proposal staging, correctness audits and
token readback from its marked trace intervals. Those costs remain included in
ordinary committed TG measurements. Instrumented durations are neither TG nor
an end-to-end critical path. There is no serving change or performance promotion.

The run must finish and its artifacts pass independent validation before any
operation is identified as the next bottleneck.
