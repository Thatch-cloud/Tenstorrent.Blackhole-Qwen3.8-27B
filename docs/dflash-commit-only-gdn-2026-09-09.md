# Commit-only GDN verification

**Status: simulator and host gates pass; hardware comparison ready.**
The best complete single-stream result remains **66.76 committed tok/s**.
The target is 200; this change has no measured performance result yet.

## What changes

The captured T8 DFlash2 request spends about 65.27 ms per verifier block.
Its GDN adapter copies state and publishes a speculative final state before
the request decides how many tokens to accept. The request then publishes the
chosen prefix again.

| During multirow verification | Control | Candidate |
| --- | --- | --- |
| Save immutable entry state | Yes | Yes |
| Calculate all recurrent/convolution prefixes | Yes | Yes, same kernels/math |
| Copy entry into temporary working state | Yes | No |
| Publish speculative convolution state | Yes | No |
| Restore checkpoint and speculative final state | Yes | No |
| Write native GDN state before acceptance | Yes | No |
| Publish accepted prefix after decision | Existing global DMA | Same global DMA |

This removes redundant writes, not precision or rollback guarantees. Native
single-token decoding is unchanged. The candidate requires retained packed
histories with an explicit request-level commit owner; it is off by default.

## Validation

- TT-Sim: T8 normalized GDN, every accepted prefix 0–8, unchanged native state
  and checkpoint before decision, actual commit DMA, then a real T2 adapter
  continuation. Reject a deliberately stale-state continuation.
- Hardware: audit one complete control request and one candidate request.
  The candidate additionally hashes native GDN around every multirow verifier
  replay, before acceptance. These requests are excluded from throughput.
- Then measure four complete requests in **control / candidate / candidate /
  control** order. All must have identical proposals, acceptance and output.
- Require native-exact tokens, GDN, valid attention KV and inactive slots;
  exact target feature rows and same-context eager/trace draft outputs.

## Reproducible comparison

TT-Sim `20260908T145528Z-299-multitoken` exits 0 after clean mesh close:
9 adapter prefixes, 9 unchanged-before-commit checks, 9 real T2 continuations
and 1 detected stale-state control. Normalization and packed convolution DMA
are enabled. This is functional comparison against serial T1 of the same GDN
kernel, not native full-model certification, fabric validation or timing.
Report SHA256: `8ea51a9ae0f61a0f09ed6720b5a7a87964ee2aea8f6e5bbdfc0f4489186c0540`.
The simulator adapter hash matches the hardware candidate source. All 859 host
tests pass (799 CI + 60 speculative harness), as do Python/YAML/shell syntax
and whitespace checks.

Opt-in CI suite: `full-dflash-commit-request`. Both arms use the same captured
five-layer BF16 DFlash2 T8 proposer, CTX170 coding task, physical `p150_x2`
descriptor, four fabric links, target precision and B1 KV allocation.
The report includes runtime source hashes, per-arm TG, setup-inclusive request
time and the candidate/control ratio. No audit time or setup amortization is
hidden in the decode rate. One task is not held-out coding-quality certification.

The remaining 31.27 ms draft cost also matters: removing target-state copies
alone is not evidence that 200 tok/s is achievable. Promote only a measured,
correct improvement; otherwise retain the current control.
