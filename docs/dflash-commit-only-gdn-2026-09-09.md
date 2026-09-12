# Commit-only GDN verification

**Status: hardware ABBA passes; new best 70.34 committed tok/s.**
The matched control reaches 65.50: **7.38% improvement**. The target remains
200, not achieved. This is one complete coding task, not held-out quality.

## Hardware result

[Run 34242926044](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34242926044),
revision `20dbd99`, passes both full-request audits and four timed requests.

| CTX / streams / verify rows | Control | Commit-only candidate |
| --- | --- | --- |
| 170 / 1 / up to 8: committed TG | 65.501887 | **70.336300** |
| Individual timed TG | 63.661800 / 67.451512 | 71.665727 / 69.055297 |
| PP tok/s | 496.892908 | 530.317135 |
| Mean draft ms/block | 32.326778 | 32.374450 |
| Mean verifier/readback ms/block | 65.427248 | 58.078125 |
| Mean publication ms/block | 3.364984 | 3.142621 |
| Complete prefill + setup + decode | 8.148 / 8.588 s | 6.904 / 7.061 s |

All six requests emit the same 150 committed tokens through EOS, with 22 blocks
and 129/154 accepted proposals. Native tokens, GDN, valid KV and inactive slots
are exact. Each arm's audit checks 1500 feature row/chip/tap comparisons and
22 eager-versus-trace proposals. The candidate also passes all 22 native-GDN
hash comparisons around verification, before the commit decision.

The measured order is control/candidate/candidate/control. Audits are excluded;
setup is fresh and unamortized. Target model loading is outside the reported
complete-request time. PP/setup variation is not attributed to this decode
optimization. Report SHA256:
`dce5700d3a1735e3d36ff7babcf6186b9396b863e8db932acebfba42fc345582`.

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

The remaining 32.37 ms draft cost also matters. Next candidate: fuse the
drafter's grouped causal convolution, currently repeated four times per learned
layer with separate expansion, shift, casts and arithmetic. Preserve every BF16
rounding boundary and gate against the existing composed implementation before
hardware integration. No convolution speedup is claimed yet. The commit-only
path is the new experimental lead; serving defaults remain unchanged.
