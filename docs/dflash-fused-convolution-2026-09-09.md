# Fused DFlash2 convolution

**Status: simulator and host gates pass; hardware comparison ready.**
The current measured lead remains 70.34 committed tok/s. This kernel has no
hardware speed result yet; 200 remains unachieved.

## Change

Each learned layer calls grouped causal convolution four times. The composed
implementation makes 38 tensor-operation calls per convolution: causal shift,
group expansion, casts, multiplication and addition. The candidate replaces
that sequence with one dispatch on 80 workers per card, each processing two
32-column tiles.

The kernel retains all eight BF16 rounding boundaries. It does not combine
coefficients before multiplication, change weights, reduce precision or alter
the learned selector. Inputs remain read-only; only the output is allocated.

## Simulator evidence

`20260908T153551Z-310-draft-convolution-fused-probe` exits 0 after clean close.
Source hashes match the candidate.

| Gate | Result |
| --- | --- |
| Widths / inputs | 1, 8, 32 rows; dyadic and random BF16, different chip data |
| CPU / composed / fused / replay | 36 exact BF16-bit comparisons |
| Actual request-wrapper comparisons | 12 exact chip comparisons |
| Borrowed operands unchanged | 60 checks |
| Deliberately stale input | Detected at all three widths |
| Removed-rounding negative control | Detected on both chips at all widths |
| Host tests | 807 CI + 60 speculative harness pass |

Report SHA256:
`bfb4a2012dd6d63ae3ae318aabbff48a621194f170b178a420e79e83b230e44d`.

The initial kernel incorrectly held seven FP32 destination tiles under half
synchronization. The first 80 output tiles matched, but the second 80 did not.
Capture `20260908T152753Z-302` preserves the failing operands/output; merely
resetting copy configuration did not fix it. Streaming terms through four
destination tiles fixes the register-capacity error. Failed runs are not passes.

## Hardware experiment

Suite: `full-dflash-convolution-request`. Both arms retain the qualified
commit-only GDN verifier, captured T8 DFlash2, target formats, CTX170 coding
request, physical P150A pair and four fabric links.

First audit both complete requests. The candidate compares every convolution
in all five learned layers, on both chips, against the composed implementation
at every proposal. Both arms retain exact native token/GDN/valid-KV/inactive
checks, feature checks, proposal replay checks and pre-decision GDN hashes.

Then measure complete requests in control/candidate/candidate/control order.
Require identical proposals, acceptance and outputs. Audit time is excluded;
fresh prefill/setup costs remain visible. No serving-default changes or held-out
coding-quality certification are implied.
