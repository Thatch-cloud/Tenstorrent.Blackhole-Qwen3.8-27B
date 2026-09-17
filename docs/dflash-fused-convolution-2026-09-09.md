# Fused DFlash2 convolution

**Status: hardware ABBA passes; new best 78.06 committed tok/s.**
The matched control reaches 72.21: **8.10% improvement**. The target remains
200, not achieved. This is one complete coding task, not held-out quality.

## Hardware result

[Run 34246322267](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34246322267),
revision `1948a21`, passes both complete audits and all four timed requests.

| CTX / streams / verify rows | Control | Fused candidate |
| --- | --- | --- |
| 170 / 1 / up to 8: committed TG | 72.213017 | **78.061082** |
| Individual timed TG | 70.774090 / 73.711668 | 77.955701 / 78.166749 |
| PP tok/s | 517.894822 | 510.646465 |
| Mean draft ms/block | 31.054947 | 24.811769 |
| Mean verifier/readback ms/block | 58.114565 | 58.174193 |
| Mean publication ms/block | 3.340334 | 3.284827 |
| Complete prefill + setup + decode | 7.026 / 8.010 s | 6.559 / 6.286 s |

All six requests emit the same 150 committed tokens through EOS: 22 blocks and
129/154 accepted drafts. Native tokens, GDN, valid KV and inactive slots match.
Each arm's audit checks 1500 feature comparisons, 22 eager/trace proposals and
22 unchanged-before-decision GDN hashes. The fused audit additionally passes
all **880 convolution comparisons** across five layers, four convolutions,
22 proposals and both chips. No tolerance was relaxed.

The measured order is control/candidate/candidate/control. Audits are excluded
from TG, setup is unamortized and target model loading is outside the reported
complete-request time. The speedup is against this run's control, not the
previous run's 70.34 TG. PP/setup variation is not attributed to decode fusion.
Report SHA256:
`b9947cffc7e596f51215da3ec0e585bafbaddd9e76366168e251510b6415b197`.

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
