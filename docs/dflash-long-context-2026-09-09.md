# DFlash2 long-prefill initialization

**Status: first 4K hardware run failed during feature capture; the chunk-aware fix passes local gates and is ready for hardware retry.**
Best measured performance remains PP510.65 / CTX170 / TG78.06, B1.

## What changes

| Area | Candidate behavior |
| --- | --- |
| Target | Prefills the complete prompt and retains the full paged KV context; no cache-capacity reduction. |
| Draft features | Captures only the last 2,048 valid rows from each of five TP2 feature taps, excluding padding. |
| Initialization | Explicit feature-window start prevents accidentally projecting the first 2K rows of a longer prompt. |
| Positions | Keeps the absolute target frontier and RoPE positions; only draft history length is capped at 2,048. |
| Publication | Existing committed-prefix rolling history is unchanged. Rejected rows do not enter history. |
| Runtime | Retains captured T8 proposals, commit-only GDN, fused convolution and four physical fabric links. |
| Serving | Unchanged; only `full-dflash-4k-request` enables the new context pilot. |

## First hardware failure and correction

[Run34282865344](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34282865344),
code `78cc9a9`, reached actual CTX4096 but failed before request measurements.
The guard correctly rejected a 2K chunk presented to the full-prompt capture.
Report SHA256: `a578e5e1cb8f8b960f95b293bac0e54307f162cac31aa845ff0514cba0096ae4`.

The earlier one-full-sequence-forward assumption was wrong for this experiment:
`max_batch_size=8` selects TP slot-prefill, which invokes masked 2K chunks even
when the caller supplies `enable_trace=False`. The initial simulator fixture
tested full-feature copies, not that native dispatch route.

`PrefillWindowCapture` now observes the native chunk boundary, preserves its
computation unchanged, and retains only each chunk's intersection with the
final 2K window. It rejects gaps, repeated chunks and trace bypasses. A window
straddling two chunks is stitched in absolute order. Hardware audits must check
both the individual snapshots and the assembled window on both chips.

## Local gates

Initial full-feature copy run: `20260908T212513Z-419-dflash-prefill-window-probe`, terminal exit0,
clean device close. Report SHA256:
`db39a9d7b334f37e6a1b6445b49ee6e23474d83b1e35d199b0b20ca88a9f38ea`.

| Check | Result |
| --- | --- |
| Real TTNN tail slice/clone at CTX170/2048/2049/4093/4096, five taps, two chips | 50 bit-exact snapshots |
| Snapshots after overwriting the borrowed source | 50 exact ownership checks |
| Borrowed source before overwrite | 10 unchanged-input checks |
| Wrong first-window or padded-tail selection | Eight detected negative controls |
| Captured proposal inputs at initial CTX4093 and continuation, plus short-context regressions | 280 exact comparisons; T8 and T32, both chips, separate live trace preserved |
| Native-boundary chunked capture, including CTX2049/4093 straddling windows | 70 exact snapshots, 50 exact assembled/owned outputs, 16 unchanged inputs and eight detected negative controls |
| Host tests for the chunk-aware candidate | 829 CI tests + 60 harness tests passed |

Replay run: `20260908T213305Z-397-dflash-proposal-trace-probe`, terminal exit0,
clean device close. Report SHA256:
`0a67aa173b70ede73fbe093655c49a3501dd81f999f7f122cff59af29c5467c5`.
The proposal replay files are unchanged. The old full-feature fixture alone is
not sufficient for the retry.

Chunk-aware run: `20260908T220717Z-304-dflash-prefill-window-probe`, terminal exit0,
clean device close. Report SHA256:
`c345962c45f41473e4d119525aefe27638a5813638bafee1fcac716c730f8599`.
It exercises `PrefillWindowCapture` with real TTNN slice/clone/concat at
CTX170/2048/2049/4093/4096 and checks ownership after source chunks are overwritten
and released. CTX4093 retains three rows from the first chunk and 2,045 from the
second. Capture source SHA256:
`ba9131ace5de88195a241ddfad55768a23ca6dfcb25115ef379950faa1ee58af`.

The simulator checks copies, masks, positions and buffer lifetimes, not learned
proposal accuracy, physical four-link behavior or tok/s. Learned proposals,
native token/state equality and PP/CTX/TG require the complete hardware request.

## Hardware gate

`full-dflash-4k-request` uses one recorded repository-prefix coding prompt,
between4,064 and4,096 actual template-inclusive tokens. It records source hashes,
excerpt hash, prompt-token hash and actual CTX; no answer-derived padding.
The unchanged coding task is at the end of the prompt. This is a latency pilot,
not a held-out repository-editing quality benchmark.

Run one complete correctness audit, then two uninstrumented complete requests.
Require native token/GDN/valid-KV/inactive equality, all committed feature rows,
every eager/trace proposal and fused convolution, plus exact tail snapshots for
both audited prefills. Keep a 513-token output budget, require EOS, and report
actual committed output counts. PP/CTX/TG and unamortized setup remain separate.

## Next optimization to isolate

Source inspection confirms that `execute_attention_branch` concatenates the
committed feature history with proposal inputs, then reprojects the entire K/V
sequence in every layer on every proposal. `DFlashDevice.execute_proposal`
supplies the same committed history to all five learned layers. Target KV is
cached, but these drafter history projections are not.

After recording the 4K control, test a separate per-layer cache of historical
projected K/V. Initialize once, append only committed feature rows and evict the
expired window. Preserve absolute rotated K positions, BF16 rounding, stable
trace addresses and atomic publication/abort ownership. Require exact eager
proposal parity and a matched complete-request comparison; no speed gain is
claimed from this source inspection alone.
