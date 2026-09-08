# DFlash2 long-prefill initialization

**Status: implemented for the opt-in 4K qualification suite; no 4K hardware rate yet.**
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

Native TP2 eager prefill invokes each decoder layer once with the full sequence;
the capture therefore sees complete features even when layer internals chunk
their computation. A partial chunk cannot pass as the full prompt output.

## Local gates

Tail-copy run: `20260908T212513Z-419-dflash-prefill-window-probe`, terminal exit0,
clean device close. Report SHA256:
`db39a9d7b334f37e6a1b6445b49ee6e23474d83b1e35d199b0b20ca88a9f38ea`.

| Check | Result |
| --- | --- |
| Real TTNN tail slice/clone at CTX170/2048/2049/4093/4096, five taps, two chips | 50 bit-exact snapshots |
| Snapshots after overwriting the borrowed source | 50 exact ownership checks |
| Borrowed source before overwrite | 10 unchanged-input checks |
| Wrong first-window or padded-tail selection | Eight detected negative controls |
| Captured proposal inputs at initial CTX4093 and continuation, plus short-context regressions | 280 exact comparisons; T8 and T32, both chips, separate live trace preserved |
| Host tests | 825 CI tests + 60 harness tests pass |

Replay run: `20260908T213305Z-397-dflash-proposal-trace-probe`, terminal exit0,
clean device close. Report SHA256:
`0a67aa173b70ede73fbe093655c49a3501dd81f999f7f122cff59af29c5467c5`.
Both simulator reports' source hashes match the candidate files.

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
