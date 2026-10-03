# tp4/drafter-bf16: the DFlash2 drafter in bfloat16

`QWEN_FAST_DRAFTER_BF16=1` keeps every drafter projection in bfloat16. Default off; off, nothing changes (the dtype is `QWEN_FAST_DRAFT_BF8`'s, as before).

## Where the conversion happens

The serving image bakes `QWEN_FAST_DRAFT_BF8=1`. `draft_mlp_branch.draft_projection_dtype` is the one place that turns that into `bfloat8_b`, and every drafter
projection upload asks it: the five layers' q/k/v/o (`draft_attention_branch`), gate/up/down (`draft_mlp_branch.prepare_mlp_branch`) and the fc feature projection
(`dflash_device.PreparedDraftWeights`), 36 tensors. Norms, convolution kernels, bases and the selector are bfloat16 always. The quad block (`quad_draft`) and
`draft_wide_tp` reuse those tensors and convert nothing. The flag returns bfloat16 from that function and prints one `[DRAFTER_BF16] engaged` line per process.
`dflash_device.py` is not edited (pinned): its lend line already reports the dtype the lent tensors carry, so the smoke rule is the marker plus no
`projections dtype=bf8` lend line.

## Cost (`scripts/ci/drafter_bf16.py cost`)

Shapes: hidden 5120, q 4096x5120, k and v 1024x5120, o 5120x4096, MLP 17408, fc 5120x25600, five layers: 1.730 G elements, sharded evenly over the chips.

| per chip, TP4 | bf16 | bf8 (1.0625 B/elem) | extra |
|---|---|---|---|
| weight bytes | 865 MB | 460 MB | 405 MB |
| DRAM read per drafter pass at 420 GB/s achieved (512 peak) | 2.06 ms | 1.09 ms | 0.97 ms (0.79 at peak) |

- Four seats: one 64-row quad pass per round, +0.97 ms of a ~100 ms round: 1% (break-even: tau must rise 0.97%, about 0.06 tokens at tau 6).
- Eight seats: two quad blocks, two passes per round, +1.9 ms of a ~299 ms round: 0.65%.
- The drafter trace is 18.8 ms at four seats, so it is not weight-stream bound; the 0.97 ms assumes the extra bytes add straight to the stream. A bf16 matmul may also change math cost (unmeasured): the timed ABAB decides.
- DRAM: +405 MB per chip, one copy (the weights are shared by every borrower). The dspark design measured 7.9-12.9 GB free per chip at four seats x 262k; the eight-seat 262k pooled-KV budget must be rechecked at the integration.
- Trace region (256 MB) holds commands, not weights: unchanged. The weights are uploaded at attach, before any capture: nothing is allocated after a capture.
- L1: bfloat16 weight tiles are 2048 B against 1088 B, so matmul weight circular buffers grow. The pair served bfloat16 at twice the per-core widths (TP2); at TP4 each core's columns are half of that, so the buffers fit with margin. The audited smoke (S1) is the proof.

## Run

`scripts/ci/references/tp4-dbf16-jobs`: build `tp4-dbf16-1`, audited smoke on `c2-packed-tp4-best-gate-dbf16`, ABAB timed `c2-packed-tp4-best-strace` against
`c2-packed-tp4-best-dbf16`, then `drafter_bf16.py compare A.log B.log --live 4` (committed counts per round per user, tau, round, per-user tok/s, ratios).
The eight-seat 262k twin (`c2-packed-tp4-8x262k` plus the flag) is added at the integration with `tp4/seats8-262k`.
