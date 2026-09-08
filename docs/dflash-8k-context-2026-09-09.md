# DFlash2 at 8K context

**Repeat passed: PP3298.59 / CTX8192 / TG53.48, one stream, up to T8.**
The first run remains recorded at46.20 TG, including its publication stall.
Neither run is a serving measurement or a matched context-scaling comparison.

## Same-code repeat

[Hardware run34289671592](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34289671592)
uses the same `868dcbc` code and prompt, without a new optimization.

| Metric | Timed request 1 | Timed request 2 |
| --- | ---: | ---: |
| PP tok/s | 3,299.08 | 3,298.11 |
| Committed TG tok/s | 53.40 | 53.57 |
| Committed decode tokens through EOS | 121 | 121 |
| Complete prefill + fresh setup + decode | 8.63 s | 8.78 s |

Combined TG is53.484297; mean prefill2.483 s and complete request8.704 s.
The separate native-reference and feature/proposal audits pass. These two
repeat timings are more consistent, but do not identify the first run's stall.
No original sample is discarded or silently replaced.
Artifact SHA256: `272bae1fda0885196330ca6b49e9d8392015d91502eb8608b0181ddd97c88142`.

## First run

[Hardware run34286889429](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34286889429),
code `868dcbc`, retains the captured DFlash2 + commit-only GDN + fused convolution
lead. Artifact SHA256:
`9c7408fcbcb1fa73a1e0af1e708c662e4bf49bfbc4438f4c85c814994fa89099`.

| Metric | Timed request 1 | Timed request 2 |
| --- | ---: | ---: |
| PP tok/s | 3,052.67 | 3,252.31 |
| Committed TG tok/s | 41.94 | 51.43 |
| Committed decode tokens through EOS | 121 | 121 |
| Complete prefill + fresh setup + decode | 9.57 s | 10.90 s |
| Mean draft ms/block | 60.67 | 57.76 |
| Mean verify/readback ms/block | 66.38 | 63.83 |
| Mean publication ms/block | 30.71 | 5.84 |
| Accepted / proposed drafts | 104 / 126 | 104 / 126 |
| Blocks | 18 | 18 |

Combined rates use summed tokens divided by summed measured time, excluding the
separate audit. Mean prefill is2.601 s; mean full request10.24 s, excluding model
loading. The different PP, setup and TG boundaries explain why the faster-decode
request has the longer total time.

## Correctness and input

- All three requests match native tokens, GDN, valid target KV and inactive slots.
- The separate audit checks all121 committed feature rows across five taps and
  two chips, 18 eager/trace proposals and720 learned convolutions.
- Both audited prefills have20 exact tail snapshots and20 exact assembly checks.
- Sampling uses four links. The target retains its full context; only the
  drafter history rolls over2K.
- Prompt-token SHA256: `ae59d60a5f32e7e6d777fadcce441e419fb14a3455b0b81d8c66b96967bfa733`.
  The artifact records all six source-file hashes and the36,645-character excerpt.

## Timing caveat

Request1 contains a **424.46 ms publication stall**, versus mostly4-14 ms for
that phase. Some draft and verifier calls also have outliers. The stall remains
in the reported46.20 TG; do not remove it or promote the51.43 sample as the
combined result. The artifact locates the pause but does not identify whether
allocation, garbage collection, host scheduling or device work caused it.

Next measurements need bounded host/runtime diagnostics and repeatability,
alongside the cached-drafter and target-verifier optimizations. Neither this
pilot nor the earlier4K run certifies held-out coding quality.
