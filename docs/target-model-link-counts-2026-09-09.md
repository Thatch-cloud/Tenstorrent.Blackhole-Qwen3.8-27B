# Target-model link counts: separate from the sampler

**[CI 34327029099](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34327029099)
passes correctness on `1c72ef3`, but four target links do not improve TG.**
This is separate from streamed-MLP kernel work. Serving defaults are unchanged.

## Result: one coding stream

| Target links | PP tok/s | CTX tokens | TG committed tok/s | Prefill + setup + decode |
| --- | ---: | ---: | ---: | ---: |
| Two, control | 3,324.76 | 4,096 | **62.39** | 6.60 s |
| Four, candidate | 3,430.28 | 4,096 | 61.41 | 7.27 s |

Four links are **1.57% slower in TG** in this ABBA; no promotion. This single
comparison does not establish that two links always win. Both arms emit the
same 121 committed decode tokens per request, plus the prefill seed, through EOS.
Final GDN, valid KV and inactive-state digests match across arms.

Rates retain all four timed requests and use total tokens / total measured time.
CTX includes the chat template. PP includes feature capture and first-token
selection; TG includes drafting, verification/readback and publication. Setup
is not amortized. The total excludes model loading; this is not a serving test.

The artifact report SHA256 is
`e68fbe62d5e4ee848ec8b8f4f5782ad814b1727ca1b4d12d03b2d5ac53b6b4b3`.
The independent validator and PP / CTX / TG report generator both pass.

## Why test this?

The pinned Python CCL helper maps the compatibility device name `P300` to two
links per axis. Our sampler has its own four-link override, on a separate CCL
instance. The native C++ discovery fix avoids unnecessary fallback discovery;
it does not change an explicit Python request for two links into four.

The physical setup is still **two P150A cards, two QSFP-DD cables, four links**.
The compatibility descriptor name is not evidence of which link count each
caller requests. This test records the original and effective target policies.

## Matched request test

| Setting | Control | Candidate |
| --- | --- | --- |
| Prompt context, including template | 4,096 tokens | Same |
| Coding streams | One | One |
| Target verification | T8, commit-only GDN | Same |
| Drafter | Captured DFlash2, fused convolution, cached K/V | Same |
| Target CCL helper request | Two links | Four links |
| Sampler link request | Four links | Four links |
| Drafter collective policy | Unchanged | Unchanged |
| MLP compute | Existing native kernels | Same; streamed MLP disabled |

CI suite: `full-dflash-target-links-request`. Two audited requests precede
measured control/candidate/candidate/control requests. PP / CTX / TG tables label
the target link count explicitly. All measured requests and stalls remain in
the result; prefill, verifier setup and complete decode stay separately visible.

## Correctness and ownership

- Validate all 193 target owners: the model plus each layer, attention/GDN and MLP.
- Change only their shared CCL instance, inside one request; restore it even on failure.
- Record calls by axis, not just an environment variable or configured link count.
- Keep proposals, accepted tokens, EOS and native-state checks identical across arms.
- Compare final active GDN, valid KV and inactive-state digest vectors **across arms**.
- Bind results to the exact experiment sources; require a completed successful CI run.

This reuses native collective kernels already exercised by the four-link probes
and real-weight MLP comparison. No new kernel is sent to hardware. The new
streamed-MLP kernels remain behind their independent simulator-first gate.
