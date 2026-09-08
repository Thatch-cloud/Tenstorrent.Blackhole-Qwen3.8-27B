# Captured DFlash2 proposal experiment

**Target remains 200 committed tok/s for one coding stream. Not achieved.**
The measured T8 eager drafter spends110.58ms per proposal block. This opt-in
change captures the complete five-layer proposal computation, including the
borrowed embedding, four-link collectives, full-vocabulary top16 head and selector
projection. The learned greedy selector remains on CPU.

## Hardware result: 66.76 committed TG

[34238134003](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34238134003),
revision `eadaa41`, passes all three complete coding requests. This is the new
best measured single-stream request rate, but still below200 TG.

| Metric | Result |
| --- | --- |
| Workload | CTX170, B1, K7/T8, `merge_intervals_v1` |
| Measured TG | 68.409353 / 65.197390; aggregate **66.764763** |
| Completion | 150 committed decode tokens through EOS,22 blocks/request |
| Acceptance | 129/154 drafts;6.818182 committed tokens/block |
| Mean draft / verifier / publication | 31.265333 / 65.266694 / 3.384118 ms/block |
| Target PP | 517.112935 tok/s |
| Complete prefill + setup + decode | 7336.383891 / 7431.058295 ms; target already loaded |
| Prepared history contexts | 256 / 512 / 1024; no knowledge of future accepted tokens |
| Target gates | Exact native tokens, active GDN, valid KV and inactive slots |
| Audit request | 1500 row/chip/tap comparisons;22 exact same-context eager/trace proposal comparisons |

The audit request is excluded from TG. Two uninstrumented complete requests
include draft input updates, replay, readback, verification and publication.
Fresh feature/proposal setup costs2541.62/2503.83ms and is reported separately.
This changes context padding and KV allocation as well as capture, so it is not
a matched capture-only or MTP comparison. It is one coding task, not held-out
quality or an adopted serving configuration.
Report SHA256: `eb7b8d3e3694c7d32da683f182ba10913c777e9d4e3e540df424184338a742cb`.

Captured T32 is the next full-request test; eager T32 measured41.03 TG. Target
verification is now the larger measured cost, not host-side proposal dispatch.

## Implementation boundaries

| Boundary | Policy |
| --- | --- |
| Suites | `full-dflash-trace-request` (T8), `full-dflash-wide-trace-request` (T32 extrapolation) |
| Captures | Fixed history capacities256/512/1024/2048; prepare only those covering the declared request budget |
| Inputs | Stable anchor, history, mask and RoPE buffers; update values before replay |
| Masking | Unused history rows stay invisible; proposal rows move to the fixed history boundary |
| Position | Preserve absolute query/key rotary positions and the2048-token sliding window |
| Allocation | Prepare inputs and retain captured intermediates before target-verifier capture |
| Publication | Existing double-buffer committed history; rejected target rows never enter it |
| CPU readback | Chunked full-vocabulary candidates and rank256 selector features only |
| Audit | Every proposal compares captured tensors exactly against the same fixed-context eager execution |
| Timing | First full request is audit-only; two complete unfenced requests measure committed TG |

Fixed-context padding can change floating-point reduction order relative to the
unpadded eager drafter. Therefore acceptance is measured again; it is not assumed
identical. The target still enforces exact native output tokens, GDN state, valid
KV and inactive slots, with every committed feature row audited on both chips.

The single-stream experiment uses1032 physical KV pages, not8200. Its65536-token
addressable capacity, eight GDN slots and cache format are unchanged. This
allocation is not an eight-stream serving configuration.

## Validation boundaries

All848 host tests pass (788 CI helpers +60 speculative harness tests).
The simulator ownership probe passes140 exact chip/tensor checks across T8/T32,
CTX170/178/256/257/300, both history buffers and another live trace. It closes
cleanly; report source hashes match the candidate. Report SHA256:
`87ee2902cd341ef0b2c5c1db6be48c676d53aa7ff882fc55e7ec6ee400b99f21`.
The first attempts stopped before device work (an overly restrictive simulator
guard) and at the host UInt32-versus-INT64 fixture comparison, respectively;
both fixture errors are corrected. They are not counted as passes.

- Host tests cover exact visible-token masks, absolute RoPE relocation, changed
  contexts, request-budget coverage and exclusion of mixed/unaudited capture results.
- The simulator proposal-buffer probe uses copy operations, not learned weights.
  It checks fixed addresses, changed anchors/history/masks/RoPE, both history
  buffers, context transitions and interleaving with another live trace.
- This is ownership evidence only. Full learned drafting, physical four-link
  collectives, complete coding requests and TG are established separately by
  the hardware result above, not by this copy fixture.
- Setup is fresh per request and reported separately; no cross-request reuse or
  setup amortization is claimed. Serving defaults remain unchanged.
