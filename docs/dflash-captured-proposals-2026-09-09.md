# Captured DFlash2 proposal experiment

**Target remains 200 committed tok/s for one coding stream. Not achieved.**
The measured T8 eager drafter spends110.58ms per proposal block. This opt-in
change captures the complete five-layer proposal computation, including the
borrowed embedding, four-link collectives, full-vocabulary top16 head and selector
projection. The learned greedy selector remains on CPU.

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
  collectives, complete coding requests and TG require the hardware suite.
- Setup is fresh per request and reported separately; no cross-request reuse or
  setup amortization is claimed. Serving defaults remain unchanged.
