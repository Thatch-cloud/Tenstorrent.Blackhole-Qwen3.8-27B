# Two-P150A execution ledger

Updated 2026-09-05. Target: 200 committed tokens/s for one coding stream, not
aggregate throughput. No adoption or serving restart is authorized by a test pass.

## Verified control

- Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
- TT-Metal: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` with recorded graft changes.
- Plugin: `bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c`; vLLM `0.25.1+empty`.
- Weights/tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, cache discovered
  and mounted directly read-only; full-model startup and K/M engagement verified.
- PCIe x16/x4 host attachment; inter-chip collectives use QSFP-DD/P300 fabric.
- Operator-confirmed exclusive card allocation; no host process-scan claim.
- Runner repository access and tested-ref exceptions remain enabled at operator request.

## Execution queue

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Benchmarked: run 33943034757 | 19.45 to 18.39 client-estimated tok/s at B=1; no-logprobs and engine timing still required |
| E1 cache | Occupancy observed, no active-zero recurrence | 0-12.4893% occupancy; full-model cache lifecycle gate remains required |
| E2 interleaving | Operator primitives passed; prototype host tests only | Whole/chunked model continuation, cancellation and slot reuse before mixed-traffic benchmark |
| E3 verifier | Historical harness not yet a correctness gate | Assert every accepted-prefix state and output, forced rejections; then measure T=1/2/4/8/16 |
| E4 fusion/pipeline | Planned, needs implementation | E3 verified state contract, one native change per arm, full-path timing |
| E5 drafting | Lookup proposal policy passes 11 host tests; integration dependency-gated | Passing E3 before enabling lookup/MTP; no non-greedy semantic substitution |
| E6 coding quality | Corpus not frozen | 200 independent executable fixtures, isolated code execution and paired outcomes |
| E7 prefix reuse | Dependency-gated | E1/E2 lifecycle gate; hybrid KV plus recurrent/conv state identity and isolation |
| E8 precomputation | Planned audit | Bound removable cost before table prototypes; no unapproved arithmetic changes |
| E9 spare-core work | Planned profile | Actual bottleneck attribution before reader/core-map/L1 changes |
| E10 disaggregation | Eight-slot TP1 pool fails analytical capacity bound | Smaller-pool TP1 and hybrid-state handoff remain unverified; E2 mixed-traffic control first |

The queue is not a claim that all tracks already have runnable implementations.
Run dependent arms only after their entry gates pass. Preserve failed and negative
results; do not convert skipped or blocked experiments into successful test counts.

## Baseline protocol

Manual workflow suite `baseline`, `cards_allocated=true`. Network-disabled disposable
container, loopback-only endpoint, read-only existing weight cache, separate labeled
experimental weight/kernel cache volume, unchanged K-image flags and host sampling.
No production volumes are written. Only the container created by the job is removed.

Warm every workload, then three alternating-order repetitions at concurrency 1 with
target prompt lengths 128/4096/32768/65536, capped below the 65536 context limit to
reserve 1024 output tokens. Actual templated lengths are recorded. Concurrency 2 and
8 at 4096 tokens are separate aggregate controls. Ignore-EOS is verified by output
counts, logprob token counts cross-checked against usage, metrics scraped every 250 ms.
Client timing is labeled an estimate; no engine-token timestamps means this cannot
certify 200 committed tokens/s. Raw logprob token strings are not exact token IDs.
Coding quality and greedy token-ID equivalence require their separate gates.

Initial full-model run 33942471680 reached readiness and logged K/M engagement,
but no throughput requests ran: Transformers returned a BatchEncoding rather than
a token list. The client now normalizes the actual input_ids and has regression
tests for both return shapes. Run 33943034757 retries the baseline with the frozen
snapshot and unchanged model flags. Do not count the failed warm-up as a benchmark.

The [payload feasibility estimate](decode-payload-bound.md) gives approximately
19.92 GB of dense projection value payload per ordinary decode step under current
mixed bfloat4/bfloat8 layouts. It is not measured traffic, but establishes the
approximately 3.98 TB/s payload requirement of a non-amortised 200-step/s path.
