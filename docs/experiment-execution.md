# Two-P150A execution ledger

Updated 2026-09-05. Target: 200 committed tokens/s for one coding stream, not
aggregate throughput. No adoption or serving restart is authorized by a test pass.

## Verified control

- Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
- TT-Metal: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` with recorded graft changes.
- Plugin: `bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c`; vLLM `0.25.1+empty`.
- PCIe x16/x4 host attachment; inter-chip collectives use QSFP-DD/P300 fabric.
- Operator-confirmed exclusive card allocation; no host process-scan claim.
- Runner repository access and tested-ref exceptions remain enabled at operator request.

## Execution queue

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Harness ready, not benchmarked | Pinned weights, warmed B=1 context matrix, B=2/8 separately; engine commit timing still unavailable |
| E1 cache | Passive capture in baseline | Investigate zero only if it recurs; full-model cache lifecycle gate remains required |
| E2 interleaving | Operator primitives passed; prototype host tests only | Whole/chunked model continuation, cancellation and slot reuse before mixed-traffic benchmark |
| E3 verifier | Historical harness not yet a correctness gate | Assert every accepted-prefix state and output, forced rejections; then measure T=1/2/4/8/16 |
| E4 fusion/pipeline | Planned, needs implementation | E3 verified state contract, one native change per arm, full-path timing |
| E5 drafting | Dependency-gated | Passing E3, then greedy request-local lookup and MTP; no non-greedy semantic substitution |
| E6 coding quality | Corpus not frozen | 200 independent executable fixtures, isolated code execution and paired outcomes |
| E7 prefix reuse | Dependency-gated | E1/E2 lifecycle gate; hybrid KV plus recurrent/conv state identity and isolation |
| E8 precomputation | Planned audit | Bound removable cost before table prototypes; no unapproved arithmetic changes |
| E9 spare-core work | Planned profile | Actual bottleneck attribution before reader/core-map/L1 changes |
| E10 disaggregation | Planned feasibility gates | TP1 fit and correct hybrid-state handoff; E2 mixed-traffic control first |

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
