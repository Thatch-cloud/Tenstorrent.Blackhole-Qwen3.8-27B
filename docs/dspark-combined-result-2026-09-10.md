# Combined attention and norm result

Run `34456312393`, source `a0884c0`, passes correctness but shows **no speed
benefit** from adding the scatter norm reader to folded T16 target attention.
Do not promote the combination or add the earlier isolated gains together.

| Path | Streams | Draft / verify rows | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Folded target attention | 1 | 15 / 16 | 3315.87 | 4096 | 90.25 |
| Folded attention + scatter norm | 1 | 15 / 16 | 3297.14 | 4096 | 90.24 |

Two audits precede four timed A/B/B/A requests. Both arms commit 234 timed
tokens and accept 214/330 proposals. Relative TG change is -0.0105%.
Setup-inclusive mean latency is 6080.03 ms control versus 6451.35 ms combined.
This single run does not establish a significant regression, but supplies no
evidence of a useful gain. No repeat of the unchanged combination is scheduled.

Independent recomputation reproduces the saved summary. All six requests retain
exact target tokens/state and inactive slots, matched proposals, complete feature
audits and declared folded attention engagement. Scatter-kernel engagement and
restoration pass. Source/native fingerprints remain unchanged and closure passes.
The pre-dispatch host suite passes 1408 tests; both retained kernel simulator
gates remain mandatory. No serving defaults changed.

Artifact: `runner-evidence.local/34456312393/qwen-hardware-inventory-34456312393/dspark-combined-request-hardware.json`.
SHA256: `85764913f375b799c79ba77927d60b85c5cc4248a7ef305683266798fc1b8ad9`.

The repeat-confirmed folded-attention result remains 89.86 committed TG at 4K.
Held-out coding quality, long-context scaling and the 200 TG objective remain
unqualified. Further experiments must reduce the full draft/verify/publication
cycle, not substitute isolated kernel timings for committed throughput.
