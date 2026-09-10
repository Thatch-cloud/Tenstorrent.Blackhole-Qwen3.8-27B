# Native DSpark attention: matched hardware results

[Run 34446713555](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34446713555),
revision `332a600`: two P150A cards, one stream, CTX4096, T16 verifier and
15 draft queries. Both arms use captured proposals and commit-only GDN.

| Attention | PP tok/s | CTX | Committed TG tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Composed control | 3303.51 | 4096 | 60.15 | 220/360 (61.11%) |
| Precise-native | 3310.11 | 4096 | **86.69** | 222/330 (67.27%) |

**+44.12% committed TG** in the matched comparison. Each arm has one instrumented
audit and two timed requests, ordered A/B/B/A after audits. Native timed results
are87.31 and86.08 TG. All six requests have identical target output hashes and
exact target state/inactive slots. Each timed arm commits242 tokens total.
Draft proposals differ between backends, and each backend reproduces its own
audited proposals. No numerical threshold or target acceptance rule was relaxed.

| Mean ms/block | Composed | Native |
|---|---:|---:|
| Draft, including copies/readback | 78.78 | 37.78 |
| Verify + readback | 76.35 | 76.17 |
| Select + publish | 11.44 | 11.70 |
| Whole cycle | 167.59 | 126.84 |
| Committed tokens/block | 10.083 | 11 |

Native SDPA avoids the external FP32/head-replication attention path. The measured
drafting reduction is40.998ms/block, not a per-operation attribution. Target
verification barely changes and is now the largest component. At11 committed
tokens/block,200TG requires a55ms whole cycle: the verifier alone exceeds it.

The changed kernel passed82 simulator checks before hardware. The full learned
synthetic-layer simulator timed out in its baseline and remains incomplete;
it is not counted as a pass. The hardware result is full-request target correctness,
not a separate CPU numerical qualification of every learned intermediate.

Source and native-runtime hashes match before/after; the precise exponential
patch is scoped to the native arm and restored. The prompt hash is
`647801186828b64da09ce25736a818d75feb9555e07e05c9d1056c78b36b7485`, matching the
previous allocation-order comparison. Serving defaults remain unchanged.
Held-out coding quality, endpoint latency and long-context scaling are not certified.

Raw report SHA256: `cbafa38784a26bf57f10e7ba461ada85bd1cc59a2efc01eee1566608c761ac9d`.
An independent repeat uses the same immutable tag in
[run34447570149](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34447570149).
