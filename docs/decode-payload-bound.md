# Decode payload feasibility check

2026-09-05, analytical estimate, **not measured DRAM traffic or throughput**.

Inputs are the model config captured in run 33942471680, snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and the pinned TT-Metal loaders:
`tt/mlp.py` (gate/up bfloat4, down bfloat8), `tt/gdn/tp.py`,
`tt/attention/tp.py`, and `tt/model.py` (projection/head bfloat8).
Paths are relative to `models/demos/blackhole/qwen36` at the reviewed revision.
The image diff audit lists no changes to the MLP loader or model config.

For conventional single-token decode, assume each dense projection streams its
weights once per step, partitioned across the pair. Count only packed value payload:
0.5 byte for bfloat4 and 1 byte for bfloat8. Exponent blocks, padding, activations,
KV/state traffic, collectives, vision and startup are excluded. Embedding lookup
does not read the entire embedding matrix and is not counted as a dense projection.

| Projection family | Payload per complete step, decimal GB |
| --- | ---: |
| 64 MLPs: `64 * 5120 * 17408 * (0.5 + 0.5 + 1)` | 11.4085 |
| 48 GDNs: `48 * 5120 * (2*2048 + 3*6144 + 2*48)` | 5.5601 |
| 16 gated attention layers: `16 * 5120 * (3*24 + 2*4) * 256` | 1.6777 |
| LM head: `5120 * 248320` | 1.2714 |
| Total | **19.9177** |

200 ordinary single-token steps/s would require **3.9835 TB/s combined payload
bandwidth**, before all excluded costs. This is a conditional bandwidth requirement,
not a measured hardware ceiling. Re-measure achieved bank/read bandwidth and full
trace attribution before predicting attainable rate. More active cores alone do not
remove these reads. Multi-token verification can amortise them, but must include
acceptance, recurrence, rollback, draft and sampling costs before claiming a gain.

## Disaggregation capacity implication

The current eight-slot, 65536-context pool reports 524288 token capacity before
scheduler padding. At TP1, 16 attention layers with four KV heads, head dimension
256 and bfloat8 K/V require at least 16 GiB of value payload for that pool.
Together with the 18.55 GiB projection payload above, this exceeds a 32 GiB card
before embedding weights, exponents, traces, GDN state or scratch.

Therefore E10b cannot preserve that entire eight-slot pool on each TP1 replica
at unchanged layouts. A B=1 or smaller-pool TP1 arm remains a separate feasibility
experiment: disclose the capacity change and compare against an equivalent TP1
unified control. This does not prove a single-stream TP1 replica is infeasible,
nor does it establish that handoff or split serving would be faster.
