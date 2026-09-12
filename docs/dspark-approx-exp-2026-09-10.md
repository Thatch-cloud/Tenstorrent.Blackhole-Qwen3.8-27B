# Approximate proposal exponential: numerical failure

The isolated approximate-exp candidate fails the existing simulator attention
tolerance. It is not enabled in hardware requests or serving, and supplies no
throughput result. Target-model arithmetic is unchanged.

Only native proposal SDPA's `exp_approx_mode` changes to true. BF16 operands,
HiFi4 math, FP32 accumulation, 64-key chunks and the fixed-history mask remain.
The simulator uses capacity 4384, fifteen live proposals and the original
FP32-reference tolerances `rtol=0.01, atol=0.01`.

| First eager case, chip 0 | Result |
| --- | ---: |
| Elements outside tolerance | 1627 |
| Maximum absolute difference | 0.6718368530 |
| Replay and remaining cases | Not reached |
| Exit status | 1 |
| Mesh closed cleanly | Yes |

Run `20260910T085012Z-386` fails after about 55 simulator seconds. Retained
report: `scripts/ci/dspark-approx-fixed-attention-failure.json`. The outer owner
restored the original packer and both SDPA sources; both ownership locks are
absent. No tolerance was relaxed and no hardware retry was dispatched.

This rejects the candidate under the current component accuracy contract; it
does not prove target-output corruption, because target verification was never
executed. A different approximate proposal policy would need explicit numerical,
acceptance and full-request qualification rather than relabelling this a pass.

## Remaining latency budget

The folded-attention control in hardware run `34456312393` averages 10.636
committed tokens per block, with 36.967 ms drafting, 69.057 ms verify/readback,
10.804 ms selection/publication and 117.820 ms whole cycles. These are measured
host block boundaries, not an isolated kernel critical-path decomposition.

At unchanged acceptance, 200 TG requires roughly **53.18 ms per whole cycle**.
Even deleting drafting entirely leaves roughly 80.85 ms, about 131.6 TG under
that idealized assumption. Draft optimization alone cannot meet the target.
Verification and/or accepted tokens per block must improve substantially too.
