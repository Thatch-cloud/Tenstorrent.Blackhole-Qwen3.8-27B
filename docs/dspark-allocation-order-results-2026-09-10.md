# DSpark trace allocation repair: hardware results

[Run 34442013592](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34442013592),
revision `84f7718`, passes and closes cleanly. Two P150A cards, one stream,
CTX4096, fifteen draft queries and T16 target verification. Serving remains unchanged.

| Arm | PP tok/s | CTX | Committed TG tok/s | Versus matched eager |
|---|---:|---:|---:|---:|
| Eager | 3304.00 | 4096 | 54.93 | baseline |
| Captured proposal | 3355.78 | 4096 | 56.92 | +3.62% |
| Captured proposal + commit-only GDN | 3319.26 | 4096 | 59.81 | +8.89% |

Each arm has one instrumented audit and two timed requests, in forward/reverse
order. All nine requests retain exact target output, state and inactive slots.
Each timed arm commits 242 tokens total, accepting 220/360 proposals (61.11%).
TG includes proposal copies, verification, publication, readback and stalls.
This establishes correctness for this experiment, not held-out coding quality.

## Where time remains

Mean milliseconds per block across both timed requests:

| Arm | Draft | Verify + readback | Select + publish | Whole cycle |
|---|---:|---:|---:|---:|
| Eager | 85.41 | 83.40 | 13.04 | 183.50 |
| Captured | 79.96 | 83.60 | 12.42 | 177.10 |
| Captured + commit-only | 78.77 | 76.49 | 12.10 | 168.53 |

At 10.083 committed tokens/block, 200 TG requires about 50.42 ms for the
entire cycle. Drafting and verification each exceed that budget independently.
Capture alone is not enough: next work must reduce device arithmetic/conversion
and verifier cost, not just host launch overhead.

The repair allocates verifier persistent buffers before proposal trace capture.
Previous proposal calls overwrote 138 saved verifier shard hashes; the protected
snapshot and full feature/state checks now pass. No tolerance was relaxed.

The prompt includes repository source, including the edited verifier. Its hash
is `647801186828b64da09ce25736a818d75feb9555e07e05c9d1056c78b36b7485`.
Do not compare acceptance or speed against older prompts as a matched experiment.
The historical 74.27 TG DFlash2 result remains higher, but is not this run's control.

Raw report: `runner-evidence.local/34442013592/qwen-hardware-inventory-34442013592/dspark-request-variants-hardware.json`.
SHA256: `44a5c129ad722c987480a452bc9114cf78ce584f8afddd6180c74bdb6cb3ce3b`.
