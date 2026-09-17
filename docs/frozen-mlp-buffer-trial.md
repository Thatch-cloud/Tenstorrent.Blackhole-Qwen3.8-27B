# T16 MLP buffering trial

Hypothesis: additional input/weight buffering may hide producer stalls in the
11.39 ms fused-MLP group of the 32K verifier. The aggregate profile does not
prove producer stalls. No speedup is assumed.

| Setting | Control | Candidate |
| --- | ---: | ---: |
| Input blocks buffered | 2 | 4 |
| Weight blocks buffered | 2 | 4 |
| Token rows / pairs per worker | 16 / 3 | 16 / 3 |
| Arithmetic, reader code, tile order | Historical | Unchanged |

Candidate adds 32,768 input bytes per participating core and 55,296 weight
bytes per compute worker. There are 91 compute workers in a 99-core multicast
rectangle. Additional L1 use must also fit the integrated runtime.

## Simulator result

Run **35078247907** (`63770ba`) passes in **3m57s**. It uses one geometry-matched
MLP fixture, not a full model load. Both chips pass eager comparison, all twelve
changed-input replay checks, four stale-input controls, and four byte-exact
gate/up weight checks (43,520 pages per check). Exit status is zero.

| Pinned evidence | SHA256 |
| --- | --- |
| Simulator report `fused-batch.json` | `a37834f16393996a9fe2e1c0142c21e37cbde2100722dce9cdfe827b775db7dc` |
| Candidate `fused_1d.py` | `ded6f11da9b2f501d925e6809467538499549fed799a97624550072faee92d46` |
| Original `fused_1d.py` | `4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93` |

Only T16 is qualified by this run. Hardware speed, integrated L1 capacity,
other row widths and coding quality are not established. The staging manifest's
qualification fields remain false: they describe pre-execution state, not the
subsequent simulator result.

## Matched hardware result: not adopted

Run **35079244639** (`b01c3cd`) passes in **15m20s**, with shared-Q/K enabled
in both arms. All six requests pass exact output/state/inactive-slot checks;
both arms reproduce the same proposals and acceptance. Two audited requests
are excluded from timing; four timed requests use A/B/B/A order.

| CTX 32768, one stream | Two-block control | Four-block candidate |
| --- | ---: | ---: |
| Committed TG tok/s | 71.807 | 71.314 |
| Verifier/readback ms/block | 73.936 | 74.399 |
| Complete cycle ms/block | 148.061 | 149.081 |
| Committed tokens/block | 10.636 | 10.636 |

The candidate is approximately 0.69% slower in this sample; two timed requests
per arm do not establish a general regression. They do establish **no demonstrated
gain**, so retain two-block buffering and do not spend another full run on the
same candidate without new evidence. Integrated L1 capacity and correctness pass.
Serving defaults remain unchanged; the earlier 72.27 TG result remains the
recorded baseline, not a speedup from this experiment.

Hardware report SHA256:
`cb1b8fbe5b3034cb17a058254015d4c40b4216ad8e8fc201306b5e3277c3ee17`.
