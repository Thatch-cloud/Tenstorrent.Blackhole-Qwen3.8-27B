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

Next: source-bound admission, then a matched combined-request comparison with
shared-Q/K enabled in **both** arms and only buffering changed. Reusing the old
native-versus-shared-Q/K schedule would confound the comparison. Keep serving
defaults and the existing 72.27 TG baseline unchanged until hardware acceptance.
