# Direct-scatter GDN norm reader

## Stronger follow-up: passed

Run `20260910T073624Z-398` exits 0 with 12 eager comparisons, 24 changed-input
trace replay comparisons and 30 verified output-poison operations. Every
comparison is bit-exact, borrowed inputs remain unchanged, and inactive output
rows are zero. Chips receive distinct FP32 bridge inputs. Flipping input columns
changes the expected output, detecting stale-input reuse.

The complete output is poisoned before each eager invocation, changed-input
control and replay. Its 32-row logical view exposes physical padding. This
closes the missing-write blind spot described below. The independent report
validator accepts the retained evidence and current source fingerprints.

Retained report: `scripts/ci/gdn-norm-scatter-simulator.json`.
SHA256: `195e12fd7fb49710bdc2c63ffa942e89796336739aaab56da8c35df3a4b20c91`.
Ten scatter/layout/report host tests pass. This qualifies only the isolated
norm experiment, not a full GDN/request path or any throughput improvement.

## Earlier evidence and limitations

Status: initial isolated comparisons passed, but their reused output buffer is
not a sufficient missing-write control. No hardware qualification, speed result
or serving integration.

The candidate replaces each temporary-stick read, wait and CPU copy with two
64-byte reads directly into the FP32 tiled norm input. One final read barrier
replaces the per-stick barriers. Norm arithmetic, output writer and recurrence
sources are unchanged. The existing qualified implementation remains untouched.

| Check | Result |
| --- | --- |
| Token rows | 1, 16, 32 |
| Fixtures | Two random input scales per shape |
| Simulated chips | Both |
| Output comparisons | 12/12 bit-exact against existing norm reader |
| Borrowed inputs | Unchanged in all comparisons |
| Output finiteness | Passed |
| Source fingerprints | Before, after and current files match |
| Cleanup | Mesh closed; pinned packer restored; owner lock removed |
| Host tests | Five scatter tests and three runtime-owner tests pass |

Evidence: `/opt/ttsim/results/20260910T073050Z-495-gdn-norm-scatter-probe.json`
and its companion log/exit-status, exit 0. Two earlier launches stopped on
missing source paths before comparison; the successful run uses the retained,
hash-checked native GDN source artifact.

This is not trace-replay, complete recurrence, full-request correctness or
performance qualification. Inputs are replicated across the simulated chips;
distinct chip patterns, poisoned output padding and changed-input trace replay
remain necessary before integrating the candidate into a hardware request.
In particular, poison the output before every candidate invocation and replay:
otherwise the preceding control invocation leaves the expected result in place,
which could mask a missing candidate write. The initial equality results alone
must not authorize hardware integration.
The additional NoC packets may offset the reduced waits: measure, do not infer
a throughput improvement. The 200 committed tok/s objective remains unmet.
