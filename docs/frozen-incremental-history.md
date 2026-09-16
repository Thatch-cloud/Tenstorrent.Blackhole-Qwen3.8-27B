# Incremental history in the frozen 32K runtime

## Matched hardware result

Run **35084843498**, revision `84d8dc6`, passes in **14m10s**. Two audited
requests precede four A/B/B/A timed requests. This is one short coding fixture,
not held-out coding-quality or serving acceptance.

| One stream | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Full-bank publication | 2935.22 | 32768 | 79.46 |
| Incremental publication | 2961.67 | 32768 | **87.53** |

The matched improvement is **10.15%**. Candidate repeats are 87.49/87.57 TG;
control repeats are 79.97/78.96. Both arms commit 234 timed tokens and accept
216/300 proposals. All six requests have identical prompt/output hashes and
exact target output, target state and inactive state. Each candidate request
records ten completed publications, at most 64 touched rows, and restored hooks.
All 847 reported Python/source entries and 1,520 native entries remain unchanged
between the report's before/after snapshots. The container exits cleanly with
no OOM. Pre-load full I/O stall is 0.0393%; this is not ongoing isolation proof.

| Mean block phase | Control ms | Incremental ms |
| --- | ---: | ---: |
| Draft | 50.80 | 52.07 |
| Verification/readback | 73.84 | 73.98 |
| Selection/commit | 21.56 | **6.17** |
| Complete cycle | 147.18 | **133.61** |

At the observed 11.7 committed tokens/block, 200 TG requires a 58.5 ms cycle.
Even removing all remaining publication work cannot meet that budget: verifier
cost alone exceeds it. Keep this qualified publication candidate, then pursue
draft/verifier cost and accepted-token efficiency. Do not add rejected MLP
buffering or treat the marginal GDN-cache result as an established win.
Cross-run comparisons to the earlier 72.27 TG result are not a matched gain;
acceptance/block count differs. Serving defaults remain unchanged.

Report SHA256:
`f7fabefae99412da4a27f6566528c625150cbdd01331ca92d02c8d224dc45ab1`.

## Integration and retained qualification

The frozen recipe rebuilds, pads and copies ten full history banks on each
commit. Its prepared proposal also copies full banks before replay. These are
separate costs: this experiment changes **publication only**, not proposal
updates or draft attention. The measured 32K select/commit envelope is roughly
22 ms; that is not a measurement of the writer alone.

Reuse the existing dirty-tile writer rather than reimplementing it. Its simulator
and physical addressing/replay evidence already exists:

| Evidence | Result |
| --- | --- |
| Simulator 35023307310 | 84 exact checks, changed-input replay |
| Hardware 35024279412 | 84 exact checks across the 64K boundary, 42 seconds |
| Current source audit | Retained hardware report and every reported source hash match |
| New host transaction tests | Full 33,024-row banks match at starting positions 32,767 and 32,768, including commit/discard and poisoned delta tails |

Hardware report SHA256:
`7a962f5b7cc519d61cead689a225c68995ce0a600951b65c4fe0c8f2a24301cd`.

The host tests are not device acceptance at 32K. Next run a matched integrated
comparison with shared-Q/K, original MLP buffering and original draft attention
in both arms. Candidate changes only captured publication to the retained writer;
verify exact full history, target output/state, proposal acceptance and source
identities. Do not mix in the small GDN-cache candidate. No serving changes.

The deployment adapter now provides `--incremental-history`, isolated from
profiling, MLP-buffer and GDN-cache experiments. Both arms retain shared Q/K;
only the candidate enters the incremental publication scope. Host tests verify
arm identity, transaction accounting and restoration, including exceptional exit.
The adapted historical checkout passes the retained hardware source-hash gate.
Local validation: 61 frozen tests, three optional checks skipped; all five
incremental tests pass when the retained hardware report is supplied.
The integrated 32K hardware comparison is recorded above. Broader quality and
longer-context qualification remain pending.
