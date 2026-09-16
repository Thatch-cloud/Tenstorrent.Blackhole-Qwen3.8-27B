# Incremental history in the frozen 32K runtime

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
