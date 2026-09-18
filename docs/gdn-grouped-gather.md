# Grouped local Q/K copies

**Simulator passed; no hardware speedup is established.**

| Gate | Evidence | Result |
|---|---|---|
| CPU source checks | 35341749555 | Exact source/destination word coverage; only the copy loop changes |
| Full GDN simulator | 35341952099 | 24 exact output/state/bridge comparisons; 48 immutable-input checks; eager and three changed-input replays, both chips |
| Cleanup | Same simulator run | Probe exit zero; devices close; all container cleanup statuses zero |
| Combined hardware comparison | Pending | Must preserve the working DFlash2 runtime and compare complete requests |

The candidate loads eight local words before their stores, in groups of four
per face. It changes neither arithmetic nor precision, CB size, input order or
checkpoint publication. The hypothesis is reduced scalar load/use stalls; this
is different from the rejected Q/K double-buffer and V-first reorder candidates.
Simulator timing is not a performance prediction. The GDN group alone cannot
close the complete measured 40.5 ms/block gap to 200 TG.

Report SHA256:
`e645f7de77e3a31b086c388ef120ebd6522d51087c66084945c0960b2faff91b`.
Candidate reader SHA256:
`14f7d21ccb81b07f5cb581fbdd765d32e25e06f8ac5d2eaf25a080301a6a52b4`.

The admission gate checks the complete replay matrix, immutable sources, exact
artifact/runtime, clean exit, local/native dependencies and reconstructed reader
hashes. The runtime scope restores the original loader and preserves compute,
writer and norm-prefetch composition. It is not enabled in serving defaults.

## Cheap rejected alternative

## Combined serving comparison

Image build **35342832559** and CPU checks **35342808631** passed. The
candidate is packaged in image
`sha256:29153355a914393e98345ff77c5bd3619943f1d93b6fd0042d7fe538665c2ff1`.

The hardware canary loads the model once: warm control, warm candidate, then
control/candidate/candidate/control. All six requests use the same 4K prompt,
122-token exact reference, T16 DFlash2 recipe, precision and four fabric links.
Only the local recurrence reader copy schedule changes. Two warmups are excluded.

`gather-comparison.json` requires exact HTTP outputs, contiguous decode accounting,
identical acceptance/buckets, matching candidate kernel hashes and clean device
shutdown. It reports whole-cycle TG and verifier time separately; nested trace
time is not added twice. This is a paired screen, not production acceptance or
proof of 200 TG. The candidate remains off by default.

## Lookup screen

A retrospective lookup screen against the existing 4K reference found no full
15-token candidates with an eight-token suffix match across 121 output positions.
At four matching tokens, six positions qualified but averaged only 0.167 matching
proposal tokens. This overlapping-position screen is not a measured decoding run
or a general coding-workload conclusion. It does not justify replacing the
current neural drafter or spending a hardware run on this fixture's lookup path.
