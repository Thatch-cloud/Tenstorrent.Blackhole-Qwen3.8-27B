# Draft KV assembly: pad the tail, not the history

Status: simulator and full-size hardware component passed; combined comparison prepared.
No serving changes or new committed-TG result.

## Why test this

The current native draft layer slices 7/15 proposal rows, concatenates them onto
the entire fixed history, then pads that large result. The history capacity is
tile aligned, but the intermediate concatenation is not.

The candidate pads only the small proposal tail to 32/64 rows first, then joins
two tile-aligned tensors. It preserves every historical row, every valid query
row and every zero-padding row. Attention, masks, precision and acceptance rules
are unchanged. Poisoned unused query rows must never reach the result.

The incomplete capture **35151664646** motivates this test, but does not qualify
it: inferred draft trace 2, chip 0, replay 4 contains roughly 5.0 ms concat,
3.8 ms padding and 6.6 ms untilize/tilize kernel-duration sums. These groups are
not all necessarily attributable to KV assembly; they are not additive critical
path savings or a throughput prediction. The clean combined capture is separate.

## Gates

| Gate | Scope | State |
| --- | --- | --- |
| Host tests | Original versus reordered row assembly; both tail sizes; poisoned padding; alignment and layout rejection | Three tests pass |
| Weight-free two-chip simulator | BF16 bitwise eager output and changed-input traces; 64/96 history rows, 7/15 proposals, both chips | 35157432497 passes all 112 checks |
| Hardware component | Exact full-size output and replay; measure original versus candidate | 35158163367 passes in 37 seconds |
| Combined 32K runtime | Same qualified recipe, candidate changes assembly only; exact proposals/output/state; PP/CTX/TG | Prepared; not yet qualified |

The simulator has a six-minute whole-job cap, uses no model weights and has no
physical device access. Its 112 checks cover original and candidate eager output,
three changed-input replays and unchanged inputs on both chips. Simulator time
is never reported as hardware throughput. Neither the prototype nor its test
adds a runtime hook or modifies serving defaults.

Simulator run **35155862154** did not execute the probe: root checkout encountered
root-owned artifacts from an older marker workflow and failed cleaning the shared
workspace. The retry uses `draft-tail-source/` as an isolated checkout and results
directory, leaving unrelated artifacts untouched. This is not a numerical failure.

The isolated retry **35157432497** passed; its simulator step took 2m15s. Independent
validation checked all 112 unique check identities, exact results, clean close,
zero exit status, the pinned TT-Metal revision and all six reported source hashes.
Simulator report SHA256:
`57ef37a3f2fbc44943e33c0e636e96afedd453bf992c60b9e5b3c030535af279`.

The hardware harness preserves these operations and replay patterns, extending
history to 33,024/33,056 rows. It adds eight blocking trace measurements per arm
per shape in ABBA order and checks the final outputs again. The gate requires
128 exact checks and 64 timing samples. These component milliseconds are not TG.
The job remains weight-free, rejects occupied cards, and has a seven-minute cap.

## Full-size component result

Run **35158163367** passes all 128 bitwise checks and 64 ABBA timing samples.
Independent validation confirms the report, every source, generated harness hash,
clean exit and no OOM. These are blocking assembly-trace medians on the two-card
mesh, **not token-generation throughput**.

| History capacity | Proposals | Original ms | Tail-first ms |
| --- | --- | --- | --- |
| 33,024 | 7 | 1.411 | 0.208 |
| 33,024 | 15 | 1.412 | 0.212 |
| 33,056 | 7 | 1.236 | 0.205 |
| 33,056 | 15 | 1.241 | 0.209 |

Hardware report SHA256:
`cb31fac3a1767be5d6c278d4cbf758745c1b0382091a3615449d5ef75e208bb0`.

The combined comparison keeps shared Q/K, fused T16 verification, norm prefetch,
incremental publication and the original MLP readers in **both** arms. Only
proposal K/V assembly changes. It preserves the 33,024-row capacity, exact request
audits and A/B/B/A timed requests. Profiling is disabled; no additional fences are
inserted. The candidate must execute all ten K/V assemblies per five-layer draft.

## Combined run: incomplete, not a performance result

Run **35158707469** hit the 900-second command timeout (exit 124).
Both feature-audited requests completed: each committed 117 tokens over 11
blocks, with exact output, target state and inactive state. Norm prefetch was
enabled in both arms; only the candidate used the tail hook (140 construction
calls, including setup). No timed request completed.

| Phase | Previous successful run 35087582465 | This run |
| --- | ---: | ---: |
| Target load, stage-to-stage | 50.41 s | 292.30 s |
| Control feature audit | 230.17 s | 249.04 s |
| Candidate feature audit | 231.31 s | 277.67 s |
| First timed request starts, from harness start | 531.79 s | 849.35 s |

Harness elapsed time excludes launcher setup. The outer command therefore
expired shortly after the first timed request began. Audit-mode draft timings
include verification overhead and must not be compared with ordinary decode TG.
The pre-load I/O admission passed (0.859% full stall); this does not establish
the cause of the later slow load or prove isolation throughout the run.

The partial report has `passed=false`, `closed_cleanly=false`, and no final
source-immutability checks. Its two completed audits are useful diagnostic
evidence, **not a reusable full-run qualification**. Do not silently substitute
them for fresh audits or promote this candidate. The accepted 32K result stays
at 89.01 TG. Before retrying, the test cycle needs a phase budget that accounts
for load, both audits, all four timed requests and cleanup, rather than another
unchanged 900-second attempt.

Partial report SHA256:
`00658e8b2f9f04090b7c34aa3dbc707ae29d702debeead2699927a733e7efbbd`.

## Shorter qualification and timing cycles

The next run separates the existing six-request schedule into two phases:

| Phase | Work | Evidence required |
| --- | --- | --- |
| Qualification | Both full feature-audited requests, final weights/source checks and clean shutdown | Fresh execution; the timed-out report is not admitted |
| Timing | Four fresh uninstrumented A/B/B/A requests with exact output/state checks | SHA256-pinned, clean qualification with identical runtime sources, native sources, model fingerprints, context and cache formats |

The qualification-only job has a 720-second hardware command cap, a 13-minute
step cap (including cleanup), and an 18-minute whole-job cap including staging.
It refuses to begin the audits if harness setup has already exceeded 100 seconds.
This avoids spending another nine minutes auditing after a five-minute load.
Normal setup previously took about 70 seconds; the budget is not a guarantee
against future host contention.

The audit report explicitly contains no PP or TG. The later timing report must
identify the retained audit run and digest, preserve the original per-arm proposal
and acceptance checks, and obtain four fresh complete timed requests. Changing
any bound source or model fingerprint invalidates reuse. The full numerical
schedule is preserved across phases; no serving configuration, tensor operation,
precision, or timing boundary changes. The timing phase is not dispatched until
the new qualification completes successfully.
