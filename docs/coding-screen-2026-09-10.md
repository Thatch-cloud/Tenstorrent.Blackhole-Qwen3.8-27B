# Untuned local coding screen

These three frozen tasks were not used to tune the current kernels. They are
not a standardized benchmark or evidence of exclusion from model training.
Target-token equivalence and functional correctness are separate checks.

## Score-layout rerun

The repeat-confirmed score-layout kernel is being screened on the same frozen
tasks, in order: `stable_unique_v1`, `run_length_encode_v1`, `rotate_right_v1`.
Each task gets its own matched native-score-layout control with folded T16 target
attention in both arms, exact request audits and A/B/B/A timing. Different-task
rates will not be pooled or substituted for the 4K merge-intervals baseline.

Stable-unique run
[34541480692](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34541480692)
completed on immutable `83f679f`. All 688 revision-bound source fingerprints,
learned score/token checks, request correctness checks and four isolated
functional cases pass. Performance does **not** pass: the candidate regressed.

| Score layout | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native control | 1 | 2499.98 | 4096 | 73.52 |
| Fused candidate | 1 | 1459.38 | 4096 | 22.37 |

Both arms commit 104 timed tokens with 98/120 accepted proposals. Candidate
decode durations are 2214.52 and 2434.94 ms; control durations are 903.93 and
510.67 ms. Prefill also varies substantially (candidate 1374.58/4238.76 ms).
This suggests broad timing instability but does not establish its cause or
excuse the regression. Container exit is zero with no OOM kill. Repeat the
same frozen task before advancing the screen; do not promote this result.
Report SHA256: `26e914466f6b3d70b7072b60072373546bf9d3da90f54707e3cbffcd48bf7d1d`.

Identical immutable repeat `34542359051` passes all revision-bound correctness
checks and four isolated functional cases, before any runtime change.

| Repeat path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native score-layout control | 1 | 3038.15 | 4096 | 59.66 |
| Combined fused score layout | 1 | 3024.83 | 4096 | 112.55 |

Candidate decode durations are 465.73/458.31 ms for 52 tokens each. Control
durations are 1236.54/506.54 ms. Both arms retain 98/120 acceptance across their
two timed requests. This reverses the previous regression but does not establish
a repeatable 88.64% gain: the control is unstable and the prior candidate was slow.
Do not discard the bad samples or promote the fastest result alone.
Repeat report SHA256: `9d39791a3f92aabc8312d0e7660d8cffa4b0acaf1965a9ab3bfb6e2fc3365b16`.

For subsequent revisions, `request_host_health.py` records cgroup CPU throttling,
memory events and raw pressure snapshots around each complete request, outside
its timed work. Missing counters remain unknown, not zero. These diagnostics
include setup and audits and cannot independently prove decode interference or
exclusive host use. They are not present in the immutable repeat above.

### Host-instrumented combined result

Run `34543303342`, immutable `f49cee9`, passes independent validation of all 707
source fingerprints, exact learned/request checks and the same four functional
cases. Runtime kernels are unchanged; host snapshots are outside timed work.

| Path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native score-layout control | 1 | 2531.27 | 4096 | 107.36 |
| Combined fused score layout | 1 | 3346.45 | 4096 | 114.43 |

Both arms commit 104 timed tokens with 98/120 acceptance. Candidate decode
durations are 472.16/436.71 ms; control durations are 477.27/491.44 ms.
The matched TG improvement is 6.58%, not the previous unstable 88.64% estimate.
Control prefill still varies (1227.07/2009.25 ms). Across all six whole-request
windows, CPU quota throttling and memory-limit/OOM event deltas are zero; memory
pressure totals do not increase. CPU pressure is nonzero. This does not establish
the cause of earlier outliers or prove an exclusive host. Do not discard them.
Setup-inclusive means are 5768.67 ms candidate and 6753.29 ms control.
Report SHA256: `62ebfea992b011765bf66bf43ed71b9bfb8e91346c933516d91c4be7663a0895`.

### Run-length encoding: combined correctness passes, timing regresses

Run `34543886854` on the same immutable `f49cee9` passes all source, learned
proposal, target-state and four isolated functional checks. Both arms commit 210
timed tokens with 196/240 accepted proposals.

| Path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native score-layout control | 1 | 2001.16 | 4096 | 100.31 |
| Combined fused score layout | 1 | 1451.70 | 4096 | 33.39 |

Candidate request durations are 5267.59/1021.13 ms; control 1013.21/1080.27 ms.
The slow candidate localizes to `session.commit` / selection-publication: its first
block spends 2533.32 ms there, followed by several 368-534 ms blocks. Drafting
remains about 26-32 ms and verification/readback about 69-70 ms. This is not
evidence that the score-layout kernel itself stalls. No CPU quota throttling,
OOM kills or memory-pressure total increases occur in the request windows.
CPU pressure is nonzero; GC, compilation, publication substeps and other host
interference are not yet independently attributed. Keep all outliers in the result.
Report SHA256: `61f5809340abb134a8fd466f2566d2bcf093b9bb2acf67c67cc3cd54089bd689`.

The diagnostic repeat is recorded below. `rotate_right_v1` is now running as
`34545607581` on the same immutable `1c27221` combined runtime. Tasks are not submitted
concurrently because the workflow
concurrency group retains only one pending run. This screen uses no experimental
batched-publication kernel and does not change serving defaults.

Revision `1c27221` records host-wall/process-CPU duration and Python GC pauses for
feature access, drafter-history preparation, target publication and history commit.
It adds no device fences and does not disable GC or alter kernels. Device work may
be charged to a later synchronization; these are host-stage boundaries, not device
kernel durations. The diagnostics are inside measured request time and must be
labelled instrumented. Transaction failure/discard and prefix-zero tests remain
mandatory; all 47 focused tests and retained request simulator prerequisites pass.

Run `34544974175` independently passes source, request, publication-stage and four
functional checks. Both arms commit 210 timed tokens with 196/240 acceptance.

| Instrumented path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native score-layout control | 1 | 3343.22 | 4096 | 99.83 |
| Combined fused score layout | 1 | 3174.52 | 4096 | 113.29 |

The multi-second stalls do not recur. Timed history preparation is approximately
9-37 ms, with no recorded GC pauses in those stages; this does not identify or fix
the previous outlier. Candidate decode requests take 884.01/969.63 ms, control
953.63/1149.87 ms. Setup-inclusive means remain worse for the candidate:
6915.46 versus 6400.77 ms. No outliers are removed and no promotion is claimed.
Report SHA256: `51443079b34484a412b7acfb0b7f508ff6889a9eb95be4fe80741af521a4cead`.

## Stable unique: completed

Hardware run `34458413589`, source `348c61b`, passes all six matched requests
on the two P150A cards. Independent recomputation reproduces the saved summary,
with exact target outputs/state/inactive slots, complete proposal audits,
unchanged source/native fingerprints and clean closure.
Hardware report SHA256:
`a3b4873bed3b0d2106a674f9c56c08d3c1df9eb11788f967afd5f184c88d8c01`.

| Path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native target attention | 1 | 3350.17 | 4096 | 103.20 |
| Folded T16 target attention | 1 | 3351.31 | 4096 | 109.04 |

Two timed requests per arm commit 104 tokens total; acceptance is 98/120.
The short function reaches EOS. Relative TG gain is 5.66%, while mean
setup-inclusive time worsens from 5458.41 to 5510.43 ms. This is one run on a
different task, not a replacement for the 89.86 repeat-confirmed tuning result.

The identical generated function passes all four private functional cases in
the local isolated evaluator, including empty input, negatives and duplicate
handling. Input-mutation checks pass. Generated-source SHA256:
`be5425920cfe6f480cec6f038d37ce9c2107852c7e2826438c16242966165b3a`.
Task SHA256:
`75e3df4078665be31e9f6bfa17ff82f20cc50443e83caee75e270308932eb949`.

Evaluation uses a separate local Linux root, unprivileged dynamic user, private
network/devices, read-only system and process/memory/time limits. Live controls
reject wrong output, input mutation and an infinite loop. Workspace, accelerator
devices and the TT runtime are not mounted into the evaluator. This is a bounded
functional screen, not a proof against adversarial generated-code deception.

## Run-length encoding: completed

Run `34459087621` on the same `348c61b` source passes all six requests with
exact target outputs/state, matched proposals, unchanged fingerprints and clean
closure. Independent recomputation reproduces its summary.

| Path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native target attention | 1 | 3346.55 | 4096 | 104.33 |
| Folded T16 target attention | 1 | 3320.98 | 4096 | 107.18 |

Two timed requests per arm commit 210 tokens total and accept 196/240 proposals.
TG improves 2.73%; setup-inclusive latency is 5916.93 versus 5933.83 ms.
The EOS-terminated generated function passes all four isolated functional cases,
including Unicode code points and separated runs of the same character.
Generated-source SHA256:
`ae28580a32a815ea53bd50863be9d91a00490334438937d4c21a9c92db8f3529`.
Hardware report SHA256:
`9913eefe60a905a6163dc0bee2386bab830724d58a90971fc1df2ace6494bdc6`.

## Rotate right: completed

Run `34459661914` passes all six matched requests on the same immutable source.
Independent recomputation reproduces the summary; experiment/native fingerprints
are unchanged and closure is clean.

| Path | Streams | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Native target attention | 1 | 3325.73 | 4096 | 73.06 |
| Folded T16 target attention | 1 | 3382.31 | 4096 | 76.93 |

Two timed requests per arm commit 128 tokens and accept 116/210 proposals.
Setup-inclusive latency is 5710.46 versus 5657.55 ms. The EOS-terminated function
passes all five isolated functional cases, with input preservation.
Generated-source SHA256:
`84e739616807a222ead11e1abbdae128c9c6f52817e1a3fca30e14fa8768a76e`.
Hardware report SHA256:
`06020a5887cac7558ef3b4725b11658e2bb426aa7b3929c4c92dca61dc8228bc`.

## Remaining work

All three task summaries and isolated functional results have been rechecked.
Results are recorded separately for each task; do not pool different-task rates as a matched
speedup. Broader coding quality, long contexts and the 200 TG target remain open.
