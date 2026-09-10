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

The other two tasks are pending, not submitted concurrently because the workflow
concurrency group retains only one pending run. This screen uses no experimental
batched-publication kernel and does not change serving defaults.

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
