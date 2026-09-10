# Untuned local coding screen

These three frozen tasks were not used to tune the current kernels. They are
not a standardized benchmark or evidence of exclusion from model training.
Target-token equivalence and functional correctness are separate checks.

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

## Remaining work

Run-length encoding is running in hardware `34459087621` using the same immutable
source. Rotate-right has not run. Functional results and PP/CTX/TG must be
recorded separately for each task; do not pool different-task rates as a matched
speedup. Broader coding quality, long contexts and the 200 TG target remain open.
