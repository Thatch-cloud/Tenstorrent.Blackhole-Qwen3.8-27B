# Banked drafting: combined-runtime admission

Status: combined hardware correctness passed; no meaningful speedup. Not promoted.

## Hardware result: 11 September

Run [34547542628](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34547542628),
revision `53595894d9ea7cb227696291be5d2642d233c706`. Independent validation checked
717 source files, learned score-layout evidence, exact proposals, target outputs,
state, bank use, and complete request timing. This is not held-out coding certification.

| One stream, merge-intervals prompt | PP tok/s | CTX | Committed TG tok/s | Mean setup-inclusive ms |
| --- | ---: | ---: | ---: | ---: |
| Combined control, copied history | 3324.92 | 4096 | 100.13 | 5947.50 |
| Same runtime, bank-bound traces | 3289.95 | 4096 | 100.17 | 6416.36 |

Two audited requests preceded four A/B/B/A timed requests. Each arm committed
242 timed tokens with 222/330 accepted drafts. Candidate bank replay counts were
7 and 5 in every request, including warmup. The +0.036% difference is not evidence
of an improvement; extra capture setup makes setup-inclusive latency worse.

| Mean per block, 22 timed blocks per arm | Control ms | Banked ms |
| --- | ---: | ---: |
| Draft | 27.072 | 28.221 |
| Verify and readback | 69.219 | 69.077 |
| Select and commit | 12.391 | 11.399 |
| Whole cycle | 109.809 | 109.774 |

At 11 committed tokens per block, 200 TG requires a 55 ms whole cycle. The verifier
alone exceeds this. Do not pursue bank binding as a demonstrated throughput win;
the next candidate must reduce verifier cost or increase accepted tokens per unit
verification work, then be judged on combined requests again.

Hardware report SHA256:
`a3951f1c6d69939ffafb0d08834c98f5240531fc42aa63caf3c94ce97de421f8`.

## Admission design

The candidate binds one captured draft graph to each existing history bank. After
publication swaps banks, it selects the matching graph instead of copying all
history into a third bank. At capacity 4352, the avoided destination payload is
42.5 MiB per chip per proposal; this is not a measured time saving.

| Evidence | Result | Does not prove |
| --- | --- | --- |
| Revised TTsim bank lifetime probe | 80 output and 200 bank checks exact; both graphs replay twice | Learned drafting or speed |
| Focused host tests | 42 pass, including matched-arm admission | Accelerator correctness |
| Combined learned request | Exact audits and matched timings pass | Meaningful speedup or held-out quality |

Simulator report: `/opt/ttsim/results/20260911T002837Z-443-dspark-banked-trace-probe.json`.
SHA256: `967dcbee569e215993e5fd29d2cadcba85a51f5f6a34137e67b9502bc832f248`.
The checked-in JSON is reserialized with identical parsed contents; its SHA256 is
`0815094d6fc1d166b095231bb67fa56383d5d6b3b8b30f144f1d58e19b5fdc78`.
The independent gate checks every chip, operand, bank transition, post-close
readback, replay audit, exit status, and current source digest. It deliberately
labels this synthetic evidence. The active-trace allocation warning still occurs
during eager auditing; learned request checks must establish whether it is safe.

`measure_dspark_request` accepts explicit `banked_proposal=True` and a
`banked_proposal_evidence` report path. It requires the combined traced native
drafter, commit-only GDN, and folded T16 target path. Defaults are unchanged.
Reports include per-bank replay counts. The workflow accepts
`dspark_banked_proposal=true` with `dspark_score_layout=true` and the
`dspark-target-attention-request` suite. Both arms use fused scores; only the
candidate binds history banks. Two audited requests precede A/B/B/A timed requests.

Future reuse must retain learned feature/history/proposal/target-state audits,
PP / CTX / committed TG and setup-inclusive latency, both-bank execution, and
coding functional tests. Do not promote based on the simulator pass alone.
