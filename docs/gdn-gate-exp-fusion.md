# Gate conversion and exponentiation fusion

Status: simulator and combined HTTP correctness screen pass; performance
promotion rejected. Not enabled in the normal combined recipe or serving defaults.

## Combined hardware result

[Run 35348721064](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35348721064)
at `1956d5e` completed in **3m17s**. One model load, two compilation warmups and
four timed ABBA requests passed exact HTTP token identity. All candidate requests
recorded 96 admitted constructions and restored their scope. Worker and devices
closed cleanly; server exit was zero. Independent artifact validation passed.

| CTX / streams | Arm | Committed cycle TG | Verifier replay | Whole cycle |
|---|---|---:|---:|---:|
| 4096 / 1 | Control | 114.38 | 59.124 ms | 96.169 ms |
| 4096 / 1 | Gate-exp fusion | 116.41 | 59.035 ms | 94.494 ms |

Both timed arms committed 242 tokens across 22 blocks. Aggregate TG is **+1.77%**,
but paired changes are **-2.10% / +5.80%**: not repeatable. Mean blocking verifier
replay improves only **0.0885 ms**, while most of the aggregate difference is in
other host intervals. Do not promote, attribute the entire difference to this
fusion, or rerun it unchanged. PP was not separately measured by this HTTP screen.

At eleven committed tokens per block, 200 TG requires **55 ms for the whole
cycle**. This control takes 96.17 ms, including roughly 21.72 ms draft, 62.74 ms
verification/readback, 10.53 ms commit and 1.17 ms outside those phases. Blocking
replay is nested within verification; do not add it again. Eliminating this one
intermediate does not address the roughly 41 ms remaining whole-cycle gap.

Report SHA256: `bd979cf4f9d23bcce457ce5181d106adb4ad26ba35d4988547ed5bfc4e852501`.
Preload I/O stall was 0.5934%; that observation does not prove later isolation.
This remains a one-fixture screen, not held-out coding-quality acceptance.

## Simulator result

[Run 35347681287](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35347681287)
at `2419678` passed in **3m9s**, without model weights. Independent artifact
validation confirms 24 exact output/bridge/prefix-state comparisons and 48
input-immutability comparisons across eager execution and three changed-input
replays on both simulated chips. Source fingerprints stayed unchanged, exit
status was zero, and container cleanup passed.

Report SHA256: `ce1be67c12fc03a6514fca5e69882f9642d7ec902ebd5446123991cec4dc55c2`.
The admission gate pins this artifact and reconstructs the generated kernel;
the optional scope changes only recurrence compute and restores the loader.
The `fast-serving-canary-v*-gate-exp` tag selects a matched combined HTTP
comparison using the cached image with six explicitly listed, read-only helper
overlays. Their hashes are recorded. Admission runs before model loading, then
again before attaching the worker runtime. Both compilation warmups are excluded;
four timed requests run control/candidate/candidate/control after one model load.
All three candidate requests must record 96 admitted recurrence constructions.
The comparison retains native DFlash, serial BF4 weights, direct DMA draft KV,
four fabric links, CTX4096 and one stream in both arms. HTTP token identity is
checked against the retained native-reference request. This is a performance
screen, not held-out coding quality or a full hardware state-parity qualification.
The hardware result is recorded above; no default enablement follows from it.

## Change

| Path | Operation |
|---|---|
| Current | BF16 gate -> FP32 scratch CB23 -> exp -> FP32 CB13 |
| Candidate | BF16 gate -> FP32 destination registers -> exp -> FP32 CB13 |

The existing `expc` helper already copies its input into destination registers
before applying SFPU exp. Use it directly on the BF16 gate and remove the
intermediate pack, unpack and CB handoff. Keep the same exp operation, FP32
destination configuration, decay arithmetic, reader, writer, state feedback,
worker count and buffer allocation. CB23 remains allocated but unused, so this
experiment does not mix fusion with an L1 layout change.

This is a limited reduction in recurrence overhead, not a predicted 200 tok/s
solution. The latest combined progressive-input experiment measured roughly
59.5 ms verifier replay alone. At 11 committed tokens per block, the complete
cycle must take 55 ms to reach 200 tok/s; draft and commit still cost time.

## Gates

1. Lightweight transformation and scope-restoration tests.
2. Weight-free simulator: exact outputs and all prefix states against the
   native reference, changed-input trace replay, immutable inputs, both chips,
   clean close. Seven-minute whole-job cap; no model weight loading.
3. Only after simulator qualification, add a source-pinned admission gate and
   evaluate the unchanged combined recipe with matched control/candidate runs.
   Report PP, CTX, complete-cycle TG and verifier replay separately.

A simulator pass is correctness evidence only. Do not promote based on a
standalone kernel timing or reduce numerical tolerances to admit this change.
