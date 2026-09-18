# Gate conversion and exponentiation fusion

Status: simulator correctness qualified; hardware/performance unqualified.
Not enabled in the combined runtime or serving defaults.

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
Next: wire the admission and scope into a matched combined hardware comparison.
No throughput result exists for this candidate yet.

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
