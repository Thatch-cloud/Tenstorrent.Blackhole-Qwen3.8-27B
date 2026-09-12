# GDN paired-copy experiment

Status: simulator passed; complete-request hardware comparison pending. No speed claim yet.

| Item | Decision |
|---|---|
| Current matched baseline | PP 3331.48 / CTX 4096 / TG 106.38; one stream |
| Bottleneck under investigation | Recurrence conversion register handoffs inside the verifier |
| Change | Copy two tiles per register acquire/commit/wait/release instead of one |
| Unchanged | Arithmetic, conversion boundaries, state rounding, circular buffers, reader/writer, norm/gate |
| Simulator scope | One synthetic T16 GDN operation per chip, changed-input trace replay; no model weights |
| Resource limit | Runner CPU only: 16 CPUs, 64 GiB; probe timeout 15 minutes |
| Hardware admission | Requires exact outputs, every prefix state, FP32 bridge, immutable inputs and clean close in simulator |
| Performance acceptance | Matched complete requests with all accepted runtime improvements; never add isolated gains |

The recurrence already keeps its next-token state locally. This experiment does
not claim to eliminate a per-token DRAM state read. It reduces synchronization
around existing format-conversion copies, without deleting the BF16 rounding
that defines the recurrence's numerical behavior.

The source generator changes only the recurrence copy helper and rejects changed
anchors. The candidate is not wired into serving. The experimental hardware route
changes only T16 recurrence construction against the combined T16 runtime.
Other token widths and norm kernels stay native.
The prior wider-output-grid and L1-output experiments remain disabled.

Simulator run 34695921492 passed 24 exact comparisons covering gated output,
every prefix state and FP32 bridge, plus 48 input-immutability checks. Both chips
closed cleanly. Retained report SHA256:
`14d9fa6ae9b2e5ec0674d5b846f382c141e38ee73fa423ddb82a8ac73258bd99`.
