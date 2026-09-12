# GDN paired-copy experiment

Status: candidate only. No speed or hardware correctness result yet.

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
anchors. The candidate is not wired into the serving or hardware request paths.
After simulator admission, compare against the unchanged combined T16 runtime.
The prior wider-output-grid and L1-output experiments remain disabled.
