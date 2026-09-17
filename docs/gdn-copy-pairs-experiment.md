# GDN paired-copy experiment

Status: exact hardware comparison passed; no demonstrated trace speedup. Candidate disabled.

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

## Complete-request result

Run **34696377960**, revision `fcc4cae`, one stream / batch 1. Each arm has one
instrumented correctness request and two timed requests, 121 committed tokens per
response, CTX 4096. Timings exclude the instrumented requests.

| Arm | PP tok/s | CTX | Committed TG tok/s | Mean blocking trace ms/block | Setup-inclusive ms/request |
|---|---:|---:|---:|---:|---:|
| Native copies | 3323.97 | 4096 | 105.49 | 68.009 | 6030.20 |
| Paired copies | 3311.16 | 4096 | 106.26 | 68.151 | 6133.32 |

TG improves 0.73%, but the targeted blocking-trace interval worsens by 0.141 ms.
Host input staging and publication are faster in the candidate measurements;
this experiment does not establish that paired copies accelerate the kernel.
Do not promote this as a repeat-confirmed gain or add it to other speedups.

All six requests retain exact target output, recurrent state and inactive slots.
Both arms accept 222/330 proposed tokens across the timed requests. Candidate
requests record 96 T16 recurrence program constructions and scope restoration.
All 779 script hashes match the measured revision before and after execution;
both pooled TG values were independently recomputed from committed tokens/time.
No held-out coding-quality or context-ladder qualification is claimed.

Hardware report SHA256:
`65393049cbfcf91cb4c4040a90407e02d101b057495b381f6413d6df8d970812`.
