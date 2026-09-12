# Sixteen weight producers: test issue capacity

**Correct on hardware, but 36.17% slower than native. Not promoted.**
This follows the [device profile](tensix-mlp-device-profile-2026-09-09.md), which
locates the eight-producer design's regression inside gate/up/down.

The real-weight hardware comparison passes correctness in
[CI34337968780](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34337968780)
on immutable tag `ci-qwen-hardware-ca22384`. The downloaded artifact also passes
independent local validation. All nine timing blocks lose against native.

## Hardware result

| Complete Layer-0 MLP | Latency, ms | Evidence |
| --- | ---: | --- |
| Native control, this run | **0.334690** | Same real weights, input copy and four-link reduction |
| Sixteen producers, this run | **0.455749** | 36.17% slower; loses all nine ABBA blocks |
| Eight producers, earlier run | 0.743766 | Historical comparison, not a paired arm in this run |

Sixteen producers reduce the earlier prototype's latency by **38.72%**, with
native controls effectively unchanged between runs. That is progress within
the prototype, not a win over the existing runtime and not a new TG result.
More producer capacity helps this design; this alone does not separate DRAM
command issue, transfer scheduling, NoC contention and consumer waits.

All118 hardware checks pass: six eager, twelve trace, four stale-input controls,
24 raw-input/weight checks and72 timed-output checks. All36 samples are retained,
with50 replays per sample. Native weight metadata/buffers stay unchanged and
teardown completes cleanly. `eligible_for_full_model_gate=false`.

Hardware report SHA256: `afbb889e934c36669094e0c00b287111cd923529cf65ceef97059288f3bb0e8c`.
Artifact: `10098683116`, 70,058 bytes, SHA256
`306648e1e1e0fce3c6f95f16a3a27d7ee579cd6e1e218830331aac9cb2b6696c`.

The next kernel investigation should target producer/transfer overhead at a
fixed mapping, rather than assuming another core-count increase will beat
native. Any new candidate still needs its own simulator gate and matched
hardware comparison; neither streamed version enters the full model.

| Setting | Eight-producer prototype | New candidate |
| --- | --- | --- |
| Weight-reading workers per card | 8 | 16 |
| Sender coordinates | x0-7, y9 | x0-7, y8-9 |
| Gate/up compute workers | 68 | Same |
| Down compute workers | 80 | Same |
| Total active workers, including multicast drains | 85 gate/up; 96 down | 93 gate/up; 104 down |
| Maximum receivers per sender | 9 gate/up; 10 down | 5 |
| Arithmetic and precision | Native LoFi/FP32, BF4/BF4/BF8 weights | Unchanged |
| FIFO protocol and capacity per receiver | Two pages; two pooled GCBs | Unchanged |
| Input copy, activation multicast, product, collective | Qualified existing paths | Unchanged |

Only Python geometry/routing changes; no new C++ kernel instructions. More
producer issue capacity may help, but neither the profile nor spare-core count
establishes that it will. The native MLP remains the hardware control.

## Gates

- 998 CI tests and 57 simulator-harness tests pass. Mapping tests prove complete,
  single assignment, no sender/activation overlap, and identical compute arguments.
- Simulator run `20260909T092252Z-403` uses two complete weight fixtures, all32
  physical output rows, native controls, changed-input traces and raw-word audits.
  All188 checks pass: 32 native, 32 eager, 48 replay, 70 raw-input/weight,
  four stale-input controls and two distinct-fixture checks. Teardown and the
  outer wrapper both finish successfully; this is not a speed measurement.
- Simulator and hardware reports record the exact producer count and full sender
  mappings. Eight-producer evidence cannot qualify a sixteen-producer runtime.
- CI suite `tensix-stream-mlp-16` requires `tensix-mlp-simulator-16.json` and its
  successful outer exit artifact before build/device access. Both are supplied
  byte-identically from the completed run. All20 experiment and29 native source
  hashes match. Historical eight-producer results retain their original tags.
- The original native packer is restored byte-identically; both native Python
  binaries are unchanged. The independent MLP and native-weight-view gates pass
  again against the restored runtime, and the owned simulator lock is released.
- The real Layer-0 nine-block hardware ABBA completes with four links in both
  arms. All samples and exact checks pass, but the greater-than2% win in every
  block is absent, so the candidate is not promoted.

Report SHA256: `6e3d1bb41a4a91c8c8f92741a703f512cb220740763ebc5c1fcaeec6787711fe`.
Outer-exit SHA256: `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The existing reviewed packer compatibility patch is scoped to this TT-Sim run.
No firmware or serving defaults change. Component latency remains separate from
the full objective: **200 committed TG for one coding stream**, preserving quality.
