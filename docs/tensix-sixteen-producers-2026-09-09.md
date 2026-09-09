# Sixteen weight producers: test issue capacity

**Implemented; full-size simulation is running. No hardware result yet.**
This follows the [device profile](tensix-mlp-device-profile-2026-09-09.md), which
locates the eight-producer design's regression inside gate/up/down.

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
  It is still running; a JSON body alone is not success.
- Simulator and hardware reports record the exact producer count and full sender
  mappings. Eight-producer evidence cannot qualify a sixteen-producer runtime.
- CI suite `tensix-stream-mlp-16` requires `tensix-mlp-simulator-16.json` and its
  successful outer exit artifact before build/device access. These are not yet
  supplied. Historical eight-producer results retain their original source tags.
- After simulation, restore the original native packer, then run the same real
  Layer-0 nine-block ABBA. Both arms request four links; all samples and exact
  checks remain required. No greater-than2% win in every block means no promotion.

The existing reviewed packer compatibility patch is scoped to this TT-Sim run.
No firmware or serving defaults change. Component latency remains separate from
the full objective: **200 committed TG for one coding stream**, preserving quality.
