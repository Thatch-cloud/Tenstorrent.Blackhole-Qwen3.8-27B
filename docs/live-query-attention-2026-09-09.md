# Skip unused draft-query rows

**Hardware attention latency improves7.3% short /9.5% long; not a PP/CTX/TG result.**
The cached draft QK kernel computes32 query rows, but a T8 proposal has only8
live queries. This opt-in candidate evaluates the native SFPU row-broadcast
multiply for two four-row bands instead of all eight bands. The live arithmetic,
accumulation order, FP32 formats, cached inputs and64-worker cap stay unchanged.
The usual 32-row tile remains the storage format; this is not smaller MLP tiles.

| Work | Status |
| --- | --- |
| Live QK and zeroed padding | Initial simulator: 4 eager /12 traced comparisons pass on two chips |
| Captured learned operands through complete attention | Initial simulator: 12 eager /36 traced component comparisons pass |
| Final source-pinned short gate | `20260909T034508Z-606`: 12 eager /36 traced component checks and4 stale controls pass |
| Full2K draft history | `20260909T034627Z-647`: 12 eager /36 traced component checks and4 stale controls pass |
| Hardware ABBA | Run34310168821 passes exactness and wins every block at both history lengths |
| Learned stack / complete coding requests | Not integrated; existing lead and serving defaults unchanged |

## Why padding can be skipped

Each padded query's mask allows exactly one zero-bias key. Replacing its finite
QK scores with zero therefore preserves the one-hot softmax row and the complete
attention output. A host validator rejects other padding masks before capture.
Live masks, sliding-window behavior and all value rows remain unchanged.
The probe compares **all32 rows** of probabilities and final attention output,
not just the live8. Raw QK compares the live8 exactly and requires zero padding.

The candidate writes defined zeros to padding and rewrites every live score.
Both traces remain live while inputs change. Input immutability, stable bindings
and four stale-input negative controls are required. There is no native source
graft, dtype relaxation, approximation or learned-weight modification.

## Gates and timing boundary

- Short gate replays a hash-pinned layer0/rank0 learned Q/K/V/mask and a changed
  synthetic pattern. Rank0 is replicated on two simulator chips; it does not
  certify the original rank1 or the whole learned layer.
- Long gate covers2080 physical key rows from2048 draft-history rows plus the
  proposal block and padding. **Draft history is not target request CTX.**
- Hardware compares captured **complete attention**, including softmax and PV,
  using uploaded synthetic operands. Six ABBA blocks,50 replays/sample, exact
  timed outputs and unchanged inputs/bindings are mandatory.
- A greater-than2% win in every block is required before learned integration.
  Model loading, projection, communication and full-request overhead are not
  included in this component timing. It cannot establish200 committed tok/s.

The CI route preflights both reports before opening devices, uses the physical
P150-pair descriptor and skips model-weight downloads and native runtime rebuilds.
Both hardware artifacts are mandatory and independently validated on the host.
895 CI host tests and55 simulator-harness tests pass. The final short report's
12 experiment-source hashes and3 native header hashes validate independently.
The long report passes the same independent gate. Both use the original native
packer, with no simulator source or library graft. Long simulation takes1493.8s;
that is simulator wall time, not measured device latency or hardware throughput.

Initial evidence: `20260909T033107Z-414` (QK), `033234Z-553` (learned attention),
and `033838Z-415` (padding initialized once). `034053Z-402` was stopped while
finalizing the hardware-reporting interface; it is not a long-context pass.
Reports and logs are retained under `hardware-evidence.local/live-query-sim/`.

## Hardware result

[Run34310168821](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34310168821)
on `88ff247` passes the independent host validator for both artifacts. Each history
has12 eager and36 replay component checks, four stale controls and six ABBA
blocks with50 replays/sample. Every timed output and input/binding check passes.

| Draft-history rows | Control attention | Live-query attention | Latency reduction |
| ---: | ---: | ---: | ---: |
| 31 | 0.241856 ms | 0.224225 ms | 7.29% |
| 2,048 | 3.544460 ms | 3.207765 ms | 9.50% |

These are captured **complete attention** timings on uploaded synthetic operands,
not target request context sizes, learned layers, whole-drafter timing or TG.
All six blocks at each history exceed the2% threshold. Next is learned-stack
integration and then a matched complete coding request; serving stays unchanged.

Artifacts under `hardware-evidence.local/34310168821/artifacts/qwen-hardware-inventory-34310168821/`:
`live-qk-31.json`, SHA256 `20251df3b7bb49a08297c4aee54da388d1ff62c7bfc6674d180fa8f28a490e38`;
`live-qk-2048.json`, SHA256 `ab6319efa978afa77c1d034d1cb63045ffac955e45c9d2d486651a6703e7efe2`.

## Request integration

[Hardware run34314276820](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34314276820)
is underway on `a1ccc70`; no request timing or correctness verdict is available yet.

`full-dflash-live-query-request` compares the existing cached T8 lead against the
same runtime with live-query attention enabled in all five draft layers.
Two instrumented correctness requests precede uninstrumented control / candidate /
candidate / control requests. Target context is 4,096 tokens; draft history is 2,048.
Both arms retain fused convolution, commit-only GDN, native sampling and four links.

- The candidate uses the original BF16 GQA inputs, FP32 arithmetic, softmax and PV.
- Every host mask is validated before upload. Failed updates revoke registration;
  closing a capture removes registrations before its buffers are freed.
- Both source-pinned integration simulator reports are mandatory before hardware.
- Hardware must preserve proposals, acceptance, committed tokens and native state.
  A missing artifact or failed independent validator fails CI.
- The run reads DRAM-core tensor-prefetcher capability for the next verifier
  experiment. It does not start a prefetcher or change firmware.

Short integration report `20260909T044143Z-431` passes: four eager output checks,
12 changed-input replay checks and four stale-input controls; all32 rows are exact
on both chips. The learned fixture is saved layer0/rank0, replicated across the
simulated pair, not proof of original rank1 or full-model coding quality.
The 2K synthetic integration report `20260909T044521Z-376` also passes the full
four eager / 12 replay / four stale-control matrix and clean closure. Independent
qualification verifies 19 experiment sources and three native sources for each
report. Its 1,858.5 seconds of simulator time are not a hardware latency estimate.
Host tests: 915 CI and 55 simulator harness tests pass.

| Checked-in integration report | SHA256 |
| --- | --- |
| `live-attention-simulator-31.json` | `9b43e0a5a80b6b538b59416d2d1a5e00a1535a356e8d3401678949dbaf2fd698` |
| `live-attention-simulator-2048.json` | `7130e685b5fcd5c6ec4904171df5757d1c41271ecd738fe46ba74af4a62be2b1` |

Logs and exit statuses are retained in `hardware-evidence.local/live-query-integration-sim/`.
No additional PP/TG claim or serving-default change.
