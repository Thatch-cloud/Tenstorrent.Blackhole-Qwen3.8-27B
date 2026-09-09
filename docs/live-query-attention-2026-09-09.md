# Skip unused draft-query rows

**Simulator work; no hardware speedup or PP/CTX/TG claim yet.**
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
| Full2K draft history | Simulator run `20260909T034627Z-647` underway; not yet qualified |
| Hardware ABBA | Gated on both complete-attention reports and unchanged source hashes |
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

Initial evidence: `20260909T033107Z-414` (QK), `033234Z-553` (learned attention),
and `033838Z-415` (padding initialized once). `034053Z-402` was stopped while
finalizing the hardware-reporting interface; it is not a long-context pass.
Reports and logs are retained under `hardware-evidence.local/live-query-sim/`.
