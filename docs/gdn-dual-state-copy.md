# GDN state-copy fan-out candidate

**Host prototype only. Not simulator-admitted, hardware-tested or enabled.**

The shared-Q/K recurrence copies each new state twice for intermediate tokens:
once into the externally published state buffer and once into local next-token
feedback. Both destinations have four BF16 tiles per worker. The final token
needs only the published copy.

`gdn_dual_state_copy.py` proposes one tile unpack/register handoff followed by
two packs to the existing buffers. It retains both outputs, token ordering,
BF16 conversion boundaries and the original final-token path. It allocates no
additional circular buffers and does not alias independently consumed buffers.
For T16 this removes 60 repeated tile unpacks/handoffs per recurrence worker
per layer, while preserving both sets of packs. This is not a latency estimate.

This differs from the rejected paired-copy experiment: that grouped adjacent
tiles but still executed both complete copies. It also avoids revisiting the
rejected outer-add and input-cache scheduling changes.

Three host tests check source transformation, both destination formats and
rejection of changed anchors. Native generated-source validation and device
pack/CB semantics remain unproven. Next gates:

1. Apply to the exact pinned shared-Q/K recurrence source; keep native helper
   setup and all non-state-copy arithmetic unchanged.
2. Simulator eager/replay checks of every prefix state, output, input integrity
   and clean teardown, including changed inputs and final-token behavior.
3. Source-bound integration into the unchanged winning T16/HiFi4 combined
   runtime, followed by native token/state/feature audits and matched TG timing.

Do not claim a speedup from saved instructions. The retained recurrence-like
group occupies about 9.53 ms per verifier replay, including waits; the fraction
spent on these copies has not been measured. Serving defaults remain unchanged.

The candidate now has an isolated program-construction wrapper and frozen
simulator staging. It retains the existing 24 full output/prefix-state/bridge
comparisons and 48 input-integrity checks across eager and three changed-input
replays. Generated control/candidate kernel hashes are recorded, and construction
must occur exactly once. Five host tests pass, including construction failure
restoration; fresh `8c102b2` staging imports successfully and passes shell syntax
checks. CI performs native-source transformation and compilation before the
device correctness checks, without model weights or physical-card execution.
