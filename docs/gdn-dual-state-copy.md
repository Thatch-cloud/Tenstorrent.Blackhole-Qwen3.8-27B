# GDN state-copy fan-out candidate

**Simulator execution passes; combined hardware performance remains unqualified.**

## Simulator result

[Run 35231694382](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35231694382)
at `225720f33b7d4879eac62fe4e9e4a69dbba1e026` passes in **3m05s**.
Independent inspection confirms all 24 output/prefix-state/bridge comparisons
and 48 immutable-input checks, including changed-input replay. All 791 captured
source hashes remain stable; the 14 selected local probe/dependency hashes match
the staged files. Runtime is the pinned `9f9cd4f`; exit and all cleanup statuses
are zero. This supplies no TG measurement or held-out coding acceptance.

| Evidence | SHA-256 |
| --- | --- |
| Report | `30910463b67866b0ccad62784cedf9e82d88cc9b6f074368d71ea983f09c5f9e` |
| Generated control recurrence | `ce404cf287f9962243dc65f30d9736f45b0c95cfff503c6c20d68f8981a41153` |
| Generated candidate recurrence | `abf6a1d2ce0fd9f8380656a09d7963fcd7ac8ed9017f6dff49bcbcccb3cd7a68` |

Exactly one transformed recurrence is recorded, with four state tiles and
unchanged output/feedback CB18/CB30. Next: bind hardware admission to this
generated source and integrate only the copy change into the winning combined
T16/HiFi4 runtime, preserving its norm-prefetch path and target correctness gates.

## Implementation and qualification history

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
