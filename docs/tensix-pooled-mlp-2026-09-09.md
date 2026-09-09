# Complete streamed MLP: shared buffers, real boundaries

**Simulation and real-weight hardware correctness pass; performance fails.**
The streamed MLP is 2.22 times as slow as native. No promotion or serving change.

## Hardware result

[CI 34330511791](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34330511791)
completes on `b668f14`, after the metadata-only loader fix.

| Complete Layer-0 MLP, T8, TP2 | Mean latency | Decision |
| --- | ---: | --- |
| Native control | **0.334683 ms** | Retain |
| Pooled streamed weights | 0.743766 ms | Reject; 122.23% higher latency |

Both arms include the captured DRAM input copy and native four-link collective.
All nine ABBA blocks lose; all 36 samples remain included. The 118 eager,
replay, stale-control, raw-input/weight and timed-output checks pass, as do
native weight-view audits and clean teardown. The independent artifact gate
passes correctness and explicitly returns `eligible_for_full_model_gate=false`.

Report SHA256:
`c755e250f37750d859caf94d9197014bbdecb2e559c79709b84251a654ae064d`.
These are component milliseconds, **not PP or TG**. No full-model promotion run
is justified. Next diagnostic: measure producer/forwarding/consumer costs before
changing the streaming pipeline; contention and credit waits remain hypotheses.
The separate [device attribution suite](tensix-mlp-device-profile-2026-09-09.md)
uses unchanged qualified kernels and cannot qualify instrumented timing for promotion.

## What this adds

| Part | Implementation |
| --- | --- |
| Workload | T8 target verification, not eight independent serving streams |
| Compute | Full gate, up, native product and down; unchanged BF4/BF4/BF8 weights |
| Workers per card | Eight weight producers; 68 gate/up or 80 down receivers |
| Gate/up weight FIFO | One shared 36,864-byte allocation per participating receiver |
| Down weight FIFO | One shared 34,816-byte allocation per participating receiver |
| Workspace | Caller-owned gate/up/hidden/partial tensors plus one L1 input |
| Input boundary | Native DRAM-to-preallocated-L1 copy inside the captured operation |
| Concurrency | Sequential CQ0 only; overlapping traces cannot share this pool |

The pool avoids allocating separate global circular buffers for every projection
of every layer. A host test prepares 64 layers with only two allocations. The
device simulation switches between two distinct synthetic weight sets using the
same FIFOs and output workspace. This is not yet a 64-layer model result.

## Qualification ladder

| Gate | Evidence or requirement | State |
| --- | --- | --- |
| Full gate/up/down separately | Both chips, exact native controls, all 32 physical rows, changed-input traces | Pass |
| Pooled complete MLP | Two weight sets, shared buffers, captured input copy, native product, changed-input replay | Pass |
| Native weight views | Full 2D native shards, exact 4D aliases on both chips, original packer | Pass |
| Host safeguards | 983 CI tests plus 57 simulator-harness tests | Pass; not device evidence |
| Real-weight complete MLP | Layer 0, three input patterns, native four-link reduction, nine ABBA blocks | Correctness pass; 2.22x latency regression |
| Complete request | Exact target verification/state, committed tokens, PP / CTX / TG | Not dispatched: component performance gate fails |

The complete-MLP simulator gate requires 32 native-control comparisons, 32 eager
comparisons, 48 replay comparisons, 70 raw input/weight checks, four stale-input
controls and two different-weight controls. Twenty experiment files and 29 native
files bind the result. A successful JSON body without a zero outer wrapper exit
is rejected. The explicit simulator packer graft is never used on hardware.

Run `20260909T073101Z-390` passes all 188 checks and closes both devices cleanly;
the outer wrapper exits zero. Report: `scripts/ci/tensix-mlp-simulator.json`,
SHA256 `8dfa6e6b1f5a41cede8a8c0166319286971d359b94856ddfe9b8f7b8ee992c6c`.
Its `.exit-status` companion is checked in too. The native simulator packer was
restored to its original hash, both native Python binaries remain unchanged,
and the owned graft lock is removed. Qualification passes again after restoration.

## First hardware attempt: loader shape mismatch

[CI 34327911787](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34327911787)
passes simulator/source preflight but stops during preparation, before any
candidate kernel, output comparison or timing. Both devices close cleanly.

The native loader returns 2D weights: gate/up `[5120,8704]`, down `[8704,5120]`
per chip. The qualified streamed path expects `[1,1,K,N]`. The fix uses native
`ttnn.experimental.view`, borrowing the same two physical buffers. It does not
repack, loosen kernel validation, mutate `mlp.weights`, or change arithmetic.

The dedicated simulator probe exercises full-size native column-sharded
gate/up and row-sharded down weights. It checks both chip addresses, exact
dequantized contents, original metadata/lifetime, and no new program-cache entry.
The original packer is used. This separate evidence supplements, rather than
replaces, the unchanged 20-file complete-MLP arithmetic qualification.

View probe `20260909T083448Z-382` passes all six chip/weight checks and exits
zero after clean teardown. Report SHA256:
`0e824e62d3a4eec10ad7ffcf4b4f4063a866857436652d301a4894a347feb097`.
The report and outer exit artifact are checked in. Both independent simulator
gates pass again against current sources and the original native runtime.

The retry keeps both arms at four links, isolating the shape fix. Four links
showed no whole-request TG gain; changing link policy during this retry would
mix two changes. The retry establishes the latency regression, not a throughput gain.

[Retry 34330511791](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34330511791)
passes on the immutable `ci-qwen-hardware-b668f14` tag. Native view-source hashes and
the new simulator report/exit are checked before any hardware device is opened.

## Collective ownership matters

The pinned native `tt_all_reduce` TP2 branch calls
`reduce_scatter_minimal_async`, then forcibly deallocates its input. Passing our
pooled partial output to that wrapper would destroy the buffer needed by later
calls and traces.

The adapter calls the **same native reduce-scatter operation with the same
parameters and semaphore sequence**, but leaves the caller-owned input alive.
There is no extra output copy and no replacement collective kernel. Source hashes
pin the native wrapper; tests verify every argument and prohibit deallocation.
Hardware comparisons still use the actual native MLP forward as the control.

The opt-in full-verifier adapter is prepared but not enabled. Its host tests
require all 64 MLP calls in order, one shared workspace, exception-safe method
restoration and no deallocation of aliased caller inputs. It retains the native
input-memory conversion. Device correctness and full-request speed remain open.

## Hardware decision

CI suite: `tensix-stream-mlp`. Source/simulator preflight runs before the CCL build
or device probes. It requires the simulator report **and** its successful exit
artifact; missing or changed evidence stops the job.

The component test records the helper's original link requests and explicitly
matches control and candidate at four links. A separate
[whole-request two-versus-four-link test](target-model-link-counts-2026-09-09.md)
keeps that policy change separate from weight streaming.

Each timed trace includes input copying, all MLP operations and the four-link
native collective. The test retains all 36 ABBA samples, 50 replays per sample,
and verifies both outputs after every sample. Packed weights and caller input
must stay unchanged. Clean teardown is mandatory.

Only a gain greater than 2% in every matched block makes the candidate eligible
for full-model testing. Correct but slower results are retained, not promoted.
Even a component win does not establish 200 committed TG: the complete verifier
and request must be measured next.
