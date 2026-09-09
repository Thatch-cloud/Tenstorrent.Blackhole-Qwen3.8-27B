# Complete streamed MLP: shared buffers, real boundaries

**All full-size projections pass simulation. Complete-MLP simulation is running.**
There is no new hardware latency or PP / CTX / TG result yet. Serving is unchanged.

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
| Pooled complete MLP | Two weight sets, shared buffers, captured input copy, native product, changed-input replay | Running |
| Host safeguards | 959 CI tests plus 57 simulator-harness tests | Pass; not device evidence |
| Real-weight complete MLP | Layer 0, three input patterns, native four-link reduction, nine ABBA blocks | Wired; not dispatched |
| Complete request | Exact target verification/state, committed tokens, PP / CTX / TG | Not qualified |

The complete-MLP simulator gate requires 32 native-control comparisons, 32 eager
comparisons, 48 replay comparisons, 70 raw input/weight checks, four stale-input
controls and two different-weight controls. Twenty experiment files and 29 native
files bind the result. A successful JSON body without a zero outer wrapper exit
is rejected. The explicit simulator packer graft is never used on hardware.

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

## Hardware decision

CI suite: `tensix-stream-mlp`. Source/simulator preflight runs before the CCL build
or device probes. It requires the simulator report **and** its successful exit
artifact; missing or changed evidence stops the job.

Each timed trace includes input copying, all MLP operations and the four-link
native collective. The test retains all 36 ABBA samples, 50 replays per sample,
and verifies both outputs after every sample. Packed weights and caller input
must stay unchanged. Clean teardown is mandatory.

Only a gain greater than 2% in every matched block makes the candidate eligible
for full-model testing. Correct but slower results are retained, not promoted.
Even a component win does not establish 200 committed TG: the complete verifier
and request must be measured next.
