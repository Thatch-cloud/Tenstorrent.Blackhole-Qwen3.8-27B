# Batched GDN publication experiment

Status: implementation and host checks only. No simulator or hardware pass yet.
The qualified publication path and serving defaults remain unchanged.

`scripts/ci/gdn-batched-publication-probe.py` now implements the first simulator
comparison: native and candidate against an independent host reference, two
distinct chip inputs, all 17 prefixes at one layer, and selected boundary prefixes
at 48 layers. It resets inputs and NaN-poisoned checkpoints before every eager
execution and changed-input captured replay. It checks all logical source, native
and checkpoint elements and stable device addresses. Checkpoints are poisoned
including physical tile padding. Readback expands the host tensor to its padded
shape, requiring checkpoint padding to become zero while source/native padding
canaries remain NaN. The host-only TTNN tiling/readback mechanism passed a local
check without opening a device; device execution is still unproven.
The previous full-chain simulator was deliberately stopped as incomplete and its
runtime restoration verified. One-layer execution is now running as
`qwen-batched-publication-1-20260911.service`, report
`20260910T224336Z-417-gdn-batched-publication-probe.json`. No pass is claimed yet.

`gdn_batched_publication_gate.py` independently requires the complete ordered
matrices for both layer counts, exact source hashes before/after, clean closure,
exit zero and all padding/poison checks. Its rejection tests pass; this does not
substitute for executing the simulator or the later hardware continuation gates.

After the existing owner exits and restores its runtime, use the bounded launcher:

```sh
QWEN_SIM_LAYER_PROBE=gdn-batched-publication-probe QWEN_SIM_BOUNDED_MEMORY=1 \
  KERNEL_TIMEOUT=1800 bash optimisation/sim/run-native-layer-dispatch-probe.sh --layers 1
```

Require complete passing evidence before repeating with `--layers 48`. Neither
report alone replaces the full accepted-prefix continuation hardware gate.

## Why test it

The matched score-layout request spends about 11.77 ms/block in selection and
commit, including more than this kernel. Source inspection shows the current
publication kernel waits after each recurrent tile and each convolution tile.
That is not evidence that all 11.77 ms is removable.

The isolated `gdn_commit_batched_dma` candidate keeps two workers per layer
(96 workers for 48 layers). Each worker batches eight independent transfers
before a read barrier and eight publications before a write barrier. Scratch
increases from 4 KiB to 32 KiB per worker. Convolution reads fetch only the two
32-byte face rows actually used, instead of each complete 2048-byte tile.
The arithmetic, selected prefix, checkpoint padding and inactive-slot writes
are unchanged by design; simulator evidence must verify that claim.

## Admission sequence

1. Compare native and candidate publication in TTsim on both chips, T16 prefixes
   0 through 16, with distinct recurrent/convolution histories and inactive slots.
2. Require bit-exact native and checkpoint outputs, zeroed checkpoint padding,
   unchanged entry/history inputs and all seven inactive native slots preserved.
3. Capture and replay changed histories, poison outputs, and check stable binding
   and clean release. Include a 48-layer case to exercise all 96 workers.
4. Only after simulator qualification, compare learned-state hardware publication
   and full accepted-prefix continuation. Keep native publication as control.
5. Run matched full requests with the repeat-confirmed score layout in both arms;
   compare committed TG and the complete selection/publication boundary, not
   kernel timing alone. Reject any token, state, acceptance or quality regression.

The current verifier profile is independent: even removing
publication completely would not make the existing request reach 200 TG.

## Simulator readback bottleneck

A read-only `py-spy` stack inspection located the first-case delay inside
`FDMeshCommandQueue::wait_for_outstanding_reads`, called by the probe's physical
readback of recurrent history on chip 1. The kernel had returned; this is not
evidence of a publication deadlock. The run subsequently advanced through the
first candidate case to the second native input pattern.

An isolated `tensor_bit_compare` diagnostic is being prepared to compare every
physical BF16 word on device and return 32 mismatch counters, instead of repeatedly
reading megabytes through the simulator command queue. It performs direct bit
comparison, not a probabilistic hash. **It is unqualified and is not used by the
publication probe.** Its control probe now passed as described below. Integration
must preserve the complete publication matrices, not sample fewer values.

The direct-readback publication run was deliberately stopped with exit 143 after
six completed per-chip checks and 96 padding checks. The saved report is incomplete
and unqualified. Its last saved stage is `eager_candidate_0_1`; no complete replay
or accepted-prefix matrix is claimed.

Comparator run `20260910T225407Z-408` failed in simulation because its four-byte
counter writes had different source/destination NoC alignment. The fix gives each
worker's scratch counter the same offset as its destination. Aligned run
`20260910T225452Z-384` passed in about 22 seconds, closed cleanly and exited zero.
Independent validation accepts all 28 controls: both chips, one and 64 tiles,
known bit flips, physical-padding differences, unused workers, changed-input
replay, unchanged physical inputs and complete poisoned-counter replacement.

Evidence is retained as `scripts/ci/tensor-bit-compare-simulator.json` with its
exit status. `tensor_bit_compare_gate.py` verifies the exact matrix and source
hashes. This qualifies only the comparator controls, not the publication kernel,
model correctness, hardware performance or serving.

## Comparator-backed full matrix

The publication probe now requires the qualified comparator and retains the same
238 one-layer checks / 2688 48-layer checks, full physical padding checks and
poisoned-checkpoint checks. Each comparison covers every physical word against
an independently generated host reference; only the 32 mismatch counters return
through the simulator command queue. Counter poisoning also prevents stale zero
results from passing. No tensor regions or accepted prefixes were dropped.

Host fixtures are regenerated deterministically per pattern/layer rather than
retaining both complete 48-layer host copies. A single 20-tensor reference set is
shared across sequential comparisons; all device buffers and programs are
prepared before trace capture. The host reference tests cover both chip offsets,
all 17 accepted prefixes and inactive native slots. Seven host/gate tests pass.

Revised one-layer run `20260910T230103Z-389` is active under
`qwen-publication-compared-1-20260911.service`. It remains unqualified until
complete independent validation and clean exit. The 48-layer run follows only
after that succeeds.

The first comparator-integrated run was stopped as incomplete with exit 143
after repeated stack samples remained at the first poison comparison's readback.
An ordering difference from the qualified comparator probe was identified:
the integration lacked its explicit mesh fence before reading counters. The
retry restores that fence and logs enqueue/fence milestones for the first case.
This is a diagnostic hypothesis, not a proven fix or a numerical pass.

## Hardware integration preparation

`gdn_batched_publication_scope.py` provides an unused opt-in preparation hook.
It requires both completed simulator reports before installation, verifies the
declared mesh and all 48 native state owners, routes only T16 to the candidate,
and keeps smaller verifier buckets native. It restores its own binding and refuses
to overwrite an externally replaced hook. Its summary requires all 17 prefixes
but explicitly does not claim hardware execution or serving qualification.
Eight scope/geometry tests pass. No workflow or serving path activates this hook
yet; learned-state continuation and matched full-request audits remain mandatory.
