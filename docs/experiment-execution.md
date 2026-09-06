# Two-P150A execution ledger

Updated 2026-09-06. Target: 200 committed tokens/s for one coding stream, not
aggregate throughput. No adoption or serving restart is authorized by a test pass.

## Verified control

- Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
- TT-Metal: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` with recorded graft changes.
- Plugin: `bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c`; vLLM `0.25.1+empty`.
- Weights/tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, cache discovered
  and mounted directly read-only; full-model startup and K/M engagement verified.
- PCIe x16/x4 host attachment; inter-chip collectives use QSFP-DD/P300 fabric.
- Operator-confirmed exclusive card allocation; no host process-scan claim.
- Runner repository access and tested-ref exceptions remain enabled at operator request.

## Execution queue

### Transfer health isolation (hardware pending)

Diagnostic run 34014926676 (`5cdd443`) timed out before any native oracle or
custom kernel: all three stack dumps identify `host(initial)` / `ttnn.to_torch`
at seed 0 / T1. The 10114-byte artifact's last stage is `fixture-upload`.
This does not reproduce or explain the original post-T1 stall. Residual unhealthy
device/dispatch state after forced termination is a hypothesis, not a confirmed cause.

`device-readback` isolates transfers in the same pinned image and 1x2 fabric mesh.
It uploads BF16 TILE DRAM tensors of shapes `[1,1,32,32]` and `[1,24,128,128]`,
using three distinct exact integer patterns and reading each chip separately.
Twelve exact checks, synchronization and successful mesh close are required.
Durable stages identify the shape/pattern/chip; stack dumps repeat every 30 seconds.
The process is bounded to 180 seconds plus 15-second kill grace. No model imports,
weight loading, trace capture, custom kernels, card resets or serving changes.
Do not retry the GDN experiment until this gate passes or recovery is authorized.

### E4 multi-token norm/gate prerequisite (timed out 34013517498)

Run 34013517498 (`3e0846f`) timed out after seed 0 / T1 passed eager, trace,
all four restored continuations and the stale-state control. The 11311-byte
artifact does not locate the next blocking call; T2 and larger are not certified.
The unused-reader-helper compiler warning is nonfatal. A diagnostic retry preserves
the same kernels, writes durable stage markers before blocking phases, and dumps
Python stacks every 120 seconds. Its test timeout is reduced to 420 seconds plus
a 30-second kill grace. Do not infer a CB5 deadlock or blame trace allocation from
the last printed line alone; native oracle execution also occurs before each candidate.

`gdn-multitoken-norm` extends the token loop with the native fused output math,
including BF16 rounding of norm-times-weight, SiLU and final multiply. CB30/31
retain their native norm-weight/scratch roles. Recurrent feedback moves to CB5:
the reader pushes exactly one complete 16-tile initial-state ring, compute consumes
it, then compute alone produces/consumes full-ring BF16 feedback blocks. No reader
state pushes occur for subsequent tokens. CB storage is 630784 bytes/core.

Each of 24 head-owning cores assembles its T rows into four local output tiles,
zeroing padding and writing full pages to `[1,T,3072]` TILE L1 after the loop.
No cross-core assembly semaphore is needed for this fixed T<=16 geometry. Every
prefix state is still exported to DRAM. The native oracle uses L1 state, so this
first gate measures correctness only, not comparative speed. Required coverage:
30 eager/trace cases, 216 exact restored-prefix continuations, 15 stale-state controls.
Synthetic QKV/beta/g/z and norm weights only; convolution, real model weights,
device rollback integration and full-model performance remain subsequent gates.

### E4 paired recurrence latency (passed 34013199242)

Run [34013199242](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34013199242)
(`2272b91`) passed 30 exact cases, 216 continuations, 15 negative controls and all
1800 timed replays. Across three seeds, T16 serial medians were 0.531716-0.532623 ms
versus 0.339825-0.340785 ms for the device loop (about 36% lower median latency).
All nine T16 paired blocks favored the candidate, but spikes affected both arms:
per-seed median paired ratios ranged 1.169-1.620. Do not hide those spikes or treat
the ratio of marginal medians as the paired estimator. The 14710-byte artifact is
recurrence-only evidence, not a full-model or committed-token speedup.

The `gdn-multitoken` suite now retains the passing correctness matrix and adds
native serial B1 versus device-token-loop captured traces. Both arms export every
token output and every prefix state, with an immutable initial state. Three ABBA
blocks, ten blocking replays per sample, across three seeds and T1/2/4/8/16 yield
1800 timed replays. Each arm is warmed and validated before timing; every timing
sample is followed by exact output/all-prefix/initial-state checks on both chips.
Capture, compilation, input packing/upload, host reads and continuation checks are
outside timing. Report raw samples and paired block ratios; these are fenced host
trace latencies, not device-only kernel time or committed-token/model throughput.
The verified result above is bounded to this synthetic recurrence fixture.

### E4 multi-token recurrent kernel prerequisite (passed 34012883902)

Run [34012883902](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34012883902)
(`855f8de`) compiled and passed on both cards: 30 exact seed/width/eager-trace
cases, 216 exact restored-prefix continuations and 15 detected stale-state negative
controls. The artifact is 12602 bytes. This certifies the bounded recurrence-only
prototype below, not full GDN integration or a performance improvement.

`gdn-multitoken` is the first true device-side token-loop prototype, not another
Python unroll. It derives three kernels from hash-pinned native source without
patching the installed runtime. Each of 24 head-owning cores per chip processes
T1/2/4/8/16 sequential packed rows. The reader loads the initial recurrent state
only for token zero. Compute retains subsequent state in a dedicated 16-tile BF16
L1 feedback ring (32 KiB/core), preserving the native inter-token state rounding;
FP32 math scratch is not silently carried as persistent higher-precision state.
The writer exports every token output and every prefix state to disjoint buffers.
Total configured CB storage is 618496 bytes/core, including feedback. Compilation
and resource compatibility passed for this configuration on the paired cards.

This initial operator gate uses synthetic packed QKV/beta/g inputs and the native
packed recurrence without fused output norm/gate as oracle. It does not include
real weights, convolution, output normalization/gating, attention or full-model
integration. Native arithmetic statements are preserved; strict replacement anchors
change only iteration/state supply/feedback. Independent initial state must remain
unchanged, and every emitted state/output must equal the corresponding serial B1
result on both chips. Reloaded prefix states must produce exact native continuation
outputs and states. Inputs are replicated synthetic fixtures, not coding evaluation.

Required coverage: 30 seed/width/eager-trace cases, 216 restored-prefix continuation
checks and 15 stale-state negative controls. No performance timing or throughput
claim is attached to this prerequisite. Subsequent gates must add native norm/gate,
real-weight conv integration, device checkpoint/rollback integration and paired
full-model timing against run 34011273093 before adoption. Keeping all-prefix DRAM
exports intentionally preserves observability; this prototype removes repeated
state reads but does not yet eliminate checkpoint writes.

### E3 remaining-work attribution (passed 34004475100)

Run 34004475100 (`f4443cf`) passed all four T1/T16 context fixtures, including
three fenced passes, unfenced eager controls and three captured replays each.
All checked full logits, active GDN state and complete valid KV prefixes matched
native serial B1 exactly. The artifact is 167635 bytes, without raw profiler dumps.

| Context | T | Trace median ms | Unfenced eager median ms | Fenced root median ms | GDN native-row exclusive median ms |
| --- | --- | --- | --- | --- | --- |
| 4095 | 1 | 44.997 | 95.125 | 127.052 | 18.613 |
| 4095 | 16 | 235.188 | 670.426 | 783.768 | 421.565 |
| 16383 | 1 | 45.589 | 145.552 | 264.602 | 33.916 |
| 16383 | 16 | 243.955 | 828.717 | 1006.306 | 563.365 |

Each stage median is computed independently across three passes. Fences and host
dispatch substantially inflate and vary these intervals: do not divide the GDN
numbers by trace time or interpret them as device critical-path percentages.
The repeated trace controls corroborate the previous 235/244 ms block costs;
this run does not demonstrate a speed improvement or committed-token throughput.

The pinned native GDN source shows that B1 input with Bmax8 slices recurrent state
and invokes `_write_recurrent_state_prefix` for every token. Consequently
`QWEN_GDN_FUSED_INPLACE=1` does not request in-place recurrence on this path:
the native guard requires B == Bmax. The prefix writer already avoids idle rows
using a sharded active-prefix write, so this is **not** a whole-B8 copy finding.
Conv/gates and recurrence/norm/gate math are already fused; the remaining question
is how much redundant movement surrounds them.

The second bounded attribution revision (passed 34004870686, `28b8dd4`) adds separate coverage for
projected-row slice/clone, fused conv/gates, packed recurrence/norm/gate, active
state slicing and active-prefix writeback. Each must engage exactly 48*T times.
It uses only cloned function namespaces and reversible instance wrappers, preserving
native state geometry, precision, kernels and execution order. An active-B1
working-state/in-place candidate remains a hypothesis, not an implemented speedup;
it will need exact rollback, continuation, idle-slot and trace-address gates.

All four fixtures and twelve fenced passes passed exact logits/state/valid-KV
checks. Each of the five new categories engaged exactly 48*T times in every pass.
The downloaded artifact is 228636 bytes. T16 trace medians remain 235.112 ms at
4095 tokens and 243.885 ms at 16383 tokens; this is instrumentation, not a speedup.

| T16 fenced exclusive category | 4095-token median ms | 16383-token median ms |
| --- | --- | --- |
| Projected-row slice/clone | 87.957 | 90.081 |
| Fused conv/gates | 137.818 | 138.192 |
| Packed recurrence/norm/gate | 97.843 | 97.943 |
| Active recurrent-state slice | 31.583 | 31.730 |
| Active recurrent-state writeback | 62.328 | 62.268 |

These independent medians include fences/dispatch and must not be subtracted from
trace time to predict savings. Fenced root medians were 732.041/734.234 ms versus
unfenced eager medians of 439.437/425.633 ms. The native-row exclusive category now
excludes its five measured children and is not directly comparable with revision 1.

The next implementation candidate keeps an isolated active recurrent working state
through the T-token GDN block, enabling native B1 in-place updates instead of 768
active-state slice/write pairs across 48 layers. Copy into/out of that working
state at bounded block/checkpoint boundaries, preserving native live-buffer addresses
and idle slots. First certify a single-layer candidate with every rollback prefix
and corrected continuation in eager/trace; only then integrate and run paired
full-model timing. Keep projected-row clone removal separate: ownership/aliasing
must be established before deleting it. No arithmetic or precision changes are
justified by this attribution pass, and no serving promotion has been made.

### E3 isolated in-place working state (passed 34005306635)

Run 34005306635 (`7625f31`) passed 216 exact prefix/continuation checks, all 30
stale-state negative controls, all 15 isolation checks, and 279 in-place request
and both-chip buffer-alias checks. This certifies the tested single-layer state
geometry change, not full-model integration. T16 single-sample trace times were
3.750/3.749/3.746 ms across three seeds, including all-prefix staging. These are
diagnostics without a contemporaneous paired control; do not infer a speedup.

`gdn-inplace-timing` (completed 34005485199, `94c5726`) reruns the full correctness gate and
captures a control with the same batched input/output projections and all-prefix
checkpoint semantics, but native B1-in-B8 state handling. Candidate traces include
working-state entry/publication transfers. Each seed/width uses three ABBA blocks,
ten replays per arm sample, restoring identical initial live state outside every
timed replay. Full outputs, final state including idle slots, and every prefix
checkpoint must match the serial oracle after each sample. Fifteen fixtures and
1800 timed replays are required; report paired block-cost ratios, not tok/s.

All 15 timing fixtures and 1800 timed replays completed with exact validation.
The gate again passed 216 prefix/continuation checks, 30 negative controls,
15 isolation checks and 279 in-place request/alias checks. However, the performance
result rejects promotion of this candidate: all 15 fixture median paired ratios
were below 1.0. T16 was consistently slower across every ABBA block and seed.

| T16 seed | Control mean ms | Candidate mean ms | Median paired control/candidate | Paired slowdown |
| --- | --- | --- | --- | --- |
| 0 | 3.607 | 3.764 | 0.956 | 4.61% |
| 1 | 3.591 | 3.753 | 0.955 | 4.69% |
| 2 | 3.564 | 3.758 | 0.944 | 5.89% |

Means summarize six samples per arm; slowdown uses the median of the three paired
ratios, not the ratio of the displayed means. Shorter widths are noisier but none
has a winning median ratio. A green CI run certifies completion/exactness, not a
performance improvement. Keep native B1-in-B8 state handling as the control; do
not integrate this all-prefix working-state variant into the full model.

The precise cause is not isolated. This comparison includes compact entry/exit
transfers and different checkpoint implementations: five generic compact copies
per prefix versus one direct active-slot DMA in the control. Next distinguish
checkpoint overhead from recurrence savings using paired no-checkpoint and
end-prefix-only diagnostic arms, with identical checkpoint semantics per pair.
The end-prefix arm would match the current full-model timing fixture, but neither
arm alone certifies dynamic rollback or a complete speculative commit pipeline.
Retain the full all-prefix correctness gate; do not hide the failed all-prefix
performance result or predict savings from fenced eager attribution.

`gdn-checkpoint-cost` implements that diagnostic (passed 34005748778, `5fac974`). It retains
the all-prefix correctness/rollback/continuation/isolation matrix and reruns the
all-prefix paired control alongside `none` and `end` checkpoint policies. Both
arms use the same policy; compact entry/publication transfers remain timed even
when checkpoints are disabled. Native fused arithmetic and precision are unchanged.
Each warmup/capture asserts the exact checkpoint callback sequence. End-only
staging is overwritten with the initial active state outside every timed replay,
and that poison is required to differ from the expected end state, preventing a
missing checkpoint write from passing on stale results. Every timing sample checks
all outputs and full final live state, plus the selected checkpoints.

The expanded matrix requires 45 seed/width/policy fixtures, 5400 timed replays and
651 eager/capture in-place request/alias checks. Report policy-specific paired
ratios rather than mixing samples or treating reduced-checkpoint diagnostics as
a complete speculative verifier. This can separate checkpoint-related costs from
the combined recurrence/working-state-transfer cost; it does not isolate recurrence
kernel latency alone. No full-model integration or serving promotion is enabled.

Run 34005748778 passed all 45 fixtures / 5400 timed replays, 216 exact prefix and
continuation checks, 30 negative controls, 15 isolation checks and 651 in-place
request/alias checks. T16 paired control/candidate ratios across seeds were:

| Policy | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| All-prefix | 0.942 | 0.942 | 0.942 |
| None | 1.060 | 1.062 | 1.057 |
| End-only | 1.050 | 1.085 | 1.052 |

End-only candidate means were 3.302/3.301/3.299 ms. Every T16 none/end paired block
favored the candidate, while all-prefix remained slower. This supports checkpoint
cost as the factor erasing the benefit in this fixture. Policies run sequentially,
so do not interpret cross-policy differences as a precise isolated kernel cost.
These are layer block-cost ratios, not full-model or committed-token throughput.

### E3 compact checkpoint DMA (passed 34005970668)

Run 34005970668 (`07f2b59`) passed 45 fixtures / 5400 timed replays, 216 exact
prefix/continuation checks, 30 negative controls, 15 isolation checks, 651 in-place
alias checks and 354 compact checkpoint DMA calls. T16 median paired ratios across
seeds were 1.042-1.054x with all-prefix checkpoints, 1.054-1.060x end-only and
1.058-1.062x without checkpoints. The all-prefix median regression is removed,
although one of nine T16 all-prefix paired blocks was slightly below 1.0 (0.996x).
T1 remained slower across all policies and seeds. These are single-layer gains,
not a measured full-model improvement or committed-token rate.

`gdn-checkpoint-dma` changes only compact checkpoint saving from five generic
copies to one mesh generic-op invocation, reusing the existing unchanged 48-core
data-movement kernel. A new strict compact-to-compact mode validates both ends as
one recurrent state plus four conv taps, BF16 interleaved DRAM tiles, with ten
non-aliased buffers per chip. Existing active-slot transfers remain a separate
full-to-compact/compact-to-full mode. Recurrence copies complete tiles; conv copies
only the two logical-row face segments, not padded rows. No arithmetic or native
fused kernel changes are made; padding is not promoted to logical model state.

The hardware gate repeats all three checkpoint policies, the full exactness matrix
and 5400 paired replays, with 354 compact checkpoint DMA calls required during
warmup/capture/eager (trace replay does not increment Python counters). It also
records the unchanged C++ kernel hash and new Python adapter hash. The old generic
copy path remains selectable; no speedup or full-model integration is assumed.

### E3 full-model compact GDN integration (passed 34006233354)

Run 34006233354 (`1be2572`) passed 20 width/mode full-model checks, 16 rollback
cases with corrected continuations, four stale-GDN/wrong-page negative-control
pairs, and ten exact timing fixtures with 2400 timed decode replays. Full active
logits, active GDN state, selected end checkpoints and complete valid KV prefixes
matched the native serial oracle. Both 4K/16K contexts passed eager and trace gates.

| Context | T | Batched native-state control ms | Compact candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 1 | 45.025 | 45.023 | 1.000x |
| 4095 | 2 | 58.764 | 57.930 | 1.014x |
| 4095 | 4 | 84.246 | 82.031 | 1.027x |
| 4095 | 8 | 134.541 | 130.705 | 1.029x |
| 4095 | 16 | 235.197 | 227.075 | 1.036x |
| 16383 | 1 | 45.629 | 45.594 | 1.001x |
| 16383 | 2 | 59.900 | 59.065 | 1.015x |
| 16383 | 4 | 86.480 | 84.227 | 1.026x |
| 16383 | 8 | 138.942 | 135.106 | 1.028x |
| 16383 | 16 | 243.981 | 235.797 | 1.035x |

Times are means over six samples per arm; ratios are medians of three paired
ABBA blocks. Every T>1 paired block favored compact state. T1 explicitly did not
enable compact state; tiny timing differences there are not optimization gains.
T16 saves about 8.1-8.2 ms per full-logit block against the existing batched control.
This is a measured full-model verification gain, not committed-token throughput.
With perfect acceptance and zero drafting/selection/commit overhead, 16 divided
by these block times gives only 70.46/67.85 tok/s for this T16 implementation.
Reaching 200 requires blocks below 80 ms even before that overhead; this improvement
does not close the remaining gap or establish a hardware-wide speed ceiling.

Retain the opt-in exact candidate for subsequent experiments; do not change serving
defaults. Next investigate projected-row preparation/copy overhead and remaining
per-token work using the new candidate as a contemporaneous control. Keep attention
numerics and GDN precision unchanged, and require full-model paired gains rather
than extrapolating layer timing or fenced attribution.

### E3 redundant GDN input preparation (passed 34007693367)

Run 34007693367 (`8b6f429`) passed 20 width/mode exact checks, 16 rollback cases,
four negative-control pairs and ten timing fixtures with 2400 timed decode replays.
The contemporaneous control already uses compact GDN state and checkpoint DMA.

| Context | T | Compact control ms | Reused-input candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 4 | 82.033 | 77.671 | 1.056x |
| 4095 | 8 | 130.708 | 118.293 | 1.105x |
| 4095 | 16 | 227.118 | 201.741 | 1.126x |
| 16383 | 4 | 84.396 | 80.090 | 1.053x |
| 16383 | 8 | 135.122 | 122.758 | 1.101x |
| 16383 | 16 | 235.884 | 210.584 | 1.120x |

Every T16 paired block favored input reuse. T2 showed no meaningful median gain;
T1 remained unchanged. T16 saves about 25.3 ms versus the compact-state control.
These full-model verification gains preserve exact logits, state, valid KV and
corrected rollback but do not include drafting or a complete commit pipeline.
Idealized perfect-acceptance/zero-overhead T16 bounds are now 79.31/75.98 tok/s,
still well short of 200. No serving defaults are changed.

Next, inspect pinned slice/copy and tensor ownership code before attempting to
remove the projected-row clone. The `gdn-source` exporter now includes bounded
source-only ownership roots (maximum 200 files / 4 MiB additional source), recording
missing optional roots rather than assuming them. This audit uses no cards and
does not execute model code. The existing clone path remains unchanged until its
aliasing and deallocation behavior can be established from the actual image.

### E3 projected-row layout hoisting (passed 34011273093)

Run 34011273093 (`6ded69e`) passed 20 width/mode checks, 16 corrected rollback
cases, four negative-control pairs and ten exact timing fixtures / 2400 timed
decode replays. Full active logits, GDN state, selected end checkpoints and valid
KV matched the native serial oracle. T1 did not enable the optimization.

| Context | T | Selective-clone control ms | Hoisted-layout candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 57.554 | 56.142 | 1.026x |
| 4095 | 4 | 76.834 | 72.868 | 1.054x |
| 4095 | 8 | 116.988 | 106.337 | 1.100x |
| 4095 | 16 | 197.494 | 173.065 | 1.141x |
| 16383 | 2 | 58.635 | 57.281 | 1.024x |
| 16383 | 4 | 79.062 | 75.086 | 1.053x |
| 16383 | 8 | 121.388 | 110.716 | 1.096x |
| 16383 | 16 | 206.296 | 181.846 | 1.135x |

Every T>1 paired block favored layout hoisting. T16 saves about 24.4 ms against
the contemporaneous selective-clone control. This is a full-model verification
gain, not committed-token throughput. Perfect acceptance with zero drafting/commit
overhead gives only 92.45/87.99 tok/s at T16; the 200 goal remains unmet.
The bounded layout arm is complete. Retain the opt-in candidate as the next control
and prioritize E4's true multi-token state-retaining recurrence. No serving defaults
are changed and no new hardware experiment is dispatched by recording this result.

`full-gdn-row-layout` converts the packed projection to row-major once per T>1
block rather than allowing each nonzero TILE row slice to repeat that conversion.
It slices B1 row-major tensors and tiles each output separately. Both-chip ownership
guards require independent converted/source/sliced/output buffers before releasing
intermediates. The first-row clone remains; T1 and default behavior are unchanged.
Conversion setup failures release the original projection without changing the hook.

The direct paired control retains compact state, checkpoint DMA, input reuse and
selective clone removal from 34009858516. Layout conversion adds live scratch and
changes padding preparation, so savings and correctness must not be assumed.
The full 4K/16K exactness/rollback gate precedes 2400 timed decode replays. Once this
bounded arm is resolved, prioritize E4's multi-token state-retaining recurrent kernel.

### E3 selective projected-row clone removal (passed 34009858516)

Run 34009858516 (`215b933`) passed 20 width/mode checks, 16 corrected rollback
cases, four negative-control pairs and ten exact timing fixtures / 2400 timed
decode replays. Slice ownership guards and per-layer clone-skip engagement passed.
The paired control already uses compact state, checkpoint DMA and input reuse.

| Context | T | Reused-input control ms | Selective-clone candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 57.944 | 57.508 | 1.008x |
| 4095 | 4 | 77.665 | 76.868 | 1.010x |
| 4095 | 8 | 118.283 | 116.995 | 1.011x |
| 4095 | 16 | 201.794 | 197.479 | 1.022x |
| 16383 | 2 | 59.048 | 58.648 | 1.007x |
| 16383 | 4 | 79.885 | 79.052 | 1.010x |
| 16383 | 8 | 122.857 | 121.524 | 1.011x |
| 16383 | 16 | 210.808 | 206.547 | 1.020x |

Every T>1 paired block favored clone removal. T1 explicitly kept cloning and
native state handling; its small timing variation is not a gain. T16 saves another
4.3 ms per full-model verification block. Full active logits/GDN state/end
checkpoints/valid KV and corrected continuations remain exact. The opt-in candidate
is suitable as the next experiment control, not an automatic serving promotion.
Idealized perfect-acceptance/zero-overhead T16 bounds are 81.02/77.46 tok/s, still
far from 200 committed tok/s. Current evidence does not justify extrapolating to
a pure kernel speedup or removing the first-row alias-protection clone.

Ownership audit 34009341359 passed and exported the pinned slice implementation.
The no-op path can return input storage; nonzero row starts within these T<=16
tiles take the row-major slicing path, whose device op allocates a fresh output
when no preallocated output is supplied. Native GDN consumes/deallocates the
projected row, so the first-row clone is retained conservatively, including T1.

`full-gdn-row-clones` skips only clones after row zero. The exact slice frontend
and device-op source hashes are required before model startup; every skipped clone
also requires distinct source/destination buffer addresses on both chips. A
failure aborts rather than silently falling back or reporting a fake optimization.
Each active compact layer must skip exactly T-1 clones per eager/capture invocation,
removing 720 clone calls at T16. Native slice, fused math and precision are unchanged.

The paired control is the current compact/DMA/reused-input candidate, not the
older native-state baseline. Full-logit/GDN/valid-KV/rollback exactness at 4K/16K
must pass before 2400 timed replays. T1/default paths stay unchanged. Ownership
source evidence and local tests do not substitute for hardware exactness, trace
replay or a measured speedup; no serving promotion is enabled.

`full-gdn-input-reuse` isolates input-row preparation against the exact full-model
compact/DMA control from 34006233354. Batched projection already consumes the full
input and the projected-row hook supplies each token's distinct projected data.
Pinned native decode otherwise reads its input only for shape/reshape preparation.
The candidate therefore prepares one B1 input row per layer and reuses that
shape-only argument across T native row calls, instead of preparing T input slices.
At T16 this removes 720 input-slice calls across 48 layers; it does not remove
projected-row slicing/cloning or change their ownership. A source AST guard rejects
new input consumers outside shape/projection preparation, in addition to pinned
source hashes and existing exact projection-hook engagement checks. One owned
input tensor is released once, not once per repeated reference. T1 stays unchanged.

The gate repeats full-logit/GDN/valid-KV exactness and corrected rollback at 4K/16K
before 2400 timed replays. Its direct paired control now also enables compact state
and checkpoint DMA: only distinct versus reused input rows differ. Reports identify
that control explicitly; do not mislabel this as another native-state comparison.
Both simultaneously captured arms allocate compact working sets (192 MiB/chip
combined at T>1, separate from shared snapshots). No speedup or serving-default
change is assumed before hardware evidence.

`full-compact-gdn` opts the static verifier into compact state plus direct compact
checkpoint DMA for T2/4/8/16 across all 48 GDN layers. T1 and the default path keep
native B1-in-B8 state handling. B1 SDPA reads, projection batching, model precision
and live native buffer ownership are unchanged. Every eager/capture invocation
requires all 48 compact layers to perform exactly T in-place updates and one
selected-prefix checkpoint. Working sets are fixture-owned, preallocated before
capture and released only after their traces; they add 96 MiB per chip at T>1.

The gate repeats 4095/16383-token full-logit/state/valid-KV exactness at all widths
in eager/trace and T16 rollback prefixes 0/1/8/16 with corrected continuations and
negative controls. Only after that matrix passes, timing captures native serial,
the existing batched native-state control, and the compact candidate. Both batched
arms include one end-prefix checkpoint, checked against native serial state after
each sample. Restore remains outside each timed replay; no prefill occurs while
candidate trace buffers are live. Three ABBA blocks compare compact versus the
existing batched control in addition to the serial-versus-batch comparison.
Ten fixtures require 2400 timed decode replays across the two comparisons, plus
separate restore measurements. Do not extrapolate layer gains or call verifier
block throughput committed tok/s. No serving promotion or default change is made.

`gdn-inplace` runs the existing real-weight single-layer matrix with an opt-in
compact B1 working set (recurrence and all four conv taps), copied from native B8
slot zero once per block. Instance-local B/state bindings enable the native
in-place guard during the sequential fused rows. A cloned function namespace
requires the in-place request and verifies returned recurrence buffer addresses on
both chips at every eager/capture call. No installed source or math is changed.
All-prefix checkpoints copy the compact state directly; final publication uses
the already-certified active-slot DMA and restores native B8 Python bindings even
on failure. Working tensors are preallocated and remain stable across trace replay.

The gate requires 216 exact prefix/continuation checks over three seeds and
T1/2/4/8/16 in eager/trace, 30 stale-state negative controls, full-state equality
immediately after each candidate block (including idle slots), stable live
addresses, and 279 successful in-place wrapper calls across warmup/capture/eager.
Trace replays execute captured device work, not Python counters. The existing
15 active-restore isolation cases remain enabled. This changes working-state
geometry, so local tests alone do not certify native-kernel compatibility or
numerical equivalence. Reported single-sample layer times are diagnostic only;
paired measurements above establish a performance regression. Full-model
integration remains withheld pending a genuinely faster exact candidate.

`full-batch-attribution` targets the remaining 235-244 ms T16 path with a bounded,
JSON-only in-situ profiler, avoiding the historic multi-GB raw profiler exports.
It compares T1/T16 at 4095/16383 tokens, three eager profiling passes each, with
separate unfenced eager and captured trace controls. Exact full logits, active GDN
states and the complete valid KV prefix are checked against native serial B1.

Instance-local wrappers distinguish GDN input/output projections, each native
fused row, selected-prefix checkpoint, attention projections/KV/SDPA with row
packing, decoder direct-call components, final norm and LM head. Nested exclusive
times reconcile to the full-model root; call-count gates require all 64 layers,
48*T GDN rows and all attention paths. The profiler synchronizes the mesh at each
boundary, so it changes overlap and includes dispatch/CPU overhead. Report fence
calibration and unfenced/trace controls; do not treat category sums as a decomposition
of the earlier traced 235-244 ms critical path or as a throughput improvement.

### E3 coding-context verification cost (passed 34002876975)

`full-coding-cost` extends exactness to 4095/16383-token deterministic repeated-code
fixtures, crossing the 4K/16K boundaries during verification. It checks all T widths
in eager/trace and rollback prefixes 0/1/8/16 with corrected continuations. Valid KV
hashing now covers the whole request prefix in bounded 64-page chunks across both
chips and all attention layers; no raw KV/logit arrays are exported. This is a
synthetic context-length probe, not a coding-quality benchmark.

Only after the full correctness matrix passes, compare captured native serial B1
and batched full-logit blocks using three ABBA blocks, ten replays per sample. Every
replay restores the initial GDN state outside timing; captured candidate execution
includes one preselected end-prefix checkpoint. Full logits/state/KV are rechecked
before timing and after each sample. Separately measure captured active-state
restore. These measurements exclude drafter generation, dynamic acceptance,
all-prefix checkpoint staging/selection and a complete speculative commit pipeline.
Report block milliseconds and paired ratios, not committed-token throughput.

Attempt 34002210390 (`c31021a`) found a real long-context divergence: at length
4095, eager T=1 and T=2 passed exact logits/state/KV, but T=4 differed in 921936
logit values per chip (maximum absolute difference 0.59765625). Timing was not
entered. The previous short-context pass therefore cannot justify long-context use.

The next controlled arm changes only the SDPA read in the candidate: keep batched
QKV preparation/output projection and serial cache writes, but run native B1 SDPA
for each query before concatenating results. B1 query geometry preserves the native
per-query grid/reduction path, unlike a larger batch that shares cores among rows.
This is an isolation hypothesis until hardware passes; the exactness gate is not
relaxed. `full-coding-cost` now explicitly selects `--serial-sdpa`; other suites
retain their original batching policy.

Retry 34002555664 (`c74b2be`) passed all five 4K eager widths, with exact logits,
active GDN states and complete valid KV prefixes, plus the four eager rollback
cases. This controlled result isolates the earlier eager drift to the batched SDPA
read path. The first T1 traced candidate then produced non-finite logits after the
harness replayed long prefill between candidate capture and execution. This is a
separate trace-lifetime problem, not a reason to relax arithmetic comparison.

The next retry saves candidate-entry GDN state in a fourth preallocated compact set
and restores it in-place after capture, without replaying prefill while candidate
trace buffers are live. KV writes touch only candidate/future positions, which the
position mask excludes until rewritten; exact KV checks remain mandatory. Timing
also prefills before allocating candidate fixtures. Snapshot allocation is now
384 MiB/chip. Structural regression tests guard both trace/prefill boundaries.

Run 34002876975 (`6937529`) passed the complete selected matrix: 20 width/mode
checks, 16 rollback cases, four native baseline mode checks and four negative-control
pairs. All full active-token logits, active GDN states and entire valid KV prefixes
matched exactly on both chips. Ten timing fixtures passed all pre/post checks,
covering 30 ABBA blocks and 1200 individually state-reset timed replays.

| Context | T | Serial block mean (ms) | Candidate block mean (ms) | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 1 | 43.799 | 44.985 | 0.973x |
| 4095 | 2 | 87.650 | 58.716 | 1.493x |
| 4095 | 4 | 175.233 | 84.217 | 2.081x |
| 4095 | 8 | 350.618 | 134.474 | 2.607x |
| 4095 | 16 | 701.383 | 235.098 | 2.983x |
| 16383 | 1 | 44.349 | 45.555 | 0.974x |
| 16383 | 2 | 88.756 | 59.833 | 1.483x |
| 16383 | 4 | 177.423 | 86.422 | 2.053x |
| 16383 | 8 | 354.974 | 138.879 | 2.556x |
| 16383 | 16 | 710.095 | 243.876 | 2.912x |

Separate active-state restore averaged 0.866-0.867 ms. Candidate timing includes
batched projections, sequential native fused GDN recurrence, B1 SDPA reads, ordered
KV writes, native MLP/norms/LM head and one preselected end checkpoint. T1 is slower
than native serial, so this wrapper is not a replacement for the normal B1 path.

**200 tok/s assessment:** the optimistic `16 / block_seconds` bound is only
68.06 tokens/s at 4K and 65.61 at 16K for this implementation, assuming perfect
acceptance and zero drafter/selection/commit overhead. These are derived ceilings,
not measured committed throughput or hardware-wide limits. A 200-token/s T16 path
would need at most 80 ms per block before overhead: another 2.94-3.05x reduction in
the measured candidate cost. The previous 8.45x result was one short-context
attention layer, not this full-model/long-context execution policy.

Next prioritize attribution of remaining per-row GDN and B1 attention work and
evaluate a time-fused GDN block or B1-equivalent parallel attention kernel against
this exact oracle. A neural drafter alone cannot overcome this target-path bound.
All-prefix checkpoint staging, dynamic acceptance and true commit costs still need
implementation; no serving promotion is enabled. Local c31021a passed 113 CI/helper
and 26 drafting tests; subsequent affected helper/trace-boundary tests passed on
Windows Python 3.11 after WSL startup timed out (no VM/service reset attempted).

### E3 full-model batch integration (passed 34001403540)

The `full-batch` suite reuses the serial oracle without changing its default path.
The candidate invokes the native 64-layer `_forward_decode` with instance-local
GDN and attention adapters. It retains native normalization, residuals, MLP and LM
head. Each GDN layer batches input/output projection, runs the pinned fused native
recurrence per token, and saves its own active state at the requested prefix;
attention batches arithmetic with ordered B1 shared-page writes. A third reusable
all-layer compact snapshot set holds candidate checkpoints separately from reference
and scratch, so the reference snapshot cannot accidentally repair the candidate.

Gate: 30 full-logit/state/KV comparisons (T=1/2/4/8/16, eager/trace, prompt
lengths 63/64/65), then all 102 T=16 prefix rollback cases and existing negative
controls. All candidate shapes are compiled before parking native prefill/decode
traces. No installed-source patch, serving eligibility, drafter or speed claim is introduced.

First attempt 34001190409 (`7423438`) compiled all candidate widths and passed the
first native eager baseline comparison, but stopped before batched comparison:
the serial Generator API returned Bmax8-padded host logits, not a single row.
The comparator now explicitly accepts native one/eight-row API geometry and selects
the active row, with padding-canary and invalid-geometry regression tests. No model
arithmetic was changed and this harness failure is not evidence of numerical drift.

Retry 34001403540 (`028c23f`) passed in 8m11s:

- All 30 batched-width cases: full active-token vocabulary logits on both chips,
  active states across all 48 GDN layers, and valid KV prefixes across all 16 full
  attention layers exactly match native sequential B1.
- All 102 T=16 rollback cases: every prefix 0..16, prompt lengths 63/64/65,
  eager/trace; candidate-owned per-layer checkpoints restore the reference state,
  and two corrected continuation steps preserve exact full logits/state/KV.
- Six native eager/trace baseline comparisons and six pairs of stale-GDN/wrong-page
  negative controls passed. Rejected future rows do not change accepted-prefix
  logits in the tested cases.
- Three reusable compact snapshot sets consume 301989888 bytes per chip (288 MiB).
  Candidate checkpoints are separate from reference and scratch buffers.

Local validation: 108 CI/helper tests, 26 drafting tests, shell syntax and diff
checks passed. This closes the first integrated **short-context correctness** gate;
it does not measure full-model block latency, prove long-context equivalence, or
enable a serving/drafter integration. Next measure paired T=1/2/4/8/16 verification
and commit cost at coding-length contexts, repeating exactness gates there first.

### E3 attention shared-page gate (passed 34000512864)

Source-only run 34000023393 exported the actual K-image fused attention source
(`attention/tp.py`, SHA256 `e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a`).
Its decode batch means independent users, with one KV writer core per user.
The paged update validator explicitly rejects `share_cache`. This is a shared-page
write hazard to avoid, not evidence that production B1 cache writes are broken.

The new `attention-batch` suite batches native fused QKV preparation, attention and
output projection while replacing only the two paged writes with ordered B1 calls
through an instance-local tail function. Installed source and global TTNN functions
are untouched. Two seeds, T=1/2/4/8/16 and start positions 31/63/65 cover tile/page
boundaries with nonzero BF8 caches and a nonidentity shared page map. Eager and trace
outputs plus the entire physical K/V pool must match native sequential B1 exactly
on both chips; omitted writes and wrong-page writes must be detected. Static host
position/page fixtures are not a device-dynamic verifier or serving integration.
This gate deliberately makes no throughput claim; numerical drift blocks promotion.

Hardware run 34000512864 (`fbfefa4`) passed all 60 checks, with zero unequal
output/KV values on either chip, and 30 pairs of omitted-write/wrong-page controls.
This validates one real-weight attention layer (layer 3), not all 16 attention
layers or a full-model batched verifier. `attention-timing` adds paired captured
serial/batched layer timing: three ABBA blocks, 30 replays per sample, with exact
output and complete KV revalidation before and after measurement. No device-dynamic
page/position integration has been implemented.

Timing run 34000694699 (`9d9c6ec`) passed the same 60 correctness checks and 30
negative-control pairs, plus all pre/post timing checks and 90 paired ABBA blocks.
For each T, 18 blocks cover two seeds and three start positions, with three repeats
of the ABBA sequence per fixture and 30 trace replays per sample:

| T | Serial block mean (ms) | Batched block mean (ms) | Median speedup | Speedup range |
| --- | --- | --- | --- | --- |
| 1 | 0.2875 | 0.2826 | 1.017x | 1.015-1.018x |
| 2 | 0.5610 | 0.3071 | 1.826x | 1.822-1.830x |
| 4 | 1.1106 | 0.3388 | 3.278x | 3.272-3.285x |
| 8 | 2.2090 | 0.3985 | 5.544x | 5.530-5.552x |
| 16 | 4.4063 | 0.5218 | 8.451x | 8.425-8.460x |

These are warmed host-wall timings of captured **one-layer** token blocks, not
per-token full-model latency. Candidate timing includes the serialized shared-page
KV writes; reset and host validation are excluded from both arms. Positions are
31-80, not coding-length contexts. Neither this gain nor T=1's small difference
establishes serving throughput. Next combine this attention path with the exact
GDN batching/rollback path in a full-model verifier, then measure verification and
commit cost at realistic contexts before introducing a drafter.

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Benchmarked: run 33943034757 | 19.45 to 18.39 client-estimated tok/s at B=1; no-logprobs and engine timing still required |
| E1 cache | Occupancy observed, no active-zero recurrence | 0-12.4893% occupancy; full-model cache lifecycle gate remains required |
| E2 interleaving | Boundary and three-request mixed-traffic gates passed at all ratios | Repeated workload/long-context/load/cancellation sweeps; full device KV lifecycle remains unverified |
| E3 verifier | Exact full-model layout-hoisted candidate passed 34011273093; T16 costs 173.065/181.846 ms at 4K/16K | Prioritize multi-token GDN; reach <80 ms before drafting overhead and implement dynamic selection/commit |
| E4 fusion/pipeline | State/copy changes now win in full-model paired tests; earlier 91-core MLP fusion missed repeatability | True multi-token state-retaining recurrent kernel and memory-pipeline experiments; no serving promotion |
| E5 drafting | Lookup policy host-tested; MTP historical integration; DFlash2/DSpark checkpoint/config review only, not card-tested | Shared E3 verifier/rollback gate, then matched MTP/lookup/DFlash2/DSpark runs; no non-greedy semantic substitution |
| E6 coding quality | Corpus not frozen | 200 independent executable fixtures, isolated code execution and paired outcomes |
| E7 prefix reuse | Dependency-gated | E1/E2 lifecycle gate; hybrid KV plus recurrent/conv state identity and isolation |
| E8 precomputation | Planned audit | Bound removable cost before table prototypes; no unapproved arithmetic changes |
| E9 spare-core work | Layer/full-model attribution, direct DMA and guarded force-argmax measured | Persistent L1 state, dedicated staging and broader mappings; fenced attribution is not traced critical-path timing |
| E10 disaggregation | Eight-slot TP1 pool fails analytical capacity bound | Smaller-pool TP1 and hybrid-state handoff remain unverified; E2 mixed-traffic control first |

The queue is not a claim that all tracks already have runnable implementations.
Run dependent arms only after their entry gates pass. Preserve failed and negative
results; do not convert skipped or blocked experiments into successful test counts.

## Baseline protocol

Manual workflow suite `baseline`, `cards_allocated=true`. Network-disabled disposable
container, loopback-only endpoint, read-only existing weight cache, separate labeled
experimental weight/kernel cache volume, unchanged K-image flags and host sampling.
No production volumes are written. Only the container created by the job is removed.

Warm every workload, then three alternating-order repetitions at concurrency 1 with
target prompt lengths 128/4096/32768/65536, capped below the 65536 context limit to
reserve 1024 output tokens. Actual templated lengths are recorded. Concurrency 2 and
8 at 4096 tokens are separate aggregate controls. Ignore-EOS is verified by output
counts, exact streamed token IDs cross-checked against usage, metrics scraped every 250 ms.
Client timing is labeled an estimate; no engine-token timestamps means this cannot
certify 200 committed tokens/s. Historical run 33943034757 used logprobs and token
strings; new runs request exact token IDs without logprobs by default. The old
results remain valid only for that original host-sampling workload.
Coding quality and greedy token-ID equivalence require their separate gates.

Initial full-model run 33942471680 reached readiness and logged K/M engagement,
but no throughput requests ran: Transformers returned a BatchEncoding rather than
a token list. The client now normalizes the actual input_ids and has regression
tests for both return shapes. Run 33943034757 retries the baseline with the frozen
snapshot and unchanged model flags. Do not count the failed warm-up as a benchmark.

The [payload feasibility estimate](decode-payload-bound.md) gives approximately
19.92 GB of dense projection value payload per ordinary decode step under current
mixed bfloat4/bfloat8 layouts. It is not measured traffic, but establishes the
approximately 3.98 TB/s payload requirement of a non-amortised 200-step/s path.

## Layer profile gate

Run 33944471561 passed GDN and MLP numerical checks with two-chip device timing
and no dropped markers. Some operations use all 110 available workers; this is
not a uniform 36-core workload. These eager fixture timings are not full-model
traced decode timings. Attention failed before completing its numerical check:
the stock fixture allocated BF16 KV while the shipped BF8 flag casts fill inputs
to BF8. The retry stages only the fixture cache allocation dtype to match that
flag, recording original/staged hashes. Model precision and PCC thresholds are
unchanged. The failed attention profile is not accepted as performance evidence.

Retry [33945221856](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33945221856)
passed all three fixtures on both chips with no dropped markers. Attention
paged-versus-concat PCC: prefill 0.999696, decode 0.999757. GDN position-zero
PCC 0.99985; MLP PCC 0.999230. No thresholds were relaxed.

Device-0 eager attribution: the GDN fixture's two steps used 333.358 us in four
matmuls (32/43 cores), 93.131 us in two recurrent kernels (24 cores), and 70.693 us
in two fused conv/gate kernels (81 cores). The MLP fixture's single step used
303.913 us in three matmuls (32/39 cores). Some copy/elementwise operations use
110 cores. Attention includes both prefill and decode, reference and paged paths;
its aggregate operation totals must not be called a single decode latency.

Local validation at commit 35a0d3f: 84 host tests passed (31 CI, 38 continuation/
interleaving, four stream rig, 11 lookup policy). These are not 84 silicon tests.
Interleaving run 33945305689 passed the control's eight boundary prompts and
post-cancellation output-ID check without requesting logprobs. The ratio-1 arm
failed before model execution because vLLM offline mode rewrote the canonical
model ID to the pinned local snapshot path. The guard now accepts only that exact
reviewed snapshot in addition to the canonical model ID; arbitrary model paths
and other revisions remain rejected. Ratio 2/4 did not execute. Historical
zero-cache reports remain observation-only unless current measurements reproduce them.

Retry 33945632554 again passed the control, then failed during ratio-1 API startup:
`Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger than
max_num_batched_tokens (2048)`. This is multimodal admission/encoder budgeting,
not a KV-usage or kernel failure. Interleaving v1 is deliberately text-only but
the launcher had left image/video limits at their defaults. All interleaving
arms, including the control, now set image/video limits to zero and disable MM
embedding inputs. The 2048-token chunk budget and numerical gates are unchanged.
The prior baseline/profile suites retain their original settings.

Run 33945913888 confirmed the encoder-budget fix: the ratio-1 endpoint loaded and
became ready. Its first text request then hit `Continuation v1 is text-only`.
The pinned plugin represents an absent per-request image as `pixel_values=[None]`,
not only `None`. The continuation guard now accepts exactly one absent visual
row, while rejecting tensors, populated/nested rows, malformed row counts and
vision tokens. Regression cases cover both acceptance and rejection. This is
still a failing integration result, not a passed interleaving correctness gate.

Run 33946277920 progressed past the visual-row guard and failed on position
validation. The pinned runner builds prefill positions and prompt lengths as
NumPy integer arrays; our validator accepted only Python integers and torch
integer scalars. It now accepts `numbers.Integral`, normalizes to Python int,
and still rejects floating-point, Boolean and array-valued positions. The
integration regression now feeds the actual plugin visual-metadata gatherer
and NumPy position/prompt-length representations through `submit_prefill`.

Run [33946645240](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33946645240)
cleared all three runtime/configuration errors. Control and ratio-1 each completed
eight boundary requests and the post-cancellation check. Cross-arm comparison
correctly failed: the 2049-token prompt differs at generated token index 1
(the second output token). First four control IDs: `[74830,8,198,727]`;
ratio-1 IDs: `[74830,1590,198,262]`. All 32 IDs match for the other seven lengths
(63/64/65/2047/2048/4096/8193) and after cancellation. This is now a genuine
full-model equivalence failure, not a launcher error. Its cause is not yet
established; the matching first output token does not prove equal logits or state.
Do not relax the exact-output gate or adopt interleaving. Ratios 2/4 did not run.
Next diagnostic: repeat the isolated 2049 case and compare recurrent/conv/KV state
and logits around positions 2048/2049 and the first decode step. The previous
raw startup errors remain preserved in their original artifacts.

Diagnostic run 33947341479 localized the failure: for 2049-token prompts, ratio-1
records one 2048-token intermediate prefill, then enters decode at position 2048.
There is **no final GDN slot write**. The 8193-token case likewise has no final
slot write, despite previously matching output tokens by coincidence. The pinned
runner's `_is_still_prefilling` assumes one remaining token means decode. That
assumption is invalid for this prototype's separate B=1 prefill scratch: even a
one-token final prefill must execute and commit the request's GDN state to its
decode slot. The fix uses the existing request/epoch completion ledger only when
continuation is enabled; ordinary decode and the flag-off path keep their original
classification. A regression executes the transformed nested runner classifier
for a one-token tail, completed request, no remaining tokens, and flag-off mode.
Diagnostic reruns also gate three isolated 2049-token repetitions against control,
in addition to the original nine checks. Host fingerprint instrumentation performs
no additional device operations; its timings must not be used as speed evidence.

Fix verification [33947778844](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33947778844)
passed: ratios 1/2/4 each match the control on all 12 exact-token checks (eight
boundary prompts, three isolated 2049 repetitions, and post-cancellation output).
Each arm also has 13 paired diagnostic request records, including the cancelled
request. All host recurrent/conv snapshots supplied to the decode-slot writer
match the control byte-for-byte, as do the recorded final-prefill and first-three
decode logits. No missing final slot writes remain. This is not a complete direct
device KV snapshot or multi-request isolation certification.

Mixed-traffic run 33948332548 separately tests one 512-token decoder, an injected
8193-token prefill and a short coding request. It requires exact token-ID equality
against isolated requests and verifies actual request overlap. Boundary checks
remain enabled in every arm. Diagnostic hashing is disabled for this run, and
KV metrics continue to be collected. No production serving changes or automatic
adoption follow from these tests.

Mixed-traffic [33948332548](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33948332548)
passed all four arms. The live decoder, long request and short request each match
their isolated output IDs exactly; post-run comparison also matches all three
against the ratio-0 isolated control. Actual overlap and short-request arrival
before the long request's first token were verified. One smoke trial per ratio:

| Decode steps/chunk | Live decoder maximum gap (s) | p99 gap (s) | Client decode tok/s | Long TTFT (s) | Short TTFT (s) |
| --- | --- | --- | --- | --- | --- |
| Control, no interleaving | 3.116 | 0.055 | 17.671 | 2.590 | 2.813 |
| 1 | 0.654 | 0.556 | 17.470 | 2.836 | 3.308 |
| 2 | 0.646 | 0.552 | 17.249 | 3.078 | 3.605 |
| 4 | 0.647 | 0.546 | 17.148 | 3.517 | 4.132 |

The maximum interruption falls approximately 79%, but one long stall becomes
multiple chunk-sized interruptions: p99 worsens, and new-request TTFT increases.
This is a responsiveness tradeoff, **not** a decode-throughput improvement or
evidence of 200 committed tok/s. Ratio 1 is the next candidate for repeated
paired measurements, not an adopted default. Cache occupancy was nonzero
(peak 1.60-1.61%), with zero active-zero-cache samples across 1230 active samples
and zero scrape errors. Production serving and default-off experiment flags
remain unchanged.

## E9f: guarded TP2 greedy device sampling

Native sampler prerequisite [33949480633](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33949480633)
passed all 30 checks: real 248320-token vocabulary, random/boundary/tie/near-tie
cases, eager and two trace replays, on both chips. Median synchronized sampler
latency was 2.607 ms over 20 warm iterations. This is a 32-row sampler-only
measurement, not B=1 full-model throughput.

The pinned model disables generic device sampling on TP2 because each vocabulary
shard exceeds its TopK limit. The opt-in experiment enables only force-argmax,
with a runtime guard that fails before any generic TopK execution. Original
sampling eligibility checks remain intact; non-greedy requests, penalties,
multi-request batches and unsupported requests retain host sampling.

The full-model test uses three ABBA blocks per prompt length (approximately 128
and 4096 tokens), B=1 and 512 output tokens. Both arms omit explicit seeds to
permit internal greedy sampler tracing, and omit logprobs in timed requests.
Every output must match exact token IDs and text; device-path engagement is
recorded per request. A device-labelled logprobs request must fall back to host.
This is separate from the historical logprobs-enabled baseline. KV metrics are
collected throughout; experiment flags and production defaults remain unchanged.

Full-model attempt 33950052748 stopped safely during startup: the shared warmup
sweep requested non-greedy TopK. The experiment now passes the existing
`greedy_only=True` warmup option only for its TP2 model. Host/no-sampler warmup
remains included, and the runtime TopK guard is retained.

Full-model [33950377324](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33950377324)
passed at `dd42e2e`. All 24 measured requests completed 512 tokens and matched
their prompt-length control exactly (token IDs and text). All 14 device requests
(12 measured, two warmup) recorded force-argmax engagement with model trace mode
`all`; both logprobs negative controls correctly selected host sampling.

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 109 | 19.850 | 21.317 | +7.389% | +7.334% to +7.455% |
| 4078 | 19.800 | 21.113 | +6.629% | +6.278% to +6.975% |

Means weight the three ABBA blocks equally. These are B=1 client estimates,
not engine-committed timing or evidence of 200 tok/s. Exact equality on these two
coding prompts is not a general coding-quality benchmark. KV occupancy peaked
at 0.8782%; no active-zero occupancy occurred across 2364 active samples out of
2454 scrapes, with zero scrape errors and zero preemptions.

Decision: retain guarded device force-argmax as a promising experimental candidate,
not a production default. Next gates are longer-context paired runs, explicit-seed
and non-greedy/penalty fallback checks on hardware, and full-model traced operation
attribution before changing projection grids or low-level GDN kernels. Multi-request
device sampling and composition with prefill interleaving are not certified here.

The `sampling-extended` suite next runs three ABBA blocks at approximately 32K
and 64K input tokens, still B=1 with 512 output tokens and exact ID/text gates.
Before timing it compares host/device-labelled requests with explicit seeds 123
and 456, seeded non-greedy sampling, each penalty type separately, and a final
greedy recovery request. Non-greedy and penalty cases must actually select host;
seeded greedy and post-fallback greedy must engage device force-argmax. These
checks change request parameters only, not eligibility or sampling mathematics.

Extended [33952227890](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33952227890)
passed at `9ca28d0`, following 47 passing local CI-helper tests. All seven
fallback/seed pairs matched exact IDs and text, with the expected actual sampler
selection; device greedy resumed correctly after penalty requests. Both long-context
logprobs negative controls also selected host and matched greedy control tokens.
All 24 long-context measured requests completed 512 tokens and matched their
prompt-length control exactly across three ABBA blocks:

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 32752 | 19.231 | 20.522 | +6.709% | +6.553% to +6.828% |
| 64504 | 18.724 | 19.968 | +6.642% | +6.464% to +6.831% |

KV occupancy peaked at 12.3918%; zero active-zero observations across 2563 active
samples out of 4581 scrapes, zero scrape errors and zero preemptions. This is
exporter evidence, not direct certification of every attention/GDN state tensor.

The sampling improvement now reproduces at all four tested context lengths.
Explicit-seed checks establish correctness and path selection, not seeded throughput
or internal sampler-trace reuse. Timings still exclude logprobs and use client
stream events, not engine commit timestamps. Production defaults remain unchanged.
The next performance investigation is full-model traced operation attribution:
separate projection/GDN math, collectives, sampling and host dispatch before tuning
core grids. Existing eager module profiles cannot establish that critical path.

The `model-profile` probe loads every model layer through the served Qwen Generator,
retaining max-batch 8, 64K context and the observed 8200-block KV pool. Only the B=1
host-sampling decode bucket is warmed/captured; HTTP, scheduling and other resident
batch traces are excluded. It checks the first 16 tokens against endpoint run
33950377324, flushes device profiler buffers between decode steps, and requires
15 complete trace replay sessions on both chips. Attribution filters by the actual
decode trace ID, excluding eager compilation and prefill. Summed kernel durations
are attribution evidence, not non-overlapping critical-path time or tok/s.

Attempt 33954595293 completed all 16 exact-token checks on the full 64-layer
Generator, but is not a passing profile. Warmup overflowed profiler buffers and
the report fell into device-only Python analysis of an 11.5 GB raw log; it was
cancelled after exceeding its inner timeout. No kernel bottleneck conclusion is
drawn from that attempt. The retry explicitly enables host operation metadata
before Tracy imports, uses partial Python tracing and runtime C++ analysis without
the large raw CSV dump, and adds a hard kill-after timeout. Warmup overflow warnings
are reported separately: any overflow during the explicitly bounded decode region
still fails, as do missing or inconsistent replay rows on either chip.

Retry 33957235279 again passes exact generation and now captures host operation
metadata, but the all-operations report correctly rejects missing warmup data
(operation 11738113, device 1). The next revision scopes the pinned report join to
the recorded decode trace before joining device measurements. Its existing missing
operation assertions remain active for every selected operation; the final checker
also retains both-chip, replay-count, per-replay coverage and measured-overflow gates.
Profiler CSV sidecars are preserved explicitly even on failure, rather than relying
on hidden-directory artifact upload. No model/kernel arithmetic is modified.

Full-model trace [33957724963](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33957724963)
passes at `f62c298`: all 64 layers, exact first-16 endpoint tokens, 15 measured
replays on each chip, and 1433 operations per replay. Native C++ operation counts
also match the joined report by chip, replay and operation name. No marker drops
occur in measured decode; 1100 excluded warmup warnings remain explicitly recorded.
This certifies the selected decode evidence, not the discarded warmup profile.

| Operation group | Chip 0 summed kernel ms/step | Chip 1 summed kernel ms/step |
| --- | --- | --- |
| Matrix multiplications | 32.069 | 32.051 |
| GDN recurrent kernel | 2.256 | 2.255 |
| GDN convolution/gates | 1.699 | 1.699 |
| All-gather | 1.999 | 1.770 |
| Reduce-scatter | 1.413 | 1.606 |
| All operation groups | 42.670 | 42.657 |

Each entry is the median of per-replay summed durations; the last row sums those
group medians. Matmul accounts for approximately 75% of this attribution budget,
GDN recurrence plus convolution/gates for 9.3%, and the two collectives for 8%.
Do not sum chips or equate these sums to non-overlapping critical-path latency.

Largest matrix shapes, chip 0 (chip 1 agrees closely):

| Weight dimensions (padded) | Calls/step | Weight dtype | Cores | Summed kernel ms/step |
| --- | --- | --- | --- | --- |
| 5120 x 8704 | 128 | BFLOAT4_B | 39 | 11.803 |
| 8704 x 5120 | 64 | BFLOAT8_B | 32 | 7.819 |
| 5120 x 8256 (8240 logical) | 48 | BFLOAT8_B | 43 | 5.754 |
| 3072 x 5120 | 64 | BFLOAT8_B | 32 | 2.984 |
| 5120 x 7168 | 16 | BFLOAT8_B | 56 | 1.939 |
| 5120 x 124160 | 1 | BFLOAT8_B | 108 | 1.769 |

Decision: prioritize MLP gate/up and down projection grid/blocking experiments,
then the GDN input projection. Preserve the existing BF4/BF8 weight dtypes and
accumulation fidelity; extra cores are not automatically a bandwidth improvement.
The vocabulary projection already uses 108 cores, disproving a universal 36-core
limit. Next gates are real-shape numerical checks and synchronized traced kernel
A/B timing before any full-model adoption. No speedup beyond the earlier sampling
result is claimed from profiling. The host scheduler and device-sampling path
remain outside this probe. The enormous raw host-times CSV is not preserved in
future runs; final operation reports, native C++ data and host operation metadata
remain available. Local CI-helper validation now includes shape/coverage checks.

The `mlp-sweep` gate loads real layer-0 weights and compares the frozen production
44/33 gate/down grid budgets against separate larger/smaller grids and K-block
sizes 4/16 (control 8). It preserves BF4 gate/up, BF8 down, LoFi, FP32 destination
and packer L1 accumulation. Fifteen configurations run three seeded B=1 vectors
against the existing Torch PCC threshold (0.97) and exact baseline outputs, eager
and traced. All compilation precedes trace capture. Three ABBA blocks per candidate
time 20 replay submissions followed by synchronization per sample. Promotion to
a full-model gate requires exact control equality and over 2% lower latency in
every block; this is only a single-layer screen, with replicated DRAM input and
both-card reduce-scatter, not a serving-throughput or coding-quality benchmark.

Grid/block sweep [33958859849](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33958859849)
passes at `50d63d7`: all 90 eager/traced seeded outputs match control exactly;
Torch PCC ranges 0.999203-0.999417. No candidate clears the predeclared 2% latency
gate in every block. The best, gate grid budget 110, reduces whole-layer latency
only 1.691% (about 0.338 to 0.332 ms). Larger down grids are 0.48-2.43% slower;
K blocks 4 and 16 are 3.96% and 5.72% slower. The frozen control remains unchanged.

Follow-up `mlp-packing` keeps the control grid budgets but separately sets gate/up
or down output tiles per core to 8, 12 or 16, permitting a four-tile FP32 subblock
instead of the control's one-tile subblock. It retains the same precision, seeded
exact-output and paired timing gates. This tests tile packing, not a promise that
more active cores improve bandwidth.

Packing [33959155664](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33959155664)
passes at `86b9d83`, with all 42 seeded eager/traced outputs exactly equal to control.
No candidate meets the promotion gate:

| Change from frozen control | Mean whole-layer latency change |
| --- | --- |
| Gate/up 8 output tiles/core | -0.062% (one block slower) |
| Gate/up 12 output tiles/core | +2.300% |
| Gate/up 16 output tiles/core | +4.204% |
| Down 8 output tiles/core | +1.686% |
| Down 12 output tiles/core | +4.354% |
| Down 16 output tiles/core | +8.817% |

Across the two runs, 20 non-control configurations were screened without changing
weight or accumulation precision. None qualifies for full-model adoption under
the existing rule; the 44/33 control remains the default. These single-layer
results do not prove DRAM saturation, but they do rule out a large gain from the
tested grid/K-block/output-packing changes. They are not a full-model speedup.
Next investigation: combine gate/up work or reduce surrounding memory transfers,
with weight-layout/extra-allocation accounting and exact-output gates before
serving tests. Do not extrapolate a one-layer gain to the 200 tok/s objective.
Local CI-helper suite: 56 tests pass; hardware access and serving remain unchanged.

## GDN upstream parity audit

Read-only source audit [33960203974](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33960203974)
passed. The image includes ctxbot's fused recurrent kernel hardening plus packed
QKV, native norm/gate folding and broadcast rank-one updates. No missing listed
core fusion fix was identified; do not overwrite these extensions with the PR.
See [source comparison and next experiment](gdn-source-audit-2026-09-05.md) for
exact upstream heads, evidence limits, two failed audit attempts and the proposed
active-prefix state-write experiment. B1/Bmax8 currently cannot use the existing
in-place flag. No new throughput result or production change is claimed.

Offline attribution of the existing 33957724963 hardware trace now isolates the
three recurrent-state copy stages across all 48 GDN layers, 15 replays and both
chips. Median summed copy time is 0.548383/0.548504 ms per step on chip 0/1,
about 1.285%/1.286% of summed kernel time (not critical-path time). Native call
coverage, chain adjacency and BF16 state shapes are checked. Active-prefix state
writeback remains a secondary candidate; combined gate/up projection work takes
priority. The analyzer is added to future full-model profiles; eight regression
tests pass. This is new analysis of old measurements, not a new speed benchmark.

## Fused gate/up decode probe

The `mlp-fusion` suite compares the frozen real-weight layer-0 MLP with native
`minimal_matmul(fuse_swiglu=True)` using tile-pair-interleaved gate/up weights.
It retains BF4 gate/up, BF8 down, LoFi FP32 destination and packer L1 accumulation;
fusion can still alter rounding/order, so exact control output remains mandatory
for promotion. Three candidates use M/K blocks 1/8, N blocks 8/16/32 and grid
11x2. The existing minimal matmul partitions M across grid rows, so adding rows
for B1 is not automatically useful N-parallel work. No custom kernel is added.

The timed path includes the original input-to-L1 transfer, unchanged down
projection and TP reduction. All candidates compile before traces; three seeds
are checked eager and traced against Torch PCC >=0.97 and exact control output.
Three ABBA blocks with 20 replays/sample require >2% lower latency in every block
and exact outputs before any full-model gate. A non-exact result is not promoted.
An additional packed layer weight is allocated only inside the experiment;
host packing is checked reversible and no weight cache is written. Full-model
duplicate-weight memory capacity remains unverified. This is not serving adoption.

Attempt 33962274337 passed the three eager control seeds but rejected the first
fused candidate before execution: the operator requires grid dimensions >=2x2.
Retry uses 11x2 rather than 11x1, without relaxing the native validation. No fused
correctness or timing result is claimed from the rejected attempt.

Retry [33962376720](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33962376720)
completed at `460959c`. All three seeds passed Torch PCC >=0.97 eagerly and under
trace for control and every candidate (24 checks total). All 18 candidate
seed/mode comparisons failed bit-exact equality against control. Candidate PCC
was 0.999201477-0.999417543, close to control's 0.999202549-0.999416769;
this is not evidence of coding-quality equivalence or degradation.

| Fused N block | Mean layer latency increase | Three-block range | Approximate candidate ms |
| --- | --- | --- | --- |
| 8 | +70.565% | +70.481% to +70.674% | 0.576 |
| 16 | +68.031% | +67.801% to +68.285% | 0.568 |
| 32 | +57.923% | +57.889% to +57.942% | 0.535 |

Control was approximately 0.338 ms. No candidate qualifies on either exactness
or speed; no full-model arm is warranted for these configurations. Execution
success is not an optimization pass. This does not rule out a decode-specialized
fused kernel: the existing implementation uses a 2D M/N partition instead of the
control's 1D N-parallel map. Its SwiGLU stage multiplies gate/up in destination
registers before the final pack, unlike control's separately rounded BF16 gate
and up intermediates. These source differences are hypotheses for the result,
not experimentally isolated causes.

Next native candidate must preserve the control's 1D N-parallel distribution,
keep complete gate/up tile pairs together, reproduce BF16 intermediate rounding
and account for packed-weight allocation before repeating the same gates. Do not
extend the negative prefill-oriented grid sweep or relax precision/exactness to
claim a win. The 65 local CI-helper tests pass; production remains unchanged.

## Decode-specialized 1D gate/up prototype

`projection-1d` uses a separate `generic_op` program, not the prefill-oriented
minimal matmul. The native 1D compute source is copied into a source-code kernel
descriptor; its final-output branch and a post-loop epilogue change. The original K-block loop and
FP32 packer-L1 partial accumulation remain. No installed source or binary is edited.
Native and generated compute-source SHA256 values are recorded in the result.

The program maps 272 output-tile pairs to the same 39 N-workers (seven pairs/core,
six on the last), within an 11x4 rectangle. An activation-multicast reader includes
the five non-compute receiver cores. Separate weight-reader/output-writer code
streams pair-interleaved BF4 weights, including zero padding on the last worker,
and writes compact BF16 output directly. These readers are new code, not a claim
of byte-identical native reader scheduling. K-block size stays eight; paired
subblocks have two tiles instead of control's single tile.

The epilogue runs native pack-side SiLU on the gate tile only, packs gate and up
to a private fourteen-tile BF16 circular buffer (seven pairs per worker), then multiplies those rounded values
and packs BF16 output. Packed device weights must dequantize bit-identically to
the original gate/up tensors on each chip. This establishes the intended rounding
contract, not a correctness claim before silicon testing.

First gate is projection-only (no down projection or CCL reduction): three real
layer-0 input seeds, both chips, eager and traced exact comparisons, finite/PCC
diagnostics, then three ABBA blocks with 40 trace replays/sample. Inputs already
reside in L1 for both arms. Only exact output and >2% lower latency in every block
permit a whole-MLP experiment. Preserve non-exact/negative results and do not
replace the serving path on the strength of this prerequisite test.

### Initial 1D silicon attempts

All runs below are isolated projection probes, not full-MLP or serving results.
Execution/PCC success is separate from the exact-output promotion gate.

| Run | Outcome |
| --- | --- |
| 33963133460 | Descriptor setup failed: unpack-mode vector constructor not exported; use the exposed property instead. |
| 33963238683 | Descriptor validation required 64 unpack-mode entries rather than 32. |
| 33963346556 | Inline per-pair product interfered with subsequent native matmul state; first comparison PCC 0.95914, stopped before timing. |
| 33963532517 | Deferring product until after all matmul work restored PCC above 0.99988, but LoFi FPU multiply left 8,040–8,121 mismatches per 8,704-element shard. Projection latency increased 83.89–85.39%. |
| 33963794203 | SFPU product JIT failed because its binary API header was missing. |
| 33963923741 | Explicit SFPU API plus waiting for all rounded tiles reduced discrepancies to 52–83 elements per shard, maximum absolute error 0.015625. Still not exact. Control 0.19695–0.19733 ms versus candidate 0.36276–0.36490 ms, 83.83–85.29% slower. Not eligible for whole-MLP testing. |

The next probe isolates rounded gate and up outputs independently before checking
their product. It also corrects the new readers to the native 1D NoC assignments:
input multicast on NoC 1, weight reads on NoC 0. NoC 1 multicast endpoints reverse,
but receiver acknowledgements still target the original sender. Compare only the
ordinary product arm in timing; diagnostic intermediates compile before traces.

Run 33964284985 confirmed all twelve rounded gate/up comparisons exact across
three seeds and both chips. Product mismatches were unchanged. With corrected NoC
routing, candidate projection latency was 0.20523–0.20580 ms versus control
0.19672–0.19693 ms: only 4.33–4.50% slower, rather than approximately 85% slower.
This remains a rejected candidate, but identifies a substantial reader defect.

The native Blackhole `calculate_sfpu_binary_mul` explicitly applies BF16
round-to-nearest-even and zero-input handling when its arithmetic template flag
`is_fp32_dest_acc_en` is false. Calling the ordinary SFPU API inside our FP32
matmul kernel passes true and skips that narrowing. The next candidate keeps
FP32 destination addressing/accumulation for the matmul but calls the same native
multiply calculation with its BF16 narrowing branch enabled. No accumulation
precision is lowered, and the existing exact-output gate remains mandatory.

Run 33964555688 confirmed the correction: every rounded intermediate, eager
product, and traced product comparison was exact on both chips for all three
seeds. Projection latency remained 5.00–5.10% above control (0.20688–0.20733 ms
versus 0.19695–0.19732 ms). Correctness is established only for this bounded gate;
there is still no performance promotion or whole-model quality claim.

The subsequent isolated grid sweep retains this arithmetic, K-block eight,
two-tile subblocks and NoC routing. It compares 39/55/68/91 N-workers using
7/5/4/3 pairs per worker respectively. Only the N partition, corresponding buffers
and multicast receiver rectangle vary. All candidates compile before trace
capture and each receives its own three ABBA timing blocks and exact-output gate.

Run 33964688595 retained exact outputs for every grid but promoted none:
39 cores +13.22–13.30%, 55 cores +96.00–96.54%, 68 cores +11.38–11.70%,
91 cores +14.29–14.56% projection latency versus control. The 39-core arm itself
regressed relative to the preceding single-candidate probe. Generalizing the
reader had inadvertently changed its inner tile-loop bound from compile-time
to runtime; the retry restores compile-time specialization and compact core
ranges before drawing conclusions about the additional cores.

Run 33964808840 restored the 39-core control candidate to +5.07–5.16%. All
four candidates remained exact. The 55-core variant was +85.50–85.57%; the
68-core variant improved 1.94–2.06% but failed the >2% requirement in one block.
The 91-core variant passed the prerequisite: 0.18833–0.18882 ms and 4.07–4.32%
lower projection latency in every block. This is not end-to-end tok/s.

`projection-1d` now conditionally follows its projection probe with the existing
whole-MLP harness. It independently verifies each selected candidate's twelve
eager/traced shard checks and all three timing blocks, then requires an identical
compute manifest. Rejected candidates are not forwarded. Whole-MLP testing keeps
the native BF8 down projection, DRAM input/output and TP2 reduction; three seeds,
Torch PCC checks, exact native-control outputs and paired traces still apply.

Run [33964993645](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33964993645)
repeated the projection gate and completed the conditional whole-MLP test:

| Stage | Result |
| --- | --- |
| 91-core projection | Exact on both chips, eager and traced, all three seeds; 4.24–4.47% lower latency across all three paired blocks. |
| Complete layer-0 MLP, including down projection and TP2 reduction | Exact native-control output for all three seeds, eager and traced; Torch PCC 0.99920–0.99942, unchanged from control. |
| Complete MLP latency | Control 0.33735–0.33777 ms versus fused 0.32892–0.32962 ms; 2.37–2.50% lower latency, mean 2.447%. |
| Promotion | Eligible for the full-model gate only. No serving default changed and no end-to-end tok/s result claimed. |

The 68-core arm again missed the >2% projection requirement in one block and was
not forwarded; 39 and 55 cores remained slower. Local helper validation passed
75 tests, with shell syntax and whitespace checks passing. These results correct
the prior inference from the prefill-oriented 2D kernel: properly routed,
rounding-preserving 1D fusion can improve this decode workload, although the
measured gain is small and does not establish the 200 tok/s target.

Next gate: opt-in full-model integration with pair-packed weights replacing
(not retaining alongside) gate/up allocations, source/precision checks on every
layer, trace-safe B1-only dispatch and native fallback elsewhere. Verify output
tokens and memory headroom at short and long contexts before paired serving
timings. Do not extrapolate the layer microbenchmark to end-to-end speed or
claim coding-quality coverage from three layer-input seeds.

## Full-model fusion gate (2026-09-06)

Source inspection found an important allocation detail: the native TP model already
loads a pair-packed `w_gate_up` tensor for its fused prefill path. The full-model
experiment reuses that existing allocation instead of building another copy.
Native `w1`/`w3` remain available for unmodified fallback, with no weight-memory
increase relative to the current baseline. This replaces the preceding proposed
weight-replacement approach; hardware must verify all 64 layers and both shards.

`full-model-fusion` is an isolated Generator test, not a serving deployment.
It checks pinned compute/reader hashes, BF4 gate/up and BF8 down precision,
dequantized packed-weight equality for every layer/shard, and unchanged weight
addresses. It records per-chip DRAM/L1/trace allocator snapshots. Both arms share
the model and its KV pool but use separate decode trace stores. All decode kernels
compile before trace capture; prefill always uses the native path. Only explicit
B1 decode inputs select the fused path, with capture engagement checked on all
64 layers. No installed source or serving default is patched.

Predeclared work: short (~109 tokens) and long (~64,504 tokens) coding prompts,
16-token eager/traced token and full-logit equality checks, then three ABBA blocks
of 128-token generations per context. Timing includes host argmax and readback,
excludes prefill and HTTP/scheduler overhead, and counts 127 decode steps rather
than counting the prefill-produced first token as decode. Every request resets
state through native prefill. Require exact outputs and >2% lower latency in every
block before marking eligibility for a subsequent serving gate. Preserve failures
and smaller/negative gains; none establishes 200 committed tok/s.

### Completed run 33996306217

[CI evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33996306217),
code `84db0ab`: execution/correctness passed, `eligible_for_serving_gate=false`.
All six eager/traced logit/token checks were exact, as were paired generation
tokens and packed weights on all 64 layers/both chips. Weight addresses were
unchanged and additional weight allocation was zero.

| Context | Control host tok/s, three blocks | Fused host tok/s, three blocks | Decode latency change |
| --- | --- | --- | --- |
| Short (~109 input) | 19.576 / 19.652 / 19.713 | 19.789 / 20.245 / 20.067 | -1.076% / -2.931% / -1.766% |
| Long (~64,504 input) | 18.410 / 18.626 / 18.201 | 18.406 / 18.705 / 18.746 | +0.018% / -0.421% / -2.907% |

Only two of six blocks clear the predeclared >2% latency threshold. The exact
kernel is viable but the full-path improvement is not repeatable enough for
promotion. These are Generator timings with host sampling, not the separate
device-argmax serving arm or engine-committed throughput. No serving change made.

### Multi-drafter groundwork

Added lookup-first routing to one explicitly selected neural callback, with native
target fallback when mode/verifier readiness is unsupported or no proposal exists.
Nine new host tests cover selection, skipped neural work, invalid proposals,
explicit commit, failures and isolation, alongside the existing eleven lookup tests.
Callbacks are test doubles: no DFlash2, DSpark or EAGLE3 device adapter is claimed.
The E5 plan now separates routing, adaptive selection, cascades and tree ensembles.
The shared exact E3 verifier and device state lifecycle remain the next dependency.

## E3a native GDN prefix gate (2026-09-06)

Added the `gdn-prefix` hardware suite. The historical multi-token implementation
used older composed conv/gate/norm operations, so it is not reused as the current
control. The new candidate calls the audited native packed input projection once
for T rows, then supplies independent row copies to the unchanged fused native
decode remainder. Output projections and collectives remain per token in this
first isolation test; it is not a complete batched verifier or optimal kernel.

The gate loads one real GDN layer, retains the pinned K-image BF16 recurrence and
decode settings, and uses B1 within an eight-slot state allocation on both chips.
Nonzero priming precedes three seeds at T=1/2/4/8/16. Both eager and trace replay
must match every native sequential output and recurrent/conv snapshot exactly.
Each prefix 0..T is restored in place, followed by two correction-input steps,
with exact outputs and final state required. Deliberately restoring stale
recurrence or convolution separately must be detected in state and continuation.

Device snapshots are preallocated outside trace capture and all live state
addresses must stay fixed. Timings include snapshot copies and are diagnostic
only. Attention KV, full-model logits, EOS/cancellation/slot epochs and all-layer
verification remain later gates; no host test or layer pass certifies those.

Run [33997659654](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997659654),
code `b4da5c4`, passed all 216 prefix/continuation cases (both chips, three seeds,
five row widths, eager/trace) and all 30 stale recurrence/convolution negative
controls. This establishes input-projection batching with native fused GDN state
evolution for this layer fixture, not a complete verifier.

Next `gdn-block` arm batches the output projection and TP reduction as well. It
extracts the pre-projection native decode body from the SHA-pinned audited source,
with a structural tail check, preserving the internal fused GDN arithmetic. This
is an instance-local experimental function, not an installed source patch.

The host `greedy_verify.py` contract selects the longest exact proposal prefix
against K+1 target argmax rows. Retain state after seed plus accepted proposals;
the target correction/bonus is emitted but remains the next unconsumed input.
Accepted EOS suppresses later rows; correction EOS terminates without consuming
it. Six tests cover all rejection positions, bonus/off-by-one cases and EOS.
This selector neither calculates target predictions nor commits device state.

Run [33997961604](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997961604),
code `82b9e3b`, passed the output-batched arm: all 216 exact prefix/continuation
cases and all 30 stale-state negative controls again passed. The layer now reads
input and output projection weights once per T-row block and performs one TP
output reduction; convolution/recurrence/norm-gate still use the native fused
per-token path, not a new time-fused recurrence kernel.

Single synchronized traced diagnostic observations, three seeds:

| T | Input-batched only, snapshot-inclusive ms | Input + output batched, snapshot-inclusive ms |
| --- | --- | --- |
| 1 | 0.411-0.468 | 0.409-0.414 |
| 2 | 0.666-0.675 | 0.703-0.720 |
| 4 | 1.195-1.222 | 1.195-1.199 |
| 8 | 2.247-2.251 | 2.191-2.194 |
| 16 | 4.354-4.389 | 4.176-4.193 |

These are separate correctness runs, not paired ABBA performance trials. Each
row snapshots the entire eight-slot recurrent/conv allocation even though only
one slot is active. Do not use these costs as an optimized verifier curve,
extrapolate them over 48 layers, or claim a serving speedup. Next isolate active-slot
snapshot/restore cost, retain full-state comparisons for correctness, and test
attention KV masks/positions plus all-layer target logits before drafter integration.
Current local validation: 87 CI/helper tests and 26 drafting/selector tests passed;
shell syntax and whitespace checks passed. No serving default or runner access changed.

## E3 active-slot storage (2026-09-06)

The `gdn-active` arm snapshots only logical slot zero: recurrent axis 0,
convolution axis 1. It retains BF16 dtype and padded tile storage, so convolution
snapshots do not necessarily shrink physically. Restore clones the saved tensors
before the native consuming writers and changes only slot zero, preserving live
buffer addresses. No installed source or arithmetic changes.

Repeat the 216 prefix/continuation and 30 stale-state gates, now comparing complete
live state against the full-state reference after each active restore. Additionally
change all eight slots, restore only zero and assert slots 1..7 stay unchanged on
both chips. Compare full versus active save and restore separately with three ABBA
blocks, 30 trace replays per sample, synchronized outside the measured loop.
This is a snapshot-cost experiment, not full-model speculative throughput.

Run [33998325637](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33998325637),
code `da39648`: all 216 prefix cases, 30 stale-state controls and 15 changed-idle-slot
checks passed. Padded snapshot allocation fell from 7,602,176 to 2,097,152 bytes per
chip (72.4% smaller). However active save was 11.1-11.4% slower (~0.0593 vs 0.0533 ms),
and restore was 13.61-13.66 times the full-copy cost (~0.732 vs 0.0537 ms).
Do not promote this native-writer restore as a latency optimization.

The native convolution slot writer reconstructs the whole buffer with slice/concat
and copy. Next `gdn-direct` tests a 48-worker generic-op DMA kernel: full tiles for
slot-zero recurrent state, but only the two 32-byte BF16 row-zero face segments per
convolution tile. It uses no compute kernel or arithmetic, does not overwrite other
rows, and avoids native restore's reconstruction. Both directions use preallocated
snapshot/live addresses and the identical correctness/isolation and ABBA gates.
Shapes, interleaved DRAM placement, dtype, both chips and non-aliasing are guarded.

Direct-copy runs 33998799270 and 33998989061 failed exactness; preserve their timings
as invalid-for-promotion. Diagnostics isolated 2,560 wrong values per convolution
tap/shard, starting at channel 16, before trace capture; recurrence matched. The
second face-row used a scratch offset of 32 against a DRAM face offset of 512.
The retry preserves the same NoC address alignment with scratch offset 512;
do not confuse this with changing the BF16 face layout or arithmetic.

Run [33999114362](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999114362),
code `2b38da1`, passed 216 exact prefix/continuation cases, 30 stale-state controls
and 15 changed-idle-slot checks. The alignment fix retained exact recurrence and
all convolution channels on both chips. Copy kernel SHA256:
`4f921500b63817a5f26f288725a9da1fee9223841a68d058d96fac0f67a23428`.
In three paired ABBA blocks, save took 0.02266-0.02282 ms versus full-copy
0.05319-0.05346 ms (57.16-57.48% less); restore took 0.02265-0.02272 ms versus
0.05320-0.05336 ms (57.35-57.55% less). Padded allocation remains 72.4% smaller.
This is a validated per-layer snapshot microbenchmark, not a decode-speed claim.
The full-model oracle pins this kernel hash as its prerequisite.

## Full-model rollback oracle

Added `full-prefix`, gated behind the direct-copy layer test. It retains native
sequential target decode, not the experimental multi-row GDN forward. Compare
eager and trace logits, then freshly prefill each branch at lengths 63/64/65.
For retained prefixes 0..16, advance through a deliberately wrong rejected tail,
restore all 48 GDN slot-zero snapshots, rewind the explicit decode position and
compare two corrected steps against an uninterrupted native reference.

Attention KV rollback is logical: future rejected entries remain physically present,
but only positions below the current valid length may affect attention. Compare
dequantized valid KV tensor values on both chips for all 16 attention layers,
including the 64-token page transition, and all active GDN states/full logits.
Wrong logical-to-physical page mapping and omitted GDN restore are negative controls.
Fresh prefill per branch prevents one branch's correction writes contaminating the
next branch's valid prefix. Snapshot buffers are preallocated before trace capture.

This oracle is a prerequisite for the batched target verifier. It does not validate
tree branches, stochastic acceptance, request cancellation/epoch reuse, long contexts,
or a serving speedup. It must not turn the policy's `verifier_ready` on in serving.

Initial full-model run 33999230778 loaded and warmed native decode, then failed in
the harness's KV reader: native `ttnn.Shape` supports integer indexing, not Python
slice indexing. No full-model correctness case ran. Convert the shape to a tuple
before extracting dimensions, add a native-like shape regression fixture and retry.

Run [33999532634](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999532634),
code `d203018`, completed successfully. Artifact evidence confirms:

- All six eager/trace baseline comparisons matched full logits exactly.
- All 102 rollback cases passed: lengths 63/64/65, eager and trace, every retained
  prefix 0..16. Both correction steps matched full logits, all 48 GDN active states
  and logically valid KV values in all 16 attention layers, on both chips.
- All six negative-control configurations detected both stale GDN state and wrong
  page mapping (12 checks), rather than merely exiting successfully.
- Two reusable all-layer compact snapshot sets occupied 201,326,592 bytes per chip.

This establishes full-model serial rollback correctness for the tested page-boundary
fixtures. Rejected KV entries need not be physically erased in this path: rewinding
the decode position and restoring GDN state kept the rejected suffix from affecting
the checked continuation. It does not establish that concurrent multi-row KV writes
to shared pages are safe. The next gate is batched attention/target verification
against this oracle, including shared-page write hazards, then its measured cost
curve. No speculative serving flag, precision setting or deployment changed.
