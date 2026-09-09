# Qwen3.8-27B: two-card experiment programme

## Current programme position - 2026-09-10

**Completed target-kernel experiment: fixed-packet weight reads.** The sixteen-producer
mapping and native math stay unchanged; only fixed-size BF4/BF8 read dispatch
changes. BF4 generated reader code falls from 1,156 to 832 bytes, not a latency
claim. Full-size BF4 and BF8 transport each pass 24 exact checks, clean exit and
independent qualification. Full-MLP simulator run `20260909T145410Z-414` passes all
188 checks and clean teardown. The original runtime is restored and both MLP and
native-weight-view gates pass independently. Hardware ABBA
[34371489865](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34371489865)
passes 118 correctness checks, but candidate 0.424554 ms loses to native
0.334514 ms in all nine blocks: **26.92% slower**. Independent reconciliation
rejects full-model promotion. No new TG result or serving change.
[Experiment and gates](weight-read-packets-2026-09-10.md).

**Latest matched 4K gain: PP3,322.74 / CTX4,096 / TG75.42, B1.** Native
approximate draft attention improves control61.47TG by22.68% in
[34342721182](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34342721182).
All six requests retain exact target tokens/GDN/valid KV/inactive state; each
policy reproduces its own audited proposals. Drafting falls40.32 to20.35ms/block;
acceptance stays105/119 per request. The unchanged verifier remains61.72ms/block,
above the entire35.59ms cycle budget for200TG at this acceptance rate.
Independent artifact/source/teardown validation passes. Same-tag repeat
[34343945544](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34343945544)
also passes: PP3,326.31 / CTX4,096 / TG73.16 versus control61.08. Pooling four
timed requests per arm across both runs gives **74.27 versus61.28TG (+21.21%)**;
all12 requests retain exact target tokens/state. Held-out coding and serving
acceptance remain open. No candidate sample or stall is discarded.
[Full PP / CTX / TG and acceptance table](drafter-numerics-experiment-2026-09-09.md).

**Next wider-drafter prerequisite: pinned DSpark intake.** The published v2
checkpoint is not DFlash2-compatible: full attention, YaRN rotary tables and a
sequential full-vocabulary Markov head replace the sliding-window/convolution/
top16 path. The initial metadata intake and14 CPU tests qualified compatibility
only; subsequent native selector work is below. Training16 does not qualify serving15
proposals: the published serving convention is7 proposals/8 target rows.
[Port boundaries and next gates](dspark-port-contract-2026-09-09.md).

The DSpark native selector now has a simulator-only device-feedback composition:
embedding -> HiFi4 full-vocabulary bias -> FP32 add -> argmax -> next embedding.
Small gate20260909T113815Z-406 passes80 checks, both chips, exact toy scores and
clean exit0, without native runtime changes. Learned gate20260909T113949Z-365
fails the first FP32 score comparison across248,320 IDs:21.5% mismatch, maximum
absolute difference0.0025434494. It closes cleanly with outer exit1; no learned
selector qualification or hardware dispatch follows. Diagnostic120503Z-397
isolates native BF16 grouped-product matmul rounding:64 learned columns match
the existing Blackhole arithmetic reference bitwise, while addition is exact.
The failed FP32 gate remains rejected. The separate native-arithmetic policy now
passes80 small simulator checks and **100 learned248,320-ID/7-proposal checks**.
Run`20260909T123008Z-401` completes eager, changed-input replay, immutability,
stale controls and clean exit on both chips. All28 FP32 score comparisons still
fail the old tolerance; these fixtures have zero greedy-token differences against
independent FP32 trajectories. Neither result certifies target integration.
DSpark-specific YaRN CPU
tables match pinned upstream functions bitwise at21 selected positions through
262143; this is not a262K model run. Both Markov matrices agree with the verified
3.714GB checkpoint. Native rotary fails one padded value in the unchanged accuracy
gate; exact input/output casts isolate the error inside native rotary math. A
separate composed FP32 candidate **passes all234 simulator checks**, with bitwise
padded CPU outputs and exact replay/ownership/position controls, clean exit0.
This is not a fused-kernel speed claim. The full five-layer CPU backbone matches reviewed upstream methods
bitwise at all48 checked stages across four synthetic-input cases. The composed
rotary's widened arithmetic is distinct from that backbone's BF16 product rounding.
All1,102 host
and58 harness tests pass. This establishes a CPU reference, not a TT backbone,
real-target integration, coding acceptance or hardware throughput result.
The serving input contract is now explicit: anchor plus six masks, sample all
seven draft rows, then verify anchor plus seven proposals. No DFlash2 row-dropping.
The full-context attention matrix now passes all **236 simulator checks** with
explicit precise exponential and 64-key chunks: run `20260909T154029Z-441`.
All 20 short/4K numerical comparisons and 24 exact replays pass with the unchanged
0.01/0.01 CPU threshold. Historical 32-key failures remain preserved. The saved
failing operands and FP32 reference are unchanged; only the chunking policy and
masked 4K padding differ. Native files/binaries are restored/unchanged, ownership
locks released, and independent qualification passes. Learned TT backbone and
real-target integration remain next; this is not a hardware or coding-quality result.
[Detailed boundaries](dspark-port-contract-2026-09-09.md).

The learned DSpark FC/normalization now **passes all 162 simulator checks** in
run `20260909T165726Z-382`: full checkpoint tensors, all 5,120 outputs, two frozen
CPU patterns and exact changed-input traces. Explicit FP32 normalization fixes
the original accumulated rounding errors without changing the tolerance.
Ownership, clean exit 0 and independent source/native reconciliation pass;
1,142 host and 59 harness tests pass. Original native-RMS failures remain rejected.
The explicit host handoff does not qualify fabric or a complete pipeline.
One learned TT layer, then all five and target integration remain next.
[Evidence](dspark-learned-projection-2026-09-10.md).

The complete learned layer-zero port is prepared with all eleven tensors,
all Q/K/V heads, full 17,408-channel MLP and explicit TP partial-sum boundaries.
CPU preflight matches all six upstream checkpoints exactly. Its 664-check
simulator matrix is separate from hardware/fabric qualification; the initial
eager diagnostic cannot substitute for that matrix. Eager-only run
`20260909T173511Z-440` completes all phases but fails 102/192 eager comparisons;
all input, parameter, padding and CPU checks pass. Native grafts are restored,
the failed evidence is retained, and no hardware promotion follows. Own-input
attribution separates accumulated error from attention/down-projection and
residual-rounding differences. All 1,154 host and 59 harness tests pass at this checkpoint.
The captured-input residual follow-up passes **48/48 exact checks** after
explicitly widening both operands. Both layer residuals now use that candidate;
attention/down-projection diagnostics and the full 664-check gate remain open.
This component fix does not change the measured 74.27 TG hardware result.
[Layer bring-up](dspark-learned-layer-2026-09-10.md).

**Reporting and next measurement: PP / CTX / TG.** The 78.06 TG result is at
CTX170, B1, up to T8, with PP510.65; it is not a 4K/32K/64K result.
Request summaries now expose these metrics together, excluding correctness
audits and keeping ABBA arms separate. An offline artifact reporter recomputes
the rates and preserves run/hash provenance; it does not rerun hardware.
The 4K row now passes: PP3355.04 / CTX4096 / TG58.18, B1. Run34285614832
(`afaedc7`) produces two121-token EOS responses at57.28/59.10TG, with separate
native token/GDN/valid-KV/inactive and learned-feature/proposal audits. Drafting
costs54.57ms/block, verification61.87ms; complete mean request7.05s. This is not
a matched comparison against the different CTX170 prompt/output. The8K run
34286889429 (`868dcbc`) also passes: PP3149.33 / CTX8192 / TG46.20, with
121-token EOS responses. Samples41.94/51.43 include a424ms publication stall;
the combined rate retains it. Host/runtime attribution is needed before making
a context-scaling claim. [8K detail](dflash-8k-context-2026-09-09.md).
The same-code repeat34289671592 passes PP3298.59 / CTX8192 / TG53.48;
samples53.40/53.57, complete mean request8.70s. Keep both runs: the repeat
does not isolate the original stall or establish a new optimization gain.
16K/32K/64,504 remain unmeasured. The former 2,048-token prefill limit now has
an opt-in tail-window initializer with absolute positions; full target KV stays
unchanged. The first 4K hardware run34282865344 failed before timing: slot-prefill
delivers 2K chunks, not a full-sequence layer output. Chunk-aware capture now
passes its simulator gate, 889 host tests and the complete hardware retry.
[Initialization gate](dflash-long-context-2026-09-09.md). Multi-stream DFlash
remains unmeasured. [Matrix and gates](pp-ctx-tg-benchmark-matrix.md).

**Current optimization: cached historical draft K/V.** Incremental learned
projection passes20 exact simulator comparisons. The opt-in4K ABBA integration
now updates only committed rows, preserves atomic cache publication and audits
against full-history recomputation. Integrated gate20260908T230732Z-297 passes
eight attention-operand,36 replay,12 unchanged-state and four full-history
comparisons, plus four detected stale controls. All11 source hashes match;
846 CI and60 harness tests pass. The opt-in4K ABBA run34291073085 (`20915fe`)
passes: cached PP3307.88 /CTX4096 /TG60.33 versus uncached PP3293.42 /TG58.81
(+2.58%). All six121-token requests and cached/full-history audits are exact.
Drafting falls53.59->41.11ms/block, but publication rises3.75->11.72ms;
verification stays61.69ms. Full request time worsens7.09->7.41s. No serving
adoption. Next capture incremental K/V projection, while retaining the separate
target-verifier optimization requirement; neither is a200-TG claim.
[Gates](dflash-kv-cache-2026-09-09.md).

**Captured K/V update simulator gate passed.** Fixed32-row learned projections now have
an opt-in trace with stable feature/absolute-RoPE inputs; bank-tail assembly and
atomic publication remain unchanged. Simulator235318Z-417 passes12 exact
projection comparisons with changed features/positions and all earlier cache,
attention-operand and live-trace checks. All12 source hashes match `8a12962`;
851 CI plus60 harness tests pass. The new4K hardware ABBA retains caching in
both arms and isolates projection capture. Run34294263149 on `7780527` passes
correctness but is effectively flat: PP3292.61 /CTX4096 /TG62.11 versus cached
eager-update PP3297.43 /TG62.01 (+0.15%, within timing spread). Publication falls
only10.72->10.29ms/block; verify/readback stays61.90ms. No performance promotion.
The earlier60.33-TG result remains eager-update, not the matched control here.
Next prioritize attribution of the actual commit-only target verifier; the old
static-profile sums are not a current critical-path measurement.

**Current-request verifier profiler passed.** The opt-in4K suite observes the
actual cached T8 commit-only verifier calls, retains all native-reference audits
and emits no PP/TG result. Runtime replay counts must match every marked request
call on both chips; report per-chip envelopes, operation/core sums and host call
costs separately. No new kernel math or trace contents;860 CI and60 harness tests
pass. Hardware run34296336943 on `8460b69` passed its complete request audit but
hit the96 GiB host limit during profiler finalization (`oom_kill=1`); no device
CSV survived. The collection fix requires incremental profiler dumps, keeping
all17 actual verifier calls and every correctness audit. Hardware retry34298049648
on `da69ef8` passes: device envelopes61.539/61.542 ms, with16 steady replays/chip.
Chip0 summed matmul time32.050 ms, GDN recurrence6.290 ms and native SDPA5.827 ms
are attribution, not throughput or additive cross-chip critical-path costs.
No OOM occurs;92.871 GiB peak still leaves tight headroom. Next test16-row
activation tiles with unchanged compressed weights, including conversion and
trace costs; below16 is unsupported on the pinned compressed-weight mcast path.
This is simulator-first work, not a claimed speedup or another grid sweep.
The first16-row projection composition failed output retile allocation on one
core (2,061,184 bytes versus1,572,864 L1 bytes). Bounded face-copy DMA now passes
gate/up/down separately. The complete MLP passes16 eager and32 traced bitwise
comparisons on both simulated chips, including both DMA boundaries. Stock
multiplication dropped small tiles; FPU rounding also failed exactness. The new
product keeps native SFPU semantics and16-row buffers. No timing claim: next is
real layer0 hardware ABBA against actual native forward, including the collective.
Run34303979499 on `6457ec8` exits after the link prerequisite, despite a green
CI status: no MLP test or timing exists. Routing is corrected and executed shell
tests cover the transition. A mandatory host-side artifact validator now rejects
missing/incomplete MLP comparisons or timing blocks. The corrected run34305753974
on `87d4cab` passes6 eager and12 changed-input traced exact checks using real
layer0 weights, but all9 ABBA blocks regress: native0.334811ms versus
candidate0.350553ms, **4.70% slower**, including staging, both conversions and
the same four-link collective. Independent artifact validation passes and
`eligible_for_full_model_gate=false`. Do not integrate this candidate or present
the simulator pass as a performance result. No PP/CTX/TG gain or serving change.
[Evidence](tiny-tile-projections-2026-09-09.md).
[Scope and gates](current-verifier-profile-2026-09-09.md).

**Upstream comparison: MiniMax-M3 pipeline zone profiling.**
[PR55654](https://github.com/tenstorrent/tt-metal/pull/55654) adds stage-local
prefill profiling, not a Qwen decode kernel. Its attached chart compares worst-chip
zone times at EP32/16/8 with1D fabric and a5K chunk attending50K cache. Parent and
child zones must not be summed. Transfer the measurement method: named semantic
zones, worst-chip comparisons, and separate uninstrumented application timing.
Do not transplant its MoE timing or infer two-card B1 decode speed from it.

**Qualified attention kernel: skip padded draft-query QK work.** T8 has8 live query rows but
the cached SFPU dot evaluates32. The opt-in kernel retains live FP32 arithmetic
and zeros only raw padded scores; the single-allowed-key padding mask preserves
all final attention rows. Both short learned-operand and 2K synthetic component
simulations pass exactness and changed-input replay. Hardware run34310168821
passes both contexts: complete attention latency falls 7.29% and 9.50%, with
every ABBA block winning. This is not whole-drafter or request throughput.
Integration now routes the opt-in candidate through the actual cached T8 drafter,
with host mask validation before capture/replay and unchanged native arithmetic.
Both short learned-operand and 2K synthetic integration simulations pass all32
output rows on both chips, including changed inputs and both traces live.
The hardware gate is a matched 4K complete-request ABBA: cached history, fused
convolution, commit-only target verification and four links remain in both arms.
It reports PP / CTX / committed TG, not the attention microkernel rate.
Run34314276820 passes all complete-request correctness audits, but candidate
TG54.21 is below control59.42. The stalled candidate sample remains included.
Same-code run34315861448 also passes correctness: candidate61.53/control59.13 TG.
Pooling all eight measured requests across both runs gives candidate57.64 versus
control59.27 TG (-2.76%). No samples are removed; there is no TG promotion.
[Scope, padding proof and gates](live-query-attention-2026-09-09.md).

The pair's programmable-DRAM prefetch capability is false, with the API present;
the specific firmware/harvesting cause is not established. Next verifier work
uses [real Tensix weight-stream producers](dram-prefetch-mlp-2026-09-09.md), without
spoofing DRISC support. Eight sender cores feed 68 gate/up or 80 down receivers.
The five-block BF4 and BF8 simulations each pass eight eager, twelve changed-input
trace and four stale-input checks across both chips, with exact compressed words
and clean closure. Full BF4 gate transport also passes. Full BF8 kernel checks
complete, but editing its running shell wrapper breaks final status recording;
the failed launch is retained and a regression-tested compound terminal block
fixes that race. It is not counted as a clean suite pass.
The new zero-copy projection reuses native matmul compute without relaxing the
DRISC-only native entry's validator. Its five-block BF4/BF8 simulations each pass
all32 physical output rows, native lead/expanded controls, changed-input replay
and raw input/weight immutability. Full gate/up/down tests now also pass, including
clean outer wrapper exits. The complete MLP also passes simulation with two
shared FIFOs, shared workspace, two weight fixtures and captured input copying.
Its 188 control/eager/replay/input/negative checks pass, with clean teardown and
outer exit zero. Real-weight hardware ABBA now passes correctness but fails
performance:0.743766ms streamed versus0.334683ms native, all nine blocks slower.
No full-model promotion. Its native
collective adapter preserves the borrowed partial output instead of letting the
native wrapper forcibly deallocate it. This is not a bandwidth or TG result.
[Projection gates](tensix-streamed-projection-2026-09-09.md).
[Pooled MLP and hardware gate](tensix-pooled-mlp-2026-09-09.md).
The sampler's four-link override is separate from the target model's CCL helper,
whose pinned P300 lookup defaults to two links. A separate
[cached 4K whole-request ABBA](target-model-link-counts-2026-09-09.md) measures
target two-versus-four-link requests with unchanged native MLP kernels, sampler
and drafter. It records actual helper calls and cross-arm final GDN/KV/inactive
digests. Run34327029099 passes correctness: at CTX4096, two-link control is
PP3324.76/TG62.39; four-link candidate is PP3430.28/TG61.41. The candidate is
1.57% slower in TG in this ABBA and is not promoted.

The first pooled-MLP hardware attempt,34327911787, fails during preparation:
native weights are2D, while the qualified candidate requires4D. No candidate
kernel or timing executes; teardown is clean. A separate full-size metadata-view
simulation now passes exact borrowed-buffer/content/lifetime checks on both
chips and the actual native shard axes, with clean teardown and zero exit.
The unchanged full-MLP kernel gate still passes. Both hardware arms stay at
four links for the metadata-only retry;983 CI and57 simulator-harness tests pass.
Retry34330511791 passes all118 hardware checks and clean teardown, but takes
2.22x native MLP latency. Preserve this negative result; profile the transport
and consumer before another kernel change. It provides no new PP/CTX/TG result.
The [device attribution suite](tensix-mlp-device-profile-2026-09-09.md) passes
34332550656: same qualified kernels and real weights, twenty fully accounted
trace replays on both chips. Gate/up/down are roughly222/217/244us streamed
versus96/88/122us native. Copy and collective do not explain the regression.
Next: test16 rather than8 producers, keeping compute/maths/FIFOs/CCL unchanged;
full-size simulator qualification precedes hardware. No instrumented-speed promotion.
The [sixteen-producer variant](tensix-sixteen-producers-2026-09-09.md) is now
simulator-qualified. Run20260909T092252Z-403 passes all188 complete-MLP checks,
clean teardown and outer exit0;998 CI and57 simulator-harness tests pass.
Its own source-bound sixteen-producer report and exit are checked in unchanged.
The original packer and both native binaries are verified, and both independent
simulator gates pass against the restored runtime. Hardware ABBA completes
as[34337968780](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34337968780)
on immutable `ci-qwen-hardware-ca22384`. All118 checks and independent artifact
validation pass, but native0.334690ms beats candidate0.455749ms in every block:
36.17% slower, no promotion. The historical eight-producer0.743766ms falls38.72%,
which supports further producer/transfer investigation but is not a runtime win.
All36 samples remain included; native weights and serving defaults stay unchanged.
No new PP/CTX/TG result follows from this component experiment.
A separate
[approximate-drafter experiment](drafter-numerics-experiment-2026-09-09.md) can
allow different proposals while retaining exact target outputs/state; it must
not turn failed native-attention numerical tests into passing exactness claims.
Its native-attention primitive now passes76 mask/replay checks at each of31
and2,048 draft-history rows, with unchanged SDPA sources and restored native
packer/binaries. The learned maximum error0.392848 remains recorded as a failed
original numerical comparison. The separate six-request policy comparison is
now implemented with exact native target token/GDN/KV audits, independently
reconciled proposal counts, per-policy deterministic trajectories and clean
teardown/source gates. All1,021 CI and57 simulator-harness tests pass. Hardware
suite `full-dflash-native-proposal-request` passes as
[34342721182](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34342721182)
on immutable `ci-qwen-hardware-f1c1b69`: candidate75.42TG versus control61.47,
CTX4,096, independently validated. The exact attention numerical gate remains
failed; this separate approximate-proposal policy preserves target correctness.

**New best: captured DFlash2 T8, commit-only GDN and fused convolution: 78.06 TG.**
Run34246322267 (`1948a21`) passes matched ABBA: control72.213017,
candidate78.061082 (+8.10%). Drafting falls31.054947 ->24.811769ms/block;
verification stays58.17ms. All six150-token EOS requests preserve identical
proposals, acceptance, native tokens, GDN, valid KV and inactive slots. The
candidate additionally passes880 exact learned-convolution comparisons.
Candidate PP510.65; complete prefill/setup/decode6.56/6.29s, no amortization.
Simulator and867 host tests passed for that candidate. The target remains200,
not achieved. Verifier normalization/data movement remains the kernel priority
after the long-context initialization gate.
[Matched result](dflash-fused-convolution-2026-09-09.md).

**Previous best: captured DFlash2 T8 with commit-only GDN, 70.34 committed TG.**
Hardware run34242926044 (`20dbd99`) passes matched ABBA: control65.501887,
candidate70.336300 (+7.38%). Verification falls65.427248 ->58.078125ms/block;
drafting is unchanged at32.37ms. All six150-token EOS requests preserve identical
proposals, acceptance, native tokens, GDN, valid KV and inactive slots. Separate
audits check all feature/proposal outputs and22 pre-decision GDN snapshots.
Candidate PP530.32; complete prefill/setup/decode6.90/7.06s, no amortization.
Simulator passes all9 prefixes and actual continuations;859 host tests pass.
The target remains200, not achieved. Next: fuse the drafter's repeated grouped
convolution while preserving BF16 rounding; no new fusion speed claim yet.
[Matched result](dflash-commit-only-gdn-2026-09-09.md).

**Qualified fused DFlash2 convolution.** One dispatch replaces
the repeated shift/expand/cast/arithmetic sequence while preserving all BF16
rounding. Simulator passes36 exact output/replay comparisons,12 request-wrapper
comparisons and stale/rounding negative controls;867 host tests pass. The first
register-capacity bug is fixed and retained as failed evidence. Hardware ABBA
keeps commit-only GDN in both arms and audits every learned convolution.
[Experiment](dflash-fused-convolution-2026-09-09.md). Serving defaults stay unchanged.

**Previous best: captured DFlash2 T8,66.76 committed TG.**
Run34238134003 (`eadaa41`) passes three complete150-token coding requests through
EOS. The two uninstrumented requests measure68.409353/65.197390 TG, aggregate
66.764763. All native token/GDN/valid KV/inactive checks pass; the audit also
checks1500 row/chip/tap comparisons and22 exact eager/trace proposal comparisons.
Mean draft31.265333ms, verify65.266694ms, publication3.384118ms; PP517.112935.
Complete prefill/setup/decode7.34/7.43s, no setup amortization. This is one coding
task, not held-out quality, serving adoption or a matched MTP comparison.
Captured T32 run34239197088 passes but is slower:60.743458 TG,18 blocks/request,
draft32.156012ms, verify98.436125ms,133/558 accepted proposals. T8 remains the
lead candidate; target-verifier kernel cost is next, not another wider-draft
assumption. [Captured results](dflash-captured-proposals-2026-09-09.md).

**Earlier eager integration: T8 37.64 / T32 41.03 TG; MTP58.33 TG.**
The opt-in `full-dflash-request` suite connects all five learned layers to the
target's embedding/head and five post-layer feature taps. Only committed target
input rows enter the drafter history; two preallocated history buffers survive
verifier trace replay. The first complete coding request audits every published
feature row against native decoding; two further requests measure committed TG,
with setup reported separately. All three must match native tokens, GDN state,
valid KV and inactive slots. Run34232609121 (`5dc0878`) passes all three complete
150-token requests through EOS. The two uninstrumented requests accept129/154
drafts in22 blocks each and measure37.11/38.18 TG, aggregate37.641552 TG.
Drafting costs110.58ms/block, verification65.37ms; mean6.82 committed tokens/block.
PP568.82 tok/s; complete prefill/setup/decode8.24/8.17s. This is one coding task,
not held-out quality, a serving result or a matched comparison against MTP.
The earlier CTX178 stall was a V-head tiled reshape, fixed with native head split
and concatenation after40 exact simulator checks. Attention math is unchanged;
the failed native SDPA candidate remains disabled.

**Eager T32 passes after fixing the single-stream harness allocation.**
The checkpoint was trained for eight-token blocks. `full-dflash-wide-request`
tests31 parallel proposals and up to32 target rows without changing defaults.
The dense draft layers already use physical32-row tiles. Full target token/state
and feature-publication checks remain mandatory. Run34236992387 (`7539fae`)
passes three150-token requests through EOS; two timed requests aggregate41.027829
TG in17 blocks each, mean8.82 committed/block. Draft111.00ms, verify98.26ms;
PP554.23, complete prefill/setup/decode9.82/9.22s. This is not a matched T8 gain.
The first T32 attempt exhausted DRAM: the harness reserved8200 physical KV pages
while one request addresses1024. Reserving1032 retains65536-token capacity and
all eight GDN slots. Serving defaults stay unchanged.
The next opt-in candidate captures the complete five-layer proposer with
fixed-context masked padding; every audited proposal must match its eager
equivalent before the two complete timed requests count.
[Integration details](dflash-full-request-2026-09-09.md).

**Parallel-drafter work: native DFlash2 SDPA is implemented but not qualified.**
The native kernel hardcodes an approximate main exponential despite
`exp_approx_mode=False`. A geometry-scoped precision correction passes synthetic
CTX0/31/2048 and changed-input trace replay without relaxing tolerances.
Learned layer-zero attention still fails: 65/16384 values exceed the existing
bound; FP32 intermediate buffers reduce this to41 but do not pass. A single
64-key chunk also fails44 values on captured rank-zero operands. No hardware
job was dispatched for these known failures; latest committed TG stays58.33.
Captured, hash-pinned operands now reproduce the failure without reloading
weights or repeating projections/fabric setup. That native-kernel investigation
is retained separately; it no longer blocks the composed-path request integration.
[Evidence and scope](native-draft-sdpa-2026-09-08.md). Serving defaults unchanged.

**Latest qualified:34216164140 (`3772f3b`), device chain exact, small TG gain.**
ABBA57.244897TG host-stepped versus58.333256TG chained, ratio1.019012 (+1.90%).
All four requests emit150 tokens, accept125/178 proposals in26 blocks and pass
target token/GDN/valid KV/inactive checks plus both chips'319-row MTP K/V hashes.
Drafting27.147793 ->25.151618ms/block; candidate verify65.036741, repair7.769537,
cycle98.867127ms. The ownership fix passes full real-weight/four-link execution.
Mean complete prefill/setup/decode worsens7.596229 ->9.270743s. Chain setup is
3584.509647ms on first use and157.570025ms on second; neither is amortized away.
Retain as a qualified decode candidate, not a serving-latency promotion.200TG
remains unmet; next needs parallel proposals/wider verification, not another
small host-overhead experiment. Report SHA256:
`9d8d530a826b4a4fede6dd39b4cc23ef42f66c3228f53e96b0cf419446fc109e`.

**Resolved chain failure:34215001860 (`bf40963`) failed during warmup cleanup.**
The native `tt_all_gather` wrapper explicitly deallocates its input; the chain
mistakenly retained that consumed embedding/view for later cleanup. This is not
a link-discovery failure or a measured chain result. The consuming input is no
longer owned twice. Simulator103111Z-304 now models that ownership contract and
passes42 comparisons/14 stale controls, exit0 and clean close. SHA256:
`c14a78568765edd29f9b30476a11fc5eb71e854f0af69250599d6b0f35ba5247`.
752 host and60 harness tests pass. The complete hardware retry passes above.

**Prior qualified:34213743464 (`5c0e60c`), retain exact KV-only repair.**
ABBA55.067351TG control versus56.854631TG candidate, ratio1.032456 (+3.25%).
All four requests emit150 tokens, accept125/178 proposals in26 blocks and pass
target token/GDN/valid KV/inactive checks. Both chips' complete319-row valid MTP
K/V hashes also match across every arm. Repair/commit11.623757 ->7.874864ms/block;
mean total prefill/setup/decode7.682483 ->7.250688s. Target200TG remains unmet.
Report SHA256:`451b750fa24e6cdfea0b47dc4bd205673b0eee49dd57a20152265ac3386430d3`.
The identical returned function also passes205 local interval/nonmutation cases;
this checks one actual response, not held-out coding quality. Output SHA256:
`3e7de1eedd8c91838c33c575f3edbd99643d1ce9c5b98ef57bbdd487662ecafc`.

**Device-chain experiment specification (completed):**
Borrow the target BF16 embedding, gather the hidden shards, feed each native
argmax token into the next MTP step and collect all proposals in one readback.
Capture every requested count1--7 before prompt initialization; proposal math,
full vocabulary, acceptance and both target/draft-cache equality stay fixed.
Simulator101731Z-299 passes42 width/chip/seed checks and14 stale controls, with
native keepdim-false argmax, full5120-wide embeddings, exit0 and clean close.
It tests256 embedding rows without physical fabric, not full model throughput.
An earlier simulation caught borrowed-seed view deallocation; ownership is fixed.
Report SHA256:`a83dfb11d3baf3191d57b33fddc3de9464836a30042824e78a7b53026afcf736`.
752 host tests and60 harness tests pass. Full hardware chain result is recorded above.

**KV-only repair specification (completed):** The candidate runs the native
MTP embedding/hidden norms, fusion projection, attention input norm, fused QKV
preparation and both paged writes. It omits SDPA, output projection, residual,
MLP and final norm whose output teacher-forced initialization/repair ignores.
Proposal steps remain the full native MTP layer. No new kernel math: source and
fused-prep flag are pinned. Both arms require identical proposals/acceptance and
complete valid MTP K/V hashes on both chips, in addition to all target checks.
Its pre-launch validation passed746 host tests and60 harness tests.

**Prior run:34212022749 (`1e3738b`), reject approximate cache reuse.**
All four complete requests pass exact target tokens/GDN/valid KV/inactive slots.
Control54.942525TG versus candidate51.116004TG; ratio0.930354. Each emits150
committed tokens. Full repair accepts125/178 in26 blocks; reuse accepts120/209
in31 blocks, identically on both repetitions. Candidate reuses143 input rows
and teacher-forces7, versus0/150 control. Repair/commit drops11.705157 to2.305442
ms/block, but the five additional verifier blocks erase that saving. Mean total
prefill/setup/decode is7.896076s versus7.873359s; no setup amortization claimed.
Report SHA256:`065ec74bdfd46f2f87b0f6a21c1a80fafa35e08e8eb9681b111b516d481c14be`.
Next: exact KV-only MTP repair, skipping unused attention output and MLP; then
device-chained drafting to remove per-token CPU round trips. Target remains200TG.

**Cache-reuse experiment specification (completed):** Control keeps
teacher-forcing every committed draft input. Candidate reuses already-written
accepted draft KV, computes only a missing tail input, and anchors the next
proposal on the last verified target hidden row. This is explicitly approximate
draft history, not equivalent MTP state; the full-vocabulary target stays exact.
Both arms retain serial attention, K7, native-row sampling and four links.
Acceptance may differ between arms but must repeat within each arm. Reports
count reused and teacher-forced rows against every committed decode token.
740 host tests and 60 harness tests pass, including rejection, EOS, fallback,
tail failure and publication guards. No new kernel math or simulator speed claim.

**Prior run:34208889762 (`2f5ab9c`) passes correctness, not a speedup.**
The repaired instrumented request passes all150 tokens, final active GDN/valid
KV/inactive slots and all16 attention layers on both chips. It reports no TG.
The separate four-request ABBA is exact, with identical proposals and routes:

| CTX / B / verification | PP tok/s | Committed TG | Mean prefill + setup + decode |
| --- | ---: | ---: | ---: |
| 170 / 1 / native T1 reference, all four | 532.68 | 19.48 | Target already loaded |
| 170 / 1 / serial-attention MTP, up to T8 | 546.41 | 54.90 | 7.71 s |
| 170 / 1 / repaired parallel-attention MTP | 540.80 | 54.36 | 9.12 s |

Decode ratio0.99014 (candidate about1% slower); **do not adopt short-context
parallel attention for performance**. The old 53.60/49.79 paired sampler gain
remains valid; this run's54.90 control is a current measurement, not evidence
of another isolated optimization. Both MTP arms accept125/178 proposals over
26 blocks/request. Mean serial costs:27.465ms drafts,64.889ms verify/readback,
11.686ms repair/commit,105.050ms cycle. Candidate verification saves1.111ms but
whole-cycle costs increase; extra capture families also increase preparation.
Report SHA256: `a65212b5337d03bf1f96d4be7a22b9d894a15424f1f5be4e8062125945f4cd1f`.

Next structural gate: reduce serial proposal/repair work and integrate wider
parallel drafting, retaining the full-vocabulary lossless target verifier. Do
not spend another matrix on short-context attention or rejected MLP grid sweeps.
200 TG remains unmet; this one coding prompt does not certify general coding
quality, serving throughput, or longer-context performance. No serving defaults
or device reset policy changed.

**Identified fault and simulator-qualified repair:** run34206948191 (`079e552`)
finds 12,277/12,276 corrupt static-prefix mask values on the two chips, with
zero refreshed-tail differences. The saved mask/query/KV fixture SHA is
`6504ecb77022c9a843e6cd77a96e0b9269c01ce0dd9835ecf333f4ad50b69688`.
Simulator090326Z reproduced the exact11275-element hardware output error when
given that mask; a clean mask was exact. This identifies the causal-mask prefix,
not SDPA math or T4 routing, as the immediate failure. The writer that corrupted
the resident prefix is not yet identified; immutable-prefix residency was unsafe.

The short-context reader now enqueues native in-place constant-zero fill before
tail generation, inside every captured mask refresh. It neither multiplies
corrupt/NaN data by zero nor depends on a separately resident zero tensor.
Shared-mask forwards still initialize once, then use the mask for16 layers.
No SDPA arithmetic, target precision, native B1 or default long-context path changes.
Simulator090714Z reproduces the old failure and fixes the saved corrupt-mask
A/B/A case exactly on both chips (18 comparisons, including the failing control).
Report SHA256: `e9a4b87f3931ee5274ee413f2ba438c1a74f03332879e73a08f51a4201f1886d`.

Next hardware gate first audits an entire repaired coding request against native
attention in all16 layers, then runs uninstrumented serial/parallel/parallel/serial
MTP requests. The audit is stored separately and cannot enter TG aggregation.
The multi-family prerequisite now poisons the entire mask with NaNs before each
replay, so a tail-only initialization cannot accidentally pass again.
Simulator090902Z passes all24 output checks,24 mask checks and12 immutable-KV
checks after12 full-mask poison injections, plus6 stale-query controls, at
capacities256/512/768. Report SHA256:
`382558f2631bbc2e127a3c86c3d40e2cb06fc7d6f2255f911fc2e72a1e56c708`.
All733 host tests and60 speculative harness tests pass for the repair.
Final real-data simulator091157Z re-injects the saved corrupt mask before every
A/B/A replay, rather than just before warmup. Old behavior still reproduces the
hardware failure and the captured fill repairs every output exactly on both chips.
Report SHA256: `f7a4b8090589b563648b8018edc1e0a7959f72184b5159afe3ae9cdfffddf8e0`.

Real-query audit **34205529748**, code `2a9c1d1`, localizes the first attention
failure to position258, attention index0, chip0: 11,275/24,576 output elements
differ, max absolute error2.6953125. Only the first four query rows differ;
the final four match exactly. All earlier blocks passed the sixteen-layer audit.
The serial request still passes 150 tokens/state checks (54.47 TG, one control).
Saved query/KV SHA256:
`c720f80b33fecd489b3fb3c52eec63a6246b5779bc3d7a838016677b57f7b372`.

Local real-data simulator084618Z matches the saved native output exactly and
the ordinary folded path also matches native on A/B/A, both chips. It does NOT
reproduce the integrated failure. An exploratory dynamic-chunk masked arm was
rejected by the native API (explicit mask requires nonzero chunk size); the run
exited1, not a pass. Next capture the actual live mask with the same failing
query/KV. Do not change arithmetic or claim a kernel fix from an isolated pass.

Latest hardware attempt **34202741880**, code `f5a4403`, is rejected. The short
attention component gate passed, but the first full-request parallel candidate
changed emitted token 138. The serial arm passed all 150 tokens and state checks
at 54.16 TG; that one control is not a new paired best. The qualified paired best
remains **53.60 TG**, run 34200129693.

Both arms crossed position 254 using native T4 and resumed T8 at 258. The serial
control passed that transition, so blaming routing is unsupported. The first
observed acceptance-count difference is position 284. The pinned image's native
SDPA uses the same non-approximate exponent mode and no explicit compute config;
a dropped compute config is not the explanation either.

Next diagnostic compares folded and native B1 attention on identical captured
real queries and KV in all sixteen layers, on both chips, before accepting a
block. It saves the first mismatching query/cache fixture for local simulation.
Instrumented candidate timing is not TG. Full requests now stop on the first
committed-token mismatch rather than continuing wrong generation to the limit.

| Priority | Verified position / next gate |
| --- | --- |
| Single-stream target | 200 committed TG not achieved; native-row MTP: 53.60 TG, CTX 170; matched padded MTP 49.79 and native reference 19.82 |
| Qualified paired hardware result | [34200129693](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34200129693) passes K7/T8 ABBA: all four requests finish 150 decode tokens to EOS with exact tokens/GDN/valid KV/inactive slots |
| MTP integration | Full-vocabulary device drafting, 169-row prompt initialization and rejection repair execute; 125/182 proposals accepted in 26 blocks; 709 host tests pass |
| Measurement boundary | Decode includes proposal, verify/readback, cache repair and commit; prefill and all setup reported separately and in inclusive totals |
| Fabric fix | Hardware run 34189506734 passes without fallback discovery; no meaningful T8/T32 collective speed gain |
| DFlash2 | Complete captured stack remains unqualified; cancelled simulator runs are not passes |

The native-row sampler improves complete decode by 7.66% in one paired ABBA block,
without changing any proposal, acceptance or token. Mean candidate block costs:
65.312 ms verification/readback, 29.007 ms drafting, 12.178 ms repair/commit and
0.709 ms staging, 107.597 ms total. The 200 TG budget at this acceptance remains
28.846 ms. Keep the qualified sampler in the next MTP candidate, then address
the verifier; do not repeat earlier larger-grid sweeps that failed to show a
useful gain. Short-context parallel attention remains gated, not assumed exact.
Mean prefill + preparation + decode is 7.691 s candidate versus 7.890 s native
in the candidate repetitions. The target is already loaded; this is not a cold
process launch or sustained serving result. Setup is not amortized.
Latest report SHA256: `cda6127029826c86cc59b9c2e3ea775d0850840a0ef580cce0afc113d5398b9d`.

Rejected run 34202741880 compared short-context serial versus parallel attention with
the qualified native-row sampler fixed in both arms. Simulator075510Z-308 passes
24 native-output checks, 24 mask checks, 12 KV-preservation checks and 6 stale
controls at capacities256/512/768, including actual CTX170. The default long-
context path remains unchanged. A shared bounded routing plan retains native
small-width boundary fallback without padding the prompt; full-request exactness
and paired TG, not the simulator pass, determine whether this is adopted. The
real-query diagnostic above replaces an unchanged benchmark retry.
Current diagnostic host suite: 727 tests pass; the prior harness suite has 60 tests.

### First complete MTP request (historical baseline)

No device resets or serving-default changes. One coding request is not a held-out
coding-quality suite. Decode is 3.072 s; prefill + setup + decode is 10.773 s versus
7.900 s native. The 2.47x decode gain does not yet improve this cold request's total
latency. The target model is already loaded for both totals; these are not cold
process-launch measurements. Setup/trace reuse across requests remains unimplemented.

Mean block budget: verification/readback 66.285 ms, seven drafts 38.816 ms,
cache repair/commit 12.040 ms, staging 0.536 ms; total 118.099 ms for 5.769 committed
tokens. The 200 TG budget at this acceptance is 28.846 ms. Prioritize the actual
verifier and draft costs, not more link/fusion microbenchmarks. Next experiment:
exact native-row force argmax instead of sampling 32 rows; qualify the primitive
on changing simulator inputs, then compare complete hardware MTP requests.
Short-context parallel attention requires its own numerical/replay qualification;
do not relax the long-context guard or pad the coding input to hide the mismatch.
Report SHA256: `86a92b560ced248ce91e20a1ec19ea2007ac7a73be64748b5c517b724bfc305c`.

Native-row simulator reduction gate `20260908T072747Z-398` passes: T1/T2/T4/T8,
48 exact token checks across both chips, 48 input checks and 8 stale controls.
The scope is post-gather native untilize/argmax, not simulated fabric or speed.
Hardware run 34200129693 on `fb17bbb` passes both complete four-link samplers:
96 exact token checks, 48 input checks and 16 stale controls, then the four
complete coding requests summarized above. All 716 host tests and 60 harness
tests pass. This resolves the pending native-row hardware gate.

### MTP bring-up failures (resolved by run 34196777661)

First MTP launch 34191983233 failed fresh-prefill seed equality before drafting;
it produced no MTP throughput. The retry restores the required prefill/feature
warmup before parking the native decode trace and keeps all equality guards.
Run 34193704110 confirms matching prefill seeds (`71093`) but stops on an overly
strict embedding-path guard. The corrected guard permits this pinned HF cache's
snapshot-to-blob symlinks. All 16 required tensor headers have been checked on
the host; the next CI run repeats that preflight before its native build.
Run 34195514722 passes checkpoint loading and executes MTP prompt initialization,
then fails on an unaligned slice's tiled output padding. The corrected composition
passes 90 real-TTNN simulator row checks, 24 input-preservation checks and 30 stale
controls. Run 34196777661 repeats the row checks on hardware and completes the
coding request. These failures are resolved, not outstanding qualification gates.

### Earlier checkpoints (historical, not current run status)

Coding retry 34182900581 passed exact native tokens, final state and inactive-slot
checks. CTX 170, one stream, 128 committed tokens: norm-off 15.0196 TG; norm-on 16.1075
TG; native reference in the candidate repetitions 19.6694 TG. Lookup is a regression,
not an overall acceleration: 20/1234 proposals accepted across 108 verification calls.
The output truncates before completing the function and does not reach EOS.
Report SHA256:`b4da1d807d44a3c1b178afeff2c16f68bf5519b8e219149ef2cd9f8bd08fed44`.

Offline replay first reproduces every recorded routing decision, then finds that
an eight-row cap retains all 20 accepted tokens with 450 rather than 1234 proposals.
This is a counterfactual work-count result, not measured speed. A new opt-in ABBA
compares T32/T8 lookup caps with identical native attention and norm batching,
including only the needed trace widths. The coding generation budget rises to 513
in both arms to permit completion; the previous 128-token timing is not a matched
control for that longer run. Output/state equality remains mandatory; proposal
routes may differ across policy arms but must repeat exactly within each arm.
No serving or default synthetic-workload changes. Learned-stack simulator remains
active and is not promoted to hardware.

Coding run34181363845 failed before request verification: the short prompt was
incorrectly routed into an attention replay family qualified only from capacity
4096. No committed throughput result was produced. Do not relax that kernel
guard or pad the prompt to hide the mismatch. The corrected short coding run uses
`full-norm-engine`: native serial attention with paired norm batching, not the
T4/T8 attention comparison. Invalid short-workload/replay combinations now fail
before Docker build or model loading. The failed run spent about18 minutes in
the disposable native build; this native-attention retry does not need that build.

The next learned-draft integration gate is a single captured five-layer stack,
including final normalization. Attention parameters are now prepared once outside
the forward, matching the existing prepared MLP path. The simulator-only opt-in
compares two changing input patterns against the independently checked eager
stack, then replays A/B/A with input-ownership and stale-input controls. All646
Linux host tests pass; simulator correctness is not yet established. This is
synthetic feature history, not a live target-feature cache, shared-head selection,
accepted proposals or an end-to-end speed result. Hardware promotion is disabled.

Priority correction after operator feedback: stop expanding minor kernel matrices.
The next hardware measurement is one non-repeated `merge_intervals` coding request
through the existing certified request engine, comparing T4/T8 attention grouping
in ABBA order and retaining exact native token/state checks, setup costs and output
text. This remains lookup drafting, not completed DFlash integration or a coding
quality certification. It replaces repeated-code input for this opt-in run only;
actual templated context length is reported and thinking is explicitly disabled.
No serving default changes. Future component work must justify its expected
end-to-end impact rather than being promoted on isolated green checks alone.

Deferred-publication T8 simulator023840Z-315 has completed cleanly:9 adapter-prefix
checks,9 two-token continuation checks and1 stale-state control, with norm batching
enabled. Further width expansion is paused in favor of the integrated request
measurement. The new coding prompt and existing suite pass640 Linux host tests.

Captured fusion hardware run34179917205 on35654ca passed on the restored exact
runtime. Transfer health, standalone learned MLP and integrated MLP also passed.
The fusion report contains12 eager checks,4 byte/source weight checks,36 exact
changing-input replay checks,12 stale-input negative controls and9 ABBA blocks.
All timed outputs match on both chips. Median block means:

| Rows | Native trace ms | Fused trace ms | Isolated reduction |
| ---: | ---: | ---: | ---: |
| 1 | 0.208136 | 0.203761 | 2.10% |
| 8 | 0.210611 | 0.201286 | 4.43% |
| 32 | 0.210916 | 0.201526 | 4.45% |

Timing is blocking captured execution, excluding uploads, allocation, capture and
validation. This reverses the eager regression but is a small isolated gain on
geometry-matched draft weights, not target-model throughput or quality evidence.
No serving/full-model default adopts it. Raw report SHA256:
`10589e25ab06196af1591706d0054492285c89d30ceb25e28f027ac8d74a3ae8`.

Next target-state candidate: `DeviceLoopState.decode` copies entry to working
state before the batched path, which also publishes final convolution state;
the caller subsequently reconstructs the final working state with `restore_prefix`.
An opt-in immutable-history/deferred-publication path could avoid both preliminary
copies for packed batched checkpoints. It must preserve T1 fallback, every accepted
prefix including zero, native state publication, rollback and continuation. Do not
remove copies merely by assuming aliasing is harmless. The48-worker profile group
remains a combined attribution, not a measured saving for this proposed change.

The deferred-publication candidate is now implemented behind explicit constructor
and helper options; no production environment flag or serving default enables it.
For packed batched T>1, convolution reads the immutable entry snapshot and leaves
publication to the existing selected/final restores. T1 keeps the old copy path.
Routing/engagement tests and the636-test Linux host suite pass. Simulator
`20260908T023342Z-492` is the active T2/seed0 gate for all prefixes, immutable entry,
inactive slots and two-step continuation. An earlier invocation used the wrong
source root and failed before simulator execution; the active run uses the original
hash-pinned GDN audit from34009341359. No latency claim or hardware promotion yet.

T2 simulator023342Z-492 completed cleanly: all3 selected prefixes and all3
two-token continuations pass, with1 stale-state negative control and immutable
entry/inactive-slot checks. The next active gate023840Z-315 usesT8/seed1 and the
96-worker recurrence plus batched norm. The simulator model adapter now forwards
the same norm selection and audited source root as its component control; this
avoids comparing an optimized component with a differently configured adapter.
The11 targeted adapter tests pass. Full-model wiring and hardware remain gated
on this wider simulator result; existing production defaults are unchanged.

Full-model library wiring is prepared via explicit `ModelBatch` and timing-helper
arguments. The paired control retains every current attention, normalization,
cache and prefix-zero-reuse setting; only deferred publication differs. Existing
CLI/CI paths do not enable it. The638-test Linux host suite passes. T8 simulator
023840Z-315 is still running through all-prefix checks; do not treat this prepared
wiring or passing host tests as hardware correctness or a latency improvement.

Runtime recovery superseding the failure notes below: the operator supplied a
jump-host route. Read-only registry inspection established that `f1e9b1a64b4f`
is the OCI image-index digest, not the platform config digest. The original tag
has not been shown to change; the earlier config comparison was incorrect.
Pulling `tt-vllm@sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`
through the existing authorized host login restored the exact image; Docker
inspection returned that full ID. No credentials were copied, cards opened,
workloads stopped or registry objects modified. A subsequent root fuser check
found no device users, but that snapshot does not guarantee future exclusivity.

Learned-attention capture prerequisite `20260908T021855Z-300` passed TTSim:
four eager FP32-reference checks, six exact A/B/A replay checks and two stale-input
negative controls across both chips. Q/K/V and mask all change at persistent
addresses; caller inputs remain exact. Optional trace ownership removes the
helper's internal synchronization/deallocation and retains buffers until release.
This is the attention core, not a complete learned layer/drafter or a timing claim.
Default eager behavior is unchanged. Full Linux host suite now passes634 tests.

Latest runtime checkpoint: captured fusion hardware run34178476409 failed before
container creation/card access because pinned image `f1e9b1a64b4f` is missing.
Read-only inventory34178884716 passed: both cards remain visible at PCIe x16/x4;
the available TT serving candidate is `thatch-serving-tt:d9a3216` (`e6ba124656d0`),
not the benchmark runtime. No resets, serving changes or model execution occurred.
Device idleness remains unverified; other agents may be using the host.
Historical inventory identifies the missing image as
`zot.thatch.local:5000/tt-vllm:qwen38-k`; registry recovery must verify the full
pinned image ID before any benchmark. Do not substitute the available image.
The [README](../README.md) now separates serving, synthetic request and kernel
results; absent PP measurements are explicit rather than inferred from TTFT.

Exact-image registry recovery34179195465 reached the historical registry but
failed with `no basic auth credentials`; no image was pulled or cards opened.
This repo has no repository secrets configured. Recovery needs a pull-only
registry account or operator restoration of the exact pinned image. The optional
login uses a job-owned temporary Docker config, never another agent's credentials.
Host suite on the Linux TT-Sim environment passed630 tests at f5c60dd; the Windows
full-suite attempt is not a valid pass because Linux-only dependencies failed.

The200 committed coding tokens/s objective is not achieved. Serving defaults
remain unchanged. The following is the current checkpoint; dated entries below
are preserved historical evidence, including failures and superseded next steps.

| Track | Validated result | Remaining gate |
| --- | --- | --- |
| Target verifier | Best T8 static block62--64ms; exact logits/state/KV/rollback | Reduce below40ms including eventual draft/commit overhead |
| Matched attribution | CI34170503918 passed; matmul32ms, generic18ms summed costs | Optimize dominant kernels; profile sums are not critical-path latency |
| Learned drafter | Five-layer hardware correctness; MLP trace1.911ms median | Full captured draft, live feature history, shared head/selection, request acceptance |
| Actual request TPS | Last synthetic lookup20--24 committed tok/s | Real coding workload with learned proposals and held-out quality |
| Ethernet/spare column | Simulator descriptor work only | Firmware/fabric/real-card grid gates before expanded-grid kernels |
| KV reporting | Cache/state correctness gates pass | Classify the original zero-valued metric observation |
| Disaggregation | Experiment design and state prerequisites | Integrated placement/scheduling and responsiveness measurements |

The host suite passes624 tests. CI34174232113 read-only native source audit
passed against the pinned hardware image. Native MLP uses44 configured workers
for gate/up and33 for down. Actual occupied cores are39 for272 output tiles
at7 tiles/core, and32 for160 tiles at5/core. The39-core128-call profile group
matches two gate/up calls across64 layers. The32-core128-call group combines
64 MLP down calls and64 attention/GDN output calls; do not label it all MLP down.
The96-worker recurrence specification in gdn_vsplit.py maps to the48-call
generic group. Fusion passed the corrected fused-SiLU native control at all six
row widths in simulator005625Z with verified source/device operands; hardware
34176158014 also passes correctness but eager fusion regresses. Captured replay
is the next simulator prerequisite, not an assumed speedup. Historical source-integrity failures remain
unresolved; a current pass does not erase those failed reports.

## Historical measured results - 2026-09-07

Hardware34109548525 passes full projection and learned normalization; every
projection output matches TTsim. A shared-fabric gather-plus-FP32-add simulator
test now reproduces the host sum exactly on both chips. Native reduce-scatter
failed FP32 equality and is not adopted. Next connect live projection outputs
to collective/norm without intermediate host serialization, then validate CI.

The simulator fabric-startup blocker is resolved: a missing adjacent cluster
descriptor caused isolated simulator instances. Opt-in shared BDF loading now
passes FABRIC_1D startup and changed-input traces. This enables simulator-first
collective tests; it is not yet a collective, ETH-dispatch or spare-column result.

Full-width simulator projection now passes all5120 outputs on both chips;
learned RMSNorm on its host-reduced result passes within one BF16 ULP. The
80-core grid is simulator-tested, not yet a target-model throughput result.
Hardware suite `feature-projection-full` is prepared. Fabric reduction and
trained-drafter layers/acceptance remain outstanding.

Hardware projection diagnostic34107875990 now passes both Kblock100/4 arms,
12 arithmetic-reference checks total; saved first-row values match TTsim.
The float64 discrepancy also reproduces on silicon. This validates the numerical
diagnostic for the first32 learned outputs, not a complete neural drafter.
Next: full5120-output projection, learned normalization and fabric reduction.
The complete projection and norm tensors are now downloaded and hash-pinned.
A simulator-only all-output projection probe uses80 cores per chip; its first
single-row run is pending, not a demonstrated utilization or speed improvement
for the target model. Normalization and fabric reduction are not implemented.

Learned-drafter projection remains simulator-gated: a single active input term
is exact across both chips and rows1/8/32, but two terms already fail three of
six checks at the unchanged float32 tolerance. The sparse controls retain the
full tensor layout and checkpoint coefficients; they narrow the arithmetic
investigation, not certify a drafter or demonstrate a throughput improvement.
Matched operand permutations now locate this discrepancy within eight-product
groups, consistent with the pinned TTsim MVMUL shared-exponent rounding code.
Separating the same two terms by eight positions passes the unchanged gate;
separating by four fails. This is not a viable dense-projection optimization
by itself. Next distinguish ISA-expected error from implementation error, then
measure trained-drafter acceptance with exact target verification rather than
treating an arbitrary float64 comparison tolerance as coding quality.
The new grouped-product CPU diagnostic exactly matches the two-term simulator
case; on the dense12800-term slice it reduces unexplained maximum errors to
0.00067..0.00109. Destination rounding and accumulation order remain to be
modeled before this reference can serve as an implementation-correctness gate.
Update: explicit accumulator modeling reduces this residual below0.000029,
and the dense slice passes the opt-in arithmetic-reference comparison at the
same1e-4 tolerance. Default float64 comparison still fails and stays visible.
The reference is not bitwise exact; next is hardware numerical comparison,
then full-width learned projection and trained-drafter acceptance validation.

Actual matched request run34084598829 passes native token/state/KV checks at
**23.883 committed decode tok/s at4078 tokens and20.608 at16363 tokens**
for eight-row replay. Same-run four-row controls achieve23.929/20.231 tok/s;
both arms share masks and native compact scratch. Width expansion is flat
at4K (-0.19%) and improves16K by1.87%, not a material breakthrough.
Candidate setup-inclusive rates are10.746/10.345 tok/s versus control
11.558/10.691 and regress at both contexts; all these rates exclude
prefill. These are synthetic lookup requests, not coding-quality certification.
The200 tok/s single-stream target remains unmet; serving defaults are unchanged.
Generating128 committed tokens still needs89/99 verifier blocks, including
54/58 single-row blocks. Better wide-block latency alone cannot compensate for
this observed low-acceptance lookup workload. Real-target feature boundary
retry34088626033 passes30 exact eager B1 tap/chip comparisons with unchanged
native logits/state/KV. Observed chip-local features are[1,1,1,2560]. Batched
alignment gate34089347478 also passes90 exact matrices atT8/T16/T32, with
chip-local shape[1,1,T,2560] and unchanged logits/state/KV. Changed-input traced
feature gate34091904236 passes720 exact matrices and24 stale-snapshot controls,
including rollback and corrected continuation. Accepted-prefix publication
gate34094867083 fails retained feature exactness at layer19/chip0 in the first
nonempty-first-prefix fixture. Diagnostic34097205308 isolates overwrite by the
first native correction trace, not the verifier or publication copy. A pool
allocated before all traces passes simulator with a failing late-allocation
control and hardware34099243313:72 publication events,600 retained matrices and
12 empty aborts across all36 fixtures, including native traced continuation.
Eager single-chunk prefill34102267362 also passes60 exact tap/chip matrices at
contexts63/64/65/127/128/129, excluding padding and preserving logits/state/KV
and decode continuation. Learned feature projection, transactional history and
long/chunked/traced prefill remain port prerequisites. These feature gates are
not yet a neural drafter or an acceptance result.

Learned projection packing now has a pinned first32-output-neuron checkpoint
slice and TP2 tap-order tests. TTsim input/weight layouts are exact, but its
float32 matmul does not meet the current numerical gate (maximum absolute error
up to0.02122 across T1/T8/T32); the tolerance remains unchanged and no hardware
projection pass is claimed. Full projection, normalization and drafter execution
are still outstanding.

Full-model parallel attention verification reaches95.884/102.058ms atT32;
retained replay and actual-request integration are exact. Native compact scratch
also passes hardware correctness and component timing; its combination with
eight-row DMA/parallel grouping passes simulator, hardware microbenchmark,
real-weight attention-layer and full-model gate34075945160. Matched T32 static
verification improves96.407->93.218ms at4K and102.225->96.775ms at16K.
These are full-logit block costs, not measured committed decode throughput.
Materially better drafting and lower verification cost are both still required.

## Topology experiments - Ethernet dispatch and fabric weight loading

User topology: one P150A on PCIe x16, the other on PCIe x4 behind a switch,
with QSFP-DD inter-card fabric. Treat dispatch placement and upload routing as
separate experiments, not a single switch or an assumed decode improvement.

Pinned TT-Metal9f9cd4 has an explicit Blackhole Ethernet dispatch descriptor and
loader branch. Unharvested compute grids are130->140 cores; the two-harvested
descriptors are110->120. Confirm the actual per-card harvest masks and exposed
grid before claiming recovered capacity. Source:
[Ethernet descriptor](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml).

Local TTsim runtime constructor probe passed `DispatchCoreConfig(DispatchCoreType.ETH)`
with resolved COL axis. Explicit ETH+COL is rejected by the TTNN constructor;
default WORKER resolves COL. This probe opened no mesh and proves neither
fast-dispatch operation nor compatibility with the active fabric configuration.

The subsequent fast-dispatch mesh probe is not yet an Ethernet pass:

- TTsim20260907T022544Z-412: WORKER with fabric disabled passes all six
  changed-input trace checks on both chips and closes successfully; grid11x10.
- TTsim20260907T022756Z-417: ETH with fabric disabled fails during mesh opening,
  before tensor uploads, at `No core coordinate found at location: (0, 12, ETH, LOGICAL)`.
- TTsim20260907T022254Z-424: WORKER with FABRIC_1D fails during router handshake
  at mesh opening. Fabric coexistence remains a separate prerequisite.

The P300 simulator mock has14 physical Ethernet cores and eth_harvesting_mask288
on each chip, leaving12 logical Ethernet cores. The stock dispatch descriptor
lists14 logical cores. Native `core_descriptor.cpp` filters active fabric cores
but does not filter unavailable harvested cores; the allocator then translates
every listed dispatch core and fails at logical12. This is simulator evidence,
not confirmation of the physical P150A Ethernet harvesting masks. Do not clear
harvesting bits or fabricate an MMIO topology to make this test pass.

Before a native fix, account for descriptor caching too: the current cache key
contains product, dispatch/fabric configuration, queue count and dispatch mode,
but not chip identity or Ethernet availability. A per-chip availability filter
must not reuse another chip's dispatch-core list. Preserve worker dispatch and
active-fabric exclusions, and validate differing chip availability in addition
to the symmetric mock. The bounded probe is `scripts/ci/dispatch-probe.py`, with
`optimisation/sim/run-dispatch-probe.sh` selecting fast dispatch explicitly.

Native follow-up, simulator only:

- `eth-dispatch-harvesting.patch` filters against actual logical Ethernet cores,
  uses the Ethernet grid for relative dispatch coordinates, and keys Ethernet
  descriptors by chip identity. It compiles; worker probe20260907T025320Z-567
  passes all six checks. Ethernet probe20260907T025332Z-613 instead reaches
  `Expected logical cores to match across user exposed devices` during mesh
  opening. Distinct active fabric ports make per-chip idle lists differ.
- `eth-dispatch-common-pool.patch` intersects availability across user-exposed
  chips and excludes their combined active Ethernet cores, retaining the runtime
  consistency guard. Both patches compile together with simulator library hash
  `dff38587220fdb1a1e651deebe7fba4c85cf63449ea29a41337a43e5f758517b`.
  Worker probe20260907T025734Z-293 passes all six checks with grid11x10.
- Ethernet probe20260907T025739Z-413 gets past those mapping checks, but emits
  native ELF-size errors during mesh opening: idle-ERISC dispatch code0x3594
  bytes and prefetch code0x47fc bytes exceed the0x2a90-byte region. No Ethernet
  transfer, trace, expanded-grid or throughput pass is established. Do not
  enlarge the region or disable its guard without validating the memory map.

`optimisation/sim/build-eth-dispatch.sh` audits source hashes, backs up libraries,
applies both patches and builds locally. Its `harvesting` argument supports the
audited intermediate source state only. Install both resulting native libraries:
the build output and runtime `lib/` copies are distinct in this environment.
Probe reports include native-library and mock-topology hashes, so source edits
alone cannot masquerade as an executed fix. The host CI-script suite passes407
tests in WSL. These patches are not wired into hardware CI or serving defaults;
Ethernet firmware sizing and fabric coexistence remain prerequisites.

- D1a: resolve Ethernet firmware sizing, then matched WORKER/ETH dispatch health,
  trace replay, fabric collectives and actual per-chip core-grid enumeration.
  Audit Ethernet resource assignments; do not assume idle dispatch resources
  are interchangeable with active fabric channels. No kernel grid expansion
  is eligible until this prerequisite passes on the actual cards.
- D1b: spare-column kernel work starts immediately after D1a, independently of
  weight-loading D2. Compare WORKER/110, ETH/110 and ETH/120 using measured
  available grids: first isolate dispatch placement at the same compute grid,
  then isolate the extra column under identical dispatch. Prioritize MLP and
  QKV matmuls atB1/T8/T32. Compare extra compute workers with dedicated DRAM
  prefetch workers and double buffering, preserving precision and operation
  ordering. Check DRAM bandwidth, L1 footprint, worker utilization and complete
  layer latency; do not assume a larger grid helps a bandwidth-bound kernel.
- D1c: promote a D1b winner only after native-exact model/state/KV and rollback
  checks, then matched actual-request timing and held-out coding quality.
  Keep drafter policy, weights, sampling, context and trace setup identical.
  Report setup separately from committed decode throughput. Simulator timings
  do not certify physical DRAM bandwidth or a decode-rate improvement.
- D2: independently compare direct per-card PCIe uploads with x16-card staging
  followed by fabric distribution into the final TP weight shards. Verify
  byte-exact destination shards and measure PCIe/fabric traffic, bandwidth,
  temporary DRAM use and total load time. No supported routing shortcut is
  certified yet; do not fabricate a hardware MMIO topology using simulator YAML.
- D3: report model loading, prefill, setup and committed decode separately.
  Current traced decode consumes resident device weights, so faster host uploads
  do not by themselves remove the per-token device-DRAM bandwidth bottleneck.

Simulator first for new kernels; hardware dispatch/PCIe/fabric performance needs
isolated CI measurement. Preserve serving defaults and the x16/x4 topology.

## Earlier request-engine checkpoints

T32 changed-input replay and synchronized
publication passed run34041390068 (36 two-block fixtures). Device force-argmax
passed34040918809; T32 verification plus readback is131.190/148.752ms at4K/16K.
These component results do not establish200 committed tokens/s. The request pilot
uses actual lookup proposals, exact native token/state comparison, complete
proposal-to-commit timing and separately reported per-request capture cost.
Repeated-code pilot prompts cannot establish representative coding quality.

Actual lookup pilot34045603132 now passes exact native token/state/KV checks
but yields only21.415/17.741 committed decode tok/s at4078/16363-token contexts,
with3.701/3.234s request-specific setup excluded from those rates. Acceptance
is39/537 and30/658 proposed tokens. This confirms the reusable request pipeline,
not a useful general lookup speedup or200 tok/s. Further kernel work and better
proposal economics are required; defaults remain unchanged.

External-drafter width audit (2026-09-07): the official
[DFlash2 config](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/config.json)
declares block_size8. The [authors' Qwen comparison](https://inco.ai/blog/dflash2/)
also uses width8 and reports mean acceptance4.39 on HumanEval and4.79 on MBPP
under default sampling, not our greedy TT backend. Width32 verifier support
does not certify a width32 neural draft. At our measured T8 verifier/selection
cost72.660/77.054ms, even perfect8-token acceptance with zero draft/commit cost
caps this configuration at about110/104 committed tokens/s. Reaching200 at
width8 requires at most40ms for the ENTIRE cycle. Thus a drafter port alone
cannot meet the target on the current verifier; further kernel acceleration
or a separately validated wider high-acceptance strategy remains necessary.

## Programme checkpoint — 2026-09-06

Latest kernel checkpoint: run34019033933 (`36b9eed`) passed synthetic multi-token
recurrence plus fused norm/gate on both cards after simulator-first debugging:
30 exact eager/trace cases, 216 restored-prefix continuations and 15 negative
controls. The CB5 counter handoff fix is hardware-validated. No timing was measured;
full-model integration remains next. Historical entries
below explain the earlier stalls and recovery, not the current kernel status.

Latest integration checkpoint: native serial convolution/gates feeding the device-loop
recurrence/norm passed seven simulator fixtures (seed0 T1/2/4/8/16, seeds1/2 T2),
including every output and recurrent/convolution prefix. Run34019928513 (`8e75a95`)
then passed all30 real-weight native-oracle eager/trace hardware cases. Composed
state rollback and two-step corrected continuation passed simulator T1/T2/T4
checks and hardware run34021612139 (`241be95`):216 two-step prefix-continuation
cases and15 stale-state controls passed. Full-layer run34022668338 (`6b7d2b2`)
also passed native output projection/fabric reduction and all correctness checks.
T16 mean layer time improved5.176 to3.372ms against native serial; T1 regressed and
remains native. Full-model run34023117059 (`6a2ca23`) passed logits/state/KV/rollback
but regressed at every T>1 width: T16 was210.138/218.918ms versus173.072/181.859ms
at4K/16K. It is not adopted. Compact-prologue follow-up34024642720 (`8cb31c8`)
passed the full correctness matrix and reduced T16 to163.227/172.023ms against
173.067/181.856ms paired controls. T8 also improved; T2 regressed and T4 was marginal.
Experimental routing now keeps the previous path below T8. These are fixed verifier
blocks with a preselected checkpoint, not dynamic speculative committed throughput.
This is not convolution token-loop fusion or a committed-throughput result.

Newer checkpoint: parallel causal convolution windows use the unchanged native
conv/gates kernel once across token rows, with aligned DMA window construction.
Run34027510486 (`de6fd22`) passed full logits/state/valid-KV/rollback and reduced
T16 verification to94.882/103.655ms at4K/16K, versus163.262/172.040ms paired controls.
T8 improved to74.177/78.592ms; T1/T2/T4 retained previous paths. Packed convolution
histories then passed native-oracle layer run34028407207 (`7a20340`), including
every rollback prefix, and reduced T16 layer time3.006 to0.912ms. Its full-model
integration is the next gate. This removes prefix materialization, not arithmetic;
engine-level dynamic acceptance/commit and executable coding evaluation remain open.

Packed histories passed34028729821 and ordered cache writes passed34034319922:
T16 is now89.552/98.325ms at4K/16K. All multirow widths improved; T1 stays native.
Post-verification greedy commit passed34029984214. Fused96-worker publication
passed34034469074, including every inactive native slot. Captured commit passed
34036778172: selected publication plus binding guards/synchronization costs
1.852-4.571ms, with setup separately recorded. These are still forced-draft
correctness fixtures, not a measured end-to-end speculative coding engine.

T32 now has full-model exactness/timing evidence in34038451865:128.738ms at4K
and146.288ms at16K, with24 width/mode checks and16 corrected rollback cases.
The workflow failed only its final outdated10-versus12 timing-fixture counter;
that bookkeeping check is fixed, but the original workflow is not a green run.
Ideal perfect-acceptance/no-overhead ceilings are248.57/218.75 tok/s. Actual
committed throughput and coding-quality evaluation remain open; target200 is
not yet achieved. T32 dynamic publication and request-trace reuse are next.

Active implementation: `ci/qwen-hardware-correctness` (PR #7), in the
`Tenstorrent.Qwen-Runner-CI` worktree. Other branches may contain older versions
of this programme. The [execution ledger](experiment-execution.md) records individual
passes, failures and timing evidence. No serving defaults have been promoted.

| Track | Verified progress | Remaining programme gate |
| --- | --- | --- |
| E0 baseline | Pinned runtime and card-backed endpoint benchmarks | Complete engine-commit accounting and gateway comparison |
| E1 cache | Nonzero occupancy observed; exact full-model active state/valid-KV checks | Full serving lifecycle, cancellation and slot-reuse coverage; historical zero not reproduced |
| E2 scheduling | Boundary and mixed-traffic interleaving gates passed | Broader load, long-context, cancellation and repeatability sweeps |
| E3 verifier | Exact T32 verification, dynamic commit and reused request pipeline | Substantially lower V(T), multi-request trace lifecycle and representative repeatability |
| E4 fusion/pipeline | Parallel attention full-model34067681095 exact: matched T32 verification95.88ms at4K,102.06ms at16K. Retained replay34070163839 and actual requests34072489815 pass. Scratch rebuild34070379248 repaired and exact, T32 component cost1.108->0.716ms /1.524->1.020ms | Combine scratch with DMA/parallel groups simulator-first; component gains do not establish committed throughput. Preserve native T1/2/4 and bounded family routing |
| E5 drafting | Matched replay requests34072489815 exact at23.763/20.127 decode tok/s versus22.799/18.588 control; setup-inclusive rates regressed to10.284/9.524. Native-tape48-policy lookup analysis completed; historical MTP groundwork | Faster verifier and materially better drafting required:128 committed tokens require89/99 verifier blocks. No DFlash2/DSpark/EAGLE3 TT adapter certified; synthetic lookup is not coding-quality evidence |
| E6 coding/adoption | Exactness maintained on experiment fixtures | Freeze 200-task executable corpus; quality, serving lifecycle and adoption gates |
| E7 prefix reuse | Dependency analysis | Validated hybrid-state reuse and request isolation |
| E8 precomputation | Planning and cost investigation | Measured table/precomputation candidate; no LUT gain established |
| E9 spare cores | Profiles, DMA experiments and guarded force-argmax results | Persistent L1 recurrence, dedicated staging and broader worker mappings |
| E10 disaggregation | Capacity/feasibility analysis | TP1 feasibility, hybrid-state handoff and actual split-workload benchmarks |

Latest verified full-model T16 block costs (run **34034319922**, `2391339`) are
**89.552 ms at 4095 tokens** and **98.325 ms at 16383 tokens**. Full logits,
active GDN state, valid KV and corrected rollback passed. These are static verifier
costs, not committed-token throughput: even perfect acceptance with zero drafting
and commit overhead gives only about178.67/162.73 tok/s for T16. The 200 committed
token/s goal is not met; a sixteen-token cycle must fit within 80 ms including
drafting and commit. T1 retains native handling.

Post-verification record lifetime and greedy commit passed run34029984214:
all48 layers retained,16 post-output decisions across4K/16K eager/trace, including
abort and rejection corrections. Readback/selection/eager commit alone costs
25.45-41.57ms; this is not an actual drafter or reusable serving engine.
The ordered cache writer passed hardware layer gate34033619168:60 exact cases,
30 negative-control pairs and90 paired timing blocks. T16 attention improved
0.722 to0.592ms; T1 regressed and remains native. After correcting and simulator-
certifying1024-column metadata, full-model run34034319922 passed all correctness
and timing gates: the real full-model gain is about2%, not21%.
Attention retains exact B1 SDPA; GDN recurrence remains sequential inside its
device loop. Fused96-worker publication passed full-model hardware certification;
warm eager commit costs7.35-9.09ms, while full-logit readback remains about8ms.
Target200 remains unachieved, and serving defaults remain unchanged.

Historical recurrence-kernel development:
The first recurrence-only prototype retains BF16 state between tokens in L1 and
passed hardware compilation/exactness in run 34012883902 (`855f8de`): 30 eager/trace
cases, 216 restored-prefix continuations and 15 stale-state controls. No timing was
measured in that run. Paired recurrence timing passed in run 34013199242: 1800
replays, T16 medians about 0.532 ms serial versus 0.340 ms device loop. All nine T16
paired blocks favored the candidate, with timing spikes; no full-model speedup is
established. The norm/gate extension timed out in run 34013517498 after T1 passed;
larger widths remain uncertified. A bounded stage/stack diagnostic retry keeps
the same CB5 feedback and head-local assembly kernels to identify the blocking call.
That retry (34014926676) instead stalled reading the initial tensor, before either
oracle or custom kernel. A minimal paired-card transfer health check is now staged;
the original kernel stall remains unresolved. That health check (34015497253)
failed during firmware initialization, before transfers. The operator has now
authorized a controlled two-card reset followed by the same health check. Recovery
passed in run 34016364842: reset/reinitialization succeeded and all 12 transfer
checks plus clean mesh close passed. The unchanged instrumented norm/gate test is
next, without another reset; the original kernel stall remains unresolved.
Real-weight convolution and full-model integration remain; this is not yet a
complete multi-token GDN layer or speculative serving.

Kernel development is now **simulator-first**: new device-loop changes must pass
local liveness/output/prefix-state gates before silicon native-oracle, trace and
performance validation. Run34016749007 reproduced T2 candidate warm-up stalling
after native reference success on recovered cards. The operator requested another
CI reset; run34017283126 reset/reinitialized both cards and passed all12 transfer
checks. Cards remain idle while the same generated kernels are tested locally in
TT-Sim. See [simulator entry point and limits](../optimisation/sim/README.md).
Local debugging identified a missing packer-private CB5 tile-count handoff after
the reader's initial-state push. Seeding that count fixed T2 liveness and preserved
exact output/prefix states in simulation. Runtime-header hashes now guard that
dependency; all15 fixtures in the three-seed T1/2/4/8/16 simulator matrix passed
exact output/prefix-state comparisons on both chips and clean close. Hardware
native-oracle/trace validation and any performance benefit remain unverified.
Keep E5 end-to-end deployment gated on verifier economics; do not label host drafter
tests or oracle verification as real coding throughput.

## Initial execution context

Status: execution started, 2026-09-05. Hardware operator prerequisites passed in
[run 33941853075](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33941853075):
80 fused-kernel reference cases, 224 recurrent/KV-state checks and 8 eager/traced
QSFP-DD all-gather output checks across both cards. These do not establish full-model
quality, scheduler correctness or throughput. See [execution ledger](experiment-execution.md).
Local simulator infrastructure now passes single-chip arithmetic, dual-chip sharding
smokes and 40 standalone Qwen kernel reference cases; see
[setup and scope](../optimisation/sim/README.md). This does not mark E0-E10
complete. We are offsite: physical-card measurements require a verified card-backed CI
runner, not simply an image build.
Keep Qwen3.8-27B and exactly two P150A cards. Optimise single-stream coding performance
without silently changing sampling or sacrificing correctness. Treat 200 committed tokens/s
as a stretch target, not a promised result. Existing results are historical controls, not
proof of the currently running image's performance.

## 1. Execution rules and evidence

- E0 and E1 establish runtime and measurement truth before any performance changes.
- Use an isolated endpoint and exclusive access to the pair for device experiments.
  Do not run existing rig scripts blindly: some remove containers, use fixed device IDs,
  private image tags and mutable host-side patch directories. No production restarts or
  device resets without an agreed maintenance window.
- Record repository commits, image digest, model/tokenizer revision, TT runtime/plugin/vLLM
  revisions, firmware, board IDs, PCIe widths, measured fabric links, flags, sampler,
  trace buckets, context limit, KV dtype, pool size and host load for every arm.
- Freeze the latest verified A/C/D/K/M stack as control. Record whether shard-greedy is
  absent. Never enable that path on requests with nonzero temperature; preserve request
  sampling semantics. Keep current precision fixed in the first experiments.
- Warm each shape before timing. Use at least three interleaved A/B/B/A blocks, identical
  inputs and generation budgets. Report each run, paired changes and variability, not
  only a winning run. Profiling runs are separate from unprofiled acceptance runs.
- Store each run under an operator-selected output root as `<experiment>/<run-id>/`:
  manifest.json, requests.jsonl, metrics.prom, server.log, results.json, and profiler
  output when applicable. Keep prompts, outputs and credentials out of public commits.
- Results must distinguish planned, failed, correctness-passed, benchmarked and adopted.
  Failed correctness blocks adoption regardless of speed. A performance win needs a
  positive paired effect beyond observed run variability and confirmation end to end.

### Measurement contract

Start with concurrency 1 and prompt lengths 128, 4096, 32768 and 65536 tokens, reserving
space for generation inside the configured context limit. Count the final templated
prompt with the pinned tokenizer; reduce input length where necessary. Generate 1024
tokens for throughput runs, with an explicit ignore-EOS capability check; otherwise
report actual termination and output count. Use separate natural-stop coding evaluations.

Measure request-to-first-generated-token TTFT, total request duration, engine committed
token count/timestamps, steady-state decode tokens/s, and client text-event gap p50/p95/p99/max.
Count reasoning and answer tokens separately where supported and report their total.
Define engine decode rate as `(N - first_commit_count) / (last_commit_time - first_commit_time)`;
mark it unavailable if the window is empty. Also report all committed tokens over the
complete generation phase, including drafting, verification and commit, for speculative runs.
Never equate an SSE event with one token: speculation, UTF-8 buffering and proxies can
coalesce output. Endpoint usage gives total output counts, not exact per-token timing.
If engine commit timestamps are unavailable, label client measurements as estimates and
do not use them alone to certify the 200-token/s target.

Compare container-local and gateway paths sequentially using the same workload. Report
concurrency 2/8 and aggregate throughput separately; neither can satisfy the single-stream
goal. Replace or extend `optimisation/rig/bench_itl.py` before speculative endpoint scoring:
it currently counts text-bearing events and averages per-stream median gaps.

## 2. E0-E2: baseline, cache truth and responsiveness

### E0: reproduce the current baseline

Inspect the running container and its actual imported model/plugin files, not merely the
Dockerfile. Verify optimisation engagement and trace replay. Run the measurement matrix
above and save logs plus raw metrics before, during and after requests. Compare to the
historical 36 ms/token local probe and the traced demo independently; their workloads and
timing definitions differ. If reproduction fails, resolve configuration or measurement
differences before attributing gains to new kernels.

Exit: reproducible B=1 control, verified runtime identity, reliable token accounting and
a bounded estimate of gateway buffering/overhead. No requirement to reproduce an old
number exactly under a different workload.

### E1: explain the reported 0% KV cache

Do not assume zero means cache is disabled. These are separate quantities:

| Quantity | Meaning | Current repository evidence |
| --- | --- | --- |
| Attention KV | Per-request K/V history for 16 full-attention layers | Paged cache path and pool sizing are documented |
| GDN state | Recurrent and convolution state for 48 hybrid layers | Separate persistent buffers, not ordinary growing attention KV |
| Prefix hits | Reuse of a previous request's prompt computation | Explicitly disabled; model declares no prefix-cache support |
| KV occupancy | Used scheduler blocks divided by configured pool blocks | Must be checked against the live plugin and metrics exporter |
| Allocated device memory | Weight/cache/trace/scratch buffers reserved on the cards | Not equivalent to occupancy or a prefix hit rate |

The repository's `--no-enable-prefix-caching` is not a switch that disables attention KV.
The generic vLLM metric `vllm:kv_cache_usage_perc` is a fraction from 0 to 1, not already
a percentage. Metric names and TT propagation must be verified at the deployed revision.
See [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/) and
`gotchas.md`'s prefix-caching section. These are hypotheses until live evidence is captured.

Procedure:

1. Obtain the exact zero-valued log line or dashboard series, its timestamp, labels,
   query and source endpoint. Distinguish prefix hit rate, occupancy, obsolete GPU
   metric names, missing-series-to-zero conversion, rounding and post-request sampling.
2. Scrape `/metrics` at 250 ms intervals on the isolated endpoint, retaining raw values
   and request lifecycle timestamps. Compare to dashboard values. If export updates
   less often, extend the request long enough to cross multiple export intervals.
3. Run idle -> 4k prompt with sustained generation -> idle, then a 32k prompt and two
   concurrent 16k prompts, within verified capacity. Keep prefix caching off. Record
   running/waiting requests, allocated/free blocks, cache gauge and preemptions.
4. Read the live block allocator and model adapter to derive the exact occupancy
   numerator/denominator, block padding, reserved blocks and any per-group accounting.
   Cross-check `get_num_available_blocks_tt`, `get_max_tokens_all_users`, page tables
   and the scheduler-stats-to-exporter path. The repo documents model-based sizing,
   not GPU-style free-memory profiling. Never guess total blocks from card memory.
5. For a simple single-pool configuration, predict occupancy from allocated request
   blocks divided by available pool blocks, allowing measured reservation/padding.
   For illustration only, 4k live tokens in a 524k-token pool is about 0.8%, despite
   a large preallocated device buffer. A short request or idle scrape can look like 0%.
6. In a diagnostic model run, inspect attention cache writes, decode cache reads,
   positions and page-table mappings. Compare cached continuation logits against a
   whole-prefix reference at the same positions and precision. Test positions 63/64/65
   and 2047/2048/2049, longer continuation, cancellation and slot reuse. Instrumentation
   must be out of performance runs and must not emit customer cache contents.
7. Repeat an identical long prompt in separate requests. With prefix reuse disabled,
   no cross-request hit is expected; lower second-run TTFT could instead be warm traces
   or kernels. This is distinct from KV reuse within each request's decode.

Decision table:

- Prefix hits zero with prefix caching off: expected; record missing prompt-reuse capability.
- Idle/rounded occupancy or oversized denominator: correct presentation/capacity reporting;
  do not increase occupied memory just to make a utilisation chart look better.
- Allocator nonzero but exported gauge zero: repair metric propagation or dashboard mapping.
- Invalid cache reads/writes, positions or state reuse: correctness defect; stop speed work
  on that path and repair it first.
- Valid within-request cache but repeated prompt recomputation: evaluate E7, not a KV toggle.

Exit: an evidence-backed classification of the observed zero and a passing cache lifecycle
test. Do not claim the issue resolved from configuration inspection alone.

### E2: resumable prefill and fair decode scheduling

Offsite prerequisite: [prefill state operator gate](../optimisation/sim/PREFILL-STATE.md)
compares carried GDN recurrence/convolution and paged KV writes using synthetic
TP=2-local-shape inputs. Its 8-run matrix passed 112 checks, with exact chunked/whole
matches and all 24 deliberate state/page-offset faults detected. It does not exercise the model entry point or enable plugin
chunking; the unmodified pinned wrapper does not forward the continuation position.
The [opt-in continuation prototype](../optimisation/sim/CONTINUATION.md) now patches
the disposable checkout and passes 22 host tests, including extracted upstream
method bodies with a fake TTNN runtime. It remains disabled: device traces,
full-model prefill-to-decode numerics and actual scheduler isolation are unvalidated.
The [single-lane plugin integration](../optimisation/sim/INTERLEAVE.md) additionally
passes 16 host tests for admission, alternation, cancellation and generation tracking;
the endpoint harness passes four stream-accounting tests. Real-card CI begins with
an inventory-only workflow, not an assumed compatible serving image or a throughput claim.

Implement the model continuation contract before enabling plugin chunking, following
`lever-N-prefill-decode-interleave.md`. Validate the exact deployed input contract: chunk
end versus full prompt length, explicit final-chunk indication, RoPE range availability,
scratch ownership, exception cleanup and request cancellation. Keep one partial prefill
per scratch owner initially; do not let a short arrival reset another request's state.

Compare whole prefill against scheduler-visible 2048-token chunks with 1/2/4 decode
steps between chunks. Start seven decoding streams in an eight-slot endpoint, then inject
a 45k prompt; repeat with one active decoder. An eight-decoder/full-capacity case is a
separate queueing test, not evidence of prefill interleaving. Also admit a short prompt
during a partial prefill to verify safe waiting and eventual completion.

Gate: chunked/whole greedy equivalence, stable state through cancellation/slot reuse,
no starvation, no idle-workload decode regression beyond variability. Report long-prompt
TTFT and existing-stream p99/max gaps; expect chunk-sized pauses, not uninterrupted 5 ms
tokens. Smaller chunks require separate trace/state support and are not a flag-only change.

## 3. E3-E6: single-stream throughput experiments

### E3: verification cost curve before investing in a drafter

Current bounded evidence: `gdn-prefix` and `gdn-block` CI runs validate one real
GDN layer with batched input/output projections and exact every-prefix state plus
two-step rollback continuation on both chips. Each arm passed 216 cases and 30
stale-state negative controls. Separately, `full-prefix` run 33999532634 passed 102
full-model serial rollback cases across the 64-token page boundary, six eager/trace
baseline comparisons and both stale-GDN/wrong-page controls in six configurations.
  Attention-layer runs 34000512864/34000694699 additionally passed 60 exact
  eager/trace cases each with native QKV/SDPA/output batching and ordered B1
  shared-page KV writes. Timing measured T=16 at 0.5218 ms versus 4.4063 ms serial
  (8.451x median speedup), one real-weight layer at short context only. Static
  position fixtures are not a full-model/device-dynamic verifier integration.
  Integrated `full-batch` run 34001403540 subsequently passed 30 full-model
  T=1/2/4/8/16 eager/trace cases and all 102 T=16 prefix rollback cases, at prompt
  lengths 63/64/65, with exact active-token logits, all 48 active GDN states and
  valid KV prefixes across all 16 attention layers. Per-layer candidate snapshots
  are independent of the reference buffers. Six baseline mode checks and six
  stale-GDN/wrong-page negative-control pairs also passed. This remains a static,
  short-context correctness gate: full-model timings, coding-length contexts and
  a device-dynamic serving/drafter integration are still required.
  Coding-context run 34002876975 subsequently passed 20 width/mode and 16 selected
  rollback cases at 4095/16383 tokens, plus paired timings. Batched SDPA caused
  long-context drift; retaining B1 SDPA reads restored exactness while projections
  stayed batched. T16 full-logit blocks measured 235.098/243.876 ms versus serial
  701.383/710.095 ms (2.983x/2.912x median paired speedup). Separate active-state
  restore costs about 0.867 ms. Timings include one preselected end checkpoint,
  not dynamic all-prefix selection/commit. Even perfect acceptance and zero other
  overhead imply only a 68.06/65.61-token/s bound for this target path, not 200.
  Prioritize reducing target block time below 80 ms before relying on a neural
  drafter to approach 200 committed tokens/s; this is not a hardware-wide ceiling.
  The full-model serial oracle verifies full logits, active GDN states and valid KV values, not batched target
execution or safe concurrent writes to shared KV pages. Keep the old multi-token
harness diagnostic-only; its composed operations
are not the current native fused control. See the execution ledger for run IDs.

Use the existing multi-token GDN harness and latest fused control. Define T as target
verification rows, including the seed row; K=T-1 draft proposals. Sweep T=1/2/4/8/16.
Existing code already batches projections across rows; investigate remaining serial
conv/recurrence/normalisation and layout work rather than assuming no weight reuse exists.

Generate known continuation tokens from the control, then verify them from identical
snapshots. This removes draft acceptance as a confounder but is an oracle diagnostic,
never an end-to-end speed claim. Measure target verify time, state snapshot/commit cost,
readback, collectives, projection time, memory traffic where observable and peak memory.
Use traced measurements and separate synchronized attribution runs to expose work
misattributed to the next host synchronization.

Gate every prefix length: token/logit agreement, attention position/mask correctness,
GDN recurrent and convolution state after commit, and continuation after rollback.
Use deliberately wrong-state and forced-rejection controls. Assert failures: the existing
generation harness reports output divergence without asserting it, so its successful exit
is not a correctness certificate. Do not call tolerated numerical drift lossless.

Exit: measured V(T) and commit curve with passing correctness. Report optimistic
`T / (V(T) + commit(T))` only as a no-draft upper-bound diagnostic. If that cannot approach
the desired rate, improve verification before searching for a better drafter.

### E4: multi-token fused-state and memory pipeline prototypes

One change per arm, in this order:

1. Extend the current conv/gates and recurrence fusion to T rows, retaining per-head
   recurrent state through the token loop and producing accepted-prefix state snapshots.
2. Fold compatible norm/gate/layout stages into that path without changing baseline
   rounding; remove avoidable intermediate device-memory round trips.
3. Prototype bank-local contiguous weight reads and double buffering on the actual
   TP=2 gate/up/down and GDN projection shapes. Compare to the tuned 1D path at T=1/4/8/16.
4. Only where read-stall evidence supports it, test separate reader/staging cores and
   consumer placement. Record L1/circular-buffer capacity, bank/NoC contention, transfer
   size, effective bandwidth, unpack and compute time. Spare cores do not create bandwidth.

Keep negative stock grid/DRAM-sharding/packed-gate experiments as controls. Those results
close their tested configurations, not every custom pipeline. Reject a microbenchmark
win that loses after conversions, state snapshots or collectives in the full trace.
Preserve trace buffer addresses, request isolation and original precision. No lower-bit
weights or state in this phase. Stop an arm after a correct full-path result shows no
repeatable gain; retain the negative result.

### E5: real coding draft proposals

#### Drafter comparison update (2026-09-06)

Do not restrict the experiment to MTP. Neither DFlash2 nor DSpark has a recorded
P150A hardware result in this repository. Lookup has an integrated exact request
pilot at21.415/17.741 committed decode tokens/s, not a representative coding-quality
result. The older EAGLE3 discussion is analysis,
not evidence that an EAGLE3 drafter ran here.

| Arm | Upstream implementation to audit | TT port work beyond the shared verifier |
| --- | --- | --- |
| DFlash2 (first external drafter candidate) | [Official checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), [code](https://github.com/z-lab/dflash), [configuration](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/config.json) | Five-layer draft backbone, 2048-token sliding attention, two-tap dynamic convolutions, top-16 candidate path selector of rank 256; checkpoint uses target feature taps 5/19/33/47/61. |
| DSpark | [RadixArk Qwen3.8-27B checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark), [method paper](https://arxiv.org/abs/2607.05147) | Five full-attention layers, BF16 1.86B-parameter draft, VanillaMarkov rank-256 head; the card specifies the same target feature layer indices. Audit exact feature definitions, not just matching indices. |
| MTP | Existing checkpoint and local speculative branch | Repair trace integration and compare on the same verified target path. |
| Request-local lookup | Existing host-tested proposal policy | Device verification/commit integration; no model or cross-user cache. |
| EAGLE3 (compatibility-gated) | [Official implementation/checkpoint list](https://github.com/SafeAILab/EAGLE), [related Qwen3.6 PRISM head](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3) | No Qwen3.8-27B-specific public head verified in this audit. The related 3.6 head uses feature taps 1/31/60 and offers full/compressed vocabulary variants. Transfer requires explicit feature/tokenizer/map compatibility and measured acceptance, or retraining; it is not established 3.8 support. |

Additional candidate families, not runnable Qwen3.8 TT implementations:

- Request-local suffix/retrieval drafting extends lookup without draft weights.
  Keep any global cross-request suffix cache disabled for this experiment. It is
  a low-cost candidate for edits/refactors, not an assumed fresh-code speedup.
- A small independent autoregressive draft model avoids target-feature capture
  but adds its own weights/KV and sequential drafting. Require token alignment;
  cross-vocabulary adapters are additional port work, not implicit compatibility.
  [vLLM method and vocabulary documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).
- PARD is another parallel-draft family. The documented AMD checkpoint targets
  Qwen3-4B/8B, not a verified Qwen3.8-27B setup. Treat it as an adaptation/training
  track. [PARD documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/parallel_draft_model/).
- Medusa/Hydra-style trained heads are another architecture option, not ready
  Qwen3.8 checkpoints verified here. [Hydra implementation](https://github.com/zankner/Hydra).

EAGLE3 remains a meaningful comparison: a smaller/compressed draft may save memory
traffic, while sequential draft calls can cost latency. That tradeoff must be
measured against parallel block drafting, not decided by paper headline speedups.
Prioritize the published 3.8 DFlash2/DSpark checkpoints and lookup/MTP controls;
keep EAGLE3 as a compatibility/transfer candidate rather than silently substituting
a 3.6 head. Engine support for an `EAGLE`/MTP flag is not evidence of an external
EAGLE3 checkpoint or a TT backend port.

The upstream MLX implementation at
[`07ebd93`, target hooks and generation loop](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
provides a concrete feature-capture contract for the next port: zero-based layer
IDs index decoder layers, hooks save each layer's returned hidden output, and
those outputs concatenate along the hidden dimension in configured tap order.
They are not GDN recurrent states or final-normalized logits inputs. The draft
applies its learned feature projection and hidden normalization. Prefill retains
a bounded hidden suffix when configured; verification captures new block
features. A TT adapter must establish the same post-layer residual boundary,
token alignment and accepted-prefix feature ownership before draft execution.
The public implementation is a reference, not a TT-certified feature adapter.

The [DFlash2 authors' Qwen3.8 model-card comparison](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
uses one H200, SGLang, default sampled xhigh reasoning and seven proposals per
verification. At concurrency one, reported HumanEval end-to-end output rates are
69.0 (ordinary), 151.9 (MTP), 159.9 (DSpark), and 214.6 (DFlash2) tok/s; MBPP gives
69.0, 153.1, 163.3 and 226.9 respectively. These are publisher measurements, not our
greedy control, not TT-Metal support and not speed forecasts for two P150As.

Inference: DFlash2 is the stronger first external candidate for the coding workload,
but port cost, acceptance against our BF4/BF8 target and draft/verify wall time must
decide. Both external drafts need feature capture and extra device memory; neither
is a flag-only extension of the current TT plugin. Keep the target TP2 across both
cards. Audit checkpoint/runtime revisions and exact feature semantics before any
port; never execute downloaded custom model code as part of a metadata audit.

The historical `speculative-decoding` branch retains only post-final-norm hidden
states for MTP (`patch_hidden_retention.py`); that is not the five-layer feature
interface these external drafters need. Its `spec_generate.py` also explicitly
reports long-generation token divergence without asserting identity. Retain it
as historical diagnostic evidence, not a ready lossless verifier or a current
performance baseline. Audit feature lifetime, trace addresses and accepted-prefix
selection together with recurrent/conv/KV rollback before adapting either drafter.

Sequence: E3 forced-prefix/rollback correctness and T=1/2/4/8/16 verification curve;
then checkpoint conformance and native draft kernels; then matched MTP/lookup/
DFlash2/DSpark coding runs. Report emitted tokens per entire draft+verify+commit
cycle, including rejected proposals and state/feature movement. As an arithmetic
budget, 4–5 committed tokens per cycle requires a complete 20–25 ms cycle to reach
200 tok/s. No GPU multiplier bypasses that measured requirement on these cards.

#### Multiple drafters: staged comparisons

Start with routing, not an unconditional ensemble. The host-only
`speculative-decoding/harness/hybrid_draft.py` policy tries request-local lookup
and, on a miss, calls only the selected neural adapter. Defaults disable drafting
unless greedy mode and verifier readiness are explicitly asserted. All returned
tokens are unverified proposals; only the target verifier may authorize emission.
This is a tested proposal interface, not a TT neural port or a state rollback engine.
Adapters receive a bounded committed-history suffix, not full prompt context or
feature tensors. A future device adapter must own request-local synchronized state,
track the full committed position and catch up after lookup-only cycles. Never
rebuild a long-context neural cache from that truncated suffix alone. Device errors
propagate: do not silently fall back while target/drafter state may be inconsistent.

| Arm | Work per cycle | Entry gate |
| --- | --- | --- |
| Single controls | Lookup only, MTP only, then each compatible external drafter separately | E3 exact verifier and each adapter's conformance gate |
| Lookup + selected neural | Lookup hit skips neural work; miss calls one neural drafter; target verifies either proposal | Request isolation, committed-only history and neural catch-up after skipped cycles |
| Adaptive routing | Select one drafter and block length using measured committed tokens per total cycle time | Enough paired per-workload measurements; include switching, feature and cache costs |
| Hierarchical cascade | Cheap draft proposes to a larger draft, then the target verifies | Intermediate verification semantics, compatible inputs, rollback and incremental memory budget |
| Multi-branch ensemble | Several drafts propose branches for joint target verification | Explicit tree attention plus branch-local GDN recurrence/conv state; no assumed TT tree support |

[Hierarchical speculative decoding](https://arxiv.org/abs/2510.19705) establishes
the multi-model cascade approach, not a speed guarantee on Blackhole. Arbitrarily
chaining EAGLE3, DFlash2 and DSpark is not plug-and-play: their feature interfaces
and proposal mechanisms differ. Keep both cards serving the TP2 target initially.
Spare compute cores do not provide independent DRAM bandwidth for extra drafts.

Record lookup time/hit rate, selected drafter, proposal length, accepted prefix,
target bonus/correction tokens, draft/verify/rollback/commit time, peak DRAM/L1,
and total committed tok/s. Include zero-acceptance cycles and neural catch-up.
Test every rejection position, EOS, cancellation, stale request IDs, slot reuse,
lookup-to-neural transitions and a forced state-corruption negative control before
hardware promotion. Host routing tests do not satisfy these E3 device-state gates.

After E3 correctness, repair MTP trace/buffer integration and its documented eager/trace
stall interaction. Compare speculation off, MTP, and exact token-sequence lookup proposals
from the current request's supplied source/prompt and generated history. No other user's
code may enter a proposal cache. Use longest matching suffix, bounded by K, with most
recent occurrence as deterministic tie-break; no match falls back to ordinary decode.

Sweep K=1/3/7/15 only for verified T buckets. Count accepted draft tokens, seed/bonus
tokens and all committed tokens separately; state the convention in every result.
Include actual draft, verify, rollback, sampling, synchronization and emission costs.
Compare fresh code, small edits, refactoring, tests and tool/JSON output separately.

Start greedy. Require exact baseline-token agreement before describing the path as
quality-preserving. For non-greedy serving, implement and validate a correct speculative
sampling algorithm; never substitute greedy acceptance for a sampled request. Until then,
route sampled requests through unchanged ordinary decode. Add adaptive K only after
fixed-K results: select the measured winning bucket per workload, and fall back to normal
decode when rolling measured speculation cost exceeds the control estimate.

Exit: repeated end-to-end speedup on real coding prompts with passing output/state gates.
Never count proposed or rejected tokens as throughput, and never present the oracle curve
as production performance.

### E6: coding-quality and serving adoption gate

Freeze a versioned 200-task corpus before tuning: 40 new-function tasks, 40 bug fixes,
40 repository edits/refactors, 40 test-writing tasks and 40 tool/structured-output tasks.
Use user-approved repositories/fixtures with executable checks; exclude evaluation data
from tuning. Run generated code only in an isolated sandbox without network or secrets.
Report pass@1, build/test pass, tool schema validity, truncations and latency per category.

For greedy execution-equivalent changes, require identical token IDs plus unchanged task
outcomes. If a numerically different variant is explored later, label it separately and
do not adopt automatically based on a aggregate score: report paired regressions and
obtain a quality tradeoff decision. GSM8K remains supplementary, not the coding gate.

Validate cancellation, EOS/stop handling, context exhaustion, mixed sampling requests,
slot reuse and concurrency 1/2/8. Package a digest-pinned opt-in image with fallback to
the unchanged control. Promote only after isolated serving validation and an approved
deployment window; keep the old digest available for rollback.

## 4. E7: hybrid prefix reuse for repeated coding turns

Run after E1 classification and E2 state lifecycle correctness. This targets repeated-turn
TTFT, not ordinary single-stream decode rate. First inspect live TT plugin support; do
not transplant assumptions from current GPU vLLM or simply remove the unsupported flag.

Prototype block-boundary reuse of both attention KV and all required GDN recurrent/conv
state. Cache identity must include token prefix, model/tokenizer/template revision,
precision and position-affecting settings, plus tenant isolation. Enforce a measured
memory budget, eviction and cleanup; shared system-prefix reuse across tenants is out of
scope. Start with one request resuming its own unchanged prefix, then isolated request reuse.

Test identical prefixes, one-token edits before/after the boundary, different system
prompts, position settings, cancellation and eviction. Require equality to uncached
continuation and no cross-request leakage. Measure snapshot/restore cost, retained bytes,
actual avoided prefill tokens and TTFT. Adopt only if reuse pays for its state-management
cost on repeated coding turns. Do not interpret warm kernel caches as prefix hits.

## 5. E8: lookup tables and precomputation

This is an investigation track, not a claim that a DGX Spark optimisation transfers to
Tensix. Obtain the specific Spark implementation before assessing a direct port. Separate
static model/position data from activation-dependent computation and token proposals.

| Arm | Experiment | Required evidence and gate |
| --- | --- | --- |
| E8a | Audit RoPE tables, position indices, masks, immutable gate constants and layout metadata; stage reusable inputs once and gather/update positions inside the trace | Inventory what is already precomputed; measure eliminated host work and transfers, table bytes and gather cost; exact baseline rounding and context-boundary correctness |
| E8b | Token-sequence lookup drafting from request-local source code and history | Use E5's bounded deterministic proposal policy and target verification; measure proposal latency, acceptance and total committed rate; no cross-user cache |
| E8c | Prototype sigmoid/softplus lookup only if profiling shows material remaining cost | Compare existing fused special-function instructions to an L1 table, including indexing, replication, gather and interpolation costs; measure input-domain coverage and recurrent-state drift; reject approximations from the execution-equivalent ship path |
| E8d | Feasibility study for lookup-based low-bit matrix multiplication | Identify the exact arithmetic/weight format first; include activation-dependent table construction, lookup traffic, unpacking and reductions; compare real TP=2 shapes against tuned matmul, without assuming CPU/GPU LUT speedups apply |

E8a is partly present already: the attention-prep kernel consumes RoPE inputs, the GDN
conv/gates kernel consumes `neg_exp_A`, and the historical plan describes a persistent
device rotation table. Verify current wiring rather than rebuilding these tables under
a new name. The MTP path needs a separate audit because its host RoPE enqueue and trace
interaction were previously expensive.

For E8c, distinguish a fully enumerated exact mapping of a fixed finite input format
from a coarser approximate table. Account for NaN/infinity, signed zero, saturation,
rounding and every supported input dtype. Even an exact table can lose to the existing
special-function implementation because its gathers consume memory bandwidth. Do not
combine table changes with fusion changes in one A/B arm.

Static weights do not make projection outputs static: each token supplies new activations.
Any LUT scheme depending on those activations must rebuild the relevant table or prove
valid reuse. Low-bit codebook arithmetic is not automatically equivalent to the current
block-floating formats. A changed representation requires E6's separate quality decision.

Priority: E8a during E3's trace audit, E8b as part of E5, E8c only after a profile-based
cost bound, E8d as a bounded feasibility prototype rather than a full model conversion.
For each arm report startup/precompute cost separately from warm decode savings. Stop
if the measured removable cost is negligible or lookup traffic outweighs the saved work.

## 6. E9: spare-core and L1 utilisation experiments

Status: planned, not benchmarked. Optimise committed-token latency, not the percentage
of cores lit up. A core idle in one operation may be required by another operation in
the same trace; there is no verified permanent pool of spare cores. Inspect the live
core maps, allocator reservations and runtime scheduling before assigning workers.

Entry gate: E0/E1 complete, latest fused B=1 control reproduced, and a bounded profile
of each candidate operation. Record participating cores, work per core, L1/CB footprint,
DRAM bytes, NoC transfers and wait/compute time where observable. Mark unavailable
counters explicitly rather than deriving utilisation from a configured grid size.

| Arm | Control and intervention | Measurements | Correctness and adoption gate |
| --- | --- | --- | --- |
| E9a: prefetch/staging | Tuned projection reader versus bank-local staging workers and two L1 buffers; test current-operation tiles before cross-operation prefetch | Reader wait, effective bandwidth, NoC traffic, buffer occupancy, full projection and trace time | No overwrite before consumption or trace-address changes; improve full trace, not just enqueue time |
| E9b: streaming stages | Existing materialised intermediates versus direct L1 producer-consumer transfer for one compatible projection/norm/gate/recurrence boundary | Removed DRAM bytes, added core-to-core bytes, synchronization and stage overlap | Preserve baseline rounding/layout and state ownership; net end-to-end win after communication |
| E9c: persistent state | DRAM-backed GDN state versus reserved L1 state for one layer, then a capacity-bounded subset | State bytes moved, reserved L1, snapshot/restore cost, recurrence and complete-step time | Survive intervening operations, trace replay, cancellation, slot reuse and speculative rollback; no allocator aliasing |
| E9d: attention-prep partitioning | Current B*4 work-instance mapping versus finer head/dimension partitioning at B=1 | Per-worker work, normalization reduction cost, gathers, kernel and complete-step time | Correct Q/K norm, RoPE, head ordering and paged-KV output layout; reproduce baseline numerics |
| E9e: long-context attention | Inspect existing paged SDPA partitioning first; compare only genuinely different legal partition counts | 4k/32k/64k attention latency, KV traffic, partial-result reduction cost | Numerically validated stable softmax combination, masks, positions and page boundaries; report short-context tradeoff |
| E9f: sampling workers | Host sampling versus device vocabulary-shard reductions, first greedy then separately general sampling | Logits bytes returned, device reductions, host synchronization, local and gateway decode rate | Exact greedy token/tie handling; non-greedy requests retain correct sampling semantics or use unchanged host path |
| E9g: multi-token worker mapping | E3/E4 verifier with existing mapping versus mappings tuned for T=2/4/8/16 | Weight reuse, useful tile occupancy, actual core participation, V(T), rollback and committed rate | All E3 prefix-state and output gates; multiple tokens may fill existing tiles without needing more cores |

Concrete starting points in this repository:

- `optimisation/ttnn-op/attn_prep/device/attn_prep_program_factory.cpp` defines
  `n_inst = attrs.B * 4`: four work instances at B=1. This is a parallelism candidate,
  not proof of a bottleneck. Bound the gain by its measured share of the full step.
- `optimisation/ttnn-op/gdn_conv_gates/device/gdn_conv_gates_program_factory.cpp`
  already assigns gates to an additional core where possible. Preserve this as the
  specialized-worker control rather than claiming specialization is entirely new.
- Historical matmul grid widening and DRAM-sharding negatives remain controls.
  E9a/E9b must change data movement or overlap, not repeat a larger-grid flag sweep.

### Resource and dependency discipline

For E9a, sweep 0/4/8/16 staging workers only when the live physical mapping and available
L1 permit it; record exact bank and consumer placement. Existing reader/compute/writer
engines already overlap within cores, so extra staging is justified only by measured
remaining stalls. Prefetch hides latency; it does not remove the weight-bandwidth floor.

For E9b, implement explicit producer-consumer synchronization in a supported fused
program or validated runtime mechanism. Two ordinary TTNN calls on different core maps
do not establish concurrent execution. Start with one boundary and count the added
layout conversions before attempting a larger pipeline.

For E9c, budget recurrent and conv state together with code, circular buffers, other
live tensors and reserved runtime regions. Start with B=1; never assume all layers or
the draft model fit in L1. Persistent storage requires ownership across every operation
that could otherwise reuse those addresses. Park/restore or fall back to DRAM for other
buckets until their own resource and lifecycle gates pass.

Do not allocate a whole card to drafting: both cards remain TP=2 for the target. Draft
and target work share DRAM/NoC bandwidth and target-feature dependencies. Any proposed
overlap must identify the exact independent operations and include contention costs.
Likewise, prefill on another core subset is not disaggregated prefill and is not a
substitute for E2's safe scheduling. Neither overlap is an assumed performance gain.

### Run order and result sheet

Prioritise E9g with E3/E4, then E9b/E9c for traffic reduction. Run E9d as a bounded
small-kernel probe; run E9a only with reader-stall evidence. E9e targets long-context
workloads, and E9f targets measured sampling/readback overhead. Keep one independent
change per arm and apply E6 before adoption.

Every arm appends this result record; use null for unmeasured fields, never zero:

| Field | Required content |
| --- | --- |
| Identity | Experiment/variant, run IDs, image and source revisions, control flags |
| Workload | Context tokens, B, T, output count, sampling, trace bucket |
| Resources | Requested and participating cores, placement, reserved L1/CB bytes |
| Movement | DRAM read/write bytes, NoC bytes, measured versus estimated attribution |
| Timing | Operation time, complete decode/verify cycle time, committed tokens/s, variability |
| Gates | Numerical/output/state tests, negative controls, lifecycle tests and failures |
| Decision | Planned/failed/correctness-passed/benchmarked/adopted, reason and next dependency |

Reject extra core activity without a repeatable full-path latency benefit. Preserve
negative results, and distinguish a long-context-only gain from a general decode win.

## 7. E10: prefill/decode disaggregation

Status: proposed following the Discord suggestion, not implemented or benchmarked.
Primary hypothesis: isolate ongoing decode from newly arriving long coding prompts.
This is a tail-latency/responsiveness experiment, not evidence that isolated B=1
decode will reach 200 tokens/s. A lone request still needs its own prefill before
its first decode step; any benefit there must come from measured phase optimisation,
not overlap of that request's causally dependent phases.

Distinguish three mechanisms: scheduler interleaving on the same TP=2 pair (E2),
separate prefill/decode devices, and spatially partitioned workers sharing devices.
Two Python processes or queues on the same pair do not establish concurrent execution
or memory-bandwidth isolation. More active cores do not imply more DRAM bandwidth.

| Arm | Placement and comparison | Entry/stop gate |
| --- | --- | --- |
| E10a: scheduling control | Existing monolithic TP=2 prefill versus E2's resumable chunks and decode-priority scheduling on both cards | Correct start-position handling, bounded decode drain and full state continuity before enabling scheduler chunking |
| E10b: physical 1P/1D | Card 0 holds a complete prefill replica; card 1 holds a complete decode replica, both TP=1 | Verify full-model weights, attention KV, GDN state, traces and scratch fit each card at unchanged precision/context; port and validate TP=1 shapes first. Stop if infeasible; no implicit quantisation or extra card |
| E10c: shared-pair spatial split | Preserve TP=2 weights; reserve explicit core/L1 groups for concurrent prefill and decode within a compatible runtime | Prove scheduler/trace support, disjoint mutable state and resource ownership; measure shared DRAM/NoC contention. This is a new kernel/runtime design, not a serving flag |

Keep a TP=1 unified serving control for E10b to separate the cost of losing TP=2
from the benefit of isolation. Decode loses one card's memory resources but also
removes TP collectives: measure the net result rather than assuming a factor of two.
The existing TP=2 kernel fixtures do not certify TP=1. A separate prefill accelerator
with both P150As retained for decode is outside the two-card constraint.

### Hybrid-state handoff gate

Start with a same-layout, host-staged correctness oracle, then evaluate device-to-device
transport. Do not assume a CUDA KV connector supports TTNN or this hybrid model.
The handoff must identify and transfer:

- Attention K/V for all full-attention layers, logical token positions, valid lengths,
  page/block mapping, layout/dtype and any TP shard mapping.
- Every GDN layer's recurrent state and convolution history/carry at the same prompt
  boundary; attention KV alone cannot resume the model correctly.
- Exact templated token IDs, model/tokenizer/weight revision, RoPE configuration and
  position, plus sampling/first-token ownership so no token is sampled or emitted twice.
- Request ID and slot generation/epoch, completion fences, acknowledgement and buffer
  lifetime; reject stale, partial, duplicate or incompatible transfers before decode.

Compare uninterrupted prefill+decode against handoff+decode over multiple continuation
tokens, not just first-token agreement. Exercise cancellation, interrupted transfers,
retry, slot reuse and interleaved requests. Explicitly drop/corrupt one GDN state or
KV page as a negative control. Repartitioning or different phase arithmetic needs
numerical validation and E6's coding-quality gate, not just successful deserialization.

Record transfer bytes from actual tensor allocations (including padding), source-read,
pack/reshard, transport, destination-write and synchronization costs separately. Measure
both during idle decode and simultaneous prefill; raw link bandwidth alone is not the
handoff cost. Report attention KV and GDN bytes separately to avoid E1's metric confusion.

### Workload, decision and reusable upstream code

Use isolated B=1 as the speed control, then one and seven ongoing decoders with a
new 4k/32k/64k prompt, plus repeated long-prompt bursts. Report TTFT, committed-token
rate, p95/p99/max token gap, prefill queue wait, handoff latency, peak memory and
goodput against an explicitly chosen latency SLO. Keep output/sampling identical.
Adopt only with a paired end-to-end improvement over E10a and acceptable isolated
decode performance; a loaded-service win must not be labelled a B=1 kernel speedup.

The pinned TT-Metal checkout already has a
[common prefill adapter/runner](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md)
with socket orchestration, migration handshakes and model-owned runtime integration.
Audit its cache abstraction for GDN state before reuse; its presence does not prove
Qwen3.8-27B support. Tenstorrent also documents
[separate prefill/decode deployment](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/tt-inference-server/#disaggregated-prefill-decode).
[vLLM's explanation](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
distinguishes phase tuning and tail-ITL isolation from throughput gains.

Offsite order: inspect adapter/transport contracts, validate small hybrid-state handoffs
in simulation, then run the fixed-card A/B matrix on a verified hardware CI runner.
Simulator wall time cannot decide which placement is faster. E2 remains the first
responsiveness intervention; investigate E10b/c without blocking E3-E5's decode work.

## 8. Order and next inputs

Execution order: E0 -> E1; then E2 for responsiveness and E3 -> E4/E5 for decode speed;
E6 gates deployment, E7 follows safe state lifecycle support. Analysis can proceed in
parallel, but silicon runs share the pair and must be serialized.
E8 adds lookup/precomputation probes; E9 refines E4 into explicit spare-core and L1
experiments. They share the same controls and quality gates, not separate competing baselines.
E10 compares disaggregation against E2, retaining the fixed two-card budget and separate
single-stream versus mixed-traffic acceptance criteria.

The operator has allocated the runner's pair and requested keeping scoped runner access.
The historic 0% signal is no longer a required input: capture current metrics and
investigate if it recurs rather than blocking baseline testing on an old dashboard.
The model
and plugin implementation are fetched into serving images rather than fully vendored
here, so inspect/export their exact running revisions before preparing runtime patches.
Do not modify Thatch.Server or its production configuration as part of this repo-only
experiment setup.
# Integrated drafter projection update (2026-09-07)

Simulator `20260907T102134Z-309` passes the live one-row full projection,
FP32 fabric gather/add, BF16 cast and learned RMSNorm pipeline on both ranks.
Sums are exact; normalization differs by at most one BF16 ULP (bound two).
The existing opt-in full-projection CI suite now also checks this integrated
path, retaining its host-reduction control. Hardware run `34111774033` on
`731a3ea` now passes: both sums exact, normalization within one BF16 ULP,
all 10,240 projection values identical to the separate control. Eight-row
batching now passes simulator validation (`20260907T103148Z-411`): exact
fabric sums across both complete eight-row outputs and normalization within
one BF16 ULP. Hardware run `34112856023` on `3cb7381` now passes the same
eight-row gate. Next, the composed DFlash2 group-16 dynamic causal convolution
passes exact synthetic T1/T8 simulator checks (`20260907T104613Z-391`);
learned convolution weights and draft-layer integration remain outstanding.
Learned layer-zero convolution plumbing now passes T8 simulator checks
(`20260907T105950Z-754`), with exact prepare/finish arithmetic on both ranks.
The fix requires explicit FP32 operands/output and BF16 rounding per operation;
FP32 output alone did not suffice. This is a correctness control with extra
casts, not a fused speedup. Real attention/MLP transforms between prepare and
finish and hardware convolution validation remain outstanding.
Hardware `34114708492` now confirms the learned convolution plumbing gate:
both phases exact for both branches/ranks, RMSNorm within one BF16 ULP.
Draft attention is the next unresolved gate: the explicit noncausal-block,
sliding-context GQA probe fails its random-query numerical bound; uniform-query
isolation reaches an unsupported TTSim pack-zero instruction. Neither is an
native attention pass. An unfused device attention reference now passes the
same random-query gate on context0/31/2048 (`20260907T111306Z-301`), including
masked-key poison and causal-mask negative controls. This does not resolve the
native SDPA numerical/unsupported-instruction issues or certify learned attention.
Hardware `34115875438` confirms the composed attention reference. Warm attention
median times are 0.633/0.825/1.637ms at context0/31/2048 respectively, excluding
the other draft and target operations, not token throughput. The pinned learned
Q/K/V/O and head-norm fixture is downloaded and verified for the next integration.
The learned attention pipeline is now implemented as a simulator diagnostic,
but does not pass: explicit FP32 RoPE fixes its first failure, then learned
attention exceeds the unchanged numerical bound. QK/softmax/PV stage errors
are isolated in `20260907T113235Z-303`; no hardware promotion of this learned
path. Further isolation fixes softmax via pairwise FP32 reduction (maximum
error1.19e-7 on both ranks), but QK/PV matmuls still leave two attention values
outside the unchanged bound (`20260907T114726Z-297`). No learned-layer hardware
promotion or speedup claim. The host test suite now passes 515 tests.
This removes a diagnostic host handoff, not the remaining draft layers or
request-history integration. No committed-token throughput gain is established.

### Learned attention precision reference (2026-09-08)

Simulator `20260907T115603Z-287` passes the learned layer-zero Q/K/V projection,
head normalization, FP32 RoPE, attention, output head merge, O projection and
fabric sum checks on both ranks. The fixture uses synthetic hidden features,
context31, eight proposal rows and absolute start4096, not a coding request.
Explicit FP32 broadcast products and pairwise reductions replace both QK and PV
matmuls, alongside the explicit pairwise softmax. Attention maximum errors are
1.431e-5/1.669e-5 against the unchanged reference bound; output head order and
fabric sums are exact. Intermediate inspection measures QK error at most
3.052e-5, softmax error1.193e-7 and PV error5.723e-6.

This is an allocation-heavy numerical reference, not a fused kernel or speedup.
Its outer-product memory guard deliberately rejects long-context inputs;
production attention still requires a bounded tiled/fused implementation and
long-context validation. The earlier hardware attention timings do not measure
this new path. No complete draft layer, coding-quality result or 200tok/s result
is established. The inspection-free repeat `20260907T120553Z-446` also passes:
eight checks, zero intermediate diagnostic readbacks, unchanged attention errors
and exact fabric sums. All 516 host tests pass in the WSL Torch environment;
Windows Python lacks Torch and is not the supported test environment. The next
gate is isolated learned-attention hardware validation, followed by bounded
tiled/fused arithmetic and complete draft-layer integration.

The opt-in `learned-attention` CI suite now performs owner/runtime auditing,
transfer health and this precise inspection-free learned-attention gate only.
Its pinned fixture cache is rehashed before reuse; failed health prevents the
probe. Hardware rejects unvalidated arithmetic flag combinations. All 518 host
tests pass. This gate neither loads the target model nor changes serving defaults;
it establishes component correctness only, not end-to-end latency or throughput.

Hardware run `34120652422` on `a6829df` now passes this gate on both P150As.
All eight reported check records match the inspection-free simulator result,
including attention errors1.431e-5/1.669e-5, one-ULP head normalization and exact
fabric sums. This comparison is of reported metrics, not a saved full-tensor
bitwise comparison. The arithmetic implementation hash matches the simulator;
the hardware report contains zero intermediate inspection records. No timing
was measured for this precise path. Next work must remove the outer-product
allocation/long-context limitation and integrate complete draft layers; this
correctness pass alone supplies no new committed-token throughput result.

### Fused SFPU denominator reduction (2026-09-08)

`draft_row_sum` now implements FP32 row summation in one generic-op dispatch,
with 16 workers per card, two input tiles and one output tile of L1 buffers per
worker (12KiB), independent of reduction width. The compute kernel accumulates
column tiles in FP32 DEST through SFPU addition, then performs SFPU row reduction.
This replaces the explicit softmax denominator's repeated slice/add dispatches;
it does not fuse QK, PV, exponentiation or the entire attention operation.

Simulator `20260907T122332Z-548` passes positive-input row sums at widths32/64/2080
on both ranks against FP64 host sums cast to FP32. The inspection-free learned
attention integration `20260907T122502Z-302` also passes the unchanged gates.
All 519 host tests pass. Hardware remains disabled for this new option until its
own promotion; no timing improvement is claimed from simulator wall time.
The QK/PV outer-product reference remains the next large allocation bottleneck.

The learned-attention CI suite now retains its pairwise control, runs a matched
fused/pairwise row-sum comparison, then checks the fused learned-attention path.
The comparison has one warmup and five samples per arm/width, alternating arm
order, checking every output against FP64 reference sums and repeated outputs
for exact stability. Timing includes dispatch, allocation, synchronization and
temporary cleanup, excluding uploads/readback/final output release. Simulator
`20260907T123043Z-305` passes all twelve width/arm/rank correctness records.
This is a reduction-only cost comparison, not full attention or request timing.

Hardware `34122410907` on `53ccfef` passes twelve reduction checks, all thirty
timed repeated-output checks, and both learned-attention control/candidate gates.
The fused candidate retains the same reported attention errors and exact fabric
sums. Five-sample warm medians (milliseconds) are:

| Padded key width | Pairwise reduction | Fused SFPU reduction |
| --- | ---: | ---: |
| 32 | 0.617333 | 0.148081 |
| 64 | 0.687814 | 0.146431 |
| 2080 | 1.289776 | 0.169591 |

The isolated measured speedup is about4.2x/4.7x/7.6x, saving0.469/0.541/1.120ms
per measured reduction invocation across the mesh. These are synchronized host
latencies including allocation/temporary cleanup, not pure device kernel time.
Small sample counts and observed latency variation require end-to-end retesting;
do not multiply this speedup into model throughput. QK/PV remain allocation-heavy,
the full drafter remains incomplete, and200 committed tokens/s remains unproven.

### Bounded SFPU QK/PV dot prototype (2026-09-08)

`draft_dot` now streams tile products through SFPU row broadcast, multiplication,
FP32 tile accumulation and row reduction in one generic-op dispatch. It does not
materialize the earlier four-dimensional outer-product tensor. Seven FP32 tiles
of circular-buffer/scratch storage (28KiB) are allocated per worker, with16..64
workers per card. This is independent of key/reduction width; output scores and
input casts still consume DRAM. No Ethernet dispatch or extra-column claim.

Signed random FP32 inputs pass against FP64 host matmul at unchanged1e-5 relative,
1e-4 absolute bounds on both ranks:

- `20260907T123900Z-305`: QK width128/keys32, maximum error3.815e-6.
- `20260907T123947Z-381`: QK width128/keys2080, maximum error7.630e-6.
- `20260907T124150Z-472`: PV width2080/outputs128, maximum error3.434e-5.

The prototype rereads tiles and serializes output columns within each worker;
bounded memory is not proof of adequate latency. Learned-attention integration
and hardware cost must gate adoption. The simulator-only option is `--fused-dots`;
hardware rejects it for now. All 522 host tests pass.

Inspection-free learned attention `20260907T124315Z-759` also passes all eight
checks with fused QK/PV and fused denominator reduction. Output head ordering and
fabric sums remain exact. This establishes the short-context integrated numerical
gate only; synthetic long-context dot checks are not full long-context attention.

The opt-in CI suite now adds hardware gates for the three simulator-validated
dot shapes plus fused learned attention. Each dot probe takes five warm samples,
checking exact repeat stability on both ranks; timing includes output allocation,
dispatch and synchronization, excluding uploads/readback/output release. This
measures the new dot implementation's absolute cost, not a matched speedup over
the earlier outer-product reference. Existing pairwise learned-attention and
fused-denominator controls remain in the suite. Host tests:522 passed.

Hardware `34123851597` on `290b6d8` passes all three dot shapes, fifteen timed
repeat checks and the fused learned-attention integration. Reported numerical
errors match the corresponding simulator checks; short-context learned attention
errors are1.383e-5/1.193e-5, with exact head order and fabric sums.

| Dot shape | Five-sample median ms | Min ms | Max ms |
| --- | ---: | ---: | ---: |
| QK, width128/keys32 | 0.314661 | 0.306142 | 0.489972 |
| QK, width128/keys2080 | 3.879600 | 3.820399 | 4.029841 |
| PV, width2080/outputs128 | 3.379377 | 3.249936 | 3.402227 |

The sum of the two long-context component medians is7.259ms, not a measured
combined attention latency. This is too costly to treat as a completed fast
drafter: five layers also need projections, normalization, softmax, convolution,
MLP and selection, followed by target verification. The next kernel experiment
should reuse left/right tiles in L1 instead of rereading them for every output
column, explicitly measuring the L1-capacity versus DRAM-traffic tradeoff. The
bounded streaming implementation remains the correctness control. No new
end-to-end throughput result or200 committed tokens/s claim follows from this run.

### L1 dot-tile reuse candidate (2026-09-08)

The simulator-only `--cache-tiles` dot variant retains a complete left tile row
and right tile block per worker/output tile. Each operand tile is fetched once
instead of once per each of32 output columns. SFPU multiplication/accumulation
order is unchanged; the existing streaming variant remains selectable. Static
input tile reads per output tile decrease32-fold, not a measured latency gain.

The deliberate L1 tradeoff is48KiB per worker for QK width128 and536KiB for PV
width2080, versus28KiB for streaming. Output assembly and broadcast buffers are
included. Capacity alongside a complete drafter/target/trace is not yet validated.
Simulator cached-versus-streaming full-output equality and the FP64 reference
gate pass on both ranks for short QK (`20260907T125336Z-310`), long QK
(`20260907T125413Z-619`) and long PV (`20260907T125729Z-657`). Host tests:524 passed.
The integrated option is `--cache-dot-tiles`; hardware rejects it pending promotion.

Inspection-free learned-attention simulator `20260907T125934Z-819` passes all eight
checks with cached dots, including exact head order/fabric sums and the same
reported attention errors as streaming fused dots. Hardware timing is the next
decision gate; no simulator wall-time comparison is used as a speedup claim.

The opt-in hardware suite now runs both streaming and cached versions of all
three validated dot shapes, retaining identical seeded inputs and five warm
samples per shape/variant. Cached probes also require full-output exact equality
with their streaming control. A separate cached learned-attention gate retains
the unchanged numerical bounds. Arms run sequentially, not interleaved, so any
timing comparison must retain that limitation and the five-sample variability.
Host tests:524 passed. No serving defaults or reset authorization change.

Hardware `34125324127` on `3c8e53c` passes streaming/cached dot correctness,
exact cached-versus-streaming outputs, repeated-output stability, and the cached
learned-attention integration (eight checks). Five-sample medians in milliseconds:

| Shape | Streaming | L1 cached |
| --- | ---: | ---: |
| Short QK | 0.330212 | 0.248592 |
| Long QK | 3.785210 | 2.361222 |
| Long PV | 3.387797 | 1.581398 |

Long QK/PV decrease approximately38%/53%, respectively. The sum of their medians
drops7.173007->3.942620ms (45%), not a measured combined attention latency. This
confirms L1 reuse helps, but32-fold fewer input tile reads does not imply32-fold
speedup: row-broadcast/compute/output-column serialization remain. Follow-up
should measure wider worker distribution and/or multiple output columns per
compute iteration, retaining exact controls and the large PV L1-capacity gate.
There is still no complete drafter, full request timing, or200tok/s result.

### Wider dot worker distribution (2026-09-08)

The opt-in dot worker cap now supports64/80/110, bounded by actual output-tile
tasks and the reported compute grid. Full/ragged core rows and strided task
coverage are tested. Long QK has1040 tasks and can use110 workers (11x10), whereas
the current long PV geometry has only64 tasks and gains no workers from the cap.
This uses the existing compute grid; it is not an Ethernet-dispatch column gain.

Simulator `20260907T131006Z-307` passes cached long QK on110 active workers,
including exact full-output equality against cached64 workers and the unchanged
FP64 reference bound. The hardware suite adds only this validated110-worker
shape;80-worker hardware remains gated. Learned-attention defaults still use64.

Hardware `34126247829` on `6d358c2` confirms110 active workers per card for the
cached long-QK probe. Full outputs equal the64-worker control, both ranks pass
the FP64 reference bound (maximum error7.630e-6), and five repeated outputs are
stable. Same-run five-sample warm medians improve2.327681ms at64 workers to
1.593638ms at110 (31.5% lower); ranges2.294072..2.558523ms and
1.574168..1.807489ms respectively. These are sequential-arm component timings,
not a whole-model speedup. This demonstrates useful extra-core work without
changing dispatch placement. PV still needs finer task partitioning to exploit
more than64 cores; complete long-context learned attention and request integration
remain unvalidated. The200 committed-token/s goal remains unmet.

### Split PV output-tile tasks (2026-09-08)

An opt-in eight-column task partitions each32-column output tile four ways.
Long PV therefore has256 tasks instead of64 and can occupy110 cores. Each worker
writes only its own32-byte-aligned row segments; no whole-tile write races or
cross-worker floating-point reductions are introduced. Host coverage tests check
that the four partitions cover every output-tile byte exactly once. Cached L1
storage remains536KiB per worker, and tile fetches are duplicated across the four
partitions, so a hardware timing comparison must decide whether parallelism wins.

Simulator `20260907T132015Z-307` passes long PV on110 workers with eight-column
tasks, including exact full-output equality against cached64-worker whole tiles
and the unchanged FP64 reference bound. Host tests:526 passed. The hardware suite
adds only this validated split shape; integrated learned attention still uses
whole-tile tasks. This is not yet a complete drafter or committed-token speedup.

Hardware `34127121901` on `de7c0f7` passes the110-worker split PV gate: exact
output equality against the cached64-worker whole-tile control, unchanged FP64
reference error3.434e-5 on both ranks, and five stable repeated outputs. Same-run
medians are1.706819ms for64-worker whole tiles versus1.457718ms for110-worker
eight-column tasks (14.6% lower). Ranges1.516678..1.790549ms and
1.427718..1.629718ms overlap, so treat this as a modest preliminary component
improvement, not a robust whole-request gain. Next integrate the measured QK/PV
placements into complete long-context learned attention and measure that pipeline
before further microkernel tuning. No200 committed-token/s result is established.

### Long-context learned-attention integration diagnostic (2026-09-08)

The simulator diagnostic now accepts context2048 as well as its context31 control.
It derives2080 padded K/V rows,2056 valid projection rows and absolute query
position6144 from the same4096 context start. Projection validation processes all
valid rows in32-row CPU reference chunks to bound reference memory, not to sample
away rows. Head layout, normalization, RoPE, sliding/noncausal-block attention,
O projection and fabric-sum gates retain their existing numerical bounds.

`--wide-dot-placement` selects110-worker cached QK and eight-column110-worker
cached PV tasks. It requires cached fused dots and remains simulator-only.
Run `20260907T132957Z-306` was launched for the long-context integrated gate and
is still running; there is no pass claim yet. Host tests:527 passed. Full target
feature history, convolution/MLP integration and request throughput remain separate
outstanding requirements; this diagnostic does not complete them.

The first long-context run terminated at the wrapper's1800-second limit
(exit-status124), with no final numerical report; it is not a correctness pass
or a demonstrated kernel failure. A CPU reference benchmark of the pinned K
projection takes6.777s for32 rows and29.641s for128 rows (the overlapping results
are exact), so keep32-row chunks. Four full K/V rank references alone require
roughly260 such chunks, making the old whole-process deadline inadequate once
device simulation and other checks are included. The probe now writes explicit
progress checkpoints with `passed=false` until all gates finish. Retry with a
3600-second whole-process deadline, without dropping rows or changing tolerances.

### Draft MLP preparation (2026-09-08)

While retry `20260907T140423Z-302` runs, a bounded-range fetcher now selects only
the pinned layer-zero gate/up/down BF16 matrices from the already audited header.
Each matrix is178257920 bytes, total534773760 bytes. The download is in progress;
content hashes and a verified tensor loader are not yet established for this new
subset. No remote checkpoint code is executed.

CPU helper tests establish gate/up output-axis splitting and down input-axis
splitting for TP2, plus explicit BF16 SwiGLU rounding. These are packing/reference
tests, not a device MLP correctness or performance result. Host tests:530 passed.
The next independent step is to finish hashing the MLP fixture and build its
device projection/activation/reduction gate, without interrupting the active
full-row long-context attention validation.

The initial MLP download terminated on a network read timeout. Its partial file
is preserved and is not accepted as a fixture. The shared bounded-range reader
now retries transient timeout/connection/incomplete-read failures at most three
times for the identical range, retaining status/range/size validation and not
retrying invalid responses. Tests cover retry identity, exhaustion and rejection
without retry; all533 host tests pass. A fresh download is active under
`hardware-evidence.local/dflash2-mlp-dedf8df-retry1`; no partial-file overwrite or
unverified resume was performed. Attention retry remains a separate live process.

The fresh MLP download completed. All534773760 bytes were rehashed and loaded
as finite BF16 matrices; `draft_mlp_fixture` now pins each tensor's SHA256 and
checks model revision, header, shape, dtype and size before loading/reuse.
Corrupt cached content fails without overwrite/redownload. The three matrices
are available in `hardware-evidence.local/dflash2-mlp-dedf8df-retry1` for the
device MLP gate. All534 host tests pass. The long-context attention retry has
completed chip0's full K projection/normalization/RoPE checks and is advancing
through full V reference validation; final status remains pending, not passed.

The simulator-only `learned-mlp-probe` is now implemented for eight synthetic
input rows padded to32. It feeds learned TP2 gate/up projections through explicit
BF16 rounding with FP32 SiLU/multiplication, then learned row-parallel down
projection and FP32 fabric gather/add. Planned gates cover every valid gate/up
and down projection value against the existing ISA-aware reference, activation
within two BF16 ULPs, and exact all-row fabric sums. Progress is checkpointed.
The operation-sequence CPU test and existing suite pass535 tests; the device gate
has not yet run because the long-context attention simulator remains active.
No second simulator process is started concurrently, and no MLP hardware pass,
fused-kernel speedup or complete draft-layer integration is claimed.

The long-context simulator diagnostic `20260907T140423Z-302` completed all
numerical checks and mesh cleanup, writing `passed=true`. Both ranks validate
all2056 valid K/V rows, eight Q rows, normalization within one BF16 ULP,
unchanged RoPE tolerances, attention/output projection and exact fabric sums.
Maximum attention errors are1.8119812e-5 and1.4305115e-5. This validates the
cached110-worker QK and split-column PV integration, not request throughput.
The outer shell exited1 after the diagnostic completed: editing its live script
to add the MLP probe changed its read offsets, producing an EOF parse error.
Consequently there is no wrapper exit-status file and this is a numerical
diagnostic pass, not a clean wrapper run. The JSON is written true only after
successful tensor release and mesh close. Do not edit running shell scripts.

The hardware attention suite now includes only this precise validated long
configuration, with3600 seconds for its complete CPU references inside the
existing4800-second suite deadline. Short-context controls remain intact.
No long-context hardware result is claimed yet. The separate learned MLP
simulator gate has started after the attention process terminated.

The learned MLP simulator `20260907T144652Z-388` completed with both
`passed=true` and wrapper exit-status0. All learned gate/up and down values
for eight valid rows pass the unchanged ISA-aware projection rtol/atol1e-4
checks on both ranks. SwiGLU has zero BF16 ULP difference on all32 padded
rows, and fabric sums are exact. Maximum gate/up absolute error is4.7683716e-6;
maximum down absolute error is0.0001220703125 (within the combined relative
and absolute tolerance). The7.5-minute simulator wall time is not hardware
latency. Convolution/residual integration, complete learned draft layers and
coding acceptance/committed-throughput validation remain outstanding.

The isolated `learned-mlp` hardware suite now checks transfer health before
the same learned projection/activation/fabric diagnostic. It downloads only
the pinned three layer-zero matrices with full hash validation, uses a1800s
probe limit, and retains the existing allocation/no-reset requirements.
No MLP hardware result is claimed yet. Long-attention hardware run34134992019
is in progress at commit7d0287f.

The next simulator-only gate connects the learned MLP branch end to end:
post-attention RMS normalization, learned dynamic convolution kernels,
prepare convolution, TP2 gate/up/SwiGLU/down, fabric sum, finish convolution
and residual addition. Both convolution banks are generated from the same
normalized branch input; no host readback feeds the device execution path.
The diagnostic checks each stage against the existing rounding-aware references,
including exact prepare/finish convolution and residual arithmetic, and retains
the separate projection/activation/fabric gates. Inputs are eight synthetic rows
padded to32, not actual attention outputs or a complete five-layer drafter.
Run `20260907T145919Z-495` is active; no numerical pass is claimed yet.
Hardware rejects this integrated option until simulator validation. All538 host
tests pass, including the pre-fixture hardware rejection check. The standalone
MLP hardware gate34135694410 remains queued behind long-attention34134992019.

Hardware run34134992019 completed successfully. The downloaded
`learned-attention-long-wide.json` confirms all2056 K/V rows on both ranks,
normalization within one BF16 ULP, unchanged projection/RoPE checks and exact
fabric sums. Attention maximum absolute errors are1.9073486e-5 and1.5258789e-5.
This is a real two-card long-context component correctness pass, not committed
coding tokens/s. The standalone MLP hardware run is now in progress.

Integrated MLP simulator145919Z-495 failed its down-projection reference check:
four of40960 valid rank0 values exceeded the existing combined rtol/atol1e-4
bound, with greatest failing absolute difference0.000457763671875. Prior
convolution/residual, gate/up and activation checks passed. The bound remains
unchanged and integrated hardware remains blocked. A failure capture now saves
actual activation/output, ISA-aware and ideal FP64 references plus checkpoint
identity for bounded local replay; the diagnostic retry is intended to locate
the numerical cause rather than declare the discrepancy acceptable.

The captured rank0 down-projection failure is explained by CPU reference
ordering, not relaxed tolerance. Pinned Blackhole LLK `matmul_configure_mop`
in `llk_math_matmul.h` processes both16-wide reduction halves of a32-wide
tile before advancing fidelity. The old reference advanced fidelity after each
16-wide half. A configurable32-wide fidelity span now retains separate MVMUL
accumulations while matching this instruction order. CPU replay of all40960
captured outputs reproduces the old reference's four tolerance failures and
maximum error0.0078125, but the corrected schedule is bitwise exact with
zero error everywhere. Evidence: `hardware-evidence.local/mlp-down-fidelity-replay.json`.
The accumulator arithmetic remains based on
https://github.com/tenstorrent/ttsim/blob/v1.10.3/src/tensix.cpp .
Tests assert phase/half ordering and reject unsupported geometry. The MLP
diagnostic explicitly selects this schedule for its32-wide tiles; existing
other diagnostics retain their old default until separately revalidated.
The integrated branch still requires a complete new simulator pass on both
ranks; the saved single-rank replay is not that pass and is not a speed result.

The complete integrated MLP simulator151519Z-509 now passes on both ranks,
with wrapper exit-status0. All gate/up/down projections match the corrected
reference exactly (zero maximum error). SwiGLU, prepare/finish convolution,
residual addition and fabric sums are exact; RMS normalization is within one
BF16 ULP. Tolerances were not relaxed. This is the learned layer-zero MLP
branch on synthetic input, not a complete attention-plus-MLP layer or drafter.

Standalone MLP hardware34135694410 also passed, with matching simulator
numerical values using the older16-wide reference. The `learned-mlp` suite
now retains that standalone control using the corrected32-wide reference and
adds the simulator-validated integrated branch. Both fixtures are hash-verified;
each probe has1800s inside the existing4800s suite deadline. No integrated
hardware pass or request-throughput improvement is claimed yet.

The attention diagnostic now has a simulator-only integrated branch matching
the pinned DFlash2 layer contract: normalize proposal inputs, compute both
dynamic convolution banks once, prepare proposals, project Q from prepared
proposals and K/V from unchanged context plus prepared proposals, run attention
and output projection/fabric sum, then finish convolution and add the residual.
Device execution does not use intermediate host readbacks. Exact context/proposal
assembly, convolution/residual arithmetic and all prior attention checks are
retained; integrated projections select the corrected32-wide fidelity schedule.
The first context31 gate152816Z-525 is running, not yet passed. Hardware rejects
this option before fixture/device access. All541 host tests pass. This is not yet
a full attention-plus-MLP layer, five-layer drafter or request-history adapter.

Integrated MLP hardware34138213690 passed on both cards: gate/up/down
reference errors are zero, SwiGLU and convolution/residual/fabric arithmetic
are exact, and normalization is within one BF16 ULP. The downloaded report
confirms the actual integrated option, not only the standalone control.

Integrated attention simulator152816Z-525 also completed with exit-status0
and passed=true. Both ranks have exact Q/K/V projections, context/proposal
assembly, convolution, residual and fabric sums. Attention maximum absolute
errors are2.0980835e-5 and1.5258789e-5 within unchanged bounds. The hardware
suite now includes only this short-context cached fused integration, retaining
the existing non-integrated long-context test. Integrated long-context/wide
hardware combinations remain rejected pending their simulator gate. Neither
branch has yet been connected to the other as a complete learned draft layer.

The simulator-only `--mlp-fixture` attention option now connects both branches
as a complete layer-zero diagnostic. `draft_mlp_branch` consumes the actual
attention residual tensor on the existing TP2 mesh, executes the learned MLP
branch, and returns device tensors for independent stage checks. No host tensor
reconstruction lies between the branches. The gate retains attention checks,
MLP projection/fidelity references, normalization/activation bounds and exact
convolution/residual/fabric checks, plus the tensor-identity handoff check.
Run153623Z-521 is active at context31; no complete-layer pass is claimed yet.
All542 host tests pass, including rejection of complete-layer hardware before
fixture access. Hardware attention run34138908364 remains a separate gate.
Five learned layers, projected target-feature inputs, selector, transactional
history and real coding acceptance/committed throughput remain outstanding.

While the complete layer-zero simulator runs, the remaining four learned
layers are being staged from the same pinned DFlash2 checkpoint. Each layer
selects15 attention/convolution/normalization/MLP tensors,665948672 bytes;
four layers require2663794688 bytes. Layer-zero fixtures are not downloaded
again. The staging utility retains bounded range reads, pinned header and
geometry checks, streaming hashes and refusal to overwrite existing data.
Each layer has a separate directory under
`hardware-evidence.local/dflash2-remaining-layers-dedf8df`. No remote code is
executed and these new tensors are not yet accepted by a loader: complete
download, hash pinning and finite BF16 verification remain required. All545
host tests pass, including exact tensor counts/bytes and invalid-selection
rejection before network access. No five-layer runtime pass is claimed.

Complete layer-zero simulator153623Z-521 passed on both ranks with wrapper
exit-status0. Attention retains all previous gates; connected MLP convolution,
gate/up/down projections have zero reference error, activation/convolution/
residual/fabric checks are exact, and normalization is within one BF16 ULP.
The attention-to-MLP handoff is the same device tensor, with no host reconstruction.
The hardware suite now adds this validated short-context complete-layer gate
with1800s timeout and all three hash-verified fixtures. Existing branch controls
remain. Integrated long-context configurations are still rejected on hardware.
The separate current attention CI34138908364 is not cancelled or modified.
This completes only the simulator layer-zero gate, not the five-layer drafter,
coding acceptance or200 committed tokens/s objective.

Hardware34138908364 completed successfully; its downloaded integrated-attention
report confirms both ranks' convolution/residual and attention/output checks,
including exact fabric sums and attention maximum errors2.0980835e-5 and
1.5258789e-5. Complete-layer hardware34140315007 is now in progress.

The next active simulator gate155120Z-300 exercises the complete layer at
context2048, with cached110-worker QK and split-column PV. It retains every
valid2056-row K/V reference and both connected-MLP rank checks. Its5400-second
whole-process deadline accommodates the previously measured CPU reference cost;
no check is sampled away or tolerance relaxed. This run has no final result yet.
The remaining-layer download continues independently; neither active process is
restarted. Integrated long-context hardware remains blocked until this gate passes.

Remaining layer1 finished downloading. All15 tensors (665948672 bytes) were
independently rehashed on Windows, pinned in the loader, then rehashed and
verified as finite BF16 tensors in WSL. The loader rejects layers without
audited hash pins and rejects corrupted cached content; layers2-4 continue
downloading and are not accepted yet. All547 host tests pass. This is checkpoint
readiness only, not a layer1 device or complete five-layer numerical pass.

The complete-layer diagnostic is now factored into a repeatable layer operation
with simulator-only stacking. `--stack-fixtures` preloads only hash-pinned
remaining layers, then passes each layer's actual device result into the next;
host readbacks are validation-only and are never reconstructed as execution
inputs. Context features remain unchanged across layers, as in the pinned
DFlash2 contract. Reports identify the original checkpoint for every layer and
label checks/progress by layer index. The original layer-zero arithmetic and
all stage checks are retained. All548 host tests pass; this refactor and the
two-layer path still require device validation. A two-layer test is queued
behind active long-context155120Z-300, conditional on that run's clean success.
No second simulator is launched concurrently, and multi-layer hardware remains
blocked. Layers2-4 are still downloading and cannot load without audited pins.

Selector/final-normalization staging is now active separately: hidden projection
[256,5120], predecessor and successor codebooks [248320,256] each, and final
norm [5120], totaling256911360 bytes. Shapes and ranges were checked against
the pinned header before adding the bounded subset. The five layer families,
feature projection/norm and selector/final norm account for the complete
3848817896-byte checkpoint including its8936-byte header area; this arithmetic
coverage test is not a completed download or numerical gate. All550 host tests pass.

The pinned upstream `CandidateSelector.select` reranks top16 LM-head candidates
using the projected hidden state and predecessor/successor codebooks, choosing
the next predecessor sequentially. It does not remove the LM-head prerequisite.
Source: https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py .
Selector hashes/loading, arithmetic, token path and shared target LM-head wiring
remain unvalidated. Long-context simulator155120Z-300 has completed device
execution and is advancing through all-row CPU references, without a final pass yet.

Complete layer-zero hardware34140315007 passed at context31. The downloaded
`learned-layer-complete.json` confirms the actual connected branch option:
both ranks have zero MLP projection error, exact activation/convolution/residual/
fabric checks and one-ULP normalization. This is a complete single learned layer
on the two physical cards, not a five-layer drafter or measured coding throughput.

`draft_selector.greedy_selector_reference` now provides an ideal FP64 oracle for
the sequential candidate path. Tests cover predecessor-dependent choices,
independent batches, unary-vs-transition scoring, supplied candidate-order ties,
input immutability and invalid geometry/IDs/nonfinite data. It consumes already
projected hidden states and supplied LM-head candidates; it does not implement
the shared LM head or claim BF16/device equivalence. Selector arithmetic must
still be calibrated and validated on simulator/hardware. All555 host tests pass.

Selector/final norm download completed. All four tensors (256911360 bytes)
are now SHA256-pinned, rehashed by the loader and verified as finite BF16 in
WSL. Reuse rejects corrupted content without overwriting or redownloading it.
All556 host tests pass. These verified weights enable the forthcoming learned
selector numerical gate; no selector device implementation, LM-head integration
or measured proposal/coding throughput is implied by fixture validation.

The simulator-only selector fixture option now attaches final learned RMSNorm
and the256-rank hidden projection to the actual device output of a complete
layer/stack. Checks retain the two-ULP norm bound,32-wide ISA-aware projection
reference and exact BF16 cast, identifying proposal positions1-7 separately
from the anchor row. This is implemented but has not run in TTSim; it neither
produces LM-head candidates nor executes the selector codebook path yet.
Hardware rejects the new option before fixture access. All557 host tests pass.

Remaining layer2 completed downloading and all15 tensors (665948672 bytes)
were independently rehashed, pinned and verified as finite BF16 in WSL.
Layers0-2 and selector/final norm are now available as verified fixtures;
layers3-4 continue downloading. No layer2 or selector device pass is claimed.
The existing long-context simulation and queued two-layer test are unchanged.

## Eight-token cycle budget: verifier optimization is mandatory

Reinspection of hardware34075945160's actual T8 timing records gives median
static verifier costs64.4438705ms at4095 context and66.1075116ms at16383.
These are captured full-logit blocks with one preselected end checkpoint,
excluding drafting, dynamic selection and complete speculative commit. They are
historical measurements, not a fresh end-to-end DFlash2 benchmark.

For the trained block-eight path (seven proposals plus one target correction/
bonus), even perfect acceptance and zero other overhead would allow only
124.139/121.015 committed tokens/s if those verifier costs remain applicable.
200 tokens/s requires the entire eight-token cycle to fit within40ms. Thus the
verifier alone needs at least37.9/39.5 percent lower latency before allocating
any time to drafting, selection or commit; real acceptance requires more headroom.
Finishing the drafter port alone cannot satisfy the target on this baseline.

`verification_budget.py` derives these bounds from the saved hardware JSON,
records its SHA256 and assumptions, and rejects failed/missing timing evidence.
Output: `hardware-evidence.local/dflash2-eight-row-verifier-budget.json`.
All560 host tests pass. Priority alongside completing the trained drafter is a
matched T8 verifier-cost attribution/optimization gate, not reliance on a T32
timing or a selector speedup as proof of200 committed tokens/s. Wider proposal
experiments must separately establish acceptance and target-state correctness;
the trained block-eight contract is not silently widened.

### Matched T8 trace instrumentation preparation

`full-prefix.py --device-profile` now supports the existing packed-GDN,
ordered-cache static configuration only, with operation/device/trace-tracking
profilers explicitly enabled. It does not enable the legacy fenced attribution
path or alter the `ModelBatch` kernels. At T8 and contexts4095/16383 it replays
the already captured native, paired-control and candidate traces three times,
with uniquely named Tracy begin/end signposts. Initial-state restoration,
profiler dumps and exact logits/GDN/KV/end-checkpoint validation are outside
the marked interval. All existing correctness matrices remain required.

This is instrumentation preparation, not a hardware-validated profile yet.
The dedicated `verifier-profile` CI suite now launches Tracy and validates
both-chip device reports for all six context/arm traces. It retains the full
correctness matrix, but replaces timing sweeps with three T8 attribution
replays per arm. Missing replay coverage or dropped markers fails the gate.
Summed per-operation kernel durations are not critical-path latency.
Instrumented runs are explicitly marked and
rejected by `verification_budget.py`; throughput must be measured separately
without profiler overhead. No serving configuration changes are involved.

Remaining learned layer3 is now pinned: all15 tensors (665,948,672 bytes)
were independently SHA256-checked on Windows, then rehashed and verified finite
as BF16 by the WSL loader. Layer4 continues downloading. This fixture audit is
not a multi-layer execution or coding-quality result.

### Selector transition-table experiment

`draft_selector_transitions.py` supplies a mathematical oracle for parallel
transition scoring: at each proposal position the possible predecessor IDs are
the preceding position's16 candidates, with the known anchor at position1.
All seven16x16 score tables can therefore be computed before following the
greedy path. Only seven tiny row selections remain sequential; codebook reads
and dot products need not wait for preceding argmax decisions.

This spends up to16 times more dot-product work to expose parallelism; it is
not an assumed speedup. The block requires7x16x16 scores (7KiB in FP32) and
uses the original candidate order for ties. The FP64 oracle agrees with the
sequential greedy path and selected score rows on bounded randomized tests.
Finite-precision device equivalence, simulator gating, measured cost versus
serial selection, shared-LM-head candidate generation and coding acceptance
remain outstanding. No alternate drafter contract or serving change is implied.

The next simulator-only scoring gate uses `draft-dot-probe.py --keys 32
--width 256 --cache-tiles --columns-per-task 8 --selector-fixture <pinned path>`.
It packs learned codebooks into the existing SFPU dot kernel, compares split
tasks with its whole-tile control, and checks every active transition score and
the seven-token greedy path. Candidate IDs, hidden states and unary logits
remain synthetic; gathers, predecessor weighting and final selection are host
prepared. This isolates device scoring, not an implemented full device selector.

Long-context complete layer0 simulation `20260907T155120Z-300` passed with
shell exit0 and clean mesh close. Both ranks validated2056 K/V rows, exact
projection references, exact connected MLP outputs/residuals/fabric sums and
one-ULP normalization. Attention max errors were1.1444092e-5/1.3828278e-5;
existing tolerances were unchanged. Two-layer stack simulation
`20260907T164514Z-3101` is now running, replacing the completed simulator slot.

The final remaining layer4 download is complete. Its15 tensors (665,948,672
bytes) passed independent Windows SHA256 comparison and WSL hash/finite-BF16
verification. All four remaining layers now have complete source-pinned hashes;
the fail-closed unpinned-loader test explicitly removes pins rather than relying
on a permanently unavailable layer. Together with layer0, feature projection,
final normalization and selector fixtures, the staged selections cover all81
tensors and3,848,808,960 payload bytes of the pinned DFlash2 checkpoint.

Shared target head integration was checked against the saved native `model.py`
whose SHA256 is c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391.
Its `_lm_head` applies `ttnn.linear` with borrowed vocab-sharded weights and
then gathers logits. The learned drafter must use its own final normalization,
not the target's `_final_norm_decode`; sharded top16 selection could avoid the
full-logit gather, but still requires a separate exact candidate/ID merge gate.
No shared-head adapter is claimed implemented or benchmarked yet.

### First matched verifier profile: correctness passed, attribution failed

Hardware34144356177 completed24 batch checks and16 rollback checks; its
numerical report passed. All18 T8 profiling replays also passed logits/GDN/KV/
selected-checkpoint comparisons. However, every native serial replay dropped
profiler markers with the10000-op buffer, and the Python host-log importer was
killed with exit137. The overall CI gate failed; this is not a valid complete
attribution result or a throughput improvement.

The retry uses20000-op buffers and the already emitted C++ device-operation
CSV directly, avoiding expansion of all warmup host metadata. The checker
still rejects any dropped measured markers and missing chip/replay coverage;
the saved failed run is rejected by this same checker. Diagnostic candidate
rows showed matmul and generic operations dominating summed device durations,
but summed durations are not a traced critical path. No performance claim is
made from the failed profile.

`draft_shared_head.py` now contains an experimental adapter borrowing the
native vocabulary-sharded head after learned normalization. Four32768-wide
top16 chunks per chip avoid a full-logit gather, with negative-infinity padding
for the final chunk and checked global-ID merging. Host tests cover shard/chunk
boundaries, missing/padded IDs and proposal rows1..7. This adapter is not yet
simulator/hardware certified or integrated with the full drafter. Ties are
deterministic among received candidates; matching MLX's unordered candidate
selection at tied cutoffs is not claimed.

### Kernel-level diagnostic and next copy optimization

Streaming host metadata joined to selected candidate CSV rows identifies the
packed baseline's convolution-prefix copies as144 calls and about4.24ms summed
kernel duration per T8 block on chip0 at4095 context. State copies contribute
about2.58ms and convolution-window construction about2.42ms. The streaming
reader rejects absent metadata and changed per-kernel replay coverage, and
does not expand the full warmup host-time log. This selected-arm diagnostic
does not turn the failed full profiling run into a pass.

The profiled configuration lacks the norm-batch/grouped-attention improvements
of the historical64.44/66.11ms best static result; its costs must not be presented
as attribution of that best result. Convolution-prefix copying is common work
worth testing independently. `copy_prefix(..., reuse_zero_tile=True)` now offers
an opt-in candidate that initializes each worker's output padding once, rather
than once per page. Only the two active row segments are overwritten between
pages, with the existing write barrier retained. The default remains unchanged;
exact padded-output simulator and hardware timing gates are still required.

Two-layer simulation164514Z failed at learned layer1's down projection:6 of40960
values exceeded the unchanged comparison tolerance. It was closed normally,
and the dependent selector gate correctly did not launch. Retry171550Z keeps
the same numerical work but captures down inputs, actual/reference outputs,
layer identity and pinned manifest on failure for offline ISA replay. No
tolerance increase or hardware promotion was made.

The independent `gdn-prefix-copy` simulator gate is now prepared and queued
after the active MLP capture run. It checks all63 supported prefix/width pairs,
both ranks and all four slots, comparing default and zero-tile-reuse kernels
against the selected source row plus every physical padding element. Output
buffers start with nonzero padding canaries; packed sources must remain
unchanged. The slow-dispatch wrapper preserves its existing commit-gate
default. This scheduling dependency requires the prior process to terminate,
not the unrelated learned-layer numerical test to pass.

### Profiler resource isolation retry

CI34146165134 failed when the profiled child was killed during the second
context; it produced no complete runtime device CSV. Resource pressure is a
hypothesis, not a confirmed OOM diagnosis. The next retry separates the full
24-case batch and 16-case rollback correctness matrix, including all four
negative-control pairs, into an uninstrumented process. A fresh process then
profiles only the two T8 contexts with three native/control/candidate replays.
The checker requires both reports; this does not waive broad correctness.
Runtime C++ post-processing is enabled before Python starts, Tracy child exit
codes are checked, and cgroup memory events/peak are saved for diagnosis.
Instrumented durations remain attribution only, never throughput evidence.

The171550Z two-layer rerun completed with passed=true, exit0 and clean mesh
closure. Both layers/ranks report zero MLP projection error and exact connected
convolution/residual/fabric checks; the earlier six-value mismatch did not
reproduce, so no failure capture was generated. This is not an explained fix
or sufficient evidence to promote the multi-layer drafter to hardware. The
down-replay diagnostic now supports preserving all input terms/rows while
selecting only failing output columns, for a faster ISA-versus-FP64 comparison
if the failure recurs. The independent prefix-copy gate starts next; the old
queue was stopped after its terminal-PID check encountered null output.

Prefix-copy174127Z and174322Z failed in canary setup, before the candidate
kernel: logical-volume-changing reshapes are not supported. The corrected
gate uploads compact tensors with explicit pad_value=17 and checks the initial
physical canary via host to_torch_with_padded_shape, without a reshape.
Run174502Z passed all63 prefix/width cases on both chips, including every
padding element, unchanged sources, exit0 and clean mesh closure. This clears
the simulator correctness gate for opt-in zero-tile reuse, not its hardware
timing or full-model promotion gates. A five-distinct-layer simulator run with
the final learned norm/selector projection is next; the earlier intermittent
two-layer mismatch remains unexplained and numerical tolerances stay fixed.

The checkpoint-cost CI suite now begins with the simulator-validated prefix-copy
gate on explicitly allocated hardware. It preserves all63 exact padding cases,
then captures48 copies to shared hot buffers per arm for each width, alternating
five measured control/candidate replay pairs after warmup. Every replay checks
both ranks' physical output. Timings include blocking host dispatch and exclude
uploads/readback; this isolates the copy candidate but is not a model-level
speedup or a realistic distinct-layer working-set measurement. Existing GDN
checkpoint diagnostics still run afterward. No production call site enables
zero-tile reuse. The five-layer simulator and profiler CI remain independent.

The next shared-head simulator gate uses the production candidate-chunk helper
on synthetic full248320-vocabulary logits, sharded across both chips. It checks
each local score/index association, top16 scores, exclusion of padded tail IDs,
and exact global proposal IDs across chunk/chip boundaries, including248319.
This does not test the borrowed LM-head matrix multiplication or tie-equivalence
at the learned-model cutoff. It is prepared, not executed; it must wait for the
active five-layer simulator to terminate before taking the simulator devices.

Profiler CI34148441153 failed again, now with confirmed cgroup OOM evidence:
memory.peak103079301120 bytes (96GiB), oom3 and oom_kill1. The separate broad
correctness report was saved, but the profiling child was killed during the
second context's final serial replay. Child-exit checking correctly failed CI;
this is not a complete attribution result. The next resource fix must isolate
contexts in separate processes and preserve per-process trace identities rather
than rerunning the same memory-growing job or increasing its limit blindly.
Checkpoint-cost CI34149243028 has started independently.

Checkpoint-cost CI34149243028 passed: all63 prefix-copy correctness cases and
60 measured replay records. At T8, median blocking48-copy trace time is
1.445458ms control versus0.746584ms zero-reuse (about48.35percent less).
This is a shared-hot-buffer microbenchmark, not an end-to-end coding speedup.
Production copy call sites remain unchanged pending full-model validation.

The next profiler retry retains the broad uninstrumented correctness process,
then starts one fresh instrumented process per4095/16383 context. Context-local
CSV, generation report and console are checked independently, so reused numeric
trace IDs cannot cross-contaminate attribution. Both contexts, three arms,
three replays and both physical chips remain mandatory in the combined result.
The scoped context flag cannot narrow a correctness-only matrix.

The full-model prefix-copy experiment now forwards an explicit default-false
option through ModelBatch, DeviceLoopState, convolution final-state copying
and both checkpoint publications. Prefix0 still restores the original snapshot
without invoking a packed copy; native T1 remains unchanged. The matched timing
control retains norm batching and all grouped/DMA/parallel/T8 attention settings,
changing only zero reuse. CI selects this using prefix_zero_reuse=true with the
full-attention-tree suite; other suite combinations fail before hardware work.
This retains the full static correctness/rollback matrix and end-to-end verifier
timing, but does not establish committed coding throughput or alter serving defaults.

### Complete profiler evidence: CI34149690873

The context-isolated profiler passed CI and local revalidation of its downloaded
CSV/report/console artifacts. The independent correctness matrix contains24
batch cases,16 rollback cases and4 negative-control pairs. Both contexts have
three exact replays of native/control/candidate on both physical chips, without
dropped markers inside measured boundaries. Cgroup oom and oom_kill are zero;
peak usage remains near96GiB, so this is not evidence of generous memory headroom.

For the packed-GDN/ordered-cache candidate, chip0 median summed kernel durations
are32.023/32.007ms for matmuls,22.463/22.460ms for generic GDN/cache operations,
and5.682/10.081ms for SDPA at4095/16383 contexts. Each replay has2152 operations
versus11448 native serial operations. These sums are attribution, not critical
path or throughput, and this profile lacks the norm-batch/grouped-attention
settings of the best static verifier. Matmul core counts vary32/39/43/56/108;
SDPA uses110 workers. The workload is not uniformly limited to36 cores.

The full-model zero-reuse A/B run34150091732 is now active on the stronger
attention configuration. Its result, not the copy microbenchmark or instrumented
kernel sums, must establish any verifier-level improvement. The separate
five-layer learned simulation continues; no learned-request rate is certified.

The prior91-worker fused gate/up kernel was measured only at B1 and missed
repeatable full-model improvement. A new simulator-only gate prepares explicit
T1/2/4/8/16/32 inputs without changing its generated compute or reader kernels;
the default callable still accepts only T1. It uses pinned draft MLP weights as
geometry-matched operands, checks separate versus pair-packed BF4 quantization,
and requires exact native separate-projection/SwiGLU output on both chips.
This is prepared, not yet executed, and does not validate target-model weights,
coding quality, or batched performance. It follows the active learned-stack and
queued shared-head gates rather than contending for simulator devices.

### Five-layer simulator gate completed

Run174646Z passed all five distinct pinned learned layers, both chips and the
final learned normalization/selector hidden projection; exit0 and clean mesh
closure were verified. Final normalization is within one BF16 ULP, selector
projection error is zero and its BF16 cast is exact on both chips. The earlier
intermittent two-layer down mismatch did not recur. Evidence is preserved as
hardware-evidence.local/sim-five-layers-20260907T174646Z.json with its exit status.
This remains a synthetic-input31-context mathematical gate, not real target
feature history, full shared-head/selector token selection, acceptance, coding
quality or throughput validation. The independent full-vocabulary shared-head
simulator test started only after this process terminated.

Shared-head simulator184946Z also passed, with exit0 and clean closure: all
eight chip/chunk combinations return exact local top16 scores with in-range
IDs, and the merged seven proposal rows match full-vocabulary global IDs and
scores exactly. This tests candidate extraction from synthetic logits, not the
borrowed LM-head multiplication. Multi-row fused projection is the next simulator
gate; the existing fast-dispatch wrapper now explicitly allows both probe names.

Five-layer hardware promotion is opt-in via learned_stack=true with the
learned-attention CI suite. It stages all four remaining pinned layer fixtures
and selector tensors into the isolated container, rehashing cached files and
failing rather than replacing corrupt cache entries. A health check precedes
the exact simulator-validated short-context five-layer/selector-projection
configuration. This branch does not rerun the unrelated long one-layer sweep;
the existing suite remains unchanged when the option is false. No request or
throughput certification follows merely from this mathematical hardware gate.

### Prefix-copy full-model result and next gates

Hardware run34150091732 passed the full-model static correctness, rollback and
negative-control gate with prefix_zero_reuse=true. T8 interleaved A/B median
costs were64.4831ms control versus62.3926ms candidate at4095, and66.1415ms
versus64.0525ms at16383. Median paired speed ratios were1.033388 and1.032637.
Separate candidate block medians were62.3475/64.0012ms. These are verifier
costs, not committed coding throughput: even perfect acceptance with zero
draft/commit overhead gives only128.31/125.00tok/s. Reaching200 still requires
at least35.84/37.50% further verifier reduction before paying draft overhead.
The archived compact_comparison.scope string incorrectly describes the older
T4/T8 attention experiment; paired_control and the executed control flags
correctly identify the zero-copy-only A/B. Future report scope is corrected;
the downloaded evidence is preserved unchanged.

Five-layer learned hardware run34154133218 was dispatched from bb17e86 after
607 host tests passed. No reset or serving-default change was requested.
Multi-row fusion simulator185201Z failed before kernel execution because its
separate/paired BF4 weight equality prerequisite failed; no width passed.
Diagnostic simulator190512Z retains that strict prerequisite and records
source packing equality, finite checks and differing coordinates on both chips.
It is not approved for hardware until the simulator gate genuinely passes.

Simulator190512Z localized the packing discrepancy to one chip0 up-weight at
row1458/column3022: packed0.000732421875 versus separate0.1875, from source
BF16 value0.1767578125. All source packing checks and the other three chip/
projection checks were exact. No fused kernel ran. Diagnostic191304Z then
failed pinned fixture hashing before opening devices. A standalone Python
read loop (no TTNN/Torch) produced8 failing full-file hashes in15 reads;
rehashing each already-read buffer was stable. Windows and WSL sha256sum
checks matched the pins. Subsequent streaming/whole-file diagnostics and a
new18-read test across DrvFS and an independently staged ext4 copy all passed.
This is an unresolved intermittent local data-integrity issue, not evidence
that the fusion algorithm or quantization tolerances should be changed.

The reusable fixture_integrity_probe.py compares two retained whole-file
buffers with streamed bytes and records mismatch coordinates without retrying
away failures. Simulator191717Z uses the verified ext4 fixture and retains the
strict packing gate, with host tile quantization and repeated-readback
diagnostics if a mismatch recurs. This does not establish a filesystem fix.
The independent five-layer hardware CI remains in progress.

Simulator191717Z exited1 after reaching native projection compilation/execution:
`UnsupportedFunctionality: tensix_pacr: Disable_pack_zero_flags`. This is the
known TTSim packer-L1-accumulation limitation, not a fused numerical pass. The
process died before its finally block could write the report; subsequent probes
persist weight validation and dispatch phase before native calls. Do not disable
packer accumulation and label that a validation of the unchanged target kernel.
The separate learned selector codebook dot gate is the next executable simulator
prerequisite; fusion remains unvalidated and is not promoted to hardware.

Learned selector scoring simulator192510Z passed with clean close and exit0:
both chips' dot maximum error3.57628e-7, complete transition-score maximum
error3.11447e-7, and exact seven-token greedy proposal IDs. This uses pinned
codebooks with synthetic hidden/candidate/unary inputs; gathers and greedy
selection are still host-side. It is not full-device selector or request TPS.
The opt-in learned_stack hardware suite now runs this cached64-worker,
keys32/width256 gate after health and before the five-layer gate, including
bounded repeated latency measurements. Other shapes/placements remain guarded;
the default suite and serving configuration are unchanged.

### Upstream packer compatibility investigation

Upstream [PR53805](https://github.com/tenstorrent/tt-metal/pull/53805), merged
as a00d91e585e49282e54c28a442a8fcc31764afca, removes the unused pack-zero
tracking flag write while retaining Pack_L1_Acc. Its stated rationale is no
change to the pack data path. The exact Blackhole hunk is preserved in
optimisation/sim/blackhole-packer-zero-flags.patch. Local source header SHA
changes from87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181
to8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7.

Simulator193027Z tests this audited graft with packer accumulation still on,
unchanged exact equality requirements and an isolated JIT cache. The wrapper's
opt-in QWEN_SIM_PACKER_ZERO_GRAFT=1 verifies the modified header hash; the probe
records that hash and flag. The local simulator header is temporarily patched
for this run and must be restored after it finishes. Neither CI runtime nor
serving is changed. This is an explicit upstream-grafted configuration, not
proof that the original unpatched kernel passed simulation.

Simulator193027Z completed weight validation and executed both native and fused
kernels under that graft, then failed T1/chip0 equality with3618 differences.
Review found an incorrect probe oracle: it used an unfused gate matmul followed
by standalone SiLU, inserting BF16 rounding before activation. The existing
target projection gate in projection-1d.py and MLP configuration in mlp-sweep.py
use native gate-linear fused SiLU, then BF16 pack. The probe now uses that
control through native_gate_up_control, with a regression test protecting the
activation placement. Neither candidate math nor exact equality is loosened.
Simulator194000Z is testing the corrected control under the same audited graft;
the local header remains temporarily patched until this run finishes.

Simulator194000Z failed its weight prerequisite before corrected-control
execution: chip0 gate[242,8110] initially read as-0.00048828125 instead of-0.125.
Host tile quantization and both repeated device reads returned-0.125 without
any intervening device write. This local readback-path discrepancy is preserved
as a failure, not retried into a pass. The patched header was restored after
terminal exit; its SHA again matches87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181.

Hardware34154133218 failed at the2400-second remaining-layer fixture staging
timeout, before device tests. Queued34155661944 then failed because the
incomplete directory was intentionally not overwritten. Recovery now hashes
every completed tensor against its audited pin before reusing it, stages missing
tensors in independent temporary directories with at most three concurrent
downloads, verifies each before exclusive hard-link publication, and writes a
manifest only after all succeed. Old partial files are preserved; corrupted
completed files still fail closed. Unbuffered progress and a4800-second staging
bound replace the silent40-minute timeout. Both prior hardware failures are
infrastructure results, not learned-layer or selector numerical failures.

### Bytewise device packing prerequisite

The isolated packed_weight_check dataflow kernel compares every144 uint32 word
in each576-byte BF4 tile against the corresponding separate gate/up tile. No
floating-point conversion or tolerance is involved. Each worker publishes page
coverage, a completion sentinel, mismatch count and its bitwise complement;
the host checks all counters and untouched output padding. This avoids returning
entire dequantized weight matrices through the unreliable local bulk-readback
path rather than accepting a failed read on retry.

Simulator195634Z passed18 cases/36 chip checks at32x32,64x96 and320x256,
including12 single-chip negative cases, both tile offsets, fewer-than64 and
more-than64 page distributions. It closed cleanly with exit0. Host tests cover
complete full-size tile mapping and rejected incomplete/corrupt readback reports.
The fusion probe's explicit --device-weight-check option retains host source
packing equality and replaces only device-weight readback validation with this
byte-exact check. Simulator195732Z tests full5120x8704 geometry and the corrected
native gate-linear fused-SiLU control. It uses the audited upstream packer graft;
the local header must again be restored when this active run terminates.

Simulator195732Z failed before any fused projection: the device comparator
found one mismatched word in the chip1 gate and host source packing equality
also failed for that same chip/projection. The other three comparisons covered
all43520 pages each with zero mismatches. This rules out treating the issue as
only bulk output readback; local source/data integrity remains unresolved.
No retry was accepted as success. The local packer header was restored and
its original87b9c251... hash verified after the process exited1 cleanly.

### Five-layer and selector hardware gates passed

Hardware34157187322, commit062fedd, passed after verified staging recovery.
The five-layer report contains62 checks, including all10 chip/layer connected
MLP checks with zero gate/up/down/conv projection errors. All five distinct
learned layers completed on hardware at context31/block8. Final normalization
is within one BF16 ULP and final selector projection has zero error/exact cast
on both chips. The selector codebook dot gate matches simulator errors and
exact proposal paths; five isolated dispatch samples are0.39884,0.33107,
0.31394,0.30609,0.30485ms (median0.31394ms). These include dispatch/allocation/
synchronization, not codebook gathering, shared LM head or complete selection.

Artifacts are retained under hardware-evidence.local/34157187322. This proves
the short synthetic-input mathematical gates, not real target feature history,
learned drafting acceptance, complete request latency or coding quality. The
200 committed-token/s objective is still open; serving defaults are unchanged.

### Reusable learned MLP parameters

The learned MLP execution helper now separates prepare_mlp_branch from
execute_mlp_branch. Preparation owns nine device tensors (norm, dynamic-kernel
projection, four convolution bases and gate/up/down weights); repeated execution
can reuse them without weight splitting or host uploads. Prepared assets are
bound to the exact operations/mesh and learned-layer dictionaries. Per-block
activation ownership remains separate, so releasing a completed block does not
release its reusable weights. Default diagnostic callers still prepare their
own assets and retain the same arithmetic/rounding configuration.

Host tests verify two executions reuse all nine uploads, preserve projection
order, and reject assets from another mesh/layer. Simulator220731Z executes
three real-weight MLP blocks with seeds731/732/731, validating all intermediate
stages and both chips, checking changed inputs and exact repeated outputs, then
freeing each block while retaining stable parameter addresses. It uses the
original restored LLK header, not the packer graft. This gate is in progress;
no timing gain or complete reusable five-layer drafter is certified yet.

Simulator220731Z passed all three blocks and both chips (exit0, clean close).
All six connected-MLP checks have zero projection errors, zero activation ULP
distance and one normalization ULP; exact repeated outputs and stable nine-buffer
parameter ownership passed after freeing each block. Hardware promotion adds the
same replay gate to learned_stack=true after health, before selector/stack checks.
With --hardware --timing it runs one warmup plus five measured changing-input
executions, retaining full stage validation outside each timed region. Samples
cover eager MLP dispatch, activation allocation, collectives and synchronization;
weight/input transfer and host reference checks are excluded. This is a branch
cost measurement, not traced latency, an upload-inclusive A/B or full drafter TPS.

### Learned MLP trace-replay prerequisite

Simulator223954Z preallocates all nine learned parameter tensors, both host input
payloads and one persistent device input before capture. It first validates eager
outputs for seeds731/732 against the existing stage references, then captures the
unchanged prepared MLP execution. Input copies update the same device allocation
for the731/732/731 replay sequence. Both-chip final outputs must exactly match
their validated eager references, with immutable input contents and stable input,
parameter and output addresses. An extra replay without an input update must
retain the old output and fail comparison to the other input's reference.
Trace ownership is released before activation/parameter buffers. This new gate
has no hardware mode or timing claim until simulation passes. Hardware34167225933
separately measures the already validated eager parameter-reuse path.

Hardware34167225933 passed all six changing-input eager MLP iterations and the
five-layer gate. Five post-warmup branch samples were5.543028,3.938810,9.372616,
4.182100,3.930729ms (median4.182100ms). This excludes weight/input transfer and
validation and is neither full-drafter latency nor committed-token throughput.

Simulator223954Z exited1 after both eager references passed: capture rejected
the host synchronize_device call inside grouped_causal_convolution. No traced
output passed. The failed report is retained locally. The fix adds explicit
caller-owned temporary retention to convolution and gather-add, avoiding their
internal host waits and frees only in the opt-in trace-safe MLP path. Existing
eager callers remain unchanged. Captured temporaries stay alive until trace
release;620 host tests pass, including deferred ownership checks. A fresh
simulator gate must pass before this path can be promoted to hardware.

Simulator230240Z on5a7d576 passed with exit0 and clean mesh close: four eager
stage validations, six exact changing-input trace comparisons and two stale-input
negative controls. The trace-safe ownership path is now eligible for opt-in
learned-stack hardware testing. That suite retains eager timing and adds the same
captured MLP gate with one warmup and five blocking trace timings. Input copies,
weight uploads, capture and output validation remain outside timed regions.
This is an isolated branch cost, not paired full-drafter or committed throughput.

### Matched best-verifier attribution

The earlier profiler34149690873 omitted batched normalization and the grouped
attention optimizations present in the62--64ms best static T8 verifier. Its summed
kernel durations therefore cannot identify the remaining critical path of that
configuration. The verifier-profile suite now builds the already validated
disposable compact native SDPA scratch and uses all six best flags together:
norm-batch, grouped-attention, attention-dma, attention-parallel, attention-tree,
prefix-zero-reuse. No new kernel arithmetic is introduced. The separate broad
correctness process and both instrumented context processes use identical flags;
the artifact checker rejects mismatched or partial configuration metadata.
Three exact replays per arm/chip/context remain required. Profile sums remain
attribution, not latency or committed throughput. Hardware execution is pending.

### Hardware MLP trace result34170293087

Run34170293087 on1fdceef passed the health gate, eager MLP gate, captured MLP
gate, selector dot gate and five-layer gate. The captured MLP report contains
four eager stage validations, twelve exact replay comparisons across both chips
and two stale-input negative controls. Five post-warmup trace samples were
1.912000,1.910169,1.911340,1.907680,1.914480ms (median1.911340ms).
The separate eager process measured3.829159,3.634698,3.950039,3.897369,
4.293291ms (median3.897369ms). These sequential process measurements are not
an interleaved paired benchmark: they establish the isolated traced branch cost,
not a full-drafter speedup. Uploads and validation are excluded. The connected
five-layer gate also passed, but it does not yet capture the whole learned draft
or provide committed coding-stream throughput. Matched best-verifier attribution
run34170503918 is now in progress;200 committed tok/s remains unachieved.

### Matched best-verifier result34170503918

Run34170503918 on4dff922 passed with identical complete best-configuration flags
in correctness and both context profiles:24 batch checks,16 rollback checks,
four negative controls and three exact replays of each arm on both chips.
Candidate operation count is2040 per replay. Chip0 median summed kernel costs
at4095/16383 are matmul32.061658/32.009741ms, generic18.241194/18.260368ms,
SDPA2.546763/4.211429ms and all-gather1.992017/2.005627ms. These instrumented
sums are not critical-path time and must not be converted into committed TPS.

The retained C++ CSV separates the largest candidate groups at4095:39-core
matmuls11.808597ms across128 operations;32-core matmuls10.805054ms across128;
48-core generic7.198087ms across336;96-core generic6.252838ms across48;
43-core matmuls5.733327ms across48. Core count alone does not identify a kernel
or model projection. Next prioritization is source/dispatch mapping of the two
dominant matmul groups and the48/96-core custom groups, rather than assuming
unused cores imply attention is the main limit. The analyzer now retains these
operation/core-count groups and rejects changing group coverage across replays.

### Multi-row fusion simulator005625Z

After a standalone retained-buffer/streamed SHA check passed on all three pinned
MLP source files, simulator005625Z passed bytewise packed/separate BF4 comparisons
for gate/up on both chips:43520 pages each, zero mismatched words, source_exact
true. Fused output then matched the corrected native fused-SiLU control exactly
atT1/T2/T4/T8/T16/T32 on both chips. Exit0 and clean close were verified. The
temporary upstream packer-zero-flags compatibility header was restored afterward;
hardware runtime and serving remain untouched. These are DFlash-weight geometry
operands, not full target-model quality or a resolution of historical RAM failures.

The learned-mlp hardware suite now runs the byte-checked fusion gate after health,
before its existing learned MLP gates. All six widths must match; T1/T8/T32 each
receive three eager ABBA timing blocks, with every timed output validated on both
chips outside the timer. Timings include dispatch/allocation but exclude uploads,
validation and deallocation. No captured-fusion or full-model speedup is claimed.

### Multi-row fusion hardware34176158014

Run34176158014 onf1a4dc5 passed all twelve width/chip output checks and all four
byte-exact packed-weight/source checks. Every timed output was also exact. Eager
ABBA median block means (native/fused) were0.302722/0.763854ms atT1,
0.239781/0.760774ms atT8 and0.247471/0.649273ms atT32. This is a regression,
not an optimization winner; no full-model adoption is authorized by this result.

FusedProjection.__call__ constructs mesh programs, per-core runtime arguments
and descriptors on each eager invocation. That host work is included in these
timings. Its contribution versus device execution has not been isolated, so do
not conclude that fusion itself is intrinsically slower or claim that tracing
will fix it. Next test changing-input captured fusion and native control in the
simulator, preserving exactness/ownership, then compare matched hardware trace
latencies before deciding whether full-target integration is worthwhile.

### Captured fusion prerequisite

The simulator-only --trace-replay gate retains all six eager width checks and
adds T1/T8/T32 native/control versus fused traces. Each width owns one input
allocation, two prebuilt host input payloads and both captured output sets.
References come from native eager execution for both inputs. Replay follows
A/B/A with exact outputs and immutable inputs on both chips, stable addresses,
and missing-update negative controls for both arms. Both traces are released
before captured buffers, and before allocating the next width. Hardware trace
promotion remains rejected until this simulator gate passes. Eager hardware
results and kernel arithmetic are unchanged by this new test path.

Simulator012750Z on a3ed767 passed all six eager widths plus captured T1/T8/T32:
36 exact changing-input arm/chip comparisons and12 stale-input negative controls,
with four byte-exact source/device packing checks. Exit0 and clean close verified.
The local compatibility header was restored. Hardware trace timing is now enabled
in the learned-mlp suite, using the same traces and three ABBA blocks per width.
Each blocking execution is checked outside the timer; capture, allocations,
uploads and readback validation are excluded. Eager regression remains recorded;
no captured performance improvement is established until hardware reports it.
