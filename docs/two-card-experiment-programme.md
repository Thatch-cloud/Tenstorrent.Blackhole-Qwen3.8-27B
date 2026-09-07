# Qwen3.8-27B: two-card experiment programme

## Current measured result - 2026-09-07

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
