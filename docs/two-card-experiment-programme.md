# Qwen3.8-27B: two-card experiment programme

Current request-engine checkpoint: T32 changed-input replay and synchronized
publication passed run34041390068 (36 two-block fixtures). Device force-argmax
passed34040918809; T32 verification plus readback is131.190/148.752ms at4K/16K.
These component results do not establish200 committed tokens/s. The next pilot
uses actual lookup proposals, exact native token/state comparison, complete
proposal-to-commit timing and separately reported per-request capture cost.
Repeated-code pilot prompts cannot establish representative coding quality.

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
| E3 verifier | Exact 4K/16K multi-token verification, corrected rollback and paired timing | Dynamic acceptance/commit pipeline and substantially lower V(T) |
| E4 fusion/pipeline | Packed histories, ordered cache writes and captured96-worker commit passed full-model gates; T16 costs89.552/98.325ms at4K/16K | T32 simulator prerequisites and native hardware cost curve; device selection; FP32-preserving wider GDN worker mappings |
| E5 drafting | Request-local lookup and greedy session/accounting host tests; historical MTP groundwork | Reusable card-backed verifier and matched end-to-end evaluation; no DFlash2/DSpark/EAGLE3 TT adapter certified |
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
P150A hardware result in this repository. Lookup proposal policy has host tests,
not an integrated device-throughput result. The older EAGLE3 discussion is analysis,
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
