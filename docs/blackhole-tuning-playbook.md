# Tuning models on two Blackhole P150A cards

A reusable method, not a promise that Qwen settings transfer to another model.
Keep this guide curated; detailed logs belong in the experiment programme.

## Start here

| Document | Purpose |
|---|---|
| [README](../README.md) | Latest qualified results and coverage |
| This playbook | How to tune, measure and decide |
| [Experiment record template](tuning-experiment-template.md) | One reproducible record per hypothesis |
| [Programme](two-card-experiment-programme.md) | Detailed backlog, history and acceptance requirements |
| [8K investigation](draft-8k-numerical-investigation.md) | Attention precision and context-extension case study |
| [Bias cache experiment](markov-bias-cache-experiment.md) | Correctness success that became a performance rejection |
| [Context and user capacity](context-and-user-capacity-plan.md) | 131K/262K window qualification and concurrent-user sweeps |

## 1. Record the machine and model before tuning

| Inventory | Record and verify |
|---|---|
| Cards | Product, device IDs, firmware, harvesting masks, exposed compute grids, DRAM capacity |
| Host | CPU/RAM, NUMA placement, PCIe negotiated width/speed, switches, competing processes |
| Fabric | Physical cables, discovered active links, descriptor, requested and actual collective link counts |
| Runtime | Container digest, TT-Metal commit, native-library hashes, patches, driver and simulator versions |
| Model | Weight/tokenizer/config revisions and hashes; dense/MoE/hybrid layers; per-operator precision |
| Workload | Coding datasets, templated input lengths, output lengths, sampling, streams, prefix reuse and latency budget |

Our reference host has one x16 card and one x4 card behind a switch, connected
by two QSFP-DD cables. Do not infer physical topology from the logical P300
mesh name. Verify it. Four requested links are not proof of four used links.

For a new model, inventory attention KV storage and recurrent state separately.
An allocator reporting 0% KV usage is not proof that attention does not cache:
trace allocation, append, read and reset paths, and check counter ownership.

## 2. Establish a trustworthy baseline

1. Pin code, weights, tokenizer, sampling and topology. Retain a rollback revision.
2. Run native target correctness before introducing a drafter or new kernels.
3. Measure cold loading, prefill, request setup and warm generation separately.
4. Repeat complete requests; retain raw timings, output IDs and actual EOS counts.
5. Measure multiple coding prompts. A single favourable fixture is not quality acceptance.

| Metric | Definition and boundary |
|---|---|
| CTX | Actual templated input token count; report generation headroom separately |
| PP | Input tokens / measured prefill seconds; exclude loading and disclose setup separately |
| Complete-generation TG | Committed output tokens / complete generation-loop seconds, including draft, verify, commit, copies and stalls |
| Steady-state TG | `(N - first_commit_count) / (last_commit_time - first_commit_time)`; unavailable for an empty window |
| TTFT | Request arrival to first generated token; state whether measured at engine or client |
| Total request time | Request arrival to completion, including prefill and setup |
| Streaming gaps | Client event-gap p50/p95/p99/max; an event is not necessarily one token |
| Aggregate throughput | All completed output tokens / wall time; report concurrency, never label it single-stream TG |

Our initial ladder is 128/4096/8192/32768/65536 input tokens, batch 1.
Aim for 1024 generated tokens for sustained measurements; first verify ignore-EOS
support. If generation stops naturally, report the real count rather than padding
or relabelling the run. Extend coverage to 131072/262144 total-token windows
with generation headroom inside the limit. Then sweep active users 1/2/4/8/16
where memory and scheduler support permit; record actual batch size separately.
Follow the [capacity plan](context-and-user-capacity-plan.md); these extensions
are planned, not supported-runtime or user-capacity claims.

## 3. Find the bottleneck, not the emptiest core

Profile the combined runtime, then map expensive zones to operators and shapes.
Record per-chip times, transfers, layout conversions, synchronization and memory
traffic. Distinguish measured attribution from inferred operator-family labels.

| Evidence | First experiment |
|---|---|
| Device waits while host works | Reduce synchronization/readbacks; overlap independent bookkeeping |
| Repeated intermediate DRAM traffic | Fuse producer/consumer stages; retain intermediates in L1 where feasible |
| Weight bandwidth dominates | Sharding, multicast, prefetch and double buffering; do not assume more workers help |
| Recurrent state traffic dominates | Avoid unnecessary snapshots; preserve exact accepted-prefix commit/rollback |
| Verification dominates speculation | Reduce verifier cost and tune draft width against committed TG |
| Long context hurts | Attribute target attention and draft-history attention separately; inspect padding and chunk count |

For speculation, optimize `committed tokens / (draft + verify + commit time)`.
A longer draft or higher acceptance can still lose. At 11 committed tokens per
block, 200 TG requires a complete cycle of at most 55 ms; this is a budget,
not a measured capability or a guarantee across prompts.

## 4. Run experiments through explicit gates

| Gate | Required evidence | Does not prove |
|---|---|---|
| Host tests | Shapes, indexing, guards, ownership, fixture sensitivity | Device arithmetic or speed |
| Synthetic simulator | Actual native rank/dtype/layout; eager and changed-input replay; poison, stale-state and cleanup checks | Physical DRAM/fabric performance |
| Hardware correctness | Real weights, full vocabulary, target token/state/KV equality and rollback | Speed or held-out coding quality |
| Combined performance | Matched complete requests, same workload and optimizations, repeated positive effect beyond variability | Serving behaviour |
| Model/serving acceptance | Held-out coding tasks, context ladder, natural stops, streaming and concurrency regressions | Universal portability to other models |

Use planned, host-tested, simulator-passed, hardware-correct, benchmarked,
adopted or rejected as distinct statuses. Never reduce a correctness threshold
to rescue a speed candidate. Algorithmic equivalence is not bitwise equivalence.

Change one hypothesis per paired comparison. Keep all other winning optimizations
in both arms. Run an audited request separately from timed requests, then use
balanced ordering such as A/B/B/A. Repeat promising wins in a separate run.

## 5. Treat topology as three different experiments

| Experiment | Benefit being tested | Prerequisite and comparison |
|---|---|---|
| Ethernet dispatch | Move dispatch off Tensix; potentially expose another column | Resolve firmware sizing, fabric coexistence and per-chip grid discovery |
| Recovered-column kernels | Lower generation latency using additional compute or prefetch workers | Worker/110 vs Ethernet/110, then Ethernet/120 only if those grids actually exist |
| Fabric weight loading | Avoid slow x4 host upload via x16 staging and fabric distribution | Exact final shards, measured traffic, temporary memory and total loading time |

Fabric loading does not automatically reclaim dispatch cores. Resident-weight
decode does not reload the model over PCIe each token. Dispatch/core recovery
belongs alongside verifier optimization after profiling; upload routing is a
separate startup experiment. Neither path is qualified on our pair yet.

## 6. Lessons from this programme

| Observation | Reusable lesson | Evidence |
|---|---|---|
| Shared-QK combined runtime repeated at 106.58 TG/4K and 101.59 TG/8K | Preserve combined winners; qualify context separately | Runs 34702526963 and 34730226400; README |
| Bias cache passed correctness but fell from 100.90 to 93.91 TG at 8K | Saved arithmetic can cost more in traffic/dispatch; reject on complete-loop timing | Run 34736316338; bias-cache case study |
| Native embedding returned rank 3, while a precision selector expected rank 4 | Synthetic fixtures must use the real producer's shape, not merely equivalent contents | Bias-cache case study, runs 34734406268 and 34735206013 |
| Rebuilt native code was rejected by the old source gate | Admit the exact validated transformation and both loaded libraries; never disable provenance checks | Runs 34735942778 and 34736316338 |
| Ethernet dispatch hit mapping and firmware-size failures before mesh operation | Constructor support is not operational support; do not count hypothetical cores | Programme topology section |

The quoted TG results use a 121-token EOS coding fixture, one stream, not a
1024-token sustained test. They do not establish held-out quality or 200 TG.

## 7. Keep iteration fast and recoverable

- Run cheap host checks locally; use CPU CI for synthetic tensors, not full-model simulation.
- Build once per compatible shape family; run each risky probe in a fresh bounded process.
- Cache native builds by image, builder, patch and source hashes; verify both library locations on hits.
- Reserve hardware for simulator-qualified arithmetic and complete-runtime comparisons.
- Distinguish queued, running, timed out and failed. Check the exact handle before retrying; no duplicate runs or unrelated resets.
- Archive failure evidence before cleanup. Keep compact reports and hashes beyond CI artifact expiry; do not rely only on local evidence folders.

## 8. Documentation is part of experiment completion

For every experiment, save the filled record, exact invocation, immutable revision,
run/attempt IDs, report hashes, raw measurements and adoption decision. Record why
it failed and what would justify revisiting it. Preserve commit history.

Update the README only with qualified measurements. Update the programme with
pending gates. Update this playbook only when a lesson is supported by evidence,
and label model-specific assumptions. Research leads remain hypotheses until
implemented and tested; external GPU results are not Blackhole acceptance.
