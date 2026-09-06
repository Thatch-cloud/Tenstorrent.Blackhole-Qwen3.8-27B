# Qwen hardware CI inventory

The manual workflow targets the existing organization runner
`thatch-build-amd64-02-cp-temp` using its dedicated custom label
`thatch-qwen-p150a-pair`, plus an exact-name check before checkout. Do not put the
label on shared ARC runners. It was added through the GitHub API without removing
existing labels or restarting the runner. Re-registration must preserve this label.

Access gate discovered on 2026-09-05: this repository is public, but the runner's
Default organization group has `allows_public_repositories=false`. The first manual
run stayed queued and was cancelled before any host steps executed. The dedicated
label does not override repository access policy. Before hardware execution, use an
approved private-repository workflow or obtain explicit approval for narrowly scoped
runner access; do not enable public repositories across the shared Default group.

Thatch.Server owns runner provisioning and isolation. Its runner template enables
`PrivateDevices`, `ProtectHome` and an isolated HOME. This workflow does not change
those settings or assume operator weights/caches reside in the runner's HOME.

The script inventories Docker containers and their mount paths without printing
environment variables. It selects an already-local Qwen/TT serving image by its
immutable image ID; it does not pull an image. A disposable, network-disabled,
read-only, capability-dropped container with a numeric unprivileged UID reads
daemon-host `/dev/tenstorrent` and `/sys` metadata through read-only mounts.
The image entrypoint is overridden. No accelerator is opened or reset.

Only the probe container created by this invocation is removed on exit. Production
containers are never started, stopped or modified. A missing local image, missing
host devices, inaccessible Docker daemon or failed probe causes a failing job.
Inventory is not a test of device idleness, P150A harvesting, fabric health or model
performance. Container device mappings are incomplete evidence of device users;
an exclusive-use check is still required before accelerator execution.

GitHub concurrency serializes this workflow only, not jobs in other repositories
or host services. The dedicated runner label establishes placement, not exclusive
ownership of the cards. Keep serving disabled until the operator restores it.

Host-only script tests: `python3 scripts/ci/test_qwen_inventory.py`.

## Hardware correctness

The manual `hardware` suite opens both allocated cards using a pinned local image.
It audits the runtime, checks fused operators on each chip, validates chunked
prefill recurrent/KV state, and checks eager/traced all-gather over the QSFP-DD
fabric. The asymmetric PCIe x16/x4 attachment is not the inter-card transport.
This is not a full-model decode throughput benchmark.

By default, a read-only host-process descriptor scan must prove no device owners.
If the operator has explicitly allocated both cards, set `cards_allocated=true`.
That mode verifies the physical board mapping but deliberately does not inspect
host processes or claim an OS-enforced reservation. It requires no host PID
namespace or ptrace capability. Do not infer idleness from TT-SMI activity or an
unreadable process scan. Neither mode resets cards or changes serving services.

Hardware entry-point guard tests require Python 3.10 or newer:
`python3 scripts/ci/test_hardware_guards.py`.

## Programme suites

- `full-prefix`: full 64-layer native sequential rollback oracle at prompt lengths
  63/64/65, every retained prefix 0..16, eager and trace. Checks all target logits,
  active states in all 48 GDN layers and logically valid KV values in all 16
  attention layers, followed by two correction-input steps. Keeps rejected KV
  bytes physically present and relies on explicit target positions/masks to hide
  them; wrong-page and stale-GDN negative controls must be detected. This is not
  batched full-model verification, cancellation certification or throughput.

- `gdn-direct`: active-slot snapshots through a 48-worker copy-only DMA kernel,
  retaining the full prefix, continuation and changed-idle-slot isolation gates.
  Pinned BF16 tile geometry only; not a serving patch or a full-model verifier.

- `gdn-active`: stores slot-zero recurrence/conv snapshots instead of all eight
  slots, restores in place with native state writers, and repeats every-prefix
  correctness. Adds changed-idle-slot canaries and paired three-block ABBA
  save/restore microbenchmarks. Tile padding is included in reported snapshot bytes.

- `gdn-block`: extends `gdn-prefix` by batching output projection and TP reduction;
  the fused recurrent/conv/norm/gate body is taken from the SHA-pinned native source.
  Uses the same every-prefix and stale-state gates, without changing installed code.

- `gdn-prefix`: E3a real-weight GDN prerequisite on both chips, B1 in Bmax8.
  Compare native sequential decode with batched input projection plus the same
  native fused per-token remainder. Test T=1/2/4/8/16, three seeds, eager/trace,
  every prefix snapshot and two correction steps after restore. Stale recurrence
  and stale convolution negative controls must fail exact state/output checks.
  Uses unchanged K-image BF16 state/decode flags. Snapshot-heavy diagnostic layer
  timing is not the full-model verifier cost curve or speculative throughput.

- `baseline`: frozen model/tokenizer snapshot, unchanged K-image control, warmed
  context/concurrency matrix and passive 250 ms metrics capture. Client estimates
  are not engine-commit throughput. Weights are read-only; experimental caches are
  isolated in a labeled Docker volume retained between runs.
- `interleave`: opt-in model/plugin patches staged only in disposable containers;
  whole-prefill versus 1/2/4 decode-credit arms, boundary outputs and cancellation
  checks. Stop on missing or divergent evidence. This initial gate does not yet
  certify direct state snapshots, exact output token IDs or mixed-traffic fairness.
- `profile`: separate eager GDN/MLP/paged-attention layer profiles and their reference
  tests. Reject dropped markers, missing device timings, skipped tests or a missing
  chip. Never score these instrumented runs as full-model decode throughput.

- `gdn-inplace`: opt-in compact B1 working-state gate over the real-weight GDN
  prefix matrix. Verifies native in-place request and buffer aliasing on both chips,
  all rollback prefixes and corrected continuations, idle slots and trace-stable
  native B8 state. No full-model integration or serving promotion on this gate.
- `gdn-multitoken`: synthetic recurrence-only device token loop with BF16 L1
  feedback, derived from pinned kernels. Checks all prefix outputs/states and
  restored continuations against native serial B1. Adds three paired ABBA timing
  blocks per seed/width, with all-prefix exports and exact post-sample checks on
  both arms. Excludes conv/norm-gate, input packing and full-model throughput.
- `gdn-inplace-timing`: repeats that gate and measures three ABBA blocks per
  seed/width against captured projection-batched native state handling. Restores
  initial state outside every timed replay and checks every staged prefix afterward.
- `gdn-checkpoint-cost`: retains the full correctness gate and compares matched
  all-prefix, no-checkpoint and end-only policies. End staging is poisoned outside
  each timed replay. Includes working-state transfers; diagnostic, not serving-ready.
- `gdn-checkpoint-dma`: same matrix with one direct compact-state DMA per checkpoint
  instead of five generic copies. Uses the unchanged active-copy C++ kernel with
  explicit compact-to-compact geometry checks; no serving promotion.
- `full-compact-gdn`: coding-context full-model gate for the compact-state/DMA
  candidate at T2/4/8/16; T1 stays native. Requires exact logits/state/KV/rollback
  before paired timing against both native serial and the existing batched control.
- `full-gdn-input-reuse`: same full-model gate, reusing one shape-only GDN input row
  instead of T redundant slices. Paired control uses compact state/DMA too; projected
  row cloning remains unchanged, and T1 is not optimized by this experiment.
- `full-gdn-row-layout`: hoists projected input conversion to row-major once per
  block, then extracts/tiles independently owned B1 rows. Uses the selective-clone
  candidate as paired control; preserves T1 and requires full exactness before timing.
- `attention-batch`: real-weight attention layer gate, T=1/2/4/8/16 at positions
  31/63/65, two seeds, eager/trace, both chips. Batch native fused QKV preparation,
  SDPA and output projection, but serialize paged K/V writes using B1 shards.
  Compare every output and all physical KV values, including unused-page canaries,
  exactly against native sequential B1. Positions/page views are preallocated static
  fixtures, not a serving integration. Negative controls omit writes or map the
  same token sequence to different pages. No speed claim or promotion on divergence.
- `attention-timing`: repeat the attention gate, then compare captured sequential
  B1 and batched layer calls in three ABBA blocks per fixture, 30 replays per sample.
  Validate outputs and all KV values before and after timing. Cache reset and host
  validation are outside the timing interval; serial KV write overhead remains
  inside the candidate. This short-context layer cost is not full-model tok/s.
- `full-batch`: extend the passing `full-prefix` oracle with the integrated native
  64-layer batched target. GDN projections batch across tokens, while fused GDN
  recurrence remains sequential; attention uses ordered shared-page writes.
  Check T=1/2/4/8/16 full logits/state/KV in eager/trace, then every T=16 prefix
  checkpoint and two correction steps at prompt lengths 63/64/65. Snapshots are
  captured per GDN layer at the selected prefix, not by a global mid-layer save.
  This is a static-fixture correctness gate; it does not enable a serving verifier.
- `full-coding-cost`: exactness first at 4095/16383-token repeated-code contexts,
  T=1/2/4/8/16 eager/trace, and T16 rollback prefixes 0/1/8/16. Hash the entire
  valid KV prefix in bounded 64-page chunks, not just the latest pages. After all
  cases pass, time captured native serial and batched full-logit blocks using
  three ABBA blocks and ten state-reset replays per sample. Reset/readback is
  outside timing; the batched arm includes one preselected end checkpoint.
  Measure active GDN restore separately. This omits draft generation, runtime
  acceptance selection, all-prefix staging and a complete commit pipeline; it
  is not end-to-end speculative throughput or a coding-quality benchmark.
  Following the T4/4K divergence in run 34002210390, this suite selects B1 SDPA
  reads while retaining batched projections, to test native reduction parity.
- `full-batch-attribution`: bounded JSON-only in-situ stage profiling at T1/T16
  (revision 2 also separates GDN projected-row copying, fused conv/gates,
  recurrence/norm/gate, active-state slicing and active-prefix writeback).
  and 4095/16383-token contexts. Three synchronized eager passes separate GDN
  native-row work, projections/checkpoints, attention KV/SDPA, native decoder
  components and LM head. Nested exclusive times reconcile without double counting.
  Full logits/state/KV must remain exact against native serial B1, including an
  unfenced captured trace control. Fences change overlap and include dispatch costs;
  these intervals are not a traced device critical-path decomposition. Export no
  raw profiler CSVs or tensors. Profiling is off by default in all other suites.

Run card-backed suites serially under an explicit operator allocation. See the
[execution ledger](../../docs/experiment-execution.md) for dependencies and results.
The planned research tracks are not all implemented tests; a passed prerequisite
must not be reported as completion of the whole programme.
