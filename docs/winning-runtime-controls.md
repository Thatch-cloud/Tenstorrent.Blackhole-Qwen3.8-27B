# Preserve the winning combined runtime

Use the actual successful commits as controls, not a reconstruction of their
individual kernels on the latest branch. The target remains 200 committed TG
for one correct coding stream; these results do not meet it.

| CTX | Original run | Exact revision | PP | Committed TG |
| ---: | --- | --- | ---: | ---: |
| 4096 | 34702526963 | `4b3f90b001c8c91011666194e04f74b1157c70a6` | 3279.29 | 106.58 |
| 8192 | 34730226400 | `8c102b20df22329106955b4006bf4d650bb94e40` | 3304.32 | 101.59 |

Each original experiment includes a native control and shared-Q/K candidate,
one audited and two timed requests per arm. Each timed response commits 121
tokens before EOS, with a 256-token allowance. They are single-stream offline
coding fixtures, not sustained serving or held-out coding-quality acceptance.

## Frozen recipe

- Original `request-target-attention` entry, `merge_intervals` fixture.
- Score layout, T16 fusion and captured publication enabled.
- Down-only MLP, banked proposal and direct native-slot experiments disabled.
- Original four-link setup and pinned runtime image retained by the scripts.
- Original simulator admissions, learned-weight checks and exact request/state
  checks retained. No new kernel requires simulation for this unchanged replay.

Report hashes: 4K `73151825723aee8fa1fadb2e64754f797dff151adc0c7439c460b1a30129eacb`;
8K `c4ab877ca9db65a5491b04afa5456ceedff5f4451154f96d16d39ebd4a0b804c`.
The old shutdown logs contain a subdevice-manager warning; retain that caveat.

## Next comparisons

1. Replay both exact revisions through `qwen-winning-controls.yml`, sequentially
   on the same cards. Each job is bounded to ten minutes of experiment execution;
   original jobs took about six and seven minutes. Preserve provenance for both
   the old runtime commit and new workflow commit.
2. Compare original and replay outputs, acceptance, enabled paths, source/kernel
   hashes and complete PP/CTX/TG before claiming reproduction.
3. Use the 8K recipe as the starting point for longer contexts. Change only the
   capacity/numerical constraints demonstrated to require a change; reuse already
   qualified long-context components where necessary rather than rerunning their
   exploration. Keep the full combined recipe and compare complete requests.
4. Evaluate further optimizations against this control, not against whichever
   isolated kernel experiment most recently passed. Serving defaults stay unchanged.

Use `scripts/ci/winning_control_report.py ORIGINAL_JSON REPLAY_JSON` to reconcile
each replay. It pins the original report hash, compares source/kernel identity,
precision, all six requests, proposals, committed output and acceptance, and
recomputes TG from complete decode-loop durations. A timing regression remains a
regression even when identity and correctness reproduce. Unit mutation checks and
original-report self-comparisons test this offline checker, not the hardware replay.

The separate 64K history diagnostic is not the new baseline. Its second pass
(35044914165) passed but did not reproduce the hundreds-of-milliseconds spikes;
it does not establish a root cause or justify another chain of profiling jobs.

## Replay outcome: 35045581089

The unchanged 4K revision reached the ten-minute cap; 8K was cancelled by matrix
fail-fast before starting. Runtime build was a **three-second cache hit**, not a
cold compilation. The partial report retains two audited requests and one timed
control, but stops during the first timed candidate. It has no clean shutdown or
complete comparison and must not supply an accepted PP/TG row.

Late candidate block logs retain approximately 27–33 ms drafting and 66.8 ms
blocking verifier time, consistent with the original recipe. One selection/
publication call spikes to 452 ms. Thus the intermittent wait also occurs on
the unchanged winning runtime; it cannot be attributed solely to the later
sixteen-worker experiment. Keep the frozen recipe while investigating execution
conditions. Do not extend deadlines blindly or restart kernel exploration.

The replay's existing cgroup telemetry shows a substantial storage-pressure
change: the two audited requests accumulate **17.33 s and 6.93 s of full I/O
stall**, versus **0.0015 s and 0.0018 s** originally. The completed timed control
records 1.26 s versus 0.0008 s. These span setup and auditing as well as decode;
they do not prove which storage call caused an individual token-latency spike.
CPU quota throttling and OOM counters remain zero. A bounded read-only runner
storage observation checks backing mounts, capacity and device pressure before
changing the frozen model recipe. It does not load weights or touch the cards.

Read-only run **35046558059** confirms heavy storage activity with Qwen stopped:
the root filesystem, Docker backing store and checkpoints share RAID1 `md1`.
Sampled `nvme0n1` writes have 237–271 ms average latency and 94–97% utilization;
host full I/O pressure reaches 54% over the last ten seconds. The unprivileged
process sample cannot identify the system-wide writer. Follow-up only reads
RAID synchronization state and process I/O counters; it does not stop workloads
or change storage configuration. This is evidence of contention, not yet proof
of the writer or attribution of every publication spike.

Follow-up **35046860907** reports both RAID arrays non-degraded with synchronization
idle. A different UID 1001 `Runner.Worker` reports about 141,385 KiB/s writes;
`cargo` and `tar` also write heavily. The Qwen observer runs under UID 1000.
These counters identify competing build activity, not its repository/job owner;
no process was stopped. The user confirms another workload is highly likely.

Future winning-control launches sample host full I/O pressure for 15 seconds and
exit before weight loading if the interval exceeds 1%. This is a conservative
experiment-start policy, not a performance qualification or guarantee that the
host stays quiet. Existing per-request telemetry remains necessary. No full-model
rerun is launched while arranging a quiet window.

After the user reported reclaimed disk space, guarded replay **35047650211**
sampled 15 seconds and measured **12.48% full I/O stall** (1.872344 seconds),
above the 1% start limit. It correctly exited before loading weights; 8K was
cancelled without starting. This is an environment-admission rejection, not a
model correctness or throughput failure. A quiet execution window is still
needed; do not lower the threshold merely to obtain a green run.

## Offline recipe comparison

Comparison uses the original 8K report (34730226400) and completed eight-worker
64K report (35039344557), not the contended sixteen-worker timing.

| Winning element | Current 64K path | Required action |
| --- | --- | --- |
| Folded T16 target and commit-only GDN | Enabled in both complete requests | Retain |
| Fused T16 MLP | Reintegration wrapper; same `fused_t16_scope.py` hash | Retain, do not rebuild the kernel |
| Shared Q/K recurrence | Same scope and pipeline source hashes | Retain |
| Captured publication | Same publication adapter; incremental writer added | Retain full-history state/commit checks |
| Fused Markov score layout | Disabled in the latest clean 64K timing | Restore behind one explicit experimental flag |
| Draft attention | Long-history split-K replaces the short-context path | Reuse its completed admission, not its exploration |

The score-layout kernel source itself is unchanged. Its scope adds only an
overridable feedback method around the same candidate call. The old 64K score
comparison timed out and suffered variable publication stalls; it did not
establish a causal reason to permanently omit the winning fusion.

`QWEN_MATCHED_SCORE_LAYOUT=1` now selects `matched_score_request.py` around the
existing admitted eight-worker combined runtime. It reuses the learned-weight
score audit, requires actual fused execution in both complete requests, and
retains native output/state, source, loader and incremental-writer checks. It
cannot be combined with the sixteen-worker or history-profiling experiment.
Default paths and frozen 4K/8K controls are unchanged. Local scope and mutation
tests pass; the newly composed entry is **not yet hardware validated**, and no
performance gain is qualified. Hardware attempts are recorded below.

The prepared comparison also supports `QWEN_MATCHED_SCORE_PAIR=1`: one model load,
one complete native-score control and one complete fused-score candidate. It
reuses the existing paired-request scope and its degraded-control stop, and
retains two full requests with exact output/state checks. Per-arm timings are
reported separately; pooled PP/TG fields are cleared. One fixed-order pair is a
diagnostic, not repeatability or causal acceptance. The workflow has a separate
15-second pre-load I/O gate. No new kernel implementation or simulator replay is needed
for this source-identical score kernel; composed hardware correctness remains
mandatory before any performance promotion.

### Score comparison attempts (16 September)

Run **35048450847** passed pre-load admission and completed both 64K requests,
then failed the outer report validator. These are diagnostic observations only:

| Path | CTX | Committed tokens | Decode ms | TG (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| Native score | 65,536 | 135 | 3,643.03 | 37.06 |
| Fused score | 65,536 | 135 | 3,324.23 | 40.61 |

Both requests report exact output, active state and inactive state. The fused
scope reports two calls and restored hooks; device closure completed. The
serialized pair requests match the serialized main request records, but the
validator compared live Python objects against JSON-decoded objects. Commit
`73d9855` normalizes the comparison representation without dropping fields;
local tests cover tuple/list equivalence and rejection of changed values.
The failed artifact remains failed; it has not been retroactively qualified.
The observed 9.6% difference is one fixed-order pair, not a repeatable gain.

Retry **35049234170**, revision `2458f52`, stopped at the 15-second admission
check: **45.997% full I/O stall**, versus the 1% limit. No model load or hardware
measurement ran. The fix still needs an uncontended hardware rerun; do not
interpret host pressure as a measured decode slowdown or stop unrelated jobs.
The 200 committed tokens/s objective remains unachieved.

The completed pair's own cgroup counters show 3.323 seconds full I/O stall over
133.559 seconds for the control (2.49%), and 15.746 over 166.622 seconds for the
candidate (9.45%). These include setup and audits, not isolated decode. The
candidate's lower decode time despite higher whole-request pressure demonstrates
why host pressure alone cannot classify the decode result.

Tag `experiment/matched-score-pair-v3` therefore treats a valid high-pressure
sample as advisory for this diagnostic comparison only. Missing/invalid pressure
evidence, timeout, and all correctness/admission failures still stop the job.
The report remains diagnostic-only with unqualified per-arm timing; no serving
or performance promotion follows from a green diagnostic run. Other tags retain
the quiet-host cutoff. This is not a change to the kernel or benchmark timings.

### Passing combined score pair

Run **35049608823**, revision `402e2cc`, completed with exact output/state checks,
clean closure and successful outer validation. Report SHA-256:
`773d7fe34923669952e8dd07fa4c64b875967c77ea8665432fb0e3650e02d9f0`.

| One stream, CTX 65,536 | PP tokens/s | Committed TG | Draft ms/block | Verify/readback ms/block | Select/commit ms/block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native score | 2,554.47 | 38.39 | 84.07 | 81.81 | 8.06 |
| Fused score | 2,597.57 | 41.21 | 75.22 | 81.51 | 5.69 |

Each arm committed 135 tokens across 20 blocks: 6.75 committed tokens per block.
The fused arm averaged 163.75 ms per cycle. This is a correctness-passed diagnostic
pair, not a repeatability or held-out coding-quality qualification. Its 7.3% TG
improvement is observed, not adopted as a serving default.

At this acceptance rate, 200 TG requires **33.75 ms per complete cycle**, versus
163.75 ms observed. Eliminating select/commit alone cannot bridge this gap. Even
eliminating the entire draft stage leaves the measured verifier above that budget.
The next optimization needs verifier reduction and/or materially more committed
tokens per verification, while retaining exact recurrence/KV rollback and coding
quality. Repeating host-I/O admission tuning or isolated score tweaks is not a
credible route to the target. Use this composed fused path as the next diagnostic
control; do not restart its component kernel search.

### Same-recipe ladder correction

The requested ladder is 4,096 / 8,192 / 16,384 / 32,768 / 65,536 / 131,072 /
262,144 prompt tokens, with 256 additional generation rows. Selection alone is
not admission. The full historical runtime still needs its context routing,
history checks and target-attention admission parameterized together.

Simulator attempts 35051035331 and 35051933391 **cannot qualify the intended
larger-context recipe**, independently of their timeouts: the first adapter
updated operand shapes and the kernel assertion but missed the factory's
`Skt == 272` condition. Consequently it did not select the original FP32
statistics format at larger shapes. The split retry 35053293462 was cancelled
when this was discovered. Do not use any of these as numerical or speed evidence
for the winning recipe at 32K/64K.

The adapter now derives both selectors from the same context geometry. Local
tests evaluate the generated factory for all seven sizes and verify that only
the allowed context selector differs from the historical replacement. The
factory admits the explicit seven shapes in one compiled library, rather than
rebuilding that library for each environment value. Per-shape device kernel
compilation is still possible. This invalidates
the old build-cache key intentionally. No corrected numerical run is claimed.

### Cancelled-run audit: timeout scope and remaining work

Run 35053293462 is terminal (`cancelled`). No replacement run has been launched.
The proposed `experiment/frozen-context-build-v1` tag runs preparation only;
the entire numerical/diagnostic matrix is explicitly skipped for that tag.
The tag has not been published and no runner permission has been added for it.
The current branch has the following **planned**, not hardware-validated budgets:

| Boundary | Factory preparation job | Each numerical/diagnostic job |
| --- | ---: | ---: |
| GitHub job, including checkout and artifact upload | 13 min | 16 min |
| Execution step | 11 min | 14 min |
| Host launcher, including downloads and container setup | 600 s + 45 s kill grace | 780 s + 45 s kill grace |
| In-container preparation | 510 s, build allowed | 120 s, cache hit required |
| In-container probe | Not run | 510 s |

These are nested limits, not independent allowances. The historical launcher
allowed up to 180 seconds **each** for two asset downloads before container
preparation. The adapter now reuses SHA256-verified cached copies; misses have
20-second curl deadlines and a 60-second total asset-stage limit (five-second
kill grace). Asset hashes and staged copies are checked on every run. Local
tests cover reuse, corruption, wrong downloads and timeout failures; no runner
speedup is measured yet. These operations still consume outer budgets. Cleanup
now stops the owned simulator container before collecting logs/results, then
removes it. Stop/log/copy/removal deadlines are 5/5/15/5 seconds, each with a
one-second kill grace; their exit codes are retained in `container-cleanup.json`.
A cleanup error cannot turn a failed test into success, and makes an otherwise
successful launcher fail. No unrelated containers are addressed. This is covered
by local shell tests, not a live Docker timeout test. A 510-second probe allowance
still does not guarantee 510 seconds
are available to the probe. The current design does **not** yet meet the requested
single-digit-minute end-to-end cycle. Four serialized probe jobs plus preparation
can still make the whole workflow lengthy; there is no single workflow-wide timer
in this file.

The deployment audit also found the simulator adapter copied a newer hardware
runtime-cache module over the historical checkout. It now installs a separate
binary-cache helper instead. A local deployment regression test checks the
historical hardware module remains byte-for-byte unchanged. This is source-level
protection, not a performance result.

Before another ladder run:
- Verify the asset/cleanup bounds on the runner and measure remaining setup time.
- Complete runtime context routing and admission. Deployment now invokes the
  runtime plumbing helper alongside the simulator adapter, including target
  sequence allocation and stable-history geometry; this does not yet remove
  the historical 8K-only entry, build and target-attention admission checks.
- Replace the hard-coded 8K admission with matching per-context evidence, without
  bypassing source/binary checks or substituting the different 64K runtime.
- Retain one numerical recipe and report PP / CTX / committed TG independently.

The current simulator matrix covers only 32K and 64K, not the complete requested
ladder. Its split-report coverage checker is not yet a runtime admission gate.
The diagnostics shard now rejects missing, non-finite or numerically mismatched
results before returning success; a successful shard is labeled
`diagnostics_complete`, not full numerical qualification. Its validator and
dependencies are included in the probe's source hashes. Local mutation tests
cover these failure paths. No corrected simulator result is claimed.
The 200 committed-TG objective remains unachieved.

The target allocation adapter now scales **both** model sequence capacity and
physical KV pages. Previously it increased only `max_seq_len` while retaining
1,024 addressable 64-row pages, which cannot hold a 65,536-token prompt plus
generation. Below 64K the original 1,024 pages / 1,032 cache blocks are unchanged.
At 64K, 131K and 262K, addressable pages are respectively 1,028, 2,052 and 4,100,
with the same eight spare cache blocks. Each covers the prompt plus 256 rows;
model sequence capacity remains separately rounded up. Local geometry tests
verify coverage, not available device memory, model positional limits or hardware
correctness. No KV precision or attention algorithm changes are introduced.

There is also a positional-limit blocker at the largest requested prompt:
`DSparkRotary` pins 262,144 maximum positions and rejects positions at or above
that limit. A 262,144-token **prompt** plus 256 output rows needs at least 262,400
positions. The adapter now checks target and drafter configuration limits before
opening devices/loading model weights. It does not truncate the prompt, change
RoPE or relabel a smaller prompt as 262K. Testing a 262K total window versus
qualifying a positional extension for a 262K prompt is an explicit remaining
decision; allocation coverage alone does not resolve it.

### Winning 8K recipe: where the 200-TG gap actually is

Recomputed from retained run 34730226400, report SHA256
`c4ab877ca9db65a5491b04afa5456ceedff5f4451154f96d16d39ebd4a0b804c`.
Only the two uninstrumented publication requests are included: 22 blocks,
242 committed tokens. Audited requests are excluded.

| Mean per block | Observed |
| --- | ---: |
| Committed tokens | 11 |
| Draft | 29.78 ms |
| Verify including readback | 68.45 ms |
| Select and commit | 9.17 ms |
| Complete cycle | 108.23 ms |
| Required complete cycle at this acceptance for 200 TG | 55.00 ms |

The verify/readback timer includes the blocking trace; do not add those nested
measurements together. The recorded full-request result remains **101.59 TG**,
not a new measurement.

Two optimistic bounds clarify the next performance work:
- Removing every cost except verification, at the observed acceptance, gives
  only **160.69 TG**. Host-overhead removal alone cannot reach 200.
- Accepting all 16 rows at the observed complete-cycle cost gives **147.84 TG**.
  Better draft acceptance alone cannot reach 200 at that cost either.

These are arithmetic bounds with fixed measured costs, not predicted benchmark
results. The same-recipe context ladder remains necessary, but extending its
context flag is not itself a route to 200 TG. The optimization must reduce target
verification work alongside draft/commit costs, or increase useful verified rows
without proportional cost. The existing combined trace attribution identifies
recurrence, fused MLP and down/output projections as substantial kernel groups;
it does not support treating spare cores or host dispatch as the dominant cause.

### Corrected 32K native-recipe failure

Run **35056799088** fails numerically, not by timeout. Cached preparation takes
2.46 seconds; the probe exits in 197.23 seconds and closes cleanly. The precise
statistics factory is enabled. No hardware admission follows this result.

The first eager comparison fails six elements on chip 0, head 1, rows 3 and 14,
channels 20, 31 and 54. All six actual values are **-46.25**; references range
from -45.7754 to -45.7809. Errors are 0.4691–0.4746 against allowed errors of
about 0.4678 (`atol=rtol=0.01`). They are finite, and narrowly outside the
unchanged tolerance. Other chips/replay are not qualified by this stopped test.

The earlier [reciprocal investigation](context-ladder-investigation.md#reciprocal-reload-boundary-identified)
identified a reload-rounding contribution to a similar -46.25 result. Its
scalar-reciprocal candidate passed a different 32K component configuration:
512-key chunks and 1,024 output-headroom rows. That is a useful hypothesis,
**not proof of the cause or a pass for this 256-key/256-headroom recipe**.

The adapter now offers explicit `--scalar-reciprocal` for a separately labelled
simulator candidate using that existing implementation. It leaves the default
native recipe, factory formats, chunk size and tolerances unchanged. Candidate
reports identify `scalar-fp32`; the shard join rejects mixed variants. Local
source tests pass, but this candidate has not run in the corrected geometry.
Do not simply rerun the full matrix under the same timeout: a successful probe
executes more attention calls than this early failure and needs a measured
execution budget or correctly partitioned checks.

### Reciprocal candidate: 32K eager screen passes

Run **35057655308**, revision `726033c`, passes both eager fixtures on both chips
at the unchanged `rtol=atol=0.01`. All four comparisons have zero failing elements;
24 input-immutability, eight physical-layout and eight fixture-control checks
are retained. The probe takes **410.99 seconds**, cached preparation **3.02
seconds**, and device/container cleanup succeeds. The native-library cache key
and binary match the preceding failed native-reciprocal run.

All 48 reported script/support source hashes match the historical sources plus
the recorded adapter changes; all 18 deployment hashes were reconstructed.
Report SHA256:
`0534367dfb1fb3556fd6c9ac65d807cafcfbe2d62373647a20a397f28b6181a7`.

This confirms the candidate resolves the observed eager-fixture failures, not
that reciprocal rounding is the only possible long-context error. The artifact
correctly records `eager_complete=true`, `passed=false` and
`complete_probe_coverage=false`: replay and stale-input checks were not run.
Diagnostics, replay, full-model correctness and PP/CTX/TG remain unqualified.
The candidate has not been enabled in hardware or serving.
