# Split history preparation before changing it

The two completed requests in run **34926705012** contain 40 committed blocks.
Their existing publication timers report these means:

| Publication stage | Host wall ms/block |
| --- | ---: |
| Feature selection | 0.057 |
| Prepare history | 44.109 |
| Publish target state | 1.978 |
| Commit history frontier | 0.009 |

These are nested inside selection/commit, not additional cycle costs. Process
CPU time is not device-kernel time. The current source places captured feature
projection and full-bank history assembly inside `prepare_history`; the existing
record does not separate their costs. Do not attribute all 44 ms to copies.

The opt-in `dspark-64k-publication-profile-request.py` wrapper samples up to 32
publications from one complete control request. It separates
projection, bank assembly and remaining host work, without extra fences, altered
cache contents or reordered calls. It retains the existing lazy-loader and
combined-runtime correctness admissions. Profiling TG is labeled instrumented,
not promoted as a clean speed measurement.

Local tests cover bounded sampling, unchanged results, callback restoration
and error handling. After the exact output/state request returns, an explicit
diagnostic exception unwinds the runtime and closes devices before a second arm
or final parameter audit. A completed diagnostic requires clean shutdown but
keeps `passed=false`, `full_request_passed=false` and all accepted TG fields null.
It is not a substitute for those audits on a performance candidate. The process
budget is 300 seconds, with a 420-second outer limit. Hardware attribution is
reported below. A 44 ms saving alone cannot establish 200 committed TG.

## Hardware diagnostic 35020108364

Completed in **4m12s**. One 135-token request passes exact output, final active
state and inactive-slot checks; device and checkpoint closure pass. This is
diagnostic completion, not full performance acceptance or a final weight audit.

| Sampled stage, 20 blocks | Mean ms | Maximum ms |
| --- | ---: | ---: |
| Captured feature projection | 1.653 | 2.324 |
| Full history-bank assembly | 46.803 | 63.831 |

The multi-second stalls from the previous run do not recur, so their cause is
still unresolved. The persistent publication cost is now attributed to bank
assembly, not projection. Report SHA256:
`d134bd0b278c23873f19a7fa5d628d0adcf772e03b5d7905e4d467882e9084f1`.

Next candidate: replace full-history slice/concat/pad/copy with a bit-preserving
dirty-tile writer. Keep the active bank untouched until commit. Repair the spare
bank's previous accepted/discarded tail, append only accepted rows, and zero
invalid rows within the touched tiles. A bounded planner covers at most 96 rows
per append instead of rebuilding 66,560 rows. CPU tests compare entire banks
through 480 randomized commit/discard transactions and tile boundaries, including
64K positions. An arithmetic-free BF16 DMA kernel now implements this plan,
but it is **not hardware-qualified or a measured speedup**. Next gates:
simulator bank contents/address lifetime and changed-input trace replay checks,
then combined-runtime hardware correctness and matched PP/CTX/TG.
Serving defaults remain unchanged.

Simulator run 35021706331 isolated five negative-zero to positive-zero changes
on chip 1 during input upload, before the writer executes. The revised probe
permits only zero-sign changes at that boundary and checks the writer bitwise
against the actual uploaded input. Nonzero upload changes still fail. This does
not qualify signed-zero preservation through upload. Run 35022225254 attempt 1
timed out in the 30-second Docker result-write preflight, before any simulator
execution; it supplies no kernel evidence.

Attempt 2 completed in 26 seconds: all 48 checks pass on both simulated chips,
with clean shutdown. It covers six append/commit/discard actions, whole-bank
contents, unchanged active banks and input tensors, poisoned padding and stable
bank addresses. Report SHA256:
`0a038068c36de2561b9bedeaa4bc1745c043561a48060ce102668a1cba209044`.
This is a small synthetic fixture, not 64K addressing or hardware acceptance.

The next probe adds changed-input captured replay with 84 checks. Run
35022680629 attempt 1 failed in Docker preflight before executing the probe;
attempt 2 also timed out in preflight. Run 35023307310 separates Docker create
and start diagnostics and completes in 30 seconds. All 84 checks pass, including
changed-input trace replay on both chips and clean shutdown. Report SHA256:
`25db87cc6706729736699ed4a30c40c10be0f1cdfef9a0558c9dc356be47cb43`.

`incremental_history_scope.py` provides a reversible opt-in combined-runtime
binding: captured 32-row projection outputs feed the writer without slicing,
all ten bank updates validate before execution, and commit/discard retain the
existing bank-swap interface. A partial write failure prohibits further reuse.
Local transaction tests pass; this binding is not hardware/model-qualified.

The next hardware probe uses capacity 66,560 and starts at row 65,535, crossing
the 64K boundary. It pins the simulator report and checks the planner/writer
source hashes before opening devices. It loads no model weights, has a
180-second probe limit, and makes no TG claim. Combined-model correctness and
matched PP/CTX/TG remain required after this addressing/replay gate.

Hardware run **35024279412** completes in **42 seconds**, with all 84 checks
passing, clean shutdown and exact changed-input trace replay across the 64K
boundary. Report SHA256:
`7a962f5b7cc519d61cead689a225c68995ce0a600951b65c4fe0c8f2a24301cd`.
The first dispatch failed before device access because its artifact download
lacked a token; the corrected workflow uses read-only Actions access.

The next comparison keeps one audited lazy model load, the same 65,536-token
prompt and 256-token output budget, and changes only publication in the second
request. It requires identical emitted tokens, exact final active/inactive state,
all candidate publications using the new writer, final audits and clean closure.
It reports separate PP/CTX/committed TG for control and candidate, never pooled
throughput. One pair is not sustained performance or held-out coding acceptance.

Run **35025092454** stops after its exact 135-token control: PP **2418.94**,
CTX **65536**, committed TG **11.6558**. It closes devices and checkpoint cleanly,
but does not run the candidate or finish the full comparison. Mean draft and
verification times remain **88.10/82.42 ms**; selection/commit averages **406.72 ms**.
The first `prepare_history` takes **5374.47 ms** (162.62 ms process CPU, no GC
pause). This is a recurrence of the publication stall, not a candidate failure.

The next run may continue diagnostically past a slow control only when matched
per-block history records account for over half of generation time, mean draft
stays at or below 120 ms and verification at or below 100 ms. Otherwise it still
stops. A degraded baseline labels the comparison diagnostic-only; it must not
support a claimed speedup. Exact output/state gates remain unchanged. The model
process budget is 480 seconds, outer budget 590 seconds: the observed first
control alone took about 204 seconds after 66 seconds of shared setup, including
native reference generation and request setup. No repeated full-model simulator
run or unbounded retry is introduced.

## Complete combined pair: 35026222541

Completed in **7m35s**, with two identical 135-token EOS outputs, exact active
and inactive state, 60 before/60 after exact learned-weight checks, unchanged
recorded sources, and clean device/checkpoint closure. The control is not
degraded. Report SHA256:
`5e29489e85e1780650a7837673c636ad4967fe4ef53ed69cbece9e45f789d455`.

| One stream, CTX 65536 | Control | Incremental history |
| --- | ---: | ---: |
| PP tok/s | 2603.19 | 2205.49 |
| Committed TG tok/s | 30.55 | 37.95 |
| Complete decode seconds | 4.420 | 3.557 |
| Prefill + setup + decode seconds | 54.02 | 56.92 |
| Draft ms/block | 83.61 | 87.34 |
| Verify/readback ms/block | 81.85 | 82.19 |
| Selection/commit ms/block | 54.16 | 6.84 |
| Nested history preparation ms/block | 51.91 | 4.48 |

Generation improves **24.25% in this ordered pair**, but slower prefill means
setup-inclusive latency does not improve. This is not a sustained/endpoint or
held-out coding-quality result. Repeat confirmation is required before the
matched context ladder; order effects are not ruled out.

The writer is exercised for every one of 20 committed blocks, touching at most
64 rows here. One additional discarded warmup takes 385 ms during request setup,
outside generation timing but retained in setup-inclusive costs. All hooks restore.
At 6.75 committed tokens/block, 200 TG needs a 33.75 ms cycle versus the measured
177.79 ms. Drafting and verification now dominate; publication is no longer the
primary steady-state bottleneck in this run.

## Repeat: 35027433446

The identical runtime repeats successfully in **7m26s**. Control is PP 2511.45 /
CTX 65536 / TG **30.6937**; candidate is PP 2597.34 / CTX 65536 / TG **38.3215**.
History preparation averages **49.40 versus 4.28 ms/block**. Both emit the same
135 committed tokens as the first run. Exact active/inactive state, all 120
before/after learned-weight checks, source stability and clean closure pass.
Report SHA256:
`54ad14089eed8cb9bb1c89d118d5d08872a3aa29917410b5c5808900ff463456`.

Candidate request time including prefill/setup is 51.82 seconds versus 58.06
control in this repeat; its direction differs from the first pair. Do not infer
a general prefill or setup win from these two ordered runs. Generation and
publication gains repeat, but order-independent/sustained and coding-quality
acceptance remain open. Proceed to the same-runtime context ladder.
