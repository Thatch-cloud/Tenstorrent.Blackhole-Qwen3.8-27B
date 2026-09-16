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
