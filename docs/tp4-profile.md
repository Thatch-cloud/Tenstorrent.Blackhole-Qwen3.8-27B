# The TP4 op-level device profile

One four-card job (`scripts/ci/references/tp4-profile-jobs/P1-trace-profile.env`) profiles the TIMED verify of the four-card fast path
(`c2-packed-tp4-speed`: the verify's T1/T2 audits off, the drafter's K/V slide on), and an offline analysis
(`scripts/ci/tp4_profile_report.py`) turns tracy's CPP report into the decomposition that sizes the levers between the measured
round (about 70.8 ms of verify trace with the audits inside it; about 62 ms projected with them off) and the goals (several users at
75 tok/s each, one fast lane at 150 tok/s).

## What runs

`C2_ACTIONS=status reset build gate`, `C2_GATE_PLAN=ops-twin,ops-trace` (`scripts/ci/ops_profile_plan.py`):

| arm | what | why |
|---|---|---|
| `ops-twin` | unprofiled: 4 real-text users at 4,096 / 8,192 / 16,384 / 24,576 tokens; user 0 asks 192 tokens, users 1-3 ask 64, all ignore_eos | the unperturbed `[PACKED-PHASE]` round times and the reference texts |
| `ops-trace` | the same, under tracy (v138's recipe): `-p --check-exit-code --disable-device-data-dump-to-files --disable-device-data-push-to-tracy --dump-device-data-mid-run --op-support-count 20000` | the CPP device report: every op of every replayed trace, on every chip |

Fixed in code, not in the job file, so a job cannot drift from what the analysis expects: the shapes, the tracy argument list (any
op-support other than 20000 is refused: 200000 segfaulted the dispatch thread three times), the profiler environment, and the
requirement that the profile be a four-card one with both verify audits off. 24k rather than 32k: the exactness divergence at 32k and
above is unresolved.

One container yields 4-live 64-row rounds, then 3- and 2-live padded rounds as users 1-3 finish, then user 0 alone on the 1/2/4-row
sequential engines. A lone user has no 16-row step on this profile (the padded block needs two live users), so the analysis composes
that lane from the 4-row step and the 64-row block.

**The read-back cadence.** The profiler's per-core DRAM buffer holds 20000 programs and nothing is read back until a drain. A 4-live
TP4 round is about 5,000 programs per core (verify about 3,600, two pair drafts about 1,100, eager commits about 300), so
`QWEN_FAST_PROFILE_DUMP_EVERY=2` (`packed_verifier.dump_device_profiler_every`, called after every packed round and every sequential
verify step) reads the profiler back after every second verify replay of any kind and keeps every window under about 11k. The
prefill (24k = 12 chunks x 64 layers x 7.4, about 5.7k programs) fits one window, so no prefill flush is used (it has never run at
20000). Anything before the first read-back costs at most round 1.

**Traps handled.** Never cancel the job once the gate step has started: the profiler writes as root, and a cancelled profile run once
left root-owned logs so the next checkout died with EACCES. The gate hands each arm's profile tree back when the arm ends, and the
workflow's `always()` step "Hand back the TP4 op-profile output" does it again however the gate step ended, keeps the CPP report
(compressed) if the gate did not get to it, and prunes what is left above 8 MB. A disk guard stops the profiled arm past 4 GB of
profile tree or 85% disk. A segfault in `read_core_data_from_completion_queue` is an infrastructure fault in cause, but the plan reports it as FAIL (the harness records it as fatal): read the log, and do not raise op-support.

**Stop rules.** STOP on an attach refusal, a hang, garbage text, or ops-trace texts differing from ops-twin (profiling must not
change arithmetic; the plan FAILs). Everything else is soft: a report that finds fewer than 8 complete 4-live verify sessions on all
chips says so in its notes and the job still passes.

## Running the analysis

The gate runs it after the profiled arm and writes `gate/ops/tp4-profile-report.{json,md}` beside the compressed CPP report
(`gate/ops/cpp_device_perf_report.csv.gz`). To run it again, or on a downloaded artifact (stdlib only, Python 3.7+):

    python3 scripts/ci/tp4_profile_report.py --results <the artifact's gate/ directory>

or on a bare CSV:

    python3 scripts/ci/tp4_profile_report.py cpp_device_perf_report.csv.gz --server-log ops-trace/server.log \
        --gate-json ops-trace/m3native-gate.json --twin-log ops-twin/server.log --twin-json ops-twin/m3native-gate.json --chips 4

It prints the markdown and writes both files (next to the CSV, or into `--out`); the exit code is 1 when the validity section has a
problem.

## Reading the report

* **Validity.** Four chips; the trace is complete on every chip (late sessions are truncated where the per-core buffers filled, and
  are dropped); no audit line in the server log; the launched configuration has QWEN_FAST_TP=4 and both audits at 0; at least 8
  complete 4-live verify sessions and 8 complete 4-row sessions (a note when fewer); at least 20 read-back lines; dropped-marker
  lines; the texts identical to the twin's.
* **Traces are identified by content**: a trace of 64 layers is a verify; one SDPA launch per user per attention layer means the
  packed block (four) or a lone user (one). The lone lane's widths (1, 2, 4 rows) are told apart by kernel sum.
* **Categories are by role, not core count**: a layer is GDN if it holds GdnConvGates; matmuls by position; the recurrence is the
  longest generic after the last conv-gates launch; the attention core is the SDPA op, or per user the longest generic between
  AttnPrep and the heads concat.
* The sections: groups against the research projection and TP2 (which terms did not scale), categories per chip, per-layer means,
  weight matmuls (GB/s, % of DRAM, ns per tile per core: the bf4 grid question), collectives per call (the minimum over chips is the
  intrinsic time, the skew is waiting on the slowest chip), SDPA fixed cost and slope per 1k tokens (four contexts per round), kernel
  time by live-user count, the projected lone 16-row verify, the device time between verify replays (a read-back inflates half of
  them: read the best), and the profiling overhead against the twin (flagged above 3%).
* The session id is the replay counter, so session k of the verify trace is `[PACKED-PHASE] round=k`; the report checks that against
  the host `trace_ms` and says when the assumption is doubtful.

## What the profile settles

Collective cost per call inside the trace (the largest unknown: J0b's eager 58-97 us against TP2's 18 us in-trace), the TP4 matmul
grids and whether the bf4 gate/up are tile-rate bound (about 95 ns per tile per core: 39 cores, about 231 GB/s), the GDN recurrence
time at 12 heads (it does not shrink: 10.3 ms projected), and the SDPA fixed cost at one KV head per chip. Every lever in the
research ranking is re-sized from its own measured category: L3a (ring fabric for the fast path's collectives, a profile flip),
L1 (split the recurrence's value columns over the idle cores), L4 (68-core grids for the bf4 gate/up; decide the fused gate/up before
the TP4 references are re-recorded), L2 (one conv-gates launch, pieces folded into the windows copy), the rest of L3, then L6-L8.
