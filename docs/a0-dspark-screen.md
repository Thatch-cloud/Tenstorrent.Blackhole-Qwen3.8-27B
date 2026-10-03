# A0: the DSpark v2 against DFlash2 acceptance screen

Status: harness built and tested on CPU (fakes, a tiny random Qwen3.5 on the pinned transformers, and the upstream sources where a person
holds them); nothing has been run on a GPU, and nothing in this change touches the GPU
host, any service on it, the rig or production. The run needs the owner (decisions D-A0-1 to D-A0-4 below).

## What it answers

Is DSpark v2's acceptance (tokens committed per verify round, tau) on our own agent coding turns higher than DFlash2's, enough
to be worth a Tenstorrent port? It is a PAIRED, TEACHER-FORCED screen: S2 is lossless greedy, so the text a turn produced does not
depend on the drafter, and a drafter's proposals depend only on the target's features, the anchor and its own weights. Both
drafters therefore walk the SAME logged text with the SAME target features (one BF16 target forward per prefix group serves every
arm) and can be compared round for round. It measures the model-level ratio R = tau_S / tau_D only; whether R survives the Tenstorrent
round time, DRAM and pool admission is Phase A's job (break-even is about 1.02-1.06 at 4 seats and 1.15-1.3 at 9 seats of 262k).

## Inputs

The tau lab's thinking-ON turns that finished ok: A1 (SWE held-out, 240 turns in 60 clusters) and A2 (our own sessions, 96 turns in 35
clusters). `scripts/ci/a0_bundle.py` builds the bundle ON THE RIG HOST (the prompt ids live only there): per turn the prompt and
answer token ids, the inverse-probability weight, finish and think / tool offsets, the logged round schedule, and an OPAQUE index
(a seeded shuffle; the map back to lab ids stays on the rig). Token ids are equivalent to text, so the bundle is lab-store data:
mode 0700 / 0600, sha256 in `MANIFEST.json`, never in a repository, never in a log.

Decision D-A0-1: may A2's ids leave the rig for the GPU host (both machines are ours)? If not, the fallback is an A1-only screen
(`--arms`/`--expect swe=240`): lower power and no answer about our own traffic.

## Arms (registered priority order)

| Arm | Turns | Purpose |
|---|---:|---|
| `dflash2-t16` | all | control: 15 proposals from 16 rows |
| `dspark-t16` | all | primary: 15 rows, Markov greedy chain |
| `dspark-t16-w2048` / `-w8192` / `-w32768` | 120 (80 SWE + 40 own) | model effect at an equal window; decides Phase B's DRAM choice |
| `dflash2-forced` | 100 | follows the logged schedule: V2 calibration |
| `dspark-t8`, `dflash2-t8` | 60 | depth diagnostic, never converted to tok/s |

Subsets are a seeded stratified draw over set x bucket. The run order is a seeded, set-interleaved order of prefix groups, so a
stop at the deadline still leaves a stratified sample.

## The round rule (`tf_pair_walk.py`)

Anchor at answer index j (j = 0 is the prefill seed), absolute start s = prompt length + j, rows = proposals + 1. The drafter sees
target features of rows below s and the token AT s (the walk hands it a view that ends there: a look-ahead raises). accepted a =
the matching prefix against the logged answer; committed = min(a + 1, cap) where cap = min(accept_limit(s, rows), tokens left),
accept_limit being the served extent rule; uncapped = min(a + 1, tokens left). The seed and the terminal round are not counted (the
lab's rule). DFlash2 drops the anchor row and takes the selector's greedy path; DSpark keeps every row and chains the Markov bias
(`predecessor[prev] @ successor.T`, w1 / w2 of the checkpoint) from the anchor. The checkpoint's own `spec_generate` is NOT used: it
skips the Markov head.

## Statistics (registered before the window)

* tau per arm: within a set, sum(weight x committed) / sum(weight x rounds); the two sets pooled with equal weight.
* R = tau_S / tau_D at T16 under the SERVED capped rule; the uncapped R is reported beside it.
* 95% interval: a PAIRED cluster bootstrap (10,000 resamples; conversations drawn with replacement within each set, BOTH arms
  recomputed on the same draw; an unpaired draw is a tested mutation that widens the interval several-fold).
* Guardrails (non-inferiority): G-long (R on prompts over 65,536 tokens): point >= 1.00 and lower bound >= 0.90; G-p10 (ratio of the
  IPW per-turn tau p10, turns with >= 3 counted rounds in BOTH arms): lower bound >= 0.95; G-w8 (ratio of the worst-of-8 p10, 2,000
  resamples x 5,000 draws, paired on the same drawn turns): lower bound >= 0.95; R_own (A2 alone) below 1.00 forces HOLD.
* Secondary: mean per-turn log ratio and the share of turns DSpark wins; R by bucket, set, region and answer-offset band
  (0-511, 512-2,047); P(accept j) for j = 1..15; R over positions 1-7; the 0.25 / 0.75 production-leaning R; window retention
  ((R_w - 1) / (R - 1) on the same subset); DSpark's equivalent share of drafter misses removed against the 46% the 5.95 worst-seat bar needs.

## Go / kill rule

* NOT_ESTABLISHED (fix and rerun, never a kill): any validity gate V0, V1, V3, V4, V5, V6 failed or did not run, or the coverage gate
  below failed. A run whose V0 or V4 is not PASS stops before any turn (the window is not spent on a verdict that cannot be reached).
* KILL: R's upper bound below 1.05. The censored-kill exception: if R in the 512-2,047 band exceeds R in the 0-511 band by 0.05 or
  more, the verdict is HOLD with `run_arm_e` (a greedy continuation past 2,048 tokens) before closing. Arm E is NOT implemented: the
  answers are capped at 2,048 tokens, which pulls R toward 1 (it protects a GO and threatens a KILL), so a would-be KILL on that
  pattern is a HOLD until arm E exists. A GO is not affected.
* GO (proceed to the Tenstorrent Phase A): R's point >= 1.10 and G-long, G-p10, G-w8 pass and R_own >= 1.00.
* HOLD: everything else; no Phase A spend.
* Steer: if w8192 keeps at least 90% of (R - 1), Phase B uses a bounded window.

Power (`a0_power.py`, synthetic clusters of the screen's shape, parameters in the script; rerun on the real metadata before the
window): true R 1.00 -> KILL 99%, GO 0%; R 1.05 -> KILL 4%; R 1.10 -> GO about 60%; R 1.15 and above -> GO about 100%; the interval
half-width about 0.02. A true R of 1.10 therefore sits on the bar: a point estimate just under it is a HOLD, not evidence of no effect.

## Validity gates

V0 (every check must be true; the booleans are written to `calibration.json`): the source and checkpoint sha256 pins; the bundle
manifest; the pinned transformers and flash-linear-attention versions (`--transformers-version`, `--fla-version`); memory caps set;
the fast GDN kernel actually BOUND by the model's module (the closure of the decorated function is inspected: `find_spec` succeeding
is not enough, and the torch fallback would take days); for BOTH drafters the scalars the port and walkers do not apply are neutral
(input embedding scale 1, output multiplier 1, no logit softcap) and the tapped layers equal the hooked layers (the control's as the
loaded model resolved them); the mask token; and the branch self-check on the real cache. V1 HF argmax equals the logged token on
>= 95% of answer rows overall and per set (the first divergence offset is recorded). V2 forced DFlash2 against the logged emitted
counts: report-only. V3 free-walk DFlash2 tau over the lab's served tau on the same turns (paired): the pooled ratio inside
[0.93, 1.12] AND its CI lower bound >= 0.98, so a control that walks a few percent below the served one cannot inflate R; the report
also gives R times that ratio as a sensitivity. The control arm is upstream's own code: `DFlash2DraftModel.propose` (its output
multiplier, softcap and selector) and its input embedding scale. V4 DSpark on the GPU against the repository's CPU reference
(`dspark_backbone_reference.py`) on 7 rows, context <= 1,024, 8 turns x 5 rounds (the shortest turns of each set in turn): >= 99% token
agreement where the reference top-2 margin >= 0.25, accepted length equal in >= 97% of rounds; fewer than 20 rounds is NOT_RUN. V4 and
the branch self-check run at the start of the canary AND of the main run. V5 the public summary passes `assert_public`. V6 process
peak below the cap and no watchdog trip.

## Coverage gate

A verdict needs the longest turns. A turn the run ATTEMPTED and did not finish in both core arms (deferred because its projected peak
would break the memory floor, or an arm failed on it) is LOST; turns the run never reached (the deadline; the order is stratified, so
those are an unbiased thinning) are not. The report counts them from `deferred.jsonl` and the failed rows, publishes coverage by length
bucket, and returns NOT_ESTABLISHED when paired / (paired + lost) < 98% or any bucket lost more than one turn. The admission
projection counts the target KV, the raw taps and DSpark's context K/V cache (doubled while it grows), is evaluated after the
allocator's cache is released, and any allocation failure from any layer (Triton, cuBLAS, the driver, not only torch's) stops the run.

## Files

`a0_bundle.py` (rig), `a0_bundle_io.py`, `a0_run.py` (driver, V4 collection, the self-check gate, the page-cache dropper),
`a0_target.py` (taps, chunked prefill, prefix branching, V1, `HFTarget` for the transformers 5.x cache layers), `a0_drafters.py` (the
two walkers, the upstream proposer, the Markov chain, V4 rows), `dflash2_torch.py` (a plain-torch DFlash2 / DSpark-shaped backbone),
`a0_upstream.py` (pinned intake, the neutral-scalar check), `a0_watchdog.py` (abort rule, heartbeat, verified kill, `flags`,
`handback`), `a0_power.py`, `tf_pair_walk.py`, `tf_pair_report.py`, `docker/a0-spark/Dockerfile`.

Verified on CPU: `HFTarget` against the real transformers 5.17 `Qwen3_5` classes on a tiny random model (hooks equal
`output_hidden_states`, chunked equals one-shot, snapshot / restore equals a fresh prefill, a prefix group equals independent traces:
`test_a0_hf_target.py`, the CI install pins that version); the port against z-lab's `model.py` and the loss against the recipe's own
functions, when a person holds the upstream sources (`test_upstream_parity.py`; skipped in CI). NOT verified without the GPU host:
`UpstreamBackbone` against the real checkpoint, `real_environment`, DSpark's backbone port against the real weights (V4 is the check),
and every speed and memory number below.

## Memory budget on the GPU host (estimates; W0 measures)

Unified memory, about 119 GiB visible. Every other large memory user on the host must be fully stopped for the window (the
operator's private plan says how). Target BF16 text-only 54-56 GB; the two drafters 7.6 GB; target KV at the longest turn (122,774
tokens x 65,536 B) 8.0 GB; GDN state plus one snapshot 0.4 GB; raw taps (51,200 B/token) 6.3 GB; prefill transients 2-4 GB;
CUDA context 3-6 GB; Python and bundle 2-3 GB. Process peak 82-92 GB; with the OS and agents (about 7 GB) 89-99 GB, leaving 29-39 GB.
Caps: torch per-process 96 GB; container `--memory 100g` (whether CUDA allocations on this machine are charged to the cgroup is
unknown: W0 measures it). While each model loads, a thread drops the page cache of its files every 0.5 s (`CacheDropper`:
`posix_fadvise(DONTNEED)` reaches the shards the loader has released), so the checkpoint never stacks on its device copy, and it logs
MemFree / MemAvailable to `load-memory.jsonl`; `--load-only` is the rehearsal that does exactly this and runs no turn. If the
canary's peak exceeds 92 GB: 2,048-token chunks, or post-fc features per drafter instead of raw taps (-3.8 GB). Disk: about 85-95 GB.

## Run time (estimates)

Prefill (shared prefix groups) 1.2-2.2 h, not shared 2.6-4.7 h; core walks 0.8-1.1 h; optional arms about 0.75 h; V1 / V4 about
10 min; setup, load, canary, teardown and handing the host back about 1.2 h. Window with prefix sharing about 4-5.5 h
(5.5-8.5 h without). A stratified half (168 turns) about 2.5-3.5 h. The host is not available to anything else for the whole window.

## Operation

Scheduling, how the host is cleared and handed back, who can recover it, and every host and image name belong to the operator's
private operations plan, not to this repository. What this repository specifies:

* a load rehearsal (`a0_run.py --load-only`: load, V0, V4, the branch self-check, the memory log, no turn), then a bring-up window (W0,
  the canary: three short turns plus the longest) before the long run (W1, resumable: a second invocation continues where the first
  stopped);
* the abort rule, FAIL CLOSED: `a0_watchdog.py` starts BEFORE the screen's container and polls every 0.5 s reading only
  `/proc/meminfo` and `/proc/vmstat` (the kernel log and `docker events` are read by threads; nothing forks in the poll). It kills that
  container (and only that one) on low available memory (two consecutive polls), low free memory while CUDA loads, any swap-in, a new
  Xid, any other container starting, or the deadline. It rewrites a heartbeat file every poll and the harness stops when it is missing
  or older than 10 s. After a trip the kill is verified with `docker inspect` and escalated (`docker kill`, `docker rm -f`, SIGKILL to
  the last known PID) on a worker thread until the container is gone; the watchdog does not exit before that. It sets its own
  `oom_score_adj` to -1000 and `mlockall`, and the container runs with a hard memory limit, swap equal to it and the highest OOM score
  (`a0_watchdog.py flags --memory-gib N` prints the flags);
* inside the harness an out-of-memory stops the run and no turn starts whose projected peak would break the floor;
* hand-back (`a0_watchdog.py handback --drop-dir D ... --avail-floor-gib A --free-floor-gib F`): the screen's files are dropped from the
  page cache, then both MemAvailable and MemFree must reach the floors the operator names (cached files count as available, not free);
* results are counts only (`validity.json`, `arm-*.jsonl`, `deferred.jsonl`, `meta.jsonl`, `calibration.json`); `tf_pair_report.py` reads
  them. A results file whose last line was cut by a kill is read without it.

## Decisions for the owner

D-A0-1 may A2's token ids leave the lab store for the GPU host (else the A1-only fallback); D-A0-2 schedule W0 and W1 and name
who can recover the host; D-A0-3 confirm how the host is cleared and handed back; D-A0-4 log each round's proposal ids privately
for exact lookup counterfactuals (default off; not implemented in this harness: it records counts).
