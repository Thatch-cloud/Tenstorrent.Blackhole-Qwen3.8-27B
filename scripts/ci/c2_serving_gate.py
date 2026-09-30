"""The C2 serving image's gates, run in the node agent's container shape (qwen-c2-serving.yml 'gate').

    python3 scripts/ci/c2_serving_gate.py --image zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag> \
        --profile exact --plan bringup --results "$RUNNER_TEMP/c2-results/gate" [--budget-seconds S]

The fast path has never completed a request inside the serving image (c2-serve-for-real-plan 2.0 item
7): every C2 rate was measured in the m3native gate's own container (16 CPUs, bind mounts, the gate's
argv). This runs that gate's harness (lever_n_m3native_gate.py, mounted at /bench as
lever_n_m3native_run_arm.sh mounts it) INSIDE the serving image, shaped like the agent's container
(read-only root, the agent's tmpfs set, 8 CPUs, 80g, 4g shm, cards M then A by board id, /models =
~/hf-cache/hub, QWEN_C2_SERVING=1 and the job's profile, a p300 descriptor and
QWEN36_BATCHED_DECODE_MODE=host the contract must undo - held against the agent's own recorded
container, references/c2-serving/agent-container-36104200953.json), with the harness launching vLLM from the
platform's argv (--server-argv platform), so what serves is the contract's argv, not the gate's. The
one addition to the agent's shape is a bind mount for the gate's results (the agent's tmpfs cannot be
copied out). Each arm is one container; the image's engine skips tt-metal's teardown at exit, so the
next arm reopens the pair (the platform replay's docker stop/start proved it).

PLANS (C2_GATE_PLAN, run in order). Every plan is checked against the IMAGE's profile before any
container starts (plan_arms): a prompt past what the contract admits (profile_limits, the contract's
own prompt_room: context less the output ceiling or less min_answer_tokens, lowered by
max_prompt_tokens), an answer budget past the ceiling or past what a prompt leaves of the context (the
contract would cut it silently), or a length below the compact framing's template
(real_text_prompts.COMPACT_MIN_TARGET) is refused - except the default ladder's top rung, which is
lowered to the profile's largest admitted prompt with a logged note.
  bringup  4 users x 131072-token real-text prompts, 256 out, EOS on, 0.25 s apart (v235's shape),
           against the tracked v235 texts: IDENTICAL/DIVERGED per user (real_text_compare.
           reference_verdicts). Passes when all four are IDENTICAL, the harness's own verdict (every
           lever's marker, the packed rounds) holds, every stream spent exactly what it asked, and -
           on the profiles v235's engine is (REFERENCE_PROFILES) - the launched argv is v235's engine
           (real_text_compare.served_engine_diff). First, in a throwaway container without devices,
           the image's real-text corpus (its vLLM source) is checked against v235's: another corpus
           builds other prompts, so the plan is NOT_COMPARABLE without spending M+A on it. Gate table
           row Bring-up.
  matrix   real text at one prompt length per user (C2_GATE_LENGTHS, default the G4 ladder) with long
           answers (C2_GATE_MAX_TOKENS), EOS on: a concurrent arm (every user at once, 0.25 s apart;
           more users than seats queue) and a solo arm (the same prompts one at a time, the reference),
           under the exactness policy - a first divergence re-runs both arms once (real_text_compare.
           exactness_policy). Any arm's stream problem, fatal error, a prompt built to another length
           than asked, a 'length' finish short of its budget, or a user whose full-draft acceptance
           collapsed (both arms share one engine, so a corruption common to both would otherwise
           match) FAILS it. Gate table row G4 part 1.
  memory   4 users x the profile's largest admitted prompt x what the contract leaves it (the output
           ceiling, or context less the prompt), concurrent; on a C2-any profile also memory-short, 4
           users x MEMORY_SHORT_PROMPT x the whole ceiling (every proposal bucket per drafter, R3):
           every '[PINDIAG] dram after engine' line and every per-request engine's proposal ladder
           recorded, and the floor printed. Gate table row G5.
  lifecycle  six users on four seats, two event arms (LIFECYCLE_EVENTS) and a solo reference:
           lifecycle-drops  user 0 (the first arrival) leaves while its engine builds, users 1-3 at 4, 3
                            and 2 live streams, user 4 takes user 0's freed seat (max_tokens 256, so it
                            leaves while user 3 still streams), user 5 (ignore_eos)
                            user 1's;
           lifecycle-edges  user 0 (60,000 tokens, the first arrival) is cancelled 5 s into its
                            prefill, users 1-3 leave together once each has 60 chunks (a triple drop
                            in one round), user 4 asks max_tokens=1, user 5 takes a freed seat.
           The harness's StreamWatch times every drop against the server log's prefill/build markers
           and records the phase it hit; an event that missed its phase makes the arm NOT_EXERCISED.
           What each event leaves must agree with the solo text (real_text_compare.lifecycle_pass,
           under the policy). Then, per arm, as many requests at once as the profile has seats must all
           answer (slots were released), and the ledger's idle allocation after every stream, less the
           one before the first, must match the solo arm's to LEDGER_FLAT_GB per chip (the ledger stays
           flat). Gate table row Lifecycle.
Every arm prints the contract's launched-argv line (read-the-launched-argv) and its DRAM lines. Under
a profile that turns C2-any on (QWEN_FAST_ANY_REQUEST=1) every arm that served must show the D2
quarantine consumer's live line (QUARANTINE_LIVE) in its server log, and under any other profile none
of the C2-any markers may appear (ANY_REQUEST_MARKERS): either is a problem that fails the arm.

S2 PLANS (C2-packed-any, s2-design.md W11 and 6.3; the S2 image, graft K64j, with its profiles c2-packed
(traffic: c2 plus the block and QWEN_FAST_EXTENT_REPLAY=1) and c2-packed-gate (c2-gate plus the flag)).
Each arm may serve its own profile and add gate-only environment (Arm: -e NAME=value, never a profile's key):
QWEN_FAST_EXTENT_AUDIT=1 on every arm whose profile has the flag but the control plan's timing arms, the
capture-position knob on G3b's, the forced cap on M4's. Each knob must reach the harness's own process (its
qwen_configuration) or the arm fails (read-the-launched-argv). The exactness policy is strict (CB2a K2 passed
bitwise): a reproduced divergence FAILS, and a flag the card evidence claims exact never makes one
NOT_COMPARABLE (real_text_compare.S2_EXACT_CLAIMS); --policy dc-i (the user's decision D-c(i), with
--policy-decision) is the only relaxation.
  warm, warm-off  M1: solo on c2-packed over every later length (WARM_LENGTHS), short prompts decoding
           past 2048 of history, then 4 x 131072, 4 x 16384 and 4 x 4096 on c2-packed-gate (warm) and on
           c2-gate (warm-off), the two shorter with the capture knob; warm-off also runs exact at v235's shape,
           so M2's exact bring-up (judged for zero new cache entries) finds exact's programs compiled.
           Completes; kernel-cache entries recorded.
  control  M3-M4, G3: --pairs A/B pairs (default 2: ABAB) of c2-gate (A) and c2-packed-gate (B) at v235's
           shape, then one audited B arm (control-audit, QWEN_FAST_EXTENT_AUDIT=1). Every arm IDENTICAL to
           v235; B shows the S2 lines and A none of them. The pairs' B arms are the timing arms and run WITHOUT
           the audit (no audit line may appear in them): runs 36270917139 and 36272682217 showed the audit's
           cost is not confined to the ms its line reports, so a round net of its audit's ms is not the flag's
           round. control-audit judges exactness and the audit (clean, every round), never time.
           RULE CONTROL_RULE (the user's decision, 2026-09-28; control_timing). G3 is judged on two things, both
           hard:
           - exactness: every arm IDENTICAL to v235, control-audit's audit clean and complete, and no audit line
             in a timing arm (a timing arm that ran the audit FAILs its pair).
           - the net: each pair's flag-on median four-live round / its flag-off median at most
             CONTROL_NET_TOLERANCE, else FAIL.
           What the net itself needs is fail-closed: exactly CONTROL_PAIRS pairs (ABAB; any other number is a
           shortfall, never a pass); each arm's timed four-live packed rounds (flag_phase_rounds), matched round k
           with round k by fingerprint - the sorted (segment, position, emitted) of the round's four [PACKED] lines
           - within each pair and across the pairs (sequences that differ are NOT_COMPARABLE; one that stops
           early, or a round either arm did not time, is a shortfall).
           INFORMATIONAL, by the same decision (two pairs on the shared rig cannot bound the flag's own cost
           tighter than about c + 3.5% of a round, so a 0.4% flag almost never passed a 2% limit and a zero-cost
           flag failed about half the time): the flag-phase cost is computed and reported, and never decides the
           verdict - no PASS, FAIL or re-run comes from it. F_k, the summed ms of the phases the extent flag
           touches - the verify trace replay (its device time and the replay deadline's arming), verify-time
           staging, the extent bookkeeping (the rest of the verify), the predictions' readback, the commits (with
           the round's deferred-commit flush) and the window pre-stage, each read from its own line (the comment
           above FLAG_PHASES) - and D_k = F_on,k - F_off,k per pair over the rounds both arms measured, pooled over
           the pairs: c = the trimmed mean of D_k (CONTROL_TRIM_PERCENT from each tail) / the median flag-off whole
           round, u = the untrimmed mean over the same median, their CONTROL_BOUND_PERCENT upper bounds (a
           bootstrap over the rounds plus the arms' own offsets, flag_phase_cost), each phase's differences, the
           arms' offset sigma (the floor CONTROL_ARM_SIGMA_MS, raised by the A/A reads) and the A/A reads. A round
           whose phase line is absent or repeated is reported as unmeasured, never a shortfall. A c or u above
           CONTROL_PHASE_NOTICE is raised as a NOTICE in the lines, the result, the summary and the gate's log -
           reported, not judged. Nothing is NOT_RESOLVED, so there is no pooled re-run.
  forced-cap  M4, Q5: c2-packed-gate with QWEN_FAST_GATE_FORCE_CAP=8 against without, at v235's shape: both
           IDENTICAL to v235 and to each other, and the capped arm's [PACKED] caps all at most 8.
  control-below  M5, G3b: per family F (--families, default 16640 and 4352) and pair, c2-gate against
           c2-packed-gate, both with the capture knob at F - 256, 4 x (F - 256) prompts, 224 out (every ticket
           stays in F). Every user IDENTICAL across the arms; the flag-off arm must serve packed rounds inside
           F, else NO_DECISION (Q22).
  lifecycle-arrival  M6's added event: the arrival that makes the live count two (user 1) leaves in its build,
           with user 0 decoding (one live stream when the drop fired, else NOT_EXERCISED). (lifecycle itself on
           an S2 profile also needs one 'survivor narrowed' line across its arms, else NOT_EXERCISED, and every
           refuse_round request ended FINISHED_ABORTED.)
  mixed, short, boundaries, staggered  M7-M10, G4 (--lengths replaces a plan's default set): a concurrent and
           a solo arm under the policy with one re-run, plus: every arm's extent audit clean and complete,
           zero idle commits, refuse_round rounds, cap refusals and other audit mismatches, the four-live
           per-user rate (net of the audit) above GENERAL_RATE; mixed (M7) always runs the prestage, pair-mask
           and fused-commit audits too (G4_ALL_AUDITS; --audits all adds them to the other G4 plans) and needs a
           packed round of two families, short the one-bucket ladder (2048,) - on every engine line and on the
           executed path's 'proposal buckets built' lines - no DRAM hold with a seat free, no lifted hold or
           refusal, the before-point floors (FLOOR_GB and RESERVE_GB, the S2 admission's split) and no
           request-time buffer above the split's largest-buffer bound, boundaries
           (ignore_eos, 8192 out) cap events and BOUNDARY_MIN_CROSSINGS 256-key crossings per user, staggered
           (STAGGER_SECONDS apart) padded rounds at two or three live and a padded-probe arm.
  churn    M11, G5: CHURN_LENGTHS (12 users, 8 replacements) over four seats, each user's seat freed and
           retaken: no engine death, no DRAM hold with a seat free (holds while every seat decodes are the fifth
           user waiting, not a failed fit), no lifted hold or refusal, the floor, and at every departure while a
           quad was formed a released quad (W6c: the release line just ahead of that step's [PHASE] line).
  permuted  built, required only under D-c(i): two concurrent arms, the second admitted in reverse order;
           at least two SAME_PATH users (else NOT_EXERCISED), and no same-path user may differ.
PARKED PLANS (Stage E, parked per-slot engines, QWEN_FAST_PARKED_ENGINES; the design's G-E0..G-E3). Run on a parked
profile's gate twin (--profile c2-packed-prefix-parked-gate: the audit, negative-control, fault and ballast knobs are
gate-only, and the contract refuses them outside a -gate profile); the arms' flag-off reference is its twin, the profile less '-parked'
(c2-packed-prefix), on the same image and the same prompts. Every parked arm's server log is read with parked_markers
(4 of 4 engines built, the drafter warmed at 2048 with a publish prewarm that counted a pair, no unpark unless the arm
injects one), every flag-off arm's must show none of it, and the judgements are parked_judge's. The strict policy holds
throughout (QWEN_FAST_PARKED_ENGINES is an exactness claim, real_text_compare.S2_EXACT_CLAIMS).
  parked-exact    G-E1 (a) and (f): one solo chain on slot 0 over parked_judge.RUNGS (123,136 first; the design's rungs
           below 45 tokens cannot be sent as real text) with the budgets parked_judge.CHAIN_BUDGETS and ignore_eos on
           the even rungs, sent descending on the flag-off twin (the reference) and on the parked profile in three
           orders (descending, ascending, shuffled: --start-order, the same prompts): every user IDENTICAL to the
           reference and the accepted-prefix sequences equal (G-E1 (g), solo); and the first request after an attach
           (three attaches) IDENTICAL to the reference's.
  parked-drafter  G-E1 (b), (g) and the negative controls, on parked_judge.DRAFTER_LENGTHS (eight users, four seats):
           solo and concurrent arms on the twin and on the parked profile (QWEN_FAST_SHARD_CHECK=1 on the concurrent
           ones), then the parked profile under QWEN_FAST_PARKED_NEGATIVE=carry and =drafter. Parked solo equals twin
           solo in text and in drafting; parked concurrent equals parked solo in text; concurrent acceptance within
           parked_judge.ACCEPTANCE_TOLERANCE; no pair fallback, no "Committed history exceeds", no quad disable, no
           KV_SHARED. The carry control must change tokens; the drafter control must keep them and FAIL the
           equivalence judge.
  parked-lifecycle  G-E2: the lifecycle plan's event arms and solo reference on the parked profile (every row of the
           design's 6.2 that the harness can time), and an arm that injects one park-time fault
           (QWEN_FAST_PARKED_FAULT=park): the slot unparks once, the rest serve, the texts equal the solo arm's.
  parked-churn    G-E3 (i): four arms of PARKED_CHURN_USERS users on four seats, random lengths and budgets
           (parked_judge.churn_script, 200 admissions), QWEN_FAST_PARKED_AUDIT=1: no engine death, no hold with a seat
           free, no refusal, the floors (rebind before-points included), zero unparks, the idle allocation back to the
           attach's (each arm against the plan's median, within IDLE_RETURN_GB plus its idle single rebuilds), the trace region's use within TRACE_SPREAD_GB.
  parked-corner   G-E3 (ii): four users decoding with a quad formed, one leaves, and a 123,136-token arrival takes its
           seat, on the twin and the parked profile: recorded against the design's 5.3; only an engine death or a
           text that differs fails it.
  parked-ballast  G-E3 (iii): parked profile with QWEN_FAST_GATE_DRAM_BALLAST (--ballast-mb, sized from the G-E0 warm
           arm's P7p reading: ballast_advice) - the 123,136-token arrival on an idle server, and beside three
           decoders: no out-of-memory. On the warm plan a parked profile also runs G-E0: the attach at 131,072
           (4 of 4), the P7p reading, the rebind's measured peak (QWEN_FAST_PARKED_AUDIT) and the single rebuild's
           footprint, recorded, with the ballast advice.
QUICKWIN PLANS (the phase-1 quick wins, docs/engine-warm-skip.md; each usable without Stage E). Run on any S2 profile
(--profile c2-packed, c2-packed-prefix, or the parked profile): four arms each, real text, the flag off (the reference) and
on, solo and four concurrent, judged by the strict policy: every user of an on arm IDENTICAL to the off arm's.
  quickwin-ledger  QWEN_FAST_MEMORY_LEDGER_OFF=1 against the image's ledger on. The off arms must carry '[MEMLEDGER]'
           lines (else the comparison switched off nothing: NOT_EXERCISED), the on arms none. Its arms use no
           --drops, which time a drop on the ledger's prefill lines.
  quickwin-warm  QWEN_FAST_ENGINE_WARM_SKIP=1 against the warm forwards run. The solo chain (QUICKWIN_LENGTHS with
           QUICKWIN_BUDGETS: aligned and unaligned prompts, answer budgets that build one-bucket and three-bucket
           engines, ignore_eos throughout) makes a process build engines of both shapes in turn. The on arms must log
           '[PINDIAG] engine warm forwards' lines whose first build warmed and of which a later one skipped (else
           NOT_EXERCISED), the off arms none. The on arms' kernel-cache growth is judged (a skip that left a program
           uncompiled would compile inside a capture).
On an S2 profile the four S1 plans add their S2 checks too (memory: no hold or refusal and the floor).
Every arm of an S2 plan or on an S2 profile (--jit auto) must leave the kernel cache as it found it: a judged
arm that compiled is a problem (M1 warms it first), and a judged arm whose cache could not be counted leaves
its plan NOT_EXERCISED, never PASS. The cache is counted only on such a run (an S2 plan or profile, or --jit
judge/record): a run of S1 plans on an S1 profile reads nothing more than before (counts_cache). A ledger
before-point or DRAM hold that read no DRAM is unjudged (NOT_EXERCISED), never a pass. A non-identical user of
an S2 arm gets a divergence record (t*, character, round, positions, E, paths), and any NOT_COMPARABLE or
UNSTABLE one is listed in the summary's s2_exit_blockers with it (<results>/<plan>-divergence-records.json).

SALTED ARMS (--salt fresh|none, C2_GATE_SALT; the fast path's sticky sessions, harness item B3). Every arm must
then serve a prefix-reuse profile (QWEN_PREFIX_REUSE=1 and the prefix cache on: c2-packed-prefix and its gate
twin), else the gate refuses before any container; warm on c2-packed-prefix serves the prefix twins
(c2-packed-prefix for the solo arm, c2-packed-prefix-gate for the four-user ones) in place of c2-packed and
c2-packed-gate. fresh: the gate writes a salt key (<results>/salt.key), mounts it read-only at SALT_KEY_MOUNT with
QWEN_PREFIX_SALT_KEY_FILE pointing the contract at it (never the operator's key), and the harness gives every
stream its own salt minted under it (lever_n_m3native_gate --cache-salt fresh): each request prefills through the
prefix route's capture path and publishes, and none can hit. none: the harness says so and sends no salt. Either
way, and on a prefix profile with no --salt, each arm's server log is read with prefix_markers
(prefix_log_check): the scheduler graft's install line (and on a sticky profile its sticky install line and the
model graft's warmup line), no row that restored Q > 0 and no sticky admit (no two streams share a salt), no
stale grant, no row that compiled a program (on a sticky profile: no row that compiled a prefill shape an earlier
row of its engine already ran - the fast path compiles each shape at its first request, prefix_judge.fast_path_growth);
under fresh every salted request's grant plans exactly
prefix_judge.fresh_plan (C0 = floor2048(L) - 2048 on a sticky profile) and its row captured what it planned (or a
'capture skipped' line says why), no more 4096+-token requests ran unsalted than the arm's --alive-check (whose
requests are never salted), the salt key was present, and at least one row captured (else NOT_EXERCISED); under
none no request got a grant or captured. Everything else - the S2 checks, the ledger, the floors - is judged as
on c2-packed.

SAFETY. Before each arm: no thatch-inference-* container may exist (a placement reloading onto M+A
mid-gate), and a leftover gate container of the same name is removed; after each arm, however it
ended (a SIGTERM from a cancelled or timed-out step included), its container is removed. An arm
whose server log shows tt-metal's ethernet-core wedge (llrt.cpp:594, the reason the m3native gate
resets before every run) makes its plan INFRA - not a fast-path FAIL - and the arms after it are
skipped. --budget-seconds (the workflow passes what is left of its step and job) refuses a plan list
whose worst case (every arm to its limit, every re-run) does not fit, before anything runs.

Writes <results>/<arm>/ (the gate's stdout, report, server.log, prompts, docker argv),
<results>/c2-gate-summary.json and, for control, <results>/control-phase-rounds.json (every paired round's phases
and the flag-phase report: informational, never judged); exits 0 only when every plan passed, 2 when refused up
front.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import binascii
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_serving_job  # noqa: E402
import lever_n_m3native_gate as harness  # noqa: E402  (stdlib only; real_text_compare imports it too)
import real_text_compare  # noqa: E402
import parked_judge  # noqa: E402  (stdlib and real_text_compare only)
import parked_markers  # noqa: E402  (stdlib only)
import prefix_judge  # noqa: E402  (stdlib only)
import prefix_markers  # noqa: E402  (stdlib only)
import quickwin_markers  # noqa: E402  (stdlib only)
import real_text_prompts  # noqa: E402
import serving_c2_contract  # noqa: E402  (stdlib only at import: json, os, sys)

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'
HUB = '/home/thatch/hf-cache/hub'
SNAPSHOT = '/models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
P300 = '/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto'
IMAGE_PROFILES = '/opt/qwen-c2/profiles.json'
SERVED_NAME = 'Qwen/Qwen3.8-27B'
RESULTS_IN_CONTAINER = '/gate-results'
CONTAINER_PREFIX = 'qwen-c2-gate-'
PLATFORM_PREFIX = 'thatch-inference-'
# What the harness imports from scripts/ci, mounted at /bench over the image's (bundle) copies.
BENCH_SCRIPTS = ('lever_n_m3native_gate.py', 'longctx_cycle_bench.py', 'm3native_ttft_profile.py',
                 'acceptance_report.py', 'real_text_prompts.py')
REFERENCE = os.path.join(HERE, 'references', 'c2-serving', 'v235-real-text-4x131072.json')
# v235's served engine is what these profiles rewrite the platform argv to (test_real_text_gate; c2-gate
# is exact's engine with c2's environment, test_serving_c2_contract): on them a launched argv that is
# not v235's engine fails the bring-up; on any other it is reported.
REFERENCE_PROFILES = ('exact', 'c2-gate', 'c2-packed-gate')
BRINGUP_USERS, BRINGUP_PROMPT, BRINGUP_MAX_TOKENS = 4, 131072, 256
MEMORY_USERS = 4
# G5's short-prompt arm, on a profile that takes any request (ANY_REQUEST_FLAG): four users with prompts
# this short and the profile's whole output ceiling. Each drafter then captures every proposal bucket
# (256..2048: one per rung from min(position, 2048) to min(2048, position + budget), serving_request_factory),
# the per-request worst case the large-prompt arm never reaches (R3).
MEMORY_SHORT_PROMPT = 60
# C2-any (QWEN_FAST_ANY_REQUEST=1, the c2 profiles). The D2 quarantine's consumer logs
# QUARANTINE_LIVE_PREFIX + the running scheduler's class from inside the wrapper, on the engine's first
# step (QUARANTINE_LIVE on the TT platform): a consumer on a class the engine never builds is silent
# until a refusal strands a request (serving_request_quarantine, memory graft-mounted-is-not-graft-
# executed), so every arm under the switch must show it. A subclass of the wrapped class still runs the
# inherited wrapper, so any class counts; the class is recorded and one other than TTScheduler is noted.
# Every per-request
# engine logs its build with its proposal ladder (ANY_REQUEST_ENGINE; R3, G5), recorded per arm. None of
# ANY_REQUEST_MARKERS may appear under a profile that leaves the switch off: exact runs v235's path.
# The one-fresh-prefill cap (serving_prefill_admission) logs ADMISSION_LIVE_PREFIX + the class from its
# first prefill step (the platform warmup). Without it the TTScheduler batches simultaneous arrivals into
# one step and the lifecycle kills the engine (run 36211578069), so every arm under the switch must show it.
ANY_REQUEST_FLAG = 'QWEN_FAST_ANY_REQUEST'
QUARANTINE_LIVE_PREFIX = '[PINDIAG] request quarantine consumer live in '
QUARANTINE_LIVE = QUARANTINE_LIVE_PREFIX + 'TTScheduler'
ANY_REQUEST_ENGINE = '[PINDIAG] any-request engine for '
ADMISSION_LIVE_PREFIX = '[PINDIAG] one fresh prefill per step live in '
ANY_REQUEST_MARKERS = ('[PINDIAG] request quarantine', ANY_REQUEST_ENGINE, '[PINDIAG] attach source check',
                       '[PINDIAG] one fresh prefill per step')
ANY_REQUEST_LINES_KEPT = 32
STAGGER = 0.25            # v235's M3NATIVE_STAGGER: admission in user order
READINESS_SECONDS = 1800
ARM_SECONDS = dict(bringup=3600, matrix=5400, memory=5400, lifecycle=3300)
STREAM_SECONDS = dict(bringup=900, matrix=3600, memory=3600, lifecycle=1800)
ARM_OVERHEAD_SECONDS = 120   # docker start and removal, the profile read, per arm
RERUN_PLANS = ('matrix', 'lifecycle')   # a first divergence runs every arm of these once more
# Lifecycle (gate table row Lifecycle): six users on four seats, so users 4 and 5 are the 5th and 6th
# requests and take seats others free. Two event arms and one solo reference build the same prompts;
# user 0 is the first arrival and long enough that 5 s into its prefill is well inside it (a 60k-token
# prefill takes tens of seconds). --drops grammar: lever_n_m3native_gate.DROP_GRAMMAR.
LIFECYCLE_LENGTHS = (60000, 4096, 8192, 2049, 16384, 255)
LIFECYCLE_MAX_TOKENS = 2048
LIFECYCLE_EVENTS = (
    # User 4 answers at most 256 tokens: run 36222651529 left user 3's live=2 drop NOT_EXERCISED because
    # user 4's natural answer kept three streams live until user 3 reached EOS on its own.
    ('lifecycle-drops', ['--drops', '0:build,1:live=4,2:live=3,3:live=2', '--user-ignore-eos', '5',
                         '--user-max-tokens', '4:256']),
    ('lifecycle-edges', ['--drops', '0:prefill+5,1+2+3:60', '--user-max-tokens', '4:1']),
)
# The phase each drop kind must hit for its event to count (the harness's report['lifecycle']).
EVENT_PHASES = dict(chunks=('decode',), live=('decode',), barrier=('decode',), prefill=('prefill',),
                    build=('build',), seconds=('queued', 'prefill', 'prefill-or-build', 'build'))
BARRIER_SPREAD_SECONDS = 0.5     # a multiple drop's closes, first to last
LEDGER_FLAT_GB = 0.064           # idle drift beyond the solo arm's, per chip (the ledger's own cap, 64 MiB)
# A user whose full-draft rounds emit this few tokens on average has lost its target: real coding text
# accepts 2.76-4.0 per step (c2-serve-for-real-plan 2.2), a corrupt target about one (the bonus token).
MIN_ACCEPTANCE_MEAN, MIN_ACCEPTANCE_ROUNDS = 1.5, 50
WEDGE = 'Timed out while waiting for active ethernet core'
VERDICT_ORDER = ('FAIL', 'INFRA', 'RERUN', 'NOT_COMPARABLE', 'UNSTABLE', 'NOT_EXERCISED', 'NO_DECISION')

# S2 (C2-packed-any; the module docstring's S2 PLANS). The profiles are W8's; the knobs are gate-only.
EXTENT_FLAG = harness.EXTENT_REPLAY_FLAG
AUDIT_ENV = ((harness.EXTENT_AUDIT_FLAG, '1'),)
# M7 (mixed) judges zero mismatches from these three (s2-design 6.3), so its arms always carry them
# (ALL_AUDIT_PLANS); C2_GATE_AUDITS=all adds them to the other G4 arms too. Each is already in the image under its
# parent flag, and its mismatch lines FAIL an S2 arm whenever they appear.
G4_ALL_AUDITS = (('QWEN_FAST_PRESTAGE_AUDIT', '1'), ('QWEN_FAST_PAIR_MASK_AUDIT', '1'),
                 ('QWEN_FAST_FUSED_COMMIT_AUDIT', '1'))
ALL_AUDIT_PLANS = ('mixed',)
PADDED_PROBE_ENV = (('QWEN_FAST_PADDED_PROBE', '1'),)
# What an arm may add with -e: the gate-only knobs, the padded probe and the G4 audits - never a profile key.
ARM_ENV_NAMES = frozenset(harness.S2_GATE_KNOBS + ('QWEN_FAST_PADDED_PROBE',) + tuple(name for name, _ in G4_ALL_AUDITS)
                          # Stage E (the parked plans): the audit, the two controls, the fault, the ballast, the shard check
                          + ('QWEN_FAST_PARKED_AUDIT', 'QWEN_FAST_PARKED_NEGATIVE', 'QWEN_FAST_PARKED_FAULT',
                             'QWEN_FAST_GATE_DRAM_BALLAST', 'QWEN_FAST_SHARD_CHECK')
                          # the phase-1 quick wins (the quickwin plans): the two flags the arms hold on against off
                          + ('QWEN_FAST_MEMORY_LEDGER_OFF', 'QWEN_FAST_ENGINE_WARM_SKIP'))
S2_OFF_PROFILE, S2_GATE_PROFILE, S2_TRAFFIC_PROFILE = 'c2-gate', 'c2-packed-gate', 'c2-packed'
EXACT_PROFILE = 'exact'
S2_PLANS = c2_serving_job.S2_GATE_PLANS
PARKED_PLANS = c2_serving_job.PARKED_GATE_PLANS
QUICKWIN_PLANS = c2_serving_job.QUICKWIN_GATE_PLANS
WARM_PLANS = ('warm', 'warm-off')
DEFAULT_PAIRS = 2
BELOW_FAMILIES = (16640, 4352)
BELOW_MAX_TOKENS = 224           # F - 256 + 223 + 16 <= F: every ticket of a 224-token answer stays in F (B1)
CONTROL_AUDIT_ARM = 'control-audit'   # G3's audited flag-on arm: exactness and the audit, never timed
# G3's rule (the module docstring's control), revision 3: the user's decision of 2026-09-28. Revisions 1 and 2
# ('g3-flag-phase-2026-09-28', '-r2') judged the flag-phase cost against 2% with a NOT_RESOLVED re-run; no run was
# judged under either. Two pairs on the shared rig cannot bound that cost tighter than about c + 3.5% of a round, so
# the flag as measured (0.4%) almost never passed and a zero-cost flag failed about half the time. Revision 3 judges
# exactness and the whole-round net (both hard) and reports the flag-phase cost as facts that never decide the
# verdict; nothing is NOT_RESOLVED, so nothing is re-run or pooled. No number here may move after a run.
CONTROL_RULE = 'g3-net-2026-09-28-r3'   # carried by the result, the summary and the phase record
CONTROL_JUDGED = ('exactness (every arm IDENTICAL to v235, control-audit\'s extent audit clean and complete, no audit '
                  'line in a timing arm) and the net (each pair\'s flag-on / flag-off median four-live round at most '
                  '1.05), both hard')
CONTROL_INFORMATIONAL = ('the flag-phase cost (c, its bound, the untrimmed mean and its bound, the per-phase trimmed '
                         'differences, the arm-noise sigma) is informational only, by the user\'s decision of '
                         '2026-09-28: computed and reported, it never decides the verdict')
CONTROL_PAIRS = 2                # ABAB: a run of any other number of pairs is a shortfall, never a pass
CONTROL_NET_TOLERANCE = 1.05     # the net: each pair's flag-on / flag-off median four-live round (whole round)
# Reported only (flag_phase_cost): how the flag-phase cost and its bounds are computed.
CONTROL_TRIM_PERCENT = 10        # of the paired rounds' D_k, dropped from EACH tail: count * 10 // 100 rounds
CONTROL_BOUND_PERCENT = 95       # c's (and u's) one-sided upper bound: this percentile of the bootstrap's
CONTROL_BOUND_Z = 1.6449         # ... and the standard normal's same percentile, for the arms' offsets
CONTROL_BOOTSTRAP = 10000        # bootstrap resamples of the paired rounds (with replacement)
CONTROL_SEED = 20260928          # the bootstrap's fixed seed (random.Random(int): the same draws on 3.7 and 3.11)
# One arm's own host offset in F (sd, ms), which resampling rounds never sees: every round of an arm carries it, so a
# pair's D carries two (on and off), and the pooled c 2 * sigma^2 / pairs of variance. The floor is the calibration at
# c's own statistic: the trimmed mean of F's paired-round differences between two arms of one flag state, over the
# four A/A reads of arms that ran as the timing arms now run (unaudited: the flag-off pairs of runs 36270917139,
# 36272682217 and 36276533585, and the last's flag-on pair) - -1.33, +2.42, +9.05 and -2.64 ms, sigma = rms / sqrt 2
# = 3.47 ms, rounded up. A run's own A/A reads raise it (flag_phase_sigma), never lower it.
CONTROL_ARM_SIGMA_MS = 3.5
CONTROL_PHASE_NOTICE = 0.02      # a c or u above this fraction of the round is a NOTICE: reported, never judged
CONTROL_PHASE_RECORD = 'control-phase-rounds.json'   # <results>/...: every paired round's phases, reported
# The phases the extent flag touches, read per four-live round (flag_phase_rounds; the module docstring's control).
FLAG_PHASES = ('replay', 'staging', 'readback', 'bookkeeping', 'commit', 'window')
FORCE_CAP = 8
GENERAL_RATE, TARGET_RATE = 13.0, 27.0   # tok/s per user at four live: general's (MEM goal-c2-serving.md:11), S2's aim
# The before-point floors, GB per chip (s2-design 3.3, M8), on the S2 admission's split (serving_prefill_admission;
# gate v79, GitHub run 36368363993, where replacement engines were built wholly in the holes departed users left):
# every point's free less the stranded bytes less its operation's estimate at least FLOOR_GB, and its largest free
# block less the largest single buffer at least RESERVE_GB, the coordinator's DRAM reserve (256 MiB,
# dflash_packed_proposal_coordinator.DRAM_RESERVE_DEFAULT_MB), the one the profiles serve. v79's floor was the largest
# block less the estimate: admitting at its 1079.7 MB block would have read 79.7 MB ahead of the engine build. A
# prefill point's block also loses its prefill transient first, as the admission asks it
# (serving_prefill_admission.admission_contiguous_need): a long prompt's prefill needs a 696.4 MB block. And no ledger
# item allocated at request time (harness.REQUEST_ITEMS) may hold a buffer above the split's LARGEST_BUFFER_BYTES
# (harness.LARGEST_BUFFER_GB): the contiguous term would then be judged on a bound the run broke, so it fails, and the
# remedy is a larger bound. At or below it the bound is not shown to be enough (the walk sees no transient peak).
FLOOR_GB = 0.25
RESERVE_GB = 256 * 2 ** 20 / 1e9
MIXED_LENGTHS = (1536, 20000, 60000, 120000)       # M7's first set; the second, 5000,40000,90000,123136, by --lengths
SHORT_LENGTHS = (60, 255, 2047, 120000)
BOUNDARY_LENGTHS = (8192, 30000, 70000, 110000)
BOUNDARY_MAX_TOKENS = 8192
BOUNDARY_MIN_CROSSINGS = 16
STAGGER_SECONDS = 30.0
# M11: at least 8 replacements (s2-design 3.3): 12 users over the four seats, prompts 100k-123k and one short
# (the one-bucket ladder under churn), last. Budgets 256 tokens apart, so departures are spread and each
# replacement's quad re-forms before the next member leaves; the short user decodes past 2048 of history.
CHURN_LENGTHS = (110000, 120000, 123136, 110000, 120000, 123136, 110000, 120000, 123136, 110000, 120000, 1536)
CHURN_MAX_TOKENS = (768, 1024, 1280, 1536, 768, 1024, 1280, 1536, 768, 1024, 1280, 2304)
CHURN_MIN_REPLACEMENTS = 8
SINGLE_BUCKET = [2048]           # W6a: the one proposal bucket every engine builds under the extent flag
# M1: every length a later arm serves (G4's sets, the lifecycle's, G3b's), solo; below 2048 decoding past 2048 of
# history (WARM_SHORT_TOKENS, ignore_eos) so the drafter's ramp shapes are compiled too.
WARM_LENGTHS = (60, 255, 1536, 2047, 2049, 4096, 5000, 8192, 16384, 20000, 30000, 40000, 60000, 70000, 90000,
                110000, 120000, 123136)
WARM_TOKENS, WARM_SHORT_TOKENS = 256, 2304
ARM_SECONDS.update({'warm': 5400, 'warm-off': 3600, 'control': 3600, 'forced-cap': 3600, 'control-below': 2400,
                    'lifecycle-arrival': 3300, 'mixed': 5400, 'short': 5400, 'boundaries': 5400, 'staggered': 4200,
                    'churn': 9000, 'permuted': 5400})
STREAM_SECONDS.update({'warm': 1800, 'warm-off': 900, 'control': 900, 'forced-cap': 900, 'control-below': 900,
                       'lifecycle-arrival': 1800, 'mixed': 3600, 'short': 3600, 'boundaries': 3600,
                       'staggered': 3600, 'churn': 7200, 'permuted': 3600})
WARM_SHORTER_SECONDS = 2400      # the 4 x 16384 and 4 x 4096 warm arms
# Stage E's parked plans (the module docstring's PARKED PLANS). Every arm limit is what the plan's worst case is summed
# from (worst_case_seconds), so they are the arms' own ceilings, not the plan's estimates: the chain arms take about 25
# minutes, the first-request arms about 12.
ARM_SECONDS.update({'parked-exact': 3600, 'parked-drafter': 3000, 'parked-lifecycle': 2700, 'parked-churn': 3600,
                    'parked-corner': 3600, 'parked-ballast': 2400})
STREAM_SECONDS.update({'parked-exact': 3600, 'parked-drafter': 1800, 'parked-lifecycle': 1800, 'parked-churn': 1800,
                       'parked-corner': 3600, 'parked-ballast': 1800})
PARKED_FIRST_SECONDS = 1500      # G-E1 (f): one request after an attach
# The phase-1 quick-win plans (the module docstring's QUICKWIN PLANS): a solo chain and four concurrent users, each with
# the flag off and on. The chain's lengths mix page-aligned and unaligned prompts and its budgets (2 gives a one-bucket
# engine, 256 a three-bucket one) vary the warm's bucket set; ignore_eos on every user so a budget is a length.
ARM_SECONDS.update({'quickwin-ledger': 3600, 'quickwin-warm': 3600})
STREAM_SECONDS.update({'quickwin-ledger': 3600, 'quickwin-warm': 3600})
QUICKWIN_LENGTHS = (4096, 2049, 32768, 8192, 60000, 4096, 123136, 63)
QUICKWIN_BUDGETS = (2, 64, 256, 2, 128, 256, 64, 32)
QUICKWIN_LIVE_LENGTHS = (4096, 8192, 16384, 32768)
QUICKWIN_LIVE_TOKENS = 256
QUICKWIN_FLAGS = dict(ledger=('QWEN_FAST_MEMORY_LEDGER_OFF', '1'), warm=('QWEN_FAST_ENGINE_WARM_SKIP', '1'))
PARKED_FIRST_PROMPT, PARKED_FIRST_TOKENS = 4096, 64
PARKED_CHURN_ARMS, PARKED_CHURN_USERS, PARKED_CHURN_LONGEST = 4, 50, 100000   # 4 x 50 = 200 admissions
# A window of the corpus per user: 50 users still read at least 123k tokens each, 200 would not (real_text_prompts).
CORNER_LENGTHS = (4096, 8192, 16384, 32768, 123136)     # G-E3 (ii): four decoders, then the arrival
CORNER_MAX_TOKENS, CORNER_ARRIVAL_TOKENS = 2048, 256
CORNER_DROP = '0:400'            # user 0 leaves after 400 chunks: every other decoder past 2048 of history, a quad formed
BALLAST_DECODER_LENGTHS = (20000, 20000, 20000, 123136)   # G-E3 (iii): three decoders, then the arrival
PARKED_AUDIT_ENV = (('QWEN_FAST_PARKED_AUDIT', '1'),)
SHARD_CHECK_ENV = (('QWEN_FAST_SHARD_CHECK', '1'),)
# The kernel cache (B6): the image's TT_METAL_CACHE, under /experiment-cache (a link to /models/.qwen-c2, the
# hub's .qwen-c2 on the host: Dockerfile) or /models (the hub).
KERNEL_CACHE_ENV = 'TT_METAL_CACHE'
CACHE_MOUNTS = (('/experiment-cache/', '.qwen-c2'), ('/models/', ''))
JIT_MODES = c2_serving_job.JIT_MODES
# Salted arms (the module docstring's SALTED ARMS): the gate's own salt key, where the contract reads it.
SALT_KEY_MOUNT = '/c2-gate/salt.key'
SALT_KEY_ENV = serving_c2_contract.SALT_KEY_ENV
SALT_KEY_FILE = 'salt.key'
PREFIX_FLAG = 'QWEN_PREFIX_REUSE'
STICKY_FLAG = 'QWEN_FAST_STICKY_SESSIONS'
SALT_KEY_ABSENT = 'cache_salt kept only when it verifies against the salt key (ABSENT'


class PlanError(ValueError):
    """A plan this profile cannot serve as asked; refused before any container starts."""


# The agent's own container (references/c2-serving/agent-container-36104200953.json, the replay's copy of
# thatch-inference-Qwen-Qwen3.8-27B): its tmpfs set - wider than the smoke step's - and the environment
# the platform adds that is not the image's, with the p300 descriptor and the batched decode mode the
# contract must undo. The Thatch layer's own variables (THATCH_*, PYTHONPATH=/opt/thatch/py,
# VLLM_PLUGINS) belong to its runtime, which the base image does not carry.
AGENT_TMPFS = ('/opt/tt-metal/generated:rw,exec,nosuid,size=1g', '/root/.cache/flashinfer:rw,exec,nosuid,size=512m',
               '/root/.cache/huggingface/hub:rw,nosuid,size=64m', '/root/.cache/tt-metal-cache:rw,exec,nosuid,size=8g',
               '/root/.cache/ttnn:rw,exec,nosuid,size=1g', '/root/.cache/vllm:rw,nosuid,size=256m',
               '/root/.triton:rw,exec,nosuid,size=512m', '/tmp:rw,exec,nosuid,size=512m')
AGENT_ENV = ('HF_HOME=/models', 'HF_HUB_CACHE=/models', 'MESH_DEVICE=P300', 'QWEN36_BATCHED_DECODE_MODE=host',
             'VLLM_RPC_TIMEOUT=100000', 'TT_MESH_GRAPH_DESC_PATH=%s' % P300, 'TRITON_CACHE_DIR=/root/.triton',
             'DO_NOT_TRACK=1', 'VLLM_NO_USAGE_STATS=1', 'PYTHONDONTWRITEBYTECODE=1')


def agent_shape(image, name, profile, devices, hub=HUB, env=()):
    """`docker run` of the node agent's container, up to the image: read-only root, the agent's tmpfs
    set, 8 CPUs, 80g, 4g shm, the two cards in the order given, hugepages, SYS_NICE, the hub at
    /models, the environment the platform adds (AGENT_ENV), and QWEN_C2_SERVING=1 with the profile.
    `env` (an S2 arm's gate-only knobs, ARM_ENV_NAMES) follows as more -e pairs; empty, the argv is
    exactly the agent's."""
    arguments = ['docker', 'run', '--rm', '--name', name, '--read-only']
    for tmpfs in AGENT_TMPFS:
        arguments += ['--tmpfs', tmpfs]
    arguments += ['--shm-size', '4g', '--memory', '80g', '--cpus', '8']
    for device in devices:
        arguments += ['--device', device]
    arguments += ['-v', '/dev/hugepages-1G:/dev/hugepages-1G', '--cap-add', 'SYS_NICE', '-v', '%s:/models' % hub]
    for variable in AGENT_ENV + ('QWEN_C2_SERVING=1', 'QWEN_C2_PROFILE=%s' % profile):
        arguments += ['-e', variable]
    for name_, value in env:
        if name_ not in ARM_ENV_NAMES:
            raise PlanError('%s is not an environment an arm may add (%s): a profile\'s keys are the profile\'s'
                            % (name_, ', '.join(sorted(ARM_ENV_NAMES))))
        arguments += ['-e', '%s=%s' % (name_, value)]
    return arguments


def gate_run(image, name, profile, devices, checkout, arm_dir, gate_args, hub=HUB, env=(), salt=None,
             salt_key_path=None):
    """The whole `docker run` of one arm: the agent's shape, the harness mounted read-only at /bench,
    the arm's results directory, and the harness as the entrypoint. `salt` (the module docstring's SALTED
    ARMS): fresh mounts the gate's salt key and points the contract at it, and both modes tell the harness."""
    arguments = agent_shape(image, name, profile, devices, hub, env)
    for script in BENCH_SCRIPTS:
        arguments += ['--mount', 'type=bind,src=%s,dst=/bench/%s,readonly' % (
            os.path.join(checkout, 'scripts', 'ci', script), script)]
    arguments += ['--mount', 'type=bind,src=%s,dst=%s' % (arm_dir, RESULTS_IN_CONTAINER)]
    salted = []
    if salt == 'fresh':
        if not salt_key_path:
            raise PlanError('--salt fresh needs the gate\'s salt key file')
        arguments += ['--mount', 'type=bind,src=%s,dst=%s,readonly' % (salt_key_path, SALT_KEY_MOUNT),
                      '-e', '%s=%s' % (SALT_KEY_ENV, SALT_KEY_MOUNT)]
        salted = ['--cache-salt', 'fresh', '--cache-salt-key', SALT_KEY_MOUNT]
    elif salt == 'none':
        salted = ['--cache-salt', 'none']
    return (arguments + ['--entrypoint', 'python3', image, '-B', '/bench/lever_n_m3native_gate.py'] + list(gate_args)
            + salted)


def profile_limits(profiles, name):
    """(max-model-len, output ceiling, largest admitted prompt) of a profile, the prompt limit computed
    by the contract's own functions (serving_c2_contract.request_limits and prompt_room, which
    enforce_request uses): context less the output ceiling - or less min_answer_tokens where the profile
    names it (c2: 123,136; c2-gate: 131,072) - lowered by max_prompt_tokens. test_c2_serving_gate holds
    this against enforce_request itself on every profile of the checkout."""
    profile = profiles['profiles'][name]
    context = int(profile['engine']['max-model-len'])
    limits = serving_c2_contract.request_limits(profile)
    room = serving_c2_contract.prompt_room(context, limits['budget'], limits['max_prompt_tokens'],
                                           limits['min_answer_tokens'])
    return context, limits['budget'], room


def profile_seats(profiles, name):
    return int(profiles['profiles'][name]['engine'].get('max-num-seqs') or 1)


def any_request_profile(profiles, name):
    """Whether the profile turns C2-any on (serving_fast_policy.any_request_enabled's switch)."""
    return str((profiles['profiles'][name].get('env') or {}).get(ANY_REQUEST_FLAG, '0')) == '1'


def s2_profile(profiles, name):
    """Whether the profile builds the S2 extent readers (QWEN_FAST_EXTENT_REPLAY=1: c2-packed, c2-packed-gate)."""
    return str((profiles['profiles'][name].get('env') or {}).get(EXTENT_FLAG, '0')) == '1'


def prefix_profile(profiles, name):
    """Whether the profile turns prefix reuse on (QWEN_PREFIX_REUSE=1 with the prefix cache: general-prefix,
    c2-packed-prefix and its gate twin)."""
    profile = profiles['profiles'][name]
    return (str((profile.get('env') or {}).get(PREFIX_FLAG, '0')) == '1'
            and (profile.get('engine') or {}).get('enable-prefix-caching') is True)


def sticky_profile(profiles, name):
    """Whether the profile turns the fast path's sticky sessions on (c2-packed-prefix and its gate twin)."""
    return prefix_profile(profiles, name) and str(
        (profiles['profiles'][name].get('env') or {}).get(STICKY_FLAG, '0')) == '1'


def parked_profile(profiles, name):
    """Whether the profile turns Stage E's parked engines on (QWEN_FAST_PARKED_ENGINES=1: c2-packed-prefix-parked and its
    gate twin)."""
    return name in profiles['profiles'] and str(
        (profiles['profiles'][name].get('env') or {}).get('QWEN_FAST_PARKED_ENGINES', '0')) == '1'


def parked_pair(profiles, profile, plan):
    """(the parked profile, its flag-off twin) a parked plan serves: --profile, and the profile less '-parked'.
    PlanError when --profile is not a parked profile, or its twin is missing, parked itself, or lacks the extent flag."""
    if not parked_profile(profiles, profile):
        raise PlanError('%s is a parked-engines plan: run it on c2-packed-prefix-parked-gate (a -gate profile with '
                        'QWEN_FAST_PARKED_ENGINES=1), not %s' % (plan, profile))
    if not serving_c2_contract.gate_profile(dict(profiles['profiles'][profile], name=profile)):
        raise PlanError('%s adds gate-only knobs (audit, negative controls, fault, ballast) that the contract refuses '
                        'outside a -gate profile: run it on %s-gate, not %s' % (plan, profile, profile))
    twin = profile.replace('-parked', '', 1)
    if twin == profile or twin not in profiles['profiles'] or parked_profile(profiles, twin) \
            or not s2_profile(profiles, twin):
        raise PlanError('%s: profile %s has no flag-off twin %s (the profile less -parked, with %s=1 and without the '
                        'parked flag) to hold it against' % (plan, profile, twin, EXTENT_FLAG))
    return profile, twin


def warm_profiles(profiles, profile):
    """(traffic, gate) the warm plan serves: c2-packed and c2-packed-gate, or - with --profile a sticky prefix
    profile that has a '-gate' twin (c2-packed-prefix) - that profile and its twin, so their capture path warms."""
    if profile.endswith('-gate') and parked_profile(profiles, profile) and profile[:-5] in profiles['profiles']:
        profile = profile[:-5]  # a parked gate profile names its traffic twin: warm-solo serves the traffic profile
    twin = '%s-gate' % profile
    if (profile in profiles['profiles'] and twin in profiles['profiles'] and sticky_profile(profiles, profile)
            and sticky_profile(profiles, twin)):
        return profile, twin
    return S2_TRAFFIC_PROFILE, S2_GATE_PROFILE


def arg_value(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def prefix_log_check(log_text, salt, sticky, alive=0):
    """What an arm served on a prefix-reuse profile shows of the prefix route (the module docstring's SALTED
    ARMS): salt fresh or none (None reads as none). -> (problems, not exercised, facts)."""
    if log_text is None:
        return ['no server.log: the prefix route cannot be checked'], [], {}
    scanned = prefix_markers.scan(log_text.splitlines())
    rows, problems, missing = scanned['rows'], [], []
    if not scanned['installs']:
        problems.append('no "[PINDIAG] prefix: install" line: the scheduler graft never ran in this engine')
    if sticky:
        if not scanned['sticky_installs']:
            problems.append('no "[PINDIAG] prefix: install sticky=1" line on a sticky profile: the scheduler graft does '
                            'not serve the fast path\'s DFlash lookahead')
        if not scanned['model_warm']:
            problems.append('no "[PINDIAG] prefix: model warm" line: the model graft\'s warmup (the restore path, the '
                            'mid-loop capture declaration) did not run on the fast path\'s warmup')
        for entry in scanned['model_warm_skipped']:
            problems.append('the model graft\'s warm was skipped (%s): it chose no restore path' % entry['line'])
    restored = [row for row in rows if row.get('q')]
    if restored:
        problems.append('%d prefill rows restored Q > 0 (%s): no two streams of a gate arm share a salt, so none may '
                        'hit' % (len(restored), ', '.join('%s Q=%s L=%s' % (row.get('req'), row.get('q'), row.get('l'))
                                                         for row in restored[:3])))
    if scanned['sticky_admits']:
        problems.append('%d "[PINDIAG] sticky admit" lines in an arm where nothing may resume' % len(scanned['sticky_admits']))
    for entry in scanned['refused']:
        problems.append('a stale grant was refused at commit (%s start_pos=%s Q=%s)' % (
            entry['req'], entry['start_pos'], entry['q']))
    compiled = []
    if sticky:
        # The fast path never warms its prefill: each shape's first request compiles, a repeated shape must not.
        growth, compiled = prefix_judge.fast_path_growth(rows)
        problems += growth
    else:
        for row in rows:
            before, after = row.get('programs_before'), row.get('programs')
            if isinstance(before, int) and isinstance(after, int) and after > before:
                problems.append('the prefill row of %s (L=%s) compiled %d programs: a compile after the traces were '
                                'parked (F3)' % (row.get('req'), row.get('l'), after - before))
    grants = dict((entry['req'], entry) for entry in scanned['grants'])
    skipped = set(entry['req'] for entry in scanned['capture_skipped'])
    planned, unsalted = 0, 0
    for row in rows:
        grant = grants.get(row.get('req'))
        want = prefix_judge.fresh_plan(row.get('l') or 0, sticky)
        captured = sorted(row.get('captured') or ())
        if grant is None:
            if captured:
                problems.append('%s (L=%s) captured %s with no grant' % (row.get('req'), row.get('l'), captured))
            elif want:
                unsalted += 1
            continue
        if salt != 'fresh':
            problems.append('%s (L=%s) got a grant (%s) in an unsalted arm: fail-closed tenancy broken' % (
                row.get('req'), row.get('l'), grant.get('plan')))
            continue
        planned += 1
        if sorted(grant.get('plan') or ()) != want:
            problems.append('%s (L=%s) was planned captures %s, a fresh salt\'s plan is %s' % (
                row.get('req'), row.get('l'), grant.get('plan'), want))
        if captured != sorted(grant.get('plan') or ()) and row.get('req') not in skipped:
            problems.append('%s (L=%s) captured %s of its plan %s and no "capture skipped" line says why' % (
                row.get('req'), row.get('l'), captured, grant.get('plan')))
    if salt == 'fresh':
        if SALT_KEY_ABSENT in log_text:
            problems.append('the contract logged its salt key ABSENT: every salt was dropped and every request ran '
                            'unsalted (the key mount or %s did not reach it)' % SALT_KEY_ENV)
        if unsalted > alive:
            problems.append('%d requests of 4096+ tokens ran with no grant, more than the arm\'s %d unsalted alive '
                            'requests: their salts were not verified' % (unsalted, alive))
        if not any(row.get('captured') for row in rows):
            missing.append('no prefill row captured a checkpoint: the salted capture path was not exercised')
    facts = dict(rows=len(rows), planned=planned, captured=sum(len(row.get('captured') or ()) for row in rows),
                 unsalted_4096=unsalted, capture_ms=[row.get('capture_ms') for row in rows if row.get('captured')][:16],
                 first_shape_compiles=compiled)
    return problems, missing, facts


class Arm(tuple):
    """One arm: (name, harness arguments, docker timeout) - a plain 3-tuple to everything that unpacks or
    compares it - carrying what an S2 plan adds: the profile it serves when not the gate's --profile, the
    environment it adds (gate-only knobs, ARM_ENV_NAMES), whether a first divergence re-runs it (None: its
    plan's rule, RERUN_PLANS), whether its kernel-cache growth may be judged, and its role in the plan."""

    def __new__(cls, name, args, timeout, profile=None, env=(), rerun=None, judged=True, role=None, **extra):
        arm = tuple.__new__(cls, (name, args, timeout))
        arm.profile, arm.env, arm.rerun, arm.judged, arm.role = profile, tuple(env), rerun, judged, role
        arm.extra = extra
        return arm


def arm_runs(plan, arm):
    """How often an arm may run in the worst case: twice when a first divergence re-runs it."""
    rerun = getattr(arm, 'rerun', None)
    return 2 if (plan in RERUN_PLANS if rerun is None else rerun) else 1


def answer_room(context, lengths, max_tokens):
    """The largest max_tokens every prompt in `lengths` can be served in full: the contract clamps a
    request to context less its prompt, and a 'length' finish short of what the harness asked reads as
    a budget cut (request_problems)."""
    return min([max_tokens] + [context - length for length in lengths])


def common_args(profile, context, stream_seconds, readiness=READINESS_SECONDS):
    return ['--server-argv', 'platform', '--expect-profile', profile, '--served-model-name', SERVED_NAME,
            '--snapshot', SNAPSHOT, '--context', str(context), '--readiness-seconds', str(readiness),
            '--stream-timeout', str(stream_seconds), '--prompt-source', 'real-text', '--allow-missing-references',
            '--eos', 'stop', '--results', RESULTS_IN_CONTAINER]


def check_lengths(profile, lengths, room, what):
    short = [length for length in lengths if length < real_text_prompts.COMPACT_MIN_TARGET]
    if short:
        raise PlanError('%s %s: below %d tokens the compact framing\'s template leaves no room for code '
                        '(real_text_prompts.COMPACT_TEMPLATE_TOKENS)' % (what, short, real_text_prompts.COMPACT_MIN_TARGET))
    long_ = [length for length in lengths if length > room]
    if long_:
        raise PlanError('%s %s exceed %d, the largest prompt profile %s admits today (profile_limits): its edge '
                        'would answer 400' % (what, long_, room, profile))


def check_budget(profile, max_tokens, ceiling, what):
    if max_tokens > ceiling:
        raise PlanError('%s %d exceeds profile %s\'s output ceiling %d: the contract would cut every answer to %d '
                        'without a word' % (what, max_tokens, profile, ceiling, ceiling))


def plan_arms(plan, profile, profiles, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
              memory_prompt=None, notes=None, s2=None):
    """The arms one plan runs, in order: [(arm name, harness arguments, docker timeout seconds)].
    Refuses (PlanError) what the profile cannot serve as asked; `lengths` None is the default ladder,
    whose rungs past the profile's largest admitted prompt are lowered to it (noted in `notes`). An S2
    plan's arms, and the S1 plans' on an S2 profile, are Arms (s2: pairs, families, audits); on any
    other profile the S1 plans' arms are exactly what they were."""
    if plan in S2_PLANS:
        return s2_plan_arms(plan, profile, profiles, lengths, max_tokens, notes, s2 or {})
    if plan in PARKED_PLANS:
        return parked_plan_arms(plan, profile, profiles, lengths, notes, s2 or {})
    if plan in QUICKWIN_PLANS:
        return quickwin_plan_arms(plan, profile, profiles, lengths, notes)
    arms = base_plan_arms(plan, profile, profiles, lengths, max_tokens, memory_prompt, notes)
    if profile in profiles['profiles'] and s2_profile(profiles, profile):
        env = s2_env(profiles, profile, (s2 or {}).get('audits'), g4=plan == 'matrix')
        arms = [Arm(name, args, timeout, env=env) for name, args, timeout in arms]
    return arms


def base_plan_arms(plan, profile, profiles, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
                   memory_prompt=None, notes=None):
    """plan_arms for the four S1 plans."""
    context, ceiling, room = profile_limits(profiles, profile)
    notes = notes if notes is not None else []
    if plan == 'bringup':
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(BRINGUP_USERS), '--prompt-tokens', str(BRINGUP_PROMPT),
            '--max-tokens', str(BRINGUP_MAX_TOKENS), '--stagger', str(STAGGER)]
        return [('bringup-concurrent', args, ARM_SECONDS[plan])]
    if plan == 'matrix':
        if lengths is None:
            lengths = [min(length, room) for length in c2_serving_job.LADDER]
            lowered = [length for length in c2_serving_job.LADDER if length > room]
            if lowered:
                notes.append('matrix: the default ladder\'s %s lowered to %d, the largest prompt profile %s admits'
                             % (lowered, room, profile))
        lengths = list(lengths)
        check_lengths(profile, lengths, room, 'matrix prompt lengths')
        check_budget(profile, max_tokens, ceiling, 'matrix --max-tokens')
        fits = answer_room(context, lengths, max_tokens)
        if fits < max_tokens:
            raise PlanError('matrix --max-tokens %d: a %d-token prompt leaves %d tokens of profile %s\'s context %d, '
                            'so the contract would cut that answer to %d and its \'length\' finish would read as a '
                            'budget cut' % (max_tokens, context - fits, fits, profile, context, fits))
        text = ','.join(str(length) for length in lengths)
        base = common_args(profile, context, STREAM_SECONDS[plan]) + ['--prompt-lengths', text,
                                                                       '--max-tokens', str(max_tokens)]
        return [('matrix-concurrent', base + ['--users', str(len(lengths)), '--stagger', str(STAGGER)],
                 ARM_SECONDS[plan]),
                ('matrix-solo', base + ['--users', '1', '--sequential-users', str(len(lengths))], ARM_SECONDS[plan])]
    if plan == 'memory':
        # The largest prompt gets what the contract leaves it: the ceiling, or context less the prompt
        # when that is smaller (c2: 123,136 + 8,192), so every stream can run to its budget.
        prompt = memory_prompt or room
        check_lengths(profile, [prompt], room, 'memory prompt')
        args = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--users', str(MEMORY_USERS), '--prompt-lengths', ','.join([str(prompt)] * MEMORY_USERS),
            '--max-tokens', str(answer_room(context, [prompt], ceiling)), '--stagger', str(STAGGER)]
        arms = [('memory-concurrent', args, ARM_SECONDS[plan])]
        if any_request_profile(profiles, profile):
            short = [MEMORY_SHORT_PROMPT] * MEMORY_USERS
            check_lengths(profile, short, room, 'memory short prompts')
            arms.append(('memory-short', common_args(profile, context, STREAM_SECONDS[plan]) + [
                '--users', str(MEMORY_USERS), '--prompt-lengths', ','.join(str(length) for length in short),
                '--max-tokens', str(answer_room(context, short, ceiling)), '--stagger', str(STAGGER)],
                ARM_SECONDS[plan]))
        return arms
    if plan == 'lifecycle':
        check_lengths(profile, LIFECYCLE_LENGTHS, room, 'lifecycle prompt lengths')
        check_budget(profile, LIFECYCLE_MAX_TOKENS, ceiling, 'lifecycle --max-tokens')
        base = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--prompt-lengths', ','.join(str(length) for length in LIFECYCLE_LENGTHS),
            '--max-tokens', str(LIFECYCLE_MAX_TOKENS)]
        users = str(len(LIFECYCLE_LENGTHS))
        seats = str(profile_seats(profiles, profile))
        arms = [(arm, base + ['--users', users, '--stagger', str(STAGGER), '--alive-check', seats] + events,
                 ARM_SECONDS[plan]) for arm, events in LIFECYCLE_EVENTS]
        return arms + [('lifecycle-solo', base + ['--users', '1', '--sequential-users', users, '--alive-check', '1'],
                        ARM_SECONDS[plan])]
    raise ValueError('unknown plan %r' % plan)


G4_PLANS = ('mixed', 'short', 'boundaries', 'staggered')


def s2_env(profiles, profile, audits=None, g4=False):
    """What an arm on `profile` adds: the extent audit where the profile has the flag (every packed round of
    every S2 gate arm is audited, s2-design 6.1, but the control plan's timing arms, which s2_plan_arms builds
    without it), and on a G4 arm under audits 'all' the other three audits."""
    if profile not in profiles['profiles'] or not s2_profile(profiles, profile):
        return ()
    return AUDIT_ENV + (G4_ALL_AUDITS if g4 and audits == 'all' else ())


def need_profile(profiles, name, s2, plan):
    """Refuse a plan whose fixed profile the image lacks, or has without (with) the extent flag."""
    if name not in profiles['profiles']:
        raise PlanError('%s serves profile %s, which the image\'s profiles do not carry (s2-design W8 adds c2-packed '
                        'and c2-packed-gate)' % (plan, name))
    if s2_profile(profiles, name) != s2:
        raise PlanError('%s needs profile %s %s %s=1' % (plan, name, 'with' if s2 else 'without', EXTENT_FLAG))


def v235_args(profiles, profile, plan):
    """v235's shape (the bring-up's arguments) on `profile`, or PlanError where its edge refuses it."""
    warning = bringup_warning(profiles, profile)
    if warning:
        raise PlanError('%s: %s' % (plan, warning))
    context = profile_limits(profiles, profile)[0]
    return common_args(profile, context, STREAM_SECONDS[plan]) + [
        '--users', str(BRINGUP_USERS), '--prompt-tokens', str(BRINGUP_PROMPT), '--max-tokens', str(BRINGUP_MAX_TOKENS),
        '--stagger', str(STAGGER)]


def four_user_args(profiles, profile, plan, prompt, max_tokens=BELOW_MAX_TOKENS):
    """Four users at one prompt length (G3b's and M1's 4 x 16384 and 4 x 4096)."""
    context, ceiling, room = profile_limits(profiles, profile)
    check_lengths(profile, [prompt], room, '%s prompt' % plan)
    check_budget(profile, max_tokens, ceiling, '%s --max-tokens' % plan)
    if prompt + max_tokens > context:
        raise PlanError('%s: a %d-token prompt leaves %d of profile %s\'s context, not %d' % (
            plan, prompt, context - prompt, profile, max_tokens))
    return common_args(profile, context, STREAM_SECONDS[plan]) + [
        '--users', '4', '--prompt-tokens', str(prompt), '--max-tokens', str(max_tokens), '--stagger', str(STAGGER)]


def below_families(families):
    """G3b's families, each a multiple of 256 the flag-off block serves (c2_serving_job.BELOW_FAMILY_RANGE)."""
    low, high = c2_serving_job.BELOW_FAMILY_RANGE
    for family in families:
        if family % 256 or not low <= family <= high:
            raise PlanError('control-below: family %d is not one the flag-off block serves (a multiple of 256 in %d..%d)'
                            % (family, low, high))
    return list(families)


def sized_arm_args(profile, profiles, plan, lengths, budget):
    """(common arguments, lengths text, context) for a per-user-length plan, refused as the matrix is."""
    context, ceiling, room = profile_limits(profiles, profile)
    check_lengths(profile, lengths, room, '%s prompt lengths' % plan)
    check_budget(profile, budget, ceiling, '%s --max-tokens' % plan)
    fits = answer_room(context, lengths, budget)
    if fits < budget:
        raise PlanError('%s --max-tokens %d: a %d-token prompt leaves %d tokens of profile %s\'s context %d' % (
            plan, budget, context - fits, fits, profile, context))
    return common_args(profile, context, STREAM_SECONDS[plan]), ','.join(str(length) for length in lengths), context


def s2_plan_arms(plan, profile, profiles, lengths, max_tokens, notes, s2):
    """plan_arms for S2_PLANS (the module docstring's S2 PLANS). control, forced-cap, control-below and the
    warm plans serve their own fixed profiles (c2-gate, c2-packed-gate, c2-packed) whatever --profile is;
    every other S2 plan serves --profile, which must have the extent flag."""
    notes = notes if notes is not None else []
    pairs = int(s2.get('pairs') or DEFAULT_PAIRS)
    seconds = ARM_SECONDS[plan]
    if plan in ('control', 'forced-cap'):
        need_profile(profiles, S2_GATE_PROFILE, True, plan)
        on_env = s2_env(profiles, S2_GATE_PROFILE)
        on_args = v235_args(profiles, S2_GATE_PROFILE, plan)
        if plan == 'forced-cap':
            return [Arm('forced-cap-off', on_args, seconds, profile=S2_GATE_PROFILE, env=on_env, role='off'),
                    Arm('forced-cap-on', on_args, seconds, profile=S2_GATE_PROFILE,
                        env=on_env + ((harness.FORCE_CAP_FLAG, str(FORCE_CAP)),), role='on')]
        need_profile(profiles, S2_OFF_PROFILE, False, plan)
        off_args = v235_args(profiles, S2_OFF_PROFILE, plan)
        arms = []
        for index in range(1, pairs + 1):
            # The timing pair: the flag-on arm without the audit (the module docstring's control).
            arms.append(Arm('control-off-%d' % index, off_args, seconds, profile=S2_OFF_PROFILE, role='off', pair=index))
            arms.append(Arm('control-on-%d' % index, on_args, seconds, profile=S2_GATE_PROFILE, role='on', pair=index))
        # Exactness and the extent audit, after the pairs so their order (ABAB) and spacing are unchanged.
        arms.append(Arm(CONTROL_AUDIT_ARM, on_args, seconds, profile=S2_GATE_PROFILE, env=on_env, role='audit'))
        return arms
    if plan == 'control-below':
        need_profile(profiles, S2_OFF_PROFILE, False, plan)
        need_profile(profiles, S2_GATE_PROFILE, True, plan)
        arms = []
        for family in below_families(s2.get('families') or BELOW_FAMILIES):
            prompt = family - 256
            knob = ((harness.CAPTURE_POSITION_FLAG, str(prompt)),)
            for index in range(1, pairs + 1):
                for role, name in (('off', S2_OFF_PROFILE), ('on', S2_GATE_PROFILE)):
                    arms.append(Arm('below-%d-%s-%d' % (family, role, index), four_user_args(profiles, name, plan, prompt),
                                    seconds, profile=name, env=knob + s2_env(profiles, name), role=role, pair=index,
                                    family=family))
        return arms
    if plan in WARM_PLANS:
        # warm on a sticky prefix profile warms its twins (warm_profiles): the prefix route's capture path.
        traffic, gate = warm_profiles(profiles, profile) if plan == 'warm' else (S2_TRAFFIC_PROFILE, S2_OFF_PROFILE)
        need_profile(profiles, gate, plan == 'warm', plan)
        arms = []
        if plan == 'warm':
            need_profile(profiles, traffic, True, plan)
            parked_warm = parked_profile(profiles, traffic)
            # G-E0: on a parked profile the solo chain also runs every G-E1 rung once (the kernel cache is warm for
            # G-E1's judgement), and every arm reads the parked attach (4 of 4 at 131,072, P7p, R, the single).
            warm = list(lengths or (sorted(set(WARM_LENGTHS) | set(parked_judge.RUNGS)) if parked_warm
                                    else WARM_LENGTHS))
            short = [index for index, length in enumerate(warm) if length < 2048]
            common, text, _ = sized_arm_args(traffic, profiles, plan, warm, WARM_SHORT_TOKENS if short
                                             else WARM_TOKENS)
            args = common + ['--prompt-lengths', text, '--max-tokens', str(WARM_TOKENS), '--users', '1',
                             '--sequential-users', str(len(warm))]
            if short:
                args += ['--user-max-tokens', ','.join('%d:%d' % (index, WARM_SHORT_TOKENS) for index in short),
                         '--user-ignore-eos', ','.join(str(index) for index in short)]
            arms.append(Arm('warm-solo', args, ARM_SECONDS['warm'], profile=traffic,
                            env=s2_env(profiles, traffic), judged=False,
                            role='solo', parked=dict(on=True) if parked_warm else None))
        gate_parked = plan == 'warm' and parked_profile(profiles, gate)
        arms.append(Arm('%s-4x%d' % (plan, BRINGUP_PROMPT), v235_args(profiles, gate, plan), ARM_SECONDS['warm-off'],
                        profile=gate, env=s2_env(profiles, gate) + (PARKED_AUDIT_ENV if gate_parked else ()),
                        judged=False, role='four', parked=dict(on=True) if gate_parked else None))
        for prompt in (BELOW_FAMILIES[0] - 256, BELOW_FAMILIES[1] - 256):
            arms.append(Arm('%s-4x%d' % (plan, prompt), four_user_args(profiles, gate, plan, prompt), WARM_SHORTER_SECONDS,
                            profile=gate, env=((harness.CAPTURE_POSITION_FLAG, str(prompt)),) + s2_env(profiles, gate),
                            judged=False, role='four', parked=dict(on=True) if gate_parked else None))
        if plan == 'warm-off':
            # M2 judges exact's bring-up for zero new cache entries, and exact's environment is neither c2-gate's nor
            # an S2 profile's: its programs are warmed here, at the shape M2 serves.
            need_profile(profiles, EXACT_PROFILE, False, plan)
            arms.append(Arm('warm-off-exact-4x%d' % BRINGUP_PROMPT, v235_args(profiles, EXACT_PROFILE, plan),
                            ARM_SECONDS['warm-off'], profile=EXACT_PROFILE, judged=False, role='four'))
        return arms
    if profile not in profiles['profiles'] or not s2_profile(profiles, profile):
        raise PlanError('%s is an S2 plan: run it on c2-packed (a profile with %s=1), not %s' % (plan, EXTENT_FLAG, profile))
    context, ceiling, room = profile_limits(profiles, profile)
    env = s2_env(profiles, profile, 'all' if plan in ALL_AUDIT_PLANS else s2.get('audits'), g4=plan in G4_PLANS)
    seats = str(profile_seats(profiles, profile))
    if plan == 'lifecycle-arrival':
        check_lengths(profile, LIFECYCLE_LENGTHS, room, 'lifecycle prompt lengths')
        check_budget(profile, LIFECYCLE_MAX_TOKENS, ceiling, 'lifecycle --max-tokens')
        base = common_args(profile, context, STREAM_SECONDS[plan]) + [
            '--prompt-lengths', ','.join(str(length) for length in LIFECYCLE_LENGTHS),
            '--max-tokens', str(LIFECYCLE_MAX_TOKENS)]
        users = str(len(LIFECYCLE_LENGTHS))
        # The event is VR4:150's only while user 0 decodes when user 1's build is dropped: one live stream then.
        return [Arm('lifecycle-arrival', base + ['--users', users, '--stagger', str(STAGGER), '--alive-check', seats,
                                                 '--drops', '1:build'], seconds, env=env, rerun=True, role='event',
                    expect_live={1: 1}),
                Arm('lifecycle-arrival-solo', base + ['--users', '1', '--sequential-users', users, '--alive-check', '1'],
                    seconds, env=env, rerun=True, role='solo')]
    if plan == 'churn':
        churn = list(lengths or CHURN_LENGTHS)
        budgets = list(CHURN_MAX_TOKENS) if lengths is None else [max_tokens] * len(churn)
        common, text, _ = sized_arm_args(profile, profiles, plan, churn, max(budgets))
        args = common + ['--prompt-lengths', text, '--max-tokens', str(max(budgets)), '--users', str(len(churn)),
                         '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(budgets)),
                         '--user-ignore-eos', ','.join(str(index) for index in range(len(churn))),
                         '--stagger', str(STAGGER), '--alive-check', seats]
        if len(churn) - int(seats) < CHURN_MIN_REPLACEMENTS:
            notes.append('churn: %d users over %s seats is %d replacements, fewer than the %d M11 asks for'
                         % (len(churn), seats, len(churn) - int(seats), CHURN_MIN_REPLACEMENTS))
        return [Arm('churn', args, seconds, env=env, role='churn')]
    if plan == 'permuted':
        chosen = list(lengths or MIXED_LENGTHS)
        common, text, _ = sized_arm_args(profile, profiles, plan, chosen, max_tokens)
        base = common + ['--prompt-lengths', text, '--max-tokens', str(max_tokens), '--users', str(len(chosen)),
                         '--stagger', str(STAGGER)]
        reverse = ','.join(str(index) for index in reversed(range(len(chosen))))
        return [Arm('permuted-forward', base, seconds, env=env, role='forward'),
                Arm('permuted-reverse', base + ['--start-order', reverse], seconds, env=env, role='reverse')]
    defaults = dict(mixed=MIXED_LENGTHS, short=SHORT_LENGTHS, boundaries=BOUNDARY_LENGTHS, staggered=MIXED_LENGTHS)
    chosen = list(lengths or defaults[plan])
    budget = BOUNDARY_MAX_TOKENS if plan == 'boundaries' else max_tokens
    if plan == 'boundaries' and max_tokens != BOUNDARY_MAX_TOKENS:
        notes.append('boundaries: every user answers %d tokens with ignore_eos (--max-tokens %d not used), so each '
                     'crosses at least %d 256-key boundaries' % (BOUNDARY_MAX_TOKENS, max_tokens, BOUNDARY_MIN_CROSSINGS))
    common, text, _ = sized_arm_args(profile, profiles, plan, chosen, budget)
    base = common + ['--prompt-lengths', text, '--max-tokens', str(budget)]
    if plan == 'boundaries':
        base += ['--user-ignore-eos', ','.join(str(index) for index in range(len(chosen)))]
    stagger = STAGGER_SECONDS if plan == 'staggered' else STAGGER
    users = str(len(chosen))
    arms = [Arm('%s-concurrent' % plan, base + ['--users', users, '--stagger', str(stagger)], seconds, env=env, rerun=True,
                role='concurrent'),
            Arm('%s-solo' % plan, base + ['--users', '1', '--sequential-users', users], seconds, env=env, rerun=True,
                role='solo')]
    if plan == 'staggered':
        arms.append(Arm('staggered-probe', base + ['--users', users, '--stagger', str(stagger)], seconds,
                        env=env + PADDED_PROBE_ENV, rerun=False, role='probe'))
    return arms


def user_list(values):
    return ','.join(str(value) for value in values)


def parked_chain_args(profiles, profile, plan, lengths, order):
    """The G-E1 solo chain's harness arguments on `profile` in `order` (parked_judge.chain): one server, the requests
    one after another, each user's own budget and ignore_eos, and --start-order for the two other orders."""
    rungs, budgets, ignore, permutation = parked_judge.chain(order)
    if lengths:
        rungs = list(lengths)
        budgets = [parked_judge.CHAIN_BUDGETS[index % len(parked_judge.CHAIN_BUDGETS)] for index in range(len(rungs))]
        ignore = [index for index in range(len(rungs)) if index % 2 == 0]
        if order == 'descending':
            permutation = None
        elif order == 'ascending':
            permutation = list(reversed(range(len(rungs))))
        else:
            permutation = list(range(0, len(rungs), 2)) + list(range(1, len(rungs), 2))
    common, text_, _ = sized_arm_args(profile, profiles, plan, rungs, max(budgets))
    args = common + ['--prompt-lengths', text_, '--max-tokens', str(max(budgets)), '--users', '1',
                     '--sequential-users', str(len(rungs)),
                     '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(budgets)),
                     '--user-ignore-eos', user_list(ignore)]
    if permutation:
        args += ['--start-order', user_list(permutation)]
    return args


def parked_script_args(profiles, profile, plan, lengths, budgets, ignore_all=True, concurrent=False, extra=()):
    """A per-user-length script's harness arguments: solo (--sequential-users) or concurrent (staggered, alive check)."""
    common, text_, _ = sized_arm_args(profile, profiles, plan, lengths, max(budgets))
    args = common + ['--prompt-lengths', text_, '--max-tokens', str(max(budgets)),
                     '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(budgets))]
    if ignore_all:
        args += ['--user-ignore-eos', user_list(range(len(lengths)))]
    if concurrent:
        args += ['--users', str(len(lengths)), '--stagger', str(STAGGER)]
    else:
        args += ['--users', '1', '--sequential-users', str(len(lengths))]
    return args + list(extra)


def parked_plan_arms(plan, profile, profiles, lengths, notes, s2):
    """plan_arms for PARKED_PLANS (the module docstring's PARKED PLANS): the parked profile's arms beside the flag-off
    twin's on the same prompts. `lengths` replaces parked-exact's rungs."""
    notes = notes if notes is not None else []
    parked, twin = parked_pair(profiles, profile, plan)
    seconds = ARM_SECONDS[plan]
    off_env, on_env = s2_env(profiles, twin), s2_env(profiles, parked)
    seats = str(profile_seats(profiles, parked))
    context, ceiling, room = profile_limits(profiles, parked)
    off = dict(parked=dict(on=False))
    if plan == 'parked-exact':
        if lengths is None:
            notes.append('parked-exact: the design\'s rungs 1, 2, 15, 16 and 17 are not sent: a real-text prompt below '
                         '%d tokens has no room for its template (real_text_prompts.COMPACT_MIN_TARGET); the CPU '
                         'tests hold those host checks' % real_text_prompts.COMPACT_MIN_TARGET)
        arms = [Arm('exact-f-desc', parked_chain_args(profiles, twin, plan, lengths, 'descending'), seconds,
                    profile=twin, env=off_env, judged=True, role='ref', **off)]
        for name, order in (('exact-p-desc', 'descending'), ('exact-p-asc', 'ascending'), ('exact-p-shuf', 'shuffled')):
            arms.append(Arm(name, parked_chain_args(profiles, parked, plan, lengths, order), seconds, profile=parked,
                            env=on_env + PARKED_AUDIT_ENV, role='p', order=order, parked=dict(on=True)))
        first = four_user_args(profiles, twin, plan, PARKED_FIRST_PROMPT, PARKED_FIRST_TOKENS)
        single = first[:first.index('--users') + 1] + ['1'] + first[first.index('--users') + 2:]
        for name, served, env, role in (('exact-first-f', twin, off_env, 'first-ref'),
                                        ('exact-first-a', parked, on_env + PARKED_AUDIT_ENV, 'first'),
                                        ('exact-first-b', parked, on_env + PARKED_AUDIT_ENV, 'first')):
            args = list(single)
            args[args.index('--expect-profile') + 1] = served
            arms.append(Arm(name, args, PARKED_FIRST_SECONDS, profile=served, env=env, role=role,
                            parked=dict(on=served == parked)))
        return arms
    if plan == 'parked-drafter':
        chosen = list(lengths or parked_judge.DRAFTER_LENGTHS)
        budgets = [parked_judge.DRAFTER_BUDGET] * len(chosen)
        arms = []

        def script(served, concurrent):
            return parked_script_args(profiles, served, plan, chosen, budgets, concurrent=concurrent)
        arms.append(Arm('drafter-f-solo', script(twin, False), ARM_SECONDS[plan], profile=twin, env=off_env,
                        role='f-solo', **off))
        arms.append(Arm('drafter-p-solo', script(parked, False), ARM_SECONDS[plan], profile=parked,
                        env=on_env + PARKED_AUDIT_ENV, role='p-solo', parked=dict(on=True, zero_counts=True)))
        arms.append(Arm('drafter-f-live', script(twin, True), ARM_SECONDS[plan], profile=twin,
                        env=off_env + SHARD_CHECK_ENV, role='f-live', **off))
        arms.append(Arm('drafter-p-live', script(parked, True), ARM_SECONDS[plan], profile=parked,
                        env=on_env + PARKED_AUDIT_ENV + SHARD_CHECK_ENV, role='p-live',
                        parked=dict(on=True, zero_counts=True)))
        for kind in parked_judge_negatives():
            arms.append(Arm('drafter-neg-%s' % kind, script(parked, False), ARM_SECONDS[plan], profile=parked,
                            env=on_env + (('QWEN_FAST_PARKED_NEGATIVE', kind),), role='neg-%s' % kind, kind=kind,
                            parked=dict(on=True, negative=kind)))
        return arms
    if plan == 'parked-lifecycle':
        check_lengths(parked, LIFECYCLE_LENGTHS, room, 'parked-lifecycle prompt lengths')
        check_budget(parked, LIFECYCLE_MAX_TOKENS, ceiling, 'parked-lifecycle --max-tokens')
        base = common_args(parked, context, STREAM_SECONDS[plan]) + [
            '--prompt-lengths', user_list(LIFECYCLE_LENGTHS), '--max-tokens', str(LIFECYCLE_MAX_TOKENS)]
        users = str(len(LIFECYCLE_LENGTHS))
        arms = [Arm('parked-%s' % arm, base + ['--users', users, '--stagger', str(STAGGER), '--alive-check', seats]
                    + events, seconds, profile=parked, env=on_env + PARKED_AUDIT_ENV, rerun=True, role='event',
                    parked=dict(on=True)) for arm, events in LIFECYCLE_EVENTS]
        solo = base + ['--users', '1', '--sequential-users', users, '--alive-check', '1']
        arms.append(Arm('parked-lifecycle-solo', solo, seconds, profile=parked, env=on_env + PARKED_AUDIT_ENV,
                        rerun=True, role='solo', parked=dict(on=True)))
        arms.append(Arm('parked-lifecycle-fault', solo, seconds, profile=parked,
                        env=on_env + PARKED_AUDIT_ENV + (('QWEN_FAST_PARKED_FAULT', 'park'),), rerun=False,
                        role='fault', parked=dict(on=True, fault=True, allow_unparks=True)))
        return arms
    if plan == 'parked-churn':
        arms = []
        for index in range(PARKED_CHURN_ARMS):
            churn_lengths, churn_budgets = parked_judge.churn_script(index, PARKED_CHURN_USERS, PARKED_CHURN_LONGEST)
            common, text_, _ = sized_arm_args(parked, profiles, plan, churn_lengths, max(churn_budgets))
            args = common + ['--prompt-lengths', text_, '--max-tokens', str(max(churn_budgets)),
                             '--users', str(len(churn_lengths)),
                             '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(churn_budgets)),
                             '--user-ignore-eos', user_list(range(len(churn_lengths))),
                             '--stagger', str(STAGGER), '--alive-check', seats]
            arms.append(Arm('parked-churn-%d' % (index + 1), args, seconds, profile=parked,
                            env=on_env + PARKED_AUDIT_ENV, role='churn', parked=dict(on=True)))
        return arms
    if plan == 'parked-corner':
        budgets = [CORNER_MAX_TOKENS] * 4 + [CORNER_ARRIVAL_TOKENS]
        extra = ['--drops', CORNER_DROP]
        arms = []
        for name, served, env, flags in (('corner-f', twin, off_env, off), ('corner-p', parked, on_env,
                                                                        dict(parked=dict(on=True)))):
            common, text_, _ = sized_arm_args(served, profiles, plan, list(CORNER_LENGTHS), max(budgets))
            args = common + ['--prompt-lengths', text_, '--max-tokens', str(CORNER_MAX_TOKENS), '--users',
                             str(len(CORNER_LENGTHS)), '--user-max-tokens',
                             ','.join('%d:%d' % item for item in enumerate(budgets)), '--user-ignore-eos', '0,1,2,3',
                             '--stagger', str(STAGGER), '--alive-check', seats] + extra
            arms.append(Arm(name, args, seconds, profile=served, env=env, role='ref' if served == twin else 'p',
                            **flags))
        return arms
    if plan == 'parked-ballast':
        megabytes = s2.get('ballast_mb')
        if not megabytes:
            raise PlanError('parked-ballast needs --ballast-mb (C2_GATE_BALLAST_MB): the ballast in MB per chip, sized '
                            'from the G-E0 warm arm\'s P7p reading (ballast_advice)')
        knob = (('QWEN_FAST_GATE_DRAM_BALLAST', str(int(megabytes) * 2 ** 20)),)
        arms = []
        common, text_, _ = sized_arm_args(parked, profiles, plan, [123136], 64)
        arms.append(Arm('ballast-arrival', common + ['--prompt-lengths', text_, '--max-tokens', '64', '--users', '1'],
                        seconds, profile=parked, env=on_env + knob, role='arrival',
                        parked=dict(on=True, ballast=True)))
        lengths_ = list(BALLAST_DECODER_LENGTHS)
        budgets = [CORNER_MAX_TOKENS] * 3 + [CORNER_ARRIVAL_TOKENS]
        common, text_, _ = sized_arm_args(parked, profiles, plan, lengths_, max(budgets))
        arms.append(Arm('ballast-live', common + [
            '--prompt-lengths', text_, '--max-tokens', str(CORNER_MAX_TOKENS), '--users', str(len(lengths_)),
            '--user-max-tokens', ','.join('%d:%d' % item for item in enumerate(budgets)),
            '--user-ignore-eos', '0,1,2', '--stagger', str(STAGGER), '--alive-check', seats],
            seconds, profile=parked, env=on_env + knob, role='live', parked=dict(on=True, ballast=True)))
        return arms
    raise ValueError('unknown parked plan %r' % plan)


def quickwin_plan_arms(plan, profile, profiles, lengths, notes):
    """plan_arms for QUICKWIN_PLANS (the module docstring's QUICKWIN PLANS): --profile served four times, the flag off and
    on, solo (the chain) and four concurrent. `lengths` replaces the chain's. The off arms are the reference."""
    notes = notes if notes is not None else []
    if profile not in profiles['profiles'] or not s2_profile(profiles, profile):
        raise PlanError('%s serves --profile %s, which must have %s=1 (an S2 profile: c2-packed, c2-packed-prefix or its '
                        'parked twin)' % (plan, profile, EXTENT_FLAG))
    kind = plan.split('-', 1)[1]
    knob = (QUICKWIN_FLAGS[kind],)
    chain = list(lengths or QUICKWIN_LENGTHS)
    budgets = [QUICKWIN_BUDGETS[index % len(QUICKWIN_BUDGETS)] for index in range(len(chain))]
    seconds, seats = ARM_SECONDS[plan], str(profile_seats(profiles, profile))
    env = s2_env(profiles, profile)
    solo = parked_script_args(profiles, profile, plan, chain, budgets)
    live = parked_script_args(profiles, profile, plan, list(QUICKWIN_LIVE_LENGTHS),
                              [QUICKWIN_LIVE_TOKENS] * len(QUICKWIN_LIVE_LENGTHS), concurrent=True,
                              extra=['--alive-check', seats])
    if lengths is not None:
        notes.append('%s: the chain runs the lengths asked for, %s' % (plan, chain))
    arms = []
    for label, args in (('solo', solo), ('live', live)):
        for state, added in (('off', ()), ('on', knob)):
            quick = dict(kind=kind, on=state == 'on')
            arms.append(Arm('%s-%s-%s' % (plan, state, label), args, seconds, profile=profile, env=env + added,
                            judged=state == 'on', role='%s-%s' % (state, label), quick=quick))
    return arms


def parked_judge_negatives():
    """The negative controls' names, in the order their arms run (serving_parked_engines.NEGATIVES)."""
    return ('carry', 'drafter')


def worst_case_seconds(plans, arms_of):
    """Every arm to its docker limit, every re-run a first divergence can ask for, plus per-arm overhead."""
    total = 0
    for plan in plans:
        for arm in arms_of[plan]:
            total += arm_runs(plan, arm) * (arm[2] + ARM_OVERHEAD_SECONDS)
    return total


def asked(args):
    """What one arm's harness arguments ask for: streams, a prompt length and a budget per user."""
    def value(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    streams = int(value('--sequential-users', 0) or 0) or int(value('--users', 4))
    if '--prompt-lengths' in args:
        lengths = [int(part) for part in value('--prompt-lengths').split(',')]
    else:
        lengths = [int(value('--prompt-tokens', 32768))] * streams
    max_tokens = int(value('--max-tokens', 256))
    budgets = [max_tokens] * streams
    for part in (value('--user-max-tokens') or '').split(','):
        if part.strip():
            user, _, budget = part.partition(':')
            budgets[int(user)] = int(budget)
    return dict(streams=streams, lengths=lengths, max_tokens=max_tokens, budgets=budgets)


def launched_argv_line(report):
    """The contract's launched argv as its own log line spells it, or None."""
    platform = (report or {}).get('platform') or {}
    if platform.get('served_argv') is None:
        return None
    return '[QWEN-C2] profile %s: vLLM argv %s' % (platform.get('served_profile'), json.dumps(platform['served_argv']))


def dram_lines(report):
    return [event.get('line') for event in ((report or {}).get('dram') or {}).get('events') or []]


def stream_problems(report):
    return list((report or {}).get('real_text_stream_problems') or [])


def request_problems(report, want):
    """What the arm served against what it asked (`want`, from asked()): the prompt lengths built, the
    stream count, the answer budget, and every stream that neither errored nor was dropped: its
    completion_tokens reported, never past its budget, and exactly its budget when it ended 'length' -
    a shorter 'length' is a budget cut somewhere between the request and the engine."""
    problems = []
    built = ((report or {}).get('real_text') or {}).get('prompt_lengths')
    if built is not None and list(built) != list(want['lengths']):
        problems.append('prompts built at %s tokens, asked %s' % (built, want['lengths']))
    if report.get('max_tokens') is not None and report['max_tokens'] != want['max_tokens']:
        problems.append('the harness ran max_tokens %s, asked %d' % (report['max_tokens'], want['max_tokens']))
    streams = report.get('streams') or []
    if len(streams) != want['streams']:
        problems.append('%d streams, asked %d' % (len(streams), want['streams']))
    for index, stream in enumerate(streams):
        stream = stream or {}
        if stream.get('error') or stream.get('dropped'):
            continue
        budget = want['budgets'][index] if index < len(want['budgets']) else want['max_tokens']
        finish, count = stream.get('finish_reason'), stream.get('completion_tokens')
        if count is None:
            problems.append('user %d: no completion_tokens reported' % index)
        elif count > budget:
            problems.append('user %d: %d completion tokens, past its %d budget' % (index, count, budget))
        elif finish == 'length' and count != budget:
            problems.append('user %d: ended "length" after %d of its %d tokens: its budget was cut on the way to '
                            'the engine' % (index, count, budget))
    return problems


def acceptance_problems(report):
    """Users whose full-draft rounds collapsed (MIN_ACCEPTANCE_MEAN over at least MIN_ACCEPTANCE_ROUNDS)."""
    problems = []
    for entry in ((report or {}).get('acceptance') or {}).get('users') or []:
        full = entry.get('full_draft') or {}
        if (full.get('rounds') or 0) >= MIN_ACCEPTANCE_ROUNDS and full.get('mean_emitted') is not None \
                and full['mean_emitted'] < MIN_ACCEPTANCE_MEAN:
            problems.append('user %s: full-draft rounds emit %.2f tokens on average over %d rounds (floor %.1f): '
                            'a target that lost its arithmetic' % (entry.get('user'), full['mean_emitted'],
                                                                   full['rounds'], MIN_ACCEPTANCE_MEAN))
    return problems


def arm_problems(label, report, want=None, acceptance=False):
    """Everything besides the texts that fails an arm: the contract's problems, stream problems, a fatal
    error, what it served against what it asked, and (the matrix) a collapsed acceptance."""
    problems = ['%s: %s' % (label, p) for p in ((report.get('platform') or {}).get('problems') or [])]
    problems += ['%s: %s' % (label, p) for p in report.get('c2_gate_problems') or []]
    problems += ['%s: %s' % (label, p) for p in stream_problems(report)]
    if report.get('fatal'):
        problems.append('%s: fatal: %s' % (label, report['fatal']))
    if want is not None:
        problems += ['%s: %s' % (label, p) for p in request_problems(report, want)]
    if acceptance:
        problems += ['%s: %s' % (label, p) for p in acceptance_problems(report)]
    return problems


def bringup_verdict(report, reference, profile=None, want=None):
    """IDENTICAL/DIVERGED per user against the reference, and PASS only when all four are IDENTICAL,
    the contract served the expected profile - with v235's engine argv, on REFERENCE_PROFILES - every
    stream spent what it asked, and the harness's own verdict held."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report', users=[])
    result = real_text_compare.reference_verdicts(report, reference)
    problems = arm_problems('bring-up', report, want)
    served_argv = (report.get('platform') or {}).get('served_argv')
    argv_diff = real_text_compare.served_engine_diff(served_argv, reference.get('command'))
    if argv_diff and profile in REFERENCE_PROFILES:
        problems.append('bring-up: the launched argv is not v235\'s engine (served | v235): %s' % ', '.join(
            '%s=%s|%s' % (flag, json.dumps(values[0]), json.dumps(values[1])) for flag, values in sorted(argv_diff.items())))
    missing = ((report.get('flag_markers') or {}).get('missing')) or []
    passed = result['verdict'] == 'IDENTICAL' and not problems and bool(report.get('gate_passed'))
    lines = real_text_compare.render_reference(result).split('\n') + ['problem: %s' % p for p in problems]
    if argv_diff and profile not in REFERENCE_PROFILES:
        lines.append('launched argv differs from v235\'s engine in: %s' % ', '.join(sorted(argv_diff)))
    return dict(verdict='PASS' if passed else 'FAIL', texts=result['verdict'], users=result['users'],
                arithmetic_diff=result['arithmetic_diff'], configuration_diff=result['configuration_diff'],
                gate_passed=report.get('gate_passed'), platform_problems=problems, argv_diff=argv_diff,
                missing_markers=missing, lines=lines)


def memory_verdict(report, users=MEMORY_USERS, want=None):
    """G5 records, it does not judge a threshold: PASS when every stream completed as asked and every
    user's engine logged its DRAM line."""
    if report is None:
        return dict(verdict='FAIL', reason='no gate report')
    dram = report.get('dram') or {}
    problems = arm_problems('memory', report, want)
    if (dram.get('engines') or 0) < users:
        problems.append('%s dram-after-engine lines for %d users' % (dram.get('engines') or 0, users))
    return dict(verdict='FAIL' if problems else 'PASS', problems=problems, engines=dram.get('engines'),
                min_free_gb=dram.get('min_free_gb'), min_largest_free_mb=dram.get('min_largest_free_mb'),
                lines=dram_lines(report) + ['problem: %s' % p for p in problems])


def matrix_verdict(concurrent, solo, rerun=None, wants=None, relaxation=None):
    """The exactness policy over the concurrent arm and its solo reference (and their re-run), and
    every arm's own problems (arm_problems, acceptance included): any problem FAILS the plan outright
    (no re-run can clear it). `relaxation` is the user's D-c decision (Runner.relaxation), never a default."""
    if concurrent is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    wants = wants or {}
    result = real_text_compare.exactness_policy(concurrent, solo, rerun, relaxation=relaxation)
    problems = []
    for label, report in (('concurrent', concurrent), ('solo', solo)) + (
            (('concurrent re-run', rerun[0]), ('solo re-run', rerun[1])) if rerun else ()):
        problems += arm_problems(label, report, wants.get(label.replace(' re-run', '')), acceptance=True)
    verdict = 'FAIL' if problems else result['verdict']
    return dict(verdict=verdict, policy=result, platform_problems=problems,
                lines=real_text_compare.render_policy(result).split('\n') + ['problem: %s' % p for p in problems])


def event_shortfalls(report):
    """Every lifecycle drop that did not land where it was meant to (EVENT_PHASES; a live=K drop at K
    live streams; a multiple drop complete and within BARRIER_SPREAD_SECONDS), as text."""
    asked_drops = (report.get('user_events') or {}).get('drops') or {}
    if not asked_drops:
        return []
    lifecycle = report.get('lifecycle')
    if not lifecycle:
        return ['no lifecycle record: the drops were never watched']
    shortfalls = []
    events = lifecycle.get('events') or {}
    for user, event in sorted(events.items(), key=lambda item: int(item[0])):
        kind, spec = event.get('kind'), event.get('spec')
        if not event.get('fired'):
            shortfalls.append('user %s: %s never fired (%s)' % (user, spec, event.get('missed')))
            continue
        if event.get('delivered') is False:
            shortfalls.append('user %s: %s fired but its cancel never reached the socket (%s): the stream ran on' % (
                user, spec, event.get('undelivered')))
        if event.get('phase') not in EVENT_PHASES.get(kind, ()):
            shortfalls.append('user %s: %s fired during %s, not %s' % (
                user, spec, event.get('phase'), ' or '.join(EVENT_PHASES.get(kind, ('?',)))))
        if kind == 'live' and ('live=%s' % event.get('live')) != spec:
            shortfalls.append('user %s: %s fired at %s live streams' % (user, spec, event.get('live')))
    for barrier in lifecycle.get('barriers') or []:
        members = [str(member) for member in barrier.get('members') or []]
        fired = [events.get(member, {}).get('fired_s') for member in members]
        if not barrier.get('complete'):
            shortfalls.append('users %s: the multiple drop released before all had %s chunks' % (
                '+'.join(members), barrier.get('chunks')))
        elif None in fired or max(fired) - min(fired) > BARRIER_SPREAD_SECONDS:
            shortfalls.append('users %s: the multiple drop closed over %s s, not one moment' % (
                '+'.join(members), None if None in fired else round(max(fired) - min(fired), 3)))
    return shortfalls


def ledger_drift(report):
    return ((report or {}).get('ledger') or {}).get('idle_drift_gb')


def lifecycle_verdict(event, solo, rerun=None, want=None, solo_want=None, relaxation=None):
    """One lifecycle event arm against the solo reference, under the exactness policy (a dropped or
    budget-cut user must be a prefix of its solo text, an ignore_eos user must run past it, every other
    user identical), plus what the row asks beyond the texts: the engine answered every seat's request
    afterwards, every stream that was not dropped completed as asked, the contract served the expected
    profile, the ledger's idle drift matches the solo arm's (LEDGER_FLAT_GB), and every drop hit its
    phase (else NOT_EXERCISED). The P7 residual status (attach time) is reported, not judged."""
    if event is None or solo is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    policy = real_text_compare.exactness_policy(event, solo, rerun, real_text_compare.lifecycle_pass,
                                                relaxation=relaxation)
    problems = []
    for label, report, report_want in (('event', event, want), ('solo', solo, solo_want)) + (
            (('event re-run', rerun[0], want), ('solo re-run', rerun[1], solo_want)) if rerun else ()):
        problems += arm_problems(label, report, report_want)
        if not report.get('alive'):
            entries = report.get('alive_after') or []
            entries = entries if isinstance(entries, list) else [entries]
            errors = [str((entry or {}).get('error')) for entry in entries if (entry or {}).get('error')]
            problems.append('%s: the engine did not answer every seat after the streams (%s)' % (
                label, '; '.join(errors) or 'no alive check'))
    shortfalls = event_shortfalls(event) + (['re-run: %s' % s for s in event_shortfalls(rerun[0])] if rerun else [])
    mine, theirs = ledger_drift(event), ledger_drift(solo)
    if mine is None or theirs is None:
        shortfalls.append('the ledger\'s idle readings are missing (event %s, solo %s): flatness not judged' % (
            mine is not None, theirs is not None))
    else:
        grown = {chip: round(mine[chip] - theirs.get(chip, 0.0), 3) for chip in sorted(mine)
                 if abs(mine[chip] - theirs.get(chip, 0.0)) > LEDGER_FLAT_GB}
        if grown:
            problems.append('the ledger did not stay flat: idle allocation after the events differs from the solo '
                            'arm\'s by %s GB per chip (limit %s)' % (grown, LEDGER_FLAT_GB))
    residual = (event.get('ledger') or {}).get('residual_status')
    if problems:
        verdict = 'FAIL'
    elif policy['verdict'] == 'PASS' and shortfalls:
        verdict = 'NOT_EXERCISED'
    else:
        verdict = policy['verdict']
    lines = real_text_compare.render_policy(policy).split('\n') + ['problem: %s' % p for p in problems] + [
        'not exercised: %s' % s for s in shortfalls] + [
        'ledger idle drift (GB per chip) event %s, solo %s; P7 residual %s (attach time, reported only)' % (
            mine, theirs, residual)]
    for user, entry in sorted(((event.get('lifecycle') or {}).get('events') or {}).items(), key=lambda i: int(i[0])):
        lines.append('user %s: %s -> %s at %s s during %s, %s live' % (
            user, entry.get('spec'), entry.get('reason') or 'not fired', entry.get('fired_s'), entry.get('phase'),
            entry.get('live')))
    return dict(verdict=verdict, policy=policy, problems=problems, shortfalls=shortfalls, ledger_residual=residual,
                ledger_drift=dict(event=mine, solo=theirs), events=(event.get('lifecycle') or {}).get('events'),
                lines=lines)


def worst(verdicts):
    for verdict in VERDICT_ORDER:
        if verdict in verdicts:
            return verdict
    return 'PASS' if verdicts else 'FAIL'


def extract(stdout_text):
    try:
        return real_text_compare.extract_report(stdout_text)
    except ValueError:
        return None


def remove_container(name):
    try:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass


def platform_containers():
    """Names of the platform's serving containers present (running or not), or [] if docker cannot say."""
    try:
        output = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [name for name in output.stdout.decode('utf-8', 'replace').split() if name.startswith(PLATFORM_PREFIX)]


def server_log(arm_dir):
    """The arm's server log (the harness writes it beside its report), or None."""
    try:
        with open(os.path.join(arm_dir, 'server.log'), errors='replace') as handle:
            return handle.read()
    except OSError:
        return None


def any_request_check(log_text, any_request):
    """(problems, per-request engine lines, consumer classes) of one served arm's server log. Under a
    C2-any profile the D2 consumer and the one-fresh-prefill cap must both have logged their live lines
    (QUARANTINE_LIVE_PREFIX and ADMISSION_LIVE_PREFIX, any class); under any other no C2-any marker may
    appear."""
    if log_text is None:
        return (['no server.log: the D2 quarantine consumer\'s live line (%s) cannot be checked' % QUARANTINE_LIVE]
                if any_request else []), [], []
    lines = log_text.splitlines()
    engines = [line.strip() for line in lines if ANY_REQUEST_ENGINE in line]
    consumers = sorted(set(line.split(QUARANTINE_LIVE_PREFIX, 1)[1].strip().split(' ')[0]
                           for line in lines if QUARANTINE_LIVE_PREFIX in line))
    if any_request:
        if not consumers:
            return (['%s=1 but the server log has no "%s": the quarantine consumer never ran in the scheduler '
                     'this engine built, so a refusal would strand its request (serving_request_quarantine)' % (
                         ANY_REQUEST_FLAG, QUARANTINE_LIVE)], engines, consumers)
        if not any(ADMISSION_LIVE_PREFIX in line for line in lines):
            return (['%s=1 but the server log has no "%s<class>": the one-fresh-prefill cap never ran, so '
                     'prompts arriving together share a prefill step and fail the engine '
                     '(serving_prefill_admission, run 36211578069)' % (ANY_REQUEST_FLAG, ADMISSION_LIVE_PREFIX)],
                    engines, consumers)
        return [], engines, consumers
    leaked = sorted(marker for marker in ANY_REQUEST_MARKERS if marker in log_text)
    if leaked:
        return (['the profile leaves %s off, but the server log carries C2-any markers (%s): this is not the '
                 'profile\'s path' % (ANY_REQUEST_FLAG, '; '.join(leaked))], engines, consumers)
    return [], engines, consumers


def s2_log_check(log_text, s2_on, env, report):
    """An arm's S2 problems as the host sees them: the harness's own (report['s2']['problems']), or without
    that record an S2 arm's missing record and a flag-off arm's S2 lines; and every -e knob the arm added
    must be in the harness process's configuration with its value (read-the-launched-argv: three runs were
    lost to variables that never crossed into the container)."""
    problems = []
    configuration = (report or {}).get('qwen_configuration')
    for name, value in env:
        if configuration is None:
            problems.append('-e %s=%s: the report records no configuration, so the knob cannot be shown to have '
                            'reached the container' % (name, value))
        elif configuration.get(name) != value:
            problems.append('-e %s=%s was passed, but the harness process carries %s=%r: the knob never reached the '
                            'container' % (name, value, name, configuration.get(name)))
    s2 = (report or {}).get('s2')
    if s2 is not None:
        return problems + ['S2: %s' % problem for problem in s2.get('problems') or []]
    if s2_on:
        return problems + ['%s=1 but the report carries no S2 record (report[\'s2\']): the harness mounted is not '
                           'W11\'s, or it died before reading the log' % EXTENT_FLAG]
    if log_text is not None:
        leaked = sorted(marker for marker in harness.S2_MARKERS if marker in log_text)
        if leaked:
            problems.append('the profile leaves %s off, but the server log carries S2 markers (%s): this is not the '
                            'profile\'s path' % (EXTENT_FLAG, '; '.join(leaked)))
    return problems


def host_live_rate(log_text):
    """acceptance_report.live_rate over the whole server log (both arms of a G3 pair read the same way), or
    an error record; never raises."""
    try:
        import acceptance_report
        return acceptance_report.live_rate(log_text, 4)
    except Exception as error:
        return dict(error='%s: %s' % (type(error).__name__, error))


def host_cache_dir(container_path, hub=HUB):
    """Where the image's kernel cache (its TT_METAL_CACHE) lives on the host: the image links /experiment-cache
    to /models/.qwen-c2, and /models is the hub (CACHE_MOUNTS); None for any other path."""
    for prefix, below in CACHE_MOUNTS:
        if container_path and container_path.startswith(prefix):
            rest = container_path[len(prefix):].strip('/')
            return os.path.join(*([hub] + ([below] if below else []) + ([rest] if rest else [])))
    return None


def count_entries(path):
    """The files under a kernel-cache directory (every JIT build adds some), or None when it cannot be read:
    as this user first, then with sudo -n (the container writes it as root)."""
    if not path:
        return None
    if os.path.isdir(path):
        total, failed = 0, []
        for _root, _dirs, files in os.walk(path, onerror=failed.append):
            total += len(files)
        if not failed:
            return total
    try:
        output = subprocess.run(['sudo', '-n', 'find', path, '-type', 'f'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode:
        return None
    return len([line for line in output.stdout.decode('utf-8', 'replace').splitlines() if line.strip()])


def image_kernel_cache(image, hub=HUB):
    """The image's kernel cache directory on the host (its ENV TT_METAL_CACHE through host_cache_dir), or None."""
    try:
        output = subprocess.run(['docker', 'image', 'inspect', '--format', '{{json .Config.Env}}', image],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        environ = dict(item.split('=', 1) for item in json.loads(output.stdout.decode('utf-8') or 'null') or []
                       if '=' in item)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return host_cache_dir(environ.get(KERNEL_CACHE_ENV), hub)


def wedged(arm_dir):
    """Whether the arm's server log (or the gate's stdout, which quotes its tail) shows the wedge."""
    for name in ('server.log', 'gate-stdout.log'):
        path = os.path.join(arm_dir, name)
        try:
            with open(path, errors='replace') as handle:
                if WEDGE in handle.read():
                    return True
        except OSError:
            continue
    return False


class Runner(object):
    """Runs arms with docker and keeps their files; `execute` and `containers` are injectable for the
    CPU tests. Once an arm hits infrastructure trouble (`infra`: a platform container on M+A, the
    ethernet-core wedge) every later arm is skipped."""

    def __init__(self, image, profile, results, checkout, devices, hub=HUB, execute=None, log=print,
                 containers=None, corpus=None, any_request=False, profiles=None, cache_entries=None, jit='auto',
                 policy='strict', decision=None, salt=None, salt_key_path=None):
        self.image, self.profile, self.results, self.checkout = image, profile, results, checkout
        self.devices, self.hub, self.log = devices, hub, log
        self.execute = execute or self._execute
        self.containers = containers or platform_containers
        self.corpus = corpus or (lambda: image_corpus(image, checkout))
        self.any_request = any_request   # the profile turns C2-any on (any_request_profile)
        # S2: the profiles (an arm may serve another than --profile), the kernel-cache entry counter (None:
        # not counted), whose growth is judged (JIT_MODES) and the exactness policy (strict, or dc-i with the
        # user's recorded decision).
        self.profiles, self.cache_entries, self.jit = profiles, cache_entries, jit
        self.policy, self.decision = policy, decision
        # Salted arms (the module docstring's SALTED ARMS): the mode and the gate's key file.
        self.salt, self.salt_key_path = salt, salt_key_path
        self.arms = {}
        self.infra = None
        # What an arm could not judge (a judged arm whose kernel cache could not be counted): run_plan turns a
        # plan that would pass into NOT_EXERCISED with these.
        self.unjudged = []

    def any_request_for(self, profile):
        if self.profiles is not None and profile in self.profiles['profiles']:
            return any_request_profile(self.profiles, profile)
        return self.any_request

    def s2_for(self, profile):
        return bool(self.profiles is not None and profile in self.profiles['profiles']
                    and s2_profile(self.profiles, profile))

    def prefix_for(self, profile):
        return bool(self.profiles is not None and profile in self.profiles['profiles']
                    and prefix_profile(self.profiles, profile))

    def seats_for(self, profile):
        """The profile's seats (max-num-seqs), MEMORY_USERS when the profiles are not known."""
        if self.profiles is not None and profile in self.profiles['profiles']:
            return profile_seats(self.profiles, profile)
        return MEMORY_USERS

    def judges(self, plan, profile, judged=True):
        """Whether an arm's kernel-cache growth fails it: never a warm arm, and under --jit auto only an S2
        plan's or an S2 profile's (M1 warms exactly those); judge takes every other arm too (M2), record none."""
        if plan in WARM_PLANS or not judged or self.jit == 'record':
            return False
        return (self.jit == 'judge' or plan in S2_PLANS or plan in PARKED_PLANS or plan in QUICKWIN_PLANS
                or self.s2_for(profile))

    def relaxation(self, profile):
        """The relaxation a verdict on `profile` applies: dc-i only on an S2 profile, only with the decision."""
        return 'dc-i' if self.policy == 'dc-i' and self.decision and self.s2_for(profile) else None

    def count_cache(self):
        if self.cache_entries is None:
            return None
        try:
            return self.cache_entries()
        except Exception:
            return None

    @staticmethod
    def _execute(arguments, stdout_path, timeout, name):
        """docker run, the container removed before (a leftover of the same name makes docker exit 125)
        and after, however the run ended - a timeout, or the SIGTERM/SIGINT of a cancelled or timed-out
        step (main turns SIGTERM into SystemExit, so this finally runs)."""
        remove_container(name)
        try:
            with open(stdout_path, 'w') as handle:
                try:
                    return subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout).returncode
                except subprocess.TimeoutExpired:
                    return 'timeout'
        finally:
            remove_container(name)

    def run(self, arm, gate_args, timeout, profile=None, env=(), judged=False, measure=False, parked=None, quick=None):
        """One arm on --profile, or (an S2 arm) on `profile` with `env` added; `judged`: its kernel-cache
        growth is a problem (judges()); `measure`: the host reads its four-live rounds from the whole server
        log (report['c2_gate_live4']: both arms of a G3 pair read the same way). `parked` (a parked plan's arms, and the warm
        plan's on a parked profile): what the arm's server log must show of the parked engines (parked_arm_check).
        `quick` (a quickwin plan's arms): what it must show of the phase-1 quick-win flags (quick_arm_check)."""
        profile = profile or self.profile
        arm_dir = os.path.join(self.results, arm)
        os.makedirs(arm_dir, exist_ok=True)
        os.chmod(arm_dir, 0o777)
        if self.infra is None:
            held = self.containers()
            if held:
                self.infra = 'a platform serving container exists (%s): M+A may be reloaded under the gate' % (
                    ', '.join(held))
        if self.infra is not None:
            self.log('[C2-GATE] arm %s: skipped - %s' % (arm, self.infra))
            self.arms[arm] = dict(exit='skipped', infra=self.infra)
            return None
        name = CONTAINER_PREFIX + arm
        arguments = gate_run(self.image, name, profile, self.devices, self.checkout, arm_dir, gate_args, self.hub,
                             env=env, salt=self.salt, salt_key_path=self.salt_key_path)
        with open(os.path.join(arm_dir, 'docker-run.json'), 'w') as handle:
            json.dump(arguments, handle, indent=1)
        self.log('[C2-GATE] arm %s: %s%s%s' % (arm, ' '.join(gate_args),
                                                ' [profile %s]' % profile if profile != self.profile else '',
                                                ''.join(' [-e %s=%s]' % pair for pair in env)))
        cache_before = self.count_cache()
        started = time.time()
        status = self.execute(arguments, os.path.join(arm_dir, 'gate-stdout.log'), timeout, name)
        seconds = round(time.time() - started, 1)
        cache_after = self.count_cache()
        cache = dict(before=cache_before, after=cache_after, judged=judged,
                     added=None if cache_before is None or cache_after is None else cache_after - cache_before)
        with open(os.path.join(arm_dir, 'gate-stdout.log'), errors='replace') as handle:
            report = extract(handle.read())
        engines, consumers = [], []
        if report is not None:
            # What the server log says about C2-any, judged with the arm's own problems (arm_problems).
            log_text = server_log(arm_dir)
            problems, engines, consumers = any_request_check(log_text, self.any_request_for(profile))
            # S2: the harness's S2 problems, the S2 lines off the flag, the knobs reaching the container, the
            # kernel cache, and the four-live rate from the whole server log.
            problems += s2_log_check(log_text, self.s2_for(profile), env, report)
            if judged and cache['added']:
                problems.append('the kernel cache grew by %d entries during this arm: it compiled what M1 (warm) did '
                                'not (s2-design B6)' % cache['added'])
            if judged and cache['added'] is None:
                self.unjudged.append('%s: its kernel-cache growth is judged but could not be counted (%s): zero new '
                                     'entries is unshown (s2-design B6)' % (
                                         arm, 'no counter: the image\'s cache is not on the hub'
                                         if self.cache_entries is None else 'a count failed'))
            if self.prefix_for(profile):
                # The prefix route under the arm's salts (the module docstring's SALTED ARMS).
                prefix_problems, prefix_missing, facts = prefix_log_check(
                    log_text, self.salt, sticky_profile(self.profiles, profile),
                    alive=int(arg_value(list(gate_args), '--alive-check', 0) or 0))
                problems += ['prefix: %s' % text for text in prefix_problems]
                self.unjudged.extend('%s: %s' % (arm, text) for text in prefix_missing)
                report['c2_gate_prefix'] = dict(salt=self.salt, problems=prefix_problems, not_exercised=prefix_missing,
                                                facts=facts)
            if parked is not None:
                parked_problems, parked_missing, record = parked_arm_check(arm, log_text, parked)
                problems += ['parked: %s' % item for item in parked_problems]
                self.unjudged.extend('%s: %s' % (arm, item) for item in parked_missing)
                report['c2_gate_parked'] = record
            if quick is not None:
                quick_problems, quick_missing, record = quick_arm_check(log_text, quick)
                problems += ['quickwin: %s' % item for item in quick_problems]
                self.unjudged.extend('%s: %s' % (arm, item) for item in quick_missing)
                report['c2_gate_quickwin'] = record
            if log_text is not None and (measure or self.s2_for(profile) or env):
                report['c2_gate_live4'] = host_live_rate(log_text)
            if cache['before'] is not None or self.s2_for(profile) or env:
                report['c2_gate_kernel_cache'] = cache
            if problems:
                report['c2_gate_problems'] = list(report.get('c2_gate_problems') or []) + problems
            with open(os.path.join(arm_dir, 'm3native-gate.json'), 'w') as handle:
                json.dump(report, handle, indent=2)
        line = launched_argv_line(report)
        self.log('[C2-GATE] arm %s: exit %s after %s s; launched: %s' % (arm, status, seconds, line or
                 'NO [QWEN-C2] argv line (%s)' % '; '.join(((report or {}).get('platform') or {}).get('problems')
                                                             or ['no report'])))
        for text in dram_lines(report):
            self.log('[C2-GATE] arm %s: %s' % (arm, text))
        for problem in stream_problems(report):
            self.log('[C2-GATE] arm %s: stream problem: %s' % (arm, problem))
        if report and report.get('fatal'):
            self.log('[C2-GATE] arm %s: fatal: %s' % (arm, report['fatal']))
        for problem in (report or {}).get('c2_gate_problems') or []:
            self.log('[C2-GATE] arm %s: problem: %s' % (arm, problem))
        for text in engines[:8]:
            self.log('[C2-GATE] arm %s: %s' % (arm, text))
        if consumers and consumers != ['TTScheduler']:
            self.log('[C2-GATE] arm %s: note: the D2 consumer ran in %s, not TTScheduler (the scheduler class v235 logged)'
                     % (arm, ', '.join(consumers)))
        infra = None
        if wedged(arm_dir):
            infra = self.infra = ('arm %s hit tt-metal\'s ethernet-core wedge ("%s", llrt.cpp:594): reset M+A '
                                  'before the next run; nothing after it ran' % (arm, WEDGE))
            self.log('[C2-GATE] arm %s: INFRA - %s' % (arm, infra))
        self.arms[arm] = dict(exit=status, seconds=seconds, launched=line, gate_passed=(report or {}).get('gate_passed'),
                              fatal=(report or {}).get('fatal'), infra=infra,
                              any_request_engines=engines[:ANY_REQUEST_LINES_KEPT], quarantine_consumers=consumers)
        if profile != self.profile or env or cache['before'] is not None:
            # S2 arms (and any counted cache): what served and what the kernel cache did.
            self.arms[arm].update(profile=profile, env=['%s=%s' % pair for pair in env], kernel_cache=cache)
            if cache['added'] is not None:
                self.log('[C2-GATE] arm %s: kernel cache %s -> %s entries (%+d%s)' % (
                    arm, cache['before'], cache['after'], cache['added'], ', judged' if judged else ''))
        return report


def parked_arm_check(arm, log_text, expect):
    """(problems, not exercised, record) of one arm's server log against what the arm expects of the parked engines
    (`expect`: on, allow_unparks, fault, negative, ballast, zero_counts): parked_markers.problems for the flag-on arm
    (4 of 4, the 2048 warm with its prewarm, no unpark), none of it for the flag-off one, the gate-only controls
    printing exactly what the arm set, the ballast line when the arm holds a ballast, and - where asked - none of
    parked_judge.LOG_COUNTS. The record keeps the facts the plans judge and print."""
    if log_text is None:
        return ['no server.log: the parked markers cannot be checked'], [], dict(on=bool(expect.get('on')))
    facts = parked_markers.scan(log_text.splitlines())
    on = bool(expect.get('on'))
    problems, missing = parked_markers.problems(facts, on, allow_unparks=bool(expect.get('allow_unparks')))
    if on:
        seen = [(entry['kind'], entry['value']) for entry in facts['negative']]
        wanted = []
        if expect.get('negative'):
            wanted = [('mode', expect['negative'])]
        elif expect.get('fault'):
            wanted = [('fault', 'park')]
        if seen != wanted:
            problems.append('the gate-only controls the server printed are %s, the arm set %s' % (seen or 'none',
                                                                                                 wanted or 'none'))
        if expect.get('ballast') and not facts['ballast']:
            problems.append('no "%s" line: the ballast never reached the attach' % parked_markers.BALLAST.strip())
        if not expect.get('ballast') and facts['ballast']:
            problems.append('a ballast line in an arm that asked for none')
    counts = parked_judge.log_counts(log_text)
    if on and expect.get('zero_counts'):
        problems += parked_judge.count_problems(counts, arm)
    record = dict(on=on, counts=counts, problems=problems, not_exercised=missing, facts=parked_facts_brief(facts))
    return problems, missing, record


def quick_arm_check(log_text, expect):
    """(problems, not exercised, record) of one quickwin arm's server log against what it expects (`expect`: kind 'ledger' or
    'warm', on): the ledger plan's on arm carries no ledger line and its off arm some; the warm plan's on arm the warm-forward
    lines (quickwin_markers.problems) and its off arm none."""
    on, kind = bool(expect.get('on')), expect.get('kind')
    if log_text is None:
        return ['no server.log: the quick-win markers cannot be checked'], [], dict(on=on, kind=kind)
    facts = quickwin_markers.scan(log_text.splitlines())
    problems, missing = quickwin_markers.problems(
        facts, ledger_off=kind == 'ledger' and on, ledger_reference=kind == 'ledger' and not on,
        warm_skip=kind == 'warm' and on)
    return problems, missing, dict(on=on, kind=kind, ledger_lines=facts['ledger'], warm=facts['warm'][:16],
                                   warm_lines=facts['warm_lines'], problems=problems, not_exercised=missing)


def parked_facts_brief(facts):
    """parked_markers.scan's facts as the gate report keeps them: every attach fact whole, the per-request ones counted."""
    peaks = [entry['held'] for entry in facts['peaks']]
    rebuilt = facts['single_rebuilt']
    return dict(built=facts['built'], stopped=facts['stopped'], warm=facts['warm'], ballast=facts['ballast'],
                negative=facts['negative'], prewarm=facts['prewarm'], p7p=facts['p7p'][:8],
                rebinds=len(facts['rebinds']), rebind_ms=[entry['ms'] for entry in facts['rebinds']][:64],
                slots=sorted(set(entry['slot'] for entry in facts['rebinds'])),
                peak_held_max=max(peaks) if peaks else None, peaks=len(peaks), digests=len(facts['digests']),
                digests_unequal=len([entry for entry in facts['digests'] if not entry['equal']]), unparked=facts['unparked'],
                rebinds_with_single=len([entry for entry in facts['rebinds'] if entry['single_rebuilt']]),
                reparked=len(facts['reparked']), single_rebuilt=rebuilt[:8], single_rebuilt_count=len(rebuilt),
                single_kept=len(facts['single_kept']), released=len(facts['released']),
                rebind_ops=len(facts['rebind_ops']))


def parked_of(report):
    """An arm's parked record (Runner.run's report['c2_gate_parked']) - its facts and counts - or {}."""
    return ((report or {}).get('c2_gate_parked') or {})


def parked_facts_of(report):
    return parked_of(report).get('facts') or {}


def arm_report(runner, name):
    """The report an arm wrote (its m3native-gate.json), for a plan that needs an arm the runner did not hand back."""
    try:
        with open(os.path.join(runner.results, name, 'm3native-gate.json')) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def parked_result(lines, problems, shortfalls, facts=None, verdicts=()):
    """A parked plan's result: its lines, then with_checks (FAIL on a problem, NOT_EXERCISED where it would pass)."""
    result = dict(verdict=worst(list(verdicts) + ['PASS']), lines=list(lines))
    return with_checks(result, problems, shortfalls, facts)


def bringup_warning(profiles, profile):
    """Why v235's shape may be refused at this profile's edge, or None. The contract caps a prompt at
    context less the output ceiling, or less min_answer_tokens (profile_limits): c2 admits at most
    123,136, so v235's 131,072-token prompts come back as 400s; c2-gate (min_answer_tokens 256) admits
    them, which is why the c2 bring-up runs on c2-gate."""
    context, ceiling, room = profile_limits(profiles, profile)
    if room >= BRINGUP_PROMPT and context >= BRINGUP_PROMPT + BRINGUP_MAX_TOKENS:
        return None
    return ('profile %s admits prompts up to %d tokens (context %d, output ceiling %d; profile_limits); the '
            'bring-up sends %d-token prompts, which its edge will refuse' % (
                profile, room, context, ceiling, BRINGUP_PROMPT))


def spec_profile(runner, spec):
    """The profile an arm serves: its own (an S2 Arm's), or the gate's --profile."""
    return getattr(spec, 'profile', None) or runner.profile


def run_arm(runner, plan, spec, suffix=''):
    """Run one arm - a plain (name, args, timeout) or an Arm - with its profile, environment and kernel-cache
    judgement (Runner.judges); `suffix` names a re-run ('-rerun')."""
    name, args, timeout = spec
    profile = spec_profile(runner, spec)
    return runner.run(name + suffix, args, timeout, profile=getattr(spec, 'profile', None), env=getattr(spec, 'env', ()),
                      judged=runner.judges(plan, profile, getattr(spec, 'judged', True)),
                      measure=plan in S2_PLANS or plan in PARKED_PLANS or plan in QUICKWIN_PLANS,
                      parked=(getattr(spec, 'extra', None) or {}).get('parked'),
                      quick=(getattr(spec, 'extra', None) or {}).get('quick'))


def s2_of(report):
    return (report or {}).get('s2') or {}


def s2_g4_problems(label, report):
    """What fails an S2 serving arm beyond the harness's problems (s2-design M7): an idle commit, a round that
    went to refuse_round, and any prestage, pair-mask or fused-commit audit mismatch."""
    s2 = s2_of(report)
    if not s2:
        return []
    problems = []
    if s2.get('idle_commits'):
        problems.append('%s: %d "%s" lines' % (label, s2['idle_commits'], harness.PADDED_IDLE_COMMIT_MARKER))
    refused = s2.get('refused') or {}
    if refused.get('refused'):
        problems.append('%s: %d rounds went to refuse_round (%d requests aborted): a serving arm must serve every '
                        'round' % (label, refused['refused'], refused.get('aborted_lines') or 0))
    for name, count in sorted((s2.get('other_audits') or {}).items()):
        if count:
            problems.append('%s: %d %s audit mismatches' % (label, count, name))
    return problems


def memory_s2_checks(label, report, seats=MEMORY_USERS, required_op='engine'):
    """(problems, shortfalls) of G5 on an S2 profile (s2-design B4, B8; M8, M11). A DRAM hold is a failed fit only
    with a seat free: its decodes below `seats`. W6b now asks only then and logs a hold line per (request,
    decodes); its first cut asked with every seat decoding too, where a hold changes nothing (the prompt could not
    be admitted anyway - churn has more users than seats), and such a line is no failure. A deferred step (a stale
    reading, harness.DRAM_DEFERRED_MARKER) is no hold. Also: no lifted hold (admitted with nothing left to wait
    for) and no refused request; the before-point floors of the S2 admission's split over W6d's points and the
    prefill points (harness.before_points): the free less the stranded bytes less the estimate at least FLOOR_GB
    per chip, a negative one included, and the largest free block less the largest single buffer (and, at a prefill
    point, its transient) at least RESERVE_GB; and no request-time ledger item (harness.request_buffers) holding a
    buffer above the split's largest-buffer bound. A hold or a before-point that read no DRAM, or no engine
    before-point under the flag, leaves the floors unjudged: a shortfall, never a pass."""
    s2 = s2_of(report)
    if not s2:
        return [], []
    problems, shortfalls = [], []
    hold = s2.get('dram_hold') or {}
    free = sorted(set(decodes for decodes in hold.get('hold_decodes') or [] if decodes < seats))
    if free:
        problems.append('%s: DRAM held a prompt with a seat free (decodes %s of %d seats): the block and the engines '
                        'did not fit (%s)' % (label, free, seats, '; '.join(hold.get('lines') or [])))
    if hold.get('lifted'):
        problems.append('%s: %d "%s" lines: a prompt that did not fit was admitted with no decode left to wait for '
                        '(%s)' % (label, hold['lifted'], harness.DRAM_LIFTED_MARKER.strip(),
                                  '; '.join(hold.get('lifted_lines') or [])))
    if hold.get('unavailable'):
        shortfalls.append('%s: the DRAM hold read no DRAM for %d requests (%s): their fit is unjudged' % (
            label, hold['unavailable'], '; '.join(hold.get('unavailable_lines') or [])))
    if s2.get('quarantined'):
        problems.append('%s: %d requests refused (RequestRefused, quarantined) (%s)' % (
            label, s2['quarantined'], '; '.join(s2.get('quarantined_lines') or [])))
    before = s2.get('before') or {}
    if before.get('floor_gb') is None:
        problems.append('%s: no ledger before-point carries an estimate: the floor cannot be judged' % label)
    elif before['floor_gb'] < FLOOR_GB:
        problems.append('%s: before-point floor %.3f GB per chip (free less the stranded bytes less the estimate), '
                        'below %.2f (%s %s chip%s)' % (
                            label, before['floor_gb'], FLOOR_GB, (before.get('floor_point') or {}).get('op'),
                            (before.get('floor_point') or {}).get('detail'),
                            (before.get('floor_point') or {}).get('chip')))
    if before.get('floor_gb') is not None and before.get('contiguous_floor_gb') is None:
        problems.append('%s: no ledger before-point carries a largest free block: the contiguous floor cannot be '
                        'judged' % label)
    elif before.get('contiguous_floor_gb') is not None and before['contiguous_floor_gb'] < RESERVE_GB:
        point = before.get('contiguous_point') or {}
        problems.append('%s: before-point contiguous floor %.3f GB per chip (the largest free block less the largest '
                        'buffer, and a prefill\'s transient), below the %.3f GB reserve (%s %s chip%s)' % (
                            label, before['contiguous_floor_gb'], RESERVE_GB, point.get('op'), point.get('detail'),
                            point.get('chip')))
    buffers = s2.get('request_buffers') or {}
    if buffers.get('over'):
        problems.append('%s: %d request-time ledger items hold a buffer above the split\'s %.3f GB largest-buffer bound '
                        '(%s): raise serving_prefill_admission.LARGEST_BUFFER_BYTES' % (
                            label, buffers['over'], buffers.get('limit_gb') or 0.0,
                            '; '.join(request_buffer_text(entry) for entry in buffers.get('over_entries') or [])))
    if before.get('unread'):
        shortfalls.append('%s: %d ledger before-points read no DRAM (%s): the floor over them is unjudged' % (
            label, before['unread'], '; '.join(before.get('unread_lines') or [])))
    if s2.get('extent_replay') and not (before.get('by_op') or {}).get(required_op):
        shortfalls.append('%s: no "[MEMLEDGER] before op=%s" point (W6d): the floor was judged without the '
                          '%s' % (label, required_op, 'engine builds' if required_op == 'engine' else 'rebinds'))
    return problems, shortfalls


def request_buffer_text(entry):
    """One request-time ledger item's largest buffer (harness.request_buffers) as text."""
    entry = entry or {}
    return '%s at %s%s: %.4f GB' % (entry.get('item'), entry.get('phase'),
                                    ' ' + entry['point'] if entry.get('point') else '', entry.get('largest_gb') or 0.0)


def request_buffers_text(buffers):
    """The request-time ledger items' largest buffer against the split's bound, as a line (recorded; above the bound
    is memory_s2_checks' problem)."""
    buffers = buffers or {}
    if not buffers.get('items'):
        return 'request-time largest buffer: not logged (no request item line carries largest=)'
    return 'request-time largest buffer %s over %d items (bound %.3f GB, %d above it; a lower bound only)' % (
        request_buffer_text(buffers.get('largest_entry')), buffers['items'], buffers.get('limit_gb') or 0.0,
        buffers.get('over') or 0)


def live4_of(report):
    """The four-live packed rounds of an arm: the host's reading of its whole log, else the harness's."""
    live = (report or {}).get('c2_gate_live4') or s2_of(report).get('live4')
    return live if live and 'error' not in live else None


# G3's flag phases (the module docstring's control: reported, never judged), each read from one four-live round's own
# stretch of the server log - its '[PHASE] execute' line up to the next - in the producers' formats (a line 'missing'
# below leaves that round's F unmeasured in the report; it never fails or shortens the plan):
#   replay       '[PACKED-PHASE] ... trace_ms=T sync_ms=S ...' T + S (packed_verifier.verify: T is the blocking
#                execute_trace inside the replay deadline's scope - the device time and the deadline's arming - and
#                S the retained replay's own checks and fence around it)
#   staging      '[PACKED-FENCES] ... diff_ms=D write_ms=W' D + W (verify_prestage.BlockPrestage.stage: the
#                verify-time diff against the window's snapshot, and the write; serving_packed_step.FENCES_LINE)
#   readback     '[PACKED-PHASE] ... readback_ms=R' (the predictions read back to the host)
#   bookkeeping  '[PHASE] packed_verify <requests> end V ms' less replay, staging and readback: the rest of the
#                verify (the extent bookkeeping - the segments' starts, the extent round's line - the binding check,
#                the stage call's own overhead, the idle commits and the phase line; serving_worker_hook.phase)
#   commit       the four '[PHASE] packed_commit <request> end C ms' lines, summed (the commits' decisions, the
#                extent's accept_limit cap among them), plus the round's deferred-commit flush: '[PACKED-GDN-AFTER-PAIRS]
#                round=N commits=K site=S enqueue_ms=Q ...' Q (packed_verifier.flush_commits: the commit traces enqueued
#                inside the flag's replay deadline, replay_deadline('flush'), which exists only under the flag). One
#                per four-live round at site window or end (early_draft.IN_STEP_SITES) - in every four-live round of
#                all 13 arms of runs 36270917139, 36272682217 and 36276533585, site=window; absent, repeated, at
#                another site (R1's backstops, the next step's) or a 'dropped=' line, the round's commit is missing
#   window       '[PACKED-PRESTAGE-WINDOW] round=N buffers=B ms=X' (verify_prestage's window pre-stage of the next
#                round, inside this round's early draft). Due exactly when the next step serves a packed round (its
#                stretch carries a [PACKED] line): absent then - or a 'dropped=' line, which carries no ms - it is
#                missing; not due and absent, nothing was pre-staged and it is 0 (the last four-live round before a
#                sequential step, in every arm of runs 36270917139, 36272682217 and 36276533585)
# F_k = V + the commits + Q + X: the four verify phases are the whole packed_verify phase, so no cost inside it
# escapes. Outside F, and too small to judge: the deadline watchdog's own wakeups and the admission checks.
LOG_STAMP = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?) \|')
STEP_FIELDS = re.compile(r'\[PHASE\] execute total=[0-9]+ new=([0-9]+) cached=([0-9]+)')
PHASE_END_FIELDS = re.compile(r'\[PHASE\] (packed_verify|packed_commit) \S+ end (-?[0-9.]+) ms')
VERIFY_SPLIT_FIELDS = re.compile(r'\[PACKED-PHASE\] round=[0-9]+ users=[0-9]+ bind_ms=(-?[0-9.]+) input_ms=(-?[0-9.]+) '
                                 r'trace_ms=(-?[0-9.]+) sync_ms=(-?[0-9.]+) readback_ms=(-?[0-9.]+)')
STAGING_FIELDS = re.compile(r'\[PACKED-FENCES\] round=[0-9]+ [^\n]*?(?<![A-Za-z_])diff_ms=(-?[0-9.]+) '
                            r'write_ms=(-?[0-9.]+)')
WINDOW_FIELDS = re.compile(r'\[PACKED-PRESTAGE-WINDOW\] round=[0-9]+ buffers=[0-9]+ ms=(-?[0-9.]+)')
FLUSH_FIELDS = re.compile(r'\[PACKED-GDN-AFTER-PAIRS\] round=[0-9]+ commits=[0-9]+ site=(\S+) enqueue_ms=(-?[0-9.]+) '
                          r'segments=\S+( dropped=)?')
FLUSH_SITES = ('window', 'end')   # early_draft.IN_STEP_SITES: inside the step that decided the commits
PACKED_KEY_FIELDS = re.compile(r'\[PACKED\] request=\S+ segment=([0-9]+) position=([0-9]+) prefix=[0-9]+ '
                               r'emitted=([0-9]+)')
FLAG_PHASE_LINES = (('verify', "'[PHASE] packed_verify ... end'", 1), ('split', '[PACKED-PHASE]', 1),
                    ('staging', '[PACKED-FENCES]', 1), ('commit', "'[PHASE] packed_commit ... end'", 4),
                    ('flush', "'[PACKED-GDN-AFTER-PAIRS] ... site=window|end enqueue_ms=' (not dropped)", 1))
WINDOW_LINE_NAME = '[PACKED-PRESTAGE-WINDOW] ... ms='


def flag_phase_rounds(log_text):
    """Every four-live packed round of an arm's whole server log - a '[PHASE] execute ... new=0 cached=4' step whose
    stretch, up to the next execute line, carries four [PACKED] lines - in log order: its fingerprint (the sorted
    (segment, position, emitted) of those four lines), its whole round (round_ms: to the next execute line when that
    is a timed decode step, as acceptance_report.decode_steps times it; None when untimed), its flag phases
    (FLAG_PHASES, ms), F (flag_ms) and the flag-phase lines it is missing (a line absent or repeated: that round's F
    is unmeasured - None, never zero)."""
    from datetime import datetime
    steps, current = [], None
    for line in (log_text or '').split('\n'):
        if '[PHASE] execute total=' in line:
            fields = STEP_FIELDS.search(line)
            if fields is None:
                continue
            stamp = LOG_STAMP.search(line)
            at = None
            if stamp:
                text = stamp.group(1)
                at = datetime.strptime(text, '%Y-%m-%d %H:%M:%S.%f' if '.' in text else '%Y-%m-%d %H:%M:%S')
            current = dict(at=at, new=int(fields.group(1)), live=int(fields.group(2)), packed=[], verify=[], commit=[],
                           split=[], staging=[], window=[], flush=[])
            steps.append(current)
            continue
        if current is None:
            continue
        match = PACKED_KEY_FIELDS.search(line)
        if match:
            current['packed'].append([int(match.group(1)), int(match.group(2)), int(match.group(3))])
            continue
        match = PHASE_END_FIELDS.search(line)
        if match:
            current['verify' if match.group(1) == 'packed_verify' else 'commit'].append(float(match.group(2)))
            continue
        match = VERIFY_SPLIT_FIELDS.search(line)
        if match:
            current['split'].append([float(value) for value in match.groups()])
            continue
        match = STAGING_FIELDS.search(line)
        if match:
            current['staging'].append([float(value) for value in match.groups()])
            continue
        match = WINDOW_FIELDS.search(line)
        if match:
            current['window'].append(float(match.group(1)))
            continue
        match = FLUSH_FIELDS.search(line)
        if match:
            # A dropped flush or one at another site is present but not this round's measure: None (missing).
            current['flush'].append(float(match.group(2)) if match.group(1) in FLUSH_SITES and not match.group(3)
                                    else None)
    rounds = []
    for index, step in enumerate(steps):
        if step['new'] != 0 or step['live'] != 4 or len(step['packed']) != 4:
            continue
        following = steps[index + 1] if index + 1 < len(steps) else None
        round_ms = None
        if (following is not None and following['new'] == 0 and following['live'] > 0 and step['at'] is not None
                and following['at'] is not None):
            round_ms = (following['at'] - step['at']).total_seconds() * 1000.0
        if round_ms is not None and round_ms <= 0:
            round_ms = None
        gaps = ['%s x%d' % (what, want) for key, what, want in FLAG_PHASE_LINES
                if len(step[key]) != want or None in step[key]]
        # The window pre-stage is due when the next step serves a packed round (the comment above FLAG_PHASES).
        due = following is not None and bool(following['packed'])
        if len(step['window']) > 1 or (due and not step['window']):
            gaps.append('%s x1%s' % (WINDOW_LINE_NAME, ' (due: the next step is packed)' if due else ''))
        phases = flag_ms = None
        if not gaps:
            _bind, _input, trace, sync, readback = step['split'][0]
            verify, commit, window = step['verify'][0], sum(step['commit']) + step['flush'][0], sum(step['window'])
            replay, staging = trace + sync, sum(step['staging'][0])
            phases = dict(replay=replay, staging=staging, readback=readback,
                          bookkeeping=verify - replay - staging - readback, commit=commit, window=window)
            phases = dict((name, round(value, 3)) for name, value in phases.items())
            flag_ms = round(verify + commit + window, 3)
        rounds.append(dict(fingerprint=sorted(step['packed']), round_ms=None if round_ms is None else round(round_ms, 3),
                           flag_ms=flag_ms, phases=phases, missing=gaps))
    return rounds


def pair_phase_rounds(first, second, names=('flag-off', 'flag-on')):
    """Two arms' four-live packed rounds (flag_phase_rounds) matched by fingerprint, round k with round k:
    (paired, mismatch, shortfalls, unmeasured). `mismatch` says where the sequences differ (NOT_COMPARABLE: not the
    same rounds). Shortfalls are what the net needs (never a pass): no round at all, a sequence that stops early, a
    round either arm did not time. `paired`: [{fingerprint, first, second, measured}] over the rounds both arms timed;
    `measured` is whether both measured every flag phase. `unmeasured` (reported, never judged): None, or how many of
    the timed pairs' flag phases went unmeasured and which lines were missing."""
    for index, (one, other) in enumerate(zip(first, second)):
        if one['fingerprint'] != other['fingerprint']:
            return [], ('four-live packed round %d is %s in the %s arm and %s in the %s arm: not the same rounds' % (
                index, one['fingerprint'], names[0], other['fingerprint'], names[1])), [], None
    shortfalls = []
    if not first or not second:
        shortfalls.append('no four-live packed round in the %s arm' % (names[0] if not first else names[1]))
    elif len(first) != len(second):
        shortfalls.append('the %s arm has %d four-live packed rounds and the %s arm %d: only %d pair' % (
            names[0], len(first), names[1], len(second), min(len(first), len(second))))
    paired, untimed, gaps = [], [], {}
    for index, (one, other) in enumerate(zip(first, second)):
        if one['round_ms'] is None or other['round_ms'] is None:
            untimed.append(index)
            continue
        found = ['%s %s' % (names[0], gap) for gap in one['missing']] + ['%s %s' % (names[1], gap)
                                                                        for gap in other['missing']]
        for gap in found:
            gaps.setdefault(gap, []).append(index)
        paired.append(dict(fingerprint=one['fingerprint'], first=one, second=other, measured=not found))
    if untimed:
        shortfalls.append('%d of %d paired four-live rounds not timed whole (no timed decode step after them): rounds '
                          '%s' % (len(untimed), min(len(first), len(second)), untimed))
    count = len([entry for entry in paired if not entry['measured']])
    unmeasured = dict(rounds=count, of=len(paired), missing=[
        '%s in %d (first round %d)' % (gap, len(rounds), rounds[0]) for gap, rounds in sorted(gaps.items())]) \
        if count else None
    return paired, None, shortfalls, unmeasured


def trimmed_mean(values, percent=CONTROL_TRIM_PERCENT):
    """(mean, k): the mean of `values` less its k lowest and k highest, k = len(values) * percent // 100."""
    ordered = sorted(values)
    k = len(ordered) * percent // 100
    kept = ordered[k:len(ordered) - k]
    return sum(kept) / len(kept), k


def flag_phase_sigma(reads, floor=CONTROL_ARM_SIGMA_MS):
    """(sigma, seen): one arm's own host offset in F (sd, ms) for flag_phase_cost's bounds - the pre-registered floor,
    raised to what the A/A reads show - and that estimate (None without a read). Each read (the trimmed mean of F's
    paired-round differences between two arms of one flag state, control_timing) carries two arms' offsets:
    seen^2 = mean(read^2) / 2."""
    reads = [value for value in reads if value is not None]
    if not reads:
        return floor, None
    seen = math.sqrt(sum(value * value for value in reads) / (2.0 * len(reads)))
    return max(floor, seen), seen


def flag_phase_cost(rounds, pairs=CONTROL_PAIRS, arm_sigma_ms=CONTROL_ARM_SIGMA_MS, seed=CONTROL_SEED,
                    resamples=CONTROL_BOOTSTRAP):
    """The flag-phase cost over paired rounds [(D_k, flag-off whole round), ...] in ms, from `pairs` pairs of arms
    (the module docstring's control) - REPORTED, never a verdict: c = the trimmed mean of D_k / the flag-off median
    round, u = the untrimmed mean over the same median, and their upper bounds. Each bound joins two parts: the
    rounds' - the CONTROL_BOUND_PERCENT percentile of c (or u) over `resamples` bootstrap resamples of the paired
    rounds (each D_k drawn with its own flag-off round, fixed seed), less the estimate - and the arms' -
    CONTROL_BOUND_Z sds of the pooled arms' offsets, arm_sigma_ms * sqrt(2 / pairs) over the same median, which
    resampling rounds cannot see: bound = estimate + sqrt(rounds^2 + arms^2), never below the bootstrap's own
    percentile."""
    deltas = [delta for delta, _ in rounds]
    median_off = statistics.median([off for _, off in rounds])
    trimmed, k = trimmed_mean(deltas)
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    count, draws, untrimmed_draws = len(rounds), [], []
    for _ in range(resamples):
        sample = [rounds[int(rng.random() * count)] for _ in range(count)]
        median = statistics.median([off for _, off in sample])
        sample_deltas = [delta for delta, _ in sample]
        draws.append(trimmed_mean(sample_deltas)[0] / median)
        untrimmed_draws.append(sum(sample_deltas) / count / median)
    draws.sort()
    untrimmed_draws.sort()
    at = resamples * CONTROL_BOUND_PERCENT // 100 - 1
    cost, untrimmed = trimmed / median_off, mean / median_off
    round_upper, untrimmed_round_upper = draws[at], untrimmed_draws[at]
    arms = CONTROL_BOUND_Z * arm_sigma_ms * math.sqrt(2.0 / pairs) / median_off
    upper = cost + math.sqrt(max(round_upper - cost, 0.0) ** 2 + arms ** 2)
    untrimmed_upper = untrimmed + math.sqrt(max(untrimmed_round_upper - untrimmed, 0.0) ** 2 + arms ** 2)
    ordered = sorted(deltas)
    return dict(rounds=count, pairs=pairs, median_off_round_ms=round(median_off, 3),
                trimmed_mean_ms=round(trimmed, 4), mean_ms=round(mean, 4), cost=round(cost, 6),
                untrimmed=round(untrimmed, 6), upper=round(upper, 6), untrimmed_upper=round(untrimmed_upper, 6),
                round_upper=round(round_upper, 6), untrimmed_round_upper=round(untrimmed_round_upper, 6),
                arm_sigma_ms=round(arm_sigma_ms, 4), arm_term=round(arms, 6), trimmed_low=k, trimmed_high=k,
                trimmed_low_ms=[round(value, 3) for value in ordered[:k]],
                trimmed_high_ms=[round(value, 3) for value in ordered[count - k:]],
                bound_percent=CONTROL_BOUND_PERCENT, trim_percent=CONTROL_TRIM_PERCENT, resamples=resamples, seed=seed)


def phase_contrast(paired):
    """Recorded, never judged: per phase and for F, the mean and trimmed mean of second - first over paired rounds."""
    out = {}
    for name in FLAG_PHASES + ('flag_ms',):
        if name == 'flag_ms':
            deltas = [entry['second']['flag_ms'] - entry['first']['flag_ms'] for entry in paired]
        else:
            deltas = [entry['second']['phases'][name] - entry['first']['phases'][name] for entry in paired]
        if deltas:
            out[name] = dict(mean_ms=round(sum(deltas) / len(deltas), 4), trimmed_mean_ms=round(trimmed_mean(deltas)[0], 4))
    return out


def sequence_difference(first, second, names):
    """(mismatch, shortfall) between two four-live round sequences (fingerprint lists): where they differ (NOT_COMPARABLE:
    not the same rounds), or that one stops early (a shortfall); (None, None) when they are the same."""
    for index, (one, other) in enumerate(zip(first, second)):
        if one != other:
            return ('four-live packed round %d is %s in %s and %s in %s: not the same rounds' % (
                index, one, names[0], other, names[1])), None
    if len(first) != len(second):
        return None, '%s has %d four-live packed rounds and %s %d' % (names[0], len(first), names[1], len(second))
    return None, None


def record_rounds(paired):
    return [dict(fingerprint=entry['fingerprint'], off_round_ms=entry['first']['round_ms'],
                 on_round_ms=entry['second']['round_ms'], off_flag_ms=entry['first']['flag_ms'],
                 on_flag_ms=entry['second']['flag_ms'], off_phases=entry['first']['phases'],
                 on_phases=entry['second']['phases']) for entry in paired]


def aa_reads(arm_rounds):
    """The A/A contrasts: each later arm of a flag state against that state's first arm (flag-off against flag-off,
    flag-on against flag-on), paired round by round as the pairs are, over the rounds both measured. Each read's F
    trimmed mean (trimmed_ms) is one draw of two arms' offsets (flag_phase_sigma). Reported, never judged."""
    aa = []
    for state in ('off', 'on'):
        names = sorted((name for name in arm_rounds if name.startswith('control-%s-' % state)),
                       key=lambda name: int(name.rsplit('-', 1)[1]))
        for name in names[1:]:
            paired, mismatch, short, _ = pair_phase_rounds(arm_rounds[names[0]], arm_rounds[name],
                                                           names=(names[0], name))
            paired = [entry for entry in paired if entry['measured']]
            entry = dict(arms='%s - %s' % (name, names[0]), state=state)
            if mismatch or not paired:
                entry['not_compared'] = mismatch or '; '.join(short) or 'no paired round measured'
            else:
                median_off = statistics.median([one['first']['round_ms'] for one in paired])
                contrast = phase_contrast(paired)
                entry.update(rounds=len(paired), phases=contrast, trimmed_ms=contrast['flag_ms']['trimmed_mean_ms'],
                             cost=round(contrast['flag_ms']['trimmed_mean_ms'] / median_off, 6))
            aa.append(entry)
    return aa


def control_timing(runner, arms, reports):
    """G3's timing (the module docstring's control, CONTROL_RULE) over the control plan's pairs.
    JUDGED (hard; with exactness, which run_reference_pairs judges per arm): the pairs' number (CONTROL_PAIRS: any other
    is a shortfall), per pair the refusal of a timing arm that ran the extent audit (FAIL), the four-live rounds of
    both arms matched by fingerprint (pair_phase_rounds: another sequence is NOT_COMPARABLE and leaves that pair's net
    unjudged; no round, one that stops early or a round either arm did not time is a shortfall), the same sequence in
    every pair (sequence_difference), both arms' timed four-live rounds (live4_of) and the net - the flag-on median
    four-live round / the flag-off, at most CONTROL_NET_TOLERANCE, else FAIL.
    REPORTED, never judged (CONTROL_INFORMATIONAL): the flag-phase cost over every pair's rounds whose phases both arms
    measured (flag_phase_cost: c, u, their bounds with the arms' offset, flag_phase_sigma over the A/A reads,
    aa_reads), each pair's and phase's split, the rounds left unmeasured, and a NOTICE when c or u is above
    CONTROL_PHASE_NOTICE. Returns dict(problems, shortfalls, verdicts, pairs, flag_phase, lines, record, notices)."""
    problems, shortfalls, verdicts, net_lines, net = [], [], [], [], []
    measured, arm_rounds, record_pairs, per_pair, unmeasured = [], {}, [], [], []
    indices = sorted(set(spec.extra['pair'] for spec in arms if 'pair' in spec.extra))
    if len(indices) != CONTROL_PAIRS:
        shortfalls.append('the rule times %d pairs (ABAB) and this run has %d: its net is not the rule\'s, never a '
                          'pass' % (CONTROL_PAIRS, len(indices)))
    sequence = None
    for index in indices:
        off_name, on_name = 'control-off-%d' % index, 'control-on-%d' % index
        off_report, on_report = reports.get(off_name), reports.get(on_name)
        audit_lines = (s2_of(on_report).get('extent_audit') or {}).get('lines') or 0
        if on_report is not None and (audit_lines or s2_of(on_report).get('audit')):
            # Fail closed: an audited round is not the flag's round (the module docstring's control).
            marker = harness.EXTENT_AUDIT_MARKER.strip()
            problems.append('pair %d: the flag-on timing arm ran the extent audit (%d "%s" lines): its round is '
                            'not the flag\'s round, so the pair is not timed' % (index, audit_lines, marker))
            continue
        if off_report is None or on_report is None:
            shortfalls.append('pair %d: the %s arm left no gate report: the pair is not timed' % (
                index, 'flag-off' if off_report is None else 'flag-on'))
            continue
        for name in (off_name, on_name):
            arm_rounds[name] = flag_phase_rounds(server_log(os.path.join(runner.results, name)))
        paired, mismatch, short, gaps = pair_phase_rounds(arm_rounds[off_name], arm_rounds[on_name])
        if mismatch:
            # Not the same rounds: the net would compare different work, so it is not judged.
            verdicts.append('NOT_COMPARABLE')
            net_lines.append('pair %d: NOT_COMPARABLE - %s (its net is not judged)' % (index, mismatch))
            continue
        off, on = live4_of(off_report), live4_of(on_report)
        if off is None or on is None:
            shortfalls.append('pair %d: no timed four-live packed round in the %s arm: the round is not compared' % (
                index, 'flag-off' if off is None else 'flag-on'))
        else:
            ratio = on['median_round_ms'] / off['median_round_ms']
            net.append(dict(pair=index, off_ms=off['median_round_ms'], on_ms=on['median_round_ms'], ratio=round(ratio, 4)))
            if ratio > CONTROL_NET_TOLERANCE:
                problems.append('pair %d: the net - the flag-on median four-live round, %.2f ms (unaudited), is %.4f x '
                                'the flag-off %.2f ms, past %.2f' % (index, on['median_round_ms'], ratio,
                                                                     off['median_round_ms'], CONTROL_NET_TOLERANCE))
        # Every pair serves the same rounds (all 12 timing arms of runs 36270917139, 36272682217 and 36276533585 did).
        mine = [one['fingerprint'] for one in arm_rounds[off_name]]
        if sequence is None:
            sequence = dict(pair=index, arm=off_name, fingerprints=mine)
        else:
            mismatch, stops = sequence_difference(sequence['fingerprints'], mine, (sequence['arm'], off_name))
            if mismatch:
                verdicts.append('NOT_COMPARABLE')
                net_lines.append('pair %d: NOT_COMPARABLE - not pair %d\'s rounds: %s' % (index, sequence['pair'],
                                                                                         mismatch))
            elif stops:
                shortfalls.append('pair %d: not pair %d\'s rounds: %s' % (index, sequence['pair'], stops))
        shortfalls += ['pair %d: %s' % (index, text) for text in short]
        if gaps:
            unmeasured.append(dict(gaps, pair=index))
        record_pairs.append(dict(pair=index, rounds=record_rounds(paired)))
        whole = [entry for entry in paired if entry['measured']]
        measured += whole
        entry = dict(pair=index, rounds=len(paired), measured=len(whole))
        if whole:
            deltas = [one['second']['flag_ms'] - one['first']['flag_ms'] for one in whole]
            median_off = statistics.median([one['first']['round_ms'] for one in whole])
            entry.update(
                median_off_round_ms=round(median_off, 3),
                median_off_flag_ms=round(statistics.median([one['first']['flag_ms'] for one in whole]), 3),
                median_on_flag_ms=round(statistics.median([one['second']['flag_ms'] for one in whole]), 3),
                mean_ms=round(sum(deltas) / len(deltas), 4), trimmed_mean_ms=round(trimmed_mean(deltas)[0], 4),
                cost=round(trimmed_mean(deltas)[0] / median_off, 6))
        per_pair.append(entry)
    verdict = 'FAIL' if problems else worst(verdicts) if verdicts else 'NOT_EXERCISED' if shortfalls else 'PASS'
    # REPORTED from here on: nothing below touches problems, shortfalls or verdicts.
    aa = aa_reads(arm_rounds)
    reads = [entry['trimmed_ms'] for entry in aa if 'trimmed_ms' in entry]
    sigma, seen = flag_phase_sigma(reads)
    flag_phase = dict(rule=CONTROL_RULE, informational=CONTROL_INFORMATIONAL, pairs=per_pair, net=net,
                      unmeasured=unmeasured, arm_sigma=dict(floor_ms=CONTROL_ARM_SIGMA_MS,
                                                            seen_ms=None if seen is None else round(seen, 4),
                                                            used_ms=round(sigma, 4), reads_ms=reads), aa=aa,
                      statistic=None, phases=None)
    notices, lines = [], ['G3 rule %s: judged on %s; %s' % (CONTROL_RULE, CONTROL_JUDGED, CONTROL_INFORMATIONAL)]
    lines += net_lines
    lines += ['pair %(pair)d: flag-off %(off_ms)s ms, flag-on %(on_ms)s ms (unaudited): %(ratio)s x (net, at most '
              '%(limit).2f)' % dict(entry, limit=CONTROL_NET_TOLERANCE) for entry in net]
    statistic = None
    if measured:
        statistic = flag_phase_cost([(one['second']['flag_ms'] - one['first']['flag_ms'], one['first']['round_ms'])
                                     for one in measured],
                                    pairs=len([entry for entry in per_pair if entry['measured']]), arm_sigma_ms=sigma)
        flag_phase.update(statistic=statistic, phases=phase_contrast(measured))
        above = [name for name, value in (('c', statistic['cost']), ('the untrimmed mean u', statistic['untrimmed']))
                 if value > CONTROL_PHASE_NOTICE]
        if above:
            notices.append('the flag-phase cost is above %.1f%% of the flag-off median four-live round in %s: c %.3f%% '
                           '(%d%% upper bound %.3f%%), u %.3f%% (bound %.3f%%)%s - informational only, by the user\'s '
                           'decision of 2026-09-28: it does not decide the verdict (exactness and the net do)' % (
                               100 * CONTROL_PHASE_NOTICE, ' and '.join(above), 100 * statistic['cost'],
                               CONTROL_BOUND_PERCENT, 100 * statistic['upper'], 100 * statistic['untrimmed'],
                               100 * statistic['untrimmed_upper'],
                               '' if statistic['cost'] > CONTROL_PHASE_NOTICE else
                               '; u alone is a cost on few rounds or host stalls on the flag-on arms, which the trim '
                               'drops'))
    flag_phase['notices'] = notices
    lines += ['NOTICE (informational, not judged): %s' % text for text in notices]
    for entry in per_pair:
        if entry['measured']:
            lines.append('pair %(pair)d flag phases (informational): %(measured)d of %(rounds)d paired rounds measured, '
                         'F flag-off median %(median_off_flag_ms)s ms, flag-on %(median_on_flag_ms)s ms, D mean '
                         '%(mean_ms)+.3f ms, trimmed %(trimmed_mean_ms)+.3f ms' % entry)
        else:
            lines.append('pair %(pair)d flag phases (informational): unmeasured - none of %(rounds)d paired rounds '
                         'carried every flag-phase line' % entry)
    for entry in unmeasured:
        lines.append('pair %d flag phases (informational): %d of %d paired rounds unmeasured (%s)' % (
            entry['pair'], entry['rounds'], entry['of'], '; '.join(entry['missing'])))
    if statistic:
        lines.append('flag-phase cost (informational, not judged): %d paired four-live rounds of %d pairs, D trimmed mean '
                     '%+.3f ms (%d rounds trimmed from each tail: low %s, high %s), untrimmed %+.3f ms, flag-off median '
                     'round %.2f ms: c %.3f%%, untrimmed %.3f%%; %d%% upper bounds c %.3f%%, untrimmed %.3f%% (rounds '
                     'alone %.3f%% and %.3f%%; the arms\' offsets %.3f%%: sigma %.2f ms - floor %.2f, A/A %s)' % (
                         statistic['rounds'], statistic['pairs'], statistic['trimmed_mean_ms'], statistic['trimmed_low'],
                         statistic['trimmed_low_ms'], statistic['trimmed_high_ms'], statistic['mean_ms'],
                         statistic['median_off_round_ms'], 100 * statistic['cost'], 100 * statistic['untrimmed'],
                         CONTROL_BOUND_PERCENT, 100 * statistic['upper'], 100 * statistic['untrimmed_upper'],
                         100 * statistic['round_upper'], 100 * statistic['untrimmed_round_upper'],
                         100 * statistic['arm_term'], sigma, CONTROL_ARM_SIGMA_MS,
                         'none' if seen is None else '%.2f ms over %d reads' % (seen, len(reads))))
        for name in FLAG_PHASES:
            phase = flag_phase['phases'].get(name) or {}
            lines.append('phase %s (informational): D mean %+.3f ms, trimmed %+.3f ms' % (
                name, phase.get('mean_ms', 0.0), phase.get('trimmed_mean_ms', 0.0)))
    else:
        lines.append('flag-phase cost (informational, not judged): unmeasured - no paired four-live round carried '
                     'every flag-phase line in both arms')
    for entry in aa:
        if 'not_compared' in entry:
            lines.append('A/A %s: not compared (%s) (a noise read, informational)' % (entry['arms'],
                                                                                     entry['not_compared']))
        else:
            lines.append('A/A %s: %d rounds, F trimmed %+.3f ms (%.3f%% of the round), mean %+.3f ms (a noise read, '
                         'informational)' % (entry['arms'], entry['rounds'], entry['trimmed_ms'], 100 * entry['cost'],
                                             entry['phases']['flag_ms']['mean_ms']))
    record = dict(rule=CONTROL_RULE, judged=CONTROL_JUDGED, informational=CONTROL_INFORMATIONAL, net_verdict=verdict,
                  sequence=sequence['fingerprints'] if sequence else None, net=net, pairs=record_pairs,
                  unmeasured=unmeasured, aa=[dict(arms=entry['arms'], state=entry['state'],
                                                  trimmed_ms=entry.get('trimmed_ms')) for entry in aa],
                  arm_sigma=flag_phase['arm_sigma'], statistic=statistic, notices=notices)
    return dict(problems=problems, shortfalls=shortfalls, verdicts=verdicts, pairs=net, flag_phase=flag_phase,
                lines=lines, record=record, notices=notices)


def g4_checks(plan, concurrent, solo, rerun=None, probe=None, seats=MEMORY_USERS):
    """(problems, shortfalls, facts) of a G4 plan beyond the policy and matrix_verdict (module docstring)."""
    problems, shortfalls = [], []
    arms = [('concurrent', concurrent), ('solo', solo)]
    if rerun is not None and None not in rerun:
        arms += [('concurrent re-run', rerun[0]), ('solo re-run', rerun[1])]
    for label, report in arms:
        if report is None:
            continue
        problems += s2_g4_problems(label, report)
        if plan == 'short':
            memory, unjudged = memory_s2_checks(label, report, seats)
            problems += memory
            shortfalls += unjudged
            ladders = s2_of(report).get('ladders') or []
            wrong = sorted(set(str(ladder) for ladder in ladders if ladder != SINGLE_BUCKET))
            if wrong or not ladders:
                problems.append('%s: proposal ladders %s, not the one 2048 bucket every engine builds under %s (W6a)'
                                % (label, wrong or 'unlogged', EXTENT_FLAG))
            # The executed path: the buckets read from each built capture (graft-mounted-is-not-graft-executed).
            built = s2_of(report).get('buckets_built') or []
            wrong = sorted(set(str(buckets) for buckets in built if buckets != SINGLE_BUCKET))
            if wrong:
                problems.append('%s: the engines built proposal buckets %s, not the one 2048 bucket (W6a)'
                                % (label, wrong))
            elif len(built) < len(ladders):
                shortfalls.append('%s: %d "%s" lines for %d engines: the buckets built are unseen for the rest'
                                  % (label, len(built), '[PINDIAG] proposal buckets built', len(ladders)))
    s2 = s2_of(concurrent)
    live = live4_of(concurrent)
    facts = dict(live4=live, rounds=s2.get('rounds'))
    rate = (live or {}).get('net_per_user_tok_s')
    if rate is None and plan == 'staggered':
        # Staggered arrivals (STAGGER_SECONDS apart, EOS on) need not ever have four users live at once on real text
        # (v76, v78: at most three). Its own evidence is the padded path at two or three live and the probe below;
        # the four-live rate is judged when such rounds exist and measured by mixed (M7) regardless.
        by_live = (s2.get('rounds') or {}).get('by_live') or {}
        most = max([int(live_count) for live_count, count in by_live.items() if count] or [0])
        facts['four_live_rate'] = 'not measured: at most %d live (staggered arrivals)' % most
    elif rate is None:
        shortfalls.append('concurrent: no timed four-live packed round: the per-user rate is unmeasured')
    elif rate <= GENERAL_RATE:
        problems.append('concurrent: %.1f tok/s per user at four live (net of the audit), not above general\'s %.1f '
                        '(target %.0f)' % (rate, GENERAL_RATE, TARGET_RATE))
    rounds = s2.get('rounds') or {}
    if plan == 'mixed' and not rounds.get('multi_family_rounds'):
        shortfalls.append('concurrent: no packed round of two or more live families: mixed positions not exercised')
    if plan == 'boundaries':
        users = (s2.get('paths') or {}).get('users') or {}
        if not users:
            shortfalls.append('concurrent: no path records: cap events and boundary crossings unseen')
        for user, entry in sorted(users.items(), key=lambda item: int(item[0])):
            if not entry.get('cap_fields'):
                shortfalls.append('user %s: its [PACKED] lines carry no cap= (W4): cap events unseen' % user)
            elif not entry.get('cap_events'):
                shortfalls.append('user %s: no commit capped at a boundary' % user)
            if (entry.get('boundaries') or 0) < BOUNDARY_MIN_CROSSINGS:
                shortfalls.append('user %s crossed %s 256-key boundaries, fewer than %d' % (
                    user, entry.get('boundaries'), BOUNDARY_MIN_CROSSINGS))
    if plan == 'staggered':
        by_live = rounds.get('by_live') or {}
        if not (by_live.get('2') or by_live.get('3')):
            shortfalls.append('concurrent: no packed round at two or three live: the padded path not exercised')
        if probe is None:
            problems.append('probe: the arm left no gate report')
        else:
            problems += arm_problems('probe', probe)
            problems += s2_g4_problems('probe', probe)
            markers = probe.get('flag_markers') or {}
            problems += ['probe: %s' % line for line in markers.get('missing') or []
                         if line.startswith('QWEN_FAST_PADDED_PROBE')]
            if not markers.get('padded_probe'):
                shortfalls.append('probe: no padded probe line')
    return problems, shortfalls, facts


def with_checks(result, problems, shortfalls, facts=None):
    """A verdict with a plan's own problems (FAIL) and shortfalls (NOT_EXERCISED where it would pass)."""
    if problems:
        result['verdict'] = 'FAIL'
    elif shortfalls and result.get('verdict') == 'PASS':
        result['verdict'] = 'NOT_EXERCISED'
    result['s2_problems'], result['shortfalls'] = problems, shortfalls
    if facts:
        result['facts'] = facts
    result['lines'] = list(result.get('lines') or []) + ['problem: %s' % p for p in problems] + [
        'not exercised: %s' % s for s in shortfalls]
    return result


def run_matrix(plan, runner, arms, checks=None):
    """A concurrent arm against its solo reference under the policy, one re-run on a first divergence (matrix,
    and the G4 plans with `checks`)."""
    by_role = dict((getattr(spec, 'role', None) or ('concurrent', 'solo')[index], spec)
                   for index, spec in enumerate(arms[:2]))
    by_role.update((spec.role, spec) for spec in arms[2:])
    c_spec, s_spec = by_role['concurrent'], by_role['solo']
    wants = dict(concurrent=asked(c_spec[1]), solo=asked(s_spec[1]))
    relaxation = runner.relaxation(spec_profile(runner, c_spec))
    concurrent, solo = run_arm(runner, plan, c_spec), run_arm(runner, plan, s_spec)
    result = matrix_verdict(concurrent, solo, wants=wants, relaxation=relaxation)
    rerun = None
    if result['verdict'] == 'RERUN':
        runner.log('[C2-GATE] %s: a first divergence - re-running both arms once (the exactness policy)' % plan)
        rerun = (run_arm(runner, plan, c_spec, '-rerun'), run_arm(runner, plan, s_spec, '-rerun'))
        result = matrix_verdict(concurrent, solo, rerun if None not in rerun else None, wants=wants,
                                relaxation=relaxation)
        if None in rerun:
            result.update(verdict='FAIL', reason='a re-run arm left no gate report')
    if checks is None and runner.s2_for(spec_profile(runner, c_spec)):
        checks = 'matrix'
    if checks is not None:
        probe = run_arm(runner, plan, by_role['probe']) if 'probe' in by_role else None
        if concurrent is not None and solo is not None:
            problems, shortfalls, facts = g4_checks(checks, concurrent, solo, rerun, probe,
                                                    seats=runner.seats_for(spec_profile(runner, c_spec)))
            result = with_checks(result, problems, shortfalls, facts)
    return result


def live_shortfalls(spec, report, suffix=''):
    """An event arm's drops that fired at another live count than the arm expects (Arm expect_live: {user: live
    streams when that user's drop fires}); a drop that never fired is event_shortfalls'."""
    expected = (getattr(spec, 'extra', None) or {}).get('expect_live') or {}
    if not expected or report is None:
        return []
    events = ((report or {}).get('lifecycle') or {}).get('events') or {}
    shortfalls = []
    for user, live in sorted(expected.items()):
        event = events.get(str(user), events.get(user)) or {}
        if event.get('fired') and event.get('live') != live:
            shortfalls.append('%s%s: user %s\'s %s fired at %s live streams, not %d: the arrival that makes the live '
                              'count two was not the one dropped (VR4:150)' % (spec[0], suffix, user, event.get('spec'),
                                                                               event.get('live'), live))
    return shortfalls


def run_lifecycle(plan, runner, arms):
    """The event arms against the last (solo) arm, each divergent one re-run with the solo once; on an S2
    profile also every refuse_round request ended FINISHED_ABORTED (W5b) and, for lifecycle itself, at least
    one narrowed survivor across the event arms (D1 exercised, M6)."""
    reports = [(spec, run_arm(runner, plan, spec)) for spec in arms]
    s_spec, solo = reports[-1]
    solo_want = asked(s_spec[1])
    relaxation = runner.relaxation(spec_profile(runner, s_spec))
    arms_results = dict((spec[0], lifecycle_verdict(report, solo, want=asked(spec[1]), solo_want=solo_want,
                                                    relaxation=relaxation))
                        for spec, report in reports[:-1])
    again = [spec for spec, _ in reports[:-1] if arms_results[spec[0]]['verdict'] == 'RERUN']
    reruns = {}
    if again:
        runner.log('[C2-GATE] %s: a first divergence - re-running %s and the solo arm once (the exactness '
                   'policy)' % (plan, ', '.join(spec[0] for spec in again)))
        solo_again = run_arm(runner, plan, s_spec, '-rerun')
        first = dict((spec[0], report) for spec, report in reports)
        for spec in again:
            event_again = reruns[spec[0]] = run_arm(runner, plan, spec, '-rerun')
            rerun = (event_again, solo_again) if event_again is not None and solo_again is not None else None
            arms_results[spec[0]] = lifecycle_verdict(first[spec[0]], solo, rerun, want=asked(spec[1]),
                                                      solo_want=solo_want, relaxation=relaxation)
            if rerun is None:
                arms_results[spec[0]].update(verdict='FAIL', reason='a re-run arm left no gate report')
    result = dict(verdict=worst([r['verdict'] for r in arms_results.values()]), arms=arms_results,
                  lines=['%s: %s' % (arm, line) for arm, r in arms_results.items() for line in r['lines']] +
                  ['%s %s' % (arm, r['verdict']) for arm, r in arms_results.items()])
    if runner.s2_for(spec_profile(runner, s_spec)):
        problems, shortfalls = [], []
        for spec, report in reports[:-1]:
            refused = s2_of(report).get('refused') or {}
            if refused and not refused.get('accounted', True):
                problems.append('%s: %d refuse_round rounds, but %d complete FINISHED_ABORTED groups: a refused '
                                'request was left to kill the engine (W5b)' % (spec[0], refused.get('refused'),
                                                                               refused.get('complete_groups')))
        for spec, report in reports[:-1]:
            shortfalls += live_shortfalls(spec, report) + live_shortfalls(spec, reruns.get(spec[0]), ' re-run')
        narrowed = sum(s2_of(report).get('narrowed') or 0 for _, report in reports[:-1])
        if plan == 'lifecycle' and not narrowed:
            shortfalls.append('no "%s" line across the event arms: D1 was not exercised (M6: re-run)'
                              % harness.NARROWED_MARKER)
        result = with_checks(result, problems, shortfalls, dict(narrowed=narrowed))
    return result


def run_memory(plan, runner, arms):
    """G5: each arm judged alone (memory_verdict), the plan the worst of them; on an S2 profile with
    memory_s2_checks."""
    results = []
    for spec in arms:
        report = run_arm(runner, plan, spec)
        one = memory_verdict(report, want=asked(spec[1]))
        if report is not None and runner.s2_for(spec_profile(runner, spec)):
            problems, shortfalls = memory_s2_checks(spec[0], report, runner.seats_for(spec_profile(runner, spec)))
            one = with_checks(one, problems, shortfalls,
                              dict(before=s2_of(report).get('before'), dram_hold=s2_of(report).get('dram_hold')))
        one['any_request_engines'] = (runner.arms.get(spec[0]) or {}).get('any_request_engines') or []
        results.append((spec[0], one))
    if len(results) == 1:
        return results[0][1]
    # memory-short on a C2-any profile: each arm judged alone, the plan the worst of them; every per-request
    # engine's proposal ladder recorded with its arm (R3).
    return dict(verdict=worst([one['verdict'] for _, one in results]), arms=dict(results),
                lines=['%s: %s' % (arm, line) for arm, one in results
                       for line in (one.get('lines') or []) + one['any_request_engines']] +
                ['%s %s' % (arm, one['verdict']) for arm, one in results])


def corpus_first(plan, runner, reference):
    """The image's corpus against the reference's before any card is opened: the mismatch, or None."""
    try:
        return corpus_mismatch(runner.corpus(), reference)
    except Exception as error:
        runner.log('[C2-GATE] %s WARNING: the image\'s corpus could not be read first (%s)' % (plan, error))
        return None


def run_reference_pairs(plan, runner, reference, arms):
    """control (G3) and forced-cap (Q5): every arm against v235 (bringup_verdict); control's flag-on timing arms
    (unaudited) timed against the flag-off arm of their pair, and its audited arm (CONTROL_AUDIT_ARM) judged for
    exactness and the audit only; forced-cap's arms against each other and its caps."""
    mismatch = corpus_first(plan, runner, reference)
    if mismatch:
        return dict(verdict='NOT_COMPARABLE', reason=mismatch, lines=['corpus: %s' % mismatch])
    reports, results = {}, {}
    for spec in arms:
        report = reports[spec[0]] = run_arm(runner, plan, spec)
        verdict = bringup_verdict(report, reference, spec.profile, asked(spec[1]))
        if spec.role in ('on', 'audit') and report is not None and not (s2_of(report).get('rounds') or {}).get('count'):
            verdict.update(verdict='FAIL')
            verdict['lines'] = list(verdict.get('lines') or []) + ['problem: the flag-on arm served no packed extent round']
        results[spec[0]] = verdict
    problems, shortfalls, pairs, audited, timing = [], [], [], None, None
    if plan == 'control':
        timing = control_timing(runner, arms, reports)
        problems, shortfalls, pairs = list(timing['problems']), list(timing['shortfalls']), timing['pairs']
        audit_report = reports.get(CONTROL_AUDIT_ARM)
        if audit_report is not None and not s2_of(audit_report).get('audit'):
            problems.append('%s: the extent audit did not run (the harness saw no %s=1): exactness under the audit '
                            'is unjudged' % (CONTROL_AUDIT_ARM, harness.EXTENT_AUDIT_FLAG))
        live = live4_of(audit_report)
        if live is not None:
            # Recorded, never judged: the audit's reads lengthen more than the ms its line reports.
            audited = dict(arm=CONTROL_AUDIT_ARM, median_round_ms=live['median_round_ms'],
                           median_audit_ms=live.get('median_audit_ms'))
    else:
        off, on = reports.get('forced-cap-off'), reports.get('forced-cap-on')
        if off is not None and on is not None:
            compared = real_text_compare.policy_pass(on, off)
            for user in compared['users']:
                if user['verdict'] != 'IDENTICAL':
                    problems.append('user %s: the capped arm is %s against the uncapped at character %s: commit '
                                    'granularity changed the text (Q5)' % (user['user'], user['verdict'],
                                                                          user.get('first_divergence')))
            caps = (s2_of(on).get('paths') or {}).get('users') or {}
            seen = [entry for entry in caps.values() if entry.get('cap_fields')]
            if not seen:
                shortfalls.append('the capped arm\'s [PACKED] lines carry no cap= (W4): the forced cap is unseen')
            for user, entry in sorted(caps.items(), key=lambda item: int(item[0])):
                if entry.get('max_cap') is not None and entry['max_cap'] > FORCE_CAP:
                    problems.append('user %s: a packed commit capped at %d under %s=%d' % (
                        user, entry['max_cap'], harness.FORCE_CAP_FLAG, FORCE_CAP))
    verdicts = [result['verdict'] for result in results.values()] + (timing['verdicts'] if timing else [])
    result = dict(verdict=worst(verdicts), arms=results, pairs=pairs,
                  lines=['%s: %s' % (arm, line) for arm, one in results.items() for line in one.get('lines') or []] +
                  ['%s %s' % (arm, one['verdict']) for arm, one in results.items()] +
                  (timing['lines'] if timing else []) +
                  (['%(arm)s: median four-live round %(median_round_ms)s ms, audit %(median_audit_ms)s ms (recorded, '
                    'not timed)' % audited] if audited else []))
    if audited:
        result['audited'] = audited
    if timing:
        # The phase record (CONTROL_PHASE_RECORD) is written by main with the plan's final verdict, never kept here.
        # The rule and its informational flag-phase cost are stated in the result itself, with any NOTICE.
        result.update(rule=CONTROL_RULE, judged=CONTROL_JUDGED, informational=CONTROL_INFORMATIONAL,
                      flag_phase=timing['flag_phase'], phase_record=timing['record'], notices=timing['notices'])
    return with_checks(result, problems, shortfalls)


def below_pair(family, off_spec, off, on_spec, on):
    """One G3b pair at family F: both arms served as asked, the flag-off block served packed rounds inside F
    (else NO_DECISION, Q22), the flag-on arm served packed extent rounds, and every user IDENTICAL across them."""
    problems, shortfalls = [], []
    if off is None or on is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    problems += arm_problems('flag-off', off, asked(off_spec[1])) + arm_problems('flag-on', on, asked(on_spec[1]))
    low, high = family - 256, family - 16
    packed, outside = 0, []
    for user, entry in sorted(((s2_of(off).get('paths') or {}).get('users') or {}).items()):
        for record in real_text_compare.decode_paths(entry.get('encoded')):
            if record['path'] == 'P':
                packed += 1
                if record['position'] is not None and not low <= record['position'] <= high:
                    outside.append('user %s round %d at %d' % (user, record['round'], record['position']))
    if outside:
        problems.append('flag-off: packed rounds outside family %d (%s)' % (family, '; '.join(outside[:3])))
    if not (s2_of(on).get('rounds') or {}).get('count'):
        shortfalls.append('flag-on: no packed extent round')
    compared = real_text_compare.policy_pass(on, off)
    texts = [user['verdict'] for user in compared['users']]
    for user in compared['users']:
        if user['verdict'] != 'IDENTICAL':
            problems.append('user %s: flag-on is %s against flag-off at character %s (%s)' % (
                user['user'], user['verdict'], user.get('first_divergence'), user.get('reason') or user.get('detail')))
    if problems:
        verdict = 'FAIL'
    elif not packed:
        verdict = 'NO_DECISION'
        shortfalls.append('flag-off: the block served no packed round at family %d (Q22): G3b decides nothing here; '
                          'fall back to CB2b R2 and the audit' % family)
    elif shortfalls:
        verdict = 'NOT_EXERCISED'
    else:
        verdict = 'PASS'
    return dict(verdict=verdict, family=family, users=texts, flag_off_packed_rounds=packed, problems=problems,
                shortfalls=shortfalls, lines=['problem: %s' % p for p in problems] + [
                    'not exercised: %s' % s for s in shortfalls] + ['users %s; flag-off packed rounds %d' % (texts, packed)])


def run_below(plan, runner, arms):
    """control-below (G3b): each family's pairs in order, each pair judged alone (below_pair)."""
    reports = dict((spec[0], (spec, run_arm(runner, plan, spec))) for spec in arms)
    pairs = {}
    for spec in arms:
        if spec.role != 'off':
            continue
        family, index = spec.extra['family'], spec.extra['pair']
        off_spec, off = reports['below-%d-off-%d' % (family, index)]
        on_spec, on = reports['below-%d-on-%d' % (family, index)]
        pairs['%d-%d' % (family, index)] = below_pair(family, off_spec, off, on_spec, on)
    return dict(verdict=worst([pair['verdict'] for pair in pairs.values()]), pairs=pairs,
                lines=['family %s: %s' % (key, line) for key, pair in pairs.items() for line in pair['lines']] +
                ['family %s %s' % (key, pair['verdict']) for key, pair in pairs.items()])


def run_warm(plan, runner, arms):
    """M1: every arm must complete as asked; the kernel cache's growth is recorded, never judged. On a parked profile
    (G-E0) every arm also shows the parked attach (4 of 4, the 2048 warm and its prewarm: Runner.run's parked check),
    and the result records the attach, the P7p reading, R's measured lower bound, the single's footprint and the
    ballast advice (parked_g_e0)."""
    results = {}
    parked_reports = {}
    for spec in arms:
        report = run_arm(runner, plan, spec)
        problems = arm_problems(spec[0], report, asked(spec[1])) if report is not None else ['%s: no gate report' % spec[0]]
        results[spec[0]] = dict(verdict='FAIL' if problems else 'PASS', problems=problems,
                                kernel_cache=(runner.arms.get(spec[0]) or {}).get('kernel_cache'))
        if report is not None and parked_of(report).get('on'):
            parked_reports[spec[0]] = report
    lines = (['%s: problem: %s' % (arm, p) for arm, one in results.items() for p in one['problems']] +
             ['%s %s kernel cache %s' % (arm, one['verdict'], one['kernel_cache']) for arm, one in results.items()])
    result = dict(verdict=worst([one['verdict'] for one in results.values()]), arms=results, lines=lines)
    if parked_reports:
        problems, shortfalls, facts, more = parked_g_e0(parked_reports)
        result['lines'] = lines + more
        result = with_checks(result, problems, shortfalls, facts)
    return result


def parked_g_e0(reports):
    """G-E0 (design section 9) over the parked warm arms' reports: (problems, shortfalls, facts, lines). The attach
    built 4 of 4 at every arm (Runner.run's parked check judges it), the gate twin's at 131,072; P7p is recorded per
    chip; R is the rebind's measured lower bound (the audit's peak reading, and the ledger's op=rebind bracket) with
    the design's estimate beside it; the single's trace and DRAM footprints come from its rebuild lines; the ballast
    advice sizes parked-ballast's job key."""
    import serving_prefill_admission as admission

    problems, shortfalls, lines, facts = [], [], [], {}
    for name, report in sorted(reports.items()):
        brief = parked_facts_of(report)
        built = (brief.get('built') or [{}])[-1]
        p7p = brief.get('p7p') or []
        facts[name] = dict(attach_ms=built.get('attach_ms'), k=built.get('k'), free=built.get('free'),
                           largest_free=built.get('largest_free'), trace_used=built.get('trace_used'),
                           p7p=p7p, peak_held_max=brief.get('peak_held_max'), rebinds=brief.get('rebinds'),
                           single=brief.get('single_rebuilt'))
        lines.append('%s: parked attach k=%s attach_ms=%s free=%s largest_free=%s trace_used=%s; P7p %s; R measured '
                     '(caller-held, lower bound) %s B over %s rebinds, design estimate %d B; single rebuilt %d times %s' % (
                         name, built.get('k'), built.get('attach_ms'), built.get('free'), built.get('largest_free'),
                         built.get('trace_used'),
                         '; '.join('chip%d free %.3f GB largest %.1f MB' % (entry['chip'], entry['free_gb'],
                                                                             entry['largest_free_mb'])
                                   for entry in p7p[:2]) or 'not read', brief.get('peak_held_max'), brief.get('peaks'),
                         admission.PARKED_REBIND_BYTES, brief.get('single_rebuilt_count') or 0,
                         [(entry['trace_delta'], entry['dram_delta']) for entry in brief.get('single_rebuilt') or []]))
        # M1: the admission's constants are what the hold and the backstop are sized by; a reading above one fails.
        if brief.get('peak_held_max') is not None and brief['peak_held_max'] > admission.PARKED_REBIND_BYTES:
            problems.append('%s: the rebind held %d B at its peak, above the admission constant R = %d B' % (
                name, brief['peak_held_max'], admission.PARKED_REBIND_BYTES))
        for entry in brief.get('single_rebuilt') or []:
            if (entry.get('dram_delta') or 0) > admission.MEASURED_SINGLE_CAPTURE_BYTES:
                problems.append('%s: the single rebuild took %d B of DRAM, above the admission constant %d B' % (
                    name, entry['dram_delta'], admission.MEASURED_SINGLE_CAPTURE_BYTES))
            if (entry.get('trace_delta') or 0) > admission.SINGLE_TRACE_BYTES:
                problems.append('%s: the single rebuild took %d B of trace region, above the admission constant %d B' % (
                    name, entry['trace_delta'], admission.SINGLE_TRACE_BYTES))
        if not p7p:
            shortfalls.append('%s: no "[MEMLEDGER] phase=P7p" line (QWEN_FAST_MEMORY_LEDGER=1): the parked baseline is '
                              'unrecorded' % name)
        if brief.get('peak_held_max') is None and name != 'warm-solo':
            shortfalls.append('%s: no "%s" line (QWEN_FAST_PARKED_AUDIT=1): R is unmeasured' % (name, parked_markers.PEAK.strip()))
    gate = [name for name in reports if '4x%d' % BRINGUP_PROMPT in name]
    frees = [entry['free_gb'] for name in gate for entry in facts[name]['p7p']]
    if frees:
        advice = parked_judge.ballast_advice(int(min(frees) * 1e9))
        facts['ballast_advice_bytes'] = advice
        lines.append('ballast advice: C2_GATE_BALLAST_MB=%d (the smallest P7p free reading %.3f GB per chip less the '
                     'parked need - transient, R, reserve - and the stranded bytes)' % (advice // 2 ** 20, min(frees)))
    return problems, shortfalls, facts, lines


def run_parked_plan(plan, runner, profiles, arms):
    """The parked plans (the module docstring's PARKED PLANS)."""
    return dict(zip(PARKED_PLANS, (run_parked_exact, run_parked_drafter, run_parked_lifecycle, run_parked_churn,
                                   run_parked_corner, run_parked_ballast)))[plan](plan, runner, profiles, arms)


def run_parked_arms(plan, runner, arms):
    """Every arm in order: {name: report}, and the problems each arm's own judgement adds (arm_problems against what it
    asked, and the S2 arms' audit and idle-commit checks)."""
    reports, problems = {}, []
    for spec in arms:
        report = reports[spec[0]] = run_arm(runner, plan, spec)
        if report is None:
            problems.append('%s: the arm left no gate report' % spec[0])
            continue
        problems += arm_problems(spec[0], report, asked(spec[1])) + s2_g4_problems(spec[0], report)
    return reports, problems


def compare_arms(label, reference, other, problems):
    """Every user of `other` against `reference` under the strict policy; the verdicts (or None, a report missing)."""
    if reference is None or other is None:
        return None
    verdicts, differences = parked_judge.token_verdicts(reference, other)
    problems += ['%s: %s' % (label, item) for item in differences]
    return verdicts


def parked_rebind_problems(name, report, streams, problems, shortfalls):
    """A parked arm rebinds once per request that reached the engine: its rebind lines equal its streams (a request the
    lifecycle ended before its engine has none)."""
    brief = parked_facts_of(report)
    if brief.get('rebinds') is None:
        return
    if brief['rebinds'] != streams:
        shortfalls.append('%s: %d rebind lines for %d streams' % (name, brief['rebinds'], streams))
    if brief.get('digests_unequal'):
        problems.append('%s: %d rebind digest lines report unequal state' % (name, brief['digests_unequal']))
    if brief.get('peaks') and brief.get('digests') != brief['peaks']:
        shortfalls.append('%s: %s rebind digest lines for %s audited rebinds (QWEN_FAST_PARKED_AUDIT=1): the rebound state '
                          'is unchecked' % (name, brief.get('digests'), brief['peaks']))


def run_parked_exact(plan, runner, profiles, arms):
    """G-E1 (a), (f) and (g)'s solo half."""
    reports, problems = run_parked_arms(plan, runner, arms)
    shortfalls, lines, facts = [], [], {}
    reference = reports.get('exact-f-desc')
    wanted = asked(next(spec for spec in arms if spec[0] == 'exact-f-desc')[1])
    users = wanted['streams']
    # A request with a budget of 1 ends at its first token: no bridge, no rebind, no round to record (the 2048 rung).
    first_token = [index for index, budget in enumerate(wanted['budgets']) if budget == 1]
    for spec in arms:
        if spec.role == 'p':
            verdicts = compare_arms('%s against exact-f-desc' % spec[0], reference, reports.get(spec[0]), problems)
            lines.append('%s: users %s' % (spec[0], verdicts))
            facts[spec[0]] = dict(users=verdicts)
            if reference is not None and reports.get(spec[0]) is not None:
                more, unseen, detail = parked_judge.equivalence(reference, reports[spec[0]], users=users,
                                                                skip=first_token)
                problems += ['%s: %s' % (spec[0], item) for item in more]
                shortfalls += ['%s: %s' % (spec[0], item) for item in unseen]
                facts[spec[0]]['drafting_identical'] = detail['solo']['identical']
            parked_rebind_problems(spec[0], reports.get(spec[0]), users - len(first_token), problems, shortfalls)
    first_ref = reports.get('exact-first-f')
    for spec in arms:
        if spec.role == 'first':
            verdicts = compare_arms('%s against exact-first-f' % spec[0], first_ref, reports.get(spec[0]), problems)
            lines.append('%s: the first request after an attach: %s' % (spec[0], verdicts))
            facts[spec[0]] = dict(users=verdicts)
    return parked_result(lines, problems, shortfalls, facts)


def run_parked_drafter(plan, runner, profiles, arms):
    """G-E1 (b), (g) and both negative controls."""
    reports, problems = run_parked_arms(plan, runner, arms)
    shortfalls, lines, facts = [], [], {}
    f_solo, p_solo = reports.get('drafter-f-solo'), reports.get('drafter-p-solo')
    f_live, p_live = reports.get('drafter-f-live'), reports.get('drafter-p-live')
    users = asked(next(spec for spec in arms if spec[0] == 'drafter-f-solo')[1])['streams']
    verdicts = compare_arms('drafter-p-solo against drafter-f-solo', f_solo, p_solo, problems)
    lines.append('parked solo against flag-off solo: users %s' % verdicts)
    verdicts = compare_arms('drafter-p-live against drafter-p-solo', p_solo, p_live, problems)
    lines.append('parked concurrent against parked solo: users %s' % verdicts)
    if None not in (f_solo, p_solo):
        more, unseen, detail = parked_judge.equivalence(f_solo, p_solo, f_live, p_live, users=users)
        problems += more
        shortfalls += unseen
        facts['equivalence'] = detail
        live = detail.get('live')
        lines.append('drafting: solo sequences identical %s; concurrent acceptance %s' % (
            detail['solo']['identical'], 'unmeasured' if live is None else '%(other)s against %(reference)s '
                                                                            '(%(delta)+.4f)' % live))
    for spec in arms:
        if spec.role in ('neg-carry', 'neg-drafter'):
            report = reports.get(spec[0])
            if report is None or f_solo is None:
                continue
            more, detail = parked_judge.negative_verdict(spec.extra['kind'], f_solo, report)
            problems += ['%s: %s' % (spec[0], item) for item in more]
            facts[spec[0]] = detail
            lines.append('%s: users %s%s' % (spec[0], detail['users'], ' - the control did its job' if not more
                                             else ' - the control did NOT do its job'))
    return parked_result(lines, problems, shortfalls, facts)


def run_parked_lifecycle(plan, runner, profiles, arms):
    """G-E2: the lifecycle plan's event arms and solo on the parked profile (run_lifecycle), and the injected park-time
    fault: one unpark with its reason, the slot served by today's builds, its texts equal to the solo arm's."""
    events_and_solo, fault = arms[:-1], arms[-1]
    base = run_lifecycle(plan, runner, events_and_solo)
    report = run_arm(runner, plan, fault)
    solo = arm_report(runner, 'parked-lifecycle-solo')
    problems, shortfalls, lines, facts = [], [], [], {}
    if report is None:
        problems.append('%s: the arm left no gate report' % fault[0])
    else:
        problems += arm_problems(fault[0], report, asked(fault[1])) + s2_g4_problems(fault[0], report)
        brief = parked_facts_of(report)
        unparked = brief.get('unparked') or []
        if len(unparked) != 1 or unparked[0].get('reason') != 'injected park fault (gate only)':
            problems.append('%s: the injected fault should leave exactly one unpark with its reason, the log shows %s'
                            % (fault[0], unparked))
        if not brief.get('reparked'):
            shortfalls.append('%s: the slot was never re-parked at an idle moment (the lifecycle\'s idle callback fires '
                              'only on a step that schedules nothing): the rest of the arm ran on per-request builds'
                              % fault[0])
        facts['fault'] = dict(unparked=unparked, reparked=brief.get('reparked'), rebinds=brief.get('rebinds'))
        if solo is not None:
            verdicts = compare_arms('%s against parked-lifecycle-solo' % fault[0], solo, report, problems)
            lines.append('%s: users %s; unparks %d, re-parks %s, rebinds %s' % (fault[0], verdicts, len(unparked),
                                                                               brief.get('reparked'), brief.get('rebinds')))
        else:
            shortfalls.append('parked-lifecycle-solo left no report to hold the fault arm\'s texts against')
    result = dict(base)
    result['verdict'] = worst([base['verdict'], 'PASS'])
    return with_checks(result, problems, shortfalls, facts)


def run_parked_churn(plan, runner, profiles, arms):
    """G-E3 (i): each arm's floors, holds and refusals (memory_s2_checks over the rebind before-points), its alive check,
    its idle return to the attach's allocation, the trace region's spread and its rebinds."""
    problems, shortfalls, lines, facts = [], [], [], {}
    drifts, rebuilt = {}, {}
    for spec in arms:
        report = run_arm(runner, plan, spec)
        if report is None:
            problems.append('%s: the arm left no gate report' % spec[0])
            continue
        want = asked(spec[1])
        seats = profile_seats(profiles, spec_profile(runner, spec))
        memory, unjudged = memory_s2_checks(spec[0], report, seats, required_op='rebind')
        problems += arm_problems(spec[0], report, want) + s2_g4_problems(spec[0], report) + memory
        shortfalls += unjudged
        if not report.get('alive'):
            problems.append('%s: the engine did not answer every seat after the streams' % spec[0])
        drifts[spec[0]] = ledger_drift(report)
        rebuilt[spec[0]] = parked_facts_of(report).get('single_rebuilt_count') or 0
        region = s2_of(report).get('trace_region') or {}
        more, unread = parked_judge.trace_spread(region)
        problems += ['%s: %s' % (spec[0], item) for item in more]
        shortfalls += ['%s: %s' % (spec[0], item) for item in unread]
        parked_rebind_problems(spec[0], report, want['streams'], problems, shortfalls)
        releases = s2_of(report).get('releases') or {}
        facts[spec[0]] = dict(before=s2_of(report).get('before'), dram_hold=s2_of(report).get('dram_hold'),
                              trace_region=region, releases=releases, ledger_drift=ledger_drift(report),
                              rebinds=parked_facts_of(report).get('rebinds'))
        lines.append('%s: %d users; rebinds %s; unparks %d; ledger drift %s; trace region %s; quad departures %s' % (
            spec[0], want['streams'], parked_facts_of(report).get('rebinds'),
            len(parked_facts_of(report).get('unparked') or []), ledger_drift(report), trace_region_text(region),
            releases.get('quad_departures')))
        lines.append(full_seat_admissions_text((s2_of(report).get('dram_hold') or {}).get('fit_readings'), seats))
    more, unread = parked_judge.idle_return_across(drifts, rebuilt)
    problems += more
    shortfalls += unread
    return parked_result(lines, problems, shortfalls, facts)


def run_parked_corner(plan, runner, profiles, arms):
    """G-E3 (ii): the worst corner on the twin and the parked profile. Recorded against the design's 5.3 (the hold or the
    fit, the floors); only an engine death, a failed arm or a differing text fails it."""
    reports, problems = run_parked_arms(plan, runner, arms)
    shortfalls, lines, facts = [], [], {}
    for name in ('corner-f', 'corner-p'):
        report = reports.get(name)
        if report is None:
            continue
        s2 = s2_of(report)
        hold = s2.get('dram_hold') or {}
        releases = s2.get('releases') or {}
        facts[name] = dict(dram_hold=hold, before=s2.get('before'), releases=releases,
                           lifecycle=(report.get('lifecycle') or {}).get('events'))
        lines.append('%s: DRAM holds %s (decodes %s), lifted %s, refused %s; floor %s GB; quad departures %s, released '
                     'quad lines %s' % (name, hold.get('holds'), hold.get('hold_decodes'), hold.get('lifted'),
                                         s2.get('quarantined'), (s2.get('before') or {}).get('floor_gb'),
                                         releases.get('quad_departures'), releases.get('quad')))
        if not releases.get('quad_departures'):
            shortfalls.append('%s: no request left while a quad was formed: the corner (a quad retiring as a seat '
                              'frees) was not exercised' % name)
        if name == 'corner-p':
            brief = parked_facts_of(report)
            if not brief.get('rebinds_with_single') and not brief.get('single_kept'):
                shortfalls.append('%s: no arrival rebuilt the released single of its slot at its rebind (single_rebuilt=1) and '
                                  'none was kept released: the S term of the parked need was not exercised' % name)
        events = (report.get('lifecycle') or {}).get('events') or {}
        if not any(entry.get('fired') for entry in events.values()):
            shortfalls.append('%s: the drop %s never fired: no seat freed for the arrival' % (name, CORNER_DROP))
    if reports.get('corner-f') is not None and reports.get('corner-p') is not None:
        compared = real_text_compare.lifecycle_pass(reports['corner-p'], reports['corner-f'])
        verdicts = [user['verdict'] for user in compared['users']]
        problems += ['corner-p user %s is %s against the flag-off arm at character %s' % (
            user['user'], user['verdict'], user.get('first_divergence')) for user in compared['users']
            if user['verdict'] != 'IDENTICAL']
        lines.append('corner-p against corner-f: users %s' % verdicts)
    return parked_result(lines, problems, shortfalls, facts)


def run_quickwin_plan(plan, runner, profiles, arms):
    """quickwin-ledger and quickwin-warm: every user of each on arm identical to the off arm's, solo and four concurrent
    (the strict policy: the warm skip is an exactness claim and the ledger switch arithmetic-neutral, real_text_compare),
    with the arms' logs read for what the flag promises (quick_arm_check)."""
    reports, problems = run_parked_arms(plan, runner, arms)
    shortfalls, lines, facts = [], [], {}
    for label in ('solo', 'live'):
        off, on = reports.get('%s-off-%s' % (plan, label)), reports.get('%s-on-%s' % (plan, label))
        verdicts = compare_arms('%s-on-%s against %s-off-%s' % (plan, label, plan, label), off, on, problems)
        lines.append('%s: users %s' % (label, verdicts))
        facts[label] = dict(users=verdicts)
    for spec in arms:
        record = (reports.get(spec[0]) or {}).get('c2_gate_quickwin') or {}
        facts[spec[0]] = record
        lines.append('%s: flag %s; ledger lines %s; warm forwards %s' % (
            spec[0], 'on' if record.get('on') else 'off', record.get('ledger_lines'),
            [(entry['run'], entry['skipped']) for entry in record.get('warm') or []][:8] or 'none logged'))
    return parked_result(lines, problems, shortfalls, facts)


def run_parked_ballast(plan, runner, profiles, arms):
    """G-E3 (iii): the ballast arms. No out-of-memory: both arms complete, every stream as asked, the ballast line seen.
    A hold beside the three decoders is recorded (its fit against the design's 5.3), never failed."""
    reports, problems = run_parked_arms(plan, runner, arms)
    shortfalls, lines, facts = [], [], {}
    for spec in arms:
        report = reports.get(spec[0])
        if report is None:
            continue
        s2 = s2_of(report)
        hold = s2.get('dram_hold') or {}
        ballast = (parked_facts_of(report).get('ballast') or [{}])[0]
        facts[spec[0]] = dict(dram_hold=hold, ballast=ballast, p7p=parked_facts_of(report).get('p7p'),
                              before=s2.get('before'))
        lines.append('%s: ballast %s B in %s buffers; DRAM holds %s (decodes %s), lifted %s, refused %s; floor %s GB' % (
            spec[0], ballast.get('bytes'), ballast.get('buffers'), hold.get('holds'), hold.get('hold_decodes'),
            hold.get('lifted'), s2.get('quarantined'), (s2.get('before') or {}).get('floor_gb')))
        if s2.get('quarantined'):
            problems.append('%s: %d requests refused: the parked need did not cover the arrival' % (
                spec[0], s2['quarantined']))
        brief = parked_facts_of(report)
        if not brief.get('rebinds') or brief.get('peak_held_max') is None:
            shortfalls.append('%s: no arrival rebind and peak line (rebinds %s, peak %s): the ballast did not exercise R' % (
                spec[0], brief.get('rebinds'), brief.get('peak_held_max')))
    return parked_result(lines, problems, shortfalls, facts)



def trace_region_text(region):
    """The trace region's readings as a line: its peak use and smallest free block, or why there are none (Q18)."""
    region = region or {}
    if region.get('readings'):
        return '%d readings, at most %s GB used, smallest largest-free %s GB' % (
            region['readings'], region.get('max_used_gb'), region.get('min_largest_free_gb'))
    if region.get('unavailable'):
        return 'unavailable at %d points: this ttnn exposes no TRACE view (Q18)' % region['unavailable']
    return 'not logged (Q18)'


def run_churn(plan, runner, profiles, arms):
    """M11 (G5 churn): no engine death, no DRAM hold with a seat free, no lifted hold or refusal, the floor, and a
    released quad at every departure while a quad was formed (harness.proposal_releases), over at least
    CHURN_MIN_REPLACEMENTS replacements seen as departures; the trace region's readings recorded (Q18)."""
    spec, = arms
    report = run_arm(runner, plan, spec)
    if report is None:
        return dict(verdict='FAIL', reason='the arm left no gate report', lines=[])
    seats = profile_seats(profiles, spec_profile(runner, spec))
    memory, shortfalls = memory_s2_checks('churn', report, seats)
    problems = arm_problems('churn', report, asked(spec[1])) + memory
    if not report.get('alive'):
        problems.append('churn: the engine did not answer every seat after the streams')
    s2 = s2_of(report)
    releases = s2.get('releases') or {}
    users = asked(spec[1])['streams']
    replacements = users - seats
    if (releases.get('departures') or 0) < replacements:
        shortfalls.append('%s departures seen ([PHASE] finished ids) for %d replacements: the seats did not churn '
                          'as asked' % (releases.get('departures') or 0, replacements))
    if not releases.get('quad_rounds'):
        shortfalls.append('no quad round ran: no departure met a formed quad (W6c unexercised)')
    elif not releases.get('quad_departures'):
        shortfalls.append('%d quad rounds, but no request left while a quad was formed (W6c unexercised)'
                          % releases['quad_rounds'])
    elif releases.get('unreleased'):
        problems.append('%d of %d departures while a quad was formed carry no "[PACKED-PROPOSE] released quad=1" '
                        'line (%s): dead proposal traces were not released at detach (W6c)' % (
                            releases['unreleased'], releases['quad_departures'],
                            '; '.join(releases.get('unreleased_steps') or [])))
    region = s2.get('trace_region') or {}
    facts = dict(releases=releases, replacements=replacements, quads_built=s2.get('quads_built'),
                 before=s2.get('before'), dram_hold=s2.get('dram_hold'), trace_region=region,
                 request_buffers=s2.get('request_buffers'), draft_outputs=s2.get('draft_outputs'))
    result = dict(verdict='PASS', lines=[
        'departures %s (%s while a quad was formed, %s released it) over %d replacements; release lines %s (%s '
        'quads, %s pairs); trace region %s' % (
            releases.get('departures'), releases.get('quad_departures'),
            (releases.get('quad_departures') or 0) - (releases.get('unreleased') or 0), replacements,
            releases.get('lines'), releases.get('quad'), releases.get('pairs'), trace_region_text(region)),
        full_seat_admissions_text((s2.get('dram_hold') or {}).get('fit_readings'), seats),
        request_buffers_text(s2.get('request_buffers')), draft_outputs_text(s2.get('draft_outputs'))])
    return with_checks(result, problems, shortfalls, facts)


def draft_outputs_text(record):
    """S2 v86 (run 36416471352, the churn arm that died on a fresh pair's clobbered outputs): the drafts' pooled head
    outputs as the harness read them (lever_n_m3native_gate.draft_outputs_report), as a line - per kind the builds
    that pooled against the builds seen, the refusals and the rejected readbacks. Recorded here; the harness's own
    problems judge them."""
    if not record:
        return 'draft outputs: not recorded'
    pooled, built = record.get('pooled') or {}, record.get('built') or {}
    return 'draft outputs pooled by %s; %s refused; %s rejected readbacks' % (
        ', '.join('%d %s (%d built)' % (pooled.get(kind) or 0, kind, built.get(kind) or 0)
                  for kind in ('single', 'pair', 'quad')), record.get('refused') or 0, record.get('rejected') or 0)


def full_seat_admissions_text(fits, seats):
    """The split's admissions with every other seat decoding (the replacements M11 churns, gate v79's holds), as a
    line: how many, and the smallest largest DRAM block and trace-region block they were admitted beside, so a
    fragmentation ratchet across the cycles shows (recorded, not judged)."""
    full = [fit for fit in fits or [] if fit.get('decodes') == seats - 1]
    if not full:
        return 'admissions at decodes=%d: none logged' % (seats - 1)

    def smallest(key):
        values = [fit[key] for fit in full if fit.get(key) is not None]
        return '%.4f GB' % min(values) if values else 'unread'

    return 'admissions at decodes=%d: %d, smallest largest block %s, smallest trace-region block %s, smallest free %s' % (
        seats - 1, len(full), smallest('largest_free_gb'), smallest('trace_largest_free_gb'), smallest('free_gb'))


def run_permuted(plan, runner, arms):
    """The permuted arm (A4; required only under D-c(i)): two concurrent arms, the second admitted in reverse;
    the users whose every round took the same path in both (SAME_PATH) must match, and there must be two."""
    forward_spec, reverse_spec = arms
    forward, reverse = run_arm(runner, plan, forward_spec), run_arm(runner, plan, reverse_spec)
    if forward is None or reverse is None:
        return dict(verdict='FAIL', reason='an arm left no gate report', lines=[])
    problems = arm_problems('forward', forward, asked(forward_spec[1])) + arm_problems('reverse', reverse,
                                                                                       asked(reverse_spec[1]))
    shortfalls = []
    same = real_text_compare.same_path_users(forward, reverse)
    if same is None:
        shortfalls.append('an arm carries no path records: SAME_PATH users cannot be counted')
        same = []
    elif len(same) < 2:
        shortfalls.append('%d SAME_PATH users, fewer than two: NOT_EXERCISED (A4)' % len(same))
    streams_f, streams_r = forward.get('streams') or [], reverse.get('streams') or []
    for user in same:
        compared = real_text_compare.compare_user(streams_f[user] if user < len(streams_f) else None,
                                                  streams_r[user] if user < len(streams_r) else None)
        if compared['verdict'] != 'exact':
            problems.append('user %d took the same path in both arms but is %s at character %s: a same-path '
                            'divergence FAILS under any policy' % (user, compared['verdict'],
                                                                   compared.get('first_divergence')))
    result = dict(verdict='PASS', same_path_users=same,
                  lines=['SAME_PATH users %s (required only under decision D-c(i))' % same])
    return with_checks(result, problems, shortfalls)


def run_s2_plan(plan, runner, profiles, reference, arms):
    """The S2 plans (module docstring's S2 PLANS)."""
    if plan in ('control', 'forced-cap'):
        return run_reference_pairs(plan, runner, reference, arms)
    if plan == 'control-below':
        return run_below(plan, runner, arms)
    if plan in WARM_PLANS:
        return run_warm(plan, runner, arms)
    if plan == 'lifecycle-arrival':
        return run_lifecycle(plan, runner, arms)
    if plan == 'churn':
        return run_churn(plan, runner, profiles, arms)
    if plan == 'permuted':
        return run_permuted(plan, runner, arms)
    return run_matrix(plan, runner, arms, checks=plan)


def divergence_records(result):
    """Every divergence record a plan's policies carry (exactness_policy's 'records'), as (where, record)."""
    found = []
    policy = result.get('policy') or {}
    found += [('', record) for record in policy.get('records') or []]
    for name, one in sorted((result.get('arms') or {}).items()):
        found += [(name, record) for record in ((one or {}).get('policy') or {}).get('records') or []]
    return found


def counts_cache(plans, profiles, profile, jit):
    """Whether a run counts the kernel cache before and after every arm: an S2 plan or profile, or an explicit --jit
    judge or record. S1 plans on an S1 profile under --jit auto read nothing more than before W11 (no docker image
    inspect, no find) and their reports and summary gain no kernel-cache keys."""
    return (any(plan in S2_PLANS or plan in PARKED_PLANS or plan in QUICKWIN_PLANS for plan in plans)
            or s2_profile(profiles, profile)
            or jit != 'auto')


def with_unjudged(result, unjudged):
    """A plan's result with what its arms could not judge (Runner.unjudged): NOT_EXERCISED where it would pass."""
    if not unjudged:
        return result
    result['shortfalls'] = list(result.get('shortfalls') or []) + list(unjudged)
    result['lines'] = list(result.get('lines') or []) + ['not exercised: %s' % s for s in unjudged]
    if result.get('verdict') == 'PASS':
        result['verdict'] = 'NOT_EXERCISED'
    return result


def run_plan(plan, runner, profiles, reference=None, lengths=None, max_tokens=c2_serving_job.DEFAULT_MAX_TOKENS,
             memory_prompt=None, s2=None):
    notes = []
    unjudged_from = len(runner.unjudged)
    arms = plan_arms(plan, runner.profile, profiles, lengths, max_tokens, memory_prompt, notes, s2)
    for note in notes:
        runner.log('[C2-GATE] note: %s' % note)
    if plan in S2_PLANS:
        result = run_s2_plan(plan, runner, profiles, reference, arms)
    elif plan in PARKED_PLANS:
        result = run_parked_plan(plan, runner, profiles, arms)
    elif plan in QUICKWIN_PLANS:
        result = run_quickwin_plan(plan, runner, profiles, arms)
    elif plan == 'bringup':
        warning = bringup_warning(profiles, runner.profile)
        if warning:
            runner.log('[C2-GATE] bringup WARNING: %s' % warning)
        spec, = arms
        try:
            mismatch = corpus_mismatch(runner.corpus(), reference)
        except Exception as error:
            mismatch = None
            runner.log('[C2-GATE] bringup WARNING: the image\'s corpus could not be read first (%s)' % error)
        if mismatch:
            result = dict(verdict='NOT_COMPARABLE', reason=mismatch, lines=['corpus: %s' % mismatch])
        else:
            result = bringup_verdict(run_arm(runner, plan, spec), reference, runner.profile, asked(spec[1]))
        if warning:
            result['warning'] = warning
    elif plan == 'memory':
        result = run_memory(plan, runner, arms)
    elif plan == 'lifecycle':
        result = run_lifecycle(plan, runner, arms)
    else:
        result = run_matrix(plan, runner, arms)
    result = with_unjudged(result, runner.unjudged[unjudged_from:])
    if runner.infra is not None:
        result.update(verdict='INFRA', reason=runner.infra)
    if notes:
        result['notes'] = notes
    for line in result.get('lines') or ():
        runner.log('[C2-GATE] %s %s' % (plan, line))
    runner.log('[C2-GATE] %s %s%s' % (plan, result['verdict'], ' (%s)' % result['reason'] if result.get('reason') else ''))
    return result


def serving_pair(root='/dev/tenstorrent/by-id'):
    """Cards M then A, resolved by board id now (minors renumber on every reset)."""
    devices = []
    for board in (CARD_M, CARD_A):
        path = os.path.realpath(os.path.join(root, board))
        if not os.path.exists(path) or path == os.path.join(root, board):
            raise RuntimeError('card %s has no device node under %s' % (board, root))
        devices.append(path)
    return devices


# The card sets a hardware step can open (qwen-c2-serving.yml C2_CARDS): the pair above, or every Blackhole board
# present - the four-card (1, 4) mesh - resolved by board id at the moment of use and never named (card_set.sh is
# the shell's copy of the rule: exactly QUAD_BOARDS by-id links named blackhole-*, on distinct device nodes).
CARD_SETS = ('pair', 'quad')
QUAD_BOARDS = 4
BOARD_ID = re.compile(r'blackhole-[A-Za-z0-9._-]+\Z')


def card_set(root='/dev/tenstorrent/by-id', expect=QUAD_BOARDS, listdir=os.listdir, realpath=os.path.realpath,
             is_device=None):
    """Every Blackhole board present, resolved by board id now: their device nodes in board-id order. Refuses
    (RuntimeError) anything but exactly `expect` boards on distinct nodes."""
    if is_device is None:
        import stat

        def is_device(path):
            try:
                return stat.S_ISCHR(os.stat(path).st_mode)
            except OSError:
                return False
    boards = sorted(name for name in listdir(root) if BOARD_ID.match(name))
    if len(boards) != expect:
        raise RuntimeError('%d Blackhole boards under %s (%s), not %d' % (len(boards), root, ', '.join(boards), expect))
    devices = []
    for board in boards:
        path = realpath(os.path.join(root, board))
        if path == os.path.join(root, board) or not is_device(path):
            raise RuntimeError('board %s has no device node under %s' % (board, root))
        if path in devices:
            raise RuntimeError('two boards resolve to %s' % path)
        devices.append(path)
    return devices


def devices_for(cards):
    """The device nodes a step opens for C2_CARDS `cards`, resolved now."""
    if cards not in CARD_SETS:
        raise ValueError('cards must be one of %s, got %r' % (', '.join(CARD_SETS), cards))
    return serving_pair() if cards == 'pair' else card_set()


def cards_problem(cards, profiles, names):
    """Why the named profiles cannot open card set `cards` (a four-card profile on the pair, or a pair profile on all
    four), or None. Reads each profile's mesh_device as the contract does (serving_c2_contract.mesh_of)."""
    for name in names:
        if not name or name == 'none' or name not in profiles['profiles']:
            continue
        device = profiles['profiles'][name].get('mesh_device')
        if (device != 'P150x4') if cards == 'quad' else (device not in (None, 'P300')):
            return ('profile %s opens %s, but C2_CARDS=%s gives %s' % (
                name, 'the four-card (1, 4) mesh' if device == 'P150x4' else 'the (1, 2) pair', cards,
                'all four cards' if cards == 'quad' else 'cards M and A'))
    return None


def image_corpus(image, checkout):
    """The real-text corpus the image's harness would build (real_text_prompts.build_corpus over the
    image's installed vLLM), read in a throwaway container - no network, no devices, the contract's
    boot off: {root, files, characters, sha256, ...}. v235's prompts came from corpus sha 6cf533fb; an
    image whose vLLM source differs builds other prompts, and every bring-up user reads NOT_COMPARABLE."""
    script = ('import json, sys; sys.path.insert(0, "/bench"); import real_text_prompts as r; '
              'corpus, info = r.build_corpus(r.package_root()); print(json.dumps(info))')
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0', '--mount',
                             'type=bind,src=%s,dst=/bench/real_text_prompts.py,readonly' % os.path.join(
                                 checkout, 'scripts', 'ci', 'real_text_prompts.py'),
                             '--entrypoint', 'python3', image, '-B', '-c', script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if output.returncode:
        raise RuntimeError('corpus read exited %d: %s' % (output.returncode, output.stderr[-400:]))
    return json.loads(output.stdout.decode('utf-8').strip().splitlines()[-1])


def corpus_mismatch(corpus, reference):
    """Why the image's corpus is not the reference's, or None."""
    theirs = ((reference or {}).get('real_text') or {}).get('corpus') or {}
    if not theirs.get('sha256') or corpus.get('sha256') == theirs['sha256']:
        return None
    return ('the image builds its real-text prompts from corpus %s (%s files, %s characters), the reference from %s '
            '(%s files): its prompts are not the reference\'s' % (
                str(corpus.get('sha256'))[:16], corpus.get('files'), corpus.get('characters'),
                theirs['sha256'][:16], theirs.get('files')))


def image_profiles(image):
    """The profiles the image itself carries (not the checkout's: the image may predate it)."""
    output = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'cat', image,
                             IMAGE_PROFILES], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if output.returncode:
        raise RuntimeError('cannot read %s from %s: %s' % (IMAGE_PROFILES, image, output.stderr[-400:]))
    return json.loads(output.stdout.decode('utf-8'))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--plan', default='bringup', help='comma-separated: %s' % ', '.join(c2_serving_job.ALL_GATE_PLANS))
    parser.add_argument('--results', required=True)
    parser.add_argument('--lengths', default=None, help='matrix prompt lengths (default: the G4 ladder, its top '
                                                        'rung lowered to what the profile admits); an S2 plan\'s '
                                                        'set in place of its default')
    parser.add_argument('--max-tokens', type=int, default=c2_serving_job.DEFAULT_MAX_TOKENS)
    parser.add_argument('--memory-prompt', type=int, default=None)
    parser.add_argument('--budget-seconds', type=int, default=None,
                        help='refuse a plan list whose worst case (every arm to its limit, every re-run) is longer')
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--checkout', default=os.path.dirname(os.path.dirname(HERE)))
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--hub', default=HUB)
    parser.add_argument('--dry-run', action='store_true', help='print every arm\'s docker argv and run nothing')
    parser.add_argument('--cards', choices=CARD_SETS, default='pair',
                        help='pair: cards M and A (the default); quad: every Blackhole board present, the four-card '
                             '(1, 4) mesh the TP4 profiles open (qwen-c2-serving.yml C2_CARDS)')
    # S2 (the module docstring's S2 PLANS; c2_serving_job's C2_GATE_* keys).
    parser.add_argument('--pairs', type=int, default=None, help='control and control-below A/B pairs (default %d; '
                                                                'control adds one audited arm after them)'
                                                                % DEFAULT_PAIRS)
    parser.add_argument('--families', default=None, help='control-below families (default %s)'
                                                          % ','.join(str(family) for family in BELOW_FAMILIES))
    parser.add_argument('--jit', choices=JIT_MODES, default='auto',
                        help='whose kernel-cache growth fails an arm: auto (S2 arms and plans), judge (every arm but '
                             'warm), record (none)')
    parser.add_argument('--policy', choices=c2_serving_job.POLICIES, default='strict',
                        help='strict (default), or dc-i: the user\'s decision D-c(i) (needs --policy-decision)')
    parser.add_argument('--policy-decision', default=None, help='where the user\'s D-c decision is recorded')
    parser.add_argument('--audits', choices=c2_serving_job.AUDIT_SETS, default='extent',
                        help='extent (the extent audit on every S2 arm but control\'s timing arms) or all (also '
                             'prestage, pair-mask and fused-commit on the G4 arms)')
    parser.add_argument('--ballast-mb', type=int, default=None,
                        help='parked-ballast: the ballast (QWEN_FAST_GATE_DRAM_BALLAST) in MB per chip, sized from the '
                             'G-E0 warm arm P7p reading (the warm plan prints the advice)')
    parser.add_argument('--salt', choices=c2_serving_job.SALT_MODES, default=None,
                        help='fresh or none: every arm on a prefix-reuse profile, each stream salted under the gate\'s '
                             'own key (fresh) or explicitly unsalted (the module docstring\'s SALTED ARMS)')
    return parser


def salt_refusal(plans, arms_of, profile, profiles):
    """Why --salt cannot run these plans (an arm whose profile has no prefix reuse), or None."""
    for plan in plans:
        for spec in arms_of[plan]:
            served = getattr(spec, 'profile', None) or profile
            if served not in profiles['profiles'] or not prefix_profile(profiles, served):
                return ('--salt needs every arm on a prefix-reuse profile (QWEN_PREFIX_REUSE=1 and the prefix cache '
                        'on): plan %s arm %s serves %s, where a salt does nothing' % (plan, spec[0], served))
    return None


def write_salt_key(path, urandom=os.urandom):
    """A fresh 64-character key the contract reads (stripped, at least 32 bytes), for this run's arms only."""
    with open(path, 'wb') as handle:
        handle.write(binascii.hexlify(urandom(32)) + b'\n')
    return path


def s2_options(options, log):
    """The S2 options as plan_arms takes them, or None (logged) when they are refused."""
    try:
        if options.pairs is not None and not 1 <= options.pairs <= c2_serving_job.MAX_PAIRS:
            raise c2_serving_job.JobError('--pairs must be 1..%d, got %d' % (c2_serving_job.MAX_PAIRS, options.pairs))
        families = [c2_serving_job.below_family('--families', part)
                    for part in c2_serving_job.split_list(options.families)] if options.families else None
    except c2_serving_job.JobError as error:
        log('refused: %s' % error)
        return None
    if options.policy == 'dc-i' and not (options.policy_decision or '').strip():
        log('refused: --policy dc-i needs --policy-decision: only the user\'s D-c decision relaxes the strict policy')
        return None
    if options.ballast_mb is not None and not 1 <= options.ballast_mb <= c2_serving_job.MAX_BALLAST_MB:
        log('refused: --ballast-mb must be 1..%d, got %d' % (c2_serving_job.MAX_BALLAST_MB, options.ballast_mb))
        return None
    return dict(pairs=options.pairs, families=families, audits=options.audits, ballast_mb=options.ballast_mb)


def write_phase_record(results, result):
    """control's phase record (control_timing: every paired round's phases and the flag-phase report, informational)
    to <results>/CONTROL_PHASE_RECORD with the plan's final verdict - a reported artifact, kept out of the summary; its
    path goes in the result instead."""
    record = result.pop('phase_record', None)
    if record is None:
        return None
    record['plan_verdict'] = result.get('verdict')
    path = os.path.join(results, CONTROL_PHASE_RECORD)
    with open(path, 'w') as handle:
        json.dump(record, handle, indent=1)
    result['phase_record'] = path
    return path


def write_records(results, plan, result):
    """Every divergence record of a plan to <results>/<plan>-divergence-records.json, and the S2 exit blockers
    among them: a NOT_COMPARABLE or UNSTABLE user blocks S2 exit until its record is read (s2-design W11)."""
    found = divergence_records(result)
    if not found:
        return []
    path = os.path.join(results, '%s-divergence-records.json' % plan)
    with open(path, 'w') as handle:
        json.dump([dict(record, arm=where) for where, record in found], handle, indent=2)
    return [dict(plan=plan, arm=where, user=record.get('user'), verdict=record.get('verdict'), record=path)
            for where, record in found if record.get('verdict') in ('NOT_COMPARABLE', 'UNSTABLE')]


def main(argv=None, execute=None, devices=None, log=print, containers=None, corpus=None, cache_entries=None):
    options = build_parser().parse_args(argv)
    plans = c2_serving_job.split_list(options.plan)
    unknown = sorted(set(plans) - set(c2_serving_job.ALL_GATE_PLANS))
    if not plans or unknown:
        log('unknown plan(s): %s' % ', '.join(unknown or ['(none)']))
        return 2
    s2 = s2_options(options, log)
    if s2 is None:
        return 2
    lengths = [c2_serving_job.positive_int('--lengths', part) for part in c2_serving_job.split_list(options.lengths)] \
        if options.lengths else None
    if options.profiles:
        with open(options.profiles, encoding='utf-8') as handle:
            profiles = json.load(handle)
    else:
        profiles = image_profiles(options.image)
    if options.profile not in profiles['profiles']:
        log('profile %r is not in the image\'s profiles (%s)' % (options.profile, ', '.join(sorted(profiles['profiles']))))
        return 2
    problem = cards_problem(options.cards, profiles, (options.profile,))
    if problem:
        log('refused: %s' % problem)
        return 2
    # Every plan is checked against the image's profile before any container starts.
    arms_of = {}
    for plan in plans:
        try:
            arms_of[plan] = plan_arms(plan, options.profile, profiles, lengths, options.max_tokens, options.memory_prompt,
                                      s2=s2)
        except PlanError as error:
            log('refused: %s' % error)
            return 2
    if options.salt is not None:
        refusal = salt_refusal(plans, arms_of, options.profile, profiles)
        if refusal:
            log('refused: %s' % refusal)
            return 2
    worst_case = worst_case_seconds(plans, arms_of)
    if options.budget_seconds is not None and worst_case > options.budget_seconds:
        log('refused: plans %s may take %d s (every arm to its limit, every re-run), past the %d s this step and job '
            'leave: run fewer plans per tag' % (','.join(plans), worst_case, options.budget_seconds))
        return 2
    with open(options.reference, encoding='utf-8') as handle:
        reference = json.load(handle)
    os.makedirs(options.results, exist_ok=True)
    salt_key_path = None
    if options.salt == 'fresh':
        salt_key_path = os.path.join(os.path.abspath(options.results), SALT_KEY_FILE)
        if not options.dry_run:
            write_salt_key(salt_key_path)
    if options.dry_run:
        log(json.dumps(dict(worst_case_seconds=worst_case, budget_seconds=options.budget_seconds)))
        for plan in plans:
            for spec in arms_of[plan]:
                arm, args, timeout = spec
                log(json.dumps(dict(arm=arm, timeout=timeout, docker=gate_run(
                    options.image, CONTAINER_PREFIX + arm, getattr(spec, 'profile', None) or options.profile,
                    devices or (['<M>', '<A>'] if options.cards == 'pair' else
                                ['<card %d>' % n for n in range(QUAD_BOARDS)]),
                    options.checkout, os.path.join(options.results, arm), args, options.hub,
                    env=getattr(spec, 'env', ()), salt=options.salt, salt_key_path=salt_key_path))))
        return 0
    cache_dir = None
    s2_run = (any(plan in S2_PLANS or plan in PARKED_PLANS or plan in QUICKWIN_PLANS for plan in plans)
              or s2_profile(profiles, options.profile))
    if cache_entries is None and execute is None and counts_cache(plans, profiles, options.profile, options.jit):
        # The kernel cache the image's arms read and write (B6), counted before and after every arm - only on a run
        # that judges or records it: S1 plans on an S1 profile (--jit auto) read nothing more than before W11.
        cache_dir = image_kernel_cache(options.image, options.hub)
        if cache_dir is not None:
            cache_entries = lambda: count_entries(cache_dir)
        else:
            log('[C2-GATE] note: the image\'s kernel cache (%s) is not on the hub: kernel-cache growth is not counted, '
                'and every arm that judges it leaves its plan NOT_EXERCISED' % KERNEL_CACHE_ENV)
    runner = Runner(options.image, options.profile, options.results, options.checkout,
                    devices if devices is not None else devices_for(options.cards), options.hub, execute, log, containers, corpus,
                    any_request=any_request_profile(profiles, options.profile), profiles=profiles,
                    cache_entries=cache_entries, jit=options.jit, policy=options.policy,
                    decision=options.policy_decision, salt=options.salt, salt_key_path=salt_key_path)
    context, ceiling, room = profile_limits(profiles, options.profile)
    summary = dict(image=options.image, profile=options.profile, plans=plans, context=context,
                   output_ceiling=ceiling, largest_prompt=room, worst_case_seconds=worst_case,
                   budget_seconds=options.budget_seconds, results={}, arms=runner.arms)
    if s2_run:
        summary.update(policy=options.policy, policy_decision=options.policy_decision, jit=options.jit,
                       kernel_cache=cache_dir, s2_exit_blockers=[])
    if options.salt is not None:
        summary['salt'] = options.salt
    if 'control' in plans:
        # G3's rule, stated plainly where a reader of the summary looks first: what is judged, and that the flag-phase
        # cost is informational by the user's decision.
        summary['control_rule'] = dict(rule=CONTROL_RULE, judged=CONTROL_JUDGED, flag_phase_cost=CONTROL_INFORMATIONAL)
    try:
        for plan in plans:
            summary['results'][plan] = run_plan(plan, runner, profiles, reference, lengths, options.max_tokens,
                                                options.memory_prompt, s2=s2)
            write_phase_record(options.results, summary['results'][plan])
            for notice in summary['results'][plan].get('notices') or []:
                # Reported prominently, never a verdict (the module docstring's control).
                summary.setdefault('notices', []).append(dict(plan=plan, notice=notice))
                log('[C2-GATE] %s NOTICE (informational, not judged): %s' % (plan, notice))
            blockers = write_records(options.results, plan, summary['results'][plan])
            if blockers:
                summary.setdefault('s2_exit_blockers', []).extend(blockers)
                for blocker in blockers:
                    log('[C2-GATE] S2 EXIT BLOCKED: %(plan)s %(arm)s user %(user)s is %(verdict)s - its record is '
                        '%(record)s' % blocker)
    finally:
        summary['passed'] = bool(summary['results']) and len(summary['results']) == len(plans) and all(
            result['verdict'] == 'PASS' for result in summary['results'].values())
        summary['infra'] = runner.infra
        with open(os.path.join(options.results, 'c2-gate-summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
    log('C2_GATE profile=%s plans=%s passed=%s%s' % (options.profile, ','.join(plans), summary['passed'],
                                                     ' infra=%s' % runner.infra if runner.infra else ''))
    return 0 if summary['passed'] else 1


def _terminate(signum, frame):
    """SIGTERM (a cancelled or timed-out step) as SystemExit, so every finally - the arm's container
    removal first - runs before the runner's SIGKILL."""
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
