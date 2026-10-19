"""The tp4/sched-1 window pack (scripts/ci/references/fusion-jobs/SCH), generated (stdlib only).

    python3 scripts/ci/make_sched_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_sched_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The scheduler-side levers of docs/tp4-sched-1.md on the production checkpoint: ONE image (tp4-sched-1, a placeholder: B0 builds it from the pushed tp4/sched-1 head WITH the K64j-OQ graft), ONE
window, ONE control (the approved ship-4 profile) and its six gate-only twins (make_sched_profiles.py), the checkpoint production serves (ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys
C2_MODEL_REPO and C2_MODEL_REVISION: qwen_c2_checkpoint.py). The pack answers, in value order: does everything stay exact with all the levers on and every audit on (A1, S1), what do the four
together do to a production-shaped multi-turn run against the approved profile (the ABAB C1 L1 C2 L2 on agent_turns), which lever did it (LE1 lean admission, PA1 the returning-turn pacing, FD0 FD1
the few-decoder policy) and what do complete spills do to a restore beside seven seats (the tier ABAB T0 T1 T2 T3). Nothing here changes the approved profile, its image or its cache; every result is
UNQUALIFIED.

THE MEASUREMENT is the agent_turns smoke test: a production-shaped multi-turn run (six sessions on 64k / 128k / 200k-token contexts, each returning twelve times with a tool result of a few hundred
tokens, a few thousand old tokens re-prefilled, an engine rebind, a few hundred tokens decoded, a tool gap of 1-8 s; about 13 minutes). It lives on the branch tp4/agent-turns (pushed; scripts/ci/agent_turns.py
plans it, reads the container log and judges it; the branch merges into this one without a conflict) and is not in c2_serving_smoke.py of this one: merge it into the window head BEFORE B0, or the timed jobs
name a test the image does not have.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_sched_profiles as twins
import make_ship4_profiles as ship4

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'SCH'
IMAGE = 'tp4-sched-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = REPO
SHIP = ship4.SHIP                                   # the approved ship-4 profile: the control of every timed pair
ARM = {suffix: twins.NAMESPACE + suffix for suffix in ('lean', 'spill', 'pace', 'adaptive', 'all', 'all-audit')}
MEASURE = 'agent_turns'                             # the production-shaped multi-turn test (another branch)
TIMED_TESTS = 'warmup,%s,concurrent8_steady' % MEASURE
AUDIT_TESTS = 'warmup,coding,parked_turns,prefill_ladder_solo,concurrent8_steady'
FEW_TESTS = 'warmup,prefill_few_decoders,concurrent8_steady'          # (the eight-seat rule: the smoke check judges the quad profile on concurrent8_steady alone)

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/sched-1 window (the scheduler-side levers; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/sched-1 (with the branch tp4/agent-turns merged: see ORDER.txt) and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a smoke job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the
# same in every smoke job of this pack, so each pair differs by the profile alone. The prefix-gate jobs (T0-T3, S1) do not forward the checkpoint (c2_serving_job.MODEL_ACTIONS): they run the base
# weights, which the tier's mechanics (what is spilled, restored and counted) do not depend on."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: found" line (the cache the Swift window\'s W0 built; this pack never builds it) and the worker\'s "[QWEN-C2] checkpoint bound ..." '
                   'line; the models answer names %s.' % (REPO, REVISION, SERVED))
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
REPORT = ('READ the arm with python3 scripts/ci/sched_report.py <arm container log> --second <control container log> reversed as needed (B over A): the engine builds of returning turns '
          '(kind=rebind) p50 / p90, the lean split A / B / C, the readbacks skipped, the backstop readings reused, prefill\'s share of the wall time while a prefill is pending, the rounds after each '
          'returning turn, the longest stretch without a decode round, the guard steps and the owed cap. Every figure prints "-" when no line is behind it.')
LEVER_LINES = ('the lines each lever logs (a lever whose line is absent did not engage, and its arm reads NOT_EXERCISED): "[PINDIAG] admission lean split request=... a_ms=... b_ms=... c_ms=... total_ms=..." '
               'once per rebound admission, "[PINDIAG] admission lean rebind phases request=... fence_ms=... rezero_ms=... project_ms=... copy_ms=... history_ms=... engine_ms=..." once per rebind (where the 188 ms go) '
               'and "[PINDIAG] admission lean adopt ..." per slot adoption (QWEN_FAST_ADMISSION_LEAN); "[PINDIAG] lever N sched: owed_cap_ms=1500 short_share=0.65 short_wait_s=5" once and '
               'short=/cap_ms=/guard= on every "lever N step" line (the pacing); "[PINDIAG] lever N adaptive: decoders<=2 share=0.9 ttft=120 gap_floor=8" once and base=/target=/adaptive= on every step line '
               '(the few-decoder policy); "[PINDIAG] prefix: host tier spill complete on cap_blocks=512 hard_max_blocks=4096 batch_blocks=512 warn_ms=1000" once and batches=/lifted= on every spill line '
               '(complete spills)')


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/sched-1 head (= tp4/swift, the ship-4 stack with the checkpoint selection, plus the scheduler-side levers) WITH the branch tp4/agent-turns '
                         'merged: the approved ship-4 profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. Every lever is a gate instrument of the twins and is off in the baked '
                         'profile, so the image serves exactly what production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report with no '
                         'problem line, its K64j-OQ graft line, the image id. A build, not a card job; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: '
                         '35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host (the Swift window\'s W0 built it). Stop rule: '
                         'anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='A1-sch-audited-attach', mode='stop', minutes=120, actions='reset smoke', profile=ARM['all-audit'], tests=AUDIT_TESTS, telemetry=True, box=180, needs=('A1 <- X0',), model=True, extra=[],
                    why=('A1 (EXACTNESS with every lever on, STOP): %s, the ship-4 audited twin (every audit of the candidate and the tier\'s instruments) with the four levers of docs/tp4-sched-1.md on, on Swift: the lean '
                         'admission (the conv-slice readback after a slot\'s first adoption only, no "dram after engine" line for a rebind, the backstop reading reused, no trace view where none applies), the '
                         'returning-turn pacing, the few-decoder policy and complete spills. None of them touches what a round computes: the pacing and the policy decide WHEN a decode round or a prefill step runs, '
                         'the lean admission drops host-side checks around a rebind, the spills move KV bytes the digests compare. warmup; coding (one user); parked_turns (eight seats in a closed loop of turns, 4k-24k '
                         'tokens: every admission is a rebind); prefill_ladder_solo; concurrent8_steady (the octo rounds). READ: every audit line exact=True with entries above 0, NO "audit mismatch", NO "fell back", '
                         'no stall; %s. %s %s RUN WITH THE CI PAUSED. ANY failure is NO-GO for the lever it names and halts the window.'
                         % (ARM['all-audit'], LEVER_LINES, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='S1-sch-tier-audited', mode='stop', minutes=100, actions='reset prefix', profile=None, tests=None, telemetry=False, box=None, needs=('S1 <- A1',), model=False,
                    extra=['C2_PREFIX_PLAN=tier-concurrent', 'C2_PREFIX_PROFILE=%s' % ARM['all-audit'], 'C2_PREFIX_BASELINE=none', 'C2_BOX_MINUTES=150'],
                    why=('S1 (EXACTNESS of the complete spills, STOP): the concurrent tier plan (c2_prefix_gate.py --plan tier-concurrent: seven seats decoding, a 254k session built and evicted by ten 120k floods, then returning '
                         'beside the seats; the restore audited block by block) on %s. The flood is what the cap of 512 blocks a flush cut in the TC1 card run (3,393 of 3,905 displaced blocks dropped when the 254k restore '
                         'took its blocks). READ: the plan verdict PASS; the line "host tier spill complete on cap_blocks=512 hard_max_blocks=4096" in the container log; the restore\'s flush as one spill line with '
                         'blocks above 512 and lifted= above 0 (and the counters tier_spill_cap_lifted above 0, tier_spill_dropped_cap 0); every pair IDENTICAL; zero tier_digest_failures and tier_audit_mismatches; '
                         'no "host tier spill failed" line; the largest flush\'s milliseconds RECORDED (ms a block from sched_report.py: ESTIMATE 0.4 ms once the slab is resident, 1.7 ms while it fills). The '
                         'prefix step does not forward the checkpoint keys: this job runs the base weights. Duration (estimate): 100 minutes.' % ARM['all-audit'])))
    for index, (control, arm) in enumerate((('C1', 'L1'), ('C2', 'L2')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-control-%s' % (control, letter), mode='soft', minutes=50, actions='reset smoke', profile=SHIP, tests=TIMED_TESTS, telemetry=True, box=90,
                        needs=('C1 <- S1',) if index == 1 else ('C2 <- L1',), model=True, extra=[],
                        why=('%s (TIMED, the CONTROL of the %s ABAB pair): the approved ship-4 profile %s on Swift, the stack production runs. Tests: warmup; %s (the production-shaped multi-turn run of the branch tp4/agent-turns: six sessions at 64k-200k tokens of context that return twelve times each, so the returning-turn prefill, the engine rebind '
                             'and the decoders\' gaps are the production ones); concurrent8_steady (eight users: the parity row and the boot-to-boot floor, and the test the smoke check judges the eight-seat profile on). Every boot a fresh '
                             'reset. READ: the agent_turns entry (python3 scripts/ci/agent_turns.py report <container log>): committed tokens a second, TTFT by context, the re-prefill, the per-live-users table, prefill\'s and decode\'s share '
                             'of the run. %s %s %s' % (control, 'first' if index == 1 else 'second', SHIP, MEASURE, ARC, CHECKPOINT_READ, TELEMETRY))))
        out.append(dict(name='%s-all-%s' % (arm, letter), mode='soft', minutes=50, actions='reset smoke', profile=ARM['all'], tests=TIMED_TESTS, telemetry=True, box=90,
                        needs=('L1 <- C1',) if index == 1 else ('L2 <- C2',), model=True, extra=[],
                        why=('%s (TIMED, the ARM of the %s ABAB pair): %s, the approved ship-4 profile with the gate marker in place of its owner waiver id and the four levers on (QWEN_FAST_ADMISSION_LEAN=1, '
                             'QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete, QWEN_FAST_LEVERN_OWED_CAP_MS=1500, QWEN_FAST_LEVERN_SHORT_SHARE=0.65, QWEN_FAST_LEVERN_SHORT_WAIT_S=5, QWEN_FAST_LEVERN_ADAPTIVE=1), the same '
                             'tests as %s, on Swift. %s READ: c2_smoke_check clean; %s. Then, after the four boots of the pair (C1 L1 C2 L2): per user and turn the paired comparison of %s (turn wall time, the '
                             'time to the first token, the longest gap between two tokens of the stream, decode tokens a second; paired by index, never by unpaired medians) and of concurrent8_steady, each arm against '
                             'the two controls and the control-to-control floor; the answers must be EQUAL turn for turn (the same checkpoint, greedy: a difference is a token a lever changed and is NO-GO). %s %s'
                             % (arm, 'first' if index == 1 else 'second', ARM['all'], control, ARC, LEVER_LINES, MEASURE, REPORT, CHECKPOINT_READ))))
    out.append(dict(name='LE1-lean-a', mode='soft', minutes=50, actions='reset smoke', profile=ARM['lean'], tests=TIMED_TESTS, telemetry=True, box=90, needs=('LE1 <- L2',), model=True, extra=[],
                    why=('LE1 (ATTRIBUTION, the admission lever alone): %s, the approved profile with the gate marker and QWEN_FAST_ADMISSION_LEAN=1, the tests of C1, on Swift. ESTIMATE (docs/tp4-sched-1.md section 4): '
                         '541 ms (p50) of a returning turn\'s engine build become about 240 ms (the readback after a slot\'s first adoption only, 30 to 145; the backstop reading reused, about 70; the ledger\'s op=rebind bracket '
                         'after a slot\'s first rebind only, about 127; and no "dram after engine" line, about 66, outside the sticky line\'s own clock). READ: sched_report.py against C1 or C2 (the nearer control): the rebind p50 and p90 and the lean split A / B / C, the readbacks '
                         'skipped (all but the first of each slot), the backstop reuses (every admission with the room), the ledger brackets (one per slot), the "dram after engine" lines (one per cold build only); the answers EQUAL to the control\'s. '
                         '%s %s %s' % (ARM['lean'], ARC, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='PA1-pace-a', mode='soft', minutes=50, actions='reset smoke', profile=ARM['pace'], tests=TIMED_TESTS, telemetry=True, box=90, needs=('PA1 <- LE1',), model=True, extra=[],
                    why=('PA1 (ATTRIBUTION, the returning-turn pacing alone): %s, the approved profile with the gate marker and QWEN_FAST_LEVERN_OWED_CAP_MS=1500, QWEN_FAST_LEVERN_SHORT_SHARE=0.65, '
                         'QWEN_FAST_LEVERN_SHORT_WAIT_S=5, the tests of C1, on Swift. ESTIMATE (the simulation of docs/tp4-sched-1.md section 6, which moves delay and removes none): the decoders\' share after a '
                         'returning turn rises from one round to about seven, the seat rate at five to seven live users by about 9%% and the returning turn\'s TTFT p90 by about 30%% (the guard bounds it). READ: '
                         'sched_report.py: "rounds after a returning turn beside decoders" about 7 at the median, the guard steps, the cap 1500; the smoke check\'s alternation rules clean (a decode round after every short '
                         'step, the 8 s floor); the paired per-turn TTFT and gap against the control. %s %s %s' % (ARM['pace'], ARC, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='FD0-few-control', mode='soft', minutes=50, actions='reset smoke', profile=SHIP, tests=FEW_TESTS, telemetry=True, box=90, needs=('FD0 <- PA1',), model=True, extra=[],
                    why=('FD0 (TIMED, the control of the few-decoder pair): the approved ship-4 profile %s on Swift: prefill_few_decoders (one decoder and a 120k cold prompt, then two decoders and a 254k one: the '
                         'few-decoder policy\'s own shape); concurrent8_steady (the eight-seat rule of the smoke check, and the policy\'s regression row: with eight live it must change nothing). %s %s %s' % (SHIP, ARC, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='FD1-few-adaptive', mode='soft', minutes=50, actions='reset smoke', profile=ARM['adaptive'], tests=FEW_TESTS, telemetry=True, box=90, needs=('FD1 <- FD0',), model=True, extra=[],
                    why=('FD1 (TIMED, the arm of the few-decoder pair): %s, the approved profile with the gate marker and QWEN_FAST_LEVERN_ADAPTIVE=1 (a prefill share of 0.9 and a 120 s deadline while at most two '
                         'decoders are live), the same tests as FD0, on Swift. ESTIMATE (the policy\'s note, docs/lever-n-adaptive-governor.md): the cold prompt\'s TTFT falls 31-44%% (254k: 182-187 s to 114-119 s), '
                         'paid for by the one or two decoders\' rate in that window. READ: python3 scripts/ci/prefill_ladder_report.py <FD0 smoke log> --second <FD1 smoke log>: the TTFT ratio B over A of each shape, '
                         'both arms\' worst decoder gap, the arrival\'s answer hash EQUAL; the step lines say adaptive=1 for exactly the prefill steps with one or two decoders (the smoke check\'s '
                         'levern_adaptive_problems). %s %s %s' % (ARM['adaptive'], ARC, CHECKPOINT_READ, TELEMETRY))))
    for index, (control, arm) in enumerate((('T0', 'T1'), ('T2', 'T3')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-tier-control-%s' % (control, letter), mode='soft', minutes=50, actions='reset prefix', profile=None, tests=None, telemetry=False, box=None,
                        needs=('T0 <- FD1',) if index == 1 else ('T2 <- T1',), model=False,
                        extra=['C2_PREFIX_PLAN=tier-concurrent-timed', 'C2_PREFIX_PROFILE=%s' % SHIP, 'C2_PREFIX_BASELINE=none', 'C2_BOX_MINUTES=90'],
                        why=('%s (TIMED, the CONTROL of the tier ABAB\'s %s pair): the concurrent tier plan on the approved ship-4 profile %s, the cap of 512 blocks a flush in force (c2_prefix_gate.py --plan '
                             'tier-concurrent-timed: no row digests, no extent audit, the tier\'s own audits off). The prefix step does not forward the checkpoint keys (the base weights). %s READ: the plan verdict '
                             'PASS (the stall budget judged: no seat waited longer than the Lever N decode gap floor, 8 s, in any window); the container log\'s spill lines (sched_report.py: ms a block, the largest flush, '
                             'tier_spill_dropped_cap in the counters line) and the returning session\'s TTFT.' % (control, 'first' if index == 1 else 'second', SHIP, ARC))))
        out.append(dict(name='%s-tier-spill-%s' % (arm, letter), mode='soft', minutes=50, actions='reset prefix', profile=None, tests=None, telemetry=False, box=None,
                        needs=('T1 <- T0',) if index == 1 else ('T3 <- T2',), model=False,
                        extra=['C2_PREFIX_PLAN=tier-concurrent-timed', 'C2_PREFIX_PROFILE=%s' % ARM['spill'], 'C2_PREFIX_BASELINE=none', 'C2_BOX_MINUTES=90'],
                        why=('%s (TIMED, the ARM of the tier ABAB\'s %s pair): the same plan on %s, the approved profile with the gate marker and QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete. ESTIMATE: the 254k restore\'s '
                             'flush spills the 3,905 blocks it displaces (8.7 GB; 0.4 ms a block once the slab is resident, 1.7 ms while it fills: 1.6 to 6.6 s on top of the restore\'s 0.96 s, one engine step) '
                             'where the cap spilled 512; the sessions the flood loses to the cap come back warm instead of cold. READ: the plan verdict PASS; then python3 scripts/ci/tier_concurrent_compare.py '
                             '<T0 results> <T1 results> <T2 results> <T3 results> (A control, B arm, C control repeat, D arm repeat): exit 0 needs every seat IDENTICAL over its common tokens and the answers IDENTICAL, '
                             'no longest wait over the budget; it prints the arm against the control with the control-to-control FLOOR. THE COST this lever must show: the seats\' longest wait in the restore '
                             'window grows by the flush (about 1.6 to 6.6 s ESTIMATED): a NO-GO for shipping WITHOUT a sliced restore if it passes the 8 s budget. %s'
                             % (arm, 'first' if index == 1 else 'second', ARM['spill'], ARC))))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, '
                         'link measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    return out


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row['model']:
        lines += ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    lines += row['extra']
    if row['telemetry']:
        lines.append('C2_TELEMETRY=1')
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def render_order(table):
    total = sum(row['minutes'] for row in table)
    names = [row['name'] for row in table]
    cards = total - next(row['minutes'] for row in table if row['name'].startswith('B0'))

    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())

    blocks = (('GATES', 'B0 X0 A1 S1'), ('COMBINED ABAB', 'C1 L1 C2 L2'), ('ATTRIBUTION', 'LE1 PA1 FD0 FD1'), ('TIER ABAB', 'T0 T1 T2 T3'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/sched-1 window (docs/tp4-sched-1.md; scripts/ci/make_sched_jobs.py): the scheduler-side levers on the approved ship-4 profile %s and its six gate-only twins (make_sched_profiles.py), "
        "the checkpoint production serves (%s, revision %s), ONE image (%s, a placeholder: B0 builds it from the pushed tp4/sched-1 head WITH the K64j-OQ graft and the agent_turns branch merged) and ONE window."
        % (SHIP, REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether production should turn on, one by one, the four levers: the lean admission (QWEN_FAST_ADMISSION_LEAN: the part of a returning turn's 541 ms engine build that is "
        "accounting, ESTIMATED -300 ms of the 541 and a further 66 ms outside it), complete spills of the host tier (QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete), the returning-turn pacing (an owed cap in ms, a share for returning turns, its guard) and the few-decoder policy "
        "(QWEN_FAST_LEVERN_ADAPTIVE). Only the first removes device time; the pacing and the policy move delay between a returning turn's wait and the decoders' gaps, and the note's simulation says by how much. Every result is UNQUALIFIED.",
        "# THE MEASUREMENT: the agent_turns smoke test (the production-shaped multi-turn run: six sessions at 64k-200k tokens of context, twelve returning turns each, about 13 minutes; scripts/ci/agent_turns.py reads and judges it) is built on the branch tp4/agent-turns and is NOT in this branch's c2_serving_smoke.py. P0: merge that branch into the head B0 builds (it merges without a conflict), or every timed job below names a test the image does not have. "
        "parked_turns (in this tree) is a closed loop of fresh 4k-24k prompts with no growing context and no prefix hits: it is A1's exactness shape, not the measurement.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
        "first boot) has built the keyed weight cache, so every smoke job here logs \"weight cache ...: found\". This pack builds nothing of it. The prefix-gate jobs (S1, T0-T3) run the base weights." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO A1 T3",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER T3, i.e. around "
        "every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean where it runs and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) every audit line exact=True with entries above 0, no 'audit mismatch', no 'fell back', the lever lines of the four levers (sched_report.py prints them), no stall.",
        "#   S1   (STOP) the tier-concurrent plan PASS on %s (every pair IDENTICAL, zero digest failures and audit mismatches), the 'spill complete on' line, a restore flush above 512 blocks with lifted= above 0, tier_spill_dropped_cap 0." % ARM['all-audit'],
        "#   C1 L1 C2 L2   the agent_turns pairs per turn by index, the answers EQUAL turn for turn, concurrent8_steady within the control-to-control floor; sched_report.py --second on the container logs. GO for a lever needs a GAIN beyond the floor and no LOSS elsewhere.",
        "#   LE1 PA1 FD0 FD1   the attribution of the combined arm: the lean split and the rebind p50 (LE1), the rounds after a returning turn, the guard and the TTFT cost (PA1), prefill_ladder_report.py --second (FD1 over FD0).",
        "#   T0 T1 T2 T3   the tier-concurrent-timed plan PASS each (the stall budget judged), then tier_concurrent_compare.py over the four: IDENTICAL texts, the arm against the control, the floor; the cost of the flush in the seats' longest wait.",
        "# NO-GO for a lever: any text that differs from the control's, an audit mismatch, a stall, a refusal at the attach, an 'engaged' line that is absent, a LOSS in a timed test beyond the floor, and for complete spills a longest wait over the 8 s floor (they then need the sliced restore beside them). "
        "Adopting a lever is the owner's decision: each is a gate instrument today and a ship profile needs an owner traffic waiver naming it (docs/tp4-sched-1.md).",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z)." % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def generate():
    table = rows()
    out = dict(('%s.env' % row['name'], render_env(row)) for row in table)
    out['ORDER.txt'] = render_order(table)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args(argv)
    if not (arguments.write or arguments.check):
        parser.error('--write or --check')
    wanted = generate()
    found = dict((path.name, path.read_text(encoding='utf-8')) for path in FOLDER.glob('*') if path.is_file()) if FOLDER.is_dir() else {}
    stale = sorted(name for name in set(wanted) | set(found) if wanted.get(name) != found.get(name))
    if arguments.check:
        for name in stale:
            sys.stderr.write('%s/%s is not what make_sched_jobs.py generates: run it with --write\n' % (FOLDER, name))
        return 1 if stale else 0
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name in stale:
        if name in wanted:
            with open(str(FOLDER / name), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted[name])
        else:
            (FOLDER / name).unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
