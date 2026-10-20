"""The tp4/s6 window pack (scripts/ci/references/fusion-jobs/S6), generated (stdlib only).

    python3 scripts/ci/make_s6_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_s6_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The ship-6 candidate (docs/tp4-next2.md): the traffic profile `-fx-traffic-s6` (scripts/ci/fusion-wp/S6.json, owner traffic waiver PENDING) = the ship-5 profile that serves production plus W3 WC
(QWEN_FAST_TP4_GDN_WC=1) and the adaptive Lever N governor (QWEN_FAST_LEVERN_ADAPTIVE=1), its audited twin (make_s6_profiles.py) and THE SHIP-5 PROFILE ITSELF as the control of every pair, ONE image
(tp4-s6-1, a placeholder: B0 builds it from the pushed tp4/s6 head WITH the K64j-OQ graft), the checkpoint production serves (selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION:
qwen_c2_checkpoint.py). Templates only: no tag is pushed from the branch.

TWO ORDER FILES share the folder (the window driver reads a fixed ORDER.txt, so the ship flow is run from a commit where its file is ORDER.txt: `git mv ORDER-SHIP.txt ORDER.txt`):

  ORDER.txt        THE WINDOW: B0 build (while production serves), X0, A1 the audited attach of the candidate's twin (STOP; WC and the governor under every audit, with the production-shaped agent_turns),
                   T1A and T1B the timed pair at one, two and four live seats, the governor's own shape (prefill_few_decoders), eight beside them and agent_turns (the main judge), T2A and T2B the
                   production-load pair on agent_turns8 (eight sessions: every seat busy; optional), Z.
  ORDER-SHIP.txt   AFTER THE OWNER APPROVES (and only then): B1 the ship build and SR the platform replay of the candidate, as the ship-5 pack's: B1 is refused by the job reader while the waiver is PENDING.

Every minute below is an ESTIMATE from the ship-5 window's jobs (docs/tp4-next1.md section 7) and the test estimates of agent_turns.py; the card decides.
"""

import argparse
import json
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_s5_jobs as s5jobs
import make_s5_profiles as s5
import make_s6_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'S6'
IMAGE = 'tp4-s6-1'
REPO, REVISION, SERVED, MODEL = s5jobs.REPO, s5jobs.REVISION, s5jobs.SERVED, s5jobs.MODEL
SHIP5 = s5.SHIP                         # the ship-5 profile: production, the control of every pair, and what B0 bakes as the window image's default
SHIP6 = twins.SHIP                      # the candidate (a traffic profile, waiver PENDING)
AUDIT = twins.AUDIT                     # the audited twin of the candidate (every audit, WC with its audit, the governor)
# the audited attach: the live counts WC acts at (2 and 4), the governor's shape (one or two decoders and a long cold prompt), the drain through every live count, the octo rounds, the agent sessions
A1_TESTS = 'warmup,coding,np1_live2,np1_live4,prefill_few_decoders,concurrent8_drain,concurrent8_steady,agent_turns'
# the timed pair: the lone user (NP1) and the live counts WC acts at, the governor's shape, eight beside them, and agent_turns (the main judge; the ship-5 window's T1 numbers are its baseline)
T1_TESTS = 'warmup,np1_live1,np1_live2,np1_live4,prefill_few_decoders,concurrent8_steady,agent_turns'
# the production-load pair: eight sessions of the same plan, every seat busy (WC and the governor decline or sit out at eight live; the pair says whether anything is LOST there)
T2_TESTS = 'warmup,concurrent8_steady,agent_turns8'
MINUTES = {'B0': 40, 'X0': 15, 'A1': 115, 'T1A': 60, 'T1B': 60, 'T2A': 40, 'T2B': 40, 'Z': 5, 'B1': 35, 'SR': 25}
BOX = {'A1': 190, 'T1A': 115, 'T1B': 115, 'T2A': 80, 'T2B': 80}
ORDERS = ('ORDER.txt', 'ORDER-SHIP.txt')
# the estimates of the tests that are not in the ship-5 window's jobs, for the order's header (minutes inside a boot; agent_turns and agent_turns8 from agent_turns.estimate, the fewer-decoders shape from the 120k and 254k cold prefills)
TEST_MINUTES = (('agent_turns', 15), ('agent_turns8', 20), ('prefill_few_decoders', 12), ('np1_live1 + np1_live2 + np1_live4', 9), ('concurrent8_steady', 6))

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/s6 window (the ship-6 candidate; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/s6 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt and
# ORDER-SHIP.txt have the order, the stop rules, the minutes and the read rules of the two runs.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the same in every
# smoke job of this pack, so each pair differs by the profile alone. The replay and build steps do not forward the selection (the job reader refuses the keys beside them): they run the base checkpoint."""

WC_READ = ('W3 WC: the "[PINDIAG] tp4 gdn wc engaged" line (conv_cores, gate_cores and z_cores above 0) once per engine start that has an M3 block, a "gdn wc audit <n> exact=True" line on the audited twin and no "audit mismatch"; '
           'the octo block (8 users of 8 rows) declines it with ONE "[PINDIAG] tp4 gdn wc fell back reason=8 users (WC takes 2 or 4)" line, which is the design (w3_wc_smoke accepts exactly that line); any other "fell back" line fails.')
GOVERNOR_READ = ('GOVERNOR: the install line "[PINDIAG] lever N adaptive: decoders<=2 share=0.9 ttft=120 gap_floor=8" once; every scheduler step line carries base=, target= and adaptive=; adaptive=1 exactly on the prefill steps that ran '
                 'with one or two decoders live (and the base share 0.9, the target 120); adaptive=0 and today\'s 0.5/180 with none or three or more; f_eff never below base (c2_smoke_check.levern_adaptive_problems judges all of it; '
                 'prefill_few_decoders NOT EXERCISED means no step was adaptive). The decode-gap floor (8 s) holds: no gap over it in the decoders of prefill_few_decoders.')
FEW_READ = ('prefill_few_decoders (ONE decoder and a ~120k cold arrival, then TWO decoders and a ~254k one): per shape the arrival\'s first-token time and prefill rate, each decoder\'s longest gap from the arrival on and its '
            'progress inside the arrival\'s window. ESTIMATE (docs/lever-n-adaptive-governor.md, from the timing model, never measured at 254k): the candidate\'s first token comes 31 to 44 percent sooner (254k: about 182-187 s to '
            '114-119 s; 120k: about 85 s to 50 s) and the decoders get about a fifth of the control\'s rate inside that window. Read the pair with `python3 scripts/ci/prefill_ladder_report.py <T1A smoke log> --second <T1B smoke log>`; '
            'the answers of the cold arrivals must hash EQUAL between the arms (greedy text, only WHEN the rounds run moves).')
AGENT_READ = ('agent_turns (6 sessions of 64k, 128k and 200k tokens, 12 returning turns each) is the MAIN JUDGE: the ship-5 window\'s T1B number (50.36 tok/s aggregate, returning turn first token p50 1.46 s, p90 2.33 s) is its baseline. '
              'agent_turns8 is the same plan with EIGHT sessions (its sessions 0 to 5 are agent_turns\' six; two more contexts of about 195k and 64k tokens are added), every seat busy, no new turn after 25 minutes, an estimate of 15 '
              'minutes: the production load, where WC (2 or 4 users) and the governor (1 or 2 decoders) mostly sit out, so the pair says whether anything is LOST there. Read both legs with `python3 scripts/ci/agent_turns.py report '
              '--smoke-log <smoke> --container-log <container> --test <name>`: the rates, the TTFT by context bucket, the tokens re-prefilled, the per-live table; the per-turn answer hashes must be EQUAL between the legs.')
NO_GO = ('NO-GO for the candidate: any text difference against the control, an audit mismatch, a fall-back line beyond the octo block\'s one, a floor violation (a decode gap over 8 s), a step line that does not carry the governor\'s fields, '
         'an eight-live window outside the boot-to-boot drift of the control, an errored or timed-out agent request. A failing lever is taken out by deleting its line from the ship entry of scripts/ci/fusion-wp/S6.json and regenerating '
         '(WC: QWEN_FAST_TP4_GDN_WC; the governor: QWEN_FAST_LEVERN_ADAPTIVE and its name in the waiver\'s levers; a new build).')


def agent_read_for(tests):
    return AGENT_READ if any(name.startswith('agent_turns') for name in tests.split(',')) else ''


def rows():
    """[row]: every template of the folder; row['orders'] says which ORDER files list it."""
    out = []
    out.append(dict(name='B0-build-window', orders=('w',), mode='stop', minutes=MINUTES['B0'], actions='build', profile=None, tests=None, telemetry=False, box=None, model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP5, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/s6 head (the shipped ship-5 stack and its checkpoint selection plus the adaptive governor and W3 WC, both default off): the profile that serves '
                         'production, %s, baked as the serving default with a session cap of 8 (its waiver is APPROVED, so the build is not refused), plus C2_BAKE_GRAFT=K64j-OQ. The candidate %s is a profile of the file and is chosen by a '
                         'job\'s C2_PROFILE; it cannot be the baked default while its owner traffic waiver is PENDING (ORDER-SHIP.txt, B1). READ: "baking the serving default %s with a session cap of 8", the provenance report with no '
                         'problem line, its K64j-OQ graft line, the image id. A build, not a card job: it may run while production serves; no privileged builder container may be up and no card job may run during it. Estimate from the '
                         'last builds: 35-40 minutes.' % (IMAGE, SHIP5, SHIP6, SHIP5))))
    out.append(dict(name='X0-status-rescan-reset', orders=('w',), mode='stop', minutes=MINUTES['X0'], actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host: "weight cache ...: found" is the read of every smoke job below. Stop rule: anything else. '
                         'The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='A1-s6-audited-attach', orders=('w',), mode='stop', minutes=MINUTES['A1'], actions='reset smoke', profile=AUDIT, tests=A1_TESTS, telemetry=True, box=BOX['A1'], model=True, extra=[],
                    why=('A1 (EXACTNESS of the whole candidate, STOP): %s, the audited twin of the ship-6 candidate: every audit of the shipped ship-5 stack (the K/V page writer, the verify T2 audit, the pre-stage audits, the multi-user '
                         'SDPA launch, the vglue, conv-gates and octo-lever audits, the 13x10 grids, the host tier with every instrument on, the carry\'s audit, the hand-off\'s audit) and W3 WC with its audit (it rode the ship-5 twin '
                         'in two card runs, 0 mismatches over ~12,800 audited rounds) PLUS the adaptive governor, on Swift. Tests: warmup; coding (one user, 1,500 tokens); np1_live2 and np1_live4 (two and four live seats: the '
                         'counts WC acts at); prefill_few_decoders (the governor\'s own shape, audited); concurrent8_drain (eight users whose budgets end one by one: every live count from 8 to 1); concurrent8_steady (the octo rounds, '
                         'required by the smoke check on this profile); agent_turns last. READ: c2_smoke_check clean (its fusion, octo, grid13, WC and governor rules read the container log); every audit line exact=True with entries above 0; '
                         'octo_judge.py --container-log <container log> --profile %s clean; NO "audit mismatch", NO "fell back" but the octo block\'s one WC line, NO "kv shared", no stall. %s %s Then python3 scripts/ci/sched_report.py '
                         '<container log>. The audited profile holds about 2.6 GB more DRAM per chip than the unaudited one (the ship-5 window: A1 timed out three agent-turn reads on admission holds that T1B did not have): a hold '
                         'line on this job is the audit\'s own weight, not the candidate\'s, and it is not a stop reason by itself. ANY audit mismatch, a WC or governor refusal, or a stall is NO-GO for the candidate and halts the '
                         'window. %s %s %s %s %s' % (AUDIT, AUDIT, WC_READ, GOVERNOR_READ, (agent_read_for(A1_TESTS)), s5jobs.CHECKPOINT_READ, s5jobs.TELEMETRY, s5jobs.ARC, ''))))
    for letter, role, profile, tests, minutes, box, what in (
            ('T1A', 's5-control', SHIP5, T1_TESTS, MINUTES['T1A'], BOX['T1A'], 'CONTROL'),
            ('T1B', 's6-candidate', SHIP6, T1_TESTS, MINUTES['T1B'], BOX['T1B'], 'CANDIDATE'),
            ('T2A', 's5-control-8', SHIP5, T2_TESTS, MINUTES['T2A'], BOX['T2A'], 'CONTROL'),
            ('T2B', 's6-candidate-8', SHIP6, T2_TESTS, MINUTES['T2B'], BOX['T2B'], 'CANDIDATE')):
        pair = 'T1' if letter.startswith('T1') else 'T2'
        suffix = ('the profile that serves production (ship 5), the CONTROL of the pair' if what == 'CONTROL' else
                  'the ship-6 candidate, a TRAFFIC profile whose owner waiver is PENDING (a profile whose decision does not start with APPROVED boots in a gate or a smoke and logs the word)')
        if pair == 'T1':
            body = ('Tests: np1_live1, np1_live2 and np1_live4 (one, two and four users started together on ~3,500-token real-code prompts, every answer to EXACTLY 2,000 tokens with ignore_eos: the live counts WC acts at '
                    'two and four of, and the lone user NP1 serves); prefill_few_decoders (the governor\'s shape); concurrent8_steady (eight users: the octo block, which neither WC nor the governor touches, so a parity row and the '
                    'boot-to-boot floor); agent_turns last (the main judge). %s %s %s' % (FEW_READ, AGENT_READ, ''))
            reads = ('READ%s: c2_smoke_check clean on both legs%s; then python3 scripts/ci/np1_report.py --control-a <T1A smoke> --arm-a <T1B smoke> --control-b <T1A smoke> --arm-b <T1B smoke> --arm-container <T1B container> '
                     '--profile %s (an ABAB reader fed one boot a side: each smoke is passed twice, the boot-to-boot floors read exactly 1; a LOSS of 2 to 5 percent at a parity row is boot noise until a second pair says otherwise) '
                     'and python3 scripts/ci/octo_compare.py --control <T1A smoke> --octo <T1B smoke> --octo-container <T1B container> --profile %s (every answer equal, user by user). %s %s %s'
                     % ('' if what == 'CONTROL' else ' (T1B)', '' if what == 'CONTROL' else ' and octo_judge.py clean on the candidate\'s log', SHIP6, SHIP6, WC_READ if what == 'CANDIDATE' else
                        'The control logs none of WC\'s or the governor\'s lines (the smoke check fails a line on a profile that does not ask for it).', GOVERNOR_READ if what == 'CANDIDATE' else '', NO_GO if what == 'CANDIDATE' else ''))
        else:
            body = ('Tests: concurrent8_steady (the octo rounds, required by the smoke check) and agent_turns8 (EIGHT sessions of the agent_turns plan, every seat busy). OPTIONAL: it runs when the window has the minutes. %s' % AGENT_READ)
            reads = ('READ: c2_smoke_check clean on both legs; python3 scripts/ci/agent_turns.py report --test agent_turns8 on each leg side by side; the answer hashes equal; no LOSS of the candidate against the control '
                     'outside the boot-to-boot drift. %s' % (NO_GO if what == 'CANDIDATE' else ''))
        out.append(dict(name='%s-%s' % (letter, role), orders=('w',), mode='soft' if pair == 'T1' else 'opt', minutes=minutes, actions='reset smoke', profile=profile, tests=tests, telemetry=True, box=box, model=True, extra=[],
                        why=('%s (TIMED, the %s of the %s pair): %s on Swift. %s Every boot a fresh reset. %s %s %s %s' % (letter, what, pair, suffix + ' ' + profile, body, reads, s5jobs.ARC, s5jobs.CHECKPOINT_READ, s5jobs.TELEMETRY))))
    out.append(dict(name='B1-ship-build', orders=('ship',), mode='soft', minutes=MINUTES['B1'], actions='build', profile=None, tests=None, telemetry=False, box=None, model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP6, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B1 (AFTER THE OWNER APPROVES; refused until then, by design): build the SHIP image from the commit that records the owner\'s approval, with %s baked as the image\'s serving default (ENV QWEN_C2_PROFILE) and '
                         'THATCH_SERVING_SESSION_CAP baked as its max-num-seqs (8). REFUSED by c2_serving_job.read_bake and by build-c2-serving-image.sh while the profile\'s owner_traffic_waiver (id %s) has a decision that does '
                         'not start with APPROVED: the owner approves first, the approval is committed (make_fusion_profiles.py --approve-ship --ship traffic-s6), the CPU suite runs on that commit, and the tag is pushed on THAT '
                         'commit. READ: "baking the serving default %s with a session cap of 8", the provenance report "(c) the image default is the BAKED profile ...", no problem line, the image id. THE IMAGE BAKES THE '
                         'K64j-OQ OP GRAFT. Estimate 35 minutes.' % (SHIP6, waiver_id(), SHIP6))))
    out.append(dict(name='SR-platform-replay', orders=('ship',), mode='soft', minutes=MINUTES['SR'], actions='reset replay', profile=None, tests=None, telemetry=False, box=60, model=False, extra=[],
                    replay=SHIP6,
                    why=('SR (AFTER THE OWNER APPROVES, on the image B1 builds): the four-card platform replay of the PRODUCTION image: the node agent\'s serving sequence in a copy of its container running the THIN LAYER built on the '
                         'B1 image (C2_PLATFORM_IMAGE=local:thin-layer: the step reads the image name from a file the operator writes on the runner; it is never committed). The replay sets no profile, so it boots the ship profile BY THE '
                         'IMAGE DEFAULT, which is why it needs the baked B1 image and so the APPROVED decision. The replay needs the recorded source image present on the rig host (c2_platform_replay.source_problem); the image name stays '
                         'out of this file. READ the vLLM subprocess log: \'[QWEN-C2] profile %s: vLLM argv\', the OWNER TRAFFIC WAIVER line, the P150x4 mesh, admission passed with no UNQUALIFIED admission line and no 262k evidence '
                         'waiver, prefix installed, the Lever N install line and the governor\'s ("lever N adaptive: decoders<=2 ..."), the parked build line, WC\'s engaged line, every op-fusion lever\'s engaged line, the :tt alias before '
                         'and after the restart, and no kill-switch line. PASS: \'PLATFORM_REPLAY passed=True\'. The replay runs the base checkpoint (the replay step does not forward the selection); production picks Swift from its node '
                         'environment. Measured basis for the estimate: 25 minutes.' % SHIP6)))
    out.append(dict(name='Z-reset', orders=('w',), mode='soft', minutes=MINUTES['Z'], actions='status reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, link '
                         'measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    return out


def waiver_id():
    profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    return profiles[SHIP6]['owner_traffic_waiver']['id']


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(' '.join(why.split()))]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row.get('replay'):
        lines += ['C2_PLATFORM_IMAGE=local:thin-layer', 'C2_REPLAY_PROFILE=%s' % row['replay']]
    if row['model']:
        lines += MODEL
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    lines += row['extra']
    if row['telemetry']:
        lines.append('C2_TELEMETRY=1')
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def table_of(table, key):
    return [row for row in table if key in row['orders']]


def totals(table):
    total = sum(row['minutes'] for row in table)
    builds = sum(row['minutes'] for row in table if row['actions'] == 'build')
    return total, total - builds, builds


COMMON_FORMAT = "# One template per line: <template name without .env> <stop|soft|opt> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it). opt: optional, admitted only if it fits."

CANDIDATE = ("# THE CANDIDATE (docs/tp4-next2.md): %s, a TRAFFIC profile under ONE owner traffic waiver (id %s, decision PENDING: the owner approves at ship time) = the ship-5 profile %s (the profile that serves production, the control "
             "of every pair) plus exactly two switches: W3 WC (QWEN_FAST_TP4_GDN_WC=1: the GDN layer's windows, conv and gates in one launch at two or four live users, declined at the octo block's eight) and the adaptive Lever N "
             "governor (QWEN_FAST_LEVERN_ADAPTIVE=1 at its default knobs: while one or two decoders are live the prefill share is 0.9 and the first-token deadline 120 s instead of 0.5 and 180 s; with none or three or more the "
             "pacing is today's, step for step). Its audited twin is %s. ESTIMATES (nothing here has run on a card): a cold long prompt beside one or two decoders reaches its first token 31 to 44 percent sooner, paid for by those "
             "decoders' rate in that window; WC saves device time at two to four live seats and gives nothing at six or more.")


def render_order_window(table):
    total, cards, builds = totals(table)
    names = [row['name'].split('-')[0] for row in table]
    lines = [
        "# The order of the tp4/s6 WINDOW (docs/tp4-next2.md; scripts/ci/make_s6_jobs.py): the ship-6 candidate %s on top of the shipped ship-5 stack, the checkpoint production serves (%s, revision %s), ONE image (%s, a "
        "placeholder: B0 builds it from the pushed tp4/s6 head WITH the K64j-OQ graft) and ONE window. The control of every pair is the ship-5 profile %s on the same image." % (SHIP6, REPO, REVISION[:12], IMAGE, SHIP5),
        COMMON_FORMAT,
        CANDIDATE % (SHIP6, waiver_id(), SHIP5, AUDIT),
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the candidate goes to production in place of the ship-5 profile. A1 is the exactness gate of both additions together with every audit of the shipped stack; T1A T1B time the "
        "candidate against production at one, two and four live seats, on the governor's own shape (prefill_few_decoders), with eight beside them and on agent_turns (the main judge); T2A T2B repeat the comparison at the production load "
        "(agent_turns8: eight sessions) and are optional. Every result is UNQUALIFIED.",
        "# ESTIMATES OF THE TESTS (minutes inside a boot; the card decides): %s. A boot is 12 to 15 minutes before the first test." % ', '.join('%s %d' % pair for pair in TEST_MINUTES),
        "# THE NEW DEVICE FACTS (first on a card here): the governor has never run on a card, audited or not (it changes WHEN the decode rounds run between prefill steps, never what a step computes); W3 WC has run only audited "
        "(two card runs, 0 mismatches) and never in a timed run. A1 is the first card to run both with every audit on; T1B the first timed run of either. Possible JIT compile at the first WC block launch is NOT new (the "
        "kernels compiled in the ship-5 window); the governor compiles nothing.",
    ] + s5jobs.COMMON_P0 + [
        "# ARC-ZERO A1 T2B",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER the "
        "last timed job that runs (T1B, or T2B when the window has the minutes), i.e. around every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): GATES: X0 A1 (%d min); DECODE PAIR: T1A T1B (%d min); PRODUCTION LOAD: T2A T2B (%d min, optional); END: Z. B0 is the WINDOW build and runs while production serves, so it is outside the card minutes."
        % (MINUTES['X0'] + MINUTES['A1'], MINUTES['T1A'] + MINUTES['T1B'], MINUTES['T2A'] + MINUTES['T2B']),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for need in ('X0 <- B0', 'A1 <- X0', 'T1A <- A1', 'T1B <- T1A', 'T2A <- T1B', 'T2B <- T2A'):
        lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, whose fusion, octo, grid13, exact-resume, WC and governor rules read the container log, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP5,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) every audit line exact=True with entries above 0, octo_judge.py clean under %s, WC's engaged and audit lines (the octo block's one 'fell back reason=8 users' line is the design), the governor's install line and "
        "adaptive fields on every step line, agent_turns judged (no errored or timed-out request; reuse on at least 90%% of the returning turns), no 'audit mismatch', no other 'fell back', no 'kv shared', sched_report.py read. ANY other failure is NO-GO and halts the window." % AUDIT,
        "#   T1A T1B   c2_smoke_check clean on both; np1_report.py (no LOSS at np1_live1, np1_live2, np1_live4; WC's gain, if any, at two and four), prefill_ladder_report.py --second (the cold arrival's first-token ratio B over A, the decoders' longest gaps and window progress, the answer hashes equal), "
        "octo_compare.py (every answer equal), agent_turns.py report on both legs side by side (the main judge).",
        "#   T2A T2B   (OPT) c2_smoke_check clean on both; agent_turns.py report --test agent_turns8 side by side; no LOSS outside the boot-to-boot drift.",
        "# " + NO_GO,
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z), %d of build." % (', '.join('%s %d' % (name, row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards, builds),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def render_order_ship(table):
    total, cards, builds = totals(table)
    names = [row['name'].split('-')[0] for row in table]
    lines = [
        "# The order of the tp4/s6 SHIP FLOW (docs/tp4-next2.md; scripts/ci/make_s6_jobs.py): B1 builds the ship image and SR replays the platform's serving sequence on it. BOTH WAIT FOR THE OWNER'S APPROVAL of the owner traffic waiver of %s (id %s): "
        "B1 is refused by c2_serving_job.read_bake while its decision starts with PENDING, and SR boots the profile BY THE IMAGE DEFAULT (c2_platform_replay), so it needs B1's image. Order: the owner approves, `python3 scripts/ci/make_fusion_profiles.py "
        "--approve-ship \"APPROVED ...\" --ship traffic-s6` regenerates the profile, the CPU suite runs on that commit, B1's tag is pushed on THAT commit, then SR. RUN IT FROM A COMMIT WHERE THIS FILE IS ORDER.txt (git mv ORDER-SHIP.txt ORDER.txt)." % (SHIP6, waiver_id()),
        COMMON_FORMAT,
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE SR and restores it after, with the cutover pack's own steps (B1 is a build and may run while production serves). The card jobs of the window are over.",
        "# P0. The CPU suite is green on the commit that records the approval, and each tag is pushed on THAT commit. Push ONE tag at a time and wait for its run to finish; never cancel a running card job.",
        "# THE JOBS: SHIP BUILD: B1 (%d min); PLATFORM REPLAY: SR (%d min)." % (MINUTES['B1'], MINUTES['SR']),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
        "# NEEDS SR <- B1",
        "# READ RULES:",
        "#   B1   'baking the serving default %s with a session cap of 8', the provenance report '(c) the image default is the BAKED profile ...', no problem line, the image id." % SHIP6,
        "#   SR   'PLATFORM_REPLAY passed=True'; the lever reads of the template's header (the governor's install line, WC's engaged line).",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards." % (', '.join('%s %d' % (name, row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def generate():
    table = rows()
    out = {}
    for row in table:
        out['%s.env' % row['name']] = render_env(row)
    out['ORDER.txt'] = render_order_window(table_of(table, 'w'))
    out['ORDER-SHIP.txt'] = render_order_ship(table_of(table, 'ship'))
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
            sys.stderr.write('%s/%s is not what make_s6_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
