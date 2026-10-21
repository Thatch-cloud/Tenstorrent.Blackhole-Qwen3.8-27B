"""The tp4/s7 window pack (scripts/ci/references/fusion-jobs/S7), generated (stdlib only).

    python3 scripts/ci/make_s7_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_s7_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The ship-7 candidate (docs/tp4-s7.md): the traffic profile `-fx-traffic-s7` (scripts/ci/fusion-wp/S7.json, owner traffic waiver ship7-fx-262k-2026-10-21) = the ship-6 profile plus the host tier's threaded upload,
admission pipelining, the memory-ledger trim and prompt lookup beside engine reuse (the traced masked tails are NOT in it: docs/tp4-s7.md); its audited twin (make_s7_profiles.py); THE SHIP-6 PROFILE ITSELF as the
control of every pair; ONE image (tp4-s7-1, a placeholder: B0 builds it from the pushed tp4/s7b head WITH the K64j-OQ graft AND THE SHIP-7 PROFILE BAKED AS THE SERVING DEFAULT, so the same image is the window's and,
if the gates pass, the ship's: the waiver's decision must already be APPROVED when B0 is pushed, the job reader refuses a PENDING one), the checkpoint production serves (selected by the job keys C2_MODEL_REPO and
C2_MODEL_REVISION: qwen_c2_checkpoint.py). Templates only: no tag is pushed from the branch.

TWO ORDER FILES share the folder (the window driver reads a fixed ORDER.txt, so the flow after the window is run from a commit where its file is ORDER.txt: `git mv ORDER-SHIP.txt ORDER.txt`):

  ORDER.txt        THE WINDOW: B0 the ship-image build (while production serves), X0, A1 the audited attach of the candidate's twin (STOP), T1A and T1B the timed pair at one, two and four live seats, the
                   governor's own shape (prefill_few_decoders), eight beside them and agent_turns (the main judge), R1 and R2 the restore pair (a returning session evicted to the host tier and restored: the
                   threaded upload's own test), T2A and T2B the production-load pair on agent_turns8 (optional), Z.
  ORDER-SHIP.txt   AFTER THE GATES (and the owner's reading of them): SR the platform replay of the candidate on the B0 image. There is no B1: B0 already baked the ship profile.

Every minute below is an ESTIMATE from the 2026-10-21 ship-6 window's jobs (X0 1, A1 61, T1A and T1B 18 each, T2A and T2B 15 each) and, for R1 and R2, from the upload-1 pack's own estimates; the card decides.
"""

import argparse
import json
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_s5_jobs as s5jobs
import make_s6_jobs as s6jobs
import make_s6_profiles as s6
import make_s7_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'S7'
IMAGE = 'tp4-s7-1'
REPO, REVISION, SERVED, MODEL = s5jobs.REPO, s5jobs.REVISION, s5jobs.SERVED, s5jobs.MODEL
SHIP6 = s6.SHIP                         # the ship-6 profile: the control of every pair (approved 2026-10-21, about to ship)
SHIP7 = twins.SHIP                      # the candidate (a traffic profile; the window image bakes it as the serving default)
AUDIT = twins.AUDIT                     # the audited twin of the candidate
# the audited attach: ship 6's A1 tests (the live counts WC acts at, the governor's shape, the drain through every live count, the octo rounds, the agent sessions with their returning turns)
A1_TESTS = s6jobs.A1_TESTS
T1_TESTS = s6jobs.T1_TESTS
T2_TESTS = s6jobs.T2_TESTS
RESTORE_PLAN, RESTORE_CONTROL_PLAN = 'tier-upload-timed', 'tier-upload-timed-control'
MINUTES = {'B0': 40, 'X0': 1, 'A1': 61, 'T1A': 18, 'T1B': 18, 'R1': 45, 'R2': 45, 'T2A': 15, 'T2B': 15, 'Z': 5, 'SR': 25}
BOX = {'A1': 110, 'T1A': 45, 'T1B': 45, 'R1': 90, 'R2': 90, 'T2A': 40, 'T2B': 40}
ORDERS = ('ORDER.txt', 'ORDER-SHIP.txt')
TEST_MINUTES = s6jobs.TEST_MINUTES

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/s7 window (the ship-7 candidate; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/s7 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt and
# ORDER-SHIP.txt have the order, the stop rules, the minutes and the read rules of the two runs.
# ONE IMAGE FOR THE WHOLE WINDOW AND THE SHIP: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft and the ship-7 profile baked as the serving default
# (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the same in every
# smoke and prefix job of this pack, so each pair differs by the profile alone. The replay and build steps do not forward the selection (the job reader refuses the keys beside them): they run the base checkpoint."""

UPLOAD_READ = ("UPLOAD (QWEN_PREFIX_TIER_IO_THREADS=4, QWEN_PREFIX_TIER_VERIFY_THREADS=4): the line \"[PINDIAG] prefix: host tier upload on io_threads=4 io_chunk_blocks=.. verify_threads=4 io_wrapped=1\" once, a returning "
               "session restored through the threaded IO (the restore line and tier_io_write_blocks above 0 in the counters), NO \"host tier upload latched off\" line and NO \"host tier upload fell back\" line (a latch is a FAIL: the "
               "call was repeated through the serial code, so the answers are right and the lever is not working), zero digest failures. ESTIMATE: a 254k restore of about 1 s falls toward 0.4 to 0.6 s, so a returning turn's first "
               "token (5.4 to 6.5 s) gains under 0.5 s.")
SEED_READ = ("SEED FIRST (QWEN_FAST_ADMISSION_SEED_FIRST=1): the install line \"[PINDIAG] admission pipeline installed: seed_first=1 hold_reuse=0 ledger_trim=<1 on the candidate, 0 on the audited twins>\" once; a "
             "\"[PINDIAG] admission seed first deferred\" line per returning turn that rebinds a parked engine and a \"... seed first built request=.. ms=.. after_seed_ms=..\" line after each; NO \"admission seed first: the bind of "
             "... failed\" line; a first token's client time that leads its bind's built line; python3 scripts/ci/sched2_report.py <container log> prints the segments. ESTIMATE: a returning turn's first token p50 1.46 s to about 1.19 s.")
LEDGER_READ = ("LEDGER TRIM (QWEN_FAST_ADMISSION_LEDGER_TRIM=1, the candidate only: the audited twins never trim): \"[PINDIAG] admission ledger trim: the memory ledger ended after the attach\" once and no per-rebind "
               "bracket lines after the attach. ESTIMATE: nothing on a warm process, about 1.7 s of a 15.4 s burst into a fresh process.")
LOOKUP_READ = ("LOOKUP (QWEN_FAST_LOOKUP_DRAFT=n3m12, the first card run of the lookup anywhere): \"[LOOKUP-DRAFT] engaged policy=n3m12\" per rebound request (build_ms in it: the index build runs on the step thread and "
               "stalls every decoding seat; about 0.07 s at the 32,768-token cap, ESTIMATE) and a \"[LOOKUP-ROUND]\" line per user per round. The target verifies every row, so the answers must be EQUAL to the control's; "
               "only the tokens a round commits can move. If the container log carries no engaged line the engine-core subprocess's output is not reaching it: say so, do not read the arm.")
INHERITED_READ = ("The ship-6 reads stand: " + s6jobs.WC_READ + " " + s6jobs.GOVERNOR_READ)
NO_GO = ("NO-GO for the candidate: any text difference against the control, an audit mismatch, a latch or fall-back line of any of the four groups, a fall-back line beyond the octo block's one WC line, a floor violation "
         "(a decode gap over 8 s), an allocator or admission failure, an eight-live window outside the boot-to-boot drift of the control, a loss in the restore pair, an errored or timed-out agent request. A failing lever is "
         "taken out by deleting its line from the ship entry of scripts/ci/fusion-wp/S7.json (and, for seed first and the ledger trim, from the waiver's levers) and regenerating (a new build): "
         "QWEN_PREFIX_TIER_IO_THREADS with QWEN_PREFIX_TIER_VERIFY_THREADS, QWEN_FAST_ADMISSION_SEED_FIRST, QWEN_FAST_ADMISSION_LEDGER_TRIM, QWEN_FAST_LOOKUP_DRAFT.")
READS = ' '.join((UPLOAD_READ, SEED_READ, LEDGER_READ, LOOKUP_READ))


def agent_read_for(tests):
    return s6jobs.AGENT_READ if any(name.startswith('agent_turns') for name in tests.split(',')) else ''


def rows():
    """[row]: every template of the folder; row['orders'] says which ORDER files list it."""
    out = []
    out.append(dict(name='B0-ship-build', orders=('w',), mode='stop', minutes=MINUTES['B0'], actions='build', profile=None, tests=None, telemetry=False, box=None, model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP7, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW AND SHIP image %s, built from the pushed tp4/s7b head (the ship-6 stack plus the four groups of levers, each default off in its code) with the ship-7 candidate %s baked as the '
                         'serving default (ENV QWEN_C2_PROFILE) and THATCH_SERVING_SESSION_CAP baked as its max-num-seqs (8), plus C2_BAKE_GRAFT=K64j-OQ. REFUSED by c2_serving_job.read_bake and by build-c2-serving-image.sh while the '
                         'profile\'s owner_traffic_waiver has a decision that does not start with APPROVED: the owner\'s approval (conditional on this window\'s gates) is committed first (make_fusion_profiles.py --approve-ship '
                         '--ship traffic-s7), the CPU suite runs on that commit and the tag is pushed on THAT commit. The control %s is a profile of the file chosen by a job\'s C2_PROFILE, so one image serves both arms of every pair. '
                         'READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line ("(c) the image default is the BAKED profile ..."), its K64j-OQ graft line, the image id. A build, '
                         'not a card job: it may run while production serves; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: 35-40 minutes.'
                         % (IMAGE, SHIP7, SHIP6, SHIP7))))
    out.append(dict(name='X0-status-rescan-reset', orders=('w',), mode='stop', minutes=MINUTES['X0'], actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host: "weight cache ...: found" is the read of every smoke job below. Stop rule: anything else. '
                         'The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='A1-s7-audited-attach', orders=('w',), mode='stop', minutes=MINUTES['A1'], actions='reset smoke', profile=AUDIT, tests=A1_TESTS, telemetry=True, box=BOX['A1'], model=True, extra=[],
                    why=('A1 (EXACTNESS of the whole candidate, STOP): %s, the audited twin of the ship-7 candidate: every audit of the shipped stack (the K/V page writer, the verify T2 audit, the '
                         'pre-stage audits, the multi-user SDPA launch, the vglue, conv-gates and octo-lever audits, the 13x10 grids, the host tier with every instrument on, the carry\'s audit, the hand-off\'s audit), W3 WC with its audit '
                         'and the adaptive governor (all three passed their card gates in the ship-6 window) PLUS the threaded upload, seed first and the lookup, on Swift. The ledger trim is NOT in '
                         'the twin (the audit\'s judge reads the ledger\'s points). Tests: warmup; coding; np1_live2 and np1_live4; prefill_few_decoders (the governor\'s shape); concurrent8_drain (every '
                         'live count from 8 to 1); concurrent8_steady (the octo rounds, required by the smoke check on this profile); agent_turns last (about seventy returning turns, and the '
                         'returning sessions the host tier restores). READ: c2_smoke_check clean (its fusion, octo, grid13, WC and governor rules read the container log); every audit line exact=True with entries above 0; '
                         'octo_judge.py --container-log <container log> --profile %s clean; NO "audit mismatch", NO "fell back" but the octo block\'s one WC line, NO "latched off", NO "kv shared", no stall. %s %s %s Then '
                         'python3 scripts/ci/sched_report.py <container log>. The audited profile holds about 2.6 GB more DRAM per chip than the unaudited one (the ship-5 and ship-6 windows\' A1 timed out three agent-turn reads on '
                         'admission holds that T1B did not have; the owner accepted that for ships 5 and 6): a hold line or a client timeout on '
                         'this job is the audit\'s own weight, not the candidate\'s, and it is not a stop reason by itself. ANY audit mismatch, a latch, or a stall is NO-GO for '
                         'the group it names and halts the window. %s %s %s %s' % (AUDIT, AUDIT, READS, INHERITED_READ, agent_read_for(A1_TESTS), NO_GO, s5jobs.CHECKPOINT_READ, s5jobs.TELEMETRY, s5jobs.ARC))))
    for letter, role, profile, tests, minutes, box, what in (
            ('T1A', 's6-control', SHIP6, T1_TESTS, MINUTES['T1A'], BOX['T1A'], 'CONTROL'),
            ('T1B', 's7-candidate', SHIP7, T1_TESTS, MINUTES['T1B'], BOX['T1B'], 'CANDIDATE'),
            ('T2A', 's6-control-8', SHIP6, T2_TESTS, MINUTES['T2A'], BOX['T2A'], 'CONTROL'),
            ('T2B', 's7-candidate-8', SHIP7, T2_TESTS, MINUTES['T2B'], BOX['T2B'], 'CANDIDATE')):
        pair = 'T1' if letter.startswith('T1') else 'T2'
        suffix = ('the profile ship 6 serves (the CONTROL of the pair)' if what == 'CONTROL' else
                  'the ship-7 candidate, a TRAFFIC profile under the owner waiver the ship-7 decision records (a PENDING one boots in a smoke and logs the word)')
        if pair == 'T1':
            body = ('Tests: np1_live1, np1_live2 and np1_live4 (one, two and four users started together on ~3,500-token real-code prompts, every answer to EXACTLY 2,000 tokens with ignore_eos); prefill_few_decoders (the '
                    'governor\'s shape); concurrent8_steady (eight users: the octo block); agent_turns last (the main judge: returning turns with tails, restores, rebinds and the lookup). %s %s' % (s6jobs.FEW_READ, s6jobs.AGENT_READ))
            reads = ('READ%s: c2_smoke_check clean on both legs%s; then python3 scripts/ci/np1_report.py --control-a <T1A smoke> --arm-a <T1B smoke> --control-b <T1A smoke> --arm-b <T1B smoke> --arm-container <T1B container> '
                     '--profile %s (an ABAB reader fed one boot a side: each smoke is passed twice, the boot-to-boot floors read exactly 1; a LOSS of 2 to 5 percent at a parity row is boot noise until a second pair says otherwise) '
                     'and python3 scripts/ci/octo_compare.py --control <T1A smoke> --octo <T1B smoke> --octo-container <T1B container> --profile %s (every answer equal, user by user), python3 scripts/ci/agent_turns.py '
                     'report on both legs side by side (the per-turn answer hashes EQUAL; the returning-turn first token p50 and p90 against the control, the ship-6 window\'s T1B being the baseline), python3 scripts/ci/sched2_report.py '
                     '<T1B container>. %s %s' % (
                         '' if what == 'CONTROL' else ' (T1B)', '' if what == 'CONTROL' else ' and octo_judge.py clean on the candidate\'s log', SHIP7, SHIP7,
                         ('The control logs none of the four groups\' lines (the smoke check fails a line on a profile that does not ask for it).' if what == 'CONTROL' else READS + ' ' + INHERITED_READ), NO_GO if what == 'CANDIDATE' else ''))
        else:
            body = ('Tests: concurrent8_steady (the octo rounds, required by the smoke check) and agent_turns8 (EIGHT sessions of the agent_turns plan, every seat busy). OPTIONAL: it runs when the window has the minutes. %s' % s6jobs.AGENT_READ)
            reads = ('READ: c2_smoke_check clean on both legs; python3 scripts/ci/agent_turns.py report --test agent_turns8 on each leg side by side; the answer hashes equal; no LOSS of the candidate against the control '
                     'outside the boot-to-boot drift. %s' % (NO_GO if what == 'CANDIDATE' else ''))
        out.append(dict(name='%s-%s' % (letter, role), orders=('w',), mode='soft' if pair == 'T1' else 'opt', minutes=minutes, actions='reset smoke', profile=profile, tests=tests, telemetry=True, box=box, model=True, extra=[],
                        why=('%s (TIMED, the %s of the %s pair): %s on Swift. %s Every boot a fresh reset. %s %s %s %s' % (letter, what, pair, suffix + ' ' + profile, body, reads, s5jobs.ARC, s5jobs.CHECKPOINT_READ, s5jobs.TELEMETRY))))
    for letter, role, profile, plan, what in (('R1', 'restore-s6-control', SHIP6, RESTORE_CONTROL_PLAN, 'CONTROL'), ('R2', 'restore-s7-candidate', SHIP7, RESTORE_PLAN, 'CANDIDATE')):
        suffix = ('the profile ship 6 serves, without the threaded upload (the CONTROL of the pair)' if what == 'CONTROL' else
                  'the ship-7 candidate, whose profile carries the threaded upload; the plan derives the same two values over it (the pair differs by the profile alone)')
        out.append(dict(name='%s-%s' % (letter, role), orders=('w',), mode='soft', minutes=MINUTES[letter], actions='reset prefix', profile=None, tests=None, telemetry=False, box=BOX[letter], model=True,
                        extra=['C2_PREFIX_PLAN=%s' % plan, 'C2_PREFIX_PROFILE=%s' % profile, 'C2_PREFIX_BASELINE=none'],
                        why=('%s (TIMED, the %s of the RESTORE pair; the test that exists for the threaded upload, a returning session evicted to the host tier and restored): the tier-timed cycle at ~64k, ~128k and ~254k '
                             '(c2_prefix_gate.py --plan %s: a cold prefill, the device-resident hit, floods the size of the KV pool, the returning hit that restores from the host tier, and its cold twin; no row digests, no extent '
                             'audit, the tier\'s own audits off, the digest check sampled) on %s. There is no restore-heavy smoke test (the smoke tests\' returning turns restore too, but untimed and mixed with prefill): this is the '
                             'prefix gate\'s own plan, one job a side, no new harness. THE PAIR: R1 control, R2 candidate, every boot a fresh reset, the ARC runner sets at ZERO for the job. READ each: the plan verdict PASS (every cold '
                             'twin IDENTICAL to its returning hit), the "host tier restore" line of each returning turn (blocks, GB, ms, slices); then python3 scripts/ci/upload_restore_compare.py <R1 results> <R2 results> <R1 results> '
                             '<R2 results> (it wants an ABAB; the single pair is passed twice, so its repeat floor reads 1: say so in the verdict). ESTIMATE: the restore ms down by at least 25 percent at 128k and 254k, the returning '
                             'hit\'s TTFT not above the control\'s by more than 5 percent; the lever cannot change the returning request\'s own final prefill step or engine build (4.2 to 4.8 s at 254k). %s %s %s' % (
                                 letter, what, plan, suffix + ' ' + profile, UPLOAD_READ, s5jobs.ARC, s5jobs.CHECKPOINT_READ))))
    out.append(dict(name='SR-platform-replay', orders=('ship',), mode='soft', minutes=MINUTES['SR'], actions='reset replay', profile=None, tests=None, telemetry=False, box=60, model=False, extra=[],
                    replay=SHIP7,
                    why=('SR (AFTER THE GATES, on the image B0 built): the four-card platform replay of the PRODUCTION image: the node agent\'s serving sequence in a copy of its container running the THIN LAYER built on the '
                         'B0 image (C2_PLATFORM_IMAGE=local:thin-layer: the step reads the image name from a file the operator writes on the runner; it is never committed). The replay sets no profile, so it boots the ship profile BY THE '
                         'IMAGE DEFAULT, which is why it needs the baked B0 image and so the APPROVED decision. The replay needs the recorded source image present on the rig host (c2_platform_replay.source_problem); the image name stays '
                         'out of this file. READ the vLLM subprocess log: \'[QWEN-C2] profile %s: vLLM argv\', the OWNER TRAFFIC WAIVER line, the P150x4 mesh, admission passed with no UNQUALIFIED admission line and no 262k evidence '
                         'waiver, prefix installed, the host tier upload line, the Lever N install line and the governor\'s, the admission pipeline install line, the parked build line, WC\'s engaged line, the lookup\'s engaged line, '
                         'every op-fusion lever\'s engaged line, the :tt alias before and after the restart, and no kill-switch line. PASS: \'PLATFORM_REPLAY passed=True\'. The replay runs the base checkpoint '
                         '(the replay step does not forward the selection); production picks Swift from its node environment. Measured basis for the estimate: 25 minutes.' % SHIP7)))
    out.append(dict(name='Z-reset', orders=('w',), mode='soft', minutes=MINUTES['Z'], actions='status reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, link '
                         'measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    order = ('B0', 'X0', 'A1', 'T1A', 'T1B', 'R1', 'R2', 'T2A', 'T2B', 'SR', 'Z')
    return sorted(out, key=lambda row: order.index(row['name'].split('-')[0]))


def waiver_id():
    profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    return profiles[SHIP7]['owner_traffic_waiver']['id']


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


COMMON_FORMAT = s6jobs.COMMON_FORMAT

CANDIDATE = ("# THE CANDIDATE (docs/tp4-s7.md): %s, a TRAFFIC profile under ONE owner traffic waiver (id %s, decision %s) = the ship-6 profile %s (the control of every pair) plus four groups of levers, "
             "each one line of its ship entry: the host tier's threaded upload (QWEN_PREFIX_TIER_IO_THREADS=4, QWEN_PREFIX_TIER_VERIFY_THREADS=4), admission pipelining (QWEN_FAST_ADMISSION_SEED_FIRST=1), the memory-ledger trim "
             "(QWEN_FAST_ADMISSION_LEDGER_TRIM=1) and prompt lookup beside engine reuse (QWEN_FAST_LOOKUP_DRAFT=n3m12). Its audited twin is %s. LEFT OUT and default off: the traced masked tails (QWEN_FAST_PREFILL_TAIL_TRACE: "
             "tp4/s7's audited attaches died at engine start on a device read inside the trace capture), the hold reuse, the K-split of the tail passes (a numerics waiver of its own), the tree verify and the lookup shadow. "
             "ESTIMATES (nothing here has run on a card): a returning turn's first token p50 1.46 s to about 1.19 s (seed first), +3 percent tokens a second from the lookup, a 254k restore from about 1 s to 0.4 to 0.6 s.")


def decision_word():
    profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    return profiles[SHIP7]['owner_traffic_waiver']['decision'].split(None, 1)[0].rstrip(':')


def render_order_window(table):
    total, cards, builds = totals(table)
    names = [row['name'].split('-')[0] for row in table]
    core = sum(row['minutes'] for row in table if row['name'].split('-')[0] in ('X0', 'A1', 'T1A', 'T1B', 'Z'))
    lines = [
        "# The order of the tp4/s7 WINDOW (docs/tp4-s7.md; scripts/ci/make_s7_jobs.py): the ship-7 candidate %s on top of the ship-6 stack, the checkpoint production serves (%s, revision %s), ONE image (%s, a "
        "placeholder: B0 builds it from the pushed tp4/s7 head WITH the K64j-OQ graft AND THE CANDIDATE BAKED AS THE SERVING DEFAULT, so the window's image is the ship image) and ONE window. The control of every pair is the ship-6 "
        "profile %s on the same image." % (SHIP7, REPO, REVISION[:12], IMAGE, SHIP6),
        COMMON_FORMAT,
        CANDIDATE % (SHIP7, waiver_id(), decision_word(), SHIP6, AUDIT),
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the candidate goes to production in place of the ship-6 profile. The owner approved the ship conditionally (\"Yes approve s7 if it passes\"); the "
        "waiver's decision sentence records the conditions. A1 is the exactness gate of the four groups with every audit of the shipped stack; T1A T1B time the candidate against ship 6 at one, two and four live seats, on "
        "the governor's shape, with eight beside them and on agent_turns (the main judge); R1 R2 are the restore pair; T2A T2B repeat the comparison at the production load (agent_turns8) and are optional. Every result is UNQUALIFIED.",
        "# ESTIMATES OF THE TESTS (minutes inside a boot; the card decides): %s. A boot is 12 to 15 minutes before the first test." % ', '.join('%s %d' % pair for pair in TEST_MINUTES),
        "# THE NEW DEVICE FACTS (first on a card here): the threaded upload, seed first, the ledger trim and the lookup have never run on a card, alone or together, audited or not; they compile "
        "nothing new for the device (no new kernel); W3 WC and the governor passed their card gates in the ship-6 window. A1 is the first card to run everything with every audit on; "
        "T1B the first timed run of any of the four.",
    ] + s5jobs.COMMON_P0 + [
        "# ARC-ZERO A1 T2B",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER the "
        "last timed job that runs (R2, or T2B when the window has the minutes), i.e. around every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): GATES: X0 A1 (%d min); DECODE PAIR: T1A T1B (%d min); RESTORE PAIR: R1 R2 (%d min); PRODUCTION LOAD: T2A T2B (%d min, optional); END: Z. "
        "The core (X0 A1 T1A T1B Z) is %d minutes of cards. B0 is the build of the ship image and runs while production serves, so it is outside the card minutes."
        % (MINUTES['X0'] + MINUTES['A1'], MINUTES['T1A'] + MINUTES['T1B'], MINUTES['R1'] + MINUTES['R2'], MINUTES['T2A'] + MINUTES['T2B'], core),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for need in ('X0 <- B0', 'A1 <- X0', 'T1A <- A1', 'T1B <- T1A', 'R1 <- T1B', 'R2 <- R1', 'T2A <- T1B', 'T2B <- T2A'):
        lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, whose fusion, octo, grid13, exact-resume, WC and governor rules read the container log, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP7,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) every audit line exact=True with entries above 0, octo_judge.py clean under %s, WC's engaged and audit lines (the octo block's one 'fell back reason=8 users' line is the design), the governor's install line and "
        "adaptive fields, the upload line and no latch, the admission pipeline install line, the lookup's engaged lines, "
        "agent_turns judged (no errored or timed-out request but the audit's own three; reuse on at least 90%% of the returning turns), no 'audit mismatch', no other 'fell back', no 'kv shared', sched_report.py read. ANY other "
        "failure is NO-GO and halts the window." % AUDIT,
        "#   T1A T1B   c2_smoke_check clean on both; np1_report.py (no LOSS at np1_live1, np1_live2, np1_live4), prefill_ladder_report.py --second (the cold arrival's first-token ratio B over A, the decoders' longest gaps, the answer "
        "hashes equal), octo_compare.py (every answer equal), agent_turns.py report on both legs side by side (the main judge), sched2_report.py on the candidate's log.",
        "#   R1 R2   the tier-upload-timed plans PASS each (every returning answer IDENTICAL to its cold twin), then upload_restore_compare.py: the restore ms down at 128k and 254k, the returning TTFT not up beyond the floor.",
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
        "# The order of the tp4/s7 SHIP FLOW (docs/tp4-s7.md; scripts/ci/make_s7_jobs.py): SR replays the platform's serving sequence on the image B0 built, AFTER the window's gates have been read and the owner's conditions are met. "
        "The owner traffic waiver of %s (id %s) is already APPROVED conditionally in the profile (B0 would have been refused otherwise); SR boots the profile BY THE IMAGE DEFAULT (c2_platform_replay). "
        "RUN IT FROM A COMMIT WHERE THIS FILE IS ORDER.txt (git mv ORDER-SHIP.txt ORDER.txt)." % (SHIP7, waiver_id()),
        COMMON_FORMAT,
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE SR and restores it after, with the cutover pack's own steps. The card jobs of the window are over.",
        "# P0. The CPU suite is green on the commit that records the approval, and the tag is pushed on THAT commit. Push ONE tag at a time and wait for its run to finish; never cancel a running card job.",
        "# THE JOB: PLATFORM REPLAY: SR (%d min)." % MINUTES['SR'],
        "# DEPENDENCIES (machine-greppable): none: SR needs the image B0 built and the gates read.",
        "# READ RULES:",
        "#   SR   'PLATFORM_REPLAY passed=True'; the lever reads of the template's header (the upload, admission pipeline, governor, WC and lookup lines).",
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
            sys.stderr.write('%s/%s is not what make_s7_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
