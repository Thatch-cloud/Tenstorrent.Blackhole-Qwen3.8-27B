"""The tp4/sched-2 window pack (scripts/ci/references/fusion-jobs/SCH2), generated (stdlib only).

    python3 scripts/ci/make_sched2_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_sched2_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The admission levers of docs/tp4-sched2.md on the shipped ship-5 stack: ONE image (tp4-sched-2, a placeholder: B0 builds it from the pushed tp4/sched-2 head WITH the K64j-OQ graft and bakes the
ship-5 candidate, whose owner waiver is APPROVED, as the serving default), ONE window, ONE control (the ship-5 candidate, what production serves) and its five gate-only twins
(make_sched2_profiles.py), the checkpoint production serves (selected by C2_MODEL_REPO and C2_MODEL_REVISION). Templates only: no tag is pushed from the branch by its author.

THE PACK CUTS AT BLOCK ENDS (ORDER.txt has the minutes): the shortest useful window is X0 A1 C1 L1 Z, 175 minutes of cards after the build; the second pair (C2 L2) and the attribution (SE1 HO1 LT1)
follow in value order. A1 the audited attach (exactness: every audit of the candidate with seed first and the
hold's reading reuse on, on eight fresh prompts of ~3.5k tokens, the drain of eight, and the production-shaped agent_turns run), then the ABAB of the combined arm against the control on
`concurrent8_steady` (eight fresh prompts of ~3.5k tokens into a FRESH process: the burst this branch is written for, whose last first token is 15.4 s at ship 5) and `agent_turns` (returning turns:
their first token p50 is 1.46 s at ship 5), then the three arms alone to say which lever moved what. The test order puts concurrent8_steady right after warmup on purpose: the ledger trim and the
slots' first-use costs are only visible in a process whose eight slots are all new.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_s5_jobs as s5
import make_sched2_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'SCH2'
IMAGE = 'tp4-sched-2'
REPO, REVISION, SERVED = s5.REPO, s5.REVISION, s5.SERVED
SHIP = twins.SHIP                                   # the ship-5 candidate (production): the control of every pair, and what B0 bakes as the default
ARM = {suffix: twins.NAMESPACE + suffix for suffix in ('seed', 'hold', 'ledger', 'all', 'audit')}
AUDIT_TESTS = 'warmup,coding,concurrent8_drain,concurrent8_steady,agent_turns'
TIMED_TESTS = 'warmup,concurrent8_steady,agent_turns'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/sched-2 window (the admission levers; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/sched-2 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a smoke job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the
# same in every smoke job of this pack, so each pair differs by the profile alone."""

REPORT = ('READ the arm with python3 scripts/ci/sched2_report.py <arm container log> --second <control container log> (and --burst for the first eight fresh prompts, one line an arrival): the serial chain '
          '(hold + drain + bind) of returning turns and of fresh prompts, the bind\'s parts (A, B, C and C\'s phases), the ledger bracket and the conv-slice readback (only at a slot\'s first use), the seed '
          'first lines (deferred, bound in the sampler and why, how long the first token left ahead of the bind), the hold reuse lines (how many holds took no allocator reading, their age). Every figure '
          'prints "-" when no line is behind it. The model of the same arithmetic is python3 scripts/ci/sched2_model.py (ESTIMATES: docs/tp4-sched2.md section 5).')
LEVER_LINES = ('the lines each lever logs (a lever whose line is absent did not engage, and its arm reads NOT_EXERCISED): "[PINDIAG] admission pipeline installed: seed_first=.. hold_reuse=.. ledger_trim=.." once; '
               'SEED FIRST: "[PINDIAG] admission seed first deferred request=.. slot=.." and "[PINDIAG] admission seed first built request=.. ms=.. after_seed_ms=.." per deferred admission (ms is the bind the first token did not wait for; after_seed_ms is how '
               'long the token had been out when the bind began), "[PINDIAG] admission seed first eager request=.. reason=.." for each bind that stayed in the sampler; HOLD REUSE: "[PINDIAG] admission hold '
               'reading reused prompt=.. age_ms=.. reuses=.." per reused hold (the dram admission fit lines then number the real readings); LEDGER TRIM: "[PINDIAG] admission ledger trim: the memory ledger '
               'ended after the attach" once and no "memory_ledger_before" line with "op": "rebind" in the whole log')
EXACT_READ = ('EXACT READ: every user\'s answer is byte-identical to the control\'s (the same checkpoint, greedy: a difference is a token a lever changed and is NO-GO); no "audit mismatch", no "fell back", '
              'no "admission seed first: the bind of" error line (a refusal after a seed had gone: it stops the engine, and the preflight makes it unexpected), no engine death, no stall.')
CHECKPOINT_READ = s5.CHECKPOINT_READ
TELEMETRY = s5.TELEMETRY
ARC = s5.ARC


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/sched-2 head (= tp4/next-1, the shipped ship-5 stack, plus the three admission levers, all default off): the ship-5 candidate '
                         '%s (its owner waiver is APPROVED) baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. Every lever is a gate instrument of the twins and is off in the baked '
                         'profile, so the image serves exactly what production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report with no '
                         'problem line, its K64j-OQ graft line, the image id. A build, not a card job; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: '
                         '35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=15, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host. Stop rule: anything else. The owner stops the '
                         'node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='A1-sch2-audited-attach', mode='stop', minutes=80, actions='reset smoke', profile=ARM['audit'], tests=AUDIT_TESTS, telemetry=True, box=150, needs=('A1 <- X0',), model=True, extra=[],
                    why=('A1 (EXACTNESS with seed first and the hold\'s reading reuse on, STOP): %s, the ship-5 audited twin (every audit of the candidate) with the two levers of docs/tp4-sched2.md that touch the order of '
                         'work on, on Swift: the seed token leaves the engine before the request\'s engine is bound (the bind runs at the start of the next scheduling call, ahead of the scheduler\'s own clock, hold and '
                         'gate), and the DRAM hold reuses a recent allocator reading while it clears the terms. Neither changes what a round computes: the order of the device work, the scheduler\'s view and the pacing are '
                         'the eager order\'s; the audits (the rebind\'s digests against a cold twin, the carry, the K/V page writer, the tier) run over every admission. Tests: warmup; coding (one user); concurrent8_drain '
                         '(eight users whose budgets end one by one: every live count from 8 to 1, so every hold and every bind happens beside every number of decoders); concurrent8_steady (eight fresh prompts together '
                         'into a process whose slots are new: eight deferred binds in a row, the hold reuse at its limit); agent_turns (the returning turns that reach the rebind and the exact resume). READ: c2_smoke_check '
                         'clean; every audit line exact=True with entries above 0. LEVERS: %s %s %s %s'
                         % (ARM['audit'], LEVER_LINES, EXACT_READ, CHECKPOINT_READ, TELEMETRY))))
    for index, (control, arm) in enumerate((('C1', 'L1'), ('C2', 'L2')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-control-%s' % (control, letter), mode='soft', minutes=35, actions='reset smoke', profile=SHIP, tests=TIMED_TESTS, telemetry=True, box=80,
                        needs=('C1 <- A1',) if index == 1 else ('C2 <- L1',), model=True, extra=[],
                        why=('%s (TIMED, the CONTROL of the %s ABAB pair): the ship-5 candidate %s on Swift, the stack production runs. Tests: warmup; concurrent8_steady (eight fresh prompts of ~3.5k tokens together into '
                             'a FRESH process, right after the warmup: every one of the eight slots is new, so the ledger\'s bracket and the conv-slice readback are paid eight times; the last first token is 15.4 s at '
                             'ship 5 and the engine builds are 497 / 736 / 622 / 640 / 556 / 557 / 598 / 522 ms); agent_turns (six sessions of 64k-200k tokens that return twelve times: the returning-turn first token is '
                             'p50 1.46 s / p90 2.33 s). Every boot a fresh reset. %s READ: %s %s %s'
                             % (control, 'first' if index == 1 else 'second', SHIP, ARC, REPORT, CHECKPOINT_READ, TELEMETRY))))
        out.append(dict(name='%s-all-%s' % (arm, letter), mode='soft', minutes=35, actions='reset smoke', profile=ARM['all'], tests=TIMED_TESTS, telemetry=True, box=80,
                        needs=('L1 <- C1',) if index == 1 else ('L2 <- C2',), model=True, extra=[],
                        why=('%s (TIMED, the ARM of the %s ABAB pair): %s, the ship-5 candidate with the gate marker in place of its owner waiver id and the three levers on (QWEN_FAST_ADMISSION_SEED_FIRST=1, '
                             'QWEN_FAST_ADMISSION_HOLD_REUSE=1, QWEN_FAST_ADMISSION_LEDGER_TRIM=1), the same tests as %s, on Swift. ESTIMATE (docs/tp4-sched2.md section 5, arithmetic over the control run\'s own '
                             'segments): concurrent8_steady\'s last first token 15.4 s -> about 12.7 s (-17%%; the ledger trim 1.7 s, the hold reuse 0.6 s, the seed first 0.35 s), up to 11.8 s if the owed decode rounds '
                             'shrink with the step; a returning turn\'s first token p50 1.46 s -> about 1.12 s (-23%%: the seed first 275 ms, the hold reuse 63 ms). The decoders\' tokens a second and agent_turns\' '
                             'committed rate must NOT fall. READ: c2_smoke_check clean; %s; %s Then, after the four boots (C1 L1 C2 L2): per user the first-token time, paired by index against the two controls and the '
                             'control-to-control floor, for concurrent8_steady (the eight arrivals in arrival order) and agent_turns (python3 scripts/ci/agent_turns.py report <smoke> <container> on each leg); '
                             'the decoders\' rate and the longest gap between two tokens of a stream (a lever that only moves delay shows here). %s %s %s'
                             % (arm, 'first' if index == 1 else 'second', ARM['all'], control, LEVER_LINES, EXACT_READ, ARC, REPORT, CHECKPOINT_READ))))
    for name, suffix, what, need, lever in (
            ('SE1-seed-a', 'seed', 'the seed first lever alone', 'SE1 <- L2', 'SEED FIRST: the first token leaves ahead of the bind by the bind\'s own time (the built lines\' ms, p50 about 275 ms on returning turns, '
                                                                                  '500-740 ms on the cold slots); the answers EQUAL; the decoders\' gaps and the rounds after each arrival unchanged against the control'),
            ('HO1-hold-a', 'hold', 'the hold\'s reading reuse alone', 'HO1 <- SE1', 'HOLD REUSE: of every four back-to-back holds one takes an allocator reading (the "dram admission fit" lines count the readings, '
                                                                                  'the "hold reading reused" lines the reuses); no "dram hold" line the control does not have; the hold segment p50 84 ms -> about 20 ms'),
            ('LT1-ledger-a', 'ledger', 'the ledger trim alone', 'LT1 <- HO1', 'LEDGER TRIM: no op=rebind bracket in the log (the control has eight, one per slot) and the bind of each fresh prompt about 170-310 ms '
                                                                                   'shorter; nothing changes on returning turns into warm slots')):
        out.append(dict(name=name, mode='soft', minutes=35, actions='reset smoke', profile=ARM[suffix], tests=TIMED_TESTS, telemetry=True, box=80, needs=(need,), model=True, extra=[],
                        why=('%s (ATTRIBUTION, %s): %s, the ship-5 candidate with the gate marker and its one switch, the tests of C1, on Swift. READ: %s. sched2_report.py against C1 or C2 (the nearer control). %s %s %s'
                             % (name.split('-')[0], what, ARM[suffix], lever, ARC, CHECKPOINT_READ, TELEMETRY))))
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

    blocks = (('GATES', 'B0 X0 A1'), ('COMBINED ABAB', 'C1 L1 C2 L2'), ('ATTRIBUTION', 'SE1 HO1 LT1'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/sched-2 window (docs/tp4-sched2.md; scripts/ci/make_sched2_jobs.py): the admission levers (seed first, the hold's reading reuse, the memory-ledger trim) on the ship-5 candidate %s and its "
        "five gate-only twins (make_sched2_profiles.py), the checkpoint production serves (%s, revision %s), ONE image (%s, a placeholder: B0 builds it from the pushed tp4/sched-2 head WITH the K64j-OQ graft) and ONE window."
        % (SHIP, REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether production should turn on, one by one, the three levers. Seed first and the hold's reuse change when work happens at an admission and the ledger trim removes "
        "a diagnostic: none of them changes a token, and each is a gate instrument today (a ship profile needs an owner traffic waiver naming it). Every result is UNQUALIFIED.",
        "# THE MEASUREMENT: concurrent8_steady first after the warmup (eight fresh prompts of ~3.5k tokens into a process whose slots are new: the burst) and agent_turns (production-shaped returning turns); both are in this tree.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, so every smoke job here logs \"weight cache ...: found\"." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# P0. MERGE POINTS: the adaptive Lever N governor (levern_policy.py, levern_scheduler.py) is ported on tp4/next-2 by another branch; this branch does not edit it (docs/tp4-sched2.md section 6). Run this window from a commit of tp4/sched-2 alone.",
        "# ARC-ZERO A1 LT1",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER LT1, i.e. around "
        "every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build. THE SHORTEST USEFUL WINDOW is X0 A1 C1 L1 Z: %d minutes of cards (the audited exactness gate and one control/arm pair); C2 L2 make it an ABAB (+70) and SE1 HO1 LT1 attribute (+105)." % ('; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks), minutes('X0 A1 C1 L1 Z')),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean where it runs and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) every audit line exact=True with entries above 0, no 'audit mismatch', no 'fell back', the lever lines of seed first and the hold reuse (sched2_report.py prints them), every answer equal where a control exists, no stall.",
        "#   C1 L1 C2 L2   the first token of each of the eight arrivals of concurrent8_steady and of every returning turn of agent_turns, paired against the two controls and their floor; the answers EQUAL; the decoders' rate and gaps not worse; sched2_report.py --second on the container logs.",
        "#   SE1 HO1 LT1   the attribution of the combined arm: the seed first lines' ms (SE1), the real readings against the reuses and the hold segment (HO1), the rebind brackets (none) and the fresh prompts' bind (LT1).",
        "# NO-GO for a lever: any text that differs from the control's, an audit mismatch, a stall, a refusal at the attach, an 'engaged' line that is absent, a LOSS in a timed test beyond the floor (the decoders' rate, the agent_turns rate, a first token later than the control's), "
        "and for seed first any 'admission seed first: the bind of' error line. Adopting a lever is the owner's decision: each is a gate instrument today and a ship profile needs an owner traffic waiver naming it (docs/tp4-sched2.md).",
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
            sys.stderr.write('%s/%s is not what make_sched2_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
