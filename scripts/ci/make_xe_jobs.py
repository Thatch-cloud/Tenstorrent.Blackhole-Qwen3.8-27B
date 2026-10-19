"""The tp4/exact-end window pack (scripts/ci/references/fusion-jobs/XE), generated (stdlib only).

    python3 scripts/ci/make_xe_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_xe_jobs.py --check   exit 1 if the checked-in pack is not what this generates

Exact resume P0 (QWEN_PREFIX_STICKY_UNDROP + QWEN_PREFIX_WINDOW_CARRY, docs/tp4-exact-resume.md 5.2) on the PRODUCTION stack and checkpoint (docs/tp4-exact-end.md): ONE image (tp4-xe-1, a placeholder: B0 builds
it from the pushed tp4/exact-end head WITH the K64j-OQ graft), ONE window, the approved ship-4 profile and its two gate-only twins (make_xe_profiles.py), and the checkpoint production serves
(ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION, which the prefix gate now forwards: c2_prefix_gate.py --model-repo/--model-revision). The pack answers, in value order:
does the split-invariance attribution probe say WHY a resume off the 2048 grid is not byte-exact (SP1, no build), is a returning turn that resumes one chunk higher byte-identical to its cold twin on real multi-turn
text with every audit on, the carried drafter window included (XX), and how much sooner does it answer (the ABAB XC1 XL1 XC2 XL2, then the eight-seat replay XB1 XB2). Nothing here changes the approved profile,
its image or its cache; every result is UNQUALIFIED.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_ship4_profiles as ship4
import make_xe_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'XE'
IMAGE = 'tp4-xe-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP = ship4.SHIP                      # the approved ship-4 profile: the control of every timed pair
ARM = twins.ARM                        # its gate-only twin with the undrop and the window carry: the timed arm
AUDIT = twins.AUDIT                    # the ship-4 audited twin with the same switches and the carry's audit: the exactness arm
PROBE = 'split-attribute'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/exact-end window (exact resume P0 on the production stack; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/exact-end and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory, in the prefix gate's
# containers too (c2_prefix_gate.py reads every arm's log for the checkpoint lines and fails the arm without them); the keys are the same in every serving job of this pack, so each pair differs by the profile alone."""

CHECKPOINT_READ = ('CHECKPOINT READ (the prefix gate checks every arm\'s container log, the smoke step passes c2_smoke_check.py --expect-checkpoint): the log holds "[QWEN-C2] checkpoint %s@%s key=... '
                   'weight_cache=.../weights-ckpt/<key> base_cache=... (the base cache is not read or written)" once, a "weight cache ...: found" line (the cache the Swift window built; this pack never builds it) and '
                   'the worker\'s "[QWEN-C2] checkpoint bound ..." line; the arm\'s result line says "checkpoint %s@%s: read".' % (REPO, REVISION, REPO, REVISION[:12]))
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
SHAPE = ('the prefix gate\'s exact-resume plan (prefix_replay.scenario_exact_resume): three conversations of real repository text whose first prompts are about 9k, 33k and 120k tokens, then six returning tool turns '
         'each of 150, 400, 900, 1,700, 2,600 and 300 new tokens, so the tail a turn re-prefills crosses both sides of 2,048 - (P mod 2,048): the window carry engaged and not engaged')
WINDOW_LINES = ('the carry\'s lines: "[PINDIAG] prefix: window carry engaged undrop=1 carry=1" once, "... carried req= Q= P= rows=" for every carried admission and "... stored req= pos= rows= bytes= ms=" for every '
                'export, and, beside the tile-aligned window of this profile, "[PINDIAG] tap window aligned engaged ... parts=2" on a carried turn (the carried rows are the first aligned part: '
                'prefix_window_carry.Carried.align) with no "fell back" line')


def rows():
    out = []
    out.append(dict(name='B0-xe-build', mode='stop', minutes=40, actions='build', cards=True, plan=None, profile=None, probe=None, model=False, box=None, needs=(),
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/exact-end head (= tp4/swift, the ship-4 stack with the checkpoint selection, plus exact resume P0 and its tile-aligned window join): '
                         'the approved ship-4 profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. Every exact-resume switch is a gate instrument of the two twins and is off '
                         'in the baked profile, so the image serves exactly what production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report '
                         'with no problem line, its K64j-OQ graft line, the image id. A build, not a card job; no privileged builder container may be up and no card job may run during it. '
                         'Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-xe-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', cards=True, plan=None, profile=None, probe=None, model=False, box=None,
                    needs=('X0 <- B0',), extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host (the Swift window\'s W0 built it): "weight cache '
                         '...: found" is the read of every serving job below. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the '
                         'cutover pack\'s own steps.')))
    out.append(dict(name='SP1-split-attribution', mode='soft', minutes=30, actions='reset fabric', cards=True, plan=None, profile=None, probe=PROBE, model=False, box=40,
                    needs=('SP1 <- X0',), extra=[],
                    why=('SP1 (SOFT; the decision gate for everything that resumes OFF the 2048 grid, docs/tp4-exact-end.md 2 and 5): tp4_split_probe.py --attribute under the ship-4 candidate\'s own environment, '
                         'from the checkout, inside the window image (the probe needs no build: it uses only the image\'s model code, the C2 graft and the G1 stage; it may also run before B0 on any existing four-card image). '
                         'The model is the first real layers of the checkpoint up to and including the first full-attention layer; per synthetic prompt length (9,000 and 41,000) the arms A (cold), A2 (determinism control), C (split at a 2048 '
                         'multiple, the harness control), B+128, B+1024, B+1920 and Bend (splits at 128-aligned R inside a chunk, each segment in the bucket the engine would pick) and F+128, F+1024, F+1920 and Fend (the same splits with BOTH '
                         'segments forced into the 2048 bucket, the cold run\'s own program masked to the valid rows). READ the lines "SPLIT_PROBE attribution L= R= B+x=... F+x=... cause=...": cause=bucket (B differs, F identical) means the '
                         'segments\' PROGRAMS differ (another bucket is another matmul program config: a resume at R is exact only in the cold run\'s own bucket, which saves nothing over P0); cause=handoff (F differs) means the STORE '
                         'at R (the bf16 GDN state and the conv carry) or the mask is not the cold run\'s own, whatever the bucket (a kernel that emits the in-program state is the only exact way); cause=none on every split means '
                         'a 128-aligned resume IS byte-exact and the verdict line says IDENTICAL (then Option A of the doc needs no owner decision on exactness). DIFFERS is a valid measured answer (exit 2, the step ends red); '
                         'FAIL (exit 1) is no answer (a control differed, the mesh did not open, an error: read split-probe.json). A failed run needs a reset before the next hardware job. RUN WITH THE CI PAUSED '
                         '(the weight load is the long part).')))
    out.append(dict(name='XX-xe-exactness', mode='stop', minutes=120, actions='reset prefix', cards=True, plan='exact-resume', profile=AUDIT, probe=None, model=True, box=170,
                    needs=('XX <- X0',), extra=[],
                    why=('XX (EXACTNESS of exact resume P0 on the production stack, STOP): %s, the ship-4 audited twin with the undrop, the window carry and the carry\'s audit, on Swift. %s. Every turn is a cold run under a '
                         'fresh salt and the hit under the conversation\'s salt, compared token for token (strict), the oracle\'s Q strict (prefix_judge.Oracle with the undrop and the carry: a carried resume above P - 2048 '
                         'wherever the new content stays under 2048 - (P mod 2048)). READ: the plan PASS: every cold/hit pair IDENTICAL; every continuation\'s Q the oracle\'s (none above it; under it only with an eviction on record); '
                         'exact_resume_findings: the engaged line with undrop=1 carry=1, at least one carried admission, every window carry audit line exact=True (the carried rows read back as stored) and every cold/hit pair\'s two '
                         'window digests EQUAL (the hit\'s spliced drafter window is the cold twin\'s byte for byte); %s; every other audit line of the candidate exact=True; no program growth after the warm; %s '
                         'ANY difference is NO-GO. RUN WITH THE CI PAUSED. Duration (estimate): about 2 hours (the ~120k conversation\'s cold twins dominate: seven ~40 s cold prefills, and an audited 8x262k boot).'
                         % (AUDIT, SHAPE, WINDOW_LINES, CHECKPOINT_READ))))
    for index, (control, arm) in enumerate((('XC1', 'XL1'), ('XC2', 'XL2')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-control-%s' % (control, letter), mode='soft', minutes=45, actions='reset prefix', cards=True, plan='exact-resume-timed', profile=SHIP, probe=None, model=True, box=90,
                        needs=('XC1 <- XX',) if index == 1 else ('XC2 <- XL1',), extra=[],
                        why=('%s (TIMED, the CONTROL of the %s returning-turn ABAB pair): %s, the approved ship-4 profile on Swift, the stack production runs: a returning turn resumes at floor2048(P_prev) - 2048. The prefix gate\'s '
                             'exact-resume-timed plan: %s, no audit, no row digest. Every boot a fresh reset. %s %s'
                             % (control, 'first' if index == 1 else 'second', SHIP, SHAPE, ARC, CHECKPOINT_READ))))
        out.append(dict(name='%s-xe-%s' % (arm, letter), mode='soft', minutes=45, actions='reset prefix', cards=True, plan='exact-resume-timed', profile=ARM, probe=None, model=True, box=90,
                        needs=('XL1 <- XC1 XX',) if index == 1 else ('XL2 <- XC2',), extra=[],
                        why=('%s (TIMED, the ARM of the %s returning-turn ABAB pair): %s, the approved ship-4 profile with the gate marker in place of its owner waiver id, the undrop and the window carry, the same plan and '
                             'conversations as %s, on Swift. READ after the four boots of the pair (XC1 XL1 XC2 XL2): python3 scripts/ci/exact_resume_compare.py <XC1 results>/prefix/exact-resume-timed <XL1 ...> and the same for '
                             '(XC2, XL2): every paired hit IDENTICAL across the arms (both are exact against their cold twins, so the same messages give the same answers), the lever\'s Q at floor2048(P) or above where the control\'s '
                             'is floor2048(P) - 2048, and the TTFT delta per conversation (~9k, ~33k, ~120k). ESTIMATE to beat (docs/tp4-exact-end.md 4, modelled from the card medians): about -0.9 s of 2.1 s at 165k '
                             'for the typical returning turn (two 2048-row forwards become one); a pair whose delta is not negative beyond the control-to-control spread (XC1 against XC2) is no gain. GO needs both pairs negative '
                             'beyond the spread and the texts equal. %s %s %s' % (arm, 'first' if index == 1 else 'second', ARM, control, ARC, WINDOW_LINES, CHECKPOINT_READ))))
    out.append(dict(name='XB1-xe-busy-control', mode='soft', minutes=45, actions='reset prefix', cards=True, plan='agent-turns-prefix', profile=SHIP, probe=None, model=True, box=90,
                    needs=('XB1 <- XX',), extra=[],
                    why=('XB1 (OPTIONAL, TIMED, the CONTROL of the eight-seat replay): the prefix gate\'s agent-turns-prefix plan (prefix_replay.scenario_agent_turns: eight agents holding conversations on real text, the metering '
                         'shape) on %s, Swift. %s %s' % (SHIP, ARC, CHECKPOINT_READ))))
    out.append(dict(name='XB2-xe-busy-lever', mode='soft', minutes=45, actions='reset prefix', cards=True, plan='agent-turns-prefix', profile=ARM, probe=None, model=True, box=90,
                    needs=('XB2 <- XB1',), extra=[],
                    why=('XB2 (OPTIONAL, TIMED, the ARM of the eight-seat replay): the same plan on %s, Swift. READ: python3 scripts/ci/prefix_agent_turns.py <XB1 results> <XB2 results>: the turns IDENTICAL across the arms, the paired '
                         'continuation TTFT, the stall of the other seats. %s %s' % (ARM, ARC, CHECKPOINT_READ))))
    out.append(dict(name='Z-xe-reset', mode='soft', minutes=10, actions='status reset', cards=True, plan=None, profile=None, probe=None, model=False, box=None, needs=(), extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, '
                         'link measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    return out


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['probe']:
        lines.append('C2_FABRIC_PROBE=%s' % row['probe'])
    if row['plan']:
        lines += ['C2_PREFIX_PLAN=%s' % row['plan'], 'C2_PREFIX_PROFILE=%s' % row['profile'], 'C2_PREFIX_BASELINE=none']
    if row['model']:
        lines += ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]
    lines += row['extra']
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def render_order(table):
    total = sum(row['minutes'] for row in table)
    names = [row['name'] for row in table]
    cards = total - next(row['minutes'] for row in table if row['name'].startswith('B0'))

    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())

    blocks = (('GATES', 'B0 X0 SP1 XX'), ('RETURNING-TURN ABAB', 'XC1 XL1 XC2 XL2'), ('EIGHT-SEAT REPLAY (optional)', 'XB1 XB2'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/exact-end window (docs/tp4-exact-end.md; scripts/ci/make_xe_jobs.py): exact resume P0 (QWEN_PREFIX_STICKY_UNDROP=1 + QWEN_PREFIX_WINDOW_CARRY=1) on the approved ship-4 profile %s and its two gate-only twins "
        "(make_xe_profiles.py), the checkpoint production serves (%s, revision %s), ONE image (%s, a placeholder: B0 builds it from the pushed tp4/exact-end head WITH the K64j-OQ graft) and ONE window."
        % (SHIP, REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether a returning turn should resume one 2048 chunk higher than today. Today it resumes at floor2048(P_prev) - 2048 and re-prefills 2,048 + (P_prev mod 2,048) tokens it "
        "already processed (about 3,070 on average; production: 2.95 million tokens over 996 turns). P0 resumes at floor2048(P_prev), the cold prefill's own last 2048 boundary: the same programs on the same bytes, so a hit equals its cold twin "
        "(E0), and the drafter window rows below it are carried from the turn that computed them. XX is the exactness gate (every audit on, every pair against its cold twin, the window digests); XC1 XL1 XC2 XL2 time it. Every result is UNQUALIFIED.",
        "# WHAT IS NEW ON THE DEVICE: nothing the cold prefill does not already run (the same chunk programs at the same absolute positions); the carried rows are uploaded once per carried turn and join the tile-aligned window as its first part. "
        "WHAT IS NEW IN THE HARNESS: the prefix gate serves the Swift checkpoint (--model-repo/--model-revision) and reads every arm's log for it.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
        "first boot) has built the keyed weight cache, so every serving job here logs \"weight cache ...: found\". This pack builds nothing of it." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO SP1 XB2",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE SP1 and restores them AFTER XB2, i.e. around "
        "every weight load, every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs the prefix gate or c2_smoke_check clean and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   SP1  'SPLIT_PROBE verdict=IDENTICAL' (exit 0): a 128-aligned resume is byte-exact; 'verdict=DIFFERS' (exit 2, a valid measured answer) plus the 'SPLIT_PROBE attribution' lines: cause=bucket or cause=handoff per split (the doc's 2); "
        "'verdict=FAIL' is no answer. Soft: P0 does not depend on it.",
        "#   XX   (STOP) the exact-resume plan PASS: every cold/hit pair IDENTICAL, the oracle's Q strict, at least one carried admission, every window audit exact=True and every pair's window digests EQUAL, the checkpoint read. ANY difference is NO-GO.",
        "#   XC1 XL1 XC2 XL2   exact_resume_compare.py (XC1, XL1) and (XC2, XL2): the hits IDENTICAL across the arms, the lever's Q one chunk higher where expected, the TTFT delta per conversation against the control-to-control spread (XC1 against XC2). "
        "GO needs both pairs negative beyond the spread and the texts equal.",
        "#   XB1 XB2  (optional) prefix_agent_turns.py: the turns IDENTICAL across the arms, the paired continuation TTFT, the stall of the other seats.",
        "# NO-GO for P0: any pair that differs (XX), a window digest that differs, an audit line with exact=False, a Q above the oracle's, any program compiled after the warm, a lever answer that differs from the control's, a missing checkpoint line. "
        "Adopting is the owner's decision: the switches are gate instruments today (serving_c2_contract.exact_resume_problems) and a ship profile needs an owner traffic waiver naming them (docs/tp4-exact-end.md 7).",
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
            sys.stderr.write('%s/%s is not what make_xe_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
