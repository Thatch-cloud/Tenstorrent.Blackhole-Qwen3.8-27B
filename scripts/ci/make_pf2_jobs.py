"""The tp4/prefill-2 window pack (scripts/ci/references/fusion-jobs/PF2), generated (stdlib only).

    python3 scripts/ci/make_pf2_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_pf2_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The device-side hand-off of the final step's GDN state (QWEN_FAST_PREFILL_HANDOFF, docs/tp4-prefill-2.md) on the production checkpoint: ONE image (tp4-pf2-1, a placeholder: B0 builds it from the pushed
tp4/prefill-2 head WITH the K64j-OQ graft), ONE window, ONE serving profile (the approved ship-4 profile) and its three gate-only twins (make_pf2_profiles.py), the checkpoint production serves
(ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION: qwen_c2_checkpoint.py) in every smoke job. The pack answers, in value order: is the hand-off byte-exact on
the audited twin, on cold prompts and on returning turns at 128k and 254k (A1, AP1), and how many milliseconds does a final step save (T1-T4 on the smoke shapes, R1-R2 on the returning turns), read
from the '[PF2] handoff' line both arms print. Nothing here changes the approved profile, its image or its cache; every result is UNQUALIFIED.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_pf2_profiles as twins
import make_ship4_profiles as ship4

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'PF2'
IMAGE = 'tp4-pf2-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP = ship4.SHIP                      # the approved ship-4 profile: what production serves
CONTROL = twins.CONTROL                # its gate-only twin with QWEN_FAST_PREFILL_HANDOFF=0: the control of every timed pair
ARM = twins.ARM                        # ... with =1: the timed arm
AUDIT = twins.AUDIT                    # the ship-4 audited twin with =1 and its audit: the audited attach arm
SMOKE_TESTS = 'warmup,coding,parked_turns,prefill_ladder_solo,prefill_few_decoders,concurrent8_steady'      # (the eight-seat profile's smoke rule requires concurrent8_steady)
TIMED_TESTS = SMOKE_TESTS

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/prefill-2 window (the device-side hand-off of the final step's GDN state; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/prefill-2 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a smoke job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the same
# in every smoke job of this pack, so each pair differs by the profile alone. The prefix steps (AP1, R1, R2) do not forward the selection into their container and serve the base checkpoint: the hand-off moves
# the GDN state of a finished prefill, whose shape and bytes the checkpoint does not change."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: found" line (the cache the Swift window built; this pack never builds it) and the worker\'s "[QWEN-C2] checkpoint bound ..." line; '
                   'the models answer names %s.' % (REPO, REVISION[:12], SERVED))
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
SHAPES = ('warmup; coding (one user); parked_turns (eight seats, forty closed-loop turns of 4k to 24k tokens: forty final steps, each a hand-off beside seven decoding seats); prefill_ladder_solo (one request at a time '
          'at about 4k, 32k, 128k and 254k tokens: four final steps); prefill_few_decoders (one decoding seat beside a 120k-token arrival and two beside a 254k one: two more final steps, at 120k and 254k of '
          'context, beside seats that wait for them); concurrent8_steady (eight users: eight final steps, then the octo rounds, which the lever never touches)')
PF2_LINES = ('the lines in the container log: "[PINDIAG] lever N route installed: route=1 audit=0 merged=COLD,CHECKPOINT,SCRATCH,PARKED handoff=%s" once; on the arms "[PINDIAG] prefill handoff engaged mode=device layers=48 clones=240" '
             'once and one "[PF2] handoff mode=device ms=.. read_ms=0.0 clone_ms=.. write_ms=.. layers=48" per final step; on the control one "[PF2] handoff mode=host ms=.. read_ms=.. write_ms=.. layers=48" per final step and no '
             '"engaged" line; no "fell back" line and no program growth after the warm anywhere')


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, prefix=None,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/prefill-2 head (= tp4/swift, the ship-4 stack with the checkpoint selection, plus prefill_handoff.py and the hand-off in the Lever N route): '
                         'the approved ship-4 profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. The flags are off in the baked profile (the route does not even import the module), so the image '
                         'serves exactly what production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, its K64j-OQ graft '
                         'line, the image id. A build, not a card job; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, prefix=None, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host (the Swift window\'s W0 built it): "weight cache '
                         '...: found" is the read of every smoke job below. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the '
                         'cutover pack\'s own steps.')))
    out.append(dict(name='A1-pf2-audited-attach', mode='stop', minutes=100, actions='reset smoke', profile=AUDIT, tests=SMOKE_TESTS, telemetry=True, box=150, needs=('A1 <- X0',), model=True, prefix=None, extra=[],
                    why=('A1 (EXACTNESS of the device-side hand-off on cold prompts, STOP): %s, the ship-4 audited twin (every audit of the candidate) with QWEN_FAST_PREFILL_HANDOFF=1 and QWEN_FAST_PREFILL_HANDOFF_AUDIT=1, on Swift. '
                         'Each of the first 8 final steps and every 24th after reads the whole batched decode state of all 48 GDN layers before and after the slot write and requires (a) the slot\'s rows equal to the scratch\'s host copy '
                         'byte for byte on every chip and (b) every other row unchanged. %s. READ: c2_smoke_check clean; %s; "[PINDIAG] prefill handoff audit n=<k> exact=True slot=.. layers=48 tensors_checked=.." for n=1..8 and then every 24th '
                         'final step, NO "audit mismatch", NO "fell back"; the memory ledger\'s free DRAM per chip after the attach (RECORD it: the device path holds the clones of the 240 scratch tensors, about 70 MB a chip (48 layers x 1.4 MB with the fp32 recurrent state), until the last layer is written, against the '
                         'roughly 4 GB the traffic arm keeps free after the first round). ANY failure is NO-GO for the lever and halts the window. %s %s RUN WITH THE CI PAUSED.'
                         % (AUDIT, SHAPES, PF2_LINES % 'device audit', CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='AP1-pf2-audited-returning', mode='stop', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('AP1 <- A1',), model=False,
                    prefix=dict(plan='tier-timed', profile=AUDIT), extra=[],
                    why=('AP1 (EXACTNESS of the hand-off on RETURNING turns, STOP): c2_prefix_gate.py --plan tier-timed on %s: the cold prefill, the device-resident hit and the host-tier restore at about 32k, 128k and 254k tokens, '
                         'every final step of them audited (the first eight final steps all are). The returning turns are the production shape: a hit resumes on a 2,048-token boundary, its first step is a CHECKPOINT restore, '
                         'its last a hand-off at 125k and 245k of context. READ: the plan verdict PASS (its own exactness reads: the returning answers equal the cold twins\'); "[PINDIAG] prefill handoff audit n=<k> exact=True" for every '
                         'final step it runs and NO "audit mismatch", NO "fell back". ANY failure is NO-GO for the lever. This step does not forward the Swift selection: it serves the base checkpoint. RUN WITH THE CI PAUSED.' % AUDIT)))
    pair = (('T1', 'T2'), ('T3', 'T4'))
    for index, (control, arm) in enumerate(pair, 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-ctl-%s' % (control, letter), mode='soft', minutes=55, actions='reset smoke', profile=CONTROL, tests=TIMED_TESTS, telemetry=True, box=90,
                        needs=('T1 <- A1 AP1',) if index == 1 else ('T3 <- T2',), model=True, prefix=None, extra=[],
                        why=('%s (TIMED, the CONTROL of the %s smoke pair): %s, the approved ship-4 profile with the gate marker for its waiver id and QWEN_FAST_PREFILL_HANDOFF=0: the host path of the final step, byte for byte '
                             'production\'s, which also prints the "[PF2] handoff mode=host" timing line, on Swift. Tests: %s. Every boot a fresh reset. %s %s %s'
                             % (control, 'first' if index == 1 else 'second', CONTROL, SHAPES, ARC, CHECKPOINT_READ, TELEMETRY))))
        out.append(dict(name='%s-pf2-%s' % (arm, letter), mode='soft', minutes=55, actions='reset smoke', profile=ARM, tests=TIMED_TESTS, telemetry=True, box=90,
                        needs=('T2 <- T1',) if index == 1 else ('T4 <- T3',), model=True, prefix=None, extra=[],
                        why=('%s (TIMED, the ARM of the %s smoke pair): %s, the control with QWEN_FAST_PREFILL_HANDOFF=1 and nothing else, the same tests as %s, on Swift. %s READ: c2_smoke_check clean; %s. Then, after the four boots of the pair '
                             '(T1 T2 T3 T4): python3 -B scripts/ci/pf2_report.py --control <T1 container log> --control <T3 container log> --arm <T2 container log> --arm <T4 container log>: the hand-off milliseconds per final step '
                             'of the host path against the device path (count, mean, median, p90), the saving per final step and the route milliseconds of the final steps by context band; exit 0 needs the arm\'s engaged line, no fell-back '
                             'or mismatch line and a host-only control. The cold ladder\'s TTFTs: python3 -B scripts/ci/prefill_ladder_report.py <T1 results> --second <T2 results> (and the same for T3 T4): a ratio below 1 is the arm faster; '
                             'the answer text of every rung must be EQUAL between the arms (greedy, same checkpoint). The control-to-control floor is T1 against T3. %s %s'
                             % (arm, 'first' if index == 1 else 'second', ARM, control, ARC, PF2_LINES % 'device', CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='R1-ctl-returning', mode='soft', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('R1 <- AP1',), model=False,
                    prefix=dict(plan='tier-timed', profile=CONTROL),
                    why=('R1 (TIMED, the CONTROL of the returning-turn pair): c2_prefix_gate.py --plan tier-timed on %s: the host path of the final step at about 32k, 128k and 254k of context (a cold prefill, a device-resident hit, a host-tier '
                         'restore), with the "[PF2] handoff mode=host" timing line on every final step. %s NO CI load while it runs. This step serves the base checkpoint.' % (CONTROL, ARC)), extra=[]))
    out.append(dict(name='R2-pf2-returning', mode='soft', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('R2 <- R1',), model=False,
                    prefix=dict(plan='tier-timed', profile=ARM),
                    why=('R2 (TIMED, the ARM of the returning-turn pair): the same plan on %s: the device-side hand-off at the same sizes. READ: the plan verdict PASS (the returning answers equal the cold twins\') and, with R1: '
                         'python3 -B scripts/ci/pf2_report.py --control <R1 container log> --arm <R2 container log>: the saving per final step at 125k and 245k of context, where a returning turn\'s step is about 1.2 s and 2.7 s long. '
                         'There is one hit of each kind per size, so six final steps an arm: the figure is read for its size (a step\'s hand-off does not depend on the context), not for its spread. %s' % (ARM, ARC)), extra=[]))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, prefix=None, extra=[],
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
    if row.get('prefix'):
        lines += ['C2_PREFIX_PLAN=%s' % row['prefix']['plan'], 'C2_PREFIX_PROFILE=%s' % row['prefix']['profile'], 'C2_PREFIX_BASELINE=none']
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

    blocks = (('GATES', 'B0 X0 A1 AP1'), ('SMOKE ABAB', 'T1 T2 T3 T4'), ('RETURNING TURNS', 'R1 R2'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/prefill-2 window (docs/tp4-prefill-2.md; scripts/ci/make_pf2_jobs.py): the device-side hand-off of the final step's GDN state (QWEN_FAST_PREFILL_HANDOFF) on the approved ship-4 profile %s and "
        "its three gate-only twins (make_pf2_profiles.py), the checkpoint production serves (%s, revision %s) in every smoke job, ONE image (%s, a placeholder: B0 builds it from the pushed tp4/prefill-2 head WITH the K64j-OQ graft) and ONE window."
        % (SHIP, REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the last step of every split prompt should write the decode slot from device-side clones of the GDN scratch instead of reading the scratch to the host and uploading it "
        "back. Today the step pays 240 blocking reads and 240 uploads before 48 layers of slot writes; the PF1 device profile of the ship-4 stack measured 48.6 ms of idle device for the reads and 115.2 ms for the writes, 8.0 ms of it device "
        "kernels, on a profiled run (so the upper end of what production pays). The expected saving is 20 to 55 ms a final step, a few percent of a returning turn's 1.1 to 1.2 s step and about twice that once the first re-prefilled chunk is gone "
        "(tp4/exact-end); the expectation is arithmetic on that profile, not a measurement. A1 and AP1 are the exactness gates; T1-T4 and R1-R2 read the saving. Every result is UNQUALIFIED.",
        "# THE ONE NEW DEVICE FACT: the final step makes 240 device clones of the persistent GDN scratch (ttnn.clone, two program shapes, compiled by the route's warm before the traces) and hands them to the graft's own write_slot, which consumes them. "
        "No kernel, trace or buffer of the serving path is new; the scratch is never given to the consuming write. A1 is the first card to run it, with the audit on.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
        "first boot) has built the keyed weight cache, so every smoke job here logs \"weight cache ...: found\". This pack builds nothing of it." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO A1 R2",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER R2, i.e. around "
        "every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean where it is a smoke job, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) the hand-off engaged line once, every audit line exact=True (the first eight final steps and every 24th), no 'audit mismatch', no 'fell back', the memory ledger recorded. ANY failure is NO-GO and halts the window.",
        "#   AP1  (STOP) the tier-timed plan PASS on %s and every final step's audit exact=True at 125k and 245k of context." % AUDIT,
        "#   T1 T2 T3 T4   pf2_report.py (the hand-off milliseconds of both arms, both pairs, and the floor T1 against T3), prefill_ladder_report.py (the rungs' TTFT ratio and equal answers). GO: a SAVING in the hand-off line and no LOSS in a rung.",
        "#   R1 R2   pf2_report.py on the returning-turn pair: the saving at 125k and 245k of context; the plan verdict PASS on both.",
        "# NO-GO for the hand-off: any stall, any audit mismatch, any fallback line, a text difference against the control, a program compiled after the warm, an engaged line missing on an arm, a LOSS in the hand-off line. Adopting is the owner's decision: "
        "the flag is a gate instrument today and a ship profile needs an owner traffic waiver naming it (docs/tp4-prefill-2.md).",
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
            sys.stderr.write('%s/%s is not what make_pf2_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
