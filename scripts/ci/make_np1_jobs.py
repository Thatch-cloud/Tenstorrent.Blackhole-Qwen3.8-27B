"""The tp4/np1-fastlane window pack (scripts/ci/references/fusion-jobs/NP1), generated (stdlib only).

    python3 scripts/ci/make_np1_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_np1_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The lone-user padded round (QWEN_FAST_SOLO_PACKED=1, docs/tp4-np1-fastlane.md) on the production checkpoint: ONE image (tp4-np1-1, a placeholder: B0 builds it from the pushed tp4/np1-fastlane head WITH the
K64j-OQ graft), ONE window, ONE serving profile (the approved ship-4 profile) and its two gate-only twins (make_np1_profiles.py), the checkpoint production serves (ukisai/Swift-1.5-Qwen3.8-27b, selected by the job
keys C2_MODEL_REPO and C2_MODEL_REVISION: qwen_c2_checkpoint.py). The pack answers, in value order: does the lone-user round boot and stay exact on the audited twin, with every audit on (A1), and how much faster does a
lone user, a pair and a four-seat block decode with it than without (the ABAB C1 L1 C2 L2 at one, two and four live seats, with eight beside them as the regression row). Nothing here changes the approved profile, its image or
its cache; every result is UNQUALIFIED.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_np1_profiles as twins
import make_ship4_profiles as ship4

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'NP1'
IMAGE = 'tp4-np1-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP = ship4.SHIP                      # the approved ship-4 profile: the control of every timed pair
ARM = twins.ARM                        # its gate-only twin with QWEN_FAST_SOLO_PACKED=1: the timed arm
AUDIT = twins.AUDIT                    # the ship-4 audited twin with QWEN_FAST_SOLO_PACKED=1: the audited attach arm
TIMED_TESTS = 'warmup,np1_live1,np1_live2,np1_live4,concurrent8_steady'
AUDIT_TESTS = 'warmup,coding,np1_live2,concurrent5_split,concurrent8_drain,concurrent8_steady'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/np1-fastlane window (the lone-user padded round; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/np1-fastlane and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the same
# in every serving job of this pack, so each pair differs by the profile alone."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: found" line (the cache W0 of the Swift window built; this pack never builds it) and the worker\'s "[QWEN-C2] checkpoint bound ..." line; '
                   'the models answer names %s.' % (REPO, REVISION, SERVED))
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
SHAPES = ('np1_live1, np1_live2 and np1_live4 (one, two and four users started together on about 3,500-token real-code prompts, every answer to EXACTLY 2,000 tokens with ignore_eos, so the all-live window is long '
          'enough to read at either rate) and concurrent8_steady (eight users: the octo block, which the flag never touches, so a parity row and the boot-to-boot floor)')
NP1_LINES = ('the NP1 lines in the container log: "[NP1] kv scratch: Qwen36ForCausalLM.allocate_kv_cache allocates 1 block(s) beyond the pool vLLM sized (pool 16500 blocks, scratch ids 16500..16500)" once, '
             '"[SOLO-PACKED] admitted min_users=1 scratch_pages=16500 (gate only)" once and, for each of the two M3 blocks, "packed padded block admitted min_users=1 users=4 max_idle=4 ... scratch_pages=16500"; '
             'the octo block\'s own admitted line is unchanged (min_users=6, max_idle=2, no scratch pages)')


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/np1-fastlane head (= tp4/swift, the ship-4 stack with the checkpoint selection, plus idle_scratch.py and the lone-user padded round): the approved '
                         'ship-4 profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. The flag is a gate instrument of the two twins and is off in the baked profile, so the image '
                         'serves exactly what production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, its K64j-OQ graft '
                         'line, the image id. A build, not a card job; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host (the Swift window\'s W0 built it): "weight cache '
                         '...: found" is the read of every serving job below. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the '
                         'cutover pack\'s own steps.')))
    out.append(dict(name='A1-np1-audited-attach', mode='stop', minutes=120, actions='reset smoke', profile=AUDIT, tests=AUDIT_TESTS, telemetry=True, box=180, needs=('A1 <- X0',), model=True, extra=[],
                    why=('A1 (EXACTNESS of the lone-user padded round, STOP): %s, the ship-4 audited twin (every audit of the candidate: the K/V page writer\'s audit of every unit it writes, the verify T2 audit, the pre-stage diff and two-block pre-stage audits, the multi-user SDPA launch\'s audit, '
                         'the vglue, conv-gates and octo-lever audits, the host tier, the host read-backs SAMPLED) with QWEN_FAST_SOLO_PACKED=1, on Swift. The flag adds ONE block to the KV cache tensors (a scratch page past vLLM\'s pool) and lets a '
                         'lone live user (and a lone user of the other block) take the padded 64-row round with its third idle segment on that page. warmup; coding (ONE user for 1,500 tokens: every round a lone-user padded round, the '
                         'audited ones compare the served composition with the sequential one on the idle segments too); np1_live2 (a pair: two idle segments, page 0 alone, the flag changes nothing); concurrent5_split (four '
                         'users in block A and a fifth ALONE in block B, then a sixth: the split the placement leaves); concurrent8_drain (eight users whose budgets end one by one: every live count from 8 to 1, every split '
                         'of the two blocks); concurrent8_steady (the octo rounds, required by the smoke check on this profile). READ: octo_judge.py --container-log <container log> --profile %s clean (the solo rule: at least '
                         'one "packed padded round live=1" line, no "packed padded skipped live=1 eligible=1" line, beside the octo rules); %s; every audit line exact=True with entries above 0, in particular the K/V page '
                         'writer\'s audit on a round with idle=[...] of three segments (the unit that writes the scratch page); NO "audit mismatch", NO "fell back", NO "kv shared", NO "page0 ... refused" for an eligible lone user, no stall. The '
                         'memory ledger\'s free DRAM per chip after the attach: RECORD it (the scratch page is 557,056 bytes a chip). ANY failure is NO-GO for the lone-user round and halts the window. %s %s RUN WITH THE CI PAUSED.'
                         % (AUDIT, AUDIT, NP1_LINES, CHECKPOINT_READ, TELEMETRY))))
    for index, (control, arm) in enumerate((('C1', 'L1'), ('C2', 'L2')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-control-%s' % (control, letter), mode='soft', minutes=45, actions='reset smoke', profile=SHIP, tests=TIMED_TESTS, telemetry=True, box=80,
                        needs=('C1 <- A1',) if index == 1 else ('C2 <- L1',), model=True, extra=[],
                        why=('%s (TIMED, the CONTROL of the %s decode ABAB pair): the approved ship-4 profile %s on Swift, the stack production runs: a lone live user decodes on the 4-row per-request engine, two or three on the '
                             'padded 64-row block, four on the full block, six or more on the octo block. Tests: %s. Every boot a fresh reset. %s %s %s'
                             % (control, 'first' if index == 1 else 'second', SHIP, SHAPES, ARC, CHECKPOINT_READ, TELEMETRY))))
        out.append(dict(name='%s-np1-%s' % (arm, letter), mode='soft', minutes=45, actions='reset smoke', profile=ARM, tests=TIMED_TESTS, telemetry=True, box=80,
                        needs=('L1 <- C1',) if index == 1 else ('L2 <- C2',), model=True, extra=[],
                        why=('%s (TIMED, the ARM of the %s decode ABAB pair): %s, the approved ship-4 profile with the gate marker in place of its owner waiver id and QWEN_FAST_SOLO_PACKED=1, the same tests as %s, on Swift. %s '
                             'READ: c2_smoke_check clean AND python3 scripts/ci/octo_judge.py --container-log <container log> --profile %s clean (the solo rule beside the octo rules); %s. Then, after the four boots of the pair '
                             '(C1 L1 C2 L2): python3 scripts/ci/np1_report.py --control-a <C1 smoke> --arm-a <L1 smoke> --control-b <C2 smoke> --arm-b <L2 smoke> --arm-container <L1 container> --arm-container <L2 container> '
                             '--profile %s: per test the aggregate of the all-live window of the arm against the control\'s in BOTH pairs and the floors; GO needs a GAIN at np1_live1 and no LOSS elsewhere. The texts must be '
                             'EQUAL, user by user: python3 scripts/ci/octo_compare.py --control <C1 smoke> --octo <L1 smoke> --octo-container <L1 container> --profile %s (and the same for C2 L2): the same checkpoint, greedy, so '
                             'a difference is a token the lone-user round changed and is NO-GO. %s %s'
                             % (arm, 'first' if index == 1 else 'second', ARM, control, ARC, ARM, NP1_LINES, ARM, ARM, CHECKPOINT_READ, TELEMETRY))))
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

    blocks = (('GATES', 'B0 X0 A1'), ('DECODE ABAB', 'C1 L1 C2 L2'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/np1-fastlane window (docs/tp4-np1-fastlane.md; scripts/ci/make_np1_jobs.py): the lone-user padded round (QWEN_FAST_SOLO_PACKED=1) on the approved ship-4 profile %s and its two gate-only twins "
        "(make_np1_profiles.py), the checkpoint production serves (%s, revision %s), ONE image (%s, a placeholder: B0 builds it from the pushed tp4/np1-fastlane head WITH the K64j-OQ graft) and ONE window."
        % (SHIP, REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether a lone user (and a lone user of the other block) should decode on the padded 64-row block. Today a lone live user runs the 4-row per-request engine "
        "(83.9 ms a pass plus a 15.3 ms host gap, 3.70 tokens a round: ~37 tokens a second); two live users run the padded block (52.8 ms plus 19.6 ms, 13.56 tokens a round for the pair: ~94 a user). A1 is the exactness gate "
        "(every audit on, the lone user's rounds among them); C1 L1 C2 L2 time the arm against the production profile at one, two and four live seats with eight beside them. The expected gain at one seat is x2 to x3 and "
        "nothing at two or more; the expectation is the arithmetic above, not a measurement. Every result is UNQUALIFIED.",
        "# THE ONE NEW DEVICE FACT: the flag makes the KV cache tensors one block longer than vLLM's pool (a scratch page, 557,056 bytes a chip) and parks the third idle segment's K/V there. No program, trace or "
        "kernel is new; a block id past the pool in a page table is a value the K/V writers and the extent reader have never been given. A1 is the first card to give it to them, with the audits on.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
        "first boot) has built the keyed weight cache, so every serving job here logs \"weight cache ...: found\". This pack builds nothing of it." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO A1 L2",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER L2, i.e. around "
        "every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) octo_judge.py clean under %s (the solo rule: a padded round of a lone live user ran, none went to the per-request engines), the NP1 lines (scratch page allocated, admitted, two M3 blocks at max_idle=4), "
        "every audit line exact=True, no 'audit mismatch', no 'fell back', no 'kv shared', the memory ledger recorded. ANY failure is NO-GO and halts the window." % AUDIT,
        "#   C1 L1 C2 L2   np1_report.py (the all-live window of each test, both pairs, the floors) GO, octo_compare.py (every answer equal) and octo_judge.py on both arm logs. GO: a GAIN at np1_live1, no LOSS elsewhere, every text equal.",
        "# NO-GO for the lone-user round: any stall, any text difference against the control, any audit mismatch, any fallback line, a refusal at the attach (the cache not one block longer, a request block id past the pool), a lone-user "
        "round that never ran or an eligible one that went to the engines, a LOSS in a timed test. Adopting is the owner's decision: the flag is a gate instrument today and a ship profile needs an owner traffic waiver naming it (docs/tp4-np1-fastlane.md).",
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
            sys.stderr.write('%s/%s is not what make_np1_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
