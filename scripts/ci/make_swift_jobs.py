"""The Swift evaluation window pack (scripts/ci/references/fusion-jobs/SW), generated (stdlib only).

    python3 scripts/ci/make_swift_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_swift_jobs.py --check   exit 1 if the checked-in pack is not what this generates

ONE image (tp4-swift-1, a placeholder: B0 builds it from the pushed tp4/swift head WITH the K64j-OQ graft), ONE window, ONE serving profile (the approved ship-4 profile, scripts/ci/fusion-wp/S4.json),
TWO checkpoints of the same architecture: the base (Qwen/Qwen3.8-27B, what production serves) and the RL post-train ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys C2_MODEL_REPO and
C2_MODEL_REVISION (qwen_c2_checkpoint.py, docs/tp4-swift.md). The pack answers, in value order: does it boot on a weight cache of its own (W0, SW1), what does it answer on the coding set the base
answers too (SW1, SB1: pass counts, thinking tokens, answer tokens), how well does the base's DFlash drafter accept its text (T1, T2: tau), how fast does it decode on the ship profile (C1 L1 C2 L2) and is the
fast path still exact on its weights (SA1). Nothing here changes the base's profile, alias or cache.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_fusion_profiles as fusion
import make_ship4_jobs as ship4
import make_ship4_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'SW'
IMAGE = 'tp4-swift-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP = twins.SHIP                      # the approved ship-4 profile: the profile of every smoke job here
AUDIT = twins.AUDIT                    # its audited twin (SA1)
PRODUCTION = fusion.PARENT
SMOKE_TESTS = 'warmup,coding,prefill_ladder_solo,concurrent8_steady,coding_set'
FIRST_BOOT_TESTS = 'warmup,concurrent8_steady'
DECODE_TESTS = ship4.GRID_TESTS        # warmup,concurrent8_steady,concurrent8_code_32k_long
TAULAB_PROFILE = 'c2-packed-tp4'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the Swift evaluation window (tp4/swift; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/swift and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint with its own converted-weight directory (never the base's); a job without them is the base,
# byte for byte what production serves."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: created|found" line, and the worker\'s "[QWEN-C2] checkpoint bound ..." line; the models answer names %s.' % (REPO, REVISION, SERVED))
BASE_READ = ('BASE READ: the container log holds NO "[QWEN-C2] checkpoint" line and the smoke check was run without --expect-checkpoint: this is the stack production runs, on the base\'s own warm cache.')
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the checkpoint.')
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
MODEL = ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/swift head (= tp4/ship-4 plus the checkpoint selection): the approved ship-4 profile %s baked as the serving default with a session cap of 8, '
                         'plus C2_BAKE_GRAFT=K64j-OQ. The image serves the BASE unless a job names a checkpoint; qwen_c2_checkpoint.py and qwen_c2_checkpoints.json are in it (the overlay manifest). READ: "baking the serving '
                         'default %s with a session cap of 8", the provenance report with no problem line, its K64j-OQ graft line, the image id. A build, not a card job; no privileged builder container may be up and no card '
                         'job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. Also before W0, by the operator: the Swift weights are on the card host whole '
                         '(python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s [--hash-shards]: "ok", exit 0) and the hub filesystem has at least 80 GB free '
                         '(the Swift converted weights are about as large as the base\'s, tens of GB, and W0 writes them). Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and '
                         'restores production after Z with the cutover pack\'s own steps.' % (REPO, REVISION))))
    out.append(dict(name='W0-swift-first-boot', mode='stop', minutes=60, actions='reset smoke', profile=SHIP, tests=FIRST_BOOT_TESTS, telemetry=True, box=150, needs=('W0 <- X0',), model=True, extra=[],
                    why=('W0 (STOP): the FIRST BOOT of Swift, which builds its converted-weight cache. The cache is keyed by the checkpoint (TT_CACHE_PATH becomes <base path>-ckpt/<repo>--<revision>), so it starts EMPTY: every tensor '
                         'is converted from the checkpoint\'s safetensors (bf16 to the TP4 shards in bfp8 and bfp4, the packed MLP blocks) and written, then the engines are built and the traces captured. The base\'s cache '
                         'is neither read nor written. ESTIMATE of the conversion: 10 to 30 minutes on top of the warm boot (the base\'s warm boot is 3 to 14 minutes by load; its converted weights are about 30 GB; the first '
                         'cold four-card boot of the TP4 stack was ready after 376 s on the general path), so 20 to 45 minutes in all; the BOX is 150 minutes and the smoke step\'s own readiness wait is 80 minutes: a conversion '
                         'that outlasts it kills the container in the middle of a write, and the next boot then logs "its first boot never completed" for the directory (delete <hub>/.qwen-c2/weights-ckpt/<key> and re-run W0). '
                         'RUN WITH THE CI PAUSED (a model load under CI load has hit the platform\'s load timeout). Tests: warmup and concurrent8_steady (the smoke check requires it on this profile): the first multi-user rounds '
                         'on the new weights. READ: %s "ready=1 after N s" in the step log: RECORD N, it is the cold load; the warm one is SW1\'s, and it is the number a production deploy of Swift has to fit under the platform\'s '
                         'bounded load attempt (about 13.5 minutes). "[QWEN-C2] weight cache complete (first boot of this directory)" once. The octo rounds of concurrent8_steady, no stall, no "audit mismatch". Any refusal line '
                         '("cannot serve the asked checkpoint") means the snapshot is not the base\'s in config/tokenizer/template or is incomplete: read it, fix the download, do NOT work around it. %s'
                         % (CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='SW1-swift-smoke', mode='stop', minutes=70, actions='reset smoke', profile=SHIP, tests=SMOKE_TESTS, telemetry=True, box=110, needs=('SW1 <- W0',), model=True, extra=[],
                    why=('SW1 (STOP): the SMOKE of Swift on the approved ship-4 profile, on the warm cache W0 built. warmup; coding (a Python rewrite of a shell script); prefill_ladder_solo (the solo prefill at about 4k, 32k, 128k and '
                         '254k tokens: the long-context rung is the nearest this stack has to a needle test, and the smoke check reads its text for garbage); concurrent8_steady (eight users, the octo rounds); coding_set (twenty '
                         'self-checking Python problems, thinking on, greedy as the request contract forces, eight at a time, 12,288 tokens at most each; the answers are saved whole as coding-set-<profile>.json in the results). %s READ: '
                         'c2_smoke_check clean (no garbage text, no stall, the steady rule), the warm "ready=1 after N s" recorded (against W0\'s cold one). The coding set is READ ON A CPU HOST with SB1\'s: python3 '
                         'scripts/ci/swift_coding_report.py --run base=<SB1 coding-set-<profile>.json> --run swift=<this job\'s coding-set-<profile>.json> (it runs model-written code: a disposable host only). %s'
                         % (CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='SB1-base-smoke', mode='soft', minutes=70, actions='reset smoke', profile=SHIP, tests=SMOKE_TESTS, telemetry=True, box=110, needs=('SB1 <- X0',), model=False, extra=[],
                    why=('SB1 (the BASE CONTROL of SW1): the same profile, the same image and the same five tests with NO checkpoint keys: the base checkpoint, as production serves it. It is the reference of SW1\'s coding set '
                         '(the same twenty problems, the same budget, the same greedy decoding), of its solo prefill ladder (python3 scripts/ci/prefill_ladder_report.py <SB1 smoke log> --second <SW1 smoke log>: the TTFT ratio '
                         'per rung, Swift over base: the weights are the same shape, so a ratio away from 1 would be the clocks or the load) and of its eight-user steady window. %s %s'
                         % (BASE_READ, TELEMETRY))))
    out.append(dict(name='T1-tau-base', mode='soft', minutes=90, actions='reset taulab', profile=None, tests=None, telemetry=False, box=None, needs=('T1 <- X0',), model=False,
                    extra=['C2_TAULAB_PROFILE=%s' % TAULAB_PROFILE, 'C2_TAULAB_ARMS=A3 A1 A2', 'C2_TAULAB_DEADLINE=85', 'C2_TAULAB_IN_FLIGHT=8'],
                    why=('T1 (the BASE tau, control of T2): the W-T1 tau lab (c2_tau_lab.py) on the base: the served DFlash drafter (the base\'s, unchanged) proposing for the base\'s own greedy answers to coding-agent turns, thinking on. '
                         'Arms A3 (the calibration: the lanes\' 4 x 4k and 4 x 32k prompts, thinking OFF; it must reproduce the served drafter\'s own reference within +/-7%%, which says this stack still measures what it measured), '
                         'A1 (public coding-agent turns) and A2 (the scrubbed own-session set). The data directory is rig-local (C2_TAULAB_DATA, default as in the earlier tau jobs, checked against its manifest before the container starts); the artifact gets only taulab/tau-lab-summary.json, aggregates. The profile is %s (the lab\'s own tested profile: the drafter proposes 15 tokens per round; the octo block commits '
                         'at most 8 and would change tau by construction, so tau is read on the profile without it). READ: "[TAULAB-REPORT] ... complete=True", the A3 verdict PASS. Kill rule: a FAIL of A3 voids T2 (the stack, '
                         'not the checkpoint, moved tau). Duration (estimate): 90 minutes, the load included.' % TAULAB_PROFILE)))
    out.append(dict(name='T2-tau-swift', mode='soft', minutes=75, actions='reset taulab', profile=None, tests=None, telemetry=False, box=None, needs=('T2 <- W0',), model=True,
                    extra=['C2_TAULAB_PROFILE=%s' % TAULAB_PROFILE, 'C2_TAULAB_ARMS=A1 A2', 'C2_TAULAB_DEADLINE=70', 'C2_TAULAB_IN_FLIGHT=8'],
                    why=('T2 (the SWIFT tau): the same lab, image, profile, drafter (the BASE\'s DFlash drafter: its acceptance on Swift is what this measures), prompts and arms A1 and A2 on Swift (A3 is the base stack\'s calibration '
                         'against a reference made on the base; it is left out). The lab refuses the arm, before anything is sent, when the container\'s log does not show the checkpoint lines (the keyed cache and the bound weights). '
                         'READ: "[TAULAB-REPORT] ... complete=True" and the container-log excerpt the step prints ("[QWEN-C2] checkpoint ..."). THE COMPARISON (a CPU host): python3 scripts/ci/swift_tau_compare.py --base '
                         '<T1 artifact>/taulab/tau-lab-summary.json --other <T2 artifact>/taulab/tau-lab-summary.json: the pooled tau and its interval, each arm, each context bucket and each draft position, Swift over base; the verdict is '
                         'LOWER, INDISTINGUISHABLE or HIGHER on the intervals. The prompts are the same and the answers are not, so this is an unpaired read (drafter_pair_report.py, which pairs identical texts, does not apply). '
                         'Duration (estimate): 75 minutes, the load included.')))
    for index, (control, arm) in enumerate((('C1', 'L1'), ('C2', 'L2')), 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-base-%s' % (control, letter), mode='soft', minutes=40, actions='reset smoke', profile=SHIP, tests=DECODE_TESTS, telemetry=True, box=70,
                        needs=('C1 <- X0',) if index == 1 else ('C2 <- L1',), model=False, extra=[],
                        why=('%s (TIMED, the BASE of the %s decode ABAB pair): the approved ship-4 profile %s on the base: warmup, concurrent8_steady (eight users live) and concurrent8_code_32k_long (eight real-code prompts of about 32k '
                             'tokens with 3,000-token budgets, ignore_eos, that outlast the prefill queue so that eight seats decode together at depth). Every boot a fresh reset. %s %s %s'
                             % (control, 'first' if index == 1 else 'second', SHIP, ARC, BASE_READ, TELEMETRY))))
        out.append(dict(name='%s-swift-%s' % (arm, letter), mode='soft', minutes=40, actions='reset smoke', profile=SHIP, tests=DECODE_TESTS, telemetry=True, box=70,
                        needs=('L1 <- W0 C1',) if index == 1 else ('L2 <- C2',), model=True, extra=[],
                        why=('%s (TIMED, SWIFT of the %s decode ABAB pair): the same profile and tests as %s, on Swift. %s %s READ: c2_smoke_check clean AND python3 scripts/ci/octo_judge.py --container-log <container log> --profile %s '
                             'clean (the octo rounds ran on the new weights). Then, after %s: python3 scripts/ci/octo3_report.py --control-a <C1 smoke> --arm-a <L1 smoke> --control-b <C2 smoke> --arm-b <L2 smoke> --arm-container <L1 container> '
                             '--arm-container <L2 container> --profile %s: per test (concurrent8_steady, concurrent8_code_32k_long) the aggregate of the all-live window of Swift against the base\'s in BOTH pairs and the floors; here GAIN / PARITY / LOSS '
                             'read Swift over base. The texts DIFFER by construction (another checkpoint: do not run octo_compare.py), and so do the lengths of the answers and the drafter\'s acceptance: the rate moves with tau (T1, T2), '
                             'not with the engine, which is the same. %s'
                             % (arm, 'first' if index == 1 else 'second', control, ARC, CHECKPOINT_READ, SHIP, 'the four boots of the pair (C1 L1 C2 L2)', SHIP, TELEMETRY))))
    out.append(dict(name='SA1-swift-audited-attach', mode='soft', minutes=110, actions='reset smoke', profile=AUDIT, tests=ship4.AUDIT_TESTS, telemetry=True, box=150, needs=('SA1 <- SW1',), model=True, extra=[],
                    why=('SA1 (EXACTNESS of the fast path on Swift\'s weights, SOFT, the lowest value of the window: cut it first): %s, the candidate with EVERY audit on (the ship-4 audited attach, its A1), on Swift. The fast '
                         'path\'s exactness evidence (the packed-any records, the W2, engine reuse and fusion audits) was made on the base\'s weights; the engine is the same code on tensors of the same shape, so the argument carries over, '
                         'and this job is the card check of it: every audit compares the served composition against the sequential one on device and logs exact=True or a mismatch. warmup, coding, parked_turns, the solo prefill '
                         'ladder and concurrent8_steady. An audited 8x262k profile costs about 8.4 minutes per request. RUN WITH THE CI PAUSED. READ as the ship-4 pack\'s A1 (references/fusion-jobs/ship4/A1-s4-audited-attach.env): every '
                         'audit line exact=True, no "audit mismatch", no "fell back", no stall, plus %s' % (AUDIT, CHECKPOINT_READ))))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, '
                         'link measurement, topology republish, agent restart, deploy and the post-checks. Production is on the BASE and nothing in this window touched its image, profile, alias or cache.')))
    return out


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
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


def render_order(table):
    total = sum(row['minutes'] for row in table)
    names = [row['name'] for row in table]
    cards = total - next(row['minutes'] for row in table if row['name'].startswith('B0'))

    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())

    blocks = (('GATES', 'B0 X0 W0 SW1'), ('QUALITY', 'SB1'), ('ACCEPTANCE', 'T1 T2'), ('DECODE ABAB', 'C1 L1 C2 L2'), ('EXACTNESS', 'SA1'), ('END', 'Z'))
    lines = [
        "# The order of the Swift evaluation window (docs/tp4-swift.md; scripts/ci/make_swift_jobs.py): the RL post-train %s (revision %s) against the base on the approved ship-4 profile %s, ONE image (%s, a placeholder: "
        "B0 builds it from the pushed tp4/swift head WITH the K64j-OQ graft) and ONE window. The checkpoint is selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION; its converted weights have a directory of "
        "their own (qwen_c2_checkpoint.py), so the base's warm cache, image, profile and alias are untouched." % (REPO, REVISION[:12], SHIP, IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether Swift is worth a production cutover. The model card's claims are fewer thinking tokens and better coding scores at temperature 1.0; this stack serves greedy decoding only (the "
        "request contract), so the claim has to be re-read under greedy: the coding set (SW1 against SB1) counts passes and the thinking and answer tokens of each model; the tau pair (T1, T2) says how well the base's drafter accepts Swift's text (a lower tau "
        "is a slower decode, whatever the weights are worth); the decode ABAB (C1 L1 C2 L2) measures the rate on the ship profile; SA1 checks the fast path's exactness on the new weights. Every result is UNQUALIFIED.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST BEFORE W0: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0 (add --hash-shards to hash the 55 GB against the hub's own sha256: minutes of disk reads, "
        "an operator's choice, off the cards' clock). The hub filesystem has at least 80 GB free. The licence of the checkpoint is the owner's decision (docs/tp4-swift.md), not this window's." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO W0 SA1",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE W0 and restores them AFTER SA1, i.e. around "
        "every Swift boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; qwen_c2_checkpoint.py check ok; 80 GB free.",
        "#   W0   (STOP) the checkpoint lines (c2_smoke_check --expect-checkpoint), the cache created, 'weight cache complete', the cold 'ready=1 after N s' RECORDED. A refusal line is the download's or the catalog's, never worked around.",
        "#   SW1  (STOP) the same lines, the warm 'ready=1 after N s' recorded, c2_smoke_check clean, coding_set saved (20 problems, 0 errors).",
        "#   SB1  no checkpoint line at all (the base); the same tests; coding_set saved. CPU side: swift_coding_report.py --run base=<SB1 coding-set-*.json> --run swift=<SW1 coding-set-*.json>; prefill_ladder_report.py <SB1 smoke log> --second <SW1 smoke log>.",
        "#   T1 T2   the lab's report complete=True (T1: the A3 calibration PASS); CPU side: swift_tau_compare.py --base <T1 summary> --other <T2 summary> (the drafter is the base's in both).",
        "#   C1 L1 C2 L2   octo3_report.py (the all-live window of each test, both pairs, the floors; Swift over base) and octo_judge.py on both Swift logs. NO text comparison: the checkpoints answer differently by construction.",
        "#   SA1  every audit line exact=True, no 'audit mismatch', no 'fell back', the checkpoint lines. Any mismatch is NO-GO for Swift on this stack.",
        "# NO-GO for Swift on the ship-4 stack: a refusal at W0 or SW1 (the snapshot is not the base's in config, tokenizer or chat template, or is incomplete), garbage text, a stall, an audit mismatch (SA1), a coding set that errors. "
        "A LOWER tau or a LOSS in the decode pair is a finding, not a failure; weigh it against the coding set. Adopting is the owner's decision.",
        "# PRODUCTION LATER (not this window): a Swift deployment needs (1) its cache built beforehand (W0), (2) QWEN_C2_MODEL_REPO and QWEN_C2_MODEL_REVISION in the serving image's environment (a thin layer's ENV lines; the node agent forwards nothing), "
        "(3) its own platform model id (docs/tp4-swift.md: alias plan) and (4) the replay job taught the new served name (C2_REPLAY_SERVED_MODEL; the replay's model pin is the base's).",
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
            sys.stderr.write('%s/%s is not what make_swift_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
