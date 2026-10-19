"""The NP9 long-answer window pack (scripts/ci/references/fusion-jobs/NP9), generated (stdlib only).

    python3 scripts/ci/make_np9_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_np9_jobs.py --check   exit 1 if the checked-in pack is not what this generates

ONE image (tp4-np9-1, a placeholder: B0 builds it from the pushed tp4/np9-answer32k head WITH the K64j-OQ graft), the approved ship-4 profile (scripts/ci/fusion-wp/S4.json) on the Swift checkpoint, whose catalog
entry carries the 32,768-token output ceiling (qwen_c2_checkpoints.json, docs/tp4-np9-answer32k.md). The pack answers one question: does a seat run 30,000 tokens, alone and eight at once, at a short and at a long
prompt, on the cards, with the text unchanged by the ceiling and the KV reservation behaving as the arithmetic says. Nothing here changes the profile, the base or the image's defaults. Templates only: no tag is pushed.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_ship4_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'NP9'
IMAGE = 'tp4-np9-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SHIP = twins.SHIP                      # the approved ship-4 profile: the profile of every smoke job here
MODEL = ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]
SOLO_TESTS = 'warmup,concurrent8_steady,long_answer_prefix,long_answer_solo_4k,long_answer_solo_200k'
SHORT_TESTS = 'warmup,concurrent8_steady,long_answer8_4k'
DEEP_TESTS = 'warmup,concurrent8_steady,long_answer8_96k'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the NP9 long-answer window (tp4/np9-answer32k; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/np9-answer32k and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint with its own converted-weight directory (never the base's) and, since NP9, the output ceiling of its
# catalog entry (qwen_c2_checkpoints.json "request": output_budget 32768, default_max_tokens 16384). A job without them is the base: its ceiling is the profile's 16,384, so the long-answer tests would read as clamped."""

CEILING_READ = ('CEILING READ: the container log holds "[QWEN-C2] request output budget 32768 for this checkpoint (its catalog entry; the profile says 16384)" and "[QWEN-C2] request contract installed: greedy, max_tokens <= 32768" '
                '(c2_smoke_check requires the second whenever a long-answer test ran), the checkpoint lines (c2_smoke_check --expect-checkpoint) and no "kv reservation too large" line.')
ANSWER_READ = ('ANSWER READ (c2_smoke_check, long_answer_problems): every user of a long_answer test ends with finish "length" and exactly 30,000 tokens (ignore_eos, so the stop is the ceiling, not the model); fewer tokens at "length" '
               'is a server whose ceiling clamped the request.')
NO_MEMORY = ('MEMORY READ: the output ceiling allocates no verifier capture, proposal bucket or page table (docs/tp4-np9-answer32k.md), so the DRAM lines are the ship-4 profile\'s own: no "dram backstop refused", no "dram admission hold" '
             'that the same prompts would not have drawn at 16,384, no engine-build failure, no "audit mismatch", no stall.')
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. A 30,000-token decode is '
             'minutes at the board power limit: a rate that falls while AICLK falls is the clock, not the ceiling.')
ARC = 'RUN WITH THE CI PAUSED (a model load under CI load has hit the platform\'s load timeout; the decodes here run 7 to 30 minutes).'


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/np9-answer32k head (= tp4/swift plus the output ceiling): the approved ship-4 profile %s baked as the serving default with a session cap of 8, '
                         'plus C2_BAKE_GRAFT=K64j-OQ. The image serves the BASE unless a job names a checkpoint; qwen_c2_checkpoint.py and qwen_c2_checkpoints.json (whose Swift entry carries the output_budget) are in it '
                         '(the overlay manifest). READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, its K64j-OQ graft line, the image id. A build, not a card job; '
                         'no privileged builder container may be up and no card job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run (13x10 since the '
                         'firmware unlock), no other window driver alive. Also by the operator: Swift\'s converted weights are on the card host (the Swift window\'s W0 built them under a directory of their own; without them N1 is the '
                         'first boot and builds them, box 150 minutes). The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='N1-long-answer-solo', mode='stop', minutes=75, actions='reset smoke', profile=SHIP, tests=SOLO_TESTS, telemetry=True, box=150, needs=('N1 <- X0',), model=True, extra=[],
                    why=('N1 (STOP): ONE request at a time, 30,000 tokens, on Swift. warmup; concurrent8_steady (the profile\'s required eight-user check, which also proves the octo rounds still run); long_answer_prefix (one 4k real-code '
                         'prompt asked at max_tokens 16,384 and then 30,000, both ignore_eos: the long answer must begin with the short one but for its last 512 characters - greedy decoding does not change with the ceiling, the budget '
                         'enters a round only at the last few tokens); long_answer_solo_4k (a 4k prompt, 30,000 tokens, ignore_eos) and long_answer_solo_200k (a 200,000-token prompt as the server counts it, 30,000 tokens, ignore_eos: '
                         'the answer ends at position 230,000 and the request reserves 3,596 of the pool\'s 16,499 blocks). %s %s %s %s READ also: the solo decode rate is recorded, never gated; the stream never stalls for the '
                         'answer\'s 7 to 15 minutes (a gap over the profile\'s QWEN_FAST_LEVERN_MAX_DECODE_GAP_S, 8 s, is a fault). Stop rule: a request that ends short of 30,000 tokens, a prefix difference, a stall, any audit mismatch or a refusal line. '
                         '%s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, ARC, TELEMETRY))))
    out.append(dict(name='N2-long-answer8-short', mode='soft', minutes=70, actions='reset smoke', profile=SHIP, tests=SHORT_TESTS, telemetry=True, box=120, needs=('N2 <- N1',), model=True, extra=[],
                    why=('N2: EIGHT requests at once, 4k prompts, 30,000 tokens each (ignore_eos), on Swift: eight seats decode together for the whole answer (octo rounds at eight live, the engine-reuse rebinds, the page tables growing by '
                         '470 blocks each, the host KV tier idle). The KV reservation holds 535 blocks a seat, 4,280 of 16,499: nothing is held. %s %s %s READ also: the all-live window\'s per-seat rate (live4_tok_s) is recorded, never gated; '
                         'octo_judge.py --container-log <log> --profile %s is clean. %s %s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, SHIP, ARC, TELEMETRY))))
    out.append(dict(name='N3-long-answer8-deep', mode='soft', minutes=80, actions='reset smoke', profile=SHIP, tests=DEEP_TESTS, telemetry=True, box=150, needs=('N3 <- N1',), model=True, extra=[],
                    why=('N3: EIGHT requests at once, prompts of about 96,000 tokens (as the server counts them), 30,000 tokens each (ignore_eos), on Swift. The longest prompt at which eight 30,000-token requests are all resident: '
                         '8 x 1,971 = 15,768 of the 16,499 blocks (eight 30,000-token requests fit up to 101,872-token prompts, eight at the full 32,768 ceiling up to 99,104; docs/tp4-np9-answer32k.md), so no seat waits: the container log holds NO "kv reservation hold" '
                         'line (a hold here means the pool arithmetic is wrong). The eight prefills queue (about 26 s each), then eight seats decode at depth 96,000 to 126,000 together for 30,000 tokens. %s %s %s READ also: the '
                         'per-seat rate at depth is recorded against concurrent8_code_128k_long of the Swift window, never gated. %s %s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, ARC, TELEMETRY))))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, link '
                         'measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
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
    lines = [
        "# The order of the NP9 long-answer window (docs/tp4-np9-answer32k.md; scripts/ci/make_np9_jobs.py): Swift (%s, revision %s) on the approved ship-4 profile %s with the 32,768-token output ceiling of its catalog entry, "
        "ONE image (%s, a placeholder: B0 builds it from the pushed tp4/np9-answer32k head WITH the K64j-OQ graft) and ONE window. Nothing in the profile, the base or the image's defaults moves." % (REPO, REVISION[:12], SHIP, IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the 32,768-token ceiling goes to production. The CPU side is settled (docs/tp4-np9-answer32k.md: the ceiling allocates no device memory on the serving shape; it costs KV admission "
        "only); what only a card can show is that a seat runs 30,000 tokens, alone and eight at once, at a short and a long prompt, with the text unchanged by the ceiling. Every result is UNQUALIFIED.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
        "to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and its converted weights exist from the Swift window (otherwise N1 is the first boot, box 150)." % (REPO, REVISION),
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE N1 and restores them AFTER N3.",
        "# THE JOBS, in value order (cut the window at a block end): N1 is the whole question for one seat (the ceiling, the text, 200k); N2 and N3 are the eight-seat shapes. B0 is the WINDOW build.",
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, with the container log, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; Swift on the host.",
        "#   N1   (STOP) the ceiling lines (request output budget 32768; request contract installed ... <= 32768), the checkpoint lines, long_answer_prefix prefix_equal=True, every long_answer user finish length with 30000 tokens.",
        "#   N2   the same ceiling and answer rules at eight live; octo_judge clean; no hold line (4,280 of 16,499 blocks).",
        "#   N3   the same at 96,000-token prompts; NO 'kv reservation hold' line (15,768 of 16,499 blocks); the per-seat rate at depth recorded.",
        "# NO-GO for the 32,768 ceiling: a request that ends short of its 30,000 tokens, a prefix difference in long_answer_prefix, a stall, an audit mismatch, an engine-build or DRAM refusal that 16,384 would not have drawn, a 'kv reservation hold' at N3.",
        "# THEN (not this window): production takes the ceiling with a new serving image built from the branch head that passed (the catalog entry is the whole production change); the owner decides and restores production with the cutover pack's own steps.",
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
            sys.stderr.write('%s/%s is not what make_np9_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
