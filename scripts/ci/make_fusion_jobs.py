"""The op-fusion programme's card pack (scripts/ci/references/fusion-jobs), generated from the work packages' manifests (stdlib only).

    python3 scripts/ci/make_fusion_jobs.py --write   regenerate the templates and ORDER.txt from scripts/ci/fusion-wp/*.json
    python3 scripts/ci/make_fusion_jobs.py --check   exit 1 if the checked-in pack is not what the manifests generate

ONE image (tp4-fusion-1, built by B0 from the pushed branch head), ONE window, one tag at a time. The pack is B0 (build), X0 (status, rescan, reset), then for every lever of every
merged package (make_fusion_profiles.py: scripts/ci/fusion-wp/*.json; an extra profile twin joins when a manifest names it under pack_arms):

  <ID>A   the AUDITED attach smoke: the lever's -audit twin (the lever plus its _AUDIT flag, which runs the served composition beside the lever on the device) over warmup and
          concurrent8_steady; the exactness gate. Skipped for a lever with no audit flag (its timed jobs then need the control only).
  <ID>C1 <ID>L1 <ID>C2 <ID>L2   the timed ABAB at eight live (concurrent8_steady): control (the production profile), lever, control, lever, each a separate boot.

then Z (status, reset). Nothing here stops, starts or hands back the node agent, builds anything but the image, or names a rig, a card, a host or a registry. With no merged lever
the pack is B0, X0 and Z. Every template parses through scripts/ci/c2_serving_job.py with rc 0 (test_tp4_fusion_jobs holds it).
"""

import argparse
import json
from pathlib import Path
import re
import sys

import make_fusion_profiles as fusion

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs'

IMAGE = 'tp4-fusion-1'
SHIP = fusion.PARENT
SMOKE_TESTS = 'warmup,concurrent8_steady'
MINUTES = dict(build=40, status=20, audit=40, timed=35, reset=10)

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the op-fusion programme's four-card window (scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/fusion-1 and push a tag experiment/c2-serving-vN. Nothing here names a rig, a card, a host or a registry: the cards
# are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file the image
# carries needs a new B0 build and a new tag in every template. The cards are under development: this window never stops, starts or hands back the node agent."""


def arms(plan):
    """[dict(prefix, id, wp, name, reason, timed, audit, env)] in the order the arms are written: one per lever (its timed twin and, with an audit flag, its audit twin), then one
    per extra profile twin a manifest names under pack_arms (`<suffix>` timed, `<suffix>-audit` its audit twin; an audit twin with no timed one is an audit-only arm); the other
    extra twins (variants, combinations, controls) belong to their package's own folder. The prefix is the id in capitals
    ('s1' -> S1, 'kv-writer' -> KVWRITER) and must be unique."""
    out, by_id, seen = [], {}, {}
    chosen = set(arm for _wp, arm in plan.get('pack_arms', ()))
    for twin in fusion.profile_twins(plan):
        suffix = twin['name'][len(fusion.NAMESPACE):]
        if twin['kind'] == 'audit' and suffix.endswith('-audit'):
            ident, slot = suffix[:-len('-audit')], 'audit'
        elif twin['kind'] == 'audit':
            ident, slot = suffix, 'audit'
        else:
            ident, slot = suffix, 'timed'
        if twin['lever'] is None and ident not in chosen and suffix not in chosen:
            continue            # an extra twin (a variant, a combination, a control) is in the pack only when a manifest names it under pack_arms
        arm = by_id.get(ident)
        if arm is None:
            prefix = ident.upper().replace('-', '')
            if prefix in seen or prefix in ('B0', 'X0', 'Z'):
                raise fusion.ManifestError('arm %s (%s) and %s give the same job prefix %s' % (ident, twin['wp'], seen.get(prefix, 'the pack'), prefix))
            seen[prefix] = ident
            lever = twin['lever']
            arm = by_id[ident] = dict(prefix=prefix, id=ident, wp=twin['wp'], timed=None, audit=None, env={},
                                      name=lever['name'] if lever else ident, reason=lever['reason'] if lever else twin['why'])
            out.append(arm)
        arm[slot] = twin['name']
        if slot == 'timed':
            arm['env'] = twin['env']
        elif not arm['env']:
            arm['env'] = twin['env']
    last = dict((arm_id, reason) for _wp, arm_id, reason in plan.get('pack_last', ()))
    for arm in out:
        arm['last'] = last.get(arm['id'])
    return [arm for arm in out if not arm['last']] + [arm for arm in out if arm['last']]


def jobs(plan):
    """[dict(name, mode, minutes, actions, profile, tests, why, needs)] in the order they are run: B0, X0, every audit, every ABAB, Z."""
    rows = [
        dict(name='B0-build', mode='stop', minutes=MINUTES['build'], actions='build', profile='c2-packed-tp4', tests='', bake=SHIP, needs=(),
             why=("B0 (STOP): the window image, built from the pushed branch head with the SHIP profile baked as the image's serving default (ENV QWEN_C2_PROFILE) and THATCH_SERVING_SESSION_CAP baked as its "
                  "max-num-seqs (8): the SAME default production runs (%s), so the image could serve; the gate-only fusion twins are in its profiles file and the window selects them per job. "
                  "It carries every package's modules (the overlay manifest, which make_fusion_profiles.py fills from the manifests). REFUSED by c2_serving_job.read_bake while the profile's "
                  "owner_traffic_waiver decision is not APPROVED. READ: 'baking the serving default %s with a session cap of 8', the provenance report with no problem line, the image id. "
                  "A build, not a card job: it may run while the cards serve, never during a timing job. Duration (estimate): %d minutes." % (SHIP, SHIP, MINUTES['build']))),
        dict(name='X0-status-rescan-reset', mode='stop', minutes=MINUTES['status'], actions='status rescan reset', profile=None, tests='', needs=('X0 <- B0',),
             why=("X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, no other window driver alive. "
                  "Stop rule: anything else. (The owner stops the node agent and unserves production before it, and restores production after Z, with the cutover pack's own steps.) Duration (estimate): %d minutes."
                  % MINUTES['status'])),
    ]
    every = arms(plan)
    for arm in every:
        if not arm['audit']:
            continue
        prefix = arm['prefix']
        rows.append(dict(
            name='%sA-%s-audit-attach' % (prefix, arm['id']), mode='soft', minutes=MINUTES['audit'], actions='reset smoke', profile=arm['audit'], tests=SMOKE_TESTS,
            needs=('%sA <- X0' % prefix,),
            why=("%sA (EXACTNESS, %s): %s = the production profile plus %s; the audit runs the served composition beside the lever on the device and logs exact=True or a mismatch. "
                 "READ: c2_smoke_check clean with the fusion rules (the engaged line for the lever, no fell-back line, at least one audit line carrying exact=True, no 'audit mismatch'), no stall, every answer complete. "
                 "ANY mismatch, fallback or missing audit line is NO-GO for the lever and skips its timed jobs. %s Duration (estimate): %d minutes."
                 % (prefix, arm['wp'], arm['audit'], env_text(arm['env']), fusion.sentence(arm['reason']), MINUTES['audit']))))
    for arm in every:
        if not arm['timed']:
            continue
        prefix = arm['prefix']
        gate = ['%sA' % prefix] if arm['audit'] else []
        legs = (('C1', 'control-timed', SHIP, 'the CONTROL (A1): the production profile itself, no lever', ['X0']),
                ('L1', 'lever-timed', arm['timed'], 'the LEVER (B1): the production profile plus %s' % env_text(arm['env']), gate + ['%sC1' % prefix]),
                ('C2', 'control-repeat', SHIP, 'the CONTROL again (A2)', ['%sL1' % prefix]),
                ('L2', 'lever-repeat', arm['timed'], 'the LEVER again (B2)', ['%sC2' % prefix]))
        for leg, label, profile, what, needs in legs:
            rows.append(dict(
                name='%s%s-%s-%s' % (prefix, leg, arm['id'], label), mode='soft', minutes=MINUTES['timed'], actions='reset smoke', profile=profile, tests=SMOKE_TESTS,
                needs=('%s%s <- %s' % (prefix, leg, ' '.join(needs)),),
                why=("%s%s (TIMED, %s, %s): %s; eight live (concurrent8_steady) in the %s pair of the ABAB, every boot a fresh reset. READ: c2_smoke_check clean (a lever arm also has its engaged line "
                     "and no fell-back line), no stall; then PAIRED per round, never by unpaired medians: python3 scripts/ci/w2ln_timing_compare.py pair <A container log> <B container log> "
                     "--window steady --a-smoke <A smoke log> --b-smoke <B smoke log> for (%sC1, %sL1) and (%sC2, %sL2), and its floor subcommand over the two control logs; the texts equal "
                     "(text_mismatches empty). %s Duration (estimate): %d minutes."
                     % (prefix, leg, arm['name'], arm['wp'], what, 'first' if leg in ('C1', 'L1') else 'repeat', prefix, prefix, prefix, prefix,
                        fusion.sentence(arm['reason']), MINUTES['timed']))))
    rows.append(dict(name='Z-reset', mode='soft', minutes=MINUTES['reset'], actions='status reset', profile=None, tests='', needs=(),
                     why=("Z: the window's end: status and an all-four reset, no agent start (the cards stay under development). (The owner restores production afterwards: card reset, link measurement, "
                          "topology republish, agent restart, deploy and the post-checks.) Duration (estimate): %d minutes." % MINUTES['reset'])))
    return rows


def env_text(env):
    return ', '.join('%s=%s' % pair for pair in sorted(env.items())) or 'nothing'


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    lines += ['# ' + piece for piece in wrap(row['why'])]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row.get('bake'):
        lines.append('C2_BAKE_DEFAULT_PROFILE=%s' % row['bake'])
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    return '\n'.join(lines) + '\n'


def wrap(text, width=170):
    """`text` in lines of at most `width` characters, broken at spaces."""
    lines, current = [], ''
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + ' ' + word) if current else word
    if current:
        lines.append(current)
    return lines


def package_folders(folder):
    """The packages' own template folders, by name relative to references/: the sub-folders of the pack (fusion-jobs/WP1) and, for the pack itself (fusion-jobs), the sibling folders that start with
    `fusion-` (fusion-jobs-wp5, fusion-wp7-jobs: a package may keep its templates beside the pack so that nothing sits among the files the generator owns)."""
    folder = Path(folder)
    found = []
    if folder.is_dir():
        found += ['%s/%s' % (folder.name, path.name) for path in folder.iterdir() if path.is_dir()]
        if folder.name == 'fusion-jobs' and folder.parent.is_dir():
            found += [path.name for path in folder.parent.iterdir() if path.is_dir() and path.name.startswith('fusion-') and path != folder]
    return sorted(found)


def render_order(rows, plan, folder=FOLDER):
    total = sum(row['minutes'] for row in rows)
    cards = total - MINUTES['build']
    repeats = sum(row['minutes'] for row in rows if re.match(r'^[A-Z0-9]+(C2|L2)-', row['name']))
    every = arms(plan)
    packs = package_folders(folder)
    lines = [
        "# The order of the op-fusion programme's four-card DEVELOPMENT window (docs/tp4-fusion.md): ONE image (%s, built by B0 from tp4/fusion-1 = tp4/next2-1 + the merged work packages) and ONE window for the" % IMAGE,
        "# fused-op arms of the packages, each behind default-off flags: %s." % ('; '.join('%s (%s: %s)' % (arm['name'], arm['wp'], env_text(arm['env'])) for arm in every) or 'none merged yet'),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. The minutes are generic ESTIMATES (nothing has run on a card): a package's own folder has its own.",
        "# stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (a lever's timed jobs are skipped by the NEEDS lines below, the other levers still run).",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z (the cutover pack's restore steps and post-checks; nothing in this pack does either, and no job here starts or",
        "#     hands back the agent). B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and the tag is pushed on THAT commit. Push ONE tag at a time and wait for its run to finish (the workflow keeps one",
        "#     pending run; queued tags are cancelled). Never cancel a running card job.",
        "# P0. CI PAUSED: the owner scales the rig's ARC runner sets to zero from the first timed job to the last (CI shares the host and skews round times; the hardware host runner that runs these jobs is NOT scaled down) and restores",
        "#     them after it. The load average is read with every pair (w2ln_timing_compare.py load-limit).",
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run (AVAILABLE WORKER CORE COUNT), never assume 11x10: every pair runs on one grid, and a number from an 11x10 run is not this pack's control.",
        "# Every result is UNQUALIFIED (gate-only twins of the production profile; the control is the production profile itself, which carries an owner traffic waiver). A lever that passes its audit and its ABAB is a go to",
        "#     confirm it in the ship profile, not to ship. Each twin is the production profile plus exactly one lever (and, for the audit twin, its _AUDIT flag): make_fusion_profiles.py generated it from the package's manifest.",
        "# THE JOBS OF A LEVER <ID>: <ID>A the audited attach (exactness: the lever's served-composition comparison in the trace), then the timed ABAB at eight live (concurrent8_steady): <ID>C1 control, <ID>L1 lever, <ID>C2 control, <ID>L2 lever.",
        "#     The audits of every lever run before any timed pair, so a lever that is not exact costs one short job, not an hour of timing.",
        "# THE PACKAGES' OWN FOLDERS (card-M and detail templates, their read rules and decision rules; under references/): %s." % (', '.join(packs) or 'none yet'),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for arm in every:
        if arm['last']:
            lines.insert(-1, "# LAST: %s (%s) runs after every other arm, audit and ABAB alike: %s" % (arm['prefix'], arm['name'], ' '.join(arm['last'].split())))
    for row in rows:
        lines.extend('# NEEDS ' + need for need in row['needs'])
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, whose fusion rules read the container log, and no stall):",
        "#   B0     'baking the serving default %s with a session cap of 8', the provenance report with no problem line, the image id." % SHIP,
        "#   X0     four cards present and healthy, the fabric as measured, no other window driver alive.",
        "#   <ID>A  the engaged line, no fell-back line, an audit line with exact=True, no 'audit mismatch'. ANY failure is NO-GO for the lever.",
        "#   <ID>C1 <ID>L1 <ID>C2 <ID>L2  w2ln_timing_compare.py pair, the steady window, 8 live: per pair the lever's round time against the control's; GO for a lever needs both pairs measured (at least 100 matched rounds, the load rule",
        "#     applied) and negative beyond the control-to-control floor, the texts equal, the lever's engaged line in both lever boots. A pair with fewer matched rounds is VOID (never GO, never NO-GO).",
        "# NO-GO for adopting a lever: any stall, any text or accepted-prefix difference against the control, any audit mismatch, any fallback line, a pair that is positive, a lever boot that logs no engaged line.",
        "# Time (estimate, minutes): B0 %d, X0 %d, an audit %d, a timed job %d, Z %d: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z), %d minutes without the repeat pairs; about 3 minutes of queue per tag on top."
        % (MINUTES['build'], MINUTES['status'], MINUTES['audit'], MINUTES['timed'], MINUTES['reset'], total, total // 60, total % 60, cards, total - repeats),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in rows]
    return '\n'.join(lines) + '\n'


def generate(plan, folder=FOLDER):
    """{file name: text} of the whole pack; `folder` is where the pack lives (the packages' own sub-folders beside it are named in ORDER.txt)."""
    rows = jobs(plan)
    out = dict(('%s.env' % row['name'], render_env(row)) for row in rows)
    out['ORDER.txt'] = render_order(rows, plan, folder)
    return out


def stale(folder, wanted):
    """[file names] that differ from `wanted` or are not in it."""
    found = dict((path.name, path.read_text(encoding='utf-8')) for path in Path(folder).glob('*') if path.is_file())
    names = sorted(set(found) | set(wanted))
    return [name for name in names if found.get(name) != wanted.get(name)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--manifests', default=str(fusion.MANIFESTS))
    parser.add_argument('--folder', default=str(FOLDER))
    arguments = parser.parse_args(argv)
    if not (arguments.write or arguments.check):
        parser.error('--write or --check')
    try:
        wanted = generate(fusion.normalise(fusion.read_manifests(arguments.manifests)), Path(arguments.folder))
    except fusion.ManifestError as error:
        sys.stderr.write('make_fusion_jobs: %s\n' % error)
        return 2
    folder = Path(arguments.folder)
    differ = stale(folder, wanted)
    if arguments.check:
        for name in differ:
            sys.stderr.write('%s is not what make_fusion_jobs.py generates from scripts/ci/fusion-wp: run it with --write\n' % (folder / name))
        return 1 if differ else 0
    folder.mkdir(parents=True, exist_ok=True)
    for name in differ:
        path = folder / name
        if name in wanted:
            with open(str(path), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted[name])
        else:
            path.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
