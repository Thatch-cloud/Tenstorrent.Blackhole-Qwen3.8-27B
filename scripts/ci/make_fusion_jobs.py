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

INCLUDED_NEEDS = re.compile(r'^# NEEDS (.+?) <- (.+)$', re.M)

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs'

IMAGE = 'tp4-fusion-1'
SHIP = fusion.PARENT
SMOKE_TESTS = 'warmup,concurrent8_steady'
MINUTES = dict(build=40, status=20, audit=40, timed=35, reset=10, combined_audit=60)

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
    first = [arm_id for _wp, arm_id in plan.get('pack_order', ())]
    rank = lambda arm: first.index(arm['id']) if arm['id'] in first else len(first)      # noqa: E731
    middle = sorted((arm for arm in out if not arm['last']), key=rank)       # (stable: the unlisted keep the manifests' order)
    return middle + [arm for arm in out if arm['last']]


def included(plan, references):
    """[(section, row)] of the other packs' templates folded in (pack_include): each template's text with the pack's one image tag, its name behind the
    include's prefix, its minutes and mode from its own ORDER.txt, and its NEEDS lines renamed (its B0 is this pack's X0)."""
    rows, needs, taken = [], [], set()
    for item in plan.get('pack_include', ()):
        source = Path(references) / item['folder']
        order = source / 'ORDER.txt'
        if not order.is_file():
            raise fusion.ManifestError('%s: pack_include folder %s has no ORDER.txt under %s' % (item['wp'], item['folder'], references))
        text = order.read_text(encoding='utf-8')
        lines = dict((line.split()[0], line.split()) for line in text.splitlines() if line.strip() and not line.startswith('#'))
        for section, names in (('audit', item['audits']), ('timed', item['timed'])):
            for name in names:
                path = source / (name + '.env')
                if not path.is_file() or name not in lines:
                    raise fusion.ManifestError('%s: pack_include template %s is not in %s (a .env and a line of its ORDER.txt)' % (item['wp'], name, item['folder']))
                body = path.read_text(encoding='utf-8')
                if not re.search(r'^C2_IMAGE_TAG=\S+$', body, re.M):
                    raise fusion.ManifestError('%s: %s/%s.env names no C2_IMAGE_TAG' % (item['wp'], item['folder'], name))
                body = re.sub(r'^C2_IMAGE_TAG=\S+$', 'C2_IMAGE_TAG=%s' % IMAGE, body, flags=re.M)
                note = '# Folded into the op-fusion pack by make_fusion_jobs.py from references/%s/%s.env (pack_include, %s): the one image, the template otherwise as its package wrote it.\n' % (
                    item['folder'], name, item['wp'])
                new_name = item['prefix'] + name
                if new_name in taken:
                    raise fusion.ManifestError('%s: the folded template name %s is used twice' % (item['wp'], new_name))
                taken.add(new_name)
                rows.append((section, dict(name=new_name, mode='soft', minutes=int(lines[name][3]), text=note + body, needs=(), folded=item)))
        short = lambda token: item['prefix'] + token       # noqa: E731
        for left, right in INCLUDED_NEEDS.findall(text):
            mine = [short(token) for token in left.split() if token not in ('B0', 'Z')]
            theirs = [('X0' if token == 'B0' else short(token)) for token in right.split()]
            kept = set(row['name'].split('-')[0] for _s, row in rows)
            mine = [token for token in mine if token in kept]
            theirs = [token for token in theirs if token == 'X0' or token in kept]
            if mine and theirs:
                needs.append(('%s <- %s' % (' '.join(mine), ' '.join(theirs)), mine[0]))
    owners = dict((row['name'].split('-')[0], row) for _section, row in rows)
    for line, first in needs:
        owners[first]['needs'] = owners[first]['needs'] + (line,)
    return rows


def jobs(plan, references=None):
    """[dict(name, mode, minutes, actions, profile, tests, why, needs)] in the order they are run: B0, X0, every audit (the arms that wait last), every ABAB (the
    arms that wait last), Z; the other packs' folded templates (pack_include) after the ordinary arms of their section, before the arms that wait last."""
    head = [
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
    folded = included(plan, references) if references is not None else []
    audits, timed = {False: [], True: []}, {False: [], True: []}
    for arm in every:
        prefix = arm['prefix']
        if arm['audit']:
            audits[bool(arm['last'])].append(dict(
                name='%sA-%s-audit-attach' % (prefix, arm['id']), mode='soft', minutes=MINUTES['audit'], actions='reset smoke', profile=arm['audit'], tests=SMOKE_TESTS,
                needs=('%sA <- X0' % prefix,),
                why=("%sA (EXACTNESS, %s): %s = the production profile plus %s; the audit runs the served composition beside the lever on the device and logs exact=True or a mismatch. "
                     "READ: c2_smoke_check clean with the fusion rules (the engaged line for the lever, no fell-back line, at least one audit line carrying exact=True, no 'audit mismatch'), no stall, every answer complete. "
                     "ANY mismatch, fallback or missing audit line is NO-GO for the lever and skips its timed jobs. %s Duration (estimate): %d minutes."
                     % (prefix, arm['wp'], arm['audit'], env_text(arm['env']), fusion.sentence(arm['reason']), MINUTES['audit']))))
        if not arm['timed']:
            continue
        gate = ['%sA' % prefix] if arm['audit'] else []
        legs = (('C1', 'control-timed', SHIP, 'the CONTROL (A1): the production profile itself, no lever', ['X0']),
                ('L1', 'lever-timed', arm['timed'], 'the LEVER (B1): the production profile plus %s' % env_text(arm['env']), gate + ['%sC1' % prefix]),
                ('C2', 'control-repeat', SHIP, 'the CONTROL again (A2)', ['%sL1' % prefix]),
                ('L2', 'lever-repeat', arm['timed'], 'the LEVER again (B2)', ['%sC2' % prefix]))
        for leg, label, profile, what, needs in legs:
            timed[bool(arm['last'])].append(dict(
                name='%s%s-%s-%s' % (prefix, leg, arm['id'], label), mode='soft', minutes=MINUTES['timed'], actions='reset smoke', profile=profile, tests=SMOKE_TESTS,
                needs=('%s%s <- %s' % (prefix, leg, ' '.join(needs)),),
                why=("%s%s (TIMED, %s, %s): %s; eight live (concurrent8_steady) in the %s pair of the ABAB, every boot a fresh reset. READ: c2_smoke_check clean (a lever arm also has its engaged line "
                     "and no fell-back line), no stall; then PAIRED per round, never by unpaired medians: python3 scripts/ci/w2ln_timing_compare.py pair <A container log> <B container log> "
                     "--window steady --a-smoke <A smoke log> --b-smoke <B smoke log> for (%sC1, %sL1) and (%sC2, %sL2), and its floor subcommand over the two control logs; the texts equal "
                     "(text_mismatches empty). %s Duration (estimate): %d minutes."
                     % (prefix, leg, arm['name'], arm['wp'], what, 'first' if leg in ('C1', 'L1') else 'repeat', prefix, prefix, prefix, prefix,
                        fusion.sentence(arm['reason']), MINUTES['timed']))))
    rows = (head + audits[False] + [row for section, row in folded if section == 'audit'] + audits[True]
            + timed[False] + [row for section, row in folded if section == 'timed'] + timed[True])
    rows.append(dict(name='Z-reset', mode='soft', minutes=MINUTES['reset'], actions='status reset', profile=None, tests='', needs=(),
                     why=("Z: the window's end: status and an all-four reset, no agent start (the cards stay under development). (The owner restores production afterwards: card reset, link measurement, "
                          "topology republish, agent restart, deploy and the post-checks.) Duration (estimate): %d minutes." % MINUTES['reset'])))
    for row in rows:
        row.setdefault('text', None)
    return rows


def env_text(env):
    return ', '.join('%s=%s' % pair for pair in sorted(env.items())) or 'nothing'


COMBINED_FOLDER = 'combined'


def combined_prefix(name):
    return 'FX' if name == 'all' else 'FX' + name.upper().replace('-', '')


def env_pairs(env):
    return ', '.join('%s=%s' % pair for pair in sorted(env.items()))


def combined_twins(plan, spec):
    """(timed twin, audit twin) of one combined spec, from the profile generator."""
    twins = dict((twin['name'], twin) for twin in fusion.profile_twins(plan))
    return twins[fusion.NAMESPACE + spec['name']], twins[fusion.NAMESPACE + spec['name'] + '-audit']


def combined_rows(plan, spec):
    """The rows of one combined spec's mini-pack: B0, X0, the audited attach of the combination, its timed ABAB, Z."""
    timed, audited = combined_twins(plan, spec)
    levers = [lever for lever in plan['levers'] if lever['id'] in spec['levers']]
    prefix, name = combined_prefix(spec['name']), spec['name']
    judged = ', '.join('%s (%s)' % (lever['name'], lever['flag']) for lever in levers)
    every = jobs(plan)
    rows = [every[0], every[1], dict(
        name='%sA-%s-audited-attach' % (prefix, name), mode='soft', minutes=MINUTES['combined_audit'], actions='reset smoke', profile=audited['name'], tests=SMOKE_TESTS,
        telemetry=True, needs=('%sA <- X0' % prefix,),
        why=("%sA (EXACTNESS of the combination %s, %s): %s = the production profile plus %s. Every lever of it runs its audit, which compares the served composition beside the lever on the device. "
             "READ: c2_smoke_check clean under EVERY lever's rule at once (%s): each lever logs its engaged line and an audit line with exact=True, no fell-back line, no 'audit mismatch', no stall, every "
             "answer complete. ANY mismatch, fallback or missing audit line is NO-GO for the combination: read which lever's rule names it, and ablate with the per-lever pack (references/fusion-jobs). "
             "%s Duration (estimate): %d minutes." % (prefix, name, spec['wp'], audited['name'], env_pairs(audited['env']), judged, fusion.sentence(spec['reason']), MINUTES['combined_audit'])))]
    legs = (('C1', 'control-timed', SHIP, 'the CONTROL (A1): the production profile itself, no lever', ['X0']),
            ('L1', 'lever-timed', timed['name'], 'the LEVERS (B1): the production profile plus %s' % env_pairs(timed['env']), ['%sA' % prefix, '%sC1' % prefix]),
            ('C2', 'control-repeat', SHIP, 'the CONTROL again (A2)', ['%sL1' % prefix]),
            ('L2', 'lever-repeat', timed['name'], 'the LEVERS again (B2)', ['%sC2' % prefix]))
    for leg, label, profile, what, needs in legs:
        rows.append(dict(
            name='%s%s-%s-%s' % (prefix, leg, name, label), mode='soft', minutes=MINUTES['timed'], actions='reset smoke', profile=profile, tests=SMOKE_TESTS,
            telemetry=True, needs=('%s%s <- %s' % (prefix, leg, ' '.join(needs)),),
            why=("%s%s (TIMED, the combination %s, %s): %s; eight live (concurrent8_steady) in the %s pair of the ABAB, every boot a fresh reset. READ: c2_smoke_check clean (a lever arm also has every engaged "
                 "line and no fell-back line), no stall; then PAIRED per round, never by unpaired medians: python3 scripts/ci/w2ln_timing_compare.py pair <A container log> <B container log> "
                 "--window steady --a-smoke <A smoke log> --b-smoke <B smoke log> for (%sC1, %sL1) and (%sC2, %sL2), and its floor subcommand over the two control logs; the texts equal "
                 "(text_mismatches empty). Levers on together: %s. %s Duration (estimate): %d minutes."
                 % (prefix, leg, name, spec['wp'], what, 'first' if leg in ('C1', 'L1') else 'repeat', prefix, prefix, prefix, prefix, judged, fusion.sentence(spec['reason']), MINUTES['timed']))))
    rows.append(every[-1])
    for row in rows:
        row.setdefault('text', None)
    return rows


def render_combined_order(rows, plan, spec):
    total = sum(row['minutes'] for row in rows)
    prefix = combined_prefix(spec['name'])
    timed, _audited = combined_twins(plan, spec)
    lines = [
        "# The order of the op-fusion programme's COMBINED window (docs/tp4-fusion.md): the levers %s ON TOGETHER, one audited attach and one timed control/lever ABAB at eight live; ONE image (%s)." % (', '.join(spec['levers']), IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. The same B0, X0 and Z as the per-lever pack (references/fusion-jobs), which is the ABLATION: run it for the levers of a combination that fails.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing here does either. CI PAUSED from the first timed job to the last. The CPU suite is green on the pushed head;",
        "#     push ONE tag at a time and wait for its run to finish. Never cancel a running card job. Every result is UNQUALIFIED (gate-only twins). Read the grid from the run (13x10 since the firmware unlock), never assume 11x10.",
        "# THE TWINS: %s%s (the timed arm) and %s%s-audit are generated by make_fusion_profiles.py from the combined entry of %s: the production profile plus %s." % (
            fusion.NAMESPACE, spec['name'], fusion.NAMESPACE, spec['name'], spec['file'], env_pairs(timed['env'])),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in rows:
        lines.extend('# NEEDS ' + need for need in row['needs'])
    lines += [
        "# READ RULES: B0 and X0 as the per-lever pack. %sA: every lever's engaged line and an audit line with exact=True, no fell-back line, no 'audit mismatch'. ANY failure is NO-GO for the combination." % prefix,
        "#   %sC1 %sL1 %sC2 %sL2: w2ln_timing_compare.py pair, the steady window, 8 live; GO needs both pairs measured (at least 100 matched rounds, the load rule applied) and negative beyond the control-to-control floor, the texts equal." % (prefix, prefix, prefix, prefix),
        "# Time (estimate, minutes): %d in all (%d h %d min), %d of cards." % (total, total // 60, total % 60, total - MINUTES['build']),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in rows]
    return '\n'.join(lines) + '\n'


def render_env(row):
    if row.get('text') is not None:
        return row['text']
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why']
    if row.get('telemetry'):
        why += (' TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py; on the control and the lever legs alike, so the pairs stay paired): the read-only ARC sidecar samples every chip once a second through '
                'the smoke step (AICLK and the limiter holding it, power, current, temperature, kernel NOPs); read telemetry-smoke/telemetry-summary.txt beside the timing, and the first [TELEMETRY] line must say '
                'decode check ok. It reads only (no ARC message, no write) and runs on the host: it is not part of any arm.')
    lines += ['# ' + piece for piece in wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row.get('bake'):
        lines.append('C2_BAKE_DEFAULT_PROFILE=%s' % row['bake'])
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    if row.get('telemetry'):
        lines.append('C2_TELEMETRY=1')
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
        found += ['%s/%s' % (folder.name, path.name) for path in folder.iterdir() if path.is_dir() and path.name != COMBINED_FOLDER]
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
    for spec in plan.get('combined', ()):
        lines.insert(-1, "# THE COMBINED PACK (references/fusion-jobs/%s: %s%s, the levers %s on together; its own ORDER.txt) runs FIRST; this ORDER is the ablation, for the levers of a combination that fails." % (
            COMBINED_FOLDER, fusion.NAMESPACE, spec['name'], ', '.join(spec['levers'])))
    for item in plan.get('pack_include', ()):
        lines.insert(-1, "# FOLDED IN: %s* are the templates of references/%s (%s), renamed and on the one image; their read and decision rules are that folder's ORDER.txt. %s" % (
            item['prefix'], item['folder'], item['wp'], ' '.join(item['reason'].split())))
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
    """{file name: text} of the whole pack; `folder` is where the pack lives (the packages' own sub-folders beside it are named in ORDER.txt, and the packs named under
    pack_include are read from beside it)."""
    rows = jobs(plan, Path(folder).parent)
    out = dict(('%s.env' % row['name'], render_env(row)) for row in rows)
    out['ORDER.txt'] = render_order(rows, plan, folder)
    for spec in plan.get('combined', ()):
        mini = combined_rows(plan, spec)
        for row in mini:
            out['%s/%s.env' % (COMBINED_FOLDER, row['name'])] = render_env(row)
        out['%s/ORDER.txt' % COMBINED_FOLDER] = render_combined_order(mini, plan, spec)
    return out


def stale(folder, wanted):
    """[file names] that differ from `wanted` or are not in it (the pack's own files and the combined/ sub-folder, which the generator owns whole)."""
    found = dict((path.name, path.read_text(encoding='utf-8')) for path in Path(folder).glob('*') if path.is_file())
    found.update(('%s/%s' % (COMBINED_FOLDER, path.name), path.read_text(encoding='utf-8')) for path in (Path(folder) / COMBINED_FOLDER).glob('*') if path.is_file())
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
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in wanted:
            with open(str(path), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted[name])
        else:
            path.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
