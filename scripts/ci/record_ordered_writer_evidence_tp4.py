"""Record the E1 evidence: the report of the ordered-writer card window into ordered_writer_evidence_tp4.json.

    py -3.11 -B scripts/ci/record_ordered_writer_evidence_tp4.py \
        --report <reports>/ordered-<stamp>.json --run <run id> --tag <experiment tag> --commit <40-hex> --image <tag name> \
        [--watcher <reports>/ordered-<stamp>.json --watcher-run <run id> --watcher-tag <tag>]

It is the twin of record_packed_any_evidence_tp4.py for the one section page_width_tp4 reads: every value is taken from the report
JSON that scripts/ci/ordered_writer_tp4_card_test.py wrote (K64J_HARNESS=ordered_writer through run_card_b.sh) and checked BEFORE
anything is written: the verdict (PASS, scope full, width 4096, chips 1of4), no error, no cut case, every required check recorded
and exact, both writers, both widths, seeds 0-2, all three cases (eager, replay_changed, replay_unchanged) and the complete-cache
readback, the page-table entries and the anchor positions, the scoped admission patch (recorded as the harness wrote it), and the
sha256 of ordered_cache.py against the live file. A report that does not qualify is an error, never a record.

The page64 arm (tp4/fx-wp2, kv_page_writer_tp4.py): a report that also ran --writers page64 at full page scope (every regime, both sources, all three modes,
both widths, seeds 0-2, K and V, the write-back proofs, no negative control, a PASS page_decision) adds a 'page_writer' block to the record, bound to the
design the run exercised (kv_page_writer_tp4.design_signature: the kernel's sha256, the pinned compute kernel's, the geometry), checked against the LIVE design
before anything is written. The lever (kv_page_writer_tp4.evidence_problems) engages only on that block. A page-only report (no E1 writers) is recorded with
--merge-page: it adds or replaces the block in the record already on disk (which must stand at its pin) and touches nothing else but the pin. A record
without the block is exactly what it was: the E1 writers' record, admitting width 4,096, and the page writer refused.

Writes, in one go: ordered_writer_evidence_tp4.json (LF, one-space indent) and ORDERED_WRITER_EVIDENCE_TP4_SHA256 in
page_width_tp4.py. Commit both by explicit path (scripts/ci/__pycache__ is tracked: never `git add` a directory or -A).

PUBLIC-REPO HYGIENE: run ids, experiment tags, commit shas, report file names and content hashes only; never an image digest or
registry, a host path, an address or a board id (checked before the write; the image is its tag name).

Stdlib only, Python 3.7 syntax.
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import page_width_tp4  # noqa: E402
import record_packed_any_evidence_tp4 as base  # noqa: E402

PAGE_WRITER = 'page64'

RecordError = base.RecordError
TOP_KEYS = ('schema', 'what', 'status', 'provenance', 'run', 'tag', 'commit', 'card', 'image', 'harness', 'report',
            'report_sha256', 'verdict_line', 'scope', 'chips', 'kv_heads', 'failures', 'widths', 'writers', 'seeds',
            'sections', 'entries', 'anchor_positions', 'blocks', 'counts', 'admission_patch', 'sources', 'page_writer')
PAGE_KEY = 'page_writer'
PAGE_BLOCK_KEYS = ('status', 'scope', 'failures', 'design_signature', 'wt', 'units', 'kernel_sha256', 'compute_sha256', 'widths', 'sources',
                   'regimes', 'modes', 'caches', 'seeds', 'counts', 'proofs', 'grid', 'run', 'tag', 'commit', 'image', 'report', 'report_sha256',
                   'page_verdict')


def page_problems(report, root=HERE):
    """Every reason the report's page64 arm does not qualify the page writer (empty when it does)."""
    import kv_page_writer_tp4 as kvpw
    import ordered_writer_page_arm_tp4 as arm

    problems = []
    requested = report.get('requested') or {}
    page = requested.get('page') or {}
    if arm.PAGE_WRITER not in (requested.get('writers') or ()):
        return ['the report did not run --writers page64']
    words = base.line_words(report.get('verdict_line'))
    decision = report.get('page_decision') or {}
    if decision.get('verdict') != 'PASS' or words.get('page') != 'PASS':
        problems.append('page verdict %s (line %s), not PASS' % (decision.get('verdict'), words.get('page')))
    if words.get('page_scope') != 'full' or arm.page_scope(report) != 'full':
        problems.append('page scope %s, not full' % words.get('page_scope'))
    if report.get('in_progress'):
        problems.append('a checkpoint (in progress %s), not the final report' % report['in_progress'])
    if report.get('error'):
        problems.append('error %s' % str(report['error'])[:100])
    if page.get('negative'):
        problems.append('a negative-control run (%s) is never evidence' % page['negative'])
    if page.get('noop') is not True:
        problems.append('the write-back (noop) proofs did not run')
    for key, wanted in (('sources', arm.SOURCES), ('regimes', arm.REGIMES)):
        missing = [item for item in wanted if item not in (page.get(key) or ())]
        if missing:
            problems.append('requested page %s lack %s' % (key, missing))
    for key, wanted in (('widths', kvpw.PROOF_WIDTHS), ('seeds', (0, 1, 2)), ('modes', kvpw.PROOF_MODES)):
        missing = [item for item in wanted if item not in (requested.get(key) or ())]
        if missing:
            problems.append('requested %s lack %s' % (key, missing))
    checks = [check for check in report.get('checks') or [] if check.get('kind') in arm.PAGE_KINDS]
    exact = sum(1 for check in checks if check.get('exact'))
    if not checks or exact != len(checks):
        problems.append('%d of %d page checks exact' % (exact, len(checks)))
    names = {check.get('name') for check in checks}
    for wanted in ('served_equal_k', 'served_equal_v', 'host_equal_k', 'host_equal_v', 'twice_equal_k', 'twice_equal_v', 'noop_equal_k',
                   'noop_equal_v', 'fill_equal', 'wrote_k', 'wrote_v', 'stored_nonzero_k', 'stored_nonzero_v'):
        if wanted not in names:
            problems.append('no %s check recorded' % wanted)
    seen = {(check.get('regime'), check.get('source'), check.get('mode'), check.get('width')) for check in checks if check.get('kind') == 'page'}
    for regime, source, mode in arm.MATRIX:
        for width in kvpw.PROOF_WIDTHS:
            if (regime, source, mode, width) not in seen:
                problems.append('no page case ran %s %s %s at width %d' % (regime, source, mode, width))
                break
    recorded = {(check['case'], check['name'], check.get('step')) for check in checks}
    for entry in report.get('plan') or ():
        if entry.get('kind') in arm.PAGE_KINDS:
            lacking = [item for item in entry.get('required') or () if (entry['name'], item[0], item[1]) not in recorded]
            if lacking:
                problems.append('page case %s lacks %d required checks' % (entry['name'], len(lacking)))
                break
    for name, state in sorted((report.get('cases') or {}).items()):
        if name.startswith(('page64-', 'noop-')) and (state.get('error') or state.get('skipped') or state.get('exact') is not True):
            problems.append('page case %s: %s' % (name, 'raised' if state.get('error') else 'cut' if state.get('skipped') else 'not exact'))
            break
    proofs = arm.proofs_of(report)
    for label, entry in sorted(proofs.items()):
        if entry['checks'] <= 0 or entry['checks'] != entry['exact']:
            problems.append('proof %s: %d of %d checks exact' % (label, entry['exact'], entry['checks']))
    info = report.get('page') or {}
    wt = info.get('wt')
    if wt not in kvpw.WIDTHS:
        problems.append('the report names unit width %r' % (wt,))
    else:
        live = kvpw.design_signature(wt)
        if info.get('design_signature') != live:
            problems.append('the run exercised design %s, the live one is %s (run the page arm on these bytes)' % (str(info.get('design_signature'))[:16], live[:16]))
        if (report.get('sources') or {}).get('kv_page_writer_tp4.cpp') != kvpw.source_sha256():
            problems.append('the run loaded kv_page_writer_tp4.cpp at another sha256 than the live file')
        if info.get('negative'):
            problems.append('the page design ran with a negative control (%s)' % info['negative'])
    return problems


def page_block(report, digest, path, run, tag, commit, image):
    """The 'page_writer' record of a qualifying report (page_problems empty)."""
    import ordered_writer_page_arm_tp4 as arm

    info, requested = report['page'], report['requested']
    page = requested['page']
    words = base.line_words(report['verdict_line'])
    counts = arm.tally_page(report)
    return dict(status='PASS', scope='full', failures=0, design_signature=info['design_signature'], wt=info['wt'], units=info['units'],
                kernel_sha256=info['kernel_sha256'], compute_sha256=info['design']['compute_sha256'],
                widths=sorted(requested['widths']), sources=sorted(page['sources']), regimes=sorted(page['regimes']),
                modes=sorted(requested['modes']), caches=['k', 'v'], seeds=sorted(requested['seeds']), counts=dict(checks=counts['checks'], exact=counts['exact']),
                proofs=arm.proofs_of(report), grid=list(info.get('grid') or []), run=int(run), tag=tag, commit=commit, image=image,
                report=base.report_name(path), report_sha256=digest,
                page_verdict='page=%s page_scope=%s page_checks=%s page_exact=%s' % (
                    words.get('page'), words.get('page_scope'), words.get('page_checks'), words.get('page_exact')))


def common_ref_problems(commit, image, run, tag):
    problems = []
    if not (isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit)):
        problems.append('--commit must be the 40-hex commit the run tested')
    if not (isinstance(image, str) and re.fullmatch(r'[0-9A-Za-z._-]{3,60}', image)):
        problems.append('--image is the image TAG NAME only (no registry, no digest)')
    if run is None or not tag:
        problems.append('--run and --tag are required')
    return problems


def build(report, digest, path, run, tag, commit, image, watcher=None, root=HERE):
    problems = []
    words = base.line_words(report.get('verdict_line'))
    decision = report.get('decision') or {}
    if report.get('in_progress'):
        problems.append('a checkpoint (in progress %s), not the final report' % report['in_progress'])
    if decision.get('verdict') != 'PASS' or words.get('verdict') != 'PASS':
        problems.append('verdict %s (line %s), not PASS' % (decision.get('verdict'), words.get('verdict')))
    if report.get('error'):
        problems.append('error %s' % str(report['error'])[:100])
    if words.get('scope') != 'full':
        problems.append('scope %s, not full' % words.get('scope'))
    if str(words.get('width')) != str(page_width_tp4.WIDE_WIDTH_TP4):
        problems.append('width %s on the verdict line, not %d' % (words.get('width'), page_width_tp4.WIDE_WIDTH_TP4))
    if words.get('chips') != page_width_tp4.CHIPS or report.get('tp') != 4:
        problems.append('chips %s (tp %s), not %s of four' % (words.get('chips'), report.get('tp'), page_width_tp4.CHIPS))
    if report.get('kv_heads') != 1:
        problems.append('kv_heads %r, not 1' % (report.get('kv_heads'),))
    requested = report.get('requested') or {}
    for key, wanted in (('widths', page_width_tp4.WIDTHS_RECORDED), ('writers', page_width_tp4.WRITERS),
                        ('seeds', page_width_tp4.SEEDS), ('modes', ('eager', 'replay_changed', 'replay_unchanged'))):
        missing = [item for item in wanted if item not in (requested.get(key) or ())]
        if missing:
            problems.append('requested %s lack %s' % (key, missing))
    checks = report.get('checks') or []
    exact = sum(1 for check in checks if check.get('exact'))
    if not checks or exact != len(checks):
        problems.append('%d of %d checks exact' % (exact, len(checks)))
    for name, state in sorted((report.get('cases') or {}).items()):
        if state.get('error') or state.get('skipped') or state.get('exact') is not True:
            problems.append('case %s: %s' % (name, 'raised' if state.get('error') else 'cut' if state.get('skipped') else 'not exact'))
            break
    recorded = {(check['case'], check['name'], check.get('step')) for check in checks}
    for entry in report.get('plan') or ():
        lacking = [item for item in entry.get('required') or () if (entry['name'], item[0], item[1]) not in recorded]
        if lacking:
            problems.append('case %s lacks %d required checks' % (entry['name'], len(lacking)))
            break
    if not report.get('plan'):
        problems.append('the report holds no plan')
    modes_seen = {check.get('mode') for check in checks}
    if not {'eager', 'replay_changed', 'replay_unchanged'} <= modes_seen:
        problems.append('checks cover modes %s' % sorted(modes_seen))
    if not any(check.get('name') == 'complete_cache' for check in checks):
        problems.append('no complete-cache readback recorded')
    patch = report.get('admission_patch') or {}
    if patch.get('width') != page_width_tp4.WIDE_WIDTH_TP4:
        problems.append('admission_patch %r is not the scoped 4,096 patch' % (patch,))
    ran = (report.get('sources') or {}).get('ordered_cache.py')
    live_path = os.path.join(root, 'ordered_cache.py')
    live = page_width_tp4.sha256_file(live_path) if os.path.isfile(live_path) else None
    if ran != live:
        problems.append('the run loaded ordered_cache.py at %s, but the live file is %s (run E1 on these bytes)'
                        % (str(ran)[:16], str(live)[:16]))
    if not (isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit)):
        problems.append('--commit must be the 40-hex commit the run tested')
    if not (isinstance(image, str) and re.fullmatch(r'[0-9A-Za-z._-]{3,60}', image)):
        problems.append('--image is the image TAG NAME only (no registry, no digest)')
    if run is None or not tag:
        problems.append('--run and --tag are required')
    with_page = PAGE_WRITER in (requested.get('writers') or ())
    if with_page:
        problems.extend(page_problems(report, root))
    if problems:
        raise RecordError(problems)
    evidence = dict(
        schema=page_width_tp4.EVIDENCE_SCHEMA,
        what='The four-card record of page_width_tp4: the packed block\'s chained K/V write (chained64) and the engines\' 32-row-tile '
             'write (tiles32) at page-table widths 2,052 (control) and 4,096 on one chip of four, eager and under trace replay '
             '(table rewritten in place, and untouched), the COMPLETE cache read back and compared with the host prediction after '
             'every step. page_width_tp4.admitted reads it at its pin (ORDERED_WRITER_EVIDENCE_TP4_SHA256).',
        status='PASS',
        provenance='Recorded by scripts/ci/record_ordered_writer_evidence_tp4.py from run %s (%s), report cardm/%s (sha256 %s).%s' % (
            run, tag, os.path.basename(path), digest[:16],
            '' if not watcher else ' The watcher pass before it is run %s (%s).' % (watcher['run'], watcher['tag'])),
        run=int(run), tag=tag, commit=commit, card='M', image=image,
        harness='scripts/ci/ordered_writer_tp4_card_test.py through run_card_b.sh (K64J_HARNESS=ordered_writer), '
                '--widths %s --writers %s --seeds %s (the cardm action)' % (
                    ','.join(str(item) for item in requested['widths']), ','.join(requested['writers']),
                    ','.join(str(item) for item in requested['seeds'])),
        report=base.report_name(path), report_sha256=digest, verdict_line=report['verdict_line'].strip(),
        scope='full', chips=words['chips'], kv_heads=1, failures=0,
        widths=sorted(requested['widths']), writers=list(requested['writers']), seeds=sorted(requested['seeds']),
        sections=['eager', 'replay_changed', 'replay_unchanged', 'complete_cache'],
        entries=list(report['entries']), anchor_positions=list(report['anchor_positions']), blocks=report.get('blocks'),
        counts=dict(checks=len(checks), exact=exact), admission_patch=patch,
        sources={'ordered_cache.py': ran})
    if watcher:
        evidence['watcher_pass'] = watcher
    if with_page:
        evidence[PAGE_KEY] = page_block(report, digest, path, run, tag, commit, image)
    return evidence


def merge_page(report, digest, path, run, tag, commit, image, evidence_path, sources_root=HERE):
    """The record on disk with the page64 report's 'page_writer' block added or replaced (the E1 part untouched). The record must still stand: PASS, the
    live ordered_cache.py bytes."""
    problems = common_ref_problems(commit, image, run, tag) + page_problems(report, sources_root)
    try:
        with open(evidence_path, 'rb') as handle:
            existing = json.loads(handle.read().decode('utf-8'))
    except (OSError, ValueError) as error:
        raise RecordError(problems + ['the record to extend does not read: %s' % str(error)[:80]])
    problems.extend('the record on disk: ' + item for item in page_width_tp4.evidence_problems(existing, sources_root))
    if problems:
        raise RecordError(problems)
    evidence = dict(existing)
    evidence[PAGE_KEY] = page_block(report, digest, path, run, tag, commit, image)
    return evidence


def dump(evidence):
    """The bytes of the file: LF, one-space indent, the key order above, a final newline."""
    ordered = {key: evidence[key] for key in TOP_KEYS if key in evidence}
    ordered.update({key: value for key, value in evidence.items() if key not in ordered})
    return (json.dumps(ordered, indent=1, ensure_ascii=False) + '\n').encode('utf-8')


def repin(module_path, digest):
    """page_width_tp4.py with ORDERED_WRITER_EVIDENCE_TP4_SHA256 set to `digest`, the bytes otherwise untouched."""
    with open(module_path, 'rb') as handle:
        text = handle.read().decode('utf-8')
    new, count = re.subn(r"^(ORDERED_WRITER_EVIDENCE_TP4_SHA256 = )'[0-9a-f]{64}'", r"\1'%s'" % digest, text, flags=re.M)
    if count != 1:
        raise RecordError(['%s has %d ORDERED_WRITER_EVIDENCE_TP4_SHA256 assignments, not 1' % (os.path.basename(module_path), count)])
    return new.encode('utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence', default=str(page_width_tp4.EVIDENCE))
    parser.add_argument('--module', default=os.path.join(HERE, 'page_width_tp4.py'))
    parser.add_argument('--sources-root', default=HERE, help='where ordered_cache.py is read (default: here)')
    parser.add_argument('--report', required=True, help='the final report of the E1 run')
    parser.add_argument('--run', type=int, help='the workflow run id of the E1 job')
    parser.add_argument('--tag', help='the experiment tag the run was pushed under')
    parser.add_argument('--commit', help='the 40-hex commit the run tested')
    parser.add_argument('--image', help='the image TAG NAME (no registry, no digest)')
    parser.add_argument('--watcher', help='the watcher pass\'s report (optional)')
    parser.add_argument('--watcher-run', type=int)
    parser.add_argument('--watcher-tag')
    parser.add_argument('--merge-page', action='store_true',
                        help='the report is a page64-only run: add or replace the page_writer block in the record on disk (the E1 part is kept)')
    parser.add_argument('--dry-run', action='store_true', help='check and print the result; write nothing')
    return parser.parse_args(argv)


def main(argv=None, out=print):
    args = parse_args(argv)
    try:
        report, digest = base.read_report(args.report)
        watcher = base.watcher_entry(args.watcher_run, args.watcher_tag, None) if args.watcher_run else None
        if args.merge_page:
            evidence = merge_page(report, digest, args.report, args.run, args.tag, args.commit, args.image, args.evidence, args.sources_root)
        else:
            evidence = build(report, digest, args.report, args.run, args.tag, args.commit, args.image, watcher, args.sources_root)
    except RecordError as error:
        for problem in error.problems:
            out('REFUSED: ' + problem)
        return 1
    dirty = base.hygiene_problems(evidence)
    if dirty:
        for problem in dirty:
            out('REFUSED: ' + problem)
        return 1
    payload = dump(evidence)
    sha = hashlib.sha256(payload).hexdigest()
    remaining = page_width_tp4.evidence_problems(evidence, args.sources_root)
    if PAGE_KEY in evidence:
        import kv_page_writer_tp4 as kvpw

        remaining = remaining + kvpw.block_problems(evidence[PAGE_KEY], evidence[PAGE_KEY].get('wt'))
    out('evidence sha256 %s; %d problems left for page_width_tp4%s' % (sha, len(remaining), '' if not remaining else ':'))
    for problem in remaining:
        out('  - ' + problem)
    if remaining:
        return 1
    pinned = repin(args.module, sha)             # both payloads are made before either file is opened for writing
    if args.dry_run:
        return 0
    with open(args.evidence, 'wb') as handle:
        handle.write(payload)
    with open(args.module, 'wb') as handle:
        handle.write(pinned)
    out('wrote %s and re-pinned ORDERED_WRITER_EVIDENCE_TP4_SHA256 in %s' % (args.evidence, args.module))
    return 0


if __name__ == '__main__':
    sys.exit(main())
