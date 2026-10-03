"""record_ordered_writer_evidence_tp4: a report that qualifies becomes the record and its pin, one that does not is an error and writes nothing,
and nothing outside the record and the pin changes (the 131k packed-any record and its pin are never touched)."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ordered_writer_tp4_card_test as card  # noqa: E402
import packed_any_admission as admission  # noqa: E402
import page_width_tp4 as pw  # noqa: E402
import record_ordered_writer_evidence_tp4 as rec  # noqa: E402
import tp_shapes  # noqa: E402

TP4 = {tp_shapes.TP_SWITCH: '4'}
COMMIT = 'c' * 40


def make_report(root=pw.HERE):
    specs = card.build_cases()
    checks, cases, plan = [], {}, []
    for spec in specs:
        required = card.required_checks(spec)
        plan.append(dict(name=spec['name'], required=[list(item) for item in required]))
        cases[spec['name']] = dict(exact=True)
        for name, step in required:
            checks.append(dict(case=spec['name'], writer=spec['writer'], width=spec['width'], mode=spec['mode'],
                               seed=spec['seed'], name=name, step=step, exact=True))
    report = dict(checks=checks, cases=cases, plan=plan, tp=4, kv_heads=1, blocks=card.BLOCKS,
                  entries=sorted(set(card.WIDE_ENTRIES) | set(card.CONTROL_ENTRIES)),
                  anchor_positions=[262080, 262111], admission_patch=dict(module='page_width_tp4.admitted', width=4096, scope='x'),
                  requested=dict(widths=list(card.WIDTHS), writers=list(card.WRITERS), seeds=list(card.SEEDS),
                                 modes=list(card.MODES)),
                  sources={'ordered_cache.py': pw.sha256_file(Path(root) / 'ordered_cache.py')})
    report['decision'] = card.decide(report)
    report['verdict_line'] = card.verdict_line(report)
    return report


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ('ordered_cache.py', 'page_width_tp4.py', 'ordered_writer_evidence_tp4.json'):
            shutil.copyfile(pw.HERE / name, self.dir / name)
        self.report = make_report(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_recorder(self, report=None, extra=(), **overrides):
        path = self.dir / 'ordered-20261004T010203.json'
        path.write_text(json.dumps(self.report if report is None else report, indent=1))
        argv = ['--evidence', str(self.dir / 'ordered_writer_evidence_tp4.json'), '--module', str(self.dir / 'page_width_tp4.py'),
                '--sources-root', str(self.dir), '--report', str(path), '--run', '36900000001', '--tag', 'v300',
                '--commit', COMMIT, '--image', 'tp4-serve-8'] + list(extra)
        lines = []
        for key, value in overrides.items():
            argv += ['--' + key.replace('_', '-'), value]
        code = rec.main(argv, out=lines.append)
        return code, lines

    def test_a_qualifying_report_writes_the_record_and_its_pin_and_4096_is_then_admitted(self):
        before = (self.dir / 'page_width_tp4.py').read_bytes()
        code, lines = self.run_recorder()
        self.assertEqual(code, 0, lines)
        payload = (self.dir / 'ordered_writer_evidence_tp4.json').read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        text = (self.dir / 'page_width_tp4.py').read_bytes().decode()
        self.assertIn("ORDERED_WRITER_EVIDENCE_TP4_SHA256 = '%s'" % digest, text)
        self.assertEqual(before.decode().replace(pw.ORDERED_WRITER_EVIDENCE_TP4_SHA256, digest), text, 'only the pin changed')
        evidence = json.loads(payload.decode())
        self.assertEqual((evidence['status'], evidence['scope'], evidence['chips'], evidence['kv_heads'], evidence['run']),
                         ('PASS', 'full', '1of4', 1, 36900000001))
        self.assertEqual(rec.dump(evidence), payload, 'LF, one-space indent, the key order')
        self.assertNotIn(b'\r', payload)
        self.assertTrue(pw.admitted(4096, TP4, path=self.dir / 'ordered_writer_evidence_tp4.json', expected=digest,
                                    sources_root=self.dir))
        self.assertFalse(pw.admitted(4096, {}, path=self.dir / 'ordered_writer_evidence_tp4.json', expected=digest,
                                     sources_root=self.dir), 'never at the pair')

    def test_a_dry_run_writes_nothing(self):
        before = {name: (self.dir / name).read_bytes() for name in ('page_width_tp4.py', 'ordered_writer_evidence_tp4.json')}
        code, lines = self.run_recorder(extra=['--dry-run'])
        self.assertEqual(code, 0, lines)
        self.assertEqual(before, {name: (self.dir / name).read_bytes() for name in before})

    def test_the_packed_any_record_and_its_pin_are_never_touched(self):
        pin, record = admission.EVIDENCE_TP4_SHA256, admission.EVIDENCE_TP4.read_bytes()
        self.assertEqual(self.run_recorder()[0], 0)
        self.assertEqual((admission.EVIDENCE_TP4_SHA256, admission.EVIDENCE_TP4.read_bytes()), (pin, record))

    def test_a_report_that_does_not_qualify_is_refused_and_nothing_is_written(self):
        mutations = {
            'a checkpoint': lambda r: r.update(in_progress='x'),
            'a failure': lambda r: r['checks'][5].update(exact=False),
            'an error': lambda r: r.update(error='boom'),
            'one seed': lambda r: r['requested'].update(seeds=[0]),
            'one width': lambda r: r['requested'].update(widths=[4096]),
            'one writer': lambda r: r['requested'].update(writers=['tiles32']),
            'no replay': lambda r: r['requested'].update(modes=['eager']),
            'two chips': lambda r: r.update(tp=2),
            'two heads': lambda r: r.update(kv_heads=2),
            'a cut case': lambda r: r['cases'][next(iter(r['cases']))].update(skipped=True),
            'a raised case': lambda r: r['cases'][next(iter(r['cases']))].update(error='x'),
            'a missing check': lambda r: r['checks'].pop(),
            'no plan': lambda r: r.update(plan=[]),
            'no patch': lambda r: r.update(admission_patch={}),
            'other writer bytes': lambda r: r['sources'].update({'ordered_cache.py': 'f' * 64}),
        }
        for label, mutate in mutations.items():
            report = copy.deepcopy(self.report)
            mutate(report)
            if label not in ('a checkpoint', 'an error', 'one seed', 'one width', 'one writer', 'no replay', 'two chips'):
                report['decision'] = card.decide(report)
                report['verdict_line'] = card.verdict_line(report)
            else:
                report['decision'] = card.decide(report) if label != 'an error' else dict(verdict='NO-DECISION')
                report['verdict_line'] = card.verdict_line(report) if label != 'an error' else 'ORDERED_WRITER verdict=NO-DECISION'
            before = {name: (self.dir / name).read_bytes() for name in ('page_width_tp4.py', 'ordered_writer_evidence_tp4.json')}
            code, lines = self.run_recorder(report)
            self.assertEqual(code, 1, (label, lines))
            self.assertTrue(any(line.startswith('REFUSED') or 'problems left' in line for line in lines), (label, lines))
            self.assertEqual(before, {name: (self.dir / name).read_bytes() for name in before}, label)

    def test_a_changed_ordered_cache_since_the_run_is_refused(self):
        (self.dir / 'ordered_cache.py').write_bytes((self.dir / 'ordered_cache.py').read_bytes() + b'\n# edit\n')
        code, lines = self.run_recorder()
        self.assertEqual(code, 1)
        self.assertTrue(any('ordered_cache.py' in line for line in lines), lines)

    def test_commit_and_image_and_run_are_checked_and_the_record_is_public_safe(self):
        for key, value, word in (('commit', 'abc', '--commit'), ('image', 'registry.example/x@sha256:' + 'a' * 64, '--image'),
                                 ('tag', '', '--run and --tag')):
            code, lines = self.run_recorder(**{key: value})
            self.assertEqual(code, 1, key)
        report = copy.deepcopy(self.report)
        report['verdict_line'] += ' /opt/results/x'
        code, lines = self.run_recorder(report)
        self.assertEqual(code, 1, lines)
        self.assertTrue(any('host path' in line for line in lines), lines)

    def test_the_shipped_skeleton_is_pending_and_pinned(self):
        evidence = json.loads(pw.EVIDENCE.read_text())
        self.assertEqual(evidence['status'], 'PENDING')
        self.assertEqual(hashlib.sha256(pw.EVIDENCE.read_bytes()).hexdigest(), pw.ORDERED_WRITER_EVIDENCE_TP4_SHA256)
        self.assertNotIn(b'\r', pw.EVIDENCE.read_bytes())
        self.assertEqual(len(rec.re.findall(r'^ORDERED_WRITER_EVIDENCE_TP4_SHA256 = ', (pw.HERE / 'page_width_tp4.py').read_text(),
                                            flags=rec.re.M)), 1)


if __name__ == '__main__':
    unittest.main()
