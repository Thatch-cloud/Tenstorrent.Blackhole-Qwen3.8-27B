"""page_width_tp4: width 4,096 is admitted only at four cards, on a passing E1 record at its pin, for the live ordered_cache.py.

The pinned ordered_cache is not edited: its answer (up to 1,024 entries, and 2,052) is page_width_tp4's answer at the pair and at
four cards, byte for byte, whatever the record says."""

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

import ordered_cache  # noqa: E402
import ordered_cache_tp  # noqa: E402
import packed_ordered_cache as poc  # noqa: E402
import page_width_tp4 as pw  # noqa: E402
import tp_shapes  # noqa: E402

TP4 = {tp_shapes.TP_SWITCH: '4'}
PAIR = {}


def passing(root):
    return dict(
        schema=pw.EVIDENCE_SCHEMA, status='PASS', run=123, tag='v1', commit='a' * 40, card='M', image='tt-vllm:test',
        scope='full', chips='1of4', kv_heads=1, failures=0, widths=[2052, 4096], writers=['chained64', 'tiles32'],
        seeds=[0, 1, 2], sections=list(pw.SECTIONS), entries=list(pw.ENTRIES), anchor_positions=[262080, 262111],
        counts=dict(checks=10, exact=10),
        sources={'ordered_cache.py': pw.sha256_file(Path(root) / 'ordered_cache.py')})


def write(directory, evidence):
    path = Path(directory) / 'ordered_writer_evidence_tp4.json'
    payload = (json.dumps(evidence, indent=1) + '\n').encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


class PinnedAnswerTests(unittest.TestCase):
    def test_every_pinned_width_stays_admitted_and_the_rest_do_not_move(self):
        for environ in (PAIR, TP4):
            for width in (1, 512, 1024, 2052):
                self.assertTrue(pw.admitted(width, environ), (environ, width))
            for width in (0, -1, 1025, 2048, 2051, 2053, 4100, 4104, 8192, 4096.0, True, '4096', None):
                if width == 4096.0 or width == 4096:
                    continue
                self.assertEqual(pw.admitted(width, environ), ordered_cache.page_width_admitted(width), (environ, width))
        self.assertFalse(ordered_cache.page_width_admitted(4096), 'the pinned file must not admit 4,096')
        self.assertEqual(ordered_cache.WIDE_PAGE_WIDTHS, frozenset({2052}))

    def test_the_shipped_record_is_pass_and_4096_is_admitted_at_four_cards_only(self):
        """Recorded by ship/262k-prefix (the E1 window on the tp4/w1 image): four cards admit 4,096, the pair never does."""
        ok, problems = pw.evidence_state()
        self.assertEqual(problems, [])
        self.assertTrue(ok)
        self.assertEqual(json.loads(pw.EVIDENCE.read_text())['status'], 'PASS')
        self.assertTrue(pw.admitted(4096, TP4))
        self.assertFalse(pw.admitted(4096, PAIR))
        self.assertEqual(hashlib.sha256(pw.EVIDENCE.read_bytes()).hexdigest(), pw.ORDERED_WRITER_EVIDENCE_TP4_SHA256)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        shutil.copyfile(pw.HERE / 'ordered_cache.py', self.dir / 'ordered_cache.py')
        self.evidence = passing(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def admitted(self, evidence=None, pin=None, environ=TP4, root=None):
        path, digest = write(self.dir, self.evidence if evidence is None else evidence)
        return pw.admitted(4096, environ, path=path, expected=digest if pin is None else pin,
                           sources_root=self.dir if root is None else root)

    def test_a_passing_record_at_its_pin_admits_4096_at_four_cards_only(self):
        self.assertTrue(self.admitted())
        self.assertFalse(self.admitted(environ=PAIR), 'the pair never admits 4,096')

    def test_a_wrong_pin_is_refused(self):
        self.assertFalse(self.admitted(pin='1' * 64))

    def test_a_record_that_does_not_qualify_is_refused(self):
        for label, mutate in (
                ('pending', lambda e: e.update(status='PENDING')),
                ('failures', lambda e: e.update(failures=1)),
                ('reduced scope', lambda e: e.update(scope='reduced')),
                ('two-chip view', lambda e: e.update(chips='1of2')),
                ('two heads', lambda e: e.update(kv_heads=2)),
                ('no control width', lambda e: e.update(widths=[4096])),
                ('one writer', lambda e: e.update(writers=['chained64'])),
                ('one seed', lambda e: e.update(seeds=[0])),
                ('no replay', lambda e: e.update(sections=['eager', 'complete_cache'])),
                ('an entry missing', lambda e: e.update(entries=[0, 1])),
                ('anchors elsewhere', lambda e: e.update(anchor_positions=[0, 31])),
                ('inexact checks', lambda e: e.update(counts=dict(checks=10, exact=9))),
                ('no checks', lambda e: e.update(counts=dict(checks=0, exact=0))),
                ('no run', lambda e: e.pop('run')),
                ('wrong schema', lambda e: e.update(schema='other/1')),
                ('no source hash', lambda e: e.update(sources={})),
                ('other writer bytes', lambda e: e.update(sources={'ordered_cache.py': 'f' * 64}))):
            evidence = copy.deepcopy(self.evidence)
            mutate(evidence)
            self.assertFalse(self.admitted(evidence), label)

    def test_a_changed_ordered_cache_invalidates_the_record(self):
        (self.dir / 'ordered_cache.py').write_bytes((self.dir / 'ordered_cache.py').read_bytes() + b'\n# edit\n')
        self.assertFalse(self.admitted(evidence=self.evidence))

    def test_4100_and_4104_and_other_wide_widths_never_admit_even_with_the_record(self):
        path, digest = write(self.dir, self.evidence)
        for width in (4100, 4104, 4097, 4095, 3000, 2053, 8192, 4096.5):
            self.assertFalse(pw.admitted(width, TP4, path=path, expected=digest, sources_root=self.dir), width)

    def test_a_missing_or_malformed_file_is_refused(self):
        self.assertFalse(pw.admitted(4096, TP4, path=self.dir / 'absent.json', sources_root=self.dir))
        path = self.dir / 'bad.json'
        path.write_bytes(b'{not json')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertFalse(pw.admitted(4096, TP4, path=path, expected=digest, sources_root=self.dir))


class CallersTests(unittest.TestCase):
    """The writers ask page_width_tp4 through the module, so one patch moves them all (the harness's scoped patch)."""

    def setUp(self):
        import test_262k_evidence_waiver as waiver

        waiver.pending_records(self)                     # the unrecorded state: "refused: no record yet"
        patcher = unittest.mock.patch.dict(os.environ, {tp_shapes.TP_SWITCH: '4'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_four_card_writers_validate_through_page_width_tp4(self):
        cache = (4112, 1, 64, 256)
        for rows in (32,):
            args = (cache, (1, rows, 32, 256), (rows,), (rows, 4096))
            with self.assertRaisesRegex(ValueError, 'page-table'):
                ordered_cache_tp.validate_shapes(*args)          # refused: no record yet
        with unittest.mock.patch.object(pw, 'admitted', lambda width, *a, **k: width == 4096 or ordered_cache.page_width_admitted(width)):
            self.assertEqual(ordered_cache_tp.validate_shapes(cache, (1, 32, 32, 256), (32,), (32, 4096)), 32)
            self.assertEqual(poc.validate_chained(cache, (1, 64, 32, 256), (64,), (64, 4096), 64), 64)
            self.assertEqual(poc.validate_chained(cache, (1, 32, 32, 256), (32,), (32, 2052), 32), 32)
        with self.assertRaises(ValueError):
            poc.validate_chained(cache, (1, 64, 32, 256), (64,), (64, 4096), 64)

    def test_the_pinned_validate_shapes_still_refuses_4096(self):
        with self.assertRaises(ValueError):
            ordered_cache.validate_shapes((4112, 2, 64, 256), (1, 32, 32, 256), (32,), (32, 4096))

    def test_the_cb3_page_is_the_cost_of_the_wider_table(self):
        self.assertEqual(poc.cb_bytes(2052), 390160)
        self.assertEqual(poc.cb_bytes(4096), 390160 + 8176)


import unittest.mock  # noqa: E402

if __name__ == '__main__':
    unittest.main()
