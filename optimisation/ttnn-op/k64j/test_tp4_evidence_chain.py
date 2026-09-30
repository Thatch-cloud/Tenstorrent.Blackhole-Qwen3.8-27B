"""CPU: the four-card evidence chain end to end on the fakes - harness report, recorder, admission.

Each one-card evidence job's harness runs on its fake ttnn (the flows of test_k64j_one_head and test_extent_reader_card_b):
k64j_card_b.py --kv-heads 1 for CB1-TP4 (EV-F1) and CB2a-TP4 (EV-F2), extent_reader_card_b.py --width 4 for CB2b-TP4 (EV-F3).
scripts/ci/record_packed_any_evidence_tp4.py's builders then record each report exactly as its main() does, and
packed_any_admission.evidence_problems judges the resulting record at four cards (tp=4). The record made this way must
qualify four cards and nothing else, and the pair's reports (two KV heads, 0x27, chips=1of2, the pinned reader) must be
refused by the recorder.

The fakes run small scopes, so the design's sizes (the seeds, K1's extents, K2's ticket set and the X7 / Z floors, CB2b's
capacity, families and residues) are narrowed to what each run covers - in the harness's own coverage judgement and in the
admission alike - and the fake binary's sha stands in for K64j's while a report is read. Everything else, every field, word
and count the recorder takes, is the harness's own output: a harness that stops writing one breaks the chain here, not on
the card after the window.

    py -3.11 -B -m unittest test_tp4_evidence_chain      (from this directory)
"""

import contextlib
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_k64j_one_head as one_head_tests  # noqa: E402 - the one-head flows; sys.path: k64j_probe, sdpa_decode_qwen, scripts/ci
import test_extent_reader_card_b as reader_tests  # noqa: E402 - the width-4 flow

import packed_any_admission as admission  # noqa: E402 - scripts/ci
import record_packed_any_evidence_tp4 as recorder  # noqa: E402 - scripts/ci

card_b = one_head_tests.card_b
reader_b = reader_tests.reader_b
CI = reader_tests.CI
RUNS = dict(CB1=36400000001, CB2a=36400000002, CB2b=36400000003)
TAGS = dict(CB1='experiment/c2-serving-v201', CB2a='experiment/c2-serving-v202', CB2b='experiment/c2-serving-v204')
COMMIT = 'f3e016d1' * 5
IMAGE = 'tp4-stackfix-3'

# The narrowed design: exactly what the fake runs below cover.
SEEDS = (0,)
K1_EXTENTS = (512, 2304, 4352)                                   # one_head_tests.OneHeadFlow.BASE
CB2_EXTENTS = (2304, 4352)                                       # one_head_tests.OneHeadCB2aFlow.BASE
Z_STARTS = (0, 32, 240)
CB2A_EXTRA = ['--variants', 'normal,peaky', '--z-starts', ','.join(map(str, Z_STARTS))]
K2_TICKETS = ((card_b.K2_SWEEP[1] - card_b.K2_SWEEP[0] + 1 + len(CB2_EXTENTS) * len(card_b.CB2_STARTS))
              * len(SEEDS) * len(admission.CB2A_VARIANTS))       # 183 tickets per (seed, variant): 366
CB2B_CAPACITY = 2304                                             # reader_tests.FlowBase.FULL
CB2B_NAMED = (256, 512, 2304)
CB2B_FAMILIES = 6
CB2B_RESIDUES = (0, 7, 240, 255)


def patched(target, **values):
    return [mock.patch.object(target, name, value) for name, value in values.items()]


@contextlib.contextmanager
def narrowed_design(binary_sha256=None):
    """The admission's (and so the recorder's) design sizes narrowed to the fake runs'; with `binary_sha256`, the fake binary
    stands in for K64j's (only while a report is read: the record keeps the skeleton's K64j binary)."""
    values = dict(SEEDS=SEEDS, K1_EXTENTS=K1_EXTENTS, CB2_EXTENTS=CB2_EXTENTS, Z_STARTS=Z_STARTS, CB2A_K2_TICKETS=K2_TICKETS,
                  X7_FLOOR=len(CB2_EXTENTS) * len(admission.CB2_STARTS) * len(SEEDS) * len(admission.CB2A_VARIANTS) * 2,
                  Z_FLOOR=len(admission.Z_FAMILIES) * len(Z_STARTS) * len(SEEDS) * 2,
                  CB2B_CAPACITY=CB2B_CAPACITY, CB2B_SEEDS=SEEDS, CB2B_RESIDUES=CB2B_RESIDUES,
                  CB2B_R2_MIN_FAMILIES=CB2B_FAMILIES, CB2B_R2_NAMED=CB2B_NAMED)
    if binary_sha256 is not None:
        values['K64J_TTNNCPP_SHA256'] = binary_sha256
    with contextlib.ExitStack() as stack:
        for patch in patched(admission, **values):
            stack.enter_context(patch)
        yield


def run_flow(cls, method, run, *patches):
    """One of the harness flow tests' runs (its fixture set up and torn down): (report, sha256 of its bytes, file name, the
    fake binary's sha256). The report is read back from the file the harness wrote, as the recorder reads it."""
    case = cls(method)
    case.setUp()
    try:
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            status, _report = run(case)
        path = sorted(case.dir.glob('*.json'))
        assert len(path) == 1, path
        payload = path[0].read_bytes()
        binary = hashlib.sha256(case.binary.read_bytes()).hexdigest()
        return json.loads(payload.decode('utf-8')), hashlib.sha256(payload).hexdigest(), path[0].name, binary, status
    finally:
        case.tearDown()


class EvidenceChainTests(unittest.TestCase):
    """The three one-card evidence runs at four cards (and two of the pair's), each once for the class."""

    @classmethod
    def setUpClass(cls):
        import torch

        one, cb2a, reader = one_head_tests.OneHeadFlow, one_head_tests.OneHeadCB2aFlow, reader_tests.QuadFlowTests
        cls.runs = dict(
            CB1=run_flow(one, 'test_cb1_passes_end_to_end_at_one_kv_head',
                         lambda case: case.run_card(one_head_tests.FakeExtentTtnn(case.torch))),
            # K2's full coverage is the design's set: narrowed to the run's seed and CB2 extents (both variants), so the
            # harness itself judges K2 a PASS, not a REDUCED-PASS.
            CB2a=run_flow(cb2a, 'test_cb2a_passes_end_to_end_at_one_kv_head',
                          lambda case: case.run_card(one_head_tests.FakeExtentTtnn(case.torch), CB2A_EXTRA),
                          *patched(card_b, K2_DESIGN_SEEDS=SEEDS, CB2_EXTENTS=CB2_EXTENTS)),
            # CB2b's full scope narrowed to FlowBase.FULL (C = 2,304, six families, seed 0), so the harness says scope=full.
            CB2b=run_flow(reader, 'test_pass_end_to_end',
                          lambda case: case.run_reader(reader_tests.FakeQuadTtnn(torch), patches=patched(
                              reader_b, CAPACITY=CB2B_CAPACITY, R2_NAMED=CB2B_NAMED, R2_MIN_FAMILIES=CB2B_FAMILIES,
                              DESIGN_RESIDUES=CB2B_RESIDUES, SEEDS=SEEDS))))
        pair_cb1 = [word for word in one.BASE if word not in ('--kv-heads', '1')]
        cls.pair = dict(
            CB1=run_flow(one, 'test_the_pair_report_has_no_one_head_words',
                         lambda case: case.run_card(one_head_tests.FakeExtentTtnn(case.torch), ['--sections', 'N']),
                         mock.patch.object(one, 'BASE', pair_cb1)),
            CB2b=run_flow(reader_tests.DryRunTests, 'test_pass_end_to_end',
                          lambda case: case.run_reader(reader_tests.FakeReaderTtnn(torch), patches=patched(
                              reader_b, CAPACITY=CB2B_CAPACITY, R2_NAMED=CB2B_NAMED, R2_MIN_FAMILIES=CB2B_FAMILIES,
                              DESIGN_RESIDUES=CB2B_RESIDUES, SEEDS=SEEDS))))

    def build(self, name, runs=None):
        """recorder.build_<name> on a run's report, as record_packed_any_evidence_tp4.main calls it."""
        report, digest, file_name, binary, _status = (runs or self.runs)[name]
        path = str(Path('results') / file_name)
        with narrowed_design(binary):
            if name == 'CB1':
                return recorder.build_cb1(report, digest, path, RUNS[name], TAGS[name])
            if name == 'CB2a':
                return recorder.build_cb2a(report, digest, path, RUNS[name], TAGS[name])
            return recorder.build_cb2b(report, digest, path, RUNS[name], TAGS[name], COMMIT, IMAGE, root=str(CI))

    def test_the_runs_pass_at_full_scope(self):
        """Each run is a PASS of the (narrowed) design's whole set by the harness's own judgement: never a reduced pass."""
        for name, (report, _digest, _file, _binary, status) in self.runs.items():
            with self.subTest(section=name):
                self.assertEqual((status, report['passed'], report['decision']['verdict'], report['failures']),
                                 (0, True, 'PASS', []))
        self.assertEqual(self.runs['CB2a'][0]['decision']['k2'], 'PASS')
        self.assertEqual(self.runs['CB2a'][0]['decision']['k2_coverage'],
                         dict(full=True, covered=K2_TICKETS, design=K2_TICKETS, short=[]))
        self.assertEqual(self.runs['CB2b'][0]['decision']['scope'], 'full')

    def test_each_report_is_recorded_with_the_four_card_words(self):
        cb1 = self.build('CB1')
        self.assertEqual((cb1['status'], cb1['kv_heads'], cb1['seeds'], cb1['extents'], cb1['failures']),
                         ('PASS', 1, list(SEEDS), list(K1_EXTENTS), 0))
        self.assertIs(type(cb1['kv_heads']), int)
        self.assertEqual(cb1['combos'], [dict(geometry='G4B3', flags=['0x21', '0x23']),
                                         dict(geometry='G8B2', flags=['0x21', '0x23'])])
        self.assertTrue(all(total and passed == total for passed, total in cb1['counts'].values()), cb1['counts'])
        cb2a = self.build('CB2a')
        self.assertEqual((cb2a['status'], cb2a['kv_heads'], cb2a['local_heads'], cb2a['k2']['verdict'], cb2a['k2']['tickets']),
                         ('PASS', 1, 6, 'PASS', [K2_TICKETS, K2_TICKETS]))
        self.assertEqual(cb2a['z']['families'], list(admission.Z_FAMILIES))
        cb2b, sha = self.build('CB2b')
        live = hashlib.sha256((CI / recorder.READER_TP).read_bytes()).hexdigest()
        self.assertEqual((cb2b['status'], cb2b['scope'], cb2b['chips'], cb2b['capacity'], cb2b['served']['flags'], sha),
                         ('PASS', 'full', '1of4', CB2B_CAPACITY, '0x23', live))
        self.assertEqual(cb2b['served'], dict(flags='0x23', rows=8, batch=2, segments=4))
        self.assertEqual(sorted(cb2b['r1_geometries']), sorted(admission.CB2B_R1_GEOMETRIES))
        self.assertEqual(set(cb2b['pinned_modules']), set(recorder.PINNED_SIBLINGS))
        self.assertTrue(all(len(value) == 64 for value in cb2b['pinned_modules'].values()))

    def test_the_record_qualifies_four_cards_and_not_the_pair(self):
        sections = dict(CB1=self.build('CB1'), CB2a=self.build('CB2a'))
        sections['CB2b'], sha = self.build('CB2b')
        skeleton = json.loads((CI / admission.EVIDENCE_TP4.name).read_text(encoding='utf-8'))
        evidence = recorder.record(skeleton, sections, (36400000000, 'experiment/c2-serving-v200'))
        evidence['sources'] = dict(skeleton.get('sources') or {}, **{recorder.READER_TP: sha})
        self.assertEqual(recorder.hygiene_problems(evidence), [])
        self.assertEqual(json.loads(recorder.dump(evidence).decode('utf-8')), evidence)
        with narrowed_design():                  # the skeleton's K64j binary and kernels, as the image's attach reads them
            self.assertEqual(admission.evidence_problems(evidence, CI, tp=4), [])
            # The same record is not the pair's: its chips, its flags and its reader are the four-card ones.
            pair = admission.evidence_problems(evidence, CI, tp=2)
        self.assertTrue(any('chips 1of4' in problem for problem in pair), pair)
        self.assertTrue(any('no G8B2 0x27 combo' in problem for problem in pair), pair)
        self.assertTrue(any('extent_attention_replay.py' in problem for problem in pair), pair)
        # And the design's real sizes are not met by these small runs: the narrowing is what lets them qualify.
        self.assertTrue(admission.evidence_problems(evidence, CI, tp=4))

    def test_the_pairs_reports_are_refused_by_the_four_card_recorder(self):
        with self.assertRaises(recorder.RecordError) as caught:
            self.build('CB1', self.pair)
        problems = '\n'.join(caught.exception.problems)
        self.assertIn('kv_heads is None, not 1', problems)
        with self.assertRaises(recorder.RecordError) as caught:
            self.build('CB2b', self.pair)
        problems = '\n'.join(caught.exception.problems)
        self.assertIn('chips 1of2 on the verdict line', problems)
        self.assertIn('served flags 0x27, not 0x23', problems)
        pair_reader = hashlib.sha256((CI / 'extent_attention_replay.py').read_bytes()).hexdigest()
        self.assertIn('the run loaded extent_attention_replay_tp.py at %s' % pair_reader[:16], problems)
        self.assertIn('holds no sha256 of %s' % ','.join(recorder.PINNED_SIBLINGS), problems)

    def test_a_report_that_lost_a_four_card_word_is_refused(self):
        """What the recorder reads is the harness's own output: without it the section stays PENDING."""
        for name, key, needle in (('CB1', 'kv_heads', 'kv_heads is None, not 1'),
                                  ('CB2a', 'kv_heads', 'kv_heads is None, not 1'),
                                  ('CB2b', 'served', 'served flags None, not 0x23'),
                                  ('CB2b', 'chip_view', 'chips None on the verdict line')):
            with self.subTest(section=name, key=key):
                report, digest, file_name, binary, status = self.runs[name]
                report = json.loads(json.dumps(report))
                report.pop(key)
                if key == 'kv_heads':
                    report['verdict_line'] = report['verdict_line'].replace(' kv_heads=1', '')
                if key == 'chip_view':
                    report['verdict_line'] = report['verdict_line'].replace(' chips=1of4', '')
                with self.assertRaises(recorder.RecordError) as caught:
                    self.build(name, {name: (report, digest, file_name, binary, status)})
                self.assertIn(needle, '\n'.join(caught.exception.problems))


if __name__ == '__main__':
    unittest.main()
