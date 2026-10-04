"""packed_any_admission at the 262,144-token window (QWEN_FAST_MAX_POSITION=262144, page-table width 4,096, four cards).

The same admission at a second capacity, behind its OWN evidence: packed_any_evidence_tp4_262144.json at EVIDENCE_TP4_262K_SHA256
(a PENDING skeleton until E2 records it) and the ordered writers' E1 record (page_width_tp4). The 131,328 record and its pin are
never re-pinned, the gate-only UNQUALIFIED waiver is never applied at 262,144, and the 131,328 attach logs exactly what it always
did."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import page_width_tp4  # noqa: E402
import test_packed_any_admission as pair_tests  # noqa: E402
import test_packed_any_admission_tp4 as quad_tests  # noqa: E402

M3 = pair_tests.M3
WIDE = dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='262144')
WIDE_GATE = dict(WIDE, QWEN_C2_GATE='1', QWEN_C2_GATE_PROFILE='1')
CAPACITY = 262144
E1_OK = (True, [])
E1_PENDING = (False, ['status PENDING, not PASS'])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def qualifying262():
    """The 131,328 four-card record with every section widened to the 262,144 design set, as the recorder writes it."""
    evidence = quad_tests.qualifying()
    evidence['capacity'] = CAPACITY
    cb1 = evidence['sections']['CB1']
    cb1.update(extents=list(admission.K1_EXTENTS_262K), capacity=CAPACITY)
    cb2a = evidence['sections']['CB2a']
    cb2a.update(capacity=CAPACITY, cb2_extents=list(admission.CB2_EXTENTS_262K))
    cb2a['k2'] = dict(cb2a['k2'], tickets=[2030, 2030])
    cb2a['x7'] = [600, 600]
    cb2b = evidence['sections']['CB2b']
    cb2b.update(capacity=CAPACITY, r2_families=sorted(set(cb2b['r2_families']) | {CAPACITY}))
    return evidence


class Lines(list):
    def __call__(self, template, *values):
        self.append(template.format(*values))


class DesignTests(unittest.TestCase):
    def test_the_131k_entry_is_exactly_todays_constants(self):
        design = admission.capacity_design(131328)
        self.assertEqual((design['k1_extents'], design['cb2_extents'], design['k2_tickets'], design['x7_floor'],
                          design['z_floor'], design['r2_named']),
                         (admission.K1_EXTENTS, admission.CB2_EXTENTS, 1980, 500, 900, admission.CB2B_R2_NAMED))
        self.assertEqual(admission.CB2A_K2_TICKETS, 1980)
        self.assertEqual(admission.CB2B_CAPACITY, 131328)

    def test_the_262k_entry_reaches_the_full_window(self):
        design = admission.capacity_design(CAPACITY)
        self.assertEqual(design['k1_extents'], admission.K1_EXTENTS + (196864, 262144))
        self.assertEqual(design['cb2_extents'], admission.CB2_EXTENTS + (262144,))
        self.assertEqual((design['k2_tickets'], design['x7_floor'], design['z_floor']), (2030, 600, 900))
        self.assertEqual(design['r2_named'], admission.CB2B_R2_NAMED + (262144,))

    def test_k2_coverage_of_the_harness_matches_the_floor_at_both_capacities(self):
        sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'k64j'))
        sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'k64j_probe'))
        sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'sdpa_decode_qwen'))
        import k64j_card_b as harness

        for capacity, tickets in ((131328, 1980), (CAPACITY, 2030)):
            extents = harness.cb2_extents_for(capacity)
            self.assertEqual(extents, admission.capacity_design(capacity)['cb2_extents'])
            comparisons = []
            for seed in harness.K2_DESIGN_SEEDS:
                for variant in harness.K2_DESIGN_VARIANTS:
                    for start in range(harness.K2_SWEEP[0], harness.K2_SWEEP[1] + 1):
                        comparisons.append(dict(kind='k2_native_vs_extent', seed=seed, variant=variant, ticket='sweep',
                                                start=start))
                    for extent in extents:
                        for offset in harness.CB2_STARTS:
                            comparisons.append(dict(kind='k2_native_vs_extent', seed=seed, variant=variant,
                                                    ticket='family', extent=extent,
                                                    start=extent - harness.K_CHUNK + offset))
            coverage = harness.k2_coverage(dict(capacity=capacity, comparisons=comparisons))
            self.assertEqual((coverage['full'], coverage['covered'], coverage['design']), (True, tickets, tickets), capacity)
            # a 131k design run is short at 262k (the full window's tickets are missing) and the reverse is not full
            other = 262144 if capacity == 131328 else 131328
            self.assertEqual(harness.k2_coverage(dict(capacity=other, comparisons=comparisons))['full'], capacity == 262144)
        self.assertIsNone(harness.cb2_extents_for(200000))

    def test_the_served_capacity_is_max_position_rounded_to_a_page(self):
        self.assertEqual(admission.served_capacity({}), 131328)
        self.assertEqual(admission.served_capacity({'QWEN_FAST_MAX_POSITION': ''}), 131328)
        self.assertEqual(admission.served_capacity({'QWEN_FAST_MAX_POSITION': '131328'}), 131328)
        self.assertEqual(admission.served_capacity({'QWEN_FAST_MAX_POSITION': '131312'}), 131328)
        self.assertEqual(admission.served_capacity({'QWEN_FAST_MAX_POSITION': '262144'}), 262144)
        self.assertEqual(admission.served_capacity({'QWEN_FAST_MAX_POSITION': '262080'}), 262080)
        for bad in ('abc', '-1', '0', '1.5', '0262144', ' 262144'):
            with self.assertRaises(ValueError, msg=bad):
                admission.served_capacity({'QWEN_FAST_MAX_POSITION': bad})
        self.assertIsNone(admission.capacity_problem(131328))
        self.assertIsNone(admission.capacity_problem(262144))
        self.assertIn('neither 131328 nor 262144', admission.capacity_problem(200000))


class RecordTests(unittest.TestCase):
    def setUp(self):
        import test_262k_evidence_waiver as waiver

        waiver.pending_records(self)       # these tests describe the unrecorded state; test_ship_262k_prefix reads the real files

    def test_the_262k_skeleton_is_pinned_pending_lf_and_the_131k_pin_did_not_move(self):
        data = admission.EVIDENCE_TP4_262K.read_bytes()
        self.assertEqual(sha(data), admission.EVIDENCE_TP4_262K_SHA256)
        self.assertNotIn(b'\r', data)
        record = json.loads(data.decode())
        self.assertEqual(record['capacity'], CAPACITY)
        self.assertEqual({name: section['status'] for name, section in record['sections'].items()},
                         {'CB1': 'PENDING', 'CB2a': 'PENDING', 'CB2b': 'PENDING'})
        # the 131,328 record's pin is the one E-less I1 shipped with (a 262k record never re-pins it)
        self.assertEqual(admission.EVIDENCE_TP4_SHA256, 'd221f68f494e7b1a7571cefdfc28a986458d1ae4511e00a8bf56bfc79aa262cf')
        self.assertEqual(sha(admission.EVIDENCE_TP4.read_bytes()), admission.EVIDENCE_TP4_SHA256)
        self.assertEqual(admission.EVIDENCE_SHA256, 'a9407e9a54bfbaa7a0264456644ef6136e376f600ce8f0490417f2b5406515ee')

    def test_the_skeleton_qualifies_nothing_at_262k(self):
        problems = admission.evidence_problems(json.loads(admission.EVIDENCE_TP4_262K.read_text()), HERE, tp=4, capacity=CAPACITY)
        self.assertTrue(any(problem.startswith('CB1: status PENDING') for problem in problems), problems)
        self.assertTrue(any('no sha256 recorded' in problem for problem in problems), problems)

    def test_the_widened_record_qualifies_at_262k_and_the_131k_record_does_not(self):
        self.assertEqual(admission.evidence_problems(qualifying262(), HERE, tp=4, capacity=CAPACITY), [])
        self.assertNotEqual(admission.evidence_problems(quad_tests.qualifying(), HERE, tp=4, capacity=CAPACITY), [])
        record = json.loads(admission.EVIDENCE_TP4.read_text())
        problems = admission.evidence_problems(record, HERE, tp=4, capacity=CAPACITY)
        self.assertTrue(any(problem.startswith('capacity:') for problem in problems), problems)
        self.assertTrue(any('CB2b: capacity 131328, not the served C = 262144' in problem for problem in problems), problems)
        self.assertTrue(any('K2 tickets' in problem and '2030' in problem for problem in problems), problems)
        self.assertTrue(any(problem.startswith('CB1: extents lack') for problem in problems), problems)
        self.assertTrue(any(problem.startswith('CB2b: R2 families lack the named [262144]') for problem in problems), problems)

    def test_the_262k_record_is_not_a_131k_record_either(self):
        problems = admission.evidence_problems(qualifying262(), HERE, tp=4)
        self.assertTrue(any('capacity 262144, not the served C = 131328' in problem for problem in problems), problems)

    def test_each_262k_condition_refuses_alone(self):
        cases = {
            'record capacity missing': lambda e: e.pop('capacity'),
            'record capacity 131k': lambda e: e.update(capacity=131328),
            'CB1 without 262144': lambda e: e['sections']['CB1'].update(extents=list(admission.K1_EXTENTS_262K[:-1])),
            'CB1 without 196864': lambda e: e['sections']['CB1'].update(
                extents=[x for x in admission.K1_EXTENTS_262K if x != 196864]),
            'CB1 capacity 131k': lambda e: e['sections']['CB1'].update(capacity=131328),
            'CB2a capacity missing': lambda e: e['sections']['CB2a'].pop('capacity'),
            'CB2a families without 262144': lambda e: e['sections']['CB2a'].update(cb2_extents=list(admission.CB2_EXTENTS)),
            'K2 at the 131k ticket count': lambda e: e['sections']['CB2a']['k2'].update(tickets=[1980, 1980]),
            'K2 reduced': lambda e: e['sections']['CB2a']['k2'].update(tickets=[2000, 2030]),
            'X7 at the 131k floor': lambda e: e['sections']['CB2a'].update(x7=[500, 500]),
            'Z short': lambda e: e['sections']['CB2a']['z'].update(passed=[10, 10]),
            'CB2b at 131k': lambda e: e['sections']['CB2b'].update(capacity=131328),
            'CB2b without the named window': lambda e: e['sections']['CB2b'].update(
                r2_families=[f for f in e['sections']['CB2b']['r2_families'] if f != CAPACITY]),
            'CB2b family beyond the window': lambda e: e['sections']['CB2b']['r2_families'].append(262400),
            'CB2b reduced': lambda e: e['sections']['CB2b'].update(scope='reduced'),
            'one seed': lambda e: e['sections']['CB1'].update(seeds=[0]),
        }
        for name, mutate in cases.items():
            evidence = qualifying262()
            mutate(evidence)
            with self.subTest(name):
                self.assertNotEqual(admission.evidence_problems(evidence, HERE, tp=4, capacity=CAPACITY), [], name)

    def test_there_is_no_pair_record_for_the_wide_window(self):
        problems = admission.evidence_problems(qualifying262(), HERE, tp=2, capacity=CAPACITY)
        self.assertEqual(len(problems), 1)
        self.assertIn('four-card window', problems[0])

    def test_the_width_and_capacity_select_the_file_and_the_pin(self):
        self.assertEqual((admission.evidence_path(4, CAPACITY), admission.evidence_pin(4, CAPACITY)),
                         (admission.EVIDENCE_TP4_262K, admission.EVIDENCE_TP4_262K_SHA256))
        self.assertEqual((admission.evidence_path(4), admission.evidence_pin(4)),
                         (admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_SHA256))
        self.assertEqual((admission.evidence_path(2), admission.evidence_pin(2)),
                         (admission.EVIDENCE, admission.EVIDENCE_SHA256))

    def test_a_record_on_disk_is_checked_at_its_own_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wide.json'
            path.write_text(json.dumps(qualifying262(), indent=1) + '\n')
            digest = sha(path.read_bytes())
            record = admission.check_evidence(path, expected_sha256=digest, sources_root=HERE, tp=4, capacity=CAPACITY)
            self.assertEqual(record['capacity'], CAPACITY)
            with self.assertRaises(admission.AdmissionRefused) as refused:
                admission.check_evidence(path, expected_sha256='1' * 64, sources_root=HERE, tp=4, capacity=CAPACITY)
            self.assertIn('not the reviewed', refused.exception.problems[0])
            with self.assertRaises(admission.AdmissionRefused) as refused:
                admission.check_evidence(sources_root=HERE, tp=4, capacity=CAPACITY)       # the shipped skeleton
            self.assertTrue(any('PENDING' in problem for problem in refused.exception.problems))
            with self.assertRaisesRegex(ValueError, 'not one this admission has evidence for'):
                admission.check_evidence(path, expected_sha256=digest, sources_root=HERE, tp=4, capacity=200000)


class AdmitTests(unittest.TestCase):
    def setUp(self):
        import test_262k_evidence_waiver as waiver

        waiver.pending_records(self)
        patcher = mock.patch.dict(admission._STATE, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = Lines()
        self.runtime = dict(binaries={'build_Release/lib/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256,
                                      'build_Release/ttnn/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})

    def admit(self, environ, wide=None, writer=E1_OK, m3=M3, shipped=False):
        """`wide`: the 262k record a passing read returns (None: the real, shipped skeleton). The 131k record is the committed one."""
        real = admission.check_evidence

        def read(path=None, **kwargs):
            if kwargs.get('capacity') == CAPACITY and wide is not None:
                return wide
            return real(path, **kwargs)

        with mock.patch.object(admission, 'check_runtime', side_effect=lambda root, binaries: self.runtime), \
                mock.patch.object(admission, 'check_evidence', side_effect=read), \
                mock.patch.object(page_width_tp4, 'evidence_state', return_value=writer), \
                mock.patch.dict(os.environ, environ, clear=True):
            return admission.admit('/opt/tt-metal', m3=m3, environ=dict(environ), log=self.lines)

    def refused(self, environ, **kwargs):
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.admit(environ, **kwargs)
        self.assertFalse(admission.admitted())
        return caught.exception.problems

    def test_a_262k_attach_on_131k_only_evidence_is_refused_naming_the_capacity(self):
        problems = self.refused(WIDE)
        self.assertTrue(any('capacity 262144' in problem for problem in problems), problems)
        self.assertTrue(any('PENDING' in problem for problem in problems), problems)

    def test_a_262k_gate_run_is_refused_too_the_waiver_is_131ks_alone(self):
        self.assertTrue(admission.unqualified_allowed(WIDE_GATE))
        problems = self.refused(WIDE_GATE)
        self.assertTrue(any('capacity 262144' in problem for problem in problems), problems)
        self.assertFalse([line for line in self.lines if admission.UNQUALIFIED_MARKER in line], self.lines)
        self.assertFalse(any('passed' in line for line in self.lines))

    def test_without_e1_the_262k_attach_is_refused_even_on_a_qualifying_record(self):
        problems = self.refused(WIDE, wide=qualifying262(), writer=E1_PENDING)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('page-table width 4096 is not qualified', problems[0])
        self.assertIn('capacity 262144', problems[0])
        self.refused(WIDE_GATE, wide=qualifying262(), writer=E1_PENDING)

    def on_disk(self, evidence):
        """The admission reading `evidence` from a file at its own pin where it reads the shipped 262k record."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'wide.json'
        path.write_text(json.dumps(evidence, indent=1) + chr(10))
        for patcher in (mock.patch.object(admission, 'EVIDENCE_TP4_262K', path),
                        mock.patch.object(admission, 'EVIDENCE_TP4_262K_SHA256', sha(path.read_bytes()))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_reduced_262k_record_is_refused(self):
        reduced = qualifying262()
        reduced['sections']['CB2b']['scope'] = 'reduced'
        self.on_disk(reduced)
        problems = self.refused(WIDE)
        self.assertTrue(any('CB2b: scope reduced' in problem for problem in problems), problems)
        self.assertTrue(all(problem.startswith('capacity 262144: ') for problem in problems), problems)

    def test_the_records_read_from_disk_admit_through_the_real_checks(self):
        self.on_disk(qualifying262())
        record = self.admit(WIDE)
        self.assertEqual(record['capacity'], CAPACITY)
        self.assertTrue(self.lines[-1].endswith(' capacity=262144'), self.lines[-1])

    def test_both_records_admit_and_the_line_and_record_name_the_capacity(self):
        record = self.admit(WIDE, wide=qualifying262())
        self.assertTrue(admission.admitted())
        self.assertEqual(record['capacity'], CAPACITY)
        self.assertTrue(self.lines[-1].startswith('[PINDIAG] packed-any admission passed: K64j'), self.lines[-1])
        self.assertTrue(self.lines[-1].endswith(' capacity=262144'), self.lines[-1])
        self.assertIn(admission.EVIDENCE_TP4_262K_SHA256[:16], self.lines[-1])
        self.assertLess(len(self.lines[-1]), pair_tests.LOG_LINE_LIMIT)

    def test_two_blocks_at_262k_name_both(self):
        record = self.admit(dict(WIDE, QWEN_FAST_M3_BLOCKS='2'), wide=qualifying262(),
                            m3=quad_tests.AdmitTests.blocks_m3(8, 2))
        self.assertEqual((record['blocks'], record['capacity']), (2, CAPACITY))
        self.assertTrue(self.lines[-1].endswith(' blocks=2 capacity=262144'), self.lines[-1])

    def test_the_131k_attach_is_byte_for_byte_what_it_was(self):
        outputs = []
        for environ in (quad_tests.GOOD_ENV, dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='131328'),
                        dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='131312'),
                        dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='')):
            with mock.patch.dict(admission._STATE, clear=True):
                self.lines.clear()
                record = self.admit(environ)
                outputs.append((list(self.lines), sorted(record)))
        self.assertTrue(all(output == outputs[0] for output in outputs), outputs)
        self.assertNotIn('capacity', outputs[0][1])
        self.assertNotIn('capacity=', outputs[0][0][-1])
        self.assertIn(admission.EVIDENCE_TP4_SHA256[:16], outputs[0][0][-1])

    def test_the_131k_record_is_read_with_the_default_capacity_exactly_as_before(self):
        calls = []
        real = admission.check_evidence

        def spy(path=None, **kwargs):
            calls.append(kwargs)
            return real(path, **kwargs)

        with mock.patch.object(admission, 'check_runtime', side_effect=lambda root, binaries: self.runtime), \
                mock.patch.object(admission, 'check_evidence', side_effect=spy):
            admission.admit('/opt/tt-metal', m3=M3, environ=dict(quad_tests.GOOD_ENV), log=self.lines)
        self.assertEqual(calls, [dict(tp=4)])

    def test_any_other_capacity_is_refused_by_name(self):
        for value in ('200000', '131329', '262145', '65536', '524288'):
            with self.subTest(value=value), mock.patch.dict(admission._STATE, clear=True):
                problems = self.refused(dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION=value))
                self.assertTrue(any('neither 131328 nor 262144' in problem for problem in problems), problems)
        problems = self.refused(dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='wide'))
        self.assertTrue(any('QWEN_FAST_MAX_POSITION must be a positive decimal integer' in problem for problem in problems))

    def test_the_pair_never_serves_the_wide_window(self):
        pair = dict(pair_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='262144')
        problems = self.refused(pair, wide=qualifying262())
        self.assertTrue(any('four-card window' in problem for problem in problems), problems)

    def test_every_refusal_is_one_short_line_each(self):
        with self.assertRaises(admission.AdmissionRefused):
            self.admit(WIDE)
        self.assertTrue(self.lines)
        self.assertTrue(all(len(line) < pair_tests.LOG_LINE_LIMIT for line in self.lines), [len(l) for l in self.lines])


class PoolTests(unittest.TestCase):
    STATISTICS = [dict(chip=0, largest_free=4_000_000_000)] * 4

    def setUp(self):
        patcher = mock.patch.dict(admission._STATE, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def pool(self, page_width):
        return SimpleNamespace(extent_replay=True, page_width=page_width, dram_statistics=mock.Mock(return_value=self.STATISTICS))

    def test_a_262k_admission_holds_the_pool_to_4096_pages(self):
        admission._STATE['record'] = dict(capacity=CAPACITY)
        self.assertIs(admission.admit_pool(self.pool(4096), log=Lines()), self.STATISTICS)
        for width in (2052, 4100, 4092, None, '4096'):
            lines = Lines()
            with self.subTest(width=width), self.assertRaises(admission.AdmissionRefused) as caught:
                admission.admit_pool(self.pool(width), log=lines)
            self.assertIn('not the admitted capacity 262144 (4096 pages)', caught.exception.problems[0])
            self.assertEqual(len(lines), 1)

    def test_a_131k_admission_checks_nothing_new(self):
        admission._STATE['record'] = dict(flag=admission.FLAG)
        for width in (2052, 4096, None):
            self.assertIs(admission.admit_pool(self.pool(width), log=Lines()), self.STATISTICS)
        admission._STATE.clear()
        self.assertIs(admission.admit_pool(self.pool(4096), log=Lines()), self.STATISTICS)


class GuardTests(unittest.TestCase):
    def setUp(self):
        import test_262k_evidence_waiver as waiver

        waiver.pending_records(self)

    def test_tp_guard_at_262k_needs_both_records_in_a_gate_run_too(self):
        lines = Lines()
        with mock.patch.object(page_width_tp4, 'evidence_state', return_value=E1_OK):
            with self.assertRaises(admission.AdmissionRefused) as caught:
                admission.tp_guard(WIDE_GATE, log=lines)
        self.assertTrue(any('capacity 262144' in problem for problem in caught.exception.problems))
        with mock.patch.object(admission, 'check_window_262k', return_value=qualifying262()) as window:
            self.assertEqual(admission.tp_guard(WIDE, log=lines), [])
            window.assert_called_once()

    def test_tp_guard_at_131k_does_not_look_at_the_wide_window(self):
        with mock.patch.object(admission, 'check_window_262k', side_effect=AssertionError('read')):
            self.assertEqual(admission.tp_guard(quad_tests.GOOD_ENV, log=Lines()), [])
            self.assertEqual(admission.tp_guard(dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='131328'), log=Lines()), [])


if __name__ == '__main__':
    unittest.main()
