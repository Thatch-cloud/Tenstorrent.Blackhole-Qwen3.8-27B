"""record_packed_any_evidence_tp4: the four-card evidence recorder, on sample reports shaped like the harnesses' own.

The sample reports carry exactly the fields k64j_card_b.py (CB1, CB2a, at --kv-heads 1) and extent_reader_card_b.py (CB2b, at width 4)
write; the recorder must turn three passing ones into a record packed_any_admission qualifies at four cards, re-pin the file, and
refuse every report that is not evidence (a pair's report, a reduced scope, a wrong binary or reader, a differing comparison), and
everything that must not reach a public repository."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import record_packed_any_evidence_tp4 as recorder  # noqa: E402

COMMIT = 'a' * 40


def comparisons(kinds_counts):
    rows = []
    for kind, count in kinds_counts.items():
        rows += [dict(kind=kind, decisive=True, differing=0, label='%s/%d' % (kind, index)) for index in range(count)]
    return rows


def tally_of(rows):
    out = {}
    for entry in rows:
        row = out.setdefault(entry['kind'], dict(runs=0, equal=0, decisive=entry['decisive']))
        row['runs'] += 1
        row['equal'] += entry['differing'] == 0
    return out


def cb1_report():
    rows = comparisons(dict(extent_vs_reference=2100, mixed_vs_reference=75, share_slot0=20, share_skip_slot0_wins=5,
                            trace_vs_eager=100, trace_vs_reference=44, skip_live=10, trace_skip_live=14, refusal=4))
    report = dict(card='K64J_CARD', passed=True, kv_heads=1, seeds=[0, 1, 2, 3, 4], extents=list(admission.K1_EXTENTS),
                  sections=list(recorder.CB1_SECTIONS), combos=['G4B3:0x21', 'G4B3:0x23', 'G8B2:0x21', 'G8B2:0x23'],
                  failures=[], warnings=[], comparisons=rows, tally=tally_of(rows),
                  liveness=[dict(section='X', label='a', live=True), dict(section='T', label='b', live=True)],
                  trace_families_distinct=114, skip_written='partial-unpoisoned',
                  decision=dict(verdict='PASS', reasons=[], decisive=len(rows), decisive_differing=0, k2='not_run'),
                  binary=dict(sha256=admission.K64J_TTNNCPP_SHA256, k64j=True, stage=4),
                  kernels=dict(root='/k', found=dict(admission.K64J_KERNELS)))
    report['verdict_line'] = ('K64J_CARD verdict=PASS extent=2100/2100 mixed=75/75 share_slot0=25/25 trace=144/144 fence=none '
                              'skip=24/24 refusals=4/4 live=2/2 skipped_rows=partial-unpoisoned families=114 k2=none '
                              'k2_verdict=not_run x7=none z=none k4=not_run kv_heads=1')
    return report


def cb2a_report():
    rows = comparisons(dict(k2_native_vs_extent=1980, x7_narrow_vs_wide=250, x7_extent_vs_wide=250, z_trace_vs_eager=450,
                            z_trace_vs_reference=450))
    report = dict(card='K64J_CARD', passed=True, kv_heads=1, local_heads=6, seeds=[0, 1, 2, 3, 4], variants=['normal', 'peaky'],
                  sections=list(recorder.CB2A_SECTIONS), failures=[], warnings=[], comparisons=rows, tally=tally_of(rows),
                  liveness=[dict(section='K2', label='l%d' % index, live=True) for index in range(1035)],
                  z_families=list(admission.Z_FAMILIES), k2_rows=dict(compared=29730, equal=29730, floor=3280, floor_differing=0),
                  decision=dict(verdict='PASS', reasons=[], decisive=len(rows), decisive_differing=0, k2='PASS',
                                k2_coverage=dict(full=True, covered=1980, design=1980, short=[])),
                  binary=dict(sha256=admission.K64J_TTNNCPP_SHA256, k64j=True, stage=4),
                  kernels=dict(root='/k', found=dict(admission.K64J_KERNELS)))
    report['verdict_line'] = ('K64J_CARD verdict=PASS extent=none k2=1980/1980 k2_rows=29730/29730 k2_floor_differing=0/3280 '
                              'k2_verdict=PASS k2_coverage=1980/1980 x7=500/500 z=900/900 z_families=15 kv_heads=1')
    return report


def cb2b_report():
    rows = comparisons(dict(r1_eager=21, r1_trace=21, r1_reader=672, staging_word=1000, restage_word=932,
                            r2_construction_vs_wide=342, r2_trace_vs_wide=342, r2_trace_vs_eager=168, r4_live_unchanged=264))
    families = [256, 512, 2304, 16640, 65792, 131328] + [3328 + 256 * index for index in range(45)]
    families = sorted(set(families))
    shas = {name: hashlib.sha256(name.encode()).hexdigest() for name in recorder.PINNED_SIBLINGS}
    reader = recorder.reader_sha(str(HERE))
    shas[recorder.READER_TP] = reader
    report = dict(reader='K64J_READER', passed=True, capacity=131328, sections=['R1', 'S', 'R2', 'R4'], seeds_run=[0, 1, 2],
                  variants_run=['normal', 'peaky'], r1_run={name: list(admission.CB2B_RESIDUES) + [32, 128] for name in
                                                            admission.CB2B_R1_GEOMETRIES},
                  r2_families_replayed=families, idle_starts_run=[0, 32], failures=[], warnings=[], comparisons=rows,
                  tally=tally_of(rows), liveness=[dict(label='x', live=True)] * 12, segments=[[0, 1]] * 4,
                  served=dict(flags='0x23', reference_flags='0x3', rows=8, batch=2),
                  decision=dict(verdict='PASS', reasons=[], decisive=len(rows), decisive_differing=0, scope='full', scope_short=[]),
                  binary=dict(sha256=admission.K64J_TTNNCPP_SHA256, k64j=True, stage=4),
                  kernels=dict(root='/k', found=dict(admission.K64J_KERNELS)), modules=dict(sha256=shas))
    report['verdict_line'] = ('K64J_READER verdict=PASS scope=full r1=42/42 r1_reader=672/672 staging=1932/1932 r2=684/684 '
                              'r2_trace=168/168 r4=264/264 live=12/12 families=%d chips=1of4 phantom=2895 extent_sha256=%s'
                              % (len(families), reader))
    return report


class Workspace:
    """A temp directory with the skeleton record, a copy of the admission module and the sample reports."""

    def __init__(self, test):
        self.dir = Path(tempfile.mkdtemp(prefix='recorder-test-'))
        test.addCleanup(shutil.rmtree, str(self.dir), True)
        self.evidence = self.dir / 'packed_any_evidence_tp4.json'
        self.admission = self.dir / 'packed_any_admission.py'
        # The skeleton (the committed record is filled in now), with the admission copy re-pinned to it.
        from test_packed_any_admission_tp4 import SKELETON_TP4_TEXT
        self.evidence.write_bytes(SKELETON_TP4_TEXT.encode('utf-8'))
        source = (HERE / 'packed_any_admission.py').read_text(encoding='utf-8')
        pin = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        assert source.count(admission.EVIDENCE_TP4_SHA256) == 1, 'the pin is in the admission source exactly once'
        self.admission.write_text(source.replace(admission.EVIDENCE_TP4_SHA256, pin), encoding='utf-8', newline=chr(10))

    def report(self, name, report):
        path = self.dir / name
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return str(path)

    def argv(self, cb1=None, cb2a=None, cb2b=None, extra=()):
        args = ['--evidence', str(self.evidence), '--admission', str(self.admission), '--sources-root', str(HERE)]
        if cb1 is not None:
            args += ['--cb1', self.report('card-ev-f1.json', cb1), '--cb1-run', '1001', '--cb1-tag', 'experiment/c2-serving-v901']
        if cb2a is not None:
            args += ['--cb2a', self.report('card-ev-f2.json', cb2a), '--cb2a-run', '1002', '--cb2a-tag',
                     'experiment/c2-serving-v902']
        if cb2b is not None:
            args += ['--cb2b', self.report('reader-ev-f3.json', cb2b), '--cb2b-run', '1003', '--cb2b-tag',
                     'experiment/c2-serving-v903', '--cb2b-commit', COMMIT, '--cb2b-image', 'tp4-stackfix-3']
        return args + list(extra)

    def run(self, **kwargs):
        lines = []
        code = recorder.main(self.argv(**kwargs), out=lines.append)
        return code, lines

    def pin(self):
        import re
        return re.search(r"^EVIDENCE_TP4_SHA256 = '([0-9a-f]{64})'", self.admission.read_text(encoding='utf-8'), re.M).group(1)


class RecordsTests(unittest.TestCase):
    def test_three_passing_reports_become_a_record_the_admission_qualifies_at_four_cards_and_the_pin_follows(self):
        work = Workspace(self)
        code, lines = work.run(cb1=cb1_report(), cb2a=cb2a_report(), cb2b=cb2b_report(),
                               extra=['--extent-audit-run', '36758864992', '--extent-audit-tag', 'experiment/c2-serving-v185'])
        self.assertEqual(code, 0, lines)
        payload = work.evidence.read_bytes()
        self.assertNotIn(b'\r', payload)
        self.assertTrue(payload.endswith(b'\n'))
        self.assertEqual(work.pin(), hashlib.sha256(payload).hexdigest())
        record = json.loads(payload.decode('utf-8'))
        self.assertEqual(admission.evidence_problems(record, str(HERE), tp=4), [])
        # check_evidence is the admission's own gate: the file at its own pin, the sections qualifying
        admission.check_evidence(str(work.evidence), expected_sha256=work.pin(), sources_root=str(HERE), tp=4)
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
        self.assertEqual(record['sources'], {recorder.READER_TP: recorder.reader_sha(str(HERE))})
        self.assertNotIn('SKELETON', record['what'])
        self.assertIn('36758864992', record['provenance'])

    def test_the_repos_admission_file_is_not_touched_by_a_test_run_and_the_pair_record_is_not_the_four_card_one(self):
        before = (admission.EVIDENCE_TP4.read_bytes(), (HERE / 'packed_any_admission.py').read_bytes())
        work = Workspace(self)
        work.run(cb1=cb1_report())
        self.assertEqual(before, (admission.EVIDENCE_TP4.read_bytes(), (HERE / 'packed_any_admission.py').read_bytes()))

    def test_cb1_takes_every_value_from_the_report(self):
        work = Workspace(self)
        self.assertEqual(work.run(cb1=cb1_report())[0], 0)
        cb1 = json.loads(work.evidence.read_text(encoding='utf-8'))['sections']['CB1']
        self.assertEqual(cb1['status'], 'PASS')
        self.assertEqual((cb1['run'], cb1['tag'], cb1['card'], cb1['kv_heads']), (1001, 'experiment/c2-serving-v901', 'M', 1))
        self.assertEqual(cb1['seeds'], [0, 1, 2, 3, 4])
        self.assertEqual(cb1['combos'], [dict(geometry='G4B3', flags=['0x21', '0x23']), dict(geometry='G8B2', flags=['0x21', '0x23'])])
        self.assertEqual(cb1['counts'], dict(extent=[2100, 2100], mixed=[75, 75], share_slot0=[25, 25], trace=[144, 144],
                                             skip=[24, 24]))
        self.assertEqual((cb1['families'], cb1['failures'], cb1['skipped_rows']), (114, 0, 'partial-unpoisoned'))
        self.assertEqual(cb1['report'], 'cardm/card-ev-f1.json')
        self.assertEqual(len(cb1['report_sha256']), 64)

    def test_cb2a_takes_every_value_from_the_report(self):
        work = Workspace(self)
        self.assertEqual(work.run(cb2a=cb2a_report())[0], 0)
        cb2a = json.loads(work.evidence.read_text(encoding='utf-8'))['sections']['CB2a']
        self.assertEqual(cb2a['k2'], dict(verdict='PASS', tickets=[1980, 1980], rows=[29730, 29730], floor_rows_differing=[0, 3280]))
        self.assertEqual((cb2a['x7'], cb2a['z']['passed'], cb2a['liveness']), ([500, 500], [900, 900], [1035, 1035]))
        self.assertEqual(cb2a['z']['families'], list(admission.Z_FAMILIES))
        self.assertEqual((cb2a['variants'], cb2a['local_heads'], cb2a['kv_heads']), (['normal', 'peaky'], 6, 1))

    def test_cb2b_takes_every_value_from_the_report_and_the_verdict_line(self):
        work = Workspace(self)
        self.assertEqual(work.run(cb2b=cb2b_report())[0], 0)
        record = json.loads(work.evidence.read_text(encoding='utf-8'))
        cb2b = record['sections']['CB2b']
        self.assertEqual((cb2b['scope'], cb2b['chips'], cb2b['capacity'], cb2b['image'], cb2b['commit']),
                         ('full', '1of4', 131328, 'tp4-stackfix-3', COMMIT))
        self.assertEqual(cb2b['counts'], dict(R1=[714, 714], S=[1932, 1932], R2=[852, 852], R4=[264, 264], liveness=[12, 12]))
        self.assertEqual(cb2b['served'], dict(flags='0x23', rows=8, batch=2, segments=4))
        self.assertEqual(cb2b['phantom_programs'], 2895)
        self.assertEqual(sorted(cb2b['pinned_modules']), sorted(recorder.PINNED_SIBLINGS))
        self.assertEqual(cb2b['sources'], record['sources'])
        self.assertTrue(cb2b['verdict_line'].startswith('K64J_READER verdict=PASS scope=full'))

    def test_one_section_recorded_leaves_the_others_pending_and_the_skeleton_text_until_all_three_are_in(self):
        work = Workspace(self)
        code, lines = work.run(cb1=cb1_report())
        self.assertEqual(code, 0)
        record = json.loads(work.evidence.read_text(encoding='utf-8'))
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS', 'PENDING', 'PENDING'])
        self.assertIn('SKELETON', record['what'])
        self.assertTrue(any('CB2a' in line for line in lines), 'the problems still left for the admission are printed')
        self.assertEqual(work.pin(), hashlib.sha256(work.evidence.read_bytes()).hexdigest())

    def test_a_watcher_pass_is_cited_with_its_reports_hash_and_scope(self):
        work = Workspace(self)
        watcher = cb1_report()
        watcher['decision']['scope'] = 'reduced'
        path = work.report('watcher.json', watcher)
        argv = work.argv(cb1=cb1_report(), extra=['--cb1-watcher', path, '--cb1-watcher-run', '900', '--cb1-watcher-tag',
                                                  'experiment/c2-serving-v900'])
        self.assertEqual(recorder.main(argv, out=lambda line: None), 0)
        entry = json.loads(work.evidence.read_text(encoding='utf-8'))['sections']['CB1']['watcher_pass']
        self.assertEqual((entry['run'], entry['tag'], entry['status'], entry['scope']), (900, 'experiment/c2-serving-v900', 'PASS',
                                                                                       'reduced'))
        self.assertEqual(len(entry['report_sha256']), 64)

    def test_dry_run_writes_nothing(self):
        work = Workspace(self)
        before = (work.evidence.read_bytes(), work.admission.read_bytes())
        code, _ = work.run(cb1=cb1_report(), extra=['--dry-run'])
        self.assertEqual(code, 0)
        self.assertEqual(before, (work.evidence.read_bytes(), work.admission.read_bytes()))

    def test_recording_twice_is_stable(self):
        work = Workspace(self)
        work.run(cb1=cb1_report(), cb2a=cb2a_report(), cb2b=cb2b_report())
        first = work.evidence.read_bytes(), work.admission.read_bytes()
        work.run(cb1=cb1_report(), cb2a=cb2a_report(), cb2b=cb2b_report())
        self.assertEqual(first, (work.evidence.read_bytes(), work.admission.read_bytes()))


class RefusesTests(unittest.TestCase):
    def refused(self, which, mutate, needle):
        work = Workspace(self)
        reports = dict(cb1=cb1_report(), cb2a=cb2a_report(), cb2b=cb2b_report())
        mutate(reports[which])
        before = (work.evidence.read_bytes(), work.admission.read_bytes())
        code, lines = work.run(**{which: reports[which]})
        self.assertEqual(code, 1, lines)
        self.assertTrue(any(needle in line for line in lines), (needle, lines))
        self.assertEqual(before, (work.evidence.read_bytes(), work.admission.read_bytes()), 'nothing is written on a refusal')

    def test_a_differing_decisive_comparison_is_not_evidence(self):
        def mutate(report):
            report['tally']['extent_vs_reference']['equal'] -= 1
            report['decision']['verdict'] = 'FAIL'
            report['verdict_line'] = report['verdict_line'].replace('verdict=PASS', 'verdict=FAIL')
        self.refused('cb1', mutate, 'not PASS')

    def test_a_failed_count_with_a_pass_verdict_is_still_refused(self):
        def mutate(report):
            report['tally']['mixed_vs_reference']['equal'] -= 1
        self.refused('cb1', mutate, 'mixed [74, 75] is not a full pass')

    def test_the_pairs_report_is_not_four_card_evidence(self):
        def mutate(report):
            report['kv_heads'] = 2
            report['verdict_line'] = report['verdict_line'].replace('kv_heads=1', 'kv_heads=2')
        self.refused('cb1', mutate, 'kv_heads is 2')

    def test_a_report_without_the_kv_heads_word_is_refused(self):
        def mutate(report):
            del report['kv_heads']
            report['verdict_line'] = report['verdict_line'].replace(' kv_heads=1', '')
        self.refused('cb2a', mutate, 'kv_heads is None')

    def test_another_binary_is_refused(self):
        def mutate(report):
            report['binary']['sha256'] = 'b' * 64
        self.refused('cb1', mutate, 'not K64j')

    def test_another_kernel_is_refused(self):
        def mutate(report):
            report['kernels']['found']['compute/sdpa_flash_decode_qwen.cpp'] = 'c' * 64
        self.refused('cb2a', mutate, 'compute/sdpa_flash_decode_qwen.cpp')
        self.refused('cb2b', mutate, 'compute/sdpa_flash_decode_qwen.cpp')
        self.refused('cb2b', lambda report: report.pop('kernels'), 'dataflow/reader_decode_qwen.cpp')

    def test_a_cb1_without_the_g8b2_0x23_combo_or_a_seed_or_an_extent_is_refused(self):
        self.refused('cb1', lambda report: report.update(combos=['G8B2:0x21', 'G4B3:0x23']), 'no G8B2 0x23 combo')
        self.refused('cb1', lambda report: report.update(seeds=[0, 1, 2]), 'lack [3, 4]')
        self.refused('cb1', lambda report: report.update(extents=report['extents'][:-1]), 'extents')

    def test_a_q_slice_combo_in_a_one_head_record_is_refused(self):
        self.refused('cb1', lambda report: report['combos'].append('G8B2:0x27'), 'needs a second KV head')

    def test_a_cb1_that_skipped_a_section_is_refused(self):
        self.refused('cb1', lambda report: report.update(sections=['X', 'M', 'K', 'L', 'T']), 'N did not run')

    def test_a_reduced_k2_is_refused(self):
        def mutate(report):
            report['decision']['k2'] = 'REDUCED-PASS'
            report['decision']['k2_coverage'] = dict(full=False, covered=38, design=1980, short=['seeds'])
        self.refused('cb2a', mutate, 'K2 verdict REDUCED-PASS')

    def test_short_x7_or_z_is_refused(self):
        self.refused('cb2a', lambda report: report['tally']['x7_narrow_vs_wide'].update(runs=10, equal=10), 'X7')
        self.refused('cb2a', lambda report: report.update(z_families=report['z_families'][:-1]), 'Z families lack')

    def test_a_dead_liveness_control_or_a_failure_or_an_error_is_refused(self):
        self.refused('cb2a', lambda report: report['liveness'].__setitem__(0, dict(section='K2', label='x', live=False)),
                     'liveness controls did not move')
        self.refused('cb1', lambda report: report['failures'].append('X/seed0: boom'), 'failures')
        self.refused('cb1', lambda report: report.update(error='watchdog'), 'error')

    def test_a_checkpoint_is_not_a_final_report(self):
        self.refused('cb1', lambda report: report.update(in_progress='after X/seed0'), 'checkpoint')

    def test_a_reduced_reader_scope_is_refused(self):
        def mutate(report):
            report['decision'].update(scope='reduced', scope_short=['seeds'])
            report['verdict_line'] = report['verdict_line'].replace('scope=full', 'scope=reduced')
        self.refused('cb2b', mutate, 'scope reduced')

    def test_the_pairs_chip_view_is_refused(self):
        self.refused('cb2b', lambda report: report.update(verdict_line=report['verdict_line'].replace('1of4', '1of2')), 'chips 1of2')

    def test_a_reader_served_with_the_slice_is_refused(self):
        self.refused('cb2b', lambda report: report['served'].update(flags='0x27'), 'served flags 0x27')

    def test_another_reader_sha_is_refused(self):
        def mutate(report):
            report['verdict_line'] = report['verdict_line'].replace(report['modules']['sha256'][recorder.READER_TP], 'd' * 64)
        self.refused('cb2b', mutate, 'run CB2b on these bytes')

    def test_a_capacity_other_than_the_served_one_or_a_missing_seed_is_refused(self):
        self.refused('cb2b', lambda report: report.update(capacity=65792), 'capacity 65792')
        self.refused('cb2b', lambda report: report.update(seeds_run=[0]), 'seeds run')

    def test_a_missing_pinned_sibling_hash_is_refused(self):
        self.refused('cb2b', lambda report: report['modules']['sha256'].pop('attention_mask_replay_tp.py'), 'attention_mask_replay_tp.py')

    def test_a_wrong_commit_or_an_image_with_a_registry_or_digest_is_refused(self):
        work = Workspace(self)
        for extra, needle in ((['--cb2b-commit', 'abc'], '40-hex'),
                              (['--cb2b-image', 'registry.example:5000/tt-vllm:x'], 'TAG NAME only'),
                              (['--cb2b-image', 'sha256:' + 'e' * 64], 'TAG NAME only')):
            argv = work.argv(cb2b=cb2b_report()) + extra
            lines = []
            self.assertEqual(recorder.main(argv, out=lines.append), 1)
            self.assertTrue(any(needle in line for line in lines), (needle, lines))

    def test_a_host_path_board_id_or_digest_never_reaches_the_record(self):
        work = Workspace(self)
        for tag in ('/home/runner/tag', 'blackhole-0123456789ABCDEF', 'zot.thatch.local:5000/x'):
            argv = work.argv(cb1=cb1_report())
            argv[argv.index('--cb1-tag') + 1] = tag
            lines = []
            self.assertEqual(recorder.main(argv, out=lines.append), 1, tag)
            self.assertTrue(any('host path, registry' in line for line in lines), (tag, lines))

    def test_the_hygiene_check_flags_what_the_pairs_record_carries(self):
        pair = json.loads((HERE / 'packed_any_evidence.json').read_text(encoding='utf-8'))
        self.assertTrue(recorder.hygiene_problems(pair), 'the pair\'s record names /home/thatch/opgraft-K64j: not a pattern to copy')
        self.assertEqual(recorder.hygiene_problems(dict(image='tp4-stackfix-3', report='cardm/card-1.json')), [])

    def test_with_nothing_to_record_it_says_so(self):
        lines = []
        self.assertEqual(recorder.main(['--dry-run'], out=lines.append), 2)


class FilesTests(unittest.TestCase):
    def test_the_dump_is_lf_one_space_and_in_the_skeletons_key_order(self):
        skeleton = json.loads(admission.EVIDENCE_TP4.read_text(encoding='utf-8'))
        payload = recorder.dump(copy.deepcopy(skeleton))
        self.assertNotIn(b'\r', payload)
        self.assertEqual(list(json.loads(payload.decode('utf-8'))), list(recorder.TOP_KEYS))
        self.assertTrue(payload.startswith(b'{\n "schema"'))

    def test_repin_changes_only_the_pin_line_and_keeps_crlf_untouched(self):
        text = (HERE / 'packed_any_admission.py').read_bytes()
        new = recorder.repin(str(HERE / 'packed_any_admission.py'), 'f' * 64)
        self.assertEqual(len(new.splitlines()), len(text.splitlines()))
        changed = [(a, b) for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][1], b"EVIDENCE_TP4_SHA256 = '" + b'f' * 64 + b"'")

    def test_repin_refuses_a_file_without_exactly_one_pin(self):
        work = Workspace(self)
        work.admission.write_text('x = 1\n', encoding='utf-8')
        with self.assertRaises(recorder.RecordError):
            recorder.repin(str(work.admission), 'f' * 64)

    def test_the_recorder_is_stdlib_python37_syntax_and_takes_the_constants_from_the_admission(self):
        self.assertEqual(recorder.SERVED_FLAGS, admission.CB1_FLAG_TP4)
        self.assertEqual(recorder.SCHEMA, admission.EVIDENCE_SCHEMA)
        source = (HERE / 'record_packed_any_evidence_tp4.py').read_text(encoding='utf-8')
        self.assertNotIn(':=', source)
        self.assertNotIn('removeprefix', source)


if __name__ == '__main__':
    unittest.main()
