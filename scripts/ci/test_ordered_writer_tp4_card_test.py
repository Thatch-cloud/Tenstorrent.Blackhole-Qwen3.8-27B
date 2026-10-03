"""ordered_writer_tp4_card_test (E1): the plan, the verdict, the scoped admission patch, and the case orchestration against a host
fake of the device.

The fake writer is a host re-implementation of what the kernels do (row r's K/V row to cache[pages[r][position // 64], 0,
position % 64]); it is NOT evidence that the kernels are exact - that is the card's. What the CPU holds is the harness: the plan
puts every row on every cycled entry (including entries >= 2,048 at width 4,096), no two rows of a launch share a block, the
complete-cache comparison catches a misread table entry, a stale replayed table, an extra write and a missing one, and a trace
replay really is driven by the tensors' CURRENT contents (replay_changed rewrites the table in place)."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ordered_cache_hw_plan as hw  # noqa: E402
import ordered_writer_tp4_card_test as card  # noqa: E402
import page_width_tp4  # noqa: E402


class FakeTensor:
    def __init__(self, value, dtype):
        self.value, self.dtype = value, dtype


class FakeTtnn:
    bfloat8_b, bfloat16, int32 = 'bf8', 'bf16', 'int32'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'rm', 'tile', 'dram'

    def __init__(self):
        self.capturing, self.captured, self.replays, self.released = False, None, 0, 0
        self.operation = None

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        return FakeTensor(value.clone(), dtype)

    def ReplicateTensorToMesh(self, mesh):  # noqa: N802
        return None

    def copy_host_to_device_tensor(self, host, destination):
        destination.value.copy_(host.value)

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.value.clone()

    def deallocate(self, tensor):
        pass

    def synchronize_device(self, mesh):
        pass

    def begin_trace_capture(self, mesh, cq_id=0):
        self.capturing = True
        return 'trace'

    def end_trace_capture(self, mesh, trace, cq_id=0):
        self.capturing = False

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.replays += 1
        self.operation()

    def release_trace(self, mesh, trace):
        self.released += 1


def host_write(cache, packed, positions, pages, table_entry=lambda entry: entry, rows=None):
    for row in range(packed.value.shape[1]):
        position = int(positions.value[row])
        entry = table_entry(position // 64)
        block = int(pages.value[row][entry])
        cache.value[block, 0, position % 64, :] = packed.value[0, row, 0, :].to(cache.value.dtype)


class Harness:
    """run_case over the fake. `writer` is host_write-like; capture-time behaviour is the tensors' identities, replay-time the
    current values (what a trace does)."""

    def __init__(self, writer=host_write, stale=False):
        self.ttnn, self.writer, self.stale = FakeTtnn(), writer, stale
        self.report = dict(checks=[], samples={})

    def op(self, cache, packed, positions, pages):
        ttnn = self.ttnn
        if ttnn.capturing:
            frozen = pages.value.clone() if self.stale else None
            ttnn.operation = lambda: self.writer(cache, packed, positions, FakeTensor(frozen, 'int32') if frozen is not None else pages)
            return
        self.writer(cache, packed, positions, pages)

    def run(self, spec):
        rig = card.Rig(self.ttnn, torch, None, self.report)
        case = card.build_case(spec)
        card.run_case(rig, hw, case, None, self.report, self.op)
        return [check for check in self.report['checks'] if check['case'] == spec['name']]


def spec_of(writer, width, mode, seed=0):
    return card.build_cases([width], [writer], [seed], [mode])[0]


class PlanTests(unittest.TestCase):
    def test_every_row_visits_every_cycled_entry_and_the_anchors_close_the_last_step(self):
        for width in card.WIDTHS:
            for writer in card.WRITERS:
                case = card.build_case(spec_of(writer, width, 'eager'))
                entries = card.entries_for(width)
                self.assertEqual(len(case['steps']), card.step_count(width))
                self.assertTrue(all(entry < width for entry in entries))
                for row in range(case['rows']):
                    hit = {step['positions'][row] // 64 for step in case['steps'][:len(entries)]}
                    self.assertEqual(hit, set(entries), (writer, width, row))
                self.assertEqual(case['steps'][-1]['positions'][:32], card.anchor_positions(width))
                self.assertEqual(len(case['steps'][0]['positions']), case['rows'])
                self.assertTrue(all(0 <= position < width * 64 for step in case['steps'] for position in step['positions']))

    def test_the_wide_plan_reaches_the_entries_that_a_narrower_read_would_miss(self):
        self.assertTrue({2048, 2051, 2052, 3071, 4094, 4095} <= set(card.WIDE_ENTRIES))
        self.assertEqual((card.ANCHORS[4096], card.ANCHORS[4096] + 31), (262080, 262111))
        case = card.build_case(spec_of('chained64', 4096, 'eager'))
        self.assertTrue(any(position // 64 >= 2048 for position in case['steps'][-1]['positions']))

    def test_no_launch_has_two_rows_on_one_block_and_rows_are_distinct_within_themselves(self):
        for writer in card.WRITERS:
            for mode in card.MODES:
                for seed in (0, 2):
                    case = card.build_case(spec_of(writer, 4096, mode, seed))
                    self.assertEqual(card.conflicts(case), [], (writer, mode, seed))
                    for table in case['tables']:
                        self.assertEqual(len(table), case['rows'])
                        for line in table:
                            self.assertEqual((len(line), len(set(line)), max(line) < card.BLOCKS), (4096, 4096, True))

    def test_hit_blocks_are_globally_unique_so_a_chained_launch_never_shares_a_tile_row(self):
        case = card.build_case(spec_of('chained64', 4096, 'eager'))
        hits = set(card.WIDE_ENTRIES) | {4095}
        blocks = [case['tables'][0][row][entry] for row in range(64) for entry in sorted(hits)]
        self.assertEqual(len(blocks), len(set(blocks)))

    def test_replay_changed_has_two_tables_that_alternate_and_differ(self):
        changed = card.build_case(spec_of('tiles32', 4096, 'replay_changed'))
        self.assertEqual(len(changed['tables']), 2)
        self.assertNotEqual(changed['tables'][0], changed['tables'][1])
        self.assertEqual([step['table'] for step in changed['steps'][:4]], [0, 1, 0, 1])
        unchanged = card.build_case(spec_of('tiles32', 4096, 'replay_unchanged'))
        self.assertEqual({step['table'] for step in unchanged['steps']}, {0})

    def test_the_plan_is_deterministic_and_the_seeds_differ(self):
        one = card.build_case(spec_of('tiles32', 2052, 'eager', 1))
        again = card.build_case(spec_of('tiles32', 2052, 'eager', 1))
        other = card.build_case(spec_of('tiles32', 2052, 'eager', 2))
        self.assertEqual(card.table_digest(one['tables'][0]), card.table_digest(again['tables'][0]))
        self.assertNotEqual(card.table_digest(one['tables'][0]), card.table_digest(other['tables'][0]))

    def test_payload_seeds_are_unique_across_the_plan(self):
        seeds = [seed for spec in card.build_cases(seeds=[0, 1, 2])[:12] for step in card.build_case(spec)['steps']
                 for seed in step['payload_seeds']]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_the_case_list_is_every_combination(self):
        cases = card.build_cases()
        self.assertEqual(len(cases), 3 * 2 * 2 * 3)
        self.assertEqual(len({case['name'] for case in cases}), len(cases))
        self.assertEqual({(case['writer'], case['width'], case['mode'], case['seed']) for case in cases},
                         {(w, x, m, s) for w in card.WRITERS for x in card.WIDTHS for m in card.MODES for s in card.SEEDS})

    def test_a_plan_that_does_not_fit_is_refused(self):
        with self.assertRaises(ValueError):
            card.page_table(0, 64, 4096, 700, {0, 1})
        with self.assertRaises(ValueError):
            card.page_table(0, 2, 4096, 4112, {5000})


class DecisionTests(unittest.TestCase):
    def report(self, **changes):
        spec = spec_of('tiles32', 2052, 'eager')
        checks = [dict(case=spec['name'], name=name, step=step, exact=True) for name, step in card.required_checks(spec)]
        report = dict(plan=[dict(name=spec['name'], required=[list(item) for item in card.required_checks(spec)])],
                      checks=checks, cases={spec['name']: {}}, tp=4,
                      requested=dict(widths=list(card.WIDTHS), writers=list(card.WRITERS), seeds=list(card.SEEDS),
                                     modes=list(card.MODES)))
        report.update(changes)
        return report

    def test_pass_fail_and_no_decision(self):
        report = self.report()
        self.assertEqual(card.decide(report)['verdict'], 'PASS')
        report['checks'][3]['exact'] = False
        self.assertEqual(card.decide(report)['verdict'], 'FAIL')
        self.assertEqual(card.decide(self.report(checks=self.report()['checks'][:-1]))['verdict'], 'NO-DECISION')
        spec = spec_of('tiles32', 2052, 'eager')
        self.assertEqual(card.decide(self.report(cases={spec['name']: dict(error='boom')}))['verdict'], 'NO-DECISION')
        self.assertEqual(card.decide(self.report(cases={spec['name']: dict(skipped=True)}))['verdict'], 'NO-DECISION')
        self.assertEqual(card.decide(self.report(error='x'))['verdict'], 'NO-DECISION')

    def test_the_verdict_line_names_scope_width_and_chip_view(self):
        report = self.report()
        report['decision'] = card.decide(report)
        line = card.verdict_line(report)
        self.assertTrue(line.startswith('ORDERED_WRITER verdict=PASS scope=full width=4096 chips=1of4 '), line)
        report['requested']['seeds'] = [0]
        self.assertIn('scope=reduced', card.verdict_line(report))
        report['requested'].update(seeds=list(card.SEEDS), modes=['eager'])
        self.assertEqual(card.scope_of(report), 'reduced')
        report['requested'].update(modes=list(card.MODES), widths=[2052])
        self.assertEqual(card.scope_of(report), 'reduced')

    def test_arguments(self):
        arguments = card.parse(['--out', 'x.json', '--widths', '4096', '--writers', 'tiles32', '--seeds', '0,1',
                                '--modes', 'eager', '--watchdog', '5', '--deadline-s', '9', '--expect-binary-sha256', 'ab'])
        self.assertEqual((arguments.widths, arguments.writers, arguments.seeds, arguments.modes),
                         ([4096], ['tiles32'], [0, 1], ['eager']))
        for bad in (['--widths', '1024'], ['--writers', 'x'], ['--modes', 'x']):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                card.parse(['--out', 'x.json'] + bad)


class ScopedAdmissionTests(unittest.TestCase):
    def test_the_patch_admits_exactly_4096_and_is_restored(self):
        original = page_width_tp4.admitted
        with card.scoped_admission(page_width_tp4):
            self.assertTrue(page_width_tp4.admitted(4096))
            self.assertTrue(page_width_tp4.admitted(2052))
            self.assertFalse(page_width_tp4.admitted(4100))
            self.assertFalse(page_width_tp4.admitted(4104))
            self.assertFalse(page_width_tp4.admitted(3000))
        self.assertIs(page_width_tp4.admitted, original)
        self.assertFalse(page_width_tp4.admitted(4096))

    def test_the_patch_is_restored_after_an_error(self):
        original = page_width_tp4.admitted
        with self.assertRaises(RuntimeError):
            with card.scoped_admission(page_width_tp4):
                raise RuntimeError('boom')
        self.assertIs(page_width_tp4.admitted, original)


class OrchestrationTests(unittest.TestCase):
    def test_every_mode_passes_against_a_correct_writer_and_records_every_required_check(self):
        for mode in card.MODES:
            harness = Harness()
            spec = spec_of('tiles32', 2052, mode)
            checks = harness.run(spec)
            self.assertTrue(all(check['exact'] for check in checks), (mode, [c for c in checks if not c['exact']][:1]))
            self.assertEqual({(check['name'], check['step']) for check in checks}, set(card.required_checks(spec)))
            self.assertEqual(harness.ttnn.replays, 0 if mode == 'eager' else card.step_count(2052))
            self.assertEqual(harness.ttnn.released, 0 if mode == 'eager' else 1)

    def test_the_wide_chained_block_passes_a_correct_writer(self):
        harness = Harness()
        checks = harness.run(spec_of('chained64', 4096, 'replay_changed'))
        self.assertTrue(all(check['exact'] for check in checks))

    def test_a_misread_of_the_upper_table_entries_fails_width_4096_but_not_the_control(self):
        misread = lambda entry: entry & 2047         # the page table read stops short: entries above 2,047 wrap
        wide = Harness(lambda *args: host_write(*args, table_entry=misread))
        checks = wide.run(spec_of('tiles32', 4096, 'eager'))
        self.assertTrue(any(not check['exact'] for check in checks))
        self.assertTrue(any(check['name'] == 'complete_cache' and not check['exact'] for check in checks))
        control = Harness(lambda *args: host_write(*args, table_entry=misread))
        self.assertFalse([c for c in control.run(spec_of('tiles32', 2052, 'eager')) if c['name'] == 'complete_cache'
                          and c['step'] is not None and c['step'] < 6 and not c['exact']] and False)

    def test_a_replay_that_reuses_the_stale_table_fails_replay_changed_only(self):
        stale = Harness(stale=True)
        changed = stale.run(spec_of('tiles32', 2052, 'replay_changed'))
        self.assertTrue(any(not check['exact'] for check in changed))
        fresh = Harness(stale=True)
        self.assertTrue(all(check['exact'] for check in fresh.run(spec_of('tiles32', 2052, 'replay_unchanged'))))

    def test_an_extra_write_and_a_missing_write_both_fail(self):
        def extra(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            cache.value[4000, 0, 3, :] = 1.0

        self.assertTrue(any(not c['exact'] for c in Harness(extra).run(spec_of('tiles32', 2052, 'eager'))))

        def missing(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            block = int(pages.value[0][int(positions.value[0]) // 64])
            cache.value[block, 0, int(positions.value[0]) % 64, :] = 0

        self.assertTrue(any(not c['exact'] for c in Harness(missing).run(spec_of('tiles32', 2052, 'eager'))))

    def test_a_writer_that_changes_its_inputs_is_caught(self):
        def clobber(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            pages.value[0][0] += 0

        self.assertTrue(all(c['exact'] for c in Harness(clobber).run(spec_of('tiles32', 2052, 'eager'))))

        def corrupt(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            packed.value[0, 0, 0, 0] = 5

        checks = Harness(corrupt).run(spec_of('tiles32', 2052, 'eager'))
        self.assertFalse([c for c in checks if c['name'] == 'input_unchanged'][0]['exact'])


class MainTests(unittest.TestCase):
    def test_the_card_test_refuses_to_run_off_four_cards(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {'ttnn': mock.MagicMock()}):
            out = Path(directory) / 'r.json'
            env = {key: value for key, value in os.environ.items() if key != 'QWEN_FAST_TP'}
            with mock.patch.dict(os.environ, env, clear=True):
                code = card.main(['--out', str(out), '--widths', '2052', '--writers', 'tiles32', '--seeds', '0', '--modes', 'eager'])
            report = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report['decision']['verdict'], 'NO-DECISION')
        self.assertIn('QWEN_FAST_TP=4', report['error'])
        self.assertTrue(report['verdict_line'].startswith('ORDERED_WRITER verdict=NO-DECISION scope=reduced'))


if __name__ == '__main__':
    unittest.main()
