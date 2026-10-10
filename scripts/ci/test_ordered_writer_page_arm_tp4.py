"""The page64 arm of the E1 card test (ordered_writer_page_arm_tp4) and its record (record_ordered_writer_evidence_tp4 --merge-page).

The arm is the card-M proof of plan unknown U1 (bfloat8_b read-modify-write idempotence on the hardware packer): the page writer against the served
writer on identical caches, step after step, the complete caches compared as bit patterns; and every stored row written back to where it came from. What the CPU
holds is the HARNESS: the plan is the stack's shapes (four users of 16 consecutive positions on distinct tile rows, every offset class, both widths, the
window's end), its payloads reach the edge values (-0, denormals, an underflowing outlier, rounding carries, the largest and smallest normals), a correct pair
of writers passes every check and every mode, and a page writer that drops a row, writes the wrong tile row, or moves one bit of a stored block FAILS - the
comparison is not vacuous. Then the record: a qualifying report becomes a 'page_writer' block bound to the live design, anything less is refused, and a
record with the block lets kv_page_writer_tp4 engage while the shipped record (no block) does not.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_page_writer_tp4 as kvpw  # noqa: E402
import ordered_cache_hw_plan as hw  # noqa: E402
import ordered_writer_page_arm_tp4 as arm  # noqa: E402
import ordered_writer_tp4_card_test as card  # noqa: E402
import page_width_tp4 as pw  # noqa: E402
import record_ordered_writer_evidence_tp4 as rec  # noqa: E402
import test_record_ordered_writer_evidence_tp4 as e1  # noqa: E402
from test_ordered_writer_tp4_card_test import FakeTensor, FakeTtnn, host_write  # noqa: E402

COMMIT = 'd' * 40


class PlanTests(unittest.TestCase):
    def test_the_case_matrix(self):
        specs = arm.build_page_cases()
        pages = [spec for spec in specs if spec['kind'] == 'page']
        noops = [spec for spec in specs if spec['kind'] == 'noop']
        self.assertEqual(len(pages), 3 * 2 * len(arm.MATRIX))
        self.assertEqual(len(noops), 2 * (1 + 2 * 2))
        self.assertEqual(len({spec['name'] for spec in specs}), len(specs))
        self.assertEqual([spec['index'] for spec in specs], list(range(len(specs))))
        self.assertEqual({(spec['regime'], spec['source'], spec['mode']) for spec in pages}, set(arm.MATRIX))
        self.assertEqual({spec['width'] for spec in pages}, {2052, 4096})
        self.assertTrue({'eager', 'replay_changed', 'replay_unchanged'} <= {spec['mode'] for spec in pages})
        self.assertEqual({spec['source'] for spec in pages}, {'dram', 'l1'})
        self.assertEqual({spec['writer'] for spec in noops}, {'chained64', 'page64'})

    def test_filters(self):
        self.assertEqual({spec['regime'] for spec in arm.build_page_cases(regimes=('exact',), noop=False)}, {'exact'})
        self.assertEqual({spec['source'] for spec in arm.build_page_cases(sources=('l1',), noop=False)}, {'l1'})
        self.assertEqual({spec['mode'] for spec in arm.build_page_cases(modes=('eager',), noop=False)}, {'eager'})
        self.assertFalse([spec for spec in arm.build_page_cases(noop=False) if spec['kind'] == 'noop'])
        self.assertEqual(arm.build_page_cases(first_index=7)[0]['index'], 7)
        self.assertEqual([spec for spec in arm.build_page_cases(seeds=(1,)) if spec['kind'] == 'noop'], [], 'the noop proofs run with seed 0')

    def test_every_step_puts_four_users_on_distinct_tile_rows_inside_the_window(self):
        for width in card.WIDTHS:
            for seed in card.SEEDS:
                for step in range(arm.PAGE_STEPS + 1):
                    starts = arm.group_starts(width, step, seed, arm.PAGE_STEPS)
                    self.assertEqual(len(starts), 4)
                    self.assertTrue(arm.conflict_free(starts), (width, seed, step, starts))
                    self.assertTrue(all(0 <= start and start + 16 <= width * 64 for start in starts), (width, seed, step, starts))
                    positions = arm.positions_of(starts)
                    self.assertEqual(len(positions), 64)
                    for group in range(4):
                        self.assertIsNone(kvpw.group_problem(positions[group * 16:(group + 1) * 16]))
                        self.assertEqual(positions[group * 16:(group + 1) * 16], list(range(starts[group], starts[group] + 16)))

    def test_the_offset_classes_the_stack_writes_all_occur(self):
        for width in card.WIDTHS:
            seen = set()
            for seed in card.SEEDS:
                for step in range(arm.PAGE_STEPS):
                    for start in arm.group_starts(width, step, seed, arm.PAGE_STEPS):
                        slots = len({(((start + row) >> 6), ((start + row) >> 5) & 1) for row in range(16)})
                        seen.add(('aligned16' if start % 16 == 0 else 'unaligned', 'crossing' if slots == 2 else 'one tile row',
                                  'page' if (start % 64) + 16 > 64 else 'inside'))
            self.assertTrue({('aligned16', 'one tile row', 'inside'), ('unaligned', 'one tile row', 'inside'), ('unaligned', 'crossing', 'inside'),
                             ('unaligned', 'crossing', 'page')} <= seen, (width, seen))

    def test_the_last_step_ends_at_the_window_end_and_reaches_high_entries(self):
        for width in card.WIDTHS:
            starts = arm.group_starts(width, arm.PAGE_STEPS, 0, arm.PAGE_STEPS)
            self.assertEqual(max(arm.positions_of(starts)), width * 64 - 1)
            self.assertTrue(any(position >> 6 >= 2048 for position in arm.positions_of(starts)) or width < 4096)
            self.assertTrue(arm.conflict_free(starts))

    def test_tables_give_every_user_and_hit_entry_a_block_no_other_has(self):
        for mode in card.MODES:
            case = arm.build_page_case(arm.build_page_cases(modes=(mode,), noop=False)[3])
            self.assertEqual(len(case['tables']), 2 if mode == 'replay_changed' else 1)
            for table in case['tables']:
                self.assertEqual(len(table), 4)
                hits = [table[group][entry] for group in range(4) for entry in case['hits']]
                self.assertEqual(len(hits), len(set(hits)))
                self.assertTrue(max(hits) < arm.plan_blocks(case['width']))
                for line in table:
                    self.assertEqual((len(line), len(set(line))), (case['width'], case['width']))
            for step in case['steps']:
                self.assertEqual(len(step['blocks']), 64)
                self.assertEqual(len(set(step['payload_seeds'])), 64)
                for row, position in enumerate(step['positions']):
                    self.assertEqual(step['blocks'][row], case['tables'][step['table']][row // 16][position >> 6])
                self.assertTrue(set(step['blocks']) <= set(case['touched']))
            if mode == 'replay_changed':
                self.assertNotEqual(case['tables'][0], case['tables'][1])
                self.assertEqual([step['table'] for step in case['steps'][:4]], [0, 1, 0, 1])

    def test_payload_seeds_are_unique_across_the_whole_plan(self):
        seeds = []
        for spec in arm.build_page_cases(seeds=(0, 1, 2), noop=False):
            seeds.extend(seed for step in arm.build_page_case(spec)['steps'] for seed in step['payload_seeds'])
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_required_checks(self):
        exact = [spec for spec in arm.build_page_cases() if spec['kind'] == 'page' and spec['regime'] == 'exact'][0]
        random_ = [spec for spec in arm.build_page_cases() if spec['kind'] == 'page' and spec['regime'] == 'random'][0]
        names = lambda spec: {name for name, step in arm.required_checks(spec)}
        self.assertTrue({'host_equal_k', 'host_equal_v', 'served_equal_k', 'served_equal_v', 'twice_equal_k', 'twice_equal_v'} <= names(exact))
        self.assertNotIn('fill_equal', names(exact))
        self.assertIn('fill_equal', names(random_))
        self.assertNotIn('host_equal_k', names(random_))
        self.assertEqual(len([1 for name, step in arm.required_checks(random_) if name == 'served_equal_k']), arm.PAGE_STEPS + 1)
        for spec in arm.build_page_cases():
            if spec['kind'] == 'noop':
                self.assertEqual(len(arm.required_checks(spec)), 2 * arm.step_count(spec))

    def test_noop_cases_use_distinct_blocks_and_cover_every_offset(self):
        for spec in arm.build_page_cases():
            if spec['kind'] != 'noop':
                continue
            case = arm.build_noop_case(spec)
            self.assertEqual(len(case['blocks']), 4 if spec['writer'] == 'page64' else 64)
            self.assertEqual(len(set(case['blocks'])), len(case['blocks']))
            self.assertTrue(max(case['blocks']) < arm.plan_blocks(case['width']))
        covered = {offset + row for offset in arm.NOOP_STARTS for row in range(16)}
        self.assertEqual(covered, set(range(64)))
        self.assertTrue(all(start + 16 <= 64 for start in arm.NOOP_STARTS))
        self.assertEqual(set(arm.NOOP_SERVED_STEPS) & set(range(0, 32)) != set() and set(arm.NOOP_SERVED_STEPS) & set(range(32, 64)) != set(), True)


class PayloadTests(unittest.TestCase):
    def test_shapes_dtypes_and_determinism(self):
        for regime in ('exact', 'random', 'edge'):
            seeds = [1000 + row for row in range(64)]
            k = arm.step_payloads(hw, regime, seeds, 0)
            v = arm.step_payloads(hw, regime, seeds, 1)
            self.assertEqual((tuple(k.shape), k.dtype), ((64, 32, 256), torch.bfloat16), regime)
            self.assertFalse(torch.equal(k.view(torch.int16), v.view(torch.int16)), 'K and V payloads differ')
            self.assertTrue(torch.equal(k.view(torch.int16), arm.step_payloads(hw, regime, seeds, 0).view(torch.int16)), 'deterministic')

    def test_the_exact_regime_is_bfloat8_exact_and_the_others_are_not(self):
        exact = arm.step_payloads(hw, 'exact', list(range(64)), 0)
        self.assertTrue(all(hw.bf8_exact(exact[row, 0]) for row in range(8)))
        random_ = arm.step_payloads(hw, 'random', list(range(64)), 0)
        self.assertFalse(all(hw.bf8_exact(random_[row, 0]) for row in range(8)))

    def test_the_edge_regime_reaches_the_values_a_packer_can_get_wrong(self):
        values = arm.step_payloads(hw, 'edge', [5000 + row for row in range(64)], 0)[:, 0, :].contiguous().view(torch.int16)
        bits = {int(value) & 0xFFFF for value in values.flatten().tolist()}
        self.assertIn(0x8000, bits, '-0')
        self.assertTrue({0x0001, 0x8001, 0x007F, 0x807F} & bits, 'denormals')
        self.assertIn(0x4980, bits, 'the 2^20 outlier that underflows its block mates')
        self.assertIn(0x3FFF, bits)
        self.assertIn(0x3F7F, bits, 'rounding carries')
        self.assertTrue(any(0x7B00 <= (bit & 0x7FFF) <= 0x7F7F for bit in bits), 'huge')
        self.assertTrue(any(0x0080 <= (bit & 0x7FFF) <= 0x0600 for bit in bits), 'tiny normals')
        kinds = {arm.EDGE_KINDS[(row + 5000 + row) % len(arm.EDGE_KINDS)] for row in range(64)}
        self.assertEqual(kinds, set(arm.EDGE_KINDS), 'every kind occurs in one step')
        self.assertTrue(all((value & 0x7F80) != 0x7F80 for value in bits), 'no infinity or NaN')

    def test_padded_heads_carry_values_so_a_reader_that_copies_one_is_caught(self):
        payload = arm.step_payloads(hw, 'random', [1, 2], 0)
        self.assertTrue(bool((payload[:, 1:, :].float() != 0).any()))
        edge = arm.step_payloads(hw, 'edge', [1, 2], 0)
        self.assertTrue(bool((edge[:, 1:, :].float() != 0).any()))

    def test_fill_payloads_are_device_packing_inputs(self):
        for regime in ('exact', 'random', 'edge'):
            self.assertEqual(tuple(arm.fill_payloads(regime, 3, 0).shape), (64, 32, 256))
        self.assertFalse(torch.equal(arm.fill_payloads('random', 3, 0).view(torch.int16), arm.fill_payloads('random', 3, 1).view(torch.int16)))


# ---------------------------------------------------------------------------------------------
# The orchestration over a host fake of the device.
# ---------------------------------------------------------------------------------------------

class PageFake(FakeTtnn):
    """FakeTtnn with two traces' worth of capture (a list of operations replayed with the tensors' current contents), clone and placement."""
    DRAM_MEMORY_CONFIG = 'dram'
    int32 = 'int32'

    def __init__(self):
        FakeTtnn.__init__(self)
        self.current = []

    def clone(self, tensor, memory_config=None):
        return FakeTensor(tensor.value.clone(), tensor.dtype)

    def begin_trace_capture(self, mesh, cq_id=0):
        self.capturing, self.current = True, []
        return self.current

    def end_trace_capture(self, mesh, trace, cq_id=0):
        self.capturing = False

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.replays += 1
        for operation in trace:
            operation()


class FakePageRig(object):
    def __init__(self, report, page_writer=host_write, served_writer=host_write):
        self.ttnn, self.torch, self.mesh, self.hw, self.negative = PageFake(), torch, None, hw, None
        self.rig = card.Rig(self.ttnn, torch, None, report)
        self.page_writer, self.served_writer = page_writer, served_writer
        self.page_launches = 0
        self.wt = 2

    def _do(self, operation):
        if self.ttnn.capturing:
            self.ttnn.current.append(operation)
        else:
            operation()

    def served(self, caches, packed, positions, pages):
        self._do(lambda: [self.served_writer(cache, value, positions, pages) for cache, value in zip(caches, packed)])

    def fill_served(self, caches, packed, positions, pages):
        for cache, value in zip(caches, packed):
            host_write(cache, value, positions, pages)

    def page(self, caches, packed, positions, pages, source):
        def run():
            self.page_launches += 1
            for cache, value in zip(caches, packed):
                self.page_writer(cache, value, positions, pages)

        self._do(run)


def drop_last_row(cache, packed, positions, pages):
    positions = FakeTensor(positions.value.clone(), 'int32')
    packed = FakeTensor(packed.value.clone(), packed.dtype)
    packed.value[0, 63, 0, :] = cache.value[int(pages.value[63][int(positions.value[63]) // 64]), 0, int(positions.value[63]) % 64, :]
    host_write(cache, packed, positions, pages)


def wrong_tile_row(cache, packed, positions, pages):
    positions = FakeTensor(positions.value.clone(), 'int32')
    for row in range(64):
        if int(positions.value[row]) % 64 >= 32 and row % 16 == 15:
            positions.value[row] -= 32
    host_write(cache, packed, positions, pages)


def flips_one_bit(cache, packed, positions, pages):
    host_write(cache, packed, positions, pages)
    block = int(pages.value[0][int(positions.value[0]) // 64])
    bits = cache.value[block, 0].contiguous().view(torch.int16)
    bits[0, 0] ^= 1


def case_of(**selector):
    for spec in arm.build_page_cases():
        if all(spec[key] == value for key, value in selector.items()):
            return spec
    raise KeyError(selector)


class Orchestration(unittest.TestCase):
    def run_case(self, spec, **writers):
        report = dict(checks=[], cases={spec['name']: {}}, samples={})
        prig = FakePageRig(report, **writers)
        if spec['kind'] == 'page':
            arm.run_page_case(prig, hw, arm.build_page_case(spec), report)
        else:
            arm.run_noop_case(prig, hw, arm.build_noop_case(spec), report)
        return [check for check in report['checks'] if check['case'] == spec['name']], prig

    def test_a_correct_pair_passes_every_regime_source_and_mode_and_records_every_required_check(self):
        for selector in (dict(regime='exact', source='dram', mode='eager', width=2052, seed=0),
                         dict(regime='random', source='l1', mode='replay_changed', width=2052, seed=0),
                         dict(regime='edge', source='dram', mode='replay_changed', width=2052, seed=1),
                         dict(regime='edge', source='l1', mode='replay_unchanged', width=2052, seed=2)):
            spec = case_of(kind='page', **selector)
            checks, prig = self.run_case(spec)
            self.assertTrue(all(check['exact'] for check in checks), (selector, [c for c in checks if not c['exact']][:1]))
            self.assertEqual({(check['name'], check['step']) for check in checks}, set(arm.required_checks(spec)), selector)
            self.assertTrue(all(check['kind'] == 'page' and check['writer'] == 'page64' for check in checks))
            replay = spec['mode'] != 'eager'
            self.assertEqual(prig.ttnn.replays, 2 * (arm.PAGE_STEPS + 1) + 1 if replay else 0, selector)

    def test_a_page_writer_that_drops_a_row_fails_served_equal(self):
        spec = case_of(kind='page', regime='random', source='l1', mode='eager', width=2052, seed=0)
        checks, prig = self.run_case(spec, page_writer=drop_last_row)
        self.assertTrue(any(not check['exact'] and check['name'].startswith('served_equal') for check in checks))

    def test_a_page_writer_that_writes_the_wrong_tile_row_fails(self):
        spec = case_of(kind='page', regime='exact', source='dram', mode='eager', width=2052, seed=0)
        checks, prig = self.run_case(spec, page_writer=wrong_tile_row)
        failed = {check['name'] for check in checks if not check['exact']}
        self.assertTrue({'served_equal_k', 'host_equal_k'} & failed, failed)

    def test_one_flipped_bit_in_one_cache_fails_in_every_regime(self):
        for regime in ('exact', 'random', 'edge'):
            source = 'dram' if regime == 'exact' else 'l1'
            spec = case_of(kind='page', regime=regime, source=source, mode='eager', width=2052, seed=0)
            checks, prig = self.run_case(spec, page_writer=flips_one_bit)
            self.assertTrue(any(not check['exact'] for check in checks), regime)

    def test_a_served_writer_that_differs_from_a_correct_page_writer_fails_too(self):
        spec = case_of(kind='page', regime='random', source='l1', mode='eager', width=2052, seed=0)
        checks, prig = self.run_case(spec, served_writer=flips_one_bit)
        self.assertTrue(any(not check['exact'] for check in checks))

    def test_a_page_writer_that_is_not_idempotent_fails_twice_equal(self):
        counter = {'calls': 0}

        def drifting(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            counter['calls'] += 1
            if counter['calls'] > 2 * (arm.PAGE_STEPS + 1):
                block = int(pages.value[0][int(positions.value[0]) // 64])
                cache.value[block, 0, int(positions.value[0]) % 64, 0] += 1

        spec = case_of(kind='page', regime='random', source='l1', mode='eager', width=2052, seed=0)
        checks, prig = self.run_case(spec, page_writer=drifting)
        self.assertEqual({check['name'] for check in checks if not check['exact']}, {'twice_equal_k', 'twice_equal_v'})

    def test_the_input_and_the_table_must_be_left_alone(self):
        def corrupt(cache, packed, positions, pages):
            host_write(cache, packed, positions, pages)
            packed.value[0, 0, 0, 0] = 5

        spec = case_of(kind='page', regime='exact', source='dram', mode='eager', width=2052, seed=0)
        checks, prig = self.run_case(spec, page_writer=corrupt)
        self.assertFalse([check for check in checks if check['name'] == 'input_unchanged'][0]['exact'])

    def test_noop_cases_pass_when_nothing_moves_and_fail_when_a_stored_block_moves(self):
        for writer, source in (('chained64', 'dram'), ('page64', 'dram'), ('page64', 'l1')):
            spec = case_of(kind='noop', writer=writer, regime='random', source=source, width=2052)
            checks, prig = self.run_case(spec)
            self.assertTrue(all(check['exact'] for check in checks), (writer, source))
            self.assertEqual({(check['name'], check['step']) for check in checks}, set(arm.required_checks(spec)))
        spec = case_of(kind='noop', writer='page64', regime='edge', source='dram', width=2052)
        checks, prig = self.run_case(spec, page_writer=flips_one_bit)
        self.assertTrue(any(not check['exact'] for check in checks))
        spec = case_of(kind='noop', writer='chained64', regime='random', source='dram', width=2052)
        checks, prig = self.run_case(spec, served_writer=flips_one_bit)
        self.assertTrue(any(not check['exact'] for check in checks), 'a served writer that moves a stored block is the U1 failure itself')


# ---------------------------------------------------------------------------------------------
# The card test's integration and the verdicts.
# ---------------------------------------------------------------------------------------------

def synthetic_page_report(base=None, mutate=None):
    """E1's synthetic report plus a page64 arm that passed everything (no device)."""
    report = base if base is not None else e1.make_report()
    report['requested']['writers'] = list(card.WRITERS) + [card.PAGE_WRITER]
    report['requested']['page'] = dict(sources=list(arm.SOURCES), regimes=list(arm.REGIMES), noop=True, negative=None, wt=None)
    specs = arm.build_page_cases(first_index=len(report['plan']))
    for spec in specs:
        required = arm.required_checks(spec)
        report['plan'].append(dict(name=spec['name'], kind=spec['kind'], writer=spec['writer'], required=[list(item) for item in required]))
        report['cases'][spec['name']] = dict(exact=True)
        for name, step in required:
            report['checks'].append(dict(case=spec['name'], kind=spec['kind'], writer=spec['writer'], width=spec['width'], mode=spec['mode'],
                                         seed=spec['seed'], source=spec['source'], regime=spec['regime'], name=name, step=step, exact=True))
    wt = kvpw.DEFAULT_WT
    report['page'] = dict(design=kvpw.design(wt), design_signature=kvpw.design_signature(wt), wt=wt, units=kvpw.unit_count(wt),
                          kernel_sha256=kvpw.source_sha256(), grid=[13, 10], negative=None)
    report['sources']['kv_page_writer_tp4.cpp'] = kvpw.source_sha256()
    if mutate is not None:
        mutate(report)
    report['decision'] = card.decide(report)
    report['page_decision'] = arm.decide_page(report)
    report['verdict_line'] = card.verdict_line(report)
    return report


class VerdictTests(unittest.TestCase):
    def test_a_full_pass(self):
        report = synthetic_page_report()
        self.assertEqual(report['decision']['verdict'], 'PASS')
        self.assertEqual(report['page_decision']['verdict'], 'PASS')
        self.assertEqual(arm.page_scope(report), 'full')
        line = report['verdict_line']
        self.assertTrue(line.startswith('ORDERED_WRITER verdict=PASS scope=full width=4096 chips=1of4 '), line)
        self.assertIn(' page=PASS page_scope=full page_checks=', line)
        self.assertIn('writers=chained64,tiles32,page64', line)

    def test_the_e1_verdict_tally_and_scope_ignore_the_page_checks(self):
        report = synthetic_page_report()
        self.assertEqual(card.tally(report)['checks'], len(card.build_cases()) and sum(len(card.required_checks(spec)) for spec in card.build_cases()))
        report['checks'][-1]['exact'] = False                       # a page check
        report['decision'] = card.decide(report)
        report['page_decision'] = arm.decide_page(report)
        self.assertEqual(report['decision']['verdict'], 'PASS', 'E1 is still E1')
        self.assertEqual(report['page_decision']['verdict'], 'FAIL')
        self.assertIn(' page=FAIL ', card.verdict_line(dict(report, decision=report['decision'])))

    def test_a_missing_check_a_raised_case_and_a_cut_case_are_no_decision(self):
        report = synthetic_page_report()
        page = [index for index, check in enumerate(report['checks']) if check.get('kind') == 'page']
        short = copy.deepcopy(report)
        short['checks'].pop(page[3])
        self.assertEqual(arm.decide_page(short)['verdict'], 'NO-DECISION')
        for key in ('error', 'skipped'):
            broken = copy.deepcopy(report)
            name = next(entry['name'] for entry in broken['plan'] if entry.get('kind') == 'page')
            broken['cases'][name] = {key: True}
            self.assertEqual(arm.decide_page(broken)['verdict'], 'NO-DECISION', key)
        errored = copy.deepcopy(report)
        errored['error'] = 'boom'
        self.assertEqual(arm.decide_page(errored)['verdict'], 'NO-DECISION')
        self.assertEqual(arm.decide_page(e1.make_report())['verdict'], 'absent')

    def test_scope_reduced_without_everything(self):
        for mutation in (lambda r: r['requested']['page'].update(sources=['dram']), lambda r: r['requested']['page'].update(regimes=['exact']),
                         lambda r: r['requested']['page'].update(noop=False), lambda r: r['requested']['page'].update(negative='drop'),
                         lambda r: r['requested'].update(seeds=[0]), lambda r: r['requested'].update(modes=['eager']),
                         lambda r: r['requested'].update(widths=[4096])):
            report = synthetic_page_report()
            mutation(report)
            self.assertEqual(arm.page_scope(report), 'reduced')

    def test_the_proof_counters(self):
        report = synthetic_page_report()
        proofs = arm.proofs_of(report)
        for label in ('noop_served', 'noop_page', 'twice_page'):
            self.assertGreater(proofs[label]['checks'], 0, label)
            self.assertEqual(proofs[label]['checks'], proofs[label]['exact'])
        self.assertEqual(arm.tally_page(report)['checks'], arm.tally_page(report)['exact'])

    def test_arguments(self):
        arguments = card.parse(['--out', 'x.json', '--writers', 'page64', '--page-sources', 'l1', '--page-regimes', 'exact,edge', '--page-wt', '4',
                                '--page-negative', 'drop', '--no-page-noop'])
        self.assertEqual((arguments.writers, arguments.page_sources, arguments.page_regimes, arguments.page_wt, arguments.page_negative, arguments.no_page_noop),
                         (['page64'], ['l1'], ['exact', 'edge'], 4, 'drop', True))
        default = card.parse(['--out', 'x.json'])
        self.assertEqual(default.writers, list(card.WRITERS), 'the page arm is opt-in: E1 jobs are unchanged')
        self.assertEqual(card.build_cases(writers=['page64', 'tiles32'], widths=[2052], seeds=[0], modes=['eager'])[0]['writer'], 'tiles32')
        with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
            card.parse(['--out', 'x.json', '--page-negative', 'bogus'])
        self.assertEqual(card.page_cases_of(default), [])
        self.assertTrue(card.page_cases_of(arguments))

    def test_main_runs_the_plan_and_refuses_off_four_cards(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {'ttnn': mock.MagicMock()}):
            out = Path(directory) / 'r.json'
            env = {key: value for key, value in os.environ.items() if key != 'QWEN_FAST_TP'}
            with mock.patch.dict(os.environ, env, clear=True):
                code = card.main(['--out', str(out), '--widths', '2052', '--writers', 'page64', '--seeds', '0', '--modes', 'eager'])
            report = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report['page_decision']['verdict'], 'NO-DECISION')
        self.assertIn('QWEN_FAST_TP=4', report['error'])
        kinds = {entry.get('kind') for entry in report['plan']}
        self.assertEqual(kinds, {'page', 'noop'})
        self.assertTrue(all(entry['required'] for entry in report['plan']))
        self.assertIn(' page=NO-DECISION ', report['verdict_line'])
        self.assertEqual(report['requested']['page']['sources'], ['dram', 'l1'])


# ---------------------------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------------------------

class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ('ordered_cache.py', 'page_width_tp4.py', 'ordered_writer_evidence_tp4.json'):
            shutil.copyfile(pw.HERE / name, self.dir / name)
        self.evidence = self.dir / 'ordered_writer_evidence_tp4.json'

    def tearDown(self):
        self.tmp.cleanup()

    def run_recorder(self, report, merge=False, **overrides):
        path = self.dir / 'ordered-20261010T010203.json'
        path.write_text(json.dumps(report, indent=1))
        argv = ['--evidence', str(self.evidence), '--module', str(self.dir / 'page_width_tp4.py'), '--sources-root', str(self.dir),
                '--report', str(path), '--run', '36900000009', '--tag', 'v700', '--commit', COMMIT, '--image', 'tp4-fusion-1']
        if merge:
            argv.append('--merge-page')
        for key, value in overrides.items():
            argv += ['--' + key.replace('_', '-'), value]
        lines = []
        code = rec.main(argv, out=lines.append)
        return code, lines

    def page_only(self, mutate=None):
        report = synthetic_page_report(base=e1.make_report(self.dir), mutate=mutate)
        report['requested']['writers'] = [card.PAGE_WRITER]
        report['plan'] = [entry for entry in report['plan'] if entry.get('kind')]
        report['checks'] = [check for check in report['checks'] if check.get('kind')]
        report['cases'] = {name: state for name, state in report['cases'].items() if name.startswith(('page64-', 'noop-'))}
        return report

    def test_the_shipped_record_cannot_engage_the_lever_and_a_recorded_page_block_does(self):
        self.assertEqual(kvpw.evidence_problems(), ['no page_writer record'])
        report = synthetic_page_report(base=e1.make_report(self.dir))
        code, lines = self.run_recorder(report)
        self.assertEqual(code, 0, lines)
        payload = self.evidence.read_bytes()
        evidence = json.loads(payload.decode())
        block = evidence['page_writer']
        self.assertEqual((block['status'], block['scope'], block['failures'], block['wt'], block['units']), ('PASS', 'full', 0, 2, 64))
        self.assertEqual(block['design_signature'], kvpw.design_signature(2))
        self.assertEqual((block['widths'], block['sources'], block['caches']), ([2052, 4096], ['dram', 'l1'], ['k', 'v']))
        self.assertEqual(block['counts']['checks'], block['counts']['exact'])
        self.assertEqual((block['run'], block['tag'], block['commit'], block['image']), (36900000009, 'v700', COMMIT, 'tp4-fusion-1'))
        self.assertEqual(rec.dump(evidence), payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.assertIn("ORDERED_WRITER_EVIDENCE_TP4_SHA256 = '%s'" % digest, (self.dir / 'page_width_tp4.py').read_text())
        for width in (2052, 4096):
            for source in ('dram', 'l1'):
                self.assertEqual(kvpw.evidence_problems(2, width, source, path=self.evidence, expected=digest, sources_root=self.dir), [])
        self.assertTrue(kvpw.evidence_problems(2, 1024, 'dram', path=self.evidence, expected=digest, sources_root=self.dir))
        self.assertTrue(kvpw.evidence_problems(4, 4096, 'dram', path=self.evidence, expected=digest, sources_root=self.dir))
        self.assertTrue(pw.admitted(4096, {'QWEN_FAST_TP': '4'}, path=self.evidence, expected=digest, sources_root=self.dir), 'the E1 admission is unchanged')

    def test_merge_adds_the_block_to_the_record_on_disk_and_touches_nothing_else(self):
        before = json.loads(self.evidence.read_text())
        code, lines = self.run_recorder(self.page_only(), merge=True)
        self.assertEqual(code, 0, lines)
        after = json.loads(self.evidence.read_text())
        self.assertEqual({key: value for key, value in after.items() if key != 'page_writer'}, before)
        self.assertEqual(kvpw.block_problems(after['page_writer'], 2), [])
        digest = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        self.assertIn("ORDERED_WRITER_EVIDENCE_TP4_SHA256 = '%s'" % digest, (self.dir / 'page_width_tp4.py').read_text())
        code, lines = self.run_recorder(self.page_only(), merge=True, run='36900000010')
        self.assertEqual(code, 0, lines)
        self.assertEqual(json.loads(self.evidence.read_text())['page_writer']['run'], 36900000010, 're-recording replaces the block')

    def test_a_page_only_report_is_not_an_e1_record_without_merge(self):
        before = self.evidence.read_bytes()
        code, lines = self.run_recorder(self.page_only())
        self.assertEqual(code, 1)
        self.assertEqual(self.evidence.read_bytes(), before)

    def test_a_dry_run_writes_nothing(self):
        before = {name: (self.dir / name).read_bytes() for name in ('page_width_tp4.py', 'ordered_writer_evidence_tp4.json')}
        code, lines = self.run_recorder(self.page_only(), merge=True, **{})
        self.assertEqual(code, 0)
        dry = {name: (self.dir / name).read_bytes() for name in before}
        self.assertNotEqual(before, dry)
        for name, data in before.items():
            (self.dir / name).write_bytes(data)
        path = self.dir / 'r.json'
        path.write_text(json.dumps(self.page_only()))
        lines = []
        self.assertEqual(rec.main(['--evidence', str(self.evidence), '--module', str(self.dir / 'page_width_tp4.py'), '--sources-root', str(self.dir),
                                   '--report', str(path), '--run', '1', '--tag', 't', '--commit', COMMIT, '--image', 'tp4-x', '--merge-page', '--dry-run'],
                                  out=lines.append), 0, lines)
        self.assertEqual({name: (self.dir / name).read_bytes() for name in before}, before)

    def test_reports_that_do_not_qualify_are_refused_and_nothing_is_written(self):
        mutations = {
            'a negative control': lambda r: (r['requested']['page'].update(negative='drop'), r['page'].update(negative='drop')),
            'no noop': lambda r: r['requested']['page'].update(noop=False),
            'one source': lambda r: r['requested']['page'].update(sources=['dram']),
            'one regime': lambda r: r['requested']['page'].update(regimes=['exact', 'random']),
            'one seed': lambda r: r['requested'].update(seeds=[0]),
            'one width': lambda r: r['requested'].update(widths=[2052]),
            'one mode': lambda r: r['requested'].update(modes=['eager']),
            'an inexact check': lambda r: next(c for c in r['checks'] if c.get('kind') == 'page' and c['name'] == 'served_equal_k').update(exact=False),
            'an inexact noop': lambda r: next(c for c in r['checks'] if c.get('kind') == 'noop').update(exact=False),
            'a missing check': lambda r: r['checks'].pop(next(i for i, c in enumerate(r['checks']) if c.get('kind') == 'page')),
            'a raised case': lambda r: r['cases'].update({next(n for n in r['cases'] if n.startswith('page64-')): dict(error='x')}),
            'another design': lambda r: r['page'].update(design_signature='0' * 64),
            'another kernel': lambda r: r['sources'].update({'kv_page_writer_tp4.cpp': 'f' * 64}),
            'no design': lambda r: r.pop('page'),
            'a checkpoint': lambda r: r.update(in_progress='x'),
            'an error': lambda r: r.update(error='boom'),
            'a bad unit width': lambda r: r['page'].update(wt=3),
        }
        for label, mutate in mutations.items():
            report = self.page_only()
            mutate(report)
            report['page_decision'] = arm.decide_page(report) if 'checkpoint' not in label else report['page_decision']
            report['verdict_line'] = card.verdict_line(report)
            before = {name: (self.dir / name).read_bytes() for name in ('page_width_tp4.py', 'ordered_writer_evidence_tp4.json')}
            code, lines = self.run_recorder(report, merge=True)
            self.assertEqual(code, 1, (label, lines))
            self.assertEqual(before, {name: (self.dir / name).read_bytes() for name in before}, label)

    def test_a_record_on_disk_that_no_longer_stands_cannot_be_extended(self):
        evidence = json.loads(self.evidence.read_text())
        evidence['status'] = 'PENDING'
        self.evidence.write_text(json.dumps(evidence))
        code, lines = self.run_recorder(self.page_only(), merge=True)
        self.assertEqual(code, 1)
        self.assertTrue(any('the record on disk' in line for line in lines), lines)
        self.evidence.write_text('not json')
        code, lines = self.run_recorder(self.page_only(), merge=True)
        self.assertEqual(code, 1)

    def test_the_commit_image_and_run_are_checked_for_a_merge_too_and_the_block_is_public_safe(self):
        for key, value in (('commit', 'abc'), ('image', 'registry.example/x@sha256:' + 'a' * 64), ('tag', '')):
            code, lines = self.run_recorder(self.page_only(), merge=True, **{key: value})
            self.assertEqual(code, 1, key)
        self.assertEqual(self.run_recorder(self.page_only(), merge=True)[0], 0)
        self.assertEqual(rec.base.hygiene_problems(json.loads(self.evidence.read_text())), [])
        combined = synthetic_page_report(base=e1.make_report(self.dir))
        combined['verdict_line'] += ' /opt/results/x'
        code, lines = self.run_recorder(combined)
        self.assertEqual(code, 1)
        self.assertTrue(any('host path' in line for line in lines), lines)

    def test_a_record_without_the_page_arm_is_what_it_was(self):
        report = e1.make_report(self.dir)
        code, lines = self.run_recorder(report)
        self.assertEqual(code, 0, lines)
        self.assertNotIn('page_writer', json.loads(self.evidence.read_text()))
        self.assertEqual(kvpw.evidence_problems(2, 4096, 'dram', path=self.evidence, expected=hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
                                                sources_root=self.dir), ['no page_writer record'])


if __name__ == '__main__':
    unittest.main()
