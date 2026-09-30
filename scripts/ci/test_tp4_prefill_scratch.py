"""The four-card garbage/EOS defect (v172 D1): every prefill after the first packed replay came out wrong.

The fast path never ran the model's prefill warmup (serving runs trace_mode=decode_only, so warmup_model_prefill returns
early), so the first prefill compiled its programs and allocated the B=1 GDN scratch after the attach-time packed traces
existed. serving_runtime.prefill_warm_before_traces now warms the whole eager prefill before the packed blocks are built at
four cards (the scratch, every bucket at the plugin's page-table width, the slot writes); the pair (QWEN_FAST_TP unset or 2)
keeps its order. The smoke check judges the text (instant EOS, foreign script), the first prefill's ledger line, the warm
line's place before Metal's 'unsafe allocation' warning, and each prefill's late programs.
"""
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import c2_smoke_check as check
import dflash_prefill_window
import memory_ledger
import serving_runtime
import test_tp4_draft_window as window
from test_memory_ledger import FakeOperations, ledger_for

HERE = Path(__file__).resolve().parent
LEDGER = ('(EngineCore pid=66) 2026-09-30 14:15:09.482 | INFO     | memory_ledger:log_line:121 - [MEMLEDGER] phase=prefill '
          'point=after req=4b3-b36e8145 item=model_after_prefill chip0=%s chip1=0.081GB chip2=0.081GB chip3=0.081GB buffers=%d '
          'largest=0.4MB unreadable=0')


class Model:
    """A model double that records the warm's calls in order."""

    def __init__(self, log=None, cache=None):
        self.log = [] if log is None else log
        self._gdn_prefill_scratch = None
        self.mesh_device = SimpleNamespace(num_program_cache_entries=(lambda: cache[0]) if cache else None)

    def _ensure_gdn_prefill_scratch(self):
        self.log.append('ensure')
        self._gdn_prefill_scratch = ['scratch']

    def _bind_gdn_prefill_scratch(self):
        self.log.append('bind')
        return 'decode-bindings'

    def _unbind_gdn_prefill_scratch(self, previous):
        self.log.append(('unbind', previous))

    def warmup_prefill_masked_buckets(self, page_table):
        self.log.append(('buckets', tuple(page_table.shape)))

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None):
        self.log.append(('prefill', tuple(token_ids_list[0].shape), tuple(page_table.shape), list(empty_slots), list(valid_lens)))


FOUR = {'QWEN_FAST_TP': '4'}


def warm(model, runner=None, slots=4, environ=FOUR):
    runner = SimpleNamespace(max_num_blocks_per_req=2052) if runner is None else runner
    with patch.object(serving_runtime, 'pindiag'):
        return serving_runtime.prefill_warm_before_traces(runner, model, slots, environ)


class WarmBeforeTraces(unittest.TestCase):
    def test_four_cards_warms_in_order_at_the_plugin_width(self):
        model = Model()
        self.assertTrue(warm(model))
        self.assertEqual(model.log, [
            'ensure', 'bind', ('buckets', (1, 2052)), ('unbind', 'decode-bindings'),
            ('prefill', (1, 64), (1, 2052), [0], [64]), ('prefill', (1, 64), (1, 2052), [1], [64]),
            ('prefill', (1, 64), (1, 2052), [2], [64]), ('prefill', (1, 64), (1, 2052), [3], [64]),
            ('prefill', (1, 2112), (1, 2052), [0], [2112]), ('prefill', (1, 4096), (1, 2052), [0], [4096])])

    def test_the_ledger_claims_the_scratch_then_the_warm(self):
        model = Model()
        with patch.object(serving_runtime.memory_ledger, 'record') as record:
            warm(model)
        self.assertEqual([call.args + tuple(sorted(call.kwargs.items())) for call in record.call_args_list], [
            ('P5', ('model_prefill_scratch', ['scratch']), ('point', 'prefill_scratch')), ('P5', ('point', 'prefill_warm'))])

    def test_the_log_line_names_the_width_slots_and_programs(self):
        cache = [117]
        model = Model(cache=cache)
        model.prefill_paged_slots = lambda *args, **kwargs: cache.__setitem__(0, cache[0] + 20)
        with patch.object(serving_runtime, 'pindiag') as line:
            serving_runtime.prefill_warm_before_traces(SimpleNamespace(max_num_blocks_per_req=2052), model, 2, FOUR)
        self.assertTrue(line.call_args.args[0].startswith(serving_runtime.WARM_MARKER))
        self.assertEqual(line.call_args.args[1:6], (2052, 2, [2112, 4096], 117, 117 + 20 * 4))

    def test_the_masked_warmup_is_unbound_even_when_it_fails(self):
        model = Model()

        def fail(page_table):
            raise RuntimeError('compile failed')

        model.warmup_prefill_masked_buckets = fail
        with self.assertRaises(RuntimeError):
            warm(model)
        self.assertEqual(model.log, ['ensure', 'bind', ('unbind', 'decode-bindings')])

    def test_a_runner_without_the_width_refuses_the_attach_before_allocating(self):
        for runner in (SimpleNamespace(), SimpleNamespace(max_num_blocks_per_req=None), SimpleNamespace(max_num_blocks_per_req=0)):
            model = Model()
            with self.assertRaisesRegex(ValueError, 'max_num_blocks_per_req'):
                warm(model, runner)
            self.assertEqual(model.log, [])

    def test_the_pair_is_unchanged(self):
        for environ in ({}, {'QWEN_FAST_TP': '2'}):
            model = Model()
            with patch.object(serving_runtime.memory_ledger, 'record') as record:
                self.assertFalse(warm(model, SimpleNamespace(), environ=environ))   # not even the width is read
            self.assertEqual(model.log, [])
            record.assert_not_called()

    def test_a_model_without_the_method_is_left_alone(self):
        self.assertFalse(warm(SimpleNamespace()))

    def test_the_attach_warms_after_the_p4_scopes_and_before_the_first_block_is_built(self):
        # Fails without the call: the warm must run after the draft weights (P5) and before any capture, whether or not
        # the attach builds a packed block.
        source = (HERE / 'serving_runtime.py').read_text(encoding='utf-8')
        call = source.index("prefill_warm_before_traces(runner, model, policy['scheduler_requests'], operations=operations)")
        self.assertEqual(source.count("prefill_warm_before_traces(runner, model, policy['scheduler_requests']"), 1)
        self.assertLess(source.index("memory_ledger.record('P4', combined_runtime=audit)"), call)
        self.assertLess(source.index("memory_ledger.record('P5', draft_weights=weights)"), call)
        self.assertLess(call, source.index('from packed_verifier import PackedVerifierEngine'))
        self.assertLess(call, source.index('packed_block = PackedVerifierEngine('))
        self.assertNotIn('prefill_scratch_before_traces', source)

    def test_every_prefill_shape_is_warmed(self):
        # The model's bucket warm covers the buckets and fill widths, and the long prompts run the chunk loop through the
        # served entry.
        self.assertEqual(serving_runtime.WARM_LONG_PROMPTS, (2048 + 64, 4096))
        self.assertEqual(serving_runtime.WARM_SLOT_TOKENS, 64)
        fixture = (HERE / 'fixtures' / 'qwen36_model.py').read_text(encoding='utf-8')
        self.assertRegex(fixture, r'_PREFILL_MASK_BUCKETS\s*=\s*[\(\[]128,\s*256,\s*512,\s*1024,\s*2048[\)\]]')


class Tripwire(unittest.TestCase):
    def setUp(self):
        self.addCleanup(dflash_prefill_window.set_program_counter, None)

    def factories(self, cache):
        def capture_factory(position, start=0):
            cache[0] += 3
            return SimpleNamespace(position=position)

        def bridge_factory(state, capture):
            return 'bridge'

        return capture_factory, bridge_factory

    def test_each_prefill_logs_its_programs_and_its_windows(self):
        cache = [100]
        model = Model(cache=cache)
        capture_factory, bridge_factory = serving_runtime.prefill_tripwire(model, *self.factories(cache), FOUR)
        with patch.object(serving_runtime, 'pindiag') as line:
            capture = capture_factory(8376)
            cache[0] += 10   # the prefill's programs, two of them the window snapshot's
            dflash_prefill_window._WINDOW_PROGRAMS[0] = 2
            self.assertEqual(bridge_factory(SimpleNamespace(), capture), 'bridge')
        self.assertEqual(line.call_args.args, (serving_runtime.PREFILL_PROGRAMS_MARKER + '{}->{} window={} prompt={}',
                                               103, 113, 2, 8376))

    def test_the_window_snapshot_counts_its_own_programs(self):
        cache = [5]
        dflash_prefill_window.set_program_counter(lambda: cache[0])
        dflash_prefill_window.window_programs(reset=True)
        with patch.object(dflash_prefill_window, '_snapshot_prefill_tail', lambda *a, **k: cache.__setitem__(0, 9)):
            dflash_prefill_window.snapshot_prefill_tail(None, None, 10)
        self.assertEqual(dflash_prefill_window.window_programs(reset=True), 4)
        self.assertEqual(dflash_prefill_window.window_programs(), 0)

    def test_the_pair_and_a_model_without_a_cache_get_the_factories_back(self):
        cache = [1]
        originals = self.factories(cache)
        self.assertEqual(serving_runtime.prefill_tripwire(Model(cache=cache), *originals, {'QWEN_FAST_TP': '2'}), originals)
        self.assertEqual(serving_runtime.prefill_tripwire(Model(), *originals, FOUR), originals)

    def test_the_attach_wraps_the_factories_before_the_lifecycle(self):
        source = (HERE / 'serving_runtime.py').read_text(encoding='utf-8')
        self.assertLess(source.index('capture_factory, bridge_factory = prefill_tripwire('),
                        source.index('lifecycle = FastServingLifecycle('))


class TextJudge(unittest.TestCase):
    def results(self, users):
        return {'concurrent4_steady': {'users': users}}

    def good(self):
        return dict(tokens=800, finish='length', text='This code defines a class that parses config files. ' * 8)

    def test_instant_eos_fails(self):
        users = [dict(tokens=1, finish='stop', text='')] + [self.good()] * 3
        problems = check.smoke_problems(self.results(users))
        self.assertTrue(any('user 0' in p and 'stop after 1 token' in p for p in problems), problems)

    def test_foreign_script_garbage_fails(self):
        garbage = dict(tokens=800, finish='length', text=u'กขฃ 中文字 ЖЗК ' * 30 + '!?;{}')
        problems = check.smoke_problems(self.results([self.good(), self.good(), garbage, self.good()]))
        self.assertTrue(any('user 2' in p and 'Latin' in p for p in problems), problems)

    def test_four_coherent_answers_pass(self):
        self.assertEqual(check.smoke_problems(self.results([self.good()] * 4)), [])

    def test_a_long_stop_is_not_an_instant_eos(self):
        users = [dict(tokens=300, finish='stop', text='An answer that ends. ' * 20)] + [self.good()] * 3
        self.assertEqual(check.smoke_problems(self.results(users)), [])


WARM = '(EngineCore pid=66) | INFO | [PINDIAG] four-card eager prefill warmed before the packed traces: page_table_blocks=2052 slots=4 programs=117->503 ms=9000'
UNSAFE = ('(EngineCore pid=66) [warning] Allocating device buffers is unsafe due to the existence of an active trace. '
          'These buffers may be corrupted once a trace is executed.')
PROGRAMS = '(EngineCore pid=66) | INFO | [PINDIAG] four-card prefill programs=%s->%s window=%s prompt=%s'


def log_of(*lines):
    return '\n'.join(lines)


CLEAN = log_of(WARM, UNSAFE, PROGRAMS % (600, 603, 3, 54), LEDGER % ('0.000GB', 0), PROGRAMS % (603, 603, 0, 8376))


class FirstPrefillLedger(unittest.TestCase):
    smoke = 'SMOKE_JSON ' + '{"warmup": {"value": 200}}'

    def run_check(self, log, tp):
        return check.check(self.smoke, log, False, env={'QWEN_FAST_TP': tp})

    def test_late_allocation_fails_at_four_cards(self):
        problems, facts = self.run_check(log_of(WARM, UNSAFE, PROGRAMS % (600, 603, 3, 54), LEDGER % ('0.081GB', 1484)), '4')
        self.assertEqual(facts['first_prefill_model_buffers'], 1484)
        self.assertTrue(any('first prefill' in p for p in problems), problems)

    def test_a_clean_first_prefill_passes(self):
        problems, facts = self.run_check(CLEAN, '4')
        self.assertEqual(facts['first_prefill_model_buffers'], 0)
        self.assertEqual(facts['prefills_counted'], 2)
        self.assertEqual(problems, [])

    def test_the_pair_is_not_judged(self):
        problems, facts = self.run_check(LEDGER % ('0.081GB', 1484), '2')
        self.assertNotIn('first_prefill_model_buffers', facts)
        self.assertEqual(problems, [])
        self.assertEqual(self.run_check('', '2')[0], [])


class LateProgramTripwire(unittest.TestCase):
    def problems(self, *lines):
        return check.check('SMOKE_JSON {"warmup": {"value": 200}}', log_of(*lines), False, env=FOUR)[0]

    def test_growth_above_the_window_fails(self):
        problems = self.problems(WARM, UNSAFE, PROGRAMS % (600, 650, 3, 8376))
        self.assertTrue(any('8376 tokens compiled 47 program' in p for p in problems), problems)

    def test_growth_inside_the_window_passes(self):
        self.assertEqual(self.problems(WARM, UNSAFE, PROGRAMS % (600, 650, 50, 8376)), [])

    def test_the_warm_after_the_metal_warning_fails(self):
        problems = self.problems(UNSAFE, WARM, PROGRAMS % (600, 600, 0, 54))
        self.assertTrue(any('came after the first allocation made with a trace live' in p for p in problems), problems)

    def test_a_missing_warm_fails(self):
        problems = self.problems(UNSAFE, PROGRAMS % (600, 600, 0, 54))
        self.assertTrue(any('eager prefill warm never ran' in p for p in problems), problems)

    def test_no_prefill_line_fails(self):
        problems = self.problems(WARM, UNSAFE)
        self.assertTrue(any('tripwire did not run' in p for p in problems), problems)

    def test_the_facts_carry_the_order_and_the_count(self):
        facts = check.check('SMOKE_JSON {"warmup": {"value": 200}}', CLEAN, False, env=FOUR)[1]
        self.assertEqual((facts['prefill_warm_line'], facts['unsafe_allocation_line'], facts['prefills_with_late_programs']), (1, 2, 0))


class LedgerEndToEnd(unittest.TestCase):
    """The ledger's own item line, the helper's claim and c2_smoke_check's four-card rule together: a scratch allocated by the
    first prefill is reported there, one allocated (and claimed) before the block is not, and the block's walk plays no part."""

    def first_prefill(self, before_traces):
        operations = FakeOperations()
        ledger, lines, _ = ledger_for(operations)
        model = SimpleNamespace(weights=[operations.tensor((32, 1024))], _gdn_prefill_scratch=None)

        def ensure():
            if model._gdn_prefill_scratch is None:
                model._gdn_prefill_scratch = [operations.tensor((1, 32, 2560)) for _ in range(3)]

        model._ensure_gdn_prefill_scratch = ensure
        model._bind_gdn_prefill_scratch = lambda: None
        model._unbind_gdn_prefill_scratch = lambda previous: None
        model.warmup_prefill_masked_buckets = lambda page_table: None
        model.prefill_paged_slots = lambda *args, **kwargs: None
        with patch.object(memory_ledger, '_active', ledger), patch.object(serving_runtime, 'pindiag'):
            memory_ledger.record('P0', model=model)
            if before_traces:
                serving_runtime.prefill_warm_before_traces(SimpleNamespace(max_num_blocks_per_req=64), model, 1, FOUR)
            # A block whose walk does not reach the model.
            memory_ledger.record('P6', point='block1', packed_block=SimpleNamespace(rows=[operations.tensor((64, 64))]))
            ensure()   # the first prefill binds the scratch, allocating it only if nothing did before
            memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id('chatcmpl-4b3-b36e8145'),
                                 request='chatcmpl-4b3-b36e8145', model_after_prefill=model)
        return check.first_prefill_buffers('\n'.join(lines))

    def test_the_late_scratch_is_reported_and_the_claimed_one_is_not(self):
        self.assertEqual(self.first_prefill(False), 6)   # three tensors on two chips
        self.assertIsNone(self.first_prefill(True))


class Templates(unittest.TestCase):
    folder = str(HERE / 'references' / 'tp4-garbage-jobs')

    def test_the_warm_chain_is_one_image_built_by_v0b(self):
        rows = {row[0]: row for row in window.read_order(self.folder)}
        for name in ('V0b-build-stackfix3', 'V1b-fix-gate-pairs-smoke', 'V3-fix-staggered-trigger'):
            self.assertEqual(rows[name][2], 'tp4-stackfix-3', name)
        build = window.parsed('V0b-build-stackfix3', self.folder)[0]
        self.assertEqual((build['C2_ACTIONS'], build['C2_IMAGE_TAG']), ('status reset build', 'tp4-stackfix-3'))
        self.assertEqual(window.parsed('V1b-fix-gate-pairs-smoke', self.folder)[0]['C2_SMOKE_TESTS'],
                         'warmup,coding,concurrent4,concurrent4_steady')

    def test_every_template_parses_and_is_ordered_with_its_image(self):
        rows = window.read_order(self.folder)
        on_disk = sorted(path.stem for path in Path(self.folder).glob('*.env'))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)
        for name, kind, image, *_ in rows:
            self.assertIn(kind, ('stop', 'soft'))
            values, _ = window.parsed(name, self.folder)
            self.assertEqual(values['C2_IMAGE_TAG'], image, name)
            self.assertEqual(values['C2_CARDS'], 'quad')

    def test_the_isolation_jobs_run_the_images_that_exist(self):
        self.assertEqual({row[0]: row[2] for row in window.read_order(self.folder)[:3]},
                         {'G3-stack-gate-steady': 'tp4-stack-1', 'R3-stack-speed-d1-sequence': 'tp4-stack-1',
                          'R2-speed-gate-d1-sequence': 'tp4-speed-1'})

    def test_the_isolation_jobs_are_soft(self):
        # Their images allocate the scratch at the first prefill, so this checkout's ledger rule fails them whatever the
        # text: a 'stop' there would halt a chain on a verdict the exit status does not carry.
        for name, kind, *_ in window.read_order(self.folder)[:3]:
            self.assertEqual(kind, 'soft', name)

    def test_the_trigger_is_verified_under_the_exactness_policy(self):
        # V2's matrix admits all four users before round 1 and its solo arm serves one at a time, so no prefill there follows
        # a packed replay; the staggered plan's later arrivals prefill after padded packed rounds, judged against solo.
        rows = {row[0]: row for row in window.read_order(self.folder)}
        values, _ = window.parsed('V3-fix-staggered-trigger', self.folder)
        self.assertEqual((values['C2_GATE_PLAN'], values['C2_IMAGE_TAG']), ('staggered', 'tp4-stackfix-3'))
        self.assertEqual(rows['V3-fix-staggered-trigger'][2], 'tp4-stackfix-3')
        self.assertEqual(window.parsed('V2-fix-s3a-matrix-pairs', self.folder)[0]['C2_GATE_PLAN'], 'matrix')
        order = [row[0] for row in window.read_order(self.folder)]
        self.assertLess(order.index('V0b-build-stackfix3'), order.index('V1b-fix-gate-pairs-smoke'))
        self.assertLess(order.index('V1b-fix-gate-pairs-smoke'), order.index('V3-fix-staggered-trigger'))
        self.assertNotIn('exercised by the matrix', window.text_of('V2-fix-s3a-matrix-pairs', self.folder))


if __name__ == '__main__':
    unittest.main()
