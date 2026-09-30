"""The four-card garbage/EOS defect (v172 D1): every prefill after the first packed replay came out wrong.

The model's persistent B=1 GDN prefill scratch was allocated by the FIRST prefill (serving runs trace_mode=decode_only, so
warmup_model_prefill returns early), after the attach-time packed traces existed. serving_runtime.prefill_scratch_before_traces
now allocates it before the packed blocks are built at four cards; the pair (QWEN_FAST_TP unset or 2) keeps its order. The smoke
check now judges the text (instant EOS, foreign script) and the first prefill's ledger line.
"""
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import c2_smoke_check as check
import serving_runtime
import test_tp4_draft_window as window

HERE = Path(__file__).resolve().parent
LEDGER = ('(EngineCore pid=66) 2026-09-30 14:15:09.482 | INFO     | memory_ledger:log_line:121 - [MEMLEDGER] phase=prefill '
          'point=after req=4b3-b36e8145 item=model_after_prefill chip0=%s chip1=0.081GB chip2=0.081GB chip3=0.081GB buffers=%d '
          'largest=0.4MB unreadable=0')


class Model:
    def __init__(self):
        self.calls = 0

    def _ensure_gdn_prefill_scratch(self):
        self.calls += 1


class ScratchBeforeTraces(unittest.TestCase):
    def test_four_cards_allocates_once_before_the_blocks(self):
        model = Model()
        with patch.object(serving_runtime, 'pindiag'):
            self.assertTrue(serving_runtime.prefill_scratch_before_traces(model, {'QWEN_FAST_TP': '4'}))
        self.assertEqual(model.calls, 1)

    def test_the_pair_is_unchanged(self):
        for environ in ({}, {'QWEN_FAST_TP': '2'}):
            model = Model()
            self.assertFalse(serving_runtime.prefill_scratch_before_traces(model, environ))
            self.assertEqual(model.calls, 0)

    def test_a_model_without_the_method_is_left_alone(self):
        self.assertFalse(serving_runtime.prefill_scratch_before_traces(SimpleNamespace(), {'QWEN_FAST_TP': '4'}))

    def test_the_attach_calls_it_before_the_first_block_is_built(self):
        source = (HERE / 'serving_runtime.py').read_text(encoding='utf-8')
        call = source.index('prefill_scratch_before_traces(model)')
        self.assertLess(call, source.index('packed_block = PackedVerifierEngine('))
        self.assertLess(source.index('packed_blocks, slot = [], 0') - 1, source.index('packed_block = PackedVerifierEngine('))


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


class FirstPrefillLedger(unittest.TestCase):
    smoke = 'SMOKE_JSON ' + '{"warmup": {"value": 200}}'

    def run_check(self, log, tp):
        return check.check(self.smoke, log, False, env={'QWEN_FAST_TP': tp})

    def test_late_allocation_fails_at_four_cards(self):
        problems, facts = self.run_check(LEDGER % ('0.081GB', 1484), '4')
        self.assertEqual(facts['first_prefill_model_buffers'], 1484)
        self.assertTrue(any('first prefill' in p for p in problems), problems)

    def test_a_clean_first_prefill_passes(self):
        problems, facts = self.run_check(LEDGER % ('0.000GB', 0), '4')
        self.assertEqual(facts['first_prefill_model_buffers'], 0)
        self.assertEqual(problems, [])

    def test_the_pair_is_not_judged(self):
        problems, facts = self.run_check(LEDGER % ('0.081GB', 1484), '2')
        self.assertNotIn('first_prefill_model_buffers', facts)
        self.assertEqual(problems, [])


class Templates(unittest.TestCase):
    folder = str(HERE / 'references' / 'tp4-garbage-jobs')

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


if __name__ == '__main__':
    unittest.main()
