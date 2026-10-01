"""verifier_engine_tp.VerifierEngine.proposal_rows under QWEN_FAST_BUDGET_CAP (tail caps, H3).

The engine is built without a device (object.__new__): proposal_rows reads only the session's budget and
position, the engine's captured widths, its page table and its replay plan.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import trace_census
from verifier_engine_tp import VerifierEngine

PAGES = 2052  # the served geometry: 131328 positions


def engine(remaining, *, widths=(1, 2, 4), position=5000, verifier_rows=16, replay_plan=None):
    live = object.__new__(VerifierEngine)
    live.session = SimpleNamespace(max_new_tokens=remaining + 1, emitted=[0], position=position,
                                   verifier_rows=verifier_rows)
    live.position = position
    live.widths, live.pages, live.replay_plan = tuple(widths), SimpleNamespace(shape=(1, PAGES)), replay_plan
    return live


class BudgetCapProposalRowsTests(unittest.TestCase):
    def setUp(self):
        stack = patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'})
        stack.start()
        self.addCleanup(stack.stop)

    def test_a_request_at_its_last_tokens_drafts_the_engines_widest_width(self):
        for remaining in (1, 2, 3, 4, 40):
            with self.subTest(remaining=remaining):
                self.assertEqual(engine(remaining).proposal_rows(), 4)

    def test_without_the_flag_the_widest_width_that_fits_the_remaining_tokens(self):
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            self.assertEqual([engine(remaining).proposal_rows() for remaining in (3, 2, 1)], [2, 2, 1])
        os.environ.pop('QWEN_FAST_BUDGET_CAP')
        self.assertEqual([engine(remaining).proposal_rows() for remaining in (3, 2, 1)], [2, 2, 1])

    def test_the_scheduler_cannot_offer_four_rows_at_the_end_of_the_context(self):
        # vLLM offers max_model_len - 1 - position rows: four fit through position 131323
        top = PAGES * 64 - 1
        self.assertEqual([engine(1, position=top - rows).proposal_rows() for rows in (4, 3, 2, 1)], [4, 2, 2, 1])

    def test_an_engine_built_without_a_four_row_width_still_narrows(self):
        for widths, expected in (((1, 2), 2), ((1,), 1)):
            self.assertEqual(engine(1, widths=widths).proposal_rows(), expected)

    def test_a_block_round_answers_the_blocks_rows_from_one_token_left(self):
        self.assertEqual(engine(1).proposal_rows(packed_rows=16), 16)
        self.assertEqual(engine(40).proposal_rows(packed_rows=16), 16)
        for rows in (3, 128, '16'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                engine(40).proposal_rows(packed_rows=rows)
        with self.assertRaises(ValueError):
            engine(40, verifier_rows=16).proposal_rows(packed_rows=32)

    def test_a_replay_plan_keeps_its_own_width_selection(self):
        plan = SimpleNamespace(max_rows=lambda position, remaining: 2)
        self.assertEqual(engine(1, replay_plan=plan).proposal_rows(), 2)
        self.assertEqual(engine(1, replay_plan=plan).proposal_rows(packed_rows=16), 16, 'a block round, as without a plan')

    def test_the_inherited_answer_without_the_flag_for_a_block_round(self):
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            self.assertEqual(engine(1).proposal_rows(packed_rows=16), 1)
            self.assertEqual(engine(16).proposal_rows(packed_rows=16), 16)


class Host:
    """The readback's host tensor: reshape(-1)[:n].tolist() is the predicted ids."""

    def reshape(self, *shape):
        return self

    def __getitem__(self, selected):
        return SimpleNamespace(tolist=lambda: [7, 8, 9, 10])


def verifying_engine(*, first, retained):
    """An engine whose every device call is a fake that records its name, so the order of the stage lines is the order
    of the calls they precede."""
    calls = []
    live = object.__new__(VerifierEngine)
    ticket = SimpleNamespace(tokens=[1, 2, 3, 4], position=10)
    live.session = SimpleNamespace(request_id='request-V', check_ticket=lambda request, ticket: None,
                                   fail_verification=lambda request, ticket: None)
    live.phase, live.pending, live.position = 'idle', None, 10
    fixture = SimpleNamespace(retained=SimpleNamespace(replay=lambda operation: calls.append('replay') or operation())
                              if retained else None, replay_reader=None)
    live.buckets = {'key': dict(fixture=fixture, trace='trace', output=(None, 'ids'), first=first)}
    live.bucket_key = lambda ticket: 'key'
    live.validate_bindings = lambda: calls.append('validate')
    live.restore_carry = lambda: calls.append('restore') or True
    live.mesh, live.model = SimpleNamespace(num_program_cache_entries=lambda: 41), SimpleNamespace(
        args=SimpleNamespace(vocab_size=8))
    live.operations = SimpleNamespace(
        execute_trace=lambda mesh, trace, cq_id, blocking: calls.append('execute'),
        synchronize_device=lambda mesh: calls.append('sync'),
        get_device_tensors=lambda tensor: [object()] * 4, to_torch=lambda part: calls.append('readback') or Host())
    return live, ticket, calls


class StageLogTests(unittest.TestCase):
    """QWEN_FAST_SEQ_STAGE_LOG: a 'begin' line before each stage of verify and around publish; nothing without the flag."""

    def verify(self, *, first, retained, flag='1'):
        live, ticket, calls = verifying_engine(first=first, retained=retained)
        lines = []
        environ = {'QWEN_FAST_TP': '4', **({'QWEN_FAST_SEQ_STAGE_LOG': flag} if flag else {})}
        with patch.dict(os.environ, environ), patch.object(trace_census, 'log', lines.append), \
                patch('verifier_engine_tp.stage_inputs', lambda *args: calls.append('stage_inputs')):
            if not flag:
                os.environ.pop('QWEN_FAST_SEQ_STAGE_LOG', None)
            predictions, timings = live.verify(ticket)
        return predictions, lines, calls

    def test_the_first_replay_path_logs_the_six_stages_in_order_and_one_census_line(self):
        predictions, lines, calls = self.verify(first=True, retained=False)
        self.assertEqual(predictions, [7, 8, 9, 10])
        stages = [line.split('stage=')[1] for line in lines if line.startswith('[SEQ-STAGE]')]
        self.assertEqual(stages, [name + ' begin' for name in
                                  ('validate', 'restore', 'stage_inputs', 'execute_trace', 'sync', 'readback')])
        self.assertTrue(all(line.startswith('[SEQ-STAGE] request=request-V rows=4 first=1 stage=') for line in lines
                            if line.startswith('[SEQ-STAGE]')))
        census = [line for line in lines if line.startswith('[PINDIAG] first replay')]
        self.assertEqual(census, ['[PINDIAG] first replay request=request-V rows=4 built_after_packed_round=n/a '
                                  'packed_rounds_since_build=n/a program_cache=41'])
        self.assertEqual(calls, ['validate', 'restore', 'stage_inputs', 'execute', 'sync', 'readback'])

    def test_the_retained_replay_path_logs_the_same_stages_and_no_census_line(self):
        _, lines, calls = self.verify(first=False, retained=True)
        stages = [line.split('stage=')[1].split()[0] for line in lines]
        self.assertEqual(stages, ['validate', 'restore', 'stage_inputs', 'execute_trace', 'sync', 'readback'])
        self.assertTrue(all(' first=0 ' in line for line in lines))
        self.assertEqual(calls, ['validate', 'restore', 'stage_inputs', 'replay', 'execute', 'readback'])

    def test_the_census_line_counts_the_packed_steps_since_the_engine_was_built(self):
        live, ticket, calls = verifying_engine(first=True, retained=False)
        lines = []
        with patch.dict(os.environ, {'QWEN_FAST_SEQ_STAGE_LOG': '1', 'QWEN_FAST_TP': '4'}), patch.object(trace_census, 'log', lines.append), \
                patch.object(trace_census, 'PACKED_STEPS', 9), \
                patch.dict(trace_census.BUILT_AT, {id(live): 5}), \
                patch('verifier_engine_tp.stage_inputs', lambda *args: None):
            live.verify(ticket)
        self.assertIn('built_after_packed_round=5 packed_rounds_since_build=4 program_cache=41',
                      [line for line in lines if 'first replay' in line][0])

    def test_without_the_flag_nothing_is_logged_and_the_calls_are_the_same(self):
        _, lines, calls = self.verify(first=True, retained=False, flag=None)
        self.assertEqual(lines, [])
        self.assertEqual(calls, ['validate', 'restore', 'stage_inputs', 'execute', 'sync', 'readback'])

    def test_publish_logs_begin_and_end_around_the_inherited_body_and_failed_on_a_raise(self):
        for refuses in (False, True):
            with self.subTest(refuses=refuses):
                live = object.__new__(VerifierEngine)
                live.session, live.pending = SimpleNamespace(request_id='request-V'), SimpleNamespace(tokens=[0] * 4)
                lines = []
                with patch.dict(os.environ, {'QWEN_FAST_SEQ_STAGE_LOG': '1'}), patch.object(trace_census, 'log', lines.append):
                    if refuses:
                        # phase is not 'verified': the inherited publish refuses
                        live.phase = 'idle'
                        with self.assertRaises(ValueError):
                            live.publish(1)
                    else:
                        with patch('verifier_engine.VerifierEngine.publish', lambda self, prefix: None):
                            live.publish(1)
                self.assertEqual([line.split('stage=')[1] for line in lines],
                                 ['publish begin', 'publish failed' if refuses else 'publish end'])


if __name__ == '__main__':
    unittest.main()
