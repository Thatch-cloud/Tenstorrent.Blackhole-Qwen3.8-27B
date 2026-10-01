"""QWEN_FAST_REQUEST_SHARD_ARGMAX / QWEN_FAST_REQUEST_SHARD_AUDIT (verifier_engine_tp.VerifierEngine at four cards).

The request engine's rows=1/2/4 verify traces pick tokens with the packed block's per-chip shard argmax instead of the pinned
sampler. The engine is built without a device: ttnn is a fake that computes each chip's argmax and max from a real torch logits
row, so the combined ids are compared with what the pinned sampler (a whole-row torch.argmax) returns on the same logits.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import verifier_engine_tp
import verify_trace_t1 as t1
from verifier_engine_tp import VerifierEngine
import verifier_engine as pair_engine

CHIPS, WIDTH = 4, 62080
VOCABULARY = CHIPS * WIDTH
ENVIRON = {'QWEN_FAST_TP': '4'}
ARM = dict(ENVIRON, QWEN_FAST_REQUEST_SHARD_ARGMAX='1')
AUDITED = dict(ARM, QWEN_FAST_REQUEST_SHARD_AUDIT='1')


def greedy_sampler(**changes):
    options = dict(force_argmax_sampling=True, vocab_size=VOCABULARY, padded_vocab_size=VOCABULARY)
    state = dict(_penalties_active=False, _log_probs_active=False, seed=False)
    for key, value in changes.items():
        (options if key in options else state)[key] = value
    return SimpleNamespace(tt_sampling=SimpleNamespace(**options), _penalties_active=state['_penalties_active'],
                           _log_probs_active=state['_log_probs_active'],
                           seed_manager=SimpleNamespace(has_active_request_seed=lambda: state['seed']))


def logits_with_ties(rows):
    """Rows whose maximum is tied across shards (the earliest shard must win), plus ordinary rows."""
    values = torch.zeros(rows, VOCABULARY)
    for row in range(rows):
        values[row, 5 + row] = 1.0
        values[row, 7] = -1.0
    values[0, WIDTH + 11] = 1.0 + 5  # row 0: a later shard strictly greater
    values[0, 5] = 1.0
    if rows > 1:
        values[1, 2 * WIDTH + 6] = values[1, 6]  # row 1: a tie across shards 0 and 2 keeps shard 0
    if rows > 2:
        values[2, 3 * WIDTH + 1] = 9.0  # the last shard wins alone
    return values.bfloat16()


class Tensor:
    def __init__(self, name, parts=None, shape=None):
        self.name, self.parts, self.shape = name, parts, shape
        self.dtype, self.layout = 'bf16', 'tile'


class FakeOperations:
    """ttnn: every op logs its name; argmax and max compute each chip's own answer from the logits row."""
    bfloat16, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'row_major', 'dram'

    def __init__(self):
        self.ops = []
        self.freed = []

    def to_layout(self, value, layout, memory_config=None):
        self.ops.append('to_layout')
        return Tensor('row-major', shape=value.shape, parts=value.parts)

    def argmax(self, value, **options):
        self.ops.append('argmax')
        return Tensor('ids', parts=[part.float().argmax(dim=-1).to(torch.int32) for part in value.parts])

    def max(self, value, **options):
        self.ops.append('max')
        return Tensor('values', parts=[part.amax(dim=-1) for part in value.parts])

    def deallocate(self, value):
        self.freed.append(value.name)

    def get_device_tensors(self, value):
        self.ops.append('get_device_tensors')
        return value.parts

    def to_torch(self, part):
        self.ops.append('to_torch')
        return part


def logits_tensor(full, rows):
    tensor = Tensor('logits', parts=[full[:, chip * WIDTH:(chip + 1) * WIDTH] for chip in range(CHIPS)],
                    shape=(1, 1, rows, WIDTH))
    tensor.full = full
    return tensor


class EngineCase(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, ENVIRON)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = []
        for target, value in (('log_line', self.lines.append),):
            patcher = patch.object(t1, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        verifier_engine_tp._LOGGED.clear()
        self.addCleanup(verifier_engine_tp._LOGGED.clear)
        self.sampler_calls = []
        for module in (verifier_engine_tp, pair_engine):
            patcher = patch.object(module, 'sample_rows', side_effect=self.pinned)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.pinned_override = None

    def pinned(self, sampler, logits, rows, operations, *, native_rows):
        """The pinned sampler: an AllGather, untilize and a whole-row first-occurrence argmax (torch.argmax)."""
        self.sampler_calls.append(rows)
        operations.ops.extend(['all_gather', 'untilize', 'sampler_argmax'])
        ids = logits.full.float().argmax(dim=-1) if self.pinned_override is None else self.pinned_override
        return Tensor('pinned', parts=[ids.to(torch.int32)] * CHIPS)

    def engine(self, environ, rows=4, sampler=None, build=True):
        operations = FakeOperations()
        with patch.object(pair_engine.VerifierEngine, '__init__', Mock(return_value=None)), \
                patch.dict(os.environ, environ, clear=True):
            live = VerifierEngine(None, sampler=greedy_sampler() if sampler is None else sampler)
        live.operations, live.sampler, live.native_sampling_rows = operations, greedy_sampler(), False
        return live

    def operation(self, live, environ, rows=4, full=None):
        full = logits_with_ties(rows) if full is None else full
        logits = logits_tensor(full, rows)
        fixture = SimpleNamespace(rows=rows, run=lambda sharded_logits: logits)
        with patch.dict(os.environ, environ, clear=True):
            return full, live.operation(fixture)

    def verify_ids(self, live, output, rows):
        with patch.dict(os.environ, ENVIRON, clear=True):
            return live.shard_predictions(output, rows) if len(output) > 2 else None


class FlagTests(EngineCase):
    def test_the_arm_is_off_unless_exactly_one(self):
        for value, expected in (('1', True), ('0', False), ('', False), ('true', False)):
            with self.subTest(value=value):
                self.assertEqual(verifier_engine_tp.shard_arm_enabled({'QWEN_FAST_REQUEST_SHARD_ARGMAX': value}), expected)
        self.assertFalse(verifier_engine_tp.shard_arm_enabled({}))

    def test_the_audit_needs_the_arm_and_its_name_ends_in_audit(self):
        self.assertTrue(verifier_engine_tp.SHARD_AUDIT_FLAG.endswith('_AUDIT'))
        self.assertFalse(verifier_engine_tp.shard_audit_enabled({'QWEN_FAST_REQUEST_SHARD_AUDIT': '1'}))
        self.assertTrue(verifier_engine_tp.shard_audit_enabled(AUDITED))
        self.assertFalse(verifier_engine_tp.shard_audit_enabled(ARM))

    def test_it_is_read_at_construction(self):
        self.assertFalse(self.engine(ENVIRON).request_shard)
        live = self.engine(ARM)
        self.assertTrue(live.request_shard and not live.request_shard_audit)
        live = self.engine(AUDITED)
        self.assertTrue(live.request_shard and live.request_shard_audit)

    def test_the_pairs_engine_file_is_untouched_by_the_flag(self):
        with open(pair_engine.__file__, encoding='utf-8') as handle:
            source = handle.read()
        self.assertNotIn('REQUEST_SHARD', source)


class FlagOffTests(EngineCase):
    def test_the_pinned_sampler_runs_and_nothing_else_changes(self):
        live = self.engine(ENVIRON)
        with patch.object(pair_engine.VerifierEngine, 'operation', autospec=True,
                          return_value=('logits', 'ids')) as inherited:
            fixture = SimpleNamespace(rows=4)
            self.assertEqual(live.operation(fixture), ('logits', 'ids'))
        inherited.assert_called_once()
        self.assertEqual(self.lines, [])

    def test_the_inherited_operation_samples_with_the_pinned_sampler(self):
        live = self.engine(ENVIRON)
        full, output = self.operation(live, ENVIRON)
        self.assertEqual(len(output), 2)
        self.assertEqual(self.sampler_calls, [4])
        self.assertEqual(self.lines, [])


class ShardArgmaxTests(EngineCase):
    def test_no_pinned_sampler_op_is_recorded_and_the_ids_equal_the_pinned_ones_including_cross_shard_ties(self):
        for rows in (1, 2, 4):
            with self.subTest(rows=rows):
                live = self.engine(ARM)
                full, output = self.operation(live, ARM, rows=rows)
                self.assertEqual(len(output), 3)
                self.assertEqual(self.sampler_calls, [])
                self.assertEqual(live.operations.ops, ['to_layout', 'argmax', 'max'])
                self.assertFalse({'all_gather', 'untilize', 'sampler_argmax'} & set(live.operations.ops))
                served = live.shard_predictions(output, rows)
                self.assertEqual(served, full.float().argmax(dim=-1).tolist())
                self.assertTrue(all(type(value) is int for value in served))

    def test_the_tied_rows_keep_the_earliest_shard(self):
        live = self.engine(ARM)
        full, output = self.operation(live, ARM, rows=4)
        served = live.shard_predictions(output, 4)
        self.assertEqual(served[0], WIDTH + 11)      # a later shard strictly greater
        self.assertEqual(served[1], 6)               # tied with shard 2: shard 0 wins
        self.assertEqual(served[2], 3 * WIDTH + 1)   # the last shard alone

    def test_one_marker_per_width_when_it_engages(self):
        live = self.engine(ARM)
        for rows in (1, 2, 4, 4):
            self.operation(live, ARM, rows=rows)
        markers = [line for line in self.lines if line.startswith(verifier_engine_tp.ENGAGED_MARKER)]
        self.assertEqual(markers, ['[PINDIAG] request shard argmax engaged rows=%d audit=0' % rows for rows in (1, 2, 4)])

    def test_the_wrong_number_of_chips_is_refused(self):
        live = self.engine(ARM)
        _, output = self.operation(live, ARM, rows=4)
        output[1].parts = output[1].parts[:2]
        with patch.dict(os.environ, ENVIRON, clear=True), self.assertRaises(AssertionError):
            live.shard_predictions(output, 4)

    def test_every_output_is_released_by_the_engines_close_shape(self):
        # close() frees every non-None member of bucket['output']: the shard tuple's members are all device tensors
        live = self.engine(AUDITED)
        _, output = self.operation(live, AUDITED, rows=2)
        self.assertEqual([value.name for value in output], ['logits', 'ids', 'values', 'pinned'])


class FallbackTests(EngineCase):
    def test_an_ineligible_sampler_keeps_the_pinned_sampler_and_says_why_once(self):
        for changes in (dict(seed=True), dict(_penalties_active=True), dict(_log_probs_active=True),
                        dict(force_argmax_sampling=False), dict(vocab_size=VOCABULARY + 64)):
            with self.subTest(changes=changes):
                self.lines.clear()
                verifier_engine_tp._LOGGED.clear()
                live = self.engine(ARM, sampler=greedy_sampler(**changes))
                self.assertFalse(live.request_shard)
                self.assertIsNotNone(live.request_shard_problem)
                self.engine(ARM, sampler=greedy_sampler(**changes))
                self.assertEqual(len(self.lines), 1)
                self.assertTrue(self.lines[0].startswith(verifier_engine_tp.FALLBACK_MARKER + ': '))
                self.assertIn(live.request_shard_problem, self.lines[0])

    def test_no_sampler_is_a_fallback(self):
        with patch.object(pair_engine.VerifierEngine, '__init__', Mock(return_value=None)), \
                patch.dict(os.environ, ARM, clear=True):
            live = VerifierEngine(None)
        self.assertFalse(live.request_shard)
        self.assertIn('no device sampler', live.request_shard_problem)

    def test_the_ineligible_engine_runs_the_inherited_operation(self):
        live = self.engine(ARM, sampler=greedy_sampler(seed=True))
        with patch.object(pair_engine.VerifierEngine, 'operation', autospec=True, return_value=('logits', 'ids')) as inherited:
            self.assertEqual(live.operation(SimpleNamespace(rows=4)), ('logits', 'ids'))
        inherited.assert_called_once()

    def test_logits_that_are_not_bf16_tile_fall_back_per_trace_and_say_why_once(self):
        live = self.engine(ARM)
        for dtype, layout in (('bf8_b', 'tile'), ('bf16', 'row_major')):
            with self.subTest(dtype=dtype, layout=layout):
                self.lines.clear()
                verifier_engine_tp._LOGGED.clear()
                full = logits_with_ties(4)
                logits = logits_tensor(full, 4)
                logits.dtype, logits.layout = dtype, layout
                fixture = SimpleNamespace(rows=4, run=lambda sharded_logits: logits)
                with patch.dict(os.environ, ARM, clear=True):
                    output = live.operation(fixture)
                    live.operation(fixture)
                self.assertEqual(len(output), 2)
                self.assertEqual(len(self.lines), 1)
                self.assertIn('not bf16 TILE', self.lines[0])

    def test_the_value_audit_beside_the_gathered_maxima_keeps_the_pinned_sampler(self):
        live = self.engine(ARM)
        environ = dict(ARM, QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1')
        with patch('tp4_vglue.audit_enabled', return_value=True), patch('tp4_vglue.enabled', return_value=True):
            _, output = self.operation(live, environ)
        self.assertEqual(len(output), 2)
        self.assertIn('VGLUE_AUDIT', self.lines[0])


class AuditTests(EngineCase):
    def test_both_run_and_the_pinned_ids_are_served(self):
        live = self.engine(AUDITED)
        full, output = self.operation(live, AUDITED, rows=4)
        self.assertEqual(self.sampler_calls, [4])
        self.assertEqual(len(output), 4)
        self.assertEqual(live.operations.ops, ['to_layout', 'argmax', 'max', 'all_gather', 'untilize', 'sampler_argmax'])
        served = live.shard_predictions(output, 4)
        self.assertEqual(served, full.float().argmax(dim=-1).tolist())
        self.assertFalse([line for line in self.lines if verifier_engine_tp.AUDIT_MISMATCH in line])
        self.assertTrue([line for line in self.lines if line.startswith('[PINDIAG] request shard argmax audit exact=True rows=4')])
        self.assertTrue([line for line in self.lines if line == '[PINDIAG] request shard argmax engaged rows=4 audit=1'])

    def test_a_mismatch_is_logged_and_the_pinned_sampler_still_serves(self):
        live = self.engine(AUDITED)
        self.pinned_override = torch.tensor([9, 6, 3 * WIDTH + 1, 0])
        _, output = self.operation(live, AUDITED, rows=4)
        served = live.shard_predictions(output, 4)
        self.assertEqual(served, [9, 6, 3 * WIDTH + 1, 0])
        mismatches = [line for line in self.lines if line.startswith(verifier_engine_tp.AUDIT_MISMATCH)]
        self.assertEqual(len(mismatches), 1)
        self.assertIn('differing=[0, 3]', mismatches[0])

    def test_a_fold_that_cannot_run_is_logged_and_the_pinned_ids_are_served(self):
        live = self.engine(AUDITED)
        full, output = self.operation(live, AUDITED, rows=4)
        with patch.object(verifier_engine_tp.verify_trace_t1, 'combine_shards', side_effect=ValueError('bad shard id')):
            served = live.shard_predictions(output, 4)
        self.assertEqual(served, full.float().argmax(dim=-1).tolist())
        self.assertTrue([line for line in self.lines if line.startswith(verifier_engine_tp.AUDIT_MISMATCH) and 'combine failed' in line])

    def test_a_failing_pinned_sampler_frees_the_shard_outputs_and_the_logits(self):
        live = self.engine(AUDITED)
        # the four-card process rebinds release_owned to tp_addresses' twin; the fake frees each tensor once
        with patch.object(verifier_engine_tp, 'sample_rows', side_effect=RuntimeError('sampler')),                 patch.object(verifier_engine_tp, 'release_owned',
                             lambda operations, tensors: [operations.deallocate(value) for value in tensors]):
            with self.assertRaises(RuntimeError):
                self.operation(live, AUDITED, rows=4)
        self.assertEqual(sorted(live.operations.freed), ['ids', 'logits', 'row-major', 'values'])


class VerifyReadbackTests(EngineCase):
    """verify hands GreedySession the same list of ints whichever tuple the bucket holds."""

    def verifying(self, live, output, rows):
        ticket = SimpleNamespace(tokens=list(range(rows)), position=10)
        live.session = SimpleNamespace(request_id='r', check_ticket=lambda request, ticket: None,
                                       fail_verification=lambda request, ticket: None)
        live.phase, live.pending, live.position = 'idle', None, 10
        fixture = SimpleNamespace(retained=None, replay_reader=None)
        live.buckets = {'key': dict(fixture=fixture, trace='trace', output=output, first=True)}
        live.bucket_key = lambda ticket: 'key'
        live.validate_bindings = lambda: None
        live.restore_carry = lambda: True
        live.mesh, live.model = 'mesh', SimpleNamespace(args=SimpleNamespace(vocab_size=VOCABULARY))
        live.operations.execute_trace = lambda mesh, trace, cq_id, blocking: None
        live.operations.synchronize_device = lambda mesh: None
        with patch('verifier_engine_tp.stage_inputs', lambda *args: None), patch.dict(os.environ, ENVIRON, clear=True):
            return live.verify(ticket)

    def test_the_arm_and_the_pinned_path_return_the_same_predictions(self):
        rows = 4
        pinned_live = self.engine(ENVIRON)
        full, pinned_output = self.operation(pinned_live, ENVIRON, rows=rows)
        shard_live = self.engine(ARM)
        _, shard_output = self.operation(shard_live, ARM, rows=rows)
        expected, _ = self.verifying(pinned_live, pinned_output, rows)
        got, metrics = self.verifying(shard_live, shard_output, rows)
        self.assertEqual(got, expected)
        self.assertEqual(got, full.float().argmax(dim=-1).tolist())
        self.assertIn('output_readback_host_ms', metrics)

    def test_an_audited_bucket_serves_the_pinned_ids(self):
        live = self.engine(AUDITED)
        self.pinned_override = torch.tensor([1, 2, 3, 4])
        _, output = self.operation(live, AUDITED, rows=4)
        predictions, _ = self.verifying(live, output, 4)
        self.assertEqual(predictions, [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
