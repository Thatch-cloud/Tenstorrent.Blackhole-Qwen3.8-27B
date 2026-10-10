"""QWEN_FAST_DRAFT_HEAD64 (draft_head64_tp, F-F4): the quad's LM head as one 64-row matmul.

Held on the CPU: the head's ops with the flag on are ONE linear on the 64-row block (not two on the halves), the same four TopK of 32 rows by 32,768 columns, the same pads and the same four
concats, the chunk slices cutting their rows as well as their columns (a shape-tracking fake); on a torch fake whose matmul is row independent (exactly representable data) the candidates
equal the served two-half head tensor for tensor - the ops below the matmul see the same bytes iff the matmul's rows at M = 64 equal its rows at M = 32, which no CPU test can say: that is the
audit's question (QWEN_FAST_DRAFT_HEAD64_AUDIT: the eager warm pass compares the served head beside the lever on every chip; a difference makes the lever tau-only and the audit arm its NO-GO
for bit-identity); the refusals and the flag-off path.

Run: `python -m unittest test_draft_head64_tp` from scripts/ci (py 3.11).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import draft_head64_tp as head64
import draft_permute_tp as perm
import draft_shared_head_tp
import quad_draft_tp as twin
import test_quad_draft as base
import tp4_sampdraft
from test_quad_draft_tp4 import ShapeOps4
from tp_test_support import four_cards

HERE = Path(__file__).resolve().parent
VOCAB = 62080


class Quiet(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', head64.FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        perm._LOGGED.clear()
        perm._PASS['eager'] = None
        perm._SERVED['depth'] = 0
        self.lines = []
        logger = patch.object(tp4_sampdraft, 'log_line', side_effect=self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)


def model(devices=4):
    return SimpleNamespace(num_devices=devices, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight='head')


class OpLogTests(Quiet):
    def run_head(self, **flags):
        with four_cards(), patch.dict(os.environ, flags):
            ops, owned = ShapeOps4(4), []
            chunks = twin.head_candidates(ops, model(), base.Tensor((1, 1, 64, 5120)), owned, lambda value: owned.append(value) or value)
        return ops, chunks

    def test_one_64_row_linear_the_served_topk_pads_and_concats_and_row_cutting_chunk_slices(self):
        os.environ.pop(head64.FLAG, None)
        served_ops, served_chunks = self.run_head()
        ops, chunks = self.run_head(**{head64.FLAG: '1'})
        self.assertEqual([event for event in served_ops.events if event[0] == 'linear'], [('linear', (1, 1, 32, 5120))] * 2)
        self.assertEqual([event for event in ops.events if event[0] == 'linear'], [('linear', (1, 1, 64, 5120))])
        for name in ('topk', 'pad', 'concat'):
            self.assertEqual([event for event in ops.events if event[0] == name], [event for event in served_ops.events if event[0] == name], name)
        chunk_slices = [event for event in ops.events if event[0] == 'slice' and event[1] == (1, 1, 64, VOCAB)]
        self.assertEqual([(event[2], event[3]) for event in chunk_slices],
                         [((0, 0, 0, 0), (1, 1, 32, 32768)), ((0, 0, 0, 32768), (1, 1, 32, VOCAB)), ((0, 0, 32, 0), (1, 1, 64, 32768)), ((0, 0, 32, 32768), (1, 1, 64, VOCAB))])
        self.assertNotIn('slice', [event[0] for event in ops.events if event[1] == (1, 1, 64, 5120)], 'no half slice of the normalised block')
        self.assertEqual([(chunk['start'], chunk['stop'], chunk['values'].shape) for chunk in chunks],
                         [(chunk['start'], chunk['stop'], chunk['values'].shape) for chunk in served_chunks])
        engaged = [line for line in self.lines if line.startswith(head64.ENGAGED)]
        self.assertEqual(engaged, ['%s site=quad rows=64 halves=2 chunks=2' % head64.ENGAGED])

    def test_flag_off_the_head_never_imports_the_module_and_runs_the_served_ops(self):
        script = ('import sys, os; sys.path.insert(0, %r); os.environ.pop(%r, None); os.environ["QWEN_FAST_TP"] = "4"; import quad_draft_tp; print("draft_head64_tp" in sys.modules)'
                  % (str(HERE), head64.FLAG))
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=120, cwd=str(HERE))
        if result.returncode != 0:
            self.skipTest('the serving modules do not import here: %s' % result.stderr[-200:])
        self.assertEqual(result.stdout.strip(), 'False')

    def test_a_model_the_block_head_refuses_falls_back_to_the_served_head_which_refuses_it_too(self):
        with four_cards():
            with self.assertRaises(ValueError):
                twin.head_candidates(ShapeOps4(4), model(devices=2), base.Tensor((1, 1, 64, 5120)), [], lambda value: value)
        self.assertTrue(any(line.startswith(head64.FALLBACK) and 'site=quad' in line for line in self.lines))

    def test_the_flag_is_refused_at_the_pair_and_the_audit_needs_the_lever(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ[head64.FLAG] = '1'
            with self.assertRaisesRegex(ValueError, 'TP4 lever'):
                head64.enabled()
        with patch.dict(os.environ, {head64.FLAG: '0', head64.AUDIT_FLAG: '1'}), self.assertRaisesRegex(ValueError, 'needs'):
            head64.audit_enabled()


class TorchOps(object):
    """linear (a float64 matmul on exactly representable data: a row's result cannot depend on the other rows), slice, pad, topk on torch tensors."""

    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def linear(self, left, right):
        return Tensor((left.value.double() @ right.double()).float())

    def slice(self, value, start, end, step=None):
        return Tensor(value.value[tuple(slice(low, high) for low, high in zip(start, end))].clone())

    def pad(self, value, padding, fill):
        out = value.value
        for dim, (low, high) in enumerate(padding):
            if high:
                shape = list(out.shape)
                shape[dim] = high
                out = torch.cat([out, torch.full(shape, fill)], dim=dim)
        return Tensor(out)

    def topk(self, value, k, dim, largest, sorted):
        values, indices = value.value.topk(k, dim=dim, largest=largest, sorted=sorted)
        return Tensor(values), Tensor(indices.to(torch.int32))

    def get_device_tensors(self, tensor):
        return [tensor]

    def to_torch(self, tensor):
        return tensor.value


class Tensor(object):
    def __init__(self, value):
        self.value = value
        self.shape = tuple(value.shape)
        self.dtype, self.layout = 'bf16', 'tile'

    def memory_config(self):
        return 'dram'


class NumericTests(Quiet):
    def setUp(self):
        super().setUp()
        generator = torch.Generator().manual_seed(4)
        self.normalized = Tensor(torch.randint(-3, 4, (1, 1, 64, 96), generator=generator).float())
        self.weight = torch.randint(-3, 4, (96, VOCAB), generator=generator).float()
        self.model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight=self.weight)

    def test_on_a_row_independent_matmul_the_candidates_are_the_served_two_half_heads_tensor_for_tensor(self):
        ops = TorchOps()
        owned = []
        served = []
        for half in range(2):
            block = Tensor(self.normalized.value[:, :, 32 * half:32 * half + 32])
            logits = ops.linear(block, self.weight)
            served.append(draft_shared_head_tp.local_head_candidates(ops, logits, owned))
        # the fused head: one matmul, the chunk slices cut their rows
        logits = ops.linear(self.normalized, self.weight)
        fused = [draft_shared_head_tp.local_head_candidates(ops, logits, owned, row=32 * half, rows=32) for half in range(2)]
        for half in range(2):
            for mine, theirs in zip(fused[half], served[half]):
                self.assertEqual((mine['start'], mine['stop']), (theirs['start'], theirs['stop']))
                self.assertTrue(torch.equal(mine['values'].value, theirs['values'].value), half)
                self.assertTrue(torch.equal(mine['indices'].value, theirs['indices'].value), half)

    def test_a_row_window_is_validated(self):
        ops = TorchOps()
        logits = Tensor(torch.zeros(1, 1, 64, VOCAB))
        for row, rows in ((16, 32), (64, 32), (0, 24), (-32, 32)):
            with self.subTest(row=row, rows=rows), self.assertRaises(ValueError):
                draft_shared_head_tp.local_head_candidates(ops, logits, [], row=row, rows=rows)
        with self.assertRaises(ValueError):
            draft_shared_head_tp.local_head_candidates(ops, Tensor(torch.zeros(1, 1, 64, VOCAB)), [])        # the served default still wants 8, 16 or 32 rows

    def test_the_default_call_is_the_served_one(self):
        ops, owned = TorchOps(), []
        logits = Tensor(torch.randn(1, 1, 32, VOCAB))
        found = draft_shared_head_tp.local_head_candidates(ops, logits, owned)
        self.assertEqual([(chunk['start'], chunk['stop']) for chunk in found], [(0, 32768), (32768, VOCAB)])


class AuditTests(Quiet):
    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {head64.AUDIT_FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def chunks(self, values_seed):
        generator = torch.Generator().manual_seed(values_seed)
        return [dict(start=start, stop=stop, values=Tensor(torch.randn(1, 1, 32, 16, generator=generator)),
                     indices=Tensor(torch.randint(0, 100, (1, 1, 32, 16), generator=generator).to(torch.int32))) for start, stop in ((0, 32768), (32768, VOCAB))]

    def call(self, fused, reference, eager=True):
        calls = []
        ops = TorchOps()
        with patch.object(draft_shared_head_tp, 'block_head_candidates', return_value=fused), four_cards():
            previous = perm.set_pass(eager)
            try:
                result = head64.candidates(ops, model(), None, [], served=lambda: calls.append(1) or reference, site='quad')
            finally:
                perm.set_pass(previous)
        return result, calls

    def test_equal_candidates_log_exact_and_a_difference_raises_the_mismatch_marker(self):
        one = [self.chunks(1), self.chunks(2)]
        result, calls = self.call(one, [list(half) for half in one])
        self.assertIs(result, one)
        self.assertEqual(calls, [1])
        self.assertIn('%s exact=True site=head shape=quad tensors=8' % head64.AUDIT, self.lines)
        other = [self.chunks(1), self.chunks(3)]
        with self.assertRaises(AssertionError):
            self.call(one, other)
        self.assertTrue(self.lines[-1].startswith(head64.MISMATCH))
        self.assertIn("'half1.chunk0.values'", self.lines[-1])

    def test_a_capture_pass_does_not_run_the_served_head(self):
        one = [self.chunks(1), self.chunks(2)]
        result, calls = self.call(one, None, eager=False)
        self.assertIs(result, one)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
