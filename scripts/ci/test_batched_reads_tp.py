"""Op-fusion programme, host-gap package WPH, lever 3: the verify and collect read-backs as a few reads (QWEN_FAST_BATCHED_READS=1|async, audit twin
QWEN_FAST_BATCHED_READS_AUDIT; batched_reads_tp.py, hooked in packed_verifier.shard_predictions and quad_draft_tp.read_quad_outputs; default off, host only).

Over a fake four-chip mesh that counts what the host asks of it, pinned here:
  - the flag is 0, 1 or async (anything else fails the attach), the audit needs the lever;
  - the verify readback and the quad collect hand the downstream code the SAME per-chip tensors (shape, dtype, bits) as the per-shard blocking reads: composed (one
    blocking read a tensor: 8 become 2 and 36 become 9) and asynchronous (one non-blocking copy a tensor into host tensors allocated once, one fence), with the replicated-
    feature guard on or sampled out, and the READ merge of round_host unchanged on top;
  - the pieces of a composed read do not alias one another;
  - a tensor whose composed size is not four equal shards, a binding that is missing, an exception: the served read, a `fell back` line, the lever latched off;
  - the audit twin reads the served way beside the batched one (the first reads, then every 16th), logs exact=True, and on a difference returns the SERVED bytes and
    latches the lever off;
  - with the flag off the hooks read exactly as before: the same calls in the same order, and batched_reads_tp is not imported."""

import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import batched_reads_tp as reads
import packed_verifier
import round_host
import test_quad_draft_tp4 as tq
import verify_prestage
from tp_test_support import four_cards

HERE = Path(__file__).resolve().parent
FLAG, AUDIT = 'QWEN_FAST_BATCHED_READS', 'QWEN_FAST_BATCHED_READS_AUDIT'
CHIPS = 4


class Mesh:
    """A mesh tensor: its shards, one tensor a chip."""

    def __init__(self, chips, layout='row_major', dtype='int32'):
        self.chips = [chip.clone() for chip in chips]
        self.shape, self.dtype, self.layout = tuple(self.chips[0].shape), dtype, layout


class Shard:
    def __init__(self, values):
        self.values = values


class Host:
    def __init__(self, shape, dtype, layout):
        self.shape, self.dtype, self.layout = shape, dtype, layout
        self.chips = None


class MeshOps:
    """The ttnn calls the readbacks use, counting the blocking reads, the non-blocking copies and the fences."""

    def __init__(self):
        self.single_reads = self.composed_reads = self.copies = self.fences = self.host_reads = 0
        self.order = []
        self.allocated = []

    def get_device_tensors(self, tensor):
        return [Shard(chip) for chip in tensor.chips]

    def to_torch(self, value, mesh_composer=None):
        if isinstance(value, Shard):
            self.single_reads += 1
            self.order.append('single')
            return value.values.clone()
        if isinstance(value, Host):
            self.host_reads += 1
            return torch.cat(value.chips, dim=mesh_composer[1])
        self.composed_reads += 1
        self.order.append('composed')
        return torch.cat([chip.clone() for chip in value.chips], dim=mesh_composer[1])

    def ConcatMeshToTensor(self, mesh, dim=0):
        return ('composer', dim)

    def allocate_tensor_on_host(self, shape, dtype, layout, mesh):
        host = Host(tuple(shape), dtype, layout)
        self.allocated.append(host)
        return host

    def copy_device_to_host_tensor(self, tensor, host, blocking=True, cq_id=None):
        assert blocking is False
        self.copies += 1
        host.chips = [chip.clone() for chip in tensor.chips]

    def synchronize_device(self, mesh):
        self.fences += 1


def verify_outputs(rows=64):
    generator = torch.Generator().manual_seed(7)
    ids = Mesh([torch.randint(0, 62080, (1, 1, 1, rows), generator=generator).to(torch.int32) for _ in range(CHIPS)], dtype='uint32')
    values = Mesh([torch.randn(1, 1, 1, rows, generator=generator).bfloat16() for _ in range(CHIPS)], dtype='bf16')
    return ids, values


def mesh_quad(quad):
    """tq's four-chip quad outputs with every tensor a Mesh (shape, dtype and layout, for the asynchronous mode)."""
    chunks = [dict(chunk, values=Mesh(chunk['values'].chips, dtype='bf16'), indices=Mesh(chunk['indices'].chips, dtype='int32')) for chunk in quad.chunks]
    return SimpleNamespace(chunks=chunks, projected=Mesh(quad.projected.chips, layout='tile', dtype='bf16'))


class Clean(unittest.TestCase):
    def setUp(self):
        reads.reset()
        self.addCleanup(reads.reset)
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        logger = patch.object(verify_prestage, 'log_line', side_effect=self.log.append)
        logger.start()
        self.addCleanup(logger.stop)
        round_host.refresh({})
        self.addCleanup(round_host.refresh, {})
        context = four_cards()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def lines(self, marker):
        return [line for line in self.log if line.startswith(marker)]


class FlagTests(Clean):
    def test_the_flag_is_zero_one_or_async_and_the_audit_needs_the_lever(self):
        self.assertEqual([reads.mode({FLAG: value}) for value in ('0', '1', 'async')], [None, 'compose', 'async'])
        self.assertIsNone(reads.mode({}))
        self.assertFalse(reads.enabled({}))
        for value in ('yes', '2', 'compose', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reads.mode({FLAG: value})
        self.assertFalse(reads.audit_enabled({AUDIT: '1'}))
        self.assertTrue(reads.audit_enabled({FLAG: 'async', AUDIT: '1'}))
        with self.assertRaises(ValueError):
            reads.audit_enabled({FLAG: '1', AUDIT: 'x'})


class VerifyReadTests(Clean):
    def served(self, ids, values, rows=64):
        ops = MeshOps()
        return ([ops.to_torch(part).reshape(-1)[:rows] for part in ops.get_device_tensors(ids)],
                [ops.to_torch(part).reshape(-1)[:rows] for part in ops.get_device_tensors(values)]), ops

    def same(self, left, right):
        return len(left) == len(right) and all(a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b)) for a, b in zip(left, right))

    def test_composed_two_reads_for_eight_and_the_same_per_chip_tensors(self):
        ids, values = verify_outputs()
        (want_ids, want_values), served_ops = self.served(ids, values)
        self.assertEqual(served_ops.single_reads, 8)
        os.environ[FLAG] = '1'
        ops = MeshOps()
        got_ids, got_values = reads.verify_reads(ops, 'mesh', ids, values, 64)
        self.assertTrue(self.same(got_ids, want_ids) and self.same(got_values, want_values))
        self.assertEqual((ops.composed_reads, ops.single_reads, ops.copies, ops.fences), (2, 0, 0, 0))
        self.assertEqual(len(self.lines(reads.ENGAGED_MARKER)), 1)
        self.assertRegex(self.lines(reads.ENGAGED_MARKER)[0], r'site=verify mode=compose tensors=2 chips=4$')
        reads.verify_reads(ops, 'mesh', ids, values, 64)
        self.assertEqual(len(self.lines(reads.ENGAGED_MARKER)), 1, 'engaged once a site')

    def test_asynchronous_two_copies_and_one_fence_and_the_same_bytes(self):
        ids, values = verify_outputs()
        (want_ids, want_values), unused = self.served(ids, values)
        os.environ[FLAG] = 'async'
        ops = MeshOps()
        got_ids, got_values = reads.verify_reads(ops, 'mesh', ids, values, 64)
        self.assertTrue(self.same(got_ids, want_ids) and self.same(got_values, want_values))
        self.assertEqual((ops.copies, ops.fences, ops.composed_reads, ops.single_reads, ops.host_reads), (2, 1, 0, 0, 2))
        self.assertEqual(len(ops.allocated), 2)
        reads.verify_reads(ops, 'mesh', ids, values, 64)
        self.assertEqual((len(ops.allocated), ops.copies, ops.fences), (2, 4, 2), 'the host tensors are allocated once a tensor')

    def test_the_pieces_of_a_composed_read_do_not_alias_each_other(self):
        ids, values = verify_outputs()
        os.environ[FLAG] = '1'
        got_ids, unused = reads.verify_reads(MeshOps(), 'mesh', ids, values, 64)
        before = got_ids[1].clone()
        got_ids[0].fill_(5)
        self.assertTrue(torch.equal(got_ids[1], before))

    def test_a_tensor_that_is_not_four_equal_shards_is_read_the_served_way_and_latches_the_lever(self):
        ids, values = verify_outputs()
        values.chips = values.chips[:3]               # a mesh tensor of three shards
        os.environ[FLAG] = '1'
        ops = MeshOps()
        with patch.object(reads, 'served', wraps=reads.served) as served:
            got = reads.read(ops, 'mesh', [ids, values], site='verify')
        self.assertEqual(len(got[1]), 3, 'the served read hands back what the shards are')
        self.assertEqual(reads.STATE.off, 'verify:ValueError')
        self.assertEqual(len(self.lines(reads.FELL_BACK_MARKER)), 1)
        self.assertEqual(served.call_count, 1)
        ops2 = MeshOps()
        reads.read(ops2, 'mesh', [ids, values], site='verify')
        self.assertEqual((ops2.composed_reads, ops2.single_reads), (0, 7), 'once latched, every read is served')

    def test_a_missing_binding_or_an_exception_is_the_served_read_and_a_latch(self):
        class WithoutComposer(MeshOps):
            ConcatMeshToTensor = None

        class WithoutHostAllocation(MeshOps):
            allocate_tensor_on_host = None

        ids, values = verify_outputs()
        for mode, ops_class in (('1', WithoutComposer), ('async', WithoutHostAllocation)):
            with self.subTest(mode=mode):
                reads.reset()
                self.log.clear()
                os.environ[FLAG] = mode
                ops = ops_class()
                got = reads.read(ops, 'mesh', [ids, values], site='verify')
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(got[0], ids.chips)))
                self.assertTrue(reads.STATE.off.startswith('verify:'), reads.STATE.off)
                self.assertEqual(len(self.lines(reads.FELL_BACK_MARKER)), 1)
                self.assertEqual(ops.single_reads, 8)

    def test_off_the_hook_reads_per_shard_and_the_module_is_not_imported(self):
        ids, values = verify_outputs()
        sys.modules.pop('batched_reads_tp', None)
        try:
            ops = MeshOps()
            engine = SimpleNamespace(output=(None, ids, values), operations=ops, mesh='mesh', block_rows=64, shard_audit=False)
            with patch.object(packed_verifier, 'audit_sampdraft'), patch.object(packed_verifier, 'audit_shard_values'):
                host = packed_verifier.PackedVerifierEngine.shard_predictions(engine)
            self.assertEqual((ops.single_reads, ops.composed_reads), (8, 0))
            self.assertEqual(ops.order, ['single'] * 8)
            self.assertNotIn('batched_reads_tp', sys.modules)
            self.assertEqual(len(host), 64)
            self.assertEqual(len(engine.readback_split), 2)
        finally:
            sys.modules['batched_reads_tp'] = reads

    def test_the_hook_with_the_flag_returns_the_same_predictions(self):
        ids, values = verify_outputs()
        ids = Mesh([torch.randint(0, 62080, (1, 1, 1, 64)).to(torch.int32) for _ in range(CHIPS)], dtype='uint32')
        results = {}
        for mode in ('0', '1', 'async'):
            os.environ[FLAG] = mode
            reads.reset()
            ops = MeshOps()
            engine = SimpleNamespace(output=(None, ids, values), operations=ops, mesh='mesh', block_rows=64, shard_audit=False)
            with patch.object(packed_verifier, 'audit_sampdraft'), patch.object(packed_verifier, 'audit_shard_values'):
                results[mode] = packed_verifier.PackedVerifierEngine.shard_predictions(engine)
        self.assertEqual(results['1'], results['0'])
        self.assertEqual(results['async'], results['0'])


class AuditTests(Clean):
    def test_the_audit_compares_the_first_reads_and_every_sixteenth_and_logs_exact(self):
        ids, values = verify_outputs()
        os.environ.update({FLAG: '1', AUDIT: '1'})
        ops = MeshOps()
        for _ in range(40):
            reads.verify_reads(ops, 'mesh', ids, values, 64)
        audited = self.lines(reads.AUDIT_MARKER)
        self.assertEqual(len(audited), 16 + 1, 'reads 1-16, then the 32nd')
        for line in audited:
            self.assertRegex(line, r'site=verify mode=compose reads=\d+ tensors=2 chips=4 exact=True$')
        self.assertEqual(ops.single_reads, 17 * 8)

    def test_a_difference_returns_the_served_bytes_and_latches_the_lever(self):
        ids, values = verify_outputs()
        os.environ.update({FLAG: '1', AUDIT: '1'})
        ops = MeshOps()
        real = ops.to_torch

        def corrupting(value, mesh_composer=None):
            result = real(value, mesh_composer=mesh_composer)
            return result + 1 if mesh_composer is not None else result

        ops.to_torch = corrupting
        got_ids, got_values = reads.verify_reads(ops, 'mesh', ids, values, 64)
        self.assertTrue(all(torch.equal(a, b.reshape(-1)[:64]) for a, b in zip(got_ids, ids.chips)), 'the served bytes')
        mismatches = self.lines(reads.MISMATCH_MARKER)
        self.assertEqual(len(mismatches), 1)
        self.assertIn('exact=False', mismatches[0])
        self.assertEqual(reads.STATE.off, 'audit_mismatch')

    def test_a_negative_zero_is_a_difference_to_the_audit(self):
        plus = torch.tensor([0.0], dtype=torch.bfloat16)
        self.assertFalse(reads.same_tensor_bits(plus, -plus))


class QuadReadTests(Clean):
    def quad(self):
        generator = torch.Generator().manual_seed(21)
        return mesh_quad(tq.four_chip_users(generator)[1])

    def read(self, quad, ops, **kwargs):
        import quad_draft_tp

        return quad_draft_tp.read_quad_outputs(SimpleNamespace(operations=ops, mesh='mesh'), quad, **kwargs)

    def equal(self, left, right):
        for user in range(4):
            for key in ('hidden', 'candidates', 'unary'):
                a, b = left[user][key], right[user][key]
                if not (a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))):
                    return False
        return True

    def test_composed_nine_reads_for_thirty_six_and_the_same_selection_inputs(self):
        quad = self.quad()
        chunks = len(quad.chunks)
        ops = MeshOps()
        want = self.read(quad, ops)
        served_reads = ops.single_reads
        self.assertEqual(served_reads, chunks * 2 * CHIPS + CHIPS)
        for mode, expected in (('1', dict(composed_reads=2 * chunks + 1, single_reads=0, copies=0, fences=0)),
                               ('async', dict(composed_reads=0, single_reads=0, copies=2 * chunks + 1, fences=1))):
            with self.subTest(mode=mode):
                reads.reset()
                os.environ[FLAG] = mode
                ops = MeshOps()
                got = self.read(quad, ops)
                self.assertTrue(self.equal(got, want))
                self.assertEqual({key: getattr(ops, key) for key in expected}, expected)

    def test_the_guard_sampled_out_reads_chip_zero_alone_and_still_the_same_bytes(self):
        quad = self.quad()
        want = self.read(quad, MeshOps())
        for mode in ('0', '1', 'async'):
            with self.subTest(mode=mode):
                reads.reset()
                os.environ.update({FLAG: mode, 'QWEN_FAST_TP4_ROUND_HOST_READ': '1'})
                round_host.refresh(dict(os.environ))
                round_host.reset_counters()
                round_host._GUARD['reads'] = round_host.GUARD_FIRST
                got = self.read(quad, MeshOps())
                self.assertTrue(self.equal(got, want))
                self.assertEqual(round_host.COUNTS['guard_skipped'], 1)
                self.assertEqual(round_host.COUNTS['read_fast'], 2)

    def test_a_replicated_feature_that_differs_is_still_refused_with_the_flag(self):
        quad = self.quad()
        quad.projected.chips[3][0, 0, 5, 3] += 1
        os.environ[FLAG] = '1'
        with self.assertRaises(AssertionError) as raised:
            self.read(quad, MeshOps())
        self.assertEqual(str(raised.exception), 'Replicated learned selector features differ')

    def test_the_reference_selection_is_always_the_served_read(self):
        quad = self.quad()
        os.environ[FLAG] = '1'
        ops = MeshOps()
        self.read(quad, ops, reference=True)
        self.assertEqual((ops.composed_reads, ops.single_reads), (0, len(quad.chunks) * 2 * CHIPS + CHIPS))

    def test_off_the_calls_are_the_served_ones_in_the_served_order(self):
        quad = self.quad()
        ops = MeshOps()
        self.read(quad, ops)
        self.assertEqual(ops.order, ['single'] * (len(quad.chunks) * 2 * CHIPS + CHIPS))

    def test_the_audit_on_the_collect_logs_exact(self):
        quad = self.quad()
        os.environ.update({FLAG: '1', AUDIT: '1'})
        self.read(quad, MeshOps())
        audited = self.lines(reads.AUDIT_MARKER)
        self.assertEqual(len(audited), 1)
        self.assertRegex(audited[0], r'site=collect mode=compose reads=1 tensors=%d chips=4 exact=True$' % (2 * len(quad.chunks) + 1))


class ShippingTests(unittest.TestCase):
    def test_the_hooks_import_the_module_only_with_the_flag(self):
        for name in ('packed_verifier.py', 'quad_draft_tp.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertIn("os.environ.get('QWEN_FAST_BATCHED_READS', '0') != '0'", text, name)
            self.assertNotRegex(text, r'(?m)^import batched_reads_tp', name)

    def test_the_names_are_the_ones_the_manifest_reads(self):
        import json

        manifest = json.loads((HERE / 'fusion-wp' / 'WPH.json').read_text(encoding='utf-8'))
        lever = next(item for item in manifest['levers'] if item['id'] == 'reads')
        self.assertEqual((lever['flag'], lever['audit_flag'], lever['marker']), (reads.FLAG, reads.AUDIT_FLAG, 'tp4 batched reads'))
        self.assertTrue(reads.ENGAGED_MARKER.startswith('[PINDIAG] ' + lever['marker']))


if __name__ == '__main__':
    unittest.main()
