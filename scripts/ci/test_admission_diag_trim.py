"""QWEN_FAST_ADMISSION_DIAG_TRIM (F2 of the four-card freeze plan): the diagnostics of a burst of concurrent arrivals.

When four requests arrive together each admission rebuilds its engine inside the prefill gate and nobody decodes,
so diagnostics that cost host time on every admission are paid by every waiting user. Under the flag (set only by
the traffic-profile twins c2-packed-tp4-f2 and -f12, never by a gate profile) the per-admission ledger points are
taken for the first prefill and the first engine only, the conv-slice readback of the GDN slot adoption is taken at
each slot's first adoption only, and the 'dram after engine' line is off. Unset, every call is what it always was.

    py -3.11 -B -m unittest test_admission_diag_trim      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

import gdn_snapshot
import memory_ledger
import serving_request_factory
import test_serving_extent_memory   # module imports, so their TestCases are not collected here again
import test_serving_runtime

TRIM = {'QWEN_FAST_ADMISSION_DIAG_TRIM': '1'}


def environ(on, **extra):
    """The process environment with the trim set or absent, the ledger's first-admission state and the adoption's
    verified slots cleared before and after."""
    base = {name: value for name, value in os.environ.items() if name != memory_ledger.TRIM_FLAG}
    if on:
        base.update(TRIM)
    base.update(extra)
    return patch.dict(os.environ, base, clear=True)


class Fresh(unittest.TestCase):
    def setUp(self):
        memory_ledger.reset_admission_diag()
        serving_request_factory._ADOPT_VERIFIED.clear()
        self.addCleanup(memory_ledger.reset_admission_diag)
        self.addCleanup(serving_request_factory._ADOPT_VERIFIED.clear)


class FlagTests(Fresh):
    def test_the_flag_is_strict_and_off_by_default(self):
        self.assertIs(memory_ledger.trim_enabled({}), False)
        for value in ('0', '', 'true', '2', ' 1'):
            self.assertIs(memory_ledger.trim_enabled({memory_ledger.TRIM_FLAG: value}), False, value)
        self.assertIs(memory_ledger.trim_enabled(TRIM), True)

    def test_every_kind_is_taken_every_time_without_the_flag(self):
        for _ in range(5):
            for kind in ('prefill_before', 'prefill_after', 'before_engine', 'engine'):
                self.assertIs(memory_ledger.admission_diag(kind, {}), True)

    def test_each_kind_is_taken_once_under_the_flag(self):
        for kind in ('prefill_before', 'prefill_after', 'before_engine', 'engine'):
            self.assertIs(memory_ledger.admission_diag(kind, TRIM), True, kind)
        for _ in range(3):
            for kind in ('prefill_before', 'prefill_after', 'before_engine', 'engine'):
                self.assertIs(memory_ledger.admission_diag(kind, TRIM), False, kind)
        memory_ledger.reset_admission_diag()
        self.assertIs(memory_ledger.admission_diag('engine', TRIM), True)


class LedgerSiteTests(Fresh):
    """The attach's prefill and engine points, over four admissions (test_serving_runtime's harness)."""

    def admissions(self, on, count=4):
        calls, diagnostics = [], []

        def record(phase, point=None, request=None, **walked):
            calls.append((phase, point))

        def admitted(request_id, **walked):
            calls.append(('engine', request_id))

        def probe(install, diagnostic):
            capture_factory = install.call_args.kwargs['capture_factory']
            bridge_factory = install.call_args.kwargs['bridge_factory']
            test_serving_runtime.serving_runtime.ServingCacheOwner.return_value.physical_pages = 100
            for index in range(count):
                capture_factory(4096)
                state = SimpleNamespace(req_id='request-%d' % index, block_ids=([3, 4],))
                runtime = test_serving_runtime.serving_runtime
                with patch.object(runtime, 'from_prefill', return_value=SimpleNamespace(engine=object(), close=Mock())), \
                        patch.object(runtime, 'VerifierPageBinding'), \
                        patch.object(runtime, 'FastRunnerBridge', return_value='bridge'):
                    bridge_factory(state, 'capture')
            diagnostics.extend(call.args[0] for call in diagnostic.call_args_list)

        with environ(on), patch.object(memory_ledger, 'record', side_effect=record), \
                patch.object(memory_ledger, 'engine_admitted', side_effect=admitted), \
                patch('dflash_prefill_window.PrefillWindowCapture'):
            test_serving_runtime.RuntimeAttachmentTests().exercise(packed=True, users=4, four_as_two=False, probe=probe)
        per_admission = [item for item in calls if item[0] in ('prefill', 'engine')]
        return per_admission, [line for line in diagnostics if line.startswith('[PINDIAG] dram after engine')]

    def test_without_the_flag_every_admission_is_walked_and_logged(self):
        points, lines = self.admissions(False)
        self.assertEqual(len([item for item in points if item[0] == 'prefill' and item[1].startswith('before')]), 4)
        self.assertEqual(len([item for item in points if item[0] == 'prefill' and item[1].startswith('after')]), 4)
        self.assertEqual([item for item in points if item[0] == 'engine'], [('engine', 'request-%d' % i) for i in range(4)])
        self.assertEqual(len(lines), 4)

    def test_the_trim_keeps_the_first_prefill_and_the_first_engine_and_drops_the_line(self):
        points, lines = self.admissions(True)
        self.assertEqual(points, [('prefill', 'before prompt=4096'), ('prefill', 'after req=request-0'),
                                  ('engine', 'request-0')])
        self.assertEqual(lines, [], 'the dram after engine line is off under the trim')


class EngineBeforePointTests(Fresh):
    """serving_request_factory.from_prefill's W6d point: once per process under the trim, every engine without it."""

    def builds(self, on, count=3):
        extent = test_serving_extent_memory.ExtentMemoryTests()
        points = []
        with environ(on, QWEN_FAST_EXTENT_REPLAY='1'), \
                patch.object(memory_ledger, 'before', side_effect=lambda *args, **kwargs: points.append(kwargs['point'])):
            for index in range(count):
                _, _, _, build = extent.short_request()
                build().close('request')
        return points

    def test_every_engine_has_its_before_point_without_the_flag(self):
        self.assertEqual(self.builds(False), ['req=request'] * 3)

    def test_only_the_first_engine_has_one_under_the_flag(self):
        self.assertEqual(self.builds(True), ['req=request'])


class SlotAdoptionTests(Fresh):
    def adopt(self, on, slots):
        helpers = [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2}) for _ in range(3)]
        with environ(on), patch.object(serving_request_factory, '_log'):
            for slot in slots:
                serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=slot), 'request')
        return helpers

    def test_without_the_flag_every_adoption_is_called_as_it_always_was(self):
        helpers = self.adopt(False, [1, 1, 2, 1])
        for layer, helper in enumerate(helpers):
            self.assertEqual([item for item in helper.adopt_slot.call_args_list],
                             [((slot,), dict(layer=layer)) for slot in (1, 1, 2, 1)], layer)

    def test_the_readback_is_taken_at_each_slots_first_adoption_only(self):
        helpers = self.adopt(True, [1, 1, 2, 1, 2, 3])
        for layer, helper in enumerate(helpers):
            first, again = dict(layer=layer), dict(layer=layer, readback=False)
            self.assertEqual([tuple(item) for item in helper.adopt_slot.call_args_list],
                             [((1,), first), ((1,), again), ((2,), first), ((1,), again), ((2,), again), ((3,), first)],
                             layer)

    def test_slot_zero_adopts_nothing_and_marks_nothing(self):
        helpers = self.adopt(True, [0, None])
        for helper in helpers:
            helper.adopt_slot.assert_not_called()
        self.assertEqual(serving_request_factory._ADOPT_VERIFIED, set())

    def test_a_refused_first_adoption_does_not_count_as_verified(self):
        helpers = [Mock(spec=['adopt_slot']) for _ in range(2)]
        helpers[0].adopt_slot.side_effect = ValueError('differs from the row')
        helpers[1].adopt_slot.return_value = 2
        with environ(True), patch.object(serving_request_factory, '_log'):
            with self.assertRaises(ValueError):
                serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=1), 'request')
            self.assertEqual(serving_request_factory._ADOPT_VERIFIED, set())
            helpers[0].adopt_slot.side_effect = None
            helpers[0].adopt_slot.return_value = 2
            serving_request_factory.adopt_prefill_slot(helpers, SimpleNamespace(prefill_slot=1), 'request')
        self.assertEqual(helpers[0].adopt_slot.call_args_list[-1], ((1,), dict(layer=0)),
                         'the retry still reads back: the failed attempt verified nothing')


class ReadbackTests(unittest.TestCase):
    """gdn_snapshot.ActiveSnapshot.adopt_slot(readback=False) on real per-chip torch shards."""

    def fixture(self, wrong_shape=False):
        reads = []

        def make(shape, dimension):
            rows = [torch.full([1 if axis == dimension else size for axis, size in enumerate(shape)], float(row),
                               dtype=torch.bfloat16) for row in range(8)]
            return SimpleNamespace(chips=[torch.cat(rows, dim=dimension) for _ in range(2)])

        live = {'rec': make((8, 2, 2, 2), 0), 'conv': make((1, 8, 4), 1)}

        def slice_along(tensor, dimension, start, stop):
            parts = [shard.narrow(dimension, start, stop - start).clone() for shard in tensor.chips]
            if wrong_shape and tensor is live['conv']:
                parts = [part[..., :2].clone() for part in parts]
            return SimpleNamespace(chips=parts)

        def to_torch(shard):
            reads.append(tuple(shard.shape))
            return shard

        layer = SimpleNamespace(B=8, _stable_state=True, rec_state=live['rec'], conv_states=[live['conv']],
                                _slice_along=Mock(side_effect=slice_along), _write_recurrent_state_prefix=Mock(),
                                _write_index=Mock())
        operations = SimpleNamespace(clone=Mock(side_effect=lambda source, **kwargs: source), copy=Mock(),
                                     deallocate=Mock(), get_device_tensors=lambda value: value.chips, to_torch=to_torch,
                                     DRAM_MEMORY_CONFIG='DRAM')
        return layer, reads, gdn_snapshot.ActiveSnapshot(layer, operations)

    def test_the_default_reads_the_conv_slices_back_and_the_trimmed_form_does_not(self):
        layer, reads, snapshots = self.fixture()
        self.assertEqual(snapshots.adopt_slot(3), 2)
        self.assertEqual(len(reads), 4, 'two chips, the live conv tensor and its slice each')
        layer, reads, snapshots = self.fixture()
        self.assertEqual(snapshots.adopt_slot(3, readback=False), 2)
        self.assertEqual(reads, [])
        layer._write_recurrent_state_prefix.assert_called_once()
        layer._write_index.assert_called_once()

    def test_the_trimmed_form_keeps_the_metadata_checks(self):
        layer, reads, snapshots = self.fixture(wrong_shape=True)
        with self.assertRaisesRegex(ValueError, 'differs from the row'):
            snapshots.adopt_slot(3, readback=False)
        self.assertEqual(reads, [])
        layer._write_index.assert_not_called()


if __name__ == '__main__':
    unittest.main()
