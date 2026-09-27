"""S2 B6: the extent block's attach-time eager publication warm (publication_warm.py).

What is held here, on CPU:
  - the plan: every segment's row offset (0, 16, 32 and 48 in M3) x prefixes 1..16, then each pooled capture
    width x 1..width, in that order; the line's fields;
  - the warm runs today's code, not a copy: DFlashDevice.prepare_publication then discard_publication once for
    every shape of the plan, in order, on one scratch drafter - the packed ones through dflash_traced_publish.
    install_publish_options with the round's QWEN_FAST_PIPELINED_PUBLISH and QWEN_FAST_TRACED_PUBLISH (nothing
    installed with both off), the sequential ones through nothing - on a torch fake running the real
    DFlashDevice, DraftKVHistory and dflash_traced_publish code, B1 on and off;
  - in the served configuration (the qualified slide candidate built by draft_kv_slide_adapter from the real
    draft_kv_history source, B1, the pipelined and traced publication): ten slides per publication at
    history_rows 2048 and the shape's prefix, into scratch;
  - scratch only: no live tensor (a live drafter's history, spare and banks, the block's taps, a pool slot's
    history, banks and query) is handed to any operation, every write lands in a buffer the warm allocated,
    and every live tensor is bit for bit what it was; nothing is left pending or installed;
  - every scratch buffer released exactly once, after a final fence; on a failure released without a fence
    after it, the failure propagating and no warmed line;
  - skipped: without the extent block (nothing allocated, nothing logged), and with a logged reason when the
    block has no collectives or no prepared draft weights;
  - the block: warm() called once under the extent pool, at its own stage, after every capture, with the
    block's pool, weights, collectives, device and log; its summary in describe(); a raise fails the attach
    closed (the extent storage handed back, the failure logged at the stage); never called without the extent
    pool;
  - shipping: the C2 overlay manifest, both P8 copy lists and the C2 closure's S2 set name the module; no
    qualification pins it; the CPU suite runs this file; LF.
"""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
import draft_kv_history
from draft_kv_history import DraftKVHistory
import draft_kv_slide
import packed_verifier
from packed_shapes import m1_shape, m3_shape
import publication_warm
import test_packed_extent_block as extent_block
import test_packed_verifier as base
from test_publish_prewarm import build_device, project_key_value, real_publication_code, torch_operations

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WIDTH = 2052
POSITION = publication_warm.POSITION
SERVED = dict(QWEN_FAST_ROUND_B1='1', QWEN_FAST_PIPELINED_PUBLISH='1', QWEN_FAST_TRACED_PUBLISH='1')


def environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(flags)
    return patch.dict(os.environ, environ, clear=True)


def packed_plan(offsets=(0, 16, 32, 48), rows=64, prefixes=16):
    return [(offset, rows, prefix) for offset in offsets for prefix in range(1, prefixes + 1)]


def sequential_plan(widths=(1, 2, 4)):
    return [(0, width, prefix) for width in widths for prefix in range(1, width + 1)]


def tensors_in(value):
    """Every torch tensor in an argument (lists, tuples and dicts opened)."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors_in(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors_in(item)]
    return []


def storage(value):
    return value.untyped_storage().data_ptr()


class Recorder:
    """The torch fake of test_publish_prewarm, with what the warm also calls (a feature-axis shard mapper, the
    drafter's kernel config), the storage of every tensor each operation is handed and of every write's
    destination (addresses only: holding the tensors would keep every temporary alive), and every fence and
    release on one event list."""

    NAMES = ('slice', 'pad', 'concat', 'copy', 'matmul', 'typecast', 'rms_norm', 'zeros_like')

    def __init__(self):
        operations = torch_operations()
        self.events, self.inputs, self.written = [], set(), set()
        for name in self.NAMES:
            setattr(operations, name, self.recording(name, getattr(operations, name)))
        operations.ShardTensorToMesh = lambda mesh, dim: ('shard', dim)
        operations.ReplicateTensorToMesh = lambda mesh: ('replicate',)
        operations.MathFidelity = SimpleNamespace(HiFi4='hifi4')
        operations.WormholeComputeKernelConfig = lambda **fields: ('kernel-config',) + tuple(sorted(fields.items()))

        def from_torch(value, *, mesh_mapper=None, **keywords):
            # A feature-axis shard is one chip's half of the columns, as project_features reads it.
            if mesh_mapper == ('shard', 3):
                return value[..., :value.shape[-1] // 2].clone()
            return value.clone()

        operations.from_torch = from_torch
        operations.synchronize_device = Mock(side_effect=lambda mesh: self.events.append(('sync',)))
        operations.deallocate = Mock(side_effect=lambda value: self.events.append(('free', id(value))))
        self.operations = operations

    def recording(self, name, function):
        def call(*args, **keywords):
            self.inputs.update(storage(value) for value in tensors_in(args) + tensors_in(keywords))
            if name == 'copy':
                self.written.add(storage(args[1]))
            return function(*args, **keywords)
        return call


def shared_weights():
    projection, norm = torch.tensor(0.5), torch.tensor(1.0, dtype=torch.bfloat16)
    return SimpleNamespace(closed=False, layers=[(dict(layer=layer), 'mlp', 'weights', 'convolution') for layer in range(5)],
                           projection=projection, feature_norm=norm, tensors=[projection, norm])


def live_slot():
    """A pool slot as serving_buffer_pool lends it: a history pair, five layers of K/V banks and a query."""
    generator = torch.Generator().manual_seed(11)
    randn = lambda *shape: torch.randn(shape, generator=generator).bfloat16()
    return SimpleNamespace(history=randn(1, 1, 2048, 5120), spare_history=randn(1, 1, 2048, 5120),
                           kv=[{side: {head: randn(1, 4, 2048, 128) for head in ('k', 'v')} for side in ('active', 'spare')}
                               for layer in range(5)], query=randn(1, 1, 32, 2048))


def live_block(shape=None, extent=True):
    shape = shape or m3_shape(WIDTH)
    generator = torch.Generator().manual_seed(7)
    taps = tuple(torch.randn((1, 1, shape.block_rows, 2560), generator=generator).bfloat16() for tap in range(5))
    return SimpleNamespace(extent=extent, shape=shape, users=shape.users, rows_per_user=shape.rows_per_user,
                           block_rows=shape.block_rows, taps=taps, segment_slots=())


def slot_tensors(slot):
    return [slot.history, slot.spare_history, slot.query,
            *(bank[side][head] for bank in slot.kv for side in ('active', 'spare') for head in ('k', 'v'))]


def bits(value):
    return value.contiguous().view(torch.int16).clone()


def torch_transport(calls):
    """draft_kv_slide.prepare on torch: the spare becomes rows drop..history_rows of the active bank, then the
    accepted rows of the delta, then zeros (draft_kv_slide.row_source's map)."""
    def prepare(mesh, active, delta, spare, *, history_rows, prefix):
        shape = draft_kv_slide.geometry(history_rows, prefix)

        def run():
            calls.append((id(active), id(delta), id(spare), history_rows, prefix))
            combined = torch.cat([active[:, :, shape['drop']:history_rows], delta[:, :, :prefix]], dim=2)
            spare.zero_()
            spare[:, :, :shape['rows']] = combined
        return run
    return prepare


def served_candidate(transport):
    """The class-level DraftKVHistory.prepare draft_kv_slide_scope.scoped_publication installs, built the same
    way (draft_kv_slide_adapter.build_prepare over the real draft_kv_history source) and marked as it marks it,
    so dflash_traced_publish recognises it and installs the steady slide fusion."""
    from draft_kv_slide_adapter import build_prepare

    candidate, source = build_prepare((HERE / 'draft_kv_history.py').read_text(encoding='utf-8'),
                                      vars(draft_kv_history), transport)

    def prepare(cache, *args, **keywords):
        return candidate(cache, *args, **keywords)

    prepare.__module__ = 'draft_kv_slide_scope'
    prepare.__qualname__ = 'scoped_publication.<locals>.prepare'
    prepare._draft_kv_slide = True
    return prepare


@contextmanager
def publication_code(slides=None):
    """The real publication code on torch (test_publish_prewarm.real_publication_code, plus the projection the
    traced fusion imports itself); with `slides` a list, the qualified slide candidate is live and its
    transport records into it."""
    with ExitStack() as stack:
        stack.enter_context(real_publication_code())
        stack.enter_context(patch('draft_kv_projection.project_key_value', side_effect=project_key_value))
        stack.enter_context(patch('dflash_packed_proposal._ROUND_B1_NOTED', []))
        if slides is not None:
            transport = torch_transport(slides)
            stack.enter_context(patch.object(draft_kv_slide, 'prepare', transport))
            stack.enter_context(patch.object(DraftKVHistory, 'prepare', served_candidate(transport)))
        yield


def recording_scratch():
    """Every buffer publication_warm.Scratch allocates, in order."""
    allocated = []
    original = publication_warm.Scratch.zeros

    def zeros(self, shape, **keywords):
        value = original(self, shape, **keywords)
        allocated.append(value)
        return value
    return allocated, patch.object(publication_warm.Scratch, 'zeros', zeros)


def warm(recorder, block, pool, *, weights=None, collectives='ccl', mesh='mesh', log=None):
    lines = []
    summary = publication_warm.warm(block, operations=recorder.operations, mesh=mesh, pool=pool,
                                    shared_weights=shared_weights() if weights is None else weights,
                                    collectives=collectives, log=lines.append if log is None else log)
    return summary, lines


# ------------------------------------------------------------------------------------------------------
# The plan and the line
# ------------------------------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    def test_every_segment_offset_by_every_prefix_then_every_pooled_width_by_its_prefixes(self):
        shapes = publication_warm.plan(live_block(), SimpleNamespace(bucket_rows=(1, 2, 4, 1)))
        self.assertEqual(len(shapes), 71)
        self.assertEqual([(shape.offset, shape.rows, shape.prefix) for shape in shapes if shape.path == 'packed'],
                         packed_plan())
        self.assertEqual([(shape.offset, shape.rows, shape.prefix) for shape in shapes if shape.path == 'sequential'],
                         sequential_plan())
        self.assertEqual([shape.path for shape in shapes], ['packed'] * 64 + ['sequential'] * 7)
        self.assertEqual(publication_warm.describe_plan(shapes), ('0,16,32,48:1-16', '1:1,2:1-2,4:1-4'))

    def test_the_offsets_are_the_blocks_segments_and_a_pool_without_widths_adds_none(self):
        shapes = publication_warm.plan(live_block(m1_shape(WIDTH)), SimpleNamespace())
        self.assertEqual([(shape.offset, shape.rows, shape.prefix) for shape in shapes], packed_plan((0, 16), 32))
        self.assertEqual(publication_warm.describe_plan(shapes), ('0,16:1-16', 'none'))
        self.assertEqual(publication_warm.runs([1, 3, 4, 7]), '1,3-4,7')

    def test_the_line(self):
        summary = dict(shapes=71, ms=12.25, packed='0,16,32,48:1-16', sequential='1:1,2:1-2,4:1-4', merge_release=True,
                       fused_steady_state=False, program_cache=('n/a', 'n/a'))
        self.assertEqual(publication_warm.warmed_line(summary),
                         '[PINDIAG] eager publication warmed: 71 shapes in 12.2 ms packed=0,16,32,48:1-16 '
                         'sequential=1:1,2:1-2,4:1-4 merge_release=1 fused_steady_state=0 program_cache=n/a->n/a')


# ------------------------------------------------------------------------------------------------------
# Today's publication, every shape, on scratch
# ------------------------------------------------------------------------------------------------------

class EnumerationTests(unittest.TestCase):
    def publications(self, flags, slides=None):
        recorder, calls, discards = Recorder(), [], []
        prepare, discard = DFlashDevice.prepare_publication, DFlashDevice.discard_publication

        def recording(device, features, prefix, *, position, merge_release=False, fused_steady_state=False):
            calls.append((device, getattr(features, 'row_offset', 0), tuple(features[0].shape)[2], prefix, position,
                          merge_release, fused_steady_state, 'prepare_publication' in vars(device),
                          'prepare' in vars(device.kv_history)))
            return prepare(device, features, prefix, position=position, merge_release=merge_release,
                           fused_steady_state=fused_steady_state)

        def discarding(device, publication):
            discards.append((device, publication.prefix, publication.status))
            return discard(device, publication)

        allocated, scratch = recording_scratch()
        with environment(**flags), publication_code(slides), scratch, \
                patch.object(DFlashDevice, 'prepare_publication', recording), \
                patch.object(DFlashDevice, 'discard_publication', discarding):
            summary, lines = warm(recorder, live_block(), SimpleNamespace(bucket_rows=(1, 2, 4)))
        return recorder, calls, discards, allocated, summary, lines

    def test_every_shape_is_todays_prepare_then_discard_through_todays_installers(self):
        for flags in (SERVED, dict(SERVED, QWEN_FAST_ROUND_B1='0'), dict(QWEN_FAST_ROUND_B1='1'),
                      dict(QWEN_FAST_TRACED_PUBLISH='1'), dict(QWEN_FAST_PIPELINED_PUBLISH='1')):
            with self.subTest(flags=flags):
                recorder, calls, discards, allocated, summary, lines = self.publications(flags)
                merge = flags.get('QWEN_FAST_PIPELINED_PUBLISH') == '1'
                fused = flags.get('QWEN_FAST_TRACED_PUBLISH') == '1'
                expected = ([(offset, rows, prefix, POSITION, merge, fused, merge or fused, fused)
                             for offset, rows, prefix in packed_plan()]
                            + [(offset, rows, prefix, POSITION, False, False, False, False)
                               for offset, rows, prefix in sequential_plan()])
                self.assertEqual([call[1:] for call in calls], expected)
                drafter = calls[0][0]
                self.assertTrue(all(call[0] is drafter for call in calls), 'one scratch drafter')
                self.assertEqual([(device, prefix, status) for device, prefix, status in discards],
                                 [(drafter, call[3], 'prepared') for call in calls])
                self.assertIsNone(drafter.pending)
                self.assertIsNone(drafter.kv_history.pending)
                self.assertEqual((drafter.position, drafter.history_rows), (POSITION, 2048))
                self.assertNotIn('prepare_publication', vars(drafter))
                self.assertNotIn('prepare', vars(drafter.kv_history))
                self.assertEqual((summary['shapes'], summary['merge_release'], summary['fused_steady_state']),
                                 (71, merge, fused))
                self.assertEqual(len(lines), 1)
                self.assertRegex(lines[0], r'^\[PINDIAG\] eager publication warmed: 71 shapes in [0-9]+\.[0-9] ms '
                                           r'packed=0,16,32,48:1-16 sequential=1:1,2:1-2,4:1-4 merge_release=%d '
                                           r'fused_steady_state=%d program_cache=n/a->n/a$' % (merge, fused))

    def test_the_drafter_is_built_as_serving_builds_it_over_the_shared_weights(self):
        recorder, calls, discards, allocated, summary, lines = self.publications(SERVED)
        drafter = calls[0][0]
        self.assertIsInstance(drafter, DFlashDevice)
        self.assertIsInstance(drafter.kv_history, DraftKVHistory)
        # C7 holds (a K/V cache, no reporter, a captured proposal): the packed publications skip the history write.
        from dflash_device import history_unread

        self.assertTrue(history_unread(drafter))
        self.assertEqual(drafter.kernel, recorder.operations.WormholeComputeKernelConfig(
            math_fidelity='hifi4', math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
        self.assertEqual([parameter['layer'] for parameter in drafter.kv_history.parameters], [0, 1, 2, 3, 4])
        # Scratch: the query, a bank pair, the history pair, five 64-row taps and five taps per pooled width.
        self.assertEqual([tuple(value.shape) for value in allocated],
                         [(1, 1, 32, 2048), (1, 4, 2048, 128), (1, 4, 2048, 128), (1, 1, 2048, 5120), (1, 1, 2048, 5120)]
                         + [(1, 1, 64, 2560)] * 5 + [(1, 1, 1, 2560)] * 5 + [(1, 1, 2, 2560)] * 5
                         + [(1, 1, 4, 2560)] * 5)
        scratch = {id(value) for value in allocated}
        self.assertTrue({id(drafter.history), id(drafter.spare_history), id(drafter.kv_history.query)} <= scratch)
        self.assertTrue(all(id(bank[head]) in scratch for side in (drafter.kv_history.active, drafter.kv_history.spare)
                            for bank in side for head in ('k', 'v')))

    def test_the_served_configuration_slides_ten_banks_per_publication_into_scratch(self):
        slides = []
        recorder, calls, discards, allocated, summary, lines = self.publications(SERVED, slides=slides)
        self.assertEqual(len(calls), 71)
        self.assertEqual(len(slides), 71 * 10)
        drafter = calls[0][0]
        active, spare = drafter.kv_history.active[0]['k'], drafter.kv_history.spare[0]['k']
        prefixes = [prefix for offset, rows, prefix in packed_plan() + sequential_plan()]
        self.assertEqual([(entry[0], entry[2], entry[3], entry[4]) for entry in slides],
                         [(id(active), id(spare), 2048, prefix) for prefix in prefixes for bank in range(10)])
        self.assertTrue(all(call[8] for call in calls[:64]), 'the packed K/V publication is the steady slide fusion')
        self.assertFalse(any(call[8] for call in calls[64:]), 'the sequential one is the class candidate')


class ScratchOnlyTests(unittest.TestCase):
    def test_no_live_tensor_is_read_or_written_and_every_write_lands_in_scratch(self):
        slides = []
        recorder = Recorder()
        block, slot = live_block(), live_slot()
        pool = SimpleNamespace(bucket_rows=(1, 2, 4), slots=[slot])
        allocated, scratch = recording_scratch()
        with environment(**SERVED), publication_code(slides), scratch:
            device = build_device(recorder.operations, 3)
            live = list(block.taps) + slot_tensors(slot) + [device.history, device.spare_history] + [
                bank[head] for side in (device.kv_history.active, device.kv_history.spare) for bank in side
                for head in ('k', 'v')]
            before = [bits(value) for value in live]
            frontier = (device.position, device.history_rows, device.kv_history.position, device.kv_history.history_rows)
            recorder.inputs.clear()
            recorder.written.clear()
            summary, lines = warm(recorder, block, pool)
        self.assertEqual(summary['shapes'], 71)
        # Every live tensor stays allocated throughout, so no other tensor can have held its storage.
        self.assertEqual(recorder.inputs & {storage(value) for value in live}, set(),
                         'a live tensor was handed to an operation')
        self.assertTrue(recorder.written)
        self.assertEqual(recorder.written - {storage(value) for value in allocated}, set(),
                         'a write landed outside the scratch')
        scratch_ids = {id(value) for value in allocated}
        self.assertEqual(len(slides), 71 * 10)
        self.assertTrue(all(entry[0] in scratch_ids and entry[2] in scratch_ids for entry in slides),
                        'every slide reads a scratch bank and writes a scratch spare')
        for index, (value, old) in enumerate(zip(live, before)):
            self.assertTrue(torch.equal(bits(value), old), 'live tensor %d changed' % index)
        self.assertIsNone(device.pending)
        self.assertIsNone(device.kv_history.pending)
        self.assertEqual((device.position, device.history_rows, device.kv_history.position,
                          device.kv_history.history_rows), frontier)


class ReleaseTests(unittest.TestCase):
    def test_every_scratch_buffer_is_released_once_after_a_final_fence(self):
        recorder = Recorder()
        allocated, scratch = recording_scratch()
        mesh = SimpleNamespace(num_program_cache_entries=Mock(side_effect=[40, 95]))
        with environment(**SERVED), publication_code(), scratch:
            summary, lines = warm(recorder, live_block(), SimpleNamespace(bucket_rows=(1, 2, 4)), mesh=mesh)
        frees = [event[1] for event in recorder.events if event[0] == 'free']
        self.assertEqual(sorted(frees), sorted(id(value) for value in allocated))
        first_free = next(index for index, event in enumerate(recorder.events) if event[0] == 'free')
        self.assertEqual(recorder.events[first_free - 1], ('sync',), 'the scratch is released after a fence')
        self.assertTrue(all(event[0] == 'free' for event in recorder.events[first_free:]))
        self.assertEqual(summary['program_cache'], (40, 95))
        self.assertTrue(lines[0].endswith(' program_cache=40->95'))

    def test_a_failure_releases_the_scratch_without_a_fence_and_propagates(self):
        recorder = Recorder()
        allocated, scratch = recording_scratch()
        gathers = []

        def gather(operations, mesh, collectives, partial, **keywords):
            gathers.append(partial)
            if len(gathers) == 20:
                recorder.events.append(('raise',))
                raise RuntimeError('gather failed')
            return partial * 2

        with environment(**SERVED), publication_code(), scratch, \
                patch('dflash_device.gather_add_projection', side_effect=gather):
            with self.assertRaisesRegex(RuntimeError, 'gather failed'):
                warm(recorder, live_block(), SimpleNamespace(bucket_rows=(1, 2, 4)), log=Mock(side_effect=AssertionError))
        failed = recorder.events.index(('raise',))
        after = recorder.events[failed + 1:]
        self.assertNotIn(('sync',), after, 'no fence after the failure')
        self.assertEqual(sorted(event[1] for event in after if event[0] == 'free'),
                         sorted(id(value) for value in allocated))

    def test_a_publication_left_pending_fails_the_warm(self):
        recorder = Recorder()
        with environment(**SERVED), publication_code(), \
                patch.object(DFlashDevice, 'discard_publication', lambda device, publication: None):
            with self.assertRaisesRegex(AssertionError, 'leave nothing pending'):
                warm(recorder, live_block(), SimpleNamespace())
        self.assertTrue(recorder.operations.deallocate.called)


class SkipTests(unittest.TestCase):
    def test_without_the_extent_block_nothing_is_allocated_or_logged(self):
        for block in (live_block(extent=False), SimpleNamespace(), live_block(extent=Mock())):
            with self.subTest(block=block):
                recorder = Recorder()
                allocated, scratch = recording_scratch()
                with scratch:
                    summary, lines = warm(recorder, block, SimpleNamespace(bucket_rows=(1, 2, 4)))
                self.assertIsNone(summary)
                self.assertEqual((lines, allocated, recorder.events), ([], [], []))

    def test_an_extent_block_that_cannot_publish_says_why_and_allocates_nothing(self):
        for options, reason in ((dict(collectives=None), 'no-collectives'),
                                (dict(weights=SimpleNamespace(layers=(), tensors=[])), 'no-prepared-draft-weights'),
                                (dict(weights=SimpleNamespace(layers=shared_weights().layers, projection=None,
                                                              feature_norm=torch.tensor(1.0))),
                                 'no-prepared-draft-weights')):
            with self.subTest(reason=reason):
                recorder = Recorder()
                allocated, scratch = recording_scratch()
                with scratch:
                    summary, lines = warm(recorder, live_block(), SimpleNamespace(bucket_rows=(1, 2, 4)), **options)
                self.assertIsNone(summary)
                self.assertEqual(lines, ['[PINDIAG] eager publication warm skipped reason=%s' % reason])
                self.assertEqual((allocated, recorder.events), ([], []))


# ------------------------------------------------------------------------------------------------------
# The block
# ------------------------------------------------------------------------------------------------------

SUMMARY = dict(shapes=71, ms=1.5, packed='0,16,32,48:1-16', sequential='1:1,2:1-2,4:1-4', merge_release=True,
               fused_steady_state=True, program_cache=(10, 20))


class BlockTests(extent_block.ExtentFixture):
    def test_the_extent_block_warms_once_after_every_capture_at_its_own_stage(self):
        seen = []

        def spy(block, **options):
            seen.append((block.stage, packed_verifier.capture_operation.call_count, block.phase, options))
            return dict(SUMMARY)

        collectives = SimpleNamespace(name='shared-ccl')
        with patch.object(publication_warm, 'warm', side_effect=spy):
            block = self.build(collectives=collectives)
        self.assertEqual(len(seen), 1)
        stage, captures, phase, options = seen[0]
        self.assertEqual((stage, phase), ('eager publication warm', 'preparing'))
        self.assertEqual(captures, packed_verifier.capture_operation.call_count, 'no capture after the warm')
        self.assertEqual(captures, 1 + 64, 'the verify trace and every commit trace before it')
        self.assertIs(options['operations'], self.ttnn)
        self.assertIs(options['mesh'], self.model.mesh_device)
        self.assertIs(options['pool'], self.pool)
        self.assertIs(options['shared_weights'], self.weights)
        self.assertIs(options['collectives'], collectives)
        self.assertIs(options['log'], packed_verifier.diagnostic)
        self.assertEqual(block.phase, 'idle')
        self.assertEqual(block.describe()['publication_warm'], dict(SUMMARY, program_cache=[10, 20]))

    def test_a_failed_warm_fails_the_attach_closed(self):
        with patch.object(publication_warm, 'warm', side_effect=RuntimeError('warm failed')), \
                self.assertRaisesRegex(RuntimeError, 'warm failed'):
            self.build(collectives=SimpleNamespace(name='shared-ccl'))
        self.assertFalse(self.pool.storage.taken, 'the extent storage is handed back')
        self.assertTrue(any(line.startswith('[PINDIAG] packed block warm failed with RuntimeError: warm failed; '
                                            'stage eager publication warm') for line in self.lines), self.lines)

    def test_the_real_warm_skips_a_block_without_collectives_and_says_so(self):
        block = self.build()
        self.assertIsNone(block.publication_warm)
        self.assertNotIn('publication_warm', block.describe())
        self.assertEqual([line for line in self.lines if line.startswith(publication_warm.SKIPPED_MARKER)],
                         ['[PINDIAG] eager publication warm skipped reason=no-collectives'])
        block.close()
        other = self.build(collectives=SimpleNamespace(name='shared-ccl'))
        self.assertIsNone(other.publication_warm)
        self.assertIn('[PINDIAG] eager publication warm skipped reason=no-prepared-draft-weights', self.lines)


class FlagOffBlockTests(base.FourUserFixture):
    def test_without_the_extent_pool_the_warm_is_never_called(self):
        with patch.object(publication_warm, 'warm', Mock(side_effect=AssertionError('warmed'))) as warm_mock:
            block = self.build(collectives=SimpleNamespace(name='shared-ccl'))
        warm_mock.assert_not_called()
        self.assertFalse(block.extent)
        self.assertIsNone(block.publication_warm)
        self.assertNotIn('publication_warm', block.describe())


# ------------------------------------------------------------------------------------------------------
# Shipping
# ------------------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_c2_image_carries_it_and_the_p8_lists_both_name_it(self):
        from test_c2_overlay_closure import C2_MANIFEST, C2_PACKED_ANY_MODULES, manifest_modules, read
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        overlay = manifest_modules(read(C2_MANIFEST))
        for name in ('publication_warm.py', 'packed_verifier.py'):
            with self.subTest(module=name):
                self.assertIn(name, overlay)
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())
        self.assertIn('publication_warm.py', C2_PACKED_ANY_MODULES)
        self.assertNotIn('test_publication_warm.py', overlay, 'it imports test modules the image does not carry')

    def test_nothing_pins_it(self):
        import c2_overlay
        import dflash_t16_native_attention_gate
        import target_t16_attention_gate

        for pinned in (c2_overlay.FORBIDDEN, target_t16_attention_gate.SOURCES, dflash_t16_native_attention_gate.SOURCES):
            self.assertNotIn('publication_warm.py', pinned)
            self.assertNotIn('packed_verifier.py', pinned)
        entries = c2_overlay.parse_manifest((ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8'))
        self.assertIn('scripts/ci/publication_warm.py', [entry.source for entry in entries])

    def test_the_cpu_suite_runs_this_file(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_publication_warm\b')

    def test_every_touched_file_is_lf(self):
        for relative in ('scripts/ci/publication_warm.py', 'scripts/ci/test_publication_warm.py',
                         'scripts/ci/packed_verifier.py', 'scripts/ci/lever_n_m3native_gate.py',
                         'scripts/ci/test_c2_serving_gate_s2.py', 'scripts/ci/test_c2_overlay_closure.py',
                         'scripts/ci/test_dflash_round_b1.py',
                         'docker/qwen-c2-overlay.txt', 'docker/qwen-fast-serving.Dockerfile',
                         '.github/workflows/qwen-fast-serving-image.yml', '.github/workflows/qwen-integration-cpu.yml'):
            with self.subTest(file=relative):
                self.assertNotIn(b'\r', (ROOT / relative).read_bytes())


if __name__ == '__main__':
    unittest.main()
