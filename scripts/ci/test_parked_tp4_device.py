"""Engine reuse, E2': the device's rebind and park (four chips) (serving_parked_engines.rebind_device, park_device, project_window)
and the pool's rezero (design 3.4).

Held here:
  - the windowed projection issues the whole call's per-chunk operations - the same slices at the same
    offsets, pads, feature concats, matmuls with the same program configuration, gathers, typecasts and
    norms - and gives a bit-identical history on a torch fake that computes, at every window and length;
  - on the census world (test_parked_census): a rebind with the whole-call projection repeats the
    constructor's device work event for event, from the slot's zeroing to the K/V seed; a rebound
    device's host state is a fresh device's for the same prompt; a rebind allocates nothing that outlives
    it; history_stale is cleared, rebind_generation advances, the capture's K/V cache is the new one, a
    pair's view is unwrapped and a released capture is rebuilt at the single 2048 bucket below 2048;
  - park and rebind fence before they discard a pending proposal;
  - rezero zeroes what acquire zeroes, in its order, and leaves the loan and the taken buckets alone;
  - a rebound device drafts and publishes census-clean.
"""

from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import serving_parked_engines as parked  # noqa: E402
from test_parked_tp4_census import World  # noqa: E402


# -- the windowed projection on values ----------------------------------------------------------------------

class ValueOps:
    """The ttnn calls project_features makes, computed in torch on one chip, each logged with its geometry."""

    bfloat16, float32 = torch.bfloat16, torch.float32
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.log = []

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return SimpleNamespace(**options)

    def slice(self, value, start, end):
        self.log.append(('slice', tuple(start), tuple(end)))
        return value[tuple(slice(first, last) for first, last in zip(start, end))].clone()

    def pad(self, value, padding, fill):
        self.log.append(('pad', tuple(tuple(pair) for pair in padding)))
        flat = [size for pair in reversed(padding) for size in pair]
        return torch.nn.functional.pad(value, flat, value=fill)

    def concat(self, values, dim, memory_config=None):
        dim = dim % values[0].dim()
        self.log.append(('concat', dim, tuple(tuple(value.shape) for value in values)))
        return torch.cat(values, dim=dim)

    def matmul(self, left, right, dtype=None, compute_kernel_config=None, program_config=None, memory_config=None):
        self.log.append(('matmul', tuple(left.shape), tuple(right.shape), dtype, vars(program_config)))
        return (left.float() @ right.float()).to(dtype or left.dtype)

    def typecast(self, value, dtype):
        self.log.append(('typecast', tuple(value.shape), dtype))
        return value.to(dtype)

    def rms_norm(self, value, epsilon=None, weight=None, compute_kernel_config=None, memory_config=None):
        self.log.append(('rms_norm', tuple(value.shape)))
        wide = value.float()
        return (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + epsilon) * weight.float()).to(value.dtype)

    def synchronize_device(self, mesh):
        self.log.append(('synchronize_device',))

    def deallocate(self, value):
        pass

    def get_device_tensors(self, value):
        return [SimpleNamespace(buffer_address=lambda value=value: id(value)),
                SimpleNamespace(buffer_address=lambda value=value: id(value) + 1)]


class Taps(tuple):
    """A packed block's taps name this user's first row (packed_verifier.PackedFeatureTaps)."""
    row_offset = 0


def value_device(ops, seed=0):
    from dflash_device import DFlashDevice

    generator = torch.Generator().manual_seed(seed)
    device = SimpleNamespace(operations=ops, mesh=None, collectives=None, kernel=None,
                             projection=torch.randn((12800, 64), generator=generator),
                             feature_norm=torch.randn((64,), generator=generator).bfloat16(),
                             temporaries=lambda protected: ([], lambda value: value),
                             release_except=lambda owned, output: None)
    device.project_features = partial(DFlashDevice.project_features, device)
    return device


def value_taps(rows, offset=0, seed=1):
    generator = torch.Generator().manual_seed(seed)
    taps = Taps(torch.randn((1, 1, offset + rows, 2560), generator=generator).bfloat16() for _ in range(5))
    taps.row_offset = offset
    return taps


def per_chunk(log):
    """The per-chunk operations: without the fences and the concats that join chunk or window outputs."""
    return [entry for entry in log if entry[0] != 'synchronize_device' and not (entry[0] == 'concat' and entry[1] == 2)]


class ProjectionWindowTests(unittest.TestCase):
    def setUp(self):
        import dflash_device

        patcher = patch.object(dflash_device, 'gather_add_projection',
                               lambda operations, mesh, collectives, value, retain_temporaries=None, observe=None: value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def project(self, rows, window, offset=0):
        ops = ValueOps()
        output = parked.project_window(value_device(ops), value_taps(rows, offset), rows, window=window)
        return output, ops.log

    def test_every_window_issues_the_whole_calls_chunks_and_gives_its_bits(self):
        for rows in (1, 31, 32, 33, 255, 256, 257, 1000, 2048):
            whole, whole_log = self.project(rows, 0)
            self.assertEqual(tuple(whole.shape), (1, 1, rows, 64))
            for window in (32, 64, 256, 1024, 2048):
                with self.subTest(rows=rows, window=window):
                    windowed, windowed_log = self.project(rows, window)
                    self.assertTrue(torch.equal(windowed.view(torch.int16), whole.view(torch.int16)))
                    self.assertEqual(per_chunk(windowed_log), per_chunk(whole_log))
                    windows = -(-rows // window)
                    # one fence per window call; the whole call fences once
                    self.assertEqual(windowed_log.count(('synchronize_device',)), windows)
                    self.assertEqual(whole_log.count(('synchronize_device',)), 1)

    def test_a_packed_row_offset_is_kept_across_the_windows(self):
        whole, whole_log = self.project(1000, 0, offset=64)
        windowed, windowed_log = self.project(1000, 256, offset=64)
        self.assertTrue(torch.equal(windowed.view(torch.int16), whole.view(torch.int16)))
        self.assertEqual(per_chunk(windowed_log), per_chunk(whole_log))
        starts = [entry[1][2] for entry in whole_log if entry[0] == 'slice' and entry[1][3] == 0 and entry[2][3] == 2560]
        self.assertEqual(starts[0], 64)

    def test_the_window_is_refused_unless_chunk_aligned(self):
        for window in (-32, 16, 100, 4096, 256.0):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, 'multiple of 32'):
                parked.project_window(value_device(ValueOps()), value_taps(64), 64, window=window)

    def test_the_window_flag_parses_strictly(self):
        self.assertEqual(parked.project_rows({}), 256)
        self.assertEqual(parked.project_rows({parked.PROJECT_ROWS_FLAG: '0'}), 0)
        self.assertEqual(parked.project_rows({parked.PROJECT_ROWS_FLAG: '2048'}), 2048)
        for value in ('', '32.0', ' 256', '48', '4096', '-32', 'whole'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, parked.PROJECT_ROWS_FLAG):
                parked.project_rows({parked.PROJECT_ROWS_FLAG: value})


# -- the device on the census world ------------------------------------------------------------------------

def taps(world, position):
    from test_parked_tp4_census import CensusCapture

    return CensusCapture(world.ops, world.mesh, position).outputs()


def construct_device(world, position, features, budget=16):
    """DFlashDevice as from_prefill constructs it under S2 (its capture deferred to the engine build)."""
    from dflash_device import DFlashDevice

    _, layers, projection, selector = world.fixtures
    device = DFlashDevice(world.ops, world.model, world.collectives, layers, projection, selector, features,
        position=position, block_rows=16, proposal_capture=True, max_new_tokens=budget, fused_convolution=True,
        feature_start=max(0, position - 2048), cache_history=True, cache_projection_capture=False,
        live_query_qk=False, native_proposal_attention=True, defer_proposal_capture=True,
        buffer_pool=world.pool, shared_weights=world.weights)
    world.stack.callback(lambda: device.closed or device.close())
    return device


def attach_capture(device, budget=16):
    from dflash_proposal_trace import PreparedDFlashProposal
    from serving_request_factory import single_proposal_bucket

    with single_proposal_bucket():
        device.proposal_capture = PreparedDFlashProposal(device, max_new_tokens=budget)


def build_device(world, position, budget=16):
    """A device and its single-bucket capture, as from_prefill builds them under S2."""
    features = taps(world, position)
    device = construct_device(world, position, features, budget)
    for value in features:
        world.ops.deallocate(value)
    attach_capture(device, budget)
    return device


def serve(world, device, rounds=3, prefix=2):
    """Draft and publish as DFlashRequestRuntime does: a proposal, then the verified features' publication."""
    features = device.pool_slot.verifier.buckets[2].target_features
    for _ in range(rounds):
        device.propose(7, 7)
        publication = device.prepare_publication(features, prefix, position=device.position)
        device.commit_publication(publication)


def normalized(events):
    """Serials renamed by first appearance, so two spans over different tensors compare by structure."""
    names = {}

    def name(value):
        return names.setdefault(value, len(names))
    result = []
    for event in events:
        kind = event[0]
        if kind in ('alloc', 'read', 'write', 'free', 'upload'):
            result.append((kind, name(event[1])) + tuple(event[2:]))
        else:
            result.append(event)
    return result


def span_from_zeroing(trail):
    first = next(index for index, event in enumerate(trail) if event[:1] == ('write',) and event[2] == 'full_like')
    return trail[first:]


def device_state(device):
    from gdn_multitoken_conv import addresses

    operations = device.operations

    def place(value):
        return None if value is None else tuple(addresses(operations, value))
    kv = device.kv_history
    capture = parked.single_capture(device)
    masks = {place(bucket.mask) for bucket in capture.buckets.values()}
    return dict(
        position=device.position, history_rows=device.history_rows, name=device.name.split(' ', 1)[1],
        block_rows=device.block_rows, max_drafts=device.max_drafts, owned=len(device.owned), pending=device.pending,
        history=place(device.history), spare_history=place(device.spare_history), slot=device.pool_slot.index,
        weights=device.shared_weights is not None, borrowed=[place(value) for value in device.borrowed],
        flags=(device.cache_history, device.live_query_qk, device.native_proposal_attention, device.fused_convolution),
        validated=(device.validated_live_masks == set(), device.validated_native_proposal_masks == masks),
        convolution_checks=device.convolution_checks, closed=device.closed, calls=device.proposal_calls,
        published=device.published_rows, audit_digest=device.audit_digest, progress=device.progress,
        kernel=vars(device.kernel), stale=getattr(device, 'history_stale', False),
        kv=dict(position=kv.position, history_rows=kv.history_rows, pending=kv.pending, closed=kv.closed,
                owned=len(kv.owned), projection=kv.projection, checks=kv.checks,
                active=[{name: place(value) for name, value in layer.items()} for layer in kv.active],
                spare=[{name: place(value) for name, value in layer.items()} for layer in kv.spare],
                borrowed=[place(value) for value in kv.borrowed], query=place(kv.query)),
        capture=dict(contexts=tuple(capture.buckets), cache=capture.kv_history is kv, closed=capture.closed,
                     pending=capture._pending))


class DeviceRebindTests(unittest.TestCase):
    def test_the_whole_call_rebind_repeats_the_constructors_device_work_event_for_event(self):
        for position in (4096, 2048, 1000, 1):
            with self.subTest(position=position), World() as world:
                ops = world.ops
                features = taps(world, position)
                ops.tracking, start = True, len(ops.trail)
                fresh = construct_device(world, position, features)
                constructed = span_from_zeroing(ops.trail[start:])
                ops.tracking = False
                for value in features:
                    ops.deallocate(value)
                attach_capture(fresh)
                fresh_state = device_state(fresh)
                fresh.close()
                device = build_device(world, 1)
                serve(world, device)
                self.assertIsNone(parked.park_device(device))
                features = taps(world, position)
                live = set(ops.live)
                ops.tracking, start = True, len(ops.trail)
                result = parked.rebind_device(device, features, position=position, window=0)
                rebound = span_from_zeroing(ops.trail[start:])
                ops.tracking = False
                self.assertEqual(set(ops.live), live, 'nothing a rebind allocates outlives it')
                self.assertEqual(result['single_rebuilt'], False)
                self.assertGreater(len(constructed), 100)
                self.assertEqual(normalized(rebound), normalized(constructed))
                self.assertEqual(device_state(device), fresh_state)
                self.assertEqual(ops.violations, [])

    def test_a_windowed_rebind_leaves_the_state_a_whole_call_rebind_leaves(self):
        states = {}
        for window in (0, 256, None):
            with World() as world:
                device = build_device(world, 1)
                result = parked.rebind_device(device, taps(world, 4096), position=4096, window=window)
                self.assertEqual(result['window'], 256 if window is None else window, 'the flag default is 256')
                states[window] = device_state(device)
                self.assertEqual(world.ops.violations, [])
        self.assertEqual(states[256], states[0])
        self.assertEqual(states[None], states[0])

    def test_a_rebound_device_drafts_and_publishes_census_clean(self):
        with World() as world:
            device = build_device(world, 1)
            for position in (4096, 300, 2049, 1, 2048):
                with self.subTest(position=position):
                    parked.rebind_device(device, taps(world, position), position=position)
                    self.assertIs(parked.single_capture(device).kv_history, device.kv_history)
                    serve(world, device, rounds=4, prefix=4)
                    self.assertEqual(device.position, position + 16)
                    self.assertIsNone(parked.park_device(device))
            self.assertEqual(world.ops.violations, [])

    def test_the_host_state_a_rebind_resets(self):
        with World() as world:
            device = build_device(world, 1)
            serve(world, device)
            device.history_stale = True
            device.audit_digest, device.convolution_checks = 'digest', ['check']
            self.assertIsNone(parked.park_device(device))
            parked.rebind_device(device, taps(world, 300), position=300)
            self.assertEqual((device.history_stale, device.audit_digest, device.convolution_checks), (False, None, []))
            self.assertEqual((device.proposal_calls, device.published_rows, device.pending), (0, 0, None))
            self.assertEqual(device.rebind_generation, 1)
            self.assertIn('position=300', device.name)
            parked.rebind_device(device, taps(world, 5000), position=5000)
            self.assertEqual(device.rebind_generation, 2)

    def test_a_pairs_view_is_unwrapped_and_a_released_capture_rebuilt_at_the_single_bucket(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        with World() as world:
            device = build_device(world, 4096)
            original = device.proposal_capture
            device.proposal_capture = _PackedCaptureView(object(), 'a', original)
            parked.rebind_device(device, taps(world, 3000), position=3000)
            self.assertIs(device.proposal_capture, original)
            # released by a pair while at 2048, rebound below it: the rebuilt capture keeps the 2048 bucket and
            # survives the history growing past 512
            original.close()
            device.proposal_capture = _PackedCaptureView(object(), 'a', None)
            device._packed_capture_released = True
            live = len(world.ops.live)
            result = parked.rebind_device(device, taps(world, 300), position=300)
            self.assertTrue(result['single_rebuilt'])
            self.assertEqual(tuple(device.proposal_capture.buckets), (2048,))
            self.assertIs(device.proposal_capture.kv_history, device.kv_history)
            self.assertFalse(device._packed_capture_released)
            self.assertGreater(len(world.ops.live), live, 'the rebuilt capture is the one allocation kept')
            serve(world, device, rounds=60, prefix=4)
            self.assertGreater(device.history_rows, 512)
            self.assertEqual(world.ops.violations, [])

    def test_a_device_that_cannot_be_rebound_is_refused_before_anything_is_written(self):
        with World() as world:
            device = build_device(world, 1)
            ops = world.ops
            device.proposal_capture = None
            with self.assertRaisesRegex(ValueError, 'keeps its single-user proposal capture'):
                parked.rebind_device(device, taps(world, 300), position=300)
            device = build_device(world, 1)
            device.kv_history.projection = object()
            with self.assertRaisesRegex(ValueError, 'no K/V projection capture'):
                parked.rebind_device(device, taps(world, 300), position=300)
            device.kv_history.projection = None
            device.close()
            with self.assertRaisesRegex(ValueError, 'Only an open pooled device'):
                parked.rebind_device(device, taps(world, 300), position=300)
            self.assertEqual(ops.violations, [])


class FenceOrderTests(unittest.TestCase):
    def order(self, world, device, call):
        events = []
        capture = parked.single_capture(device)
        original_discard, original_sync = capture.discard_pending, world.ops.synchronize_device
        with patch.object(capture, 'discard_pending', lambda: (events.append('discard'), original_discard())[1]), \
                patch.object(world.ops, 'synchronize_device', lambda mesh: (events.append('fence'), original_sync(mesh))[1]):
            call()
        return events

    def test_park_and_rebind_fence_before_they_discard_a_pending_proposal(self):
        with World() as world:
            device = build_device(world, 4096)
            self.assertTrue(device.prepare_device(7))
            events = self.order(world, device, lambda: self.assertIsNone(parked.park_device(device)))
            self.assertEqual(events[:2], ['fence', 'discard'])
            self.assertTrue(device.prepare_device(7))
            events = self.order(world, device, lambda: parked.rebind_device(device, taps(world, 300), position=300))
            self.assertEqual(events[:2], ['fence', 'discard'])
            self.assertIsNone(parked.single_capture(device)._pending)

    def test_park_says_why_a_device_cannot_park(self):
        with World() as world:
            device = build_device(world, 4096)
            features = device.pool_slot.verifier.buckets[2].target_features
            publication = device.prepare_publication(features, 2, position=device.position)
            self.assertEqual(parked.park_device(device), 'a pending publication')
            device.discard_publication(publication)
            self.assertIsNone(parked.park_device(device))
            saved = device.pool_slot.addresses
            device.pool_slot.addresses = ((1, 2),) + saved[1:]
            self.assertIn('pool slot moved', parked.park_device(device))
            device.pool_slot.addresses = saved
            device.close()
            self.assertEqual(parked.park_device(device), 'device closed')


class RezeroTests(unittest.TestCase):
    def test_rezero_zeroes_what_acquire_zeroes_and_keeps_the_loan_and_the_buckets(self):
        with World() as world:
            ops, pool = world.ops, world.pool
            ops.tracking, start = True, len(ops.trail)
            slot = pool.acquire(owner='engine')
            acquired = ops.trail[start:]
            taken = [bucket for bucket in (slot.verifier.take(1), slot.verifier.take(4))]
            start = len(ops.trail)
            pool.rezero(slot)
            rezeroed = ops.trail[start:]
            ops.tracking = False
            self.assertEqual(rezeroed, acquired)
            self.assertEqual([event[1] for event in rezeroed], [value.serial for value in slot.zeroed])
            self.assertTrue(all(bucket.taken for bucket in taken))
            self.assertEqual((slot.lent, slot.owner), (True, 'engine'))
            pool.release(slot)
            # an unlent slot too (the attach zeroes it before its synthetic build), still unlent afterwards
            pool.rezero(pool.slots[1])
            self.assertFalse(pool.slots[1].lent)

    def test_rezero_refuses_a_foreign_or_moved_slot_and_a_closed_pool(self):
        with World() as world:
            pool = world.pool
            with self.assertRaisesRegex(ValueError, 'Only a slot of this pool'):
                pool.rezero(SimpleNamespace(pool=pool))
            slot = pool.slots[2]
            saved = slot.addresses
            slot.addresses = ((5, 6),) + saved[1:]
            with self.assertRaisesRegex(AssertionError, 'moved'):
                pool.rezero(slot)
            slot.addresses = saved
        closed = SimpleNamespace(closed=True)
        from serving_buffer_pool import ServingBufferPool

        with self.assertRaisesRegex(ValueError, 'Closed serving buffer pool'):
            ServingBufferPool.rezero(closed, object())


if __name__ == '__main__':
    unittest.main()
