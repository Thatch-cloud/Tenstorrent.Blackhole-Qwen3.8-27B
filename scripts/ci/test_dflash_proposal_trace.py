"""Variable-user packed rounds M0: the pair-mask audit and refresh (dflash_proposal_trace
.PreparedPackedDFlashProposal under QWEN_FAST_PAIR_MASK_REFRESH / QWEN_FAST_PAIR_MASK_AUDIT)
and its fallback, QWEN_FAST_PAIRS_PACKED_ONLY (dflash_packed_proposal_coordinator, fed by
serving_worker_hook._drafts).

With every flag off the modules must behave byte for byte as they did before M0. That is
proved here against the sources themselves: the pinned commit's dflash_proposal_trace.py and
dflash_packed_proposal_coordinator.py (git show PINNED_COMMIT) run the same scenario over the
same recording fakes, and the two call logs - every runtime call with every argument, host
tensors by their bytes - must be equal. Without git history the comparison skips; every other
test here runs anywhere.

The fakes keep what a device tensor holds on each chip, so the audit's read-back and the
refresh's copy are checked by value, not only by call.

S2 v86 (run 36416471352) is here too: the G5 churn round whose older single-user replay overwrote
a fresh pair's head outputs (V86SequenceTests), the pool's pre-trace output sets every traced
draft copies into (PooledOutputTests) and the per-chip report a refused readback logs before its
error (RejectedOutputsTests)."""

import hashlib
from itertools import count
import heapq
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import dflash_proposal_trace  # noqa: E402
import dflash_packed_proposal_coordinator as coordinator_module  # noqa: E402
from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture, FakeTrace, make_bridge, make_device  # noqa: E402

# experiment/t32-score-reuse before the variable-user M0/M1 changes.
PINNED_COMMIT = '24887d15'
# The S2 gates head the pooled draft outputs branched from (image s2-7f029fa's code).
V86_PARENT = '486d8cb7'
FLAGS = ('QWEN_FAST_PAIR_MASK_REFRESH', 'QWEN_FAST_PAIR_MASK_AUDIT', 'QWEN_FAST_PAIRS_PACKED_ONLY',
         'QWEN_FAST_PADDED_PROBE', 'QWEN_FAST_ROUND_B1', 'QWEN_FAST_ROUND_B1_AUDIT', 'QWEN_FAST_PACKED_AUDIT')
# The gate's own parse of the audit line (lever_n_m3native_gate.PAIR_MASK_AUDIT_LINE).
AUDIT_LINE = re.compile(r'\[PACKED-PROPOSE\] mask round=([0-9]+|None) pair=\[([0-9]+|None),([0-9]+|None)\] '
                        r'intact=([01]) mismatched=([0-9]+) chip=([0-9]+)')


def pinned_module(relative, name, commit=PINNED_COMMIT):
    """The module at `commit`, loaded under `name` beside today's modules (its own imports
    resolve to today's siblings, which the change did not touch), or None without history."""
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/%s' % (commit, relative)], capture_output=True,
                                cwd=str(HERE), timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    module = ModuleType(name)
    exec(compile(result.stdout.decode('utf-8'), '%s@%s' % (relative, commit), 'exec'), module.__dict__)
    return module


class Normalizer:
    """A call argument as something two runs can compare: host tensors by shape, dtype and
    bytes; containers element-wise; every other object by the order it was first seen in, so
    two runs creating the same objects in the same order compare equal. Objects are kept
    alive so an id is never reused."""

    def __init__(self):
        self.ids, self.keep = {}, []

    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            flat = value.detach().contiguous().reshape(-1)
            try:
                raw = flat.view(torch.uint8).numpy().tobytes()
            except (RuntimeError, TypeError):
                raw = repr(flat.tolist()).encode()
            return ('tensor', tuple(value.shape), str(value.dtype), hashlib.sha256(raw).hexdigest())
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(self(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted(((repr(key), self(item)) for key, item in value.items()), key=lambda pair: pair[0]))
        if id(value) not in self.ids:
            self.ids[id(value)] = len(self.ids)
            self.keep.append(value)
        return ('obj', self.ids[id(value)])


class FakeShard:
    def __init__(self, owner, chip, address):
        self.owner, self.chip, self.address = owner, chip, address

    def buffer_address(self):
        return self.address


class FakeTensor:
    """A device tensor holds one value per chip; a host payload holds one value."""

    def __init__(self, value, dtype, layout, on_device, addresses):
        self.value, self.dtype, self.layout = value.clone(), dtype, layout
        self.shape = tuple(value.shape)
        self.chips = [value.clone(), value.clone()] if on_device else None
        self.shards = [FakeShard(self, chip, next(addresses)) for chip in range(2)] if on_device else None

    def write(self, value):
        self.value = value.clone()
        if self.chips is not None:
            self.chips = [value.clone(), value.clone()]


class RecordingOps:
    """The ttnn surface the pair trace uses, every call logged normalized in `events`."""

    bfloat16, uint32 = 'bf16', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'

    def __init__(self):
        self.events, self.normalize, self.addresses = [], Normalizer(), count(0x1000)

    def event(self, name, *args):
        self.events.append((name,) + tuple(self.normalize(arg) for arg in args))

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        tensor = FakeTensor(value, dtype, layout, device is not None, self.addresses)
        self.event('from_torch', value, device is not None, dtype, layout, memory_config, mesh_mapper, tensor)
        return tensor

    def copy_host_to_device_tensor(self, payload, destination):
        destination.write(payload.value)
        self.event('copy_host_to_device_tensor', payload, destination, payload.value)

    def get_device_tensors(self, tensor):
        self.event('get_device_tensors', tensor)
        return tensor.shards

    def to_torch(self, shard):
        self.event('to_torch', shard.owner, shard.chip)
        return shard.owner.chips[shard.chip].clone()

    def synchronize_device(self, mesh):
        self.event('synchronize_device')

    def begin_trace_capture(self, mesh, cq_id=0):
        trace = SimpleNamespace(kind='trace')
        self.event('begin_trace_capture', cq_id, trace)
        return trace

    def end_trace_capture(self, mesh, trace, cq_id=0):
        self.event('end_trace_capture', trace, cq_id)

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.event('execute_trace', trace, cq_id, blocking)

    def release_trace(self, mesh, trace):
        self.event('release_trace', trace)

    def copy(self, source, destination):
        self.event('copy', source, destination)

    def slice(self, value, start, end):
        result = SimpleNamespace(kind='slice')
        self.event('slice', value, start, end, result)
        return result

    def deallocate(self, tensor):
        self.event('deallocate', tensor)


def pair_devices(ops, *, context=300):
    mesh = SimpleNamespace(kind='mesh')

    def device(slot, position):
        kv_history = SimpleNamespace(active=[{'k': SimpleNamespace(), 'v': SimpleNamespace()} for _ in range(5)],
                                     pending=None, owned=[], borrowed=[])
        return SimpleNamespace(operations=ops, mesh=mesh, block_rows=16, native_proposal_attention=True,
                               kv_history=kv_history, position=position, history_rows=context, history=None,
                               spare_history=None, closed=False, pending=None, progress=None,
                               validated_native_proposal_masks=set(), pool_slot=SimpleNamespace(index=slot),
                               temporaries=lambda protected: ([], lambda value: value),
                               execute_proposal=Mock(return_value=SimpleNamespace(projected='projected', chunks=())))
    return device(0, 4096), device(1, 1200)


def pair_scenario(module, ops, *, rounds=3, geometry_change=True):
    """Build a pair, run `rounds` prepare/finish rounds, change geometry once, close. Returns
    (trace, the call log: every ops call plus every execute_proposal call, normalized)."""
    device_a, device_b = pair_devices(ops)
    with patch('dflash_packed_proposal.select_device_outputs', return_value=((1, 2, 3), (4, 5))):
        trace = module.PreparedPackedDFlashProposal(device_a, device_b)
        for index in range(rounds):
            trace.prepare_device(11 + index, 22 + index)
            trace.finish('a', 2)
            trace.finish('b', 1)
            device_a.position += 3
            device_b.position += 2
        if geometry_change:
            device_a.history_rows = device_b.history_rows = 500
            trace.prepare_device(99, 98)
            trace.finish('b', 1)
            trace.finish('a', 1)
        trace.close()
    calls = [('execute_proposal', which, ops.normalize(call.args), ops.normalize(call.kwargs))
             for which, device in (('a', device_a), ('b', device_b)) for call in device.execute_proposal.call_args_list]
    return trace, ops.events + calls


def clean_environment():
    return patch.dict(os.environ, {name: value for name, value in os.environ.items() if name not in FLAGS}, clear=True)


def logged(target):
    """Capture every line the module logs (loguru absent: print), as the server log would hold it."""
    lines = []
    stub = ModuleType('loguru')
    stub.logger = SimpleNamespace(info=lambda template, *values, **named: lines.append(
        template.format(*values, **named)))
    return lines, patch.dict('sys.modules', {'loguru': stub})


class FlagOffIsTodayTests(unittest.TestCase):
    """Every flag off: the pinned sources and today's make the same calls with the same bytes."""

    def test_the_pair_trace_matches_the_pinned_source_call_for_call(self):
        pinned = pinned_module('dflash_proposal_trace.py', 'dflash_proposal_trace_pinned')
        if pinned is None:
            self.skipTest('no git history for %s' % PINNED_COMMIT)
        for round_b1 in (False, True):
            with self.subTest(round_b1=round_b1), clean_environment():
                if round_b1:
                    os.environ['QWEN_FAST_ROUND_B1'] = '1'
                _, before = pair_scenario(pinned, RecordingOps())
                _, today = pair_scenario(dflash_proposal_trace, RecordingOps())
            self.assertGreater(len(before), 100)
            self.assertEqual(today, before)

    def test_the_coordinator_matches_the_pinned_source_call_for_call(self):
        pinned = pinned_module('dflash_packed_proposal_coordinator.py', 'dflash_packed_proposal_coordinator_pinned')
        if pinned is None:
            self.skipTest('no git history for %s' % PINNED_COMMIT)

        def run(module, **prepare):
            log = []
            operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: log.append(('sync',))))
            mesh = SimpleNamespace()

            class Trace(FakeTrace):
                def prepare_device(self, seed_a, seed_b):
                    log.append(('pair', seed_a, seed_b, vars(self).get('round_number', 'unset')))
                    return super().prepare_device(seed_a, seed_b)

            with clean_environment(), patch('dflash_proposal_trace.PreparedPackedDFlashProposal', Trace), \
                    patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture):
                os.environ['QWEN_FAST_PACKED_AUDIT'] = '1'
                bridges = []
                for index, slot in enumerate((0, 1, 2, 3)):
                    device = make_device(operations, mesh, slot=slot)
                    device.prepare_device = Mock(side_effect=lambda seed, slot=slot: log.append(('single', slot, seed)) or True)
                    bridges.append(make_bridge('r%d' % slot, device, seed=100 + index))
                coordinator = module.PackedProposalCoordinator()
                lines, loguru = logged(module)
                with loguru:
                    coordinator.prepare(bridges, **prepare)
                    coordinator.prepare(bridges[:3], **prepare)
                    bridges[3].request.session.finished = True
                    coordinator.prepare(bridges, **prepare)
            # the whole timing list: propose_ms=['0.1', '0.0'] has a space, so \S+ left all but its first element
            return log, [re.sub(r'propose_ms=\[[^\]]*\]', '', line) for line in lines]

        before = run(pinned)
        self.assertEqual(run(coordinator_module), before)
        # packed_round only matters under QWEN_FAST_PAIRS_PACKED_ONLY: unset, a sequential round
        # still pairs exactly as it always has.
        self.assertEqual(run(coordinator_module, packed_round=False), before)
        self.assertEqual(run(coordinator_module, packed_round=True), before)


class PairMaskRefreshTests(unittest.TestCase):
    def setUp(self):
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        del dflash_proposal_trace._PAIR_MASK_REFRESH_NOTED[:]

    def mask_copies(self, ops, trace):
        """Each copy_host_to_device_tensor into any bucket's mask, as (event index, the value copied)."""
        masks = {ops.normalize(bucket.mask) for bucket in trace.buckets.values()}
        return [(index, event[3]) for index, event in enumerate(ops.events)
                if event[0] == 'copy_host_to_device_tensor' and event[2] in masks]

    def build(self, ops):
        device_a, device_b = pair_devices(ops)
        trace = dflash_proposal_trace.PreparedPackedDFlashProposal(device_a, device_b)
        return trace, device_a, device_b

    def one_round(self, trace, seeds=(11, 22)):
        with patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
            self.assertTrue(trace.prepare_device(*seeds))
            trace.finish('a', 1)
            trace.finish('b', 1)

    def test_the_mask_is_copied_on_every_update_with_the_flag_and_never_without(self):
        for flag in (False, True):
            with self.subTest(refresh=flag):
                if flag:
                    os.environ['QWEN_FAST_PAIR_MASK_REFRESH'] = '1'
                ops = RecordingOps()
                trace, _, _ = self.build(ops)
                lines, loguru = logged(dflash_proposal_trace)
                with loguru:
                    for round_index in range(3):
                        self.one_round(trace, (round_index, round_index + 1))
                copies = self.mask_copies(ops, trace)
                # the build's own blocking _update, then one per prepare_device
                self.assertEqual(len(copies), 4 if flag else 0)
                bucket = next(iter(trace.buckets.values()))
                for _, payload in copies:
                    self.assertEqual(payload, ops.normalize(bucket.host_mask), 'the build mask, byte for byte')
                self.assertEqual(sum(1 for line in lines if 'pair mask refresh' in line), 1 if flag else 0)
                os.environ.pop('QWEN_FAST_PAIR_MASK_REFRESH', None)

    def test_the_deferred_prepare_device_path_refreshes_before_its_replay(self):
        os.environ['QWEN_FAST_PAIR_MASK_REFRESH'] = '1'
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        with logged(dflash_proposal_trace)[1]:
            self.one_round(trace)
            start = len(ops.events)
            self.one_round(trace, (5, 6))
        copies = [index for index, _ in self.mask_copies(ops, trace) if index >= start]
        replays = [index for index, event in enumerate(ops.events)
                   if index >= start and event[0] == 'execute_trace' and event[3] is False]
        self.assertEqual(len(copies), 1)
        self.assertEqual(len(replays), 1, "prepare_device's own non-blocking replay")
        self.assertLess(copies[0], replays[0])
        # the deferred path still fences nothing itself: the only synchronize is the caller's
        self.assertNotIn(('synchronize_device',), ops.events[start:replays[0]])

    def test_the_refresh_heals_a_clobbered_mask_on_both_chips(self):
        os.environ['QWEN_FAST_PAIR_MASK_REFRESH'] = '1'
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        with logged(dflash_proposal_trace)[1]:
            self.one_round(trace)
            bucket = next(iter(trace.buckets.values()))
            bucket.mask.chips[0].fill_(0.0)
            bucket.mask.chips[1][0, 0, 3, :] = 1.0
            self.one_round(trace, (7, 8))
        for chip in bucket.mask.chips:
            self.assertTrue(torch.equal(chip.view(torch.int16), bucket.host_mask.view(torch.int16)))

    def test_without_the_refresh_a_clobbered_mask_stays_clobbered(self):
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        self.one_round(trace)
        bucket = next(iter(trace.buckets.values()))
        bucket.mask.chips[1].fill_(0.0)
        self.one_round(trace, (7, 8))
        self.assertFalse(torch.equal(bucket.mask.chips[1], bucket.host_mask))

    def test_the_audit_reads_both_chips_before_the_refresh_and_flags_a_mutated_mask(self):
        os.environ['QWEN_FAST_PAIR_MASK_REFRESH'] = '1'
        os.environ['QWEN_FAST_PAIR_MASK_AUDIT'] = '1'
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            trace.round_number = 4
            self.one_round(trace)
            bucket = next(iter(trace.buckets.values()))
            host = bucket.host_mask
            allowed = (host == 0).nonzero()
            # two visible keys hidden on chip 1: what a replay writing into freed holes would do
            for row in allowed[:2]:
                bucket.mask.chips[1][tuple(row.tolist())] = float('-inf')
            trace.round_number = 30
            start = len(ops.events)
            self.one_round(trace, (9, 10))
            trace.round_number = 31
            self.one_round(trace, (11, 12))
        parsed = [AUDIT_LINE.search(line).groups() for line in lines if AUDIT_LINE.search(line)]
        # the build's own update and the first prepare (round 4), then rounds 30 and 31, two chips each
        self.assertEqual([(r, chip, intact, n) for r, a, b, intact, n, chip in parsed],
                         [('4', '0', '1', '0'), ('4', '1', '1', '0'), ('4', '0', '1', '0'), ('4', '1', '1', '0'),
                          ('30', '0', '1', '0'), ('30', '1', '0', '2'), ('31', '0', '1', '0'), ('31', '1', '1', '0')])
        self.assertTrue(all((a, b) == ('0', '1') for r, a, b, intact, n, chip in parsed), 'the pair is its pool slots')
        # round 30 read the mask back before copying it
        reads = [index for index, event in enumerate(ops.events) if index >= start and event[0] == 'to_torch']
        copies = [index for index, _ in self.mask_copies(ops, trace) if index >= start]
        self.assertEqual(len(reads), 4)
        self.assertLess(max(reads[:2]), copies[0])

    def test_the_audit_alone_changes_no_device_byte(self):
        os.environ['QWEN_FAST_PAIR_MASK_AUDIT'] = '1'
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        with logged(dflash_proposal_trace)[1]:
            self.one_round(trace)
        writes = [event for event in ops.events if event[0] in ('copy_host_to_device_tensor', 'copy')]
        ops_off = RecordingOps()
        os.environ.pop('QWEN_FAST_PAIR_MASK_AUDIT')
        trace_off, _, _ = self.build(ops_off)
        self.one_round(trace_off)
        self.assertEqual(len(writes), len([event for event in ops_off.events
                                           if event[0] in ('copy_host_to_device_tensor', 'copy')]))
        self.assertEqual(self.mask_copies(ops, trace), [])

    def test_a_read_back_of_another_shape_is_every_element_mismatched(self):
        os.environ['QWEN_FAST_PAIR_MASK_AUDIT'] = '1'
        ops = RecordingOps()
        trace, _, _ = self.build(ops)
        with logged(dflash_proposal_trace)[1]:
            self.one_round(trace)
            bucket = next(iter(trace.buckets.values()))
            bucket.mask.chips[0] = bucket.mask.chips[0][..., :32]
            results = trace.audit_mask(bucket)
        self.assertEqual(results[0], (0, 0, bucket.host_mask.numel()))
        self.assertEqual(results[1], (1, 1, 0))


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        FakeTrace.instances, FakeSingleUserCapture.instances = [], []
        for target, value in (('dflash_proposal_trace.PreparedPackedDFlashProposal', FakeTrace),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        del coordinator_module._PAIRS_PACKED_ONLY_NOTED[:]
        self.operations = SimpleNamespace(synchronize_device=Mock())
        self.mesh = object()

    def bridges(self, slots=(0, 1, 2, 3)):
        return [make_bridge('r%d' % slot, make_device(self.operations, self.mesh, slot=slot), seed=100 + slot)
                for slot in slots]

    def test_the_round_is_set_on_the_trace_only_under_the_audit(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges()
        coordinator.prepare(bridges)
        self.assertEqual([vars(trace).get('round_number', 'unset') for trace in FakeTrace.instances], ['unset'] * 2)
        os.environ['QWEN_FAST_PAIR_MASK_AUDIT'] = '1'
        coordinator.prepare(bridges)
        coordinator.prepare(bridges)
        self.assertEqual([trace.round_number for trace in FakeTrace.instances], [3, 3])

    def test_pairs_packed_only_unpairs_every_member_of_a_sequential_round(self):
        os.environ['QWEN_FAST_PAIRS_PACKED_ONLY'] = '1'
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges()
        lines, loguru = logged(coordinator_module)
        with loguru:
            prepared = coordinator.prepare(bridges, packed_round=False)
        self.assertEqual(FakeTrace.instances, [], 'no pair trace is built or replayed')
        for bridge in bridges:
            bridge.request.runtime.drafter.prepare_device.assert_called_once_with(bridge.request.session.seed)
        self.assertEqual(len(prepared), 4)
        self.operations.synchronize_device.assert_called_once_with(self.mesh)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('[PINDIAG] pairs packed only: round=1 unpaired=[[0, 1], [2, 3]]'))
        with loguru:
            coordinator.prepare(bridges, packed_round=False)
        self.assertEqual(len(lines), 1, 'once per process')

    def test_pairs_packed_only_still_packs_a_packed_round_and_unpacks_the_next_sequential_one(self):
        os.environ['QWEN_FAST_PAIRS_PACKED_ONLY'] = '1'
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges()
        with logged(coordinator_module)[1]:
            coordinator.prepare(bridges, packed_round=True)
            self.assertEqual([len(trace.prepared) for trace in FakeTrace.instances], [1, 1])
            for bridge in bridges:
                bridge.request.runtime.drafter.prepare_device.assert_not_called()
            # the pair's first capture released each member's own single-user capture
            self.assertTrue(all(getattr(bridge.request.runtime.drafter, '_packed_capture_released', False)
                                for bridge in bridges))
            coordinator.prepare(bridges, packed_round=False)
        self.assertEqual([len(trace.prepared) for trace in FakeTrace.instances], [1, 1], 'no pair replay')
        # each member's single-user capture is rebuilt, once, and prepared through its view
        self.assertEqual(len(FakeSingleUserCapture.instances), 4)
        for bridge in bridges:
            bridge.request.runtime.drafter.prepare_device.assert_called_once()
            capture = bridge.request.runtime.drafter.proposal_capture
            self.assertIsInstance(capture, coordinator_module._PackedCaptureView)
            self.assertIsInstance(capture._original, FakeSingleUserCapture)

    def test_unset_a_sequential_round_still_pairs(self):
        coordinator = coordinator_module.PackedProposalCoordinator()
        coordinator.prepare(self.bridges(), packed_round=False)
        self.assertEqual([len(trace.prepared) for trace in FakeTrace.instances], [1, 1])

    def test_a_lone_member_is_unaffected(self):
        os.environ['QWEN_FAST_PAIRS_PACKED_ONLY'] = '1'
        coordinator = coordinator_module.PackedProposalCoordinator()
        bridges = self.bridges((0, 2))
        lines, loguru = logged(coordinator_module)
        with loguru:
            coordinator.prepare(bridges, packed_round=False)
        self.assertEqual(lines, [], 'no full pair was split')
        self.assertEqual(coordinator_module.unpair_groups([(0,), (2,)], 1), [(0,), (2,)])


class HookPassesThePolicyTests(unittest.TestCase):
    """serving_worker_hook._drafts hands the coordinator packed_round only under the fallback."""

    def run_hook(self, environ, proposal_rows=None):
        import test_serving_worker_hook as hook_tests
        from serving_worker_hook import FastWorkerHook

        fixture = hook_tests.WorkerHookTests()
        worker, bridge, events, scheduled = fixture.fixture()
        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False)
        original = hook.bridges
        order = []
        operations = SimpleNamespace(synchronize_device=Mock())
        hook.bridges = fixture.make_pipelined_bridges('abcd', operations=operations, mesh=object(), order=order)
        if proposal_rows is not None:
            hook.packed_step = SimpleNamespace(proposal_rows=lambda requests: proposal_rows)
        calls = []

        class Coordinator:
            def prepare(self, bridges, *args, **kwargs):
                calls.append((len(bridges), args, kwargs))
                return []

            def close(self):
                pass

        outputs = ModuleType('vllm.v1.outputs')
        outputs.DraftTokenIds = SimpleNamespace
        try:
            with clean_environment(), patch.dict('sys.modules', {'vllm.v1.outputs': outputs}), \
                    patch.object(coordinator_module, 'PackedProposalCoordinator', Coordinator):
                os.environ.update(QWEN_FAST_PIPELINED_PROPOSALS='1', QWEN_FAST_PACKED_PROPOSAL='1', **environ)
                worker.take_draft_token_ids()
        finally:
            hook.bridges = original
            hook.packed_step = None
            hook.close()
        return calls

    def test_unset_the_call_is_todays(self):
        self.assertEqual(self.run_hook({}), [(4, (), {})])
        self.assertEqual(self.run_hook({}, proposal_rows=16), [(4, (), {})])

    def test_set_the_coordinator_learns_whether_the_round_is_packed(self):
        flag = dict(QWEN_FAST_PAIRS_PACKED_ONLY='1')
        self.assertEqual(self.run_hook(flag), [(4, (), dict(packed_round=False))])
        self.assertEqual(self.run_hook(flag, proposal_rows=16), [(4, (), dict(packed_round=True))])


class HoleOps(RecordingOps):
    """RecordingOps over an allocator that reuses freed addresses, lowest first, plus request traces that bake the
    addresses of the intermediates they free after capture: a replay writes each chip's own garbage over whatever
    live tensor now sits at one - serving_buffer_pool's WHY, the M0 class, on a CPU."""

    def __init__(self):
        super().__init__()
        self.fresh, self.holes, self.live = count(0x100000, 0x1000), [], {}

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        if device is None:
            return super().from_torch(value, device, dtype, layout, memory_config, mesh_mapper)
        address = heapq.heappop(self.holes) if self.holes else next(self.fresh)
        tensor = FakeTensor(value, dtype, layout, True, iter((address, address)))
        self.live[address] = tensor
        self.event('from_torch', value, True, dtype, layout, memory_config, mesh_mapper, tensor)
        return tensor

    def deallocate(self, tensor):
        super().deallocate(tensor)
        address = tensor.shards[0].address
        if self.live.get(address) is tensor:
            del self.live[address]
            heapq.heappush(self.holes, address)

    def full_like(self, tensor, value, *, optional_tensor):
        optional_tensor.write(torch.full_like(optional_tensor.value, value))

    def capture_request_trace(self, intermediates=4):
        """A request engine's verify/commit capture: its intermediates allocated, baked, then freed."""
        made = [self.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), device='mesh') for _ in range(intermediates)]
        baked = [tensor.shards[0].address for tensor in made]
        for tensor in made:
            self.deallocate(tensor)
        return baked

    def replay_request_trace(self, baked):
        """A sequential round's replay: the intermediates written at their baked addresses, different on each chip."""
        for address in baked:
            tensor = self.live.get(address)
            if tensor is not None and tensor.chips is not None:
                tensor.chips = [torch.full_like(chip, float(chip_index + 1)) for chip_index, chip in enumerate(tensor.chips)]


class V73SequenceTests(unittest.TestCase):
    """Run 36358821640 (v73), on CPU. Pair [2, 3] is re-formed after a detach: slot 3's request finishes, W6c
    releases the pair's trace, a new request takes the slot and captures its engine's traces, and the pair is built
    again over it. Then v73's rounds 594 and 595: a sequential round - the per-request traces of slots 1-3 replay,
    and the sequential publish swaps each drafter's K/V banks, so the pair's next round normalises the live banks
    ('live banks pair=[2, 3] normalised=20' after one swap, nothing after two) - and the pair's packed round, whose
    audit reads the mask before the refresh's copy and again after it.

    With the pool's pre-trace masks (serving_buffer_pool draft_masks=, at the shapes serving_runtime asks the
    coordinator for under S2) the mask survives every replay. Uploaded by the pair as before, it sits in a hole a
    live request trace baked, and the first read finds the whole mask clobbered on both chips, as v73 did."""

    # v235's image flags on this path (docker/qwen-c2-v235-environment.json) and the gate's audit.
    FLAGS = dict(QWEN_FAST_PACKED_PROPOSAL='1', QWEN_FAST_PAIR_ROW_EXACT='1', QWEN_FAST_FUSED_COMMIT='1',
                 QWEN_FAST_FUSED_COMMIT_INPLACE='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1',
                 QWEN_FAST_PAIR_MASK_REFRESH='1', QWEN_FAST_PAIR_MASK_AUDIT='1')

    def setUp(self):
        import fused_commit
        import pair_row_exact
        import serving_buffer_pool

        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.update(self.FLAGS)
        for target, name, value in ((serving_buffer_pool, 'HISTORY_SHAPE', (1, 1, 32, 64)),
                                    (serving_buffer_pool, 'KV_SHAPE', (1, 4, 32, 128)),
                                    (fused_commit, '_LIVE_NOTED', []), (pair_row_exact, '_NOTED', []),
                                    (dflash_proposal_trace, '_PAIR_MASK_REFRESH_NOTED', [])):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def device(ops, mesh, slot, position):
        """A request's drafter on its pool slot, its live banks on the slot's active side."""
        kv_history = SimpleNamespace(active=[dict(layer['active']) for layer in slot.kv], pending=None, owned=[],
                                     borrowed=[value for layer in slot.kv for side in layer.values()
                                               for value in side.values()])
        return SimpleNamespace(operations=ops, mesh=mesh, block_rows=16, native_proposal_attention=True,
                               kv_history=kv_history, position=position, history_rows=2048, history=None,
                               spare_history=None, closed=False, pending=None, progress=None,
                               validated_native_proposal_masks=set(), pool_slot=slot,
                               temporaries=lambda protected: ([], lambda value: value),
                               execute_proposal=Mock(return_value=SimpleNamespace(projected='projected', chunks=())))

    @staticmethod
    def swap_banks(device):
        """The sequential publish's bank swap (draft_kv_history): the live banks move to the other pool side."""
        slot = device.pool_slot
        on_active = device.kv_history.active[0]['k'] is slot.kv[0]['active']['k']
        device.kv_history.active = [dict(layer['spare' if on_active else 'active']) for layer in slot.kv]

    def packed_round(self, trace, round_number, seeds):
        trace.round_number = round_number
        with patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
            self.assertTrue(trace.prepare_device(*seeds))
            trace.finish('a', 1)
            trace.finish('b', 1)

    def run_v73(self, *, pooled):
        """The sequence; returns (ops, pool, the re-formed pair's trace, the log lines, the addresses the live
        request traces baked at the sequential rounds)."""
        from serving_buffer_pool import ServingBufferPool

        ops, mesh = HoleOps(), 'mesh'
        # The attach: the pool, before any trace - with the packed drafts' masks when pooled.
        draft_masks = coordinator_module.pooled_draft_mask_shapes(4, 16) if pooled else None
        pool = ServingBufferPool(ops, mesh, users=4, draft_masks=draft_masks)
        lines, loguru = logged(dflash_proposal_trace)
        coordinator = coordinator_module.PackedProposalCoordinator()
        with loguru:
            slots = [pool.acquire(owner='r%d' % index) for index in range(4)]
            # Four engines capture their per-request traces: the holes they leave are baked.
            baked = {index: ops.capture_request_trace() for index in range(4)}
            devices = {index: self.device(ops, mesh, slots[index], 130000 + index) for index in range(4)}
            # Pair [2, 3] forms and serves packed rounds.
            first = coordinator._trace_for((2, 3), devices[2], devices[3])
            for round_number in (10, 11):
                self.packed_round(first, round_number, (5, 6))
            # Slot 3's request finishes: its device closes, W6c releases the pair's trace at detach, and the slot
            # and the engine (whose traces never replay again) go back.
            devices[3].closed = True
            self.assertEqual(coordinator.release_closed(), dict(quad=0, pairs=[[2, 3]]))
            self.assertTrue(first.closed)
            del baked[3]
            pool.release(slots[3])
            # A new request takes slot 3 and builds its engine: new traces, new baked holes.
            slots[3] = pool.acquire(owner='r4')
            baked[3] = ops.capture_request_trace()
            devices[3] = self.device(ops, mesh, slots[3], 127000)
            # The pair re-forms over it.
            trace = coordinator._trace_for((2, 3), devices[2], devices[3])
            self.assertIsNot(trace, first)
            for round_number in (580, 581):
                self.packed_round(trace, round_number, (7, 8))
            # v73's rounds 594 and 595: a sequential round, then the pair's packed round.
            for round_number in (594, 595):
                for index in (3, 2, 1):
                    ops.replay_request_trace(baked[index])
                for index in (1, 2, 3):
                    self.swap_banks(devices[index])
                self.packed_round(trace, round_number, (9, 10))
        live = {address for index in (1, 2, 3) for address in baked[index]}
        return ops, pool, trace, lines, live

    @staticmethod
    def reads(lines, pattern, rounds=(594, 595)):
        return [match.groups() for match in map(pattern.search, lines)
                if match is not None and match.group(1) in {str(value) for value in rounds}]

    def test_uploaded_by_the_pair_the_mask_sits_in_a_replayed_hole_and_reads_clobbered(self):
        # Today's path (a pool without masks), v73 reproduced: the whole mask on both chips, healed only by the
        # refresh's copy.
        ops, pool, trace, lines, live = self.run_v73(pooled=False)
        bucket = trace.buckets[(2048, 2048)]
        self.assertFalse(hasattr(bucket, 'lent_mask'))
        self.assertIn(bucket.mask.shards[0].address, live, 'the mask was uploaded into a live trace\'s hole')
        before = self.reads(lines, AUDIT_LINE)
        self.assertEqual([(r, a, b, chip, intact) for r, a, b, intact, n, chip in before],
                         [('594', '2', '3', '0', '0'), ('594', '2', '3', '1', '0'),
                          ('595', '2', '3', '0', '0'), ('595', '2', '3', '1', '0')])
        self.assertEqual({int(n) for *_, n, chip in before}, {bucket.host_mask.numel()}, 'every element')
        self.assertEqual([line for line in lines if 'live banks pair=[2, 3] normalised' in line],
                         ['[PACKED-PROPOSE] live banks pair=[2, 3] normalised=20'])
        self.assertFalse([line for line in lines if line.startswith('[PINDIAG] draft mask pooled')])

    def test_with_the_pools_pre_trace_mask_the_pair_mask_survives_the_sequential_rounds(self):
        ops, pool, trace, lines, live = self.run_v73(pooled=True)
        bucket = trace.buckets[(2048, 2048)]
        held = pool.draft_masks[(2, 3)]
        self.assertIs(bucket.mask, held, 'the re-formed pair borrows the pool mask again')
        self.assertEqual(bucket.lent_mask, (held,))
        self.assertNotIn(held.shards[0].address, live)
        self.assertIs(ops.live.get(held.shards[0].address), held, 'never freed by the detach\'s close')
        before = self.reads(lines, AUDIT_LINE)
        self.assertEqual([(r, a, b, chip, intact, n) for r, a, b, intact, n, chip in before],
                         [('594', '2', '3', '0', '1', '0'), ('594', '2', '3', '1', '1', '0'),
                          ('595', '2', '3', '0', '1', '0'), ('595', '2', '3', '1', '1', '0')])
        # every (pre-refresh) audit read of both pair traces found it intact: nothing to heal
        every = [match.group(4) for match in map(AUDIT_LINE.search, lines) if match is not None]
        self.assertEqual(len(every), 2 * (3 + 5), 'two chips x (3 first-pair + 5 re-formed) updates')
        self.assertEqual(set(every), {'1'})
        self.assertEqual([line for line in lines if line.startswith('[PINDIAG] draft mask pooled')],
                         ['[PINDIAG] draft mask pooled slots=[2,3] shape=1x1x32x2080'] * 2)
        self.assertEqual([line for line in lines if 'live banks pair=[2, 3] normalised' in line],
                         ['[PACKED-PROPOSE] live banks pair=[2, 3] normalised=20'])
        for chip in bucket.mask.chips:
            self.assertTrue(torch.equal(chip.view(torch.int16), bucket.host_mask.view(torch.int16)))
        trace.close()
        self.assertIs(ops.live.get(held.shards[0].address), held, 'nor by the re-formed pair\'s close')

    def test_the_pool_masks_are_the_ones_serving_runtime_asks_for(self):
        self.assertEqual(coordinator_module.pooled_draft_mask_shapes(4, 16),
                         {(0, 1): (1, 1, 32, 2080), (2, 3): (1, 1, 32, 2080)})
        self.assertEqual(coordinator_module.pooled_draft_mask_shapes(2, 16), {(0, 1): (1, 1, 32, 2080)})
        with patch.dict(os.environ, QWEN_FAST_PAIR_ROW_EXACT='0'):
            self.assertEqual(coordinator_module.pooled_draft_mask_shapes(4, 16),
                             {(0, 1): (1, 1, 32, 4160), (2, 3): (1, 1, 32, 4160)}, 'the unfolded pair mask')
        with patch.dict(os.environ, QWEN_FAST_QUAD_DRAFT='1'):
            shapes = coordinator_module.pooled_draft_mask_shapes(4, 16)
            self.assertEqual(shapes[(0, 1, 2, 3)], (1, 1, 32, 2080))
            self.assertNotIn((0, 1, 2, 3), coordinator_module.pooled_draft_mask_shapes(3, 16))
        with patch.dict(os.environ, QWEN_FAST_PACKED_PROPOSAL='0'):
            self.assertEqual(coordinator_module.pooled_draft_mask_shapes(4, 16), {}, 'no pair forms: nothing pooled')

    def test_a_pool_mask_of_another_shape_is_refused_and_the_pair_uploads_its_own(self):
        from serving_buffer_pool import ServingBufferPool

        ops = HoleOps()
        pool = ServingBufferPool(ops, 'mesh', users=4, draft_masks={(2, 3): (1, 1, 32, 4160)})
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            slots = [pool.acquire(owner='r%d' % index) for index in range(4)]
            trace = dflash_proposal_trace.PreparedPackedDFlashProposal(self.device(ops, 'mesh', slots[2], 130000),
                                                                       self.device(ops, 'mesh', slots[3], 130001))
            self.packed_round(trace, 1, (1, 2))
        bucket = trace.buckets[(2048, 2048)]
        self.assertIsNot(bucket.mask, pool.draft_masks[(2, 3)])
        self.assertIn(bucket.mask, trace.owned, 'uploaded and owned, as before')
        self.assertEqual([line for line in lines if line.startswith('[PINDIAG] draft mask pooled')],
                         ['[PINDIAG] draft mask pooled refused slots=[2,3] shape=1x1x32x2080: the pool holds '
                          '[([2, 3], [1, 1, 32, 4160])]'])


class ReplayingHoleOps(HoleOps):
    """HoleOps whose traces replay: what a captured operation registered (on_replay) - a draft head writing its
    outputs, and every ttnn.copy the capture recorded - runs again at each execute_trace of that trace, in enqueue
    order (one in-order command queue). A copy writes each chip's value into its destination."""

    uint16 = 'u16'

    def __init__(self):
        super().__init__()
        self.capturing, self.replays = None, {}
        # every replay (trace, blocking) and every TracedSingle replay, in enqueue order
        self.order = []

    def begin_trace_capture(self, mesh, cq_id=0):
        trace = super().begin_trace_capture(mesh, cq_id)
        self.capturing = trace
        self.replays[id(trace)] = []
        return trace

    def end_trace_capture(self, mesh, trace, cq_id=0):
        super().end_trace_capture(mesh, trace, cq_id)
        self.capturing = None

    def on_replay(self, action):
        if self.capturing is not None:
            self.replays[id(self.capturing)].append(action)

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        super().execute_trace(mesh, trace, cq_id, blocking)
        self.order.append((trace, blocking))
        for action in self.replays.get(id(trace), ()):
            action()

    def copy(self, source, destination):
        super().copy(source, destination)

        def write():
            if getattr(source, 'chips', None) is not None and getattr(destination, 'chips', None) is not None:
                destination.chips = [chip.clone() for chip in source.chips]
        write()
        self.on_replay(write)

    def fill_holes(self):
        """Persistent allocations (an engine build) that take every hole there is."""
        return [self.from_torch(torch.zeros(1, 1, 32, 16), device='mesh', dtype=self.bfloat16, layout=self.TILE_LAYOUT)
                for _ in range(len(self.holes))]


def head_outputs(ops, rows=32, projected_rows=32):
    """What a draft pass's shared head leaves on device, allocated now (inside a capture: the trace's outputs): per
    candidate chunk the top-16 values (BF16) and indices (UINT16) at `rows` rows, and the selector projection, plus
    the write that fills them with a well-formed block (finite descending values, distinct in-range ids)."""
    from draft_shared_head import candidate_chunks

    chunks = []
    for start, stop in candidate_chunks():
        values = ops.from_torch(torch.zeros(1, 1, rows, 16), device='mesh', dtype=ops.bfloat16, layout=ops.TILE_LAYOUT)
        indices = ops.from_torch(torch.zeros(1, 1, rows, 16, dtype=torch.int32), device='mesh', dtype=ops.uint16,
                                 layout=ops.TILE_LAYOUT)
        chunks.append(dict(start=start, stop=stop, values=values, indices=indices))
    projected = ops.from_torch(torch.zeros(1, 1, projected_rows, 256), device='mesh', dtype=ops.bfloat16,
                               layout=ops.TILE_LAYOUT)

    def write():
        for number, chunk in enumerate(chunks):
            chunk['values'].write((torch.linspace(10.0, 1.0, 16).repeat(rows, 1) - number).reshape(1, 1, rows, 16))
            chunk['indices'].write((torch.arange(16, dtype=torch.int32) * 7 + number).repeat(rows, 1)
                                   .reshape(1, 1, rows, 16))
        projected.write(torch.full((1, 1, projected_rows, 256), 0.5))

    write()
    return SimpleNamespace(projected=projected, chunks=chunks), write


def output_tensors(outputs):
    return [*(chunk[key] for chunk in outputs.chunks for key in ('values', 'indices')), outputs.projected]


class TracedSingle:
    """A request's single-user PreparedDFlashProposal reduced to what matters here: its capture allocates its outputs
    and op-internal scratch and frees the scratch (holes it baked); every replay rewrites the baked addresses with
    its scratch (a -inf pad, out-of-range ids) and its own outputs with a good block; finish() reads them through
    merge_chunk_candidates."""

    def __init__(self, ops, *, scratch=24):
        from draft_shared_head import merge_chunk_candidates

        self.ops, self.merge = ops, merge_chunk_candidates
        made = [ops.from_torch(torch.zeros(1, 1, 32, 16), device='mesh', dtype=ops.bfloat16, layout=ops.TILE_LAYOUT)
                for _ in range(scratch)]
        self.outputs, self.write = head_outputs(ops, rows=16)
        self.baked = [tensor.shards[0].address for tensor in made]
        for tensor in made:
            ops.deallocate(tensor)
        self.pending, self.closed = None, False

    def prepare_device(self, seed):
        self.ops.order.append((self, False))
        for address in self.baked:
            tensor = self.ops.live.get(address)
            if tensor is not None and tensor.chips is not None:
                tensor.chips = [torch.full_like(chip, float('-inf')) if chip.is_floating_point()
                                else torch.full_like(chip, 1 << 20) for chip in tensor.chips]
        self.write()
        self.pending = seed
        return True

    def has_pending(self, seed):
        return self.pending == seed

    def finish(self, count):
        self.pending = None
        self.merge([dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                         values=self.ops.to_torch(chunk['values'].shards[chip]).float().reshape(16, 16),
                         indices=self.ops.to_torch(chunk['indices'].shards[chip]).long().reshape(16, 16))
                    for chunk in self.outputs.chunks for chip in range(2)], block_rows=16)
        return (1,) * count

    def discard_pending(self):
        self.pending = None

    def close(self):
        self.closed = True


class V86SequenceTests(unittest.TestCase):
    """Run 36416471352 (v86, G5 churn), on CPU: a fresh pair's head outputs overwritten, before its collect, by the
    replay of an OLDER single-user trace that the coordinator enqueues after the pair in the same round.

    v86, 11:47:44.98-11:48:00.97: slot 2 loses its partner (slot 3 departs) and its single-user capture is rebuilt
    ('recapture slot=2'); its capture frees its op-internal intermediates - holes its trace baked, which every replay
    of it rewrites. Nothing persistent is allocated after it. Slot 0 (the prompt-1536 user) reaches history_rows 2048,
    so pair [0, 1] forms for the first time and is captured, all of it from freed fragments; its outputs (the head's
    top-16 values and indices, the selector projection) are allocated then. PackedProposalCoordinator.prepare walks
    FOUR_AS_TWO_PAIRS in order: pair (0, 1) is built and replayed, THEN slot 2's single (reason 'absent') is prepared
    and replayed, then the one fence, then select_round -> collect -> read_device_outputs -> merge_chunk_candidates
    raised 'Finite complete-block top16 values and in-range integer indices required'.

    With the pool's pre-trace output sets (serving_buffer_pool draft_outputs=, at the shapes serving_runtime asks the
    coordinator for under S2) the pair's pass ends by copying its outputs into its set, which no trace's capture ever
    had free; the older single's replay still writes over the pair's own capture outputs, after the copy, and collect
    reads the set. The replay order is unchanged. Without the sets the round fails as v86 did.

    The pair is the real PreparedPackedDFlashProposal, the coordinator the real PackedProposalCoordinator
    (QWEN_FAST_ROUND_B1: select_round/collect), the check the real draft_shared_head.merge_chunk_candidates."""

    FLAGS = dict(QWEN_FAST_PACKED_PROPOSAL='1', QWEN_FAST_PAIR_ROW_EXACT='1', QWEN_FAST_FUSED_COMMIT='1',
                 QWEN_FAST_FUSED_COMMIT_INPLACE='1', QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1',
                 QWEN_FAST_PAIR_MASK_REFRESH='1', QWEN_FAST_ROUND_B1='1', QWEN_FAST_PACKED_AUDIT='1')
    ERROR = 'Finite complete-block top16 values and in-range integer indices required'

    def setUp(self):
        import fused_commit
        import pair_row_exact
        import serving_buffer_pool

        environment = clean_environment()
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.update(self.FLAGS)
        for target, name, value in ((serving_buffer_pool, 'HISTORY_SHAPE', (1, 1, 32, 64)),
                                    (serving_buffer_pool, 'KV_SHAPE', (1, 4, 32, 128)),
                                    (fused_commit, '_LIVE_NOTED', []), (pair_row_exact, '_NOTED', []),
                                    (dflash_proposal_trace, '_PAIR_MASK_REFRESH_NOTED', [])):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def drafter(ops, slot, position):
        """A DFlashDevice on its pool slot: live banks on the pool's active side, its own single-user capture, and
        (as a pair's device_a) an execute_proposal whose outputs the pair trace's replay rewrites."""
        kv_history = SimpleNamespace(active=[dict(layer['active']) for layer in slot.kv], pending=None, owned=[],
                                     borrowed=[value for layer in slot.kv for side in layer.values()
                                               for value in side.values()])
        device = SimpleNamespace(operations=ops, mesh='mesh', block_rows=16, native_proposal_attention=True,
                                 kv_history=kv_history, layers=[object()] * len(slot.kv), position=position,
                                 history_rows=min(position, 2048), history=None, spare_history=None, closed=False,
                                 pending=None, progress=None, validated_native_proposal_masks=set(), pool_slot=slot,
                                 predecessors='codebook', successors='codebook',
                                 temporaries=lambda protected: ([], lambda value: value))

        def execute_proposal(*args, **kwargs):
            outputs, write = head_outputs(ops)
            ops.on_replay(write)
            return outputs

        device.execute_proposal = execute_proposal
        device.proposal_capture = TracedSingle(ops)
        device.prepare_device = lambda seed: device.proposal_capture.prepare_device(seed)
        return device

    @staticmethod
    def bridge(name, device, seed):
        session = SimpleNamespace(request_id=name, seed=seed, pending=None, finished=False)
        return SimpleNamespace(request=SimpleNamespace(session=session, runtime=SimpleNamespace(drafter=device),
                                                       closed=False, cancelled=False), failed=False)

    def run_v86(self, *, pooled, recapture_slot2_last=True):
        """Slots 1 and 2 serve long prompts, slot 0's short prompt ramps (position 2046). Engine builds fill every
        hole; then slot 2's single-user capture is rebuilt (the newest trace, its holes the only ones), a singles
        round runs, slot 0 commits to 2048 and pair [0, 1] forms. Returns (ops, pool, devices, the log lines, the pair
        trace, the ValueError the pair round raised or None)."""
        from serving_buffer_pool import ServingBufferPool

        ops = ReplayingHoleOps()
        # The attach: the pool, before any trace - with the drafts' head-output sets when pooled. serving_runtime
        # passes the keyword only when there are sets; without it the pool is the M0 one.
        outputs = dict(draft_outputs=coordinator_module.pooled_draft_output_shapes(4, 16)) if pooled else {}
        pool = ServingBufferPool(ops, 'mesh', users=4, draft_masks=coordinator_module.pooled_draft_mask_shapes(4, 16),
                                 **outputs)
        slots = [pool.acquire(owner='r%d' % index) for index in range(3)]
        devices = [self.drafter(ops, slots[0], 2046), self.drafter(ops, slots[1], 120357),
                   self.drafter(ops, slots[2], 110787)]
        ops.fill_holes()
        if recapture_slot2_last:
            # 'recapture slot=2': slot 2's partner left; nothing allocates after it.
            devices[2].proposal_capture = TracedSingle(ops)
        bridges = [self.bridge('r%d' % index, device, 100 + index) for index, device in enumerate(devices)]
        coordinator = coordinator_module.PackedProposalCoordinator()
        lines, loguru = logged(coordinator_module)
        failure = None
        with loguru, patch('dflash_packed_proposal.select_packed_batched',
                           side_effect=lambda parts, *rest: [(7,) for _ in parts]):
            # 'singles round=638 slots=[0, 1, 2] reasons=['ramp', 'ramp', 'absent']'
            coordinator.prepare(bridges)
            for device in devices:
                device.proposal_capture.finish(1)
            # the ramp commit at position 2046, prefix 2: slot 0 reaches history_rows 2048
            devices[0].position = devices[0].history_rows = 2048
            try:
                coordinator.prepare(bridges)
            except ValueError as error:
                failure = error
        trace = coordinator.pairs[(0, 1)][2] if (0, 1) in coordinator.pairs else None
        return ops, pool, devices, lines, trace, failure

    def test_with_the_pools_output_sets_the_fresh_pair_survives_the_older_singles_replay(self):
        # The regression test: the v86 round, pooled. Red before the fix (no draft_outputs anywhere), green with it.
        ops, pool, devices, lines, trace, failure = self.run_v86(pooled=True)
        self.assertIsNone(failure, 'v86: %s' % failure)
        self.assertTrue(any('PACKED-SELECT' in line and 'pairs=[[0, 1]]' in line for line in lines), lines[-5:])
        bucket = trace.buckets[(2048, 2048)]
        held = pool.draft_outputs[(0, 1)]
        self.assertEqual([id(tensor) for tensor in output_tensors(bucket.outputs)],
                         [id(tensor) for tensor in output_tensors(held)], 'collect reads the pool set')
        holes = set(devices[2].proposal_capture.baked)
        self.assertFalse({tensor.shards[0].address for tensor in output_tensors(held)} & holes,
                         'no trace capture ever had the set free')
        # The mechanism still runs: the pair's own capture outputs sit in slot 2's holes and its replay, after the
        # pair's, overwrote them - after the pair's copy, so nothing reads them.
        own = [ops.live[address] for address in holes if address in ops.live]
        clobbered = [tensor for tensor in own if tensor.chips is not None and tensor.chips[0].is_floating_point()
                     and torch.isinf(tensor.chips[0]).all()]
        self.assertTrue(clobbered, 'the older replay still writes its holes')
        self.assertEqual(['[PINDIAG] draft outputs pooled slots=[0,1] head=1x1x32x16 projected=1x1x32x256'],
                         [line for line in lines if line.startswith('[PINDIAG] draft outputs pooled')])
        self.assertFalse([line for line in lines if 'draft outputs rejected' in line])

    def test_the_replay_order_is_unchanged(self):
        # The fix is the pooled sets, not a reorder: the pair still replays first and slot 2's older single after it.
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                ops, pool, devices, lines, trace, failure = self.run_v86(pooled=pooled)
                pair = trace.buckets[(2048, 2048)].trace
                built = ops.order.index((pair, True))
                self.assertEqual(ops.order[built + 1:], [(pair, False), (devices[2].proposal_capture, False)],
                                 "the pair's round replay, then slot 2's older single, as v86 enqueued them")
                self.assertEqual(failure is None, pooled, 'the same order fails only without the sets')

    def test_without_them_the_older_single_overwrites_the_fresh_pairs_outputs_and_each_chip_is_reported(self):
        # The diagnosis, on a pool without output sets (today's attach): the same round fails as v86 did, and the
        # readback says what it held on each chip before raising the same error.
        ops, pool, devices, lines, trace, failure = self.run_v86(pooled=False)
        self.assertIsNotNone(trace, 'pair [0, 1] was built')
        outputs = trace.buckets[(2048, 2048)].outputs
        holes = set(devices[2].proposal_capture.baked)
        self.assertTrue({chunk['values'].shards[0].address for chunk in outputs.chunks} & holes,
                        "the fresh pair's head outputs were allocated in slot 2's single-user capture holes")
        self.assertIsNotNone(failure, 'the pair round fails, as v86 did')
        self.assertEqual(str(failure), self.ERROR)
        rejected = [line for line in lines if line.startswith('[PINDIAG] draft outputs rejected')]
        self.assertEqual(len(rejected), 2, 'one line per chip')
        for chip, line in enumerate(rejected):
            self.assertIn('device_slot=0 chip=%d ' % chip, line)
            self.assertRegex(line, r' neg_inf=[1-9][0-9]* ')
            self.assertIn('projected_finite=1 projected_equal=1', line)
            self.assertIn('error=Finite_complete-block_top16', line)
        self.assertFalse([line for line in lines if line.startswith('[PINDIAG] draft outputs pooled')])

    def test_a_single_recaptured_before_the_pairs_slots_filled_holes_is_harmless(self):
        # The control: no hole of a trace that replays after the pair is free when the pair captures (v79's churn,
        # and every other pair capture in 60 gate logs) - the same round passes, pooled or not.
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                ops, pool, devices, lines, trace, failure = self.run_v86(pooled=pooled, recapture_slot2_last=False)
                self.assertIsNone(failure)
                self.assertTrue(any('PACKED-SELECT' in line and 'pairs=[[0, 1]]' in line for line in lines))

    def test_the_output_sets_are_the_ones_serving_runtime_asks_for(self):
        from draft_shared_head import candidate_chunks

        chunks = tuple(candidate_chunks())

        def spec(head_rows, projected_rows):
            return dict(chunks=chunks, head=(1, 1, head_rows, 16), projected=(1, 1, projected_rows, 256))

        singles = {(slot,): spec(16, 32) for slot in range(4)}
        self.assertEqual(coordinator_module.pooled_draft_output_shapes(4, 16),
                         {**singles, (0, 1): spec(32, 32), (2, 3): spec(32, 32)})
        self.assertEqual(coordinator_module.pooled_draft_output_shapes(3, 16),
                         {**{(slot,): spec(16, 32) for slot in range(3)}, (0, 1): spec(32, 32)})
        with patch.dict(os.environ, QWEN_FAST_QUAD_DRAFT='1'):
            shapes = coordinator_module.pooled_draft_output_shapes(4, 16)
            self.assertEqual(shapes[(0, 1, 2, 3)], spec(64, 64))
            self.assertNotIn((0, 1, 2, 3), coordinator_module.pooled_draft_output_shapes(3, 16))
        with patch.dict(os.environ, QWEN_FAST_PACKED_PROPOSAL='0', QWEN_FAST_QUAD_DRAFT='1'):
            self.assertEqual(coordinator_module.pooled_draft_output_shapes(4, 16), singles,
                             'no pair or quad forms: the singles only')
        self.assertEqual(coordinator_module.pooled_draft_output_shapes(2, 8)[(1,)], spec(8, 32))


def pooled_lines(lines):
    return [line for line in lines if line.startswith('[PINDIAG] draft outputs')]


class PooledOutputTests(unittest.TestCase):
    """dflash_proposal_trace.borrow_pooled_outputs / pool_outputs / publish_outputs: a draft build over a pool that
    holds an output set for its slots copies its warm-up's outputs in (compiling the copies), records the copies at
    the end of its capture and hands the set to every reader; a set it cannot use is refused with the reason and the
    trace reads its own outputs, as before."""

    FLAGS = V86SequenceTests.FLAGS

    def setUp(self):
        V86SequenceTests.setUp(self)

    def pool(self, ops, outputs=None):
        from serving_buffer_pool import ServingBufferPool

        outputs = coordinator_module.pooled_draft_output_shapes(4, 16) if outputs is None else outputs
        return ServingBufferPool(ops, 'mesh', users=4, draft_outputs=outputs)

    def pair(self, ops, pool, *, rows=32):
        slots = [pool.acquire(owner='r%d' % index) for index in range(2)]
        devices = [V86SequenceTests.drafter(ops, slot, 5000 + index) for index, slot in enumerate(slots)]
        for device in devices:
            device.history_rows = 2048

            def execute_proposal(*args, rows=rows, **kwargs):
                outputs, write = head_outputs(ops, rows=rows)
                ops.on_replay(write)
                return outputs
            device.execute_proposal = execute_proposal
        return dflash_proposal_trace.PreparedPackedDFlashProposal(*devices), slots

    def one_round(self, trace):
        with patch('dflash_packed_proposal.select_device_outputs', return_value=((1,), (2,))):
            self.assertTrue(trace.prepare_device(11, 22))
            trace.finish('a', 1)
            trace.finish('b', 1)

    def test_the_pair_copies_its_outputs_into_the_set_in_its_warm_up_and_at_the_end_of_its_capture(self):
        ops = ReplayingHoleOps()
        pool = self.pool(ops)
        trace, _ = self.pair(ops, pool)
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            self.one_round(trace)
        bucket = trace.buckets[(2048, 2048)]
        held = pool.draft_outputs[(0, 1)]
        self.assertEqual([id(tensor) for tensor in output_tensors(bucket.outputs)],
                         [id(tensor) for tensor in output_tensors(held)])
        self.assertIsNot(bucket.outputs, held, 'a fresh namespace: no reader holds the pool record itself')
        targets = {ops.normalize(tensor) for tensor in output_tensors(held)}
        begin = next(index for index, event in enumerate(ops.events) if event[0] == 'begin_trace_capture')
        end = next(index for index, event in enumerate(ops.events) if event[0] == 'end_trace_capture')
        copies = [index for index, event in enumerate(ops.events) if event[0] == 'copy' and event[2] in targets]
        self.assertEqual(len([index for index in copies if index < begin]), 9, 'the warm-up: 4 x 2 chunks + 1')
        self.assertEqual(len([index for index in copies if begin < index < end]), 9, 'recorded at the capture')
        self.assertEqual(len(copies), 18, 'nothing outside the build copies: the replays run the recorded ones')
        # every replay rewrote the set: it holds the head's block on both chips
        for number, chunk in enumerate(held.chunks):
            for chip in chunk['indices'].chips:
                self.assertEqual(int(chip.max()), 105 + number)
        self.assertEqual(pooled_lines(lines),
                         ['[PINDIAG] draft outputs pooled slots=[0,1] head=1x1x32x16 projected=1x1x32x256'])
        trace.close()
        self.assertFalse([event for event in ops.events if event[0] == 'deallocate'
                          and event[1] in targets], 'the pool set is never freed by a trace')
        # the pair re-formed borrows the same set again
        again = dflash_proposal_trace.PreparedPackedDFlashProposal(trace.device_a, trace.device_b)
        with logged(dflash_proposal_trace)[1]:
            self.one_round(again)
        self.assertEqual([id(tensor) for tensor in output_tensors(again.buckets[(2048, 2048)].outputs)],
                         [id(tensor) for tensor in output_tensors(held)])

    def test_a_set_of_another_shape_is_refused_and_the_trace_reads_its_own_outputs(self):
        ops = ReplayingHoleOps()
        outputs = coordinator_module.pooled_draft_output_shapes(4, 16)
        outputs[(0, 1)] = dict(outputs[(0, 1)], head=(1, 1, 16, 16))
        pool = self.pool(ops, outputs)
        trace, _ = self.pair(ops, pool)
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            self.one_round(trace)
        bucket = trace.buckets[(2048, 2048)]
        self.assertFalse({id(tensor) for tensor in output_tensors(bucket.outputs)}
                         & {id(tensor) for tensor in output_tensors(pool.draft_outputs[(0, 1)])})
        self.assertEqual(pooled_lines(lines), ['[PINDIAG] draft outputs pooled refused slots=[0,1]: values0 shape '
                                               '1x1x32x16, the pool holds 1x1x16x16'])
        targets = {ops.normalize(tensor) for tensor in output_tensors(pool.draft_outputs[(0, 1)])}
        self.assertFalse([event for event in ops.events if event[0] == 'copy' and event[2] in targets])

    def test_a_pool_without_a_set_for_the_group_or_a_refused_copy_is_a_logged_refusal_never_a_failed_build(self):
        ops = ReplayingHoleOps()
        outputs = coordinator_module.pooled_draft_output_shapes(4, 16)
        del outputs[(0, 1)]
        trace, _ = self.pair(ops, self.pool(ops, outputs))
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            self.one_round(trace)
        self.assertEqual(pooled_lines(lines), ['[PINDIAG] draft outputs pooled refused slots=[0,1]: the pool holds '
                                               'sets for [[0], [1], [2], [2, 3], [3]]'])
        ops = ReplayingHoleOps()
        pool = self.pool(ops)
        trace, _ = self.pair(ops, pool)
        held = {ops.normalize(tensor) for tensor in output_tensors(pool.draft_outputs[(0, 1)])}
        original = ReplayingHoleOps.copy

        def refusing(self, source, destination):
            if ops.normalize(destination) in held:
                raise RuntimeError('ttnn.copy only supports ... inputs')
            return original(self, source, destination)

        lines, loguru = logged(dflash_proposal_trace)
        with loguru, patch.object(ReplayingHoleOps, 'copy', refusing):
            self.one_round(trace)
        self.assertEqual(pooled_lines(lines), ['[PINDIAG] draft outputs pooled refused slots=[0,1]: copy refused: '
                                               'RuntimeError: ttnn.copy only supports ... inputs'])
        self.assertFalse({id(tensor) for tensor in output_tensors(trace.buckets[(2048, 2048)].outputs)}
                         & {id(tensor) for tensor in output_tensors(pool.draft_outputs[(0, 1)])})

    def test_a_pool_without_output_sets_is_the_parents_pair_call_for_call(self):
        # The M0 pool (masks, no output sets): the pair makes exactly the calls it made before the output sets.
        from serving_buffer_pool import ServingBufferPool

        parent = pinned_module('dflash_proposal_trace.py', 'dflash_proposal_trace_v86_parent', commit=V86_PARENT)
        if parent is None:
            self.skipTest('no git history for %s' % V86_PARENT)

        def run(module):
            ops = ReplayingHoleOps()
            pool = ServingBufferPool(ops, 'mesh', users=4,
                                     draft_masks=coordinator_module.pooled_draft_mask_shapes(4, 16))
            slots = [pool.acquire(owner='r%d' % index) for index in range(2)]
            devices = [V86SequenceTests.drafter(ops, slot, 5000 + index) for index, slot in enumerate(slots)]
            for device in devices:
                device.history_rows = 2048
            trace = module.PreparedPackedDFlashProposal(*devices)
            with logged(dflash_proposal_trace)[1]:
                self.one_round(trace)
                self.one_round(trace)
            trace.close()
            return ops.events

        before = run(parent)
        self.assertGreater(len(before), 100)
        self.assertEqual(run(dflash_proposal_trace), before)

    def single(self, ops, slot, *, pooled_select):
        """A DFlashDevice at position 200 (one 256-row bucket) on `slot`, its head at the T16 16 rows; its
        select_proposal records the outputs it is handed."""
        kv_history = SimpleNamespace(active=[{'k': object(), 'v': object()} for _ in range(5)], pending=None,
                                     owned=[], borrowed=[], position=200, history_rows=200)
        device = SimpleNamespace(operations=ops, mesh='mesh', block_rows=16, position=200, history_rows=200,
                                 kv_history=kv_history, owned=[], history=None, spare_history=None, progress=None,
                                 live_query_qk=False, native_proposal_attention=False, pool_slot=slot,
                                 temporaries=lambda protected: ([], lambda value: value))

        def execute_proposal(*args, **kwargs):
            outputs, write = head_outputs(ops, rows=16, projected_rows=32)
            ops.on_replay(write)
            device.own.append(outputs)
            return outputs

        device.own = []
        device.execute_proposal = execute_proposal
        device.select_proposal = lambda outputs, seed, count: pooled_select.append(outputs) or (3,) * count
        return device

    def test_the_single_user_draft_copies_into_its_slots_set_and_finish_reads_it(self):
        from serving_buffer_pool import ServingBufferPool

        ops = ReplayingHoleOps()
        pool = ServingBufferPool(ops, 'mesh', users=4,
                                 draft_outputs=coordinator_module.pooled_draft_output_shapes(4, 16))
        slots = [pool.acquire(owner='r%d' % index) for index in range(3)]
        selected = []
        device = self.single(ops, slots[2], pooled_select=selected)
        lines, loguru = logged(dflash_proposal_trace)
        with loguru:
            capture = dflash_proposal_trace.PreparedDFlashProposal(device, max_new_tokens=1)
        self.assertEqual(list(capture.buckets), [256])
        held = pool.draft_outputs[(2,)]
        bucket = capture.buckets[256]
        self.assertEqual([id(tensor) for tensor in output_tensors(bucket.outputs)],
                         [id(tensor) for tensor in output_tensors(held)])
        self.assertEqual(pooled_lines(lines),
                         ['[PINDIAG] draft outputs pooled slots=[2] head=1x1x16x16 projected=1x1x32x256'])
        # the capture's own outputs clobbered after its replay: finish still reads the good block
        self.assertTrue(capture.prepare_device(5))
        captured = device.own[-1]
        for tensor in output_tensors(captured):
            tensor.chips = [torch.full_like(chip, float('nan')) if chip.is_floating_point()
                            else torch.full_like(chip, 1 << 20) for chip in tensor.chips]
        self.assertEqual(capture.finish(2), (3, 3))
        self.assertEqual([id(tensor) for tensor in output_tensors(selected[-1])],
                         [id(tensor) for tensor in output_tensors(held)])
        for tensor in output_tensors(held):
            for chip in tensor.chips:
                self.assertTrue(torch.isfinite(chip.float()).all())
        capture.close()
        targets = {ops.normalize(tensor) for tensor in output_tensors(held)}
        self.assertFalse([event for event in ops.events if event[0] == 'deallocate' and event[1] in targets])


class RejectedOutputsTests(unittest.TestCase):
    """S2 v86: a readback whose merge refuses the head outputs logs, per chip, what it read - non-finite, -inf and NaN
    counts, the index range and out-of-range count, whether the chips read equal, the projection's finiteness and
    equality, and every output buffer's address - then raises the same ValueError."""

    ERROR = V86SequenceTests.ERROR

    def outputs(self, ops, rows=32):
        outputs, _ = head_outputs(ops, rows=rows)
        chips = outputs.chunks[0]['values'].chips
        chips[0] = torch.full_like(chips[0], float('-inf'))
        chips = outputs.chunks[2]['indices'].chips
        chips[1] = chips[1].clone()
        chips[1][..., 3, :] = 40000
        return outputs

    def check(self, lines, outputs, *, slot, rows):
        self.assertEqual(len(lines), 2)
        addresses = ['[%s]' % ','.join('0x%x' % tensor.shards[chip].address for tensor in output_tensors(outputs))
                     for chip in range(2)]
        self.assertEqual(lines[0], '[PINDIAG] draft outputs rejected device_slot=%s chip=0 nonfinite=%d neg_inf=%d '
                                   'nan=0 index_min=0 index_max=108 index_out_of_range=0 chips_equal=values:3/4,'
                                   'indices:3/4 projected_finite=1 projected_equal=1 addresses=%s '
                                   'error=Finite_complete-block_top16_values_and_in-range_integer_indices_required'
                         % (slot, rows * 16, rows * 16, addresses[0]))
        self.assertEqual(lines[1], '[PINDIAG] draft outputs rejected device_slot=%s chip=1 nonfinite=0 neg_inf=0 '
                                   'nan=0 index_min=0 index_max=40000 index_out_of_range=16 chips_equal=values:3/4,'
                                   'indices:3/4 projected_finite=1 projected_equal=1 addresses=%s '
                                   'error=Finite_complete-block_top16_values_and_in-range_integer_indices_required'
                         % (slot, addresses[1]))

    def test_the_pair_readback_reports_each_chip_then_raises_the_same_error(self):
        import dflash_packed_proposal

        for read in (lambda device, outputs: dflash_packed_proposal.read_device_outputs(device, outputs, 2, 16),
                     lambda device, outputs: dflash_packed_proposal.select_device_outputs(device, outputs, (1, 2),
                                                                                          (15, 15), 2, 16)):
            ops = RecordingOps()
            ops.uint16 = 'u16'
            outputs = self.outputs(ops)
            device = SimpleNamespace(operations=ops, pool_slot=SimpleNamespace(index=2), predecessors=None,
                                     successors=None)
            lines, loguru = logged(dflash_packed_proposal)
            with loguru, self.assertRaises(ValueError) as raised:
                read(device, outputs)
            self.assertEqual(str(raised.exception), self.ERROR)
            self.check(lines, outputs, slot=2, rows=32)

    def test_the_single_users_selection_reports_each_chip_then_raises_the_same_error(self):
        from dflash_device import DFlashDevice

        ops = RecordingOps()
        ops.uint16 = 'u16'
        outputs = self.outputs(ops, rows=16)
        device = SimpleNamespace(operations=ops, block_rows=16, pool_slot=SimpleNamespace(index=1))
        lines, loguru = logged(dflash_proposal_trace)
        with loguru, self.assertRaises(ValueError) as raised:
            DFlashDevice.select_proposal(device, outputs, 5, 3)
        self.assertEqual(str(raised.exception), self.ERROR)
        self.check(lines, outputs, slot=1, rows=16)

    def test_healthy_outputs_log_nothing_and_a_broken_diagnostic_never_replaces_the_error(self):
        import dflash_packed_proposal

        ops = RecordingOps()
        ops.uint16 = 'u16'
        good, _ = head_outputs(ops)
        device = SimpleNamespace(operations=ops, pool_slot=None)
        lines, loguru = logged(dflash_packed_proposal)
        with loguru:
            parts = dflash_packed_proposal.read_device_outputs(device, good, 2, 16)
        self.assertEqual(len(parts), 2)
        self.assertEqual(lines, [])
        bad = self.outputs(ops)
        bad.projected = 'not a tensor'
        with loguru, self.assertRaises(ValueError) as raised:
            dflash_packed_proposal.read_device_outputs(device, bad, 2, 16)
        self.assertEqual(str(raised.exception), self.ERROR)
        self.assertEqual(len(lines), 2)
        self.assertIn('device_slot=None chip=0 ', lines[0])
        self.assertIn('projected_finite=unavailable:AttributeError projected_equal=unavailable:AttributeError',
                      lines[0])
        self.assertIn('addresses=unavailable:AttributeError', lines[0])


class ShippingTests(unittest.TestCase):
    def test_the_changed_modules_reach_the_image(self):
        from test_serving_image_copy_closure import copied_modules, dockerfile_text

        copied = copied_modules(dockerfile_text())
        for name in ('dflash_proposal_trace.py', 'dflash_packed_proposal_coordinator.py', 'serving_worker_hook.py'):
            with self.subTest(module=name):
                self.assertIn(name, copied)


if __name__ == '__main__':
    unittest.main()
