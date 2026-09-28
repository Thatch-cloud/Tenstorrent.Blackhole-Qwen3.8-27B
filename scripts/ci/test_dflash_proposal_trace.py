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
refresh's copy are checked by value, not only by call."""

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


class ShippingTests(unittest.TestCase):
    def test_the_changed_modules_reach_the_image(self):
        from test_serving_image_copy_closure import copied_modules, dockerfile_text

        copied = copied_modules(dockerfile_text())
        for name in ('dflash_proposal_trace.py', 'dflash_packed_proposal_coordinator.py', 'serving_worker_hook.py'):
            with self.subTest(module=name):
                self.assertIn(name, copied)


if __name__ == '__main__':
    unittest.main()
