"""tp4/u1 on the CPU: the packed verify's all-reduce as ONE reduce-scatter on the unit-major view (QWEN_FAST_TP4_RS_UNIT_MAJOR),
its refusals, its audit, the profiles and the job pack. Nothing here ran on four cards: the exactness argument is the X1 spike's
(docs/tp4-exact-ring-parity.md) and the audit arm is what proves it again on the serving image."""

import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import c2_smoke_check  # noqa: E402
import tile_collective_tp as collective  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
PACK = HERE / 'references' / 'tp4-u1-jobs'
CHIPS = 4
WIDTH = 5120
SHARD = WIDTH // CHIPS
FOUR = {'QWEN_FAST_TP': '4'}


class Memory:
    def __init__(self, name, sharded=False):
        self.name, self.sharded = name, sharded

    def is_sharded(self):
        return self.sharded

    def __eq__(self, other):
        return isinstance(other, Memory) and (self.name, self.sharded) == (other.name, other.sharded)

    def __hash__(self):
        return hash((self.name, self.sharded))

    def __repr__(self):
        return 'Memory(%s)' % self.name


DRAM, L1, SHARDED = Memory('DRAM'), Memory('L1'), Memory('L1_SHARDED', True)


class Tensor:
    """Four chips' partials (torch, one per chip) with a buffer id: views share it."""
    counter = 0

    def __init__(self, chips, memory=DRAM, dtype='bf16', layout='tile', buffer=None):
        self.chips, self.memory, self.dtype, self.layout = chips, memory, dtype, layout
        if buffer is None:
            Tensor.counter += 1
            buffer = Tensor.counter
        self.buffer = buffer
        self.freed = False

    @property
    def shape(self):
        return tuple(self.chips[0].shape)

    def memory_config(self):
        return self.memory

    def buffer_address(self):
        return self.buffer


def partials(rows, seed=0, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 1, rows, width, generator=generator) * torch.exp(torch.randn(1, 1, rows, width, generator=generator))
             ).to(torch.bfloat16) for _ in range(CHIPS)]


def reduce_scatter(chips, units_major):
    """The sum of the chips' partials, chip k keeping its columns. The ring's order is not modelled: both paths use this one sum
    unless `units_major` is wrapped by a test."""
    total = sum(chip.to(torch.float32) for chip in chips).to(torch.bfloat16)
    width = total.shape[-1] // CHIPS
    return [total[..., k * width:(k + 1) * width].contiguous() for k in range(CHIPS)]


class Mesh:
    shape = (1, CHIPS)

    def get_num_devices(self):
        return CHIPS


class Collective:
    def __init__(self, links=2):
        self.links, self.cycles = links, 0

    def get_num_links(self, axis):
        return self.links

    def get_and_cycle_rs_semaphore_handles(self):
        self.cycles += 1
        return ('rs', self.cycles)

    def get_and_cycle_barrier_semaphore_handle(self):
        return ('barrier',)


class Operations:
    bfloat16, float32, TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'bf16', 'f32', 'tile', 'row'
    DRAM_MEMORY_CONFIG = DRAM

    class Topology:
        Ring, Linear = 'Ring', 'Linear'

    def __init__(self, bad_unit_major=False, copying_reshape=False):
        self.calls = []
        self.bad_unit_major = bad_unit_major
        self.copying_reshape = copying_reshape
        self.experimental = types.SimpleNamespace(reduce_scatter_minimal_async=self.reduce_scatter_minimal_async)
        self.live = set()

    def made(self, tensor):
        self.live.add(id(tensor))
        return tensor

    def reshape(self, tensor, shape):
        assert not tensor.freed
        self.calls.append(('reshape', tuple(tensor.shape), tuple(shape)))
        chips = [chip.reshape(*shape) for chip in tensor.chips]
        if self.copying_reshape:
            return self.made(Tensor(chips, tensor.memory, tensor.dtype, tensor.layout))
        return self.made(Tensor(chips, tensor.memory, tensor.dtype, tensor.layout, buffer=tensor.buffer))

    def reduce_scatter_minimal_async(self, tensor, **kwargs):
        assert not tensor.freed
        self.calls.append(('reduce_scatter', tuple(tensor.shape), kwargs))
        out = reduce_scatter(tensor.chips, True)
        if self.bad_unit_major and len(tensor.shape) == 4 and tensor.shape[1] > 1:
            out[1] = out[1].clone()
            out[1][0, 0, 0, 0] = out[1][0, 0, 0, 0] + 1
        return self.made(Tensor(out, kwargs['memory_config']))

    def slice(self, tensor, start, stop, **kwargs):
        assert not tensor.freed
        self.calls.append(('slice', start[2], stop[2]))
        return self.made(Tensor([chip[:, :, start[2]:stop[2], :].contiguous() for chip in tensor.chips],
                                kwargs.get('memory_config', tensor.memory), tensor.dtype, tensor.layout))

    def concat(self, tensors, dim, **kwargs):
        assert dim == 2 and not any(tensor.freed for tensor in tensors)
        self.calls.append(('concat', len(tensors)))
        return self.made(Tensor([torch.cat([tensor.chips[k] for tensor in tensors], dim=2) for k in range(CHIPS)],
                                kwargs.get('memory_config', tensors[0].memory), tensors[0].dtype, tensors[0].layout))

    def clone(self, tensor, memory_config=None):
        assert not tensor.freed
        self.calls.append(('clone', tuple(tensor.shape)))
        return self.made(Tensor([chip.clone() for chip in tensor.chips], memory_config or tensor.memory, tensor.dtype, tensor.layout))

    def deallocate(self, tensor):
        assert not tensor.freed, 'freed twice'
        self.calls.append(('deallocate', tuple(tensor.shape)))
        tensor.freed = True

    def get_device_tensors(self, tensor):
        return list(tensor.chips)

    def to_torch(self, chip):
        return chip

    def names(self):
        return [call[0] for call in self.calls]


class ModelAllReduce:
    """tt_all_reduce as far as bits go: the reduce-scatter on the flattened tensor, consuming the input (ccl.py)."""

    def __init__(self, operations):
        self.operations = operations
        self.calls = []

    def __call__(self, tensor, mesh, ccl, cluster_axis=0, dim=3, topology='Ring', memory_config=DRAM):
        assert not tensor.freed
        self.calls.append((tuple(tensor.shape), cluster_axis, dim, topology, memory_config))
        out = self.operations.made(Tensor(reduce_scatter(tensor.chips, False), memory_config))
        tensor.freed = True
        return out


class Fixture(unittest.TestCase):
    def setUp(self):
        collective._HELD.clear()
        collective._STATE['reasons'].clear()
        collective._STATE['owners'] = 0
        collective._STATE['replayed'] = None
        self.operations = Operations()
        self.model = ModelAllReduce(self.operations)
        self.wrapper = collective.TileSplitAllReduce(self.model, self.operations)
        self.mesh, self.collective = Mesh(), Collective()
        self.options = dict(cluster_axis=0, dim=3, topology='Ring', memory_config=DRAM)

    def call(self, tensor, **changes):
        options = dict(self.options, **changes)
        return self.wrapper(tensor, self.mesh, self.collective, **options)

    def scope(self, rows=64, **options):
        return collective.block_scope(rows, unit_major=True, **options)


class SettingsTests(unittest.TestCase):
    def test_the_lever_is_off_by_default_and_strict(self):
        self.assertEqual(collective.unit_major_settings({}), (False, 0))
        self.assertEqual(collective.unit_major_settings({'QWEN_FAST_TP4_RS_UNIT_MAJOR': '0'}), (False, 0))
        self.assertEqual(collective.unit_major_settings(dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR='1')), (True, 0))
        for bad in ('true', '2', '', 'yes'):
            with self.assertRaises(ValueError, msg=bad):
                collective.unit_major_settings(dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR=bad))

    def test_the_audit_needs_the_lever_and_defaults_to_32_calls(self):
        self.assertEqual(collective.unit_major_settings(dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR='1',
                                                             QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT='1')), (True, 32))
        with self.assertRaises(ValueError):
            collective.unit_major_settings(dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT='1'))
        env = dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR='1', QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT='1')
        self.assertEqual(collective.unit_major_settings(dict(env, QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT_CALLS='5')), (True, 5))
        for bad in ('0', '-1', 'x', '05', ''):
            with self.assertRaises(ValueError, msg=bad):
                collective.unit_major_settings(dict(env, QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT_CALLS=bad))
        with self.assertRaises(ValueError):
            collective.unit_major_settings(dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR='1', QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT_CALLS='5'))

    def test_the_pair_refuses_every_flag(self):
        for env in ({'QWEN_FAST_TP4_RS_UNIT_MAJOR': '1'}, {'QWEN_FAST_TP': '2', 'QWEN_FAST_TP4_RS_UNIT_MAJOR': '1'}):
            with self.assertRaises(ValueError):
                collective.unit_major_settings(env)

    def test_the_census_is_what_the_x1_spike_measured(self):
        self.assertEqual(collective.CENSUS_ROWS, (64,))        # 128 has one X1 seed and no audit line: it falls back
        self.assertEqual((collective.CENSUS_WIDTH, collective.CENSUS_CHIPS, collective.CENSUS_LINKS), (5120, 4, 2))


class ViewArithmeticTests(Fixture):
    def test_a_64_row_call_is_one_reduce_scatter_on_the_two_unit_view_and_a_view_back(self):
        out = None
        with self.scope():
            out = self.call(Tensor(partials(64)))
        self.assertEqual(self.operations.names(), ['reshape', 'reduce_scatter', 'reshape', 'deallocate'])
        self.assertEqual(self.operations.calls[0][1:], ((1, 1, 64, 5120), (1, 2, 32, 5120)))
        self.assertEqual(self.operations.calls[1][1], (1, 2, 32, 5120))
        self.assertEqual(self.operations.calls[2][1:], ((1, 2, 32, 1280), (1, 1, 64, 1280)))
        self.assertEqual(out.shape, (1, 1, 64, SHARD))
        self.assertEqual(self.model.calls, [])

    def test_a_128_row_call_is_not_proven_and_falls_back_by_name(self):
        tensor = Tensor(partials(128))
        self.assertIn('rows 128', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective), self.options))

    def test_the_reduce_scatter_is_the_spikes_call(self):
        with self.scope():
            self.call(Tensor(partials(64)))
        kwargs = self.operations.calls[1][2]
        self.assertEqual(kwargs, dict(persistent_output_buffers=None, dim=3, multi_device_global_semaphore=('rs', 1),
                                      barrier_semaphore=('barrier',), num_links=2, memory_config=DRAM,
                                      intermediate_memory_config=DRAM, topology='Ring', chunks_per_sync=10,
                                      num_workers_per_link=2, num_buffers_per_channel=2))

    def test_the_output_is_the_split_paths_bytes_and_memory_config(self):
        source = partials(64, seed=3)
        with self.scope():
            mine = self.call(Tensor([chip.clone() for chip in source]))
        with collective.block_scope(64):
            served = self.call(Tensor([chip.clone() for chip in source]))
        self.assertEqual(mine.shape, served.shape)
        self.assertEqual((mine.dtype, mine.layout, mine.memory_config()), (served.dtype, served.layout, served.memory_config()))
        for left, right in zip(mine.chips, served.chips):
            self.assertTrue(torch.equal(left.view(torch.int16), right.view(torch.int16)))

    def test_the_output_follows_the_requested_memory_config_and_the_inputs_by_default(self):
        with self.scope():
            out = self.call(Tensor(partials(64), memory=L1), memory_config=L1)
            self.assertEqual(out.memory_config(), L1)
        self.assertEqual(self.operations.calls[1][2]['intermediate_memory_config'], DRAM)

    def test_a_view_is_not_freed_twice_and_the_input_is_consumed_once(self):
        source = Tensor(partials(64))
        with self.scope():
            self.call(source)
        self.assertTrue(source.freed)
        self.assertEqual(self.operations.names().count('deallocate'), 1)

    def test_a_reshape_that_copies_frees_its_copies(self):
        self.operations = Operations(copying_reshape=True)
        self.model = ModelAllReduce(self.operations)
        self.wrapper = collective.TileSplitAllReduce(self.model, self.operations)
        source = Tensor(partials(64))
        with self.scope():
            self.call(source)
        self.assertEqual(self.operations.names().count('deallocate'), 3)    # the view copy, the input, the scattered copy

    def test_the_semaphores_cycle_once_per_all_reduce(self):
        with self.scope():
            self.call(Tensor(partials(64)))
            self.call(Tensor(partials(64)))
        self.assertEqual(self.collective.cycles, 2)


class RefusalTests(Fixture):
    def refused(self, tensor, **changes):
        before = list(self.operations.calls)
        with self.scope(tensor.shape[2]):
            self.call(tensor, **changes)
        # the split ran in its place: slices, per-tile model calls, a concat
        self.assertIn('concat', self.operations.names()[len(before):])
        self.assertEqual(len(self.model.calls), tensor.shape[2] // 32)

    def reasons(self):
        return sorted(collective._STATE['reasons'])

    def test_each_call_outside_the_census_falls_back_to_the_split_by_name(self):
        cases = {
            'rows 96': (lambda: Tensor(partials(96)), {}, 'rows 96'),
            'width': (lambda: Tensor(partials(64, width=2560)), {}, 'width 2560'),
            'dtype': (lambda: Tensor(partials(64), dtype='f32'), {}, 'dtype'),
            'layout': (lambda: Tensor(partials(64), layout='row'), {}, 'layout'),
            'sharded input': (lambda: Tensor(partials(64), memory=SHARDED), {}, 'sharded'),
            'sharded output': (lambda: Tensor(partials(64)), dict(memory_config=SHARDED), 'sharded'),
            'topology': (lambda: Tensor(partials(64)), dict(topology='Linear'), 'topology'),
            'axis': (lambda: Tensor(partials(64)), dict(cluster_axis=1), 'cluster_axis'),
            'dim': (lambda: Tensor(partials(64)), dict(dim=2), 'dim'),
        }
        for label, (build, changes, word) in cases.items():
            with self.subTest(label):
                self.setUp()
                with patch.object(collective, '_log') as log:
                    self.refused(build(), **changes)
                self.assertEqual(len(self.reasons()), 1, self.reasons())
                self.assertIn(word, self.reasons()[0])
                self.assertIn('fell back', log.call_args[0][0])
                self.assertIn('reason=', log.call_args[0][0])
                self.assertNotIn('reduce_scatter', [call[0] for call in self.operations.calls])

    def test_an_unknown_keyword_and_a_positional_argument_are_refused(self):
        with patch.object(collective, '_log'), self.scope():
            self.wrapper(Tensor(partials(64)), self.mesh, self.collective, cluster_axis=0, dim=3, topology='Ring',
                         memory_config=DRAM)
            self.assertEqual(collective._STATE['fallbacks'], 0)
            tensor = Tensor(partials(64))
            self.assertIn('keyword num_links', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective),
                                                                    dict(self.options, num_links=1)))
            self.assertIn('positional', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective, 0),
                                                              self.options))
            tensor = Tensor(partials(64))
            self.assertIn('topology', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective),
                                                           dict(cluster_axis=0, dim=3, memory_config=DRAM)))

    def test_the_wrong_mesh_and_the_wrong_link_count_are_refused(self):
        tensor = Tensor(partials(64))
        wrong_mesh = types.SimpleNamespace(get_num_devices=lambda: 2)
        self.assertIn('4 chips', self.wrapper.refusal(tensor, tensor.shape, (wrong_mesh, self.collective), self.options))
        self.assertIn('2 links', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, Collective(links=1)), self.options))
        self.assertIsNone(self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective), self.options))
        for shape in ((2, 2), (4, 1), None):
            odd_mesh = types.SimpleNamespace(get_num_devices=lambda: 4, shape=shape)
            self.assertIn('mesh shape', self.wrapper.refusal(tensor, tensor.shape, (odd_mesh, self.collective), self.options))
        broken = types.SimpleNamespace(get_num_links=lambda axis: 1 / 0)
        self.assertIn('could not say', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, broken), self.options))

    def test_an_omitted_dim_is_refused_because_the_model_defaults_it_to_zero(self):
        tensor = Tensor(partials(64))
        options = dict(cluster_axis=0, topology='Ring', memory_config=DRAM)
        self.assertIn('dim None', self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective), options))

    def test_the_census_shapes_are_the_only_unit_major_ones(self):
        for rows in (32, 64, 96, 128, 160):
            tensor = Tensor(partials(rows)) if rows >= 32 else None
            reason = self.wrapper.refusal(tensor, tensor.shape, (self.mesh, self.collective), self.options)
            self.assertEqual(reason is None, rows in (64,), (rows, reason))

    def test_a_reason_is_logged_once_per_scope_family_and_every_call_counts(self):
        with patch.object(collective, '_log') as log:
            with self.scope(96):
                for _ in range(3):
                    self.call(Tensor(partials(96)))
        self.assertEqual(log.call_count, 1)
        self.assertEqual(collective._STATE['fallbacks'], 3)


class FlagOffIdentityTests(Fixture):
    def test_without_the_flag_the_split_runs_as_it_always_did(self):
        with collective.block_scope(64):
            out = self.call(Tensor(partials(64)))
        self.assertEqual(self.operations.names().count('slice'), 2)
        self.assertEqual(self.operations.names().count('concat'), 1)
        self.assertNotIn('reshape', self.operations.names())
        self.assertNotIn('reduce_scatter', self.operations.names())
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(out.shape, (1, 1, 64, SHARD))

    def test_the_default_scope_state_is_off(self):
        with collective.block_scope(64):
            self.assertFalse(collective._STATE['unit_major'])
        self.assertFalse(collective._STATE['unit_major'])

    def test_scope_for_without_the_flag_passes_unit_major_false(self):
        batch = types.SimpleNamespace(rows=64, model=types.SimpleNamespace(layers=[0] * 2), native_m3=True)
        with patch.dict(os.environ, {}, clear=False), patch.object(collective, 'block_scope') as scope, \
                patch.dict(sys.modules, {'dflash_device': types.SimpleNamespace(pindiag=lambda *a: None)}):
            os.environ.pop('QWEN_FAST_TP4_RS_UNIT_MAJOR', None)
            collective.scope_for(batch)
        self.assertEqual(scope.call_args[1]['unit_major'], False)
        self.assertEqual(scope.call_args[1]['audit_calls'], 0)

    def test_scope_for_with_the_flags_passes_them(self):
        batch = types.SimpleNamespace(rows=64, model=types.SimpleNamespace(layers=[0] * 2), native_m3=True)
        env = dict(FOUR, QWEN_FAST_TP4_RS_UNIT_MAJOR='1', QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT='1')
        with patch.dict(os.environ, env), patch.object(collective, 'block_scope') as scope, \
                patch.dict(sys.modules, {'dflash_device': types.SimpleNamespace(pindiag=lambda *a: None)}):
            collective.scope_for(batch)
        self.assertEqual((scope.call_args[1]['unit_major'], scope.call_args[1]['audit_calls']), (True, 32))
        self.assertEqual(scope.call_args[1]['expected'], 4)


class GuardTests(Fixture):
    def test_the_engaged_marker_counts_unit_major_calls_and_fallbacks(self):
        lines = []
        with collective.block_scope(64, expected=3, log=lambda template, *values: lines.append(template.format(*values)),
                                    unit_major=True), patch.object(collective, '_log'):
            self.call(Tensor(partials(64)))
            self.call(Tensor(partials(64)))
            self.call(Tensor(partials(64)), topology='Linear')
        marker = [line for line in lines if line.startswith(collective.ENGAGED_MARKER)]
        self.assertEqual(marker, ['[PINDIAG] tp4 u1 engaged rows=64 calls=3 unit_major=2 fallbacks=1 audited=0'])

    def test_no_marker_without_the_lever(self):
        lines = []
        with collective.block_scope(64, expected=1, log=lambda template, *values: lines.append(template.format(*values))):
            self.call(Tensor(partials(64)))
        self.assertFalse([line for line in lines if collective.ENGAGED_MARKER in line])

    def test_a_scope_that_reduced_fewer_than_expected_is_refused_with_the_lever_on_too(self):
        with self.assertRaises(AssertionError):
            with collective.block_scope(64, expected=2, unit_major=True):
                self.call(Tensor(partials(64)))

    def test_the_audit_needs_the_lever(self):
        with self.assertRaises(ValueError):
            with collective.block_scope(64, audit_calls=4):
                pass

    def test_an_odd_reduce_scatter_count_is_refused_when_the_scope_states_its_expectation(self):
        with patch.object(collective, '_log'):
            with self.assertRaises(AssertionError) as raised:
                with collective.block_scope(64, expected=127 + 1, unit_major=True):
                    for _ in range(127):
                        self.call(Tensor(partials(64)))
                    self.call(Tensor(partials(64)), topology='Linear')       # one fallback: 127 + 2 reduce-scatters
            self.assertIn('odd', str(raised.exception))
            with collective.block_scope(64, expected=4, unit_major=True):
                for _ in range(2):
                    self.call(Tensor(partials(64)))
                for _ in range(2):
                    self.call(Tensor(partials(64)), topology='Linear')       # two fallbacks: 2 + 4, even

    def test_the_state_is_clean_after_a_scope(self):
        with self.scope():
            self.call(Tensor(partials(64)))
        self.assertEqual((collective._STATE['rows'], collective._STATE['unit_major'], collective._STATE['audit_calls']),
                         (None, False, 0))
        out = self.call(Tensor(partials(64)))                       # outside any scope: the model's own call
        self.assertEqual(out.shape, (1, 1, 64, SHARD))
        self.assertEqual(len(self.model.calls), 1)


class AuditTests(Fixture):
    def hold(self, calls, rows=64, audit=2):
        with patch.object(collective, '_log'), collective.block_scope(rows, unit_major=True, audit_calls=audit):
            return [self.call(Tensor(partials(rows, seed=seed))) for seed in range(calls)]

    def test_only_the_first_calls_per_shape_are_audited_and_the_split_result_is_served(self):
        outputs = self.hold(4)
        self.assertEqual(len(collective._HELD), 2)
        self.assertEqual(len(self.model.calls), 2 * 2)                   # two audited calls x two tile calls
        self.assertEqual(self.operations.names().count('reduce_scatter'), 4)  # two audited (+ their splits' none) and two plain
        self.assertEqual(self.collective.cycles, 4)
        for pair in collective._HELD:
            self.assertEqual(pair['shape'], (64, 5120))
            self.assertIsNone(pair['owner'])
        self.assertEqual([out.shape for out in outputs], [(1, 1, 64, SHARD)] * 4)

    def test_the_marker_counts_the_audited_calls(self):
        lines = []
        with patch.object(collective, '_log'), collective.block_scope(
                64, expected=4, unit_major=True, audit_calls=2, log=lambda template, *values: lines.append(template.format(*values))):
            for seed in range(4):
                self.call(Tensor(partials(64, seed=seed)))
        self.assertIn('[PINDIAG] tp4 u1 engaged rows=64 calls=4 unit_major=4 fallbacks=0 audited=2', lines)

    def test_a_pair_is_claimed_compared_and_released(self):
        self.hold(3)
        owner = object()
        self.assertEqual(collective.audit_claim(owner, 'capture'), 2)
        self.assertEqual(collective.audit_claim(object()), 0)
        collective.audit_replayed(owner)
        lines = []
        self.assertEqual(collective.audit_round(self.operations, owner, 1, log=lines.append), 2)
        self.assertEqual(lines, ['[PINDIAG] tp4 u1 audit shape=64x5120 owner=capture1 round=1 calls=2 chips=4 elements=%d exact=True'
                                 % (2 * CHIPS * 64 * SHARD)])
        lines.clear()
        collective.audit_round(self.operations, owner, 7, log=lines.append)
        self.assertEqual(lines, [])                                      # logged on rounds 0-3 and every 50th
        collective.audit_round(self.operations, owner, 50, log=lines.append)
        self.assertEqual(len(lines), 1)
        held = [tensor for pair in collective._HELD for tensor in (pair['mine'], pair['served'])]
        self.assertEqual(collective.audit_release(self.operations, owner), 2)
        self.assertTrue(all(tensor.freed for tensor in held))
        self.assertEqual(collective._HELD, [])

    def test_two_blocks_are_labelled_apart_and_each_is_read_only_after_its_own_replay(self):
        self.hold(1)
        first, second = object(), object()
        collective.audit_claim(first, 'capture')
        self.hold(1)
        collective.audit_claim(second, 'capture')
        lines = []
        collective.audit_replayed(first)
        collective.audit_replayed(second)                                # the second block replayed after the first
        with self.assertRaises(AssertionError) as raised:
            collective.audit_round(self.operations, first, 1, log=lines.append)
        self.assertIn('another block replayed', str(raised.exception))
        collective.audit_replayed(first)
        collective.audit_round(self.operations, first, 1, log=lines.append)
        collective.audit_replayed(second)
        collective.audit_round(self.operations, second, 1, log=lines.append)
        self.assertEqual([line.split(' owner=')[1].split(' ')[0] for line in lines], ['capture1', 'capture2'])
        collective.audit_round(self.operations, first, 0, log=lines.append)      # the warm forward is not a replay

    def test_an_owner_with_nothing_held_compares_nothing(self):
        self.assertEqual(collective.audit_round(self.operations, object(), 1), 0)
        self.assertEqual(collective.audit_release(self.operations, object()), 0)

    def test_one_flipped_bit_on_one_chip_is_a_mismatch(self):
        self.operations = Operations(bad_unit_major=True)
        self.model = ModelAllReduce(self.operations)
        self.wrapper = collective.TileSplitAllReduce(self.model, self.operations)
        self.hold(1)
        owner = object()
        collective.audit_claim(owner)
        collective.audit_replayed(owner)
        lines = []
        with self.assertRaises(AssertionError) as raised:
            collective.audit_round(self.operations, owner, 1, log=lines.append)
        self.assertIn('[PINDIAG] tp4 u1 audit mismatch', str(raised.exception))
        self.assertIn('chip 1: 1 of %d elements differ' % (64 * SHARD), str(raised.exception))
        self.assertEqual(lines, [str(raised.exception)])

    def test_plus_and_minus_zero_differ(self):
        self.hold(1)
        owner = object()
        collective.audit_claim(owner)
        collective.audit_replayed(owner)
        pair = collective._HELD[0]
        pair['mine'].chips[0][0, 0, 0, 0] = 0.0
        pair['served'].chips[0][0, 0, 0, 0] = -0.0
        with self.assertRaises(AssertionError):
            collective.audit_round(self.operations, owner, 1, log=lambda text: None)

    def test_a_layout_difference_is_a_mismatch(self):
        self.hold(1)
        owner = object()
        collective.audit_claim(owner)
        collective.audit_replayed(owner)
        collective._HELD[0]['layout'] = 'memory DRAM against L1'
        with self.assertRaises(AssertionError) as raised:
            collective.audit_round(self.operations, owner, 1, log=lambda text: None)
        self.assertIn('layout memory DRAM against L1', str(raised.exception))

    def test_the_layout_check_sees_the_consumers_view(self):
        self.hold(1)
        self.assertIsNone(collective._HELD[0]['layout'])
        left = collective._view_of(Tensor(partials(64)))
        right = collective._view_of(Tensor(partials(64), memory=L1))
        self.assertIn('memory', collective._differences(left, right))
        self.assertEqual(collective._differences(left, left), '')

    def test_every_forward_audits_its_own_first_calls(self):
        with patch.object(collective, '_log'):
            for _ in range(2):
                with collective.block_scope(64, unit_major=True, audit_calls=1):
                    for seed in range(3):
                        self.call(Tensor(partials(64, seed=seed)))
        self.assertEqual(len(collective._HELD), 2)

    def test_an_audit_of_a_refused_call_holds_nothing(self):
        with patch.object(collective, '_log'), collective.block_scope(96, unit_major=True, audit_calls=4):
            self.call(Tensor(partials(96)))
        self.assertEqual(collective._HELD, [])

    def test_a_failing_split_frees_the_unit_major_clone(self):
        def broken(*args, **kwargs):
            raise RuntimeError('tile call failed')

        self.wrapper = collective.TileSplitAllReduce(broken, self.operations)
        with self.assertRaises(RuntimeError):
            with patch.object(collective, '_log'), collective.block_scope(64, unit_major=True, audit_calls=4):
                self.call(Tensor(partials(64)))
        self.assertEqual(collective._HELD, [])


class SmokeRuleTests(unittest.TestCase):
    ENGAGED = '[PINDIAG] tp4 u1 engaged rows=64 calls=128 unit_major=128 fallbacks=0 audited=0'
    AUDIT = '[PINDIAG] tp4 u1 audit shape=64x5120 owner=capture3 round=1 calls=32 chips=4 elements=1 exact=True'

    def problems(self, env, *lines):
        return c2_smoke_check.u1_problems(env, '\n'.join(lines))

    def test_a_profile_without_the_flag_is_untouched(self):
        self.assertEqual(self.problems({}, 'nothing'), [])
        self.assertEqual(self.problems(None, 'nothing'), [])
        self.assertTrue(self.problems({}, self.ENGAGED))

    def test_the_flag_needs_the_engaged_marker_with_unit_major_calls(self):
        env = {c2_smoke_check.U1_FLAG: '1'}
        self.assertEqual(self.problems(env, self.ENGAGED), [])
        self.assertTrue(self.problems(env, 'nothing'))
        self.assertTrue(self.problems(env, self.ENGAGED.replace('unit_major=128', 'unit_major=0')))

    def test_a_fall_back_line_fails(self):
        env = {c2_smoke_check.U1_FLAG: '1'}
        self.assertTrue(self.problems(env, self.ENGAGED, '[PINDIAG] tp4 u1 fell back rows=96 reason=rows 96'))

    def test_the_audit_flag_needs_an_exact_line_for_every_served_shape(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1'}
        self.assertEqual(self.problems(env, self.ENGAGED, self.AUDIT), [])
        self.assertTrue(self.problems(env, self.ENGAGED))
        self.assertTrue(self.problems(env, self.ENGAGED, self.AUDIT.replace('64x5120', '128x5120')))

    def test_a_warm_forward_line_alone_does_not_pass(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1'}
        warm = self.AUDIT.replace('round=1', 'round=0').replace('capture3', 'warm1')
        found = self.problems(env, self.ENGAGED, warm)
        self.assertTrue(any('no replay was compared' in text for text in found), found)

    def test_an_audit_line_that_compared_nothing_fails(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1'}
        for change in (('chips=4', 'chips=0'), ('elements=1', 'elements=0'), ('chips=4', 'chips=2')):
            with self.subTest(change):
                self.assertTrue(any('compared nothing' in text for text in
                                    self.problems(env, self.ENGAGED, self.AUDIT.replace(*change))))

    def test_every_block_needs_its_own_replay_audit(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1', 'QWEN_FAST_M3_BLOCKS': '2'}
        self.assertTrue(any('1 block owner' in text for text in self.problems(env, self.ENGAGED, self.AUDIT)))
        self.assertEqual(self.problems(env, self.ENGAGED, self.AUDIT, self.AUDIT.replace('capture3', 'capture4')), [])

    def test_the_served_shape_follows_the_engaged_rows(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1'}
        engaged = self.ENGAGED.replace('rows=64', 'rows=128')
        self.assertTrue(any('128x5120' in text for text in self.problems(env, engaged, self.AUDIT)))

    def test_a_mismatch_line_fails_and_is_not_an_audit_line(self):
        env = {c2_smoke_check.U1_FLAG: '1', c2_smoke_check.U1_AUDIT_FLAG: '1'}
        mismatch = '[PINDIAG] tp4 u1 audit mismatch round=1 shape=64x5120 call=0 chip 1: 1 of 2 elements differ exact=True'
        found = self.problems(env, self.ENGAGED, mismatch)
        self.assertTrue(any('difference' in text for text in found))
        self.assertTrue(any('no replay was compared' in text for text in found))

    def test_check_wires_the_rule_in(self):
        problems, _ = c2_smoke_check.check('', 'nothing', True, env={c2_smoke_check.U1_FLAG: '1', 'QWEN_FAST_TP': '4'})
        self.assertTrue(any('unit-major' in text for text in problems))


class ProfileTests(unittest.TestCase):
    TIMED, AUDIT = 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-audit'
    U1_TIMED, U1_AUDIT = 'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit'

    def profiles(self):
        return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']

    def flat(self, profile):
        profile = json.loads(json.dumps(profile))
        profile.pop('description')
        return profile

    def test_the_timed_arm_is_the_control_plus_the_flag_alone(self):
        found = self.profiles()
        control = self.flat(found[self.TIMED])
        control['env']['QWEN_FAST_TP4_RS_UNIT_MAJOR'] = '1'
        self.assertEqual(self.flat(found[self.U1_TIMED]), control)

    def test_the_audited_arm_is_the_best_audit_plus_the_flag_and_the_audit(self):
        found = self.profiles()
        base = self.flat(found[self.AUDIT])
        base['env']['QWEN_FAST_TP4_RS_UNIT_MAJOR'] = '1'
        base['env']['QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT'] = '1'
        self.assertEqual(self.flat(found[self.U1_AUDIT]), base)

    def test_both_are_gate_only_and_carry_the_waiver_and_no_other_profile_carries_the_flag(self):
        found = self.profiles()
        for name in (self.U1_TIMED, self.U1_AUDIT):
            self.assertTrue(found[name]['gate_only'], name)
            self.assertEqual(found[name]['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
            self.assertEqual(found[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
        self.assertEqual(sorted(name for name, entry in found.items() if 'QWEN_FAST_TP4_RS_UNIT_MAJOR' in entry['env']),
                         sorted((self.U1_TIMED, self.U1_AUDIT, 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-dbf16')))  # + tp4/w1's stack, + the ship-prefix pair arms
        self.assertNotIn('QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT', found[self.U1_TIMED]['env'])
        self.assertEqual(json.loads(PROFILES.read_text(encoding='utf-8'))['default'], 'c2-packed-tp4')


class PackTests(unittest.TestCase):
    def jobs(self):
        order = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
                 if line.strip() and not line.startswith('#')]
        return order

    def env(self, name):
        values = {}
        for line in (PACK / (name + '.env')).read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.startswith('#'):
                key, _, value = line.partition('=')
                values[key] = value
        return values

    def test_the_order_is_the_asked_for_pack_on_one_image(self):
        jobs = self.jobs()
        self.assertEqual([job[0].split('-')[0] for job in jobs],
                         ['X0', 'B0', 'S0c', 'A1', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'T4', 'Z'])
        self.assertEqual({job[2] for job in jobs}, {'tp4-u1-1'})
        for job in jobs:
            self.assertTrue((PACK / (job[0] + '.env')).is_file(), job)
        modes = {job[0].split('-')[0]: job[1] for job in jobs}
        for name in ('X0', 'B0', 'S0c', 'A1', 'H1', 'H2', 'H3'):
            self.assertEqual(modes[name], 'stop', name)

    def test_no_job_stops_starts_or_hands_back_the_agent_and_the_first_quad_job_rescans(self):
        for job in self.jobs():
            env = self.env(job[0])
            actions = env['C2_ACTIONS'].split()
            self.assertFalse({'agentstop', 'agentstart', 'handback'} & set(actions), job)
        self.assertEqual(self.env(self.jobs()[0][0])['C2_ACTIONS'], 'status rescan reset')
        self.assertEqual(self.env(self.jobs()[0][0])['C2_CARDS'], 'quad')

    def test_arms_and_tests(self):
        control, u1 = 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-time-gate-u1'
        by = {job[0].split('-')[0]: self.env(job[0]) for job in self.jobs()}
        self.assertEqual(by['S0c']['C2_PROFILE'], 'c2-packed-tp4-8x262k-gate')
        self.assertIn('concurrent8_steady', by['S0c']['C2_SMOKE_TESTS'].split(','))
        self.assertEqual(by['A1']['C2_PROFILE'], 'c2-packed-tp4-8x262k-best-u1-audit')
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k'} <= set(by['A1']['C2_SMOKE_TESTS'].split(',')))
        for name in ('H1', 'H2', 'H3'):
            self.assertEqual(by[name]['C2_PROFILE'], u1)
        for name, profile in (('T1', control), ('T2', u1), ('T3', control), ('T4', u1)):
            self.assertEqual(by[name]['C2_PROFILE'], profile, name)
            tests = by[name]['C2_SMOKE_TESTS'].split(',')
            self.assertIn('concurrent8_steady', tests)
            self.assertTrue({'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests), name)
        self.assertEqual(by['B0']['C2_ACTIONS'], 'build')
        for env in by.values():
            self.assertEqual(env['C2_IMAGE_TAG'], 'tp4-u1-1')
            self.assertEqual(env['C2_CARDS'], 'quad')

    def test_every_env_parses_with_the_job_script(self):
        import subprocess
        for path in sorted(PACK.glob('*.env')):
            with self.subTest(env=path.name):
                done = subprocess.run([sys.executable, '-B', str(HERE / 'c2_serving_job.py'), str(path)], capture_output=True,
                                      text=True, cwd=str(HERE.parent.parent))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class ShipsTests(unittest.TestCase):
    def test_the_module_is_already_in_both_image_copy_lists_and_the_overlay(self):
        root = HERE.parent.parent
        self.assertIn('scripts/ci/tile_collective_tp.py', (root / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8'))
        self.assertIn('tile_collective_tp.py', (root / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text(encoding='utf-8'))
        self.assertIn('scripts/ci/tile_collective_tp.py', (root / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8'))

    def test_the_new_test_module_is_on_the_cpu_allowlist(self):
        root = HERE.parent.parent
        self.assertIn('test_tp4_u1', (root / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
