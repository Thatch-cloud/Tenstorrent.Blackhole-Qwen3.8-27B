"""QWEN_FAST_DEVICE_ZEROS (qwen_device_zeros, P0 of the upload plan): the big zero buffers of an engine start filled on the device.

On a recording fake of the ttnn surface it uses (no card): the flag-off op sequence is the served one (the model method and the pool); the flag-on sequence is
`empty` then an in-place `full_like(t, 0.0, optional_tensor=t)` per tensor and no host tensor, no `as_tensor`, no `from_torch`; what keeps the host path (sharded,
row-major, small, non-DRAM); the audit passes only on equal packed bytes of the first AND last blocks and refuses, latches and rebuilds on the host path on any
difference (an exponent-only one included), on a fill that stopped short, on bytes it cannot read, and on an op that raises; the KV twin differs from the pinned
model method in `_mk` alone (its source digest is pinned); the strict flags."""

import importlib.util
import inspect
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy
import torch

import qwen_device_zeros as zeros

HERE = Path(__file__).resolve().parent
MODEL_FIXTURE = HERE / 'fixtures' / 'qwen36_model.py'
SMALL = {'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES': '0'}
GARBAGE = 0xAA


class Distributed(object):
    def __init__(self, shards):
        self.shards = shards

    def get_shard(self, coordinate):
        return self.shards[coordinate.index]


class HostBuffer(object):
    """The HostBuffer binding: __array__ and __iter__ (no buffer protocol, as the pinned nanobind has it)."""

    def __init__(self, data):
        self.data = bytes(data)

    def __array__(self, *args, **kwargs):
        return numpy.frombuffer(self.data, dtype=numpy.uint8)

    def __iter__(self):
        return iter(self.data)


class HostTensor(object):
    def __init__(self, shards, raw=True):
        self.shards, self.raw = shards, raw

    def host_buffer(self):
        if not self.raw:
            raise RuntimeError('Tensor must be on host to access host_buffer')
        return Distributed([HostBuffer(shard) for shard in self.shards])


class Tensor(object):
    def __init__(self, shape, dtype, layout, memory, chips, data=None):
        self.shape, self.dtype, self.layout, self.memory = tuple(shape), dtype, layout, memory
        self.chips = [bytearray(data[chip] if data is not None else b'') for chip in range(chips)]


class FakeTTNN(object):
    """The ttnn surface qwen_device_zeros and the model/pool call, recording every call in `log`.

    fill: 'zeros' | 'exponent' (the device packer leaves byte 3 of every bf8 tile non-zero) | 'mantissa' (byte 100) | 'short' (stops after the first half of the
    blocks, leaving the rest as the allocator found it) | 'garbage_all'. host: 'zeros' | 'exponent' (the HOST packer writes the odd exponent)."""

    bfloat8_b, bfloat16, float32, uint32, int32, uint16 = 'bf8', 'bf16', 'fp32', 'u32', 'i32', 'u16'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'tile', 'row_major', 'dram', 'l1'

    def __init__(self, chips=4, fill='zeros', host='zeros', raw=True, fail=None):
        self.chips, self.fill, self.host, self.raw, self.fail = chips, fill, host, raw, fail
        self.log, self.live, self.freed = [], [], []
        self.mesh = SimpleNamespace(shape=(1, chips))

    # -- sizes ------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def tile_bytes(dtype):
        return {'bf8': 1088, 'bf16': 2048, 'fp32': 4096}[dtype]

    def nbytes(self, shape, dtype):
        return zeros.tile_count(shape) * self.tile_bytes(dtype)

    def block_bytes(self, tensor):
        return self.nbytes((1,) + tensor.shape[1:], tensor.dtype)

    # -- ops --------------------------------------------------------------------------------------------------------------------------------
    def empty(self, shape, *, dtype, layout, device, memory_config):
        self.log.append(('empty', tuple(shape), dtype, layout, device is self.mesh, memory_config))
        if self.fail == 'empty':
            raise RuntimeError('out of DRAM')
        tensor = Tensor(shape, dtype, layout, memory_config, self.chips,
                        [bytes([GARBAGE]) * self.nbytes(shape, dtype)] * self.chips)
        self.live.append(tensor)
        return tensor

    def full_like(self, tensor, value, *, optional_tensor=None):
        self.log.append(('full_like', id(tensor), value, optional_tensor is tensor))
        if self.fail == 'full_like':
            raise RuntimeError('unsupported dtype for fill')
        if optional_tensor is not tensor:
            raise AssertionError('in-place fill expected')
        size = len(tensor.chips[0])
        tile = self.tile_bytes(tensor.dtype)
        for chip in tensor.chips:
            if self.fill == 'short':
                chip[:size // 2] = bytes(size // 2)
                continue
            chip[:] = bytes(size)
            if self.fill == 'exponent' and tensor.dtype == 'bf8':
                for offset in range(3, size, tile):
                    chip[offset] = 0x7F
            if self.fill == 'mantissa' and tensor.dtype == 'bf8':
                for offset in range(100, size, tile):
                    chip[offset] = 0x01
            if self.fill == 'garbage_all':
                chip[:] = bytes([GARBAGE]) * size
        return tensor

    def from_torch(self, host, *, dtype, layout, device, memory_config, mesh_mapper):
        self.log.append(('from_torch', tuple(host.shape), dtype, layout, memory_config, mesh_mapper))
        size = self.nbytes(tuple(host.shape), dtype)
        data = bytearray(size)
        if self.host == 'exponent' and dtype == 'bf8':
            for offset in range(3, size, self.tile_bytes(dtype)):
                data[offset] = 0x7F
        tensor = Tensor(host.shape, dtype, layout, memory_config, self.chips, [bytes(data)] * self.chips)
        self.live.append(tensor)
        return tensor

    def as_tensor(self, host, *, device, dtype, layout, memory_config, mesh_mapper):
        self.log.append(('as_tensor', tuple(host.shape), dtype, layout, memory_config, mesh_mapper))
        tensor = Tensor(host.shape, dtype, layout, memory_config, self.chips, [bytes(self.nbytes(tuple(host.shape), dtype))] * self.chips)
        self.live.append(tensor)
        return tensor

    def slice(self, tensor, starts, ends):
        self.log.append(('slice', tuple(starts), tuple(ends)))
        block = self.block_bytes(tensor)
        lo, hi = starts[0], ends[0]
        shape = (hi - lo,) + tensor.shape[1:]
        part = Tensor(shape, tensor.dtype, tensor.layout, tensor.memory, self.chips,
                      [bytes(chip[lo * block:hi * block]) for chip in tensor.chips])
        self.live.append(part)
        return part

    def from_device(self, tensor):
        self.log.append(('from_device', tuple(tensor.shape)))
        return HostTensor([bytes(chip) for chip in tensor.chips], raw=self.raw)

    def deallocate(self, tensor):
        self.log.append(('deallocate', id(tensor)))
        if any(tensor is freed for freed in self.freed):
            raise AssertionError('double free')
        self.freed.append(tensor)

    def synchronize_device(self, mesh):
        self.log.append(('synchronize',))

    def ReplicateTensorToMesh(self, mesh):
        return ('replicate', mesh is self.mesh)

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim)

    def MeshCoordinate(self, row, column):
        return SimpleNamespace(index=column, row=row)

    def names(self):
        return [entry[0] for entry in self.log]


def logger():
    lines = []

    def log(template, *values):
        lines.append(template.format(*values))
    log.lines = lines
    return log


def filler_for(ttnn, extra=None, log=None):
    environ = dict(SMALL, **(extra or {}))
    log = log or logger()
    return zeros.DeviceZeros(ttnn, torch, environ, log=log), log


class FlagTests(unittest.TestCase):
    def test_the_flags_are_strict(self):
        self.assertFalse(zeros.enabled({}))
        self.assertFalse(zeros.enabled({'QWEN_FAST_DEVICE_ZEROS': '0'}))
        self.assertTrue(zeros.enabled({'QWEN_FAST_DEVICE_ZEROS': '1'}))
        for bad in ('', 'true', '2', ' 1', 'on'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                zeros.enabled({'QWEN_FAST_DEVICE_ZEROS': bad})
            with self.subTest(audit=bad), self.assertRaises(ValueError):
                zeros.audited({'QWEN_FAST_DEVICE_ZEROS_AUDIT': bad})

    def test_flag_problems_name_each_misuse(self):
        self.assertEqual(zeros.flag_problems({}), [])
        self.assertEqual(zeros.flag_problems({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}), [])
        cases = (({'QWEN_FAST_DEVICE_ZEROS': 'yes'}, 'must be 0 or 1'),
                 ({'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, 'without QWEN_FAST_DEVICE_ZEROS=1'),
                 ({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES': '-1'}, 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES must be an integer'),
                 ({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS': '0'}, 'QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS must be an integer in 1..64'),
                 ({'QWEN_FAST_DEVICE_ZEROS': '0', 'QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS': '4'}, 'would do nothing'))
        for env, needle in cases:
            with self.subTest(env=env):
                problems = zeros.flag_problems(env)
                self.assertTrue(problems and any(needle in problem for problem in problems), problems)

    def test_off_creates_no_filler_and_arms_nothing(self):
        zeros.forget()
        self.assertIsNone(zeros.filler(FakeTTNN(), torch, {}))
        self.assertIsNone(zeros.filler(FakeTTNN(), torch, {'QWEN_FAST_DEVICE_ZEROS': '0'}))
        self.assertEqual(zeros._FILLERS, {})
        calls = []
        self.assertFalse(zeros.arm({}, lambda name, callback: calls.append(name)))
        self.assertFalse(zeros.arm({'QWEN_FAST_DEVICE_ZEROS': '0'}, lambda name, callback: calls.append(name)))
        self.assertEqual(calls, [])
        self.assertTrue(zeros.arm({'QWEN_FAST_DEVICE_ZEROS': '1'}, lambda name, callback: calls.append(name)))
        self.assertEqual(calls, [zeros.MODEL_MODULE])
        with self.assertRaises(ValueError):
            zeros.arm({'QWEN_FAST_DEVICE_ZEROS': '2'}, lambda name, callback: calls.append(name))

    def test_one_filler_per_process_and_ttnn(self):
        zeros.forget()
        ttnn = FakeTTNN()
        first = zeros.filler(ttnn, torch, {'QWEN_FAST_DEVICE_ZEROS': '1'})
        self.assertIs(first, zeros.filler(ttnn, torch, {'QWEN_FAST_DEVICE_ZEROS': '1'}))
        self.assertIsNot(first, zeros.filler(FakeTTNN(), torch, {'QWEN_FAST_DEVICE_ZEROS': '1'}))
        zeros.forget()


class BuildTests(unittest.TestCase):
    def build(self, ttnn, filler, shape=(6, 1, 64, 256), dtype='bf8', **options):
        options = dict(dict(layout='tile', memory_config='dram', tag='kv_cache'), **options)
        return filler.build(shape, dtype, options['layout'], ttnn.mesh, options['memory_config'], options['tag'], **{
            key: value for key, value in options.items() if key == 'replicated'})

    def test_the_device_sequence_is_empty_then_one_in_place_fill_and_no_host_tensor(self):
        ttnn = FakeTTNN()
        filler, log = filler_for(ttnn)
        tensor = self.build(ttnn, filler)
        self.assertIsNotNone(tensor)
        self.assertEqual(ttnn.names(), ['empty', 'full_like'])
        self.assertEqual(ttnn.log[0], ('empty', (6, 1, 64, 256), 'bf8', 'tile', True, 'dram'))
        self.assertEqual(ttnn.log[1], ('full_like', id(tensor), 0.0, True))
        for chip in tensor.chips:
            self.assertEqual(bytes(chip), bytes(len(chip)))
        self.assertEqual(len(tensor.chips[0]), 6 * 16 * 1088)
        self.assertEqual(len([line for line in log.lines if line.startswith(zeros.ENGAGED)]), 1)
        self.assertIn('tag=kv_cache', log.lines[0])
        self.assertIn('audit=off', log.lines[0])
        self.build(ttnn, filler)
        self.assertEqual(len([line for line in log.lines if line.startswith(zeros.ENGAGED)]), 1, 'the first tensor of a tag logs once')

    def test_each_dtype_family_is_built_on_the_device(self):
        for dtype in ('bf8', 'bf16', 'fp32'):
            with self.subTest(dtype=dtype):
                ttnn = FakeTTNN()
                filler, _ = filler_for(ttnn)
                self.assertIsNotNone(self.build(ttnn, filler, dtype=dtype))
                self.assertEqual(ttnn.names(), ['empty', 'full_like'])

    def test_what_keeps_the_host_path(self):
        cases = (('sharded mapper', dict(replicated=False)), ('row major', dict(layout='row_major')), ('integer', dict(dtype='i32')),
                 ('sharded or L1 memory', dict(memory_config='l1')))
        for label, options in cases:
            with self.subTest(label):
                ttnn = FakeTTNN()
                filler, log = filler_for(ttnn)
                self.assertIsNone(self.build(ttnn, filler, **options))
                self.assertEqual(ttnn.log, [])
                self.assertEqual(log.lines, [])
        ttnn = FakeTTNN()
        filler, _ = filler_for(ttnn, {'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES': str(10 ** 9)})
        self.assertIsNone(self.build(ttnn, filler))
        self.assertEqual(ttnn.log, [])

    def test_the_default_floor_keeps_small_tensors_off_the_device(self):
        ttnn = FakeTTNN()
        filler = zeros.DeviceZeros(ttnn, torch, {'QWEN_FAST_DEVICE_ZEROS': '1'}, log=logger())
        self.assertEqual(filler.min_bytes, 4 * 1024 * 1024)
        self.assertIsNone(self.build(ttnn, filler, shape=(1, 4, 2048, 128), dtype='bf16'))
        self.assertIsNotNone(self.build(ttnn, filler, shape=(1, 1, 2048, 5120), dtype='bf16'))

    def test_a_raising_op_frees_the_tensor_latches_and_every_later_tensor_is_the_hosts(self):
        for failing in ('full_like', 'empty'):
            with self.subTest(failing=failing):
                ttnn = FakeTTNN(fail=failing)
                filler, log = filler_for(ttnn)
                self.assertIsNone(self.build(ttnn, filler))
                self.assertTrue(filler.latched and failing in ' '.join(ttnn.names()) or filler.latched)
                if failing == 'full_like':
                    self.assertEqual(len(ttnn.freed), 1, 'the half-built tensor is freed')
                refusals = [line for line in log.lines if line.startswith(zeros.REFUSED)]
                self.assertEqual(len(refusals), 1)
                self.assertIn('every later zero buffer takes the host path', refusals[0])
                before = len(ttnn.log)
                self.assertIsNone(self.build(ttnn, filler))
                self.assertEqual(len(ttnn.log), before, 'a latched lever touches the device no more')

    def test_report_totals_and_synchronises_once(self):
        ttnn = FakeTTNN()
        filler, log = filler_for(ttnn)
        for _ in range(3):
            self.build(ttnn, filler)
        state = filler.report('kv_cache', synchronize=lambda: ttnn.synchronize_device(ttnn.mesh))
        self.assertEqual((state['built'], state['host']), (3, 0))
        self.assertEqual(state['bytes'], 3 * 6 * 16 * 1088)
        self.assertEqual(ttnn.names().count('synchronize'), 1)
        self.assertTrue(log.lines[-1].startswith('[PINDIAG] tp4 device zeros engaged summary tag=kv_cache device=3 host=0 '))


class AuditTests(unittest.TestCase):
    def audited(self, ttnn, shape=(6, 1, 64, 256), dtype='bf8', extra=None):
        filler, log = filler_for(ttnn, dict({'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, **(extra or {})))
        tensor = filler.build(shape, dtype, 'tile', ttnn.mesh, 'dram', 'kv_cache')
        return filler, log, tensor

    def lines(self, log, marker):
        if marker == zeros.AUDIT_LINE:
            return [line for line in log.lines if line.startswith(marker + ' exact=True')]          # (the mismatch marker begins with it)
        return [line for line in log.lines if line.startswith(marker)]

    def test_equal_bytes_log_exact_true_and_keep_the_device_tensor(self):
        ttnn = FakeTTNN()
        filler, log, tensor = self.audited(ttnn)
        self.assertIsNotNone(tensor)
        passes = self.lines(log, zeros.AUDIT_LINE)
        self.assertEqual(len(passes), 1)
        self.assertIn('exact=True', passes[0])
        self.assertIn('tag=kv_cache', passes[0])
        self.assertIn('chips=4', passes[0])
        self.assertIn('raw=numpy', passes[0])
        self.assertIsNone(filler.latched)
        self.assertEqual(self.lines(log, zeros.AUDIT_MISMATCH), [])
        # the host-built twin goes through the host path of record (from_torch, replicated), and both it and every sample are freed
        self.assertEqual([entry for entry in ttnn.log if entry[0] == 'from_torch'][0][1:4], ((6, 1, 64, 256), 'bf8', 'tile'))
        self.assertIn(('from_torch', (6, 1, 64, 256), 'bf8', 'tile', 'dram', ('replicate', True)), ttnn.log)
        self.assertEqual(len(ttnn.freed), 1)
        self.assertNotIn(tensor, ttnn.freed)

    def test_only_the_first_tensors_of_a_tag_are_audited(self):
        ttnn = FakeTTNN()
        filler, log, _ = self.audited(ttnn)
        for _ in range(4):
            filler.build((6, 1, 64, 256), 'bf8', 'tile', ttnn.mesh, 'dram', 'kv_cache')
        self.assertEqual(len(self.lines(log, zeros.AUDIT_LINE)), zeros.DEFAULT_AUDIT_TENSORS)
        filler.build((6, 1, 64, 256), 'bf8', 'tile', ttnn.mesh, 'dram', 'buffer_pool')
        self.assertEqual(len(self.lines(log, zeros.AUDIT_LINE)), zeros.DEFAULT_AUDIT_TENSORS + 1, 'a second tag has its own count')

    def test_a_large_tensor_is_sampled_by_blocks_that_include_the_first_and_the_last(self):
        with mock.patch.object(zeros, 'WHOLE_READ_LIMIT', 1):
            ttnn = FakeTTNN()
            filler, log, tensor = self.audited(ttnn, shape=(40, 1, 64, 256), extra={'QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS': '5'})
            self.assertIsNotNone(tensor)
            slices = [entry for entry in ttnn.log if entry[0] == 'slice']
            starts = [entry[1][0] for entry in slices]
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1], 39)
            self.assertEqual(len(starts), 5)
            self.assertEqual(starts, sorted(set(starts)))
            self.assertEqual([entry[2][0] - entry[1][0] for entry in slices], [1] * 5)
            self.assertEqual(ttnn.log.count(('from_torch', (1, 1, 64, 256), 'bf8', 'tile', 'dram', ('replicate', True))), 1, 'one twin serves every window')
            self.assertEqual(len(self.lines(log, zeros.AUDIT_LINE)), 1)

    def test_a_fill_that_stops_short_is_caught_by_the_last_block(self):
        with mock.patch.object(zeros, 'WHOLE_READ_LIMIT', 1):
            ttnn = FakeTTNN(fill='short')
            filler, log, tensor = self.audited(ttnn, shape=(40, 1, 64, 256))
            self.assertIsNone(tensor)
            self.assertTrue(filler.latched)
            mismatch = self.lines(log, zeros.AUDIT_MISMATCH)
            self.assertTrue(any('exact=False' in line for line in mismatch), mismatch)

    def test_a_bf8_exponent_only_difference_is_a_mismatch_and_says_so(self):
        for fill, host in (('exponent', 'zeros'), ('zeros', 'exponent')):
            with self.subTest(fill=fill, host=host):
                ttnn = FakeTTNN(fill=fill, host=host)
                filler, log, tensor = self.audited(ttnn)
                self.assertIsNone(tensor, 'the lever refuses: the caller builds the host tensor')
                self.assertEqual(self.lines(log, zeros.AUDIT_LINE), [])
                line = [line for line in log.lines if 'exact=False' in line][0]
                self.assertIn('exponent_only=True', line)
                self.assertIn('dtype=bf8', line)
                self.assertTrue(filler.latched)
                self.assertIn(ttnn.live[0], ttnn.freed, 'the device tensor under audit is freed')

    def test_a_mantissa_difference_is_a_mismatch_that_is_not_exponent_only(self):
        ttnn = FakeTTNN(fill='mantissa')
        filler, log, tensor = self.audited(ttnn)
        self.assertIsNone(tensor)
        line = [line for line in log.lines if 'exact=False' in line][0]
        self.assertIn('exponent_only=False', line)
        self.assertIn('nonzero_bytes=', line)

    def test_unwritten_memory_is_a_mismatch(self):
        ttnn = FakeTTNN(fill='garbage_all')
        filler, log, tensor = self.audited(ttnn, dtype='bf16')
        self.assertIsNone(tensor)
        self.assertIn('exact=False', ' '.join(log.lines))

    def test_bytes_the_audit_cannot_read_fail_closed(self):
        ttnn = FakeTTNN(raw=False)
        filler, log, tensor = self.audited(ttnn)
        self.assertIsNone(tensor)
        self.assertEqual(self.lines(log, zeros.AUDIT_LINE), [])
        text = ' '.join(self.lines(log, zeros.AUDIT_MISMATCH))
        self.assertIn('host_buffer', text)
        self.assertTrue(filler.latched)

    def test_a_latched_audit_sends_every_later_tensor_to_the_host(self):
        ttnn = FakeTTNN(fill='mantissa')
        filler, log, _ = self.audited(ttnn)
        before = len(ttnn.log)
        self.assertIsNone(filler.build((6, 1, 64, 256), 'bf8', 'tile', ttnn.mesh, 'dram', 'kv_cache'))
        self.assertEqual(len(ttnn.log), before)

    def test_byte_views_are_tried_in_order_and_named(self):
        class Plain(object):
            def __iter__(self):
                return iter(b'\x00\x01\x02')

        class Dlpack(object):
            def __dlpack__(self, *args, **kwargs):
                return torch.arange(3, dtype=torch.uint8).__dlpack__(*args, **kwargs)

            def __dlpack_device__(self):
                return torch.arange(3, dtype=torch.uint8).__dlpack_device__()

        self.assertEqual(zeros.buffer_bytes(HostBuffer(b'\x00\x01\x02')), (b'\x00\x01\x02', 'numpy'))
        self.assertEqual(zeros.buffer_bytes(Plain()), (b'\x00\x01\x02', 'iter'))
        data, how = zeros.buffer_bytes(Dlpack())
        self.assertEqual((data, how), (b'\x00\x01\x02', 'dlpack'))
        with self.assertRaises(zeros.RawBytesUnavailable):
            zeros.buffer_bytes(object())

    def test_differing_offsets_count_every_difference_and_a_length_mismatch(self):
        self.assertEqual(zeros.differing_offsets(b'abc', b'abc'), ([], 0))
        self.assertEqual(zeros.differing_offsets(b'abc', b'axc'), ([1], 1))
        self.assertEqual(zeros.differing_offsets(b'abc', b'abcde'), ([], 2))
        offsets, count = zeros.differing_offsets(bytes(100), bytes([1]) * 100, limit=5)
        self.assertEqual((len(offsets), count), (5, 100))


class PinnedModel(object):
    """The image's model.py (fixtures/qwen36_model.py) executed under a stub tree and the fake ttnn."""

    @staticmethod
    def stubs(ttnn):
        modules = {}
        for name in ('models', 'models.common', 'models.common.rmsnorm', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                     'models.demos.blackhole.qwen36.tt', 'models.demos.blackhole.qwen36.tt.layer', 'models.demos.blackhole.qwen36.tt.model_config',
                     'models.demos.blackhole.qwen36.tt.rope', 'models.tt_transformers', 'models.tt_transformers.tt',
                     'models.tt_transformers.tt.common', 'loguru', 'tqdm'):
            modules[name] = types.ModuleType(name)
        modules['ttnn'] = ttnn
        modules['loguru'].logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
        modules['tqdm'].tqdm = lambda iterable, **kwargs: iterable
        modules['models.common.rmsnorm'].RMSNorm = type('RMSNorm', (), {})
        modules['models.demos.blackhole.qwen36.tt.layer'].Qwen36DecoderLayer = type('Qwen36DecoderLayer', (), {})
        modules['models.demos.blackhole.qwen36.tt.model_config'].Qwen36ModelArgs = type('Qwen36ModelArgs', (), {})
        modules['models.demos.blackhole.qwen36.tt.rope'].Qwen36RoPESetup = type('Qwen36RoPESetup', (), {})
        common = modules['models.tt_transformers.tt.common']
        common.Mode = SimpleNamespace(PREFILL='prefill', DECODE='decode')
        common.get_block_size = lambda caches: 64
        common.num_blocks_in_seq = lambda seq, block: -(-seq // block)
        return modules

    @classmethod
    def load(cls, ttnn):
        spec = importlib.util.spec_from_file_location('qwen36_model_under_test', str(MODEL_FIXTURE))
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, cls.stubs(ttnn)):
            spec.loader.exec_module(module)
        return module

    @staticmethod
    def model(module, ttnn, attention_layers=(1, 3), layers=5):
        events = []

        class Attention(object):
            def __init__(self, index):
                self.index = index

            def set_paged_kv_cache(self, k, v):
                events.append(('set_paged', self.index))

            def reset_state(self):
                events.append(('reset_state', self.index))

        instance = object.__new__(module.Qwen36Model)
        instance.device = ttnn.mesh
        instance._attention_layer_indices = list(attention_layers)
        instance._deltanet_external_states = None
        instance.layers = [SimpleNamespace(is_full_attention=index in attention_layers, attention=Attention(index)) for index in range(layers)]
        return instance, events


class KvTwinTests(unittest.TestCase):
    SHAPE = (6, 1, 64, 256)

    def run_method(self, method, ttnn, module):
        model, events = PinnedModel.model(module, ttnn)
        caches = method(model, self.SHAPE, ttnn.bfloat8_b, 8)
        return model, events, caches

    def test_the_pinned_source_digest_is_the_fixture_methods(self):
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        self.assertEqual(zeros.source_digest(module.Qwen36Model._allocate_kv_caches_tp), zeros.KV_SOURCE_SHA256)

    def test_off_the_model_method_is_the_image_s_and_arm_hooks_nothing(self):
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        original = module.Qwen36Model._allocate_kv_caches_tp
        self.assertFalse(zeros.install_kv(module, {}))
        self.assertFalse(zeros.install_kv(module, {'QWEN_FAST_DEVICE_ZEROS': '0'}))
        self.assertIs(module.Qwen36Model._allocate_kv_caches_tp, original)
        self.assertFalse(getattr(original, '_qwen_device_zeros', False))

    def test_with_no_filler_the_twin_runs_the_images_op_sequence_exactly(self):
        base, other = FakeTTNN(), FakeTTNN()
        original = PinnedModel.load(base)
        _, events_a, caches_a = self.run_method(original.Qwen36Model._allocate_kv_caches_tp, base, original)
        twin_module = PinnedModel.load(other)
        twin = zeros.kv_twin(twin_module, twin_module.Qwen36Model._allocate_kv_caches_tp, lambda: None)
        _, events_b, caches_b = self.run_method(twin, other, twin_module)
        self.assertEqual(base.log, other.log)
        self.assertEqual(events_a, events_b)
        self.assertEqual([entry[0] for entry in base.log], ['as_tensor'] * 4)
        self.assertEqual(len(caches_a), len(caches_b))
        self.assertEqual(events_a, [('set_paged', 1), ('set_paged', 3), ('reset_state', 0), ('reset_state', 2), ('reset_state', 4)])

    def test_on_each_cache_is_empty_plus_an_in_place_fill_and_nothing_else_touches_the_host(self):
        zeros.forget()
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        log = logger()
        with mock.patch.dict(os.environ, SMALL):
            self.assertTrue(zeros.install_kv(module, dict(SMALL), log=log))
        model, events, caches = self.run_method(module.Qwen36Model._allocate_kv_caches_tp, ttnn, module)
        zeros.forget()
        names = ttnn.names()
        self.assertEqual(names.count('empty'), 4)
        self.assertEqual(names.count('full_like'), 4)
        self.assertEqual(names.count('as_tensor'), 0)
        self.assertEqual(names.count('from_torch'), 0)
        self.assertEqual(names.count('synchronize'), 1)
        self.assertEqual(names[:8], ['empty', 'full_like'] * 4)
        self.assertEqual(len(caches), 2)
        self.assertTrue(all(len(pair) == 2 for pair in caches))
        self.assertEqual(events, [('set_paged', 1), ('set_paged', 3), ('reset_state', 0), ('reset_state', 2), ('reset_state', 4)])
        self.assertTrue(model._deltanet_external_states == [])
        self.assertTrue(all(layer.attention.B == 8 for layer in model.layers if not layer.is_full_attention))
        self.assertTrue(any(line.startswith(zeros.ENGAGED + ' tag=kv_cache') for line in log.lines))
        self.assertTrue(any('engaged summary tag=kv_cache device=4 host=0' in line for line in log.lines))

    def test_on_with_a_small_floor_unmet_the_twin_falls_back_to_the_images_expression(self):
        zeros.forget()
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        environ = {'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES': str(10 ** 12)}
        self.assertTrue(zeros.install_kv(module, environ, log=logger()))
        self.run_method(module.Qwen36Model._allocate_kv_caches_tp, ttnn, module)
        zeros.forget()
        self.assertEqual(ttnn.names().count('as_tensor'), 4)
        self.assertEqual(ttnn.names().count('empty'), 0)

    def test_install_is_idempotent_and_refuses_a_method_it_does_not_know(self):
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        log = logger()
        self.assertTrue(zeros.install_kv(module, dict(SMALL), log=log))
        first = module.Qwen36Model._allocate_kv_caches_tp
        self.assertTrue(zeros.install_kv(module, dict(SMALL), log=log))
        self.assertIs(module.Qwen36Model._allocate_kv_caches_tp, first)
        zeros.forget()
        # another model.py: one changed line in the method
        edited = MODEL_FIXTURE.read_text(encoding='utf-8').replace('"""TP paged KV allocation (B=1). Replicated per device; GDN self-manages state."""',
                                                                    '"""TP paged KV allocation (B=1), edited."""')
        self.assertNotEqual(edited, MODEL_FIXTURE.read_text(encoding='utf-8'))
        path = Path(os.environ.get('TMPDIR', '/tmp')) / ('edited_model_%d.py' % os.getpid())
        path.write_text(edited, encoding='utf-8')
        try:
            spec = importlib.util.spec_from_file_location('qwen36_model_edited', str(path))
            other = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, PinnedModel.stubs(ttnn)):
                spec.loader.exec_module(other)
            before = other.Qwen36Model._allocate_kv_caches_tp
            log = logger()
            self.assertFalse(zeros.install_kv(other, dict(SMALL), log=log))
            self.assertIs(other.Qwen36Model._allocate_kv_caches_tp, before)
            self.assertTrue(any(line.startswith(zeros.REFUSED) and 'not the pinned one' in line for line in log.lines), log.lines)
        finally:
            path.unlink()
        empty = types.ModuleType('empty_model')
        empty.ttnn, empty.torch = ttnn, torch
        empty.Qwen36Model = type('Qwen36Model', (), {})
        log = logger()
        self.assertFalse(zeros.install_kv(empty, dict(SMALL), log=log))
        self.assertTrue(log.lines[0].startswith(zeros.REFUSED))

    def test_arm_installs_when_the_model_module_imports(self):
        zeros.forget()
        ttnn = FakeTTNN()
        module = PinnedModel.load(ttnn)
        registered = []
        self.assertTrue(zeros.arm(dict(SMALL), lambda name, callback: registered.append((name, callback)), log=logger()))
        self.assertEqual([name for name, _ in registered], [zeros.MODEL_MODULE])
        registered[0][1](module)
        self.assertTrue(module.Qwen36Model._allocate_kv_caches_tp._qwen_device_zeros)
        zeros.forget()

    def test_the_audited_twin_rebuilds_a_refused_cache_on_the_host_path(self):
        zeros.forget()
        ttnn = FakeTTNN(fill='mantissa')
        module = PinnedModel.load(ttnn)
        log = logger()
        environ = dict(SMALL, QWEN_FAST_DEVICE_ZEROS_AUDIT='1')
        self.assertTrue(zeros.install_kv(module, environ, log=log))
        _, _, caches = self.run_method(module.Qwen36Model._allocate_kv_caches_tp, ttnn, module)
        zeros.forget()
        self.assertEqual(len(caches), 2)
        self.assertEqual(ttnn.names().count('as_tensor'), 4, 'all four caches are the host path\'s')
        self.assertTrue(any('exact=False' in line for line in log.lines))
        self.assertTrue(any('latched=audit of kv_cache tensor 1' in line for line in log.lines), log.lines)


class PoolTests(unittest.TestCase):
    def operations(self):
        from test_serving_buffer_pool import FakeOperations

        class Operations(FakeOperations):
            bfloat8_b, float32 = 'bf8', 'fp32'

            def __init__(self):
                FakeOperations.__init__(self)
                self.empties = []

            def empty(self, shape, *, dtype, layout, device, memory_config):
                self.empties.append((tuple(shape), dtype, layout, memory_config))
                return self.allocate(tuple(shape), dtype=dtype, layout=layout, mapper=('replicate', device), memory=memory_config)

        return Operations()

    def test_flag_off_the_pool_never_looks_at_the_device_fill(self):
        from serving_buffer_pool import ServingBufferPool

        zeros.forget()
        operations = self.operations()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('QWEN_FAST_DEVICE_ZEROS', None)
            ServingBufferPool(operations, 'mesh', users=1)
        self.assertEqual(operations.empties, [])
        self.assertEqual(operations.fills, [])

    def test_flag_on_the_big_history_buffers_are_filled_on_the_card_and_the_small_banks_are_not(self):
        from serving_buffer_pool import DRAFT_LAYERS, HISTORY_SHAPE, KV_SHAPE, ServingBufferPool

        zeros.forget()
        operations = self.operations()
        with mock.patch.dict(os.environ, {'QWEN_FAST_DEVICE_ZEROS': '1'}):
            pool = ServingBufferPool(operations, 'mesh', users=2)
        zeros.forget()
        self.assertEqual(operations.empties, [(HISTORY_SHAPE, 'bf16', 'tile', 'dram')] * 4)
        self.assertEqual(len(operations.fills), 4)
        self.assertTrue(all(value == 0.0 for _, value in operations.fills))
        self.assertEqual(operations.zeros_like_calls, 0)
        self.assertEqual(len(pool.slots), 2)
        # every pooled buffer still owns independent storage on every chip, history and banks alike
        for chip in range(2):
            addresses = {address[chip] for slot in pool.slots for address in slot.addresses}
            self.assertEqual(len(addresses), 2 * (2 + 4 * DRAFT_LAYERS))
        host_shapes = [value.shape for value in operations.live if value.shape == KV_SHAPE]
        self.assertEqual(len(host_shapes), 2 * 4 * DRAFT_LAYERS, 'the 2 MB banks keep the host path under the default floor')


if __name__ == '__main__':
    unittest.main()
