"""QWEN_FAST_LAZY_SHARD_W (qwen_lazy_shard, P1a of the upload plan): a weight-cache hit without the host bf16 transpose.

On a fake ttnn.as_tensor that follows the pinned core.py (the cache name is built from the caller's path, the dtype and the layout; a hit returns the file and never
looks at the tensor or `preprocess`; a miss runs `preprocess`, converts and writes): the twin hands as_tensor the stock call's arguments, so it opens the stock loader's
files and an old cache is valid for it (and its cache for the stock loader); a hit touches the weight not at all; a miss converts the same bytes; the audit compares
packed bytes with the stock conversion and latches the lever off on a difference; the pinned function's source digest; the flags; off installs nothing."""

import hashlib
import os
import textwrap
import types
import unittest
from types import SimpleNamespace

import numpy
import torch

import qwen_lazy_shard as lazy

# tp_common.shard_w as the image has it (tp_common.py sha256 bb43f0cde336c3f84725d47a64ed2b506b5287bdd0e910cd24b13feed0a0826a, the simulator gates' pin).
PINNED_SHARD_W = '''def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row."""
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )
'''


class HostTensor(object):
    def __init__(self, shards):
        self.shards = shards

    def host_buffer(self):
        shards = self.shards

        class Distributed(object):
            def get_shard(self, coordinate):
                class Buffer(object):
                    def __array__(self, *args, **kwargs):
                        return numpy.frombuffer(shards[coordinate.index], dtype=numpy.uint8)
                return Buffer()
        return Distributed()


class DeviceTensor(object):
    def __init__(self, shards):
        self.shards = shards


class FakeTTNN(object):
    """as_tensor as core.py has it; a conversion is the weight's bytes cut along `dim` (the packed bytes depend on the transposed values and the shard cut)."""

    bfloat8_b, bfloat4_b = 'bf8', 'bf4'
    TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'dram'

    def __init__(self, chips=4):
        self.chips, self.cache, self.log = chips, {}, []
        self.mesh = SimpleNamespace(shape=(1, chips))

    def pack(self, tensor, dim, dtype):
        flat = tensor.to(torch.float32).contiguous()
        parts = torch.chunk(flat, self.chips, dim=dim)
        return [bytes(dtype, 'ascii') + part.numpy().tobytes() for part in parts]

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim)

    def MeshCoordinate(self, row, column):
        return SimpleNamespace(index=column)

    def as_tensor(self, tensor, dtype=None, *, layout=None, device=None, memory_config=None, cache_file_name=None, preprocess=None, mesh_mapper=None):
        self.log.append(('as_tensor', dtype, layout, memory_config, mesh_mapper, cache_file_name, preprocess is not None))
        name = None if cache_file_name is None else '%s_dtype_%s_layout_%s.tensorbin' % (cache_file_name, dtype, layout)
        if name is not None and name in self.cache:
            self.log.append(('hit', name))
            return self.cache[name]
        if preprocess:
            tensor = preprocess(tensor)
        result = DeviceTensor(self.pack(tensor, mesh_mapper[1], dtype))
        self.log.append(('miss', name))
        if name is not None:
            self.cache[name] = result
        return result

    def from_torch(self, tensor, *, dtype, layout, mesh_mapper):
        self.log.append(('from_torch', dtype))
        return HostTensor(self.pack(tensor, mesh_mapper[1], dtype))

    def from_device(self, tensor):
        return HostTensor(list(tensor.shards))


class Spy(object):
    """A weight that counts how it is touched."""

    def __init__(self, tensor):
        self.tensor, self.touched = tensor, []

    @property
    def shape(self):
        return self.tensor.shape

    def to(self, *args, **kwargs):
        self.touched.append('to')
        return self.tensor.to(*args, **kwargs)


def logger():
    lines = []

    def log(template, *values):
        lines.append(template.format(*values))
    log.lines = lines
    return log


def tp_common(ttnn):
    module = types.ModuleType('tp_common_under_test')
    module.ttnn, module.torch = ttnn, torch
    exec(compile(PINNED_SHARD_W, 'tp_common_under_test.py', 'exec'), module.__dict__)
    import linecache
    linecache.cache['tp_common_under_test.py'] = (len(PINNED_SHARD_W), None, PINNED_SHARD_W.splitlines(True), 'tp_common_under_test.py')
    module.shard_w.__code__ = module.shard_w.__code__.replace(co_filename='tp_common_under_test.py')
    return module


def weight(seed=0, shape=(8, 16)):
    return torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape).add(seed).to(torch.bfloat16)


class PinTests(unittest.TestCase):
    def test_the_pinned_source_digest_is_the_images(self):
        self.assertEqual(hashlib.sha256(textwrap.dedent(PINNED_SHARD_W).encode('utf-8')).hexdigest(), lazy.SHARD_W_SOURCE_SHA256)
        module = tp_common(FakeTTNN())
        self.assertEqual(lazy.source_digest(module.shard_w), lazy.SHARD_W_SOURCE_SHA256)
        self.assertEqual(tuple(lazy.inspect.signature(module.shard_w).parameters), lazy.SHARD_W_PARAMETERS)

    def test_the_flags_are_strict(self):
        self.assertFalse(lazy.enabled({}))
        self.assertTrue(lazy.enabled({'QWEN_FAST_LAZY_SHARD_W': '1'}))
        for bad in ('', 'true', '2'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                lazy.enabled({'QWEN_FAST_LAZY_SHARD_W': bad})
        self.assertEqual(lazy.flag_problems({}), [])
        self.assertEqual(lazy.flag_problems({'QWEN_FAST_LAZY_SHARD_W': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS': '3'}), [])
        for env, needle in (({'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1'}, 'without QWEN_FAST_LAZY_SHARD_W=1'),
                            ({'QWEN_FAST_LAZY_SHARD_W': 'x'}, 'must be 0 or 1'),
                            ({'QWEN_FAST_LAZY_SHARD_W': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS': '0'}, 'integer in 1..64'),
                            ({'QWEN_FAST_LAZY_SHARD_W': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS': '2'}, 'would do nothing')):
            with self.subTest(env=env):
                self.assertTrue(any(needle in problem for problem in lazy.flag_problems(env)), lazy.flag_problems(env))


class LazyTests(unittest.TestCase):
    def setUp(self):
        self.ttnn = FakeTTNN()
        self.module = tp_common(self.ttnn)
        self.stock = self.module.shard_w
        self.log = logger()
        self.environ = {'QWEN_FAST_LAZY_SHARD_W': '1'}

    def install(self, **extra):
        self.environ.update(extra)
        self.assertTrue(lazy.install(self.module, self.environ, log=self.log))
        return self.module.shard_w

    def load(self, loader, tensor, path='cache/mlp.w1', dim=-1, dtype='bf4'):
        return loader(tensor, self.ttnn.mesh, dim, 'dram', path, dtype)

    def test_off_installs_nothing_and_arm_registers_nothing(self):
        self.assertFalse(lazy.install(self.module, {}, log=self.log))
        self.assertFalse(lazy.install(self.module, {'QWEN_FAST_LAZY_SHARD_W': '0'}, log=self.log))
        self.assertIs(self.module.shard_w, self.stock)
        registered = []
        self.assertFalse(lazy.arm({}, lambda name, callback: registered.append(name)))
        self.assertTrue(lazy.arm({'QWEN_FAST_LAZY_SHARD_W': '1'}, lambda name, callback: registered.append(name)))
        self.assertEqual(registered, [lazy.TP_COMMON])
        with self.assertRaises(ValueError):
            lazy.arm({'QWEN_FAST_LAZY_SHARD_W': '2'}, lambda name, callback: None)

    def test_the_twin_hands_as_tensor_the_stock_calls_arguments(self):
        lazy_loader = self.install()
        self.assertEqual(lazy.inspect.signature(lazy_loader), lazy.inspect.signature(self.stock))
        w = weight()
        self.load(self.stock, w)
        stock_call = self.ttnn.log[0]
        self.ttnn.cache.clear()
        self.ttnn.log.clear()
        self.load(lazy_loader, w)
        lazy_call = self.ttnn.log[0]
        self.assertEqual(stock_call[:6], lazy_call[:6], 'dtype, layout, memory config, mapper and the cache path are the stock call\'s')
        self.assertEqual((stock_call[6], lazy_call[6]), (False, True), 'the only difference: the transpose moved into preprocess')

    def test_a_cache_hit_never_touches_the_weight_and_a_miss_converts_the_same_bytes(self):
        lazy_loader = self.install()
        w = weight()
        stock_result = self.load(self.stock, w)
        spy = Spy(w)
        hit = self.load(lazy_loader, spy)
        self.assertIs(hit, stock_result)
        self.assertEqual(spy.touched, [], 'the stock loader\'s cache file served the twin without a transpose')
        self.assertEqual(self.ttnn.log[-1][0], 'hit')
        self.ttnn.cache.clear()
        spy = Spy(w)
        miss = self.load(lazy_loader, spy)
        self.assertEqual(spy.touched, ['to'])
        self.assertEqual(miss.shards, stock_result.shards)

    def test_the_two_loaders_share_one_cache_in_both_directions(self):
        lazy_loader = self.install()
        w = weight()
        self.load(lazy_loader, w, path='cache/a')
        self.load(self.stock, w, path='cache/a')
        self.assertEqual([entry[0] for entry in self.ttnn.log if entry[0] in ('hit', 'miss')], ['miss', 'hit'])
        self.assertEqual(len(self.ttnn.cache), 1)
        self.assertEqual(list(self.ttnn.cache), ['cache/a_dtype_bf4_layout_tile.tensorbin'])

    def test_an_uncached_load_still_transposes(self):
        lazy_loader = self.install()
        w = weight()
        first = self.load(lazy_loader, w, path=None)
        second = self.load(self.stock, w, path=None)
        self.assertEqual(first.shards, second.shards)

    def test_the_summary_line_every_64_loads_and_engaged_once(self):
        lazy_loader = self.install()
        for index in range(130):
            self.load(lazy_loader, weight(), path='cache/t%d' % (index % 3))
        engaged = [line for line in self.log.lines if line.startswith(lazy.ENGAGED)]
        loads = [line for line in self.log.lines if line.startswith(lazy.LOADS)]
        self.assertEqual(len(engaged), 1)
        self.assertEqual([line.split()[-4:] for line in loads], [['calls=64', 'misses=3', 'hits=61', 'latched=no'], ['calls=128', 'misses=3', 'hits=125', 'latched=no']])

    def test_the_audit_passes_on_a_valid_cache_and_a_fresh_conversion(self):
        lazy_loader = self.install(QWEN_FAST_LAZY_SHARD_W_AUDIT='1')
        w = weight()
        self.load(self.stock, w, path='cache/hit')                      # the stock loader's cache
        self.load(lazy_loader, w, path='cache/hit')                     # a hit
        self.load(lazy_loader, w, path='cache/miss')                    # a miss
        self.load(lazy_loader, w, path='cache/third')                   # past the default two audited loads
        passes = [line for line in self.log.lines if line.startswith(lazy.AUDIT_LINE + ' exact=True')]
        self.assertEqual(len(passes), 2)
        self.assertIn('name=hit_dtype' if False else 'name=hit', passes[0])
        self.assertIn('chips=4', passes[0])
        self.assertIsNone(self.module._qwen_lazy_shard_record.latched)

    def test_the_audit_catches_a_stale_cache_and_latches_the_lever_off(self):
        lazy_loader = self.install(QWEN_FAST_LAZY_SHARD_W_AUDIT='1')
        self.load(self.stock, weight(seed=0), path='cache/stale')       # a cache made from other weights
        got = self.load(lazy_loader, weight(seed=1), path='cache/stale')
        mismatch = [line for line in self.log.lines if line.startswith(lazy.AUDIT_MISMATCH)]
        self.assertEqual(len(mismatch), 1)
        self.assertIn('exact=False', mismatch[0])
        self.assertTrue(self.module._qwen_lazy_shard_record.latched)
        self.assertIsNotNone(got, 'the load still returns what the stock loader would have loaded from that file')
        spy = Spy(weight())
        before = len(self.ttnn.log)
        self.load(lazy_loader, spy, path='cache/other')
        self.assertEqual(spy.touched, ['to'], 'latched: the stock function runs (its transpose is eager)')
        self.assertEqual([entry for entry in self.ttnn.log[before:] if entry[0] == 'as_tensor'][0][6], False)

    def test_a_changed_function_is_left_alone(self):
        changed = tp_common(self.ttnn)
        source = PINNED_SHARD_W.replace('.T.contiguous()', '.T')
        changed.shard_w = None
        namespace = {'ttnn': self.ttnn, 'torch': torch}
        exec(compile(source, 'changed_tp_common.py', 'exec'), namespace)
        import linecache
        linecache.cache['changed_tp_common.py'] = (len(source), None, source.splitlines(True), 'changed_tp_common.py')
        changed.shard_w = namespace['shard_w']
        original = changed.shard_w
        log = logger()
        self.assertFalse(lazy.install(changed, self.environ, log=log))
        self.assertIs(changed.shard_w, original)
        self.assertTrue(log.lines[0].startswith(lazy.REFUSED))
        self.assertIn('not the pinned function', log.lines[0])
        wrong_signature = types.ModuleType('wrong')
        wrong_signature.shard_w = lambda tensor: tensor
        log = logger()
        self.assertFalse(lazy.install(wrong_signature, self.environ, log=log))
        self.assertIn('another signature', log.lines[0])

    def test_install_is_idempotent(self):
        first = self.install()
        self.assertTrue(lazy.install(self.module, self.environ, log=self.log))
        self.assertIs(self.module.shard_w, first)


if __name__ == '__main__':
    unittest.main()
