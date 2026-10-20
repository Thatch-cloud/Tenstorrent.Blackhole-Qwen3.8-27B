"""The host tier's upload levers (qwen_tier_upload; docs/tp4-upload1.md): threaded, pipelined staging copies of the raw block IO and parallel digest checks. CPU only.

* the strict flags and the registry's refusals, and that an engine without them builds none of it (no wrapper, no counter, no import);
* ThreadedTierIO against the model graft's own adapter (the class in qwen_prefix_model_patch.MODEL_ADAPTER, built from its source over the same fake raw ops): the same payload
  bytes read, the same device bytes written, the same device calls in the same order, one at a time, with the next call's staging running beside the current call; errors
  surface and leave no worker behind; the adapter's refusals are the wrapper's;
* the digest checks on threads against the sequential loop on the real scheduler graft: the same verdicts, counters and sampling, the same record dropped;
* the checkpoint store's arithmetic (qwen_tier_upload.checkpoint_bytes) against the numbers docs/prefix-store-hygiene.md and docs/tp4-upload1.md quote.
"""

import ast
import hashlib
import os
import random
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import qwen_prefix_model_patch as patcher  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_tier_upload as upload  # noqa: E402
import test_qwen_prefix_tiers as tiers  # noqa: E402

IO_THREADS, IO_CHUNK, VERIFY_THREADS = upload.ENV_IO_THREADS, upload.ENV_IO_CHUNK, upload.ENV_VERIFY_THREADS


# -- the fake device and the stock adapter ----------------------------------------------------------------------

class FakeDevice(object):
    """`caches` cache tensors on `chips` chips, `blocks` blocks each, `slice_bytes` per chip per block: the raw ops of the qwen_kv_read extension on numpy arrays, with
    the extension's own argument checks, a log of every call and a count of calls in flight."""

    def __init__(self, caches=3, chips=4, blocks=64, slice_bytes=48, seed=1):
        rng = random.Random(seed)
        self.chips, self.slice_bytes = chips, slice_bytes
        self.data = [np.frombuffer(bytes(rng.randrange(256) for _ in range(chips * blocks * slice_bytes)), dtype=np.uint8).reshape(chips, blocks, slice_bytes).copy()
                     for _ in range(caches)]
        self.log = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.delay = 0.0
        self.fail_on = None            # (op, call number)
        self.hook = None               # called inside every call
        self.guard = threading.Lock()
        self.cache_index = {id(cache): index for index, cache in enumerate(self.cache_objects())}

    def cache_objects(self):
        if not hasattr(self, '_objects'):
            self._objects = [SimpleNamespace(shape=(self.data[0].shape[1], 1, 64, 256), dtype='bfloat8_b', index=index) for index in range(len(self.data))]
        return self._objects

    def block_bytes(self, cache):
        return self.slice_bytes

    def _enter(self, op, cache, blocks):
        with self.guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            number = sum(1 for row in self.log if row[0] == op)
            self.log.append((op, cache.index, tuple(blocks)))
        if self.fail_on == (op, number):
            raise RuntimeError('%s refused (call %d)' % (op, number))
        if self.hook is not None:
            self.hook(op, number, cache, blocks)
        if self.delay:
            time.sleep(self.delay)

    def _leave(self):
        with self.guard:
            self.in_flight -= 1

    def read_raw(self, cache, out, blocks):
        try:
            self._enter('read', cache, blocks)
            if out.dtype != np.uint8 or tuple(out.shape) != (self.chips, len(blocks), self.slice_bytes) or not out.flags['C_CONTIGUOUS']:
                raise RuntimeError('qwen_read_blocks_raw: out %s %s' % (out.dtype, out.shape))
            for chip in range(self.chips):
                out[chip] = self.data[cache.index][chip, list(blocks), :]
        finally:
            self._leave()

    def write_raw(self, cache, data, blocks):
        try:
            self._enter('write', cache, blocks)
            if data.dtype != np.uint8 or tuple(data.shape) != (self.chips, len(blocks), self.slice_bytes) or not data.flags['C_CONTIGUOUS']:
                raise RuntimeError('qwen_write_blocks_raw: data %s %s' % (data.dtype, data.shape))
            for chip in range(self.chips):
                self.data[cache.index][chip, list(blocks), :] = data[chip]
        finally:
            self._leave()

    def snapshot(self):
        return [cache.copy() for cache in self.data]


def stock_adapter(device, chunk=None):
    """The model graft's own _QwenKvTierIO, from its source in the patch, over `device`'s raw ops (the ttnn its methods see is a namespace of those ops)."""
    segment = [ast.get_source_segment(patcher.MODEL_ADAPTER, node) for node in ast.parse(patcher.MODEL_ADAPTER).body
               if isinstance(node, ast.ClassDef) and node.name == '_QwenKvTierIO'][0]
    fake = SimpleNamespace(qwen_block_bytes=device.block_bytes, qwen_read_blocks_raw=device.read_raw, qwen_write_blocks_raw=device.write_raw)
    namespace = {'ttnn': fake, 'os': os}
    exec(segment, namespace)
    model = SimpleNamespace(_paged_kv_caches=[tuple(device.cache_objects()[i:i + 2]) for i in range(0, len(device.data), 2)], num_devices=device.chips)
    io = namespace['_QwenKvTierIO'](model)
    if chunk:
        io.CHUNK_BLOCKS = chunk
    return io


def payload_set(io, count, seed):
    rng = random.Random(seed)
    return [np.frombuffer(bytes(rng.randrange(256) for _ in range(io.block_bytes)), dtype=np.uint8).copy() for _ in range(count)]


# -- (1) the flags -------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    TIER = {'QWEN_PREFIX_HOST_TIER_GIB': '0.5'}

    def plan(self, **environ):
        return upload.upload_plan(environ, SimpleNamespace())

    def test_unset_zero_and_empty_are_off(self):
        for environ in ({}, {IO_THREADS: ''}, {IO_THREADS: '0'}, {VERIFY_THREADS: '0'}, {IO_THREADS: '0', VERIFY_THREADS: '', IO_CHUNK: ''}):
            self.assertIsNone(upload.upload_plan(environ, SimpleNamespace()), environ)
            self.assertIsNone(upload.upload_plan(environ, None), 'off is off with or without a tier: nothing to refuse')

    def test_the_numbers_are_the_plan(self):
        plan = upload.upload_plan({IO_THREADS: '4', VERIFY_THREADS: '8', IO_CHUNK: '256'}, SimpleNamespace())
        self.assertEqual((plan.io_threads, plan.io_chunk, plan.verify_threads), (4, 256, 8))
        plan = upload.upload_plan({VERIFY_THREADS: '2'}, SimpleNamespace())
        self.assertEqual((plan.io_threads, plan.io_chunk, plan.verify_threads), (0, 0, 2))
        plan = upload.upload_plan({IO_THREADS: '16'}, SimpleNamespace())
        self.assertEqual((plan.io_threads, plan.io_chunk, plan.verify_threads), (16, 0, 0))

    def test_every_malformed_value_refuses_to_start(self):
        for environ, needle in (({IO_THREADS: '-1'}, 'digits only'), ({IO_THREADS: '1e1'}, 'digits only'), ({IO_THREADS: ' 4'}, 'digits only'), ({IO_THREADS: '4.0'}, 'digits only'),
                                ({IO_THREADS: '٤'}, 'digits only'), ({IO_THREADS: '17'}, '0..16'.replace('0', '1', 1)), ({VERIFY_THREADS: '99'}, '1..16'),
                                ({IO_THREADS: '4', IO_CHUNK: '8'}, '16..4096'), ({IO_THREADS: '4', IO_CHUNK: '4097'}, '16..4096'), ({IO_THREADS: '4', IO_CHUNK: '0'}, '16..4096'),
                                ({IO_THREADS: '4', IO_CHUNK: 'x'}, 'digits only'), ({IO_CHUNK: '256'}, 'inert'), ({IO_THREADS: '0', IO_CHUNK: '256'}, 'inert')):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError) as caught:
                    upload.upload_plan(environ, SimpleNamespace())
                self.assertIn(needle, str(caught.exception))

    def test_it_needs_the_tier(self):
        for environ in ({IO_THREADS: '4'}, {VERIFY_THREADS: '4'}, {IO_THREADS: '2', VERIFY_THREADS: '2'}):
            with self.assertRaises(ValueError) as caught:
                registry_module.PrefixRegistry(budget_bytes=100, environ=environ)
            self.assertIn('need the host KV tier', str(caught.exception))
        with self.assertRaises(ValueError):
            registry_module.PrefixRegistry(budget_bytes=100, environ={IO_CHUNK: '256'})

    def test_the_contract_helper_agrees_with_the_parser(self):
        self.assertEqual(upload.flag_problems({IO_THREADS: '4', VERIFY_THREADS: '4', IO_CHUNK: '512'}), [])
        self.assertEqual(upload.flag_problems({}), [])
        self.assertTrue(upload.flag_problems({IO_THREADS: '17'}))
        self.assertTrue(upload.flag_problems({IO_THREADS: '4', IO_CHUNK: '7'}))
        self.assertTrue(any('inert' in problem for problem in upload.flag_problems({IO_CHUNK: '512'})))
        self.assertTrue(any('inert' in problem for problem in upload.flag_problems({IO_THREADS: '0', IO_CHUNK: '512'})))
        self.assertTrue(upload.flag_problems({VERIFY_THREADS: 'many'}))

    def test_the_registry_and_the_module_name_the_same_flags_and_counters(self):
        self.assertEqual(registry_module.ENV_UPLOAD_NAMES, upload.NAMES)
        self.assertEqual(registry_module.UPLOAD_STAT_NAMES, upload.STAT_NAMES)
        self.assertEqual(len(set(upload.STAT_NAMES)), len(upload.STAT_NAMES))
        self.assertFalse(set(upload.STAT_NAMES) & set(registry_module.STAT_NAMES), 'a counter the registry always had is not this module\'s')

    def test_split_covers_the_range_in_order_without_empty_pieces(self):
        for count in (0, 1, 2, 5, 16, 33, 512):
            for parts in (1, 2, 3, 4, 16, 600):
                pieces = upload.split(count, parts)
                self.assertEqual([index for low, high in pieces for index in range(low, high)], list(range(count)), (count, parts))
                self.assertTrue(all(high > low for low, high in pieces))
                self.assertLessEqual(len(pieces), max(1, min(parts, count)))
                if pieces:
                    sizes = [high - low for low, high in pieces]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)


# -- (2) the registry wires it only when asked -----------------------------------------------------------------------

class RegistryWiringTests(unittest.TestCase):
    def rig(self, **extra):
        made = tiers.Rig(num_blocks=300, budget=1 << 20, tier=64, tier_env=extra)
        self.addCleanup(made.close)
        return made

    def test_off_the_io_is_the_models_own_object_and_the_stats_are_what_they_were(self):
        made = self.rig()
        self.assertIs(made.registry.tier_io, made.kv)
        self.assertIsNone(made.registry.upload_plan)
        self.assertFalse(set(registry_module.UPLOAD_STAT_NAMES) & set(made.registry.stats))
        self.assertIsNone(made.graft.tier_hooks.verifier)
        self.assertFalse([line for line in made.lines if 'host tier upload' in line])

    def test_off_does_not_even_import_the_module(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop('qwen_tier_upload', None)
            made = tiers.Rig(num_blocks=300, budget=1 << 20, tier=64)
            try:
                self.assertNotIn('qwen_tier_upload', sys.modules)
                self.assertIs(made.registry.tier_io, made.kv)
            finally:
                made.close()

    def test_on_the_io_is_wrapped_and_delegates_the_geometry(self):
        made = self.rig(**{IO_THREADS: '3', IO_CHUNK: '32'})
        io = made.registry.tier_io
        self.assertIsInstance(io, upload.ThreadedTierIO)
        self.assertIs(io.io, made.kv)
        self.assertEqual((io.threads, io.chunk_blocks), (3, 32))
        self.assertEqual((io.block_bytes, io.fingerprint), (made.kv.block_bytes, made.kv.fingerprint))
        self.assertEqual(made.registry.tier.fingerprint, made.kv.fingerprint)
        for name in registry_module.UPLOAD_STAT_NAMES:
            self.assertEqual(made.registry.stats[name], 0, name)
        self.assertIs(io.stats, made.registry.stats)
        self.assertTrue([line for line in made.lines if line == 'host tier upload on io_threads=3 io_chunk_blocks=32 verify_threads=0 io_wrapped=1'], made.lines)

    def test_attaching_twice_never_nests_wrappers(self):
        made = self.rig(**{IO_THREADS: '2'})
        first = made.registry.tier_io
        made.registry.attach_tier_io(made.kv)
        second = made.registry.tier_io
        self.assertIsNot(first, second)
        self.assertIs(second.io, made.kv)
        made.registry.attach_tier_io(second)
        self.assertIs(made.registry.tier_io.io, made.kv)

    def test_verify_threads_alone_leave_the_io_alone_and_build_the_verifier(self):
        made = self.rig(**{VERIFY_THREADS: '4'})
        self.assertIs(made.registry.tier_io, made.kv)
        self.assertIsInstance(made.graft.tier_hooks.verifier, upload.Verifier)
        self.assertEqual(made.graft.tier_hooks.verifier.threads, 4)
        self.assertTrue([line for line in made.lines if line == 'host tier upload on io_threads=0 io_chunk_blocks=0 verify_threads=4 io_wrapped=0'], made.lines)


# -- (3) the threaded IO against the adapter ------------------------------------------------------------------------

class ThreadedIOTests(unittest.TestCase):
    def pair(self, chunk=4, threads=4, blocks=64, caches=3, **kwargs):
        stock_device, threaded_device = FakeDevice(blocks=blocks, caches=caches), FakeDevice(blocks=blocks, caches=caches)
        stock = stock_adapter(stock_device, chunk)
        stats = dict.fromkeys(upload.STAT_NAMES, 0)
        wrapped = upload.ThreadedTierIO(stock_adapter(threaded_device, chunk), threads, stats=stats, **kwargs)
        self.addCleanup(wrapped.close)
        return stock_device, stock, threaded_device, wrapped, stats

    def test_it_discovers_the_ops_in_the_adapters_own_namespace(self):
        device = FakeDevice()
        io = stock_adapter(device)
        read, write = upload.device_ops(io)
        self.assertEqual((read, write), (device.read_raw, device.write_raw))

    def test_a_write_leaves_the_device_exactly_as_the_adapters_does(self):
        for threads in (1, 2, 4, 16):
            for count in (1, 3, 4, 5, 9, 21):
                with self.subTest(threads=threads, count=count):
                    stock_device, stock, threaded_device, wrapped, stats = self.pair(threads=threads)
                    rng = random.Random(count * 31 + threads)
                    ids = rng.sample(range(64), count)
                    payloads = payload_set(stock, count, count)
                    stock.write_blocks(ids, payloads)
                    wrapped.write_blocks(ids, payloads)
                    for want, got in zip(stock_device.data, threaded_device.data):
                        self.assertTrue(np.array_equal(want, got))
                    self.assertEqual(threaded_device.log, stock_device.log, 'the same calls, in the same order, with the same blocks')
                    self.assertEqual(stats['tier_io_write_calls'], 1)
                    self.assertEqual(stats['tier_io_write_blocks'], count)

    def test_a_read_returns_the_adapters_bytes_with_and_without_a_destination(self):
        for threads in (1, 3, 8):
            for count in (1, 4, 6, 17):
                with self.subTest(threads=threads, count=count):
                    stock_device, stock, threaded_device, wrapped, stats = self.pair(threads=threads)
                    ids = random.Random(count).sample(range(64), count)
                    want = stock.read_blocks(ids)
                    got = wrapped.read_blocks(ids)
                    self.assertEqual([bytes(a) for a in want], [bytes(b) for b in got])
                    self.assertEqual(threaded_device.log, stock_device.log)
                    slab = [bytearray(stock.block_bytes) for _ in ids]
                    returned = wrapped.read_blocks(ids, into=slab)
                    self.assertEqual([bytes(a) for a in want], [bytes(b) for b in slab], 'the destination buffers hold the blocks')
                    self.assertEqual([bytes(a) for a in want], [bytes(b) for b in returned])
                    self.assertEqual(stats['tier_io_read_calls'], 2)

    def test_a_round_trip_through_the_wrapper_restores_what_was_lost(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=5, threads=4)
        before = threaded_device.snapshot()
        ids = [60, 3, 17, 4, 5, 30, 31, 2, 9, 11, 12]
        payloads = wrapped.read_blocks(ids)
        for cache in threaded_device.data:
            cache[:, ids, :] = 0
        wrapped.write_blocks(ids, payloads)
        for was, now in zip(before, threaded_device.data):
            self.assertTrue(np.array_equal(was, now))

    def test_the_call_size_knob_changes_the_calls_and_not_the_bytes(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=2, chunk_blocks=16)
        ids = list(range(40))
        payloads = payload_set(stock, 40, 5)
        wrapped.write_blocks(ids, payloads)
        sizes = sorted({len(row[2]) for row in threaded_device.log})
        self.assertEqual(sizes, [8, 16], '40 blocks at 16 a call: 16, 16, 8 (per cache), not the adapter\'s 4')
        stock.write_blocks(ids, payloads)
        for want, got in zip(stock_device.data, threaded_device.data):
            self.assertTrue(np.array_equal(want, got))

    def test_the_default_call_size_is_the_adapters_own_and_follows_it(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=7, threads=2)
        wrapped.read_blocks(list(range(20)))
        self.assertEqual({len(row[2]) for row in threaded_device.log}, {7, 6})
        wrapped.io.CHUNK_BLOCKS = 5
        threaded_device.log.clear()
        wrapped.read_blocks(list(range(20)))
        self.assertEqual({len(row[2]) for row in threaded_device.log}, {5})

    def test_no_two_device_calls_are_ever_in_flight_together(self):
        for op in ('write', 'read'):
            stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=3, threads=8)
            threaded_device.delay = 0.002
            ids = list(range(30))
            if op == 'write':
                wrapped.write_blocks(ids, payload_set(stock, 30, 9))
            else:
                wrapped.read_blocks(ids)
            self.assertEqual(threaded_device.max_in_flight, 1, op)
            self.assertEqual(len(threaded_device.log), 10 * 3)

    def test_staging_the_next_call_runs_beside_the_current_device_call(self):
        """The first device write waits (up to 10 s) for the buffer of the SECOND call to hold that call's bytes: only a worker staging beside the device call can have put them there."""
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=2)
        payloads = payload_set(stock, 8, 11)
        seen = {}

        def hook(op, number, cache, blocks):
            if op == 'write' and number == 0:
                deadline = time.time() + 10.0
                want = np.stack([np.frombuffer(bytes(payloads[offset]), dtype=np.uint8).reshape(len(stock.caches), stock.chips, stock.slice_bytes)[1]
                                 for offset in range(4)], axis=1)      # job 1 is blocks 0-3 of the second cache (chunk by chunk, cache by cache)
                while time.time() < deadline:
                    buffers = wrapped._buffers
                    if len(buffers) > 1 and buffers[1] is not None and np.array_equal(buffers[1][:want.size].reshape(want.shape), want):
                        seen['staged_during_call'] = True
                        return
                    time.sleep(0.001)
                seen['staged_during_call'] = False

        threaded_device.hook = hook
        wrapped.write_blocks(list(range(8)), payloads)
        self.assertTrue(seen.get('staged_during_call'), 'the next call\'s bytes were staged while the first device call was running')

    def test_a_read_copies_out_beside_the_next_device_call(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=2)
        sink = [bytearray(stock.block_bytes) for _ in range(8)]
        seen = {}

        def hook(op, number, cache, blocks):
            if op == 'read' and number == 1:
                # call 0 (cache 0, blocks 0-3) was copied out by a worker while this call runs, or is about to be: wait for it
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if any(any(chunk) for chunk in (bytes(view[:stock.slice_bytes]) for view in sink[:4])):
                        seen['copied_during_call'] = True
                        return
                    time.sleep(0.001)
                seen['copied_during_call'] = False

        threaded_device.hook = hook
        wrapped.read_blocks(list(range(8)), into=sink)
        self.assertTrue(seen.get('copied_during_call'))

    def test_a_refusing_device_raises_to_the_caller_and_the_next_call_works(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=4)
        ids = list(range(12))
        payloads = payload_set(stock, 12, 3)
        threaded_device.fail_on = ('write', 4)
        with self.assertRaisesRegex(RuntimeError, 'write refused'):
            wrapped.write_blocks(ids, payloads)
        self.assertEqual(stats['tier_io_write_calls'], 0, 'a failed call is not counted')
        threaded_device.fail_on = ('read', 2)
        with self.assertRaisesRegex(RuntimeError, 'read refused'):
            wrapped.read_blocks(ids)
        threaded_device.fail_on = None
        threaded_device.log.clear()
        wrapped.write_blocks(ids, payloads)
        stock.write_blocks(ids, payloads)
        got = wrapped.read_blocks(ids)
        self.assertEqual([bytes(a) for a in got], [bytes(a) for a in stock.read_blocks(ids)])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(threaded_device.data, stock_device.data)))

    def test_a_failing_worker_surfaces_and_leaves_none_running(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=3)
        ids = list(range(12))
        payloads = payload_set(stock, 12, 4)
        with mock.patch.object(upload, 'split', lambda count, parts: [(0, count + 3)]):
            with self.assertRaises(IndexError):
                wrapped.write_blocks(ids, payloads)
            with self.assertRaises(IndexError):
                wrapped.read_blocks(ids)
        self.assertEqual(threaded_device.in_flight, 0)
        wrapped.close()
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith('qwen-tier-io') and t.is_alive()], 'close() joins every worker')
        wrapped.write_blocks(ids, payloads)           # the wrapper is reusable after a close: a new pool
        stock.write_blocks(ids, payloads)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(threaded_device.data, stock_device.data)))

    def test_the_adapters_refusals_are_the_wrappers(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair()
        for call in ('write_short', 'write_count', 'read_into'):
            with self.subTest(call=call):
                errors = []
                for io in (stock, wrapped):
                    with self.assertRaises(ValueError) as caught:
                        if call == 'write_short':
                            io.write_blocks([1], [b'short'])
                        elif call == 'write_count':
                            io.write_blocks([1, 2], [b'x'])
                        else:
                            io.read_blocks([1, 2], into=[bytearray(3)])
                    errors.append(str(caught.exception))
                self.assertEqual(errors[0], errors[1])
        self.assertEqual(threaded_device.log, [], 'refused before the device is touched')

    def test_nothing_to_do_is_nothing_done(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair()
        self.assertEqual(wrapped.read_blocks([]), [])
        wrapped.write_blocks([], [])
        self.assertEqual(threaded_device.log, [])

    def test_a_block_set_larger_than_the_pipeline_depth_is_exact(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=2, threads=5, blocks=64, caches=4)
        ids = random.Random(77).sample(range(64), 31)
        payloads = payload_set(stock, 31, 77)
        wrapped.write_blocks(ids, payloads)
        stock.write_blocks(ids, payloads)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(threaded_device.data, stock_device.data)))
        self.assertEqual(threaded_device.log, stock_device.log)

    def test_the_counters_account_for_the_work(self):
        stock_device, stock, threaded_device, wrapped, stats = self.pair(chunk=4, threads=2)
        threaded_device.delay = 0.001
        wrapped.write_blocks(list(range(8)), payload_set(stock, 8, 1))
        wrapped.read_blocks(list(range(8)))
        self.assertEqual((stats['tier_io_write_calls'], stats['tier_io_write_blocks'], stats['tier_io_read_calls'], stats['tier_io_read_blocks']), (1, 8, 1, 8))
        for name in ('tier_io_write_ms', 'tier_io_write_device_ms', 'tier_io_write_stage_ms', 'tier_io_read_ms', 'tier_io_read_device_ms', 'tier_io_read_stage_ms'):
            self.assertGreater(stats[name], 0.0, name)
        self.assertGreaterEqual(stats['tier_io_write_ms'], stats['tier_io_write_device_ms'])
        self.assertGreaterEqual(stats['tier_io_read_ms'], stats['tier_io_read_device_ms'])

    def test_a_wrapper_without_a_stats_dict_counts_nothing_and_works(self):
        device = FakeDevice()
        io = upload.ThreadedTierIO(stock_adapter(device, 4), 2)
        self.addCleanup(io.close)
        io.write_blocks([1, 2], payload_set(io, 2, 1))
        self.assertEqual(io.stats, {})

    def test_the_zero_thread_wrapper_is_refused(self):
        with self.assertRaises(ValueError):
            upload.ThreadedTierIO(stock_adapter(FakeDevice()), 0)


# -- (3b) the whole graft over the real adapter and a numpy device ---------------------------------------------------

class DeviceView(object):
    """The Rig's `kv.device` (block id -> the 32 bytes the block holds) over a FakeDevice of 2 caches x 2 chips x 8 bytes: the layout of a payload (cache, chip, slice)."""

    def __init__(self, device):
        self.device = device

    def get(self, block, default=None):
        return b''.join(bytes(self.device.data[cache][chip, block, :]) for cache in range(len(self.device.data)) for chip in range(self.device.chips))

    def __setitem__(self, block, payload):
        payload = bytes(payload)
        size = self.device.slice_bytes
        for cache in range(len(self.device.data)):
            for chip in range(self.device.chips):
                start = (cache * self.device.chips + chip) * size
                self.device.data[cache][chip, block, :] = np.frombuffer(payload[start:start + size], dtype=np.uint8)


class AdapterKvIO(object):
    """tiers.FakeKvIO's replacement: the model graft's own adapter over a numpy device (instantiating the class returns the adapter, dressed with the attributes the Rig reads)."""

    block_bytes = 32
    pool_blocks = 400

    def __new__(cls):
        device = FakeDevice(caches=2, chips=2, blocks=cls.pool_blocks, slice_bytes=8, seed=7)
        for cache in device.data:
            cache[:] = 0
        io = stock_adapter(device, chunk=5)
        io.raw_device = device
        io.device = DeviceView(device)
        io.reads, io.writes = [], []
        return io


class WholeGraftTests(unittest.TestCase):
    """A returning session restored through the real scheduler graft, with the adapter's own IO and with the threaded one: the same bytes in the same blocks."""

    def served(self, **env):
        with mock.patch.object(tiers, 'FakeKvIO', AdapterKvIO):
            made = tiers.Rig(num_blocks=POOL, budget=1 << 20, tier=64, tier_env=env)
        self.addCleanup(made.close)
        a = tiers.Request('a1', tiers.span(FIRST_TURN, 1), salt=tiers.salt_of('alpha'))
        made.serve(a)
        made.serve(tiers.Request('f1', tiers.span(FILLER, 2), salt=tiers.salt_of('bravo')))
        start, grant = made.serve(tiers.Request('a2', a.all_token_ids + tiers.span(700, 3), salt=a.cache_salt))
        return made, start, grant

    def test_the_threaded_io_restores_what_the_adapters_does(self):
        plain, start, grant = self.served()
        for threads in ('1', '4'):
            with self.subTest(threads=threads):
                threaded, start_t, grant_t = self.served(**{IO_THREADS: threads, VERIFY_THREADS: threads})
                self.assertIsInstance(threaded.registry.tier_io, upload.ThreadedTierIO)
                self.assertEqual((start_t, grant_t.q), (start, grant.q))
                self.assertEqual(start_t, 8192)
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(plain.kv.raw_device.data, threaded.kv.raw_device.data)), 'every cache, every chip, every block')
                self.assertTrue(threaded.check_kv(), 'every cached block, restored ones included, holds its hash\'s bytes')
                stats = threaded.registry.stats
                self.assertEqual((stats['tier_restore_requests'], stats['tier_restore_blocks'], stats['tier_digest_failures']), (1, 49, 0))
                self.assertGreater(stats['tier_io_read_blocks'], 0, 'the spill went through the threaded read')
                self.assertEqual(stats['tier_io_write_blocks'], 49, 'the restore through the threaded write')
                self.assertEqual({key: value for key, value in stats.items() if not key.endswith('_ms') and key not in registry_module.UPLOAD_STAT_NAMES},
                                 {key: value for key, value in plain.registry.stats.items() if not key.endswith('_ms')})

    def test_a_corrupting_device_is_caught_by_the_audit_through_the_threaded_io(self):
        with mock.patch.object(tiers, 'FakeKvIO', AdapterKvIO):
            made = tiers.Rig(num_blocks=POOL, budget=1 << 20, tier=64, tier_env={'QWEN_PREFIX_HOST_TIER_AUDIT': '1', IO_THREADS: '3'})
        self.addCleanup(made.close)
        a = tiers.Request('a1', tiers.span(FIRST_TURN, 1), salt=tiers.salt_of('alpha'))
        made.serve(a)
        made.serve(tiers.Request('f1', tiers.span(FILLER, 2), salt=tiers.salt_of('bravo')))
        raw_write = made.registry.tier_io.io.raw_device.write_raw

        def corrupting(cache, data, blocks):
            raw_write(cache, data, blocks)
            cache_array = made.registry.tier_io.io.raw_device.data[cache.index]
            cache_array[0, blocks[0], 0] ^= 1

        made.registry.tier_io._ops = (made.registry.tier_io.io.raw_device.read_raw, corrupting)
        with self.assertRaises(Exception) as caught:
            made.serve(tiers.Request('a2', a.all_token_ids + tiers.span(700, 3), salt=a.cache_salt))
        self.assertIn('read back differently', str(caught.exception))


# -- (4) the digest checks on threads --------------------------------------------------------------------------

POOL, FIRST_TURN, FILLER = 220, 8300, 9000      # blocks, tokens, tokens: the filler takes 49 of A's blocks into the tier and A's next turn brings all 49 back


def run_traffic(test, **extra):
    """A's first turn, another tenant's filler that pushes 49 of A's blocks to the tier, then A's next turn (restored from the tier). The record number `corrupt` (1-based)
    of the tier is changed in host memory before the return, when given. -> (rig, the return's start position, its grant)."""
    corrupt = extra.pop('corrupt', 0)
    made = tiers.Rig(num_blocks=POOL, budget=1 << 20, tier=64, tier_env=extra)
    test.addCleanup(made.close)
    a = tiers.Request('a1', tiers.span(FIRST_TURN, 1), salt=tiers.salt_of('alpha'))
    made.serve(a)
    made.serve(tiers.Request('f1', tiers.span(FILLER, 2), salt=tiers.salt_of('bravo')))
    if corrupt:
        victim = list(made.registry.tier.records.values())[corrupt - 1]
        victim.payload = bytes([victim.payload[0] ^ 1]) + victim.payload[1:]
    start, grant = made.serve(tiers.Request('a2', a.all_token_ids + tiers.span(700, 3), salt=a.cache_salt))
    return made, start, grant


class VerifyThreadTests(unittest.TestCase):
    def stats_of(self, made):
        """The registry's counters but the timers (they are wall time) and this module's own."""
        return {name: value for name, value in made.registry.stats.items()
                if name not in registry_module.UPLOAD_STAT_NAMES and not name.endswith('_ms')}

    def test_the_parallel_check_restores_the_same_session_with_the_same_counters(self):
        for mode in ('all', 'sample'):
            with self.subTest(mode=mode):
                plain, start, grant = run_traffic(self, QWEN_PREFIX_HOST_TIER_VERIFY=mode)
                threaded, start_t, grant_t = run_traffic(self, QWEN_PREFIX_HOST_TIER_VERIFY=mode, **{VERIFY_THREADS: '4'})
                self.assertEqual((start_t, grant_t.q), (start, grant.q))
                self.assertEqual(self.stats_of(threaded), self.stats_of(plain), 'every counter the sequential loop moves, moved the same')
                self.assertEqual(threaded.graft.tier_hooks.checked, plain.graft.tier_hooks.checked)
                self.assertTrue(threaded.check_kv())
                self.assertEqual(threaded.registry.stats['tier_verify_blocks'], {'all': 49, 'sample': 49 // registry_module.TIER_VERIFY_EVERY}[mode])
                self.assertEqual(threaded.registry.stats['tier_verify_batches'], 1)
                self.assertEqual(threaded.registry.stats['tier_restore_blocks'], 49)
                self.assertEqual(start_t, 8192)

    def test_a_corrupt_record_is_dropped_and_the_restore_refused_either_way(self):
        for mode, positions in (('all', (1, 2, 5, 49)), ('sample', (1, 5, 16, 32, 48))):
            for position in positions:
                with self.subTest(mode=mode, position=position):
                    plain, start, _ = run_traffic(self, corrupt=position, QWEN_PREFIX_HOST_TIER_VERIFY=mode)
                    threaded, start_t, _ = run_traffic(self, corrupt=position, QWEN_PREFIX_HOST_TIER_VERIFY=mode, **{VERIFY_THREADS: '4'})
                    self.assertEqual(start_t, start)
                    self.assertEqual(self.stats_of(threaded), self.stats_of(plain))
                    self.assertEqual(threaded.graft.tier_hooks.checked, plain.graft.tier_hooks.checked, 'the sampling count stops where the sequential loop stopped')
                    self.assertEqual(sorted(threaded.registry.tier.records), sorted(plain.registry.tier.records), 'the same record was dropped')
                    self.assertEqual(threaded.kv.device, plain.kv.device)
                    if mode == 'all' or position % registry_module.TIER_VERIFY_EVERY == 0:
                        self.assertEqual(start_t, 0, 'a record that fails its digest is a miss')
                        self.assertEqual(threaded.registry.stats['tier_digest_failures'], 1)
                        self.assertTrue(threaded.check_kv())

    def test_digest_failures_is_the_sequential_loop_when_there_is_no_verifier(self):
        made, _, _ = run_traffic(self)
        hooks = made.graft.tier_hooks
        records = list(made.registry.tier.records.values())[:6]
        self.assertIsNone(hooks.verifier)
        before = hooks.checked
        self.assertEqual(list(hooks.digest_failures(records)), [])
        self.assertEqual(hooks.checked, before + len(records) if made.registry.tier_config.verify != 'off' else before)

    def test_off_mode_checks_nothing_and_counts_nothing(self):
        made, start, grant = run_traffic(self, QWEN_PREFIX_HOST_TIER_VERIFY='off', **{VERIFY_THREADS: '4'})
        self.assertEqual(made.registry.stats['tier_verify_batches'], 0)
        self.assertEqual(made.graft.tier_hooks.checked, 0)
        self.assertGreater(made.registry.stats['tier_restore_blocks'], 0)

    def test_a_record_with_no_digest_yet_claims_nothing(self):
        made, _, _ = run_traffic(self, **{VERIFY_THREADS: '4'})
        hooks = made.graft.tier_hooks
        record = list(made.registry.tier.records.values())[0]
        digest, record.digest = record.digest, None
        checks = made.registry.stats['tier_digest_checks']
        self.assertEqual(list(hooks.digest_failures([record] * 3)), [])
        self.assertEqual(made.registry.stats['tier_digest_checks'], checks)
        record.digest = digest

    def test_the_verifier_returns_verdicts_in_order_and_survives_a_worker_error(self):
        store = registry_module.TierStore(1 << 16, digest_inline=True)
        store.configure(64)
        verifier = upload.Verifier(4, stats=dict.fromkeys(upload.STAT_NAMES, 0))
        self.addCleanup(verifier.close)
        records = []
        for index in range(10):
            slots = store.reserve(1)
            slots[0].view[:] = bytes([index]) * 64
            records.append(store.put(b'k%02d' % index, slots[0].view, 'tenant', b'chain', index))
        records = [store.get(b'k%02d' % index) for index in range(10)]
        self.assertEqual(verifier.verdicts(store, records), [True] * 10)
        records[3].payload[0] ^= 1
        verdicts = verifier.verdicts(store, records)
        self.assertEqual([i for i, v in enumerate(verdicts) if v is False], [3])
        with mock.patch.object(store, 'verify', side_effect=OSError('boom')):
            with self.assertRaises(OSError):
                verifier.verdicts(store, records)
        self.assertEqual(verifier.verdicts(store, records[:1]), [True])


# -- (4b) the profile contract -----------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s5'
    AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s5-audit'

    @classmethod
    def setUpClass(cls):
        import json

        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            cls.profiles = json.load(handle)['profiles']

    def profile(self, name=None, **env):
        import copy

        profile = copy.deepcopy(self.profiles[name or self.SHIP])
        for key, value in env.items():
            if value is None:
                profile['env'].pop(key, None)
            else:
                profile['env'][key] = value
        return profile

    def problems(self, profile):
        import serving_c2_contract as contract

        return contract.prefix_policy_problems(profile)

    def test_the_contract_names_what_the_module_names(self):
        import serving_c2_contract as contract

        self.assertEqual(contract.PREFIX_UPLOAD_NAMES, upload.NAMES)

    def test_the_shipped_profiles_name_none_of_them_and_are_clean(self):
        for name, profile in self.profiles.items():
            self.assertFalse(set(upload.NAMES) & set(profile.get('env') or {}), '%s: no profile names an upload lever until the owner moves it' % name)
        self.assertEqual(self.problems(self.profile()), [])

    def test_a_traffic_profile_whose_waiver_names_the_tier_may_name_them(self):
        self.assertEqual(self.problems(self.profile(**{IO_THREADS: '4', VERIFY_THREADS: '4', IO_CHUNK: '256'})), [])
        self.assertEqual(self.problems(self.profile(**{IO_THREADS: '8'})), [])

    def test_a_traffic_profile_whose_waiver_does_not_name_the_tier_may_not(self):
        import serving_c2_contract as contract

        profile = self.profile(**{IO_THREADS: '4'})
        profile[contract.TRAFFIC_WAIVER]['levers'].pop('QWEN_PREFIX_HOST_TIER_GIB')
        found = [problem for problem in self.problems(profile) if 'upload levers' in problem]
        self.assertTrue(found, self.problems(profile))
        self.assertIn(IO_THREADS, found[0])

    def test_a_gate_only_profile_may_name_them_without_a_waiver(self):
        self.assertEqual(self.problems(self.profile(self.AUDIT, **{IO_THREADS: '4', VERIFY_THREADS: '8'})), [])

    def test_the_values_are_checked_as_the_registry_checks_them(self):
        for env, needle in (({IO_THREADS: '17'}, '1..16'), ({IO_THREADS: '-2'}, 'digits only'), ({VERIFY_THREADS: 'x'}, 'digits only'),
                            ({IO_THREADS: '4', IO_CHUNK: '8'}, '16..4096'), ({IO_CHUNK: '256'}, 'inert')):
            with self.subTest(env=env):
                found = self.problems(self.profile(self.AUDIT, **env))
                self.assertTrue(any(needle in problem for problem in found), found)

    def test_they_need_the_tier_and_the_registry(self):
        found = self.problems(self.profile(self.AUDIT, QWEN_PREFIX_HOST_TIER_GIB=None, QWEN_PREFIX_HOST_TIER_SPILL_MODE=None, QWEN_PREFIX_TIER_RESTORE_SLICE=None,
                                           QWEN_PREFIX_TIER_RESTORE_SLICE_MS=None, QWEN_PREFIX_CKPT_PRECONVERTED=None, QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT=None,
                                           **{IO_THREADS: '4'}))
        self.assertTrue(any('nothing to move' in problem for problem in found), found)
        found = self.problems(self.profile(self.AUDIT, QWEN_PREFIX_REUSE=None, **{IO_THREADS: '4'}))
        self.assertTrue(any('registry does not exist' in problem for problem in found), found)

    def test_zero_is_off_and_needs_nothing(self):
        found = self.problems(self.profile(self.AUDIT, QWEN_PREFIX_HOST_TIER_GIB=None, QWEN_PREFIX_HOST_TIER_SPILL_MODE=None, QWEN_PREFIX_TIER_RESTORE_SLICE=None,
                                           QWEN_PREFIX_TIER_RESTORE_SLICE_MS=None, QWEN_PREFIX_CKPT_PRECONVERTED=None, QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT=None,
                                           **{IO_THREADS: '0', VERIFY_THREADS: '0'}))
        self.assertFalse(any('nothing to move' in problem for problem in found), found)

    def test_an_inherited_value_is_dropped_under_a_profile_that_never_named_it(self):
        import serving_c2_contract as contract

        environ = {IO_THREADS: '4', VERIFY_THREADS: '4', IO_CHUNK: '256', 'UNRELATED': 'kept'}
        contract.apply_environment(self.profile(), environ)
        self.assertEqual({name: environ.get(name) for name in upload.NAMES}, dict.fromkeys(upload.NAMES))
        self.assertEqual(environ['UNRELATED'], 'kept')
        environ = {IO_THREADS: '4'}
        contract.apply_environment(self.profile(**{IO_THREADS: '2'}), environ)
        self.assertEqual(environ[IO_THREADS], '2', 'a profile that names it sets it')


# -- (5) the checkpoint store's arithmetic -------------------------------------------------------------------------

class CheckpointBudgetTests(unittest.TestCase):
    STORE = 16 * (1 << 30)

    def test_the_sizes_are_the_ones_the_docs_quote(self):
        self.assertEqual(upload.checkpoint_bytes(state_bf16=True), 78446592)
        self.assertEqual(upload.checkpoint_bytes(state_bf16=False), 153944064)
        self.assertEqual(upload.checkpoint_bytes(state_bf16=True, preconverted=True), 106954752)
        self.assertEqual(upload.checkpoint_bytes(state_bf16=False, preconverted=True), 153944064 + 28508160)
        self.assertEqual(upload.checkpoint_bytes(state_bf16=True, preconverted=True, window=True), 106954752 + 104857600)

    def test_they_agree_with_the_registrys_own_arithmetic(self):
        self.assertEqual(upload.checkpoint_bytes(state_bf16=True), registry_module.checkpoint_nbytes({'QWEN35_GDN_STATE_BF16': '1'}))
        self.assertEqual(upload.checkpoint_bytes(state_bf16=False), registry_module.checkpoint_nbytes({}))

    def test_the_padded_size_is_the_models_tile_padding(self):
        import torch

        chips = 4
        rec = torch.zeros(chips, 12, 128, 128, dtype=torch.bfloat16)        # one layer's state over the mesh: 48 value heads, 12 a chip
        carry = torch.zeros(chips, 3, 2560, dtype=torch.bfloat16)           # its conv carry: [3, 10240] over the mesh, 2560 a chip
        segment = [ast.get_source_segment(patcher.MODEL_ADAPTER, node) for node in ast.parse(patcher.MODEL_ADAPTER).body
                   if isinstance(node, ast.FunctionDef) and node.name == '_qwen_prefix_padded_nbytes'][0]
        namespace = {'_QWEN_PREFIX_TILE': 32}
        exec(segment, namespace)
        padded = namespace['_qwen_prefix_padded_nbytes']
        self.assertEqual(padded(rec, chips), rec.numel() * 2, 'the state is already whole tiles')
        self.assertEqual(padded(carry, chips), chips * 32 * 2560 * 2, 'the carry\'s 3 rows are 32')
        self.assertEqual(48 * (padded(rec, chips) + padded(carry, chips)), upload.checkpoint_bytes(state_bf16=True, preconverted=True))
        self.assertEqual(48 * (rec.numel() + carry.numel()) * 2, upload.checkpoint_bytes(state_bf16=True))

    def test_how_many_fit_in_the_stores_the_doc_discusses(self):
        eight, sixteen = 8 * (1 << 30), 16 * (1 << 30)
        self.assertEqual(upload.checkpoints_that_fit(eight, state_bf16=True), 109)
        self.assertEqual(upload.checkpoints_that_fit(eight, state_bf16=True, preconverted=True), 80)
        self.assertEqual(upload.checkpoints_that_fit(sixteen, state_bf16=True), 219)
        self.assertEqual(upload.checkpoints_that_fit(sixteen, state_bf16=True, preconverted=True), 160)
        self.assertEqual(upload.checkpoints_that_fit(sixteen, state_bf16=True, preconverted=True, window=True), 81)
        self.assertEqual(upload.checkpoints_that_fit(sixteen, state_bf16=True, window=True), 93)


if __name__ == '__main__':
    unittest.main()
