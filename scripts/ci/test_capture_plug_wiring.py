"""The capture plug's seams: the packed block opens it before its warm forward and seals it after its last capture, an engine build
opens its own zone under the packed one, and both are off unless QWEN_FAST_CAPTURE_PLUG=1 at QWEN_FAST_TP=4.

The block and the runtime are driven through their own methods on objects built without their constructors (a constructor needs a
device); the plug is a fake that records what it is asked."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capture_plug  # noqa: E402
import packed_verifier  # noqa: E402
import serving_runtime  # noqa: E402
import trace_census  # noqa: E402


class FakePlug:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.calls, self.sealed = args, kwargs, [], False

    def open(self):
        self.calls.append('open')

    def seal(self):
        self.calls.append('seal')
        self.sealed = True

    def verify_extents(self, ranges):
        self.calls.append(('verify', list(ranges)))

    def close(self):
        self.calls.append('close')

    def abandon(self):
        self.calls.append('abandon')

    def monitor(self, where=''):
        self.calls.append(('monitor', where))

    def zone_lo(self):
        return 0x4000


class Isolated(unittest.TestCase):
    def setUp(self):
        stack = patch.dict(os.environ)
        stack.start()
        self.addCleanup(stack.stop)
        for name in list(os.environ):
            if name.startswith('QWEN_FAST_CAPTURE_PLUG') or name == 'QWEN_FAST_TP':
                del os.environ[name]
        del capture_plug.PACKED[:]
        self.addCleanup(lambda: capture_plug.PACKED.clear())
        trace_census.reset()
        self.addCleanup(trace_census.reset)


class PackedBlockTests(Isolated):
    def block(self):
        block = object.__new__(packed_verifier.PackedVerifierEngine)
        block.plug, block.mesh, block.stage = None, 'mesh', 'start'
        return block

    def test_the_pair_and_an_unset_flag_open_nothing(self):
        block = self.block()
        os.environ['QWEN_FAST_CAPTURE_PLUG'] = '1'
        block.open_capture_plug('ops')                      # QWEN_FAST_TP unset: the pair
        self.assertIsNone(block.plug)
        os.environ['QWEN_FAST_TP'] = '4'
        del os.environ['QWEN_FAST_CAPTURE_PLUG']
        block.open_capture_plug('ops')                      # four cards, flag off
        self.assertIsNone(block.plug)
        block.seal_capture_plug()                           # nothing to seal
        self.assertEqual(capture_plug.PACKED, [])

    def test_the_flag_at_four_cards_opens_the_zone_before_the_captures_and_seals_after_them(self):
        os.environ.update({'QWEN_FAST_TP': '4', 'QWEN_FAST_CAPTURE_PLUG': '1'})
        block = self.block()
        with patch.object(capture_plug.Plug, 'packed', classmethod(lambda cls, settings, operations, mesh, log: FakePlug(settings, mesh))):
            trace_census.SEQUENCE = 3
            block.open_capture_plug('ops')
            self.assertEqual(block.plug.calls, ['open'])
            self.assertEqual(block.plug.args[1], 'mesh')
            self.assertEqual(block.plug.args[0]['leave'], 4096 * capture_plug.MB)
            trace_census.TRACES.extend([dict(seq=2, site='before', ranges=[(1, 2, 'DRAM')], handle='a'),
                                        dict(seq=4, site='packed', ranges=[(10, 20, 'DRAM'), (5, 6, 'L1')], handle='b')])
            block.seal_capture_plug()
        self.assertEqual(block.plug.calls, ['open', 'seal', ('verify', [(10, 20, 'DRAM'), (5, 6, 'L1')])])
        self.assertEqual(block.stage, 'capture plug seal')
        self.assertEqual(capture_plug.PACKED, [block.plug])

    def closing(self, block, wait):
        block.phase, block.operations = 'idle', SimpleNamespace(synchronize_device=lambda mesh: None,
                                                               release_trace=lambda mesh, trace: None)
        block.commits, block.trace, block.output, block.fixture, block.feature_capture = [{}], None, None, None, None
        block.taps, block.checkpoints, block.initial, block.carries, block.carry_addresses = [], [], [], [], []
        block.pending_segments = {}
        block.close(wait=wait)

    def test_closing_the_block_gives_the_plugs_back_after_its_traces_and_a_failed_attach_frees_nothing(self):
        block = self.block()
        block.plug = FakePlug()
        capture_plug.PACKED.append(block.plug)
        plug = block.plug
        self.closing(block, True)
        self.assertEqual((plug.calls, capture_plug.PACKED, block.plug), (['close'], [], None))
        failed = self.block()
        failed.plug = FakePlug()
        plug = failed.plug
        self.closing(failed, False)
        self.assertEqual((plug.calls, failed.plug), (['abandon'], None))


class EnginePlugTests(Isolated):
    def model(self):
        return SimpleNamespace(mesh_device='mesh')

    def test_off_for_the_pair_an_unset_flag_and_the_packed_only_setting(self):
        self.assertIsNone(serving_runtime.open_engine_plug('ops', self.model(), {}))
        self.assertIsNone(serving_runtime.open_engine_plug('ops', self.model(), {'QWEN_FAST_CAPTURE_PLUG': '1', 'QWEN_FAST_CAPTURE_PLUG_ENGINES': '1'}))
        self.assertIsNone(serving_runtime.open_engine_plug('ops', self.model(), {'QWEN_FAST_TP': '4'}))
        self.assertIsNone(serving_runtime.open_engine_plug('ops', self.model(), {'QWEN_FAST_TP': '4', 'QWEN_FAST_CAPTURE_PLUG': '1'}))

    def test_the_engine_zone_is_checked_then_opened_under_the_packed_zone(self):
        environ = {'QWEN_FAST_TP': '4', 'QWEN_FAST_CAPTURE_PLUG': '1', 'QWEN_FAST_CAPTURE_PLUG_ENGINES': '1'}
        packed = FakePlug()
        packed.sealed = True
        capture_plug.PACKED.append(packed)
        made = []

        def engine(cls, settings, operations, mesh, log=None, ceiling=None, **extra):
            made.append(FakePlug(settings, operations, mesh, ceiling))
            return made[-1]

        with patch.object(capture_plug.Plug, 'engine', classmethod(engine)):
            plug = serving_runtime.open_engine_plug('ops', self.model(), environ)
        self.assertIs(plug, made[0])
        self.assertEqual(plug.calls, [('monitor', 'before engine build'), 'open'])
        self.assertEqual(plug.args[1:], ('ops', 'mesh', 0x4000))
        self.assertEqual(plug.args[0]['engine_leave'], 1024 * capture_plug.MB)

    def test_sealing_plugs_the_engine_verifies_its_extents_and_ties_the_plugs_life_to_the_engines(self):
        closed = []
        request = SimpleNamespace(close=lambda *args: closed.append(('engine', args)) or 'closed')
        plug = FakePlug()
        plug.close = lambda: closed.append('plug')
        trace_census.TRACES.extend([dict(seq=1, site='old', ranges=[(1, 2, 'DRAM')], handle='a'),
                                    dict(seq=7, site='mine', ranges=[(30, 40, 'DRAM')], handle='b')])
        serving_runtime.seal_engine_plug(plug, request, 5)
        self.assertEqual(plug.calls, ['seal', ('verify', [(30, 40, 'DRAM')])])
        self.assertEqual(request.close('request-1'), 'closed')
        self.assertEqual(closed, [('engine', ('request-1',)), 'plug'], 'the engine releases its traces first, then the plugs go')

    def test_the_plug_is_given_back_even_when_the_engines_close_raises(self):
        closed = []

        def broken(*args):
            raise RuntimeError('close failed')

        request = SimpleNamespace(close=broken)
        plug = FakePlug()
        plug.close = lambda: closed.append('plug')
        serving_runtime.seal_engine_plug(plug, request, 0)
        with self.assertRaises(RuntimeError):
            request.close('r')
        self.assertEqual(closed, ['plug'])


if __name__ == '__main__':
    unittest.main()
