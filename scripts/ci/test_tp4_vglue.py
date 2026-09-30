"""The TP4 verify-glue levers (tp4/vglue): flags, and per lever a CPU equality test that fails if the lever's path
differs, byte for byte, from the path it replaces. No hardware; the DMA kernels are proven by emulating their
published schedules over byte pages.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue` from scripts/ci.
"""

import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_commit_dma_tp
import tp4_vglue
from test_gdn_tp_twins import commit_layers, fake_ttnn, kernels_of, pair, four

HERE = Path(__file__).parent
REPO = HERE.parent.parent


def env(**values):
    return patch.dict(os.environ, values)


class FlagTests(unittest.TestCase):
    def test_flags_are_strict_zero_one(self):
        with four(), env():
            for name in tp4_vglue.LEVERS:
                self.assertFalse(tp4_vglue.enabled(name))
        with four(), env(QWEN_FAST_TP4_COMMIT_LANES='0'):
            self.assertFalse(tp4_vglue.enabled(tp4_vglue.COMMIT_LANES))
        with four(), env(QWEN_FAST_TP4_COMMIT_LANES='1'):
            self.assertTrue(tp4_vglue.enabled(tp4_vglue.COMMIT_LANES))
        for bad in ('true', '2', '', ' 1'):
            with four(), env(QWEN_FAST_TP4_ATTN_FOLD=bad):
                with self.assertRaises(ValueError):
                    tp4_vglue.enabled(tp4_vglue.ATTN_FOLD)

    def test_any_flag_at_two_cards_raises_and_unset_does_not(self):
        with pair():
            for name in tp4_vglue.LEVERS:
                self.assertFalse(tp4_vglue.enabled(name))
            self.assertFalse(tp4_vglue.audit_enabled())
        for name in tp4_vglue.ALL_FLAGS:
            with pair(), env(**{name: '1'}):
                with self.assertRaises(ValueError):
                    tp4_vglue.enabled(tp4_vglue.COMMIT_LANES)
        with env(QWEN_FAST_TP='2', QWEN_FAST_TP4_GDN_GLUE='1'):
            with self.assertRaises(ValueError):
                tp4_vglue.engaged_levers()

    def test_block_conv_needs_the_glue_dma(self):
        with four(), env(QWEN_FAST_TP4_GDN_BLOCK_CONV='1'):
            with self.assertRaises(ValueError):
                tp4_vglue.enabled(tp4_vglue.GDN_BLOCK_CONV)
        with four(), env(QWEN_FAST_TP4_GDN_BLOCK_CONV='1', QWEN_FAST_TP4_GDN_GLUE='1'):
            self.assertEqual(tp4_vglue.engaged_levers(), (tp4_vglue.GDN_GLUE, tp4_vglue.GDN_BLOCK_CONV))

    def test_audit_needs_a_lever(self):
        with four(), env(QWEN_FAST_TP4_VGLUE_AUDIT='1'):
            self.assertFalse(tp4_vglue.audit_enabled())
        with four(), env(QWEN_FAST_TP4_VGLUE_AUDIT='1', QWEN_FAST_TP4_SHARD_VALUES='1'):
            self.assertTrue(tp4_vglue.audit_enabled())

    def test_marker_format(self):
        self.assertEqual(tp4_vglue.marker('packed_verify', gdn_glue=48, attn_fold=16),
                         '[PINDIAG] tp4 vglue engaged site=packed_verify gdn_glue=48 attn_fold=16')

    def test_unknown_lever_is_refused(self):
        with four():
            with self.assertRaises(ValueError):
                tp4_vglue.enabled('QWEN_FAST_TP4_NOPE')


# ---------------------------------------------------------------------------------------------------------------------
# C1a: the pipelined commit against the served commit

PAGE = 2048
STATE_PAGES, CONV_PAGES = 192, 80
CONV_TASKS = 4 * CONV_PAGES


class Noc:
    """Pages of one buffer as bytearrays: the kernel emulations move real bytes so a wrong page, offset or lane shows."""

    def __init__(self, count, seed=None, blank=False):
        rng = random.Random(seed)
        self.pages = [bytearray(PAGE) if blank else bytearray(rng.randbytes(PAGE)) for _ in range(count)]


def fixture(rows, seed):
    """One layer's twenty buffers by the kernel's runtime-arg order; the destinations start as recognisable garbage."""
    rng = random.Random(seed)

    def source(count):
        return Noc(count, rng.getrandbits(32))

    return dict(
        entry_rec=source(STATE_PAGES), entry_conv=[source(CONV_PAGES) for _ in range(4)],
        history_rec=source(rows * STATE_PAGES), history_conv=[source(CONV_PAGES) for _ in range(4)],
        native_rec=source(STATE_PAGES), native_conv=[source(CONV_PAGES) for _ in range(4)],
        checkpoint_rec=source(STATE_PAGES), checkpoint_conv=[source(CONV_PAGES) for _ in range(4)])


def clone(layer):
    out = {}
    for key, value in layer.items():
        if isinstance(value, list):
            out[key] = [SimpleNamespace(pages=[bytearray(page) for page in item.pages]) for item in value]
        else:
            out[key] = SimpleNamespace(pages=[bytearray(page) for page in value.pages])
    return out


def conv_offset(prefix):
    token = 0 if prefix == 0 else prefix - 1
    return ((token // 16) * 512 + (token % 16) * 16) * 2


def served_kernel(layer, prefix, worker, scratch):
    """gdn_commit_dma_tp.cpp: one tile, a barrier, two writes, a barrier; zero the 2 KB output per conv task."""
    staged, output = 0, 2048
    for page in range(worker, STATE_PAGES, 2):
        source = layer['entry_rec'] if prefix == 0 else layer['history_rec']
        index = page if prefix == 0 else (prefix - 1) * STATE_PAGES + page
        scratch[staged:staged + PAGE] = source.pages[index]
        layer['native_rec'].pages[page][:] = scratch[staged:staged + PAGE]
        layer['checkpoint_rec'].pages[page][:] = scratch[staged:staged + PAGE]
    offset = conv_offset(prefix)
    for task in range(worker, CONV_TASKS, 2):
        slot, page = divmod(task, CONV_PAGES)
        source = layer['entry_conv'][slot] if prefix == 0 else layer['history_conv'][slot]
        scratch[staged:staged + PAGE] = source.pages[page]
        scratch[output:output + PAGE] = bytes(PAGE)
        for face in range(2):
            scratch[output + face * 512:output + face * 512 + 32] = scratch[staged + offset + face * 512:staged + offset + face * 512 + 32]
        layer['checkpoint_conv'][slot].pages[page][:] = scratch[output:output + PAGE]
        layer['native_conv'][slot].pages[page][0:32] = scratch[output:output + 32]
        layer['native_conv'][slot].pages[page][512:544] = scratch[output + 512:output + 544]


def lanes_kernel(layer, prefix, worker, scratch):
    """gdn_commit_lanes_tp.cpp: eight lanes per barrier, whole-tile reads, output scratch zeroed once per lane."""
    for batch in range(0, STATE_PAGES, 16):
        pages = [batch + lane * 2 + worker for lane in range(8)]
        for lane, page in enumerate(pages):
            source = layer['entry_rec'] if prefix == 0 else layer['history_rec']
            index = page if prefix == 0 else (prefix - 1) * STATE_PAGES + page
            scratch[lane * 4096:lane * 4096 + PAGE] = source.pages[index]
        for lane, page in enumerate(pages):
            layer['native_rec'].pages[page][:] = scratch[lane * 4096:lane * 4096 + PAGE]
            layer['checkpoint_rec'].pages[page][:] = scratch[lane * 4096:lane * 4096 + PAGE]
    offset = conv_offset(prefix)
    for lane in range(8):
        scratch[lane * 4096 + 2048:lane * 4096 + 4096] = bytes(2048)
    for batch in range(0, CONV_TASKS, 16):
        tasks = [batch + lane * 2 + worker for lane in range(8)]
        for lane, task in enumerate(tasks):
            slot, page = divmod(task, CONV_PAGES)
            source = layer['entry_conv'][slot] if prefix == 0 else layer['history_conv'][slot]
            scratch[lane * 4096:lane * 4096 + PAGE] = source.pages[page]
        for lane, task in enumerate(tasks):
            slot, page = divmod(task, CONV_PAGES)
            staged, output = lane * 4096, lane * 4096 + 2048
            for face in range(2):
                scratch[output + face * 512:output + face * 512 + 32] = scratch[staged + offset + face * 512:staged + offset + face * 512 + 32]
            layer['checkpoint_conv'][slot].pages[page][:] = scratch[output:output + PAGE]
            layer['native_conv'][slot].pages[page][0:32] = scratch[output:output + 32]
            layer['native_conv'][slot].pages[page][512:544] = scratch[output + 512:output + 544]


def snapshot(layer):
    return [bytes(page) for key in ('native_rec', 'checkpoint_rec') for page in layer[key].pages] + \
        [bytes(page) for key in ('native_conv', 'checkpoint_conv') for item in layer[key] for page in item.pages]


class CommitLanesTests(unittest.TestCase):
    def test_the_lanes_schedule_writes_the_served_bytes_for_every_prefix(self):
        for prefix in range(0, 17):
            served, lanes = fixture(16, prefix), None
            lanes = clone(served)
            garbage = random.Random(prefix)
            for worker in (0, 1):
                served_kernel(served, prefix, worker, bytearray(garbage.randbytes(4096)))
                lanes_kernel(lanes, prefix, worker, bytearray(garbage.randbytes(32768)))
            self.assertEqual(snapshot(served), snapshot(lanes), 'prefix %d' % prefix)

    def test_the_two_workers_partition_every_page_and_task(self):
        for count in (STATE_PAGES, CONV_TASKS):
            workers = [[batch + lane * 2 + worker for batch in range(0, count, 16) for lane in range(8)]
                       for worker in range(2)]
            self.assertFalse(set(workers[0]) & set(workers[1]))
            self.assertEqual(sorted(workers[0] + workers[1]), list(range(count)))

    def test_lane_scratch_fits_the_circular_buffer(self):
        self.assertEqual(7 * 4096 + 2048 + 2048, gdn_commit_dma_tp.LANES_CB_BYTES)

    def test_kernel_text_has_no_pair_literals_and_takes_the_defines(self):
        text = (HERE / 'gdn_commit_lanes_tp.cpp').read_text()
        for literal in ('384', '640', '160'):
            self.assertIsNone(re.search(r'\b%s\b' % literal, text), literal)
        for name in ('QWEN_STATE_PAGES', 'QWEN_CONV_TASKS', 'QWEN_CONV_PAGES'):
            self.assertIn('#ifndef %s' % name, text)
        # one zeroing pass per lane, none inside the task loop
        self.assertEqual(text.count('words[word] = 0'), 1)
        # whole-tile reads only on the conv path (a 32-byte DRAM read at a token offset is unaligned)
        self.assertNotIn('noc_async_read(', text)

    def launch(self, chips=4, layers=3, rows=8, prefix=5):
        captured = []
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        with patch.dict(sys.modules, {'ttnn': fake_ttnn(captured, chips)}):
            gdn_commit_dma_tp.prepare(mesh, commit_layers(layers, rows, chips), prefix)()
        return captured[0][1]

    def test_flag_off_is_the_served_program_and_on_changes_only_kernel_and_buffer(self):
        with four(), env():
            served = self.launch()
        with four(), env(QWEN_FAST_TP4_COMMIT_LANES='1'):
            lanes = self.launch()
        self.assertEqual(sorted(served), sorted(lanes))
        for key in served:
            left, right = served[key].kernels[0], lanes[key].kernels[0]
            self.assertEqual(left.kernel_source, str(HERE / 'gdn_commit_dma_tp.cpp'))
            self.assertEqual(right.kernel_source, str(HERE / 'gdn_commit_lanes_tp.cpp'))
            self.assertEqual(left.defines, right.defines)
            self.assertEqual(left.compile_time_args, right.compile_time_args)
            self.assertEqual(dict(left.runtime_args), dict(right.runtime_args))
            self.assertEqual(served[key].cbs[0].total_size, 4096)
            self.assertEqual(lanes[key].cbs[0].total_size, 32768)
            self.assertEqual(served[key].cbs[0].core_ranges, lanes[key].cbs[0].core_ranges)

    def test_the_flag_at_the_pair_raises(self):
        with pair(), env(QWEN_FAST_TP4_COMMIT_LANES='1'):
            with self.assertRaises(ValueError):
                self.launch(chips=2)


if __name__ == '__main__':
    unittest.main()
