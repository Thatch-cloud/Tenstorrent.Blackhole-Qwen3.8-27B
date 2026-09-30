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

import torch

import gdn_commit_dma_tp
import tp4_vglue
import verify_trace_t1 as t1
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


# ---------------------------------------------------------------------------------------------------------------------
# V4a: the gathered shard maximum against ttnn.max

class T:
    """A torch-backed stand-in for a device tensor: `.t` is the data, the rest is what sample_shards reads."""

    def __init__(self, t, name='t'):
        self.t, self.name = t, name
        self.shape, self.dtype, self.layout = tuple(t.shape), 'bf16', 'tile'


def torch_operations(calls, gather_error=None):
    ops = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm', DRAM_MEMORY_CONFIG='dram')
    ops.to_layout = lambda value, layout, memory_config=None: T(value.t, 'row_major')
    ops.argmax = lambda value, dim, keepdim, memory_config=None: T(torch.argmax(value.t, dim=dim, keepdim=keepdim).to(torch.int64), 'ids')

    def maximum(value, dim, keepdim, memory_config=None):
        calls.append('max')
        return T(torch.max(value.t.to(torch.float32), dim=dim, keepdim=keepdim).values.to(torch.bfloat16), 'values')

    def gather(value, dim, index, memory_config=None):
        calls.append('gather')
        if gather_error is not None:
            raise gather_error
        return T(torch.gather(value.t, dim, index.t), 'gathered')

    ops.max, ops.gather, ops.deallocate = maximum, gather, lambda value: calls.append('free ' + value.name)
    return ops


def shard_logits(rows, width, seed):
    """bf16 shards seeded with ties, signed zeros, all-zero rows and a flat row, the cases a value swap could show."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, rows, width, generator=generator).to(torch.bfloat16)
    logits[0, 0, 0, :] = 0.0                                       # all +0: first occurrence is column 0
    logits[0, 0, 1, :] = -0.0                                      # all -0
    logits[0, 0, 2, 3], logits[0, 0, 2, 5] = 9.0, 9.0             # a tie inside the shard
    logits[0, 0, 3, :] = -1.0
    logits[0, 0, 3, 7], logits[0, 0, 3, 8] = -0.0, 0.0            # -0 first, +0 later: equal maxima
    logits[0, 0, 4, :] = -3.0e38
    logits[0, 0, 5, :] = 1e-40                                     # denormal-scale values
    return logits


class ShardValuesTests(unittest.TestCase):
    def sample(self, logits, **flags):
        calls = []
        with four(), env(**flags):
            ids, values = t1.sample_shards(torch_operations(calls), T(logits, 'logits'), logits.shape[2])
        return ids, values, calls

    def test_the_gathered_values_are_the_max_bits_on_every_row(self):
        for seed in range(4):
            logits = shard_logits(16, 62080, seed)
            ids, gathered, on_calls = self.sample(logits, QWEN_FAST_TP4_SHARD_VALUES='1')
            ids_off, maximum, off_calls = self.sample(logits)
            self.assertTrue(torch.equal(ids.t, ids_off.t))
            self.assertEqual(gathered.t.shape, maximum.t.shape)
            self.assertEqual(t1.compare_values(gathered.t, maximum.t), [])
            # not just equal as numbers: identical bits except a zero of the other sign
            same_bits = (gathered.t.view(torch.int16) == maximum.t.view(torch.int16)).reshape(-1)
            zero = (gathered.t == 0).reshape(-1)
            self.assertTrue(bool((same_bits | zero).all()))
            self.assertIn('gather', on_calls)
            self.assertNotIn('max', on_calls)
            self.assertIn('max', off_calls)
            self.assertNotIn('gather', off_calls)

    def test_the_four_shard_combine_is_unchanged_by_the_value_source(self):
        rows = 16
        parts = [shard_logits(rows, 62080, seed) for seed in range(4)]
        combined = {}
        for name, flags in (('max', {}), ('gather', {'QWEN_FAST_TP4_SHARD_VALUES': '1'})):
            shard_ids, shard_values_ = [], []
            for part in parts:
                ids, values, _ = self.sample(part, **flags)
                shard_ids.append(ids.t.reshape(-1)), shard_values_.append(values.t.reshape(-1))
            with four():
                combined[name] = t1.combine_shards(shard_ids, shard_values_).tolist()
        self.assertEqual(combined['max'], combined['gather'])
        whole = torch.cat([part.reshape(rows, -1) for part in parts], dim=1).to(torch.float32)
        self.assertEqual(combined['gather'], torch.argmax(whole, dim=1).tolist())

    def test_flag_off_makes_exactly_the_served_calls(self):
        logits = shard_logits(8, 62080, 0)
        _, _, calls = self.sample(logits)
        self.assertEqual(calls, ['max', 'free row_major'])

    def test_the_row_major_shard_lives_until_the_gather_has_read_it(self):
        _, _, calls = self.sample(shard_logits(8, 62080, 1), QWEN_FAST_TP4_SHARD_VALUES='1')
        self.assertLess(calls.index('gather'), calls.index('free row_major'))

    def test_a_refusing_gather_falls_back_to_max_and_says_so(self):
        logits = shard_logits(8, 62080, 2)
        calls, lines = [], []
        with four(), env(QWEN_FAST_TP4_SHARD_VALUES='1'), patch.object(t1, 'log_line', side_effect=lines.append):
            ids, values = t1.sample_shards(torch_operations(calls, gather_error=RuntimeError('row-major gather')),
                                           T(logits, 'logits'), 8)
        self.assertEqual(calls[:2], ['gather', 'max'])
        self.assertEqual(values.name, 'values')
        self.assertTrue(any(line.startswith(tp4_vglue.FALLBACK) for line in lines))

    def test_the_audit_keeps_max_beside_the_gather_and_compares(self):
        logits = shard_logits(8, 62080, 3)
        t1.VALUE_REFERENCES.clear()
        ids, values, calls = self.sample(logits, QWEN_FAST_TP4_SHARD_VALUES='1', QWEN_FAST_TP4_VGLUE_AUDIT='1')
        self.assertEqual(calls.count('max'), 1)
        self.assertIn(id(values), t1.VALUE_REFERENCES)
        self.assertEqual(t1.compare_values(values.t, t1.VALUE_REFERENCES[id(values)].t), [])
        t1.VALUE_REFERENCES.clear()

    def test_the_value_compare_rule(self):
        one = torch.tensor([1.0, -0.0, float('nan'), 2.0])
        self.assertEqual(t1.compare_values(one, torch.tensor([1.0, 0.0, float('nan'), 2.0])), [])
        self.assertEqual(t1.compare_values(one, torch.tensor([1.5, 0.0, 1.0, 2.0])), [0, 2])

    def test_at_the_pair_the_flag_raises_before_any_op(self):
        with pair(), env(QWEN_FAST_TP4_SHARD_VALUES='1'):
            with self.assertRaisesRegex(ValueError, 'TP4 levers'):
                t1.sample_shards(torch_operations([]), T(shard_logits(8, 124160, 0), 'logits'), 8)


# ---------------------------------------------------------------------------------------------------------------------
# the image: every runtime file of every lever ships, and its tests run in CI

class ShipmentTests(unittest.TestCase):
    def read(self, path):
        return (REPO / path).read_text(encoding='utf-8')

    def test_every_runtime_file_exists_and_is_in_all_three_copy_lists(self):
        dockerfile = self.read('docker/qwen-fast-serving.Dockerfile')
        workflow = self.read('.github/workflows/qwen-fast-serving-image.yml')
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        for name in tp4_vglue.RUNTIME_FILES:
            with self.subTest(name=name):
                self.assertTrue((HERE / name).is_file())
                self.assertIn('scripts/ci/%s ' % name, dockerfile)
                self.assertRegex(workflow, r'for name in [^\n]*\b%s\b' % re.escape(name))
                self.assertIn('scripts/ci/%s' % name, overlay)

    def test_the_modules_that_import_the_flags_are_shipped_beside_them(self):
        importing = [path.name for path in HERE.glob('*.py') if not path.name.startswith('test_')
                     and re.search(r'^\s*(import|from) tp4_vglue\b', path.read_text(encoding='utf-8'), re.M)]
        self.assertGreater(len(importing), 4)
        for name in importing:
            if name == 'tp4_vglue.py':
                continue
            self.assertTrue((HERE / name).is_file())

    def test_the_cpu_suite_names_every_vglue_test_module(self):
        workflow = self.read('.github/workflows/qwen-integration-cpu.yml')
        for module in ('test_tp4_vglue', 'test_tp4_vglue_attention', 'test_tp4_vglue_gdn', 'test_tp4_vglue_twin',
                       'test_tp4_vglue_block', 'test_tp4_vglue_window'):
            self.assertRegex(workflow, r'unittest [^\n]*\b%s\b' % module)
            self.assertTrue((HERE / (module + '.py')).is_file())

    def test_pinned_tp2_sources_are_untouched(self):
        import subprocess

        base = subprocess.run(['git', '-C', str(REPO), 'merge-base', 'HEAD', 'origin/tp4/stack'], capture_output=True)
        if base.returncode:
            self.skipTest('origin/tp4/stack is not reachable from this checkout')
        changed = subprocess.run(['git', '-C', str(REPO), 'diff', '--name-only', base.stdout.decode().strip(), 'HEAD',
                                  '--', 'scripts/ci'], capture_output=True).stdout.decode().split()
        guarded = {'scripts/ci/gdn_device_loop_state.py', 'scripts/ci/gdn_user_batch.py', 'scripts/ci/gdn_records.py',
                   'scripts/ci/gdn_multitoken.py', 'scripts/ci/gdn_multitoken_conv.py', 'scripts/ci/gdn_commit_dma.py',
                   'scripts/ci/gdn_commit_dma.cpp', 'scripts/ci/attention_fold_dma.py', 'scripts/ci/attention_fold_dma.cpp',
                   'scripts/ci/gdn_batched_conv.py', 'scripts/ci/gdn_conv_windows.py', 'scripts/ci/gdn_conv_windows.cpp'}
        self.assertEqual(sorted(guarded & set(changed)), [])


if __name__ == '__main__':
    unittest.main()
