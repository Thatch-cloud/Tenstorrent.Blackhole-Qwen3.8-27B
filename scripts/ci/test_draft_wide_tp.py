"""draft_wide_tp (QWEN_FAST_TP4_DRAFT_WIDE): the drafter's hidden-width RMS norms on a wide sharded grid at four cards.

CPU only. A fake `operations` records every call. What is proved here: with the flag off every call is the plain operations.rms_norm
with the arguments the drafter modules passed before the lever existed (so production is unchanged); the flag is strict and refused
at the pair; the plan covers every tile of every accepted shape exactly (no padded or missing core, shard and block agree, the
subblock divides the block); an input the wide program cannot take falls back to the plain call and says why; the wide path
shards, normalises sharded with the plan's program config and returns to the interleaved layout the caller asked for, freeing its
intermediates. What only hardware shows: that the sharded program is accepted at these shapes and the proposals it yields equal the
interleaved ones (the job compares accepted prefixes and the acceptance mean).
"""

import os
import re
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import draft_wide_tp as wide  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TP4 = {'QWEN_FAST_TP': '4', wide.FLAG: '1'}


class Tensor:
    def __init__(self, shape, dtype='bf16', layout='tile', memory='dram'):
        self.shape, self.dtype, self.layout, self.memory = tuple(shape), dtype, layout, memory


class Operations:
    """The slice of ttnn the module touches, recording calls."""
    bfloat16 = 'bf16'
    TILE_LAYOUT = 'tile'
    ROW_MAJOR_LAYOUT = 'row_major'
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.calls, self.freed = [], []
        self.TensorMemoryLayout = types.SimpleNamespace(WIDTH_SHARDED='width_sharded')
        self.BufferType = types.SimpleNamespace(L1='l1')
        self.ShardOrientation = types.SimpleNamespace(ROW_MAJOR='row_major_orientation')

    def CoreCoord(self, x, y):
        return ('core', x, y)

    def CoreRange(self, start, end):
        return ('range', start, end)

    def CoreRangeSet(self, ranges):
        return ('set', tuple(sorted(ranges)))

    def ShardSpec(self, grid, shape, orientation):
        return ('shard', grid, tuple(shape), orientation)

    def MemoryConfig(self, layout, buffer, spec):
        return ('memcfg', layout, buffer, spec)

    def LayerNormShardedMultiCoreProgramConfig(self, **options):
        return ('program', tuple(sorted(options.items())))

    def to_memory_config(self, tensor, config):
        self.calls.append(('to_memory_config', config))
        return Tensor(tensor.shape, tensor.dtype, tensor.layout, config)

    def rms_norm(self, tensor, **options):
        self.calls.append(('rms_norm', tensor, tuple(sorted(options.items()))))
        return Tensor(tensor.shape, tensor.dtype, tensor.layout, options.get('memory_config'))

    def deallocate(self, tensor):
        self.freed.append(tensor)


def hidden(rows=64, **options):
    return Tensor((1, 1, rows, 5120), **options)


def weight(**options):
    options.setdefault('layout', 'row_major')
    return Tensor((1, 1, 160, 32), **options)


class FlagTests(unittest.TestCase):
    def test_strict_zero_or_one_and_refused_at_the_pair(self):
        self.assertFalse(wide.enabled({}))
        self.assertFalse(wide.enabled({wide.FLAG: '0'}))
        self.assertTrue(wide.enabled(TP4))
        for bad in ('', 'true', '2', ' 1'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                wide.enabled({'QWEN_FAST_TP': '4', wide.FLAG: bad})
        with self.assertRaises(ValueError):
            wide.enabled({wide.FLAG: '1'})
        with self.assertRaises(ValueError):
            wide.enabled({'QWEN_FAST_TP': '2', wide.FLAG: '1'})

    def test_it_reads_the_process_environment_by_default(self):
        with mock.patch.dict(os.environ, TP4):
            self.assertTrue(wide.enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(wide.enabled())


class PlanTests(unittest.TestCase):
    def test_every_accepted_shape_is_covered_exactly(self):
        for rows in (32, 64, 96, 128):
            found = wide.plan(rows)
            with self.subTest(rows=rows):
                x, y = found['grid']
                self.assertEqual(found['cores'], x * y)
                self.assertEqual(found['cores'] * found['block_w'], 5120 // 32, 'every tile of the row is on exactly one core')
                self.assertEqual(found['shard'], (rows, found['block_w'] * 32))
                self.assertEqual(found['block_h'] * 32, rows)
                self.assertEqual(found['block_w'] % found['subblock_w'], 0)
                self.assertLessEqual(found['subblock_w'], wide.SUBBLOCK_CAP)
                self.assertLessEqual(found['cores'], 110, 'the worker grid')

    def test_the_default_grid_is_twenty_cores_of_eight_tiles(self):
        found = wide.plan(64)
        self.assertEqual((found['cores'], found['block_w'], found['subblock_w'], found['block_h']), (20, 8, 4, 2))
        self.assertEqual(found['grid'], wide.WIDE_GRID)

    def test_refused_shapes(self):
        for rows in (0, 16, 33, 160, 64.0, '64'):
            with self.assertRaises(ValueError, msg=repr(rows)):
                wide.plan(rows)
        with self.assertRaises(ValueError):
            wide.plan(64, width=5120 + 32)           # 161 tiles over 20 cores
        with self.assertRaises(ValueError):
            wide.plan(64, width=100)

    def test_other_grids_that_divide_the_row_are_planned_and_the_rest_refused(self):
        self.assertEqual(wide.plan(32, grid=(8, 5))['block_w'], 4)
        self.assertEqual(wide.plan(32, grid=(10, 4))['block_w'], 4)
        with self.assertRaises(ValueError):
            wide.plan(32, grid=(7, 3))

    def test_the_shard_fits_one_cores_l1_with_room(self):
        found = wide.plan(wide.MAX_ROWS)
        shard_bytes = found['shard'][0] * found['shard'][1] * 2
        self.assertLess(2 * shard_bytes, 200 * 1024, 'input plus output shard of the largest accepted block')


class FlagOffTests(unittest.TestCase):
    def test_off_is_the_plain_call_with_the_callers_arguments_and_nothing_else(self):
        ops = Operations()
        tensor, norm = hidden(), weight()
        with mock.patch.dict(os.environ, {}, clear=True):
            out = wide.rms_norm(ops, tensor, epsilon=1e-6, weight=norm, compute_kernel_config='kernel', memory_config='dram')
        self.assertEqual([call[0] for call in ops.calls], ['rms_norm'])
        _, seen, options = ops.calls[0]
        self.assertIs(seen, tensor)
        self.assertEqual(dict(options), dict(epsilon=1e-6, weight=norm, compute_kernel_config='kernel', memory_config='dram'))
        self.assertEqual(ops.freed, [])
        self.assertEqual(out.memory, 'dram')

    def test_off_never_inspects_the_tensor(self):
        ops = Operations()
        odd = Tensor((3, 7), dtype='fp32')
        wide.rms_norm(ops, odd, epsilon=1e-6, weight=None, compute_kernel_config=None, memory_config='dram', environ={})
        self.assertEqual([call[0] for call in ops.calls], ['rms_norm'])


class FlagOnTests(unittest.TestCase):
    def setUp(self):
        wide._LOGGED.clear()
        for key in wide.STATS:
            wide.STATS[key] = 0
        self.lines = []
        patcher = mock.patch.object(wide, 'diagnostic', self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_wide(self, tensor, norm=None, memory_config='dram', ops=None):
        ops = Operations() if ops is None else ops
        out = wide.rms_norm(ops, tensor, epsilon=1e-6, weight=weight() if norm is None else norm, compute_kernel_config='kernel',
                            memory_config=memory_config, site='mlp', environ=TP4)
        return ops, out

    def test_a_drafter_block_is_sharded_normalised_sharded_and_returned_interleaved(self):
        ops, out = self.run_wide(hidden(64))
        kinds = [call[0] for call in ops.calls]
        self.assertEqual(kinds, ['to_memory_config', 'rms_norm', 'to_memory_config'])
        shard_config = ops.calls[0][1]
        self.assertEqual(shard_config[1:3], ('width_sharded', 'l1'))
        spec = shard_config[3]
        self.assertEqual(spec[2], (64, 256), 'a (rows, block_w x 32) shard')
        self.assertEqual(spec[1], ('set', (('range', ('core', 0, 0), ('core', 4, 3)),)))
        _, sharded_input, options = ops.calls[1]
        options = dict(options)
        self.assertEqual(sharded_input.memory, shard_config)
        self.assertEqual(options['memory_config'], shard_config, 'the norm writes its output sharded')
        program = dict(options['program_config'][1])
        self.assertEqual((program['block_h'], program['block_w'], program['subblock_w'], program['inplace']), (2, 8, 4, False))
        self.assertEqual(program['compute_with_storage_grid_size'], (5, 4))
        self.assertEqual((options['epsilon'], options['compute_kernel_config']), (1e-6, 'kernel'))
        self.assertEqual(ops.calls[2][1], 'dram', 'back to the layout the caller asked for')
        self.assertEqual(out.memory, 'dram')
        self.assertEqual(out.shape, (1, 1, 64, 5120))

    def test_the_sharded_intermediates_are_freed_and_the_result_and_input_are_not(self):
        tensor = hidden(32)
        ops, out = self.run_wide(tensor)
        self.assertEqual(len(ops.freed), 2)
        self.assertNotIn(out, ops.freed)
        self.assertNotIn(tensor, ops.freed)
        self.assertEqual({tensor_.memory[0] for tensor_ in ops.freed}, {'memcfg'}, 'only the two L1 sharded tensors')

    def test_a_failing_norm_frees_the_shard_and_raises_it_is_not_swallowed(self):
        ops = Operations()
        ops.rms_norm = mock.Mock(side_effect=RuntimeError('program refused'))
        with self.assertRaises(RuntimeError):
            self.run_wide(hidden(64), ops=ops)
        self.assertEqual(len(ops.freed), 1)

    def test_engaged_is_logged_once_per_distinct_call(self):
        for _ in range(5):
            self.run_wide(hidden(64))
        engaged = [line for line in self.lines if line.startswith(wide.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('site=mlp', engaged[0])
        self.assertIn('cores=20', engaged[0])
        self.assertEqual(wide.STATS['wide'], 5)

    def test_every_ineligible_input_falls_back_to_the_plain_call_and_says_why(self):
        cases = (('width', Tensor((1, 1, 64, 4096)), None, 'shape'),
                 ('rows', Tensor((1, 1, 160, 5120)), None, 'rows'),
                 ('dtype', hidden(64, dtype='fp32'), None, 'bfloat16'),
                 ('layout', hidden(64, layout='row_major'), None, 'bfloat16'),
                 ('weight shape', hidden(64), Tensor((1, 1, 1, 5120), layout='row_major'), 'weight'),
                 ('weight layout', hidden(64), Tensor((1, 1, 160, 32), layout='tile'), 'weight'))
        for name, tensor, norm, word in cases:
            wide._LOGGED.clear()
            self.lines.clear()
            with self.subTest(case=name):
                ops, out = self.run_wide(tensor, norm=norm)
                self.assertEqual([call[0] for call in ops.calls], ['rms_norm'])
                self.assertNotIn('program_config', dict(ops.calls[0][2]))
                self.assertEqual(len([line for line in self.lines if line.startswith(wide.FALLBACK)]), 1)
                self.assertIn(word, self.lines[0])
        self.assertEqual(wide.STATS['fallback'], len(cases))

    def test_a_caller_that_wants_another_layout_gets_the_plain_call(self):
        ops, _ = self.run_wide(hidden(64), memory_config='l1_interleaved')
        self.assertEqual([call[0] for call in ops.calls], ['rms_norm'])
        self.assertIn('layout', self.lines[0])


class CallSiteTests(unittest.TestCase):
    """The drafter modules call draft_wide_tp.rms_norm at the four 5,120-wide norms and nowhere else; the per-head norms stay plain."""

    def read(self, name):
        with open(os.path.join(HERE, name), encoding='utf-8') as handle:
            return handle.read()

    def test_the_hidden_width_norms_go_through_the_module(self):
        sites = {'dflash_device.py': ("site='feature'", "site='final'"), 'draft_attention_branch.py': ("site='attention'",),
                 'draft_mlp_branch.py': ("site='mlp'",)}
        for name, markers in sites.items():
            text = self.read(name)
            self.assertIn('import draft_wide_tp', text, name)
            for marker in markers:
                self.assertEqual(text.count(marker), 1, (name, marker))
            self.assertEqual(text.count('draft_wide_tp.rms_norm('), len(markers), name)

    def test_the_head_norms_and_every_other_rms_norm_stay_the_plain_call(self):
        text = self.read('draft_attention_branch.py')
        self.assertEqual(len(re.findall(r'operations\.rms_norm\(head,', text)), 1, 'the per-head norm is not widened')
        for name in ('draft_kv_projection_tp.py', 'quad_draft_tp.py'):
            self.assertNotIn('draft_wide_tp', self.read(name), name)

    def test_the_module_is_stdlib_and_tp_shapes_only(self):
        text = self.read('draft_wide_tp.py')
        imports = sorted(set(re.findall(r'^(?:import|from) ([a-z_0-9]+)', text, flags=re.M)))
        self.assertEqual(imports, ['os', 'sys', 'tp_shapes'])


class ShipTests(unittest.TestCase):
    """The module reaches the serving image through every list the other runtime modules travel in."""

    def read(self, *parts):
        with open(os.path.join(ROOT, *parts), encoding='utf-8') as handle:
            return handle.read()

    def test_it_is_in_the_overlay_the_dockerfile_and_the_image_workflow(self):
        self.assertRegex(self.read('docker', 'qwen-c2-overlay.txt'), r'(?m)^scripts/ci/draft_wide_tp\.py$')
        self.assertRegex(self.read('docker', 'qwen-fast-serving.Dockerfile'), r'(?m)^COPY scripts/ci/draft_wide_tp\.py /experiment-scripts/ci/$')
        self.assertRegex(self.read('.github', 'workflows', 'qwen-fast-serving-image.yml'), r'for name in draft_wide_tp\.py; do')

    def test_every_importer_of_it_is_in_the_same_lists(self):
        overlay = self.read('docker', 'qwen-c2-overlay.txt')
        dockerfile = self.read('docker', 'qwen-fast-serving.Dockerfile')
        for name in ('dflash_device.py', 'draft_attention_branch.py', 'draft_mlp_branch.py'):
            self.assertIn('scripts/ci/' + name, overlay, name)
            self.assertIn('scripts/ci/' + name, dockerfile, name)

    def test_the_cpu_workflow_runs_this_module(self):
        self.assertIn('test_draft_wide_tp', self.read('.github', 'workflows', 'qwen-integration-cpu.yml'))


if __name__ == '__main__':
    unittest.main()
