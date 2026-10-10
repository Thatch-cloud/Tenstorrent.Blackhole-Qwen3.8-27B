"""CPU tests of the WPH-3 card-M probe: its tensor sets, its byte compare, its verdict and pick rules, and its whole run on a fake one-chip ttnn; and of its harness script.

The fake ttnn holds a mesh tensor of ONE shard (a card sees one chip) and answers the three reads the probe times: a blocking ttnn.to_torch of a shard, a blocking to_torch of the mesh tensor with a
mesh composer, and a non-blocking copy_device_to_host_tensor into a host tensor allocated by allocate_tensor_on_host followed by one synchronize_device. A second fake corrupts one path to show the
probe failing it, and one without a binding to show a path reported as an error without failing the others.

Run: python -B -m unittest discover -s optimisation/ttnn-op/batched_reads -p 'test_*.py'
"""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import batched_reads_card_m as probe  # noqa: E402
import batched_reads_tp as reads  # noqa: E402


class Shard:
    def __init__(self, value):
        self.value = value


class Mesh:
    def __init__(self, value, dtype, layout):
        self.value, self.dtype, self.layout = value, dtype, layout
        self.shape = tuple(value.shape)


class Host:
    def __init__(self, shape, dtype, layout):
        self.shape, self.dtype, self.layout, self.value = shape, dtype, layout, None


class FakeTTNN:
    """A one-chip ttnn as the probe uses it."""

    bfloat16, uint32 = 'bfloat16', 'uint32'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'ROW_MAJOR_LAYOUT', 'TILE_LAYOUT', 'dram'

    def __init__(self, corrupt=None, without=()):
        self.corrupt = corrupt
        self.opened = []
        self.closed = False
        self.singles = self.composed = self.copies = self.fences = 0
        for name in without:
            setattr(self, name, None)

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.opened.append(shape)
        return SimpleNamespace(name='mesh')

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def from_torch(self, value, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        return Mesh(value.clone(), dtype, layout)

    def deallocate(self, tensor):
        pass

    def get_device_tensors(self, tensor):
        return [Shard(tensor.value)]

    def to_torch(self, value, mesh_composer=None):
        if isinstance(value, Shard):
            self.singles += 1
            return value.value.clone()
        if isinstance(value, Host):
            return value.value.clone()
        self.composed += 1
        result = value.value.clone()
        if self.corrupt == 'compose' and result.is_floating_point():
            result = result + 1
        return result

    def ConcatMeshToTensor(self, mesh, dim=0):
        return ('composer', dim)

    def allocate_tensor_on_host(self, shape, dtype, layout, mesh):
        return Host(shape, dtype, layout)

    def copy_device_to_host_tensor(self, tensor, host, blocking=True, cq_id=None):
        assert blocking is False
        self.copies += 1
        host.value = tensor.value.clone()
        if self.corrupt == 'async' and host.value.is_floating_point():
            host.value = host.value + 1

    def synchronize_device(self, mesh):
        self.fences += 1


class ShapeTests(unittest.TestCase):
    def test_the_sets_are_the_two_read_backs_at_the_quads_rows(self):
        sets = probe.tensor_sets(2)
        self.assertEqual([item[0] for item in sets['verify']], ['ids', 'maxima'])
        self.assertEqual([item[1] for item in sets['verify']], [(1, 1, 1, 64)] * 2)
        self.assertEqual([item[0] for item in sets['collect']], ['values0', 'indices0', 'values1', 'indices1', 'features'])
        self.assertEqual(sets['collect'][-1][1], (1, 1, 64, 256))
        self.assertEqual(len(probe.tensor_sets(3)['collect']), 7)

    def test_the_four_card_shard_has_two_candidate_chunks(self):
        import draft_shared_head_tp

        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.assertEqual(len(draft_shared_head_tp.candidate_chunks()), 2)

    def test_same_bits_tells_a_sign_of_zero_and_a_dtype(self):
        plus, minus = torch.tensor([0.0], dtype=torch.bfloat16), torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertFalse(probe.same_bits(torch, plus, minus))
        self.assertTrue(probe.same_bits(torch, plus, plus.clone()))
        self.assertFalse(probe.same_bits(torch, torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64)))


class RuleTests(unittest.TestCase):
    def timing(self, served, compose=None, async_=None, enqueue=10.0):
        """The collect set's one-shard figures: the served read of the whole set, the composed and asynchronous ones, and the enqueue loop of the asynchronous path (q times the tensors)."""
        collect = {'served': {'median_us': served}, 'tensors': 5}
        if compose is not None:
            collect['compose'] = {'median_us': compose}
        if async_ is not None:
            collect['async'] = {'median_us': async_}
            collect['async_parts'] = {'enqueue': {'median_us': enqueue}, 'fence': {'median_us': 5.0}, 'convert': {'median_us': 5.0}, 'tensors': 5}
        return {'collect': collect}

    def test_the_projection_adds_the_extra_shards_enqueues_to_the_one_shard_figure(self):
        projected = probe.project(self.timing(100.0, 90.0, 40.0, enqueue=10.0)['collect'], 4)
        self.assertEqual(projected, {'served': 400.0, 'compose': 90.0 + 30.0, 'async': 40.0 + 30.0})
        without = probe.project(self.timing(100.0, 90.0)['collect'], 4)
        self.assertEqual(without, {'served': 400.0, 'compose': 90.0 + 300.0}, 'no q: an extra shard is priced as a whole blocking read')

    def test_a_path_wins_at_a_quarter_or_more_below_the_four_chip_served_read(self):
        self.assertEqual(probe.verdicts(self.timing(100.0, 90.0, 40.0)), ({'compose': 'WIN', 'async': 'WIN'}, 'async'))
        self.assertEqual(probe.verdicts(self.timing(100.0, 90.0, 100.0, enqueue=60.0)), ({'compose': 'WIN', 'async': 'WIN'}, '1'), 'the extra enqueues cost the async path its lead')
        self.assertEqual(probe.verdicts(self.timing(100.0, 420.0, 450.0, enqueue=10.0)), ({'compose': 'LOSS', 'async': 'LOSS'}, 'none'))
        self.assertEqual(probe.verdicts(self.timing(100.0, 280.0, 330.0, enqueue=10.0)), ({'compose': 'NEUTRAL', 'async': 'NEUTRAL'}, 'none'))
        self.assertEqual(probe.verdicts(self.timing(100.0, 90.0)), ({'compose': 'NEUTRAL', 'async': 'NOT-RUN'}, 'none'))
        self.assertEqual(probe.verdicts(self.timing(100.0, 60.0, 60.0)), ({'compose': 'WIN', 'async': 'WIN'}, '1'), 'a tie goes to compose')
        self.assertEqual(probe.verdicts({})[1], 'none')

    def test_the_verdict_fails_a_path_that_differs_and_does_not_fail_one_that_raised(self):
        exact = {'verify': {'compose': {'exact': True}, 'async': {'exact': True}}}
        self.assertEqual(probe.verdict(exact), ('PASS', 0))
        self.assertEqual(probe.verdict({'verify': {'compose': {'exact': True}, 'async': {'exact': None, 'error': 'x'}}}), ('PASS', 0))
        self.assertEqual(probe.verdict({'verify': {'compose': {'exact': False, 'differing': ['ids']}, 'async': {'exact': True}}}), ('FAIL', 1))
        self.assertEqual(probe.verdict({'verify': {'compose': {'exact': None, 'error': 'x'}, 'async': {'exact': None, 'error': 'y'}}}), ('NOT-RUN', 4))


class RunTests(unittest.TestCase):
    def run_probe(self, ttnn, *extra, tp='4'):
        out = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        out.close()
        self.addCleanup(os.unlink, out.name)
        stdout = io.StringIO()
        import draft_shared_head_tp
        import tp_shapes

        with patch.dict(os.environ, {'QWEN_FAST_TP': tp}), contextlib.redirect_stdout(stdout):
            status = probe.main(['--out', out.name, '--rounds', '5', '--warmup', '1', *extra], torch=torch, ttnn=ttnn, reads=reads, tp_shapes=tp_shapes,
                                draft_shared_head_tp=draft_shared_head_tp)
        report = json.loads(Path(out.name).read_text())
        return status, report, stdout.getvalue()

    def test_a_clean_run_passes_every_compare_and_times_all_three_paths(self):
        ttnn = FakeTTNN()
        status, report, text = self.run_probe(ttnn)
        self.assertEqual(status, 0)
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(report['kind'], probe.KIND)
        for name in ('verify', 'collect'):
            self.assertTrue(report['compare'][name]['compose']['exact'] and report['compare'][name]['async']['exact'], name)
            self.assertTrue(report['compare'][name]['served']['upload_unaltered'])
            self.assertEqual(sorted(path for path in report['timing'][name] if path in probe.PATHS), ['async', 'compose', 'served'])
            self.assertEqual(sorted(report['timing'][name]['async_parts']), ['convert', 'enqueue', 'fence', 'tensors'])
        self.assertEqual(report['chunks'], 2)
        self.assertTrue(all(report['bindings'].values()))
        self.assertIn('BATCHED_READS verdict=PASS sets=2', text)
        self.assertIn('BATCHED_READS timing_verdict compose=', text)
        self.assertRegex(text, r'BATCHED_READS pick reads=(1|async|none)')
        self.assertEqual(ttnn.opened, [(1, 1)])
        self.assertTrue(ttnn.closed)
        self.assertGreater(ttnn.copies, 0)
        self.assertGreater(ttnn.fences, 0)
        self.assertTrue(text.rstrip().splitlines()[-1].startswith('{"'), 'the last line is the JSON object')

    def test_a_path_that_differs_fails_the_run_and_is_not_timed(self):
        for path in ('compose', 'async'):
            with self.subTest(path=path):
                status, report, text = self.run_probe(FakeTTNN(corrupt=path))
                self.assertEqual((status, report['verdict']), (1, 'FAIL'))
                self.assertFalse(report['compare']['verify'][path]['exact'])
                self.assertEqual(report['compare']['verify'][path]['differing'], ['maxima'])
                self.assertNotIn(path, report['timing']['verify'] if 'timing' in report else {})

    def test_a_missing_binding_is_that_paths_error_alone(self):
        status, report, text = self.run_probe(FakeTTNN(without=('allocate_tensor_on_host',)))
        self.assertEqual((status, report['verdict']), (0, 'PASS'))
        self.assertIn('error', report['compare']['verify']['async'])
        self.assertTrue(report['compare']['verify']['compose']['exact'])
        self.assertEqual(report['timing_verdict']['async'], 'NOT-RUN')
        self.assertFalse(report['bindings']['allocate_tensor_on_host'])

    def test_no_batched_path_at_all_is_not_run(self):
        status, report, text = self.run_probe(FakeTTNN(without=('allocate_tensor_on_host', 'ConcatMeshToTensor')))
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))

    def test_the_four_card_geometry_is_required_and_a_dead_device_is_not_run(self):
        status, report, text = self.run_probe(FakeTTNN(), tp='2')
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))
        self.assertIn('QWEN_FAST_TP=4 required', report['error'])
        broken = FakeTTNN()
        broken.open_mesh_device = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('no device'))
        status, report, text = self.run_probe(broken)
        self.assertEqual((status, report['verdict']), (4, 'NOT-RUN'))

    def test_the_arguments_are_checked(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(probe.main(['--out', '/dev/null', '--sets', 'bogus'], torch=torch, ttnn=FakeTTNN(), reads=reads, tp_shapes=None,
                                        draft_shared_head_tp=None), 2)


class HarnessTests(unittest.TestCase):
    SCRIPT = HERE / 'run_card_m.sh'

    def test_it_parses_embeds_the_board_block_and_selects_before_any_use(self):
        text = self.SCRIPT.read_text(encoding='utf-8')
        self.assertTrue(text.startswith('#!/usr/bin/env bash\n'))
        self.assertTrue(subprocess.run(['bash', '-n', str(self.SCRIPT)]).returncode == 0)
        self.assertTrue(text.count('qual_card_select') >= 2)
        self.assertNotIn('\r', text)
        self.assertLess(text.index('qual_refuse_holders\n'), text.index('qual_card_recheck   #'))
        self.assertLess(text.index('qual_card_recheck   #'), text.index('timeout -k 30'))
        self.assertIn('--network none', text)
        self.assertIn('--entrypoint env', text)
        self.assertIn('-e QWEN_FAST_TP=4 -e PYTHONPATH=/bench/ci', text)
        self.assertEqual(text.count('--device "$node"'), 1)

    def test_it_refuses_an_unset_card_and_card_b_before_anything_else(self):
        for environment in ({}, {'QUAL_CARD': 'blackhole-F36F768B9A5CAFA0'}):
            with self.subTest(environment=environment):
                env = {name: value for name, value in os.environ.items() if name not in ('QUAL_CARD', 'ALLOW_SERVING_CARD')}
                env.update(environment)
                result = subprocess.run(['bash', str(self.SCRIPT)], capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 1)
                self.assertIn('refusing', result.stderr)

    def test_the_canonical_board_block_is_embedded_byte_for_byte(self):
        sys.path.insert(0, str(CI))
        import test_qual_card

        text = self.SCRIPT.read_text(encoding='utf-8')
        start, end = test_qual_card.block_span(text)
        self.assertEqual(text[start:end], test_qual_card.canonical())
        self.assertTrue(text[end:].startswith('qual_card_select\n'))

    def test_no_host_address_registry_or_digest_is_written_outside_the_board_block(self):
        import re

        text = self.SCRIPT.read_text(encoding='utf-8')
        outside = text[:text.index('# >>> qual_card.sh')] + text[text.index('# <<< qual_card.sh'):]
        self.assertIsNone(re.search(r'\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|zot\.|thatch\.local|/home/', outside))


if __name__ == '__main__':
    unittest.main()
