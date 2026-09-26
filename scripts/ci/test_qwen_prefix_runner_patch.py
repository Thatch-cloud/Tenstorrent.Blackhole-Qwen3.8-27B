"""qwen_prefix_runner_patch: row request ids to the model, and the worker's block-size assertion.

Runs in the 3.11 CPU suite (no vLLM, no ttnn). The fixtures are the pinned plugin's own bytes
(fixtures/vllm_tt_plugin_bf77cd63, git show bf77cd63:src/vllm_tt_plugin/<file>); the P8 image's
worker.py is derived from the fixture by serving_plugin_patch.patch_worker, exactly as the P8
Dockerfile derives it (serving_plugin_patch.py is unchanged since the P8 base be9e184e).

Beyond the text: the patched submit_prefill, _prepare_model_inputs' id block and
initialize_from_config are executed against the originals, so "byte-identical when
QWEN_PREFIX_REUSE is unset" is checked on the kwargs the model would receive.
"""

import ast
import dataclasses
import hashlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_prefix_runner_patch as runner  # noqa: E402
import serving_plugin_patch  # noqa: E402

FIXTURES = HERE / 'fixtures' / 'vllm_tt_plugin_bf77cd63'


def fixture(name):
    return (FIXTURES / name).read_bytes().replace(b'\r\n', b'\n')


def p8_worker():
    return serving_plugin_patch.patch_worker(fixture('worker.py').decode('utf-8')).encode('utf-8')


def method_source(source, class_name, name):
    start, end = runner.method_span(source, class_name, name)
    return ''.join(source.splitlines(keepends=True)[start:end])


def compile_method(source, class_name, name, namespace):
    """The method as a plain function, dedented one level, executed in namespace."""
    text = method_source(source, class_name, name)
    lines = [line[4:] if line.startswith('    ') else line for line in text.splitlines(keepends=True)]
    # The plugin files use postponed annotations; so does the method compiled on its own.
    code = 'from __future__ import annotations' + chr(10) + ''.join(lines)
    exec(compile(code, '<%s.%s>' % (class_name, name), 'exec'), namespace)
    return namespace[name]


class PinTests(unittest.TestCase):
    def test_the_fixtures_are_the_pinned_blobs(self):
        self.assertEqual(hashlib.sha256(fixture('model_runner.py')).hexdigest(), runner.MODEL_RUNNER_SHA256)
        self.assertEqual(hashlib.sha256(fixture('model_input.py')).hexdigest(), runner.MODEL_INPUT_SHA256)
        self.assertEqual(runner.WORKER_SHA256[hashlib.sha256(fixture('worker.py')).hexdigest()], 'bf77cd63 blob')

    def test_the_p8_worker_is_derived_and_pinned(self):
        digest = hashlib.sha256(p8_worker()).hexdigest()
        self.assertIn('P8 image', runner.WORKER_SHA256[digest])

    def test_the_kwarg_is_the_registry_s(self):
        import qwen_prefix_registry

        self.assertEqual(runner.REQUEST_IDS_KWARG, qwen_prefix_registry.REQUEST_IDS_KWARG)
        self.assertIn('kwargs["request_ids"]', runner.SUBMIT_NEW)


class StageTests(unittest.TestCase):
    def package(self, worker=None):
        directory = Path(tempfile.mkdtemp(prefix='qwen-prefix-runner-'))
        self.addCleanup(shutil.rmtree, str(directory), True)
        for name in ('model_input.py', 'model_runner.py'):
            (directory / name).write_bytes(fixture(name))
        (directory / 'worker.py').write_bytes(worker if worker is not None else fixture('worker.py'))
        return directory

    def test_stage_on_the_stock_and_the_p8_tree(self):
        for worker, origin in ((None, 'bf77cd63 blob'), (p8_worker(), 'P8 image')):
            package = self.package(worker)
            report = runner.stage(package)
            self.assertIn(origin, report['worker.py'][1])
            for name, (before, _, after) in report.items():
                written = (package / name).read_bytes()
                self.assertEqual(hashlib.sha256(written).hexdigest(), after)
                self.assertNotEqual(before, after)
                self.assertNotIn(b'\r\n', written)
                ast.parse(written)
            with self.assertRaisesRegex(ValueError, 'is not a pinned'):
                runner.stage(package)

    def test_check_and_refusals_write_nothing(self):
        package = self.package()
        runner.stage(package, check_only=True)
        for name in ('model_input.py', 'model_runner.py', 'worker.py'):
            self.assertEqual((package / name).read_bytes(), fixture(name))
        (package / 'worker.py').write_bytes(fixture('worker.py') + b'# drift\n')
        with self.assertRaisesRegex(ValueError, 'worker.py: sha256'):
            runner.stage(package)
        self.assertEqual((package / 'model_input.py').read_bytes(), fixture('model_input.py'))

    def test_only_the_anchored_methods_change(self):
        source = fixture('model_runner.py').decode('utf-8')
        patched = runner.patch_model_runner(source)
        changed = {'_prepare_model_inputs', 'submit_prefill'}
        tree = ast.parse(source)
        runner_class = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TTModelRunner'][0]
        for node in runner_class.body:
            if isinstance(node, ast.FunctionDef) and node.name not in changed:
                self.assertEqual(method_source(source, 'TTModelRunner', node.name),
                                 method_source(patched, 'TTModelRunner', node.name), node.name)
        worker = fixture('worker.py').decode('utf-8')
        patched_worker = runner.patch_worker(worker)
        self.assertEqual(patched_worker.replace(runner.WORKER_NEW, runner.WORKER_ANCHOR), worker)
        model_input = fixture('model_input.py').decode('utf-8')
        self.assertEqual(runner.patch_model_input(model_input).replace(runner.FIELD_NEW, runner.FIELD_ANCHOR),
                         model_input)

    def test_refuses_a_patched_or_changed_source(self):
        source = fixture('model_runner.py').decode('utf-8')
        with self.assertRaisesRegex(ValueError, 'already carries'):
            runner.patch_model_runner(runner.patch_model_runner(source))
        with self.assertRaisesRegex(ValueError, 'expected one anchor'):
            runner.patch_model_runner(source.replace('        slot_remap = None\n', '        slot_remap = 0\n', 1))
        model_input = fixture('model_input.py').decode('utf-8')
        with self.assertRaisesRegex(ValueError, 'no longer ends'):
            runner.patch_model_input(model_input + '    extra: int = 0\n')


class ModelInputTests(unittest.TestCase):
    def load(self, source):
        # model_input.py imports torch. Import it OUTSIDE patch.dict(sys.modules): a module first
        # imported inside is dropped on exit, and torch's C extension cannot be imported twice.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch is not installed')
        logits = types.ModuleType('vllm.v1.sample.logits_processor')
        logits.LogitsProcessors = object
        modules = {'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'),
                   'vllm.v1.sample': types.ModuleType('vllm.v1.sample'), 'vllm.v1.sample.logits_processor': logits}
        # dataclasses resolves postponed annotations through sys.modules[cls.__module__].
        module = types.ModuleType('qwen_prefix_model_input_test')
        modules[module.__name__] = module
        with mock.patch.dict(sys.modules, modules):
            exec(compile(source, 'model_input.py', 'exec'), module.__dict__)
        return module.TTModelInput

    def test_the_field_is_last_and_defaults_to_none(self):
        before = self.load(fixture('model_input.py').decode('utf-8'))
        after = self.load(runner.patch_model_input(fixture('model_input.py').decode('utf-8')))
        names_before = [field.name for field in dataclasses.fields(before)]
        fields_after = dataclasses.fields(after)
        self.assertEqual([field.name for field in fields_after], names_before + ['prefill_request_ids'])
        self.assertIsNone(fields_after[-1].default)
        self.assertTrue(after.__dataclass_params__.frozen)


class SubmitPrefillTests(unittest.TestCase):
    NAMES = {'fields': dataclasses.fields, 'TTSamplingParams': object, 'SEED_NONE_SENTINEL': -1}

    def functions(self):
        source = fixture('model_runner.py').decode('utf-8')
        original = compile_method(source, 'TTModelRunner', 'submit_prefill', dict(self.NAMES))
        patched = compile_method(runner.patch_model_runner(source), 'TTModelRunner', 'submit_prefill',
                                 dict(self.NAMES))
        return original, patched

    def run_one(self, function, request_ids):
        seen = {}

        def prefill_forward(**kwargs):
            seen.update(kwargs)
            return 'logits'

        model_input = SimpleNamespace(
            input_tokens='tokens', block_tables='pages', input_positions=[0, 4096], prompt_lens=[100, 5000],
            block_tables_per_layer=None, multi_modal_kwargs={}, perform_device_sampling=False,
            prefill_empty_slots=[2, 0], prefill_request_ids=request_ids)
        owner = SimpleNamespace(kv_caches='kv', trace_mode='all', request_specific_rope=False,
                                tt_per_lane_max_num_seqs=4, model=SimpleNamespace(prefill_forward=prefill_forward))
        self.assertEqual(function(owner, model_input, [2]), 'logits')
        return seen

    def test_ids_reach_the_model_only_when_set(self):
        original, patched = self.functions()
        baseline = self.run_one(original, None)
        self.assertEqual(self.run_one(patched, None), baseline, 'unset: the plugin\'s own kwargs')
        with_ids = self.run_one(patched, ['req-a', 'req-b'])
        self.assertEqual(with_ids.pop('request_ids'), ['req-a', 'req-b'])
        self.assertEqual(with_ids, baseline)


class PrepareIdsTests(unittest.TestCase):
    def ids(self, is_prompt, environ):
        code = ('def prepare(input_batch, req_indices, is_prompt):\n' + runner.IDS_NEW +
                '        return prefill_request_ids\n')
        code = '\n'.join(line[4:] if line.startswith('    ') else line for line in code.split('\n'))
        namespace = {}
        exec(compile(code, '<ids>', 'exec'), namespace)
        batch = SimpleNamespace(req_ids=['r0', 'r1', 'r2'])
        with mock.patch.dict(os.environ, environ, clear=False):
            if 'QWEN_PREFIX_REUSE' not in environ:
                os.environ.pop('QWEN_PREFIX_REUSE', None)
            return namespace['prepare'](batch, [2, 0], is_prompt)

    def test_only_prompt_steps_under_reuse_carry_ids(self):
        self.assertEqual(self.ids(True, {'QWEN_PREFIX_REUSE': '1'}), ['r2', 'r0'])
        self.assertIsNone(self.ids(False, {'QWEN_PREFIX_REUSE': '1'}))
        self.assertIsNone(self.ids(True, {}))
        self.assertIsNone(self.ids(True, {'QWEN_PREFIX_REUSE': '0'}))

    def test_the_block_sits_where_the_row_ids_are_built(self):
        patched = runner.patch_model_runner(fixture('model_runner.py').decode('utf-8'))
        body = method_source(patched, 'TTModelRunner', '_prepare_model_inputs')
        self.assertIn(runner.IDS_NEW, body)
        self.assertIn('prefill_request_ids=prefill_request_ids,\n        )\n', body)
        self.assertLess(body.index('prefill_request_ids = ('), body.index('return TTModelInput('))


class WorkerTests(unittest.TestCase):
    def function(self, source):
        self.lines = []
        namespace = {'os': os, 'logger': SimpleNamespace(info=lambda message, *values: self.lines.append(
            message % values))}
        return compile_method(source, 'TTWorker', 'initialize_from_config', namespace)

    def call(self, function, block_size, group_sizes, environ, kv_caches=None):
        calls = []
        owner = SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size),
                                model_runner=SimpleNamespace(initialize_kv_cache=calls.append, kv_caches=kv_caches))
        config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size))
                                                  for size in group_sizes])
        with mock.patch.dict(os.environ, environ, clear=False):
            if 'QWEN_PREFIX_REUSE' not in environ:
                os.environ.pop('QWEN_PREFIX_REUSE', None)
            function(owner, config)
        return calls

    def test_the_assertion(self):
        for worker in (fixture('worker.py'), p8_worker()):
            function = self.function(runner.patch_worker(worker.decode('utf-8')))
            on = {'QWEN_PREFIX_REUSE': '1'}
            self.assertEqual(len(self.call(function, 64, [64], on)), 1)
            for block_size, groups in ((128, [128]), (64, [64, 32]), (64, [832]), (832, [64])):
                with self.assertRaisesRegex(RuntimeError, 'exactly 64 is required'):
                    self.call(function, block_size, groups, on)
            self.assertEqual(len(self.call(function, 128, [128], {})), 1, 'unset: the plugin\'s own behaviour')
            original = self.function(worker.decode('utf-8'))
            self.assertEqual(len(self.call(original, 128, [128], on)), 1)

    def test_the_marker_names_the_device_kv_dtype_after_allocation(self):
        function = self.function(runner.patch_worker(p8_worker().decode('utf-8')))
        on = {'QWEN_PREFIX_REUSE': '1'}
        self.call(function, 64, [64], on, kv_caches=[[SimpleNamespace(dtype='DataType.BFLOAT8_B'), None]])
        self.assertEqual(self.lines, ['[PINDIAG] prefix: worker block_size=64 kv_groups=1 '
                                      'kv_dtype=DataType.BFLOAT8_B'])
        self.lines[:] = []
        self.call(function, 64, [64], on, kv_caches={'unexpected': 'shape'})
        self.assertEqual(self.lines, ['[PINDIAG] prefix: worker block_size=64 kv_groups=1 kv_dtype=?'])
        self.lines[:] = []
        self.call(function, 64, [64], {}, kv_caches=[[SimpleNamespace(dtype='x')]])
        self.assertEqual(self.lines, [], 'unset: no marker')


if __name__ == '__main__':
    unittest.main()
