"""llk_zone_override and its call site: off by default, the serving path unchanged when off, refused without the
device profiler, and - when asked - the K5-A build instrumented after qualification with its JIT cache key moved
off the served binaries."""
import ast
import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llk_zone_override as override  # noqa: E402
import llk_zones as zones  # noqa: E402

PROFILED = {'QWEN_LLK_ZONES': 'stages', 'TT_METAL_DEVICE_PROFILER': '1', 'TT_METAL_PROFILER_SUM': '1'}


def fake_module(build):
    """A stand-in for gdn_seq_block: its Build class and a served_kernels that returns one cached build."""
    import gdn_seq_block
    calls = []

    def served_kernels(root=None, environ=None):
        calls.append((root, environ))
        return build
    return types.SimpleNamespace(Build=gdn_seq_block.Build, served_kernels=served_kernels, calls=calls)


def k5a_build():
    import test_gdn_seq_block
    return test_gdn_seq_block.build()


class Log(list):
    """pindiag's shape: a brace template and its values."""

    def __call__(self, template, *values):
        self.append(template.format(*values))


class RequestedTests(unittest.TestCase):
    def test_levels(self):
        self.assertIsNone(override.requested({}))
        self.assertIsNone(override.requested({'QWEN_LLK_ZONES': ''}))
        self.assertEqual(override.requested({'QWEN_LLK_ZONES': 'tag'}), 'tag')
        for bad in ('1', 'on', 'STAGES', 'all'):
            with self.assertRaisesRegex(ValueError, 'QWEN_LLK_ZONES must be one of tag, stages'):
                override.requested({'QWEN_LLK_ZONES': bad})


class InstallTests(unittest.TestCase):
    def test_unset_changes_nothing(self):
        module = fake_module(object())
        before = module.served_kernels
        self.assertIsNone(override.install(log=Log(), environ={}, module=module))
        self.assertIs(module.served_kernels, before)
        self.assertFalse(hasattr(module, '_llk_served_kernels'))

    def test_refused_without_the_profiler(self):
        module = fake_module(object())
        with self.assertRaisesRegex(ValueError, 'without TT_METAL_DEVICE_PROFILER=1'):
            override.install(log=Log(), environ={'QWEN_LLK_ZONES': 'stages'}, module=module)
        self.assertFalse(hasattr(module, '_llk_served_kernels'))

    def test_the_build_is_instrumented_after_qualification(self):
        build = k5a_build()
        module = fake_module(build)
        log = Log()
        result = override.install(log=log, environ=PROFILED, module=module)
        self.assertEqual(result, dict(env='QWEN_LLK_ZONES', level='stages', wrapped=True, sums=True))
        instrumented = module.served_kernels('/opt/tt-metal', {'X': '1'})
        self.assertEqual(module.calls, [('/opt/tt-metal', {'X': '1'})])
        self.assertIs(instrumented.served, build)
        self.assertEqual((instrumented.level, instrumented.variant, instrumented.diag, instrumented.qualified),
                         (build.level, build.variant, build.diag, build.qualified))
        for part in ('compute', 'reader', 'writer'):
            self.assertEqual(zones.remove(instrumented[part]), build[part])
            # SRC_TAG keys the JIT cache on content: the copy never reuses (or overwrites) a served binary.
            self.assertNotEqual(instrumented.tag(part), build.tag(part))
        self.assertIn('DeviceZoneScopedN("QWEN_LLK_K5A_T45")', instrumented['compute'])
        self.assertIs(module.served_kernels(), instrumented)   # one instrumentation per served build
        records = [json.loads(line.split('[LLK] record ', 1)[1]) for line in log if '[LLK] record ' in line]
        self.assertEqual(sorted(record['key'] for record in records), ['K5A', 'K5A_RD', 'K5A_WR'])
        self.assertTrue(any('[LLK] zones installed on gdn_seq_block (level stages, sums on)' in line for line in log))
        # The record lines carry braces: they reached pindiag as a value, never as its template.
        self.assertTrue(all('{' in line for line in log if '[LLK] record ' in line))

    def test_install_is_idempotent_and_tag_has_no_sums(self):
        build = k5a_build()
        module = fake_module(build)
        environ = dict(PROFILED, QWEN_LLK_ZONES='tag', TT_METAL_PROFILER_SUM='0')
        override.install(log=Log(), environ=environ, module=module)
        wrapped = module.served_kernels
        override.install(log=Log(), environ=environ, module=module)
        self.assertIs(module.served_kernels, wrapped)
        instrumented = module.served_kernels()
        self.assertNotIn('DeviceZoneScopedSum', instrumented['compute'])
        self.assertEqual(instrumented.llk_level, 'tag')


class CallSiteTests(unittest.TestCase):
    """serving_runtime: the override is reached only under QWEN_LLK_ZONES, after the profiler admission and before
    combined_runtime() builds the K5-A kernels; unset, not even the import runs."""

    def setUp(self):
        with open(os.path.join(HERE, 'serving_runtime.py'), encoding='utf-8') as handle:
            self.source = handle.read()

    def test_guarded_lazy_import(self):
        tree = ast.parse(self.source)
        module_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(getattr(node, 'module', None) == 'llk_zone_override' or
                             any(alias.name == 'llk_zone_override' for alias in node.names) for node in module_level))
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and "os.environ.get('QWEN_LLK_ZONES')" in ast.get_source_segment(self.source, node.test)]
        self.assertEqual(len(guarded), 1)
        body = ast.get_source_segment(self.source, guarded[0])
        self.assertIn('from llk_zone_override import install as install_llk_zones', body)
        self.assertIn('install_llk_zones(log=pindiag)', body)
        admitted = self.source.index('admit_profiled_block_stream(log=pindiag)')
        installed = self.source.index('install_llk_zones(log=pindiag)')
        combined = self.source.index('combined_runtime(', installed)
        self.assertLess(admitted, installed)
        self.assertLess(installed, combined)

    def test_no_served_profile_carries_the_flag_or_the_profiler(self):
        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            profiles = json.load(handle)['profiles']
        for name, profile in profiles.items():
            for key in profile.get('env') or {}:
                self.assertFalse(key.startswith('QWEN_LLK_') or 'PROFILER' in key or key.startswith('TT_METAL_PROFILE'),
                                 '%s carries %s' % (name, key))

    def test_the_image_overlay_carries_the_modules(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-overlay.txt'), encoding='utf-8') as handle:
            listed = [line.split()[0] for line in handle if line.strip() and not line.startswith('#')]
        for name in ('llk_zone_override.py', 'llk_kernels.py', 'llk_zones.py'):
            self.assertIn('scripts/ci/' + name, listed)


class SyntaxTests(unittest.TestCase):
    def test_module_parses_as_python_37(self):
        with open(os.path.join(HERE, 'llk_zone_override.py'), encoding='utf-8') as handle:
            ast.parse(handle.read(), 'llk_zone_override.py', feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
