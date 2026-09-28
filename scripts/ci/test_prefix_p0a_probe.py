"""prefix_p0a_probe in a G1 image's condition, on the CPU: no vLLM, no device, no model.

Run 36236920928 (image g1-7e1296a, the first built with G1's stages) failed the P0a probe with 10
FAIL (4 6 7 8 9 10 12 13 14 16) and the oracle check outright, while both passed locally against the
plain TT plugin and a stub model. The general-prefix switch the probe sets, QWEN_PREFIX_REUSE, is
read in the image by two staged grafts:

  - the plugin's TTScheduler.__init__ (qwen_prefix_scheduler_patch.INIT_HOOK, scheduler.py:93 in the
    image) installs the plugin's own graft on the process's one shared registry at every
    construction. The probe's no-graft controls ran with a graft, its explicit installs got the
    constructor's instead (default registry, kill switch and clock), and every scheduler built
    while another was alive was refused ("one scheduler per process");
  - the model (qwen_prefix_model_patch) reports supports_prefix_caching when the switch was 1 at its
    import, and build_config set it for every variant, so the no-capability control had the
    capability and the platform kept prefix caching.

These tests build that condition from the stages' own text - a package whose TTScheduler.__init__
is patch_scheduler's output beside the real graft and registry, a model module carrying the model
stage's constants block and capability line - and hold the probe to it.
"""

import importlib
import importlib.util
import io
import itertools
import json
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

import qwen_prefix_model_patch as model_patch  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as scheduler_patch  # noqa: E402

REUSE = prefix_registry.ENV_REUSE
STATE = {}
COUNTER = itertools.count()

# The TT plugin's scheduler.py around TTScheduler.__init__ (bf77cd63), without vLLM: the base class
# only keeps what it was given. patch_scheduler accepts it because __init__ is the pinned body.
PLUGIN_SCHEDULER = (
    'import enum\n'
    '\n'
    '\n'
    'class TTSchedulingMode(enum.Enum):\n'
    '    DEFAULT = 0\n'
    '\n'
    '\n'
    'class Scheduler(object):\n'
    '    def __init__(self, *args, **kwargs):\n'
    '        self.init_kwargs = kwargs\n'
    '\n'
    '\n'
    'class TTScheduler(Scheduler):\n'
    + scheduler_patch.INIT_ANCHOR)

# qwen36_vllm around what the G1 model stage edits (patch_vllm_source): its constants anchor and the
# stock capability dict.
MODEL_SOURCE = (
    'import os\n'
    '\n'
    + model_patch.VLLM_CONSTANTS_ANCHOR +
    '\n'
    '\n'
    'class Qwen36ForCausalLM(object):\n'
    '    model_capabilities = {\n'
    '        "supports_async_decode": False,\n'
    + model_patch.VLLM_CAPABILITY_OLD +
    '        "supports_sample_on_device": True,\n'
    '    }\n')


def staged_model_source():
    """MODEL_SOURCE as the G1 model stage leaves it (the two edits patch_vllm_source makes first)."""
    source = MODEL_SOURCE.replace(model_patch.VLLM_CONSTANTS_ANCHOR,
                                  model_patch.VLLM_CONSTANTS_ANCHOR + model_patch.VLLM_CONSTANTS, 1)
    return source.replace(model_patch.VLLM_CAPABILITY_OLD, model_patch.VLLM_CAPABILITY_NEW, 1)


def import_probe():
    """prefix_p0a_probe, with this process's sys.path, sys.modules and environment left as they were:
    at import it takes its own directory off sys.path and loads the registry and the graft under
    their plain names."""
    names = ('prefix_p0a_probe', 'qwen_prefix_registry', 'qwen_prefix_scheduler_patch')
    saved_modules = dict((name, sys.modules.get(name)) for name in names)
    saved_path = list(sys.path)
    saved_stats = os.environ.get('QWEN_PREFIX_STATS_PATH')
    sys.modules.pop('prefix_p0a_probe', None)
    try:
        return importlib.import_module('prefix_p0a_probe')
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_stats is None:
            os.environ.pop('QWEN_PREFIX_STATS_PATH', None)
        else:
            os.environ['QWEN_PREFIX_STATS_PATH'] = saved_stats


def setUpModule():
    STATE['probe'] = import_probe()
    STATE['root'] = tempfile.mkdtemp(prefix='p0a-g1-')
    STATE['reuse'] = os.environ.get(REUSE)


def tearDownModule():
    for name in [name for name in sys.modules if name.startswith('p0a_g1_')]:
        del sys.modules[name]
    if STATE['root'] in sys.path:
        sys.path.remove(STATE['root'])
    shutil.rmtree(STATE['root'], True)
    if STATE['reuse'] is None:
        os.environ.pop(REUSE, None)
    else:
        os.environ[REUSE] = STATE['reuse']


def switch(on):
    """REUSE on or absent in os.environ, for a test; setUp-style (tearDownModule restores it)."""
    if on:
        os.environ[REUSE] = '1'
    else:
        os.environ.pop(REUSE, None)


class StagedPlugin(object):
    """The image's plugin package where it matters: TTScheduler.__init__ as patch_scheduler leaves it,
    and the graft and registry copied beside it as qwen_prefix_scheduler_patch.stage copies them. The
    hook's maybe_install is recorded; the real one runs underneath it."""

    def __init__(self, alter_registry=False):
        self.name = 'p0a_g1_plugin_%d' % next(COUNTER)
        package = Path(STATE['root']) / self.name
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        with io.open(str(package / 'scheduler.py'), 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(scheduler_patch.patch_scheduler(PLUGIN_SCHEDULER))
        for name in scheduler_patch.RUNTIME_FILES:
            shutil.copyfile(str(HERE / name), str(package / name))
        if alter_registry:
            with io.open(str(package / 'qwen_prefix_registry.py'), 'a', encoding='utf-8', newline='\n') as handle:
                handle.write('# not the checkout\'s bytes\n')
        if STATE['root'] not in sys.path:
            sys.path.insert(0, STATE['root'])
        importlib.invalidate_caches()  # the path finder may have listed STATE['root'] before this package
        self.scheduler = importlib.import_module(self.name + '.scheduler')
        self.graft = importlib.import_module(self.name + '.qwen_prefix_scheduler_patch')
        self.calls = []
        real = self.graft.maybe_install

        def recorded(scheduler, environ=None):
            try:
                result = real(scheduler, environ)
            except Exception as error:  # the fake scheduler lacks vLLM's internals: install refuses
                self.calls.append('%s: %s' % (type(error).__name__, str(error)[:160]))
                return None
            self.calls.append(result)
            return result

        self.graft.maybe_install = recorded


def sched_env(scheduler_cls):
    """A SchedEnv past its vLLM half: what make() reads."""
    env = object.__new__(STATE['probe'].SchedEnv)
    env.vllm_config = SimpleNamespace(role='vllm_config')
    env.kv_cache_config = SimpleNamespace(num_blocks=4100)
    env.structured = object()
    env.scheduler_block = env.hash_block = 64
    env.scheduler_cls = scheduler_cls
    env.logs = []
    return env


def import_model(source):
    """A fresh module from source, executed now: under the environment the test set."""
    path = os.path.join(STATE['root'], 'p0a_g1_model_%d.py' % next(COUNTER))
    with io.open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(source)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, module.Qwen36ForCausalLM


class ContractTests(unittest.TestCase):
    def test_the_probe_reads_the_switch_and_the_sign_the_stages_write(self):
        probe = STATE['probe']
        self.assertEqual(probe.REUSE_ENV, REUSE)
        self.assertEqual(probe.MODEL_STAGED_SIGN, model_patch.STAGED_SIGN_VLLM)
        self.assertIn('%s = os.environ.get("%s") == "1"' % (probe.MODEL_STAGED_SIGN, REUSE), model_patch.VLLM_CONSTANTS)
        self.assertIn('"supports_prefix_caching": %s,' % probe.MODEL_STAGED_SIGN, model_patch.VLLM_CAPABILITY_NEW)
        self.assertIn('environ.get("%s") == "1"' % REUSE, scheduler_patch.INIT_HOOK)
        self.assertEqual(probe.REUSE_VARIANTS, ('general-prefix', 'no-chunking'))


class SchedulerHookTests(unittest.TestCase):
    """Checks 6-14 and 16, and the oracle: the probe decides what each scheduler carries."""

    def tearDown(self):
        switch(STATE['reuse'] == '1')

    def test_make_keeps_a_g1_plugin_s_constructor_hook_off(self):
        plugin = StagedPlugin()
        switch(True)  # as build_config('general-prefix') leaves it for the scheduler checks
        env = sched_env(plugin.scheduler.TTScheduler)
        scheduler, state = env.make(install=False)
        self.assertEqual(plugin.calls, [], 'the G1 TTScheduler.__init__ hook ran the plugin\'s graft')
        self.assertIsNone(state)
        self.assertNotIn('_qwen_prefix', scheduler.__dict__)
        self.assertEqual(os.environ.get(REUSE), '1', 'the switch is back after construction')
        self.assertEqual(scheduler.init_kwargs['block_size'], 64)

    def test_make_builds_every_scheduler_the_same_way_with_the_switch_absent(self):
        plugin = StagedPlugin()
        switch(False)
        env = sched_env(plugin.scheduler.TTScheduler)
        env.make(install=False)
        self.assertEqual(plugin.calls, [])
        self.assertIsNone(os.environ.get(REUSE))

    def test_hook_true_constructs_as_the_engine_does(self):
        plugin = StagedPlugin()
        switch(False)
        env = sched_env(plugin.scheduler.TTScheduler)
        env.make(install=False, hook=True)
        self.assertEqual(len(plugin.calls), 1, 'the hook runs once, with the switch set')
        self.assertIn('prefix reuse refused', plugin.calls[0], 'the real install ran (and refused the fake)')
        self.assertIsNone(os.environ.get(REUSE), 'the switch is back after construction')

    def test_the_served_row_fails_a_g1_hook_that_installed_nothing(self):
        plugin = StagedPlugin()
        env = sched_env(plugin.scheduler.TTScheduler)
        scheduler, _ = env.make(install=False, hook=True)
        ok, detail = STATE['probe'].served_install(scheduler, env)
        self.assertFalse(ok)
        self.assertIn('carries the G1 constructor hook', detail)

    def test_the_served_row_takes_the_hook_s_install_only_on_the_checked_bytes(self):
        for altered in (False, True):
            plugin = StagedPlugin(alter_registry=altered)
            env = sched_env(plugin.scheduler.TTScheduler)
            scheduler, _ = env.make(install=False)
            # What the hook's install leaves on the instance: the plugin copy's graft, the wrappers.
            scheduler._qwen_prefix = object.__new__(plugin.graft.SchedulerGraft)
            scheduler.schedule = lambda: None
            ok, detail = STATE['probe'].served_install(scheduler, env)
            self.assertEqual(ok, not altered, detail)
            self.assertIn('installed by the plugin\'s TTScheduler.__init__ hook', detail)
            self.assertEqual('NOT the graft the checks drove' in detail, altered, detail)


class CapabilityTests(unittest.TestCase):
    """Checks 1 and 4: which variant runs with the switch, and whose capability it sees."""

    def tearDown(self):
        switch(STATE['reuse'] == '1')

    def child_switch(self, variant):
        """The switch in the environment run_variant starts the child with (without env=, this
        process's own). Only the switch: the environment itself stays out of the test log."""
        probe = STATE['probe']
        captured = {}

        def run(command, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(returncode=0, stdout=probe.VARIANT_TAG + json.dumps({'variant': variant}) + '\n')

        with mock.patch.object(probe.subprocess, 'run', run):
            result, _ = probe.run_variant(variant)
        self.assertEqual(result, {'variant': variant})
        environ = captured.get('env') if captured.get('env') is not None else os.environ
        return environ.get(REUSE)

    def test_the_no_capability_control_never_inherits_the_switch(self):
        switch(True)  # the parent after building general-prefix
        self.assertIsNone(self.child_switch('no-capability'))
        self.assertEqual(self.child_switch('no-chunking'), '1')
        switch(False)
        self.assertEqual(self.child_switch('no-chunking'), '1')
        self.assertIsNone(self.child_switch('no-capability'))

    def build_config_switch(self, variant):
        """os.environ's switch after build_config(variant) as it would import the model (the platform
        stand-in is not TTPlatform, so the build stops right after the switch and the argv)."""
        probe = STATE['probe']
        platforms = types.ModuleType('vllm.platforms')
        platforms.current_platform = type('NotTTPlatform', (object,), {})()
        vllm = types.ModuleType('vllm')
        vllm.__path__ = []
        vllm.platforms = platforms
        with mock.patch.dict(sys.modules, {'vllm': vllm, 'vllm.platforms': platforms}), \
                mock.patch.object(probe, 'variant_argv', lambda variant: ('profiles', 'contract', 'snapshot', [])):
            config, evidence = probe.build_config(variant)
        self.assertIsNone(config)
        self.assertIn('not TTPlatform', evidence['error'])
        return os.environ.get(REUSE)

    def test_build_config_sets_the_switch_only_for_the_variants_whose_profile_does(self):
        for parent in (True, False):
            for variant, wanted in (('general-prefix', '1'), ('no-chunking', '1'), ('no-capability', None),
                                    ('fallback-general', None)):
                switch(parent)
                self.assertEqual(self.build_config_switch(variant), wanted, (parent, variant))

    def capability(self, source, enable):
        probe = STATE['probe']
        with mock.patch.object(probe, 'tt_model_class', lambda: import_model(source)):
            return probe.simulate_capability(enable)

    def test_a_staged_model_reports_the_capability_itself_under_the_switch(self):
        switch(True)
        capability = self.capability(staged_model_source(), True)
        self.assertTrue(capability['image_capabilities']['supports_prefix_caching'])
        self.assertFalse(capability['simulated'], 'nothing to simulate: the G1 model reports it')
        self.assertTrue(capability['staged'])
        self.assertIs(capability['reuse_at_import'], True)

    def test_a_staged_model_ships_without_it_where_the_switch_is_absent(self):
        switch(False)  # the no-capability child, after build_config('no-capability')
        capability = self.capability(staged_model_source(), False)
        self.assertFalse(capability['image_capabilities']['supports_prefix_caching'])
        self.assertFalse(capability['used_capabilities']['supports_prefix_caching'])
        self.assertFalse(capability['simulated'])
        self.assertIs(capability['reuse_at_import'], False)

    def test_an_unstaged_model_is_simulated_and_says_so(self):
        switch(True)
        capability = self.capability(MODEL_SOURCE, True)
        self.assertFalse(capability['image_capabilities']['supports_prefix_caching'])
        self.assertTrue(capability['used_capabilities']['supports_prefix_caching'])
        self.assertTrue(capability['simulated'])
        self.assertFalse(capability['staged'])

    def check_4(self, capability):
        probe = STATE['probe']
        evidence = dict(final=dict(mamba_block_size=64, block_size=64, enable_prefix_caching=True),
                        capability=capability)
        config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_block_size=64,
                                                              block_size=64))
        control = dict(error_type='ValidationError',
                       error='Value error, --mamba-block-size can only be set with --enable-prefix-caching',
                       platform_after=dict(prefix=False),
                       capability=dict(image_capabilities=dict(supports_prefix_caching=False), reuse_at_import=False))
        with mock.patch.dict(sys.modules, {'vllm.config': None}):  # the validator's explicit re-call: not here
            return probe.check_4(evidence, config, control, '')

    def test_check_4_requires_a_g1_model_s_own_capability(self):
        ok, detail = self.check_4(dict(staged=True, simulated=True, reuse_at_import=False))
        self.assertFalse(ok, detail)
        self.assertIn('general-prefix would serve without the capability', detail)
        ok, detail = self.check_4(dict(staged=True, simulated=False, reuse_at_import=True))
        self.assertTrue(ok, detail)
        self.assertIn("the model's own", detail)
        ok, detail = self.check_4(dict(staged=False, simulated=True, reuse_at_import=None))
        self.assertTrue(ok, detail)
        self.assertIn('SIMULATED on the class', detail)


if __name__ == '__main__':
    unittest.main()
