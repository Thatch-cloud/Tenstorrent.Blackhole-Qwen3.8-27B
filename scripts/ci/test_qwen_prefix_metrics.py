"""qwen_prefix_metrics: the prefix registry's export from the engine process and its collection in the
API server. Overlaid into the C2 image, so it also runs there at build time (where prometheus_client, a
vLLM dependency, is installed: the exposition test runs for real there and is skipped where it is not)."""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_scheduler_graft as graft  # noqa: E402
import qwen_prefix_metrics as metrics  # noqa: E402

try:
    import prometheus_client  # noqa: F401
    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False


def registry_with_traffic():
    registry = graft.PrefixRegistry(budget_bytes=1 << 30)
    registry.put('k1', 2048, list(range(2048)), nbytes=1000)
    registry.put('k2', 4096, list(range(4096)), nbytes=3000)
    registry.stats.update(grants=3, grant_tokens=6144, trim_loss_tokens=128, orphans=1, token_mismatches=2,
                          restore_ms=1500, capture_ms=250)
    return registry


def always(pid, ticks=None):
    return True


def never(pid, ticks=None):
    return False


class TheSharedMechanism(unittest.TestCase):
    def test_the_constants_are_the_scheduler_graft_s(self):
        """The exporter finds the registry where the graft parks it, and names every counter it keeps."""
        self.assertEqual(metrics.REGISTRY_KEY, graft.REGISTRY_KEY)
        self.assertEqual(metrics.KILL_SWITCH_PATH, graft.KILL_SWITCH_PATH)
        self.assertEqual(sorted(set(graft.STAT_NAMES) - set(metrics.COUNTERS)), [])
        snapshot = graft.PrefixRegistry(budget_bytes=1).snapshot()
        self.assertEqual(sorted(set(snapshot) - set(graft.STAT_NAMES) - set(metrics.GAUGES)), [])

    def test_the_process_registry_is_the_shared_one(self):
        self.assertIsNone(metrics.process_registry({}))
        holder = types.ModuleType(metrics.REGISTRY_KEY)
        holder.registry = object()
        self.assertIs(metrics.process_registry({metrics.REGISTRY_KEY: holder}), holder.registry)
        saved = sys.modules.pop(graft.REGISTRY_KEY, None)
        try:
            shared = graft.shared_registry()
            self.assertIs(metrics.process_registry(), shared)
        finally:
            sys.modules.pop(graft.REGISTRY_KEY, None)
            if saved is not None:
                sys.modules[graft.REGISTRY_KEY] = saved


class Values(unittest.TestCase):
    def test_read_registry_keeps_numbers_and_the_budget(self):
        values = metrics.read_registry(registry_with_traffic())
        self.assertEqual((values['entries'], values['bytes'], values['budget_bytes']), (2, 4000, 1 << 30))
        self.assertEqual((values['grants'], values['restore_ms']), (3, 1500))
        self.assertEqual(metrics.numbers({'a': True, 'b': 'x', 'c': 1.5, 'Bad-Name': 1, 3: 4}), {'a': 1, 'c': 1.5})

    def test_a_snapshot_mutated_during_iteration_is_retried(self):
        calls = []

        class Flaky(object):
            budget_bytes = 7

            def snapshot(self):
                calls.append(1)
                if len(calls) < 3:
                    raise RuntimeError('OrderedDict mutated during iteration')
                return {'grants': 1}

        self.assertEqual(metrics.read_registry(Flaky()), {'grants': 1, 'budget_bytes': 7})
        calls[:] = []
        with self.assertRaises(RuntimeError):
            metrics.read_registry(Flaky(), attempts=2)

    def test_metric_names(self):
        self.assertEqual(metrics.metric_name('bytes')[:2], ('qwen_prefix_registry_bytes', 'gauge'))
        self.assertEqual(metrics.metric_name('pins')[:2], ('qwen_prefix_registry_pins', 'gauge'))
        self.assertEqual(metrics.metric_name('staged_now')[:2], ('qwen_prefix_staged_now', 'gauge'))
        self.assertEqual(metrics.metric_name('trim_loss_tokens')[:2], ('qwen_prefix_trim_loss_tokens', 'counter'))
        self.assertEqual(metrics.metric_name('restore_ms')[:2], ('qwen_prefix_restore_seconds', 'counter'))
        self.assertEqual(metrics.metric_name('some_new_counter')[:2], ('qwen_prefix_some_new_counter', 'counter'))

    def test_rows_sum_live_writers_and_report_the_export_s_health(self):
        one = dict(values=dict(grants=2, bytes=100, restore_ms=500), written_unix=90.0, kill_switch_file=False)
        two = dict(values=dict(grants=3, bytes=50, capture_ms=2000), written_unix=99.0, kill_switch_file=True)
        rows = {name: (kind, value) for kind, name, _, value in metric_rows_of([one, two], dead=2, unreadable=1,
                                                                                 now=100.0, local={'grants': 1})}
        self.assertEqual(rows['qwen_prefix_grants'], ('counter', 6))
        self.assertEqual(rows['qwen_prefix_registry_bytes'], ('gauge', 150))
        self.assertEqual(rows['qwen_prefix_restore_seconds'], ('counter', 0.5))
        self.assertEqual(rows['qwen_prefix_capture_seconds'], ('counter', 2.0))
        self.assertEqual(rows['qwen_prefix_export_writers'], ('gauge', 3))
        self.assertEqual(rows['qwen_prefix_export_age_seconds'], ('gauge', 10.0))
        self.assertEqual(rows['qwen_prefix_export_dead_files'], ('gauge', 2))
        self.assertEqual(rows['qwen_prefix_export_unreadable_files'], ('gauge', 1))
        self.assertEqual(rows['qwen_prefix_kill_switch_file'], ('gauge', 1))
        empty = {name: value for _, name, _, value in metrics.metric_rows([], now=5.0)}
        self.assertEqual((empty['qwen_prefix_export_writers'], empty['qwen_prefix_export_age_seconds']), (0, 0.0))
        self.assertNotIn('qwen_prefix_grants', empty)


def metric_rows_of(documents, **kwargs):
    return metrics.metric_rows(documents, **kwargs)


class Liveness(unittest.TestCase):
    def test_this_process_is_alive_and_nonsense_is_not(self):
        self.assertTrue(metrics.process_alive(os.getpid()))
        for pid in (0, -1, None, 'x'):
            self.assertFalse(metrics.process_alive(pid))

    def test_a_reused_pid_is_not_the_writer(self):
        proc = tempfile.mkdtemp(prefix='qwen-prefix-proc-')
        try:
            os.makedirs(os.path.join(proc, '4242'))
            fields = ['S'] + ['0'] * 18 + ['777'] + ['0'] * 10
            with open(os.path.join(proc, '4242', 'stat'), 'w') as handle:
                handle.write('4242 (python3 (x)) ' + ' '.join(fields) + '\n')
            self.assertEqual(metrics.start_ticks(4242, proc), 777)
            self.assertTrue(metrics.process_alive(4242, 777, proc=proc))
            self.assertFalse(metrics.process_alive(4242, 778, proc=proc))
            self.assertIsNone(metrics.start_ticks(4343, proc))
        finally:
            shutil.rmtree(proc, ignore_errors=True)

    def test_without_proc_it_asks_the_kernel(self):
        def gone(pid, signal):
            raise ProcessLookupError(pid)

        def foreign(pid, signal):
            raise PermissionError(pid)

        self.assertFalse(metrics.process_alive(99, None, proc='/nonexistent', kill=gone))
        self.assertTrue(metrics.process_alive(99, None, proc='/nonexistent', kill=foreign))
        self.assertTrue(metrics.process_alive(99, 5, proc='/nonexistent', kill=lambda pid, signal: None))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='qwen-prefix-metrics-')
        self.directory = os.path.join(self.tmp, 'metrics')
        self.registry = None
        self.logs = []

    def tearDown(self):
        metrics.stop_exporter()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def exporter(self, pid=4242, **kwargs):
        kwargs.setdefault('kill_switch_path', os.path.join(self.tmp, 'prefix-reuse.off'))
        return metrics.Exporter(self.directory, 0.5, clock=lambda: 1000.0, lookup=lambda: self.registry, pid=pid,
                                logger=lambda *a: self.logs.append(a[0] % a[1:]), **kwargs)

    def test_nothing_is_written_without_a_registry(self):
        exporter = self.exporter()
        self.assertFalse(exporter.publish())
        self.assertFalse(os.path.exists(self.directory))

    def test_a_snapshot_is_written_whole_and_read_back(self):
        self.registry = registry_with_traffic()
        exporter = self.exporter()
        self.assertTrue(exporter.publish())
        self.assertTrue(exporter.publish())
        self.assertEqual(sorted(os.listdir(self.directory)), ['4242.json'])
        document = metrics.load_document(os.path.join(self.directory, '4242.json'))
        self.assertEqual((document['pid'], document['writes'], document['written_unix']), (4242, 2, 1000.0))
        self.assertEqual(document['values']['grant_tokens'], 6144)
        self.assertFalse(document['kill_switch_file'])
        self.assertEqual(len([line for line in self.logs if line.startswith('metrics export pid=4242')]), 1)
        with open(os.path.join(self.tmp, 'prefix-reuse.off'), 'w'):
            pass
        exporter.publish()
        self.assertTrue(metrics.load_document(os.path.join(self.directory, '4242.json'))['kill_switch_file'])

    def test_the_collector_reads_live_files_and_counts_the_rest(self):
        self.registry = registry_with_traffic()
        self.exporter(pid=1).publish()
        self.exporter(pid=2).publish()
        with open(os.path.join(self.directory, '3.json'), 'w') as handle:
            handle.write('{not json')
        with open(os.path.join(self.directory, 'notes.txt'), 'w') as handle:
            handle.write('ignored')
        live, dead, unreadable = metrics.read_documents(self.directory, alive=lambda pid, ticks=None: pid == 1)
        self.assertEqual(([document['pid'] for document in live], dead, unreadable), ([1], 1, 1))
        collector = metrics.PrefixCollector(self.directory, alive=lambda pid, ticks=None: pid == 1,
                                            clock=lambda: 1003.0, lookup=lambda: None)
        rows = {name: value for _, name, _, value in collector.rows()}
        self.assertEqual(rows['qwen_prefix_grants'], 3)
        self.assertEqual(rows['qwen_prefix_export_age_seconds'], 3.0)
        self.assertEqual((rows['qwen_prefix_export_dead_files'], rows['qwen_prefix_export_unreadable_files']), (1, 1))
        in_process = metrics.PrefixCollector(os.path.join(self.tmp, 'absent'), lookup=lambda: self.registry)
        rows = {name: value for _, name, _, value in in_process.rows()}
        self.assertEqual((rows['qwen_prefix_grants'], rows['qwen_prefix_export_writers']), (3, 1))

    def test_a_new_engine_sweeps_the_files_of_gone_writers(self):
        self.registry = registry_with_traffic()
        for pid in (7, 8):
            self.exporter(pid=pid).publish()
        exporter = self.exporter(pid=9, alive=lambda pid, ticks=None: pid == 8)
        self.assertEqual(exporter.sweep(), 1)
        self.assertEqual(sorted(os.listdir(self.directory)), ['8.json'])

    def test_the_thread_exports_until_stopped_and_starts_once_per_process(self):
        self.registry = registry_with_traffic()
        made = []

        def factory(directory, interval):
            made.append((directory, interval))
            return metrics.Exporter(directory, interval, lookup=lambda: self.registry,
                                    logger=lambda *a: None)

        environ = {'QWEN_PREFIX_METRICS_DIR': self.directory, 'QWEN_PREFIX_METRICS_INTERVAL_S': '0.1'}
        first = metrics.start_exporter(environ, factory)
        self.assertIs(metrics.start_exporter(environ, factory), first)
        self.assertEqual(made, [(self.directory, 0.1)])
        path = os.path.join(self.directory, '%d.json' % os.getpid())
        deadline = time.time() + 10
        while time.time() < deadline and (metrics.load_document(path) or {}).get('writes', 0) < 2:
            time.sleep(0.05)
        self.assertGreaterEqual(metrics.load_document(path)['writes'], 2)
        self.assertTrue(metrics.stop_exporter())
        self.assertFalse(metrics.stop_exporter())

    def test_a_failing_export_is_counted_and_logged_not_raised(self):
        class Broken(object):
            def snapshot(self):
                raise ValueError('boom')

        self.registry = Broken()
        exporter = self.exporter()
        stop = threading.Event()
        stop.set()
        exporter.run(stop)
        self.assertEqual(exporter.errors, 1)
        self.assertTrue(any('metrics export failed (1 so far): ValueError: boom' in line for line in self.logs))

    def test_settings(self):
        self.assertEqual(metrics.settings({}), (metrics.DEFAULT_DIRECTORY, metrics.DEFAULT_INTERVAL_S))
        self.assertEqual(metrics.settings({'QWEN_PREFIX_METRICS_INTERVAL_S': 'x'})[1], metrics.DEFAULT_INTERVAL_S)
        self.assertEqual(metrics.settings({'QWEN_PREFIX_METRICS_INTERVAL_S': '0'})[1], 0.1)


class ForkTests(unittest.TestCase):
    def tearDown(self):
        metrics._EXPORTER.pop('fork_hook', None)

    def test_a_forked_child_exports_only_a_registry_of_its_own(self):
        started = []
        with mock.patch.object(metrics, 'start_exporter', lambda environ=None, factory=None: started.append(1) or 'e'):
            self.assertIsNone(metrics.after_fork_in_child(lookup=lambda: object()))
            self.assertEqual(metrics.after_fork_in_child(lookup=lambda: None), 'e')
        self.assertEqual(started, [1])

    def test_the_fork_hook_is_registered_once(self):
        metrics._EXPORTER.pop('fork_hook', None)
        registered = []
        self.assertTrue(metrics.export_in_forked_children(register=lambda **hooks: registered.append(hooks)))
        self.assertFalse(metrics.export_in_forked_children(register=lambda **hooks: registered.append(hooks)))
        self.assertEqual([sorted(hooks) for hooks in registered], [['after_in_child']])


class FakeRegistry(object):
    def __init__(self):
        self.collectors = []

    def register(self, collector):
        self.collectors.append(collector)


def chat_module():
    """A stand-in for vllm.entrypoints.openai.chat_completion.serving (0.25.1: an async method)."""
    module = types.ModuleType(metrics.CHAT_MODULE)

    class OpenAIServingChat(object):
        async def create_chat_completion(self, request, raw_request=None):
            """Chat Completion API."""
            return ('served', request, raw_request)

    module.OpenAIServingChat = OpenAIServingChat
    return module


class RequestShapeTests(unittest.TestCase):
    def setUp(self):
        metrics._REQUESTS.clear()

    def tearDown(self):
        metrics._REQUESTS.clear()

    def test_the_labels(self):
        request = types.SimpleNamespace(reasoning_effort='high', cache_salt='abc', chat_template_kwargs={
            'preserve_thinking': False, 'enable_thinking': True, 'reasoning_effort': 'xhigh'})
        self.assertEqual(metrics.request_labels(request), ('high', 'xhigh', 'false', 'true', 'set'))
        self.assertEqual(metrics.request_labels(types.SimpleNamespace()), ('unset',) * 4 + ('unset',))
        odd = types.SimpleNamespace(reasoning_effort='extreme', chat_template_kwargs={'preserve_thinking': 'true',
                                                                                      'enable_thinking': 1})
        # C1: the template strips reasoning for "true" and 1 too: they must not read as the boolean
        self.assertEqual(metrics.request_labels(odd), ('other', 'unset', 'other', 'other', 'unset'))
        self.assertEqual(metrics.request_labels(types.SimpleNamespace(chat_template_kwargs=['x']))[2], 'unset')

    def test_a_chat_completion_is_counted_and_still_served(self):
        import asyncio

        module = chat_module()
        self.assertTrue(metrics.wrap_chat_serving(module))
        self.assertFalse(metrics.wrap_chat_serving(module))
        method = module.OpenAIServingChat.create_chat_completion
        self.assertEqual((method.__name__, method.__doc__), ('create_chat_completion', 'Chat Completion API.'))
        request = types.SimpleNamespace(reasoning_effort=None, chat_template_kwargs=None, cache_salt=None)
        served = asyncio.run(module.OpenAIServingChat().create_chat_completion(request, 'raw'))
        self.assertEqual(served, ('served', request, 'raw'))
        asyncio.run(module.OpenAIServingChat().create_chat_completion(request))
        self.assertEqual(metrics.request_counts(), {('unset',) * 5: 2})
        with mock.patch.object(metrics, 'request_labels', side_effect=ValueError('x')):
            asyncio.run(module.OpenAIServingChat().create_chat_completion(request))
        self.assertEqual(metrics.request_counts()[('other',) * 5], 1)
        self.assertFalse(metrics.wrap_chat_serving(types.ModuleType('no_chat_class')))


class CollectorInstallTests(unittest.TestCase):
    def setUp(self):
        metrics._COLLECTED.clear()

    def tearDown(self):
        metrics._COLLECTED.clear()

    def test_the_registry_vllm_serves_carries_the_collector_once(self):
        shared = FakeRegistry()
        module = types.ModuleType('fake_vllm_prometheus')
        module.get_prometheus_registry = lambda: shared
        self.assertTrue(metrics.wrap_prometheus_module(module, '/tmp/x'))
        self.assertFalse(metrics.wrap_prometheus_module(module, '/tmp/x'))
        self.assertIs(module.get_prometheus_registry(), shared)
        self.assertIs(module.get_prometheus_registry(), shared)
        self.assertEqual(len(shared.collectors), 1)
        self.assertIsInstance(shared.collectors[0], metrics.PrefixCollector)
        # multiprocess mode hands out a new registry per call (v1/metrics/prometheus.py:46-50)
        fresh = []
        module = types.ModuleType('fake_vllm_prometheus_mp')
        module.get_prometheus_registry = lambda: fresh.append(FakeRegistry()) or fresh[-1]
        metrics.wrap_prometheus_module(module)
        module.get_prometheus_registry()
        module.get_prometheus_registry()
        self.assertEqual([len(registry.collectors) for registry in fresh], [1, 1])

    def test_a_registry_that_refuses_the_collector_still_serves(self):
        class Refusing(object):
            def register(self, collector):
                raise ValueError('Duplicated timeseries')

        refusing = Refusing()
        module = types.ModuleType('fake_vllm_prometheus_refusing')
        module.get_prometheus_registry = lambda: refusing
        metrics.wrap_prometheus_module(module)
        self.assertIs(module.get_prometheus_registry(), refusing)

    def test_install_wraps_now_or_on_import(self):
        hooks = []
        module = types.ModuleType(metrics.PROMETHEUS_MODULE)
        module.get_prometheus_registry = lambda: FakeRegistry()
        self.assertTrue(metrics.install_collector(lambda name, callback: hooks.append((name, callback)), modules={}))
        self.assertEqual([name for name, _ in hooks], [metrics.PROMETHEUS_MODULE, metrics.CHAT_MODULE])
        hooks[0][1](module)
        self.assertTrue(module.get_prometheus_registry._qwen_prefix)
        other = types.ModuleType(metrics.PROMETHEUS_MODULE)
        other.get_prometheus_registry = lambda: FakeRegistry()
        chat = chat_module()
        self.assertTrue(metrics.install_collector(hooks.append, modules={metrics.PROMETHEUS_MODULE: other,
                                                                          metrics.CHAT_MODULE: chat}))
        self.assertEqual(len(hooks), 2)
        self.assertTrue(other.get_prometheus_registry._qwen_prefix)
        self.assertTrue(chat.OpenAIServingChat.create_chat_completion._qwen_prefix)

    def test_collect_builds_prometheus_families(self):
        made = []

        class Family(object):
            def __init__(self, kind, name, documentation, value=None, labels=None):
                self.samples = []
                made.append((kind, name, value if labels is None else tuple(labels)))

            def add_metric(self, labels, value):
                self.samples.append((tuple(labels), value))

        core = types.ModuleType('prometheus_client.core')
        core.CounterMetricFamily = lambda *a, **k: Family('counter', *a, **k)
        core.GaugeMetricFamily = lambda *a, **k: Family('gauge', *a, **k)
        package = types.ModuleType('prometheus_client')
        package.core = core
        collector = metrics.PrefixCollector('/nonexistent', lookup=lambda: registry_with_traffic())
        with mock.patch.dict(metrics._REQUESTS, {('high',) + ('unset',) * 3 + ('set',): 2}, clear=True), \
                mock.patch.dict(sys.modules, {'prometheus_client': package, 'prometheus_client.core': core}):
            families = list(collector.collect())
        self.assertEqual(len(families), len(made))
        self.assertIn(('counter', 'qwen_prefix_grants', 3), made)
        self.assertIn(('gauge', 'qwen_prefix_registry_bytes', 4000), made)
        self.assertIn(('counter', 'qwen_prefix_restore_seconds', 1.5), made)
        self.assertEqual(made[-1], ('counter', 'qwen_prefix_chat_requests', metrics.REQUEST_LABELS))
        self.assertEqual(families[-1].samples, [(('high', 'unset', 'unset', 'unset', 'set'), 2)])
        self.assertEqual(collector.describe(), [])

    @unittest.skipUnless(HAVE_PROMETHEUS, 'prometheus_client is a vLLM dependency: this runs in the image')
    def test_the_real_exposition(self):
        from prometheus_client import CollectorRegistry, generate_latest

        tmp = tempfile.mkdtemp(prefix='qwen-prefix-expo-')
        try:
            registry = CollectorRegistry()
            exporter = metrics.Exporter(tmp, 1.0, lookup=registry_with_traffic, logger=lambda *a: None)
            exporter.publish()
            metrics.register_collector(registry, tmp)
            with mock.patch.dict(metrics._REQUESTS, {('medium', 'unset', 'unset', 'true', 'set'): 4}, clear=True):
                text = generate_latest(registry).decode('utf-8')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        sample = [line for line in text.splitlines() if line.startswith('qwen_prefix_chat_requests_total{')]
        self.assertEqual(len(sample), 1, text)
        for pair in ('reasoning_effort="medium"', 'template_reasoning_effort="unset"', 'preserve_thinking="unset"',
                     'enable_thinking="true"', 'cache_salt="set"'):
            self.assertIn(pair, sample[0])
        self.assertTrue(sample[0].endswith('} 4.0'), sample[0])
        self.assertIn('qwen_prefix_grants_total 3.0', text)
        self.assertIn('qwen_prefix_grant_tokens_total 6144.0', text)
        self.assertIn('qwen_prefix_trim_loss_tokens_total 128.0', text)
        self.assertIn('qwen_prefix_restore_seconds_total 1.5', text)
        self.assertIn('qwen_prefix_registry_bytes 4000.0', text)
        self.assertIn('qwen_prefix_export_writers 1.0', text)
        self.assertIn('# TYPE qwen_prefix_orphans_total counter', text)


class ContractBootTests(unittest.TestCase):
    """serving_c2_contract.boot starts the right half in the right process under a prefix profile."""

    def boot(self, profile_name, orig_argv):
        import serving_c2_contract as contract

        here = os.path.dirname(os.path.abspath(__file__))
        calls = []
        environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILE': profile_name,
                   'QWEN_C2_PROFILES': os.path.join(here, 'qwen_c2_profiles.json')}
        argv = list(sys.argv)
        meta_path = list(sys.meta_path)
        path = list(sys.path)
        try:
            with mock.patch.object(metrics, 'install_collector', lambda on_import: calls.append('collector')), \
                    mock.patch.object(metrics, 'start_exporter', lambda: calls.append('exporter')), \
                    mock.patch.object(metrics, 'export_in_forked_children', lambda: calls.append('fork')), \
                    mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
                    mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][0]), \
                    mock.patch.object(contract, 'log', lambda *a: calls.append(a[0] % a[1:] if len(a) > 1 else a[0])), \
                    mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': profile_name}):
                sys.argv[:] = ['-m', '--port', '8001']
                profile = contract.boot(environ=environ, orig_argv=orig_argv)
        finally:
            sys.argv[:] = argv
            sys.meta_path[:] = meta_path
            sys.path[:] = path
        return profile, environ, calls

    def test_the_api_server_collects_and_the_engine_exports(self):
        import serving_c2_contract as contract

        api = ['python3', '-m', contract.API_SERVER, '--port', '8001']
        profile, environ, calls = self.boot('general-prefix', api)
        self.assertEqual(environ[contract.PREFIX_SWITCH], '1')
        self.assertEqual([call for call in calls if call in ('collector', 'exporter', 'fork')], ['collector', 'fork'])
        self.assertTrue(any('prefix reuse on (QWEN_PREFIX_REUSE=1' in str(call) for call in calls))
        profile, environ, calls = self.boot('general-prefix', ['python3', '-c', 'from multiprocessing.spawn import x'])
        self.assertEqual([call for call in calls if call in ('collector', 'exporter', 'fork')], ['exporter', 'fork'])
        for name in ('general', 'exact', 'c2'):
            with self.subTest(profile=name):
                profile, environ, calls = self.boot(name, api)
                self.assertNotIn(contract.PREFIX_SWITCH, environ)
                self.assertEqual([call for call in calls if call in ('collector', 'exporter', 'fork')], [])


if __name__ == '__main__':
    unittest.main()
