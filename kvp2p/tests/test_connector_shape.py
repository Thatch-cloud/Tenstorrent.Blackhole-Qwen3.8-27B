"""The shape of the connector: what importing it does, how the two roles meet, how the tier starts, refuses and stays off."""

import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import fakes
from fakes import FakeRuntime
from harness import RigCase, tokens

PROMPT = 2300
FULL_BLOCKS = PROMPT // 128
OBJECTS = FULL_BLOCKS + 40

# The shape of the production kwarg (design 3.1), with generic names for the paths.
PRODUCTION_KWARG = {
    'spec_name': 'TTP2PNvmeSpec', 'spec_module_path': 'ttkvp2p.spec',
    'blocks_per_chunk': 1, 'offload_prompt_only': True, 'layout': 'gemma-chipmajor-v1', 'persistence': 'process',
    'retention_ack': True, 'nvme_device': '/dev/kv-drive', 'nvme_status_dir': '/run/kv-drive-status',
    'p2pmem_glob': '/run/kv-p2pmem/*/allocate', 'p2p_window_id': 202, 'staging_bytes_per_bank': 67108864,
    'spill_banks': [0, 1], 'restore_banks': [2, 3, 4, 5, 6, 7], 'spill_subrun_objects': 8, 'io_threads': 8,
    'io_queue_depth_objects': 8, 'device_budget_objects_per_step': 96, 'idle_wait_ms': 3,
    'load_deadline_base_s': 2.0, 'load_deadline_gbps': 1.0, 'stall_latch_s': 20.0, 'strikes_to_latch': 3,
    'aer_latch': 'uncorrectable', 'ttl_s': 86400, 'free_floor_pct': 5, 'reserve_bytes': 8589934592,
    'store_threshold': 1, 'min_store_tokens': 4096, 'daily_write_cap_bytes': 329000000000, 'require_sed': True,
    'require_salt': False, 'persist_multimodal': False, 'control_dir': '/var/lib/kvp2p/ctl',
    'state_file': '/var/lib/kvp2p/state.json', 'kv_epoch': 1,
}


class ImportTests(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        """I14: the API-server process imports the connector for the HMA check. Nothing may start, open, connect or load a device library."""
        code = textwrap.dedent('''
            import importlib, os, sys, threading
            sys.modules['ttnn'] = None                 # an import of either raises ImportError
            sys.modules['qwen_kv_read'] = None
            import torch, numpy                        # what vLLM loads anyway
            import vllm.distributed.kv_transfer.kv_connector.factory
            import vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector
            import vllm.distributed.kv_transfer.kv_connector.v1.offloading.common
            import vllm.distributed.kv_transfer.kv_connector.v1.offloading.config
            import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler
            import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker
            import vllm.v1.kv_offload.factory, vllm.v1.kv_offload.base, vllm.v1.outputs, vllm.config
            before_threads = threading.active_count()
            before_fds = sorted(os.listdir('/proc/self/fd'))
            before_cwd = sorted(os.listdir('.'))
            events = []
            def hook(event, args):
                if event == 'open':
                    mode = args[1] if len(args) > 1 else None
                    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                    if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
                        events.append((event, args))
                elif event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system', 'os.fork', 'os.exec',
                               'os.posix_spawn', 'ctypes.dlopen', 'ctypes.dlsym', 'mmap.__new__'):
                    events.append((event, args))
            sys.addaudithook(hook)
            for name in ('ttkvp2p', 'ttkvp2p.registry', 'ttkvp2p.spec', 'ttkvp2p.worker', 'ttkvp2p.connector'):
                importlib.import_module(name)
            assert not events, events
            assert threading.active_count() == before_threads, 'a thread was started'
            assert sorted(os.listdir('/proc/self/fd')) == before_fds, 'a file was left open'
            assert sorted(os.listdir('.')) == before_cwd, 'a file was created'
            for lazy in ('ttkvp2p.runtime', 'ttkvp2p.manager', 'ttkvp2p.cardlink', 'ttkvp2p.p2pio', 'qwen_kv_read', 'ttnn'):
                assert sys.modules.get(lazy) is None, lazy + ' was imported'
            print('clean')
        ''')
        import tempfile
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([fakes.STUB_DIR, fakes.PACKAGE_ROOT])
        env.setdefault('VLLM_USE_V2_MODEL_RUNNER', '0')
        with tempfile.TemporaryDirectory() as cwd:
            done = subprocess.run([sys.executable, '-B', '-c', code], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-3000:])
        self.assertIn('clean', done.stdout)


class ConfigTests(unittest.TestCase):
    def test_the_production_kwarg_parses_and_has_the_designed_values(self):
        from ttkvp2p.spec import P2PConfig

        config = P2PConfig.from_extra_config(PRODUCTION_KWARG)
        self.assertEqual((config.spill_banks, config.restore_banks), ((0, 1), (2, 3, 4, 5, 6, 7)))
        self.assertEqual((config.spill_subrun_objects, config.device_budget_objects_per_step), (8, 96))
        self.assertEqual((config.load_deadline_base_s, config.load_deadline_gbps, config.stall_latch_s, config.strikes_to_latch),
                         (2.0, 1.0, 20.0, 3))
        self.assertEqual(config, P2PConfig.from_extra_config({k: v for k, v in PRODUCTION_KWARG.items() if k not in ('spec_name',)}),
                         'the defaults are the production values')

    def test_malformed_configuration_is_refused_with_the_key_named(self):
        from ttkvp2p.spec import P2PConfig

        for bad, fragment in (
            ({'strikes_to_latch': 'three'}, 'strikes_to_latch'),
            ({'retention_ack': 'yes'}, 'retention_ack'),
            ({'blocks_per_chunk': 4}, 'blocks_per_chunk'),
            ({'spill_banks': [0, 1], 'restore_banks': [1, 2]}, 'disjoint'),
            ({'slot_objects': 500}, 'slot_objects'),
            ({'load_deadline_gbps': 0}, 'deadline'),
        ):
            with self.assertRaises(ValueError, msg=str(bad)) as caught:
                P2PConfig.from_extra_config(bad)
            self.assertIn(fragment, str(caught.exception))

    def test_object_bytes_is_the_tt_geometry_not_the_gpu_page_size(self):
        from ttkvp2p.spec import LAYOUTS, TTP2PNvmeSpec

        geometry = LAYOUTS['gemma-chipmajor-v1']
        self.assertEqual(geometry.object_bytes, 2785280)
        self.assertEqual(geometry.shard_object_bytes, 696320)
        config = SimpleNamespace(extra_config=PRODUCTION_KWARG, enable_kv_cache_events=False,
                                 cache=SimpleNamespace(tokens_per_hash=128, blocks_per_chunk=1),
                                 groups=(SimpleNamespace(tokens_per_block=128),) * 6,
                                 parallel=SimpleNamespace(world_size=1, rank=0))
        spec = TTP2PNvmeSpec(config)
        self.assertIsNone(spec.refusal)
        self.assertEqual(spec.object_bytes, 2785280)
        with self.assertRaises(RuntimeError):
            spec.get_worker(None)

    def test_the_registry_checks_its_geometry_against_the_layout(self):
        from ttkvp2p.registry import TTKVRegistry
        from ttkvp2p.spec import LAYOUTS

        layout = LAYOUTS['gemma-chipmajor-v1']
        good = dict(mesh_device=None, tensors=tuple(range(10)), layer_to_tensor_idx={}, group_layers=(), num_blocks=19305,
                    pages_per_block=64, row_bytes_per_chip=69632, shard_bdf=('a', 'b', 'c', 'd'))
        self.assertEqual(TTKVRegistry(**good).problems(layout), [])
        self.assertEqual(TTKVRegistry(**good).object_bytes, 2785280)
        for change, fragment in (
            ({'tensors': tuple(range(9))}, 'tensors'), ({'shard_bdf': ('a', 'b')}, 'shards'),
            ({'row_bytes_per_chip': 69632 + 64}, 'row bytes'), ({'pages_per_block': 32}, 'pages per block'),
            ({'num_blocks': 1}, 'num_blocks'),
        ):
            problems = TTKVRegistry(**{**good, **change}).problems(layout)
            self.assertTrue(any(fragment in p for p in problems), (change, problems))


class TopologyTests(unittest.TestCase):
    def test_the_tier_runs_only_on_the_uni_executor_with_one_worker(self):
        from ttkvp2p.connector import topology_refusal

        def config(world_size, backend, policy='recompute'):
            return SimpleNamespace(parallel_config=SimpleNamespace(world_size=world_size, distributed_executor_backend=backend),
                                   kv_transfer_config=SimpleNamespace(kv_load_failure_policy=policy))

        self.assertIsNone(topology_refusal(config(1, 'uni')))
        self.assertIsNone(topology_refusal(config(1, None)))
        self.assertIsNone(topology_refusal(config(1, type('UniProcExecutor', (), {}))))
        self.assertIn('uni', topology_refusal(config(1, 'mp')))
        self.assertIn('world_size 2', topology_refusal(config(2, 'uni')))

    def test_the_tier_needs_the_recompute_policy(self):
        from ttkvp2p.connector import topology_refusal

        config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=1, distributed_executor_backend='uni'),
                                 kv_transfer_config=SimpleNamespace(kv_load_failure_policy='fail'))
        self.assertIn('recompute', topology_refusal(config))


class SharedRuntimeTests(RigCase):
    def test_two_roles_share_one_runtime_and_the_last_release_closes_it(self):
        rig = self.rig()
        self.assertEqual(len(FakeRuntime.instances), 1, 'one runtime per drive path')
        self.assertEqual(FakeRuntime.acquire_log, [('/dev/test-kv', 'worker'), ('/dev/test-kv', 'scheduler')],
                         'the worker opens first, the scheduler attaches')
        self.assertIs(rig.manager.runtime, rig.worker.rt.runtime)
        self.assertEqual(rig.runtime.roles, ['worker', 'scheduler'])
        rig.worker_conn.shutdown()                                      # the executor shuts down before the scheduler
        self.assertEqual(rig.runtime.released, ['worker'])
        self.assertFalse(rig.runtime.closed)
        self.assertEqual(rig.runtime.state, 'AVAILABLE', 'the scheduler role is still attached')
        rig.scheduler.shutdown()
        self.assertEqual(rig.runtime.released, ['worker', 'scheduler'])
        self.assertTrue(rig.runtime.closed)
        rig.closed = True                                               # already shut down in the order under test
        rig.drive.close()
        from ttkvp2p import worker as worker_module
        worker_module.BACKEND_FACTORY = rig.saved_factory
        fakes.restore_modules(rig.saved_modules)


class StartupTests(RigCase):
    def test_a_failed_host_check_leaves_the_tier_unavailable_with_zero_device_footprint(self):
        rig = self.rig(start_worker=False, host_ok=False)
        rig.worker_conn.register_kv_caches(rig.registry)                # never raises
        self.assertIsNone(rig.worker)
        self.assertIn('host checks failed', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(rig.factory.builds, 0, 'no staging, no windows, no programs')
        self.assertEqual(rig.card.log, [])
        self.assertEqual(rig.runtime.state, 'UNAVAILABLE')
        self.assertEqual(rig.runtime.released, ['worker'], 'the worker role let go of the runtime')
        # serving carries on: lookups miss, stores drop, requests finish
        prompt = tokens(PROMPT, 'host')
        rig.add('r1', prompt)
        rig.drain()
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.drain()
        self.assertEqual(rig.manager.written(), 0)
        self.assertEqual(rig.loads, [])

    def test_a_registry_that_does_not_match_the_layout_is_refused_before_any_device_work(self):
        import dataclasses

        rig = self.rig(start_worker=False)
        wrong = dataclasses.replace(rig.registry, row_bytes_per_chip=rig.registry.row_bytes_per_chip * 2)
        rig.worker_conn.register_kv_caches(wrong)
        self.assertIsNone(rig.worker)
        self.assertIn('row bytes', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(rig.factory.builds, 0)
        self.assertEqual(rig.runtime.released, [], 'the runtime was never acquired for a worker that could not be built')

    def test_a_runtime_without_what_the_worker_needs_is_refused_loudly(self):
        import sys

        rig = self.rig(start_worker=False)

        class Bare:
            state = 'UNAVAILABLE'
            host_ok = True
            released = []

            @classmethod
            def acquire(cls, cfg, role):
                return cls()

            def release(self, role):
                self.released.append(role)

        sys.modules['ttkvp2p.runtime'].KVP2PRuntime = Bare
        rig.worker_conn.register_kv_caches(rig.registry)
        self.assertIsNone(rig.worker)
        reason = rig.worker_conn.connector_worker.unavailable_reason
        self.assertIn('lacks', reason)
        for name in ('mark_bad', 'slot_offset', 'allocator.io_begin'):
            self.assertIn(name, reason)
        self.assertEqual(rig.factory.builds, 0)
        self.assertEqual(Bare.released, ['worker'])

    def test_a_backend_that_fails_to_start_leaves_the_tier_off_and_the_runtime_released(self):
        from ttkvp2p import worker as worker_module

        rig = self.rig(start_worker=False)

        def broken(**kwargs):
            raise RuntimeError('the warm-up threw')

        worker_module.BACKEND_FACTORY = broken
        rig.worker_conn.register_kv_caches(rig.registry)                # never raises
        self.assertIsNone(rig.worker)
        self.assertIn('the warm-up threw', rig.worker_conn.connector_worker.unavailable_reason)
        self.assertEqual(rig.runtime.state, 'UNAVAILABLE')
        self.assertEqual(rig.runtime.released, ['worker'])

    def test_with_no_worker_every_job_fails_at_once_and_is_reported_once(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata, TransferJob
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from ttkvp2p.spec import NvmeLoadStoreSpec

        rig = self.rig(with_scheduler=False, start_worker=False, host_ok=False)
        connector = rig.worker_conn
        connector.register_kv_caches(rig.registry)
        load = TransferJob('req-1', NvmeLoadStoreSpec([1], [1], None), GPULoadStoreSpec([1], [1], [0]))
        store = TransferJob('req-2', GPULoadStoreSpec([1], [1], [0]), NvmeLoadStoreSpec([2], [1]))
        connector.bind_connector_metadata(OffloadingConnectorMetadata(load_jobs={1: load}, store_jobs={2: store}))
        connector.handle_preemptions(connector._get_connector_metadata())
        connector.start_load_kv(None)
        results = connector.get_transfer_results(set())                 # prepare_store_kv queues the store for the next step
        self.assertEqual((results.finished_recving, results.failed_recving), ({'req-1'}, {'req-1'}))
        connector.clear_connector_metadata()
        connector.bind_connector_metadata(OffloadingConnectorMetadata(load_jobs={}, store_jobs={}))
        connector.start_load_kv(None)
        connector.get_transfer_results(set())
        meta = connector.build_connector_worker_meta()
        self.assertEqual(set(meta.completed_jobs), {1, 2})


class RefusedTierTests(RigCase):
    def test_an_engine_on_another_executor_serves_normally_and_the_tier_stays_a_no_op(self):
        from ttkvp2p.spec import RefusedManager

        rig = self.rig(executor='mp')
        self.assertIsInstance(rig.connector_scheduler.manager._manager, RefusedManager)
        self.assertIn('uni', rig.connector_scheduler.manager._manager.reason)
        self.assertIsNone(rig.worker)
        prompt = tokens(PROMPT, 'mp')
        rig.add('r1', prompt)
        rig.drain()
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.drain()
        self.assertEqual(rig.loads, [])
        self.assertTrue({entry[1] for entry in rig.card.log} <= {'forward', 'step'}, 'no tier op ever ran')

    def test_a_refused_manager_takes_the_real_managers_complete_store_signature(self):
        from ttkvp2p.spec import RefusedManager

        manager = RefusedManager('off')
        self.assertIsNone(manager.complete_store(['k'], None, success=True, digests=None))
        self.assertIsNone(manager.complete_store(['k'], None, success=False))

    def test_an_unknown_layout_refuses_the_tier(self):
        from ttkvp2p.spec import RefusedManager

        rig = self.rig(extra={'layout': 'no-such-layout'})
        self.assertIsInstance(rig.connector_scheduler.manager._manager, RefusedManager)
        self.assertIsNone(rig.worker)
        rig.add('r1', tokens(PROMPT, 'x'))
        rig.drain()

    def test_a_scheduler_role_that_cannot_attach_to_the_runtime_stays_off_and_serving_continues(self):
        import sys
        from ttkvp2p.spec import RefusedManager

        rig = self.rig(with_scheduler=False)
        saved = sys.modules.pop('ttkvp2p.manager')
        self.addCleanup(lambda: sys.modules.__setitem__('ttkvp2p.manager', saved))
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.structured_output import StructuredOutputManager
        scheduler = Scheduler(vllm_config=rig.vllm_config, kv_cache_config=rig.kv_cache_config,
                              structured_output_manager=StructuredOutputManager(rig.vllm_config), block_size=128,
                              hash_block_size=128, include_finished_set=False, log_stats=False)
        self.assertIsInstance(scheduler.connector.connector_scheduler.manager._manager, RefusedManager)


    def test_a_manager_that_fails_to_build_releases_the_schedulers_runtime_reference(self):
        import sys
        from ttkvp2p.spec import RefusedManager

        rig = self.rig(with_scheduler=False)

        def broken(cfg, runtime):
            raise RuntimeError('the manager could not be built')

        sys.modules['ttkvp2p.manager'].NvmeKVManager = broken
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.structured_output import StructuredOutputManager
        scheduler = Scheduler(vllm_config=rig.vllm_config, kv_cache_config=rig.kv_cache_config,
                              structured_output_manager=StructuredOutputManager(rig.vllm_config), block_size=128,
                              hash_block_size=128, include_finished_set=False, log_stats=False)
        self.assertIsInstance(scheduler.connector.connector_scheduler.manager._manager, RefusedManager)
        self.assertEqual(FakeRuntime.acquire_log, [('/dev/test-kv', 'worker'), ('/dev/test-kv', 'scheduler')])
        self.assertEqual(rig.runtime.released, ['scheduler'], 'the reference acquired for the manager that never came to be was given back')
        self.assertEqual(rig.runtime.roles, ['worker'])
        self.assertFalse(rig.runtime.closed, 'the worker role is still attached')


class ConnectorSurfaceTests(unittest.TestCase):
    def test_the_connector_keeps_hma_and_asks_for_no_layout(self):
        from vllm.distributed.kv_transfer.kv_connector.v1 import SupportsHMA
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
        from ttkvp2p.connector import TTP2PConnector

        self.assertTrue(issubclass(TTP2PConnector, OffloadingConnector))
        self.assertTrue(issubclass(TTP2PConnector, SupportsHMA))
        self.assertIsNone(TTP2PConnector.get_required_kvcache_layout(SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None))))

    def test_vllm_resolves_the_connector_by_module_path_and_sees_hma_support(self):
        from vllm.config import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        config = KVTransferConfig(kv_connector='TTP2PConnector', kv_connector_module_path='ttkvp2p.connector', kv_role='kv_both',
                                  kv_load_failure_policy='recompute', kv_connector_extra_config=dict(PRODUCTION_KWARG))
        self.assertTrue(KVConnectorFactory.supports_hma_config(config))
        self.assertEqual(KVConnectorFactory.get_connector_class(config).__name__, 'TTP2PConnector')


class RequestTagTests(RigCase):
    def test_every_request_is_tagged_with_a_tenant_and_a_multimodal_flag_before_the_manager_sees_it(self):
        from ttkvp2p.connector import blake2b8
        from ttkvp2p.spec import ReqTag

        rig = self.rig()
        rig.add('plain', tokens(300, 'a'))
        rig.add('salted', tokens(300, 'b'), salt='tenant-a')
        tags = {rid: rig.connector_scheduler._req_status[rid].req_context.get_state(ReqTag) for rid in ('plain', 'salted')}
        self.assertEqual((tags['plain'].tenant, tags['plain'].has_mm, tags['plain'].admitted, tags['plain'].no_retry),
                         (None, False, None, False))
        self.assertEqual(tags['salted'].tenant, blake2b8('tenant-a'))
        self.assertEqual(len(tags['salted'].tenant), 8)
        self.assertNotIn(b'tenant-a', tags['salted'].tenant)

    def test_a_request_with_multimodal_features_is_flagged(self):
        from ttkvp2p.spec import ReqTag

        rig = self.rig()
        request = SimpleNamespace(request_id='mm', kv_transfer_params=None, cache_salt=None, mm_features=[object()],
                                  block_hashes=[], num_prompt_tokens=10, num_tokens=10)
        rig.connector_scheduler.on_new_request(request)
        tag = rig.connector_scheduler._req_status['mm'].req_context.get_state(ReqTag)
        self.assertTrue(tag.has_mm)


if __name__ == '__main__':
    unittest.main()
