"""A rig that drives vLLM 0.30.0's REAL Scheduler, KV cache manager and OffloadingConnectorScheduler (through the ttkvp2p
subclasses) against the numpy card and the file drive.

The model runner is played by ``FakeRunner``, which calls the connector exactly as the plugin hooks of WP5 will (design 2.7 and
5.1): ``handle_preemptions`` and ``bind_connector_metadata`` before the forward; the forward writes KV into the card (as an op in
the card's queue, so a gather enqueued before it reads the old rows and one after it the new ones); then ``_kv_finish``:
``start_load_kv``, ``get_transfer_results``, ``get_block_ids_with_load_errors``, ``build_connector_worker_meta``,
``clear_connector_metadata``. Its output carries the ``KVConnectorOutput`` into ``update_from_output``.

The KV rows are a deterministic function of the block hash and the group, so after every step the rig checks, for every running
request, that every non-null block below its computed tokens holds exactly that content: restored, locally cached or
recomputed, wrong KV is never accepted.
"""

import os
import random
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np

from fakes import (  # noqa: F401  (prepares the environment before vLLM is imported)
    CHIPS, ROW, TENSORS, TINY,
    FakeBackendFactory, FakeCard, FakeDrive, FakeRuntime, FakeTensor,
    install_fake_modules, kv_rows, restore_modules,
)

import torch  # noqa: E402
from transformers import GPT2Config  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from ttkvp2p import worker as worker_module  # noqa: E402
from ttkvp2p.connector import TTP2PConnector  # noqa: E402
from ttkvp2p.registry import TTKVRegistry  # noqa: E402
from ttkvp2p.spec import register_layout  # noqa: E402

BLOCK = 128
WINDOW = 1024
GROUPS = 6                      # one full-attention group and five sliding-window groups, as Gemma's
SHARDS = ('card-0', 'card-1', 'card-2', 'card-3')       # stand-ins for the cards' PCI addresses

_MODEL = {}


def model_config() -> ModelConfig:
    """A stand-in model config (the scheduler reads its sizes and flags only); built once per process."""
    if 'config' not in _MODEL:
        root = tempfile.mkdtemp(prefix='ttkvp2p-model-')
        GPT2Config(n_positions=65536, n_embd=256, n_layer=1, n_head=4, architectures=['GPT2LMHeadModel']).save_pretrained(root)
        _MODEL['dir'] = root
        _MODEL['config'] = ModelConfig(model=root, dtype='float32', max_model_len=65536, skip_tokenizer_init=True, seed=0)
    return _MODEL['config']


def tokens(count: int, seed) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 200000) for _ in range(count)]


def default_extra_config(slot_objects: int = 8, **overrides) -> dict:
    extra = {
        'spec_name': 'TTP2PNvmeSpec', 'spec_module_path': 'ttkvp2p.spec', 'blocks_per_chunk': 1, 'offload_prompt_only': True,
        'layout': TINY.name, 'persistence': 'process', 'retention_ack': True,
        'nvme_device': '/dev/test-kv', 'staging_bytes_per_bank': 16384,
        'spill_banks': [0, 1], 'restore_banks': [2, 3, 4, 5, 6, 7], 'spill_subrun_objects': 4,
        'device_budget_objects_per_step': 96, 'idle_wait_ms': 0.0,
        'load_deadline_base_s': 2.0, 'load_deadline_gbps': 1.0, 'stall_latch_s': 20.0, 'strikes_to_latch': 3,
        'slot_objects': slot_objects, 'shutdown_drain_s': 2.0,
    }
    extra.update(overrides)
    return extra


class FakeRunner:
    """The TT model runner's connector duties plus a forward that writes deterministic KV."""

    def __init__(self, rig: 'Rig'):
        self.rig = rig
        self.blocks: Dict[str, List[List[int]]] = {}
        self.computed: Dict[str, int] = {}
        self.forwards = 0

    def _track(self, so) -> Dict[str, int]:
        computed = {}
        for new in so.scheduled_new_reqs:
            self.blocks[new.req_id] = [list(ids) for ids in new.block_ids]
            computed[new.req_id] = new.num_computed_tokens
        cached = so.scheduled_cached_reqs
        for position, req_id in enumerate(cached.req_ids):
            new_ids = cached.new_block_ids[position]
            if req_id in cached.resumed_req_ids:
                self.blocks[req_id] = [list(ids) for ids in new_ids]
            elif new_ids is not None:
                for group, ids in enumerate(new_ids):
                    self.blocks[req_id][group].extend(ids)
            computed[req_id] = cached.num_computed_tokens[position]
        return computed

    def _forward(self, so, computed: Dict[str, int]) -> Dict[str, List[int]]:
        card = self.rig.card
        card.mark('forward', n=self.forwards)
        self.forwards += 1
        sampled = {}
        for req_id, scheduled in so.num_scheduled_tokens.items():
            request = self.rig.scheduler.requests[req_id]
            start = computed[req_id]
            end = start + scheduled
            for index in range(start // BLOCK, end // BLOCK):
                if index >= len(request.block_hashes):
                    break
                for group in range(GROUPS):
                    table = self.blocks[req_id][group]
                    if index >= len(table) or table[index] == 0:
                        continue
                    for tensor in range(TENSORS):
                        card.write_rows(tensor, table[index], kv_rows(request.block_hashes[index], group, tensor))
            sampled[req_id] = [7] if end >= request.num_tokens else []
        return sampled

    def execute_model(self, so) -> ModelRunnerOutput:
        conn = self.rig.worker_conn
        meta = so.kv_connector_metadata
        for job in meta.load_jobs.values():
            self.rig.loads.append((job.req_id, tuple(job.dst_spec.group_sizes)))
        self.rig.flushes.append(set(meta.jobs_to_flush or ()))
        self.rig.card.mark('step', n=self.rig.steps)
        conn.handle_preemptions(meta)
        conn.bind_connector_metadata(meta)
        computed = self._track(so)
        sampled = self._forward(so, computed)
        # _kv_finish
        conn.start_load_kv(None)
        results = conn.get_transfer_results(so.finished_req_ids)
        invalid = conn.get_block_ids_with_load_errors()
        worker_meta = conn.build_connector_worker_meta()
        conn.clear_connector_metadata()
        kv_out = KVConnectorOutput(
            finished_sending=results.finished_sending or None, finished_recving=results.finished_recving or None,
            failed_recving=results.failed_recving, invalid_block_ids=invalid, kv_connector_worker_meta=worker_meta)
        req_ids = list(so.num_scheduled_tokens)
        return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                 sampled_token_ids=[sampled[r] for r in req_ids], kv_connector_output=kv_out)


class Rig:
    def __init__(self, *, slot_objects: int = 8, num_slots: int = 400, num_blocks: int = 600, extra: Optional[dict] = None,
                 max_batched: int = 4096, workers: int = 1, tmp: Optional[str] = None, start_worker: bool = True,
                 checks: bool = True, with_scheduler: bool = True, executor: Optional[str] = None, host_ok: bool = True,
                 async_scheduling: bool = False, real_tier=None):
        register_layout(TINY)
        self.real_tier = real_tier              # WP3a: a tier_kit.RealTier makes the rig use the REAL runtime and manager instead of the fakes
        self.tmp = tmp or tempfile.mkdtemp(prefix='ttkvp2p-rig-')
        self.slot_objects = slot_objects
        self.num_blocks = num_blocks
        self.checks = checks
        self.steps = 0
        if real_tier is None:
            self.saved_modules = install_fake_modules()
            FakeRuntime.reset(num_slots)
            FakeRuntime.host_ok_default = host_ok
        else:
            self.saved_modules = {}
            real_tier.prepare(self, num_slots)
        self.card = FakeCard(num_blocks=num_blocks, slot_objs=slot_objects)
        self.drive = FakeDrive(os.path.join(self.tmp, 'drive.img'), num_slots)
        self.factory = FakeBackendFactory(self.card, self.drive, slot_objects, workers=workers)
        self.saved_factory = worker_module.BACKEND_FACTORY
        worker_module.BACKEND_FACTORY = self.factory
        self.extra = default_extra_config(slot_objects, **(extra or {}))
        if real_tier is not None:
            self.extra.update(real_tier.extra_config(self))
            self.extra.update(extra or {})

        kv_transfer = KVTransferConfig(
            kv_connector='TTP2PConnector', kv_connector_module_path='ttkvp2p.connector', kv_role='kv_both',
            kv_load_failure_policy='recompute', kv_connector_extra_config=self.extra)
        self.vllm_config = VllmConfig(
            model_config=model_config(), device_config=DeviceConfig(device='cpu'),
            scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=max_batched, max_model_len=65536,
                                             is_encoder_decoder=False, enable_chunked_prefill=True,
                                             async_scheduling=async_scheduling),
            cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=True), parallel_config=ParallelConfig(),
            kv_transfer_config=kv_transfer)
        self.vllm_config.cache_config.num_gpu_blocks = num_blocks
        if executor is not None:
            self.vllm_config.parallel_config.distributed_executor_backend = executor
        register_all_kvcache_specs(self.vllm_config)
        spec_args = dict(block_size=BLOCK, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
        groups = [KVCacheGroupSpec(['g0.l0', 'g0.l1'], FullAttentionSpec(**spec_args))]
        for group in range(1, GROUPS):
            groups.append(KVCacheGroupSpec(['g%d.l0' % group, 'g%d.l1' % group], SlidingWindowSpec(sliding_window=WINDOW, **spec_args)))
        self.kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)

        hash_fn = get_hash_fn_by_name(self.vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        self.hasher = get_request_block_hasher(BLOCK, hash_fn)

        self.tensors = tuple(FakeTensor(i) for i in range(TENSORS))
        self.registry = TTKVRegistry(
            mesh_device=SimpleNamespace(name='mesh'), tensors=self.tensors,
            layer_to_tensor_idx={name: i % TENSORS for i, name in enumerate(l for g in groups for l in g.layer_names)},
            group_layers=tuple(tuple(g.layer_names) for g in groups), num_blocks=num_blocks, pages_per_block=TINY.pages_per_block,
            row_bytes_per_chip=ROW, shard_bdf=SHARDS)

        # The worker role first, as under the uni executor (V core.py: the worker initialises first).
        self.worker_conn = TTP2PConnector(self.vllm_config, KVConnectorRole.WORKER, self.kv_cache_config)
        if start_worker:
            self.worker_conn.register_kv_caches(self.registry)
            if real_tier is not None:
                real_tier.after_start(self)
        self.scheduler = None
        self.async_mode = async_scheduling
        self.pending: List[tuple] = []          # async scheduling: batches executed but not yet through update_from_output
        if with_scheduler:
            scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
            self.scheduler = scheduler_cls(vllm_config=self.vllm_config, kv_cache_config=self.kv_cache_config,
                                       structured_output_manager=StructuredOutputManager(self.vllm_config), block_size=BLOCK,
                                       hash_block_size=BLOCK, include_finished_set=False, log_stats=False)
        self.runner = FakeRunner(self)
        self.requests: Dict[str, Request] = {}
        self.loads: List[tuple] = []            # (req_id, per-group block counts) of every load job the scheduler created
        self.flushes: List[set] = []            # jobs_to_flush of every step's metadata
        self.expected_internal_errors = 0       # a test that provokes some says how many (None: it checks them itself)
        self.closed = False

    # --- the pieces under test, by name
    @property
    def connector_scheduler(self):
        return self.scheduler.connector.connector_scheduler

    @property
    def manager(self):
        return self.connector_scheduler.manager._manager

    @property
    def worker(self):
        return self.worker_conn.connector_worker.worker

    @property
    def runtime(self) -> FakeRuntime:
        if self.real_tier is not None:
            return self.real_tier.runtime(self)
        return next(iter(FakeRuntime.instances.values()))

    @property
    def io(self):
        return self.factory.io

    # --- requests and steps
    def add(self, rid: str, prompt: List[int], max_tokens: int = 1, salt: Optional[str] = None) -> Request:
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        request = Request(request_id=rid, prompt_token_ids=list(prompt), sampling_params=params, pooling_params=None,
                          cache_salt=salt, block_hasher=self.hasher)
        self.requests[rid] = request
        self.scheduler.add_request(request)
        return request

    def step(self):
        so = self.scheduler.schedule()
        out = self.runner.execute_model(so)
        if self.async_mode:
            # two batches in flight (EngineCore.step_with_batch_queue): the output of step n reaches update_from_output after
            # schedule(n + 1), and an empty step flushes the queue
            self.pending.append((so, out))
            while len(self.pending) > 1 or (so.total_num_scheduled_tokens == 0 and self.pending):
                done_so, done_out = self.pending.pop(0)
                self.scheduler.update_from_output(done_so, done_out)
        else:
            self.scheduler.update_from_output(so, out)
        self.steps += 1
        if self.checks:
            self.assert_kv_exact()
        return so, out

    def idle(self) -> bool:
        return not self.pending and not self.scheduler.has_requests()

    def run_until(self, condition: Callable[[], bool], limit: int = 3000, what: str = 'the condition') -> int:
        for count in range(limit):
            if condition():
                return count
            so, _ = self.step()
            if so.total_num_scheduled_tokens == 0:
                time.sleep(0.0005)              # waiting on an I/O thread
        raise AssertionError('%s not reached in %d steps (running=%s waiting=%s)' % (
            what, limit, [r.request_id for r in self.scheduler.running],
            [(r.request_id, r.status.name) for r in list(self.scheduler.waiting) + list(self.scheduler.skipped_waiting)]))

    def finish(self, *rids: str, limit: int = 3000) -> None:
        self.run_until(lambda: all(self.requests[r].is_finished() for r in rids), limit, 'requests %s finished' % (rids,))

    def drain(self, limit: int = 3000) -> None:
        """Run until nothing is running and the connector has no job in flight."""
        self.run_until(self.idle, limit, 'the engine idle')

    def reset_device_cache(self) -> None:
        """POST /reset_prefix_cache?reset_external=false: the device prefix cache only."""
        assert self.scheduler.reset_prefix_cache(reset_running_requests=False, reset_connector=False)

    # --- the invariant: wrong KV is never accepted
    def expected_object(self, key: bytes) -> np.ndarray:
        block_hash, group = key[:-4], int.from_bytes(key[-4:], 'big')
        return np.concatenate([np.concatenate([kv_rows(block_hash, group, t)[d] for t in range(TENSORS)]) for d in range(CHIPS)])

    def assert_drive_exact(self) -> int:
        """Every WRITTEN entry's slot on the drive holds exactly the object its key names. Returns how many were checked."""
        checked = 0
        for key, entry in self.manager.index.items():
            if entry.state == 'WRITTEN':
                got = self.drive.slot_object(entry.slot)
                assert np.array_equal(got, self.expected_object(key)), 'slot %d does not hold its key\'s object' % entry.slot
                checked += 1
        return checked

    def assert_kv_exact(self) -> None:
        manager = self.scheduler.kv_cache_manager
        for request in self.scheduler.running:
            if request.status != RequestStatus.RUNNING:
                continue
            tables = manager.get_block_ids(request.request_id)
            full = min(request.num_computed_tokens // BLOCK, len(request.block_hashes))
            for group in range(GROUPS):
                for index in range(min(full, len(tables[group]))):
                    block = tables[group][index]
                    if block == 0:
                        continue
                    for tensor in range(TENSORS):
                        want = kv_rows(request.block_hashes[index], group, tensor)
                        got = self.card.row(tensor, block)
                        assert np.array_equal(got, want), (
                            'request %s group %d block index %d (device block %d) tensor %d holds the wrong KV'
                            % (request.request_id, group, index, block, tensor))

    # --- teardown
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        connector_worker = self.worker_conn.connector_worker
        worker = connector_worker.worker if connector_worker is not None else None
        try:
            if self.io is not None:
                self.io.release_all()
            if self.scheduler is not None:
                self.scheduler.shutdown()
            self.worker_conn.shutdown()
        finally:
            if self.io is not None:
                self.io.close()                 # idempotent; lets every held call go and joins the pool threads
            self.drive.close()
            worker_module.BACKEND_FACTORY = self.saved_factory
            restore_modules(self.saved_modules)
            if self.real_tier is not None:
                self.real_tier.finish(self)
        if worker is not None and self.expected_internal_errors is not None:
            # An unexpected exception inside the state machine is latched and survived in production; in the tests it is a failure.
            assert worker.stats['internal_error'] == self.expected_internal_errors, (
                'the worker hit %d internal errors' % worker.stats['internal_error'])


# --------------------------------------------------------------------------------------------------------------------
# Worker-level kit: jobs by hand, without the scheduler
# --------------------------------------------------------------------------------------------------------------------

from vllm.v1.kv_offload.base import GPULoadStoreSpec  # noqa: E402

from ttkvp2p.spec import NvmeLoadStoreSpec  # noqa: E402


class WorkerKit:
    """Submit store and load jobs straight to the TTP2PWorker of a rig and pump it, the way ``_kv_finish`` does."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.worker = rig.worker
        self.results: Dict[int, object] = {}
        self.reports: Dict[int, int] = {}
        self.written: Dict[int, tuple] = {}           # store job id -> (slots, gens, blocks)
        self.clock_now = 1000.0

    # --- a fake clock for deadline and stall tests
    def use_fake_clock(self):
        self.worker._clock = lambda: self.clock_now
        self.worker._last_progress = self.clock_now

    def advance(self, seconds: float):
        self.clock_now += seconds

    # --- content
    def fill_blocks(self, blocks: List[int], seed) -> None:
        rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
        for block in blocks:
            for tensor in range(TENSORS):
                self.rig.card.write_rows(tensor, block, rng.integers(0, 256, size=(CHIPS, ROW), dtype=np.uint8))

    def object_of(self, block: int) -> np.ndarray:
        return self.rig.card.block_bytes(block)

    # --- jobs
    def alloc(self, n: int):
        allocator = self.rig.runtime.allocator
        slots = allocator.alloc(n)
        assert slots is not None
        return slots, [allocator.gen[s] for s in slots]

    def store(self, job_id: int, blocks: List[int], slots=None, gens=None) -> tuple:
        if slots is None:
            slots, gens = self.alloc(len(blocks))
        self.written[job_id] = (list(slots), list(gens), list(blocks))
        src = GPULoadStoreSpec(list(blocks), group_sizes=[len(blocks)], block_indices=[0])
        assert self.worker.submit_store(job_id, src, NvmeLoadStoreSpec(list(slots), list(gens)))
        return slots, gens

    def expected_for(self, store_job: int) -> np.ndarray:
        result = self.rig.runtime.store_results[store_job]
        assert result.success, 'the store job did not succeed (%s)' % result.reason
        return result.digests

    def load(self, job_id: int, store_job: int, dst_blocks: List[int], expected: Optional[np.ndarray] = None) -> None:
        slots, gens, _ = self.written[store_job]
        if expected is None:
            expected = self.expected_for(store_job)
        dst = GPULoadStoreSpec(list(dst_blocks), group_sizes=[len(dst_blocks)], block_indices=[0])
        assert self.worker.submit_load(job_id, NvmeLoadStoreSpec(list(slots), list(gens), expected), dst)

    # --- pumping
    def pump_once(self, **hint):
        if hint:
            self.worker.begin_step(**hint)
        for result in self.worker.get_finished():
            self.reports[result.job_id] = self.reports.get(result.job_id, 0) + 1
            self.results[result.job_id] = result

    def pump(self, until: Callable[[], bool], limit: int = 2000, sleep: float = 0.0005, what: str = 'the condition') -> int:
        for count in range(limit):
            self.pump_once()
            if until():
                return count
            time.sleep(sleep)
        raise AssertionError('%s not reached in %d pumps (active=%s results=%s)' % (
            what, limit, self.worker.active_job_ids, {k: (v.success, v.reason) for k, v in self.results.items()}))

    def pump_until_reported(self, *job_ids: int, limit: int = 2000) -> None:
        self.pump(lambda: all(j in self.results for j in job_ids), limit, what='jobs %s reported' % (job_ids,))


class RigCase(unittest.TestCase):
    """Base class: ``self.rig(...)`` builds a rig that is closed when the test ends."""

    def rig(self, **kwargs) -> Rig:
        rig = Rig(**kwargs)
        self.addCleanup(rig.close)
        return rig
