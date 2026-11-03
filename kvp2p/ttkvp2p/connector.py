"""TTP2PConnector: vLLM's OffloadingConnector, with the TT worker and the failure handling the stock worker lacks.

What is kept from the stock connector: the whole scheduler side (lookups, sliding-window windows, store reachability,
fences, flush, preemption, LRU touch), ``SupportsHMA``, and the worker's submission discipline. What is replaced:

  * the worker role registers a ``TTKVRegistry`` (not a dict of torch tensors) and builds a ``TTP2PWorker``;
  * ``get_finished`` never asserts: a failed job is reported exactly once, in the same output that carries
    ``finished_recving`` / ``failed_recving`` (invariant I1);
  * ``get_transfer_results`` keeps ``super()`` (the only path that runs ``get_finished`` and so ``prepare_store_kv``,
    invariant I9) and adds the failed receives;
  * the scheduler role tags each request (tenant, multimodal, admission verdict, no-retry), flags a request no-retry when
    its load failed for a reason that leaves the keys valid, and checks that every store job pairs the right block with
    the right key (invariant I6);
  * the scheduler role settles every completed store job itself: vLLM calls ``manager.complete_store(keys, ctx)`` with no job id,
    no digests and no way to say the store failed, so ``update_connector_output`` takes the job's ``StoreResult`` from the runtime
    and hands the manager ``complete_store(keys in the manager's order, ctx, success=..., digests=...)`` instead. A missing or
    inconsistent result is a failed store.

Importing this module imports neither ttnn nor the extension (invariant I14).
"""

import collections
import dataclasses
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata, KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import PrepareStoreOutput, ReqContext, TransferResult
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.outputs import KVConnectorOutput

from ttkvp2p.registry import TTKVRegistry
from ttkvp2p.spec import NvmeLoadStoreSpec, ReqTag, TierRefused
from ttkvp2p.worker import BAD_KEY_REASONS, TTTransferResult

logger = logging.getLogger(__name__)

TOMBSTONE_LIMIT = 65536              # I1: the last 64k job ids the connector worker has reported


def blake2b8(salt: str) -> bytes:
    """The tenant tag of a cache salt: 8 bytes, never the salt itself."""
    return hashlib.blake2b(salt.encode('utf-8'), digest_size=8).digest()


def topology_refusal(vllm_config: VllmConfig) -> Optional[str]:
    """Why the tier will not run on this engine, or None. It needs the ``uni`` executor, exactly one worker, and the recompute
    policy for failed loads (any other policy would turn a failed restore into a failed request)."""
    parallel = vllm_config.parallel_config
    if parallel.world_size != 1:
        return 'world_size %d (the tier needs exactly one worker)' % parallel.world_size
    backend = parallel.distributed_executor_backend
    if not (backend is None or backend == 'uni' or getattr(backend, '__name__', '') == 'UniProcExecutor'):
        return 'distributed executor %r (the tier needs the uni executor)' % (backend,)
    policy = getattr(vllm_config.kv_transfer_config, 'kv_load_failure_policy', 'recompute')
    if policy != 'recompute':
        return 'kv_load_failure_policy %r (the tier needs "recompute": a failure must become a miss, not a failed request)' % (policy,)
    return None


# --------------------------------------------------------------------------------------------------------------------
# Worker -> scheduler metadata
# --------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class TTWorkerMetadata(OffloadingWorkerMetadata):
    """The stock completed-jobs map plus WHY each failed load failed (the scheduler role needs it for the no-retry flag)."""

    failed_loads: Dict[int, str] = dataclasses.field(default_factory=dict)

    def aggregate(self, other):
        merged = super().aggregate(other)
        failed = dict(self.failed_loads)
        failed.update(getattr(other, 'failed_loads', {}))
        return TTWorkerMetadata(completed_jobs=merged.completed_jobs, transfer_stats=merged.transfer_stats, failed_loads=failed)


# --------------------------------------------------------------------------------------------------------------------
# Scheduler role
# --------------------------------------------------------------------------------------------------------------------

class _RecordingManager:
    """Stands between the stock scheduler and the manager.

    * ``prepare_store``: remembers, per store spec, the keys the manager accepted and in what order (the I6 check and the settling of
      the job need them: the scheduler keeps a set).
    * ``complete_store``: vLLM's call carries only the job's key SET, always as a success. The scheduler role stages the real outcome of
      each completed store job in ``staged`` (keyed by the identity of that key set) just before the stock loop runs, and this proxy
      then makes the one call the manager is meant to get: keys in the manager's order, ``success``, and the digests. A call for a
      job nobody staged cannot be trusted (there are no digests to hand over), so it reaches the manager as a failed store.
    """

    def __init__(self, manager, record, on_error):
        self._manager = manager
        self._record = record
        self._on_error = on_error
        self.staged: Dict[int, Tuple[Any, List[bytes], bool, Any]] = {}      # id(job keys) -> (job keys, ordered keys, success, digests)

    def prepare_store(self, keys, req_context: ReqContext) -> Optional[PrepareStoreOutput]:
        output = self._manager.prepare_store(keys, req_context)
        if output is not None and output.keys_to_store:
            self._record(output)
        return output

    def complete_store(self, keys, req_context: ReqContext, success: bool = True) -> None:
        staged = self.staged.pop(id(keys), None)
        if staged is None or staged[0] is not keys:
            logger.error('ttkvp2p: vLLM completed a store job this connector did not settle (%d keys); it is treated as failed', len(keys))
            self._call(list(keys), req_context, False, None)
            return
        _, ordered_keys, ok, digests = staged
        self._call(ordered_keys, req_context, ok, digests)

    def _call(self, keys, req_context: ReqContext, success: bool, digests) -> None:
        try:
            if success:
                self._manager.complete_store(keys, req_context, success=True, digests=digests)
            else:
                self._manager.complete_store(keys, req_context, success=False)      # the frozen signature, nothing added
            return
        except Exception:
            logger.exception('ttkvp2p: manager.complete_store raised for %d keys (success=%s)', len(keys), success)
            self._on_error('manager.complete_store raised')
        if success:
            # The keys must not stay ALLOCATED for good (every lookup of them would defer): give them up as a failed store.
            try:
                self._manager.complete_store(keys, req_context, success=False)
            except Exception:
                logger.exception('ttkvp2p: manager.complete_store(success=False) raised too')

    def __getattr__(self, name):
        return getattr(self._manager, name)


class TTP2PConnectorScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec, vllm_config, kv_cache_config):
        super().__init__(spec, vllm_config, kv_cache_config)
        self._spec = spec
        self._block_pool = None
        self._accepted: Dict[int, Tuple[Any, List[bytes]]] = {}       # id(store_spec) -> (store_spec, keys in the manager's order)
        self._store_jobs: Dict[int, Tuple[Any, List[bytes]]] = {}     # store job id -> (its store spec, keys in the manager's order)
        self.misalignments = 0
        self.manager = _RecordingManager(self.manager, self._record_store, self._manager_failed)

    def _record_store(self, output: PrepareStoreOutput) -> None:
        self._accepted[id(output.store_spec)] = (output.store_spec, list(output.keys_to_store))

    def _manager_failed(self, why: str) -> None:
        """The manager raised inside a completion: the tier goes off (the engine carries on, as with any internal error)."""
        latch = getattr(self._spec, 'latch_tier', None)
        if latch is not None:
            latch(why)

    def reset_cache(self) -> None:
        super().reset_cache()
        self._store_jobs.clear()
        self._accepted.clear()

    def bind_block_pool(self, pool) -> None:
        self._block_pool = pool

    # --- requests
    def on_new_request(self, request) -> None:
        super().on_new_request(request)
        try:
            salt = getattr(request, 'cache_salt', None)
            tag = ReqTag(tenant=blake2b8(salt) if salt else None,
                         has_mm=bool(getattr(request, 'mm_features', None)),
                         admitted=None, no_retry=False)
            self._req_status[request.request_id].req_context.set_state(tag)
        except Exception:      # an untagged request is treated as a plain one by the manager; the engine must not stop for it
            logger.exception('ttkvp2p: could not tag request %s', request.request_id)

    # --- I6: every store job pairs the right block with the right key
    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        accepted, self._accepted = self._accepted, {}
        for job_id, job in meta.store_jobs.items():
            spec = job.dst_spec
            record = accepted.get(id(spec))
            if record is None or record[0] is not spec:
                continue
            self._store_jobs[job_id] = record          # the job's keys in the manager's order, for its completion
            try:
                problem = self._misalignment(job, record[1])
            except Exception as error:      # a check that cannot run is a store that does not happen
                logger.exception('ttkvp2p: the alignment check of store job %d failed', job_id)
                problem = 'the check itself failed: %s' % (type(error).__name__,)
            if problem is not None:
                self.misalignments += 1
                logger.error('ttkvp2p: store job %d is not aligned (%s); the worker will fail it instead of storing it (I6)', job_id, problem)
                if isinstance(spec, NvmeLoadStoreSpec):
                    spec.poison = 'I6: ' + problem
        return meta

    def _misalignment(self, job, keys: List[bytes]) -> Optional[str]:
        block_ids = [int(b) for b in job.src_spec.block_ids]
        slot_ids = getattr(job.dst_spec, 'slot_ids', None)
        if len(block_ids) != len(keys) or (slot_ids is not None and len(slot_ids) != len(keys)):
            return '%d keys, %d slots, %d source blocks' % (len(keys), -1 if slot_ids is None else len(slot_ids), len(block_ids))
        # (1) The scheduler pairs blocks with slots by position in ITS order (group by group, chunk by chunk). The keys the
        # manager accepted must come back in that same order, or slot i would receive the data of another key's block.
        req_status = self._req_status.get(job.req_id)
        if req_status is not None:
            intended = {}
            per_chunk = self.config.blocks_per_chunk
            for group_config, state in zip(self.config.kv_group_configs, req_status.group_states):
                for idx, key in enumerate(state.offload_keys):
                    at = idx * per_chunk + per_chunk - 1
                    if at < len(state.block_ids):
                        intended[key] = state.block_ids[at]
            if all(key in intended for key in keys):
                for position, key in enumerate(keys):
                    if intended[key] != block_ids[position]:
                        return 'position %d: key belongs to block %d, the job reads block %d' % (position, intended[key], block_ids[position])
        # (2) Where the source block is already published in the prefix cache, its hash IS the key.
        pool = self._block_pool
        if pool is not None:
            for position, (key, block_id) in enumerate(zip(keys, block_ids)):
                block_hash = pool.blocks[block_id].block_hash
                if block_hash is not None and bytes(block_hash) != bytes(key):
                    return 'position %d: block %d is hashed for another key' % (position, block_id)
        return None

    # --- failures: a load that failed for a reason that leaves the keys valid must not be retried by its own recompute (I13)
    def update_connector_output(self, connector_output: KVConnectorOutput):
        try:
            failed = getattr(connector_output.kv_connector_worker_meta, 'failed_loads', None)
            for job_id, reason in (failed or {}).items():
                if job_id < self._stale_job_threshold:
                    continue
                job_status = self._jobs.get(job_id)
                if job_status is None or job_status.is_store or reason in BAD_KEY_REASONS:
                    continue
                req_status = self._req_status.get(job_status.req_id)
                tag = req_status.req_context.get_state(ReqTag) if req_status is not None else None
                if tag is not None:
                    tag.no_retry = True
        except Exception:
            logger.exception('ttkvp2p: flagging failed loads no-retry failed')
        try:
            self._stage_store_outcomes(connector_output)
        except Exception:
            # Anything not staged reaches the manager as a failed store (see _RecordingManager.complete_store): never as a success.
            logger.exception('ttkvp2p: settling the completed store jobs failed')
        try:
            super().update_connector_output(connector_output)
        finally:
            self.manager.staged.clear()

    # --- stores: the job's real outcome and its digests reach the manager (design 2.5 steps 5 and 6)
    def _stage_store_outcomes(self, connector_output: KVConnectorOutput) -> None:
        """For each store job this output completes, decide ``complete_store``'s arguments and stage them for the stock loop.

        The stock loop calls ``manager.complete_store(job.keys, ctx)`` for it: no job id, no digests, and a ``success`` that is
        always True. The worker's verdict and the source digests live in the runtime's ``store_results[job_id]``, so it is here,
        where the job id, the keys and the result meet, that the outcome is settled."""
        completed = getattr(connector_output.kv_connector_worker_meta, 'completed_jobs', None)
        threshold = self._stale_job_threshold
        for job_id in [j for j in self._store_jobs if j < threshold]:
            del self._store_jobs[job_id]
        if not completed:
            return
        for job_id, count in completed.items():
            if job_id < threshold:
                self._spec.take_store_result(job_id)         # a job from before a reset: the stock loop skips it; drop what it left
                continue
            job_status = self._jobs.get(job_id)
            if job_status is None or not job_status.is_store or job_status.pending_count - count > 0:
                continue                                      # a load, an unknown id, or a job other workers have yet to finish
            ordered_keys, success, digests = self._store_outcome(job_id, job_status)
            self.manager.staged[id(job_status.keys)] = (job_status.keys, ordered_keys, success, digests)

    def _store_outcome(self, job_id: int, job_status) -> Tuple[List[bytes], bool, Any]:
        """(keys in the manager's order, success, digests [n, chips] in that order). A missing or inconsistent result is a failure."""
        record = self._store_jobs.pop(job_id, None)
        result = self._spec.take_store_result(job_id)
        if record is None:
            logger.error('ttkvp2p: store job %d has no record of its keys in the manager\'s order; it is treated as failed', job_id)
            return list(job_status.keys), False, None
        store_spec, keys = record
        if result is None:
            # The worker prunes reported results after STORE_RESULT_LIMIT, and a job it never saw has none: neither is a success.
            logger.warning('ttkvp2p: store job %d has no result from the worker; it is treated as failed', job_id)
            return keys, False, None
        if not getattr(result, 'success', False):
            return keys, False, None
        problem = self._result_problem(result, store_spec, keys, job_status)
        if problem is not None:
            logger.error('ttkvp2p: the result of store job %d cannot be trusted (%s); it is treated as failed', job_id, problem)
            return keys, False, None
        return keys, True, result.digests

    @staticmethod
    def _result_problem(result, store_spec, keys: List[bytes], job_status) -> Optional[str]:
        if len(keys) != len(job_status.keys) or set(keys) != set(job_status.keys):
            return 'the keys the manager accepted are not the keys of the job'
        digests = getattr(result, 'digests', None)
        if digests is None or getattr(digests, 'ndim', 0) != 2 or digests.shape[0] != len(keys):
            return 'digests %s for %d keys' % (getattr(digests, 'shape', None), len(keys))
        if tuple(result.slot_ids) != tuple(store_spec.slot_ids) or tuple(result.generations) != tuple(store_spec.generations):
            return 'the result names other drive slots than the job\'s store spec'
        return None


# --------------------------------------------------------------------------------------------------------------------
# Worker role
# --------------------------------------------------------------------------------------------------------------------

class TTP2PConnectorWorker(OffloadingConnectorWorker):
    def __init__(self, spec, vllm_config, kv_cache_config):
        super().__init__(spec, vllm_config, kv_cache_config)
        self._connector_worker_meta = TTWorkerMetadata()
        self._failed_recving: Set[str] = set()
        self._unserved: List[TTTransferResult] = []           # jobs failed here because there is no worker
        self._reported: 'collections.OrderedDict[int, None]' = collections.OrderedDict()    # tombstones (I1)
        self.unavailable_reason: Optional[str] = None

    # --- start-up (design 2.9): never raises
    def register_kv_caches(self, kv_caches: TTKVRegistry) -> None:
        if self.worker is not None:
            return
        try:
            self.worker = self.spec.get_tt_worker(kv_caches)
        except TierRefused as error:
            self.unavailable_reason = str(error)
            logger.warning('ttkvp2p: the tier stays off: %s', error)
        except Exception as error:
            self.unavailable_reason = '%s: %s' % (type(error).__name__, error)
            logger.error('ttkvp2p: the tier failed to start and stays off', exc_info=True)

    # --- submission: with no worker every job fails at once, through the normal report
    def _fail_unserved(self, job_id: int, reason: str = 'unavailable') -> None:
        self._unserved.append(TTTransferResult(job_id, False, reason=reason))

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        if self.worker is not None:
            super().handle_preemptions(kv_connector_metadata)
            return
        for job_id in kv_connector_metadata.jobs_to_flush or ():
            entry = kv_connector_metadata.store_jobs.pop(job_id, None)
            if entry is not None:
                self._fail_unserved(job_id)
        for job_id, _src, _dst in self._unsubmitted_store_jobs:
            self._fail_unserved(job_id)
        self._unsubmitted_store_jobs.clear()

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        if self.worker is not None:
            super().start_kv_transfers(metadata)
            return
        for job_id, _src, _dst in self._unsubmitted_store_jobs:
            self._fail_unserved(job_id)
        self._unsubmitted_store_jobs.clear()
        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            self._fail_unserved(job_id)

    def begin_step(self, *, decode_active: bool = True, idle: bool = False) -> None:
        """Hint for the next ``get_finished`` (see ``TTP2PWorker.begin_step``). Optional."""
        if self.worker is not None:
            self.worker.begin_step(decode_active=decode_active, idle=idle)

    # --- completion (I1): never asserts; every terminal job reaches mark_completed
    def get_finished(self, finished_req_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
        results: List[TransferResult] = list(self._unserved)
        self._unserved.clear()
        if self.worker is not None:
            results.extend(self.worker.get_finished())
        finished_recving: Set[str] = set()
        for result in results:
            job_id = result.job_id
            if job_id in self._reported:
                # A second report for an id already completed would be a KeyError (or a tripped assert) in the scheduler.
                logger.warning('ttkvp2p: job %s reported twice by the worker; the second report is dropped (I1)', job_id)
                continue
            self._reported[job_id] = None
            while len(self._reported) > TOMBSTONE_LIMIT:
                self._reported.popitem(last=False)
            is_load = job_id in self._load_jobs
            if result.success and result.transfer_time is not None and result.transfer_size is not None:
                stats = (self._connector_worker_meta.transfer_stats.load if is_load
                         else self._connector_worker_meta.transfer_stats.store)
                stats.record(result.transfer_size, result.transfer_time)
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
                if not result.success:
                    self._failed_recving.add(req_id)
                    self._connector_worker_meta.failed_loads[job_id] = getattr(result, 'reason', '') or 'failed'
        return set(), finished_recving

    def take_failed_recving(self) -> Set[str]:
        failed, self._failed_recving = self._failed_recving, set()
        return failed

    def build_connector_worker_meta(self) -> Optional[OffloadingWorkerMetadata]:
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = TTWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unserved.clear()
        self._failed_recving.clear()
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = TTWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()


# --------------------------------------------------------------------------------------------------------------------
# The connector
# --------------------------------------------------------------------------------------------------------------------

class TTP2PConnector(OffloadingConnector):
    """Keeps ``SupportsHMA`` (inherited). Under the uni executor both roles live in the EngineCore process and share one runtime."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig):
        # Not super().__init__: it would build the stock scheduler (and its manager) just to have it replaced.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        self._canonical_layout = offloading_config.canonical_layout
        spec = OffloadingSpecFactory.create_spec(offloading_config)
        refusal = topology_refusal(vllm_config)
        if refusal is not None and hasattr(spec, 'refuse'):
            spec.refuse(refusal)          # logs; the manager always misses and no worker is built

        self.connector_scheduler: Optional[TTP2PConnectorScheduler] = None
        self.connector_worker: Optional[TTP2PConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TTP2PConnectorScheduler(spec, vllm_config, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = TTP2PConnectorWorker(spec, vllm_config, kv_cache_config)

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.bind_block_pool(gpu_block_pool)

    def register_kv_caches(self, kv_caches: TTKVRegistry) -> None:         # type: ignore[override]
        if self.connector_worker is not None:
            self.connector_worker.register_kv_caches(kv_caches)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> Optional[str]:
        return None

    def begin_step(self, *, decode_active: bool = True, idle: bool = False) -> None:
        if self.connector_worker is not None:
            self.connector_worker.begin_step(decode_active=decode_active, idle=idle)

    def get_transfer_results(self, finished_req_ids: Set[str]) -> KVConnectorTransferResults:
        res = super().get_transfer_results(finished_req_ids)          # runs get_finished, and with it prepare_store_kv (I9)
        res.failed_recving |= self.connector_worker.take_failed_recving()
        assert res.failed_recving <= res.finished_recving             # V base.py: a failed receive is also a finished one
        return res
