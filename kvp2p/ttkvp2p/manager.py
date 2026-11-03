"""NvmeKVManager: the scheduler-side index of the drive tier (a subclass of vLLM's ``CPUOffloadingManager``, medium STORAGE).

It decides, in RAM and on the ENGINE THREAD only (invariant I7), which keys live in which drive slot, who may store, what is reusable and
what to evict. It never does device or drive I/O. All the lifecycle facts it relies on are in ``runtime`` (the process-wide
``KVP2PRuntime``): the slot allocator with its generations and in-flight counts (I5), the worker's per-job store results, the BAD marks.

What it keeps, per key: the slot (``chunk_id`` of vLLM's LRU ``ChunkStatus``), and in structure-of-arrays form per slot: the entry state
(ALLOCATED = store in flight, WRITTEN, BAD), the tenant tag, created and last-used times and the 4 device digests copied from the store
result. The vLLM LRU policy is reused as it is (``self._policy``): its ``chunks`` dict is the index, its ``evictable_chunks`` the LRU order.

The rules this class owns:

* I6  ``prepare_store`` returns slot ids 1:1 with ``keys_to_store`` in the order it was asked, and asserts it (a pairing that does not
      hold is rolled back, dropped and latches the tier: a digest cannot catch a key/slot swap).
* I7  Only the engine thread mutates anything. "Now" for the TTL is frozen once per ``schedule()`` (it is refreshed by the first
      lookup of a pass and held until ``on_schedule_end``); expired entries and LRU victims go in ``on_schedule_end``, which vLLM runs
      after every lookup and ``prepare_load`` of the pass. An entry that answered HIT cannot vanish before its ``prepare_load``.
* I12 Admission is all or nothing per request boundary, decided once at the request's first store and cached in its ``ReqTag``: the whole
      prompt boundary (every full-attention block plus every sliding group's window tail) must fit the free list, the capacity (with the
      free floor and the standing reserve) and the daily write cap, and the request must clear ``min_store_tokens``, the salt rule and
      the multimodal rule. The slots and the bytes are RESERVED for the request, so later stores of the same prompt cannot be starved
      by other requests; the unused rest is released when the request finishes.
* I13 An entry goes BAD only on the worker's proof (``runtime.mark_bad``); a deadline or stall never gets here. A BAD entry is removed
      when its last load completes.
* TTL 86,400 s since last use; capacity = slots minus the free floor; a standing reserve below it so that a new maximum session always
  fits; a rolling 24 hour write cap charged at admission; ``reset_cache`` = in-RAM invalidate plus an epoch bump, the slots going back
  through TRIM under I5 (never a wipe).
"""

import collections
import logging
from collections import OrderedDict
from collections.abc import Collection
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

from ttkvp2p.runtime import STATE_AVAILABLE, STATE_CLOSING
from ttkvp2p.spec import NvmeLoadStoreSpec, ReqTag

logger = logging.getLogger(__name__)

# entry states (the slot table's ``state`` column; 0 = no entry)
E_NONE, E_ALLOCATED, E_WRITTEN, E_BAD = 0, 1, 2, 3

TOKENS_PER_OBJECT = 128           # one object = one block of one group = 128 tokens (Gemma; the connector forces blocks_per_chunk = 1)
ALLOCATED_REAP_S = 300.0          # a store that has not completed in this long is given up (a stuck store must not defer lookups forever)
EVICT_PER_PASS = 4096             # eviction work per on_schedule_end (the engine thread must stay light)
VERDICT_CACHE_MAX = 8192

# The KV groups of a layout as sliding-window sizes in objects (None = full attention). Gemma 4 26B-A4B: one full-attention group and five
# sliding groups of an 8-block window (cdiv(1023, 128) = 8). Used to size a request's whole boundary (I12); the connector can bind the real
# groups with ``bind_groups``.
LAYOUT_GROUPS: Dict[str, Tuple[Optional[int], ...]] = {'gemma-chipmajor-v1': (None, 8, 8, 8, 8, 8)}


def estimate_objects(prompt_tokens: int, groups: Sequence[Optional[int]], tokens_per_object: int = TOKENS_PER_OBJECT) -> int:
    """An upper bound of the objects one prompt's boundary stores: every full block of a full-attention group, the last ``window`` blocks of
    a sliding group. 262,144 tokens on Gemma: 2,048 + 5 x 8 = 2,088."""
    full = max(0, int(prompt_tokens)) // tokens_per_object
    return sum(full if window is None else min(full, int(window)) for window in groups)


class _RollingBytes:
    """Bytes charged in the last 24 hours, in hourly buckets (the bucket of the current hour and the 24 before it: never less than 24 h)."""

    def __init__(self) -> None:
        self.buckets: Dict[int, int] = {}

    def add(self, now: float, nbytes: int) -> int:
        hour = int(now // 3600)
        self.buckets[hour] = self.buckets.get(hour, 0) + int(nbytes)
        self._prune(hour)
        return hour

    def refund(self, now: float, hour: int, nbytes: int) -> None:
        if hour in self.buckets and hour >= int(now // 3600) - 24:
            self.buckets[hour] = max(0, self.buckets[hour] - int(nbytes))

    def total(self, now: float) -> int:
        hour = int(now // 3600)
        self._prune(hour)
        return sum(self.buckets.values())

    def _prune(self, hour: int) -> None:
        for old in [k for k in self.buckets if k < hour - 24]:
            del self.buckets[old]


class _Reservation:
    __slots__ = ('slots', 'nbytes', 'hour')

    def __init__(self, slots: int, nbytes: int, hour: int):
        self.slots, self.nbytes, self.hour = slots, nbytes, hour


class PairingError(AssertionError):
    """keys_to_store and the slot ids do not pair 1:1 with the manager's own key -> slot map (I6)."""


class NvmeKVManager(CPUOffloadingManager):
    """See the module docstring. ``runtime`` is the shared ``KVP2PRuntime``; the manager may be built before or after the drive is open."""

    def __init__(self, cfg, runtime, *, clock=None, groups: Optional[Sequence[Optional[int]]] = None,
                 tokens_per_object: int = TOKENS_PER_OBJECT):
        super().__init__(num_chunks=max(1, runtime.allocator.num_slots), cache_policy='lru', store_threshold=cfg.store_threshold)
        self.medium = Medium.STORAGE
        self.cfg = cfg
        self.runtime = runtime
        self._clock = clock or runtime.env.monotonic
        self.tokens_per_object = int(tokens_per_object)
        self.groups: Optional[Tuple[Optional[int], ...]] = tuple(groups) if groups is not None else LAYOUT_GROUPS.get(cfg.layout)
        self.epoch = 0
        self._now = self._clock()
        self._now_valid = False
        self._tables_n = -1
        self._state = np.zeros(0, dtype=np.uint8)
        self._tenant = np.zeros(0, dtype=np.uint64)
        self._created = np.zeros(0, dtype=np.float64)
        self._last_used = np.zeros(0, dtype=np.float64)
        self._digest = np.zeros((0, 4), dtype=np.uint64)
        self._seq = np.zeros(0, dtype=np.uint64)                           # allocation order: a job's keys are a set, the LRU order must not be
        self._next_seq = 0
        self._slot_key: List[Optional[bytes]] = []
        self._pending: 'OrderedDict[int, float]' = OrderedDict()           # slot -> time its store was admitted (ALLOCATED entries)
        self._reservations: Dict[str, _Reservation] = {}
        self._reserved_total = 0
        self._verdicts: 'OrderedDict[str, bool]' = OrderedDict()            # for requests that carry no ReqTag
        self._bytes = _RollingBytes()
        self._down = False
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.admitted_objects = 0
        self.stored_objects = 0
        self.dropped: 'collections.Counter[str]' = collections.Counter()
        self.evictions: 'collections.Counter[str]' = collections.Counter()
        self.miss_reasons: 'collections.Counter[str]' = collections.Counter()
        self._sync_tables()
        runtime.attach_manager(self)

    # ------------------------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------------------------

    @property
    def object_bytes(self) -> int:
        store = self.runtime.store
        if store is not None:
            return store.object_bytes
        layout = getattr(self.runtime, 'layout', None)
        return layout.object_bytes if layout is not None else 0

    @property
    def num_slots(self) -> int:
        return self.runtime.allocator.num_slots

    @property
    def floor_slots(self) -> int:
        return -(-self.num_slots * self.cfg.free_floor_pct // 100)

    @property
    def capacity_slots(self) -> int:
        """The most entries the index may hold: every slot minus the free floor."""
        return max(0, self.num_slots - self.floor_slots)

    @property
    def reserve_slots(self) -> int:
        obj = self.object_bytes
        return -(-self.cfg.reserve_bytes // obj) if obj and self.cfg.reserve_bytes else 0

    @property
    def entries(self) -> int:
        return len(self._policy.chunks)

    @property
    def index(self) -> Dict[bytes, int]:
        """key -> slot (a copy; for tests and the state file)."""
        return {key: chunk.chunk_id for key, chunk in self._policy.chunks.items()}

    def entry_state(self, key: bytes) -> str:
        chunk = self._policy.get(key)
        if chunk is None:
            return 'none'
        return ('none', 'allocated', 'written', 'bad')[int(self._state[chunk.chunk_id])]

    def written(self) -> int:
        return int((self._state == E_WRITTEN).sum())

    def _sync_tables(self) -> None:
        n = self.runtime.allocator.num_slots
        if n == self._tables_n:
            return
        if self._policy.chunks:
            raise RuntimeError('the slot table was resized while the index is not empty')
        self._tables_n = n
        self._state = np.zeros(n, dtype=np.uint8)
        self._tenant = np.zeros(n, dtype=np.uint64)
        self._created = np.zeros(n, dtype=np.float64)
        self._last_used = np.zeros(n, dtype=np.float64)
        self._digest = np.zeros((n, 4), dtype=np.uint64)
        self._seq = np.zeros(n, dtype=np.uint64)
        self._slot_key = [None] * n

    def _freeze(self) -> float:
        """The pass's "now": refreshed by the first manager call of a pass, held until ``on_schedule_end``."""
        if not self._now_valid:
            self._now = self._clock()
            self._now_valid = True
            if self.runtime.allocator.num_slots != self._tables_n:
                self._sync_tables()
        return self._now

    def _tag(self, ctx) -> Optional[ReqTag]:
        return ctx.get_state(ReqTag) if ctx is not None else None

    @staticmethod
    def _tenant_value(tag: Optional[ReqTag]) -> int:
        if tag is None or not tag.tenant:
            return 0
        return int.from_bytes(bytes(tag.tenant)[:8], 'big') or 1

    def _miss(self, reason: str) -> LookupResult:
        self.misses += 1
        self.miss_reasons[reason] += 1
        return LookupResult.MISS

    def _drop_reason(self, reason: str) -> PrepareStoreOutput:
        self.dropped[reason] += 1
        return PrepareStoreOutput(keys_to_store=[], store_spec=NvmeLoadStoreSpec([], []), evicted_keys=[])

    # ------------------------------------------------------------------------------------------------------------
    # OffloadingManager
    # ------------------------------------------------------------------------------------------------------------

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        now = self._freeze()
        self.lookups += 1
        runtime = self.runtime
        if runtime.state != STATE_AVAILABLE:
            return self._miss('unavailable')
        tag = req_context.get_state(ReqTag)
        if tag is not None and tag.no_retry:
            return self._miss('no_retry')
        chunk = self._policy.chunks.get(key)
        if chunk is None:
            return self._miss('absent')
        slot = chunk.chunk_id
        state = self._state[slot]
        if state == E_BAD or (runtime.bad and (slot, int(runtime.allocator.gen[slot])) in runtime.bad):
            return self._miss('bad')
        if self.cfg.require_salt:
            if tag is None or not tag.tenant:
                return self._miss('no_salt')
            if int(self._tenant[slot]) != self._tenant_value(tag):
                return self._miss('foreign_tenant')
        if state == E_ALLOCATED:
            if now - self._pending.get(slot, now) > ALLOCATED_REAP_S:
                return self._miss('stuck_store')
            return LookupResult.HIT_PENDING
        if now - self._last_used[slot] >= self.cfg.ttl_s:
            return self._miss('expired')
        self.hits += 1
        return LookupResult.HIT

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        now = self._freeze()
        chunks = self._policy.chunks
        for key in keys:
            chunk = chunks.get(key)
            if chunk is not None and self._state[chunk.chunk_id] == E_WRITTEN:
                self._last_used[chunk.chunk_id] = now
        self._policy.touch(keys, req_context)

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        """Pin the entries and copy their slot, generation and expected digests into the spec (I4). The scheduler only asks after a HIT in the
        same pass, and eviction runs in ``on_schedule_end`` (I7), so every key is there; if one is not, the spec is POISONED so that the worker
        fails the job and the request recomputes (never a crash of the engine thread, never a wrong slot)."""
        self._freeze()
        allocator = self.runtime.allocator
        slots: List[int] = []
        gens: List[int] = []
        for key in keys:
            chunk = self._policy.chunks.get(key)
            if chunk is None or not chunk.is_ready or self._state[chunk.chunk_id] != E_WRITTEN:
                logger.error('ttkvp2p: prepare_load of a key that is not WRITTEN (I7 violated); the load will be failed')
                self.runtime.count('mgr_prepare_load_missing')
                n = len(keys)
                return NvmeLoadStoreSpec([-1] * n, [0] * n, np.zeros((n, 4), dtype=np.uint64), poison='manager: a key vanished before prepare_load')
            slots.append(chunk.chunk_id)
        for key, slot in zip(keys, slots):
            chunk = self._policy.chunks[key]
            if chunk.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
            chunk.ref_cnt += 1
            gens.append(int(allocator.gen[slot]))
        index = np.asarray(slots, dtype=np.int64)
        expected = self._digest[index].copy() if len(slots) else np.zeros((0, 4), dtype=np.uint64)
        return NvmeLoadStoreSpec(slots, gens, expected)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        now = self._clock()
        for key in keys:
            chunk = self._policy.chunks.get(key)
            if chunk is None:
                continue                                  # a reset or a wipe in between
            if chunk.ref_cnt <= 0:
                logger.error('ttkvp2p: complete_load of a key with ref_cnt %d', chunk.ref_cnt)
                continue
            chunk.ref_cnt -= 1
            if chunk.ref_cnt > 0:
                continue
            slot = chunk.chunk_id
            if self._state[slot] == E_BAD or (slot, int(self.runtime.allocator.gen[slot])) in self.runtime.bad:
                self._remove(key, 'bad')                  # I13: removed at ref_cnt 0
                continue
            self._last_used[slot] = now
            self._policy.mark_evictable(key)

    # ------------------------------------------------------------------------------------------------------------
    # stores: admission (I12), allocation (I6), completion
    # ------------------------------------------------------------------------------------------------------------

    def _verdict(self, ctx, tag: Optional[ReqTag], todo: List[bytes]) -> Tuple[bool, str, int]:
        """(admit?, reason, whole-boundary objects to reserve). Called once per request; the answer is cached in the ReqTag."""
        cfg = self.cfg
        prompt_tokens = getattr(tag, 'num_prompt_tokens', None) if tag is not None else None
        if prompt_tokens is not None and self.groups is not None:
            if prompt_tokens < cfg.min_store_tokens:
                return False, 'short_prompt', 0
            total = estimate_objects(prompt_tokens, self.groups, self.tokens_per_object)
            total = max(total, len(todo))
            known = True
        else:
            # the request's size is not known to the manager (no ReqTag.num_prompt_tokens or no group map): judge this call's keys alone, and
            # take the tokens the call covers as the size (a full-attention key is one object = tokens_per_object tokens)
            per_group = collections.Counter(get_offload_group_idx(k) for k in todo)
            if max(per_group.values()) * self.tokens_per_object < cfg.min_store_tokens:
                return False, 'short_prompt', 0
            total = len(todo)
            known = False
        obj = self.object_bytes
        free_for_new = self.runtime.allocator.n_free - self._reserved_total
        if total > free_for_new:
            return False, 'no_slot', 0
        if self.entries + self._reserved_total + total > self.capacity_slots:
            return False, 'capacity', 0
        now = self._clock()
        if self._bytes.total(now) + total * obj > cfg.daily_write_cap_bytes:
            return False, 'cap', 0
        return True, 'admitted' if known else 'admitted_per_call', total

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> Optional[PrepareStoreOutput]:
        """Allocate slots for the keys the tier does not hold yet, in the order asked (I6). Never returns None (that would be a warning and a
        retry on every step); a refusal is an EMPTY ``keys_to_store``, which the scheduler treats as a drop."""
        self._freeze()
        runtime = self.runtime
        if runtime.state != STATE_AVAILABLE:
            return self._drop_reason('unavailable')
        tag = self._tag(req_context)
        if tag is not None and tag.has_mm and not self.cfg.persist_multimodal:
            return self._drop_reason('multimodal')
        if self.cfg.require_salt and (tag is None or not tag.tenant):
            return self._drop_reason('no_salt')
        keys = list(keys)
        if self.counts is not None:                        # store_threshold >= 2: store a key only from its Nth offer (as the CPU manager)
            self._record_accesses(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
        chunks = self._policy.chunks
        todo = list(dict.fromkeys(k for k in keys if k not in chunks))
        if not todo:
            return PrepareStoreOutput(keys_to_store=[], store_spec=NvmeLoadStoreSpec([], []), evicted_keys=[])
        req_id = req_context.req_id
        cached = tag.admitted if tag is not None else self._verdicts.get(req_id)
        reservation = self._reservations.get(req_id)
        if cached is False:
            return self._drop_reason('denied')
        obj = self.object_bytes
        if cached is None:
            admit, reason, total = self._verdict(req_context, tag, todo)
            self._remember_verdict(req_id, tag, admit)
            if not admit:
                return self._drop_reason(reason)
            nbytes = total * obj
            hour = self._bytes.add(self._clock(), nbytes)
            reservation = _Reservation(total, nbytes, hour)
            self._reservations[req_id] = reservation
            self._reserved_total += total
            self.admitted_objects += total
        need = len(todo)
        own = reservation.slots if reservation is not None else 0
        # slots: the free list minus what OTHER requests have reserved
        if need > runtime.allocator.n_free - (self._reserved_total - own):
            return self._drop_reason('no_slot')
        extra = max(0, need - own)
        if extra and (self.entries + self._reserved_total + extra > self.capacity_slots):
            return self._drop_reason('capacity')
        if extra and self._bytes.total(self._clock()) + extra * obj > self.cfg.daily_write_cap_bytes:
            return self._drop_reason('cap')
        slots = runtime.allocator.alloc(need)
        if slots is None or len(slots) != need or len(set(slots)) != need:
            if slots:
                runtime.allocator.release_many(slots)
            return self._drop_reason('no_slot')
        if reservation is not None:
            used = min(own, need)
            reservation.slots -= used
            reservation.nbytes -= used * obj
            self._reserved_total -= used
        if extra:
            self._bytes.add(self._clock(), extra * obj)
        now = self._now
        gens = [int(runtime.allocator.gen[s]) for s in slots]
        tenant = self._tenant_value(tag)
        for key, slot in zip(todo, slots):
            self._policy.insert(key, ChunkStatus(slot))        # ref_cnt -1: not ready, not evictable
            self._state[slot] = E_ALLOCATED
            self._tenant[slot] = tenant
            self._created[slot] = now
            self._last_used[slot] = now
            self._slot_key[slot] = key
            self._pending[slot] = now
            self._seq[slot] = self._next_seq
            self._next_seq += 1
        output = PrepareStoreOutput(keys_to_store=todo, store_spec=NvmeLoadStoreSpec(list(slots), gens), evicted_keys=[])
        try:
            self._check_pairing(output)
        except PairingError as error:
            logger.error('ttkvp2p: %s; the store is rolled back and the tier latched', error)
            self._rollback(todo, slots)
            runtime.latch('internal error: store pairing (%s)' % (error,), 'op_throw')
            return self._drop_reason('pairing')
        return output

    def _remember_verdict(self, req_id: str, tag: Optional[ReqTag], admit: bool) -> None:
        if tag is not None:
            tag.admitted = admit
            return
        self._verdicts[req_id] = admit
        while len(self._verdicts) > VERDICT_CACHE_MAX:
            self._verdicts.popitem(last=False)

    def _check_pairing(self, output: PrepareStoreOutput) -> None:
        """I6: slot i of the spec is the slot the index gives key i, and carries the generation the allocator holds. A digest cannot see a swap."""
        spec = output.store_spec
        keys = output.keys_to_store
        if len(keys) != len(spec.slot_ids) or len(keys) != len(spec.generations):
            raise PairingError('%d keys, %d slots, %d generations' % (len(keys), len(spec.slot_ids), len(spec.generations)))
        chunks = self._policy.chunks
        gen = self.runtime.allocator.gen
        for position, (key, slot, generation) in enumerate(zip(keys, spec.slot_ids, spec.generations)):
            chunk = chunks.get(key)
            if chunk is None or chunk.chunk_id != slot or int(gen[slot]) != generation or self._slot_key[slot] != key:
                raise PairingError('position %d: key and slot %d do not pair' % (position, slot))

    def _rollback(self, keys: List[bytes], slots: List[int]) -> None:
        for key in keys:
            chunk = self._policy.chunks.get(key)
            if chunk is not None:
                self._policy.remove(key)
        for slot in slots:
            self._clear_slot(slot)
        self.runtime.allocator.release_many(slots)

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True, result: Any = None) -> None:
        """A store job reported. The digests come from the job's ``StoreResult`` (``result`` when the caller has it, else the runtime's results,
        found by an index on (slot, generation): no scan), paired with the entry through its slot and generation, so the order of ``keys`` (a set,
        for vLLM) does not matter. ``success=False``, a failed result, a result that was pruned or one whose generation moved on drops the entry:
        its slot goes back through TRIM. Nothing here loads a half-written object."""
        now = self._clock()
        allocator = self.runtime.allocator
        local = None
        if result is not None and success:
            slots_ = list(getattr(result, 'slot_ids', ()) or ())
            gens_ = list(getattr(result, 'generations', ()) or ())
            local = {(int(s), int(g)): i for i, (s, g) in enumerate(zip(slots_, gens_))}
        pending = []
        for key in keys:
            chunk = self._policy.chunks.get(key)
            if chunk is not None and not chunk.is_ready:       # (a reset in between, or a repeated report, leaves nothing to complete)
                pending.append((int(self._seq[chunk.chunk_id]), key, chunk))
        pending.sort(key=lambda item: item[0])                 # the job's keys are a set: complete them in the order they were allocated
        for _seq, key, chunk in pending:
            slot = chunk.chunk_id
            digest = None
            if success:
                generation = int(allocator.gen[slot])
                if local is not None:
                    found = (result, local[(slot, generation)]) if (slot, generation) in local else None
                else:
                    found = self.runtime.store_results.lookup(slot, generation)
                if found is not None:
                    res, position = found
                    digests = getattr(res, 'digests', None)
                    if getattr(res, 'success', False) and digests is not None and position < len(digests):
                        digest = np.asarray(digests[position], dtype=np.uint64)
            if digest is None:
                self._remove(key, 'store_failed')
                continue
            self._digest[slot] = digest
            self._state[slot] = E_WRITTEN
            self._last_used[slot] = self._created[slot] = now
            self._pending.pop(slot, None)
            chunk.ref_cnt = 0
            self._policy.mark_evictable(key)
            self.stored_objects += 1

    # ------------------------------------------------------------------------------------------------------------
    # removal
    # ------------------------------------------------------------------------------------------------------------

    def _clear_slot(self, slot: int) -> None:
        self._state[slot] = E_NONE
        self._slot_key[slot] = None
        self._pending.pop(slot, None)
        self.runtime.bad.discard((slot, int(self.runtime.allocator.gen[slot])))

    def _remove(self, key: bytes, reason: str) -> None:
        chunk = self._policy.chunks.get(key)
        if chunk is None:
            return
        slot = chunk.chunk_id
        self._policy.remove(key)
        self._clear_slot(slot)
        self.runtime.allocator.release(slot)
        self.evictions[reason] += 1

    def on_slot_bad(self, slot: int, generation: int) -> None:
        """The worker proved (slot, generation) bad. Mark it; it is removed when its last load completes (or now, if nobody holds it)."""
        if not 0 <= slot < self._tables_n or int(self.runtime.allocator.gen[slot]) != generation:
            return
        key = self._slot_key[slot]
        chunk = self._policy.chunks.get(key) if key is not None else None
        if chunk is None:
            return
        self._state[slot] = E_BAD
        if chunk.is_ready and chunk.ref_cnt == 0:
            self._remove(key, 'bad')

    def on_request_finished(self, req_context: ReqContext) -> None:
        req_id = req_context.req_id
        reservation = self._reservations.pop(req_id, None)
        self._verdicts.pop(req_id, None)
        if reservation is not None and (reservation.slots or reservation.nbytes):
            self._reserved_total -= reservation.slots
            self._bytes.refund(self._clock(), reservation.hour, reservation.nbytes)

    # ------------------------------------------------------------------------------------------------------------
    # the pass boundary: eviction, the runtime's pump
    # ------------------------------------------------------------------------------------------------------------

    def on_schedule_end(self, context) -> None:
        now = self._freeze()
        runtime = self.runtime
        try:
            runtime.heartbeat()
            runtime.service()
            if runtime.state != STATE_CLOSING:
                self._maintain(now)
        except Exception:
            logger.exception('ttkvp2p: maintenance at the end of a schedule failed')
        finally:
            self._now_valid = False

    def _maintain(self, now: float) -> None:
        budget = EVICT_PER_PASS
        # a store that never completed (its job vanished): give it up, so lookups stop deferring on it
        while self._pending and budget:
            slot, since = next(iter(self._pending.items()))
            if now - since <= ALLOCATED_REAP_S:
                break
            key = self._slot_key[slot]
            self._pending.pop(slot, None)
            if key is not None:
                self._remove(key, 'stuck_store')
            budget -= 1
        evictable = self._policy.evictable_chunks
        # TTL: entries not used for ttl_s go, oldest first (the LRU order is the last-used order)
        ttl = self.cfg.ttl_s
        while evictable and budget:
            key = next(iter(evictable))
            slot = self._policy.chunks[key].chunk_id
            if now - self._last_used[slot] < ttl:
                break
            del evictable[key]
            self._remove(key, 'ttl')
            budget -= 1
        # capacity: keep the standing reserve free below the floor, so a new maximum session always fits
        target = self.capacity_slots - self.reserve_slots
        while evictable and budget and self.entries > target:
            key = next(iter(evictable))
            del evictable[key]
            self._remove(key, 'lru')
            budget -= 1

    def reset_cache(self) -> None:
        """vLLM's pause and sleep paths (``reset_connector=True``): invalidate the index in RAM, bump the epoch, send the slots back through
        TRIM under I5. Never a wipe of the drive."""
        self.invalidate_all('reset')
        self.epoch += 1

    def invalidate_all(self, kind: str = 'reset') -> None:
        slots = [chunk.chunk_id for chunk in self._policy.chunks.values()]
        self._policy.clear()
        if slots:
            self._state[np.asarray(slots, dtype=np.int64)] = E_NONE
        for slot in slots:
            self._slot_key[slot] = None
        self._pending.clear()
        self._reservations.clear()
        self._reserved_total = 0
        self._verdicts.clear()
        if self.counts is not None:
            self.counts.clear()
        if self.runtime.bad:
            self.runtime.bad.clear()
        self.runtime.allocator.release_many(slots)
        self.evictions[kind] += len(slots)

    def shutdown(self) -> None:
        if self._down:
            return
        self._down = True
        self.runtime.detach_manager(self)
        self.runtime.release('scheduler')

    def get_stats(self):
        return None            # the tier's numbers go to the state file; vLLM's own offload families come from the worker's transfer stats

    # ------------------------------------------------------------------------------------------------------------
    # state file
    # ------------------------------------------------------------------------------------------------------------

    def bind_groups(self, groups: Sequence[Optional[int]], tokens_per_object: Optional[int] = None) -> None:
        """Tell the manager the real KV groups (sliding window in objects, None = full attention), for the whole-boundary admission size."""
        self.groups = tuple(groups)
        if tokens_per_object:
            self.tokens_per_object = int(tokens_per_object)

    def snapshot(self) -> Dict[str, Any]:
        return {
            'lookups': self.lookups, 'hits': self.hits, 'misses': self.misses, 'miss_reasons': dict(self.miss_reasons),
            'entries': self.entries, 'capacity_slots': self.capacity_slots, 'reserve_slots': self.reserve_slots,
            'admitted_objects': self.admitted_objects, 'stored_objects': self.stored_objects,
            'dropped': dict(self.dropped), 'evictions': dict(self.evictions), 'epoch': self.epoch,
            'bytes_written_today': self._bytes.total(self._clock()), 'reserved_slots': self._reserved_total,
        }
