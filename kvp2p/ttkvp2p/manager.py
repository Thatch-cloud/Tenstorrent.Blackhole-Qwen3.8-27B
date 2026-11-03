"""NvmeKVManager: the scheduler-side index of the drive tier (an ``OffloadingManager`` for the storage medium).

It decides, in RAM and on the ENGINE THREAD only (invariant I7), which keys live in which drive slot, who may store, what is reusable and
what to evict. It never does device or drive I/O. All the lifecycle facts it relies on are in ``runtime`` (the process-wide
``KVP2PRuntime``): the slot allocator with its generations and in-flight counts (I5) and the BAD marks.

Its OWN index and LRU. The manager subclasses vLLM's public ``OffloadingManager`` and nothing else of vLLM's: not ``CPUOffloadingManager``,
not its ``CachePolicy`` classes, not ``ChunkStatus``. Those are internals that moved between vLLM releases (0.31 stops evicting through
them without an error, 0.33 crashes), and the tier must keep evicting whatever vLLM does to them. What the manager keeps instead:

* ``_index``   key -> slot (one dict; the key objects are shared with the two structures below);
* ``_lru``     an ``OrderedDict`` of the WRITTEN keys, oldest first: touch, insert and evict are O(1). A key a load pins stays in it (moved to
               the back when the load completes) and the eviction passes it over: pinning and unpinning never insert or delete, so a
               read-heavy engine does not make CPython rebuild a 340,000-entry hash table every few thousand loads;
* per slot, in structure-of-arrays form: the entry state (ALLOCATED = store in flight, WRITTEN, BAD), the load pin count, the tenant tag,
  created and last-used times, the 4 device digests, the batch of the ``prepare_store`` call that allocated it, and its key.

Every call costs O(keys in the call). Nothing scans the index or the slot table (a 1 TB drive holds about 340,000 entries): the entry
counts are kept incrementally, the slot allocator's too, and evictions are collected and released in ONE batch per pass.

The rules this class owns:

* I6  ``prepare_store`` returns slot ids 1:1 with ``keys_to_store`` in the order it was asked, and asserts it (a pairing that does not
      hold is rolled back, dropped and latches the tier: a digest cannot catch a key/slot swap).
* A1  ``complete_store(keys, ctx, success, digests)``: ``keys`` in ``prepare_store`` order, ``digests`` ``[len(keys), chips]`` uint64 in
      that same order, attached BY POSITION. The manager never reads the runtime's store results and never pairs a digest by slot or
      generation. What it does check is that a key's entry is still the one this job's ``prepare_store`` allocated (a batch number kept
      per slot): a completion that arrives after a wipe or the stuck-store reap must not mark a NEW entry of the same key WRITTEN.
* I7  Only the engine thread mutates anything. "Now" for the TTL is frozen once per ``schedule()`` (it is refreshed by the first
      lookup of a pass and held until ``on_schedule_end``); expired entries and LRU victims go in ``on_schedule_end``, which vLLM runs
      after every lookup and ``prepare_load`` of the pass. An entry that answered HIT cannot vanish before its ``prepare_load``.
* I12 Admission is all or nothing per request boundary, decided once at the request's first store and cached in its ``ReqTag``: the whole
      prompt boundary (every full-attention block plus every sliding group's window tail) must fit the free list, the capacity (with the
      free floor and the standing reserve) and the daily write cap, and the request must clear ``min_store_tokens``, the salt rule and
      the multimodal rule. The slots and the bytes are RESERVED for the request, so later stores of the same prompt cannot be starved
      by other requests; the unused rest is released when the request finishes. The size comes from ``ReqTag.num_prompt_tokens`` (set by
      the connector's scheduler role) and the group map (bound by it from the real KV groups). A request whose size is UNKNOWN on a
      layout with a group map is refused (`size_unknown`), never judged chunk by chunk: a chunk denied mid-prompt leaves a hole that
      vLLM never revisits, and everything after it is unreachable.
* I13 An entry goes BAD only on the worker's proof (``runtime.mark_bad``); a deadline or stall never gets here. A BAD entry is removed
      when its last load completes.
* TTL 86,400 s since last use; capacity = slots minus the free floor; a standing reserve below it so that a new maximum session always
  fits (eviction keeps ``entries + reservations + reserve <= capacity``); a rolling 24 hour write cap charged at admission;
  ``reset_cache`` = in-RAM invalidate plus an epoch bump, the slots going back through TRIM under I5 (never a wipe).
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
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)

from ttkvp2p.runtime import STATE_AVAILABLE, STATE_CLOSING
from ttkvp2p.spec import NvmeLoadStoreSpec, ReqTag

logger = logging.getLogger(__name__)

# entry states (the slot table's ``state`` column; 0 = no entry)
E_NONE, E_ALLOCATED, E_WRITTEN, E_BAD = 0, 1, 2, 3

TOKENS_PER_OBJECT = 128           # one object = one block of one group = 128 tokens (Gemma; the connector forces blocks_per_chunk = 1)
ALLOCATED_REAP_S = 300.0          # a store that has not completed in this long is given up (a stuck store must not defer lookups forever)
EVICT_PER_PASS = 4096             # eviction work per on_schedule_end (the engine thread must stay light)
VERDICT_CACHE_MAX = 8192
ACCESS_TRACKER_MAX = 64_000       # store_threshold >= 2: how many keys' offers are counted at once

# The KV groups of a layout as sliding-window sizes in objects (None = full attention). Gemma 4 26B-A4B: one full-attention group and five
# sliding groups of an 8-block window (cdiv(1023, 128) = 8). Used to size a request's whole boundary (I12) when nothing binds the real groups;
# the connector's scheduler role binds the real ones (``bind_groups``) from the offloading config.
LAYOUT_GROUPS: Dict[str, Tuple[Optional[int], ...]] = {'gemma-chipmajor-v1': (None, 8, 8, 8, 8, 8)}


def estimate_objects(prompt_tokens: int, groups: Sequence[Optional[int]], tokens_per_object: int = TOKENS_PER_OBJECT, hit_tokens: int = 0) -> int:
    """An upper bound of the objects one prompt's boundary stores: every full block of a full-attention group, the last ``window`` blocks of
    a sliding group. 262,144 tokens on Gemma: 2,048 + 5 x 8 = 2,088. ``hit_tokens`` (a later turn of a conversation: the prefix already on the
    device or in the tier) takes its blocks off the full-attention groups only; a sliding group still stores the window at the NEW end."""
    full = max(0, int(prompt_tokens)) // tokens_per_object
    new_full = max(0, full - max(0, int(hit_tokens)) // tokens_per_object)
    return sum(new_full if window is None else min(full, int(window)) for window in groups)


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


class _StoreLedger:
    """The ``prepare_store`` calls of ONE request whose jobs have not reported yet, oldest first: ``(batch number, keys in prepare order)``.

    It lives in the request's ``ReqContext`` (``set_state``), the object vLLM hands to ``complete_store`` too, so it outlives
    ``on_request_finished``: a completion that arrives after the request finished still finds it. ``take`` names the batch a completion
    belongs to, which is how a late completion is told from the entries a LATER ``prepare_store`` made for the same keys (after a wipe or
    the reap of a stuck store). Jobs of one request report in the order they were issued (the worker serves them first in first out), so
    the oldest ledger entry with these keys is the one reporting."""

    __slots__ = ('batches',)

    def __init__(self) -> None:
        self.batches: 'collections.deque[Tuple[int, Tuple[bytes, ...]]]' = collections.deque()

    def add(self, batch: int, keys: Sequence[bytes]) -> None:
        self.batches.append((batch, tuple(keys)))

    def discard(self, batch: int) -> None:
        for index, (number, _keys) in enumerate(self.batches):
            if number == batch:
                del self.batches[index]
                return

    def take(self, keys: Sequence[bytes]) -> Optional[int]:
        """Remove and return the batch number of the oldest entry holding exactly these keys (in this order, or as a set), or None."""
        listed = tuple(keys)
        for index, (number, held) in enumerate(self.batches):
            if len(held) == len(listed) and (held == listed or set(held) == set(listed)):
                del self.batches[index]
                return number
        return None


class PairingError(AssertionError):
    """keys_to_store and the slot ids do not pair 1:1 with the manager's own key -> slot map (I6)."""


class NvmeKVManager(OffloadingManager):
    """See the module docstring. ``runtime`` is the shared ``KVP2PRuntime``; the manager may be built before or after the drive is open."""

    def __init__(self, cfg, runtime, *, clock=None, groups: Optional[Sequence[Optional[int]]] = None,
                 tokens_per_object: int = TOKENS_PER_OBJECT):
        self.medium = Medium.STORAGE
        self.cfg = cfg
        self.runtime = runtime
        self.store_threshold = cfg.store_threshold
        self.max_tracker_size = ACCESS_TRACKER_MAX
        # store_threshold >= 2: a key is stored only from its Nth offer (the same rule as vLLM's CPU manager, kept here as our own code)
        self.counts: 'Optional[OrderedDict[bytes, int]]' = OrderedDict() if self.store_threshold >= 2 else None
        self._clock = clock or runtime.env.monotonic
        self.tokens_per_object = int(tokens_per_object)
        self.groups: Optional[Tuple[Optional[int], ...]] = tuple(groups) if groups is not None else LAYOUT_GROUPS.get(cfg.layout)
        self.epoch = 0
        self._now = self._clock()
        self._now_valid = False
        # the index and the LRU (our own: see the module docstring)
        self._index: Dict[bytes, int] = {}
        self._lru: 'OrderedDict[bytes, None]' = OrderedDict()
        self._n_state = [0, 0, 0, 0]                                       # entries per state (index 0 unused): O(1) counts
        self._tables_n = -1
        self._chips = 4
        self._state = np.zeros(0, dtype=np.uint8)
        self._ref = np.zeros(0, dtype=np.int32)                            # loads pinning the entry
        self._tenant = np.zeros(0, dtype=np.uint64)
        self._created = np.zeros(0, dtype=np.float64)
        self._last_used = np.zeros(0, dtype=np.float64)
        self._digest = np.zeros((0, 4), dtype=np.uint64)
        self._batch = np.zeros(0, dtype=np.uint64)                         # the prepare_store call that allocated the entry (A1: stale completions)
        self._next_batch = 1
        self._slot_key = np.empty(0, dtype=object)
        self._pending: 'OrderedDict[int, float]' = OrderedDict()           # slot -> time its store was admitted (ALLOCATED entries)
        self._reservations: Dict[str, _Reservation] = {}
        self._reserved_total = 0
        self._verdicts: 'OrderedDict[str, bool]' = OrderedDict()            # for requests that carry no ReqTag
        self._bytes = _RollingBytes()
        self._down = False
        self._warned_size_unknown = False
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.admitted_objects = 0
        self.stored_objects = 0
        self.admission_size_unknown = 0        # requests whose prompt size the manager could not learn (refused, or judged per call without a group map)
        self.admission_overrun = 0             # a request that stored more than its reservation (the estimate was short)
        self.store_completion_stale = 0        # completions for entries a later prepare_store replaced (a wipe or the reap in between)
        self.store_completion_invalid = 0      # a success without usable digests: treated as a failed store
        self.store_completion_unmatched = 0    # completions whose request context holds no matching prepare_store record (judged by majority)
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
        return len(self._index)

    @property
    def index(self) -> Dict[bytes, int]:
        """key -> slot (a COPY, O(entries): for tests and diagnostics, never for a per-call path)."""
        return dict(self._index)

    def entry_state(self, key: bytes) -> str:
        slot = self._index.get(key)
        if slot is None:
            return 'none'
        return ('none', 'allocated', 'written', 'bad')[int(self._state[slot])]

    def written(self) -> int:
        return self._n_state[E_WRITTEN]

    def _sync_tables(self) -> None:
        n = self.runtime.allocator.num_slots
        if n == self._tables_n:
            return
        if self._index:
            raise RuntimeError('the slot table was resized while the index is not empty')
        self._tables_n = n
        layout = getattr(self.runtime, 'layout', None)
        self._chips = int(getattr(layout, 'chips', 4) or 4)
        self._state = np.zeros(n, dtype=np.uint8)
        self._ref = np.zeros(n, dtype=np.int32)
        self._tenant = np.zeros(n, dtype=np.uint64)
        self._created = np.zeros(n, dtype=np.float64)
        self._last_used = np.zeros(n, dtype=np.float64)
        self._digest = np.zeros((n, self._chips), dtype=np.uint64)
        self._batch = np.zeros(n, dtype=np.uint64)
        self._slot_key = np.empty(n, dtype=object)
        self._n_state = [0, 0, 0, 0]
        self._lru.clear()

    def _freeze(self) -> float:
        """The pass's "now" for the TTL: refreshed by the first lookup or prepare_load of a pass (or by ``on_schedule_end``), held until
        ``on_schedule_end`` ends the pass. Only those calls arm it: calls that vLLM makes AFTER ``on_schedule_end`` in the same pass
        (``prepare_store``, ``touch``, ``on_request_finished``, the completions) must not, or the next pass would start from a stale
        time (after an idle engine, by the length of the idle)."""
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

    def _set_state(self, slot: int, new: int) -> None:
        old = int(self._state[slot])
        if old != new:
            self._n_state[old] -= 1
            self._n_state[new] += 1
            self._state[slot] = new

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
        slot = self._index.get(key)
        if slot is None:
            return self._miss('absent')
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
        """Mark the keys recently used. vLLM calls this AFTER ``on_schedule_end`` of the pass, so it never arms the frozen clock: it stamps
        with the pass's "now" while one is held and with the live clock otherwise."""
        now = self._now if self._now_valid else self._clock()
        index, lru = self._index, self._lru
        listed = list(keys)
        found = [slot for slot in map(index.get, listed) if slot is not None]
        if found:
            slots = np.asarray(found, dtype=np.int64)
            self._last_used[slots[self._state[slots] == E_WRITTEN]] = now
        for key in reversed(listed):                      # the first key ends up the most recent: a prefix is evicted after its tail
            if key in lru:
                lru.move_to_end(key)

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        """Pin the entries and copy their slot, generation and expected digests into the spec (I4). The scheduler only asks after a HIT in the
        same pass, and eviction runs in ``on_schedule_end`` (I7), so every key is there; if one is not, the spec is POISONED so that the worker
        fails the job and the request recomputes (never a crash of the engine thread, never a wrong slot)."""
        self._freeze()
        keys = list(keys)
        n = len(keys)
        slots = list(map(self._index.get, keys))
        slot_array = np.asarray([-1 if slot is None else slot for slot in slots], dtype=np.int64)
        if n and (slot_array.min() < 0 or bool((self._state[slot_array] != E_WRITTEN).any())):
            logger.error('ttkvp2p: prepare_load of a key that is not WRITTEN (I7 violated); the load will be failed')
            self.runtime.count('mgr_prepare_load_missing')
            return NvmeLoadStoreSpec([-1] * n, [0] * n, np.zeros((n, self._chips), dtype=np.uint64), poison='manager: a key vanished before prepare_load')
        ref = self._ref                                   # a pinned entry stays in the LRU; the eviction passes it over until its loads complete
        if n and np.unique(slot_array).size == n:
            ref[slot_array] += 1
        else:
            for slot in slots:
                ref[slot] += 1
        gens = self.runtime.allocator.gen[slot_array].tolist() if n else []
        expected = self._digest[slot_array].copy() if n else np.zeros((0, self._chips), dtype=np.uint64)
        return NvmeLoadStoreSpec(slots, gens, expected)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        now = self._clock()
        index, ref, state = self._index, self._ref, self._state
        runtime = self.runtime
        present: List[bytes] = []
        slots: List[int] = []
        for key in keys:
            slot = index.get(key)
            if slot is not None:                          # (a reset or a wipe in between leaves nothing to complete)
                present.append(key)
                slots.append(slot)
        if not slots:
            return
        slot_array = np.asarray(slots, dtype=np.int64)
        if np.unique(slot_array).size != len(slots) or bool((ref[slot_array] <= 0).any()):
            self._complete_load_slowly(present, slots, now)
            return
        ref[slot_array] -= 1
        free = ref[slot_array] == 0                       # the last load of the entry
        if not bool(free.any()):
            return
        gen = runtime.allocator.gen
        bad_marks = runtime.bad
        is_bad = state[slot_array] == E_BAD
        bad_keys: List[bytes] = []
        lru = self._lru
        keep: List[int] = []
        for position in np.nonzero(free)[0].tolist():
            slot = slots[position]
            if is_bad[position] or (bad_marks and (slot, int(gen[slot])) in bad_marks):
                bad_keys.append(present[position])        # I13: removed at ref_cnt 0
            else:
                keep.append(slot)
                if present[position] in lru:
                    lru.move_to_end(present[position])    # evictable again, the most recently used
        if keep:
            self._last_used[np.asarray(keep, dtype=np.int64)] = now
        if bad_keys:
            self._remove_many(bad_keys, 'bad')

    def _complete_load_slowly(self, present: List[bytes], slots: List[int], now: float) -> None:
        """The unusual shapes (a key twice in one call, a key with no load pinning it): one key at a time, as the contract reads."""
        ref, state, runtime = self._ref, self._state, self.runtime
        bad_keys: List[bytes] = []
        for key, slot in zip(present, slots):
            if ref[slot] <= 0:
                logger.error('ttkvp2p: complete_load of a key with ref_cnt %d', ref[slot])
                continue
            ref[slot] -= 1
            if ref[slot] > 0:
                continue
            if state[slot] == E_BAD or (runtime.bad and (slot, int(runtime.allocator.gen[slot])) in runtime.bad):
                bad_keys.append(key)
                continue
            self._last_used[slot] = now
            if key in self._lru:
                self._lru.move_to_end(key)
        if bad_keys:
            self._remove_many(bad_keys, 'bad')

    # ------------------------------------------------------------------------------------------------------------
    # stores: admission (I12), allocation (I6), completion (A1)
    # ------------------------------------------------------------------------------------------------------------

    def _verdict(self, tag: Optional[ReqTag], todo: List[bytes]) -> Tuple[bool, str, int]:
        """(admit?, reason, whole-boundary objects to reserve). Called once per request; the answer is cached in the ReqTag."""
        cfg = self.cfg
        prompt_tokens = getattr(tag, 'num_prompt_tokens', None) if tag is not None else None
        if self.groups is not None:
            if prompt_tokens is None:
                # The tier cannot size this request's boundary, so it cannot promise all-or-nothing: refuse it whole (a chunk denied later
                # would leave a hole vLLM never revisits). The connector sets the size on every request; this counter tells if it does not.
                self.admission_size_unknown += 1
                if not self._warned_size_unknown:
                    self._warned_size_unknown = True
                    logger.warning('ttkvp2p: a request reached the tier without its prompt size (ReqTag.num_prompt_tokens); requests of unknown '
                                   'size are refused whole (counted as admission_size_unknown)')
                return False, 'size_unknown', 0
            if prompt_tokens < cfg.min_store_tokens:
                return False, 'short_prompt', 0
            total = estimate_objects(prompt_tokens, self.groups, self.tokens_per_object, getattr(tag, 'hit_tokens', 0) or 0)
            total = max(total, len(todo))
            known = True
        else:
            # no group map for this layout (a test layout): judge this call's keys alone, and take the tokens the call covers as the size
            # (a full-attention key is one object = tokens_per_object tokens). Counted, because it is not all-or-nothing.
            self.admission_size_unknown += 1
            per_group = collections.Counter(get_offload_group_idx(k) for k in todo)
            if max(per_group.values()) * self.tokens_per_object < cfg.min_store_tokens:
                return False, 'short_prompt', 0
            total = len(todo)
            known = False
        obj = self.object_bytes
        free_for_new = self.runtime.allocator.n_free - self._reserved_total
        if total > free_for_new:
            # the open-time discard of the data region is still running: its slots become allocatable as their runs complete
            return False, ('warming' if getattr(self.runtime, 'warming', False) else 'no_slot'), 0
        if self.entries + self._reserved_total + total > self.capacity_slots:
            return False, 'capacity', 0
        now = self._clock()
        if self._bytes.total(now) + total * obj > cfg.daily_write_cap_bytes:
            return False, 'cap', 0
        return True, 'admitted' if known else 'admitted_per_call', total

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> Optional[PrepareStoreOutput]:
        """Allocate slots for the keys the tier does not hold yet, in the order asked (I6). Never returns None (that would be a warning and a
        retry on every step); a refusal is an EMPTY ``keys_to_store``, which the scheduler treats as a drop."""
        if self.runtime.allocator.num_slots != self._tables_n:
            self._sync_tables()
        runtime = self.runtime
        if runtime.state != STATE_AVAILABLE:
            return self._drop_reason('unavailable')
        tag = self._tag(req_context)
        if tag is not None and tag.has_mm and not self.cfg.persist_multimodal:
            return self._drop_reason('multimodal')
        if self.cfg.require_salt and (tag is None or not tag.tenant):
            return self._drop_reason('no_salt')
        keys = list(keys)
        if self.counts is not None:                        # store_threshold >= 2: store a key only from its Nth offer
            self._record_accesses(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
        index = self._index
        todo = list(dict.fromkeys(k for k in keys if k not in index))
        if not todo:
            return PrepareStoreOutput(keys_to_store=[], store_spec=NvmeLoadStoreSpec([], []), evicted_keys=[])
        req_id = req_context.req_id
        cached = tag.admitted if tag is not None else self._verdicts.get(req_id)
        reservation = self._reservations.get(req_id)
        if cached is False:
            return self._drop_reason('denied')
        obj = self.object_bytes
        if cached is None:
            admit, reason, total = self._verdict(tag, todo)
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
        if extra and reservation is not None:
            self.admission_overrun += 1                    # the estimate was short: this part is judged on its own (it can still be denied)
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
        now = self._clock()
        slot_array = np.asarray(slots, dtype=np.int64)
        gens = runtime.allocator.gen[slot_array].tolist()
        batch = self._next_batch
        self._next_batch += 1
        self._state[slot_array] = E_ALLOCATED
        self._ref[slot_array] = 0
        self._tenant[slot_array] = self._tenant_value(tag)
        self._created[slot_array] = now
        self._last_used[slot_array] = now
        self._batch[slot_array] = batch
        slot_key, pending = self._slot_key, self._pending
        for key, slot in zip(todo, slots):
            index[key] = slot
            slot_key[slot] = key
            pending[slot] = now
        self._n_state[E_ALLOCATED] += need
        ledger = req_context.get_state(_StoreLedger)
        if ledger is None:
            ledger = _StoreLedger()
            req_context.set_state(ledger)
        ledger.add(batch, todo)
        output = PrepareStoreOutput(keys_to_store=todo, store_spec=NvmeLoadStoreSpec(list(slots), gens), evicted_keys=[])
        try:
            self._check_pairing(output)
        except PairingError as error:
            logger.error('ttkvp2p: %s; the store is rolled back and the tier latched', error)
            ledger.discard(batch)
            self._rollback(todo)
            runtime.latch('internal error: store pairing (%s)' % (error,), 'op_throw')
            return self._drop_reason('pairing')
        return output

    def _record_accesses(self, keys: Collection[OffloadKey]) -> None:
        """Count an offer without evicting its tracked candidates (store_threshold >= 2)."""
        counts = self.counts
        assert counts is not None
        protected = set()
        for key in keys:
            if key in counts:
                counts.move_to_end(key)
                counts[key] += 1
                protected.add(key)
        unprotected = len(counts) - len(protected)
        for key in keys:
            if key in counts:
                continue
            if len(counts) >= self.max_tracker_size:
                if unprotected == 0:
                    continue
                counts.popitem(last=False)
                unprotected -= 1
            counts[key] = 1

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
        index = self._index
        gen = self.runtime.allocator.gen
        slot_key = self._slot_key
        for position, (key, slot, generation) in enumerate(zip(keys, spec.slot_ids, spec.generations)):
            if index.get(key) != slot or int(gen[slot]) != generation or slot_key[slot] != key:
                raise PairingError('position %d: key and slot %d do not pair' % (position, slot))

    def _rollback(self, keys: List[bytes]) -> None:
        slots = self._drop_entries(keys)
        if slots:
            self.runtime.allocator.release_many(slots)

    def _digest_rows(self, digests: Any, n: int) -> Optional[np.ndarray]:
        """``digests`` as a ``[n, chips]`` uint64 array, or None when it is not one (a success without usable digests is a failed store)."""
        if digests is None:
            return None
        try:
            rows = np.asarray(digests)
        except Exception:
            return None
        if rows.ndim != 2 or rows.shape[0] != n or rows.shape[1] != self._chips or rows.dtype.kind not in 'ui':
            return None
        if rows.dtype.kind == 'i' and n and bool((rows < 0).any()):
            return None
        return rows.astype(np.uint64, copy=False)

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True, digests: Any = None) -> None:
        """A store job reported (amendment A1 of INTERFACE.md).

        ``keys`` are in this manager's ``prepare_store`` order, ``digests`` is the ``[len(keys), chips]`` uint64 array in that same order:
        row i belongs to ``keys[i]``, whatever slot the key was given. The digests are required when ``success`` is true and ignored when
        it is false. A success without usable digests is a failed store. A failed store drops its entries; their slots go back through TRIM
        (I5). Nothing here reads the runtime's store results or pairs a digest by slot or generation; and nothing here marks an entry
        WRITTEN unless it is still the entry this job's ``prepare_store`` allocated (its batch), so a late completion cannot turn a NEW
        entry of the same key, allocated after a wipe or the reap of a stuck store, into a written one."""
        keys = list(keys)
        count = len(keys)
        ledger = req_context.get_state(_StoreLedger) if req_context is not None else None
        job_batch = ledger.take(keys) if ledger is not None else None          # this report settles the job's record, whatever is left of its entries
        rows = None
        if success:
            rows = self._digest_rows(digests, count)
            if rows is None:
                self.store_completion_invalid += 1
                logger.error('ttkvp2p: complete_store(success) for %d keys without usable digests; treated as a failed store', count)
                success = False
        index, state = self._index, self._state
        positions: List[int] = []
        slots: List[int] = []
        for position, key in enumerate(keys):
            slot = index.get(key)
            if slot is not None and state[slot] == E_ALLOCATED:      # (a reset in between, or a repeated report, leaves nothing to complete)
                positions.append(position)
                slots.append(slot)
        if not slots:
            return
        slot_array = np.asarray(slots, dtype=np.int64)
        batches = self._batch[slot_array]
        if job_batch is None:
            self.store_completion_unmatched += 1
            if ledger is not None:
                # The request's context knows its jobs and this is not one of them: a second report of a job already settled, or a report
                # for keys this manager never handed out. Whatever entries the keys have now are somebody else's; leave them alone.
                return
            # A context that never saw a prepare_store (a caller that did not pass the one prepare_store got): fall back to the batch most
            # of the entries belong to. In the connector's flow this does not happen; the counter says if it does.
            values, occurrences = np.unique(batches, return_counts=True)
            job_batch = int(values[int(occurrences.argmax())])
        mine = batches == np.uint64(job_batch)
        if not bool(mine.all()):
            # entries of the same keys that a LATER prepare_store allocated (after a wipe or the reap of a stuck store): their own job has
            # not reported yet, and this one's digests are not theirs
            self.store_completion_stale += int(mine.size - np.count_nonzero(mine))
            positions = [p for p, keep in zip(positions, mine.tolist()) if keep]
            slot_array = slot_array[mine]
            slots = slot_array.tolist()
        done = len(slots)
        if not done:
            return
        if success:
            now = self._clock()
            self._digest[slot_array] = rows[np.asarray(positions, dtype=np.int64)]
            self._state[slot_array] = E_WRITTEN
            self._last_used[slot_array] = now
            self._created[slot_array] = now
            self._ref[slot_array] = 0
            pending, lru = self._pending, self._lru
            for position, slot in zip(positions, slots):
                pending.pop(slot, None)
                lru[keys[position]] = None                          # evictable now, the most recent
            self._n_state[E_ALLOCATED] -= done
            self._n_state[E_WRITTEN] += done
            self.stored_objects += done
        else:
            self._remove_many([keys[p] for p in positions], 'store_failed')

    # ------------------------------------------------------------------------------------------------------------
    # removal
    # ------------------------------------------------------------------------------------------------------------

    def _drop_entries(self, keys: Sequence[bytes]) -> List[int]:
        """Take the entries out of the index and the LRU and clear their slot rows. Returns the slots; the caller releases them to the allocator
        in ONE call (a release per key would be one numpy pass per key)."""
        index, lru, state, n_state = self._index, self._lru, self._state, self._n_state
        bad = self.runtime.bad
        gen = self.runtime.allocator.gen
        slots: List[int] = []
        for key in keys:
            slot = index.pop(key, None)
            if slot is None:
                continue
            lru.pop(key, None)
            n_state[int(state[slot])] -= 1
            slots.append(slot)
            if bad:
                bad.discard((slot, int(gen[slot])))
        if slots:
            array = np.asarray(slots, dtype=np.int64)
            state[array] = E_NONE
            self._ref[array] = 0
            self._slot_key[array] = None
            pending = self._pending
            if pending:
                for slot in slots:
                    pending.pop(slot, None)
        return slots

    def _remove_many(self, keys: Sequence[bytes], reason: str) -> None:
        slots = self._drop_entries(keys)
        if slots:
            self.runtime.allocator.release_many(slots)
            self.evictions[reason] += len(slots)

    def on_slot_bad(self, slot: int, generation: int) -> None:
        """The worker proved (slot, generation) bad. Mark it; it is removed when its last load completes (or now, if nobody holds it)."""
        if not 0 <= slot < self._tables_n or int(self.runtime.allocator.gen[slot]) != generation:
            return
        key = self._slot_key[slot]
        if key is None or self._state[slot] not in (E_WRITTEN, E_BAD):      # only a written entry can be proven bad (a store in flight decides itself)
            return
        self._set_state(slot, E_BAD)
        self._lru.pop(key, None)                          # a BAD entry is never evicted by the LRU, only removed when its last load completes
        if self._ref[slot] == 0:
            self._remove_many([key], 'bad')

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
        """Reap stuck stores, expire by TTL, evict by LRU: at most ``EVICT_PER_PASS`` entries, released to the allocator in one batch."""
        budget = EVICT_PER_PASS
        # a store that never completed (its job vanished): give it up, so lookups stop deferring on it
        stuck: List[bytes] = []
        while self._pending and budget:
            slot, since = next(iter(self._pending.items()))
            if now - since <= ALLOCATED_REAP_S:
                break
            self._pending.pop(slot, None)
            key = self._slot_key[slot]
            if key is not None and self._state[slot] == E_ALLOCATED:       # (only a store still in flight is given up; never a written entry)
                stuck.append(key)
            budget -= 1
        if stuck:
            self._remove_many(stuck, 'stuck_store')
        lru, index, last_used, ref = self._lru, self._index, self._last_used, self._ref
        # TTL: entries not used for ttl_s go, oldest first (the LRU order is the last-used order). An entry a load is reading is passed over
        # (moved to the back: its load completion stamps a use).
        ttl = self.cfg.ttl_s
        expired: List[bytes] = []
        while lru and budget:
            key = next(iter(lru))
            slot = index[key]
            budget -= 1
            if ref[slot] > 0:
                lru.move_to_end(key)
                continue
            if now - last_used[slot] < ttl:
                break
            del lru[key]
            expired.append(key)
        if expired:
            self._remove_many(expired, 'ttl')
        # capacity: keep the standing reserve free below the floor AND the reservations admitted requests hold, so a new maximum session
        # always fits (an admission uses up the reserve it was let in on; the entries it will write must make room before the next arrives)
        target = max(0, self.capacity_slots - self.reserve_slots - self._reserved_total)
        over = len(index) - target
        victims: List[bytes] = []
        while lru and budget and over > 0:
            key = next(iter(lru))
            budget -= 1
            if ref[index[key]] > 0:
                lru.move_to_end(key)
                continue
            del lru[key]
            victims.append(key)
            over -= 1
        if victims:
            self._remove_many(victims, 'lru')

    def reset_cache(self) -> None:
        """vLLM's pause and sleep paths (``reset_connector=True``): invalidate the index in RAM, bump the epoch, send the slots back through
        TRIM under I5. Never a wipe of the drive."""
        self.invalidate_all('reset')
        self.epoch += 1

    def wipe(self) -> None:
        """The control verb ``wipe``: invalidate the index and bump the epoch (the runtime then sends every slot back through TRIM)."""
        self.invalidate_all('wipe')
        self.epoch += 1

    def invalidate_all(self, kind: str = 'reset') -> None:
        slots = list(self._index.values())
        self._index.clear()
        self._lru.clear()
        if slots:
            array = np.asarray(slots, dtype=np.int64)
            self._state[array] = E_NONE
            self._ref[array] = 0
            self._slot_key[array] = None
        self._n_state = [0, 0, 0, 0]
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
    # state file, diagnostics
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
            'admission_size_unknown': self.admission_size_unknown, 'admission_overrun': self.admission_overrun,
            'store_completion_stale': self.store_completion_stale, 'store_completion_invalid': self.store_completion_invalid,
            'store_completion_unmatched': self.store_completion_unmatched,
            'dropped': dict(self.dropped), 'evictions': dict(self.evictions), 'epoch': self.epoch,
            'bytes_written_today': self._bytes.total(self._clock()), 'reserved_slots': self._reserved_total,
        }

    def check_invariants(self) -> None:
        """O(entries): the index, the LRU, the per-state counts and the slot table agree. For tests; never called on a serving path."""
        index = self._index
        assert len(set(index.values())) == len(index), 'two keys share a slot'
        counts = [0, 0, 0, 0]
        for key, slot in index.items():
            assert self._slot_key[slot] == key, 'slot_key disagrees with the index'
            state = int(self._state[slot])
            assert state != E_NONE, 'an indexed slot has no state'
            counts[state] += 1
            assert (key in self._lru) == (state == E_WRITTEN), 'LRU membership disagrees with the state of slot %d' % slot
            assert (self._ref[slot] == 0) or state in (E_WRITTEN, E_BAD), 'a store in flight is pinned by a load (slot %d)' % slot
            assert (slot in self._pending) == (state == E_ALLOCATED), 'pending set disagrees with the state of slot %d' % slot
        assert counts[1:] == self._n_state[1:], 'entry counts %r != %r' % (counts, self._n_state)
        assert set(self._lru) <= set(index), 'an LRU key is not indexed'
        assert int(np.count_nonzero(self._state)) == len(index), 'a state row is set for a slot without an entry'
