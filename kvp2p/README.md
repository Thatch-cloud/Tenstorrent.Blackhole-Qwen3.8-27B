# ttkvp2p: the connector, spec and worker

A vLLM KV connector that spills finished KV blocks from card DRAM to an NVMe drive and restores them, peer to peer, with no host
copies of the KV bytes. It subclasses vLLM 0.30.0's `OffloadingConnector`, so the whole scheduler side (lookups, sliding-window
windows, store reachability, fences, flush, preemption) is vLLM's own.

This directory holds the connector side:

| Module | What it is |
|---|---|
| `ttkvp2p/registry.py` | `TTKVRegistry`: the allocated KV tensors, as the plugin runner hands them over, and the shard map (mesh coordinate to the PCI address of its card): `resolve_shard_bdf` reads it from the extension's `qwen_p2p_mesh_devices`, an op that allocates nothing on a card, so the start-up order keeps its zero device footprint. The plugin fills the map when it builds the registry; the spec completes a registry that arrives without one |
| `ttkvp2p/spec.py` | `TTP2PNvmeSpec`, `P2PConfig` (the `kv_connector_extra_config` keys), `NvmeLoadStoreSpec`, `ReqTag`, the layout table |
| `ttkvp2p/connector.py` | `TTP2PConnector` and its scheduler and worker roles |
| `ttkvp2p/worker.py` | `TTP2PWorker`: the job state machine, the staging ring, budgets, deadlines, strikes and latch, `wait()` |
| `ttkvp2p/runtime.py` | `KVP2PRuntime` (one per drive path per process, shared by both connector roles), `SlotAllocator` (invariant I5), the TRIM thread, the control directory and the state file |
| `ttkvp2p/manager.py` | `NvmeKVManager`: the scheduler-side index (invariants I6, I7, I12, I13), TTL, LRU, capacity, the daily write cap |
| `ttkvp2p/store.py` | the drive store, v1a: superblocks, geometry, the exclusive drive open, discard (standard library only) |
| `ttkvp2p/health.py` | the start-up host checks, the status-file reader, the control-directory parser, the state-file writer (standard library only) |
| `ttkvp2p/claim.py` | the claim tool, the only writer of a superblock; owner-run, never automatic (standard library only) |
| `ttkvp2p/cardlink.py` | `CardLink`: one card's end of the peer-to-peer path. TLB window 202 (re-aimed per staging bank, every aim proved by a canary read), the uncached CPU mapping, the mapping of the published slice, bounds-checked per-object `slot_view`s |
| `ttkvp2p/p2pio.py` | `P2PIO`: the drive side, both directions. Aim epochs, restore first, bounded spill sub-runs, vectored O_DIRECT `preadv` / `pwritev` with hard iovec bounds, the DMA canary |

Importing any of the connector-side modules imports neither `ttnn` nor the device extension, starts no thread and opens nothing: the API-server process
imports the connector class to decide whether the hybrid KV cache manager can stay on.

## The store, the runtime and the manager (WP3a)

**The drive.** A raw namespace carries a `gemma-kv` format v2 store, mode "process": two 4 KiB superblock slots, an 8 KiB DMA canary block,
then the data region on a MiB boundary with object `i` at `data_off + i * object_bytes`. The tier **never formats**: it refuses a drive with an
absent or unknown superblock. `python3 -m ttkvp2p.claim` is the only writer. It is gated: the by-id path, the name typed back, a fresh status
file that names the same drive and records (or is told) its size, an exclusive open, and `--overwrite=<format seen>` for a drive that carries
anything else. Mode "process" keeps nothing across a restart: the open verifies the superblock and queues a discard of the whole data
region in runs of at most 1 GiB, ALL of them at once to the one I/O thread (not two runs per engine pass); a slot is allocatable as its run
completes, and the engine thread applies the completions at its next pass. Until the last run is applied the tier is `warming`
(`capacity.warming` in the state file): a request whose boundary does not fit the slots already freed is refused for the reason `warming`.

**The runtime.** The worker role `acquire`s with role "worker" and opens it: the host checks of the design's start-up order run first and
leave no device footprint when one fails. They are, in order: the kill switch (`off` in the control directory), the configuration (v1a keeps
nothing across a restart, the TTL is at most a day, no gate-only value outside test mode), the zero-retention signals (the forwarded flag, the
log control, the node's attestation key, a marker file beside the control directory) and `retention_ack`, the cache-drive status file (fresh,
the drive named, live, unlocked, at rest by Opal, Gen4 or better), the drive (exclusive open, superblock, layout, size, the claim recomputed
from the status file's by-id name), capacity for one maximum-context session, and one allocate file per card with the slice whole and
mappable. A refusal carries a reason code from the closed vocabulary and is written to the state file. The scheduler role only attaches.
`release` is reference counted; the last one closes the runtime.

**Slots (I5).** A drive slot becomes allocatable only when no index entry names it, no bio of any job is in flight on it and its TRIM
completed; each allocation bumps its generation. `io_begin` refuses a slot that is not in use or whose generation moved on, so a stale job's
read never starts on a reallocated slot.

**The manager.** Engine thread only (I7). It subclasses vLLM's public `OffloadingManager` and nothing else of vLLM's: it keeps its OWN index
(one dict, key -> slot), its OWN LRU (an `OrderedDict`) and its own slot table, and imports nothing from `vllm.v1.kv_offload.cpu` (those
classes moved between vLLM releases: 0.31 stops evicting through them without an error, 0.33 crashes). Every call costs O(keys in the call):
nothing scans the index or the slot table, the per-state counts are incremental (a recount of 359,030 slots per release stalled the engine thread
for seconds at the slot count of a 1 TB drive), evictions are collected and released in one batch per pass, and pinning an entry for a load never
inserts or deletes in the LRU. "Now" for the TTL is frozen for a pass: the first manager call of a pass reads the clock and `on_schedule_end` ends
the pass, so an entry that answered HIT is still there for `prepare_load`; calls vLLM makes after `on_schedule_end` (`prepare_store`, `touch`, the
completions) never arm it. Expired entries and LRU victims go in `on_schedule_end`, which also pumps the runtime (completed TRIMs, the control
directory once a second, the status file, the state file every 10 s). Eviction keeps `entries + reservations + reserve <= capacity`, so an
admission does not use up the room the next one needs.

`prepare_store` allocates in the order asked and checks the pairing (I6). Admission (I12) is all or nothing per request boundary: the whole
prompt boundary must fit the free list, the capacity and the daily cap, the slots and the bytes are reserved for the request (the unused rest is
released and refunded when it finishes), and the verdict is cached in its `ReqTag`. The size comes from `ReqTag.num_prompt_tokens`, which the
connector's scheduler role sets from the request, and the group map, which it binds from the offloading config (`manager.bind_groups`). A
request whose size is unknown on a layout with a group map is refused whole (`size_unknown`, counted as `admission_size_unknown`), never judged
chunk by chunk: a chunk denied mid-prompt is a hole vLLM never revisits. A store that cannot be admitted is an EMPTY `keys_to_store`, never `None`.

`complete_store(keys, ctx, success=True, digests=None)` is amendment A1 of the interface: `keys` come in the order `prepare_store` returned
them, `digests` is a `[len(keys), chips]` uint64 array in that order, and row i belongs to `keys[i]`. The manager never reads the runtime's
`store_results` and never pairs a digest by slot or generation: the connector consumes the worker's result and hands the verdict and the rows
over. A success without usable digests is a failed store, a failed store drops its entries, and their slots go back through TRIM under I5. What
the manager does check is that an entry is still the one this job's `prepare_store` allocated: it keeps a batch number per slot and, in the
request's `ReqContext`, a ledger of the request's `prepare_store` calls, so a completion that arrives after a wipe or after the stuck-store reap
(300 s) cannot mark a NEW entry of the same key WRITTEN, or drop it. `reset_cache` is an in-RAM invalidate plus an epoch bump; `wipe` the same
plus a discard of the whole region; the slots go back through TRIM under I5. Neither is a format.

**Files the tier writes.** The state file (`state_file` in the kwarg), atomically, every 10 s while the engine is stepping AND at once on every
state change (opened, available, latched, a latch cleared, closing). The key names are frozen (additions are allowed, removals and renames are
not; a test pins them):

| key | meaning |
|---|---|
| `schema` (1), `spawn_id`, `written_at` | the standard's three additive fields (`spawn_id` echoes `THATCH_KVTIER_SPAWN_ID`, null when absent; `written_at` is ISO-8601 UTC) |
| `state` | `UNAVAILABLE`, `AVAILABLE`, `LATCHED` or `CLOSING` |
| `reason` | the CLOSED code of the state (`host_code` while `UNAVAILABLE`, `latch_code` while `LATCHED`, else null): the only reason field a metric label may be built from |
| `latch_reason`, `latch_code`, `latches` | the first active latch (text and code) and the codes of every active latch, oldest first |
| `host_ok`, `host_code`, `host_reason` | the host checks: pass/fail, the closed code, free text for people |
| `persistence`, `layout`, `kv_epoch`, `engine_heartbeat` | configuration echo; the wall time of the last `schedule()` pass |
| `counters` | `hits`, `misses`, `lookups`, `loads`, `stores` (jobs), `store_admitted_objects`, `store_dropped` ({reason: n}), `admission_size_unknown`, `admission_overrun`, `store_completion_stale`, `store_completion_invalid`, `store_completion_unmatched`, `evictions` ({reason: n}), `failures` ({cause: n}), `latched` ({code: n}), `trim_runs`, `trim_failed`, `bad_marked` |
| `bytes_written_today`, `write_cap_bytes` | the rolling 24 hour write budget |
| `capacity` | `slots`, `object_bytes`, `entries`, `capacity_slots`, `slot_states` ({free, used, trim_wait, trimming, draining, quarantined, slots, inflight, trim_failed}), `warming` |
| `restore` | `count`, `p50_s`, `p99_s` |
| `drive` | `state`, `link_gts`, `aer_uncorrectable`, `aer_correctable`, `status_age_s` |
| `control` | `acks` (the last 16 control acknowledgements) and `last_edge_seq` |

It reads the control directory (`control_dir`): `NNNN-off|on|wipe|delete_tenant|fault|selftest`; `off` and `on` are levels (the highest number
wins; `on` clears an `off` and nothing else, so a latch that tripped while `off` was in force stays), the rest apply once per process, and
`fault` and `selftest` need `THATCH_KVP2P_TEST=1`. `delete_tenant` is v1b and is acknowledged as unsupported.

**What it needs from the connector side** (all of it is in `connector.py`): `ReqTag.num_prompt_tokens`, set in `on_new_request`; the KV groups
(sliding window in objects, `None` = full attention) bound in the scheduler role's constructor from `config.kv_group_configs`; and, for every
completed store job, one `complete_store(keys, ctx, success, digests)` as above. The built-in Gemma group map (`LAYOUT_GROUPS`) only serves a
manager that nothing binds.

## What the worker needs from its neighbours

The worker reaches the card, the drive and the process-wide runtime only through injected collaborators.

**Card ops and drive I/O** (`worker.Backend`): the `qwen_kv_read` functions `qwen_blocks_to_staging`,
`qwen_write_blocks_from_staging`, `qwen_blocks_digest_async` and `qwen_p2p_record_event` (called with keyword arguments, always
`cq_id=0`), a staging handle with its `qwen_p2p_staging_info` dict, and a P2PIO with `read_piece(drive_off, bank, n_obj)` and
`write_subrun(drive_off, bank, n_obj)`. When the objects of a staging slot do not start at the slot's first object (slots on the
drive are not adjacent), the worker passes `first_obj=<k>` as a keyword; a P2PIO must accept it (default 0). This keyword is not in
the frozen interface: it needs WP2b's acceptance and a note in INTERFACE.md before integration.

The production factory (`worker.default_backend_factory`) checks the p2pmem allocate file of every card (by PCI address, exactly one
each) before it allocates anything on the cards, then names each card's character device by the tt-kmd index in the staging record
(`pcie_device`; tt-metal's logical `device_id` is not that index) and refuses a record that carries none. It checks that the highest staging
slot of every configured bank (each bank has its own address in the record) stays below the top 16 MiB of the window.

The index in `/dev/tenstorrent/<N>` comes from the staging record, which is a number the host may resolve differently from the PCI address
the shard map names. The factory therefore does not rely on it: `CardLink.open` must confirm that the character device it opened is the card
at the PCI address it was given (and that the record's index matches), and refuse otherwise, before any window is touched.

**The runtime** (`worker.RuntimePort` checks these up front and refuses a runtime that lacks one, so a mismatch shows as a tier that
stays UNAVAILABLE with a log line, never as a half-used runtime):

| Member | Used for |
|---|---|
| `state`, `latch(reason)`, `set_available()` | health; the worker latches on strikes, a stall, a device-op throw; it flips AVAILABLE as the last start-up step |
| `host_ok` (and optional `host_reason`) | the host checks passed: only then does any device work start |
| `store_results[job_id]` | a mutable mapping. The worker leaves a `worker.StoreResult` (`success`, `slot_ids`, `generations`, `digests` of the SOURCE rows) in it when a store job ends, done or failed. The connector's scheduler role **consumes** it (`pop`) and settles the job; the manager never reads it. The worker keeps at most the last 1,024 reported results, so a result that was never consumed can disappear: a missing result is a FAILED store |
| `allocator.io_begin(slots, generations) -> bool`, `allocator.io_end(slots)` | the in-flight count per drive slot (a slot is reusable only when no bio is in flight); `io_begin` refuses, and counts nothing, when a generation moved on |
| `mark_bad(slot, generation)` | keys go BAD only on proof: a digest mismatch after one re-read, or EIO / EILSEQ on a read |
| `slot_offset(slot)` | the byte offset of a slot on the drive |
| optional `count`, `observe_restore`, `take_fault`, `release` | telemetry; the test-only fault commands; reference-counted release |

**The manager** (`spec.get_manager` builds `ttkvp2p.manager.NvmeKVManager(cfg, runtime)`): its `lookup` answers MISS for a request whose
`ReqTag.no_retry` is set (read with `req_context.get_state(ReqTag)`), and its `complete_load` reads the worker's `mark_bad` marks. It keeps its
own LRU and eviction structure: nothing here depends on vLLM's internal LRU classes.

**How a store job ends in the manager (amendment A1).** vLLM's own `update_connector_output` calls `manager.complete_store(job.keys, ctx)` for
every completed store job: the key SET (unordered), no job id, no digests, and no way to say the store failed. That call cannot carry the
outcome, so `TTP2PConnectorScheduler.update_connector_output` settles each store job itself, where the job id, the keys and the result
meet, and suppresses vLLM's call for it. The manager's side of the contract is one signature:

```python
def complete_store(self, keys, req_context, success: bool = True, digests: "np.ndarray | None" = None) -> None
```

- `keys` is a list **in the order the manager's own `prepare_store` returned them** (never vLLM's unordered `job.keys`);
- `digests` is an `[len(keys), chips]` uint64 array in that same order. It is required when `success=True` and `None` when `success=False`
  (the connector passes the keyword on every call);
- the connector takes the job's `StoreResult` from `store_results` (through `spec.take_store_result`) and checks it against the job before it
  calls the manager with `success=True`: the same keys; the same drive slots and generations as the store spec the manager returned; and
  digests that are an ndarray of dtype uint64 with exactly the shape `[keys, chips]` (chips from the layout). Any other result is a failed
  store, like a failed job, a missing result (the worker keeps only the last 1,024 and a job it never ran has none) or an exception while
  settling: `complete_store(keys, ctx, success=False, digests=None)`, the entries are dropped and their drive slots return to the free
  list under I5. A refused result is a miss, never a strike;
- the manager NEVER reads `store_results` and never pairs a digest with a key by slot or generation: digest row `i` belongs to `keys[i]`,
  whatever slots the manager handed out (a manager may permute them);
- an exception from `complete_store` latches the tier and gives the keys up with a second `success=False` call, so the engine never sees
  it and no key stays ALLOCATED (an ALLOCATED entry answers HIT_PENDING, and every lookup of its key would defer).

`get_manager` checks the manager it built with `inspect.signature` (`spec.store_completion_problem`): `complete_store` must name a `digests`
parameter and must bind `(keys, ctx, success=True|False, digests=None)`. A manager that does not (the stock vLLM signature, an older draft, a
`**kwargs` that would swallow the digests) gets the `RefusedManager` instead: the runtime reference taken for it is given back, the log says
why, and the tier is a no-op that never stores from boot, instead of raising in the first successful store and latching then.

A job from before a `reset_cache` is skipped by vLLM; its result is taken and dropped. So the manager implements `complete_store` as
"for each key, in order: on success attach digest row i and make the entry WRITTEN; on failure drop the entry", and nothing more (the WP3a
manager adds one check of its own: that the entry is still the one this job's `prepare_store` allocated, so a late report cannot touch a newer
entry of the same key).

**Gate for the manager (WP3a): I6 beyond what the connector can see.** The connector checks that the keys come back in the scheduler's own
order and that a hashed source block matches its key. It cannot see a manager that permutes `slot_ids` against its OWN key-to-slot map:
the bytes of key `i` then land in a slot that the manager's entry for key `i` does not name, so a restore of that key reads another key's
object. The digest check fails it (a spurious BAD and a lost session) only because the digests are attached by key position; a manager that
attached them by slot would make wrong KV verify. `NvmeKVManager.prepare_store` therefore asserts, before it returns, that slot `i` of
the store spec is the slot it recorded for key `i` (a corrupted spec and each kind of swap trip it, and a permuting allocator is exercised
through the real scheduler), and `complete_store` attaches the digests by key position and by nothing else. The tests here run a manager with
exactly the signature above (`A1Manager`, which raises on any call the amendment does not allow) and a slot-permuting one (`PermutingKVManager`:
key `i` gets neither the i-th allocated slot nor a monotone one), and check that every key ends up with the digest of its own object and
restores exactly.

## The drive side: `CardLink` and `P2PIO` (WP2b)

Both are stdlib only (no numpy, no vLLM, no ttnn) and are imported by the production wiring at start-up (`worker.default_backend_factory`).

**Geometry.** One *object* is one KV block of one group: four *shards* of `shard_object_bytes` (696,320 for Gemma), one per card, laid out on the drive as
`[shard 0][shard 1][shard 2][shard 3]`. Each card's staging bank holds a 4 KiB canary page and then the slot's objects, so object `j` of a slot is at
`canary + 4096 + j * shard_object_bytes` on every card. One `preadv` / `pwritev` covers one contiguous run of objects and its iovecs are, per object, shard 0..3
(the cards' `slot_view`s), so the drive offset of every byte follows from the iovec order.

**The window.** Window 202 of each card is the only way to reach a bank, so it is *aimed* at one bank at a time and all four cards at the same bank (an *aim epoch*).
`CardLink.aim(bank)` is `CONFIGURE_TLB` followed by an 8-byte non-posted read of that bank's canary page through the uncached mapping; any other value raises
`CanaryMismatch` and the link is unusable. `CardLink.open` writes a fresh canary page (a per-open random nonce) into **every** bank and reads them all back
before any data is stored. `begin_io` / `end_io` count the bios in flight on a card: `aim` refuses while there is one, and `begin_io` refuses an I/O that was
planned for an older aim. Every CPU access to the uncached window is one aligned 8-byte load or store (`ops.words`, a `Q` view of the mapping), never a `memcpy` of
the range; `open` waits `settle_s` (default 1 s, as the Qwen link does) after the power state; the published slice must start on a page boundary.

**Scheduling.** `P2PIO` has one scheduler thread and `io_threads` I/O threads (`kvp2p-sched`, `kvp2p-io_N`, for the per-thread CPU gate). A call for the bank of the
current epoch runs (up to `io_queue_depth_objects` objects and `io_threads` calls in flight); a call for another bank waits until everything in flight has returned,
then all four cards are re-aimed. Restore (`read_piece`) is served before spill (`write_subrun`, at most `spill_subrun_objects` objects, so the drain is bounded).
The two directions use disjoint banks. `stats()` reports the aims, the longest aim, and how long a restore waited behind a spill (the drain the W1 probe bounds).

**What `P2PIO` asks of the runtime** (the names `KVP2PRuntime` has): `runtime.drive.fd` (the open descriptor of the cache drive, which the runtime opened exclusively),
`runtime.store.data_off` / `num_slots` / `object_bytes` (the data region), optionally `runtime.canary_offset`, and `runtime.latch(reason)`. The layer never opens the
device itself, never guesses a data region (a runtime that states none, or objects of another size, is refused and the tier stays UNAVAILABLE), and refuses a
descriptor that is not open O_DIRECT (buffered I/O would make the CPU copy the cards' BAR pages; only a test passes `allow_buffered`).

**Bounds.** Before every syscall each iovec is checked against the plan (the card is the shard's card, the buffer is exactly where the slot of that bank has the
object, the windows are aimed at the call's bank in the call's epoch), and the drive range against the data region the runtime states (never the superblock or the
DMA canary block). A violation raises `BoundsViolation`, issues no syscall and stops the layer. Once the layer is dead nothing more is issued, not even a call that
was already decided.

**Futures.** A future resolves only when no call of its request is still in flight: the worker releases the bank and the slots' in-flight counts when it sees the
future done. This matters when a request is cut into several calls (`restore_run_objects` shorter than the run).

**Threads.** The layer's own threads never call into the runtime (the runtime belongs to the engine thread). A fault found there stops the layer and is recorded in
`P2PIO.dead`, a reason that begins `P2PIO bounds violation` (the runtime files that text under the `bounds_violation` latch code); the worker reads it at its next
`service` and latches the tier from the engine thread. A fault found on the caller's thread (a bad request, the start-up DMA canary) is latched there.

**Not proven on a card** (the W1 kit settles each; the CPU tests use a fake kernel and cannot): the uncached mapping of window 202 (only write-combining ran on a
card), that an 8-byte non-posted read cannot complete before the posted `CONFIGURE_TLB` writes landed, that the card accepts the 8-byte accesses and the ordering value
`relaxed` for the canary writes, that `O_DIRECT` `preadv` / `pwritev` accept iovecs from four cards' p2pmem mappings in one call, and the sysfs file names under
`p2pmem`. Only the bank's first NOC endpoint is used; the Qwen link fell back to the bank's other endpoints when the first did not store. This module fails closed
instead, so the risk is availability, not safety.

**Errors.** The future of a call carries the drive's `OSError` with its `errno` (EIO / EILSEQ are what the worker counts as media errors); a short count is continued;
an end of file is `P2PShortIO` (no errno).

**Tests** (`tests/fake_card.py` is a fake tt-kmd, four fake cards whose p2pmem mapping and CPU window alias whichever bank the window is aimed at, and a real file as
the drive, opened O_DIRECT where the file system allows it):

```sh
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_cardlink.py'      # stdlib only
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_p2pio.py'         # stdlib only
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_p2pio_mutants.py' # the safety checks, weakened one at a time (~1 min)
```

`test_wp4_wiring.py` drives the WP4 factory against the real modules (and the worker's latch from `P2PIO.dead`) and needs the vLLM environment.

## What the model runner calls, in this order

This is the sequence the tests drive (it is what the plugin hooks must do each step):

```
connector.register_kv_caches(TTKVRegistry)         once, after the KV tensors are allocated and before the warm-up; never raises
connector.handle_preemptions(meta)                 before the forward: submits deferred stores, and gathers the jobs in jobs_to_flush now
connector.bind_connector_metadata(meta)
    ... the forward is enqueued ...
connector.start_load_kv(None)                      hands loads to the drive and submits the stores deferred from the last step
connector.get_transfer_results(finished_req_ids)   NOT get_finished: only this path runs prepare_store_kv, so only it submits stores
connector.get_block_ids_with_load_errors()
connector.build_connector_worker_meta()            completed job ids, and why each failed load failed
connector.clear_connector_metadata()
connector.begin_step(decode_active=..., idle=...)  optional hint before get_transfer_results: no decode batch lifts the per-step device
                                                   budget; a zero-token step may wait idle_wait_ms for a completion instead of spinning
connector.shutdown()                               the executor's connector first, the scheduler's second (the runtime is reference counted)
```

A failed load is in both `finished_recving` and `failed_recving` of the same output, exactly once; the request then recomputes.
Every job is reported exactly once, whatever happens to it: a bug in the state machine latches the tier, and a job whose submit hit it is
reported failed (`internal`) like any other.

A store that cannot get a spill bank for `store_wait_s` (default 10 s; an additive `kv_connector_extra_config` key, not in the frozen kwarg and
not yet in INTERFACE.md) while none comes back is dropped (`no_slot`, not a strike): a later miss, never a stall. The stall clock (`stall_latch_s`) starts when work does: it does not count idle time before a job.

## Running the tests

The tests drive vLLM's real `Scheduler`, `KVCacheManager` and `OffloadingConnectorScheduler` (through the ttkvp2p subclasses)
against a numpy "card" and a file "drive". They need vLLM 0.30.0 with the empty target and its common requirements (see the
`qwen-fast-vllm-cpu.yml` recipe), and no device:

```sh
PYTHONPATH=kvp2p python -m pytest kvp2p/tests          # or: python -m unittest discover -s kvp2p/tests
```

The WP3a tests use a file-backed drive claimed by the real claim tool (a regular file is allowed only under `THATCH_KVP2P_TEST=1`, which the
test kit sets), a temporary status directory, fake p2pmem files and sysfs, and a controllable clock. `harness.Rig()` runs the integrated path by
default: the connector, the REAL `KVP2PRuntime` and `NvmeKVManager` and the worker run together inside vLLM's real scheduler, and only the devices and
the I/O (the numpy card, the file drive, `FakeIO`) are fakes; nothing is patched (the connector sets the prompt size and binds the groups itself).
The stand-in runtime and manager of `tests/fakes.py` stay available, for the tests that need a fault knob on the runtime, a manager with another
signature or a start-up refusal: `Rig(fake_tier=True)`, `RigCase.FAKE_TIER = True` or `Rig(manager_class=...)`. Most of the connector and worker tests
run on both (`StandIn*` classes repeat them on the stand-ins) and ask the rig, not the tier, for the books (`written_slots`, `entry_digest`,
`allocated_keys`, `slots_all_returned`, `bad_marked`, `complete_store_calls`, ...). `test_tier_rig.py` and `test_object_count_262k.py` use the same
real tier with their own `RealTier` options. `test_runtime_p2pio_seam.py` joins the other end: the worker's production factory builds the real
`CardLink` and `P2PIO` over the fake kernel and hands them the real runtime (descriptor, fence, canary offset, slot offsets, latch code).
`test_scale_production.py` builds an index of 340,732 objects on a 359,030-slot drive (the slot count of 1 TB of Gemma objects; a sparse file
stands in for the namespace) and bounds the latency of every call against it.

`tests/nvml_stub` hides a GPU driver from vLLM's platform probe (set `TTKVP2P_TEST_HIDE_NVML=0` to leave it alone);
`TTKVP2P_PACKAGE_ROOT` points the tests at another copy of the package.
