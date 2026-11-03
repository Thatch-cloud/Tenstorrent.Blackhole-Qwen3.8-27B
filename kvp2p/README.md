# ttkvp2p: the connector, spec and worker

A vLLM KV connector that spills finished KV blocks from card DRAM to an NVMe drive and restores them, peer to peer, with no host
copies of the KV bytes. It subclasses vLLM 0.30.0's `OffloadingConnector`, so the whole scheduler side (lookups, sliding-window
windows, store reachability, fences, flush, preemption) is vLLM's own.

This directory holds the connector side:

| Module | What it is |
|---|---|
| `ttkvp2p/registry.py` | `TTKVRegistry`: the allocated KV tensors, as the plugin runner hands them over |
| `ttkvp2p/spec.py` | `TTP2PNvmeSpec`, `P2PConfig` (the `kv_connector_extra_config` keys), `NvmeLoadStoreSpec`, `ReqTag`, the layout table |
| `ttkvp2p/connector.py` | `TTP2PConnector` and its scheduler and worker roles |
| `ttkvp2p/worker.py` | `TTP2PWorker`: the job state machine, the staging ring, budgets, deadlines, strikes and latch, `wait()` |
| `ttkvp2p/runtime.py` | `KVP2PRuntime` (one per drive path per process, shared by both connector roles), `SlotAllocator` (invariant I5), the TRIM thread, the control directory and the state file |
| `ttkvp2p/manager.py` | `NvmeKVManager`: the scheduler-side index (invariants I6, I7, I12, I13), TTL, LRU, capacity, the daily write cap |
| `ttkvp2p/store.py` | the drive store, v1a: superblocks, geometry, the exclusive drive open, discard (standard library only) |
| `ttkvp2p/health.py` | the start-up host checks, the status-file reader, the control-directory parser, the state-file writer (standard library only) |
| `ttkvp2p/claim.py` | the claim tool, the only writer of a superblock; owner-run, never automatic (standard library only) |

Importing any of the connector-side modules imports neither `ttnn` nor the device extension, starts no thread and opens nothing: the API-server process
imports the connector class to decide whether the hybrid KV cache manager can stay on.

## The store, the runtime and the manager (WP3a)

**The drive.** A raw namespace carries a `gemma-kv` format v2 store, mode "process": two 4 KiB superblock slots, an 8 KiB DMA canary block,
then the data region on a MiB boundary with object `i` at `data_off + i * object_bytes`. The tier **never formats**: it refuses a drive with an
absent or unknown superblock. `python3 -m ttkvp2p.claim` is the only writer. It is gated: the by-id path, the name typed back, a fresh status
file that names the same drive and records (or is told) its size, an exclusive open, and `--overwrite=<format seen>` for a drive that carries
anything else. Mode "process" keeps nothing across a restart: the open verifies the superblock and queues a discard of the whole data
region in runs of at most 1 GiB; a slot is allocatable as its run completes.

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

**The manager.** Engine thread only (I7). "Now" for the TTL is frozen for a pass: the first manager call of a pass reads the clock and
`on_schedule_end` ends the pass, so an entry that answered HIT is still there for `prepare_load`. Expired entries and LRU victims go in
`on_schedule_end`, which also pumps the runtime (completed TRIMs, the control directory once a second, the status file, the state file every
10 s). `prepare_store` allocates in the order asked and checks the pairing (I6), and admission (I12) is all or nothing per request boundary:
the whole prompt boundary must fit the free list, the capacity and the daily cap, the slots and the bytes are reserved for the request, and the
verdict is cached in its `ReqTag`. A store that cannot be admitted is an EMPTY `keys_to_store`, never `None`. `complete_store(keys, ctx, ...)`
pairs each key with the digest of its own store job through the entry's (slot, generation) and an index over the worker's results (no scan),
so a failed job, a missing result or a result of another generation drops the entry. `reset_cache` is an in-RAM invalidate plus an epoch
bump; the slots go back through TRIM under I5. It is never a wipe.

**Files the tier writes.** The state file (`state_file` in the kwarg) every 10 s while the engine is stepping, atomically, with `schema`,
`spawn_id` (the echo of `THATCH_KVTIER_SPAWN_ID`, null when absent) and `written_at`. Nothing else. It reads the control directory
(`control_dir`): `NNNN-off|on|wipe|delete_tenant|fault|selftest`; `off` and `on` are levels (the highest number wins), the rest apply once per
process, and `fault` and `selftest` need `THATCH_KVP2P_TEST=1`.

**What it needs from the connector side** (the changes WP4 owns are listed in the WP3a report, none is needed to run): an optional
`ReqTag.num_prompt_tokens`, set from the request, makes admission judge the whole boundary at the first store (without it the manager judges
each call's keys alone); `manager.bind_groups(...)` can name the real KV groups (the Gemma layout has a built-in map: one full-attention
group and five sliding groups of 8 objects); `complete_store(..., result=<the job's StoreResult>)` hands the job's own result over directly
(without it the manager finds the result through the runtime).

## What the worker needs from its neighbours

The worker reaches the card, the drive and the process-wide runtime only through injected collaborators.

**Card ops and drive I/O** (`worker.Backend`): the `qwen_kv_read` functions `qwen_blocks_to_staging`,
`qwen_write_blocks_from_staging`, `qwen_blocks_digest_async` and `qwen_p2p_record_event` (called with keyword arguments, always
`cq_id=0`), a staging handle with its `qwen_p2p_staging_info` dict, and a P2PIO with `read_piece(drive_off, bank, n_obj)` and
`write_subrun(drive_off, bank, n_obj)`. When the objects of a staging slot do not start at the slot's first object (slots on the
drive are not adjacent), the worker passes `first_obj=<k>` as a keyword; a P2PIO must accept it (default 0).

**The runtime** (`worker.RuntimePort` checks these up front and refuses a runtime that lacks one, so a mismatch shows as a tier that
stays UNAVAILABLE with a log line, never as a half-used runtime):

| Member | Used for |
|---|---|
| `state`, `latch(reason)`, `set_available()` | health; the worker latches on strikes, a stall, a device-op throw; it flips AVAILABLE as the last start-up step |
| `host_ok` (and optional `host_reason`) | the host checks passed: only then does any device work start |
| `store_results[job_id]` | the worker leaves a `worker.StoreResult` (`success`, `slot_ids`, `generations`, `digests` of the SOURCE rows) for the manager's `complete_store` |
| `allocator.io_begin(slots, generations) -> bool`, `allocator.io_end(slots)` | the in-flight count per drive slot (a slot is reusable only when no bio is in flight); `io_begin` refuses, and counts nothing, when a generation moved on |
| `mark_bad(slot, generation)` | keys go BAD only on proof: a digest mismatch after one re-read, or EIO / EILSEQ on a read |
| `slot_offset(slot)` | the byte offset of a slot on the drive |
| optional `count`, `observe_restore`, `take_fault`, `release` | telemetry; the test-only fault commands; reference-counted release |

**The manager** (`spec.get_manager` builds `ttkvp2p.manager.NvmeKVManager(cfg, runtime)`): its `lookup` answers MISS for a request whose
`ReqTag.no_retry` is set (read with `req_context.get_state(ReqTag)`), and its `complete_store` / `complete_load` read the worker's
`store_results` and `mark_bad` marks.

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

## Running the tests

The tests drive vLLM's real `Scheduler`, `KVCacheManager` and `OffloadingConnectorScheduler` (through the ttkvp2p subclasses)
against a numpy "card" and a file "drive". They need vLLM 0.30.0 with the empty target and its common requirements (see the
`qwen-fast-vllm-cpu.yml` recipe), and no device:

```sh
PYTHONPATH=kvp2p python -m pytest kvp2p/tests          # or: python -m unittest discover -s kvp2p/tests
```

The WP3a tests use a file-backed drive claimed by the real claim tool (a regular file is allowed only under `THATCH_KVP2P_TEST=1`, which the
test kit sets), a temporary status directory, fake p2pmem files and sysfs, and a controllable clock. `test_tier_rig.py` and
`test_object_count_262k.py` run the real runtime and manager inside vLLM's real scheduler through `harness.Rig(real_tier=...)`.

`tests/nvml_stub` hides a GPU driver from vLLM's platform probe (set `TTKVP2P_TEST_HIDE_NVML=0` to leave it alone);
`TTKVP2P_PACKAGE_ROOT` points the tests at another copy of the package.
