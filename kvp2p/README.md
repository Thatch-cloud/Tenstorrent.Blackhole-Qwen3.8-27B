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
| `ttkvp2p/cardlink.py` | `CardLink`: one card's end of the peer-to-peer path. TLB window 202 (re-aimed per staging bank, every aim proved by a canary read), the uncached CPU mapping, the mapping of the published slice, bounds-checked per-object `slot_view`s |
| `ttkvp2p/p2pio.py` | `P2PIO`: the drive side, both directions. Aim epochs, restore first, bounded spill sub-runs, vectored O_DIRECT `preadv` / `pwritev` with hard iovec bounds, the DMA canary |

Importing any of them imports neither `ttnn` nor the device extension, starts no thread and opens nothing: the API-server process
imports the connector class to decide whether the hybrid KV cache manager can stay on.

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
"for each key, in order: on success attach digest row i and make the entry WRITTEN; on failure drop the entry", and nothing more.

**Gate for the manager (WP3a): I6 beyond what the connector can see.** The connector checks that the keys come back in the scheduler's own
order and that a hashed source block matches its key. It cannot see a manager that permutes `slot_ids` against its OWN key-to-slot map:
the bytes of key `i` then land in a slot that the manager's entry for key `i` does not name, so a restore of that key reads another key's
object. The digest check fails it (a spurious BAD and a lost session) only because the digests are attached by key position; a manager that
attached them by slot would make wrong KV verify. `NvmeKVManager.prepare_store` must therefore assert, before it returns, that slot `i` of
the store spec is the slot it recorded for key `i` (with a test that a permuting allocator trips it); `complete_store` may re-check
each entry's slot and generation against the store spec it handed out. The tests here run a manager with exactly the signature above
(`A1Manager`, which raises on any call the amendment does not allow) and a slot-permuting one (`PermutingKVManager`: key `i` gets neither the
i-th allocated slot nor a monotone one), and check that every key ends up with the digest of its own object and restores exactly.

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
planned for an older aim.

**Scheduling.** `P2PIO` has one scheduler thread and `io_threads` I/O threads (`kvp2p-sched`, `kvp2p-io_N`, for the per-thread CPU gate). A call for the bank of the
current epoch runs (up to `io_queue_depth_objects` objects and `io_threads` calls in flight); a call for another bank waits until everything in flight has returned,
then all four cards are re-aimed. Restore (`read_piece`) is served before spill (`write_subrun`, at most `spill_subrun_objects` objects, so the drain is bounded).
The two directions use disjoint banks. `stats()` reports the aims, the longest aim, and how long a restore waited behind a spill (the drain the W1 probe bounds).

**Bounds.** Before every syscall each iovec is checked against the plan (the card is the shard's card, the buffer is exactly where the slot of that bank has the
object, the windows are aimed at the call's bank in the call's epoch), and the drive range against the data region (default: everything from 1 MiB, never the
superblock or the DMA canary block). A violation raises `BoundsViolation`, issues no syscall, stops the layer and latches the tier.

**Errors.** The future of a call carries the drive's `OSError` with its `errno` (EIO / EILSEQ are what the worker counts as media errors); a short count is continued;
an end of file is `P2PShortIO` (no errno).

**Tests** (`tests/fake_card.py` is a fake tt-kmd, four fake cards whose p2pmem mapping and CPU window alias whichever bank the window is aimed at, and a real file as
the drive, opened O_DIRECT where the file system allows it):

```sh
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_cardlink.py'      # stdlib only
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_p2pio.py'         # stdlib only
PYTHONPATH=kvp2p python3 -m unittest discover -s kvp2p/tests -p 'test_p2pio_mutants.py' # the safety checks, weakened one at a time (~40 s)
```

`test_wp4_wiring.py` drives the WP4 factory against the real modules and needs the vLLM environment.

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

`tests/nvml_stub` hides a GPU driver from vLLM's platform probe (set `TTKVP2P_TEST_HIDE_NVML=0` to leave it alone);
`TTKVP2P_PACKAGE_ROOT` points the tests at another copy of the package.
