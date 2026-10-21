# tp4/kv-nvme-1: an NVMe cold tier for the prefix state

Status: **built and tested on CPU, never run on a card, default off.** Branched from `tp4/next-1` (`758264ac`, the ship-5 production stack). With none of the `QWEN_PREFIX_NVME*` names set the
host tier, the checkpoint store and the scheduler graft behave exactly as they did (a pinned digest of twelve random-traffic runs holds that, `test_qwen_kv_nvme.FlagOffTests`).

Every effect in this note that is not a CPU measurement is labelled **ESTIMATE**; nothing of it has run on a card or on the target drive.

## 0. Owner decisions (2026-10-21) this build follows

1. **Retention is for all tenants, for now.** No tenant allowlist. Keys still carry the tenant's cache salt (vLLM's chained block hash), so one tenant's blocks can never be named by another. **TTL 7 days** (`QWEN_PREFIX_NVME_TTL_S`, default 604,800), counted from the last use.
2. **Seats are released at the end of a turn** (a separate change); the tiers are what keep a session's KV warm. So spilling to the NVMe has to be reliable when a session's KV leaves the card or the host tier, and there is a **write-through option** (`QWEN_PREFIX_NVME_SPILL=through`, default off, section 5) that puts a warm copy on the drive soon after a session goes idle. Its write rate is in section 8.
3. **A zero-retention node must not run the tier by accident.** The tier refuses to start (a loud line and a counter; the engine serves without it) when the process sees a zero-retention signal, unless an explicit override says otherwise, and it always needs an explicit retention acknowledgement. Section 9.

## 1. What it is

The host KV tier (32 GiB at the ship setting) holds the attention-KV blocks vLLM evicted from the device pool; the checkpoint store (16 GiB) holds the GDN checkpoints that make a hit exact. Both are RAM and
both die with the engine process. The NVMe tier is a third level under them on a **raw namespace** (no filesystem, no md, no LVM, no dm: the later peer-to-peer path needs plain namespace LBAs):

```
 device pool (cards) --evict--> host KV tier (RAM, 32 GiB) --evict--> NVMe tier (raw namespace, ~1 TB)
        ^                              ^   (slab slot retained until written)           |
        |   existing restore           |                                                |
        +--- (TierHooks.load/advance) -+<--- read into slab slots (backend 'host') -----+
 checkpoint store (RAM, 16 GiB) --budget eviction / with its boundary block--> NVMe tier   (adopted back by the trim)
```

* **Spill on host-tier eviction, asynchronously.** `TierStore.remove(evicted=True)` calls the NVMe tier's sink (`NvmeTier.offer`) on the scheduler thread. The sink only enqueues: it keeps the record's **slab slot** (the slab is mapped with spare slots for exactly this) and the writer thread hashes, writes and journals the batch, then gives the slot back. No copy, no wait for the drive. A full queue (`QWEN_PREFIX_NVME_QUEUE_BYTES`) drops the record, as the host tier always did (`nvme_spill_dropped_queue`). A record no checkpoint reaches is not written (`nvme_spill_dropped_useless`), one already on the drive is not written twice (`nvme_spill_known`).
* **Checkpoints go with their sessions.** A GDN checkpoint is written (packed on the writer thread: header, token ids, the tensors, the drafter window it carries) when the store evicts it for its byte budget, and when the KV record of its boundary block leaves the host tier. A block of the same chain is *not* "dead" while its checkpoint is queued (`queued_ckpt`): the tail of a session leaves first, and the head must not be judged useless because the writer is a few milliseconds behind.
* **Restore on a miss in both host stores.** `effective_hit` counts the host tier's consecutive blocks *and* the NVMe's; a checkpoint missing from the store is read back and filed as its own (`PrefixRegistry.adopt`; the trim then checks its token ids like any other). The NVMe-resident blocks of a restore are read into **host-tier records** (backend `host`: NVMe -> slab slot, digest-checked on the reader threads), after which the restore is the existing code, unchanged, with every digest it already checks. In the sliced restore the NVMe blocks are read a slice at a time (`fill`), inside the slice's time budget.
* **Never worse than a miss.** A block that is gone, expired, fails its digest, has no room, or a device that errors is a miss and the turn prefills cold (the model verifies it against the cold chain in every test). Three device errors in a row latch the NVMe level off (`nvme_latched`); the host tier goes on without it.
* **Restart.** The index is a snapshot plus a journal on the drive; a new process opens the device, replays, drops entries of another layout fingerprint, and restores what was spilled. What is only in host RAM at the moment of a restart is lost, unless write-through put it on the drive first (section 5).

Code: `scripts/ci/qwen_nvme_store.py` (the store), `scripts/ci/qwen_kv_nvme.py` (flags, fingerprint, codec, backends, the tier), hooks in `qwen_prefix_registry.py` and `qwen_prefix_scheduler_patch.py` (docstring item j), `serving_c2_contract.py` (names, gate, waiver), `prefix_markers.py`, `qwen_prefix_metrics.py`. `nvme_store_bench.py` is a micro-benchmark of the store for the target drive.

## 2. The store (`qwen_nvme_store`)

```
 [0, 4 KiB) [4 KiB, 8 KiB)   two superblock slots: magic, generation, geometry, the active snapshot slot, the journal epoch, crc32
 snapshot slot 0 | slot 1     the whole index; written to the slot that is not active, then the superblock flips to it
 journal                      records since the snapshot: PUT (96 entries a record), DEL, DEL_TENANT, WIPE, TOUCH; each (epoch, seq, crc32)
 data region (from a MiB)     the objects, 4 KiB-aligned extents from a coalescing first-fit allocator (rebuilt from the index at open)
```

* **O_DIRECT, aligned.** Every device access is a whole number of 4 KiB blocks at an aligned offset, enforced for a plain file too (so the CPU tests find the bug a raw namespace would). A block is 2,228,224 bytes = 544 x 4,096, so a slab slot is already aligned and a run of blocks is one `preadv`; anything else bounces through an aligned scratch.
* **Crash safety.** A put is: write the data, `fdatasync`, append the journal record that names it, `fdatasync`. A delete is the other way round: the DEL record is durable before the extent is allocatable again, and an extent a reader has pinned waits for the unpin. A snapshot writes the inactive slot, syncs, then writes the superblock (generation + 1, into the slot the older generation does not occupy) and syncs. A journal group torn mid-write is cut at the first record whose crc, epoch or sequence does not check. The result after any crash: the old state, the new state, or (for a multi-record batch) a prefix of it - the blocks of a session in index order, so a shorter restorable session. `test_qwen_nvme_store.CrashTests` kills the process at **every device call** of a put, a multi-record put, a delete (with immediate reuse of the freed extents), a tenant delete, a wipe, a snapshot and a journal wrap, on a simulated disk with a volatile write cache (a random subset of the writes since the last sync, and 512-byte torn pieces of them, reach the media), and checks that the reopened store lists only entries whose bytes verify, with no overlapping extents. A mutation that drops the data sync is caught by it.
* **A put that fails.** Before the journal was touched nothing names the extents, so they are given back; once it was, a record may be on the media whatever the error says, so the extents are *not* given back until the next open rebuilds the allocator from the index (a surviving record must never name bytes something else has overwritten). The writer retries a failed batch once, then counts it; three failures in a row latch the NVMe level off.
* **Format safety.** The store writes to a device only if it carries its own superblock, or its first MiB is all zero, or `QWEN_PREFIX_NVME_FORMAT=1` says to destroy whatever is there. A device with the store's magic and no valid superblock is *corrupt*, not blank. The device is locked exclusively (`flock`) while open: two engines on one namespace would corrupt it.
* **Entries.** Kind (KV block / checkpoint), key (the host tier's block hash; the checkpoint's boundary block hash), extent, size, the tenant tag as an 8-byte blake2b (the tag itself never reaches the device), the chain (the prompt's first block hash), the index (block number / checkpoint position), created, last used, sha256 of the bytes, the layout fingerprint. About 160 bytes each.
* **TTL, age, LRU, floor.** `QWEN_PREFIX_NVME_TTL_S` (default 7 days) is counted from the last use (a restore, or a re-spill of a held block); `..._MAX_AGE_S` (default off) is an absolute cap from the write, for a retention policy that must hold whatever the use. The LRU order is the order of last use (persisted by the snapshot and by TOUCH records at most once a minute). When the tier is full a put evicts the least recently used, never a pinned entry. The free-space floor (`..._FREE_FLOOR_BYTES`, default 5% of the data region) keeps contiguous room for the allocator and over-provisioning for the drive. `..._MAX_BYTES` caps the tier below the device.
* **Deletion.** `delete_tenant` (one journal record), `wipe` (one record; the index is cleared wholesale), `drop_where` (the fingerprint sweep). Logical deletion is immediate and durable; the bytes stay on the media until they are overwritten. `QWEN_PREFIX_NVME_DISCARD=1` asks the drive to discard (TRIM) freed extents.

## 3. Keys, the layout fingerprint, integrity

* **Key.** The host tier's own: vLLM's chained block hash, which carries the tenant's `cache_salt`. Tenants never meet. The checkpoint is keyed by its boundary block's hash, in its own kind (the same hash names the KV block and the checkpoint).
* **Fingerprint** (8 bytes of a sha256 over): payload layout (`packed-v1`: the host tier's raw packed block, `[cache tensor][chip][slice]`), model repo and revision (`base` when none is selected), the serving profile, the tier IO's KV layout string (cache count, chips, slice bytes, dtype, page dims, GDN dtype), the block bytes, the page size (64), the chunk (2,048), the mesh (`tp<chips>`), the GDN state dtype, the checkpoint format, and the **code version** (`QWEN_PREFIX_NVME_CODE_VERSION`, else the image's `source-revision` label file; the tier refuses to start if it can tell neither). An entry whose fingerprint differs is dropped at open (`nvme_fp_drops`, one line) and never restored. **Every image build changes the code version**, so a deploy empties the tier unless the operator pins a version across images known to be KV-compatible (an owner decision, open question 6).
* **Digest.** sha256 of every object, taken on the writer's hash threads and checked at restore on the reader threads right after each run lands: `QWEN_PREFIX_NVME_VERIFY=sample` (the default: the first block of a restore and one in sixteen, like the host tier), `all`, `off`. A checkpoint is always checked unless `off`. A mismatch drops the entry (out of the index at once, its journal record on the writer thread), counts `nvme_digest_failures`, and the turn falls back to a cold prefill. `QWEN_PREFIX_NVME_AUDIT=1` (gate only) reads every object back after its write and compares, and forces `all`.

## 4. The restore interface and its two backends

`qwen_kv_nvme.RestoreBackend` has two methods and a flag:

| | `host` (built) | `p2p` (stub) |
| --- | --- | --- |
| `direct` | `False` | `True` |
| `read_to_host(tier, entries, views)` | NVMe -> the slab slots the caller reserved (reader threads, merged runs, digest checks) | - |
| `restore_to_device(tier, entries, block_ids, io)` | - | NVMe -> the device blocks the pool allocated, no host copy |

`TierHooks` asks `backend.direct`: a host backend promotes the NVMe-resident blocks to host records and the existing upload writes them; a direct backend takes the NVMe-resident blocks to the device blocks the pool allocated for them (`TierHooks.load_direct`; the host-resident blocks are written as before; the sliced restore is not offered to a direct backend and falls back to the whole restore). `load_direct` is tested with an in-memory double (`test_qwen_kv_nvme.DirectBackendTests`), so the seam a P2P backend plugs into is exercised. `QWEN_PREFIX_NVME_RESTORE=p2p` is refused at start (`P2PBackend()` raises `NotBuilt`); its contract is in section 11.

## 5. Flags

All default off; all strict (plain digits, no sign, no exponent, no space); a name set without `QWEN_PREFIX_NVME=1` is refused as inert; the switch needs the host tier (`QWEN_PREFIX_HOST_TIER_GIB`), `..._DEVICE` and the acknowledgement. Parsed by `qwen_kv_nvme.parse`, refused by the engine at start (`ValueError`) and by the contract in a profile.

| Name | Default | Meaning |
| --- | --- | --- |
| `QWEN_PREFIX_NVME` | off | `1` turns the tier on |
| `QWEN_PREFIX_NVME_DEVICE` | - | the raw namespace (an absolute path); a regular file works for tests |
| `QWEN_PREFIX_NVME_RETENTION_ACK` | - | **required** `1`: the operator confirms the owner's retention decision (all tenants, up to the TTL) and that this node does not advertise zero-retention (section 9) |
| `QWEN_PREFIX_NVME_TTL_S` | 604800 | seconds since last use after which an object is gone (1 to 366 days) |
| `QWEN_PREFIX_NVME_MAX_AGE_S` | 0 (off) | absolute cap, seconds since written |
| `QWEN_PREFIX_NVME_MAX_BYTES` | 0 (all) | most bytes the tier holds (at least 1 MiB) |
| `QWEN_PREFIX_NVME_FREE_FLOOR_BYTES` | 5% of the data region | bytes of the data region never used |
| `QWEN_PREFIX_NVME_VERIFY` | `sample` | digest check at restore: `sample`, `all`, `off` |
| `QWEN_PREFIX_NVME_RESTORE` | `host` | `host`; `p2p` is refused (not built) |
| `QWEN_PREFIX_NVME_QUEUE_BYTES` | 2 GiB | bytes waiting for the writer; also the number of spare slab slots mapped (resident only while in use) |
| `QWEN_PREFIX_NVME_THREADS` | 4 | reader and hash threads |
| `QWEN_PREFIX_NVME_SPILL` | `evict` | `evict`: write on host-tier eviction (the specification). `through`: also copy a session toward the drive once idle |
| `QWEN_PREFIX_NVME_THROUGH_BLOCKS` | 32 | `through` only: most device blocks one scheduler step copies |
| `QWEN_PREFIX_NVME_THROUGH_IDLE_S` | 60 | `through` only: seconds a finished request must be idle before its blocks are copied |
| `QWEN_PREFIX_NVME_DISCARD` | 0 | `1`: TRIM freed extents |
| `QWEN_PREFIX_NVME_CODE_VERSION` | the image's `source-revision` | a short token naming the code in the fingerprint |
| `QWEN_PREFIX_NVME_FORMAT` | 0 | **gate only.** `1`: initialize the device whatever it holds |
| `QWEN_PREFIX_NVME_AUDIT` | 0 | **gate only.** read back every object after its write; digests `all` |
| `QWEN_PREFIX_NVME_CMD_DIR` | - | **gate only.** a directory polled for `wipe` and `tenant.<tag>` files |
| `QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION` | 0 | **gate only.** start the tier although a zero-retention signal is present (section 9) |

**Contract** (`serving_c2_contract`): a profile that names any of them needs the host tier; a **traffic** profile may name `QWEN_PREFIX_NVME=1` only under an owner traffic waiver that names it (`NVME_WAIVABLE_LEVERS`), next to the host tier's own waiver; the four gate-only names stay a gate's even under the waiver (a baked zero-retention override would defeat the guard on every node the image runs on, so overriding in production is a contract change, a decision); an inherited value is dropped under a profile that does not name it. No profile names the tier yet: profiles are generated, and the gate profile twins are part of the gate work (section 12).

**Write-through** (`through`). Two parts, both bounded and neither on a decode round's critical path beyond a small slice: (a) every record that enters the host tier is queued to the writer at once (it is *held* in the tier until written: never a victim, its slot kept if it is removed meanwhile); (b) at the end of a `schedule()` call, `TierHooks.copy_idle` reads up to `..._THROUGH_BLOCKS` still-cached device blocks of requests that have been idle for `..._THROUGH_IDLE_S` into the host tier (the device idle, nothing yet written for the step: `flush`'s own moment; blocks cached in this step, blocks no checkpoint reaches and blocks already held anywhere are skipped), and (a) queues them. A warm copy is on its way before any eviction, and a restart loses only the last idle seconds. Cost: section 8.

## 6. Metrics and log lines

Counters `qwen_prefix_nvme_*_total` (ms timers as `_seconds_total`) and gauges, present in the registry snapshot only while the tier is configured: `nvme_spill_{offered,blocks,bytes,ms,batches,known,dropped_useless,dropped_queue,dropped_full,failures}`, `nvme_ckpt_{spilled,spill_bytes,spill_failures,spill_dropped,restored,restore_bytes,restore_ms,restore_failures}`, `nvme_restore_{requests,blocks,bytes,ms,refused_room,refused_digest,failures}`, `nvme_miss_blocks`, `nvme_digest_{checks,failures}`, `nvme_evicted_lru`, `nvme_expired_{ttl,age}`, `nvme_fp_drops`, `nvme_deleted_tenant`, `nvme_wipes`, `nvme_audit_{reads,mismatches}`, `nvme_latched`, `nvme_recovered_{entries,records}`, `nvme_recover_dropped`, `nvme_journal_records`, `nvme_snapshots`, `nvme_store_{hits,misses}`, `nvme_discarded_bytes`, `nvme_through_{queued,known,dropped_queue,copied_blocks,copied_bytes,copy_ms,copy_steps,failures}`, `nvme_spill_retries`, **`nvme_refused_zero_retention`**, **`nvme_unavailable`**; gauges `nvme_{on,bytes,entries,kv_entries,ckpt_entries,capacity_bytes,free_bytes,queue_bytes,journal_used_bytes,largest_free_bytes,through_backlog_blocks}`. The periodic `tier` line gains `nvme_bytes= nvme_entries= nvme_spilled= nvme_restored= ...`.

```
[PINDIAG] prefix: nvme tier open device=<path> backend=host ttl_s=<n> ... entries=<n> kv=<n> ckpt=<n> recovered_records=<n> fp_drops=<n> verify=<mode> ... spill=<evict|through> retention_ack=1 fingerprint=<hex>
[PINDIAG] prefix: nvme tier on backend=host ttl_s=<n> max_age_s=<n> verify=<mode> audit=<0|1> queue_bytes=<n> capacity_bytes=<n> spill=<mode> through_blocks=<n> through_idle_s=<n> fingerprint=<hex>
[PINDIAG] prefix: nvme tier spill blocks=<n> bytes=<n> ckpts=<n> ckpt_bytes=<n> ms=<f> queue=<n>
[PINDIAG] prefix: nvme tier restore req=<id> blocks=<n> bytes=<n> from=<tokens> to=<tokens> backend=host verify=<mode>
[PINDIAG] prefix: nvme tier checkpoint restored pos=<n> bytes=<n> ms=<f>
[PINDIAG] prefix: nvme tier REFUSED to start on a zero-retention node: <signals>; ...        (counter nvme_refused_zero_retention = 1, nvme_on = 0)
[PINDIAG] prefix: nvme tier UNAVAILABLE, the host tier serves without it: <error>            (counter nvme_unavailable = 1)
[PINDIAG] prefix: nvme tier latched off: <why> | spill failed | record failed its digest | ...
```

`prefix_markers.scan` reads them into `nvme_open`, `nvme_on`, `nvme_spills`, `nvme_restores`, `nvme_ckpt_restores`, `nvme_trouble`. A gate that asks for the tier needs `nvme_on` and an empty `nvme_trouble`.

## 7. Tests (CPU, stdlib only, `qwen-integration-cpu.yml` line `# NVME`)

`test_qwen_nvme_store` (store, on a sparse file with the real alignment rules and a simulated volatile-cache disk): alignment, the format policy, round trip through reopen, TTL / age / touch persistence, LRU and the floor, `max_bytes`, index capacity, piecemeal placement under fragmentation, tenant delete and wipe durability, TRIM calls, pins, journal wrap, the device lock, crash at every device call (put, multi-record put, delete with reuse, tenant delete, wipe, snapshot, wrap), concurrent readers and a writer, the bench tool.

`test_qwen_kv_nvme` (the tier under the real scheduler graft on the host-tier tests' fake world, whose model checks every restored checkpoint against the cold chain): flags and the contract; flag-off identity (pinned digest) and "nothing exists when off"; the checkpoint codec (bytes, tensors, windows, preconverted items through the model's own conversion); the P2P stub; spill and restore of mixed host/NVMe sessions, restart persistence, the scheduler never waiting for the drive (the writer's lock held for the whole scenario), a full queue, no double writes; TTL, age, LRU on a tiny device, a full device; fingerprint drops, digest and sampled digest, a corrupt checkpoint, device errors and the latch, the audit; a crash in the middle of a spill; tenant delete, wipe, `reset_prefix_cache`, both kill switches, the command directory; a restore beside a busy writer; the sliced restore (and its abort); the direct-backend seam; the markers and metrics; write-through (bounded steps, restart without an eviction, the idle delay, held records); the zero-retention guard and the acknowledgement; an unavailable drive; random traffic: exact, and the NVMe only adds hits.

## 8. Capacity, endurance and effects (ESTIMATE unless it says CPU)

Sizes (measured on the cards, `docs/prefix-store-hygiene.md`): a KV block is 2,228,224 bytes = 34,816 B per token. A 254k-token session is 3,841 blocks = 8.56 GB; the average production prompt (~59k tokens) is ~920 blocks = 2.05 GB; a checkpoint is ~107 MB (+ up to ~105 MB of carried drafter window). An average session image on the drive is ~2.2 GB.

**Capacity of a 1 TB namespace.** Metadata is 2 snapshot slots (device/2,048 = 488 MB each) + a journal (device/4,096 = 244 MB) + 8 KiB: 1.2 GB (0.12%). With the 5% floor about 950 GB of objects: **~430 average sessions** (~370,000 index entries, ~190 MB of host RAM for the index: ESTIMATE ~500 B per entry in CPython objects).

**Retention is capacity-bound before it is TTL-bound.** At the specification's spill rate of ~416 GB/day (~190 average sessions a day):

| spill rate (GB/day) | 100 | 136 | 200 | 416 | 800 |
| --- | --- | --- | --- | --- | --- |
| retention of the least recently used session (days) = min(7, 950 / rate) | 7 (9.5) | 7.0 | 4.7 | **2.3** | 1.2 |

To hold 7 days at 416 GB/day the namespace must be ~3.1 TB. At 1 TB the TTL is a ceiling the drive reaches only below ~136 GB/day (~62 average sessions a day); the real retention is "until the LRU gets to it".

**Endurance.** 416 GB/day is ~152 TB a year. A 600 TBW consumer-class 1 TB drive: ~3.9 years; an enterprise 1 DWPD / 5 year drive (1,825 TBW): ~12 years. Write amplification inside the drive should stay near 1 to 1.3 (large sequential writes, whole-extent frees, 5% free floor, optional TRIM): 152 to 198 TB a year. Journal and snapshot writes are negligible (~30 MB/day of journal; a ~60 MB snapshot every few days).

**Write-through costs.** `evict` writes a block when the host tier evicts it and the drive does not hold it; `through` writes every published block of every idle session once (blocks of a follow-up turn that the drive holds are not written again, so a conversation costs its new tokens). The specification's 416 GB/day is the volume of sessions leaving the cards; `evict` writes at most that and less when a session returns from host RAM first, `through` writes all of it plus the checkpoints (+5 to 10%): **~420 to 460 GB/day** at the stated load, i.e. the same endurance row as above, never more than ~10% above it. The device-read cost is on the scheduler thread: 32 blocks a step is ~13 ms at the 5.6 GB/s the host tier measured for a full-tier spill (T3), for as many steps as the backlog needs: ~29 steps for an average session, ~120 for a 254k one; against a decode round of ~150 ms that is a slowdown of the rounds that carry a slice by under 10%, and it never exceeds `QWEN_PREFIX_NVME_THROUGH_BLOCKS`. The Lever N decode-gap floor (8 s) is unaffected unless the step budget is raised by hand.

**Restore time (host backend).** The drive was measured at 6.9 GB/s sequential into host RAM at a Gen4 x4 link; the host tier's own restore to the device ran 8.9 GB/s (T3). Sequential (not yet pipelined): 59k ~0.30 + 0.25 = **~0.55 s**; 128k ~0.64 + 0.50 = **~1.1 s**; 254k ~1.24 + 0.96 = **~2.2 s**, against a cold re-prefill of 75 to 145 s (35 to 65 times). With `verify=all` the digests run on 4 reader threads at ~6 GB/s (1.5 GB/s each), about the drive's rate. The first token of a returning 254k turn: T3's host-tier return was 6.49 s (device-resident hit 4.65 s), so **~7.7 s** from the NVMe.

**Decode stalls.** *Spill:* the scheduler thread's cost is the sink: **2.5 microseconds a record on the CPU (mean over 143 offers on a busy host; p99 25 us; measured, no payload copy)**, ~1.3 ms for a 512-record flush; the writer thread does the rest. A burst larger than the queue is dropped, not waited for. *Restore, whole, in the trim:* the NVMe read adds to the host tier's restore inside one scheduler step: `docs/prefix-store-hygiene.md` estimates the longest wait of a whole restore at 254k at ~6.5 s (T3's returning first token, 6.49 s), so ~7.7 s with the NVMe read, **0.3 s under the 8 s floor**. Run the NVMe arms **sliced** (`QWEN_PREFIX_TIER_RESTORE_SLICE`): the host tier's sliced estimate is ~5 s, and the read is spread over ~12 slices of about 0.1 s. This is the one number in this note that looks too close to leave to the whole-restore path.

**Checkpoint adoption.** A returning session whose checkpoint the store lost reads it back on the scheduler thread, once, in the trim (or in `start` of a sliced restore): the read is ~107 to 215 MB (16 to 32 ms at 6.9 GB/s), the digest ~70 to 140 ms at 1.5 GB/s, and the model's own conversion of the 96 tensors ~60 ms (measured at capture), so **~0.1 to 0.25 s** (ESTIMATE) of one scheduler step per returning session, not per block.

**Host RAM.** The writer queue (KV blocks and checkpoints together) is bounded by `QWEN_PREFIX_NVME_QUEUE_BYTES` (2 GiB): that many bytes of spare slab slots are mapped beyond the host tier's 48 GiB, resident only while records are in flight (peak +2 GiB, the checkpoints in the queue being objects the store already dropped), plus ~190 MB of index and one transient checkpoint buffer (107 to 215 MB). The 48 GiB budget the owner approved does not include these (open question 3).

**Open time.** Reading and parsing a ~60 MB snapshot of ~370,000 entries and replaying a journal: ~1 s (ESTIMATE; the first thing `nvme_store_bench.py` should time on the real drive). The fingerprint sweep and a first TTL sweep follow on the writer thread.

## 9. Retention, privacy, and the zero-retention conflict

**What is kept.** The drive holds KV derived from tenants' prompts (and the GDN state at 2,048-token boundaries, and drafter window rows): data that can in principle be turned back into text. It is plaintext on the media unless encrypted (below). Keys carry the tenant's salt; the tenant tag is stored as an 8-byte hash; no salt, tag or token is logged or exported.

**Policy (owner decision 2026-10-21).** All tenants. TTL 7 days since the last use. There is no allowlist. An absolute cap (`..._MAX_AGE_S`) exists for a policy that must bound the age of the data whatever its use (open question 1). A tenant's data can be deleted by `NvmeTier.delete_salt(salt)` / `delete_tenant(tag)` (one durable journal record; admin thread) or a `tenant.<tag>` file in the command directory (gate only), the whole tier by `NvmeTier.wipe()` (also what a `reset_prefix_cache`, the prefix kill switch and the host tier's kill switch do). Deletion is logical and immediate; the bytes remain on the media until overwritten. `QWEN_PREFIX_NVME_DISCARD=1` issues TRIM, which the drive may honour late or not at all. Strong erasure is a crypto-erase or a full secure-erase of the namespace.

**The conflict.** A node whose agent holds an attestation key advertises `zero_retention` as a node label, gates no-retention work onto itself, and signs a teardown confirmation each time it destroys an inference container: *nothing survived*. A tier that keeps KV on a drive for days after the container is gone makes that statement false for any node it runs on.

**The guard.** The serving process **cannot know** what its node agent advertises: the label and the signature belong to the agent, which tells its containers nothing today (and must never put its key in one). So there are two layers:

1. **Signals the process can see** (`zero_retention_signals`): the agent's attestation key variable present in the process (a conflict by itself); `THATCH_ZERO_RETENTION=1` (the name proposed for the node agent to forward into the serving containers of an attesting node - **not set by anything today**); `THATCH_NO_LOG=1` (the container runs under the zero-retention log control); a `zero-retention` marker file in the kill-switch directory. Any of them present and `QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION` not `1`: the tier does not start, logs `nvme tier REFUSED to start on a zero-retention node: <signals>`, `nvme_refused_zero_retention` = 1 and `nvme_on` = 0 are exported, no byte of the drive is touched, no slab slot is mapped for it, and the engine serves with the host tier alone. With the override the tier starts and says so.
2. **A required acknowledgement**, because layer 1 is best effort: `QWEN_PREFIX_NVME_RETENTION_ACK=1` must be in every profile that names the tier (the contract refuses the profile otherwise, the engine refuses to start otherwise). It records that the operator and the profile's waiver accept retention for all tenants for up to the TTL and that the image is only placed on nodes that do not advertise zero-retention. **Placement is the platform's control, not this process's.** The follow-ups that close the gap are the platform's: the node agent forwards `THATCH_ZERO_RETENTION=1` to the serving containers of an attesting node, and the teardown attestation either excludes nodes that run the tier or lists it.

The override is a gate's own name: a traffic profile cannot carry it, and an inherited value is dropped (`apply_environment`). A drive that is missing, not the store's, or in use does not stop the engine: `nvme tier UNAVAILABLE`, `nvme_unavailable` = 1, the host tier serves.

**Encryption at rest.** Not built; the options:

| | drive self-encryption (Opal, one locked range per namespace) | software AES on the host path |
| --- | --- | --- |
| host CPU | none | AES-XTS/GCM at the drive's rate: ~2 to 4 cores for 6.9 GB/s (ESTIMATE) on the reader and writer threads |
| P2P (NVMe -> card) | **works**: the drive decrypts, the card receives plaintext | **cannot work**: the host never touches the bytes |
| per-tenant crypto-shred | no (one key per range) | yes, with a key per tenant |
| erase | revoke the key: the whole namespace | per tenant by dropping its key |
| integrity | the digest still applies on the host path | encrypt-then-MAC |

The recommendation is Opal for the namespace (it keeps the P2P path possible and costs nothing); per-tenant crypto-shred needs software keys and gives up P2P. This is the owner's decision (open question 2); the store's object layout (`write_buffers`, `read_run`) is the one place a software cipher would sit.

## 10. What it does not do

* A session in host RAM at the moment of a restart is lost in `evict` mode, and a session only on the device pool is lost in both modes until it is copied (`through`). A block evicted from the host tier is for a few milliseconds to seconds *in flight* (queued, not yet on the drive): a request for it in that window is a miss.
* A restore reads the host-tier-missing blocks, then the existing code uploads them: the two are sequential (a pipelined slice would overlap them: not built).
* It adds no new device op and changes no card path. The checkpoint codec reads the model's preconverted tensors with `ttnn.to_torch` and converts them back with the model's own `_qwen_prefix_preconvert` (the capture's call on the same values): exact in theory, **never run on a card**; the gate's exactness arm is its check.
* The host RAM tier's slab, the checkpoint store and the registry keep their sizes and policies; no NVMe knob changes them.
* **Merging with `tp4/upload-1`** (threaded tier IO, parallel digest checks): both branches start from `tp4/next-1` and both edit `PrefixRegistry.attach_tier_io`, `TierHooks.load` and `advance`, so the merge has textual conflicts and no semantic one: the NVMe tier takes whatever IO the registry holds (the codec and the fingerprint look through a wrapper's `.io`), its promotion runs before the digest loop the upload levers parallelise, and neither touches the other's counters.

## 11. The peer-to-peer follow-up (NVMe -> card, no host bounce)

Proven on 2026-10-21, outside this repository: a read from the namespace into a card's BAR4 window through the patched kernel driver at 6.95 GB/s, byte-exact. A usable window is about 3 GiB per card at a 1 GiB offset (the top of a 4 GiB window failed its self-test). What `RestoreBackend` with `direct = True` (`P2PBackend`) needs, as written in its docstring:

* **A staging window per card**, at most 3 GiB at the 1 GiB offset: one fill moves `floor(3 GiB / block_bytes)` = 1,445 blocks; a 254k session is three to four fills.
* **Alignment.** Every object is a whole number of 4 KiB blocks at a 4 KiB-aligned LBA, and the block is 544 pages, so a run of blocks is one aligned read; the window offset and every chunk stay 4 KiB-aligned.
* **A device-side scatter op**, e.g. `ttnn.qwen_scatter_blocks_from_window(cache_tensors, window, block_ids, block_bytes, slice_bytes)`: for each block in the window, copy its `[cache tensor][chip][slice]` pieces to the page range of `block_ids[i]` in every cache tensor on every chip (the inverse of `qwen_write_blocks_raw` without the host buffer). A chip's piece of a block is 17,408 bytes = 4.25 pages, so it is not page-aligned inside the block: either the scatter distributes the whole block to all four chips over the fabric, or a later layout version (chip-major, page-padded) rewrites the cache - a new `layout` in the fingerprint, which drops the old entries cleanly.
* **Integrity without the host.** The bytes never visit host RAM, so the sha256 cannot run on them: a device-side checksum over the window compared with a CRC stored beside the digest, or the drive's end-to-end protection information. Until one exists a P2P restore verifies nothing, and `QWEN_PREFIX_NVME_VERIFY` must say so.
* **Encryption** only at the drive (Opal).
* The same sliced hold the host path has, and the same fall back to a miss.

On the host side nothing else moves: `TierHooks.load_direct` is the call site, the entries are pinned for the length of the call, and the miss paths are tested.

## 12. Card gate plan (not run; minutes of card time per job, ESTIMATE)

**Prerequisites that are not built** (each is CPU work plus a review, none needs a card): (1) the job runner gives the engine container the raw namespace (a `--device` mapping and a `C2_NVME_DEVICE` key that becomes `QWEN_PREFIX_NVME_DEVICE`) - the container-launch path is the node agent's/the runner's, not this branch's; (2) gate profile twins and plans, generated, never hand-edited: the ship-5 candidate and its audit twin plus the `QWEN_PREFIX_NVME*` names, the retention acknowledgement and the override (the gate's host is a rig, not a zero-retention node, but may carry the signals), with the host tier's KV share cut to `QWEN_PREFIX_STORE_GIB + 4` (4 GiB, smaller than a 254k session's 8.56 GB, so every restore in the arm has to come from the NVMe); (3) `c2_prefix_gate --plan nvme-exact | nvme-timed | nvme-concurrent | nvme-restart`, built from the existing scenarios (`scenario_tier_upload` at 64k/128k/254k, `scenario_tier_concurrent`) plus one new scenario (restart in the middle); (4) a dedicated namespace, **formatted by the job** (`QWEN_PREFIX_NVME_FORMAT=1` in the gate profile only) - the owner must name the namespace and agree that the run formats it.

| Job | What | Cards | Stop / read rule |
| --- | --- | --- | --- |
| B0 | image build from the pushed head | - (40 min, while production serves) | `test_qwen_nvme_store` and `test_qwen_kv_nvme` green on the commit first |
| D0 | `nvme_store_bench.py` on the namespace (`--gib 64 --verify all` and `sample`, threads 1/4/8) | idle, 10 | seq read >= 5 GB/s, shuffled read >= 4 GB/s, write >= 2 GB/s, 0 failures; open time of the 64 GiB store |
| A1 | audited attach, one 64k session through the tier: `nvme tier on`, an `open` line with the fingerprint, no trouble, `nvme_audit_mismatches` 0 | 25 | stop on any `nvme_trouble` |
| N1 | **exactness**: `nvme-exact`, audited, 64k/128k/254k: build, push out of the pool *and* the 4 GiB host share, return from the NVMe, against the cold twin | 60 | every pair identical token for token; `nvme_digest_failures` 0, `nvme_ckpt_restored` >= 1, restore lines `from=0`; stop on the first difference |
| N2 | **timed restore**, unaudited, `verify=sample`: same sizes; arms host tier (48 GiB share) vs NVMe (4 GiB share): TTFT of cold / device-resident / host return / NVMe return | 2 x 35 | NVMe return at 254k <= ~8 s; the figures replace the ESTIMATE rows in section 8 |
| N3 | **spill under 8-live load**: seven seats + one, floods that push sessions to the NVMe (writes at the drive's rate), `evict` and `through` | 2 x 40 | no seat waits longer than the 8 s floor in any window; round p99 beside the control within the host tier's own TC numbers; `nvme_spill_dropped_queue` read, not judged |
| N4 | **concurrent restore**: `tier-concurrent-timed` ABAB, control = host tier, arm = NVMe, whole and sliced (`QWEN_PREFIX_TIER_RESTORE_SLICE=512`) | 4 x 45 | whole-restore wait vs the 8 s floor (the estimate says 0.3 s of margin); the sliced arm must clear it |
| N5 | **restart persistence**: serve S, push it to the NVMe, stop the container gracefully, start the same image, return S (a), repeat with the container killed (SIGKILL) during a spill (b) | 2 x 40 | (a) S restored from the NVMe, identical to its cold twin; (b) the store opens, every entry verifies, S is a hit or a cold turn, never a wrong answer |
| N6 | `through`: idle copy visible (`nvme_through_copied_blocks`), a session never evicted survives a graceful restart, per-step cost `nvme_through_copy_ms / nvme_through_copy_steps` | 30 | mean step cost <= 20 ms |
| Z | reset | 5 | |

About 380 minutes of cards after the build (D0 and B0 need none). Read rules: identical means token-identical to the cold twin, never a similarity; a digest failure or an audit mismatch anywhere is a stop, not a note.

## 13. Open questions for the owner

1. **TTL since last use, or an absolute age?** As built a session used daily lives as long as it is used. If the policy is "no prompt-derived data on a drive older than 7 days", set `..._MAX_AGE_S=604800`; a session older than that is then cold.
2. **Encryption at rest:** Opal for the namespace now (keeps P2P possible, no per-tenant shred) or software keys (per-tenant shred, no P2P)?
3. **Host RAM:** the queue's 2 GiB and the index (~190 MB) are outside the 48 GiB budget. Accept, or shrink `QWEN_PREFIX_NVME_QUEUE_BYTES`?
4. **`through` in production?** Its write cost is section 8; it is what makes a restart not lose the last hours. It is a flag, default off.
5. **Capacity.** At 416 GB/day a 1 TB namespace retains ~2.3 days, not 7. Buy 3 TB, or accept "until the LRU gets to it"?
6. **The code version in the fingerprint** empties the tier at every deploy. Pin it across KV-compatible images (needs a qualification record that says which images are), or accept?
7. **Zero-retention placement.** The guard and the acknowledgement are in; the node agent forwarding `THATCH_ZERO_RETENTION=1` and the teardown attestation's wording are the platform's work.
8. **Gate prerequisites** (section 12): who writes the runner's device mapping and the plan wiring, and which namespace the gate may format.
