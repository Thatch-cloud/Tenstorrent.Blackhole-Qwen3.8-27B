# tp4/p2p-host: the peer-to-peer KV restore, host side

Status: **built and tested on CPU against fakes, never run on a card, default off** (`QWEN_PREFIX_NVME_P2P`, a gate's name in the contract until a card has qualified it). Branched from `tp4/kv-nvme-1`
(`0f644692`); the kernel patch and the P1 probe come from `tp4/p2p-nvme` (`patches/tt-kmd-p2pmem`, `scripts/p2p`). The device ops (the staging buffer, the checksum, the copy into the cache) are another branch's
(`ttnn.qwen_p2p_staging_alloc` ...); this branch is everything the host does around them.

Owner, 2026-10-30: "with nvme we want the p2p working". The path:

```
 raw NVMe namespace --O_DIRECT read--> p2pmem (the card's BAR4 window) ---> staging buffer in card DRAM --device op--> paged KV cache
        (the drive's PCIe DMA, no host RAM)        (TLB window aimed at the staging bytes)        (checksum first, then the copy)
```

Everything that is not a CPU measurement is labelled **UNTESTED** and listed in section 10. P1 (2026-10-21, outside this repository) proved the first two arrows: 6.95 GB/s from the drive into a card's DRAM at
Gen4, byte-exact, to an x16 card and to an x4 card behind the switch, no AER, no NVMe timeouts.

## 1. The object: what the contract said, what the host tier really holds, and the layout this branch defines

The contract (INTERFACE.md section 1) assumed that "the P2P object for card d is data_t[d] (len(blocks) x slice_bytes) concatenated over the cache tensors, the same bytes the host tier already holds". **That is
not what the host tier holds.**

* **The host tier's record** (`_QwenKvTierIO.read_blocks`, `qwen_prefix_model_patch.py`) is **one payload per block**: `[cache tensor][chip][slice]`, all four chips interleaved, `slice_bytes` = 17,408 per
  (tensor, chip) at one KV head per chip (16 bf8 tiles of 1,088 bytes), 32 tensors, 4 chips: **2,228,224 bytes** a block. The NVMe tier (`qwen_kv_nvme`, layout `packed-v1`) stores exactly that payload,
  one object per block, keyed by the block hash, with the sha256 of the payload.
* **Why a card cannot read its own share of it.** Chip d's bytes of cache tensor t sit at `(t * 4 + d) * 17,408` inside the object. 17,408 = 4.25 pages: no chip's slice starts or ends on the 4 KiB boundary
  that `O_DIRECT` needs, and a slice is interleaved with three others. A read of the whole object into one card's staging buffer would put all four chips' bytes in one card and read every byte four times from
  the drive.
* **The contract's per-run, tensor-major object cannot be read in one aligned run either**, and the store's unit is the block (entries come and go one block at a time: write-through, eviction, TTL, a
  partial prefix restored).

So this branch changes what the drive holds, behind the flag, as the contract allowed ("if the per-card object layout changes the stored bytes, give it a new `layout` in the fingerprint"):

**The P2P layout `chipmajor-v1`** (`QWEN_PREFIX_NVME_P2P=1`): the same block, the same bytes, with the chip axis first: `[chip][cache tensor][slice]`.

```
 object (one KV block)      = shard 0 | shard 1 | shard 2 | shard 3                 2,228,224 bytes = 544 x 4,096
 shard d (chip d's share)   = cache tensor 0's slice | tensor 1's slice | ... | tensor 31's slice    32 x 17,408 = 557,056 bytes = 136 x 4,096
 chip d's share of block b  = bytes [d * 557,056, +557,056) of block b's object
```

* Every shard starts and ends on a 4 KiB boundary of the object, and the object starts on one (the store's extents are 4 KiB-aligned): **one `O_DIRECT` read per card per block**, each landing in that card's
  staging buffer. A restore of n blocks is 4 n reads of 557,056 bytes.
* The store entry keeps the **sha256 of the stored bytes** (what the host path checks) and now carries, beside it, `aux` = **four little-endian u64, one `p2p_checksum` per shard** (section 3), recorded by the
  spill thread. `qwen_nvme_store` entries gained an optional auxiliary field (a flag bit and a length byte after the fingerprint; an entry without it packs byte for byte as before; at most 64 bytes).
* The **fingerprint's `layout` component** is `chipmajor-v1` instead of `packed-v1` when the flag is on, so a drive written one way is dropped (never restored) by an engine of the other: **switching the flag
  clears the warm tier**. The knobs (`..._P2P_THREADS`, `..._P2P_STAGING_MIB`, `..._P2PMEM`) are not in the fingerprint. Chip-major is chosen by the flag alone, not by whether the path started at boot: an
  engine whose P2P path cannot start (say, a reboot returned the stock module) keeps reading and writing the same layout through the host backend instead of discarding a day of warm cache.
* **The host backend reads chip-major objects too** (it is the fallback): `NvmeTier._read_many` reads a few blocks at a time into aligned scratch, checks the sha256 of the stored bytes, and converts to the
  host tier's `[cache][chip][slice]` with `ShardLayout.from_chipmajor`. The spill thread converts the other way (`_chipmajor`, into one page-aligned arena per batch, so the device write needs no bounce); batches
  are capped at 128 MiB in this layout because the conversion needs a second buffer. GDN checkpoints are not touched.

**What the device ops must therefore do (changes to INTERFACE.md sections 3 and 4; the host adapter, `qwen_p2p_restore.StagingApi`, is written to this):**

1. **Staging layout.** A chunk of m blocks lands in a half of the staging buffer **block-major**: `staging[block j][cache tensor t][slice]`, block j at `j * shard_bytes`, tensor t at `+ t * slice_bytes`.
2. `ttnn.qwen_write_blocks_from_staging(device_tensor, staging_handle, byte_offset, blocks, row_stride)` takes a new `row_stride`: row i of device d's `[len(blocks), slice_bytes]` is
   `staging_d[byte_offset + i * row_stride : + slice_bytes]`. The host calls it once per cache tensor t with `byte_offset = half + t * slice_bytes` and `row_stride = shard_bytes` (557,056 = 512 pages of 1,088
   bytes; `slice_bytes` = 16 pages, so every row starts on a tile page). With `row_stride = slice_bytes` it is the contract's op.
3. `ttnn.qwen_staging_checksum(staging_handle, byte_offset, length, count)` returns, per device, **`count` uint64**: the checksums of the consecutive ranges `[byte_offset + i * length, + length)`, each with k
   counted from its own first byte. The host calls it with `length = shard_bytes` and `count = blocks in the chunk`. (The checksum is not combinable from sub-ranges, so one value per block is the only way to
   check blocks individually; the per-block value is what the store holds.) `count = 1` is the contract's op. The return value is read as a (devices, count) matrix of integers (lists, a numpy array, a torch
   tensor).
4. `ttnn.qwen_p2p_staging_info(handle)` must name the card of each mesh device: `pci_bus_id` (a PCI address) **or** `pcie_device_id` (the N of `/dev/tenstorrent/N`), besides `noc_x`, `noc_y`, `address`
   (**4096-aligned**), `size`. The host resolves the card against the driver's own table at run time (the nodes renumber on every reset).
5. The staging buffer is **bank-contiguous inside one 4 GiB window and inside the published slice** (the host refuses anything else, `plan_aim`), 2 halves of `chunk_blocks x 557,056` bytes. P1 found the top of
   a 4 GiB window at bank 0 unusable (a 3 GiB transfer from the 1 GiB mark failed its canary): the staging address comes from the allocator and the canary checks both ends of it.

`qwen_staging_checksum` for a range of 557,056 bytes is the only per-block device cost besides the copy; both are streaming passes over DRAM.

## 2. The window: how a process takes it and aims it

The patched tt-kmd (`patches/tt-kmd-p2pmem`, module parameter `p2pmem_devices=<cards>`) publishes the first 4 GiB TLB window of the card's BAR4 (window id 202, BAR4 offset 0 unless `p2pmem_offset_mb` says
otherwise) as PCI peer-to-peer memory. It programs no TLB: **whoever uses the memory takes and aims the window with the driver's ordinary ioctls on `/dev/tenstorrent/<N>`**, and keeps the fd. The host side
(`qwen_p2p_restore.TtLink`, the P1 probe's calls with the engine's lifetime) does, per card, in this order:

| step | call | notes |
|---|---|---|
| find the card | `/sys/class/tenstorrent/tenstorrent!N/device` -> the PCI address | at run time, from the staging record's `pci_bus_id` / `pcie_device_id` |
| read the slice | `.../tenstorrent!N/p2pmem/{bar,offset,size,bus_addr,tlb_first,tlb_last}` (root) and `/sys/bus/pci/devices/<bdf>/p2pmem/{size,available}` | refuse unless `available == size` (nobody holds the pool), BAR 4, one window |
| power | `SET_POWER_STATE` (`_IO(0xFA, 15)`), busy flags `MRISC_PHY_WAKEUP \| TENSIX_ENABLE \| L2CPU_ENABLE`, validity 4 | the DRAM PHY is asleep when no client has the card open; the state is this fd's and ends with it |
| take the window | `ALLOCATE_TLB` (`_IO(0xFA, 11)`, size 4 GiB) until the id equals `tlb_first`; `FREE_TLB` (12) the ones taken on the way | the driver hands out the lowest free window; if another process (the UMD) holds it, the path is unavailable |
| aim it | `CONFIGURE_TLB` (`_IO(0xFA, 13)`): `addr` = the staging address rounded down to 4 GiB, `x_end, y_end` = the staging bank's NOC tile, `noc` 0, `ordering` relaxed | byte x of the slice is at BAR4 `offset + x`, window offset `offset + x - window_base`, NOC address `addr + that` |
| pin it | `EXPORT_TLB_DMABUF` (`_IO(0xFA, 16)`), nothing imports the export | keeps the window past `FREE_TLB`/`close()` and makes the driver refuse a destructive `RESET_DEVICE` |
| prove it | `mmap(fd, 4 GiB, offset = mmap_offset_wc)`: write a pattern at both ends of the staging buffer through the window, read it back | the canary; before any DMA; the CPU path is slow (the kernel maps p2pmem uncached) but it touches 8 KiB |
| map the slice | `open(/sys/bus/pci/devices/<bdf>/p2pmem/allocate, O_RDWR)` + `mmap` of the **whole** slice | one allocation: a second mapper gets ENOMEM (the lease); mapping from the pool's start puts byte x of the mapping at byte x of the slice; if `/proc/self/pagemap` can say (CAP_SYS_ADMIN) the first page, the staging start and its last page are checked against the BAR |

The staging buffer's place in the slice is arithmetic (`plan_aim`, equal to the probe's `make_plan` in a test): staging address A, slice at BAR4 offset S in window W (window_base = (W - 202) x 4 GiB):
window offset = A mod 4 GiB, slice offset = that - (S - window_base), which must be >= 0, 4096-aligned and inside the slice.

While the engine lives, `p2pmem/available` is 0 for each card, a destructive `RESET_DEVICE` returns EBUSY, and system suspend is refused (the patch's guards): **the card-reset flow must stop the engine first**, as
it already must.

## 3. The checksum

`p2p_checksum(bytes) -> int` (`qwen_p2p_restore.py`): words `x_k` = little-endian uint32 at byte `4k`; `w_k = (k * 2654435761 + 1) mod 2^32`; `S = sum(w_k * x_k) mod 2^64`, k from the first byte of the range.
numpy: weights cached per length, 1 Mi words a pass (the products are below 2^64, the sum wraps). Three shared vectors (INTERFACE section 4) with their values in `scripts/ci/p2p_checksum_vectors.json`
(also pinned in the test against a plain-python reference):

| vector | definition | S |
|---|---|---|
| `zero_4096` | 4096 zero bytes | 0 |
| `mod251_4096` | byte i = i mod 251, 4096 bytes | 17453227578024978262 |
| `prng_1mib` | `numpy.random.default_rng(20261030).integers(0, 256, 1 << 20, dtype=numpy.uint8)` | 4942361704597259676 |

It is a weighted sum, not a cryptographic digest: it catches a flipped byte, a truncated or misrouted DMA, a stale staging buffer. The spill thread computes it for every shard of every block it writes (measured on a CPU host:
0.5 ms for the four shards of a block, and 0.07 ms for the chip-major conversion, numpy, off the scheduler thread; ~2 s of one core for a 254k session); the sha256 stays the integrity check of the host path, the audit and the scrubbers.

## 4. The read

`P2PEngine` (one per mesh) owns the staging buffer (`StagingApi.alloc`, once, for the engine's life), a `TtLink` per card, and a pool of `threads` reader threads. A restore is a `Stream`:

* chunks of up to `chunk_blocks` positions (half the staging buffer / 557,056 = 120 at the default 2 x 64 MiB; the sliced restore's block cap shrinks it);
* for every stored block in a chunk and every card, one task: `pread` (the store's `BlockFile.pread_into`: alignment checks, short-read loop, test hooks) of that card's shard, `entry.offset + d x 557,056`,
  straight into the card's p2pmem mapping at the staging slot; positions that need no read (a block the host tier holds) are gaps and keep their place;
* two halves: chunk k+1 is on the wire while chunk k is checked and copied; **a half the device has read is synced (`ttnn.synchronize_device`) before the drive's DMA writes into it again** (the drive ignores the
  device's queue); the dirty state is the engine's, not the stream's;
* a chunk is ready when all its tasks returned; a read error names the chunk; `close()` waits for the reads in flight (they cannot be taken back, and they write into the staging buffer).

Then, per chunk, on the scheduler thread: `verify` (one checksum op, the comparison, one retry after a device sync and 2 ms if it differs, because a posted write can trail the drive's completion; a second
difference raises `P2PMismatch` with the (block, chip) pairs) and `write` (one copy op per cache tensor). **Nothing is written for a chunk that failed.** Counters: `nvme_p2p_late_data` is a first
comparison that failed and the retry passed.

## 5. The backend: flags, fallbacks, counters

`QWEN_PREFIX_NVME_P2P=1` (with the tier's usual names; **gate-only** in `serving_c2_contract`: a traffic profile cannot carry it even under the owner waiver; promoting it is a contract change). Knobs, refused as
inert unless the switch is on: `QWEN_PREFIX_NVME_P2P_THREADS` (reads in flight over all cards, default 8, 1-64), `QWEN_PREFIX_NVME_P2P_STAGING_MIB` (per card, default 128 = two halves of 64, 2-4096),
`QWEN_PREFIX_NVME_P2PMEM` (the PCI addresses of the cards whose p2pmem is published; optional: when given the mesh must run on exactly these cards). `QWEN_PREFIX_NVME_RESTORE=p2p` is refused with a pointer to the
flag. All four cards need the slice published; fewer is `UNAVAILABLE` (the copy op writes every chip).

| situation | what happens |
|---|---|
| start: no staging op, no published window, a held TLB window, a failed canary, a staging buffer outside the slice, a card not found | one line `nvme tier p2p UNAVAILABLE, the host backend serves the restores ...`, `nvme_p2p_unavailable` = 1, the host backend; the engine is never stopped by it |
| the IO has no chip geometry (`caches`, `chips`, `slice_bytes`) | the tier does not open (`UNAVAILABLE`, `needs the tier IO's geometry`) |
| restore, checksums match | the blocks are in the cache; `nvme_p2p_restores`, `_blocks`, `_bytes`, `_read_ms`, `_check_ms`, `_write_ms` |
| **checksum mismatch after the retry** | **a miss**: nothing written for the chunk, the turn prefills cold, `nvme_p2p_mismatch` += shards. The first 8 blocks are re-read through the host path: bytes whose sha256 fail are bad on the drive (the entry is dropped, `nvme_digest_failures`); bytes that verify mean the path delivered them wrong (a strike; three latch the path off). It does not count toward the host tier's three-failures latch |
| any other failure (a read error, a device op that raises, a stalled drive) | counted (`nvme_p2p_errors`), a strike, and the same blocks come through the host path (`nvme_p2p_fallbacks`); if the host path fails too the restore is a miss |
| three strikes in a row, a device-op error (at once), a stalled drive (no chunk for 20 s) | the path latches off: `nvme tier p2p latched off`, `nvme_p2p_latched` = 1, `direct` turns False and `TierHooks` goes back to host records; until a restart |
| audit (`QWEN_PREFIX_HOST_TIER_AUDIT=1`, gates) | the restored blocks are read back through the model's own IO, converted chip-major and compared with the drive's sha256 (`TierHooks.audit_direct`): the end-to-end drive -> card -> cache -> host check |

Log lines (read by `prefix_markers`): `nvme tier p2p on cards= threads= staging_bytes= chunk_blocks= shard_bytes= mapping_verified= window=`, `nvme tier p2p restore blocks= bytes= ms=`, and the trouble lines `p2p
UNAVAILABLE`, `p2p latched off`, `p2p restore failed`, `p2p CHECKSUM MISMATCH`. The open line carries `backend=p2p`. The smoke table (`c2_smoke_check`) wants the `p2p on` line when a profile sets the flag.

`nvme_p2p_map_verified` is 1 when `/proc/self/pagemap` confirmed the mapping (needs CAP_SYS_ADMIN). Without it the guard is the whole-slice mapping: it succeeds only if every page of the pool was free, so the
mapping is the pool in order and the offset arithmetic holds.

## 6. Async restore: the answer

DESIGN section 0: `TierHooks.load` writes synchronously inside `schedule()`, so every seat stalls for the length of a restore (~1 s per 254k). A P2P restore moves the same bytes at the same speed, so run
inside `load_direct` it would stall just as long. **The minimal change that lets it run beside the other seats' decodes is the sliced restore that already exists** (`QWEN_PREFIX_TIER_RESTORE_SLICE`: a
`RestoreJob` advanced one slice per scheduler step by the one-fresh-prefill admission's hold), extended to direct backends:

* `P2PBackend.sliced = True`; `TierHooks.start` now marks the NVMe blocks `Pending` for it (before, a direct backend had no sliced restore and the trim restored whole);
* the first `advance` of a job begins a **stream** over the job's positions (`begin_stream`: pins the entries, starts the reads, chunks of at most the slice cap); **the reads run on the engine's pool, between
  steps**, not on the scheduler thread;
* each step, `advance_direct` takes the next chunk **only if all its shards have landed**; if not it does nothing and the pass goes on without the request (a decode-only pass: the seats keep getting
  rounds); if so it runs the checksum op, takes the chunk's blocks from the pool (nothing is allocated for a chunk that does not verify), writes the host-tier records of the chunk as before, enqueues the copy
  ops, and lets the next read start (after a device sync of the half);
* the only work left on the scheduler thread per step is the checksum op with its readback, the block allocation (with the existing `flush` of what it evicts), and the copy enqueue: estimated **ms to tens of ms
  per 120-block chunk** (the copy is 67 MB per card of DRAM-to-DRAM), against ~40 ms of drive read per chunk that is now off-thread. **UNTESTED/UNMEASURED**: P3a prints the check and copy times per chunk and
  the restore's end-to-end rate; G-C3 judges the disturbance.
* failures: a mismatch abandons the job (a miss; the request's tier hit is ignored); a read or device error before any block was taken hands the rest of the job to the host `fill` path
  (`job.host_only`); a path that latches under a job finishes it through the host path; a drive that lands nothing for 20 s abandons the job and latches the path; the trim reaching an unfinished job finishes
  it whole and **waits** for the reads (`advance(whole=True)`), as before.

It needs the slice flag **and** the admission wrapper that holds a request between slices (`hold_ready`); without them a direct backend still restores whole in the trim (`load_direct`), blocking for the
restore's length. **What must wait for a later step:** the device-side copy still runs on the same queue as the decode rounds (tens of ms of device time a 254k restore, not measured), and nothing here
changes `flush` (the device read of the blocks an allocation evicts), which is synchronous; both belong to the G-C3 concurrency gate.

## 7. Files

| file | what |
|---|---|
| `scripts/ci/qwen_p2p_restore.py` | `p2p_checksum`, `ShardLayout`, `plan_aim`, `TtLink`, `StagingApi`, `P2PEngine`, `Stream` |
| `scripts/ci/qwen_kv_nvme.py` | the flags, the `chipmajor-v1` layout, `P2PBackend`, `NvmeTier._start_p2p` / `_chipmajor` / `_read_many` / `_restore_via_host`, counters |
| `scripts/ci/qwen_nvme_store.py` | the entry's auxiliary bytes |
| `scripts/ci/qwen_prefix_scheduler_patch.py` | the sliced restore for a direct backend (`begin_stream`, `advance_direct`), `audit_direct` |
| `scripts/ci/tp4_p2p_probe.py` | P3a |
| `scripts/ci/p2p_checksum_vectors.json` | the three shared vectors |
| `scripts/ci/test_qwen_p2p_restore.py`, `test_qwen_p2p_tier.py`, `test_tp4_p2p_probe.py` | the CPU tests (named in `qwen-integration-cpu.yml` through `fusion-wp/NVME.json`) |
| `scripts/ci/references/tp4-p2p-jobs/` | the P3a job template |
| `patches/tt-kmd-p2pmem/`, `scripts/p2p/` | from `tp4/p2p-nvme`: the kernel patch, the P1 probe, the topology tool, the layout and bandwidth model |

`qwen_p2p_restore.py` is in the serving image's one copy list (`docker/qwen-c2-overlay.txt`, generated from `fusion-wp/NVME.json`; the P8 copy lists are not involved for the C2 image). `qwen_kv_nvme`
imports it only when the flag is on.

## 8. P3a: the standalone card probe

`scripts/ci/tp4_p2p_probe.py`, run by the `fabric` action of `qwen-c2-serving.yml`: **job key `C2_FABRIC_PROBE=p2p`** with `C2_CARDS=quad`, `C2_ACTIONS=fabric` (template
`scripts/ci/references/tp4-p2p-jobs/P3a-p2p-restore.env`; `C2_IMAGE_TAG` names an image with the device-op extension). The probe runs from the checkout the step mounts at `/c2`: an image, not a build of this
commit. Order, and what it refuses before it does anything:

0. **The drive, first.** Identity by model and serial through `/dev/disk/by-id` (a link `nvme-<model>_<serial>[_n]` that resolves to the node; a link of that name pointing elsewhere refuses) and,
   independently, the kernel's own model and serial for the node; the values come from the job's environment (`P2P_DRIVE_BY_ID`, `P2P_DRIVE_MODEL`, `P2P_DRIVE_SERIAL`, `P2P_CARDS`), which the step reads from
   **a private file on the runner** (`P2P_ENV_FILE`, owned by the runner user, not group- or world-writable, a strict `KEY=VALUE` parser, never sourced). The step checks the by-id name against the model and serial
   before `docker run`; the probe checks again inside. Then **R1's readings, printed**: `P3A R1 controller_state=live namespace_bytes=N`. **Anything but `live` (or an empty namespace) refuses, nothing written.**
   A drive carrying the NVMe tier's superblock refuses unless `--allow-store-overwrite`. The model, serial and PCI addresses are never printed (the Actions log of a public repository is public): the report has an
   8-hex fingerprint of model and serial.
1. The (1, 4) mesh as the kv-read probe opens it; the ops must exist (refusal otherwise); scratch KV caches allocated as production allocates its own.
2. N blocks (default 96, ~214 MB) of random bytes in the host tier's layout, laid out chip-major, written `O_DIRECT` at `--scratch-offset` (default 1 GiB), with their per-chip checksums.
3. Every published card's window taken, aimed, pinned, proved (engine construction; the staging description is printed without PCI addresses).
4. **One card first**: the widest link among the published cards (the x16 card), alone: its shards of the first blocks read into its staging buffer, its device checksums compared with the host's.
5. **Every published card**: the read on its own (GB/s drive -> cards), then the whole restore (checksum ops, `qwen_write_blocks_from_staging` into the scratch caches), then `qwen_read_blocks_raw` back and a
   byte-for-byte comparison with what was written to the drive. Printed: read GB/s, end-to-end GB/s, checksum and copy ms per chunk.
6. **Negative arm**: one byte of one chip's shard of one block flipped on the drive: the restore must be refused by the checksum, the cache block it was for must still be zeros, and after the byte is put back the same
   restore must pass.
7. The windows released and every `p2pmem/available` equal to `size`.

Last lines: `P3A verdict=PASS|FAIL|NOT-MEASURED ...` and one JSON object (kind `p2p-p3a-quad`); exit 0 / 1 / 2 (a refusal or a setup failure) / 3 (the watchdog, 30 minutes).

## 9. What production needs from the platform (for a software-factory ticket)

The node agent launches the serving container. For the P2P restore it must, in addition to what the NVMe tier already needs (the drive's by-id path as `QWEN_PREFIX_NVME_DEVICE`, the Opal status file, the
stop grace; `docs/tp4-kv-nvme.md` section 9):

1. **Devices.** `/dev/tenstorrent/*` for all four cards (already) and the cache drive's block node read-write (the tier writes it too).
2. **sysfs, writable where it must be.** The container's `/sys` is read-only. For **each card with a published slice**, the directory `/sys/bus/pci/devices/<bdf>/p2pmem` must be bind-mounted **read-write**
   (the engine opens `allocate` `O_RDWR` and maps it shared). `/sys/class/tenstorrent` (the `p2pmem` attributes are root-readable) and `/sys/bus/pci/devices/<bdf>/resource` must stay readable. The agent finds
   `<bdf>` at launch from the same table the engine uses (`/sys/class/tenstorrent/tenstorrent!N/device`); the nodes renumber on every reset.
3. **Capabilities.** None are needed for the data path (the ioctls need only the open fd; `O_DIRECT` and the mapping need root in the container). **Recommended on the first deployments: `CAP_SYS_ADMIN`**, so
   `/proc/self/pagemap` shows physical addresses and the engine can confirm where the mapping sits (`nvme_p2p_map_verified`); without it the whole-slice mapping is the guard.
4. **Environment.** `QWEN_PREFIX_NVME_P2P=1` (a gate profile today), optionally `QWEN_PREFIX_NVME_P2PMEM=<bdf>,<bdf>,...` (the runner's `C2_P2PMEM`) and the staging and thread knobs; plus the tier's existing
   names (`C2_NVME_DEVICE` -> `QWEN_PREFIX_NVME_DEVICE`).
5. **The kernel module.** The patched tt-kmd (`p2pmem_devices` naming **all four cards**; today two are published and a reboot returns the stock module) installed through DKMS or a boot unit, with
   `p2pmem_size_mb` (default one 4 GiB window; 64 MiB of host RAM per card). A node without it serves from the host backend.
6. **Reset ordering.** While an engine holds a card, `p2pmem/available` is 0 and a destructive `RESET_DEVICE` returns EBUSY: the agent's reset (and the ZR job) must stop the serving container first and treat EBUSY
   as "something still holds the card", never retry blindly. A container that cannot be stopped holds the pool.
7. **The window.** The engine needs TLB window 202 (the published one) free: confirm with the UMD's mappings (`/sys/kernel/debug/tenstorrent/*/mappings`) that the serving process holds none of the 4 GiB windows.
   Nothing else on the node may map the published range (the kernel maps it uncached).
8. **DRAM budget.** The staging buffer is `QWEN_PREFIX_NVME_P2P_STAGING_MIB` (128) per card, allocated by the device-op extension after the KV pool; the profile's DRAM ledger must leave it.
9. **Topology.** The drive and the cards must be on paths the kernel allows peer-to-peer on (P1: all `THRU_HOST_BRIDGE`, hairpin through the CPU measured fine at 6.95 GB/s); the drive must stay a plain
   NVMe namespace (no md, device-mapper, LVM or dm-crypt: they do not carry the peer-to-peer flag). A PCI rescan that takes the drive away (R1) latches the tier, as before.

## 10. What is untested until cards

Everything in the data path past the CPU tests: that a process in the serving container can take TLB window 202 while the UMD holds the card; the `ALLOCATE_TLB` / `CONFIGURE_TLB` / `EXPORT_TLB_DMABUF` call
sequence on a live card as `TtLink` makes it (the structs are the probe's, pinned equal in a test); the canary at the allocator's staging address; the bind mount of a sysfs `p2pmem` directory read-write
over a read-only `/sys`; the pagemap check inside a container; the O_DIRECT reads of 557,056-byte shards into p2pmem at 4 x 8 threads (P1 measured 16 MiB chunks; the per-read size here is smaller and the
drive's rate at it is unmeasured); whether a posted write ever trails the drive's completion (the retry exists for it); the device ops themselves and their `row_stride` / `count` forms; the copy's cost beside a
decode round; the disturbance of a 254k restore on eight live seats (G-C3); a reset with the engine stopped, and the EBUSY paths; the DRAM bank contention of a single-bank staging buffer; the `chipmajor-v1`
conversion cost on the spill thread and the host fallback's cost under load; the stock-module reboot (the path must come up `UNAVAILABLE`, not crash).
