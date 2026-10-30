# P2P KV restore, the device side: staging buffer, scatter op, checksum op

Branch `tp4/p2p-devop` (from `tp4/kv-nvme-1`). Scope: the device ops the host side needs to restore KV blocks from an NVMe drive that DMAs straight into card
DRAM (peer to peer, no host bounce). The drive writes into a **staging range** in one DRAM bank; a device op then copies the staged bytes into the paged KV
cache; a second op checksums the staged bytes so a mis-aimed or truncated DMA is a miss. This document is the record of how the staging range is placed (and what it
costs), what was built, what is checked on a CPU and what only a card can settle. **Nothing here has run on a card.**

## 0. Status

| | |
|---|---|
| Built | `qwen_kv_read.so` version 3 (the existing extension, grown): compiles and links against the serving tt-metal tree (9f9cd4fd), every non-Python symbol resolves. **The two ops now carry `row_stride` (copy) and `count` (checksum)** (section 2): a build older than that has neither keyword, and the host runs its slow compat mode against it (`docs/tp4-p2p-host.md` section 1) |
| Checked on a CPU | the arithmetic, the argument checks, **both device kernels executed against a host stand-in of the dataflow API** (random block sets, page sizes, banks, offsets incl. the end of the ring, launch splits, **rows apart by a stride, many checksum ranges in one launch**), a mutation check of that harness, the card test against an in-memory fake extension |
| Compiled with the real RISC-V toolchain | both kernels, front end only (`rv_kernel_check.sh`: the build container's sfpi g++ with the JIT's flags, `-Wall -Werror -ftt-no-dyninit`, behind the firmware's include prelude; negative controls fail it). The 64-bit product is `mul` + `mulhu`, and neither kernel calls a multiply or divide library routine. Nothing was linked or run |
| Not checked | the JIT's own link step, any execution on a device, the lockstep assumption (checked at run time by a `TT_FATAL`), NOC coordinates, throughput |

## 1. Where the staging range goes, and what it costs

Requirement (INTERFACE section 2): on every device, **one contiguous byte range in one DRAM bank** that the allocator never hands to anything else.

What tt-metal does (read in the source of the served tree, tag v0.77.0 = commit 9f9cd4fd; the headers and the allocator files are byte-identical to the build container's):

* The DRAM allocator (`BankManager`, `tt_metal/impl/allocator/bank_manager.cpp`) keeps **one free list for all banks**. `allocate_buffer(size, page_size, ...)` computes
  `size_per_bank` and asks that one free list for `size_per_bank` bytes; the address it returns applies to **every bank of the device**. So an allocation reserves
  the SAME address range in all 8 banks, whatever it holds, and a one-shard sharded buffer reserves just as much as an interleaved one. There is no single-bank reservation.
  On a mesh the allocation is made once for the mesh (`MeshBuffer`, "allocated in lock-step across all devices"), so the address is the same on all four cards.
* An interleaved buffer puts page `p` on bank `p mod 8` at `address + (p div 8) * page`. So **the pages of one bank are contiguous**: in an interleaved buffer of
  `8 x S` bytes with page size `P` (`P` divides `S`), bank `b` holds bytes `[address, address + S)` as one contiguous range.
* The allocator's range per bank is `[dram_unreserved_base, dram_view_size - trace_region_per_bank)` with `dram_view_size = 4,278,190,080` (0xFF000000, **4 GiB minus 16 MiB**,
  `blackhole_140_arch.yaml`). Everything the allocator returns is real memory, reachable through a 4 GiB TLB window at bank address 0, and addresses are far below the window's top.
  This explains P1's "3 GiB failed": the destination `[1 GiB, 4 GiB)` ends in the last 16 MiB of the window, which is not memory. Nothing about 2 GiB being a limit; an
  allocator-placed range never has the problem.

So the cheapest way to get a protected single-bank range is the obvious one: **allocate an interleaved DRAM buffer of `8 x size` and use bank `b`'s range**
(`ttnn.qwen_p2p_staging_alloc`, one `MeshBuffer` kept alive for the engine's life; the allocator cannot give any of it to anything else while the handle lives).

**Cost per card = `8 x size_bytes`** (plus 2.5 MiB, spread over the banks, for the checksum's partial-sum buffer: one 64-byte page per core and range of a launch, 512 ranges x 80 cores; and the kernels' binaries in DRAM at their first dispatch).
Default `size_bytes` = 2 x 64 MiB = 128 MiB, so **1 GiB per card**: 128 MiB in the chosen bank, and the other seven banks' ranges (896 MiB) are kept but unused by default.
The guarantee is that THIS MeshDevice's allocator never gives the range away: a submesh or a second MeshDevice over the same chips has its own free list (and so does a single-device
allocation), so the engine must allocate DRAM only through the MeshDevice it passes to `qwen_p2p_staging_alloc`.
They are not wasted if the host wants them: every op takes `bank=`, so the same buffer is eight staging ranges of 128 MiB (the host can stripe a restore over banks to spread
the write bandwidth, with one TLB window aim per bank). `size_bytes` is a parameter: 2 x 32 MiB costs 512 MiB, 2 x 16 MiB costs 256 MiB. The allocator's free bytes per bank before and after are in `qwen_p2p_staging_info`.

Placement: top-down (`bottom_up=False`) so the range sits at the high end of the free space, away from the bottom-up allocations of weights, KV pool and transient tensors.
Allocate it **once at engine start, after the KV pool and before any decode trace is captured** (allocating after a trace capture warns: buffers made then can
overlap addresses a captured trace re-uses). The range start is 64-byte aligned (the allocator's guarantee); `qwen_p2p_staging_info` also reports where the first 4096-aligned
bank address is (`aligned_4k_offset`) so the host can keep its O_DIRECT objects page aligned.

## 2. Ops as built (`ttnn.*`, from `qwen_kv_read` version 3)

```
qwen_p2p_staging_alloc(mesh_device, size_bytes=128 MiB, bank=0, page_bytes=64 KiB, bottom_up=False) -> QwenP2PStaging
qwen_p2p_staging_info(staging) -> [dict per device, mesh row-major order]
qwen_write_blocks_from_staging(device_tensor, staging, byte_offset, blocks, bank=None, row_stride=None)   enqueued, does not wait
qwen_staging_checksum(staging, byte_offset, length, bank=None, cores=64, count=None) -> [int per device], or with count=N [[int] * N per device]   blocking
qwen_p2p_staging_write(staging, byte_offset, data[devices, n], bank=None)                    diagnostic, blocking
qwen_p2p_staging_read(staging, byte_offset, out[devices, n], bank=None)                      diagnostic, blocking
QwenP2PStaging: .release(), .size_bytes, .page_bytes, .bank, .num_banks, .address, .released
```

* **info** per device: `device_id`, `mmio_chip_id`/`pcie_device` (tt-metal's `GetPCIeDeviceID`: the logical chip id of the MMIO chip; it is the tt-kmd index N of `/dev/tenstorrent/N` only when every card is visible
  to the process, as in the serving container), `pci_bus_id` (sysfs, derived from that: **verify it against the card actually opened**; `None` when unreadable), `dram_bank`, `noc_x`, `noc_y`, `noc_endpoints`
  (all three NOC endpoints of the bank, translated coordinates, tt-metal's order: index 0 is the view's NOC0 worker endpoint), `endpoint_index` (0), `address` (bank address of the range start = the TLB-window offset, bank offset is 0 on Blackhole), `size`,
  `aligned_4k_offset/address/size`, `page_bytes`, `bytes_reserved` (the cost above), `dram_free_per_bank_before/after`, and `banks`: the same description for all 8 banks.
  **Aim the window at `(noc_x, noc_y)` from this record**, not at the probe's formula (section 5).
* **write_blocks_from_staging**: the semantics of `qwen_write_blocks_raw`; device d's slice `blocks[i]` gets `staging_d[byte_offset + i*row_stride : + slice_bytes]`. **`row_stride=None` is `slice_bytes`: contiguous rows, byte for
  byte what the op did before the keyword existed.** A given `row_stride` must be a multiple of 64 and at least `slice_bytes`, and the whole range, up to the last row's last byte (`byte_offset + (len(blocks) - 1) * row_stride + slice_bytes`),
  must lie inside the staging range (else a `ValueError`, nothing launched). The bytes between two rows are never read, so a chunk the host laid out block-major, each block's shard `[cache][slice]`, is copied tensor by tensor
  with `byte_offset = chunk start + t * slice_bytes` and `row_stride = shard bytes`: for the host's chipmajor-v1 object one cache tensor's rows of consecutive blocks are `32 * 17,408 = 557,056` bytes apart. The stride is a runtime
  argument of the kernel (section 3), so one cached program serves every stride. The tensor
  must be an interleaved DRAM tensor laid out alike on every device; its page at most 8192 bytes (L1 scratch); `slice_bytes` a whole number of its pages (1,088-byte bf8 tile pages in production); `byte_offset` a multiple of 64 (NOT of 1,088: objects
  start at multiples of 4,096, which is not); block ids distinct; at most 512 blocks per launch (more are several launches). Every DRAM access is one whole page.
  Runtime args (the tensor's address, the staging address, the block ids) are rewritten from the live buffers on every call, as a program-cache hit would.
* **checksum**: INTERFACE section 4 exactly: `S = sum_k ((k*2654435761 + 1) mod 2^32) * x_k mod 2^64` over the little-endian uint32 words of `staging[byte_offset : + length]`,
  `k` counted from `byte_offset`. `length` a positive multiple of 4096, `byte_offset` a multiple of 64. **`count=None` returns exactly that: `[int per device]`.**
  **`count=N` computes N consecutive ranges of `length` bytes, range r at `byte_offset + r * length`, in one call, and returns `[[int] * N per device]`** (a list of N ints per device, mesh order). Each range is checksummed
  with the same algorithm and **k counted from ITS OWN first word**, so the same bytes in two ranges give the same sum in both, and range r alone is `qwen_staging_checksum(staging, byte_offset + r * length, length)`.
  The whole span `N * length` must lie inside the staging range; `count >= 1`, and it has no limit of its own: one launch carries up to 512 ranges, more are several launches (each read back before the next, since they share the
  partials buffer). `count=1` returns `[[int]]` per device, not `[int]`.
  The three shared vectors (the numpy reference is `p2p_staging_ref.checksum_numpy`, equal to an independent plain-Python implementation):

  | vector | S |
  |---|---|
  | 4096 zero bytes | 0 |
  | 4096 bytes, byte i = i % 251 | 17453227578024978262 (0xf2365202d43ae756) |
  | `numpy.random.default_rng(20261030).integers(0, 256, 1<<20, numpy.uint8)` | 4942361704597259676 (0x4496cbc8da03859c) (numpy 2.4.6; the stream is numpy's) |

  The ranged form has two vectors of its own, four ranges each and every range one of the vectors above (`p2p_staging_ref.vector_ranged_small` / `vector_ranged_prng`): `count=4, length=4096` over
  `zeros | i%251 | i%251 | zeros` is `[0, 17453227578024978262, 17453227578024978262, 0]`, and `count=4, length=1 MiB` over `prng | 1 MiB of zeros | prng | 1 MiB of zeros` is
  `[4942361704597259676, 0, 4942361704597259676, 0]`.

* Both ops run **programs**, unlike the rest of the extension: the kernels (`qwen_p2p_scatter.cpp`, `qwen_p2p_checksum.cpp`) are read from the directory of the `.so` and JIT-compiled
  at the first dispatch (seconds), and the programs are cached in the handle (the program binaries take a little DRAM at that first dispatch). **Warm them at engine start**
  (one scatter into a scratch block and one checksum of 4096 bytes), before the decode trace is captured. Call them from the thread that issues the model's ops.
  `release()` the handle **before** `close_mesh_device` (and after the last op has completed): after the mesh is closed it leaks the cached programs on purpose, because their destructors would
  touch destroyed devices.

## 3. The kernels

* **scatter** (`qwen_p2p_scatter.cpp`): 16 cores (8 x 2), RISCV_0. Each takes up to 32 of the launch's blocks (a contiguous slice), reads the block's pages from the staging bank in
  chunks of 16 pages into one of two L1 halves and writes them to the destination pages, the reads of chunk c+1 overlapping the writes of chunk c. L1: `2 x 16 x page_bytes + 64`
  (34,880 bytes for bf8 tile pages) per core. The staging source is addressed as `bank, address + offset` through a `TensorAccessor` on the staging buffer (page id = bank id), so the
  staging buffer's own page interleaving never enters the arithmetic. **The row stride is a runtime argument** (arg 5, the block ids moved to args 6..37; arg 3 is the byte offset of the core's first row,
  `byte_offset + first * row_stride`): page `p` of a core's run is read from `arg3 + (p div pages_per_block) * row_stride + (p mod pages_per_block) * page_bytes`. No division by a runtime value: `pages_per_block` is a compile-time argument.
* **checksum** (`qwen_p2p_checksum.cpp`): `cores` cores (default 64, a multiple of 8, at most 80), RISCV_0. The range is split in 4096-byte units, core c takes
  `[floor(c*u/C), floor((c+1)*u/C))`, starts its weight at `w_(k0)` (computed by the host), walks it with `w += 2654435761` and accumulates `uint64(w) * x` (`mul` + `mulhu` on the rv32im core),
  reading 32 KiB chunks into two L1 halves. Each core writes its 64-bit partial as a whole 64-byte page; the host adds the partials mod 2^64. L1: 65,664 bytes per core.
  **Many ranges in one launch** (`count`): every range is split among the cores the same way, so a core takes the same run of units of EACH range, restarts its weight at the start of each range (`w_(k0)` is the
  same for every range), and writes one partial page per range, range r of core c at partials page `r * cores + c` (runtime args 7..9: the number of ranges, the bytes between two range starts, the pages between two slots). The chunks
  of consecutive ranges are one stream, so the read of a range's first chunk overlaps the arithmetic on the previous range's last. A core with no units still writes its zero partial for every range (the partials
  buffer is reused between calls). The host reads `launch ranges x cores` pages back per device and adds each range's `cores` partials mod 2^64; the loops need no division by a runtime value (counters step through the chunks).
  Rate (an estimate from the instruction count, to be measured): about 0.6 GB/s per core, so 64 cores check 64 MiB in roughly 2 ms. It uses compute cores the decode needs: it
  is a short, blocking op per staged chunk, not a background job.

## 4. Checks on a CPU (all named in `qwen-integration-cpu.yml`)

`optimisation/ttnn-op/kv_region_read/test_p2p_staging.py`, `test_p2p_staging_card.py` and `scripts/ci/test_pin_kvread_p2p.py`:

* checksum: numpy = plain Python = the three expected values; k restarts at the range start; one flipped bit changes it; partial sums over 1/8/64/80 cores add up; **the two ranged vectors, and every range restarting k**;
* `qwen_p2p_plan.hpp` (every split, weight, argument vector, CB size and argument check, **the strided range check at its boundaries (a last row ending on the last byte, one past it, a stride that wraps 64 bits), the
  checksum launch split**) compiled and compared with its Python mirror; the arithmetic block of the checksum kernel
  compiled as it stands in the kernel, under many chunkings; `row_stride=None` is checked to be accepted and refused exactly where the contiguous op was;
* **both kernels run on the CPU** (`emu/`): the real kernel source is compiled against a stand-in for `api/dataflow/dataflow_api.h` and driven by the same argument builders the extension uses.
  The stand-in models DRAM banks, an interleaved `TensorAccessor`, L1, and NOC transfers that land late (at the barrier) or at once (`-DEMU_EAGER`), and refuses a sub-page access, an
  unaligned DRAM address, an access outside the allowed range and a kernel that finishes with a transfer pending. Scenarios: one block; runs; shuffled ids; every bank; page
  sizes 64..32768; 1..48 pages per block; 17, 100, 513, 600 blocks (cores and launches); offsets of 64, 192, 4160+k*17408; the end of the ring; **strides of block + 64 up to 557,056 bytes (the bytes between two rows
  are noise, and the stand-in refuses a read that reaches into a gap), the same 17..600 blocks over cores and launches, a strided range that ends on the last byte of the ring**; the checksum vectors on 8/24/64/80
  cores at several offsets and banks; **7 ranges of 136 units (a chipmajor-v1 shard) and 60 ranges of 9 units in one launch, idle cores (more cores than units) with 40 ranges, 4 ranges of 1 MiB, and the two ranged vectors**;
* **mutation check**: plausible slips in the kernels (a barrier dropped, the wrong half, an off-by-one page, a half-page write, a restarted weight, a truncated product, a wrong
  slot, **the stride ignored, applied per page or one unit short, read from the wrong argument; a weight or sum not restarted for the next range, a range read from the previous one, a partial in the
  previous range's slot, the last range dropped, an idle core writing no partial**, ...) are each caught under at least one of the two schedules, and the unbroken kernels pass both;
* the card test (`p2p_staging_card.py`) against an in-memory fake extension: a correct fake passes, and each of fifteen plausible mistakes (a flipped byte, a block forgotten, a spill into
  an unnamed block, `bank=` ignored, a wrong checksum, a short info list, a differing or misaligned address, **a stride ignored or one unit short, the ranges swapped, k running on from the first range, a range missing**, ...) fails it.
  Its strided cases stage a chipmajor-v1 chunk (block-major, each block's shard `[cache][slice]`), checksum it with `count=` and copy it tensor by tensor with `row_stride=`; a build without the keywords is reported `device_ops: compat`
  and the cases are skipped (`strided=True` / `--require-strided` makes that a failure);
* the extension source's signatures: the two keywords last, with `None` defaults, so every existing positional call stays valid (`bank` is still the 4th argument of the checksum).

## 5. Corrections to the contract (INTERFACE.md), for the orchestrator

1. **Cost**: the single-bank range costs `8 x size` per card, not `size`: the allocator is lockstep over banks (section 1). The other seven banks' ranges stay usable through `bank=`.
2. **P1's "~2 GiB usable"**: not a memory limit. The top 16 MiB of each 4 GiB window is not DRAM (`dram_view_size` = 0xFF000000); P1's 3 GiB test ended inside it. A range the allocator
   hands out is always inside memory.
3. **`byte_offset` need not be a multiple of 1,088**: objects start at multiples of 4,096 in the ring, so a second tensor's data starts at a multiple of 64 only. The ops require 64.
4. **NOC coordinates**: take `noc_x`/`noc_y` (or any of `noc_endpoints`) from `qwen_p2p_staging_info`. They are tt-metal's own translated coordinates of the bank's endpoints
   (`logical DRAM core (bank, endpoint index)`); the default is endpoint 0, the view's NOC0 worker endpoint, which the kernels and tt-metal's own host reads of DRAM use (for banks 0 and 4-7 it is
   also the sub-port the UMD's static TLBs use; for banks 1-3 that is endpoint 2). Whether the three endpoints of a channel are interchangeable for a PCIe window is a hardware fact the
   source cannot show: the host's window self-test settles it (try another endpoint from the list if the default fails). The probe's formula (banks 0-3 in column 17, 4-7 in
   column 18, `y = 12 + 3*(bank mod 4) + port`) is an assumption from the UMD; the card test reports whether it agrees with the allocator's answer (`formula_first_endpoint_in_info`).
   P1 verified one window by reading back through it, which does not prove the bank number; the checksum does.
5. **Object bytes (section 1)**: "a host-path restore and a P2P restore read the same object bytes" holds for the **slice bytes** (the packed bf8 bytes of one block of one cache tensor on one
   chip), not for the arrangement. The host tier's record (`_QwenKvTierIO`) is per block, `[cache][chip][slice]`, chip-interleaved; the P2P object of a card is per run of blocks,
   `[cache][block][slice]` for that chip. The spill side must transpose (`kv_shard_layout.py` is the model of it); the device ops follow section 1 and do not care.
6. **`address`** in `info` is the range start in the bank as the window sees it (bank offset 0 on Blackhole is added anyway). The host mmaps the whole published slice and DMAs at
   `address + byte_offset` from the mapping's start; nothing needs to be MiB aligned.
7. **Flush before the first device read.** A posted PCIe write finishing at the NVMe is not an ordering point on the card's NOC and DRAM path. After the NVMe completion and before enqueueing the
   checksum or the scatter, the host should do a non-posted read through the same TLB window (any word of the range): the usual flush of posted writes. It does not strictly order writes of
   another requester (the drive), so the checksum stays the arbiter (a miss is a cold prefill); a scatter alone would not catch a stale range.
8. **Program binaries**: the first dispatch of each op allocates a little DRAM (its kernel binary) and compiles; do it at engine start (section 2).

## 6. What only a card settles

* the JIT's link of the two kernels (they compile with the real RISC-V compiler and the JIT's flags, `rv_kernel_check.sh`, but nothing was linked or dispatched);
* that `MeshBuffer::create` replicated gives the same address on all four cards (the constructor checks and throws if not) and the page size asked for;
* the NOC coordinates and the bank (card test: `staging` block of the report), the lockstep, the cost in free DRAM per bank (`dram_free_per_bank_*`);
* whether the three NOC endpoints of a channel are interchangeable for a PCIe window, and that `pci_bus_id` is the card the host opened;
* L1 fit of the circular buffers on cores (0..7, 0..9) next to a served model's L1 buffers (a clash is a `TT_FATAL` at the first dispatch, not corruption);
* the rates: the write op's GB/s per card (report `write_gbps_per_card`), the checksum's (`checksum_gbps_per_card`, and `ranged_checksum_gbps_per_card` for a chunk with `count=`), the effect of reading 16 cores' worth of pages from one bank;
* the two new kernel paths on silicon: the scatter's reads at `row_stride` apart (one DRAM page per read, 64-byte aligned by the plan's check), and the checksum's per-range partial pages (a 2.5 MiB partials buffer; the read-back of `ranges x cores` pages per device);
* running while decode traces are live (ordering on the one command queue is assumed, as for every other op).

Run: `python3 optimisation/ttnn-op/kv_region_read/p2p_staging_card.py --devices 4 --staging-mib 32` in a card window (the host side's probe calls `run()` from the same file).

## 7. Delivery into an image

The module is the existing `qwen_kv_read` extension, version 3. The image copies its **whole build directory** (`COPY opgraft-KVR/`), so the two kernels and the header ship beside the `.so`.

1. On the rig host, from a checkout of this branch: `bash optimisation/ttnn-op/kv_region_read/build_kv_read.sh ~/opgraft-KVR3` (compiles in the build container, then imports the module in
   the production image with no network and no device; it also compiles both kernels with the real RISC-V toolchain, and refuses unless all ten ops import and the kernels sit beside the `.so`).
2. `python3 scripts/ci/pin_kvread.py --dir ~/opgraft-KVR3 --write` moves the four pin sites to the new hash (it refuses a build whose source or shipped kernels are not this
   checkout's, or that lacks the P2P op names); commit that.
3. `build-c2-serving-image.sh` now reads `~/opgraft-KVR3` by default (the version 2 and 1 directories are untouched). The Dockerfile's RUN fails the image build unless the baked module is
   version 3 with the ten ops and both kernels.
4. Image build job (a build, no card; `.github/c2-serving-job.env` on a throwaway commit, then push the tag `experiment/c2-serving-vN`; one tag at a time):

```
C2_CARDS=quad
C2_ACTIONS=build
C2_IMAGE_TAG=<new tag>
C2_PROFILE=c2-packed-tp4
```

Read the build log for the provenance report with no problem line (check (g) at the new pin) and the Dockerfile's `[KVREAD] qwen_kv_read 3 ...` line.

The `.so` built from this source carries `row_stride` and `count`; the pin moves with the rebuild. An image that still carries an older version 3 `.so` (contiguous copy, one-range checksum) keeps working: the host finds the forms from
the ops' signatures and runs its slow compat mode against it (`docs/tp4-p2p-host.md` section 1), and the card check reports `device_ops: compat` instead of failing. To make a build without the keywords a failure, run the card check with `--require-strided`.
