# TP4 fabric upload: feeding the x4 cards over the fabric instead of their own PCIe

Status: **design and two measurement jobs; nothing has been measured on cards yet, and nothing in serving changes.**
The owner directive of 2026-10-10 says the four cards should use the fabric, not PCIe, for upload.
The branch is `tp4/fabric-upload`.

The hardware:
- Two of the four p150a cards train PCIe Gen5 x16 on CPU root ports.
- The other two sit behind a PCIe switch and train Gen5 x4.
- The four cards form a (1, 4) mesh. Every card pair is cabled with two 400G ethernet links, and the fabric routes the
  2x2 ring (scripts/ci/tp4_mesh.py).

Every number below carries a label:
- **M** measured, with its source.
- **C** computed from code or config.
- **I** inferred from measured numbers.
- **A** assumed. Each A is replaced by one of the two jobs in section 3.

File:line references to `ttnn/...`, `tt_metal/...` and `tests/...` are into the pinned tt-metal
`9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`; "UMD" is its UMD submodule.

## 0. Bottom line

1. **What PCIe carries today is not what makes engine start slow.**
   - Engine start takes 2 min 47 s on an idle rig and 9 min 43 s under CI load (M).
   - At engine start each card takes about 6.9 GB of weights (C) and about 14 GB of zero-filled buffers (M) over its
     own PCIe.
   - The layer-loading phase takes 75 s idle and 409 s under CI load (M). The bytes it moves are the same in both runs,
     so host CPU work sets that phase, not any link.
   - Moving every x4 byte onto the fabric would save under 1 s of either start (section 2.6).
2. **The cheapest cut in x4 PCIe traffic needs no fabric.**
   - The 14 GB of zeros per card is built on the host and written over PCIe. The KV pool is 11.1 GB of it.
   - At the pinned runtime, `ttnn.zeros_like(device_tensor)` fills a TILE bf8/bf16/fp32 device tensor on the device
     with an SFPU kernel (`ttnn/cpp/ttnn/operations/creation/creation.cpp:237-244`).
   - Allocating and filling on the device removes about two thirds of every card's engine-start PCIe bytes.
3. **A relay of the weights can be built with the pinned runtime. It pays only when an x16 card writes more than
   twice as fast as an x4 card AND the fabric op keeps up. Neither is likely.**
   - Model for symmetric shards: direct takes `B / r4`, relay takes `max(2B / r16, B / f)`.
   - Upstream's own Blackhole write benchmark reaches **21.5 GB/s** (M upstream, link width unstated). A Gen5 x4 link's
     payload ceiling is 15.75 GB/s (C). Unless the x4 cards fall below about 10.7 GB/s, an x16 card is not twice as fast.
   - `ttnn.point_to_point`, the only ready-made unicast, drives **one worker core and link 0**. Upstream's one-core
     fabric unicast benchmark targets 6.3 GB/s at 1 MiB (M upstream). That is below an x4 link.
   - The multi-link path is `send_async/recv_async` over sockets between submeshes. It is measured too, but submeshes
     bring their own DRAM allocators.
4. **Today's 1.15 GB/s restore rate is not a link limit.**
   - Measured: 78.4 MB of GDN checkpoint in 66.5-68.0 ms for the whole mesh, conversion included (M, Lever N park-in
     lines).
   - The 96 small tensors go through a single-threaded host tilize and one dispatch each. One x4 card's 26.7 MB share
     would take 1.7 ms at its link ceiling.
   - For a future warm-tier KV restore (1.16 GB per card at 128k, 2.23 GB at 254k, C), the relay saves at most 39 ms
     and 74 ms. That needs r16 = 40 GB/s and a multi-link fabric path (A), against a 1.6-9 s warm restore.
5. **Two card jobs, about 1 h together, settle every A in this document.**
   - F1: per-card host-to-device rates, x16 against x4, alone and together; the restore's conversion/copy split; and
     whether the host can write one mesh coordinate.
   - F2: `point_to_point` on every edge; socket send/recv; and the **relay end to end against a direct write of the
     same bytes, in one job**, byte for byte.
   - `fabric_upload_plan.decide` and F2's `relay_over_direct` give RELAY, DIRECT or HOST-BOUND per use. The rule is
     registered here before any run.

What to do regardless of the jobs: P0 (zero buffers on the device) and P1 (host-side work on cache hits and restores),
in section 5.

## 1. Where host-to-device bytes go today

### 1.1 Engine start, per card

This is the production profile's tensor set, under the image's environment: `QWEN_FAST_SINGLE_GATEUP=1`,
`QWEN_FAST_SKIP_BLOCK_STREAM=1` and `QWEN_SDPA_BF8=1`.

Model shape: hidden 5120, 64 layers (16 full attention, 48 GDN), intermediate 17,408, vocabulary 248,320, 4 KV heads
of 256.

| What | Per card | Mesh total | Placement | Label |
|---|---:|---:|---|---|
| MLP w1 + w3 (bf4) | 1.604 GB | 6.42 GB | sharded (`ShardTensorToMesh` dim -1) | C, equal to the memory ledger (M) |
| MLP w2 (bf8) | 1.515 GB | 6.06 GB | sharded, dim 0 | C = M |
| GDN projections (bf8) | 1.511 GB | 6.04 GB | sharded | C |
| Attention projections (bf8) | 0.446 GB | 1.78 GB | sharded | C |
| Embedding | 0.636 GB | 2.54 GB | sharded on hidden (`ShardTensor2dMesh`) | C = M |
| LM head (bf8) | 0.338 GB | 1.35 GB | sharded on vocabulary | C = M |
| Drafter projections (bf8) | 0.460 GB | 1.84 GB | sharded, converted on the host every boot (no cache file) | C |
| **Replicated weights**: norms, RoPE tables, drafter norms and small projections | ~0.22 GB | ~0.87 GB | `ReplicateTensorToMesh`: every card takes the same bytes | C/I |
| **Weights in all** | **~6.9 GB** | **~27.5 GB** | | C |
| KV pool (32 tensors of bf8 zeros) | **11.123 GB** | 44.5 GB | `as_tensor(torch.zeros(...), Replicate)`, no cache file | M = C (19,968 blocks x 557,056 B) |
| Fast-path buffer pool (zeros) | 2.530 GB | 10.1 GB | `from_torch(torch.zeros, Replicate)` (scripts/ci/serving_buffer_pool.py) | M |
| GDN states and scratch | ~0.3 GB | ~1.2 GB | | I |
| **Engine-start upload per card** | **~21 GB** | **~84 GB** | | C/M |

- Every projection is interleaved DRAM (`proj_1d_decode` and `mlp_1d_decode` are on in the graft's model_config), and
  that is what `point_to_point` writes.
- Replicated weights uploaded four times cost about 0.65 GB of extra PCIe traffic for the mesh, about 2.4% of the
  weights (C).
- The zero buffers are two thirds of every card's engine-start upload.
- **Cache hits still pay host work.**
  - `tp_common.shard_w` runs `.to(bf16).T.contiguous()` before `as_tensor` looks at its cache.
  - The checkpoint is paged in through `from_pretrained`.
  - This repo already proved a cache-hit loader that skips both: docs/loading-diagnostic-results.md, 12.2 s against
    313.8 s at TP2. Serving does not use it.

**Engine-start timeline (M, two CI runs of the production-lineage profile, container logs):**

| Phase | idle rig (load 3-7) | CI load (load 16-36) |
|---|---:|---:|
| vLLM init to mesh/fabric open | 22 s | 23 s |
| 64 layers, every tensor a cache hit | **74.7 s** | **408.8 s** |
| LM head, final norm, vision | 11 s | 32 s |
| KV allocation, GDN reset, prefix warm | 20 s | 23 s |
| Buffer pool | 1.1 s | 2.8 s |
| Drafter upload | 6 s | 7 s |
| Captures (attach) | 17 s | 59 s |
| To "Application startup complete" | 2 min 47 s | 9 min 43 s |

- The layer phase is 5.5x slower under CPU load. That makes it host-bound.
- In the idle run it moved about 20 GB in 75 s: about 0.27 GB/s for the whole mesh (I).
- The production deploy's ~13.5 min load timeout was nearly hit under CI load. With CI paused, the load finished.

### 1.2 Per request: the prefix restore

**Today: a GDN checkpoint only.** The attention KV stays in the device pool.
- `_qwen_prefix_restore` (scripts/ci/qwen_prefix_model_patch.py, h2d mode) handles 48 layers of rec_state and
  conv_carry.
- Each of the 96 tensors goes through `ttnn.from_torch(..., mesh_mapper=ShardTensorToMesh(dim=0))`, which tilizes on
  the host, and then `ttnn.copy_host_to_device_tensor`.
- Size: 78,446,592 B logical for the mesh, 19.6 MB per card (M).
- On the device the conv carry is tile-padded from 3 to 32 rows, so about 26.7 MB per card crosses PCIe (C).
- Measured: 66.5-68.0 ms, about 1.15 GB/s for the mesh, conversion included (M).

**Planned warm tier (the KV tiers plan; another workstream, whose files are not touched here):**
- Content: raw bf8 KV pages, 8,704 B per token per card (16 layers x K and V x one 256-wide head per card, 1,088 B per
  tile), plus the GDN checkpoint.
- Size: **1.16 GB per card at 128k, 2.23 GB per card at 254k (C).**
- The pages are scattered over vLLM's paged pool by block id, 557,056 B per block per card.
- There is one command queue, so a restore serializes with the decode traces on CQ0.

### 1.3 The path the bytes take in the pinned runtime

**The Python calls.**
- `ttnn.copy_host_to_device_tensor`, `to_device` and the `as_tensor` cache hit all end in the mesh command queue's
  `enqueue_write`.
- Call path: `ttnn/core/tensor/tensor_ops.cpp:128-182` → `impl/tensor/tensor_apis.cpp:79-110, 152-231` →
  `distributed/mesh_command_queue_base.cpp:231-323` → `FDMeshCommandQueue::write_shard_to_device`
  (`fd_mesh_command_queue.cpp:878-926`) → `buffer_dispatch::write_to_device_buffer` (`impl/buffers/dispatch.cpp:1145+`).

**Fast dispatch.**
- The host CPU copies each payload into **that card's own hugepage issue queue**. The copy is an AVX non-temporal
  memcpy (`dispatch/memcpy.hpp:56`, `device_command.cpp:1341-1345`).
- The card's prefetcher then pulls it over **that card's own PCIe**. Only the small fetch-queue entry crosses the TLB
  (`impl/dispatch/system_memory_manager.cpp:194-198, 257-258`).
- Commands are at most 128 KB. Pages larger than that are split into 4 KB pieces (`dispatch_settings.cpp:60-101`,
  `dispatch.cpp:426-459`).
- Blackhole has no host DMA engine (`llrt/tt_cluster.cpp:840-853`).
- `TT_METAL_SLOW_DISPATCH_MODE=1` switches to per-page TLB writes instead.

**Pinned zero-copy.**
- A write over **32 MiB** can skip the memcpy: the prefetcher reads the user buffer directly
  (`tensor_apis.cpp:40-48, 174-216`, `dispatch.cpp:1004-1040`).
- That happens **only if the IOMMU is on** (`distributed/pinned_memory.cpp:389-405`).
- Python cannot reach the pinning API (`api/tt-metalium/experimental/pinned_memory.hpp:48-200`). The probe records
  each card's IOMMU group and times 16 MiB against 64 MiB writes.

**Four cards in parallel.**
- Each shard write is a task on the dispatch thread pool, which has one NUMA-pinned thread per card
  (`mesh_command_queue_base.cpp:261-266`, `impl/threading/thread_pool.cpp:295-343`).
- `TT_MESH_PASS_THROUGH_THREAD_POOL` makes it serial.

**Replicated tensors** are N uploads.
- One host buffer is written to every card: `distributed_tensor.cpp:268-287`, `mesh_command_queue_base.cpp:182-196,
  310-322`.
- A replicated cache file is stored unsharded and fanned out the same way (`ttnn/ttnn/operations/core.py:703-704`,
  `distributed_tensor_apis.cpp:201-255, 286-294`).

**Host conversion.**
- `from_torch(device=None)` tilizes and packs on the host, one shard after another in the calling thread, with a
  single-threaded tilize (`distributed_tensor.cpp:404-451`, `impl/data_format/tilize_utils.cpp`).
- `from_torch(device=mesh)` can tilize on the card after a row-major upload (`py_to_tt_tensor.cpp:65-99, 254-289`;
  for bfp8/bfp4 only with `enable_bfloat_opt`).

**No host route through another card's PCIe.**
- UMD's Blackhole discovery never opens a remote chip (UMD `topology_discovery_blackhole.cpp:36-42`).
- The tunneling machinery is Wormhole only (`dispatch.cpp:1022-1036`, `impl/dispatch/topology.cpp:496-510`).
- On these cards **only a device-side fabric kernel can carry bytes between cards.**

## 2. Design: x4 cards fed over the fabric

### 2.1 The primitives

**`ttnn.point_to_point(input, sender_coord, receiver_coord, *, output_tensor=None, intermediate_tensor=None,
topology=Linear)`**
- Binding: `ttnn/cpp/ttnn/operations/point_to_point/point_to_point_nanobind.cpp:167-177`. In Python the sender comes
  first; the C++ `point_to_point.hpp:15-21` takes the receiver first.
- Guarantees (`device/host/point_to_point_device_op.cpp`):
  - interleaved DRAM or L1 only (`:103`);
  - TILE or ROW_MAJOR, with pages a multiple of 16 B (`:135-140`);
  - the output has the input's exact TensorSpec and lives on the same mesh (`:118-134`);
  - only the sender's and receiver's programs run (`:242-256`), so only the receiver's shard of `output_tensor` is
    written, in place, at its address.
- Costs:
  - **One worker core and link 0** (`send_program_factory.cpp:36-37, 124`; `receive_program_factory.cpp:36, 123`).
  - An input-sized intermediate tensor, allocated mesh-wide unless passed in (`:161-190`). Bytes land in the
    receiver's intermediate and are copied to the output there.
  - On a program-cache miss, a global semaphore and a mesh-wide `Synchronize` (`:226-231`).
  - Ring routing never takes the wrap: `ring_hops = abs(line_hops) + N` is never shorter (`:45-98`). Position 0 to 3
    is three hops.

**`ttnn.experimental.send_async(input, socket)` / `recv_async(output, socket)` over `ttnn.create_socket_pair`**
- Needs two mesh devices, i.e. submeshes (`tests/nightly/t3000/ccl/test_send_recv_async.py:110-111`).
- One core per socket connection; connection i uses link `link_indices[i % n]` (`send_async_op_program_factory.cpp:241-247`).
  This is the multi-link path.
- FABRIC_1D is allowed; the sender and receiver must share a row (`distributed/mesh_socket_utils.cpp:63-120`).
- `recv_async` writes into the tensor it is given.

**A custom kernel**
- Device side: `fabric_unicast_noc_unicast_write` (`fabric/hw/inc/linear/api.h:81, 113`, plus `_with_state` and
  `_set_state` variants) and the TensorAccessor-addressed `to_noc_unicast_write` (`linear/addrgen_api.h:86-100`).
- Host side: `append_fabric_connection_rt_args` (`experimental/fabric/fabric.hpp:60-73`, adjacent router only) and
  `get_forwarding_link_indices` (`:115`).
- Default fabric payload: 4,352 B (`fabric/erisc_datamover_builder.hpp:460-483`).

**The host leg: writing one coordinate only.**
- `copy_host_to_device_tensor(single_device_host, ttnn.get_device_tensors(t)[i])` **writes all four cards**. A 1x1
  host buffer is replicated to the mesh (`distributed_tensor_apis.cpp:254, 286-294`; `tensor_ops.cpp:176-178`).
- What writes coordinate i only is a host tensor built with
  `ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig([PlacementReplicate(), PlacementReplicate()], MeshShape(1, 1),
  MeshCoordinate(0, i)))`.
  - Its other shards are absent, and `enqueue_write` writes populated shards only (`mesh_command_queue_base.cpp:310-322`;
    `distributed_nanobind.cpp:663-688`).
  - It narrows the handle it is given (`tensor_ops.cpp:178`), so pass a `get_device_tensors` view, never the tensor a
    later `point_to_point` reads.
- F1 checks both writes per coordinate (`subset`). F2 checks the mapper write again before its relay arms.
- **Submeshes are not a staging alternative.** Each one builds its own allocator, so parent and submesh DRAM can
  overlap (`mesh_device.cpp:611-705`, `distributed/mesh_buffer.cpp:145-161`).

### 2.2 Who sends to whom

`fabric_upload_plan.relay_pairs` gives every x4 position its own x16 sender: fewest total line hops first, then the
smallest worst hop. Which positions are x16 depends on how the ring maps the cards. The probes read it at run time:
chip id → UMD `chips_with_mmio` → `/dev/tenstorrent/N` → the sysfs link.

| x16 positions | pairs (sender -> receiver, hops) |
|---|---|
| 0, 1 | 0->2 (2), 1->3 (2); both cross edge 1-2 in one direction |
| 2, 3 | 2->0 (2), 3->1 (2); both cross edge 1-2 |
| 0, 3 | 0->1 (1), 3->2 (1) |
| 1, 2 | 1->0 (1), 2->3 (1) |
| 0, 2 | 0->1 (1), 2->3 (1) |
| 1, 3 | 1->0 (1), 3->2 (1) |

On link capacity:
- Two links per edge carry about 42-45 GB/s per direction (I): the TP2 all-gather's 83.74-90.43 GB/s counts both
  directions of every card (docs/fabric-bandwidth-2026-09-19.md).
- `point_to_point` uses one of the two links.

### 2.3 Weights (one time, at engine start), when `QWEN_FABRIC_UPLOAD=weights`

Per destination tensor (same spec as today: an interleaved DRAM mesh tensor):

1. Write each x16 card's own shard into the destination with a one-coordinate write.
2. Write each x4 card's shard into a **staging tensor with the destination's spec**, at its sender's coordinate.
   - The staging tensor and the op's intermediate are reused per spec.
   - Tensors over 256 MiB per card are chunked by rows.
3. Move each x4 card's shard on with `point_to_point(staging, sender, receiver, output_tensor=destination,
   intermediate_tensor=scratch)`, or a socket pair when F2 shows that is faster. Both pairs are issued back to back
   with one synchronize.

No x4 card takes a byte over its PCIe. With the same spec on both ends, the pages move as opaque bytes: no tilize, no
repack, no reshard.

**Exceptions**
- **Sharded destinations.** None of the projections are sharded here; the inventory step lists any that are. Relay
  them into an interleaved staging tensor on the receiver, then run `ttnn.to_memory_config` on the x4 card: an
  on-device copy at DRAM speed (420 GB/s achieved per card, M, docs/tp4-drafter-bf16.md).
- **Replicated tensors** (~0.22 GB per card). One x16 card takes them once, and the fabric copies them on. These are
  small tensors, so the per-call cost may exceed the 0.65 GB saved; F2's 4 MiB rows decide.
- **Zero buffers** are never relayed. They are filled on the device instead (P0).

### 2.4 Warm-tier KV restore (per request, latency-critical)

**Data path when relayed**
1. The x16 card takes its own blocks plus its x4 partner's, as one contiguous staging slab of raw bf8 pages per chunk.
2. The fabric moves the partner's slab to an x4 staging slab.
3. An **on-device scatter** writes the slab's pages into the paged pool at the request's block ids.

**The missing piece is that scatter.** `point_to_point` and `recv_async` write whole tensors, while the pool takes
blocks at arbitrary ids. There are two options:
- (a) A device op that copies N pages from a slab to a page list: the device-side twin of the tiers plan's
  host-to-device region write. **Preferred:** no fabric code of our own, and a direct restore can reuse it.
- (b) A fabric kernel that writes each page to its pool address with `to_noc_unicast_write` (2.1).

**Pipelining**
- Chunks of 32-64 blocks: 17.8-35.7 MB per card per chunk.
- The x16 host write of chunk k+1 overlaps the fabric move of chunk k.
- Staging is two slabs per pair, under 0.15 GB per card. It must stay chunked: about 2 GB of DRAM is free per card
  after attach with engine reuse (docs/tp4-engine-reuse.md).

**CQ0**
- The fabric and scatter programs run through CQ0, as the direct writes do, so either way a restore serializes with
  the decode traces.
- The relay adds device work: the receiver's double DRAM write plus the scatter, about 5 ms for 2.2 GB (I, at
  420 GB/s).

**The GDN checkpoint stays direct.**
- It is 96 tensors of 0.4 MB per card. A fabric call per tensor costs more than the 1.7 ms its x4 bytes take over PCIe.
- Its real cost is host conversion, so store it as converted host tensors and a restore is copies only (P1).
- F1's `restore` arm splits conversion from copy.

### 2.5 What the relay cannot change

- CQ0 serialization.
- Host conversion.
- Any byte an x16 card takes for itself.

It doubles the x16 cards' PCIe load; hence the condition `r16 > 2 * r4`, together with `f > r4`.

### 2.6 Estimates

Setup: per card, symmetric shards, relay pairs 0->2 and 1->3, overlapping per-card writes, pipelined 64 MiB chunks.
The arithmetic is `fabric_upload_plan.direct_seconds` / `relay_seconds`. Direct uses r16 = 21.5 and r4 = 12 GB/s.

| Upload | bytes per card | direct | relay: r16 21.5, f 40 | relay: r16 40, f 40 | relay: r16 40, f 6.3 (p2p one core) |
|---|---:|---:|---:|---:|---:|
| Weights | 6.9 GB (C) | 0.575 s | 0.642 s (worse) | 0.345 s | 1.106 s (worse) |
| Zero buffers | 14 GB (M) | 1.17 s | 1.30 s | 0.70 s | 2.23 s |
| Zero buffers, filled on the device | 0 over PCIe | **~0.07 s** (A: ~200 GB/s fill) | | | |
| Restore 32k | 0.30 GB (C) | 25 ms | 28 ms | 15 ms | 59 ms |
| Restore 128k | 1.16 GB (C) | 97 ms | 108 ms | 58 ms | 195 ms |
| Restore 254k | 2.23 GB (C) | 186 ms | 207 ms | 112 ms | 365 ms |
| Restore 254k at today's measured 1.15 GB/s (mesh, host-bound) | | **7.8 s** (I) | | | |

The best case saves 0.23 s of a 2.8-9.7 min engine start and 74 ms of a 254k restore. The likely case loses.

**Measured against assumed**

| Quantity | Value used | Label | Replaced by |
|---|---|---|---|
| Gen5 x16 / x4 payload ceiling | 63.0 / 15.75 GB/s | C (32 GT/s x lanes x 128/130 / 8) | - |
| Link widths: 2 x16 on root ports, 2 x4 behind a switch | | M (sysfs, read-only, 2026-10-10) | F1 `links` |
| x16 host-to-device, one card | 21.5 GB/s (upstream golden, `benchmark_rw_buffer`, link unstated); 40 optimistic | **A** (M upstream only) | F1 `alone/rm16k/1024MiB` |
| x4 host-to-device, one card | 12 GB/s | **A** | F1 `alone/rm16k/*` |
| The four cards' writes overlap | yes (one dispatch thread per card, read from code) | **A** | F1 `together` against `alone` |
| The x4 cards' shared switch uplink does not halve them | | **A** | F1 `together` |
| Pinned zero-copy above 32 MiB | only with the IOMMU on | C (code), state **A** | F1 `links.iommu_group`, 16 vs 64 MiB rows |
| Restore today | 1.15-1.18 GB/s for the mesh, conversion included | M (Lever N park-in lines) | F1 `restore` (split) |
| Fabric, two links, per direction | 42-45 GB/s | **I** (TP2 all-gather) | F2 `socket/*/2conn` |
| `point_to_point` (one core, link 0) | 6.3 GB/s (upstream one-core unicast target at 1 MiB, FABRIC_2D) up to one link | **A** (M upstream only) | F2 `pair/*`, `edge/*` |
| Two streams across one edge | half each | **A** | F2 `both/*` |
| First fabric call per shape and pair | seconds (programs + barrier) | **A** | F2 `first_call_s` |
| One-coordinate host write | works (code reading) | **A** | F1 `subset`, F2 `mapper_write` |
| Relay against direct, end to end | modelled above | **A** | F2 `relay/*` (`relay_over_direct`) |
| Engine-start phases, weight and zero-buffer bytes | tables 1.1 | M and C | - |

## 3. Measurement jobs (scripts/ci/references/fabric-upload-jobs)

**What both jobs share**
- They run in the `fabric` step of `qwen-c2-serving.yml` with `C2_CARDS=quad` and `C2_ACTIONS=reset fabric`.
- The probe comes from the checkout, mounted read-only, inside any image built on the pinned tt-metal.
- Each job opens the mesh once, under the ring descriptor only.
- The JSON report is rewritten after every arm. A failed arm becomes a problem line, and the next arm still runs.
- Exit codes:
  - 0: MEASURED.
  - 1: INEXACT. A transfer left bytes nobody wrote.
  - 2: PARTIAL or NOT-MEASURED.
  - 3: the watchdog fired, after 28 min.
- Every template parses with `python3 -s scripts/ci/c2_serving_job.py <template>` (rc 0), and `test_fabric_upload`
  holds them.

| Job | Probe | Arms | Estimate |
|---|---|---|---|
| **F1-h2d-per-card** | `C2_FABRIC_PROBE=h2d`, `tp4_h2d_probe.py` | links (widths, port chains, IOMMU groups); subset (view and mapper one-coordinate writes: exact / broadcast / missing / corrupt / refused, and the tensor's coordinates afterwards); alone per card (the mapper write, else (1, 1) submeshes), ROW_MAJOR 16 KiB pages at 4/16/64/256/1024 MiB, TILE bf16 and bf8 at 64/256 MiB, first call apart, read back and compared, D2H timed; together (sharded) and replicated at 256 MiB per card; restore (the production checkpoint's 96 tensors, conversion and copies apart); convert (host from_torch per format) | reset 3-12 + 8-20 min |
| **F2-p2p-x16-to-x4** | `C2_FABRIC_PROBE=p2p`, `tp4_p2p_probe.py` | links and pairs; local (the op's on-card copy); every directed line edge at 256 MiB; far (0 -> 3); each relay pair at 4/64/256 MiB in rm16k, tile_bf8, tile_bf16; both pairs at once; reverse; **relay** (the mapper write re-checked, then the full relay against a direct write of the same bytes, per pair and all pairs, plus the production sharded write); socket (send/recv between (1, 1) submeshes, one and two connections, DRAM FIFO, last) | reset 3-12 + 10-25 min |

**Running them**
- Push one tag at a time.
- The owner stops production for the window.
- Pause ARC CI for both jobs: these are timed host-side numbers.
- `NEEDS F2 <- F1` in ORDER.txt.

## 4. Decision rule (registered before any run)

**For weight-sized bulk uploads, F2 gives the empirical answer directly.** `relay/<fmt>/256MiB/all` times the relay
and the direct write of the same bytes in one process, and so is `relay_over_direct`.
- **RELAY** needs `relay_over_direct <= 0.8` on rm16k and tile_bf8, with every relay arm exact.
- Otherwise the verdict is **DIRECT**.

**For other byte counts (a 128k or 254k restore, the full 6.9 GB), `fabric_upload_plan.decide(h2d, fabric,
card_bytes, min_saving_s)` extrapolates.**
- Inputs: F1's `decide_inputs`, and F2's per-pair GB/s (the both-pairs-at-once rate; the socket rate if it is higher
  and exact).
- **NOT-MEASURED**: an input is missing.
- **HOST-BOUND**: the x4 cards alone reach at least 80% of the x16 senders' rate. The links do not set the rate, so
  there is no relay; fix the host path.
- **RELAY**: in the measured overlap mode, the modelled relay saves at least 20% of the direct time **and** at least
  `min_saving_s` (2 s for weights, 20 ms for a 128k restore).
- **DIRECT**: otherwise.

**Stops that apply whatever the times are**
- INEXACT in either job means no relay on that path.
- If F1's `subset` shows no exact one-coordinate write (and F2's `mapper_write` agrees), there is no relay host leg.
  A relay would then need a new host path, for example pinned shard transfers from C++.

## 5. Implementation plan

Every phase is opt-in, and defaults stay byte-identical. Effort is in engineer-days; card sessions are listed apart.

| Phase | What | Effort | Gate |
|---|---|---|---|
| **P0 zero buffers on the device** (independent of F1/F2) | Allocate the KV pool and the buffer pool with `allocate_tensor_on_device`, then fill with `ttnn.zeros_like(t, optional_output_tensor=t)`, the on-device SFPU fill. This removes ~13.7 GB per card of host-to-device traffic, x4 cards included, and the host bf8 packing of zeros. `QWEN_DEVICE_ZERO_FILL=1` | 1-2 d + one card job | an attach-time audit reads every zero buffer back once and compares it to zeros; the ledger is unchanged except time |
| **P1 host path** (independent) | Cache hits skip `shard_w`'s transpose, using the proven lazy-loader pattern. The GDN checkpoint is stored converted, so a restore is copies only | 3-5 d | weight-check audit digests equal; the prefix exactness gate is unchanged |
| F1 + F2 | the two jobs | 0.5 d + ~1 h of cards | section 4 |
| **P2 relay library** (only on RELAY for weights) | `fabric_upload.py`: pairs, a staging pool, the mapper write, `point_to_point` or a socket into the destination, a per-tensor fallback to direct | 4-6 d | CPU fakes as in `test_fabric_upload`; one card job: every relayed shard's digest equals a direct load's |
| **P3 weights through the relay** | a hook in the cache-hit load for interleaved destinations; replicated tensors deduplicated | 2-3 d + one card window | the 120-exact weight-check audit on all four cards; the attach ledger; a gate smoke |
| **P4 restore through the relay** (only on RELAY for restores, after the tiers work lands its region write) | the on-device page scatter (option (a) in 2.4) plus the chunk pipeline | 6-9 d + one card window | the tiers plan's restore exactness (region-read the restored blocks, compare to the spill digests); the salted-cold exactness gate |

**Exactness**

A relay moves pages as opaque bytes:
- `point_to_point` refuses any output spec that differs from its input's (`:118-129`).
- Its kernels are dataflow only (`device/kernels/dataflow`): no compute runs.
- The staged bytes are the host bytes a direct write would send: same spec, same host conversion.

So a relayed shard is byte-identical by construction. It is also **verified**:
- F2 compares every transfer and every relay byte for byte, including that untouched shards keep their sentinel.
- P2 and P3 card jobs run with `QWEN_FABRIC_UPLOAD_VERIFY=1`. Every relayed shard is read back from its receiver and
  compared to the host bytes it came from. A mismatch is fatal at load.
- The weight-check audit compares digests on all four cards against a direct load.
- In production the verify flag is off, since it doubles the load's reads; the audit digests remain the gate.

**Kill switch**
- `QWEN_FABRIC_UPLOAD=off|weights|restore`, default off. Off is today's code path, untouched.
- A file kill switch on the hub mount (`fabric-upload.off`), read at attach as the other levers do.
- **Automatic fallback.** Any exception from the mapper write, `point_to_point` or the socket latches the relay off
  for the process, and that tensor or chunk is written direct.
  - The latch logs once (`[QWEN-FABRIC-UPLOAD] latched off: <reason>`) and is counted.
  - With verify on, a mismatch is fatal instead.
- The relay never runs inside a captured trace. Weights load before capture; restores run between steps, where the
  direct writes already run.

**Order of work.**
1. P0 and P1 now: they shorten engine start and restores whatever F1/F2 say.
2. F1 and F2 next: about 1 h of cards.
3. P2-P4 only on a RELAY verdict for that use.
