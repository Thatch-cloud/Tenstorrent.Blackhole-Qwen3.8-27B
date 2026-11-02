// qwen_kv_read: a standalone nanobind extension adding ttnn.qwen_read_blocks (the prefix audit's region read) WITHOUT
// replacing any of the image's pinned binaries (_ttnn.so, _ttnncpp.so, libtt_metal.so keep their hashes and their QWEN_
// strings). It is optimisation/sim/kv-region-read.patch re-cut as a module: the same function body, compiled against the
// served tt-metal tree, linking the image's own libraries. Import it after ttnn; it sets ttnn.qwen_read_blocks.
//
// Version 2 adds the host KV tier's device side (docs/prefix-store-hygiene.md, QWEN_PREFIX_HOST_TIER_GIB): the same region transfers of whole
// dimension-0 slices (paged KV cache blocks), but of the RAW PACKED BYTES, to and from a plain numpy buffer - no ttnn host tensor, no unpack,
// no repack, so what is written back is bit for bit what was read:
//   ttnn.qwen_block_bytes(device_tensor) -> bytes of one dim-0 slice on one device
//   ttnn.qwen_read_blocks_raw(device_tensor, out, blocks)   out: uint8 C-contiguous [devices, len(blocks), slice bytes]
//   ttnn.qwen_write_blocks_raw(device_tensor, data, blocks) data: the same shape; device d's slice blocks[i] gets data[d, i]
// Devices are in the mesh's row-major coordinate order. Runs of consecutive block ids are one region transfer each; both calls return when the
// transfers have completed (blocking), release the GIL while they wait, run no program and allocate nothing on the device. (These are the v2/v3
// ORACLES: card tests only.)
//
// Version 3 added the DEVICE side of the peer-to-peer KV restore (docs/tp4-p2p-restore.md): a staging buffer in card DRAM that a drive can DMA
// into (one contiguous range in one DRAM bank, never handed to anything else) and device programs, unlike everything above, which run no program.
//
// Version 4 ("v4-lite", the P2P KV offload of the Gemma serving image; ported to tt-metal 62443b62) makes the staging a RING OF BANKS and adds the
// spill direction and the integrity digest. All ops enqueue on one command queue (CQ0), never block, and validate every argument BEFORE they enqueue anything:
//   ttnn.qwen_p2p_staging_alloc(mesh_device, bytes_per_bank, bottom_up=False, page_bytes=64 KiB) -> P2PStaging
//       one lockstep interleaved allocation of 8 x bytes_per_bank per card: bank b of every card holds the contiguous range [A, A + bytes_per_bank)
//   ttnn.qwen_p2p_staging_info(staging) -> dict(address, size, page_bytes, aligned_4k_offset / _address / _size, banks=[{bank, noc_x, noc_y, ...}],
//       devices=[{mesh_coord, device_id, pci_bus_id, ...}], ...)
//   ttnn.qwen_blocks_to_staging(tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0)       GATHER: KV rows -> staging
//       staging[byte_offset + i * row_stride + j * page_bytes] = page (blocks[i] * pages_per_block + j) of the tensor
//   ttnn.qwen_write_blocks_from_staging(tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0)   SCATTER: staging -> KV rows
//   ttnn.qwen_blocks_digest_async(tensors, blocks, pages_per_block, cq_id=0) -> DigestHandle(ready(), result() -> [device][block] uint64)
//       the checksum of each block's SHARD (its rows in the tensors, in the order given); one 64-byte partial per block, read back without blocking
//   ttnn.qwen_p2p_record_event(mesh_device, cq_id=0) -> P2PEvent(query() -> bool)          distributed::EventQuery on an event recorded to the host
//   ttnn.qwen_p2p_warm(staging, tensors, null_block=0)   gather -> digest -> scatter-back of the null block, each program run to completion
//   ttnn.qwen_p2p_freeze(staging)                         after it a program key that was not warmed raises (no JIT, no allocation behind live traces)
//   ttnn.qwen_staging_checksum / qwen_p2p_staging_write / qwen_p2p_staging_read       (ORACLES and diagnostics: card tests only; blocking)
// Every op refuses (QwenP2PRefused, a ValueError: nothing was enqueued) during trace capture, on bad arguments, after freeze for a new program key, and on a
// released staging; an enqueue that THROWS raises QwenP2PEnqueueError (a RuntimeError: the queue may hold part of that op, the caller quarantines the slot).
// The kernels (qwen_p2p_scatter.cpp, qwen_p2p_gather.cpp, qwen_p2p_checksum.cpp, qwen_p2p_digest_rows.cpp) are read from the directory of this module and
// JIT-compiled at the first dispatch (qwen_p2p_warm does that, before any trace); qwen_p2p_plan.hpp is the pure arithmetic (compiled and checked by the CPU
// test). NOT YET RUN ON A CARD: p2p_staging_card.py.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "qwen_p2p_plan.hpp"
#include "tt_metal/impl/tensor/mesh_tensor_impl.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace nb = nanobind;

namespace qwen_kv {

// Reads whole slices along dimension 0 of a mesh tensor (for a paged KV cache: whole blocks) into the front of a host
// tensor's shards, every device's shard of the mesh, slice indices[i] to host slice i. Runs of consecutive indices are
// one region read each; the call returns when every read has landed. Reads only: no program, no device allocation.
void read_dim0_slices(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const tt::tt_metal::MeshTensor& device_tensor,
    tt::tt_metal::HostTensor& host_tensor,
    const std::vector<uint32_t>& indices) {
    using namespace tt::tt_metal;
    const auto& spec = device_tensor.tensor_spec();
    const uint64_t slices = spec.logical_shape()[0];
    const uint64_t total_bytes = spec.compute_packed_buffer_size_bytes();
    TT_FATAL(slices > 0 && total_bytes % slices == 0, "A dim-0 slice read needs a whole number of bytes per slice");
    const uint64_t slice_bytes = total_bytes / slices;
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");
    TT_FATAL(host_tensor.logical_shape()[0] == indices.size(), "Host tensor holds a different number of slices");
    for (uint32_t index : indices) {
        TT_FATAL(index < slices, "Slice {} is outside the {} slices of the device tensor", index, slices);
    }

    std::vector<std::pair<uint64_t, uint64_t>> runs;  // (first index, count) of each run of consecutive indices
    for (uint32_t index : indices) {
        if (!runs.empty() && runs.back().first + runs.back().second == index) {
            ++runs.back().second;
        } else {
            runs.emplace_back(index, 1);
        }
    }

    distributed::MeshCoordinateRange all_coords(queue.device()->shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    std::vector<distributed::ShardDataTransfer> transfers;
    transfers.reserve(coords.size() * runs.size());
    for (const auto& coord : coords) {
        auto host_buffer = host_tensor.buffer().get_shard(coord);
        TT_FATAL(host_buffer.has_value(), "Host shard for device shard {} is not populated.", coord);
        auto bytes = host_buffer->view_bytes();
        TT_FATAL(
            bytes.size() >= indices.size() * slice_bytes,
            "Host shard for device shard {} is too small: {} < {}",
            coord,
            bytes.size(),
            indices.size() * slice_bytes);
        std::byte* base = const_cast<std::byte*>(bytes.data());
        uint64_t host_offset = 0;
        for (const auto& [first, count] : runs) {
            transfers.push_back(distributed::ShardDataTransfer{coord}
                                    .host_data(base + host_offset)
                                    .region(BufferRegion(first * slice_bytes, count * slice_bytes)));
            host_offset += count * slice_bytes;
        }
    }
    queue.enqueue_read_shards(transfers, device_tensor.impl().raw_mesh_buffer(), /*blocking=*/true);
}

// Bytes of one dimension-0 slice of a mesh tensor on one device (for a paged KV cache: one block).
uint64_t slice_bytes_of(const tt::tt_metal::MeshTensor& device_tensor) {
    const auto& spec = device_tensor.tensor_spec();
    const uint64_t slices = spec.logical_shape()[0];
    const uint64_t total_bytes = spec.compute_packed_buffer_size_bytes();
    TT_FATAL(slices > 0 && total_bytes % slices == 0, "A dim-0 slice transfer needs a whole number of bytes per slice");
    return total_bytes / slices;
}

// The raw twin of read_dim0_slices and its mirror: moves slices indices[i] of every device's shard between the device tensor and a plain host
// buffer laid out [device in row-major coordinate order][i][slice bytes]. write = true copies host to device, false device to host. The
// indices must be distinct (two writes to one slice have no defined order). Blocking.
void transfer_raw_dim0_slices(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const tt::tt_metal::MeshTensor& device_tensor,
    std::byte* host,
    uint64_t host_bytes,
    const std::vector<uint32_t>& indices,
    bool write) {
    using namespace tt::tt_metal;
    const uint64_t slices = device_tensor.tensor_spec().logical_shape()[0];
    const uint64_t slice_bytes = slice_bytes_of(device_tensor);
    for (uint32_t index : indices) {
        TT_FATAL(index < slices, "Slice {} is outside the {} slices of the device tensor", index, slices);
    }
    {
        std::vector<uint32_t> sorted(indices);
        std::sort(sorted.begin(), sorted.end());
        TT_FATAL(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "A raw slice transfer needs distinct slice indices");
    }

    std::vector<std::pair<uint64_t, uint64_t>> runs;  // (first index, count) of each run of consecutive indices
    for (uint32_t index : indices) {
        if (!runs.empty() && runs.back().first + runs.back().second == index) {
            ++runs.back().second;
        } else {
            runs.emplace_back(index, 1);
        }
    }

    distributed::MeshCoordinateRange all_coords(queue.device()->shape());
    std::vector<distributed::MeshCoordinate> coords(all_coords.begin(), all_coords.end());
    const uint64_t per_device = indices.size() * slice_bytes;
    TT_FATAL(
        host_bytes == coords.size() * per_device,
        "Host buffer holds {} bytes, a transfer of {} slices on {} devices needs {}",
        host_bytes,
        indices.size(),
        coords.size(),
        coords.size() * per_device);
    std::vector<distributed::ShardDataTransfer> transfers;
    transfers.reserve(coords.size() * runs.size());
    for (size_t device = 0; device < coords.size(); ++device) {
        std::byte* base = host + device * per_device;
        uint64_t host_offset = 0;
        for (const auto& [first, count] : runs) {
            transfers.push_back(distributed::ShardDataTransfer{coords[device]}
                                    .host_data(base + host_offset)
                                    .region(BufferRegion(first * slice_bytes, count * slice_bytes)));
            host_offset += count * slice_bytes;
        }
    }
    if (write) {
        queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), transfers, /*blocking=*/true);
    } else {
        queue.enqueue_read_shards(transfers, device_tensor.impl().raw_mesh_buffer(), /*blocking=*/true);
    }
}

}  // namespace qwen_kv

// ---------------------------------------------------------------------------------------------------------------------------------------
// Versions 3 and 4: the device side of the peer-to-peer KV offload.
//
// HOW THE STAGING IS PLACED (docs/tp4-p2p-restore.md, section 1). tt-metal's DRAM allocator (BankManager, one free list) places every DRAM
// buffer in LOCKSTEP over all banks: an allocation reserves the SAME address range [a, a + size_per_bank) in every one of the 8 banks of every
// device of the mesh, whatever it holds, so a single-bank reservation does not exist. An interleaved buffer of 8 x S bytes (page size P, S a
// multiple of P) puts its pages p = 0, 1, 2, ... on bank p mod 8 at a + (p div 8) * P, so the pages of ONE bank are contiguous: bytes [a, a + S) of
// bank b are one contiguous range, the allocator never hands it to anything else while the buffer lives, and a TLB window aimed at bank b
// reaches it. The cost is the buffer: 8 x S per card. Version 4 uses all 8 banks as 8 staging slots (banks 0-1 spill, 2-7 restore); the ops name the bank.
//
// THE LIFETIME RULES. The staging handle (P2PStaging) owns a shared state; every op holds a reference to that state for its whole call, so a handle dropped
// by the garbage collector in another thread cannot pull the state out from under a running op. The host buffers of an asynchronous digest readback outlive
// their handle (the staging keeps the in-flight ones until their event has completed): the completion-queue reader thread writes into them.
namespace qwen_p2p_host {

using tt::tt_metal::BufferRegion;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::ShardDataTransfer;

// Raised BEFORE anything was enqueued: a bad argument, a trace capture in progress, a released staging, a program key that was not warmed before freeze, a closed
// mesh. The caller may carry on (it has nothing to undo on the device). Registered as QwenP2PRefused, a ValueError.
struct P2PRefused : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// Raised when an enqueue itself THREW: the command queue may hold part of that op (and the earlier ops of the caller's sequence ran or will run). The caller
// quarantines the staging slot, enqueues a trailing event, and latches the tier. Registered as QwenP2PEnqueueError, a RuntimeError.
struct P2PEnqueueError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void refuse(const std::string& op, const std::string& what) { throw P2PRefused(op + ": " + what); }

// A cached program: built once per distinct compile-time arguments, then only its runtime args change between calls.
struct CachedProgram {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::KernelHandle kernel{};
    std::vector<CoreCoord> cores;
    bool ran = false;        // enqueued at least once
    bool completed = false;  // ran to completion under a blocking finish (qwen_p2p_warm, or a blocking oracle): freeze requires it of every program
};
using ProgramMap = std::map<std::vector<uint32_t>, std::unique_ptr<CachedProgram>>;

// One asynchronous digest: the host buffer the partials are read back into ([device][block][64 bytes]) and the event recorded behind that read.
struct DigestPending {
    std::vector<uint8_t> host;
    size_t devices = 0;
    size_t blocks = 0;
    std::weak_ptr<MeshDevice> mesh;
    std::optional<tt::tt_metal::distributed::MeshEvent> event;
};

struct StagingState {
    std::weak_ptr<MeshDevice> mesh;
    uint64_t size_bytes = 0;  // bytes per bank
    uint64_t page_bytes = 0;
    uint32_t num_banks = 0;
    uint64_t address = 0;  // the allocator's address of the range (the same in every bank of every device)
    std::vector<int32_t> bank_offsets;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>> partials;  // the digest's device-side partials: qwen_p2p::kDigestRingBuffers of them
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> checksum_partials;      // the oracle's, allocated at its first use (card tests only)
    tt::tt_metal::Statistics dram_before{};
    tt::tt_metal::Statistics dram_after{};
    std::mutex mutex;
    ProgramMap scatter, gather, digest, checksum;
    bool frozen = false;
    bool warmed = false;
    uint32_t digest_serial = 0;
    std::vector<std::shared_ptr<DigestPending>> inflight;

    bool released() const { return buffer == nullptr; }
};

namespace {
std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}
// One live staging per mesh: the digest op has no staging argument and finds its partials and program cache through the mesh of its tensors.
std::map<const MeshDevice*, std::weak_ptr<StagingState>>& registry() {
    static std::map<const MeshDevice*, std::weak_ptr<StagingState>> map;
    return map;
}
}  // namespace

std::shared_ptr<StagingState> find_staging(const MeshDevice* mesh) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = registry().find(mesh);
    if (it == registry().end()) {
        return nullptr;
    }
    auto state = it->second.lock();
    if (!state) {
        registry().erase(it);
    }
    return state;
}

// True once the event of a pending digest has completed (or can never complete: the mesh is gone and nothing writes the buffer any more).
bool pending_done(const DigestPending& pending) {
    auto mesh = pending.mesh.lock();
    if (!mesh || !mesh->is_initialized() || !pending.event.has_value()) {
        return true;
    }
    return tt::tt_metal::distributed::EventQuery(*pending.event);
}

// Drops the in-flight digests that have completed. Call with the staging mutex held.
void reap(StagingState& state) {
    auto& list = state.inflight;
    list.erase(std::remove_if(list.begin(), list.end(), [](const std::shared_ptr<DigestPending>& pending) { return pending_done(*pending); }), list.end());
}

// True while a trace is being captured on the queue. QWEN_P2P_TEST_TRACE_ACTIVE=1 forces it (a test hook for the mock device, which cannot capture a trace: the
// refusal path is otherwise first exercised by a real capture on a card).
bool trace_active(MeshDevice& mesh, uint32_t cq_id) {
    if (const char* forced = std::getenv("QWEN_P2P_TEST_TRACE_ACTIVE")) {
        if (forced[0] == '1') {
            return true;
        }
    }
    return mesh.mesh_command_queue(static_cast<uint8_t>(cq_id)).trace_id().has_value();
}

// Refusal during trace capture: the ops allocate, JIT and read back, none of which is allowed behind a live trace (a new binary during capture is a TT_FATAL).
void check_queue_usable(MeshDevice& mesh, uint32_t cq_id, const std::string& op) {
    if (!mesh.is_initialized()) {
        refuse(op, "the mesh device is closed");
    }
    if (cq_id != 0) {
        // Never CQ1: it needs num_command_queues=2 at open, every trace re-qualified, and it gives no concurrency without sub-devices. cq_id is in the signature so that
        // the call says which queue it means, and so that this refusal is the one place that says no.
        refuse(op, "cq_id " + std::to_string(cq_id) + ": only command queue 0 is used (never CQ1)");
    }
    if (mesh.num_hw_cqs() == 0) {
        refuse(op, "the mesh has no command queue");
    }
    if (trace_active(mesh, 0)) {
        refuse(op, "refused while a trace is being captured on command queue 0 (warm up and freeze before the first capture)");
    }
}

std::vector<MeshCoordinate> mesh_coordinates(MeshDevice& mesh) {
    const MeshCoordinateRange all(mesh.shape());
    return std::vector<MeshCoordinate>(all.begin(), all.end());
}

// The directory holding the kernels: next to this module, or QWEN_P2P_KERNEL_DIR.
std::string kernel_directory() {
    if (const char* forced = std::getenv("QWEN_P2P_KERNEL_DIR")) {
        if (*forced != 0) {
            return forced;
        }
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&kernel_directory), &info) != 0 && info.dli_fname != nullptr) {
        return std::filesystem::absolute(std::filesystem::path(info.dli_fname)).parent_path().string();
    }
    return std::string();
}

std::string kernel_path(const char* name) {
    const std::string path = kernel_directory() + "/" + name;
    if (!std::filesystem::exists(path)) {
        throw P2PRefused("the P2P kernel " + path + " is missing (it ships beside qwen_kv_read.so; QWEN_P2P_KERNEL_DIR overrides)");
    }
    return path;
}

// PCI address of a tt-kmd device N (/dev/tenstorrent/N) from sysfs, "" when it cannot be read (a container without /sys/class).
std::string pci_bus_id_of(uint32_t pcie_device) {
    std::error_code error;
    const auto link = std::filesystem::read_symlink("/sys/class/tenstorrent/tenstorrent!" + std::to_string(pcie_device) + "/device", error);
    return error ? std::string() : link.filename().string();
}

std::vector<CoreCoord> core_list(uint32_t count, uint32_t columns) {
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < count; ++i) {
        cores.emplace_back(i % columns, i / columns);
    }
    return cores;
}

tt::tt_metal::CoreRangeSet core_range_set(uint32_t count, uint32_t columns) {
    std::set<tt::tt_metal::CoreRange> ranges;
    const uint32_t full_rows = count / columns;
    if (full_rows > 0) {
        ranges.insert(tt::tt_metal::CoreRange(CoreCoord(0, 0), CoreCoord(columns - 1, full_rows - 1)));
    }
    if (count % columns != 0) {
        ranges.insert(tt::tt_metal::CoreRange(CoreCoord(0, full_rows), CoreCoord(count % columns - 1, full_rows)));
    }
    return tt::tt_metal::CoreRangeSet(ranges);
}

void check_grid(MeshDevice& mesh, uint32_t count, uint32_t columns, const char* what) {
    const auto grid = mesh.compute_with_storage_grid_size();
    const uint32_t rows = qwen_p2p::ceil_div(count, columns);
    if (grid.x < columns || grid.y < rows) {
        refuse(what, "the compute grid is smaller than " + std::to_string(columns) + " x " + std::to_string(rows));
    }
}

// Builds the program (one kernel on `count` cores in rows of `columns`, one circular buffer of cb_bytes as L1 scratch) with runtime args of
// `arg_count` words, as a MeshWorkload over the whole mesh. The runtime args are placeholders until the caller fills them. Nothing is compiled here: the
// JIT runs at the first enqueue.
std::unique_ptr<CachedProgram> build_program(
    MeshDevice& mesh,
    const char* kernel_name,
    const std::vector<uint32_t>& compile_args,
    uint32_t count,
    uint32_t columns,
    uint32_t cb_bytes,
    uint32_t arg_count) {
    using namespace tt::tt_metal;
    check_grid(mesh, count, columns, kernel_name);
    auto cached = std::make_unique<CachedProgram>();
    Program program = CreateProgram(mesh);
    const CoreRangeSet cores = core_range_set(count, columns);
    CreateCircularBuffer(program, cores, CircularBufferConfig(cb_bytes, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 64));
    DataMovementConfig config;
    config.processor = DataMovementProcessor::RISCV_0;
    config.noc = NOC::RISCV_0_default;
    config.compile_args = compile_args;
    cached->kernel = CreateKernel(program, kernel_path(kernel_name), cores, config);
    cached->cores = core_list(count, columns);
    const std::vector<uint32_t> placeholder(arg_count, 0);
    for (const auto& core : cached->cores) {
        SetRuntimeArgs(program, cached->kernel, core, placeholder);
    }
    cached->workload.add_program(MeshCoordinateRange(mesh.shape()), std::move(program));
    return cached;
}

// The program of a cached workload (it holds exactly one, over the whole mesh).
tt::tt_metal::Program& program_of(CachedProgram& cached) { return cached.workload.get_programs().begin()->second; }

void set_args(CachedProgram& cached, size_t core, const std::vector<uint32_t>& values) {
    auto& args = tt::tt_metal::GetRuntimeArgs(program_of(cached), cached.kernel, cached.cores[core]);
    for (size_t i = 0; i < values.size(); ++i) {
        args[i] = values[i];
    }
}

// The cached program for `key`, built on first use. After freeze a missing key is refused: building it would JIT-compile and load a binary, which behind live
// traces is a TT_FATAL or a corrupt capture. Call with the staging mutex held.
template <typename Builder>
CachedProgram& program_for(StagingState& state, ProgramMap& map, const std::vector<uint32_t>& key, const char* op, Builder build) {
    auto it = map.find(key);
    if (it != map.end()) {
        return *it->second;
    }
    if (state.frozen) {
        refuse(op, "a program key that was not warmed before freeze (the compile-time arguments differ from every warmed call)");
    }
    std::unique_ptr<CachedProgram> built = build();
    auto& slot = map[key];
    slot = std::move(built);
    return *slot;
}

// Enqueues the program non-blocking on `cq_id`. A throw here is NOT a refusal: the queue may hold part of the op.
void enqueue_program(MeshDevice& mesh, uint32_t cq_id, CachedProgram& program, const char* op) {
    try {
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh.mesh_command_queue(static_cast<uint8_t>(cq_id)), program.workload, /*blocking=*/false);
        program.ran = true;
    } catch (const std::exception& error) {
        throw P2PEnqueueError(std::string(op) + ": the enqueue threw (" + error.what() + "): the command queue may hold part of this op");
    }
}

// What an op needs to know about one KV tensor: its buffer, its geometry (slices = blocks, bytes per slice, page size) and its address.
struct TensorView {
    MeshDevice* mesh = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    const tt::tt_metal::Buffer* reference = nullptr;
    uint64_t slices = 0;
    uint64_t slice_bytes = 0;
    uint64_t page_bytes = 0;
    uint32_t address = 0;
};

TensorView view_tensor(const ttnn::Tensor& tensor, const char* op) {
    using namespace tt::tt_metal;
    TensorView view;
    view.mesh = tensor.device();
    if (view.mesh == nullptr || !tensor.is_allocated()) {
        refuse(op, "the tensor must be allocated on a mesh device");
    }
    const MeshTensor& mesh_tensor = tensor.mesh_tensor();
    view.buffer = mesh_tensor.impl().raw_mesh_buffer();
    view.reference = view.buffer->get_reference_buffer();
    if (view.reference->buffer_type() != BufferType::DRAM || view.reference->buffer_layout() != TensorMemoryLayout::INTERLEAVED) {
        refuse(op, "the tensor must be an interleaved DRAM tensor");
    }
    try {
        view.slices = mesh_tensor.tensor_spec().logical_shape()[0];
        view.slice_bytes = qwen_kv::slice_bytes_of(mesh_tensor);
    } catch (const std::exception& error) {
        refuse(op, std::string("the tensor has no whole number of bytes per block (") + error.what() + ")");
    }
    view.page_bytes = view.reference->page_size();
    if (view.reference->aligned_page_size() != view.page_bytes) {
        refuse(op, "the tensor's page must be DRAM aligned");
    }
    const uint64_t reference_address = view.reference->address();  // (a DeviceAddr: 32 bits in some trees, 64 in others)
    if (reference_address >= (uint64_t(1) << 32)) {
        refuse(op, "the tensor's address does not fit 32 bits");
    }
    view.address = static_cast<uint32_t>(reference_address);
    // The program is built once for the reference buffer and its addresses go to every device: every device's buffer must be laid out alike.
    for (const auto& coord : mesh_coordinates(*view.mesh)) {
        const Buffer* device_buffer = view.buffer->get_device_buffer(coord);
        if (device_buffer == nullptr || device_buffer->address() != view.reference->address() || device_buffer->page_size() != view.reference->page_size() ||
            device_buffer->buffer_layout() != view.reference->buffer_layout() || device_buffer->buffer_type() != view.reference->buffer_type()) {
            refuse(op, "the tensor's buffers differ between devices (address, page size or layout)");
        }
    }
    return view;
}

std::string with_index(const char* problem, size_t index, const std::vector<uint32_t>& blocks) {
    return std::string(problem) + " (blocks[" + std::to_string(index) + "] = " + std::to_string(index < blocks.size() ? blocks[index] : 0) + ")";
}

}  // namespace qwen_p2p_host

using tt::tt_metal::distributed::MeshDevice;

// The handle Python holds. The state lives as long as the handle OR any running op.
class P2PStaging {
public:
    P2PStaging(MeshDevice& mesh, uint64_t size_bytes, uint64_t page_bytes, bool bottom_up) : state_(std::make_shared<qwen_p2p_host::StagingState>()) {
        using namespace tt::tt_metal;
        using qwen_p2p_host::refuse;
        const char* op = "qwen_p2p_staging_alloc";
        qwen_p2p_host::check_queue_usable(mesh, 0, op);
        auto& state = *state_;
        state.mesh = mesh.shared_from_this();
        state.size_bytes = size_bytes;
        state.page_bytes = page_bytes;
        {
            std::lock_guard<std::mutex> lock(qwen_p2p_host::registry_mutex());
            auto it = qwen_p2p_host::registry().find(&mesh);
            if (it != qwen_p2p_host::registry().end()) {
                // (an entry whose mesh is gone is stale: a handle that outlived its mesh, and a new mesh at the same address)
                const auto existing = it->second.lock();
                if (existing && !existing->mesh.expired()) {
                    refuse(op, "this mesh already has a staging buffer (one lockstep allocation per mesh; release() the first)");
                }
                qwen_p2p_host::registry().erase(it);
            }
        }
        auto* allocator = mesh.allocator().get();
        state.num_banks = allocator->get_num_banks(BufferType::DRAM);
        if (const char* problem = qwen_p2p::check_staging(size_bytes, page_bytes, state.num_banks, allocator->get_bank_size(BufferType::DRAM))) {
            refuse(op, problem);
        }
        state.dram_before = allocator->get_statistics(BufferType::DRAM);
        try {
            distributed::DeviceLocalBufferConfig local;
            local.page_size = page_bytes;
            local.buffer_type = BufferType::DRAM;
            local.bottom_up = bottom_up;
            state.buffer = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = size_bytes * state.num_banks}, local, &mesh);
            state.address = state.buffer->address();
            // The digest's device-side partials: one whole 64-byte page per block of a call, kDigestRingBuffers buffers that take turns. They are allocated here, before
            // any trace is captured (an allocation after the capture is flagged unsafe).
            distributed::DeviceLocalBufferConfig partial_local;
            partial_local.page_size = qwen_p2p::kDigestPartialBytes;
            partial_local.buffer_type = BufferType::DRAM;
            for (uint32_t i = 0; i < qwen_p2p::kDigestRingBuffers; ++i) {
                state.partials.push_back(distributed::MeshBuffer::create(
                    distributed::ReplicatedBufferConfig{.size = uint64_t(qwen_p2p::kDigestMaxBlocks) * qwen_p2p::kDigestPartialBytes}, partial_local, &mesh));
            }
        } catch (const std::exception& error) {
            state.partials.clear();
            state.buffer.reset();
            refuse(op, std::string("the allocation failed (") + error.what() + ")");
        }
        state.dram_after = allocator->get_statistics(BufferType::DRAM);
        for (uint32_t bank = 0; bank < state.num_banks; ++bank) {
            state.bank_offsets.push_back(allocator->get_bank_offset(BufferType::DRAM, bank));
        }
        auto fail = [&](const std::string& what) {
            state.partials.clear();
            state.buffer.reset();
            refuse(op, what);
        };
        if (state.address % qwen_p2p::kDramAlignment != 0) {
            fail("the staging buffer address " + std::to_string(state.address) + " is not 64-byte aligned");
        }
        // Sanity of the MeshBuffer (it creates every device's Buffer at the one address, so this cannot fail by itself): same address, page size asked for.
        // What it does NOT prove is that no other allocator hands out the range: the guarantee is that THIS MeshDevice's allocator never does (a submesh or
        // a second MeshDevice over the same chips has its own free list, and so has a single-device allocation), so the engine must allocate DRAM only
        // through the MeshDevice it passes here.
        for (const auto& coord : qwen_p2p_host::mesh_coordinates(mesh)) {
            const Buffer* device_buffer = state.buffer->get_device_buffer(coord);
            if (device_buffer == nullptr || device_buffer->address() != state.address || device_buffer->page_size() != page_bytes ||
                device_buffer->aligned_page_size() != page_bytes || device_buffer->buffer_layout() != TensorMemoryLayout::INTERLEAVED) {
                fail("the staging buffer is not an interleaved buffer at one address with the requested page size");
            }
        }
        std::lock_guard<std::mutex> lock(qwen_p2p_host::registry_mutex());
        qwen_p2p_host::registry()[&mesh] = state_;
    }

    ~P2PStaging() { release(); }
    P2PStaging(const P2PStaging&) = delete;
    P2PStaging& operator=(const P2PStaging&) = delete;

    // Frees the staging and everything cached with it (the programs hold device memory for their binaries). Call it before the mesh is closed. If the
    // mesh is already closed the Programs are LEAKED on purpose: their destructors unregister from Device objects that close() has destroyed, which
    // is a use after free (the MeshBuffers are safe, they know a closed mesh). On an open mesh it first waits (at most 10 s) for the queue to pass an event recorded
    // behind everything enqueued so far; if the queue does not, nothing is freed (see below).
    void release() {
        auto& state = *state_;
        std::lock_guard<std::mutex> lock(state.mutex);
        {
            std::lock_guard<std::mutex> registry_lock(qwen_p2p_host::registry_mutex());
            for (auto it = qwen_p2p_host::registry().begin(); it != qwen_p2p_host::registry().end();) {
                auto alive = it->second.lock();
                it = (alive.get() == &state || !alive) ? qwen_p2p_host::registry().erase(it) : std::next(it);
            }
        }
        const auto locked = state.mesh.lock();
        const bool open = locked && locked->is_initialized();
        // What is freed below (the programs' binaries, the staging, the partials) must not be in use by commands still in the queue: an event recorded behind
        // everything enqueued so far has to complete first, within a bounded time (a hung device must not hang the shutdown). If it does not, everything is LEAKED on
        // purpose, like the closed-mesh case, and so are the host buffers of the digests in flight (the completion reader thread writes into them).
        bool drained = !open;
        static std::vector<std::shared_ptr<qwen_p2p_host::DigestPending>> leaked_pending;
        static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>> leaked_buffers;
        if (open) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            try {
                const tt::tt_metal::distributed::MeshEvent event = locked->mesh_command_queue(0).enqueue_record_event_to_host();
                while (!tt::tt_metal::distributed::EventQuery(event) && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                drained = tt::tt_metal::distributed::EventQuery(event);
            } catch (const std::exception&) {
                drained = false;
            }
            for (auto& pending : state.inflight) {
                if (!drained || !qwen_p2p_host::pending_done(*pending)) {
                    leaked_pending.push_back(pending);
                }
            }
        }
        if (!drained || !open) {
            for (auto* map : {&state.scatter, &state.gather, &state.digest, &state.checksum}) {
                for (auto& entry : *map) {
                    (void)entry.second.release();
                }
            }
            if (open) {
                for (auto& buffer : state.partials) {
                    leaked_buffers.push_back(buffer);
                }
                leaked_buffers.push_back(state.checksum_partials);
                leaked_buffers.push_back(state.buffer);
            }
        }
        try {
            state.scatter.clear();
            state.gather.clear();
            state.digest.clear();
            state.checksum.clear();
            state.inflight.clear();
            state.partials.clear();
            state.checksum_partials.reset();
            state.buffer.reset();
        } catch (const std::exception&) {
            // The mesh may already be closed: its memory is gone with it.
        }
    }

    bool released() const { return state_->released(); }
    bool frozen() const { return state_->frozen; }
    bool warmed() const { return state_->warmed; }
    uint64_t size_bytes() const { return state_->size_bytes; }
    uint64_t page_bytes() const { return state_->page_bytes; }
    uint32_t num_banks() const { return state_->num_banks; }
    uint64_t address() const { return state_->address; }
    const std::shared_ptr<qwen_p2p_host::StagingState>& state() const { return state_; }

private:
    std::shared_ptr<qwen_p2p_host::StagingState> state_;
};

// The result of qwen_blocks_digest_async: ready() is a non-blocking EventQuery; result() is the digest of each block on each device once it is.
class DigestHandle {
public:
    explicit DigestHandle(std::shared_ptr<qwen_p2p_host::DigestPending> pending) : pending_(std::move(pending)) {}

    bool ready() const {
        auto mesh = pending_->mesh.lock();
        if (!mesh || !mesh->is_initialized()) {
            qwen_p2p_host::refuse("DigestHandle.ready", "the mesh device is closed");
        }
        return tt::tt_metal::distributed::EventQuery(*pending_->event);
    }

    // [device][block] -> the uint64 digest. Only after ready(): it never waits (a wait would spin a core, and tell nothing the worker's service loop does not).
    std::vector<std::vector<uint64_t>> result() const {
        if (!ready()) {
            qwen_p2p_host::refuse("DigestHandle.result", "called before ready() was true");
        }
        const auto& pending = *pending_;
        std::vector<std::vector<uint64_t>> out(pending.devices, std::vector<uint64_t>(pending.blocks, 0));
        for (size_t device = 0; device < pending.devices; ++device) {
            for (size_t block = 0; block < pending.blocks; ++block) {
                // partial page: the 64-bit sum as low word, high word (little endian), the rest zero
                std::memcpy(&out[device][block], pending.host.data() + (device * pending.blocks + block) * qwen_p2p::kDigestPartialBytes, 8);
            }
        }
        return out;
    }

    size_t blocks() const { return pending_->blocks; }
    size_t devices() const { return pending_->devices; }

private:
    std::shared_ptr<qwen_p2p_host::DigestPending> pending_;
};

// An event recorded to the host behind the ops enqueued so far: query() is distributed::EventQuery.
class P2PEvent {
public:
    P2PEvent(const std::shared_ptr<MeshDevice>& mesh, tt::tt_metal::distributed::MeshEvent event) : mesh_(mesh), event_(std::move(event)) {}

    bool query() const {
        auto mesh = mesh_.lock();
        if (!mesh || !mesh->is_initialized()) {
            qwen_p2p_host::refuse("P2PEvent.query", "the mesh device is closed");
        }
        return tt::tt_metal::distributed::EventQuery(event_);
    }

private:
    std::weak_ptr<MeshDevice> mesh_;
    tt::tt_metal::distributed::MeshEvent event_;
};

namespace qwen_p2p_host {

// The shared body of the gather (KV rows -> staging) and the scatter (staging -> KV rows): same cores, same chunks, same runtime args; the kernel differs.
struct RowsOp {
    const char* name;
    const char* kernel;
    bool gather;
};
constexpr RowsOp kGatherOp{"qwen_blocks_to_staging", "qwen_p2p_gather.cpp", true};
constexpr RowsOp kScatterOp{"qwen_write_blocks_from_staging", "qwen_p2p_scatter.cpp", false};

// No GIL needed (the binding releases it). Every check comes before the first enqueue; one launch (at most 512 blocks) is the only enqueue, so a throw there is
// the only way to leave a half-enqueued op, and it is reported as such.
void rows_op(
    const RowsOp& kind,
    const ttnn::Tensor& tensor,
    P2PStaging& handle,
    uint32_t bank,
    uint64_t byte_offset,
    const std::vector<uint32_t>& blocks,
    uint64_t row_stride,
    uint32_t pages_per_block,
    uint32_t cq_id,
    CachedProgram** used = nullptr) {
    using namespace tt::tt_metal;
    const std::string op = kind.name;
    const auto state_keeper = handle.state();
    StagingState& state = *state_keeper;
    const TensorView view = view_tensor(tensor, kind.name);
    const auto mesh = state.mesh.lock();
    if (!mesh || !mesh->is_initialized()) {
        refuse(op, "the mesh device of this staging buffer is closed");
    }
    if (view.mesh != mesh.get()) {
        refuse(op, "the tensor and the staging buffer are on different meshes");
    }
    check_queue_usable(*mesh, cq_id, op);
    if (const char* problem = qwen_p2p::check_bank(bank, state.num_banks)) {
        refuse(op, problem);
    }
    if (const char* problem = qwen_p2p::check_block_rows(byte_offset, view.slice_bytes, view.page_bytes, pages_per_block, blocks.size(), state.size_bytes, row_stride)) {
        refuse(op, problem);
    }
    size_t bad = 0;
    if (const char* problem = qwen_p2p::check_block_ids(blocks, view.slices, /*require_distinct=*/!kind.gather, &bad)) {
        refuse(op, with_index(problem, bad, blocks));
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.released()) {
        refuse(op, "the staging buffer was released");
    }
    std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(view.page_bytes), pages_per_block, qwen_p2p::kScatterChunkPages, qwen_p2p::kScatterMaxBlocksPerCore};
    TensorAccessorArgs(*view.buffer).append_to(compile_args);
    TensorAccessorArgs(*state.buffer).append_to(compile_args);
    CachedProgram& program = program_for(state, kind.gather ? state.gather : state.scatter, compile_args, kind.name, [&]() {
        return build_program(
            *mesh,
            kind.kernel,
            compile_args,
            qwen_p2p::kScatterCores,
            qwen_p2p::kScatterColumns,
            qwen_p2p::scatter_cb_bytes(static_cast<uint32_t>(view.page_bytes)),
            qwen_p2p::kScatterArgs);
    });
    // The addresses are read from the live buffers on EVERY call and written over the cached program's runtime args (what a program-cache hit's
    // override does): a tensor or staging buffer that moved is followed, never remembered.
    const uint32_t staging_address = static_cast<uint32_t>(state.address);
    const auto runs = qwen_p2p::split_blocks(static_cast<uint32_t>(blocks.size()), qwen_p2p::kScatterCores);
    for (size_t core = 0; core < runs.size(); ++core) {
        set_args(
            program,
            core,
            kind.gather ? qwen_p2p::gather_core_args(view.address, staging_address, bank, byte_offset, row_stride, blocks, 0, runs[core])
                        : qwen_p2p::scatter_core_args(view.address, staging_address, bank, byte_offset, row_stride, blocks, 0, runs[core]));
    }
    enqueue_program(*mesh, cq_id, program, kind.name);
    if (used != nullptr) {
        *used = &program;
    }
}

// qwen_blocks_digest_async. No GIL needed. The digest of each block is the checksum of its SHARD: its rows in the tensors, in the order given (k continues from
// one tensor to the next). One launch, then a non-blocking read of the partials into a host buffer that the handle (and the staging, until the event completes)
// keeps alive, then an event recorded to the host behind the read.
DigestHandle digest_blocks(
    const std::vector<ttnn::Tensor>& tensors, const std::vector<uint32_t>& blocks, uint32_t pages_per_block, uint32_t cq_id, CachedProgram** used = nullptr) {
    using namespace tt::tt_metal;
    const char* op = "qwen_blocks_digest_async";
    if (tensors.empty() || tensors.size() > qwen_p2p::kDigestMaxTensors) {
        refuse(op, "tensors must be 1 to 16");
    }
    std::vector<TensorView> views;
    for (const auto& tensor : tensors) {
        views.push_back(view_tensor(tensor, op));
    }
    const TensorView& first = views.front();
    MeshDevice* mesh_raw = first.mesh;
    const auto state_keeper = find_staging(mesh_raw);
    if (!state_keeper) {
        refuse(op, "no staging buffer is allocated for this mesh (it owns the digest's partials and program cache: qwen_p2p_staging_alloc first)");
    }
    StagingState& state = *state_keeper;
    const auto mesh = state.mesh.lock();
    if (!mesh || !mesh->is_initialized() || mesh.get() != mesh_raw) {
        refuse(op, "the mesh device of the staging buffer is closed");
    }
    check_queue_usable(*mesh, cq_id, op);
    if (const char* problem = qwen_p2p::check_digest_rows(tensors.size(), blocks.size(), pages_per_block, first.page_bytes, first.slices)) {
        refuse(op, problem);
    }
    std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(first.page_bytes), pages_per_block, qwen_p2p::kDigestChunkPages, qwen_p2p::kDigestMaxBlocksPerCore, qwen_p2p::kDigestMaxTensors};
    TensorAccessorArgs(*first.buffer).append_to(compile_args);
    std::vector<uint32_t> addresses;
    for (size_t t = 0; t < views.size(); ++t) {
        const TensorView& view = views[t];
        if (view.mesh != mesh_raw) {
            refuse(op, "tensor " + std::to_string(t) + " is on a different mesh");
        }
        if (view.page_bytes != first.page_bytes || view.slice_bytes != uint64_t(pages_per_block) * first.page_bytes) {
            refuse(op, "tensor " + std::to_string(t) + ": its page size or block size differs from pages_per_block * page_bytes of the first tensor");
        }
        std::vector<uint32_t> own;
        TensorAccessorArgs(*view.buffer).append_to(own);
        if (own != std::vector<uint32_t>(compile_args.begin() + 5, compile_args.end())) {
            refuse(op, "tensor " + std::to_string(t) + " has a different accessor layout from the first tensor");
        }
        size_t bad = 0;
        if (const char* problem = qwen_p2p::check_block_ids(blocks, view.slices, /*require_distinct=*/false, &bad)) {
            refuse(op, "tensor " + std::to_string(t) + ": " + with_index(problem, bad, blocks));
        }
        addresses.push_back(view.address);
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.released()) {
        refuse(op, "the staging buffer was released");
    }
    reap(state);
    const auto& partials = state.partials[state.digest_serial % state.partials.size()];
    TensorAccessorArgs(*partials).append_to(compile_args);
    CachedProgram& program = program_for(state, state.digest, compile_args, op, [&]() {
        return build_program(
            *mesh,
            "qwen_p2p_digest_rows.cpp",
            compile_args,
            qwen_p2p::kDigestMaxCores,
            qwen_p2p::kDigestColumns,
            qwen_p2p::digest_cb_bytes(static_cast<uint32_t>(first.page_bytes)),
            qwen_p2p::kDigestArgs);
    });
    const auto runs = qwen_p2p::split_blocks(static_cast<uint32_t>(blocks.size()), qwen_p2p::kDigestMaxCores);
    for (size_t core = 0; core < runs.size(); ++core) {
        set_args(program, core, qwen_p2p::digest_core_args(static_cast<uint32_t>(partials->address()), addresses, blocks, runs[core]));
    }
    const auto coords = mesh_coordinates(*mesh);
    auto pending = std::make_shared<DigestPending>();
    pending->devices = coords.size();
    pending->blocks = blocks.size();
    pending->mesh = mesh;
    pending->host.assign(coords.size() * blocks.size() * qwen_p2p::kDigestPartialBytes, 0);
    ++state.digest_serial;

    enqueue_program(*mesh, cq_id, program, op);
    if (used != nullptr) {
        *used = &program;
    }
    try {
        auto& queue = mesh->mesh_command_queue(static_cast<uint8_t>(cq_id));
        const uint64_t per_device = uint64_t(blocks.size()) * qwen_p2p::kDigestPartialBytes;
        std::vector<ShardDataTransfer> transfers;
        for (size_t device = 0; device < coords.size(); ++device) {
            transfers.push_back(ShardDataTransfer{coords[device]}.host_data(pending->host.data() + device * per_device).region(BufferRegion(0, per_device)));
        }
        // Non-blocking: the read waits its turn in the queue behind the digest program (and behind the forward before it), and the event behind the read tells
        // the host when the bytes are in `pending->host`. A blocking read here would drain CQ0 behind the in-flight forward on every serviced step.
        queue.enqueue_read_shards(transfers, partials, /*blocking=*/false);
        pending->event = queue.enqueue_record_event_to_host();
    } catch (const std::exception& error) {
        // The digest program is in the queue, its readback is not (or only partly): the host buffer may still be written, so it is kept until the mesh closes.
        static std::vector<std::shared_ptr<DigestPending>> leaked;
        leaked.push_back(pending);
        throw P2PEnqueueError(std::string(op) + ": the partials readback or its event threw (" + error.what() + "): the digest program is in the queue, the sequence is half-enqueued");
    }
    state.inflight.push_back(pending);
    return DigestHandle(pending);
}

// qwen_p2p_record_event: an event recorded to the host behind everything enqueued on the queue so far.
P2PEvent record_event(MeshDevice& mesh, uint32_t cq_id) {
    const char* op = "qwen_p2p_record_event";
    check_queue_usable(mesh, cq_id, op);
    try {
        return P2PEvent(mesh.shared_from_this(), mesh.mesh_command_queue(static_cast<uint8_t>(cq_id)).enqueue_record_event_to_host());
    } catch (const std::exception& error) {
        throw P2PEnqueueError(std::string(op) + ": the event enqueue threw (" + error.what() + ")");
    }
}

// Blocks until every command enqueued on the queue has completed.
void finish_queue(MeshDevice& mesh, uint32_t cq_id, const char* op) {
    try {
        tt::tt_metal::distributed::Finish(mesh.mesh_command_queue(static_cast<uint8_t>(cq_id)));
    } catch (const std::exception& error) {
        throw P2PEnqueueError(std::string(op) + ": the blocking finish threw (" + error.what() + ")");
    }
}

// qwen_p2p_warm: gather -> digest -> scatter-back of the null block (the same bytes go back, so the null block is unchanged), every program run to completion under
// a blocking finish. Start-up only, before the first trace capture and with nothing else on the queue. No GIL needed.
void warm(P2PStaging& handle, const std::vector<ttnn::Tensor>& tensors, uint32_t null_block, uint32_t bank, std::optional<uint64_t> byte_offset_opt, uint32_t cq_id) {
    const char* op = "qwen_p2p_warm";
    const auto state_keeper = handle.state();
    StagingState& state = *state_keeper;
    if (tensors.empty() || tensors.size() > qwen_p2p::kDigestMaxTensors) {
        refuse(op, "tensors must be 1 to 16");
    }
    const TensorView first = view_tensor(tensors.front(), op);
    const auto mesh = state.mesh.lock();
    if (!mesh || !mesh->is_initialized() || first.mesh != mesh.get()) {
        refuse(op, "the tensors are not on the mesh of this staging buffer (or it is closed)");
    }
    check_queue_usable(*mesh, cq_id, op);
    if (const char* problem = qwen_p2p::check_bank(bank, state.num_banks)) {
        refuse(op, problem);
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.released()) {
            refuse(op, "the staging buffer was released");
        }
        if (state.frozen) {
            refuse(op, "the staging buffer is frozen (warm before freeze)");
        }
    }
    if (null_block >= first.slices) {
        refuse(op, "null_block is outside the tensors");
    }
    const uint32_t pages_per_block = static_cast<uint32_t>(first.slice_bytes / first.page_bytes);
    // The warm scratch: just after the first 4 KiB-aligned page of the bank (the canary page the host writes at start-up), one row per tensor.
    const uint64_t start = state.address + (state.bank_offsets.empty() ? 0 : state.bank_offsets[0]);
    const uint64_t lead = (4096 - start % 4096) % 4096;
    const uint64_t offset = byte_offset_opt.value_or(lead + 4096);
    const uint64_t span = uint64_t(tensors.size()) * first.slice_bytes;
    if (offset % qwen_p2p::kDramAlignment != 0 || offset > state.size_bytes || span > state.size_bytes - offset) {
        refuse(op, "the warm scratch (one row per tensor, after the first 4 KiB page of the bank) does not fit the staging range");
    }

    std::vector<CachedProgram*> used;
    const std::vector<uint32_t> null_only = {null_block};
    for (size_t t = 0; t < tensors.size(); ++t) {
        CachedProgram* program = nullptr;
        rows_op(kGatherOp, tensors[t], handle, bank, offset + t * first.slice_bytes, null_only, first.slice_bytes, pages_per_block, cq_id, &program);
        finish_queue(*mesh, cq_id, op);
        used.push_back(program);
    }
    {
        CachedProgram* program = nullptr;
        DigestHandle digest = digest_blocks(tensors, null_only, pages_per_block, cq_id, &program);
        finish_queue(*mesh, cq_id, op);
        used.push_back(program);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!digest.ready()) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw P2PEnqueueError(std::string(op) + ": the digest event did not complete within 30 s after a blocking finish");
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        const auto sums = digest.result();
        if (sums.size() != mesh_coordinates(*mesh).size()) {
            throw P2PEnqueueError(std::string(op) + ": the digest answered for the wrong number of devices");
        }
    }
    for (size_t t = 0; t < tensors.size(); ++t) {
        CachedProgram* program = nullptr;
        rows_op(kScatterOp, tensors[t], handle, bank, offset + t * first.slice_bytes, null_only, first.slice_bytes, pages_per_block, cq_id, &program);
        finish_queue(*mesh, cq_id, op);
        used.push_back(program);
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    for (CachedProgram* program : used) {
        if (program != nullptr) {
            program->completed = true;
        }
    }
    state.warmed = true;
}

// qwen_p2p_freeze: every program of the staging must have run to completion, and the three production families (gather, scatter, digest) must exist. After it a
// call whose compile-time arguments were not warmed is refused (program_for).
void freeze(P2PStaging& handle) {
    const char* op = "qwen_p2p_freeze";
    StagingState& state = *handle.state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.released()) {
        refuse(op, "the staging buffer was released");
    }
    if (state.frozen) {
        return;
    }
    if (state.scatter.empty() || state.gather.empty() || state.digest.empty()) {
        refuse(op, "the gather, scatter and digest programs were not all warmed (qwen_p2p_warm first)");
    }
    for (auto* map : {&state.scatter, &state.gather, &state.digest, &state.checksum}) {
        for (const auto& entry : *map) {
            if (!entry.second->completed) {
                refuse(op, "a program was enqueued but never run to completion under a blocking finish");
            }
        }
    }
    state.frozen = true;
}

// ttnn.qwen_staging_checksum (ORACLE, card tests only): the per-device checksums of `count` consecutive staging byte ranges of `length` bytes (the algorithm of
// qwen_p2p_checksum.cpp, k from 0 at the start of EACH range). Blocking. -> [device][range]. One launch carries up to kChecksumMaxRanges ranges; more are several
// launches, each read back before the next because they share the partials buffer. The partials buffer is allocated at the first call, so an oracle call after
// freeze needs a buffer that was allocated before it.
std::vector<std::vector<uint64_t>> staging_checksum(
    P2PStaging& handle, uint64_t byte_offset, uint64_t length, uint32_t bank, uint32_t cores, uint32_t count) {
    using namespace tt::tt_metal;
    const char* op = "qwen_staging_checksum";
    const auto state_keeper = handle.state();
    StagingState& state = *state_keeper;
    if (const char* problem = qwen_p2p::check_checksum(byte_offset, length, state.size_bytes, cores, count)) {
        refuse(op, problem);
    }
    if (const char* problem = qwen_p2p::check_bank(bank, state.num_banks)) {
        refuse(op, problem);
    }
    const auto mesh_keeper = state.mesh.lock();
    if (!mesh_keeper || !mesh_keeper->is_initialized()) {
        refuse(op, "the mesh device of this staging buffer is closed");
    }
    MeshDevice& mesh = *mesh_keeper;
    check_queue_usable(mesh, 0, op);

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.released()) {
        refuse(op, "the staging buffer was released");
    }
    if (!state.checksum_partials) {
        if (state.frozen) {
            refuse(op, "the oracle's partials buffer was not allocated before freeze");
        }
        distributed::DeviceLocalBufferConfig partial_local;
        partial_local.page_size = qwen_p2p::kChecksumPartialBytes;
        partial_local.buffer_type = BufferType::DRAM;
        state.checksum_partials = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{
                .size = uint64_t(qwen_p2p::kChecksumMaxRanges) * qwen_p2p::kChecksumMaxCores * qwen_p2p::kChecksumPartialBytes},
            partial_local,
            &mesh);
    }
    const std::shared_ptr<distributed::MeshBuffer> source = state.buffer;
    const std::shared_ptr<distributed::MeshBuffer> partials = state.checksum_partials;

    std::vector<uint32_t> compile_args = {qwen_p2p::kChecksumChunkBytes, qwen_p2p::kChecksumPartialBytes};
    TensorAccessorArgs(*source).append_to(compile_args);
    TensorAccessorArgs(*partials).append_to(compile_args);
    compile_args.push_back(cores);  // not read by the kernel: it keys the cached program (the core set is part of the program)

    CachedProgram& program = program_for(state, state.checksum, compile_args, op, [&]() {
        return build_program(
            mesh, "qwen_p2p_checksum.cpp", compile_args, cores, qwen_p2p::kChecksumColumns, qwen_p2p::checksum_cb_bytes(), qwen_p2p::kChecksumArgs);
    });
    const auto units = qwen_p2p::split_units(length / qwen_p2p::kChecksumUnit, cores);
    auto& queue = mesh.mesh_command_queue(0);
    const std::vector<MeshCoordinate> coords = mesh_coordinates(mesh);
    std::vector<std::vector<uint64_t>> sums(coords.size(), std::vector<uint64_t>(count, 0));
    for (const auto& [launch_first, launch_count] : qwen_p2p::checksum_launches(count)) {
        const uint64_t launch_offset = byte_offset + uint64_t(launch_first) * length;
        for (uint32_t core = 0; core < cores; ++core) {
            set_args(
                program,
                core,
                qwen_p2p::checksum_core_args(
                    static_cast<uint32_t>(state.address),
                    static_cast<uint32_t>(partials->address()),
                    bank,
                    launch_offset,
                    units[core],
                    core,
                    launch_count,
                    length,
                    cores));
        }
        enqueue_program(mesh, 0, program, op);

        // The partial of range r, core c is partials page r * cores + c.
        const uint64_t per_device = uint64_t(launch_count) * cores * qwen_p2p::kChecksumPartialBytes;
        std::vector<uint8_t> host(coords.size() * per_device);
        std::vector<ShardDataTransfer> transfers;
        for (size_t device = 0; device < coords.size(); ++device) {
            transfers.push_back(ShardDataTransfer{coords[device]}.host_data(host.data() + device * per_device).region(BufferRegion(0, per_device)));
        }
        queue.enqueue_read_shards(transfers, partials, /*blocking=*/true);
        program.completed = true;
        for (size_t device = 0; device < coords.size(); ++device) {
            for (uint32_t range = 0; range < launch_count; ++range) {
                for (uint32_t core = 0; core < cores; ++core) {
                    uint64_t partial = 0;
                    std::copy_n(
                        host.data() + device * per_device + (uint64_t(range) * cores + core) * qwen_p2p::kChecksumPartialBytes,
                        8,
                        reinterpret_cast<uint8_t*>(&partial));
                    sums[device][launch_first + range] += partial;  // mod 2^64
                }
            }
        }
    }
    return sums;
}

// Diagnostics (card tests only): bytes of the staging range <-> host memory through the command queue. The range is page aligned (byte_offset and the length are
// multiples of the staging page size) because a buffer region is whole pages; bank byte x lives in page (x div P) * banks + bank.
void staging_transfer(P2PStaging& handle, uint64_t byte_offset, uint8_t* host, uint64_t host_bytes, uint32_t bank, bool write) {
    using namespace tt::tt_metal;
    const char* op = "qwen_p2p_staging_write/read";
    const auto state_keeper = handle.state();
    StagingState& state = *state_keeper;
    const uint64_t page = state.page_bytes;
    const auto mesh_keeper = state.mesh.lock();
    if (!mesh_keeper || !mesh_keeper->is_initialized()) {
        refuse(op, "the mesh device of this staging buffer is closed");
    }
    MeshDevice& mesh = *mesh_keeper;
    check_queue_usable(mesh, 0, op);
    const std::vector<MeshCoordinate> coords = mesh_coordinates(mesh);
    if (bank >= state.num_banks || host_bytes % coords.size() != 0) {
        refuse(op, "bad bank or a buffer that is not [devices, n]");
    }
    const uint64_t per_device = host_bytes / coords.size();
    if (per_device == 0 || byte_offset % page != 0 || per_device % page != 0 || byte_offset > state.size_bytes || per_device > state.size_bytes - byte_offset) {
        refuse(op, "byte_offset and the per-device length must be multiples of the staging page size and inside the range");
    }
    std::vector<ShardDataTransfer> transfers;
    transfers.reserve(coords.size() * (per_device / page));
    for (size_t device = 0; device < coords.size(); ++device) {
        for (uint64_t at = 0; at < per_device; at += page) {
            const uint64_t page_index = (byte_offset + at) / page;
            transfers.push_back(ShardDataTransfer{coords[device]}
                                    .host_data(host + device * per_device + at)
                                    .region(BufferRegion((page_index * state.num_banks + bank) * page, page)));
        }
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.released()) {
        refuse(op, "the staging buffer was released");
    }
    auto& queue = mesh.mesh_command_queue(0);
    if (write) {
        queue.enqueue_write_shards(state.buffer, transfers, /*blocking=*/true);
    } else {
        queue.enqueue_read_shards(transfers, state.buffer, /*blocking=*/true);
    }
}

// The NOC coordinates of the endpoints of a DRAM bank, as the device itself translates them.
struct BankPoint {
    bool ok = false;
    uint32_t x = 0, y = 0;  // endpoint 0
    std::vector<std::pair<uint32_t, uint32_t>> endpoints;
    std::string error;
};

BankPoint bank_point(tt::tt_metal::IDevice* device, uint32_t bank) {
    // A Blackhole DRAM channel (a "DRAM view") has three NOC endpoints. tt-metal's logical DRAM coordinate is (view, endpoint index): index 0 is the view's
    // NOC0 worker endpoint, index 1 its NOC1 worker endpoint, then the remaining endpoint (metal_soc_descriptor.cpp). The result is the TRANSLATED coordinate
    // the TLB registers take, from the same tables the device uses for its own bank addresses. The default (noc_x, noc_y) is endpoint 0: the one the kernels
    // and tt-metal's own host reads of DRAM (Cluster::read_dram_vec) go through. Any endpoint reaches the same channel; the host's window self-test settles it.
    BankPoint point;
    try {
        for (uint32_t index = 0; index < 3; ++index) {
            const CoreCoord endpoint = device->virtual_core_from_logical_core(CoreCoord(bank, index), tt::CoreType::DRAM);
            point.endpoints.emplace_back(static_cast<uint32_t>(endpoint.x), static_cast<uint32_t>(endpoint.y));
        }
        point.x = point.endpoints[0].first;
        point.y = point.endpoints[0].second;
        point.ok = true;
    } catch (const std::exception& error) {
        point.error = error.what();
    }
    return point;
}

// qwen_p2p_staging_info: what the host needs to aim its window and map the cards. One dict (the banks and the staging range are the same on every device; the host
// checks that, and so does this: if the devices disagree about a bank's NOC coordinates the call refuses rather than let a window be aimed at the wrong place).
nb::dict staging_info(P2PStaging& handle) {
    using namespace tt::tt_metal;
    const char* op = "qwen_p2p_staging_info";
    const auto state_keeper = handle.state();
    StagingState& state = *state_keeper;
    const auto mesh_keeper = state.mesh.lock();
    if (!mesh_keeper || !mesh_keeper->is_initialized() || state.released()) {
        refuse(op, "the mesh device of this staging buffer is closed, or the staging buffer was released");
    }
    MeshDevice& mesh = *mesh_keeper;
    const std::vector<MeshCoordinate> coords = mesh_coordinates(mesh);

    // The first byte address of the range in bank k, as a TLB window aimed at the bank sees it: the allocator's address plus the bank's offset.
    std::vector<uint64_t> bank_address;
    for (uint32_t bank = 0; bank < state.num_banks; ++bank) {
        bank_address.push_back(static_cast<uint64_t>(static_cast<int64_t>(state.address) + state.bank_offsets[bank]));
    }
    // An O_DIRECT object wants 4096-byte aligned bank addresses: the first one is `lead` bytes into the range, and `aligned_4k_size` bytes of it are usable from there
    // in whole 4096-byte units. Every bank must give the same lead (a staging slot is laid out once for all banks).
    const uint64_t lead = (4096 - bank_address[0] % 4096) % 4096;
    for (uint32_t bank = 1; bank < state.num_banks; ++bank) {
        if ((4096 - bank_address[bank] % 4096) % 4096 != lead) {
            refuse(op, "the banks start at addresses with different 4 KiB alignment");
        }
    }

    nb::list devices;
    nb::list banks;
    std::vector<BankPoint> first_points;
    for (size_t index = 0; index < coords.size(); ++index) {
        const auto found = mesh.get_view().get_devices(MeshCoordinateRange(coords[index]));
        if (found.empty()) {
            refuse(op, "a device of the mesh is not local");
        }
        IDevice* device = found.front();
        nb::dict entry;
        nb::list coordinate;
        for (uint32_t value : coords[index].coords()) {
            coordinate.append(value);
        }
        entry["mesh_coord"] = coordinate;
        entry["mesh_index"] = index;
        entry["device_id"] = static_cast<int>(device->id());
        // GetPCIeDeviceID is the logical chip id of the MMIO chip that controls this one. It equals the tt-kmd device index N (/dev/tenstorrent/N) when every
        // card is visible to the process, in the order the kernel numbered them (the serving container's case) and NOT when a subset is: pci_bus_id below is
        // derived from it, so the host must compare it with the card it actually opened (the window self-test does) and not trust it blindly.
        uint32_t pcie = 0;
        bool have_pcie = false;
        try {
            pcie = static_cast<uint32_t>(GetPCIeDeviceID(device->id()));
            have_pcie = true;
        } catch (const std::exception&) {
        }
        entry["mmio_chip_id"] = have_pcie ? nb::object(nb::int_(pcie)) : nb::object(nb::none());
        entry["pcie_device"] = have_pcie ? nb::object(nb::int_(pcie)) : nb::object(nb::none());
        const std::string bus = have_pcie ? pci_bus_id_of(pcie) : std::string();
        entry["pci_bus_id"] = bus.empty() ? nb::object(nb::none()) : nb::object(nb::str(bus.c_str()));
        devices.append(entry);

        // The banks' NOC coordinates: from the first device, and every other device must agree.
        for (uint32_t bank = 0; bank < state.num_banks; ++bank) {
            const BankPoint point = bank_point(device, bank);
            if (index == 0) {
                first_points.push_back(point);
            } else if (point.ok != first_points[bank].ok || point.x != first_points[bank].x || point.y != first_points[bank].y) {
                refuse(op, "bank " + std::to_string(bank) + " has different NOC coordinates on device " + std::to_string(index));
            }
        }
    }
    for (uint32_t bank = 0; bank < state.num_banks; ++bank) {
        const BankPoint& point = first_points[bank];
        nb::dict description;
        description["bank"] = bank;
        if (point.ok) {
            description["noc_x"] = point.x;
            description["noc_y"] = point.y;
            description["endpoint_index"] = 0;
        } else {
            description["noc_x"] = nb::none();
            description["noc_y"] = nb::none();
            description["endpoint_error"] = point.error.c_str();
        }
        nb::list endpoints;
        for (const auto& endpoint : point.endpoints) {
            nb::list pair;
            pair.append(endpoint.first);
            pair.append(endpoint.second);
            endpoints.append(pair);
        }
        description["noc_endpoints"] = endpoints;
        description["bank_offset"] = state.bank_offsets[bank];
        description["address"] = bank_address[bank];
        banks.append(description);
    }
    nb::dict info;
    info["address"] = bank_address[0];
    info["size"] = state.size_bytes;
    info["page_bytes"] = state.page_bytes;
    info["aligned_4k_offset"] = lead;
    info["aligned_4k_address"] = bank_address[0] + lead;
    info["aligned_4k_size"] = state.size_bytes > lead ? (state.size_bytes - lead) / 4096 * 4096 : uint64_t(0);
    info["banks"] = banks;
    info["devices"] = devices;
    // What the allocation took: the same range in every bank (lockstep), and what the allocator had left around it.
    info["num_banks"] = state.num_banks;
    info["bytes_reserved_per_bank"] = state.size_bytes;
    info["bytes_reserved"] = state.size_bytes * state.num_banks;
    info["allocator_address"] = state.address;
    info["dram_free_per_bank_before"] = static_cast<uint64_t>(state.dram_before.total_free_bytes);
    info["dram_free_per_bank_after"] = static_cast<uint64_t>(state.dram_after.total_free_bytes);
    info["dram_largest_free_block_after"] = static_cast<uint64_t>(state.dram_after.largest_free_block_bytes);
    info["frozen"] = state.frozen;
    info["warmed"] = state.warmed;
    return info;
}

}  // namespace qwen_p2p_host

NB_MODULE(qwen_kv_read, m) {
    m.doc() = "ttnn.qwen_read_blocks: region read of a paged KV cache (docs/prefix-audit-cost.md); the P2P KV offload's device ops (version 4)";
    // The module must load after ttnn: it names ttnn.Tensor, whose nanobind type ttnn registers.
    nb::module_ ttnn_module = nb::module_::import_("ttnn");
    nb::exception<qwen_p2p_host::P2PRefused>(m, "QwenP2PRefused", PyExc_ValueError);
    nb::exception<qwen_p2p_host::P2PEnqueueError>(m, "QwenP2PEnqueueError", PyExc_RuntimeError);
    m.def(
        "qwen_read_blocks",
        [](const ttnn::Tensor& device_tensor, ttnn::Tensor& host_tensor, const std::vector<uint32_t>& blocks) {
            auto* mesh = device_tensor.device();
            if (mesh == nullptr || !device_tensor.is_allocated()) {
                throw nb::value_error("qwen_read_blocks: device_tensor must be a tensor allocated on a mesh device");
            }
            if (host_tensor.storage_type() != ttnn::StorageType::HOST) {
                throw nb::value_error("qwen_read_blocks: host_tensor must be a host tensor (ttnn.allocate_tensor_on_host)");
            }
            auto& cq = mesh->mesh_command_queue(0);
            qwen_kv::read_dim0_slices(cq, device_tensor.mesh_tensor(), host_tensor.host_storage().host_tensor(), blocks);
        },
        nb::arg("device_tensor"),
        nb::arg("host_tensor"),
        nb::arg("blocks"),
        "Reads the named dimension-0 slices (paged KV cache blocks) of a device tensor into a host tensor of just those "
        "slices, on every device of the mesh. Only those pages move; no program is compiled.");
    // Version 2: the raw packed-byte transfers of the host KV tier.
    m.def(
        "qwen_block_bytes",
        [](const ttnn::Tensor& device_tensor) -> uint64_t {
            if (device_tensor.device() == nullptr || !device_tensor.is_allocated()) {
                throw nb::value_error("qwen_block_bytes: device_tensor must be a tensor allocated on a mesh device");
            }
            return qwen_kv::slice_bytes_of(device_tensor.mesh_tensor());
        },
        nb::arg("device_tensor"),
        "Bytes of one dimension-0 slice (one paged KV cache block) of a device tensor, on one device.");
    m.def(
        "qwen_read_blocks_raw",
        [](const ttnn::Tensor& device_tensor,
           nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> out,
           const std::vector<uint32_t>& blocks) {
            auto* mesh = device_tensor.device();
            if (mesh == nullptr || !device_tensor.is_allocated()) {
                throw nb::value_error("qwen_read_blocks_raw: device_tensor must be a tensor allocated on a mesh device");
            }
            auto* host = reinterpret_cast<std::byte*>(out.data());
            const uint64_t host_bytes = out.nbytes();
            nb::gil_scoped_release release;
            qwen_kv::transfer_raw_dim0_slices(
                mesh->mesh_command_queue(0), device_tensor.mesh_tensor(), host, host_bytes, blocks, /*write=*/false);
        },
        nb::arg("device_tensor"),
        nb::arg("out"),
        nb::arg("blocks"),
        "Reads the named dimension-0 slices (paged KV cache blocks) of a device tensor, as raw packed bytes, into a uint8 buffer "
        "[devices, len(blocks), qwen_block_bytes]. No unpack; no program is compiled; nothing is allocated on the device.");
    m.def(
        "qwen_write_blocks_raw",
        [](const ttnn::Tensor& device_tensor,
           nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu> data,
           const std::vector<uint32_t>& blocks) {
            auto* mesh = device_tensor.device();
            if (mesh == nullptr || !device_tensor.is_allocated()) {
                throw nb::value_error("qwen_write_blocks_raw: device_tensor must be a tensor allocated on a mesh device");
            }
            auto* host = const_cast<std::byte*>(reinterpret_cast<const std::byte*>(data.data()));
            const uint64_t host_bytes = data.nbytes();
            nb::gil_scoped_release release;
            qwen_kv::transfer_raw_dim0_slices(
                mesh->mesh_command_queue(0), device_tensor.mesh_tensor(), host, host_bytes, blocks, /*write=*/true);
        },
        nb::arg("device_tensor"),
        nb::arg("data"),
        nb::arg("blocks"),
        "Writes raw packed bytes [devices, len(blocks), qwen_block_bytes] into the named dimension-0 slices (paged KV cache blocks) of a "
        "device tensor, on every device, and returns when the device has them. No pack; no program is compiled.");


    // Versions 3 and 4: the P2P KV offload's device side.
    nb::class_<P2PStaging>(m, "P2PStaging")
        .def("release", &P2PStaging::release, "Frees the staging buffer and its cached programs. Call it before the mesh is closed.")
        .def_prop_ro("size_bytes", &P2PStaging::size_bytes)
        .def_prop_ro("page_bytes", &P2PStaging::page_bytes)
        .def_prop_ro("num_banks", &P2PStaging::num_banks)
        .def_prop_ro("address", &P2PStaging::address)
        .def_prop_ro("released", &P2PStaging::released)
        .def_prop_ro("frozen", &P2PStaging::frozen)
        .def_prop_ro("warmed", &P2PStaging::warmed);
    nb::class_<DigestHandle>(m, "DigestHandle")
        .def("ready", &DigestHandle::ready, "True once the digest's partials are in the host buffer (distributed::EventQuery; never waits).")
        .def(
            "result",
            &DigestHandle::result,
            "The digest of each block on each device, [device][block], as Python ints. Only after ready() is true; it never waits.")
        .def_prop_ro("blocks", &DigestHandle::blocks)
        .def_prop_ro("devices", &DigestHandle::devices);
    nb::class_<P2PEvent>(m, "P2PEvent")
        .def("query", &P2PEvent::query, "True once the command queue has completed everything enqueued before this event (distributed::EventQuery; never waits).");
    m.def(
        "qwen_p2p_staging_alloc",
        [](MeshDevice& mesh, uint64_t bytes_per_bank, bool bottom_up, uint64_t page_bytes) {
            std::unique_ptr<P2PStaging> staging;
            {
                nb::gil_scoped_release release;
                staging = std::make_unique<P2PStaging>(mesh, bytes_per_bank, page_bytes, bottom_up);
            }
            return staging.release();
        },
        nb::arg("mesh"),
        nb::arg("bytes_per_bank"),
        nb::arg("bottom_up") = false,
        nb::arg("page_bytes") = uint64_t(64) << 10,
        nb::rv_policy::take_ownership,
        "Allocates the P2P staging buffer: on every device of the mesh ONE contiguous range of bytes_per_bank in EACH of the DRAM banks that the allocator never hands to "
        "anything else while the handle lives (the DRAM allocator reserves the same range in all banks: the cost is bytes_per_bank x banks per card). Also allocates "
        "the digest's partials, so call it before the first trace capture. Keep the handle for the engine's life; release() it before closing the mesh. Refused during "
        "trace capture and for a mesh that already has one.");
    m.def(
        "qwen_p2p_staging_info",
        [](P2PStaging& staging) { return qwen_p2p_host::staging_info(staging); },
        nb::arg("staging"),
        "A dict: address (byte address of the range in bank 0 as seen through a TLB window aimed at it), size (bytes per bank), page_bytes, aligned_4k_offset / "
        "aligned_4k_address / aligned_4k_size (the first 4096-aligned byte of the range and the whole 4096-byte units from there), banks [{bank, noc_x, noc_y, "
        "noc_endpoints, bank_offset, address}] (translated NOC coordinates of an endpoint of each bank), devices [{mesh_coord, device_id, pcie_device, pci_bus_id "
        "(None when sysfs cannot be read)}] in mesh (row-major) order, and the allocation's cost.");
    m.def(
        "qwen_blocks_to_staging",
        [](const ttnn::Tensor& tensor,
           P2PStaging& staging,
           uint32_t bank,
           uint64_t byte_offset,
           const std::vector<uint32_t>& blocks,
           uint64_t row_stride,
           uint32_t pages_per_block,
           uint32_t cq_id) {
            nb::gil_scoped_release release;
            qwen_p2p_host::rows_op(qwen_p2p_host::kGatherOp, tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id);
        },
        nb::arg("tensor"),
        nb::arg("staging"),
        nb::arg("bank"),
        nb::arg("byte_offset"),
        nb::arg("blocks"),
        nb::arg("row_stride"),
        nb::arg("pages_per_block"),
        nb::arg("cq_id") = 0,
        "GATHER: staging bank `bank` [byte_offset + i * row_stride + j * page_bytes] = page (blocks[i] * pages_per_block + j) of the tensor, on every device. One launch "
        "(at most 512 blocks); enqueued on cq_id, it does not wait. row_stride is a multiple of 64 and at least the block size; the whole range must lie inside the "
        "bank's range. Full tile pages only; repeated block ids are fine (reads). Refused during trace capture or with bad arguments; QwenP2PEnqueueError if the "
        "enqueue itself threw.");
    m.def(
        "qwen_write_blocks_from_staging",
        [](const ttnn::Tensor& tensor,
           P2PStaging& staging,
           uint32_t bank,
           uint64_t byte_offset,
           const std::vector<uint32_t>& blocks,
           uint64_t row_stride,
           uint32_t pages_per_block,
           uint32_t cq_id) {
            nb::gil_scoped_release release;
            qwen_p2p_host::rows_op(qwen_p2p_host::kScatterOp, tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id);
        },
        nb::arg("tensor"),
        nb::arg("staging"),
        nb::arg("bank"),
        nb::arg("byte_offset"),
        nb::arg("blocks"),
        nb::arg("row_stride"),
        nb::arg("pages_per_block"),
        nb::arg("cq_id") = 0,
        "SCATTER: device d's page (blocks[i] * pages_per_block + j) of the tensor gets staging bank `bank` [byte_offset + i * row_stride + j * page_bytes]. One launch "
        "(at most 512 blocks); enqueued on cq_id, it does not wait. The block ids must be distinct. Full tile pages only. Refused during trace capture or with bad "
        "arguments; QwenP2PEnqueueError if the enqueue itself threw.");
    m.def(
        "qwen_blocks_digest_async",
        [](const std::vector<ttnn::Tensor>& tensors, const std::vector<uint32_t>& blocks, uint32_t pages_per_block, uint32_t cq_id) {
            nb::gil_scoped_release release;
            return qwen_p2p_host::digest_blocks(tensors, blocks, pages_per_block, cq_id);
        },
        nb::arg("tensors"),
        nb::arg("blocks"),
        nb::arg("pages_per_block"),
        nb::arg("cq_id") = 0,
        "Digest of KV ROWS: for each block, the checksum S = sum_k ((k * 2654435761 + 1) mod 2^32) * x_k mod 2^64 over the little-endian uint32 words x_k of the block's "
        "shard, its rows in the given tensors (at most 16) concatenated in the order given (k counts from the shard start and continues from one tensor to the next). "
        "At most 128 blocks. Enqueues one program, a NON-BLOCKING read of the partials and an event; returns a DigestHandle: ready() / result() -> [device][block]. "
        "Needs the mesh's staging (it owns the partials and the program cache).");
    m.def(
        "qwen_p2p_record_event",
        [](MeshDevice& mesh, uint32_t cq_id) {
            nb::gil_scoped_release release;
            return qwen_p2p_host::record_event(mesh, cq_id);
        },
        nb::arg("mesh"),
        nb::arg("cq_id") = 0,
        "Records an event to the host behind everything enqueued on the command queue so far: returns a P2PEvent whose query() is distributed::EventQuery.");
    m.def(
        "qwen_p2p_warm",
        [](P2PStaging& staging,
           const std::vector<ttnn::Tensor>& tensors,
           uint32_t null_block,
           uint32_t bank,
           std::optional<uint64_t> byte_offset,
           uint32_t cq_id) {
            nb::gil_scoped_release release;
            qwen_p2p_host::warm(staging, tensors, null_block, bank, byte_offset, cq_id);
        },
        nb::arg("staging"),
        nb::arg("tensors"),
        nb::arg("null_block") = 0,
        nb::arg("bank") = 0,
        nb::arg("byte_offset") = nb::none(),
        nb::arg("cq_id") = 0,
        "Runs every program once, each to completion under a blocking finish: gather the null block's row of each tensor into the bank's scratch (default: just after the "
        "first 4 KiB-aligned page of bank 0), digest it, scatter it back (the same bytes). Start-up only: before the first trace capture, with nothing else on the queue.");
    m.def(
        "qwen_p2p_freeze",
        [](P2PStaging& staging) { qwen_p2p_host::freeze(staging); },
        nb::arg("staging"),
        "After it a call whose program key was not warmed raises QwenP2PRefused. Requires the gather, scatter and digest programs to exist and every program to have run "
        "to completion.");
    // The ORACLES and diagnostics of versions 2 and 3 (card tests only; blocking).
    m.def(
        "qwen_staging_checksum",
        [](P2PStaging& staging, uint64_t byte_offset, uint64_t length, std::optional<uint32_t> bank, uint32_t cores, std::optional<uint32_t> count) -> nb::object {
            std::vector<std::vector<uint64_t>> sums;
            {
                nb::gil_scoped_release release;
                sums = qwen_p2p_host::staging_checksum(staging, byte_offset, length, bank.value_or(0), cores, count.value_or(1));
            }
            if (count.has_value()) {
                return nb::cast(sums);  // [device][range]
            }
            std::vector<uint64_t> flat;
            for (const auto& device : sums) {
                flat.push_back(device[0]);
            }
            return nb::cast(flat);  // [device]
        },
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("length"),
        nb::arg("bank") = nb::none(),
        nb::arg("cores") = qwen_p2p::kChecksumDefaultCores,
        nb::arg("count") = nb::none(),
        "ORACLE (card tests only). Per device, S = sum_k (((k * 2654435761 + 1) mod 2^32) * x_k) mod 2^64 over the little-endian uint32 words x_k of "
        "staging[byte_offset : + length] in bank `bank` (default 0) (k from 0 at byte_offset; length a multiple of 4096). Blocking: returns a list with one int per "
        "device in mesh order. count=N checksums N consecutive ranges of `length` bytes (the range r starts at byte_offset + r * length) in one call, k counted from each "
        "range's own start, and returns a list with a list of N ints per device.");
    m.def(
        "qwen_p2p_staging_write",
        [](P2PStaging& staging, uint64_t byte_offset, nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu> data, std::optional<uint32_t> bank) {
            auto* host = const_cast<uint8_t*>(data.data());
            const uint64_t bytes = data.nbytes();
            nb::gil_scoped_release release;
            qwen_p2p_host::staging_transfer(staging, byte_offset, host, bytes, bank.value_or(0), /*write=*/true);
        },
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("data"),
        nb::arg("bank") = nb::none(),
        "Diagnostic: writes uint8 data [devices, n] into the staging range of bank `bank` (default 0) through the command queue (what a drive's DMA would write). "
        "byte_offset and n are multiples of the staging page size. Blocking.");
    m.def(
        "qwen_p2p_staging_read",
        [](P2PStaging& staging, uint64_t byte_offset, nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> out, std::optional<uint32_t> bank) {
            auto* host = out.data();
            const uint64_t bytes = out.nbytes();
            nb::gil_scoped_release release;
            qwen_p2p_host::staging_transfer(staging, byte_offset, host, bytes, bank.value_or(0), /*write=*/false);
        },
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("out"),
        nb::arg("bank") = nb::none(),
        "Diagnostic: reads the staging range of bank `bank` (default 0) into uint8 out [devices, n]. byte_offset and n are multiples of the staging page size. Blocking.");
    m.attr("QWEN_KV_READ_VERSION") = "4";
    for (const char* name : {"qwen_read_blocks",
                             "qwen_block_bytes",
                             "qwen_read_blocks_raw",
                             "qwen_write_blocks_raw",
                             "qwen_p2p_staging_alloc",
                             "qwen_p2p_staging_info",
                             "qwen_blocks_to_staging",
                             "qwen_write_blocks_from_staging",
                             "qwen_blocks_digest_async",
                             "qwen_p2p_record_event",
                             "qwen_p2p_warm",
                             "qwen_p2p_freeze",
                             "qwen_staging_checksum",
                             "qwen_p2p_staging_write",
                             "qwen_p2p_staging_read",
                             "QwenP2PRefused",
                             "QwenP2PEnqueueError"}) {
        if (!nb::hasattr(ttnn_module, name)) {
            ttnn_module.attr(name) = m.attr(name);
        }
    }
}
