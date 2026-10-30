// qwen_kv_read: a standalone nanobind extension adding ttnn.qwen_read_blocks (the prefix audit's region read) WITHOUT
// replacing any of the image's pinned binaries (_ttnn.so, _ttnncpp.so, libtt_metal.so keep their hashes and their QWEN_
// strings). It is optimisation/sim/kv-region-read.patch re-cut as a module: the same function body, compiled against the
// served tt-metal tree (9f9cd4fd), linking the image's own libraries. Import it after ttnn; it sets ttnn.qwen_read_blocks.
//
// Version 2 adds the host KV tier's device side (docs/prefix-store-hygiene.md, QWEN_PREFIX_HOST_TIER_GIB): the same region transfers of whole
// dimension-0 slices (paged KV cache blocks), but of the RAW PACKED BYTES, to and from a plain numpy buffer - no ttnn host tensor, no unpack,
// no repack, so what is written back is bit for bit what was read:
//   ttnn.qwen_block_bytes(device_tensor) -> bytes of one dim-0 slice on one device
//   ttnn.qwen_read_blocks_raw(device_tensor, out, blocks)   out: uint8 C-contiguous [devices, len(blocks), slice bytes]
//   ttnn.qwen_write_blocks_raw(device_tensor, data, blocks) data: the same shape; device d's slice blocks[i] gets data[d, i]
// Devices are in the mesh's row-major coordinate order. Runs of consecutive block ids are one region transfer each; both calls return when the
// transfers have completed (blocking), release the GIL while they wait, run no program and allocate nothing on the device. NOT YET RUN ON A
// CARD (kv_region_read_card.py run_raw is its check); version 1's functions are unchanged.
//
// Version 3 adds the DEVICE side of the peer-to-peer KV restore (docs/tp4-p2p-restore.md): a staging buffer in card DRAM that a drive can DMA
// into (one contiguous range in one DRAM bank, never handed to anything else) and two device programs, unlike everything above, which run
// no program:
//   ttnn.qwen_p2p_staging_alloc(mesh_device, size_bytes=128 MiB, bank=0, page_bytes=64 KiB, bottom_up=False) -> QwenP2PStaging
//   ttnn.qwen_p2p_staging_info(staging) -> one dict per device (dram_bank, noc_x, noc_y, address, size, ...)
//   ttnn.qwen_write_blocks_from_staging(device_tensor, staging, byte_offset, blocks, bank=None, row_stride=None)   staging bytes -> KV blocks, enqueued
//       (row i of a device comes from staging[byte_offset + i * row_stride : + slice bytes]; row_stride=None is the slice bytes: contiguous rows)
//   ttnn.qwen_staging_checksum(staging, byte_offset, length, bank=None, cores=64, count=None) -> [uint64 per device]   (blocking)
//       (count=N: N consecutive ranges of `length` bytes, one checksum each, k counted from each range's own start -> [[uint64] * N per device])
//   ttnn.qwen_p2p_staging_write / qwen_p2p_staging_read (diagnostics: host bytes into / out of the staging range through the command queue)
// The two kernels (qwen_p2p_scatter.cpp, qwen_p2p_checksum.cpp) are read from the directory of this module and JIT-compiled at the first
// dispatch; qwen_p2p_plan.hpp is the pure arithmetic (compiled and checked by the CPU test). NOT YET RUN ON A CARD: p2p_staging_card.py.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
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
// Version 3: the device side of the peer-to-peer KV restore.
//
// HOW THE STAGING IS PLACED (docs/tp4-p2p-restore.md, section 1). tt-metal's DRAM allocator (BankManager, one free list) places every DRAM
// buffer in LOCKSTEP over all banks: an allocation reserves the SAME address range [a, a + size_per_bank) in every one of the 8 banks of every
// device of the mesh, whatever it holds, so a single-bank reservation does not exist. An interleaved buffer of 8 x S bytes (page size P, S a
// multiple of P) puts its pages p = 0, 1, 2, ... on bank p mod 8 at a + (p div 8) * P, so the pages of ONE bank are contiguous: bytes [a, a + S) of
// bank b are one contiguous range, the allocator never hands it to anything else while the buffer lives, and a TLB window aimed at bank b
// reaches it. The cost is the buffer: 8 x S per card (the other 7 banks' ranges are kept, usable as further staging through `bank=`).
class P2PStaging {
public:
    P2PStaging(
        tt::tt_metal::distributed::MeshDevice& mesh,
        uint64_t size_bytes,
        uint32_t bank,
        uint64_t page_bytes,
        bool bottom_up) :
        mesh_(mesh.shared_from_this()), size_bytes_(size_bytes), page_bytes_(page_bytes), bank_(bank) {
        using namespace tt::tt_metal;
        auto* allocator = mesh.allocator().get();
        num_banks_ = allocator->get_num_banks(BufferType::DRAM);
        if (const char* problem =
                qwen_p2p::check_staging(size_bytes, page_bytes, bank, num_banks_, allocator->get_bank_size(BufferType::DRAM))) {
            throw nb::value_error(problem);
        }
        dram_free_before_ = allocator->get_statistics(BufferType::DRAM);
        distributed::DeviceLocalBufferConfig local;
        local.page_size = page_bytes;
        local.buffer_type = BufferType::DRAM;
        local.bottom_up = bottom_up;
        buffer_ = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size_bytes * num_banks_}, local, &mesh);
        address_ = buffer_->address();
        // The checksum's partial sums: one whole 64-byte page per core and range of a launch (up to kChecksumMaxRanges ranges: 2.5 MiB, spread over the banks).
        distributed::DeviceLocalBufferConfig partial_local;
        partial_local.page_size = qwen_p2p::kChecksumPartialBytes;
        partial_local.buffer_type = BufferType::DRAM;
        partials_ = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{
                .size = uint64_t(qwen_p2p::kChecksumMaxRanges) * qwen_p2p::kChecksumMaxCores * qwen_p2p::kChecksumPartialBytes},
            partial_local,
            &mesh);
        dram_free_after_ = allocator->get_statistics(BufferType::DRAM);
        bank_offset_ = allocator->get_bank_offset(BufferType::DRAM, bank_);
        TT_FATAL(address_ % qwen_p2p::kDramAlignment == 0, "The staging buffer address {} is not 64-byte aligned", address_);
        // Sanity of the MeshBuffer (it creates every device's Buffer at the one address, so this cannot fail by itself): same address, page size asked for.
        // What it does NOT prove is that no other allocator hands out the range: the guarantee is that THIS MeshDevice's allocator never does (a submesh or
        // a second MeshDevice over the same chips has its own free list, and so has a single-device allocation), so the engine must allocate DRAM only
        // through the MeshDevice it passes here.
        distributed::MeshCoordinateRange all(mesh.shape());
        for (const auto& coord : all) {
            const Buffer* device_buffer = buffer_->get_device_buffer(coord);
            TT_FATAL(
                device_buffer->address() == address_ && device_buffer->page_size() == page_bytes &&
                    device_buffer->aligned_page_size() == page_bytes &&
                    device_buffer->buffer_layout() == TensorMemoryLayout::INTERLEAVED,
                "The staging buffer is not an interleaved buffer at one address with the requested page size on {}",
                coord);
        }
    }

    ~P2PStaging() { release(); }
    P2PStaging(const P2PStaging&) = delete;
    P2PStaging& operator=(const P2PStaging&) = delete;

    // Frees the staging and everything cached with it (the programs hold device memory for their binaries). Call it before the mesh is closed. If the
    // mesh is already closed the Programs are LEAKED on purpose: their destructors unregister from Device objects that close() has destroyed, which
    // is a use after free (the MeshBuffers are safe, they know a closed mesh).
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto locked = mesh_.lock();
        if (!locked || !locked->is_initialized()) {
            for (auto& entry : scatter_) {
                (void)entry.second.release();
            }
            for (auto& entry : checksum_) {
                (void)entry.second.release();
            }
        }
        try {
            scatter_.clear();
            checksum_.clear();
            partials_.reset();
            buffer_.reset();
        } catch (const std::exception&) {
            // The mesh may already be closed: its memory is gone with it.
        }
    }

    bool released() const { return buffer_ == nullptr; }
    // The mesh, kept alive by the caller for as long as it uses it.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh() const {
        auto locked = mesh_.lock();
        if (!locked || !locked->is_initialized()) {
            throw nb::value_error("the mesh device of this staging buffer is closed");
        }
        return locked;
    }
    uint64_t size_bytes() const { return size_bytes_; }
    uint64_t page_bytes() const { return page_bytes_; }
    uint32_t bank() const { return bank_; }
    uint32_t num_banks() const { return num_banks_; }
    uint64_t address() const { return address_; }
    int64_t bank_offset() const { return bank_offset_; }
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& buffer() const { return require_buffer(); }
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& partials() const {
        require_buffer();
        return partials_;
    }
    const tt::tt_metal::Statistics& dram_before() const { return dram_free_before_; }
    const tt::tt_metal::Statistics& dram_after() const { return dram_free_after_; }

    // A cached program: built once per distinct compile-time arguments, then only its runtime args change between calls.
    struct Cached {
        tt::tt_metal::distributed::MeshWorkload workload;
        tt::tt_metal::KernelHandle kernel{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    std::map<std::vector<uint32_t>, std::unique_ptr<Cached>>& scatter_programs() { return scatter_; }
    std::map<std::vector<uint32_t>, std::unique_ptr<Cached>>& checksum_programs() { return checksum_; }
    std::mutex& mutex() { return mutex_; }

private:
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& require_buffer() const {
        if (!buffer_) {
            throw nb::value_error("this staging buffer was released");
        }
        return buffer_;
    }

    std::weak_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    uint64_t size_bytes_ = 0;
    uint64_t page_bytes_ = 0;
    uint32_t bank_ = 0;
    uint32_t num_banks_ = 0;
    uint64_t address_ = 0;
    int64_t bank_offset_ = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer_;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> partials_;
    tt::tt_metal::Statistics dram_free_before_{};
    tt::tt_metal::Statistics dram_free_after_{};
    std::mutex mutex_;
    std::map<std::vector<uint32_t>, std::unique_ptr<Cached>> scatter_;
    std::map<std::vector<uint32_t>, std::unique_ptr<Cached>> checksum_;
};

namespace qwen_p2p_host {

using tt::tt_metal::BufferRegion;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::ShardDataTransfer;

// The directory holding the two kernels: next to this module, or QWEN_P2P_KERNEL_DIR.
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
        throw nb::value_error(("the P2P kernel " + path + " is missing (it ships beside qwen_kv_read.so; QWEN_P2P_KERNEL_DIR overrides)").c_str());
    }
    return path;
}

// PCI address of a tt-kmd device N (/dev/tenstorrent/N) from sysfs, "" when it cannot be read (a container without /sys/class).
std::string pci_bus_id_of(uint32_t pcie_device) {
    std::error_code error;
    const auto link = std::filesystem::read_symlink(
        "/sys/class/tenstorrent/tenstorrent!" + std::to_string(pcie_device) + "/device", error);
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
        throw nb::value_error(
            (std::string(what) + ": the compute grid is smaller than " + std::to_string(columns) + " x " + std::to_string(rows)).c_str());
    }
}

// Builds the program (one kernel on `count` cores in rows of `columns`, one circular buffer of cb_bytes as L1 scratch) with runtime args of
// `arg_count` words, as a MeshWorkload over the whole mesh. The runtime args are placeholders until the caller fills them.
std::unique_ptr<P2PStaging::Cached> build_program(
    MeshDevice& mesh,
    const char* kernel_name,
    const std::vector<uint32_t>& compile_args,
    uint32_t count,
    uint32_t columns,
    uint32_t cb_bytes,
    uint32_t arg_count) {
    using namespace tt::tt_metal;
    check_grid(mesh, count, columns, kernel_name);
    auto cached = std::make_unique<P2PStaging::Cached>();
    Program program = CreateProgram();
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
tt::tt_metal::Program& program_of(P2PStaging::Cached& cached) { return cached.workload.get_programs().begin()->second; }

void set_args(P2PStaging::Cached& cached, size_t core, const std::vector<uint32_t>& values) {
    auto& args = tt::tt_metal::GetRuntimeArgs(program_of(cached), cached.kernel, cached.cores[core]);
    for (size_t i = 0; i < values.size(); ++i) {
        args[i] = values[i];
    }
}

// ttnn.qwen_write_blocks_from_staging: the semantics of qwen_write_blocks_raw with the source bytes in the staging range. Returns without waiting.
// Row i (the bytes of blocks[i]) comes from staging[byte_offset + i * row_stride : + slice bytes]; no row_stride is the slice bytes (contiguous rows).
void write_blocks_from_staging(
    const ttnn::Tensor& device_tensor,
    P2PStaging& staging,
    uint64_t byte_offset,
    const std::vector<uint32_t>& blocks,
    std::optional<uint32_t> bank_opt,
    std::optional<uint64_t> row_stride_opt) {
    using namespace tt::tt_metal;
    auto* mesh = device_tensor.device();
    if (mesh == nullptr || !device_tensor.is_allocated()) {
        throw nb::value_error("qwen_write_blocks_from_staging: device_tensor must be a tensor allocated on a mesh device");
    }
    const auto staging_mesh = staging.mesh();
    if (mesh != staging_mesh.get()) {
        throw nb::value_error("qwen_write_blocks_from_staging: the tensor and the staging buffer are on different meshes");
    }
    if (blocks.empty()) {
        return;
    }
    const MeshTensor& mesh_tensor = device_tensor.mesh_tensor();
    const std::shared_ptr<distributed::MeshBuffer>& destination = mesh_tensor.impl().raw_mesh_buffer();
    const Buffer* reference = destination->get_reference_buffer();
    if (reference->buffer_type() != BufferType::DRAM || reference->buffer_layout() != TensorMemoryLayout::INTERLEAVED) {
        throw nb::value_error("qwen_write_blocks_from_staging: the tensor must be an interleaved DRAM tensor");
    }
    const uint64_t slices = mesh_tensor.tensor_spec().logical_shape()[0];
    const uint64_t slice_bytes = qwen_kv::slice_bytes_of(mesh_tensor);
    const uint64_t page_bytes = reference->page_size();
    if (reference->aligned_page_size() != page_bytes) {
        throw nb::value_error("qwen_write_blocks_from_staging: the tensor's page must be DRAM aligned");
    }
    // The program is built once for the reference buffer and its addresses go to every device: every device's buffer must be laid out alike.
    {
        const distributed::MeshCoordinateRange all(mesh->shape());
        for (const auto& coord : all) {
            const Buffer* device_buffer = destination->get_device_buffer(coord);
            if (device_buffer == nullptr || device_buffer->address() != reference->address() || device_buffer->page_size() != reference->page_size() ||
                device_buffer->buffer_layout() != reference->buffer_layout() || device_buffer->buffer_type() != reference->buffer_type()) {
                throw nb::value_error("qwen_write_blocks_from_staging: the tensor's buffers differ between devices (address, page size or layout)");
            }
        }
    }
    const uint32_t bank = bank_opt.value_or(staging.bank());
    if (bank >= staging.num_banks()) {
        throw nb::value_error("qwen_write_blocks_from_staging: bank is outside the device's DRAM banks");
    }
    const uint64_t row_stride = row_stride_opt.value_or(slice_bytes);
    if (const char* problem =
            qwen_p2p::check_scatter(byte_offset, slice_bytes, page_bytes, blocks.size(), staging.size_bytes(), row_stride)) {
        throw nb::value_error((std::string("qwen_write_blocks_from_staging: ") + problem).c_str());
    }
    for (uint32_t index : blocks) {
        if (index >= slices) {
            throw nb::value_error(("qwen_write_blocks_from_staging: block " + std::to_string(index) + " is outside the tensor").c_str());
        }
    }
    {
        std::vector<uint32_t> sorted(blocks);
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw nb::value_error("qwen_write_blocks_from_staging: the block ids must be distinct (two writes to one block have no order)");
        }
    }
    const uint32_t pages_per_block = static_cast<uint32_t>(slice_bytes / page_bytes);
    const std::shared_ptr<distributed::MeshBuffer>& source = staging.buffer();

    std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(page_bytes), pages_per_block, qwen_p2p::kScatterChunkPages, qwen_p2p::kScatterMaxBlocksPerCore};
    TensorAccessorArgs(*destination).append_to(compile_args);
    TensorAccessorArgs(*source).append_to(compile_args);

    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(staging.mutex());
    if (staging.released()) {
        throw nb::value_error("qwen_write_blocks_from_staging: the staging buffer was released");
    }
    auto& slot = staging.scatter_programs()[compile_args];
    if (!slot) {
        const uint32_t cb_bytes = qwen_p2p::scatter_cb_bytes(static_cast<uint32_t>(page_bytes));
        slot = build_program(
            *mesh,
            "qwen_p2p_scatter.cpp",
            compile_args,
            qwen_p2p::kScatterCores,
            qwen_p2p::kScatterColumns,
            cb_bytes,
            qwen_p2p::kScatterArgs);
    }
    // The addresses are read from the live buffers on EVERY call and written over the cached program's runtime args (what a program-cache hit's
    // override does): a tensor or staging buffer that moved is followed, never remembered.
    const uint32_t destination_address = reference->address();
    const uint32_t staging_address = static_cast<uint32_t>(staging.address());
    for (const auto& [launch_first, launch_count] : qwen_p2p::scatter_launches(static_cast<uint32_t>(blocks.size()))) {
        const auto runs = qwen_p2p::split_blocks(launch_count, qwen_p2p::kScatterCores);
        for (size_t core = 0; core < runs.size(); ++core) {
            set_args(
                *slot,
                core,
                qwen_p2p::scatter_core_args(
                    destination_address, staging_address, bank, byte_offset, row_stride, blocks, launch_first, runs[core]));
        }
        distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), slot->workload, /*blocking=*/false);
    }
}

// ttnn.qwen_staging_checksum: the per-device checksums of `count` consecutive staging byte ranges of `length` bytes (the algorithm of qwen_p2p_checksum.cpp, k
// from 0 at the start of EACH range). Blocking. -> [device][range]. One launch carries up to kChecksumMaxRanges ranges; more are several launches, each read back
// before the next because they share the partials buffer.
std::vector<std::vector<uint64_t>> staging_checksum(
    P2PStaging& staging, uint64_t byte_offset, uint64_t length, std::optional<uint32_t> bank_opt, uint32_t cores, uint32_t count) {
    using namespace tt::tt_metal;
    if (const char* problem = qwen_p2p::check_checksum(byte_offset, length, staging.size_bytes(), cores, count)) {
        throw nb::value_error((std::string("qwen_staging_checksum: ") + problem).c_str());
    }
    const uint32_t bank = bank_opt.value_or(staging.bank());
    if (bank >= staging.num_banks()) {
        throw nb::value_error("qwen_staging_checksum: bank is outside the device's DRAM banks");
    }
    const auto mesh_keeper = staging.mesh();
    MeshDevice& mesh = *mesh_keeper;
    const std::shared_ptr<distributed::MeshBuffer> source = staging.buffer();
    const std::shared_ptr<distributed::MeshBuffer> partials = staging.partials();

    std::vector<uint32_t> compile_args = {qwen_p2p::kChecksumChunkBytes, qwen_p2p::kChecksumPartialBytes};
    TensorAccessorArgs(*source).append_to(compile_args);
    TensorAccessorArgs(*partials).append_to(compile_args);
    compile_args.push_back(cores);  // not read by the kernel: it keys the cached program (the core set is part of the program)

    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(staging.mutex());
    if (staging.released()) {
        throw nb::value_error("qwen_staging_checksum: the staging buffer was released");
    }
    auto& slot = staging.checksum_programs()[compile_args];
    if (!slot) {
        const uint32_t cb_bytes = qwen_p2p::checksum_cb_bytes();
        slot = build_program(
            mesh,
            "qwen_p2p_checksum.cpp",
            compile_args,
            cores,
            qwen_p2p::kChecksumColumns,
            cb_bytes,
            qwen_p2p::kChecksumArgs);
    }
    const auto units = qwen_p2p::split_units(length / qwen_p2p::kChecksumUnit, cores);
    auto& queue = mesh.mesh_command_queue();
    const MeshCoordinateRange all(mesh.shape());
    const std::vector<MeshCoordinate> coords(all.begin(), all.end());
    std::vector<std::vector<uint64_t>> sums(coords.size(), std::vector<uint64_t>(count, 0));
    for (const auto& [launch_first, launch_count] : qwen_p2p::checksum_launches(count)) {
        const uint64_t launch_offset = byte_offset + uint64_t(launch_first) * length;
        for (uint32_t core = 0; core < cores; ++core) {
            set_args(
                *slot,
                core,
                qwen_p2p::checksum_core_args(
                    static_cast<uint32_t>(staging.address()),
                    static_cast<uint32_t>(partials->address()),
                    bank,
                    launch_offset,
                    units[core],
                    core,
                    launch_count,
                    length,
                    cores));
        }
        distributed::EnqueueMeshWorkload(queue, slot->workload, /*blocking=*/false);

        // The partial of range r, core c is partials page r * cores + c.
        const uint64_t per_device = uint64_t(launch_count) * cores * qwen_p2p::kChecksumPartialBytes;
        std::vector<uint8_t> host(coords.size() * per_device);
        std::vector<ShardDataTransfer> transfers;
        for (size_t device = 0; device < coords.size(); ++device) {
            transfers.push_back(ShardDataTransfer{coords[device]}
                                    .host_data(host.data() + device * per_device)
                                    .region(BufferRegion(0, per_device)));
        }
        queue.enqueue_read_shards(transfers, partials, /*blocking=*/true);
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

// Diagnostics: bytes of the staging range <-> host memory through the command queue. The range is page aligned (byte_offset and the length are
// multiples of the staging page size) because a buffer region is whole pages; bank byte x lives in page (x div P) * banks + bank.
void staging_transfer(
    P2PStaging& staging, uint64_t byte_offset, uint8_t* host, uint64_t host_bytes, std::optional<uint32_t> bank_opt, bool write) {
    using namespace tt::tt_metal;
    const uint64_t page = staging.page_bytes();
    const uint32_t bank = bank_opt.value_or(staging.bank());
    const auto mesh_keeper = staging.mesh();
    MeshDevice& mesh = *mesh_keeper;
    const MeshCoordinateRange all(mesh.shape());
    const std::vector<MeshCoordinate> coords(all.begin(), all.end());
    if (bank >= staging.num_banks() || host_bytes % coords.size() != 0) {
        throw nb::value_error("qwen_p2p_staging_write/read: bad bank or a buffer that is not [devices, n]");
    }
    const uint64_t per_device = host_bytes / coords.size();
    if (per_device == 0 || byte_offset % page != 0 || per_device % page != 0 || byte_offset > staging.size_bytes() ||
        per_device > staging.size_bytes() - byte_offset) {
        throw nb::value_error("qwen_p2p_staging_write/read: byte_offset and the per-device length must be multiples of the staging page size and inside the range");
    }
    std::vector<ShardDataTransfer> transfers;
    transfers.reserve(coords.size() * (per_device / page));
    for (size_t device = 0; device < coords.size(); ++device) {
        for (uint64_t at = 0; at < per_device; at += page) {
            const uint64_t page_index = (byte_offset + at) / page;
            transfers.push_back(ShardDataTransfer{coords[device]}
                                    .host_data(host + device * per_device + at)
                                    .region(BufferRegion((page_index * staging.num_banks() + bank) * page, page)));
        }
    }
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(staging.mutex());
    if (staging.released()) {
        throw nb::value_error("qwen_p2p_staging_write/read: the staging buffer was released");
    }
    auto& queue = mesh.mesh_command_queue();
    if (write) {
        queue.enqueue_write_shards(staging.buffer(), transfers, /*blocking=*/true);
    } else {
        queue.enqueue_read_shards(transfers, staging.buffer(), /*blocking=*/true);
    }
}

nb::dict bank_description(tt::tt_metal::IDevice* device, uint32_t bank) {
    // A Blackhole DRAM channel (a "DRAM view") has three NOC endpoints. tt-metal's logical DRAM coordinate is (view, endpoint index): index 0 is the view's
    // NOC0 worker endpoint, index 1 its NOC1 worker endpoint, then the remaining endpoint (metal_soc_descriptor.cpp). The result is the TRANSLATED coordinate
    // the TLB registers take, from the same tables the device uses for its own bank addresses. The default (noc_x, noc_y) is endpoint 0: the one the kernels
    // and tt-metal's own host reads of DRAM (Cluster::read_dram_vec) go through. Any endpoint reaches the same channel; the host's window self-test settles it.
    nb::list endpoints;
    nb::dict bank_info;
    bank_info["dram_bank"] = bank;
    try {
        uint32_t first_x = 0, first_y = 0;
        for (uint32_t index = 0; index < 3; ++index) {
            const CoreCoord endpoint = device->virtual_core_from_logical_core(CoreCoord(bank, index), tt::CoreType::DRAM);
            nb::list triple;
            triple.append(static_cast<uint32_t>(endpoint.x));
            triple.append(static_cast<uint32_t>(endpoint.y));
            endpoints.append(triple);
            if (index == 0) {
                first_x = endpoint.x;
                first_y = endpoint.y;
            }
        }
        bank_info["noc_x"] = first_x;
        bank_info["noc_y"] = first_y;
        bank_info["endpoint_index"] = 0;
    } catch (const std::exception& error) {
        bank_info["noc_x"] = nb::none();
        bank_info["noc_y"] = nb::none();
        bank_info["endpoint_error"] = error.what();
    }
    bank_info["noc_endpoints"] = endpoints;
    return bank_info;
}

nb::list staging_info(P2PStaging& staging) {
    using namespace tt::tt_metal;
    const auto mesh_keeper = staging.mesh();
    MeshDevice& mesh = *mesh_keeper;
    const auto devices = mesh.get_devices();
    const MeshCoordinateRange all(mesh.shape());
    const std::vector<MeshCoordinate> coords(all.begin(), all.end());
    nb::list result;
    for (size_t index = 0; index < devices.size(); ++index) {
        IDevice* device = devices[index];
        nb::dict entry = bank_description(device, staging.bank());
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
        entry["address"] = static_cast<uint64_t>(static_cast<int64_t>(staging.address()) + staging.bank_offset());
        entry["size"] = staging.size_bytes();
        // The range start is 64-byte aligned (what the DRAM allocator guarantees). An O_DIRECT object wants 4096-byte aligned bank addresses: the first one
        // is `aligned_4k_offset` bytes into the range, and `aligned_4k_size` bytes of it are usable from there in whole 4096-byte units.
        const uint64_t start = static_cast<uint64_t>(static_cast<int64_t>(staging.address()) + staging.bank_offset());
        const uint64_t lead = (4096 - start % 4096) % 4096;
        entry["aligned_4k_offset"] = lead;
        entry["aligned_4k_address"] = start + lead;
        entry["aligned_4k_size"] = staging.size_bytes() > lead ? (staging.size_bytes() - lead) / 4096 * 4096 : uint64_t(0);
        entry["bank_offset"] = staging.bank_offset();
        entry["page_bytes"] = staging.page_bytes();
        // What the allocation took: the same range in every bank (lockstep), and what the allocator had left around it.
        entry["bytes_reserved_per_bank"] = staging.size_bytes();
        entry["bytes_reserved"] = staging.size_bytes() * staging.num_banks();
        entry["dram_free_per_bank_before"] = static_cast<uint64_t>(staging.dram_before().total_free_bytes);
        entry["dram_free_per_bank_after"] = static_cast<uint64_t>(staging.dram_after().total_free_bytes);
        entry["dram_largest_free_block_after"] = static_cast<uint64_t>(staging.dram_after().largest_free_block_bytes);
        nb::list banks;
        for (uint32_t bank = 0; bank < staging.num_banks(); ++bank) {
            banks.append(bank_description(device, bank));
        }
        entry["banks"] = banks;
        result.append(entry);
    }
    return result;
}

}  // namespace qwen_p2p_host

NB_MODULE(qwen_kv_read, m) {
    m.doc() = "ttnn.qwen_read_blocks: region read of a paged KV cache (docs/prefix-audit-cost.md)";
    // The module must load after ttnn: it names ttnn.Tensor, whose nanobind type ttnn registers.
    nb::module_ ttnn_module = nb::module_::import_("ttnn");
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
            auto& cq = mesh->mesh_command_queue();
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
                mesh->mesh_command_queue(), device_tensor.mesh_tensor(), host, host_bytes, blocks, /*write=*/false);
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
                mesh->mesh_command_queue(), device_tensor.mesh_tensor(), host, host_bytes, blocks, /*write=*/true);
        },
        nb::arg("device_tensor"),
        nb::arg("data"),
        nb::arg("blocks"),
        "Writes raw packed bytes [devices, len(blocks), qwen_block_bytes] into the named dimension-0 slices (paged KV cache blocks) of a "
        "device tensor, on every device, and returns when the device has them. No pack; no program is compiled.");
    // Version 3: the P2P restore's device side.
    nb::class_<P2PStaging>(m, "QwenP2PStaging")
        .def("release", &P2PStaging::release, "Frees the staging buffer and its cached programs. Call it before the mesh is closed.")
        .def_prop_ro("size_bytes", &P2PStaging::size_bytes)
        .def_prop_ro("page_bytes", &P2PStaging::page_bytes)
        .def_prop_ro("bank", &P2PStaging::bank)
        .def_prop_ro("num_banks", &P2PStaging::num_banks)
        .def_prop_ro("address", &P2PStaging::address)
        .def_prop_ro("released", &P2PStaging::released);
    m.def(
        "qwen_p2p_staging_alloc",
        [](tt::tt_metal::distributed::MeshDevice& mesh, uint64_t size_bytes, uint32_t bank, uint64_t page_bytes, bool bottom_up) {
            return new P2PStaging(mesh, size_bytes, bank, page_bytes, bottom_up);
        },
        nb::arg("mesh_device"),
        nb::arg("size_bytes") = uint64_t(128) << 20,
        nb::arg("bank") = 0,
        nb::arg("page_bytes") = uint64_t(64) << 10,
        nb::arg("bottom_up") = false,
        nb::rv_policy::take_ownership,
        "Allocates the P2P staging buffer: on every device of the mesh ONE contiguous range of size_bytes in DRAM bank `bank` that the allocator never "
        "hands to anything else while the handle lives. The DRAM allocator reserves the same range in all banks, so the cost is size_bytes x banks per card "
        "(the other banks' ranges are usable through bank=). Keep the handle for the engine's life; release() it before closing the mesh.");
    m.def(
        "qwen_p2p_staging_info",
        [](P2PStaging& staging) { return qwen_p2p_host::staging_info(staging); },
        nb::arg("staging"),
        "One dict per device in mesh (row-major) order: device_id, pcie_device, pci_bus_id (None when sysfs cannot be read), dram_bank, noc_x, noc_y "
        "(translated coordinates of a NOC endpoint of the bank), noc_endpoints (all of the bank's endpoints), address (byte address in the bank as "
        "seen through a TLB window aimed at (noc_x, noc_y)), size, page_bytes, and the allocation's cost.");
    m.def(
        "qwen_write_blocks_from_staging",
        [](const ttnn::Tensor& device_tensor,
           P2PStaging& staging,
           uint64_t byte_offset,
           const std::vector<uint32_t>& blocks,
           std::optional<uint32_t> bank,
           std::optional<uint64_t> row_stride) {
            qwen_p2p_host::write_blocks_from_staging(device_tensor, staging, byte_offset, blocks, bank, row_stride);
        },
        nb::arg("device_tensor"),
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("blocks"),
        nb::arg("bank") = nb::none(),
        nb::arg("row_stride") = nb::none(),
        "qwen_write_blocks_raw with the source in the staging range: device d's slice blocks[i] of device_tensor gets "
        "staging_d[byte_offset + i * row_stride : + slice_bytes]. row_stride=None is slice_bytes (contiguous rows); otherwise a multiple of 64 and at least "
        "slice_bytes, and the whole range must lie inside the staging range. Full tile pages only; enqueued, it does not wait.");
    m.def(
        "qwen_staging_checksum",
        [](P2PStaging& staging,
           uint64_t byte_offset,
           uint64_t length,
           std::optional<uint32_t> bank,
           uint32_t cores,
           std::optional<uint32_t> count) -> nb::object {
            const auto sums = qwen_p2p_host::staging_checksum(staging, byte_offset, length, bank, cores, count.value_or(1));
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
        "Per device, S = sum_k (((k * 2654435761 + 1) mod 2^32) * x_k) mod 2^64 over the little-endian uint32 words x_k of staging[byte_offset : + length] "
        "(k from 0 at byte_offset; length a multiple of 4096). Blocking: returns a list with one int per device in mesh order. count=N checksums N consecutive "
        "ranges of `length` bytes (the range r starts at byte_offset + r * length) in one call, k counted from each range's own start, and returns a list "
        "with a list of N ints per device.");
    m.def(
        "qwen_p2p_staging_write",
        [](P2PStaging& staging, uint64_t byte_offset, nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu> data, std::optional<uint32_t> bank) {
            qwen_p2p_host::staging_transfer(
                staging, byte_offset, const_cast<uint8_t*>(data.data()), data.nbytes(), bank, /*write=*/true);
        },
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("data"),
        nb::arg("bank") = nb::none(),
        "Diagnostic: writes uint8 data [devices, n] into the staging range through the command queue (what a drive's DMA would write). "
        "byte_offset and n are multiples of the staging page size. Blocking.");
    m.def(
        "qwen_p2p_staging_read",
        [](P2PStaging& staging, uint64_t byte_offset, nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu> out, std::optional<uint32_t> bank) {
            qwen_p2p_host::staging_transfer(staging, byte_offset, out.data(), out.nbytes(), bank, /*write=*/false);
        },
        nb::arg("staging"),
        nb::arg("byte_offset"),
        nb::arg("out"),
        nb::arg("bank") = nb::none(),
        "Diagnostic: reads the staging range into uint8 out [devices, n]. byte_offset and n are multiples of the staging page size. Blocking.");
    m.attr("QWEN_KV_READ_VERSION") = "3";
    for (const char* name : {"qwen_read_blocks", "qwen_block_bytes", "qwen_read_blocks_raw", "qwen_write_blocks_raw", "qwen_p2p_staging_alloc",
                             "qwen_p2p_staging_info", "qwen_write_blocks_from_staging", "qwen_staging_checksum", "qwen_p2p_staging_write",
                             "qwen_p2p_staging_read"}) {
        if (!nb::hasattr(ttnn_module, name)) {
            ttnn_module.attr(name) = m.attr(name);
        }
    }
}
