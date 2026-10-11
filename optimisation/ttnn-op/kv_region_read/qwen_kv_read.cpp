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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
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
    m.attr("QWEN_KV_READ_VERSION") = "2";
    for (const char* name : {"qwen_read_blocks", "qwen_block_bytes", "qwen_read_blocks_raw", "qwen_write_blocks_raw"}) {
        if (!nb::hasattr(ttnn_module, name)) {
            ttnn_module.attr(name) = m.attr(name);
        }
    }
}
