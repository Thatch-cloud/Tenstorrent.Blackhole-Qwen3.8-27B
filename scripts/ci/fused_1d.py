"""Isolated B1 TP2 gate/up kernel; reuse the native 1D K loop with a rounded epilogue."""

import hashlib
from pathlib import Path


COMPUTE = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp"


def fused_compute(source, intermediates=False):
    start = source.index("                            if (last_out) {")
    end = source.index("                            } else {\n                                tile_regs_commit();", start)
    if source.count("                            if (last_out) {") != 1:
        raise ValueError("Native final-pack anchor changed")
    epilogue = """                            if (last_out) {
                                static_assert(out_subblock_num_tiles == 2);
                                constexpr uint32_t rounded_cb = 30;
                                tile_regs_commit();
                                cb_reserve_back(rounded_cb, 2);
                                apply_activation_from_pack<KernelActivation::SILU>(1);
                                PACK((pack_reconfig_data_format(rounded_cb)));
                                PACK((llk_pack_reconfig_l1_acc(0)));
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, rounded_cb, 2);
                                tile_regs_release();
                                cb_push_back(rounded_cb, 2);
"""
    result = source[:start] + epilogue + source[end:]
    finish = result.rfind("}")
    result = result[:finish] + """
    constexpr uint32_t rounded_cb = 30;
    cb_wait_front(rounded_cb, 14);
    reconfig_data_format_srca(rounded_cb);
    copy_tile_to_dst_init_short(rounded_cb);
    mul_binary_tile_init();
    for (uint32_t pair = 0; pair < 7; ++pair) {
        cb_wait_front(rounded_cb, 2);
        tile_regs_acquire();
        copy_tile(rounded_cb, 0, 0);
        copy_tile(rounded_cb, 1, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(out_dfb_id, 1);
        tile_regs_wait();
        PACK((pack_reconfig_data_format(out_dfb_id)));
        pack_tile(0, out_dfb_id);
        tile_regs_release();
        cb_push_back(out_dfb_id, 1);
        cb_pop_front(rounded_cb, 2);
    }
""" + result[finish:]
    result = '#include "api/compute/eltwise_binary_sfpu.h"\n' + result
    if intermediates:
        result = result.replace("    mul_binary_tile_init();\n", "")
        result = result.replace("        mul_binary_tile(0, 1, 0);\n", "")
        result = result.replace("cb_reserve_back(out_dfb_id, 1);", "cb_reserve_back(out_dfb_id, 2);")
        result = result.replace("pack_tile(0, out_dfb_id);", "pack_tile(0, out_dfb_id);\n        pack_tile(1, out_dfb_id);")
        result = result.replace("cb_push_back(out_dfb_id, 1);", "cb_push_back(out_dfb_id, 2);")
    return result.replace('"bmm_fused_activation.hpp"',
        '"ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"')


def mapping():
    return [(index % 11, index // 11, index * 7, min(7, 272 - index * 7)) for index in range(39)]


class FusedProjection:
    def __init__(self, mesh, weights, intermediates=False):
        self.mesh = mesh
        self.weights = weights
        self.source = Path("/opt/tt-metal") / COMPUTE
        original = self.source.read_text()
        self.intermediates = intermediates
        self.compute = fused_compute(original, intermediates=intermediates)
        self.manifest = dict(native_compute_sha256=hashlib.sha256(original.encode()).hexdigest(),
                             fused_compute_sha256=hashlib.sha256(self.compute.encode()).hexdigest(),
                             workers=39, grid=[11, 4], pairs_per_worker=7, k_block=8,
                             intermediates=intermediates, input_noc=1, weight_noc=0,
                             epilogue="BF16(silu(gate)), BF16(up), then BF16 multiply")

    def __call__(self, value):
        import ttnn
        if list(value.shape) != [1, 1, 1, 5120] or value.dtype != ttnn.bfloat16:
            raise ValueError("Only frozen BF16 B1 projection input is supported")
        if self.weights.dtype != ttnn.bfloat4_b or list(self.weights.shape)[-2:] != [5120, 17408]:
            raise ValueError("Expected local TP2 pair-packed BF4 weights")
        output_tiles = 2 if self.intermediates else 1
        output = ttnn.empty((1, 1, 1, 8704 * output_tiles), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=self.mesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 3))])
        workers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(5, 3))])
        def cb(index, dtype, page, count, cores):
            return ttnn.CBDescriptor(total_size=page * count, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        buffers = [cb(0, ttnn.bfloat16, 2048, 16, all_cores),
                   cb(1, ttnn.bfloat4_b, 576, 224, workers),
                   cb(4, ttnn.bfloat16, 2048, 7 * output_tiles, workers),
                   cb(5, ttnn.float32, 4096, 14, workers),
                   cb(30, ttnn.bfloat16, 2048, 14, workers)]
        mesh_program = ttnn.MeshProgramDescriptor()
        shards = zip(ttnn.get_device_tensors(value), ttnn.get_device_tensors(self.weights),
                     ttnn.get_device_tensors(output))
        for chip, (local_input, local_weight, local_output) in enumerate(shards):
            device = local_input.device()
            first = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            last = device.worker_core_from_logical_core(ttnn.CoreCoord(10, 3))
            input_kernel = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("fused_1d_input.cpp")), core_ranges=all_cores,
                compile_time_args=ttnn.TensorAccessorArgs(local_input).get_compile_time_args(),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                         noc=ttnn.NOC.RISCV_1_default))
            input_args = ttnn.RuntimeArgs()
            for index in range(44):
                input_args[index % 11][index // 11] = [local_input.buffer_address(), index,
                                                       first.x, first.y, last.x, last.y]
            input_kernel.runtime_args = input_args
            writer = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("fused_1d_weights.cpp")), core_ranges=workers,
                compile_time_args=ttnn.TensorAccessorArgs(local_weight).get_compile_time_args()
                                  + ttnn.TensorAccessorArgs(local_output).get_compile_time_args(),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.RISCV_0_default))
            writer_args = ttnn.RuntimeArgs()
            for core_x, core_y, begin, count in mapping():
                writer_args[core_x][core_y] = [local_weight.buffer_address(), local_output.buffer_address(), begin, count, output_tiles]
            writer.runtime_args = writer_args
            compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True, math_approx_mode=False)
            unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
            unpack_modes[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
            compute_config.unpack_to_dest_mode.extend(unpack_modes)
            compute = ttnn.KernelDescriptor(kernel_source=self.compute,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=workers,
                compile_time_args=[8, 1, 8, 8, 7, 112, 14, 20, 1, 1, 1, 2, 2, 1, 14, 0, 0, 0],
                named_compile_time_args=[("cb_in0", 0), ("cb_in1", 1), ("cb_out", 4),
                    ("cb_intermed0", 5), ("cb_in0_transposed", 10), ("activation_type", 4),
                    ("activation_param0", 0), ("activation_param1", 0), ("activation_param2", 0)],
                defines=[("FP32_DEST_ACC_EN", "1"), ("PACKER_L1_ACC", "1"), ("SFPU_ACTIVATION", "1")],
                config=compute_config)
            program = ttnn.ProgramDescriptor(kernels=[input_kernel, writer, compute], cbs=buffers,
                semaphores=[ttnn.SemaphoreDescriptor(id=index, core_ranges=all_cores, initial_value=0)
                            for index in (0, 1)])
            coordinate = ttnn.MeshCoordinate(0, chip)
            mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = program
        ttnn.generic_op([value, self.weights, output], mesh_program)
        return output
