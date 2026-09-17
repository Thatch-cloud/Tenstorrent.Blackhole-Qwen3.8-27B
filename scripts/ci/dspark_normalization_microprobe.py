"""Bounded simulator diagnostic of the SDPA normalization primitive, not timing."""

import argparse
import json
import os
from pathlib import Path
import struct


READER = r'''
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr uint32_t repeats = get_compile_time_arg_val(1);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
    cb_reserve_back(0, tiles);
    cb_reserve_back(1, 1);
    auto* numerator = reinterpret_cast<volatile uint16_t*>(get_write_ptr(0));
    auto* reciprocal = reinterpret_cast<volatile uint32_t*>(get_write_ptr(1));
    for (uint32_t index = 0; index < tiles * 1024; ++index) {
        numerator[index] = 0xc260 + repeat * tiles + index / 1024;
    }
    for (uint32_t index = 0; index < 1024; ++index) {
        const uint32_t face = index / 256;
        const uint32_t row = (face / 2) * 16 + (index % 256) / 16;
        const uint32_t column = (face % 2) * 16 + index % 16;
        reciprocal[index] = column == 0 ? get_arg_val<uint32_t>(row) : 0x40e00000;
    }
    cb_push_back(0, tiles);
    cb_push_back(1, 1);
    }
}
'''

COMPUTE = r'''
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    binary_op_init_common(0, 1, 16);
    reconfig_data_format(0, 1);
    pack_reconfig_data_format(16);
    mul_bcast_cols_init(0, 1);
    CircularBuffer(0).wait_front(1);
    CircularBuffer(1).wait_front(1);
    PACK((llk_pack_reconfig_l1_acc(false)));
    CircularBuffer(16).reserve_back(1);
    tile_regs_acquire();
    mul_tiles_bcast_cols(0, 1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    CircularBuffer(1).pop_front(1);
    CircularBuffer(0).pop_front(1);
    CircularBuffer(16).push_back(1);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32-output', action='store_true')
    parser.add_argument('--sfpu-scalar', action='store_true')
    parser.add_argument('--sfpu-column', action='store_true')
    parser.add_argument('--vary-rows', action='store_true')
    parser.add_argument('--scratch-copy', action='store_true')
    parser.add_argument('--tiles', type=int, choices=(1, 4), default=1)
    parser.add_argument('--repeats', type=int, choices=(1, 3), default=1)
    options = parser.parse_args()
    if (options.tiles != 1 or options.repeats != 1) and not options.scratch_copy:
        parser.error('Multiple tiles or repeats require scratch-copy mode')
    if options.scratch_copy and not options.sfpu_column:
        parser.error('Scratch copy requires SFPU column mode')
    if options.sfpu_scalar and (options.sfpu_column or options.vary_rows):
        parser.error('Scalar mode requires constant rows and cannot use column mode')
    if not os.environ.get('TT_METAL_SIMULATOR') or Path('/dev/tenstorrent').exists():
        raise RuntimeError('Simulator without physical devices required')
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        output_dtype = ttnn.float32 if options.fp32_output else ttnn.bfloat16
        output_bytes = 4096 if options.fp32_output else 2048
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape([options.repeats, 1, 32, 32 * options.tiles]), output_dtype, ttnn.TILE_LAYOUT,
            device, ttnn.DRAM_MEMORY_CONFIG)
        shape_input = ttnn.from_torch(torch.full((options.repeats, 1, 32, 32 * options.tiles), -56.0),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        buffers = [ttnn.CBDescriptor(total_size=size * (options.tiles if index in (0, 16) else 1), core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=size)])
            for index, dtype, size in ((0, ttnn.bfloat16, 2048),
                (1, ttnn.float32, 4096), (16, output_dtype, output_bytes))]
        reciprocal = struct.unpack('<f', struct.pack('<f', 0.806062460))[0]
        reciprocals = [reciprocal + row / 128 if options.vary_rows else reciprocal for row in range(32)]
        reader_args = ttnn.RuntimeArgs()
        reader_args[0][0] = [struct.unpack('<I', struct.pack('<f', value))[0] for value in reciprocals]
        writer_args = ttnn.RuntimeArgs()
        writer_args[0][0] = [output.buffer_address(), options.tiles * options.repeats, 0]
        compute = COMPUTE
        if options.sfpu_scalar:
            compute = compute.replace('#include "api/compute/bcast.h"',
                '#include "api/compute/bcast.h"\n'
                '#include "api/compute/tile_move_copy.h"\n'
                '#include "api/compute/eltwise_unary/binop_with_scalar.h"')
            compute = compute.replace('mul_bcast_cols_init(0, 1);', 'copy_tile_init(0);')
            compute = compute.replace('mul_tiles_bcast_cols(0, 1, 0, 0, 0);',
                'copy_tile(0, 0, 0);\n    mul_unary_tile(0, get_arg_val<uint32_t>(0));')
        config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False)
        if options.sfpu_column:
            compute = compute.replace('#include "api/compute/bcast.h"',
                '#include "api/compute/bcast.h"\n'
                '#include "api/compute/tile_move_copy.h"\n'
                '#include "api/compute/sfpu_binary_bcast.h"')
            compute = compute.replace('mul_bcast_cols_init(0, 1);', 'sfpu_mul_bcast_col_init();')
            compute = compute.replace('mul_tiles_bcast_cols(0, 1, 0, 0, 0);',
                'reconfig_data_format_srca(0);\n    copy_tile_init(0);\n    copy_tile(0, 0, 0);\n'
                '    reconfig_data_format_srca(1);\n    copy_tile_init(1);\n    copy_tile(1, 0, 1);\n'
                '    sfpu_mul_bcast_col(0, 1);')
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
            if options.scratch_copy:
                from dspark_ladder_normalization import HELPER
                compute = compute[:compute.index('void kernel_main()')] + HELPER + '''
void kernel_main() {
    binary_op_init_common(0, 1, 16);
    for (uint32_t repeat = 0; repeat < get_compile_time_arg_val(1); ++repeat) {
        qwen_normalize_scratch<get_compile_time_arg_val(0)>(0, 1, 2, 16);
    }
}
'''
                buffers.append(ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(
                        buffer_index=2, data_format=ttnn.float32, page_size=4096)]))
                modes[1] = ttnn.UnpackToDestMode.Default
                modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
            config.unpack_to_dest_mode = modes
        kernels = [ttnn.KernelDescriptor(kernel_source=READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
            compile_time_args=[options.tiles, options.repeats],
            runtime_args=reader_args, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=compute,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=[options.tiles, options.repeats],
                runtime_args=reader_args,
                config=config),
            ttnn.KernelDescriptor(
                kernel_source='ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp',
                core_ranges=cores, runtime_args=writer_args,
                compile_time_args=[16] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                config=ttnn.WriterConfigDescriptor())]
        print(json.dumps({'stage': 'normalization_microprobe', 'tiles': options.tiles, 'repeats': options.repeats}), flush=True)
        ttnn.generic_op([shape_input, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=buffers, semaphores=[]))
        actual = ttnn.to_torch(output).float()
        print(json.dumps({'scope': 'isolated simulator primitive; not SDPA acceptance',
            'output_dtype': 'fp32' if options.fp32_output else 'bf16',
            'multiply': 'sfpu_column' if options.sfpu_column else 'sfpu_scalar' if options.sfpu_scalar else 'column_broadcast',
            'vary_rows': options.vary_rows,
            'scratch_copy': options.scratch_copy,
            'tiles': options.tiles, 'repeats': options.repeats,
            'numerator': -56, 'reciprocal': reciprocal,
            'fp32_product': -56 * reciprocal,
            'bf16_rne_product': torch.tensor(-56 * reciprocal).bfloat16().float().item(),
            'actual_unique_sample': actual.unique()[:8].tolist(),
            'actual_unique_count': actual.unique().numel(),
            'finite': bool(actual.isfinite().all()), 'elements': actual.numel()}), flush=True)
        if actual.numel() != 1024 * options.tiles * options.repeats or not bool(actual.isfinite().all()):
            raise AssertionError('Incomplete or non-finite normalization output')
        if options.sfpu_scalar or options.sfpu_column:
            numerator = (-56 - torch.arange(options.tiles * options.repeats) * 0.25).reshape(
                options.repeats, 1, 1, options.tiles).repeat_interleave(32, dim=-1)
            expected = (numerator * torch.tensor(reciprocals).reshape(1, 1, 32, 1)).expand_as(actual)
            if not options.fp32_output:
                expected = expected.bfloat16().float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        ttnn.close_device(device)


if __name__ == '__main__':
    main()
