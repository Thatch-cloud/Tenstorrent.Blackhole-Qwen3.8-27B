"""attention_fold_dma's tile permutation of bounded query groups, at any served width.

attention_fold_dma.py / .cpp are frozen-recipe evidence (target_t16_attention_gate.SOURCES) and carry the pair's fold
as literals: 12 query-head rows per token (two KV heads of six), a (12, 256) native query, two chips. At four cards a
chip holds six query heads on ONE KV head, so the fold is a token-major repack of rows * 6 head rows and the query is
(T, 6, 256). This twin is the same launch with those numbers from tp_shapes, the chip loop over the mesh's devices, and
the sibling kernel attention_fold_dma_tp.cpp, whose head-row count is a define (tp_kernels.fold_defines). At the pair
it selects the pinned kernel file and no defines, so a pair launch is what the pinned module builds.
"""

from pathlib import Path

import tp_kernels
import tp_shapes


def source_row(rows, output_index, *, inverse=False):
    """The source row (in a token's head rows for the inverse, in the token-major query for the forward) that
    output row `output_index` reads: attention_fold_dma.source_row at this width."""
    found = tp_shapes.active()
    head_rows = found.attn_fold_rows
    if (type(inverse) is not bool or type(rows) is not int or not 1 <= rows <= 8 or type(output_index) is not int
            or not 0 <= output_index < rows * head_rows):
        raise ValueError('One to eight complete query rows required')
    group = found.attn_group
    if inverse:
        token, head = divmod(output_index, head_rows)
        return (head // group) * rows * group + token * group + head % group
    key_head, remainder = divmod(output_index, rows * group)
    token, head = divmod(remainder, group)
    return token * head_rows + key_head * group + head


def device_layout_dma(mesh, source, rows, owned, *, inverse=False, offset=0):
    import ttnn

    found = tp_shapes.active()
    head_rows = found.attn_fold_rows
    if type(rows) is not int or not 1 <= rows <= 8 or type(inverse) is not bool or type(offset) is not int or offset < 0:
        raise ValueError('Bounded explicit tile permutation required')
    expected = (1, 1, rows * head_rows, 256) if inverse else None
    shape = tuple(source.shape)
    if (inverse and (shape != expected or offset)) or (not inverse and
            (len(shape) != 4 or shape[0] != 1 or shape[2:] != (head_rows, 256) or offset + rows > shape[1])):
        raise ValueError('Native tiled TP%d query geometry required' % found.tp)
    if source.dtype != ttnn.bfloat16 or source.layout != ttnn.TILE_LAYOUT or source.memory_config() not in (
            ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        raise ValueError('Interleaved BF16 tile input required')
    output_shape = (1, rows, head_rows, 256) if inverse else (1, 1, rows * head_rows, 256)
    tiles = rows * 8 if inverse else ((rows * head_rows + 31) // 32) * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tiles // 8:
        raise ValueError('Direct permutation requires the bounded 8-column worker grid')
    output = ttnn.empty(output_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    source_shards, output_shards = ttnn.get_device_tensors(source), ttnn.get_device_tensors(output)
    chips = tp_shapes.chip_count()
    if len(source_shards) != chips or len(output_shards) != chips:
        raise ValueError('%s chips required' % tp_shapes.all_chips())
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tiles // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=(max(4, rows) + 1) * 2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local_source, local_output = source_shards[chip], output_shards[chip]
        descriptor = ttnn.KernelDescriptor(
            kernel_source=tp_kernels.source(Path(__file__).with_name('attention_fold_dma.cpp')), core_ranges=cores,
            defines=tp_kernels.fold_defines(),
            compile_time_args=[argument for tensor in (local_source, local_output)
                for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker in range(tiles):
            runtime[worker % 8][worker // 8] = [local_source.buffer_address(), local_output.buffer_address(),
                rows, int(inverse), offset, worker]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    ttnn.generic_op([source, output], program)
    return output
