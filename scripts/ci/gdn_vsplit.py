"""Opt-in, uncertified 96-worker recurrence and 24-worker native FNG prototype."""

import hashlib
import os
from pathlib import Path
import struct

import gdn_multitoken as native


HELPER_HASH = '6436703e10ac46485a95b6c371edcbc1cd5fbb98d6973d29c2ec6ceeea14c4b4'
LOOP = '    for (uint32_t it = 0; it < n_inst; ++it) {'
READER_LOOP = '    for (uint32_t token = 0; token < n_inst; ++token) {'
FNG_START = '            WAIT(cb_vread, Vt);\n            ew(cb_vread, cb_vread, cb_qsq, Vt, 2);'
FNG_END = '        } else {\n            // ---- o = qn @ new_h'
STATE_BASE = 'token * 384 + (bh_start / 4) * 16 + bh_start % 4'
DEFAULT_ROOT = Path('/opt/tt-metal')


def checked_section(source, start, end):
    native.replace_section(source, start, end, '')
    return source[source.index(start):source.index(end)]


def validate_helper():
    data = Path(native.__file__).read_text(encoding='utf-8').encode()
    if hashlib.sha256(data).hexdigest() != HELPER_HASH:
        raise ValueError('Pinned gdn_multitoken helper changed')


def validate_runtime(root):
    validate_helper()
    for relative, expected in native.HANDOFF_HASHES.items():
        if hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'CB handoff runtime changed: {relative}')


def state_page(token, worker, key_tile):
    if not 0 <= token < 32 or not 0 <= worker < 96 or not 0 <= key_tile < 4:
        raise ValueError('Expected token <32, worker <96, key_tile <4')
    return token * 384 + (worker // 4) * 16 + key_tile * 4 + worker % 4


def packed_pages(worker):
    if not 0 <= worker < 96:
        raise ValueError('Expected worker <96')
    return dict(q=4 * (worker // 12), k=32 + 4 * (worker // 12),
                v=64 + worker, scalar=worker // 4)


def bridge_page(token, head, partition):
    if not 0 <= token < 32 or not 0 <= head < 24 or not 0 <= partition < 4:
        raise ValueError('Invalid FP32 bridge coordinate')
    return token * 96 + head * 4 + partition


def output_element(token, head, column):
    if not 0 <= token < 32 or not 0 <= head < 24 or not 0 <= column < 128:
        raise ValueError('Invalid TILE output coordinate')
    return (head * 4 + column // 32,
            ((token // 16) * 2 + (column % 32) // 16) * 256
            + (token % 16) * 16 + column % 16)


def cb_plan(stage):
    if stage == 'recurrence':
        io = {0: 4, 1: 4, 2: 1, 3: 1, 4: 1, 5: 4, 18: 4, 27: 1, 30: 4}
        fp32 = {6: 1, 7: 4, 8: 4, 9: 1, 10: 4, 11: 4, 12: 4, 13: 1,
                14: 4, 15: 1, 16: 2, 17: 4, 19: 2, 20: 4, 21: 4, 22: 1,
                23: 1, 24: 1, 25: 4, 26: 4, 28: 1, 29: 1}
    elif stage == 'norm_gate':
        io = {2: 8, 19: 8, 27: 4, 30: 8}
        fp32 = {5: 1, 6: 1, 7: 4, 9: 1, 10: 4, 11: 4, 15: 4, 16: 8,
                22: 4, 28: 1, 31: 4}
    else:
        raise ValueError('Unknown stage')
    return io, fp32


def split_reader(source):
    for scalar in ('beta_acc, cb_beta', 'g_acc, cb_g'):
        source = native.replace_once(source,
            f'gather_scalar({scalar}, (b / 32) * NVT + h / 32, r, h % 32);',
            f'gather_scalar({scalar}, (b / 32) * NVT + (h / 4) / 32, r, (h / 4) % 32);')
    source = native.replace_once(source, 'const uint32_t base_page = bh * kv;',
                                 f'const uint32_t base_page = {STATE_BASE};')
    return native.replace_once(source,
        'noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});',
        'noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t * 4}, {.offset_bytes = t * tb_io});')


def split_writer(source):
    source = native.replace_once(source, 'const auto s_acc = TensorAccessor(s_a, s_addr, tb_io);',
        'const uint32_t tb_state = get_tile_size(cb_sout);\n'
        '    const auto s_acc = TensorAccessor(s_a, s_addr, tb_state);')
    source = native.replace_once(source, 'const uint32_t base_page = bh * kv;',
                                 f'const uint32_t base_page = {STATE_BASE};')
    source = native.replace_once(source,
        'noc.async_write(src, s_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = base_page + t});',
        'noc.async_write(src, s_acc, tb_state, {.offset_bytes = t * tb_state}, {.page_id = base_page + t * 4});')
    return native.replace_once(source,
        '            noc.async_write(CoreLocalMem<uint32_t>(stage), o_acc, o_page, {}, {.page_id = bh});',
        '            scb.push_back(1);\n            scb.wait_front(1);\n'
        '            noc.async_write(CoreLocalMem<uint32_t>(stage), o_acc, o_page, {}, {.page_id = bh});')


NORM_READER_LOOP = '''    const auto pre_acc = TensorAccessor(q_a, q_addr, 128);
    constexpr uint32_t cb_pre = 15, cb_stick = 5;
    for (uint32_t token = 0; token < n_inst; ++token) {
        CircularBuffer pre(cb_pre);
        pre.reserve_back(4);
        const uint32_t destination = pre.get_write_ptr();
        zero(destination, 4 * 4096 / 4);
        for (uint32_t partition = 0; partition < 4; ++partition) {
            CircularBuffer stick(cb_stick);
            stick.reserve_back(1);
            noc.async_read(pre_acc, stick, 128,
                {.page_id = token * 96 + bh_start * 4 + partition}, {.offset_bytes = 0});
            noc.async_read_barrier();
            stick.push_back(1);
            stick.wait_front(1);
            const uint32_t src = stick.get_read_ptr();
            const uint32_t dst = destination + partition * 4096;
            asm volatile("" ::: "memory");
            auto input = CoreLocalMem<volatile uint32_t>(src);
            auto output = CoreLocalMem<volatile uint32_t>(dst);
            for (uint32_t word = 0; word < 16; ++word) {
                output[word] = input[word];
                output[256 + word] = input[16 + word];
            }
            asm volatile("" ::: "memory");
            stick.pop_front(1);
        }
        pre.push_back(4);
        gather_row(z_acc, cb_v, Vt, (token / 32) * WTZ + ZOT + bh_start * Vt, token % 32);
    }
}
'''


def norm_reader(source):
    if source.count(READER_LOOP) != 1 or not source.endswith('    }\n}\n'):
        raise ValueError('Native reader loop/tail changed')
    source = native.replace_once(source, 'get_tile_size(cb_q)', 'get_tile_size(cb_v)')
    return source[:source.index(READER_LOOP)] + NORM_READER_LOOP


def norm_compute(source):
    if source.count(LOOP) != 1:
        raise ValueError('Native compute loop changed')
    block = checked_section(source, FNG_START, FNG_END)
    prefix = source[:source.index(LOOP)]
    prefix = native.replace_once(prefix, 'compute_kernel_hw_startup(cb_q, cb_k, cb_out);',
                                 'compute_kernel_hw_startup(cb_vread, cb_vread, cb_out);')
    return prefix + LOOP + '\n        WAIT(cb_ones, 1);\n' + block + '    }\n}\n'


def norm_writer(source):
    source = native.local_norm_writer(source)
    source = native.replace_section(source, '        } else {\n        // o: stage',
                                     '        // new state:', '        }\n\n')
    return native.replace_section(source, '        // new state:',
                                  '    }  // per-instance loop', '')


def load_kernels(root=DEFAULT_ROOT):
    """Generate in memory only; native kernel and helper drift is fatal."""
    validate_helper()
    kernels = native.load_kernels(Path(root), False)
    return {
        'recurrence': dict(reader=split_reader(kernels['reader']),
                           writer=split_writer(kernels['writer']), compute=kernels['compute']),
        'norm_gate': dict(reader=norm_reader(kernels['reader']),
                          writer=norm_writer(kernels['writer']), compute=norm_compute(kernels['compute'])),
    }


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def stage_spec(stage, rows):
    if rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Expected T in 1,2,4,8,16,32')
    if stage == 'recurrence':
        return dict(workers=96,
            compute=[4, 1, 1, bits(1e-6), bits(128 ** -0.5), 0, 0, 0],
            reader=[4, 1, 1, bits(1e-6), bits(128 ** -0.5), 96, 1, 160, 0, 32, 64, 12, 1, 0, 0, 0],
            writer=[4, 1, 0, 1, 96, rows],
            reader_accessors=[0, 0, 0, 1, 2, 3, 3, 3], writer_accessors=[4, 5],
            reader_addresses=[0, 0, 0, 1, 2, 3], writer_addresses=[4, 5], output_page=128)
    if stage == 'norm_gate':
        return dict(workers=24,
            compute=[4, 4, 0, bits(1e-6), bits(128 ** -0.5), 1, bits(128e-6), bits(128 ** 0.5)],
            reader=[4, 4, 0, bits(1e-6), bits(128 ** -0.5), 24, 1, 96, 0, 0, 0, 3, 1, 1, 96, 0],
            writer=[4, 4, 1, 1, 24, rows],
            reader_accessors=[4, 6, 6, 6, 6, 6, 6, 7], writer_accessors=[8, 8],
            reader_addresses=[4, 6, 6, 6, 6, 6, 6, 7], writer_addresses=[8, 8], output_page=2048)
    raise ValueError('Unknown stage')


def core_coordinates(horizontal, vertical, count):
    if horizontal < 1 or vertical < 1 or horizontal * vertical < count:
        raise ValueError(f'At least {count} worker cores required')
    return [(worker // vertical, worker % vertical) for worker in range(count)]


def runtime_args(spec, role, worker, rows, addresses):
    if role == 'reader':
        return [worker, rows] + [addresses[index] for index in spec['reader_addresses']]
    if role == 'writer':
        output, state = [addresses[index] for index in spec['writer_addresses']]
        return [worker, rows, output, spec['output_page'], state]
    if role == 'compute':
        return [rows]
    raise ValueError('Unknown kernel role')


def build_program(ttnn, mesh, shards, kernels, stage, rows):
    spec = stage_spec(stage, rows)
    grid = mesh.compute_with_storage_grid_size()
    coordinates = core_coordinates(grid.x, grid.y, spec['workers'])
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*point), ttnn.CoreCoord(*point))
                               for point in coordinates])
    buffers = []
    io, fp32 = cb_plan(stage)
    for counts, dtype, page in ((io, ttnn.bfloat16, 2048), (fp32, ttnn.float32, 4096)):
        for index, count in counts.items():
            buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))]))
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [value[chip] for value in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Prototype inputs and outputs must not alias')
        descriptors = []
        configs = dict(
            reader=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                     noc=ttnn.NOC.RISCV_1_default),
            writer=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default),
            compute=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, math_approx_mode=False))
        for role in ('reader', 'writer', 'compute'):
            args = list(spec[role])
            for index in spec.get(role + '_accessors', []):
                args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
            runtime = ttnn.RuntimeArgs()
            for worker, (horizontal, vertical) in enumerate(coordinates):
                runtime[horizontal][vertical] = runtime_args(spec, role, worker, rows, addresses)
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[stage][role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=configs[role])
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    return program


def execute(mesh, qkv, beta, gate, initial, *, z, norm_w, root=DEFAULT_ROOT, experimental=False):
    """Return (BF16 gated output, BF16 prefix states, FP32 pre-norm bridge).

    Two fenced programs on an existing 1x2 mesh. No input state is modified.
    Not a trace/performance harness and not evidence of device correctness.
    """
    if not experimental:
        raise ValueError('Uncertified prototype requires experimental=True')
    import ttnn

    validate_runtime(Path(os.environ.get('TT_METAL_HOME', str(DEFAULT_ROOT))))
    kernels = load_kernels(root)
    inputs = [qkv, beta, gate, initial, z, norm_w]
    rows = native.validate_geometry(*(tuple(value.shape) for value in inputs[:4]))
    if tuple(z.shape) != (1, rows, 3072) or tuple(norm_w.shape) != (1, 1, 128):
        raise ValueError('Expected z [1,T,3072] and norm_w [1,1,128]')
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in inputs):
        raise ValueError('Expected interleaved DRAM BF16 TILE inputs')
    if tuple(mesh.shape) != (1, 2):
        raise ValueError('Expected a 1x2 mesh')
    grid = mesh.compute_with_storage_grid_size()
    core_coordinates(grid.x, grid.y, 96)
    if any(len(ttnn.get_device_tensors(value)) != 2 for value in inputs):
        raise ValueError('Both chips required in a 1x2 mesh')
    pre_norm = ttnn.empty((rows, 1, 96, 32), device=mesh, dtype=ttnn.float32,
                          layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    states = ttnn.empty((rows, 24, 128, 128), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.empty((1, rows, 3072), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tensors = inputs[:4] + [pre_norm, states] + inputs[4:] + [output]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(value) != 2 for value in shards):
        raise ValueError('Both chips required for every prototype tensor')
    programs = [build_program(ttnn, mesh, shards, kernels, stage, rows)
                for stage in ('recurrence', 'norm_gate')]
    for program in programs:
        ttnn.generic_op(tensors, program)
        ttnn.synchronize_device(mesh)
    return output, states, pre_norm


def audit(root):
    kernels = load_kernels(root)
    stages = {}
    for stage, sources in kernels.items():
        io, fp32 = cb_plan(stage)
        cb_bytes = sum(io.values()) * 2048 + sum(fp32.values()) * 4096
        stages[stage] = dict(workers=stage_spec(stage, 32)['workers'], cb_bytes=cb_bytes,
            static_end_estimate_with_111488_reserved=cb_bytes + 111488,
            bf16_cb_tiles=io, fp32_cb_tiles=fp32,
            generated_sha256={role: hashlib.sha256(source.encode()).hexdigest()
                              for role, source in sources.items()})
    return dict(status='host-source-audit-only; no compilation, simulator or hardware certification',
                helper_sha256=HELPER_HASH, native_sha256=native.HASHES,
                runtime_sha256_required_at_execute=native.HANDOFF_HASHES,
                runtime_headers_checked=False, stages=stages)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only host source/CB audit; never opens a device')
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
