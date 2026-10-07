"""gdn_user_batch at four cards: one launch for the packed users' GDN recurrence and fused norm/gate, 12 value heads a chip.

gdn_user_batch.py is a pinned reference: the K5-A card-B qualification compares against the served launch it builds
(test_gdn_seq_block.ShippingTests holds it, with gdn_records, gdn_commit_dma and gdn_device_loop_state, unedited since
the K5 parent), and it carries the pair's geometry as literals (24 heads, 160 conv pages, a 1x2 mesh, 3,072-wide z). So
the four-card launch is this sibling: the same builder, the same kernels (gdn_multitoken.load_kernels(root, True)
verbatim, hash-checked), the same descriptors, with every literal drawn from tp_shapes for the width the process serves
at. gdn_seq_block picks between the two per call (its `batch` proxy); at the pair nothing here runs.

Four users x 12 heads = 48 worker cores of the 11 x 10 grid: one wave, as the pair's 4 x 24 = 96 are (the chain does not
get faster, docs/tp4 S2T-O1 is the plan that uses the freed cores). Uncertified: host construction only. The kernels take
their geometry as compile-time arguments (H, Ct and the offsets), so no kernel source changes, but nothing here has run at
H = 12 on a device; gdn_user_batch_device_test with a geometry argument is the spike (S2T-01).
"""

import hashlib
import os
import sys
from pathlib import Path

import gdn_multitoken as native
import gdn_user_batch as pair
import tp_shapes
import verify_trace_t1

# What the pair's module owns and this one shares unchanged: flags, kernel loading, the float packing, the descriptor
# coalescing (geometry free), the placement helpers' constants.
DEFAULT_ROOT = pair.DEFAULT_ROOT
MAX_USERS = pair.MAX_USERS
FLAG = pair.FLAG
MIN_USERS_FLAG = pair.MIN_USERS_FLAG
enabled = pair.enabled
min_users = pair.min_users
load_kernels = pair.load_kernels
bits = pair.bits
accessor_layout = pair.accessor_layout
coalesced_descriptors = pair.coalesced_descriptors
ACCESSORS = pair.ACCESSORS


def geometry():
    """tp_shapes' per-chip row for the width this process serves at."""
    return tp_shapes.geometry(tp_shapes.chip_count())


def heads():
    """GDN value heads per chip: 12 at four cards (24 at the pair, which this module does not serve)."""
    return geometry().gdn_nv


def core_shares(horizontal, vertical, users, workers=None):
    """One disjoint contiguous share of `workers` (default: the width's head count) cores per user, worker w at
    (w // vertical, w % vertical) exactly as gdn_user_batch.core_shares places them."""
    if workers is None:
        workers = heads()
    if type(users) is not int or not 1 <= users <= MAX_USERS:
        raise ValueError('One to %d packed users per batched GDN launch' % MAX_USERS)
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive worker count per user required')
    total = users * workers
    if horizontal < 1 or vertical < 1 or horizontal * vertical < total:
        raise ValueError('At least %d worker cores required for %d packed users' % (total, users))
    points = [(worker // vertical, worker % vertical) for worker in range(total)]
    if any(horizontal <= point[0] for point in points):
        raise ValueError('At least %d worker cores required for %d packed users' % (total, users))
    return [points[index * workers:(index + 1) * workers] for index in range(users)]


def compile_args(role, rows):
    """gdn_user_batch.compile_args with the width's head count, conv pages and tile offsets: at the pair
    [4, 4, 1, b, b, 24, 1, 160, 0, 32, 64, 3, 1, 1, 96, 0]; at four cards H 12, Ct 80, offsets 16 / 32, z 48 tiles."""
    found = geometry()
    key_tiles = found.gdn_key // tp_shapes.TILE
    if role == 'reader':
        return [4, 4, 1, bits(1e-6), bits(128 ** -0.5), found.gdn_nv, 1, found.gdn_conv_pages, 0, key_tiles,
                2 * key_tiles, 3, 1, 1, found.gdn_value // tp_shapes.TILE, 0]
    if role == 'writer':
        return [4, 4, 1, 1, found.gdn_nv, rows]
    if role == 'compute':
        return [4, 4, 1, bits(1e-6), bits(128 ** -0.5), 1, bits(128e-6), bits(128 ** 0.5)]
    raise ValueError('Unknown kernel role')


def runtime_args(role, head, rows, addresses):
    """gdn_user_batch.runtime_args with the head bound of the width."""
    if type(head) is not int or not 0 <= head < heads():
        raise ValueError('Head index within the %d-head TP%d shard required' % (heads(), tp_shapes.chip_count()))
    if len(addresses) != 8:
        raise ValueError('Eight per-user buffer addresses required')
    if role == 'reader':
        return [head, rows, addresses[0], addresses[0], addresses[0],
                addresses[1], addresses[2], addresses[3], addresses[6], addresses[7]]
    if role == 'writer':
        return [head, rows, addresses[4], 2048, addresses[5]]
    if role == 'compute':
        return [rows]
    raise ValueError('Unknown kernel role')


def validate_geometry(qkv, beta, gate, initial):
    """gdn_multitoken.validate_geometry against the width's per-chip shapes: qkv [1, T, 2560], beta and gate
    [1, T, 12], one 12-head initial state."""
    found = geometry()
    rows = qkv[1] if len(qkv) == 3 else 0
    if rows not in (1, 2, 4, 8, 16, 32) or tuple(qkv) != (1, rows, found.gdn_qkv):
        raise ValueError('Expected packed TP%d QKV [1,T,%d]' % (found.tp, found.gdn_qkv))
    if (tuple(beta) != (1, rows, found.gdn_nv) or tuple(gate) != (1, rows, found.gdn_nv)
            or tuple(initial) != (1, found.gdn_nv, 128, 128)):
        raise ValueError('Expected T sequential rows sharing one %d-head initial state' % found.gdn_nv)
    return rows


def validate_users(shapes):
    """Per user, `(qkv, beta, gate, initial, z, norm_w)` shapes; returns each user's rows (gdn_user_batch.validate_users
    at the width's shapes: z is [1, T, 1536] at four cards)."""
    if not 1 <= len(shapes) <= MAX_USERS:
        raise ValueError('One to %d packed users per batched GDN launch' % MAX_USERS)
    widths = []
    for user in shapes:
        if len(user) != 6:
            raise ValueError('Six input shapes per packed user required')
        rows = validate_geometry(*user[:4])
        if tuple(user[4]) != (1, rows, geometry().gdn_value) or tuple(user[5]) != (1, 1, 128):
            raise ValueError('Expected z [1,T,%d] and norm_w [1,1,128] per packed user' % geometry().gdn_value)
        widths.append(rows)
    return widths


def mesh_chips(mesh):
    """The serving mesh is 1 x tp; one chip is allowed so a single-card rig can run the equality test."""
    shape = tuple(mesh.shape)
    chips = tp_shapes.chip_count()
    if len(shape) != 2 or shape[0] != 1 or shape[1] not in (1, chips):
        raise ValueError('Expected a 1x%d mesh, or a 1x1 mesh on a single-card rig' % chips)
    return shape[1]


def build_program(operations, mesh, user_shards, kernels, widths):
    """gdn_user_batch.build_program, statement for statement, over this module's width-aware helpers: one mesh program
    holding every user's reader/writer/compute on its own cores.

    `user_shards[u]` is that user's eight per-chip tensor lists, in the order
    `[qkv, beta, gate, initial, output, states, z, norm_w]`."""
    if len(user_shards) != len(widths):
        raise ValueError('One token width per packed user required')
    chips = mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    shares = core_shares(grid.x, grid.y, len(widths))
    coalesce = verify_trace_t1.cut('coalesce')
    if coalesce:
        ranges = [verify_trace_t1.rectangle_set(operations, share) for share in shares]
        union = verify_trace_t1.rectangle_set(operations, [point for share in shares for point in share])
    else:
        ranges = [operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*point),
                                                                operations.CoreCoord(*point))
                                           for point in share]) for share in shares]
        union = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*point),
                                                              operations.CoreCoord(*point))
                                         for share in shares for point in share])
    buffers = []
    io, fp32 = native.cb_plan(True)
    for counts, dtype, page in ((io, operations.bfloat16, 2048), (fp32, operations.float32, 4096)):
        for index, count in counts.items():
            buffers.append(operations.CBDescriptor(total_size=count * page, core_ranges=union,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    configs = dict(
        reader=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1,
                                                       noc=operations.NOC.RISCV_1_default),
        writer=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                       noc=operations.NOC.RISCV_0_default),
        compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
                                                   fp32_dest_acc_en=True, math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    coalesced = []
    for chip in range(chips):
        descriptors = []
        private, weights = [], []
        planned = []
        for shards, rows, share, cores in zip(user_shards, widths, shares, ranges, strict=True):
            local = [value[chip] for value in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('One user inputs, initial state and prefix outputs must not alias')
            if any(address in private for address in addresses[:7]):
                raise ValueError('Packed users must not share any buffer but the norm weight')
            private.extend(addresses[:7])
            weights.append(addresses[7])
            for role in ('reader', 'writer', 'compute'):
                args = list(compile_args(role, rows))
                for index in ACCESSORS[role]:
                    args.extend(operations.TensorAccessorArgs(local[index]).get_compile_time_args())
                if coalesce:
                    planned.append((role, args, cores, [(point, runtime_args(role, head, rows, addresses))
                                                        for head, point in enumerate(share)]))
                    continue
                runtime = operations.RuntimeArgs()
                for head, (horizontal, vertical) in enumerate(share):
                    runtime[horizontal][vertical] = runtime_args(role, head, rows, addresses)
                descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=configs[role])
                descriptor.runtime_args = runtime
                descriptors.append(descriptor)
        if coalesce:
            descriptors, merged = coalesced_descriptors(operations, kernels, configs, planned, union)
            coalesced.append(merged)
        if any(weight in private for weight in weights):
            raise ValueError('The shared norm weight must not alias any packed user buffer')
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    if coalesce:
        verify_trace_t1.note('coalesced' if all(coalesced) else 'coalesce_fallback')
    return program


def execute(mesh, users, kernels, operations=None, *, output_memory=None):
    """gdn_user_batch.execute at the width's shapes: every packed user's recurrence and fused norm/gate in ONE launch;
    returns `[(output, states), ...]`, each output `(1, T, 1536)` and each state block `(T, 12, 128, 128)` at four
    cards."""
    if operations is None:
        import ttnn as operations

    groups = [tuple(user) for user in users]
    widths = validate_users([[tuple(value.shape) for value in user] for user in groups])
    native.validate_handoff_runtime(Path(os.environ.get('TT_METAL_HOME', str(DEFAULT_ROOT))))
    if output_memory is None:
        output_memory = operations.L1_MEMORY_CONFIG
    if output_memory not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG):
        raise ValueError('Output memory must be interleaved DRAM or L1')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT or
           value.memory_config() != operations.DRAM_MEMORY_CONFIG for user in groups for value in user):
        raise ValueError('Batched GDN requires interleaved DRAM BF16 TILE inputs')
    chips = mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    core_shares(grid.x, grid.y, len(groups))

    # QWEN_FAST_GDN_SHARED_HISTORY (gdn_shared_history, gate profiles only): inside a packed block's verify capture the states
    # are the pool's one history set, not a private allocation. sys.modules, not an import: nothing loads the module unless the
    # flag is on, and `active()` is None outside a capture.
    shared_module = sys.modules.get('gdn_shared_history')
    shared = shared_module.active() if shared_module is not None else None
    produced, outputs, histories = [], [], []
    try:
        for user, rows in zip(groups, widths, strict=True):
            output = operations.empty((1, rows, geometry().gdn_value), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=output_memory)
            produced.append(output)
            outputs.append(output)
            if shared is not None:
                # Owned by the pool: never freed here, not even when this launch fails.
                states = shared.take(operations, mesh, rows, heads())
            else:
                states = operations.empty((rows, heads(), 128, 128), device=mesh, dtype=operations.bfloat16,
                                          layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG)
                produced.append(states)
            histories.append(states)
        tensor_groups = [[*user[:4], outputs[index], histories[index], *user[4:]] for index, user in enumerate(groups)]
        flat = []
        for group in tensor_groups:
            for value in group:
                if not any(value is kept for kept in flat):
                    flat.append(value)
        shards = {id(value): operations.get_device_tensors(value) for value in flat}
        if any(len(shards[id(value)]) != chips for value in flat):
            raise ValueError('Every chip of the mesh required for every batched GDN tensor')
        program = build_program(operations, mesh, [[shards[id(value)] for value in group]
                                                   for group in tensor_groups], kernels, widths)
        operations.generic_op(flat, program)
        return list(zip(outputs, histories))
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise


def audit(root=DEFAULT_ROOT, users=MAX_USERS, rows=16):
    """Host-only description of what one four-card batched launch would build (gdn_user_batch.audit's)."""
    kernels = load_kernels(root)
    io, fp32 = native.cb_plan(True)
    shares = core_shares(11, 10, users)
    return dict(status='host-source-and-placement audit only; no compilation or hardware certification',
                transforms='none; fused sources are gdn_multitoken.load_kernels(root, True) verbatim',
                native_sha256=native.HASHES, handoff_sha256=native.HANDOFF_HASHES,
                users=users, workers_per_user=heads(), workers=users * heads(),
                core_shares=shares, rows_per_user=rows, tp=tp_shapes.chip_count(),
                launches_per_layer=dict(per_user_recurrence_and_norm=2 * users, batched=1),
                cb_bytes_per_worker=sum(io.values()) * 2048 + sum(fp32.values()) * 4096,
                generated_sha256={role: hashlib.sha256(source.encode()).hexdigest()
                                  for role, source in kernels.items()})
