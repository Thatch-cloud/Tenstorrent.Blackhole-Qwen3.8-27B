"""One launch for the packed users' GDN recurrence and fused norm/gate.

Four 16-row segments of a packed decode round are four independent recurrences: each
has its own carried recurrent state, its own convolution history and its own rows. Today
each one is its own pair of launches (`gdn_vsplit`'s 96-worker recurrence stage and its
24-worker norm_gate stage), so a layer costs eight launches and the round 192 + 192 of
them - 47.4 ms of the 172 ms verify trace.

The kernels here are `gdn_multitoken.load_kernels(root, True)` VERBATIM: the fused
24-core source that runs one whole head per core and folds norm/gate into the recurrence.
Nothing is transformed, so `gdn_multitoken.HASHES` and `gdn_vsplit.HELPER_HASH` are
untouched and no new pinned-source surface appears. The batching is entirely in how the
program is built: four users, four DISJOINT 24-core shares of the 11x10 grid, four
kernel-descriptor triples, one `MeshProgramDescriptor`, one `generic_op`.

Because each user keeps its own tensors, every downstream shape is what the per-user path
already produces - `(T, 24, 128, 128)` prefix states, `(1, T, 3072)` gated output - so
`restore_prefix`, `gdn_conv_prefix_copy`, `gdn_commit_dma` and `gdn_records` need no
change at all. See docs/gdn-user-batch-launches.md for why the token-axis alternative
(one T=64 recurrence) is blocked by those same pinned files.

Uncertified: host construction only. Nothing here has run on a device; the equality and
timing evidence is `gdn_user_batch_device_test.py`.
"""

import hashlib
import os
from pathlib import Path
import struct

import gdn_multitoken as native
import verify_trace_t1


DEFAULT_ROOT = Path('/opt/tt-metal')
HEADS = 24
MAX_USERS = 4
FLAG = 'QWEN_FAST_GDN_USER_BATCH'
MIN_USERS_FLAG = 'QWEN_FAST_GDN_USER_BATCH_MIN_USERS'


def enabled(environ=None):
    """The opt-in flag. Default OFF; anything but '0' or '1' is a configuration error
    rather than a silent fallback, because a typo here silently costs 29 ms a round."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % FLAG)
    return value == '1'


def min_users(environ=None):
    """How many packed users a block must carry before the batched launch engages.

    Default 1: the flag alone decides, which is the behaviour the flag shipped with.
    Above `MAX_USERS` the batched path can never engage, which is the clean off switch
    for a bisect that keeps every other part of the arm identical.

    Two notes for whoever sets this. The packed serving block always carries
    `shape.users` segments - placeholder users fill the slots no live request holds
    (packed_verifier.py:415-421) - so in serving the threshold is a straight on/off at 5,
    not a live-user count. And on the measured single-user numbers the fused 24-core
    launch is SLOWER than the per-user value split it replaces, so 2 is the throughput
    optimum even though 1 is the compatible default.
    """
    value = (os.environ if environ is None else environ).get(MIN_USERS_FLAG, '1')
    if type(value) is not str or not value.isdigit() or value != str(int(value)):
        raise ValueError('%s must be a non-negative decimal integer' % MIN_USERS_FLAG)
    return int(value)


def load_kernels(root=DEFAULT_ROOT):
    """The fused sources, unchanged. `native.load_kernels` re-checks every pinned hash."""
    return native.load_kernels(Path(root), True)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def core_shares(horizontal, vertical, users, workers=HEADS):
    """One disjoint contiguous share of `workers` cores per user.

    Worker `w` sits at `(w // vertical, w % vertical)`, which is exactly the placement
    `gdn_multitoken.execute` uses for its 24 heads, so user zero lands on the same
    physical cores the single-user launch uses and a one-user batch is placement
    identical to it.
    """
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
    """Exactly `gdn_multitoken.execute`'s fused (`fuse_norm_gate=True`) compile arguments.

    Kept as one function so the unit tests can hold it against the native builder rather
    than against a copy of the same literals.
    """
    if role == 'reader':
        args = [4, 4, 1, bits(1e-6), bits(128 ** -0.5), HEADS, 1, 160, 0, 32, 64, 3, 1, 0, 0, 0]
        args[13:16] = [1, 96, 0]
        return args
    if role == 'writer':
        return [4, 4, 1, 1, HEADS, rows]
    if role == 'compute':
        return [4, 4, 1, bits(1e-6), bits(128 ** -0.5), 1, bits(128e-6), bits(128 ** 0.5)]
    raise ValueError('Unknown kernel role')


ACCESSORS = dict(reader=(0, 0, 0, 1, 2, 3, 6, 7), writer=(4, 5), compute=())


def runtime_args(role, head, rows, addresses):
    """One core's runtime arguments, from that user's eight buffer addresses in
    `[qkv, beta, gate, initial, output, states, z, norm_w]` order."""
    if type(head) is not int or not 0 <= head < HEADS:
        raise ValueError('Head index within the 24-head TP2 shard required')
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


def validate_users(shapes):
    """Per user, `(qkv, beta, gate, initial, z, norm_w)` shapes; returns each user's rows.

    Users may carry different widths - a ragged packed block is legal, every descriptor
    carries its own `rows` - but every user must be one of the supported token widths.
    """
    if not 1 <= len(shapes) <= MAX_USERS:
        raise ValueError('One to %d packed users per batched GDN launch' % MAX_USERS)
    widths = []
    for user in shapes:
        if len(user) != 6:
            raise ValueError('Six input shapes per packed user required')
        rows = native.validate_geometry(*user[:4])
        if tuple(user[4]) != (1, rows, 3072) or tuple(user[5]) != (1, 1, 128):
            raise ValueError('Expected z [1,T,3072] and norm_w [1,1,128] per packed user')
        widths.append(rows)
    return widths


def accessor_layout(operations, tensors):
    return [operations.TensorAccessorArgs(value).get_compile_time_args() for value in tensors]


def mesh_chips(mesh):
    """The serving mesh is 1x2. One chip is allowed so that a single-card rig can run the
    equality test (`gdn_user_batch_device_test.py`); nothing else takes that path."""
    shape = tuple(mesh.shape)
    if len(shape) != 2 or shape[0] != 1 or shape[1] not in (1, 2):
        raise ValueError('Expected a 1x2 mesh, or a 1x1 mesh on a single-card rig')
    return shape[1]


def coalesced_descriptors(operations, kernels, configs, planned, union):
    """QWEN_FAST_VERIFY_T1 (#12): the planned per-user descriptors as few as possible.

    `planned` is `(role, compile_args, user_ranges, [(core, runtime_args), ...])` per user and
    role, in build order. A role whose users all compile identically - equal row counts, and
    the interleaved accessors carry no address - becomes ONE descriptor over `union` holding
    every user's per-core runtime arguments; otherwise each user keeps its own descriptor on
    its own (rectangle) ranges. Every core runs the same kernel with the same compile-time
    and runtime arguments either way; only the descriptor and range counts change."""
    roles = ('reader', 'writer', 'compute')
    by_role = {role: [entry for entry in planned if entry[0] == role] for role in roles}
    merged = all(len(entries) > 0 and all(entry[1] == entries[0][1] for entry in entries)
                 for entries in by_role.values())
    if merged:
        parts = [(role, by_role[role][0][1], union, [core for entry in by_role[role] for core in entry[3]])
                 for role in roles]
    else:
        # Never one role merged and another not: a mixed program would still be exact, but
        # the per-user order is what the flag-off build makes, so keep it whole.
        parts = [(role, args, cores, runtime) for role, args, cores, runtime in planned]
    descriptors = []
    for role, args, cores, runtime_by_core in parts:
        runtime = operations.RuntimeArgs()
        for (horizontal, vertical), values in runtime_by_core:
            runtime[horizontal][vertical] = values
        descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
            source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
            compile_time_args=args, config=configs[role])
        descriptor.runtime_args = runtime
        descriptors.append(descriptor)
    return descriptors, merged


def build_program(operations, mesh, user_shards, kernels, widths):
    """One mesh program holding every user's reader/writer/compute on its own cores.

    `user_shards[u]` is that user's eight per-chip tensor lists, in the order
    `[qkv, beta, gate, initial, output, states, z, norm_w]`.
    """
    if len(user_shards) != len(widths):
        raise ValueError('One token width per packed user required')
    chips = mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    shares = core_shares(grid.x, grid.y, len(widths))
    # QWEN_FAST_VERIFY_T1 (#12, verify_trace_t1): rectangle core ranges, and one descriptor
    # per role over every user's cores (coalesced_descriptors); per core, the same kernel,
    # compile-time and runtime arguments as the 24 single-core ranges per user below.
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
            # Only the shared normalization weight may be held by two users at once; every
            # other buffer is that user's alone, and two users sharing one would silently
            # interleave their recurrences or overwrite each other's prefix states.
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
    """Run every packed user's recurrence and fused norm/gate in ONE launch.

    `users` is a sequence of `(qkv, beta, gate, initial, z, norm_w)` tensor tuples, one
    per packed user, exactly the arguments the per-user `gdn_multitoken.execute` takes.
    Returns `[(output, states), ...]` in user order, each the same shape, dtype, layout
    and placement the per-user call returns, so the caller cannot tell the difference
    except by counting launches.
    """
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

    produced = []
    try:
        for user, rows in zip(groups, widths, strict=True):
            output = operations.empty((1, rows, 3072), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=output_memory)
            produced.append(output)
            states = operations.empty((rows, 24, 128, 128), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG)
            produced.append(states)
        tensor_groups = [[*user[:4], produced[2 * index], produced[2 * index + 1], *user[4:]]
                         for index, user in enumerate(groups)]
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
        return [(produced[2 * index], produced[2 * index + 1]) for index in range(len(groups))]
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise


def audit(root=DEFAULT_ROOT, users=MAX_USERS, rows=16):
    """Host-only description of what one batched launch would build."""
    kernels = load_kernels(root)
    io, fp32 = native.cb_plan(True)
    shares = core_shares(11, 10, users)
    return dict(status='host-source-and-placement audit only; no compilation or hardware certification',
                transforms='none; fused sources are gdn_multitoken.load_kernels(root, True) verbatim',
                native_sha256=native.HASHES, handoff_sha256=native.HANDOFF_HASHES,
                users=users, workers_per_user=HEADS, workers=users * HEADS,
                core_shares=shares, rows_per_user=rows,
                launches_per_layer=dict(per_user_recurrence_and_norm=2 * users, batched=1),
                cb_bytes_per_worker=sum(io.values()) * 2048 + sum(fp32.values()) * 4096,
                generated_sha256={role: hashlib.sha256(source.encode()).hexdigest()
                                  for role, source in kernels.items()})


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only host audit; never opens a device')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--users', type=int, default=MAX_USERS)
    parser.add_argument('--rows', type=int, default=16)
    options = parser.parse_args()
    print(json.dumps(audit(options.root, options.users, options.rows), indent=2))
