"""V5: the K5-A recurrence launch with each (user, head)'s value columns split over TWO cores (QWEN_FAST_GDN_SPLIT_V=2).

K5-A (gdn_seq_block) runs one (user, head) per core: four users x 12 heads = 48 of the 110 cores, 193.4 us a layer. The
chain is sequential over the 16 tokens but never couples value columns (every helper is a loop over the column j with its
own DST acquire), so the same chain can run on half the columns per core, two cores per (user, head), 96 cores:

  owner  (half 0)  value tiles 0, 1: the chain, then the K5-A epilogue on all four tiles, the output;
  helper (half 1)  value tiles 2, 3: the chain, then its fp32 O rows go to the owner (one remote write of two 4 KiB pages
                   and one semaphore increment), and it ends.

The gated RMSNorm is the one place columns meet: rowsum_k adds the four squared value tiles in the order 0, 1, 2, 3 inside
one DST. A split that summed partials would change that order, so the OWNER-REDUCER scheme moves data, not partial sums:
the owner's O ring ends up holding the four fp32 tiles exactly as K5-A's writer assembles them (raw bit copies of the T7
outputs) and its epilogue is K5-A's, byte for byte. Everything else is per column, so the split must be exact; the
single-card byte gate (optimisation/ttnn-op/v5split) is the proof, and QUALIFIED below stays empty until it has passed
at full scope.

Nothing here edits a pinned file. gdn_seq_block.py's sources and QUALIFIED triple are pinned by test_gdn_seq_block, so this
sibling module reuses its pieces by import (the hash-checked native prefix, the audit helpers, the placement helper of
gdn_user_batch_tp) and generates its own three kernels from gdn_seq_block_split_{reader,writer,compute}.cpp. It binds
through the tp_addresses TWINS seam, flagged: with QWEN_FAST_GDN_SPLIT_V unset nothing is imported and production is
byte-identical.

Flag (read when the process installs the twins and when the verify trace is built):
  QWEN_FAST_GDN_SPLIT_V   '' / '0' / '1' (default): the K5-A launch, unchanged. '2': the split launch. Anything else
                          raises. '2' requires QWEN_FAST_GDN_SEQ_BLOCK=1 (this REPLACES K5-A's launch) and
                          QWEN_FAST_TP=4 (the K5-A launch at the pair keeps 24 heads on 96 cores already).

The split launch is refused unless QUALIFIED holds the generated sha256 triple; the single-card probe builds anything
else with the explicit `unqualified=True` builder argument, never from the environment. The negative controls N5r (the
owner's rowsum in the order 2, 3, 0, 1) and N5x (no exchange, no wait) and the timing diagnostics ('nosnap',
'passthrough') are builder arguments too and can never be qualified.

Runtime arguments are the same LENGTH on every core and every launch (generic_op's program cache does not hash runtime
argument lengths: f945486e): the reader gets 9 words, the writer 6, the compute 1. The owner's writer is given its
helper's coordinates (unused), so the lists never differ in shape between roles.
"""

import hashlib
import os
from pathlib import Path

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_user_batch_tp as quad_batch
import tp_shapes
import verify_trace_t1

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = seq.DEFAULT_ROOT
FLAG = 'QWEN_FAST_GDN_SPLIT_V'
FACTOR = 2                  # value tiles per head split two ways
LVT = 2                     # value tiles a core runs (of 4)
LEVEL = 0
ROLES = seq.ROLES
SOURCES = dict(reader='gdn_seq_block_split_reader.cpp', writer='gdn_seq_block_split_writer.cpp',
               compute='gdn_seq_block_split_compute.cpp')
BUILD_ANCHOR = seq.BUILD_ANCHOR
VARIANTS = ('A', 'N5r', 'N5x')                  # A: the served candidate; N5r, N5x: negative controls
DIAGNOSTICS = (None, 'nosnap', 'passthrough')   # timing builds; never exact, never served
DEPTHS = (1, 2)                                 # TOKA / TOKB depth in tokens; 2 is the plan, 1 is K5-A's
DEPTH = 2
# Runtime words per role: the length every core's list has, every launch. Each kernel static_asserts that the
# highest get_arg_val index it reads is below its RT_WORDS compile argument.
RT_WORDS = dict(reader=9, writer=6, compute=1)

# level-0 variant-A sha256 triple of the generated sources, committed ONLY after the single-card byte gate
# (optimisation/ttnn-op/v5split) passes at full scope: 0 differing bytes against K5-A on every regime, the traced
# replays exact, both negative controls blind. Empty until then: the flag refuses to serve an unqualified build.
QUALIFIED = {}

MARKER = '[PINDIAG] gdn split_v build'
ENGAGED_TEMPLATE = MARKER + ' split={} users={} qualified={}'

# ---- the CB plan: K5-A's, with TOKA and TOKB two tokens deep ----
PAGE_BYTES = seq.PAGE_BYTES
TOKEN_STAGES = {22: 'TOKA', 23: 'TOKB'}


def plan(depth=DEPTH):
    """index -> (name, pages, dtype, producer RISC, consumer RISC). gdn_seq_block.CB_PLAN with the operand rings
    `depth` tokens deep: the reader stages token t+1's rows while token t's chain runs."""
    if depth not in DEPTHS:
        raise ValueError('TOKA/TOKB depth %r is not one of %s' % (depth, DEPTHS))
    entries = dict(seq.CB_PLAN)
    for index in TOKEN_STAGES:
        name, pages, dtype, producer, consumer = entries[index]
        entries[index] = (name, pages * depth, dtype, producer, consumer)
    return entries


def cb_plan(depth=DEPTH):
    """(bf16 pages by index, fp32 pages by index), the shape of gdn_multitoken.cb_plan."""
    entries = plan(depth)
    return ({index: entry[1] for index, entry in entries.items() if entry[2] == 'bf16'},
            {index: entry[1] for index, entry in entries.items() if entry[2] == 'fp32'})


def cb_bytes(depth=DEPTH):
    return sum(entry[1] * PAGE_BYTES[entry[2]] for entry in plan(depth).values())


# ---- the flag ----

def raw_flag(environ=None):
    """The flag's text as the process was given it ('' when unset); never raises."""
    return (os.environ if environ is None else environ).get(FLAG, '')


def factor(environ=None):
    """QWEN_FAST_GDN_SPLIT_V: 1 (unset, '' or '1') or 2. '2' needs K5-A on (QWEN_FAST_GDN_SEQ_BLOCK=1 with its own
    requirements) and four-card serving; anything else is a configuration error, never silently the default."""
    environ = os.environ if environ is None else environ
    value = environ.get(FLAG, '')
    if value in ('', '0', '1'):
        return 1
    if value != '2':
        raise ValueError('%s must be unset, 1 or 2' % FLAG)
    if not seq.enabled(environ):
        raise ValueError('%s=2 requires %s=1 (the split replaces the K5-A launch)' % (FLAG, seq.FLAG))
    if tp_shapes.chip_count(environ) != 4:
        raise ValueError('%s=2 is written for four-card serving (QWEN_FAST_TP=4)' % FLAG)
    return FACTOR


def enabled(environ=None):
    return factor(environ) == FACTOR


# ---- sources ----

def build_header(variant, diag, depth):
    return ('// gdn_seq_block_split generated build: variant=%s diag=%s depth=%d\n'
            '#define GDN_SEQ_BLOCK_LEVEL %d\n'
            '#define GDN_SEQ_BLOCK_VARIANT %d\n'
            '#define GDN_SEQ_BLOCK_DIAG %d\n'
            '#define GDN_SPLIT_LVT %d\n'
            '#define GDN_SPLIT_DEPTH %d\n') % (variant, diag or 'none', depth, LEVEL, VARIANTS.index(variant),
                                              DIAGNOSTICS.index(diag), LVT, depth)


def _source(directory, role):
    data = (Path(directory) / SOURCES[role]).read_bytes().decode()
    if '\r' in data:
        raise ValueError('%s must be LF-only: its sha256 is its identity' % SOURCES[role])
    return data


def generate(root=DEFAULT_ROOT, *, variant='A', diag=None, depth=DEPTH, sources=HERE):
    """role -> generated source text: the compute is the served native prefix (checked against its pin before it is
    sliced) plus the split compute; the reader and writer are the split ones."""
    if variant not in VARIANTS:
        raise ValueError('Unknown gdn_seq_block_split variant %r' % (variant,))
    if diag not in DIAGNOSTICS:
        raise ValueError('Unknown gdn_seq_block_split diagnostic %r' % (diag,))
    if depth not in DEPTHS:
        raise ValueError('TOKA/TOKB depth %r is not one of %s' % (depth, DEPTHS))
    header = build_header(variant, diag, depth)
    kernels = {}
    for role in ROLES:
        text = seq._replace_once(_source(sources, role), BUILD_ANCHOR, header, SOURCES[role])
        kernels[role] = seq.native_prefix(root) + text if role == 'compute' else text
    return kernels


def sha256(kernels):
    return {role: hashlib.sha256(kernels[role].encode()).hexdigest() for role in ROLES}


class Build(dict):
    """role -> generated source, plus what it was generated as. Only load_kernels makes one."""

    split = FACTOR
    level = LEVEL

    def __init__(self, kernels, variant, diag, depth, qualified):
        super().__init__(kernels)
        self.variant, self.diag, self.depth, self.qualified = variant, diag, depth, qualified

    def tag(self, role):
        return seq.src_tag(self[role])


def load_kernels(root=DEFAULT_ROOT, *, variant='A', diag=None, depth=DEPTH, unqualified=False, sources=HERE):
    """The generated sources, refused unless QUALIFIED holds their sha256 triple. `unqualified=True` is the
    single-card probe's builder argument and nothing else's: never read from the environment, and the only way to
    build a control, a diagnostic or a depth the plan does not serve."""
    if type(unqualified) is not bool:
        raise ValueError('unqualified must be an explicit bool')
    kernels = generate(root, variant=variant, diag=diag, depth=depth, sources=sources)
    qualified = variant == 'A' and diag is None and depth == DEPTH and QUALIFIED.get(LEVEL) == sha256(kernels)
    if not qualified and not unqualified:
        raise ValueError('gdn_seq_block_split (variant %s, diag %s, depth %d) is not qualified: QUALIFIED holds no '
                         'matching sha256 triple; only the single-card probe builds it unqualified'
                         % (variant, diag or 'none', depth))
    return Build(kernels, variant, diag, depth, qualified)


_SERVED = {}


def served_kernels(root=None, environ=None):
    """The qualified split build, generated once per root."""
    root = Path(root if root is not None else (os.environ if environ is None else environ).get(
        'TT_METAL_HOME', str(DEFAULT_ROOT)))
    if str(root) not in _SERVED:
        _SERVED[str(root)] = load_kernels(root)
    return _SERVED[str(root)]


# ---- placement ----

def heads():
    return quad_batch.heads()


def placement(horizontal, vertical, users):
    """Per user, `[(head, half, (x, y)), ...]` over 2 x heads cores: worker w of a user is core (24u + w) of the grid in
    column-major order; head = w // 2 and half = w % 2. Half 0 is the owner. A pair is two consecutive cores of one
    column (the owner's row is even, the helper's the next), so the exchange is one hop."""
    shares = quad_batch.core_shares(horizontal, vertical, users, workers=FACTOR * heads())
    found = []
    for share in shares:
        pairs = [(worker // FACTOR, worker % FACTOR, point) for worker, point in enumerate(share)]
        for (head, half, point), (_, _, peer) in zip(pairs[0::2], pairs[1::2]):
            if point[0] != peer[0] or peer[1] != point[1] + 1:
                raise ValueError('A head pair must be two consecutive cores of one grid column; the grid height '
                                 '%d does not place it so' % vertical)
        found.append(pairs)
    return found


def peers(pairs):
    """head -> {half: point} for one user's placement."""
    found = {}
    for head, half, point in pairs:
        found.setdefault(head, {})[half] = point
    return found


# ---- program ----

def compile_args(role, kernels):
    """Per role, identical on every core of every user (so the coalescing merges them). The reader's and writer's
    leading values are K5-A's; RT_WORDS then SRC_TAG close each list, and the compute's LEVEL is K5-A's."""
    if role == 'reader':
        # Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT, RT_WORDS, SRC_TAG
        return [*tp_shapes.k5_reader_arguments(tp_shapes.chip_count()), RT_WORDS[role], kernels.tag(role)]
    if role == 'writer':
        return [4, 4, heads(), RT_WORDS[role], kernels.tag(role)]
    if role == 'compute':
        return [seq.batch.bits(1e-6), seq.batch.bits(128 ** -0.5), seq.batch.bits(128e-6), seq.batch.bits(128 ** 0.5),
                LEVEL, RT_WORDS[role], kernels.tag(role)]
    raise ValueError('Unknown kernel role')


def runtime_args(role, head, half, addresses, peer=(0, 0)):
    """One core's runtime arguments from that user's eight buffer addresses; `peer` is the other half's NOC
    coordinates. The length is RT_WORDS[role] for every core, owner or helper."""
    if type(head) is not int or not 0 <= head < heads():
        raise ValueError('Head index within the %d-head TP%d shard required' % (heads(), tp_shapes.chip_count()))
    if half not in (0, 1):
        raise ValueError('Half 0 (owner) or 1 (helper) required')
    if len(addresses) != 8:
        raise ValueError('Eight per-user buffer addresses required')
    if len(peer) != 2:
        raise ValueError('Peer NOC coordinates (x, y) required')
    if role == 'reader':
        values = [head, half, addresses[0], addresses[1], addresses[2], addresses[3], addresses[6], addresses[7],
                  addresses[5]]
    elif role == 'writer':
        values = [head, half, addresses[4], addresses[5], peer[0], peer[1]]
    elif role == 'compute':
        values = [half]
    else:
        raise ValueError('Unknown kernel role')
    if len(values) != RT_WORDS[role]:
        raise AssertionError('runtime argument count of %s drifted from RT_WORDS' % role)
    return values


def build_program(operations, mesh, user_shards, kernels):
    """gdn_seq_block.build_program's structure with these kernels on twice the cores: its descriptor coalescing under
    QWEN_FAST_VERIFY_T1, its configs, one chip program per chip; plus the CB plan of `plan(depth)`, ONE semaphore (id 0)
    over the union of the cores (the helper's increment lands in its owner's), and each core's peer coordinates.

    `user_shards[u]` is that user's eight per-chip tensor lists, in the order
    `[qkv, beta, gate, initial, output, states, z, norm_w]`."""
    if not isinstance(kernels, Build):
        raise ValueError('gdn_seq_block_split kernels must come from gdn_seq_block_split.load_kernels')
    users = len(user_shards)
    chips = seq.batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    placed = placement(grid.x, grid.y, users)
    shares = [[point for _, _, point in pairs] for pairs in placed]
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
    io, fp32 = cb_plan(kernels.depth)
    for counts, dtype, page in ((io, operations.bfloat16, 2048), (fp32, operations.float32, 4096)):
        for index, count in counts.items():
            buffers.append(operations.CBDescriptor(total_size=count * page, core_ranges=union,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    semaphore = operations.SemaphoreDescriptor(id=0, core_ranges=union, initial_value=0)
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
        for shards, pairs, cores in zip(user_shards, placed, ranges, strict=True):
            local = [value[chip] for value in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('One user inputs, initial state and prefix outputs must not alias')
            if any(address in private for address in addresses[:7]):
                raise ValueError('Packed users must not share any buffer but the norm weight')
            private.extend(addresses[:7])
            weights.append(addresses[7])
            device = local[0].device()
            where = peers(pairs)
            noc = {}
            for head, halves in where.items():
                for half, point in halves.items():
                    core = device.worker_core_from_logical_core(operations.CoreCoord(*point))
                    noc[(head, half)] = (core.x, core.y)
            for role in ROLES:
                args = list(compile_args(role, kernels))
                for index in seq.ACCESSORS[role]:
                    args.extend(operations.TensorAccessorArgs(local[index]).get_compile_time_args())
                runtime = [(point, runtime_args(role, head, half, addresses, noc[(head, 1 - half)]))
                           for head, half, point in pairs]
                if coalesce:
                    planned.append((role, args, cores, runtime))
                    continue
                values = operations.RuntimeArgs()
                for (horizontal, vertical), words in runtime:
                    values[horizontal][vertical] = words
                descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=configs[role])
                descriptor.runtime_args = values
                descriptors.append(descriptor)
        if coalesce:
            descriptors, merged = seq.batch.coalesced_descriptors(operations, kernels, configs, planned, union)
            coalesced.append(merged)
        if any(weight in private for weight in weights):
            raise ValueError('The shared norm weight must not alias any packed user buffer')
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers, semaphores=[semaphore])
    if coalesce:
        # The T1 gate counts one 'coalesced' per GDN layer; this launch is that layer's launch.
        verify_trace_t1.note('coalesced' if all(coalesced) else 'coalesce_fallback')
    return program


_ENGAGED = set()


def engaged_line(users, qualified):
    return ENGAGED_TEMPLATE.format(FACTOR, int(users), int(bool(qualified)))


def note_engaged(users, qualified):
    """One engagement line per user count per process, so the gate can require it (mounted is not executed)."""
    if users not in _ENGAGED:
        _ENGAGED.add(users)
        seq.log_line(engaged_line(users, qualified))


def resolve_kernels(kernels):
    """The split build for whatever the caller passed: nothing or a qualified K5-A build (the twin's route, where
    gdn_user_batch_conv hands the K5-A build it took for its own level report) means the served split build; a split
    build is used as is; anything else is refused."""
    if isinstance(kernels, Build):
        return kernels
    if kernels is None or (isinstance(kernels, seq.Build) and kernels.qualified and kernels.level == LEVEL
                           and kernels.variant == 'A' and kernels.diag is None):
        return served_kernels()
    raise ValueError('gdn_seq_block_split.execute takes the qualified K5-A build (replaced by the qualified split build) '
                     'or a split build from gdn_seq_block_split.load_kernels')


def execute(mesh, users, operations=None, *, output_memory=None, kernels=None):
    """gdn_seq_block.execute with the split kernels: every packed user's T=16 recurrence and fused norm/gate in ONE
    launch on 2 x 12 cores a user. Same signature, same return (`[(output, states), ...]`, each the served shape, dtype,
    layout and placement), same refusals."""
    if isinstance(operations, dict):
        raise ValueError('gdn_seq_block_split.execute takes operations third; pass a build as kernels=')
    if operations is None:
        import ttnn as operations
    build = resolve_kernels(kernels)
    groups = [tuple(user) for user in users]
    widths = seq.validate_users([[tuple(value.shape) for value in user] for user in groups])
    # The served launch's runtime pin, kept so the flag never drops it (gdn_seq_block.execute does the same).
    native.validate_handoff_runtime(Path(os.environ.get('TT_METAL_HOME', str(DEFAULT_ROOT))))
    if output_memory is None:
        output_memory = operations.L1_MEMORY_CONFIG
    if output_memory not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG):
        raise ValueError('Output memory must be interleaved DRAM or L1')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT or
           value.memory_config() != operations.DRAM_MEMORY_CONFIG for user in groups for value in user):
        raise ValueError('Batched GDN requires interleaved DRAM BF16 TILE inputs')
    chips = seq.batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    placement(grid.x, grid.y, len(groups))
    note_engaged(len(groups), build.qualified)

    produced = []
    try:
        for rows in widths:
            output = operations.empty((1, rows, seq.batch.geometry().gdn_value), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=output_memory)
            produced.append(output)
            states = operations.empty((rows, heads(), 128, 128), device=mesh, dtype=operations.bfloat16,
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
                                                   for group in tensor_groups], build)
        operations.generic_op(flat, program)
        return [(produced[2 * index], produced[2 * index + 1]) for index in range(len(groups))]
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise


def audit(root=DEFAULT_ROOT, users=quad_batch.MAX_USERS, variant='A', diag=None, depth=DEPTH):
    """Host-only description of what one split launch would build; never opens a device."""
    kernels = load_kernels(root, variant=variant, diag=diag, depth=depth, unqualified=True)
    io, fp32 = cb_plan(depth)
    return dict(status='host-source-and-placement audit only; no compilation or hardware certification',
                transforms='compute = the served native prefix (sha256-checked, sliced at kernel_main) + '
                           'gdn_seq_block_split_compute.cpp; reader and writer new',
                variant=variant, diag=diag, depth=depth, qualified=kernels.qualified,
                qualified_triple=QUALIFIED.get(LEVEL), native_sha256=native.HASHES[seq.NATIVE_COMPUTE],
                generated_sha256=sha256(kernels), src_tag={role: kernels.tag(role) for role in ROLES},
                compile_args={role: compile_args(role, kernels) for role in ROLES},
                runtime_words=dict(RT_WORDS), users=users, cores_per_user=FACTOR * heads(),
                cores=users * FACTOR * heads(),
                cb_indices=sorted(list(io) + list(fp32)), cb_bytes_per_core=cb_bytes(depth),
                k5a_cb_bytes_per_core=seq.cb_bytes(), launches_per_layer=1)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only host audit; never opens a device')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--users', type=int, default=quad_batch.MAX_USERS)
    parser.add_argument('--variant', choices=VARIANTS, default='A')
    parser.add_argument('--diag', choices=[value for value in DIAGNOSTICS if value], default=None)
    parser.add_argument('--depth', type=int, choices=DEPTHS, default=DEPTH)
    options = parser.parse_args()
    print(json.dumps(audit(options.root, options.users, options.variant, options.diag, options.depth), indent=2,
                     sort_keys=True, default=str))
