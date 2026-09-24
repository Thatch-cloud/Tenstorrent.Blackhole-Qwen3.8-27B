"""K5-A: the packed verify's GDN recurrence with the work around the chain batched per block.

The served launch (`gdn_user_batch.execute`: `gdn_multitoken.load_kernels(root, True)` on 96
cores, one (user, head) per core) runs the whole T=1 decode step sixteen times per block: per
token it re-reads 18 DRAM pages, converts every input to fp32, normalises q and k, runs the
recurrence, then the fused norm/gate - 230 tile ops and 38 helper drains per token (K5 plan,
section 2). K5-A keeps the recurrence chain sequential and exact and moves everything that is
row-independent out of it:

  prologue  (once per block) the bf16 -> fp32 copies of q, k, v, z, g, beta and norm_w, exp(g)
            over the whole g tile, and the q/k L2 norms over all 16 token rows at once;
  chain     (per token) 124 tile ops: the served LLK calls on the served operand values, in the
            served srcA/srcB roles and K order, with the state add done in DST (DEST_TO_SRCB)
            and the state copy unpacked once and packed twice;
  epilogue  (once per block) the fused rms_norm * w * silu(z) over the 16 rows.
The reader stages each token's operands as bit copies of the prologue's packed fp32 words, so a
core reads 38 DRAM pages per block instead of 308.

Exactness class B* (K5 plan, 1.1 A1): every step is either the served call on the same values or
a property shown exact on this pinned tt-metal (the DEST_TO_SRCB add, docs/gdn-outer-add-
experiment.md; the double pack, docs/gdn-dual-state-copy.md). The rest - row independence of
the batched prologue/epilogue, SFPU exp position independence, TF32 idempotence of the staged
k~ column - is what the card-B probe's P0 byte compare (gdn_seq_block_device_test.py) decides.

Sources. The compute kernel is the served native compute prefix (decode_gated_delta_rule.cpp up
to `void kernel_main() {`, sha256-checked against gdn_multitoken.HASHES before slicing, the
gdn_shared_qk_compute.py pattern) followed by gdn_seq_block_compute.cpp: the served helpers are
reused verbatim. The reader and writer are new (gdn_seq_block_reader.cpp, _writer.cpp). Each
source carries a generated build header; SRC_TAG, the first 32 bits of each generated source's
sha256, is a compile arg so the JIT cache is keyed on content. No pinned file is changed.

Every CB has one producer RISC and one consumer RISC, with no exception: norm_w arrives in W
(reader -> compute) and the epilogue's bf16 round trips use RT, a compute-only ring - not the
norm_w ring, as the served kernel does, where the packer's local count never sees the reader's
push. So no CB changes hands and nothing needs the served tiles_received handoff. `execute` still
checks the served runtime pin (gdn_multitoken.validate_handoff_runtime, HANDOFF_HASHES) exactly as
gdn_user_batch.execute does, so turning the flag on never drops a runtime check the served launch
makes.

Flags (all read when the verify trace is built):
  QWEN_FAST_GDN_SEQ_BLOCK        '0' (default) or '1'; anything else raises. '1' requires
                                 QWEN_FAST_GDN_USER_BATCH=1.
  QWEN_FAST_GDN_SEQ_BLOCK_LEVEL  the A+ increments bitmask, default 0. A level whose generated
                                 sha256 triple is not in QUALIFIED is refused.
  QWEN_FAST_GDN_SEQ_BLOCK_AUDIT  audited arms only: layers (e.g. 0,23,47) whose served launch also
                                 runs on the same inputs (served_audit_launch, which keeps it out
                                 of the T1 gate's one-'coalesced'-per-layer count), both results
                                 compared after the replay.

QUALIFIED starts empty: only the card-B probe builds, and it does so with the explicit
`unqualified=True` builder argument, never from the environment. The probe-only builds A0 (T5/T6
unfused, bisection) and N (the SFPU state add, the negative control), and the timing diagnostics
('nosnap', 'passthrough'), are builder arguments too, and can never be qualified.

Uncertified: host construction only. Nothing here has run on a device; the equality and timing
evidence is gdn_seq_block_device_test.py (card B, scripts/ci/gdn-seq-block-rig.sh).
"""

import hashlib
import os
from pathlib import Path
import re

import gdn_multitoken as native
import gdn_user_batch as batch
import verify_trace_t1


DEFAULT_ROOT = batch.DEFAULT_ROOT
HERE = Path(__file__).resolve().parent
ROLES = ('reader', 'writer', 'compute')
SOURCES = dict(reader='gdn_seq_block_reader.cpp', writer='gdn_seq_block_writer.cpp',
               compute='gdn_seq_block_compute.cpp')
NATIVE_COMPUTE = 'compute/decode_gated_delta_rule.cpp'
MAIN = 'void kernel_main() {'
PREFIX_END = '}  // namespace\n\n'
BUILD_ANCHOR = '// @@GDN_SEQ_BLOCK_BUILD@@\n'
# The served helpers and names the generated compute calls; each must appear exactly once in the
# sliced prefix (the hash check already pins them - these name what the new code depends on).
NATIVE_ANCHORS = (
    'constexpr uint32_t cb_state = 5, cb_ones = 6;',
    'inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }',
    'inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }',
    'void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {',
    'void ew_off(uint32_t a, uint32_t ia0, uint32_t b, uint32_t ib0, uint32_t o, uint32_t n, int op) {',
    'void expc(uint32_t in, uint32_t o, uint32_t n) {',
    'void silu_tiles(uint32_t in, uint32_t o, uint32_t n) {',
    'void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {',
    'void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {',
    'void rowbcast_delta(uint32_t delta, uint32_t o, uint32_t Vt) {',
    'void rowsum_k(uint32_t in, uint32_t o, uint32_t Kt) {',
    'void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {',
    'void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {',
)

FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK'
LEVEL_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_LEVEL'
AUDIT_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT'
ROWS = 16
HEADS = batch.HEADS
LAYERS = 48
LEVEL_BITS = 4                 # the A+ increments (i)-(iv) of the plan
IMPLEMENTED_LEVELS = (0,)      # none of the increments exists yet
VARIANTS = ('A', 'A0', 'N')    # A: served candidate; A0: bisection only; N: negative control
DIAGNOSTICS = (None, 'nosnap', 'passthrough')  # P1 (iii) timing builds; never exact, never served

# level -> {'reader': sha256, 'writer': sha256, 'compute': sha256} of the generated variant-A
# sources. Empty until card-B P0 shows 0 differing bytes (then commit that level's triple here, the
# fused_commit.QUALIFIED_KERNEL_SHA256 pattern).
QUALIFIED = {}

# model_batch's per-capture marker (K5 plan 3.1) and the gate's regex for it (3.9).
MARKER = '[PINDIAG] gdn seq_block calls this captured forward:'
GATE_PATTERN = re.compile(r'gdn seq_block calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers '
                          r'level=([0-9]+)')
AUDIT_MARKER = '[GDN-SEQ-BLOCK-AUDIT]'
AUDIT_PATTERN = re.compile(r'\[GDN-SEQ-BLOCK-AUDIT\] layer=([0-9]+) user=([0-9]+) mismatches=([0-9]+)')

# ---- the CB plan: index -> (name, pages, dtype, producer RISC, consumer RISC) ----
# The K5 plan's section 3.3 table, with one change: its W (8 pages, reader -> compute for norm_w,
# then compute -> compute for the epilogue's round trips, the served full ring) is split into W
# (4 pages, reader -> compute) and RT (4 pages, compute -> compute). Same bytes, and no CB with two
# producer RISCs: in the served pattern the packer's local tiles_received never counts the reader's
# push, so the second round trip's wait passes before its pack lands (llk_io_pack.h:58,68;
# llk_io_unpack.h:30) and only pipeline depth kept it safe.
PAGE_BYTES = dict(bf16=2048, fp32=4096)
RISCS = ('reader', 'compute', 'writer')
ONES = 6
CB_PLAN = {
    0: ('IN_QK', 8, 'bf16', 'reader', 'compute'),
    1: ('IN_V', 4, 'bf16', 'reader', 'compute'),
    2: ('IN_Z', 4, 'bf16', 'reader', 'compute'),
    3: ('IN_GB', 2, 'bf16', 'reader', 'compute'),
    4: ('S0', 16, 'bf16', 'reader', 'compute'),
    5: ('FB', 16, 'bf16', 'compute', 'compute'),
    6: ('ONES', 1, 'fp32', 'reader', 'compute'),
    7: ('SOUT', 16, 'bf16', 'compute', 'writer'),
    8: ('W', 4, 'bf16', 'reader', 'compute'),
    9: ('OUT', 4, 'bf16', 'compute', 'writer'),
    10: ('X', 8, 'fp32', 'compute', 'compute'),
    11: ('QKN', 8, 'fp32', 'compute', 'reader'),
    12: ('VF', 4, 'fp32', 'compute', 'reader'),
    13: ('ZF', 4, 'fp32', 'compute', 'compute'),
    14: ('GF', 1, 'fp32', 'compute', 'compute'),
    15: ('BF', 1, 'fp32', 'compute', 'reader'),
    16: ('GEXP', 1, 'fp32', 'compute', 'reader'),
    17: ('WF', 4, 'fp32', 'compute', 'compute'),
    18: ('NSQ', 4, 'fp32', 'compute', 'compute'),
    19: ('SUM', 1, 'fp32', 'compute', 'compute'),
    20: ('FAC_Q', 1, 'fp32', 'compute', 'compute'),
    21: ('FAC_K', 1, 'fp32', 'compute', 'compute'),
    22: ('TOKA', 10, 'fp32', 'reader', 'compute'),
    23: ('TOKB', 8, 'fp32', 'reader', 'compute'),
    24: ('S', 16, 'fp32', 'compute', 'compute'),
    25: ('H', 16, 'fp32', 'compute', 'compute'),
    26: ('UD', 4, 'fp32', 'compute', 'compute'),
    27: ('DL', 8, 'fp32', 'compute', 'compute'),
    28: ('OT', 4, 'fp32', 'compute', 'writer'),
    29: ('O', 4, 'fp32', 'writer', 'compute'),
    30: ('RT', 4, 'bf16', 'compute', 'compute'),
}
# A0 (bisection only): T5 unfused, the outer product in its own 16-page ring (+64 KiB).
A0_EXTRA = {31: ('OUTER', 16, 'fp32', 'compute', 'compute')}
SERVED_CB_BYTES = sum(native.cb_plan(True)[0].values()) * 2048 + sum(native.cb_plan(True)[1].values()) * 4096


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK=1. Default OFF; anything but '0' or '1' is a configuration error, and
    so is '1' without QWEN_FAST_GDN_USER_BATCH=1 (the new launch replaces the batched one)."""
    environ = os.environ if environ is None else environ
    on = _flag(FLAG, environ)
    if on and not batch.enabled(environ):
        raise ValueError('%s=1 requires %s=1' % (FLAG, batch.FLAG))
    return on


def level(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_LEVEL: a decimal bitmask of the A+ increments, default 0."""
    value = (os.environ if environ is None else environ).get(LEVEL_FLAG, '0')
    if type(value) is not str or not value.isdigit() or value != str(int(value)):
        raise ValueError('%s must be a non-negative decimal integer' % LEVEL_FLAG)
    wanted = int(value)
    if wanted >= 1 << LEVEL_BITS:
        raise ValueError('%s holds %d increment bits' % (LEVEL_FLAG, LEVEL_BITS))
    return wanted


def audit_layers(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT: comma-separated GDN layer indices, e.g. '0,23,47'. Unset or
    empty: no audit. Anything else - a blank item, a sign, a leading zero, a repeat, a layer
    outside 0..47 - is a configuration error."""
    value = (os.environ if environ is None else environ).get(AUDIT_FLAG, '')
    if value == '':
        return ()
    layers = []
    for item in value.split(','):
        if not item.isdigit() or item != str(int(item)) or int(item) >= LAYERS or int(item) in layers:
            raise ValueError('%s must list distinct GDN layers 0..%d, e.g. 0,23,47' % (AUDIT_FLAG, LAYERS - 1))
        layers.append(int(item))
    return tuple(layers)


def marker(calls, layers, built_level):
    """model_batch's line after a captured forward: how many GDN layers ran this launch."""
    return '%s %d of %d GDN layers level=%d' % (MARKER, calls, layers, built_level)


def audit_line(layer, user, mismatches):
    return '%s layer=%d user=%d mismatches=%d' % (AUDIT_MARKER, layer, user, mismatches)


# ---- CB plan ----

def plan(variant='A'):
    if variant not in VARIANTS:
        raise ValueError('Unknown gdn_seq_block variant %r' % (variant,))
    entries = dict(CB_PLAN)
    if variant == 'A0':
        entries.update(A0_EXTRA)
    return entries


def cb_plan(variant='A'):
    """(bf16 pages by index, fp32 pages by index), the shape of gdn_multitoken.cb_plan."""
    entries = plan(variant)
    return ({index: entry[1] for index, entry in entries.items() if entry[2] == 'bf16'},
            {index: entry[1] for index, entry in entries.items() if entry[2] == 'fp32'})


def cb_bytes(variant='A'):
    return sum(entry[1] * PAGE_BYTES[entry[2]] for entry in plan(variant).values())


# ---- sources ----

def _replace_once(source, before, after, what):
    if source.count(before) != 1:
        raise ValueError('%s: expected one %r' % (what, before.strip()))
    return source.replace(before, after, 1)


def prefix(source):
    """The served native compute up to its entry point: includes, CB names, helpers."""
    if source.count(MAIN) != 1:
        raise ValueError('Expected one pinned compute entry point')
    head = source[:source.index(MAIN)]
    if not head.endswith(PREFIX_END):
        raise ValueError('Native prefix no longer ends with its helper namespace')
    for anchor in NATIVE_ANCHORS:
        if head.count(anchor) != 1:
            raise ValueError('Native helper anchor changed: %s' % anchor)
    return head


def native_prefix(root=DEFAULT_ROOT):
    """Read the served compute, check it against the pin BEFORE slicing, slice the prefix."""
    data = (Path(root) / native.KERNEL_ROOT / NATIVE_COMPUTE).read_bytes()
    if hashlib.sha256(data).hexdigest() != native.HASHES[NATIVE_COMPUTE]:
        raise ValueError('Native compute hash changed: %s' % NATIVE_COMPUTE)
    return prefix(data.decode())


def build_header(built_level, variant, diag):
    return ('// gdn_seq_block generated build: level=%d variant=%s diag=%s\n'
            '#define GDN_SEQ_BLOCK_LEVEL %d\n'
            '#define GDN_SEQ_BLOCK_VARIANT %d\n'
            '#define GDN_SEQ_BLOCK_DIAG %d\n') % (built_level, variant, diag or 'none', built_level,
                                                   VARIANTS.index(variant), DIAGNOSTICS.index(diag))


def _source(directory, role):
    data = (Path(directory) / SOURCES[role]).read_bytes().decode()
    if '\r' in data:
        raise ValueError('%s must be LF-only: its sha256 is its identity' % SOURCES[role])
    return data


def generate(root=DEFAULT_ROOT, built_level=0, *, variant='A', diag=None, sources=HERE):
    """role -> generated source text. Refuses an unimplemented level, variant or diagnostic."""
    if type(built_level) is not int or built_level not in IMPLEMENTED_LEVELS:
        raise ValueError('gdn_seq_block level %r is not implemented (levels %s)' % (built_level, IMPLEMENTED_LEVELS))
    if variant not in VARIANTS:
        raise ValueError('Unknown gdn_seq_block variant %r' % (variant,))
    if diag not in DIAGNOSTICS:
        raise ValueError('Unknown gdn_seq_block diagnostic %r' % (diag,))
    header = build_header(built_level, variant, diag)
    kernels = {}
    for role in ROLES:
        text = _replace_once(_source(sources, role), BUILD_ANCHOR, header, SOURCES[role])
        kernels[role] = native_prefix(root) + text if role == 'compute' else text
    return kernels


def sha256(kernels):
    return {role: hashlib.sha256(kernels[role].encode()).hexdigest() for role in ROLES}


def src_tag(source):
    """The first 32 bits of the source's sha256: a compile arg, so the JIT cache key covers content."""
    return int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)


class Build(dict):
    """role -> generated source, plus what it was generated as. Only load_kernels makes one."""

    def __init__(self, kernels, built_level, variant, diag, qualified):
        super().__init__(kernels)
        self.level, self.variant, self.diag, self.qualified = built_level, variant, diag, qualified

    def tag(self, role):
        return src_tag(self[role])


def load_kernels(root=DEFAULT_ROOT, built_level=0, *, variant='A', diag=None, unqualified=False, sources=HERE):
    """The generated sources, refused unless QUALIFIED holds their sha256 triple.

    `unqualified=True` is the card-B probe's builder argument and nothing else's: it is never read
    from the environment, and it is the only way to build A0, N or a diagnostic."""
    if type(unqualified) is not bool:
        raise ValueError('unqualified must be an explicit bool')
    kernels = generate(root, built_level, variant=variant, diag=diag, sources=sources)
    qualified = variant == 'A' and diag is None and QUALIFIED.get(built_level) == sha256(kernels)
    if not qualified and not unqualified:
        raise ValueError('gdn_seq_block level %d (variant %s, diag %s) is not qualified: QUALIFIED holds no '
                         'matching sha256 triple; only the card-B probe builds it unqualified'
                         % (built_level, variant, diag or 'none'))
    return Build(kernels, built_level, variant, diag, qualified)


_SERVED = {}


def served_kernels(root=None, environ=None):
    """The qualified build for QWEN_FAST_GDN_SEQ_BLOCK_LEVEL, generated once per (root, level)."""
    root = Path(root if root is not None else (os.environ if environ is None else environ).get(
        'TT_METAL_HOME', str(DEFAULT_ROOT)))
    key = (str(root), level(environ))
    if key not in _SERVED:
        _SERVED[key] = load_kernels(root, key[1])
    return _SERVED[key]


# ---- program ----

def compile_args(role, kernels):
    """Per role; every value but SRC_TAG and LEVEL is the served fused build's
    (gdn_user_batch.compile_args, which the unit tests hold these against)."""
    if role == 'reader':
        # Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT, SRC_TAG
        return [4, 4, HEADS, 3, 160, 0, 32, 64, 96, 0, kernels.tag(role)]
    if role == 'writer':
        return [4, 4, HEADS, kernels.tag(role)]
    if role == 'compute':
        return [batch.bits(1e-6), batch.bits(128 ** -0.5), batch.bits(128e-6), batch.bits(128 ** 0.5),
                kernels.level, kernels.tag(role)]
    raise ValueError('Unknown kernel role')


# Indices into the per-user eight [qkv, beta, gate, initial, output, states, z, norm_w].
ACCESSORS = dict(reader=(0, 1, 2, 3, 6, 7), writer=(4, 5), compute=())


def runtime_args(role, head, addresses):
    """One core's runtime arguments from that user's eight buffer addresses. The compute reads
    none; it is still given the served [rows] so every core's argument shape is the served one."""
    if type(head) is not int or not 0 <= head < HEADS:
        raise ValueError('Head index within the 24-head TP2 shard required')
    if len(addresses) != 8:
        raise ValueError('Eight per-user buffer addresses required')
    if role == 'reader':
        return [head, addresses[0], addresses[1], addresses[2], addresses[3], addresses[6], addresses[7]]
    if role == 'writer':
        return [head, addresses[4], addresses[5]]
    if role == 'compute':
        return [ROWS]
    raise ValueError('Unknown kernel role')


def validate_users(shapes):
    """gdn_user_batch.validate_users, plus the one width this kernel is written for."""
    widths = batch.validate_users(shapes)
    if any(rows != ROWS for rows in widths):
        raise ValueError('gdn_seq_block runs 16-row segments only; other widths take the served launch')
    return widths


def build_program(operations, mesh, user_shards, kernels):
    """gdn_user_batch.build_program's geometry exactly - its core shares, its rectangle ranges and
    one descriptor per role under QWEN_FAST_VERIFY_T1 (#12), its configs - with these kernels,
    their arguments and this CB plan on the union of the cores.

    `user_shards[u]` is that user's eight per-chip tensor lists, in the order
    `[qkv, beta, gate, initial, output, states, z, norm_w]`.
    """
    if not isinstance(kernels, Build):
        raise ValueError('gdn_seq_block kernels must come from gdn_seq_block.load_kernels')
    users = len(user_shards)
    chips = batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    shares = batch.core_shares(grid.x, grid.y, users)
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
    io, fp32 = cb_plan(kernels.variant)
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
        for shards, share, cores in zip(user_shards, shares, ranges, strict=True):
            local = [value[chip] for value in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('One user inputs, initial state and prefix outputs must not alias')
            if any(address in private for address in addresses[:7]):
                raise ValueError('Packed users must not share any buffer but the norm weight')
            private.extend(addresses[:7])
            weights.append(addresses[7])
            for role in ROLES:
                args = list(compile_args(role, kernels))
                for index in ACCESSORS[role]:
                    args.extend(operations.TensorAccessorArgs(local[index]).get_compile_time_args())
                if coalesce:
                    planned.append((role, args, cores, [(point, runtime_args(role, head, addresses))
                                                        for head, point in enumerate(share)]))
                    continue
                runtime = operations.RuntimeArgs()
                for head, (horizontal, vertical) in enumerate(share):
                    runtime[horizontal][vertical] = runtime_args(role, head, addresses)
                descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=configs[role])
                descriptor.runtime_args = runtime
                descriptors.append(descriptor)
        if coalesce:
            descriptors, merged = batch.coalesced_descriptors(operations, kernels, configs, planned, union)
            coalesced.append(merged)
        if any(weight in private for weight in weights):
            raise ValueError('The shared norm weight must not alias any packed user buffer')
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    if coalesce:
        # The T1 gate counts one 'coalesced' per GDN layer; this launch is that layer's launch.
        verify_trace_t1.note('coalesced' if all(coalesced) else 'coalesce_fallback')
    return program


def execute(mesh, users, operations=None, *, output_memory=None, kernels=None):
    """Every packed user's T=16 recurrence and fused norm/gate in ONE launch, K5-A kernels.

    The return is gdn_user_batch.execute's; the signature is the plan's (section 3.1), which is
    gdn_user_batch.execute's less its positional `kernels`: the build is the qualified one for
    QWEN_FAST_GDN_SEQ_BLOCK_LEVEL unless the card-B probe passes its own BY KEYWORD. So the served
    call site `gdn_user_batch.execute(mesh, users, kernels, operations, output_memory=...)`
    (gdn_user_batch_conv.py:139) becomes `execute(mesh, users, operations, output_memory=...)`; a
    build or a source dict in the `operations` slot (the drop-in mistake) is refused.
    `users` is a sequence of `(qkv, beta, gate, initial, z, norm_w)` tuples; returns
    `[(output, states), ...]` in user order, each the served shape, dtype, layout and placement -
    `(1, 16, 3072)` gated output in `output_memory` (L1 by default), `(16, 24, 128, 128)` bf16
    prefix states in DRAM.
    """
    if isinstance(operations, dict):
        raise ValueError('gdn_seq_block.execute takes operations third; pass a build as kernels=')
    if operations is None:
        import ttnn as operations

    if kernels is None:
        kernels = served_kernels()
    if not isinstance(kernels, Build):
        raise ValueError('gdn_seq_block kernels must come from gdn_seq_block.load_kernels')
    groups = [tuple(user) for user in users]
    widths = validate_users([[tuple(value.shape) for value in user] for user in groups])
    # The served launch's runtime pin (gdn_user_batch.py:306), kept so the flag never drops it.
    native.validate_handoff_runtime(Path(os.environ.get('TT_METAL_HOME', str(DEFAULT_ROOT))))
    if output_memory is None:
        output_memory = operations.L1_MEMORY_CONFIG
    if output_memory not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG):
        raise ValueError('Output memory must be interleaved DRAM or L1')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT or
           value.memory_config() != operations.DRAM_MEMORY_CONFIG for user in groups for value in user):
        raise ValueError('Batched GDN requires interleaved DRAM BF16 TILE inputs')
    chips = batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    batch.core_shares(grid.x, grid.y, len(groups))

    produced = []
    try:
        for rows in widths:
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
                                                   for group in tensor_groups], kernels)
        operations.generic_op(flat, program)
        return [(produced[2 * index], produced[2 * index + 1]) for index in range(len(groups))]
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise


def served_audit_launch(mesh, users, served, operations=None, *, output_memory=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT's second launch of an audited layer: the served batched launch
    (gdn_user_batch.execute with the served build) on the same inputs, returning its
    `[(output, states), ...]` for the after-replay compare.

    Its verify_trace_t1 note is NOT counted. Under QWEN_FAST_VERIFY_T1 the served build notes
    'coalesced' once per launch (gdn_user_batch.py:288), and the T1 gate requires exactly one per
    GDN layer per capture (lever_n_m3native_gate.py:1190-1194): counted, three audited layers would
    log 51 of 48 and fail an arm whose launches were all correct. So the counts are taken before
    the launch and restored after it, whatever it raises."""
    kept = verify_trace_t1.take()
    try:
        return batch.execute(mesh, users, served, operations, output_memory=output_memory)
    finally:
        verify_trace_t1.take()
        for name, count in kept.items():
            verify_trace_t1.note(name, count)


def audit(root=DEFAULT_ROOT, users=batch.MAX_USERS, built_level=0, variant='A', diag=None):
    """Host-only description of what one K5-A launch would build; never opens a device."""
    kernels = load_kernels(root, built_level, variant=variant, diag=diag, unqualified=True)
    io, fp32 = cb_plan(variant)
    return dict(status='host-source-and-placement audit only; no compilation or hardware certification',
                transforms='compute = the served native prefix (sha256-checked, sliced at kernel_main) + '
                           'gdn_seq_block_compute.cpp; reader and writer new',
                level=built_level, variant=variant, diag=diag, qualified=kernels.qualified,
                qualified_triple=QUALIFIED.get(built_level),
                native_sha256=native.HASHES[NATIVE_COMPUTE],
                generated_sha256=sha256(kernels),
                src_tag={role: kernels.tag(role) for role in ROLES},
                compile_args={role: compile_args(role, kernels) for role in ROLES},
                users=users, workers_per_user=HEADS, workers=users * HEADS,
                core_shares=batch.core_shares(11, 10, users), rows_per_user=ROWS,
                cb_indices=sorted(list(io) + list(fp32)),
                cb_bytes_per_worker=cb_bytes(variant), served_cb_bytes_per_worker=SERVED_CB_BYTES,
                cb_owners={index: dict(name=entry[0], pages=entry[1], dtype=entry[2], producer=entry[3],
                                       consumer=entry[4]) for index, entry in sorted(plan(variant).items())},
                dram_page_reads_per_worker=8 + 4 + 4 + 2 + 16 + 4,
                launches_per_layer=1)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only host audit; never opens a device')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--users', type=int, default=batch.MAX_USERS)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--variant', choices=VARIANTS, default='A')
    parser.add_argument('--diag', choices=[value for value in DIAGNOSTICS if value], default=None)
    options = parser.parse_args()
    print(json.dumps(audit(options.root, options.users, options.level, options.variant, options.diag), indent=2))
