"""F1: the GDN conv-gates launch with its gate tiles on cores of their own, and a/b gathered by words
(QWEN_FAST_TP4_CONV_GATES_SPREAD, default off; tp4/w2).

WHY. gdn_decode_conv_gates puts EVERY gate tile on one extra core (gdn_conv_gates_program_factory.cpp: the gates "run on one
extra core when the grid has one spare"), whose reader gathers a and b one bfloat16 at a time with a barrier per page. The 64-row
block launch has two gate tiles, so that core runs two tiles one after the other: 42.3 us a launch against 20.6 us for the same
launch with one gate tile (the 4-row trace's), 48 launches a verify, two blocks a round.

WHAT. One generic_op per launch (a K-jit twin, no graft build, no change to the served .so) that runs the SERVED compute kernel
unchanged (read from the image tree and checked against its sha256) with two new data-movement kernels, generated from the served
reader and writer by two changes only (gdn_conv_gates_spread_reader.cpp / _writer.cpp):

  F1a  a gate-start runtime word. Gate tile k sits on core conv_cores + k, alone. The served compute loops `g_n` gate tiles after
       `n_inst` conv instances and re-initialises everything each time, so one tile on one core runs the same instructions as
       tile k of 2 on the served gate core; the served reader and writer start their gate loop at 0 (the source tile row is
       gi / Nvt, the output page is gi), which is why they cannot be reused: the new ones start it at the core's own tile.
  F1b  a and b are gathered by 32-bit words (a at columns 0-11 of its tile and b at columns 12-23, both even, so 6 words a row;
       b's run is split at the 16-column face boundary), and the gate's four reads (a, b, dt_bias, neg_exp_A) issue behind ONE
       barrier. The same bytes land in the same positions of the same zeroed tile.

Neither touches arithmetic: the conv instances are the served ones (the instance partition over cores is the served one whenever
the gate cores fit beside it, which they do at 64 rows: 160 conv instances, 80 cores of two, and 2 gate cores), every kernel instruction that
computes is the served binary's, and the data movement copies the same bytes to the same addresses. The audit (below) is the proof
on the card; exactness is "by construction", the audit is how a wrong construction is caught.

BINDING. gdn_block_conv_tp.stage (V1, QWEN_FAST_TP4_GDN_BLOCK_CONV) calls launch() in place of its one conv_gates(...) call when the
flag is set; with the flag unset nothing here is imported and the call is the served one, byte for byte. launch() answers None
(after one logged fall-back line) for any call it cannot take: another dtype, layout, placement, shape, a gate that is not read
from the projection itself, a tree whose conv-gates sources are not the pinned ones, a grid that cannot hold the gate cores. The
caller then makes the served call.

AUDIT (QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1, needs QWEN_FAST_TP4_VGLUE_AUDIT=1). Before the launch, the block windows are cloned
and the SERVED op runs on the same projection and the clones; after it, the launch's conv, beta, g and four advanced windows are
cloned and every pair is held on the layer's audit entries (labels 'spread conv gates ...', seven per launch). The replay's audit
(tp4_vglue.compare_entry, which gdn_conv_gates_spread.audit_round calls on exactly these entries) compares them on every chip as
int16 bit patterns (-0 and +0 differ). A launch that fell back leaves an audited layer without its seven entries and fails the
audit rather than passing on what is left.

TRACE. No buffer is allocated before the first capture for this launch: its outputs are allocated where the served op's are, inside
the capture, and it needs no scratch. Runtime argument lengths are fixed per role (reader 17, writer 11, compute 2): generic_op does
not hash them (f945486e).

Stdlib only at import (ttnn-free until launch), py 3.7.
"""

import hashlib
import os
from pathlib import Path

import tp4_vglue
import tp_shapes

HERE = Path(__file__).resolve().parent
FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'
AUDIT_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'
DEFAULT_ROOT = Path('/opt/tt-metal')
DIRECTORY = 'ttnn/cpp/ttnn/operations/transformer/gdn_conv_gates/device'

# The served op's four sources, as the pinned convolution graft ships them (gdn_direct_window_device.HASHES holds the same
# values). The compute kernel is run as is; the factory, reader and writer are what the new kernels were derived from and what the
# core plan below mirrors: any other tree is a fall-back, never a guess.
SERVED = {
    'gdn_conv_gates_program_factory.cpp': '0d02d3422ede6f814d9663ee4d739f16399b5bdf543c66f754db5f20127429f5',
    'kernels/compute/gdn_conv_gates.cpp': '9bdc2e38ef8415c2d4c9b4c7211972d7d61e39c6defed34acf19d7b9549391da',
    'kernels/dataflow/reader_gdn_conv_gates.cpp': '6662bbbefd64c13341af3839bb8543c38520fe0b7ed91c0b5f6a9fe5a55c418f',
    'kernels/dataflow/writer_gdn_conv_gates.cpp': '0e4193a0aa4eff3aaae2ce9b6fcf96eae282dffb3a65a59cf1eeb81882b9dce3',
}
COMPUTE_FILE = 'kernels/compute/gdn_conv_gates.cpp'
SOURCES = dict(reader='gdn_conv_gates_spread_reader.cpp', writer='gdn_conv_gates_spread_writer.cpp')
ROLES = ('reader', 'writer', 'compute')
# Runtime words per role: the length every core's list has, every launch. The kernels static_assert it against their RT_WORDS
# compile argument.
RT_WORDS = dict(reader=17, writer=11, compute=2)

K = 4                     # conv taps
TILE = 32
ONE_BITS = 0x3F800000     # 1.0f: the softplus beta and 1 / beta
TWENTY_BITS = 0x41A00000  # 20.0f: the softplus threshold
TILE_BYTES = dict(bf16=2048, fp32=4096)
# CB index -> (pages, dtype): the factory's plan (n_tiles x buffers, io bfloat16 here), except 13 (the gather scratch): four pages
# so one barrier covers a's and b's two source tiles each.
CB_PLAN = {0: (K * 2, 'bf16'), 1: (K * 2, 'bf16'), 2: (K, 'fp32'), 3: (2, 'fp32'), 4: (2, 'bf16'), 5: (K * 2, 'bf16'),
           6: (2, 'bf16'), 7: (2, 'bf16'), 8: (2, 'bf16'), 9: (2, 'bf16'), 10: (2, 'bf16'), 11: (2, 'bf16'),
           12: (2, 'bf16'), 13: (4, 'bf16')}

ENGAGED = '[PINDIAG] tp4 conv gates spread engaged'
FELL_BACK = '[PINDIAG] tp4 conv gates spread fell back'
AUDIT_MARKER = '[PINDIAG] tp4 conv gates spread audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 conv gates spread audit mismatch'
LABEL = 'spread conv gates '
ENTRIES_PER_LAUNCH = 3 + K     # conv, beta, g and the four advanced windows
RUNTIME_FILES = ('gdn_conv_gates_spread.py', 'gdn_conv_gates_spread_reader.cpp', 'gdn_conv_gates_spread_writer.cpp')


# ---- the flags ----

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def enabled(environ=None):
    """QWEN_FAST_TP4_CONV_GATES_SPREAD. Strict (unset or 0 off, 1 on, anything else raises). It is a four-card lever on the V1 block
    conv (it replaces that stage's one launch), so it raises at the pair and without QWEN_FAST_TP4_GDN_BLOCK_CONV=1."""
    source = os.environ if environ is None else environ
    on = _read(FLAG, source)
    if not on:
        if _read(AUDIT_FLAG, source):
            raise ValueError('%s=1 needs %s=1 (the audit compares the launch it replaces)' % (AUDIT_FLAG, FLAG))
        return False
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    if not tp4_vglue.enabled(tp4_vglue.GDN_BLOCK_CONV, source):
        raise ValueError('%s needs %s=1 (it replaces the block conv launch)' % (FLAG, tp4_vglue.GDN_BLOCK_CONV))
    return True


def audit_enabled(environ=None):
    """QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1: needs the lever and QWEN_FAST_TP4_VGLUE_AUDIT=1 (the layer's audit entries are the vglue
    audit's, freed with the retained records)."""
    source = os.environ if environ is None else environ
    if not enabled(source):
        return False
    if not _read(AUDIT_FLAG, source):
        return False
    if not tp4_vglue.audit_enabled(source):
        raise ValueError('%s=1 needs %s=1 (its entries ride that audit)' % (AUDIT_FLAG, tp4_vglue.AUDIT))
    return True


# ---- the core plan ----

def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def served_plan(grid_x, grid_y, bmax, channels):
    """(per_core, conv_cores, own_gate_core) of the served factory: conv instances over as many cores as needed, the gates on the next
    core when one is left."""
    n_conv = ceil_div(bmax, TILE) * (channels // TILE)
    cores = grid_x * grid_y
    per_core = ceil_div(n_conv, cores)
    conv_cores = ceil_div(n_conv, per_core)
    return per_core, conv_cores, conv_cores < cores


def plan(grid_x, grid_y, bmax, rows, channels, heads):
    """The F1 core plan for a launch over `bmax` state rows, `rows` active rows, `channels` conv channels and `heads` value heads:
    the core list in the factory's order (core c is (c // grid_y, c % grid_y)), each core's (start, n_inst, gate_start, g_n).
    Conv instances take the served partition when the gate cores fit after it; otherwise instances per core grow, by one at most, until they do
    (an instance's result does not depend on its core). Gate tile k is core conv_cores + k, one tile per core. ValueError when the
    grid cannot hold the gate tiles."""
    if min(grid_x, grid_y) < 1 or channels % TILE or rows < 1 or heads < 1 or bmax < 1:
        raise ValueError('bad conv gates geometry')
    n_conv = ceil_div(bmax, TILE) * (channels // TILE)
    n_gate = ceil_div(rows, TILE) * ceil_div(heads, TILE)
    cores = grid_x * grid_y
    per_core = ceil_div(n_conv, cores)
    while ceil_div(n_conv, per_core) + n_gate > cores:
        per_core += 1
        if per_core > n_conv:
            raise ValueError('a grid of %d cores cannot hold %d gate tiles beside one conv core' % (cores, n_gate))
    if per_core > served_plan(grid_x, grid_y, bmax, channels)[0] + 1:
        # the gate cores would cost every conv core more than one extra instance: the conv would slow by more than the gate saves
        raise ValueError('the %d gate tiles need conv cores of %d instances where the served plan has %d' % (
            n_gate, per_core, served_plan(grid_x, grid_y, bmax, channels)[0]))
    conv_cores = ceil_div(n_conv, per_core)
    work = []
    for index in range(conv_cores + n_gate):
        if index < conv_cores:
            start = index * per_core
            work.append(((index // grid_y, index % grid_y), start, min(per_core, n_conv - start), 0, 0))
        else:
            work.append(((index // grid_y, index % grid_y), 0, 0, index - conv_cores, 1))
    return dict(work=work, n_conv=n_conv, n_gate=n_gate, per_core=per_core, conv_cores=conv_cores, cores=len(work),
                served_per_core=served_plan(grid_x, grid_y, bmax, channels)[0])


def runtime_arguments(role, start, n_inst, gate_start, g_n, addresses):
    """One core's runtime words. `addresses` is the dict of buffer addresses this chip's launch uses (x, st0-st3, tap0-tap3, dt_bias,
    neg_exp_A, conv, beta, g). The length is RT_WORDS[role] for every core, conv or gate."""
    if role == 'reader':
        words = [start, n_inst, g_n, addresses['x'], *addresses['st'], *addresses['tap'], addresses['x'], addresses['x'],
                 addresses['dt_bias'], addresses['neg_exp_A'], gate_start]
    elif role == 'writer':
        words = [start, n_inst, g_n, addresses['conv'], *addresses['st'], addresses['beta'], addresses['g'], gate_start]
    elif role == 'compute':
        words = [n_inst, g_n]
    else:
        raise ValueError('Unknown kernel role')
    if len(words) != RT_WORDS[role]:
        raise AssertionError('runtime argument count of %s drifted from RT_WORDS' % role)
    return words


def compile_arguments(role, geometry, accessors):
    """Compile words of a role. `geometry` is dict(channels, heads, rows, x_rows, x_width, a_col, b_col); `accessors` the
    per-tensor accessor words in the kernel's order, flattened."""
    ct = geometry['channels'] // TILE
    nvt = ceil_div(geometry['heads'], TILE)
    if role == 'reader':
        wt = ceil_div(geometry['x_width'], TILE)
        return [K, ct, nvt, geometry['rows'], ceil_div(geometry['x_rows'], TILE), wt, 1, wt, geometry['a_col'], wt,
                geometry['b_col'], geometry['heads'], *accessors, RT_WORDS[role]]
    if role == 'writer':
        return [K, ct, nvt, *accessors, RT_WORDS[role]]
    if role == 'compute':
        return [K, nvt, ONE_BITS, TWENTY_BITS]
    raise ValueError('Unknown kernel role')


# ---- sources ----

class SourceMismatch(ValueError):
    pass


_SOURCES = {}


def sources(root=None, environ=None):
    """role -> kernel source text. The compute is the image tree's own file, accepted only when all four served files match their
    pins; the reader and writer are this directory's. Cached per root."""
    root = Path(root if root is not None else (os.environ if environ is None else environ).get('TT_METAL_HOME', str(DEFAULT_ROOT)))
    if str(root) in _SOURCES:
        return _SOURCES[str(root)]
    texts = {}
    for name, expected in SERVED.items():
        try:
            payload = (root / DIRECTORY / name).read_bytes()
        except OSError as error:
            raise SourceMismatch('the conv-gates source %s is not readable under %s (%s)' % (name, root, error.__class__.__name__))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise SourceMismatch('the conv-gates source %s is not the pinned one' % name)
        texts[name] = payload.decode()
    found = dict(compute=texts[COMPUTE_FILE])
    for role, name in SOURCES.items():
        data = (HERE / name).read_bytes().decode()
        if '\r' in data:
            raise ValueError('%s must be LF-only: its sha256 is its identity' % name)
        found[role] = data
    _SOURCES[str(root)] = found
    return found


def source_sha256():
    """The new kernels' sha256 (a host-side description; no tree needed)."""
    return {role: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for role, name in SOURCES.items()}


# ---- fall-back and engaged lines ----

_NOTED = set()


def log_line(message):
    tp4_vglue.log_line(message)


def fall_back(reason):
    """One FELL_BACK line per distinct reason per process; always returns None (the caller makes the served call)."""
    if reason not in _NOTED:
        _NOTED.add(reason)
        log_line('%s reason=%s' % (FELL_BACK, reason))
    return None


def note_engaged(rows, work_plan, audit):
    key = ('engaged', rows, work_plan['per_core'], work_plan['conv_cores'], work_plan['n_gate'])
    if key not in _NOTED:
        _NOTED.add(key)
        log_line('%s rows=%d conv_cores=%d gate_cores=%d per_core=%d served_per_core=%d audit=%d' % (
            ENGAGED, rows, work_plan['conv_cores'], work_plan['n_gate'], work_plan['per_core'], work_plan['served_per_core'],
            int(audit)))


# ---- the launch ----

def problem(operations, x, windows, taps, dt_bias, neg_exp_A, rows, channels, a_col, b_col):
    """Why this call cannot take the F1 launch (a short reason), or None."""
    if tp_shapes.chip_count() != 4:
        return 'not four cards'
    if len(windows) != K or len(taps) != K:
        return 'not %d windows and taps' % K
    tensors = [x, *windows, *taps, dt_bias, neg_exp_A]
    for tensor in tensors:
        if getattr(tensor, 'dtype', None) != operations.bfloat16 or getattr(tensor, 'layout', None) != operations.TILE_LAYOUT:
            return 'an operand is not bfloat16 TILE'
        try:
            if tensor.memory_config() != operations.DRAM_MEMORY_CONFIG:
                return 'an operand is not interleaved DRAM'
        except Exception:  # noqa: BLE001 - a diagnostic read
            return 'an operand placement is unreadable'
    xs = tuple(x.shape)
    if len(xs) != 3 or xs[0] != 1 or xs[2] < channels or xs[2] < b_col + int(dt_bias.shape[-1]) or xs[2] < a_col + int(dt_bias.shape[-1]):
        return 'x shape %r does not hold the channels and both gate windows' % (xs,)
    heads = int(dt_bias.shape[-1])
    if channels % TILE or not 1 <= rows <= xs[1] or heads < 1 or heads > TILE:
        return 'geometry rows=%r heads=%d channels=%d is not the 64-row block conv' % (rows, heads, channels)
    bmax = int(windows[0].shape[1])
    for window in windows:
        if tuple(window.shape) != (1, bmax, channels):
            return 'a window is not (1, %d, %d)' % (bmax, channels)
    if xs[1] > bmax:
        return 'x has more rows than the windows'
    for tap in taps:
        if tuple(tap.shape) != (1, 1, channels):
            return 'a tap is not (1, 1, %d)' % channels
    if tuple(dt_bias.shape) != (1, 1, heads) or tuple(neg_exp_A.shape) != (1, 1, heads):
        return 'dt_bias and neg_exp_A are not (1, 1, %d)' % heads
    return None


def _cores(operations, points):
    import verify_trace_t1

    return verify_trace_t1.rectangle_set(operations, points)


def _cb(operations, cores, index, pages, kind):
    data_format = operations.bfloat16 if kind == 'bf16' else operations.float32
    page = TILE_BYTES[kind]
    return operations.CBDescriptor(total_size=pages * page, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page,
                                                          tile=operations.TileDescriptor(operations.Tile([32, 32])))])


def build_program(operations, mesh, tensors, outputs, work_plan, geometry, texts):
    """The per-chip programs. `tensors` is dict(x, st (4), tap (4), dt_bias, neg_exp_A), `outputs` (conv, beta, g)."""
    chips = tp_shapes.chip_count()
    shards = {name: ([operations.get_device_tensors(value) for value in tensors[name]] if name in ('st', 'tap')
                     else operations.get_device_tensors(tensors[name])) for name in tensors}
    out_shards = [operations.get_device_tensors(value) for value in outputs]
    if (any(len(parts) != chips for parts in out_shards) or any(len(shards[name]) != chips for name in ('x', 'dt_bias', 'neg_exp_A'))
            or any(len(parts) != chips for name in ('st', 'tap') for parts in shards[name])):
        raise ValueError('every chip of the mesh is required for every conv gates tensor')
    points = [entry[0] for entry in work_plan['work']]
    cores = _cores(operations, points)
    cbs = [_cb(operations, cores, index, pages, kind) for index, (pages, kind) in sorted(CB_PLAN.items())]
    configs = dict(
        reader=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1,
                                                       noc=operations.NOC.RISCV_1_default),
        writer=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                       noc=operations.NOC.RISCV_0_default),
        compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                   math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    for chip in range(chips):
        local = dict(x=shards['x'][chip], st=[parts[chip] for parts in shards['st']], tap=[parts[chip] for parts in shards['tap']],
                     dt_bias=shards['dt_bias'][chip], neg_exp_A=shards['neg_exp_A'][chip])
        conv, beta, gate = (parts[chip] for parts in out_shards)
        addresses = dict(x=local['x'].buffer_address(), st=[value.buffer_address() for value in local['st']],
                         tap=[value.buffer_address() for value in local['tap']], dt_bias=local['dt_bias'].buffer_address(),
                         neg_exp_A=local['neg_exp_A'].buffer_address(), conv=conv.buffer_address(), beta=beta.buffer_address(),
                         g=gate.buffer_address())
        read_order = [local['x'], *local['st'], *local['tap'], local['x'], local['x'], local['dt_bias'], local['neg_exp_A']]
        write_order = [conv, *local['st'], beta, gate]
        descriptors = []
        for role in ROLES:
            accessors = []
            if role == 'reader':
                for value in read_order:
                    accessors.extend(operations.TensorAccessorArgs(value).get_compile_time_args())
            elif role == 'writer':
                for value in write_order:
                    accessors.extend(operations.TensorAccessorArgs(value).get_compile_time_args())
            runtime = operations.RuntimeArgs()
            for (horizontal, vertical), start, n_inst, gate_start, g_n in work_plan['work']:
                runtime[horizontal][vertical] = runtime_arguments(role, start, n_inst, gate_start, g_n, addresses)
            descriptor = operations.KernelDescriptor(kernel_source=texts[role],
                source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=compile_arguments(role, geometry, accessors), config=configs[role])
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=descriptors, cbs=cbs)
    return program


def launch(operations, mesh, x, windows, taps, dt_bias, neg_exp_A, rows, channels, a_col, b_col, held=None, entries=None):
    """(conv, beta, g) by the F1 launch, the block windows advanced in place as the served op advances them; or None after one
    logged line when this call cannot take it (the caller makes the served call; nothing is left allocated and nothing advanced).
    Under the audit it also runs the served op on cloned windows first and appends the seven audit entries to `entries` (and
    every tensor it holds to `held`)."""
    reason = problem(operations, x, windows, taps, dt_bias, neg_exp_A, rows, channels, a_col, b_col)
    if reason is not None:
        return fall_back(reason)
    try:
        texts = sources()
    except SourceMismatch as error:
        return fall_back(str(error))
    grid = mesh.compute_with_storage_grid_size()
    heads = int(dt_bias.shape[-1])
    bmax = int(windows[0].shape[1])
    try:
        work_plan = plan(grid.x, grid.y, bmax, rows, channels, heads)
    except ValueError as error:
        return fall_back(str(error))
    audit = audit_enabled()
    dram = operations.DRAM_MEMORY_CONFIG
    reference_windows, reference, mine = [], (), []
    produced = []
    try:
        if audit:
            from gdn_user_batch_conv import conv_gates

            # The served op on the same projection and clones of the not-yet-advanced windows, before the launch advances them.
            reference_windows = [operations.clone(window, memory_config=dram) for window in windows]
            reference = conv_gates(operations, x, reference_windows, taps, dt_bias, neg_exp_A, rows)
        geometry = dict(channels=channels, heads=heads, rows=rows, x_rows=int(x.shape[1]), x_width=int(x.shape[2]),
                        a_col=a_col, b_col=b_col)
        shapes = [(1, int(x.shape[1]), channels), (1, rows, heads), (1, rows, heads)]
        for shape in shapes:
            produced.append(operations.empty(shape, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh,
                                             memory_config=dram))
        tensors = dict(x=x, st=list(windows), tap=list(taps), dt_bias=dt_bias, neg_exp_A=neg_exp_A)
        program = build_program(operations, mesh, tensors, produced, work_plan, geometry, texts)
        io, seen = [], set()
        for value in (x, *windows, *taps, dt_bias, neg_exp_A, *produced):
            if id(value) not in seen:
                seen.add(id(value))
                io.append(value)
        operations.generic_op(io, program)
        if audit:
            mine = [operations.clone(value, memory_config=dram) for value in (*produced, *windows)]
            theirs = [*reference, *reference_windows]
            labels = ['conv', 'beta', 'g'] + ['window %d' % slot for slot in range(K)]
            for label, copy, served in zip(labels, mine, theirs, strict=True):
                if entries is not None:
                    entries.append(dict(label=LABEL + label, mine=copy, served=served))
                if held is not None:
                    held.extend([copy, served])
    except BaseException:
        # nothing reached `entries` or `held` yet (they are appended last): free what this call made, the served op's outputs too
        for value in (*produced, *mine, *reference, *reference_windows):
            try:
                operations.deallocate(value)
            except BaseException:
                pass
        raise
    note_engaged(rows, work_plan, audit)
    return tuple(produced)


# ---- the replay audit ----

_AUDIT = dict(rounds=0)


def audit_round(operations, records, round_number):
    """After a replay: every audited layer (verify_trace_t2.audit_layers, as the vglue audit) must carry the seven F1 entries and each
    must match its served twin on every chip, as int16 bits. Logs AUDIT_MARKER '<n> exact=True layers=<L> entries=<k>' or
    AUDIT_MISMATCH and raises. Returns the entries compared."""
    import verify_trace_t2

    layers = verify_trace_t2.audit_layers(round_number, len(records) or verify_trace_t2.GDN_LAYERS)
    compared, mismatches = 0, []
    for layer in layers:
        found = [entry for entry in tp4_vglue.audit_entries(records[layer][1]) if entry['label'].startswith(LABEL)]
        if len(found) % ENTRIES_PER_LAUNCH or not found:
            mismatches.append('layer %d: %d spread entries (a multiple of %d, at least one, required)' % (
                layer, len(found), ENTRIES_PER_LAUNCH))
        for entry in found:
            compared += 1
            mismatches.extend('layer %d %s' % (layer, text) for text in tp4_vglue.compare_entry(operations, entry))
    label = verify_trace_t2.layers_label(layers)
    if mismatches or not compared:
        message = '%s round=%d layers=%s %s' % (AUDIT_MISMATCH, round_number, label, '; '.join(mismatches[:4]) or 'nothing compared')
        log_line(message)
        raise AssertionError(message)
    _AUDIT['rounds'] += 1
    log_line('%s %d exact=True layers=%s entries=%d' % (AUDIT_MARKER, _AUDIT['rounds'], label, compared))
    return compared


def describe(grid_x=11, grid_y=10, bmax=64, rows=64, channels=2560, heads=12):
    """Host-only description of one launch (never opens a device)."""
    work_plan = plan(grid_x, grid_y, bmax, rows, channels, heads)
    return dict(status='host plan only; no compilation or hardware certification', per_core=work_plan['per_core'],
                served_per_core=work_plan['served_per_core'], conv_cores=work_plan['conv_cores'], gate_cores=work_plan['n_gate'],
                cores=work_plan['cores'], runtime_words=dict(RT_WORDS), new_kernels_sha256=source_sha256(),
                served_sha256=dict(SERVED))
