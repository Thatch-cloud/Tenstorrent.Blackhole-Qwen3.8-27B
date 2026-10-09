"""WC: the GDN layer's windows, convolution and gates in ONE launch (QWEN_FAST_TP4_GDN_WC, default off; tp4/w3).

WHAT IT REPLACES. With the V2 glue and the V1 block conv, one GDN layer's convolution costs six data-movement and conv launches around the
in-projection (fusion audit 2, G3-G8): the split of the block into per-user pieces, the T2 window build, the canon block, the window stack, the
(F1 or served) conv-gates launch and the unstack. This module runs ONE generic_op instead (a K-jit twin: no tt-metal rebuild, no graft), per chip:

  reader   per conv instance (tile row bt, channel tile cc): the projection tile of the block, its second user's half canonicalised in place
           (V2's rule, once), the three history rows each of the two users need (64-byte spans of the history pages), and the three windows
           st1..st3 BUILT IN PLACE in the conv circular buffer from them (T2's row map, gdn_conv_windows_packed.window_copies); the taps; and on
           the gate cores F1's gate gather (a and b by words) with the second user's half canonicalised; and on the z cores the z tiles.
  compute  the SERVED conv-gates compute kernel, unchanged (read from the image tree, all four served conv-gates sources pinned to their sha256,
           JIT-compiled here with the served compile arguments and compute config, as F1 does).
  writer   each output tile split into its two users' halves and written to the per-user tensors the recurrence, the commit and the retained
           records read: conv (1, 16, 2560), beta and g (1, 16, 12), z (1, 16, 1536), and the sixteen advanced windows (1, 16, 2560), as whole
           2 KiB pages (half 0 the user's rows, half 1 zero) composed in a scratch ring with local NoC copies.

WHY IT IS EXACT BY CONSTRUCTION, and what is not proved here. The compute kernel's inputs are byte-identical to the V1 chain's: the conv circular
buffer holds [st1, st2, st3, x] where x is the block tile with the second user's rows canonical (the canon block's tile) and st_s is the stack of
the two users' T2 windows (rows r < 4 - s from history r + s, the rest from the piece rows 0..11 + s; users' raw history, the odd user's canonical
piece), exactly the tile the window stack writes; the gate tiles hold a and b gathered from the canonical block. The outputs are the unstack's:
half 0 of the tile's user half, half 1 zero. Nothing is computed outside the served compute kernel; the new code moves bits. CPU emulation of the
kernels' loops against the V1 chain and the served per-user calls (test_gdn_wc, test_tp4_w3_wc) shows the same bytes on random and edge inputs. What
no CPU test shows is that the compiled kernels move those bytes: that is the card-M byte gate (optimisation/ttnn-op/gdn_wc) and, in a serving arm,
QWEN_FAST_TP4_GDN_WC_AUDIT.

BINDING. gdn_device_loop_state_tp._recurrence_glued asks `enabled()`; when it is on the V2 split launch is skipped (the pieces are made only when an
audit needs them or WC declines) and `stage()` runs after the state moves, handing run_user_batched_projected a BlockStage that carries the
sixteen windows (`prebuilt`). `stage()` answers None after ONE logged fall-back line for any call it cannot take, and the caller then runs the V2 split
and the V1 chain unchanged. With the flag unset nothing here is imported.

AUDIT (QWEN_FAST_TP4_GDN_WC_AUDIT=1, needs the lever and QWEN_FAST_TP4_VGLUE_AUDIT=1). The V1 stage's audit already holds a block launch's per-user
conv, beta, g, z and advanced windows against the SERVED per-user path (gdn_block_conv_tp.hold_served: the served T2 windows and one conv_gates call on
the user's own piece, in-trace, compared on every chip as int16 bit patterns after the replay). WC feeds the same machinery: in audit mode the pieces
are made (as V1 makes them) and hold_served compares WC's outputs with the served path's. The entries are labelled 'block conv user <u> wc <what>'
(the V1 prefix, so the V1 audit's expected-entry count still holds, plus ' wc ' so audit_round can tell them apart); an audited layer without its
WC entries fails the audit, as F1's does.

Stdlib only at import (ttnn-free until launch), py 3.7 syntax.
"""

import hashlib
import os
from pathlib import Path

import gdn_block_conv_tp as block
import gdn_conv_gates_spread as spread
import tp4_vglue
import tp_shapes

HERE = Path(__file__).resolve().parent
FLAG = 'QWEN_FAST_TP4_GDN_WC'
AUDIT_FLAG = 'QWEN_FAST_TP4_GDN_WC_AUDIT'
T2_FLAG = 'QWEN_FAST_VERIFY_T2'
T2_SKIP_FLAG = 'QWEN_FAST_VERIFY_T2_SKIP'
T2_WINDOWS_CUT = 'windows'

SOURCES = dict(reader='gdn_wc_reader.cpp', writer='gdn_wc_writer.cpp')
ROLES = ('reader', 'writer', 'compute')
RUNTIME_FILES = ('gdn_wc.py', 'gdn_wc_reader.cpp', 'gdn_wc_writer.cpp')
# Runtime words a role's per-core list holds, every core, every launch (generic_op does not hash their length; the kernels static_assert it
# against their RT_WORDS compile argument). reader and writer: [inst_start, n_inst, g_n, gate_start, z_start, n_z, 0, 0]; compute [n_inst, g_n].
RT_WORDS = dict(reader=8, writer=8, compute=2)
# The common runtime words (the buffer addresses; generic_op hashes their COUNT): the reader's [x, H[u][1..3] x 4 users, tap0..3, dt_bias,
# neg_exp_A], the writer's [conv x4, beta x4, g x4, z x4, windows 4 x 4].
READER_COMMON_WORDS = 19
WRITER_COMMON_WORDS = 32

TILE = 32
PAGE = 2048
HALF = 1024
FACE = 512
ROW = 32                  # bytes of one face row
HISTORY = 4
HISTORY_USED = 3          # histories 1..3: slot 0 of the window (history 0 .. 3) is never read by the conv
SLOTS = 4
K = spread.K
USER_ROWS = 16
SUPPORTED_USERS = (2, 4)  # an odd user count leaves half a tile to zero; the V1 chain handles it, WC declines it
SPAN = 64                 # bytes read of a history page row (two reads, faces 0 and 1)

ENGAGED = '[PINDIAG] tp4 gdn wc engaged'
FELL_BACK = '[PINDIAG] tp4 gdn wc fell back'
AUDIT_MARKER = '[PINDIAG] tp4 gdn wc audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 gdn wc audit mismatch'
WC_TAG = ' wc '          # inside every audit label WC makes: 'block conv user <u> wc <conv|beta|g|z|window s>'
ENTRIES_PER_USER = tp4_vglue.BLOCK_ENTRIES_PER_USER

# CB index -> (pages, dtype). 0-12 are the served factory's plan as F1 builds it; the rest are private to the data movement kernels:
# 13 the gate gather ring (four pages), 14 the history spans, 15 the z tile queue (reader -> writer), 16 the output staging ring (ten
# user pages and one zero source page).
CB_PLAN = dict(spread.CB_PLAN)
CB_GSRC, CB_HIST, CB_Z, CB_USTAGE = 13, 14, 15, 16
CB_PLAN[CB_HIST] = (1, 'bf16')
CB_PLAN[CB_Z] = (2, 'bf16')
CB_PLAN[CB_USTAGE] = (11, 'bf16')
USTAGE_PAGES = 10


# ---- the flags ----

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def t2_windows_cut(environ=None):
    """QWEN_FAST_VERIFY_T2=1 and the 'windows' cut not skipped (the T2 packed windows are the tensors WC writes; verify_trace_t2.cut)."""
    source = os.environ if environ is None else environ
    if source.get(T2_FLAG) != '1':
        return False
    skipped = set(name.strip() for name in source.get(T2_SKIP_FLAG, '').split(',') if name.strip())
    return T2_WINDOWS_CUT not in skipped


def enabled(environ=None):
    """QWEN_FAST_TP4_GDN_WC. Strict (unset or 0 off, 1 on, anything else raises). A four-card lever on the V1 block conv (it replaces that stage's
    chain and the T2 window build), so it raises at the pair, without QWEN_FAST_TP4_GDN_GLUE=1 and QWEN_FAST_TP4_GDN_BLOCK_CONV=1, and without
    the T2 windows cut. F1 (QWEN_FAST_TP4_CONV_GATES_SPREAD) may be set beside it: it stays the fall-back launch for a layer WC declines."""
    source = os.environ if environ is None else environ
    on = _read(FLAG, source)
    if not on:
        if _read(AUDIT_FLAG, source):
            raise ValueError('%s=1 needs %s=1 (the audit compares the launch it replaces)' % (AUDIT_FLAG, FLAG))
        return False
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    for lever in (tp4_vglue.GDN_GLUE, tp4_vglue.GDN_BLOCK_CONV):
        if not tp4_vglue.enabled(lever, source):
            raise ValueError('%s needs %s=1 (it replaces the V1 block conv chain)' % (FLAG, lever))
    if not t2_windows_cut(source):
        raise ValueError('%s needs %s=1 with the windows cut (the T2 packed windows are what it writes)' % (FLAG, T2_FLAG))
    if tp4_vglue.audit_enabled(source) and not _read(AUDIT_FLAG, source):
        # The V1 audit expects eight 'block conv user' entries per user and layer; a layer WC served has none unless WC's audit makes them.
        raise ValueError('%s=1 with %s=1 needs %s=1 (a layer WC serves has no V1 block entries to audit)' % (tp4_vglue.AUDIT, FLAG, AUDIT_FLAG))
    return True


def audit_enabled(environ=None):
    """QWEN_FAST_TP4_GDN_WC_AUDIT=1: needs the lever and QWEN_FAST_TP4_VGLUE_AUDIT=1 (the layer's audit entries are the vglue audit's)."""
    source = os.environ if environ is None else environ
    if not enabled(source):
        return False
    if not _read(AUDIT_FLAG, source):
        return False
    if not tp4_vglue.audit_enabled(source):
        raise ValueError('%s=1 needs %s=1 (its entries ride that audit)' % (AUDIT_FLAG, tp4_vglue.AUDIT))
    return True


# ---- geometry and the core plan ----

def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def geometry(users, width, found=None):
    """The launch's constants for `users` packed users over a projection `width` columns wide: tile rows, conv instances, channel tiles,
    gate tiles, z tile columns, the projection's width in tiles."""
    found = tp_shapes.active() if found is None else found
    if users not in SUPPORTED_USERS:
        raise ValueError('WC takes %s users, not %r' % (' or '.join(str(count) for count in SUPPORTED_USERS), users))
    rows = users * USER_ROWS
    tile_rows = ceil_div(rows, TILE)
    channel_tiles = found.gdn_qkv // TILE
    z_first = found.gdn_qkv // TILE
    z_columns = (found.gdn_a_col - found.gdn_qkv) // TILE
    return dict(users=users, rows=rows, tile_rows=tile_rows, channel_tiles=channel_tiles, instances=tile_rows * channel_tiles,
                heads=found.gdn_nv, gate_tiles=tile_rows * ceil_div(found.gdn_nv, TILE), gate_columns=ceil_div(found.gdn_nv, TILE),
                z_first=z_first, z_columns=z_columns, z_tiles=tile_rows * z_columns, width_tiles=ceil_div(width, TILE),
                a_col=found.gdn_a_col, b_col=found.gdn_b_col, channels=found.gdn_qkv)


def plan(grid_x, grid_y, users, width=None, found=None):
    """The core plan: F1's (conv instances over the served partition, one gate tile per core after the conv cores) plus the z tiles over the cores that
    are left. `work` lists, in core order (core c is (c // grid_y, c % grid_y)), dict(point, start, n_inst, gate_start, g_n, z_start, n_z).
    ValueError when the grid cannot hold it."""
    found = tp_shapes.active() if found is None else found
    shape = geometry(users, found.gdn_qkvzab_padded if width is None else width, found)
    base = spread.plan(grid_x, grid_y, shape['rows'], shape['rows'], shape['channels'], shape['heads'])
    spare = grid_x * grid_y - base['cores']
    if spare < 1:
        raise ValueError('no core is left for the %d z tiles' % shape['z_tiles'])
    z_cores = min(spare, shape['z_tiles'])
    each, extra = divmod(shape['z_tiles'], z_cores)
    work, z_next = [], 0
    for index, (point, start, n_inst, gate_start, g_n) in enumerate(base['work']):
        work.append(dict(point=point, start=start, n_inst=n_inst, gate_start=gate_start, g_n=g_n, z_start=0, n_z=0))
    for offset in range(z_cores):
        index = base['cores'] + offset
        count = each + (1 if offset < extra else 0)
        work.append(dict(point=(index // grid_y, index % grid_y), start=0, n_inst=0, gate_start=0, g_n=0, z_start=z_next, n_z=count))
        z_next += count
    return dict(work=work, shape=shape, conv_cores=base['conv_cores'], gate_cores=base['n_gate'], z_cores=z_cores, per_core=base['per_core'],
                served_per_core=base['served_per_core'], cores=len(work), n_conv=base['n_conv'])


def reader_runtime(item):
    """One core's reader (and writer) words: [inst_start, n_inst, g_n, gate_start, z_start, n_z, 0, 0]."""
    return [item['start'], item['n_inst'], item['g_n'], item['gate_start'], item['z_start'], item['n_z'], 0, 0]


def compute_runtime(item):
    return [item['n_inst'], item['g_n']]


def runtime_arguments(role, item):
    if role in ('reader', 'writer'):
        words = reader_runtime(item)
    elif role == 'compute':
        words = compute_runtime(item)
    else:
        raise ValueError('Unknown kernel role')
    if len(words) != RT_WORDS[role]:
        raise AssertionError('runtime argument count of %s drifted from RT_WORDS' % role)
    return words


def reader_common(addresses, users):
    """The reader's common words: x, then H[u][1..3] for u in 0..3 (zeros past `users`), the four taps, dt_bias, neg_exp_A."""
    words = [addresses['x']]
    for user in range(4):
        for index in range(1, HISTORY):
            words.append(addresses['history'][user][index] if user < users else 0)
    words.extend(addresses['tap'])
    words.extend([addresses['dt_bias'], addresses['neg_exp_A']])
    if len(words) != READER_COMMON_WORDS:
        raise AssertionError('reader common words drifted')
    return words


def writer_common(addresses, users):
    """The writer's common words: conv x4, beta x4, g x4, z x4, then the windows 16 + 4 u + s (zeros past `users`)."""
    words = []
    for name in ('conv', 'beta', 'g', 'z'):
        words.extend(addresses[name][user] if user < users else 0 for user in range(4))
    for user in range(4):
        words.extend(addresses['window'][user][slot] if user < users else 0 for slot in range(SLOTS))
    if len(words) != WRITER_COMMON_WORDS:
        raise AssertionError('writer common words drifted')
    return words


def compile_arguments(role, shape, accessors, found=None):
    """Compile words of a role. Reader: [K, Ct, Nvt, users, Wt, ACOL, BCOL, NV, ZFIRST, ZCOLS] + accessor args (projection, history, small) +
    RT_WORDS. Writer: [K, Ct, Nvt, users, ZCOLS] + accessor args (output) + RT_WORDS. Compute: the served [K, Nvt, ONE_BITS, TWENTY_BITS]."""
    if role == 'reader':
        return [K, shape['channel_tiles'], shape['gate_columns'], shape['users'], shape['width_tiles'], shape['a_col'], shape['b_col'],
                shape['heads'], shape['z_first'], shape['z_columns'], *accessors, RT_WORDS[role]]
    if role == 'writer':
        return [K, shape['channel_tiles'], shape['gate_columns'], shape['users'], shape['z_columns'], *accessors, RT_WORDS[role]]
    if role == 'compute':
        return [K, shape['gate_columns'], spread.ONE_BITS, spread.TWENTY_BITS]
    raise ValueError('Unknown kernel role')


def cb_bytes():
    return sum(pages * spread.TILE_BYTES[kind] for pages, kind in CB_PLAN.values())


# ---- the destination map: which page of which tensor each unit of work writes (the planner invariant the tests hold) ----

def output_pages(item, shape):
    """Every (tensor name, user, slot or 0, page) a core's work writes: the writer's loops, as data."""
    written = []
    for inst in range(item['start'], item['start'] + item['n_inst']):
        bt, cc = divmod(inst, shape['channel_tiles'])
        for half in range(2):
            user = 2 * bt + half
            written.append(('conv', user, 0, cc))
            for slot in range(SLOTS):
                written.append(('window', user, slot, cc))
    for gi in range(item['gate_start'], item['gate_start'] + item['g_n']):
        bt, t = divmod(gi, shape['gate_columns'])
        for half in range(2):
            user = 2 * bt + half
            written.append(('beta', user, 0, t))
            written.append(('g', user, 0, t))
    for zi in range(item['z_start'], item['z_start'] + item['n_z']):
        bt, column = divmod(zi, shape['z_columns'])
        for half in range(2):
            written.append(('z', 2 * bt + half, 0, column))
    return written


def input_pages(item, shape):
    """Every projection page a core's reader reads (the planner invariant: each is read by the work that needs it)."""
    pages = []
    for inst in range(item['start'], item['start'] + item['n_inst']):
        bt, cc = divmod(inst, shape['channel_tiles'])
        pages.append(bt * shape['width_tiles'] + cc)
    for gi in range(item['gate_start'], item['gate_start'] + item['g_n']):
        bt = gi // shape['gate_columns']
        t = gi % shape['gate_columns']
        h0 = t * TILE
        h1 = min(h0 + TILE, shape['heads'])
        for column in (shape['a_col'], shape['b_col']):
            for page in range((column + h0) // TILE, (column + h1 - 1) // TILE + 1):
                pages.append(bt * shape['width_tiles'] + page)
    for zi in range(item['z_start'], item['z_start'] + item['n_z']):
        bt, column = divmod(zi, shape['z_columns'])
        pages.append(bt * shape['width_tiles'] + shape['z_first'] + column)
    return pages


# ---- sources ----

class SourceMismatch(ValueError):
    pass


_SOURCES = {}


def sources(root=None, environ=None):
    """role -> kernel source text. The compute is the image tree's own conv-gates file, accepted only when all four served files match their pins
    (F1's pins); the reader and writer are this directory's. Cached per root."""
    root = Path(root if root is not None else (os.environ if environ is None else environ).get('TT_METAL_HOME', str(spread.DEFAULT_ROOT)))
    if str(root) in _SOURCES:
        return _SOURCES[str(root)]
    texts = {}
    for name, expected in spread.SERVED.items():
        try:
            payload = (root / spread.DIRECTORY / name).read_bytes()
        except OSError as error:
            raise SourceMismatch('the conv-gates source %s is not readable under %s (%s)' % (name, root, error.__class__.__name__))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise SourceMismatch('the conv-gates source %s is not the pinned one' % name)
        texts[name] = payload.decode()
    found = dict(compute=texts[spread.COMPUTE_FILE])
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


def kernel_defines(role, canon_denorm=True):
    """The defines of a role: the canonical rule as gdn_rows_dma_tp launches it (CANON_DENORM, on), and the source sha as the JIT key. The compute
    kernel gets none (it is the served source)."""
    if role == 'compute':
        return []
    sha = hashlib.sha256((HERE / SOURCES[role]).read_bytes()).hexdigest()[:8]
    return [('CANON_DENORM', '1' if canon_denorm else '0'), ('WC_SRC_SHA', '0x' + sha)]


# ---- fall-back and engaged lines ----

_NOTED = set()


def log_line(message):
    tp4_vglue.log_line(message)


def fall_back(reason):
    """One FELL_BACK line per distinct reason per process; always returns None (the caller runs the V1 chain)."""
    if reason not in _NOTED:
        _NOTED.add(reason)
        log_line('%s reason=%s' % (FELL_BACK, reason))
    return None


def engaged_line(work_plan, audit):
    return '%s rows=%d conv_cores=%d gate_cores=%d z_cores=%d per_core=%d served_per_core=%d audit=%d' % (
        ENGAGED, work_plan['shape']['rows'], work_plan['conv_cores'], work_plan['gate_cores'], work_plan['z_cores'], work_plan['per_core'],
        work_plan['served_per_core'], int(audit))


def note_engaged(work_plan, audit):
    key = ('engaged', work_plan['shape']['rows'], work_plan['per_core'], work_plan['conv_cores'], work_plan['gate_cores'], work_plan['z_cores'])
    if key not in _NOTED:
        _NOTED.add(key)
        log_line(engaged_line(work_plan, audit))


# ---- the launch ----

def _interleaved(operations, tensor):
    try:
        return tensor.memory_config() in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG)
    except Exception:  # noqa: BLE001 - a diagnostic read
        return False


def _dram(operations, tensor):
    try:
        return tensor.memory_config() == operations.DRAM_MEMORY_CONFIG
    except Exception:  # noqa: BLE001
        return False


def problem(operations, projected, histories, taps, dt_bias, neg_exp_A, users, found=None):
    """Why this call cannot take the WC launch (a short reason), or None. Pure checks: nothing is allocated."""
    found = tp_shapes.active() if found is None else found
    if tp_shapes.chip_count() != 4:
        return 'not four cards'
    if users not in SUPPORTED_USERS:
        return '%r users (WC takes %s)' % (users, ' or '.join(str(count) for count in SUPPORTED_USERS))
    if len(histories) != users or any(len(user) != HISTORY for user in histories):
        return 'not %d histories per user' % HISTORY
    if len(taps) != K:
        return 'not %d taps' % K
    shape = tuple(projected.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] != users * USER_ROWS:
        return 'projection shape %r is not (1, %d, W)' % (shape, users * USER_ROWS)
    if shape[2] not in (found.gdn_qkvzab, found.gdn_qkvzab_padded):
        return 'projection width %d is not %d or %d' % (shape[2], found.gdn_qkvzab, found.gdn_qkvzab_padded)
    flat = [projected, *[value for user in histories for value in user], *taps, dt_bias, neg_exp_A]
    for tensor in flat:
        if getattr(tensor, 'dtype', None) != operations.bfloat16 or getattr(tensor, 'layout', None) != operations.TILE_LAYOUT:
            return 'an operand is not bfloat16 TILE'
        if not _interleaved(operations, tensor):
            return 'an operand is not interleaved DRAM or L1'
    if len({_dram(operations, value) for user in histories for value in user}) != 1:
        return 'history placements are mixed'
    for user in histories:
        for state in user:
            if tuple(state.shape) != (1, 1, found.gdn_qkv):
                return 'a history is not (1, 1, %d)' % found.gdn_qkv
    for tap in taps:
        if tuple(tap.shape) != (1, 1, found.gdn_qkv):
            return 'a tap is not (1, 1, %d)' % found.gdn_qkv
    for small in (dt_bias, neg_exp_A):
        if tuple(small.shape) != (1, 1, found.gdn_nv):
            return 'dt_bias and neg_exp_A are not (1, 1, %d)' % found.gdn_nv
    if not all(_dram(operations, value) for value in (*taps, dt_bias, neg_exp_A)):
        return 'a tap or gate parameter is not in DRAM'
    if found.gdn_nv > TILE:
        return 'more value heads than one gate tile'
    return None


def _cores(operations, points):
    import verify_trace_t1

    return verify_trace_t1.rectangle_set(operations, points)


def _cb(operations, cores, index, pages, kind):
    return spread._cb(operations, cores, index, pages, kind)


def class_arguments(operations, shard_lists, chips):
    """Per chip, the one accessor-args word list every member of a class yields; None when a class is not uniform."""
    out = []
    for chip in range(chips):
        found = {tuple(operations.TensorAccessorArgs(parts[chip]).get_compile_time_args()) for parts in shard_lists}
        if len(found) != 1:
            return None
        out.append(found.pop())
    return out


def build_program(operations, mesh, inputs, outputs, work_plan, texts, chips=None):
    """The per-chip programs. `inputs` is dict(x, history (users lists of four), tap (4), dt_bias, neg_exp_A); `outputs` dict(conv, beta, g, z
    (one per user each), window (users lists of four)). Returns the MeshProgramDescriptor; raises ValueError for a non-uniform class."""
    chips = tp_shapes.chip_count() if chips is None else chips
    shape = work_plan['shape']
    users = shape['users']
    x_shards = operations.get_device_tensors(inputs['x'])
    history_flat = [value for user in inputs['history'] for value in user]
    history_shards = [operations.get_device_tensors(value) for value in history_flat]
    small = [*inputs['tap'], inputs['dt_bias'], inputs['neg_exp_A']]
    small_shards = [operations.get_device_tensors(value) for value in small]
    out_flat = [*outputs['conv'], *outputs['beta'], *outputs['g'], *outputs['z'], *[w for user in outputs['window'] for w in user]]
    out_shards = [operations.get_device_tensors(value) for value in out_flat]
    if any(len(parts) != chips for parts in [x_shards, *history_shards, *small_shards, *out_shards]):
        raise ValueError('every chip of the mesh is required for every WC tensor')
    projection_args = class_arguments(operations, [x_shards], chips)
    history_args = class_arguments(operations, history_shards, chips)
    small_args = class_arguments(operations, small_shards, chips)
    output_args = class_arguments(operations, out_shards, chips)
    if None in (projection_args, history_args, small_args, output_args):
        raise ValueError('a WC tensor class has different accessor arguments')
    points = [item['point'] for item in work_plan['work']]
    cores = _cores(operations, points)
    cbs = [_cb(operations, cores, index, pages, kind) for index, (pages, kind) in sorted(CB_PLAN.items())]
    configs = dict(
        reader=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default),
        writer=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default),
        compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    for chip in range(chips):
        addresses = dict(
            x=x_shards[chip].buffer_address(),
            history=[[history_shards[HISTORY * user + index][chip].buffer_address() for index in range(HISTORY)] for user in range(users)],
            tap=[small_shards[index][chip].buffer_address() for index in range(K)],
            dt_bias=small_shards[K][chip].buffer_address(), neg_exp_A=small_shards[K + 1][chip].buffer_address())
        base = 0
        out_addresses = {}
        for name, count in (('conv', users), ('beta', users), ('g', users), ('z', users)):
            out_addresses[name] = [out_shards[base + index][chip].buffer_address() for index in range(count)]
            base += count
        out_addresses['window'] = [[out_shards[base + SLOTS * user + slot][chip].buffer_address() for slot in range(SLOTS)] for user in range(users)]
        accessors = dict(reader=[*projection_args[chip], *history_args[chip], *small_args[chip]], writer=list(output_args[chip]), compute=[])
        descriptors = []
        for role in ROLES:
            runtime = operations.RuntimeArgs()
            for item in work_plan['work']:
                horizontal, vertical = item['point']
                runtime[horizontal][vertical] = runtime_arguments(role, item)
            options = dict(kernel_source=texts[role], source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                           compile_time_args=compile_arguments(role, shape, accessors[role]), config=configs[role])
            defines = kernel_defines(role)
            if defines:
                # a constructor argument, as gdn_rows_dma_tp.launch and gdn_conv_windows_packed pass it; the served compute kernel gets none
                options['defines'] = defines
            descriptor = operations.KernelDescriptor(**options)
            descriptor.runtime_args = runtime
            if role == 'reader':
                descriptor.common_runtime_args = reader_common(addresses, users)
            elif role == 'writer':
                descriptor.common_runtime_args = writer_common(out_addresses, users)
            descriptors.append(descriptor)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=descriptors, cbs=cbs)
    return program


def launch(operations, mesh, projected, histories, taps, dt_bias, neg_exp_A, chips=None):
    """The per-user tensors by ONE launch: dict(conv, beta, g, z (one (1, 16, .) tensor per user each), window (per user, four (1, 16, 2560))), all freshly
    allocated in DRAM; or None after one logged line when this call cannot take it (nothing is left allocated). `chips` is the mesh's chip count when it
    is not the configured width (the one-card probe)."""
    found = tp_shapes.active()
    users = len(histories)
    reason = problem(operations, projected, histories, taps, dt_bias, neg_exp_A, users, found)
    if reason is not None:
        return fall_back(reason)
    try:
        texts = sources()
    except SourceMismatch as error:
        return fall_back(str(error))
    grid = mesh.compute_with_storage_grid_size()
    try:
        work_plan = plan(grid.x, grid.y, users, int(projected.shape[-1]), found)
    except ValueError as error:
        return fall_back(str(error))
    dram = operations.DRAM_MEMORY_CONFIG
    made = []

    def make(shape):
        tensor = operations.empty(shape, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh, memory_config=dram)
        made.append(tensor)
        return tensor

    try:
        outputs = dict(
            conv=[make((1, USER_ROWS, found.gdn_qkv)) for user in range(users)],
            beta=[make((1, USER_ROWS, found.gdn_nv)) for user in range(users)],
            g=[make((1, USER_ROWS, found.gdn_nv)) for user in range(users)],
            z=[make((1, USER_ROWS, found.gdn_a_col - found.gdn_qkv)) for user in range(users)],
            window=[[make((1, USER_ROWS, found.gdn_qkv)) for slot in range(SLOTS)] for user in range(users)])
        inputs = dict(x=projected, history=[list(user) for user in histories], tap=list(taps), dt_bias=dt_bias, neg_exp_A=neg_exp_A)
        try:
            program = build_program(operations, mesh, inputs, outputs, work_plan, texts, chips)
        except ValueError as error:
            for tensor in made:
                operations.deallocate(tensor)
            return fall_back(str(error))
        io, seen = [], set()
        for value in (projected, *[w for user in histories for w in user], *taps, dt_bias, neg_exp_A, *made):
            if id(value) not in seen:
                seen.add(id(value))
                io.append(value)
        operations.generic_op(io, program)
    except BaseException:
        for tensor in made:
            try:
                operations.deallocate(tensor)
            except BaseException:
                pass
        raise
    note_engaged(work_plan, audit_enabled())
    return outputs


# ---- the stage: what run_user_batched_projected takes in place of the T2 window build and the V1 block stage ----

def stage(mesh, projected, histories, taps, dt_bias, neg_exp_A, operations, groups=None, note_fallback=None, chips=None):
    """A gdn_block_conv_tp.BlockStage (with `.windows`, one list of four per user) from ONE launch, or None (one logged line; nothing allocated, nothing
    advanced) when the launch cannot serve this layer: the caller then runs the V2 split and the V1 chain. `groups` is the (piece, initial, history)
    triple per user the audit needs (QWEN_FAST_TP4_GDN_WC_AUDIT: gdn_block_conv_tp.hold_served builds the served per-user path from the pieces)."""
    made = launch(operations, mesh, projected, histories, taps, dt_bias, neg_exp_A, chips)
    if made is None:
        if note_fallback is not None:
            note_fallback('WC declined this layer')
        return None
    users = len(histories)
    owned = [value for name in ('conv', 'beta', 'g', 'z') for value in made[name]]
    owned.extend(value for user in made['window'] for value in user)
    held, entries = [], []
    try:
        if audit_enabled():
            if groups is None:
                raise ValueError('the WC audit needs the pieces (groups) to build the served path from')
            entries = block.hold_served(mesh, groups, made['window'], made['conv'], made['beta'], made['g'], made['z'], taps, dt_bias, neg_exp_A,
                                        operations, held)
            for user, own in enumerate(entries):
                prefix = 'block conv user %d ' % user
                for entry in own:
                    if entry['label'].startswith(prefix):
                        entry['label'] = prefix + WC_TAG.strip() + ' ' + entry['label'][len(prefix):]
    except BaseException:
        from tp_addresses import release_owned

        release_owned(operations, owned + held)
        raise
    per_user = [((made['conv'][user], made['beta'][user], made['g'][user]), made['z'][user]) for user in range(users)]
    stage_ = block.BlockStage(per_user, [], held, entries)
    stage_.windows = made['window']
    return stage_


# ---- the replay audit ----

_AUDIT = dict(rounds=0)


def audit_round(operations, records, round_number):
    """After a replay: every audited layer (verify_trace_t2.audit_layers, as the vglue audit) must carry the WC entries of every user (a positive
    multiple of ENTRIES_PER_USER labelled 'block conv user <u> wc ...') and each must match its served twin on every chip, as int16 bits. Logs
    AUDIT_MARKER '<n> exact=True layers=<L> entries=<k>' or AUDIT_MISMATCH and raises. Returns the entries compared."""
    import verify_trace_t2

    layers = verify_trace_t2.audit_layers(round_number, len(records) or verify_trace_t2.GDN_LAYERS)
    compared, mismatches = 0, []
    for layer in layers:
        found = [entry for entry in tp4_vglue.audit_entries(records[layer][1])
                 if entry['label'].startswith('block conv user ') and WC_TAG in entry['label']]
        if not found or len(found) % ENTRIES_PER_USER:
            mismatches.append('layer %d: %d wc entries (a positive multiple of %d required)' % (layer, len(found), ENTRIES_PER_USER))
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


def describe(grid_x=11, grid_y=10, users=4):
    """Host-only description of one launch (never opens a device)."""
    work_plan = plan(grid_x, grid_y, users)
    return dict(status='host plan only; no compilation or hardware certification', per_core=work_plan['per_core'],
                served_per_core=work_plan['served_per_core'], conv_cores=work_plan['conv_cores'], gate_cores=work_plan['gate_cores'],
                z_cores=work_plan['z_cores'], cores=work_plan['cores'], runtime_words=dict(RT_WORDS), common_words=dict(
                    reader=READER_COMMON_WORDS, writer=WRITER_COMMON_WORDS), cb_bytes=cb_bytes(), new_kernels_sha256=source_sha256(),
                served_sha256=dict(spread.SERVED))
