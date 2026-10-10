"""F-B1: the page-parallel K/V cache writer for the packed 64-row verify block (QWEN_FAST_KV_PAGE_WRITER, default off; tp4/fx-wp2).

WHY. Per full-attention layer the served block writes its K and V rows with two sharded-to-interleaved copies and two chained launches of
the audited ordered kernel (packed_ordered_cache.update_chained, one semaphore chain per packed user, 16 rows a chain): 4.4 + 46.3 + 4.4 +
45.0 us a layer, 1.64 ms a pass (fusion plan, v676 census rows A4-A7). A chain step is a full DRAM round trip (read the 8 cache tiles of the
row's tile row, untilize, insert ONE row, tilize, write) at about 2.9 us, so the 16 rows of one user cost 16 round trips that all land on
one or two tile rows.

WHAT. ONE generic_op for the layer's K AND V: a unit is (cache, packed user, tile row, WT column tiles) and does the read-modify-write of
its tiles once, with every row of the user that targets that tile row inserted. 64 cores (an 8x8 rectangle, WT = 2), each one DRAM read
of its cache tiles, one write, and the update rows read straight from the prepared K/V's L1 shards (no copy to interleaved DRAM).
kv_page_writer_tp4.cpp has the kernels; the COMPUTE kernel is the served writer's own, run unchanged (ordered_cache.load_kernels(...)
['compute']): the unpack, untilize, tilize and pack are the served LLK calls.

EXACTNESS. The served chain applies pack(unpack(.)) once per row; this applies it once per tile. They are the same bytes exactly when
bfloat8_b pack(unpack(x)) is the identity on every block the packer writes, and then only if no two users write one (page, tile row)
(verify_trace_t2.kv_conflict, the rule the chained writer already needs). Whether the identity holds on the hardware packer is plan unknown U1,
settled only by a card: ordered_writer_tp4_card_test.py's page64 arm writes the same pages twice, writes partial pages, and compares the
COMPLETE cache with the served writer's after every step (both caches, both sources, the shapes the stack writes, page widths 2,052 and
4,096). record_ordered_writer_evidence_tp4.py puts the result in ordered_writer_evidence_tp4.json under 'page_writer', bound to this
design (design_signature: the kernel's sha256, the pinned compute kernel's, the geometry). THIS LEVER REFUSES TO ENGAGE unless that record
is present at its pin and matches the live design, the live page-table width and the live source (evidence_problems): a repository
without the card run keeps the chained writer, and says so once (FELL_BACK).

SHAPES. Four packed users of 16 consecutive rows (packed_host_inputs: positions start + arange, every row of a user carries the user's one
table). 16 consecutive positions touch at most two 32-row tile rows ("slots"). Anything else is refused at construction and the chained
writer runs: another row count, spans that are not four 16-row users (or the warm forward's single span), launch rows 32, a grid under 8x8,
a pair, a page width the record does not cover. The warm forward (one span over all 64 rows, placeholders that all sit on one tile row)
runs the ORDERED mode: the same program, group g waiting for group g-1's unit, which is the served single chain's order.

BINDING. packed_ordered_cache.ChainedOrderedCacheWriter, when QWEN_FAST_KV_PAGE_WRITER is set, builds one KVPageWriter and hands it each
call: the first call of a layer (K) is held, the second (V) launches both. The model calls K then V back to back and frees the prepared
tensors after the second (qwen36_attention_tp._decode_from_prep), which is the contract this relies on; a K call followed by another K
call, or a forward that ends on a held call, is a bug that raises. Flag unset: nothing here is imported and the chained writer is byte for
byte what it was (test_kv_page_writer_tp4 holds it).

AUDIT (QWEN_FAST_KV_PAGE_WRITER_AUDIT=1, a correctness arm, never timed). Around each page launch: audit prep copies the unit tiles
of the real cache into a small shadow cache and writes shadow positions; the SERVED chained writer runs on the shadow with the same prepared
K/V; the page launch runs on the real cache; audit check counts the 32-bit words of every unit that differ between the real cache and the
shadow. audit_round reads the counters after a replay and logs '<n> exact=True' or the mismatch line.

Stdlib only at import (ttnn-free until launch), py 3.7 syntax.
"""

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAG = 'QWEN_FAST_KV_PAGE_WRITER'
AUDIT_FLAG = 'QWEN_FAST_KV_PAGE_WRITER_AUDIT'
WT_FLAG = 'QWEN_FAST_KV_PAGE_WRITER_WT'

ENGAGED = '[PINDIAG] tp4 kv page writer engaged'
FELL_BACK = '[PINDIAG] tp4 kv page writer fell back'
AUDIT_MARKER = '[PINDIAG] tp4 kv page writer audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 kv page writer audit mismatch'

SOURCE_NAME = 'kv_page_writer_tp4.cpp'
RUNTIME_FILES = ('kv_page_writer_tp4.py', 'kv_page_writer_tp4.cpp', 'kv_page_writer_tp4_smoke.py')

# ---- the block's geometry ----
GROUPS = 4              # packed users in the 64-row block (M3)
GROUP_ROWS = 16         # rows per user
BLOCK_ROWS = GROUPS * GROUP_ROWS
SLOTS = 2               # tile rows 16 consecutive positions can touch
CACHES = 2              # K and V
ROW_TILES = 8           # column tiles of a 256-wide K or V row
TOKENS_PER_PAGE = 64
TILE_ROWS = 32
HEADS = 1               # KV heads per chip at four cards
CACHE_TILE = 1088       # a bfloat8_b tile
TILE = 2048             # a bfloat16 tile
DEFAULT_WT = 2
WIDTHS = (1, 2, 4, 8)
GRID_WIDTH = 8          # the 64-core layout is the served chained writer's 8x8 rectangle

CB_CACHE, CB_IN, CB_SCRATCH, CB_OUT, CB_UNTILIZED, CB_UNTILIZED2, CB_UNTILIZED_IN, CB_META = 0, 1, 2, 16, 24, 25, 26, 27
META_BYTES = 64
AUDIT_BLOCKS = 16       # shadow cache blocks (units use GROUPS * SLOTS = 8; the served writer needs blocks >= page-table width)
AUDIT_TABLE_WIDTH = 16
AUDIT_RECORD_WORDS = 16

READER_WORDS = 8 + 2 * GROUP_ROWS
WRITER_WORDS = 8
AUDIT_PREP_WORDS = 10
AUDIT_CHECK_WORDS = 9
ROLES = ('reader', 'writer', 'compute')
ROLE_DEFINES = dict(reader='KVPW_ROLE_READER', writer='KVPW_ROLE_WRITER', audit_prep='KVPW_ROLE_AUDIT_PREP',
                    audit_check='KVPW_ROLE_AUDIT_CHECK')
NEGATIVE_CONTROLS = dict(drop='KVPW_NEG_DROP', slot='KVPW_NEG_SLOT')
SOURCES = ('dram', 'l1')


class Unsupported(ValueError):
    """The page writer cannot serve this block (shape, grid, evidence, source): the chained writer runs instead."""


# ---------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def enabled(environ=None):
    """QWEN_FAST_KV_PAGE_WRITER: strict (unset or 0 off, 1 on, anything else raises). A four-card lever: it raises at the pair. The audit flag
    needs it."""
    import tp_shapes

    source = os.environ if environ is None else environ
    on = _read(FLAG, source)
    if not on:
        if _read(AUDIT_FLAG, source):
            raise ValueError('%s=1 needs %s=1 (the audit compares the launch it replaces)' % (AUDIT_FLAG, FLAG))
        return False
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    return True


def audit_enabled(environ=None):
    """QWEN_FAST_KV_PAGE_WRITER_AUDIT=1 beside the lever."""
    source = os.environ if environ is None else environ
    return enabled(source) and _read(AUDIT_FLAG, source)


def tiles_per_core(environ=None):
    """QWEN_FAST_KV_PAGE_WRITER_WT: column tiles per unit, one of 1, 2, 4, 8 (default 2: 64 cores in the served 8x8 rectangle). 1 needs 128 cores
    (a 13x10 grid); a smaller number of larger units uses fewer cores."""
    value = (os.environ if environ is None else environ).get(WT_FLAG)
    if value is None or value == '':
        return DEFAULT_WT
    if value not in tuple(str(width) for width in WIDTHS):
        raise ValueError('%s must be one of %s, got %r' % (WT_FLAG, ','.join(str(width) for width in WIDTHS), value))
    return int(value)


# ---------------------------------------------------------------------------------------------
# Pure planners (the kernel's arithmetic, in Python; CPU-tested and transliterated by kv_page_model_tp4).
# ---------------------------------------------------------------------------------------------

def pairs(wt):
    """Unit column groups per cache tile row."""
    if wt not in WIDTHS:
        raise ValueError('unit width %r is not one of %s' % (wt, WIDTHS))
    return ROW_TILES // wt


def unit_count(wt=DEFAULT_WT):
    return CACHES * GROUPS * SLOTS * pairs(wt)


def unit_index(cache, group, slot, pair, wt=DEFAULT_WT):
    if not (0 <= cache < CACHES and 0 <= group < GROUPS and 0 <= slot < SLOTS and 0 <= pair < pairs(wt)):
        raise ValueError('unit (%r, %r, %r, %r) is outside the block' % (cache, group, slot, pair))
    return ((cache * GROUPS + group) * SLOTS + slot) * pairs(wt) + pair


def units(wt=DEFAULT_WT):
    """Every unit in core order: dict(index, cache, group, slot, pair, first_column, first_row)."""
    found = []
    for cache in range(CACHES):
        for group in range(GROUPS):
            for slot in range(SLOTS):
                for pair in range(pairs(wt)):
                    found.append(dict(index=unit_index(cache, group, slot, pair, wt), cache=cache, group=group, slot=slot,
                                      pair=pair, first_column=pair * wt, first_row=group * GROUP_ROWS))
    return found


def core_layout(wt, grid_x, grid_y):
    """(points, width): unit i -> core (i % width, i // width). Width 8 while the units fit an 8-wide rectangle of at most 8 rows (64 cores, the
    served chained writer's), else the grid's own width (WT = 1: 128 cores, which needs a 13x10 grid). Unsupported when the grid cannot hold them."""
    count = unit_count(wt)
    width = GRID_WIDTH if count <= GRID_WIDTH * GRID_WIDTH else grid_x
    if width < 1 or grid_x < width or grid_y * width < count:
        raise Unsupported('%d units of %d tiles need %d cores in rows of %d; this grid is %dx%d' % (count, wt, count, width, grid_x, grid_y))
    return [(index % width, index // width) for index in range(count)], width


def predecessor(unit_for, wt=DEFAULT_WT):
    """The unit group g waits for in ORDERED mode: the same cache, slot and columns one group earlier (None for group 0)."""
    if unit_for['group'] == 0:
        return None
    return unit_index(unit_for['cache'], unit_for['group'] - 1, unit_for['slot'], unit_for['pair'], wt)


def key_of(position):
    """(page-table entry, tile row) of a position as one number: entry * 2 + tile row (the kernel's key_of)."""
    return ((position >> 6) << 1) | ((position >> 5) & 1)


def target_key(positions, slot):
    """(valid, key) of slot `slot` of one group's positions: slot 0 the first row's key, slot 1 the first key that differs from it
    (valid False when every row shares the first key). The kernel's target_key."""
    first = key_of(positions[0])
    if slot == 0:
        return True, first
    for position in positions[1:]:
        key = key_of(position)
        if key != first:
            return True, key
    return False, first


def row_mask(positions, key):
    """Bit o set when a row of the group with this key writes cache row o of its tile row (the kernel's row_mask)."""
    mask = 0
    for position in positions:
        if key_of(position) == key:
            mask |= 1 << (position & 31)
    return mask


def resolve_unit(positions, table, slot):
    """What the reader leaves in the meta record for slot `slot` of a group: dict(valid, block, tile_row, mask, rows), rows being the group
    rows (indices into `positions`) that write this tile row, in order. `table` is the group's page-table row (indexable by entry)."""
    valid, key = target_key(positions, slot)
    if not valid:
        return dict(valid=False, block=0, tile_row=key & 1, mask=0, rows=[])
    return dict(valid=True, block=int(table[key >> 1]), tile_row=key & 1, mask=row_mask(positions, key),
                rows=[row for row, position in enumerate(positions) if key_of(position) == key])


def group_problem(positions):
    """Why this group's positions are not the shape the kernel assumes, or None: GROUP_ROWS positions touching at most SLOTS tile rows."""
    if len(positions) != GROUP_ROWS:
        return '%d positions, not %d' % (len(positions), GROUP_ROWS)
    keys = []
    for position in positions:
        if type(position) is not int or position < 0:
            return 'position %r is not a non-negative int' % (position,)
        if key_of(position) not in keys:
            keys.append(key_of(position))
    if len(keys) > SLOTS:
        return 'the group touches %d tile rows, the unit table holds %d' % (len(keys), SLOTS)
    return None


def face_row_offset(row, half):
    """Byte offset of row `row`, half `half` (columns 16 half .. 16 half + 15) in a face-ordered 32x32 bfloat16 tile (the kernel's)."""
    return (((row >> 4) * 2 + half) * 512) + (row & 15) * 32


def cb_table(wt=DEFAULT_WT):
    """(indices, pages, dtype name, page bytes, tiled) per circular buffer of the page launch."""
    pages = 2 * wt
    return [((CB_CACHE,), pages, 'bfloat8_b', CACHE_TILE, True),
            ((CB_IN,), pages, 'bfloat16', TILE, True),
            ((CB_UNTILIZED, CB_UNTILIZED2), pages, 'bfloat16', TILE, True),
            ((CB_UNTILIZED_IN,), pages, 'bfloat16', TILE, True),
            ((CB_OUT,), pages, 'bfloat8_b', CACHE_TILE, True),
            ((CB_SCRATCH,), 1, 'int32', scratch_bytes(wt), False),
            ((CB_META,), 1, 'int32', META_BYTES, False)]


def scratch_bytes(wt=DEFAULT_WT):
    """The raw scratch CB: positions, the table window, the staged update spans (reader) or the audit tiles and record."""
    reader = 320 + GROUP_ROWS * wt * 2 * 64
    audit = 512 + 2 * wt * CACHE_TILE + 64
    size = max(reader, audit)
    return (size + 63) // 64 * 64


def compute_args(wt=DEFAULT_WT):
    """The served ordered writer's compute compile arguments (ordered_cache.update: [0, 1, 24, 25, 26, 16, 8, heads]) with Wt = the unit
    width and ONE head: one block of everything per unit."""
    return (CB_CACHE, CB_IN, CB_UNTILIZED, CB_UNTILIZED2, CB_UNTILIZED_IN, CB_OUT, wt, 1)


def cb_bytes(wt=DEFAULT_WT):
    return sum(pages * size for indices, pages, dtype, size, tiled in cb_table(wt))


# ---------------------------------------------------------------------------------------------
# The design and its signature (what the evidence record is bound to).
# ---------------------------------------------------------------------------------------------

def source_path(directory=None):
    return (Path(directory) if directory is not None else HERE) / SOURCE_NAME


def source_text(directory=None):
    data = source_path(directory).read_bytes()
    if b'\r' in data:
        raise ValueError('%s must be LF-only: its sha256 is its identity' % SOURCE_NAME)
    return data.decode()


def source_sha256(directory=None):
    return hashlib.sha256(source_path(directory).read_bytes()).hexdigest()


def design(wt=DEFAULT_WT, compute_sha256=None):
    """Everything the evidence is about, as plain data. `compute_sha256` is the pinned compute kernel's (ordered_cache.HASHES['compute'] unless
    given)."""
    if compute_sha256 is None:
        import ordered_cache

        compute_sha256 = ordered_cache.HASHES['compute']
    return dict(version=1, kernel=SOURCE_NAME, kernel_sha256=source_sha256(), compute_sha256=compute_sha256, groups=GROUPS,
                group_rows=GROUP_ROWS, slots=SLOTS, caches=CACHES, row_tiles=ROW_TILES, wt=wt, units=unit_count(wt),
                compute_args=list(compute_args(wt)),
                cb=[[list(indices), pages, dtype, size, tiled] for indices, pages, dtype, size, tiled in cb_table(wt)],
                reader_words=READER_WORDS, writer_words=WRITER_WORDS)


def design_signature(wt=DEFAULT_WT, compute_sha256=None):
    payload = json.dumps(design(wt, compute_sha256), sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------------------------
# The evidence gate.
# ---------------------------------------------------------------------------------------------

EVIDENCE_BLOCK = 'page_writer'
PROOF_REGIMES = ('exact', 'random', 'edge')
PROOF_MODES = ('eager', 'replay_changed', 'replay_unchanged')
PROOF_CACHES = ('k', 'v')
PROOF_WIDTHS = (2052, 4096)
PROOF_NEGATIVES = ('drop', 'slot')


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def block_problems(block, wt=DEFAULT_WT, width=None, source=None, compute_sha256=None):
    """Every reason the 'page_writer' record `block` does not qualify this design (wt) at page-table `width` and `source` ('dram' | 'l1');
    [] when it does. width / source None skip those two checks (the recorder checks the record's own coverage separately)."""
    if not isinstance(block, dict):
        return ['no page_writer record']
    problems = []
    if block.get('status') != 'PASS':
        return ['page_writer status %s, not PASS' % (block.get('status'),)]
    live = design_signature(wt, compute_sha256)
    if block.get('design_signature') != live:
        problems.append('the record qualified design %s, this one is %s (the kernel, the pinned compute kernel or the geometry changed)' % (
            str(block.get('design_signature'))[:16], live[:16]))
    if block.get('wt') != wt:
        problems.append('the record qualified unit width %r, not %d' % (block.get('wt'), wt))
    if block.get('scope') != 'full':
        problems.append('page_writer scope %s, not full' % (block.get('scope'),))
    if block.get('failures') != 0:
        problems.append('page_writer failures %r, not 0' % (block.get('failures'),))
    counts = block.get('counts') or {}
    for key in ('checks', 'exact'):
        if type(counts.get(key)) is not int or counts[key] <= 0:
            problems.append('page_writer counts.%s %r is not a positive count' % (key, counts.get(key)))
    if counts.get('checks') != counts.get('exact'):
        problems.append('page_writer counts: %r of %r checks exact' % (counts.get('exact'), counts.get('checks')))
    for key, wanted in (('widths', PROOF_WIDTHS), ('regimes', PROOF_REGIMES), ('modes', PROOF_MODES), ('caches', PROOF_CACHES),
                        ('sources', SOURCES)):
        found = block.get(key)
        missing = [item for item in wanted if not isinstance(found, list) or item not in found]
        if missing:
            problems.append('page_writer %s lack %s' % (key, missing))
    negatives = block.get('negatives')
    kinds = sorted(str(entry.get('kind')) for entry in negatives if isinstance(entry, dict)) if isinstance(negatives, list) else []
    if kinds != sorted(PROOF_NEGATIVES) or any(not (type(entry.get('failing')) is int and entry['failing'] > 0 and entry.get('verdict') == 'FAIL')
                                                for entry in negatives if isinstance(entry, dict)):
        problems.append('page_writer negative controls %s are not one failing drop and one failing slot' % (kinds,))
    for key in ('noop_served', 'noop_page', 'twice_page'):
        entry = (block.get('proofs') or {}).get(key)
        if not isinstance(entry, dict) or type(entry.get('checks')) is not int or entry['checks'] <= 0 or entry.get('checks') != entry.get('exact'):
            problems.append('page_writer proof %s is missing or not exact' % key)
    if width is not None and width not in (block.get('widths') or ()):
        problems.append('page-table width %r is not among the recorded %r' % (width, block.get('widths')))
    if source is not None and source not in (block.get('sources') or ()):
        problems.append('source %r is not among the recorded %r' % (source, block.get('sources')))
    return problems


def evidence_problems(wt=DEFAULT_WT, width=None, source=None, path=None, expected=None, sources_root=HERE, compute_sha256=None):
    """Why the lever may not engage (a list), [] when ordered_writer_evidence_tp4.json is present at its pin (page_width_tp4.evidence_state),
    carries a passing 'page_writer' block and that block matches the live design, width and source."""
    import page_width_tp4

    ok, problems = page_width_tp4.evidence_state(path, expected, sources_root)
    if not ok:
        return ['the ordered-writer record does not stand: ' + '; '.join(problems)]
    evidence_path = Path(page_width_tp4.EVIDENCE if path is None else path)
    try:
        evidence = json.loads(evidence_path.read_bytes().decode('utf-8'))
    except (OSError, ValueError) as error:
        return ['the ordered-writer record does not parse: %s' % str(error)[:80]]
    return block_problems(evidence.get(EVIDENCE_BLOCK), wt, width, source, compute_sha256)


# ---------------------------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------------------------

_NOTED = set()


def log_line(message):
    import verify_trace_t2

    verify_trace_t2.log_line(message)


def fall_back(reason):
    """One FELL_BACK line per distinct reason per process; always returns None (the chained writer runs)."""
    if reason not in _NOTED:
        _NOTED.add(reason)
        log_line('%s reason=%s' % (FELL_BACK, reason))
    return None


def note_engaged(wt, cores, source, ordered, width, audit):
    key = ('engaged', wt, cores, source, ordered, width, audit)
    if key not in _NOTED:
        _NOTED.add(key)
        log_line('%s cores=%d wt=%d source=%s ordered=%d width=%d audit=%d design=%s' % (
            ENGAGED, cores, wt, source, int(ordered), width, int(audit), design_signature(wt)[:16]))


# ---------------------------------------------------------------------------------------------
# The launch.
# ---------------------------------------------------------------------------------------------

def sha_define(directory=None):
    """First 8 hex of the kernel's sha256, passed as a define: defines are hashed into the JIT key, so a revised kernel at the same path can
    never reuse a stale binary."""
    return '0x' + source_sha256(directory)[:8]


def kernel_defines(role, source, negative=None, sha=None):
    defines = {ROLE_DEFINES[role]: '1', 'KVPW_SRC_SHA': sha if sha is not None else sha_define()}
    if source == 'l1' and role == 'reader':
        defines['KVPW_SRC_L1'] = '1'
    if negative is not None:
        if negative not in NEGATIVE_CONTROLS:
            raise ValueError('unknown negative control %r' % (negative,))
        defines[NEGATIVE_CONTROLS[negative]] = '1'
    return sorted(defines.items())


def shard_cores(spec):
    """The logical cores of a height-sharded tensor's shards in shard order, or Unsupported. `spec` is ttnn's ShardSpec (grid, shape,
    orientation) or any object with those attributes; the grid is a CoreRangeSet whose `ranges()` (or iteration) yields CoreRange(start, end)."""
    orientation = str(getattr(spec, 'orientation', 'ROW_MAJOR'))
    if 'ROW_MAJOR' not in orientation.upper():
        raise Unsupported('the prepared K/V shards are %s, not row-major' % orientation)
    grid = spec.grid
    ranges = grid.ranges() if callable(getattr(grid, 'ranges', None)) else list(grid)
    cores = []
    for core_range in ranges:
        start, end = core_range.start, core_range.end
        start_x, start_y = (start.x, start.y)
        end_x, end_y = (end.x, end.y)
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                cores.append((x, y))
    return cores


def source_mode(operations, packed):
    """'dram' for an interleaved DRAM prepared K/V, 'l1' for the AttnPrep output's height shards (one (32, 256) shard per row), else
    Unsupported. Returns (mode, shard cores or None)."""
    config = packed.memory_config()
    if config == operations.DRAM_MEMORY_CONFIG:
        return 'dram', None
    spec = getattr(config, 'shard_spec', None)
    if spec is None:
        spec = getattr(packed, 'shard_spec', None)
    if spec is None:
        raise Unsupported('the prepared K/V is neither interleaved DRAM nor height-sharded')
    layout = str(getattr(config, 'memory_layout', 'HEIGHT_SHARDED')).upper()
    buffer = str(getattr(config, 'buffer_type', 'L1')).upper()
    if 'HEIGHT' not in layout or 'L1' not in buffer:
        raise Unsupported('the prepared K/V is %s in %s, not height-sharded in L1' % (layout, buffer))
    shape = tuple(getattr(spec, 'shape', ()))
    if shape != (TILE_ROWS, 256):
        raise Unsupported('the prepared K/V shard shape %r is not (%d, 256)' % (shape, TILE_ROWS))
    cores = shard_cores(spec)
    if len(cores) != BLOCK_ROWS:
        raise Unsupported('the prepared K/V has %d shards, not one per row (%d)' % (len(cores), BLOCK_ROWS))
    return 'l1', cores


def validate_tensors(operations, caches, packed, positions, pages, width):
    """The served input checks (packed_ordered_cache.update_chained's), for both caches and both prepared tensors."""
    import page_width_tp4
    import tp_shapes

    if tp_shapes.active().attn_kv_heads != HEADS:
        raise Unsupported('the page writer serves one KV head per chip (four cards)')
    for cache in caches:
        shape = tuple(cache.shape)
        if len(shape) != 4 or shape[0] < 1 or shape[1:] != (HEADS, 64, 256) or cache.dtype != operations.bfloat8_b \
                or cache.layout != operations.TILE_LAYOUT or cache.memory_config() != operations.DRAM_MEMORY_CONFIG:
            raise ValueError('Interleaved DRAM native BF8 one-head 64-row paged caches required')
        if width > shape[0]:
            raise ValueError('Paired position vector and page-table rows required')
    for value in packed:
        if tuple(value.shape) != (1, BLOCK_ROWS, TILE_ROWS, 256) or value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT:
            raise ValueError('Native BF16 prepared tiles (1, 64, 32, 256) required')
    if tuple(positions.shape) != (BLOCK_ROWS,) or tuple(pages.shape) != (BLOCK_ROWS, width) or not page_width_tp4.admitted(width):
        raise ValueError('Paired position vector and page-table rows required')
    for value in (positions, pages):
        if value.dtype != operations.int32 or value.layout != operations.ROW_MAJOR_LAYOUT or value.memory_config() != operations.DRAM_MEMORY_CONFIG:
            raise ValueError('Int32 row-major DRAM metadata required')


def reader_words(addresses, unit_for, wt, ordered, source_noc):
    """One reader core's runtime words (READER_WORDS, every core, every launch). `addresses` is dict(cache=[K, V], packed=[K, V], positions,
    pages); `source_noc` the NOC (x, y) of each group row's source core (L1 source) or None."""
    first_row = unit_for['first_row']
    words = [addresses['cache'][unit_for['cache']], addresses['positions'], addresses['pages'], addresses['packed'][unit_for['cache']],
             first_row, unit_for['slot'], unit_for['first_column'], int(bool(ordered) and unit_for['group'] > 0)]
    for row in range(GROUP_ROWS):
        words.extend(source_noc[first_row + row] if source_noc is not None else (0, 0))
    if len(words) != READER_WORDS:
        raise AssertionError('reader runtime words drifted from READER_WORDS')
    return words


def writer_words(addresses, unit_for, wt, ordered, noc_of):
    """One writer core's runtime words (WRITER_WORDS): the cache, the first column tile, and in ORDERED mode the signal to the next group's
    unit (the same cache, slot and columns)."""
    signal, target = 0, (0, 0)
    if ordered and unit_for['group'] < GROUPS - 1:
        signal = 1
        target = noc_of(unit_index(unit_for['cache'], unit_for['group'] + 1, unit_for['slot'], unit_for['pair'], wt))
    words = [addresses['cache'][unit_for['cache']], unit_for['first_column'], signal, target[0], target[1], 0, 0, 0]
    if len(words) != WRITER_WORDS:
        raise AssertionError('writer runtime words drifted from WRITER_WORDS')
    return words


def build_program(operations, mesh, tensors, kernels, texts, *, wt=DEFAULT_WT, source='dram', source_cores=None, ordered=False,
                  negative=None, grid=None):
    """The per-chip programs of one page launch. `tensors` is dict(caches=[K, V], packed=[K, V], positions, pages); `kernels` the served
    ordered writer's sources (the compute is taken from it); `texts` the new reader and writer sources (dict reader/writer, usually
    source_text() twice). `source_cores` the logical cores of the prepared K/V shards (L1 source)."""
    from verify_trace_t1 import rectangle_set

    coordinates = mesh_coordinates(tuple(mesh.shape))
    chips = len(coordinates)
    grid = mesh.compute_with_storage_grid_size() if grid is None else grid
    points, width = core_layout(wt, grid.x, grid.y)
    cores = rectangle_set(operations, points)
    width_entries = int(tensors['pages'].padded_shape[-1])
    page_bytes = width_entries * 4
    shards = dict(caches=[operations.get_device_tensors(value) for value in tensors['caches']],
                  packed=[operations.get_device_tensors(value) for value in tensors['packed']],
                  positions=operations.get_device_tensors(tensors['positions']),
                  pages=operations.get_device_tensors(tensors['pages']))
    if any(len(parts) != chips for parts in (*shards['caches'], *shards['packed'], shards['positions'], shards['pages'])):
        raise ValueError('Every tensor must have one shard per mesh device')
    unit_list = units(wt)
    dtypes = dict(bfloat8_b=operations.bfloat8_b, bfloat16=operations.bfloat16, int32=operations.int32)
    buffers = []
    for indices, count, dtype, size, tiled in cb_table(wt):
        formats = [operations.CBFormatDescriptor(buffer_index=index, data_format=dtypes[dtype], page_size=size,
                   **(dict(tile=operations.TileDescriptor(operations.Tile([32, 32]))) if tiled else {})) for index in indices]
        buffers.append(operations.CBDescriptor(total_size=count * size, core_ranges=cores, format_descriptors=formats))
    semaphore = operations.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)
    configs = dict(
        reader=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default),
        writer=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default),
        compute=operations.ComputeConfigDescriptor(fp32_dest_acc_en=False))
    program = operations.MeshProgramDescriptor()
    sha = sha_define()
    for chip, (row_index, column_index) in enumerate(coordinates):
        local = dict(caches=[parts[chip] for parts in shards['caches']], packed=[parts[chip] for parts in shards['packed']],
                     positions=shards['positions'][chip], pages=shards['pages'][chip])
        addresses = dict(cache=[value.buffer_address() for value in local['caches']],
                         packed=[value.buffer_address() for value in local['packed']],
                         positions=local['positions'].buffer_address(), pages=local['pages'].buffer_address())
        everything = [*addresses['cache'], *addresses['packed'], addresses['positions'], addresses['pages']]
        if len(set(everything)) != len(everything):
            raise ValueError('Caches, prepared K/V and metadata must not alias')
        device = local['caches'][0].device()

        def noc_of(index, device=device):
            point = device.worker_core_from_logical_core(operations.CoreCoord(*points[index]))
            return point.x, point.y

        source_noc = None
        if source == 'l1':
            if source_cores is None or len(source_cores) != BLOCK_ROWS:
                raise ValueError('the L1 source needs the logical cores of the %d prepared K/V shards' % BLOCK_ROWS)
            source_noc = [(lambda point: (point.x, point.y))(device.worker_core_from_logical_core(operations.CoreCoord(x, y)))
                          for x, y in source_cores]
        reader_args = [GROUP_ROWS, wt, page_bytes, BLOCK_ROWS, READER_WORDS]
        for tensor in (local['caches'][0], local['positions'], local['pages']):
            reader_args.extend(operations.TensorAccessorArgs(tensor).get_compile_time_args())
        if source != 'l1':
            reader_args.extend(operations.TensorAccessorArgs(local['packed'][0]).get_compile_time_args())
        writer_args = [wt, WRITER_WORDS]
        writer_args.extend(operations.TensorAccessorArgs(local['caches'][0]).get_compile_time_args())
        descriptors = []
        for role in ROLES:
            runtime = operations.RuntimeArgs()
            for unit_for in unit_list:
                x, y = points[unit_for['index']]
                if role == 'reader':
                    runtime[x][y] = reader_words(addresses, unit_for, wt, ordered, source_noc)
                elif role == 'writer':
                    runtime[x][y] = writer_words(addresses, unit_for, wt, ordered, noc_of)
                else:
                    runtime[x][y] = []
            if role == 'compute':
                descriptor = operations.KernelDescriptor(kernel_source=kernels['compute'],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=list(compute_args(wt)), config=configs['compute'])
            else:
                descriptor = operations.KernelDescriptor(kernel_source=texts[role],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=reader_args if role == 'reader' else writer_args,
                    defines=kernel_defines(role, source, negative, sha), config=configs[role])
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = operations.MeshCoordinate(row_index, column_index)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers, semaphores=[semaphore])
    return program


def mesh_coordinates(shape):
    """The mesh's chips as (row, column): the four-card mesh [1, 4], or the one-card harness's [1, 1]."""
    import tp_shapes

    if shape not in ((1, 1), (1, tp_shapes.chip_count())):
        raise Unsupported('mesh %s is not [1, 1] or [1, %d]' % (shape, tp_shapes.chip_count()))
    rows, columns = shape
    return [(index // columns, index % columns) for index in range(rows * columns)]


def launch(operations, mesh, tensors, kernels, texts, **options):
    """One page launch for the layer's K and V (generic_op over the caches, the prepared tensors and the metadata)."""
    program = build_program(operations, mesh, tensors, kernels, texts, **options)
    io = [*tensors['caches'], *tensors['packed'], tensors['positions'], tensors['pages']]
    operations.generic_op(io, program)


# ---------------------------------------------------------------------------------------------
# The audit launches.
# ---------------------------------------------------------------------------------------------

def audit_cb_table(wt):
    return [((CB_SCRATCH,), 1, 'int32', scratch_bytes(wt), False)]


def build_audit_program(operations, mesh, role, tensors, texts, *, wt=DEFAULT_WT, grid=None, negative=None):
    """The audit prep ('audit_prep') or check ('audit_check') program: one dataflow kernel per unit core, on the page launch's cores. `tensors`
    is dict(caches=[K, V], shadows=[K, V], positions, pages, shadow_positions | counters)."""
    from verify_trace_t1 import rectangle_set

    coordinates = mesh_coordinates(tuple(mesh.shape))
    grid = mesh.compute_with_storage_grid_size() if grid is None else grid
    points, width = core_layout(wt, grid.x, grid.y)
    cores = rectangle_set(operations, points)
    page_bytes = int(tensors['pages'].padded_shape[-1]) * 4
    last = tensors['shadow_positions'] if role == 'audit_prep' else tensors['counters']
    names = ('caches', 'shadows', 'positions', 'pages')
    shards = {name: ([operations.get_device_tensors(value) for value in tensors[name]] if name in ('caches', 'shadows')
                     else operations.get_device_tensors(tensors[name])) for name in names}
    last_shards = operations.get_device_tensors(last)
    buffers = [operations.CBDescriptor(total_size=scratch_bytes(wt), core_ranges=cores, format_descriptors=[
        operations.CBFormatDescriptor(buffer_index=CB_SCRATCH, data_format=operations.int32, page_size=scratch_bytes(wt))])]
    config = operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default)
    program = operations.MeshProgramDescriptor()
    sha = sha_define()
    for chip, (row_index, column_index) in enumerate(coordinates):
        cache_k, cache_v = (parts[chip] for parts in shards['caches'])
        shadow_k, shadow_v = (parts[chip] for parts in shards['shadows'])
        positions, pages, final = shards['positions'][chip], shards['pages'][chip], last_shards[chip]
        arguments = [GROUP_ROWS, wt, page_bytes, BLOCK_ROWS, AUDIT_PREP_WORDS if role == 'audit_prep' else AUDIT_CHECK_WORDS]
        for tensor in (cache_k, shadow_k, positions, pages, final):
            arguments.extend(operations.TensorAccessorArgs(tensor).get_compile_time_args())
        runtime = operations.RuntimeArgs()
        for unit_for in units(wt):
            x, y = points[unit_for['index']]
            real, shadow = (cache_k, shadow_k) if unit_for['cache'] == 0 else (cache_v, shadow_v)
            if role == 'audit_prep':
                writes = int(unit_for['cache'] == 0 and unit_for['slot'] == 0 and unit_for['pair'] == 0)
                runtime[x][y] = [real.buffer_address(), shadow.buffer_address(), positions.buffer_address(), pages.buffer_address(),
                                 final.buffer_address(), unit_for['first_row'], unit_for['slot'], unit_for['first_column'], writes,
                                 unit_for['group']]
            else:
                runtime[x][y] = [real.buffer_address(), shadow.buffer_address(), positions.buffer_address(), pages.buffer_address(),
                                 final.buffer_address(), unit_for['first_row'], unit_for['slot'], unit_for['first_column'], unit_for['index']]
        descriptor = operations.KernelDescriptor(kernel_source=texts[role], source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE,
                                                 core_ranges=cores, compile_time_args=arguments,
                                                 defines=kernel_defines(role, 'dram', negative, sha), config=config)
        descriptor.runtime_args = runtime
        coordinate = operations.MeshCoordinate(row_index, column_index)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=[descriptor], cbs=buffers)
    return program


class Audit:
    """The audit's device state for one layer's writer: a shadow cache per cache (small), the shadow positions and identity table the served
    writer reads, and the per-unit counters. Allocated once at construction, before any warm forward or capture."""

    def __init__(self, operations, mesh, wt):
        import torch

        self.operations, self.mesh, self.wt = operations, mesh, wt
        dram = operations.DRAM_MEMORY_CONFIG

        def upload(value, dtype, layout):
            return operations.from_torch(value, device=mesh, dtype=dtype, layout=layout, memory_config=dram,
                                         mesh_mapper=operations.ReplicateTensorToMesh(mesh))

        self.shadows = [upload(torch.zeros(AUDIT_BLOCKS, HEADS, 64, 256).bfloat16(), operations.bfloat8_b, operations.TILE_LAYOUT)
                        for unused in range(CACHES)]
        self.positions = upload(torch.zeros(BLOCK_ROWS, dtype=torch.int32), operations.int32, operations.ROW_MAJOR_LAYOUT)
        self.table = upload(torch.arange(AUDIT_TABLE_WIDTH, dtype=torch.int32).repeat(BLOCK_ROWS, 1), operations.int32,
                            operations.ROW_MAJOR_LAYOUT)
        self.counters = upload(torch.zeros(unit_count(wt), AUDIT_RECORD_WORDS, dtype=torch.int32), operations.int32,
                               operations.ROW_MAJOR_LAYOUT)
        self.rounds = 0


_REGISTRY = []
_LAYERS = [0]


def registry():
    return _REGISTRY


def audit_round(operations, round_number=0):
    """After a replay: every registered writer's counters must read zero mismatched words with at least one valid unit per chip. Logs
    '<AUDIT_MARKER> <n> exact=True writers=<w> units=<u>' or AUDIT_MISMATCH and raises. Returns the number of valid units compared."""
    mismatches, valid_units, writers = [], 0, 0
    for writer in list(_REGISTRY):
        audit = writer.audit
        if audit is None:
            continue
        writers += 1
        for chip, shard in enumerate(operations.get_device_tensors(audit.counters)):
            values = operations.to_torch(shard).tolist()
            for unit, record in enumerate(values):
                if record[1]:
                    valid_units += 1
                if record[0]:
                    mismatches.append('layer %d chip %d unit %d: %d words differ (block %d tile row %d)' % (
                        writer.layer, chip, unit, record[0], record[2], record[3]))
    if mismatches or not valid_units:
        message = '%s round=%d %s' % (AUDIT_MISMATCH, round_number, '; '.join(mismatches[:4]) or 'no valid unit was compared')
        log_line(message)
        raise AssertionError(message)
    _NOTED.add(('audit', round_number))
    log_line('%s %d exact=True writers=%d units=%d' % (AUDIT_MARKER, round_number + 1, writers, valid_units))
    return valid_units


# ---------------------------------------------------------------------------------------------
# The writer the binding holds.
# ---------------------------------------------------------------------------------------------

class KVPageWriter:
    """One layer's page writer. `call(cache, packed)` is the chained writer's call contract: the first call of a pair (K) is held, the second
    (V) launches both and returns True; a call that cannot be served (the source, the first time) returns False BEFORE anything is held and
    every later call too, so the chained writer serves the layer from the start."""

    def __init__(self, mesh, operations, kernels, *, positions, pages, spans, launch_rows, layer=0, environ=None):
        import tp_shapes

        if not enabled(environ):
            raise Unsupported('the flag is off')
        self.mesh, self.operations, self.kernels = mesh, operations, kernels
        self.positions, self.pages, self.layer = positions, pages, layer
        self.wt = tiles_per_core(environ)
        self.audit_on = audit_enabled(environ)
        self.width = int(pages.shape[1])
        if launch_rows != BLOCK_ROWS:
            raise Unsupported('launch rows %r, not %d (the 32-row mode is the chained writer\'s)' % (launch_rows, BLOCK_ROWS))
        if tuple(positions.shape) != (BLOCK_ROWS,) or tuple(pages.shape) != (BLOCK_ROWS, self.width):
            raise Unsupported('the block metadata is not (%d,) positions and (%d, width) tables' % (BLOCK_ROWS, BLOCK_ROWS))
        spans = tuple((int(first), int(last)) for first, last in spans)
        users = tuple((group * GROUP_ROWS, (group + 1) * GROUP_ROWS) for group in range(GROUPS))
        if spans == users:
            self.ordered = False
        elif spans == ((0, BLOCK_ROWS),):
            self.ordered = True
        else:
            raise Unsupported('spans %r are neither %d users of %d rows nor the warm forward\'s single span' % (spans, GROUPS, GROUP_ROWS))
        if tp_shapes.chip_count(environ) != 4:
            raise Unsupported('not four cards')
        grid = mesh.compute_with_storage_grid_size()
        core_layout(self.wt, grid.x, grid.y)
        problems = evidence_problems(self.wt, self.width)
        if problems:
            raise Unsupported('no usable page-writer evidence: ' + '; '.join(problems)[:300])
        self.texts = dict(reader=source_text(), writer=source_text(), audit_prep=source_text(), audit_check=source_text())
        self.source, self.source_cores = None, None
        self.pending = None
        self.dead = False
        self.calls = 0
        self.launches = 0
        # The audit compares the launch with the served chained writer on a shadow cache, which is valid only where no two users share a tile row: not in
        # the warm forward's ORDERED mode (placeholders that all sit on one tile row), so that writer carries no audit and is never registered.
        self.audit = Audit(operations, mesh, self.wt) if self.audit_on and not self.ordered else None
        if self.audit is not None:
            _REGISTRY.append(self)

    def call(self, cache, packed, chained=None):
        """True when this call was taken (held or launched); False when the chained writer must serve it. `chained` is the callable that
        serves one cache on the audit's shadow (audit only)."""
        operations = self.operations
        if self.dead:
            return False
        if self.source is None:
            try:
                self.source, self.source_cores = source_mode(operations, packed)
            except Unsupported as reason:
                self.dead = True
                fall_back(str(reason))
                return False
            problems = evidence_problems(self.wt, self.width, self.source)
            if problems:
                self.dead = True
                fall_back('no usable page-writer evidence: ' + '; '.join(problems)[:300])
                return False
        if self.pending is None:
            self.pending = (cache, packed)
            self.calls += 1
            return True
        held_cache, held_packed = self.pending
        if held_cache is cache:
            raise ValueError('two K/V cache writes of one cache without the other between them: the page writer takes K then V')
        self.pending = None
        self.calls += 1
        self.launch_pair((held_cache, cache), (held_packed, packed), chained)
        return True

    def launch_pair(self, caches, packed, chained):
        operations = self.operations
        tensors = dict(caches=list(caches), packed=list(packed), positions=self.positions, pages=self.pages)
        validate_tensors(operations, tensors['caches'], tensors['packed'], self.positions, self.pages, self.width)
        converted = []
        try:
            if self.audit is not None:
                self.audit_before(tensors, chained, converted)
            launch(operations, self.mesh, tensors, self.kernels, self.texts, wt=self.wt, source=self.source,
                   source_cores=self.source_cores, ordered=self.ordered)
            if self.audit is not None:
                self.audit_after(tensors)
        finally:
            for value in converted:
                operations.deallocate(value)
        self.launches += 1
        note_engaged(self.wt, unit_count(self.wt), self.source, self.ordered, self.width, self.audit is not None)

    def audit_before(self, tensors, chained, converted):
        operations, audit = self.operations, self.audit
        audit_tensors = dict(caches=tensors['caches'], shadows=audit.shadows, positions=self.positions, pages=self.pages,
                             shadow_positions=audit.positions)
        operations.generic_op([*tensors['caches'], *audit.shadows, audit.positions, self.pages],
                              build_audit_program(operations, self.mesh, 'audit_prep', audit_tensors, self.texts, wt=self.wt))
        for shadow, value in zip(audit.shadows, tensors['packed']):
            interleaved = value
            if value.memory_config() != operations.DRAM_MEMORY_CONFIG:
                interleaved = operations.to_memory_config(value, operations.DRAM_MEMORY_CONFIG)
                converted.append(interleaved)
            chained(shadow, interleaved, audit.positions, audit.table)

    def audit_after(self, tensors):
        operations, audit = self.operations, self.audit
        audit_tensors = dict(caches=tensors['caches'], shadows=audit.shadows, positions=self.positions, pages=self.pages,
                             counters=audit.counters)
        operations.generic_op([*tensors['caches'], *audit.shadows, audit.counters, self.pages],
                              build_audit_program(operations, self.mesh, 'audit_check', audit_tensors, self.texts, wt=self.wt))
        audit.rounds += 1

    def idle(self):
        """True when no K call is held (the forward ended between layers' pairs)."""
        return self.pending is None


def attach(mesh, operations, kernels, *, positions, pages, spans, launch_rows, environ=None):
    """A KVPageWriter, or None after one logged line when this block cannot take the lever (the chained writer then runs, unchanged). A
    misconfigured flag raises; an unsupported block or missing evidence never does."""
    if not enabled(environ):
        return None
    try:
        layer = _LAYERS[0]
        _LAYERS[0] += 1
        return KVPageWriter(mesh, operations, kernels, positions=positions, pages=pages, spans=spans, launch_rows=launch_rows, layer=layer,
                            environ=environ)
    except Unsupported as reason:
        return fall_back(str(reason))


def describe(wt=DEFAULT_WT, grid_x=11, grid_y=10):
    """Host-only description of one launch (never opens a device)."""
    points, width = core_layout(wt, grid_x, grid_y)
    return dict(status='host plan only; no compilation or hardware certification', units=unit_count(wt), wt=wt, cores=len(points),
                core_rows_of=width, grid=[grid_x, grid_y], reader_words=READER_WORDS, writer_words=WRITER_WORDS,
                cb_bytes=cb_bytes(wt), kernel_sha256=source_sha256(), design_signature=design_signature(wt))
