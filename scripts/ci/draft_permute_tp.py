"""The drafter's K/V assembly, query fold and output unfold as permutation launches at four cards (QWEN_FAST_DRAFT_PERMUTE, default off; F-F2 of the op-fusion programme).

WHY. The device profile of the shipped stack (run 38030902670, trace of the quad) shows the drafter quad spending 2.85 ms of its 14.26 ms on assembling K and V and 0.65 ms on
folding the query and unfolding the attention output: 505 launches a quad, of which 280 build the two (1, 2, 8320, 128) tensors a layer. Each user's 16 live rows are cut from the
shared 64-row block with slices that start inside a tile, round-tripped through untilize / tilize, and joined with the users' cached banks by a row-major concat (12 untilizes,
a 64 us concat on 110 cores, a 40 us tilize on 104) - about 285 us a tensor and layer. None of it computes: every byte moves whole tiles or 8- or 16-row blocks of a tile, plus
one value rule.

WHAT. One launch per site and layer (draft_permute_tp.cpp): `kv` builds K and V together, `fold` builds the folded query, `unfold` the packed attention rows. The host
planners below derive, from the served composition itself, which destination tile comes from which source tile or quarter tile and whether it takes the canonical rule.

THE CANONICAL RULE. The served untilize / tilize round trip maps a bf16 with a zero exponent (-0 and every denormal) to +0 (gdn_prefill_conv_exact.py; U2 in
optimisation/ttnn-op/canon_probe measures it on every pattern). The served composition applies it to the pieces below, and the planners apply it to exactly those and to
nothing else:
  - a slice whose start is INSIDE a tile (start % 32 != 0) goes through untilize / slice / tilize: canonical. A slice that starts on a tile boundary is a raw tile copy
    (also when its length is 16 rows: the profile shows a plain Slice op there);
  - a concat on the row axis whose pieces are not all whole tiles goes through untilize of EVERY piece, a row-major concat and one tilize of the result: every byte of its
    output is canonical, the tile-aligned pieces (the users' 2,048-row cached banks) included. The profile of the quad shows exactly this: four 64-core untilizes of 22.8 us
    (the banks), eight 2-core untilizes (the live and pad pieces), one 110-core concat and one 104-core tilize of the whole (1, 2, 8320, 128) result. A concat whose pieces are
    all whole tiles is a tile copy, raw (the single-user path: cached 2,048 rows + 32);
  - a concat on the head axis, a reshape and a slice on the head axis move whole tiles: raw.
So the K/V assembly canonicalises every tile of its output; the fold canonicalises the rotated copies (the swapped-halves tile of the pair and quad fold, every rotation but the
first at the octo fold) and passes the unrotated copy raw; the unfold canonicalises everything (its blocks are joined by a row-major concat). The plan of the programme
(section 3 F-F2) said the tile-aligned pieces stay raw; the profile shows the bank untilizes, so they do not, and the planner follows the profile. Which of the two a card
does is the first job of the window (optimisation/ttnn-op/draft_permute: the permutation bytes against the served assembly on edge data, both variants), and
`canon_cached` is the planner's one switch for it.

FLAGS (strict 0 or 1, read at call time, never at import; QWEN_FAST_TP=4 only):
  QWEN_FAST_DRAFT_PERMUTE         the three sites above. Flag off, nothing in this module is imported by the twins and every served op runs as before.
  QWEN_FAST_DRAFT_PERMUTE_AUDIT   needs the lever. In the bucket's eager warm pass (quad_draft_tp's _execute marks it) the served composition runs beside each launch and
                                  every output pair is byte-compared on every chip; the capture records the launches only. A pass that is not marked is not audited.
A call the planners or the launch cannot take (a layout, a row count that is not a multiple of 8, a list longer than a core's argument budget) runs the served ops and logs
a 'fell back' line.

MARKERS: '[PINDIAG] tp4 draft permute engaged site=kv|fold|unfold shape=pair|quad|octo ...', '... fell back site=... reason=...', '... audit exact=True site=...',
'... audit mismatch ...' (raises). draft_permute_smoke.py is the log rule.

Pure planners first (no ttnn), the launch builder last. A record is
    ('run', destination index, destination page, source index, source page, tile count, canonical)          consecutive tiles
    ('mix', destination index, destination page, (q0, q1, q2, q3))                                         one tile from four quarters,
with each quarter (source index, source page, mode) as in gdn_rows_dma8_tp (mode 0 zeros, 1..4 raw quarter, 5..8 canonical quarter).

Stdlib only at import, py 3.7 (torch inside the audit).
"""

from bisect import bisect_right
import os
from pathlib import Path

import tp4_sampdraft
import tp_shapes

KERNEL = 'draft_permute_tp.cpp'
RUNTIME_FILES = ('draft_permute_tp.py', 'draft_permute_tp.cpp')

FLAG = 'QWEN_FAST_DRAFT_PERMUTE'
AUDIT_FLAG = 'QWEN_FAST_DRAFT_PERMUTE_AUDIT'
ENGAGED = '[PINDIAG] tp4 draft permute engaged'
FALLBACK = '[PINDIAG] tp4 draft permute fell back'
AUDIT = '[PINDIAG] tp4 draft permute audit'
MISMATCH = '[PINDIAG] tp4 draft permute audit mismatch'

TILE = 32
QUARTER = 8                       # rows of a quarter tile
QUARTERS = TILE // QUARTER
HEAD_DIM = 128
COLUMN_TILES = HEAD_DIM // TILE   # 4 tiles a row of a head
TILE_BYTES = 2048
LANES = 8                         # tiles in flight a barrier (the kernel's)
PROCESSORS = 2                    # worker kernels a core: RISCV_0 on NOC 0 and RISCV_1 on NOC 1, each with its own tiles
MAX_ARGUMENT_WORDS = 256
MIN_LANE_WEIGHT = 2

TYPE_RUN, TYPE_MIX = 1, 2
RUN_WORDS, MIX_WORDS = 4, 6
RAW, CANON = 1, 5                 # quarter mode = RAW / CANON + the source quarter
MIX_WEIGHT = 2


class Unsupported(ValueError):
    """The launch this planner would build does not fit the kernel: the caller takes the served path and says so."""


# ---------------------------------------------------------------------------------------------
# Flags, markers and the audit pass.
# ---------------------------------------------------------------------------------------------

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def enabled(environ=None):
    """Whether QWEN_FAST_DRAFT_PERMUTE is on. Strict; on at the pair is refused (a four-card lever)."""
    if not _read(FLAG, environ):
        return False
    if tp_shapes.chip_count(os.environ if environ is None else environ) == tp_shapes.PAIR:
        raise ValueError('%s=1 is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    return True


def audit_enabled(environ=None):
    """Whether QWEN_FAST_DRAFT_PERMUTE_AUDIT is on. An audit without the lever is a misconfigured arm and raises: it would pass having audited nothing."""
    if not _read(AUDIT_FLAG, environ):
        return False
    if not enabled(environ):
        raise ValueError('%s=1 needs %s=1: the audit would compare nothing' % (AUDIT_FLAG, FLAG))
    return True


def validate(environ=None):
    """Both flags strict and paired, checked once at attach. Raises ValueError."""
    enabled(environ)
    audit_enabled(environ)


_LOGGED = set()
_PASS = {'eager': None}


def note(kind, text):
    """One marker line per (kind, text): five layers and two quads print their few distinct lines."""
    key = (kind, text)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    tp4_sampdraft.log_line('%s %s' % (kind, text))


def set_pass(eager):
    """The caller that knows (quad_draft_tp's _execute): True while the bucket's eager warm pass runs, False while its capture records, None otherwise. The audit runs only in
    an eager pass - a capture cannot read a tensor back."""
    previous = _PASS['eager']
    _PASS['eager'] = eager
    return previous


def auditing():
    return _PASS['eager'] is True and audit_enabled()


_SERVED = {'depth': 0}


class served_only(object):
    """While a served reference (an audit's, or a fallback's) runs, the hooks inside it take the served ops: the quad's served fold calls the pair's fold, which is itself a
    hooked site."""

    def __enter__(self):
        _SERVED['depth'] += 1

    def __exit__(self, *exc):
        _SERVED['depth'] -= 1
        return False


def hook_enabled(environ=None):
    """What a hooked twin asks before it imports anything else of this module's: the lever is on and this call is not a served reference."""
    return _SERVED['depth'] == 0 and enabled(environ)


# ---------------------------------------------------------------------------------------------
# The records.
# ---------------------------------------------------------------------------------------------

class Emitter(object):
    """Collects records, merging consecutive whole-tile moves of one (destination, source, canonical) into one run."""

    def __init__(self):
        self.records = []
        self._run = None

    def flush(self):
        if self._run is not None:
            self.records.append(tuple(self._run))
            self._run = None

    def tile(self, destination, destination_page, source, source_page, canonical):
        run = self._run
        if (run is not None and run[1] == destination and run[3] == source and run[6] == canonical
                and run[2] + run[5] == destination_page and run[4] + run[5] == source_page):
            run[5] += 1
            return
        self.flush()
        self._run = ['run', destination, destination_page, source, source_page, 1, canonical]

    def mix(self, destination, destination_page, quarters):
        self.flush()
        self.records.append(('mix', destination, destination_page, tuple(quarters)))

    def done(self):
        self.flush()
        return self.records


def _check_rows(value, what):
    if type(value) is not int or value <= 0 or value % QUARTER:
        raise Unsupported('%s must be a positive multiple of %d rows, got %r' % (what, QUARTER, value))


def served_canon_slice(start):
    """Whether the served slice of the row axis canonicalises: it starts inside a tile (untilize / slice / tilize)."""
    return start % TILE != 0


def served_canon_concat(row_counts):
    """Whether the served concat on the row axis canonicalises every byte of its output: some piece is not a whole number of tiles (every piece is untilized, the pieces
    joined row-major, the result tilized)."""
    return any(rows % TILE for rows in row_counts)


def kv_records(plan, cache_rows, live_rows, kv_heads, *, destination=0, cached_source=None, live_source=None, canon_cached=True):
    """The K (or V) assembly: one tensor (1, kv_heads, sum of the plan's rows, 128) from the plan's pieces. `plan` is draft_attention_branch's list of dicts
    (kind cached / live / pad, user, rows, source slice of the live block - none: rows from 0), `cache_rows` {user: rows of the cached bank}, `live_rows` the rows of the
    live K (or V) tensor. Sources: cached bank of user u is `cached_source(u)` (default u), the live tensor `live_source` (default: one past the highest user).
    Returns (records, total rows). `canon_cached=False` leaves the cached banks raw (the other reading of the served concat; the card probe decides)."""
    users = sorted(set(part['user'] for part in plan))
    cached_source = cached_source or (lambda user: user)
    live_source = (max(users) + 1) if live_source is None else live_source
    unaligned = served_canon_concat([part['rows'] for part in plan])
    pieces, offset = [], 0
    for part in plan:
        rows = part['rows']
        _check_rows(rows, 'a plan piece')
        if part['kind'] == 'cached':
            have = cache_rows[part['user']]
            if have != rows or rows % TILE:
                raise Unsupported('cached bank of user %r has %r rows for a piece of %r (whole tiles required)' % (part['user'], have, rows))
            pieces.append(dict(first=offset, rows=rows, source=cached_source(part['user']), start=0, tiles=rows // TILE,
                               canonical=bool(unaligned and canon_cached)))
        elif part['kind'] in ('live', 'pad'):
            if 'source' in part:
                start = part['source'].start
                if part['source'].stop - start != rows:
                    raise Unsupported('a %s piece of %d rows names %r' % (part['kind'], rows, part['source']))
            elif part['kind'] == 'live':
                raise Unsupported('a live piece without its source rows')
            else:
                start = 0                                    # the pair's pad: rows 0 .. rows of the block
            if start % QUARTER:
                raise Unsupported('a live piece starts at row %d, not a multiple of %d' % (start, QUARTER))
            if start + rows > live_rows:
                raise Unsupported('rows %d..%d of a %d-row live block' % (start, start + rows, live_rows))
            pieces.append(dict(first=offset, rows=rows, source=live_source, start=start, tiles=-(-live_rows // TILE),
                               canonical=bool(unaligned or served_canon_slice(start))))
        else:
            raise Unsupported('plan piece kind %r' % (part['kind'],))
        if offset % QUARTER:
            raise Unsupported('piece offset %d is not a multiple of %d' % (offset, QUARTER))
        offset += rows
    if offset % TILE:
        raise Unsupported('%d assembled rows are not a whole number of tiles' % offset)
    tile_rows = offset // TILE
    starts = [piece['first'] for piece in pieces]
    emit = Emitter()
    for head in range(kv_heads):
        for tile_row in range(tile_rows):
            quarters = []
            for quarter in range(QUARTERS):
                row = tile_row * TILE + quarter * QUARTER
                piece = pieces[bisect_right(starts, row) - 1]
                source_row = piece['start'] + row - piece['first']
                quarters.append((piece['source'], head * piece['tiles'] + source_row // TILE, (source_row % TILE) // QUARTER,
                                 piece['canonical']))
            first = quarters[0]
            whole = all(entry[0] == first[0] and entry[1] == first[1] and entry[2] == index and entry[3] == first[3]
                        for index, entry in enumerate(quarters))
            for column in range(COLUMN_TILES):
                page = (head * tile_rows + tile_row) * COLUMN_TILES + column
                if whole:
                    emit.tile(destination, page, first[0], first[1] * COLUMN_TILES + column, first[3])
                else:
                    emit.mix(destination, page, [(source, source_page * COLUMN_TILES + column, (CANON if canonical else RAW) + quarter)
                                                 for source, source_page, quarter, canonical in quarters])
    return emit.done(), offset


def _fold_geometry(kv_heads, group, halves, users, block):
    for name, value in (('kv_heads', kv_heads), ('group', group), ('halves', halves), ('users', users)):
        if type(value) is not int or value < 1:
            raise Unsupported('%s must be a positive integer, got %r' % (name, value))
    _check_rows(block, 'a user block')
    if users * block != TILE:
        raise Unsupported('%d users of %d rows are not one 32-row tile half' % (users, block))
    return block // QUARTER


def fold_records(kv_heads, group, halves, users, block, *, source=0, destination=0):
    """The query fold: (1, kv_heads * group, 32 * halves, 128) -> (1, kv_heads * halves * users * group, 32, 128). Each 32-row half holds `users` blocks of `block` rows;
    folded head (kv, half, k, j) = kv * halves * users * group + half * users * group + k * group + j is query head kv * group + j of that half with its blocks rotated by k
    (block m takes the half's block (m + k) % users). k = 0 is the half as it is (a raw tile copy); every rotation is the served row-major concat of the rotated pieces
    (canonical). The pair is one half of two 16-row users, the quad two halves of two, the octo pass two halves of four 8-row users."""
    per_block = _fold_geometry(kv_heads, group, halves, users, block)
    emit = Emitter()
    for kv in range(kv_heads):
        for half in range(halves):
            for k in range(users):
                for j in range(group):
                    folded = (kv * halves + half) * users * group + k * group + j
                    query_head = kv * group + j
                    for column in range(COLUMN_TILES):
                        page = folded * COLUMN_TILES + column
                        source_page = (query_head * halves + half) * COLUMN_TILES + column
                        if k == 0:
                            emit.tile(destination, page, source, source_page, False)
                            continue
                        quarters = [None] * QUARTERS
                        for m in range(users):
                            for i in range(per_block):
                                quarters[m * per_block + i] = (source, source_page, CANON + ((m + k) % users) * per_block + i)
                        emit.mix(destination, page, quarters)
    return emit.done()


def unfold_records(kv_heads, group, halves, users, block, *, source=0, destination=0):
    """The output unfold, the inverse layout: (1, kv_heads * halves * users * group, 32, 128) -> (1, kv_heads * group, 32 * halves, 128). Query head kv * group + j of half h
    takes, for each of the half's `users` blocks m, rows 0 .. block - 1 of folded head (kv, h, m, j); the served join is a row-major concat, so every byte is canonical."""
    per_block = _fold_geometry(kv_heads, group, halves, users, block)
    emit = Emitter()
    for kv in range(kv_heads):
        for half in range(halves):
            for j in range(group):
                head = kv * group + j
                for column in range(COLUMN_TILES):
                    quarters = [None] * QUARTERS
                    for m in range(users):
                        folded = (kv * halves + half) * users * group + m * group + j
                        for i in range(per_block):
                            quarters[m * per_block + i] = (source, folded * COLUMN_TILES + column, CANON + i)
                    emit.mix(destination, (head * halves + half) * COLUMN_TILES + column, quarters)
    return emit.done()


# ---------------------------------------------------------------------------------------------
# Words and lanes.
# ---------------------------------------------------------------------------------------------

def weight(record):
    return record[5] if record[0] == 'run' else MIX_WEIGHT


def encode(record):
    """A record's words for the kernel's runtime args."""
    if record[0] == 'run':
        _, destination, destination_page, source, source_page, count, canonical = record
        if not 0 <= destination < 256 or not 0 <= source < 256 or count < 1:
            raise Unsupported('a run outside the kernel\'s index range')
        return [(TYPE_RUN << 28) | ((1 << 24) if canonical else 0) | (destination << 8) | source, count, destination_page, source_page]
    _, destination, destination_page, quarters = record
    words = [(TYPE_MIX << 28) | (destination << 8), destination_page]
    for quarter in quarters:
        if quarter is None or quarter[2] == 0:
            words.append(0)
            continue
        source, source_page, mode = quarter
        if not 0 <= source < 256 or not 0 <= source_page < 65536 or not 1 <= mode <= 8:
            raise Unsupported('a quarter outside the kernel\'s range')
        words.append((source << 24) | (mode << 16) | source_page)
    return words


def words_of(records):
    return [word for record in records for word in encode(record)]


def split_run(record, tiles):
    """(the first `tiles` tiles of a run, the rest)."""
    _, destination, destination_page, source, source_page, count, canonical = record
    return (('run', destination, destination_page, source, source_page, tiles, canonical),
            ('run', destination, destination_page + tiles, source, source_page + tiles, count - tiles, canonical))


def assign(records, lanes):
    """Deal `records`, in order, over at most `lanes` lanes in contiguous runs of about equal weight (runs are split to fill a lane exactly). Lists may be empty at the end."""
    if not records:
        raise Unsupported('At least one move required')
    total = sum(weight(record) for record in records)
    lanes = max(1, min(lanes, total // MIN_LANE_WEIGHT or 1))
    per_lane = -(-total // lanes)
    out = [[] for _ in range(lanes)]
    lane, room = 0, per_lane
    for record in records:
        while True:
            size = weight(record)
            if size <= room or lane == lanes - 1:
                out[lane].append(record)
                room -= size
                if room <= 0 and lane < lanes - 1:
                    lane, room = lane + 1, per_lane
                break
            if record[0] == 'run' and room >= 1:
                head, record = split_run(record, room)
                out[lane].append(head)
                lane, room = lane + 1, per_lane
                continue
            lane, room = lane + 1, per_lane
    return [lane_records for lane_records in out if lane_records]


def capacity(per_lane, sources, destinations):
    """The padded length of every lane's runtime args: 1 + the two address tables + the longest record list."""
    return 1 + sources + destinations + max(len(words_of(records)) for records in per_lane)


def runtime_arguments(per_lane, source_addresses, destination_addresses):
    """Each lane's [words used, source addresses, destination addresses, records..., zero padding to the capacity] from per-chip address lists."""
    size = capacity(per_lane, len(source_addresses), len(destination_addresses))
    if size > MAX_ARGUMENT_WORDS:
        raise Unsupported('a lane needs %d runtime-arg words, over the budget of %d' % (size, MAX_ARGUMENT_WORDS))
    lists = []
    for records in per_lane:
        body = words_of(records)
        words = [len(body)] + list(source_addresses) + list(destination_addresses) + body
        lists.append(words + [0] * (size - len(words)))
    return lists


def plan_lanes(records, grid, sources, destinations, processors=PROCESSORS):
    """(per-lane record lists, capacity, rows). `grid` is the device's (width, height) of worker cores. The launch covers ONE RECTANGLE of the grid - the first `rows` rows, the
    grid's full width (the profile of the 13 x 10 grid shows the served untilizes on a non-rectangular 128-core set paying a 27-38 us dispatch gap before each) - sized to the work:
    lane L runs on core L // processors (row-major), processor L % processors, and a lane without records carries an empty list. Unsupported when a lane's list is over the argument
    budget."""
    width, height = grid
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise Unsupported('a compute grid of %r' % ((width, height),))
    total = sum(weight(record) for record in records)
    wanted = max(1, min(width * height * processors, total // MIN_LANE_WEIGHT or 1))
    cores = -(-wanted // processors)
    rows = min(height, -(-cores // width))
    lanes = rows * width * processors
    per_lane = assign(records, lanes)
    per_lane = per_lane + [[] for _ in range(lanes - len(per_lane))]
    size = capacity(per_lane, sources, destinations)
    if size > MAX_ARGUMENT_WORDS:
        raise Unsupported('%d records do not fit %d lanes: %d words a lane against %d' % (len(records), lanes, size, MAX_ARGUMENT_WORDS))
    return per_lane, size, rows


# ---------------------------------------------------------------------------------------------
# The launch.
# ---------------------------------------------------------------------------------------------

def _placement_problem(operations, tensors):
    for tensor in tensors:
        if tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT:
            return 'an operand is not bfloat16 TILE'
        if tensor.memory_config() != operations.DRAM_MEMORY_CONFIG:
            return 'an operand is not interleaved DRAM'
    return None


def grid_size(mesh):
    """(width, height) of the device's compute grid, read from the device (11 x 10 before the firmware unlock, 13 x 10 after); never a literal."""
    grid = mesh.compute_with_storage_grid_size()
    return grid.x, grid.y


def launch(operations, mesh, sources, destinations, per_lane, size, rows, *, canon_denorm=True, processors=PROCESSORS):
    """One generic_op running the planned lanes (plan_lanes) over per-chip programs, on the one rectangle of `rows` full rows of the grid. Every source shares one accessor layout and
    every destination another; anything else is Unsupported. Returns the number of lanes with records and the number of cores."""
    chips = tp_shapes.chip_count()
    source_shards = [operations.get_device_tensors(tensor) for tensor in sources]
    destination_shards = [operations.get_device_tensors(tensor) for tensor in destinations]
    if any(len(shards) != chips for shards in source_shards + destination_shards):
        raise Unsupported('%s chips required' % tp_shapes.all_chips())
    width, height = grid_size(mesh)
    if rows > height or len(per_lane) != rows * width * processors:
        raise Unsupported('%d lanes do not fill %d rows of a %d x %d grid at %d processors' % (len(per_lane), rows, width, height, processors))
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(width - 1, rows - 1))])
    program = operations.MeshProgramDescriptor()
    for chip in range(chips):
        local_sources = [shards[chip] for shards in source_shards]
        local_destinations = [shards[chip] for shards in destination_shards]
        source_layouts = [list(operations.TensorAccessorArgs(tensor).get_compile_time_args()) for tensor in local_sources]
        destination_layouts = [list(operations.TensorAccessorArgs(tensor).get_compile_time_args()) for tensor in local_destinations]
        if any(layout != source_layouts[0] for layout in source_layouts) or any(layout != destination_layouts[0] for layout in destination_layouts):
            raise Unsupported('Every source (and every destination) of one launch must share an accessor layout')
        arguments = runtime_arguments(per_lane, [tensor.buffer_address() for tensor in local_sources],
                                      [tensor.buffer_address() for tensor in local_destinations])
        if len(arguments[0]) != size:
            raise AssertionError('the planned capacity %d is not the argument length %d' % (size, len(arguments[0])))
        kernels, buffers = [], []
        for processor in range(processors):
            runtime = operations.RuntimeArgs()
            for lane in range(processor, len(per_lane), processors):
                core = lane // processors
                runtime[core % width][core // width] = arguments[lane]
            buffers.append(operations.CBDescriptor(total_size=LANES * TILE_BYTES, core_ranges=cores,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=processor, data_format=operations.bfloat16,
                    page_size=TILE_BYTES, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
            kernels.append(operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(KERNEL)), core_ranges=cores,
                defines=[('CANON_DENORM', '1' if canon_denorm else '0')],
                compile_time_args=[*source_layouts[0], *destination_layouts[0], size, len(sources), len(destinations), processor],
                runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                    processor=getattr(operations.DataMovementProcessor, 'RISCV_%d' % processor),
                    noc=getattr(operations.NOC, 'RISCV_%d_default' % processor))))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=kernels, cbs=buffers)
    seen, io = set(), []
    for tensor in [*sources, *destinations]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            io.append(tensor)
    operations.generic_op(io, program)
    return sum(1 for lane in per_lane if lane), rows * width


# ---------------------------------------------------------------------------------------------
# The audit.
# ---------------------------------------------------------------------------------------------

def compare(operations, pairs, site, shape):
    """Byte-compare each (name, engaged, served) pair on every chip as int16 bit patterns (-0 and +0 differ). Logs the audit marker and returns True, or logs the mismatch
    marker and raises AssertionError."""
    import torch

    bad = []
    for name, mine, theirs in pairs:
        for chip, (left, right) in enumerate(zip(operations.get_device_tensors(mine), operations.get_device_tensors(theirs))):
            a = operations.to_torch(left).contiguous().view(torch.int16)
            b = operations.to_torch(right).contiguous().view(torch.int16)
            if tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
                bad.append((name, chip))
    if bad:
        message = '%s site=%s shape=%s differing=%s' % (MISMATCH, site, shape, bad[:8])
        tp4_sampdraft.log_line(message)
        raise AssertionError(message)
    tp4_sampdraft.log_line('%s exact=True site=%s shape=%s tensors=%d' % (AUDIT, site, shape, len(pairs)))
    return True


def _fall_back(site, shape, reason):
    note(FALLBACK, 'site=%s shape=%s reason=%s' % (site, shape, reason))


def _signature(plan):
    return tuple((part['kind'], part['user'], part['rows'], (part['source'].start, part['source'].stop) if 'source' in part else None) for part in plan)


_CACHE = {}


def _cached(key, build):
    found = _CACHE.get(key)
    if found is None:
        found = _CACHE[key] = build()
    return found


# ---------------------------------------------------------------------------------------------
# The three sites.
# ---------------------------------------------------------------------------------------------

def _plan_for(mesh, records, sources, destinations, processors=PROCESSORS):
    """plan_lanes at this device's grid."""
    return plan_lanes(records, grid_size(mesh), sources, destinations, processors)


def assemble_kv(operations, plan, caches, live, retain, *, served, site, canon_cached=True, processors=PROCESSORS):
    """dict(k=, v=) for the packed users' K and V: one launch, or `served(name)` for each name when this call cannot take it. `caches[u]` is user u's {k, v} bank (1, kv, rows, 128),
    `live` the {k, v, ...} of the proposal block (1, kv, rows, 128); `served(name)` is the served assembly of one tensor (it retains its own outputs); `site` the
    shape label (pair / quad / octo)."""
    kv_heads = tp_shapes.active().draft_kv_heads
    names = ('k', 'v')
    users = len(caches)
    reason, both, total, per_lane, size, rows, mesh = None, None, None, None, None, None, None

    def serve(name):
        with served_only():
            return served(name)

    try:
        live_rows = live['k'].shape[2]
        everything = [live[name] for name in names] + [caches[user][name] for user in range(users) for name in names]
        shapes = [tuple(tensor.shape) for tensor in everything]
        if any(len(shape) != 4 or shape[0] != 1 or shape[1] != kv_heads or shape[3] != HEAD_DIM for shape in shapes):
            reason = 'shapes %r are not (1, %d, rows, %d)' % (sorted(set(shapes))[:3], kv_heads, HEAD_DIM)
        elif live['v'].shape[2] != live_rows or any(caches[user]['v'].shape[2] != caches[user]['k'].shape[2] for user in range(users)):
            reason = 'a K and its V differ in rows'
        else:
            reason = _placement_problem(operations, everything)
        if reason is None:
            cache_rows = dict((user, caches[user]['k'].shape[2]) for user in range(users))
            signature = (_signature(plan), tuple(sorted(cache_rows.items())), live_rows, kv_heads, canon_cached)

            def build():
                # K and V in one launch: sources [K banks per user, live K, V banks per user, live V], destinations [K, V].
                records, rows = [], None
                for index in range(2):
                    part, rows = kv_records(plan, cache_rows, live_rows, kv_heads, destination=index, canon_cached=canon_cached,
                                            cached_source=lambda user, index=index: index * (users + 1) + user,
                                            live_source=index * (users + 1) + users)
                    records.extend(part)
                return records, rows

            both, total = _cached(('kv',) + signature, build)
            mesh = live['k'].device()
            per_lane, size, rows = _plan_for(mesh, both, 2 * (users + 1), 2, processors)
    except Unsupported as failure:
        reason = str(failure)
    except (AttributeError, KeyError, IndexError, TypeError) as failure:
        reason = 'plan or operands not understood: %s' % (failure,)
    if reason is not None:
        _fall_back('kv', site, reason)
        return dict((name, serve(name)) for name in names)
    outputs = [operations.empty((1, kv_heads, total, HEAD_DIM), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh,
                                memory_config=operations.DRAM_MEMORY_CONFIG) for _ in names]
    ordered = []
    for name in names:
        ordered.extend(caches[user][name] for user in range(users))
        ordered.append(live[name])
    try:
        lanes, cores = launch(operations, mesh, ordered, outputs, per_lane, size, rows, processors=processors)
    except Unsupported as failure:
        for tensor in outputs:
            operations.deallocate(tensor)
        _fall_back('kv', site, str(failure))
        return dict((name, serve(name)) for name in names)
    except BaseException:
        for tensor in outputs:
            operations.deallocate(tensor)
        raise
    engaged = dict((name, retain(tensor)) for name, tensor in zip(names, outputs))
    if auditing():
        compare(operations, [(name, engaged[name], serve(name)) for name in names], 'kv', site)
    tiles = sum(weight(record) for record in both if record[0] == 'run') + sum(1 for record in both if record[0] == 'mix')
    note(ENGAGED, 'site=kv shape=%s users=%d kv_heads=%d rows=%d tiles=%d records=%d lanes=%d cores=%d canon_cached=%d' % (
        site, users, kv_heads, total, tiles, len(both), lanes, cores, 1 if canon_cached else 0))
    return engaged


def _fold_site(operations, tensor, retain, *, served, site, halves, users, block, folding, name, processors=PROCESSORS):
    found = tp_shapes.active()
    query_heads, kv_heads = found.draft_heads, found.draft_kv_heads
    group = query_heads // kv_heads
    folded_heads = kv_heads * halves * users * group
    expected = (1, query_heads, TILE * halves, HEAD_DIM) if folding else (1, folded_heads, TILE, HEAD_DIM)
    reason = None
    if tuple(tensor.shape) != expected:
        reason = 'shape %r is not %r' % (tuple(tensor.shape), expected)
    else:
        reason = _placement_problem(operations, [tensor])
    records, mesh, per_lane, size, rows = None, None, None, None, None

    def serve():
        with served_only():
            return served()

    if reason is None:
        try:
            build = fold_records if folding else unfold_records
            records = _cached((name, kv_heads, group, halves, users, block), lambda: build(kv_heads, group, halves, users, block))
            mesh = tensor.device()
            per_lane, size, rows = _plan_for(mesh, records, 1, 1, processors)
        except Unsupported as failure:
            reason = str(failure)
    if reason is not None:
        _fall_back(name, site, reason)
        return serve()
    shape = (1, folded_heads, TILE, HEAD_DIM) if folding else (1, query_heads, TILE * halves, HEAD_DIM)
    output = operations.empty(shape, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    try:
        lanes, cores = launch(operations, mesh, [tensor], [output], per_lane, size, rows, processors=processors)
    except Unsupported as failure:
        operations.deallocate(output)
        _fall_back(name, site, str(failure))
        return serve()
    except BaseException:
        operations.deallocate(output)
        raise
    result = retain(output)
    if auditing():
        compare(operations, [(name, result, serve())], name, site)
    note(ENGAGED, 'site=%s shape=%s halves=%d users=%d block=%d kv_heads=%d records=%d lanes=%d cores=%d' % (
        name, site, halves, users, block, kv_heads, len(records), lanes, cores))
    return result


def fold_query(operations, query, retain, *, served, site, halves, users, block):
    """The folded query (1, kv * halves * users * group, 32, 128) of a (1, heads, 32 * halves, 128) query: one launch, or `served()`."""
    return _fold_site(operations, query, retain, served=served, site=site, halves=halves, users=users, block=block, folding=True, name='fold')


def unfold_output(operations, output, retain, *, served, site, halves, users, block):
    """The packed attention rows (1, heads, 32 * halves, 128) of a folded SDPA output: one launch, or `served()`."""
    return _fold_site(operations, output, retain, served=served, site=site, halves=halves, users=users, block=block, folding=False, name='unfold')
