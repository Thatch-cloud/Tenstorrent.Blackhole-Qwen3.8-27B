"""QWEN_FAST_TP4_SDPA=multi: ONE SDPA decode launch per attention layer for every live user of a packed block, byte for byte
what the per-user launches compute (tp4/sdpa-multi; design K2 of the long-context SDPA design).

SERVED TODAY. The packed verify makes one K64j decode launch per user per attention layer: G8B2, flags 0x23 (tail 0x1 | share
0x2 | extent 0x20), two entries of 8 tokens x 6 heads (48 rows, 2 row tiles), 16 cores per entry, 32 of 110 cores active (16
leaders read the KV, 16 twins receive it by multicast). Four users are four launches run one after another, each reading its
own KV at about 150 GB/s on 32 cores.

MULTI. One launch for the whole block: the query is (1, U, 96, 256), ONE entry per user holding that user's 16 tokens x 6 heads
(96 rows, 3 row tiles, flags 0x21 = tail | extent: no share, every entry reads its own KV once), a (U, width) page table with
each user's own row, a (U,) cur_pos with each user's E - 1, and a (U, 1, 96, 256) narrow tail mask. The users run concurrently
on 16 x U cores.

EXACTNESS (byte-identical to the per-user launches; the argument, with the factory lines it rests on, is `exactness()` below and
docs/sdpa-multi.md; the audit proves it row by row on the card):
  1. CORES PER ENTRY STAY 16. The factory's core allocation (sdpa_decode_program_factory.3e0a69af.cpp:196-209) gives
     min(110, max_cores_per_head_batch * B * kv_heads) / B cores per entry with max_cores_per_head_batch 16 (the config's
     default) and B = the page table's batch (:117): 16 for B <= 6, 15 at B = 7, 13 at B = 8. `plan()` computes it from that
     arithmetic and REFUSES any shape where it is not 16 (7 and 8 users at the served 110-core grid).
  2. SAME PARTITION AND TREE. get_workload_for_core (rt_args_common.1b52c60d.hpp:35-93) depends on (cur_pos, the core's index in
     its entry, cores per entry, the 256-key chunk), and the tree on the core index: all unchanged per user.
  3. SAME PER-ROW OP SEQUENCE. Rows never interact (matmul rows, row max, row sum, column broadcasts). The subblock shapes
     (qk/out_subblock_h = min(PNHt, 8 / 8) = 1, :398, :407) and the granularity defines (MUL_BCAST_GRANULARITY = min(PNHt * 8, 8)
     = 8, :785) do not depend on PNHt at 2 and 3, so regrouping rows from two 48-row entries into one 96-row entry keeps every
     row's instruction sequence. The compute config, chunk size and exp mode are the served ones.
  4. SAME DATA MOVEMENT. The fold-in, the fold-out, the page-table gather and the mask are row and page copies.

Everything below is stdlib at import (torch and ttnn are imported inside the functions that need them) and py 3.7 clean.
"""

from collections import namedtuple
from contextlib import contextmanager
import os
from pathlib import Path

import attention_block_fold_tp as fold
import tp_shapes

HERE = Path(__file__).resolve().parent

MULTI_NAME = 'multi'
AUDIT_FLAG = 'QWEN_FAST_TP4_SDPA_AUDIT'
EXTENT_AUDIT_FLAG = 'QWEN_FAST_EXTENT_AUDIT'

# The markers the smoke rules (c2_smoke_check) read. ENGAGED is sdpa_long_tp.ENGAGED (a test holds the two equal): this module
# does not import it (sdpa_long_tp imports this one, lazily).
ENGAGED = '[PINDIAG] tp4 sdpa engaged'
CALL_MARKER = '[PINDIAG] tp4 sdpa multi call'
AUDIT_MARKER = '[PINDIAG] tp4 sdpa audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 sdpa audit MISMATCH'

QWEN_DECODE_MAGIC = 0x51DEC000                 # pooled_attention_replay.QWEN_DECODE_MAGIC (a test holds the two equal)
TAIL, EXTENT = 0x1, 0x20
FLAGS = TAIL | EXTENT                          # 0x21: no KV share, so every entry reads its own KV
ROWS = 16                                      # tokens of one user in a packed block (T16)
HEAD_DIM = 256
CHUNK = 256
TILE = 32
TILE_COLUMNS = 8                               # 256 / 32 tile columns per token
ENTRY_CORES = 16                               # max_cores_per_head_batch, the SDPAProgramConfig default
MAX_USERS = 8                                  # runtime-argument slots of the mask and gather kernels
SLOTS = 16                                     # attention layers: the shared-mask budget's bound (model_batch: shared_masks(16))
AUDIT_CORES = 64                               # cores of one audit compare; its counters are SLOTS * AUDIT_CORES pages
COUNTER_WORDS = 16                             # int32 words of a counter page
KERNEL_MASK = 'sdpa_multi_mask_tp.cpp'
KERNEL_GATHER = 'sdpa_multi_gather_tp.cpp'
KERNEL_AUDIT = 'sdpa_multi_audit_tp.cpp'
RUNTIME_FILES = ('sdpa_multi_tp.py', KERNEL_MASK, KERNEL_GATHER, KERNEL_AUDIT)

# Where the exactness argument stands in the factory fixture (optimisation/ttnn-op/k64j/fixtures/
# sdpa_decode_program_factory.3e0a69af.cpp) and the work partition header (optimisation/ttnn-op/k64j_probe/fixtures/
# rt_args_common.1b52c60d.hpp). A test reads each quoted line out of those files.
FACTORY_CITATIONS = (
    ('sdpa_decode_program_factory.3e0a69af.cpp', 117, 'B = page_table_tensor->is_sharded() ?'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 196, 'program_config.has_value() ? program_config->max_cores_per_head_batch : num_cores_available;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 198, 'const uint32_t max_num_cores_for_compute = max_cores_per_head * B * num_kv_heads;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 199, 'const uint32_t num_cores_per_batch_uncapped = std::min(num_cores_available, max_num_cores_for_compute) / B;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 200, 'const uint32_t num_cores_per_head = std::max(1u, num_cores_per_batch_uncapped / num_kv_heads);'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 206, 'const uint32_t num_cores_per_batch = num_cores_per_head * num_kv_heads / num_heads_per_core;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 209, 'const uint32_t num_active_cores = num_cores_per_head * num_kv_heads * B / num_heads_per_core;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 240, 'const uint32_t num_tree_reduction_rounds = num_cores_per_head > 1 ? 32 - __builtin_clz(num_cores_per_head - 1) : 0;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 398, 'qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? std::min(PNHt, dst_size / qk_out_subblock_w) : 1;'),
    ('sdpa_decode_program_factory.3e0a69af.cpp', 785, 'add_granularity("MUL_BCAST_GRANULARITY", std::min(PNHt * Sk_chunk_t, dst_size));'),
    ('rt_args_common.1b52c60d.hpp', 35, 'inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_workload_for_core('),
)

Plan = namedtuple('Plan', 'users cores_per_entry active_cores rounds scratch_slots pnht flags')


def head_rows():
    """Folded query rows per token at this process's width: 6 at four cards (one KV head, six query heads)."""
    return tp_shapes.active().attn_fold_rows


def entry_rows():
    """Query rows of one user's entry: 16 tokens x 6 heads = 96."""
    return ROWS * head_rows()


def entry_tiles():
    """Row tiles of one entry (PNHt): 3."""
    return (entry_rows() + TILE - 1) // TILE


# ---------------------------------------------------------------------------------------------------------------------------
# Env flags.
# ---------------------------------------------------------------------------------------------------------------------------

def audit_enabled(environ=None):
    """QWEN_FAST_TP4_SDPA_AUDIT: exactly '1' audits, unset, empty or '0' does not, anything else is a configuration error."""
    source = os.environ if environ is None else environ
    value = source.get(AUDIT_FLAG)
    if value is None or value.strip() in ('', '0'):
        return False
    if value.strip() != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (AUDIT_FLAG, value))
    return True


# ---------------------------------------------------------------------------------------------------------------------------
# The factory's core allocation, as arithmetic.
# ---------------------------------------------------------------------------------------------------------------------------

def factory_cores(users, grid_cores, max_cores=ENTRY_CORES, kv_heads=1, scratch_rounds_env=True):
    """The program factory's allocation for B = `users` entries on `grid_cores` cores, statement for statement
    (sdpa_decode_program_factory.3e0a69af.cpp:196-209 and :240). Returns the numbers the factory line prints."""
    if users < 1 or grid_cores < 1 or max_cores < 1 or kv_heads < 1:
        raise ValueError('Positive users, cores and heads required')
    if grid_cores < users:
        raise ValueError('Cores available (%d) must be >= batch size (%d)' % (grid_cores, users))
    max_num_cores_for_compute = max_cores * users * kv_heads                              # :198
    uncapped = min(grid_cores, max_num_cores_for_compute) // users                       # :199
    per_head = max(1, uncapped // kv_heads)                                               # :200
    heads_per_core = max(1, -(-kv_heads // uncapped)) if uncapped else 1                  # :201-203 (ceil of kv_heads / cores)
    while kv_heads % heads_per_core:
        heads_per_core += 1
    per_batch = per_head * kv_heads // heads_per_core                                     # :206
    active = per_head * kv_heads * users // heads_per_core                                # :209
    rounds = (per_head - 1).bit_length() if per_head > 1 else 0                           # :240
    return dict(cores_per_entry=per_batch, cores_per_head=per_head, active_cores=active, rounds=rounds,
                scratch_slots=rounds if scratch_rounds_env else per_head - 1)


def plan_problem(users, mesh_grid):
    """Why `users` entries cannot be one multi launch on a mesh whose worker grid is `mesh_grid` (x, y), or None. The multi
    launch keeps each user's partition and tree only if the factory still gives it ENTRY_CORES (16) cores."""
    if type(users) is not int or users < 1:
        return 'at least one user required, got %r' % (users,)
    if users > MAX_USERS:
        return '%d users exceed the %d runtime-argument slots of the multi kernels' % (users, MAX_USERS)
    grid_cores = mesh_grid[0] * mesh_grid[1]
    if grid_cores < users:
        return 'the %dx%d grid has %d cores for %d entries' % (mesh_grid[0], mesh_grid[1], grid_cores, users)
    found = factory_cores(users, grid_cores)
    if found['cores_per_entry'] != ENTRY_CORES:
        return ('the factory gives %d cores per entry at B=%d on %d cores (min(%d, 16 * B) / B, sdpa_decode_program_factory '
                ':198-200), not %d: every user would run a different partition and tree than the per-user launch'
                % (found['cores_per_entry'], users, grid_cores, grid_cores, ENTRY_CORES))
    return None


def plan(users, mesh_grid):
    """-> Plan for `users` entries on `mesh_grid`, or ValueError naming why not."""
    problem = plan_problem(users, mesh_grid)
    if problem:
        raise ValueError('QWEN_FAST_TP4_SDPA=multi: %s' % problem)
    found = factory_cores(users, mesh_grid[0] * mesh_grid[1])
    return Plan(users, found['cores_per_entry'], found['active_cores'], found['rounds'], found['scratch_slots'],
                entry_tiles(), FLAGS)


def census(mesh_grid=(11, 10), users=range(1, MAX_USERS + 1)):
    """{users: cores per entry or the refusal} for every user count: the proof table the report quotes."""
    table = {}
    for count in users:
        problem = plan_problem(count, mesh_grid)
        table[count] = factory_cores(count, mesh_grid[0] * mesh_grid[1])['cores_per_entry'] if problem is None else problem
    return table


def exactness():
    """The argument in one place, as lines (the report and the README quote it)."""
    return [
        'cores per entry: min(110, 16 * B) / B = 16 for B <= 6 (factory :198-200); 7 and 8 users give 15 and 13 and are refused',
        'partition and tree: get_workload_for_core(cur_pos, core, cores per entry, 256) per user (rt_args_common :35-93), tree rounds :240',
        'per-row ops: subblock h = min(PNHt, 8 / 8) = 1 (:398, :407), MUL_BCAST_GRANULARITY = min(PNHt * 8, 8) = 8 (:785): PNHt-free at 2 and 3',
        'data movement: fold-in, fold-out, page-table gather and the mask are row and page copies, audited bit for bit on the card',
    ]


def citation_problems(directory):
    """For each FACTORY_CITATIONS entry, what is wrong with it under `directory` (the fixture files), as [string]."""
    problems = []
    for name, line, text in FACTORY_CITATIONS:
        matches = [path for path in Path(directory).rglob(name)]
        if not matches:
            problems.append('%s is not under %s' % (name, directory))
            continue
        lines = matches[0].read_text(encoding='utf-8').splitlines()
        if line > len(lines) or text not in lines[line - 1]:
            problems.append('%s:%d does not hold %r' % (name, line, text))
    return problems


# ---------------------------------------------------------------------------------------------------------------------------
# The block's shape: what the multi launch is written for.
# ---------------------------------------------------------------------------------------------------------------------------

def shape_problem(segments, segment_rows, bundle_counts, mesh_grid):
    """Why this block cannot be one multi launch, or None. Every segment must be one packed T16 user in one bundle of the
    served two eight-row groups (the pinned reader built and qualified them), the segments must tile the block in order, and the
    factory must still give every entry 16 cores."""
    users = len(segments)
    if users < 1:
        return 'no segments'
    if len(segment_rows) != users or len(bundle_counts) != users:
        return 'one row count and one bundle count per segment required'
    if any(rows != ROWS for rows in segment_rows):
        return 'every segment must be a T%d user, got rows %s' % (ROWS, list(segment_rows))
    if any(count != 1 for count in bundle_counts):
        return 'every T%d segment must be one bundle, got %s' % (ROWS, list(bundle_counts))
    if [tuple(segment) for segment in segments] != [(ROWS * user, ROWS * (user + 1)) for user in range(users)]:
        return 'segments %s do not tile the block in 16-row users' % ([tuple(segment) for segment in segments],)
    return plan_problem(users, mesh_grid)


def query_problem(query, memory_config, ttnn, rows):
    """Why the fold launches cannot read this query / write this output, or None."""
    if tuple(query.shape) != (1, rows, head_rows(), HEAD_DIM):
        return 'query %r is not (1, %d, %d, %d)' % (tuple(query.shape), rows, head_rows(), HEAD_DIM)
    if query.dtype != ttnn.bfloat16 or query.layout != ttnn.TILE_LAYOUT:
        return 'query is not bf16 TILE'
    interleaved = (ttnn.DRAM_MEMORY_CONFIG, getattr(ttnn, 'L1_MEMORY_CONFIG', ttnn.DRAM_MEMORY_CONFIG))
    if query.memory_config() not in interleaved:
        return 'query is not interleaved DRAM or L1'
    if memory_config not in interleaved:
        return 'output memory config is not interleaved DRAM or L1'
    return None


# ---------------------------------------------------------------------------------------------------------------------------
# Pure planners (no ttnn): what each launch carries.
# ---------------------------------------------------------------------------------------------------------------------------

def forward_plan(segments):
    """Per user, the fold-in tasks (source base, destination base, task, rows) in attention_block_fold_tp's format: the source
    base is the user's first block token (the kernel reads token tiles first .. first + 15), the destination base the user's
    batch slot in the stacked (1, U, 96, 256) query (3 row tiles x 8 columns per batch)."""
    tiles = entry_tiles()
    return [[(first, user * tiles * TILE_COLUMNS, task, ROWS) for task in range(tiles * TILE_COLUMNS)]
            for user, (first, last) in enumerate(segments)]


def inverse_plan(segments):
    """Per user, the fold-out tasks: the source base is the user's first row tile in the (1, U, 96, 256) result, the destination
    base the user's first token's page in the block output."""
    tiles = entry_tiles()
    return [[(user * tiles, first * TILE_COLUMNS, task, ROWS) for task in range(ROWS * TILE_COLUMNS)]
            for user, (first, last) in enumerate(segments)]


def mask_tasks(users):
    """The mask launch's output tiles: users x 3 row tiles x 8 columns, the page written being the task id."""
    return list(range(users * entry_tiles() * TILE_COLUMNS))


def distribute(tasks, cores):
    """Round-robin `tasks` over at most `cores` cores (no core empty)."""
    count = min(len(tasks), cores)
    if count == 0:
        raise ValueError('At least one task required')
    return [tasks[index::count] for index in range(count)]


def mask_arguments(mask_address, positions_addresses, per_core, capacity):
    """The runtime args of each mask core: [mask address, rows, tasks, positions address of batch 0..7, task ids], zero padded to
    11 + capacity words."""
    if len(positions_addresses) > MAX_USERS:
        raise ValueError('At most %d users' % MAX_USERS)
    padded = list(positions_addresses) + [0] * (MAX_USERS - len(positions_addresses))
    lists = []
    for core in per_core:
        words = [mask_address, ROWS, len(core)] + padded + list(core)
        lists.append(words + [0] * (11 + capacity - len(words)))
    return lists


def gather_arguments(task, users, table_address, position_address, table_addresses, position_addresses):
    """The runtime args of one gather core (20 words): [task, users, stacked table, stacked cur_pos, then per user the lent table
    and lent cur_pos addresses], zero padded."""
    if not len(table_addresses) == len(position_addresses) == users or users > MAX_USERS:
        raise ValueError('One lent table and one lent cur_pos per user, at most %d' % MAX_USERS)
    words = [task, users, table_address, position_address]
    for table, position in zip(table_addresses, position_addresses):
        words += [table, position]
    return words + [0] * (4 + 2 * MAX_USERS - len(words))


def audit_ranges(tiles, cores=AUDIT_CORES):
    """(first tile, tile count) per audit core: contiguous ranges that cover [0, tiles) exactly once; a core past the end gets an
    empty range (and still writes its zero counter page)."""
    per_core = -(-tiles // cores)
    return [(min(core * per_core, tiles), max(0, min(per_core, tiles - core * per_core))) for core in range(cores)]


def audit_verdict(counters, slots_seen):
    """From one chip's counter pages ((SLOTS * AUDIT_CORES) rows of COUNTER_WORDS ints) the verdict
    (differing words, live words, compared words, [reasons])."""
    differing = live = compared = 0
    reasons = []
    for slot in range(slots_seen):
        pages = counters[slot * AUDIT_CORES:(slot + 1) * AUDIT_CORES]
        slot_differing = sum(int(page[0]) for page in pages)
        slot_live = sum(int(page[1]) for page in pages)
        slot_compared = sum(int(page[2]) for page in pages)
        differing += slot_differing
        live += slot_live
        compared += slot_compared
        if slot_differing:
            reasons.append('layer %d: %d of %d words differ' % (slot, slot_differing, slot_compared))
        elif not slot_live or not slot_compared:
            reasons.append('layer %d: nothing was compared (%d words, %d live)' % (slot, slot_compared, slot_live))
    if not slots_seen:
        reasons.append('no attention layer ran the multi launch')
    return differing, live, compared, reasons


def mask_host(words, rows=ROWS):
    """The (U, 1, rows * 6, 256) bf16 mask the mask kernel writes, as int16 bit patterns: row r of entry u masks the keys past
    word_u + r // 6 (0xff80 = -inf in bf16), +0.0 elsewhere."""
    import torch

    heads = head_rows()
    masked = torch.tensor(-128, dtype=torch.int16)           # 0xff80 as int16
    out = torch.zeros(len(words), 1, rows * heads, CHUNK, dtype=torch.int16)
    for user, word in enumerate(words):
        for row in range(rows * heads):
            out[user, 0, row, word + row // heads + 1:] = masked
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Launch builders (ttnn imported inside).
# ---------------------------------------------------------------------------------------------------------------------------

def aligned_page_bytes(shard):
    """The aligned page size of a device buffer, from the runtime: a kernel's stride must be the buffer's own."""
    method = getattr(shard, 'buffer_aligned_page_size', None)
    if method is None:
        raise ValueError('this runtime cannot report a buffer\'s aligned page size (buffer_aligned_page_size), which the multi '
                         'SDPA kernels need for their page stride')
    return int(method())


def _cores(mesh, tasks):
    grid = mesh.compute_with_storage_grid_size()
    return [(index % grid.x, index // grid.x) for index in range(min(tasks, grid.x * grid.y))]


def _shards(ttnn, tensors):
    chips = tp_shapes.chip_count()
    shards = [ttnn.get_device_tensors(tensor) for tensor in tensors]
    if any(len(value) != chips for value in shards):
        raise ValueError('%s chip-local buffers required' % tp_shapes.count_word())
    return shards


def _layouts(ttnn, locals_, what):
    layouts = [ttnn.TensorAccessorArgs(tensor).get_compile_time_args() for tensor in locals_]
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError('Every %s buffer must share one accessor layout' % what)
    return layouts[0]


def _circular_buffer(ttnn, cores, size):
    return ttnn.CBDescriptor(total_size=size, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048,
                                                    tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])


def _unique(tensors):
    seen, io = set(), []
    for tensor in tensors:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            io.append(tensor)
    return io


def _program(ttnn, mesh, kernel, core_points, cb_bytes, per_chip):
    """One program per chip: `per_chip(chip)` -> (compile-time args, [runtime args per core point])."""
    import verify_trace_t1
    import tp_kernels

    cores = verify_trace_t1.rectangle_set(ttnn, core_points)
    buffer = _circular_buffer(ttnn, cores, cb_bytes)
    program = ttnn.MeshProgramDescriptor()
    for chip in range(tp_shapes.chip_count()):
        compile_args, runtime_lists = per_chip(chip)
        descriptor = ttnn.KernelDescriptor(kernel_source=str(HERE / kernel), core_ranges=cores,
            defines=tp_kernels.fold_defines(), compile_time_args=compile_args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for (x, y), arguments in zip(core_points, runtime_lists):
            runtime[x][y] = arguments
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    return program


def build_mask_program(ttnn, mesh, positions, mask, users):
    """The mask launch: every user's own positions word -> the (users, 1, 96, 256) mask. -> (program, io tensors)."""
    tasks = mask_tasks(users)
    points = _cores(mesh, len(tasks))
    per_core = distribute(tasks, len(points))
    capacity = max(len(core) for core in per_core)
    position_shards, (mask_shards,) = _shards(ttnn, positions), _shards(ttnn, [mask])

    def per_chip(chip):
        local = [shards[chip] for shards in position_shards]
        local_mask = mask_shards[chip]
        if any(value.buffer_address() == local_mask.buffer_address() for value in local):
            raise ValueError('Mask and position input must not alias')
        compile_args = [*_layouts(ttnn, local, 'positions'), *ttnn.TensorAccessorArgs(local_mask).get_compile_time_args(), capacity]
        return compile_args, mask_arguments(local_mask.buffer_address(), [value.buffer_address() for value in local],
                                            per_core, capacity)

    return _program(ttnn, mesh, KERNEL_MASK, points, 4096, per_chip), _unique([*positions, mask])


def build_gather_program(ttnn, mesh, lent_tables, lent_positions, table, position):
    """The gather launch: the users' lent tables and cur_pos words -> the stacked (users, width) table and (users,) cur_pos.
    -> (program, io tensors)."""
    users = len(lent_tables)
    points = _cores(mesh, users + 1)
    table_shards, position_shards = _shards(ttnn, lent_tables), _shards(ttnn, lent_positions)
    (table_out,), (position_out,) = _shards(ttnn, [table]), _shards(ttnn, [position])

    def per_chip(chip):
        tables = [shards[chip] for shards in table_shards]
        positions = [shards[chip] for shards in position_shards]
        table_bytes = aligned_page_bytes(table_out[chip])
        position_in_bytes = aligned_page_bytes(positions[0])
        position_out_bytes = aligned_page_bytes(position_out[chip])
        if any(aligned_page_bytes(value) != table_bytes for value in tables):
            raise ValueError('Every lent table must have the stacked table\'s page size')
        compile_args = [*_layouts(ttnn, tables, 'lent table'), *ttnn.TensorAccessorArgs(table_out[chip]).get_compile_time_args(),
                        *_layouts(ttnn, positions, 'lent cur_pos'),
                        *ttnn.TensorAccessorArgs(position_out[chip]).get_compile_time_args(),
                        table_bytes, position_in_bytes, position_out_bytes]
        runtime = [gather_arguments(task, users, table_out[chip].buffer_address(), position_out[chip].buffer_address(),
                                    [value.buffer_address() for value in tables], [value.buffer_address() for value in positions])
                   for task in range(users + 1)]
        return compile_args, runtime

    # The staging area holds one table page, or the stacked cur_pos page and a scratch page.
    cb_bytes = 2048 * (-(-(max(aligned_page_bytes(table_out[0]), 4096) + 256) // 2048))
    return (_program(ttnn, mesh, KERNEL_GATHER, points, cb_bytes, per_chip),
            _unique([*lent_tables, *lent_positions, table, position]))


def build_audit_program(ttnn, mesh, served, multi, counters, slot):
    """One compare launch: the served block output against the multi launch's, into counter pages slot * 64 .. slot * 64 + 63."""
    tiles = served.shape[1] * TILE_COLUMNS
    ranges = audit_ranges(tiles)
    points = _cores(mesh, len(ranges))
    (left_shards, right_shards, counter_shards) = _shards(ttnn, [served, multi, counters])

    def per_chip(chip):
        left, right, count = left_shards[chip], right_shards[chip], counter_shards[chip]
        compile_args = [*_layouts(ttnn, [left, right], 'audit tensor'), *ttnn.TensorAccessorArgs(count).get_compile_time_args(),
                        aligned_page_bytes(count)]
        runtime = [[left.buffer_address(), right.buffer_address(), count.buffer_address(), slot * AUDIT_CORES + core, first, amount]
                   for core, (first, amount) in enumerate(ranges[:len(points)])]
        return compile_args, runtime

    return _program(ttnn, mesh, KERNEL_AUDIT, points, 8192, per_chip), _unique([served, multi, counters])


def _log(message):
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except Exception:  # noqa: BLE001 - a log line never fails a serve
        pass


def marker(plan_, grid, audit):
    return '%s config=%s grid=%dx%d entries=%d users=%d flags=0x%x cores_per_entry=%d active_cores=%d audit=%d' % (
        ENGAGED, MULTI_NAME, grid[0], grid[1], plan_.users, plan_.users, plan_.flags, plan_.cores_per_entry,
        plan_.active_cores, int(audit))


# ---------------------------------------------------------------------------------------------------------------------------
# The block.
# ---------------------------------------------------------------------------------------------------------------------------

def attach(reader, environ=None):
    """Build the multi launch's buffers and programs for `reader` (the packed extent reader twin, after its pinned constructor
    built and qualified the per-user readers), refresh once, log the ENGAGED marker. -> MultiBlock. Raises ValueError on any
    shape the launch is not written for; the caller closes the reader."""
    source = os.environ if environ is None else environ
    audit = audit_enabled(source)
    if source.get(EXTENT_AUDIT_FLAG) == '1' and not audit:
        raise ValueError('%s=1 reads back every user\'s narrow masks, which the multi launch does not refresh; run it with %s=1 '
                         '(the audit runs the per-user launches beside it and keeps them refreshed)' % (EXTENT_AUDIT_FLAG, AUDIT_FLAG))
    mesh_grid = reader.mesh.compute_with_storage_grid_size()
    grid = (mesh_grid.x, mesh_grid.y)
    problem = shape_problem(reader.segments, [segment.rows for segment in reader.readers],
                            [len(segment.metadata) for segment in reader.readers], grid)
    if problem:
        raise ValueError('QWEN_FAST_TP4_SDPA=multi: %s' % problem)
    block = MultiBlock(reader, plan(len(reader.readers), grid), grid, audit)
    try:
        block.refresh_once()
    except BaseException:
        block.close()
        raise
    _log(marker(block.plan, grid, audit))
    return block


class MultiBlock(object):
    """The multi launch of one packed block: persistent stacked table, cur_pos and mask (allocated here, before any capture),
    the SDPA config, the gather and mask programs, and under audit the compare counters."""

    def __init__(self, reader, plan_, grid, audit):
        self.reader, self.plan, self.grid, self.audit = reader, plan_, grid, audit
        self.operations, self.mesh = reader.operations, reader.mesh
        self.users = plan_.users
        self.owned = []
        self.closed = False
        self.scope_expected = self.scope_used = None
        self.slots_seen = 0
        self.calls = self.audit_rounds = 0
        self.call_logged = False
        operations = self.operations
        try:
            import torch

            width = reader.page_width
            readers = reader.readers
            self.table = self._upload(torch.zeros(self.users, width, dtype=torch.int32), operations.int32)
            self.cur_pos = self._upload(torch.zeros(self.users, dtype=torch.int32), operations.int32)
            self.mask = self._upload(torch.zeros(self.users, 1, entry_rows(), CHUNK, dtype=torch.bfloat16), operations.bfloat16)
            if self.audit:
                self.counters = self._upload(torch.zeros(SLOTS * AUDIT_CORES, COUNTER_WORDS, dtype=torch.int32), operations.int32)
            self.config = operations.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                                       q_chunk_size=QWEN_DECODE_MAGIC | FLAGS, k_chunk_size=CHUNK)
            positions = [segment.positions for segment in readers]
            lent_tables = [segment.metadata[0][1] for segment in readers]
            lent_positions = [segment.cur_pos[0] for segment in readers]
            for tensors in (positions, lent_tables, lent_positions):
                if len({id(tensor) for tensor in tensors}) != len(tensors):
                    raise ValueError('Every user must own its own positions word, table and cur_pos')
            self.mask_program, self.mask_io = build_mask_program(operations, self.mesh, positions, self.mask, self.users)
            self.gather_program, self.gather_io = build_gather_program(operations, self.mesh, lent_tables, lent_positions,
                                                                       self.table, self.cur_pos)
        except BaseException:
            self.close()
            raise

    def _upload(self, value, dtype):
        operations = self.operations
        tensor = operations.from_torch(value, device=self.mesh, dtype=dtype,
            layout=operations.ROW_MAJOR_LAYOUT if dtype == operations.int32 else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        self.owned.append(tensor)
        return tensor

    # -- the per-forward refresh -------------------------------------------------------------------------------------

    def refresh_once(self):
        """One eager refresh at attach, so a kernel that does not build fails the attach and not the first capture."""
        self.refresh(count=False)
        self.operations.synchronize_device(self.mesh)

    def refresh(self, count=True):
        """Gather the users' tables and cur_pos into the stacked ones and write the stacked mask: once per forward (in the trace).
        `count`: books the refresh the way the per-user readers would have (model_batch checks one per bundle per forward), unless
        the audit's per-user launches book it themselves."""
        operations = self.operations
        operations.generic_op(self.gather_io, self.gather_program)
        operations.generic_op(self.mask_io, self.mask_program)
        if count and not self.audit:
            for segment in self.reader.readers:
                segment.refresh_calls += len(segment.metadata)

    @contextmanager
    def scope(self, expected_calls):
        """The shared-mask forward (PackedExtentReplayReader.shared_masks, for the multi launch): refresh once on entry, exactly
        `expected_calls` multi calls inside, none nested."""
        if self.scope_expected is not None:
            raise RuntimeError('Shared-mask forwards cannot nest')
        if type(expected_calls) is not int or not 1 <= expected_calls <= SLOTS:
            raise ValueError('Explicit one-to-%d attention call budget required' % SLOTS)
        for segment in self.reader.readers:
            segment.validate(segment.start)
        self.scope_expected, self.scope_used = expected_calls, 0
        try:
            self.refresh()
            yield
            if self.scope_used != expected_calls:
                raise AssertionError('Shared-mask forward did not consume its exact attention call budget')
        except BaseException:
            for segment in self.reader.readers:
                segment.failed = True
            raise
        finally:
            self.scope_expected = None

    # -- the call -----------------------------------------------------------------------------------------------------

    def call(self, query, keys, values, *, scale, memory_config, served=None):
        """One attention layer for the whole block. `served`: under audit, a callable that runs the per-user launches and returns
        their block output (a tensor this call compares with the multi launch's and releases)."""
        from tp_addresses import addresses, release_owned

        reader, operations = self.reader, self.operations
        reader.check_open()
        problem = query_problem(query, memory_config, operations, reader.rows)
        if problem:
            raise ValueError('QWEN_FAST_TP4_SDPA=multi cannot serve this call: %s' % problem)
        for segment in reader.readers:
            segment.validate(segment.start)
        if self.scope_expected is None:
            self.refresh()
            slot = self.calls % SLOTS
        elif self.scope_used >= self.scope_expected:
            raise AssertionError('Shared-mask forward exceeded its attention call budget')
        else:
            slot = self.scope_used
        owned = []
        protected = {addresses(operations, value) for value in (query, keys, values)}
        try:
            served_output = None
            if served is not None:
                served_output = served()
                owned.append(served_output)
            segments = [tuple(segment) for segment in reader.segments]
            stacked = operations.empty((1, self.users, entry_rows(), HEAD_DIM), dtype=operations.bfloat16,
                                       layout=operations.TILE_LAYOUT, device=self.mesh,
                                       memory_config=operations.DRAM_MEMORY_CONFIG)
            owned.append(stacked)
            fold._launch(self.mesh, [query] * self.users, [stacked] * self.users, forward_plan(segments), inverse=False,
                         tag='multi fold-in')
            result = operations.transformer.paged_scaled_dot_product_attention_decode(stacked, keys, values,
                page_table_tensor=self.table, cur_pos_tensor=self.cur_pos, is_causal=False, attn_mask=self.mask, scale=scale,
                program_config=self.config, memory_config=memory_config)
            owned.append(result)
            output = operations.empty((1, reader.rows, head_rows(), HEAD_DIM), dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, device=self.mesh, memory_config=memory_config)
            owned.append(output)
            fold._launch(self.mesh, [result] * self.users, [output] * self.users, inverse_plan(segments), inverse=True,
                         tag='multi fold-out')
            if served_output is not None:
                program, io = build_audit_program(operations, self.mesh, served_output, output, self.counters, slot)
                operations.generic_op(io, program)
            protected.add(addresses(operations, output))
            if self.scope_expected is not None:
                self.scope_used += 1
            self.slots_seen = max(self.slots_seen, slot + 1)
            self.calls += 1
            if served is None:
                for segment in reader.readers:
                    segment.calls += 1
                reader.calls += 1
            if not self.call_logged:
                self.call_logged = True
                _log('%s users=%d flags=0x%x cores_per_entry=%d audit=%d' % (CALL_MARKER, self.users, FLAGS,
                                                                           self.plan.cores_per_entry, int(self.audit)))
            return output
        except BaseException:
            for segment in reader.readers:
                segment.failed = True
            raise
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])

    # -- the audit ---------------------------------------------------------------------------------------------------

    def audit_round(self, round_number):
        """After a replay: read each chip's counter pages, require every layer's words equal and live, log one AUDIT_MARKER line
        (or AUDIT_MISMATCH and raise). -> the words compared (0 when the audit is off)."""
        if not self.audit:
            return 0
        operations = self.operations
        differing = live = compared = 0
        reasons = []
        for chip, shard in enumerate(operations.get_device_tensors(self.counters)):
            counters = operations.to_torch(shard).reshape(SLOTS * AUDIT_CORES, COUNTER_WORDS)
            chip_differing, chip_live, chip_compared, why = audit_verdict(counters, self.slots_seen)
            differing += chip_differing
            live += chip_live
            compared += chip_compared
            reasons.extend('chip %d %s' % (chip, text) for text in why)
        if reasons:
            message = '%s round=%d users=%d differing=%d %s' % (AUDIT_MISMATCH, round_number, self.users, differing,
                                                               '; '.join(reasons[:4]))
            _log(message)
            raise AssertionError(message)
        self.audit_rounds += 1
        _log('%s %d exact=True users=%d layers=%d chips=%d words=%d live=%d' % (
            AUDIT_MARKER, self.audit_rounds, self.users, self.slots_seen, tp_shapes.chip_count(), compared, live))
        return compared

    def close(self):
        if self.closed:
            return
        from tp_addresses import release_owned

        release_owned(self.operations, self.owned)
        self.owned = []
        self.closed = True
