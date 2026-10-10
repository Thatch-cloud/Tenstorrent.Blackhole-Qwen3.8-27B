"""QWEN_FAST_OCTO_ATTN_BUNDLE (default off, GATE ONLY): the octo block's attention in ceil(8 / N) SDPA launches per layer instead of eight, byte for byte what the
eight single-entry launches compute (tp4/octo-2, Lever 3).

SERVED TODAY (the octo block, docs/tp4-octo.md section 5). Eight users of EIGHT rows: every attention layer makes one K64j decode launch per user, G8B1, flags 0x21
(tail | extent, no KV share at one entry), ONE entry of 8 tokens x 6 query heads (48 rows, 2 row tiles) on 16 cores, so a layer is eight launches one after another and a
pass is 128 of them. The M3 blocks make the same count (4 users x 16 layers x 1 bundle of two shared entries, twice), so the octo pass saves nothing on attention.

BUNDLED. N users' entries go in ONE launch: a stacked query (1, N, 48, 256) (one entry per user, exactly the entry the single launch takes), an (N, width) page table with
each user's own row, an (N,) cur_pos with each user's E - 1 and an (N, 1, 48, 256) narrow tail mask, flags 0x21 again. The users of one launch run concurrently on 16 x N
cores. N is the flag's value (2 to 6); the 8 users are split evenly and contiguously: 2 gives 2 + 2 + 2 + 2, 4 (the recommended value), 5 and 6 give 4 + 4. 3 would give 3 + 3 + 2, which the
fold launch has never run over (stacked queries of two shapes in one launch) and is refused by name.

NO K64j KERNEL OR FACTORY CHANGE, NO GRAFT, NO OP BUILD. The pieces already exist from QWEN_FAST_TP4_SDPA=multi (sdpa_multi_tp, docs/sdpa-multi.md), written for the M3 block's
16-row users, and are reused unchanged: the page-table / cur_pos gather kernel (sdpa_multi_gather_tp.cpp), the compare kernel of the audit (sdpa_multi_audit_tp.cpp), the
block fold kernel (attention_block_fold_tp.cpp, which the V3a lever already runs at eight segments of one group: here its bundles are N groups, N consecutive users'
eight tokens, the Chunk the planner already describes) and the mask kernel (sdpa_multi_mask_tp.cpp, whose `rows` is a runtime argument and whose position is
`word + head / 6` for the entry's own word: only its launch builder is rows-parameterised here). All are generic_op JIT sources launched from Python.

EXACTNESS (byte-identical to the eight single launches; the argument, with the factory lines it rests on, is `exactness()` below). Each entry of the bundle IS the entry of the
single launch: the same 48 query rows in the same tile rows, the same page-table row, the same cur_pos word, the same mask bits, the same program constants (PNHt = 2).
  1. CORES PER ENTRY STAY 16. The factory gives min(110, 16 * B) / B cores to each of B entries (sdpa_decode_program_factory.3e0a69af.cpp:196-209, B the page table's batch at
     :117): 16 for B <= 6, 15 at 7, 13 at 8. plan_problem REFUSES any group where it is not 16, so a group never exceeds 6 users (the reason 8 users are two launches).
  2. SAME PARTITION AND TREE. get_workload_for_core (rt_args_common.1b52c60d.hpp:35-93) depends on (cur_pos, the core's index inside its entry, cores per entry, the 256-key
     chunk) only, and the tree on the core index inside the entry (:240 gives 4 rounds at 16): all unchanged per user. Core PLACEMENT on the grid differs; the arithmetic order
     does not.
  3. SAME PER-ROW OP SEQUENCE. The entry shape is the single launch's (PNHt = 2), so the subblock heights (:398, :407), the granularity define (:785) and the compute config are
     the single launch's exactly. Rows never interact.
  4. SAME DATA. The fold-in, fold-out, page-table gather and mask are row and page copies; the mask of entry u is `mask_host([word_u], 8)[0]`, which a test holds equal bit for
     bit to the pinned narrow mask of the single launch for every start-word class.
What no card has run: a K64j 0x21 launch of B > 1 entries of EIGHT rows. QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT=1 (gate profiles only) runs the eight single launches beside the bundled
ones in every layer and compares the two block outputs word for word in-trace (the multi audit's compare kernel); it is the exactness proof of the arm, and the timed arm never
carries it.

FLAGS (strict; read at the attach, never at import):
    QWEN_FAST_OCTO_ATTN_BUNDLE        unset, '' or '0' off. '2' .. '6': at most that many users per launch. Needs QWEN_FAST_OCTO=live|alternate (the octo block), QWEN_FAST_TP=4
                                      and no QWEN_FAST_TP4_SDPA. Anything else raises.
    QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT  '1': the audit above. Needs the bundle flag. Unset, '' or '0' off; anything else raises.
Markers (the smoke rule is `bundle_problems`): ENGAGED once at the attach, CALL once at the first layer, AUDIT per audited replay, UNQUALIFIED once at the attach.

Stdlib only at import (torch and ttnn are imported by the functions that need them), py 3.7 clean.
"""

from contextlib import contextmanager
import os

import attention_block_fold_tp as fold
import sdpa_multi_tp as multi
import tp_shapes

FLAG = 'QWEN_FAST_OCTO_ATTN_BUNDLE'
AUDIT_FLAG = 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT'
OCTO_FLAG = 'QWEN_FAST_OCTO'
SDPA_FLAG = 'QWEN_FAST_TP4_SDPA'
MIN_PER_LAUNCH, MAX_PER_LAUNCH = 2, 6                         # the factory keeps 16 cores per entry up to 6 entries on the 110-core grid (multi.census)
TP4 = 4
ROWS = 8                                                      # tokens of one octo user
USERS = 8
FLAGS = multi.FLAGS                                           # 0x21: tail | extent
RUNTIME_FILES = ('octo_attn_bundle.py',)
# The JIT kernels and modules this lever reuses unchanged (they ship in the overlay for QWEN_FAST_TP4_SDPA=multi; a test holds them there).
REUSED_FILES = ('sdpa_multi_tp.py', multi.KERNEL_MASK, multi.KERNEL_GATHER, multi.KERNEL_AUDIT, 'attention_block_fold_tp.py', 'attention_block_fold_tp.cpp')

MARKER = '[OCTO-ATTN-BUNDLE]'
ENGAGED = MARKER + ' engaged'
CALL_MARKER = MARKER + ' call'
AUDIT_MARKER = MARKER + ' audit'
AUDIT_MISMATCH = MARKER + ' audit MISMATCH'
UNQUALIFIED = MARKER + ' UNQUALIFIED (gate only)'
UNQUALIFIED_TEXT = ('K64j 0x21 with B > 1 entries of eight rows on 16 cores each == the same entries as single launches (no card has run it) '
                    '[job OA1 audited, OA2 text read and paired timing]')


def head_rows():
    return multi.head_rows()


def entry_rows():
    """Query rows of one octo user's entry: 8 tokens x 6 heads = 48."""
    return ROWS * head_rows()


def entry_tiles():
    """Row tiles of one entry (PNHt): 2."""
    return (entry_rows() + multi.TILE - 1) // multi.TILE


# ---------------------------------------------------------------------------------------------------------------------------
# Env flags.
# ---------------------------------------------------------------------------------------------------------------------------

def _value(name, environ):
    source = os.environ if environ is None else environ
    value = source.get(name)
    return None if value is None else value.strip()


def per_launch(environ=None):
    """The users per launch QWEN_FAST_OCTO_ATTN_BUNDLE asks for (an int in 2 .. 6), or None when it is off. Strict: anything else raises."""
    value = _value(FLAG, environ)
    if value in (None, '', '0'):
        return None
    if not value.isdigit() or value != str(int(value)) or not MIN_PER_LAUNCH <= int(value) <= MAX_PER_LAUNCH:
        raise ValueError('%s must be 0 (off) or the users per launch, %d to %d, got %r' % (FLAG, MIN_PER_LAUNCH, MAX_PER_LAUNCH, value))
    return int(value)


def audit_enabled(environ=None):
    """QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT: exactly '1' audits, unset, empty or '0' does not, anything else raises."""
    value = _value(AUDIT_FLAG, environ)
    if value in (None, '', '0'):
        return False
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (AUDIT_FLAG, value))
    return True


def requested(environ=None):
    """Whether either flag is set to anything but off: what the reader twin asks before it imports this module (a flag-off process never does)."""
    source = os.environ if environ is None else environ
    return any((source.get(name) or '').strip() not in ('', '0') for name in (FLAG, AUDIT_FLAG))


def admission_problems(environ=None):
    """[why the flags cannot be served by this process], the strings serving_octo's admission can refuse with. [] when both are off or the set is servable."""
    source = os.environ if environ is None else environ
    problems = []
    try:
        users = per_launch(source)
        audit = audit_enabled(source)
    except ValueError as failure:
        return [str(failure)]
    if users is None and not audit:
        return []
    if audit and users is None:
        problems.append('%s=1 audits the bundled launches: it needs %s' % (AUDIT_FLAG, FLAG))
    if users is not None and len({len(group) for group in launch_groups(USERS, users)}) != 1:
        problems.append('%s=%d: %s' % (FLAG, users, unequal_reason(launch_groups(USERS, users))))
    if (source.get(OCTO_FLAG) or 'off') not in ('live', 'alternate'):
        problems.append('%s is the octo block\'s attention: it needs %s=live or alternate' % (FLAG, OCTO_FLAG))
    if tp_shapes.chip_count(source) != TP4:
        problems.append('%s is a TP4 lever: it needs QWEN_FAST_TP=4' % FLAG)
    sdpa = (source.get(SDPA_FLAG) or '').strip().lower()
    if sdpa not in ('', '0', 'off'):
        problems.append('%s and %s are two ways to bundle the SDPA launches: set one' % (FLAG, SDPA_FLAG))
    return problems


def selected(environ=None):
    """The users per launch, or None when the lever is off; a flag set that this process cannot serve raises ValueError naming why."""
    users = per_launch(environ)
    audit = audit_enabled(environ)
    if users is None and not audit:
        return None
    problems = admission_problems(environ)
    if problems:
        raise ValueError('; '.join(problems))
    return users


# ---------------------------------------------------------------------------------------------------------------------------
# The shape: what the bundle is written for.
# ---------------------------------------------------------------------------------------------------------------------------

def unequal_reason(groups):
    """Why launch groups of different sizes are refused: attention_block_fold_tp._launch asks every stacked query of one fold launch to share one accessor layout, and
    stacked queries of different batch counts (3, 3 and 2 users) are different shapes. No card has run a fold launch over mixed shapes, so the split must be even."""
    return ('launch groups of unequal sizes %s would put stacked queries of different shapes in one fold launch (attention_block_fold_tp._launch shares one accessor '
            'layout, never run over mixed shapes): use a value that splits the users evenly' % ([len(group) for group in groups],))


def launch_groups(users, size):
    """The users of each launch: `users` consecutive indices split into ceil(users / size) groups of near-equal size, the larger ones first.
    launch_groups(8, 4) is ((0, 1, 2, 3), (4, 5, 6, 7)), launch_groups(8, 3) ((0, 1, 2), (3, 4, 5), (6, 7))."""
    if type(users) is not int or users < 1 or type(size) is not int or size < 1:
        raise ValueError('Positive user and group counts required')
    count = -(-users // size)
    base, extra = divmod(users, count)
    groups, first = [], 0
    for index in range(count):
        width = base + (1 if index < extra else 0)
        groups.append(tuple(range(first, first + width)))
        first += width
    return tuple(groups)


def shape_problem(segments, segment_rows, bundle_counts, groups, mesh_grid):
    """Why this block cannot be bundled, or None. Every segment must be one octo user (eight rows, one bundle of one group), the segments must tile the block in order,
    and the factory must still give every entry of every launch 16 cores."""
    users = len(segments)
    if users < 2:
        return 'at least two users required, got %d' % users
    if len(segment_rows) != users or len(bundle_counts) != users:
        return 'one row count and one bundle count per segment required'
    if any(rows != ROWS for rows in segment_rows):
        return 'every segment must be a T%d octo user, got rows %s' % (ROWS, list(segment_rows))
    if any(count != 1 for count in bundle_counts):
        return 'every octo segment must be one bundle, got %s' % (list(bundle_counts),)
    if [tuple(segment) for segment in segments] != [(ROWS * user, ROWS * (user + 1)) for user in range(users)]:
        return 'segments %s do not tile the block in 8-row users' % ([tuple(segment) for segment in segments],)
    if [user for group in groups for user in group] != list(range(users)):
        return 'the launch groups %s do not cover the users in order' % (list(groups),)
    for group in groups:
        problem = multi.plan_problem(len(group), mesh_grid)
        if problem:
            return problem
    if len({len(group) for group in groups}) != 1:
        return unequal_reason(groups)
    return None


def chunks_of(groups):
    """Per launch, the fold chunk (attention_block_fold_tp.Chunk): the first user's first block token and one eight-token group per user, at offsets 0, 8, 16, ... - the
    consecutive users' tokens, which is what the stacked query's batch slots hold."""
    return [fold.Chunk(ROWS * group[0], [(ROWS * index, ROWS) for index in range(len(group))]) for group in groups]


def mask_tasks(users, rows=ROWS):
    """The mask launch's output tiles: users x ceil(rows * 6 / 32) row tiles x 8 columns, the page written being the task id."""
    tiles = (rows * head_rows() + multi.TILE - 1) // multi.TILE
    return list(range(users * tiles * multi.TILE_COLUMNS))


def mask_arguments(mask_address, positions_addresses, per_core, capacity, rows=ROWS):
    """sdpa_multi_tp.mask_arguments with the entry's rows (the kernel reads it as a runtime argument): [mask address, rows, tasks, positions address of batch 0..7, task
    ids], zero padded to 11 + capacity words."""
    if len(positions_addresses) > multi.MAX_USERS:
        raise ValueError('At most %d users' % multi.MAX_USERS)
    padded = list(positions_addresses) + [0] * (multi.MAX_USERS - len(positions_addresses))
    lists = []
    for core in per_core:
        words = [mask_address, rows, len(core)] + padded + list(core)
        lists.append(words + [0] * (11 + capacity - len(words)))
    return lists


def mask_host(words):
    """The (U, 1, 48, 256) bf16 mask the mask kernel writes for U octo users with these positions words, as int16 bit patterns. Entry u is the pinned narrow mask of the
    single eight-row launch (a test holds it)."""
    return multi.mask_host(words, rows=ROWS)


def exactness():
    """The argument in one place, as lines (the report and the doc quote it)."""
    return [
        'cores per entry: min(110, 16 * B) / B = 16 for B <= 6 (factory :198-200); a launch never holds more than 6 users, so 8 users are 2 launches',
        'partition and tree: get_workload_for_core(cur_pos, core, cores per entry, 256) per user (rt_args_common :35-93), tree rounds :240: unchanged per user',
        'per-row ops: the entry is the single launch\'s 48 rows (PNHt = 2): subblock heights (:398, :407) and MUL_BCAST_GRANULARITY (:785) are the single launch\'s',
        'data movement: fold-in, fold-out, page-table gather and the mask are row and page copies, audited word for word in-trace (QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT)',
    ]


def marker(groups, plans, grid, audit):
    return '%s users=%d launches=%d per_launch=%s flags=0x%x cores_per_entry=%d active_cores=%s grid=%dx%d audit=%d' % (
        ENGAGED, sum(len(group) for group in groups), len(groups), ','.join(str(len(group)) for group in groups), FLAGS,
        plans[0]['cores_per_entry'], ','.join(str(plan['active_cores']) for plan in plans), grid[0], grid[1], int(audit))


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


# ---------------------------------------------------------------------------------------------------------------------------
# Launch builders (ttnn imported inside): the multi builders, with the entry's rows.
# ---------------------------------------------------------------------------------------------------------------------------

def build_mask_program(ttnn, mesh, positions, mask, users, rows=ROWS):
    """sdpa_multi_tp.build_mask_program at `rows` tokens per entry: every user's own positions word -> the (users, 1, rows * 6, 256) mask. -> (program, io tensors)."""
    tasks = mask_tasks(users, rows)
    points = multi._cores(mesh, len(tasks))
    per_core = multi.distribute(tasks, len(points))
    capacity = max(len(core) for core in per_core)
    position_shards, (mask_shards,) = multi._shards(ttnn, positions), multi._shards(ttnn, [mask])

    def per_chip(chip):
        local = [shards[chip] for shards in position_shards]
        local_mask = mask_shards[chip]
        if any(value.buffer_address() == local_mask.buffer_address() for value in local):
            raise ValueError('Mask and position input must not alias')
        compile_args = [*multi._layouts(ttnn, local, 'positions'), *ttnn.TensorAccessorArgs(local_mask).get_compile_time_args(), capacity]
        return compile_args, mask_arguments(local_mask.buffer_address(), [value.buffer_address() for value in local], per_core, capacity, rows)

    return multi._program(ttnn, mesh, multi.KERNEL_MASK, points, 4096, per_chip), multi._unique([*positions, mask])


# ---------------------------------------------------------------------------------------------------------------------------
# The block.
# ---------------------------------------------------------------------------------------------------------------------------

def apply(reader, environ=None):
    """Called by the reader twin after its constructor built the segment readers (and QWEN_FAST_TP4_SDPA's apply, which returned). Flags off: returns None and touches
    nothing. A block that is not the octo block (the M3 blocks' sixteen-row segments): None, silently. The octo block: builds the bundle's buffers and programs, refreshes
    once, logs the ENGAGED and UNQUALIFIED lines, sets reader.multi and returns it. Raises ValueError on any shape the launches are not written for; the caller closes the
    reader."""
    source = os.environ if environ is None else environ
    size = selected(source)
    audit = audit_enabled(source)
    if size is None or not reader_is_octo(reader):
        return None
    if getattr(reader, 'multi', None) is not None:
        raise ValueError('%s: the reader already runs a multi launch (%s)' % (FLAG, SDPA_FLAG))
    if source.get('QWEN_FAST_EXTENT_AUDIT') == '1' and not audit:
        raise ValueError('QWEN_FAST_EXTENT_AUDIT=1 reads back every user\'s narrow masks, which the bundled launches do not refresh; run it with %s=1 (the audit runs the '
                         'single launches beside them and keeps them refreshed)' % AUDIT_FLAG)
    mesh_grid = reader.mesh.compute_with_storage_grid_size()
    grid = (mesh_grid.x, mesh_grid.y)
    users = len(reader.readers)
    groups = launch_groups(users, size)
    for segment in reader.readers:
        applied = tuple(getattr(segment, 'sdpa_modes_applied', None) or ())
        if len(applied) != len(segment.metadata) or any(value != FLAGS for value in applied):
            raise ValueError('%s is qualified at flags 0x%x only, the octo reader runs %s' % (FLAG, FLAGS, ['0x%x' % value for value in applied]))
    problem = shape_problem(reader.segments, [segment.rows for segment in reader.readers], [len(segment.metadata) for segment in reader.readers], groups, grid)
    if problem:
        raise ValueError('%s: %s' % (FLAG, problem))
    block = OctoBundle(reader, groups, grid, audit)
    try:
        block.refresh_once()
    except BaseException:
        block.close()
        raise
    reader.multi = block
    _log(marker(groups, block.plans, grid, audit))
    _log('%s: %s' % (UNQUALIFIED, UNQUALIFIED_TEXT))
    return block


def reader_is_octo(reader):
    """Whether this packed reader is the octo block's: every segment exactly eight rows."""
    segments = getattr(reader, 'segments', None)
    try:
        return len(segments) > 0 and all(last - first == ROWS for first, last in segments)
    except (TypeError, ValueError):
        return False


class OctoBundle(object):
    """The bundled launches of one octo block, duck-typed as the reader twin's `multi` (audit, call, scope, rebound, audit_round, close): per launch a persistent stacked
    table, cur_pos and mask (allocated here, before any capture), the shared SDPA config, the gather and mask programs, and under audit the compare counters."""

    def __init__(self, reader, groups, grid, audit):
        self.reader, self.groups, self.grid, self.audit = reader, tuple(groups), grid, audit
        self.operations, self.mesh = reader.operations, reader.mesh
        self.users = sum(len(group) for group in self.groups)
        self.chunks = chunks_of(self.groups)
        self.plans = []
        self.owned = []
        self.closed = False
        self.scope_expected = self.scope_used = None
        self.slots_seen = 0
        self.calls = self.audit_rounds = 0
        self.call_logged = False
        self.launches = []
        operations = self.operations
        try:
            import torch

            width = reader.page_width
            readers = reader.readers
            if audit:
                self.counters = self._upload(torch.zeros(multi.SLOTS * multi.AUDIT_CORES, multi.COUNTER_WORDS, dtype=torch.int32), operations.int32)
            self.config = operations.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                                       q_chunk_size=multi.QWEN_DECODE_MAGIC | FLAGS, k_chunk_size=multi.CHUNK)
            positions = [segment.positions for segment in readers]
            lent_tables = [segment.metadata[0][1] for segment in readers]
            lent_positions = [segment.cur_pos[0] for segment in readers]
            for tensors in (positions, lent_tables, lent_positions):
                if len({id(tensor) for tensor in tensors}) != len(tensors):
                    raise ValueError('Every user must own its own positions word, table and cur_pos')
            # The gather and mask programs bake THESE buffers (the attach-time positions words, tables and cur_pos): a reader rebound to others later would leave the
            # bundled launches reading stale ones (scope() and rebound() check).
            self.bound = (positions, lent_tables, lent_positions)
            for group in self.groups:
                found = multi.factory_cores(len(group), grid[0] * grid[1])
                self.plans.append(found)
                table = self._upload(torch.zeros(len(group), width, dtype=torch.int32), operations.int32)
                cur_pos = self._upload(torch.zeros(len(group), dtype=torch.int32), operations.int32)
                mask = self._upload(torch.zeros(len(group), 1, entry_rows(), multi.CHUNK, dtype=torch.bfloat16), operations.bfloat16)
                mask_program, mask_io = build_mask_program(operations, self.mesh, [positions[user] for user in group], mask, len(group))
                gather_program, gather_io = multi.build_gather_program(operations, self.mesh, [lent_tables[user] for user in group],
                                                                       [lent_positions[user] for user in group], table, cur_pos)
                self.launches.append(dict(users=group, table=table, cur_pos=cur_pos, mask=mask, mask_program=mask_program, mask_io=mask_io,
                                          gather_program=gather_program, gather_io=gather_io))
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
        """Gather each launch's users' tables and cur_pos into its stacked ones and write its stacked mask: once per forward (in the trace). `count`: books the refresh the
        way the single readers would have (model_batch checks one per bundle per forward), unless the audit's single launches book it themselves."""
        operations = self.operations
        for launch in self.launches:
            operations.generic_op(launch['gather_io'], launch['gather_program'])
            operations.generic_op(launch['mask_io'], launch['mask_program'])
        if count and not self.audit:
            for segment in self.reader.readers:
                segment.refresh_calls += len(segment.metadata)

    def rebound(self):
        """None while every segment still holds the buffers the programs were built on; else why the bundled launches must not run (by identity)."""
        readers = self.reader.readers
        now = ([segment.positions for segment in readers], [segment.metadata[0][1] for segment in readers],
               [segment.cur_pos[0] for segment in readers])
        for name, was, current in zip(('positions word', 'page table', 'cur_pos'), self.bound, now):
            if len(was) != len(current) or any(old is not new for old, new in zip(was, current)):
                return ('The bundled octo attention was built on other %ss than the segments hold now (rebound after attach): its gather and mask programs would read '
                        'stale buffers' % name)
        return None

    @contextmanager
    def scope(self, expected_calls):
        """The shared-mask forward (PackedExtentReplayReader.shared_masks, for the bundled launches): refresh once on entry, exactly `expected_calls` calls inside, none nested."""
        if self.scope_expected is not None:
            raise RuntimeError('Shared-mask forwards cannot nest')
        if type(expected_calls) is not int or not 1 <= expected_calls <= multi.SLOTS:
            raise ValueError('Explicit one-to-%d attention call budget required' % multi.SLOTS)
        rebound = self.rebound()
        if rebound:
            for segment in self.reader.readers:
                segment.failed = True
            raise RuntimeError(rebound)
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
        """One attention layer for the whole block. `served`: under audit, a callable that runs the single launches and returns their block output (a tensor this call
        compares with the bundled launches' and releases)."""
        from tp_addresses import addresses, release_owned

        reader, operations = self.reader, self.operations
        reader.check_open()
        problem = multi.query_problem(query, memory_config, operations, reader.rows)
        if problem:
            raise ValueError('%s cannot serve this call: %s' % (FLAG, problem))
        for segment in reader.readers:
            segment.validate(segment.start)
        if self.scope_expected is None:
            self.refresh()
            slot = self.calls % multi.SLOTS
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
            # One fold-in launch for every launch's stacked query (a Chunk per launch: its users' consecutive eight-token groups), the SDPA launches, one fold-out launch.
            stacked = fold.fold_in(self.mesh, query, self.chunks, owned)
            results = []
            for launch, stack in zip(self.launches, stacked):
                result = operations.transformer.paged_scaled_dot_product_attention_decode(stack, keys, values,
                    page_table_tensor=launch['table'], cur_pos_tensor=launch['cur_pos'], is_causal=False, attn_mask=launch['mask'], scale=scale,
                    program_config=self.config, memory_config=memory_config)
                owned.append(result)
                results.append(result)
            output = fold.fold_out(self.mesh, results, self.chunks, memory_config, owned)
            if served_output is not None:
                program, io = multi.build_audit_program(operations, self.mesh, served_output, output, self.counters, slot)
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
                _log('%s users=%d launches=%d flags=0x%x cores_per_entry=%d audit=%d' % (
                    CALL_MARKER, self.users, len(self.groups), FLAGS, self.plans[0]['cores_per_entry'], int(self.audit)))
            return output
        except BaseException:
            for segment in reader.readers:
                segment.failed = True
            raise
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])

    # -- the audit ---------------------------------------------------------------------------------------------------

    def audit_round(self, round_number):
        """After a replay: read each chip's counter pages, require every layer's words equal and live, log one AUDIT_MARKER line (or AUDIT_MISMATCH and raise). -> the words
        compared (0 when the audit is off)."""
        if not self.audit:
            return 0
        operations = self.operations
        differing = live = compared = 0
        reasons = []
        for chip, shard in enumerate(operations.get_device_tensors(self.counters)):
            counters = operations.to_torch(shard).reshape(multi.SLOTS * multi.AUDIT_CORES, multi.COUNTER_WORDS)
            chip_differing, chip_live, chip_compared, why = multi.audit_verdict(counters, self.slots_seen)
            differing += chip_differing
            live += chip_live
            compared += chip_compared
            reasons.extend('chip %d %s' % (chip, text) for text in why)
        if reasons:
            message = '%s round=%d users=%d launches=%d differing=%d %s' % (AUDIT_MISMATCH, round_number, self.users, len(self.groups), differing,
                                                                          '; '.join(reasons[:4]))
            _log(message)
            raise AssertionError(message)
        self.audit_rounds += 1
        _log('%s %d exact=True users=%d launches=%d layers=%d chips=%d words=%d live=%d' % (
            AUDIT_MARKER, self.audit_rounds, self.users, len(self.groups), self.slots_seen, tp_shapes.chip_count(), compared, live))
        return compared

    def close(self):
        if self.closed:
            return
        from tp_addresses import release_owned

        release_owned(self.operations, self.owned)
        self.owned = []
        self.closed = True


# ---------------------------------------------------------------------------------------------------------------------------
# The smoke rule (pure: the log text and the profile's env).
# ---------------------------------------------------------------------------------------------------------------------------

def bundle_problems(text, env, users=USERS):
    """[problem] for a container log against the profile env. Flag off: not one marker line may appear (a leak). Flag on: the engaged line once with the launch split
    this value gives and flags 0x21 and 16 cores per entry, the call line, the UNQUALIFIED line, no audit mismatch, and under the audit flag at least one passing audit line.
    `text` is the container log, `env` the profile's env dict (strings)."""
    env = env or {}
    lines = (text or '').splitlines()
    mine = [line for line in lines if MARKER in line]
    try:
        size = per_launch(env)
        audit = audit_enabled(env)
    except ValueError as failure:
        return ['the profile\'s flags are malformed: %s' % failure]
    if size is None and not audit:
        return ['%s line in a profile that does not set %s: %s' % (MARKER, FLAG, line.strip()[:160]) for line in mine[:4]]
    problems = []
    if size is None:
        return ['%s=1 is set without %s: nothing is bundled to audit' % (AUDIT_FLAG, FLAG)]
    wanted = ','.join(str(len(group)) for group in launch_groups(users, size))
    engaged = [line for line in lines if ENGAGED in line]
    if len(engaged) != 1:
        problems.append('%s=%d is set and %d engaged lines (%s) were logged, not exactly one: the bundled attention %s' % (
            FLAG, size, len(engaged), ENGAGED, 'never attached' if not engaged else 'attached more than once'))
    else:
        line = engaged[0]
        for fragment in ('users=%d' % users, 'launches=%d' % len(launch_groups(users, size)), 'per_launch=%s' % wanted, 'flags=0x21', 'cores_per_entry=16',
                         'audit=%d' % int(audit)):
            if (fragment + ' ') not in (line + ' '):
                problems.append('the engaged line does not say %s: %s' % (fragment, line.strip()[:200]))
    if not any(CALL_MARKER in line for line in lines):
        problems.append('%s=%d is set and no call line (%s) was logged: the bundled launch was built and never called' % (FLAG, size, CALL_MARKER))
    if not any(UNQUALIFIED in line for line in lines):
        problems.append('no UNQUALIFIED line (%s) was logged: the attach no longer says no card has run B > 1 eight-row entries' % UNQUALIFIED)
    audits = [line for line in lines if AUDIT_MARKER in line]
    mismatched = [line.strip()[:200] for line in audits if AUDIT_MISMATCH in line or 'exact=False' in line]
    problems += ['the bundled octo attention audit found a difference: %s' % line for line in mismatched[:4]]
    if audit and not any('exact=True' in line for line in audits if AUDIT_MISMATCH not in line):
        problems.append('%s=1 is set and no passing audit line (%s <n> exact=True) was logged: nothing was compared' % (AUDIT_FLAG, AUDIT_MARKER))
    if not audit and audits:
        problems.append('an audit line was logged by a profile that does not set %s=1' % AUDIT_FLAG)
    return problems
