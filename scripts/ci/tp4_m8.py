"""The one 128-row packed verify block at eight seats (QWEN_FAST_M8_BLOCK, default off): phase 1, the pure-Python rebinding twins.

WHY. Eight seats run two 64-row verify blocks back to back, reading the weights twice. One 128-row block over the same eight
users reads them once (design: tp4/wave3 block128-design.md, GO WITH CHANGES in block128-review.md, an estimated -25 to -33 ms
per eight-user round at 4k contexts). The review found seven 64-row / 4-user limits the design missed, five of them in
sha256-pinned or evidence-qualified files, and this phase found three more (verifier_engine's widths, the packed rows tables,
the unstack's destination layout). None of the pinned files may be edited: this module REBINDS, at runtime, the names that carry
the limits, the way tp_addresses.install rebinds the two-chip helpers.

PHASE 1 (this file) is what a CPU can prove:

  * the flag grammar (strict 0 / 1, a TP4 lever) and its refusal: with the flag on, serving_startup stops the worker with
    PHASE2_REFUSAL, so nothing serves a 128-row block before phase 2 has built the rest;
  * the twins for the limits that are pure Python or configuration (`TWINS` below), each of which logs an engaged marker the
    first time it changes a decision, and each of which is the original's own call below the 128-row block;
  * install() / uninstall(): called by the census test and, from phase 2, by the startup. Nothing calls install() in a served
    process in phase 1.

PHASE 2 (not here): the 128-row matmul configs and the graft gates, the model forward at 128 rows (AttnPrep and the concat heads
by 64-row halves, the K/V writer's halves mode, the LM head), the packed verifier / buffer pool / admission / serving step wiring
for one block of eight, the idle-segment cap, and the trace and DRAM budgets. m8_limits.py is the table of every limit and what
phase 1 did about it; test_tp4_m8_limits holds each row to the code.

Stdlib only at import, py 3.7. Every twin below, with the flag off or not installed, changes nothing: the originals stay bound.
"""

import os
import sys

import tp_shapes

FLAG = 'QWEN_FAST_M8_BLOCK'
USERS, ROWS_PER_USER, BLOCK_ROWS = 8, 16, 128
# The graft attribute that says the 128-row matmul configs exist (model_config's `_progcfg_128` siblings of the `_64` ones, a
# phase 2 graft section). Without it the 128-row block cannot take the one native call, and the twins say so.
NATIVE_ATTRIBUTE = 'attn_wo_decode_1d_progcfg_128'

PHASE2_REFUSAL = ('%s=1 is refused: phase 1 of the 128-row block (rebinding twins, limits census, matmul byte harness) is built, '
                  'phase 2 (128-row matmul configs and graft gates, the forward at 128 rows, the K/V halves writer, the verifier / '
                  'pool / admission wiring for one block of eight) is not' % FLAG)

ENGAGED = '[PINDIAG] tp4 m8 engaged'
FALLBACK = '[PINDIAG] tp4 m8 fell back'

RUNTIME_FILES = ('tp4_m8.py',)

# What this process did, by site (a test reads and clears it).
STATS = {'engaged': 0, 'fallback': 0}
_LOGGED = set()


def enabled(environ=None):
    """QWEN_FAST_M8_BLOCK: strict 0 or 1 (unset is 0); on at the pair raises (a TP4 lever); anything else raises."""
    source = os.environ if environ is None else environ
    value = source.get(FLAG)
    if value is None or value == '0':
        return False
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    return True


def refuse_until_phase2(environ=None):
    """The startup's one call (serving_startup.start): nothing when the flag is off, ValueError(PHASE2_REFUSAL) when it is on."""
    if enabled(environ):
        raise ValueError(PHASE2_REFUSAL)


def diagnostic(text):
    """One [PINDIAG] line into the server log (loguru where it exists, stderr otherwise). Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def note(kind, site, text):
    """One marker line per (kind, site, text)."""
    key = (kind, site, text)
    STATS['engaged' if kind == ENGAGED else 'fallback'] += 1
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    diagnostic('%s site=%s %s' % (kind, site, text))


# ---------------------------------------------------------------------------------------------------------------------------
# The shape.
# ---------------------------------------------------------------------------------------------------------------------------

def m8_shape(page_width):
    """Eight T16 users in one 128-row block over the serving page-table width. Needs the install()ed packed_shapes (its
    BLOCK_ROWS holds 128): without it validate_shape refuses the shape, as it must."""
    import packed_shapes

    return packed_shapes.validate_shape(packed_shapes.PackedShape(USERS, ROWS_PER_USER, BLOCK_ROWS, page_width,
                                                                  page_width * packed_shapes.PAGE_TOKENS))


# ---------------------------------------------------------------------------------------------------------------------------
# The twins. Each is a function with the original's signature; `original` is the function install() found bound.
# ---------------------------------------------------------------------------------------------------------------------------

def make_sequential_capture_rows(original):
    """packed_shapes.sequential_capture_rows keys the trim on `block_rows == 64`: a 128-row block would answer 16 and the per-request
    engines would capture T16 beside it (the v52 OOM, 32.9 of 33.1 GB at eight engines). Key it on the block being the 64-row M3 block
    or wider (review item 7)."""
    def sequential_capture_rows(shape):
        if shape is not None:
            import packed_shapes

            if packed_shapes.validate_shape(shape).block_rows == BLOCK_ROWS:
                note(ENGAGED, 'sequential_capture_rows', 'rows=%d capture_rows=%d' % (BLOCK_ROWS, packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS))
                return packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS
        return original(shape)

    return sequential_capture_rows


def make_project_qkvzab_by_tile(original):
    """gdn_device_loop_state.project_qkvzab_by_tile (pinned) takes one native call to 64 rows and, above its cap, silently runs one
    32-row call per tile (four at 128 rows: +8.4 to 9.1 ms a verify, with no marker, review item 2). At 128 rows: one native call when
    the graft carries the 128-row configs, else the original's tiled path and a loud 'fell back' line."""
    def project_qkvzab_by_tile(operations, layer, packed, rows):
        if rows != BLOCK_ROWS:
            return original(operations, layer, packed, rows)
        if hasattr(getattr(layer, 'args', None), NATIVE_ATTRIBUTE):
            note(ENGAGED, 'gdn_in_project', 'rows=%d calls=1' % rows)
            return layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        note(FALLBACK, 'gdn_in_project', 'rows=%d: the graft has no %s, so the projection runs one call per 32-row tile (4 weight passes)'
             % (rows, NATIVE_ATTRIBUTE))
        return original(operations, layer, packed, rows)

    return project_qkvzab_by_tile


def make_validate_segments(original, limit=BLOCK_ROWS):
    """pooled_attention_replay.validate_segments's default limit is bound to 64 where it is defined; the evidence-pinned extent reader
    (extent_attention_replay_tp, hashed by packed_any_evidence_tp4.json) calls it with no limit, so eight 16-row segments are refused at
    construction (review item 4). The twin passes 128; the reader's bytes and its per-segment unit are untouched."""
    def validate_segments(segments, block_rows_limit=limit):
        found = original(segments, block_rows_limit)
        if found and found[-1][1] > 64:
            note(ENGAGED, 'extent_segments', 'rows=%d segments=%d limit=%d' % (found[-1][1], len(found), block_rows_limit))
        return found

    return validate_segments


def make_windows_unsupported(original):
    """gdn_conv_windows_packed.unsupported bounds the users by the pinned PAIR module's MAX_USERS (4), not the TP4 sibling's: at eight users
    the T2 packed windows return Unsupported, which also switches off V1's block conv (review item 5). The twin asks the original about
    each group of up to four users (so every per-user check runs unchanged) and answers the first reason."""
    def unsupported(operations, mesh, users):
        users = list(users)
        if len(users) <= 4:
            return original(operations, mesh, users)
        if len(users) > USERS:
            return '%d users outside 1..%d' % (len(users), USERS)
        for first in range(0, len(users), 4):
            reason = original(operations, mesh, users[first:first + 4])
            if reason is not None:
                return 'users %d-%d: %s' % (first, min(first + 4, len(users)) - 1, reason)
        note(ENGAGED, 'windows_users', 'users=%d groups=%d' % (len(users), -(-len(users) // 4)))
        return None

    return unsupported


def unstack_layout(users, width):
    """gdn_block_conv_tp.plan_unstack with the destination layout derived from the user count. The original hard-codes the four-user
    layout (conv u at 0-3, beta 4-7, g 8-11, z 12-15, the windows at 16 + 4u + slot): at eight users conv u 4-7 would land on beta 0-3.
    Here conv u is at u, beta at users + u, g at 2 * users + u, z at 3 * users + u and the windows at 4 * users + 4u + slot: the original's
    exact list at four users (test_tp4_m8_limits holds it), and the order of the destination list stage() builds at any count."""
    import gdn_rows_dma_tp as rows_dma
    import tp_shapes as shapes

    found = shapes.active()
    qkv_columns = rows_dma.tile_columns(found.gdn_qkv)
    z_first = found.gdn_qkv // 32
    z_columns = (found.gdn_a_col - found.gdn_qkv) // 32
    block_columns = rows_dma.tile_columns(width)
    slots = 4
    tasks = []

    def add(plan, source, destination_base, destination_stride=1, destination_offset=0):
        for destination, page, first, second in plan:
            tasks.append((destination_base + destination * destination_stride + destination_offset, page,
                          (source, first[1], first[2]), (source, second[1], second[2])))

    add(rows_dma.unstack_users(users, qkv_columns, 0, qkv_columns), 0, 0)
    add(rows_dma.unstack_users(users, 1, 0, 1), 1, users)
    add(rows_dma.unstack_users(users, 1, 0, 1), 2, 2 * users)
    add(rows_dma.unstack_users(users, block_columns, z_first, z_columns), 3, 3 * users)
    for slot in range(slots):
        add(rows_dma.unstack_users(users, qkv_columns, 0, qkv_columns), 4 + slot, 4 * users, slots, slot)
    return tasks


def make_plan_unstack(original):
    """The original's own list up to four users, the user-count layout above four (the only counts the 128-row block reaches)."""
    def plan_unstack(users, width):
        if users <= 4:
            return original(users, width)
        tasks = unstack_layout(users, width)
        note(ENGAGED, 'unstack_layout', 'users=%d tasks=%d' % (users, len(tasks)))
        return tasks

    return plan_unstack


def launch_capacity(cores, words=None, task_words=None):
    """The most tasks one gdn_rows_dma_tp launch carries on `cores` cores: a core's runtime args are 1 + 8 words a task up to 256."""
    import gdn_rows_dma_tp as rows_dma

    limit = rows_dma.MAX_ARGUMENT_WORDS if words is None else words
    per_word = rows_dma.TASK_WORDS if task_words is None else task_words
    return cores * ((limit - 1) // per_word)


def split_tasks(tasks, cores):
    """The task list as consecutive chunks that each fit one launch on `cores` cores (one chunk when it already fits). The tasks are
    independent page moves into distinct destination pages, so the chunks may run one after another in any split."""
    tasks = list(tasks)
    capacity = launch_capacity(cores)
    if capacity < 1:
        raise ValueError('no task fits a core')
    if not tasks:
        return [tasks]
    parts = max(1, -(-len(tasks) // capacity))
    size = -(-len(tasks) // parts)
    return [tasks[first:first + size] for first in range(0, len(tasks), size)]


def make_launch(original):
    """gdn_rows_dma_tp.launch raises Unsupported when a core would carry more than 31 tasks (3,410 on 110 cores); the eight-user unstack is
    3,600 tasks (review item 6), so V1 was refused AFTER canon, window stack and conv had run in the trace. The twin splits an oversize
    list into the fewest equal launches (1,800 tasks each here, the N1 figure) and is the original's call otherwise."""
    def launch(mesh, sources, destinations, tasks, **options):
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        parts = split_tasks(tasks, min(len(tasks), cores)) if tasks else [tasks]
        if len(parts) == 1:
            return original(mesh, sources, destinations, tasks, **options)
        note(ENGAGED, 'rows_dma_split', 'tasks=%d launches=%d' % (len(tasks), len(parts)))
        for part in parts:
            original(mesh, sources, destinations, part, **options)

    return launch


def validate_users(users):
    """target_packed_pages.validate_users with 128 among the legal block totals (its literal tuple stops at 64)."""
    users = tuple(users)
    if not users:
        raise ValueError('At least one packed user required')
    total = 0
    for user in users:
        pages = user.get('pages') if isinstance(user, dict) else None
        if (not isinstance(user, dict) or type(user.get('start')) is not int
                or type(user.get('rows')) is not int or user['rows'] < 1
                or user['start'] < 0 or pages is None
                or getattr(pages, 'ndim', None) != 2 or pages.shape[0] != 1):
            raise ValueError('Each packed user needs an absolute start, a row count and its own page table')
        total += user['rows']
    # 64: the M3 block, four T16 users (docs/packed-device-step-plan-2026-09-20.md section 5).
    if total not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError('Packed rows must total a legal verifier block width')
    if len({user['pages'].shape[1] for user in users}) != 1:
        raise ValueError('Every packed page table must cover the same number of blocks')
    return users, total


def build_pack(participants, *, block_rows=32):
    """verifier_pack.build_pack with 128 among the legal block widths (its literal tuple stops at 64)."""
    from verifier_pack import GDN_LAYERS

    users = tuple(participants)
    if not users:
        raise ValueError('A packed block needs at least one participant')
    total = sum(user['rows'] for user in users)
    # 64: the M3 block, four T16 participants (docs/packed-device-step-plan-2026-09-20.md section 5).
    if total != block_rows or block_rows not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError('Packed participants must fill exactly one legal block width')
    if len({id(user['pages']) for user in users}) != len(users):
        raise ValueError('Each packed user needs its own page table; a shared one means a shared cache')
    if len({id(slot) for user in users for slot in user['slots']}) != len(users) * GDN_LAYERS:
        raise ValueError('Carried GDN states must not be shared between packed users')
    return [dict(user) for user in users]


# (module, attribute, twin factory or twin function, kind): kind 'factory' calls factory(original) once; 'function' binds the function.
TWINS = (
    ('packed_shapes', 'sequential_capture_rows', make_sequential_capture_rows, 'factory'),
    ('gdn_device_loop_state', 'project_qkvzab_by_tile', make_project_qkvzab_by_tile, 'factory'),
    ('pooled_attention_replay', 'validate_segments', make_validate_segments, 'factory'),
    ('gdn_conv_windows_packed', 'unsupported', make_windows_unsupported, 'factory'),
    ('gdn_block_conv_tp', 'plan_unstack', make_plan_unstack, 'factory'),
    ('gdn_rows_dma_tp', 'launch', make_launch, 'factory'),
    ('target_packed_pages', 'validate_users', validate_users, 'function'),
    ('verifier_pack', 'build_pack', build_pack, 'function'),
)

# (module, attribute, values added): tuples the limits live in. Each is replaced by the original + the new widths; every module that holds the SAME
# tuple object under the same name (`from verifier_engine import VERIFY_WIDTHS`) and, through `ALIASES`, under another name is pointed at the new one.
TABLES = (
    ('packed_shapes', 'BLOCK_ROWS', (BLOCK_ROWS,)),
    ('serving_fast_policy', 'PACKED_BLOCK_WIDTHS', (BLOCK_ROWS,)),
    ('verifier_engine', 'VERIFY_WIDTHS', (BLOCK_ROWS,)),
    ('verifier_engine', 'BLOCK_WIDTHS', (BLOCK_ROWS,)),
    ('gdn_prefix', 'ROW_WIDTHS', (BLOCK_ROWS,)),
    ('gdn_records', 'BLOCK_ROWS', (BLOCK_ROWS,)),
    ('force_argmax', 'SAMPLE_WIDTHS', (BLOCK_ROWS,)),
    ('model_batch', 'BLOCK_WIDTHS', (BLOCK_ROWS,)),
)
ALIASES = (('serving_buffer_pool', 'PACKED_BLOCK_WIDTHS', 'packed_shapes', 'BLOCK_ROWS'),)
# (module, attribute, value): plain numbers.
SCALARS = (
    ('gdn_user_batch_tp', 'MAX_USERS', USERS),
    ('serving_buffer_pool', 'PACKED_BLOCK_ROWS', BLOCK_ROWS),
)

# What install() changed, newest last: (namespace, name, the original object, or _ABSENT).
_ABSENT = object()
_REBOUND = []


def _bind(namespace, name, value):
    _REBOUND.append((namespace, name, namespace.get(name, _ABSENT)))
    namespace[name] = value


def install(environ=None):
    """Rebind the 64-row / 4-user limits to the 128-row / 8-user twins in every loaded module that holds them. Returns how many names
    were rebound (0 when already installed). TP4 only: at the pair nothing may change. The census test and, from phase 2, the
    startup call it; in phase 1 no served process does."""
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        raise ValueError('The 64-row / 4-user limits stay in place at the pair')
    if _REBOUND:
        return 0
    import importlib

    before = len(_REBOUND)
    swaps = []
    for module_name, name, twin, kind in TWINS:
        module = importlib.import_module(module_name)
        old = getattr(module, name)
        swaps.append((name, old, twin(old) if kind == 'factory' else twin))
    tables = []
    for module_name, name, extra in TABLES:
        module = importlib.import_module(module_name)
        old = getattr(module, name)
        tables.append((module_name, name, old, tuple(old) + tuple(value for value in extra if value not in old)))
    for module_name, name, value in SCALARS:
        _bind(importlib.import_module(module_name).__dict__, name, value)
    # Function names: every namespace whose global of the same name IS the original (from m import f, or the module's own).
    for name, old, new in swaps:
        for module in list(sys.modules.values()):
            namespace = getattr(module, '__dict__', None)
            if isinstance(namespace, dict) and namespace.get(name) is old:
                _bind(namespace, name, new)
    for module_name, name, old, new in tables:
        # the table's own module, and every module that holds the SAME tuple under the same name (`from verifier_engine import VERIFY_WIDTHS`)
        _bind(importlib.import_module(module_name).__dict__, name, new)
        for module in list(sys.modules.values()):
            namespace = getattr(module, '__dict__', None)
            if isinstance(namespace, dict) and namespace.get(name) is old:
                _bind(namespace, name, new)
    for module_name, name, source_module, source_name in ALIASES:
        namespace = importlib.import_module(module_name).__dict__
        _bind(namespace, name, importlib.import_module(source_module).__dict__[source_name])
    bound = len(_REBOUND) - before
    note(ENGAGED, 'limits', 'rebound=%d twins=%s tables=%s scalars=%s aliases=%s' % (
        bound, ','.join(name for name, _old, _new in swaps), ','.join('%s.%s' % (module, name) for module, name, _extra in TABLES),
        ','.join('%s.%s' % (module, name) for module, name, _value in SCALARS),
        ','.join('%s.%s' % (module, name) for module, name, _source, _attr in ALIASES)))
    return bound


def uninstall():
    """Put back everything install() replaced (tests only: a served process never goes back)."""
    count = len(_REBOUND)
    while _REBOUND:
        namespace, name, old = _REBOUND.pop()
        if old is _ABSENT:
            namespace.pop(name, None)
        else:
            namespace[name] = old
    _LOGGED.clear()
    STATS['engaged'] = STATS['fallback'] = 0
    return count


def installed():
    return bool(_REBOUND)
