"""fused_commit at any served width: the four-card twin of the round-fence plan's H1b fused commit.

fused_commit.py stays as it is. It is written for the pair (four KV heads per chip, two chips, 16 transport workers per bank, the
pair's text-patched slide scope), it is in the image at the bytes test_quad_draft and test_fused_commit hold, and the pair's
evidence (M-F0, the v200-v205 gate, the C2 production run, the M7 audits) is about those bytes. tp_addresses.install() puts THIS
module in its place in sys.modules at QWEN_FAST_TP=4 only (MODULE_TWINS: packed_verifier, serving_packed_step, verify_prestage and
dflash_proposal_trace import fused_commit lazily, so they reach the twin; the pair never installs and keeps the pinned module).

WHAT CHANGES, and nothing else. Every piece below is the pinned one with its literal width read from tp_shapes; at two chips each
is call for call the pinned one (test_fused_commit_tp4 holds that):
  - the slide scope. The pair admits the served slide only through draft_kv_slide_scope's text-patched DraftKVHistory.prepare,
    which the four-card process never installs (attach_scopes_tp patches nothing). The four-card drafter is
    draft_kv_history_tp.DraftKVHistory, whose prepare calls draft_kv_slide_tp under QWEN_FAST_TP_KV_SLIDE=1: that flag and that
    cache class are the scope here (reason `scope` when either is missing);
  - the banks and deltas: (1, kv, 2048, 128) and (1, kv, 32, 128) with kv = tp_shapes draft_kv_heads (2 at four cards, 4 at the pair);
  - the workers: one per (KV head, 32-column tile) = 4 x kv per bank, 8 at four cards. The served kernel is head-count generic
    (page (head * 64 + tile) * 4 + column, head = worker / 4), so draft_kv_slide.cpp is used AS IT IS, through draft_kv_slide_tp.KERNEL,
    and the in-place argument (each worker reads tiles t and t + 1 before writing tile t, behind its barriers) holds at any head count;
  - the layout: a program carries the same 80 workers as the pair's (its grid check is the pair's), which is TEN banks at four cards
    (8 workers each) - one program per user per chip, where the pair needs two;
  - the chips: tp_shapes.mesh_width(mesh) shards per tensor on the (1, tp) mesh, where the pair loops over its two chips;
  - the seams: the transport is draft_kv_slide_tp.prepare and the K/V projection draft_kv_projection_tp.project_key_value, named
    here so neither the pair's slide driver (not served at four cards, test_tp4_closure_literals) nor the TWINS rebinding is relied on.

WHAT DOES NOT CHANGE. The flags and their readers, every marker and log line (the gate's regexes read one set of lines: the names not
defined here are the pinned module's own objects, module __getattr__), the guards and their reasons, the RoPE tables and the drafts'
fence window, T_proj's op sequence, the audit and its repair, the commit and discard overrides, install_fused_commit and the live-bank
helpers the pair trace and the quad twin call. The pair-only names (KV_SHAPE, DELTA_SHAPE, WORKERS, BANKS_PER_PROGRAM) are refused here
rather than read from the pinned module: a caller that meant the served width must ask kv_shape() / delta_shape() /
workers_per_bank() / banks_per_program().

NOT qualified on hardware: the kernel has run in place at four KV heads only (M-F0). scripts/ci/references/tp4-fcommit-jobs runs the
one-card 2-head proof first, then the audited smoke (QWEN_FAST_FUSED_COMMIT_AUDIT=1: every fused publication is shadowed by today's eager
path and byte-compared on every chip).

Stdlib and torch only at call time (torch inside functions), importable on py 3.7 beside the pinned module.
"""

import os
import time

import draft_kv_slide_tp
import fused_commit as _pinned
import tp_shapes
from fused_commit import (DRAFT_LAYERS, ENGAGED_MARKER, HEADS, HISTORY_ROWS, MARKER, Refused, SegmentStorage, TABLE_SHAPE,
                          FEATURE_WIDTH, audit_enabled, enabled, groups, inplace_enabled, io_list, kernel_kind,
                          live_banks_enabled, log_line)

# The pinned circular buffer (one tuple assignment there: the image's copy-closure check reads simple assignments), read off the module.
CB_BYTES, PAGE_BYTES = _pinned.CB_BYTES, _pinned.PAGE_BYTES

# The pair's own literals, which this module exists to replace: refused by name.
PAIR_ONLY = ('KV_SHAPE', 'DELTA_SHAPE', 'WORKERS', 'BANKS_PER_PROGRAM')
# One served in-place program: the pair's 2 x 80, that is 80 workers a program (the compute grid's own limit is held below).
PROGRAM_WORKERS = _pinned.WORKERS * _pinned.BANKS_PER_PROGRAM
BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_TP_BANKS'   # banks per in-place program at four cards (default: as many as 80 workers hold)


def __getattr__(name):
    """Every name this twin does not define is the pinned module's own (flags, markers, EXPECTED_REFUSALS, log_line, same_bits,
    host_tables, install_fused_commit, live_bank_history, note_live_banks ...), so the two modules cannot describe different fused
    commits. The pair's literal widths are the exception."""
    if name.startswith('__'):
        raise AttributeError(name)
    if name in PAIR_ONLY:
        raise AttributeError('%s is the pair\'s literal width: use kv_shape(), delta_shape(), workers_per_bank() or '
                             'banks_per_program()' % name)
    try:
        return getattr(_pinned, name)
    except AttributeError:
        raise AttributeError('module %r has no attribute %r' % (__name__, name)) from None


# ---------------------------------------------------------------------------------------------
# The width.
# ---------------------------------------------------------------------------------------------

def kv_shape():
    """One chip's K or V bank: (1, kv heads, 2048, 128)."""
    return draft_kv_slide_tp.bank_shape()


def delta_shape():
    """One chip's projected accepted-prefix block: (1, kv heads, 32, 128)."""
    return draft_kv_slide_tp.delta_shape()


def workers_per_bank():
    """One transport worker per (KV head, 32-column tile): 8 at four cards, 16 at the pair."""
    return draft_kv_slide_tp.worker_count()


def banks_per_program(environ=None):
    """Banks one in-place program carries: 80 workers' worth (10 at four cards, 5 at the pair, which is its pinned layout),
    or the divisor of the user's ten banks QWEN_FAST_FUSED_COMMIT_TP_BANKS names (a fallback layout, e.g. 5 = 2 x 40 workers)."""
    environ = os.environ if environ is None else environ
    full = PROGRAM_WORKERS // workers_per_bank()
    text = environ.get(BANKS_FLAG)
    if text is None:
        return full
    if text not in ('1', '2', '5', '10') or int(text) > full:
        raise ValueError('%s must be 1, 2, 5 or 10 and at most %d' % (BANKS_FLAG, full))
    return int(text)


def kernel_path():
    """The slide kernel draft_kv_slide_tp serves: the bundle's draft_kv_slide.cpp, the file draft_kv_slide.prepare's sibling names."""
    return draft_kv_slide_tp.KERNEL


def slide_scope_live():
    """The four-card slide's scope: QWEN_FAST_TP_KV_SLIDE=1 (the class check is the guard's, per cache)."""
    return draft_kv_slide_tp.enabled()


def cache_class():
    from draft_kv_history_tp import DraftKVHistory

    return DraftKVHistory


def slide_program(ttnn, mesh, source, banks, *, history_rows, prefix, in_place=True):
    """One MeshProgramDescriptor for the (1, tp) mesh: draft_kv_slide_tp.prepare's per-chip program, field for field, for any number
    of banks - `banks` is [(active, delta)] in place (the output IS the active bank) or [(active, delta, spare)] out of place. Bank
    i's workers_per_bank() workers take cores w i .. w i + w - 1 in row-major order over the grid, with the served runtime args
    [active, delta, output, history_rows, prefix, drop, rows, worker]. At the pair this is fused_commit.slide_program, and one bank
    out of place is the served driver's program."""
    shape = draft_kv_slide_tp.geometry(history_rows, prefix)
    chips = tp_shapes.mesh_width(mesh)
    if chips is None:
        raise ValueError('The (1, %d) mesh of this width is required' % tp_shapes.chip_count())
    workers = workers_per_bank()
    banks = [tuple(bank) for bank in banks]
    needed = workers * len(banks)
    grid = mesh.compute_with_storage_grid_size()
    if not banks or grid.x * grid.y < needed:
        raise ValueError('%d banks need %d transport workers; the grid has %d cores'
                         % (len(banks), needed, grid.x * grid.y))
    coordinates = [ttnn.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(needed)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=CB_BYTES, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=PAGE_BYTES, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        compile_args, rows = None, []
        for bank in banks:
            if in_place:
                if len(bank) != 2:
                    raise ValueError('In place: one (bank, delta) per bank')
                tensors = [bank[0], bank[1], bank[0]]
            else:
                if len(bank) != 3:
                    raise ValueError('Out of place: one (active, delta, spare) per bank')
                tensors = list(bank)
            shards = [ttnn.get_device_tensors(value) for value in tensors]
            if any(len(parts) != chips for parts in shards):
                raise ValueError('%s independent device shards required' % tp_shapes.count_word())
            local = [parts[chip] for parts in shards]
            for value, expected in zip(local, (kv_shape(), delta_shape(), kv_shape())):
                if (tuple(value.shape) != expected or value.dtype != ttnn.bfloat16
                        or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                        or tuple(value.tile.tile_shape) != (32, 32)
                        or value.tile.transpose_of_faces or value.tile.transpose_within_face):
                    raise ValueError('Exact non-transposed interleaved BF16 cache tiles required')
            addresses = [value.buffer_address() for value in local]
            if in_place:
                if addresses[2] != addresses[0] or addresses[1] == addresses[0]:
                    raise ValueError('In place: the output is the bank, and the delta is not')
            elif len(set(addresses)) != 3:
                raise ValueError('Active, delta and spare storage must not alias')
            arguments = [argument for value in local
                         for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()]
            if compile_args is None:
                compile_args = arguments
            elif arguments != compile_args:
                raise ValueError('Banks with different accessor layouts cannot share one kernel')
            rows.append(addresses)
        kernel = ttnn.KernelDescriptor(kernel_source=str(source), core_ranges=cores, compile_time_args=compile_args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for index, addresses in enumerate(rows):
            for worker in range(workers):
                core = coordinates[index * workers + worker]
                runtime[core.x][core.y] = addresses + [history_rows, prefix, shape['drop'], shape['rows'], worker]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return program


# Seams, looked up at call time (the tests patch them): today's feature projection (the device's own method, unbound, which reads
# its widths from tp_shapes), the four-card K/V projection and the four-card slide transport.
def _project_features(owner, features, count, row_offset, retain):
    from dflash_device import DFlashDevice

    return DFlashDevice.project_features(owner, features, count, row_offset, retain=retain)


def _project_key_value(operations, inputs, query, tables, retain, *, parameters):
    from draft_kv_projection_tp import project_key_value

    return project_key_value(operations, inputs, query, tables, retain, parameters=parameters)


def _transport():
    return draft_kv_slide_tp.prepare


def _addresses():
    from tp_addresses import addresses

    return addresses


class FusedCommit(_pinned.FusedCommit):
    """The pinned FusedCommit with this module's width. Every method that touches a width, the scope or a seam is overridden with the
    pinned body, call for call, the widths replaced; what it inherits (allocate, owner, run_slides, trace_count, the RoPE tables and
    their window, note, commit_kv, discard_kv, audit_round, describe, release_buffers, close) reads none of them."""

    def __init__(self, block, *, operations, mesh, pool, shared_weights, collectives, inplace, audit):
        self.block, self.operations, self.mesh, self.pool = block, operations, mesh, pool
        self.inplace, self.audit = bool(inplace), bool(audit)
        self.rows = block.rows_per_user
        self.closed = False
        self.timing = None
        self.chips = tp_shapes.mesh_width(mesh)
        if self.chips is None:
            raise Refused('the mesh is not the (1, %d) mesh of this width' % tp_shapes.chip_count())
        if collectives is None:
            raise Refused('no collectives were passed to the block (serving_runtime passes the shared TT_CCL)')
        layers = tuple(getattr(shared_weights, 'layers', None) or ())
        if len(layers) != DRAFT_LAYERS or getattr(shared_weights, 'projection', None) is None \
                or getattr(shared_weights, 'feature_norm', None) is None:
            raise Refused('the shared draft weights carry no five prepared layers, projection and norm')
        if not 1 <= self.rows <= 16:
            raise Refused('rows per user %d outside 1..16' % self.rows)
        self.collectives = collectives
        self.projection, self.feature_norm = shared_weights.projection, shared_weights.feature_norm
        # The K/V projection parameters, as DFlashDevice hands DraftKVHistory its layers.
        self.parameters = tuple(layer[0] for layer in layers)
        self.weight_tensors = tuple(getattr(shared_weights, 'tensors', None) or ())
        self.kernel_source = kernel_path()
        self.kernel_kind = kernel_kind(self.kernel_source)
        if self.kernel_kind is None:
            raise Refused('the slide kernel %s is not a qualified kernel' % self.kernel_source)
        self.per_program = banks_per_program()
        grid = mesh.compute_with_storage_grid_size()
        needed = workers_per_bank() * self.per_program
        if self.inplace and grid.x * grid.y < needed:
            raise Refused('the grid has %d cores; the in-place slide needs %d' % (grid.x * grid.y, needed))
        if not slide_scope_live():
            raise Refused('the four-card slide (QWEN_FAST_TP_KV_SLIDE=1) is not live')
        from packed_shapes import segment_rows

        self.segments = []
        for segment, slot in enumerate(block.segment_slots):
            kv = getattr(slot, 'kv', None)
            if (kv is None or len(kv) != DRAFT_LAYERS or getattr(slot, 'query', None) is None
                    or any(set(layer) != {'active', 'spare'} or any(set(layer[side]) != set(HEADS)
                           for side in ('active', 'spare')) for layer in kv)):
                raise Refused('pool slot %s carries no five-layer K/V banks and query' % getattr(slot, 'index', segment))
            self.segments.append(SegmentStorage(segment, slot, segment_rows(block.shape, segment)[0]))
        # The kernel config DFlashDevice builds for its own feature projection.
        self.kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        self.counts = dict(fused=0, today=0, window=0, late=0, audited=0, mismatches=0, discards=0)
        self.refusals = {}
        try:
            for storage in self.segments:
                storage.tables = tuple(self.allocate(storage, TABLE_SHAPE) for name in ('cos', 'sin'))
                storage.deltas = [{name: self.allocate(storage, delta_shape()) for name in HEADS}
                                  for layer in range(DRAFT_LAYERS)]
        except BaseException:
            self.release_buffers()
            raise

    # -- capture (after the GDN commit traces) -----------------------------------------------------
    def retainer(self, storage, owned):
        """T_proj's retain(): keeps every temporary for the trace's life, never a protected buffer (taps, tables, deltas, the slot's
        query, the shared weights), and refuses a partial alias - per chip, at every chip of the mesh."""
        addresses = _addresses()
        operations = self.operations
        protected = [addresses(operations, value) for value in (
            *storage.taps, *storage.tables, *(storage.deltas[layer][name] for layer in range(DRAFT_LAYERS) for name in HEADS),
            storage.query, self.projection, self.feature_norm, *self.weight_tensors)]
        exact = set(protected)
        chips = [set() for _ in range(tp_shapes.chip_count())]
        for identity in protected:
            for chip, address in enumerate(identity):
                chips[chip].add(address)

        def retain(value):
            identity = addresses(operations, value)
            if identity in exact:
                return value
            if any(address in known for address, known in zip(identity, chips)):
                raise ValueError('A fused-commit temporary partially aliases protected storage')
            owned.append(value)
            return value
        return retain

    def project(self, storage, owned):
        """T_proj's body: today's feature projection at count = rows_per_user, today's project_inputs slice and pad, and today's four-card
        K/V projection per layer against the staged tables; each layer's k and v copied into this segment's fixed deltas."""
        operations = self.operations
        retain = self.retainer(storage, owned)
        projected = retain(_project_features(self.owner(), storage.taps, self.rows, storage.row_offset, retain))
        valid = retain(operations.slice(projected, (0, 0, 0, 0), (1, 1, self.rows, FEATURE_WIDTH)))
        inputs = retain(operations.pad(valid, [(0, 0), (0, 0), (0, 32 - self.rows), (0, 0)], 0.0))
        for layer, parameter in enumerate(self.parameters):
            result = _project_key_value(operations, inputs, storage.query, storage.tables, retain, parameters=parameter)
            for name in HEADS:
                operations.copy(result[name], storage.deltas[layer][name])

    def capture(self, capture_operation):
        """Warm and capture every segment's T_proj, then (in place) every (segment, prefix) slide; then the timing replays."""
        from tp_addresses import release_owned

        operations, mesh = self.operations, self.mesh
        taps = tuple(self.block.taps)
        for storage in self.segments:
            storage.taps = taps
            transient = []
            try:
                self.project(storage, transient)
                operations.synchronize_device(mesh)
            finally:
                release_owned(operations, transient)
            storage.projection_trace, unused = capture_operation(
                operations, mesh, lambda storage=storage: self.project(storage, storage.projection_owned))
        if not self.inplace:
            self.measure()
            return
        for storage in self.segments:
            banks = storage.banks()
            storage.slide_programs = {
                prefix: [(chunk, slide_program(operations, mesh, self.kernel_source, chunk,
                                               history_rows=HISTORY_ROWS, prefix=prefix))
                         for chunk in groups(banks, self.per_program)]
                for prefix in range(1, self.rows + 1)}
        # Every program once before any capture (compiled; the pool's banks are lent zeroed on every acquire, so what the warm slides
        # write is wiped before any request reads it).
        for storage in self.segments:
            for prefix in range(1, self.rows + 1):
                self.run_slides(storage.slide_programs[prefix])
        operations.synchronize_device(mesh)
        for storage in self.segments:
            for prefix in range(1, self.rows + 1):
                programs = storage.slide_programs[prefix]
                storage.slides[prefix], unused = capture_operation(operations, mesh,
                                                                   lambda programs=programs: self.run_slides(programs))
        self.measure()

    def measure(self):
        """One blocking replay of every segment's T_proj trace and, in place, of its widest slide trace, timed: the device time of
        this width's fused commit (T_proj is estimated at 1-1.5 ms a user, the slide at under 0.5 ms; nothing has measured them at four
        cards). The banks are the pool's zeroed ones, the tables zero: the bytes are not read."""
        operations, mesh = self.operations, self.mesh
        clock = time.perf_counter
        projection, slide = [], []
        for storage in self.segments:
            if storage.projection_trace is None:
                continue
            started = clock()
            operations.execute_trace(mesh, storage.projection_trace, cq_id=0, blocking=True)
            projection.append(clock() - started)
            if self.inplace and self.rows in storage.slides:
                started = clock()
                operations.execute_trace(mesh, storage.slides[self.rows], cq_id=0, blocking=True)
                slide.append(clock() - started)
        self.timing = dict(tproj_ms=round(1e3 * max(projection), 3) if projection else None,
                           slide_ms=round(1e3 * max(slide), 3) if slide else None)

    def engaged_line(self):
        """The pinned line (the gate's regex reads it) with this width's layout - programs x workers a program - and, after it, the
        width and the attach's replay timings."""
        per = self.per_program
        text = ('%s users=%d rows=%d inplace=%d live_banks=%d audit=%d kernel=%s traces=%d layout=%dx%d'
                % (ENGAGED_MARKER, len(self.segments), self.rows, int(self.inplace), int(live_banks_enabled()),
                   int(self.audit), self.kernel_kind, self.trace_count(),
                   -(-(DRAFT_LAYERS * len(HEADS)) // per), workers_per_bank() * per))
        timing = self.timing or {}
        return '%s tp=%d workers=%d tproj_ms=%s slide_ms=%s' % (
            text, self.chips, workers_per_bank(), timing.get('tproj_ms'), timing.get('slide_ms'))

    # -- the publication ------------------------------------------------------------------------------
    def refusal(self, drafter, segment, features, prefix, position):
        """Why this publication takes today's path, or None (fused_commit's module docstring, GUARDS): the pinned guards in the pinned
        order, with `scope` now the four-card slide flag and the four-card cache class."""
        cache = getattr(drafter, 'kv_history', None)
        if self.closed:
            return 'closed'
        if cache is None:
            return 'no-kv'
        if type(prefix) is not int or not 1 <= prefix <= self.rows:
            return 'prefix'
        if getattr(cache, 'history_rows', None) != HISTORY_ROWS:
            return 'ramp'
        storage = self.segments[segment]
        if storage.poisoned is not None and storage.poisoned is cache:
            return 'poisoned'
        if getattr(drafter, 'pool_slot', None) is not storage.slot:
            return 'slot'
        if getattr(drafter, 'progress', None) is not None:
            return 'progress'
        if getattr(drafter, 'proposal_capture', None) is None:
            return 'no-capture'
        if getattr(cache, 'projection', None) is not None:
            return 'projection'
        if (getattr(drafter, 'pending', None) is not None or getattr(cache, 'pending', None) is not None
                or getattr(drafter, 'closed', False) or getattr(cache, 'closed', False)
                or position != getattr(drafter, 'position', None) or position != getattr(cache, 'position', None)):
            return 'pending'
        if type(position) is not int or position < 0 or position + 32 > 262144:
            # rope_tables' own bound for the 32-row tables T_proj reads (today's needs position + prefix).
            return 'frontier'
        if (getattr(drafter, 'projection', None) is not self.projection
                or getattr(drafter, 'feature_norm', None) is not self.feature_norm):
            return 'weights'
        parameters = tuple(getattr(cache, 'parameters', ()))
        if len(parameters) != len(self.parameters) or any(
                mine is not theirs for mine, theirs in zip(parameters, self.parameters)):
            return 'parameters'
        if getattr(drafter, 'collectives', None) is not self.collectives:
            return 'collectives'
        if (getattr(features, 'row_offset', None) != storage.row_offset or len(tuple(features)) != len(storage.taps)
                or any(mine is not theirs for mine, theirs in zip(features, storage.taps))):
            return 'features'
        if self.inplace:
            if (len(cache.active) != DRAFT_LAYERS or len(cache.spare) != DRAFT_LAYERS or any(
                    cache.active[layer][name] is not storage.active[layer][name]
                    or cache.spare[layer][name] is not storage.spare[layer][name]
                    for layer in range(DRAFT_LAYERS) for name in HEADS)):
                return 'parity'
        else:
            banks = {id(value) for layer in (*storage.active, *storage.spare) for value in layer.values()}
            if {id(value) for layer in (*cache.active, *cache.spare) for value in layer.values()} != banks:
                return 'banks'
        if not slide_scope_live() or not isinstance(cache, cache_class()):
            return 'scope'
        return None

    def prepare(self, drafter, segment, features, prefix, *, position):
        """The fused publication of one user, the pinned one with the four-card transport in the out-of-place branch."""
        from types import SimpleNamespace

        from dflash_traced_publish import PUBLICATION_SPLITS, add_split

        operations, mesh = self.operations, self.mesh
        cache = drafter.kv_history
        storage = self.segments[segment]
        splits = PUBLICATION_SPLITS.get()
        clock = time.perf_counter
        started = clock()
        tables = 'window'
        if storage.staged_position != position:
            self.stage_tables(segment, position)
            tables = 'late'
        staged = clock()
        self.counts[tables] += 1
        reference = None
        if self.audit:
            reference = self.eager_reference(drafter, features, prefix, position, slide=self.inplace)
        enqueued = clock()
        try:
            operations.execute_trace(mesh, storage.projection_trace, cq_id=0, blocking=False)
            if self.inplace:
                # From here the live banks hold this publication: a failure poisons this cache's device state (the request fails
                # with it) rather than pretending nothing moved.
                storage.poisoned = cache
                operations.execute_trace(mesh, storage.slides[prefix], cq_id=0, blocking=False)
                storage.poisoned = None
            else:
                transport = _transport()
                for layer in range(DRAFT_LAYERS):
                    for name in HEADS:
                        transport(mesh, cache.active[layer][name], storage.deltas[layer][name], cache.spare[layer][name],
                                  history_rows=HISTORY_ROWS, prefix=prefix)()
        except BaseException:
            log_line('%s round=%d segment=%d prefix=%s path=failed poisoned=%d' % (
                MARKER, self.block.rounds, segment, prefix, int(storage.poisoned is cache)))
            raise
        executed = clock()
        if reference is not None:
            self.audit_round(drafter, segment, features, prefix, position, reference)
        cache.pending = SimpleNamespace(position=position, prefix=prefix, rows=HISTORY_ROWS, status='prepared',
                                        fused_inplace=self.inplace, fused_segment=segment)
        # C7: nothing on this path reads the feature history, which is no longer written.
        drafter.history_stale = True
        drafter.pending = SimpleNamespace(position=position, prefix=prefix, rows=HISTORY_ROWS,
                                          history=drafter.spare_history, kv=cache.pending, status='prepared')
        self.note(segment, prefix, 'fused', None, tables)
        if splits is not None:
            add_split(splits, 'kv_in', staged - started)
            add_split(splits, 'kv_exec', executed - enqueued)
        return drafter.pending

    # -- the audit -------------------------------------------------------------------------------------
    def eager_reference(self, drafter, features, prefix, position, *, slide):
        """Today's publication, eagerly and fenced: project_features at count = prefix, then DraftKVHistory.project_inputs and the
        four-card K/V projection per layer and, with `slide`, the four-card transport of every bank from active into the spare. Returns
        each layer's k and v rows 0..prefix-1 from every chip, on the host."""
        from tp_addresses import release_owned

        operations = self.operations
        cache = drafter.kv_history
        projected = drafter.project_features(features, prefix)
        try:
            with cache.temporaries([projected]) as retain:
                inputs, tables = cache.project_inputs(projected, prefix, position, retain)
                results = []
                for parameter, active, spare in zip(cache.parameters, cache.active, cache.spare):
                    result = _project_key_value(operations, inputs, cache.query, tables, retain, parameters=parameter)
                    results.append(result)
                    if slide:
                        for name in HEADS:
                            _transport()(cache.mesh, active[name], result[name], spare[name],
                                         history_rows=cache.history_rows, prefix=prefix)()
                operations.synchronize_device(cache.mesh)
                return [{name: [operations.to_torch(shard)[..., :prefix, :].clone()
                                for shard in operations.get_device_tensors(result[name])] for name in HEADS}
                        for result in results]
        finally:
            release_owned(operations, [projected])


def build(block, *, operations, mesh, pool, shared_weights, collectives, diagnostic):
    """The block's FusedCommit under QWEN_FAST_FUSED_COMMIT=1, or None: without the flag, or when a host check refuses it (the pinned
    REFUSED_MARKER says why; the block then serves today's publication)."""
    if not enabled():
        return None
    try:
        return FusedCommit(block, operations=operations, mesh=mesh, pool=pool, shared_weights=shared_weights,
                           collectives=collectives, inplace=inplace_enabled(), audit=audit_enabled())
    except Refused as refusal:
        diagnostic('%s users=%d reason=%s' % (_pinned.REFUSED_MARKER, block.users, str(refusal).replace(' ', '_')[:160]))
        return None
