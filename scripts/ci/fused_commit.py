"""Round-fence plan H1b: F-I, the fused commit. Four flags, every one default off:

  QWEN_FAST_FUSED_COMMIT             the projection trace (T_proj) per segment and the publication
                                     overrides; the K/V slides stay today's eager out-of-place
                                     transports into the spare bank, and the commit still swaps
  QWEN_FAST_FUSED_COMMIT_INPLACE     the slides become per-(segment, prefix) traces that slide the
                                     live bank IN PLACE; the commit bumps the frontier, no swap
  QWEN_FAST_FUSED_COMMIT_LIVE_BANKS  F4: the packed pair traces read the pool's live banks
                                     (dflash_proposal_trace), so copy_cache goes; needs _INPLACE
  QWEN_FAST_FUSED_COMMIT_AUDIT       the per-round E2 shadow audit (below); a correctness arm

WHAT IT REPLACES. Each user's packed commit today publishes eagerly (dflash_device.DFlashDevice.
prepare_publication through dflash_traced_publish): project_features (~12 dispatches), the K/V
projection's two RoPE uploads (kv_in, a ~4 ms stall site), five layers of project_key_value (~50
dispatches), ten slide transports (a descriptor built per call) and a fence to release the
temporaries. Under the flag each packed segment owns, from attach:
  - T_PROJ, one captured trace: DFlashDevice.project_features over the block's taps at this
    segment's row offset with count = rows_per_user (16), today's project_inputs slice and pad,
    and project_key_value per learned layer against RoPE tables pre-staged into fixed buffers,
    each layer's k and v copied into this segment's fixed delta buffers. The ops and shapes are
    today's at count 16; the only difference is that rows prefix..15 hold real features where
    today holds zeros. Every op is row-independent (matmul, typecast, rms_norm, the CCL add,
    the rotary; bfp exponents are shared along a face row, one row), so rows 0..prefix-1 - the
    only rows a slide reads - are today's (E2, audited every round below).
  - SLIDES (_INPLACE), one captured trace per (segment, prefix): the SERVED slide kernel file
    (draft_kv_slide.prepare's kernel_source, sha-checked against the qualified kernels) in
    place over the pool slot's ten active banks, two programs of 80 workers (5 banks x 16),
    io [bank, delta, bank] per bank. M-F0 (card B, report slide-20260924T105519): in place ==
    out of place in 88 of 88 cases, generic_op takes the aliased list, the program cache adds
    no entries, the traced 2 x 80 slide of one user's ten banks replays exactly at 0.229 ms.
  - the RoPE tables, written in the drafts' fence window for the next round's frontier
    (verify_prestage.WhileWaiting -> stage_window), or at the commit when the window did not.
A fused publication enqueues T_proj and the slides without a fence (R1: all inside the round's
execute_model; R3: one in-order CQ0, the user's own stream keeps its order - publication, then
its GDN commit trace). The drafts' fence F9, or the last commit's own fence, drains them before
anything reads the banks' next state.

THE OVERRIDES (install_fused_commit, around one commit_entry, like install_publish_options):
drafter.prepare_publication validates, marks the feature history stale (C7: nothing on this path
reads it - the pair trace takes no history, the single-user trace copies it only with an audit
or without a K/V cache) and returns a pending record; the K/V "prepare" is a pending record of
the enqueued work; under _INPLACE kv_history.commit bumps the frontier and history_rows with NO
swap (the live bank never moves, which is what F4 binds). The text of DraftKVHistory.prepare is
untouched, and every refusal takes today's path - install_publish_options' composition,
argument for argument.

GUARDS (refusal): the round takes today's path and logs MARKER path=today reason=R when
  ramp        kv_history.history_rows != 2048
  prefix      the accepted prefix outside 1..rows_per_user
  slot        the drafter's pool slot is not this segment's (block.segment_slots)
  parity      _INPLACE, and kv_history.active is not the pool's active bank (identity, all ten).
              A device holding the spare as active - an odd number of ramp swaps - takes today's
              out-of-place path once, whose commit swaps it back: one parity normalisation
  progress    a proposal audit reporter is set
  no-capture  no captured proposal (C7's condition)
  projection  the draft cache captured its own K/V projection
  weights / parameters / collectives   not the objects T_proj was captured against
  features    not the block's taps at this segment's row offset
  scope       the qualified slide scope (draft_kv_slide_scope.scoped_publication) is not live
  pending / poisoned / no-kv / frontier state the fused path cannot publish from

R2 (hard). Every buffer T_proj or the slides read that is written before the verify replay -
the RoPE tables, and (conservatively) the deltas - is allocated before the block's verify
capture (FusedCommit.__init__ runs right after the taps, before the warm forward). The verify
trace's replay writes its intermediates into the holes its capture freed, so a buffer allocated
after it would be overwritten there. T_proj and the slides are captured after the GDN commit
traces; what they bake besides is pre-trace by construction (the pool's banks and query, the
shared weights, the taps).

THE AUDIT (QWEN_FAST_FUSED_COMMIT_AUDIT). Before each fused launch, today's eager publication:
project_features at count = prefix, project_inputs, project_key_value per layer and - _INPLACE -
the served transport of every bank from active into the pool's SPARE (free scratch in place). It
is fenced and its K/V rows 0..prefix-1 read back. After the fused launch and a fence: every
delta's rows 0..prefix-1 against today's (the E2 claim, both chips) and - _INPLACE - every
active bank against its spare (the in-place claim and the E2 claim together, all ten banks, both
chips, torch.equal on the bit patterns). A mismatch is logged (AUDIT_MISMATCH_MARKER) and the
round repaired to today's bytes (in place: spare copied over active; out of place: today's
publication rerun into the spare), so the arm's text stays exact and the gate fails the arm.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

FLAG = 'QWEN_FAST_FUSED_COMMIT'
INPLACE_FLAG = 'QWEN_FAST_FUSED_COMMIT_INPLACE'
LIVE_BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'
AUDIT_FLAG = 'QWEN_FAST_FUSED_COMMIT_AUDIT'
FLAGS = (FLAG, INPLACE_FLAG, LIVE_BANKS_FLAG, AUDIT_FLAG)

# Once at attach from the block (engaged, or refused with why); one line per publication of a
# packed user (the round is the block's); one per audited publication; the pair trace's F4 once.
ENGAGED_MARKER = '[PINDIAG] fused commit engaged'
REFUSED_MARKER = '[PINDIAG] fused commit refused'
MARKER = '[PACKED-FUSED]'
AUDIT_MARKER = '[PACKED-FUSED-AUDIT]'
AUDIT_MISMATCH_MARKER = '[PINDIAG] fused commit audit mismatch'
DISCARD_MARKER = '[PINDIAG] fused commit discard after in-place slide'
LIVE_BANKS_MARKER = '[PINDIAG] pair live banks'
LIVE_BANKS_NORMALISED = '[PACKED-PROPOSE] live banks'
# The refusals that are expected in a healthy run: the prefill ramp, and one parity
# normalisation per device that reached the steady state holding the spare as active.
EXPECTED_REFUSALS = ('ramp', 'parity')

HISTORY_ROWS = 2048
KV_SHAPE = (1, 4, 2048, 128)
DELTA_SHAPE = (1, 4, 32, 128)
TABLE_SHAPE = (1, 1, 32, 128)
FEATURE_WIDTH = 5120
DRAFT_LAYERS = 5
HEADS = ('k', 'v')
WORKERS = 16                 # per bank: one per (head, 32-column tile), draft_kv_slide.prepare's
BANKS_PER_PROGRAM = 5        # M-F0's verified fast layout: 2 programs x 80 workers per user
CB_BYTES, PAGE_BYTES = 8192, 2048
IO_FORM = 'aliased'          # [bank, delta, bank] per bank: the served arity, output == input
# draft_kv_slide_gate.QUALIFIED_KERNELS' kernels: the checkout's scalar kernel and the direct-DMA
# kernel every fast-serving image serves under the name draft_kv_slide.cpp (M-F0's harness notes).
QUALIFIED_KERNEL_SHA256 = {
    'bc45d47257c844aff4bf17f478b536a48763578e544597884b7d1620083b6ba1': 'scalar',
    '1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47': 'direct',
}


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_FUSED_COMMIT=1. Any value but 0 or 1 is a configuration error."""
    return _flag(FLAG, environ)


def inplace_enabled(environ=None):
    """QWEN_FAST_FUSED_COMMIT_INPLACE=1 under QWEN_FAST_FUSED_COMMIT=1 (alone: inert here, refused
    by the arm and the gate)."""
    return _flag(INPLACE_FLAG, environ) and enabled(environ)


def live_banks_enabled(environ=None):
    """QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1 under the in-place slide (the live bank must never
    move for a pair trace to bind it)."""
    return _flag(LIVE_BANKS_FLAG, environ) and inplace_enabled(environ)


def audit_enabled(environ=None):
    """QWEN_FAST_FUSED_COMMIT_AUDIT=1 under QWEN_FAST_FUSED_COMMIT=1."""
    return _flag(AUDIT_FLAG, environ) and enabled(environ)


def any_enabled(environ=None):
    return enabled(environ)


def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def same_bits(left, right):
    """Equal shape and bits (a bf16 compared as its int16 image)."""
    import torch

    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    image = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(left.dtype)
    if image is None:
        return bool(torch.equal(left, right))
    return bool(torch.equal(left.contiguous().view(image), right.contiguous().view(image)))


def kernel_path():
    """The slide kernel draft_kv_slide.prepare serves: its SIBLING .cpp, exactly as it names it."""
    import draft_kv_slide

    return Path(draft_kv_slide.__file__).with_suffix('.cpp')


def kernel_kind(path):
    """'scalar' or 'direct' for a qualified kernel file, else None."""
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
    return QUALIFIED_KERNEL_SHA256.get(digest)


def slide_scope_live():
    """Whether the qualified slide scope (draft_kv_slide_scope.scoped_publication) is the live
    DraftKVHistory.prepare - the only place the served slide is admitted."""
    import draft_kv_history
    from dflash_traced_publish import _slide_transport_is_recognizable

    return bool(_slide_transport_is_recognizable(draft_kv_history.DraftKVHistory.prepare))


def groups(items, per_program):
    return [items[index:index + per_program] for index in range(0, len(items), per_program)]


def io_list(chunk):
    """generic_op's tensor list for in-place banks [(bank, delta)]: [bank, delta, bank] per bank."""
    tensors = []
    for bank, delta in chunk:
        tensors += [bank, delta, bank]
    return tensors


def slide_program(ttnn, mesh, source, banks, *, history_rows, prefix, in_place=True):
    """One MeshProgramDescriptor for the two chips: draft_kv_slide.prepare's per-chip program,
    field for field, for any number of banks - `banks` is [(active, delta)] in place (the output
    IS the active bank) or [(active, delta, spare)] out of place. Bank i's 16 workers take cores
    16 i .. 16 i + 15 in row-major order over the grid, with the served runtime args [active,
    delta, output, history_rows, prefix, drop, rows, worker]. One bank out of place is the served
    driver's program (test_fused_commit pins it)."""
    import draft_kv_slide

    shape = draft_kv_slide.geometry(history_rows, prefix)
    banks = [tuple(bank) for bank in banks]
    needed = WORKERS * len(banks)
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
    for chip in range(2):
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
            if any(len(parts) != 2 for parts in shards):
                raise ValueError('Two independent device shards required')
            local = [parts[chip] for parts in shards]
            for value, expected in zip(local, (KV_SHAPE, DELTA_SHAPE, KV_SHAPE)):
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
            for worker in range(WORKERS):
                core = coordinates[index * WORKERS + worker]
                runtime[core.x][core.y] = addresses + [history_rows, prefix, shape['drop'], shape['rows'], worker]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return program


def host_tables(position, rows, count):
    """The K/V projection's RoPE tables as DraftKVHistory.project_inputs builds them for `count`
    rows at `position` in a `rows`-row block: rope_tables(position, rows), rows past `count` zero."""
    from draft_head_preparation import rope_tables

    tables = rope_tables(position, rows)
    for table in tables:
        table[..., count:, :] = 0
    return tables


# Seams, looked up at call time (the tests patch them): today's feature projection (the device's
# own method, unbound), the frozen K/V projection, and the served slide transport.
def _project_features(owner, features, count, row_offset, retain):
    from dflash_device import DFlashDevice

    return DFlashDevice.project_features(owner, features, count, row_offset, retain=retain)


def _project_key_value(operations, inputs, query, tables, retain, *, parameters):
    from draft_kv_projection import project_key_value

    return project_key_value(operations, inputs, query, tables, retain, parameters=parameters)


def _transport():
    import draft_kv_slide

    return draft_kv_slide.prepare


class Refused(ValueError):
    """The block cannot build the fused commit (host checks, before anything is allocated)."""


class SegmentStorage:
    """One packed segment's fused-commit state: the pool slot it publishes into, its fixed RoPE
    tables and delta buffers (allocated before the verify capture), its T_proj trace and, in
    place, its slide trace per prefix."""

    def __init__(self, segment, slot, row_offset):
        self.segment, self.slot, self.row_offset = segment, slot, row_offset
        self.tables, self.deltas, self.buffers = (), [], []
        self.query = getattr(slot, 'query', None)
        self.active = [dict(layer['active']) for layer in slot.kv]
        self.spare = [dict(layer['spare']) for layer in slot.kv]
        self.taps = ()
        self.projection_trace, self.projection_owned = None, []
        self.slides, self.slide_programs = {}, {}
        self.staged_position = None
        self.inflight = ()
        # The draft cache (kv_history) whose in-place publication failed after, or was discarded
        # after, its slide moved the live banks - only THAT cache is refused ('poisoned'); its
        # request fails with it. A later request through this segment holds another cache, over
        # banks the pool zeroed at acquire and that cache rewrote, so it is not refused.
        self.poisoned = None

    def banks(self):
        """[(active bank, delta)] in (layer, head) order: the in-place slide's io."""
        return [(self.active[layer][name], self.deltas[layer][name])
                for layer in range(len(self.active)) for name in HEADS]


class FusedCommit:
    """The fused commit of one PackedVerifierEngine (built by it under QWEN_FAST_FUSED_COMMIT).

    Construction does host checks (raising Refused before anything is allocated) and then
    allocates every segment's tables and deltas - the block calls it before its verify capture.
    capture() warms and captures T_proj and the slides after the GDN commit traces."""

    def __init__(self, block, *, operations, mesh, pool, shared_weights, collectives, inplace, audit):
        self.block, self.operations, self.mesh, self.pool = block, operations, mesh, pool
        self.inplace, self.audit = bool(inplace), bool(audit)
        self.rows = block.rows_per_user
        self.closed = False
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
        grid = mesh.compute_with_storage_grid_size()
        if self.inplace and grid.x * grid.y < WORKERS * BANKS_PER_PROGRAM:
            raise Refused('the grid has %d cores; the in-place slide needs %d' % (grid.x * grid.y,
                                                                                  WORKERS * BANKS_PER_PROGRAM))
        if not slide_scope_live():
            raise Refused('the qualified slide scope (QWEN_DRAFT_KV_SLIDE_EXPERIMENT) is not live')
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
                storage.deltas = [{name: self.allocate(storage, DELTA_SHAPE) for name in HEADS}
                                  for layer in range(DRAFT_LAYERS)]
        except BaseException:
            self.release_buffers()
            raise

    def allocate(self, storage, shape):
        import torch

        operations = self.operations
        value = operations.from_torch(torch.zeros(shape, dtype=torch.bfloat16), device=self.mesh,
            dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        storage.buffers.append(value)
        return value

    def allocated(self):
        """Every buffer the fused commit allocated before the verify capture, in order."""
        return [value for storage in self.segments for value in storage.buffers]

    # -- capture (after the GDN commit traces) -----------------------------------------------------
    def owner(self):
        """What DFlashDevice.project_features reads off its device: the same objects every device
        borrows from the shared weights, the shared collectives and the device's kernel config."""
        return SimpleNamespace(operations=self.operations, mesh=self.mesh, collectives=self.collectives,
                               projection=self.projection, feature_norm=self.feature_norm, kernel=self.kernel)

    def retainer(self, storage, owned):
        """T_proj's retain(): keeps every temporary for the trace's life, never a protected buffer
        (taps, tables, deltas, the slot's query, the shared weights), and refuses a partial alias."""
        from gdn_multitoken_conv import addresses

        operations = self.operations
        protected = [addresses(operations, value) for value in (
            *storage.taps, *storage.tables, *(storage.deltas[layer][name] for layer in range(DRAFT_LAYERS) for name in HEADS),
            storage.query, self.projection, self.feature_norm, *self.weight_tensors)]
        exact = set(protected)
        chips = [set() for _ in range(2)]
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
        """T_proj's body: today's feature projection at count = rows_per_user, today's project_inputs
        slice and pad, and today's K/V projection per layer against the staged tables; each layer's
        k and v copied into this segment's fixed deltas."""
        operations = self.operations
        retain = self.retainer(storage, owned)
        projected = retain(_project_features(self.owner(), storage.taps, self.rows, storage.row_offset, retain))
        valid = retain(operations.slice(projected, (0, 0, 0, 0), (1, 1, self.rows, FEATURE_WIDTH)))
        inputs = retain(operations.pad(valid, [(0, 0), (0, 0), (0, 32 - self.rows), (0, 0)], 0.0))
        for layer, parameter in enumerate(self.parameters):
            result = _project_key_value(operations, inputs, storage.query, storage.tables, retain, parameters=parameter)
            for name in HEADS:
                operations.copy(result[name], storage.deltas[layer][name])

    def run_slides(self, programs):
        for chunk, program in programs:
            self.operations.generic_op(io_list(chunk), program)

    def capture(self, capture_operation):
        """Warm and capture every segment's T_proj, then (in place) every (segment, prefix) slide."""
        from gdn_multitoken_conv import release_owned

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
            return
        for storage in self.segments:
            banks = storage.banks()
            storage.slide_programs = {
                prefix: [(chunk, slide_program(operations, mesh, self.kernel_source, chunk,
                                               history_rows=HISTORY_ROWS, prefix=prefix))
                         for chunk in groups(banks, BANKS_PER_PROGRAM)]
                for prefix in range(1, self.rows + 1)}
        # Every program once before any capture (compiled; the pool's banks are lent zeroed on
        # every acquire, so what the warm slides write is wiped before any request reads it).
        for storage in self.segments:
            for prefix in range(1, self.rows + 1):
                self.run_slides(storage.slide_programs[prefix])
        operations.synchronize_device(mesh)
        for storage in self.segments:
            for prefix in range(1, self.rows + 1):
                programs = storage.slide_programs[prefix]
                storage.slides[prefix], unused = capture_operation(operations, mesh,
                                                                   lambda programs=programs: self.run_slides(programs))

    def trace_count(self):
        return sum((storage.projection_trace is not None) + len(storage.slides) for storage in self.segments)

    def engaged_line(self):
        return ('%s users=%d rows=%d inplace=%d live_banks=%d audit=%d kernel=%s traces=%d layout=%dx%d'
                % (ENGAGED_MARKER, len(self.segments), self.rows, int(self.inplace), int(live_banks_enabled()),
                   int(self.audit), self.kernel_kind, self.trace_count(),
                   -(-(DRAFT_LAYERS * len(HEADS)) // BANKS_PER_PROGRAM), WORKERS * BANKS_PER_PROGRAM))

    # -- the RoPE tables ------------------------------------------------------------------------------
    def stage_tables(self, segment, position):
        """Write this segment's T_proj RoPE tables for `position` (no fence: the window's F9 or the
        in-order CQ0 ahead of T_proj). The host tensors stay alive until the next write."""
        operations = self.operations
        storage = self.segments[segment]
        storage.staged_position = None
        payloads = [operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                                          mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
                    for value, destination in zip(host_tables(position, 32, self.rows), storage.tables)]
        for payload, destination in zip(payloads, storage.tables):
            operations.copy_host_to_device_tensor(payload, destination)
        storage.inflight = payloads
        storage.staged_position = position

    def stage_window(self, requests):
        """The drafts' fence window (verify_prestage.WhileWaiting): each live request's segment gets
        its tables for its next frontier. Never raises: a segment that fails is staged late."""
        block = self.block
        for request in requests:
            try:
                if getattr(request.session, 'finished', False):
                    continue
                segment = block.segment_of(request.engine)
                position = request.session.position
                if self.segments[segment].staged_position != position:
                    self.stage_tables(segment, position)
            except Exception as failure:
                log_line('%s round=%d window-tables failed: %s' % (MARKER, block.rounds + 1,
                                                                    repr(failure)[:160]))

    # -- the publication ------------------------------------------------------------------------------
    def refusal(self, drafter, segment, features, prefix, position):
        """Why this publication takes today's path, or None (module docstring, GUARDS)."""
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
        if not slide_scope_live():
            return 'scope'
        return None

    def note(self, segment, prefix, path, reason, tables='-'):
        key = 'fused' if path == 'fused' else 'today'
        self.counts[key] += 1
        if reason is not None:
            self.refusals[reason] = self.refusals.get(reason, 0) + 1
        log_line('%s round=%d segment=%d prefix=%s path=%s reason=%s tables=%s' % (
            MARKER, self.block.rounds, segment, prefix, path, reason or '-', tables))

    def prepare(self, drafter, segment, features, prefix, *, position):
        """The fused publication of one user: its tables (if the window did not stage them), the
        audit's reference, T_proj and the slides enqueued with no fence, the audit's comparison,
        and the pending records. Returns the drafter's pending publication."""
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
                # From here the live banks hold this publication: a failure poisons this cache's
                # device state (the request fails with it) rather than pretending nothing moved.
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

    def commit_kv(self, cache, publication):
        """kv_history.commit under the in-place slide: the frontier moves, the banks do not."""
        if not getattr(publication, 'fused_inplace', False):
            return type(cache).commit(cache, publication)
        if (cache.closed or publication is not cache.pending or publication.status != 'prepared'
                or publication.position != cache.position):
            raise ValueError('Only the current prepared draft cache may commit')
        cache.position += publication.prefix
        cache.history_rows = publication.rows
        publication.status = 'committed'
        cache.pending = None

    def discard_kv(self, cache, publication):
        """kv_history.discard: a fused in-place publication has already slid the live banks, so its
        discard leaves them ahead of the frontier - logged and this cache poisoned on its segment
        (the request fails with its runtime), then today's discard."""
        if getattr(publication, 'fused_inplace', False) and publication.status == 'prepared':
            segment = getattr(publication, 'fused_segment', None)
            if segment is not None:
                self.segments[segment].poisoned = cache
            self.counts['discards'] += 1
            log_line('%s segment=%s position=%s prefix=%s' % (DISCARD_MARKER, segment, publication.position,
                                                              publication.prefix))
        return type(cache).discard(cache, publication)

    # -- the audit -------------------------------------------------------------------------------------
    def eager_reference(self, drafter, features, prefix, position, *, slide):
        """Today's publication, eagerly and fenced: project_features at count = prefix, then
        DraftKVHistory.project_inputs and project_key_value per layer and, with `slide`, the served
        transport of every bank from active into the spare. Returns each layer's k and v rows
        0..prefix-1 from both chips, on the host."""
        from gdn_multitoken_conv import release_owned

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

    def audit_round(self, drafter, segment, features, prefix, position, reference):
        """After the fused launch: fence, then the deltas' rows 0..prefix-1 against today's and, in
        place, every active bank against its spare; a mismatch is logged and repaired."""
        operations = self.operations
        cache = drafter.kv_history
        storage = self.segments[segment]
        operations.synchronize_device(self.mesh)
        mismatched, checked = [], 0
        for layer in range(DRAFT_LAYERS):
            for name in HEADS:
                shards = operations.get_device_tensors(storage.deltas[layer][name])
                for chip, (shard, expected) in enumerate(zip(shards, reference[layer][name])):
                    checked += 1
                    if not same_bits(operations.to_torch(shard)[..., :prefix, :], expected):
                        mismatched.append('delta%d%s.%d' % (layer, name, chip))
        bad_banks = []
        if self.inplace:
            for layer in range(DRAFT_LAYERS):
                for name in HEADS:
                    pairs = zip(operations.get_device_tensors(cache.active[layer][name]),
                                operations.get_device_tensors(cache.spare[layer][name]))
                    for chip, (active, spare) in enumerate(pairs):
                        checked += 1
                        if not same_bits(operations.to_torch(active), operations.to_torch(spare)):
                            mismatched.append('bank%d%s.%d' % (layer, name, chip))
                            bad_banks.append((layer, name))
        self.counts['audited'] += 1
        self.counts['mismatches'] += len(mismatched)
        log_line('%s round=%d segment=%d prefix=%d mode=%s checked=%d mismatches=%d' % (
            AUDIT_MARKER, self.block.rounds, segment, prefix, 'inplace' if self.inplace else 'oop', checked,
            len(mismatched)))
        if not mismatched:
            return
        log_line('%s round=%d segment=%d prefix=%d at=%s' % (AUDIT_MISMATCH_MARKER, self.block.rounds, segment,
                                                             prefix, ','.join(mismatched[:8])))
        # The round must stay exact whatever the audit found: today's bytes back.
        if self.inplace:
            for layer, name in dict.fromkeys(bad_banks):
                operations.copy(cache.spare[layer][name], cache.active[layer][name])
            operations.synchronize_device(self.mesh)
        else:
            self.eager_reference(drafter, features, prefix, position, slide=True)

    # -- teardown ----------------------------------------------------------------------------------------
    def describe(self):
        return dict(inplace=self.inplace, audit=self.audit, live_banks=live_banks_enabled(), rows=self.rows,
                    kernel=self.kernel_kind, traces=self.trace_count(), counts=dict(self.counts),
                    refusals=dict(self.refusals))

    def release_buffers(self):
        from gdn_multitoken_conv import release_owned

        for storage in self.segments:
            if storage.buffers:
                release_owned(self.operations, storage.buffers)
            storage.buffers, storage.tables, storage.deltas = [], (), []

    def close(self, *, wait=True):
        """Release the traces (abandoned without `wait`, as the block abandons its own) and every
        buffer. The pool's banks and query are lent: never freed here."""
        from gdn_multitoken_conv import release_owned

        if self.closed:
            return
        self.closed = True
        for storage in self.segments:
            traces = ([storage.projection_trace] if storage.projection_trace is not None else []) + list(
                storage.slides.values())
            if wait:
                for trace in traces:
                    self.operations.release_trace(self.mesh, trace)
            storage.projection_trace, storage.slides, storage.slide_programs = None, {}, {}
            if storage.projection_owned:
                release_owned(self.operations, storage.projection_owned)
            storage.projection_owned = []
            storage.inflight = ()
        self.release_buffers()


def build(block, *, operations, mesh, pool, shared_weights, collectives, diagnostic):
    """The block's FusedCommit under QWEN_FAST_FUSED_COMMIT=1, or None: without the flag, or when a
    host check refuses it (REFUSED_MARKER says why; the block then serves today's publication)."""
    if not enabled():
        return None
    try:
        return FusedCommit(block, operations=operations, mesh=mesh, pool=pool, shared_weights=shared_weights,
                           collectives=collectives, inplace=inplace_enabled(), audit=audit_enabled())
    except Refused as refusal:
        diagnostic('%s users=%d reason=%s' % (REFUSED_MARKER, block.users, str(refusal).replace(' ', '_')[:160]))
        return None


def install_fused_commit(drafter, fused, segment, *, merge_release, fused_steady_state):
    """Around one packed commit (serving_packed_step.commit_entry): today's installers exactly as
    commit_entry makes them (dflash_traced_publish.install_publish_options when either flag is on),
    then drafter.prepare_publication wrapped - a publication the guards admit is the fused one,
    any other goes to today's, argument for argument - and, in place, kv_history.commit and discard.
    Returns a restore callable that undoes all of it, in reverse."""
    from dflash_traced_publish import install_publish_options

    if type(merge_release) is not bool or type(fused_steady_state) is not bool:
        raise ValueError('Explicit merge_release and fused_steady_state selection required')
    if 'prepare_publication' in vars(drafter):
        raise ValueError('drafter.prepare_publication is already overridden')
    restore_today = None
    if merge_release or fused_steady_state:
        restore_today = install_publish_options(drafter, merge_release=merge_release,
                                                fused_steady_state=fused_steady_state)
    today = drafter.prepare_publication
    cache = getattr(drafter, 'kv_history', None)
    installed = []
    try:
        def prepare_publication(features, prefix, *, position):
            reason = fused.refusal(drafter, segment, features, prefix, position)
            if reason is not None:
                fused.note(segment, prefix, 'today', reason)
                return today(features, prefix, position=position)
            return fused.prepare(drafter, segment, features, prefix, position=position)

        drafter.prepare_publication = prepare_publication
        if cache is not None and fused.inplace:
            if 'commit' in vars(cache) or 'discard' in vars(cache):
                raise ValueError('kv_history.commit or discard is already overridden')
            cache.commit = lambda publication: fused.commit_kv(cache, publication)
            installed.append('commit')
            cache.discard = lambda publication: fused.discard_kv(cache, publication)
            installed.append('discard')
    except BaseException:
        for name in installed:
            delattr(cache, name)
        if restore_today is not None:
            drafter.prepare_publication = today
            restore_today()
        elif 'prepare_publication' in vars(drafter):
            del drafter.prepare_publication
        raise

    def restore():
        for name in reversed(installed):
            delattr(cache, name)
        if restore_today is not None:
            drafter.prepare_publication = today
            restore_today()
        else:
            del drafter.prepare_publication

    return restore


def live_bank_history(device_a, device_b):
    """F4 (QWEN_FAST_FUSED_COMMIT_LIVE_BANKS): the packed pair's cached_history as the two pool
    slots' ACTIVE banks, user-major then layer - or None (flag off, or a device without a pooled
    five-layer cache), and the pair uploads its own placeholders as today."""
    if not live_banks_enabled():
        return None
    history = []
    for device in (device_a, device_b):
        slot = getattr(device, 'pool_slot', None)
        kv = getattr(slot, 'kv', None)
        cache = getattr(device, 'kv_history', None)
        if kv is None or cache is None or len(kv) != len(cache.active) or len(kv) != DRAFT_LAYERS:
            return None
        history.append([{name: layer['active'][name] for name in HEADS} for layer in kv])
    return history


_LIVE_NOTED = []


def note_live_banks(pair, context):
    """LIVE_BANKS_MARKER once per process, at the first pair bound to the live banks."""
    if _LIVE_NOTED:
        return False
    _LIVE_NOTED.append(True)
    log_line('%s engaged pair=%s context=%s' % (LIVE_BANKS_MARKER, pair, context))
    return True
