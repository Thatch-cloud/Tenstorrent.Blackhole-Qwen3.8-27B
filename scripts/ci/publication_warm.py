"""S2 B6: the drafter's eager publication, published once at every shape serving can ask of it, at the
packed block's attach - so no serving path compiles a publication program after attach.

WHY. The fused commit (fused_commit.py) captures every segment's projection and every (segment, prefix)
slide at attach, so a packed commit it admits compiles nothing. Every commit it refuses takes today's
eager publication instead (fused_commit.install_fused_commit -> dflash_traced_publish.
install_publish_options -> DFlashDevice.prepare_publication): the ramp (history_rows != 2048), the one
parity normalisation after a user joins holding the spare as active, any other refusal - and every
sequential step publishes eagerly too. Those programs are shaped by the accepted prefix (and, for the
feature slice, by the segment's row offset and the taps' rows): DFlashDevice.project_features slices and
pads `prefix` rows of each tap, DraftKVHistory.project_inputs slices and pads `prefix` projected rows. The
M1 warm only ever reached prefixes 1-4 there (the sequential step's), so the first refused commit at a
larger prefix compiled mid-request: the S2 gate's forced-cap and lifecycle arms each added kernel-cache
entries after the warm (a tilize_with_val_padding reader and two fill_pad programs at count 5, widths 2560
and 5120), about a second per new shape on a cold kernel cache.

WHAT. PackedVerifierEngine calls warm() at attach, right after the fused commit's capture (the block's
last capture), under the S2 extent block only (a pool with extent_replay: QWEN_FAST_EXTENT_REPLAY=1, the
c2-packed profiles). It publishes, and discards, every shape a serving publication takes (plan()):
  - packed: every segment's row offset (packed_shapes.segment_rows: 0, 16, 32 and 48 in the M3 block) x
    every accepted prefix 1..rows_per_user (16), from taps of the block's own geometry (block_rows rows,
    sharded on the feature axis, as packed_verifier allocates its taps) wrapped as the block wraps them
    (packed_verifier.PackedFeatureTaps), through today's installers as serving_packed_step.commit_entry
    and fused_commit.install_fused_commit install them for a refused round:
    dflash_traced_publish.install_publish_options with QWEN_FAST_PIPELINED_PUBLISH and
    QWEN_FAST_TRACED_PUBLISH read as the round reads them (nothing installed when both are off);
  - sequential: every per-request capture width the pool holds (its bucket_rows: 1, 2 and 4 beside the M3
    block) x every prefix 1..width, from taps of that width (the pool's bucket taps' geometry), with no
    installer: DFlashRequestRuntime.publish's own call, the sequential commit (publish_prewarm.py says
    why that is the call).
Each one is drafter.prepare_publication(features, prefix, position=...) then drafter.discard_publication -
today's code, not a copy of it: DFlashDevice's feature projection and feature-history branch (B1's twin
under QWEN_FAST_ROUND_B1), then the draft cache's K/V publication as the installers leave it - the
qualified slide candidate (draft_kv_slide_scope) or its steady fusion (dflash_traced_publish).

SCRATCH, never live state. The drafter and its draft cache are built here, bypassing their constructors
(the torch fakes of test_publish_prewarm do the same), over buffers allocated here: the taps, a
(1, 1, 2048, 5120) history and spare (the pool's HISTORY_SHAPE), one active and one spare K/V bank
(1, 4, 2048, 128) serving every (layer, head), and a zero (1, 1, 32, 2048) query. The pool's slots, the
block's taps and every request's drafter, history, banks and session are never passed, read or written.
Only the shared draft weights (read: the projection, its norm, the five layers' K/V parameters) and the
shared collectives (the projection's gather, as every eager publication and the fused commit's own capture
warm run it) are the served objects. The drafter is at the steady state (history_rows 2048), so the
sequential step's feature-history write - which it runs there, fused_steady_state being False on that path
- is warmed too. Nothing is pending afterwards (checked after every shape). Each publication fences and
releases its own temporaries (today's code does); the scratch is released after a final fence before
warm() returns - on a failure without the fence (host-side bookkeeping only, as the block's own
close(wait=False)), and the failure propagates: the attach fails closed like every other stage.

THE RAMP (history_rows < 2048). Nothing ramp-specific is warmed, because nothing ramp-specific is finite
and needed:
  - the feature-history write is shaped by (history_rows, prefix) - new shapes every round - and C7
    (dflash_device.history_unread) skips it for every served engine (a K/V cache, no audit reporter, a
    captured proposal; test_dflash_ramp_history pins that serving builds them so), in both the packed
    and the sequential publication;
  - the K/V publication in the ramp is the slide candidate, op for op the steady one: project_inputs pads
    the accepted rows to 32 whatever history_rows is, the K/V projection runs at 32 rows, and the slide
    takes history_rows, drop and rows as runtime arguments of one kernel whose compile-time arguments are
    its tensors' accessors - no new kernel binary (whether ttnn's generic op keys its program cache on
    runtime arguments is UNVERIFIED; at most that is a cache entry over the same binary, no compile);
  - the feature projection is shaped by (taps rows, row offset, prefix) only - warmed here.
Under an audit reporter or an eager proposal (QWEN_FAST_PROPOSAL_AUDIT, QWEN_FAST_EAGER_PROPOSAL; never in
the C2 image) C7 does not hold and the ramp's history write compiles per round: not warmable, not served.

SIDE EFFECTS. Program-cache entries (the point). The first B1 publication of the process is this warm, so
dflash_packed_proposal's once-per-process '[PINDIAG] round b1 engaged' line names site=publication at
attach (the gate checks only that it is there). With the qualified slide scope live, its audit counters
count these publications (nothing in serving reads them; publish_prewarm's warm counts the same way). The
shared collectives' semaphore handles cycle once per warmed projection, as for any eager publication.

COST (UNVERIFIED on hardware). 71 publications beside the M3 block (64 packed, 7 sequential): about 10 ms
each once their programs exist (v27 measured 9-11 ms for one user's eager packed publication), plus
creating each new program once - about 1-3 s per attach on a warm kernel cache; on a cold one (the first
start of a new image, which the M1 warm is) the prefix-shaped kernels are JIT-compiled here instead of in
the middle of requests. DRAM per chip: about 48 MB of scratch (the history pair 2 x 20 MiB, the bank pair
2 x 2 MiB, taps, query) plus one publication's temporaries - about 1 MB packed, up to about 80 MB for the
sequential history write (the 2048-row slice, the (2048 + prefix)-row concat and its trim) - so a peak near
130 MB, all released before the block's attach completes.

MARKER: one '[PINDIAG] eager publication warmed: <n> shapes in <ms> ms packed=<offsets>:<prefixes>
sequential=<width>:<prefixes>,... merge_release=<0|1> fused_steady_state=<0|1> program_cache=<a>-><b>'
line, or '[PINDIAG] eager publication warm skipped reason=<why>' when the block was given no collectives
or no prepared draft weights to publish with. Neither is ever logged without the extent block; the S2 gate
(lever_n_m3native_gate.s2_report) requires the warmed line under QWEN_FAST_EXTENT_REPLAY=1 and counts
either as an S2 marker leaked into a flag-off arm.
"""

from collections import namedtuple
import sys
import time

MARKER = '[PINDIAG] eager publication warmed:'
SKIPPED_MARKER = '[PINDIAG] eager publication warm skipped'
PACKED, SEQUENTIAL = 'packed', 'sequential'
HISTORY_ROWS = 2048
FEATURE_WIDTH = 5120
FEATURE_TAPS = 5
DRAFT_LAYERS = 5
HISTORY_SHAPE = (1, 1, HISTORY_ROWS, FEATURE_WIDTH)
KV_SHAPE = (1, 4, HISTORY_ROWS, 128)
QUERY_SHAPE = (1, 1, 32, 2048)
# The scratch frontier: the steady state (DraftKVHistory keeps history_rows == min(position, 2048)).
# The value only fills the RoPE tables' rows; no program is shaped by it.
POSITION = HISTORY_ROWS

Shape = namedtuple('Shape', 'path rows offset prefix')


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


def wanted(block):
    """Only the S2 extent block (its pool lent it extent storage, packed_verifier: `extent`)."""
    return getattr(block, 'extent', False) is True


def refusal(shared_weights, collectives):
    """Why the attach cannot publish as a request's drafter does, or None."""
    if collectives is None:
        return 'no-collectives'
    layers = tuple(getattr(shared_weights, 'layers', None) or ())
    if (len(layers) != DRAFT_LAYERS or getattr(shared_weights, 'projection', None) is None
            or getattr(shared_weights, 'feature_norm', None) is None):
        return 'no-prepared-draft-weights'
    return None


def plan(block, pool):
    """Every shape a serving publication takes beside this block, in the order warm() publishes them: each
    packed segment's row offset x prefixes 1..rows_per_user, then each pooled capture width x 1..width."""
    from packed_shapes import segment_rows

    shapes = [Shape(PACKED, block.block_rows, segment_rows(block.shape, segment)[0], prefix)
              for segment in range(block.users) for prefix in range(1, block.rows_per_user + 1)]
    widths = sorted(set(getattr(pool, 'bucket_rows', None) or ()))
    shapes += [Shape(SEQUENTIAL, rows, 0, prefix) for rows in widths for prefix in range(1, rows + 1)]
    return shapes


def runs(prefixes):
    """'1-16' for 1..16, '1,3' for 1 and 3."""
    found = []
    for prefix in prefixes:
        if found and found[-1][1] == prefix - 1:
            found[-1][1] = prefix
        else:
            found.append([prefix, prefix])
    return ','.join('%d' % first if first == last else '%d-%d' % (first, last) for first, last in found)


def describe_plan(shapes):
    """The line's packed= and sequential= fields: 'offsets:prefixes' and 'width:prefixes,...'."""
    packed = [shape for shape in shapes if shape.path == PACKED]
    offsets = sorted({shape.offset for shape in packed})
    prefixes = sorted({shape.prefix for shape in packed})
    widths = sorted({shape.rows for shape in shapes if shape.path == SEQUENTIAL})
    sequential = ','.join('%d:%s' % (rows, runs([shape.prefix for shape in shapes
                                                 if shape.path == SEQUENTIAL and shape.rows == rows]))
                          for rows in widths)
    return ('%s:%s' % (','.join(map(str, offsets)), runs(prefixes)) if packed else 'none'), (sequential or 'none')


def warmed_line(summary):
    return '%s %d shapes in %.1f ms packed=%s sequential=%s merge_release=%d fused_steady_state=%d program_cache=%s->%s' % (
        MARKER, summary['shapes'], summary['ms'], summary['packed'], summary['sequential'],
        int(summary['merge_release']), int(summary['fused_steady_state']), summary['program_cache'][0],
        summary['program_cache'][1])


def program_cache(mesh):
    """mesh.num_program_cache_entries(), as publish_prewarm reads it, or 'n/a'."""
    count = getattr(mesh, 'num_program_cache_entries', None)
    if not callable(count):
        return 'n/a'
    try:
        return int(count())
    except Exception:
        return 'n/a'


class Scratch:
    """Every buffer warm() allocates, released together."""

    def __init__(self, operations, mesh):
        self.operations, self.mesh, self.values = operations, mesh, []

    def zeros(self, shape, *, sharded=False):
        """Tiled BF16 zeros in DRAM: replicated, or sharded on the feature axis as the taps are."""
        import torch

        operations = self.operations
        value = operations.from_torch(torch.zeros(shape, dtype=torch.bfloat16), device=self.mesh,
            dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(self.mesh, dim=3) if sharded
            else operations.ReplicateTensorToMesh(self.mesh))
        self.values.append(value)
        return value

    def release(self):
        values, self.values = self.values, []
        seen = set()
        for value in values:
            if id(value) not in seen:
                seen.add(id(value))
                self.operations.deallocate(value)


def scratch_cache(operations, mesh, parameters, active, spare, query):
    """A DraftKVHistory at the steady state over scratch banks (one active and one spare bank serve every
    layer and head: a publication writes only the spare, and each slide only needs the three it is given
    to be distinct), without its constructor's projection of a real history."""
    from draft_kv_history import DraftKVHistory

    cache = object.__new__(DraftKVHistory)
    cache.__dict__.update(operations=operations, mesh=mesh, parameters=tuple(parameters), position=POSITION,
                          history_rows=HISTORY_ROWS, owned=[], checks=[], projection=None, pending=None,
                          closed=False, query=query, borrowed=[active, spare, query],
                          active=[dict(k=active, v=active) for layer in parameters],
                          spare=[dict(k=spare, v=spare) for layer in parameters])
    return cache


def scratch_drafter(operations, mesh, shared_weights, collectives, cache, history, spare_history):
    """A DFlashDevice at the steady state as serving builds it (a K/V cache, no audit reporter, a captured
    proposal: C7 holds), borrowing the shared draft weights, over scratch histories - without its
    constructor's pool slot, prefill projection and proposal capture."""
    from dflash_device import DFlashDevice

    drafter = object.__new__(DFlashDevice)
    drafter.__dict__.update(
        operations=operations, mesh=mesh, collectives=collectives, name='eager publication warm',
        kernel=operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False),
        layers=tuple(shared_weights.layers), projection=shared_weights.projection,
        feature_norm=shared_weights.feature_norm, borrowed=list(getattr(shared_weights, 'tensors', None) or ()),
        owned=[], closed=False, pending=None, position=POSITION, history_rows=HISTORY_ROWS, history=history,
        spare_history=spare_history, kv_history=cache, progress=None, proposal_capture='scratch (never replayed)',
        published_rows=0, pool_slot=None, shared_weights=None)
    return drafter


def check_untouched(drafter):
    cache = drafter.kv_history
    if (drafter.pending is not None or cache.pending is not None or drafter.position != POSITION
            or cache.position != POSITION or drafter.history_rows != HISTORY_ROWS
            or cache.history_rows != HISTORY_ROWS or 'prepare_publication' in vars(drafter)
            or 'prepare' in vars(cache)):
        raise AssertionError('An eager publication warm must leave nothing pending or installed and the frontier '
                             'where it was: pending=%r/%r position=%r/%r history_rows=%r/%r' % (
                                 drafter.pending, cache.pending, drafter.position, cache.position,
                                 drafter.history_rows, cache.history_rows))


def publish(drafter, features, prefix, install):
    """One publication, prepared and discarded: `install` is today's installer for the path, or None."""
    restore = install(drafter) if install is not None else None
    try:
        publication = drafter.prepare_publication(features, prefix, position=POSITION)
        drafter.discard_publication(publication)
    finally:
        if restore is not None:
            restore()
    check_untouched(drafter)


def warm(block, *, operations, mesh, pool, shared_weights, collectives, log=None):
    """Publish and discard every shape of plan(block, pool) on scratch (module docstring). Returns the
    summary it logged, or None: not the extent block (nothing done, nothing logged), or a skip (logged)."""
    log = log_line if log is None else log
    if not wanted(block):
        return None
    reason = refusal(shared_weights, collectives)
    if reason is not None:
        log('%s reason=%s' % (SKIPPED_MARKER, reason))
        return None
    from dflash_pipelined_publish import pipelined_publish_enabled
    from dflash_traced_publish import install_publish_options, traced_publish_enabled
    from packed_verifier import PackedFeatureTaps

    shapes = plan(block, pool)
    # As serving_packed_step.commit_entry reads them for each round.
    merge_release, fused_steady_state = pipelined_publish_enabled(), traced_publish_enabled()
    def installers(drafter):
        return install_publish_options(drafter, merge_release=merge_release, fused_steady_state=fused_steady_state)

    packed_install = installers if merge_release or fused_steady_state else None
    started = time.perf_counter()
    before = program_cache(mesh)
    scratch = Scratch(operations, mesh)
    try:
        query = scratch.zeros(QUERY_SHAPE)
        active, spare = scratch.zeros(KV_SHAPE), scratch.zeros(KV_SHAPE)
        history, spare_history = scratch.zeros(HISTORY_SHAPE), scratch.zeros(HISTORY_SHAPE)
        parameters = tuple(layer[0] for layer in shared_weights.layers)
        cache = scratch_cache(operations, mesh, parameters, active, spare, query)
        drafter = scratch_drafter(operations, mesh, shared_weights, collectives, cache, history, spare_history)
        taps = {}
        for shape in shapes:
            if shape.rows not in taps:
                taps[shape.rows] = tuple(scratch.zeros((1, 1, shape.rows, FEATURE_WIDTH), sharded=True)
                                         for tap in range(FEATURE_TAPS))
            if shape.path == PACKED:
                features = PackedFeatureTaps(taps[shape.rows], row_offset=shape.offset, rows=block.rows_per_user)
                publish(drafter, features, shape.prefix, packed_install)
            else:
                publish(drafter, taps[shape.rows], shape.prefix, None)
        operations.synchronize_device(mesh)
    except BaseException:
        # No fence: the device may be hung on what failed (packed_verifier's close(wait=False)); the frees are
        # host-side bookkeeping, and a failure in them must not hide the one that failed the warm.
        try:
            scratch.release()
        except BaseException:
            pass
        raise
    scratch.release()
    packed, sequential = describe_plan(shapes)
    summary = dict(shapes=len(shapes), ms=(time.perf_counter() - started) * 1000, packed=packed,
                   sequential=sequential, merge_release=merge_release, fused_steady_state=fused_steady_state,
                   program_cache=(before, program_cache(mesh)))
    log(warmed_line(summary))
    return summary
