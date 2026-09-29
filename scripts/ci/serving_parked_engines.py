"""Stage E (QWEN_FAST_PARKED_ENGINES, default off): parked per-slot engines on the S2 fast path.

WHY. Every S2 request builds its own engine after its prefill: a DFlashDevice over the pool slot it
acquires, the device's single-user proposal trace, and a VerifierEngine with three verify traces and
eight commit traces. A warm build costs 1.65-2.23 s, on the path to the request's first token and as a
stall on every live decoder. None of those traces is shaped by the request: positions, rotary tables,
page tables and the SDPA cur_pos are restaged device data (verifier_inputs.stage_inputs, the fixture's
pooled page tables), and the proposal trace takes one 2048 bucket whatever the prompt
(serving_request_factory.single_proposal_bucket). So one engine per pool slot, built once at attach on
a synthetic request, can serve every request that slot ever takes.

WHAT. At attach, after the packed block, ParkedEngineSet builds one engine per pool slot on a synthetic
request (P_cap = 1, an all-page-0 table, zero taps, native GDN slot 0 zeroed first) and PARKS it: the
engine keeps its captures and its pooled storage, the device keeps its lent slot and its proposal
trace. A request REBINDS a parked slot instead of building (rebind_device, VerifierEngine.rebind),
which allocates nothing that outlives the call, and parks it again when it finishes (park_device,
VerifierEngine.park). The pair and quad drafter traces are NOT kept across requests (Stage E2 is
deferred: QWEN_FAST_PARKED_DRAFTS is refused).

EXACTNESS. Output stays byte-identical to a fresh build. The rules the census holds the code to:
  R1  a buffer a trace reads before that replay writes it is rewritten before every replay that reads
      it, or allocated before every trace that could replay while it lives (pooled);
  R2  a buffer trace A writes and a later trace B or the host reads does not sit in the holes of a
      trace C that replays in between: the sequential step verifies, reads back and commits one
      request at a time (ReplayLedger checks it under QWEN_FAST_PARKED_AUDIT=1);
  R3  after a rebind the parked engine's host state equals a fresh engine's, apart from whitelisted
      counters, and its device state differs from a fresh engine's only in buffers every replay
      writes before it reads them.
The drafter side is a speed and crash property, not an exactness one: greedy verification emits the
target's argmax whatever was drafted. Its gate is drafter equivalence (solo parked against solo
fresh), not token equality.

THE CENSUS (E0, test_parked_census). A fake ttnn logs every allocation, read, write, capture and
replay; each trace's read set is what its capture read before writing. A tensor allocated after a
trace was captured is EXPOSED by that trace's replay (it may sit in the trace's holes) until it is
next written in full; reading an exposed tensor is an R1/R2 violation. The census runs the real
VerifierEngine, DFlashDevice, DraftKVHistory, PreparedDFlashProposal and ServingBufferPool over that
fake, today's per-request churn and the parked cycles alike, and fails on any violation, on any
live tensor it cannot place with an owner, and on any persistent allocation after attach.

SOURCE PINS. The frozen recipe's pin probe (docs/batch-spec-tasks-2026-09-19.md, run 35495461481)
measured verifier_engine.py, dflash_device.py and serving_buffer_pool.py unpinned, and every module
the C2 overlay already carries is unpinned by construction (c2_overlay.pin_breaks refuses a pinned
destination at build). serving_page_binding.py was not in that probe, so it is not edited: the full
page-table rewrite lives here (write_page_tables) and test_parked_engine_rebind holds it to
VerifierPageBinding.refresh's operation sequence. draft_kv_history.py (DraftKVHistory.prepare is
text-matched by draft_kv_slide_scope) and extent_attention_replay.py (packed-any evidence pins it)
stay byte-identical (NEVER_EDITED).
"""

import os
import re


FLAG = 'QWEN_FAST_PARKED_ENGINES'
PROJECT_ROWS_FLAG = 'QWEN_FAST_PARKED_PROJECT_ROWS'
AUDIT_FLAG = 'QWEN_FAST_PARKED_AUDIT'
# Stage E2 (keeping the pair and quad drafter traces across requests) is deferred: its quad retention
# does not fit the parked admission at a 123k arrival (design section 5.3). Named only to be refused.
DEFERRED_DRAFTS_FLAG = 'QWEN_FAST_PARKED_DRAFTS'

# The windowed drafter projection at a rebind (rebind_device): rows per project_features call. 0 is
# one call over the whole window, the constructor's order.
DEFAULT_PROJECT_ROWS = 256
PROJECTION_CHUNK_ROWS = 32
HISTORY_ROWS = 2048

# The serving modules Stage E edits (each must reach the C2 image through its overlay, and none may be a
# frozen-recipe pin), and the ones it must leave byte-identical (the module docstring).
STAGE_E_EDITS = ('verifier_engine.py', 'serving_buffer_pool.py', 'serving_runtime.py', 'serving_parked_engines.py')
NEVER_EDITED = ('draft_kv_history.py', 'extent_attention_replay.py', 'serving_page_binding.py')

# Every "for the request's life" or "permanent" claim in the drafter and coordinator sources, with
# what a parked device (rebound at any position, kept for the process) does to it. test_parked_census
# fails on a claim in those files this table does not list.
LIFETIME_INVARIANTS = (
    dict(file='dflash_packed_proposal_coordinator.py', text='the permanent steady-state context',
         status='changed: a parked device leaves 2048 when it is rebound below it',
         guard='packable() reads history_rows every round, so a rebound device below 2048 does not pack'),
    dict(file='dflash_packed_proposal_coordinator.py', text='2048 is monotonic and permanent once reached',
         status='changed: history_rows can fall across a rebind',
         guard='rebind_device rebuilds a released single capture under single_proposal_bucket (one 2048 '
               'bucket at any position); the coordinator rebuild is scoped the same way under the flag (E4)'),
    dict(file='dflash_packed_proposal_coordinator.py', text='a permanent-',
         status='unchanged: a per-round fallback, re-evaluated every round', guard=None),
    dict(file='dflash_packed_proposal_coordinator.py', text='two in a row give up for good',
         status='unchanged: the quad gives up for the coordinator (hook) life, as today', guard=None),
    dict(file='dflash_device.py', text="are fixed for a request's life once its proposal is captured",
         status='changed: they are fixed for the process; a rebind keeps kv_history, progress None and the '
                'capture (rebuilt when released)', guard='rebind_device repoints proposal_capture.kv_history'),
    dict(file='dflash_device.py', text="every proposal for the request's life",
         status='changed: the pooled query is read for the process life', guard='pooled: allocated at attach'),
    dict(file='dflash_device.py', text='permanently true once this device is past the prefill ramp',
         status='changed: a rebind below 2048 re-enters the ramp', guard='rows is recomputed per publication'),
    dict(file='dflash_device.py', text='self.history_stale is set and never cleared',
         status='changed: rebind_device clears it with the history it rewrites', guard='rebind_device'),
    dict(file='dflash_device.py', text='the history is marked stale for good',
         status='changed: cleared at the next rebind', guard='rebind_device'),
    dict(file='dflash_proposal_trace.py', text='for the life of this object',
         status='unchanged: pair traces are retired when a member parks (E4, release_parked)', guard=None),
    dict(file='dflash_proposal_trace.py', text='PERMANENT, monotonic state every request',
         status='changed: per request, not per device', guard='the pair gate reads history_rows every round'),
)
LIFETIME_PHRASES = ("request's life", 'permanent', 'monotonic', 'for the life of', 'for good', 'never cleared')
LIFETIME_FILES = ('dflash_packed_proposal_coordinator.py', 'dflash_device.py', 'dflash_proposal_trace.py')


def _strict_bool(name, environ=None):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def parked_engines_enabled(environ=None):
    """QWEN_FAST_PARKED_ENGINES: unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_PARKED_AUDIT (gate): unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(AUDIT_FLAG, environ)


def project_rows(environ=None):
    """QWEN_FAST_PARKED_PROJECT_ROWS: the rebind's projection window, a multiple of the projection's 32-row
    chunk up to 2048 (so every window starts on a chunk boundary of the whole call), or 0 for one call over
    the whole window. Default 256."""
    text = (os.environ if environ is None else environ).get(PROJECT_ROWS_FLAG, str(DEFAULT_PROJECT_ROWS))
    if type(text) is not str or re.fullmatch('0|[1-9][0-9]*', text) is None:
        raise ValueError('%s must be a decimal row count, got %r' % (PROJECT_ROWS_FLAG, text))
    rows = int(text)
    if rows and (rows % PROJECTION_CHUNK_ROWS or rows > HISTORY_ROWS):
        raise ValueError('%s must be 0 or a multiple of %d up to %d, got %d'
                         % (PROJECT_ROWS_FLAG, PROJECTION_CHUNK_ROWS, HISTORY_ROWS, rows))
    return rows


def page_table_bindings(engine):
    """{two-chip identity: (tensor, shape)} for every page table a sequential engine's fixtures read - each
    bucket's pages and singleton page table - collected and checked as VerifierPageBinding collects its
    bindings. Refused: any other page ownership (a replay reader's per-bundle tables, a grouped reader, a
    reader or writer over tables of its own), which a parked engine never has."""
    from gdn_multitoken_conv import addresses

    operations, capacity = engine.operations, engine.pages.shape[1]
    tensors = []
    for bucket in engine.buckets.values():
        fixture = bucket.get('fixture')
        if fixture is None:
            continue
        if getattr(fixture, 'replay_reader', None) is not None or fixture.grouped_readers:
            raise ValueError('A parked engine rewrites the page tables of sequential captures only')
        if (any(value is not fixture.singleton_pages for reader in fixture.readers for value in reader.pages)
                or any(value is not fixture.singleton_pages for writer in fixture.writers
                       for value in getattr(writer, 'pages', ()))):
            raise ValueError('Unexpected singleton attention page binding')
        tensors.extend((fixture.pages, fixture.singleton_pages))
    bindings = {}
    for tensor in tensors:
        shape = tuple(tensor.shape)
        identity = tuple(addresses(operations, tensor))
        if len(shape) != 2 or shape[0] < 1 or not 1 <= shape[1] <= capacity or len(identity) != 2:
            raise ValueError('Bounded two-chip page metadata required')
        if identity in bindings and bindings[identity][1] != shape:
            raise ValueError('Aliased page metadata has conflicting geometry')
        bindings[identity] = (tensor, shape)
    if not bindings:
        raise ValueError('Captured verifier page metadata required')
    return bindings


def write_page_tables(operations, mesh, bindings, host):
    """VerifierPageBinding.refresh's device write (serving_page_binding.py), for a whole table: every bound
    table's addresses and shape checked, each rewritten with the (1, width) host table's first columns
    repeated per row, through copy_host_to_device_tensor, one fence, and the addresses checked again. The
    same operations, in the same order, as refresh makes for the same host table
    (test_parked_engine_rebind); kept here because serving_page_binding.py is not edited
    (serving_parked_engines' docstring, SOURCE PINS)."""
    from gdn_multitoken_conv import addresses

    for identity, (tensor, shape) in bindings.items():
        if tuple(addresses(operations, tensor)) != identity or tuple(tensor.shape) != shape:
            raise ValueError('Captured page metadata addresses changed')
    for tensor, shape in bindings.values():
        values = host[:, :shape[1]].repeat(shape[0], 1).contiguous()
        source = operations.from_torch(values, dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT,
                                       mesh_mapper=operations.ReplicateTensorToMesh(mesh))
        operations.copy_host_to_device_tensor(source, tensor)
    operations.synchronize_device(mesh)
    for identity, (tensor, _) in bindings.items():
        if tuple(addresses(operations, tensor)) != identity:
            raise ValueError('Page upload replaced a captured device buffer')


def refuse_deferred(environ=None):
    """Stage E2 is deferred: QWEN_FAST_PARKED_DRAFTS is refused, whatever its value."""
    environ = os.environ if environ is None else environ
    if DEFERRED_DRAFTS_FLAG in environ:
        raise ValueError('%s (Stage E2: keeping the pair and quad drafter traces across requests) is deferred '
                         'and not implemented; unset it' % DEFERRED_DRAFTS_FLAG)


class ReplayLedger:
    """R2's replay ledger (QWEN_FAST_PARKED_AUDIT=1): counts every trace replay the process makes by wrapping
    `operations.execute_trace`, and hands the count to verifier_engine, whose verify notes it right after its
    own replay and whose publish asserts nothing replayed since (verifier_engine.check_replay_mark). Every
    replay site calls execute_trace through the ttnn module at call time, so one wrapper sees them all."""

    def __init__(self, operations):
        self.operations = operations
        self.count = 0
        self.original = self.previous = None
        self.installed = False

    def read(self):
        return self.count

    def install(self):
        import verifier_engine

        if self.installed:
            raise ValueError('The replay ledger is already installed')
        original = self.operations.execute_trace

        def execute_trace(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        self.original = original
        self.operations.execute_trace = execute_trace
        self.previous = verifier_engine.set_replay_count(self.read)
        self.installed = True
        return self

    def uninstall(self):
        import verifier_engine

        if not self.installed:
            return
        self.operations.execute_trace = self.original
        verifier_engine.set_replay_count(self.previous)
        self.installed = False


# -- the device side (design section 2.3 step 4, section 2.4) ----------------------------------------------

def single_capture(device):
    """The device's own single-user proposal capture, through a pair's _PackedCaptureView when one is
    installed (dflash_packed_proposal_coordinator), or None when a pair or the quad released it."""
    from dflash_packed_proposal_coordinator import _PackedCaptureView

    capture = device.proposal_capture
    return capture._original if isinstance(capture, _PackedCaptureView) else capture


def build_single_capture(device):
    """The single-user proposal capture a device's build makes under S2 - one 2048 bucket, whatever the
    position (serving_request_factory.single_proposal_bucket) - for a device whose capture was released.
    Unscoped, PreparedDFlashProposal(device, max_new_tokens=1) builds a 256-1024 bucket below 2048, which a
    history that grows past it refuses ('Committed history exceeds prepared request contexts')."""
    from dflash_proposal_trace import PreparedDFlashProposal
    from serving_request_factory import single_proposal_bucket

    with single_proposal_bucket():
        return PreparedDFlashProposal(device, max_new_tokens=1)


def project_window(device, taps, count, *, window):
    """The constructor's history projection (DFlashDevice.__init__: project_features(features, history_rows)),
    in calls of `window` rows each, or in one call when `window` is 0 or covers `count`. Each call has its own
    temporaries scope, fence and release (project_features' retain=None path), so a window keeps about a
    window's intermediates alive where one call keeps all of them until it returns (design section 2.3 4.3).
    The same chunks: project_features projects independent 32-row chunks with fixed program configurations,
    and every window starts on a chunk boundary of the whole call (project_rows keeps windows a multiple of
    32), so only the last chunk of the last window can be partial, as in the whole call; the windows' outputs
    are joined by one concat, a copy."""
    from gdn_multitoken_conv import release_owned

    if type(window) is not int or window < 0 or window % PROJECTION_CHUNK_ROWS or window > HISTORY_ROWS:
        raise ValueError('A projection window of 0 or a multiple of %d up to %d rows is required, got %r'
                         % (PROJECTION_CHUNK_ROWS, HISTORY_ROWS, window))
    if window == 0 or window >= count:
        return device.project_features(taps, count)
    operations = device.operations
    base = getattr(taps, 'row_offset', 0)
    parts = []
    try:
        for start in range(0, count, window):
            parts.append(device.project_features(taps, min(window, count - start), row_offset=base + start))
        joined = operations.concat(parts, dim=2)
    except BaseException:
        release_owned(operations, parts)
        raise
    release_owned(operations, parts)
    return joined


def park_device(device):
    """Stage E, the device's half of a park (design section 2.4): a fence first - discard_pending requires
    one (dflash_proposal_trace.PreparedDFlashProposal.discard_pending) and today's close takes it from
    engine.close - then the capture's pending proposal dropped. None when the device can park; else why,
    and its owner closes it as today. Its pair and quad traces are the coordinator's to retire."""
    device.operations.synchronize_device(device.mesh)
    capture = single_capture(device)
    if capture is not None:
        capture.discard_pending()
    if device.closed:
        return 'device closed'
    if device.pending is not None:
        return 'a pending publication'
    if device.kv_history is None or device.pool_slot is None:
        return 'no pooled K/V cache'
    if device.kv_history.pending is not None:
        return 'a pending K/V publication'
    try:
        device.pool_slot.verify()
    except AssertionError as failure:
        return 'pool slot moved: %s' % str(failure)[:120]
    return None


def rebind_device(device, taps, *, position, window=None):
    """Stage E: bind a parked DFlashDevice to a request prefilled to `position`, from its window's taps (the
    prefill capture's outputs), as its constructor seeds a fresh one, without capturing anything (design
    section 2.3 step 4):
      1. a fence, then the capture's pending proposal dropped (discard_pending requires the fence);
      2. the pool slot re-zeroed as a loan zeroes it (ServingBufferPool.rezero), its buckets kept taken;
      3. the history projected (project_window: `window` rows per call, default project_rows()), padded
         to 2048 rows and copied into the slot's history - the canonical orientation, history then spare;
      4. a new DraftKVHistory seeded over the slot's banks (the old one, pooled, owns nothing);
      5. the host state a fresh device has: position, history_rows, name, nothing pending, no calls or
         published rows, no audit digest or convolution checks, history_stale cleared, and
         rebind_generation advanced (the coordinator keys a pair or quad trace on it);
      6. the capture unwrapped from a pair's view and pointed at the new cache, or - released by a pair or
         the quad - rebuilt under the single bucket (build_single_capture).
    Returns dict(ms=, single_rebuilt=, window=)."""
    import time
    from dflash_prefill_window import prefill_window
    from draft_kv_history import DraftKVHistory
    from gdn_multitoken_conv import addresses

    operations = device.operations
    window = project_rows() if window is None else window
    slot = device.pool_slot
    if device.closed or slot is None or device.kv_history is None or not device.cache_history:
        raise ValueError('Only an open pooled device with a committed K/V cache can be rebound')
    if device.kv_history.projection is not None:
        raise ValueError('A parked device keeps no K/V projection capture')
    rows = prefill_window(position)['rows']
    started = time.perf_counter()
    operations.synchronize_device(device.mesh)
    capture = single_capture(device)
    if capture is not None:
        capture.discard_pending()
    elif not getattr(device, '_packed_capture_released', False):
        raise ValueError('A parked device keeps its single-user proposal capture')
    if device.pending is not None or device.kv_history.pending is not None:
        raise ValueError('A parked device has no publication pending')
    slot.pool.rezero(slot)
    projected = project_window(device, taps, rows, window=window)
    padded = operations.pad(projected, [(0, 0), (0, 0), (0, HISTORY_ROWS - rows), (0, 0)], 0.0)
    if addresses(operations, padded) != addresses(operations, projected):
        operations.deallocate(projected)
    operations.copy(padded, slot.history)
    operations.deallocate(padded)
    device.history, device.spare_history = slot.history, slot.spare_history
    operations.synchronize_device(device.mesh)
    device.kv_history = DraftKVHistory(operations, device.mesh, [layer[0] for layer in device.layers], device.history,
        position=position, history_rows=rows, capture_projection=False, storage=slot.kv,
        **(dict(query=slot.query) if getattr(slot, 'query', None) is not None else {}))
    if device.progress is not None:
        device.kv_history.audit(device.history)
    device.position, device.history_rows = position, rows
    device.name = 'DFlashDevice@%x position=%d' % (id(device), position)
    device.pending = None
    device.proposal_calls = device.published_rows = 0
    device.audit_digest = None
    device.convolution_checks = []
    device.history_stale = False
    device.rebind_generation = getattr(device, 'rebind_generation', 0) + 1
    rebuilt = capture is None
    if rebuilt:
        device.proposal_capture = None
        capture = build_single_capture(device)
        device._packed_capture_released = False
    device.proposal_capture = capture
    capture.kv_history = device.kv_history
    return dict(ms=(time.perf_counter() - started) * 1000, single_rebuilt=rebuilt, window=window)
