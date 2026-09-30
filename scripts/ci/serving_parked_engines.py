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
from types import SimpleNamespace


FLAG = 'QWEN_FAST_PARKED_ENGINES'
PROJECT_ROWS_FLAG = 'QWEN_FAST_PARKED_PROJECT_ROWS'
AUDIT_FLAG = 'QWEN_FAST_PARKED_AUDIT'
# Stage E2 (keeping the pair and quad drafter traces across requests) is deferred: its quad retention
# does not fit the parked admission at a 123k arrival (design section 5.3). Named only to be refused.
DEFERRED_DRAFTS_FLAG = 'QWEN_FAST_PARKED_DRAFTS'
# The gate's negative controls (GATE ONLY: serving_c2_contract.parked_problems refuses the knob outside a -gate
# profile). 'carry' skips save_carry at the rebind, so the engine restores the slot's zeroed carry on its first
# verify and the tokens MUST diverge. 'drafter' skips the rezero and the reseed of the K/V banks at the rebind, so
# the tokens stay equal (greedy verification) and the drafter-equivalence judge MUST fail. Each proves its gate can
# see the breakage it exists for.
NEGATIVE_FLAG = 'QWEN_FAST_PARKED_NEGATIVE'
NEGATIVES = ('carry', 'drafter')
NEGATIVE_MARKER = '[PINDIAG] parked negative control '
# G-E2's injected park-time fault (GATE ONLY): 'park' refuses the first park after attach, once, as an engine that
# cannot park is refused (a poisoned retained block, a failed page binding), so the slot unparks, serves today's
# per-request builds and re-parks at an idle moment (design section 6.3).
FAULT_FLAG = 'QWEN_FAST_PARKED_FAULT'
FAULTS = ('park',)
FAULT_REASON = 'injected park fault (gate only)'

# The windowed drafter projection at a rebind (rebind_device): rows per project_features call. 0 is
# one call over the whole window, the constructor's order.
DEFAULT_PROJECT_ROWS = 256
PROJECTION_CHUNK_ROWS = 32
HISTORY_ROWS = 2048

# The serving modules Stage E edits (each must reach the C2 image through its overlay, and none may be a
# frozen-recipe pin), and the ones it must leave byte-identical (the module docstring).
STAGE_E_EDITS = ('verifier_engine.py', 'serving_buffer_pool.py', 'serving_runtime.py', 'serving_parked_engines.py',
                 # E4's wiring and E5's admission
                 'serving_request_factory.py', 'serving_fast_request.py', 'serving_worker_hook.py',
                 'dflash_packed_proposal_coordinator.py', 'serving_prefill_admission.py', 'serving_lifecycle.py')
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
         guard='rebind_device and rebuild_single rebuild a released single capture under single_proposal_bucket '
               '(one 2048 bucket at any position); _ensure_single_user is scoped the same way under the flag'),
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
         status='unchanged: pair traces are retired when a member parks (after_park, the coordinator\'s '
                'release_parked), and a rebound member\'s rebind_generation retires any that were not', guard=None),
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


def negative_mode(environ=None):
    """QWEN_FAST_PARKED_NEGATIVE (gate only): None when unset or empty, else 'carry' or 'drafter'; anything else
    is refused."""
    value = (os.environ if environ is None else environ).get(NEGATIVE_FLAG)
    if value in (None, ''):
        return None
    if value not in NEGATIVES:
        raise ValueError('%s must be one of %s, got %r' % (NEGATIVE_FLAG, ', '.join(NEGATIVES), value))
    return value


def fault_mode(environ=None):
    """QWEN_FAST_PARKED_FAULT (gate only): None when unset or empty, else 'park'; anything else is refused."""
    value = (os.environ if environ is None else environ).get(FAULT_FLAG)
    if value in (None, ''):
        return None
    if value not in FAULTS:
        raise ValueError('%s must be one of %s, got %r' % (FAULT_FLAG, ', '.join(FAULTS), value))
    return value


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


def rebind_device(device, taps, *, position, window=None, negative=None):
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
    `negative` == 'drafter' (the gate's negative control, NEGATIVE_FLAG) skips steps 2 and 4: the banks keep the
    previous request's K/V and the old DraftKVHistory stays.
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
    if negative != 'drafter':
        slot.pool.rezero(slot)
    projected = project_window(device, taps, rows, window=window)
    padded = operations.pad(projected, [(0, 0), (0, 0), (0, HISTORY_ROWS - rows), (0, 0)], 0.0)
    if addresses(operations, padded) != addresses(operations, projected):
        operations.deallocate(projected)
    operations.copy(padded, slot.history)
    operations.deallocate(padded)
    device.history, device.spare_history = slot.history, slot.spare_history
    operations.synchronize_device(device.mesh)
    if negative != 'drafter':
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


# -- the set (design sections 2.2, 2.4, 6.3, 6.4) ---------------------------------------------------------

# The parked admission's terms (design section 5.2) are serving_prefill_admission's, its one set of defaults (THE
# PARKED TERMS): R, the rebind's peak - estimated at 100 MB with the windowed projection and 350 MB with the whole
# call - and S, a single-capture rebuild (227 MB measured), which applies when that slot's single is released. With
# the prefill transient and the reserve they are what a parked arrival needs free. Read at call time, so this module
# imports nothing at load.
SYNTHETIC_POSITION = 1
SYNTHETIC_BUDGET = 16
SYNTHETIC_TOKEN = SYNTHETIC_SEED = 0
TARGET_TAPS = (5, 19, 33, 47, 61)
TAP_COUNT = 5
FEATURE_WIDTH = 5120
BUILT_MARKER = '[PINDIAG] parked engines built '
WARM_MARKER = '[PINDIAG] parked drafter warm '
REBIND_MARKER = '[PINDIAG] parked rebind '
UNPARKED_MARKER = '[PINDIAG] parked slot {} unparked: {}'
REPARKED_MARKER = '[PINDIAG] parked slot {} re-parked ms={:.1f}'
STOPPED_MARKER = '[PINDIAG] parked engines stopped at k={} of {}: short of {} (free={} largest_free={} need={})'
# A released single-user proposal capture rebuilt when its slot parks (after a detach, or at an idle moment), or
# kept released because the rebuild would leave the split short for the longest parked arrival.
SINGLE_REBUILT_MARKER = '[PINDIAG] parked slot {} single rebuilt at {} ms={:.1f}'
SINGLE_KEPT_MARKER = '[PINDIAG] parked slot {} single kept released at {}: short of {} (free={} largest_free={} need={})'
# QWEN_FAST_GATE_DRAM_BALLAST (GATE ONLY, G-E3's ballast arm): bytes per chip held unread from the end of the parked
# build to close, so the admissions run at the boundary of the parked need (design section 9).
BALLAST_FLAG = 'QWEN_FAST_GATE_DRAM_BALLAST'
BALLAST_MARKER = '[PINDIAG] gate dram ballast '
# The ballast in buffers of at most this many bytes (whole 32-row tiles of 1024 bf16 columns: 64 KiB each), so it
# takes the holes as well as the largest block, as an engine's 1,400 buffers do.
BALLAST_CHUNK_BYTES = 32 * 2 ** 20
BALLAST_ROW_BYTES = 1024 * 2
BALLAST_TILE_BYTES = 32 * BALLAST_ROW_BYTES


def rebind_peak_bytes(window):
    """R (serving_prefill_admission.PARKED_REBIND_BYTES; the whole call's when `window` is 0)."""
    import serving_prefill_admission as admission

    return admission.PARKED_REBIND_BYTES if window else admission.PARKED_REBIND_WHOLE_BYTES


def single_capture_bytes():
    """S (serving_prefill_admission.MEASURED_SINGLE_CAPTURE_BYTES)."""
    import serving_prefill_admission as admission

    return admission.MEASURED_SINGLE_CAPTURE_BYTES


def parked_arrival_need(reserve, window, *, single_released=False):
    """What a parked arrival at the longest prompt needs free per chip, the reserve in it (design section 5.2;
    serving_prefill_admission.parked_need): the prefill's transient, the rebind's peak, a single-capture rebuild
    when that slot's single is released."""
    import serving_prefill_admission as admission

    return admission.parked_need(admission.PREFILL_TRANSIENT_FROM, reserve, rebind=rebind_peak_bytes(window),
                                 single=single_capture_bytes() if single_released else 0)


def ballast_bytes(environ=None):
    """QWEN_FAST_GATE_DRAM_BALLAST: a decimal byte count per chip, 0 (the default) for none; anything else refused."""
    text = (os.environ if environ is None else environ).get(BALLAST_FLAG, '0')
    if type(text) is not str or re.fullmatch('0|[1-9][0-9]*', text) is None:
        raise ValueError('%s must be a decimal byte count, got %r' % (BALLAST_FLAG, text))
    return int(text)


class DramBallast:
    """G-E3's ballast (QWEN_FAST_GATE_DRAM_BALLAST, gate only): `size` bytes per chip, rounded up to whole 64 KiB tiles,
    replicated to both chips in buffers of at most BALLAST_CHUNK_BYTES, never read; close() frees them."""

    def __init__(self, operations, mesh, size):
        import torch

        if type(size) is not int or size < 1:
            raise ValueError('A positive ballast in bytes is required, got %r' % (size,))
        self.operations = operations
        self.tensors = []
        self.size = -(-size // BALLAST_TILE_BYTES) * BALLAST_TILE_BYTES
        left = self.size
        try:
            while left:
                chunk = min(left, BALLAST_CHUNK_BYTES)
                self.tensors.append(operations.from_torch(
                    torch.zeros((1, 1, chunk // BALLAST_ROW_BYTES, 1024), dtype=torch.bfloat16), device=mesh,
                    dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(mesh)))
                left -= chunk
        except BaseException:
            self.close()
            raise

    def close(self):
        tensors, self.tensors = self.tensors, []
        for value in tensors:
            self.operations.deallocate(value)


def zero_taps(operations, mesh, rows):
    """The synthetic request's five target taps: replicated-width zeros, sharded 2560 per chip as the
    prefill capture's are (dflash_prefill_window.PrefillWindowCapture.outputs)."""
    import torch

    return tuple(operations.from_torch(torch.zeros((1, 1, rows, FEATURE_WIDTH), dtype=torch.bfloat16), device=mesh,
                                       dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                       memory_config=operations.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3))
                 for _ in range(TAP_COUNT))


def _log(template, *values):
    from dflash_device import pindiag

    pindiag(template, *values)


class ParkedSlot:
    """One pool slot of the set: its parked engine and device while it has them. `state` is 'parked' (idle,
    rebindable), 'serving' (a request holds the rebound engine), 'unparked' (no parked engine: today's
    per-request builds serve the slot until it is re-parked) or 'closed'."""

    def __init__(self, index, slot, owner=None):
        self.index, self.slot, self.owner = index, slot, owner
        self.device = self.engine = None
        self.state = 'unparked'
        self.rebinds = self.parks = 0
        # E4: why the request serving the slot left it unfit to park (a failed page binding, set by the bridge
        # factory), and why its engine refused to park (park_engine), read by park_drafter; None otherwise.
        self.unfit = self.refusal = None


class ParkedEngineSet:
    """Stage E's parked engines, one per pool slot, built at attach after the packed block (design section
    2.2) and closed before it (the attach registers it after the block).

    build(): for each slot, while the DRAM split holds an engine build plus a parked arrival (dram_short):
    the slot re-zeroed and native GDN slot 0 restored from its zeroed carry, so the synthetic warm's page-0
    K/V writes are finite; then today's device, proposal and engine build on a synthetic request - P_cap = 1,
    an all-page-0 table, zero taps, budget 16 - and a park. Slot 0 also warms the drafter at its steady
    state: a rebind at 2048 (the 2048-row projection and K/V seed) and publish_prewarm.warm, which skips below
    2048 and which no rebind runs. A slot the split cannot hold stays unparked, as does one whose park fails
    later (unpark): today's per-request build serves it until repark_idle rebuilds it at an idle moment.

    take() gives the next request the lowest free slot - the rule ServingBufferPool.acquire applies to
    unlent slots - so the same arrivals get the same slots, and segments, with the flag on or off: the
    slot's entry when it is parked, else None (today's build, whose acquire takes that same slot).
    rebind_slot() binds a taken slot to its request; park() parks it again, or unparks it.

    E4, the serving wiring: serving_request_factory.from_prefill rebinds the taken slot (rebind_parked), and the
    request's close parks it in two halves - park_engine for its engine, park_drafter for its device - so an
    engine that cannot park leaves its device to be closed with it (unpark). The worker hook's detach then
    retires the device's pair and quad drafter traces (after_park: the coordinator's release_parked) and rebuilds
    its released single-user capture when the split allows (rebuild_single); the lifecycle's idle moment does the
    same for every parked slot and re-parks the unparked ones (idle). E5, the admission: arrival_terms are the
    parked terms of the slot the next request will take, asked by the scheduler's predicate and the backstop.
    QWEN_FAST_GATE_DRAM_BALLAST (gate only) holds a ballast from the end of build() to close()."""

    def __init__(self, *, operations, model, sampler, helpers, pool, weights, fixtures, collectives, blocks,
                 capture_rows, components=None, environ=None, log=None):
        import verifier_engine

        environ = os.environ if environ is None else environ
        refuse_deferred(environ)
        self.window = project_rows(environ)
        self.audit = audit_enabled(environ)
        self.negative = negative_mode(environ)
        self.fault, self.faulted = fault_mode(environ), False
        self.ballast_size = ballast_bytes(environ)
        self.ballast = None
        if environ.get('QWEN_FAST_SHARED_CCL', '1') != '1' or environ.get('QWEN_FAST_EAGER_PROPOSAL') == '1':
            raise ValueError('%s=1 keeps one device per slot for the process: it needs the shared collectives '
                             '(QWEN_FAST_SHARED_CCL=1) and captured proposals (QWEN_FAST_EAGER_PROPOSAL unset)' % FLAG)
        if capture_rows != 4:
            raise ValueError('%s=1 parks engines with the sequential captures (1, 2, 4) beside the four-user '
                             'block; this attach caps them at %r' % (FLAG, capture_rows))
        blocks = tuple(blocks)
        if not blocks or any(getattr(block, 'carries_in_place', False) is not True for block in blocks):
            raise ValueError('%s=1 needs every packed block to read its carries in place (carries_in_place, '
                             'QWEN_FAST_VERIFY_T1 #3 in every layer): a padded round writes an idle segment\'s '
                             'carry and native slot 0 otherwise, and parked slots are always lent; blocks %r'
                             % (FLAG, [getattr(block, 'carries_in_place', None) for block in blocks]))
        if getattr(pool, 'helpers', None) is None or any(slot.lent for slot in pool.slots):
            raise ValueError('The parked engines are built over a pool with verifier storage, before any request')
        if verifier_engine._resident is not None:
            raise ValueError('The parked engines are built before any request engine exists: one is resident')
        if components is None:
            from serving_request_factory import device_components

            components = device_components()
        self.operations, self.model, self.sampler, self.helpers = operations, model, sampler, helpers
        self.pool, self.weights, self.fixtures, self.collectives = pool, weights, fixtures, collectives
        self.capture_rows, self.components, self.environ = capture_rows, components, environ
        self.log = _log if log is None else log
        self.slots = [ParkedSlot(index, slot, owner=self) for index, slot in enumerate(pool.slots)]
        self.unparks = self.reparks = 0
        self.attach_ms = None
        self.ledger = ReplayLedger(operations).install() if self.audit else None
        self.closed = False

    # -- building ------------------------------------------------------------------------------------------
    def reserve(self):
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        return dram_reserve_bytes(self.environ)

    def dram_short(self):
        """THE SPLIT's terms (serving_prefill_admission.split_short) the pool's reading is short of for one
        more engine build beside a parked arrival at the longest prompt; None when it holds or cannot be read
        (the S2 attach refuses a pool without statistics, W7)."""
        import serving_prefill_admission as admission

        reserve = self.reserve()
        reading, reason = admission.dram_reading(self.pool)
        if reading is None:
            return None
        need = admission.engine_build_peak() + parked_arrival_need(reserve, self.window)
        short = admission.split_short(reading['free'], reading['largest_free'], need, reserve,
                                      reading['trace_largest_free'],
                                      contiguous=admission.admission_contiguous_need(admission.PREFILL_TRANSIENT_FROM,
                                                                                     reserve))
        return (short, reading, need) if short else None

    def build(self):
        """Park an engine on every slot the DRAM split holds, in slot order; returns how many."""
        import time
        import memory_ledger
        import verifier_engine

        started = time.perf_counter()
        built = 0
        for entry in self.slots:
            stop = self.dram_short()
            if stop is not None:
                short, reading, need = stop
                self.log(STOPPED_MARKER, entry.index, len(self.slots), '+'.join(short), reading['free'],
                         reading['largest_free'], need)
                break
            self.build_slot(entry, warm=entry.index == 0)
            built += 1
        verifier_engine.note_prefill()
        if self.fault:
            self.log(NEGATIVE_MARKER + 'fault={} (gate only: the first park is refused once)', self.fault)
        if self.negative:
            self.log(NEGATIVE_MARKER + 'mode={} (gate only: every rebind is deliberately broken)', self.negative)
        self.attach_ms = (time.perf_counter() - started) * 1000
        self.log(BUILT_MARKER + 'k={} of {} attach_ms={:.1f} {}', built, len(self.slots), self.attach_ms,
                 self.dram_text())
        memory_ledger.record('P7p', point='parked k=%d' % built, parked_engines=self)
        if self.ballast_size:
            # G-E3's ballast, after the P7p reading, so the baseline it records is the parked set's own.
            self.ballast = DramBallast(self.operations, self.model.mesh_device, self.ballast_size)
            self.log(BALLAST_MARKER + 'bytes={} buffers={} (gate only; unread) {}', self.ballast.size,
                     len(self.ballast.tensors), self.dram_text())
        return built

    def dram_text(self):
        import serving_prefill_admission as admission

        reading, reason = admission.dram_reading(self.pool)
        if reading is None:
            return 'dram unavailable (%s)' % reason
        trace = self.pool.trace_statistics() if callable(getattr(self.pool, 'trace_statistics', None)) else {}
        used = 'unavailable' if isinstance(trace, dict) else max(int(chip['allocated']) for chip in trace)
        return 'trace_used=%s free=%d largest_free=%d' % (used, reading['free'], reading['largest_free'])

    def build_slot(self, entry, *, warm):
        """The synthetic build on one unlent slot, parked (the class docstring)."""
        import torch
        from serving_page_binding import validate_initial_capture_pages
        from serving_request_factory import single_proposal_bucket

        slot, operations, components = entry.slot, self.operations, self.components
        if slot.lent or entry.state not in ('unparked',):
            raise ValueError('Pool slot %d must be free to park an engine on it' % entry.index)
        pages = torch.zeros((1, self.pool.page_width), dtype=torch.int32)
        validate_initial_capture_pages(pages, (0,), position=SYNTHETIC_POSITION, output_budget=SYNTHETIC_BUDGET)
        # Native slot 0 from the slot's zeroed carry: the synthetic warm forwards write page 0's K/V from it.
        self.pool.rezero(slot)
        for helper, snapshot in zip(self.helpers, slot.verifier.carry, strict=True):
            helper.restore(snapshot)
        _, layers, projection, selector = self.fixtures
        device = engine = session = None
        taps = zero_taps(operations, self.model.mesh_device, SYNTHETIC_POSITION)
        try:
            try:
                device = components.device(operations, self.model, self.collectives, layers, projection, selector, taps,
                    position=SYNTHETIC_POSITION, block_rows=16, proposal_capture=True, max_new_tokens=SYNTHETIC_BUDGET,
                    fused_convolution=True, feature_start=0, cache_history=True, cache_projection_capture=False,
                    live_query_qk=False, native_proposal_attention=True, defer_proposal_capture=True,
                    buffer_pool=self.pool, shared_weights=self.weights)
            finally:
                for value in taps:
                    operations.deallocate(value)
            if device.pool_slot is not slot:
                raise ValueError('The synthetic device for pool slot %d was lent slot %r'
                                 % (entry.index, getattr(device.pool_slot, 'index', None)))
            runtime = components.runtime(device, position=SYNTHETIC_POSITION)
            session = components.session('parked-%d' % entry.index, (SYNTHETIC_TOKEN,), SYNTHETIC_SEED,
                vocab_size=self.model.args.vocab_size, max_new_tokens=SYNTHETIC_BUDGET, eos_ids=(),
                neural={'dflash2': runtime}, verifier_rows=16, lookup_enabled=False)

            def prepare_proposal(built):
                if device.proposal_capture is not None:
                    raise ValueError('Proposal trace already captured before verifier allocation')
                with single_proposal_bucket():
                    device.proposal_capture = components.proposal(device, max_new_tokens=SYNTHETIC_BUDGET)

            engine = components.engine(self.model, session, pages, self.helpers, sampler=self.sampler,
                norm_batch=True, attention_replay=False, replay_group_rows=4, max_verify_rows=16,
                native_sampling_rows=True, retain_feature_taps=TARGET_TAPS, commit_only_gdn=True,
                target_attention_t16=False, before_capture=prepare_proposal, storage=slot.verifier,
                capture_rows=self.capture_rows)
            if warm:
                self.warm_drafter(device, engine)
            reason = engine.park() or park_device(device)
            if reason is not None:
                raise ValueError('The synthetic engine of pool slot %d cannot park: %s' % (entry.index, reason))
            session.close(session.request_id)
        except BaseException:
            if engine is not None and engine.phase != 'closed':
                if engine.phase in ('verifying', 'verified', 'committing'):
                    engine.phase = 'failed'
                engine.close()
            if device is not None:
                device.close()
            raise
        entry.device, entry.engine, entry.state = device, engine, 'parked'

    def warm_drafter(self, device, engine):
        """Slot 0's drafter at its steady state: a rebind at 2048 compiles the 2048-row projection and K/V
        seed, and publish_prewarm (QWEN_FAST_PUBLISH_PREWARM=1) then finds history_rows == 2048 and warms the
        publication programs - which no rebind runs, so the first request after a restart pays neither."""
        import time

        started = time.perf_counter()
        taps = zero_taps(self.operations, self.model.mesh_device, HISTORY_ROWS)
        try:
            rebind_device(device, taps, position=HISTORY_ROWS, window=self.window)
        finally:
            for value in taps:
                self.operations.deallocate(value)
        warmed = ()
        if self.environ.get('QWEN_FAST_PUBLISH_PREWARM') == '1':
            import publish_prewarm

            warmed = publish_prewarm.warm(device, engine)
        self.log(WARM_MARKER + 'P={} ms={:.1f} window={} prewarm_pairs={}', HISTORY_ROWS,
                 (time.perf_counter() - started) * 1000, self.window, len(warmed))

    # -- serving -------------------------------------------------------------------------------------------
    def peek(self):
        """The entry take() would give the next request, without taking it: the lowest free slot's when it is
        parked, else None (today's build takes that slot, or no slot is free)."""
        if self.closed:
            return None
        for entry in self.slots:
            if entry.state == 'parked':
                return entry
            if entry.state == 'unparked' and not entry.slot.lent:
                return None
        return None

    def take(self):
        """The next request's slot (the class docstring): its entry, now 'serving', when it is parked; else
        None, and today's per-request build takes it."""
        if self.closed:
            raise ValueError('The parked engines are closed')
        entry = self.peek()
        if entry is not None:
            entry.state = 'serving'
        return entry

    def single_released(self, entry):
        """Whether the slot's device has no single-user proposal capture: a pair or the quad released it, and
        neither a park nor a rebind has rebuilt it yet."""
        return entry.device is not None and single_capture(entry.device) is None

    def arrival_rebind_bytes(self):
        """R for this set's projection window."""
        return rebind_peak_bytes(self.window)

    def slot_terms(self, entry):
        """THE PARKED TERMS of a request on `entry` (serving_prefill_admission): dict(rebind=R, single=S when its
        single was released, else 0)."""
        return dict(rebind=self.arrival_rebind_bytes(),
                    single=single_capture_bytes() if self.single_released(entry) else 0)

    def arrival_terms(self):
        """E5: the terms the next request is admitted and backstopped on (serving_prefill_admission.dram_predicate,
        serving_request_factory.dram_backstop): None when the slot it will take is served by today's per-request
        build (today's terms), else slot_terms of its parked slot. Read at each call, so it follows every park,
        unpark and rebuild."""
        entry = self.peek()
        return None if entry is None else self.slot_terms(entry)

    def rebind_slot(self, entry, taps, pages, *, position, make_session, request_id=None, budget=None):
        """Bind a taken slot to its request (design section 2.3, steps 4-6): the device from the prefill's
        taps, the drafter runtime over it, the session make_session(runtime) returns, and the engine over the
        session and the request's page table. The caller has run the request's host checks and adopted its
        prefill slot. A failure unparks the slot and propagates, as a failed build does."""
        import time

        if entry.state != 'serving' or entry not in self.slots:
            raise ValueError('Only a slot this set gave out can be rebound')
        started = time.perf_counter()
        try:
            if self.audit:
                entry.slot.verify()
            info = rebind_device(entry.device, taps, position=position, window=self.window,
                                 **(dict(negative=self.negative) if self.negative else {}))
            runtime = self.components.runtime(entry.device, position=position)
            session = make_session(runtime)
            if self.negative == 'carry':
                # The gate's negative control: this rebind seeds no carry (an instance attribute shadows the method
                # for this call only), so the engine's first verify restores the slot's zeroed carry.
                entry.engine.save_carry = lambda: None
            try:
                entry.engine.rebind(session, pages)
            finally:
                entry.engine.__dict__.pop('save_carry', None)
        except BaseException as failure:
            self.unpark(entry, 'rebind failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
            raise
        entry.rebinds += 1
        self.log(REBIND_MARKER + 'req={} slot={} gen={} P={} budget={} ms={:.1f} single_rebuilt={} window={}',
                 str(request_id if request_id is not None else session.request_id)[:48], entry.index,
                 entry.device.rebind_generation, position, budget if budget is not None else session.max_new_tokens,
                 (time.perf_counter() - started) * 1000, int(info['single_rebuilt']), info['window'])
        return SimpleNamespace(device=entry.device, engine=entry.engine, runtime=runtime, session=session, **info)

    def park(self, entry, *, reason=None):
        """The request on a slot finished: park its engine (which fences first) and its device again; or,
        when either cannot park or `reason` says the request left it unfit (a failed page binding), unpark the
        slot. Returns None when parked, else why it was unparked. park_engine then park_drafter."""
        if reason is not None:
            entry.unfit = reason
        self.park_engine(entry)
        return self.park_drafter(entry)

    def park_engine(self, entry):
        """FastRequest's release_engine for a rebound slot (serving_request_factory.rebind_parked), where close()
        called engine.close(): the engine parks - fencing first, as close() does - unless the request left the slot
        unfit; why it cannot is kept for park_drafter, which closes it with the device. An engine with a block in
        flight raises as close() raises, and the slot stays serving, as today's request stays open."""
        if entry.state != 'serving' or entry not in self.slots:
            raise ValueError('Only a serving slot of this set can park')
        if self.fault == 'park' and not self.faulted and entry.unfit is None:
            # G-E2's injected fault, once: the engine is left as it is (unparked, closed by park_drafter).
            self.faulted = True
            entry.refusal = FAULT_REASON
            return
        entry.refusal = entry.unfit if entry.unfit is not None else entry.engine.park()

    def park_drafter(self, entry):
        """FastRequest's release_drafter for a rebound slot, after park_engine: the device parks (park_device:
        a fence, its pending proposal dropped, its pooled slot verified) and the slot is parked; or, when the engine
        or the device cannot park, the slot is unparked - both closed as today's close closes them, the slot back
        to the pool - until an idle moment re-parks it. Returns None when parked, else why it was unparked."""
        if entry.state != 'serving' or entry not in self.slots:
            raise ValueError('Only a serving slot of this set can park')
        reason, entry.refusal, entry.unfit = entry.refusal, None, None
        if reason is None:
            reason = park_device(entry.device)
        if reason is None:
            entry.state = 'parked'
            entry.parks += 1
            return None
        self.unpark(entry, reason)
        return reason

    # -- after a park: the device's drafter traces and its single (design section 2.4) -----------------------
    def after_park(self, entry, coordinator=None):
        """The worker hook's release when a request on `entry` detaches, after its close parked the slot
        (release_parked, from FastWorkerHook.detach): the device's pair and quad traces retired now by the
        coordinator's release_parked - a parked device never closes, so release_closed never sees them dead - then
        its released single rebuilt when the split allows (rebuild_single). A slot the close unparked has a closed
        device, whose traces release_closed retired. A failed retirement is logged and left to the coordinator's
        generation check (a rebound device's rebind_generation differs from the one its traces were captured at);
        a failed rebuild propagates, as a failed capture does anywhere. Returns dict(released=, single=) or None."""
        if self.closed or entry.state != 'parked' or entry.device is None:
            return None
        released = None
        if coordinator is not None:
            try:
                released = coordinator.release_parked(entry.device)
            except Exception as failure:
                self.log('[PACKED-PROPOSE] release parked slot={} failed ({}: {}); a rebound device\'s generation '
                         'retires them', entry.index, type(failure).__name__, str(failure)[:160])
        return dict(released=released, single=self.rebuild_single(entry, 'park'))

    def single_short(self):
        """THE SPLIT's terms (serving_prefill_admission.split_short) a single-capture rebuild now would leave short
        for the longest parked arrival - its need counted with S, which the rebuild takes, and its trace region
        with the single's trace; None when it holds or the pool cannot be read."""
        import serving_prefill_admission as admission

        reserve = self.reserve()
        reading, reason = admission.dram_reading(self.pool)
        if reading is None:
            return None
        need = parked_arrival_need(reserve, self.window, single_released=True)
        short = admission.split_short(reading['free'], reading['largest_free'], need, reserve,
                                      reading['trace_largest_free'],
                                      contiguous=admission.admission_contiguous_need(admission.PREFILL_TRANSIENT_FROM,
                                                                                     reserve),
                                      trace_need=admission.parked_trace_need(True))
        return (short, reading, need) if short else None

    def rebuild_single(self, entry, moment):
        """A parked slot's released single-user capture rebuilt now, under the single bucket (build_single_capture),
        iff the split still admits the longest parked arrival after it (single_short); else kept released, and the
        slot's arrival terms carry S until its next rebind rebuilds it. True when rebuilt, False when kept, None
        when there was nothing to rebuild. A pair's view left on the device is dropped: its trace was retired."""
        import time

        if entry.state != 'parked' or not self.single_released(entry):
            return None
        stop = self.single_short()
        if stop is not None:
            short, reading, need = stop
            self.log(SINGLE_KEPT_MARKER, entry.index, moment, '+'.join(short), reading['free'],
                     reading['largest_free'], need)
            return False
        import memory_ledger

        started = time.perf_counter()
        device = entry.device
        # The ledger reads the rebuild either side (a no-op unless QWEN_FAST_MEMORY_LEDGER=1), at the measured S.
        token = memory_ledger.before('single', estimate=single_capture_bytes(), point='slot=%d at=%s' % (entry.index,
                                                                                                         moment))
        try:
            device.proposal_capture = None
            device.proposal_capture = build_single_capture(device)
            device._packed_capture_released = False
        finally:
            memory_ledger.after(token)
        self.log(SINGLE_REBUILT_MARKER, entry.index, moment, (time.perf_counter() - started) * 1000)
        return True

    def idle(self):
        """The lifecycle's idle moment (serving_lifecycle: no decoder, no prefill in flight, a step that schedules
        nothing): every parked slot's released single rebuilt while the split allows, then every unparked slot the
        pool has not lent re-parked (repark_idle). Returns dict(singles=[slots rebuilt], reparked=[slots])."""
        if self.closed:
            return dict(singles=[], reparked=[])
        singles = [entry.index for entry in self.slots if self.rebuild_single(entry, 'idle')]
        return dict(singles=singles, reparked=self.repark_idle())

    def unpark(self, entry, reason):
        """Close the slot's engine and device as today's request close does - which returns the slot and the
        weights - and leave it to today's per-request builds until repark_idle."""
        engine, device = entry.engine, entry.device
        entry.engine = entry.device = None
        entry.state = 'unparked'
        self.unparks += 1
        self.log(UNPARKED_MARKER, entry.index, reason)
        try:
            if engine is not None and engine.phase != 'closed':
                if engine.phase in ('verifying', 'verified', 'committing'):
                    engine.phase = 'failed'
                engine.close()
        finally:
            if device is not None and not device.closed:
                device.close()

    def repark_idle(self):
        """At an idle moment (the caller's: no decoder and no prefill in flight), rebuild every unparked slot
        the pool has not lent, lowest first, while the DRAM split holds; an engine captured at a real position
        is never parked. Returns the slots re-parked."""
        import time
        import verifier_engine

        reparked = []
        for entry in self.slots:
            if entry.state != 'unparked' or entry.slot.lent:
                continue
            stop = self.dram_short()
            if stop is not None:
                short, reading, need = stop
                self.log(STOPPED_MARKER, entry.index, len(self.slots), '+'.join(short), reading['free'],
                         reading['largest_free'], need)
                break
            started = time.perf_counter()
            self.build_slot(entry, warm=False)
            self.reparks += 1
            reparked.append(entry.index)
            self.log(REPARKED_MARKER, entry.index, (time.perf_counter() - started) * 1000)
        if reparked:
            verifier_engine.note_prefill()
        return reparked

    def describe(self):
        return dict(window=self.window, audit=self.audit, unparks=self.unparks, reparks=self.reparks,
                    slots=[dict(index=entry.index, state=entry.state, rebinds=entry.rebinds, parks=entry.parks)
                           for entry in self.slots])

    def close(self):
        """Every engine and device closed - parked, or left serving - which returns every slot and the weights.
        The attach registers the set after the block, so it closes after the lifecycle and before the block,
        the weights and the pool; the pool and the weights refuse to close while anything is still lent."""
        if self.closed:
            return
        self.closed = True
        if self.ballast is not None:
            self.ballast.close()
            self.ballast = None
        failures = []
        for entry in self.slots:
            for owner in (entry.engine, entry.device):
                if owner is None:
                    continue
                try:
                    if owner is entry.engine and owner.phase in ('verifying', 'verified', 'committing'):
                        owner.phase = 'failed'
                    owner.close()
                except BaseException as failure:
                    failures.append(failure)
            entry.engine = entry.device = None
            entry.state = 'closed'
        if self.ledger is not None:
            self.ledger.uninstall()
        if failures:
            raise failures[0]


def release_parked(coordinator, request):
    """Stage E's release at a detach (serving_worker_hook.release_parked, QWEN_FAST_PARKED_ENGINES=1 only): for a
    request that ran on a parked slot (serving_request_factory.rebind_parked sets its `parked_slot`), its set's
    after_park with the hook's proposal coordinator (None when the hook never packed a proposal). None for a request
    today's per-request build served, whose closed device release_closed already covered."""
    entry = getattr(request, 'parked_slot', None)
    owner = getattr(entry, 'owner', None)
    if owner is None:
        return None
    return owner.after_park(entry, coordinator)
