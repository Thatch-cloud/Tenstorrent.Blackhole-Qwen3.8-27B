"""Engine reuse (QWEN_FAST_PARKED_ENGINES, default off): parked per-slot engines on the four-card fast path.

WHY. Every admitted request builds its own engine after its prefill: a DFlashDevice over the pool slot it acquires, the device's
single-user proposal trace, and a VerifierEngine with three verify traces and eight commit traces. A warm build costs 2.1-2.7 s on the
path to the request's first token and as a stall on every live decoder. None of those traces is shaped by the request: positions,
rotary tables, page tables and the SDPA cur_pos are restaged device data (verifier_inputs.stage_inputs, the fixture's pooled page
tables), and the proposal trace takes one 2048 bucket whatever the prompt (serving_request_factory.single_proposal_bucket). So one
engine per pool slot, built once at attach on a synthetic request, can serve every request that slot ever takes.

WHAT. At attach, after the packed blocks, ParkedEngineSet builds one engine per pool slot on a synthetic request (P_cap = 1, an
all-page-0 table, zero taps, native GDN slot 0 zeroed first) and PARKS it: the engine keeps its captures and its pooled storage, the
device keeps its lent slot and its proposal trace. A request REBINDS a parked slot instead of building (rebind_device,
verifier_engine_tp.VerifierEngine.rebind), which allocates nothing that outlives the call, and parks it again when it finishes
(park_device, VerifierEngine.park). With QWEN_FAST_PARKED_DRAFTS=1 ("2c") the pair and quad drafter traces are bound to the slots for the
process too (DraftTraceBook), so an admission or a departure never recaptures them; without it they are retired at every park and recaptured
at the next formation (E1).

EXACTNESS. Output stays byte-identical to a fresh build. The rules the census holds the code to:
  R1  a buffer a trace reads before that replay writes it is rewritten before every replay that reads it, or allocated before every
      trace that could replay while it lives (pooled);
  R2  a buffer trace A writes and a later trace B or the host reads does not sit in the holes of a trace C that replays in between: the
      sequential step verifies, reads back and commits one request at a time (ReplayLedger checks it under QWEN_FAST_PARKED_AUDIT=1);
  R3  after a rebind the parked engine's host state equals a fresh engine's, apart from whitelisted counters, and its device state
      differs from a fresh engine's only in buffers every replay writes before it reads them;
  R4  the widths an engine asks and serves are a cold engine's for the same request (verifier_engine_tp: widths, request_widths);
  R5  the slot a request takes is the one the pool's own placement would give it (place, over serving slots).
The drafter side is a speed and crash property, not an exactness one: greedy verification emits the target's argmax whatever was
drafted. Its gate is drafter equivalence (solo parked against solo fresh), not token equality.

FALLBACK. A rebind that cannot run is refused on the HOST before any device write (rebind_refusal) or, after the first device write,
fails with RebindFailed having unparked the slot (fence first; a fence that raises is engine-fatal, as a failed build is). The request
factory then runs today's cold build on the same slot (pool.slot_order), over the SAME prefill capture, which it owns in one place and
closes once (serving_request_factory.from_prefill). The cold build is admitted on today's DRAM terms, not the parked ones.

NOT BUILT. The release ladder (make_room) is pressure-driven only at an arrival that is short: book pairs, then released singles of other
parked slots, then one idle parked slot other than the target. Everything else a parked set holds stays held for the process, which is a
shipping decision the operator still has to make. QWEN_FAST_PARKED_AUDIT digests state per chip (audit_rebound_state), not every tensor
chip against chip: how a GDN state tensor is sharded is not assumed.

SOURCE PINS. The engine's parked phase lives in the four-card subclass (verifier_engine_tp) because verifier_engine.py is held byte for byte
by the frozen bundle inventory. draft_kv_history.py (text-patched at attach), extent_attention_replay.py and serving_page_binding.py are never
edited here: the full page-table rewrite is write_page_tables, held to VerifierPageBinding.refresh's operation sequence by a test.
"""

import os
import re
import time
from types import SimpleNamespace


FLAG = 'QWEN_FAST_PARKED_ENGINES'
DRAFTS_FLAG = 'QWEN_FAST_PARKED_DRAFTS'
PROJECT_ROWS_FLAG = 'QWEN_FAST_PARKED_PROJECT_ROWS'
AUDIT_FLAG = 'QWEN_FAST_PARKED_AUDIT'
# The gate's negative controls (GATE ONLY: serving_c2_contract.parked_problems refuses the knob outside a gate profile). Each proves its gate
# can see the breakage it exists for:
#   carry    skips save_carry at the rebind, so the engine restores the slot's zeroed carry at its first verify: tokens MUST diverge;
#   drafter  skips the rezero and the reseed of the K/V banks: tokens stay equal (greedy verification), the drafter-equivalence judge MUST fail;
#   pages    skips the page-table rewrite, so the previous request's table stays: tokens MUST diverge (or the A6 sentinel guard refuses);
#   widths   switches R4's restriction off: the ticket-width line of the judge MUST fail (tokens may stay equal; they are not the judge).
NEGATIVE_FLAG = 'QWEN_FAST_PARKED_NEGATIVE'
NEGATIVES = ('carry', 'drafter', 'pages', 'widths')
NEGATIVE_MARKER = '[PINDIAG] parked negative control '
# Injected faults (GATE ONLY): 'park' refuses the first park after attach, once; 'rebind' refuses the first rebind, once, on the host before
# any device write. Either unparks the slot, which serves today's per-request build and re-parks later (design 5.9).
FAULT_FLAG = 'QWEN_FAST_PARKED_FAULT'
FAULTS = ('park', 'rebind')
FAULT_REASON = 'injected park fault (gate only)'
REBIND_FAULT_REASON = 'injected rebind fault (gate only)'
# The runtime kill switch: this file beside prefix-reuse.off and levern.off under the image's .qwen-c2 directory (levern_policy.OffSwitch).
OFF_FILE = 'parked.off'
OFF_PATH = '/models/.qwen-c2/parked.off'
OFF_PATH_ENV = 'QWEN_FAST_PARKED_OFF_PATH'
OFF_MARKER = '[PINDIAG] parked engines off (kill switch): unparked={}'

# The windowed drafter projection at a rebind (rebind_device): rows per project_features call. 0 is one call over the whole window.
DEFAULT_PROJECT_ROWS = 256
PROJECTION_CHUNK_ROWS = 32
HISTORY_ROWS = 2048

# The serving modules the port edits (each must reach the image through its overlay, none may be a frozen-recipe pin), and the ones it
# must leave byte-identical.
EDITED_MODULES = ('verifier_engine_tp.py', 'serving_buffer_pool.py', 'serving_runtime.py', 'serving_parked_engines.py',
                  'serving_request_factory.py', 'serving_fast_request.py', 'serving_worker_hook.py',
                  'dflash_packed_proposal_coordinator.py', 'serving_prefill_admission.py', 'serving_lifecycle.py',
                  'levern_policy.py')
NEVER_EDITED = ('verifier_engine.py', 'draft_kv_history.py', 'extent_attention_replay.py', 'serving_page_binding.py')

# Every "for the request's life" or "permanent" claim in the drafter and coordinator sources, with what a parked device (rebound at any
# position, kept for the process) does to it. test_parked_tp4_census fails on a claim in those files this table does not list.
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
         status='unchanged for a hook; under the draft book the counters reset at hook close', guard='coordinator.close'),
    dict(file='dflash_device.py', text="are fixed for a request's life once its proposal is captured",
         status='changed: they are fixed for the process; a rebind keeps kv_history, progress None and the capture '
                '(rebuilt when released)', guard='rebind_device repoints proposal_capture.kv_history'),
    dict(file='dflash_device.py', text="every proposal for the request's life",
         status='changed: the pooled query is read for the process life', guard='pooled: allocated at attach'),
    dict(file='dflash_device.py', text='permanently true once this device is past the prefill ramp',
         status='changed: a rebind below 2048 re-enters the ramp', guard='rows is recomputed per publication'),
    dict(file='dflash_device.py', text='self.history_stale is set and never cleared',
         status='changed: rebind_device clears it with the history it rewrites', guard='rebind_device'),
    dict(file='dflash_device.py', text='the history is marked stale for good',
         status='changed: cleared at the next rebind', guard='rebind_device'),
    dict(file='dflash_proposal_trace.py', text='for the life of this object',
         status='unchanged: pair traces are retired when a member parks (E1: the coordinator\'s release_parked) or '
                'kept for the process by the draft book (2c, which retires a member\'s traces before it closes)', guard=None),
    dict(file='dflash_proposal_trace.py', text='PERMANENT, monotonic state every request',
         status='changed: per request, not per device', guard='the pair gate reads history_rows every round'),
)
LIFETIME_PHRASES = ("request's life", 'permanent', 'monotonic', 'for the life of', 'for good', 'never cleared')
LIFETIME_FILES = ('dflash_packed_proposal_coordinator.py', 'dflash_device.py', 'dflash_proposal_trace.py')

# Instance attributes a parked device or engine may gain after its attach park without that being a leak; any other new key, or any
# instance attribute that shadows a class method (fused_commit.install_fused_commit and dflash_traced_publish.install_publish_options hang
# overrides on the drafter and its K/V history for one commit and delete them), unparks the slot at its next park (instance_problems).
ALLOWED_NEW_INSTANCE_KEYS = frozenset({'rebind_generation', 'history_stale', '_packed_capture_released', 'rebind_ms',
                                       'replay_mark', 'request_widths', 'rebind_runtime', 'last_rebind'})


class RebindRefused(ValueError):
    """The rebind was refused on the host before any device write: the slot is still parked and nothing was written."""


class RebindFailed(Exception):
    """The rebind failed after its first device write and the slot is unparked (fence returned): the caller cold-builds on the same slot."""

    def __init__(self, slot_index, reason, cause=None):
        super().__init__('parked rebind of slot %s failed: %s' % (slot_index, reason))
        self.slot_index, self.reason, self.cause = slot_index, reason, cause


def _strict_bool(name, environ=None):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def parked_engines_enabled(environ=None):
    """QWEN_FAST_PARKED_ENGINES: unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(FLAG, environ)


def drafts_enabled(environ=None):
    """QWEN_FAST_PARKED_DRAFTS (2c): unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(DRAFTS_FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_PARKED_AUDIT (gate): unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(AUDIT_FLAG, environ)


def negative_mode(environ=None):
    """QWEN_FAST_PARKED_NEGATIVE (gate only): None when unset or empty, else one of NEGATIVES; anything else is refused."""
    value = (os.environ if environ is None else environ).get(NEGATIVE_FLAG)
    if value in (None, ''):
        return None
    if value not in NEGATIVES:
        raise ValueError('%s must be one of %s, got %r' % (NEGATIVE_FLAG, ', '.join(NEGATIVES), value))
    return value


def fault_mode(environ=None):
    """QWEN_FAST_PARKED_FAULT (gate only): None when unset or empty, else one of FAULTS; anything else is refused."""
    value = (os.environ if environ is None else environ).get(FAULT_FLAG)
    if value in (None, ''):
        return None
    if value not in FAULTS:
        raise ValueError('%s must be one of %s, got %r' % (FAULT_FLAG, ', '.join(FAULTS), value))
    return value


def project_rows(environ=None):
    """QWEN_FAST_PARKED_PROJECT_ROWS: the rebind's projection window, a multiple of the projection's 32-row chunk up to 2048 (so every window
    starts on a chunk boundary of the whole call), or 0 for one call over the whole window. Default 256."""
    text = (os.environ if environ is None else environ).get(PROJECT_ROWS_FLAG, str(DEFAULT_PROJECT_ROWS))
    if type(text) is not str or re.fullmatch('0|[1-9][0-9]*', text) is None:
        raise ValueError('%s must be a decimal row count, got %r' % (PROJECT_ROWS_FLAG, text))
    rows = int(text)
    if rows and (rows % PROJECTION_CHUNK_ROWS or rows > HISTORY_ROWS):
        raise ValueError('%s must be 0 or a multiple of %d up to %d, got %d'
                         % (PROJECT_ROWS_FLAG, PROJECTION_CHUNK_ROWS, HISTORY_ROWS, rows))
    return rows


def chip_count():
    import tp_shapes

    return tp_shapes.chip_count()


def page_table_bindings(engine):
    """{per-chip address identity: (tensor, shape)} for every page table a sequential engine's fixtures read - each bucket's pages and
    singleton page table - collected and checked as VerifierPageBinding collects its bindings. The identity has one address per chip
    (tp_shapes.chip_count()). Refused: any other page ownership (a replay reader's per-bundle tables, a grouped reader, a reader or writer
    over tables of its own), which a parked engine never has."""
    from gdn_multitoken_conv import addresses

    operations, capacity, chips = engine.operations, engine.pages.shape[1], chip_count()
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
        if len(shape) != 2 or shape[0] < 1 or not 1 <= shape[1] <= capacity or len(identity) != chips:
            raise ValueError('Bounded %d-chip page metadata required' % chips)
        if identity in bindings and bindings[identity][1] != shape:
            raise ValueError('Aliased page metadata has conflicting geometry')
        bindings[identity] = (tensor, shape)
    if not bindings:
        raise ValueError('Captured verifier page metadata required')
    return bindings


def shard_digests(operations, tensor):
    """sha256 of every chip's bytes of a device tensor, in chip order."""
    import hashlib

    import torch

    return [hashlib.sha256(operations.to_torch(shard).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            .hexdigest() for shard in operations.get_device_tensors(tensor)]


def slot_zero_digests(engine):
    """Every chip's digest of every native GDN slot-0 state tensor, in helper order: a rebind reads slot 0 and never writes it."""
    return [shard_digests(engine.operations, value) for helper in engine.helpers for value in helper.live]


def zeroed_problem(pool, slot, samples=1):
    """A sample of every class of tensor the rezero wrote (one per class per layer, the first `samples`), read back on every chip: the
    first that is not all zero, or None."""
    import torch

    operations = pool.operations
    classes = {}
    for tensor in slot.zeroed:
        classes.setdefault((tuple(tensor.shape), str(tensor.dtype)), []).append(tensor)
    for key, tensors in sorted(classes.items(), key=lambda item: str(item[0])):
        for tensor in tensors[:samples]:
            for shard in operations.get_device_tensors(tensor):
                if bool(torch.any(operations.to_torch(shard) != 0)):
                    return 'a re-zeroed %s tensor of slot %d holds a non-zero value' % (key, slot.index)
    return None


def audit_rebound_state(engine, slot_index, log, *, slot0_before=None, zeroed=None):
    """QWEN_FAST_PARKED_AUDIT=1, at every rebind, after the engine's (design 6.3): digests of the rebound pooled state. The initial snapshot
    and the carry were both written from native slot 0 by the rebind, so on every chip each carry tensor's digest equals its initial tensor's
    (no assumption is made about how a state tensor is sharded across chips); native slot 0 reads the same before and after (a rebind never
    writes it); every page table the captures read is replicated, so every chip's digest is the same and equals the host table's bytes
    (D4). The active K/V banks are held by DraftKVHistory.audit at the reseed. Raises AssertionError naming the first mismatch; logs
    DIGEST_MARKER with the counts when it passes."""
    import torch

    operations = engine.operations
    snapshots = tables = 0
    for helper_index, (initial, carry) in enumerate(zip(engine.initial, engine.carry, strict=True)):
        for tensor_index, (left, right) in enumerate(zip(initial, carry, strict=True)):
            snapshots += 1
            a, b = shard_digests(operations, left), shard_digests(operations, right)
            if a != b:
                raise AssertionError('Rebound state differs: helper %d tensor %d, carry against initial (slot %d)'
                                     % (helper_index, tensor_index, slot_index))
    if slot0_before is not None and slot_zero_digests(engine) != slot0_before:
        raise AssertionError('The rebind wrote native GDN slot 0 (slot %d)' % slot_index)
    for tensor, shape in page_table_bindings(engine).values():
        tables += 1
        digests = shard_digests(operations, tensor)
        if len(set(digests)) != 1:
            raise AssertionError('A replicated page table differs between chips (slot %d)' % slot_index)
        wanted = engine.pages[:, :shape[1]].repeat(shape[0], 1).contiguous()
        host = operations.to_torch(operations.get_device_tensors(tensor)[0]).reshape(shape).to(torch.int32)
        if not torch.equal(host, wanted):
            raise AssertionError('A page table on the device differs from the request\'s host table (slot %d)' % slot_index)
    log(DIGEST_MARKER + 'slot={} snapshots={} tables={} slot0={} zeroed={} equal=1', slot_index, snapshots, tables,
        int(slot0_before is not None), int(zeroed is not None))


def write_page_tables(operations, mesh, bindings, host):
    """VerifierPageBinding.refresh's device write (serving_page_binding.py), for a whole table: every bound table's addresses and shape
    checked, each rewritten with the (1, width) host table's first columns repeated per row, through copy_host_to_device_tensor, one fence,
    and the addresses checked again. The same operations, in the same order, as refresh makes for the same host table
    (test_parked_tp4_engine); kept here because serving_page_binding.py is not edited."""
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


def write_null_tables(engine):
    """A parked engine's page tables name no request's blocks: every table the captures read is rewritten with the null block (page 0, which
    the KV reservation never allocates), so a stray replay of a parked engine could not write K/V through block ids vLLM has freed and
    may have handed to another seat. The engine's host table goes with them."""
    engine.pages.zero_()
    write_page_tables(engine.operations, engine.mesh, page_table_bindings(engine), engine.pages)


def null_tables_problem(engine):
    """The first parked page table that reads non-zero on any chip, or None."""
    import torch

    for tensor, _ in page_table_bindings(engine).values():
        for shard in engine.operations.get_device_tensors(tensor):
            if bool(torch.any(engine.operations.to_torch(shard) != 0)):
                return 'a parked engine\'s page table names a block'
    return None


class ReplayLedger:
    """R2's replay ledger (QWEN_FAST_PARKED_AUDIT=1): counts every trace replay the process makes by wrapping `operations.execute_trace`, and
    hands the count to verifier_engine_tp, whose verify notes it right after its own replay and whose publish asserts nothing replayed since
    (check_replay_mark). Every replay site calls execute_trace through the ttnn module at call time, so one wrapper sees them all. It also wraps
    begin_trace_capture, so a window of the log can say which replays were of traces captured inside it (foreign_since): a build replays the
    proposal trace it has just captured and no other. Installed by the serving runtime whenever the audit is on, with or without parked engines,
    so the flag-off control arm of the audit twin runs the same check (design K2)."""

    def __init__(self, operations):
        self.operations = operations
        self.count = 0
        self.log = []
        self.original = self.original_begin = self.previous = None
        self.installed = False

    def read(self):
        return self.count

    def mark(self):
        return len(self.log)

    def foreign_since(self, mark):
        """The handles replayed since `mark` that were not captured since it, in order."""
        captured = {entry[1] for entry in self.log[mark:] if entry[0] == 'capture'}
        return [entry[1] for entry in self.log[mark:] if entry[0] == 'replay' and entry[1] not in captured]

    def install(self):
        import verifier_engine_tp

        if self.installed:
            raise ValueError('The replay ledger is already installed')
        original, original_begin = self.operations.execute_trace, self.operations.begin_trace_capture

        def execute_trace(mesh, trace, *args, **kwargs):
            self.count += 1
            self.log.append(('replay', trace))
            return original(mesh, trace, *args, **kwargs)

        def begin_trace_capture(*args, **kwargs):
            trace = original_begin(*args, **kwargs)
            self.log.append(('capture', trace))
            return trace

        self.original, self.original_begin = original, original_begin
        self.operations.execute_trace = execute_trace
        self.operations.begin_trace_capture = begin_trace_capture
        self.previous = verifier_engine_tp.set_replay_count(self.read)
        self.installed = True
        return self

    def uninstall(self):
        import verifier_engine_tp

        if not self.installed:
            return
        self.operations.execute_trace = self.original
        self.operations.begin_trace_capture = self.original_begin
        verifier_engine_tp.set_replay_count(self.previous)
        self.installed = False


def install_replay_ledger(operations, environ=None):
    """The ledger installed, or None: only under QWEN_FAST_PARKED_AUDIT=1 (any other value but '0' is refused)."""
    if not audit_enabled(environ):
        return None
    return ReplayLedger(operations).install()


# -- the device side ---------------------------------------------------------------------------------------------------------------

def single_capture(device):
    """The device's own single-user proposal capture, through a pair's _PackedCaptureView when one is installed
    (dflash_packed_proposal_coordinator), or None when a pair or the quad released it."""
    from dflash_packed_proposal_coordinator import _PackedCaptureView

    capture = device.proposal_capture
    return capture._original if isinstance(capture, _PackedCaptureView) else capture


def build_single_capture(device):
    """The single-user proposal capture a device's build makes - one 2048 bucket, whatever the position
    (serving_request_factory.single_proposal_bucket) - for a device whose capture was released. Unscoped,
    PreparedDFlashProposal(device, max_new_tokens=1) builds a 256-1024 bucket below 2048, which a history that grows past it refuses
    ('Committed history exceeds prepared request contexts')."""
    from dflash_proposal_trace import PreparedDFlashProposal
    from serving_request_factory import single_proposal_bucket

    with single_proposal_bucket():
        return PreparedDFlashProposal(device, max_new_tokens=1)


def project_window(device, taps, count, *, window):
    """The constructor's history projection (DFlashDevice.__init__: project_features(features, history_rows)), in calls of `window` rows
    each, or in one call when `window` is 0 or covers `count`. Each call has its own temporaries scope, fence and release (project_features'
    retain=None path), so a window keeps about a window's intermediates alive where one call keeps all of them until it returns. The same
    chunks: project_features projects independent 32-row chunks with fixed program configurations, and every window starts on a chunk
    boundary of the whole call (project_rows keeps windows a multiple of 32), so only the last chunk of the last window can be partial, as in
    the whole call; the windows' outputs are joined by one concat, a copy."""
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
    """The device's half of a park: a fence first - discard_pending requires one (dflash_proposal_trace.PreparedDFlashProposal.
    discard_pending) and today's close takes it from engine.close - then the capture's pending proposal dropped. None when the device can
    park; else why, and its owner closes it as today. Its pair and quad traces are the coordinator's (E1) or the book's (2c) to retire."""
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


def peak_reading(pool):
    """The smallest free DRAM over the chips (serving_prefill_admission.dram_reading), or None when it cannot be read."""
    import serving_prefill_admission as admission

    reading, _ = admission.dram_reading(pool)
    return None if reading is None else reading['free']


def footprint(pool):
    """(trace-region bytes in use, smallest free DRAM bytes) over the chips, each None when it cannot be read: what a single rebuild's
    footprint is the difference of."""
    import serving_prefill_admission as admission

    reading, _ = admission.dram_reading(pool)
    free = None if reading is None else reading['free']
    used = None
    statistics = getattr(pool, 'trace_statistics', None)
    try:
        chips = statistics() if callable(statistics) else None
        if isinstance(chips, (list, tuple)) and chips:
            used = max(int(chip['allocated']) for chip in chips)
    except Exception:
        used = None
    return used, free


def draft_history_class():
    """The K/V history class the device's constructor picks: the pair's draft_kv_history, or at four cards its sibling draft_kv_history_tp
    (dflash_device.py; the bundle's class is text-patched at attach and carries the pair's 4 KV heads) - D3."""
    import tp_shapes

    if tp_shapes.chip_count() == tp_shapes.PAIR:
        from draft_kv_history import DraftKVHistory
    else:
        from draft_kv_history_tp import DraftKVHistory
    return DraftKVHistory


def rebind_device(device, taps, *, position, window=None, negative=None, audit=False):
    """Bind a parked DFlashDevice to a request prefilled to `position`, from its window's taps (the prefill capture's outputs), as its
    constructor seeds a fresh one, without capturing anything (design 3.4):
      1. a fence, then the capture's pending proposal dropped (discard_pending requires the fence);
      2. the pool slot re-zeroed as a loan zeroes it (ServingBufferPool.rezero), its buckets kept taken;
      3. the history projected (project_window: `window` rows per call, default project_rows()), padded to 2048 rows and copied into the
         slot's history - the canonical orientation, history then spare;
      4. a new DraftKVHistory (the class the constructor picks, D3) seeded over the slot's banks (the old one, pooled, owns nothing);
      5. the host state a fresh device has: position, history_rows, name, nothing pending, no calls or published rows, no audit digest or
         convolution checks, history_stale cleared, and rebind_generation advanced (the E1 coordinator keys a pair or quad trace on it);
      6. the capture unwrapped from a pair's or quad's view and pointed at the new cache, or - released by a pair or the quad - rebuilt under
         the single bucket (build_single_capture).
    `negative` == 'drafter' (the gate's negative control) skips steps 2 and 4: the banks keep the previous request's K/V and the old
    DraftKVHistory stays. `audit` (QWEN_FAST_PARKED_AUDIT=1) logs PEAK_MARKER: the DRAM the caller holds while the projection's outputs and
    the taps are all alive. Returns dict(ms=, single_rebuilt=, window=, zeroed=)."""
    from dflash_prefill_window import prefill_window
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
    before = peak_reading(slot.pool) if audit else None
    zeroed = None
    if negative != 'drafter':
        slot.pool.rezero(slot)
        if audit:
            zeroed = zeroed_problem(slot.pool, slot)
            if zeroed is not None:
                raise AssertionError(zeroed)
            zeroed = True
    projected = project_window(device, taps, rows, window=window)
    padded = operations.pad(projected, [(0, 0), (0, 0), (0, HISTORY_ROWS - rows), (0, 0)], 0.0)
    if audit:
        peak = peak_reading(slot.pool)
        if before is not None and peak is not None:
            _log(PEAK_MARKER + 'slot={} P={} free_before={} free_at_peak={} held={}', slot.index, position, before,
                 peak, before - peak)
    if addresses(operations, padded) != addresses(operations, projected):
        operations.deallocate(projected)
    operations.copy(padded, slot.history)
    operations.deallocate(padded)
    device.history, device.spare_history = slot.history, slot.spare_history
    operations.synchronize_device(device.mesh)
    if negative != 'drafter':
        device.kv_history = draft_history_class()(operations, device.mesh, [layer[0] for layer in device.layers], device.history,
            position=position, history_rows=rows, capture_projection=False, storage=slot.kv,
            **(dict(query=slot.query) if getattr(slot, 'query', None) is not None else {}))
        if device.progress is not None:
            device.kv_history.audit(device.history)
    else:
        # The drafter control keeps the stale banks, but its host frontier follows the request: the proposal replay refuses a cache whose
        # frontier does not match, and the control must run to a verdict, not crash the engine.
        device.kv_history.position, device.kv_history.history_rows = position, rows
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
    return dict(ms=(time.perf_counter() - started) * 1000, single_rebuilt=rebuilt, window=window, zeroed=zeroed)


# -- the set -----------------------------------------------------------------------------------------------------------------------

SYNTHETIC_POSITION = 1
SYNTHETIC_BUDGET = 16
SYNTHETIC_TOKEN = SYNTHETIC_SEED = 0
TARGET_TAPS = (5, 19, 33, 47, 61)
TAP_COUNT = 5
FEATURE_WIDTH = 5120
BUILT_MARKER = '[PINDIAG] parked engines built '
WARM_MARKER = '[PINDIAG] parked drafter warm '
REBIND_MARKER = '[PINDIAG] parked rebind '
PEAK_MARKER = '[PINDIAG] parked rebind peak '
DIGEST_MARKER = '[PINDIAG] parked rebind digest '
REFUSED_MARKER = '[PINDIAG] parked rebind refused '
UNPARKED_MARKER = '[PINDIAG] parked slot {} unparked: {}'
REPARKED_MARKER = '[PINDIAG] parked slot {} re-parked ms={:.1f}'
STOPPED_MARKER = '[PINDIAG] parked engines stopped at k={} of {}: short of {} (free={} largest_free={} need={})'
PROGRAMS_MARKER = '[PINDIAG] parked programs '
LADDER_MARKER = '[PINDIAG] parked release rung={} freed={} slot={}'
INSTANCE_MARKER = '[PINDIAG] parked instance state slot={} problem={}'
# A released single-user proposal capture rebuilt when its slot parks (after a detach, or at an idle moment), or kept released because the
# rebuild would leave the split short for the longest parked arrival.
SINGLE_REBUILT_MARKER = ('[PINDIAG] parked slot {} single rebuilt at {} ms={:.1f} trace_delta={} dram_delta={} '
                         '(bytes per chip, n/a when unread)')
IDLE_FAILED_MARKER = '[PINDIAG] parked slot {} idle {} failed, left as it was: {}'
SINGLE_KEPT_MARKER = '[PINDIAG] parked slot {} single kept released at {}: short of {} (free={} largest_free={} need={})'
# QWEN_FAST_GATE_DRAM_BALLAST (GATE ONLY, the ballast arm): bytes per chip held unread from the end of the parked build to close, so the
# admissions run at the boundary of the parked need.
BALLAST_FLAG = 'QWEN_FAST_GATE_DRAM_BALLAST'
BALLAST_MARKER = '[PINDIAG] gate dram ballast '
BALLAST_CHUNK_BYTES = 32 * 2 ** 20
BALLAST_ROW_BYTES = 1024 * 2
BALLAST_TILE_BYTES = 32 * BALLAST_ROW_BYTES


def rebind_peak_bytes(window):
    """R (serving_prefill_admission.PARKED_REBIND_BYTES; the whole call's when `window` is 0)."""
    import serving_prefill_admission as admission

    return admission.PARKED_REBIND_BYTES if window else admission.PARKED_REBIND_WHOLE_BYTES


def single_capture_bytes():
    """S (serving_prefill_admission.measured_single_capture_bytes, by chip count)."""
    import serving_prefill_admission as admission

    return admission.measured_single_capture_bytes()


def parked_arrival_need(reserve, window, *, single_released=False):
    """What a parked arrival at the longest prompt needs free per chip, the reserve in it (serving_prefill_admission.parked_need): the
    prefill's transient, the rebind's peak, a single-capture rebuild when that slot's single is released."""
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
    """The ballast (QWEN_FAST_GATE_DRAM_BALLAST, gate only): `size` bytes per chip, rounded up to whole 64 KiB tiles, replicated to every chip
    in buffers of at most BALLAST_CHUNK_BYTES, never read; close() frees them."""

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
    """The synthetic request's five target taps: replicated-width zeros, sharded per chip as the prefill capture's are
    (dflash_prefill_window.PrefillWindowCapture.outputs)."""
    import torch

    return tuple(operations.from_torch(torch.zeros((1, 1, rows, FEATURE_WIDTH), dtype=torch.bfloat16), device=mesh,
                                       dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                       memory_config=operations.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3))
                 for _ in range(TAP_COUNT))


def _log(template, *values):
    from dflash_device import pindiag

    pindiag(template, *values)


def instance_problems(owner, baseline, kind):
    """What a parked device or engine carries that a fresh one would not (design 5.10, risk R-leak): an instance attribute that shadows a
    class method (an installer's override left behind by a restore that raised part-way), or any key gained since `baseline` (the keys
    right after its attach park) other than the known per-request ones. None when clean."""
    live = vars(owner)
    shadowed = sorted(name for name in live if callable(getattr(type(owner), name, None)))
    if shadowed:
        return '%s instance attributes shadow methods: %s' % (kind, ','.join(shadowed))
    gained = sorted(set(live) - baseline - ALLOWED_NEW_INSTANCE_KEYS)
    if gained:
        return '%s gained instance attributes: %s' % (kind, ','.join(gained))
    return None


class ParkedSlot:
    """One pool slot of the set: its parked engine and device while it has them. `state` is 'parked' (idle, rebindable), 'serving' (a request
    holds the rebound engine), 'unparked' (no parked engine: today's per-request builds serve the slot until it is re-parked) or 'closed'."""

    def __init__(self, index, slot, owner=None):
        self.index, self.slot, self.owner = index, slot, owner
        self.device = self.engine = None
        self.state = 'unparked'
        self.rebinds = self.parks = 0
        # Why the request serving the slot left it unfit to park (a failed page binding, set by the bridge factory), and why its engine
        # refused to park (park_engine), read by park_drafter; None otherwise.
        self.unfit = self.refusal = None
        self.device_keys = self.engine_keys = frozenset()


class DraftTraceBook:
    """2c (QWEN_FAST_PARKED_DRAFTS=1): the pair and block-quad drafter traces bound to the pool slots for the process. The book is the SOLE
    holder: a coordinator built while it is registered uses `pairs` and `quad_blocks` themselves (the same dict objects), so a trace is listed
    once, closed once and retired through retire() by whichever path decides to. The coordinator's own closes (hook close, release_closed,
    _retire_quad, _disable_quad, _block_quad, the pair release) route here; hook close drops views and counters only.

      pairs        {slot pair: (device_a, device_b, trace, released)}   as PackedProposalCoordinator.pairs
      quad_blocks  {slots: (devices, trace, released)}                  as PackedProposalCoordinator.quad_blocks
    """

    def __init__(self, log=None):
        self.pairs, self.quad_blocks = {}, {}
        self.closes = {}
        self.log = _log if log is None else log
        self.closed = False

    def _members(self, kind, key):
        if kind == 'pair':
            entry = self.pairs.get(key)
            return () if entry is None else (entry[0], entry[1])
        entry = self.quad_blocks.get(key)
        return () if entry is None else tuple(entry[0])

    def retire(self, kind, key):
        """Close the trace listed under (kind, key) ONCE, delete the entry, and unwrap every surviving member's capture view back to its own
        single-user capture. Returns whether there was an entry."""
        table = self.pairs if kind == 'pair' else self.quad_blocks
        if key not in table:
            return False
        members = self._members(kind, key)
        trace = table.pop(key)[2 if kind == 'pair' else 1]
        self.closes[id(trace)] = self.closes.get(id(trace), 0) + 1
        try:
            trace.close()
        finally:
            for device in members:
                unwrap_view(device)
        return True

    def retire_member(self, device):
        """Retire every pair and quad `device` is in, before it closes or unparks. Returns the number of traces closed."""
        closed = 0
        for key in [key for key, entry in self.quad_blocks.items() if any(member is device for member in entry[0])]:
            closed += int(self.retire('quad', key))
        for key in [key for key, entry in self.pairs.items() if entry[0] is device or entry[1] is device]:
            closed += int(self.retire('pair', key))
        return closed

    def retire_all(self):
        closed = 0
        for key in list(self.quad_blocks):
            closed += int(self.retire('quad', key))
        for key in list(self.pairs):
            closed += int(self.retire('pair', key))
        return closed

    def drop_views(self):
        """Hook close: every member device wears no view of a trace (the traces stay)."""
        for entry in list(self.pairs.values()):
            for device in entry[:2]:
                unwrap_view(device)
        for entry in list(self.quad_blocks.values()):
            for device in entry[0]:
                unwrap_view(device)

    def bytes_estimate(self):
        import serving_prefill_admission as admission

        return (len(self.pairs) * admission.measured_pair_capture_bytes()
                + len(self.quad_blocks) * admission.measured_quad_capture_bytes())

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.retire_all()


def unwrap_view(device):
    """The device's proposal capture without a pair's or quad's _PackedCaptureView (a view never owns the shared trace)."""
    from dflash_packed_proposal_coordinator import _PackedCaptureView

    capture = getattr(device, 'proposal_capture', None)
    if isinstance(capture, _PackedCaptureView):
        device.proposal_capture = capture._original


class ParkedEngineSet:
    """Parked engines, one per pool slot, built at attach after the packed blocks and closed before them (the attach registers it after
    them).

    build(): for each slot, while the DRAM split holds an engine build plus a parked arrival (dram_short): the slot re-zeroed and native
    GDN slot 0 restored from its zeroed carry, then today's device, proposal and engine build on a synthetic request - P_cap = 1, an
    all-page-0 table, zero taps, budget 16 - pinned to that slot (pool.slot_order: at eight seats the pool places by block occupancy, and a
    parked slot is always lent) and a park. Slot 0 also warms the drafter at its steady state. A slot the split cannot hold stays unparked, as
    does one whose park fails later (unpark): today's per-request build serves it until repark_idle (or a detach) rebuilds it.

    place() decides which slot the next request takes - the pool's own rule (block occupancy at eight seats, the lowest free slot at four)
    over SERVING slots, so the same arrivals get the same slots, and segments, with the flag on or off (R5). take() gives out a parked one,
    rebind_slot() binds it to its request, park() parks it again - or unparks it - at the request's close. The request's close parks in two
    halves, park_engine and park_drafter, so an engine that cannot park leaves its device to be closed with it; both are idempotent.

    The worker hook's detach then retires the device's pair and quad traces (E1: after_park -> the coordinator's release_parked) and rebuilds
    its released single when the split allows (rebuild_single); the lifecycle's idle moment does the same for every parked slot and re-parks
    the unparked ones (idle). arrival_terms are the parked terms of the slot the next request will take, asked by the scheduler's predicate
    and the backstop; make_room is the release ladder an arrival short of them runs before it is refused."""

    def __init__(self, *, operations, model, sampler, helpers, pool, weights, fixtures, collectives, blocks,
                 capture_rows, components=None, environ=None, log=None):
        import verifier_engine
        import tp_shapes

        environ = os.environ if environ is None else environ
        self.window = project_rows(environ)
        self.audit = audit_enabled(environ)
        self.drafts = drafts_enabled(environ)
        self.negative = negative_mode(environ)
        self.fault, self.faulted = fault_mode(environ), False
        self.ballast_size = ballast_bytes(environ)
        self.ballast = None
        if tp_shapes.chip_count() == tp_shapes.PAIR:
            raise ValueError('%s=1 on this line serves the four-card mesh only (QWEN_FAST_TP=4); the pair\'s Stage E is the other line\'s' % FLAG)
        if environ.get('QWEN_FAST_SHARED_CCL', '1') != '1' or environ.get('QWEN_FAST_EAGER_PROPOSAL') == '1':
            raise ValueError('%s=1 keeps one device per slot for the process: it needs the shared collectives '
                             '(QWEN_FAST_SHARED_CCL=1) and captured proposals (QWEN_FAST_EAGER_PROPOSAL unset)' % FLAG)
        if capture_rows != 4:
            raise ValueError('%s=1 parks engines with the sequential captures (1, 2, 4) beside the packed blocks; this attach caps them at %r'
                             % (FLAG, capture_rows))
        blocks = tuple(blocks)
        if not blocks or any(getattr(block, 'carries_in_place', False) is not True for block in blocks):
            raise ValueError('%s=1 needs every packed block to read its carries in place (carries_in_place, QWEN_FAST_VERIFY_T1 #3 in every '
                             'layer): a padded round writes an idle segment\'s carry and native slot 0 otherwise, and parked slots are '
                             'always lent; blocks %r' % (FLAG, [getattr(block, 'carries_in_place', None) for block in blocks]))
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
        self.unparks = self.reparks = self.fallbacks = 0
        self.attach_ms = None
        self.off = False
        self.off_file = environ.get(OFF_PATH_ENV, OFF_PATH) or None
        self.clock = time.monotonic
        self.polled = float('-inf')
        self.book = None
        self.unregister_book = None
        self.closed = False
        self.replays = None
        if self.drafts:
            self.book = DraftTraceBook(log=self.log)
            import dflash_packed_proposal_coordinator as coordinator

            self.unregister_book = coordinator.register_draft_book(self.book)

    # -- building ------------------------------------------------------------------------------------------------------------------
    def reserve(self):
        from dflash_packed_proposal_coordinator import dram_reserve_bytes

        return dram_reserve_bytes(self.environ)

    def dram_short(self):
        """THE SPLIT's terms (serving_prefill_admission.split_short) the pool's reading is short of for one more engine build beside a parked
        arrival at the longest prompt; None when it holds or cannot be read (the attach refuses a pool without statistics)."""
        import serving_prefill_admission as admission

        reserve = self.reserve()
        reading, reason = admission.dram_reading(self.pool)
        if reading is None:
            return None
        need = admission.engine_build_peak() + parked_arrival_need(reserve, self.window)
        short = admission.split_short(reading['free'], reading['largest_free'], need, reserve, reading['trace_largest_free'],
                                      contiguous=admission.admission_contiguous_need(admission.PREFILL_TRANSIENT_FROM, reserve))
        return (short, reading, need) if short else None

    def ledger(self):
        """The audit's replay ledger when it is installed (verifier_engine_tp's counter is its bound read), else None."""
        import verifier_engine_tp

        counter = verifier_engine_tp._replay_count
        return getattr(counter, '__self__', None) if counter is not None else None

    def build(self):
        """Park an engine on every slot the DRAM split holds, in slot order; returns how many."""
        import memory_ledger
        import verifier_engine

        started = time.perf_counter()
        built = 0
        ledger = self.ledger()
        mark = None if ledger is None else ledger.mark()
        for entry in self.slots:
            stop = self.dram_short()
            if stop is not None:
                short, reading, need = stop
                self.log(STOPPED_MARKER, entry.index, len(self.slots), '+'.join(short), reading['free'], reading['largest_free'], need)
                break
            self.build_slot(entry, warm=entry.index == 0)
            built += 1
        verifier_engine.note_prefill()
        if ledger is not None:
            # The attach ordering the S8 hang class leans on (the audit arm only: the ledger that sees them is the audit's): the builds replay
            # nothing but the single proposal trace each has just captured. A replay of a block's trace here is a block that ran before
            # the engines were built.
            foreign = ledger.foreign_since(mark)
            if foreign:
                raise AssertionError('The parked engine builds replayed %d trace(s) they had not captured: the attach ordering (all builds '
                                     'before any block replay) does not hold' % len(foreign))
        if self.fault:
            self.log(NEGATIVE_MARKER + 'fault={} (gate only: the first {} is refused once)', self.fault, self.fault)
        if self.negative:
            self.log(NEGATIVE_MARKER + 'mode={} (gate only: every rebind is deliberately broken)', self.negative)
        self.attach_ms = (time.perf_counter() - started) * 1000
        self.log(BUILT_MARKER + 'k={} of {} attach_ms={:.1f} {}', built, len(self.slots), self.attach_ms, self.dram_text())
        memory_ledger.record('P7p', point='parked k=%d' % built, parked_engines=self)
        if self.ballast_size:
            # The ballast, after the P7p reading, so the baseline it records is the parked set's own.
            self.ballast = DramBallast(self.operations, self.model.mesh_device, self.ballast_size)
            self.log(BALLAST_MARKER + 'bytes={} buffers={} (gate only; unread) {}', self.ballast.size, len(self.ballast.tensors),
                     self.dram_text())
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
        """The synthetic build on one unlent slot, parked (the class docstring). Pinned to the slot: at eight seats the pool places by block
        occupancy, which would lend another free slot to the second build of a block."""
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
                with self.pool.slot_order((entry.index,)):
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
            write_null_tables(engine)
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
        entry.device_keys, entry.engine_keys = frozenset(vars(device)), frozenset(vars(engine))

    def warm_drafter(self, device, engine):
        """Slot 0's drafter at its steady state: a rebind at 2048 compiles the 2048-row projection and K/V seed, and publish_prewarm
        (QWEN_FAST_PUBLISH_PREWARM=1) then finds history_rows == 2048 and warms the publication programs - which no rebind runs, so the first
        request after a restart pays neither."""
        started = time.perf_counter()
        taps = zero_taps(self.operations, self.model.mesh_device, HISTORY_ROWS)
        try:
            rebind_device(device, taps, position=HISTORY_ROWS, window=self.window, **(dict(audit=True) if self.audit else {}))
        finally:
            for value in taps:
                self.operations.deallocate(value)
        warmed = ()
        if self.environ.get('QWEN_FAST_PUBLISH_PREWARM') == '1':
            import publish_prewarm

            warmed = publish_prewarm.warm(device, engine)
        self.log(WARM_MARKER + 'P={} ms={:.1f} window={} prewarm_pairs={}', HISTORY_ROWS, (time.perf_counter() - started) * 1000,
                 self.window, len(warmed))

    # -- placement and serving -----------------------------------------------------------------------------------------------------
    def live(self, index):
        """Whether slot `index` holds a live request: serving, or lent to a cold-built device. A parked slot is lent and not live."""
        entry = self.slots[index]
        if entry.state == 'serving':
            return True
        if entry.state == 'parked':
            return False
        return bool(entry.slot.lent)

    def place(self):
        """The slot entry the next request takes - the pool's own rule over serving slots (R5) - or None when the set no longer places (off, closed)
        or every slot is taken. The entry may be parked (rebind), or unparked and free (cold build pinned to it)."""
        if self.closed or self.off:
            return None
        if getattr(self.pool, '_blocks', None) is not None:
            slot = self.pool.placement_slot(live=self.live)
            return None if slot is None else self.slots[slot.index]
        for entry in self.slots:
            if not self.live(entry.index):
                return entry
        return None

    def peek(self):
        """The parked entry take() would give the next request, without taking it, or None when the placed slot holds no parked engine
        (today's build takes it)."""
        entry = self.place()
        return entry if entry is not None and entry.state == 'parked' else None

    def take(self):
        """The next request's slot (the class docstring): its entry, now 'serving', when it is parked; else None."""
        if self.closed:
            raise ValueError('The parked engines are closed')
        entry = self.peek()
        if entry is not None:
            entry.state = 'serving'
        return entry

    def parked_count(self):
        """How many slots hold a parked engine right now (the governor's per-pending charge reads it: those arrivals rebind)."""
        return 0 if self.closed or self.off else sum(1 for entry in self.slots if entry.state == 'parked')

    def single_released(self, entry):
        """Whether the slot's device has no single-user proposal capture: a pair or the quad released it, and neither a park nor a rebind has
        rebuilt it yet. Never under 2c (singles are kept)."""
        return entry.device is not None and single_capture(entry.device) is None

    def arrival_rebind_bytes(self):
        return rebind_peak_bytes(self.window)

    def slot_terms(self, entry):
        """THE PARKED TERMS of a request on `entry` (serving_prefill_admission): dict(rebind=R, single=S when its single was released, else 0,
        credit=what the release ladder could free for it)."""
        return dict(rebind=self.arrival_rebind_bytes(), single=single_capture_bytes() if self.single_released(entry) else 0,
                    credit=self.ladder_credit(entry))

    def arrival_terms(self):
        """The terms the next request is admitted and backstopped on (serving_prefill_admission.dram_predicate,
        serving_request_factory.dram_backstop): None when the slot it will take is served by today's per-request build (today's terms), else
        slot_terms of its parked slot. Read at each call, so it follows every park, unpark and rebuild."""
        entry = self.peek()
        return None if entry is None else self.slot_terms(entry)

    def rebind_refusal(self, entry, *, position, budget, pages_shape):
        """Why `entry`'s parked engine cannot be rebound to this request, or None - HOST ONLY, before take() and before any device write
        (design 5.9): the engine's own request-geometry checks, the device being open with its pooled cache, the single being present or
        legitimately released, and the injected rebind fault (once)."""
        if entry.state != 'parked' or entry.engine is None or entry.device is None:
            return 'the slot holds no parked engine'
        if self.fault == 'rebind' and not self.faulted:
            self.faulted = True
            return REBIND_FAULT_REASON
        reason = entry.engine.rebind_refusal(position, budget, tuple(pages_shape))
        if reason is not None:
            return reason
        device = entry.device
        if device.closed or device.pool_slot is None or device.kv_history is None or not device.cache_history:
            return 'the parked device is not an open pooled device with a committed K/V cache'
        if single_capture(device) is None and not getattr(device, '_packed_capture_released', False):
            return 'the parked device keeps no single-user proposal capture'
        return None

    def refuse(self, entry, reason, request_id=None):
        """A host refusal of a rebind: unpark the slot (closing its engine and device, returning the slot) so today's build runs on it, and
        count the fallback. The request is served."""
        self.log(REFUSED_MARKER + 'req={} slot={} reason={} fallback=build', str(request_id)[:48], entry.index, str(reason)[:160])
        self.fallbacks += 1
        self.unpark(entry, 'rebind refused: %s' % str(reason)[:120])

    def rebind_slot(self, entry, taps, pages, *, position, make_session, request_id=None, budget=None):
        """Bind a taken slot to its request (design 3.4): the device from the prefill's taps, the drafter runtime over it, the session
        make_session(runtime) returns, and the engine over the session and the request's page table. The caller has run the request's host
        checks, asked rebind_refusal and adopted its prefill slot. A failure fences, UNPARKS the slot, and raises RebindFailed, so the caller
        can cold-build on the same slot over the same prefill capture; a fence that itself raises is engine-fatal and propagates as it is."""
        if entry.state != 'serving' or entry not in self.slots:
            raise ValueError('Only a slot this set gave out can be rebound')
        started = time.perf_counter()
        engine, device = entry.engine, entry.device
        programs = self.program_count()
        slot0 = None
        try:
            if self.audit:
                entry.slot.verify()
                if self.negative != 'carry':
                    slot0 = slot_zero_digests(engine)
            info = rebind_device(device, taps, position=position, window=self.window,
                                 **(dict(negative=self.negative) if self.negative == 'drafter' else {}),
                                 **(dict(audit=True) if self.audit else {}))
            runtime = self.components.runtime(device, position=position)
            session = make_session(runtime)
            if self.negative == 'carry':
                # The gate's negative control: this rebind seeds no carry (an instance attribute shadows the method for this call only), so
                # the engine's first verify restores the slot's zeroed carry.
                engine.save_carry = lambda: None
            if self.negative == 'pages':
                engine.page_table_writer = lambda *arguments: None
            if self.negative == 'widths':
                engine.keep_captured_widths = True
            try:
                engine.rebind(session, pages)
            finally:
                for name in ('save_carry', 'page_table_writer', 'keep_captured_widths'):
                    engine.__dict__.pop(name, None)
            if self.audit and self.negative != 'carry':
                audit_rebound_state(engine, entry.index, self.log, slot0_before=slot0,
                                    zeroed=info['zeroed'] if self.negative != 'drafter' else None)
        except Exception as failure:
            reason = 'rebind failed: %s: %s' % (type(failure).__name__, str(failure)[:120])
            try:
                self.operations.synchronize_device(self.model.mesh_device)
            except BaseException:
                # No fence, no fallback: the device state is unknown, as after a failed build.
                self.unpark(entry, reason)
                raise failure
            self.unpark(entry, reason)
            raise RebindFailed(entry.index, reason, failure) from failure
        except BaseException as failure:
            self.unpark(entry, 'rebind failed: %s' % type(failure).__name__)
            raise
        entry.rebinds += 1
        after = self.program_count()
        if programs is not None and after is not None:
            self.log(PROGRAMS_MARKER + 'rebind slot={} programs={}->{}', entry.index, programs, after)
        self.log(REBIND_MARKER + 'req={} slot={} gen={} P={} budget={} ms={:.1f} single_rebuilt={} window={} widths={} capacity={}',
                 str(request_id if request_id is not None else session.request_id)[:48], entry.index, device.rebind_generation,
                 position, budget if budget is not None else session.max_new_tokens, (time.perf_counter() - started) * 1000,
                 int(info['single_rebuilt']), info['window'], ','.join(str(rows) for rows in engine.widths),
                 pages.shape[1] * 64)
        return SimpleNamespace(device=device, engine=engine, runtime=runtime, session=session, **info)

    def program_count(self):
        """The mesh's program-cache entry count (serving_runtime.program_count), or None where the mesh does not report it: the S8 tripwire's
        reading, logged per rebind (a rebind that compiles a program after the block's first replay is the hang class)."""
        count = getattr(getattr(self.model, 'mesh_device', None), 'num_program_cache_entries', None)
        try:
            return int(count()) if callable(count) else None
        except Exception:
            return None

    def park(self, entry, *, reason=None):
        """The request on a slot finished: park its engine (which fences first) and its device again; or, when either cannot park or `reason`
        says the request left it unfit (a failed page binding), unpark the slot. Returns None when parked, else why it was unparked."""
        if reason is not None:
            entry.unfit = reason
        self.park_engine(entry)
        return self.park_drafter(entry)

    def park_engine(self, entry):
        """FastRequest's release_engine for a rebound slot, where close() called engine.close(): the engine parks - fencing first, as close()
        does - unless the request left the slot unfit; why it cannot is kept for park_drafter, which closes it with the device. An engine with a
        block in flight raises as close() raises, and the slot stays serving, as today's request stays open. IDEMPOTENT: a slot that is no
        longer serving (parked already, or unparked by a failed first attempt) is left as it is, so a retried close never raises over it."""
        if entry not in self.slots:
            raise ValueError('Only a slot of this set can park')
        if entry.state != 'serving':
            return
        if self.fault == 'park' and not self.faulted and entry.unfit is None:
            # The injected fault, once: the engine is left as it is (unparked, closed by park_drafter).
            self.faulted = True
            entry.refusal = FAULT_REASON
            return
        if entry.engine is None or entry.engine.phase == 'parked':
            return
        entry.refusal = entry.unfit if entry.unfit is not None else entry.engine.park()

    def park_drafter(self, entry):
        """FastRequest's release_drafter for a rebound slot, after park_engine: the device parks (park_device: a fence, its pending proposal
        dropped, its pooled slot verified), its page tables nulled and its instance state checked, and the slot is parked; or, when the engine
        or the device cannot park - or the park itself raises - the slot is unparked: both closed as today's close closes them, the slot back
        to the pool, until a detach or an idle moment re-parks it. Returns None when parked, else why it was unparked. Idempotent past the
        first outcome."""
        if entry not in self.slots:
            raise ValueError('Only a slot of this set can park')
        if entry.state != 'serving':
            return None
        reason, entry.refusal, entry.unfit = entry.refusal, None, None
        if reason is None and self.off:
            reason = 'the kill switch is latched'
        try:
            if reason is None:
                reason = park_device(entry.device)
            if reason is None:
                if entry.engine.phase != 'parked':
                    reason = 'engine phase %s, not parked' % entry.engine.phase
            if reason is None:
                write_null_tables(entry.engine)
                reason = (instance_problems(entry.device, entry.device_keys, 'device')
                          or instance_problems(entry.engine, entry.engine_keys, 'engine'))
                if reason is not None:
                    self.log(INSTANCE_MARKER, entry.index, reason)
            if reason is None and self.audit:
                reason = null_tables_problem(entry.engine)
        except BaseException as failure:
            self.unpark(entry, 'park failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
            raise
        if reason is None:
            entry.state = 'parked'
            entry.parks += 1
            return None
        self.unpark(entry, reason)
        return reason

    # -- after a park --------------------------------------------------------------------------------------------------------------
    def after_park(self, entry, coordinator=None):
        """The worker hook's release when a request on `entry` detaches, after its close parked the slot (release_parked, from
        FastWorkerHook.detach). E1: the device's pair and quad traces retired now by the coordinator's release_parked - a parked device never
        closes, so release_closed never sees them dead - then its released single rebuilt when the split allows. 2c: nothing is retired (the
        traces are the book's, bound to the slot), the single is never released. Then ONE unparked free slot is re-parked when the split
        allows, so a faulted slot does not stay on cold builds until a whole-server idle moment. A failed retirement is logged and left to
        the coordinator's own checks; a failed single rebuild leaves it released with its S in the slot's terms, as idle() does. Returns
        dict(released=, single=, reparked=) or None."""
        if self.closed or entry.state != 'parked' or entry.device is None:
            return None
        released = None
        if coordinator is not None and self.book is None:
            try:
                released = coordinator.release_parked(entry.device)
            except Exception as failure:
                self.log('[PACKED-PROPOSE] release parked slot={} failed ({}: {}); a rebound device\'s generation retires them',
                         entry.index, type(failure).__name__, str(failure)[:160])
        single = None
        try:
            single = self.rebuild_single(entry, 'park')
        except Exception as failure:
            # The park already succeeded; the single stays released and the slot's terms carry S until its next rebind rebuilds it.
            entry.device.proposal_capture = None if single_capture(entry.device) is None else entry.device.proposal_capture
            self.log(IDLE_FAILED_MARKER, entry.index, 'single rebuild at park', '%s: %s' % (type(failure).__name__, failure))
        return dict(released=released, single=single, reparked=self.repark_one())

    def single_short(self):
        """THE SPLIT's terms a single-capture rebuild now would leave short for the longest parked arrival - its need counted with S, which
        the rebuild takes, and its trace region with the single's trace; None when it holds or the pool cannot be read."""
        import serving_prefill_admission as admission

        reserve = self.reserve()
        reading, reason = admission.dram_reading(self.pool)
        if reading is None:
            return None
        need = parked_arrival_need(reserve, self.window, single_released=True)
        short = admission.split_short(reading['free'], reading['largest_free'], need, reserve, reading['trace_largest_free'],
                                      contiguous=admission.admission_contiguous_need(admission.PREFILL_TRANSIENT_FROM, reserve),
                                      trace_need=admission.parked_trace_need(True))
        return (short, reading, need) if short else None

    def rebuild_single(self, entry, moment):
        """A parked slot's released single-user capture rebuilt now, under the single bucket (build_single_capture), iff the split still admits
        the longest parked arrival after it (single_short); else kept released, and the slot's arrival terms carry S until its next rebind
        rebuilds it. True when rebuilt, False when kept, None when there was nothing to rebuild."""
        if entry.state != 'parked' or not self.single_released(entry):
            return None
        stop = self.single_short()
        if stop is not None:
            short, reading, need = stop
            self.log(SINGLE_KEPT_MARKER, entry.index, moment, '+'.join(short), reading['free'], reading['largest_free'], need)
            return False
        import memory_ledger

        started = time.perf_counter()
        device = entry.device
        used_before, free_before = footprint(self.pool)
        token = memory_ledger.before('single', estimate=single_capture_bytes(), point='slot=%d at=%s' % (entry.index, moment))
        try:
            device.proposal_capture = None
            device.proposal_capture = build_single_capture(device)
            device._packed_capture_released = False
        finally:
            memory_ledger.after(token)
        used_after, free_after = footprint(self.pool)
        self.log(SINGLE_REBUILT_MARKER, entry.index, moment, (time.perf_counter() - started) * 1000,
                 'n/a' if None in (used_before, used_after) else used_after - used_before,
                 'n/a' if None in (free_before, free_after) else free_before - free_after)
        return True

    def idle(self):
        """The lifecycle's idle moment (no decoder, no prefill in flight, a step that schedules nothing): every parked slot's released single
        rebuilt while the split allows, then every unparked slot the pool has not lent re-parked (repark_idle). Returns
        dict(singles=[slots rebuilt], reparked=[slots])."""
        if self.closed or self.off:
            return dict(singles=[], reparked=[])
        singles = []
        for entry in self.slots:
            try:
                if self.rebuild_single(entry, 'idle'):
                    singles.append(entry.index)
            except Exception as failure:
                # Optional recovery work: a failed capture at an idle moment must not take an idle, healthy server down. The single stays
                # released (the slot's next rebind rebuilds it, S in its arrival terms).
                entry.device.proposal_capture = None if single_capture(entry.device) is None else entry.device.proposal_capture
                self.log(IDLE_FAILED_MARKER, entry.index, 'single rebuild', '%s: %s' % (type(failure).__name__, failure))
        return dict(singles=singles, reparked=self.repark_idle())

    def unpark(self, entry, reason):
        """Close the slot's engine and device as today's request close does - which returns the slot and the weights - and leave it to
        today's per-request builds until it is re-parked. Under 2c the device's pair and quad traces are retired through the book FIRST: a
        trace must not outlive a member that closes."""
        engine, device = entry.engine, entry.device
        entry.engine = entry.device = None
        entry.state = 'unparked'
        self.unparks += 1
        self.log(UNPARKED_MARKER, entry.index, reason)
        try:
            if device is not None and self.book is not None:
                self.book.retire_member(device)
            if engine is not None and engine.phase != 'closed':
                if engine.phase in ('verifying', 'verified', 'committing'):
                    engine.phase = 'failed'
                engine.close()
        finally:
            if device is not None and not device.closed:
                device.close()

    def repark_idle(self):
        """At an idle moment (the caller's: no decoder and no prefill in flight), rebuild every unparked slot the pool has not lent, lowest
        first, while the DRAM split holds. Returns the slots re-parked."""
        import verifier_engine

        reparked = []
        if self.off or self.closed:
            return reparked
        for entry in self.slots:
            if not self.repark_candidate(entry):
                continue
            if not self.repark(entry):
                break
            reparked.append(entry.index)
        if reparked:
            verifier_engine.note_prefill()
        return reparked

    def repark_candidate(self, entry):
        return entry.state == 'unparked' and not entry.slot.lent

    def repark_one(self):
        """One unparked free slot re-parked now (at a detach), when the split allows; the slot's index, or None."""
        import verifier_engine

        if self.off or self.closed:
            return None
        for entry in self.slots:
            if self.repark_candidate(entry):
                if self.repark(entry):
                    verifier_engine.note_prefill()
                    return entry.index
                return None
        return None

    def repark(self, entry):
        """Build the synthetic engine on an unparked free slot and park it; False when the DRAM split is short or the build failed (the slot
        stays unparked, to be tried at the next moment)."""
        stop = self.dram_short()
        if stop is not None:
            short, reading, need = stop
            self.log(STOPPED_MARKER, entry.index, len(self.slots), '+'.join(short), reading['free'], reading['largest_free'], need)
            return False
        started = time.perf_counter()
        try:
            self.build_slot(entry, warm=False)
        except Exception as failure:
            self.log(IDLE_FAILED_MARKER, entry.index, 're-park', '%s: %s' % (type(failure).__name__, failure))
            return False
        self.reparks += 1
        self.log(REPARKED_MARKER, entry.index, (time.perf_counter() - started) * 1000)
        return True

    # -- the kill switch -----------------------------------------------------------------------------------------------------------
    def watch(self, path, now=time.monotonic):
        """The file whose presence latches the switch (default OFF_PATH, beside prefix-reuse.off and levern.off); None disables it."""
        self.off_file = path
        self.clock = now

    def poll_off(self):
        """The lifecycle calls this at the TOP of every execute (once a second at most, latched): when parked.off exists the set switches off - every
        parked idle slot unparks now (fence, close), every serving slot unparks at its close, arrival_terms and place answer None, re-parking
        stops, and the book is retired with its devices. The latch is applied only here, never between a request's backstop and its take()."""
        if self.off or self.closed or self.off_file is None:
            return self.off
        now = self.clock()
        if now - self.polled < 1.0:
            return False
        self.polled = now
        if not os.path.exists(self.off_file):
            return False
        return self.switch_off('file')

    def switch_off(self, why='api'):
        if self.off:
            return True
        self.off = True
        unparked = 0
        for entry in self.slots:
            if entry.state == 'parked':
                self.unpark(entry, 'kill switch (%s)' % why)
                unparked += 1
        if self.book is not None:
            self.book.retire_all()
        self.log(OFF_MARKER, unparked)
        return True

    # -- the release ladder (design 5.7) -------------------------------------------------------------------------------------------
    def ladder_credit(self, target=None):
        """What make_room could free for an arrival on `target` (a parked entry), in bytes per chip, by the measured unit costs: the book's
        pairs, the released-not singles of the other parked slots (2c keeps all of them), and one other idle parked slot's engine and device."""
        import serving_prefill_admission as admission

        credit = len(self.book.pairs) * admission.measured_pair_capture_bytes() if self.book is not None else 0
        others = [entry for entry in self.slots if entry.state == 'parked' and entry is not target]
        credit += sum(single_capture_bytes() for entry in others if not self.single_released(entry))
        if others:
            credit += admission.parked_engine_bytes()
        return credit

    def make_room(self, need=None, target=None):
        """The ladder an arrival short of its parked terms runs before it is refused or held (design 5.7), device-exclusive (the final step):
        rung 1 retires the book's pairs, rung 2 releases the singles of the OTHER parked slots, rung 3 unparks one idle parked slot other
        than `target`. `need` is the bytes short (None: run every rung once, stopping when the split holds). Logs one line per rung that
        freed anything; returns the bytes the readings say were freed."""
        start = peak_reading(self.pool)
        freed_total = 0
        for rung in (1, 2, 3):
            before = peak_reading(self.pool)
            freed = 0
            if rung == 1 and self.book is not None:
                for key in list(self.book.pairs):
                    self.book.retire('pair', key)
                    freed += 1
            elif rung == 2:
                for entry in self.slots:
                    if entry.state == 'parked' and entry is not target and not self.single_released(entry):
                        capture = single_capture(entry.device)
                        if capture is not None:
                            capture.close()
                            entry.device.proposal_capture = None
                            entry.device._packed_capture_released = True
                            freed += 1
            elif rung == 3:
                for entry in reversed(self.slots):
                    if entry.state == 'parked' and entry is not target:
                        self.unpark(entry, 'release ladder rung 3')
                        freed += 1
                        break
            after = peak_reading(self.pool)
            if freed:
                gained = 0 if None in (before, after) else after - before
                freed_total += gained
                self.log(LADDER_MARKER, rung, gained, -1 if target is None else target.index)
            if need is not None and start is not None and after is not None and after - start >= need:
                break
        return freed_total

    # -- reports and close ---------------------------------------------------------------------------------------------------------
    def describe(self):
        return dict(window=self.window, audit=self.audit, drafts=self.drafts, off=self.off, unparks=self.unparks, reparks=self.reparks,
                    fallbacks=self.fallbacks,
                    slots=[dict(index=entry.index, state=entry.state, rebinds=entry.rebinds, parks=entry.parks) for entry in self.slots])

    def close(self):
        """Every engine and device closed - parked, or left serving - which returns every slot and the weights; the book first. The attach
        registers the set after the blocks, so it closes after the lifecycle and before the blocks, the weights and the pool; the pool and the
        weights refuse to close while anything is still lent."""
        if self.closed:
            return
        self.closed = True
        if self.ballast is not None:
            self.ballast.close()
            self.ballast = None
        failures = []
        if self.book is not None:
            try:
                self.book.close()
            except BaseException as failure:
                failures.append(failure)
        if self.unregister_book is not None:
            self.unregister_book()
            self.unregister_book = None
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
        if failures:
            raise failures[0]


def release_parked(coordinator, request):
    """The release at a detach (serving_worker_hook.release_parked, QWEN_FAST_PARKED_ENGINES=1 only): for a request that ran on a parked slot
    (serving_request_factory.rebind_parked sets its `parked_slot`), its set's after_park with the hook's proposal coordinator (None when the hook
    never packed a proposal). None for a request today's per-request build served, whose closed device release_closed already covered."""
    entry = getattr(request, 'parked_slot', None)
    owner = getattr(entry, 'owner', None)
    if owner is None:
        return None
    return owner.after_park(entry, coordinator)
