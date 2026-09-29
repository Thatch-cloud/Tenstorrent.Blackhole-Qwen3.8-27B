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
