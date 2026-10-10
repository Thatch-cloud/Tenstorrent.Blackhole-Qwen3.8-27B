"""Per-call options of the packed verify's tile collectives at four cards (F-C1 of the op-fusion programme, WP5).

WHY. The packed verify issues 128 reduce-scatters and 129 all-gathers a pass (M676: 18.2 and 18.4 us median, 16.5 and 17.4 us on the fastest
chip, 4.70 ms a pass). Both are latency-bound: the ring moves about 120 KB a link and direction at the 64-row block, a few us at line rate, and
the rest is the fixed part of the call (three dependent ring steps, the semaphore round trips, the fabric mux between the worker and the
router, the start barrier). The model calls both ops with fixed keyword values (ccl.py tt_all_reduce: chunks_per_sync=10, num_workers_per_link=2,
num_buffers_per_channel=2; distributed_norm.py: the same three on the gather) that were tuned for other shapes. This module names the values a
call may take instead, and only those that cannot change one output bit.

WHAT CHANGES BYTES, FROM THE PINNED SOURCE (tt-metal 9f9cd4fd, ttnn/cpp/ttnn/operations/experimental/ccl/): OPTIONS below is the whole list. The
reduce-scatter is the ring op on dim 3, so its output bytes are fixed by two things only: the direction each 8-tile chunk travels (the parity
(tiles_read / tile_granularity) % 2 of reduce_scatter_common::chunk_ring_parity, a function of the tile's flat index in the channel and of
nothing else: the source's own comment is "independent of worker distribution or total number of chunks") and the add order of the last step
(ring_reduction.cpp: DST = interm2 + (interm + input), a fixed three-term form). So num_links, num_workers_per_link, chunks_per_sync and
num_buffers_per_channel only move work between cores, links and semaphore increments: exact BY CONSTRUCTION. What does change the bytes: topology
(Linear associates local + forward + backward), compute_kernel_config (fp32_dest_acc_en halves tile_granularity to 4, which moves the parity,
and accumulates in fp32), dim, and a fabric payload below 4096 bytes (it shrinks tile_granularity from 8 to 4 for a bfloat16 tile). The
all-gather copies tiles: nothing it takes can change a bit except reverse_order (the output order) and dim; its topology, links, workers,
chunks, buffers and the via-broadcast program are different routes for the same bytes.

WHAT IS NOT BY CONSTRUCTION, AND IS NOT OFFERED. Both ops drop their start barrier when persistent buffers are passed
(use_barrier_sem = barrier_semaphore.has_value() && !using_persistent_buffers). That is a SYNCHRONISATION change: a neighbour that has entered
the next call may write into a buffer this chip is still reading unless the buffers of adjacent calls are distinct. It cannot be offered on the
reduce-scatter at all: the list is [intermediate, output, penult] (the penult buffer is the third slot, so the output slot cannot be skipped) and
the model deallocates the reduce-scatter's output after the residual add, which would free the persistent buffer under the next call. Passing
barrier_semaphore=None is the same removal with fresh buffers. The sweep probe (optimisation/ttnn-op/ccl_sweep) measures both as probe-only arms so
the owner knows what the barrier costs; no named set here can contain them.

THE FLAG. QWEN_FAST_CCL_OPTIONS=<named set> (default unset: byte-identical, nothing wrapped, nothing logged). A set is a '+'-joined list of
tokens, each an option of one op: rs-l<N> num_links, rs-w<N> num_workers_per_link, rs-c<N> chunks_per_sync, rs-b<N> num_buffers_per_channel and the
same four for the gather as ag-..., plus ag-linear (topology Linear) and ag-bcast (use_broadcast). 'served' is the empty set: the lever is
engaged and every call carries the values it carries today (the A/A control of the plumbing). The set is strict: an unknown token, a value outside
the table's ranges, a repeated option, or a reduce-scatter token without QWEN_FAST_TP4_RS_UNIT_MAJOR=1 (the options ride the unit-major call that
the X1 census covers; the per-tile split calls the model's own tt_all_reduce with its own values) is a ValueError at the first block forward.
QWEN_FAST_CCL_OPTIONS_AUDIT=1 (gate arms only) runs, for the first QWEN_FAST_CCL_OPTIONS_AUDIT_CALLS (default 32, even) calls of each op in every
block forward, the call twice: with the set and with the values the model passes today. The served composition's result is served; both are
cloned into DRAM and compared element for element as int16 bit patterns on every chip after the replay by the audit machinery of
tile_collective_tp (packed_verifier already calls audit_claim, audit_replayed, audit_round and audit_release). The call count is even so the ring's
semaphore cycling keeps the parity request_width_warm.py asks for.

MARKERS (all via the process log): [PINDIAG] tp4 ccl options engaged set=<name> rows=<R> rs=<n> ag=<m> fallbacks=<k> audited=<a> once per block
forward; [PINDIAG] tp4 ccl options fell back rows=<R> op=<rs|ag> reason=<text> once per reason; [PINDIAG] tp4 ccl options audit shape=<RxW>
owner=<label> round=<r> op=<rs|ag> calls=<n> chips=4 elements=<n> exact=True and [PINDIAG] tp4 ccl options audit mismatch ... (an exception).

Stdlib only; ttnn is touched by the callers.
"""

from collections import namedtuple
import re

OPTIONS_FLAG = 'QWEN_FAST_CCL_OPTIONS'
AUDIT_FLAG = 'QWEN_FAST_CCL_OPTIONS_AUDIT'
AUDIT_CALLS_FLAG = 'QWEN_FAST_CCL_OPTIONS_AUDIT_CALLS'
UNIT_MAJOR_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR'
DEFAULT_AUDIT_CALLS = 32
SERVED_SET = 'served'

ENGAGED_MARKER = '[PINDIAG] tp4 ccl options engaged'
FALLBACK_MARKER = '[PINDIAG] tp4 ccl options fell back'
AUDIT_MARKER = '[PINDIAG] tp4 ccl options audit'
AUDIT_MISMATCH_MARKER = '[PINDIAG] tp4 ccl options audit mismatch'

# The pinned runtime this table was read from.
SOURCE_PIN = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'

# What each option does to the output bytes of the op that takes it.
EXACT = 'exact'              # reduce-scatter: cannot change a bit, by the construction the source states
COPY = 'exact-copy'          # all-gather: the same tiles by another route
BYTES = 'changes-bytes'      # changes the output bits (or the tile order)
SYNC = 'sync-removal'        # bit-neutral only if no neighbour races ahead: not by construction
PLACEMENT = 'placement'      # where a buffer or a worker lives: allocator and L1 effects, not offered here
INERT = 'inert'              # ignored on the served path
IDENTITY = 'identity'        # part of what the call is, not a choice
UNPROVEN = 'unproven'        # not read to completion: not offered

Option = namedtuple('Option', 'op name served effect why')

OPTIONS = (
    # --- ttnn.experimental.reduce_scatter_minimal_async, the ring op on dim 3 (the stack: 1 x 4 mesh, Ring, bfloat16) ---
    Option('rs', 'input_tensor', 'the (1, R/32, 32, 5120) unit-major view', IDENTITY, 'the data'),
    Option('rs', 'persistent_output_buffers', None, SYNC,
           'a list [intermediate, output, penult]: drops the start barrier (use_barrier_sem is false with persistent buffers) and the model '
           'deallocates the output after the residual add, so the output slot cannot be a persistent buffer; probe-only'),
    Option('rs', 'dim', 3, BYTES, 'selects the scattered slices'),
    Option('rs', 'multi_device_global_semaphore', 'the model\'s cycled triple', IDENTITY,
           'cycling in pairs keeps adjacent calls on different semaphores; the forward\'s call count must stay even'),
    Option('rs', 'barrier_semaphore', 'the model\'s cycled handle', SYNC,
           'None removes the neighbour handshake at the start of the call; fresh buffers of adjacent calls may reuse one address; probe-only'),
    Option('rs', 'num_links', 2, EXACT, 'worker_id = link * workers + worker splits the channel by tile index; the chunk parity is a function of the tile index'),
    Option('rs', 'memory_config', 'DRAM', PLACEMENT, 'the output; an L1 output moves the residual add\'s read and the L1 high-water mark'),
    Option('rs', 'intermediate_memory_config', 'DRAM', INERT,
           'on Ring with dim != 0 the staging buffer is a chunk-paged interleaved DRAM tensor whatever is passed (reduce_scatter_ring_interm_staging_spec)'),
    Option('rs', 'topology', 'Ring', BYTES, 'Linear adds local, forward, backward in that order and has no chunk parity: another association'),
    Option('rs', 'subdevice_id', None, IDENTITY, 'the stack has one sub-device'),
    Option('rs', 'cluster_axis', None, UNPROVEN, 'selects the semaphore pool and the ring membership; the 1 x 4 mesh case was not read to completion'),
    Option('rs', 'chunks_per_sync', 10, EXACT, 'how many chunks one semaphore increment covers; the data path and the parity do not read it'),
    Option('rs', 'num_workers_per_link', 2, EXACT,
           'workers per direction per link; 1 drops the mux core (USE_WORKER_MUX); the tile ranges are slices of the same channel, parity by global tile index'),
    Option('rs', 'num_buffers_per_channel', 2, EXACT, 'depth of the fabric mux channel buffers'),
    Option('rs', 'compute_kernel_config', None, BYTES,
           'fp32_dest_acc_en makes max_dst 4 and tile_granularity 4 (the parity moves) and accumulates in fp32'),
    # --- ttnn.experimental.all_gather_async (the persistent-buffer overload DistributedNorm.forward calls), dim 3 ---
    Option('ag', 'input_tensor', '(1, 1, R, 1280) per chip', IDENTITY, 'the data'),
    Option('ag', 'persistent_output_buffer', None, SYNC,
           'drops the start barrier like the reduce-scatter\'s; the gathered tensor is the norm\'s input and DistributedNorm does not free it, so '
           'rotating persistent buffers are possible; probe-only until the probe says what the barrier costs'),
    Option('ag', 'dim', 3, BYTES, 'the gathered axis'),
    Option('ag', 'multi_device_global_semaphore', 'the model\'s cycled pair', IDENTITY, 'cycled per call'),
    Option('ag', 'barrier_semaphore', 'the model\'s cycled handle', SYNC, 'None removes the neighbour handshake; probe-only'),
    Option('ag', 'num_links', 2, COPY, 'a copy split over links'),
    Option('ag', 'memory_config', 'the norm\'s sharded input config', PLACEMENT, 'the gathered tensor\'s layout is the norm\'s contract'),
    Option('ag', 'topology', 'Ring', COPY, 'Linear moves the same tiles by the line route'),
    Option('ag', 'subdevice_id', None, IDENTITY, 'the stack has one sub-device'),
    Option('ag', 'cluster_axis', None, UNPROVEN, 'not offered'),
    Option('ag', 'use_optimal_ccl_for_llama', False, INERT, 'only the llama sharded shapes of composite_common::use_all_gather_async_llama_sharded'),
    Option('ag', 'use_broadcast', False, COPY, 'the via-broadcast program (all_gather_via_broadcast_factory) copies the same tiles'),
    Option('ag', 'chunks_per_sync', 10, COPY, 'semaphore cadence'),
    Option('ag', 'num_workers_per_link', 2, COPY, 'work split'),
    Option('ag', 'num_buffers_per_channel', 2, COPY, 'mux buffer depth'),
    Option('ag', 'sub_core_grids', None, PLACEMENT, 'which cores run the workers: collides with the sharded L1 tensors of the norm; not offered'),
    Option('ag', 'reverse_order', False, BYTES, 'reverses the device order of the output'),
    # --- once per process, before the mesh opens (not a per-call option; the probe runs one of each) ---
    Option('process', 'fabric_config', 'FABRIC_1D', EXACT,
           'FABRIC_1D or FABRIC_1D_RING: routing of the same transfers; the add order is the op\'s'),
    Option('process', 'max_packet_payload_size_bytes', 'the runtime default (4352)', EXACT,
           'valid from 4096: a bfloat16 tile page is 2048 bytes and tile_granularity = min(4 * min(4, payload // 2048), 8) is 8 for every payload from '
           '4096 up; below 4096 it is 4 and the chunk parity moves'),
    Option('process', 'fabric_tensix_config', 'DISABLED', PLACEMENT, 'MUX reserves worker cores for the whole run: the compute grid changes; not offered'),
)

# What a token may name and the values the stack accepts. The probe's grids are drawn from the same table.
LETTERS = {'l': 'num_links', 'w': 'num_workers_per_link', 'c': 'chunks_per_sync', 'b': 'num_buffers_per_channel'}
RANGES = {'l': (1, 2), 'w': (1, 4), 'c': (1, 100), 'b': (1, 4)}
AG_FLAGS = {'linear': ('topology', 'Linear'), 'bcast': ('use_broadcast', True)}
_VALUE_TOKEN = re.compile(r'^(rs|ag)-([lwcb])(0|[1-9][0-9]*)$')
_FLAG_TOKEN = re.compile(r'^ag-(linear|bcast)$')

Selection = namedtuple('Selection', 'name rs ag')
Settings = namedtuple('Settings', 'selection audit_calls')


def option_effect(op, name):
    for option in OPTIONS:
        if option.op == op and option.name == name:
            return option.effect
    raise KeyError((op, name))


def offered_options():
    """[(op, option name)] a token may name: every one is EXACT or COPY in the table (the test holds the grammar to the table)."""
    names = [(op, name) for op in ('rs', 'ag') for name in LETTERS.values()]
    names += [('ag', name) for name, _ in AG_FLAGS.values()]
    return names


def parse_set(text):
    """Selection(name, rs, ag) of a named set; ValueError for anything the grammar does not name. `rs` and `ag` map an option name to its value
    (topology as the string 'Linear': the caller resolves it against ttnn). The name is canonical: 'served', or the sorted tokens joined by '+'."""
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError('%s must be a named set, got %r' % (OPTIONS_FLAG, text))
    if text == SERVED_SET:
        return Selection(SERVED_SET, {}, {})
    values = {'rs': {}, 'ag': {}}
    tokens = []
    for token in text.split('+'):
        flag = _FLAG_TOKEN.match(token)
        value = _VALUE_TOKEN.match(token)
        if flag:
            name, setting = AG_FLAGS[flag.group(1)]
            op = 'ag'
        elif value:
            op, letter, number = value.group(1), value.group(2), int(value.group(3))
            low, high = RANGES[letter]
            if not low <= number <= high:
                raise ValueError('%s token %r: %s must be %d to %d' % (OPTIONS_FLAG, token, LETTERS[letter], low, high))
            name, setting = LETTERS[letter], number
        else:
            raise ValueError('%s token %r is not in the grammar (rs|ag)-(l|w|c|b)<N>, ag-linear, ag-bcast or the set "%s"'
                             % (OPTIONS_FLAG, token, SERVED_SET))
        if name in values[op]:
            raise ValueError('%s names %s %s twice' % (OPTIONS_FLAG, op, name))
        values[op][name] = setting
        tokens.append(token)
    return Selection('+'.join(sorted(tokens)), values['rs'], values['ag'])


def _flag(name, source):
    value = source.get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def settings(environ=None):
    """Settings(selection, audit_calls) for this process, None when the lever is off. Strict: four cards (QWEN_FAST_TP=4), the set parses, a
    reduce-scatter token needs the unit-major lever, the audit needs the set, and the audit's call count is a positive even integer."""
    import os

    source = os.environ if environ is None else environ
    text = source.get(OPTIONS_FLAG)
    audit = _flag(AUDIT_FLAG, source)
    calls_text = source.get(AUDIT_CALLS_FLAG)
    if text is None or text == '0':
        if audit or calls_text is not None:
            raise ValueError('%s and %s need %s=<named set>' % (AUDIT_FLAG, AUDIT_CALLS_FLAG, OPTIONS_FLAG))
        return None
    import tp_shapes

    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % OPTIONS_FLAG)
    selection = parse_set(text)
    if selection.rs and not _flag(UNIT_MAJOR_FLAG, source):
        raise ValueError('%s=%s names reduce-scatter options, which ride the unit-major call: they need %s=1'
                         % (OPTIONS_FLAG, text, UNIT_MAJOR_FLAG))
    calls = 0
    if calls_text is not None and not audit:
        raise ValueError('%s needs %s=1' % (AUDIT_CALLS_FLAG, AUDIT_FLAG))
    if audit:
        calls = DEFAULT_AUDIT_CALLS
        if calls_text is not None:
            if not calls_text.isdigit() or str(int(calls_text)) != calls_text or int(calls_text) < 2 or int(calls_text) % 2:
                raise ValueError('%s must be a positive even integer (the ring\'s semaphore parity), got %r' % (AUDIT_CALLS_FLAG, calls_text))
            calls = int(calls_text)
    return Settings(selection, calls)


# --- the scope of one block forward ------------------------------------------------------------------------------------------------------------


class Plan(object):
    """The options in force for one block forward and what this forward did with them (made by begin, read by end)."""

    def __init__(self, selection, audit_calls, rows, layers, log=None):
        self.selection = selection
        self.audit_calls = audit_calls
        self.rows = rows
        self.layers = layers
        self.log = log
        self.rs_engaged = 0
        self.ag_engaged = 0
        self.fallback_counts = {'rs': 0, 'ag': 0}
        self.audited = {'rs': 0, 'ag': 0}

    @property
    def fallbacks(self):
        return sum(self.fallback_counts.values())

    def audit_open(self, op):
        """True while the next call of `op` is inside the audit's quota (and counts it)."""
        if self.audit_calls and self.audited[op] < self.audit_calls:
            self.audited[op] += 1
            return True
        return False


_STATE = {'plan': None, 'reasons': set()}


def current():
    """The Plan of the block forward in progress, None outside one (or with the lever off)."""
    return _STATE['plan']


def begin(configured, rows, layers, log=None):
    """Open the plan of a block forward; `configured` is settings(), `log` the scope's log(format, *values). Returns the Plan (None when off)."""
    if configured is None:
        return None
    if _STATE['plan'] is not None:
        raise ValueError('A ccl options plan is already open (%d rows)' % _STATE['plan'].rows)
    plan = Plan(configured.selection, configured.audit_calls, rows, layers, log)
    _STATE['plan'] = plan
    return plan


def close(plan):
    """Close the plan (always, also after an error)."""
    if plan is not None and _STATE['plan'] is plan:
        _STATE['plan'] = None


def finish(plan):
    """After a block forward that finished: the guard that the gather wrapper was reached by every norm of the stack (2 per layer, and the final
    norm's: a wrapper bound where the layers do not look would serve the model's own calls silently; skipped when the scope was given no layer
    count), then the engaged line."""
    if plan is None:
        return
    gathers = plan.ag_engaged + plan.fallback_counts['ag']
    low = 2 * (plan.layers or 0)
    if plan.layers is not None and (gathers < low or gathers > low + 1):
        raise AssertionError('The ccl options gather wrapper saw %d norm gathers in this %d-row forward, %d to %d expected: the model\'s DistributedNorm '
                             'does not all reach it (distributed_norm_gather_tp.install binds it at QWEN_FAST_TP != 2)'
                             % (gathers, plan.rows, low, low + 1))
    if plan.log is not None:
        plan.log('{}', '%s set=%s rows=%d rs=%d ag=%d fallbacks=%d audited=%d'
                 % (ENGAGED_MARKER, plan.selection.name, plan.rows, plan.rs_engaged, plan.ag_engaged, plan.fallbacks, sum(plan.audited.values())))


def note_fallback(plan, op, reason):
    """Count a call served with the model's own values and say why, once per reason."""
    plan.fallback_counts[op] += 1
    if reason not in _STATE['reasons']:
        _STATE['reasons'].add(reason)
        if plan.log is not None:
            plan.log('{}', '%s rows=%d op=%s reason=%s' % (FALLBACK_MARKER, plan.rows, op, reason))


def rs_overrides(plan=None):
    """The reduce-scatter keyword overrides of the plan in force: {} with no plan or a set without reduce-scatter tokens."""
    plan = _STATE['plan'] if plan is None else plan
    return dict(plan.selection.rs) if plan is not None else {}


def ag_overrides(plan, operations):
    """The gather keyword overrides of the plan, the topology resolved against `operations` (ttnn, or a fake in the tests)."""
    result = {}
    for name, value in plan.selection.ag.items():
        result[name] = getattr(operations.Topology, value) if name == 'topology' else value
    return result
