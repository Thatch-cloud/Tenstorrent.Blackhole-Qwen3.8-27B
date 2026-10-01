"""Diagnostics for the four-card sequential hang, all host-side, all flag-gated and all off by default.

The hang (the audits-off concurrent4 that sat 1510 s) never named a call site: the process was stuck below Python, so no
Python-level watchdog could run and the log stopped at the last '[PHASE]' line. This module is what tells the next one apart.
Nothing here allocates device memory or changes an operation's order, so an arm with these flags on runs the device sequence
of the arm with them off.

QWEN_FAST_SEQ_DEADLINE_S=<seconds>: serving_sequential_step.step_request arms faulthandler.dump_traceback_later around each
  request.step. The timer is a C thread, so it fires while the stalled call holds the GIL: it writes every thread's Python stack
  to stderr (the EngineCore log) and exits the process. It does not unwedge the card; the next job's reset does.
QWEN_FAST_SEQ_STAGE_LOG=1: '[SEQ-STAGE] request= rows= stage= begin|end' before each stage of a sequential step (the step,
  verifier_engine_tp.VerifierEngine.verify's stages, publish, serving_packed_step's publication stages), flushed as it is
  written, so the last line names the stage the process entered and never left. Also one '[PINDIAG] first replay' line per
  captured bucket: what the engine's first trace replay runs after.
QWEN_FAST_CCL_HANDLE_LOG=1: the integer state of every collectives object the serving process cycles (TT_CCL cycles its
  semaphore handles by host-side index; a trace replay does not advance the index) appended to every '[SEQ-STAGE]' line, as
  three groups: shared (the drafter's), model (the target model's, used by the all-reduces inside the verify and packed traces
  and by eager prefill) and sampler (the pinned sampler's gathers). This is the test for the documented hang mode: an eager
  collective reusing a handle a replayed trace baked.
QWEN_FAST_TRACE_CENSUS=1: every trace capture logs its call site, the DRAM and L1 allocator views before and after, and the
  collectives' state before and after; best effort (QWEN_FAST_TRACE_CENSUS_GRAPH=0 turns it off) the address ranges the
  capture allocated and freed again - its temporaries, which a replay rewrites - from a ttnn.graph capture around it.
  census_engine then lists each width's persistent buffer extents per chip and reports every buffer that sits inside the freed
  temporaries of a trace that is still live ('[PINDIAG] trace overlap'): a replay of that trace writes into the buffer. Every
  range here is a per-bank extent (an address is the same offset in each DRAM bank), and a released trace's ranges are dropped.
  The graph part is NOT fail-soft end to end: the begin, end and parse are guarded, but the graph processor's work on each
  captured operation runs inside the capture itself, and an error there fails the trace capture. QWEN_FAST_TRACE_CENSUS_GRAPH=0
  (the c2-packed-tp4-diag-nograph profile) is the fallback that keeps everything else.

install() is a twin of attention_batch.capture_operation, bound by tp_addresses.install() at QWEN_FAST_TP=4 only; attention_batch.py
stays the pair's pinned bytes. Without the flag the twin returns the original's result and makes the original's calls.

Stdlib only at import.
"""

import json
import os
import sys

SEQ_DEADLINE_FLAG = 'QWEN_FAST_SEQ_DEADLINE_S'
STAGE_LOG_FLAG = 'QWEN_FAST_SEQ_STAGE_LOG'
CCL_LOG_FLAG = 'QWEN_FAST_CCL_HANDLE_LOG'
CENSUS_FLAG = 'QWEN_FAST_TRACE_CENSUS'
GRAPH_FLAG = 'QWEN_FAST_TRACE_CENSUS_GRAPH'
ID_WIDTH = 48
LINE_BUDGET = 240
CCL_LINE_BUDGET = 180
MAX_TRACES = 4096
MAX_OVERLAP_LINES = 20

# The collectives objects by name (serving_runtime.note: shared, model, sampler), the capture sequence number, the live traces
# so far and where the current engine's build began in that sequence. PACKED_NOTES counts calls to
# verifier_engine.note_packed_step, which a four-user packed round makes about six times (verify, each commit, the flush and
# the step's finally): it is not a count of rounds.
COLLECTIVES = {}
TRACES = []
SEQUENCE = 0
ENGINE_BOUNDARY = 0
PACKED_NOTES = 0
ENGINE_NOTES = 0
BUILT_AT = {}
UNAVAILABLE_REPORTED = set()


def flag_on(name, environ=None):
    return (os.environ if environ is None else environ).get(name) == '1'


def stage_log_enabled():
    return flag_on(STAGE_LOG_FLAG)


def census_enabled():
    return flag_on(CENSUS_FLAG)


def seq_deadline(environ=None):
    """QWEN_FAST_SEQ_DEADLINE_S as a positive number of seconds, None when unset; anything else is a configuration error."""
    text = (os.environ if environ is None else environ).get(SEQ_DEADLINE_FLAG)
    if text is None or text == '':
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = 0.0
    if not seconds > 0 or seconds == float('inf'):
        raise ValueError('%s must be a positive number of seconds, got %r' % (SEQ_DEADLINE_FLAG, text))
    return seconds


def watching(environ=None):
    return seq_deadline(environ) is not None or stage_log_enabled()


def log(message):
    """One loguru INFO line; plain flushed print where loguru is absent (host tests)."""
    if len(message) > LINE_BUDGET:
        message = message[:LINE_BUDGET - 3] + '...'
    try:
        from loguru import logger
    except ImportError:
        print(message, flush=True)
        return
    logger.info('{}', message)


def request_label(request_id):
    return str(request_id)[:ID_WIDTH]


# --- live stage markers -------------------------------------------------------------------------------------------------


def register_collectives(collectives, name='shared'):
    """Name a collectives object for the logs. One object registered under two names stays under the first."""
    if collectives is None or any(known is collectives for known in COLLECTIVES.values()):
        return
    COLLECTIVES[name] = collectives


def collective_state(collectives):
    """The integer attributes (and integer lists) of one collectives object, one compact string, or '' when it has none
    readable."""
    if collectives is None:
        return ''
    try:
        attributes = vars(collectives)
    except TypeError:
        return 'opaque'
    parts = []
    for name, value in attributes.items():
        if type(value) is int:
            parts.append('%s=%d' % (name, value))
        elif isinstance(value, (list, tuple)) and value and all(type(item) is int for item in value):
            parts.append('%s=%s' % (name, ','.join(str(item) for item in value)))
    return ' '.join(parts) or 'no-integer-attributes'


def collective_groups():
    """[(name, state)] for every registered collectives object that has a readable state."""
    groups = [(name, collective_state(collectives)) for name, collectives in COLLECTIVES.items()]
    return [(name, state) for name, state in groups if state]


def groups_text(groups):
    return ' '.join('%s{%s}' % group for group in groups)


def stage(request_id, rows, name, edge='begin', extra=''):
    """'[SEQ-STAGE]' line, flushed as it is written, when QWEN_FAST_SEQ_STAGE_LOG=1; a no-op otherwise. Under
    QWEN_FAST_CCL_HANDLE_LOG=1 the collectives' state rides the line, or, where that would pass CCL_LINE_BUDGET, follows it on
    one line per object (the same request, rows and stage)."""
    if not stage_log_enabled():
        return
    base = '[SEQ-STAGE] request=%s rows=%s%s stage=%s %s' % (request_label(request_id), rows, extra, name, edge)
    groups = collective_groups() if flag_on(CCL_LOG_FLAG) else []
    if not groups:
        return log(base)
    whole = '%s ccl{%s}' % (base, groups_text(groups))
    if len(whole) <= CCL_LINE_BUDGET:
        return log(whole)
    log(base)
    for group in groups:
        log('%s ccl{%s}' % (base, groups_text([group])))


def runtime_context(runtime):
    """(request id, rows) of the transaction a runtime is publishing, 'n/a' where it has none."""
    session = getattr(runtime, 'session', None)
    pending = getattr(session, 'pending', None)
    tokens = getattr(pending, 'tokens', None)
    return (getattr(session, 'request_id', 'n/a'), 'n/a' if tokens is None else len(tokens))


def program_cache_entries(mesh):
    count = getattr(mesh, 'num_program_cache_entries', None)
    if not callable(count):
        return 'n/a'
    try:
        return int(count())
    except Exception:
        return 'n/a'


def note_packed_step_called():
    global PACKED_NOTES
    PACKED_NOTES += 1


def first_replay(engine, request_id, rows):
    """One line when a bucket replays its captured trace for the first time (QWEN_FAST_SEQ_STAGE_LOG=1): how many packed-step
    notes (about six per four-user round, PACKED_NOTES) were made since this engine was built, and the program cache's size."""
    if not stage_log_enabled():
        return
    built = BUILT_AT.get(id(engine))
    log('[PINDIAG] first replay request=%s rows=%s packed_notes_at_build=%s packed_notes_since_build=%s program_cache=%s'
        % (request_label(request_id), rows, 'n/a' if built is None else built,
           'n/a' if built is None else PACKED_NOTES - built, program_cache_entries(getattr(engine, 'mesh', None))))


def watched_step(request_id, rows, call):
    """`call()` between its '[SEQ-STAGE] ... stage=step' lines and, under QWEN_FAST_SEQ_DEADLINE_S, a faulthandler timer
    that dumps every thread's stack and exits the process if the step has not returned in time. The timer is cancelled
    whatever the step does. A stderr faulthandler cannot write to is logged and the step runs unwatched."""
    deadline = seq_deadline()
    stage(request_id, rows, 'step', 'begin')
    armed = False
    if deadline is not None:
        try:
            import faulthandler

            faulthandler.dump_traceback_later(deadline, exit=True, file=sys.stderr)
            armed = True
        except Exception as failure:
            log('[PINDIAG] seq watchdog unavailable %s: %s' % (type(failure).__name__, str(failure)[:80]))
    try:
        result = call()
    finally:
        if armed:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()
    stage(request_id, rows, 'step', 'end')
    return result


# --- per-capture census -----------------------------------------------------------------------------------------------


def unavailable(why):
    """'[PINDIAG] trace census unavailable <why>', once per distinct reason."""
    why = why[:120]
    if why not in UNAVAILABLE_REPORTED:
        UNAVAILABLE_REPORTED.add(why)
        log('[PINDIAG] trace census unavailable %s' % why)


def caller_site(depth):
    try:
        frame = sys._getframe(depth)
        return '%s.%s:%d' % (frame.f_globals.get('__name__', '?'), frame.f_code.co_name, frame.f_lineno)
    except Exception:
        return 'unknown'


def memory_views(operations, mesh):
    """Per chip, the DRAM and L1 allocator figures in bytes over all banks, or dict(unavailable=why)."""
    try:
        import memory_ledger

        devices = mesh.get_devices() if callable(getattr(mesh, 'get_devices', None)) else [mesh]
        report = []
        for chip, device in enumerate(devices):
            report.append(dict(chip=chip,
                               dram=memory_ledger.device_view(operations, device, operations.BufferType.DRAM),
                               l1=memory_ledger.device_view(operations, device, operations.BufferType.L1)))
        return report
    except BaseException as failure:
        return dict(unavailable='%s: %s' % (type(failure).__name__, str(failure)[:80]))


def _mb(value):
    return '%.1f' % (value / 1e6)


def view_text(before, after):
    """One line per chip: allocated and largest free MB, DRAM then L1, before -> after."""
    if isinstance(before, dict) or isinstance(after, dict):
        return ['views unavailable (%s)' % (before if isinstance(before, dict) else after).get('unavailable')]
    lines = []
    for first, second in zip(before, after):
        parts = []
        for kind in ('dram', 'l1'):
            parts.append('%s allocated=%s->%s largest_free=%s->%s' % (
                kind, _mb(first[kind]['allocated']), _mb(second[kind]['allocated']),
                _mb(first[kind]['largest_free']), _mb(second[kind]['largest_free'])))
        lines.append('chip%d %s MB' % (first['chip'], ' '.join(parts)))
    return lines


def _address(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def freed_ranges(nodes, banks=1):
    """The buffers a ttnn.graph capture saw allocated and then deallocated: [(lo, hi, buffer type)], each a PER-BANK extent
    (an interleaved DRAM buffer sits at the same address in every bank, and the node's `size` is the whole buffer across all
    of them): the node's max_size_per_bank where it carries one, else a DRAM buffer's size over `banks`. `nodes` is the
    capture's node list (or its JSON text); allocate and deallocate nodes carry their buffer's address in params. Live buffers
    are keyed by (type, address); a deallocate node that names no type frees the latest buffer at its address."""
    if isinstance(nodes, (str, bytes)):
        nodes = json.loads(nodes)
    live, freed = {}, []
    for node in nodes:
        kind, params = node.get('node_type'), node.get('params') or {}
        if kind not in ('buffer_allocate', 'buffer_deallocate') or 'address' not in params:
            continue
        address = _address(params['address'])
        if kind == 'buffer_allocate':
            live[(str(params.get('type', '?')), address)] = (int(params.get('size', 0)), _per_bank(params))
            continue
        if 'type' in params:
            key = (str(params['type']), address)
        else:
            key = next((key for key in reversed(list(live)) if key[1] == address), None)
        if key not in live:
            continue
        size, per_bank = live.pop(key)
        buffer_type = key[0]
        per_bank = _per_bank(params) or per_bank
        if per_bank is None:
            per_bank = -(-size // max(banks, 1)) if 'DRAM' in buffer_type.upper() else size
        freed.append((address, address + per_bank, buffer_type))
    return freed


def _per_bank(params):
    value = params.get('max_size_per_bank')
    return None if value is None else int(value)


def census_capture(original):
    """The twin of capture_operation: the original's call and result, with the census around it under QWEN_FAST_TRACE_CENSUS=1."""
    def capture_operation(operations, mesh, operation):
        if not census_enabled():
            return original(operations, mesh, operation)
        return censused(original, operations, mesh, operation, caller_site(2))
    capture_operation.census_of = original
    return capture_operation


def censused(original, operations, mesh, operation, site):
    global SEQUENCE
    SEQUENCE += 1
    seq = SEQUENCE
    before, state_before = memory_views(operations, mesh), collective_groups()
    started = False
    if os.environ.get(GRAPH_FLAG, '1') != '0':
        try:
            operations.graph.begin_graph_capture(operations.graph.RunMode.NORMAL)
            started = True
        except BaseException as failure:
            unavailable('graph capture %s: %s' % (type(failure).__name__, str(failure)[:80]))
    try:
        trace, result = original(operations, mesh, operation)
    finally:
        nodes = None
        if started:
            try:
                nodes = operations.graph.end_graph_capture()
            except BaseException as failure:
                unavailable('graph end %s: %s' % (type(failure).__name__, str(failure)[:80]))
    ranges = None
    if nodes is not None:
        try:
            ranges = freed_ranges(nodes, dram_banks(before))
        except BaseException as failure:
            unavailable('graph parse %s: %s' % (type(failure).__name__, str(failure)[:80]))
    after, state_after = memory_views(operations, mesh), collective_groups()
    if ranges is not None and len(TRACES) < MAX_TRACES:
        TRACES.append(dict(seq=seq, site=site, ranges=ranges, handle=trace))
    total = sum(hi - lo for lo, hi, _ in ranges) if ranges else 0
    log('[PINDIAG] trace census seq=%d site=%s temporaries=%s bytes_per_bank=%s' % (
        seq, site, 'n/a' if ranges is None else len(ranges), 'n/a' if ranges is None else total))
    for line in view_text(before, after):
        log('[PINDIAG] trace census seq=%d %s' % (seq, line))
    ccl_lines(seq, state_before, state_after)
    return trace, result


def dram_banks(views):
    """The DRAM bank count of the memory views (memory_views), 1 where they are unavailable."""
    try:
        return max(int(views[0]['dram']['banks']), 1)
    except BaseException:
        return 1


def ccl_lines(seq, before, after):
    """'ccl before{shared{..} model{..} sampler{..}} after{...}' on one line, or one line per object where that would pass
    CCL_LINE_BUDGET (and one per side where even that would pass LINE_BUDGET)."""
    if not before and not after:
        return
    head = '[PINDIAG] trace census seq=%d ccl' % seq
    whole = '%s before{%s} after{%s}' % (head, groups_text(before), groups_text(after))
    if len(whole) <= CCL_LINE_BUDGET:
        return log(whole)
    before_by_name, after_by_name = dict(before), dict(after)
    for name in [name for name, _ in before] + [name for name, _ in after if name not in before_by_name]:
        pair = '%s %s before{%s} after{%s}' % (head, name, before_by_name.get(name, ''), after_by_name.get(name, ''))
        if len(pair) <= LINE_BUDGET:
            log(pair)
        else:
            log('%s %s before{%s}' % (head, name, before_by_name.get(name, '')))
            log('%s %s after{%s}' % (head, name, after_by_name.get(name, '')))


def census_release(original):
    """The twin of ttnn.release_trace: the original's call, then the trace's freed temporaries are forgotten (a released trace
    is never replayed, so nothing it wrote can be written again)."""
    def release_trace(*args, **kwargs):
        result = original(*args, **kwargs)
        forget_trace(args + tuple(kwargs.values()))
        return result
    release_trace.census_of = original
    return release_trace


def forget_trace(handles):
    """Drop the recorded traces whose handle is one of `handles`."""
    if not TRACES:
        return

    def held(trace):
        for handle in handles:
            try:
                if trace['handle'] is handle or trace['handle'] == handle:
                    return True
            except BaseException:
                continue
        return False

    TRACES[:] = [trace for trace in TRACES if not held(trace)]


def count_packed_step(original):
    def note_packed_step():
        note_packed_step_called()
        return original()
    note_packed_step.census_of = original
    return note_packed_step


# --- persistent buffers against traces' freed temporaries ------------------------------------------------------------------


def engine_begin():
    """An engine's build starts: the live traces captured before this point are the ones its buffers are checked against, and
    the packed-step notes are counted from here (first_replay)."""
    global ENGINE_BOUNDARY, ENGINE_NOTES
    ENGINE_BOUNDARY, ENGINE_NOTES = SEQUENCE, PACKED_NOTES


def overlapping(lo, hi, traces):
    """The first trace whose freed temporaries intersect the per-bank extent [lo, hi), or None."""
    for trace in traces:
        for start, stop, buffer_type in trace['ranges']:
            if 'DRAM' in buffer_type.upper() and lo < stop and start < hi:
                return trace
    return None


def census_engine(request_id, request, operations):
    """After an engine is admitted (QWEN_FAST_TRACE_CENSUS=1): each captured width's persistent buffer extents per chip (per
    bank), and '[PINDIAG] trace overlap' for each buffer inside the freed temporaries of a live trace captured before this
    engine's build.
    Never raises. Also notes, under QWEN_FAST_SEQ_STAGE_LOG, when the engine was built (first_replay)."""
    if stage_log_enabled():
        BUILT_AT[id(getattr(request, 'engine', None))] = ENGINE_NOTES
    if not census_enabled():
        return
    try:
        _census_engine(request_id, request, operations)
    except BaseException as failure:
        unavailable('engine census %s: %s' % (type(failure).__name__, str(failure)[:80]))


def banks_of(operations, shard):
    """The DRAM bank count of the chip a shard lives on."""
    return max(int(operations.get_memory_view(shard.device(), operations.BufferType.DRAM).num_banks), 1)


def bank_extent(walker, shard, banks):
    """The bytes a DRAM-interleaved shard occupies in each bank: its pages dealt round-robin over `banks`, so the buffer's
    address is the same offset in every bank and its extent there is ceil(pages / banks) pages."""
    page = walker.page_bytes(shard)
    pages = -(-walker.shard_bytes(shard) // page)
    return -(-pages // banks) * page


def _census_engine(request_id, request, operations):
    import memory_ledger
    from tp_addresses import addresses

    engine = getattr(request, 'engine', None)
    walker = memory_ledger.MemoryLedger(operations, None, log=lambda message: None, emit=lambda text: None)
    earlier = [trace for trace in TRACES if trace['seq'] <= ENGINE_BOUNDARY]
    label, reported = request_label(request_id), 0
    for key, bucket in sorted(getattr(engine, 'buckets', {}).items(), key=lambda item: str(item[0])):
        rows = bucket.get('rows', key)
        extents, count = {}, {}
        for tensor in walker.tensors([bucket.get('fixture'), bucket.get('output')]):
            try:
                shards = operations.get_device_tensors(tensor)
                if not all(walker.in_dram(shard) for shard in shards):
                    continue
                chip_addresses = addresses(operations, tensor)
                sizes = [bank_extent(walker, shard, banks_of(operations, shard)) for shard in shards]
            except BaseException:
                continue
            for chip, (address, size) in enumerate(zip(chip_addresses, sizes)):
                low, high = extents.get(chip, (address, address))
                extents[chip] = (min(low, address), max(high, address + size))
                count[chip] = count.get(chip, 0) + 1
                trace = overlapping(address, address + size, earlier)
                if trace is not None:
                    reported += 1
                    if reported <= MAX_OVERLAP_LINES:
                        log('[PINDIAG] trace overlap request=%s rows=%s buffer=chip%d@%#x+%d/bank trace=%s'
                            % (label, rows, chip, address, size, trace['site']))
        for chip, (low, high) in sorted(extents.items()):
            log('[PINDIAG] engine buffers request=%s rows=%s chip%d buffers=%d lo=%#x hi=%#x'
                % (label, rows, chip, count[chip], low, high))
    if reported > MAX_OVERLAP_LINES:
        log('[PINDIAG] trace overlap request=%s more=%d' % (label, reported - MAX_OVERLAP_LINES))
    log('[PINDIAG] trace census engine request=%s traces=%d overlaps=%d' % (label, len(earlier), reported))


def note_collectives(collectives, model=None, sampler=None):
    """serving_runtime: the collectives object every request shares ('shared', the drafter's), and the two others that cycle
    their own semaphore handles: the target model's (the all-reduces in the verify and packed traces, eager prefill) and the
    pinned sampler's (its gathers, which the packed capture holds under the T1 audit)."""
    if not (census_enabled() or stage_log_enabled() or flag_on(CCL_LOG_FLAG)):
        return
    register_collectives(collectives)
    register_collectives(getattr(model, 'tt_ccl', None), 'model')
    sampler_ccl = getattr(sampler, 'tt_ccl', None)
    if sampler_ccl is None:
        sampler_ccl = getattr(getattr(sampler, 'tt_sampling', None), 'tt_ccl', None)
    register_collectives(sampler_ccl, 'sampler')


def install(environ=None):
    """Rebind capture_operation, verifier_engine.note_packed_step and ttnn.release_trace to their censusing twins in every loaded
    module holding the original by name (`from attention_batch import capture_operation` binds a copy), and on attention_batch
    itself, which the lazy imports in quad_draft read at call time. -> [(namespace, name, original)] for each binding changed, so tp_addresses
    can put them back; [] when already installed."""
    import importlib

    changed = []
    for module_name, name, wrap in (('attention_batch', 'capture_operation', census_capture),
                                    ('verifier_engine', 'note_packed_step', count_packed_step),
                                    ('ttnn', 'release_trace', census_release)):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        original = getattr(module, name, None)
        if original is None or getattr(original, 'census_of', None) is not None:
            continue
        twin = wrap(original)
        for candidate in list(sys.modules.values()):
            namespace = getattr(candidate, '__dict__', None)
            if isinstance(namespace, dict) and namespace.get(name) is original:
                namespace[name] = twin
                changed.append((namespace, name, original))
    return changed


def reset():
    """Forget every recorded trace and counter (tests, and a process that rebuilds its engines)."""
    global SEQUENCE, ENGINE_BOUNDARY, PACKED_NOTES, ENGINE_NOTES
    SEQUENCE, ENGINE_BOUNDARY, PACKED_NOTES, ENGINE_NOTES = 0, 0, 0, 0
    COLLECTIVES.clear()
    del TRACES[:]
    BUILT_AT.clear()
    UNAVAILABLE_REPORTED.clear()
