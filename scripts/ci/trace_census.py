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
QWEN_FAST_CCL_HANDLE_LOG=1: the shared collectives object's integer state (TT_CCL cycles its semaphore handles by host-side
  index; a trace replay does not advance the index) appended to every '[SEQ-STAGE]' line. This is the test for the documented
  hang mode: an eager collective reusing a handle a replayed trace baked.
QWEN_FAST_TRACE_CENSUS=1: every trace capture logs its call site, the DRAM and L1 allocator views before and after, and the
  collectives' state before and after; best effort (QWEN_FAST_TRACE_CENSUS_GRAPH=0 turns it off) the address ranges the
  capture allocated and freed again - its temporaries, which a replay rewrites - from a ttnn.graph capture around it.
  census_engine then lists each width's persistent buffer extents per chip and reports every buffer that sits inside an
  earlier trace's freed temporaries ('[PINDIAG] trace overlap'): a replay of that trace writes into the buffer.

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
MAX_TRACES = 4096
MAX_OVERLAP_LINES = 20

# The shared collectives object (serving_runtime.note), the capture sequence number, the traces so far and where the
# current engine's build began in that sequence.
COLLECTIVES = None
TRACES = []
SEQUENCE = 0
ENGINE_BOUNDARY = 0
PACKED_STEPS = 0
ENGINE_PACKED = 0
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
    message = message[:LINE_BUDGET]
    try:
        from loguru import logger
    except ImportError:
        print(message, flush=True)
        return
    logger.info('{}', message)


def request_label(request_id):
    return str(request_id)[:ID_WIDTH]


# --- live stage markers -------------------------------------------------------------------------------------------------


def register_collectives(collectives):
    global COLLECTIVES
    COLLECTIVES = collectives


def collective_state(collectives=None):
    """The integer attributes (and integer lists) of the collectives object, one compact string, or '' when none is
    registered or none is readable."""
    collectives = COLLECTIVES if collectives is None else collectives
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


def ccl_suffix():
    if not flag_on(CCL_LOG_FLAG):
        return ''
    state = collective_state()
    return '' if not state else ' ccl{%s}' % state


def stage(request_id, rows, name, edge='begin', extra=''):
    """'[SEQ-STAGE]' line, flushed as it is written, when QWEN_FAST_SEQ_STAGE_LOG=1; a no-op otherwise."""
    if not stage_log_enabled():
        return
    log('[SEQ-STAGE] request=%s rows=%s%s stage=%s %s%s' % (request_label(request_id), rows, extra, name, edge,
                                                             ccl_suffix()))


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
    global PACKED_STEPS
    PACKED_STEPS += 1


def first_replay(engine, request_id, rows):
    """One line when a bucket replays its captured trace for the first time (QWEN_FAST_SEQ_STAGE_LOG=1): how many packed
    steps ran since this engine was built, and the program cache's size."""
    if not stage_log_enabled():
        return
    built = BUILT_AT.get(id(engine))
    log('[PINDIAG] first replay request=%s rows=%s built_after_packed_round=%s packed_rounds_since_build=%s program_cache=%s'
        % (request_label(request_id), rows, 'n/a' if built is None else built,
           'n/a' if built is None else PACKED_STEPS - built, program_cache_entries(getattr(engine, 'mesh', None))))


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


def freed_ranges(nodes):
    """The buffers a ttnn.graph capture saw allocated and then deallocated: [(lo, hi, buffer type)]. `nodes` is the capture's
    node list (or its JSON text); allocate and deallocate nodes carry their buffer's address in params."""
    if isinstance(nodes, (str, bytes)):
        nodes = json.loads(nodes)
    live, freed = {}, []
    for node in nodes:
        kind, params = node.get('node_type'), node.get('params') or {}
        if kind not in ('buffer_allocate', 'buffer_deallocate') or 'address' not in params:
            continue
        address = _address(params['address'])
        if kind == 'buffer_allocate':
            live[address] = (int(params.get('size', 0)), str(params.get('type', '?')))
        elif address in live:
            size, buffer_type = live.pop(address)
            freed.append((address, address + size, buffer_type))
    return freed


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
    before, state_before = memory_views(operations, mesh), collective_state()
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
            ranges = freed_ranges(nodes)
        except BaseException as failure:
            unavailable('graph parse %s: %s' % (type(failure).__name__, str(failure)[:80]))
    after, state_after = memory_views(operations, mesh), collective_state()
    if ranges is not None and len(TRACES) < MAX_TRACES:
        TRACES.append(dict(seq=seq, site=site, ranges=ranges))
    total = sum(hi - lo for lo, hi, _ in ranges) if ranges else 0
    log('[PINDIAG] trace census seq=%d site=%s temporaries=%s bytes=%s' % (
        seq, site, 'n/a' if ranges is None else len(ranges), 'n/a' if ranges is None else total))
    for line in view_text(before, after):
        log('[PINDIAG] trace census seq=%d %s' % (seq, line))
    if state_before or state_after:
        log('[PINDIAG] trace census seq=%d ccl before{%s} after{%s}' % (seq, state_before, state_after))
    return trace, result


def count_packed_step(original):
    def note_packed_step():
        note_packed_step_called()
        return original()
    note_packed_step.census_of = original
    return note_packed_step


# --- persistent buffers against traces' freed temporaries ------------------------------------------------------------------


def engine_begin():
    """An engine's build starts: traces captured before this point are the ones its buffers are checked against, and the
    packed steps are counted from here (first_replay)."""
    global ENGINE_BOUNDARY, ENGINE_PACKED
    ENGINE_BOUNDARY, ENGINE_PACKED = SEQUENCE, PACKED_STEPS


def overlapping(lo, hi, traces):
    """The first trace whose freed temporaries intersect [lo, hi), or None."""
    for trace in traces:
        for start, stop, buffer_type in trace['ranges']:
            if 'DRAM' in buffer_type.upper() and lo < stop and start < hi:
                return trace
    return None


def census_engine(request_id, request, operations):
    """After an engine is admitted (QWEN_FAST_TRACE_CENSUS=1): each captured width's persistent buffer extents per chip, and
    '[PINDIAG] trace overlap' for each buffer inside the freed temporaries of a trace captured before this engine's build.
    Never raises. Also notes, under QWEN_FAST_SEQ_STAGE_LOG, when the engine was built (first_replay)."""
    if stage_log_enabled():
        BUILT_AT[id(getattr(request, 'engine', None))] = ENGINE_PACKED
    if not census_enabled():
        return
    try:
        _census_engine(request_id, request, operations)
    except BaseException as failure:
        unavailable('engine census %s: %s' % (type(failure).__name__, str(failure)[:80]))


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
                sizes = [walker.shard_bytes(shard) for shard in shards]
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
                        log('[PINDIAG] trace overlap request=%s rows=%s buffer=chip%d@%#x+%d trace=%s'
                            % (label, rows, chip, address, size, trace['site']))
        for chip, (low, high) in sorted(extents.items()):
            log('[PINDIAG] engine buffers request=%s rows=%s chip%d buffers=%d lo=%#x hi=%#x'
                % (label, rows, chip, count[chip], low, high))
    if reported > MAX_OVERLAP_LINES:
        log('[PINDIAG] trace overlap request=%s more=%d' % (label, reported - MAX_OVERLAP_LINES))
    log('[PINDIAG] trace census engine request=%s traces=%d overlaps=%d' % (label, len(earlier), reported))


def note_collectives(collectives):
    """serving_runtime: the one collectives object every request shares."""
    if census_enabled() or stage_log_enabled() or flag_on(CCL_LOG_FLAG):
        register_collectives(collectives)


def install(environ=None):
    """Rebind capture_operation and verifier_engine.note_packed_step to their censusing twins in every loaded module holding the
    original by name (`from attention_batch import capture_operation` binds a copy), and on attention_batch itself, which the
    lazy imports in quad_draft read at call time. -> [(namespace, name, original)] for each binding changed, so tp_addresses
    can put them back; [] when already installed."""
    import importlib

    changed = []
    for module_name, name, wrap in (('attention_batch', 'capture_operation', census_capture),
                                    ('verifier_engine', 'note_packed_step', count_packed_step)):
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
    global COLLECTIVES, SEQUENCE, ENGINE_BOUNDARY, PACKED_STEPS, ENGINE_PACKED
    COLLECTIVES, SEQUENCE, ENGINE_BOUNDARY, PACKED_STEPS, ENGINE_PACKED = None, 0, 0, 0, 0
    del TRACES[:]
    BUILT_AT.clear()
    UNAVAILABLE_REPORTED.clear()
