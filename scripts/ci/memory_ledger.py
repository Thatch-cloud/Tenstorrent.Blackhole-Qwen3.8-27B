"""L0 memory ledger: an env-gated, read-only itemisation of per-chip DRAM at fixed phases.

QWEN_FAST_MEMORY_LEDGER=1 turns it on; unset (the default) nothing is constructed and every
hook below returns at its first line, so the serving path is unchanged.

What it does, per phase (P0 at serving_startup.start entry ... P12 after the first packed
round, P13 before shutdown, plus a reading before and after each user's prefill):

1. Reads each chip's DRAM allocator through `serving_buffer_pool.dram_statistics` - the
   get_memory_view path already proven on the rig - via one fixed probe tensor. This is the
   PRIMARY itemisation: the allocator delta between neighbouring phases.
2. Walks the host-side objects the phase just built (instance dicts, lists, tuples, dicts)
   for device tensors, and sizes every DRAM shard it has not seen before from its padded
   shape and dtype (bf16 2 B, bf8 1088/1024, bf4 576/1024 per value). A shard's
   buffer_address is its identity, so a tensor reachable from two places counts once.
3. Checks that can FAIL:
   - delta: each phase's allocator delta per chip must equal the bytes of the tensors first
     walked at that phase, within bank rounding. The allowance is bounded by each buffer's
     OWN page: an interleaved buffer wastes at most one page per bank (the allocator
     reserves ceil(pages / banks) pages in every bank), so each new buffer adds banks x its
     page (the buffer's reported page size, else its tile or row-major row, aligned to
     DRAM_ALIGNMENT), plus RELATIVE_TOLERANCE of the walked bytes, the whole capped at
     TOLERANCE_CAP per chip per phase - so walking many small buffers cannot buy slack
     large enough to hide an unwalked allocation. A phase that allocated something the walk
     does not name, or a walk that names something allocated earlier, is UNMATCHED.
   - residual: at P7 (after attach) allocated minus every walked byte must be under
     RESIDUAL_LIMIT (1.5 GB) per chip; otherwise the phases whose deltas the walk could not
     explain are listed, largest first.

Read-only by construction: no device allocation, no synchronize, no program, no trace. It
never keeps a reference to a walked tensor (it stores addresses and sizes only), so it
cannot extend a buffer's life. Its call sites are the listed phase points only, all of which
are outside any trace capture. Every hook swallows its own failure and logs it: a ledger bug
must never take the serving path down with it.

The known set is cumulative: a buffer freed after the phase that walked it stays counted, and
a later buffer at a reused address is not counted again. Through P7 that is exact (the attach
frees nothing it walked); from the first request on (prefill, P8.., P12, P13) the delta check
stays exact for what each phase newly walks, but the residual is indicative only.

Output: short '[MEMLEDGER] phase=P<n> ...' lines, each message within LINE_BUDGET (180)
characters - the log capture truncates near 250 and loguru's prefix takes some of them, the
same budget as dflash_device.AUDIT_LINE_BUDGET - with one check=delta line per chip and a
request id cut to its last SHORT_ID characters (the JSON keeps the full one). The full
per-phase report is one JSON line ({"stage": "memory_ledger", ...}) appended to a FILE, not
the log: $QWEN_FAST_MEMORY_LEDGER_REPORT when set, else memory-ledger.jsonl in the gate's
mounted results directory (/experiment-results-gate) when that exists, else stdout. The
first log line names where it went.
"""

import json
import os
import types

FLAG = 'QWEN_FAST_MEMORY_LEDGER'
RESIDUAL_LIMIT = 1.5e9
RESIDUAL_PHASE = 'P7'
# Bank rounding is bounded per buffer by one of ITS OWN pages per bank (see the module
# docstring). The MLP block stream measured 3,227,516,928 B allocated against
# 3,220,439,040 B logical over 64 buffers (docs/mlp-block-stream.md:28,73): 110,592 B per
# buffer per chip, one 13,824 B page per bank on 8 banks - within RELATIVE_TOLERANCE of
# its 3.22 GB (16.1 MB) whatever page size this ttnn reports.
DRAM_ALIGNMENT = 64
RELATIVE_TOLERANCE = 0.005
TOLERANCE_CAP = 64 * 2 ** 20
LINE_BUDGET = 180
SHORT_ID = 12
REPORT_ENV = 'QWEN_FAST_MEMORY_LEDGER_REPORT'
GATE_RESULTS = '/experiment-results-gate'
MAX_WALK_NODES = 400000
MAX_WALK_DEPTH = 12
UNMATCHED_LISTED = 50
# Never descended into: classes, modules and code objects hold no device tensor of a phase.
NOT_WALKED = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType)

_active = None


def enabled(environ=None):
    return (os.environ if environ is None else environ).get(FLAG) == '1'


def log_line(message):
    try:
        from loguru import logger
    except ImportError:
        print(message, flush=True)
        return
    logger.info('{}', message)


def short_id(value):
    """A request id for a log label: its last SHORT_ID characters (the distinct end of a
    vLLM id); the JSON report carries the full id."""
    return str(value)[-SHORT_ID:]


def report_path(environ=None):
    """Where the per-phase JSON goes: the explicit path, else the gate's mounted results
    directory when it exists, else None (stdout)."""
    environ = os.environ if environ is None else environ
    if environ.get(REPORT_ENV):
        return environ[REPORT_ENV]
    if os.path.isdir(GATE_RESULTS):
        return os.path.join(GATE_RESULTS, 'memory-ledger.jsonl')
    return None


def file_emitter(path):
    def emit(text):
        with open(path, 'a', encoding='utf-8', newline='\n') as stream:
            stream.write(text + '\n')
    return emit


def active():
    return _active


def begin(operations, probe, *, log=None, emit=None):
    """Construct and register the process's ledger when the flag is on; None otherwise."""
    global _active
    if not enabled():
        return None
    _active = MemoryLedger(operations, probe, log=log, emit=emit)
    _active.log('[MEMLEDGER] report=%s' % _active.report)
    return _active


def end():
    global _active
    _active = None


def record(phase, *, point=None, request=None, **walked):
    """The hook every call site uses: a no-op unless a ledger is active."""
    ledger = _active
    if ledger is None:
        return None
    return ledger.phase(phase, point=point, request=request, **walked)


def engine_admitted(request_id, **walked):
    """P8..P11: after each of the first four admitted requests' engines (their captures,
    retained histories, DFlashDevice and proposal buffers); engine<n> for any later one."""
    ledger = _active
    if ledger is None:
        return None
    ledger.engines += 1
    name = 'P%d' % (7 + ledger.engines) if ledger.engines <= 4 else 'engine%d' % ledger.engines
    # The log label carries the id's last SHORT_ID characters, the JSON the full id.
    return ledger.phase(name, point='req=%s' % short_id(request_id), request=str(request_id), **walked)


def first_packed_round(**walked):
    """P12, once: after the first packed round's verify and commits have returned."""
    ledger = _active
    if ledger is None or ledger.first_round_recorded:
        return None
    ledger.first_round_recorded = True
    return ledger.phase('P12', point='first_packed_round', **walked)


def _gb(value):
    return '%.3fGB' % (value / 1e9)


def _mb(value):
    return '%.1fMB' % (value / 1e6)


class MemoryLedger:
    def __init__(self, operations, probe, *, log=None, emit=None):
        self.operations, self.probe = operations, probe
        self.raw_log = log_line if log is None else log
        self.report = 'caller'
        if emit is None:
            path = report_path()
            self.report = 'stdout' if path is None else path
            emit = (lambda text: print(text, flush=True)) if path is None else file_emitter(path)
        self.emit = emit
        self.known = {}          # (chip, address) -> (category, bytes, phase)
        self.readings = []       # (phase, point, [per-chip dict])
        self.checks = []         # per phase: dict(phase, status, chips=[...])
        self.first_round_recorded = False
        self.engines = 0

    def log(self, message):
        """One message within LINE_BUDGET; anything longer continues on further lines."""
        head, rest = message[:LINE_BUDGET], message[LINE_BUDGET:]
        self.raw_log(head)
        prefix = '[MEMLEDGER] ...'
        width = LINE_BUDGET - len(prefix)
        while rest:
            self.raw_log(prefix + rest[:width])
            rest = rest[width:]

    # --- sizing -----------------------------------------------------------------------

    def value_bytes(self, dtype):
        operations = self.operations
        table = (('bfloat16', 2.0), ('float32', 4.0), ('uint32', 4.0), ('int32', 4.0), ('uint16', 2.0),
                 ('uint8', 1.0), ('bfloat8_b', 1088 / 1024), ('bfloat4_b', 576 / 1024))
        for name, size in table:
            if hasattr(operations, name) and dtype == getattr(operations, name):
                return size
        return None

    def shard_bytes(self, shard):
        shape = getattr(shard, 'padded_shape', None)
        if shape is None:
            shape = shard.shape
        count = 1
        for size in tuple(shape):
            count *= int(size)
        size = self.value_bytes(shard.dtype)
        if size is None:
            raise ValueError('unsized dtype %r' % (shard.dtype,))
        return int(count * size)

    def page_bytes(self, shard):
        """The buffer's page, bounding its bank rounding: the size the buffer reports when
        this ttnn exposes it, else a row-major row or a tile of its dtype, aligned up to
        DRAM_ALIGNMENT."""
        page = None
        for read in (lambda: shard.buffer().page_size(), lambda: shard.tensor_spec.compute_page_size_bytes()):
            try:
                value = read()
            except Exception:
                continue
            if type(value) is int and value > 0:
                page = value
                break
        if page is None:
            size = self.value_bytes(shard.dtype) or 4.0
            row_major = getattr(getattr(self.operations, 'Layout', None), 'ROW_MAJOR', None)
            shape = tuple(getattr(shard, 'padded_shape', None) or shard.shape)
            if row_major is not None and getattr(shard, 'layout', None) == row_major and shape:
                page = int(shape[-1] * size)
            else:
                try:
                    height, width = (int(value) for value in shard.tile.tile_shape)
                except Exception:
                    height, width = 32, 32
                page = int(height * width * size)
        return -(-max(page, 1) // DRAM_ALIGNMENT) * DRAM_ALIGNMENT

    def in_dram(self, shard):
        buffer_type = getattr(self.operations, 'BufferType', None)
        dram = getattr(buffer_type, 'DRAM', None)
        config = shard.memory_config() if callable(getattr(shard, 'memory_config', None)) else None
        return dram is None or config is None or getattr(config, 'buffer_type', dram) == dram

    # --- walking ------------------------------------------------------------------------

    def tensors(self, root):
        """Every device tensor reachable from `root` through containers and instance dicts."""
        tensor_type = getattr(self.operations, 'Tensor', None)
        if not isinstance(tensor_type, type):
            return []
        found, seen, stack, visited = [], set(), [(root, 0)], 0
        skip = (str, bytes, bytearray, int, float, complex, bool, type(None))
        while stack and visited < MAX_WALK_NODES:
            value, depth = stack.pop()
            if id(value) in seen or isinstance(value, skip):
                continue
            seen.add(id(value))
            visited += 1
            if isinstance(value, tensor_type):
                found.append(value)
                continue
            if depth >= MAX_WALK_DEPTH or isinstance(value, NOT_WALKED):
                continue
            module = type(value).__module__ or ''
            if module.startswith(('torch', 'numpy', 'vllm', 'transformers')):
                continue
            if isinstance(value, dict):
                children = list(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                children = list(value)
            else:
                try:
                    children = list(vars(value).values())
                except TypeError:
                    continue
            stack.extend((child, depth + 1) for child in children)
        return found

    def claim(self, category, root, phase):
        """Record every not-yet-known DRAM shard under `root`; returns, per chip, the new
        bytes, the new buffer count and the new buffers' pages summed (the rounding bound per
        bank), and how many tensors could not be read."""
        added, buffers, pages, unreadable = {}, {}, {}, 0
        for tensor in self.tensors(root):
            try:
                device_storage = getattr(getattr(self.operations, 'StorageType', None), 'DEVICE', None)
                if (device_storage is not None and callable(getattr(tensor, 'storage_type', None))
                        and tensor.storage_type() != device_storage):
                    continue
                if callable(getattr(tensor, 'is_allocated', None)) and not tensor.is_allocated():
                    continue
                shards = self.operations.get_device_tensors(tensor)
                for chip, shard in enumerate(shards):
                    if not self.in_dram(shard):
                        continue
                    key = (chip, int(shard.buffer_address()))
                    if key in self.known:
                        continue
                    size = self.shard_bytes(shard)
                    self.known[key] = (category, size, phase)
                    added[chip] = added.get(chip, 0) + size
                    buffers[chip] = buffers.get(chip, 0) + 1
                    pages[chip] = pages.get(chip, 0) + self.page_bytes(shard)
            except BaseException:
                unreadable += 1
        return added, buffers, pages, unreadable

    # --- phases -------------------------------------------------------------------------

    def reading(self):
        from serving_buffer_pool import dram_statistics

        return dram_statistics(self.operations, self.probe)

    def known_bytes(self, chip):
        return sum(size for (owner, _), (_, size, _) in self.known.items() if owner == chip)

    def phase(self, name, *, point=None, request=None, **walked):
        try:
            return self._phase(name, point, request, walked)
        except BaseException as failure:
            try:
                self.log('[MEMLEDGER] phase=%s error=%s: %s' % (name, type(failure).__name__, str(failure)[:100]))
            except BaseException:
                pass
            return None

    def tolerance(self, new_bytes, pages, banks):
        """Bank rounding allowed for one chip's new buffers at one phase: one page per bank
        per buffer, plus RELATIVE_TOLERANCE of the walked bytes, capped at TOLERANCE_CAP."""
        return int(min(banks * pages + RELATIVE_TOLERANCE * new_bytes, TOLERANCE_CAP))

    def _phase(self, name, point, request, walked):
        label = name if point is None else '%s point=%s' % (name, point)
        chips = self.reading()
        categories = {}
        for category, root in walked.items():
            added, buffers, pages, unreadable = self.claim(category, root, name)
            categories[category] = dict(bytes=added, buffers=buffers, pages=pages, unreadable=unreadable)
        if isinstance(chips, dict):
            self.log('[MEMLEDGER] phase=%s dram unavailable (%s)' % (label, str(chips.get('unavailable'))[:100]))
            report = dict(stage='memory_ledger', phase=name, point=point, request=request, chips=chips,
                          known=categories)
            self.emit(json.dumps(report, default=str))
            return report
        previous = self.readings[-1][2] if self.readings else None
        check_chips, residuals = [], []
        for chip in chips:
            index = chip['chip']
            known = self.known_bytes(index)
            residual = chip['allocated'] - known
            residuals.append(residual)
            new_bytes = sum(entry['bytes'].get(index, 0) for entry in categories.values())
            new_pages = sum(entry['pages'].get(index, 0) for entry in categories.values())
            before = 0 if previous is None else next((item['allocated'] for item in previous
                                                      if item['chip'] == index), 0)
            delta = chip['allocated'] - before
            tolerance = self.tolerance(new_bytes, new_pages, chip['banks'])
            unexplained = delta - new_bytes
            check_chips.append(dict(chip=index, delta=delta, walked=new_bytes, unexplained=unexplained,
                                    tolerance=tolerance, matched=abs(unexplained) <= tolerance,
                                    residual=residual, known=known))
            self.log('[MEMLEDGER] phase=%s chip%d allocated=%s free=%s largest_free=%s total=%s known=%s residual=%s'
                     % (label, index, _gb(chip['allocated']), _gb(chip['free']), _mb(chip['largest_free']),
                        _gb(chip['total']), _gb(known), _gb(residual)))
        for category, entry in sorted(categories.items()):
            if entry['bytes'] or entry['unreadable']:
                self.log('[MEMLEDGER] phase=%s item=%s %s buffers=%s unreadable=%d'
                         % (label, category, ' '.join('chip%d=%s' % (chip, _gb(size)) for chip, size in sorted(entry['bytes'].items())),
                            sum(entry['buffers'].values()), entry['unreadable']))
        # The first reading has no neighbour: its whole allocation is residual, not a delta.
        status = 'first' if previous is None else ('matched' if all(item['matched'] for item in check_chips) else 'UNMATCHED')
        check = dict(phase=name, point=point, status=status, chips=check_chips)
        self.checks.append(check)
        if previous is not None:
            # One line per chip, so a failing check keeps each chip's figures whole.
            for item in check_chips:
                self.log('[MEMLEDGER] phase=%s check=delta status=%s chip%d delta=%s walked=%s unexplained=%s tol=%s'
                         % (label, status, item['chip'], _gb(item['delta']), _gb(item['walked']),
                            _gb(item['unexplained']), _mb(item['tolerance'])))
        residual_check = None
        if name == RESIDUAL_PHASE:
            passed = all(value < RESIDUAL_LIMIT for value in residuals)
            residual_check = dict(status='passed' if passed else 'FAILED', limit=RESIDUAL_LIMIT,
                                  residuals=residuals, unmatched=self.unmatched())
            self.log('[MEMLEDGER] phase=%s check=residual status=%s limit=%s %s' % (label, residual_check['status'],
                     _gb(RESIDUAL_LIMIT), ' '.join('chip%d=%s' % (index, _gb(value)) for index, value in enumerate(residuals))))
        if residual_check is not None and (residual_check['status'] != 'passed' or residual_check['unmatched']):
            for rank, item in enumerate(residual_check['unmatched'][:UNMATCHED_LISTED], 1):
                self.log('[MEMLEDGER] phase=%s unmatched rank=%d at=%s chip%d unexplained=%s'
                         % (name, rank, item['phase'], item['chip'], _gb(item['unexplained'])))
        self.readings.append((name, point, chips))
        report = dict(stage='memory_ledger', phase=name, point=point, request=request, chips=chips,
                      known=categories, check=check, residual=residual_check,
                      categories_total=self.category_totals())
        self.emit(json.dumps(report, default=str))
        return report

    def unmatched(self):
        """Every (phase, chip) whose allocator delta the walk did not explain, largest first."""
        items = [dict(phase=check['phase'] if check['point'] is None else '%s:%s' % (check['phase'], check['point']),
                      chip=item['chip'], unexplained=item['unexplained'])
                 for check in self.checks if check['status'] == 'UNMATCHED'
                 for item in check['chips'] if not item['matched']]
        return sorted(items, key=lambda item: -abs(item['unexplained']))

    def category_totals(self):
        totals = {}
        for (chip, _), (category, size, _) in self.known.items():
            totals.setdefault(category, {}).setdefault(chip, 0)
            totals[category][chip] += size
        return totals
