"""Capture plug: keep what a trace capture frees from being handed to a later allocation (QWEN_FAST_CAPTURE_PLUG, off by default).

THE HYPOTHESIS. A trace capture allocates temporaries and frees them again; the replay writes them again every time. When a
buffer that is allocated AFTER the capture lands in the memory a temporary used to occupy, every replay overwrites it. The audits
change the capture's footprint (one more all-gather, +126 MB of kept windows), so whether and where such a victim lands changes
with them. This is a hypothesis, not a finding: no victim has been named (trace_census's overlap census is what names one), and
the module is the experiment arm that removes the class, not a claim that the class is the cause.

WHAT THIS DOES (code only, no kernel change). Everything lives in per-bank terms (an interleaved buffer sits at the same offset
in every bank of every chip, so one address is the whole story) and relies on one allocator property: first-fit, bottom-up.

  open()   before the captures. (a) sweep: every free hole below the main free region is plugged by an inert, never-read tensor
           (so no capture temporary lands low and later frees a low hole); (b) one ballast tensor takes the main free region up to
           a ZONE of LEAVE_MB with RESERVE_MB above it left alone. The captures then allocate in the zone, bottom-up.
  seal()   after the last capture of the owner. With the ballast STILL held the only free memory below the zone's top is the zone
           itself, so a second sweep plugs every hole in it (the freed temporaries, whatever their size or place) with inert
           tensors. Nothing in [zone lo, zone hi) is free any more.
  release() the ballast goes. Every post-capture allocation (publication warm, engines and their traces, prefill, window snapshots)
           now lands in the low region the ballast held, which is disjoint from the zone by construction: first-fit takes the
           lowest hole, and the zone has none. The plugs live as long as the traces: closed with the owner, after its traces.

WHY THE ZONE IS NOT OPENED AT THE TOP AND THE BALLAST LEFT. The first proposal (a bottom-up ballast under the capture, released
after it) put the packed temporaries directly under the top-down kernel-binary region, so the program cache - unbounded through
the window programs - grew down into memory the replays rewrite. Here the zone is full of plugs after seal(), so a kernel
binary that reaches it cannot be placed: the allocation fails (fail closed) instead of overwriting, RESERVE_MB above the zone is
what binaries may use, and monitor() refuses (CapturePlugError) when the free memory left drops under MIN_FREE_MB.

WHAT IT DOES NOT COVER, AND WHAT CHECKS IT. The temporaries' high-water mark is not observable without the graph census (a freed
extent above the zone's top would be merged into the free region), so the zone has to be large enough: a capture that does not fit
fails at attach (out of memory, fail closed); one that spills into RESERVE does not, and verify_extents() (given the census's
recorded write sets, QWEN_FAST_TRACE_CENSUS_GRAPH=1) raises for any recorded extent that ends above the zone. Every plug is
inert: nothing reads it, and nothing is allocated inside a captured region.

The pair never reaches here: the flag is read by the TP4 attach only.

Stdlib only; the device is reached through the Memory adapter, which tests replace.
"""

import os

FLAG = 'QWEN_FAST_CAPTURE_PLUG'
ENGINES_FLAG = 'QWEN_FAST_CAPTURE_PLUG_ENGINES'
L1_FLAG = 'QWEN_FAST_CAPTURE_PLUG_L1'
LEAVE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_LEAVE_MB'
RESERVE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_RESERVE_MB'
ENGINE_LEAVE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_ENGINE_LEAVE_MB'
MIN_FREE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_MIN_FREE_MB'
L1_LEAVE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_L1_LEAVE_KB'
L1_RESERVE_FLAG = 'QWEN_FAST_CAPTURE_PLUG_L1_RESERVE_KB'
PAGE = 2048              # bytes per bank of the smallest plug: one bf16 tile page
MB = 1 << 20
KB = 1 << 10
MAX_SWEEP_ITERATIONS = 20000
ABANDONED = []           # blocks of a failed attach: referenced for good, never freed (a freed address under a running kernel)
PACKED = []              # the open packed plugs of this process: an engine's zone must stay under every one of them


class CapturePlugError(RuntimeError):
    """The plug could not be placed, sealed or kept: refused (fail closed) instead of served over a hole."""


def _positive(environ, name, default, unit):
    text = environ.get(name)
    if text is None or text == '':
        return default * unit
    if not text.isdigit() or int(text) <= 0:
        raise ValueError('%s must be a positive integer, got %r' % (name, text))
    return int(text) * unit


def config(environ=None):
    """None when QWEN_FAST_CAPTURE_PLUG is not '1'; else the zone sizes in bytes PER BANK (and the engine flag). A malformed
    value is a configuration error. Only QWEN_FAST_TP=4 reaches this: the caller checks."""
    environ = os.environ if environ is None else environ
    value = environ.get(FLAG)
    if value in (None, '', '0'):
        return None
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    for flag in (ENGINES_FLAG, L1_FLAG):
        if environ.get(flag, '0') not in ('0', '1'):
            raise ValueError('%s must be 0 or 1, got %r' % (flag, environ.get(flag)))
    return dict(leave=_positive(environ, LEAVE_FLAG, 4096, MB), reserve=_positive(environ, RESERVE_FLAG, 2048, MB),
                engine_leave=_positive(environ, ENGINE_LEAVE_FLAG, 1024, MB),
                min_free=_positive(environ, MIN_FREE_FLAG, 512, MB), engines=environ.get(ENGINES_FLAG, '0') == '1',
                l1=environ.get(L1_FLAG, '0') == '1', l1_leave=_positive(environ, L1_LEAVE_FLAG, 512, KB),
                l1_reserve=_positive(environ, L1_RESERVE_FLAG, 256, KB))


def align_down(value):
    return value // PAGE * PAGE


class Block:
    """One inert allocation: the tensor that owns it, its per-bank address and per-bank size."""

    def __init__(self, handle, address, per_bank, kind):
        self.handle, self.address, self.per_bank, self.kind = handle, address, per_bank, kind

    @property
    def end(self):
        return self.address + self.per_bank


class DeviceMemory:
    """The device behind the plug: per-bank allocations of one buffer type across the whole mesh, through ttnn. A tensor of
    `banks * k` bf16 tile pages (32 x 32) interleaved over the banks takes exactly k pages in each bank, at the same address on
    every chip: allocate_tensor_on_device writes nothing and uploads nothing."""

    def __init__(self, operations, mesh, kind='DRAM'):
        self.operations, self.mesh, self.kind = operations, mesh, kind

    def views(self):
        import memory_ledger

        buffer_type = self.operations.BufferType.L1 if self.kind == 'L1' else self.operations.BufferType.DRAM
        devices = self.mesh.get_devices() if callable(getattr(self.mesh, 'get_devices', None)) else [self.mesh]
        found = []
        for device in devices:
            view = memory_ledger.device_view(self.operations, device, buffer_type)
            banks = max(int(view['banks']), 1)
            found.append(dict(banks=banks, largest_free=view['largest_free'] // banks, free=view['free'] // banks))
        return found

    def banks(self):
        if getattr(self, 'bank_count', None) is None:
            self.bank_count = self.views()[0]['banks']
        return self.bank_count

    def largest_free(self):
        """The largest contiguous free block per bank, the smallest over the chips (mesh allocations are lockstep)."""
        return min(view['largest_free'] for view in self.views())

    def free(self):
        return min(view['free'] for view in self.views())

    def allocate(self, per_bank):
        operations = self.operations
        banks = self.banks()
        pages = banks * (per_bank // PAGE)
        config = operations.L1_MEMORY_CONFIG if self.kind == 'L1' else operations.DRAM_MEMORY_CONFIG
        tensor = operations.allocate_tensor_on_device(operations.Shape([1, 1, 32, 32 * pages]), operations.bfloat16,
                                                      operations.TILE_LAYOUT, self.mesh, config)
        from tp_addresses import addresses

        located = addresses(operations, tensor)
        if len(set(located)) != 1:
            operations.deallocate(tensor)
            raise CapturePlugError('a mesh allocation landed at different addresses per chip: %s' % (located,))
        return Block(tensor, located[0], per_bank, self.kind)

    def release(self, block):
        self.operations.deallocate(block.handle)


def probe_start(memory):
    """Where the main free region starts: the address a block of the largest free size lands at (first fit takes the lowest hole
    that fits, and the largest block is the only one that fits it). The probe is freed again."""
    size = align_down(memory.largest_free())
    if size < PAGE:
        raise CapturePlugError('no free block of a page to probe')
    block = memory.allocate(size)
    address = block.address
    memory.release(block)
    return address


def sweep(memory, limit, log=None, max_iterations=MAX_SWEEP_ITERATIONS):
    """Plug every free hole that lies entirely below `limit` (a per-bank address): allocate inert blocks, large to small. A block
    that lands at or above `limit` (or straddles it) proves no hole of that size is left below it - first fit takes the lowest
    hole - so it is freed and the size halves, down to one page. -> (plugs, iterations). Raises CapturePlugError when the
    iteration cap is reached with holes possibly left."""
    plugs, iterations = [], 0
    size = align_down(memory.largest_free())
    while size >= PAGE:
        iterations += 1
        if iterations > max_iterations:
            raise CapturePlugError('plug sweep stopped at %d iterations with holes possibly left below %#x' % (max_iterations, limit))
        size = min(size, align_down(memory.largest_free()))
        if size < PAGE:
            break
        block = memory.allocate(size)
        if block.end <= limit:
            plugs.append(block)
            continue
        memory.release(block)
        size = align_down(size // 2)
    if log:
        log('[PINDIAG] capture plug sweep kind=%s limit=%#x plugs=%d plugged_mb=%.1f iterations=%d' % (
            memory.kind, limit, len(plugs), sum(block.per_bank for block in plugs) / MB, iterations))
    return plugs, iterations


def verify_sealed(memory, limit):
    """Fail closed unless no free hole of a page remains below `limit`: a page-sized probe lands at the lowest hole there is."""
    block = memory.allocate(PAGE)
    address = block.address
    memory.release(block)
    if address < limit:
        raise CapturePlugError('a free hole remains at %#x below the sealed limit %#x' % (address, limit))


class Zone:
    """One owner's capture zone in one buffer type (see the module docstring): open(), the captures, seal(), release(); the
    plugs close with close() once the owner's traces are gone."""

    def __init__(self, memory, leave, reserve, log=None, name='packed', ceiling=None):
        self.memory, self.leave, self.reserve, self.log, self.name = memory, align_down(leave), align_down(reserve), log, name
        # Where the zone must end at the latest: an engine's zone must stay below the packed zone, never in the binary reserve above.
        self.ceiling = ceiling
        self.low_plugs, self.zone_plugs, self.ballast = [], [], None
        self.lo = self.hi = None
        self.state = 'new'

    def say(self, message):
        if self.log:
            self.log('[PINDIAG] capture plug %s kind=%s %s' % (self.name, self.memory.kind, message))

    def open(self):
        if self.state != 'new':
            raise CapturePlugError('capture plug zone %s opened twice' % self.name)
        memory = self.memory
        start = probe_start(memory)
        self.low_plugs, _ = sweep(memory, start, self.log)
        largest = align_down(memory.largest_free())
        size = align_down(largest - self.leave - self.reserve)
        if size < PAGE:
            raise CapturePlugError('no room for a %d MB zone and %d MB reserve: the largest free block is %.1f MB per bank' % (
                self.leave // MB, self.reserve // MB, largest / MB))
        self.ballast = memory.allocate(size)
        self.lo = self.ballast.end
        self.hi = self.lo + self.leave
        if self.ceiling is not None and self.hi > self.ceiling:
            memory.release(self.ballast)
            self.ballast = None
            for block in self.low_plugs:
                memory.release(block)
            self.low_plugs, self.state = [], 'closed'
            raise CapturePlugError('the %s zone [%#x, %#x) would reach above %#x, into the memory reserved for kernel binaries' % (
                self.name, self.lo, self.hi, self.ceiling))
        self.state = 'open'
        self.say('open lo=%#x hi=%#x leave_mb=%d reserve_mb=%d ballast_mb=%.1f low_plugs=%d' % (
            self.lo, self.hi, self.leave // MB, self.reserve // MB, size / MB, len(self.low_plugs)))

    def seal(self):
        """Plug every hole in the zone. Raises CapturePlugError when the captures spilled past the zone's top (a persistent
        buffer above it) or a hole survives the sweep."""
        if self.state != 'open':
            raise CapturePlugError('capture plug zone %s sealed from state %s' % (self.name, self.state))
        memory = self.memory
        start = probe_start(memory)
        limit = max(self.hi, start)
        if start > self.hi:
            self.say('spill: the free region starts at %#x, above the zone top %#x; sealing up to it' % (start, self.hi))
        self.zone_plugs, _ = sweep(memory, limit, self.log)
        verify_sealed(memory, limit)
        self.hi = limit
        self.state = 'sealed'
        self.say('sealed lo=%#x hi=%#x zone_plugs=%d plugged_mb=%.1f' % (
            self.lo, self.hi, len(self.zone_plugs), sum(block.per_bank for block in self.zone_plugs) / MB))

    def release(self):
        """Give the ballast back: from here every allocation lands below the zone."""
        if self.state != 'sealed':
            raise CapturePlugError('capture plug zone %s released from state %s' % (self.name, self.state))
        self.memory.release(self.ballast)
        self.ballast = None
        self.state = 'released'
        self.say('released ballast; free_mb=%.1f largest_free_mb=%.1f' % (self.memory.free() / MB, self.memory.largest_free() / MB))

    def close(self):
        """The owner's traces are released: the plugs (and a ballast still held) go back."""
        for block in self.zone_plugs + self.low_plugs + ([self.ballast] if self.ballast is not None else []):
            self.memory.release(block)
        self.zone_plugs, self.low_plugs, self.ballast, self.state = [], [], None, 'closed'

    def abandon(self):
        """A failed attach: the device may still be running work these blocks sit under, so nothing is freed (the process does
        not allocate again). The blocks stay referenced for good: a destructor must not free them either."""
        ABANDONED.extend(self.zone_plugs + self.low_plugs + ([self.ballast] if self.ballast is not None else []))
        self.zone_plugs, self.low_plugs, self.ballast, self.state = [], [], None, 'abandoned'

    def verify_extents(self, ranges):
        """`ranges` are census write-set extents [(lo, hi, buffer type)] of this owner's captures: none may end above the zone
        (they would lie in memory the plug does not hold). Raises CapturePlugError for the first that does."""
        for lo, hi, buffer_type in ranges:
            if self.memory.kind in buffer_type.upper() and hi > self.hi:
                raise CapturePlugError('a capture freed extent [%#x, %#x) %s ends above the zone top %#x: raise %s' % (
                    lo, hi, buffer_type, self.hi, LEAVE_FLAG))


def monitor(memories, min_free, log=None, where=''):
    """Fail closed when the free memory left (kernel binaries and the program cache live in it) is under `min_free` per bank."""
    for memory in memories:
        free = memory.free()
        if log:
            log('[PINDIAG] capture plug monitor %s kind=%s free_mb=%.1f largest_free_mb=%.1f min_free_mb=%.1f' % (
                where, memory.kind, free / MB, memory.largest_free() / MB, min_free / MB))
        if free < min_free:
            raise CapturePlugError('only %.1f MB per bank of %s is free (%s): the program cache has no room above the plugs' % (
                free / MB, memory.kind, where or 'monitor'))


def engine_settings(environ=None):
    """The settings an engine build's zone takes, or None: the pair, an unset flag and QWEN_FAST_CAPTURE_PLUG_ENGINES unset get None."""
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_FAST_TP', '2') != '4':
        return None
    settings = config(environ)
    return settings if settings is not None and settings['engines'] else None


def open_engine(settings, operations, mesh, log=None, memory_factory=None):
    """One engine's capture zone (Plug.engine), checked against the free-memory floor and opened, under the packed zone."""
    extra = {} if memory_factory is None else dict(memory_factory=memory_factory)
    plug = Plug.engine(settings, operations, mesh, log, ceiling=packed_ceiling(), **extra)
    plug.monitor('before engine build')
    plug.open()
    return plug


def seal_engine(plug, request, ranges):
    """The engine is built: plug the holes its captures freed, hold the census's recorded extents of them (`ranges`, none without the
    graph census) against the zone's top, and tie the plugs' life to the engine's: they go back when request.close has released its
    traces, even if that close raises."""
    plug.seal()
    plug.verify_extents(ranges)
    inner = request.close

    def close(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        finally:
            plug.close()

    request.close = close


def packed_ceiling():
    """The address an engine's zone must end under: the lowest zone start of the packed plugs, None with no packed plug."""
    return min((plug.zone_lo() for plug in PACKED if plug.sealed), default=None)


class Plug:
    """The zones of one owner, per buffer type: the packed block's (packed(), a zone with the reserve above it) or one engine's
    (engine(), its captures' zone inside what the ballast leaves). Built by the serving code only when config() says so."""

    def __init__(self, zones, memories, min_free, log=None):
        self.zones, self.memories, self.min_free, self.log = zones, memories, min_free, log
        self.sealed = False

    @classmethod
    def packed(cls, settings, operations, mesh, log=None, memory_factory=DeviceMemory):
        kinds = [('DRAM', settings['leave'], settings['reserve'])]
        if settings['l1']:
            kinds.append(('L1', settings['l1_leave'], settings['l1_reserve']))
        memories = [memory_factory(operations, mesh, kind) for kind, _, _ in kinds]
        zones = [Zone(memory, leave, reserve, log, 'packed') for memory, (_, leave, reserve) in zip(memories, kinds)]
        return cls(zones, memories, settings['min_free'], log)

    @classmethod
    def engine(cls, settings, operations, mesh, log=None, memory_factory=DeviceMemory, ceiling=None):
        memory = memory_factory(operations, mesh, 'DRAM')
        return cls([Zone(memory, settings['engine_leave'], 0, log, 'engine', ceiling)], [memory], settings['min_free'], log)

    def zone_lo(self):
        """Where this plug's (DRAM) zone begins: the ceiling an engine's zone must stay under."""
        return self.zones[0].lo

    def open(self):
        for zone in self.zones:
            zone.open()

    def seal(self):
        """Seal every zone and give the ballasts back (open zones only)."""
        for zone in self.zones:
            zone.seal()
        for zone in self.zones:
            zone.release()
        self.sealed = True
        monitor(self.memories, self.min_free, self.log, 'after seal')

    def verify_extents(self, ranges):
        for zone in self.zones:
            zone.verify_extents(ranges)

    def close(self):
        for zone in reversed(self.zones):
            zone.close()

    def abandon(self):
        for zone in self.zones:
            zone.abandon()

    def monitor(self, where=''):
        monitor(self.memories, self.min_free, self.log, where)
