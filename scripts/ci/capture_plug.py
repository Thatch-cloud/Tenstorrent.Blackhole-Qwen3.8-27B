"""Capture plug: keep what a trace capture frees from being handed to a later allocation (QWEN_FAST_CAPTURE_PLUG, off by default).

THE HYPOTHESIS. A trace capture allocates temporaries and frees them again; the replay writes them again every time. When a
buffer that is allocated AFTER the capture lands in the memory a temporary used to occupy, every replay overwrites it. The audits
change the capture's footprint (one more all-gather, +126 MB of kept windows), so whether and where such a victim lands changes
with them. This is a hypothesis, not a finding: no victim has been named (trace_census's overlap census is what names one), and
the module is the experiment arm that removes the class, not a claim that the class is the cause.

SIZES ARE PER BANK, NOT PER CHIP. DeviceMemory divides the allocator's totals by the bank count (a p150a has 8 DRAM banks of about 4.24
GB), so every *_MB below is megabytes in each bank, and a zone of LEAVE_MB takes LEAVE_MB x 8 on a chip. Measured on a four-card
engine before the packed capture (v174): about 2.5 GB per bank free (20,138 MB per chip), the packed block 224 MB per bank, its verify
capture 168 MB per bank, one engine 62 MB per bank. The defaults (512 leave, 256 reserve, 256 engine, 128 floor) fit that with margin; a
default must never be larger than a bank (the first draft asked for 6 GiB and could not attach).

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
fails at attach (out of memory, fail closed). One that spills into RESERVE does not fail: seal() finds where the used memory ends
(plugging the zone first, then every hole above it except the last one, the reserve) and seals up to it. Every plug is
inert: nothing reads it, and nothing is allocated inside a captured region. The graph census is not needed for any of this (it is
off in every profile: its str() of a tensor inside a capture is what killed v173).

THE FREE-MEMORY FLOOR is read ABOVE the packed zone (free_above): the bytes the reserve still has for kernel binaries and the
program cache, found by plugging everything below the zone's top for the length of the read. Total free memory would count the
low region the engines fill and hide a reserve that is nearly gone.

L1 IS NOT MODELLED. tt-metal allocates interleaved L1 buffers top-down by default (only DRAM defaults to bottom-up) and circular
buffers grow up from the L1 base, so the zone would sit at the wrong end: config() refuses QWEN_FAST_CAPTURE_PLUG_L1=1.

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
# Per-bank defaults (megabytes, then the unit), sized from the four-card measurements in the module docstring: they must fit in the
# ~2.4 GiB per bank that was free before the packed capture, with margin (the test asserts it).
DEFAULT_LEAVE_MB = (512, MB)
DEFAULT_RESERVE_MB = (256, MB)
DEFAULT_ENGINE_LEAVE_MB = (256, MB)
DEFAULT_MIN_FREE_MB = (128, MB)
MAX_SWEEP_ITERATIONS = 20000
MAX_SPILL_ROUNDS = 32
# Per-bank bytes of free memory that no page-sized plug can take: slivers between buffers (a few hundred bytes each, 3136 B in all
# on the v186 attach). seal() counts them as part of "nothing but the reserve is left" - a reserve zone must never be plugged
# whole because slivers made the free total a page or two larger than its last hole (v186: the whole 256 MB reserve was plugged
# and the next allocation ran out of memory).
SLIVER_ALLOWANCE = 1 * MB   # the cap; a zone's own allowance is 1/64 of its reserve, at least a page (so a small test reserve is not swallowed)
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
    """None when QWEN_FAST_CAPTURE_PLUG is not '1'; else the zone sizes in bytes PER BANK (not per chip; see the module docstring)
    and the engine flag. A malformed value, or QWEN_FAST_CAPTURE_PLUG_L1=1, is a configuration error. Only QWEN_FAST_TP=4 reaches this: the caller checks."""
    environ = os.environ if environ is None else environ
    value = environ.get(FLAG)
    if value in (None, '', '0'):
        return None
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    for flag in (ENGINES_FLAG, L1_FLAG):
        if environ.get(flag, '0') not in ('0', '1'):
            raise ValueError('%s must be 0 or 1, got %r' % (flag, environ.get(flag)))
    if environ.get(L1_FLAG, '0') == '1':
        raise ValueError('%s=1 is refused: the L1 zone assumes bottom-up allocation and tt-metal allocates interleaved L1 top-down '
                         '(circular buffers grow up from the L1 base); it is not modelled or verified on hardware' % L1_FLAG)
    return dict(leave=_positive(environ, LEAVE_FLAG, *DEFAULT_LEAVE_MB), reserve=_positive(environ, RESERVE_FLAG, *DEFAULT_RESERVE_MB),
                engine_leave=_positive(environ, ENGINE_LEAVE_FLAG, *DEFAULT_ENGINE_LEAVE_MB),
                min_free=_positive(environ, MIN_FREE_FLAG, *DEFAULT_MIN_FREE_MB), engines=environ.get(ENGINES_FLAG, '0') == '1',
                l1=False, l1_leave=_positive(environ, L1_LEAVE_FLAG, 512, KB),
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
    try:
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
    except BaseException:
        # A sweep that fails hands back what it took (inert blocks over memory that was free): the caller never saw the list.
        for block in plugs:
            try:
                memory.release(block)
            except BaseException:
                pass
        raise
    if log:
        log('[PINDIAG] capture plug sweep kind=%s limit=%#x plugs=%d plugged_mb=%.1f iterations=%d' % (
            memory.kind, limit, len(plugs), sum(block.per_bank for block in plugs) / MB, iterations))
    return plugs, iterations


def lowest_hole(memory):
    """-> (address, size) of the lowest free hole: a page probe lands at its start (first fit), and a block of size s lands there
    exactly when the hole holds s (the lowest hole is the first one tried), so a binary search over s finds the size. Every probe
    is freed again."""
    probe = memory.allocate(PAGE)
    address = probe.address
    memory.release(probe)
    low, high = 1, max(align_down(memory.largest_free()) // PAGE, 1)
    while low < high:
        middle = (low + high + 1) // 2
        block = memory.allocate(middle * PAGE)
        fits = block.address == address
        memory.release(block)
        if fits:
            low = middle
        else:
            high = middle - 1
    return address, low * PAGE


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

    def __init__(self, memory, leave, reserve, log=None, name='packed', ceiling=None, floor=None):
        self.memory, self.leave, self.reserve, self.log, self.name = memory, align_down(leave), align_down(reserve), log, name
        # The least free memory per bank the seal may leave (the kernel-binary reserve); None checks nothing. Only a zone with a
        # reserve above it is held to it: an engine zone has the packed zone above it, not free memory.
        self.floor = floor if reserve > 0 else None
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
        try:
            self._open()
        except BaseException:
            # Every refusal (no room, over the ceiling, an allocator failure) gives back what this call took: the sweep's low plugs
            # and the ballast. A refused engine build must not leak every hole below the main region.
            self._give_back_open()
            raise

    def _give_back_open(self):
        for block in ([self.ballast] if self.ballast is not None else []) + self.low_plugs:
            try:
                self.memory.release(block)
            except BaseException:
                pass
        self.ballast, self.low_plugs, self.state = None, [], 'closed'

    def _open(self):
        memory = self.memory
        start = probe_start(memory)
        self.low_plugs, _ = sweep(memory, start, self.log)
        largest = align_down(memory.largest_free())
        size = align_down(largest - self.leave - self.reserve)
        if size < PAGE:
            raise CapturePlugError('no room for a %d MB zone and %d MB reserve: the largest free block is %.1f MB per bank '
                                   '(sizes are per bank)' % (self.leave // MB, self.reserve // MB, largest / MB))
        self.ballast = memory.allocate(size)
        self.lo = self.ballast.end
        self.hi = self.lo + self.leave
        if self.ceiling is not None and self.hi > self.ceiling:
            raise CapturePlugError('the %s zone [%#x, %#x) would reach above %#x, into the memory reserved for kernel binaries' % (
                self.name, self.lo, self.hi, self.ceiling))
        self.state = 'open'
        self.say('open lo=%#x hi=%#x leave_mb=%d reserve_mb=%d ballast_mb=%.1f low_plugs=%d' % (
            self.lo, self.hi, self.leave // MB, self.reserve // MB, size / MB, len(self.low_plugs)))

    def seal(self):
        """Plug every hole in the zone and in whatever the captures used above it. The zone is swept first (the ballast is held, so
        its holes are the only free memory below its top, however large they are against what is left above). Then the holes above
        the zone's top are taken in address order, each found exactly (lowest_hole) and plugged, until the one that holds ALL the
        free memory that is left: the reserve (the region kernel binaries grow down into; binaries are never freed during serving,
        so no hole lies above it). A capture that spilled past the zone's top and freed temporaries there leaves holes below that
        region, and they are plugged too, whatever their size against it. Raises CapturePlugError when a hole survives."""
        if self.state != 'open':
            raise CapturePlugError('capture plug zone %s sealed from state %s' % (self.name, self.state))
        memory = self.memory
        limit = self.hi
        self.zone_plugs, _ = sweep(memory, limit, self.log)
        # Slivers (free pieces under a page, which no plug can take) are part of memory.free(); with a reserve above the zone they
        # must not make the reserve look like "a hole and then more free memory", or it is plugged whole.
        slack = max(PAGE, min(SLIVER_ALLOWANCE, self.reserve // 64)) if self.reserve > 0 else PAGE
        for unused in range(MAX_SPILL_ROUNDS + 1):
            address, size = lowest_hole(memory)
            remaining = memory.free() - size
            if remaining < slack:
                # This hole holds all the free memory that is left: the reserve. It is never plugged; the used memory ends where it
                # starts (a hole that starts a sliver under the zone top, because the sweep cannot plug under a page, ends the
                # zone there).
                if self.floor is not None and size < self.floor:
                    raise CapturePlugError('capture plug zone %s: only %.1f MB per bank is free above the sealed captures (a hole at '
                                           '%#x), under the %.1f MB floor for kernel binaries: raise %s' % (
                                               self.name, size / MB, address, self.floor / MB, RESERVE_FLAG))
                limit = address
                break
            if unused == MAX_SPILL_ROUNDS:
                raise CapturePlugError('capture plug zone %s: more than %d free holes above the zone' % (self.name, MAX_SPILL_ROUNDS))
            if self.floor is not None and remaining < self.floor:
                raise CapturePlugError('capture plug zone %s: plugging the freed hole [%#x, %#x) would leave %.1f MB per bank free, '
                                       'under the %.1f MB floor for kernel binaries (refused instead of taking the last free '
                                       'memory)' % (self.name, address, address + size, remaining / MB, self.floor / MB))
            self.say('spill: a freed hole [%#x, %#x) lies above the sealed limit %#x; plugging it' % (address, address + size, limit))
            self.zone_plugs.append(memory.allocate(size))
            if self.zone_plugs[-1].address != address:
                raise CapturePlugError('a spill hole of %d bytes landed at %#x, not at its start %#x' % (
                    size, self.zone_plugs[-1].address, address))
            limit = address + size
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


def free_above(memory, address):
    """The free bytes per bank at or above `address`: everything free below it is plugged for the length of the read (the same sweep
    that seals a zone), the allocator's free total is then the free memory above, and the plugs go back at once."""
    plugs, _ = sweep(memory, address)
    try:
        return memory.free()
    finally:
        for block in plugs:
            memory.release(block)


def monitor(memories, min_free, log=None, where='', above=None):
    """Fail closed when the free memory left (kernel binaries and the program cache live in it) is under `min_free` per bank. With
    `above` (a per-bank address: the top of the packed zone) the free memory counted is the part ABOVE it, the reserve the binaries
    use; without it, the total free memory (which includes the low region the engines fill, so it cannot see the reserve go)."""
    for memory in memories:
        free = memory.free() if above is None else free_above(memory, above)
        if log:
            log('[PINDIAG] capture plug monitor %s kind=%s free_%s_mb=%.1f largest_free_mb=%.1f min_free_mb=%.1f' % (
                where, memory.kind, 'total' if above is None else 'above_%#x' % above, free / MB, memory.largest_free() / MB,
                min_free / MB))
        if free < min_free:
            raise CapturePlugError('only %.1f MB per bank of %s is free %s (%s): the program cache has no room above the plugs' % (
                free / MB, memory.kind, 'in total' if above is None else 'above the packed zone', where or 'monitor'))


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


def give_back(plug, request):
    """The engine's plugs after its close: back to the allocator only when the request says it released everything
    (request.closed is True: its traces are gone); otherwise they are abandoned, still held, because the engine's traces may be live
    and a freed plug is a hole the replay writes through. FastRequest.close raises before touching the engine when the request is
    busy or the id does not match, and the engine's own close can fail part-way."""
    if getattr(request, 'closed', False) is True:
        plug.close()
    else:
        plug.abandon()


def seal_engine(plug, request, ranges):
    """The engine is built: plug the holes its captures freed, hold the census's recorded extents of them (`ranges`, none without the
    graph census) against the zone's top, and tie the plugs' life to the engine's: they go back when request.close has released its
    traces (request.closed), and are abandoned, never freed, when the close raises or leaves the request open."""
    plug.seal()
    plug.verify_extents(ranges)
    inner = request.close

    def close(*args, **kwargs):
        try:
            result = inner(*args, **kwargs)
        except BaseException:
            give_back(plug, request)
            raise
        give_back(plug, request)
        return result

    request.close = close


def packed_top():
    """The top of the packed zone: the highest sealed hi of the open packed plugs' DRAM zones (the reserve above all of them is what the
    free-memory floor counts), None with no packed plug."""
    return max((plug.zones[0].hi for plug in PACKED if plug.sealed), default=None)


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
        zones = [Zone(memory, leave, reserve, log, 'packed', floor=settings['min_free']) for memory, (_, leave, reserve) in zip(memories, kinds)]
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
        self.monitor('after seal')

    def verify_extents(self, ranges):
        for zone in self.zones:
            zone.verify_extents(ranges)

    def close(self):
        for zone in reversed(self.zones):
            zone.close()

    def abandon(self):
        for zone in self.zones:
            zone.abandon()

    def reserve_floor(self):
        """The address the free-memory floor is read above: this plug's own sealed DRAM zone top, else the packed zone's top (an
        engine plug before its zone opens), else None (the total free memory)."""
        own = [zone.hi for zone in self.zones if zone.state in ('sealed', 'released') and zone.memory.kind == 'DRAM']
        return max(own) if own else packed_top()

    def monitor(self, where=''):
        monitor(self.memories, self.min_free, self.log, where, above=self.reserve_floor())
