"""Op-fusion programme, host-gap package WPH, lever 2: a lean write_packed (QWEN_FAST_WRITE_PACKED_LEAN, audit twin QWEN_FAST_WRITE_PACKED_LEAN_AUDIT;
default off, 0 or 1, anything else is a configuration error at the attach).

packed_verifier.write_packed copies host values into the captured staging buffers. For every call it (1) reads every destination's per-chip buffer
address (get_device_tensors, then one buffer_address a chip), checks them pairwise distinct per chip, (2) builds one ReplicateTensorToMesh mapper PER VALUE,
(3) uploads each value to a host tensor, (4) copies it, (5) reads every address again and compares with the first read. A pre-stage writes about 148 buffers a
block, so (1), (2) and (5) are about 300 address reads and 148 mapper constructions of host time in the drafts' window, which is now on the critical path.

THE LEAN WRITE (write below) keeps (3) and (4) argument for argument and changes only the host's bookkeeping around them:
  - ONE MAPPER per mesh device, built on first use and kept: the mapper (`ttnn.replicate_tensor_to_mesh_mapper`, wrapped by ttnn.ReplicateTensorToMesh) is a
    stateless description of the placement, so the host tensors it produces are the same bytes whichever instance made them;
  - the buffer addresses are CAPTURED once, the first time a destination is seen (the attach's first full stage writes every destination), together with the
    pairwise-distinct check over everything captured; there is no read before a write;
  - the address check after a write is made ONCE A ROUND, in the drafts' window (the pre-stage, which is off the verify's critical path): it re-reads the
    addresses of the destinations that call wrote plus every destination written since the last check (the verify-time writes of the round before) and compares
    them with the captured ones. A buffer that moved raises the original's AssertionError ('Packed input staging replaced a captured buffer') there, and the
    window's failure handling is the one the pre-stage has always had (its snapshot is dropped and the verify takes the full stage). The verify-time write makes
    no address read at all.

EXACT BY CONSTRUCTION. The values, the dtype and layout of every upload, the order of the copies and the fence are the original's; what is removed is a defensive
check that raises or passes and a mapper object that is rebuilt. No byte that reaches the card depends on either. The audit twin proves it on the card instead of
arguing it: it runs the ORIGINAL's before/after address checks beside the book's (they must agree: the captured addresses are the before-read and the after-read of
every write), and it reads back a rotating eight of the call's destinations from every chip and compares them with the values (verify_prestage.same_readback). A
difference is logged (`... audit mismatch`), the destination is written again by the original function, and the lever latches off: every write after it is the
original's. Any exception in the lever's own bookkeeping, before the first copy, also latches it off and the original does the call.

The lean writer is for the pre-stage and the verify-time writes (verify_prestage.BlockPrestage.writer); packed_verifier.stage_packed, the full stage, keeps the
original. The address book and the mapper are PER BLOCK (the book lives on the block's Lean config, which lives on its BlockPrestage): a block that is closed and rebuilt starts a new
book, so a buffer address that the allocator hands out again is never compared with the dead block's, and the book holds nothing past the block that owns the destinations. Host only: no kernel, no trace, no device command but the copies the original makes.
"""

import os

FLAG = 'QWEN_FAST_WRITE_PACKED_LEAN'
AUDIT_FLAG = 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT'
ENGAGED_MARKER = '[PINDIAG] tp4 write packed lean engaged'
FELL_BACK_MARKER = '[PINDIAG] tp4 write packed lean fell back'
AUDIT_MARKER = '[PINDIAG] tp4 write packed lean audit'
MISMATCH_MARKER = '[PINDIAG] tp4 write packed lean audit mismatch'
AUDIT_DESTINATIONS = 8
REPLACED = 'Packed input staging replaced a captured buffer'


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_WRITE_PACKED_LEAN=1."""
    return _flag(FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_WRITE_PACKED_LEAN_AUDIT=1 with QWEN_FAST_WRITE_PACKED_LEAN=1 (alone it audits nothing: the smoke rule refuses it)."""
    return _flag(AUDIT_FLAG, environ) and enabled(environ)


def log_line(text):
    import verify_prestage

    verify_prestage.log_line(text)


class Book:
    """The state of the lean writer: the captured addresses (a reference to each destination, so an id is never reused while the block lives), the destinations written since the
    last address check, the mappers and the counters. One per block (Lean.book); BOOK is the module's default for direct calls and the tests."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.captured = {}      # id(destination) -> (destination, per-chip addresses)
        self.dirty = {}         # id(destination) -> destination, written since the last check
        self.mappers = {}       # id(mesh device) -> (mesh device, mapper)
        self.off = None         # the reason the lever gave up, or None
        self.engaged = set()    # the sites that logged their engaged line
        self.cursor = 0
        self.counts = dict(calls=0, written=0, captured=0, checked=0, audited=0, mismatches=0)


BOOK = Book()


def reset():
    """A new attach (and the tests): the module's default book forgotten (a block's own book is made with its Lean config)."""
    BOOK.clear()


def latch(reason, book=None):
    book = BOOK if book is None else book
    if book.off is None:
        book.off = str(reason).replace(' ', '_')[:160]
        log_line('%s reason=%s' % (FELL_BACK_MARKER, book.off))


class Lean:
    """What one block's pre-stage was built with: the lever on, whether its audit twin is, and the block's own book."""

    __slots__ = ('audit', 'book')

    def __init__(self, audit):
        self.audit = bool(audit)
        self.book = Book()

    def writer(self, site):
        """A function with packed_verifier.write_packed's signature, bound to `site`: 'prestage' makes the round's address check, 'verify' only notes what it wrote."""
        audit, book = self.audit, self.book

        def write_packed(operations, model, values, readers, *, indices=None, fence=True, poison=True):
            return write(operations, model, values, readers, indices=indices, fence=fence, poison=poison, site=site, audit=audit, book=book)

        return write_packed


def engage(block, environ=None):
    """BlockPrestage.__init__ (only when the flag is set to anything but 0): both flags read strictly. Returns the Lean config, or None for FLAG=0."""
    on, audit = enabled(environ), audit_enabled(environ)
    if not on:
        return None
    return Lean(audit)


def mapper_for(operations, mesh, book=None):
    """The one ReplicateTensorToMesh of `mesh` (in this book), built on first use."""
    book = BOOK if book is None else book
    entry = book.mappers.get(id(mesh))
    if entry is None or entry[0] is not mesh:
        entry = (mesh, operations.ReplicateTensorToMesh(mesh))
        book.mappers[id(mesh)] = entry
    return entry[1]


def capture(operations, destinations, book=None):
    """First sight of destinations: their addresses, the call-for-call checks of the original (a pair per chip; pairwise distinct per chip over everything this book captured and
    these), then remembered. Raises the original's ValueError."""
    import packed_verifier
    import tp_shapes

    book = BOOK if book is None else book
    fresh, seen = [], set()
    for destination in destinations:
        if id(destination) not in book.captured and id(destination) not in seen:
            seen.add(id(destination))
            fresh.append(destination)
    if not fresh:
        return
    found = [packed_verifier.addresses(operations, destination) for destination in fresh]
    chips = tp_shapes.chip_count()
    if any(len(pair) != chips for pair in found):
        raise ValueError('%s chip-local addresses per packed input required' % tp_shapes.count_word())
    known = [pair for _destination, pair in book.captured.values()]
    everything = known + found
    if any(len({pair[chip] for pair in everything}) != len(everything) for chip in range(chips)):
        raise ValueError('%s independent chip-local buffers per packed input required' % tp_shapes.count_word())
    for destination, pair in zip(fresh, found):
        book.captured[id(destination)] = (destination, pair)
    book.counts['captured'] += len(fresh)


def check_round(operations, destinations, book=None):
    """The round's address check: every destination written since the last check, and these, against the captured addresses. Raises the original's AssertionError."""
    import packed_verifier

    book = BOOK if book is None else book
    pending = dict(book.dirty)
    for destination in destinations:
        pending[id(destination)] = destination
    for key, destination in pending.items():
        if packed_verifier.addresses(operations, destination) != book.captured[key][1]:
            raise AssertionError(REPLACED)
    book.counts['checked'] += len(pending)
    book.dirty.clear()


def original(operations, model, values, readers, **options):
    import packed_verifier

    return packed_verifier.write_packed(operations, model, values, readers, **options)


def write(operations, model, values, readers, *, indices=None, fence=True, poison=True, site='stage', audit=False, book=None):
    """packed_verifier.write_packed, lean (module docstring). Returns the staged host tensors, which the caller keeps alive while a copy may be unfenced."""
    book = BOOK if book is None else book
    options = dict(indices=indices, fence=fence, poison=poison)
    if book.off is not None:
        return original(operations, model, values, readers, **options)
    chosen = values if indices is None else [values[index] for index in indices]
    destinations = [destination for destination, value, dtype, layout in chosen]
    try:
        before = None
        if audit:
            import packed_verifier

            before = [packed_verifier.addresses(operations, destination) for destination in destinations]
        capture(operations, destinations, book)
        if before is not None:
            wrong = [index for index, (destination, pair) in enumerate(zip(destinations, before)) if book.captured[id(destination)][1] != pair]
            if wrong:
                log_line('%s site=%s calls=%d reason=captured_addresses_differ_from_the_before_read at=%s' % (
                    MISMATCH_MARKER, site, book.counts['calls'], ','.join(str(index) for index in wrong[:8])))
                book.counts['mismatches'] += len(wrong)
                latch('audit_mismatch', book)
                return original(operations, model, values, readers, **options)
        mapper = mapper_for(operations, model.mesh_device, book)
    except ValueError:
        raise
    except Exception as failure:
        # Before the first copy: the original does the call, and the lever is done.
        latch('prelude:%s' % type(failure).__name__, book)
        return original(operations, model, values, readers, **options)
    if site not in book.engaged:
        book.engaged.add(site)
        log_line('%s site=%s mappers=1 audit=%d' % (ENGAGED_MARKER, site, int(audit)))
    book.counts['calls'] += 1
    book.counts['written'] += len(destinations)
    staged = [operations.from_torch(value, device=None, dtype=dtype, layout=layout, mesh_mapper=mapper)
              for destination, value, dtype, layout in chosen]
    try:
        try:
            for source, destination in zip(staged, destinations, strict=True):
                operations.copy_host_to_device_tensor(source, destination)
        finally:
            if fence:
                operations.synchronize_device(model.mesh_device)
        if site == 'verify':
            for destination in destinations:
                book.dirty[id(destination)] = destination
        else:
            check_round(operations, destinations, book)
        if audit:
            audit_call(operations, model, values, readers, chosen, destinations, before, options, book)
    except BaseException:
        # A half-staged word or table poisons the readers, as stage_inputs poisons one.
        if poison:
            for own in readers:
                own.failed = True
        raise
    return staged


def audit_call(operations, model, values, readers, chosen, destinations, before, options, book=None):
    """The audit twin, after a call's copies: the original's after-read against its before-read (the captured addresses must be both), and a rotating eight of the
    call's destinations read back from every chip against their values. A difference is written again by the original function and latches the lever off."""
    import packed_verifier
    import verify_prestage

    book = BOOK if book is None else book
    after = [packed_verifier.addresses(operations, destination) for destination in destinations]
    problems = []
    if after != before:
        problems.append('addresses_moved_during_the_write')
    count = len(chosen)
    picked = sorted({(book.cursor + offset) % count for offset in range(min(AUDIT_DESTINATIONS, count))}) if count else []
    book.cursor = (book.cursor + AUDIT_DESTINATIONS) % count if count else 0
    mismatched = []
    for index in picked:
        destination, value = destinations[index], chosen[index][1]
        for chip, shard in enumerate(operations.get_device_tensors(destination)):
            if not verify_prestage.same_readback(value, operations.to_torch(shard)):
                mismatched.append(index)
                break
    book.counts['audited'] += len(picked)
    if mismatched or problems:
        book.counts['mismatches'] += len(mismatched) + len(problems)
        log_line('%s calls=%d checked=%d mismatches=%d problems=%s exact=False' % (
            MISMATCH_MARKER, book.counts['calls'], len(picked), len(mismatched), ','.join(problems) or '-'))
        latch('audit_mismatch', book)
        if mismatched:
            retry = [options['indices'][index] if options['indices'] is not None else index for index in mismatched]
            original(operations, model, values, readers, indices=retry, fence=options['fence'], poison=options['poison'])
        return
    log_line('%s calls=%d checked=%d destinations=%d mismatches=0 exact=True' % (AUDIT_MARKER, book.counts['calls'], len(picked), count))
