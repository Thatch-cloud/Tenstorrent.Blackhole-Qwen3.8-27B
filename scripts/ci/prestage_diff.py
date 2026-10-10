"""Op-fusion programme, host-gap package WPH, lever 1: the pre-stage writes only the destinations whose bytes differ
(QWEN_FAST_PRESTAGE_DIFF, audit twin QWEN_FAST_PRESTAGE_DIFF_AUDIT; default off, 0 or 1, anything else is a configuration error at the attach).

WHAT IT BUYS. The drafts' window pre-stages the next packed verify (verify_prestage.BlockPrestage.prestage): every verify input but the tokens,
about 148 buffers a block, each one a host tensor, a mapper, a from_torch and a copy. With the fusion levers the drafter quads got shorter and this window
(about 18 ms for two blocks) is on the critical path: the host's wait in the fence F9 after the window fell from 9.4 ms to 0.39 ms. Only about 73 of the 148
buffers change from one round to the next (the positions, the rotary tables, the 64 per-row positions, the tile positions and the readers' words); the page
derived ones (the page table, the 64 per-row tables, the tile tables, the readers' per-bundle tables) change only at a page crossing.

WHAT IT DOES. After every write of a block's staging buffers (the pre-stage, the verify-time diff, the keyed write) the block remembers, per destination, the
exact host value that is now on the device (`Resident`). At the next pre-stage the freshly computed values are compared with the resident ones BIT FOR BIT
(`same_bits`: dtype, shape and every element's bit pattern, so +0 and -0, or two NaN payloads, are different) and only the destinations that differ are
written. The values are computed exactly as before (packed_verifier.packed_values), so every check it makes is made as before.

EXACT BY CONSTRUCTION. After the pre-stage every destination holds the bytes `from_torch(value)` of the value the full pre-stage would have written:
  - a destination that was written holds them by the same call;
  - a destination that was skipped holds the bytes of a resident value that is bit-equal to the new one, and from_torch of equal bits is equal bytes
    (a pure function of the host bits, dtype and layout);
and nothing wrote that destination in between, which is the one invariant the whole pre-stage already stands on, now across rounds:
  1. EVERY OTHER WRITER of a fixture's inputs moves the write epoch (verify_prestage.bump / bump_fixture: the full stage_packed, a prefill or Lever N chunk,
     an admission or a detach, a probe restage), and a resident whose epoch (global, and the fixture's own with the per-block epochs) is not the current one is
     not used: the pre-stage is the full one. A step of a split prefill that wrote no decode slot, declared disjoint (verify_prestage.note_disjoint), writes
     only addresses that were checked disjoint from every block's staging destinations, so it cannot touch a resident byte either;
  2. THE BLOCK'S OWN WRITES are recorded as they are made: the pre-stage records what it wrote, the verify-time diff records the snapshot's values with the
     changed destinations replaced by the verify-time values (the unchanged ones keep the snapshot's, which are the bytes actually on the device), the keyed
     write records the snapshot's values, and a full stage_packed (the first round, an epoch kill, a padded probe) records nothing, so the next pre-stage is full;
  3. THE TRACES. The verify trace, the commit traces and the drafts read the staging buffers (embedding indices, rotary multiplicands, the paged cache update's
     positions and page tables, the mask's positions word, the readers' words and tables) and the captured programs name none of them as an output. That is the
     one claim no host code can prove from the host: the audit twin proves it on the card, round by round (below), and the gate is the audited attach.
  4. A destination list that is not the resident's (an object replaced, a different length) takes the full pre-stage, and so does any exception inside the
     lever (it latches the lever off for the process: a `fell back` line, and every pre-stage after it is the full one).

THE AUDIT (QWEN_FAST_PRESTAGE_DIFF_AUDIT, with the lever). After every pre-stage write, EVERY pre-staged destination (the skipped ones included) is read back from
every chip and compared with the value the full pre-stage would have written (the same comparator as the existing audits, verify_prestage.same_readback). The
read is enqueued behind the traces and the writes on the in-order queue, so it sees the trace's effect on the buffers: a trace that wrote a staging buffer, an
external writer the epoch missed, a resident that was wrong, all read as a mismatch. A mismatch is logged (`... audit mismatch`), the round is repaired by writing
every pre-staged destination in full, and the lever latches off. The existing QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT (every destination read back after the
verify-time write, path=diff, zero mismatches) is the second, independent gate: a skipped write that was wrong shows there too, because the verify-time diff
compares with the snapshot's values, which say the buffer is right.

COMPOSITION. KEYED (round_host) keeps its own claim (every staged value but the tokens is a function of the start and page table, so the verify-time recompute is
skipped); this lever works one step earlier, on the pre-stage, and records what KEYED wrote. LEAN WRITES (write_packed_lean) is the writer both use. TWO_BLOCK and
the per-block epochs are what make a resident survive the OTHER block's round: with one global epoch the other block's verify bumps it, and every pre-stage is full.

Markers: `[PINDIAG] tp4 prestage diff engaged|fell back|audit ... exact=True`, and per pre-stage `[PACKED-PRESTAGE-DIFF] block= round= path=diff|full reason= total= written=
skipped=`. Host only: no kernel, no trace, no device command but the writes and reads the full path makes.
"""

import os
import sys

FLAG = 'QWEN_FAST_PRESTAGE_DIFF'
AUDIT_FLAG = 'QWEN_FAST_PRESTAGE_DIFF_AUDIT'
ENGAGED_MARKER = '[PINDIAG] tp4 prestage diff engaged'
FELL_BACK_MARKER = '[PINDIAG] tp4 prestage diff fell back'
AUDIT_MARKER = '[PINDIAG] tp4 prestage diff audit'
MISMATCH_MARKER = '[PINDIAG] tp4 prestage diff audit mismatch'
ROUND_MARKER = '[PACKED-PRESTAGE-DIFF]'

# The process-wide latch: a reason once the lever has given up (the audit found a difference, the lever's own code raised); None while it stands.
_STATE = dict(off=None, engaged=0)
COUNTS = dict(diff=0, full=0, written=0, skipped=0, audited=0, mismatches=0)


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_PRESTAGE_DIFF=1."""
    return _flag(FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_PRESTAGE_DIFF_AUDIT=1 with QWEN_FAST_PRESTAGE_DIFF=1 (alone it audits nothing: the smoke rule refuses it)."""
    return _flag(AUDIT_FLAG, environ) and enabled(environ)


def reset():
    """A new attach (and the tests): the latch and the counters forgotten."""
    _STATE.update(off=None, engaged=0)
    for key in COUNTS:
        COUNTS[key] = 0


def latched():
    """The reason the lever gave up for this process, or None."""
    return _STATE['off']


def log_line(text):
    import verify_prestage

    verify_prestage.log_line(text)


def latch(reason):
    """Give the lever up for the process (the safe direction: every pre-stage after this is the full one). Logged once."""
    if _STATE['off'] is None:
        _STATE['off'] = str(reason).replace(' ', '_')[:160]
        log_line('%s reason=%s' % (FELL_BACK_MARKER, _STATE['off']))


class Config:
    """What one block's pre-stage was built with: the lever on, and whether its audit twin is."""

    __slots__ = ('audit',)

    def __init__(self, audit):
        self.audit = bool(audit)


def engage(block, environ=None):
    """BlockPrestage.__init__ (only when the flag is set to anything but 0): read both flags strictly and say so, once per block. Returns the Config, or None
    for QWEN_FAST_PRESTAGE_DIFF=0 (an audit flag without the lever is inert)."""
    on, audit = enabled(environ), audit_enabled(environ)
    if not on:
        return None
    _STATE['engaged'] += 1
    log_line('%s users=%d audit=%d' % (ENGAGED_MARKER, getattr(block, 'users', 0), int(audit)))
    return Config(audit)


_MEMCMP = []


def memcmp():
    """libc's memcmp as a ctypes function, or False where the process has none (then same_bits compares with torch). Looked up once."""
    if not _MEMCMP:
        try:
            import ctypes

            function = ctypes.CDLL(None).memcmp
            function.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
            function.restype = ctypes.c_int
            _MEMCMP.append(function)
        except Exception:                                   # no libc symbol, no ctypes: the torch comparison below is the same answer
            _MEMCMP.append(False)
    return _MEMCMP[0]


def same_bits(left, right):
    """Whether two host tensors hold the same dtype, shape and the same bit pattern in every element (+0 differs from -0, a NaN equals only a NaN of the same payload). Two dense CPU
    tensors are compared byte for byte with memcmp (about 3 us a small tensor where torch.equal is 7, the whole 181-tensor staging list in 0.5 ms where torch.equal and a bit view take
    1.8); anything else is compared by value after viewing floating point values as the integers of their own width, which is the same answer."""
    import torch

    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    compare = memcmp()
    if compare and left.is_cpu and right.is_cpu and left.is_contiguous() and right.is_contiguous():
        return compare(left.data_ptr(), right.data_ptr(), left.numel() * left.element_size()) == 0
    if left.is_floating_point() or left.is_complex():
        wide = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64, 16: torch.int64}[left.element_size()]
        left, right = left.contiguous(), right.contiguous()
        if left.is_complex():
            left, right = torch.view_as_real(left), torch.view_as_real(right)
        return bool(torch.equal(left.view(wide), right.view(wide)))
    return bool(torch.equal(left, right))


class Resident:
    """What a block's staging destinations hold on the device after its last own write: `destinations` as the fixture's staging list gave them, `values[i]`
    the (host tensor, dtype, layout) written into destination i (None where nothing is known: the tokens, which the pre-stage never writes), and the epochs
    (global, and the fixture's own) AFTER the write's bump, which a later pre-stage must find unmoved."""

    __slots__ = ('epoch', 'local', 'destinations', 'values', 'round')

    def __init__(self, epoch_value, local_value, destinations, values, round_number):
        self.epoch, self.local, self.round = epoch_value, local_value, round_number
        self.destinations, self.values = tuple(destinations), list(values)


def take_resident(prestage):
    """At the top of a pre-stage, BEFORE its epoch bump: (resident, None) when the block's last own write still stands (no other writer moved either epoch),
    else (None, why). The block's `resident` is forgotten either way: while the new write is made the device holds an unknown mixture."""
    import verify_prestage

    resident, prestage.resident = prestage.resident, None
    if _STATE['off'] is not None:
        return None, 'latched:%s' % _STATE['off']
    if resident is None:
        return None, 'no-resident'
    if resident.epoch != verify_prestage.epoch():
        return None, 'epoch:%s' % verify_prestage.last_bump()
    if resident.local != verify_prestage.local_epoch(prestage.block.fixture):
        return None, 'epoch:%s' % verify_prestage.last_local_bump(prestage.block.fixture)
    return resident, None


def plan(resident, why, values, indices):
    """(indices to write, path, reason): the destinations of `indices` (the pre-staged ones) whose new value is not bit-equal to the resident one, or all of them
    when there is no usable resident or the destination list is not the resident's. Never raises: the lever's own failure is a full pre-stage and a latch."""
    if resident is None:
        COUNTS['full'] += 1
        return list(indices), 'full', why
    try:
        if len(resident.destinations) != len(values) or any(kept is not value[0] for kept, value in zip(resident.destinations, values)):
            COUNTS['full'] += 1
            return list(indices), 'full', 'destinations'
        changed = []
        for index in indices:
            kept, value = resident.values[index], values[index]
            if kept is None or kept[1] != value[2] or kept[2] != value[3] or not same_bits(kept[0], value[1]):
                changed.append(index)
    except BaseException as failure:
        latch('plan:%s' % type(failure).__name__)
        COUNTS['full'] += 1
        return list(indices), 'full', 'plan-failed'
    COUNTS['diff'] += 1
    return changed, 'diff', '-'


def resident_after_prestage(prestage, snapshot):
    """The block's resident after a pre-stage wrote what the snapshot holds (and left the rest as it was): the snapshot's values, epochs as the snapshot took them."""
    return Resident(snapshot.epoch, snapshot.local, snapshot.destinations, snapshot.values, snapshot.round)


def resident_after_verify(prestage, snapshot, values, changed, round_number):
    """The block's resident after the verify-time write (call it after the write's bump and after any audit that restaged the round). `snapshot` is the pre-stage
    the write diffed against, `values` the verify-time (destination, value, dtype, layout) list the diff computed (None for a keyed write, which wrote only the
    tokens) and `changed` the indices it wrote. The unchanged destinations keep the SNAPSHOT's values: those are the bytes that are on the device."""
    import verify_prestage

    if snapshot is None:
        return None
    kept = list(snapshot.values)
    if values is not None:
        for index in changed:
            kept[index] = values[index][1:]
    return Resident(verify_prestage.epoch(), verify_prestage.local_epoch(prestage.block.fixture), snapshot.destinations, kept, round_number)


def note_round(prestage, round_number, path, reason, total, written):
    import verify_prestage

    COUNTS['written'] += written
    COUNTS['skipped'] += total - written
    log_line('%s block=%s round=%d path=%s reason=%s total=%d written=%d skipped=%d' % (
        ROUND_MARKER, verify_prestage.block_label(prestage.block), round_number, path, str(reason).replace(' ', '_')[:120], total, written, total - written))


def audit_round(prestage, values, indices, written, writer, readers):
    """The audit twin, right after a pre-stage write: every pre-staged destination read back from every chip against the full value. Returns the mismatches
    ('<index>.<chip>' strings). On a difference the round is repaired (every pre-staged destination written in full), the mismatch is logged and the lever latches off."""
    import verify_prestage

    block = prestage.block
    operations = block.operations
    mismatched, checked = [], 0
    for index in indices:
        for chip, shard in enumerate(operations.get_device_tensors(values[index][0])):
            checked += 1
            if not verify_prestage.same_readback(values[index][1], operations.to_torch(shard)):
                mismatched.append('%d.%d' % (index, chip))
    COUNTS['audited'] += checked
    COUNTS['mismatches'] += len(mismatched)
    label = verify_prestage.block_label(block)
    if mismatched:
        log_line('%s block=%s round=%d checked=%d skipped=%d mismatches=%d at=%s exact=False' % (
            MISMATCH_MARKER, label, block.rounds + 1, checked, len(indices) - len(written), len(mismatched), ','.join(mismatched[:8])))
        prestage.inflight = writer(block.operations, block.model, values, readers, indices=list(indices), fence=False, poison=False)
        latch('audit_mismatch')
    else:
        log_line('%s block=%s round=%d checked=%d skipped=%d mismatches=0 exact=True' % (
            AUDIT_MARKER, label, block.rounds + 1, checked, len(indices) - len(written)))
    return mismatched
