"""Mutation tests for the safety checks of CardLink and P2PIO (design 2.8, Bounds: "mutation tests cover off-by-one, wrong bank, wrong card and a stale epoch").

A battery of behavioural checks runs against the real modules (it must pass) and against copies of their source with ONE check weakened at a time (each
must fail). A mutant that survives means a check that no test would miss if it were broken. The copies are executed from the source text, so the
mutations are exactly the strings listed here and nothing about the real files changes. No vLLM, no numpy, no device: stdlib only."""

import logging
import os
import sys
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
for entry in (HERE, PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fake_card as fc  # noqa: E402

logging.getLogger('ttkvp2p').addHandler(logging.NullHandler())

PAGE = 4096
SLOT = fc.SMALL_SLOT
CARDLINK_PATH = os.path.join(PACKAGE_ROOT, 'ttkvp2p', 'cardlink.py')
P2PIO_PATH = os.path.join(PACKAGE_ROOT, 'ttkvp2p', 'p2pio.py')


def read(path):
    with open(path) as handle:
        return handle.read()


def load_variant(cardlink_source, p2pio_source):
    """The two modules, executed from the given source, wired to each other (and to nothing else in sys.modules afterwards)."""
    names = ('ttkvp2p.cardlink', 'ttkvp2p.p2pio')
    saved = {name: sys.modules.get(name) for name in names}
    try:
        cl_mod = types.ModuleType('ttkvp2p.cardlink')
        cl_mod.__file__ = CARDLINK_PATH
        sys.modules['ttkvp2p.cardlink'] = cl_mod
        exec(compile(cardlink_source, CARDLINK_PATH, 'exec'), cl_mod.__dict__)
        io_mod = types.ModuleType('ttkvp2p.p2pio')
        io_mod.__file__ = P2PIO_PATH
        sys.modules['ttkvp2p.p2pio'] = io_mod
        exec(compile(p2pio_source, P2PIO_PATH, 'exec'), io_mod.__dict__)
        return cl_mod, io_mod
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def must_raise(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except kind:
        return
    except Exception as error:
        raise AssertionError('%s raised %s instead of %s' % (getattr(function, '__name__', function), type(error).__name__, kind.__name__))
    raise AssertionError('%s did not raise %s' % (getattr(function, '__name__', function), kind.__name__))


class Scenario(object):
    """A fresh environment that is always cleaned up."""

    def __init__(self, modules, **kwargs):
        self.env = fc.Env(modules=modules, **kwargs)

    def __enter__(self):
        return self.env

    def __exit__(self, *exc):
        self.env.close()
        return False


def prime(e, count, tag='p'):
    for slot in range(count):
        e.drive_write(e.drive_off(slot), e.drive_object((tag, slot)))


# --------------------------------------------------------------------------------------------------------------------
# The battery: each check raises AssertionError when the system misbehaves
# ---------------------------------------------------------------------------------------------------------------------

def check_slot_view_bounds(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), start=False) as e:
        link = e.rig.open_links()[0]
        for args in ((0, -1, 1), (0, 0, SLOT + 1), (0, SLOT, 1), (0, SLOT - 1, 2), (0, 0, 0), (0, 0, -1), (99, 0, 1), (-1, 0, 1)):
            must_raise(cl_mod.BoundsViolation, link.slot_view, *args)
        for obj in (-1, SLOT):
            must_raise(cl_mod.BoundsViolation, link.expected_slice_offset, 0, obj)
        for good in ((0, 0, SLOT), (7, SLOT - 1, 1)):
            for view in link.slot_view(*good):
                view.release()


def check_aim_is_refused_while_busy(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), start=False) as e:
        link = e.rig.open_links()[0]
        link.aim(2)
        link.begin_io(2, link.aim_epoch)
        must_raise(cl_mod.AimWhileBusy, link.aim, 3)
        assert link.aimed_bank == 2


def check_begin_io_is_refused_when_stale(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), start=False) as e:
        link = e.rig.open_links()[0]
        link.aim(2)
        epoch = link.aim_epoch
        must_raise(cl_mod.StaleAim, link.begin_io, 3, epoch)         # another bank
        must_raise(cl_mod.StaleAim, link.begin_io, 2, epoch - 1)     # an older epoch
        link.aim(2)
        must_raise(cl_mod.StaleAim, link.begin_io, 2, epoch)         # the same bank, aimed again since


def check_a_window_that_does_not_move_is_found_on_aim(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), start=False) as e:
        link = e.rig.open_links()[1]
        e.rig.boards[1].ignore_configure = True
        must_raise(cl_mod.CanaryMismatch, link.aim, 4)


def check_open_proves_every_canary_before_data(cl_mod, io_mod):
    rig = fc.Rig(cardlink_module=cl_mod)
    try:
        rig.boards[0].alias = {3: 2}
        must_raise(cl_mod.CanaryMismatch, rig.make_link(0).open, rig.staging_info())
    finally:
        rig.close()


def check_canaries_carry_a_fresh_nonce(cl_mod, io_mod):
    rig = fc.Rig(cardlink_module=cl_mod)
    try:
        first, second = rig.make_link(0), rig.make_link(0)
        assert first._nonce != second._nonce
        info = rig.staging_info()
        first.open(info)
        page_one = rig.boards[0].content(0)[rig.canary_offset(0):rig.canary_offset(0) + PAGE]
        first.close()
        second.open(info)
        page_two = rig.boards[0].content(0)[rig.canary_offset(0):rig.canary_offset(0) + PAGE]
        assert page_one != page_two, 'a canary left in the card by an earlier open would satisfy this one'
    finally:
        rig.close()


def check_locate_knows_where_a_buffer_is(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), start=False) as e:
        links = e.rig.open_links()
        own = links[0].slot_view(2, 1, 1)[0]
        foreign = links[1].slot_view(2, 1, 1)[0]
        assert links[0].locate(own) == e.rig.data_offset(2, 1)
        assert links[0].locate(foreign) is None, 'another card\'s buffer'
        assert links[0].locate(bytearray(100)) is None, 'host memory'


def check_host_checks_refuse(cl_mod, io_mod):
    rig = fc.Rig(cardlink_module=cl_mod, bank_address=0xFF000000 - 8 * PAGE + 64, region_base=0xFF000000 - 8 * PAGE)
    try:
        must_raise(cl_mod.CardLinkUnavailable, rig.make_link(0).open, rig.staging_info())       # the top 16 MiB of the window
    finally:
        rig.close()
    rig = fc.Rig(cardlink_module=cl_mod)
    try:
        rig.kmd.files['%s/%s/p2pmem/available' % (fc.PCI_ROOT, fc.BDFS[0])] = '4096'
        must_raise(cl_mod.CardLinkUnavailable, rig.make_link(0).open, rig.staging_info())       # slice already allocated
    finally:
        rig.close()
    rig = fc.Rig(cardlink_module=cl_mod)
    try:
        info = rig.staging_info()
        info['devices'][1]['pci_bus_id'] = fc.BDFS[0]
        must_raise(cl_mod.CardLinkUnavailable, rig.make_link(0).open, info)                       # two mesh devices for one card
    finally:
        rig.close()


def check_direction_of_banks(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.read_piece, e.drive_off(0), 0, 1)               # a restore into a spill bank
    with Scenario((cl_mod, io_mod)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.write_subrun, e.drive_off(0), 2, 1)             # a spill from a restore bank
    with Scenario((cl_mod, io_mod)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.read_piece, e.drive_off(0), 9, 1)               # not a staging bank


def check_request_bounds(cl_mod, io_mod):
    cases = [('read_piece', (None, 2, 3), {'first_obj': SLOT - 3 + 1}), ('write_subrun', (None, 0, 3), {'first_obj': SLOT - 3 + 1}),
             ('read_piece', (None, 2, 1), {'first_obj': -1}), ('read_piece', (None, 2, 0), {}), ('read_piece', (None, 2, SLOT + 1), {}),
             ('write_subrun', (None, 0, 5), {})]
    for name, args, kwargs in cases:
        with Scenario((cl_mod, io_mod), config_kwargs=dict(spill_subrun_objects=4)) as e:
            must_raise(cl_mod.BoundsViolation, getattr(e.io, name), e.drive_off(0), *args[1:], **kwargs)
            assert e.ops.calls == [], 'a syscall was issued'


def check_drive_range(cl_mod, io_mod):
    for off in (0, 2 * PAGE, fc.Env.DATA_OFF - PAGE, fc.Env.DATA_OFF + 1, fc.Env.DATA_OFF + 512):
        with Scenario((cl_mod, io_mod)) as e:
            must_raise(cl_mod.BoundsViolation, e.io.read_piece, off, 2, 1)
            assert e.ops.calls == []
    with Scenario((cl_mod, io_mod)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.read_piece, e.data_hi - e.object_bytes + PAGE, 2, 1)       # one page past the end
        must_raise(cl_mod.BoundsViolation, e.io.read_piece, e.data_hi, 2, 1)
    with Scenario((cl_mod, io_mod)) as e:
        assert e.io.read_piece(e.data_hi - e.object_bytes, 2, 1).result(2) == e.object_bytes       # the last object of the region is fine


def planned(e, io_mod, bank=2, first=1, n=2):
    for link in e.links:
        link.aim(bank)
    req = io_mod._Request('read', bank, 0.0)
    call = io_mod._Call(req, first, n, e.drive_off(0))
    call.tokens = tuple(link.aim_epoch for link in e.links)
    return call, e.io._build_iovecs(call)


def check_iovec_off_by_one(cl_mod, io_mod):
    for delta in (-1, 1):
        with Scenario((cl_mod, io_mod)) as e:
            call, iovs = planned(e, io_mod, first=2)
            iovs[1] = e.links[1].slot_view(2, 2 + delta, 1)[0]
            must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)


def check_iovec_wrong_card(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        call, iovs = planned(e, io_mod)
        iovs[0], iovs[1] = iovs[1], iovs[0]
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)


def check_iovec_wrong_bank(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), rig_kwargs=dict(per_bank_offset=0x10000)) as e:        # addresses differ per bank: the view says which
        call, iovs = planned(e, io_mod)
        iovs[3] = e.links[3].slot_view(3, 1, 1)[0]
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)
    with Scenario((cl_mod, io_mod)) as e:                                                    # lockstep: the aim says which
        call, iovs = planned(e, io_mod)
        e.links[1].aim(5)
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)
    with Scenario((cl_mod, io_mod)) as e:                                                    # a call planned for bank 2 in an epoch whose windows are on bank 5
        call, iovs = planned(e, io_mod, bank=5)
        call.req.bank = 2
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)


def check_iovec_stale_epoch(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        call, iovs = planned(e, io_mod)
        e.links[2].aim(2)
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)


def check_iovec_shape(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        call, iovs = planned(e, io_mod)
        iovs.pop()
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)
    with Scenario((cl_mod, io_mod)) as e:
        call, iovs = planned(e, io_mod)
        iovs[3] = iovs[3][:-PAGE]
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)
    with Scenario((cl_mod, io_mod)) as e:
        call, iovs = planned(e, io_mod)
        e.links[0].shard = 1
        must_raise(cl_mod.BoundsViolation, e.io._check_iovecs, call, iovs)


def check_a_card_moved_behind_the_layers_back(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        e.io.read_piece(e.drive_off(0), 2, 1).result(2)
        e.links[3].aim(2)
        error = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1).exception(2)
        assert error is not None and not isinstance(error, TimeoutError), 'the stale call was issued'
        assert len(e.ops.calls) == 1, 'a call ran on a window whose epoch had moved'
        assert e.io.dead and e.runtime.latched, 'a card that moved behind the layer\'s back must stop the layer and latch the tier'


def check_round_trip_and_iovec_order(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        for j in range(3):
            for d in range(4):
                e.put(d, 0, 1 + j, e.card_object(d, ('o', j)))
        assert e.io.write_subrun(e.drive_off(7), 0, 3, first_obj=1).result(2) == 3 * e.object_bytes
        for j in range(3):
            assert e.drive_read(e.drive_off(7) + j * e.object_bytes, e.object_bytes) == e.drive_object(('o', j)), 'drive layout, object %d' % j
        assert e.io.read_piece(e.drive_off(7), 3, 3, first_obj=2).result(2) == 3 * e.object_bytes
        for d in range(4):
            for j in range(3):
                assert e.get(d, 3, 2 + j) == e.card_object(d, ('o', j)), 'card %d object %d' % (d, j)
        assert e.rig.reaim_while_busy == []


def check_short_io(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        prime(e, 3, 's')
        e.ops.shorts = [PAGE, 5 * PAGE, 4 * PAGE]
        e.io.read_piece(e.drive_off(0), 2, 3).result(2)
        for d in range(4):
            for j in range(3):
                assert e.get(d, 2, j) == e.card_object(d, ('s', j))
    with Scenario((cl_mod, io_mod)) as e:
        prime(e, 1, 'z')
        e.ops.zero_next = 1
        error = e.io.read_piece(e.drive_off(0), 2, 1).exception(2)
        assert isinstance(error, io_mod.P2PShortIO), 'EOF must fail the call: %r' % (error,)
    with Scenario((cl_mod, io_mod)) as e:
        prime(e, 1, 'z')
        e.ops.over_next = 1
        error = e.io.read_piece(e.drive_off(0), 2, 1).exception(2)
        assert isinstance(error, io_mod.P2PIOError), 'a count beyond the call must fail it: %r' % (error,)


def check_re_aim_only_when_idle(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        (release, entered), = e.ops.hold_next(1)
        prime(e, 2, 'i')
        first = e.io.read_piece(e.drive_off(0), 2, 1)
        assert entered.wait(2)
        before = [len(board.configures) for board in e.rig.boards]
        second = e.io.read_piece(e.drive_off(1), 3, 1)
        time.sleep(0.08)
        assert [len(board.configures) for board in e.rig.boards] == before, 'the window moved with a bio in flight'
        release.set()
        first.result(2)
        second.result(2)
        assert e.rig.reaim_while_busy == []
        assert [c['aimed'] for c in e.ops.calls] == [[2] * 4, [3] * 4]
        for d in range(4):
            assert e.get(d, 2, 0) == e.card_object(d, ('i', 0)) and e.get(d, 3, 0) == e.card_object(d, ('i', 1))


def check_a_call_runs_on_its_own_bank(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        prime(e, 2, 'b')
        a = e.io.read_piece(e.drive_off(0), 2, 1)
        a.result(2)
        b = e.io.read_piece(e.drive_off(1), 4, 1)
        b.result(2)
        assert [c['aimed'] for c in e.ops.calls] == [[2] * 4, [4] * 4], [c['aimed'] for c in e.ops.calls]


def check_restore_goes_first(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        (release, entered), = e.ops.hold_next(1)
        s1 = e.io.write_subrun(e.drive_off(0), 0, 1)
        assert entered.wait(2)
        s2 = e.io.write_subrun(e.drive_off(2), 1, 1)
        r1 = e.io.read_piece(e.drive_off(10), 3, 1)
        time.sleep(0.05)
        release.set()
        for future in (s1, s2, r1):
            future.result(2)
        assert [c['kind'] for c in e.ops.calls] == ['write', 'read', 'write'], [c['kind'] for c in e.ops.calls]


def check_queue_depth_and_threads(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod), config_kwargs=dict(io_queue_depth_objects=4)) as e:
        (release1, entered1), (release2, entered2) = e.ops.hold_next(2)
        first = e.io.read_piece(e.drive_off(0), 2, 4)
        assert entered1.wait(2)
        second = e.io.read_piece(e.drive_off(4), 2, 4)
        assert not entered2.wait(0.15), 'the queue depth is in objects'
        release1.set()
        release2.set()
        first.result(2)
        second.result(2)
    with Scenario((cl_mod, io_mod), config_kwargs=dict(io_threads=1)) as e:
        (release1, entered1), (release2, entered2) = e.ops.hold_next(2)
        first = e.io.read_piece(e.drive_off(0), 2, 1)
        assert entered1.wait(2)
        second = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1)
        assert not entered2.wait(0.15), 'one thread, one call in flight'
        release1.set()
        release2.set()
        first.result(2)
        second.result(2)
        assert e.io.stats()['inflight_calls_max'] == 1, 'the layer counted %d calls in flight on one thread' % e.io.stats()['inflight_calls_max']


def check_dma_canary_detects_a_drive_that_does_nothing(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        e.ops.on_call = lambda kind, record: setattr(e.ops, 'swallow_next', 1) if kind == 'read' else None
        must_raise(io_mod.DmaCanaryError, e.io.dma_canary)
    with Scenario((cl_mod, io_mod)) as e:
        e.io.dma_canary()
    with Scenario((cl_mod, io_mod), io_kwargs=dict(canary_offset=4 * PAGE)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.dma_canary)
        assert e.ops.calls == [], 'the canary wrote outside its block'


def check_a_violation_latches_and_stops_the_layer(cl_mod, io_mod):
    with Scenario((cl_mod, io_mod)) as e:
        must_raise(cl_mod.BoundsViolation, e.io.read_piece, e.drive_off(0), 0, 1)
        assert e.runtime.latched, 'the tier was not latched'
        assert e.io.dead
        assert isinstance(e.io.read_piece(e.drive_off(0), 2, 1).exception(2), io_mod.P2PDead)


CHECKS = [value for name, value in sorted(globals().items()) if name.startswith('check_') and callable(value)]


def battery(cl_mod, io_mod):
    """Names of the checks that failed (an exception of any kind counts as a failure, a hang as a timeout inside the check)."""
    failed = []
    for check in CHECKS:
        try:
            check(cl_mod, io_mod)
        except BaseException as error:        # noqa: B902  (a mutant may break anything)
            failed.append('%s: %s: %s' % (check.__name__, type(error).__name__, str(error)[:120]))
    return failed


# --------------------------------------------------------------------------------------------------------------------
# The mutants: (file, old text, new text). Each old text must occur exactly once.
# ---------------------------------------------------------------------------------------------------------------------

CARDLINK_MUTANTS = [
    ('slot_view lets the range run one object past the slot', "if first_obj < 0 or first_obj + n_obj > self.slot_objects:", "if first_obj < 0 or first_obj + n_obj > self.slot_objects + 1:"),
    ('slot_view accepts an empty range', "if n_obj < 1:\n            raise BoundsViolation('slot_view of", "if n_obj < 0:\n            raise BoundsViolation('slot_view of"),
    ('expected_slice_offset accepts the object after the last', "obj < 0 or obj >= self.slot_objects", "obj < 0 or obj > self.slot_objects"),
    ('expected_slice_offset accepts a negative object', "obj < 0 or obj >= self.slot_objects", "obj >= self.slot_objects"),
    ('aim is allowed with a bio in flight', "if self._inflight:\n                raise AimWhileBusy(", "if False:\n                raise AimWhileBusy("),
    ('aim does not compare the canary', "if got != self._canaries[bank][:CANARY_COMPARE_BYTES]:\n                self.stats['canary_mismatches'] += 1\n                self._poisoned",
     "if False:\n                self.stats['canary_mismatches'] += 1\n                self._poisoned"),
    ('begin_io ignores the epoch', "if self._aimed != bank or self._epoch != epoch:", "if self._aimed != bank:"),
    ('begin_io ignores the bank', "if self._aimed != bank or self._epoch != epoch:", "if self._epoch != epoch:"),
    ('locate accepts any address', "if offset < 0 or offset + nbytes > self._slice_size:", "if False:"),
    ('open does not prove the canary pages', "if got != self._canaries[bank]:", "if False:"),
    ('the top 16 MiB of the window is allowed', "if end > WINDOW_TOP:", "if False:"),
    ('a slice already allocated is accepted', "if info['core_available'] != info['core_size']:", "if False:"),
    ('two mesh devices for one card are accepted', "if len(found) != 1:", "if len(found) < 1:"),
    ('the canary has no nonce', "canary_page(self._nonce, self.bdf, bank)", "canary_page(b'', self.bdf, bank)"),
    ('the nonce is the same for every link', "self._nonce = nonce if nonce is not None else os.urandom(16)", "self._nonce = nonce if nonce is not None else b'fixed'"),
]

P2PIO_MUTANTS = [
    ('the direction of a bank is not checked', "if bank not in banks:", "if False:"),
    ('the object range may run one past the slot', "if first_obj < 0 or first_obj + n_obj > self.slot_objects:", "if first_obj < 0 or first_obj + n_obj > self.slot_objects + 1:"),
    ('a negative first object is accepted', "if first_obj < 0 or first_obj + n_obj > self.slot_objects:", "if first_obj + n_obj > self.slot_objects:"),
    ('an empty request is accepted', "        if n_obj < 1:\n            raise self._violation('%d objects' % n_obj)", "        if n_obj < 0:\n            raise self._violation('%d objects' % n_obj)"),
    ('a spill sub-run may be one object too long', "if kind == 'write' and n_obj > self.spill_subrun_objects:", "if kind == 'write' and n_obj > self.spill_subrun_objects + 1:"),
    ('an unaligned drive offset is accepted', "if drive_off % PAGE:", "if False:"),
    ('a drive offset below the data region is accepted', "if drive_off < lo or drive_off + nbytes > hi:", "if drive_off + nbytes > hi:"),
    ('a drive range past the data region is accepted', "if drive_off < lo or drive_off + nbytes > hi:", "if drive_off < lo:"),
    ('a drive range one page past the data region is accepted', "if drive_off < lo or drive_off + nbytes > hi:", "if drive_off < lo or drive_off + nbytes > hi + PAGE:"),
    ('the iovec location is not compared', "if got != want:", "if False:"),
    ('the aim epoch is not compared', "if link.aim_epoch != call.tokens[d]:", "if False:"),
    ('the aimed bank is not compared', "if link.aimed_bank != call.bank:", "if False:"),
    ('the iovec length is not compared', "if view.nbytes != self.shard_object_bytes:", "if False:"),
    ('the iovec count is not compared', "if len(iovs) != call.n * self.chips or len(iovs) > IOV_MAX:", "if False:"),
    ('the shard of a link is not compared', "if link.shard != d:", "if False:"),
    ('the windows are never re-aimed', "if self._epoch_bank != req.bank:", "if False:"),
    ('the windows are re-aimed with calls in flight', "if self._inflight_calls:\n                return None                      # drain first", "if False:\n                return None                      # drain first"),
    ('spill goes before restore', "queue = self._restore_q or self._spill_q", "queue = self._spill_q or self._restore_q"),
    ('the iovecs are card-major', "return [per_card[d][j] for j in range(call.n) for d in range(self.chips)]", "return [per_card[d][j] for d in range(self.chips) for j in range(call.n)]"),
    ('the queue depth is ignored', "if self._inflight_calls and self._inflight_objects + call.n > self._qd_objects:", "if False:"),
    ('the thread count is ignored', "if self._inflight_calls >= self._threads:", "if False:"),
    ('end of file is progress', "if count <= 0:\n                raise P2PShortIO(", "if count < 0:\n                raise P2PShortIO("),
    ('a count beyond the call is believed', "if count > total - done:", "if False:"),
    ('a bounds violation does not stop the layer', "self._die('P2PIO bounds violation: %s' % message)", "pass"),
    ('the DMA canary result is not compared', "if back != pattern:", "if False:"),
    ('the DMA canary may use any offset', "if self._canary_offset < DMA_CANARY_OFFSET or self._canary_offset + PAGE > DMA_CANARY_OFFSET + DMA_CANARY_BYTES or self._canary_offset % PAGE:", "if False:"),
    ('a fatal error does not stop the layer', "fatal = isinstance(error, (BoundsViolation, CardLinkError))", "fatal = False"),
]


class Baseline(unittest.TestCase):
    def test_the_real_modules_pass_the_whole_battery(self):
        cl_mod, io_mod = load_variant(read(CARDLINK_PATH), read(P2PIO_PATH))
        self.assertEqual(battery(cl_mod, io_mod), [])

    def test_the_battery_is_not_empty_and_covers_the_four_planted_mistakes(self):
        names = [check.__name__ for check in CHECKS]
        self.assertGreaterEqual(len(names), 25)
        for needed in ('check_iovec_off_by_one', 'check_iovec_wrong_bank', 'check_iovec_wrong_card', 'check_iovec_stale_epoch', 'check_re_aim_only_when_idle'):
            self.assertIn(needed, names)

    def test_every_mutation_applies_exactly_once(self):
        for label, old, new in CARDLINK_MUTANTS:
            self.assertEqual(read(CARDLINK_PATH).count(old), 1, label)
            self.assertNotEqual(old, new, label)
        for label, old, new in P2PIO_MUTANTS:
            self.assertEqual(read(P2PIO_PATH).count(old), 1, label)
            self.assertNotEqual(old, new, label)


class RedundantGuards(unittest.TestCase):
    """A guard that a second guard repeats cannot be mutated away on its own. These are kept as defence in depth and shown here to be redundant, not forgotten."""

    def test_slot_view_checks_the_first_object_twice(self):
        old = "if first_obj < 0 or first_obj + n_obj > self.slot_objects:"
        self.assertEqual(read(CARDLINK_PATH).count(old), 1)
        cl_mod, io_mod = load_variant(read(CARDLINK_PATH).replace(old, "if first_obj + n_obj > self.slot_objects:"), read(P2PIO_PATH))
        with Scenario((cl_mod, io_mod), start=False) as e:
            link = e.rig.open_links()[0]
            for first in (-1, -5):
                must_raise(cl_mod.BoundsViolation, link.slot_view, 0, first, 1)       # expected_slice_offset refuses a negative object all the same


class Mutants(unittest.TestCase):
    """One weakened check at a time; the battery must notice."""

    def killed(self, which, label, old, new):
        cl_source, io_source = read(CARDLINK_PATH), read(P2PIO_PATH)
        if which == 'cardlink':
            cl_source = cl_source.replace(old, new)
        else:
            io_source = io_source.replace(old, new)
        cl_mod, io_mod = load_variant(cl_source, io_source)
        failed = battery(cl_mod, io_mod)
        self.assertTrue(failed, 'SURVIVED: %s' % label)
        return failed

    def test_cardlink_mutants_are_all_killed(self):
        for label, old, new in CARDLINK_MUTANTS:
            with self.subTest(mutant=label):
                self.killed('cardlink', label, old, new)

    def test_p2pio_mutants_are_all_killed(self):
        for label, old, new in P2PIO_MUTANTS:
            with self.subTest(mutant=label):
                self.killed('p2pio', label, old, new)


if __name__ == '__main__':
    unittest.main()
