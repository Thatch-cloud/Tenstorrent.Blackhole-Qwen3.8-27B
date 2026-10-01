"""capture_plug: the zone, the sweep, the seal and the monitor, against a model of the allocator the plug relies on (first fit,
bottom-up, one address per bank), plus the control that shows the model reproduces the aliasing the plug removes.

Host only; nothing here needs a device."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capture_plug as plug  # noqa: E402

PAGE = plug.PAGE
TOTAL = 4000 * PAGE   # per bank


class Heap:
    """One bank's DRAM: bottom-up first-fit buffers and top-down kernel binaries, in bytes; addresses are the same in every bank."""

    def __init__(self, total=TOTAL):
        self.total = total
        self.used = {}   # address -> size

    def free_blocks(self):
        blocks, cursor = [], 0
        for address in sorted(self.used):
            if address > cursor:
                blocks.append((cursor, address - cursor))
            cursor = address + self.used[address]
        if cursor < self.total:
            blocks.append((cursor, self.total - cursor))
        return blocks

    def allocate(self, size):
        for address, room in self.free_blocks():
            if room >= size:
                self.used[address] = size
                return address
        raise RuntimeError('Out of Memory: no block of %d' % size)

    def allocate_top(self, size):
        for address, room in reversed(self.free_blocks()):
            if room >= size:
                at = address + room - size
                self.used[at] = size
                return at
        raise RuntimeError('Out of Memory (top-down): no block of %d' % size)

    def free(self, address):
        del self.used[address]

    def largest_free(self):
        return max((room for _, room in self.free_blocks()), default=0)

    def free_bytes(self):
        return sum(room for _, room in self.free_blocks())


class FakeMemory:
    """The Memory the plug is written against, over a Heap."""

    def __init__(self, heap, kind='DRAM'):
        self.heap, self.kind = heap, kind

    def largest_free(self):
        return self.heap.largest_free()

    def free(self):
        return self.heap.free_bytes()

    def allocate(self, per_bank):
        return plug.Block(None, self.heap.allocate(per_bank), per_bank, self.kind)

    def release(self, block):
        self.heap.free(block.address)


def sizes(count, size):
    return [size] * count


class Capture:
    """What a trace capture does to the heap: persistent buffers, and temporaries that are allocated and freed again. The
    temporaries' extents are what a replay rewrites."""

    def __init__(self, heap):
        self.heap, self.persistent, self.temporary = heap, [], []

    def run(self, persistent=(), temporaries=()):
        live = []
        for size in temporaries:
            address = self.heap.allocate(size)
            live.append((address, size))
        for size in persistent:
            self.persistent.append((self.heap.allocate(size), size))
        for address, size in live:
            self.heap.free(address)
            self.temporary.append((address, address + size))
        return self


def hits(address, size, extents):
    return any(address < hi and lo < address + size for lo, hi in extents)


def fragmented_heap():
    """A heap with low holes (earlier work freed): buffers at the bottom with gaps between them."""
    heap = Heap()
    held = [heap.allocate(40 * PAGE) for _ in range(6)]
    for index in (1, 3):
        heap.free(held[index])
    return heap


class ConfigTests(unittest.TestCase):
    def test_off_unless_exactly_1(self):
        for value in (None, '', '0'):
            self.assertIsNone(plug.config({} if value is None else {plug.FLAG: value}))
        with self.assertRaises(ValueError):
            plug.config({plug.FLAG: 'true'})

    def test_defaults_and_overrides_are_per_bank_bytes(self):
        found = plug.config({plug.FLAG: '1'})
        self.assertEqual((found['leave'], found['reserve'], found['engine_leave'], found['min_free']),
                         (512 * plug.MB, 256 * plug.MB, 256 * plug.MB, 128 * plug.MB))
        self.assertFalse(found['engines'] or found['l1'])
        found = plug.config({plug.FLAG: '1', plug.LEAVE_FLAG: '300', plug.ENGINES_FLAG: '1', plug.L1_LEAVE_FLAG: '64'})
        self.assertEqual((found['leave'], found['engines'], found['l1'], found['l1_leave']), (300 * plug.MB, True, False, 64 * plug.KB))
        for flag in (plug.LEAVE_FLAG, plug.RESERVE_FLAG, plug.MIN_FREE_FLAG):
            for bad in ('0', '-1', '1.5', 'x'):
                with self.assertRaises(ValueError, msg=(flag, bad)):
                    plug.config({plug.FLAG: '1', flag: bad})
        with self.assertRaises(ValueError):
            plug.config({plug.FLAG: '1', plug.ENGINES_FLAG: '2'})

    def test_the_l1_zone_is_refused_until_it_is_modelled_for_top_down_allocation(self):
        with self.assertRaisesRegex(ValueError, 'top-down'):
            plug.config({plug.FLAG: '1', plug.L1_FLAG: '1'})
        self.assertFalse(plug.config({plug.FLAG: '1', plug.L1_FLAG: '0'})['l1'])
        self.assertFalse(plug.config({plug.FLAG: '1'})['l1'])
        self.assertIsNone(plug.config({plug.L1_FLAG: '1'}), 'the flag alone, without the plug, stays inert')


class SweepTests(unittest.TestCase):
    def test_every_hole_below_the_limit_is_plugged_and_the_region_above_is_untouched(self):
        heap = fragmented_heap()
        memory = FakeMemory(heap)
        start = plug.probe_start(memory)
        before = heap.free_bytes()
        plugs, iterations = plug.sweep(memory, start)
        self.assertGreater(len(plugs), 0)
        self.assertTrue(all(block.end <= start for block in plugs))
        plug.verify_sealed(memory, start)
        self.assertEqual(heap.free_bytes(), before - 80 * PAGE, 'exactly the two 40-page holes were filled')
        self.assertEqual(plug.probe_start(memory), start, 'the main region did not move')
        self.assertLess(iterations, 100)

    def test_a_hole_of_awkward_size_is_filled_exactly_by_halving(self):
        heap = Heap()
        a, b, c = heap.allocate(7 * PAGE), heap.allocate(13 * PAGE), heap.allocate(9 * PAGE)
        heap.free(a)
        heap.free(c)    # holes of 7 and (9 + the region above) pages
        memory = FakeMemory(heap)
        limit = c + 9 * PAGE
        plugs, _ = plug.sweep(memory, limit)
        self.assertEqual(sum(block.per_bank for block in plugs), 16 * PAGE)
        plug.verify_sealed(memory, limit)

    def test_the_iteration_cap_fails_closed(self):
        heap = Heap()
        held = [heap.allocate(PAGE) for _ in range(60)]
        for address in held[::2]:
            heap.free(address)
        with self.assertRaisesRegex(plug.CapturePlugError, 'iterations'):
            plug.sweep(FakeMemory(heap), held[-1] + PAGE, max_iterations=10)

    def test_a_surviving_hole_fails_the_verify(self):
        heap = Heap()
        first, second = heap.allocate(8 * PAGE), heap.allocate(8 * PAGE)
        heap.free(first)
        with self.assertRaisesRegex(plug.CapturePlugError, 'hole remains'):
            plug.verify_sealed(FakeMemory(heap), second)


class LowestHoleTests(unittest.TestCase):
    def test_the_lowest_hole_is_found_exactly_and_the_probes_leave_nothing_behind(self):
        heap = Heap()
        blocks = [heap.allocate(size * PAGE) for size in (3, 50, 7, 21, 9)]
        for index in (1, 3):
            heap.free(blocks[index])
        baseline = dict(heap.used)
        self.assertEqual(plug.lowest_hole(FakeMemory(heap)), (blocks[1], 50 * PAGE))
        self.assertEqual(heap.used, baseline)
        heap.free(blocks[0])
        self.assertEqual(plug.lowest_hole(FakeMemory(heap)), (0, (3 + 50) * PAGE))

    def test_the_last_hole_is_the_main_region(self):
        heap = Heap()
        heap.allocate(10 * PAGE)
        self.assertEqual(plug.lowest_hole(FakeMemory(heap)), (10 * PAGE, TOTAL - 10 * PAGE))


class ZoneTests(unittest.TestCase):
    def zone(self, heap=None, leave=1200 * PAGE, reserve=800 * PAGE, ceiling=None):
        heap = heap or fragmented_heap()
        return heap, plug.Zone(FakeMemory(heap), leave, reserve, name='packed', ceiling=ceiling)

    def test_the_captures_run_inside_the_zone_and_the_ballast_is_the_only_thing_below_it(self):
        heap, zone = self.zone()
        zone.open()
        self.assertEqual(zone.hi - zone.lo, 1200 * PAGE)
        capture = Capture(heap).run(persistent=[60 * PAGE, 30 * PAGE], temporaries=[300 * PAGE, 100 * PAGE, 450 * PAGE])
        self.assertTrue(all(zone.lo <= lo and hi <= zone.hi for lo, hi in capture.temporary + [(a, a + s) for a, s in capture.persistent]))
        zone.seal()
        zone.release()
        self.assertEqual(zone.state, 'released')

    def test_after_seal_no_later_allocation_lands_in_a_freed_temporary_or_anywhere_in_the_zone(self):
        heap, zone = self.zone()
        zone.open()
        capture = Capture(heap).run(persistent=[60 * PAGE], temporaries=[300 * PAGE, 100 * PAGE, 450 * PAGE, 20 * PAGE])
        zone.seal()
        zone.release()
        memory, landed = FakeMemory(heap), []
        for size in [7, 120, 33, 500, 64, 64, 9, 300, 200, 1, 1, 80]:
            try:
                landed.append((heap.allocate(size * PAGE), size * PAGE))
            except RuntimeError:
                break
        self.assertGreaterEqual(len(landed), 5)
        for address, size in landed:
            self.assertFalse(hits(address, size, capture.temporary), 'a later buffer sits in a freed temporary')
            self.assertLessEqual(address + size, zone.lo, 'a later buffer sits in the zone')

    def test_the_control_without_the_plug_puts_a_later_buffer_in_a_freed_temporary(self):
        heap = fragmented_heap()
        capture = Capture(heap).run(persistent=[60 * PAGE], temporaries=[300 * PAGE, 100 * PAGE, 450 * PAGE, 20 * PAGE])
        victim = heap.allocate(120 * PAGE)
        self.assertTrue(hits(victim, 120 * PAGE, capture.temporary), 'the model must reproduce the aliasing the plug removes')

    def test_kernel_binaries_grow_down_into_the_reserve_and_then_fail_instead_of_overwriting(self):
        heap, zone = self.zone()
        zone.open()
        capture = Capture(heap).run(persistent=[60 * PAGE], temporaries=[300 * PAGE, 450 * PAGE])
        zone.seal()
        zone.release()
        binaries = []
        with self.assertRaisesRegex(RuntimeError, 'top-down'):
            while True:
                size = 25 * PAGE
                binaries.append((heap.allocate_top(size), size))
        self.assertGreater(len(binaries), 10)
        for address, size in binaries:
            self.assertFalse(hits(address, size, capture.temporary), 'a kernel binary sits in a freed temporary')
            self.assertFalse(address < zone.hi and zone.lo < address + size, 'a binary sits in the plugged zone')

    def test_a_zone_too_small_for_the_reserve_is_refused(self):
        heap, zone = self.zone(leave=2500 * PAGE, reserve=2500 * PAGE)
        with self.assertRaisesRegex(plug.CapturePlugError, 'no room'):
            zone.open()

    def test_a_capture_that_spills_past_the_zone_seals_up_to_the_spill_and_verify_extents_names_it(self):
        heap, zone = self.zone(leave=200 * PAGE, reserve=800 * PAGE)
        zone.open()
        Capture(heap).run(persistent=[260 * PAGE], temporaries=[50 * PAGE])
        zone.seal()
        self.assertGreater(zone.hi, zone.lo + 200 * PAGE, 'sealed up to where the persistent buffers end')
        with self.assertRaisesRegex(plug.CapturePlugError, 'ends above the zone top'):
            zone.verify_extents([(zone.lo, zone.hi + PAGE, 'DRAM')])
        zone.verify_extents([(zone.lo, zone.hi, 'DRAM'), (0, 1 << 60, 'L1')])

    def test_a_hole_in_the_zone_larger_than_what_is_left_above_it_does_not_hide_a_spill(self):
        """The zone's biggest hole is bigger than the free memory above the zone, so the main region does not start above the zone's
        top. Temporaries that spilled past the top and were freed above it must still be plugged."""
        heap = Heap()
        zone = plug.Zone(FakeMemory(heap), 1000 * PAGE, 300 * PAGE, name='packed')
        zone.open()
        capture = Capture(heap).run(persistent=[40 * PAGE], temporaries=[900 * PAGE, 200 * PAGE, 50 * PAGE])
        spilled = [(lo, hi) for lo, hi in capture.temporary if hi > zone.hi]
        self.assertTrue(spilled, 'the model must put a freed temporary above the zone top')
        free_above_top = [(a, r) for a, r in heap.free_blocks() if a >= zone.hi]
        self.assertLess(max(r for _, r in free_above_top), 1000 * PAGE)
        self.assertGreater(max(r for a, r in heap.free_blocks() if a < zone.hi), max(r for _, r in free_above_top) // 2)
        zone.seal()
        zone.release()
        self.assertGreater(zone.hi, spilled[0][1] - 1, 'sealed up to the end of the spilled temporaries')
        victim = None
        for size in (60, 120, 33, 7, 300, 150):
            try:
                victim = (heap.allocate(size * PAGE), size * PAGE)
            except RuntimeError:
                continue
            self.assertFalse(hits(victim[0], victim[1], capture.temporary), 'a later buffer sits in a freed temporary')
            self.assertLessEqual(victim[0] + victim[1], zone.lo)

    def test_a_refused_open_gives_back_every_low_plug_and_the_ballast(self):
        for kwargs in (dict(leave=2500 * PAGE, reserve=2500 * PAGE), dict(leave=400 * PAGE, reserve=0, ceiling=100 * PAGE)):
            heap = fragmented_heap()
            baseline = dict(heap.used)
            zone = plug.Zone(FakeMemory(heap), name='engine', **kwargs)
            with self.assertRaises(plug.CapturePlugError):
                zone.open()
            self.assertEqual(heap.used, baseline, kwargs)
            self.assertEqual((zone.low_plugs, zone.ballast, zone.state), ([], None, 'closed'))

    def test_a_sweep_that_fails_hands_back_what_it_took(self):
        heap = Heap()
        held = [heap.allocate(PAGE) for _ in range(60)]
        for address in held[::2]:
            heap.free(address)
        baseline = dict(heap.used)
        with self.assertRaisesRegex(plug.CapturePlugError, 'iterations'):
            plug.sweep(FakeMemory(heap), held[-1] + PAGE, max_iterations=10)
        self.assertEqual(heap.used, baseline)

    def test_the_free_memory_above_an_address_is_read_by_plugging_below_it_and_nothing_is_left_behind(self):
        heap = fragmented_heap()
        memory = FakeMemory(heap)
        top = 1500 * PAGE
        baseline = dict(heap.used)
        expected = sum(room for address, room in heap.free_blocks() if address >= top) + max(
            0, sum(min(room, address + room - top) for address, room in heap.free_blocks() if address < top < address + room))
        self.assertEqual(plug.free_above(memory, top), expected)
        self.assertEqual(heap.used, baseline)

    def test_the_monitor_counts_the_reserve_above_the_packed_zone_not_the_free_memory_below_it(self):
        heap, zone = self.zone()
        zone.open()
        Capture(heap).run(persistent=[60 * PAGE], temporaries=[300 * PAGE])
        zone.seal()
        zone.release()
        memory = zone.memory
        total = memory.free()
        above = plug.free_above(memory, zone.hi)
        self.assertLess(above, total, 'the low region the ballast held is free but is not the reserve')
        plug.monitor([memory], above, None, 'ok', above=zone.hi)
        with self.assertRaisesRegex(plug.CapturePlugError, 'above the packed zone'):
            plug.monitor([memory], above + PAGE, None, 'reserve nearly gone', above=zone.hi)
        plug.monitor([memory], above + PAGE, None, 'total passes', above=None)   # the total free memory would not have seen it

    def test_the_states_are_enforced(self):
        heap, zone = self.zone()
        with self.assertRaises(plug.CapturePlugError):
            zone.seal()
        zone.open()
        with self.assertRaises(plug.CapturePlugError):
            zone.open()
        with self.assertRaises(plug.CapturePlugError):
            zone.release()

    def test_close_gives_every_plug_and_the_ballast_back_and_abandon_gives_nothing_back(self):
        heap, zone = self.zone()
        baseline = dict(heap.used)
        zone.open()
        Capture(heap).run(temporaries=[100 * PAGE])
        zone.seal()
        zone.release()
        zone.close()
        self.assertEqual(heap.used, baseline)
        heap2, zone2 = self.zone()
        zone2.open()
        held = dict(heap2.used)
        del plug.ABANDONED[:]
        zone2.abandon()
        self.assertEqual(heap2.used, held, 'a failed attach frees nothing')
        self.assertTrue(plug.ABANDONED)
        del plug.ABANDONED[:]

    def test_an_engine_zone_that_would_reach_the_binary_reserve_is_refused_and_leaves_nothing_behind(self):
        heap = fragmented_heap()
        baseline = dict(heap.used)
        zone = plug.Zone(FakeMemory(heap), 400 * PAGE, 0, name='engine', ceiling=100 * PAGE)
        with self.assertRaisesRegex(plug.CapturePlugError, 'reserved for kernel binaries'):
            zone.open()
        self.assertEqual(heap.used, baseline)


class V186SealTests(unittest.TestCase):
    """The v186 attach (c2-packed-tp4-speed-fix, QWEN_FAST_CAPTURE_PLUG=1): the seal plugged the whole 256 MB reserve and the next
    allocation died with 'free: 3136 B'. The numbers below are the logged ones (per bank): zone [0xcceba240, 0xeceba240), 512 MB
    leave, 256 MB reserve; the captures' live memory ended 960 B under the zone top, a freed hole [0xeceb9e80, 0xfce8b680) was
    left above it (kernel binaries had taken the top 0x2ebc0 bytes), and 3136 B of free slivers (each under a page) sat between
    buffers, so memory.free() was a page and a half more than that hole."""

    TOP = 0xfceba240
    LO = 0xcceba240
    HI = 0xeceba240
    USED_END = 0xeceb9e80
    BINARIES = 0xfceba240 - 0xfce8b680
    SLIVERS = (1045, 1045, 1046)

    def heap_and_zone(self, floor=128 * plug.MB):
        heap = Heap(self.TOP)
        low = 0x1000000 + 0x240          # the low region: used, ends where the main free region starts (page-misaligned, as logged)
        heap.used[0] = low
        zone = plug.Zone(FakeMemory(heap), 512 * plug.MB, 256 * plug.MB, name='packed', floor=floor)
        return heap, zone

    def run_captures(self, heap, zone):
        """Persistent buffers and temporaries inside the zone; the live memory ends at USED_END; slivers between buffers."""
        slivers = []
        for size in self.SLIVERS:
            slivers.append(heap.allocate(size))
            heap.allocate(64)               # a persistent 64 B between the slivers keeps them apart
        big = heap.allocate(150 * plug.MB)
        heap.allocate(200 * plug.MB)
        heap.free(big)                      # a freed 150 MB temporary: plugged by the seal
        heap.allocate(self.USED_END - max(a + s for a, s in heap.used.items()))
        for address in slivers:
            heap.free(address)
        heap.allocate_top(self.BINARIES)    # the program cache took the top of the reserve while the captures ran

    def test_the_logged_arithmetic_is_what_the_model_builds(self):
        heap, zone = self.heap_and_zone()
        zone.open()
        self.assertEqual((zone.lo, zone.hi), (self.LO, self.HI))
        self.run_captures(heap, zone)
        holes = heap.free_blocks()
        self.assertEqual(holes[-1], (self.USED_END, 0xfce8b680 - self.USED_END))
        self.assertEqual(sum(room for _, room in holes) - holes[-1][1], sum(self.SLIVERS) + 150 * plug.MB)

    def test_the_seal_keeps_the_reserve_whole_and_never_runs_the_bank_out_of_memory(self):
        heap, zone = self.heap_and_zone()
        zone.open()
        self.run_captures(heap, zone)
        zone.seal()                         # the unfixed seal plugged the reserve, then raised 'Out of Memory' from the probe
        zone.release()
        reserve = heap.free_blocks()[-1]
        self.assertEqual(reserve[0], self.USED_END, 'the reserve hole starts where the used memory ends')
        self.assertEqual(zone.hi, self.USED_END)
        self.assertGreater(reserve[1], 255 * plug.MB, 'the whole reserve is free for kernel binaries')
        # no plug lies in the reserve
        self.assertTrue(all(block.end <= self.USED_END for block in zone.zone_plugs))
        # the freed 150 MB temporary is plugged: a later allocation cannot land in it
        later = heap.allocate(64 * PAGE)
        self.assertLessEqual(later + 64 * PAGE, zone.lo)

    def test_a_reserve_under_the_floor_is_refused_with_a_clear_error_not_an_oom(self):
        heap, zone = self.heap_and_zone(floor=300 * plug.MB)
        zone.open()
        self.run_captures(heap, zone)
        with self.assertRaisesRegex(plug.CapturePlugError, 'under the 300.0 MB floor'):
            zone.seal()

    def test_a_freed_hole_above_the_zone_is_refused_when_plugging_it_would_leave_less_than_the_floor(self):
        heap = Heap()
        zone = plug.Zone(FakeMemory(heap), 400 * PAGE, 2500 * PAGE, name='packed', floor=2400 * PAGE)
        zone.open()
        heap.allocate(450 * PAGE)                  # persistent, 50 pages over the zone top
        temporary = heap.allocate(100 * PAGE)
        heap.allocate(10 * PAGE)
        heap.free(temporary)                       # a freed temporary above the zone top, with 2340 pages of reserve above it
        with self.assertRaisesRegex(plug.CapturePlugError, 'would leave'):
            zone.seal()
        self.assertEqual(zone.state, 'open', 'a refused seal does not claim to be sealed')

    def test_the_wiring_hands_the_floor_to_the_packed_zone_only(self):
        heap = Heap()
        settings = dict(plug.config({plug.FLAG: '1'}), leave=400 * PAGE, reserve=700 * PAGE, engine_leave=100 * PAGE, min_free=90 * PAGE)
        packed = plug.Plug.packed(settings, None, None, memory_factory=lambda operations, mesh, kind: FakeMemory(heap, kind))
        engine = plug.Plug.engine(settings, None, None, memory_factory=lambda operations, mesh, kind: FakeMemory(heap, kind))
        self.assertEqual(packed.zones[0].floor, 90 * PAGE)
        self.assertIsNone(engine.zones[0].floor)


class PlugTests(unittest.TestCase):
    def build(self, heap, **extra):
        settings = dict(plug.config({plug.FLAG: '1'}), leave=1200 * PAGE, reserve=800 * PAGE, engine_leave=300 * PAGE,
                        min_free=100 * PAGE, **extra)
        made = []

        def factory(operations, mesh, kind):
            made.append(kind)
            return FakeMemory(heap, kind)

        return settings, factory, made

    def test_the_packed_plug_then_an_engine_plug_each_keep_their_captures_apart_from_later_allocations(self):
        heap = fragmented_heap()
        settings, factory, made = self.build(heap)
        lines = []
        packed = plug.Plug.packed(settings, None, None, lines.append, factory)
        packed.open()
        block = Capture(heap).run(persistent=[50 * PAGE], temporaries=[400 * PAGE, 200 * PAGE])
        packed.seal()
        engine = plug.Plug.engine(settings, None, None, lines.append, factory, ceiling=packed.zone_lo())
        engine.open()
        mine = Capture(heap).run(persistent=[20 * PAGE], temporaries=[150 * PAGE, 90 * PAGE])
        engine.seal()
        victims = []
        for size in (11, 40, 40, 13, 70):
            victims.append((heap.allocate(size * PAGE), size * PAGE))
        for address, size in victims:
            self.assertFalse(hits(address, size, block.temporary), 'packed temporary')
            self.assertFalse(hits(address, size, mine.temporary), 'engine temporary')
        self.assertTrue(any('capture plug packed' in line and 'sealed' in line for line in lines))
        engine.close()
        packed.close()
        survivors = sorted(heap.used.values())
        expected = sorted([40 * PAGE] * 4 + [size for _, size in block.persistent + mine.persistent] + [size for _, size in victims])
        self.assertEqual(survivors, expected, 'closing the plugs gave back every plug and ballast and nothing else')

    def test_the_monitor_refuses_when_the_free_memory_left_is_under_the_floor(self):
        heap = fragmented_heap()
        settings, factory, made = self.build(heap)
        packed = plug.Plug.packed(settings, None, None, None, factory)
        packed.open()
        packed.seal()
        packed.monitor('after seal')
        settings['min_free'] = heap.free_bytes() + PAGE
        with self.assertRaisesRegex(plug.CapturePlugError, 'no room above the plugs'):
            plug.monitor(packed.memories, settings['min_free'], None, 'engine admission')

    def test_l1_gets_its_own_zone_only_when_asked(self):
        heap = fragmented_heap()
        settings, factory, made = self.build(heap)
        plug.Plug.packed(settings, None, None, None, factory)
        self.assertEqual(made, ['DRAM'])
        del made[:]
        settings['l1'] = True
        built = plug.Plug.packed(settings, None, None, None, factory)
        self.assertEqual(made, ['DRAM', 'L1'])
        self.assertEqual([zone.memory.kind for zone in built.zones], ['DRAM', 'L1'])


class FakeTensor:
    def __init__(self, address):
        self.address = address


class DeviceMemoryTests(unittest.TestCase):
    """The adapter against a fake ttnn: the shape of the allocation and the lockstep check."""

    def operations(self, addresses=(0x1000,) * 4):
        calls = []
        ops = SimpleNamespace(
            BufferType=SimpleNamespace(DRAM='dram', L1='l1'), DRAM_MEMORY_CONFIG='DRAM_CFG', L1_MEMORY_CONFIG='L1_CFG',
            bfloat16='bf16', TILE_LAYOUT='tile', Shape=lambda dims: tuple(dims), calls=calls, deallocated=[])

        def allocate(shape, dtype, layout, device, config):
            calls.append((shape, dtype, layout, config))
            return FakeTensor(addresses)

        ops.allocate_tensor_on_device = allocate
        ops.get_device_tensors = lambda tensor: [SimpleNamespace(buffer_address=lambda a=a: a) for a in tensor.address]
        ops.deallocate = ops.deallocated.append
        ops.get_memory_view = lambda device, kind: SimpleNamespace(
            num_banks=8, total_bytes_per_bank=1 << 30, total_bytes_allocated_per_bank=1 << 20,
            total_bytes_free_per_bank=(1 << 30) - (1 << 20), largest_contiguous_bytes_free_per_bank=1 << 29)
        return ops

    def memory(self, ops, kind='DRAM'):
        mesh = SimpleNamespace(get_devices=lambda: [object() for _ in range(4)])
        return plug.DeviceMemory(ops, mesh, kind)

    def test_a_per_bank_size_becomes_banks_times_pages_of_tiles_and_the_address_is_the_chips_common_one(self):
        ops = self.operations()
        memory = self.memory(ops)
        with patch.dict(sys.modules):
            import tp_addresses
            with patch.object(tp_addresses.tp_shapes, 'chip_count', lambda: 4):
                block = memory.allocate(3 * PAGE)
        self.assertEqual((block.address, block.per_bank), (0x1000, 3 * PAGE))
        shape, dtype, layout, config = ops.calls[0]
        self.assertEqual((shape, dtype, layout, config), ((1, 1, 32, 32 * 8 * 3), 'bf16', 'tile', 'DRAM_CFG'))
        self.assertEqual(memory.largest_free(), 1 << 29)

    def test_l1_uses_the_l1_config_and_a_lockstep_violation_is_refused(self):
        ops = self.operations()
        memory = self.memory(ops, 'L1')
        with patch.dict(sys.modules):
            import tp_addresses
            with patch.object(tp_addresses.tp_shapes, 'chip_count', lambda: 4):
                memory.allocate(PAGE)
                self.assertEqual(ops.calls[0][3], 'L1_CFG')
                bad = self.operations(addresses=(0x1000, 0x1000, 0x2000, 0x1000))
                with self.assertRaisesRegex(plug.CapturePlugError, 'different addresses'):
                    self.memory(bad).allocate(PAGE)
                self.assertEqual(len(bad.deallocated), 1, 'the refused allocation is given back')


if __name__ == '__main__':
    unittest.main()
