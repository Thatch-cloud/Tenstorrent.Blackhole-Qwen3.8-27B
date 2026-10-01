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
                         (4096 * plug.MB, 2048 * plug.MB, 1024 * plug.MB, 512 * plug.MB))
        self.assertFalse(found['engines'] or found['l1'])
        found = plug.config({plug.FLAG: '1', plug.LEAVE_FLAG: '3000', plug.ENGINES_FLAG: '1', plug.L1_FLAG: '1',
                             plug.L1_LEAVE_KB if hasattr(plug, 'L1_LEAVE_KB') else plug.L1_LEAVE_FLAG: '64'})
        self.assertEqual((found['leave'], found['engines'], found['l1'], found['l1_leave']), (3000 * plug.MB, True, True, 64 * plug.KB))
        for flag in (plug.LEAVE_FLAG, plug.RESERVE_FLAG, plug.MIN_FREE_FLAG):
            for bad in ('0', '-1', '1.5', 'x'):
                with self.assertRaises(ValueError, msg=(flag, bad)):
                    plug.config({plug.FLAG: '1', flag: bad})
        with self.assertRaises(ValueError):
            plug.config({plug.FLAG: '1', plug.ENGINES_FLAG: '2'})


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
