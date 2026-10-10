"""card_telemetry.py / card_telemetry.sh / C2_TELEMETRY: the read-only per-chip ARC telemetry sidecar of the card steps, held on CPU with a fake telemetry source.

The fake is a chip whose ARC firmware table lives in a byte image behind the same AXI reads luwen makes (SCRATCH_RAM[13] -> pointer -> version, count, (tag, offset)
words, data words), so the real decoder runs on real bytes; it has no write, no ARC message and no NOC call at all, so a sampler that tried one would fail here.
"""
import ast
import csv
import gzip
import io
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import c2_serving_job as job
import card_telemetry as ct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODULE = HERE / 'card_telemetry.py'
LIBRARY = HERE / 'card_telemetry.sh'
WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml'
PROFILES_FILE = HERE / 'qwen_c2_profiles.json'
TEMPLATES = HERE / 'references' / 'fusion-jobs' / 'TEL'
MANIFEST = HERE / 'fusion-wp' / 'TEL.json'
BASH = shutil.which('bash')
SCRATCH = 0x1FF30434      # whatever the fake answers axi_translate with
CSM = 0x10000000
TAG = ct.TAG


# ---------------------------------------------------------------- the fake chip

def table_image(tags, pointer=CSM + 0x400, count=78, version=0x100):
    """The CSM bytes [CSM, CSM + 0x2000) holding a firmware telemetry_table (telemetry.c) at `pointer`: version, count, count (tag | offset << 16) words, count data words.
    A pointer outside the image (a test of the reader's refusals) lays the table at the default place and only the pointer the chip publishes differs; a `count` over 78 is
    the header's claim only."""
    image = bytearray(0x2000)
    if not CSM <= pointer < CSM + 0x800:
        pointer = CSM + 0x400
    base = pointer - CSM
    struct.pack_into('<II', image, base, version, count)
    count = min(count, 78)
    for index, tag in enumerate(sorted(tags)):
        struct.pack_into('<I', image, base + 8 + 4 * index, tag | (tag << 16))
        struct.pack_into('<I', image, base + 8 + count * 4 + 4 * tag, tags[tag] & 0xFFFFFFFF)
    return image


def fx(celsius):
    """A temperature as the firmware publishes it: signed 16.16 fixed point."""
    return int(round(celsius * 65536)) & 0xFFFFFFFF


def tags(aiclk=1350, fmax=1350, busy=True, limiter=None, limit_mhz=None, power=100, board=200, tdc=90, vcore=800, temp=60.5, gddr=64, nop_starts=0, nop_on=0,
         trips=0, heartbeat=100, ppm=True):
    """A firmware table as v19.12.0 publishes it for the given state. `limiter` (an ARB_MAX_NAMES entry) holds the maximum arbiter at `limit_mhz`."""
    max_name = limiter or 'doppler_slow'     # nothing limiting: the last enabled arbiter at fmax is the one named (get_aiclk_effective_arb_max keeps the last of equals)
    arb_max = (limit_mhz if limiter else fmax) | (ct.ARB_MAX_NAMES.index(max_name) << 16)
    arb_min = (fmax if busy else 200) | (ct.ARB_MIN_NAMES.index('busy') << 16)         # the same: the busy arbiter is named even at fmin
    table = {TAG['AICLK']: aiclk, TAG['AICLK_LIMIT_MAX']: fmax, TAG['AICLK_ARB_MIN']: arb_min, TAG['AICLK_ARB_MAX']: arb_max,
             TAG['ENABLED_MIN_ARB']: 0b11, TAG['ENABLED_MAX_ARB']: 0b00111111111, TAG['HOST_AICLK_LIMIT']: 0,
             TAG['VCORE']: vcore, TAG['TDP']: power, TAG['TDC']: tdc, TAG['INPUT_POWER']: board, TAG['ASIC_TEMPERATURE']: fx(temp),
             TAG['MAX_GDDR_TEMP']: gddr, TAG['TDP_LIMIT_MAX']: 150, TAG['TDC_LIMIT_MAX']: 200, TAG['THM_LIMIT_THROTTLE']: 90, TAG['BOARD_POWER_LIMIT']: 300,
             TAG['FAN_SPEED']: 0xFFFFFFFF, TAG['FAN_RPM']: 0xFFFFFFFF, TAG['TIMER_HEARTBEAT']: heartbeat, TAG['NOP_START_COUNT']: nop_starts,
             TAG['NOP_ON_DURATION']: nop_on, TAG['THERM_TRIP_COUNT']: trips, TAG['KERNEL_THROTTLER']: 0x00000000,
             TAG['GDDR_WEST_IO_POWER']: 3, TAG['GDDR_EAST_IO_POWER']: 4, TAG['ENABLED_TENSIX_COL']: 0x3FFF, TAG['FLASH_BUNDLE_VERSION']: 0x13090000}
    if ppm:
        reason = 1 if (limiter and limit_mhz is not None and limit_mhz < (fmax if busy else 200)) else 0
        table[TAG['AICLK_PPM_INFO']] = (reason << 16) | (ct.ARB_MAX_NAMES.index(limiter) if reason else 1)
    return table


STRUCT_ATTRS = dict((tag, attribute) for attribute, tag in ct.STRUCT_TAGS)


class ChipState(object):
    """One chip's script: the tag table each successive open of it publishes (the last repeats), or a chip that cannot be opened."""

    def __init__(self, script=None, dead=False, pointer=CSM + 0x400, count=78, lib=None):
        self.script, self.dead, self.pointer, self.count, self.opens, self.lib = script or [tags()], dead, pointer, count, 0, lib

    def current(self):
        return self.script[min(self.opens - 1, len(self.script) - 1)]


class FakeBlackhole(object):
    """luwen's PciBlackhole as far as a reader goes: axi_translate, axi_read32, axi_read and get_telemetry. Nothing that writes or messages exists on it."""

    def __init__(self, state, log):
        self.state, self.log = state, log
        self.image = table_image(state.current(), state.pointer, state.count)

    def axi_translate(self, name):
        self.log.append('axi_translate')
        if name != ct.SCRATCH_TABLE_POINTER:
            raise KeyError(name)
        return types.SimpleNamespace(addr=SCRATCH, size=4)

    def axi_read32(self, addr):
        self.log.append('axi_read32')
        if addr == SCRATCH:
            return self.state.pointer
        if not CSM <= self.state.pointer < CSM + 0x800:        # the table is laid at the default place whatever the chip publishes
            addr = addr - self.state.pointer + CSM + 0x400
        return struct.unpack_from('<I', self.image, addr - CSM)[0]

    def axi_read(self, addr, buffer):
        self.log.append('axi_read')
        buffer[:] = self.image[addr - CSM:addr - CSM + len(buffer)]

    def get_telemetry(self):
        self.log.append('get_telemetry')
        values = dict((STRUCT_ATTRS[tag], word) for tag, word in self.state.current().items() if tag in STRUCT_ATTRS)
        values.update(self.state.lib or {})
        return types.SimpleNamespace(**values)


class FakeChip(object):
    def __init__(self, state, log):
        self.bh = FakeBlackhole(state, log)

    def as_bh(self):
        return self.bh


class FakePyluwen(object):
    """The module: PciChip(pci_interface=N)."""
    __version__ = '0.10.0-fake'

    def __init__(self, chips):
        self.chips, self.log, self.opened = chips, [], []

    def PciChip(self, pci_interface=0):
        state = self.chips.get(pci_interface)
        if state is None or state.dead:
            raise RuntimeError('Could not open chip on pci interface %d' % pci_interface)
        state.opens += 1
        self.opened.append(pci_interface)
        return FakeChip(state, self.log)


class Time(object):
    """A clock that only moves when the sampler sleeps."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return 1700000000.0 + self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def run_sampler(world, nodes, out, duration, interval=0.1, parent=None):
    clock = Time()
    sampler = ct.Sampler(world, nodes, out, interval=interval, parent=parent, clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep)
    sampler.run(duration)
    return sampler


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_file(path):
    with open(path) as handle:
        return handle.read()


def rows_of(directory):
    with open(os.path.join(directory, ct.CSV_NAME), newline='') as handle:
        return list(csv.DictReader(handle))


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)


# ---------------------------------------------------------------- the decode

class DecodeTests(unittest.TestCase):
    def test_a_busy_unthrottled_chip(self):
        row = ct.decode(tags(aiclk=1350, power=101, board=210, tdc=88, vcore=812, temp=61.5, gddr=66))
        self.assertEqual((row['aiclk_mhz'], row['aiclk_fmax_mhz'], row['busy']), (1350, 1350, 1))
        self.assertEqual((row['vcore_mv'], row['power_w'], row['board_power_w'], row['tdc_a']), (812, 101, 210, 88))
        self.assertEqual((row['asic_temp_c'], row['gddr_temp_c']), (61.5, 66))
        self.assertEqual(row['limited_by'], ct.NOT_LIMITED)
        self.assertEqual((row['tdp_limit_w'], row['tdc_limit_a'], row['thm_limit_c'], row['board_power_limit_w']), (150, 200, 90, 300))
        self.assertEqual(row['gddr_io_w'], 7)
        self.assertEqual((row['fan_pct'], row['fan_rpm']), (None, None), 'no fan control publishes 0xFFFFFFFF: not a reading')

    def test_each_limiter_is_named_by_its_arbiter_and_only_when_it_lowers_the_target(self):
        for name in ct.ARB_MAX_NAMES[1:]:
            row = ct.decode(tags(aiclk=1210, limiter=name, limit_mhz=1210))
            self.assertEqual(row['limited_by'], name, name)
            self.assertEqual(row['arb_max_mhz'], 1210)
        # equal to the minimum arbiter's frequency: the firmware keeps the minimum (strictly lower only), so nothing is limiting
        self.assertEqual(ct.decode(tags(limiter='tdp', limit_mhz=1350))['limited_by'], ct.NOT_LIMITED)

    def test_an_idle_chip_is_not_busy_and_cannot_be_limited(self):
        row = ct.decode(tags(aiclk=200, busy=False, limiter='tdp', limit_mhz=900, power=30))
        self.assertEqual((row['busy'], row['limited_by']), (0, ct.NOT_LIMITED))
        self.assertEqual(row['arb_min'], 'busy', 'the busy arbiter is named at fmin too (ties go to the last arbiter): busy comes from the frequency')

    def test_the_ppm_info_is_read_low_arbiter_high_reason(self):
        row = ct.decode(tags(aiclk=1200, limiter='fast_tdc', limit_mhz=1200))
        self.assertEqual((row['ppm_reason'], row['ppm_arb']), ('max_arb', ct.ARB_MAX_NAMES.index('fast_tdc')))
        self.assertEqual(ct.decode(tags())['ppm_reason'], 'min_arb')

    def test_packing_of_the_words(self):
        table = tags()
        table[TAG['AICLK']] = (1400 << 16) | 1337            # older firmware: the upper half is a maximum, tt-smi masks it
        table[TAG['TDP']] = (150 << 16) | 77
        table[TAG['TDC']] = (200 << 16) | 55
        row = ct.decode(table)
        self.assertEqual((row['aiclk_mhz'], row['power_w'], row['tdc_a']), (1337, 77, 55))
        table[TAG['FAN_RPM']] = 4200
        table[TAG['FAN_SPEED']] = 55
        self.assertEqual((ct.decode(table)['fan_rpm'], ct.decode(table)['fan_pct']), (4200, 55))

    def test_temperatures(self):
        self.assertEqual(ct.signed_16_16(fx(61.5)), 61.5)
        self.assertEqual(ct.signed_16_16(fx(-3.25)), -3.25)
        self.assertIsNone(ct.signed_16_16(0x80000000), 'the firmware error value (FLT_MAX)')
        table = tags()
        del table[TAG['MAX_GDDR_TEMP']]
        table[TAG['GDDR_0_1_TEMP']] = 0x3c3e3a3c
        table[TAG['GDDR_4_5_TEMP']] = 0x41403f3e
        self.assertEqual(ct.decode(table)['gddr_temp_c'], 0x41, 'without MAX_GDDR_TEMP: the hottest die byte of the packed words')

    def test_a_table_without_the_arbiter_tags_has_no_attribution(self):
        old = dict((tag, word) for tag, word in tags().items() if tag <= 64)
        row = ct.decode(old)
        self.assertEqual((row['busy'], row['limited_by'], row['arb_max_mhz'], row['ppm_reason']), (None, None, None, None))
        self.assertEqual(row['aiclk_mhz'], 1350)

    def test_the_names_of_the_tags_and_arbiters_are_the_firmwares(self):
        self.assertEqual(TAG['AICLK'], 14)
        self.assertEqual((TAG['AICLK_ARB_MIN'], TAG['AICLK_ARB_MAX'], TAG['AICLK_PPM_INFO']), (65, 66, 69))
        self.assertEqual((TAG['NOP_START_COUNT'], TAG['NOP_ON_DURATION'], TAG['KERNEL_THROTTLER']), (76, 77, 75))
        self.assertEqual(len(ct.TAG_NAMES), 77)
        self.assertEqual(ct.ARB_MAX_NAMES[:4], ('fmax', 'tdp', 'fast_tdc', 'tdc'))
        self.assertEqual(len(ct.ARB_MAX_NAMES), 11)
        self.assertEqual(ct.ARB_MIN_NAMES, ('fmin', 'busy'))
        self.assertEqual(ct.REASON_NAMES[1], 'max_arb')


# ---------------------------------------------------------------- the read

class ReadTests(unittest.TestCase):
    def bh(self, **state):
        log = []
        return FakeBlackhole(ChipState(**state), log), log

    def test_the_table_is_read_by_the_axi_reads_luwen_makes_and_by_no_other_call(self):
        state = ChipState([tags(aiclk=1111)])
        state.opens = 1
        log = []
        table = ct.read_table(FakeBlackhole(state, log))
        self.assertEqual(table, dict((tag, word & 0xFFFFFFFF) for tag, word in tags(aiclk=1111).items()))
        self.assertEqual(set(log), {'axi_translate', 'axi_read32', 'axi_read'})
        self.assertEqual(log.count('axi_read'), 2, 'the tag block and the data block')

    def test_an_unpublished_or_wild_pointer_or_count_is_an_error_not_a_read(self):
        for pointer, words in ((0, 'not published'), (0x20000000, 'outside the CSM window'), (0x08000000, 'outside the CSM window')):
            state = ChipState(pointer=pointer)
            state.opens = 1
            with self.assertRaisesRegex(RuntimeError, words):
                ct.read_table(FakeBlackhole(state, []))
        state = ChipState(count=100000)
        state.opens = 1
        with self.assertRaisesRegex(RuntimeError, 'not plausible'):
            ct.read_table(FakeBlackhole(state, []))

    def test_read_chip_opens_the_node_index_and_checks_the_static_tags_against_pyluwen_on_the_first_read(self):
        world = FakePyluwen({3: ChipState([tags()])})
        row, table, source, result = ct.read_chip(world, 3, check=True)
        self.assertEqual((source, result), ('raw', []))
        self.assertEqual(world.opened, [3])
        self.assertEqual(row['aiclk_mhz'], 1350)
        self.assertIn('get_telemetry', world.log)
        world.log[:] = []
        ct.read_chip(world, 3, check=False)
        self.assertNotIn('get_telemetry', world.log, 'checked once per chip, not on every sample')

    def test_a_disagreeing_decode_is_reported_by_attribute(self):
        world = FakePyluwen({0: ChipState([tags()], lib=dict(tdp_limit_max=140))})
        _, _, _, result = ct.read_chip(world, 0, check=True)
        self.assertEqual(len(result), 1)
        self.assertIn('tdp_limit_max tag 64 raw 0x96 pyluwen 0x8c', result[0])

    def test_a_missing_or_failing_library_read_skips_the_check_and_never_calls_the_decode_wrong(self):
        world = FakePyluwen({0: ChipState([tags()])})
        state = world.chips[0]
        state.opens = 1
        bh = FakeBlackhole(state, [])
        table = ct.read_table(bh)
        bh.get_telemetry = lambda: (_ for _ in ()).throw(AttributeError('no get_telemetry'))
        self.assertEqual(ct.static_check(bh, table), 'skipped: get_telemetry is not in this pyluwen')
        bh.get_telemetry = lambda: (_ for _ in ()).throw(RuntimeError('ARC busy'))
        self.assertEqual(ct.static_check(bh, table), 'skipped: get_telemetry failed (ARC busy)')
        self.assertEqual(ct.check_text('skipped: x'), 'skipped: x')
        self.assertEqual(ct.check_text([]), 'ok')
        self.assertEqual(ct.check_text(['a', 'b']), 'MISMATCH: a; b')
        self.assertEqual(ct.check_text(None), 'not checked')

    def test_when_the_raw_read_fails_the_struct_of_get_telemetry_is_the_source(self):
        world = FakePyluwen({0: ChipState([tags()], pointer=0)})
        row, table, source, result = ct.read_chip(world, 0, check=True)
        self.assertEqual(source, 'struct')
        self.assertEqual(row['aiclk_mhz'], 1350)
        self.assertIsNone(row['arb_max_mhz'], 'the struct has no arbiter tags')
        self.assertIsNone(row['limited_by'])

    def test_a_chip_that_is_not_blackhole_or_cannot_open_raises(self):
        world = FakePyluwen({0: ChipState(dead=True)})
        with self.assertRaisesRegex(RuntimeError, 'Could not open'):
            ct.read_chip(world, 0)
        wormhole = types.SimpleNamespace(PciChip=lambda pci_interface=0: types.SimpleNamespace(as_bh=lambda: None))
        with self.assertRaisesRegex(RuntimeError, 'not a Blackhole'):
            ct.read_chip(wormhole, 0)

    def test_a_rust_panic_is_an_error_row_not_the_end_of_the_sampler(self):
        """pyluwen's Rust panics (an unreadable PCI config space, a driver older than 2) arrive as pyo3's PanicException: a BaseException, not an Exception."""
        class PanicException(BaseException):
            pass

        world = FakePyluwen({0: ChipState(), 1: ChipState()})
        real = world.PciChip

        def panicking(pci_interface=0):
            if pci_interface == 1:
                raise PanicException('Failed to open config space for device 1 with error No such file')
            return real(pci_interface)

        world.PciChip = panicking
        with tempfile.TemporaryDirectory() as out:
            run_sampler(world, ["/dev/tenstorrent/0", "/dev/tenstorrent/1"], out, 0.25)
            rows = rows_of(out)
        bad = [row for row in rows if row['chip'] == '1']
        self.assertTrue(bad and all(row['ok'] == '0' and row['error'].startswith('PanicException: Failed to open config space') for row in bad))
        self.assertTrue([row for row in rows if row['chip'] == '0' and row['ok'] == '1'], 'the other chip is still read')
        interrupt = FakePyluwen({0: ChipState()})
        interrupt.PciChip = lambda pci_interface=0: (_ for _ in ()).throw(KeyboardInterrupt())
        with tempfile.TemporaryDirectory() as out, self.assertRaises(KeyboardInterrupt):
            run_sampler(interrupt, ['0'], out, 0.25)

    def test_node_index(self):
        self.assertEqual(ct.node_index('/dev/tenstorrent/12'), 12)
        self.assertEqual(ct.node_index('3'), 3)
        for bad in ('/dev/tenstorrent/by-id/blackhole-x', '/dev/null', '', '1; rm'):
            with self.assertRaises(ValueError):
                ct.node_index(bad)


class ReadOnlyTests(unittest.TestCase):
    """The sampler may read the table and nothing else: no ARC message, no write, no NOC access, no power state, no reset."""
    FORBIDDEN = {'arc_msg', 'arc_msg_buf', 'axi_write', 'axi_write32', 'noc_write', 'noc_write32', 'noc_read', 'noc_read32', 'noc_multicast', 'noc_broadcast', 'set_power',
                 'set_power_state', 'spi_write', 'spi_read', 'encode_and_write_boot_fs_table', 'detect_chips', 'detect_chips_fallible', 'init', 'open_remote',
                 'setup_tlb', 'config_dma', 'dma_transfer_turbo', 'allocate_dma_buffer', 'get_neighbouring_chips', 'write', 'ioctl', 'system', 'Popen', 'check_output'}
    ALLOWED_CHIP_CALLS = {'PciChip', 'as_bh', 'axi_translate', 'axi_read32', 'axi_read', 'get_telemetry'}

    def test_the_module_never_names_a_call_that_writes_or_messages_the_chip(self):
        tree = ast.parse(MODULE.read_text(encoding='utf-8'))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
        # `write` is the csv/json file writer and `init`'s only appearances are the Sampler's __init__ and the word in a docstring: allow the file-object ones explicitly
        allowed_here = {'write', 'init'}
        self.assertEqual(sorted((names & self.FORBIDDEN) - allowed_here), [])
        pyluwen_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in ('bh', 'chip', 'pyluwen'):
                pyluwen_calls.add(node.func.attr)
        self.assertEqual(sorted(pyluwen_calls - self.ALLOWED_CHIP_CALLS), [])
        self.assertIn('axi_read', pyluwen_calls)

    def test_a_whole_run_makes_only_reads(self):
        with tempfile.TemporaryDirectory() as out:
            world = FakePyluwen({0: ChipState(), 1: ChipState()})
            run_sampler(world, ['/dev/tenstorrent/0', '/dev/tenstorrent/1'], out, 0.35)
        self.assertEqual(set(world.log), {'axi_translate', 'axi_read32', 'axi_read', 'get_telemetry'})


# ---------------------------------------------------------------- the sampler

class SamplerTests(Tmp):
    def world(self):
        busy_idle = [tags(aiclk=200, busy=False, power=30)] * 2
        full = [tags(aiclk=1350, power=100)] * 3
        throttled = [tags(aiclk=1200, limiter='tdp', limit_mhz=1200, power=150, board=260, temp=70.25, gddr=71)] * 2
        return FakePyluwen({
            0: ChipState(busy_idle + full + throttled),
            1: ChipState([tags(aiclk=1350, power=95, nop_starts=2)] * 3 + [tags(aiclk=1350, power=96, nop_starts=5, nop_on=40)] * 4),
            2: ChipState(dead=True),
            3: ChipState([dict((tag, word) for tag, word in tags(aiclk=1340).items() if tag <= 64)])})

    def run_all(self):
        out = os.path.join(self.dir, 'telemetry-smoke')
        nodes = ['/dev/tenstorrent/%d' % index for index in range(4)]
        sampler = run_sampler(self.world(), nodes, out, 0.55)
        return sampler, out

    def test_one_row_per_chip_per_tick_with_an_error_row_for_a_chip_that_cannot_be_read(self):
        sampler, out = self.run_all()
        self.assertEqual(sampler.ticks, 7)
        rows = rows_of(out)
        self.assertEqual(len(rows), 7 * 4)
        self.assertEqual(list(rows[0]), list(ct.COLUMNS))
        dead = [row for row in rows if row['chip'] == '2']
        self.assertEqual(len(dead), 7)
        self.assertTrue(all(row['ok'] == '0' and 'Could not open' in row['error'] and row['aiclk_mhz'] == '' for row in dead))
        self.assertEqual((sampler.ok_rows, sampler.error_rows), (21, 7))
        timestamps = [float(row['ts']) for row in rows if row['chip'] == '0']
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertAlmostEqual(timestamps[1] - timestamps[0], 0.1, places=5, msg='the schedule is the interval, not interval plus read time')

    def test_the_meta_names_the_interval_the_source_and_the_decode_check_and_no_identity(self):
        sampler, out = self.run_all()
        meta = load_json(os.path.join(out, ct.META_NAME))
        self.assertEqual((meta['interval_s'], meta['chips'], meta['stop_reason'], meta['ticks']), (0.1, 4, 'duration', 7))
        self.assertEqual(meta['source'], {'0': 'raw', '1': 'raw', '3': 'raw'})
        self.assertEqual(set(meta['decode_check'].values()), {'ok'})
        self.assertEqual(meta['pyluwen'], '0.10.0-fake')
        text = json.dumps(meta) + read_file(os.path.join(out, ct.CSV_NAME))
        self.assertNotRegex(text, r'/dev/tenstorrent|blackhole-|[0-9a-f]{12}', 'a chip is its index: no node, board id or serial in a public artifact')

    def test_the_interval_has_a_floor(self):
        self.assertEqual(ct.Sampler(FakePyluwen({}), ['0'], self.dir, interval=0.001).interval, ct.MIN_INTERVAL_S)

    def test_an_unreadable_chip_never_stops_the_others(self):
        sampler, out = self.run_all()
        self.assertEqual(len([row for row in rows_of(out) if row['chip'] == '0' and row['ok'] == '1']), 7)

    def test_it_stops_when_asked_and_when_its_parent_is_gone(self):
        out = os.path.join(self.dir, 'a')
        clock = Time()

        def sleep(seconds):
            clock.sleep(seconds)
            if clock.now > 0.25:
                sampler.request_stop('SIGTERM')

        sampler = ct.Sampler(FakePyluwen({0: ChipState()}), ['0'], out, interval=0.1, clock=clock.time, monotonic=clock.monotonic, sleep=sleep)
        sampler.run()
        self.assertEqual(sampler.stop_reason, 'SIGTERM')
        self.assertGreaterEqual(sampler.ticks, 2)
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        child.wait()
        gone = ct.Sampler(FakePyluwen({0: ChipState()}), ['0'], os.path.join(self.dir, 'b'), interval=0.1, parent=child.pid)
        self.assertTrue(gone.parent_gone())
        self.assertFalse(ct.Sampler(FakePyluwen({}), ['0'], self.dir, parent=os.getpid()).parent_gone())

    def test_an_interval_behind_schedule_does_not_burst(self):
        out = os.path.join(self.dir, 'slow')
        clock = Time()
        world = FakePyluwen({0: ChipState()})
        real = world.PciChip

        def slow_once(pci_interface=0):
            if not world.opened:
                clock.now += 0.45        # the first read takes 4.5 intervals
            return real(pci_interface)

        world.PciChip = slow_once
        sampler = ct.Sampler(world, ['0'], out, interval=0.1, clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep)
        sampler.run(1.0)
        stamps = [float(row['t_s']) for row in rows_of(out)]
        simultaneous = len([a for a, b in zip(stamps, stamps[1:]) if abs(b - a) < 1e-9])
        self.assertLessEqual(simultaneous, 1, 'one tick at once after the slow read, not the four it missed')
        self.assertLessEqual(sampler.ticks, 9)


# ---------------------------------------------------------------- the summary

class SummaryTests(Tmp):
    def summarise(self):
        out = os.path.join(self.dir, 'telemetry-gate')
        nodes = ['/dev/tenstorrent/%d' % index for index in range(4)]
        run_sampler(SamplerTests.world(self), nodes, out, 0.55)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(ct.main(['summary', out]), 0)
        return out, stdout.getvalue().splitlines(), load_json(os.path.join(out, ct.SUMMARY_JSON))

    def test_one_line_per_chip_with_aiclk_under_load_power_temperature_and_the_throttler(self):
        out, lines, summary = self.summarise()
        self.assertEqual(len(lines), 5)
        self.assertRegex(lines[0], r'^\[TELEMETRY\] telemetry-gate 4 chip\(s\), 28 rows, interval 0\.1 s, source raw, pyluwen 0\.10\.0-fake, decode check ok$')
        chip0 = lines[1]
        self.assertTrue(chip0.startswith('[TELEMETRY] chip0: AICLK under load min/median 1200/1350 MHz (fmax 1350; 5 of 7 samples)'), chip0)
        self.assertIn('power max 150 W (board 260 W), TDC max 90 A', chip0)
        self.assertIn('temp max 70.2 C (GDDR 71 C)', chip0)
        self.assertIn('throttler: ENGAGED tdp 40.0% of busy samples (arbiter down to 1200 MHz)', chip0)
        self.assertIn('PPM info agrees in 2 of 2', chip0)

    def test_the_idle_samples_are_not_load(self):
        _, _, summary = self.summarise()
        stat = summary['chips']['0']
        self.assertEqual((stat['samples'], stat['ok'], stat['busy'], stat['loaded']), (7, 7, 5, 5))
        self.assertEqual(stat['aiclk_all']['min'], 200)
        self.assertEqual(stat['aiclk_loaded'], dict(min=1200, median=1350, n=5))

    def test_a_chip_whose_clock_never_dipped_says_none_engaged_and_a_nop_counter_that_moved_says_engaged(self):
        _, lines, summary = self.summarise()
        self.assertIn('throttler: ENGAGED kernel NOPs started 3 times', lines[2], 'nop_starts is cumulative: the run saw 2 then 5')
        self.assertEqual(summary['chips']['1']['nop_on_samples'], 4)
        clean = ct.summarize_chip([dict(ok=1, chip=0, ts=1.0 * i, t_s=1.0 * i, aiclk_mhz=1350, aiclk_fmax_mhz=1350, arb_min_mhz=1350, arb_max_mhz=1350, arb_max='doppler_slow',
                                        busy=1, limited_by='-', power_w=100) for i in range(5)])
        self.assertFalse(clean['engaged'])
        self.assertIn('throttler: none engaged', ct.chip_line(0, clean, 1.0))

    def test_an_unreadable_chip_is_said_so_with_its_first_error(self):
        _, lines, _ = self.summarise()
        self.assertTrue(lines[3].startswith('[TELEMETRY] chip2: NO SAMPLES (7 rows, first error: Could not open chip on pci interface 2'), lines[3])

    def test_firmware_without_the_arbiter_tags_is_not_called_clean(self):
        _, lines, summary = self.summarise()
        self.assertIn('throttler: NO ATTRIBUTION (no arbiter tags in this firmware)', lines[4])
        self.assertFalse(summary['chips']['3']['attribution'])

    def test_a_clean_run_names_the_limiters_that_were_armed_so_none_engaged_is_not_none_enabled(self):
        stat = ct.summarize_chip([dict(row, en_max_arb=0b110011110, host_fmax_mhz=0) for row in self.clean_rows()])
        self.assertEqual(stat['armed'], ['tdp', 'fast_tdc', 'tdc', 'thm', 'gddr_thm', 'doppler_slow'], 'bits 1-4, 7 and 8 of the mask, never the fmax arbiter')
        self.assertIn('throttler: none engaged (armed: tdp, fast_tdc, tdc, thm, gddr_thm, doppler_slow)', ct.chip_line(0, stat, 1.0))
        ceiling = ct.summarize_chip([dict(row, en_max_arb=0b11, host_fmax_mhz=1200) for row in self.clean_rows()])
        self.assertIn('(host fmax ceiling 1200 MHz)', ct.verdict(ceiling))

    def clean_rows(self):
        return [dict(ok=1, chip=0, ts=float(i), t_s=float(i), aiclk_mhz=1350, aiclk_fmax_mhz=1350, arb_min_mhz=1350, arb_max_mhz=1350, arb_max='doppler_slow', busy=1,
                     limited_by='-', power_w=100) for i in range(5)]

    def test_a_chip_that_never_went_busy_is_not_reported_as_under_load(self):
        rows = [dict(row, busy=0, arb_min_mhz=200, aiclk_mhz=200) for row in self.clean_rows()]
        line = ct.chip_line(0, ct.summarize_chip(rows), 1.0)
        self.assertIn('AICLK never under load (no busy sample); all samples min/median 200/200 MHz (fmax 1350; 5 samples)', line)

    def test_a_clock_below_fmax_with_no_limiter_named_is_reported_not_hidden(self):
        rows = [dict(ok=1, chip=0, ts=float(i), t_s=float(i), aiclk_mhz=1350 if i < 3 else 1100, aiclk_fmax_mhz=1350, arb_min_mhz=1350, arb_max_mhz=1350, arb_max='doppler_slow',
                     busy=1, limited_by='-', power_w=100) for i in range(6)]
        stat = ct.summarize_chip(rows)
        self.assertFalse(stat['engaged'])
        self.assertEqual(stat['below_fmax']['samples'], 3)
        self.assertIn('none engaged (AICLK below fmax in 3 busy samples, no limiter named)', ct.verdict(stat))

    def test_a_ppm_layout_read_wrong_shows_as_disagreement(self):
        rows = [dict(ok=1, chip=0, ts=float(i), t_s=float(i), aiclk_mhz=1200, aiclk_fmax_mhz=1350, arb_min_mhz=1350, arb_max_mhz=1200, arb_max='tdp', busy=1, limited_by='tdp',
                     power_w=150, ppm_reason='min_arb', ppm_arb=1) for i in range(4)]
        stat = ct.summarize_chip(rows)
        self.assertEqual(stat['ppm_agrees'], dict(of=4, agree=0))
        self.assertIn('PPM info agrees in 0 of 4', ct.verdict(stat))

    def test_the_timeline_buckets_a_minute_and_the_csv_is_gzipped_after_the_summary(self):
        out, _, summary = self.summarise()
        timeline = summary['chips']['0']['timeline']
        self.assertEqual(len(timeline), 1)
        self.assertEqual((timeline[0]['t_s'], timeline[0]['n'], timeline[0]['aiclk_min'], timeline[0]['limited'], timeline[0]['busy']), (0, 7, 200, 2, 5))
        self.assertFalse(os.path.exists(os.path.join(out, ct.CSV_NAME)))
        with gzip.open(os.path.join(out, ct.CSV_GZ_NAME), 'rt') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 28)
        self.assertEqual(read_file(os.path.join(out, ct.SUMMARY_TXT)).splitlines()[1:], self.summarise_again(out))

    def summarise_again(self, out):
        """A second `summary` of the same directory (the always-run step after the step's own) prints the same lines from the gzipped csv."""
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            ct.main(['summary', out])
        return stdout.getvalue().splitlines()[1:]

    def test_an_empty_or_missing_directory_is_a_line_not_a_failure(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(ct.main(['summary', os.path.join(self.dir, 'nothing')]), 0)
        self.assertIn('no telemetry.csv', stdout.getvalue())
        out = os.path.join(self.dir, 'empty')
        os.makedirs(out)
        with open(os.path.join(out, ct.CSV_NAME), 'w') as handle:
            handle.write(','.join(ct.COLUMNS) + '\n')
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(ct.main(['summary', out]), 0)
        self.assertIn('no rows', stdout.getvalue())

    def test_the_loaded_samples_need_power_when_the_busy_flag_is_set_all_the_time(self):
        """The serving container holds the clock busy even idle: the median over busy samples would be fmax. Loaded = busy and at least half the run's top power."""
        rows = [dict(ok=1, chip=0, ts=float(i), t_s=float(i), aiclk_mhz=1350, aiclk_fmax_mhz=1350, arb_min_mhz=1350, arb_max_mhz=1350, arb_max='doppler_slow', busy=1,
                     limited_by='-', power_w=20) for i in range(8)]
        rows += [dict(row, aiclk_mhz=1250, power_w=140, ts=100.0 + i, t_s=100.0 + i) for i, row in enumerate(rows[:2])]
        stat = ct.summarize_chip(rows)
        self.assertEqual(stat['aiclk_busy']['median'], 1350)
        self.assertEqual(stat['aiclk_loaded'], dict(min=1250, median=1250, n=2))


# ---------------------------------------------------------------- the command line, with a fake pyluwen module in a subprocess

FAKE_MODULE = '''"""A fake pyluwen for card_telemetry.py's subprocess tests: FAKE_WORLD is a json file of {node: [ [tag, word] ... ] or null (dead)}."""
import json, os, sys
sys.path.insert(0, %(here)r)
import test_card_telemetry as t

world = json.load(open(os.environ['FAKE_WORLD']))
chips = {}
for node, script in world.items():
    chips[int(node)] = t.ChipState(dead=True) if script is None else t.ChipState([dict((int(tag), word) for tag, word in table.items()) for table in script])
_fake = t.FakePyluwen(chips)
__version__ = _fake.__version__
PciChip = _fake.PciChip
'''


class CommandLineTests(Tmp):
    def setUp(self):
        super().setUp()
        self.fake = os.path.join(self.dir, 'fake')
        os.makedirs(self.fake)
        with open(os.path.join(self.fake, 'pyluwen.py'), 'w') as handle:
            handle.write(FAKE_MODULE % dict(here=str(HERE)))
        self.world_file = os.path.join(self.dir, 'world.json')
        with open(self.world_file, 'w') as handle:
            json.dump({'0': [tags(aiclk=1350)], '1': [tags(aiclk=1300, limiter='tdp', limit_mhz=1300, power=149)], '2': None}, handle)
        self.env = dict(os.environ, PYTHONPATH=self.fake, FAKE_WORLD=self.world_file, PYTHONDONTWRITEBYTECODE='1')

    def run_module(self, *args, **more):
        return subprocess.run([sys.executable, '-s', str(MODULE)] + list(args), env=self.env, capture_output=True, text=True, timeout=60, **more)

    def test_check_prints_each_chip_decoded_and_exits_0_when_any_could_be_read(self):
        result = self.run_module('check', '--nodes', '/dev/tenstorrent/0', '/dev/tenstorrent/1', '/dev/tenstorrent/2', '--raw')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[TELEMETRY] chip0: source raw, %d tags, decode check ok' % len(tags()), result.stdout)
        self.assertIn('[TELEMETRY] chip2: unreadable: Could not open chip on pci interface 2', result.stdout)
        self.assertIn('aiclk_mhz=1300', result.stdout)
        self.assertIn('limited_by=tdp', result.stdout)
        self.assertRegex(result.stdout, r'chip0 raw 14 AICLK\s+0x00000546')
        dead = self.run_module('check', '--nodes', '/dev/tenstorrent/2')
        self.assertEqual(dead.returncode, 1)

    def test_check_without_pyluwen_exits_2_and_says_which_interpreter(self):
        env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-s', str(MODULE), 'check', '--nodes', '0'], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('pyluwen is not importable by', result.stderr)

    def test_sample_runs_until_sigterm_then_the_summary_gzips_the_csv(self):
        out = os.path.join(self.dir, 'telemetry-smoke')
        process = subprocess.Popen([sys.executable, '-s', str(MODULE), 'sample', '--out', out, '--nodes', '/dev/tenstorrent/0', '/dev/tenstorrent/1', '--interval', '0.1'],
                                   env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                path = os.path.join(out, ct.CSV_NAME)
                if os.path.exists(path) and len(read_file(path).splitlines()) >= 7:
                    break
                time.sleep(0.1)
            else:
                self.fail('the sampler wrote no rows')
        finally:
            process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0, process.stderr.read())
        meta = load_json(os.path.join(out, ct.META_NAME))
        self.assertEqual(meta['stop_reason'], 'SIGTERM')
        self.assertGreaterEqual(meta['ticks'], 3)
        summary = self.run_module('summary', out)
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertIn('[TELEMETRY] chip1: AICLK under load min/median 1300/1300 MHz', summary.stdout)
        self.assertIn('ENGAGED tdp 100.0% of busy samples', summary.stdout)
        self.assertTrue(os.path.exists(os.path.join(out, ct.CSV_GZ_NAME)))
        self.assertFalse(os.path.exists(os.path.join(out, ct.CSV_NAME)))

    def test_sample_stops_by_itself_when_its_parent_exits(self):
        out = os.path.join(self.dir, 'orphan')
        parent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        process = subprocess.Popen([sys.executable, '-s', str(MODULE), 'sample', '--out', out, '--nodes', '0', '--interval', '0.1', '--parent', str(parent.pid)],
                                   env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(1.0)
            self.assertIsNone(process.poll(), 'alive while its parent is')
            parent.kill()
            parent.wait()
            self.assertEqual(process.wait(timeout=30), 0)
        finally:
            if process.poll() is None:
                process.kill()
        self.assertEqual(load_json(os.path.join(out, ct.META_NAME))['stop_reason'], 'parent exited')


# ---------------------------------------------------------------- card_telemetry.sh

def pid_alive(pid):
    """Running, not gone and not a zombie that nothing has reaped (a container without an init leaves the orphans of a finished test as zombies)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open('/proc/%d/stat' % pid) as handle:
            return handle.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


@unittest.skipUnless(BASH, 'bash not found')
class ShellTests(Tmp):
    def setUp(self):
        super().setUp()
        self.fake = os.path.join(self.dir, 'fake')
        os.makedirs(self.fake)
        with open(os.path.join(self.fake, 'pyluwen.py'), 'w') as handle:
            handle.write(FAKE_MODULE % dict(here=str(HERE)))
        self.world_file = os.path.join(self.dir, 'world.json')
        with open(self.world_file, 'w') as handle:
            json.dump({'0': [tags()], '1': [tags(aiclk=1250, limiter='board_power', limit_mhz=1250)]}, handle)
        # the host's tt-smi: a script whose shebang is the interpreter that has pyluwen (here a wrapper that adds the fake module to python3)
        self.py = os.path.join(self.dir, 'smi-python')
        with open(self.py, 'w') as handle:
            handle.write('#!/bin/bash\nPYTHONPATH=%s FAKE_WORLD=%s exec %s "$@"\n' % (self.fake, self.world_file, sys.executable))
        os.chmod(self.py, 0o755)
        self.smi = os.path.join(self.dir, 'tt-smi')
        with open(self.smi, 'w') as handle:
            handle.write('#!%s\nprint("tt-smi")\n' % self.py)
        os.chmod(self.smi, 0o755)
        self.empty_path = os.path.join(self.dir, 'bin')       # a PATH with python3 (no pyluwen) and the tools the library needs, but no tt-smi
        os.makedirs(self.empty_path)
        for tool in ('python3', 'awk', 'sed', 'grep', 'nice', 'seq', 'sleep', 'timeout', 'cat', 'mkdir', 'getent', 'id', 'cut', 'dirname', 'tee', 'head', 'kill'):
            found = shutil.which(tool)
            if found:
                os.symlink(found, os.path.join(self.empty_path, tool))
        self.home = os.path.join(self.dir, 'home')       # a runner user's home with no tt-smi in it
        os.makedirs(self.home)

    def sh(self, body, env=None, timeout=90):
        script = '. %s\n%s' % (LIBRARY, body)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment.update(env or {})
        return subprocess.run([BASH, '-c', script], env=environment, capture_output=True, text=True, timeout=timeout)

    def test_off_is_a_no_op(self):
        out = os.path.join(self.dir, 'telemetry-off')
        result = self.sh('telemetry_start smoke %s /dev/tenstorrent/0; echo "pid=[$TELEMETRY_PID]"; telemetry_stop; echo done' % out, env=dict(TELEMETRY='', TELEMETRY_SMI=self.smi))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['pid=[]', 'done'])
        self.assertFalse(os.path.exists(out))

    def test_it_finds_the_interpreter_of_the_tt_smi_script_and_starts_samples_and_stops(self):
        out = os.path.join(self.dir, 'telemetry-smoke')
        body = ('telemetry_start smoke %s /dev/tenstorrent/0 /dev/tenstorrent/1; echo "python=$(telemetry_python)"; sleep 1.5; telemetry_stop; echo "pid=[$TELEMETRY_PID]"; '
                'telemetry_stop; echo twice') % out
        result = self.sh(body, env=dict(TELEMETRY='1', TELEMETRY_MS='100', TELEMETRY_SMI=self.smi, PATH=self.empty_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('python=%s' % self.py, result.stdout)
        self.assertRegex(result.stdout, r'\[TELEMETRY\] smoke: sampling 2 chip\(s\) every 0\.100 s with %s \(pid [0-9]+\) into telemetry-smoke' % re.escape(self.py))
        self.assertIn('chip0: source raw, %d tags, decode check ok' % len(tags()), result.stdout)
        self.assertRegex(result.stdout, r'\[TELEMETRY\] chip0: AICLK under load min/median 1350/1350 MHz')
        self.assertRegex(result.stdout, r'\[TELEMETRY\] chip1: AICLK under load min/median 1250/1250 MHz .*ENGAGED board_power 100\.0% of busy samples')
        self.assertTrue(result.stdout.rstrip().endswith('pid=[]\ntwice'), 'stop is idempotent and prints nothing the second time')
        for name in (ct.CSV_GZ_NAME, ct.SUMMARY_JSON, ct.SUMMARY_TXT, ct.META_NAME, 'preflight.log', 'sampler.log'):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        self.assertEqual(load_json(os.path.join(out, ct.META_NAME))['stop_reason'], 'SIGTERM')

    def test_no_interpreter_with_pyluwen_is_a_warning_and_a_note_never_a_failure(self):
        out = os.path.join(self.dir, 'telemetry-gate')
        result = self.sh('telemetry_start gate %s /dev/tenstorrent/0; echo "rc=$? pid=[$TELEMETRY_PID]"; telemetry_stop; telemetry_summaries %s' % (out, self.dir),
                         env=dict(TELEMETRY='1', TELEMETRY_SMI='', PATH=self.empty_path, HOME=self.dir, TELEMETRY_HOME=self.home))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('::warning::C2_TELEMETRY=1 but no interpreter with pyluwen was found', result.stdout)
        self.assertIn('rc=0 pid=[]', result.stdout)
        self.assertIn('UNAVAILABLE', result.stderr)
        self.assertIn('[TELEMETRY] gate: UNAVAILABLE', read_file(os.path.join(out, ct.SUMMARY_TXT)))
        self.assertIn('[TELEMETRY] gate: UNAVAILABLE', result.stdout.split('pid=[]')[1], 'the always-run step prints it too')

    def test_a_failed_preflight_starts_no_sampler(self):
        with open(self.world_file, 'w') as handle:
            json.dump({'0': None}, handle)
        out = os.path.join(self.dir, 'telemetry-gate')
        result = self.sh('telemetry_start gate %s /dev/tenstorrent/0; echo "pid=[$TELEMETRY_PID]"' % out, env=dict(TELEMETRY='1', TELEMETRY_SMI=self.smi, PATH=self.empty_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pid=[]', result.stdout)
        self.assertIn('the preflight read of the chips failed', result.stderr)
        self.assertIn('unreadable', read_file(os.path.join(out, 'preflight.log')))

    def test_the_sampler_ends_with_the_shell_that_started_it(self):
        out = os.path.join(self.dir, 'telemetry-orphan')
        result = self.sh('telemetry_start gate %s /dev/tenstorrent/0; echo "pid=$TELEMETRY_PID"; sleep 0.5' % out,
                         env=dict(TELEMETRY='1', TELEMETRY_MS='100', TELEMETRY_SMI=self.smi, PATH=self.empty_path))
        pid = int(re.search(r'pid=([0-9]+)', result.stdout).group(1))
        deadline = time.time() + 30
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.2)
        self.assertFalse(pid_alive(pid), 'the sampler outlived its step shell')
        self.assertEqual(load_json(os.path.join(out, ct.META_NAME))['stop_reason'], 'parent exited')

    def test_summaries_covers_only_directories_without_one(self):
        results = os.path.join(self.dir, 'results')
        done = os.path.join(results, 'telemetry-smoke')
        todo = os.path.join(results, 'telemetry-gate')
        for directory in (done, todo):
            run_sampler(FakePyluwen({0: ChipState()}), ['0'], directory, 0.25)
        with redirect_stdout(io.StringIO()):
            ct.main(['summary', done])
        result = self.sh('telemetry_summaries %s' % results)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('telemetry-gate', result.stdout)
        self.assertNotIn('telemetry-smoke', result.stdout)
        self.assertTrue(os.path.exists(os.path.join(todo, ct.SUMMARY_JSON)))


# ---------------------------------------------------------------- C2_TELEMETRY in the job file and the workflow

def read(**values):
    values.setdefault('C2_IMAGE_TAG', 'v1-tel')
    return job.read_job(values, ['general', 'general-tp4'], meshes={'general': None, 'general-tp4': 'P150x4'}, envs={})


class JobKeyTests(unittest.TestCase):
    def test_off_is_the_job_as_it_was(self):
        outputs = read(C2_ACTIONS='status smoke')
        self.assertEqual((outputs['telemetry'], outputs['telemetry_ms']), ('', ''))
        for actions in ('status', 'status gate', 'smoke', 'replay'):
            self.assertEqual(read(C2_ACTIONS=actions, C2_TELEMETRY='')['telemetry'], '')

    def test_on_needs_a_smoke_or_a_gate_and_defaults_to_one_second(self):
        for actions in ('smoke', 'status smoke', 'gate', 'reset smoke gate'):
            outputs = read(C2_ACTIONS=actions, C2_TELEMETRY='1', C2_CARDS='pair')
            self.assertEqual((outputs['telemetry'], outputs['telemetry_ms']), ('1', '1000'), actions)
        for actions in ('status', 'status reset', 'replay', 'push'):
            with self.assertRaisesRegex(job.JobError, 'smoke and gate steps: C2_ACTIONS has neither'):
                read(C2_ACTIONS=actions, C2_TELEMETRY='1')
        with self.assertRaisesRegex(job.JobError, 'smoke and gate steps: C2_ACTIONS has neither'):
            read(C2_ACTIONS='prefix', C2_PREFIX_PROFILE='general', C2_PREFIX_BASELINE='general', C2_TELEMETRY='1')

    def test_the_flag_is_empty_or_1_and_nothing_else(self):
        for bad in ('0', 'yes', 'true', '2', 'on'):
            with self.assertRaisesRegex(job.JobError, 'empty or 1'):
                read(C2_ACTIONS='smoke', C2_TELEMETRY=bad)

    def test_the_interval(self):
        self.assertEqual(read(C2_ACTIONS='smoke', C2_TELEMETRY='1', C2_TELEMETRY_MS='250')['telemetry_ms'], '250')
        for low_or_high in ('99', '10001', '0', '-5', 'x', '1.5'):
            with self.assertRaises(job.JobError):
                read(C2_ACTIONS='smoke', C2_TELEMETRY='1', C2_TELEMETRY_MS=low_or_high)
        self.assertEqual(read(C2_ACTIONS='smoke', C2_TELEMETRY='1', C2_TELEMETRY_MS='100')['telemetry_ms'], '100')
        self.assertEqual(read(C2_ACTIONS='smoke', C2_TELEMETRY='1', C2_TELEMETRY_MS='10000')['telemetry_ms'], '10000')
        with self.assertRaisesRegex(job.JobError, 'set C2_TELEMETRY=1'):
            read(C2_ACTIONS='smoke', C2_TELEMETRY_MS='500')

    def test_the_keys_are_documented_where_the_others_are(self):
        self.assertIn('C2_TELEMETRY ', job.__doc__)
        self.assertIn('C2_TELEMETRY_MS', job.__doc__)

    def test_the_other_outputs_are_unchanged_by_the_keys(self):
        base = read(C2_ACTIONS='smoke')
        on = read(C2_ACTIONS='smoke', C2_TELEMETRY='1', C2_TELEMETRY_MS='500')
        self.assertEqual(sorted(k for k in base if base[k] != on[k]), ['telemetry', 'telemetry_ms'])


def step(name):
    text = WORKFLOW.read_text(encoding='utf-8')
    start = text.index('      - name: ' + name + '\n')
    end = text.find('\n      - ', start)
    return text[start:end + 1] if end >= 0 else text[start:]


SMOKE_PAIR, SMOKE_QUAD, GATE = 'Smoke on cards M+A', 'Smoke on the four-card set', "Run the gate in the agent's container shape"


class WorkflowTests(unittest.TestCase):
    def test_each_card_step_passes_the_two_outputs_and_starts_after_its_holder_check_and_before_its_first_container(self):
        for name, holder, first in ((SMOKE_PAIR, 'sudo -n fuser', 'docker run -d'), (SMOKE_QUAD, 'card_set_unheld', 'docker run -d'),
                                    (GATE, 'card_set_unheld', 'python3 scripts/ci/c2_serving_gate.py')):
            text = step(name)
            self.assertIn('TELEMETRY: ${{ steps.job.outputs.telemetry }}', text, name)
            self.assertIn('TELEMETRY_MS: ${{ steps.job.outputs.telemetry_ms }}', text, name)
            self.assertEqual(text.count('telemetry_start '), 1, name)
            self.assertLess(text.index('. scripts/ci/card_telemetry.sh'), text.index('telemetry_start '), name)
            self.assertLess(text.index('telemetry_start '), text.index(first), name)
            self.assertLess(text.index(holder), text.index('telemetry_start '), '%s: after the holder check' % name)

    def test_the_sampler_is_stopped_where_the_workload_ends_and_in_the_exit_trap_of_the_smokes(self):
        pair, quad, gate = step(SMOKE_PAIR), step(SMOKE_QUAD), step(GATE)
        self.assertIn("trap 'telemetry_stop; docker logs", pair)
        self.assertIn("trap 'telemetry_stop; kill \"$load_pid\"", quad)
        self.assertLess(pair.index('tee "$results/smoke.log"'), pair.index('\n          telemetry_stop\n'))
        self.assertLess(quad.index('|| smoke_status=$?'), quad.index('\n          telemetry_stop\n'))
        self.assertLess(quad.index('\n          telemetry_stop\n'), quad.index('if [ "$smoke_status" != 0 ]'), 'also summarised when the smoke client failed')
        self.assertLess(gate.index('tee "$RUNNER_TEMP/c2-results/gate.log"'), gate.index('\n          telemetry_stop\n'))
        self.assertIn('trap remove_gate_containers EXIT', gate, 'the gate step keeps its own trap')

    def test_the_nodes_are_the_steps_own_resolved_cards(self):
        self.assertIn('telemetry_start smoke "$results/telemetry-smoke" "$m" "$a"', step(SMOKE_PAIR))
        self.assertIn('telemetry_start smoke "$results/telemetry-smoke" "${CARD_SET_NODES[@]}"', step(SMOKE_QUAD))
        gate = step(GATE)
        self.assertIn('telemetry_nodes=("${CARD_SET_NODES[@]}")', gate)
        self.assertIn('telemetry_nodes=("$m" "$a")', gate)
        self.assertIn('telemetry_start gate "$RUNNER_TEMP/c2-results/telemetry-gate" "${telemetry_nodes[@]}"', gate)

    def test_the_containers_are_untouched_by_the_sidecar(self):
        """Telemetry must not change what the workload runs: no docker run argument, environment variable or mount mentions it."""
        for name in (SMOKE_PAIR, SMOKE_QUAD):
            text = step(name)
            run = text[text.index('docker run -d'):text.index('started=$(date')]
            self.assertNotIn('telemetry', run.lower(), name)
        gate_call = step(GATE)[step(GATE).index('python3 scripts/ci/c2_serving_gate.py'):]
        self.assertNotIn('telemetry', gate_call.split('| tee', 1)[0].lower())

    def test_the_always_run_step_summarises_what_a_failed_step_left_and_comes_before_the_upload(self):
        text = WORKFLOW.read_text(encoding='utf-8')
        summary = step('Telemetry summaries')
        self.assertIn("if: always() && steps.job.outputs.telemetry == '1'", summary)
        self.assertIn('telemetry_summaries "$RUNNER_TEMP/c2-results"', summary)
        self.assertLess(text.index('- name: Start the node agent'), text.index('- name: Telemetry summaries'))
        self.assertLess(text.index('- name: Telemetry summaries'), text.index('- name: Upload results'))

    def test_the_workflow_documents_the_key(self):
        text = WORKFLOW.read_text(encoding='utf-8')
        self.assertIn('C2_TELEMETRY=1', text.split('\non:\n', 1)[0])
        self.assertNotIn('\r', text)
        try:
            import yaml
        except ImportError:
            return
        self.assertIn('jobs', yaml.safe_load(text))


# ---------------------------------------------------------------- the two job templates and the manifest

class TemplateTests(unittest.TestCase):
    def templates(self):
        return sorted(TEMPLATES.glob('*.env'))

    def test_both_templates_pass_the_job_parser_with_telemetry_on(self):
        found = {path.name: path for path in self.templates()}
        self.assertEqual(sorted(found), ['TEL1-prefill-ladder-solo-telemetry.env', 'TEL2-concurrent8-steady-telemetry.env'])
        for path in found.values():
            result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(path), str(PROFILES_FILE)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, '%s: %s' % (path.name, result.stderr))
            outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
            self.assertEqual((outputs['telemetry'], outputs['telemetry_ms']), ('1', '1000'), path.name)
            self.assertEqual((outputs['cards'], outputs['actions']), ('quad', 'reset smoke'), path.name)
            self.assertEqual(outputs['profile'], 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic', path.name)

    def test_the_tests_are_the_solo_ladder_and_the_eight_live_steady_window(self):
        solo = job.parse_env((TEMPLATES / 'TEL1-prefill-ladder-solo-telemetry.env').read_text())
        steady = job.parse_env((TEMPLATES / 'TEL2-concurrent8-steady-telemetry.env').read_text())
        self.assertEqual(solo['C2_SMOKE_TESTS'], 'warmup,prefill_ladder_solo')
        self.assertEqual(steady['C2_SMOKE_TESTS'], 'warmup,concurrent8_steady')
        self.assertIn('C2_SMOKE_PARTIAL', solo, 'the ladder knowingly omits concurrent8_steady')
        self.assertNotIn('C2_SMOKE_PARTIAL', steady)
        self.assertEqual(solo['C2_IMAGE_TAG'], steady['C2_IMAGE_TAG'])

    def test_the_order_file_lists_both_and_a_template_never_names_a_host(self):
        order = (TEMPLATES / 'ORDER.txt').read_text()
        for name in ('TEL1-prefill-ladder-solo-telemetry', 'TEL2-concurrent8-steady-telemetry'):
            self.assertRegex(order, r'(?m)^%s (stop|soft) tp4-fusion-1 [0-9]+$' % name)
        for path in self.templates() + [TEMPLATES / 'ORDER.txt']:
            self.assertNotRegex(path.read_text(), r'[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+|[.]local|/home/|blackhole-[0-9A-F]{8}', path.name)

    def test_the_manifest_has_only_keys_the_generator_places_and_names_the_test_module(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest['wp'], 'TEL')
        self.assertEqual(manifest['branch'], 'tp4/card-telemetry')
        self.assertEqual(manifest['tests'], ['test_card_telemetry'])
        self.assertTrue((HERE / (manifest['tests'][0] + '.py')).exists())
        self.assertFalse(manifest.get('levers'), 'the sidecar has no profile twin: it is a job key')
        self.assertFalse(manifest.get('image_files'), 'it runs on the host, from the checkout: the image carries none of it')


class HygieneTests(unittest.TestCase):
    """The repo is public and the results are a public artifact: nothing here may name a host, an address, a serial, a registry or a user path."""

    def test_the_new_files_name_no_host_address_serial_registry_or_home(self):
        files = [MODULE, LIBRARY, Path(__file__), MANIFEST] + sorted(TEMPLATES.glob('*'))
        for path in files:
            text = path.read_text(encoding='utf-8')
            for pattern in (r'\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b', r'\bzot\.', r'/home/[a-z]', r'blackhole-[0-9A-Fa-f]{16}', r'[.]local\b(?!/bin)',
                            r'gh[ps]_[A-Za-z0-9]', r'[.]thatch|[.]lan\b|[.]internal\b'):
                match = re.search(pattern, text)
                self.assertIsNone(match, '%s: %r' % (path.name, match and match.group(0)))

    def test_the_sampler_records_no_identity_column(self):
        for column in ct.COLUMNS:
            self.assertNotRegex(column, r'board_id|serial|asic_id|bdf|^node|^host$|hostname|path|address')


if __name__ == '__main__':
    unittest.main()
