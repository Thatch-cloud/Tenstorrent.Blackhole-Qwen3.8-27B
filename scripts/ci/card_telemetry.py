"""card_telemetry: a read-only, per-chip ARC telemetry sampler for the card steps of qwen-c2-serving.yml (C2_TELEMETRY=1), and its summary.

Why: a prefill op profile against its unprofiled twin showed a uniform device-side slowdown in the sustained run that scales with SDPA time. Throttling is one
candidate (the profiled run idles between layers): the Blackhole firmware lowers AICLK under power, current, thermal and GDDR limits, and can insert kernel NOPs.
This samples what the firmware itself publishes, once a second, for the whole step, and says which limiter (if any) held the clock down.

    python3 -s scripts/ci/card_telemetry.py check   --nodes /dev/tenstorrent/0 [...] [--raw]      one read of every chip, decoded; exit 1 if none could be read
    python3 -s scripts/ci/card_telemetry.py sample  --out DIR --nodes ... [--interval S] [--parent PID]   until SIGTERM / SIGINT / the parent's death
    python3 -s scripts/ci/card_telemetry.py summary DIR [--keep-csv]       DIR/telemetry-summary.{json,txt}, one line per chip on stdout, the CSV gzipped

`sample` and `check` need pyluwen (the interpreter of the host's tt-smi has it: scripts/ci/card_telemetry.sh finds it); `summary` is stdlib only and runs under the host's python3.
`sample` writes DIR/telemetry.csv (one row per chip per sample, COLUMNS), DIR/telemetry-meta.json and never raises into the job: a chip that cannot be read is an `error` row.
No board id, serial, ASIC id, host name or path is recorded (the results are a public artifact): a chip is its index in the --nodes list.

READ-ONLY, AND HOW IT IS THE SAME PATH AS TT-SMI. The ARC firmware (tt-zephyr-platforms lib/tenstorrent/bh_arc/telemetry.c, update_telemetry(), every telem_update_interval =
100 ms) writes a table into its CSM and publishes its address in SCRATCH_RAM[13] (status_reg.h TELEMETRY_TABLE_REG_ADDR; SCRATCH_RAM[12] holds the data block's). A reader needs only AXI reads over the PCIe BAR; no ARC message is sent, nothing is
written, no mailbox is touched, so it cannot race the workload's own ARC messages. That is luwen's Blackhole get_telemetry() (luwen crates/luwen-api/src/chip/blackhole.rs:
axi_read SCRATCH_RAM[13] -> table pointer, 0x10000000..0x1007FFFF CSM window check, version, entry_count, (tag, offset) pairs, data words), which tt-smi calls on its pyluwen
path (tt-smi tt_smi/backend.py get_smbus_board_info: `pyluwen_chip.as_bh().get_telemetry()`; its refresh loop, update_telem(), re-reads the already-open chips every
interval, and its default UMD path does the same table read as get_arc_telemetry_reader().read_entry(tag)). The host's telemetry collector runs tt-smi against every board every
30 s while the cards serve, so this read is the one the rig does all day. PciChip(pci_interface=N) is luwen's open (a file descriptor, the BAR mapping, one 2 MiB TLB and the
DMA transfer buffers; Chip::open only translates addresses: no hardware traffic) and is what tt-smi's own reset path uses (tt_smi/reset.py). It is opened and dropped for EVERY
sample, as the collector does, so nothing is held across a container start or a reset and a chip that goes away costs one error row, not the sampler.

WHY NOT pyluwen's get_telemetry() ALONE. luwen's TelemetryTags (telemetry_tags.rs) and Telemetry struct (crates/luwen-api/src/chip/mod.rs, bind/pyluwen/src/lib.rs `Telemetry`)
stop at tag 64. The AICLK arbiter tags the firmware publishes from v19.12.0 (65 to 77: effective min/max arbiter, enabled-arbiter masks, the target-frequency reason, the
host fmax, the kernel NOP throttler) are dropped by its `_ => ()` arm, and `Telemetry.throttler` / `.faults` are never filled on Blackhole (blackhole.rs has no arm for them).
So the table is read here exactly as luwen reads it, keeping every tag by id. The first successful read of each chip is checked against pyluwen's own decode of the static
tags (STATIC_CHECKS); a disagreement is written to the meta file and printed, so a wrong decode cannot pass unseen. If the raw read fails, the struct of get_telemetry() is the
fallback source (no arbiter columns, `source` = struct).

FIELD MAP (firmware tag from tt-zephyr-platforms v19.12.0 lib/tenstorrent/bh_arc/telemetry.h; unit and packing from telemetry.c update_telemetry(); pyluwen/tt-smi names):
  column                tag  firmware meaning                               pyluwen Telemetry attr      tt-smi key / decode (tt_smi/backend.py get_bh_chip_telemetry)
  aiclk_mhz              14  AICLK, low 16 bits MHz (current PLL rate)       aiclk                       AICLK & 0xFFFF
  aiclk_fmax_mhz         63  AICLK_LIMIT_MAX (fwtable asic_fmax)             aiclk_limit_max             AICLK_LIMIT_MAX
  arb_min_mhz/arb_min    65  AICLK_ARB_MIN: low 16 = MHz, high 16 = the      (not exposed)               -
                             effective minimum arbiter (aiclk_arb_min)
  arb_max_mhz/arb_max    66  AICLK_ARB_MAX: low 16 = MHz, high 16 = the      (not exposed)               -
                             effective maximum arbiter (aiclk_arb_max)
  en_min_arb/en_max_arb  67/68  bitmask of enabled min / max arbiters        (not exposed)               -
  ppm_reason/ppm_arb     69  AICLK_PPM_INFO: low 16 = arbiter, high 16 =     (not exposed)               -
                             targ_freq_reason (why the target is what it is)
  host_fmax_mhz          70  HOST_AICLK_LIMIT (0 = no host ceiling)          (not exposed)               -
  vcore_mv                6  VCORE in mV                                     vcore                       VCORE / 1000 -> V
  power_w                 7  "TDP": the VCORE rail power now, W (truncated)   tdp                         TDP & 0xFFFF (ASIC power)
  board_power_w          54  INPUT_POWER: whole board, W                     input_power                 INPUT_POWER (board_power)
  tdc_a                   8  "TDC": the VCORE rail current now, A            tdc                         TDC & 0xFFFF
  asic_temp_c            11  ASIC_TEMPERATURE, signed 16.16 fixed point C    asic_temperature            convert_signed_16_16_to_float
  gddr_temp_c            51  MAX_GDDR_TEMP, C (else the max byte of 42..45)  max_gddr_temp, gddr*_temp   MAX_GDDR_TEMP
  gddr_io_w          73+74  GDDR west + east IO rail power, W                (not exposed)               -
  tdp_limit_w            64  TDP_LIMIT_MAX (the throttler's power limit)      tdp_limit_max               TDP_LIMIT_MAX
  tdc_limit_a            55  TDC_LIMIT_MAX                                   tdc_limit_max               TDC_LIMIT_MAX
  thm_limit_c            56  THM_LIMIT_THROTTLE                              thm_limit_throttle          THM_LIMIT_THROTTLE
  board_power_limit_w    53  BOARD_POWER_LIMIT                               board_power_limit           BOARD_POWER_LIMIT
  nop_starts             76  NOP_START_COUNT: kernel NOP throttle starts since boot (cumulative, complete)  (not exposed)
  nop_on_ms              77  NOP_ON_DURATION: ms NOPs were on in the last update window                      (not exposed)
  kernel_throttler       75  KERNEL_THROTTLER: bit 0 enabled, 31:16 stop-NOPs MHz                            (not exposed)
  therm_trips            60  THERM_TRIP_COUNT (cumulative)                   therm_trip_count            THERM_TRIP_COUNT
  fan_pct / fan_rpm      31/41  fan target % / RPM, 0xFFFFFFFF = no fan control  fan_speed / fan_rpm     FAN_RPM & 0xFFFF
  heartbeat              32  TIMER_HEARTBEAT, +1 per telemetry update (10 per second)  timer_heartbeat   TIMER_HEARTBEAT
Derived columns: busy (the minimum arbiter is at fmax: the host asked for a busy clock, firmware aiclk_arb_min_busy) and limited_by (the effective maximum arbiter's name
when its frequency is strictly below the effective minimum one's, which is exactly when firmware CalculateTargAiclk lets it lower the target; '-' when it is not, blank when
tags 65 and 66 are absent). Both come from tags 65/66, whose packing the header documents. TAG_AICLK_PPM_INFO's bit layout is NOT documented in the header (it is a C bitfield:
arbiter in the low 16 bits and the reason in the high 16 on a little-endian target, which is what ppm_arb and ppm_reason assume), so it only corroborates: the summary counts the
limited samples whose PPM reason is max_arb with the same arbiter ('ppm agrees'); a count that is not all of them says the PPM layout is read wrong, not that the limiter is.

THE LIMITERS (aiclk_ppm.h v19.12.0, enum aiclk_arb_max, in this order and never reordered: ARB_MAX_NAMES), and what feeds each (throttler.c): tdp = VCORE rail power against
tdp_limit; fast_tdc and tdc = VCORE rail current against the fast and the slow TDC limits; thm = ASIC temperature against thm_limit; board_power = input power against the
board limit; gddr_thm = the hottest GDDR die; voltage = a static ceiling from the V/F curve at vdd_max; doppler_slow = the averaged board power when the Doppler feature is on
(it then replaces tdp, fast_tdc, tdc and board_power); doppler_critical = a 2x / 2.5x board-power excursion (clock to fmin and kernel NOPs); host_fmax = a ceiling the host asked
for by ARC message (disabled unless one was sent). Kernel NOP throttling (nop_starts, nop_on_ms) slows the Tensix kernels WITHOUT lowering AICLK. The firmware's per-limiter
counters (final_arbiter_count) are only readable by ARC message, which this never sends.

LIMITS OF A ONE-SECOND SAMPLE. The table is refreshed every 100 ms and the arbiters decide on a millisecond loop, so a limiter that engages for a few milliseconds between two
samples is not seen; nop_starts and therm_trips are cumulative and ARE complete. The percentages are shares of the samples, not of time. A busy flag is a client holding the clock
up (the serving container does, idle or not), so the 'loaded' statistics also need the VCORE power to be at least half of the chip's own maximum in the run.

Python 3.7 syntax (the rig host's interpreter runs `summary`).
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import shutil
import signal
import statistics
import sys
import time

FIRMWARE_TAGS = 'tt-zephyr-platforms v19.12.0'

# telemetry.h v19.12.0 (TAG_COUNT 78): the names, for --raw and the meta file.
TAG_NAMES = {
    1: 'BOARD_ID_HIGH', 2: 'BOARD_ID_LOW', 3: 'ASIC_ID', 4: 'HARVESTING_STATE', 5: 'UPDATE_TELEM_SPEED', 6: 'VCORE', 7: 'TDP', 8: 'TDC',
    9: 'VDD_LIMITS', 10: 'THM_LIMIT_SHUTDOWN', 11: 'ASIC_TEMPERATURE', 12: 'VREG_TEMPERATURE', 13: 'BOARD_TEMPERATURE', 14: 'AICLK',
    15: 'AXICLK', 16: 'ARCCLK', 17: 'L2CPUCLK0', 18: 'L2CPUCLK1', 19: 'L2CPUCLK2', 20: 'L2CPUCLK3', 21: 'ETH_LIVE_STATUS', 22: 'GDDR_STATUS',
    23: 'GDDR_SPEED', 24: 'ETH_FW_VERSION', 25: 'GDDR_FW_VERSION', 26: 'DM_APP_FW_VERSION', 27: 'DM_BL_FW_VERSION', 28: 'FLASH_BUNDLE_VERSION',
    29: 'CM_FW_VERSION', 30: 'L2CPU_FW_VERSION', 31: 'FAN_SPEED', 32: 'TIMER_HEARTBEAT', 33: 'TELEM_ENUM_COUNT', 34: 'ENABLED_TENSIX_COL',
    35: 'ENABLED_ETH', 36: 'ENABLED_GDDR', 37: 'ENABLED_L2CPU', 38: 'PCIE_USAGE', 39: 'INPUT_CURRENT', 40: 'NOC_TRANSLATION', 41: 'FAN_RPM',
    42: 'GDDR_0_1_TEMP', 43: 'GDDR_2_3_TEMP', 44: 'GDDR_4_5_TEMP', 45: 'GDDR_6_7_TEMP', 46: 'GDDR_0_1_CORR_ERRS', 47: 'GDDR_2_3_CORR_ERRS',
    48: 'GDDR_4_5_CORR_ERRS', 49: 'GDDR_6_7_CORR_ERRS', 50: 'GDDR_UNCORR_ERRS', 51: 'MAX_GDDR_TEMP', 52: 'ASIC_LOCATION', 53: 'BOARD_POWER_LIMIT',
    54: 'INPUT_POWER', 55: 'TDC_LIMIT_MAX', 56: 'THM_LIMIT_THROTTLE', 57: 'FW_BUILD_DATE', 58: 'TT_FLASH_VERSION', 59: 'ENABLED_TENSIX_ROW',
    60: 'THERM_TRIP_COUNT', 61: 'ASIC_ID_HIGH', 62: 'ASIC_ID_LOW', 63: 'AICLK_LIMIT_MAX', 64: 'TDP_LIMIT_MAX', 65: 'AICLK_ARB_MIN',
    66: 'AICLK_ARB_MAX', 67: 'ENABLED_MIN_ARB', 68: 'ENABLED_MAX_ARB', 69: 'AICLK_PPM_INFO', 70: 'HOST_AICLK_LIMIT', 71: 'SMBUS_ERRORS',
    72: 'GDDR_MRISC_NOC2AXI_PORT', 73: 'GDDR_WEST_IO_POWER', 74: 'GDDR_EAST_IO_POWER', 75: 'KERNEL_THROTTLER', 76: 'NOP_START_COUNT',
    77: 'NOP_ON_DURATION'}
TAG = dict((name, number) for number, name in TAG_NAMES.items())

# aiclk_ppm.h v19.12.0 ("The order of these enum values must be preserved for compatibility").
ARB_MAX_NAMES = ('fmax', 'tdp', 'fast_tdc', 'tdc', 'thm', 'board_power', 'voltage', 'gddr_thm', 'doppler_slow', 'doppler_critical', 'host_fmax')
ARB_MIN_NAMES = ('fmin', 'busy')
REASON_NAMES = ('min_arb', 'max_arb', 'fmin', 'sweep', 'forced')   # enum targ_freq_reason
REASON_MAX_ARB = 'max_arb'

# luwen blackhole.rs get_telemetry(): the table is behind SCRATCH_RAM[13], in the CSM window below.
SCRATCH_TABLE_POINTER = 'arc_ss.reset_unit.SCRATCH_RAM[13]'
CSM_WINDOW = (0x10000000, 0x1007FFFF)
MAX_ENTRIES = 512    # sanity: the table has TAG_COUNT = 78 today
NO_FAN = 0xFFFFFFFF

# (tag, pyluwen Telemetry attribute, mask): static values both decoders must agree on, checked on a chip's first read.
STATIC_CHECKS = ((TAG['AICLK_LIMIT_MAX'], 'aiclk_limit_max', 0xFFFFFFFF), (TAG['TDC_LIMIT_MAX'], 'tdc_limit_max', 0xFFFFFFFF),
                 (TAG['TDP_LIMIT_MAX'], 'tdp_limit_max', 0xFFFFFFFF), (TAG['THM_LIMIT_THROTTLE'], 'thm_limit_throttle', 0xFFFFFFFF),
                 (TAG['ENABLED_TENSIX_COL'], 'tensix_enabled_col', 0xFFFFFFFF), (TAG['FLASH_BUNDLE_VERSION'], 'fw_bundle_version', 0xFFFFFFFF),
                 (TAG['BOARD_POWER_LIMIT'], 'board_power_limit', 0xFFFFFFFF))
# pyluwen Telemetry attribute -> tag, for the fallback source (what get_telemetry() exposes of the tags above).
STRUCT_TAGS = (('vcore', 6), ('tdp', 7), ('tdc', 8), ('asic_temperature', 11), ('aiclk', 14), ('fw_bundle_version', 28), ('fan_speed', 31),
               ('timer_heartbeat', 32), ('tensix_enabled_col', 34), ('fan_rpm', 41), ('gddr01_temp', 42), ('gddr23_temp', 43), ('gddr45_temp', 44),
               ('gddr67_temp', 45), ('max_gddr_temp', 51), ('board_power_limit', 53), ('input_power', 54), ('tdc_limit_max', 55),
               ('thm_limit_throttle', 56), ('therm_trip_count', 60), ('aiclk_limit_max', 63), ('tdp_limit_max', 64))

COLUMNS = ('ts', 't_s', 'chip', 'ok', 'heartbeat', 'aiclk_mhz', 'aiclk_fmax_mhz', 'arb_min_mhz', 'arb_min', 'arb_max_mhz', 'arb_max', 'ppm_reason', 'ppm_arb',
           'host_fmax_mhz', 'en_min_arb', 'en_max_arb', 'busy', 'limited_by', 'vcore_mv', 'power_w', 'board_power_w', 'tdc_a', 'asic_temp_c',
           'gddr_temp_c', 'gddr_io_w', 'tdp_limit_w', 'tdc_limit_a', 'thm_limit_c', 'board_power_limit_w', 'nop_starts', 'nop_on_ms',
           'kernel_throttler', 'therm_trips', 'fan_pct', 'fan_rpm', 'error')
DEFAULT_INTERVAL_S = 1.0
MIN_INTERVAL_S = 0.1
LOADED_POWER_SHARE = 0.5     # 'loaded' = busy and power at least this share of the chip's own maximum in the run
BELOW_FMAX_MHZ = 1.5         # an AICLK this far under fmax is 'below fmax'
NOT_LIMITED = '-'            # limited_by when the attribution tags are present and no limiter holds the clock (blank: no attribution)
TIMELINE_BUCKET_S = 60
CSV_NAME, CSV_GZ_NAME = 'telemetry.csv', 'telemetry.csv.gz'
META_NAME, SUMMARY_JSON, SUMMARY_TXT = 'telemetry-meta.json', 'telemetry-summary.json', 'telemetry-summary.txt'
ERROR_WIDTH = 160


def u32(data, index):
    """The little-endian 32-bit word at word `index` of a byte buffer (luwen's u32_from_slice)."""
    start = index * 4
    return int.from_bytes(bytes(data[start:start + 4]), 'little')


# ---------------------------------------------------------------- reading one chip

def read_table(bh):
    """{tag: raw word} from the chip's telemetry table, by the AXI reads luwen's Blackhole get_telemetry() makes (and no others). `bh` has axi_translate, axi_read32, axi_read."""
    pointer = bh.axi_read32(bh.axi_translate(SCRATCH_TABLE_POINTER).addr)
    if pointer == 0:
        raise RuntimeError('telemetry table not published (ARC boot incomplete)')
    if not CSM_WINDOW[0] <= pointer <= CSM_WINDOW[1]:
        raise RuntimeError('telemetry table pointer 0x%08x outside the CSM window' % pointer)
    count = bh.axi_read32(pointer + 4)
    if not 0 < count <= MAX_ENTRIES:
        raise RuntimeError('telemetry entry count %d is not plausible' % count)
    tags = bytearray((count + 1) * 4)
    bh.axi_read(pointer + 8, tags)
    data = bytearray((count + 1) * 4)
    bh.axi_read(pointer + 8 + count * 4, data)
    table = {}
    for index in range(count):
        entry = u32(tags, index)
        tag, offset = entry & 0xFFFF, (entry >> 16) & 0xFFFF
        if tag and offset <= count:
            table[tag] = u32(data, offset)
    return table


def struct_table(bh):
    """{tag: raw word} from pyluwen's own get_telemetry() struct: the fallback, with only the tags luwen decodes."""
    telemetry = bh.get_telemetry()
    table = {}
    for attribute, tag in STRUCT_TAGS:
        value = getattr(telemetry, attribute, None)
        if isinstance(value, int):
            table[tag] = value & 0xFFFFFFFF
    return table


def static_check(bh, table):
    """[] when every STATIC_CHECKS tag of `table` equals pyluwen's own decode, else the disagreements; the string 'skipped: <why>' when the library's read is not available
    (a missing method or a failed read says nothing about the raw decode)."""
    try:
        telemetry = bh.get_telemetry()
    except Exception as error:     # noqa: BLE001 - a missing method or a failed read only skips the check
        return 'skipped: get_telemetry %s' % ('is not in this pyluwen' if isinstance(error, (AttributeError, NotImplementedError)) else 'failed (%s)' % short(error))
    bad = []
    for tag, attribute, mask in STATIC_CHECKS:
        theirs = getattr(telemetry, attribute, None)
        if not isinstance(theirs, int) or tag not in table:
            continue
        if (theirs & mask) != (table[tag] & mask):
            bad.append('%s tag %d raw 0x%x pyluwen 0x%x' % (attribute, tag, table[tag], theirs))
    return bad


def check_text(result):
    """The decode-check result as the meta file and the summary say it: ok, MISMATCH: <attributes>, skipped: <why>, or not checked."""
    if result is None:
        return 'not checked'
    if isinstance(result, str):
        return result
    return 'ok' if not result else 'MISMATCH: ' + '; '.join(result)


def short(error):
    """One line of an error: pyluwen raises Exception for a failed read but a Rust panic (a missing device node's config space, an old driver) arrives as pyo3's PanicException,
    which is a BaseException, so the callers catch that too and the class name is kept for it."""
    text = ' '.join(str(error).split()) or error.__class__.__name__
    if not isinstance(error, Exception):
        text = '%s: %s' % (error.__class__.__name__, text)
    return text[:ERROR_WIDTH]


def signed_16_16(word):
    """tt_tools_common convert_signed_16_16_to_float: None for the firmware's error value 0x80000000."""
    if word >= 1 << 31:
        word -= 1 << 32
    return None if word == -(1 << 31) else word / 65536.0


def name_of(names, index):
    return names[index] if 0 <= index < len(names) else 'id%d' % index


def decode(table):
    """The COLUMNS values (None where the tag is absent) for one table, plus the derived busy and limited_by."""
    row = dict((column, None) for column in COLUMNS)

    def word(name):
        return table.get(TAG[name])

    def low(name):
        value = word(name)
        return None if value is None else value & 0xFFFF

    row['heartbeat'] = word('TIMER_HEARTBEAT')
    row['aiclk_mhz'] = low('AICLK')
    row['aiclk_fmax_mhz'] = word('AICLK_LIMIT_MAX')
    for tag, mhz, who, names in (('AICLK_ARB_MIN', 'arb_min_mhz', 'arb_min', ARB_MIN_NAMES), ('AICLK_ARB_MAX', 'arb_max_mhz', 'arb_max', ARB_MAX_NAMES)):
        value = word(tag)
        if value is not None:
            row[mhz], row[who] = value & 0xFFFF, name_of(names, value >> 16)
    ppm = word('AICLK_PPM_INFO')
    if ppm is not None:
        row['ppm_arb'], row['ppm_reason'] = ppm & 0xFFFF, name_of(REASON_NAMES, ppm >> 16)
    row['host_fmax_mhz'] = word('HOST_AICLK_LIMIT')
    row['en_min_arb'], row['en_max_arb'] = word('ENABLED_MIN_ARB'), word('ENABLED_MAX_ARB')
    row['vcore_mv'] = word('VCORE')
    row['power_w'] = low('TDP')
    row['board_power_w'] = word('INPUT_POWER')
    row['tdc_a'] = low('TDC')
    if word('ASIC_TEMPERATURE') is not None:
        row['asic_temp_c'] = signed_16_16(word('ASIC_TEMPERATURE'))
    gddr = word('MAX_GDDR_TEMP')
    if gddr is None:
        packed = [table[tag] for tag in (TAG['GDDR_0_1_TEMP'], TAG['GDDR_2_3_TEMP'], TAG['GDDR_4_5_TEMP'], TAG['GDDR_6_7_TEMP']) if tag in table]
        if packed:
            gddr = max((value >> shift) & 0xFF for value in packed for shift in (0, 8, 16, 24))
    row['gddr_temp_c'] = gddr
    west, east = word('GDDR_WEST_IO_POWER'), word('GDDR_EAST_IO_POWER')
    row['gddr_io_w'] = None if west is None and east is None else (west or 0) + (east or 0)
    row['tdp_limit_w'], row['tdc_limit_a'] = word('TDP_LIMIT_MAX'), word('TDC_LIMIT_MAX')
    row['thm_limit_c'], row['board_power_limit_w'] = word('THM_LIMIT_THROTTLE'), word('BOARD_POWER_LIMIT')
    row['nop_starts'], row['nop_on_ms'], row['therm_trips'] = word('NOP_START_COUNT'), word('NOP_ON_DURATION'), word('THERM_TRIP_COUNT')
    if word('KERNEL_THROTTLER') is not None:
        row['kernel_throttler'] = '0x%08x' % word('KERNEL_THROTTLER')
    for column, tag in (('fan_pct', 'FAN_SPEED'), ('fan_rpm', 'FAN_RPM')):
        value = word(tag)
        row[column] = None if value is None or value == NO_FAN else (value & 0xFFFF if tag == 'FAN_RPM' else value)
    derive(row)
    return row


def derive(row):
    """busy and limited_by (module docstring). Both stay None when the tags they need are absent."""
    fmax, arb_min, arb_max = row['aiclk_fmax_mhz'], row['arb_min_mhz'], row['arb_max_mhz']
    if fmax and arb_min is not None:
        row['busy'] = 1 if arb_min >= fmax - 1 else 0
    if arb_min is not None and arb_max is not None:
        # firmware CalculateTargAiclk: the target is the effective minimum arbiter, lowered to the effective maximum one only when that is STRICTLY lower
        row['limited_by'] = row['arb_max'] if arb_max < arb_min else NOT_LIMITED
    return row


def read_chip(pyluwen, index, check=None):
    """(decoded row, table, source, check result) for the chip at /dev/tenstorrent/<index>; raises on any failure. The chip is dropped when this returns."""
    chip = pyluwen.PciChip(pci_interface=index)
    bh = chip.as_bh()
    if bh is None:
        raise RuntimeError('not a Blackhole chip')
    try:
        table, source = read_table(bh), 'raw'
    except Exception as error:     # noqa: BLE001 - the library's own decode is the fallback for any failure of the raw read
        table, source = struct_table(bh), 'struct'
        if not table:
            raise RuntimeError('raw read failed (%s) and the struct has no tags' % short(error))
    result = static_check(bh, table) if check else None
    return decode(table), table, source, result


# ---------------------------------------------------------------- the sampler

def node_index(node):
    """/dev/tenstorrent/N (or N) -> N."""
    match = re.fullmatch(r'(?:/dev/tenstorrent/)?([0-9]{1,4})', str(node))
    if not match:
        raise ValueError('%r is not /dev/tenstorrent/<n>' % (node,))
    return int(match.group(1))


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return ('%.3f' % value).rstrip('0').rstrip('.')
    return value


class Sampler(object):
    def __init__(self, pyluwen, nodes, out_dir, interval=DEFAULT_INTERVAL_S, parent=None, clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        self.pyluwen, self.indices = pyluwen, [node_index(node) for node in nodes]
        self.out_dir, self.interval, self.parent = out_dir, max(float(interval), MIN_INTERVAL_S), parent
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.stop_reason = None
        self.ticks = self.ok_rows = self.error_rows = 0
        self.checked = {}
        self.sources = {}
        self.started = None

    def request_stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason

    def parent_gone(self):
        if not self.parent:
            return False
        try:
            os.kill(self.parent, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    def meta(self, final=False):
        meta = dict(tool='card_telemetry', tags=FIRMWARE_TAGS, interval_s=self.interval, chips=len(self.indices), started_unix=self.started,
                    pyluwen=pyluwen_version(self.pyluwen), python=sys.version.split()[0], source=dict((str(chip), source) for chip, source in sorted(self.sources.items())),
                    decode_check=dict((str(chip), result) for chip, result in sorted(self.checked.items())))
        if final:
            meta.update(stopped_unix=self.clock(), stop_reason=self.stop_reason, ticks=self.ticks, rows_ok=self.ok_rows, rows_error=self.error_rows)
        return meta

    def write_meta(self, final=False):
        path = os.path.join(self.out_dir, META_NAME)
        with open(path + '.tmp', 'w') as handle:
            json.dump(self.meta(final), handle, indent=1, sort_keys=True)
        os.replace(path + '.tmp', path)

    def sample_chip(self, chip):
        """The csv row for one chip now: the decoded values, or an error row."""
        try:
            row, _, source, result = read_chip(self.pyluwen, self.indices[chip], check=chip not in self.checked)
        except BaseException as error:     # noqa: BLE001 - every failure of a read is data, never the end of the sampler
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            row = dict((column, None) for column in COLUMNS)
            row['error'] = short(error)
            return row, False
        self.sources[chip] = source
        if chip not in self.checked:
            self.checked[chip] = check_text(result)
            if isinstance(result, list) and result:
                sys.stderr.write('[TELEMETRY] chip%d: raw decode disagrees with pyluwen: %s\n' % (chip, '; '.join(result)))
        return row, True

    def run(self, duration=None):
        os.makedirs(self.out_dir, exist_ok=True)
        self.started = self.clock()
        begin = self.monotonic()
        self.write_meta()
        with open(os.path.join(self.out_dir, CSV_NAME), 'w', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(COLUMNS)
            handle.flush()
            due = begin
            while self.stop_reason is None:
                for chip in range(len(self.indices)):
                    row, ok = self.sample_chip(chip)
                    row.update(ts=self.clock(), t_s=self.monotonic() - begin, chip=chip, ok=1 if ok else 0)
                    writer.writerow([fmt(row[column]) for column in COLUMNS])
                    if ok:
                        self.ok_rows += 1
                    else:
                        self.error_rows += 1
                handle.flush()
                self.ticks += 1
                if self.ticks == 1:
                    self.write_meta()
                if duration is not None and self.monotonic() - begin >= duration:
                    self.request_stop('duration')
                due += self.interval
                now = self.monotonic()
                if due < now:
                    due = now      # behind (a slow read): do not burst to catch up
                while self.stop_reason is None and self.monotonic() < due:
                    if self.parent_gone():
                        self.request_stop('parent exited')
                        break
                    self.sleep(min(0.2, max(due - self.monotonic(), 0.0)))
        self.write_meta(final=True)
        return 0


def pyluwen_version(pyluwen):
    version = getattr(pyluwen, '__version__', None)
    if version:
        return str(version)
    try:
        from importlib import metadata
        return metadata.version('pyluwen')
    except Exception:     # noqa: BLE001 - the version is a note, not a requirement
        return 'unknown'


# ---------------------------------------------------------------- the summary

def number(text):
    if text == '' or text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value == int(value) and '.' not in text else value


def load_rows(directory):
    """[row dict] from DIR/telemetry.csv or .csv.gz (values as numbers, strings, or None when blank)."""
    path = os.path.join(directory, CSV_NAME)
    if os.path.exists(path):
        handle = open(path, newline='')
    elif os.path.exists(os.path.join(directory, CSV_GZ_NAME)):
        handle = io.TextIOWrapper(gzip.open(os.path.join(directory, CSV_GZ_NAME), 'rb'), newline='')
    else:
        return None
    with handle:
        return [dict((key, number(value)) for key, value in row.items()) for row in csv.DictReader(handle)]


def pct(part, whole):
    return 0.0 if not whole else 100.0 * part / whole


def med(values):
    if not values:
        return None
    value = statistics.median(values)
    return int(value) if value == int(value) else round(value, 1)


def summarize_chip(rows):
    """The statistics of one chip's rows (module docstring: busy, loaded, limited_by)."""
    good = [row for row in rows if row.get('ok') == 1]
    out = dict(samples=len(rows), ok=len(good), errors=len(rows) - len(good))
    errors = [row['error'] for row in rows if row.get('error')]
    if errors:
        out['first_error'] = errors[0]
    if not good:
        return out

    def values(name, subset=None):
        return [row[name] for row in (good if subset is None else subset) if isinstance(row.get(name), (int, float))]

    out['span_s'] = round(good[-1]['ts'] - good[0]['ts'], 1) if isinstance(good[0].get('ts'), (int, float)) else None
    fmax = max(values('aiclk_fmax_mhz') or [0]) or None
    out['fmax_mhz'] = fmax
    busy_known = any(row.get('busy') in (0, 1) for row in good)
    busy = [row for row in good if row.get('busy') == 1]
    powers = values('power_w')
    top_power = max(powers) if powers else None
    basis = busy if busy_known else good
    loaded = [row for row in basis if isinstance(row.get('power_w'), (int, float)) and top_power and row['power_w'] >= LOADED_POWER_SHARE * top_power] or basis
    out.update(busy_known=busy_known, busy=len(busy), loaded=len(loaded))
    aiclk_loaded, aiclk_busy, aiclk_all = values('aiclk_mhz', loaded), values('aiclk_mhz', busy), values('aiclk_mhz')
    out['aiclk_loaded'] = dict(min=min(aiclk_loaded), median=med(aiclk_loaded), n=len(aiclk_loaded)) if aiclk_loaded else None
    out['aiclk_busy'] = dict(min=min(aiclk_busy), median=med(aiclk_busy), n=len(aiclk_busy)) if aiclk_busy else None
    out['aiclk_all'] = dict(min=min(aiclk_all), median=med(aiclk_all), n=len(aiclk_all)) if aiclk_all else None
    for column, key in (('power_w', 'power_max_w'), ('board_power_w', 'board_power_max_w'), ('tdc_a', 'tdc_max_a'), ('asic_temp_c', 'temp_max_c'),
                        ('gddr_temp_c', 'gddr_temp_max_c'), ('vcore_mv', 'vcore_max_mv'), ('fan_rpm', 'fan_rpm_max')):
        found = values(column)
        out[key] = round(max(found), 1) if found else None
    vcore = values('vcore_mv', loaded)
    out['vcore_loaded_min_mv'] = min(vcore) if vcore else None
    out['limits'] = dict((key, max(found) if found else None) for key, found in (
        ('tdp_w', values('tdp_limit_w')), ('tdc_a', values('tdc_limit_a')), ('thm_c', values('thm_limit_c')), ('board_power_w', values('board_power_limit_w'))))
    attribution = any(row.get('limited_by') not in (None, '') for row in good)
    out['attribution'] = attribution
    masks = [row['en_max_arb'] for row in good if isinstance(row.get('en_max_arb'), int)]
    mask = max(set(masks), key=masks.count) if masks else None
    out['armed'] = None if mask is None else [name for index, name in enumerate(ARB_MAX_NAMES) if (mask >> index) & 1 and name != 'fmax']
    host_fmax = values('host_fmax_mhz')
    out['host_fmax_mhz'] = max(host_fmax) if host_fmax else None
    denominator = len(busy) if busy_known else len(good)
    held = {}
    for row in good:
        name = row.get('limited_by')
        if name not in (None, '', NOT_LIMITED) and (not busy_known or row.get('busy') == 1):
            held.setdefault(name, []).append(row)
    # corroboration by TAG_AICLK_PPM_INFO (module docstring): of the limited samples that carry it, those it names the same way
    corroborated = [row for items in held.values() for row in items if row.get('ppm_reason') is not None]
    out['ppm_agrees'] = dict(of=len(corroborated), agree=len([row for row in corroborated if row['ppm_reason'] == REASON_MAX_ARB
                                                              and name_of(ARB_MAX_NAMES, row['ppm_arb']) == row['limited_by']]))
    out['limiters'] = {}
    for name, items in sorted(held.items()):
        floor = values('arb_max_mhz', items)
        out['limiters'][name] = dict(samples=len(items), share_pct=round(pct(len(items), denominator), 1), arb_mhz_min=min(floor) if floor else None)
    below = [row for row in (busy if busy_known else good) if fmax and isinstance(row.get('aiclk_mhz'), (int, float)) and row['aiclk_mhz'] < fmax - BELOW_FMAX_MHZ]
    out['below_fmax'] = dict(samples=len(below), share_pct=round(pct(len(below), denominator), 1))
    starts = values('nop_starts')
    out['nop_starts_delta'] = (max(starts) - min(starts)) if starts else None
    out['nop_on_samples'] = len([v for v in values('nop_on_ms') if v > 0]) if values('nop_on_ms') else None
    trips = values('therm_trips')
    out['therm_trip_delta'] = (max(trips) - min(trips)) if trips else None
    out['engaged'] = bool(out['limiters'] or (out['nop_starts_delta'] or 0) > 0 or (out['nop_on_samples'] or 0) > 0 or (out['therm_trip_delta'] or 0) > 0)
    out['timeline'] = timeline(good)
    return out


def timeline(good):
    """Per TIMELINE_BUCKET_S bucket: samples, AICLK min and median, max power, max temp, limiter samples (the JSON only)."""
    if not good or not all(isinstance(row.get('t_s'), (int, float)) for row in good):
        return []
    buckets = {}
    for row in good:
        buckets.setdefault(int(row['t_s'] // TIMELINE_BUCKET_S), []).append(row)
    out = []
    for key, rows in sorted(buckets.items()):
        aiclk = [row['aiclk_mhz'] for row in rows if isinstance(row.get('aiclk_mhz'), (int, float))]
        power = [row['power_w'] for row in rows if isinstance(row.get('power_w'), (int, float))]
        temp = [row['asic_temp_c'] for row in rows if isinstance(row.get('asic_temp_c'), (int, float))]
        out.append(dict(t_s=key * TIMELINE_BUCKET_S, ts=rows[0].get('ts'), n=len(rows), aiclk_min=min(aiclk) if aiclk else None, aiclk_median=med(aiclk), power_max_w=max(power) if power else None,
                        temp_max_c=round(max(temp), 1) if temp else None, limited=len([row for row in rows if row.get('limited_by') not in (None, '', NOT_LIMITED)]),
                        busy=len([row for row in rows if row.get('busy') == 1])))
    return out


def summarize(rows, meta=None):
    chips = {}
    for row in rows:
        chips.setdefault(row.get('chip'), []).append(row)
    return dict(meta=meta or {}, chips=dict((str(chip), summarize_chip(items)) for chip, items in sorted(chips.items(), key=lambda pair: str(pair[0]))))


def chip_line(chip, stat, interval):
    head = '[TELEMETRY] chip%s: ' % chip
    if not stat.get('ok'):
        return head + 'NO SAMPLES (%d rows, first error: %s)' % (stat['samples'], stat.get('first_error', 'none'))
    fmax = stat['fmax_mhz'] if stat['fmax_mhz'] else '?'
    if stat['aiclk_loaded']:
        aiclk = stat['aiclk_loaded']
        parts = ['AICLK under load min/median %s/%s MHz (fmax %s; %d of %d samples)' % (aiclk['min'], aiclk['median'], fmax, stat['loaded'], stat['ok'])]
    else:
        aiclk = stat['aiclk_all'] or dict(min='?', median='?')
        parts = ['AICLK never under load (no busy sample); all samples min/median %s/%s MHz (fmax %s; %d samples)' % (aiclk['min'], aiclk['median'], fmax, stat['ok'])]
    power = 'power max %s W' % stat['power_max_w'] if stat['power_max_w'] is not None else 'power n/a'
    if stat['board_power_max_w'] is not None:
        power += ' (board %s W)' % stat['board_power_max_w']
    if stat['tdc_max_a'] is not None:
        power += ', TDC max %s A' % stat['tdc_max_a']
    parts.append(power)
    temp = 'temp max %s C' % stat['temp_max_c'] if stat['temp_max_c'] is not None else 'temp n/a'
    if stat['gddr_temp_max_c'] is not None:
        temp += ' (GDDR %s C)' % stat['gddr_temp_max_c']
    parts.append(temp)
    parts.append('throttler: ' + verdict(stat))
    if stat['errors']:
        parts.append('%d unreadable' % stat['errors'])
    return head + ' | '.join(parts)


def verdict(stat):
    bits = []
    for name, info in sorted(stat['limiters'].items(), key=lambda pair: -pair[1]['samples']):
        bits.append('%s %.1f%% of %s samples%s' % (name, info['share_pct'], 'busy' if stat['busy_known'] else 'all',
                                                   ' (arbiter down to %s MHz)' % info['arb_mhz_min'] if info.get('arb_mhz_min') is not None else ''))
    if (stat['nop_starts_delta'] or 0) > 0:
        bits.append('kernel NOPs started %d times' % stat['nop_starts_delta'])
    elif (stat['nop_on_samples'] or 0) > 0:
        bits.append('kernel NOPs on in %d samples' % stat['nop_on_samples'])
    if (stat['therm_trip_delta'] or 0) > 0:
        bits.append('thermal trips +%d' % stat['therm_trip_delta'])
    if bits:
        agrees = stat.get('ppm_agrees') or {}
        if agrees.get('of'):
            bits.append('PPM info agrees in %d of %d' % (agrees['agree'], agrees['of']))
        return 'ENGAGED ' + ', '.join(bits)
    if not stat['attribution']:
        note = 'NO ATTRIBUTION (no arbiter tags in this firmware)'
        if stat['below_fmax']['samples']:
            note += ', AICLK below fmax in %.1f%% of samples' % stat['below_fmax']['share_pct']
        return note
    note = 'none engaged'
    if stat['below_fmax']['samples']:
        note += ' (AICLK below fmax in %d busy samples, no limiter named)' % stat['below_fmax']['samples']
    if stat.get('armed'):
        note += ' (armed: %s)' % ', '.join(stat['armed'])
    if stat.get('host_fmax_mhz'):
        note += ' (host fmax ceiling %s MHz)' % stat['host_fmax_mhz']
    return note


def render(summary, label=''):
    meta = summary.get('meta') or {}
    lines = []
    interval = meta.get('interval_s')
    chips = summary['chips']
    total = sum(stat['samples'] for stat in chips.values())
    sources = sorted(set((meta.get('source') or {}).values())) or ['?']
    checks = sorted(set((meta.get('decode_check') or {}).values())) or ['?']
    lines.append('[TELEMETRY]%s %d chip(s), %d rows, interval %s s, source %s, pyluwen %s, decode check %s' % (
        ' ' + label if label else '', len(chips), total, interval if interval is not None else '?', '/'.join(sources), meta.get('pyluwen', '?'), '; '.join(checks)))
    for chip, stat in chips.items():
        lines.append(chip_line(chip, stat, interval))
    if not chips:
        lines.append('[TELEMETRY] no rows: the sampler wrote nothing readable')
    return lines


def summary_command(options):
    directory = options.directory
    rows = load_rows(directory)
    if rows is None:
        if os.path.exists(os.path.join(directory, SUMMARY_TXT)):
            with open(os.path.join(directory, SUMMARY_TXT)) as handle:
                sys.stdout.write(handle.read())
            return 0
        print('[TELEMETRY] %s: no telemetry.csv (the sampler never started or wrote nothing)' % directory)
        return 0
    meta = {}
    try:
        with open(os.path.join(directory, META_NAME)) as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        pass
    summary = summarize(rows, meta)
    label = os.path.basename(os.path.normpath(directory))
    lines = render(summary, label)
    with open(os.path.join(directory, SUMMARY_JSON), 'w') as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    with open(os.path.join(directory, SUMMARY_TXT), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    sys.stdout.write('\n'.join(lines) + '\n')
    csv_path = os.path.join(directory, CSV_NAME)
    if os.path.exists(csv_path) and not options.keep_csv:
        with open(csv_path, 'rb') as source, gzip.open(os.path.join(directory, CSV_GZ_NAME), 'wb') as target:
            shutil.copyfileobj(source, target)
        os.remove(csv_path)
    return 0


# ---------------------------------------------------------------- commands

def import_pyluwen():
    try:
        import pyluwen
    except ImportError as error:
        sys.stderr.write('[TELEMETRY] pyluwen is not importable by %s: %s\n' % (sys.executable, error))
        return None
    return pyluwen


def check_command(options):
    pyluwen = import_pyluwen()
    if pyluwen is None:
        return 2
    readable = 0
    for chip, node in enumerate(options.nodes):
        try:
            row, table, source, result = read_chip(pyluwen, node_index(node), check=True)
        except BaseException as error:     # noqa: BLE001 - reported per chip
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            print('[TELEMETRY] chip%d: unreadable: %s' % (chip, short(error)))
            continue
        readable += 1
        status = check_text(result)
        print('[TELEMETRY] chip%d: source %s, %d tags, decode check %s' % (chip, source, len(table), status))
        print('[TELEMETRY] chip%d: %s' % (chip, ' '.join('%s=%s' % (column, fmt(row[column])) for column in COLUMNS[4:-1] if row[column] is not None)))
        if options.raw:
            for tag in sorted(table):
                print('[TELEMETRY] chip%d raw %2d %-24s 0x%08x' % (chip, tag, TAG_NAMES.get(tag, '?'), table[tag]))
    return 0 if readable else 1


def sample_command(options):
    pyluwen = import_pyluwen()
    if pyluwen is None:
        return 2
    sampler = Sampler(pyluwen, options.nodes, options.out, options.interval, options.parent)
    for number_, name in ((signal.SIGTERM, 'SIGTERM'), (signal.SIGINT, 'SIGINT')):
        signal.signal(number_, lambda signum, frame, name=name: sampler.request_stop(name))
    return sampler.run(options.duration)


def parse(argv):
    parser = argparse.ArgumentParser(description='read-only ARC telemetry sampler for the card steps (module docstring)')
    commands = parser.add_subparsers(dest='command')
    commands.required = True
    check = commands.add_parser('check', help='one decoded read of every chip')
    check.add_argument('--nodes', nargs='+', required=True, help='/dev/tenstorrent/<n> (or <n>), in the order the chips are labelled chip0, chip1, ...')
    check.add_argument('--raw', action='store_true', help='also print every tag of the table')
    sample = commands.add_parser('sample', help='sample until stopped')
    sample.add_argument('--out', required=True)
    sample.add_argument('--nodes', nargs='+', required=True)
    sample.add_argument('--interval', type=float, default=DEFAULT_INTERVAL_S, help='seconds between samples (at least %s)' % MIN_INTERVAL_S)
    sample.add_argument('--parent', type=int, default=None, help='stop when this pid is gone (the step shell)')
    sample.add_argument('--duration', type=float, default=None, help='stop after this many seconds (tests)')
    summary = commands.add_parser('summary', help='summarise DIR and gzip its csv')
    summary.add_argument('directory')
    summary.add_argument('--keep-csv', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    options = parse(sys.argv[1:] if argv is None else argv)
    try:
        return dict(check=check_command, sample=sample_command, summary=summary_command)[options.command](options)
    except ValueError as error:
        sys.stderr.write('card_telemetry: %s\n' % error)
        return 2


if __name__ == '__main__':
    sys.exit(main())
