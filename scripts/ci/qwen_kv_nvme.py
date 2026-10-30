"""The NVMe cold tier of the prefix state (docs/tp4-kv-nvme.md): spill below the host KV tier, restore on a miss, keep a returning session warm for days.

Where it sits. The host KV tier (qwen_prefix_registry.TierStore, 32 GiB at the ship setting) holds the attention-KV blocks vLLM evicted from the device pool, and the
checkpoint store (16 GiB) holds the GDN checkpoints that make a hit exact. Both are RAM and both are lost at a restart. This module adds a third level under them, on a
RAW NVMe namespace (qwen_nvme_store: no filesystem, a journal, O_DIRECT, 4 KiB-aligned extents):

  SPILL    A host-tier record the tier evicts is handed to offer() (the TierStore's spill sink) on the scheduler thread, which only enqueues it: the record's slab slot is
           RETAINED (the slab is mapped with spare slots for exactly this) until the writer thread has written it, so there is no copy and no wait. When the queue is full
           the record is simply dropped, as the host tier always did. A GDN checkpoint leaves the checkpoint store the same way (offer_checkpoint): on a budget eviction,
           and with the KV record of its boundary block. Nothing here is ever on a decode round's critical path; the device is written by one thread.
  RESTORE  A returning request's effective hit (TierHooks.effective_hit) reaches through the host tier's consecutive blocks AND the NVMe's; a checkpoint missing from the
           store is read back from the NVMe (adopt). The missing blocks are read into slab slots of the host tier (backend 'host': NVMe -> host buffer), committed as host
           records and restored to the device by the existing code, byte for byte. Everything the host tier checks, it still checks; the NVMe adds its own digest check.
  THROUGH  Write-through, the DEFAULT (QWEN_PREFIX_NVME_SPILL=through; =evict is the explicit alternative): a finished request's blocks are copied toward the drive at once
           (QWEN_PREFIX_NVME_THROUGH_IDLE_S=0: idle 0 means 'queue them now', not 'off') - a bounded slice of its still-cached device blocks per scheduler step is read into the
           host tier (TierHooks.copy_idle) and every record that enters the host tier is queued to the writer (offer_through) - so a warm copy exists before the eviction that would
           otherwise be the first chance to write it, and a restart loses only the last seconds.
  STOP     A graceful stop (SIGTERM, an engine shutdown, an interpreter exit) drains: arm_stop_hooks() chains the SIGTERM handler (so the drain starts at the signal, in parallel
           with the engine's own shutdown) and registers an exit finalizer; shutdown() stops taking offers, flushes the queue for up to QWEN_PREFIX_NVME_DRAIN_S seconds (default 4,
           inside the platform's ~10 s stop grace), journals the TTL clocks, snapshots the index when time remains and closes the device - so a deploy restart finds the warm tier.
  DEVICE   The drive may vanish at run time (a PCI rescan after a card reset took it once: the controller is disabled, the namespace reads 0 bytes). The tier notices through
           the failing IO itself, a periodic look at the device's node and sysfs entries (device_problem: never an IO), and latches off at once with ONE line and a counter
           (nvme_device_lost); the host tier serves as if the NVMe level did not exist, the scheduler never waits on the drive, and a restart with a live device brings it back.
  GUARDS   The tier keeps prompt-derived data on a drive for days. A node that advertises zero-retention signs teardown attestations that nothing survived; the tier therefore
           REFUSES to start (a loud line and a counter; the engine serves without it) when the process sees a zero-retention signal (zero_retention_signals) unless
           QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION=1, and it needs QWEN_PREFIX_NVME_RETENTION_ACK=1 always: the serving process cannot know what its node agent advertises.
  AT REST  The drive must be an Opal self-encrypting drive with locking enabled and currently unlocked (QWEN_PREFIX_NVME_REQUIRE_SED, on by default; 0 is a gate's own): sed_check asks
           the drive (IOC_OPAL_DISCOVERY, read-only) or, where the container lacks CAP_SYS_ADMIN, reads the status file the host's boot unit writes; anything it cannot show is
           NOT encrypted and the tier never opens the drive (one REFUSED line, nvme_refused_not_encrypted). TRIM of freed extents is on by default (QWEN_PREFIX_NVME_DISCARD).
  BACKENDS RestoreBackend: 'host' and 'p2p' (QWEN_PREFIX_NVME_P2P=1, docs/tp4-p2p-restore.md: the drive's O_DIRECT read lands in the cards' staging buffers through the published BAR4
           window, a device checksum is compared with the one stored at spill, and a device op copies the bytes into the KV blocks; the host backend is its fallback). TierHooks asks
           backend.direct: a direct backend restores NVMe-resident blocks straight to the freshly allocated device blocks instead of through the host (TierHooks.load_direct). With the
           flag on the drive holds each block CHIP-MAJOR ([chip][cache][slice], layout 'chipmajor-v1', its own fingerprint): chip d's share is one aligned run a card can read alone.

KEYS. The host tier's own: vLLM's chained block hash (it carries the tenant's cache salt, so tenants never meet), kind KV; the checkpoint's boundary block hash, kind CKPT.
Every entry also carries a LAYOUT FINGERPRINT (KV_EPOCH, model repo + revision, drafter checkpoint, KV layout/dtype/page size, mesh, GDN dtype, the few env switches that
change stored bytes by design - FINGERPRINTED_ENV): an entry of another fingerprint is dropped at open and never restored. The image build and the profile NAME are NOT in
it: a deploy keeps the warm tier unless it changes what the stored bytes mean, and KV_EPOCH is the by-hand switch for that (docs/tp4-kv-nvme.md, "The layout fingerprint").
The tenant tag is stored only as an 8-byte hash.

Pure python; imports nothing outside the standard library (torch and ttnn only through the objects the model hands in).
"""

import array
import atexit
import collections
import ctypes
import errno
import hashlib
import importlib
import json
import os
import re
import signal
import struct
import sys
import threading
import time

import qwen_nvme_store as store_module
from qwen_nvme_store import CKPT, KV, NvmeStore, PutItem, tenant_hash

BLOCK = 64
CHUNK = 2048

ENV_SWITCH = 'QWEN_PREFIX_NVME'
ENV_DEVICE = 'QWEN_PREFIX_NVME_DEVICE'
ENV_TTL_S = 'QWEN_PREFIX_NVME_TTL_S'
ENV_MAX_AGE_S = 'QWEN_PREFIX_NVME_MAX_AGE_S'
ENV_MAX_BYTES = 'QWEN_PREFIX_NVME_MAX_BYTES'
ENV_VERIFY = 'QWEN_PREFIX_NVME_VERIFY'
ENV_RESTORE = 'QWEN_PREFIX_NVME_RESTORE'
ENV_FREE_FLOOR = 'QWEN_PREFIX_NVME_FREE_FLOOR_BYTES'
ENV_QUEUE = 'QWEN_PREFIX_NVME_QUEUE_BYTES'
ENV_THREADS = 'QWEN_PREFIX_NVME_THREADS'
ENV_FORMAT = 'QWEN_PREFIX_NVME_FORMAT'
ENV_DISCARD = 'QWEN_PREFIX_NVME_DISCARD'
ENV_COMPAT = 'QWEN_PREFIX_NVME_COMPAT'
ENV_CMD_DIR = 'QWEN_PREFIX_NVME_CMD_DIR'
ENV_AUDIT = 'QWEN_PREFIX_NVME_AUDIT'
ENV_SPILL = 'QWEN_PREFIX_NVME_SPILL'
ENV_THROUGH_BLOCKS = 'QWEN_PREFIX_NVME_THROUGH_BLOCKS'
ENV_THROUGH_IDLE_S = 'QWEN_PREFIX_NVME_THROUGH_IDLE_S'
ENV_RETENTION_ACK = 'QWEN_PREFIX_NVME_RETENTION_ACK'
ENV_ALLOW_ZERO_RETENTION = 'QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION'
ENV_DRAIN_S = 'QWEN_PREFIX_NVME_DRAIN_S'
ENV_REQUIRE_SED = 'QWEN_PREFIX_NVME_REQUIRE_SED'
ENV_SED_STATUS = 'QWEN_PREFIX_NVME_SED_STATUS'
ENV_P2P = 'QWEN_PREFIX_NVME_P2P'
ENV_P2P_THREADS = 'QWEN_PREFIX_NVME_P2P_THREADS'
ENV_P2P_STAGING_MIB = 'QWEN_PREFIX_NVME_P2P_STAGING_MIB'
ENV_P2PMEM = 'QWEN_PREFIX_NVME_P2PMEM'
NAMES = (ENV_SWITCH, ENV_DEVICE, ENV_TTL_S, ENV_MAX_AGE_S, ENV_MAX_BYTES, ENV_VERIFY, ENV_RESTORE, ENV_FREE_FLOOR, ENV_QUEUE, ENV_THREADS, ENV_FORMAT, ENV_DISCARD,
         ENV_COMPAT, ENV_CMD_DIR, ENV_AUDIT, ENV_SPILL, ENV_THROUGH_BLOCKS, ENV_THROUGH_IDLE_S, ENV_RETENTION_ACK, ENV_ALLOW_ZERO_RETENTION, ENV_DRAIN_S, ENV_REQUIRE_SED, ENV_SED_STATUS,
         ENV_P2P, ENV_P2P_THREADS, ENV_P2P_STAGING_MIB, ENV_P2PMEM)
# What only a gate sets (the contract refuses them on a traffic profile even under the owner's waiver). The peer-to-peer restore is a gate's until a card has qualified it (docs/tp4-p2p-restore.md);
# its three knobs are refused as inert unless it is on, so they cannot reach a traffic profile without it.
GATE_ONLY = (ENV_AUDIT, ENV_FORMAT, ENV_CMD_DIR, ENV_ALLOW_ZERO_RETENTION, ENV_P2P)
# Names whose OFF value is the gate's own: a traffic profile may carry the name (at 1, or by default) but never 0. The encryption check is on unless a gate turns it off.
GATE_ONLY_VALUES = ((ENV_REQUIRE_SED, '0'),)
VERIFY_MODES = ('sample', 'all', 'off')
SPILL_MODES = ('evict', 'through')
DEFAULT_SPILL = 'through'            # owner, 2026-10-21: seats are released at turn end, so a finished session is copied to the drive straight away
DEFAULT_THROUGH_BLOCKS = 32          # device blocks one scheduler step may copy toward the drive (about 13 ms of device reads at 0.4 ms a block)
MAX_THROUGH_BLOCKS = 4096
DEFAULT_THROUGH_IDLE_S = 0           # a finished request's blocks are queued for the drive at once (0 = immediately; a larger value waits that long first)
IDLE_BACKLOG = 1024                  # finished requests remembered for the copy
# What tells a process it runs on a node that advertises zero-retention (see zero_retention_signals); the node agent's own attestation key is the node agent's secret and is
# never meant to be in a serving container, so its presence here is a conflict by itself.
ZERO_RETENTION_ENV = ('THATCH_ATTESTATION_PRIVATE_KEY', 'THATCH_ZERO_RETENTION', 'THATCH_NO_LOG')
ZERO_RETENTION_MARKER = '/models/.qwen-c2/zero-retention'
VERIFY_EVERY = 16
RESTORE_BACKENDS = ('host', 'p2p')
DEFAULT_TTL_S = 24 * 3600
MAX_TTL_S = 366 * 24 * 3600
DEFAULT_QUEUE_BYTES = 2 << 30
MIN_QUEUE_BYTES = 16 << 20
DEFAULT_THREADS = 4
DEFAULT_FREE_FLOOR_FRACTION = 0.05
DEFAULT_DRAIN_S = 4                  # a graceful stop flushes the queue for up to this long; the platform's stop grace is ~10 s and the engine's own shutdown shares it
MAX_DRAIN_S = 8
LEGACY_CLOSE_S = 10.0              # close() on a tier whose stop budget is 0 (hooks off) still flushes this long, as it always did
SNAPSHOT_BASE_S = 1.0                # a stop snapshots the index only when this much plus SNAPSHOT_ENTRY_S per entry is left of the drain budget
SNAPSHOT_ENTRY_S = 4e-6
DEFAULT_SED_STATUS = '/run/thatch-cache-drive/status.json'      # written by the host's boot unit; a container without CAP_SYS_ADMIN reads it instead of asking the drive
DEFAULT_DISCARD = True               # TRIM freed extents (delete, expiry, eviction): the encryption scoping's recommendation, and what keeps a full drive's write amplification down
IOC_OPAL_DISCOVERY = 0x401070EF      # _IOW('p', 239, struct opal_discovery {u64 data; u64 size;}): a Level 0 Discovery (a Security Receive: reads, changes no state)
OPAL_DISCOVERY_BYTES = 4096
HEALTH_EVERY_S = 2.0                 # how often poll() looks at the device's node and sysfs entries (no IO)
# errnos that mean the device itself is gone (a failing write with one of these latches the tier at once instead of after three strikes)
GONE_ERRNOS = (errno.ENODEV, errno.ENXIO, errno.ESHUTDOWN, errno.ENOMEDIUM)
LAYOUT = 'packed-v1'                 # the on-disk payload layout: the host tier's raw packed block ([cache tensor][chip][slice]); a new layout is a new fingerprint
LAYOUT_CHIPMAJOR = 'chipmajor-v1'    # the same bytes as [chip][cache tensor][slice] (QWEN_PREFIX_NVME_P2P=1): chip d's share of a block is one 4 KiB-aligned run (docs/tp4-p2p-restore.md)
DEFAULT_P2P_THREADS = 8              # reads in flight over all cards
DEFAULT_P2P_STAGING_MIB = 128        # staging buffer per card: two halves
P2P_BATCH_BYTES = 128 << 20          # a chip-major spill batch is converted in a second buffer of its size: keep batches smaller than the 512 MiB of the host layout
P2P_READ_GROUP = 16                  # blocks a chip-major host read converts at a time (the scratch it needs is this many blocks)
P2P_STRIKES = 3                      # consecutive failed peer-to-peer restores (or checksum mismatches on bytes the host path proves good) that latch the path off
P2P_MISMATCH_CHECKS = 8              # blocks a checksum mismatch re-reads through the host path to tell a bad drive from a bad path
CKPT_FORMAT = 1

# ==========================================================================================================================================================================
# KV_EPOCH: THE BY-HAND COMPATIBILITY SWITCH OF THE STORED KV. READ THIS BEFORE YOU SHIP A KERNEL OR A MODEL-PATH CHANGE.
#
# The NVMe tier keeps a session's KV and GDN checkpoints for days, ACROSS deploys: an image build and a serving-profile name are deliberately NOT part of the layout
# fingerprint. What keeps it honest is this number. BUMP IT (one line, in the same commit as the change) whenever a change alters what a prefill PUBLISHES:
#   * prefill NUMERICS: any change to what the prefill path computes that is not bit-for-bit what the old path computed - an attention, matmul, norm, activation or GDN kernel
#     or program config (fidelity, fp32 accumulation, summation or reduction order, topology, tile or chunk partitioning that moves a reduction), a dtype, a quantisation, a
#     different weights file, a rounding mode. A change you have PROVEN bit-exact (an audit twin, byte-for-byte) does not need a bump.
#   * the BYTE MEANING of anything stored: the KV block layout or cache-tensor order, the GDN state or conv-carry layout or dtype, the checkpoint object format, the
#     drafter window rows (which taps, their order, their alignment).
#   * the drafter's state layout, if a stored checkpoint ever carries drafter-side state beyond today's target-tap window rows.
# A bump makes every stored entry another fingerprint: dropped at the next open, never restored, a cold prefill for every returning session (and the drive is reusable at
# once). NOT bumping a real numerics change is the dangerous direction: a returning session is then restored from the OLD numerics' KV, so its answer continues from the old
# image's state and equals the old image's continuation, NOT the new image's own cold prefill of the same prompt; nothing at run time can tell. The UPGRADE card gate
# (docs/tp4-kv-nvme.md, section 12) is the check that two images are KV-compatible: a mismatch there means bump this number.
KV_EPOCH = 1

# The env switches whose VALUES change the stored bytes by design (a different computation, not a faster route to the same bytes), read from the process environment the
# tier is opened in (the profile's values over the image's defaults). They go in the fingerprint; a switch that is bit-exact against its baseline (qualified by an audit twin)
# does not, and the many that do not touch the prefill path do not either. scripts/ci/test_qwen_kv_compat.py classifies EVERY env name of the profile table and of the image's
# ENV defaults, and fails on a name it has not seen: a new switch must be put in one of its classes, and this tuple must equal its 'kv' class.
FINGERPRINTED_ENV = (
    'QWEN_SDPA_BF8',                      # the served KV and the prefill SDPA operands are bf8 (the dtype also shows in io.fingerprint; the switch is what selects it)
    'QWEN_GDN_PREFILL_MMRS',              # fused out-projection + reduce-scatter in the GDN prefill: a different fp32 summation order, not proven exact
    'QWEN_PREFIX_CKPT_PRECONVERTED',      # the checkpoint object's form (the model's device-ready host tensors, or torch tensors): a stored form restores only as itself
)
DRAFTER_FLAG = 'QWEN_FAST_DRAFTER_CHECKPOINT'
DRAFTER_MODULE = 'drafter_checkpoint'
SWEEP_EVERY_S = 60.0
CMD_EVERY_S = 2.0
MAX_BATCH_ITEMS = 256
MAX_BATCH_BYTES = 512 << 20
WRITER_YIELD_S = 0.005
WRITER_YIELD_MAX_S = 2.0
PATH = re.compile(r'(?!-)/[A-Za-z0-9_./:@+-]{1,200}')
TOKEN = re.compile(r'[A-Za-z0-9._-]{1,64}')
DIGITS = re.compile(r'[0-9]{1,18}')
PCI_ADDRESS = re.compile(r'(?:[0-9A-Fa-f]{4}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]')

# The counters the NVMe tier adds to the registry's snapshot (exported as qwen_prefix_<name>_total; ms timers as _seconds_total). Present only while the tier is on.
COUNTER_NAMES = (
    'nvme_spill_offered', 'nvme_spill_blocks', 'nvme_spill_bytes', 'nvme_spill_ms', 'nvme_spill_batches', 'nvme_spill_known', 'nvme_spill_dropped_useless',
    'nvme_spill_dropped_queue', 'nvme_spill_dropped_full', 'nvme_spill_failures', 'nvme_ckpt_spilled', 'nvme_ckpt_spill_bytes', 'nvme_ckpt_spill_failures',
    'nvme_ckpt_spill_dropped', 'nvme_restore_requests', 'nvme_restore_blocks', 'nvme_restore_bytes', 'nvme_restore_ms', 'nvme_restore_refused_room',
    'nvme_restore_refused_digest', 'nvme_restore_failures', 'nvme_miss_blocks', 'nvme_ckpt_restored', 'nvme_ckpt_restore_bytes', 'nvme_ckpt_restore_ms',
    'nvme_ckpt_restore_failures', 'nvme_digest_checks', 'nvme_digest_failures', 'nvme_evicted_lru', 'nvme_expired_ttl', 'nvme_expired_age', 'nvme_fp_drops',
    'nvme_deleted_tenant', 'nvme_wipes', 'nvme_audit_reads', 'nvme_audit_mismatches', 'nvme_latched', 'nvme_recovered_entries', 'nvme_recovered_records',
    'nvme_recover_dropped', 'nvme_journal_records', 'nvme_snapshots', 'nvme_store_hits', 'nvme_store_misses', 'nvme_discarded_bytes', 'nvme_through_queued',
    'nvme_through_known', 'nvme_through_dropped_queue', 'nvme_through_copied_blocks', 'nvme_through_copied_bytes', 'nvme_through_copy_ms', 'nvme_through_copy_steps',
    'nvme_through_failures', 'nvme_spill_retries', 'nvme_refused_zero_retention', 'nvme_unavailable', 'nvme_device_lost', 'nvme_refused_not_encrypted',
    # the peer-to-peer restore (QWEN_PREFIX_NVME_P2P): present, at 0, while the tier is on whether or not the path is
    'nvme_p2p_restores', 'nvme_p2p_blocks', 'nvme_p2p_bytes', 'nvme_p2p_read_ms', 'nvme_p2p_check_ms', 'nvme_p2p_write_ms', 'nvme_p2p_mismatch', 'nvme_p2p_late_data',
    'nvme_p2p_errors', 'nvme_p2p_fallbacks', 'nvme_p2p_latched', 'nvme_p2p_unavailable')
GAUGE_NAMES = ('nvme_on', 'nvme_bytes', 'nvme_entries', 'nvme_kv_entries', 'nvme_ckpt_entries', 'nvme_capacity_bytes', 'nvme_free_bytes', 'nvme_queue_bytes',
               'nvme_journal_used_bytes', 'nvme_largest_free_bytes', 'nvme_through_backlog_blocks', 'nvme_at_rest', 'nvme_p2p_on', 'nvme_p2p_map_verified')


def log(message, *values):
    try:
        sys.stderr.write('[PINDIAG] prefix: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------
# The flags
# ------------------------------------------------------------------------------------------------
class NvmePlan(object):
    """The NVMe tier's settings (nvme_plan)."""

    __slots__ = ('device', 'ttl_s', 'max_age_s', 'max_bytes', 'verify', 'restore', 'free_floor_bytes', 'queue_bytes', 'threads', 'format', 'discard', 'compat',
                 'cmd_dir', 'audit', 'direct', 'snap_bytes', 'journal_bytes', 'spill', 'through_blocks', 'through_idle_s', 'allow_zero_retention', 'drain_s', 'require_sed', 'sed_status',
                 'p2p', 'p2p_threads', 'p2p_staging_bytes', 'p2pmem')

    @property
    def through(self):
        return self.spill == 'through'

    def __init__(self):
        self.direct = None             # None: O_DIRECT on a block device, buffered on a file (tests set it)
        self.snap_bytes = None
        self.journal_bytes = None

    def describe(self):
        return dict((name, getattr(self, name)) for name in self.__slots__)


def _digits(env, name, default, low, high, problems):
    raw = env.get(name)
    if raw is None or raw == '':
        return default
    if not DIGITS.fullmatch(raw):
        problems.append('%s=%r is not a whole number in plain digits' % (name, raw))
        return default
    value = int(raw)
    if not low <= value <= high:
        problems.append('%s=%d is outside %d..%d' % (name, value, low, high))
        return default
    return value


def _switch(env, name, problems, default=False):
    raw = env.get(name)
    if raw is None or raw == '':
        return default
    if raw not in ('0', '1'):
        problems.append('%s=%r must be 0 or 1' % (name, raw))
        return default
    return raw == '1'


def parse(env):
    """(NvmePlan or None, [problem]) from a mapping of environment strings. None with no problem: the tier is not asked for. Every value is strict (plain digits, no sign,
    no exponent, no space); a name set while the switch is not 1 is a problem (it would be inert); QWEN_PREFIX_NVME_RESTORE=p2p is a problem (not built)."""
    problems = []
    named = [name for name in NAMES if env.get(name) not in (None, '')]
    if not named:
        return None, problems
    on = _switch(env, ENV_SWITCH, problems)
    if env.get(ENV_RETENTION_ACK) not in (None, '', '0', '1'):
        problems.append('%s=%r must be 1' % (ENV_RETENTION_ACK, env.get(ENV_RETENTION_ACK)))
    if not on:
        others = [name for name in named if name != ENV_SWITCH]
        if others:
            problems.append('%s set without %s=1: the NVMe tier would be inert' % (', '.join(others), ENV_SWITCH))
        return None, problems
    plan = NvmePlan()
    device = env.get(ENV_DEVICE)
    if not device:
        problems.append('%s=1 needs %s (the raw namespace, an absolute path)' % (ENV_SWITCH, ENV_DEVICE))
    elif not PATH.fullmatch(device):
        problems.append('%s=%r is not a plain absolute path' % (ENV_DEVICE, device))
    plan.device = device
    plan.ttl_s = _digits(env, ENV_TTL_S, DEFAULT_TTL_S, 1, MAX_TTL_S, problems)
    plan.max_age_s = _digits(env, ENV_MAX_AGE_S, 0, 0, MAX_TTL_S, problems)
    plan.max_bytes = _digits(env, ENV_MAX_BYTES, 0, 0, 1 << 60, problems)
    if plan.max_bytes and plan.max_bytes < (1 << 20):
        problems.append('%s=%d is under 1 MiB' % (ENV_MAX_BYTES, plan.max_bytes))
    verify = env.get(ENV_VERIFY) or 'sample'
    if verify not in VERIFY_MODES:
        problems.append('%s=%r must be one of %s' % (ENV_VERIFY, verify, ', '.join(VERIFY_MODES)))
    plan.verify = verify
    restore = env.get(ENV_RESTORE) or 'host'
    if restore not in RESTORE_BACKENDS:
        problems.append('%s=%r must be one of %s' % (ENV_RESTORE, restore, ', '.join(RESTORE_BACKENDS)))
    elif restore == 'p2p':
        problems.append('%s=p2p is not a setting: the peer-to-peer restore is %s=1, which keeps the host backend as its fallback (docs/tp4-p2p-restore.md)' % (ENV_RESTORE, ENV_P2P))
    plan.restore = restore
    floor = env.get(ENV_FREE_FLOOR)
    plan.free_floor_bytes = None if floor in (None, '') else _digits(env, ENV_FREE_FLOOR, 0, 0, 1 << 60, problems)
    plan.queue_bytes = _digits(env, ENV_QUEUE, DEFAULT_QUEUE_BYTES, MIN_QUEUE_BYTES, 1 << 40, problems)
    plan.threads = _digits(env, ENV_THREADS, DEFAULT_THREADS, 1, 64, problems)
    plan.format = 'force' if _switch(env, ENV_FORMAT, problems) else 'auto'
    plan.discard = _switch(env, ENV_DISCARD, problems, DEFAULT_DISCARD)
    plan.audit = _switch(env, ENV_AUDIT, problems)
    token = env.get(ENV_COMPAT)
    if token not in (None, '') and not TOKEN.fullmatch(token):
        problems.append('%s=%r is not a short token (letters, digits, . _ -)' % (ENV_COMPAT, token))
    plan.compat = token or None
    cmd = env.get(ENV_CMD_DIR)
    if cmd not in (None, '') and not PATH.fullmatch(cmd):
        problems.append('%s=%r is not a plain absolute path' % (ENV_CMD_DIR, cmd))
    plan.cmd_dir = cmd or None
    spill = env.get(ENV_SPILL) or DEFAULT_SPILL
    if spill not in SPILL_MODES:
        problems.append('%s=%r must be one of %s' % (ENV_SPILL, spill, ', '.join(SPILL_MODES)))
    plan.spill = spill
    plan.through_blocks = _digits(env, ENV_THROUGH_BLOCKS, DEFAULT_THROUGH_BLOCKS, 1, MAX_THROUGH_BLOCKS, problems)
    plan.through_idle_s = _digits(env, ENV_THROUGH_IDLE_S, DEFAULT_THROUGH_IDLE_S, 0, 86400, problems)
    if spill != 'through':
        knobs = [name for name in (ENV_THROUGH_BLOCKS, ENV_THROUGH_IDLE_S) if env.get(name) not in (None, '')]
        if knobs:
            problems.append('%s set with %s=%s: the evict mode would ignore it (write-through is the default; drop the knob or the override)' % (', '.join(knobs), ENV_SPILL, spill))
    plan.drain_s = _digits(env, ENV_DRAIN_S, DEFAULT_DRAIN_S, 0, MAX_DRAIN_S, problems)
    # At-rest encryption (owner, 2026-10-21: Opal self-encryption on the cache drive): the tier refuses to open a drive that is not a self-encrypting drive with locking enabled and
    # unlocked, unless a gate turns the check off. On by default; 0 is a gate's own value (the contract refuses it in a traffic profile).
    plan.require_sed = _switch(env, ENV_REQUIRE_SED, problems, True)
    status = env.get(ENV_SED_STATUS)
    if status not in (None, '') and not PATH.fullmatch(status):
        problems.append('%s=%r is not a plain absolute path' % (ENV_SED_STATUS, status))
    plan.sed_status = status or DEFAULT_SED_STATUS
    # The retention acknowledgement: the serving process cannot know what its node agent advertises, so the profile says, in so many words, that this node keeps prompt-derived
    # data on a drive for days and is not a zero-retention node.
    if env.get(ENV_RETENTION_ACK) != '1':
        problems.append('%s=1 is required with %s=1: the tier keeps prompt-derived KV on a drive for up to its TTL (default 1 day; all tenants; owner decisions 2026-10-21), which a '
                        'node that advertises zero-retention must not do, and a serving process cannot tell what its node advertises' % (ENV_RETENTION_ACK, ENV_SWITCH))
    plan.allow_zero_retention = _switch(env, ENV_ALLOW_ZERO_RETENTION, problems)
    # The peer-to-peer restore (docs/tp4-p2p-restore.md): off unless named. On, the drive holds the blocks chip-major (a new layout in the fingerprint) and the restores go NVMe -> card.
    plan.p2p = _switch(env, ENV_P2P, problems)
    plan.p2p_threads = _digits(env, ENV_P2P_THREADS, DEFAULT_P2P_THREADS, 1, 64, problems)
    plan.p2p_staging_bytes = _digits(env, ENV_P2P_STAGING_MIB, DEFAULT_P2P_STAGING_MIB, 2, 4096, problems) << 20
    plan.p2pmem = ()
    listed = env.get(ENV_P2PMEM)
    if listed not in (None, ''):
        names = [name.strip() for name in listed.split(',')]
        if not all(PCI_ADDRESS.fullmatch(name) for name in names):
            problems.append('%s=%r must be a comma-separated list of PCI addresses (0000:bb:dd.f)' % (ENV_P2PMEM, listed))
        else:
            plan.p2pmem = tuple(sorted(name.lower() if name.count(':') == 2 else '0000:' + name.lower() for name in names))
    if not plan.p2p:
        knobs = [name for name in (ENV_P2P_THREADS, ENV_P2P_STAGING_MIB, ENV_P2PMEM) if env.get(name) not in (None, '')]
        if knobs:
            problems.append('%s set without %s=1: the peer-to-peer restore would ignore it' % (', '.join(knobs), ENV_P2P))
    return (None if problems else plan), problems


def flag_problems(env):
    """Every way a profile's env strings set the NVMe tier wrongly ([] when none or when it names none of them). The same parser the engine refuses to start on."""
    plan, problems = parse({key: str(value) for key, value in (env or {}).items()})
    return problems


def nvme_plan(environ, tier_config):
    """The plan from the process environment, None when QWEN_PREFIX_NVME is unset (and so is every other name). ValueError on any problem, and when the host KV tier is
    not configured (the NVMe tier sits below it)."""
    plan, problems = parse(dict(environ))
    if problems:
        raise ValueError('; '.join(problems))
    if plan is not None and tier_config is None:
        raise ValueError('%s=1 needs the host KV tier (QWEN_PREFIX_HOST_TIER_GIB > 0): the NVMe tier is its cold level' % ENV_SWITCH)
    return plan


def zero_retention_signals(environ, exists=os.path.exists):
    """Why this process looks like it runs on a node that advertises zero-retention ([] when nothing says so). BEST EFFORT, and the module docstring says why: the node agent
    advertises the label and signs the teardown attestations itself, and tells its serving containers nothing today. What a serving process can see: the agent's attestation key
    in its own environment (a conflict by itself: the key is the agent's secret), THATCH_ZERO_RETENTION=1 (the name the platform is asked to forward into the serving
    containers of an attesting node), THATCH_NO_LOG=1 (the container runs under the zero-retention log control), and a marker file in the kill-switch directory."""
    reasons = []
    if environ.get('THATCH_ATTESTATION_PRIVATE_KEY'):
        reasons.append('THATCH_ATTESTATION_PRIVATE_KEY is in this process (the node signs zero-retention attestations)')
    if str(environ.get('THATCH_ZERO_RETENTION', '')).strip().lower() in ('1', 'true', 'yes', 'on'):
        reasons.append('THATCH_ZERO_RETENTION=1 (the node advertises zero-retention)')
    if environ.get('THATCH_NO_LOG') == '1':
        reasons.append('THATCH_NO_LOG=1 (the container runs under the zero-retention log control)')
    try:
        if exists(ZERO_RETENTION_MARKER):
            reasons.append('%s exists (a zero-retention marker on the node)' % ZERO_RETENTION_MARKER)
    except Exception:
        pass
    return reasons


def refused_metrics(kind):
    """The registry snapshot's NVMe keys while the tier is configured but not running: every counter 0, the gauges 0, and the reason's own counter 1."""
    values = dict.fromkeys(COUNTER_NAMES, 0)
    values.update(dict.fromkeys(GAUGE_NAMES, 0))
    values[{'zero_retention': 'nvme_refused_zero_retention', 'not_encrypted': 'nvme_refused_not_encrypted'}.get(kind, 'nvme_unavailable')] = 1
    return values


# ------------------------------------------------------------------------------------------------
# At-rest encryption: the drive must be a self-encrypting drive (TCG Opal) with locking enabled, unlocked
# ------------------------------------------------------------------------------------------------
class NotEncrypted(store_module.StoreError):
    """QWEN_PREFIX_NVME_REQUIRE_SED=1 and the drive is not, or cannot be shown to be, an Opal self-encrypting drive with locking enabled and currently unlocked."""


OPAL_FEATURES = {0x0200: 'opal', 0x0203: 'opal'}      # Opal SSC 1.0 and 2.x feature descriptors; Pyrite (0x0302, 0x0303) does not encrypt the media and is not Opal
LOCKING_FEATURE = 0x0002
DISCOVERY_HEADER = 48


def parse_level0(data):
    """What a TCG Level 0 Discovery response says that matters here: dict(sed='opal' or None, locking_supported, locking_enabled, locked, media_encryption). The response is a
    48-byte header (its first four bytes, big-endian, count the bytes after them) and feature descriptors (code 2 bytes, version 1, length 1, data); the Locking feature's first
    data byte holds the flags (bit 0 supported, 1 enabled, 2 locked, 3 media encryption). ValueError on a response that does not parse."""
    data = bytes(data)
    if len(data) < DISCOVERY_HEADER:
        raise ValueError('a discovery response of %d bytes is shorter than its header' % len(data))
    end = min(struct.unpack('>I', data[0:4])[0] + 4, len(data))
    sed, flags, position = None, 0, DISCOVERY_HEADER
    while position + 4 <= end:
        code, _version, length = struct.unpack('>HBB', data[position:position + 4])
        body = data[position + 4:position + 4 + length]
        if len(body) < length:
            raise ValueError('a discovery feature (0x%04x) is cut short' % code)
        if code in OPAL_FEATURES:
            sed = OPAL_FEATURES[code]
        elif code == LOCKING_FEATURE and body:
            flags = body[0]
        position += 4 + length
    return dict(sed=sed, locking_supported=bool(flags & 1), locking_enabled=bool(flags & 2), locked=bool(flags & 4), media_encryption=bool(flags & 8))


def opal_discovery(device, ioctl=None):
    """IOC_OPAL_DISCOVERY on the device: the drive's Level 0 Discovery response as bytes (a Security Receive: it changes nothing). OSError from the open or the ioctl as it comes: a
    container without CAP_SYS_ADMIN gets EACCES (the kernel's sed-opal ioctls check the capability), a kernel or drive without Opal ENOTTY or EOPNOTSUPP."""
    import fcntl

    fd = os.open(device, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
    try:
        buffer = ctypes.create_string_buffer(OPAL_DISCOVERY_BYTES)
        argument = struct.pack('QQ', ctypes.addressof(buffer), OPAL_DISCOVERY_BYTES)
        (ioctl or fcntl.ioctl)(fd, IOC_OPAL_DISCOVERY, argument)
        return bytes(buffer.raw)
    finally:
        os.close(fd)


def _read_file(path):
    with open(path) as handle:
        return handle.read()


def boot_time(read=None):
    """When this host booted, in epoch seconds (/proc/stat btime, else now minus /proc/uptime). OSError when neither can be read."""
    read = read or _read_file
    try:
        for line in read('/proc/stat').splitlines():
            if line.startswith('btime '):
                return float(line.split()[1])
    except (OSError, ValueError):
        pass
    return time.time() - float(read('/proc/uptime').split()[0])


def parse_timestamp(value):
    """A status file's `ts`: epoch seconds (a number, or digits in a string) or an ISO-8601 time (a Z or an offset; no zone means UTC). ValueError otherwise."""
    if isinstance(value, bool):
        raise ValueError('ts is a boolean')
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            import datetime

            moment = datetime.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=datetime.timezone.utc)
            return moment.timestamp()
    raise ValueError('ts is not a time')


def same_drive(status, device, sysfs=store_module.SYSFS, by_id_dir='/dev/disk/by-id'):
    """(True, '') when the status file is about the drive at `device`, else (False, why). Two independent looks, each match / mismatch / unknown, at least one match and no mismatch:
    `by_id` (the device is that /dev/disk/by-id name, or resolves to the same node as its link) and `pci` (the device's sysfs path, found by device number, runs through that PCI
    function). A file that names neither, or names only what this host cannot check, is not trusted."""
    verdicts = []
    by_id = status.get('by_id')
    if isinstance(by_id, str) and by_id:
        name = os.path.basename(by_id)
        link = os.path.join(by_id_dir, name)
        if os.path.basename(device) == name:
            verdicts.append(True)
        elif os.path.dirname(device) == by_id_dir:
            verdicts.append(False)                                  # the device IS a by-id name, and another one
        elif os.path.exists(link) and os.path.exists(device):
            verdicts.append(os.path.realpath(link) == os.path.realpath(device))
    pci = status.get('pci')
    if isinstance(pci, str) and pci:
        try:
            info = os.stat(device)
            real = os.path.realpath(os.path.join(sysfs, 'dev', 'block', '%d:%d' % (os.major(info.st_rdev), os.minor(info.st_rdev))))
            if os.path.isdir(real):
                verdicts.append(('/%s/' % pci) in (real + '/'))
        except OSError:
            pass
    if False in verdicts:
        return False, 'it describes another drive'
    if True in verdicts:
        return True, ''
    return False, 'nothing in it can be checked against this device'


def sed_check(device, status_path=DEFAULT_SED_STATUS, discover=None, read=None, boot=None, sysfs=store_module.SYSFS, by_id_dir='/dev/disk/by-id'):
    """Whether the drive is a self-encrypting drive (Opal) with locking enabled and currently unlocked. -> (source, facts) with source 'ioctl' or 'status file'; NotEncrypted(why) when
    it is not, or cannot be shown to be. First an IOC_OPAL_DISCOVERY on the device (read-only). A serving container normally lacks CAP_SYS_ADMIN, which the kernel's sed-opal ioctls
    require, so EACCES or EPERM (from the open or the ioctl) sends us to the status file the host's boot unit writes: JSON with sed="opal", locking_enabled, locked, by_id, pci,
    link_gts and ts. A missing, unreadable or malformed file, one older than this boot (the drive may have been reset since), one about another drive, or a drive that answers
    nothing, all count as NOT encrypted."""
    try:
        facts = parse_level0((discover or opal_discovery)(device))
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM):
            raise NotEncrypted('the drive does not answer an Opal discovery (%s)' % (error.strerror or error))
        return 'status file', status_file_facts(device, status_path, read, boot, sysfs, by_id_dir)
    except ValueError as error:
        raise NotEncrypted('the drive\'s Opal discovery cannot be read (%s)' % error)
    if facts['sed'] != 'opal':
        raise NotEncrypted('the drive is not an Opal self-encrypting drive (no Opal feature in its discovery)')
    if not (facts['locking_supported'] and facts['locking_enabled']):
        raise NotEncrypted('the drive\'s locking is not enabled (supported=%d enabled=%d)' % (facts['locking_supported'], facts['locking_enabled']))
    if not facts['media_encryption']:
        raise NotEncrypted('the drive does not report media encryption')
    if facts['locked']:
        raise NotEncrypted('the drive is locked')
    return 'ioctl', facts


def status_file_facts(device, status_path, read=None, boot=None, sysfs=store_module.SYSFS, by_id_dir='/dev/disk/by-id'):
    """The boot unit's status file, validated (see sed_check). -> its facts; NotEncrypted(why)."""
    try:
        text = (read or _read_file)(status_path)
    except OSError as error:
        raise NotEncrypted('no Opal discovery is allowed here and the status file %s cannot be read (%s)' % (status_path, error.strerror or error))
    try:
        status = json.loads(text)
        if not isinstance(status, dict):
            raise ValueError('not an object')
    except ValueError as error:
        raise NotEncrypted('the status file %s is not a JSON object (%s)' % (status_path, error))
    if status.get('sed') != 'opal':
        raise NotEncrypted('the status file does not say the drive is an Opal self-encrypting drive (sed=%r)' % (status.get('sed'),))
    if status.get('locking_enabled') is not True:
        raise NotEncrypted('the status file does not say locking is enabled')
    if status.get('locked') is not False:
        raise NotEncrypted('the status file says the drive is locked, or does not say it is not')
    try:
        stamp = parse_timestamp(status.get('ts'))
        booted = (boot or boot_time)()
    except (ValueError, OSError, TypeError) as error:
        raise NotEncrypted('the status file\'s time (ts) or this boot\'s time cannot be read (%s)' % error)
    if stamp < booted:
        raise NotEncrypted('the status file is older than this boot (the drive may have been reset or re-locked since)')
    ok, why = same_drive(status, device, sysfs, by_id_dir)
    if not ok:
        raise NotEncrypted('the status file does not belong to this drive: %s' % why)
    return dict(sed='opal', locking_supported=True, locking_enabled=True, locked=False, media_encryption=True, link_gts=status.get('link_gts'))


def spare_slots(plan, block_bytes):
    """Slab slots beyond the host tier's cap that hold evicted records until the writer has them (the spill queue's whole size in blocks)."""
    return max(1, int(plan.queue_bytes // max(1, int(block_bytes))))


# ------------------------------------------------------------------------------------------------
# The layout fingerprint
# ------------------------------------------------------------------------------------------------
def drafter_identity(environ):
    """What names the drafter checkpoint a stored checkpoint's window rows were taken for: 'id@revision' from the pinned table (drafter_checkpoint, imported by name so this
    module's closure does not grow), the table's default when QWEN_FAST_DRAFTER_CHECKPOINT is unset; the bare id, or 'default', when the table cannot be read."""
    name = str(environ.get(DRAFTER_FLAG) or '').strip()
    try:
        table = importlib.import_module(DRAFTER_MODULE).load()
        chosen = name or table['default']
        return '%s@%s' % (chosen, str(table['checkpoints'][chosen]['revision'])[:12])
    except Exception:
        return name or 'default'


def layout_fingerprint(io, plan, environ):
    """(8-byte fingerprint, components): everything that makes a stored block mean something else. An entry whose fingerprint differs is never restored.

    NOT in it, on purpose: the image build (its source revision) and the serving profile's name. A deploy that changes neither the KV_EPOCH nor one of the components below
    keeps the tier warm; KV_EPOCH is what a kernel author bumps when a change alters prefill numerics or the byte meaning of what is stored (see its comment).
    QWEN_PREFIX_NVME_COMPAT is the explicit override: any token other than the empty one is its own fingerprint, so a gate (or an operator) can force a fresh tier."""
    repo, revision = environ.get('QWEN_C2_MODEL_REPO') or '', environ.get('QWEN_C2_MODEL_REVISION') or ''
    components = collections.OrderedDict([
        ('layout', LAYOUT_CHIPMAJOR if getattr(plan, 'p2p', False) else LAYOUT),
        ('epoch', KV_EPOCH),
        ('model', '%s@%s' % (repo, revision) if repo or revision else 'base'),
        ('drafter', drafter_identity(environ)),
        ('kv', str(getattr(io, 'fingerprint', ''))),
        ('block_bytes', int(getattr(io, 'block_bytes', 0) or 0)),
        ('page', BLOCK),
        ('chunk', CHUNK),
        ('mesh', 'tp%s' % (getattr(io, 'chips', None) or '?')),
        ('gdn_bf16', environ.get('QWEN35_GDN_STATE_BF16', '0')),
        ('ckpt_format', CKPT_FORMAT),
        ('env', ['%s=%s' % (name, environ.get(name, '')) for name in FINGERPRINTED_ENV]),
        ('compat', plan.compat or ''),
    ])
    text = json.dumps(components, sort_keys=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).digest()[:8], components


# ------------------------------------------------------------------------------------------------
# The checkpoint codec
# ------------------------------------------------------------------------------------------------
class BytesOps(object):
    """Raw bytes-like items (bytes, bytearray, memoryview, array.array): the fakes of the CPU tests, and anything already flat."""

    kind = 'b'

    def describe(self, item):
        if isinstance(item, (bytes, bytearray, memoryview, array.array)):
            view = memoryview(item)
            return ('b', 'bytes', [view.nbytes])
        return None

    def to_buffer(self, item):
        view = memoryview(item)
        return view if view.format == 'B' else view.cast('B')

    def from_buffer(self, kind, dtype, shape, buffer):
        return bytes(buffer)


class TorchOps(object):
    """torch tensors (the unconverted checkpoints, the drafter window rows): bytes through numpy, dtype and shape in the header. torch and numpy are imported by the
    first tensor only."""

    kind = 't'

    def __init__(self, torch=None):
        self._torch = torch

    @property
    def torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def describe(self, item):
        if hasattr(item, 'dtype') and hasattr(item, 'shape') and hasattr(item, 'detach'):
            return ('t', str(item.dtype), [int(dim) for dim in item.shape])
        return None

    def to_buffer(self, item):
        torch = self.torch
        flat = item.detach().contiguous().reshape(-1).view(torch.uint8)
        return memoryview(flat.numpy())

    def from_buffer(self, kind, dtype, shape, buffer):
        torch = self.torch
        name = dtype.split('.')[-1]
        flat = torch.frombuffer(bytearray(buffer), dtype=getattr(torch, name))
        return flat.reshape(shape)


class ConvertedOps(TorchOps):
    """The model's preconverted checkpoint tensors (_QwenPrefixConverted: a ttnn host tensor in the device's tile layout and dtype). Packed as the torch tensor the
    capture converted (ttnn.to_torch, the mesh concatenated on dim 0), unpacked as that tensor and converted again by the model's own _qwen_prefix_preconvert, the call
    the capture made on the same values - so the host tensor that comes back is the one that was stored."""

    kind = 'c'

    def __init__(self, model):
        TorchOps.__init__(self)
        self.model = model
        # ttnn and torch are globals of the model file; the globals of the model's own preconvert method are that file's, whether or not it is registered as a module
        method = getattr(model, '_qwen_prefix_preconvert', None)
        namespace = getattr(sys.modules.get(type(model).__module__), '__dict__', {})
        if namespace.get('ttnn') is None:
            namespace = getattr(getattr(method, '__func__', method), '__globals__', None) or {}
        self.ttnn = namespace.get('ttnn')
        self._torch = namespace.get('torch')

    def describe(self, item):
        if hasattr(item, 'host') and hasattr(item, 'padded_nbytes') and hasattr(item, 'shape') and hasattr(item, 'dtype'):
            return ('c', str(item.dtype), [int(dim) for dim in item.shape])
        return None

    def to_buffer(self, item):
        ttnn = self.ttnn
        composer = ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=0)
        return TorchOps.to_buffer(self, ttnn.to_torch(item.host, mesh_composer=composer))

    def convert(self, rec, carry):
        """torch tensors -> the model's preconverted form: (rec, carry, nbytes)."""
        return self.model._qwen_prefix_preconvert(rec, carry)


class CheckpointImage(object):
    """A checkpoint decoded: what PrefixRegistry.adopt takes."""

    __slots__ = ('pos', 'token_ids', 'rec', 'carry', 'nbytes', 'window')


class CheckpointCodec(object):
    """GDN checkpoints (and the drafter window one may carry) to bytes and back. Layout: magic, header length, a JSON header (version, pos, the registry's byte charge,
    one descriptor per tensor: kind, dtype, shape, byte count; the window's range), the token ids (int64 x pos), then the tensors' bytes, each padded to 8. A checkpoint
    is whatever its `rec` and `carry` are: one item, or a list of items (one per GDN layer); an item is anything one of the ops describes."""

    MAGIC = b'QCKP0001'

    def __init__(self, ops):
        self.ops = list(ops)

    def _ops_for(self, item):
        for ops in self.ops:
            described = ops.describe(item)
            if described is not None:
                return ops, described
        raise TypeError('no tensor adapter for a checkpoint item of type %s' % type(item).__name__)

    def _describe_group(self, group, buffers, descriptors):
        single = not isinstance(group, (list, tuple))
        items = [group] if single else list(group)
        for item in items:
            ops, (kind, dtype, shape) = self._ops_for(item)
            buffer = ops.to_buffer(item)
            nbytes = len(buffer)
            descriptors.append({'k': kind, 'dt': dtype, 'sh': shape, 'n': nbytes})
            buffers.append(buffer)
            pad = (-nbytes) % 8
            if pad:
                buffers.append(bytes(pad))
        return single

    def pack(self, checkpoint):
        """-> the list of buffers whose concatenation is the object; raises TypeError for an item no adapter describes."""
        body, rec_desc, carry_desc, win_desc = [], [], [], []
        rec_single = self._describe_group(checkpoint.rec, body, rec_desc)
        carry_single = self._describe_group(checkpoint.carry, body, carry_desc)
        window = getattr(checkpoint, 'window', None)
        window_meta = None
        if window is not None:
            parts, described = [], []
            try:
                self._describe_group(list(window.payload), parts, described)
            except TypeError:
                pass                    # a window the adapters cannot hold is left off: the next turn resumes at the window ceiling, as without the carry
            else:
                body.extend(parts)
                win_desc = described
                window_meta = {'start': int(window.start), 'end': int(window.end), 'nbytes': int(window.nbytes), 'digest': window.digest}
        tokens = array.array('q', checkpoint.token_ids)
        header = {'v': CKPT_FORMAT, 'pos': int(checkpoint.pos), 'nbytes': int(checkpoint.nbytes), 'rec': rec_desc, 'rec_single': rec_single, 'carry': carry_desc,
                  'carry_single': carry_single, 'window': window_meta, 'window_items': win_desc}
        text = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
        head = self.MAGIC + struct.pack('<I', len(text)) + text
        head += bytes((-len(head)) % 8)
        return [head, memoryview(tokens).cast('B')] + body

    def unpack(self, data):
        """bytes-like -> CheckpointImage. ValueError on anything that is not a checkpoint of this format."""
        view = memoryview(data)
        if view.format != 'B':
            view = view.cast('B')
        if len(view) < 12 or bytes(view[:8]) != self.MAGIC:
            raise ValueError('not a checkpoint object')
        (length,) = struct.unpack_from('<I', view, 8)
        header = json.loads(bytes(view[12:12 + length]).decode())
        if header.get('v') != CKPT_FORMAT:
            raise ValueError('checkpoint format %r' % (header.get('v'),))
        position = 12 + length
        position += (-position) % 8
        pos = int(header['pos'])
        tokens = array.array('q')
        tokens.frombytes(bytes(view[position:position + pos * 8]))
        if len(tokens) != pos:
            raise ValueError('the token ids are cut short')
        position += pos * 8
        by_kind = dict((ops.kind, ops) for ops in self.ops)
        converted = []

        def group(descriptors, single):
            nonlocal position
            items = []
            for desc in descriptors:
                ops = by_kind.get(desc['k'])
                if ops is None:
                    raise ValueError('no adapter for item kind %r' % (desc['k'],))
                chunk = view[position:position + desc['n']]
                if len(chunk) != desc['n']:
                    raise ValueError('a tensor is cut short')
                items.append(ops.from_buffer(desc['k'], desc['dt'], desc['sh'], chunk))
                if desc['k'] == 'c':
                    converted.append(ops)
                position += desc['n'] + (-desc['n']) % 8
            return items[0] if single else items

        rec = group(header['rec'], header['rec_single'])
        carry = group(header['carry'], header['carry_single'])
        nbytes = int(header['nbytes'])
        meta = header.get('window')
        if converted:
            rec, carry, nbytes = converted[0].convert(rec, carry)
            nbytes += int(meta['nbytes']) if meta is not None else 0          # (the registry charges a carried window to its checkpoint)
        image = CheckpointImage()
        image.pos, image.token_ids, image.rec, image.carry, image.nbytes = pos, tokens, rec, carry, nbytes
        image.window = None
        if meta is not None:
            payload = group(header['window_items'], False)
            image.window = dict(start=meta['start'], end=meta['end'], payload=payload, nbytes=meta['nbytes'], digest=meta['digest'])
        return image


def default_codec(io):
    """The codec for a tier IO: its own (io.checkpoint_codec) when it brings one, else bytes + torch, and the model's preconverted items when the IO knows its model."""
    inner = getattr(io, 'io', None)            # (a wrapper such as tp4/upload-1's ThreadedTierIO holds the model's IO as .io)
    own = getattr(io, 'checkpoint_codec', None) or getattr(inner, 'checkpoint_codec', None)
    if own is not None:
        return own
    ops = [BytesOps(), TorchOps()]
    model = getattr(io, 'model', None) or getattr(inner, 'model', None)
    if model is not None and hasattr(model, '_qwen_prefix_preconvert'):
        ops.insert(1, ConvertedOps(model))
    return CheckpointCodec(ops)


# ------------------------------------------------------------------------------------------------
# The restore backends
# ------------------------------------------------------------------------------------------------
class RestoreBackend(object):
    """How NVMe-resident blocks get back to the device. TierHooks holds one and asks `direct`:
      direct False  read_to_host(tier, entries, views): the entries' bytes into the host buffers (slab slots) the caller reserved; the existing upload code then writes
                    those records to the device. NVMe -> host RAM -> card.
      direct True   restore_to_device(tier, entries, block_ids, io): the entries' bytes straight into the device blocks `block_ids` (already allocated by the pool, in
                    order), with no host copy. NVMe -> card."""

    name = 'abstract'
    direct = False

    def read_to_host(self, tier, entries, views):
        raise NotImplementedError

    def restore_to_device(self, tier, entries, block_ids, io):
        raise NotImplementedError

    def reading(self):
        """Reads this backend has in flight outside the tier's own count (the spill writer yields to them)."""
        return 0

    def close(self):
        """Give back whatever the backend holds (the tier's shutdown)."""


class HostBounceBackend(RestoreBackend):
    """NVMe -> pinned host buffer -> the existing upload. The device reads happen on `threads` workers, runs of back-to-back blocks as one call; the digest checks (the
    tier's verify mode) run on the same workers right after each run lands, so a 254k session's 8.5 GB is read and hashed at the drive's rate."""

    name = 'host'
    direct = False

    def read_to_host(self, tier, entries, views):
        return tier.read_into(entries, views)


class P2PBackend(HostBounceBackend):
    """NVMe -> card with no host copy (QWEN_PREFIX_NVME_P2P=1; docs/tp4-p2p-restore.md). `direct` while the path is usable; the host path (read_to_host, inherited) is always there, and is
    what any failure falls back to, so a restore is never worse than a miss:
      * the engine (qwen_p2p_restore.P2PEngine) reads each card's share of every block from the drive into the card's staging buffer (O_DIRECT into the published p2pmem), checks it on
        the card against the checksums recorded at spill, and only then asks the device to copy it into the KV blocks the pool allocated;
      * a checksum that differs is a MISS: nothing is written for that chunk, the turn prefills cold, the blocks are counted (nvme_p2p_mismatch), the entries are re-read through the
        host path to tell a bad drive (sha256 fails: the entry is dropped) from a bad path (sha256 holds: a strike; three latch the path off);
      * any other failure (a read error, a device op that raises, a window problem) restores the same blocks through the host path and counts a strike; three latch the path off at
        run time (`direct` turns False and TierHooks goes back to the host records), a device-op error at once;
      * if the path cannot start (no staging op, no published window, a failed canary) the tier serves from the host backend and says so once (nvme_p2p_unavailable);
      * ENCRYPTION AT REST works only at the drive (Opal): the bytes go from the drive to the card in the clear."""

    name = 'p2p'
    sliced = True                  # TierHooks may run a sliced restore through start_stream (the reads run beside the other seats' decode rounds)

    def __init__(self, engine, tier):
        self.engine = engine
        self.tier = tier
        self.latched = None
        self.strikes = 0

    @property
    def direct(self):
        return self.engine is not None and self.latched is None

    def reading(self):
        """Shard reads in flight (the spill writer yields to them, as it does to a host read)."""
        return self.engine.inflight if self.engine is not None else 0

    def latch(self, why):
        if self.latched is None:
            self.latched = str(why)
            self.tier.c['nvme_p2p_latched'] = 1
            self.tier.log('nvme tier p2p latched off: %s: the restores go through the host path until a restart', why)

    def strike(self, why, at_once=False):
        self.strikes += 1
        if at_once or self.strikes >= P2P_STRIKES:
            self.latch(why)

    def start_stream(self, items, max_chunk_blocks=None):
        """A restore's reads, begun (TierHooks' sliced restore): `items` is one pinned entry, or None for a block the host tier holds, per position."""
        return self.engine.start(items, max_chunk_blocks)

    def restore_to_device(self, tier, entries, block_ids, io):
        """TierHooks.load_direct's call: entries pinned, block_ids allocated by the pool. Returns when the blocks hold the bytes (the host fallback included); raises when the restore is a
        miss (a checksum mismatch, or the host path failing too)."""
        import qwen_p2p_restore as p2p

        c = tier.c
        began = time.perf_counter()
        try:
            self.engine.restore(entries, block_ids)
        except p2p.P2PMismatch as error:
            self.note_mismatch(tier, error)
            raise
        except Exception as error:
            self.note_error(tier, error)
            tier._restore_via_host(entries, block_ids)
            c['nvme_p2p_fallbacks'] += 1
            c['nvme_p2p_restores'] += 1
            return
        self.strikes = 0
        c['nvme_p2p_restores'] += 1
        tier.log('nvme tier p2p restore blocks=%d bytes=%d ms=%.1f', len(entries), sum(entry.nbytes for entry in entries), (time.perf_counter() - began) * 1000.0)

    def note_error(self, tier, error):
        """A peer-to-peer restore failed for a reason other than a checksum: counted, a strike (a device-op error latches the path at once)."""
        import qwen_p2p_restore as p2p

        tier.c['nvme_p2p_errors'] += 1
        self.strike('%s: %s' % (type(error).__name__, error), at_once=isinstance(error, p2p.P2PDeviceError))
        tier.log('nvme tier p2p restore failed (%s: %s): the blocks go through the host path', type(error).__name__, error)

    def note_mismatch(self, tier, error):
        tier.c['nvme_p2p_mismatch'] += len(error.bad)
        self._mismatch(tier, error)

    def _mismatch(self, tier, error):
        """The cards' checksums disagreed with the stored ones. Re-read a few of the blocks through the host path: bytes whose sha256 fails are bad on the drive (the entry is dropped);
        bytes that verify mean the path delivered them wrong (a strike against the path; the entries stay)."""
        seen, bad_on_drive, good = [], [], 0
        for entry in error.entries:
            if entry in seen:
                continue
            seen.append(entry)
            if len(seen) > P2P_MISMATCH_CHECKS:
                break
            try:
                data = tier.store.read_bytes(entry)
                ok = hashlib.sha256(data).digest() == entry.digest
            except Exception:
                ok = False
            if ok:
                good += 1
            else:
                bad_on_drive.append(entry)
        if bad_on_drive:
            tier.c['nvme_digest_failures'] += len(bad_on_drive)
            tier._forget(bad_on_drive)
        tier.log('nvme tier p2p CHECKSUM MISMATCH on %d shard(s) of %d block(s): %d re-read through the host path verify, %d fail their sha256 and are dropped; the turn prefills cold',
                 len(error.bad), len(error.entries), good, len(bad_on_drive))
        if good:
            self.strike('the cards\' checksums disagree with the stored ones for bytes the host path proves good')

    def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()


def default_p2p_engine(tier):
    """The engine of a real mesh, built from the tier's IO: the staging ops of the device-op extension, the cards' windows, the reader on the tier's own device file. P2PUnavailable
    (or whatever went wrong) when the path cannot start; the caller serves from the host backend."""
    import importlib

    import qwen_p2p_restore as p2p

    io = tier.io
    inner = getattr(io, 'io', None)
    mesh = getattr(io, 'mesh_device', None) or getattr(getattr(io, 'model', None), 'mesh_device', None) or getattr(inner, 'mesh_device', None) or getattr(getattr(inner, 'model', None), 'mesh_device', None)
    caches = getattr(io, 'caches', None) or getattr(inner, 'caches', None)
    if mesh is None or not caches:
        raise p2p.P2PUnavailable('the tier IO exposes no mesh device or cache tensors')
    try:
        ttnn = importlib.import_module('ttnn')
    except ImportError as error:
        raise p2p.P2PUnavailable('ttnn is not importable: %s' % error)
    api = p2p.StagingApi(ttnn, mesh, caches)
    return p2p.P2PEngine(api, tier.shards, tier.p2p_reader, threads=tier.plan.p2p_threads, staging_bytes=tier.plan.p2p_staging_bytes, expected=tier.plan.p2pmem, logger=tier.log,
                         io_align=tier.store.bf.align)


def make_backend(name):
    if name == 'host':
        return HostBounceBackend()
    if name == 'p2p':
        raise ValueError('the peer-to-peer backend is built by the tier (NvmeTier._start_p2p) when QWEN_PREFIX_NVME_P2P=1: it needs the tier\'s IO and device')
    raise ValueError('restore backend %r (host)' % (name,))


# ------------------------------------------------------------------------------------------------
# The tier
# ------------------------------------------------------------------------------------------------
class SpillKV(object):
    """A host-tier record on its way to the NVMe: the key and identity, the payload (a view of the retained slab slot, or the record's own array) and the slot to give back."""

    __slots__ = ('key', 'payload', 'nbytes', 'tenant', 'chain', 'index', 'slot', 'held')

    def __init__(self, key, payload, tenant, chain, index, slot, held=False):
        self.key, self.payload, self.tenant, self.chain, self.index, self.slot = key, payload, tenant, chain, index, slot
        self.held = held             # a write-through record: still a live host-tier record, held there until it is written (TierStore.held)
        self.nbytes = len(payload)


class SpillCkpt(object):
    __slots__ = ('checkpoint', 'nbytes')

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.nbytes = max(1, int(checkpoint.nbytes))


class NvmeTier(object):
    """See the module docstring. Threads: the scheduler's (offer, offer_checkpoint, lookups, reads, adopt) and the one writer's (everything durable that spilling does),
    plus whoever calls delete_salt / wipe (they serialise on the store). Counters are split by thread so none is updated from two: self.c (the scheduler's and callers')
    and self.w (the writer's)."""

    def __init__(self, plan, io, registry, logger=log, clock=time.time, environ=None, store=None, backend=None, file_factory=None, probe=None, p2p_engine=None):
        self.plan = plan
        self.io = io
        self.registry = registry
        self.log = logger
        self.clock = clock
        environ = os.environ if environ is None else environ
        self.fp, self.components = layout_fingerprint(io, plan, environ)
        self.block_bytes = int(getattr(io, 'block_bytes', 0) or 0)
        if self.block_bytes and plan.queue_bytes < self.block_bytes:
            raise ValueError('%s=%d holds less than one KV block (%d bytes)' % (ENV_QUEUE, plan.queue_bytes, self.block_bytes))
        self.codec = default_codec(io)
        # QWEN_PREFIX_NVME_P2P=1: the drive holds each KV block chip-major, which needs the IO's geometry (cache tensors, chips, bytes per slice).
        self.shards = None
        if plan.p2p:
            import qwen_p2p_restore

            self.shards = qwen_p2p_restore.ShardLayout.of_io(io)
            if self.shards is None:
                raise store_module.StoreError('%s=1 needs the tier IO\'s geometry (caches, chips, slice_bytes) to lay the blocks out chip-major' % ENV_P2P)
            if self.shards.block_bytes != self.block_bytes:
                raise store_module.StoreError('the tier IO\'s geometry makes a block of %d bytes, its block_bytes says %d' % (self.shards.block_bytes, self.block_bytes))
        self.backend = backend if backend is not None else make_backend(plan.restore)
        self.c = dict.fromkeys(COUNTER_NAMES, 0)
        self.w = dict.fromkeys(COUNTER_NAMES, 0)
        self.latched = None
        self.closed = False
        self.at_rest = 'unchecked'               # 'opal' once the drive was shown to be a self-encrypting drive with locking enabled and unlocked (QWEN_PREFIX_NVME_REQUIRE_SED)
        self.stopping = False                    # a graceful stop began: no new offers (set by begin_stop, from a signal handler too: assignments only)
        self.stop_deadline = None                # clock() time the drain budget ends (set by begin_stop)
        self.last_health = clock()
        self._probe = probe                      # None: the store's own device_problem; tests inject a callable -> str or None
        self._hooks_armed = False
        self._pid = os.getpid()                  # a forked child must never write the drive its parent holds
        self.failures = 0
        self.read_failures = 0
        self.checked = 0
        self.reading = 0
        self.queue = collections.deque()
        self.queue_kv_bytes = 0
        self.queue_ckpt_bytes = 0
        # chain -> {position: queued count} of the checkpoints offered and not yet written: the blocks that follow them into the queue are not 'dead' just because the writer
        # has not caught up (under self.cond)
        self.queued_ckpt = {}
        self.queued_keys = set()                 # block hashes in the queue or being written (under self.cond)
        self.backlog_blocks = 0                  # finished requests' blocks waiting for the idle copy (TierHooks.copy_idle sets it)
        self.cond = threading.Condition()
        self.commands = collections.deque()
        self.last_sweep = clock()
        self.last_cmd = 0.0
        self.cmd_seen = {}
        self.min_tokens = int(getattr(getattr(registry, 'tier_config', None), 'min_tokens', 0) or 0)
        self._hash_pool = None
        if store is None:
            # (a drive that a reset or rescan took away can leave its node behind with a size of 0: say so, rather than let the format policy or a short read speak)
            gone = store_module.device_problem(plan.device) if probe is None else probe()
            if gone:
                raise store_module.DeviceGone(gone)
            if plan.require_sed:
                # (before anything touches the drive: a plaintext drive is never opened, whatever it holds)
                source, facts = sed_check(plan.device, plan.sed_status)
                self.at_rest = 'opal'
                self.log('nvme tier at-rest encryption verified: opal locking_enabled=1 locked=0 media_encryption=1 source=%s', source.replace(' ', '-'))
            options = dict(ttl_s=plan.ttl_s, max_age_s=plan.max_age_s, max_bytes=plan.max_bytes, discard=plan.discard, threads=plan.threads)
            kwargs = dict(format=plan.format, direct=plan.direct, snap_bytes=plan.snap_bytes, journal_bytes=plan.journal_bytes, clock=clock)
            if file_factory is not None:
                kwargs['file_factory'] = file_factory
            store = NvmeStore.open(plan.device, **dict(kwargs, **options))
        self.store = store
        if plan.p2p and backend is None:
            self._start_p2p(p2p_engine)
        if plan.free_floor_bytes is None:
            self.store.free_floor_bytes = int(self.store.geometry.data_bytes * DEFAULT_FREE_FLOOR_FRACTION)
        else:
            self.store.free_floor_bytes = int(plan.free_floor_bytes)
        for name in ('recovered_entries', 'recovered_records', 'recover_dropped'):
            self.w['nvme_' + name] = self.store.stats[name]
        dropped = self.store.drop_where(lambda entry: entry.fp != self.fp)
        self.w['nvme_fp_drops'] += dropped
        stats = self.store.snapshot_stats()
        self.log('nvme tier open device=%s backend=%s ttl_s=%d max_age_s=%d max_bytes=%d capacity_bytes=%d free_floor_bytes=%d entries=%d kv=%d ckpt=%d '
                 'recovered_records=%d recover_dropped=%d fp_drops=%d verify=%s queue_bytes=%d threads=%d audit=%d spill=%s retention_ack=1 at_rest=%s fingerprint=%s',
                 plan.device, self.backend.name, plan.ttl_s, plan.max_age_s, plan.max_bytes, self.store.capacity_bytes(), self.store.free_floor_bytes, stats['entries'],
                 stats['kv_entries'], stats['ckpt_entries'], self.store.stats['recovered_records'], self.store.stats['recover_dropped'], dropped, plan.verify,
                 plan.queue_bytes, plan.threads, int(plan.audit), plan.spill, self.at_rest, self.fp.hex())
        self.log('nvme tier fingerprint %s components=%s', self.fp.hex(), json.dumps(self.components, separators=(',', ':')))
        if dropped:
            self.log('nvme tier dropped %d entries of another layout fingerprint (never restored)', dropped)
        self.commands.append(('sweep', None))                  # what expired while the engine was down goes at once
        self.writer = threading.Thread(target=self._writer_loop, name='qwen-nvme-writer', daemon=True)
        self.writer.start()

    # -- state ---------------------------------------------------------------------------------------
    @property
    def active(self):
        return self.latched is None and not self.closed and not self.stopping and self.registry.tier_on

    def latch(self, reason, lost=False):
        """Stop spilling and restoring for the life of the process (the data stays on the drive). ONE log line, however many callers find the same trouble. `lost`: the cause is
        the device itself having gone away (counted as nvme_device_lost too)."""
        if self.latched is None:
            self.latched = ('device lost: %s' % reason) if lost else str(reason)
            self.c['nvme_latched'] = 1
            if lost:
                self.c['nvme_device_lost'] = 1
                self.log('nvme tier DEVICE LOST: %s: latched off, the host tier serves without it until a restart finds the device live', reason)
            else:
                self.log('nvme tier latched off: %s', reason)
        self._drop_queue()

    def _device_down(self, error=None):
        """Why the drive is gone (a sentence), None when nothing says so. From an error: DeviceGone, or an errno that only a vanished device returns. Otherwise the device's
        node and sysfs entries (device_problem: no IO, so a dead controller cannot hang this)."""
        if isinstance(error, store_module.DeviceGone):
            return str(error)
        if isinstance(error, OSError) and error.errno in GONE_ERRNOS:
            return '%s (errno %s)' % (error.strerror or type(error).__name__, error.errno)
        try:
            if self._probe is not None:
                return self._probe()
            return store_module.device_problem(self.plan.device, self.store.geometry.device_bytes)
        except Exception:
            return None

    def _io_trouble(self, error=None, strikes=None, what='IO'):
        """A device call failed. A device that is gone latches the tier at once; otherwise `strikes` (a count of failures in a row) latches it at three."""
        gone = self._device_down(error)
        if gone:
            self.latch(gone, lost=True)
        elif strikes is not None and strikes >= 3:
            self.latch('three NVMe %s in a row failed' % what)

    # -- the spill sink and the checkpoint offer (scheduler thread) ---------------------------------------
    def offer(self, record):
        """TierStore's spill sink: `record` was just evicted from the host tier. True when this tier now owns the record's slab slot (the writer gives it back)."""
        if not self.active:
            return False
        c = self.c
        c['nvme_spill_offered'] += 1
        key = record.key
        if self.store.lookup(KV, key) is not None or key in self.queued_keys:
            self.store.touch(KV, key)
            c['nvme_spill_known'] += 1
            self._offer_checkpoint_of(record)
            return False
        if self.registry._tier_dead(record):
            c['nvme_spill_dropped_useless'] += 1
            return False
        if self.queue_kv_bytes + self.queue_ckpt_bytes + record.nbytes > self.plan.queue_bytes:
            c['nvme_spill_dropped_queue'] += 1
            return False
        item = SpillKV(key, record.payload, record.tenant, record.chain, record.index, record.slot)
        with self.cond:
            self.queue.append(item)
            self.queued_keys.add(key)
            self.queue_kv_bytes += item.nbytes
            self.cond.notify()
        self._offer_checkpoint_of(record)
        return record.slot is not None

    @property
    def through(self):
        return self.plan.through

    def offer_through(self, record):
        """TierStore's commit sink under QWEN_PREFIX_NVME_SPILL=through: `record` just entered the host tier. Queue its bytes for the drive now, holding the record in the tier
        (never a victim) until the writer has them. Nothing here waits for the drive."""
        if not self.plan.through or not self.active:
            return
        c = self.c
        key = record.key
        if self.store.lookup(KV, key) is not None or key in self.queued_keys:
            c['nvme_through_known'] += 1
            return
        if self.registry._tier_dead(record):
            return
        if self.queue_kv_bytes + self.queue_ckpt_bytes + record.nbytes > self.plan.queue_bytes:
            c['nvme_through_dropped_queue'] += 1
            return
        tier = self.registry.tier
        with tier._held_lock:
            tier.held.add(key)
        item = SpillKV(key, record.payload, record.tenant, record.chain, record.index, None, held=True)
        with self.cond:
            self.queue.append(item)
            self.queued_keys.add(key)
            self.queue_kv_bytes += item.nbytes
            self.cond.notify()
        c['nvme_through_queued'] += 1
        self._offer_checkpoint_of(record)

    def _offer_checkpoint_of(self, record):
        """The checkpoint at a KV record's key leaves the store with it: persist it so the session can come back from the NVMe."""
        checkpoint = self.registry.entries.get(record.key)
        if checkpoint is not None:
            self.offer_checkpoint(checkpoint)

    def offer_checkpoint(self, checkpoint):
        """Persist a checkpoint that is leaving the registry's store (its byte budget, or the eviction of its boundary block). False when it is not worth it, already held,
        or the queue is full. The packing runs on the writer thread."""
        if not self.active or checkpoint.pos < self.min_tokens:
            return False
        c = self.c
        if self.store.lookup(CKPT, checkpoint.key) is not None:
            self.store.touch(CKPT, checkpoint.key)
            return False
        if checkpoint.chain is not None and checkpoint.pos in self.queued_ckpt.get(bytes(checkpoint.chain), ()):
            return False                                  # already on its way
        item = SpillCkpt(checkpoint)
        if self.queue_kv_bytes + self.queue_ckpt_bytes + item.nbytes > self.plan.queue_bytes:
            c['nvme_ckpt_spill_dropped'] += 1
            return False
        with self.cond:
            self.queue.append(item)
            self.queue_ckpt_bytes += item.nbytes
            if checkpoint.chain is not None:
                positions = self.queued_ckpt.setdefault(bytes(checkpoint.chain), {})
                positions[checkpoint.pos] = positions.get(checkpoint.pos, 0) + 1
            self.cond.notify()
        return True

    # -- lookups (scheduler thread; no device IO) -------------------------------------------------------------
    def has_kv(self, key):
        return self.active and self.store.lookup(KV, key) is not None

    def chain_reaches(self, chain, need_tokens):
        """Whether a checkpoint of the chain at or above need_tokens is on the NVMe or on its way there."""
        if not self.active:
            return False
        chain = bytes(chain or b'')
        if self.store.chain_reaches(chain, need_tokens):
            return True
        with self.cond:
            return any(pos >= need_tokens for pos in self.queued_ckpt.get(chain, ()))

    def checkpoint_entry(self, key):
        if not self.active:
            return None
        return self.store.lookup(CKPT, key)

    def pin_kv(self, keys):
        """The entries of these block hashes, pinned (None where a block is gone or expired). Give them back with unpin."""
        return self.store.pin([(KV, key) for key in keys])

    def unpin(self, entries):
        self.store.unpin(entries)

    def touch_kv(self, keys):
        """These blocks were just used: their TTL starts over and they become the youngest in the LRU order."""
        for key in keys:
            self.store.touch(KV, key)

    # -- reads (scheduler thread; this is the restore's NVMe time) ---------------------------------------------
    def read_into(self, entries, views):
        """Read pinned KV entries into `views` and check their digests per the verify mode. -> a list parallel to `entries`: None ok, 'io', 'digest'. Failures are counted
        and a block that failed its digest is deleted; three device errors in a row latch the tier."""
        mode = self.plan.verify
        if self.plan.audit and mode != 'off':
            mode = 'all'
        chosen = set()
        if mode == 'all':
            chosen = set(id(entry) for entry in entries)
        elif mode == 'sample':
            chosen = set(id(entry) for index, entry in enumerate(entries) if index % VERIFY_EVERY == 0)

        def verify(entry, view):
            return hashlib.sha256(view).digest() == entry.digest if id(entry) in chosen else True

        self.reading += 1
        began = time.perf_counter()
        try:
            results = self._read_many(entries, views, verify if chosen else None)
        finally:
            self.reading -= 1
        c = self.c
        c['nvme_digest_checks'] += len(chosen)
        bad = [entry for entry, result in zip(entries, results) if result == 'digest']
        c['nvme_digest_failures'] += len(bad)
        io_errors = sum(1 for result in results if result == 'io')
        if io_errors:
            self.read_failures += 1
            self._io_trouble(None, self.read_failures, 'reads')
        elif all(result is None for result in results):
            self.read_failures = 0
        if bad:
            self._forget(bad)
        c['nvme_restore_ms'] += (time.perf_counter() - began) * 1000.0
        return results

    def _read_many(self, entries, views, verify):
        """store.read_many into the host tier's views. On a chip-major drive (QWEN_PREFIX_NVME_P2P=1) the objects are read, digest-checked as stored, into aligned scratch a few blocks at a
        time and converted back to the host tier's [cache][chip][slice] layout."""
        if self.shards is None:
            return self.store.read_many(entries, views, verify=verify)
        import mmap

        results = []
        group = P2P_READ_GROUP
        for low in range(0, len(entries), group):
            part = entries[low:low + group]
            arena = mmap.mmap(-1, len(part) * self.block_bytes)
            scratch = [memoryview(arena)[index * self.block_bytes:(index + 1) * self.block_bytes] for index in range(len(part))]
            got = self.store.read_many(part, scratch, verify=verify)
            for source, target, result in zip(scratch, views[low:low + group], got):
                if result is None:
                    self.shards.from_chipmajor(source, target)
            results.extend(got)
            for source in scratch:
                source.release()
            del scratch
            try:
                arena.close()
            except BufferError:
                pass
        return results

    def _restore_via_host(self, entries, block_ids, group=P2P_READ_GROUP):
        """The host path to the device blocks, for the peer-to-peer backend's fallback: read (and verify, per the verify mode) a few blocks at a time and write them with the model\'s
        own tier IO. Raises when a block cannot be brought back; the caller's restore is then a miss."""
        for low in range(0, len(entries), group):
            part = entries[low:low + group]
            payloads = [bytearray(self.block_bytes) for _ in part]
            results = self.read_into(part, payloads)
            if any(result is not None for result in results):
                self.c['nvme_restore_failures'] += 1
                raise store_module.StoreError('%d of %d blocks could not be read back through the host path (%s)' % (sum(1 for item in results if item is not None), len(part), sorted(set(item for item in results if item))))
            self.io.write_blocks(block_ids[low:low + group], payloads)

    def p2p_reader(self, offset, view):
        """The peer-to-peer engine's read: fill `view` (a window onto a card's p2pmem) from the drive at `offset`, through the store's own device file (alignment checks, test hooks)."""
        self.store.bf.pread_into(offset, [view])

    def _start_p2p(self, factory=None):
        """Bring the peer-to-peer path up (QWEN_PREFIX_NVME_P2P=1). It is optional: when it cannot start (no staging op, no published window, a failed canary) the tier says so once, counts
        it and serves from the host backend, which reads the same chip-major objects."""
        try:
            engine = (factory or default_p2p_engine)(self)
        except Exception as error:
            self.c['nvme_p2p_unavailable'] = 1
            self.log('nvme tier p2p UNAVAILABLE, the host backend serves the restores (the drive keeps its chip-major layout): %s: %s', type(error).__name__, error)
            return
        self.backend = P2PBackend(engine, self)
        info = engine.describe()
        self.log('nvme tier p2p on cards=%d threads=%d staging_bytes=%d chunk_blocks=%d shard_bytes=%d mapping_verified=%d window=%s', info['cards'], info['threads'], info['staging_bytes'],
                 info['chunk_blocks'], info['shard_bytes'], info['mapping_verified'], info['window'])

    def _forget(self, entries):
        """Entries that failed their digest: out of the index at once, their journal record and extents on the writer thread."""
        self.store.mark_bad(entries)
        with self.cond:
            self.commands.append(('dead', None))
            self.cond.notify()

    def load_checkpoint(self, entry):
        """Read and decode a CKPT entry (pinned for the read; digest always checked unless verify=off). -> CheckpointImage or None (counted)."""
        c = self.c
        began = time.perf_counter()
        pinned = self.store.pin([(CKPT, entry.key)])[0]
        if pinned is None:
            c['nvme_ckpt_restore_failures'] += 1
            return None
        self.reading += 1
        try:
            data = self.store.read_bytes(pinned)
            if self.plan.verify != 'off':
                c['nvme_digest_checks'] += 1
                if hashlib.sha256(data).digest() != pinned.digest:
                    c['nvme_digest_failures'] += 1
                    c['nvme_ckpt_restore_failures'] += 1
                    self._forget([pinned])
                    self.log('nvme tier checkpoint failed its digest pos=%d: dropped', pinned.index)
                    return None
            image = self.codec.unpack(data)
            if image.pos != pinned.index:
                raise ValueError('checkpoint object holds position %d, its entry says %d' % (image.pos, pinned.index))
        except Exception as error:
            c['nvme_ckpt_restore_failures'] += 1
            self.log('nvme tier checkpoint restore failed pos=%d: %s: %s', pinned.index, type(error).__name__, error)
            if isinstance(error, (OSError, IOError)):
                self.read_failures += 1
                self._io_trouble(error, self.read_failures, 'reads')
            return None
        finally:
            self.reading -= 1
            self.store.unpin([pinned])
        self.read_failures = 0
        elapsed = (time.perf_counter() - began) * 1000.0
        c['nvme_ckpt_restored'] += 1
        c['nvme_ckpt_restore_bytes'] += pinned.nbytes
        c['nvme_ckpt_restore_ms'] += elapsed
        self.store.touch(CKPT, pinned.key)
        self.log('nvme tier checkpoint restored pos=%d bytes=%d ms=%.1f', image.pos, pinned.nbytes, elapsed)
        return image

    def restore_to_device(self, entries, block_ids):
        """The direct backend's call (TierHooks.load_direct)."""
        return self.backend.restore_to_device(self, entries, block_ids, self.io)

    def note_restore(self, blocks, size, first, need, request_id):
        c = self.c
        c['nvme_restore_requests'] += 1
        c['nvme_restore_blocks'] += blocks
        c['nvme_restore_bytes'] += size
        self.log('nvme tier restore req=%s blocks=%d bytes=%d from=%d to=%d backend=%s verify=%s', request_id, blocks, size, first * BLOCK, need * BLOCK,
                 self.backend.name, self.plan.verify)

    # -- deletion and administration ---------------------------------------------------------------------
    def delete_salt(self, salt):
        """Remove every entry of the tenant a cache_salt belongs to (the tenant tag the registry keys by). Synchronous: call it from an admin thread. -> entries."""
        from_registry = getattr(sys.modules.get('qwen_prefix_registry'), 'tenant_of_salt', None)
        tag = from_registry(salt) if from_registry is not None else salt
        return self.delete_tenant(tag)

    def _durable(self, what, call):
        """Run a deletion on the store. A device that is gone cannot take it: the tier latches (one line), the deletion is reported NOT APPLIED, and 0 comes back - the entries
        stay on the drive, so the operator repeats the deletion once a restart finds the device live. Any other failure is the caller's."""
        try:
            return call()
        except (OSError, IOError, store_module.StoreError) as error:
            gone = self._device_down(error)
            if gone is None:
                raise
            self.latch(gone, lost=True)
            self.log('nvme tier %s NOT APPLIED: the device is gone (%s); the entries stay on the drive until it is live and the deletion is repeated', what, gone)
            return 0

    def delete_tenant(self, tag):
        removed = self._durable('tenant delete', lambda: self.store.delete_tenant(tag))
        self.c['nvme_deleted_tenant'] += removed
        self.log('nvme tier tenant delete entries=%d', removed)
        return removed

    def wipe(self, why='requested'):
        """Remove everything, durably and at once (a reset_prefix_cache, the kill switch). The queued spills are dropped. -> entries."""
        self._drop_queue()
        removed = self._durable('wipe', self.store.wipe)
        self.c['nvme_wipes'] += 1
        self.log('nvme tier wiped entries=%d why=%s', removed, why)
        return removed

    def _drop_queue(self):
        with self.cond:
            items = list(self.queue)
            self.queue.clear()
            self.queue_kv_bytes = 0
            self.queue_ckpt_bytes = 0
        for item in items:
            self._release(item)

    def poll(self):
        """Once per scheduler step: the command directory and the periodic sweep (both run on the writer thread, never here)."""
        if self.closed:
            return
        now = self.clock()
        if self.latched is None and now - self.last_health >= HEALTH_EVERY_S:
            self.last_health = now
            gone = self._device_down()
            if gone:
                self.latch(gone, lost=True)
        if self.plan.cmd_dir and now - self.last_cmd >= CMD_EVERY_S:
            self.last_cmd = now
            self._read_commands()
        if now - self.last_sweep >= SWEEP_EVERY_S:
            self.last_sweep = now
            with self.cond:
                self.commands.append(('sweep', None))
                self.cond.notify()

    def _read_commands(self):
        """Files in QWEN_PREFIX_NVME_CMD_DIR: `wipe`, `tenant.<tag>`. Each is acted on once (by name and mtime), then removed when the directory lets us."""
        directory = self.plan.cmd_dir
        try:
            names = os.listdir(directory)
        except OSError:
            return
        for name in names:
            if name != 'wipe' and not name.startswith('tenant.'):
                continue
            path = os.path.join(directory, name)
            try:
                stamp = os.stat(path).st_mtime
            except OSError:
                continue
            if self.cmd_seen.get(name) == stamp:
                continue
            self.cmd_seen[name] = stamp
            with self.cond:
                self.commands.append(('wipe', None) if name == 'wipe' else ('tenant', name[len('tenant.'):]))
                self.cond.notify()
            try:
                os.remove(path)
            except OSError:
                pass

    # -- the writer thread -----------------------------------------------------------------------------
    def _release(self, item):
        if isinstance(item, SpillKV):
            if item.slot is not None:
                self.registry.tier.release_slot(item.slot)
                item.slot = None
            elif item.held:
                self.registry.tier.release_held(item.key)
                item.held = False
            with self.cond:
                self.queued_keys.discard(item.key)
        elif isinstance(item, SpillCkpt):
            self._unqueue_checkpoint(item)

    def _unqueue_checkpoint(self, item):
        checkpoint = item.checkpoint
        if checkpoint.chain is None:
            return
        with self.cond:
            positions = self.queued_ckpt.get(bytes(checkpoint.chain))
            if positions is None or checkpoint.pos not in positions:
                return
            positions[checkpoint.pos] -= 1
            if positions[checkpoint.pos] <= 0:
                del positions[checkpoint.pos]
            if not positions:
                del self.queued_ckpt[bytes(checkpoint.chain)]

    def _writer_loop(self):
        while True:
            try:
                if self._writer_round():
                    return
            except Exception as error:
                # whatever went wrong, the thread must not die quietly with the queue full of retained slots: the tier latches (which gives the slots back) and carries on
                self.log('nvme tier writer error: %s: %s', type(error).__name__, error)
                self.latch('the NVMe writer thread failed: %s: %s' % (type(error).__name__, error))
                time.sleep(0.1)

    def _writer_round(self):
        """One pass of the writer: True when the thread should end."""
        with self.cond:
            while not self.queue and not self.commands and not self.closed:
                self.cond.wait(timeout=1.0)
            if self.closed and not self.queue and not self.commands:
                return True
            commands = list(self.commands)
            self.commands.clear()
            batch, size = [], 0
            limit = MAX_BATCH_BYTES if self.shards is None else P2P_BATCH_BYTES
            while self.queue and len(batch) < MAX_BATCH_ITEMS and size < limit:
                item = self.queue.popleft()
                batch.append(item)
                size += item.nbytes
        for command, argument in commands:
            self._run_command(command, argument)
        if batch:
            self._wait_for_readers()
            try:
                self._write_batch(batch)
            finally:
                # the slots go back BEFORE the bytes leave the queue's count: a new offer is admitted only when its retained slot is certain to exist
                for item in batch:
                    self._release(item)
                with self.cond:
                    for item in batch:
                        if isinstance(item, SpillKV):
                            self.queue_kv_bytes = max(0, self.queue_kv_bytes - item.nbytes)      # (a latch zeroed the counts while this batch was in flight)
                        else:
                            self.queue_ckpt_bytes = max(0, self.queue_ckpt_bytes - item.nbytes)
        return False

    def _wait_for_readers(self):
        """A restore is latency-critical and a spill is not: hold the next write while a read is in flight (bounded)."""
        waited = 0.0
        while (self.reading or self.backend.reading()) and waited < WRITER_YIELD_MAX_S and not self.closed:
            time.sleep(WRITER_YIELD_S)
            waited += WRITER_YIELD_S

    def _run_command(self, command, argument):
        try:
            if command == 'sweep':
                before = dict(self.store.stats)
                self.store.sweep()
                for source, name in (('expired_ttl', 'nvme_expired_ttl'), ('expired_age', 'nvme_expired_age'), ('evicted_lru', 'nvme_evicted_lru')):
                    self.w[name] = self.store.stats[source]
                moved = self.store.stats['expired_ttl'] - before['expired_ttl'] + self.store.stats['expired_age'] - before['expired_age']
                if moved:
                    self.log('nvme tier expired %d entries (ttl %d, age %d)', moved, self.store.stats['expired_ttl'] - before['expired_ttl'],
                             self.store.stats['expired_age'] - before['expired_age'])
            elif command == 'dead':
                self.store.flush_dead()
            elif command == 'wipe':
                self.wipe('command file')
            elif command == 'tenant':
                self.delete_tenant(argument)
        except Exception as error:
            self.log('nvme tier command %s failed: %s: %s', command, type(error).__name__, error)

    def _digests(self, items):
        """sha256 of each KV payload, on the hash threads (hashlib drops the GIL on big buffers)."""
        if len(items) < 2 or self.plan.threads < 2:
            return [hashlib.sha256(item.payload).digest() for item in items]
        if self._hash_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._hash_pool = ThreadPoolExecutor(max_workers=self.plan.threads, thread_name_prefix='qwen-nvme-hash')
        return list(self._hash_pool.map(lambda item: hashlib.sha256(item.payload).digest(), items))

    def _chipmajor(self, kvs):
        """The batch's blocks laid out chip-major in one page-aligned arena (so the device write goes out from it without a bounce): -> [(buffer, sha256 of it, the per-chip checksums
        packed)] parallel to `kvs`. The per-chip checksums (qwen_p2p_restore.p2p_checksum) are what a card compares its staged shard with at a peer-to-peer restore."""
        import mmap

        shards, block_bytes = self.shards, self.block_bytes
        arena = mmap.mmap(-1, len(kvs) * block_bytes)
        view = memoryview(arena)

        def convert(index):
            buffer = view[index * block_bytes:(index + 1) * block_bytes]
            shards.to_chipmajor(kvs[index].payload, buffer)
            return buffer, hashlib.sha256(buffer).digest(), shards.pack_checksums(shards.checksums(buffer))

        if len(kvs) < 2 or self.plan.threads < 2:
            return [convert(index) for index in range(len(kvs))]
        if self._hash_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._hash_pool = ThreadPoolExecutor(max_workers=self.plan.threads, thread_name_prefix='qwen-nvme-hash')
        return list(self._hash_pool.map(convert, range(len(kvs))))

    def _write_batch(self, batch):
        w = self.w
        began = time.perf_counter()
        kvs = sorted((item for item in batch if isinstance(item, SpillKV)), key=lambda item: (bytes(item.chain or b''), item.index))
        ckpts = [item for item in batch if isinstance(item, SpillCkpt)]
        puts, owners = [], []
        try:
            if self.shards is not None:
                stored = self._chipmajor(kvs) if kvs else []
                for item, (buffer, digest, aux) in zip(kvs, stored):
                    puts.append(PutItem(KV, item.key, [buffer], item.tenant, item.chain, item.index, self.fp, digest=digest, aux=aux))
                    owners.append(item)
            else:
                digests = self._digests(kvs) if kvs else []
                for item, digest in zip(kvs, digests):
                    puts.append(PutItem(KV, item.key, [item.payload], item.tenant, item.chain, item.index, self.fp, digest=digest))
                    owners.append(item)
            packed = []
            for item in ckpts:
                try:
                    buffers = self.codec.pack(item.checkpoint)
                except Exception as error:
                    w['nvme_ckpt_spill_failures'] += 1
                    self.log('nvme tier checkpoint not packed pos=%s: %s: %s', getattr(item.checkpoint, 'pos', '?'), type(error).__name__, error)
                    continue
                checkpoint = item.checkpoint
                puts.append(PutItem(CKPT, checkpoint.key, buffers, checkpoint.tenant or '', checkpoint.chain or b'', checkpoint.pos, self.fp,
                                    extra=checkpoint.nbytes))
                owners.append(item)
                packed.append(item)
            if not puts:
                return
            try:
                results = self.store.put_batch(puts)
            except (OSError, IOError, store_module.StoreError) as error:
                if self._device_down(error):
                    raise                                 # the drive is gone: a retry would only fail the same way
                # one retry: a transient device error must not cost a session the one chance its blocks had to be written
                w['nvme_spill_retries'] += 1
                self.log('nvme tier spill retry after %s: %s', type(error).__name__, error)
                time.sleep(0.05)
                results = self.store.put_batch(puts)
        except Exception as error:
            w['nvme_spill_failures'] += 1
            self.failures += 1
            self.log('nvme tier spill failed (%d so far): %s: %s', self.failures, type(error).__name__, error)
            self._io_trouble(error, self.failures, 'writes')
            return
        self.failures = 0
        stored = stored_bytes = full = 0
        ckpt_stored = ckpt_bytes = 0
        for owner, put, ok in zip(owners, puts, results):
            if not ok:
                full += 1
                continue
            if put.kind == KV:
                stored += 1
                stored_bytes += put.nbytes
            else:
                ckpt_stored += 1
                ckpt_bytes += put.nbytes
        elapsed = (time.perf_counter() - began) * 1000.0
        w['nvme_spill_blocks'] += stored
        w['nvme_spill_bytes'] += stored_bytes
        w['nvme_spill_ms'] += elapsed
        w['nvme_spill_batches'] += 1
        w['nvme_spill_dropped_full'] += full
        w['nvme_ckpt_spilled'] += ckpt_stored
        w['nvme_ckpt_spill_bytes'] += ckpt_bytes
        for source, name in (('evicted_lru', 'nvme_evicted_lru'), ('expired_ttl', 'nvme_expired_ttl'), ('discarded_bytes', 'nvme_discarded_bytes')):
            w[name] = self.store.stats[source]
        if self.plan.audit:
            self._audit(puts, results)
        if stored or ckpt_stored:
            self.log('nvme tier spill blocks=%d bytes=%d ckpts=%d ckpt_bytes=%d ms=%.1f queue=%d', stored, stored_bytes, ckpt_stored, ckpt_bytes, elapsed,
                     self.queue_kv_bytes + self.queue_ckpt_bytes)

    def _audit(self, puts, results):
        """QWEN_PREFIX_NVME_AUDIT=1 (a gate instrument): read every object just written back from the device and compare its digest (the write path, O_DIRECT, end to end)."""
        w = self.w
        for put, ok in zip(puts, results):
            if not ok:
                continue
            entry = self.store.lookup(put.kind, put.key)
            if entry is None:
                continue
            try:
                data = self.store.read_bytes(entry)
            except Exception:
                w['nvme_audit_mismatches'] += 1
                continue
            w['nvme_audit_reads'] += 1
            if hashlib.sha256(data).digest() != entry.digest:
                w['nvme_audit_mismatches'] += 1
                self.log('nvme tier AUDIT MISMATCH kind=%d bytes=%d', put.kind, put.nbytes)

    # -- reporting -------------------------------------------------------------------------------------
    def metrics(self):
        """Counters and gauges for the registry's snapshot (read from the exporter thread)."""
        values = dict(self.w)
        for name, value in self.c.items():
            values[name] += value
        stats = self.store.snapshot_stats()
        for source, name in (('get_hits', 'nvme_store_hits'), ('get_misses', 'nvme_store_misses'), ('snapshots', 'nvme_snapshots'),
                             ('journal_records', 'nvme_journal_records'), ('evicted_lru', 'nvme_evicted_lru'), ('expired_ttl', 'nvme_expired_ttl'),
                             ('expired_age', 'nvme_expired_age'), ('discarded_bytes', 'nvme_discarded_bytes'), ('fp_drops', 'nvme_fp_drops')):
            values[name] = stats[source]
        engine = getattr(self.backend, 'engine', None)
        if engine is not None:
            for source, name in (('read_ms', 'nvme_p2p_read_ms'), ('check_ms', 'nvme_p2p_check_ms'), ('write_ms', 'nvme_p2p_write_ms'), ('blocks', 'nvme_p2p_blocks'),
                                 ('bytes', 'nvme_p2p_bytes'), ('mismatch_retries', 'nvme_p2p_late_data')):
                values[name] += engine.stats[source]
        values.update(nvme_p2p_on=int(self.active and self.backend.direct), nvme_p2p_map_verified=int(bool(engine is not None and engine.describe()['mapping_verified'])))
        values.update(nvme_on=int(self.active), nvme_bytes=stats['logical_bytes'], nvme_entries=stats['entries'], nvme_kv_entries=stats['kv_entries'],
                      nvme_ckpt_entries=stats['ckpt_entries'], nvme_capacity_bytes=stats['capacity_bytes'], nvme_free_bytes=stats['free_bytes'],
                      nvme_queue_bytes=self.queue_kv_bytes + self.queue_ckpt_bytes, nvme_journal_used_bytes=stats['journal_used'],
                      nvme_largest_free_bytes=stats['largest_free'], nvme_through_backlog_blocks=self.backlog_blocks, nvme_at_rest=int(self.at_rest == 'opal'))
        return values

    def describe(self):
        """The fields of the scheduler graft's 'nvme tier on' line."""
        return dict(device=self.plan.device, backend=self.backend.name, ttl_s=self.plan.ttl_s, max_age_s=self.plan.max_age_s, verify=self.plan.verify,
                    audit=int(self.plan.audit), queue_bytes=self.plan.queue_bytes, capacity_bytes=self.store.capacity_bytes(), fingerprint=self.fp.hex(),
                    spill=self.plan.spill, through_blocks=self.plan.through_blocks if self.plan.through else 0,
                    through_idle_s=self.plan.through_idle_s if self.plan.through else 0, drain_s=self.plan.drain_s, at_rest=self.at_rest)

    def flush(self, timeout=30.0):
        """Wait for the queue to empty (tests, a graceful stop). -> True when it did."""
        end = time.time() + timeout
        while time.time() < end:
            with self.cond:
                idle = not self.queue and not self.commands
            if idle and self._writer_idle():
                return True
            time.sleep(0.002)
        return False

    def _writer_idle(self):
        with self.cond:
            return not self.queue and not self.commands and self.queue_kv_bytes == 0 and self.queue_ckpt_bytes == 0

    # -- the graceful stop ----------------------------------------------------------------------------------
    def begin_stop(self):
        """A graceful stop has begun: no new offers or restores from now on, and the drain budget starts. Assignments only - safe in a signal handler."""
        if not self.stopping:
            self.stop_deadline = time.monotonic() + (self.plan.drain_s or LEGACY_CLOSE_S)
            self.stopping = True

    def shutdown(self, budget_s=None):
        """Drain and close: stop taking offers, let the writer flush the queue for what is left of the budget (QWEN_PREFIX_NVME_DRAIN_S from begin_stop, or `budget_s` from now),
        drop whatever it could not reach, then journal the TTL clocks, snapshot the index when time remains for it, and close the device. Bounded: it returns by the budget (a
        writer stuck in a device call is abandoned, not waited for). Idempotent; a forked child does nothing. -> True when the queue was flushed whole."""
        if self.closed or os.getpid() != self._pid:
            return True
        self.begin_stop()
        if budget_s is not None:
            self.stop_deadline = time.monotonic() + max(0.0, float(budget_s))
        began = time.monotonic()
        remaining = max(0.0, self.stop_deadline - began)
        drained = True
        if self.latched is None:
            drained = self.flush(timeout=max(0.0, remaining - min(SNAPSHOT_BASE_S, remaining / 2.0)))
        self._drop_queue()                                    # what the flush did not reach is not written: the host tier always lost these
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.writer.join(timeout=max(0.2, self.stop_deadline - time.monotonic()))
        if self.writer.is_alive():
            self.log('nvme tier stop: the writer is stuck in a device call; the device is left as it is (the journal is consistent at every record)')
            return False
        snapshotted = False
        if self.latched is None:
            entries = self.store.snapshot_stats()['entries']
            if self.stop_deadline - time.monotonic() >= SNAPSHOT_BASE_S + SNAPSHOT_ENTRY_S * entries:
                try:
                    self.store.snapshot()
                    snapshotted = True
                except Exception as error:
                    self.log('nvme tier stop: snapshot failed: %s: %s', type(error).__name__, error)
        if self._hash_pool is not None:
            self._hash_pool.shutdown(wait=True)
            self._hash_pool = None
        try:
            self.backend.close()
        except Exception as error:
            self.log('nvme tier stop: the peer-to-peer engine did not close: %s: %s', type(error).__name__, error)
        try:
            self.store.close()
        except Exception as error:
            self.log('nvme tier stop: close failed: %s: %s', type(error).__name__, error)
        self.log('nvme tier stopped drained=%d snapshot=%d latched=%d ms=%.0f', int(drained), int(snapshotted), int(self.latched is not None),
                 (time.monotonic() - began) * 1000.0)
        return drained

    def close(self, budget_s=None):
        """Close the tier (a re-attach, a test, the exit hooks): the same bounded drain as a stop."""
        return self.shutdown(budget_s)

    def arm_stop_hooks(self, signal_module=signal):
        """Make a graceful stop of this process drain the tier (QWEN_PREFIX_NVME_DRAIN_S > 0; the registry calls this once the tier is open). Three hooks, all calling
        shutdown(): the SIGTERM handler, CHAINED so that the drain clock starts at the signal and the writer flushes while the engine shuts itself down (when the previous
        handler is the default one, nothing would run at all, so this one drains and re-raises the signal with the default disposition); a multiprocessing finalizer (an
        engine-core child leaves through os._exit, which skips atexit); and atexit. -> True when armed."""
        if self._hooks_armed or self.plan.drain_s <= 0:
            return False
        self._hooks_armed = True
        try:
            previous = signal_module.getsignal(signal_module.SIGTERM)
            if previous != signal_module.SIG_IGN:
                tier = self

                def handler(signum, frame):
                    if os.getpid() != tier._pid:
                        return previous(signum, frame) if callable(previous) else None
                    tier.begin_stop()
                    if callable(previous):
                        return previous(signum, frame)
                    tier.shutdown()
                    signal_module.signal(signum, signal_module.SIG_DFL)
                    os.kill(os.getpid(), signum)

                signal_module.signal(signal_module.SIGTERM, handler)
        except (ValueError, OSError, AttributeError):
            pass                                              # not the main thread: the finalizers below still run on an exit
        try:
            from multiprocessing import util as multiprocessing_util

            multiprocessing_util.Finalize(None, self.shutdown, exitpriority=100)
        except Exception:
            pass
        atexit.register(self.shutdown)
        return True
