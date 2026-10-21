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
  THROUGH  Opt-in write-through (QWEN_PREFIX_NVME_SPILL=through, default evict): a session's blocks are also copied toward the drive soon after it goes idle - a bounded slice of
           its still-cached device blocks per scheduler step is read into the host tier (TierHooks.copy_idle) and every record that enters the host tier is queued to the writer at
           once (offer_through) - so a warm copy exists before the eviction that would otherwise be the first chance to write it, and a restart loses only the last idle seconds.
  GUARDS   The tier keeps prompt-derived data on a drive for days. A node that advertises zero-retention signs teardown attestations that nothing survived; the tier therefore
           REFUSES to start (a loud line and a counter; the engine serves without it) when the process sees a zero-retention signal (zero_retention_signals) unless
           QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION=1, and it needs QWEN_PREFIX_NVME_RETENTION_ACK=1 always: the serving process cannot know what its node agent advertises.
  BACKENDS RestoreBackend: 'host' (built) and 'p2p' (a stub that raises NotBuilt, its contract below). TierHooks asks backend.direct: a direct backend restores NVMe-resident
           blocks straight to the freshly allocated device blocks instead of through the host (TierHooks.load_direct).

KEYS. The host tier's own: vLLM's chained block hash (it carries the tenant's cache salt, so tenants never meet), kind KV; the checkpoint's boundary block hash, kind CKPT.
Every entry also carries a LAYOUT FINGERPRINT (model repo + revision, profile, KV layout/dtype/page size, mesh, GDN dtype, code version): an entry of another fingerprint
is dropped at open and never restored. The tenant tag is stored only as an 8-byte hash.

Pure python; imports nothing outside the standard library (torch and ttnn only through the objects the model hands in).
"""

import array
import collections
import hashlib
import importlib
import json
import os
import re
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
ENV_CODE_VERSION = 'QWEN_PREFIX_NVME_CODE_VERSION'
ENV_CMD_DIR = 'QWEN_PREFIX_NVME_CMD_DIR'
ENV_AUDIT = 'QWEN_PREFIX_NVME_AUDIT'
ENV_SPILL = 'QWEN_PREFIX_NVME_SPILL'
ENV_THROUGH_BLOCKS = 'QWEN_PREFIX_NVME_THROUGH_BLOCKS'
ENV_THROUGH_IDLE_S = 'QWEN_PREFIX_NVME_THROUGH_IDLE_S'
ENV_RETENTION_ACK = 'QWEN_PREFIX_NVME_RETENTION_ACK'
ENV_ALLOW_ZERO_RETENTION = 'QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION'
NAMES = (ENV_SWITCH, ENV_DEVICE, ENV_TTL_S, ENV_MAX_AGE_S, ENV_MAX_BYTES, ENV_VERIFY, ENV_RESTORE, ENV_FREE_FLOOR, ENV_QUEUE, ENV_THREADS, ENV_FORMAT, ENV_DISCARD,
         ENV_CODE_VERSION, ENV_CMD_DIR, ENV_AUDIT, ENV_SPILL, ENV_THROUGH_BLOCKS, ENV_THROUGH_IDLE_S, ENV_RETENTION_ACK, ENV_ALLOW_ZERO_RETENTION)
# What only a gate sets (the contract refuses them on a traffic profile even under the owner's waiver).
GATE_ONLY = (ENV_AUDIT, ENV_FORMAT, ENV_CMD_DIR, ENV_ALLOW_ZERO_RETENTION)
VERIFY_MODES = ('sample', 'all', 'off')
SPILL_MODES = ('evict', 'through')
DEFAULT_THROUGH_BLOCKS = 32          # device blocks one scheduler step may copy toward the drive (about 13 ms of device reads at 0.4 ms a block)
MAX_THROUGH_BLOCKS = 4096
DEFAULT_THROUGH_IDLE_S = 60          # a finished request's blocks are copied once it has been idle this long (a follow-up turn within it reuses them on the device)
IDLE_BACKLOG = 1024                  # finished requests remembered for the copy
# What tells a process it runs on a node that advertises zero-retention (see zero_retention_signals); the node agent's own attestation key is the node agent's secret and is
# never meant to be in a serving container, so its presence here is a conflict by itself.
ZERO_RETENTION_ENV = ('THATCH_ATTESTATION_PRIVATE_KEY', 'THATCH_ZERO_RETENTION', 'THATCH_NO_LOG')
ZERO_RETENTION_MARKER = '/models/.qwen-c2/zero-retention'
VERIFY_EVERY = 16
RESTORE_BACKENDS = ('host', 'p2p')
DEFAULT_TTL_S = 7 * 24 * 3600
MAX_TTL_S = 366 * 24 * 3600
DEFAULT_QUEUE_BYTES = 2 << 30
MIN_QUEUE_BYTES = 16 << 20
DEFAULT_THREADS = 4
DEFAULT_FREE_FLOOR_FRACTION = 0.05
CODE_VERSION_FILE = '/opt/qwen-c2/source-revision'
LAYOUT = 'packed-v1'                 # the on-disk payload layout: the host tier's raw packed block ([cache tensor][chip][slice]); a new layout is a new fingerprint
CKPT_FORMAT = 1
SWEEP_EVERY_S = 60.0
CMD_EVERY_S = 2.0
MAX_BATCH_ITEMS = 256
MAX_BATCH_BYTES = 512 << 20
WRITER_YIELD_S = 0.005
WRITER_YIELD_MAX_S = 2.0
PATH = re.compile(r'(?!-)/[A-Za-z0-9_./:@+-]{1,200}')
TOKEN = re.compile(r'[A-Za-z0-9._-]{1,64}')
DIGITS = re.compile(r'[0-9]{1,18}')

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
    'nvme_through_failures', 'nvme_spill_retries', 'nvme_refused_zero_retention', 'nvme_unavailable')
GAUGE_NAMES = ('nvme_on', 'nvme_bytes', 'nvme_entries', 'nvme_kv_entries', 'nvme_ckpt_entries', 'nvme_capacity_bytes', 'nvme_free_bytes', 'nvme_queue_bytes',
               'nvme_journal_used_bytes', 'nvme_largest_free_bytes', 'nvme_through_backlog_blocks')


def log(message, *values):
    try:
        sys.stderr.write('[PINDIAG] prefix: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


class NotBuilt(NotImplementedError):
    """A restore backend that is designed (its contract is in its class docstring) and not built."""


# ------------------------------------------------------------------------------------------------
# The flags
# ------------------------------------------------------------------------------------------------
class NvmePlan(object):
    """The NVMe tier's settings (nvme_plan)."""

    __slots__ = ('device', 'ttl_s', 'max_age_s', 'max_bytes', 'verify', 'restore', 'free_floor_bytes', 'queue_bytes', 'threads', 'format', 'discard', 'code_version',
                 'cmd_dir', 'audit', 'direct', 'snap_bytes', 'journal_bytes', 'spill', 'through_blocks', 'through_idle_s', 'allow_zero_retention')

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


def _switch(env, name, problems):
    raw = env.get(name)
    if raw is None or raw == '':
        return False
    if raw not in ('0', '1'):
        problems.append('%s=%r must be 0 or 1' % (name, raw))
        return False
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
        problems.append('%s=p2p is not built: the peer-to-peer restore needs a device-side scatter op (docs/tp4-kv-nvme.md, "The peer-to-peer follow-up")' % ENV_RESTORE)
    plan.restore = restore
    floor = env.get(ENV_FREE_FLOOR)
    plan.free_floor_bytes = None if floor in (None, '') else _digits(env, ENV_FREE_FLOOR, 0, 0, 1 << 60, problems)
    plan.queue_bytes = _digits(env, ENV_QUEUE, DEFAULT_QUEUE_BYTES, MIN_QUEUE_BYTES, 1 << 40, problems)
    plan.threads = _digits(env, ENV_THREADS, DEFAULT_THREADS, 1, 64, problems)
    plan.format = 'force' if _switch(env, ENV_FORMAT, problems) else 'auto'
    plan.discard = _switch(env, ENV_DISCARD, problems)
    plan.audit = _switch(env, ENV_AUDIT, problems)
    version = env.get(ENV_CODE_VERSION)
    if version not in (None, '') and not TOKEN.fullmatch(version):
        problems.append('%s=%r is not a short token (letters, digits, . _ -)' % (ENV_CODE_VERSION, version))
    plan.code_version = version or None
    cmd = env.get(ENV_CMD_DIR)
    if cmd not in (None, '') and not PATH.fullmatch(cmd):
        problems.append('%s=%r is not a plain absolute path' % (ENV_CMD_DIR, cmd))
    plan.cmd_dir = cmd or None
    spill = env.get(ENV_SPILL) or 'evict'
    if spill not in SPILL_MODES:
        problems.append('%s=%r must be one of %s' % (ENV_SPILL, spill, ', '.join(SPILL_MODES)))
    plan.spill = spill
    plan.through_blocks = _digits(env, ENV_THROUGH_BLOCKS, DEFAULT_THROUGH_BLOCKS, 1, MAX_THROUGH_BLOCKS, problems)
    plan.through_idle_s = _digits(env, ENV_THROUGH_IDLE_S, DEFAULT_THROUGH_IDLE_S, 0, 86400, problems)
    if spill != 'through':
        knobs = [name for name in (ENV_THROUGH_BLOCKS, ENV_THROUGH_IDLE_S) if env.get(name) not in (None, '')]
        if knobs:
            problems.append('%s set without %s=through: the evict mode would ignore it' % (', '.join(knobs), ENV_SPILL))
    # The retention acknowledgement: the serving process cannot know what its node agent advertises, so the profile says, in so many words, that this node keeps prompt-derived
    # data on a drive for days and is not a zero-retention node.
    if env.get(ENV_RETENTION_ACK) != '1':
        problems.append('%s=1 is required with %s=1: the tier keeps prompt-derived KV on a drive for up to its TTL (default 7 days; all tenants; owner decision 2026-10-21), which a '
                        'node that advertises zero-retention must not do, and a serving process cannot tell what its node advertises' % (ENV_RETENTION_ACK, ENV_SWITCH))
    plan.allow_zero_retention = _switch(env, ENV_ALLOW_ZERO_RETENTION, problems)
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
    values['nvme_refused_zero_retention' if kind == 'zero_retention' else 'nvme_unavailable'] = 1
    return values


def spare_slots(plan, block_bytes):
    """Slab slots beyond the host tier's cap that hold evicted records until the writer has them (the spill queue's whole size in blocks)."""
    return max(1, int(plan.queue_bytes // max(1, int(block_bytes))))


# ------------------------------------------------------------------------------------------------
# The layout fingerprint
# ------------------------------------------------------------------------------------------------
def profile_name(environ):
    """The serving profile's name: QWEN_C2_PROFILE, else the contract's default (the engine's own resolution)."""
    name = environ.get('QWEN_C2_PROFILE')
    if name:
        return name
    try:
        # (by name, not by an import statement: the registry's image closure must not grow the contract's)
        contract = importlib.import_module('serving_c2_contract')
        return contract.load_profile(environ.get('QWEN_C2_PROFILES', contract.PROFILES))['name']
    except Exception as error:
        raise ValueError('the NVMe tier cannot tell which serving profile this is (QWEN_C2_PROFILE unset, the default unreadable: %s)' % error)


def code_version(plan, environ, reader=None):
    if plan.code_version:
        return plan.code_version
    path = CODE_VERSION_FILE
    try:
        with (reader or open)(path) as handle:
            text = handle.read().strip()
    except Exception as error:
        raise ValueError('the NVMe tier cannot tell which image this is: %s unreadable (%s); name it with %s' % (path, error, ENV_CODE_VERSION))
    if not TOKEN.fullmatch(text):
        raise ValueError('%s does not hold a short token: %r' % (path, text[:80]))
    return text


def layout_fingerprint(io, plan, environ, reader=None):
    """(8-byte fingerprint, components): everything that makes a stored block mean something else. An entry whose fingerprint differs is never restored."""
    repo, revision = environ.get('QWEN_C2_MODEL_REPO') or '', environ.get('QWEN_C2_MODEL_REVISION') or ''
    components = collections.OrderedDict([
        ('layout', LAYOUT),
        ('model', '%s@%s' % (repo, revision) if repo or revision else 'base'),
        ('profile', profile_name(environ)),
        ('kv', str(getattr(io, 'fingerprint', ''))),
        ('block_bytes', int(getattr(io, 'block_bytes', 0) or 0)),
        ('page', BLOCK),
        ('chunk', CHUNK),
        ('mesh', 'tp%s' % (getattr(io, 'chips', None) or '?')),
        ('gdn_bf16', environ.get('QWEN35_GDN_STATE_BF16', '0')),
        ('ckpt_format', CKPT_FORMAT),
        ('code', code_version(plan, environ, reader)),
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


class HostBounceBackend(RestoreBackend):
    """NVMe -> pinned host buffer -> the existing upload. The device reads happen on `threads` workers, runs of back-to-back blocks as one call; the digest checks (the
    tier's verify mode) run on the same workers right after each run lands, so a 254k session's 8.5 GB is read and hashed at the drive's rate."""

    name = 'host'
    direct = False

    def read_to_host(self, tier, entries, views):
        return tier.read_into(entries, views)


class P2PBackend(RestoreBackend):
    """NOT BUILT. The contract a peer-to-peer restore (NVMe -> card, no host copy) must meet; 'p2p' is refused at start until it does.

    Proven on 2026-10-21 (not in this repository): a read from the namespace into a card's BAR4 window through the patched kernel driver at 6.95 GB/s, byte-exact. What
    that leaves to build:
      * a STAGING WINDOW per card: at most 3 GiB (STAGING_WINDOW_MAX_BYTES), at a 1 GiB offset into the 4 GiB BAR (STAGING_WINDOW_OFFSET; the top of the window failed
        its self-test), so one restore moves at most floor(3 GiB / block_bytes) blocks (1,445 at 2,228,224 bytes) per window fill;
      * CHUNK ALIGNMENT: every entry is already a whole number of 4 KiB blocks at a 4 KiB-aligned namespace LBA (qwen_nvme_store), and block_bytes = 544 x 4,096, so a
        run of blocks is one aligned read; the window offset and each chunk must stay 4 KiB-aligned;
      * a DEVICE-SIDE SCATTER OP, e.g. ttnn.qwen_scatter_blocks_from_window(cache_tensors, window, block_ids, block_bytes, slice_bytes): for each block in the window, copy
        its [cache tensor][chip][slice] pieces to the page range of block_ids[i] in every cache tensor on every chip - the inverse of qwen_write_blocks_raw without the
        host buffer. A block's payload is laid out [cache][chip][slice_bytes = 17,408 = 4.25 pages], so a chip's pieces are not page-aligned inside the block: either the
        scatter distributes the whole block to all four chips over the fabric, or a later layout version ('chip-major, page-padded') changes the fingerprint and rewrites
        the cache;
      * INTEGRITY without the host: the bytes never visit host RAM, so the sha256 check cannot run on them. The options are a device-side checksum op over the window
        (compared with the entry's digest, or a CRC stored beside it), or the drive's end-to-end protection information; until one exists a P2P restore verifies nothing,
        and QWEN_PREFIX_NVME_VERIFY must say so;
      * ENCRYPTION AT REST works only at the drive (Opal self-encrypting namespace): host software AES cannot touch bytes that go from the drive to the card;
      * the same sliced hold TierHooks gives the host path (the restore may not freeze the other seats), and the same fall back to a miss on any failure."""

    name = 'p2p'
    direct = True
    STAGING_WINDOW_MAX_BYTES = 3 << 30
    STAGING_WINDOW_OFFSET = 1 << 30
    ALIGN = 4096

    def __init__(self, *args, **kwargs):
        raise NotBuilt('the peer-to-peer NVMe restore is not built (it needs a device-side scatter op); see the P2PBackend docstring')

    @classmethod
    def window_blocks(cls, block_bytes):
        """How many blocks one staging window holds."""
        return cls.STAGING_WINDOW_MAX_BYTES // int(block_bytes)


def make_backend(name):
    if name == 'host':
        return HostBounceBackend()
    if name == 'p2p':
        return P2PBackend()
    raise ValueError('restore backend %r (host or p2p)' % (name,))


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

    def __init__(self, plan, io, registry, logger=log, clock=time.time, environ=None, store=None, backend=None, reader=None, file_factory=None):
        self.plan = plan
        self.io = io
        self.registry = registry
        self.log = logger
        self.clock = clock
        environ = os.environ if environ is None else environ
        self.fp, self.components = layout_fingerprint(io, plan, environ, reader)
        self.block_bytes = int(getattr(io, 'block_bytes', 0) or 0)
        if self.block_bytes and plan.queue_bytes < self.block_bytes:
            raise ValueError('%s=%d holds less than one KV block (%d bytes)' % (ENV_QUEUE, plan.queue_bytes, self.block_bytes))
        self.codec = default_codec(io)
        self.backend = backend if backend is not None else make_backend(plan.restore)
        self.c = dict.fromkeys(COUNTER_NAMES, 0)
        self.w = dict.fromkeys(COUNTER_NAMES, 0)
        self.latched = None
        self.closed = False
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
            options = dict(ttl_s=plan.ttl_s, max_age_s=plan.max_age_s, max_bytes=plan.max_bytes, discard=plan.discard, threads=plan.threads)
            kwargs = dict(format=plan.format, direct=plan.direct, snap_bytes=plan.snap_bytes, journal_bytes=plan.journal_bytes, clock=clock)
            if file_factory is not None:
                kwargs['file_factory'] = file_factory
            store = NvmeStore.open(plan.device, **dict(kwargs, **options))
        self.store = store
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
                 'recovered_records=%d recover_dropped=%d fp_drops=%d verify=%s queue_bytes=%d threads=%d audit=%d spill=%s retention_ack=1 fingerprint=%s',
                 plan.device, self.backend.name, plan.ttl_s, plan.max_age_s, plan.max_bytes, self.store.capacity_bytes(), self.store.free_floor_bytes, stats['entries'],
                 stats['kv_entries'], stats['ckpt_entries'], self.store.stats['recovered_records'], self.store.stats['recover_dropped'], dropped, plan.verify,
                 plan.queue_bytes, plan.threads, int(plan.audit), plan.spill, self.fp.hex())
        if dropped:
            self.log('nvme tier dropped %d entries of another layout fingerprint (never restored)', dropped)
        self.commands.append(('sweep', None))                  # what expired while the engine was down goes at once
        self.writer = threading.Thread(target=self._writer_loop, name='qwen-nvme-writer', daemon=True)
        self.writer.start()

    # -- state ---------------------------------------------------------------------------------------
    @property
    def active(self):
        return self.latched is None and not self.closed and self.registry.tier_on

    def latch(self, reason):
        """Stop spilling and restoring for the life of the process (the data stays on the drive)."""
        if self.latched is None:
            self.latched = str(reason)
            self.c['nvme_latched'] = 1
            self.log('nvme tier latched off: %s', reason)
        self._drop_queue()

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
            results = self.store.read_many(entries, views, verify=verify if chosen else None)
        finally:
            self.reading -= 1
        c = self.c
        c['nvme_digest_checks'] += len(chosen)
        bad = [entry for entry, result in zip(entries, results) if result == 'digest']
        c['nvme_digest_failures'] += len(bad)
        io_errors = sum(1 for result in results if result == 'io')
        if io_errors:
            self.read_failures += 1
            if self.read_failures >= 3:
                self.latch('three NVMe reads in a row failed')
        elif all(result is None for result in results):
            self.read_failures = 0
        if bad:
            self._forget(bad)
        c['nvme_restore_ms'] += (time.perf_counter() - began) * 1000.0
        return results

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
                if self.read_failures >= 3:
                    self.latch('three NVMe reads in a row failed')
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

    def delete_tenant(self, tag):
        removed = self.store.delete_tenant(tag)
        self.c['nvme_deleted_tenant'] += removed
        self.log('nvme tier tenant delete entries=%d', removed)
        return removed

    def wipe(self, why='requested'):
        """Remove everything, durably and at once (a reset_prefix_cache, the kill switch). The queued spills are dropped. -> entries."""
        self._drop_queue()
        removed = self.store.wipe()
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
            with self.cond:
                while not self.queue and not self.commands and not self.closed:
                    self.cond.wait(timeout=1.0)
                if self.closed and not self.queue and not self.commands:
                    return
                commands = list(self.commands)
                self.commands.clear()
                batch, size = [], 0
                while self.queue and len(batch) < MAX_BATCH_ITEMS and size < MAX_BATCH_BYTES:
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
                                self.queue_kv_bytes -= item.nbytes
                            else:
                                self.queue_ckpt_bytes -= item.nbytes

    def _wait_for_readers(self):
        """A restore is latency-critical and a spill is not: hold the next write while a read is in flight (bounded)."""
        waited = 0.0
        while self.reading and waited < WRITER_YIELD_MAX_S and not self.closed:
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

    def _write_batch(self, batch):
        w = self.w
        began = time.perf_counter()
        kvs = sorted((item for item in batch if isinstance(item, SpillKV)), key=lambda item: (bytes(item.chain or b''), item.index))
        ckpts = [item for item in batch if isinstance(item, SpillCkpt)]
        puts, owners = [], []
        try:
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
                # one retry: a transient device error must not cost a session the one chance its blocks had to be written
                w['nvme_spill_retries'] += 1
                self.log('nvme tier spill retry after %s: %s', type(error).__name__, error)
                time.sleep(0.05)
                results = self.store.put_batch(puts)
        except Exception as error:
            w['nvme_spill_failures'] += 1
            self.failures += 1
            self.log('nvme tier spill failed (%d so far): %s: %s', self.failures, type(error).__name__, error)
            if self.failures >= 3:
                self.latch('three NVMe writes in a row failed')
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
        values.update(nvme_on=int(self.active), nvme_bytes=stats['logical_bytes'], nvme_entries=stats['entries'], nvme_kv_entries=stats['kv_entries'],
                      nvme_ckpt_entries=stats['ckpt_entries'], nvme_capacity_bytes=stats['capacity_bytes'], nvme_free_bytes=stats['free_bytes'],
                      nvme_queue_bytes=self.queue_kv_bytes + self.queue_ckpt_bytes, nvme_journal_used_bytes=stats['journal_used'],
                      nvme_largest_free_bytes=stats['largest_free'], nvme_through_backlog_blocks=self.backlog_blocks)
        return values

    def describe(self):
        """The fields of the scheduler graft's 'nvme tier on' line."""
        return dict(device=self.plan.device, backend=self.backend.name, ttl_s=self.plan.ttl_s, max_age_s=self.plan.max_age_s, verify=self.plan.verify,
                    audit=int(self.plan.audit), queue_bytes=self.plan.queue_bytes, capacity_bytes=self.store.capacity_bytes(), fingerprint=self.fp.hex(),
                    spill=self.plan.spill, through_blocks=self.plan.through_blocks if self.plan.through else 0,
                    through_idle_s=self.plan.through_idle_s if self.plan.through else 0)

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

    def close(self):
        if self.closed:
            return
        self.flush(timeout=10.0)
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.writer.join(timeout=10.0)
        if self._hash_pool is not None:
            self._hash_pool.shutdown(wait=True)
            self._hash_pool = None
        self.store.close()
