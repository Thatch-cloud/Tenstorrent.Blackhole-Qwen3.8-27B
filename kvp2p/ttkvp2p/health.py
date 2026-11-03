"""Start-up host checks (design 2.9 step 1), the control directory (3.6) and the state file (3.6), plus the latch vocabulary.

Everything here is stdlib only and does no device work: it reads files and sysfs, never talks to a card. ``HostEnv`` carries every
outside input (the environment, the clocks, the sysfs root, the p2pmem probe), so a test can run the whole sequence on a temporary
tree.

The reason codes are a closed vocabulary: a REFUSED reason when the node's configuration or consent is wrong, an UNAVAILABLE reason when
the host cannot give the tier what it needs, a LATCHED code when it trips at run time. A later version may only add to them.
"""

import glob as glob_module
import json
import logging
import mmap
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------------------------------------

REFUSED_CODES = ('schema_invalid', 'gate_only_key', 'retention_ack_missing', 'zero_retention', 'ttl_over_floor', 'sed_not_required')
UNAVAILABLE_CODES = ('at_rest_unverified', 'status_stale', 'device_absent', 'device_dead', 'link_below_floor', 'claim_busy',
                     'superblock_unknown', 'layout_mismatch', 'capacity_short', 'p2pmem_unavailable', 'canary_failed',
                     'host_check_error')        # the last one is an addition to the v1 table (an exception inside the host checks): KV-TIER-STANDARD 5.1 needs it
LATCH_CODES = ('strikes', 'device_lost', 'aer_uncorrectable', 'stall', 'op_throw', 'bounds_violation', 'control_off', 'write_cap')

TTL_FLOOR_S = 86400                    # the owner's 10-21 rule: KV lives at most a day since its last use
STATUS_MAX_AGE_S = 120.0               # the cache-drive status file must be this fresh at start-up
STATUS_CLOCK_SKEW_S = 5.0
LINK_FLOOR_GTS = 16.0                  # Gen4 on this rig
SPAWN_ID_ENV = 'THATCH_KVTIER_SPAWN_ID'
TEST_ENV = 'THATCH_KVP2P_TEST'
ZERO_RETENTION_ENV = ('THATCH_ATTESTATION_PRIVATE_KEY', 'THATCH_ZERO_RETENTION', 'THATCH_NO_LOG')
ZERO_RETENTION_MARKER_NAME = 'zero-retention'     # a file next to the control directory, or the one named by cfg.zero_retention_marker; its existence is a signal
STATE_SCHEMA = 1


class HostRefusal(Exception):
    """A host check failed. ``code`` is from the closed vocabulary; ``refused`` says whether the node's configuration is at fault
    (REFUSED) or the host cannot supply what the tier needs (UNAVAILABLE)."""

    def __init__(self, code: str, detail: str):
        super().__init__('%s: %s' % (code, detail))
        self.code = code
        self.detail = detail
        self.refused = code in REFUSED_CODES


def latch_code(reason: str) -> str:
    """Classify a free-text latch reason (the worker's wording) into the closed LATCHED vocabulary."""
    text = (reason or '').lower()
    for code in LATCH_CODES:
        if text == code:
            return code
    if 'control' in text and 'off' in text:
        return 'control_off'
    if 'consecutive job failure' in text or 'strike' in text:
        return 'strikes'
    if 'no completion' in text or 'stall' in text:
        return 'stall'
    if 'aer' in text:
        return 'aer_uncorrectable'
    if 'device lost' in text or 'enodev' in text or 'enxio' in text or 'not live' in text or 'reads 0 bytes' in text:
        return 'device_lost'
    if 'bound' in text or 'canary' in text:
        return 'bounds_violation'
    if 'write cap' in text or 'daily cap' in text:
        return 'write_cap'
    return 'op_throw'       # an internal error is an exception in the tier's own state machine: the closest closed code


# --------------------------------------------------------------------------------------------------------------------
# The outside world
# --------------------------------------------------------------------------------------------------------------------

def real_p2pmem_probe(path: str, size: int) -> None:
    """Map the whole published slice read-write and let go (design 2.9: "the mmap of each slice succeeds"; it also tests the
    AppArmor profile). Raises OSError when the map fails."""
    fd = os.open(path, os.O_RDWR)
    try:
        mapped = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        mapped.close()
    finally:
        os.close(fd)


@dataclass
class HostEnv:
    """Every input from outside the process. Tests replace the whole object (``runtime.set_host_env``)."""

    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    sysfs_pci: str = '/sys/bus/pci/devices'
    wall: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic
    glob: Callable[[str], List[str]] = glob_module.glob
    p2pmem_probe: Callable[[str, int], None] = real_p2pmem_probe
    sync_io: bool = False             # tests: run drive I/O on the caller's thread (completions still reach the engine thread through the queue)

    @property
    def test_mode(self) -> bool:
        return str(self.environ.get(TEST_ENV, '')).strip() == '1'

    @property
    def spawn_id(self) -> Optional[str]:
        return self.environ.get(SPAWN_ID_ENV) or None


# --------------------------------------------------------------------------------------------------------------------
# Policy checks: configuration, retention consent, zero-retention signals
# --------------------------------------------------------------------------------------------------------------------

def zero_retention_signals(cfg: Any, env: HostEnv) -> List[str]:
    """Why this process looks like it runs on a node that advertises zero retention ([] when nothing says so): the node agent's
    attestation key or the forwarded flag in the environment, the log control, or the marker file beside the control directory."""
    reasons: List[str] = []
    environ = env.environ
    if environ.get('THATCH_ATTESTATION_PRIVATE_KEY'):
        reasons.append('the node\'s attestation key is in this process (the node signs zero-retention teardowns)')
    if str(environ.get('THATCH_ZERO_RETENTION', '')).strip().lower() in ('1', 'true', 'yes', 'on'):
        reasons.append('THATCH_ZERO_RETENTION is set (the node advertises zero retention)')
    if str(environ.get('THATCH_NO_LOG', '')).strip() == '1':
        reasons.append('THATCH_NO_LOG is set (the container runs under the zero-retention log control)')
    # The marker files: one beside the control directory (this tier's own tree) and the node-level one the node operator names in
    # ``zero_retention_marker`` (node.env; a node-wide path is the operator's to agree, so no default is invented here).
    markers: List[str] = []
    control_dir = getattr(cfg, 'control_dir', '') or ''
    if control_dir:
        markers.append(os.path.join(os.path.dirname(control_dir.rstrip('/')), ZERO_RETENTION_MARKER_NAME))
    node_marker = getattr(cfg, 'zero_retention_marker', '') or ''
    if node_marker:
        markers.append(node_marker)
    for marker in markers:
        try:
            if os.path.exists(marker):
                reasons.append('%s exists (a zero-retention marker on the node)' % ZERO_RETENTION_MARKER_NAME)
        except OSError:
            pass
    return reasons


def check_policy(cfg: Any, env: HostEnv) -> None:
    """Refuse a configuration the tier must not run under. Raises HostRefusal."""
    if cfg.persistence != 'process':
        raise HostRefusal('schema_invalid', 'persistence %r (v1a keeps nothing across a restart; "durable" is v1b)' % (cfg.persistence,))
    if cfg.ttl_s <= 0:
        raise HostRefusal('schema_invalid', 'ttl_s %r' % (cfg.ttl_s,))
    if cfg.ttl_s > TTL_FLOOR_S:
        raise HostRefusal('ttl_over_floor', 'ttl_s %d is over the %d s floor (KV is kept at most a day since its last use)' % (cfg.ttl_s, TTL_FLOOR_S))
    if cfg.free_floor_pct < 0 or cfg.free_floor_pct >= 100 or cfg.reserve_bytes < 0 or cfg.daily_write_cap_bytes < 0 or cfg.min_store_tokens < 0:
        raise HostRefusal('schema_invalid', 'a capacity, cap or threshold is out of range')
    signals = zero_retention_signals(cfg, env)
    if signals:
        raise HostRefusal('zero_retention', '; '.join(signals))
    if not cfg.retention_ack:
        raise HostRefusal('retention_ack_missing', 'retention_ack is not true: keeping KV on a drive is a per-node operator decision')
    if not cfg.require_sed and not env.test_mode:
        raise HostRefusal('sed_not_required', 'require_sed=false is a gate-only value (it needs %s=1)' % TEST_ENV)


# --------------------------------------------------------------------------------------------------------------------
# The cache-drive status file (written by the host's boot unit and monitor timer)
# --------------------------------------------------------------------------------------------------------------------

_ISO = re.compile(r'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$')


def parse_timestamp(value: Any) -> Optional[float]:
    """Seconds since the epoch from an ISO-8601 UTC string or a number; None when it cannot be read."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _ISO.match(value.strip())
    if not match:
        return None
    import calendar
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    stamp = float(calendar.timegm((year, month, day, hour, minute, second)))
    zone = match.group(7)
    if zone and zone != 'Z':
        sign = 1 if zone[0] == '+' else -1
        digits = zone[1:].replace(':', '')
        stamp -= sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
    return stamp


def parse_link_gts(text: Any) -> Optional[float]:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    match = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s*GT/s', str(text or ''))
    return float(match.group(1)) if match else None


@dataclass
class StatusFacts:
    by_id: str
    state: str
    sed: str
    locking_enabled: Optional[bool]
    locked: Optional[bool]
    link_gts: Optional[float]
    timestamp: Optional[float]
    age_s: Optional[float]
    aer_uncorrectable: int
    aer_correctable: int
    raw_state: Dict[str, Any] = field(default_factory=dict)


def _sum_aer(node: Any, wanted: Callable[[str], bool]) -> int:
    """Sum every number under ``status['aer']`` whose key satisfies ``wanted`` (the monitor's exact shape is not frozen: per device or flat)."""
    total = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                total += _sum_aer(value, wanted)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and wanted(str(key).lower()):
                total += int(value)
    elif isinstance(node, list):
        for item in node:
            total += _sum_aer(item, wanted)
    return total


def _is_uncorrectable(key: str) -> bool:
    return 'uncorrect' in key or 'fatal' in key


def _is_correctable(key: str) -> bool:
    return 'correctable' in key and 'uncorrect' not in key


def read_status(status_dir: str, env: HostEnv) -> StatusFacts:
    """Read and parse ``status.json``. Raises HostRefusal ``at_rest_unverified`` when it is absent or unreadable."""
    path = os.path.join(status_dir, 'status.json')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise HostRefusal('at_rest_unverified', 'no cache-drive status file at the status directory')
    except (OSError, ValueError) as error:
        raise HostRefusal('at_rest_unverified', 'the cache-drive status file cannot be read (%s)' % (error,))
    if not isinstance(data, dict):
        raise HostRefusal('at_rest_unverified', 'the cache-drive status file is not an object')
    stamp = parse_timestamp(data.get('timestamp', data.get('ts')))
    age = None if stamp is None else env.wall() - stamp
    locking = data.get('locking_enabled')
    locked = data.get('locked')
    aer = data.get('aer')
    return StatusFacts(
        by_id=str(data.get('by_id') or ''), state=str(data.get('state') or ''), sed=str(data.get('sed') or ''),
        locking_enabled=locking if isinstance(locking, bool) else None, locked=locked if isinstance(locked, bool) else None,
        link_gts=parse_link_gts(data.get('link_speed')), timestamp=stamp, age_s=age,
        aer_uncorrectable=_sum_aer(aer, _is_uncorrectable), aer_correctable=_sum_aer(aer, _is_correctable),
        raw_state=data)


def check_status(cfg: Any, env: HostEnv) -> StatusFacts:
    """Design 2.9 step 1, Status: the file is fresh (120 s or less), names a drive, says live, unlocked, at rest by Opal, 16 GT/s or more."""
    facts = read_status(cfg.nvme_status_dir, env)
    if facts.age_s is None:
        raise HostRefusal('status_stale', 'the status file has no readable timestamp')
    if facts.age_s > STATUS_MAX_AGE_S or facts.age_s < -STATUS_CLOCK_SKEW_S:
        raise HostRefusal('status_stale', 'the status file is %.0f s old (it must be at most %.0f s)' % (facts.age_s, STATUS_MAX_AGE_S))
    if not facts.by_id:
        raise HostRefusal('at_rest_unverified', 'the status file does not name the drive')
    if facts.state != 'ok':
        code = 'device_dead' if facts.state in ('not_live', 'probe_failed', 'setup_failed') else 'at_rest_unverified'
        raise HostRefusal(code, 'the cache drive reports state %r' % (facts.state,))
    if cfg.require_sed:
        if facts.sed != 'opal' or facts.locking_enabled is not True or facts.locked is not False:
            raise HostRefusal('at_rest_unverified', 'the drive is not verified at rest (sed=%r locking_enabled=%r locked=%r)'
                              % (facts.sed, facts.locking_enabled, facts.locked))
    if facts.link_gts is None or facts.link_gts < LINK_FLOOR_GTS:
        raise HostRefusal('link_below_floor', 'the drive link is %s GT/s (at least %.0f needed)' % (facts.link_gts, LINK_FLOOR_GTS))
    return facts


# --------------------------------------------------------------------------------------------------------------------
# p2pmem
# --------------------------------------------------------------------------------------------------------------------

def _sysfs_int(path: str) -> Optional[int]:
    try:
        with open(path, 'r') as handle:
            return int(handle.read().strip(), 0)
    except (OSError, ValueError):
        return None


def check_p2pmem(cfg: Any, chips: int, env: HostEnv) -> List[str]:
    """Design 2.9 step 1, p2pmem: one allocate file per card, each slice whole (``available == size``), and the mmap of every slice
    succeeds. Returns the allocate paths. Raises HostRefusal ``p2pmem_unavailable``."""
    paths = sorted(env.glob(cfg.p2pmem_glob))
    if len(paths) != chips:
        raise HostRefusal('p2pmem_unavailable', 'found %d p2pmem allocate files, the layout needs %d' % (len(paths), chips))
    for path in paths:
        card = os.path.basename(os.path.dirname(path))
        base = os.path.join(env.sysfs_pci, card, 'p2pmem')
        size = _sysfs_int(os.path.join(base, 'size'))
        available = _sysfs_int(os.path.join(base, 'available'))
        if not size or available is None:
            raise HostRefusal('p2pmem_unavailable', 'the kernel reports no peer-to-peer memory for %s' % card)
        if available != size:
            raise HostRefusal('p2pmem_unavailable', 'the slice of %s is already allocated (available %d of %d): another holder has it' % (card, available, size))
        try:
            env.p2pmem_probe(path, size)
        except OSError as error:
            raise HostRefusal('p2pmem_unavailable', 'cannot map the slice of %s (%s)' % (card, error.strerror or error))
    return paths


# --------------------------------------------------------------------------------------------------------------------
# The control directory
# --------------------------------------------------------------------------------------------------------------------

_CONTROL_NAME = re.compile(r'^(\d{1,9})-(off|on|wipe|delete_tenant|fault|selftest)$')
LEVEL_VERBS = ('off', 'on')
EDGE_VERBS = ('wipe', 'delete_tenant', 'fault', 'selftest')
TEST_ONLY_VERBS = ('fault', 'selftest')
FAULT_KINDS = ('eio', 'digest', 'stall', 'throw')


@dataclass
class ControlCommand:
    seq: int
    verb: str
    body: str
    name: str


def list_control(control_dir: str) -> List[ControlCommand]:
    """The command files of the control directory in order. A name that does not match is ignored (never an error)."""
    commands: List[ControlCommand] = []
    try:
        names = os.listdir(control_dir)
    except OSError:
        return commands
    for name in names:
        match = _CONTROL_NAME.match(name)
        if not match:
            continue
        body = ''
        if match.group(2) in ('fault', 'delete_tenant'):
            try:
                with open(os.path.join(control_dir, name), 'r', encoding='utf-8', errors='replace') as handle:
                    body = handle.read(256).strip()
            except OSError:
                body = ''
        commands.append(ControlCommand(int(match.group(1)), match.group(2), body, name))
    commands.sort(key=lambda command: (command.seq, command.verb))
    return commands


def switch_is_off(commands: List[ControlCommand]) -> bool:
    """The level verbs: the highest-numbered off/on file is the switch. No file means on."""
    last = None
    for command in commands:
        if command.verb in LEVEL_VERBS:
            last = command
    return last is not None and last.verb == 'off'


def parse_fault(body: str) -> Optional[Tuple[str, int]]:
    """``eio 3`` -> ('eio', 3). A fault without a count is one."""
    parts = body.split()
    if not parts or parts[0] not in FAULT_KINDS:
        return None
    count = 1
    if len(parts) > 1:
        try:
            count = int(parts[1])
        except ValueError:
            return None
    return parts[0], max(1, min(count, 1000))


# --------------------------------------------------------------------------------------------------------------------
# The state file
# --------------------------------------------------------------------------------------------------------------------

def iso_utc(seconds: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seconds))


def write_json_atomic(path: str, payload: Dict[str, Any], mode: int = 0o644) -> None:
    """Write ``payload`` to ``path`` without ever exposing a partial file. Creates missing parent directories."""
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    temp = '%s.tmp.%d.%d' % (path, os.getpid(), threading.get_ident())      # a latch can write from an I/O thread while the engine thread writes
    data = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    try:
        os.chmod(temp, mode)
    except OSError:
        pass
    os.replace(temp, path)


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]
