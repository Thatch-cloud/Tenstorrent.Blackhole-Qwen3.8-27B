"""G1 of the TT prefix-reuse design (section 2.2, S5): the prefix registry's metrics, from the EngineCore
to the API server's /metrics.

The checkpoint registry (prefix_scheduler_graft.PrefixRegistry, one per process under the fixed
sys.modules key REGISTRY_KEY) lives in the vLLM EngineCore, a child process of the API server
(SKILL:211-212). vLLM's Prometheus endpoint lives in the API server. So the engine publishes and the API
server collects, through a tmpfs file per engine process:

  engine side  start_exporter() runs a daemon thread that, while a registry exists in the process,
               writes its snapshot every QWEN_PREFIX_METRICS_INTERVAL_S seconds (default 2) to
               <QWEN_PREFIX_METRICS_DIR>/<pid>.json (default /tmp/qwen-prefix-metrics; /tmp is a
               tmpfs in the node agent's container), atomically. A process without a registry writes
               nothing. serving_c2_contract.boot starts it in every non-API-server process of a prefix
               profile (a spawned EngineCore), and export_in_forked_children starts it in every forked
               child that did not inherit a registry (vLLM forks the EngineCore by default).
  API side     install_collector() wraps vllm.v1.metrics.prometheus.get_prometheus_registry - the one
               function vLLM's /metrics route takes its registry from (entrypoints/serve/instrumentator/
               metrics.py:59,78) - so the returned registry carries PrefixCollector. Each scrape reads
               every file whose writer is alive (pid and /proc start time: an engine restarted inside
               the container leaves its old file behind), sums them, and also counts a registry in the
               API server's own process (an in-process engine). Nothing is cached between scrapes.

Why not SchedulerStats: it crosses the process boundary as a msgspec-encoded dataclass whose fields
are fixed (v1/metrics/stats.py:171-198); the only free-form one, kv_connector_stats, is handed to the
KV connector's own logging in the frontend (v1/metrics/loggers.py:184-185,1108-1110), and the scheduler
graft refuses a KV connector. A tmpfs file needs no vLLM change on either side.

The metrics (prefix qwen_prefix_): every counter the registry keeps, as <name>_total (the cumulative
*_ms timers as <name>_seconds_total) - grants, grant tokens, trim loss (h - Q), KV hit without a
checkpoint, orphans (registry hit without KV), token mismatches, same-step rejects, captures and
capture failures, LRU and coupled evictions, unsalted and kill-switch denials, restore and capture time
- and gauges for the checkpoints held, their host bytes, the byte budget, pins, and staged/committed
grants; plus the export's own health: live writers, the oldest live write's age, dead writers' files,
unreadable files, and whether the kill-switch file exists.

The hit rate to report is the model's (L - Q), from grant_tokens against vllm:prompt_tokens; never
vllm:prefix_cache_hits, which vLLM counts inside get_computed_blocks before the trim (design 2.2 need 5).

Stdlib only at import; prometheus_client is imported by the collector, in the API server only.
"""

import json
import os
import re
import sys
import threading
import time

REGISTRY_KEY = '_qwen_prefix_registry'
KILL_SWITCH_PATH = '/models/.qwen-c2/prefix-reuse.off'
DEFAULT_DIRECTORY = '/tmp/qwen-prefix-metrics'
DEFAULT_INTERVAL_S = 2.0
SCHEMA = 1
PREFIX = 'qwen_prefix_'
PROMETHEUS_MODULE = 'vllm.v1.metrics.prometheus'
NAME = re.compile(r'^[a-z_][a-z0-9_]*$')

# Registry snapshot keys that are levels, not running totals; everything else numeric is a counter.
GAUGES = {
    'entries': 'GDN checkpoints held in host RAM',
    'bytes': 'host bytes the held checkpoints occupy',
    'budget_bytes': 'the checkpoint store budget (QWEN_PREFIX_STORE_GIB)',
    'pins': 'checkpoint pins held by this step\'s committed grants',
    'staged_now': 'grants staged in the current schedule() call',
    'committed_now': 'grants committed for the current step',
}
COUNTERS = {
    'attempts': 'admission attempts the trim saw (vLLM calls get_computed_blocks once per attempt)',
    'staged': 'grants staged (idempotent per attempt)',
    'dropped_attempts': 'staged grants dropped because the step did not admit the request',
    'commit_mismatch': 'grants refused at commit because start_pos differed from Q',
    'admissions': 'admissions committed with a grant record (hit or planned capture)',
    'grants': 'admissions granted a checkpoint (a hit)',
    'grant_tokens': 'prompt tokens served from the prefix cache after the trim (sum of Q)',
    'trim_loss_tokens': 'tokens of vLLM hit the trim gave up (sum of h - Q)',
    'kv_hit_without_checkpoint': 'admissions whose KV hit reached a boundary with no usable checkpoint',
    'orphans': 'checkpoints found past the KV hit (registry hit without KV)',
    'same_step_rejects': 'hits cut back because a block was cached in the same scheduler step',
    'token_mismatches': 'checkpoints refused because their stored token ids differ from the request',
    'unsalted_denied': 'hits denied to requests without a cache_salt (fail closed)',
    'killed_denied': 'hits denied after the kill switch latched',
    'publish_capped': 'publish calls capped at the last full-chunk boundary of the prompt',
    'captures': 'checkpoints stored',
    'capture_replaced': 'captures that replaced an unpinned checkpoint of the same prefix',
    'capture_kept_pinned': 'captures that kept the pinned checkpoint of the same prefix',
    'capture_failures': 'captures skipped on an error (the request still served)',
    'capture_skipped_budget': 'captures larger than the whole store budget',
    'evicted_lru': 'checkpoints evicted by the byte budget',
    'evicted_coupled': 'checkpoints dropped because vLLM evicted their boundary block',
    'dropped': 'checkpoints dropped for another reason',
    'clears': 'registry clears (reset_prefix_cache, the kill switch)',
    'freed_requests': 'requests whose grants were dropped when vLLM freed them',
    'restore_ms': 'time restoring checkpoints into the prefill scratch',
    'capture_ms': 'time capturing checkpoints to host',
}


def log(message, *values):
    try:
        sys.stderr.write('[PINDIAG] prefix: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def settings(environ=None):
    environ = os.environ if environ is None else environ
    directory = environ.get('QWEN_PREFIX_METRICS_DIR') or DEFAULT_DIRECTORY
    try:
        interval = float(environ.get('QWEN_PREFIX_METRICS_INTERVAL_S') or DEFAULT_INTERVAL_S)
    except ValueError:
        interval = DEFAULT_INTERVAL_S
    return directory, max(interval, 0.1)


# -- the engine side ---------------------------------------------------------------------------
def process_registry(modules=None):
    """The registry this process holds under REGISTRY_KEY, or None."""
    holder = (sys.modules if modules is None else modules).get(REGISTRY_KEY)
    return getattr(holder, 'registry', None)


def numbers(values):
    """The numeric entries of a snapshot, with metric-safe names (bools as 0/1)."""
    kept = {}
    for key, value in (values or {}).items():
        if not isinstance(key, str) or not NAME.match(key):
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            kept[key] = value
    return kept


def read_registry(registry, attempts=3):
    """registry.snapshot() plus its byte budget. The scheduler thread mutates the registry while this
    thread reads it, so an OrderedDict 'mutated during iteration' is retried."""
    for attempt in range(attempts):
        try:
            values = dict(registry.snapshot())
            break
        except RuntimeError:
            if attempt == attempts - 1:
                raise
    budget = getattr(registry, 'budget_bytes', None)
    if budget is not None and 'budget_bytes' not in values:
        values['budget_bytes'] = budget
    return numbers(values)


def start_ticks(pid, proc='/proc'):
    """A process's start time in clock ticks (/proc/<pid>/stat field 22), or None where unreadable."""
    try:
        with open(os.path.join(proc, str(pid), 'stat'), 'rb') as handle:
            data = handle.read().decode('ascii', 'replace')
        return int(data.rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def process_alive(pid, ticks=None, proc='/proc', kill=None):
    """Whether `pid` is the process that wrote a file: alive, and started at `ticks` when both are known.
    Off POSIX (a developer's Windows checkout) nothing is signalled - signal 0 is CTRL_C_EVENT there -
    and a writer counts as alive."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if ticks is not None:
        now = start_ticks(pid, proc)
        if now is not None:
            return now == ticks
    if kill is None:
        if os.name != 'posix':
            return True
        kill = os.kill
    try:
        kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class Exporter(object):
    """Writes this process's registry snapshot to <directory>/<pid>.json."""

    def __init__(self, directory=None, interval_s=None, clock=time.time, lookup=process_registry, pid=None,
                 kill_switch_path=KILL_SWITCH_PATH, proc='/proc', logger=log, alive=process_alive):
        default_directory, default_interval = settings()
        self.directory = directory or default_directory
        self.interval_s = interval_s or default_interval
        self.clock, self.lookup, self.log = clock, lookup, logger
        self.pid = os.getpid() if pid is None else pid
        self.ticks = start_ticks(self.pid, proc)
        self.kill_switch_path = kill_switch_path
        self.alive = alive
        self.writes = 0
        self.errors = 0
        self.announced = False
        self.path = os.path.join(self.directory, '%d.json' % self.pid)

    def document(self, registry):
        return dict(schema=SCHEMA, pid=self.pid, start_ticks=self.ticks, written_unix=self.clock(),
                    interval_s=self.interval_s, writes=self.writes + 1, errors=self.errors,
                    kill_switch_file=bool(self.kill_switch_path and os.path.exists(self.kill_switch_path)),
                    values=read_registry(registry))

    def publish(self):
        """Write one snapshot; False when this process holds no registry (nothing is written)."""
        registry = self.lookup()
        if registry is None:
            return False
        document = self.document(registry)
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        temporary = '%s.%d.tmp' % (self.path, threading.get_ident())
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(document, handle, sort_keys=True)
        os.replace(temporary, self.path)
        self.writes += 1
        if not self.announced:
            self.announced = True
            self.log('metrics export pid=%d -> %s every %.1fs; first snapshot entries=%s bytes=%s grants=%s',
                     self.pid, self.path, self.interval_s, document['values'].get('entries'),
                     document['values'].get('bytes'), document['values'].get('grants'))
        return True

    def sweep(self):
        """Remove the files of writers that are gone (an engine restarted inside the container)."""
        removed = 0
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        for name in names:
            if not name.endswith('.json') or name == os.path.basename(self.path):
                continue
            path = os.path.join(self.directory, name)
            document = load_document(path)
            if document is not None and self.alive(document.get('pid'), document.get('start_ticks')):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed

    def run(self, stop):
        self.sweep()
        while True:
            try:
                self.publish()
            except Exception as error:
                self.errors += 1
                if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                    self.log('metrics export failed (%d so far): %s: %s', self.errors, type(error).__name__, error)
            if stop.wait(self.interval_s):
                return


_EXPORTER = {}


def start_exporter(environ=None, factory=Exporter):
    """Start this process's exporter thread, once per process. Returns the Exporter."""
    pid = os.getpid()
    state = _EXPORTER.get('state')
    if state is not None and state['pid'] == pid:
        return state['exporter']
    directory, interval = settings(environ)
    exporter = factory(directory, interval)
    stop = threading.Event()
    thread = threading.Thread(target=exporter.run, args=(stop,), name='qwen-prefix-metrics', daemon=True)
    _EXPORTER['state'] = dict(pid=pid, exporter=exporter, stop=stop, thread=thread)
    thread.start()
    return exporter


def after_fork_in_child(environ=None, factory=Exporter, lookup=process_registry):
    """In a forked child: start its own exporter, unless it inherited a registry - that is the
    parent's, copied at fork, and exporting it too would count the parent twice."""
    if lookup() is not None:
        return None
    return start_exporter(environ, factory)


def export_in_forked_children(environ=None, register=None):
    """vLLM forks the EngineCore from the API server by default (VLLM_WORKER_MULTIPROC_METHOD=fork,
    utils/system_utils.py:168-181; engine/utils.py:139), and a forked child neither re-runs the .pth
    boot nor keeps its parent's threads. So every process of a prefix profile registers this once."""
    if _EXPORTER.get('fork_hook'):
        return False
    register = register if register is not None else getattr(os, 'register_at_fork', None)
    if register is None:
        return False
    register(after_in_child=lambda: after_fork_in_child(environ))
    _EXPORTER['fork_hook'] = True
    return True


def stop_exporter(timeout=5.0):
    state = _EXPORTER.pop('state', None)
    if state is not None:
        state['stop'].set()
        state['thread'].join(timeout)
    return state is not None


# -- the API side ------------------------------------------------------------------------------
def load_document(path):
    try:
        with open(path, encoding='utf-8') as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get('schema') != SCHEMA or not isinstance(
            document.get('values'), dict):
        return None
    return document


def read_documents(directory, alive=process_alive):
    """(live documents, dead writers' files, unreadable files) under directory."""
    live, dead, unreadable = [], 0, 0
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return live, dead, unreadable
    for name in names:
        if not name.endswith('.json'):
            continue
        document = load_document(os.path.join(directory, name))
        if document is None:
            unreadable += 1
        elif alive(document.get('pid'), document.get('start_ticks')):
            live.append(document)
        else:
            dead += 1
    return live, dead, unreadable


def metric_name(key):
    """(metric name without _total, kind, help) for a snapshot key."""
    if key in GAUGES:
        return PREFIX + ('registry_' + key if key in ('bytes', 'entries', 'budget_bytes', 'pins') else key), \
            'gauge', GAUGES[key]
    help_text = COUNTERS.get(key, 'prefix registry counter %s' % key)
    if key.endswith('_ms'):
        return PREFIX + key[:-3] + '_seconds', 'counter', help_text + ' (seconds)'
    return PREFIX + key, 'counter', help_text


def metric_rows(documents, dead=0, unreadable=0, now=None, local=None):
    """[(kind, name, help, value)]: the registries' values summed over every live writer (and the
    local registry, when this process holds one), then the export's own health."""
    now = time.time() if now is None else now
    totals = {}
    for values in [document['values'] for document in documents] + ([local] if local is not None else []):
        for key, value in numbers(values).items():
            totals[key] = totals.get(key, 0) + value
    rows = []
    for key in sorted(totals):
        name, kind, help_text = metric_name(key)
        value = totals[key]
        if key.endswith('_ms') and kind == 'counter':
            value = value / 1000.0
        rows.append((kind, name, help_text, value))
    ages = [max(0.0, now - float(document.get('written_unix') or 0)) for document in documents]
    rows.extend((
        ('gauge', PREFIX + 'export_writers', 'engine processes exporting a prefix registry (live files)',
         len(documents) + (1 if local is not None else 0)),
        ('gauge', PREFIX + 'export_age_seconds', 'age of the oldest live export (0 without one)',
         max(ages) if ages else 0.0),
        ('gauge', PREFIX + 'export_dead_files', 'export files whose writer is gone (restarted engines)', dead),
        ('gauge', PREFIX + 'export_unreadable_files', 'export files that did not parse', unreadable),
        ('gauge', PREFIX + 'kill_switch_file', 'the kill-switch file exists (reuse latches off until a restart)',
         int(any(document.get('kill_switch_file') for document in documents))),
    ))
    return rows


class PrefixCollector(object):
    """A prometheus_client custom collector over the engine exports (see the module docstring)."""

    def __init__(self, directory=None, alive=process_alive, clock=time.time, lookup=process_registry):
        self.directory = directory or settings()[0]
        self.alive, self.clock, self.lookup = alive, clock, lookup

    def rows(self):
        documents, dead, unreadable = read_documents(self.directory, self.alive)
        local = None
        registry = self.lookup()
        if registry is not None:
            try:
                local = read_registry(registry)
            except Exception:
                local = None
        return metric_rows(documents, dead, unreadable, self.clock(), local)

    def collect(self):
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        for kind, name, help_text, value in self.rows():
            family = CounterMetricFamily if kind == 'counter' else GaugeMetricFamily
            yield family(name, help_text, value=value)

    def describe(self):
        # Names vary with the registry's counters; an empty describe keeps registration from calling
        # collect() before an engine exists.
        return []


_COLLECTED = {}


def register_collector(registry, directory=None, factory=PrefixCollector):
    """Put a PrefixCollector on `registry` once (keyed by identity); returns the registry."""
    if registry is None or id(registry) in _COLLECTED:
        return registry
    collector = factory(directory)
    registry.register(collector)
    _COLLECTED[id(registry)] = (registry, collector)
    log('metrics collector on %s.%s reading %s', type(registry).__module__, type(registry).__name__,
        collector.directory)
    return registry


def wrap_prometheus_module(module, directory=None):
    """Make module.get_prometheus_registry hand out registries that carry the collector."""
    original = getattr(module, 'get_prometheus_registry', None)
    if original is None or getattr(original, '_qwen_prefix', False):
        return False

    def get_prometheus_registry(*args, **kwargs):
        registry = original(*args, **kwargs)
        try:
            register_collector(registry, directory)
        except Exception as error:
            log('metrics collector not registered: %s: %s', type(error).__name__, error)
        return registry

    get_prometheus_registry._qwen_prefix = True
    get_prometheus_registry.__wrapped__ = original
    module.get_prometheus_registry = get_prometheus_registry
    return True


def install_collector(on_import, directory=None, modules=None):
    """In the API server: wrap vLLM's registry getter now if it is imported, else when it is.
    on_import(name, callback) runs callback(module) right after `name` executes
    (serving_c2_contract.PostImportHook)."""
    modules = sys.modules if modules is None else modules
    module = modules.get(PROMETHEUS_MODULE)
    if module is not None:
        return wrap_prometheus_module(module, directory)
    on_import(PROMETHEUS_MODULE, lambda loaded: wrap_prometheus_module(loaded, directory))
    return True
