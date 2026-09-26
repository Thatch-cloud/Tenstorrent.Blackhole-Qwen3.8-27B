"""G1 of the TT prefix-reuse design (section 2.2, S5): the prefix registry's metrics, from the EngineCore
to the API server's /metrics.

The checkpoint registry (qwen_prefix_registry.PrefixRegistry, one per process under the fixed
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

The API server also counts chat requests by the fields that decide whether the next turn's prompt
extends this one - top-level reasoning_effort and chat_template_kwargs' reasoning_effort,
preserve_thinking and enable_thinking (P0b, correction C6) - and by whether a cache_salt came with it
(none: no hit, fail closed): qwen_prefix_chat_requests_total{...}, labels bounded. And it counts what the
contract's salt policy did with each request's cache_salt (serving_c2_contract.salt_verdict: verified,
dropped-unverified, dropped-no-key, unset): qwen_prefix_salts_total{verdict}.

Nothing here raises into serving: the exporter and collector swallow their own failures, and the two
wraps the API server installs when vLLM imports its modules (install_collector) are guarded, because
they run inside vLLM's own import of those modules.

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
    'orphans_now': 'resident checkpoints whose KV chain is broken below them',
    'mid_loop_capture': '1 once the model graft declared captures inside its chunk loop (gap boundaries planned)',
    'disabled': '1 once the kill switch latched the registry off for the life of the engine',
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
    'dropped_hits': 'staged hits (Q > 0) dropped because the step did not admit the request (F2)',
    'program_growth': 'prefill rows whose program cache grew after warmup (F3: a compile after parking)',
    'capture_wrong_position': 'captures refused because the state was not taken after exactly pos tokens',
    'restores': 'checkpoints restored into the prefill scratch',
    'token_checks': 'token-id checks the trim ran (remembered per request and checkpoint)',
    'session_denied': 'hits denied to streaming-input sessions (their prompt holds decode-written tokens)',
    'capture_disabled': 'captures refused because the registry is latched off',
    'mid_loop_unplanned': 'boundaries below the loop drain left unplanned (the model had not declared mid-loop captures)',
    'reset_kept': 'reset_prefix_cache calls vLLM refused, so the registry was kept',
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
    if 'disabled' in values:
        # the registry holds the reason (a string) once latched, else None
        values['disabled'] = int(values['disabled'] is not None)
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
        requests = CounterMetricFamily(
            PREFIX + 'chat_requests', 'chat completions by the fields that decide whether the next turn\'s prompt '
            'extends this one (reasoning_effort, chat_template_kwargs) and by cache_salt (unset: no hit)',
            labels=REQUEST_LABELS)
        for labels, value in sorted(request_counts().items()):
            requests.add_metric(list(labels), value)
        yield requests
        salts = CounterMetricFamily(
            PREFIX + 'salts', 'requests by what the salt policy did with their cache_salt (dropped: served '
            'unsalted, so no hit)', labels=('verdict',))
        for verdict, value in sorted(salt_counts().items()):
            salts.add_metric([verdict], value)
        yield salts

    def describe(self):
        # Names vary with the registry's counters; an empty describe keeps registration from calling
        # collect() before an engine exists.
        return []


# -- request shapes that decide whether prompt N+1 extends prompt N (P0b, correction C6) -------------
CHAT_MODULE = 'vllm.entrypoints.openai.chat_completion.serving'
CHAT_CLASS = 'OpenAIServingChat'
REQUEST_LABELS = ('reasoning_effort', 'template_reasoning_effort', 'preserve_thinking', 'enable_thinking',
                  'cache_salt')
# vLLM 0.25.1 accepts these (chat_completion/protocol.py:228-229); the Qwen3.8 template renders only
# none/low/medium/xhigh and raises on the rest (P0b C7). Anything else is 'other': labels stay bounded.
EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
_REQUESTS = {}
_REQUESTS_LOCK = threading.Lock()


def label(value, allowed=()):
    if value is None:
        return 'unset'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str) and value in allowed:
        return value
    return 'other'


def request_labels(request):
    """The label values of one chat request: what the Qwen3.8 template keys its reasoning history on
    (preserve_thinking not true strips it; enable_thinking or reasoning_effort changing between turns
    breaks the prefix at token 3-9) and whether it carries a cache_salt (none: no hit, fail closed)."""
    kwargs = getattr(request, 'chat_template_kwargs', None)
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    return (label(getattr(request, 'reasoning_effort', None), EFFORTS),
            label(kwargs.get('reasoning_effort'), EFFORTS),
            label(kwargs.get('preserve_thinking')),
            label(kwargs.get('enable_thinking')),
            'set' if getattr(request, 'cache_salt', None) else 'unset')


def count_request(request):
    try:
        labels = request_labels(request)
    except Exception:
        labels = ('other',) * len(REQUEST_LABELS)
    with _REQUESTS_LOCK:
        _REQUESTS[labels] = _REQUESTS.get(labels, 0) + 1


def request_counts():
    with _REQUESTS_LOCK:
        return dict(_REQUESTS)


SALT_VERDICTS = ('verified', 'dropped-unverified', 'dropped-no-key', 'unset')
_SALTS = {}


def count_salt(verdict):
    """One request's salt verdict (serving_c2_contract.install_salt_policy); anything else is 'other'."""
    verdict = verdict if verdict in SALT_VERDICTS else 'other'
    with _REQUESTS_LOCK:
        _SALTS[verdict] = _SALTS.get(verdict, 0) + 1


def salt_counts():
    with _REQUESTS_LOCK:
        return dict(_SALTS)


def wrap_chat_serving(module):
    """Count every chat completion by request_labels before vLLM serves it; the count never raises
    into the request."""
    cls = getattr(module, CHAT_CLASS, None)
    original = getattr(cls, 'create_chat_completion', None)
    if original is None or getattr(original, '_qwen_prefix', False):
        return False
    import functools

    @functools.wraps(original)
    async def create_chat_completion(self, request, *args, **kwargs):
        try:
            count_request(request)
        except Exception:
            pass
        return await original(self, request, *args, **kwargs)

    create_chat_completion._qwen_prefix = True
    cls.create_chat_completion = create_chat_completion
    log('request shapes counted on %s.%s.create_chat_completion', module.__name__, CHAT_CLASS)
    return True


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


def guarded(name, wrap):
    """wrap(module), logging instead of raising: a wrap runs inside vLLM's own import of `name`, and an
    exception there would fail the API server's import of it."""
    def callback(module):
        try:
            return wrap(module)
        except Exception as error:
            log('metrics wrap of %s not installed: %s: %s', name, type(error).__name__, error)
            return False

    return callback


def install_collector(on_import, directory=None, modules=None):
    """In the API server: wrap vLLM's registry getter, and count chat requests by shape, now for a
    module already imported, else when it is. on_import(name, callback) runs callback(module) right
    after `name` executes (serving_c2_contract.PostImportHook). Each wrap is guarded."""
    modules = sys.modules if modules is None else modules
    for name, wrap in ((PROMETHEUS_MODULE, lambda loaded: wrap_prometheus_module(loaded, directory)),
                       (CHAT_MODULE, wrap_chat_serving)):
        callback = guarded(name, wrap)
        module = modules.get(name)
        if module is not None:
            callback(module)
        else:
            on_import(name, callback)
    return True
