"""The trace-capture rules of the device, as a wrapper any fake ttnn can be put behind (TEST SUPPORT ONLY: no served module imports it and it is not in the image).

Two audits of the op-fusion programme died on a card at engine start because the capture they ran inside broke a rule of the device that no CPU test could see:
the first synchronized inside the fused-commit capture, the second launched an audit program that had never run eagerly ("Cannot load new binaries during trace
capture"). This module is the CPU model of exactly those rules. `guard(operations)` returns an object that behaves as `operations` (every constant, descriptor
class and op is forwarded) and, while a capture is open (between begin_trace_capture and end_trace_capture), enforces:

  (a) no host round trip: synchronize_device, to_torch, from_torch (a write to a device tensor), copy_host_to_device_tensor and the event calls raise
      CaptureViolation ("Event Synchronization is not supported during trace capture");
  (b) no new binary: an op or program launch whose key (op name, input shapes / dtypes / layouts, configs and, for generic_op, the kernels' sources, compile-time
      arguments, defines, circular buffers and core ranges - NEVER the runtime arguments or buffer addresses, which a cached program overrides) was not executed
      eagerly before raises CaptureViolation ("Cannot load new binaries during trace capture").

Every violation is also recorded (rules.violations), so a lever that catches the exception and carries on is still found. `rules.warm` is the set of keys executed
eagerly so far (by every wrapped object that shares the rules); a capture is allowed to use only those. The stand-in is a model, not the device: it knows nothing a
card knows beyond these two rules (no allocator, no CB overflow, no NoC), and a key that is too coarse makes it lenient, one that is too fine makes it strict, so the
key errs on the device's side: shapes, dtypes, layouts, configs and kernel text count; values and addresses do not.
"""

from hashlib import sha1
from types import SimpleNamespace

HOST_ROUND_TRIPS = frozenset((
    'synchronize_device', 'to_torch', 'from_torch', 'copy_host_to_device_tensor', 'copy_device_to_host_tensor', 'from_device', 'to_device', 'event_synchronize',
    'record_event', 'wait_for_event', 'read_buffer', 'write_buffer', 'synchronize', 'to_host'))

# Calls that are not program launches: bookkeeping on handles and views, the capture calls themselves, frees. Anything else lower-case that is handed a device tensor
# or a program is a launch and is keyed. Classes and descriptor constructors (capitalised names) are never launches.
NOT_PROGRAMS = frozenset((
    'begin_trace_capture', 'end_trace_capture', 'release_trace', 'execute_trace', 'begin_graph_capture', 'end_graph_capture', 'deallocate', 'get_device_tensors',
    'get_memory_view', 'memory_config', 'buffer_address', 'num_banks', 'is_tensor_storage_on_device', 'num_devices', 'get_num_devices', 'as_tensor',
    'create_mesh_device', 'open_mesh_device', 'close_mesh_device', 'set_printoptions', 'empty', 'empty_like', 'allocate_tensor_on_device'))

NEW_BINARIES = 'Cannot load new binaries during trace capture'
SYNCHRONIZATION = 'TT_FATAL fd_mesh_command_queue.cpp:1042 !trace_id_.has_value(): Event Synchronization is not supported during trace capture'

SKIP_ATTRIBUTES = frozenset(('runtime_args', 'common_runtime_args'))


class CaptureViolation(RuntimeError):
    """What the device raises (the message carries the device's own words so a lever that classifies a refusal by text sees the real thing)."""


def _is_tensor(value):
    return hasattr(value, 'shape') and (hasattr(value, 'shards') or hasattr(value, 'dtype'))


def signature(value, depth=0):
    """A hashable structural key of an op argument. Tensors by shape, dtype, layout and memory config; descriptors and configs by their public attributes (without the
    runtime arguments); sequences and mappings elementwise; text by its hash; scalars by value."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 64 else 'text:' + sha1(value.encode('utf-8', 'replace')).hexdigest()[:16]
    if isinstance(value, bytes):
        return 'bytes:' + sha1(value).hexdigest()[:16]
    if _is_tensor(value):
        try:
            memory = value.memory_config()
        except Exception:  # noqa: BLE001 - a fake without a memory config
            memory = None
        return ('tensor', tuple(int(size) for size in value.shape), repr(getattr(value, 'dtype', None)), repr(getattr(value, 'layout', None)), repr(memory))
    if depth > 6:
        return type(value).__name__
    if isinstance(value, dict):
        return ('dict', tuple(sorted(((repr(key), signature(item, depth + 1)) for key, item in value.items() if key not in SKIP_ATTRIBUTES), key=repr)))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [signature(item, depth + 1) for item in value]
        return (type(value).__name__, tuple(sorted(items, key=repr)) if isinstance(value, (set, frozenset)) else tuple(items))
    if hasattr(value, 'dtype') and hasattr(value, 'numel'):          # a host torch tensor: shape and dtype
        return ('host', tuple(value.shape), str(value.dtype))
    attributes = getattr(value, '__dict__', None)
    if attributes is not None and not callable(value):
        return (type(value).__name__, tuple(sorted(((name, signature(item, depth + 1)) for name, item in attributes.items()
                                                    if not name.startswith('_') and name not in SKIP_ATTRIBUTES), key=lambda pair: pair[0])))
    if callable(value):
        return 'callable:' + getattr(value, '__name__', type(value).__name__)
    return repr(value)


# Keyword arguments that are runtime handles of a collective (semaphores cycle on every call and are overridden in a cached program), not part of its key.
SKIP_OPTIONS = frozenset(('multi_device_global_semaphore', 'barrier_semaphore', 'semaphore', 'semaphores'))


def key_of(name, args, kwargs):
    return (name, tuple(signature(item) for item in args),
            tuple(sorted((option, signature(item)) for option, item in kwargs.items() if option not in SKIP_OPTIONS)))


def describe(key):
    """A short human description of a key (the op and its tensor shapes / kernel names) for a violation message."""
    name, args, options = key
    shapes = [item[1] for item in list(args) + [value for _name, value in options] if isinstance(item, tuple) and item and item[0] == 'tensor']
    return '%s%s' % (name, ' shapes=%s' % (shapes[:4],) if shapes else '')


class CaptureRules:
    """The state the rules share: whether a capture is open, the warm keys, every violation and every capture."""

    def __init__(self):
        self.depth = 0
        self.calling_begin = self.calling_end = False
        self.warm = set()
        self.violations = []
        self.attempts = []                      # every host round trip refused, (name,)
        self.captures = 0
        self.captured = []                      # per capture, the keys it used
        self.eager_launches = 0
        self.captured_launches = 0
        self.log = []

    @property
    def open(self):
        return self.depth > 0

    def begin(self):
        if self.depth:
            raise CaptureViolation('a trace capture is already open')
        self.depth += 1
        self.captures += 1
        self.captured.append([])

    def end(self):
        self.depth = max(0, self.depth - 1)

    def round_trip(self, name):
        if self.open:
            self.attempts.append(name)
            self.violations.append('%s inside a trace capture' % name)
            raise CaptureViolation('%s (%s)' % (SYNCHRONIZATION, name))

    def launch(self, name, args, kwargs):
        key = key_of(name, args, kwargs)
        if not self.open:
            self.warm.add(key)
            self.eager_launches += 1
            return key
        self.captured_launches += 1
        self.captured[-1].append(key)
        if key not in self.warm:
            self.violations.append('%s: %s' % (NEW_BINARIES, describe(key)))
            raise CaptureViolation('%s (%s)' % (NEW_BINARIES, describe(key)))
        return key

    def problems(self):
        return list(self.violations)

    def assert_clean(self):
        if self.violations:
            raise AssertionError('%d trace-capture rule violation(s):\n  %s' % (len(self.violations), '\n  '.join(self.violations[:12])))


class GuardedNamespace:
    """A wrapped view of `target` (the ttnn module or one of its namespaces) under shared rules."""

    def __init__(self, target, rules, prefix=''):
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_rules', rules)
        object.__setattr__(self, '_prefix', prefix)
        object.__setattr__(self, '_cache', {})

    def __getattr__(self, name):
        target, rules = self._target, self._rules
        value = getattr(target, name)
        cache = self._cache
        if name in cache and cache[name][0] is value:
            return cache[name][1]
        wrapped = self._wrap(name, value)
        cache[name] = (value, wrapped)
        return wrapped

    def __setattr__(self, name, value):
        # A test or a lever installing an attribute (a twin, a hook) sets it on the real target, as it would on ttnn.
        setattr(self._target, name, value)
        self._cache.pop(name, None)

    def __dir__(self):
        return dir(self._target)

    def _wrap(self, name, value):
        rules, qualified = self._rules, self._prefix + name
        if isinstance(value, SimpleNamespace) and not callable(value):
            return GuardedNamespace(value, rules, qualified + '.')
        if not callable(value) or isinstance(value, type) or name[:1].isupper():
            return value
        if name == 'begin_trace_capture':
            def begin(*args, **kwargs):
                if rules.calling_begin:             # a lever's own wrapper of the capture call (draft_fusion_tp.track) reaching the wrapper below it: one capture, one begin
                    return value(*args, **kwargs)
                rules.calling_begin = True
                try:
                    result = value(*args, **kwargs)
                finally:
                    rules.calling_begin = False
                rules.begin()
                return result
            return begin
        if name == 'end_trace_capture':
            def end(*args, **kwargs):
                if rules.calling_end:
                    return value(*args, **kwargs)
                rules.calling_end = True
                try:
                    return value(*args, **kwargs)
                finally:
                    rules.calling_end = False
                    rules.end()
            return end
        if name in HOST_ROUND_TRIPS:
            def round_trip(*args, **kwargs):
                rules.round_trip(name)
                return value(*args, **kwargs)
            return round_trip
        if name in NOT_PROGRAMS or name.startswith('_'):
            return value

        def launch(*args, **kwargs):
            rules.launch(qualified, args, kwargs)
            return value(*args, **kwargs)
        return launch


def guard(operations, rules=None):
    """`operations` behind the capture rules; the rules are `result.rules`. Pass the same `rules` to several objects (the ttnn fake and a mesh fake) to share the warm set."""
    rules = rules or CaptureRules()
    wrapped = GuardedNamespace(operations, rules)
    object.__setattr__(wrapped, 'rules', rules)
    return wrapped


def mesh_guard(mesh, rules):
    """A mesh device whose host-visible synchronization methods are refused inside a capture (the object the levers pass to synchronize_device)."""
    return GuardedNamespace(mesh, rules, 'mesh.')
