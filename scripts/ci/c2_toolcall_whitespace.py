"""Tool-call argument whitespace (C2 serving, API server): keep what the model wrote between <parameter=NAME> and </parameter>.

The bug. vLLM 0.25.1's qwen3 / qwen3_xml / qwen3_coder tool parser (vllm/parser/qwen3.py, ParserEngine grammar
shared with the nemotron_v3 and seed_oss parsers) turns every <parameter=NAME>VALUE</parameter> into a JSON string
with ``value.strip()`` (qwen3.py:65 for a closed parameter, :74 for the one still open while streaming). Every
whitespace character at either end of every argument is lost, in both the non-streaming and the streaming chat path:
a file written with a trailing newline arrives without it, an edit whose old_string / new_string starts with
indentation arrives dedented, a value that is only spaces or newlines arrives empty. The client then executes the
corrupted call, and re-sends the corrupted call in the next turn, where the chat template re-renders it into tokens
the model never generated (a prefix-cache miss from that argument on, on top of the corrupted edit).

What the model wrote. The served chat template (the model's chat_template.jinja, assistant tool_calls branch) renders

    '<parameter=' + name + '>\\n' + value + '\\n</parameter>\\n'

for a string value (and tojson for any other), and the model emits exactly that form. The newline after the ``>`` and
the newline before ``</parameter>`` are framing; the characters between them are the value. The reference parsers
(vLLM 0.23 qwen3_coder: qwen3coder_tool_parser.py lines 148-151 and 455-456, 486-488) remove exactly one leading and
one trailing newline and nothing else. The 0.25.1 engine parser replaced that with strip().

The fix, here: ``unwrap(raw)`` removes ONE leading "\\n" and ONE trailing "\\n" from the text between the tags, and
nothing else. It is the exact inverse of the template for every string value (render, parse, render is the identity,
the empty string and "\\n" included), and it is what the 0.23 reference parser did. Stream-safe: while a parameter is
still open its last "\\n" may be the framing or the value, so the open value is shown one character short of it,
which makes every partial value a prefix of every later one and of the final one (the streaming path drops a delta
whose JSON is not an extension of what it already sent, ParserEngine._compute_arg_delta). Parameters that the stock
converter already returned unchanged (no whitespace at either end) are returned byte for byte as before.

How it is installed. A converter swap: ``vllm.parser.qwen3._qwen3_arg_converter`` is replaced when vLLM imports that
module (before qwen3_config() builds its first ParserEngineConfig, which reads the name at call time), by
serving_c2_contract.boot in the vLLM API server only (the parsers run there, not in the EngineCore), through an import
hook of this module. No pinned file is edited, nothing runs on the cards, no token the model generates changes:
only what the API server hands the client. DEFAULT ON: it is a pure correctness fix (the client now receives the
bytes the model generated; the only argument values that change are those that had whitespace at an end), so a
profile does not have to ask for it. Kill switches: the environment variable QWEN_C2_TOOLCALL_WS=0 (1 forces it on),
or "toolcall_whitespace": false in the profile; anything else is refused at boot.

install() refuses a module it does not recognise (the converter or its two patterns missing, a converter whose
behaviour on the probe below is neither the stock strip() nor already whitespace-preserving), so a vLLM bump cannot
silently serve a patch written for 0.25.1; an already whitespace-preserving converter is left alone (logged). Markers
(stderr, one line each): at arming, at install, the first time a tool argument is returned with edge whitespace kept
(graft mounted is not graft executed; lengths only, never text). Python 3.10 (the image), stdlib only.
"""

import json
import re
import sys

MODULE = 'vllm.parser.qwen3'
FUNCTION = '_qwen3_arg_converter'
CONFIG_FUNCTION = 'qwen3_config'
ENV_SWITCH = 'QWEN_C2_TOOLCALL_WS'
PROFILE_KEY = 'toolcall_whitespace'
INSTALLED = '_qwen_c2_toolcall_whitespace'
ARMED = 'toolcall whitespace armed'
DISABLED = 'toolcall whitespace disabled'
NOT_NEEDED = 'toolcall whitespace: not needed'

# vllm/parser/qwen3.py:50-56 (0.25.1), spelled for the stdlib re. install() uses the module's own compiled patterns, so
# the tokenisation is vLLM's; these are the defaults for a caller without vLLM (tests) and a pinned comparison.
PARAM_PATTERN = r'<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))'
PARTIAL_PARAM_PATTERN = r'<\s*parameter\s*=\s*([^>]+)>(.*)$'
PARAM_RE = re.compile(PARAM_PATTERN, re.DOTALL)
PARTIAL_PARAM_RE = re.compile(PARTIAL_PARAM_PATTERN, re.DOTALL)

# What the stock converter returns for PROBE_RAW (closed, partial=False): it strips. A whitespace-preserving converter
# returns the unwrapped value instead.
PROBE_RAW = '<parameter=a>\n  x \n</parameter>\n<parameter=b>\n\n</parameter>'
PROBE_STOCK = '{"a": "x", "b": ""}'
PROBE_PRESERVING = '{"a": "  x ", "b": ""}'

STATS = dict(installed=0, kept=0)
_LOGGED = set()


def log(message, *values):
    try:
        sys.stderr.write('[QWEN-C2] ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def _log_once(key, message, *values):
    if key not in _LOGGED:
        _LOGGED.add(key)
        log(message, *values)


def unwrap(raw):
    """The value between the tags: ONE leading and ONE trailing newline (the template's framing) removed, nothing else.
    The inverse of '\\n' + value + '\\n' for every string; for an open (still streaming) value it is also one character
    short of a trailing newline, which is exactly what keeps every partial value a prefix of the final one."""
    if raw.startswith('\n'):
        raw = raw[1:]
    if raw.endswith('\n'):
        raw = raw[:-1]
    return raw


def make_converter(param_re=PARAM_RE, partial_param_re=PARTIAL_PARAM_RE, on_kept=None):
    """The replacement for vllm.parser.qwen3._qwen3_arg_converter(raw_args, partial) -> JSON text: the stock structure
    (every closed parameter, then the one still open when partial), with unwrap() where the stock calls strip()."""

    def _qwen3_arg_converter(raw_args, partial):
        """Whitespace-preserving twin of vLLM 0.25.1's converter (c2_toolcall_whitespace)."""
        params = {}

        for match in param_re.finditer(raw_args):
            name = match.group(1)
            value = unwrap(match.group(2))
            params[name] = value
            if on_kept is not None and not partial and value != value.strip():
                on_kept(value)

        if partial:
            remaining = param_re.sub('', raw_args)
            m = partial_param_re.search(remaining)
            if m:
                name = m.group(1)
                value = m.group(2)
                if name:
                    params[name] = unwrap(value)

        return json.dumps(params, ensure_ascii=False)

    setattr(_qwen3_arg_converter, INSTALLED, True)
    return _qwen3_arg_converter


def kept(value):
    """The live marker: the first closed argument returned with whitespace at an end (lengths only, never the text)."""
    STATS['kept'] += 1
    stripped = value.strip()
    _log_once('live', 'toolcall whitespace: live: first tool argument returned with edge whitespace kept '
                      '(%d characters, %d after strip())', len(value), len(stripped))


def probe(converter):
    """'stock' (strips), 'preserving' (already keeps the edges), or None (neither: not a converter we know)."""
    try:
        found = converter(PROBE_RAW, False)
    except Exception:
        return None
    if found == PROBE_STOCK:
        return 'stock'
    if found == PROBE_PRESERVING:
        return 'preserving'
    return None


def install(module, emit=None):
    """Replace module._qwen3_arg_converter with the whitespace-preserving one. Refuses (raises) a module that is not
    vLLM 0.25.1's qwen3 parser: a boot that asked for the fix must not serve without it. Idempotent."""
    emit = log if emit is None else emit
    original = getattr(module, FUNCTION, None)
    if not callable(original):
        raise RuntimeError('%s has no %s: the toolcall whitespace fix cannot be installed' % (module.__name__, FUNCTION))
    if getattr(original, INSTALLED, False):
        return original
    param_re, partial_param_re = getattr(module, '_PARAM_RE', None), getattr(module, '_PARTIAL_PARAM_RE', None)
    for label, compiled, expected in (('_PARAM_RE', param_re, PARAM_PATTERN),
                                      ('_PARTIAL_PARAM_RE', partial_param_re, PARTIAL_PARAM_PATTERN)):
        if compiled is None or not hasattr(compiled, 'finditer') or getattr(compiled, 'pattern', None) != expected:
            raise RuntimeError('%s.%s is not the vLLM 0.25.1 pattern: the toolcall whitespace fix cannot be installed'
                               % (module.__name__, label))
    kind = probe(original)
    if kind == 'preserving':
        emit('%s: %s.%s already keeps edge whitespace (a newer vLLM?): left as it is' % (NOT_NEEDED, module.__name__, FUNCTION))
        return original
    if kind != 'stock':
        raise RuntimeError('%s.%s does not behave as the vLLM 0.25.1 converter (the probe returned neither the stripped '
                           'nor the preserved form): the toolcall whitespace fix cannot be installed'
                           % (module.__name__, FUNCTION))
    replacement = make_converter(param_re, partial_param_re, on_kept=kept)
    replacement.__wrapped__ = original
    setattr(module, FUNCTION, replacement)
    config = getattr(module, CONFIG_FUNCTION, None)
    if hasattr(config, 'cache_clear'):
        config.cache_clear()      # a config built before the swap holds the stock converter
    if callable(config) and getattr(config(), 'arg_converter', replacement) is not replacement:
        uninstall(module)
        raise RuntimeError('%s.%s builds a config that does not use %s: the toolcall whitespace fix cannot be installed'
                           % (module.__name__, CONFIG_FUNCTION, FUNCTION))
    STATS['installed'] += 1
    emit('toolcall whitespace: installed on %s.%s: a tool argument keeps its edge whitespace (one framing newline at '
         'each end removed, nothing else)' % (module.__name__, FUNCTION))
    return replacement


def uninstall(module):
    """Put the stock converter back (tests)."""
    current = getattr(module, FUNCTION, None)
    original = getattr(current, '__wrapped__', None)
    if original is not None and getattr(current, INSTALLED, False):
        setattr(module, FUNCTION, original)
        config = getattr(module, CONFIG_FUNCTION, None)
        if hasattr(config, 'cache_clear'):
            config.cache_clear()
    return getattr(module, FUNCTION, None)


def enabled(profile, environ):
    """Whether this boot installs the fix: the environment switch if set ('0' off, '1' on), else the profile key if
    present (true / false), else ON (a pure correctness fix: see the module docstring). Anything else is refused."""
    value = environ.get(ENV_SWITCH)
    if value is not None and value != '':
        if value not in ('0', '1'):
            raise ValueError('%s must be 0 or 1, got %r' % (ENV_SWITCH, value))
        return value == '1'
    if PROFILE_KEY in profile:
        value = profile[PROFILE_KEY]
        if value not in (True, False):
            raise ValueError('%s must be true or false, got %r' % (PROFILE_KEY, value))
        return bool(value)
    return True


class ImportHook(object):
    """Run install() on vllm.parser.qwen3 right after it executes, without importing it early. (Not the contract's
    PostImportHook: this one is armed for every profile, and the contract's hooks are the profile-gated ones.)"""

    def __init__(self, name, callback):
        self.name, self.callback = name, callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader, callback = spec.loader, self.callback
        execute = loader.exec_module

        def exec_module(module):
            execute(module)
            callback(module)

        loader.exec_module = exec_module
        return spec


def arm(profile, environ, emit=None):
    """Called by serving_c2_contract.boot in the vLLM API server: install the fix when vLLM imports its qwen3 parser
    module (at once when it already has). Returns True when armed. The module is imported by the boot, so an image
    without it fails the API server's boot instead of serving the bug."""
    emit = log if emit is None else emit
    name = profile.get('name', '?')
    if not enabled(profile, environ):
        emit('profile %s: %s (%s=0 or "%s": false in the profile): tool-call argument whitespace is stripped as vLLM does',
             name, DISABLED, ENV_SWITCH, PROFILE_KEY)
        return False
    present = sys.modules.get(MODULE)
    if present is not None:
        install(present, lambda line: emit('%s', line))
    else:
        sys.meta_path.insert(0, ImportHook(MODULE, lambda module: install(module, lambda line: emit('%s', line))))
    emit('profile %s: %s: %s.%s keeps the edge whitespace of tool-call arguments (default on: a pure correctness fix; '
         '%s=0 turns it off)', name, ARMED, MODULE, FUNCTION, ENV_SWITCH)
    return True
