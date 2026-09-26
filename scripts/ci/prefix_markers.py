"""What a general-prefix engine logs, parsed: the evidence the prefix-reuse gates judge (design 2.2).

The hit rate is read from the MODEL's own [PREFIX] rows (L - Q of each prefill row), never from
vllm:prefix_cache_hits: vLLM counts that inside get_computed_blocks, before the trim to a checkpoint
boundary, once per admission attempt (kv_cache_manager.py:238-244).

THE MARKER CONTRACT (what the harness expects each G1 track to print; a missing marker is reported,
never guessed around):

  scheduler graft (prefix_scheduler_graft.py, printed today):
    [PINDIAG] prefix: install scheduler=<module.Class> plugin=<path> coordinator=<Class> block_size=<n>
        blocks=<n> kv_spec_dtype=<dtype> QWEN_SDPA_BF8=<v> store_gib=<f> kill_switch=<path>
    [PINDIAG] prefix: grant req=<engine request id> h=<n> Q=<n> plan=[<pos>,...]
    [PINDIAG] prefix: commit refused req=<id> start_pos=<n> Q=<n>
    [PINDIAG] prefix: capture skipped req=<id> pos=<n>: <reason>
    [PINDIAG] prefix: kill switch <path> present: ...
  registry counters (the design's cross-process export; either form is read, the last one wins):
    [PINDIAG] prefix: stats {<json of PrefixRegistry.snapshot()>}      in the server log, or
    the JSON file STATS_FILE inside the container (read with docker exec)
  model graft (one line per prefill row, from inside the branch that ran):
    [PREFIX] req=<engine request id> Q=<n> L=<n> path=<traced|eager> restored_ms=<f> captured=[<pos>,...]
        capture_ms=<f> programs=<program-cache entries after the row>
    The design's own spelling (row=... ms=...) is accepted too: `row` is read as the request id
    when it is not a plain integer, `ms` as capture_ms.
  audit mode (QWEN_PREFIX_AUDIT=1, program-free):
    [PREFIX-AUDIT] req=<id> Q=<n> L=<n> kv_range=<a>:<b> kv_sha=<hex> slot_sha=<hex>
    digests of the unpacked K/V values over kv_range and of the row's GDN slot bytes; the gate
    compares a hit's digests with its cold twin's over the same range, so kv_range should be 0:L
    on every row (a hit's [0,Q) is the shared blocks, L2; its [Q,L) the resumed chunks and tail).

Other lines read: the serving contract's '[QWEN-C2] profile <name>: vLLM argv [...]' (the launched
argv, memory read-the-launched-argv), the TT platform's 'Automatic prefix caching is enabled' and
'Chunked prefill is not supported ... disabling it', the model's '[TP chunk-replay]' (the traced
chunk loop ran), '[PINDIAG] dram ...' lines (G2's reading), vLLM's 'GPU KV cache size: N tokens',
and failure signatures (a traceback, an engine death, tt-metal's ethernet-core wedge).

Log lines may carry docker's --timestamps prefix; it is split off and kept.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import json
import re

STATS_FILE = '/tmp/qwen-prefix-stats.json'
DOCKER_TIME = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z) ')
INSTALL = '[PINDIAG] prefix: install '
GRANT = re.compile(r'\[PINDIAG\] prefix: grant req=(\S+) h=(\d+) Q=(\d+) plan=\[([0-9, ]*)\]')
COMMIT_REFUSED = re.compile(r'\[PINDIAG\] prefix: commit refused req=(\S+) start_pos=(\d+) Q=(\d+)')
CAPTURE_SKIPPED = re.compile(r'\[PINDIAG\] prefix: capture skipped req=(\S+) pos=(\S+): (.*)$')
KILL_SWITCH = '[PINDIAG] prefix: kill switch '
STATS = re.compile(r'\[PINDIAG\] prefix: stats (\{.*\})\s*$')
MODEL_ROW = '[PREFIX] '
AUDIT_ROW = '[PREFIX-AUDIT] '
QWEN_C2_ARGV = re.compile(r'\[QWEN-C2\] profile (\S+): vLLM argv (\[.*\])[ \t]*$')
APC = re.compile(r'Automatic prefix caching is (enabled|disabled)')
CHUNKING_OFF = 'Chunked prefill is not supported for'
CHUNK_REPLAY = '[TP chunk-replay]'
DRAM = '[PINDIAG] dram'
KV_TOKENS = re.compile(r'GPU KV cache size: ([0-9,]+) tokens')
WEDGE = 'Timed out while waiting for active ethernet core'
TRACEBACK = 'Traceback (most recent call last)'
FAILURES = (TRACEBACK, 'EngineDeadError', 'EngineCore encountered a fatal error', 'PrefixInstallError',
            'AssertionError', WEDGE)
FIELD = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=(\[[^\]]*\]|\S+)')
ENGINE_SUFFIX = re.compile(r'^(.*)-([0-9A-Za-z]{8})$')


def split_timestamp(line):
    """(docker timestamp or None, the line without it)."""
    match = DOCKER_TIME.match(line)
    if match:
        return match.group(1), line[match.end():]
    return None, line


def _value(text):
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        return [_value(part.strip()) for part in inner.split(',') if part.strip()] if inner else []
    if re.match(r'^-?\d+$', text):
        return int(text)
    if re.match(r'^-?\d+\.\d*(e-?\d+)?$', text):
        return float(text)
    return text


def fields(text):
    """key=value tokens of a marker line: integers, floats and [a,b] lists parsed, the rest strings."""
    return dict((key, _value(value)) for key, value in FIELD.findall(text))


def request_tag(request_id):
    """The harness's tag from an engine request id. The API server names a request
    'chatcmpl-<X-Request-Id>' and the input processor appends '-<8 random characters>'
    (input_processor.py:223-240), so 'chatcmpl-pfx-a-3-hit-1a2b3c4d' is tag 'pfx-a-3-hit'."""
    text = str(request_id or '')
    if text.startswith('chatcmpl-'):
        text = text[len('chatcmpl-'):]
    match = ENGINE_SUFFIX.match(text)
    return match.group(1) if match else text


def model_row(text):
    """A [PREFIX] row's fields, normalised: req (from req, or from row when that is not a number),
    Q, L, path, restored_ms, captured, capture_ms, programs."""
    values = fields(text.split(MODEL_ROW, 1)[1])
    req = values.get('req')
    if req is None and isinstance(values.get('row'), str):
        req = values['row']
    captured = values.get('captured')
    if captured is not None and not isinstance(captured, list):
        captured = [captured]
    programs = values.get('programs', values.get('program_cache'))
    return dict(req=req, tag=request_tag(req) if req else None, q=values.get('Q'), l=values.get('L'),
                path=values.get('path'), restored_ms=values.get('restored_ms'), captured=captured or [],
                capture_ms=values.get('capture_ms', values.get('ms')), programs=programs,
                row=values.get('row'))


def audit_row(text):
    values = fields(text.split(AUDIT_ROW, 1)[1])
    req = values.get('req') or (values.get('row') if isinstance(values.get('row'), str) else None)
    return dict(req=req, tag=request_tag(req) if req else None, q=values.get('Q'), l=values.get('L'),
                kv_range=values.get('kv_range'), kv_sha=values.get('kv_sha'), slot_sha=values.get('slot_sha'))


def scan(lines):
    """Every marker in a server log (a list of lines, docker timestamps allowed), in order. Each
    entry keeps its line index and timestamp so the driver can window it against a request."""
    out = dict(installs=[], grants=[], rows=[], audits=[], refused=[], capture_skipped=[], kill_switch=[],
               stats=None, launches=[], apc=[], chunking_off=0, chunk_replay=0, dram=[], kv_tokens=None,
               failures=[])
    for index, raw in enumerate(lines):
        stamp, line = split_timestamp(raw.rstrip('\n'))
        where = dict(index=index, time=stamp)
        if INSTALL in line:
            entry = fields(line.split(INSTALL, 1)[1])
            entry.update(where)
            out['installs'].append(entry)
        match = GRANT.search(line)
        if match:
            plan = [int(part) for part in match.group(4).replace(' ', '').split(',') if part]
            out['grants'].append(dict(where, req=match.group(1), tag=request_tag(match.group(1)),
                                      h=int(match.group(2)), q=int(match.group(3)), plan=plan))
        if MODEL_ROW in line:
            entry = model_row(line)
            entry.update(where)
            out['rows'].append(entry)
        if AUDIT_ROW in line:
            entry = audit_row(line)
            entry.update(where)
            out['audits'].append(entry)
        match = COMMIT_REFUSED.search(line)
        if match:
            out['refused'].append(dict(where, req=match.group(1), tag=request_tag(match.group(1)),
                                       start_pos=int(match.group(2)), q=int(match.group(3))))
        match = CAPTURE_SKIPPED.search(line)
        if match:
            out['capture_skipped'].append(dict(where, req=match.group(1), tag=request_tag(match.group(1)),
                                               pos=match.group(2), reason=match.group(3)[:200]))
        if KILL_SWITCH in line:
            out['kill_switch'].append(dict(where, line=line.strip()[:300]))
        match = STATS.search(line)
        if match:
            try:
                out['stats'] = json.loads(match.group(1))
            except ValueError:
                pass
        match = QWEN_C2_ARGV.search(line)
        if match:
            try:
                argv = json.loads(match.group(2))
            except ValueError:
                argv = match.group(2)
            out['launches'].append(dict(where, profile=match.group(1), argv=argv))
        match = APC.search(line)
        if match:
            out['apc'].append(match.group(1))
        if CHUNKING_OFF in line:
            out['chunking_off'] += 1
        if CHUNK_REPLAY in line:
            out['chunk_replay'] += 1
        if DRAM in line:
            out['dram'].append(line.strip()[:400])
        match = KV_TOKENS.search(line)
        if match:
            out['kv_tokens'] = int(match.group(1).replace(',', ''))
        for signature in FAILURES:
            if signature in line:
                out['failures'].append(dict(where, signature=signature, line=line.strip()[:400]))
                break
    return out


def by_tag(entries):
    """Marker entries grouped by request tag (entries without one are left out)."""
    grouped = {}
    for entry in entries:
        if entry.get('tag'):
            grouped.setdefault(entry['tag'], []).append(entry)
    return grouped


def argv_flags(argv):
    """The launched argv's engine flags: {flag: value or True}, the last spelling winning."""
    flags, items = {}, list(argv or ())
    index = 0
    while index < len(items):
        token = str(items[index])
        if token.startswith('--'):
            name, separator, value = token[2:].partition('=')
            if separator:
                flags[name] = value
            elif index + 1 < len(items) and not str(items[index + 1]).startswith('--'):
                flags[name] = items[index + 1]
                index += 1
            else:
                flags[name] = True
        index += 1
    return flags


def prefix_argv_problems(argv):
    """What the launched argv of a general-prefix engine must say (design 2.0.1 item 5): the prefix
    cache on, chunked prefill on in the argv (vLLM's align-mode assertion) with the TT platform
    turning it back off - that half is read from the platform's own log line, not here - no
    'no-enable-prefix-caching', async scheduling off, block size 64."""
    if not isinstance(argv, list):
        return ['no launched argv line (the serving contract did not rewrite the API server argv)']
    flags = argv_flags(argv)
    problems = []
    if flags.get('enable-prefix-caching') is not True:
        problems.append('the launched argv does not enable the prefix cache (--enable-prefix-caching)')
    if 'no-enable-prefix-caching' in flags:
        problems.append('the launched argv still carries --no-enable-prefix-caching')
    if 'no-async-scheduling' not in flags:
        problems.append('the launched argv does not turn async scheduling off (--no-async-scheduling)')
    if str(flags.get('block-size')) != '64':
        problems.append('the launched argv block size is %r, not 64' % (flags.get('block-size'),))
    return problems


def parse_prometheus(text):
    """{metric name: summed value} over every label set; a counter's _total suffix is dropped, so
    'vllm:num_preemptions_total' reads as 'vllm:num_preemptions'."""
    values = {}
    for line in (text or '').splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        try:
            value = float(line.rsplit(' ', 1)[1])
        except (IndexError, ValueError):
            continue
        if name.endswith('_total'):
            name = name[:-len('_total')]
        values[name] = values.get(name, 0.0) + value
    return values
