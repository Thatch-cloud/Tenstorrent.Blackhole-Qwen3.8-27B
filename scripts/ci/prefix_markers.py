"""What a general-prefix engine logs, parsed: the evidence the prefix-reuse gates judge (design 2.2).

The hit rate is read from the MODEL's own [PREFIX] rows (L - Q of each prefill row), never from
vllm:prefix_cache_hits: vLLM counts that inside get_computed_blocks, before the trim to a checkpoint
boundary, once per admission attempt (kv_cache_manager.py:238-244).

THE MARKER CONTRACT (what the harness expects each G1 track to print; a missing marker is reported,
never guessed around):

  scheduler graft (qwen_prefix_scheduler_patch.py over qwen_prefix_registry.py):
    [PINDIAG] prefix: install scheduler=<module.Class> plugin=<path> coordinator=<Class> block_size=<n>
        blocks=<n> kv_spec_dtype=<dtype> QWEN_SDPA_BF8=<v> hash=<algo> executor=<backend> store_gib=<f>
        kill_switch=<path> stats=<path> vllm=<version>
    [PINDIAG] prefix: grant req=<engine request id> h=<n> Q=<n> drain=<n> plan=[<pos>,...]
        (drain: where the row's chunk loop drains, floor2048 of the tokens it prefills; the P0a
        prototype printed no drain= and is still read)
    [PINDIAG] prefix: commit refused req=<id> start_pos=<n> Q=<n>
    [PINDIAG] prefix: capture skipped req=<id> pos=<n>: <reason>
    [PINDIAG] prefix: kill switch <path> present: ...
  registry counters (the design's cross-process export; either form is read, the last one wins):
    [PINDIAG] prefix: stats {<json of PrefixRegistry.snapshot()>}      in the server log, or
    the JSON file STATS_FILE inside the container (read with docker exec)
    The lifecycle gates read REQUIRED_STATS from it, `dropped_hits` (a staged grant with Q > 0
    that commit dropped: an allocation failure after a grant, F2) included.
  model graft (qwen_prefix_model_patch; one line per prefill row, from inside the branch that ran):
    [PREFIX] row=<index> req=<engine request id> path=<traced|eager> registry=<present|absent>
        grant=<committed|none> Q=<n> L=<n> plan=[<pos>,...] restored_ms=<f or -> captured=[<pos>:stored:<n>ms |
        <pos>:refused:<n>ms | <pos>:skipped, ...] dropped=[<pos>,...] ms=<row ms> programs=<before>-><after>
        programs_across_restore=<...> [slot_sha=<hex> logits_sha=<hex>]
    read as: captured = the stored positions (refused and skipped ones go to capture_failed), capture_ms
    = the sum of the stored captures' ms, programs_before / programs = the two sides of the arrow ('None'
    when the program cache size is unavailable). slot_sha and logits_sha are printed only with
    QWEN_PREFIX_DIGESTS=1 (or QWEN_PREFIX_AUDIT=1), a gate instrument the gate passes to every prefix arm
    but timing. The harness's own spelling (req=... captured=[<pos>,...] capture_ms=<f>
    programs_before=<n> programs=<n>) is read too. Other '[PREFIX] ' lines (program growth, capture
    skipped, no registry) are not rows.
    programs_before lets the gate judge what the row itself compiled (a decode step between two rows
    may compile on its own); without it only a row right after its cold twin is measured.
    slot_sha digests the row's end-of-prefill GDN state (the bytes the row already copies to the
    host) and logits_sha the last prompt position's logits row (on the host under general, which
    samples there): every cold/hit pair in every arm compares both byte for byte (design 2.0.4).
    A request vLLM preempts and resumes prints a second row for the same request id: its L is the
    prompt plus the output so far (a resumed request re-prefills both). The first row is the
    admission the oracle judges; the admissions count names the preempted requests.
    The design's own spelling (row=... ms=...) is accepted too: `row` is read as the request id
    when it is not a plain integer, `ms` as capture_ms.
  model graft, G2's DRAM reading (each chip's DRAM allocator figures, each point once per engine):
    [PINDIAG] dram after registry: chip0 allocated=<f>GB free=<f>GB largest_free=<f>MB of <f>GB; chip1 ...
        when the prefix route first runs with the scheduler's registry present: the model is warm and
        the registry exists. The bring-up requires it (per-chip figures, not 'unavailable (<reason>)').
    [PINDIAG] dram after first capture: <the same>
        after the first row that stored a checkpoint; the bring-up requires it once the arm stored one.
    read into dram_readings: point, chips (chip, allocated_gb, free_gb, largest_free_mb, total_gb),
    unavailable (the reason, or None) and text. The fast path's 'dram after attach' and 'dram after
    engine <id>' lines carry the same text (serving_buffer_pool.format_dram) and parse the same way.
  model graft, the eager warm (trace_mode decode_only; qwen36_vllm's _qwen_prefix_warm_eager):
    [PINDIAG] prefix: eager prefill warmed before the decode trace: page_table_blocks=<n> programs=<a>-><b>
        once per engine, before the decode trace is parked; read into eager_warm (its fields). The
        eager exactness arm requires it: an image without it compiles the eager prefill after the
        traces are parked, and G1 v47's (run 36246961161) next prefill hung the device.
  audit mode (QWEN_PREFIX_AUDIT=1, program-free):
    [PREFIX-AUDIT] req=<id> Q=<n> L=<n> kv_range=<a>:<b> kv_sha=<hex> slot_sha=<hex> [logits_sha=<hex>]
    digests of the unpacked K/V values over kv_range and of the row's GDN slot bytes (the model also
    prints one line per 2048-token window, window=<w> ... kv=<hex>, which is not an audit row); the gate
    compares a hit's digests with its cold twin's over the same range, so kv_range should be 0:L
    on every row (a hit's [0,Q) is the shared blocks, L2; its [Q,L) the resumed chunks and tail).

  model graft, the warmup (qwen_prefix_model_patch MARKER_WARM, once per engine, before vLLM builds the scheduler):
    [PINDIAG] prefix: model warm restore_mode=<h2d|copy> results={...} gdn_layers=<n> ... programs=<n>
        the restore path round-tripped and the mid-loop captures declared; read into model_warm (its fields).
        The fast path's sticky sessions require it: without it the first resume stops the engine and the
        first turn after boot plans no C0 capture.
    [PINDIAG] prefix: model warm skipped - not the batched TP path (...); a resumed row will assert
        the warm did NOT run (read into model_warm_skipped, never model_warm): a gate that needs the warm fails.
  audit mode, per window (QWEN_PREFIX_AUDIT=1, the rows' window lines):
    [PREFIX-AUDIT] req=<id> Q=<n> L=<n> window=<w> tokens=[<a>,<b>) new=<0|1> kv=<hex>
        read into audit_windows (tag, q, l, window, start, end, new, kv): the KV bytes of each 2048-token
        window; new=0 marks a window a hit restored rather than wrote (prefix_judge.window_findings).
  the fast path's sticky sessions (QWEN_FAST_STICKY_SESSIONS=1):
    [PINDIAG] prefix: install sticky=1 lookahead=<n> drop_last=<bool> ceiling=floor2048(P-2048)
        the scheduler graft's second install line; read into sticky_installs, never installs.
    [PINDIAG] prefix: install chunked=levern: chunked prefill beside the Lever N cap (max_num_scheduled_tokens=<n>)
        Lever N's install line (QWEN_FAST_LEVER_N=1 beside the prefix graft), written AFTER the other two; read into chunked_installs (chunked
        'levern', max_num_scheduled_tokens), never installs: only a body that begins `scheduler=` is the install the gate reads its class, block
        size and KV dtype from. The gate takes chunked_installs to mean the request's prefill is Lever N's steps, not one.
    [PINDIAG] sticky admit req='<engine request id>' Q=<n> P=<n> tail=<n>
        serving_lifecycle, once per request admitted at a granted boundary; read into sticky_admits. tail is the FIRST scheduled chunk
        (`chunk` in the lifecycle): the whole rest of the prompt, P - Q, without Lever N, and under Lever N's chunked prefill the first
        step of the plan (2,048 tokens while seats decode, e.g. P=6197 Q=2048 tail=2048 and a continuation start=4096 tokens=2101).
    [PINDIAG] sticky engine built req=<engine request id, first 48 characters> ms=<f> frontier=<R, 0 cold> [kind=<build|rebind>: engine reuse only]
        prompt=<P> [kind=<build|rebind>]   serving_runtime.STICKY_ENGINE_MARKER, once per admitted request's engine build; read
        into sticky_builds (the TTFT split: the tail prefill, then this build). Under engine reuse (QWEN_FAST_PARKED_ENGINES=1) every
        request logs it and `kind` says whether its engine was built or a parked one rebound (the A8 rule reads the line either way).
    [PINDIAG] verify t2 kv shared ...   verify_trace_t2.KV_SHARED, a packed round's K/V conflict (the proposal
        guard's or the stage's); read into kv_shared, which a fast-path prefix arm needs empty.

  the host KV tier and the preconverted checkpoints (QWEN_PREFIX_HOST_TIER_GIB, QWEN_PREFIX_CKPT_PRECONVERTED; docs/prefix-store-hygiene.md):
    [PINDIAG] prefix: host tier on gib=<f> kv_gib=<f> block_bytes=<n> verify=<sample|all|off> audit=<0|1> spill_max_blocks=<n> min_tokens=<n> slack=<n> off_path=<path>
        the scheduler graft, once, when the tier is configured and the model graft's IO is attached; read into tier_on.
    [PINDIAG] prefix: host tier IO attached block_bytes=<n> tensors=<n> chips=<n> slice_bytes=<n> fingerprint=<s>
        the model graft, once, after the warm; read into tier_io.
    [PINDIAG] prefix: host tier spill blocks=<n> bytes=<n> ms=<f> held=<n>          one per flush; read into tier_spills.
    [PINDIAG] prefix: host tier restore req=<id> blocks=<n> bytes=<n> ms=<f> from=<n> to=<n>
        one per restored request (the blocks written to the device before vLLM's own hit logic ran); read into tier_restores.
    [PINDIAG] prefix: host tier latched off: ... | host tier spill failed ... | host tier restore failed ...    read into tier_trouble.
    [PREFIX-AUDIT-CKPT] tensors=<n> conversion_equal=<0|1> readback_equal=<0|1> differing=[...]
        QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT=1, after a restore from preconverted tensors; read into ckpt_audits.
    [PINDIAG] prefix: checkpoint preconverted req=<id> pos=<n> tensors=<n> host_bytes=<n> convert_ms=<f>      one per capture stored preconverted; counted into ckpt_preconverted.

Other lines read: the serving contract's '[QWEN-C2] profile <name>: vLLM argv [...]' (the launched
argv, memory read-the-launched-argv), the TT platform's 'Automatic prefix caching is enabled' and
'Chunked prefill is not supported ... disabling it', the model's '[TP chunk-replay]' (the traced
chunk loop ran), every '[PINDIAG] dram ...' line (raw, beside dram_readings), vLLM's 'GPU KV cache size: N tokens',
and failure signatures (a traceback, an engine death, tt-metal's ethernet-core wedge, UMD's MMIO
per-op timeout - the host's read of a hung device, G1 v47's eager arm).

Log lines may carry docker's --timestamps prefix; it is split off and kept.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import json
import re

STATS_FILE = '/tmp/qwen-prefix-stats.json'
REQUIRED_STATS = ('pins', 'commit_mismatch', 'dropped_attempts', 'dropped_hits', 'same_step_rejects',
                  'evicted_coupled', 'evicted_lru', 'unsalted_denied')
DOCKER_TIME = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z) ')
INSTALL = '[PINDIAG] prefix: install '
# The scheduler graft's sticky-session install line begins 'install sticky=' (qwen_prefix_scheduler_patch).
STICKY_INSTALL = 'sticky='
# The install line the gate reads (class, block size, KV dtype, store) begins 'install scheduler='; Lever N's begins 'install chunked=levern: ...'
# and is written after the sticky one, so it is its own list (chunked_installs): it carries none of those fields.
SCHEDULER_INSTALL = 'scheduler='
CHUNKED_INSTALL = 'chunked='
CHUNKED_CAP = re.compile(r'max_num_scheduled_tokens=(\d+)')
STICKY_ADMIT = re.compile(r"\[PINDIAG\] sticky admit req=(?:'([^']*)'|\"([^\"]*)\"|(\S+)) Q=(\d+) P=(\d+) tail=(\d+)")
STICKY_BUILT = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([0-9.]+) frontier=(\d+) prompt=(\d+)(?: kind=(build|rebind))?')
# qwen_prefix_model_patch.MARKER_WARM: the warm that chose a restore path. Its skip line (off the batched TP path,
# 'model warm skipped - ... a resumed row will assert') is MODEL_WARM_SKIPPED, never a warm.
MODEL_WARM = '[PINDIAG] prefix: model warm restore_mode='
MODEL_WARM_SKIPPED = '[PINDIAG] prefix: model warm skipped'
KV_SHARED = '[PINDIAG] verify t2 kv shared'    # verify_trace_t2.KV_SHARED
AUDIT_WINDOW = re.compile(r'\[PREFIX-AUDIT\] req=(\S+) Q=(\d+) L=(\d+) window=(\d+) tokens=\[(\d+),(\d+)\) '
                          r'new=([01]) kv=(\S+)')
GRANT = re.compile(r'\[PINDIAG\] prefix: grant req=(\S+) h=(\d+) Q=(\d+)(?: drain=(\d+))? plan=\[([0-9, ]*)\]')
COMMIT_REFUSED = re.compile(r'\[PINDIAG\] prefix: commit refused req=(\S+) start_pos=(\d+) Q=(\d+)')
CAPTURE_SKIPPED = re.compile(r'\[PINDIAG\] prefix: capture skipped req=(\S+) pos=(\S+): (.*)$')
KILL_SWITCH = '[PINDIAG] prefix: kill switch '
TIER_ON = '[PINDIAG] prefix: host tier on '
TIER_IO = '[PINDIAG] prefix: host tier IO attached '
TIER_SPILL = '[PINDIAG] prefix: host tier spill blocks='
TIER_RESTORE = '[PINDIAG] prefix: host tier restore req='
TIER_TROUBLE = ('[PINDIAG] prefix: host tier latched off', '[PINDIAG] prefix: host tier spill failed', '[PINDIAG] prefix: host tier restore failed')
CKPT_AUDIT = '[PREFIX-AUDIT-CKPT] tensors='
CKPT_PRECONVERTED = '[PINDIAG] prefix: checkpoint preconverted '
STATS = re.compile(r'\[PINDIAG\] prefix: stats (\{.*\})\s*$')
MODEL_ROW = '[PREFIX] '
# A row, not the model's other [PREFIX] lines (program growth, capture skipped, no registry).
MODEL_ROW_LINE = re.compile(r'\[PREFIX\] (?:row|req)=')
PROGRAMS_ARROW = re.compile(r'^(\d+|None)->(\d+|None)$')
AUDIT_ROW = '[PREFIX-AUDIT] '
# One per audited step: rows= reqs= tokens= mode=<region|full> reads= blocks_read= read_ms= total_ms= programs=<a>-><b> [fallback='...'].
AUDIT_COST = '[PREFIX-AUDIT-COST] rows='
# One per cross-checked audited step (QWEN_PREFIX_AUDIT_READ=cross): tensors= mismatched= region_ms= whole_read_ms= blocks_read= [fallback='...'].
AUDIT_CROSS = '[PREFIX-AUDIT-CROSS] tensors='
QWEN_C2_ARGV = re.compile(r'\[QWEN-C2\] profile (\S+): vLLM argv (\[.*\])[ \t]*$')
APC = re.compile(r'Automatic prefix caching is (enabled|disabled)')
CHUNKING_OFF = 'Chunked prefill is not supported for'
CHUNK_REPLAY = '[TP chunk-replay]'
DRAM = '[PINDIAG] dram'
# '[PINDIAG] dram after <point>: <reading>' (qwen_prefix_model_patch.MARKER_DRAM; the fast path's lines too).
DRAM_READING = re.compile(r'\[PINDIAG\] dram after ([^:]+?): (.*?)\s*$')
DRAM_CHIP = re.compile(r'chip(\d+) allocated=([0-9.]+)GB free=([0-9.]+)GB largest_free=([0-9.]+)MB of ([0-9.]+)GB')
DRAM_UNAVAILABLE = re.compile(r'^unavailable \((.*)\)$')
DRAM_REGISTRY = 'registry'              # qwen_prefix_model_patch.DRAM_REGISTRY
DRAM_FIRST_CAPTURE = 'first capture'    # qwen_prefix_model_patch.DRAM_FIRST_CAPTURE
KV_TOKENS = re.compile(r'GPU KV cache size: ([0-9,]+) tokens')
WEDGE = 'Timed out while waiting for active ethernet core'
# UMD's per-op MMIO budget (2 ms) overran and its hang check did not clear it (umd device_memcpy.cpp): G1 v47's
# eager arm (run 36246961161) logged it from fetch_queue_reserve_back, after the device stopped taking commands.
MMIO_TIMEOUT = 'MMIO per-op timeout'
TRACEBACK = 'Traceback (most recent call last)'
FAILURES = (TRACEBACK, 'EngineDeadError', 'EngineCore encountered a fatal error', 'PrefixInstallError',
            'AssertionError', WEDGE, MMIO_TIMEOUT)
# qwen_prefix_model_patch.MARKER_EAGER_WARM begins with it.
EAGER_WARM = '[PINDIAG] prefix: eager prefill warmed'
# The four-card packed path warms its eager prefill before the packed traces in serving_runtime.prefill_warm_before_traces
# (serving_runtime.WARM_MARKER); on that path the prefix model graft's own warm line is not written.
FOUR_CARD_WARM = '[PINDIAG] four-card eager prefill warmed before the packed traces'
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
    Q, L, path, restored_ms, captured, capture_failed, capture_ms, programs(_before), slot_sha, logits_sha
    (the model graft's spelling and the harness's own, see the module docstring)."""
    body = text.split(MODEL_ROW, 1)[1]
    values, raw = fields(body), dict(FIELD.findall(body))
    req = values.get('req')
    if req is None and isinstance(values.get('row'), str):
        req = values['row']
    captured = values.get('captured')
    if captured is not None and not isinstance(captured, list):
        captured = [captured]
    stored, failed, stored_ms = [], [], []
    for item in captured or ():
        if isinstance(item, int):
            stored.append(item)
            continue
        parts = str(item).split(':')
        if parts[0].isdigit() and len(parts) >= 2 and parts[1] == 'stored':
            stored.append(int(parts[0]))
            if len(parts) >= 3 and parts[2].endswith('ms'):
                try:
                    stored_ms.append(float(parts[2][:-2]))
                except ValueError:
                    pass
        else:
            failed.append(str(item))
    programs = values.get('programs', values.get('program_cache'))
    programs_before = values.get('programs_before')
    arrow = PROGRAMS_ARROW.match(programs) if isinstance(programs, str) else None
    if arrow:
        programs_before = None if arrow.group(1) == 'None' else int(arrow.group(1))
        programs = None if arrow.group(2) == 'None' else int(arrow.group(2))
    capture_ms = values.get('capture_ms')
    if capture_ms is None:
        capture_ms = sum(stored_ms) if stored_ms else (values.get('ms') if 'plan' not in values else None)
    restored_ms = values.get('restored_ms')
    return dict(req=req, tag=request_tag(req) if req else None, q=values.get('Q'), l=values.get('L'),
                path=values.get('path'), restored_ms=restored_ms if isinstance(restored_ms, (int, float)) else None,
                captured=stored, capture_failed=failed, capture_ms=capture_ms,
                programs=programs if isinstance(programs, int) else None,
                row=values.get('row'), slot_sha=raw.get('slot_sha'), logits_sha=raw.get('logits_sha'),
                programs_before=programs_before if isinstance(programs_before, int) else None)


def audit_row(text):
    """A [PREFIX-AUDIT] row; digests and the range are kept as the text printed (an all-digit hex
    digest would otherwise parse as an integer)."""
    body = text.split(AUDIT_ROW, 1)[1]
    values, raw = fields(body), dict(FIELD.findall(body))
    req = values.get('req') or (values.get('row') if isinstance(values.get('row'), str) else None)
    return dict(req=req, tag=request_tag(req) if req else None, q=values.get('Q'), l=values.get('L'),
                kv_range=raw.get('kv_range'), kv_sha=raw.get('kv_sha'), slot_sha=raw.get('slot_sha'),
                logits_sha=raw.get('logits_sha'))


def audit_cost(text):
    """A [PREFIX-AUDIT-COST] line: the read mode, what it read, and the program cache across the audit."""
    values, raw = fields(text), dict(FIELD.findall(text))
    match = PROGRAMS_ARROW.match(raw.get('programs') or '')
    count = lambda part: int(part) if part and part != 'None' else None
    return dict(mode=raw.get('mode'), rows=values.get('rows'), tokens=values.get('tokens'),
                reads=values.get('reads'), blocks_read=values.get('blocks_read'),
                read_ms=values.get('read_ms'), total_ms=values.get('total_ms'),
                programs_before=count(match.group(1)) if match else None,
                programs=count(match.group(2)) if match else None, fallback='fallback=' in text)


def audit_cross(text):
    """A [PREFIX-AUDIT-CROSS] line: how many region selections were compared with the whole-cache read, and how many differed."""
    values = fields(text)
    return dict(tensors=values.get('tensors'), mismatched=values.get('mismatched'), region_ms=values.get('region_ms'),
                whole_read_ms=values.get('whole_read_ms'), blocks_read=values.get('blocks_read'), fallback='fallback=' in text)


def dram_reading(line):
    """A '[PINDIAG] dram after <point>: <reading>' line -> dict(point, chips: [dict(chip, allocated_gb,
    free_gb, largest_free_mb, total_gb)], unavailable: the reason when there are no per-chip figures
    (else None), text), or None for any other line."""
    match = DRAM_READING.search(line)
    if not match:
        return None
    text = match.group(2)
    chips = [dict(chip=int(chip), allocated_gb=float(allocated), free_gb=float(free), largest_free_mb=float(largest),
                  total_gb=float(total))
             for chip, allocated, free, largest, total in DRAM_CHIP.findall(text)]
    unavailable = None
    if not chips:
        found = DRAM_UNAVAILABLE.match(text)
        unavailable = found.group(1) if found else 'no per-chip figures: %s' % text[:200]
    return dict(point=match.group(1).strip(), chips=chips, unavailable=unavailable, text=text[:400])


def tier_line(out, line, where):
    """One of the host tier's or the preconverted checkpoints' lines (TIER_*, CKPT_*) into the scan's lists."""
    if TIER_ON in line:
        entry = fields(line.split(TIER_ON, 1)[1])
        entry.update(where)
        out['tier_on'].append(entry)
    elif TIER_IO in line:
        entry = fields(line.split(TIER_IO, 1)[1])
        entry.update(where)
        out['tier_io'].append(entry)
    elif TIER_SPILL in line:
        entry = fields('blocks=' + line.split(TIER_SPILL, 1)[1])
        entry.update(where)
        out['tier_spills'].append(entry)
    elif TIER_RESTORE in line:
        entry = fields('req=' + line.split(TIER_RESTORE, 1)[1])
        entry.update(where)
        entry['tag'] = request_tag(entry['req']) if isinstance(entry.get('req'), str) else None
        out['tier_restores'].append(entry)
    elif any(marker in line for marker in TIER_TROUBLE):
        out['tier_trouble'].append(dict(where, line=line.strip()[:300]))
    elif CKPT_AUDIT in line:
        entry = fields('tensors=' + line.split(CKPT_AUDIT, 1)[1])
        entry.update(where)
        out['ckpt_audits'].append(entry)
    elif CKPT_PRECONVERTED in line:
        out['ckpt_preconverted'] += 1


def scan(lines):
    """Every marker in a server log (a list of lines, docker timestamps allowed), in order. Each
    entry keeps its line index and timestamp so the driver can window it against a request."""
    out = dict(installs=[], chunked_installs=[], grants=[], rows=[], audits=[], refused=[], capture_skipped=[], kill_switch=[],
               stats=None, launches=[], apc=[], chunking_off=0, chunk_replay=0, dram=[], dram_readings=[],
               kv_tokens=None, failures=[], audit_costs=[], audit_crosses=[], eager_warm=[], four_card_warm=[], sticky_installs=[], sticky_admits=[], sticky_builds=[],
               model_warm=[], model_warm_skipped=[], kv_shared=[], audit_windows=[], tier_on=[], tier_io=[], tier_spills=[], tier_restores=[],
               tier_trouble=[], ckpt_audits=[], ckpt_preconverted=0)
    for index, raw in enumerate(lines):
        stamp, line = split_timestamp(raw.rstrip('\n'))
        where = dict(index=index, time=stamp)
        if INSTALL in line:
            body = line.split(INSTALL, 1)[1]
            entry = fields(body)
            entry.update(where)
            if body.startswith(STICKY_INSTALL):
                out['sticky_installs'].append(entry)
            elif body.startswith(CHUNKED_INSTALL):
                # 'chunked=levern: chunked prefill beside the Lever N cap (max_num_scheduled_tokens=262032)': the mode is the word before the colon.
                entry['chunked'] = body[len(CHUNKED_INSTALL):].split(':', 1)[0].strip()
                cap = CHUNKED_CAP.search(body)
                if cap:
                    entry['max_num_scheduled_tokens'] = int(cap.group(1))
                out['chunked_installs'].append(entry)
            elif body.startswith(SCHEDULER_INSTALL):
                out['installs'].append(entry)
        match = STICKY_ADMIT.search(line)
        if match:
            req = match.group(1) or match.group(2) or match.group(3)
            out['sticky_admits'].append(dict(where, req=req, tag=request_tag(req), q=int(match.group(4)),
                                             p=int(match.group(5)), tail=int(match.group(6))))
        match = STICKY_BUILT.search(line)
        if match:
            out['sticky_builds'].append(dict(where, req=match.group(1), ms=float(match.group(2)),
                                             frontier=int(match.group(3)), prompt=int(match.group(4)),
                                             **({'kind': match.group(5)} if match.group(5) else {})))
        if MODEL_WARM in line:
            entry = fields('restore_mode=' + line.split(MODEL_WARM, 1)[1])
            entry.update(where)
            out['model_warm'].append(entry)
        elif MODEL_WARM_SKIPPED in line:
            out['model_warm_skipped'].append(dict(where, line=line.strip()[:300]))
        if KV_SHARED in line:
            out['kv_shared'].append(dict(where, line=line.strip()[:300]))
        match = AUDIT_WINDOW.search(line)
        if match:
            out['audit_windows'].append(dict(where, req=match.group(1), tag=request_tag(match.group(1)),
                                             q=int(match.group(2)), l=int(match.group(3)), window=int(match.group(4)),
                                             start=int(match.group(5)), end=int(match.group(6)),
                                             new=int(match.group(7)), kv=match.group(8)))
        match = GRANT.search(line)
        if match:
            plan = [int(part) for part in match.group(5).replace(' ', '').split(',') if part]
            out['grants'].append(dict(where, req=match.group(1), tag=request_tag(match.group(1)),
                                      h=int(match.group(2)), q=int(match.group(3)), plan=plan,
                                      drain=int(match.group(4)) if match.group(4) else None))
        if MODEL_ROW_LINE.search(line):
            entry = model_row(line)
            entry.update(where)
            out['rows'].append(entry)
        if AUDIT_COST in line:
            entry = audit_cost(line.split('[PREFIX-AUDIT-COST] ', 1)[1])
            entry.update(where)
            out['audit_costs'].append(entry)
        if AUDIT_CROSS in line:
            entry = audit_cross(line.split('[PREFIX-AUDIT-CROSS] ', 1)[1])
            entry.update(where)
            out['audit_crosses'].append(entry)
        if AUDIT_ROW in line and 'kv_range=' in line:
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
        if '] prefix: host tier ' in line or CKPT_AUDIT in line or CKPT_PRECONVERTED in line:
            tier_line(out, line, where)
        if EAGER_WARM in line:
            entry = fields(line.split(EAGER_WARM, 1)[1])
            entry.update(where)
            out['eager_warm'].append(entry)
        if FOUR_CARD_WARM in line:
            out['four_card_warm'].append(dict(where, line=line.strip()[:300]))
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
            reading = dram_reading(line)
            if reading is not None:
                reading.update(where)
                out['dram_readings'].append(reading)
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
