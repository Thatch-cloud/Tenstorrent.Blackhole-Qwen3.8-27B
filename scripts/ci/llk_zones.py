"""Per-thread LLK profiling zones: exact, reversible source transforms that make zone-instrumented COPIES
of kernel sources. Diagnostic only; never a serving default and never a throughput claim.

The profiler (tt-metal v0.77.0, tt_metal/tools/profiler/kernel_profiler.hpp) offers two scope macros:
  DeviceZoneScopedN(name)      a start and an end marker per execution, on the RISC that runs it. In a
                               compute kernel one source line runs on all three TRISCs (unpack, math,
                               pack), so one zone yields three per-thread intervals.
  DeviceZoneScopedSumN1/N2     no marker per execution: the cycles accumulate in one of SUM_COUNT (2)
                               slots per RISC and leave once per program as a ZONE_TOTAL row whose
                               data column is the sum. Compiled out unless TT_METAL_PROFILER_SUM=1.
Both compile to nothing when the profiler is off, so an instrumented copy that is not profiled runs the
same code as its original (the numerics are the original's either way: remove() restores the original
bytes exactly, and instrument() refuses to return a copy for which that does not hold).

Three kinds of instrumentation, every insertion an exact string tagged 'qwen-llk-zone':
  envelope   the whole body of the kernel's entry point (void kernel_main() or void MAIN) in one block
             zone named QWEN_LLK_<KEY>: per-thread kernel duration with the kernel's identity in the
             zone name, so the report needs no op table to know what ran.
  region     [begin anchor, end anchor) of the source in one block zone QWEN_LLK_<KEY>_<STAGE>. Both
             anchors are exact, unique, line-initial strings; the region must be brace-balanced, declare
             nothing at its own depth (a declaration used after it would leave scope), carry no case
             label, keep its #if/#endif balanced and name no 'hash' or 'zone' (the macro declares both).
  sync       every blocking synchronisation call statement in sum slot 1 (QWEN_LLK_WAIT_IN: waiting for
             an upstream producer) or sum slot 2 (QWEN_LLK_WAIT_OUT: waiting for a downstream consumer):
               TRISC_0 unpack  WAIT_IN  cb_wait_front: input starved by the reader (dataflow-bound)
               TRISC_1 math    WAIT_OUT tile_regs_acquire: dest still held by the packer (pack-bound)
               TRISC_2 pack    WAIT_IN  tile_regs_wait: waiting for math to commit dest (math-bound)
                               WAIT_OUT cb_reserve_back: output ring full (writer-bound)
               BRISC/NCRISC    WAIT_IN  read barriers, producer rings, semaphores
                               WAIT_OUT write barriers and flushes, free space in a ring
             The compute API compiles each of these for one TRISC only, so the slot's meaning follows the
             thread. Math waiting on unpack is a hardware stall (srcA/srcB valid), invisible to software;
             the report takes it from the counters, or marks it undetermined.
A call is wrapped only where its statement boundary is unambiguous (after ';', '{', '}', ')' or 'else',
outside comments, strings and preprocessor lines); every other site is left alone and listed.

Levels: 'tag' (the envelope only: kernel identity for the counter pass) and 'stages' (envelope, regions
and sync sums). Stdlib only, Python 3.7 syntax: the gate runs it on the rig host.
"""
import bisect
import hashlib
import re

TAG = 'qwen-llk-zone'
PREFIX = 'QWEN_LLK_'
WAIT_IN = 'QWEN_LLK_WAIT_IN'
WAIT_OUT = 'QWEN_LLK_WAIT_OUT'
LEVELS = ('tag', 'stages')
HEADER_LINE = '#include "tools/profiler/kernel_profiler.hpp"  // %s\n' % TAG
# PROFILER_L1_OPTIONAL_MARKER_COUNT (hostdevcommon/profiler_common.h:162, v0.77.0): markers per RISC per
# program. A zone costs two per execution; the sum totals one per used slot at the program's end. Past the
# budget the profiler writes nothing more (bufferHasRoom), and the zones after the overflow are lost.
MARKER_BUDGET = 250
MARKER_RESERVE = 32
SUM_SLOTS = 2
NAME = re.compile(r'[A-Z][A-Z0-9_]{0,47}')

SYNC_IN_FUNCS = ('cb_wait_front', 'tile_regs_wait', 'noc_async_read_barrier', 'noc_semaphore_wait',
                 'noc_semaphore_wait_min')
SYNC_IN_METHODS = ('wait_front', 'async_read_barrier')
SYNC_OUT_FUNCS = ('cb_reserve_back', 'tile_regs_acquire', 'noc_async_write_barrier', 'noc_async_writes_flushed')
SYNC_OUT_METHODS = ('reserve_back', 'async_write_barrier', 'async_writes_flushed')
_SYNC = re.compile(
    r'(?P<obj>\b[A-Za-z_]\w*(?:\([^;{}()]*\))?\s*(?:\.|->)\s*)?'
    r'\b(?P<name>%s)\s*\((?P<args>[^;{}]*?)\)\s*;' % '|'.join(
        sorted(set(SYNC_IN_FUNCS + SYNC_IN_METHODS + SYNC_OUT_FUNCS + SYNC_OUT_METHODS), key=len, reverse=True)))
_SUM_WRAP = re.compile(r'\{ DeviceZoneScopedSum(?:N1\("%s"\)|N2\("%s"\)); ([^{}]*?;) \}' % (WAIT_IN, WAIT_OUT))
_REGION_BEGIN = re.compile(r'^[ \t]*\{ DeviceZoneScopedN\("(%s\w+)"\);  // %s\n' % (PREFIX, TAG), re.M)
_REGION_END = re.compile(r'^[ \t]*\}  // %s (%s\w+)\n' % (TAG, PREFIX), re.M)
_ENVELOPE_BEGIN = re.compile(r'\n    \{ DeviceZoneScopedN\("(%s\w+)"\);  // %s envelope' % (PREFIX, TAG))
_ENVELOPE_END = re.compile(r'\}  // %s envelope (%s\w+)\n' % (TAG, PREFIX))
_ENTRY = re.compile(r'\bvoid\s+(?:kernel_main\s*\(\s*\)|MAIN)\s*\{')
_SHADOWED = re.compile(r'\b(?:hash|zone)\b')
_DECLARATION = re.compile(
    r'^\s*(?:(?:const|constexpr|static|volatile|register|unsigned|signed)\s+)*'
    r'(?:auto|bool|char|short|int|long|float|double|size_t|u?int(?:8|16|32|64)_t|std::\w+(?:<[^;]*>)?|'
    r'[A-Za-z_]\w*<[^;()]*>|[A-Z]\w*)\s*[&*]?\s+[A-Za-z_]\w*\s*(?:=|;|\[|\{|\()')
_RECONFIG = re.compile(r'\b(?:\w*reconfig\w*|\w+_init(?:_short)?\w*|compute_kernel_hw_startup|mm_init|'
                       r'mm_block_init)\s*(?:<[^;{}()]*>)?\s*\(')
_FUNCTION = re.compile(r'^[ \t]*(?:template\s*<[^;{}]*>\s*)?(?:(?:inline|static|constexpr|ALWI|FORCE_INLINE)\s+)*'
                       r'(?:void|bool|u?int(?:8|16|32|64)_t|float|auto)\s+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*'
                       r'(?:const\s*)?\{', re.M)


class ZoneError(ValueError):
    """A source this transform cannot instrument exactly, with the reason."""


def decode(data):
    """Kernel bytes as text, byte-exactly reversible (encode) whatever they hold."""
    return data.decode('utf-8', 'surrogateescape')


def encode(text):
    return text.encode('utf-8', 'surrogateescape')


def sha256(text):
    """The sha256 of the bytes `text` decodes (decode) - the file's own sha256."""
    return hashlib.sha256(encode(text)).hexdigest()


def zone_name(*parts):
    """QWEN_LLK_<PARTS>: [A-Z0-9_] only (a zone name is a string literal and a report key)."""
    name = PREFIX + '_'.join(parts)
    if not NAME.fullmatch(name):
        raise ZoneError('zone name %r must match %s' % (name, NAME.pattern))
    return name


# ---- lexical scan: what is not code ----

def excluded_spans(text):
    """Sorted (start, end) spans of comments, string and character literals, and preprocessor lines
    (with their backslash continuations). Raw strings are not handled: none of the kernels use them."""
    spans = []
    index, length = 0, len(text)
    line_start = True
    while index < length:
        char = text[index]
        if line_start and char in ' \t':
            index += 1
            continue
        if line_start and char == '#':
            start = index
            while index < length:
                end = text.find('\n', index)
                if end < 0:
                    index = length
                    break
                if end > 0 and text[end - 1] == '\\':
                    index = end + 1
                    continue
                index = end
                break
            spans.append((start, index))
            continue
        line_start = False
        if text.startswith('//', index):
            end = text.find('\n', index)
            end = length if end < 0 else end
            spans.append((index, end))
            index = end
            continue
        if text.startswith('/*', index):
            end = text.find('*/', index + 2)
            end = length if end < 0 else end + 2
            spans.append((index, end))
            index = end
            continue
        if char in '"\'':
            start = index
            index += 1
            while index < length and text[index] != char:
                index += 2 if text[index] == '\\' else 1
            index += 1
            spans.append((start, min(index, length)))
            continue
        if char == '\n':
            line_start = True
        index += 1
    return spans


class _Code(object):
    """Position queries over a source: in excluded text or not, brace depth."""

    def __init__(self, text):
        self.text = text
        self.spans = excluded_spans(text)
        self.starts = [start for start, _ in self.spans]

    def excluded(self, position):
        index = bisect.bisect_right(self.starts, position) - 1
        return index >= 0 and self.spans[index][0] <= position < self.spans[index][1]

    def code_chars(self, start, end):
        """(position, char) of every code character in [start, end)."""
        position = start
        index = bisect.bisect_right(self.starts, position) - 1
        if index < 0 or self.spans[index][1] <= position:
            index += 1
        spans, text = self.spans, self.text
        while position < end:
            if index < len(spans) and spans[index][0] <= position:
                position = max(position, spans[index][1])
                index += 1
                continue
            limit = end if index >= len(spans) else min(end, spans[index][0])
            for offset in range(position, limit):
                yield offset, text[offset]
            position = limit

    def matching_brace(self, open_position):
        depth = 0
        for position, char in self.code_chars(open_position, len(self.text)):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return position
        raise ZoneError('unbalanced braces after offset %d' % open_position)

    def previous_code(self, position):
        """The code text before `position` back to the previous code character, stripped."""
        index = position - 1
        while index >= 0:
            if self.text[index].isspace():
                index -= 1
                continue
            span = bisect.bisect_right(self.starts, index) - 1
            if span >= 0 and self.spans[span][0] <= index < self.spans[span][1]:
                index = self.spans[span][0] - 1
                continue
            return index
        return -1


def _line_start(text, position):
    return text.rfind('\n', 0, position) + 1


def _unique(text, anchor, what):
    count = text.count(anchor)
    if count != 1:
        raise ZoneError('%s %r must occur exactly once, found %d' % (what, anchor[:60], count))
    position = text.index(anchor)
    if position and text[position - 1] != '\n':
        raise ZoneError('%s %r must start a line' % (what, anchor[:60]))
    return position


# ---- region lint ----

def region_problems(text, start, end):
    """Why [start, end) of `text` cannot be one block zone, or []."""
    code = _Code(text)
    problems = []
    depth = 0
    body = text[start:end]
    for position, char in code.code_chars(start, end):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth < 0:
                problems.append('closes a brace it did not open')
                break
    if depth > 0:
        problems.append('leaves %d braces open' % depth)
    conditionals = 0
    for line in body.split('\n'):
        stripped = line.strip()
        if re.match(r'#\s*if', stripped):
            conditionals += 1
        elif re.match(r'#\s*endif', stripped):
            conditionals -= 1
            if conditionals < 0:
                break
    if conditionals:
        problems.append('#if/#endif unbalanced')
    # Declarations and labels at the region's own depth, by line; comment-only text is excluded.
    depth = 0
    line_offset = start
    for line in body.split('\n'):
        code_text = ''.join(char for position, char in code.code_chars(line_offset, line_offset + len(line)))
        if depth == 0 and code_text.strip():
            if _DECLARATION.match(code_text) and not re.match(r'\s*(?:return|else|if|for|while|switch|do)\b',
                                                               code_text):
                problems.append('declares at its own depth: %r' % code_text.strip()[:80])
            if re.match(r'\s*(?:case\b|default\s*:|goto\b)', code_text):
                problems.append('carries a label or goto: %r' % code_text.strip()[:80])
        depth += code_text.count('{') - code_text.count('}')
        line_offset += len(line) + 1
    shadowed = [match.group(0) for match in _SHADOWED.finditer(body) if not code.excluded(start + match.start())]
    if shadowed:
        problems.append('names %s, which the zone macro declares' % ', '.join(sorted(set(shadowed))))
    return problems


# ---- the transforms ----

def _entry(text):
    code = _Code(text)
    matches = [match for match in _ENTRY.finditer(text) if not code.excluded(match.start())]
    if len(matches) != 1:
        raise ZoneError('exactly one kernel entry point (void kernel_main() { or void MAIN {) required, found %d'
                        % len(matches))
    open_position = matches[0].end() - 1
    return code, open_position, code.matching_brace(open_position)


def add_envelope(text, name):
    """The entry point's whole body in one block zone `name`."""
    code, open_position, close_position = _entry(text)
    body = text[open_position + 1:close_position]
    shadowed = [match.group(0) for match in _SHADOWED.finditer(body)
                if not code.excluded(open_position + 1 + match.start())]
    if shadowed:
        raise ZoneError('the entry point names %s, which the zone macro declares' % ', '.join(sorted(set(shadowed))))
    begin = '\n    { DeviceZoneScopedN("%s");  // %s envelope' % (name, TAG)
    end = '}  // %s envelope %s\n' % (TAG, name)
    return text[:open_position + 1] + begin + text[open_position + 1:close_position] + end + text[close_position:]


def region_bounds(text, name, begin_anchor, end_anchor):
    """(start, end) of a region: two unique line-initial anchors, or (end_anchor None) the begin anchor to
    the line of the entry point's closing brace."""
    start = _unique(text, begin_anchor, 'region %s begin anchor' % name)
    if end_anchor is None:
        _, _, close_position = _entry(text)
        end = _line_start(text, close_position)
        if text[end:close_position].strip():
            raise ZoneError('region %s: the closing brace of the entry point does not start a line' % name)
    else:
        end = _unique(text, end_anchor, 'region %s end anchor' % name)
    if end <= start:
        raise ZoneError('region %s: the end anchor precedes the begin anchor' % name)
    return start, end


def add_region(text, name, begin_anchor, end_anchor):
    """[begin_anchor, end_anchor) in one block zone `name`, at the begin anchor's indentation. end_anchor None
    closes the region at the entry point's closing brace."""
    start, end = region_bounds(text, name, begin_anchor, end_anchor)
    problems = region_problems(text, start, end)
    if problems:
        raise ZoneError('region %s: %s' % (name, '; '.join(problems)))
    indent = re.match(r'[ \t]*', text[start:]).group(0)
    return (text[:start] + '%s{ DeviceZoneScopedN("%s");  // %s\n' % (indent, name, TAG) + text[start:end]
            + '%s}  // %s %s\n' % (indent, TAG, name) + text[end:])


def sync_sites(text):
    """([(start, end, slot, line)], [(line, call, reason)]): the sync call statements this transform wraps,
    and those it leaves (with why). slot 1 = WAIT_IN, 2 = WAIT_OUT."""
    code = _Code(text)
    wrapped, skipped = [], []
    for match in _SYNC.finditer(text):
        name, obj = match.group('name'), match.group('obj')
        line = text.count('\n', 0, match.start()) + 1
        if code.excluded(match.start()):
            continue
        method = name in SYNC_IN_METHODS + SYNC_OUT_METHODS and name not in SYNC_IN_FUNCS + SYNC_OUT_FUNCS
        if bool(obj) != method:
            skipped.append((line, match.group(0).strip()[:80], 'method/function form mismatch'))
            continue
        before = code.previous_code(match.start())
        boundary = before < 0 or text[before] in ';{})' or (
            text[before] == 'e' and re.search(r'(?:^|[^\w])else$', text[max(0, before - 5):before + 1]) is not None)
        if not boundary:
            skipped.append((line, match.group(0).strip()[:80], 'no unambiguous statement boundary before it'))
            continue
        if _SHADOWED.search(match.group(0)):
            skipped.append((line, match.group(0).strip()[:80], 'names hash or zone'))
            continue
        slot = 1 if name in SYNC_IN_FUNCS + SYNC_IN_METHODS else 2
        wrapped.append((match.start(), match.end(), slot, line))
    return wrapped, skipped


def add_sync(text):
    """Every wrappable sync call statement in its sum slot; returns (text, wrapped count by slot, skipped)."""
    wrapped, skipped = sync_sites(text)
    pieces, position = [], 0
    counts = {1: 0, 2: 0}
    for start, end, slot, _ in wrapped:
        macro = 'DeviceZoneScopedSumN1("%s")' % WAIT_IN if slot == 1 else 'DeviceZoneScopedSumN2("%s")' % WAIT_OUT
        pieces.append(text[position:start])
        pieces.append('{ %s; %s }' % (macro, text[start:end]))
        position = end
        counts[slot] += 1
    pieces.append(text[position:])
    return ''.join(pieces), counts, skipped


def add_header(text):
    """The profiler header after the first #include (or #pragma once) directive outside every #if block - a
    real directive, not one inside a block comment - else at the top."""
    code = _Code(text)
    for directive_name in ('include', 'pragma'):
        depth = 0
        for start, end in code.spans:
            if text[start] != '#':
                continue
            directive = text[start:end]
            word = re.match(r'#\s*(\w+)', directive)
            word = word.group(1) if word else ''
            if word in ('if', 'ifdef', 'ifndef'):
                depth += 1
            elif word == 'endif':
                depth -= 1
            elif depth == 0 and word == directive_name and '\\\n' not in directive and (
                    word == 'include' or re.match(r'#\s*pragma\s+once\b', directive)):
                if end >= len(text):
                    return text + '\n' + HEADER_LINE
                return text[:end + 1] + HEADER_LINE + text[end + 1:]
    return HEADER_LINE + text


def remove(text):
    """The original bytes of an instrumented copy: every tagged insertion taken out."""
    text = text.replace(HEADER_LINE, '')
    previous = None
    while previous != text:
        previous = text
        text = _SUM_WRAP.sub(lambda match: match.group(1), text)
    text = _ENVELOPE_BEGIN.sub('', text)
    text = _ENVELOPE_END.sub('', text)
    text = _REGION_BEGIN.sub('', text)
    text = _REGION_END.sub('', text)
    return text


def check_source(text, what='source'):
    if '\r' in text:
        raise ZoneError('%s must be LF-only: its sha256 is its identity' % what)
    if PREFIX in text or TAG in text:
        raise ZoneError('%s is already instrumented' % what)


def marker_count(regions, envelope=True, sums=True):
    """Optional markers one RISC writes per program: two per zone execution plus one per used sum slot."""
    total = (2 if envelope else 0) + sum(2 * int(multiplicity) for _, multiplicity in regions)
    return total + (SUM_SLOTS if sums else 0)


def instrument(text, key, level, regions=(), envelope=True, sync=True, sums_supported=True,
               budget=MARKER_BUDGET, what=None):
    """(instrumented text, record). `regions`: [(stage, begin anchor, end anchor, multiplicity)] where the
    multiplicity bounds how often the region runs per program (the marker lint's input). 'tag' keeps only the
    envelope. Refuses (ZoneError) anything it cannot do exactly, before returning a copy."""
    what = what or key
    if level not in LEVELS:
        raise ZoneError('level must be one of %s, got %r' % (', '.join(LEVELS), level))
    check_source(text, what)
    stages = level == 'stages'
    use_regions = list(regions) if stages else []
    use_sums = bool(stages and sync and sums_supported)
    if not envelope and not use_regions and not use_sums:
        raise ZoneError('%s: nothing to instrument at level %s' % (what, level))
    markers = marker_count([(stage, multiplicity) for stage, _, _, multiplicity in use_regions], envelope, use_sums)
    limit = budget - MARKER_RESERVE
    if markers > limit:
        raise ZoneError('%s: %d markers per RISC per program at level %s, past the %d the lint allows (budget %d, '
                        'reserve %d)' % (what, markers, level, limit, budget, MARKER_RESERVE))
    result = text
    zones = []
    for stage, begin_anchor, end_anchor, multiplicity in use_regions:
        name = zone_name(key, stage)
        static = reconfig_calls(text, *region_bounds(text, name, begin_anchor, end_anchor))
        result = add_region(result, name, begin_anchor, end_anchor)
        zones.append(dict(name=name, kind='region', stage=stage, multiplicity=int(multiplicity), reconfig_static=static))
    envelope_name = None
    if envelope:
        envelope_name = zone_name(key)
        result = add_envelope(result, envelope_name)
        zones.insert(0, dict(name=envelope_name, kind='envelope', stage=None, multiplicity=1,
                             reconfig_static=reconfig_total(text)))
    counts, skipped = {1: 0, 2: 0}, []
    if use_sums:
        result, counts, skipped = add_sync(result)
    result = add_header(result)
    if remove(result) != text:
        raise ZoneError('%s: removing the zones does not restore the original bytes' % what)
    record = dict(key=key, level=level, envelope=envelope_name, zones=zones, source_sha256=sha256(text),
                  instrumented_sha256=sha256(result),
                  sync=dict(enabled=use_sums, wait_in=counts[1], wait_out=counts[2],
                            skipped=[dict(line=line, call=call, reason=reason) for line, call, reason in skipped]),
                  markers=dict(per_risc_per_program=markers, budget=budget, reserve=MARKER_RESERVE))
    return result, record


# ---- static reconfiguration counts ----

def function_bodies(text):
    """name -> body text of every function defined in `text` (the last definition of an overloaded name)."""
    code = _Code(text)
    bodies = {}
    for match in _FUNCTION.finditer(text):
        if code.excluded(match.start('name')):
            continue
        open_position = match.end() - 1
        try:
            close = code.matching_brace(open_position)
        except ZoneError:
            continue
        bodies[match.group('name')] = (open_position + 1, close)
    return bodies


def reconfig_calls(text, start, end, _bodies=None, _seen=None):
    """Static count of unpack/pack/math reconfiguration and init calls in [start, end), following calls into
    functions defined in the same text (once per call site, recursion cut). A static count: how often each
    runs is the zone's instance count."""
    code = _Code(text)
    bodies = function_bodies(text) if _bodies is None else _bodies
    seen = frozenset() if _seen is None else _seen
    count = 0
    segment = text[start:end]
    for match in _RECONFIG.finditer(segment):
        if not code.excluded(start + match.start()):
            count += 1
    for name, (body_start, body_end) in bodies.items():
        if name in seen or (body_start <= start and end <= body_end):
            continue
        for match in re.finditer(r'\b%s\s*(?:<[^;{}()]*>)?\s*\(' % re.escape(name), segment):
            position = start + match.start()
            if code.excluded(position) or body_start - 1 <= position <= body_end:
                continue
            count += reconfig_calls(text, body_start, body_end, bodies, seen | {name})
    return count


def reconfig_total(text):
    """Static reconfiguration count of the entry point's body."""
    _, open_position, close_position = _entry(text)
    return reconfig_calls(text, open_position + 1, close_position)
