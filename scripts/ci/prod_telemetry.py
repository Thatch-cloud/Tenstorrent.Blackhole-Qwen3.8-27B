#!/usr/bin/env python3
"""Production telemetry from the serving container's own log: check, capture, watch, sanitise.

Why this exists. After the Lever N + prefix cutover the production image is judged on real traffic, with a rollback rule
(Tier 0 image rollback, Tier 1 kill switches, Tier 2 statistics). Round time, the hang rule and the Lever N quarantine rule all
live in the container's log, not in a metric, so something has to read it. This module is that reader. It runs ON THE SERVING HOST:
the raw log never leaves it, and what it writes is derived records only (round times, request shapes, trigger events).

    <log source> | python3 prod_telemetry.py capture --out <dir> -
    <log source> | python3 prod_telemetry.py check [--expect-levern] [--metrics-url URL] [--boot-count KEY=N]
    <log source> | python3 prod_telemetry.py sanitise > clean.log

The log source is the engine's own log. A serving container can run with its container log driver off, so `docker logs` is EMPTY there and
the engine writes one log file per engine generation INSIDE the container (the newest is live). The wrappers prod_telemetry_rig_check.sh (one
read-only look) and prod_telemetry_rig_capture.sh (follow, restart-safe) read it with `docker exec ... tail`; the file's glob is an argument
(--log-glob), never a default. For a container started by hand (a CI gate) `docker logs` carries the same lines. A platform that renames the
engine's metrics gives the name prefix with --platform-prefix.

Subcommands.
  check     Reads a log window (and optionally a /metrics scrape) and says which of the lines the rollback rule needs are present. It prints
            counts, never log text. Exit 0 = every required signal present, 2 = a required signal absent while requests ran, 3 = no
            traffic in the window, or fewer than 5 decode steps with two or more users when the missing signals are the per-user audit lines of a packed round
            (inconclusive: send two or more concurrent requests and rerun), 4 = the metrics URL did not answer. The one-off boot lines
            (`lever N installed`) scroll out of a tail window: the wrapper counts them over the whole file and passes --boot-count.
  capture   Reads a log stream and appends one JSON line per decode round, per Lever N request and per event to daily files in --out
            (rounds-DAY.jsonl, requests-DAY.jsonl, events-DAY.jsonl) and runs the watcher below. Nothing in those files carries a token id,
            a prompt, a client address, a raw request id (ids are hashed) or free text from the log. Each round carries the host's 1-minute load
            average (`load1`, sampled every 10 s on the host the capture runs on): the Tier 2 comparator reads rounds LOAD-MATCHED. It never
            writes a raw log line.
            --prom-file F publishes the capture's own health to F as Prometheus text (point F into the host's node_exporter textfile directory): a
            heartbeat timestamp and one counter per trigger code, so the log-only rules reach the alert rules and a dead capture shows as a stale heartbeat.
            --maintenance-file F: while F exists (and is under 12 h old) the Tier 0 triggers are written to triggers.jsonl as suppressed and counted apart,
            not raised (planned window work restarts the engine; the Tier 1 triggers still fire). --metrics-url-file F: the wrapper keeps the /metrics URL
            there, so a published port that changes with a redeploy is followed.
  watch     Capture's trigger half alone (prints the triggers, writes no records).
  sanitise  A log filter for the rare case a log excerpt must be shared: keeps only known telemetry lines, drops every token-id list
            (`predictions=[...]` and any list of integers) and client addresses. Unknown lines are dropped, not passed through.

Triggers (the approved rule; the watcher only REPORTS, it never writes a kill-switch file or restarts anything):
  T0-HANG             no '[PHASE] execute' line for HANG_S (300 s) while a request is live. 'Live' is, in order: the engine's /metrics gauges
                      (`vllm:num_requests_running + waiting`, --metrics-url, polled every 10 s) when they are exported and fresh; else the LAST
                      stats log line. vLLM prints that line while tokens flow and one more after they stop ("Running: 0" when the work is
                      done, "Running: N" when it is not), then nothing: a silent engine whose last line says N > 0 is hung, one whose last
                      line says 0 is idle. A restart of the follower (the capture wrapper prints a follow marker) forgets the old line
  T0-LOG-STALLED      the same silence, but tokens ARE still being generated (the generation-token counter advanced in the last minute, or the
                      last stats line shows generation throughput): the engine is alive and its LOG is not advancing (a full log filesystem,
                      the phase log switched off). Not a rollback trigger: free the log's space and read the capture gap as a gap
  T0-ENGINE-DEATH     an engine-level death marker (EngineDeadError, 'EngineCore ... died', Fatal Python error, a shell 'Segmentation fault' line) at
                      once; a ttnn/runtime error text (TT_THROW, TT_FATAL, terminate called) only when CONFIRMED: no '[PHASE] execute' line for
                      NATIVE_CONFIRM_S (45 s) with a request live, or the API unreachable. A line tagged [PREFIX] or [PINDIAG], or logged at WARNING level,
                      is never a death (a recoverable prefix capture failure carries a TT_THROW text). Only the marker's name is recorded, no log text
  T0-API-UNREACHABLE  the engine's /metrics stopped answering for API_DOWN_S (120 s) after having answered: the API server or the container is gone
                      (a hung API server counts; a capture started while the API was already down does not alarm)
  T0-SERVER-ERRORS    ERROR_COUNT (3) or more HTTP 5xx answers on /v1/ inside ERROR_WINDOW_S (1 h)
  T1-LEVERN-QUARANTINE  any '[PINDIAG] lever N quarantine' line
  T1-LEVERN-LONG-TTFT   a request of more than LONG_PROMPT (128k) tokens whose Lever N prefill has run for more than TTFT_S (240 s) from its first
                        step line to its final step (or still running with a step line in the last STALE_S seconds). This is a LOWER BOUND on its
                        TTFT (queue wait before the first step is not in the log), so a trigger is a certain breach and a silence is not proof
                        of none. A request that is aborted, quarantined or finished (the step's `finished=[...]` list, an 'Aborted request' line,
                        a quarantine line) leaves the in-flight table and cannot fire: confirm with the gateway's per-request ttft before writing
                        a kill switch on this trigger
The per-day error ratio over 24 h, the prefix grant ratio and the Tier 2 statistics are not log rules: see the comparator and the alert rules.

Stdlib only; Python 3.8 or later (the serving host's own interpreter).
"""

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# ---- the log lines (producers: serving_worker_hook, serving_packed_step, packed_verifier, levern_policy, vLLM's stat logger) -------------------------
STAMP = r'([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2})(\.[0-9]+)? \|'
EXECUTE = re.compile(STAMP + r'[^\n]*?\[PHASE\] execute total=([0-9]+) new=([0-9]+) cached=([0-9]+) spec=([0-9]+)(?: finished=\[([^\]]*)\])?')
PACKED = re.compile(r'\[PACKED\] request=(\S+) segment=([0-9]+) position=([0-9]+) prefix=([0-9]+) emitted=([0-9]+)')
PACKED_PHASE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)')
# vLLM's periodic stat line: "INFO 10-04 17:04:44 [loggers.py:273] Engine 000: Avg prompt throughput: ..., Running: 3 reqs, Waiting: 1 reqs, ..."
STATS = re.compile(r'Engine [0-9]+: Avg prompt throughput: ([0-9.]+) tokens/s, Avg generation throughput: ([0-9.]+) tokens/s, '
                   r'Running: ([0-9]+) reqs, Waiting: ([0-9]+) reqs')
ACCESS = re.compile(r'"([A-Z]+) (\S+) HTTP/[0-9.]+" ([0-9]{3})')
LEVERN_STEP = re.compile(STAMP + r'[^\n]*?\[PINDIAG\] lever N step n=([0-9]+) kind=(prefill|decode) seats=(\S+) req=(\S+) start=(\S+) '
                         r'tokens=(\S+) end=(\S+) prompt=(\S+) final=(\S+)')
LEVERN_QUARANTINE = re.compile(r'\[PINDIAG\] lever N quarantine req=(\S+): (.*)')
LEVERN_PREFILL_STEP = re.compile(r'lever N step n=\S+ kind=prefill')
REQUEST_QUARANTINED = re.compile(r'\[PINDIAG\] request quarantined: [\'"]?([^\'"\s]+)[\'"]?')
DECODING_COUNT = re.compile(r'\(([0-9]+) decoding\)')
ABORTED = re.compile(r'[Aa]borted request')
QUOTED_ID = re.compile(r'[\'"]([A-Za-z0-9_.:-]{8,})[\'"]')
BARE_ID = re.compile(r'\b(?:chat)?cmpl-[A-Za-z0-9_.-]+')
MARKERS = dict(
    levern_installed='[PINDIAG] lever N installed on',
    levern_route='[PINDIAG] lever N route installed',
    levern_kill='[PINDIAG] lever N kill switch',
    levern_refused='[PINDIAG] lever N REFUSED',
    prefix_kill='[PINDIAG] prefix: kill switch',
    quarantine_consumer='[PINDIAG] request quarantine consumer live in',
    request_quarantined='[PINDIAG] request quarantined',
)
# What counts as the engine being gone. Two classes, because a ttnn error text is not a death by itself: the prefix patch logs a RECOVERABLE capture failure
# as `[PREFIX] capture not stored req=... pos=...: TT_THROW @ ...` and serves the request on.
#   * ENGINE markers are engine-level: the engine core reports itself dead, Python aborts, or the shell reports the process killed by a signal. They fire
#     T0-ENGINE-DEATH at once.
#   * NATIVE markers (a ttnn/runtime error text) are only a SUSPICION: they fire T0-ENGINE-DEATH when confirmed (no '[PHASE] execute' line for
#     NATIVE_CONFIRM_S afterwards with a request live, or the API stopped answering).
# A line that carries a [PREFIX] or [PINDIAG] tag, or is logged at WARNING level, is never a death line. Only the matched marker's NAME is recorded, never text
# from the line (a line can carry prompt text when request logging is on).
ENGINE_DEATH_RE = (
    ('EngineDeadError', re.compile(r'EngineDeadError')),
    ('EngineCore-died', re.compile(r'Engine ?[Cc]ore(?: proc\w*)?\b[^\n]{0,80}\b(?:died|is dead|has died)\b')),
    ('Fatal-Python-error', re.compile(r'Fatal Python error')),
    ('process-signal', re.compile(r'^(?:\([^)]*\)\s*)?(?:Segmentation fault|Aborted \(core dumped\)|Bus error|Illegal instruction)')),
)
NATIVE_ERROR_RE = (
    ('TT_FATAL', re.compile(r'\bTT_FATAL\b')),
    ('TT_THROW', re.compile(r'\bTT_THROW\b')),
    ('terminate-called', re.compile(r'terminate called')),
)
NOT_A_DEATH = re.compile(r'\[PREFIX\]|\[PINDIAG\]|\|\s*WARNING\s*\||\bWARNING\s+[0-9]{2}-[0-9]{2}\b|^\s*(?:\([^)]*\)\s*)?WARNING\b')


def death_marker(line):
    """('engine'|'native', marker name) when the line is an engine-death or an unconfirmed native-error line, else None (module comment above)."""
    if NOT_A_DEATH.search(line):
        return None
    for name, pattern in ENGINE_DEATH_RE:
        if pattern.search(line):
            return 'engine', name
    for name, pattern in NATIVE_ERROR_RE:
        if pattern.search(line):
            return 'native', name
    return None

# ---- thresholds (the approved rule) -----------------------------------------------------------------------------------------------------------------
HANG_S = 300.0
ERROR_COUNT = 3
ERROR_WINDOW_S = 3600.0
TTFT_S = 240.0
LONG_PROMPT = 131072
API_DOWN_S = 120.0
LIVE_FRESH_S = 60.0          # a metrics reading older than this is not a reading
FLOW_AFTER_S = 30.0          # a stats line with generation throughput counts as 'tokens flow' only this long after the last phase line (it averages the 10 s before it)
STALE_S = 60.0               # a Lever N prefill with no step line for this long is not running (an abort leaves no final step)
FORGET_S = 3600.0            # ... and is forgotten altogether after this long
LOAD_EVERY_S = 10.0          # the host load average is sampled this often and stamped on every round
FOLLOW_MARKER = '[prod-telemetry] follow'   # printed by the capture wrapper when it (re)attaches to an engine log: the old stats line is forgotten
MIN_MULTI_STEPS = 5          # the check calls a window packed-capable only with this many decode steps carrying two or more users
NATIVE_CONFIRM_S = 45.0      # a native error text is a death only when no '[PHASE] execute' line follows for this long with a request live (or the API is gone)
ROWS_PER_USER_MAX = 16       # a decode step carries at most (draft tokens + 1) rows per live user (dflash 15 + 1); a step with more rows per user is a prefill chunk
MAINT_MAX_S = 12 * 3600.0    # a maintenance marker older than this is ignored: a forgotten marker must not switch Tier 0 off for good
TRIGGER_CODES = ('T0-HANG', 'T0-LOG-STALLED', 'T0-ENGINE-DEATH', 'T0-API-UNREACHABLE', 'T0-SERVER-ERRORS', 'T1-LEVERN-QUARANTINE', 'T1-LEVERN-LONG-TTFT')
MAX_ROUND_S = 5.0          # a gap between two decode steps longer than this is idle time, not a round
WINDOW_EDGES = ((0, 8192, 'steady'), (28672, 40960, '32k'), (106496, 139264, '128k'))   # w2ln_timing_compare.WINDOWS: mean position, low <= x < high

TIER_ACTION = {
    'T0-HANG': 'Tier 0: image rollback (runbook section Tier 0)',
    'T0-LOG-STALLED': 'Investigate, not a rollback: tokens flow but the engine log does not advance (runbook section: log filesystem headroom)',
    'T0-ENGINE-DEATH': 'Tier 0: image rollback (runbook section Tier 0)',
    'T0-API-UNREACHABLE': 'Tier 0: image rollback (runbook section Tier 0)',
    'T0-SERVER-ERRORS': 'Tier 0: image rollback (runbook section Tier 0)',
    'T1-LEVERN-QUARANTINE': 'Tier 1: write levern.off (runbook section Tier 1)',
    'T1-LEVERN-LONG-TTFT': 'Tier 1: confirm with the gateway ttft, then write levern.off (runbook section Tier 1)',
}

# ---- sanitising ---------------------------------------------------------------------------------------------------------------------------------------
PREDICTIONS = re.compile(r'predictions=\[([^\]\n]*)\]')
# a list of integers after `name=` or `name:` (two or more of them, any name; ONE of them when the name says ids or tokens: `prompt_token_ids: [5]`)
INT_LIST = re.compile(r'(\w+)\s*[=:]\s*\[\s*-?[0-9]+(?:\s*,\s*-?[0-9]+)+\s*\]')
ID_LIST = re.compile(r'(\w*(?:ids?|tokens?)\w*)\s*[=:]\s*\[\s*-?[0-9]+\s*\]', re.IGNORECASE)
PROMPT_TEXT = re.compile(r'prompt(?!\s+throughput|_tokens_total)', re.IGNORECASE)
ADDRESS = re.compile(r'\b[0-9]{1,3}(?:\.[0-9]{1,3}){3}(?::[0-9]+)?\b')
KEEP = ('[PHASE]', '[PACKED', '[PINDIAG]', '[GDN-SEQ-BLOCK', '[SEQ-PUBLISH]', '[CARRY]', 'Engine 0', 'HTTP/1.1"')


def sanitise_line(line):
    """The line with every token-id list reduced to its length and every address removed (the line's other text is untouched)."""
    line = PREDICTIONS.sub(lambda m: 'predictions=<%d ids dropped>' % (len([v for v in m.group(1).split(',') if v.strip()])), line)
    line = INT_LIST.sub(lambda m: '%s=<%d ids dropped>' % (m.group(1), m.group(0).count(',') + 1), line)
    line = ID_LIST.sub(lambda m: '%s=<1 ids dropped>' % m.group(1), line)
    return ADDRESS.sub('<addr>', line)


def sanitise_stream(lines, keep_all=False):
    for line in lines:
        line = line.rstrip('\n')
        if keep_all:
            # every line, EXCEPT one that can carry prompt text: a request log line, a `prompt: ...` echo
            if PROMPT_TEXT.search(line) and not STATS.search(line):
                continue
            yield sanitise_line(line)
        elif any(marker in line for marker in KEEP) or death_marker(line):
            yield sanitise_line(line)


def request_hash(request_id):
    return hashlib.sha256(request_id.encode('utf-8', 'replace')).hexdigest()[:10]


def scrub(text, limit=200):
    """Event text for a capture file: addresses and token-id lists removed, every request id (bare or quoted) replaced by its hash."""
    text = sanitise_line(text)
    text = QUOTED_ID.sub(lambda m: 'id:' + request_hash(m.group(1)), text)
    text = BARE_ID.sub(lambda m: 'id:' + request_hash(m.group(0)), text)
    return text[:limit]


def window_of(mean_position, edges=WINDOW_EDGES):
    for low, high, name in edges:
        if low <= mean_position < high:
            return name
    return None


def stamp_of(match, offset=1):
    """(day, 'HH:MM:SS.mmm', datetime) of the three timestamp groups starting at `offset`."""
    day, clock, fraction = match.group(offset), match.group(offset + 1), match.group(offset + 2) or ''
    moment = datetime.strptime('%s %s%s' % (day, clock, fraction), '%Y-%m-%d %H:%M:%S.%f' if fraction else '%Y-%m-%d %H:%M:%S')
    return day, (clock + (fraction + '000')[:4]), moment


def utc_day():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


LINE_STAMP = re.compile(STAMP)


def ids_in(text):
    """The request ids in a `finished=[...]` list body or an abort line: quoted tokens and bare cmpl- ids."""
    found = [m.group(1) for m in QUOTED_ID.finditer(text)]
    found += [m.group(0) for m in BARE_ID.finditer(text)]
    return found


class Extractor(object):
    """Turns log lines into records: decode rounds, Lever N requests and events. Stateful; one instance per stream.

    A ROUND is a decode step ('[PHASE] execute ... new=0 cached=C') timed to the NEXT '[PHASE] execute' line, when that is a decode step too
    and follows within MAX_ROUND_S (acceptance_report.round_split's rule: a step followed by a prefill, or by idle time, is not a round).
    Its users' positions are the '[PACKED] request= ... position=' lines logged before the next execute line; `packed` says there is one per live
    user (the comparator keeps only those, as w2ln_timing_compare does). `after_prefill` marks a round that is beside a prefill, the w2ln rule
    (w2ln_timing_compare.timed_rounds): the round's own step or the step before it carried a Lever N prefill line, or the step before it was an
    engine prefill (new > 0, a superset of w2ln's rule, applied to both sides). The comparator drops those on both sides."""

    def __init__(self, max_round_s=MAX_ROUND_S):
        self.max_round_s = max_round_s
        self.pending = None        # the open DECODE step (timed to the next decode step)
        self.current = None        # the open step of any kind: dict(new, prefill)
        self.requests = collections.OrderedDict()
        self.counts = collections.Counter()
        self.first = self.last = None
        self.running = self.waiting = None
        self.gen_tps = None
        self.max_live = 0          # the most live users any decode step of the stream carried

    def feed(self, line):
        """[(kind, record)] the line completes. kind is 'round', 'request' or 'event'."""
        out = []
        if line.startswith(FOLLOW_MARKER):
            self.counts['follow'] += 1
            parts = line[len(FOLLOW_MARKER):].strip().split(None, 1)
            if parts:
                out.append(('event', dict(code='follow', detail=scrub(' '.join([os.path.basename(parts[0])] + parts[1:])))))
            self.running = self.waiting = None
        elif '[PHASE] execute' in line:
            match = EXECUTE.search(line)
            if match:
                self._execute(match, out)
        elif '[PACKED] request=' in line:
            match = PACKED.search(line)
            self.counts['packed_lines'] += 1
            if match and self.pending is not None:
                self.pending['positions'].append(int(match.group(3)))
        elif '[PACKED-PHASE]' in line:
            self.counts['packed_phase'] += 1
        elif 'lever N' in line:
            self._levern(line, out)
        elif 'Running:' in line and 'Engine ' in line:
            match = STATS.search(line)
            if match:
                self.counts['stats'] += 1
                self.gen_tps = float(match.group(2))
                self.running, self.waiting = int(match.group(3)), int(match.group(4))
                if self.running or self.waiting:
                    self.counts['stats_busy'] += 1
        elif 'HTTP/' in line and '"' in line:
            match = ACCESS.search(line)
            if match:
                self.counts['access'] += 1
                if match.group(3).startswith('5') and match.group(2).startswith('/v1/'):
                    self.counts['http5xx'] += 1
                    out.append(('event', dict(code='http5xx', detail='%s %s %s' % (match.group(1), match.group(2).split('?')[0], match.group(3)))))
        elif ABORTED.search(line):
            self.counts['aborted'] += 1
            self.forget_ids(ids_in(line))
        else:
            for key, marker in MARKERS.items():
                if marker in line:
                    self.counts[key] += 1
                    if key == 'prefix_kill':
                        out.append(('event', dict(code=key, detail=scrub(line.split(marker, 1)[1].strip()))))
                    elif key == 'request_quarantined':
                        quarantined = REQUEST_QUARANTINED.search(line)
                        decoding = DECODING_COUNT.search(line)
                        if quarantined:
                            self.forget_ids([quarantined.group(1)])
                        out.append(('event', dict(code=key, req=request_hash(quarantined.group(1)) if quarantined else None,
                                                  detail='reason in the engine log; decoding=%s' % (decoding.group(1) if decoding else '?'))))
                    break
            else:
                death = death_marker(line)
                if death and death[0] == 'engine':
                    self.counts['engine_death'] += 1
                    out.append(('event', dict(code='engine_death', detail=death[1])))
                elif death:
                    self.counts['native_error'] += 1
                    out.append(('event', dict(code='native_error', detail=death[1])))
        for kind, record in out:
            if kind == 'event':
                record.setdefault('day', self.last[0] if self.last else utc_day())
                record.setdefault('t', self.last[1] if self.last else '')
        return out

    def _stamp(self, match):
        day, clock, moment = stamp_of(match, 1)
        self.first = self.first or (day, clock)
        self.last = (day, clock)
        return day, clock, moment

    def forget_ids(self, ids):
        """Drops the in-flight Lever N prefills named by `ids` (a request id as vLLM lists it can carry a suffix the step line lacks, so either may contain the other)."""
        for wanted in ids:
            if len(wanted) < 8:
                continue
            for request in [key for key in self.requests if key == wanted or wanted in key or key in wanted]:
                self.requests.pop(request, None)
                self.counts['inflight_cleared'] += 1

    def _execute(self, match, out):
        day, clock, moment = self._stamp(match)
        total, new, cached, _spec = (int(value) for value in match.groups()[3:7])
        if match.group(8):
            self.forget_ids(ids_in(match.group(8)))
        self.counts['execute'] += 1
        # a chunk-continuation step of a chunked prefill logs `total=<chunk> new=0 cached=1`: it is a prefill step, not a decode step, whichever image
        # runs (the baseline has no Lever N lines to mark it). More rows per live user than a draft step can carry means a prefill chunk.
        chunk = cached > 0 and total > cached * ROWS_PER_USER_MAX
        decode = new == 0 and cached > 0 and not chunk
        if decode:
            self.counts['decode'] += 1
            self.max_live = max(self.max_live, cached)
            if cached >= 2:
                self.counts['decode_multi'] += 1
        previous = self.current
        after_prefill = bool(previous and (previous['prefill'] or previous['new'] > 0))
        if self.pending is not None and decode:
            seconds = (moment - self.pending['moment']).total_seconds()
            if 0 < seconds <= self.max_round_s:
                step = self.pending
                live = step['live']
                packed = len(step['positions']) == live
                self.counts['rounds'] += 1
                self.counts['rounds_packed'] += int(packed)
                out.append(('round', dict(day=step['day'], t=step['clock'], live=live, rows=step['rows'], ms=round(seconds * 1000.0, 2),
                                          packed=packed, after_prefill=bool(step['after_prefill'] or step['prefill']),
                                          mean_pos=round(sum(step['positions']) / float(live), 1) if packed else None)))
            else:
                self.counts['gaps'] += 1
        self.current = dict(new=new, prefill=chunk)
        if decode:
            self.pending = dict(day=day, clock=clock, moment=moment, live=cached, rows=total, positions=[], after_prefill=after_prefill,
                                prefill=False)
        else:
            self.pending = None

    def _mark_prefill(self):
        if self.current is not None:
            self.current['prefill'] = True
        if self.pending is not None:
            self.pending['prefill'] = True

    def _levern(self, line, out):
        if LEVERN_PREFILL_STEP.search(line):
            self._mark_prefill()
        match = LEVERN_STEP.search(line)
        if match:
            day, clock, moment = self._stamp(match)
            self.counts['levern_steps'] += 1
            kind, request, prompt, final = match.group(5), match.group(7), match.group(11), match.group(12)
            if kind == 'prefill' and request != '-' and prompt.isdigit():
                entry = self.requests.get(request)
                if entry is None:
                    entry = self.requests[request] = dict(day=day, clock=clock, moment=moment, prompt=int(prompt), steps=0)
                    while len(self.requests) > 2000:
                        self.requests.popitem(last=False)
                entry['steps'] += 1
                if final == '1':
                    self.requests.pop(request, None)
                    out.append(('request', dict(day=entry['day'], t=entry['clock'], req=request_hash(request), prompt=entry['prompt'],
                                                steps=entry['steps'], first_to_final_s=round((moment - entry['moment']).total_seconds(), 2))))
            return
        match = LEVERN_QUARANTINE.search(line)
        if match:
            self.counts['levern_quarantine'] += 1
            self.forget_ids([match.group(1)])
            out.append(('event', dict(code='levern_quarantine', req=request_hash(match.group(1)), detail='reason in the engine log')))
            return
        for key in ('levern_installed', 'levern_route', 'levern_kill', 'levern_refused'):
            if MARKERS[key] in line:
                self.counts[key] += 1
                stamped = LINE_STAMP.search(line)
                if stamped:
                    self._stamp(stamped)
                if key in ('levern_kill', 'levern_refused'):
                    out.append(('event', dict(code=key, detail=scrub(line.split('lever N', 1)[1].strip()))))
                else:
                    out.append(('event', dict(code=key, detail=scrub(line.split('lever N', 1)[1].strip(), 80))))
                return

    def inflight(self):
        """{request: dict(moment, prompt, steps)}: Lever N prefills whose final step has not been logged and that have not been aborted, quarantined or finished."""
        return self.requests


class Watcher(object):
    """The log rules (module docstring). `feed(line, now)` and `tick(now)` return the triggers newly raised; `now` is wall-clock seconds.

    A hang is the ABSENCE of a line, so it needs a clock that moves without lines: the capture loop calls tick() when no line arrives. Each
    trigger fires once per episode (a hang until the next execute line; the error rule until the count falls below the threshold; a request once)."""

    def __init__(self, hang_s=HANG_S, error_count=ERROR_COUNT, error_window_s=ERROR_WINDOW_S, ttft_s=TTFT_S, long_prompt=LONG_PROMPT, start=None,
                 on_record=None, live_fresh_s=LIVE_FRESH_S, api_down_s=API_DOWN_S, stale_s=STALE_S, forget_s=FORGET_S,
                 native_confirm_s=NATIVE_CONFIRM_S):
        self.native_confirm_s = native_confirm_s
        self.api_down_s = api_down_s
        self.api_ever_ok = False
        self.api_fail_since = None
        self.api_fired = False
        self.on_record = on_record
        self.live = None              # (requests running + waiting, wall time) from the metrics poll
        self.stats = None             # (busy, wall time) from the engine's stats log line
        self.tokens = None            # (generation tokens counter, wall time it was read) from the metrics poll
        self.tokens_moved = None      # wall time the counter last increased
        self.flow = None              # wall time of the last stats line that showed generation throughput
        self.live_fresh_s = live_fresh_s
        self.stale_s, self.forget_s = stale_s, forget_s
        self.hang_s, self.error_count, self.error_window_s = hang_s, error_count, error_window_s
        self.ttft_s, self.long_prompt = ttft_s, long_prompt
        self.extractor = Extractor()
        self.last_execute = start
        self.silence = None           # the code raised for the current silence: 'T0-HANG' or 'T0-LOG-STALLED'
        self.errors = collections.deque()
        self.errors_armed = True
        self.first_wall = {}
        self.seen = {}                # request -> (steps, wall time the step count last changed)
        self.fired = set()
        self.native = None            # (wall time, marker name) of an unconfirmed native error text (death_marker)

    def set_live(self, count, now=None):
        """The metrics poll's reading: requests running plus waiting, as of `now`."""
        self.live = (int(count), time.time() if now is None else now)

    def set_tokens(self, total, now=None):
        """The metrics poll's generation-token counter: the log-stalled rule's evidence that the engine still works."""
        now = time.time() if now is None else now
        if self.tokens is not None and total > self.tokens[0]:
            self.tokens_moved = now
        self.tokens = (float(total), now)

    def set_api(self, ok, now=None):
        """The metrics poll's outcome: a read that worked (True) or failed (False)."""
        now = time.time() if now is None else now
        if ok:
            self.api_ever_ok, self.api_fail_since, self.api_fired = True, None, False
        elif self.api_fail_since is None:
            self.api_fail_since = now

    def busy(self, now):
        """Is a request live? The metrics reading when it is fresh (a fresh 0 is idle); else the last stats log line (see the module docstring)."""
        if self.live is not None and now - self.live[1] <= self.live_fresh_s:
            return self.live[0] > 0
        if self.stats is not None:
            return self.stats[0]
        return False

    def tokens_flowing(self, now):
        """Has the engine generated tokens in the last minute (counter or stats line), whatever its log says?"""
        if self.tokens_moved is not None and now - self.tokens_moved <= self.live_fresh_s:
            return True
        return self.flow is not None and now - self.flow <= self.live_fresh_s

    def _trigger(self, code, now, **detail):
        record = dict(code=code, at=round(now, 1), action=TIER_ACTION[code])
        record.update(detail)
        return record

    def feed(self, line, now=None):
        now = time.time() if now is None else now
        records = self.extractor.feed(line)
        if self.on_record is not None:
            for kind, record in records:
                self.on_record(kind, record)
        out = []
        if line.startswith(FOLLOW_MARKER):
            self.stats, self.silence, self.last_execute, self.flow, self.native = None, None, now, None, None
            return out
        if self.last_execute is None:
            self.last_execute = now
        if '[PHASE] execute' in line:
            self.last_execute = now
            self.silence = None
            self.native = None        # the engine stepped after the error text: it was recoverable
        elif 'Running:' in line and self.extractor.running is not None:
            self.stats = (bool(self.extractor.running or self.extractor.waiting), now)
            if self.extractor.gen_tps and now - self.last_execute > FLOW_AFTER_S:
                self.flow = now      # tokens were still generated well after the last phase line: only then is a stats line evidence
        for kind, record in records:
            if kind == 'event':
                code = record['code']
                if code == 'engine_death':
                    out.append(self._trigger('T0-ENGINE-DEATH', now, detail=record['detail']))
                elif code == 'native_error':
                    if self.native is None:
                        self.native = (now, record['detail'])
                elif code == 'http5xx':
                    self.errors.append(now)
                    while self.errors and now - self.errors[0] > self.error_window_s:
                        self.errors.popleft()
                    if len(self.errors) >= self.error_count and self.errors_armed:
                        self.errors_armed = False
                        out.append(self._trigger('T0-SERVER-ERRORS', now, count=len(self.errors), window_s=self.error_window_s))
                elif code == 'levern_quarantine':
                    out.append(self._trigger('T1-LEVERN-QUARANTINE', now, req=record['req'], detail=record['detail']))
            elif kind == 'request':
                if record['prompt'] > self.long_prompt and record['first_to_final_s'] > self.ttft_s:
                    out.append(self._trigger('T1-LEVERN-LONG-TTFT', now, req=record['req'], prompt=record['prompt'],
                                             lower_bound_s=record['first_to_final_s'], state='finished'))
        # the in-flight rule's clocks: when each Lever N prefill was first seen and when its step count last moved (wall clock)
        if 'lever N step' in line or 'finished=[' in line or ABORTED.search(line) or 'quarantine' in line:
            inflight = self.extractor.inflight()
            for request, entry in inflight.items():
                self.first_wall.setdefault(request, now)
                seen = self.seen.get(request)
                if seen is None or seen[0] != entry['steps']:
                    self.seen[request] = (entry['steps'], now)
        return out

    def tick(self, now=None):
        now = time.time() if now is None else now
        out = []
        while self.errors and now - self.errors[0] > self.error_window_s:
            self.errors.popleft()
        if len(self.errors) < self.error_count:
            self.errors_armed = True
        if self.api_ever_ok and self.api_fail_since is not None and not self.api_fired and now - self.api_fail_since >= self.api_down_s:
            self.api_fired = True
            out.append(self._trigger('T0-API-UNREACHABLE', now, down_s=round(now - self.api_fail_since, 1)))
        busy = self.busy(now)
        if self.native is not None:
            since, marker = self.native
            if self.api_fail_since is not None or (now - since >= self.native_confirm_s and busy):
                self.native = None
                out.append(self._trigger('T0-ENGINE-DEATH', now, detail=marker, confirmed='api-down' if self.api_fail_since is not None else 'no-step'))
            elif now - since >= self.native_confirm_s:
                self.native = None      # idle and answering: the error text did not stop the engine
                self.extractor.counts['native_unconfirmed'] += 1
        if busy and self.last_execute is not None and now - self.last_execute > self.hang_s:
            code = 'T0-LOG-STALLED' if self.tokens_flowing(now) else 'T0-HANG'
            if self.silence != code and self.silence != 'T0-HANG':
                self.silence = code
                out.append(self._trigger(code, now, silent_s=round(now - self.last_execute, 1), live=self.live[0] if self.live else None))
        inflight = self.extractor.inflight()
        for request in [key for key in self.first_wall if key not in inflight]:
            del self.first_wall[request]
            self.seen.pop(request, None)
        for request, entry in list(inflight.items()):
            first = self.first_wall.get(request)
            moved = self.seen.get(request, (0, first))[1]
            if moved is not None and now - moved > self.forget_s:
                self.extractor.requests.pop(request, None)
                continue
            if (first is not None and moved is not None and entry['prompt'] > self.long_prompt and now - first > self.ttft_s
                    and now - moved <= self.stale_s and request not in self.fired):
                self.fired.add(request)
                out.append(self._trigger('T1-LEVERN-LONG-TTFT', now, req=request_hash(request), prompt=entry['prompt'],
                                         lower_bound_s=round(now - first, 1), state='running'))
        return out


METRIC_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+|NaN)\s*$')
RAW_LIVE = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
RAW_TOKENS = 'vllm:generation_tokens_total'


def live_names(platform_prefix=None):
    """(the gauge names whose sum is 'requests live', the generation-token counter name) for the raw engine metrics and, when a prefix is given, the platform's."""
    platform = (platform_prefix + 'running_requests', platform_prefix + 'queue_depth') if platform_prefix else ()
    return RAW_LIVE, platform, ((platform_prefix + 'generation_tokens_total') if platform_prefix else None)


def parse_metrics(text, platform_prefix=None):
    """(live, tokens) from a /metrics text: requests running plus waiting (raw names, else the platform's), and the generation-token counter; None when absent."""
    raw, platform, platform_tokens = live_names(platform_prefix)
    wanted = raw if RAW_LIVE[0] in text or not platform else platform
    live, tokens, seen_live, seen_tokens = 0.0, 0.0, False, False
    for line in text.split(chr(10)):
        match = METRIC_LINE.match(line)
        if not match or match.group(2) == 'NaN':
            continue
        name = match.group(1)
        if name in wanted:
            live += float(match.group(2))
            seen_live = True
        elif name == RAW_TOKENS or (platform_tokens and name == platform_tokens):
            tokens += float(match.group(2))
            seen_tokens = True
    return (int(live) if seen_live else None), (tokens if seen_tokens else None)


def parse_live(text, platform_prefix=None):
    """Requests running plus waiting from a /metrics text (the raw vLLM names, or the platform's under `platform_prefix`), summed over series; None when neither is present."""
    return parse_metrics(text, platform_prefix)[0]


def current_url(url, url_file):
    """The /metrics URL to read now: the first line of `url_file` when it exists and is not empty (the capture wrapper rewrites it each time it attaches to
    the container, so a published port that changes with a redeploy is followed), else `url`."""
    if url_file:
        try:
            with open(url_file, 'r', encoding='utf-8') as handle:
                text = handle.readline().strip()
            if text:
                return text
        except OSError:
            pass
    return url


def poll_live(url, watcher, every, stop, clock=time.time, state=None, platform_prefix=None, url_file=None):
    """Thread body: read the engine's /metrics every `every` seconds into watcher.set_live. A failed read leaves the last reading to age out."""
    from urllib.request import urlopen
    state = state if state is not None else {}
    while not stop.is_set():
        try:
            with urlopen(current_url(url, url_file), timeout=5) as handle:
                count, tokens = parse_metrics(handle.read().decode('utf-8', 'replace'), platform_prefix)
            watcher.set_api(True, clock())
            if tokens is not None:
                watcher.set_tokens(tokens, clock())
            if count is not None:
                watcher.set_live(count, clock())
                state['ok'] = state.get('ok', 0) + 1
            else:
                state['no_gauge'] = state.get('no_gauge', 0) + 1
        except Exception:
            watcher.set_api(False, clock())
            state['errors'] = state.get('errors', 0) + 1
        stop.wait(every)


def host_load():
    """The host's 1-minute load average, or None where the platform has none."""
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


# ---- output ---------------------------------------------------------------------------------------------------------------------------------------------
class Sink(object):
    """Daily JSONL files in one directory, appended and flushed. Only derived records reach it."""

    FILES = dict(round='rounds', request='requests', event='events')

    def __init__(self, directory, flush_every=200):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.handles = {}
        self.flush_every, self.pending = flush_every, 0
        self.written = collections.Counter()

    def write(self, kind, record):
        name = '%s-%s.jsonl' % (self.FILES[kind], record.get('day') or 'unknown')
        handle = self.handles.get(name)
        if handle is None:
            handle = self.handles[name] = open(os.path.join(self.directory, name), 'a', encoding='utf-8', newline='\n')
        handle.write(json.dumps(dict(k=kind, **record), sort_keys=True, separators=(',', ':')) + '\n')
        self.written[kind] += 1
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self):
        for handle in self.handles.values():
            handle.flush()
        self.pending = 0

    def close(self):
        self.flush()
        for handle in self.handles.values():
            handle.close()
        self.handles = {}


def open_text(path):
    if path == '-':
        return sys.stdin
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def maintenance_active(path, now=None):
    """Is the maintenance marker present and fresh (younger than MAINT_MAX_S)? While it is, the Tier 0 triggers are recorded as suppressed, not raised."""
    if not path:
        return False
    try:
        age = (time.time() if now is None else now) - os.stat(path).st_mtime
    except OSError:
        return False
    return age <= MAINT_MAX_S


def write_prom(path, watcher, by_code, suppressed, maintenance, started, now=None):
    """The capture's own health as Prometheus text (a node_exporter textfile): a heartbeat, a counter per trigger code and the maintenance state.
    Prometheus alerts on a new trigger and on a stale heartbeat, so a log-only rule reaches a person and a dead capture is seen. Atomic replace."""
    if not path:
        return
    now = time.time() if now is None else now
    source = 'metrics' if watcher.live and now - watcher.live[1] <= watcher.live_fresh_s else ('stats-line' if watcher.stats is not None else 'none')
    rows = ['# HELP tt_prod_telemetry_heartbeat_timestamp_seconds Unix time the production telemetry capture last wrote this file (stale = the capture is down).',
            '# TYPE tt_prod_telemetry_heartbeat_timestamp_seconds gauge',
            'tt_prod_telemetry_heartbeat_timestamp_seconds %d' % now,
            '# TYPE tt_prod_telemetry_start_timestamp_seconds gauge',
            'tt_prod_telemetry_start_timestamp_seconds %d' % started,
            '# HELP tt_prod_telemetry_trigger_total Rollback-rule triggers the log watcher raised since this capture started.',
            '# TYPE tt_prod_telemetry_trigger_total counter']
    rows += ['tt_prod_telemetry_trigger_total{code="%s"} %d' % (code, by_code.get(code, 0)) for code in TRIGGER_CODES]
    rows += ['# HELP tt_prod_telemetry_trigger_suppressed_total Tier 0 triggers recorded but not raised because a maintenance marker was present.',
             '# TYPE tt_prod_telemetry_trigger_suppressed_total counter']
    rows += ['tt_prod_telemetry_trigger_suppressed_total{code="%s"} %d' % (code, suppressed.get(code, 0)) for code in TRIGGER_CODES if code.startswith('T0-')]
    rows += ['# HELP tt_prod_telemetry_maintenance 1 while a fresh maintenance marker suppresses the Tier 0 log triggers (planned window work).',
             '# TYPE tt_prod_telemetry_maintenance gauge',
             'tt_prod_telemetry_maintenance %d' % (1 if maintenance else 0),
             '# HELP tt_prod_telemetry_hang_rule_source Where the hang rule reads whether a request is live: metrics gauges, the stats log line, or none.',
             '# TYPE tt_prod_telemetry_hang_rule_source gauge']
    rows += ['tt_prod_telemetry_hang_rule_source{source="%s"} %d' % (name, 1 if name == source else 0) for name in ('metrics', 'stats-line', 'none')]
    rows += ['# TYPE tt_prod_telemetry_api_up gauge', 'tt_prod_telemetry_api_up %d' % (0 if watcher.api_fail_since is not None else (1 if watcher.api_ever_ok else 0))]
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8', newline=chr(10)) as handle:
            handle.write(chr(10).join(rows) + chr(10))
        os.replace(tmp, path)
    except OSError as error:
        print('[prod-telemetry] cannot write %s (%s): no notification path' % (path, type(error).__name__), file=sys.stderr, flush=True)


def write_status(path, watcher, sink, fired, load=None):
    if not path:
        return
    state = dict(first=watcher.extractor.first, last=watcher.extractor.last, counts=dict(watcher.extractor.counts), triggers=fired,
                 written=dict(sink.written) if sink else {}, updated_unix=int(time.time()), load1=load,
                 live=dict(count=watcher.live[0], age_s=round(time.time() - watcher.live[1], 1)) if watcher.live else None,
                 live_source='metrics' if watcher.live and time.time() - watcher.live[1] <= watcher.live_fresh_s else
                 ('stats-line' if watcher.stats is not None else 'none'), stats_line_busy=watcher.stats[0] if watcher.stats else None)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(state, handle, sort_keys=True)
    os.replace(tmp, path)


def run(lines, out_dir=None, watch=True, clock=time.time, idle_tick=None, hang_s=HANG_S, metrics_url=None, metrics_every=10.0, platform_prefix=None,
        load_fn=None, load_every=LOAD_EVERY_S, metrics_url_file=None, prom_file=None, maintenance_file=None):
    """Feeds `lines` through the extractor and the watcher. With `out_dir` the extractor's records go to daily files there and the triggers also
    to triggers.jsonl; the triggers always print to stdout. Returns the extractor's counters.

    `idle_tick` (seconds): when the stream is a pipe that goes quiet, a hang produces no line at all, so a reader thread feeds the lines through a
    queue and the loop calls tick() every `idle_tick` seconds without one. None = tick only when a line arrives (a finite file).

    `load_fn` returns the host's 1-minute load average; it is sampled at most every `load_every` seconds (of `clock`) and stamped on every round record
    as `load1`. Leave it None when the stream is a replay of an old log (the current load says nothing about then).

    `prom_file`: a Prometheus text file (a node_exporter textfile) rewritten every ~30 s and at every trigger, with a heartbeat and a counter per trigger
    code. `maintenance_file`: while this file exists and is younger than MAINT_MAX_S the Tier 0 triggers are written to triggers.jsonl with
    suppressed='maintenance' and counted apart, not raised (planned window work restarts the engine). `metrics_url_file`: see current_url."""
    sink = Sink(out_dir) if out_dir else None
    load_state = dict(value=None, at=None)

    def current_load():
        now = clock()
        if load_fn is not None and (load_state['at'] is None or now - load_state['at'] >= load_every):
            load_state.update(value=load_fn(), at=now)
        return load_state['value']

    def on_record(kind, record):
        if kind == 'round':
            load = current_load()
            if load is not None:
                record['load1'] = load
        sink.write(kind, record)

    watcher = Watcher(start=clock(), on_record=on_record if sink else None, hang_s=hang_s)
    import threading
    stop, poll_state = threading.Event(), {}
    if metrics_url or metrics_url_file:
        threading.Thread(target=poll_live, args=(metrics_url, watcher, metrics_every, stop, clock, poll_state, platform_prefix, metrics_url_file),
                         daemon=True).start()
    triggers_path = os.path.join(out_dir, 'triggers.jsonl') if out_dir else None
    state_path = os.path.join(out_dir, 'capture-state.json') if out_dir else None
    fired, last_status = 0, clock()
    by_code, suppressed = collections.Counter(), collections.Counter()
    started = int(time.time())

    def publish():
        write_prom(prom_file, watcher, by_code, suppressed, maintenance_active(maintenance_file), started)

    def emit(triggers):
        nonlocal fired
        for trigger in triggers:
            if trigger['code'].startswith('T0-') and maintenance_active(maintenance_file):
                trigger['suppressed'] = 'maintenance'
                suppressed[trigger['code']] += 1
            else:
                by_code[trigger['code']] += 1
            fired += 1
            text = json.dumps(trigger, sort_keys=True)
            print(text, flush=True)
            if triggers_path:
                with open(triggers_path, 'a', encoding='utf-8', newline='\n') as handle:
                    handle.write(text + '\n')
        if triggers:
            publish()

    def step(line):
        nonlocal last_status
        now = clock()
        emit(watcher.feed(line, now) + (watcher.tick(now) if watch else []))
        if state_path and now - last_status > 30:
            last_status = now
            sink.flush()          # a record waits at most ~30 s in a buffer: a kill loses seconds, not hundreds of records
            write_status(state_path, watcher, sink, fired, load_state['value'])
            publish()

    publish()      # the heartbeat starts with the capture
    try:
        if idle_tick:
            import queue
            lines_q = queue.Queue(maxsize=10000)
            done = object()

            def reader():
                for line in lines:
                    lines_q.put(line)
                lines_q.put(done)

            threading.Thread(target=reader, daemon=True).start()
            while True:
                try:
                    line = lines_q.get(timeout=idle_tick)
                except queue.Empty:
                    if watch:
                        emit(watcher.tick(clock()))
                    if state_path and clock() - last_status > 30:
                        last_status = clock()
                        sink.flush()
                        write_status(state_path, watcher, sink, fired, load_state['value'])
                        publish()
                    continue
                if line is done:
                    break
                step(line)
        else:
            for line in lines:
                step(line)
    finally:
        stop.set()
        if sink is not None:
            sink.close()
    if state_path:
        write_status(state_path, watcher, sink, fired, load_state['value'])
    return watcher.extractor.counts


# ---- check ----------------------------------------------------------------------------------------------------------------------------------------------
LOG_REQUIRED = (
    ('phase_execute', "'[PHASE] execute' lines (the hang rule's clock and every round)", lambda c: c['execute'] > 0),
    ('decode_steps', 'decode steps (new=0 cached>0)', lambda c: c['decode'] > 0),
    ('rounds_timed', 'rounds the comparator can time (two decode steps in a row)', lambda c: c['rounds'] > 0),
    ('packed_audit', "'[PACKED] request= ... position=' lines (the round's context window)", lambda c: c['packed_lines'] > 0),
    ('rounds_packed', 'rounds with one audit line per live user (mean position known)', lambda c: c['rounds_packed'] > 0),
)
LOG_LEVERN = (
    ('levern_installed', "'[PINDIAG] lever N installed' (the scheduler is Lever N; a boot line: counted over the whole file by the wrapper)",
     lambda c: c['levern_installed'] > 0),
    ('levern_steps', "'[PINDIAG] lever N step' lines (prefill steps: the long-TTFT rule's clock)", lambda c: c['levern_steps'] > 0),
)
LOG_INFO = (
    ('stats_line', "the engine's stats line (vLLM prints it every 10 s while active; the hang rule's fallback when no gauge is exported)",
     lambda c: c['stats'] > 0),
    ('quarantine_consumer', 'request quarantine consumer live', lambda c: c['quarantine_consumer'] > 0),
    ('access_lines', 'HTTP access lines (the 5xx rule)', lambda c: c['access'] > 0),
    ('levern_quarantine', 'Lever N quarantine lines seen (zero is the healthy state; the rule matches this exact line format)', lambda c: c['levern_quarantine'] > 0),
    ('levern_kill', 'Lever N kill-switch line seen', lambda c: c['levern_kill'] > 0),
    ('prefix_kill', 'prefix kill-switch line seen', lambda c: c['prefix_kill'] > 0),
)
# metric names the rules and the Tier 2 reads need (REQUIRED) and names that are nice to have (INFO: absence is reported, never a failure)
RAW_REQUIRED = ('vllm:request_success_total', 'vllm:prompt_tokens_total', 'vllm:prompt_tokens_cached_total', 'vllm:generation_tokens_total',
                'vllm:time_to_first_token_seconds_bucket', 'qwen_prefix_attempts_total', 'qwen_prefix_admissions_total', 'qwen_prefix_grants_total',
                'qwen_prefix_grant_tokens_total')
RAW_INFO = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'qwen_prefix_kill_switch_file', 'qwen_prefix_export_writers')
PLATFORM_REQUIRED_SUFFIXES = ('requests_total', 'prompt_tokens_total', 'cached_prompt_tokens_total', 'generation_tokens_total', 'ttft_seconds_bucket',
                              'prefix_attempts_total', 'prefix_admissions_total', 'prefix_grants_total', 'prefix_grant_tokens_total')
PLATFORM_INFO_SUFFIXES = ('running_requests', 'queue_depth')
METRIC_NAME = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+\S+')


def platform_names(prefix):
    return tuple(prefix + suffix for suffix in PLATFORM_REQUIRED_SUFFIXES), tuple(prefix + suffix for suffix in PLATFORM_INFO_SUFFIXES)


def metric_names(text):
    names = set()
    for line in text.split('\n'):
        match = METRIC_NAME.match(line)
        if match:
            names.add(match.group(1))
    return names


def check_metrics(text, family='auto', platform_prefix=None):
    names = metric_names(text)
    if family == 'auto':
        family = 'platform' if platform_prefix and any(name.startswith(platform_prefix) for name in names) else 'raw'
    if family == 'platform':
        if not platform_prefix:
            raise ValueError('the platform family needs --platform-prefix')
        required, info = platform_names(platform_prefix)
    else:
        required, info = RAW_REQUIRED, RAW_INFO
    gauges = RAW_LIVE if family == 'raw' else live_names(platform_prefix)[1]
    return dict(family=family, present=[name for name in required if name in names], absent=[name for name in required if name not in names],
                info_present=[name for name in info if name in names], info_absent=[name for name in info if name not in names],
                live_gauges=bool(gauges) and all(name in names for name in gauges))


def check_log(lines, expect_levern=False, boot=None, live_gauges=None):
    """The verdict on a log window. `boot` = {counter: n} counted over the whole log file (boot lines scroll out of a tail window); `live_gauges` = whether
    the /metrics scrape carries the running/waiting gauges (None: not scraped)."""
    extractor = Extractor()
    for line in lines:
        extractor.feed(line)
    counts = extractor.counts
    for key, value in (boot or {}).items():
        counts[key] = max(counts[key], int(value))
    rows, absent = [], []
    for group, expected in ((LOG_REQUIRED, True), (LOG_LEVERN, expect_levern), (LOG_INFO, False)):
        for key, text, test in group:
            present = bool(test(counts))
            rows.append(dict(signal=key, present=present, required=expected, what=text))
            if expected and not present:
                absent.append(key)
    source = 'metrics-gauges' if live_gauges else ('stats-line' if counts['stats'] > 0 else None)
    rows.append(dict(signal='hang_live_source', present=source is not None, required=True, source=source,
                     what="where the hang rule reads 'a request is live': fresh /metrics gauges OR at least one stats line in the window"))
    if source is None:
        absent.append('hang_live_source')
    traffic = counts['execute'] > 0 or counts['stats_busy'] > 0
    verdict = 'PASS' if not absent else ('FAIL' if traffic else 'IDLE')
    note = None
    if verdict == 'FAIL' and set(absent) <= {'packed_audit', 'rounds_packed'} and counts['decode_multi'] < MIN_MULTI_STEPS:
        # a packed round (one audit line per live user) exists only with two or more users live: a window in which one user decodes on the single path (a step or two of ramp
        # between users does not make a packed round) cannot show it
        verdict = 'IDLE'
        note = ('%d decode steps with two or more users live in the window (under %d, at most %d live at once): the per-user audit lines exist only for packed rounds; '
                'send two or more concurrent requests and rerun' % (counts['decode_multi'], MIN_MULTI_STEPS, extractor.max_live))
    return dict(verdict=verdict, absent=absent, note=note, max_live=extractor.max_live, window=dict(first=extractor.first, last=extractor.last), counts=dict(counts),
                signals=rows)


def parse_boot(values):
    out = {}
    for value in values or ():
        key, _, number = value.partition('=')
        out[key] = int(number)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    prefix_help = "the platform's metric name prefix (its names are <prefix>running_requests, <prefix>generation_tokens_total, ...); default: the raw engine names only"
    check = sub.add_parser('check', help='which of the rule\'s log lines are present in a log window')
    check.add_argument('log', nargs='?', default='-', help="log file ('-' = stdin, .gz ok)")
    check.add_argument('--expect-levern', action='store_true', help='the image carries Lever N: its lines are required')
    check.add_argument('--boot-count', action='append', default=[], metavar='KEY=N',
                       help='a boot line counted over the whole log file (levern_installed=1): a tail window has lost it')
    check.add_argument('--metrics', help='a /metrics scrape file to check the metric names in')
    check.add_argument('--metrics-url', help='a /metrics URL to scrape (read-only GET) and check the metric names of')
    check.add_argument('--metrics-family', choices=('auto', 'raw', 'platform'), default='auto')
    check.add_argument('--platform-prefix', default=os.environ.get('PROD_TELEMETRY_PLATFORM_PREFIX') or None, help=prefix_help)
    cap = sub.add_parser('capture', help='write derived records and run the watcher')
    cap.add_argument('log', nargs='?', default='-')
    cap.add_argument('--out', required=True, help='directory on this host for rounds-/requests-/events-DAY.jsonl, triggers.jsonl')
    cap.add_argument('--no-watch', action='store_true')
    cap.add_argument('--metrics-url', help="the engine's /metrics URL: the hang rule's 'a request is live' (without it the hang rule rests on the stats line)")
    cap.add_argument('--metrics-every', type=float, default=10.0)
    cap.add_argument('--metrics-url-file', help="a file whose first line is the /metrics URL, re-read on every poll (the wrapper rewrites it when the published port changes)")
    cap.add_argument('--prom-file', help='write the capture heartbeat and trigger counters here as Prometheus text (a node_exporter textfile): the notification path')
    cap.add_argument('--maintenance-file', help='while this file exists (and is under 12 h old) the Tier 0 triggers are recorded as suppressed, not raised')
    cap.add_argument('--platform-prefix', default=os.environ.get('PROD_TELEMETRY_PLATFORM_PREFIX') or None, help=prefix_help)
    cap.add_argument('--stamp-load', action='store_true', help='stamp the host load on rounds even when reading a file (a replay: the stamp is the load NOW)')
    watch = sub.add_parser('watch', help='the watcher alone')
    watch.add_argument('log', nargs='?', default='-')
    watch.add_argument('--metrics-url')
    watch.add_argument('--metrics-every', type=float, default=10.0)
    watch.add_argument('--platform-prefix', default=os.environ.get('PROD_TELEMETRY_PLATFORM_PREFIX') or None, help=prefix_help)
    san = sub.add_parser('sanitise', help='filter a log excerpt: telemetry lines only, token ids and addresses removed')
    san.add_argument('log', nargs='?', default='-')
    san.add_argument('--keep-all', action='store_true', help='strip ids and addresses but keep every line')
    args = parser.parse_args(argv)
    stream = open_text(args.log)
    try:
        return dispatch(args, stream)
    finally:
        if stream is not sys.stdin:
            stream.close()


def dispatch(args, stream):
    if args.command == 'check':
        metrics_text = None
        if args.metrics:
            with open(args.metrics, 'r', encoding='utf-8', errors='replace') as handle:
                metrics_text = handle.read()
        elif args.metrics_url:
            from urllib.request import urlopen
            try:
                with urlopen(args.metrics_url, timeout=15) as handle:
                    metrics_text = handle.read().decode('utf-8', 'replace')
            except (OSError, ValueError) as error:
                print('the metrics URL did not answer (%s): nothing to check yet' % type(error).__name__, file=sys.stderr)
                return 4
        metrics = check_metrics(metrics_text, args.metrics_family, args.platform_prefix) if metrics_text is not None else None
        report = check_log(stream, args.expect_levern, parse_boot(args.boot_count), metrics['live_gauges'] if metrics else None)
        if metrics is not None:
            report['metrics'] = metrics
            if metrics['absent']:
                report['verdict'] = 'FAIL' if report['verdict'] != 'IDLE' else 'IDLE'
        print(json.dumps(report, indent=1, sort_keys=True))
        return {'PASS': 0, 'FAIL': 2, 'IDLE': 3}[report['verdict']]
    if args.command in ('capture', 'watch'):
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))    # a stop closes the files (the sink's flush) instead of dropping the last records
    if args.command == 'capture':
        run(stream, args.out, watch=not args.no_watch, idle_tick=None if args.log != '-' else 5.0, metrics_url=args.metrics_url,
            metrics_every=args.metrics_every, platform_prefix=args.platform_prefix, load_fn=host_load if (args.log == '-' or args.stamp_load) else None,
            metrics_url_file=args.metrics_url_file, prom_file=args.prom_file, maintenance_file=args.maintenance_file)
        return 0
    if args.command == 'watch':
        run(stream, None, watch=True, idle_tick=None if args.log != '-' else 5.0, metrics_url=args.metrics_url, metrics_every=args.metrics_every,
            platform_prefix=args.platform_prefix)
        return 0
    for line in sanitise_stream(stream, args.keep_all):
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
