"""The W-T1 tau lab (Linear THA-268 tooling, THA-269 the run): Qwen3.8-27B's own greedy answers on agent turns, through
the PRODUCTION arithmetic on four cards, one container load for every arm (design: tau-on-tt/design.md section 1).

    python3 scripts/ci/c2_tau_lab.py --image zot.thatch.local:5000/tt-vllm:qwen38-c2-tp4-serve-2 \
        --profile c2-packed-tp4 --data ~/kwork64/taulab/data --results ~/kwork64/taulab/results/<run> \
        --public "$RUNNER_TEMP/c2-results/taulab" --deadline-seconds 16200

Run by qwen-c2-serving.yml's 'taulab' action (C2_CARDS=quad), after the window's other jobs. Host side: this file starts ONE
container through c2_prefix_gate.server_run (the node agent's shape) serving a DERIVED profile - the production profile with
exactly two arithmetic-neutral log flags added (QWEN_FAST_PACKED_AUDIT=1: one [PACKED] line per user per round;
QWEN_FAST_PHASE_TIMING=1: one fast_serving_phases record per request; real_text_compare.ARITHMETIC_NEUTRAL) - and drives the
arms over its OpenAI API from this host. The data directory is read (never written, never mounted into the model container,
which has no use for it); every private output goes to --results, and only AGGREGATES go to --public (what the workflow
uploads: the Qwen repo is PUBLIC, so its Actions logs and artifacts are public). Stdout carries counts and aggregates only.

THE ARMS (order, default concurrency; design table in section 1.1)
  A1  SWE-rebench held-out turns            240 turns, thinking ON, independent turns, IN_FLIGHT at once
  A2  our own Claude Code / Codex traces     96 turns, thinking ON, independent turns (rig-local data only)
  A3  calibration: the lanes 4 x 4k + 4 x 32k prompts, two waves of 4 (the lanes' four seats), must reproduce the logged
      v162/v179 per-prompt tau within +/-7% (tau_lab_report). Thinking as the prompts bake it (OFF: the references were measured
      thinking-off, so a thinking-ON calibration could not calibrate anything)
  A4  chained synthetic agent sessions       20 sessions x 6 turns, thinking ON, the model's own answers fed back as the
      assistant turns, IN_FLIGHT sessions at once
  A5  the thinking-OFF pair of the first 100 turns of A1's pre-registered order (same turns, paired)
THINKING. Production agents call Qwen3.8 with thinking ON, which is the chat template's default: an ON turn carries NO
chat_template_kwargs (the smoke's stream_reasoning sends none either); an OFF turn carries {"enable_thinking": false}.
A record with prompt_ids is already rendered, so its thinking mode is baked (A5 needs prompt_ids_nothink beside it).

DATA FORMAT (version 1; the data agent's job is to match it - FORMAT.driver.md beside the data says the same)
  <data>/manifest.json (optional) = {"format": 1, "seed": int, "arms": {"A1": {"file": "A1.jsonl", "max_tokens": int,
      "in_flight": int}, ...}, "a3_reference": {"pooled_tau": float, "source": "v162/v179"},
      "think_end_token_id": int, "tool_call_token_id": int}
      Without a manifest the files are <data>/<ARM>.jsonl (A5.jsonl is optional: absent, A5 re-sends A1's first 100).
  One JSON object per line (every field optional unless marked):
      id *        opaque, [A-Za-z0-9_.:-]{1,64}, unique within the arm (never printed)
      set         swe | own | chained | calib (default: the arm's)
      cluster     the conversation id the bootstrap resamples (default: id); turns of one conversation share it
      turn        the turn number within the conversation
      tokens      the prompt's token count where known (the driver skips a prompt past the profile's limit)
      ONE prompt form:
        messages *  [{"role", "content", ...}]  (+ "tools": [...]): sent to /v1/chat/completions, the server renders it
        prompt_ids *  [int]  (+ "prompt_ids_nothink"): sent to /v1/completions as token ids, already rendered
      A4 session records: messages (the first turn's) + "followups": [user text of turns 2..N]: the driver appends the
          model's answer (content, thinking stripped as the template does) and the next follow-up each turn
      A3 records: ref_tau (the logged per-prompt tau), wave (0, 1; default: sorted by tokens, 4 to a wave)
      max_tokens  per record (else the arm's, else --max-tokens)
  Nothing in the data directory is ever printed, committed or uploaded.

RESULTS (private, under --results; mode 0700): turns.jsonl (one line per turn, status/usage/chunk counts/ids - no text),
outputs.jsonl (the output token ids per turn; an A4 turn's answer text, which its next turn is built from), server.log
(docker logs of the one container, whole), run.json, profiles.json, docker-run.json and, from tau_lab_report, tapes.jsonl and
report.private.json. A killed run RESUMES: turns already ok in turns.jsonl are skipped (and an A4 session continues from its
stored answers). --analyze-only reruns the report over a finished results directory.

Python 3.7 syntax, stdlib only: it runs on the rig host.
"""
import argparse
import copy
import http.client
import json
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_prefix_gate as pg  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import prefix_replay as replay  # noqa: E402
import tau_lab_report as report  # noqa: E402

ARMS = ('A1', 'A2', 'A3', 'A4', 'A5')
FORMAT_VERSION = 1
LOG_FLAGS = (('QWEN_FAST_PACKED_AUDIT', '1'), ('QWEN_FAST_PHASE_TIMING', '1'))
PRODUCTION_TAG = 'tp4-serve-2'
CONTAINER = 'qwen-c2-taulab'
PORT = 8022
SEED = 20261001
A5_TURNS = 100
READINESS_SECONDS = 1800
RESERVE_SECONDS = 900            # what the stop, the log and the report need after the last request
SETTLE_SECONDS = 5.0
REQUEST_READ_TIMEOUT = 1500      # no byte for this long: the turn fails (a queued request still gets bytes from nothing, so
                                 # the timeout is on the whole wait: see send_turn's deadline)
DEFAULT_IN_FLIGHT = 8            # 4 seats + 4 queued: keeps the packed block full while one seat refills
DEFAULT_MAX_TOKENS = 768
ARM_SPEC = {
    # kind, default set, thinking (True/False/None = as baked), design wall weight (minutes), default max_tokens
    'A1': dict(kind='turns', set='swe', thinking=True, weight=80, max_tokens=DEFAULT_MAX_TOKENS),
    'A2': dict(kind='turns', set='own', thinking=True, weight=49, max_tokens=DEFAULT_MAX_TOKENS),
    'A3': dict(kind='calib', set='calib', thinking=None, weight=3, max_tokens=1024),
    'A4': dict(kind='chain', set='chained', thinking=True, weight=44, max_tokens=2048),
    'A5': dict(kind='pair', set='swe', thinking=False, weight=28, max_tokens=DEFAULT_MAX_TOKENS),
}
ID = re.compile(r'[A-Za-z0-9_.:-]{1,64}')
SETS = ('swe', 'own', 'chained', 'calib')
# The data agent scrubs; this is the lab's own last guard, FAIL CLOSED: a record that still matches a detector is dropped (never
# repaired, never printed: only the category and its count are). Transcripts carry bearer tokens, passwords, keys, addresses.
SECRET_PATTERNS = (
    ('platform_token', r'thatch_(?:sess|sk)_[A-Za-z0-9_-]{6,}'),
    ('api_key', r'\bsk-[A-Za-z0-9_-]{16,}'),
    ('github_token', r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    ('cloud_key', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    ('chat_token', r'\bxox[abprs]-[A-Za-z0-9-]{10,}'),
    ('jwt', r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}'),
    ('private_key', r'-----BEGIN [A-Z ]*PRIVATE KEY'),
    ('bearer', r'(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}'),
    ('password_arg', r'(?i)(?:\s-pw|--password|--passwd)[ =]+\S+'),
    ('password_literal', r"""(?i)\b(?:password|passwd|pwd)\b\s*[=:]\s*["'][^"'\s]{4,}["']"""),
    ('secret_literal', r"""(?i)\b(?:secret|api[_-]?key|access[_-]?key)\b\s*[=:]\s*["'][A-Za-z0-9_/+=-]{12,}["']"""),
    ('ipv4', r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])'),
    ('email', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
)
SECRETS = [(name, re.compile(pattern)) for name, pattern in SECRET_PATTERNS]


class LabError(ValueError):
    """A lab the driver refuses before any container starts."""


# -- the data --------------------------------------------------------------------------------------------------------

def read_manifest(data):
    path = os.path.join(data, 'manifest.json')
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get('format', FORMAT_VERSION) != FORMAT_VERSION:
        raise LabError('manifest.json is not format %d' % FORMAT_VERSION)
    return manifest


def strings_of(value):
    """Every string in a JSON value (keys and leaves), so a quote or a newline is scanned as the text has it, not as JSON escapes it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            for text in strings_of(item):
                yield text
    elif isinstance(value, (list, tuple)):
        for item in value:
            for text in strings_of(item):
                yield text


def secret_categories(record, extra=()):
    """The categories of detector (SECRET_PATTERNS, then the manifest's own `deny` regexes, named 'deny') the record matches:
    [str], [] for a clean one. Never the matched text."""
    text = chr(10).join(strings_of(record))
    found = [name for name, pattern in SECRETS if pattern.search(text)]
    if any(pattern.search(text) for pattern in extra):
        found.append('deny')
    return found


def compile_deny(manifest):
    patterns = []
    for text in (manifest or {}).get('deny') or ():
        try:
            patterns.append(re.compile(text))
        except (re.error, TypeError):
            raise LabError('manifest.json: a deny pattern is not a regular expression')
    return patterns


def read_records(path, arm, default_set, dropped=None, deny=()):
    """The arm's records from one JSONL file, validated: [dict]. Errors name the line number only, never a value. A record
    matching a secret detector is dropped and counted in `dropped` ({category: n}), never kept."""
    records, seen = [], set()
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                raise LabError('%s line %d is not JSON' % (os.path.basename(path), number))
            if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not ID.fullmatch(record['id']):
                raise LabError('%s line %d: id must match %s' % (os.path.basename(path), number, ID.pattern))
            if record['id'] in seen:
                raise LabError('%s line %d: a repeated id' % (os.path.basename(path), number))
            seen.add(record['id'])
            has_messages, has_ids = isinstance(record.get('messages'), list), isinstance(record.get('prompt_ids'), list)
            if has_messages == has_ids:
                raise LabError('%s line %d: exactly one of messages and prompt_ids' % (os.path.basename(path), number))
            if record.get('set', default_set) not in SETS:
                raise LabError('%s line %d: set must be one of %s' % (os.path.basename(path), number, ', '.join(SETS)))
            if arm == 'A4' and not isinstance(record.get('followups', []), list):
                raise LabError('%s line %d: followups must be a list' % (os.path.basename(path), number))
            found = secret_categories(record, deny)
            if found:
                if dropped is not None:
                    for category in found:
                        dropped[category] = dropped.get(category, 0) + 1
                    dropped['records'] = dropped.get('records', 0) + 1
                continue
            record.setdefault('set', default_set)
            record.setdefault('cluster', record['id'])
            records.append(record)
    return records


def load_data(data, arms, dropped=None):
    """({arm: [records]}, manifest, {arm: file name}): what the data directory holds for the arms asked. A missing arm file
    is an empty arm (counted, reported, never fatal): the lab runs what it has. A5 with no file of its own is derived."""
    manifest = read_manifest(data)
    entries = manifest.get('arms') or {}
    deny = compile_deny(manifest)
    loaded, files = {}, {}
    for arm in arms:
        entry = entries.get(arm) or {}
        name = entry.get('file') or '%s.jsonl' % arm
        if os.path.basename(name) != name:
            raise LabError('%s: the file must sit in the data directory itself' % arm)
        path = os.path.join(data, name)
        files[arm] = name
        mine = dropped.setdefault(arm, {}) if dropped is not None else None
        loaded[arm] = read_records(path, arm, ARM_SPEC[arm]['set'], mine, deny) if os.path.isfile(path) else []
    return loaded, manifest, files


def order_records(records, seed, arm):
    """The pre-registered random order: a seeded shuffle of the ids sorted, so the order depends on the ids and the seed only."""
    ordered = sorted(records, key=lambda record: record['id'])
    random.Random('%s:%s' % (seed, arm)).shuffle(ordered)
    return ordered


def pair_records(a1_order, own, count=A5_TURNS):
    """A5's records: its own file when the data has one, else the first `count` of A1's order (the same turns, paired)."""
    return list(own) if own else list(a1_order[:count])


def calibration_waves(records):
    """A3's two waves of four (the lanes' four seats): by the record's `wave`, else sorted by prompt size in fours."""
    if records and all('wave' in record for record in records):
        waves = {}
        for record in records:
            waves.setdefault(record['wave'], []).append(record)
        return [waves[key] for key in sorted(waves)]
    ordered = sorted(records, key=lambda record: (record.get('tokens') or 0, record['id']))
    return [ordered[at:at + 4] for at in range(0, len(ordered), 4)]


# -- the profile and the container -----------------------------------------------------------------------------------

def derive_profile(profiles, base):
    """(derived name, a profiles document holding only it): `base` with the two log flags. Refuses a base the lab cannot
    mean: not a four-card profile, a gate-only one, or one that already sets a flag to something else."""
    names = profiles['profiles']
    if base not in names:
        raise LabError('profile %s is not one the image carries' % base)
    profile = copy.deepcopy(names[base])
    if profile.get('mesh_device') != 'P150x4':
        raise LabError('profile %s does not open the four-card mesh' % base)
    if profile.get('gate_only') is True:
        raise LabError('profile %s is gate-only: the lab serves the production profile' % base)
    env = profile.setdefault('env', {})
    for name, value in LOG_FLAGS:
        if str(env.get(name, value)) != value:
            raise LabError('profile %s sets %s to %s, not %s' % (base, name, env[name], value))
        env[name] = value
    derived = '%s+taulab' % base
    profile['description'] = '%s, derived by c2_tau_lab: %s' % (base, ', '.join('%s=%s' % pair for pair in LOG_FLAGS))
    return derived, dict(default=derived, profiles={derived: profile})


def arithmetic_diff(profiles, base, derived_document):
    """The env names the derived profile adds or changes that real_text_compare does not list as arithmetic-neutral:
    [] is the design's 'derived profile's arithmetic diff is empty'."""
    import real_text_compare as compare
    before = {key: str(value) for key, value in (profiles['profiles'][base].get('env') or {}).items()}
    document = next(iter(derived_document['profiles'].values()))
    after = {key: str(value) for key, value in (document.get('env') or {}).items()}
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    return [key for key in changed
            if key not in compare.ARITHMETIC_NEUTRAL and not key.endswith(compare.ARITHMETIC_NEUTRAL_SUFFIXES)]


def image_tag(image):
    return image.rsplit(':', 1)[-1] if ':' in image.rsplit('/', 1)[-1] else ''


def is_production_image(image):
    """The design's label: the production image tp4-serve-2 (the tag ends with it); any other image's tau is labelled
    non-production in the report."""
    return image_tag(image).endswith(PRODUCTION_TAG)


# -- the requests ----------------------------------------------------------------------------------------------------

def request_body(job, thinking, max_tokens, model):
    """(path, body) for one turn. Greedy, streamed, with the continuous usage stats the third source needs and the token
    ids the tapes need. Thinking ON sends no chat_template_kwargs (the template's default is what production agents get)."""
    options = dict(include_usage=True, continuous_usage_stats=True)
    common = dict(model=model, max_tokens=int(max_tokens), temperature=0.0, top_p=1.0, stream=True,
                  stream_options=options, return_token_ids=True)
    if job.get('messages') is not None:
        body = dict(common, messages=job['messages'])
        if job.get('tools'):
            body['tools'] = job['tools']
            body['tool_choice'] = 'auto'
        if thinking is False:
            body['chat_template_kwargs'] = dict(enable_thinking=False)
        return '/v1/chat/completions', body
    ids = job.get('prompt_ids_nothink') if thinking is False and job.get('prompt_ids_nothink') else job['prompt_ids']
    return '/v1/completions', dict(common, prompt=ids)


class Chunks(object):
    """One streamed completion, fed server-sent-event lines: its id, the completion tokens per chunk (the third source),
    the kind of each chunk (reasoning, content or tool call, from the chat deltas), the output token ids, the usage and the
    finish reason. Never keeps text except `content` (the A4 chain's answer, which the next turn is built from)."""

    def __init__(self, keep_content=False, clock=time.time):
        self.clock, self.keep_content = clock, keep_content
        self.started = clock()
        self.first_at = None
        self.request_id = None
        self.chunk_tokens, self.kinds, self.token_ids = [], [], []
        self.content, self.finish, self.error, self.usage, self.done = [], None, None, None, False
        self.reported = 0

    def feed(self, raw):
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith('data:'):
            return self.done
        body = line[5:].strip()
        if body == '[DONE]':
            self.done = True
            return True
        try:
            chunk = json.loads(body)
        except ValueError:
            return False
        if chunk.get('error') is not None:
            self.error = 'server error'
            self.done = True
            return True
        self.request_id = self.request_id or chunk.get('id')
        usage = chunk.get('usage')
        if usage:
            self.usage = usage
        total = usage.get('completion_tokens') if usage else None
        for choice in chunk.get('choices') or ():
            ids = choice.get('token_ids') or []
            delta = choice.get('delta') or {}
            text = delta.get('content') or choice.get('text') or ''
            thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
            calls = delta.get('tool_calls')
            self.token_ids.extend(ids)
            kind = 't' if calls else ('r' if thought and not text else 'c')
            if self.keep_content and text and not calls:
                self.content.append(text)
            # The continuous usage stats say how many tokens this chunk brought; without them the ids do.
            step = (total - self.reported) if total is not None else len(ids)
            if step > 0:
                if self.first_at is None:
                    self.first_at = self.clock()
                self.chunk_tokens.append(step)
                self.kinds.append(kind)
                self.reported += step
            if choice.get('finish_reason'):
                self.finish = choice['finish_reason']
        return False

    def result(self):
        tool_at = next((sum(self.chunk_tokens[:at]) for at, kind in enumerate(self.kinds) if kind == 't'), None)
        content_at = next((sum(self.chunk_tokens[:at]) for at, kind in enumerate(self.kinds) if kind != 'r'), None)
        return dict(request_id=self.request_id, chunk_tokens=self.chunk_tokens, kinds=''.join(self.kinds),
                    think_tokens=content_at if self.kinds and 'r' in self.kinds else 0, tool_at=tool_at,
                    finish=self.finish, error=self.error, usage=self.usage, output_ids=self.token_ids,
                    content=''.join(self.content), ttft=None if self.first_at is None else self.first_at - self.started)


def http_stream(port, path, body, timeout, host='127.0.0.1', keep_content=False, clock=time.time, deadline=None):
    """One streamed request to the container: the Chunks result plus status. The whole wait is bounded by `timeout` seconds
    (a queued request gets no bytes while it waits, so it is not a per-read bound alone)."""
    state = Chunks(keep_content, clock)
    connection = http.client.HTTPConnection(host, int(port), timeout=min(timeout, REQUEST_READ_TIMEOUT))
    status = None
    try:
        connection.request('POST', path, body=json.dumps(body).encode('utf-8'),
                           headers={'content-type': 'application/json'})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            response.read()
            state.error = 'HTTP %s' % status
        else:
            ends = clock() + timeout
            while True:
                raw = response.readline()
                if not raw or state.feed(raw):
                    break
                if clock() > ends or (deadline is not None and clock() > deadline):
                    state.error = 'turn deadline'
                    break
    except (OSError, http.client.HTTPException, ValueError) as error:
        state.error = 'connection: %s' % type(error).__name__
    finally:
        connection.close()
    result = state.result()
    result['status'] = status
    return result


def turn_ok(result):
    return result.get('error') is None and result.get('finish') in ('stop', 'length', 'tool_calls') \
        and bool(result.get('chunk_tokens'))


# -- the lab ---------------------------------------------------------------------------------------------------------

class Results(object):
    """The private results directory: append-only turns.jsonl and outputs.jsonl, read back on resume."""

    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        self.turns_path = os.path.join(directory, 'turns.jsonl')
        self.outputs_path = os.path.join(directory, 'outputs.jsonl')
        self.done = self._read_done()

    def _lines(self, path):
        if not os.path.isfile(path):
            return []
        out = []
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def _read_done(self):
        return dict(((turn['arm'], turn['id']), turn) for turn in self._lines(self.turns_path) if turn.get('status') == 'ok')

    def outputs(self):
        return dict(((entry['arm'], entry['id']), entry) for entry in self._lines(self.outputs_path))

    def add(self, turn, output):
        with self.lock:
            with open(self.turns_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(turn, sort_keys=True) + '\n')
            with open(self.outputs_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(output, sort_keys=True) + '\n')
            if turn.get('status') == 'ok':
                self.done[(turn['arm'], turn['id'])] = turn


class Lab(object):
    """Runs the arms against one served container. `stream` (path, body, timeout) -> result dict, `clock`, `sleep` and `say`
    are injectable for the CPU tests."""

    def __init__(self, results, stream, limits, seed=SEED, in_flight=DEFAULT_IN_FLIGHT, max_tokens=None, model=replay.SERVED_NAME,
                 clock=time.time, sleep=time.sleep, say=print):
        self.results, self.stream, self.limits = results, stream, limits
        self.seed, self.in_flight, self.max_tokens, self.model = seed, in_flight, max_tokens, model
        self.clock, self.sleep, self.say = clock, sleep, say
        self.counts = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()

    # one turn
    def send(self, arm, job, thinking, max_tokens, ends, turn=None, extra=None, keep_content=False):
        """One turn: -> (turn record, output record). Never raises on a request failure: the turn is recorded failed."""
        spec = ARM_SPEC[arm]
        path, body = request_body(job, thinking, max_tokens, self.model)
        started = self.clock()
        timeout = max(60.0, ends - started + 600.0)
        try:
            result = self.stream(path, body, timeout, keep_content)
        except Exception as error:  # noqa: BLE001 - one failed request must not take its worker thread (and its seat) with it
            result = dict(error='client: %s' % type(error).__name__, chunk_tokens=[], output_ids=[], status=None)
        usage = result.get('usage') or {}
        turn_record = dict(arm=arm, id=job['id'], set=job.get('set', spec['set']), cluster=job.get('cluster', job['id']),
                           turn=turn if turn is not None else job.get('turn'), declared_tokens=job.get('tokens'),
                           thinking=thinking, max_tokens=int(max_tokens), path=path,
                           status='ok' if turn_ok(result) else 'error', finish=result.get('finish'),
                           error=result.get('error'), http=result.get('status'),
                           request_id=result.get('request_id'), chunk_tokens=result.get('chunk_tokens'),
                           kinds=result.get('kinds'), think_tokens=result.get('think_tokens'), tool_at=result.get('tool_at'),
                           prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                           ttft=result.get('ttft'), seconds=round(self.clock() - started, 3),
                           ref_tau=job.get('ref_tau'), started=round(started, 3))
        if extra:
            turn_record.update(extra)
        output = dict(arm=arm, id=job['id'], output_ids=result.get('output_ids') or [])
        if keep_content:
            output['content'] = result.get('content') or ''
        self.results.add(turn_record, output)
        return turn_record, output

    def fits(self, job, max_tokens):
        limit = self.limits.get('prompt')
        declared = job.get('tokens')
        return limit is None or declared is None or declared <= limit

    def bump(self, arm, key, amount=1):
        with self.lock:
            self.counts.setdefault(arm, dict(planned=0, ok=0, error=0, skipped=0, resumed=0, refused=0))[key] += amount

    def tally(self, arm, turn):
        self.bump(arm, 'ok' if turn['status'] == 'ok' else 'error')

    # an arm
    def arm_max_tokens(self, arm, entry):
        return int(self.max_tokens or entry.get('max_tokens') or ARM_SPEC[arm]['max_tokens'])

    def run_pool(self, arm, jobs, work, workers, ends):
        """Run `work(job)` for every job on `workers` threads, taking no new job after `ends`."""
        todo = queue.Queue()
        for job in jobs:
            todo.put(job)

        def loop():
            while not self.stop.is_set():
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                if self.clock() >= ends:
                    self.bump(arm, 'skipped')
                    continue
                work(job)

        threads = [threading.Thread(target=loop) for _ in range(max(1, min(workers, len(jobs) or 1)))]
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()

    def run_turns(self, arm, records, thinking, entry, ends):
        max_tokens = self.arm_max_tokens(arm, entry)
        jobs = []
        for record in records:
            self.bump(arm, 'planned')
            if (arm, record['id']) in self.results.done:
                self.bump(arm, 'resumed')
            elif not self.fits(record, max_tokens):
                self.bump(arm, 'refused')
            elif thinking is False and record.get('messages') is None and not record.get('prompt_ids_nothink'):
                self.bump(arm, 'refused')
            else:
                jobs.append(record)

        def work(job):
            self.tally(arm, self.send(arm, job, thinking, max_tokens, ends)[0])

        self.run_pool(arm, jobs, work, int(entry.get('in_flight') or self.in_flight), ends)

    def run_calibration(self, arm, records, entry, ends):
        max_tokens = self.arm_max_tokens(arm, entry)
        for wave in calibration_waves(records):
            jobs = []
            for record in wave:
                self.bump(arm, 'planned')
                if (arm, record['id']) in self.results.done:
                    self.bump(arm, 'resumed')
                elif not self.fits(record, max_tokens):
                    self.bump(arm, 'refused')
                else:
                    jobs.append(record)
            self.run_pool(arm, jobs, lambda job: self.tally(arm, self.send(arm, job, None if 'prompt_ids' in job else False,
                                                                        job.get('max_tokens') or max_tokens, ends)[0]),
                          len(jobs), ends)

    def run_chains(self, arm, records, entry, ends):
        max_tokens = self.arm_max_tokens(arm, entry)
        stored = self.results.outputs()

        def work(session):
            messages = copy.deepcopy(session['messages'])
            followups = list(session.get('followups') or [])
            for turn in range(1 + len(followups)):
                turn_id = '%s.t%d' % (session['id'], turn)
                self.bump(arm, 'planned')
                if (arm, turn_id) in self.results.done:
                    self.bump(arm, 'resumed')
                    messages.append(dict(role='assistant', content=(stored.get((arm, turn_id)) or {}).get('content', '')))
                else:
                    if self.clock() >= ends or self.stop.is_set():
                        self.bump(arm, 'skipped')
                        return
                    job = dict(id=turn_id, set='chained', cluster=session['cluster'], messages=copy.deepcopy(messages),
                               tools=session.get('tools'))
                    record, output = self.send(arm, job, True, max_tokens, ends, turn=turn, keep_content=True)
                    self.tally(arm, record)
                    if record['status'] != 'ok':
                        return          # the chain cannot go on without its answer
                    messages.append(dict(role='assistant', content=output.get('content', '')))
                if turn < len(followups):
                    messages.append(dict(role='user', content=followups[turn]))

        self.run_pool(arm, records, work, int(entry.get('in_flight') or self.in_flight), ends)

    def run(self, arms, data, manifest, send_ends):
        """Every arm in order; each gets the time left in proportion to the design's wall weights of the arms still to run,
        so a slow arm cannot starve the later ones, and time an arm does not use goes to the next. -> {arm: counts}."""
        entries = manifest.get('arms') or {}
        a1_order = order_records(data.get('A1') or [], self.seed, 'A1')
        pending = [arm for arm in arms]
        for arm in arms:
            entry = entries.get(arm) or {}
            now = self.clock()
            left = max(0.0, send_ends - now)
            weight = sum(ARM_SPEC[name]['weight'] for name in pending)
            ends = now + left * ARM_SPEC[arm]['weight'] / float(weight)
            pending.remove(arm)
            records = data.get(arm) or []
            started = self.clock()
            self.counts.setdefault(arm, dict(planned=0, ok=0, error=0, skipped=0, resumed=0, refused=0))
            kind = ARM_SPEC[arm]['kind']
            if arm == 'A5':
                records = pair_records(a1_order, records)
                self.run_turns(arm, records, False, entry, ends)
            elif kind == 'turns':
                self.run_turns(arm, order_records(records, self.seed, arm), True, entry, ends)
            elif kind == 'calib':
                self.run_calibration(arm, records, entry, ends)
            else:
                self.run_chains(arm, order_records(records, self.seed, arm), entry, ends)
            self.say('[TAULAB] arm %s: %s in %.0f s of %.0f' % (arm, ' '.join('%s=%d' % pair for pair in sorted(
                self.counts[arm].items())), self.clock() - started, ends - started))
            self.sleep(SETTLE_SECONDS)
            if self.stop.is_set():
                break
        return self.counts


# -- the container and the job ---------------------------------------------------------------------------------------

def docker_run(arguments, timeout=600):
    try:
        result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return result.returncode, (result.stdout or b'').decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError) as error:
        return None, repr(error)[:300]


def wait_ready(client, container, clock, sleep, seconds=READINESS_SECONDS):
    ends = clock() + seconds
    while clock() < ends:
        if not container.running():
            return 'the container exited before its API answered'
        if client.ready():
            return None
        sleep(10)
    return 'no /v1/models answer in %d s' % seconds


def launched_problems(log_lines, derived):
    """What the launched container's own log says against what was asked (memory read-the-launched-argv): the contract's
    '[QWEN-C2]' lines must name the derived profile. [str]."""
    text = '\n'.join(log_lines)
    problems = []
    if derived not in text:
        problems.append('the container log never names the derived profile %s' % derived)
    return problems


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', default=None)
    parser.add_argument('--profile', default='c2-packed-tp4')
    parser.add_argument('--data', required=True, help='the rig-local data directory (read only)')
    parser.add_argument('--results', required=True, help='the private results directory (rig-local)')
    parser.add_argument('--public', default=None, help='where the aggregates-only summary goes (the uploaded artifact)')
    parser.add_argument('--arms', default=' '.join(ARMS))
    parser.add_argument('--deadline-seconds', type=int, default=16200,
                        help='the whole run, the container load included; the last %d s are the stop and the report'
                        % RESERVE_SECONDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--in-flight', type=int, default=DEFAULT_IN_FLIGHT)
    parser.add_argument('--max-tokens', type=int, default=None, help='every arm\'s answer budget (default per arm)')
    parser.add_argument('--counters', default=None, help='production\'s spec-decode counters (a JSON file; aggregates only)')
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--cards', choices=('quad',), default='quad')
    parser.add_argument('--hub', default=gate.HUB)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--dry-run', action='store_true', help='print the plan (counts) and the docker argv; run nothing')
    parser.add_argument('--analyze-only', action='store_true', help='rerun the report over --results; no container')
    return parser


def parse_arms(text):
    arms = [arm for arm in re.split(r'[,\s]+', text.strip()) if arm]
    unknown = sorted(set(arms) - set(ARMS))
    if not arms or unknown:
        raise LabError('unknown arm(s): %s (known: %s)' % (', '.join(unknown or ['(none)']), ' '.join(ARMS)))
    if len(set(arms)) != len(arms):
        raise LabError('an arm named twice')
    return [arm for arm in ARMS if arm in arms]


def plan_counts(data, arms):
    return dict((arm, len(data.get(arm) or [])) for arm in arms)


def analyze(options, say, run_info=None):
    """The report over the results directory -> (code, public summary)."""
    info = run_info
    if info is None:
        try:
            with open(os.path.join(options.results, 'run.json'), encoding='utf-8') as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            info = {}
    manifest = read_manifest(options.data) if os.path.isdir(options.data) else {}
    counters = None
    if options.counters:
        with open(options.counters, encoding='utf-8') as handle:
            counters = json.load(handle)
    return report.build_and_write(options.results, options.public, manifest=manifest, counters=counters,
                                  seed=options.seed, say=say, info=info)


def main(argv=None, say=print, make_stream=None, make_client=None, make_container=None, make_log=None, docker=None,
         devices=None, clock=time.time, sleep=time.sleep, profiles=None):
    options = build_parser().parse_args(argv)
    try:
        arms = parse_arms(options.arms)
        dropped = {}
        # A5 with no file of its own re-sends A1's first 100 in A1's order, so A1's file is read for it even when A1 is not run.
        loading = arms + ['A1'] if 'A5' in arms and 'A1' not in arms else arms
        data, manifest, files = load_data(options.data, loading, dropped) if os.path.isdir(options.data) else (None, {}, {})
        if data is None and not options.analyze_only:
            raise LabError('the data directory does not exist')
    except (LabError, OSError, ValueError) as error:
        say('refused: %s' % error)
        return 2
    if options.analyze_only:
        summary = analyze(options, say)
        return 0 if summary and summary.get('complete') else 1
    if not options.image:
        say('refused: --image is required')
        return 2
    started = clock()
    if profiles is None:
        if options.profiles:
            with open(options.profiles, encoding='utf-8') as handle:
                profiles = json.load(handle)
        else:
            profiles = gate.image_profiles(options.image)
    try:
        derived, document = derive_profile(profiles, options.profile)
    except LabError as error:
        say('refused: %s' % error)
        return 2
    extra = arithmetic_diff(profiles, options.profile, document)
    if extra:
        say('refused: the derived profile changes arithmetic: %s' % ', '.join(extra))
        return 2
    limits = dict(prompt=pg.prompt_limit(profiles, options.profile))
    say('[TAULAB] data: %s' % ' '.join('%s=%d' % pair for pair in sorted(plan_counts(data, arms).items())))
    for arm in arms:
        if dropped.get(arm):
            say('[TAULAB] secret guard dropped from %s: %s' % (arm, ' '.join('%s=%d' % pair for pair in sorted(dropped[arm].items()))))
    say('[TAULAB] profile %s -> %s, image %s, production=%s' % (
        options.profile, derived, 'tag ' + image_tag(options.image), is_production_image(options.image)))
    devices = devices if devices is not None else (None if options.dry_run else gate.devices_for(options.cards))
    results_dir = os.path.abspath(options.results)
    derived_path = os.path.join(results_dir, 'profiles.json')
    arguments = pg.server_run(options.image, CONTAINER, derived, devices or ['<card %d>' % n for n in range(4)],
                              options.port, options.hub, derived_path)
    if options.dry_run:
        say(json.dumps(arguments))
        return 0
    if gate.platform_containers():
        say('refused: a platform serving container exists')
        return 2
    results = Results(results_dir)
    with open(derived_path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    with open(os.path.join(results_dir, 'docker-run.json'), 'w') as handle:
        json.dump(arguments, handle, indent=1)
    docker = docker or docker_run
    client = (make_client or (lambda: replay.Client(options.port)))()
    container = (make_container or (lambda name: replay.Container(name)))(CONTAINER)
    follower = (make_log or (lambda name, path: replay.LogFollower(name, path)))(CONTAINER, os.path.join(results_dir, 'server.log'))
    info = dict(image_tag=image_tag(options.image), production=is_production_image(options.image), profile=options.profile,
                derived=derived, seed=options.seed, arms=arms, arithmetic_extra=extra, dropped=dropped, data_counts=plan_counts(data, arms), started=started)
    status = 1
    lab = None
    try:
        docker(['docker', 'rm', '-f', CONTAINER], 120)
        code, output = docker(arguments, 300)
        if code != 0:
            raise RuntimeError('docker run exited %s' % code)
        follower.start()
        problem = wait_ready(client, container, clock, sleep, max(60, min(READINESS_SECONDS, options.deadline_seconds)))
        if problem:
            raise RuntimeError(problem)
        info['ready_seconds'] = round(clock() - started, 1)
        say('[TAULAB] ready after %.0f s' % info['ready_seconds'])
        context = client.context_tokens()
        info['context_tokens'] = context
        problems = launched_problems(follower.lines(), derived)
        info['launched_problems'] = problems
        for problem in problems:
            say('[TAULAB] WARNING: %s' % problem)
        stream = (make_stream or (lambda: lambda path, body, timeout, keep: http_stream(
            options.port, path, body, timeout, keep_content=keep)))()
        lab = Lab(results, stream, limits, seed=options.seed, in_flight=options.in_flight, max_tokens=options.max_tokens,
                  clock=clock, sleep=sleep, say=say)
        send_ends = started + options.deadline_seconds - RESERVE_SECONDS
        info['counts'] = lab.run(arms, data, manifest, send_ends)
        status = 0
    except Exception as error:  # noqa: BLE001 - the container must still be stopped and the report still written
        info['error'] = type(error).__name__
        say('[TAULAB] stopped: %s' % type(error).__name__)
    finally:
        if lab is not None:
            lab.stop.set()
        sleep(SETTLE_SECONDS)
        docker(['docker', 'stop', '-t', '120', CONTAINER], 300)
        sleep(SETTLE_SECONDS)
        docker(['docker', 'rm', '-f', CONTAINER], 120)
        info['finished'] = clock()
        with open(os.path.join(results_dir, 'run.json'), 'w', encoding='utf-8') as handle:
            json.dump(info, handle, indent=1, sort_keys=True, default=str)
    summary = analyze(options, say, info)
    complete = bool(summary and summary.get('complete')) and status == 0
    say('C2_TAULAB profile=%s complete=%s' % (options.profile, complete))
    return 0 if complete else 1


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
