"""agent_turns: a production-shaped multi-turn smoke test, the plan it follows, the numbers it reports and the log reader behind them.

WHY. The gates before it measure what a card does under SYNTHETIC load: eight users on ~3.5k-token prompts (concurrent8_steady), eight at 32k and
120k started together (concurrent8_code_32k / _128k). Production agent traffic is not that: sessions sit at 70-200k tokens of context, a returning
turn appends a few hundred new tokens and RE-PREFILLS a few thousand old ones (the resume checkpoint is floor2048(P_prev) - 2048, so the tail from
the checkpoint to the end of the old prompt is prefilled again), pays an engine rebind, decodes a few hundred tokens and then waits for a tool.
On the 2026-10 ship-4 stack the card's wall time in production was ~58% prefill steps, ~36% decode rounds, ~6% other, with 14-22 tok/s a user at
four to seven live users. This test reproduces that shape deterministically so a card window can read the same numbers.

THE SHAPE (c2_serving_smoke.py: agent_turns, agent_turns_long, agent_turns8). `sessions` concurrent sessions (default 6, under the eight seats; agent_turns8 has EIGHT, see VARIANT_SETTINGS), each:
  turn 0   a coding-text context of its own length (the mix cycles 64k / 128k / 200k tokens, rotated and jittered by the seed; fitted to the
           server's own token count where /tokenize answers; real code from the corpus the long-context smokes read) plus a code task, the
           opening request: the cold prefill that builds the session;
  turn 1.. `turns` more (default 12). Each appends ONE user message (a deterministic ~150-400 token "tool result + instruction": a numbered
           file excerpt or grep lines of real code, then a follow-up) after the assistant's previous answer INCLUDED VERBATIM (content, and the
           reasoning the server returned in the field it reads back), so the history is append-only and a prefix-reuse server can resume;
           generates 200-600 tokens (greedy: the API edge forces it), then waits a tool gap of 1-8 s (x SMOKE_AGENT_GAP_SCALE).
Everything but the answers is deterministic in (seed, session, turn): contexts, gaps, budgets and messages are drawn from a hash-seeded LCG, never
the random module, so two arms send the same requests until their answers differ (the answers are greedy, so on an exact stack they never do).

REUSE AND THE SALT. A prefix-reuse profile (QWEN_PREFIX_REUSE=1) keeps a request's cache_salt only when it verifies against the platform's salt
key (serving_c2_contract.mint_salt); unsalted, reuse is inert and a turn re-prefills its whole history. The test mints one salt per session
('qps1.<tag>.<hmac>') under the key file SMOKE_SALT_KEY_FILE names (the smoke step writes a throwaway key and mounts it into the container when the
job names agent_turns), so a returning turn can hit. SMOKE_AGENT_SALT=none sends no salt on purpose (the no-reuse control arm).

WHAT IT REPORTS (the entry c2_serving_smoke records under SMOKE_JSON):
  per turn     prompt tokens, cached tokens (usage.prompt_tokens_details.cached_tokens when the server reports it), the tokens re-prefilled
               (prompt - cached), time to first token, decode tok/s, completion tokens, finish, the stream's own clocks and answer hashes;
  per session  mean decode tok/s, TTFT p50 / p90 of the returning turns, the opening request's TTFT, tokens, turns done, why it stopped;
  aggregate    committed tok/s over the run, TTFT by context bucket, the re-prefill distribution, reuse counts, the steady window (every session
               built, none finished) and the client-side per-live table (live users -> tok/s a user, measured from the stream clocks).
FROM THE CONTAINER LOG (container_facts; c2_smoke_check.py calls it, `python3 agent_turns.py report` prints it): every request carries an
X-Request-Id tag, so the engine's own lines name the turn they belong to: the server-side resume point (the first '[PINDIAG] lever N route' step's
start, or a '[PREFIX]' row's Q), the route time of each step (prefill share of the run's wall time = sum of 'ms=' / wall), the engine rebind
('[PINDIAG] sticky engine built ... kind=rebind'), and the '[OCTO] round' lines inside the run (decode share = sum of step_ms / wall; the per-live
table: live -> step_ms, gap_ms, committed per live user, tok/s a user).

THE JUDGE (c2_smoke_check.agent_turns_problems; no speed threshold: it is a measurement): an error or a timeout in any request fails; a request
with no tokens or no finish fails; on a prefix-reuse profile a SALTED run must show reuse (cached tokens or the server's resume point above zero)
on at least 90% of its returning turns, and an UNSALTED run on such a profile fails unless it asked for no salt (the plausible-number trap: a run
without reuse measures a server nobody serves).

Stdlib only, Python 3.7 syntax: the smoke client runs on the rig host and the check on the runner.
"""

import argparse
import bisect
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time

NAME = 'agent_turns'
LONG_NAME = 'agent_turns_long'
EIGHT_NAME = 'agent_turns8'
TESTS = (NAME, LONG_NAME, EIGHT_NAME)
# name -> (sessions, turns): the named shapes. Environment parameters (settings) override any. agent_turns and agent_turns_long are unchanged (the baseline of past results).
VARIANTS = {NAME: (6, 12), LONG_NAME: (6, 30), EIGHT_NAME: (8, 12)}
# agent_turns8 (tp4/s6): EIGHT sessions, every seat of the profile busy, weighted like production traffic instead of like agent_turns: production prompts average about 59k tokens (p50 69k, p90 208k)
# and its answers about 690 tokens, where agent_turns' mix (64k, 128k, 200k) averages 131k and its answers 400. Eight distinct contexts (the mix has eight values and the seed only rotates it, so
# each appears once; about 545k tokens in all against the 1,055,488-token pool, where eight sessions of the agent_turns mix would be 1.06M and evict each other), answers capped at 300 to 1,080
# tokens (mean 690), and a 1,500 s budget (no new turn after 25 minutes) so that it fits a 30-minute box. Environment parameters still override each. Compare it with itself (the ship-5 control run
# on the same name), never with agent_turns.
EIGHT_MIX = (24000, 32000, 40000, 48000, 56000, 64000, 80000, 200000)
EIGHT_ANSWER_RANGE = (300, 1080)
VARIANT_SETTINGS = {EIGHT_NAME: dict(mix=EIGHT_MIX, answer_range=EIGHT_ANSWER_RANGE, budget_s=1500.0)}
SEED = 20261012
MIX = (64000, 128000, 200000)            # the context lengths the sessions cycle through (tokens); production prompts: p50 69k, p90 208k
GAP_RANGE_S = (1.0, 8.0)                 # the tool gap after a turn
ANSWER_RANGE = (200, 600)                # max_tokens of a turn
MESSAGE_RANGE = (150, 400)               # tokens of the appended tool-result + instruction message
CHARS_PER_TOKEN = 3.2                    # Qwen3.8's tokenizer on numbered code excerpts (prefix_agent_corpus.CHARS_PER_TOKEN)
MESSAGE_SLACK = 1.6                      # the most the estimate may undercount a message when the context guard reserves room for the next one
STAGGER_S = 0.5                          # session i starts i x STAGGER_S seconds after the first
BUDGET_S = 2700.0                        # no session starts a new turn after this many seconds
TIMEOUT_S = 900.0                        # the per-read timeout of a stream: a request silent this long is a timeout (an error)
PAGE = 64                                # the KV page: context targets are multiples of it
JITTER = 0.04                            # a context target is the mix value +- 4%
DEFAULT_MAX_CONTEXT = 262144
CONTEXT_MARGIN = 2048                    # room kept beyond the planned growth of a session
MAX_SESSIONS = 16
MAX_TURNS = 400
IDLE_GAP_MS = 5000.0                     # an [OCTO] round whose gap is longer is idle time between requests, not a stall a user felt
REUSE_MISS_SHARE = 0.10                  # a salted run on a reuse profile fails when more than this share of its returning turns reused nothing
LINE_CHARS = 160                         # a corpus line is cut to this many characters in a tool result
ROUND_COVER = 0.98                       # the log window ends once the rounds in it committed this share of the run's tokens
STEADY_MIN_S = 20.0                      # a steady window shorter than this is reported as absent

# The production figures the run-time estimate is built from (ship-4 stack, Swift-1.5, ~2 h of traffic): a cold prefill runs ~2.6k tok/s (a 254k
# prompt took 96 s), a returning turn re-prefills ~3.2k tokens and pays an engine rebind of ~0.54 s, a user decodes ~16 tok/s at four to seven live.
ESTIMATE = dict(cold_prefill_tok_s=2600.0, resume_prefill_s=3.0, rebind_s=0.54, user_decode_tok_s=16.0, contention=1.15)

FOLLOWUPS = (
    'Using that output, point out the single most likely defect and show the smallest change that fixes it.',
    'Summarise what this tells us about the code discussed so far in three bullet points, then suggest one test to add.',
    'Does this change the plan? If so, give the revised next step as a short numbered list.',
    'Explain why this part is written this way and whether a simpler version would behave identically.',
    'Write the unit test that would have caught what this shows.',
    'Rewrite the affected function with clearer names and type annotations, keeping its behaviour.',
    'List the edge cases this code does not handle and rank them by how likely they are to matter.',
    'Continue the review with this file: quote each problem, explain it and give the corrected code.',
    'What should be read next to confirm this? Name the file and the function, and say what to look for.',
    'Apply the fix to the code and show the full corrected function.',
    'Compare this with the earlier code in the conversation and say what differs in behaviour.',
    'Explain the control flow of this snippet step by step.',
)
SALT_VERSION = 'qps1'
SALT_KEY_MIN_BYTES = 32
TAG_PREFIX = 'agt'
NONCE = re.compile(r'\A[0-9a-z]{4,16}\Z')
TAG_IN_LINE = re.compile(r'chatcmpl-(agt-[0-9a-z]+-s(\d+)-t(\d+))-[0-9A-Za-z]{8}(?![0-9A-Za-z])')


# -- settings and the deterministic plan --------------------------------------------------------------------------------------------------

def parse_tokens(text):
    """'64k' -> 64000, '200000' -> 200000, '1.5k' -> 1500."""
    text = str(text).strip().lower()
    scale = 1
    if text.endswith('k'):
        text, scale = text[:-1], 1000
    return int(round(float(text) * scale))


def settings(name, environ=None):
    """The test's parameters: the named variant's sessions and turns, the environment's overrides (SMOKE_AGENT_*), validated.

    SESSIONS (1..16), TURNS (1..400), SEED, MIX (comma-separated context lengths in tokens, 64k style allowed), GAP_SCALE (0..100: x the 1-8 s
    tool gap), BUDGET_S (60..14400: no new turn starts after it), TIMEOUT_S, STAGGER_S, IGNORE_EOS (0/1: every answer runs to its budget),
    SALT (auto | none), MAX_CONTEXT (the profile's context when /v1/models does not say), NONCE (4..16 of [0-9a-z]: names the run's requests and
    salts; default random)."""
    environ = os.environ if environ is None else environ
    if name not in VARIANTS:
        raise ValueError('no such test %r (%s)' % (name, ', '.join(TESTS)))
    sessions, turns = VARIANTS[name]

    def read(key, cast, default, low=None, high=None):
        raw = environ.get('SMOKE_AGENT_' + key)
        if raw is None or str(raw).strip() == '':
            return default
        try:
            value = cast(raw)
        except ValueError:
            raise ValueError('SMOKE_AGENT_%s=%r is not a valid %s' % (key, raw, cast.__name__))
        if (low is not None and value < low) or (high is not None and value > high):
            raise ValueError('SMOKE_AGENT_%s=%r is outside %s..%s' % (key, raw, low, high))
        return value

    variant = VARIANT_SETTINGS.get(name, {})
    mix = tuple(parse_tokens(part) for part in str(environ.get('SMOKE_AGENT_MIX') or '').split(',') if part.strip()) or variant.get('mix', MIX)
    if any(value < 256 or value > 1048576 for value in mix):
        raise ValueError('SMOKE_AGENT_MIX holds a context outside 256..1048576 tokens: %r' % (mix,))
    salt = str(environ.get('SMOKE_AGENT_SALT') or 'auto').strip().lower()
    if salt not in ('auto', 'none'):
        raise ValueError('SMOKE_AGENT_SALT is auto or none, got %r' % salt)
    nonce = str(environ.get('SMOKE_AGENT_NONCE') or '').strip().lower() or hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    if not NONCE.match(nonce):
        raise ValueError('SMOKE_AGENT_NONCE is 4..16 of [0-9a-z], got %r' % nonce)
    return dict(name=name, sessions=read('SESSIONS', int, sessions, 1, MAX_SESSIONS), turns=read('TURNS', int, turns, 1, MAX_TURNS),
                seed=read('SEED', int, SEED), mix=mix, gap_scale=read('GAP_SCALE', float, 1.0, 0.0, 100.0),
                gap_range=GAP_RANGE_S, answer_range=variant.get('answer_range', ANSWER_RANGE), message_range=MESSAGE_RANGE,
                budget_s=read('BUDGET_S', float, variant.get('budget_s', BUDGET_S), 60.0, 14400.0), timeout_s=read('TIMEOUT_S', float, TIMEOUT_S, 10.0, 7200.0),
                stagger_s=read('STAGGER_S', float, STAGGER_S, 0.0, 600.0), ignore_eos=bool(read('IGNORE_EOS', int, 0, 0, 1)),
                salt_mode=salt, max_context=read('MAX_CONTEXT', int, DEFAULT_MAX_CONTEXT, 4096, 4194304), nonce=nonce)


class Lcg(object):
    """A linear congruential generator seeded by a hash of its parts: the same draws on every run and Python (churn_script's constants)."""

    def __init__(self, *parts):
        digest = hashlib.sha256('|'.join(str(part) for part in parts).encode('utf-8')).digest()
        self.state = int.from_bytes(digest[:4], 'big') & 0x7fffffff

    def random(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
        return ((self.state >> 8) & 0x7fffff) / float(0x800000)

    def between(self, low, high):
        return low + (high - low) * self.random()


def context_room(cfg, max_model_len=None):
    """The longest opening context (tokens) a session of this plan can start from: the profile's context, less the growth of `turns` turns
    (a message and an answer each) and a margin."""
    limit = max_model_len or cfg['max_context']
    growth = cfg['turns'] * (int(cfg['message_range'][1] * MESSAGE_SLACK) + cfg['answer_range'][1] + 64)
    return max(limit - growth - cfg['answer_range'][1] - CONTEXT_MARGIN, 1024)


def session_targets(cfg, max_model_len=None):
    """([target tokens per session], clamped): the mix cycled in a seed-rotated order, each value jittered +-4% and rounded to the KV page, then
    held under context_room. `clamped` is how many were cut."""
    rng = Lcg('targets', cfg['seed'])
    mix = cfg['mix']
    rotation = int(rng.random() * len(mix))
    room = context_room(cfg, max_model_len)
    targets, clamped = [], 0
    for index in range(cfg['sessions']):
        base = mix[(index + rotation) % len(mix)]
        value = int(base * (1.0 + rng.between(-JITTER, JITTER))) // PAGE * PAGE
        if value > room:
            value, clamped = room // PAGE * PAGE, clamped + 1
        targets.append(max(value, PAGE))
    return targets, clamped


def turn_plan(cfg, session, turn):
    """The deterministic draws of one turn: the tool gap that FOLLOWS it (seconds, scaled), its answer budget, the appended message's size in
    tokens (turn 0 has no message: the opening request), the follow-up instruction and where in the corpus its tool result reads."""
    rng = Lcg('turn', cfg['seed'], session, turn)
    gap = rng.between(*cfg['gap_range']) * cfg['gap_scale']
    answer = int(rng.between(cfg['answer_range'][0], cfg['answer_range'][1] + 1))
    message = int(rng.between(cfg['message_range'][0], cfg['message_range'][1] + 1)) if turn else 0
    return dict(turn=turn, gap_s=round(gap, 3), max_tokens=min(answer, cfg['answer_range'][1]), message_tokens=message,
                instruction=int(rng.random() * len(FOLLOWUPS)), offset=rng.random(), kind='grep' if rng.random() < 0.3 else 'read_file')


def plan(cfg, max_model_len=None):
    """The whole plan, deterministic in the settings: {targets, clamped, sessions: [[turn plan] per turn 0..turns]}."""
    targets, clamped = session_targets(cfg, max_model_len)
    return dict(targets=targets, clamped=clamped,
                sessions=[[turn_plan(cfg, session, turn) for turn in range(cfg['turns'] + 1)] for session in range(cfg['sessions'])])


def plan_digest(the_plan):
    """A short hash of a plan: two arms with equal digests send the same gaps, budgets and message sizes."""
    return hashlib.sha256(json.dumps(the_plan, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def estimate(cfg, targets=None, max_model_len=None):
    """The run's expected wall time (a dict, minutes among it) from the production figures in ESTIMATE: the cold builds queue on one prefill
    path, then the last-built session still runs its `turns` returning turns (a re-prefill, a rebind, ~mean-answer tokens of decode, the mean
    gap), x a contention allowance for the other sessions' prefill steps. An estimate: the card decides."""
    figures = ESTIMATE
    if targets is None:
        targets = session_targets(cfg, max_model_len)[0]
    prime_s = sum(targets) / figures['cold_prefill_tok_s'] + cfg['stagger_s'] * (cfg['sessions'] - 1)
    mean_answer = (cfg['answer_range'][0] + cfg['answer_range'][1]) / 2.0
    mean_gap = (cfg['gap_range'][0] + cfg['gap_range'][1]) / 2.0 * cfg['gap_scale']
    per_turn = figures['resume_prefill_s'] + figures['rebind_s'] + mean_answer / figures['user_decode_tok_s'] + mean_gap
    returning_s = cfg['turns'] * per_turn * figures['contention']
    return dict(prime_s=round(prime_s, 1), per_turn_s=round(per_turn, 1), returning_s=round(returning_s, 1),
                total_s=round(prime_s + returning_s, 1), minutes=round((prime_s + returning_s) / 60.0, 1),
                within_budget=prime_s + returning_s <= cfg['budget_s'])


# -- the appended message ----------------------------------------------------------------------------------------------------------------

def neutralise(text):
    """A space after the '<' of a chat-template marker that real code may carry as a string literal ('<|im_start|>' -> '< |im_start|>'): the
    server must not read it as a special token."""
    return text.replace('<|', '< |')


def file_at(corpus, position):
    """(path, line number of the position's line) for the corpus file that holds `position`: corpus files start with a 'File: <path>' line."""
    marker = corpus.rfind('\nFile: ', 0, position + 1)
    marker = marker + 1 if marker >= 0 else (0 if corpus.startswith('File: ') else -1)
    if marker < 0:
        return 'src/module.py', 1
    end = corpus.find('\n', marker)
    end = end if end >= 0 else len(corpus)
    return corpus[marker + 6:end].strip() or 'src/module.py', 1 + corpus.count('\n', end + 1, position)


GREP_PATTERNS = ('def ', 'class ', 'import ', 'return ')
MESSAGE_OVERHEAD_CHARS = 150            # the tool-result head and tail and the instruction, which the excerpt leaves room for (the rest is line granularity)


def excerpt_lines(corpus, start, budget, kind, pattern, number, path):
    """([rows], first line number, last line number): numbered lines of the corpus from `start` ('read_file': every non-blank line; 'grep': the lines
    that hold `pattern`, with their real line numbers) until `budget` characters or the end of the file."""
    reach = 40 * budget if kind == 'grep' else 6 * budget
    rows, used, first_line, last_line = [], 0, number, number
    for text in corpus[start:start + reach].split('\n'):
        if text.startswith('File: '):
            break
        text = neutralise(text.rstrip())[:LINE_CHARS]
        if text.strip() and (kind == 'read_file' or pattern in text):
            row = '%6d\t%s' % (number, text) if kind == 'read_file' else '%s:%d:%s' % (path, number, text)
            if used + len(row) + 1 > budget and rows:
                break
            if not rows:
                first_line = number
            rows.append(row)
            used += len(row) + 1
            last_line = number
        number += 1
    return rows, first_line, last_line


def tool_message(corpus, cfg, session, turn, turn_draws):
    """The user message of a returning turn: a tool result of real code (a numbered excerpt of a file, or grep lines) of about the planned token
    count, then the turn's follow-up instruction. Deterministic in (corpus, plan)."""
    instruction = FOLLOWUPS[turn_draws['instruction']]
    budget = max(int(turn_draws['message_tokens'] * CHARS_PER_TOKEN) - MESSAGE_OVERHEAD_CHARS, 200)
    start = int(turn_draws['offset'] * max(len(corpus) - 42 * budget - 1, 1))
    newline = corpus.find('\n', start)
    start = newline + 1 if 0 <= newline < len(corpus) - 1 else start
    path, number = file_at(corpus, start)
    kind = turn_draws['kind']
    pattern = GREP_PATTERNS[turn_draws['instruction'] % len(GREP_PATTERNS)]
    rows, first_line, last_line = excerpt_lines(corpus, start, budget, kind, pattern, number, path)
    if kind == 'grep' and len(rows) < 3:
        kind = 'read_file'
        rows, first_line, last_line = excerpt_lines(corpus, start, budget, kind, pattern, number, path)
    if kind == 'read_file':
        head = '[tool result] read_file %s lines %d-%d' % (path, first_line, last_line)
    else:
        head = '[tool result] grep -n "%s" %s' % (pattern.strip(), path)
    return '%s\n%s\n[end of tool result]\n\n%s' % (head, '\n'.join(rows), instruction)


# -- tags, salts ------------------------------------------------------------------------------------------------------------------------

def request_tag(nonce, session, turn):
    """The X-Request-Id of a turn; the engine names its request 'chatcmpl-<tag>-<8 characters>' and its log lines carry that."""
    return '%s-%s-s%d-t%02d' % (TAG_PREFIX, nonce, session, turn)


def salt_tag(nonce, session):
    return 'agentturns-%s-s%d' % (nonce, session)


def read_salt_key(path):
    with open(path, 'rb') as handle:
        key = handle.read().strip()
    if len(key) < SALT_KEY_MIN_BYTES:
        raise ValueError('%s holds %d bytes, fewer than %d' % (path, len(key), SALT_KEY_MIN_BYTES))
    return key


def mint_salt(key, tag):
    """The cache_salt the serving contract verifies for `tag` under `key` (bytes): 'qps1.<tag>.<hmac-sha256>' (serving_c2_contract.mint_salt,
    restated for the rig host; test_agent_turns holds the two equal)."""
    return '%s.%s.%s' % (SALT_VERSION, tag, hmac.new(key, ('%s.%s' % (SALT_VERSION, tag)).encode('ascii'), hashlib.sha256).hexdigest())


# -- numbers ----------------------------------------------------------------------------------------------------------------------------

def percentile(values, share):
    """The linear-interpolated `share`% (0..100) of the numbers in `values` (None entries dropped), or None."""
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * share / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def rounded(value, places=2):
    return None if value is None else round(value, places)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def count_between(stamps, begin, end):
    """How many of the sorted `stamps` fall in (begin, end]."""
    return bisect.bisect_right(stamps, end) - bisect.bisect_right(stamps, begin)


def live_table(requests, begin=None, end=None):
    """The client-side per-live table: requests are (first_at, ended_at, stamps, tokens) of streams (a stream is LIVE from its first token to its end,
    so a stall inside it counts against the user, as it did). The time is cut at every request's first and last instant; in each piece with k live
    users the tokens are every live stream's stamps in it, each stamp worth tokens / stamps (a delta can carry more than one token). Returns
    [dict(live, seconds, tokens, user_tok_s, aggregate_tok_s)] by live count, restricted to (begin, end) when given."""
    streams = [(first, last, sorted(stamps), tokens) for first, last, stamps, tokens in requests
               if first is not None and last is not None and last > first and stamps]
    inner = sorted(set([first for first, _, _, _ in streams] + [last for _, last, _, _ in streams]))
    inner = [point for point in inner if (begin is None or point > begin) and (end is None or point < end)]
    points = ([begin] if begin is not None else []) + inner + ([end] if end is not None else [])
    table = {}
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        live = [stream for stream in streams if stream[0] <= left and stream[1] >= right]
        if not live:
            continue
        tokens = sum(count_between(stamps, left, right) * float(total) / len(stamps) for _, _, stamps, total in live)
        row = table.setdefault(len(live), dict(seconds=0.0, tokens=0.0))
        row['seconds'] += right - left
        row['tokens'] += tokens
    return [dict(live=live, seconds=round(row['seconds'], 1), tokens=int(round(row['tokens'])),
                 user_tok_s=round(row['tokens'] / (live * row['seconds']), 1) if row['seconds'] > 0 else None,
                 aggregate_tok_s=round(row['tokens'] / row['seconds'], 1) if row['seconds'] > 0 else None)
            for live, row in sorted(table.items())]


def window_tokens(requests, begin, end):
    """The tokens streamed inside (begin, end] by the requests (as live_table counts them)."""
    return sum(count_between(sorted(stamps), begin, end) * float(total) / len(stamps)
               for first, last, stamps, total in requests if first is not None and last is not None and stamps)


CONTEXT_BUCKETS = (('lt32k', 0, 32768), ('32k_128k', 32768, 131072), ('ge128k', 131072, 10 ** 9))


def bucket_of(prompt_tokens):
    for name, low, high in CONTEXT_BUCKETS:
        if low <= (prompt_tokens or 0) < high:
            return name
    return CONTEXT_BUCKETS[-1][0]


def turn_row(session, turn, tag, draws, answer, previous_prompt):
    """One request's recorded row (the keys c2_smoke_check.stream_problems and the other smoke rows use, plus the agent ones)."""
    prompt = answer.get('prompt_tokens')
    cached = answer.get('cached_tokens')
    row = dict(session=session, turn=turn, tag=tag, prompt_tokens=prompt, tokens=answer.get('tokens'), finish=answer.get('finish'),
               ttft=answer.get('ttft'), decode_tok_s=answer.get('decode_tok_s'), max_tokens=draws['max_tokens'],
               gap_after_s=draws['gap_s'], text=(answer.get('text') or '')[:200], content_sha256=answer.get('content_sha256'),
               reasoning_sha256=answer.get('reasoning_sha256'), started_at=answer.get('started_at'), first_at=answer.get('first_at'),
               ended_at=answer.get('ended_at'))
    if 'cached_tokens' in answer:
        row['cached_tokens'] = cached
        row['reprefill_tokens'] = prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None
    if turn and isinstance(prompt, int) and isinstance(previous_prompt, int):
        row['appended_tokens'] = prompt - previous_prompt
    return row


def assistant_message(answer):
    """The assistant's turn as history: its content verbatim (never None: the template trims it) and the reasoning the server returned, in the field
    vLLM 0.25 reads back (prefix_agent_corpus.assistant_message)."""
    message = {'role': 'assistant', 'content': answer.get('full_content') or ''}
    if answer.get('full_reasoning'):
        message['reasoning'] = answer['full_reasoning']
    return message


# -- the run -----------------------------------------------------------------------------------------------------------------------------

def run(cfg, targets, prompts, corpus, stream_fn, key=None, sleep=time.sleep, now=time.time, say=None, the_plan=None):
    """Drive the sessions to completion and return the entry c2_serving_smoke records. `prompts[i]` is session i's opening request (its context and
    task), `targets` their planned token counts, `corpus` the real code the appended messages read, `stream_fn(messages, max_tokens, timeout=,
    keep_stamps=True, keep_text=True, headers=, **body)` a streamed chat completion (c2_serving_smoke.stream), `key` the salt key bytes (None: no
    salt). Each session is a thread: the opening request at i x stagger seconds, then its returning turns, each after the previous answer and its tool
    gap; no session starts a turn once `budget_s` is up. An error ends its session (a row with `error`), never the run."""
    say = say or (lambda *parts: print(*parts, flush=True))
    sessions = len(prompts)
    cfg = dict(cfg, salted=bool(key))
    the_plan = the_plan or plan(cfg)
    rows = [[] for _ in range(sessions)]
    stopped = [None] * sessions
    streams = {}
    lock = threading.Lock()
    began = now()
    deadline = began + cfg['budget_s']
    extra_body = dict(ignore_eos=True) if cfg['ignore_eos'] else {}

    def session_loop(index):
        salt = mint_salt(key, salt_tag(cfg['nonce'], index)) if key else None
        messages = [{'role': 'user', 'content': prompts[index]}]
        previous = None
        sleep(cfg['stagger_s'] * index)
        for turn in range(cfg['turns'] + 1):
            draws = the_plan['sessions'][index][turn]
            if turn:
                if now() >= deadline:
                    stopped[index] = 'budget'
                    return
                messages.append({'role': 'user', 'content': tool_message(corpus, cfg, index, turn, draws)})
            tag = request_tag(cfg['nonce'], index, turn)
            body = dict(extra_body)
            if salt:
                body['cache_salt'] = salt
            try:
                answer = stream_fn(messages, draws['max_tokens'], timeout=cfg['timeout_s'], keep_stamps=True, keep_text=True,
                                   headers={'X-Request-Id': tag}, **body)
            except Exception as error:      # a timeout, a refusal, a dropped connection: the session ends, the run goes on
                rows[index].append(dict(session=index, turn=turn, tag=tag, error=repr(error)[:300]))
                stopped[index] = 'error'
                say('agent_turns session', index, 'turn', turn, 'ERROR', repr(error)[:200])
                return
            row = turn_row(index, turn, tag, draws, answer, previous)
            with lock:
                streams[(index, turn)] = (answer.get('first_at'), answer.get('ended_at'), answer.pop('delta_stamps', None) or [], answer.get('tokens') or 0)
            rows[index].append(row)
            say('agent_turns session', index, 'turn', turn, json.dumps(dict((key_, row.get(key_)) for key_ in (
                'prompt_tokens', 'cached_tokens', 'ttft', 'tokens', 'decode_tok_s', 'finish'))))
            if not answer.get('tokens'):
                stopped[index] = 'error'      # nothing to carry forward; the judge reports the empty answer
                return
            messages.append(assistant_message(answer))
            previous = answer.get('prompt_tokens')
            used = (previous or 0) + (answer.get('tokens') or 0)
            if used + int(cfg['message_range'][1] * MESSAGE_SLACK) + cfg['answer_range'][1] + CONTEXT_MARGIN > cfg['max_context'] and turn < cfg['turns']:
                stopped[index] = 'context'
                return
            if turn < cfg['turns']:
                sleep(draws['gap_s'])

    threads = [threading.Thread(target=session_loop, args=(index,)) for index in range(sessions)]
    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        thread.join()
    return build_entry(cfg, the_plan, targets, rows, stopped, streams, began, now())


def build_entry(cfg, the_plan, targets, rows, stopped, streams, began, ended):
    """The recorded entry from the sessions' rows and their streams' clocks (see the module docstring for the keys)."""
    done = [row for session in rows for row in session if 'error' not in row]
    errors = [row for session in rows for row in session if 'error' in row]
    firsts = [row['started_at'] for row in done if row.get('started_at')]
    lasts = [row['ended_at'] for row in done if row.get('ended_at')]
    run_begin = min(firsts) if firsts else began
    run_end = max(lasts) if lasts else ended
    wall = max(run_end - run_begin, 0.0)
    tokens = sum(row.get('tokens') or 0 for row in done)
    requests = [(first, last, stamps, total) for (first, last, stamps, total) in streams.values()]
    opened = [row['ended_at'] for row in done if row['turn'] == 0 and row.get('ended_at')]
    closing = [session[-1]['ended_at'] for session in rows if session and 'error' not in session[-1] and session[-1].get('ended_at')]
    steady = None
    if len(opened) == len(rows) and len(closing) == len(rows) and rows:
        low, high = max(opened), min(closing)
        if high - low >= STEADY_MIN_S:
            inside = window_tokens(requests, low, high)
            steady = dict(begin_s=round(low - run_begin, 1), window_s=round(high - low, 1), committed_tok_s=round(inside / (high - low), 2),
                          live=live_table(requests, low, high))
    returning = [row for row in done if row['turn'] > 0]
    reported = [row for row in returning if 'cached_tokens' in row]
    cached_known = [row for row in reported if isinstance(row.get('cached_tokens'), int)]
    buckets = {}
    for name, _, _ in CONTEXT_BUCKETS:
        members = [row for row in returning if bucket_of(row.get('prompt_tokens')) == name]
        buckets[name] = dict(turns=len(members), ttft_p50_s=rounded(percentile([row.get('ttft') for row in members], 50)),
                             ttft_p90_s=rounded(percentile([row.get('ttft') for row in members], 90)))
    per_session = []
    for index, session in enumerate(rows):
        own = [row for row in session if 'error' not in row]
        later = [row for row in own if row['turn'] > 0]
        per_session.append(dict(
            session=index, target_tokens=targets[index], turns=session, turns_done=len(later), stopped=stopped[index],
            opening_ttft_s=own[0].get('ttft') if own and own[0]['turn'] == 0 else None,
            prompt_tokens_first=own[0].get('prompt_tokens') if own else None, prompt_tokens_last=own[-1].get('prompt_tokens') if own else None,
            completion_tokens=sum(row.get('tokens') or 0 for row in own),
            mean_decode_tok_s=rounded(mean([row.get('decode_tok_s') for row in later])),
            ttft_p50_s=rounded(percentile([row.get('ttft') for row in later], 50)), ttft_p90_s=rounded(percentile([row.get('ttft') for row in later], 90))))
    aggregate = dict(
        sessions=len(rows), turns_planned=len(rows) * (cfg['turns'] + 1), requests_done=len(done), errors=len(errors), completion_tokens=tokens,
        wall_s=round(wall, 1), committed_tok_s=round(tokens / wall, 2) if wall > 0 else None,
        returning=dict(turns=len(returning), ttft_p50_s=rounded(percentile([row.get('ttft') for row in returning], 50)),
                       ttft_p90_s=rounded(percentile([row.get('ttft') for row in returning], 90)),
                       decode_tok_s_mean=rounded(mean([row.get('decode_tok_s') for row in returning])),
                       decode_tok_s_p50=rounded(percentile([row.get('decode_tok_s') for row in returning], 50)),
                       appended_tokens_p50=rounded(percentile([row.get('appended_tokens') for row in returning], 50), 0)),
        ttft_by_context=buckets,
        reuse=dict(returning_turns=len(returning), cached_reported=len(cached_known), cached_positive=sum(1 for row in cached_known if row['cached_tokens'] > 0),
                   reprefill_p50=rounded(percentile([row.get('reprefill_tokens') for row in reported], 50), 0),
                   reprefill_p90=rounded(percentile([row.get('reprefill_tokens') for row in reported], 90), 0)),
        prime=dict(wall_s=round(max(opened) - run_begin, 1) if opened else None, cold_ttft_p50_s=rounded(percentile([
            row.get('ttft') for row in done if row['turn'] == 0], 50))),
        steady=steady, live=live_table(requests))
    return dict(test=cfg['name'], seed=cfg['seed'], nonce=cfg['nonce'], sessions_planned=cfg['sessions'], turns_planned=cfg['turns'],
                gap_scale=cfg['gap_scale'], ignore_eos=cfg['ignore_eos'], salted=bool(cfg.get('salted')),
                salt_mode=cfg['salt_mode'], targets=targets, plan_digest=plan_digest(the_plan), estimate=estimate(cfg, targets),
                started_at=round(run_begin, 3), ended_at=round(run_end, 3), aggregate=aggregate, sessions=per_session)


# -- the container log --------------------------------------------------------------------------------------------------------------------

ROUTE = re.compile(r'\[PINDIAG\] lever N route req=(\S+) start=(\d+) end=(\d+) prompt=(\d+) final=([01]) .*?\bms=([0-9.]+)')
SOURCE = re.compile(r'\bsource=([A-Z]+)')
STICKY = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([0-9.]+) frontier=(\d+) prompt=(\d+) kind=(\w+)')
PREFIX_ROW = re.compile(r'\[PREFIX\] (?:row|req)=\S*')
REQ_FIELD = re.compile(r'\breq=(\S+)')
Q_FIELD = re.compile(r'\bQ=(\d+)')
L_FIELD = re.compile(r'\bL=(\d+)')


def tag_of(line, nonce):
    """(session, turn) of the agent_turns request a log line names (its engine request id carries the X-Request-Id tag), or None."""
    prefix = 'chatcmpl-%s-%s-' % (TAG_PREFIX, nonce)
    if prefix not in line:
        return None
    match = TAG_IN_LINE.search(line)
    if match and match.group(1).startswith('%s-%s-' % (TAG_PREFIX, nonce)):
        return int(match.group(2)), int(match.group(3))
    return None


def live_rows(rounds, idle_gap_ms=IDLE_GAP_MS):
    """The server-side per-live table of [OCTO] rounds (octo_markers.rounds dicts): live -> rounds, mean step_ms, mean gap_ms, mean committed per live
    user, tok/s a user (the committed tokens a user got per round over the time of the rounds, step + gap) and the aggregate. A round whose gap is
    over idle_gap_ms followed idle time (no user was waiting), not a stall a user felt, and is counted apart."""
    kept = [item for item in rounds if item['gap_ms'] <= idle_gap_ms and item['live'] > 0]
    by_live = {}
    for item in kept:
        by_live.setdefault(item['live'], []).append(item)
    table = []
    for live, items in sorted(by_live.items()):
        seconds = sum(item['step_ms'] + item['gap_ms'] for item in items) / 1000.0
        committed = sum(item['committed'] for item in items)
        table.append(dict(live=live, rounds=len(items), step_ms=round(mean([item['step_ms'] for item in items]), 1),
                          gap_ms=round(mean([item['gap_ms'] for item in items]), 1),
                          committed_per_live=round(mean([item['committed'] / float(live) for item in items]), 2),
                          user_tok_s=round(committed / float(live) / seconds, 1) if seconds > 0 else None,
                          aggregate_tok_s=round(committed / seconds, 1) if seconds > 0 else None))
    return table, len(rounds) - len(kept)


def container_facts(entry, container_text, idle_gap_ms=IDLE_GAP_MS):
    """What the engine's own lines say about one agent_turns entry. The window is the log from the first line that names one of the run's requests
    to the last such line, then on through the [OCTO] rounds until they committed ROUND_COVER of the tokens the run streamed (its last answer's decode
    is logged after its last tagged line) or another request's prefill begins. Returns dict(readable, window=dict(lines), prefill, decode, live,
    rebind, turns={(session, turn): dict(reused, source, prefill_ms, steps, rebind_ms)})."""
    nonce = entry.get('nonce')
    wall_ms = ((entry.get('aggregate') or {}).get('wall_s') or 0) * 1000.0
    lines = (container_text or '').splitlines()
    tagged = [(number, tag_of(line, nonce)) for number, line in enumerate(lines) if nonce and 'chatcmpl-' in line]
    tagged = [(number, tag) for number, tag in tagged if tag is not None]
    if not tagged:
        return dict(readable=False, reason='no engine line names a request of this run (the server did not log lever N route / PREFIX / sticky lines '
                                           'with its request ids, or it ignored X-Request-Id)')
    first, last = tagged[0][0], tagged[-1][0]
    expected = sum(max((row.get('tokens') or 0) - 1, 0) for session in entry.get('sessions') or () for row in session.get('turns') or () if 'error' not in row)
    import octo_markers
    committed, end = 0, last
    for number in range(first, len(lines)):
        if number > last:
            if ROUTE.search(lines[number]) and tag_of(lines[number], nonce) is None:
                break           # another request's prefill: the run is over
            if committed >= ROUND_COVER * expected:
                break
        found = octo_markers.ROUND_PATTERN.search(lines[number])
        if found:
            committed += int(found.group(10))
        end = number
    window = lines[first:end + 1]
    text = '\n'.join(window)
    turns = {}
    steps = []
    for line in window:
        found = ROUTE.search(line)
        if found:
            tag = tag_of(line, nonce)
            step = dict(req=found.group(1), start=int(found.group(2)), end=int(found.group(3)), prompt=int(found.group(4)), final=found.group(5) == '1',
                        ms=float(found.group(6)), tag=tag)
            source = SOURCE.search(line)
            step['source'] = source.group(1) if source else None
            steps.append(step)
            if tag is not None:
                item = turns.setdefault(tag, dict(reused=None, source=None, prefill_ms=0.0, steps=0, rebind_ms=None, prompt=None))
                item['prefill_ms'] += step['ms']
                item['steps'] += 1
                if item['reused'] is None or step['start'] < item['reused']:
                    item['reused'], item['source'] = step['start'], step['source']
                item['prompt'] = step['prompt']
            continue
        found = STICKY.search(line)
        if found:
            tag = tag_of(line, nonce)
            if tag is not None:
                item = turns.setdefault(tag, dict(reused=None, source=None, prefill_ms=0.0, steps=0, rebind_ms=None, prompt=None))
                if found.group(5) == 'rebind':
                    item['rebind_ms'] = float(found.group(2))
            continue
        if PREFIX_ROW.search(line):
            tag = tag_of(line, nonce)
            quota, length = Q_FIELD.search(line), L_FIELD.search(line)
            if tag is not None and quota:
                item = turns.setdefault(tag, dict(reused=None, source=None, prefill_ms=0.0, steps=0, rebind_ms=None, prompt=None))
                if item['reused'] is None:       # the route's own start wins where Lever N printed one
                    item['reused'], item['source'] = int(quota.group(1)), 'PREFIX'
                    item['prompt'] = int(length.group(1)) if length else item['prompt']
    rounds = octo_markers.rounds(text)
    live, idle = live_rows(rounds, idle_gap_ms) if rounds else ([], 0)
    prefill_ms = sum(step['ms'] for step in steps)
    cold_ms = sum(item['prefill_ms'] for tag, item in turns.items() if item['reused'] == 0)
    resumed_ms = sum(item['prefill_ms'] for tag, item in turns.items() if item['reused'])
    step_ms = sum(item['step_ms'] for item in rounds)
    rebinds = [item['rebind_ms'] for item in turns.values() if item['rebind_ms'] is not None]
    return dict(readable=True, window=dict(lines=len(window), tagged_lines=len(tagged)),
                prefill=dict(route_steps=len(steps), ms=round(prefill_ms, 1), cold_ms=round(cold_ms, 1), resumed_ms=round(resumed_ms, 1),
                             share=round(prefill_ms / wall_ms, 3) if wall_ms and steps else None,
                             resumed_share=round(resumed_ms / wall_ms, 3) if wall_ms and steps else None,
                             cold_share=round(cold_ms / wall_ms, 3) if wall_ms and steps else None),
                decode=dict(rounds=len(rounds), step_ms=round(step_ms, 1), share=round(step_ms / wall_ms, 3) if wall_ms and rounds else None,
                            idle_rounds=idle),
                other_share=round(1.0 - (prefill_ms + step_ms) / wall_ms, 3) if wall_ms and steps and rounds else None,
                live=live, rebind=dict(count=len(rebinds), p50_ms=rounded(percentile(rebinds, 50), 0), p90_ms=rounded(percentile(rebinds, 90), 0)),
                turns=turns)


def reuse_evidence(entry, facts=None):
    """(returning turns, [reused tokens or None per returning turn], source): a turn's reuse is the larger of the usage's cached tokens and the
    server's resume point; None when neither is readable for it."""
    log_turns = (facts or {}).get('turns') or {}
    out, sources = [], set()
    for session in entry.get('sessions') or ():
        for row in session.get('turns') or ():
            if 'error' in row or not row.get('turn'):
                continue
            seen = []
            if isinstance(row.get('cached_tokens'), int):
                seen.append(row['cached_tokens'])
                sources.add('usage')
            item = log_turns.get((row['session'], row['turn']))
            if item and item.get('reused') is not None:
                seen.append(item['reused'])
                sources.add('log')
            out.append(max(seen) if seen else None)
    return out, '+'.join(sorted(sources)) or None


def judge(entry, env=None, container_text=''):
    """([problem], facts) for one agent_turns entry (module docstring, THE JUDGE). `env` is the served profile's environment (None: the reuse rules
    are not applied), `container_text` the container log ('' : the log facts are skipped). stream-level rules (no tokens, no finish, garbage) are
    c2_smoke_check's; this returns the shape's own."""
    if not isinstance(entry, dict):
        return ['%s: no result' % NAME], {}
    name = entry.get('test') or NAME
    if 'error' in entry:
        return ['%s: %s' % (name, entry['error'])], {}
    problems = []
    aggregate = entry.get('aggregate') or {}
    for session in entry.get('sessions') or ():
        for row in session.get('turns') or ():
            if 'error' in row:
                problems.append('%s session %s turn %s: %s' % (name, row.get('session'), row.get('turn'), row['error']))
    if not aggregate.get('requests_done'):
        problems.append('%s: no request completed' % name)
    elif not (aggregate.get('returning') or {}).get('turns'):
        problems.append('%s: no returning turn completed (the opening requests only)' % name)
    log = container_facts(entry, container_text) if container_text else None
    reused, source = reuse_evidence(entry, log if log and log.get('readable') else None)
    facts = dict(sessions=aggregate.get('sessions'), requests=aggregate.get('requests_done'), errors=aggregate.get('errors'),
                 wall_s=aggregate.get('wall_s'), committed_tok_s=aggregate.get('committed_tok_s'),
                 ttft_p50_s=(aggregate.get('returning') or {}).get('ttft_p50_s'), ttft_p90_s=(aggregate.get('returning') or {}).get('ttft_p90_s'),
                 salted=entry.get('salted'), stopped=sorted(set(session.get('stopped') for session in entry.get('sessions') or () if session.get('stopped'))))
    if log is not None:
        if log.get('readable'):
            facts['log'] = dict(prefill_share=log['prefill']['share'], prefill_resumed_share=log['prefill']['resumed_share'],
                                decode_share=log['decode']['share'], other_share=log['other_share'], live=log['live'], rebind_p50_ms=log['rebind']['p50_ms'])
        else:
            facts['log'] = 'unread: ' + log.get('reason', '')
    reuse_on = bool(env) and str(env.get('QWEN_PREFIX_REUSE')) == '1'
    returning = len(reused)
    if reuse_on:
        if not entry.get('salted') and entry.get('salt_mode') != 'none':
            problems.append('%s: the run sent no cache_salt on a prefix-reuse profile (QWEN_PREFIX_REUSE=1): reuse is inert unsalted and every returning turn '
                            're-prefills its whole history; the smoke step must mount a salt key (SMOKE_SALT_KEY_FILE), or the job asks for no salt on '
                            'purpose with SMOKE_AGENT_SALT=none' % name)
        elif entry.get('salted'):
            readable = [value for value in reused if value is not None]
            misses = [value for value in readable if value <= 0]
            facts['reuse'] = dict(source=source, returning_turns=returning, read=len(readable), reused_zero=len(misses))
            if readable and len(misses) > REUSE_MISS_SHARE * len(readable):
                problems.append('%s: %d of %d returning turns reused nothing on a salted prefix-reuse run (more than %d%%): the resume never happened '
                                '(cached tokens %s)' % (name, len(misses), len(readable), int(REUSE_MISS_SHARE * 100), source))
            if not readable:
                facts['reuse']['unread'] = True
    return problems, facts


# -- the report ---------------------------------------------------------------------------------------------------------------------------

def smoke_json(text):
    """The dict of the last SMOKE_JSON line of a smoke log, or None."""
    found = None
    for line in text.splitlines():
        marker = line.find('SMOKE_JSON ')
        if marker >= 0:
            try:
                found = json.loads(line[marker + len('SMOKE_JSON '):])
            except ValueError:
                pass
    return found


def percent(share):
    return 'n/a' if share is None else '%.1f%%' % (100.0 * share)


def format_report(entry, facts=None):
    """The entry (and the container log's facts, if any) as text: settings, per-session summaries, the turn table, the aggregate and the tables."""
    out = []
    aggregate = entry.get('aggregate') or {}
    out.append('%s seed=%s nonce=%s sessions=%s turns=%s gap_scale=%s salted=%s plan=%s' % (
        entry.get('test'), entry.get('seed'), entry.get('nonce'), entry.get('sessions_planned'), entry.get('turns_planned'), entry.get('gap_scale'),
        entry.get('salted'), entry.get('plan_digest')))
    out.append('estimate %s min; ran %.1f min (wall %s s); opening contexts %s' % (
        (entry.get('estimate') or {}).get('minutes'), (aggregate.get('wall_s') or 0) / 60.0, aggregate.get('wall_s'), entry.get('targets')))
    out.append('')
    out.append('PER SESSION   session  target  first -> last prompt  turns  decode tok/s  TTFT p50 / p90 (s)  opening TTFT (s)  stopped')
    for session in entry.get('sessions') or ():
        out.append('              %5d %7s  %9s -> %-9s %5s  %12s  %8s / %-8s  %14s  %s' % (
            session['session'], session['target_tokens'], session['prompt_tokens_first'], session['prompt_tokens_last'], session['turns_done'],
            session['mean_decode_tok_s'], session['ttft_p50_s'], session['ttft_p90_s'], session['opening_ttft_s'], session['stopped'] or '-'))
    out.append('')
    out.append('PER TURN      session turn   prompt   cached  re-prefill  appended  TTFT s  tokens  tok/s  finish')
    for session in entry.get('sessions') or ():
        for row in session.get('turns') or ():
            if 'error' in row:
                out.append('              %5d %4d  ERROR %s' % (row['session'], row['turn'], row['error']))
                continue
            out.append('              %5d %4d %8s %8s %11s %9s %7s %7s %6s  %s' % (
                row['session'], row['turn'], row.get('prompt_tokens'), row.get('cached_tokens', '-'), row.get('reprefill_tokens', '-'),
                row.get('appended_tokens', '-'), row.get('ttft'), row.get('tokens'), row.get('decode_tok_s'), row.get('finish')))
    out.append('')
    returning = aggregate.get('returning') or {}
    out.append('AGGREGATE     committed %s tok/s over %s s (%s tokens, %s errors); returning TTFT p50 %s s p90 %s s; decode tok/s mean %s p50 %s' % (
        aggregate.get('committed_tok_s'), aggregate.get('wall_s'), aggregate.get('completion_tokens'), aggregate.get('errors'),
        returning.get('ttft_p50_s'), returning.get('ttft_p90_s'), returning.get('decode_tok_s_mean'), returning.get('decode_tok_s_p50')))
    out.append('              TTFT by context: %s' % json.dumps(aggregate.get('ttft_by_context')))
    out.append('              reuse: %s' % json.dumps(aggregate.get('reuse')))
    steady = aggregate.get('steady')
    out.append('              steady window (every session built, none finished): %s' % (
        'none' if not steady else '%s s from +%s s, %s tok/s' % (steady['window_s'], steady['begin_s'], steady['committed_tok_s'])))
    out.append('')
    out.append('CLIENT LIVE TABLE (whole run)  live  seconds  tokens  tok/s a user  aggregate tok/s')
    for row in aggregate.get('live') or ():
        out.append('                               %4d %8s %7s %13s %16s' % (row['live'], row['seconds'], row['tokens'], row['user_tok_s'], row['aggregate_tok_s']))
    if facts is not None:
        out.append('')
        if not facts.get('readable'):
            out.append('CONTAINER LOG  unread: %s' % facts.get('reason'))
        else:
            prefill, decode = facts['prefill'], facts['decode']
            out.append('CONTAINER LOG  %(lines)s lines in the run window (%(tagged_lines)s name a request)' % facts['window'])
            out.append('               prefill %s of the wall (%s route steps, %s ms; resumed turns %s, cold builds %s)' % (
                percent(prefill['share']), prefill['route_steps'], prefill['ms'], percent(prefill['resumed_share']), percent(prefill['cold_share'])))
            out.append('               decode  %s of the wall (%s [OCTO] rounds, %s ms; %s idle-gap rounds left out of the table); other %s' % (
                percent(decode['share']), decode['rounds'], decode['step_ms'], decode['idle_rounds'], percent(facts['other_share'])))
            out.append('               rebind  %s' % json.dumps(facts['rebind']))
            out.append('SERVER LIVE TABLE  live  rounds  step_ms  gap_ms  committed/live  tok/s a user  aggregate tok/s')
            for row in facts['live']:
                out.append('                   %4d %7d %8s %7s %15s %13s %16s' % (row['live'], row['rounds'], row['step_ms'], row['gap_ms'],
                                                                                  row['committed_per_live'], row['user_tok_s'], row['aggregate_tok_s']))
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command')
    show = commands.add_parser('plan', help='print the deterministic plan and the run-time estimate (no server needed)')
    show.add_argument('--test', default=NAME, choices=TESTS)
    show.add_argument('--max-model-len', type=int, default=None)
    report = commands.add_parser('report', help='print the report of a finished smoke log (and container log)')
    report.add_argument('--smoke-log', required=True)
    report.add_argument('--container-log', default=None)
    report.add_argument('--test', default=NAME, choices=TESTS)
    report.add_argument('--json', default=None, help='also write the entry and the log facts here')
    options = parser.parse_args(argv)
    if options.command == 'plan':
        cfg = settings(options.test)
        the_plan = plan(cfg, options.max_model_len)
        print(json.dumps(dict(test=cfg['name'], sessions=cfg['sessions'], turns=cfg['turns'], seed=cfg['seed'], targets=the_plan['targets'],
                              clamped=the_plan['clamped'], plan_digest=plan_digest(the_plan), estimate=estimate(cfg, the_plan['targets'])), indent=1))
        return 0
    if options.command == 'report':
        with open(options.smoke_log, encoding='utf-8', errors='replace') as handle:
            results = smoke_json(handle.read())
        entry = (results or {}).get(options.test)
        if not isinstance(entry, dict) or 'error' in entry:
            print('no usable %s entry in %s: %s' % (options.test, options.smoke_log, (entry or {}).get('error') if isinstance(entry, dict) else entry))
            return 1
        facts = None
        if options.container_log:
            with open(options.container_log, encoding='utf-8', errors='replace') as handle:
                facts = container_facts(entry, handle.read())
        print(format_report(entry, facts))
        if options.json:
            with open(options.json, 'w', encoding='utf-8') as handle:
                json.dump(dict(entry=entry, log=dict((key, value) for key, value in (facts or {}).items() if key != 'turns')), handle)
        return 0
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
