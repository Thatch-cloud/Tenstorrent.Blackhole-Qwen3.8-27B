"""Smoke a C2 serving container over its OpenAI API (stdlib only; runs on the rig host)."""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8010'
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'Qwen/Qwen3.8-27B'
ONLY = set(filter(None, sys.argv[3].split(','))) if len(sys.argv) > 3 else None
results = {}


def post(path, body, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors='replace')[:400]


def stream(messages, max_tokens, drop_after=None, timeout=1800, keep_stamps=False, on_token=None, **extra):
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={'include_usage': True}, **extra)
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    started, first, pieces, usage, finish, count = time.time(), None, [], None, None, 0
    content, reasoning, stamps = [], [], []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', ()):
                delta = choice.get('delta') or {}
                text = (delta.get('content') or '') + (delta.get('reasoning_content') or delta.get('reasoning') or '')
                if text:
                    if first is None:
                        first = time.time()
                    stamps.append(time.time())
                    pieces.append(text)
                    content.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
                    count += 1
                    if on_token is not None:
                        on_token(count)     # the arrival gate of concurrent5_split reads the stream's progress here
                finish = choice.get('finish_reason') or finish
            if drop_after is not None and count >= drop_after:
                return dict(dropped_after=count, text=''.join(pieces)[:200])
    ended = time.time()
    tokens = (usage or {}).get('completion_tokens') or 0
    decode = (ended - first) if first else None
    # The full answer is kept as two hashes, one per field: the delta that carries </think> splits differently at a different
    # step width (16 rows packed, 4 solo), so a hash of the merged text would call identical tokens different.
    content, reasoning = ''.join(content), ''.join(reasoning)
    return dict(ttft=round(first - started, 2) if first else None, tokens=tokens,
                prompt_tokens=(usage or {}).get('prompt_tokens'), finish=finish,
                decode_tok_s=round((tokens - 1) / decode, 2) if decode and tokens > 1 else None,
                text=''.join(pieces)[:300], completion_tokens=tokens,
                content_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(), content_chars=len(content),
                reasoning_sha256=hashlib.sha256(reasoning.encode('utf-8')).hexdigest(), reasoning_chars=len(reasoning),
                started_at=round(started, 3), first_at=round(first, 3) if first else None, ended_at=round(ended, 3),
                **(dict(delta_stamps=stamps) if keep_stamps else {}))


def record(name, function):
    if ONLY and name not in ONLY:
        return
    started = time.time()
    try:
        value = function()
    except Exception as error:
        value = dict(error=repr(error)[:400])
    value = value if isinstance(value, dict) else dict(value=value)
    value['wall_s'] = round(time.time() - started, 1)
    results[name] = value
    print(name, json.dumps(value)[:900], flush=True)


def alive():
    status, body = post('/v1/chat/completions', dict(model=MODEL, max_tokens=4,
                        messages=[{'role': 'user', 'content': 'Say OK.'}]), timeout=300)
    return dict(status=status, text=(body['choices'][0]['message'].get('content') if status == 200 else body))


import sysconfig
source = open(sys.argv[4] if len(sys.argv) > 4 else sysconfig.get_paths()['stdlib'] + '/typing.py', encoding='utf-8').read()
coding = ('Here is a shell script:\n\n```bash\n' + open(sys.argv[5] if len(sys.argv) > 5 else __file__).read() * 3
          + '\n```\n\nRewrite it in Python 3 with argparse and subprocess, keeping every step. Output only code.')

record('warmup', lambda: post('/v1/chat/completions', dict(model=MODEL, max_tokens=1,
       messages=[{'role': 'user', 'content': 'warmup'}]))[0])


def warm_lifecycle():
    # The startup warmup (max_tokens 1) ends at its first token and builds no fast-path engine, so the first real request after
    # attach pays every per-request build: on the first four-card run (v140) an 8k prompt had TTFT 19.5 s, the same prompt
    # later 3.9 s. This drives one short answer at each prefill size (~170, ~500, ~1,000 and ~2,500 tokens: the 256/512/1024
    # buckets and the 2048 chunk with its tail) twice: the first pass is the cold cost per size, the second the warm one, and
    # everything after it is measured warm. Named explicitly in C2_SMOKE_TESTS ahead of coding / concurrent4.
    sizes = (600, 1800, 3600, 9000)
    passes = {}
    for label in ('cold', 'warm'):
        rows = []
        for size in sizes:
            prompt = 'Summarise this code in one sentence.\n\n' + source[:size]
            answer = stream([{'role': 'user', 'content': prompt}], 8)
            rows.append(dict(chars=size, prompt_tokens=answer.get('prompt_tokens'), ttft=answer.get('ttft'),
                             tokens=answer.get('tokens'), finish=answer.get('finish')))
        passes[label] = rows
    return dict(passes)


record('warm_lifecycle', warm_lifecycle)
record('coding', lambda: stream([{'role': 'user', 'content': coding}], 1500))
record('long_real_text', lambda: stream([{'role': 'user', 'content': 'Summarise what this module provides, '
       'section by section:\n\n' + source[:90000]}], 1200))


def live_report(users):
    """Per-user rates over the window all four users are live in, and the aggregate. `users` are stream() results with their
    delta stamps (keep_stamps=True). The all-four-live window runs from the LAST user's first token to the FIRST user's last:
    decode_tok_s (the existing clock) runs from each user's own first token, so the earlier users' figure includes the ramp
    while the later users were still prefilling. Each user's tokens in the window are its stamps in it, scaled by completion
    tokens over stamps (a delta can carry more than one token). The stamps are dropped from the results."""
    done = [user for user in users if isinstance(user, dict) and user.get('first_at') and user.get('delta_stamps')]
    window = None
    if len(done) == len(users):
        window = (max(user['first_at'] for user in done), min(user['ended_at'] for user in done))
    for user in done:
        stamps = user.pop('delta_stamps')
        user['live4_tok_s'] = None
        if window and window[1] - window[0] > 0.5:
            inside = sum(1 for stamp in stamps if window[0] < stamp <= window[1])
            user['live4_tok_s'] = round(inside * (user.get('tokens') or 0) / len(stamps) / (window[1] - window[0]), 2)
    for user in users:
        if isinstance(user, dict):
            user.pop('delta_stamps', None)
    rates = [user['live4_tok_s'] for user in done if user.get('live4_tok_s') is not None]
    tokens = sum(user.get('tokens') or 0 for user in done)
    first = min([user['started_at'] for user in done] or [0])
    last = max([user['ended_at'] for user in done] or [0])
    return dict(users=len(done), live4_window_s=round(window[1] - window[0], 2) if window else None,
                last_first_token_s=round(window[0] - first, 2) if window else None,
                live4_agg_tok_s=round(sum(rates), 2) if len(rates) == len(users) else None,
                live4_min_tok_s=min(rates) if rates else None,
                decode_agg_tok_s=round(sum(user.get('decode_tok_s') or 0 for user in done), 2),
                wall_tok_s=round(tokens / (last - first), 2) if last > first else None,
                ttft_max_s=max([user.get('ttft') or 0 for user in done] or [0]))


def run_users(label, prompts, max_tokens=800, order=None, stagger=0.0, timeout=1800, **extra):
    """The four-user shape (any user count): one streamed request per prompt, threads started in `order` (default the prompts'
    own) `stagger` seconds apart, per-read timeout `timeout`. users[i] is prompt i's answer, whatever the order. Per user: TTFT,
    decode tok/s on the existing clock and live4_tok_s (live_report: the window every user is live in, whatever the count);
    the aggregate is printed and returned first. `max_tokens` is one budget for all or a list, one per prompt; `extra` is
    added to every request body (the drain test's ignore_eos)."""
    out = [None] * len(prompts)
    budgets = list(max_tokens) if isinstance(max_tokens, (list, tuple)) else [max_tokens] * len(prompts)

    def run(index):
        try:
            out[index] = stream([{'role': 'user', 'content': prompts[index]}], budgets[index], timeout=timeout, keep_stamps=True,
                                **extra)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])

    threads = []
    for position, index in enumerate(order if order is not None else range(len(prompts))):
        if position and stagger:
            time.sleep(stagger)
        thread = threading.Thread(target=run, args=(index,))
        thread.start()
        threads.append(thread)
    [thread.join() for thread in threads]
    aggregate = live_report(out)
    print(label, 'AGGREGATE', json.dumps(aggregate), flush=True)
    for index, user in enumerate(out):
        print(label, 'user', index, json.dumps(dict((key, user.get(key)) for key in (
            'prompt_tokens', 'tokens', 'ttft', 'decode_tok_s', 'live4_tok_s', 'finish', 'error'))), flush=True)
    return dict(aggregate=aggregate, users=out)


def concurrent_prompts():
    return [coding, 'Write a Python LRU cache class with tests. ' * 40,
            'Explain this code:\n' + source[:30000], 'Write a Rust function that parses RFC 3339 dates, with tests.' * 30]


def concurrent():
    return run_users('concurrent4', concurrent_prompts())


record('concurrent4', concurrent)


def concurrent4_v164order():
    # Opt-in (named in the tests list). The same four prompts, threads started 0.25 s apart in v164's admission order: the
    # 8,376-token coding prompt, the 7,423-token explain prompt, then the 452-token LRU and 533-token Rust prompts (prompt
    # indices 0, 2, 1, 3). Slots 0 and 1 pair on the two long prompts before the short prompts' engines are built, which the
    # simultaneous start of concurrent4 does not reproduce. Per-read timeout 300 s: a stalled stream is an error, not a wait.
    return run_users('concurrent4_v164order', concurrent_prompts(), order=(0, 2, 1, 3), stagger=0.25, timeout=300)


if ONLY and 'concurrent4_v164order' in ONLY:
    record('concurrent4_v164order', concurrent4_v164order)


CODE_CHARS_PER_TOKEN = 3.6   # tp_decode_bench's estimate: the results carry the server's own prompt_tokens to calibrate it


def code_corpus():
    """Real code to build prompts from: the installed vLLM package source (the corpus of the real-text runs,
    real_text_prompts.build_corpus), else the tree named by SMOKE_CODE_ROOT, else the python stdlib (the smoke runs on the rig
    host, which has no vLLM). SMOKE_CODE_ROOT wins when set."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import real_text_prompts
    root = os.environ.get('SMOKE_CODE_ROOT')
    if not root:
        try:
            root = str(real_text_prompts.package_root('vllm'))
        except RuntimeError:
            root = sysconfig.get_paths()['stdlib']
    return real_text_prompts.build_corpus(root), real_text_prompts.TASKS


def code_prompts(targets):
    """One real-code prompt per target token count (about: characters at CODE_CHARS_PER_TOKEN). User i reads its own disjoint
    window of the corpus (from i * len // users, as real_text_prompts does) inside the repository framing, then a code task
    from real_text_prompts.TASKS (each asks for a long answer, so no user ends early and thins the rounds)."""
    (corpus, info), tasks = code_corpus()
    prompts = []
    for index, target in enumerate(targets):
        start = index * len(corpus) // len(targets)
        excerpt = corpus[start:start + int(target * CODE_CHARS_PER_TOKEN)]
        prompts.append('<repository_context>\n%s\n</repository_context>\n\n%s' % (excerpt, tasks[index % len(tasks)]))
    return prompts, dict(files=info['files'], characters=info['characters'])


def concurrent4_code():
    # Opt-in. The coding workload: four users on real code prompts of about 4k, 8k, 16k and 24k tokens, a code task each, 800
    # tokens out, started together.
    prompts, corpus = code_prompts((4096, 8192, 16384, 24576))
    return dict(run_users('concurrent4_code', prompts), corpus=corpus)


def concurrent4_code_equal():
    # Opt-in. Four real-code prompts of the same length (about 4k tokens), a code task each: the padded-4k lanes shape.
    prompts, corpus = code_prompts((4096,) * 4)
    return dict(run_users('concurrent4_code_equal', prompts), corpus=corpus)


def concurrent4_code_32k():
    # Opt-in. Four real-code prompts of about 32k tokens each (the long-context lane of the paired timing windows), a code task each,
    # 800 tokens out, started together: the four-user prefill queue at 32k, then the packed decode at that depth.
    prompts, corpus = code_prompts((32768,) * 4)
    return dict(run_users('concurrent4_code_32k', prompts), corpus=corpus)


def concurrent8_code():
    # Opt-in. Eight users, the target's standard seats: real code prompts of about 4k, 8k, 16k and 24k tokens twice over, a code
    # task each, 800 tokens out, started together. A profile with fewer seats queues the rest, which this measures too (the
    # all-live window is then empty and only the per-user clock and TTFT report).
    prompts, corpus = code_prompts((4096, 8192, 16384, 24576) * 2)
    return dict(run_users('concurrent8_code', prompts), corpus=corpus)


def steady_prompts(count=4):
    # About 3,500 tokens each: four (or `count`) different 14,000-character stretches of the same real code; the first four are
    # the same four whatever the count (concurrent8_steady's first four are concurrent4_steady's).
    span = 14000
    starts = [(index * span) % max(len(source) - span, 1) for index in range(count)]
    return ['Explain what this code does, then rewrite it with complete type annotations:\n\n' + source[start:start + span]
            for start in starts]


def concurrent4_steady():
    # Opt-in (named in the tests list). Four users, every one past the 2,048-row draft window from its first round (a prompt of
    # about 3,500 tokens each: four different 14,000-character stretches of the same real code), so the fixed slot pairs (0, 1)
    # and (2, 3) can pack and, with QWEN_FAST_QUAD_DRAFT=1, the four-user quad can draft. The mixed concurrent4 above holds two
    # short prompts in the ramp, which keep both pairs drafting singly for their whole answer: it can never show a batched draft.
    return run_users('concurrent4_steady', steady_prompts())


if ONLY and 'concurrent4_steady' in ONLY:
    record('concurrent4_steady', concurrent4_steady)


def concurrent8_steady():
    # Opt-in (tp4/seats8: the eight-user version of concurrent4_steady, the deterministic audits-off hang shape). Eight users,
    # every one past the 2,048-row draft window from its first round: eight different 14,000-character stretches of the same
    # real code (the first four are concurrent4_steady's), started together, so BOTH 64-row blocks run packed rounds, their
    # four draft pairs pack, and the pair, packed and commit traces replay under eight engines built while traces were live.
    return run_users('concurrent8_steady', steady_prompts(8))


if ONLY and 'concurrent8_steady' in ONLY:
    record('concurrent8_steady', concurrent8_steady)


def steady_resend():
    # Opt-in (named in the tests list, after concurrent4_steady): the first steady prompt again, alone. Its prefill reuses the
    # prompt geometry whose window-snapshot programs (keyed on the prompt's length, so no attach warm can cover them) the steady
    # test compiled after the packed traces were captured, and it runs after that test's packed and pair rounds replayed them:
    # a program compiled after the capture and reused after a replay is the four-card hang's sequence (#48536).
    return stream([{'role': 'user', 'content': steady_prompts()[0]}], 800)


if ONLY and 'steady_resend' in ONLY:
    record('steady_resend', steady_resend)
record('tool_call', lambda: post('/v1/chat/completions', dict(model=MODEL, max_tokens=400, tool_choice='auto',
       messages=[{'role': 'user', 'content': 'What is the weather in Wellington? Use the tool.'}],
       tools=[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
               'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}])))
def stream_events(messages, max_tokens, **extra):
    """One streamed chat completion, kept apart by kind (stream() merges content with reasoning): the content, the
    reasoning, the tool-call deltas accumulated by index ({name, arguments}), the finish reason, the delta count and the
    usage. Parser M (c2_parser_rechunk) runs on this path only."""
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={'include_usage': True}, **extra)
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    content, reasoning, calls, finish, deltas, usage = [], [], {}, None, 0, None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', ()):
                delta = choice.get('delta') or {}
                deltas += 1
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
                for call in delta.get('tool_calls') or ():
                    slot = calls.setdefault(call.get('index', 0), dict(name='', arguments=''))
                    function = call.get('function') or {}
                    slot['name'] += function.get('name') or ''
                    slot['arguments'] += function.get('arguments') or ''
                finish = choice.get('finish_reason') or finish
    return dict(content=''.join(content), reasoning=''.join(reasoning), calls=[calls[index] for index in sorted(calls)],
                finish=finish, deltas=deltas, tokens=(usage or {}).get('completion_tokens'))


def stream_tool_call():
    """Opt-in (named in the tests list): a STREAMED tool_choice auto request, the only path parser M re-chunks. The
    call must arrive as tool_calls deltas (a name and arguments that parse as JSON with the asked city) and the
    <tool_call> markers must not leak into the content."""
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
              'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    got = stream_events([{'role': 'user', 'content': 'What is the weather in Wellington? Use the tool.'}], 400,
                        tool_choice='auto', tools=tools)
    problems = []
    if not got['calls']:
        problems.append('no tool_calls in the stream (content %r)' % got['content'][:120])
    else:
        call = got['calls'][0]
        if call['name'] != 'get_weather':
            problems.append('the call names %r, not get_weather' % call['name'])
        try:
            arguments = json.loads(call['arguments'])
        except ValueError:
            arguments = None
            problems.append('the call arguments are not JSON: %r' % call['arguments'][:120])
        if isinstance(arguments, dict) and 'wellington' not in str(arguments.get('city', '')).lower():
            problems.append('the call asks for %r, not Wellington' % arguments.get('city'))
    for marker in ('<tool_call>', '</tool_call>'):
        if marker in got['content']:
            problems.append('%s leaked into the content' % marker)
    return dict(ok=not problems, problems=problems, calls=got['calls'][:2], finish=got['finish'], deltas=got['deltas'],
                tokens=got['tokens'], content=got['content'][:120])


def stream_reasoning():
    """Opt-in: a streamed thinking-on prompt. The reasoning arrives in reasoning_content and only the answer in content:
    a </think> in the content means the parser lost the marker at a multi-token delta (the defect parser M fixes)."""
    # 4096, not the answer's size: the reasoning comes first and a verbose one must not end the stream before any content
    # (T1's coding answer spent all of its 1,500 tokens reasoning), which would fail a stop job on the budget, not the parser.
    got = stream_events([{'role': 'user', 'content': 'How many prime numbers are there below 30? Think first.'}], 4096)
    problems = []
    if not got['reasoning'].strip():
        problems.append('no reasoning_content in the stream')
    for marker in ('</think>', '<think>'):
        if marker in got['content']:
            problems.append('%s leaked into the content' % marker)
    if not got['content'].strip():
        problems.append('no answer content after the reasoning (finish %s, %s tokens)' % (got['finish'], got['tokens']))
    return dict(ok=not problems, problems=problems, finish=got['finish'], deltas=got['deltas'], tokens=got['tokens'],
                reasoning=got['reasoning'][:120], content=got['content'][:120])


if ONLY and 'stream_tool_call' in ONLY:
    record('stream_tool_call', stream_tool_call)
if ONLY and 'stream_reasoning' in ONLY:
    record('stream_reasoning', stream_reasoning)
record('refused_n2', lambda: dict(status=post('/v1/chat/completions', dict(model=MODEL, n=2, max_tokens=8,
       messages=[{'role': 'user', 'content': 'hi'}]))[0]))
record('alive_after_refusal', alive)
record('stream_dropped', lambda: stream([{'role': 'user', 'content': coding}], 1500, drop_after=20))
time.sleep(5)
record('alive_after_drop', alive)


def concurrent4_solo():
    # Opt-in (named in the tests list, after alive_after_drop): each concurrent4 prompt alone, at the same 800-token budget, so
    # the full answers (content and reasoning hashes) are this image's own solo references for concurrent4's four users.
    # c2_smoke_check.solo_problems compares them.
    return dict(users=[stream([{'role': 'user', 'content': prompt}], 800) for prompt in concurrent_prompts()])


REPLAY_PARSERS = ('CSV', 'JSON', 'INI', 'TOML', 'YAML', 'XML', 'TSV', 'HTML')


def replay_users(count):
    # c2_platform_replay's concurrent shape: `count` NON-streamed requests, one short prompt each (a unit test for the first
    # `count` of REPLAY_PARSERS' parsers), 300 tokens, all at once: the traffic shape the platform sends.
    out = [None] * count

    def run(index):
        started = time.time()
        try:
            status, body = post('/v1/chat/completions', dict(model=MODEL, max_tokens=300, messages=[{'role': 'user',
                                'content': 'Write a unit test for a %s parser in Python.' % REPLAY_PARSERS[index]}]))
            wall = time.time() - started
            tokens = ((body.get('usage') or {}).get('completion_tokens') or 0) if status == 200 else 0
            out[index] = dict(status=status, tokens=tokens, wall_s=round(wall, 1), tok_s_e2e=round(tokens / wall, 2),
                              finish=body['choices'][0].get('finish_reason') if status == 200 else body)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    return dict(users=out)


def replay_concurrent4():
    # Opt-in (after concurrent4_solo): c2_platform_replay's concurrent4 - four NON-streamed requests (CSV, JSON, INI, TOML).
    return replay_users(4)


def replay_concurrent8():
    # Opt-in (tp4/seats8): the eight-user version of replay_concurrent4, the deterministic audits-off hang shape in the traffic
    # the platform sends: eight non-streamed requests (the four above and YAML, XML, TSV, HTML), 300 tokens, all at once.
    return replay_users(8)


if ONLY and 'concurrent4_solo' in ONLY:
    record('concurrent4_solo', concurrent4_solo)
if ONLY and 'replay_concurrent4' in ONLY:
    record('replay_concurrent4', replay_concurrent4)
# The coding-text tests run after every like-for-like test (the smoke runs in source order, not list order), so the earlier
# part keeps v166's order and clock.
if ONLY and 'concurrent4_code' in ONLY:
    record('concurrent4_code', concurrent4_code)
if ONLY and 'concurrent4_code_equal' in ONLY:
    record('concurrent4_code_equal', concurrent4_code_equal)
if ONLY and 'concurrent4_code_32k' in ONLY:
    record('concurrent4_code_32k', concurrent4_code_32k)
if ONLY and 'concurrent8_code' in ONLY:
    record('concurrent8_code', concurrent8_code)


def concurrent8_code_equal():
    # Opt-in (tp4/seats8). Eight real-code prompts of the same length (about 4k tokens), a code task each, 800 out, started together:
    # both blocks padded-equal at once; the audited S8-1 run reads zero mismatches and block B's users equal to solo.
    prompts, corpus = code_prompts((4096,) * 8)
    return dict(run_users('concurrent8_code_equal', prompts), corpus=corpus)


def concurrent8_code_32k():
    # Opt-in (tp4/seats8). Eight real-code prompts of about 32k tokens each, 800 out, started together: the eight-user prefill queue
    # at 32k, then both blocks' packed decode at that depth.
    prompts, corpus = code_prompts((32768,) * 8)
    return dict(run_users('concurrent8_code_32k', prompts), corpus=corpus)


DRAIN_BUDGETS = (200, 400, 600, 800, 1000, 1200, 1400, 1600)
SPLIT_BUDGETS = (1600, 1600, 1600, 1600, 1200, 400)
SPLIT_TOKENS = 320          # user 4's answer before the sixth user arrives: at least 50 rounds at tau up to 6.5


def concurrent8_drain():
    # Opt-in (tp4/seats8). Eight 4k real-code prompts, budgets 200, 400 ... 1,600 with ignore_eos, started together: the users end
    # one by one at their own budgets, so the live count goes 8 to 1 through every split of the two blocks the placement allows
    # (a block narrowing to one live user while the other still runs packed, then the last block emptying).
    prompts, corpus = code_prompts((4096,) * 8)
    return dict(run_users('concurrent8_drain', prompts, max_tokens=list(DRAIN_BUDGETS), ignore_eos=True), corpus=corpus,
                budgets=list(DRAIN_BUDGETS))


def concurrent5_split():
    # Opt-in (tp4/seats8). The split-block shape: users 0-3 (block A) start together; user 4 arrives alone in block B once all four
    # stream and runs at least SPLIT_TOKENS tokens (50+ rounds with its block narrowed to one live user, the other block packed);
    # then a sixth arrives into block B (a pair, narrow to padded). Every user ignores EOS, budgets 1,600 x 4, 1,200, 400. A user that
    # dies releases the next arrival (never a deadlock): user 4's death records its token count as None.
    prompts, corpus = code_prompts((4096,) * 6)
    out = [None] * 6
    streaming = [threading.Event() for _ in range(5)]      # users 0-4: set at the first token or at the end (a dead user too)
    split = dict(tokens=None)
    split_ready = threading.Event()

    def run(index, on_token=None):
        try:
            out[index] = stream([{'role': 'user', 'content': prompts[index]}], SPLIT_BUDGETS[index], on_token=on_token,
                                ignore_eos=True)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])
        finally:
            if index < 5:
                streaming[index].set()
            if index == 4:
                if split['tokens'] is None and isinstance(out[4], dict) and 'error' not in out[4]:
                    split['tokens'] = out[4].get('tokens')    # it ended before SPLIT_TOKENS: the count it reached
                split_ready.set()

    def first_token(index):
        def seen(count):
            streaming[index].set()
        return seen

    def fifth_token(count):
        streaming[4].set()
        if count == SPLIT_TOKENS and split['tokens'] is None:
            split['tokens'] = count
            split_ready.set()

    threads = [threading.Thread(target=run, args=(index, first_token(index))) for index in range(4)]
    [thread.start() for thread in threads]
    for event in streaming[:4]:
        event.wait(1800)
    fifth = threading.Thread(target=run, args=(4, fifth_token))
    fifth.start()
    split_ready.wait(1800)
    sixth = threading.Thread(target=run, args=(5,))
    sixth.start()
    [thread.join() for thread in threads + [fifth, sixth]]
    print('concurrent5_split user4_tokens_at_sixth_arrival', split['tokens'], flush=True)
    for index, user in enumerate(out):
        print('concurrent5_split user', index, json.dumps(dict((key, user.get(key)) for key in (
            'prompt_tokens', 'tokens', 'ttft', 'decode_tok_s', 'finish', 'error'))), flush=True)
    return dict(users=out, corpus=corpus, budgets=list(SPLIT_BUDGETS), user4_tokens_at_sixth_arrival=split['tokens'])


if ONLY and 'concurrent8_code_equal' in ONLY:
    record('concurrent8_code_equal', concurrent8_code_equal)
if ONLY and 'concurrent8_code_32k' in ONLY:
    record('concurrent8_code_32k', concurrent8_code_32k)
if ONLY and 'replay_concurrent8' in ONLY:
    record('replay_concurrent8', replay_concurrent8)
if ONLY and 'concurrent5_split' in ONLY:
    record('concurrent5_split', concurrent5_split)
if ONLY and 'concurrent8_drain' in ONLY:
    record('concurrent8_drain', concurrent8_drain)


def agreement():
    # Opt-in (named in the tests list): greedy answers to real repository text, with per-token log-probabilities, saved
    # for tp_agreement.py compare against another tensor-parallel configuration (AGREEMENT_OUT, else agreement.json).
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tp_agreement
    out = tp_agreement.collect(BASE, MODEL, (os.environ.get('AGREEMENT_ROOT') or '.'), os.environ.get('AGREEMENT_LABEL', 'run'))
    with open(os.environ.get('AGREEMENT_OUT', 'agreement.json'), 'w') as handle:
        json.dump(out, handle)
    # Every answer twice (host-sampled with logprobs, device-sampled without): self_check fails the step on an errored
    # or incoherent answer, a device sampler that departs from the host's outside a bfloat16 near-tie, or a garbage
    # perplexity. It cannot say the answers are RIGHT: tp_agreement.py compare against the reference does.
    ok, reasons = tp_agreement.self_check(out)
    return dict(ok=ok, reasons=reasons[:8],
                prompts=[dict(name=p['name'], tokens=len(p['tokens']), logprobs=p['logprobs_available'],
                              finish=p['finish'], error=p.get('error'),
                              device_vs_host=tp_agreement.device_check(p)['ok']) for p in out['prompts']])


if ONLY and 'agreement' in ONLY:
    record('agreement', agreement)


def bench():
    # Opt-in (named in the tests list): tp_decode_bench.py's shapes (BENCH_SHAPES, else its default ladder) against
    # this server, saved as BENCH_OUT (else bench.json) for the pair-versus-TP4 comparison.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tp_decode_bench
    shapes = [tp_decode_bench.parse_shape(part) for part in
              (os.environ.get('BENCH_SHAPES') or ','.join(tp_decode_bench.DEFAULT_SHAPES)).split(',') if part]
    out = tp_decode_bench.bench(BASE, MODEL, (os.environ.get('AGREEMENT_ROOT') or '.'),
                                os.environ.get('AGREEMENT_LABEL', 'run'), shapes)
    with open(os.environ.get('BENCH_OUT', 'bench.json'), 'w') as handle:
        json.dump(out, handle)
    return dict(shapes=[dict(name=s['name'], median=s.get('median_decode_tok_s'), error=s.get('error'))
                        for s in out['shapes']])


if ONLY and 'bench' in ONLY:
    record('bench', bench)
print('SMOKE_JSON ' + json.dumps(results))
if (results.get('agreement') or {}).get('ok') is False or 'error' in (results.get('agreement') or {}):
    print('SMOKE_FAILED agreement: %s' % json.dumps((results['agreement'].get('reasons') or results['agreement'].get('error')))[:600])
    sys.exit(1)
