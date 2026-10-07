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
    return dict(ttft=round(first - started, 2) if first else None, tokens=tokens, chunks_streamed=count,
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


def code_prompts(targets, chars_per_token=None):
    """One real-code prompt per target token count (about: characters at CODE_CHARS_PER_TOKEN, or at `chars_per_token`: one number
    for all or a list, one per prompt - fitted_code_prompts calibrates it against the server). User i reads its own disjoint
    window of the corpus (from i * len // users, as real_text_prompts does) inside the repository framing, then a code task
    from real_text_prompts.TASKS (each asks for a long answer, so no user ends early and thins the rounds)."""
    (corpus, info), tasks = code_corpus()
    ratios = (list(chars_per_token) if isinstance(chars_per_token, (list, tuple))
              else [CODE_CHARS_PER_TOKEN if chars_per_token is None else chars_per_token] * len(targets))
    prompts = []
    for index, target in enumerate(targets):
        start = index * len(corpus) // len(targets)
        excerpt = corpus[start:start + int(target * ratios[index])]
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


def server_tokens(prompt):
    """The server's own token count of a prompt (POST /tokenize, vLLM's OpenAI server), or None where it cannot say (no endpoint,
    a failure): the rig host has no tokenizer, and a character estimate at CODE_CHARS_PER_TOKEN is only an estimate."""
    try:
        status, body = post('/tokenize', dict(model=MODEL, messages=[{'role': 'user', 'content': prompt}]), timeout=600)
    except Exception:
        return None
    if status == 200 and isinstance(body, dict) and isinstance(body.get('count'), int):
        return body['count']
    return None


def fitted_code_prompts(targets, tolerance=0.01, rounds=4):
    """code_prompts calibrated to the SERVER's token counts: each prompt is rebuilt at the characters-per-token its own count
    showed until it is within `tolerance` of its target and never above it (a prompt over the profile's limit is a 400, not a
    measurement). Where the server cannot count (server_tokens is None) the estimate stands and `calibrated` says so. Returns
    (prompts, corpus info, dict(calibrated, counts, targets, ratios))."""
    ratios = [CODE_CHARS_PER_TOKEN] * len(targets)
    prompts, corpus = code_prompts(targets, ratios)
    counts = [server_tokens(prompt) for prompt in prompts]
    calibrated = all(count is not None for count in counts)
    for _ in range(rounds if calibrated else 0):
        pending = [index for index, (count, target) in enumerate(zip(counts, targets))
                   if count > target or count < target * (1 - tolerance)]
        if not pending:
            break
        for index in pending:
            # the characters each token cost this prompt, aimed at the target (a hair under it when it was over)
            ratios[index] = ratios[index] * targets[index] / counts[index] * (0.998 if counts[index] > targets[index] else 1.0)
        prompts, corpus = code_prompts(targets, ratios)
        counts = [server_tokens(prompt) for prompt in prompts]
        calibrated = all(count is not None for count in counts)
        if not calibrated:
            break
    return prompts, corpus, dict(calibrated=calibrated, counts=counts, targets=list(targets),
                                 ratios=[round(ratio, 4) for ratio in ratios])


def concurrent8_code_128k():
    # Opt-in (tp4/seats262k). Eight real-code prompts of about 120,000 tokens AS THE SERVER COUNTS THEM (fitted_code_prompts: /tokenize,
    # never above the target), a code task each, 800 out, started together. About 120,000 and not 131,072: the 131k time-gate profile's
    # room is 131,072 less the answer, so the same arm runs on the 131k and the 262k eight-seat profiles for the paired timing
    # (the 262k round must cost at most 1.03 x the 131k one), and the 262k profile reads the deep per-seat rate: 8 x 120,800 tokens
    # reserve 15,112 blocks of its pool, so every seat is live at once.
    prompts, corpus, fit = fitted_code_prompts((DEEP_PROMPT_TOKENS,) * 8)
    return dict(run_users('concurrent8_code_128k', prompts), corpus=corpus, fit=fit)


SKEW_LONG_TOKENS = 253920           # the stall arm's cold prompt: known to fit the 262k profile's room with a short answer
SKEW_SHORT_TOKENS = 4096
SKEW_TARGETS = (SKEW_LONG_TOKENS, SKEW_SHORT_TOKENS, SKEW_SHORT_TOKENS, SKEW_SHORT_TOKENS,
                SKEW_LONG_TOKENS, SKEW_SHORT_TOKENS, SKEW_SHORT_TOKENS, SKEW_SHORT_TOKENS)


def concurrent8_skew():
    # Opt-in (tp4/w2). The skewed eight: two real-code prompts of about 253,920 tokens (users 0 and 4, one in each four-seat block) and six of about
    # 4,096, as the server counts them (fitted_code_prompts), a code task each, 800 out, started together. A multi-user SDPA launch on a fixed block
    # geometry is paid at its longest user, so this is the shape where it can lose against per-user launches (the plan's row 7s: up to 9 ms a round at
    # heavy skew); the equal-length arms cannot show it.
    prompts, corpus, fit = fitted_code_prompts(SKEW_TARGETS)
    return dict(run_users('concurrent8_skew', prompts), corpus=corpus, fit=fit)


DEEP_PROMPT_TOKENS = 120000
STALL_SHORT_SEATS = 7
STALL_SHORT_TOKENS = 4096
STALL_SHORT_BUDGET = 6000
STALL_COLD_TOKENS = 253920
STALL_COLD_128K_TOKENS = 120000
STALL_WARM_CHUNKS = 12              # each decoding seat streams this many chunks before the cold prompt arrives


def longest_gap(stamps, after=None):
    """(the longest gap in seconds between consecutive delta stamps, the stamp it began at); `after` keeps only the gaps that end
    after that time (the cold arrival's). (None, None) with fewer than two stamps."""
    best, at = None, None
    for first, second in zip(stamps, stamps[1:]):
        if after is not None and second <= after:
            continue
        if best is None or second - first > best:
            best, at = second - first, first
    return (round(best, 3), round(at, 3)) if best is not None else (None, None)


def window_stats(stamps, tokens, chunks, begin, end):
    """One seat's progress inside [begin, end], the cold arrival's prefill window (Lever N, tp4/lever-n): the stream chunks it received there,
    chunks a second, and an ESTIMATE of its tokens a second (the stream carries one stamp per chunk, not per token, so the seat's own
    tokens-per-chunk over the whole stream scales the chunk rate: a speculative round commits a few tokens at once). None where the window is empty."""
    if not (isinstance(begin, (int, float)) and isinstance(end, (int, float))) or end <= begin:
        return dict(window_s=None, chunks=None, chunk_rate=None, est_tok_s=None)
    inside = len([stamp for stamp in stamps or () if begin < stamp <= end])
    span = end - begin
    per_chunk = (tokens / chunks) if tokens and chunks else None
    return dict(window_s=round(span, 3), chunks=inside, chunk_rate=round(inside / span, 3),
                est_tok_s=round(inside / span * per_chunk, 2) if per_chunk else None)


def window_aggregate(seat_windows):
    """The seats' in-window progress together: how many progressed at all, the slowest chunk rate, and the summed estimated tokens a second."""
    known = [window for window in seat_windows if window.get('chunks') is not None]
    return dict(seats=len(known), seats_progressing=len([window for window in known if window['chunks'] > 0]),
                min_chunk_rate=min([window['chunk_rate'] for window in known] or [None]) if known else None,
                total_est_tok_s=round(sum(window['est_tok_s'] or 0.0 for window in known), 2) if known else None)


def stall8_cold262k():
    return stall8_cold('stall8_cold262k', STALL_COLD_TOKENS)


def stall8_cold128k():
    # Opt-in (tp4/lever-n). The same shape with a cold arrival of about 120,000 tokens (as the server counts them): half the freeze of the 254k one.
    return stall8_cold('stall8_cold128k', STALL_COLD_128K_TOKENS)


def stall8_cold(label, cold_tokens):
    # Opt-in (tp4/seats262k). The stall shape of a cold long arrival: seven users decode 4k prompts (ignore_eos, 6,000-token budgets,
    # so they outlast the arrival's prefill), and once each has streamed STALL_WARM_CHUNKS chunks an eighth user arrives with a
    # cold_tokens-token prompt. Recorded, not gated: the arrival's time to first token, every decoding seat's longest inter-token gap
    # from the arrival on (the prefill's admission freeze: a long prefill holds the gate and no live user decodes), and (Lever N,
    # tp4/lever-n) each seat's chunks and estimated tokens a second INSIDE the arrival's prefill window (arrival to the newcomer's first
    # token): zero on the control arm, which freezes every decoder, and the interleaved arm's whole point.
    prompts, corpus = code_prompts((STALL_SHORT_TOKENS,) * STALL_SHORT_SEATS)
    cold, cold_corpus, fit = fitted_code_prompts((cold_tokens,))
    out = [None] * (STALL_SHORT_SEATS + 1)
    streaming = [threading.Event() for _ in range(STALL_SHORT_SEATS)]
    arrival = dict(at=None)

    def run(index, prompt, budget, event=None, **extra):
        counted = dict(chunks=0)

        def seen(count):
            counted['chunks'] = count
            if event is not None and count >= STALL_WARM_CHUNKS:
                event.set()

        try:
            out[index] = stream([{'role': 'user', 'content': prompt}], budget, timeout=3600, keep_stamps=True, on_token=seen,
                                **extra)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])
        finally:
            if event is not None:
                event.set()

    threads = [threading.Thread(target=run, args=(index, prompts[index], STALL_SHORT_BUDGET, streaming[index]),
                                kwargs=dict(ignore_eos=True)) for index in range(STALL_SHORT_SEATS)]
    [thread.start() for thread in threads]
    for event in streaming:
        event.wait(3600)
    arrival['at'] = time.time()
    cold_thread = threading.Thread(target=run, args=(STALL_SHORT_SEATS, cold[0], 200))
    cold_thread.start()
    [thread.join() for thread in threads + [cold_thread]]
    newcomer = out[STALL_SHORT_SEATS] if isinstance(out[STALL_SHORT_SEATS], dict) else {}
    window_end = newcomer.get('first_at')
    gaps, windows = [], []
    for index in range(STALL_SHORT_SEATS):
        user = out[index] if isinstance(out[index], dict) else {}
        stamps = user.pop('delta_stamps', None) or []
        gap, began = longest_gap(stamps, after=arrival['at'])
        gaps.append(dict(seat=index, longest_gap_s=gap, began_at=began, tokens=user.get('tokens'), error=user.get('error')))
        windows.append(dict(seat=index, **window_stats(stamps, user.get('tokens'), user.get('chunks_streamed'), arrival['at'], window_end)))
    newcomer.pop('delta_stamps', None)
    worst = max([gap['longest_gap_s'] for gap in gaps if gap['longest_gap_s'] is not None] or [None]) if any(
        gap['longest_gap_s'] is not None for gap in gaps) else None
    aggregate = window_aggregate(windows)
    print(label, 'arrival_ttft_s', arrival_ttft(newcomer), 'worst_gap_s', worst, 'window', json.dumps(aggregate), flush=True)
    return dict(users=out, corpus=corpus, fit=fit, arrival_started_at=round(arrival['at'], 3),
                arrival_ttft_s=arrival_ttft(newcomer), arrival_prompt_tokens=newcomer.get('prompt_tokens'),
                seat_gaps=gaps, longest_gap_s=worst, seat_windows=windows, window=aggregate)


COLD2_DECODERS = 6                    # six decoders + two cold arrivals = the eight seats of the profile (never a ninth request: it would queue)
COLD2_ARRIVALS = 2
COLD2_BUDGET = 6000                  # ignore_eos, so the decoders outlast both prefills

# Every shape's concurrent request count, held at or under the eight seats of the 262k profiles by test_w2ln_smoke_shapes (a request past max-num-seqs
# waits for a seat, and its time to first token then measures somebody else's budget).
SHAPE_SEATS = {'concurrent8_skew': 8, 'concurrent8_code_32k': 8, 'concurrent8_code_128k': 8,
               'stall8_cold262k': STALL_SHORT_SEATS + 1, 'stall8_cold128k': STALL_SHORT_SEATS + 1,
               'cold2_254k': COLD2_DECODERS + COLD2_ARRIVALS, 'levern_equal_busy': 7 + 1, 'levern_decoder_finishes': 7 + 1}


def cold2_254k():
    # Opt-in (the combined window). Two SIMULTANEOUS cold arrivals of about 253,920 tokens (as the server counts them) while six seats decode 4k prompts
    # (ignore_eos, 6,000-token budgets): the arrival burst of the merged route (the governor runs at f = 1.0 when two long prompts are owed, so the
    # decoders' gaps are not the single-arrival shape's). Recorded, not gated: each arrival's TTFT, every decoder's longest gap from the arrival on, and
    # each decoder's chunks inside the first arrival's window.
    out, events, threads, corpus = start_decoders([COLD2_BUDGET] * COLD2_DECODERS)
    for event in events:
        event.wait(3600)
    cold, cold_corpus, fit = fitted_code_prompts((STALL_COLD_TOKENS,) * COLD2_ARRIVALS)
    newcomers = [None] * COLD2_ARRIVALS
    arrival = time.time()
    colds = [threading.Thread(target=run_cold, args=(index, newcomers, cold[index], 200)) for index in range(COLD2_ARRIVALS)]
    [thread.start() for thread in colds]
    [thread.join() for thread in threads + colds]
    firsts = [item.get('first_at') for item in newcomers if isinstance(item, dict) and item.get('first_at')]
    window_end = min(firsts) if firsts else None
    gaps, windows = [], []
    for index in range(COLD2_DECODERS):
        user = out[index] if isinstance(out[index], dict) else {}
        stamps = user.pop('delta_stamps', None) or []
        gap, began = longest_gap(stamps, after=arrival)
        gaps.append(dict(seat=index, longest_gap_s=gap, began_at=began, tokens=user.get('tokens'), error=user.get('error')))
        windows.append(dict(seat=index, **window_stats(stamps, user.get('tokens'), user.get('chunks_streamed'), arrival, window_end)))
    arrivals = []
    for item in newcomers:
        item = item if isinstance(item, dict) else {}
        item.pop('delta_stamps', None)
        arrivals.append(dict(ttft_s=arrival_ttft(item), prompt_tokens=item.get('prompt_tokens'), error=item.get('error'),
                             first_at=item.get('first_at')))
    known = [gap['longest_gap_s'] for gap in gaps if gap['longest_gap_s'] is not None]
    worst = max(known) if known else None
    print('cold2_254k arrivals', json.dumps(arrivals), 'worst_gap_s', worst, 'window', json.dumps(window_aggregate(windows)), flush=True)
    return dict(users=out + newcomers, corpus=corpus, fit=fit, arrival_started_at=round(arrival, 3), arrivals=arrivals, seat_gaps=gaps,
                longest_gap_s=worst, seat_windows=windows, window=window_aggregate(windows))


def arrival_ttft(newcomer):
    return newcomer.get('ttft') if isinstance(newcomer, dict) else None


# ---------------------------------------------------------------------------------------------------------------------
# Lever N (tp4/lever-n): exactness of a split prefill against the whole one, and the hang shapes of an interleaved prefill.
# ---------------------------------------------------------------------------------------------------------------------

LEVERN_EQUAL_LENGTHS = (2047, 2048, 2049, 4095, 4096, 4097, 6143, 6145, 32785)
LEVERN_EQUAL_LONG_LENGTHS = (131077, 253920)
LEVERN_BUSY_LENGTHS = (4097, 6145, 32785)
LEVERN_EQUAL_TOKENS = 256
LEVERN_BUSY_SEATS = 7
LEVERN_BUSY_BUDGET = 4000
LEVERN_COLD_TOKENS = 64000


def server_token_ids(text):
    """The server's own token ids of `text` (POST /tokenize with a raw prompt, vLLM's OpenAI server), or None where it cannot say."""
    try:
        status, body = post('/tokenize', dict(model=MODEL, prompt=text, add_special_tokens=False), timeout=1200)
    except Exception:
        return None
    if status == 200 and isinstance(body, dict) and isinstance(body.get('tokens'), list):
        return body['tokens']
    return None


def exact_token_prompts(lengths, margin=1.3):
    """One prompt of EXACTLY each length in tokens, as token ids: real code text tokenized by the server (the rig host has no tokenizer),
    cut at the token. Each prompt reads its own window of the corpus. Returns (ids per length, corpus info); a length the corpus cannot
    reach, or a server that cannot tokenize, is an error naming it (a boundary test that ran at another length would test nothing)."""
    (corpus, info), _ = code_corpus()
    prompts = []
    for index, length in enumerate(lengths):
        chars = int(length * CODE_CHARS_PER_TOKEN * margin) + 4000
        for _ in range(6):
            chars = min(chars, len(corpus))
            start = (index * len(corpus) // len(lengths)) % max(len(corpus) - chars, 1)
            ids = server_token_ids(corpus[start:start + chars])
            if ids is None:
                raise RuntimeError('the server cannot tokenize (POST /tokenize with a prompt): no exact-length prompt for %d tokens' % length)
            if len(ids) >= length:
                prompts.append(ids[:length])
                break
            if chars >= len(corpus):
                raise RuntimeError('the corpus holds %d tokens of text at most: no prompt of %d tokens' % (len(ids), length))
            chars = int(chars * 1.4)
        else:
            raise RuntimeError('no prompt of %d tokens in the corpus' % length)
    return prompts, dict(files=info['files'], characters=info['characters'])


def complete_ids(ids, max_tokens, timeout=3600):
    """One non-streamed completion of a prompt given as token ids (POST /v1/completions): the answer's hashes, token count, finish reason and
    the server's own prompt token count, which must be len(ids)."""
    started = time.time()
    status, body = post('/v1/completions', dict(model=MODEL, prompt=ids, max_tokens=max_tokens), timeout=timeout)
    if status != 200:
        return dict(status=status, error=str(body)[:300], prompt_tokens_sent=len(ids))
    choice = body['choices'][0]
    text = choice.get('text') or ''
    usage = body.get('usage') or {}
    return dict(status=status, prompt_tokens_sent=len(ids), prompt_tokens=usage.get('prompt_tokens'), tokens=usage.get('completion_tokens'),
                finish=choice.get('finish_reason'), content_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), content_chars=len(text),
                text=text[:200], wall_s=round(time.time() - started, 1))


def levern_equal_run(label, lengths):
    prompts, corpus = exact_token_prompts(lengths)
    rows = {}
    for length, ids in zip(lengths, prompts):
        row = complete_ids(ids, LEVERN_EQUAL_TOKENS)
        rows[str(length)] = row
        print(label, 'prompt', length, json.dumps(dict((key, row.get(key)) for key in (
            'prompt_tokens', 'tokens', 'finish', 'content_sha256', 'wall_s', 'error'))), flush=True)
    return dict(prompts=rows, lengths=list(lengths), corpus=corpus, max_tokens=LEVERN_EQUAL_TOKENS)


def levern_equal():
    # Opt-in (tp4/lever-n, G-N1 part i). One user alone: prompts of EXACTLY 2047, 2048, 2049, 4095, 4096, 4097, 6143, 6145 and 32,785 tokens (the
    # boundaries of the model's 2,048-token chunk and the final-step coalescing) through /v1/completions as token ids, 256 tokens out. Every row's
    # answer hashes are compared between the arms by levern_compare.py (the interleaved arm must equal the control), and on the audit profiles
    # each prefill logs its digests (slot, logits, KV): the audited pair's digests are compared the same way. A prompt under 4,096 tokens
    # is not split (a control inside the interleaved arm).
    return levern_equal_run('levern_equal', LEVERN_EQUAL_LENGTHS)


def levern_equal_long():
    # Opt-in (tp4/lever-n). The same for the two long boundary prompts: 131,077 and 253,920 tokens (123 chunks and a 2,016-token tail).
    return levern_equal_run('levern_equal_long', LEVERN_EQUAL_LONG_LENGTHS)


def start_decoders(budgets, prompt_tokens=STALL_SHORT_TOKENS, warm_chunks=STALL_WARM_CHUNKS):
    """Decoding seats (4k real-code prompts, ignore_eos, one budget each) started together; returns (out, events, threads, corpus) where
    events[i] is set once seat i has streamed warm_chunks chunks (or ended)."""
    prompts, corpus = code_prompts((prompt_tokens,) * len(budgets))
    out = [None] * len(budgets)
    events = [threading.Event() for _ in budgets]

    def run(index):
        def seen(count):
            if count >= warm_chunks:
                events[index].set()

        try:
            out[index] = stream([{'role': 'user', 'content': prompts[index]}], budgets[index], timeout=3600, keep_stamps=True,
                                on_token=seen, ignore_eos=True)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])
        finally:
            events[index].set()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(budgets))]
    [thread.start() for thread in threads]
    return out, events, threads, corpus


def drop_stamps(users):
    for user in users:
        if isinstance(user, dict):
            user.pop('delta_stamps', None)
    return users


def levern_equal_busy():
    # Opt-in (tp4/lever-n, G-N1 part ii). Seven seats decode 4k prompts (ignore_eos) while the boundary prompts of LEVERN_BUSY_LENGTHS arrive one
    # at a time, so decode rounds run BETWEEN the chunks of each split prefill (pause, decode, resume): the interleaved arm's texts must equal the
    # control's and the solo run's. The seven decoders' own answers are compared between the arms too.
    out, events, threads, corpus = start_decoders([LEVERN_BUSY_BUDGET] * LEVERN_BUSY_SEATS)
    for event in events:
        event.wait(3600)
    prompts, exact_corpus = exact_token_prompts(LEVERN_BUSY_LENGTHS)
    rows = {}
    for length, ids in zip(LEVERN_BUSY_LENGTHS, prompts):
        row = complete_ids(ids, LEVERN_EQUAL_TOKENS)
        rows[str(length)] = row
        print('levern_equal_busy prompt', length, json.dumps(dict((key, row.get(key)) for key in (
            'prompt_tokens', 'tokens', 'finish', 'content_sha256', 'wall_s', 'error'))), flush=True)
    [thread.join() for thread in threads]
    return dict(users=drop_stamps(out), prompts=rows, lengths=list(LEVERN_BUSY_LENGTHS), corpus=corpus, max_tokens=LEVERN_EQUAL_TOKENS)


def run_cold(index, out, prompt, budget, **extra):
    try:
        out[index] = stream([{'role': 'user', 'content': prompt}], budget, timeout=3600, keep_stamps=True, **extra)
    except Exception as error:
        out[index] = dict(error=repr(error)[:300])


def levern_decoder_finishes():
    # Opt-in (tp4/lever-n, G-N2 a). One of seven decoders finishes MID-prefill of a cold arrival (budget 150): its plugin slot frees and the
    # partial's row can move between two chunks (v121), and the batch condenses. Every stream must complete and the arrival must be exact.
    budgets = [150] + [LEVERN_BUSY_BUDGET] * (LEVERN_BUSY_SEATS - 1)
    out, events, threads, corpus = start_decoders(budgets)
    for event in events:
        event.wait(3600)
    cold, _, fit = fitted_code_prompts((LEVERN_COLD_TOKENS,))
    newcomer = [None]
    thread = threading.Thread(target=run_cold, args=(0, newcomer, cold[0], 300))
    thread.start()
    [item.join() for item in threads + [thread]]
    return dict(users=drop_stamps(out + newcomer), corpus=corpus, fit=fit, budgets=budgets)


def levern_all_decoders_finish():
    # Opt-in (tp4/lever-n, G-N2 b). Every decoder finishes mid-prefill of the cold arrival (budgets 120, 150 and 180): the hook closes between two
    # chunks and the final step builds a fresh one. The arrival must complete exact.
    budgets = [120, 150, 180]
    out, events, threads, corpus = start_decoders(budgets)
    for event in events:
        event.wait(3600)
    cold, _, fit = fitted_code_prompts((LEVERN_COLD_TOKENS,))
    newcomer = [None]
    thread = threading.Thread(target=run_cold, args=(0, newcomer, cold[0], 300))
    thread.start()
    [item.join() for item in threads + [thread]]
    return dict(users=drop_stamps(out + newcomer), corpus=corpus, fit=fit, budgets=budgets)


def abort_stream(messages, max_tokens, after_s, timeout=3600):
    """A streamed request the CLIENT drops `after_s` seconds in, by shutting the socket from another thread (the server sees a disconnect while
    the request is still prefilling, before its first token: urllib would still be waiting for the response headers)."""
    import http.client
    import socket
    from urllib.parse import urlparse

    target = urlparse(BASE)
    body = json.dumps({'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'str' + 'eam': True}).encode()
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)
    started = time.time()

    fired = threading.Event()

    def drop():
        fired.set()
        try:
            connection.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass

    timer = threading.Timer(after_s, drop)
    timer.start()
    first_byte = None
    try:
        connection.request('POST', '/v1/chat/completions', body, {'content-type': 'application/json'})
        response = connection.getresponse()
        first_byte = round(time.time() - started, 2)
        tail = b''
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            tail = (tail + chunk)[-64:]
        # the stream ended by itself only if the server sent its [DONE]; an end after the drop (EOF on the shut socket) is the drop
        outcome = 'completed' if b'[DONE]' in tail and not fired.is_set() else 'dropped'
    except Exception as error:
        outcome = 'dropped' if fired.is_set() else 'failed: %r' % (error,)
    finally:
        timer.cancel()
        drop()
    return dict(outcome=outcome, after_s=after_s, elapsed_s=round(time.time() - started, 2), first_byte_s=first_byte)


def levern_cancel_mid_prefill():
    # Opt-in (tp4/lever-n, G-N2 c). Three seats decode; a cold arrival of about 120,000 tokens is DROPPED by its client 20 s into its prefill (mid-chunk
    # sequence: the abort frees the gate, the scratch owner and the KV reservation), then a 8k request arrives and must complete as if nothing had
    # happened, and every decoder must complete.
    budgets = [LEVERN_BUSY_BUDGET] * 3
    out, events, threads, corpus = start_decoders(budgets)
    for event in events:
        event.wait(3600)
    cold, _, fit = fitted_code_prompts((STALL_COLD_128K_TOKENS,))
    import os
    dropped = abort_stream([{'role': 'user', 'content': cold[0]}], 200, float(os.environ.get('SMOKE_LEVERN_CANCEL_AFTER_S', '20')))
    print('levern_cancel_mid_prefill dropped', json.dumps(dropped), flush=True)
    followup_prompt, _ = code_prompts((8192,))
    follow = [None]
    run_cold(0, follow, followup_prompt[0], 300)
    [item.join() for item in threads]
    return dict(users=drop_stamps(out + follow), dropped=dropped, corpus=corpus, fit=fit, budgets=budgets)


def levern_arrival_during_prefill():
    # Opt-in (tp4/lever-n, G-N2 d). Three seats decode; a cold arrival of about 64,000 tokens starts its (interleaved) prefill and, 6 s in, a 5,000-token
    # prompt arrives: it queues behind the in-flight prefill (v1 runs one prefill at a time) and is admitted when the first completes. Both
    # complete exact; the second's time to first token is recorded.
    budgets = [LEVERN_BUSY_BUDGET] * 3
    out, events, threads, corpus = start_decoders(budgets)
    for event in events:
        event.wait(3600)
    cold, _, fit = fitted_code_prompts((LEVERN_COLD_TOKENS,))
    short, _ = code_prompts((5000,))
    pair = [None, None]
    first = threading.Thread(target=run_cold, args=(0, pair, cold[0], 300))
    first.start()
    import os
    time.sleep(float(os.environ.get('SMOKE_LEVERN_ARRIVAL_AFTER_S', '6')))
    second = threading.Thread(target=run_cold, args=(1, pair, short[0], 300))
    second.start()
    [item.join() for item in threads + [first, second]]
    return dict(users=drop_stamps(out + pair), corpus=corpus, fit=fit, budgets=budgets)


def levern_seed_stops():
    # Opt-in (tp4/lever-n, G-N2 e and f). Two seats decode while a chunked prompt of about 36,000 tokens is asked for ONE token (max_tokens=1: the request ends at
    # its seed after 17 chunks, no engine is built for it), then an ordinary 4k request completes: nothing was left held by the terminal seed.
    budgets = [LEVERN_BUSY_BUDGET] * 2
    out, events, threads, corpus = start_decoders(budgets)
    for event in events:
        event.wait(3600)
    long_prompt, _, fit = fitted_code_prompts((36000,))
    one = [None]
    run_cold(0, one, long_prompt[0], 1)
    after, _ = code_prompts((4096,))
    follow = [None]
    run_cold(0, follow, after[0], 300)
    [item.join() for item in threads]
    return dict(users=drop_stamps(out + one + follow), corpus=corpus, fit=fit, budgets=budgets)


DRAIN_BUDGETS = (200, 400, 600, 800, 1000, 1200, 1400, 1600)
SPLIT_BUDGETS = (1600, 1600, 1600, 1600, 1200, 400)
SPLIT_CHUNKS = 60           # user 4's streamed chunks (about one per engine step, so rounds, NOT tokens: a lone user drafts 4 rows and
                            # emits 3.75+ tokens a chunk) before the sixth user arrives: at least 50 rounds with its block narrowed


def concurrent8_drain():
    # Opt-in (tp4/seats8). Eight 4k real-code prompts, budgets 200, 400 ... 1,600 with ignore_eos, started together: the users end
    # one by one at their own budgets, so the live count goes 8 to 1 through every split of the two blocks the placement allows
    # (a block narrowing to one live user while the other still runs packed, then the last block emptying).
    prompts, corpus = code_prompts((4096,) * 8)
    return dict(run_users('concurrent8_drain', prompts, max_tokens=list(DRAIN_BUDGETS), ignore_eos=True), corpus=corpus,
                budgets=list(DRAIN_BUDGETS))


def concurrent5_split():
    # Opt-in (tp4/seats8). The split-block shape: users 0-3 (block A) start together; user 4 arrives alone in block B once all four
    # stream and streams at least SPLIT_CHUNKS chunks (about 60 rounds with its block narrowed to one live user, the other block packed);
    # then a sixth arrives into block B (a pair, narrow to padded). Every user ignores EOS, budgets 1,600 x 4, 1,200, 400. A user that
    # dies releases the next arrival (never a deadlock): user 4's death records its chunk count as None.
    prompts, corpus = code_prompts((4096,) * 6)
    out = [None] * 6
    streaming = [threading.Event() for _ in range(5)]      # users 0-4: set at the first token or at the end (a dead user too)
    split = dict(chunks=None)
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
                if split['chunks'] is None and isinstance(out[4], dict) and 'error' not in out[4]:
                    split['chunks'] = out[4].get('chunks_streamed')    # it ended before SPLIT_CHUNKS: the count it reached
                split_ready.set()

    def first_token(index):
        def seen(count):
            streaming[index].set()
        return seen

    def fifth_token(count):
        streaming[4].set()
        if count == SPLIT_CHUNKS and split['chunks'] is None:
            split['chunks'] = count
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
    print('concurrent5_split user4_chunks_at_sixth_arrival', split['chunks'], flush=True)
    for index, user in enumerate(out):
        print('concurrent5_split user', index, json.dumps(dict((key, user.get(key)) for key in (
            'prompt_tokens', 'tokens', 'ttft', 'decode_tok_s', 'finish', 'error'))), flush=True)
    return dict(users=out, corpus=corpus, budgets=list(SPLIT_BUDGETS), user4_chunks_at_sixth_arrival=split['chunks'])


if ONLY and 'concurrent8_code_equal' in ONLY:
    record('concurrent8_code_equal', concurrent8_code_equal)
if ONLY and 'concurrent8_code_32k' in ONLY:
    record('concurrent8_code_32k', concurrent8_code_32k)
if ONLY and 'concurrent8_code_128k' in ONLY:
    record('concurrent8_code_128k', concurrent8_code_128k)
if ONLY and 'concurrent8_skew' in ONLY:
    record('concurrent8_skew', concurrent8_skew)
if ONLY and 'stall8_cold262k' in ONLY:
    record('stall8_cold262k', stall8_cold262k)
if ONLY and 'stall8_cold128k' in ONLY:
    record('stall8_cold128k', stall8_cold128k)
if ONLY and 'cold2_254k' in ONLY:
    record('cold2_254k', cold2_254k)
if ONLY and 'replay_concurrent8' in ONLY:
    record('replay_concurrent8', replay_concurrent8)
for _levern_name in ('levern_equal', 'levern_equal_long', 'levern_equal_busy', 'levern_decoder_finishes', 'levern_all_decoders_finish',
                     'levern_cancel_mid_prefill', 'levern_arrival_during_prefill', 'levern_seed_stops'):
    if ONLY and _levern_name in ONLY:
        record(_levern_name, globals()[_levern_name])
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
