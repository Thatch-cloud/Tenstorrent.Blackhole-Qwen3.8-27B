"""Smoke a C2 serving container over its OpenAI API (stdlib only; runs on the rig host)."""
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


def stream(messages, max_tokens, drop_after=None, **extra):
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, stream=True,
                stream_options={'include_usage': True}, **extra)
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    started, first, pieces, usage, finish, count = time.time(), None, [], None, None, 0
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
                text = (delta.get('content') or '') + (delta.get('reasoning_content') or delta.get('reasoning') or '')
                if text:
                    if first is None:
                        first = time.time()
                    pieces.append(text)
                    count += 1
                finish = choice.get('finish_reason') or finish
            if drop_after is not None and count >= drop_after:
                return dict(dropped_after=count, text=''.join(pieces)[:200])
    ended = time.time()
    tokens = (usage or {}).get('completion_tokens') or 0
    decode = (ended - first) if first else None
    return dict(ttft=round(first - started, 2) if first else None, tokens=tokens,
                prompt_tokens=(usage or {}).get('prompt_tokens'), finish=finish,
                decode_tok_s=round((tokens - 1) / decode, 2) if decode and tokens > 1 else None,
                text=''.join(pieces)[:300])


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


def concurrent():
    prompts = [coding, 'Write a Python LRU cache class with tests. ' * 40,
               'Explain this code:\n' + source[:30000], 'Write a Rust function that parses RFC 3339 dates, with tests.' * 30]
    out = [None] * 4

    def run(index):
        try:
            out[index] = stream([{'role': 'user', 'content': prompts[index]}], 800)
        except Exception as error:
            out[index] = dict(error=repr(error)[:300])

    threads = [threading.Thread(target=run, args=(index,)) for index in range(4)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    return dict(users=out)


record('concurrent4', concurrent)
record('tool_call', lambda: post('/v1/chat/completions', dict(model=MODEL, max_tokens=400, tool_choice='auto',
       messages=[{'role': 'user', 'content': 'What is the weather in Wellington? Use the tool.'}],
       tools=[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
               'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}])))
record('refused_n2', lambda: dict(status=post('/v1/chat/completions', dict(model=MODEL, n=2, max_tokens=8,
       messages=[{'role': 'user', 'content': 'hi'}]))[0]))
record('alive_after_refusal', alive)
record('stream_dropped', lambda: stream([{'role': 'user', 'content': coding}], 1500, drop_after=20))
time.sleep(5)
record('alive_after_drop', alive)


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
