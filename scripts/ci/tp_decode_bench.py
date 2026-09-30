"""Decode and prefill benchmark of one serving container over its OpenAI API (stdlib only; runs on the rig host beside
c2_serving_smoke.py, Python 3.7 syntax), for putting a TP4 profile's numbers next to the pair's.

    python3 tp_decode_bench.py http://127.0.0.1:8010 Qwen/Qwen3.8-27B out.json [--label tp4] [--root .]
        [--shapes 1x4096,1x32768,1x65536,1x130000,4x4096,4x32768,4x65536,8x4096,8x32768,8x65536,4x130000]
        [--max-tokens 256]

A shape is STREAMSxPROMPT: that many simultaneous streams, each a distinct real-text prompt of about PROMPT tokens
(this repository's own documents and sources, cut by characters at CHARS_PER_TOKEN and reported at the server's own
prompt_tokens, so the tokenizer is never loaded here), each decoding max_tokens tokens with ignore_eos so every stream
runs the whole length. Per stream: time to first token (the prefill), tokens, and the decode rate after the first
token. Per shape: the median per-user rate, and the STEADY rate - tokens per second per user over the window in which
every stream was decoding (from the last first-token to the first last-token), the number the pair's multi-user
figures are (a stagger of long prefills would otherwise lower a plain average). One shape at a time, in the order given,
against a server that is already up: a shape the server refuses (too long for its context, more streams than seats) is
recorded as an error and the run continues.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

CHARS_PER_TOKEN = 3.6
MAX_TOKENS = 256
DEFAULT_SHAPES = ('1x4096', '1x32768', '1x65536', '1x130000', '4x4096', '4x32768', '4x65536', '8x4096', '8x32768',
                  '8x65536', '4x130000')
SOURCES = ('docs/bringup-2026-08-24.md', 'docs/gotchas.md', 'docs/decode-payload-bound.md',
           'scripts/ci/c2_serving_gate.py', 'scripts/ci/c2_prefix_gate.py', 'scripts/ci/serving_c2_contract.py',
           'scripts/ci/lever_n_m3native_gate.py', 'scripts/ci/tp4_mesh.py')


def parse_shape(text):
    streams, _, prompt = text.partition('x')
    if not (streams.isdigit() and prompt.isdigit() and int(streams) > 0 and int(prompt) > 0):
        raise ValueError('a shape is STREAMSxPROMPT, got %r' % text)
    return int(streams), int(prompt)


def read_corpus(root):
    """One long string of the checkout's own real text (sources in order, each behind a header), never empty."""
    pieces = []
    for path in SOURCES:
        full = os.path.join(root, path)
        if os.path.isfile(full):
            with open(full, encoding='utf-8', errors='replace') as handle:
                pieces.append('### %s\n%s\n' % (path, handle.read()))
    if not pieces:
        raise SystemExit('no source text found under %s' % root)
    return ''.join(pieces)


def prompt_for(corpus, stream, tokens):
    """Stream `stream`'s prompt: about `tokens` tokens of the corpus, from an offset of its own (distinct text, so no
    two streams share a prefix), wrapped to reach the length, behind a one-line instruction."""
    characters = int(tokens * CHARS_PER_TOKEN)
    start = (stream * 7919) % len(corpus)
    text = corpus[start:] + corpus
    while len(text) < characters:
        text += corpus
    return 'Read the following and then write a long, detailed commentary on it.\n\n%s' % text[:characters]


def post_stream(base, model, message, max_tokens, timeout=3600):
    """(first_token_time, token_times, usage, finish) of one streaming completion, times as time.time()."""
    body = dict(model=model, messages=[{'role': 'user', 'content': message}], max_tokens=max_tokens, stream=True,
                temperature=0, ignore_eos=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False})
    request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    times, usage, finish = [], None, None
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
                if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                    times.append(time.time())
                finish = choice.get('finish_reason') or finish
    return times, usage, finish


def summarise_stream(started, times, usage, finish):
    """One stream's record from its chunk times: ttft, tokens (the server's count), the decode rate after the first."""
    tokens = (usage or {}).get('completion_tokens') or len(times)
    record = dict(prompt_tokens=(usage or {}).get('prompt_tokens'), tokens=tokens, finish=finish,
                  ttft_s=round(times[0] - started, 3) if times else None, decode_tok_s=None)
    if len(times) > 1 and times[-1] > times[0]:
        # chunks carry one or more tokens; the server's total, spread over the chunks, is the token count per chunk
        per_chunk = tokens / float(len(times))
        record['decode_tok_s'] = round((tokens - per_chunk) / (times[-1] - times[0]), 3)
    return record


def steady(all_times, all_tokens):
    """Tokens per second per user while every stream was decoding: from the last stream's first token to the first
    stream's last token, each stream's chunks in that window scaled to its tokens. None when the windows never overlap."""
    windows = [(times[0], times[-1]) for times in all_times if len(times) > 1]
    if len(windows) != len(all_times) or not windows:
        return None
    begin, end = max(window[0] for window in windows), min(window[1] for window in windows)
    if end <= begin:
        return None
    rates = []
    for times, tokens in zip(all_times, all_tokens):
        per_chunk = tokens / float(len(times))
        inside = [moment for moment in times if begin <= moment <= end]
        if len(inside) < 2:
            return None
        rates.append((len(inside) - 1) * per_chunk / (inside[-1] - inside[0]))
    return dict(window_s=round(end - begin, 3), per_user_tok_s=round(statistics.median(rates), 3),
                aggregate_tok_s=round(sum(rates), 3), min_user_tok_s=round(min(rates), 3))


def run_shape(base, model, corpus, streams, tokens, max_tokens, stream_fn=post_stream, clock=time.time):
    """One shape: `streams` simultaneous requests; the record of each, and the shape's summary."""
    records = [None] * streams
    times_of = [[] for _ in range(streams)]
    errors = [None] * streams
    started = [0.0] * streams

    def one(index):
        started[index] = clock()
        try:
            times, usage, finish = stream_fn(base, model, prompt_for(corpus, index, tokens), max_tokens)
            times_of[index] = times
            records[index] = summarise_stream(started[index], times, usage, finish)
        except urllib.error.HTTPError as error:
            errors[index] = 'HTTP %s: %s' % (error.code, error.read().decode(errors='replace')[:200])
        except Exception as error:
            errors[index] = '%s: %s' % (type(error).__name__, str(error)[:200])

    threads = [threading.Thread(target=one, args=(index,)) for index in range(streams)]
    wall = clock()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    shape = dict(name='%dx%d' % (streams, tokens), streams=streams, prompt_tokens_target=tokens,
                 wall_s=round(clock() - wall, 2), streams_detail=[record or dict(error=errors[index])
                                                                  for index, record in enumerate(records)])
    failed = [error for error in errors if error]
    if failed:
        shape['error'] = '; '.join(sorted(set(failed)))
        return shape
    rates = [record['decode_tok_s'] for record in records if record['decode_tok_s']]
    shape.update(median_decode_tok_s=round(statistics.median(rates), 3) if rates else None,
                 max_ttft_s=max(record['ttft_s'] for record in records if record['ttft_s'] is not None),
                 steady=steady(times_of, [record['tokens'] for record in records]))
    return shape


def bench(base, model, root, label, shapes, max_tokens=MAX_TOKENS, stream_fn=post_stream, log=print):
    corpus = read_corpus(root)
    results = []
    for streams, tokens in shapes:
        shape = run_shape(base, model, corpus, streams, tokens, max_tokens, stream_fn)
        results.append(shape)
        steady_rate = (shape.get('steady') or {}).get('per_user_tok_s')
        log('BENCH %s: %s' % (shape['name'], ('ERROR ' + shape['error']) if shape.get('error') else
                              'median %s tok/s/user, steady %s, max ttft %s s, wall %s s' % (
                                  shape.get('median_decode_tok_s'), steady_rate, shape.get('max_ttft_s'),
                                  shape['wall_s'])))
    return dict(label=label, model=model, max_tokens=max_tokens, chars_per_token=CHARS_PER_TOKEN, shapes=results)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base')
    parser.add_argument('model')
    parser.add_argument('output')
    parser.add_argument('--label', default='run')
    parser.add_argument('--root', default='.')
    parser.add_argument('--shapes', default=','.join(DEFAULT_SHAPES))
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    return parser


def main(argv=None, stream_fn=post_stream, log=print):
    options = build_parser().parse_args(argv)
    shapes = [parse_shape(part) for part in options.shapes.split(',') if part]
    result = bench(options.base, options.model, options.root, options.label, shapes, options.max_tokens, stream_fn, log)
    with open(options.output, 'w') as handle:
        json.dump(result, handle, indent=1)
    return 0 if all(not shape.get('error') for shape in result['shapes']) else 1


if __name__ == '__main__':
    sys.exit(main())
