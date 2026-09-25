"""Single-stream decode rate of a live serving container: <base url> <label> [repeats].

Streams one greedy coding request per repeat (300 output tokens, short prompt) and reports time
to first token and the decode rate after it, so a before/after on the same container is paired.
"""
import json
import statistics
import sys
import time
import urllib.request

BASE, LABEL = sys.argv[1], sys.argv[2]
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MODEL = 'Qwen/Qwen3.8-27B'


def once():
    body = dict(model=MODEL, max_tokens=300, temperature=0, stream=True, stream_options={'include_usage': True},
                messages=[{'role': 'user', 'content': 'Write a Python class implementing an LRU cache with get and '
                                                      'put, plus three unit tests. Code only.'}])
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    started, first, tokens = time.time(), None, 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                tokens = chunk['usage'].get('completion_tokens') or tokens
            for choice in chunk.get('choices', ()):
                delta = choice.get('delta') or {}
                if (delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning')) and first is None:
                    first = time.time()
    ended = time.time()
    return first - started, (tokens - 1) / (ended - first)


rates, ttfts = [], []
for _ in range(REPEATS):
    ttft, rate = once()
    ttfts.append(ttft)
    rates.append(rate)
print('DECODE %s tok/s median %.2f (runs %s) ttft median %.2f s' % (
    LABEL, statistics.median(rates), ' '.join('%.2f' % r for r in rates), statistics.median(ttfts)), flush=True)
