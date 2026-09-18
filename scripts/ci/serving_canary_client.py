"""Compare loopback HTTP streaming against the retained combined hardware oracle."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REFERENCE_SHA256 = '0be812cceaf6f91686570bb02a9991cc5a58d8aaa95e05d0633388802799a2bc'


def reference(path):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Exact retained combined hardware report required')
    report = json.loads(payload)
    checks = report.get('request_checks', [])
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('ctx_tokens') != 4096 or report.get('streams') != 1 or len(checks) != 6):
        raise ValueError('Complete closed single-stream 4K reference required')
    prompt, emitted = checks[0]['prompt_tokens'], checks[0]['emitted']
    if (len(prompt) != 4096 or not 1 <= len(emitted) <= 256
            or any(type(token) is not int or not 0 <= token < 248320 for token in prompt + emitted)
            or any(entry.get('exact') is not True or entry.get('max_new_tokens') != 256
                or entry.get('prompt_tokens') != prompt or entry.get('emitted') != emitted for entry in checks)):
        raise ValueError('Reference token identity or correctness checks disagree')
    return prompt, emitted


def endpoint(base):
    parsed = urlparse(base)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('Unauthenticated loopback-only disposable canary required')
    return base.rstrip('/') + '/v1/completions'


def consume(lines, expected, started, *, clock=time.perf_counter):
    tokens, events, finish, usage, done = [], [], None, None, False
    for line in lines:
        line = line.decode('utf-8').strip()
        if not line or line.startswith(':'):
            continue
        if not line.startswith('data: '):
            raise ValueError('Unexpected streaming record')
        data = line[6:]
        if data == '[DONE]':
            done = True
            break
        chunk = json.loads(data)
        if chunk.get('error'):
            raise ValueError('Serving returned a streaming error')
        choices = chunk.get('choices', [])
        if len(choices) > 1:
            raise ValueError('Exactly one completion required')
        if choices:
            choice = choices[0]
            ids = choice.get('token_ids') or []
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError('Explicit integer output tokens required')
            if ids:
                if finish is not None:
                    raise ValueError('Output arrived after completion')
                tokens.extend(ids)
                events.append(dict(seconds=clock() - started, tokens=len(ids)))
                if tokens != expected[:len(tokens)]:
                    raise ValueError('HTTP tokens differ from retained native target reference')
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
        if chunk.get('usage') is not None:
            usage = chunk['usage']
    elapsed = clock() - started
    if (not done or finish != 'stop' or tokens != expected or not events
            or usage is None or usage.get('prompt_tokens') != 4096
            or usage.get('completion_tokens') != len(tokens)):
        raise ValueError('Incomplete exact-token stream, EOS finish or usage accounting')
    interval = events[-1]['seconds'] - events[0]['seconds']
    rate = (len(tokens) - events[0]['tokens']) / interval if interval > 0 else None
    return dict(exact=True, context=4096, output_tokens=tokens, usage=usage,
        ttft_seconds=events[0]['seconds'], e2e_seconds=elapsed, token_events=events,
        stream_delivery_tokens_per_second=rate, pp_tokens_per_second=None,
        timing_scope='HTTP delivery after first token event; not device-only TG or isolated PP',
        serving_qualified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError('Fresh canary report destination required')
    prompt, expected = reference(arguments.reference)
    url = endpoint(arguments.base_url)
    report = dict(passed=False, reference_sha256=REFERENCE_SHA256, requests=[],
        serving_qualified=False, performance_qualified=False)
    try:
        for ordinal in range(2):
            body = dict(model='qwen-fast-canary', prompt=prompt, temperature=0, max_tokens=256,
                stream=True, stream_options=dict(include_usage=True), return_token_ids=True,
                add_special_tokens=False)
            started = time.perf_counter()
            request = Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=180) as response:
                result = consume(response, expected, started)
            report['requests'].append(dict(ordinal=ordinal, **result))
            arguments.output.write_text(json.dumps(report, indent=2) + '\n')
        report['passed'] = True
    finally:
        arguments.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
