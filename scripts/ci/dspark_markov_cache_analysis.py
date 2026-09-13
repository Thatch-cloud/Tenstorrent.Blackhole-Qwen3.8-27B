"""Host trace analysis of exact Markov bias memoization; no device speed claim."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path


VOCABULARY = 248320


def replay(tokens, slots):
    if type(slots) is not int or slots < 0:
        raise ValueError('Nonnegative integral cache capacity required')
    cache = OrderedDict()
    hits = misses = 0
    for token in tokens:
        if type(token) is not int or not 0 <= token < VOCABULARY:
            raise ValueError('Valid predecessor token required')
        if token in cache:
            hits += 1
            cache.move_to_end(token)
        else:
            misses += 1
            if slots:
                cache[token] = None
                if len(cache) > slots:
                    cache.popitem(last=False)
    return dict(slots=slots, hits=hits, misses=misses, lookups=hits + misses,
        hit_rate=hits / (hits + misses) if hits + misses else 0.,
        fp32_payload_bytes_per_card=slots * VOCABULARY * 4)


def predecessors(request):
    if (any(request.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
            or request.get('instrumented_timing') is not False):
        raise ValueError('Exact uninstrumented request required')
    tokens = []
    for block in request.get('blocks', []):
        inputs = block.get('input_tokens', [])
        if block.get('rows') != 16 or len(inputs) != 16 or block.get('source') != 'dspark':
            raise ValueError('Complete T16 DSpark input blocks required; do not infer missing proposals')
        tokens.extend(inputs[:-1])
    if not tokens or len(tokens) != request.get('proposed'):
        raise ValueError('All fifteen Markov predecessors per proposal invocation required')
    return tokens


def analyze(report):
    if report.get('passed') is not True or report.get('closed_cleanly') is not True:
        raise ValueError('Completed passing hardware report required')
    records = []
    for ordinal, request in enumerate(report.get('request_checks', [])):
        if request.get('instrumented_timing') is not False:
            continue
        tokens = predecessors(request)
        records.append(dict(ordinal=ordinal, arm=request.get('arm'),
            context=len(request['prompt_tokens']), committed=request['committed_decode_tokens'],
            unique_predecessors=len(set(tokens)),
            cold_request_lru=[replay(tokens, slots) for slots in (0, 16, 32, 64, 128)]))
    if not records:
        raise ValueError('Timed request records required')
    return dict(scope=__doc__, future_tokens_used=False, device_implemented=False,
        performance_qualified=False, numerical_qualified=False, requests=records,
        caveats=['Cache resets at every request; no prompt prewarming or cross-request reuse.',
                 'Payload excludes tags, alignment, scratch space and allocator overhead.',
                 'A cache hit must skip the matmul on device; copying a row while still executing it saves no work.',
                 'Cached rows must be produced by the exact original native dot product with unchanged weights.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError('Fresh analysis output required')
    payload = options.report.read_bytes()
    result = analyze(json.loads(payload))
    result['report_sha256'] = hashlib.sha256(payload).hexdigest()
    options.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
