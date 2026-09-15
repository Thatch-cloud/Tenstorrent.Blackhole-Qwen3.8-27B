"""Same-process control/candidate requests; refuse further work on a degraded control."""

from contextlib import contextmanager
from functools import wraps
import inspect
import math
from types import SimpleNamespace
from unittest.mock import patch


def summarize(request, label):
    tokens, elapsed = request['committed_decode_tokens'], request['decode_ms']
    if (type(tokens) is not int or tokens <= 0 or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed) or elapsed <= 0
            or any(request.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))):
        raise ValueError('Exact complete positive-timing request required')
    return dict(arm=label, committed_tokens=tokens, decode_ms=elapsed,
        committed_tg=tokens * 1000 / elapsed, prefill_ms=request['prefill_ms'],
        context=request['prompt_tokens'], performance_qualified=False)


@contextmanager
def paired_scope(module, candidate_scope, records, emit):
    original = module.measure_dspark_request
    signature = inspect.signature(original)
    ordinal = 0

    @wraps(original)
    def measure(*args, **kwargs):
        nonlocal ordinal
        arguments = signature.bind(*args, **kwargs).arguments
        if (ordinal >= 2 or arguments.get('audit_features') is not False
                or arguments.get('max_new_tokens') != 256 or len(arguments['prompt']) != 65536):
            raise ValueError('Exactly two complete clean 64K requests required')
        label = ('control', 'score_layout')[ordinal]
        ordinal += 1
        if label == 'control':
            result = original(*args, **kwargs)
        else:
            isolated = SimpleNamespace(measure_dspark_request=original)
            with candidate_scope(isolated):
                result = isolated.measure_dspark_request(*args, **kwargs)
        record = summarize(result, label)
        records.append(dict(summary=record, request=result))
        emit(dict(event='paired_request_complete', **record))
        if label == 'control' and record['committed_tg'] < 25:
            raise RuntimeError('Control below 25 committed TG; stopping before candidate rather than timing a degraded baseline')
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
    if len(records) != 2:
        raise ValueError('Both complete paired requests required')
