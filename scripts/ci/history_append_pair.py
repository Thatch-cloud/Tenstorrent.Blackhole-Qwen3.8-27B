"""Matched full-request control and incremental-publication candidate timing."""

from contextlib import contextmanager
from functools import wraps
import inspect
import math
from unittest.mock import patch

from dspark_score_pair import summarize


@contextmanager
def paired_history(module, candidate_scope, records, emit):
    original = module.measure_dspark_request
    signature = inspect.signature(original)
    ordinal = 0

    @wraps(original)
    def measure(*args, **kwargs):
        nonlocal ordinal
        arguments = signature.bind(*args, **kwargs).arguments
        if (ordinal >= 2 or arguments.get('audit_features') is not False
                or arguments.get('max_new_tokens') != 256 or len(arguments['prompt']) != 65536
                or arguments.get('proposal_trace') is not True):
            raise ValueError('Two complete captured 64K requests required')
        label = ('control', 'incremental_history')[ordinal]
        if ordinal and list(arguments['prompt']) != records[0]['request']['prompt_tokens']:
            raise ValueError('Matched prompt required')
        ordinal += 1
        if label == 'control':
            result = original(*args, **kwargs)
        else:
            with candidate_scope():
                result = original(*args, **kwargs)
        summary = summarize(result, label)
        prefill = summary['prefill_ms']
        if type(prefill) not in (int, float) or not math.isfinite(prefill) or prefill <= 0:
            raise ValueError('Positive measured prefill time required')
        summary.update(pp=1000 * summary['context'] / prefill, streams=1)
        if records and any(result[name] != records[0]['request'][name] for name in
                ('prompt_tokens', 'emitted', 'eos_ids', 'max_new_tokens', 'committed_decode_tokens')):
            raise ValueError('Identical committed output and request identity required')
        records.append(dict(summary=summary, request=result))
        emit(dict(event='history_pair_request_complete', **summary))
        if label == 'control' and summary['committed_tg'] < 25:
            raise RuntimeError('Control below 25 committed TG; refusing a degraded comparison')
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
    if len(records) != 2:
        raise ValueError('Both complete matched requests required')
