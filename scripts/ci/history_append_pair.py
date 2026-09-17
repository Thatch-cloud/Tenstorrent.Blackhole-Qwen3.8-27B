"""Matched full-request control and incremental-publication candidate timing."""

from contextlib import contextmanager
from functools import wraps
import inspect
import math
from unittest.mock import patch

from dspark_score_pair import summarize


def history_dominated(request):
    blocks = request.get('blocks', [])
    stages = [record for record in request.get('publication_diagnostics', {}).get('records', [])
        if record.get('stage') == 'prepare_history']
    if (not blocks or len(stages) != len(blocks)
            or [record.get('position') for record in stages] != [block.get('position') for block in blocks]):
        return False
    values = [record.get('host_ms') for record in stages] + [block.get(name) for block in blocks
        for name in ('draft_ms', 'verify_readback_ms')]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        return False
    history_ms = sum(record['host_ms'] for record in stages)
    return (0.5 * request['decode_ms'] < history_ms < request['decode_ms']
        and sum(block['draft_ms'] for block in blocks) / len(blocks) <= 120
        and sum(block['verify_readback_ms'] for block in blocks) / len(blocks) <= 100)


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
        if label == 'control':
            summary.update(degraded=summary['committed_tg'] < 25,
                history_dominated=history_dominated(result))
        if records and any(result[name] != records[0]['request'][name] for name in
                ('prompt_tokens', 'emitted', 'eos_ids', 'max_new_tokens', 'committed_decode_tokens')):
            raise ValueError('Identical committed output and request identity required')
        records.append(dict(summary=summary, request=result))
        emit(dict(event='history_pair_request_complete', **summary))
        if label == 'control' and summary['degraded'] and not summary['history_dominated']:
            raise RuntimeError('Control below 25 committed TG; refusing a degraded comparison')
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
    if len(records) != 2:
        raise ValueError('Both complete matched requests required')
