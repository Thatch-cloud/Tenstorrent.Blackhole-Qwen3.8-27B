"""Measured block latency budgets, not throughput predictions or qualification."""

import math


def analyze(blocks, target_tg=200):
    if not blocks or type(target_tg) not in (float, int) or not math.isfinite(target_tg) or target_tg <= 0:
        raise ValueError('Nonempty timed blocks and a positive finite target required')
    fields = ('draft_ms', 'verify_readback_ms', 'select_commit_ms', 'cycle_ms')
    totals = dict.fromkeys(fields, 0.0)
    committed = 0
    widths = set()
    for block in blocks:
        rows, tokens = block.get('rows'), block.get('committed')
        if type(rows) is not int or rows < 2 or type(tokens) is not int or not 1 <= tokens <= rows:
            raise ValueError('Speculative blocks with valid committed counts required')
        widths.add(rows)
        for field in fields:
            value = block.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative block timings required: ' + field)
            totals[field] += value
        if block['cycle_ms'] <= 0 or any(block[field] > block['cycle_ms'] for field in fields[:-1]):
            raise ValueError('Positive whole cycle must cover each measured phase')
        committed += tokens
    if len(widths) != 1:
        raise ValueError('Separate verifier widths before computing a budget')
    means = {field: value / len(blocks) for field, value in totals.items()}
    tokens = committed / len(blocks)
    required_ms = 1000 * tokens / target_tg
    return dict(rows=widths.pop(), blocks=len(blocks), committed=committed,
        mean_committed_per_block=tokens, mean_ms=means,
        observed_block_tg=1000 * committed / totals['cycle_ms'], target_tg=target_tg,
        required_cycle_ms=required_ms,
        required_cycle_reduction_percent=100 * (1 - required_ms / means['cycle_ms']),
        required_committed_at_current_latency=target_tg * means['cycle_ms'] / 1000,
        zero_draft_ceiling_tg=(1000 * tokens / (means['cycle_ms'] - means['draft_ms'])
                               if means['cycle_ms'] > means['draft_ms'] else None),
        scope='Block-only counterfactual with unchanged acceptance and all other timings; not full-request TG')
