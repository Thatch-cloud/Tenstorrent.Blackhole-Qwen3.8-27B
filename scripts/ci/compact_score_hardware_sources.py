"""Narrow source transformation for a separately admitted hardware experiment."""

from pathlib import Path


GUARD = "    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):\n"
HARDWARE_GUARD = (
    "    if (any(os.environ.get(name) != '1' for name in ('QWEN_COMPACT_SCORE_HARDWARE',\n"
    "            'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS', 'QWEN_FROZEN_COMBINED_RUNTIME'))\n"
    "            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('QWEN_SIM_ONLY') == '1'):\n"
)


def transform(source, kind):
    if kind not in ('device', 'markov'):
        raise ValueError('Explicit device or Markov adapter required')
    count = 2 if kind == 'device' else 1
    if source.count(GUARD) != count:
        raise ValueError('Exact simulator guard count required')
    changed = source.replace(GUARD, HARDWARE_GUARD)
    messages = (('Compact local winners are simulator-only', 'Allocated compact-score hardware experiment required'),
                ('Compact winner reduction is simulator-only', 'Allocated compact-score hardware experiment required')) \
        if kind == 'device' else (('Compact Markov feedback is simulator-only',
                                  'Allocated compact-score hardware experiment required'),)
    for before, after in messages:
        if changed.count(before) != 1:
            raise ValueError('Exact execution guard message required')
        changed = changed.replace(before, after)
    if kind == 'markov':
        before = 'from compact_score_device import execute_local_winners, reduce_winners'
        if changed.count(before) != 1:
            raise ValueError('Exact compact feedback binding required')
        changed = changed.replace(before, 'from compact_score_hardware_device import execute_local_winners, reduce_winners')
    compile(changed, kind, 'exec')
    return changed


def payloads(directory):
    directory = Path(directory)
    return {f'compact_score_hardware_{kind}.py': transform((directory / name).read_text(), kind)
            for kind, name in (('device', 'compact_score_device.py'), ('markov', 'compact_markov.py'))}
