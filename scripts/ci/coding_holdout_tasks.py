"""Untuned local coding screen, not a standardized benchmark or training holdout."""

import hashlib
import json


TASKS = (
    dict(name='stable_unique_v1', function='stable_unique', prompt=(
        'Implement stable_unique(values) in Python. Values is a list of integers. '
        'Return a new list containing each distinct integer once, in order of its first '
        'appearance. Do not mutate values. Handle empty input and negative integers. '
        'Return only the complete function in one Python code block, without explanation.'),
        cases=((([],), []), (([3, 1, 3, 2, 1],), [3, 1, 2]),
            (([-2, 0, -2, -1, 0],), [-2, 0, -1]), (([7, 7, 7],), [7]))),
    dict(name='run_length_encode_v1', function='run_length_encode', prompt=(
        'Implement run_length_encode(text) in Python. Return a list of pairs '
        '(character, count) describing consecutive runs in text, preserving order. '
        'Characters are Unicode code points, not bytes. Handle empty input. '
        'Return only the complete function in one Python code block, without explanation.'),
        cases=((('',), []), (('aaabbcca',), [('a', 3), ('b', 2), ('c', 2), ('a', 1)]),
            (('xyx',), [('x', 1), ('y', 1), ('x', 1)]),
            (('\u03bb\u03bb\u03b2',), [('\u03bb', 2), ('\u03b2', 1)]))),
    dict(name='rotate_right_v1', function='rotate_right', prompt=(
        'Implement rotate_right(values, steps) in Python. Return a new list rotating '
        'the input list of integers right by steps positions. Negative steps rotate '
        'left. Handle empty input and steps larger than the list length. Do not mutate '
        'values. Return only the complete function in one Python code block, without explanation.'),
        cases=((([], 9), []), (([1, 2, 3, 4], 1), [4, 1, 2, 3]),
            (([1, 2, 3], -1), [2, 3, 1]), (([1, 2, 3], 7), [3, 1, 2]),
            (([1, 2], 0), [1, 2]))),
)

EXPECTED = {
    'stable_unique_v1': '75e3df4078665be31e9f6bfa17ff82f20cc50443e83caee75e270308932eb949',
    'run_length_encode_v1': '9ac7095852eff13c54ba5b1d2b8e761c1a49d9fe0cadc50b6315dc0ac06a8f03',
    'rotate_right_v1': 'f01135f248b6a515b626873f6486a1fb6882c1a6db0d7f3a47df7cc8278b0893',
}


def task_manifest():
    return {task['name']: hashlib.sha256(json.dumps(task, ensure_ascii=True,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for task in TASKS}


def messages(name):
    if task_manifest() != EXPECTED:
        raise ValueError('Frozen task definitions changed')
    matching = [task for task in TASKS if task['name'] == name]
    if len(matching) != 1:
        raise ValueError('Known frozen coding task required')
    return [{'role': 'system', 'content': 'You are a careful coding assistant.'},
        {'role': 'user', 'content': matching[0]['prompt']}]


def make_context_prompt(tokenizer, name, *, context_tokens=4096):
    from types import SimpleNamespace
    from coding_context_request import make_context_prompt as make_original
    from coding_request import TASK

    selected = messages(name)[-1]['content']

    def encode(original, **options):
        if len(original) != 2 or not original[-1]['content'].endswith(TASK):
            raise ValueError('Original complete task boundary required')
        changed = [dict(value) for value in original]
        changed[-1]['content'] = changed[-1]['content'][:-len(TASK)] + selected
        return tokenizer.apply_chat_template(changed, **options)

    tokens, metadata = make_original(SimpleNamespace(apply_chat_template=encode),
        context_tokens=context_tokens)
    metadata.update(task=name, task_sha256=EXPECTED[name], scope=__doc__,
        functional_quality_qualified=False)
    return tokens, metadata
