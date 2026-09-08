"""Fixed non-repeated coding request; throughput workload, not a coding-quality certification."""

from collections.abc import Mapping


TASK = '''Implement merge_intervals(intervals) in Python. Input is a list of pairs
of integer endpoints with start <= end. Return a new list of pairs sorted by start,
merging overlapping or touching closed intervals. Do not mutate the input.
Handle an empty list, negative endpoints, duplicate intervals and nested intervals.
Examples:
[] -> []
[(5, 7), (1, 3), (3, 6)] -> [(1, 7)]
[(-4, -1), (-3, -2), (2, 2)] -> [(-4, -1), (2, 2)]
Return only the complete function in one Python code block, without explanation.'''


def make_prompt(tokenizer):
    encoded = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a careful coding assistant.'},
        {'role': 'user', 'content': TASK}], tokenize=True, add_generation_prompt=True,
        return_dict=False, enable_thinking=False)
    tokens = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('Expected nonempty flat token IDs')
    return tokens
