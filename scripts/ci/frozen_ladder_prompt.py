"""Pinned synthetic coding latency workload; not held-out quality evaluation."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from frozen_context_geometry import CONTEXTS


CORPUS_SHA256 = '83ba40b9d7045cc674b54c4b03cd56a1535659590463dd099bbbff87d7c97939'


def make_context_prompt(tokenizer, *, context_tokens=4096):
    if type(context_tokens) is not int or context_tokens not in CONTEXTS:
        raise ValueError('Explicit ladder context required')
    payload = Path(__file__).with_name('frozen-ladder-corpus.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != CORPUS_SHA256:
        raise ValueError('Frozen coding corpus changed')
    corpus = json.loads(payload)
    context = '\n\n'.join(f'File: {name}\n{source}' for name, source in corpus['sources'].items())
    repeated = context
    repetitions = 1

    def encode(characters):
        content = ('Repository context excerpt for background only; it may end partway through a file.\n'
            'Do not reproduce or modify this excerpt. Complete only the coding task after it.\n'
            '<repository_context>\n' + repeated[:characters] + '\n</repository_context>\n\nCoding task:\n'
            + corpus['task'])
        encoded = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a careful coding assistant.'},
            {'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
            return_dict=False, enable_thinking=False)
        tokens = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('Expected nonempty flat token IDs')
        return tokens

    while len(encode(len(repeated))) < context_tokens:
        if repetitions >= 16:
            raise ValueError('Pinned corpus cannot fill requested context within repetition bound')
        repeated += '\n\n' + context
        repetitions += 1
    low, high = 0, len(repeated)
    best = None
    while low <= high:
        characters = (low + high) // 2
        tokens = encode(characters)
        if len(tokens) <= context_tokens:
            if best is None or len(tokens) > len(best[1]):
                best = characters, tokens
            low = characters + 1
        else:
            high = characters - 1
    if best is None or not context_tokens - 32 <= len(best[1]) <= context_tokens:
        raise ValueError('Cannot fill context without changing task or template')
    characters, tokens = best
    return tokens, dict(task='merge_intervals_frozen_ladder_v1', requested_context=context_tokens,
        actual_context=len(tokens), corpus_sha256=CORPUS_SHA256,
        corpus_repetitions=(characters + 2 + len(context) - 1) // (len(context) + 2),
        excerpt_characters=characters,
        excerpt_sha256=hashlib.sha256(repeated[:characters].encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in corpus['sources'].items()},
        scope=__doc__)
