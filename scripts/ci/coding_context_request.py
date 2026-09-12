"""Recorded repository-prefix context for a coding latency pilot, not a quality benchmark."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from coding_request import TASK


CONTEXT_FILES = ('target_features.py', 'feature_projection.py', 'dflash_request_runtime.py', 'verifier_engine.py')
EXTENDED_CONTEXT_FILES = (*CONTEXT_FILES, 'model_batch.py', 'full_request.py')


def make_context_prompt(tokenizer, *, context_tokens=4096):
    if type(context_tokens) is not int or context_tokens not in (4096, 8192):
        raise ValueError('Only the explicit 4K and 8K context qualification pilots are enabled')
    filenames = CONTEXT_FILES if context_tokens == 4096 else EXTENDED_CONTEXT_FILES
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in filenames}
    context = '\n\n'.join(f'File: {name}\n{payload.decode("utf-8")}' for name, payload in sources.items())

    def encode(characters):
        content = ('Repository context excerpt for background only; it may end partway through a file.\n'
            'Do not reproduce or modify this excerpt. Complete only the coding task after it.\n'
            '<repository_context>\n' + context[:characters] + '\n</repository_context>\n\nCoding task:\n' + TASK)
        encoded = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a careful coding assistant.'},
            {'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
            return_dict=False, enable_thinking=False)
        tokens = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('Expected nonempty flat token IDs')
        return tokens

    if len(encode(len(context))) < context_tokens:
        raise ValueError('Repository excerpt cannot fill the requested context')
    low, high = 0, len(context)
    best = None
    while low <= high:
        characters = (low + high) // 2
        tokens = encode(characters)
        if len(tokens) <= context_tokens:
            if best is None or len(tokens) > len(best[1]):
                best = (characters, tokens)
            low = characters + 1
        else:
            high = characters - 1
    if best is None or not context_tokens - 32 <= len(best[1]) <= context_tokens:
        raise ValueError('Could not construct a bounded context prompt without changing the task or template')
    characters, tokens = best
    return tokens, dict(task='merge_intervals_repo_context_v1', requested_context=context_tokens,
        actual_context=len(tokens), excerpt_characters=characters,
        excerpt_sha256=hashlib.sha256(context[:characters].encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
        sources={name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()},
        scope=__doc__)
