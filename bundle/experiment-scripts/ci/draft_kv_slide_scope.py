"""Opt-in combined-request publication override; native projection and transactions unchanged."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

from draft_kv_slide import prepare as transport
from draft_kv_slide_adapter import build_prepare
from draft_kv_slide_gate import qualify


@contextmanager
def scoped_publication(directory, evidence):
    import draft_kv_history
    from mlp_block_stream_runtime import require_hardware

    require_hardware(os.environ)
    if os.environ.get('QWEN_DRAFT_KV_SLIDE_EXPERIMENT') != '1':
        raise ValueError('Explicit fused K/V publication hardware experiment required')
    directory = Path(directory)
    if Path(draft_kv_history.__file__).resolve() != (directory / 'draft_kv_history.py').resolve():
        raise ValueError('Owned request history implementation required')
    admission = qualify(directory, evidence)
    original = draft_kv_history.DraftKVHistory.prepare
    if getattr(original, '_draft_kv_slide', False):
        raise ValueError('Nested publication overrides forbidden')
    audit = dict(enabled=True, restored=False, prepare_calls=0, tensor_copies=0,
        serving_defaults_changed=False, admission=admission)

    def counted_transport(*args, **kwargs):
        operation = transport(*args, **kwargs)
        def execute():
            result = operation()
            audit['tensor_copies'] += 1
            return result
        return execute

    candidate, source = build_prepare((directory / 'draft_kv_history.py').read_text(),
        vars(draft_kv_history), counted_transport)
    audit['adapter'] = source

    def prepare(cache, *args, **kwargs):
        if len(cache.parameters) != 5:
            raise ValueError('All five learned draft layers required')
        before = audit['tensor_copies']
        publication = candidate(cache, *args, **kwargs)
        if audit['tensor_copies'] - before != 10:
            raise AssertionError('Complete five-layer K/V publication coverage required')
        audit['prepare_calls'] += 1
        return publication

    prepare._draft_kv_slide = True
    try:
        with patch.object(draft_kv_history.DraftKVHistory, 'prepare', prepare):
            yield audit
            if draft_kv_history.DraftKVHistory.prepare is not prepare:
                raise ValueError('Publication override changed outside owned scope')
    finally:
        audit['restored'] = draft_kv_history.DraftKVHistory.prepare is original
        if not audit['restored'] or qualify(directory, evidence) != admission:
            raise ValueError('Publication scope or source admission changed')
