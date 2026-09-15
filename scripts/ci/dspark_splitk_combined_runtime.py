"""Audited request-only split-K binding; no serving or timing admission."""

from contextlib import contextmanager
import inspect
import os
from pathlib import Path
from unittest.mock import patch

import dspark_64k_entry
import dspark_native_cached_layer
import dspark_splitk_attention
from dspark_splitk_combined_build import digest, require_selected, validate_combined
from dspark_splitk_hardware_scope import HEADER, kernel_scope


def require_screen():
    require_selected()
    required = ('QWEN_DSPARK_SFPU_REQUEST_SCREEN', 'QWEN_TARGET_T16_64K_REQUEST',
        'QWEN_DSPARK_CENTER_TILE_FILL')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Fresh audited combined request required before split-K timing')


@contextmanager
def runtime_scope(directory, root, build_path):
    require_screen()
    admission = validate_combined(root, directory, build_path)
    original_execute = dspark_splitk_attention.execute_folded
    state = dict(component=admission['component'],
        combined_build_binaries=admission['build']['binaries'],
        attention_calls=0, kernel_restored=False, full_request_qualified=False,
        performance_qualified=False, serving_qualified=False)

    def execute(*args, **kwargs):
        kwargs.update(key_chunk_size=256, max_cores_per_head=8, stripe_keys=False, fp32_dest_acc=True)
        result = original_execute(*args, **kwargs)
        state['attention_calls'] += 1
        return result

    try:
        with kernel_scope(root) as kernel:
            if kernel != admission['component']['kernel']:
                raise ValueError('Combined request must use the exact hardware-qualified decode kernel')
            state['kernel'] = dict(kernel)
            with patch.dict(os.environ, QWEN_SPLITK_FP32_INTERMEDIATES='1'), \
                    patch.object(dspark_splitk_attention, 'execute_folded', execute), \
                    patch.object(dspark_native_cached_layer, 'attend', dspark_splitk_attention.adapter(65536)):
                yield state
                if state['attention_calls'] == 0:
                    raise ValueError('Combined request did not execute split-K attention')
    finally:
        state['kernel_restored'] = digest(Path(root) / HEADER) == admission['component']['kernel']['source_before']
        if not state['kernel_restored']:
            raise ValueError('Combined request decode kernel was not restored')


@contextmanager
def entry_scope(records):
    require_screen()
    original = dspark_64k_entry.runtime_scope
    signature = inspect.signature(original)

    @contextmanager
    def combined(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        with original(*args, **kwargs) as legacy:
            with runtime_scope(bound.arguments['directory'], bound.arguments['factory_root'],
                    bound.arguments['build_path']) as evidence:
                records.append(evidence)
                yield legacy

    with patch.object(dspark_64k_entry, 'runtime_scope', combined):
        yield
