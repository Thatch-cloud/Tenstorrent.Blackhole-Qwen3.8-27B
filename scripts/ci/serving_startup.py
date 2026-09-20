"""Explicit fast-worker startup; no baseline serving defaults are modified."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

from serving_fast_policy import validate_fast_config


def recipe_paths(config):
    validate_fast_config(config)
    recipe = config.additional_config.get('qwen_fast_runtime')
    required = {'directory', 'runtime_root', 'fixtures', 'target_snapshot'}
    if not isinstance(recipe, dict) or set(recipe) != required:
        raise ValueError('Explicit source, runtime, DFlash fixtures and target snapshot paths required')
    paths = {}
    for name, value in recipe.items():
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f'Existing absolute directory required: {name}')
        paths[name] = Path(value).resolve()
    if Path(os.environ.get('TT_METAL_HOME', '')).resolve() != paths['runtime_root']:
        raise ValueError('Serving recipe must use the loaded TT runtime root')
    return paths


def start(worker):
    if getattr(worker, '_qwen_fast_resources', None) is not None:
        raise ValueError('Fast worker already initialized')
    paths = recipe_paths(worker.vllm_config)
    from sampling_link_policy import DESCRIPTOR, SOURCES
    import hashlib

    if (os.environ.get('TT_MESH_GRAPH_DESC_PATH') != str(paths['runtime_root'] / DESCRIPTOR)
            or any(os.environ.get(name) for name in
                ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'))):
        raise ValueError('Explicit P150 pair hardware descriptor required for serving startup')
    if any(hashlib.sha256((paths['runtime_root'] / name).read_bytes()).hexdigest() != digest
            for name, digest in SOURCES.items()):
        raise ValueError('Four-link serving descriptor or sampling sources changed')
    import ttnn
    from full_dflash_request import load_dflash_fixtures
    from mlp_block_stream_pool import owned_streams
    from serving_runtime import attach_combined_runtime

    model = worker.model_runner.model.model[0]
    config = json.loads((paths['target_snapshot'] / 'generation_config.json').read_text())
    eos_ids = config.get('eos_token_id')
    eos_ids = (eos_ids,) if type(eos_ids) is int else tuple(eos_ids) if isinstance(eos_ids, list) else ()
    if not eos_ids or any(type(token) is not int or not 0 <= token < model.args.vocab_size for token in eos_ids):
        raise ValueError('Target snapshot must provide valid EOS identifiers')
    fixtures = load_dflash_fixtures(paths['fixtures'])
    resources = ExitStack()
    try:
        streams, pool = resources.enter_context(owned_streams(ttnn, model.mesh_device,
            [layer.feed_forward.weights.w_gate_up for layer in model.layers]))
        attached = resources.enter_context(attach_combined_runtime(worker, ttnn,
            directory=paths['directory'], runtime_root=paths['runtime_root'], fixtures=fixtures,
            native_attention_evidence=paths['directory'] / 'dflash-t16-native-evidence',
            block_stream=dict(evidence=paths['directory'] / 'block-stream-evidence', streams=streams),
            kv_publication_evidence=paths['directory'] / 'draft-kv-slide-evidence',
            eos_ids=eos_ids, cancelled=lambda: False))
        worker._qwen_fast_resources = resources
        worker._qwen_fast_attachment = dict(attached, weight_pool=pool)
    except BaseException:
        resources.close()
        raise


def stop(worker):
    resources = getattr(worker, '_qwen_fast_resources', None)
    if resources is not None:
        resources.close()
        worker._qwen_fast_resources = None
        worker._qwen_fast_attachment = None


def warmup(worker):
    if os.environ.get('QWEN_FAST_FAULTHANDLER') == '1':
        # A device hang leaves the worker blocked inside a ttnn call; dumping the
        # Python stack every minute names that call in the server log (run
        # 35482551725 stalled with nothing to say).
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
    from vllm.v1.worker.worker_base import CompilationTimes

    started = time.perf_counter()
    start(worker)
    return CompilationTimes(language_model=time.perf_counter() - started, encoder=0.0)
