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


def weight_streams(resources, operations, model, owned_streams, policy, directory):
    """The serial MLP block stream over every layer's w_gate_up, entered into `resources`,
    as the (block_stream recipe, weight pool) pair attach_combined_runtime takes - or, when
    serving_runtime.register_reader_reason admits the register-epilogue reader instead,
    (None, None) with no stream built at all.

    That reason exists at the 64-row M3 block only (four scheduler requests,
    QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1), behind either of two default-off
    flags: QWEN_FAST_SKIP_BLOCK_STREAM=1, since the stream is never read there (v83's exit
    audit: fewer than 64 calls, since neither the 64-row block nor the trimmed 1/2/4-row
    captures are 16-row), which frees 3,227,516,928 B per chip (docs/mlp-block-stream.md:73);
    and QWEN_FAST_SINGLE_GATEUP=1, where the model builds no w_gate_up to stream - refused
    (ValueError) at any other shape, and refused if any layer still holds w_gate_up (the
    model graft is not applied), so its marker states what the model holds, not the flag.
    QWEN_FAST_SKIP_BLOCK_STREAM=1 at another shape builds the stream and says so, so such a
    run cannot pass for an A1 measurement. With both unset this enters owned_streams with
    exactly the arguments it always had."""
    from serving_runtime import SINGLE_GATEUP_SHAPE, SKIP_BLOCK_STREAM_FLAG, m3_shape, register_reader_reason

    skipped = register_reader_reason(policy)
    if skipped is not None:
        from dflash_device import pindiag

        if skipped[0] == SINGLE_GATEUP_SHAPE:
            layers = list(model.layers)
            present = sum(layer.feed_forward.weights.w_gate_up is not None for layer in layers)
            if present:
                raise ValueError('QWEN_FAST_SINGLE_GATEUP=1 but w_gate_up is present on %d of %d layers: the '
                                 'single gate/up model graft is not applied' % (present, len(layers)))
            pindiag('[PINDIAG] block stream skipped for {}: w_gate_up present on {} of {} layers',
                    skipped[0], present, len(layers))
        else:
            pindiag('[PINDIAG] block stream skipped for {}: {}', *skipped)
        return None, None
    if os.environ.get(SKIP_BLOCK_STREAM_FLAG) == '1':
        from dflash_device import pindiag

        pindiag('[PINDIAG] block stream NOT skipped: QWEN_FAST_SKIP_BLOCK_STREAM=1 but {}', m3_shape(policy)[1])
    streams, pool = resources.enter_context(owned_streams(operations, model.mesh_device,
        [layer.feed_forward.weights.w_gate_up for layer in model.layers]))
    return dict(evidence=directory / 'block-stream-evidence', streams=streams), pool


def model_walk(runner, model):
    """P0's categories for the memory ledger, most specific first (a buffer counts under
    the first category that reaches it, and the layers hold references to the model's shared
    TT_CCL and to the KV caches): the MLP weights by name, the model's own TT_CCL and
    SamplingGenerator (model.py builds a pair of its own beside serving_runtime's), the
    embedding, the LM head, the native GDN states and prefill scratch, the KV caches, then the
    rest of each layer (GDN / attention tensors, norms), then anything else on the model."""
    layers = list(getattr(model, 'layers', ()))

    def weights(name):
        return [getattr(getattr(getattr(layer, 'feed_forward', None), 'weights', None), name, None) for layer in layers]

    fields = vars(model) if hasattr(model, '__dict__') else {}
    named = ('tt_ccl', 'sampling', 'embd', 'lm_head_weight', '_deltanet_external_states', '_gdn_prefill_scratch',
             '_dn_zero_recurrent', '_dn_zero_conv', '_paged_kv_caches', 'layers')
    return {'target.mlp.w1_w3': weights('w1') + weights('w3'), 'target.mlp.w2': weights('w2'),
            'target.mlp.w_gate_up': weights('w_gate_up'),
            'model.tt_ccl_and_sampler': [fields.get('tt_ccl'), fields.get('sampling')],
            'model.embedding': fields.get('embd'), 'model.lm_head': fields.get('lm_head_weight'),
            'model.gdn_states_and_scratch': [fields.get(name) for name in (
                '_deltanet_external_states', '_gdn_prefill_scratch', '_dn_zero_recurrent', '_dn_zero_conv')],
            'kv_caches': [getattr(runner, 'kv_caches', None), fields.get('_paged_kv_caches')],
            'target.layers.other': layers,
            'model.other': {name: value for name, value in fields.items() if name not in named}}


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
        # QWEN_FAST_MEMORY_LEDGER=1 only: P0 (model, KV and plugin state, nothing of the
        # fast path yet) and P1 (after the weight streams). Unset, no ledger is built and
        # every record() returns at its first line.
        import memory_ledger

        if memory_ledger.enabled():
            memory_ledger.begin(ttnn, model.layers[0].feed_forward.weights.w2)
            resources.callback(memory_ledger.end)
            memory_ledger.record('P0', **model_walk(worker.model_runner, model))
        block_stream, pool = weight_streams(resources, ttnn, model, owned_streams,
                                            lambda: validate_fast_config(worker.vllm_config), paths['directory'])
        memory_ledger.record('P1', block_stream=block_stream, weight_pool=pool)
        attached = resources.enter_context(attach_combined_runtime(worker, ttnn,
            directory=paths['directory'], runtime_root=paths['runtime_root'], fixtures=fixtures,
            native_attention_evidence=paths['directory'] / 'dflash-t16-native-evidence',
            block_stream=block_stream,
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
