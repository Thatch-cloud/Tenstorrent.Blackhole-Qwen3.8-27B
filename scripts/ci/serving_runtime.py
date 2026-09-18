"""Explicit attachment of the admitted combined runtime to a loaded TT worker."""

from contextlib import ExitStack, contextmanager

from serving_cache_owner import ServingCacheOwner
from serving_fast_policy import validate_fast_config
from serving_lifecycle import FastServingLifecycle
from serving_page_binding import VerifierPageBinding
from serving_request_factory import from_prefill
from serving_runner_bridge import FastRunnerBridge


@contextmanager
def attach_combined_runtime(worker, operations, *, directory, runtime_root, fixtures,
                            native_attention_evidence, block_stream, kv_publication_evidence,
                            eos_ids, cancelled):
    import torch
    from dflash_combined_request import combined_runtime
    from dflash_prefill_window import PrefillWindowCapture
    from dflash_request_runtime import TARGET_TAPS
    from gdn_snapshot import ActiveSnapshot
    from models.common.sampling.generator import SamplingGenerator
    from models.tt_transformers.tt.ccl import TT_CCL
    from sampling_link_policy import sampler_links

    validate_fast_config(worker.vllm_config)
    if (native_attention_evidence is None or kv_publication_evidence is None
            or block_stream is None or 'pipeline_evidence' in block_stream):
        raise ValueError('Native T16, direct KV publication and serial weight-stream recipe required')
    runner = worker.model_runner
    model = runner.model.model[0]
    owner = ServingCacheOwner(operations, runner, model)
    layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    if len(layers) != 48:
        raise ValueError('All forty-eight native GDN layers required')
    helpers = [ActiveSnapshot(layer, operations, direct=True) for layer in layers]
    sampler = SamplingGenerator(args=model.args, mesh_device=model.mesh_device,
        tt_ccl=TT_CCL(model.mesh_device))
    sampler.set_trace_bucket(1)
    from serving_gather_experiment import from_environment

    experiment = from_environment(directory, runtime_root)

    def capture_factory(position):
        owner.validate()
        return PrefillWindowCapture(operations, model, position, TARGET_TAPS)

    def bridge_factory(state, capture):
        owner.validate()
        if len(state.block_ids) != 1:
            raise ValueError('One scheduler KV group required')
        blocks = tuple(state.block_ids[0])
        if (not blocks or len(blocks) > 68 or len(set(blocks)) != len(blocks)
                or any(type(block) is not int or not 0 <= block < owner.physical_pages for block in blocks)):
            raise ValueError('Unique physical pages from the admitted cache required')
        pages = torch.full((1, 68), blocks[0], dtype=torch.int32)
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        def create_request():
            return from_prefill(operations, model, sampler, pages, helpers,
                state=state, capture=capture, fixtures=fixtures, eos_ids=eos_ids)

        request = create_request() if experiment is None else experiment.create(create_request)
        try:
            binding = VerifierPageBinding(request.engine, blocks, physical_pages=owner.physical_pages)
            return FastRunnerBridge(runner, request, binding, validate_storage=owner.validate)
        except BaseException:
            request.close(state.req_id)
            raise

    scopes = ExitStack()
    try:
        scopes.enter_context(sampler_links(sampler.tt_sampling, 4))
        audit = scopes.enter_context(combined_runtime(operations, model, directory=directory,
            runtime_root=runtime_root, native_attention_evidence=native_attention_evidence,
            block_stream=block_stream, kv_publication_evidence=kv_publication_evidence))
        lifecycle = FastServingLifecycle(worker, config=worker.vllm_config,
            capture_factory=capture_factory, bridge_factory=bridge_factory, eos_ids=eos_ids,
            cancelled=cancelled)
    except BaseException:
        scopes.close()
        raise
    try:
        yield dict(lifecycle=lifecycle, runtime=audit, cache_owner=owner,
            serving_qualified=False, performance_qualified=False)
    finally:
        lifecycle.close()
        scopes.close()
