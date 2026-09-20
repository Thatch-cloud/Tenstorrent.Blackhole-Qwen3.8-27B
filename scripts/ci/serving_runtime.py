"""Explicit attachment of the admitted combined runtime to a loaded TT worker."""

from contextlib import ExitStack, contextmanager
from functools import partial
import json
import os

from dflash_device import PreparedDraftWeights, pindiag
from serving_buffer_pool import ServingBufferPool
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

    policy = validate_fast_config(worker.vllm_config)
    if (native_attention_evidence is None or kv_publication_evidence is None
            or block_stream is None or 'pipeline_evidence' in block_stream):
        raise ValueError('Native T16, direct KV publication and serial weight-stream recipe required')
    runner = worker.model_runner
    model = runner.model.model[0]
    # One draft history pair per scheduler slot, allocated FIRST: no request exists
    # yet, so no request trace does either. A request's verify trace bakes the
    # addresses of the intermediates its capture frees; the next request's
    # persistent history, allocated afterwards, lands in those holes and every
    # replay of the first trace overwrites it - per chip, since each chip's
    # allocator reuses independently (runs 35477522469, 35479238722; precedent
    # docs/experiment-execution.md, feature_prefix.py). Nothing earlier in this
    # attachment captures a trace, and the plugin's own warmup is replaced by
    # serving_startup.warmup, so this is the earliest point the fast path controls.
    # Registered first so it closes last, after the lifecycle has closed every
    # device that borrows from it.
    scopes = ExitStack()
    try:
        # The helpers allocate nothing; the pool needs them to allocate every request's
        # GDN snapshot sets (initial, carry, one checkpoint set per capture) in the
        # exact shape the engine's own allocate() would have, before any trace.
        layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        if len(layers) != 48:
            raise ValueError('All forty-eight native GDN layers required')
        helpers = [ActiveSnapshot(layer, operations, direct=True) for layer in layers]

        # The page table has to cover the admitted context. It was a fixed 68 pages,
        # which is 68 x 64 = 4352 tokens, while this image's T16 gate demands position
        # 32768 - so the fast path could not decode at ANY context, at one user or two
        # (runs 35472072127, 35473307362). 68 stays the floor because
        # ServingCacheOwner requires at least that many physical pages. The pool's
        # fixture inputs are this wide too: a request's table must match exactly.
        page_width = max(68, -(-int(worker.vllm_config.model_config.max_model_len) // 64))

        def rope(positions):
            # The native rotary construction ModelBatch uses, so the pooled tables are
            # the same kind of tensor the fixture would have built.
            from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
            return rot_mats_decode(model.mesh_device, model.args.rope_head_dim,
                                   model.args.max_seq_len, model.args.rope_theta, positions)

        # Widths with multiplicity for the geometry from_prefill gives every engine:
        # sixteen verifier rows, the output budget, a T16 cap. A request whose plan
        # needs more is refused at admission by the engine, never allocated late.
        from verifier_engine import capture_bucket_rows

        # The packed block's shape, when QWEN_FAST_PACKED_STEP=1 asks for one, follows the
        # scheduler's request count (packed_shapes.serving_shape): two requests take the
        # 32-row M1 block, four the 64-row M3 block, and any other count builds no block,
        # so the sequential step serves the rounds and the stage line says so. Chosen
        # before the pool, which lends the block its per-user replay page tables for
        # exactly that shape; with the switch unset the pool is built as it always was.
        packed_requested = os.environ.get('QWEN_FAST_PACKED_STEP') == '1'
        packed_shape = None
        if packed_requested:
            from packed_shapes import serving_shape

            packed_shape = serving_shape(policy['scheduler_requests'], page_width)
            if packed_shape is None:
                pindiag('[PINDIAG] QWEN_FAST_PACKED_STEP=1 builds no packed block for {} scheduler requests '
                        '(two take the 32-row M1 block, four the 64-row M3 block); the sequential step '
                        'serves the rounds', policy['scheduler_requests'])
        pool = ServingBufferPool(operations, model.mesh_device, users=policy['scheduler_requests'],
            helpers=helpers, page_width=page_width,
            bucket_rows=capture_bucket_rows(policy['verifier_rows'], policy['output_budget'], 16),
            feature_taps=len(TARGET_TAPS), rope=rope,
            **({} if packed_shape is None else dict(packed_shapes=((packed_shape.users, packed_shape.rows_per_user),))))
        scopes.callback(pool.close)
        owner = ServingCacheOwner(operations, runner, model)
        # Built once and shared by every request: two TT_CCL objects cycling semaphore
        # handles over one mesh is cross-request interference, not concurrency.
        collectives = TT_CCL(model.mesh_device)
        sampler = SamplingGenerator(args=model.args, mesh_device=model.mesh_device,
            tt_ccl=TT_CCL(model.mesh_device))
        sampler.set_trace_bucket(1)
        from serving_gather_experiment import from_environment

        experiment = from_environment(directory, runtime_root)

        scopes.enter_context(sampler_links(sampler.tt_sampling, 4))
        audit = scopes.enter_context(combined_runtime(operations, model, directory=directory,
            runtime_root=runtime_root, native_attention_evidence=native_attention_evidence,
            block_stream=block_stream, kv_publication_evidence=kv_publication_evidence))
        # The draft weights, uploaded once for every request: they are the first
        # buffers a request allocates, so they took the lowest hole an earlier
        # request's verify trace left, ahead of the history (PreparedDraftWeights).
        # After combined_runtime, because T16 native proposal preparation must run
        # inside its source-bound admission scope; still before any request. The
        # geometry is the one from_prefill asks every device for, and lend() refuses
        # a device asking for anything else.
        _, draft_layers, projection, selector = fixtures
        weights = PreparedDraftWeights(operations, model.mesh_device, draft_layers, projection, selector,
            block_rows=16, live_query_qk=False, native_proposal_attention=True)
        scopes.callback(weights.close)
        # The device step. By default the sequential one - correct, not yet fast: one
        # weight pass per user per round - and describe() records that cost so a benchmark
        # reading it is not mistaken for the goal. QWEN_FAST_PACKED_STEP=1 builds the
        # packed verify block instead, one pass serving every user at the shape chosen
        # above, and builds it HERE: after the pool and the shared draft weights, whose
        # buffers it restores from and checks, and before the lifecycle admits a request,
        # whose traces would otherwise bake over its buffers (packed_verifier.py,
        # CONSTRUCTION ORDER). Registered after the weights so it closes before them,
        # with the scope. Off by default: the sequential step stays the serving default
        # until the packed step's gates pass (docs/packed-device-step-plan-2026-09-20.md).
        from serving_sequential_step import describe as describe_sequential_step, sequential_packed_step

        packed_step, step_description = sequential_packed_step, describe_sequential_step()
        if packed_shape is not None:
            from packed_verifier import PackedVerifierEngine
            from serving_packed_step import describe as describe_packed_step, packed_device_step

            block = PackedVerifierEngine(operations, model, helpers, sampler, pool=pool, shared_weights=weights,
                                         shape=packed_shape, feature_taps=TARGET_TAPS)
            scopes.callback(block.close)
            packed_step = partial(packed_device_step, block=block)
            step_description = dict(describe_packed_step(), block=block.describe())
        elif packed_requested:
            step_description = dict(step_description,
                packed_block_skipped='no packed block shape for %d scheduler requests' % policy['scheduler_requests'])
        # One line with every pre-trace address - the pooled history pairs, each named
        # shared weight and, when built, the packed block's taps, checkpoints and carries -
        # so a diverged address from the shard check can be placed against what was
        # allocated before any request; and which device step serves the rounds.
        print(json.dumps(dict(stage='serving_buffer_pool', **pool.describe(), draft_weights=weights.describe(),
                              device_step=step_description)), flush=True)

        def capture_factory(position):
            owner.validate()
            return PrefillWindowCapture(operations, model, position, TARGET_TAPS)

        def bridge_factory(state, capture):
            owner.validate()
            if len(state.block_ids) != 1:
                raise ValueError('One scheduler KV group required')
            blocks = tuple(state.block_ids[0])
            if (not blocks or len(blocks) > page_width or len(set(blocks)) != len(blocks)
                    or any(type(block) is not int or not 0 <= block < owner.physical_pages for block in blocks)):
                raise ValueError('Unique physical pages from the admitted cache required')
            pages = torch.full((1, page_width), blocks[0], dtype=torch.int32)
            pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            def create_request():
                return from_prefill(operations, model, sampler, pages, helpers,
                    state=state, capture=capture, fixtures=fixtures, eos_ids=eos_ids,
                    collectives=collectives, buffer_pool=pool, shared_weights=weights)

            request = create_request() if experiment is None else experiment.create(create_request)
            try:
                binding = VerifierPageBinding(request.engine, blocks, physical_pages=owner.physical_pages)
                return FastRunnerBridge(runner, request, binding, validate_storage=owner.validate)
            except BaseException:
                request.close(state.req_id)
                raise

        lifecycle = FastServingLifecycle(worker, config=worker.vllm_config,
            capture_factory=capture_factory, bridge_factory=bridge_factory, eos_ids=eos_ids,
            cancelled=cancelled, packed_step=packed_step)
    except BaseException as failure:
        # Closing the scopes can itself raise (the block-stream scope checks at exit
        # that every layer ran the fused candidate, which nothing has at attach), and
        # run 35496483954 logged only that, losing the attach failure it was
        # handling. Keep the original: log the closing error and re-raise the first.
        try:
            scopes.close()
        except BaseException as closing:
            try:
                from loguru import logger
                logger.info('[PINDIAG] attach failed with {}: {}; closing the scopes then raised {}: {}',
                            type(failure).__name__, str(failure)[:300], type(closing).__name__, str(closing)[:200])
            except BaseException:
                pass
        raise failure
    try:
        yield dict(lifecycle=lifecycle, runtime=audit, cache_owner=owner,
            serving_qualified=False, performance_qualified=False)
    finally:
        lifecycle.close()
        scopes.close()
