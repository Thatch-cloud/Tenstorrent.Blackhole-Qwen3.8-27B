"""Explicit attachment of the admitted combined runtime to a loaded TT worker."""

from contextlib import ExitStack, contextmanager
from functools import partial
import json
import os

from dflash_device import PreparedDraftWeights, pindiag
import memory_ledger
from serving_buffer_pool import ServingBufferPool, dram_line
from serving_cache_owner import ServingCacheOwner
from serving_fast_policy import any_request_enabled, validate_fast_config
from serving_lifecycle import FastServingLifecycle
from serving_page_binding import VerifierPageBinding
from serving_request_factory import attach_source_check, from_prefill, sequential_captures
from serving_runner_bridge import FastRunnerBridge


SKIP_BLOCK_STREAM_FLAG = 'QWEN_FAST_SKIP_BLOCK_STREAM'
SINGLE_GATEUP_FLAG = 'QWEN_FAST_SINGLE_GATEUP'
SINGLE_GATEUP_SHAPE = 'the single gate/up copy'
M3_SHAPE = 'the 64-row block'
C2_ANY_SHAPE = 'C2-any with no packed block'
PADDED_BLOCK_FLAG = 'QWEN_FAST_PADDED_BLOCK'


def m3_shape(policy, environ=None):
    """(met, description): whether this is the 64-row M3 block - four scheduler requests,
    QWEN_FAST_FOUR_AS_TWO=0 and QWEN_FAST_PACKED_STEP=1 - and a short description of the
    shape actually configured, for a marker. `policy` is validate_fast_config's dict, or a
    zero-argument callable returning it."""
    environ = os.environ if environ is None else environ
    policy = policy() if callable(policy) else policy
    users, four_as_two, packed = (policy['scheduler_requests'], environ.get('QWEN_FAST_FOUR_AS_TWO', 'unset'),
                                  environ.get('QWEN_FAST_PACKED_STEP', 'unset'))
    met = users == 4 and four_as_two == '0' and packed == '1'
    return met, 'users=%s FOUR_AS_TWO=%s PACKED_STEP=%s' % (users, four_as_two, packed)


def c2_any_without_block(environ=None):
    """Whether this is C2-any with no packed block: QWEN_FAST_ANY_REQUEST=1 and QWEN_FAST_PACKED_STEP
    not 1 (the c2 profile). attach_combined_runtime then builds no block and caps every per-request
    engine at the sequential widths (1, 2, 4), so, as at the 64-row block, no target MLP ever sees 16
    rows and FusedT16Arm and the block stream behind it are never read."""
    environ = os.environ if environ is None else environ
    return environ.get('QWEN_FAST_ANY_REQUEST') == '1' and environ.get('QWEN_FAST_PACKED_STEP', 'unset') != '1'


def register_reader_reason(policy, environ=None):
    """Why the serial block stream may be replaced by the register-epilogue reader over the
    native w_gate_up (combined_runtime's block_stream=None branch), as a (shape, detail)
    pair for the startup marker - or None, the default, when the stream is required.

    Both flags are default-off and both are admitted at the 64-row M3 block ONLY (four
    scheduler requests, QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1), the one shape
    where no target MLP ever sees 16 rows - the per-request engines capture 1, 2 and 4 rows
    and the block 64 - so FusedT16Arm (fused_t16_scope.py:49-51) and the stream behind it
    are never read:
    - QWEN_FAST_SINGLE_GATEUP=1: the model builds no w_gate_up, so there is nothing to
      stream (and combined_runtime installs no FusedT16Arm either). At any OTHER shape it
      is REFUSED (ValueError): there the arm serves every 16-row verify, and replacing its
      BF4 register-epilogue projection with the native w1/w3 path changes the target's
      arithmetic, so committed tokens could change.
    - QWEN_FAST_SKIP_BLOCK_STREAM=1: elsewhere None, the stream is built as always (and
      serving_startup.weight_streams says so in a marker).

    The same holds for C2-any with no packed block (c2_any_without_block, the c2 profile): its
    engines capture 1, 2 and 4 rows and nothing else runs a verify, so both flags are admitted there
    too. Run 36219636175 refused SINGLE_GATEUP there, and turning it off would build w_gate_up and
    its stream again, about as much DRAM as dropping the block freed.

    The policy is evaluated only once either flag is set."""
    environ = os.environ if environ is None else environ
    single = environ.get(SINGLE_GATEUP_FLAG) == '1'
    if not single and environ.get(SKIP_BLOCK_STREAM_FLAG) != '1':
        return None
    met, shape = m3_shape(policy, environ)
    if not met and c2_any_without_block(environ):
        if single:
            return (SINGLE_GATEUP_SHAPE, 'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream')
        return (C2_ANY_SHAPE, 'register-epilogue reader on native w_gate_up')
    if single:
        if not met:
            raise ValueError('QWEN_FAST_SINGLE_GATEUP=1 is admitted only at the 64-row M3 block '
                             '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not ' + shape)
        return (SINGLE_GATEUP_SHAPE, 'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream')
    if not met:
        return None
    return (M3_SHAPE, 'register-epilogue reader on native w_gate_up')


def padded_block_admission(policy, environ=None):
    """QWEN_FAST_PADDED_BLOCK (variable-user packed rounds M2, default off): None while the flag
    is off - the policy is not even read - else the fewest live users a round of the block may
    serve (QWEN_FAST_PADDED_BLOCK_MIN_USERS, default 2; packed_verifier.padded_block_min_users),
    which the 64-row block is then built with. Admitted at the 64-row M3 block ONLY, by the same
    rule as QWEN_FAST_SINGLE_GATEUP (m3_shape): every other shape is REFUSED (ValueError). The
    padded round is the M3 trace with idle segments on page 0 - measured exact at that trace
    (v188) and nowhere else - and two idle segments are all page 0's two tile rows hold."""
    environ = os.environ if environ is None else environ
    if environ.get(PADDED_BLOCK_FLAG, '0') == '0':
        return None
    from packed_verifier import padded_block_min_users

    minimum = padded_block_min_users(environ)   # refuses any value but '1' here
    met, shape = m3_shape(policy, environ)
    if not met:
        raise ValueError('QWEN_FAST_PADDED_BLOCK=1 is admitted only at the 64-row M3 block '
                         '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not ' + shape)
    return minimum


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
    # Measurement-only, env-gated admission of one named grafted binary (K64 kernel
    # graft): inert unless QWEN_FAST_RUNTIME_BINARY_SHA256 is set, and it neither
    # touches the hash-pinned sources nor lowers the pin itself. Must run before
    # combined_runtime() is entered below, since dflash_t16_native_scope.admit runs inside it.
    from runtime_binary_override import install as override_runtime_binary
    binary_record = override_runtime_binary(runtime_root, log=pindiag)
    # Measurement-only, env-gated admission of the device profiler into the mandatory
    # block-stream recipe (attribution only, never a throughput claim): inert unless
    # QWEN_FAST_PROFILED_BLOCK_STREAM=1, and edits no recipe file. Must run before
    # combined_runtime() is entered below, since scoped_block_stream's require_hardware
    # check runs inside it.
    from profiled_block_stream_override import install as admit_profiled_block_stream
    admit_profiled_block_stream(log=pindiag)
    # The serial weight stream stays mandatory except at the shape register_reader_reason
    # admits (each flag default-off), where the register-epilogue reader over the native
    # w_gate_up replaces it. Read before anything is built: QWEN_FAST_SINGLE_GATEUP=1 at any
    # other shape is refused here, whatever recipe startup handed over.
    reader = register_reader_reason(policy)
    # QWEN_FAST_PADDED_BLOCK (default off): the 64-row block's fewest live users per round, or
    # None. Refused here at any other shape, before anything is built, like the single copy.
    padded_min_users = padded_block_admission(policy)
    # S2 C2-packed-any (QWEN_FAST_EXTENT_REPLAY, default off; strictly '0' or '1'): packed rounds at any
    # position through the K64j extent readers. Admitted here or nowhere, host only and before anything
    # is built, so a refusal builds nothing: the M3 shape, C2-any, eight-row groups, the tree-scratch
    # precondition, the modes, the K64j binary and kernels, and the pinned hardware evidence
    # (packed_any_admission, design W7). Off (unset or '0'), nothing here runs, not even the import - the
    # module ships in the C2 overlay only, as serving_lifecycle's lazy quarantine import does - and the
    # attach is today's.
    extent_replay = os.environ.get('QWEN_FAST_EXTENT_REPLAY', '0') != '0'
    if extent_replay:
        import packed_any_admission

        packed_any_admission.extent_replay_enabled()   # strictly '1' from here: any other value is refused
        packed_any_admission.admit(runtime_root, m3=m3_shape(policy), binary_record=binary_record, log=pindiag)
    if (native_attention_evidence is None or kv_publication_evidence is None
            or (block_stream is None and reader is None)
            or (block_stream is not None and 'pipeline_evidence' in block_stream)):
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
        # before the pool, which lends the block(s) their per-user replay page tables for
        # exactly that shape; with the switch unset the pool is built as it always was.
        #
        # Four requests are the one count with a choice: QWEN_FAST_FOUR_AS_TWO (default ON
        # at four requests) builds TWO 32-row M1 blocks instead of the single 64-row M3
        # block - block A over pool slots (0, 1), block B over (2, 3) - because two proven,
        # workaround-free 32-row rounds (M1b, 109 ms measured, run 35500352729) cost far
        # less than the 64-row round's two-tile workarounds (1453 ms, run 35544598063) for
        # the same four users. QWEN_FAST_FOUR_AS_TWO=0 keeps the single M3 block.
        packed_requested = os.environ.get('QWEN_FAST_PACKED_STEP') == '1'
        four_as_two = False
        packed_shapes = ()
        if packed_requested:
            from packed_shapes import m1_shape, serving_shape

            if policy['scheduler_requests'] == 4 and os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':
                four_as_two = True
                packed_shapes = (m1_shape(page_width), m1_shape(page_width))
            else:
                shape = serving_shape(policy['scheduler_requests'], page_width)
                if shape is None:
                    pindiag('[PINDIAG] QWEN_FAST_PACKED_STEP=1 builds no packed block for {} scheduler requests '
                            '(two take the 32-row M1 block, four the 64-row M3 block); the sequential step '
                            'serves the rounds', policy['scheduler_requests'])
                else:
                    packed_shapes = (shape,)
        # S2 (QWEN_FAST_EXTENT_REPLAY=1) serves its rounds through the packed block alone - the pool lends the
        # block its extent storage and the block keys on it - so with no block to build the flag would build
        # nothing and every round would run sequentially. The admission's M3 shape check already makes this
        # unreachable; refused here too, before the pool, the first allocation.
        if extent_replay and not packed_shapes:
            raise ValueError('QWEN_FAST_EXTENT_REPLAY=1 serves its rounds through the packed block, and this attach '
                             'builds none (QWEN_FAST_PACKED_STEP=%s, %d scheduler requests)'
                             % (os.environ.get('QWEN_FAST_PACKED_STEP', 'unset'), policy['scheduler_requests']))
        # The per-request engines' capture widths, and the pool's buckets that hold them:
        # the full T16 set by default and beside the 32-row block, the sequential widths
        # (1, 2, 4) beside the 64-row block, whose four engines' 8- and 16-row captures do
        # not fit (packed_shapes.sequential_capture_rows; run 35509307389, image v52, OOM
        # in the first engine with 32.9 of 33.1 GB per chip allocated). The rounds the
        # block serves are drafted at its width (serving_packed_step.proposal_rows); the
        # survivors decode sequentially at four rows per round.
        #
        # Two 32-row blocks retain the SAME total state as the one 64-row block - every
        # per-segment term in packed_verifier.py scales from users x rows_per_user, 4 x 16
        # either way, not from one block's own width - so nothing here has re-measured a
        # device with two 32-row blocks built beside four untrimmed engines and found the
        # headroom the single 64-row measurement did not have (run 35509307389: 32.9 of
        # 33.1 GB allocated with ONE untrimmed engine's captures still pending). Trimming
        # is the safe assumption for four_as_two too: `sequential_capture_rows(m1_shape)`
        # would answer 16 (correctly - a LONE 32-row block never needed the trim), so the
        # four-user, two-block case is decided explicitly here instead, by total rows
        # rather than by either shape alone.
        from packed_shapes import M3_SEQUENTIAL_CAPTURE_ROWS, sequential_capture_rows

        if four_as_two:
            capture_rows = M3_SEQUENTIAL_CAPTURE_ROWS
        else:
            capture_rows = sequential_capture_rows(packed_shapes[0] if packed_shapes else None)
        # C2-any with no packed block at all (the c2 profile sets QWEN_FAST_PACKED_STEP=0): its
        # engines drop replay attention and the T16 gate, so they must capture the sequential
        # widths whatever the block would have asked. The block could never engage under c2 - it
        # needs every live user inside [131072, 131328), and c2 caps prompts at 123136 - yet it held
        # 3.84 GB per chip, and run 36218104858 died building the fourth engine with 0.79 GB free
        # (engines cost 0.80 GB at one proposal bucket, 1.35 GB at four).
        no_block_any_request = any_request_enabled() and not packed_shapes
        if no_block_any_request:
            capture_rows = M3_SEQUENTIAL_CAPTURE_ROWS
        # QWEN_FAST_ANY_REQUEST (C2-any, plan S1; default off). Its engines drop replay
        # attention and the per-request T16 gate, which is only sound where every capture is
        # narrower than a replayed block (serving_request_factory.sequential_captures): beside
        # the four-user block. Anywhere else the gate would still refuse every request that is
        # not the frozen benchmark shape, so the attach is refused instead, before anything is
        # built. And the gate's source qualification - every pinned component source hashed
        # against the frozen evidence - runs here, once, in its place: a mismatch fails the
        # attach, as it failed the first admission before (image v42).
        if any_request_enabled():
            if not sequential_captures(capture_rows):
                raise ValueError('QWEN_FAST_ANY_REQUEST=1 needs per-request captures narrower than a replayed block '
                                 '(beside the four-user block: QWEN_FAST_PACKED_STEP=1 with four scheduler '
                                 'requests); this attach caps them at %d rows' % capture_rows)
            attach_source_check()
        bucket_rows = capture_bucket_rows(policy['verifier_rows'], policy['output_budget'], capture_rows)
        trimmed = capture_rows != policy['verifier_rows']
        if trimmed and no_block_any_request:
            pindiag('[PINDIAG] per-request captures trimmed to widths {} for C2-any with no packed block',
                    tuple(sorted(set(bucket_rows))))
        elif trimmed:
            pindiag('[PINDIAG] per-request captures trimmed to widths {} for the four-user block',
                    tuple(sorted(set(bucket_rows))))
        # packed_shapes covers each distinct shape once (validate_packed_shapes still
        # refuses a shape repeated there); four_as_two's two blocks share one (2, 16) shape,
        # so its multiplicity is named separately, and the pool lends each block asking for
        # it its OWN independent replay table set (ServingBufferPool.packed_replay).
        distinct_shapes = tuple(dict.fromkeys((shape.users, shape.rows_per_user) for shape in packed_shapes))
        if packed_shapes:
            # The packed block's own replay grouping (QWEN_FAST_REPLAY_GROUP_ROWS), which
            # may differ from the per-request bucket slots' fixed four-row grouping above -
            # packed_verifier.py is the only reader of this value.
            from packed_verifier import replay_group_rows
        pool = ServingBufferPool(operations, model.mesh_device, users=policy['scheduler_requests'],
            helpers=helpers, page_width=page_width, bucket_rows=bucket_rows,
            feature_taps=len(TARGET_TAPS), rope=rope,
            **({} if not packed_shapes else dict(packed_shapes=distinct_shapes,
                packed_replay_group_rows=replay_group_rows(),
                **({'packed_replicas': {distinct_shapes[0]: len(packed_shapes)}} if four_as_two else {}),
                # S2: the extent storage in place of the per-family tables; flag off, no keyword at all.
                **({'extent_replay': True} if extent_replay else {}))))
        scopes.callback(pool.close)
        if extent_replay:
            # The pool must hold the extent storage the S2 block keys on (design W2: without it the block
            # would be the per-family one, packed only in [131072, 131312], and nothing would say so), and
            # (design B4) its DRAM statistics must be readable, since the scheduler-side DRAM hold reads them
            # for every request. Either fails the attach here, before any trace (the pool closes with the
            # scopes).
            packed_any_admission.admit_pool(pool, log=pindiag)
        memory_ledger.record('P2', buffer_pool=pool)
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
        memory_ledger.record('P3', serving_collectives=collectives, serving_sampler=sampler, gather_experiment=experiment)
        audit = scopes.enter_context(combined_runtime(operations, model, directory=directory,
            runtime_root=runtime_root, native_attention_evidence=native_attention_evidence,
            block_stream=block_stream, kv_publication_evidence=kv_publication_evidence,
            **({'single_gateup_admitted': True} if reader is not None and reader[0] == SINGLE_GATEUP_SHAPE
               else {})))
        memory_ledger.record('P4', combined_runtime=audit)
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
        memory_ledger.record('P5', draft_weights=weights)
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
        if packed_shapes:
            from packed_verifier import PackedVerifierEngine
            from serving_packed_step import PackedStep, describe as describe_packed_step

            # One block per configured shape, each bound to its own disjoint pool slots in
            # scheduler order - four_as_two's block A over (0, 1), block B over (2, 3) - so
            # every block keeps its fixed carry-identity binding for the pool's whole life
            # and no round ever rebinds a segment. A single configured shape (the m1
            # two-user or m3 four-user default) gets no `pool_slots=` at all, so its block
            # is built exactly as it always was: slots 0..users-1, in order.
            packed_blocks, slot = [], 0
            for shape in packed_shapes:
                packed_block = PackedVerifierEngine(operations, model, helpers, sampler, pool=pool, shared_weights=weights,
                                                    shape=shape, feature_taps=TARGET_TAPS,
                                                    **({'pool_slots': tuple(range(slot, slot + shape.users))} if four_as_two else {}),
                                                    **({'padded_min_users': padded_min_users}
                                                       if padded_min_users is not None else {}),
                                                    # Round-fence plan H1b (QWEN_FAST_FUSED_COMMIT, default
                                                    # off): the block's T_proj traces use the one shared
                                                    # TT_CCL every request's device uses.
                                                    **({'collectives': collectives}
                                                       if os.environ.get('QWEN_FAST_FUSED_COMMIT') == '1' else {}))
                scopes.callback(packed_block.close)
                packed_blocks.append(packed_block)
                memory_ledger.record('P6', point='block%d' % len(packed_blocks), packed_block=packed_block)
                slot += shape.users
            # The step bound to its block (or blocks), carrying the per-round ticket-width
            # policy the worker hook asks before drafting.
            packed_step = PackedStep(packed_blocks if four_as_two else packed_blocks[0])
            step_description = dict(describe_packed_step(),
                **(dict(blocks=[packed_block.describe() for packed_block in packed_blocks]) if four_as_two
                   else dict(block=packed_blocks[0].describe())))
        elif packed_requested:
            step_description = dict(step_description,
                packed_block_skipped='no packed block shape for %d scheduler requests' % policy['scheduler_requests'])
        if extent_replay:
            # The executed path is the admitted one (memory graft-mounted-is-not-graft-executed): every block
            # is the extent block and every segment reader reports runtime_extent, or the attach fails here,
            # before the lifecycle admits a request (the scopes close the block, the weights and the pool).
            packed_any_admission.admit_blocks(packed_blocks, log=pindiag)
        # One line with every pre-trace address - the pooled history pairs, each named
        # shared weight and, when built, the packed block's taps, checkpoints and carries -
        # so a diverged address from the shard check can be placed against what was
        # allocated before any request; and which device step serves the rounds.
        print(json.dumps(dict(stage='serving_buffer_pool', **pool.describe(), draft_weights=weights.describe(),
                              device_step=step_description)), flush=True)
        # The device allocator after everything the attach allocates - weights, KV pool,
        # buffer pool, draft weights, the block and its traces - so the log shows the
        # headroom the per-request engines have (run 35509307389 found 214 MB of it).
        pindiag('[PINDIAG] dram after attach: {}', dram_line(pool))
        memory_ledger.record('P7', point='after_attach')

        def capture_factory(position):
            owner.validate()
            # The allocator just before this user's prefill; the bridge factory reads it
            # again just after, so the pair bounds what the prefill leaves resident.
            memory_ledger.record('prefill', point='before prompt=%d' % position)
            return PrefillWindowCapture(operations, model, position, TARGET_TAPS)

        def bridge_factory(state, capture):
            owner.validate()
            memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id(state.req_id),
                                 request=str(state.req_id), model_after_prefill=model)
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
                    collectives=collectives, buffer_pool=pool, shared_weights=weights,
                    **(dict(capture_rows=capture_rows) if trimmed else {}))

            request = create_request() if experiment is None else experiment.create(create_request)
            # The allocator after this request's engine and its captures: one line per
            # admitted request, so the log shows what each costs and what is left.
            pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))
            memory_ledger.engine_admitted(str(state.req_id), engine_request=request)
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
        # And log the failure BEFORE closing: the scopes' exits fence the device
        # (the block's close, then the admitted runtime's shared-QK scope first), and a
        # device left hung by the failed attach blocks the first fence for good - run
        # 35507675630 (image v51) sat in gdn_shared_qk_scope's exit for the rest of its
        # 900 s with nothing in the log to say so.
        pindiag('[PINDIAG] attach failed with {}: {}; closing the attach scopes now (the packed block if built, '
                'the draft weights, the admitted combined runtime from its shared-QK scope, the sampler links, '
                'the pool) - their exits fence the device, and a hung device blocks the first fence',
                type(failure).__name__, str(failure)[:300])
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
        memory_ledger.record('P13', point='before_shutdown')
        lifecycle.close()
        scopes.close()
