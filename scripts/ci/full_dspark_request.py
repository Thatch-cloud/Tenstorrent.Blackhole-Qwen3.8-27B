"""Complete full-history DSpark coding-request experiment; target exactness, not held-out quality certification."""

import json
from contextlib import ExitStack

from dspark_device import DSparkDevice
from dspark_intake import TAPS
from dspark_prefill import FullHistoryCapture
from dspark_projection import tensor_digest
from dspark_request_runtime import DSparkRequestRuntime
from gdn_multitoken_conv import addresses
from target_features import LayerOutputCapture


def measure_dspark_request(operations, model, sampler, prompt, pages, helpers, *, collectives,
        parameters, layer_weights, predecessor, successor, rotary, prefill, decode,
        live_digest, kv_digest, inactive_digest, eos_ids, audit_features=False, max_new_tokens=257,
        proposal_trace=False, commit_only_gdn=False, native_attention=False, profile_verifier=False,
        target_attention_t16=False, score_layout=False, score_layout_evidence=None,
        banked_proposal=False, banked_proposal_evidence=None, native_slot_gdn=False, fused_t16_mlp=False,
        history_profile=False, captured_publication=False, gdn_output_l1=False, gdn_output_grid=False,
        gdn_copy_pairs=False, gdn_outer_add=False, combined_profile=False, gdn_shared_qk=False):
    import torch
    from full_request import measure_request
    if type(combined_profile) is not bool or (combined_profile and not (
            audit_features and captured_publication and fused_t16_mlp and target_attention_t16
            and commit_only_gdn and native_attention and proposal_trace and not profile_verifier)):
        raise ValueError('Combined attribution requires the audited complete publication runtime')
    if type(gdn_outer_add) is not bool or (gdn_outer_add and (gdn_copy_pairs or gdn_output_l1 or gdn_output_grid or not (
            captured_publication and fused_t16_mlp and target_attention_t16 and commit_only_gdn))):
        raise ValueError('Outer-add fusion requires the isolated combined runtime')
    shared_audit = shared_admission = None
    if type(gdn_shared_qk) is not bool or (gdn_shared_qk and (
            gdn_outer_add or gdn_copy_pairs or gdn_output_l1 or gdn_output_grid or combined_profile or not (
            captured_publication and fused_t16_mlp and target_attention_t16 and commit_only_gdn))):
        raise ValueError('Shared Q/K requires the isolated combined runtime')
    if gdn_shared_qk:
        import os
        from pathlib import Path
        from gdn_shared_qk_gate import qualify
        shared_admission = qualify(Path(__file__).with_name('gdn-shared-recurrence.json'),
            Path(__file__).parent, os.environ['TT_METAL_HOME'])
    outer_audit = outer_admission = None
    if gdn_outer_add:
        import os
        from pathlib import Path
        from gdn_outer_add_gate import qualify
        outer_admission = qualify(Path(__file__).with_name('gdn-outer-add.json'),
            Path(__file__).parent, os.environ['TT_METAL_HOME'])
    if type(gdn_copy_pairs) is not bool or (gdn_copy_pairs and (gdn_output_l1 or gdn_output_grid or not (
            captured_publication and fused_t16_mlp and target_attention_t16 and commit_only_gdn))):
        raise ValueError('Copy pairs require the combined runtime without other GDN candidates')
    copy_audit = copy_admission = None
    if gdn_copy_pairs:
        import os
        from pathlib import Path
        from gdn_copy_pairs_gate import qualify
        copy_admission = qualify(Path(__file__).with_name('gdn-copy-pairs.json'),
            Path(__file__).parent, os.environ['TT_METAL_HOME'])
    if type(gdn_output_grid) is not bool or (gdn_output_grid and (gdn_output_l1 or not (
            captured_publication and fused_t16_mlp and target_attention_t16 and commit_only_gdn))):
        raise ValueError('GDN grid requires the combined runtime without a competing placement candidate')
    if type(gdn_output_l1) is not bool or (gdn_output_l1 and not (
            captured_publication and fused_t16_mlp and target_attention_t16 and commit_only_gdn)):
        raise ValueError('GDN L1 output requires the explicit combined publication runtime')
    output_arm = output_admission = None
    if gdn_output_grid:
        import os
        from pathlib import Path
        from gdn_output_grid_gate import qualify
        output_admission = qualify(Path(__file__).with_name('gdn-output-grid.json'),
            Path(__file__).parent, os.environ['TT_METAL_HOME'])
    if gdn_output_l1:
        import os
        from pathlib import Path
        from gdn_output_l1_gate import qualify
        output_admission = qualify(Path(__file__).with_name('gdn-output-l1.json'),
            Path(__file__).parent, os.environ['TT_METAL_HOME'])
    if type(captured_publication) is not bool or (captured_publication and (
            not proposal_trace or not commit_only_gdn or banked_proposal or history_profile or profile_verifier)):
        raise ValueError('Captured publication requires the traced commit-only request without competing history profiles')
    publication_arm = None
    if type(history_profile) is not bool or (history_profile and banked_proposal):
        raise ValueError('Explicit history attribution requires the standard fixed-history request path')
    history_observer = None
    if history_profile:
        from history_publication_profile import HistoryPublicationProfile
        history_observer = HistoryPublicationProfile()
    if type(fused_t16_mlp) is not bool or (fused_t16_mlp and not (
            proposal_trace and commit_only_gdn and native_attention and target_attention_t16
            and score_layout and not profile_verifier and not native_slot_gdn and not banked_proposal)):
        raise ValueError('T16 fusion requires the combined score-layout runtime without other experimental MLP/state routes')
    if type(native_slot_gdn) is not bool or (native_slot_gdn and (not score_layout or banked_proposal)):
        raise ValueError('Native-slot GDN requires the combined score-layout runtime without banked drafting')
    bank_evidence = None
    if type(banked_proposal) is not bool or (banked_proposal and not (
            proposal_trace and commit_only_gdn and native_attention and target_attention_t16 and not profile_verifier)):
        raise ValueError('Banked proposals require the combined traced native-drafter folded-target experiment')
    if banked_proposal:
        from pathlib import Path
        from dspark_banked_gate import qualify as qualify_banks
        if banked_proposal_evidence is None:
            raise ValueError('Banked proposal simulator evidence required')
        bank_evidence = qualify_banks(banked_proposal_evidence, Path(__file__).parent)
    elif banked_proposal_evidence is not None:
        raise ValueError('Bank evidence requires explicit banked proposal selection')
    if type(score_layout) is not bool or (score_layout and not (
            proposal_trace and commit_only_gdn and native_attention and target_attention_t16 and not profile_verifier)):
        raise ValueError('Score layout requires the distinct traced native-drafter folded-target experiment')
    if type(target_attention_t16) is not bool:
        raise ValueError('Choose a distinct T16 target attention experiment')
    if target_attention_t16:
        from pathlib import Path
        from target_t16_attention_gate import qualify, validate_request_option
        validate_request_option(True, rows=16, position=len(prompt), remaining=max_new_tokens - 1,
            replay=True, norm_batch=True, native_sampling=True, group_rows=4, short_context=False)
        qualify(Path(__file__).parent)
    if type(profile_verifier) is not bool or (profile_verifier and not (
            audit_features and proposal_trace and commit_only_gdn and native_attention)):
        raise ValueError('Profile only the audited native-attention commit-only request')

    if (any(type(value) is not bool for value in (audit_features, proposal_trace, commit_only_gdn, native_attention))
            or (native_attention and not proposal_trace)
            or type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 513
            or not 1 <= len(prompt) <= 8192 - max_new_tokens):
        raise ValueError('Explicit audit policy and full-history capacity for the complete request required')
    if native_attention:
        import os
        from pathlib import Path
        from dspark_native_fixed_gate import qualify
        from native_draft_sdpa import audit_active_kernel
        qualify(Path(__file__).parent)
        audit_active_kernel(os.environ['TT_METAL_HOME'])
    capture = drafter = runtime = proposal_device = None
    score_scope, score_arm = ExitStack(), None
    golden_features, prefill_hashes = {}, None
    prefill_records, feature_checks, history_checks, proposal_checks = [], [], [], []
    seed = None

    def status(stage, **values):
        print(json.dumps(dict(dspark_stage=stage, **values)), flush=True)

    def captured_prefill(tokens):
        nonlocal capture, seed, prefill_hashes
        if capture is not None:
            capture.close()
        capture = FullHistoryCapture(operations, model, len(prompt))
        status('prefill', ordinal=len(prefill_records), context=len(prompt), audit=audit_features)
        with capture.capture():
            seed = prefill(tokens)
        chunks = capture.outputs()
        records = [dict(start=chunk.start, rows=chunk.rows, bucket=chunk.features[0].shape[2]) for chunk in chunks]
        if audit_features:
            observed = []
            for chunk in chunks:
                for tap, value in zip(TAPS, chunk.features, strict=True):
                    shards = operations.get_device_tensors(value)
                    if len(shards) != 2:
                        raise AssertionError('Both actual prefill feature shards required')
                    for chip, shard in enumerate(shards):
                        observed.append(dict(start=chunk.start, rows=chunk.rows, tap=tap, chip=chip,
                            sha256=tensor_digest(operations.to_torch(shard)[..., :chunk.rows, :])))
            if prefill_hashes is not None and observed != prefill_hashes:
                raise AssertionError('Fresh candidate prefill features differ from native controls')
            prefill_hashes = observed
        prefill_records.append(records)
        return seed

    def gold_decode(token, position, traced):
        if not audit_features:
            return decode(token, position, traced)
        observed = LayerOutputCapture(model, TAPS,
            snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
            release=operations.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
        try:
            with observed.capture():
                output = decode(token, position, False)
            features = []
            for value in observed.outputs():
                shards = operations.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Both native decode feature shards required')
                features.append(tuple(operations.to_torch(shard).clone() for shard in shards))
            golden_features[position] = tuple(features)
            return output
        finally:
            observed.close()

    def validate_features(features, prefix, position):
        for index, (tap, value) in enumerate(zip(TAPS, features, strict=True)):
            shards = operations.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError('Both verifier feature shards required')
            for chip, shard in enumerate(shards):
                expected = torch.cat([golden_features[position + row][index][chip] for row in range(prefix)], dim=2)
                actual = operations.to_torch(shard)[..., :prefix, :]
                if not torch.equal(actual, expected):
                    from dspark_feature_mismatch import FeatureMismatch
                    error = FeatureMismatch(actual, expected, tap=tap, chip=chip, position=position)
                    if runtime is not None and runtime.engine is not None:
                        engine = runtime.engine
                        ticket = engine.pending
                        fixture = engine.buckets[engine.pending_key]['fixture']
                        error.evidence['ticket_tokens'] = list(ticket.tokens)
                        error.evidence['ticket_position'] = ticket.position
                        error.evidence['verifier_inputs'] = {
                            name: [dict(address=part.buffer_address(),
                                values=operations.to_torch(part).reshape(-1).tolist())
                                for part in operations.get_device_tensors(getattr(fixture, name))]
                            for name in ('tokens', 'positions')}
                        error.evidence['proposal_checks'] = list(proposal_device.prepared.checks) if proposal_trace else []
                        if proposal_trace and position == len(prompt):
                            from dspark_verifier_failure import compare_first_block
                            try:
                                error.evidence['first_block_comparison'] = compare_first_block(engine, decode,
                                    drafter.snapshot, drafter.initial_state, actual[..., :1, :].clone(), tap=tap, chip=chip,
                                    fresh_prefill=lambda: prefill(prompt))
                            except BaseException as diagnostic_error:
                                error.evidence['first_block_diagnostic_error'] = f'{type(diagnostic_error).__name__}: {diagnostic_error}'
                    raise error
                feature_checks.append(dict(tap=tap, chip=chip, position=position, rows=prefix, exact=True))

    preparing_engine = None

    def protected_verifier_snapshot():
        engine = preparing_engine if runtime is None or runtime.engine is None else runtime.engine
        if engine is None:
            return None
        result = []
        for layer, snapshot in enumerate(engine.initial):
            for operand, value in enumerate(snapshot):
                shards = operations.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Both saved verifier state shards required')
                for chip, shard in enumerate(shards):
                    result.append(dict(layer=layer, operand=operand, chip=chip,
                        address=shard.buffer_address(), sha256=tensor_digest(operations.to_torch(shard))))
        return result

    def prepare_proposal_trace(engine):
        nonlocal preparing_engine
        preparing_engine = engine
        status('prepare_proposal_trace_after_verifier_persistent_allocation')
        drafter.prepare_trace(seed, audit=audit_features)
        status('warm_fifteen_query_proposal_before_verifier_capture')
        drafter.propose(seed, 15)

    def factory():
        nonlocal drafter, runtime, proposal_device, score_arm, publication_arm
        status('project_full_prefill_history', context=len(prompt))
        implementation = DSparkDevice
        if proposal_trace:
            from dspark_prepared_proposal import TracedDSparkDevice
            implementation = TracedDSparkDevice
        if banked_proposal:
            from dspark_banked_device import BankedDSparkDevice
            implementation = BankedDSparkDevice
        drafter = implementation(operations, model, collectives, parameters, layer_weights, predecessor, successor,
            capture.outputs(), rotary, position=len(prompt), proposals=15,
            **(dict(native_attention=True) if native_attention else {}),
            history_capacity=((len(prompt) + max_new_tokens + 31) // 32) * 32)
        proposal_device = drafter
        if captured_publication:
            from dspark_publication_scope import CapturedPublicationArm
            status('capture_history_projection_before_verifier_allocation')
            publication_arm = CapturedPublicationArm(proposal_device.history, audit=audit_features)
            score_scope.enter_context(publication_arm.install())
        if history_observer is not None:
            score_scope.enter_context(history_observer.install(proposal_device.history))
        if score_layout:
            from dspark_score_layout_scope import ScoreLayoutArm
            score_arm = ScoreLayoutArm(proposal_device, hardware_audit=score_layout_evidence)
            score_scope.enter_context(score_arm.install())
        capture.close()
        if audit_features:
            from dspark_history_audit import AuditedHistoryDrafter
            drafter = AuditedHistoryDrafter(operations, drafter, history_checks)
        if proposal_trace:
            if audit_features:
                from dspark_target_state_audit import TargetStateAuditedDrafter
                drafter = TargetStateAuditedDrafter(drafter,
                    lambda: dict(gdn=live_digest(), kv=kv_digest(drafter.position), inactive=inactive_digest()),
                    protected_snapshot=protected_verifier_snapshot)
        else:
            status('warm_fifteen_query_proposal_before_verifier_capture')
            drafter.propose(seed, 15)
        runtime = DSparkRequestRuntime(drafter, position=len(prompt), validate_features=validate_features if audit_features else None)
        return runtime

    observer = native_slot_arm = fusion_arm = None
    try:
        if gdn_shared_qk:
            from gdn_shared_qk_scope import scoped_shared_qk
            shared_audit = score_scope.enter_context(scoped_shared_qk(operations, shared_admission))
        if gdn_outer_add:
            from gdn_outer_add_scope import scoped_outer_add
            outer_audit = score_scope.enter_context(scoped_outer_add(outer_admission))
        if gdn_copy_pairs:
            from gdn_copy_pairs_scope import scoped_copy_pairs
            copy_audit = score_scope.enter_context(scoped_copy_pairs(copy_admission))
        if gdn_output_grid:
            from gdn_output_grid_scope import GDNOutputGridArm
            from models.demos.blackhole.qwen36.tt.tp_common import matmul_1d_decode
            output_arm = GDNOutputGridArm(operations, model, matmul_1d_decode)
            score_scope.enter_context(output_arm.install())
        if gdn_output_l1:
            from gdn_output_l1_scope import GDNOutputL1Arm
            from models.demos.blackhole.qwen36.tt.tp_common import matmul_1d_decode
            output_arm = GDNOutputL1Arm(operations, model, matmul_1d_decode)
            score_scope.enter_context(output_arm.install())
        if fused_t16_mlp:
            from fused_t16_scope import FusedT16Arm
            from models.tt_transformers.tt.ccl import tt_all_reduce
            fusion_arm = FusedT16Arm(operations, model, tt_all_reduce)
            score_scope.enter_context(fusion_arm.install())
        if native_slot_gdn:
            from gdn_native_slot_scope import NativeSlotArm
            native_slot_arm = NativeSlotArm(operations, model)
            score_scope.enter_context(native_slot_arm.install())
        if profile_verifier or combined_profile:
            from request_verifier_profile import RequestVerifierProfile
            observer = RequestVerifierProfile(operations, model.mesh_device, full_rows=16,
                target_attention_t16=target_attention_t16)
        result = measure_request(model, sampler, prompt, pages, helpers, prefill=captured_prefill, decode=gold_decode,
            live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest, eos_ids=eos_ids,
            max_new_tokens=max_new_tokens, norm_batch=True, native_sampling_rows=True,
            commit_only_gdn=commit_only_gdn, audit_commit_only_gdn=audit_features and commit_only_gdn,
            lookup_max_rows=16, feature_factory=factory, feature_drafter_name='dspark',
            **(dict(target_attention_t16=True, attention_replay=True, family_routing=True)
               if target_attention_t16 else {}),
            **(dict(verifier_before_capture=prepare_proposal_trace) if proposal_trace else {}),
            **(dict(verifier_observer=observer) if observer is not None else {}),
            progress=lambda block: status('committed-block', **block))
        expected_checks = len(result['blocks']) * len(TAPS) * 2 if audit_features else 0
        if (runtime is None or len(prefill_records) != 2 or len(feature_checks) != expected_checks
                or runtime.committed_feature_rows != result['committed_decode_tokens']
                or drafter.position != len(prompt) + result['committed_decode_tokens']):
            raise AssertionError('Complete target-feature publication and exact full-history frontier required')
        if proposal_trace:
            proposal_checks = list(proposal_device.prepared.checks)
            replay_count = 1 + sum(block['rows'] > 1 for block in result['blocks'])
            if len(proposal_checks) != (replay_count if audit_features else 0):
                raise AssertionError('Every changing-input proposal replay and warmup must pass its eager audit')
        result['publication_diagnostics'] = dict(records=runtime.publication_diagnostics.records,
            scope='Host wall and process CPU with GC pauses; no added device fences; timing includes instrumentation')
        if publication_arm is not None:
            result['captured_publication'] = dict(enabled=True,
                checks=list(publication_arm.projection.checks),
                scope='Captured learned feature projection with transactional fixed history; full request costs retained')
        if history_observer is not None:
            result['history_publication_profile'] = dict(records=list(history_observer.records),
                scope='Nested host-wall timers; total includes projection and bank assembly; no added fences; not device-kernel timing')
        result['dspark'] = dict(proposals=15, verifier_rows=16, full_history=True, prefill_chunks=prefill_records,
            prefill_hashes=prefill_hashes, feature_checks=feature_checks, audit_features=audit_features,
            history_checks=history_checks,
            fixed_history_capacity=drafter.history.capacity, persistent_history_allocated_before_verifier=True,
            committed_feature_rows=runtime.committed_feature_rows, final_position=drafter.position,
            execution=('Captured fixed-capacity' if proposal_trace else 'Eager')
                + ' full-history proposal; batched captured target verifier; all request-loop costs retained',
            proposal_trace=proposal_trace, proposal_checks=proposal_checks, native_attention=native_attention,
            packed_token_readbacks_per_proposal=2, checkpoint_trained_block_rows=16,
            published_serving_proposals=7, wider_proposal_acceptance_qualified=False)
        if observer is not None:
            result['verifier_profile'] = observer.summary()
        if banked_proposal:
            result['dspark']['banked_proposal'] = dict(evidence=bank_evidence,
                replay_counts=list(proposal_device.prepared.replay_counts),
                scope='Combined request candidate; not a serving-default change')
        result['instrumented_timing'] = audit_features
        if audit_features:
            result['committed_tokens_per_second'] = None
        result['kind'] = 'Full-history DSpark coding-request pilot; not held-out coding quality'
        result['qualification'] = __doc__
        score_scope.close()
        if shared_audit is not None:
            result['gdn_shared_qk'] = shared_audit
        if outer_audit is not None:
            result['gdn_outer_add'] = outer_audit
        if copy_audit is not None:
            result['gdn_copy_pairs'] = copy_audit
        if output_arm is not None:
            result['gdn_output_grid' if gdn_output_grid else 'gdn_output_l1'] = dict(hits=list(output_arm.hits), restored=not output_arm.active,
                admission=output_admission)
        if fusion_arm is not None:
            result['fused_t16_mlp'] = fusion_arm.audit
        if native_slot_arm is not None:
            result['native_slot_gdn'] = native_slot_arm.summary()
        if score_arm is not None:
            result['score_layout'] = score_arm.summary()
        return result
    finally:
        try:
            score_scope.close()
        finally:
            try:
                if drafter is not None:
                    drafter.close()
            finally:
                if capture is not None:
                    capture.close()
