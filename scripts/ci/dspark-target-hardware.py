"""Real target-bound DSpark proposal integration at CTX32; serial oracle, not a committed-TG benchmark."""

import argparse
import json
import os
from pathlib import Path
import time

from attention_batch import capture_operation
from dspark_attention import full_mask, validate_mask
from dspark_hardware_gate import digest, simulator_preflight, native_fingerprints, require_compatible_native
from dspark_inputs import PROPOSALS, VOCABULARY, query_inputs
from dspark_intake import FILES, TAPS
from dspark_layer import SPECIFICATIONS
from dspark_pipeline import PARAMETERS, pack_parameter
from dspark_projection import tensor_digest
from dspark_rope_tables import DSparkRotary
from dspark_target import propose
from dspark_weights import VerifiedWeights
from dflash_prefill_window import snapshot_prefill_tail
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate
from target_features import LayerOutputCapture


CONTEXT = 32
CASES = (0, 1, 0)
PROMPTS = (
    'Write a Python function that merges two sorted lists of integers without changing either input. Include duplicate values and return a new sorted list.\n```python\ndef merge_sorted(left, right):\n',
    'Write a Rust function that checks whether a string contains balanced parentheses. Ignore all other characters and reject any unmatched closing parenthesis. Include unit tests.\n```rust\nfn balanced(input: &str) -> bool {\n',
)
TARGET_SOURCES = {
    'models/demos/blackhole/qwen36/tt/gdn/tp.py': 'f767d0648ae01b0b1c0bb7bf601f5490661707b845c11ce6dbebb86ba0f84dc9',
    'models/demos/blackhole/qwen36/tt/qwen36_vllm.py': 'cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe',
    'models/tt_transformers/tt/generator.py': '4c2633ba8e5e6b0430550ef99409e9a6f0e0a901b4c6627540c579eb9b7d5a3e',
    'models/demos/blackhole/qwen36/tt/attention/tp.py': 'e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a',
    'models/demos/blackhole/qwen36/tt/model.py': 'c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391',
}
PREREQUISITES = {
    'dspark-noise-simulator.json': '6215f1ee04d8928814ac79b56ae7fe63bd345ebf6aaa83995814c245bb5c1cc0',
    'dspark-vocabulary-simulator.json': 'e95f780502fefcafa5b07c813614450c88609044abf27e606aab11d4b293216c',
    'dspark-markov-native-simulator-learned.json': 'cd53b2cf20f4be1a2cae1b4f52b3162f6e5527fa194795dd53068cae88cd3a5a',
    'dspark-pipeline-hardware.json': '632215b0a7c90420ab53ad06ecabf82e8124ee8262f4d9ca11ccaae4851a7262',
}


def metadata(config):
    import torch

    values = {}
    for name, pair in DSparkRotary(config).block_tables(0, CONTEXT).items():
        rows = 32 if name == 'q' else 64
        for kind, source, fill in zip(('cos', 'sin'), pair, (1., 0.), strict=True):
            value = torch.full((1, 1, rows, 128), fill, dtype=torch.bfloat16)
            value[:, :, :source.shape[2]] = source
            values[name + '_' + kind] = value
    values['mask'] = full_mask(CONTEXT, key_multiple=64)
    validate_mask(values['mask'], CONTEXT, key_multiple=64)
    values['live'] = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
    values['live'][:, :, :PROPOSALS] = 1
    return values


def accepted_prefix(proposals, golden):
    if (len(proposals) != PROPOSALS or len(golden) != PROPOSALS + 1
            or any(type(token) is not int or not 0 <= token < VOCABULARY for token in (*proposals, *golden))):
        raise ValueError('Seven complete global proposals and eight serial target predictions required')
    accepted = 0
    for proposal, expected in zip(proposals, golden):
        if proposal != expected:
            break
        accepted += 1
    return accepted


def preflight(root, weights, config):
    directory = Path(__file__).parent
    if (weights.name != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
            or digest(config) != FILES['config.json'][1]):
        raise ValueError('Pinned target snapshot and DSpark config required')
    target = {name: digest(root / name) for name in TARGET_SOURCES}
    if target != TARGET_SOURCES:
        raise ValueError('Unreviewed target implementation')
    prerequisites = {}
    source_names = {'dspark-target-hardware.py', 'dspark_hardware_gate.py', 'target_features.py',
        'dflash_prefill_window.py', 'model_batch.py', 'attention_batch.py',
        'run-dspark-hardware.sh', 'dspark-hardware-suite.sh'}
    native_reference = {}
    for name, expected in PREREQUISITES.items():
        path = directory / name
        if digest(path) != expected or path.with_suffix('.exit-status').read_text().strip() != '0':
            raise ValueError('Pinned clean component evidence required: ' + name)
        report = json.loads(path.read_text())
        if report.get('passed') is not True or report.get('closed_cleanly') is not True:
            raise ValueError('Component must complete and close cleanly')
        qualified = report.get('sources', {})
        if qualified != report.get('sources_after'):
            raise ValueError('Component sources changed during qualification')
        for source, source_hash in qualified.items():
            if source == '../../optimisation/sim/run-dispatch-probe.sh':
                continue
            if source.endswith('.py') and not source.endswith('-hardware.py') and source not in (
                    'dspark_hardware_gate.py', 'dspark_runtime_cache.py', 'dspark_native_restore.py'):
                if digest(directory / source) != source_hash:
                    raise ValueError('Qualified proposal source changed: ' + source)
            source_names.add(source)
        if name == 'dspark-noise-simulator.json':
            native_reference = report['native_sources']
        prerequisites[name] = expected
    index = json.loads((weights / 'model.safetensors.index.json').read_text())
    for filename in set(index['weight_map'].values()):
        if Path(filename).name != filename or not (weights / filename).is_file():
            raise ValueError('Complete pinned local target checkpoint required')
    metadata(json.loads(config.read_text()))
    return dict(prerequisites=prerequisites, target_sources=target,
        target_index_sha256=digest(weights / 'model.safetensors.index.json'),
        target_config_sha256=digest(weights / 'config.json'),
        sources={name: digest(directory / name) for name in sorted(source_names)},
        native_reference=native_reference, simulator_preflight=simulator_preflight(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--request', action='store_true', help='Run full 4K coding requests with the T16 batched verifier')
    parser.add_argument('--request-variants', action='store_true', help='Compare eager, captured proposal and commit-only GDN in one loaded session')
    parser.add_argument('--native-attention-variants', action='store_true')
    parser.add_argument('--profile-verifier', action='store_true')
    parser.add_argument('--norm-scatter-variants', action='store_true')
    parser.add_argument('--target-attention-variants', action='store_true')
    options = parser.parse_args()
    if options.target_attention_variants and (not options.request or options.request_variants
            or options.native_attention_variants or options.profile_verifier or options.norm_scatter_variants):
        raise ValueError('Target attention requires its own complete request comparison')
    if options.norm_scatter_variants and (not options.request or options.request_variants
            or options.native_attention_variants or options.profile_verifier):
        raise ValueError('Norm scatter requires its own complete request comparison')
    if options.request_variants and not options.request:
        raise ValueError('Proposal variants require the complete request experiment')
    if options.native_attention_variants and (not options.request or options.request_variants):
        raise ValueError('Native attention requires its own complete request comparison')
    if options.profile_verifier and (not options.request or options.request_variants or options.native_attention_variants):
        raise ValueError('Verifier profiling requires its own audited request')
    if (os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE')
            or options.output.exists() or os.environ.get('QWEN_PROJECTION_LINKS') != '4'):
        raise ValueError('Fresh allocated fast-dispatch hardware run with four explicit proposal links required')
    root, weights = Path(os.environ['TT_METAL_HOME']), Path(os.environ['MODEL_WEIGHTS_DIR'])
    gate = preflight(root, weights, options.config)
    if options.request:
        from dspark_request_experiment import request_preflight
        from sampling_link_policy import audit as sampling_link_audit

        request_gate = request_preflight(Path(__file__).parent,
            prepared_proposals=options.request_variants or options.native_attention_variants or options.profile_verifier or options.norm_scatter_variants or options.target_attention_variants)
        if options.native_attention_variants or options.profile_verifier or options.norm_scatter_variants or options.target_attention_variants:
            from dspark_native_fixed_gate import qualify
            request_gate['request_prerequisites'].update(qualify(Path(__file__).parent))
        if options.norm_scatter_variants:
            from gdn_norm_scatter_report import validate as validate_norm
            request_gate['request_prerequisites']['norm_scatter'] = validate_norm(
                json.loads(Path(__file__).with_name('gdn-norm-scatter-simulator.json').read_text()), Path(__file__).parent)
        if options.target_attention_variants:
            from target_t16_attention_gate import qualify as qualify_target
            request_gate['request_prerequisites']['target_t16_attention'] = qualify_target(Path(__file__).parent)
        gate['sources'].update(request_gate['sources'])
        gate['request_prerequisites'] = request_gate['request_prerequisites']
        gate['simulator_metadata_only_sources'] = request_gate['simulator_metadata_only_sources']
        gate['sampling_link_sources'] = sampling_link_audit(root, {**os.environ, 'QWEN_FABRIC_LINK_PROBE': '1'})
    native = native_fingerprints(root, dict(native_sources=gate['native_reference']))
    require_compatible_native(native, gate['native_reference'], require_built_library=not options.preflight)
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
    config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS]
    if any(len(prompt) < CONTEXT for prompt in prompts):
        raise ValueError('Both coding fixtures must supply at least 32 actual tokens')
    prompts = [prompt[:CONTEXT] for prompt in prompts]
    context = None
    if options.request:
        from coding_context_request import make_context_prompt

        prompt, context = make_context_prompt(tokenizer, context_tokens=4096)
        prompts = [prompt]
    if options.preflight:
        options.output.write_text(json.dumps(dict(passed=True, scope='Target imports/config/tokenizer and component gates; no device execution',
            prompts=prompts, **gate), indent=2) + '\n')
        return
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    report = dict(passed=False, closed_cleanly=False, scope=__doc__, ctx_tokens=CONTEXT, drafter_history_rows=CONTEXT,
        proposal_rows=PROPOSALS, streams=1, pp=None, committed_tg=None, eligible_for_serving=False,
        retained_numerical_gate_passed=False, target_verifier='Native serial oracle, not batched verification or publication',
        source_revision=os.environ.get('QWEN_SOURCE_REVISION'), workflow_run=os.environ.get('QWEN_WORKFLOW_RUN'),
        link_policy=validate(os.environ), cases=[], stages=[], native_sources=native, **gate)
    if options.request:
        report.update(scope='Full-history DSpark coding-request screen; one feature audit and two timed requests',
            ctx_tokens=len(prompts[0]), drafter_history_rows=len(prompts[0]), proposal_rows=15,
            target_verifier='Captured T16 batched verifier with exact native token/state and committed-feature checks')
        if options.request_variants:
            report['scope'] = 'Matched full-history DSpark proposal/commit-only screen; three audits and six timed requests'
        if options.native_attention_variants:
            report['scope'] = 'Matched composed/native DSpark attention screen; two audits and four timed requests'
        if options.profile_verifier:
            report['scope'] = 'Audited native-attention DSpark T16 verifier attribution; no throughput measurement'
        if options.norm_scatter_variants:
            report['scope'] = 'Matched native-attention DSpark norm reader screen; two audits and four timed requests'
        if options.target_attention_variants:
            report['scope'] = 'Matched native versus folded T16 target attention; two audits and four timed requests'
    owned, transient, captured_owned = [], [], []
    mesh = reader = trace = capture = None
    started = time.perf_counter()

    def progress(stage):
        report['stage'] = stage
        report['stages'].append(dict(stage=stage, elapsed_seconds=time.perf_counter() - started))
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['stages'][-1]), flush=True)

    try:
        progress('verify_checkpoint')
        reader = VerifiedWeights(options.checkpoint)
        report['parameter_sha256'] = reader.fingerprints()
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        progress('load_target_once')
        generator = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=8, max_seq_len=65536)
        model = generator.model[0]
        if len(model.layers) != 64 or model.vocab_size != VOCABULARY or model._lmhead_vocab_sharded is not True:
            raise ValueError('Complete TP2 target with borrowed vocabulary-sharded head required')
        kv_cache = generator.allocate_kv_cache((1032, model.args.n_local_kv_heads, 64, model.args.head_dim), ttnn.bfloat16, 64)
        pages = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)
        recurrent = [value for layer in model.layers if not layer.is_full_attention
            for value in (layer.attention.rec_state, *layer.attention.conv_states)]
        caches = [value for pair in model._paged_kv_caches for value in pair]
        if len(recurrent) != 240 or len(caches) != 32:
            raise ValueError('All recurrent slots and 16 attention KV pairs required')

        def host(value, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        def hashes(value):
            return [tensor_digest(host(value, chip)) for chip in range(2)]

        bindings = [addresses(ttnn, value) for value in (*recurrent, *caches, model.lm_head_weight, model.embd.weights)]

        def state(valid):
            if [addresses(ttnn, value) for value in (*recurrent, *caches, model.lm_head_weight, model.embd.weights)] != bindings:
                raise AssertionError('Borrowed target state/weight addresses changed')
            result = dict(recurrent=[hashes(value) for value in recurrent], kv=[])
            for value in caches:
                page = ttnn.slice(value, (0, 0, 0, 0), (1, value.shape[1], 64, value.shape[3]))
                try:
                    result['kv'].append([tensor_digest(host(page, chip)[:, :, :valid]) for chip in range(2)])
                finally:
                    if addresses(ttnn, page) != addresses(ttnn, value):
                        ttnn.deallocate(page)
            return result

        def upload(value, *, sharded=False, integers=False, row_major=False, device=True):
            result = ttnn.from_torch(value, dtype=ttnn.uint32 if integers else
                ttnn.float32 if value.dtype == torch.float32 else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integers or row_major else ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh),
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(result)
            return result

        parameters, parameter_hashes = {}, {}
        for name in PARAMETERS:
            progress('upload_' + name)
            value, sharded = pack_parameter(name, reader.tensor(name))
            parameters[name] = upload(value, sharded=sharded)
            parameter_hashes[name] = [tensor_digest(value[chip:chip + 1] if sharded else value) for chip in range(2)]
            del value
        predecessor_value = reader.tensor('markov_head.markov_w1.weight').reshape(1, 1, VOCABULARY, 256)
        predecessor = upload(predecessor_value, row_major=True)
        successor_value = reader.tensor('markov_head.markov_w2.weight').T.contiguous().reshape(1, 1, 256, VOCABULARY)
        successor = upload(successor_value)
        parameter_hashes['predecessor'] = [tensor_digest(predecessor_value)] * 2
        parameter_hashes['successor'] = [tensor_digest(successor_value)] * 2
        del predecessor_value, successor_value
        all_parameters = dict(parameters, predecessor=predecessor, successor=successor)
        report['device_parameter_checks'] = []

        def check_parameters(phase):
            for name, value in all_parameters.items():
                actual = hashes(value)
                report['device_parameter_checks'].append(dict(name=name, phase=phase, actual=actual,
                    expected=parameter_hashes[name], exact=actual == parameter_hashes[name]))
                if actual != parameter_hashes[name]:
                    raise AssertionError('Learned parameter changed during target integration')

        progress('audit_uploaded_parameters')
        check_parameters('before')
        layers = tuple({name:parameters[f'layers.{layer}.{name}'] for name in SPECIFICATIONS} for layer in range(5))
        if options.request:
            from dspark_request_experiment import run_loaded_requests

            run_loaded_requests(ttnn, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
                layers, predecessor, successor, DSparkRotary(json.loads(options.config.read_text())), report, progress,
                prompt=prompts[0], context=context, variants=options.request_variants,
                native_attention_variants=options.native_attention_variants, profile_verifier=options.profile_verifier,
                norm_scatter_variants=options.norm_scatter_variants,
                target_attention_variants=options.target_attention_variants)
            progress('audit_parameters_after_full_requests')
            check_parameters('after')
            report['passed'] = True
            return
        inputs = {name:upload(value) for name, value in metadata(json.loads(options.config.read_text())).items()}
        inputs.update({'feature_' + str(tap):upload(torch.zeros(2, 1, 32, 2560, dtype=torch.bfloat16), sharded=True) for tap in TAPS})
        identifiers = upload(query_inputs(0, CONTEXT)['identifiers'], integers=True)
        anchor = upload(torch.zeros(1, 1, 1, 1, dtype=torch.int64), integers=True)

        def prefill(prompt, *, with_features=False):
            nonlocal capture
            if capture is not None:
                capture.close()
                capture = None
            if with_features:
                capture = LayerOutputCapture(model, TAPS,
                    snapshot=lambda value:snapshot_prefill_tail(ttnn, value, CONTEXT),
                    release=ttnn.deallocate, storage_ids=lambda value:addresses(ttnn, value))
            generator.prev_page_table = None

            def forward():
                logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), pages,
                    kv_cache, [CONTEXT], empty_slots=[0], enable_trace=False)
                return int(logits.reshape(-1, VOCABULARY)[0].float().argmax())

            if capture is None:
                return forward()
            with capture.capture():
                return forward()

        def oracle(seed):
            output = []
            token = seed
            for position in range(CONTEXT, CONTEXT + PROPOSALS + 1):
                result = generator.decode_forward(tokens=torch.tensor([[token]], dtype=torch.int32),
                    start_pos=torch.tensor([position], dtype=torch.int32), page_table=pages,
                    kv_cache=kv_cache, enable_trace=False, read_from_device=True)
                logits = result[0] if isinstance(result, tuple) else result
                token = int(logits.reshape(-1, VOCABULARY)[0].float().argmax())
                output.append(token)
            return output

        def run(destination):
            return propose(ttnn, model, mesh, collectives, identifiers, anchor, inputs, parameters,
                layers, predecessor, successor, destination, inputs_validated=True)

        def observe(result):
            values = dict(noise=result['noise'], final_norm=result['learned']['backbone']['final_norm'],
                base_logits=result['logits']['base_logits'])
            values.update({'scores_' + str(index):record['scores'] for index, record in enumerate(result['records'])})
            tokens = [[int(host(record['token'], chip).reshape(-1)[0]) for record in result['records']] for chip in range(2)]
            observed = {name:hashes(value) for name, value in values.items()}
            if tokens[0] != tokens[1] or any(value[0] != value[1] for value in observed.values()):
                raise AssertionError('Complete global proposal replicas differ')
            return dict(tokens=tokens[0], hashes=observed)

        prepared = []
        for case, prompt in enumerate(prompts):
            progress(f'case_{case}_native_oracle')
            seed = prefill(prompt)
            golden = oracle(seed)
            golden_state = state(CONTEXT + PROPOSALS + 1)
            progress(f'case_{case}_capture_actual_target_features')
            if prefill(prompt, with_features=True) != seed:
                raise AssertionError('Actual feature capture changes native prefill seed')
            features = capture.outputs()
            feature_hashes = [hashes(value) for value in features]
            copies = [ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in features]
            owned.extend(copies)
            if [hashes(value) for value in copies] != feature_hashes:
                raise AssertionError('Copied target features differ from actual post-layer captures')
            prepared.append(dict(seed=seed, golden=golden, golden_state=golden_state,
                features=copies, feature_hashes=feature_hashes,
                identifiers=upload(query_inputs(seed, CONTEXT)['identifiers'], integers=True, device=False),
                anchor=upload(torch.tensor([[[[seed]]]], dtype=torch.int64), integers=True, device=False)))
            capture.close()
            capture = None

        def update(case):
            current = prepared[case]
            for tap, value in zip(TAPS, current['features'], strict=True):
                ttnn.copy(value, inputs['feature_' + str(tap)])
            for name, destination in (('identifiers', identifiers), ('anchor', anchor)):
                ttnn.copy_host_to_device_tensor(current[name], destination)
            ttnn.synchronize_device(mesh)

        if prefill(prompts[0]) != prepared[0]['seed']:
            raise AssertionError('Native target cannot restore the integration frontier')
        before = state(CONTEXT)
        for ordinal, case in enumerate(CASES):
            current = prepared[case]
            update(case)
            entry = dict(ordinal=ordinal, case=case, prompt=prompts[case], prompt_is_first_32_tokens=True,
                seed=current['seed'], golden=current['golden'], feature_hashes=current['feature_hashes'])
            report['cases'].append(entry)
            entry['input_hashes'] = {name:hashes(value) for name, value in dict(inputs, identifiers=identifiers, anchor=anchor).items()}
            progress(f'case_{ordinal}_complete_eager_proposal')
            eager = observe(run(transient))
            release_owned(ttnn, transient)
            transient.clear()
            entry.update(proposal=eager, accepted=accepted_prefix(eager['tokens'], current['golden']))
        update(0)
        progress('capture_complete_proposal_after_all_allocations')
        trace, traced_result = capture_operation(ttnn, mesh, lambda:run(captured_owned))
        for ordinal, case in enumerate(CASES):
            entry = report['cases'][ordinal]
            update(case)
            progress(f'case_{ordinal}_complete_trace_proposal')
            ttnn.execute_trace(mesh, trace, blocking=True)
            replay = observe(traced_result)
            if replay != entry['proposal']:
                raise AssertionError('Changing-input whole-proposal replay differs from eager execution')
            if {name:hashes(value) for name, value in dict(inputs, identifiers=identifiers, anchor=anchor).items()} != entry['input_hashes']:
                raise AssertionError('Proposal mutates its persistent inputs')
            entry['replay_exact'] = True
        ttnn.release_trace(mesh, trace)
        trace = None
        release_owned(ttnn, captured_owned)
        captured_owned.clear()
        if state(CONTEXT) != before:
            raise AssertionError('Complete eager/captured proposals mutate target state')
        report['target_state_before'] = before
        report['target_state_unchanged'] = True
        report['oracle_checks'] = []
        for case, current in enumerate(prepared):
            if [hashes(value) for value in current['features']] != current['feature_hashes']:
                raise AssertionError('Proposal changes captured actual target features')
            progress(f'case_{case}_post_proposal_oracle')
            if case and prefill(prompts[case]) != current['seed']:
                raise AssertionError('Proposal changes target prefill prediction')
            actual = oracle(current['seed'])
            actual_state = state(CONTEXT + PROPOSALS + 1)
            if actual != current['golden'] or actual_state != current['golden_state']:
                raise AssertionError('Proposal changes target tokens, recurrent slots or valid KV prefix')
            report['oracle_checks'].append(dict(case=case, exact=True, tokens=actual, state=actual_state))
        if report['cases'][0]['proposal'] != report['cases'][2]['proposal']:
            raise AssertionError('Restored request must reproduce the original complete proposal')
        if report['cases'][0]['feature_hashes'] == report['cases'][1]['feature_hashes']:
            raise AssertionError('Changing-input control did not change actual target features')
        if report['cases'][0]['proposal']['hashes'] == report['cases'][1]['proposal']['hashes']:
            raise AssertionError('Whole proposal ignored changed actual target inputs')
        progress('audit_parameters_after_target_integration')
        check_parameters('after')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if hasattr(error, 'evidence'):
            report['failure_evidence'] = error.evidence
        raise
    finally:
        try:
            if capture is not None:
                capture.close()
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, captured_owned)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            if reader is not None:
                reader.__exit__(None, None, None)
            report['checkpoint_closed'] = reader is not None and reader.source is None
            report['sources_after'] = {name:digest(Path(__file__).parent / name) for name in gate['sources']}
            report['native_sources_after'] = native_fingerprints(root, dict(native_sources=gate['native_reference']))
            if report['sources_after'] != gate['sources'] or report['native_sources_after'] != native:
                raise ValueError('Source or native runtime changed during target integration')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
