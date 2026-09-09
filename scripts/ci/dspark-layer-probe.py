"""Complete learned layer-zero simulation from observed FC output; three traces and explicit host-staged TP handoffs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_attention import validate_mask
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import EXACT, PHASES, POLICY, REPLAY_CASES, SPECIFICATIONS, attention_partial, finish, mlp_partial, pack_weight
from dspark_layer_reference import CASES, PROJECTION_OPERANDS_SHA256, PROJECTION_REPORT_SHA256
from dspark_layer_reference import attention_reference, cpu_layer, finish_reference, load_fixtures, mlp_reference
from dspark_projection import TOLERANCE, difference, digest, tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from native_draft_sdpa import KERNEL_DIRECTORY, SOURCE_HASHES, audit_active_kernel, patched_sources


SOURCES = ('dspark-layer-probe.py','dspark_layer.py','dspark_layer_reference.py','dspark_layer_gate.py',
    'dspark_projection.py','dspark-projection-probe.py','dspark_norm_precision.py','dspark_projection_gate.py',
    'dspark_attention.py','draft_attention.py','dspark_rotary_device.py','dspark_rope_tables.py','dspark_residual.py',
    'dspark_backbone_reference.py','dspark-backbone-cpu.py','dspark_intake.py','dspark_checkpoint.py',
    'dspark_weights.py','dspark_markov_fixture.py','dspark_markov_gate.py','feature_projection.py',
    'attention_batch.py','gdn_multitoken_conv.py','native_draft_sdpa.py',
    'dspark-backbone-cpu-reference.json','dspark-backbone-upstream-reference.json',
    '../../optimisation/sim/run-dispatch-probe.sh')


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def fingerprints(root, *, active):
    if type(active) is not bool:
        raise ValueError('Explicit active or restored native runtime required')
    spec = importlib.util.spec_from_file_location('dspark_layer_native_base',Path(__file__).with_name('dspark-projection-probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.fingerprints(root,packer_compat=active,composed_norm=True)
    for name in ('ttnn/cpp/ttnn/operations/transformer/sdpa','tt_metal/tt-llk/tt_llk_blackhole'):
        directory = root/name
        if not directory.is_dir():
            raise ValueError('Complete native SDPA and Blackhole LLK sources required')
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.cpp','.hpp','.h'):
                result[str(path.relative_to(root))] = digest(path)
    if active:
        audit_active_kernel(root)
    else:
        directory = root/KERNEL_DIRECTORY
        if (directory/'.qwen-precise-draft.lock').exists():
            raise ValueError('Restore and release the native SDPA graft before independent qualification')
        result[module.PACKER] = module.COMPAT_PACKER
        for name,data in patched_sources({name:(directory/name).read_bytes() for name in SOURCE_HASHES}).items():
            import hashlib

            result[KERNEL_DIRECTORY+'/'+name] = hashlib.sha256(data).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','checkpoint','projection-report','projection-operands','cpu-outputs','config'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--eager-only',action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    if (os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') != '1' or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'):
        raise ValueError('Owned simulator precise-attention/packer compatibility required; hardware is prohibited')
    outputs_path = options.output.with_suffix('.operands.pt')
    if outputs_path.exists():
        raise ValueError('Refusing to overwrite observed layer outputs')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False,closed_cleanly=False,checkpoint_closed=False,backend='simulator',scope=__doc__,
        mode='eager_diagnostic' if options.eager_only else 'matrix',layer=0,context_rows=32,proposal_rows=7,
        checkpoint_sha256=CHECKPOINT_SHA256,
        cases=[list(value) for value in CASES],policy=POLICY,tolerance=TOLERANCE,precise_native=True,packer_compat=True,
        projection_report_sha256=PROJECTION_REPORT_SHA256,projection_operands_sha256=PROJECTION_OPERANDS_SHA256,
        sources=source_hashes(),native_sources=fingerprints(root,active=True),
        fabric_tested=False,full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        cpu_checks=[],eager_checks=[],replay_checks=[],input_checks=[],parameter_checks=[],stale_controls=[],padding_checks=[])
    mesh = trace = reader = None
    owned,transient,eager = [],[],{}

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    def retain(value):
        transient.append(value)
        return value

    def host(value,chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

    def check(phase,case,chip,stage,actual,expected):
        report['eager_checks'].append(dict(phase=phase,case=case,chip=chip,stage=stage,
            **difference(actual,expected,exact=(phase,stage) in EXACT)))

    try:
        progress('frozen_projection_and_complete_cpu_layer')
        fixtures,report['reference'] = load_fixtures(projection_report=options.projection_report,
            projection_operands=options.projection_operands,cpu_outputs=options.cpu_outputs,config=options.config)
        for fixture in fixtures:
            validate_mask(fixture['mask'],32,key_multiple=64)
        reader = VerifiedWeights(options.checkpoint)
        weights = {name:reader.tensor('layers.0.'+name) for name in SPECIFICATIONS}
        report['parameter_sha256'] = {name:reader.fingerprints()['layers.0.'+name] for name in SPECIFICATIONS}
        if any(checksum != report['reference']['tensor_sha256']['layers.0.'+name] for name,checksum in report['parameter_sha256'].items()):
            raise ValueError('Complete layer parameters differ from frozen CPU backbone')
        own_reference = {}
        for fixture in fixtures:
            case = fixture['case']
            noise = fixture['noise'][:,0,:7]
            upstream = cpu_layer(fixture['cpu_context'],noise,weights,fixture['tables'])
            for stage,value in upstream.items():
                if tensor_digest(value) != tensor_digest(fixture['cpu_'+stage]):
                    raise ValueError('Full CPU layer differs from the upstream-checked frozen reference')
                report['cpu_checks'].append(dict(case=case,stage=stage,exact=True))
            own_reference[case] = cpu_layer(fixture['context'][:,0],noise,weights,fixture['tables'])
        progress('complete_layer_cpu_reference_verified')
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value, *, sharded=False, device=True):
            dtype = ttnn.float32 if value.dtype == torch.float32 else ttnn.bfloat16
            mapper = ttnn.ShardTensorToMesh(mesh,dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh)
            result = ttnn.from_torch(value,dtype=dtype,layout=ttnn.TILE_LAYOUT,mesh_mapper=mapper,
                **(dict(device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(result)
            return result

        packed = {name:pack_weight(name,value) for name,value in weights.items()}
        device_weights = {name:upload(value,sharded=sharded) for name,(value,sharded) in packed.items()}
        parameter_bindings = {name:addresses(ttnn,value) for name,value in device_weights.items()}

        def audit_parameters(phase):
            if {name:addresses(ttnn,value) for name,value in device_weights.items()} != parameter_bindings:
                raise AssertionError('Learned parameter bindings changed')
            for name,value in device_weights.items():
                packed_value,sharded = packed[name]
                for chip in range(2):
                    expected = packed_value[chip:chip+1] if sharded else packed_value
                    if tensor_digest(host(value,chip)) != tensor_digest(expected):
                        raise AssertionError('Learned parameter bits changed')
                    report['parameter_checks'].append(dict(phase=phase,chip=chip,tensor=name,exact=True,bindings_stable=True))

        progress('audit_all_eleven_parameters_before')
        audit_parameters('before')
        for phase,stages in PHASES.items():
            if phase == 'attention':
                names = ('context','noise','q_cos','q_sin','k_cos','k_sin','mask','live')
                values = {fixture['case']:[fixture['context'],fixture['noise'],*fixture['tables']['q'],
                    *fixture['tables']['k'],fixture['mask'],fixture['live']] for fixture in fixtures}
            elif phase == 'mlp':
                names = ('first','second','noise')
                values = {case:[*eager['attention',case]['attention_partial'],fixtures[case]['noise']] for case in range(3)}
            else:
                names = ('first','second','residual')
                values = {case:[*eager['mlp',case]['down_partial'],eager['mlp',case]['attention_residual'][0]] for case in range(3)}
                for case in range(3):
                    residuals = eager['mlp',case]['attention_residual']
                    if tensor_digest(residuals[0]) != tensor_digest(residuals[1]):
                        raise AssertionError('Both TP branches must share the exact residual')
            inputs = {name:upload(value) for name,value in zip(names,values[0],strict=True)}
            payloads = {case:[upload(value,device=False) for value in case_values] for case,case_values in values.items()}
            bindings = {name:addresses(ttnn,value) for name,value in inputs.items()}
            all_bindings = [addresses(ttnn,value) for value in owned]
            if any(len({binding[chip] for binding in all_bindings}) != len(all_bindings) for chip in range(2)):
                raise AssertionError('All persistent inputs and parameters must be disjoint')

            def update(case):
                for name,value in zip(names,payloads[case],strict=True):
                    ttnn.copy_host_to_device_tensor(value,inputs[name])
                ttnn.synchronize_device(mesh)

            def run():
                if phase == 'attention':
                    return attention_partial(ttnn,inputs['context'],inputs['noise'],device_weights,
                        {name:(inputs[name+'_cos'],inputs[name+'_sin']) for name in ('q','k')},
                        inputs['mask'],inputs['live'],retain,mask_validated=True)
                if phase == 'mlp':
                    return mlp_partial(ttnn,inputs['first'],inputs['second'],inputs['noise'],device_weights,retain)
                return finish(ttnn,inputs['first'],inputs['second'],inputs['residual'],retain)

            def audit_inputs(mode,ordinal,case):
                if {name:addresses(ttnn,value) for name,value in inputs.items()} != bindings:
                    raise AssertionError('Persistent layer input bindings changed')
                for index,name in enumerate(names):
                    for chip in range(2):
                        if tensor_digest(host(inputs[name],chip)) != tensor_digest(values[case][index]):
                            raise AssertionError('Borrowed layer input bits changed')
                        report['input_checks'].append(dict(phase=phase,mode=mode,ordinal=ordinal,case=case,chip=chip,tensor=index,exact=True))

            for case in range(3):
                progress(f'{phase}_eager_{case}')
                update(case)
                result = run()
                if set(result) != set(stages):
                    raise AssertionError('Complete layer stage inventory required')
                ttnn.synchronize_device(mesh)
                observed = {stage:[host(result[stage],chip) for chip in range(2)] for stage in stages}
                eager[phase,case] = observed
                for chip in range(2):
                    if phase == 'attention':
                        expected = attention_reference(fixtures[case],weights,chip)
                    elif phase == 'mlp':
                        expected = mlp_reference(values[case][:2],fixtures[case]['noise'],weights,chip)
                    else:
                        expected = finish_reference(values[case][:2],values[case][2])
                    for stage in stages:
                        check(phase,case,chip,stage,observed[stage][chip],expected[stage])
                    if phase == 'mlp':
                        actual = observed['attention_residual'][chip][:,0,:7]
                        check(phase,case,chip,'upstream_attention_residual',actual,fixtures[case]['cpu_attention_residual'])
                        check(phase,case,chip,'own_context_attention_residual',actual,own_reference[case]['attention_residual'])
                        check(phase,case,chip,'native_activation',observed['activation'][chip],
                            torch.nn.functional.silu(observed['gate'][chip].float()).bfloat16())
                        check(phase,case,chip,'native_product',observed['product'][chip],observed['activation'][chip]*observed['up'][chip])
                    if phase == 'finish':
                        actual = observed['output'][chip][:,0,:7]
                        check(phase,case,chip,'upstream_output',actual,fixtures[case]['cpu_output'])
                        check(phase,case,chip,'own_context_output',actual,own_reference[case]['output'])
                    if phase in ('attention','finish'):
                        padding = observed[stages[-1]][chip][:,:,7:]
                        report['padding_checks'].append(dict(phase=phase,case=case,chip=chip,zero=bool(torch.count_nonzero(padding)==0)))
                audit_inputs('eager',case,case)
                release_owned(ttnn,transient)
                transient.clear()
            if options.eager_only:
                continue
            progress(f'{phase}_capture')
            update(0)
            trace,result = capture_operation(ttnn,mesh,run)
            output_bindings = {stage:addresses(ttnn,value) for stage,value in result.items()}
            for ordinal,case in enumerate(REPLAY_CASES):
                progress(f'{phase}_replay_{ordinal}')
                update(case)
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                if {stage:addresses(ttnn,value) for stage,value in result.items()} != output_bindings:
                    raise AssertionError('Captured layer output bindings changed')
                for stage in stages:
                    for chip in range(2):
                        if tensor_digest(host(result[stage],chip)) != tensor_digest(eager[phase,case][stage][chip]):
                            raise AssertionError('Layer replay differs from complete eager outputs')
                        report['replay_checks'].append(dict(phase=phase,ordinal=ordinal,case=case,chip=chip,stage=stage,
                            exact=True,bindings_stable=True))
                audit_inputs('replay',ordinal,case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            for chip in range(2):
                actual = tensor_digest(host(result[stages[-1]],chip))
                if actual != tensor_digest(eager[phase,0][stages[-1]][chip]) or actual == tensor_digest(eager[phase,2][stages[-1]][chip]):
                    raise AssertionError('Omitted layer input update is not detectable')
                report['stale_controls'].append(dict(phase=phase,chip=chip,detected=True))
            ttnn.release_trace(mesh,trace)
            trace = None
            release_owned(ttnn,transient)
            transient.clear()
        progress('audit_all_eleven_parameters_after')
        audit_parameters('after')
        with outputs_path.open('xb') as stream:
            torch.save(dict(sources=report['sources'],native_sources=report['native_sources'],policy=POLICY,
                fixtures=fixtures,eager=eager,parameter_sha256=report['parameter_sha256']),stream)
        report['operands'] = dict(path=str(outputs_path),sha256=digest(outputs_path))
        failed = sum(not entry['passed'] for entry in report['eager_checks'])
        if failed or any(not entry['zero'] for entry in report['padding_checks']):
            raise AssertionError(f'{failed} learned layer comparisons fail the unchanged numerical gate')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh,trace)
                release_owned(ttnn,transient)
                release_owned(ttnn,owned)
                ttnn.close_mesh_device(mesh)
            if reader is not None:
                reader.__exit__(None,None,None)
            report['checkpoint_closed'] = reader is not None and reader.source is None
            report['sources_after'],report['native_sources_after'] = source_hashes(),fingerprints(root,active=True)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Layer source or native runtime changed during execution')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
