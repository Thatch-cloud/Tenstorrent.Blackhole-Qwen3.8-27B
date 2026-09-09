"""Five learned layers/final norm in one simulated device trace; replay qualification, not numerical or hardware promotion."""

import argparse
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_attention import validate_mask
from dspark_backbone_mesh import PARAMETERS, execute, flatten, pack_parameter
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import PHASES, REPLAY_CASES, SPECIFICATIONS, WIDE_POLICY
from dspark_layer_diagnostic import COMPOSED_REPORT_SHA256, COMPOSED_OPERANDS_SHA256, error_summary, load_capture
from dspark_layer_mesh import INPUTS
from dspark_projection import TOLERANCE, difference, digest, tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate
from sim_memory_budget import require_clean, snapshot


BACKEND_REPORT = 'dspark-backend-cpu-reference.json'
BACKEND_REPORT_SHA256 = 'efd8bd223ad7adf373efe41fff93488a309efff278e7efcf138f111b00835d37'
BACKEND_OUTPUTS_SHA256 = '94c41406999c929111663c5bc1ffbaf6ed729d95b689457e95e0119f532cf713'
LAYER_REPORT = 'dspark-layer-mesh-simulator.json'
LAYER_REPORT_SHA256 = '1374f6cbd1704a04069d29b8fa0b32db523972298e19699d0b45df2848a8607a'
CHECK_COUNTS = dict(stage_checks=786,layer_zero_checks=156,replay_checks=1048,input_checks=112,
    parameter_checks=224,stale_controls=2,padding_checks=66)
REFERENCE_COUNTS = dict(backend_checks=66,context_reference_checks=6)


def dependency(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAYER = dependency('dspark-layer-mesh-probe')
BACKEND = dependency('dspark-backend-reference')
SOURCES = tuple(sorted(set(LAYER.SOURCES+BACKEND.SOURCES+('dspark-backbone-mesh-probe.py',
    'dspark_backbone_mesh.py','sim_memory_budget.py',BACKEND_REPORT,LAYER_REPORT))))


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def load_backend(path):
    import torch

    report_path = Path(__file__).with_name(BACKEND_REPORT)
    if digest(report_path)!=BACKEND_REPORT_SHA256 or digest(path)!=BACKEND_OUTPUTS_SHA256:
        raise ValueError('Pinned complete backend-matched CPU reference and observed outputs required')
    report = json.loads(report_path.read_text())
    if (report['passed'] is not True or report['closed_cleanly'] is not True or report['stage']!='complete'
            or report['sources']!=report['sources_after'] or report['checkpoint_sha256']!=CHECKPOINT_SHA256
            or report['output_tensors_sha256']!=BACKEND_OUTPUTS_SHA256
            or any(digest(Path(__file__).parent/name)!=sha for name,sha in report['sources'].items())):
        raise ValueError('Complete unchanged backend-matched CPU reference required')
    outputs = torch.load(path,weights_only=True,map_location='cpu')
    stages = {'projected_context','final_norm'}|{f'layer_{layer}_{stage}' for layer in range(5)
        for stage in ('attention_residual','output')}
    for name in ('port_stages','backend_checks'):
        rows = report[name]
        if len(rows)!=48 or {(row['case'],row['stage']) for row in rows}!={(case,stage) for case in range(4) for stage in stages}:
            raise ValueError('All backend-matched CPU stage records required')
        for row in rows:
            value = outputs[row['case']]['stages'][row['stage']]
            if tensor_digest(value)!=row['sha256'] or not torch.isfinite(value).all():
                raise ValueError('Backend-matched CPU stage payload changed')
    return report,outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','capture','operands','checkpoint','backend-outputs'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    link_policy = validate(os.environ)
    outputs_path = options.output.with_suffix('.operands.pt')
    if (options.output.exists() or outputs_path.exists() or os.environ.get('QWEN_SIM_SHARED_BDF')!='1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY')!='1'
            or Path(os.environ['TT_METAL_SIMULATOR']).parent.name!='shared-bdf'
            or any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE','QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh bounded-memory simulator evidence, shared BDF and original runtime required; hardware is prohibited')
    original,control = load_capture(options.capture,options.operands,composed_layer=True)
    backend,cpu = load_backend(options.backend_outputs)
    layer_path = Path(__file__).with_name(LAYER_REPORT)
    if digest(layer_path)!=LAYER_REPORT_SHA256:
        raise ValueError('Pinned independently qualified single-trace learned layer required')
    layer_report = json.loads(layer_path.read_text())
    sources = source_hashes()
    for name,sha in layer_report['sources'].items():
        if name!='../../optimisation/sim/run-dispatch-probe.sh' and sources.get(name)!=sha:
            raise ValueError('Qualified single-trace learned layer source changed: '+name)
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    native = LAYER.MESH.fingerprints(root)
    if native!=layer_report['native_sources'] or native!=layer_report['native_sources_after']:
        raise ValueError('Qualified single-trace native runtime changed')
    report = dict(passed=False,functional_replay_gate_passed=False,cpu_numerical_gate_passed=False,
        retained_layer_numerical_gate_passed=False,closed_cleanly=False,checkpoint_closed=False,
        scope=__doc__,backend='simulator',layers=5,context_rows=32,proposal_rows=7,policy=WIDE_POLICY,
        link_policy=link_policy,tolerance=TOLERANCE,checkpoint_sha256=CHECKPOINT_SHA256,
        sources=sources,native_sources=native,resources_before=snapshot(bounded=True),capture_sha256=COMPOSED_REPORT_SHA256,
        operands_sha256=COMPOSED_OPERANDS_SHA256,layer_report_sha256=LAYER_REPORT_SHA256,
        backend_report_sha256=BACKEND_REPORT_SHA256,backend_outputs_sha256=BACKEND_OUTPUTS_SHA256,
        reference_context='CPU uses frozen FC context; device uses retained learned native FC context',resource_samples=[],
        full_backbone_captured=False,full_pipeline_captured=False,simulated_collectives=True,
        physical_fabric_tested=False,target_integrated=False,eligible_for_hardware=False,
        **{name:[] for name in (*CHECK_COUNTS,*REFERENCE_COUNTS)})
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

    try:
        progress('verify_checkpoint_and_complete_inputs')
        fixtures = control['fixtures']
        for case,fixture in enumerate(fixtures):
            validate_mask(fixture['mask'],32,key_multiple=64)
            if (tensor_digest(fixture['cpu_context'])!=tensor_digest(cpu[case]['stages']['projected_context'])
                    or tuple(fixture['noise'].shape)!=(1,1,32,5120)):
                raise ValueError('Retained fixture and backend-matched CPU context disagree')
        reader = VerifiedWeights(options.checkpoint)
        report['parameter_sha256'] = {name:reader.fingerprints()[name] for name in PARAMETERS}
        if any(sha!=backend['tensor_sha256'][name] for name,sha in report['parameter_sha256'].items()):
            raise ValueError('All 56 parameters must match the complete backend-matched CPU reference')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)

        def upload(value, *, sharded=False, device=True):
            result = ttnn.from_torch(value,dtype=ttnn.float32 if value.dtype==torch.float32 else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0)
                if sharded else ttnn.ReplicateTensorToMesh(mesh),
                **(dict(device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(result)
            return result

        parameters,parameter_bindings,expected_parameter_hashes = {},{},{}

        def audit_parameter(name,phase):
            value = parameters[name]
            if addresses(ttnn,value)!=parameter_bindings[name]:
                raise AssertionError('Learned parameter bindings moved')
            for chip in range(2):
                actual = tensor_digest(host(value,chip))
                expected = expected_parameter_hashes[name,chip]
                exact = actual==expected
                report['parameter_checks'].append(dict(phase=phase,chip=chip,tensor=name,exact=exact,
                    bindings_stable=True,actual_sha256=actual,expected_sha256=expected))
                if not exact:
                    raise AssertionError('Learned parameter bits changed')

        for name in PARAMETERS:
            progress('upload_and_audit_'+name)
            value,sharded = pack_parameter(name,reader.tensor(name))
            for chip in range(2):
                expected_parameter_hashes[name,chip] = tensor_digest(value[chip:chip+1] if sharded else value)
            parameters[name] = upload(value,sharded=sharded)
            parameter_bindings[name] = addresses(ttnn,parameters[name])
            del value
            audit_parameter(name,'before')
            report['resource_samples'].append(dict(parameter=name,phase='before',**snapshot(bounded=True)))
        layer_weights = tuple({name:parameters[f'layers.{layer}.{name}'] for name in SPECIFICATIONS} for layer in range(5))
        values = {case:[fixture['context'],fixture['noise'],*fixture['tables']['q'],*fixture['tables']['k'],
            fixture['mask'],fixture['live']] for case,fixture in enumerate(fixtures)}
        inputs = {name:upload(value) for name,value in zip(INPUTS,values[0],strict=True)}
        host_inputs = {case:[upload(value,device=False) for value in rows] for case,rows in values.items()}
        input_bindings = {name:addresses(ttnn,value) for name,value in inputs.items()}
        all_bindings = [addresses(ttnn,value) for value in owned]
        if any(len({binding[chip] for binding in all_bindings})!=len(all_bindings) for chip in range(2)):
            raise AssertionError('Persistent backbone inputs and learned parameters must be disjoint')

        def update(case):
            for name,value in zip(INPUTS,host_inputs[case],strict=True):
                ttnn.copy_host_to_device_tensor(value,inputs[name])
            ttnn.synchronize_device(mesh)

        def run():
            return execute(ttnn,mesh,collectives,inputs,layer_weights,parameters['norm.weight'],retain,mask_validated=True)

        def audit_inputs(mode,ordinal,case):
            if {name:addresses(ttnn,value) for name,value in inputs.items()}!=input_bindings:
                raise AssertionError('Persistent backbone inputs moved')
            for index,name in enumerate(INPUTS):
                for chip in range(2):
                    exact = tensor_digest(host(inputs[name],chip))==tensor_digest(values[case][index])
                    report['input_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,chip=chip,tensor=name,exact=exact))
                    if not exact:
                        raise AssertionError('Borrowed backbone input changed')

        for case in range(3):
            progress(f'five_layer_eager_{case}')
            update(case)
            result = run()
            ttnn.synchronize_device(mesh)
            observed = {}
            eager[case] = observed
            for (layer,phase,stage),value in flatten(result).items():
                observed[layer,phase,stage] = [host(value,chip) for chip in range(2)]
                for chip,actual in enumerate(observed[layer,phase,stage]):
                    template = control['eager']['finish' if layer==-1 else phase,case]['output' if layer==-1 else stage][chip]
                    valid = actual.shape==template.shape and actual.dtype==template.dtype and bool(torch.isfinite(actual).all())
                    report['stage_checks'].append(dict(case=case,layer=layer,phase=phase,stage=stage,chip=chip,
                        valid=valid,sha256=tensor_digest(actual),shape=list(actual.shape),dtype=str(actual.dtype)))
                    if not valid:
                        raise AssertionError('Every complete learned stage must have valid shape, precision and finite data')
                    if layer==0:
                        report['layer_zero_checks'].append(dict(case=case,phase=phase,stage=stage,chip=chip,
                            **difference(actual,template,exact=True)))
                    if (phase,stage) in (('attention','attention_partial'),('finish','output'),('final','final_norm')):
                        zero = bool(torch.count_nonzero(actual[:,:,7:])==0)
                        report['padding_checks'].append(dict(case=case,layer=layer,phase=phase,chip=chip,zero=zero))
                    if (phase,stage) in (('mlp','attention_residual'),('finish','output'),('final','final_norm')):
                        name = 'final_norm' if layer==-1 else f'layer_{layer}_{stage}'
                        report['backend_checks'].append(dict(case=case,layer=layer,stage=name,chip=chip,
                            **error_summary(actual[:,0,:7],cpu[case]['stages'][name])))
            for chip in range(2):
                report['context_reference_checks'].append(dict(case=case,chip=chip,
                    **error_summary(host(inputs['context'],chip)[:,0],cpu[case]['stages']['projected_context'])))
            audit_inputs('eager',case,case)
            if (any(not entry['passed'] for entry in report['layer_zero_checks'])
                    or any(not entry['zero'] for entry in report['padding_checks'])):
                raise AssertionError('Retained layer-zero bits and all inactive rows must remain unchanged')
            release_owned(ttnn,transient)
            transient.clear()
        progress('capture_complete_five_layer_backbone')
        update(0)
        trace,result = capture_operation(ttnn,mesh,run)
        report['full_backbone_captured'] = True
        output_bindings = {key:addresses(ttnn,value) for key,value in flatten(result).items()}
        for ordinal,case in enumerate(REPLAY_CASES):
            progress(f'five_layer_replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            if {key:addresses(ttnn,value) for key,value in flatten(result).items()}!=output_bindings:
                raise AssertionError('Captured five-layer output bindings moved')
            for (layer,phase,stage),value in flatten(result).items():
                for chip in range(2):
                    actual = host(value,chip)
                    expected = eager[case][layer,phase,stage][chip]
                    report['replay_checks'].append(dict(ordinal=ordinal,case=case,layer=layer,phase=phase,stage=stage,chip=chip,
                        bindings_stable=True,actual_sha256=tensor_digest(actual),expected_sha256=tensor_digest(expected),
                        **difference(actual,expected,exact=True)))
            audit_inputs('replay',ordinal,case)
            if any(not entry['passed'] for entry in report['replay_checks']):
                raise AssertionError('Five-layer changing-input replay differs from its complete eager execution')
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
        for chip in range(2):
            actual = tensor_digest(host(result['final_norm'],chip))
            detected = actual==tensor_digest(eager[0][-1,'final','final_norm'][chip]) \
                and actual!=tensor_digest(eager[2][-1,'final','final_norm'][chip])
            report['stale_controls'].append(dict(chip=chip,detected=detected))
        ttnn.release_trace(mesh,trace)
        trace = None
        release_owned(ttnn,transient)
        transient.clear()
        for name in PARAMETERS:
            progress('audit_after_'+name)
            audit_parameter(name,'after')
            report['resource_samples'].append(dict(parameter=name,phase='after',**snapshot(bounded=True)))
        expected_counts = {**CHECK_COUNTS,**REFERENCE_COUNTS}
        if {name:len(report[name]) for name in expected_counts}!=expected_counts:
            raise AssertionError('All 2394 functional checks and 72 separate numerical diagnostics required')
        if len(report['resource_samples'])!=2*len(PARAMETERS):
            raise AssertionError('Memory-budget evidence for every parameter audit required')
        for sample in report['resource_samples']:
            require_clean(report['resources_before'],sample)
        if any(not entry['detected'] for entry in report['stale_controls']):
            raise AssertionError('Omitted backbone input updates must be detectable')
        report['cpu_numerical_gate_passed'] = all(entry['passed'] for name in REFERENCE_COUNTS for entry in report[name])
        report['functional_replay_gate_passed'] = True
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
            report['sources_after'],report['native_sources_after'] = source_hashes(),LAYER.MESH.fingerprints(root)
            if report['sources_after']!=report['sources'] or report['native_sources_after']!=report['native_sources']:
                raise ValueError('Backbone sources or native runtime changed')
            report['resources_after'] = snapshot(bounded=True)
            require_clean(report['resources_before'],report['resources_after'])
            if eager:
                with outputs_path.open('xb') as stream:
                    torch.save(dict(sources=report['sources'],native_sources=report['native_sources'],policy=WIDE_POLICY,
                        parameter_sha256=report['parameter_sha256'],observed=eager),stream)
                report['observed_operands'] = dict(path=str(outputs_path),sha256=digest(outputs_path))
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['functional_replay_gate_passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
