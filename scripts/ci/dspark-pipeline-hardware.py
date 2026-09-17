"""Hardware integration of learned FC, all five layers and final norm; synthetic inputs, not committed TG."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

from attention_batch import capture_operation
from dspark_attention import full_mask, validate_mask
from dspark_backend_reference import FP32Backbone
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_hardware_gate import digest, simulator_preflight, native_fingerprints, require_compatible_native
from dspark_intake import FILES, TAPS
from dspark_layer import SPECIFICATIONS, WIDE_POLICY
from dspark_layer_diagnostic import error_summary
from dspark_pipeline import INPUTS, PARAMETERS, execute, flatten, pack_parameter
from dspark_projection import tensor_digest
from dspark_rope_tables import DSparkRotary
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate


CASES = ((0,0),(0,8190),(1,8190))
REPLAYS = (0,2,1,0)
COUNTS = dict(stage_checks=822,replay_checks=1096,input_checks=168,parameter_checks=232,
    padding_checks=66,replica_checks=6,stale_controls=2,cpu_diagnostics=72,timing_checks=6)


def dependency(name):
    spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(config):
    import torch

    generator = dependency('dspark-backbone-cpu')
    reference = json.loads(Path(__file__).with_name('dspark-backbone-cpu-reference.json').read_text())
    rotary = DSparkRotary(config)
    result = []
    for case,(pattern,start) in enumerate(CASES):
        features,noise = generator.inputs(pattern)
        if [tensor_digest(value) for value in (*features.values(),noise)]!=reference['cases'][case]['input_sha256']:
            raise ValueError('Pinned complete backbone input fixtures changed')
        values = {'feature_'+str(tap):torch.stack(features[tap].chunk(2,dim=-1)) for tap in TAPS}
        values['noise'] = torch.nn.functional.pad(noise.unsqueeze(1),(0,0,0,25))
        for name,pair in rotary.block_tables(start,32).items():
            rows = 32 if name=='q' else 64
            for kind,source,fill in zip(('cos','sin'),pair,(1.,0.),strict=True):
                value = torch.full((1,1,rows,128),fill,dtype=torch.bfloat16)
                value[:,:,:source.shape[2]] = source
                values[name+'_'+kind] = value
        values['mask'] = full_mask(32,key_multiple=64)
        validate_mask(values['mask'],32,key_multiple=64)
        values['live'] = torch.zeros(1,1,32,1,dtype=torch.float32)
        values['live'][:,:,:7] = 1
        if set(values)!=set(INPUTS):
            raise ValueError('Complete feature, noise, rotary, mask and padding inputs required')
        result.append(dict(features=features,noise=noise,start=start,values=values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ,True)
    link_policy = validate(os.environ)
    if link_policy['requested_links']!=4 or options.output.exists():
        raise ValueError('Fresh hardware evidence and explicit four-link drafter required')
    if digest(options.config)!=FILES['config.json'][1]:
        raise ValueError('Pinned DSpark configuration required')
    gate = simulator_preflight(Path(__file__).parent)
    source_names = set(gate['arithmetic'])|{
        'dspark_pipeline.py','dspark_backbone_mesh.py','dspark-pipeline-hardware.py','dspark_hardware_gate.py',
        'dspark_backend_reference.py','dspark_backend_attribution.py','dspark_upstream_backbone.py',
        'dspark_rope_reference.py','dspark-hardware-fixtures.py','run-dspark-hardware.sh','dspark-hardware-suite.sh'}
    source_names.update(('dspark_runtime_cache.py','dspark_native_restore.py','ccl-links-build.sh','sdpa_graft_build.py','lazy_ccl_links.py'))

    def source_hashes():
        return {name:digest(Path(__file__).parent/name) for name in sorted(source_names)}

    layer_report = json.loads(Path(__file__).with_name('dspark-layer-mesh-simulator.json').read_text())
    root = Path(os.environ['TT_METAL_HOME'])
    native = native_fingerprints(root,layer_report)
    require_compatible_native(native,layer_report['native_sources'],require_built_library=not options.preflight)
    if options.preflight:
        import sys

        cases = fixtures(json.loads(options.config.read_text()))
        options.output.write_text(json.dumps(dict(passed=True,scope='Python/config/source/input compatibility; no device execution',
            python=sys.version,cases=len(cases),sources=source_hashes(),native_sources=native,simulator_preflight=gate,
            built_library_validation='deferred until build/cache restore; mandatory before device execution'),indent=2)+'\n')
        return
    report = dict(passed=False,closed_cleanly=False,checkpoint_closed=False,scope=__doc__,backend='hardware',
        context_rows=32,proposal_rows=7,layers=5,learned_parameters=58,policy=WIDE_POLICY,
        target_integrated=False,full_pipeline_captured=False,backbone_with_projection_captured=False,
        eligible_for_serving=False,committed_tg=None,pp=None,simulator_preflight=gate,link_policy=link_policy,
        checkpoint_sha256=CHECKPOINT_SHA256,sources=source_hashes(),native_sources=native,
        source_revision=os.environ.get('QWEN_SOURCE_REVISION'),workflow_run=os.environ.get('QWEN_WORKFLOW_RUN'),
        native_differs_from_simulator={name:dict(simulator=sha,hardware=native.get(name))
            for name,sha in layer_report['native_sources'].items() if not name.startswith('simulator/') and native.get(name)!=sha},
        cpu_numerical_gate_passed=False,retained_layer_numerical_gate_passed=False,
        elapsed_stages=[],timing_samples=[],**{name:[] for name in COUNTS})
    started = time.perf_counter()
    mesh = trace = reader = None
    owned,transient = [],[]

    def progress(stage):
        report['stage'] = stage
        report['elapsed_stages'].append(dict(stage=stage,elapsed_seconds=time.perf_counter()-started))
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['elapsed_stages'][-1]),flush=True)

    def retain(value):
        transient.append(value)
        return value

    try:
        import torch
        import ttnn
        from models.tt_transformers.tt.ccl import TT_CCL

        progress('checkpoint_and_inputs')
        reader = VerifiedWeights(options.checkpoint)
        report['parameter_sha256'] = reader.fingerprints()
        config = json.loads(options.config.read_text())
        cases = fixtures(config)
        cpu = []
        for case,fixture in enumerate(cases):
            progress(f'cpu_reference_{case}')
            stages = {}
            FP32Backbone(reader,config).forward(fixture['features'],fixture['noise'],context_start=fixture['start'],
                inspect=lambda name,value:stages.__setitem__(name,value))
            cpu.append(stages)
        progress('open_mesh')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        report['grid'] = dict(x=mesh.compute_with_storage_grid_size().x,y=mesh.compute_with_storage_grid_size().y)

        def host(value,chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        def upload(value,sharded=False,device=True):
            result = ttnn.from_torch(value,dtype=ttnn.float32 if value.dtype==torch.float32 else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0)
                if sharded else ttnn.ReplicateTensorToMesh(mesh),
                **(dict(device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(result)
            return result

        parameters,parameter_hashes,parameter_bindings = {},{},{}

        def audit_parameter(name,phase):
            if addresses(ttnn,parameters[name])!=parameter_bindings[name]:
                raise AssertionError('Learned parameter binding moved')
            for chip in range(2):
                actual = tensor_digest(host(parameters[name],chip))
                expected = parameter_hashes[name,chip]
                report['parameter_checks'].append(dict(name=name,phase=phase,chip=chip,
                    exact=actual==expected,actual_sha256=actual,expected_sha256=expected))
                if actual!=expected:
                    raise AssertionError('Learned parameter bits changed')

        for name in PARAMETERS:
            progress('upload_'+name)
            value,sharded = pack_parameter(name,reader.tensor(name))
            for chip in range(2):
                parameter_hashes[name,chip] = tensor_digest(value[chip:chip+1] if sharded else value)
            parameters[name] = upload(value,sharded)
            parameter_bindings[name] = addresses(ttnn,parameters[name])
            del value
            audit_parameter(name,'before')
        layers = tuple({name:parameters[f'layers.{layer}.{name}'] for name in SPECIFICATIONS} for layer in range(5))
        inputs = {name:upload(cases[0]['values'][name],name.startswith('feature_')) for name in INPUTS}
        host_inputs = [{name:upload(fixture['values'][name],name.startswith('feature_'),False) for name in INPUTS} for fixture in cases]
        input_bindings = {name:addresses(ttnn,value) for name,value in inputs.items()}
        bindings = [addresses(ttnn,value) for value in owned]
        if any(len({value[chip] for value in bindings})!=len(bindings) for chip in range(2)):
            raise AssertionError('Persistent learned tensors must not alias')

        def update(case):
            for name in INPUTS:
                ttnn.copy_host_to_device_tensor(host_inputs[case][name],inputs[name])
            ttnn.synchronize_device(mesh)

        def run():
            return execute(ttnn,mesh,collectives,inputs,parameters,layers,retain,mask_validated=True)

        def audit_inputs(mode,ordinal,case):
            if {name:addresses(ttnn,value) for name,value in inputs.items()}!=input_bindings:
                raise AssertionError('Borrowed input binding moved')
            for name in INPUTS:
                for chip in range(2):
                    value = cases[case]['values'][name]
                    expected = value[chip:chip+1] if name.startswith('feature_') else value
                    exact = tensor_digest(host(inputs[name],chip))==tensor_digest(expected)
                    report['input_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,name=name,chip=chip,exact=exact))
                    if not exact:
                        raise AssertionError('Borrowed pipeline input changed')

        eager = []
        for case in range(3):
            progress(f'eager_compile_and_execute_{case}')
            update(case)
            result = run()
            ttnn.synchronize_device(mesh)
            observed = {}
            eager.append(observed)
            for name,value in flatten(result).items():
                observed[name] = [host(value,chip) for chip in range(2)]
                for chip,actual in enumerate(observed[name]):
                    valid = bool(torch.isfinite(actual).all())
                    report['stage_checks'].append(dict(case=case,name=name,chip=chip,valid=valid,
                        sha256=tensor_digest(actual),shape=list(actual.shape),dtype=str(actual.dtype)))
                    if not valid:
                        raise AssertionError('Nonfinite learned pipeline stage')
                    if name.endswith(('.attention.attention_partial','.finish.output','.final.final_norm')):
                        zero = bool(torch.count_nonzero(actual[:,:,7:])==0)
                        report['padding_checks'].append(dict(case=case,name=name,chip=chip,zero=zero))
                        if not zero:
                            raise AssertionError('Inactive proposal rows changed')
                    reference = 'projected_context' if name=='projection.context' else 'final_norm' if name=='-1.final.final_norm' \
                        else f"layer_{name.split('.')[0]}_{name.split('.')[-1]}" if name.endswith(('.mlp.attention_residual','.finish.output')) else None
                    if reference is not None:
                        rows = 32 if reference=='projected_context' else 7
                        report['cpu_diagnostics'].append(dict(case=case,name=name,chip=chip,
                            **error_summary(actual[:,0,:rows],cpu[case][reference])))
            for name in ('projection.context','-1.final.final_norm'):
                exact = tensor_digest(observed[name][0])==tensor_digest(observed[name][1])
                report['replica_checks'].append(dict(case=case,name=name,exact=exact))
                if not exact:
                    raise AssertionError('Replicated learned outputs disagree across physical cards')
            audit_inputs('eager',case,case)
            release_owned(ttnn,transient)
            transient.clear()
        progress('capture_projection_and_all_five_layers')
        update(0)
        trace,result = capture_operation(ttnn,mesh,run)
        report['backbone_with_projection_captured'] = True
        output_bindings = {name:addresses(ttnn,value) for name,value in flatten(result).items()}
        for ordinal,case in enumerate(REPLAYS):
            progress(f'changing_input_replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            if {name:addresses(ttnn,value) for name,value in flatten(result).items()}!=output_bindings:
                raise AssertionError('Captured pipeline output binding moved')
            for name,value in flatten(result).items():
                for chip in range(2):
                    actual,expected = tensor_digest(host(value,chip)),tensor_digest(eager[case][name][chip])
                    report['replay_checks'].append(dict(ordinal=ordinal,case=case,name=name,chip=chip,exact=actual==expected,
                        actual_sha256=actual,expected_sha256=expected))
                    if actual!=expected:
                        raise AssertionError('Complete hardware pipeline replay differs from eager execution')
            audit_inputs('replay',ordinal,case)
        for chip in range(2):
            actual = tensor_digest(host(result['backbone']['final_norm'],chip))
            detected = actual==tensor_digest(eager[0]['-1.final.final_norm'][chip]) \
                and actual!=tensor_digest(eager[2]['-1.final.final_norm'][chip])
            report['stale_controls'].append(dict(chip=chip,detected=detected))
            if not detected:
                raise AssertionError('Changing inputs must distinguish stale final output')
        progress('warm_trace_timing_not_tg')
        for sample in range(3):
            start = time.perf_counter()
            for replay in range(10):
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=False)
            ttnn.synchronize_device(mesh)
            report['timing_samples'].append(dict(sample=sample,replays=10,elapsed_seconds=time.perf_counter()-start))
            for chip in range(2):
                exact = tensor_digest(host(result['backbone']['final_norm'],chip))==tensor_digest(eager[0]['-1.final.final_norm'][chip])
                report['timing_checks'].append(dict(sample=sample,chip=chip,exact=exact))
                if not exact:
                    raise AssertionError('Timed output differs from audited pipeline')
        report['backbone_projection_ms'] = 1000*sum(row['elapsed_seconds'] for row in report['timing_samples'])/30
        ttnn.release_trace(mesh,trace)
        trace = None
        release_owned(ttnn,transient)
        transient.clear()
        for name in PARAMETERS:
            progress('audit_after_'+name)
            audit_parameter(name,'after')
        if {name:len(report[name]) for name in COUNTS}!=COUNTS:
            raise AssertionError('Incomplete hardware integration matrix')
        report['cpu_numerical_gate_passed'] = all(row['passed'] for row in report['cpu_diagnostics'])
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
            report['sources_after'],report['native_sources_after'] = source_hashes(),native_fingerprints(root,layer_report)
            if report['sources']!=report['sources_after'] or report['native_sources']!=report['native_sources_after']:
                raise ValueError('Hardware integration sources or native runtime changed')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__=='__main__':
    main()
