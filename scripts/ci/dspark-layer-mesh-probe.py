"""Simulator-only complete learned layer in one device trace; exact transport differential, not CPU numerical qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_attention import validate_mask
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import PHASES, REPLAY_CASES, SPECIFICATIONS, WIDE_POLICY, pack_weight
from dspark_layer_diagnostic import COMPOSED_REPORT_SHA256, COMPOSED_OPERANDS_SHA256, load_capture
from dspark_layer_mesh import INPUTS, execute
from dspark_projection import difference, digest, tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate


def dependency(name):
    spec = importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAYER = dependency('dspark-layer-probe')
MESH = dependency('dspark-mesh-probe')
SOURCES = tuple(sorted(set(LAYER.SOURCES+MESH.SOURCES+('dspark-layer-mesh-probe.py','dspark_layer_mesh.py'))))
EXPECTED_COUNTS = dict(eager_checks=156,replay_checks=208,input_checks=112,parameter_checks=44,
    stale_controls=2,padding_checks=12)


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','capture','operands','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    policy = validate(os.environ)
    outputs_path = options.output.with_suffix('.operands.pt')
    if (options.output.exists() or outputs_path.exists() or os.environ.get('QWEN_SIM_SHARED_BDF')!='1'
            or Path(os.environ['TT_METAL_SIMULATOR']).parent.name!='shared-bdf'
            or any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE','QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh simulator evidence, shared BDF and original runtime required; hardware is prohibited')
    original,payload = load_capture(options.capture,options.operands,composed_layer=True)
    if original['policy']!=WIDE_POLICY:
        raise ValueError('Unchanged complete-layer arithmetic policy required')
    sources = source_hashes()
    for name,sha in original['sources'].items():
        if name!='../../optimisation/sim/run-dispatch-probe.sh' and sources.get(name)!=sha:
            raise ValueError('Retained learned arithmetic source changed: '+name)
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    native = MESH.fingerprints(root)
    if any(native.get(name)!=sha for name,sha in original['native_sources'].items()):
        raise ValueError('Retained learned native arithmetic runtime changed')
    report = dict(passed=False,closed_cleanly=False,checkpoint_closed=False,scope=__doc__,backend='simulator',
        layer=0,context_rows=32,proposal_rows=7,policy=WIDE_POLICY,link_policy=policy,
        checkpoint_sha256=CHECKPOINT_SHA256,sources=sources,native_sources=native,
        capture_sha256=COMPOSED_REPORT_SHA256,operands_sha256=COMPOSED_OPERANDS_SHA256,
        arithmetic_control_full_gate_passed=False,simulated_collectives=True,physical_fabric_tested=False,
        full_layer_captured=False,full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        **{name:[] for name in EXPECTED_COUNTS})
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
        progress('load_complete_learned_layer')
        fixtures = payload['fixtures']
        for fixture in fixtures:
            validate_mask(fixture['mask'],32,key_multiple=64)
        reader = VerifiedWeights(options.checkpoint)
        weights = {name:reader.tensor('layers.0.'+name) for name in SPECIFICATIONS}
        report['parameter_sha256'] = {name:reader.fingerprints()['layers.0.'+name] for name in SPECIFICATIONS}
        if report['parameter_sha256']!=payload['parameter_sha256']:
            raise ValueError('Complete learned weights differ from retained native control')
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

        packed = {name:pack_weight(name,value) for name,value in weights.items()}
        device_weights = {name:upload(value,sharded=sharded) for name,(value,sharded) in packed.items()}
        parameter_bindings = {name:addresses(ttnn,value) for name,value in device_weights.items()}

        def audit_parameters(phase):
            if {name:addresses(ttnn,value) for name,value in device_weights.items()}!=parameter_bindings:
                raise AssertionError('Learned parameter bindings moved')
            for name,value in device_weights.items():
                packed_value,sharded = packed[name]
                for chip in range(2):
                    expected = packed_value[chip:chip+1] if sharded else packed_value
                    exact = tensor_digest(host(value,chip))==tensor_digest(expected)
                    report['parameter_checks'].append(dict(phase=phase,chip=chip,tensor=name,
                        exact=exact,bindings_stable=True))
                    if not exact:
                        raise AssertionError('Learned parameter bits changed')

        progress('audit_all_eleven_parameters_before')
        audit_parameters('before')
        values = {case:[fixture['context'],fixture['noise'],*fixture['tables']['q'],*fixture['tables']['k'],
            fixture['mask'],fixture['live']] for case,fixture in enumerate(fixtures)}
        inputs = {name:upload(value) for name,value in zip(INPUTS,values[0],strict=True)}
        host_inputs = {case:[upload(value,device=False) for value in rows] for case,rows in values.items()}
        input_bindings = {name:addresses(ttnn,value) for name,value in inputs.items()}
        all_bindings = [addresses(ttnn,value) for value in owned]
        if any(len({binding[chip] for binding in all_bindings})!=len(all_bindings) for chip in range(2)):
            raise AssertionError('Persistent inputs and learned weights must be disjoint')

        def update(case):
            for name,value in zip(INPUTS,host_inputs[case],strict=True):
                ttnn.copy_host_to_device_tensor(value,inputs[name])
            ttnn.synchronize_device(mesh)

        def run():
            return execute(ttnn,mesh,collectives,inputs,device_weights,retain,mask_validated=True)

        def flattened(result):
            if set(result)!=set(PHASES) or any(set(result[phase])!=set(stages) for phase,stages in PHASES.items()):
                raise AssertionError('All complete learned layer stages required')
            return {(phase,stage):result[phase][stage] for phase,stages in PHASES.items() for stage in stages}

        def audit_inputs(mode,ordinal,case):
            if {name:addresses(ttnn,value) for name,value in inputs.items()}!=input_bindings:
                raise AssertionError('Persistent learned layer inputs moved')
            for index,name in enumerate(INPUTS):
                for chip in range(2):
                    exact = tensor_digest(host(inputs[name],chip))==tensor_digest(values[case][index])
                    report['input_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,chip=chip,tensor=name,exact=exact))
                    if not exact:
                        raise AssertionError('Borrowed learned layer input changed')

        def audit_outputs(result,mode,ordinal,case):
            observed = {}
            for (phase,stage),value in flattened(result).items():
                observed[phase,stage] = [host(value,chip) for chip in range(2)]
                for chip,actual in enumerate(observed[phase,stage]):
                    expected = payload['eager'][phase,case][stage][chip]
                    report[mode+'_checks'].append(dict(phase=phase,stage=stage,ordinal=ordinal,case=case,chip=chip,
                        actual_sha256=tensor_digest(actual),expected_sha256=tensor_digest(expected),
                        **difference(actual,expected,exact=True)))
            audit_inputs(mode,ordinal,case)
            if any(not entry['passed'] for entry in report[mode+'_checks']):
                eager[mode,ordinal,case] = observed
                raise AssertionError('Complete device layer differs from retained same-arithmetic native control')
            return observed

        for case in range(3):
            progress(f'full_layer_eager_{case}')
            update(case)
            result = run()
            ttnn.synchronize_device(mesh)
            observed = audit_outputs(result,'eager',case,case)
            eager['eager',case,case] = observed
            for phase,stage in (('attention','attention_partial'),('finish','output')):
                for chip in range(2):
                    zero = bool(torch.count_nonzero(observed[phase,stage][chip][:,:,7:])==0)
                    report['padding_checks'].append(dict(phase=phase,case=case,chip=chip,zero=zero))
            release_owned(ttnn,transient)
            transient.clear()
        progress('capture_complete_layer')
        update(0)
        trace,result = capture_operation(ttnn,mesh,run)
        report['full_layer_captured'] = True
        output_bindings = {key:addresses(ttnn,value) for key,value in flattened(result).items()}
        for ordinal,case in enumerate(REPLAY_CASES):
            progress(f'full_layer_replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            if {key:addresses(ttnn,value) for key,value in flattened(result).items()}!=output_bindings:
                raise AssertionError('Single-trace learned output bindings moved')
            audit_outputs(result,'replay',ordinal,case)
            for entry in report['replay_checks'][-52:]:
                entry['bindings_stable'] = True
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
        for chip in range(2):
            actual = tensor_digest(host(result['finish']['output'],chip))
            detected = actual==tensor_digest(payload['eager']['finish',0]['output'][chip]) \
                and actual!=tensor_digest(payload['eager']['finish',2]['output'][chip])
            report['stale_controls'].append(dict(chip=chip,detected=detected))
        ttnn.release_trace(mesh,trace)
        trace = None
        release_owned(ttnn,transient)
        transient.clear()
        progress('audit_all_eleven_parameters_after')
        audit_parameters('after')
        if {name:len(report[name]) for name in EXPECTED_COUNTS}!=EXPECTED_COUNTS:
            raise AssertionError('Complete 534-check learned layer differential matrix required')
        if (any(not entry['detected'] for entry in report['stale_controls'])
                or any(not entry['zero'] for entry in report['padding_checks'])):
            raise AssertionError('Stale-input and inactive-row controls must pass')
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
            report['sources_after'],report['native_sources_after'] = source_hashes(),MESH.fingerprints(root)
            if report['sources_after']!=report['sources'] or report['native_sources_after']!=report['native_sources']:
                raise ValueError('Learned device layer sources or native runtime changed')
            if eager:
                with outputs_path.open('xb') as stream:
                    torch.save(dict(sources=report['sources'],native_sources=report['native_sources'],policy=WIDE_POLICY,
                        parameter_sha256=report['parameter_sha256'],observed=eager),stream)
                report['observed_operands'] = dict(path=str(outputs_path),sha256=digest(outputs_path))
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
