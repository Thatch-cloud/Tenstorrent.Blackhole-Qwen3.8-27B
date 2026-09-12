"""Simulator-only captured device TP handoffs on actual learned partials; no physical-fabric or full-layer qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_layer import REPLAY_CASES, reduce_residual
from dspark_layer_diagnostic import COMPOSED_REPORT_SHA256, COMPOSED_OPERANDS_SHA256, load_capture
from dspark_mesh import gather_partials
from dspark_projection import difference, digest, tensor_digest
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate


SOURCES = ('dspark-mesh-probe.py','dspark_mesh.py','dspark_layer.py','dspark_residual.py',
    'dspark_layer_diagnostic.py','dspark_layer_reference.py','dspark_projection.py',
    'dspark-layer-probe.py','dspark-projection-probe.py','projection_link_policy.py','sampling_link_policy.py',
    'attention_batch.py','feature_projection.py','gdn_multitoken_conv.py','native_draft_sdpa.py',
    '../../optimisation/sim/run-dispatch-probe.sh')
STAGES = ('sum','narrowed','residual')


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def fingerprints(root):
    spec = importlib.util.spec_from_file_location('mesh_layer_native',Path(__file__).with_name('dspark-layer-probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.fingerprints(root,active=False,composed_attention=True)
    directory = root/'ttnn/cpp/ttnn/operations/ccl'
    if not directory.is_dir():
        raise ValueError('Complete native collective sources required')
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in ('.h','.hpp','.cpp'):
            result[str(path.relative_to(root))] = digest(path)
    result['models/tt_transformers/tt/ccl.py'] = digest(root/'models/tt_transformers/tt/ccl.py')
    for name in ('libttsim_bh_x2.so','soc_descriptor.yaml','cluster_descriptor.yaml'):
        result['simulator/shared-bdf/'+name] = digest(Path(os.environ['TT_METAL_SIMULATOR']).parent/name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','capture','operands'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    policy = validate(os.environ)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF')!='1'
            or Path(os.environ['TT_METAL_SIMULATOR']).parent.name!='shared-bdf'
            or any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE','QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh simulator evidence, shared BDF and original runtime required; hardware is prohibited')
    original,payload = load_capture(options.capture,options.operands,composed_layer=True)
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False,closed_cleanly=False,scope=__doc__,backend='simulator',link_policy=policy,
        sources=source_hashes(),native_sources=fingerprints(root),capture_sha256=COMPOSED_REPORT_SHA256,
        operands_sha256=COMPOSED_OPERANDS_SHA256,arithmetic_control_full_gate_passed=False,
        simulated_collectives=True,physical_fabric_tested=False,full_layer_captured=False,
        full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        eager_checks=[],replay_checks=[],input_checks=[],stale_controls=[])
    mesh = trace = None
    owned,transient = [],[]

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    def retain(value):
        transient.append(value)
        return value

    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

    try:
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        for phase in ('attention','finish'):
            inputs_by_case,expected = {},{}
            for case in range(3):
                mlp,finish = payload['eager']['mlp',case],payload['eager']['finish',case]
                partials = payload['eager']['attention',case]['attention_partial'] if phase=='attention' else mlp['down_partial']
                residual = payload['fixtures'][case]['noise'] if phase=='attention' else mlp['attention_residual'][0]
                inputs_by_case[case] = (torch.cat(partials,dim=0),residual)
                expected[case] = [mlp[name] for name in ('attention_sum','attention_narrowed','attention_residual')] \
                    if phase=='attention' else [finish[name] for name in ('down_sum','down_narrowed','output')]
            inputs = []
            for index,value in enumerate(inputs_by_case[0]):
                tensor = ttnn.from_torch(value,dtype=ttnn.float32 if index==0 else ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0) if index==0 else ttnn.ReplicateTensorToMesh(mesh))
                owned.append(tensor)
                inputs.append(tensor)
            bindings = [addresses(ttnn,value) for value in inputs]

            def update(case):
                for index,value in enumerate(inputs_by_case[case]):
                    host_tensor = ttnn.from_torch(value,dtype=ttnn.float32 if index==0 else ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0)
                        if index==0 else ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(host_tensor,inputs[index])
                ttnn.synchronize_device(mesh)

            def execute():
                parts = gather_partials(ttnn,mesh,collectives,inputs[0],retain)
                return reduce_residual(ttnn,*parts,inputs[1],retain)

            def audit(mode,ordinal,case):
                if [addresses(ttnn,value) for value in inputs]!=bindings:
                    raise AssertionError('Persistent mesh handoff inputs moved')
                for chip in range(2):
                    for index,value in enumerate(inputs):
                        source = inputs_by_case[case][index]
                        reference = source[chip:chip+1] if index==0 else source
                        exact = tensor_digest(host(value,chip))==tensor_digest(reference)
                        report['input_checks'].append(dict(phase=phase,mode=mode,ordinal=ordinal,case=case,
                            chip=chip,tensor=index,exact=exact))
                        if not exact:
                            raise AssertionError('Borrowed mesh handoff input changed')

            for case in range(3):
                progress(f'{phase}_eager_{case}')
                update(case)
                result = execute()
                ttnn.synchronize_device(mesh)
                for index,stage in enumerate(STAGES):
                    for chip in range(2):
                        report['eager_checks'].append(dict(phase=phase,case=case,chip=chip,name=stage,
                            **difference(host(result[index],chip),expected[case][index][chip],exact=True)))
                audit('eager',case,case)
                release_owned(ttnn,transient)
                transient.clear()
            progress(f'{phase}_capture')
            update(0)
            trace,result = capture_operation(ttnn,mesh,execute)
            output_bindings = [addresses(ttnn,value) for value in result]
            for ordinal,case in enumerate(REPLAY_CASES):
                progress(f'{phase}_replay_{ordinal}')
                update(case)
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                if [addresses(ttnn,value) for value in result]!=output_bindings:
                    raise AssertionError('Captured collective output bindings moved')
                for index,stage in enumerate(STAGES):
                    for chip in range(2):
                        report['replay_checks'].append(dict(phase=phase,ordinal=ordinal,case=case,chip=chip,name=stage,
                            bindings_stable=True,**difference(host(result[index],chip),expected[case][index][chip],exact=True)))
                audit('replay',ordinal,case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            for chip in range(2):
                actual = tensor_digest(host(result[2],chip))
                detected = actual==tensor_digest(expected[0][2][chip]) and actual!=tensor_digest(expected[2][2][chip])
                report['stale_controls'].append(dict(phase=phase,chip=chip,detected=detected))
            ttnn.release_trace(mesh,trace)
            trace = None
            release_owned(ttnn,transient)
            transient.clear()
            release_owned(ttnn,owned)
            owned.clear()
        expected_counts = dict(eager_checks=36,replay_checks=48,input_checks=56,stale_controls=4)
        if {name:len(report[name]) for name in expected_counts}!=expected_counts:
            raise AssertionError('Complete eager, replay, input and negative-control matrix required')
        if (any(not entry['passed'] for name in ('eager_checks','replay_checks') for entry in report[name])
                or any(not entry['exact'] for entry in report['input_checks'])
                or any(not entry['detected'] for entry in report['stale_controls'])):
            raise AssertionError('Device TP handoffs must reproduce the saved arithmetic exactly')
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
            report['sources_after'],report['native_sources_after'] = source_hashes(),fingerprints(root)
            if report['sources_after']!=report['sources'] or report['native_sources_after']!=report['native_sources']:
                raise ValueError('Mesh handoff sources or native runtime changed')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
