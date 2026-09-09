"""Simulator-only full-vocabulary collective with synthetic logits; target LM head and Markov feedback are not executed."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_projection import difference, digest, tensor_digest
from dspark_vocabulary import gather_logits
from dspark_vocabulary_fixture import inputs as fixture_inputs
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate


spec = importlib.util.spec_from_file_location('vocabulary_native',Path(__file__).with_name('dspark-mesh-probe.py'))
MESH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MESH)
SOURCES = tuple(sorted(set(MESH.SOURCES+('dspark-vocabulary-probe.py','dspark_vocabulary.py',
    'dspark_vocabulary_fixture.py','dspark_inputs.py'))))
REPLAYS = (0,2,1,0)
COUNTS = dict(eager_checks=12,replay_checks=16,input_checks=14,stale_controls=2,fixture_controls=6)


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    policy = validate(os.environ)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF')!='1'
            or Path(os.environ['TT_METAL_SIMULATOR']).parent.name!='shared-bdf'
            or any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE','QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh simulator evidence, shared BDF and original runtime required; hardware is prohibited')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False,closed_cleanly=False,scope=__doc__,backend='simulator',link_policy=policy,
        sources=source_hashes(),native_sources=MESH.fingerprints(root),vocabulary=248320,proposal_rows=7,
        synthetic_logits=True,target_head_executed=False,markov_executed=False,physical_fabric_tested=False,
        target_integrated=False,full_pipeline_captured=False,eligible_for_hardware=False,
        **{name:[] for name in COUNTS})
    mesh = trace = None
    owned,transient = [],[]

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
        fixtures = [fixture_inputs(pattern) for pattern in range(3)]
        for case,fixture in enumerate(fixtures):
            swapped = torch.cat(fixture['packed'].chunk(2,dim=0)[::-1],dim=-1)
            alternatives = dict(swapped_ranks=swapped[:,:,:7].float(),dropped_row_zero=fixture['full_logits'][:,:,1:8].float())
            for name,value in alternatives.items():
                detected = tensor_digest(value)!=tensor_digest(fixture['base_logits'])
                report['fixture_controls'].append(dict(case=case,name=name,detected=detected))
                if not detected:
                    raise AssertionError('Full-vocabulary fixture must detect wrong rank order and dropped row zero')
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        input_tensor = ttnn.from_torch(fixtures[0]['packed'],dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
            device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0))
        owned.append(input_tensor)
        bindings = addresses(ttnn,input_tensor)
        host_inputs = [ttnn.from_torch(fixture['packed'],dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0)) for fixture in fixtures]

        def update(case):
            ttnn.copy_host_to_device_tensor(host_inputs[case],input_tensor)
            ttnn.synchronize_device(mesh)

        def run():
            return gather_logits(ttnn,mesh,collectives,input_tensor,retain)

        def audit(result,mode,ordinal,case):
            if addresses(ttnn,input_tensor)!=bindings or set(result)!= {'full_logits','base_logits'}:
                raise AssertionError('Persistent input binding and both complete output stages required')
            for chip in range(2):
                exact = tensor_digest(host(input_tensor,chip))==tensor_digest(fixtures[case]['packed'][chip:chip+1])
                report['input_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,chip=chip,exact=exact))
                if not exact:
                    raise AssertionError('Borrowed vocabulary shard changed')
                for name,value in result.items():
                    actual = host(value,chip)
                    expected = fixtures[case][name]
                    report[mode+'_checks'].append(dict(ordinal=ordinal,case=case,chip=chip,name=name,
                        actual_sha256=tensor_digest(actual),expected_sha256=tensor_digest(expected),
                        **difference(actual,expected,exact=True)))
            if any(not entry['passed'] for entry in report[mode+'_checks']):
                raise AssertionError('All vocabulary scores and query rows must survive the device handoff exactly')

        for case in range(3):
            progress(f'eager_{case}')
            update(case)
            result = run()
            ttnn.synchronize_device(mesh)
            audit(result,'eager',case,case)
            release_owned(ttnn,transient)
            transient.clear()
        progress('capture')
        update(0)
        trace,result = capture_operation(ttnn,mesh,run)
        output_bindings = {name:addresses(ttnn,value) for name,value in result.items()}
        for ordinal,case in enumerate(REPLAYS):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            if {name:addresses(ttnn,value) for name,value in result.items()}!=output_bindings:
                raise AssertionError('Captured full-vocabulary bindings moved')
            audit(result,'replay',ordinal,case)
            for entry in report['replay_checks'][-4:]:
                entry['bindings_stable'] = True
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
        for chip in range(2):
            actual = tensor_digest(host(result['base_logits'],chip))
            detected = actual==tensor_digest(fixtures[0]['base_logits']) and actual!=tensor_digest(fixtures[2]['base_logits'])
            report['stale_controls'].append(dict(chip=chip,detected=detected))
        if {name:len(report[name]) for name in COUNTS}!=COUNTS:
            raise AssertionError('Complete 50-check full-vocabulary transport matrix required')
        if any(not entry['detected'] for entry in report['stale_controls']):
            raise AssertionError('Omitted vocabulary input updates must be detectable')
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
            report['sources_after'],report['native_sources_after'] = source_hashes(),MESH.fingerprints(root)
            if report['sources_after']!=report['sources'] or report['native_sources_after']!=report['native_sources']:
                raise ValueError('Vocabulary handoff sources or native runtime changed')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
