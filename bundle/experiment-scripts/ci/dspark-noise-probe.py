"""Seven-row embedding/gather/zero-padding layout gate; synthetic 64-row table, not target weights or TG."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from attention_batch import capture_operation
from dspark_projection import difference, digest, tensor_digest
from dspark_target import noise_embeddings
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate
from sim_memory_budget import require_clean, snapshot


spec = importlib.util.spec_from_file_location('noise_vocabulary_probe',Path(__file__).with_name('dspark-vocabulary-probe.py'))
VOCABULARY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VOCABULARY)
SOURCES = tuple(sorted(set(VOCABULARY.SOURCES+('dspark-noise-probe.py','dspark_target.py',
    'dspark_markov_device.py','dspark_pipeline.py','dspark_backbone_mesh.py','dspark_layer_mesh.py'))))
REPLAYS = (0,1,0)
COUNTS = dict(eager_checks=4,replay_checks=6,input_checks=20,padding_checks=10,stale_controls=2,fixture_controls=2)


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def fingerprints(root):
    result = VOCABULARY.MESH.fingerprints(root)
    directory = root/'ttnn/cpp/ttnn/operations/embedding'
    if not directory.is_dir():
        raise ValueError('Native embedding operation sources required')
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in ('.cpp','.hpp','.h'):
            result[str(path.relative_to(root))] = digest(path)
    return result


def fixtures():
    import torch

    rows = torch.arange(64).reshape(64,1)
    columns = torch.arange(5120).reshape(1,5120)
    table = (1+(rows*73+columns)%2048).float().div(64).bfloat16()
    packed = torch.stack(table.chunk(2,dim=-1)).unsqueeze(1)
    identifiers = (torch.tensor([[0,63,31,1,62,33,2]]),torch.tensor([[63,0,33,62,1,2,31]]))
    expected = []
    for values in identifiers:
        result = torch.zeros(1,1,32,5120,dtype=torch.bfloat16)
        result[:,:,:7] = table[values].unsqueeze(1)
        expected.append(result)
    return packed,identifiers,expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF')!='1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY')!='1'
            or Path(os.environ['TT_METAL_SIMULATOR']).parent.name!='shared-bdf'
            or any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE','QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh bounded shared-BDF simulator evidence and original runtime required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False,closed_cleanly=False,scope=__doc__,backend='simulator',link_policy=validate(os.environ),
        sources=source_hashes(),native_sources=fingerprints(root),resources_before=snapshot(bounded=True),
        synthetic_lookup_rows=64,hidden_width=5120,query_rows=7,padded_rows=32,
        target_weights_executed=False,markov_executed=False,full_pipeline_captured=False,
        target_integrated=False,physical_fabric_tested=False,**{name:[] for name in COUNTS})
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
        packed,identifiers,expected = fixtures()
        for name,wrong in (('rank_swap',expected[0].roll(2560,dims=-1)),('dropped_row_zero',expected[0].roll(-1,dims=2))):
            detected = not torch.equal(expected[0],wrong)
            report['fixture_controls'].append(dict(name=name,detected=detected))
            if not detected:
                raise AssertionError('Noise fixture must detect rank or query-row corruption')
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        table = ttnn.from_torch(packed,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0))
        tokens = ttnn.from_torch(identifiers[0],dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        owned.extend((table,tokens))
        bindings = [addresses(ttnn,value) for value in owned]
        host_tokens = [ttnn.from_torch(value,dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for value in identifiers]
        target = SimpleNamespace(mesh_device=mesh,num_devices=2,vocab_size=248320,
            embd=lambda values,memory_config:ttnn.embedding(values,table,layout=ttnn.TILE_LAYOUT,memory_config=memory_config))

        def update(case):
            ttnn.copy_host_to_device_tensor(host_tokens[case],tokens)
            ttnn.synchronize_device(mesh)

        def run():
            return noise_embeddings(ttnn,target,mesh,collectives,tokens,retain)

        def audit(result,mode,ordinal,case):
            if [addresses(ttnn,value) for value in owned]!=bindings:
                raise AssertionError('Borrowed lookup or query storage moved')
            for chip in range(2):
                actual = host(result,chip)
                row = dict(ordinal=ordinal,case=case,chip=chip,actual_sha256=tensor_digest(actual),
                    expected_sha256=tensor_digest(expected[case]),**difference(actual,expected[case],exact=True))
                report[mode+'_checks'].append(row)
                zero = bool(torch.count_nonzero(actual[:,:,7:])==0)
                report['padding_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,chip=chip,zero=zero))
                if not row['passed'] or not zero:
                    raise AssertionError('Seven exact embedding rows and zero inactive rows required')
                for name,value,reference in (('table',table,packed[chip:chip+1]),('identifiers',tokens,identifiers[case])):
                    actual_input = host(value,chip)
                    exact = torch.equal(actual_input.to(reference.dtype),reference)
                    report['input_checks'].append(dict(mode=mode,ordinal=ordinal,case=case,chip=chip,name=name,exact=bool(exact)))
                    if not exact:
                        raise AssertionError('Borrowed lookup table or query identifiers changed')

        for case in range(2):
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
        output_binding = addresses(ttnn,result)
        for ordinal,case in enumerate(REPLAYS):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            if addresses(ttnn,result)!=output_binding:
                raise AssertionError('Captured padded output binding changed')
            audit(result,'replay',ordinal,case)
        for chip in range(2):
            actual = host(result,chip)
            detected = torch.equal(actual,expected[0]) and not torch.equal(actual,expected[1])
            report['stale_controls'].append(dict(chip=chip,detected=bool(detected)))
            if not detected:
                raise AssertionError('Omitted query update must remain distinguishable')
        if {name:len(report[name]) for name in COUNTS}!=COUNTS:
            raise AssertionError('All 44 noise-layout checks required')
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
            if report['sources']!=report['sources_after'] or report['native_sources']!=report['native_sources_after']:
                raise ValueError('Noise-layout source or native runtime changed')
            report['resources_after'] = snapshot(bounded=True)
            require_clean(report['resources_before'],report['resources_after'])
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__=='__main__':
    main()
