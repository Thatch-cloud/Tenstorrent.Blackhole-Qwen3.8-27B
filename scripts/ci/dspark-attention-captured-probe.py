"""Eager-only FP32 attention diagnostic on retained learned Q/K/V; no full-layer or performance qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from draft_attention import composed_draft_attention
from dspark_attention import validate_mask
from dspark_layer_diagnostic import OPERANDS_SHA256, REPORT_SHA256, error_summary, load_capture
from dspark_projection import TOLERANCE, digest, tensor_digest
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


POLICY = 'Existing cached FP32 SFPU dots and row sum; explicit softmax; final BF16 cast'
SOURCES = ('dspark-attention-captured-probe.py','draft_attention.py','draft_dot.py','draft_row_sum.py',
    'draft_dot_compute.cpp','draft_dot_io.cpp','draft_row_sum_compute.cpp','draft_row_sum_io.cpp',
    'dspark_attention.py','dspark_layer_diagnostic.py','dspark_layer_reference.py','dspark_layer.py',
    'dspark_projection.py','dspark-projection-probe.py','feature_projection.py','gdn_multitoken_conv.py',
    '../../optimisation/sim/run-dispatch-probe.sh')


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def reference_inputs(values, chip):
    import torch

    shapes = ((2,16,32,128),(2,4,64,128),(2,4,64,128),(1,1,32,64))
    if type(chip) is not int or chip not in (0,1) or len(values)!=4:
        raise ValueError('Complete two-chip learned attention fixture required')
    if any(value.dtype!=torch.bfloat16 or value.device.type!='cpu' or tuple(value.shape)!=shape
            for value,shape in zip(values,shapes)) or any(not torch.isfinite(value).all() for value in values[:3]):
        raise ValueError('Finite complete BF16 learned Q/K/V and full mask required')
    validate_mask(values[3],32,key_multiple=64)
    return (values[0][chip:chip+1].float(),
        *(value[chip:chip+1].float().repeat_interleave(4,1) for value in values[1:3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','capture','operands'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    if options.output.exists() or options.output.with_suffix('.operands.pt').exists():
        raise ValueError('Refusing to overwrite captured-attention evidence')
    require_projection_environment(os.environ,False)
    if any(os.environ.get(name)=='1' for name in ('QWEN_HARDWARE_TESTS','QWEN_CARDS_ALLOCATED',
            'QWEN_SIM_PACKER_ZERO_GRAFT','QWEN_PRECISE_DRAFT_ACTIVE')):
        raise ValueError('Simulator diagnostic requires original native runtime and no hardware allocation')
    original,payload = load_capture(options.capture,options.operands)
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    spec = importlib.util.spec_from_file_location('dspark_captured_attention_runtime',Path(__file__).with_name('dspark-projection-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    report = dict(passed=False,closed_cleanly=False,backend='simulator',mode='captured_attention_diagnostic',
        policy=POLICY,scope=__doc__,sources=source_hashes(),native_sources=probe.fingerprints(root,composed_norm=True),
        capture_sha256=REPORT_SHA256,operands_sha256=OPERANDS_SHA256,tolerance=TOLERANCE,
        eligible_for_hardware=False,full_pipeline_captured=False,target_integrated=False,fabric_tested=False,
        candidate_checks=[],baseline_comparisons=[],stage_checks=[],input_checks=[],transform_checks=[])
    mesh = None
    owned,outputs = [],{}

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

    try:
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()
        for case in range(3):
            progress(f'attention_{case}')
            captured = payload['eager']['attention',case]
            mask = payload['fixtures'][case]['mask']
            validate_mask(mask,32,key_multiple=64)
            host_inputs = [torch.cat(captured[name],dim=0) for name in ('q_rotary','k_rotary','v_heads')]+[mask]
            inputs = []
            for index,value in enumerate(host_inputs):
                device_value = ttnn.from_torch(value,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=0) if index<3 else ttnn.ReplicateTensorToMesh(mesh))
                owned.append(device_value)
                inputs.append(device_value)

            def inspect(query, keys, values, scores, masked, probabilities, output):
                outputs[case] = {name:[host(value,chip) for chip in range(2)] for name,value in
                    (('query',query),('keys',keys),('values',values),('scores',scores),
                     ('masked',masked),('probabilities',probabilities),('output',output))}

            output = composed_draft_attention(ttnn,mesh,*inputs,inspect=inspect,explicit_softmax=True,
                fused_dots=True,cache_dot_tiles=True,fused_row_sum=True)
            owned.append(output)
            narrowed = ttnn.typecast(output,ttnn.bfloat16)
            owned.append(narrowed)
            ttnn.synchronize_device(mesh)
            outputs[case]['narrowed'] = [host(narrowed,chip) for chip in range(2)]
            for chip in range(2):
                query,keys,values = reference_inputs(host_inputs,chip)
                for name,reference in zip(('query','keys','values'),(query,keys,values)):
                    report['transform_checks'].append(dict(case=case,chip=chip,name=name,
                        exact=tensor_digest(outputs[case][name][chip])==tensor_digest(reference)))
                expected = torch.nn.functional.scaled_dot_product_attention(query,keys,values,attn_mask=mask.float())
                scores = query @ keys.transpose(-1,-2)
                probabilities = torch.softmax(scores*(128**-.5)+mask.float(),dim=-1)
                report['candidate_checks'].append(dict(case=case,chip=chip,
                    **error_summary(outputs[case]['narrowed'][chip],expected.bfloat16())))
                report['baseline_comparisons'].append(dict(case=case,chip=chip,
                    scope='Retained precise native output, not rerun in this diagnostic',
                    **error_summary(captured['attended'][chip],expected.bfloat16())))
                for name,reference in (('scores',scores),('probabilities',probabilities),('output',expected)):
                    report['stage_checks'].append(dict(case=case,chip=chip,name=name,
                        **error_summary(outputs[case][name][chip],reference)))
                for index,value in enumerate(inputs):
                    reference = host_inputs[index][chip:chip+1] if index<3 else mask
                    report['input_checks'].append(dict(case=case,chip=chip,tensor=index,
                        exact=tensor_digest(host(value,chip))==tensor_digest(reference)))
            release_owned(ttnn,owned)
            owned.clear()
        if (len(report['candidate_checks'])!=6 or len(report['stage_checks'])!=18
                or len(report['input_checks'])!=24 or len(report['transform_checks'])!=18
                or any(not entry['passed'] for name in ('candidate_checks','stage_checks') for entry in report[name])
                or any(not entry['exact'] for name in ('input_checks','transform_checks') for entry in report[name])):
            raise AssertionError('Captured attention numerical and borrowed-input checks failed')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn,owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = source_hashes()
            report['native_sources_after'] = probe.fingerprints(root,composed_norm=True)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Captured attention sources or native runtime changed during execution')
            operands = options.output.with_suffix('.operands.pt')
            with operands.open('xb') as stream:
                torch.save(dict(outputs=outputs,sources=report['sources'],native_sources=report['native_sources'],
                    policy=POLICY,capture_sha256=REPORT_SHA256,operands_sha256=OPERANDS_SHA256),stream)
            report['observed_tensors'] = dict(path=str(operands),sha256=digest(operands))
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
