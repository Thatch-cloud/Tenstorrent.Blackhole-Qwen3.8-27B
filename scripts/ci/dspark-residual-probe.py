"""Simulator-only residual rounding diagnostic on complete captured layer operands; no layer or replay qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from dspark_layer_diagnostic import OPERANDS_SHA256, REPORT_SHA256, load_capture
from dspark_projection import difference, digest, tensor_digest
from dspark_residual import POLICY, add
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


SOURCES = ('dspark-residual-probe.py','dspark_residual.py','dspark_layer_diagnostic.py',
    'dspark_projection.py','dspark-projection-probe.py','dspark_layer.py','dspark_layer_reference.py',
    'feature_projection.py','gdn_multitoken_conv.py','../../optimisation/sim/run-dispatch-probe.sh')


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','capture','operands'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    if (os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1' or os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') == '1'):
        raise ValueError('Simulator diagnostic requires the original runtime and no hardware allocation')
    original,payload = load_capture(options.capture,options.operands)
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    spec = importlib.util.spec_from_file_location('dspark_residual_runtime',Path(__file__).with_name('dspark-projection-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    report = dict(passed=False,closed_cleanly=False,backend='simulator',scope=__doc__,policy=POLICY,
        mode='residual_diagnostic',sources=source_hashes(),native_sources=probe.fingerprints(root,composed_norm=True),
        capture_sha256=REPORT_SHA256,operands_sha256=OPERANDS_SHA256,eligible_for_hardware=False,
        full_pipeline_captured=False,target_integrated=False,fabric_tested=False,
        control_checks=[],candidate_checks=[],input_checks=[])
    mesh = None
    owned,transient = [],[]

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    def retain(value):
        transient.append(value)
        return value

    try:
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value):
            result = ttnn.from_torch(value,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        def host(value,chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        for case in range(3):
            attention_residual = payload['eager']['mlp',case]['attention_residual']
            output = payload['eager']['finish',case]['output']
            values = {
                'attention':(payload['fixtures'][case]['noise'],payload['eager']['mlp',case]['attention_narrowed'][0],attention_residual),
                'finish':(attention_residual[0],payload['eager']['finish',case]['down_narrowed'][0],output),
            }
            for phase,(first,second,old) in values.items():
                progress(f'{phase}_{case}')
                inputs = [upload(value) for value in (first,second)]
                control = retain(ttnn.add(*inputs,dtype=ttnn.bfloat16,memory_config=ttnn.DRAM_MEMORY_CONFIG))
                candidate = add(ttnn,*inputs,retain)
                ttnn.synchronize_device(mesh)
                expected = first+second
                for chip in range(2):
                    report['control_checks'].append(dict(phase=phase,case=case,chip=chip,
                        **difference(host(control,chip),old[chip],exact=True)))
                    report['candidate_checks'].append(dict(phase=phase,case=case,chip=chip,
                        **difference(host(candidate,chip),expected,exact=True)))
                    for index,value in enumerate(inputs):
                        expected_input = (first,second)[index]
                        report['input_checks'].append(dict(phase=phase,case=case,chip=chip,tensor=index,
                            exact=tensor_digest(host(value,chip))==tensor_digest(expected_input)))
                release_owned(ttnn,transient)
                release_owned(ttnn,owned)
                transient.clear()
                owned.clear()
        if (any(not entry['passed'] for name in ('control_checks','candidate_checks') for entry in report[name])
                or any(not entry['exact'] for entry in report['input_checks'])):
            raise AssertionError('Residual diagnostic requires exact control, candidate and borrowed input bits')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn,transient)
                release_owned(ttnn,owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = source_hashes()
            report['native_sources_after'] = probe.fingerprints(root,composed_norm=True)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Residual diagnostic source or native runtime changed')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
