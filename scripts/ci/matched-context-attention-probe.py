"""Weight-free context admission for unchanged split-K math; not full-model or TG acceptance."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

from attention_batch import capture_operation
from dspark_cached_layer import append_queries
import dspark_full_attention
from dspark_hardware_gate import digest
from dspark_projection import tensor_digest
from dspark_splitk_hardware_build import REPORT, validate_build
from dspark_splitk_hardware_scope import HEADER, kernel_scope
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from matched_context_attention import adapter
from matched_context_geometry import geometry
from native_draft_sdpa import precise_draft_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, True)
    context = int(os.environ['QWEN_MATCHED_CONTEXT'])
    plan = geometry(context)
    if (not Path('/dev/tenstorrent').exists() or options.output.exists()
            or os.environ.get('QWEN_SPLITK_ATTENTION') != '1'):
        raise ValueError('Fresh allocated physical split-K context probe required')
    import torch
    import ttnn

    directory, root = Path(__file__).parent, Path(os.environ['TT_METAL_HOME'])
    simulator = directory / 'dspark-splitk-simulator.json'
    build = validate_build(root, directory, REPORT, simulator)
    spec = importlib.util.spec_from_file_location('context_fixture', directory / 'dspark-native-8k-attention-probe.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.CAPACITY, fixture.POSITIONS = plan['capacity'], (context, plan['capacity'] - 15)
    names = (*fixture.SOURCES, Path(__file__).name, 'matched_context_attention.py', 'matched_context_geometry.py')
    def sources():
        return {name: digest(directory / name) for name in names}
    report = dict(passed=False, closed_cleanly=False, backend='hardware', scope=__doc__, geometry=plan,
        checks=[], fixture_controls=[], sources=sources(), build=build,
        numerical_tolerances=dict(rtol=.01, atol=.01), full_request_qualified=False, performance_qualified=False)
    execute = adapter(context)
    mesh = trace = None
    owned, transient = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage, context=context)), flush=True)

    try:
        with patch.object(dspark_full_attention, 'MAX_CONTEXT', plan['capacity']), \
                precise_draft_kernel(root), kernel_scope(root) as kernel, \
                patch.dict(os.environ, QWEN_PRECISE_DRAFT_ACTIVE='1', QWEN_SPLITK_FP32_INTERMEDIATES='1'):
            report['kernel'] = kernel
            if digest(root / fixture.NATIVE.PACKER) != fixture.NATIVE.ORIGINAL_PACKER:
                raise ValueError('Stock hardware packer required')
            progress('reference')
            patterns = fixture.fixtures()
            for position, values in zip(fixture.POSITIONS, patterns, strict=True):
                fixture.validate_fixed_mask(values['mask'], position, plan['capacity'], 15)
            expected = [[fixture.reference(values, chip) for chip in range(2)] for values in patterns]
            report['fixture_controls'] = fixture.controls(patterns, expected)
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=536870912)
            mesh.enable_program_cache()

            def upload(value, name, device=True):
                tensor = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if name == 'mask' else ttnn.ShardTensorToMesh(mesh, dim=0),
                    **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
                if device:
                    owned.append(tensor)
                return tensor

            inputs = {name: upload(patterns[0][name], name) for name in fixture.NAMES}
            payloads = [{name: upload(values[name], name, False) for name in fixture.NAMES} for values in patterns]
            bindings = {name: addresses(ttnn, value) for name, value in inputs.items()}
            eager = {}

            def update(case):
                for name in fixture.NAMES:
                    ttnn.copy_host_to_device_tensor(payloads[case][name], inputs[name])
                ttnn.synchronize_device(mesh)

            def retain(value):
                transient.append(value)
                return value

            def operation():
                keys = append_queries(ttnn, inputs['history_key'], inputs['query_key'], retain,
                    position=plan['capacity'], proposals=15)
                values = append_queries(ttnn, inputs['history_value'], inputs['query_value'], retain,
                    position=plan['capacity'], proposals=15)
                output = execute(ttnn, mesh, inputs['query'], keys, values, inputs['mask'], transient)
                return dict(attention=output, key=keys, value=values)

            def audit(output, mode, case):
                if {name: addresses(ttnn, value) for name, value in inputs.items()} != bindings:
                    raise AssertionError('Persistent inputs moved')
                for chip in range(2):
                    actual = ttnn.to_torch(ttnn.get_device_tensors(output['attention'])[chip])
                    close = torch.isclose(actual.float(), expected[case][chip], rtol=.01, atol=.01)
                    checksum = tensor_digest(actual)
                    if mode == 'eager':
                        eager[case, chip] = checksum
                    passed = (actual.shape == expected[case][chip].shape and actual.dtype == torch.bfloat16
                        and bool(torch.isfinite(actual).all()) and bool(close.all()) and checksum == eager[case, chip])
                    report['checks'].append(dict(mode=mode, case=case, chip=chip, name='attention', passed=passed,
                        failed_elements=int((~close).sum()), max_abs=float((actual.float() - expected[case][chip]).abs().max())))
                    if not passed:
                        raise AssertionError('Full-history numerical or exact replay gate failed')
                    for name in (*fixture.NAMES, 'key', 'value'):
                        tensor = inputs[name] if name in inputs else output[name]
                        actual_input = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])
                        golden = (fixture.joined(patterns[case], name)[chip:chip + 1] if name in ('key', 'value')
                            else patterns[case][name] if name == 'mask' else patterns[case][name][chip:chip + 1])
                        exact = torch.equal(actual_input, golden)
                        report['checks'].append(dict(mode=mode, case=case, chip=chip, name=name, passed=exact))
                        if not exact:
                            raise AssertionError('Input or joined history changed')

            for case in range(2):
                progress('eager_' + str(case))
                update(case)
                output = operation()
                ttnn.synchronize_device(mesh)
                audit(output, 'eager', case)
                release_owned(ttnn, transient)
                transient.clear()
            update(0)
            progress('capture')
            trace, output = capture_operation(ttnn, mesh, operation)
            output_bindings = {name: addresses(ttnn, value) for name, value in output.items()}
            for case in (1, 0):
                progress('replay_' + str(case))
                update(case)
                ttnn.execute_trace(mesh, trace, blocking=True)
                if {name: addresses(ttnn, value) for name, value in output.items()} != output_bindings:
                    raise AssertionError('Captured outputs moved')
                audit(output, 'replay', case)
            if any(eager[0, chip] == eager[1, chip] for chip in range(2)):
                raise AssertionError('Frontier changes must alter outputs on both chips')
            if len(report['checks']) != 72 or len(report['fixture_controls']) != 8:
                raise AssertionError('Complete context matrix required')
            validate_build(root, directory, REPORT, simulator)
        if digest(root / HEADER) != kernel['source_before']:
            raise ValueError('Decode kernel was not restored')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = sources()
            if report['sources_after'] != report['sources']:
                raise ValueError('Context probe source changed')
            report['closed_cleanly'] = True
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
