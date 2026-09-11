"""Precise native SDPA at fixed 4384 capacity and 31 queries; simulator only, no TG qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_t32_attention import append_queries
from dspark_t32_inputs import fixed_mask, validate_fixed_mask
from dspark_t32_inputs import geometry
from dspark_t32_attention import execute
from native_draft_sdpa import run_precise_probe
from dspark_hardware_gate import digest
from dspark_projection import tensor_digest
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from sim_memory_budget import require_clean
from t32_ci_runtime import fingerprints, snapshot


SPEC = importlib.util.spec_from_file_location('fixed_full_attention', Path(__file__).with_name('dspark-full-attention-probe.py'))
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)
SOURCES = tuple(sorted(set(FULL.SOURCES + ('dspark-t32-draft-attention-probe.py', 'dspark_t32_inputs.py', 'dspark_t32_attention.py',
    'dspark_native_full_attention.py', 'draft_attention.py', 'native_draft_sdpa.py', 't32_ci_runtime.py'))))
CAPACITY, PROPOSALS = 4384, 31
POSITIONS = (4096, 4109)
REPLAYS = (1, 0)
NAMES = ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')
COUNTS = dict(eager_checks=4, replay_checks=4, input_checks=48, layout_checks=16, fixture_controls=8, stale_controls=2)


def source_hashes():
    return {name: digest(Path(__file__).parent / name) for name in SOURCES}


def numerical_details(actual, golden):
    import torch

    difference = (actual.float() - golden).abs()
    failed = ~torch.isclose(actual.float(), golden, rtol=.01, atol=.01)
    coordinates = failed.nonzero()[:16].tolist()
    return dict(actual_shape=list(actual.shape), expected_shape=list(golden.shape),
        actual_dtype=str(actual.dtype), nonfinite=int((~torch.isfinite(actual)).sum()),
        failed_by_row=failed.sum(dim=(0, 1, 3)).tolist(),
        max_abs_by_row=difference.amax(dim=(0, 1, 3)).tolist(),
        failures=[dict(index=index, actual=float(actual[tuple(index)]),
                       expected=float(golden[tuple(index)]),
                       absolute_error=float(difference[tuple(index)]),
                       allowed_error=float(.01 + .01 * golden[tuple(index)].abs()))
                  for index in coordinates])


def fixtures():
    import torch

    generator = torch.Generator().manual_seed(383928)
    query = (.125 + .125 * torch.rand(2, 16, 32, 128, generator=generator)).bfloat16()
    key = (.125 * torch.randn(2, 4, CAPACITY, 128, generator=generator)).bfloat16()
    value = torch.randn(2, 4, CAPACITY, 128, generator=generator).bfloat16()
    key[:, :, :2048] -= 3
    key[:, :, 2048:4096] += 2
    key[:, :, 0] = 6
    value[:, :, 0] = 64
    query_key = (.125 * torch.randn(2, 4, 32, 128, generator=generator) + 4).bfloat16()
    query_value = torch.randn(2, 4, 32, 128, generator=generator).bfloat16()
    query_key[:, :, PROPOSALS - 1] = 7
    query_value[:, :, PROPOSALS - 1] = -64
    result = []
    for case, position in enumerate(POSITIONS):
        values = dict(query=query.clone(), history_key=key.clone(), history_value=value.clone(),
            query_key=query_key.clone(), query_value=query_value.clone(), mask=fixed_mask(position, CAPACITY, PROPOSALS))
        if case:
            values['history_key'][:, :, 4096:position] = 9
            values['history_value'][:, :, 4096:position] = 128
        for name in ('history_key', 'history_value'):
            values[name][:, :, position:] = 8192
        for name in ('query_key', 'query_value'):
            values[name][:, :, PROPOSALS:] = -8192
        result.append(values)
    return result


def joined(values, name):
    import torch

    complete = torch.cat((values['history_' + name], values['query_' + name][:, :, :PROPOSALS]), dim=2)
    return torch.nn.functional.pad(complete, (0, 0, 0, geometry(CAPACITY, PROPOSALS)[-1][1] - complete.shape[2]))


def reference(values, chip):
    import torch

    return torch.nn.functional.scaled_dot_product_attention(values['query'][chip:chip + 1].float(),
        joined(values, 'key')[chip:chip + 1].float().repeat_interleave(4, dim=1),
        joined(values, 'value')[chip:chip + 1].float().repeat_interleave(4, dim=1),
        attn_mask=values['mask'].float(), is_causal=False)


def controls(patterns, expected):
    import torch

    records = []
    for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update'):
        case = int(name == 'frontier_update')
        altered = {key: value.clone() for key, value in patterns[case].items()}
        if name == 'gap_poison':
            for key in ('history_key', 'history_value'):
                altered[key][:, :, POSITIONS[case]:] = -8192
        elif name == 'frontier_update':
            altered['mask'] = patterns[0]['mask'].clone()
        else:
            altered['mask'][:, :, :PROPOSALS, 0 if name == 'oldest' else CAPACITY + PROPOSALS - 1] = float('-inf')
        for chip in range(2):
            value = reference(altered, chip)
            detected = torch.equal(value, expected[case][chip]) if name == 'gap_poison' else not torch.allclose(
                value, expected[case][chip], rtol=.01, atol=.01)
            records.append(dict(name=name, case=case, chip=chip, detected=bool(detected)))
            if not detected:
                raise AssertionError('Fixed-storage fixture must detect missing history, proposal keys and frontier updates')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-chunk-size', type=int, choices=(32, 64), default=64)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    kernel_audit = run_precise_probe(__file__)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF') != '1'
            or os.environ.get('QWEN_SIM_ONLY') != '1'
            or os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))):
        raise ValueError('Fresh bounded two-chip simulator and owned precise native runtime required')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        positions=POSITIONS, capacity=CAPACITY, proposal_rows=PROPOSALS, chunks=geometry(CAPACITY, PROPOSALS),
        sources=source_hashes(), native_sources=fingerprints(root),
        kernel_audit=kernel_audit, key_chunk_size=options.key_chunk_size, resources_before=snapshot(),
        numerical_tolerances=dict(rtol=.01, atol=.01), target_integrated=False, committed_tg=None,
        precise_reciprocal=os.environ.get('QWEN_T32_PRECISE_RECIP') == '1',
        **{name: [] for name in COUNTS})
    owned, transient = [], []
    mesh = trace = None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        patterns = fixtures()
        for position, values in zip(POSITIONS, patterns, strict=True):
            validate_fixed_mask(values['mask'], position, CAPACITY, PROPOSALS)
        expected = [[reference(values, chip) for chip in range(2)] for values in patterns]
        report['fixture_controls'] = controls(patterns, expected)
        progress('open')
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

        inputs = {name: upload(patterns[0][name], name) for name in NAMES}
        payloads = [{name: upload(values[name], name, False) for name in NAMES} for values in patterns]
        bindings = {name: addresses(ttnn, value) for name, value in inputs.items()}
        eager = {}

        def update(case):
            for name in NAMES:
                ttnn.copy_host_to_device_tensor(payloads[case][name], inputs[name])
            ttnn.synchronize_device(mesh)

        def retain(value):
            transient.append(value)
            return value

        def run():
            keys = append_queries(ttnn, inputs['history_key'], inputs['query_key'], retain,
                position=CAPACITY, proposals=PROPOSALS)
            values = append_queries(ttnn, inputs['history_value'], inputs['query_value'], retain,
                position=CAPACITY, proposals=PROPOSALS)
            attention = execute(ttnn, mesh, inputs['query'], keys, values, inputs['mask'], transient,
                context_rows=CAPACITY, proposals=PROPOSALS, mask_validated=True,
                key_chunk_size=options.key_chunk_size)
            return dict(attention=attention, key=keys, value=values)

        def audit(output, mode, ordinal, case):
            if {name: addresses(ttnn, value) for name, value in inputs.items()} != bindings:
                raise AssertionError('Fixed-storage input addresses changed')
            for chip in range(2):
                actual = ttnn.to_torch(ttnn.get_device_tensors(output['attention'])[chip]).clone()
                golden = expected[case][chip]
                close = torch.isclose(actual.float(), golden, rtol=.01, atol=.01)
                checksum = tensor_digest(actual)
                if mode == 'eager':
                    eager[case, chip] = checksum
                exact = mode == 'eager' or checksum == eager[case, chip]
                passed = (actual.shape == golden.shape and actual.dtype == torch.bfloat16
                    and bool(torch.isfinite(actual).all()) and bool(close.all()) and exact)
                report[mode + '_checks'].append(dict(ordinal=ordinal, case=case, chip=chip, passed=passed,
                    replay_exact=exact if mode == 'replay' else None, numerical_close=bool(close.all()),
                    sha256=checksum, expected_sha256=tensor_digest(golden), failed_elements=int((~close).sum()),
                    max_abs=float((actual.float() - golden).abs().max()),
                    diagnostics=numerical_details(actual, golden)))
                if not passed:
                    raise AssertionError('Fixed-storage attention fails retained FP32 accuracy or exact replay')
                for name in ('key', 'value'):
                    actual_layout = ttnn.to_torch(ttnn.get_device_tensors(output[name])[chip])
                    reference_layout = joined(patterns[case], name)[chip:chip + 1]
                    identical = torch.equal(actual_layout, reference_layout)
                    report['layout_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, chip=chip, name=name,
                        passed=identical, sha256=tensor_digest(actual_layout), expected_sha256=tensor_digest(reference_layout)))
                    if not identical:
                        raise AssertionError('Proposal keys must follow physical capacity without exposing the gap')
                for name, value in inputs.items():
                    original = patterns[case][name] if name == 'mask' else patterns[case][name][chip:chip + 1]
                    exact_input = torch.equal(ttnn.to_torch(ttnn.get_device_tensors(value)[chip]), original)
                    report['input_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, chip=chip, name=name, exact=exact_input))
                    if not exact_input:
                        raise AssertionError('Fixed-storage attention mutates a borrowed input')

        for case in range(2):
            progress(f'eager_{case}')
            update(case)
            output = run()
            ttnn.synchronize_device(mesh)
            audit(output, 'eager', case, case)
            release_owned(ttnn, transient)
            transient.clear()
        update(0)
        progress('capture')
        trace, output = capture_operation(ttnn, mesh, run)
        output_bindings = {name: addresses(ttnn, value) for name, value in output.items()}
        for ordinal, case in enumerate(REPLAYS):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh, trace, blocking=True)
            if {name: addresses(ttnn, value) for name, value in output.items()} != output_bindings:
                raise AssertionError('Captured fixed-storage output addresses changed')
            audit(output, 'replay', ordinal, case)
        for chip in range(2):
            detected = eager[0, chip] != eager[1, chip]
            report['stale_controls'].append(dict(chip=chip, detected=detected))
            if not detected:
                raise AssertionError('Changed committed frontier must change the full attention output')
        if {name: len(report[name]) for name in COUNTS} != COUNTS:
            raise AssertionError('Incomplete fixed-storage numerical and lifetime matrix')
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
            report['sources_after'], report['native_sources_after'] = source_hashes(), fingerprints(root)
            if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
                raise ValueError('Fixed-storage source or native runtime changed')
            report['resources_after'] = snapshot()
            require_clean(report['resources_before'], report['resources_after'])
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
