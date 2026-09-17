"""Targeted CTX4096 / fifteen-query full-history attention gate; no model weights or TG measurement."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_full_attention import POLICY, execute, full_mask, geometry, validate_mask
from dspark_hardware_gate import digest
from dspark_projection import tensor_digest
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from sim_memory_budget import require_clean, snapshot


SPEC = importlib.util.spec_from_file_location('full_attention_noise_probe', Path(__file__).with_name('dspark-noise-probe.py'))
NOISE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NOISE)
SOURCES = tuple(sorted(set(NOISE.SOURCES + ('dspark-full-attention-probe.py', 'dspark_full_attention.py',
    'draft_dot.py', 'draft_dot_io.cpp', 'draft_dot_compute.cpp', 'draft_row_sum.py',
    'draft_row_sum_io.cpp', 'draft_row_sum_compute.cpp', 'sim_memory_budget.py'))))
CONTEXT, PROPOSALS = 4096, 15
REPLAYS = (0, 1, 0)
COUNTS = dict(eager_checks=4, replay_checks=6, input_checks=40, fixture_controls=6, stale_controls=2)


def source_hashes():
    return {name:digest(Path(__file__).parent / name) for name in SOURCES}


def fixtures():
    import torch

    generator = torch.Generator().manual_seed(383927)
    padded_keys = geometry(CONTEXT, PROPOSALS)[-1][1]
    query = (.125 + .125 * torch.rand(2, 16, 32, 128, generator=generator)).bfloat16()
    key = (.125 * torch.randn(2, 4, padded_keys, 128, generator=generator)).bfloat16()
    value = torch.randn(2, 4, padded_keys, 128, generator=generator).bfloat16()
    key[:, :, :2048] -= 3
    key[:, :, 2048:4096] += 2
    key[:, :, CONTEXT:CONTEXT + PROPOSALS] += 4
    key[:, :, 0] = 6
    key[:, :, CONTEXT + PROPOSALS - 1] = 7
    value[:, :, 0] = 64
    value[:, :, CONTEXT + PROPOSALS - 1] = -64
    for tensor in (key, value):
        tensor[:, :, CONTEXT + PROPOSALS:] = 8192
    original = [query, key, value, full_mask(CONTEXT, PROPOSALS)]
    changed = [tensor.clone() for tensor in original]
    changed[1][:, :, 0] = 10
    changed[2][:, :, 0] = 128
    return original, changed


def reference(values, chip):
    import torch

    query, key, value, mask = values
    return torch.nn.functional.scaled_dot_product_attention(query[chip:chip + 1].float(),
        key[chip:chip + 1].float().repeat_interleave(4, dim=1),
        value[chip:chip + 1].float().repeat_interleave(4, dim=1), attn_mask=mask.float(), is_causal=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF') != '1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE', 'QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh bounded shared-BDF simulator gate using the unchanged native runtime required')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', policy=POLICY, scope=__doc__,
        context_rows=CONTEXT, proposal_rows=PROPOSALS, chunks=geometry(CONTEXT, PROPOSALS),
        numerical_tolerances=dict(rtol=.01, atol=.01), sources=source_hashes(), native_sources=NOISE.fingerprints(root),
        resources_before=snapshot(bounded=True), target_integrated=False, eligible_for_serving=False,
        committed_tg=None, physical_fabric_tested=False, **{name:[] for name in COUNTS})
    owned, transient = [], []
    mesh = trace = None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        patterns = fixtures()
        for values in patterns:
            validate_mask(values[3], CONTEXT, PROPOSALS)
        expected = [[reference(values, chip) for chip in range(2)] for values in patterns]
        for name in ('oldest', 'last_proposal', 'padding'):
            altered = [value.clone() for value in patterns[0]]
            if name == 'padding':
                for value in altered[1:3]:
                    value[:, :, CONTEXT + PROPOSALS:] = -8192
            else:
                altered[3][:, :, :PROPOSALS, 0 if name == 'oldest' else CONTEXT + PROPOSALS - 1] = float('-inf')
            for chip in range(2):
                actual = reference(altered, chip)
                detected = torch.equal(actual, expected[0][chip]) if name == 'padding' else not torch.allclose(actual, expected[0][chip], rtol=.01, atol=.01)
                report['fixture_controls'].append(dict(name=name, chip=chip, detected=bool(detected)))
                if not detected:
                    raise AssertionError('Full-history fixture does not expose missing history/proposals or masked padding')
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=536870912)
        mesh.enable_program_cache()

        def upload(value, index, device=True):
            tensor = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if index == 3 else ttnn.ShardTensorToMesh(mesh, dim=0),
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(tensor)
            return tensor

        inputs = [upload(value, index) for index, value in enumerate(patterns[0])]
        host_inputs = [[upload(value, index, False) for index, value in enumerate(values)] for values in patterns]
        bindings = [addresses(ttnn, value) for value in inputs]

        def host(value, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        def update(case):
            for source, destination in zip(host_inputs[case], inputs, strict=True):
                ttnn.copy_host_to_device_tensor(source, destination)
            ttnn.synchronize_device(mesh)

        def run():
            return execute(ttnn, mesh, *inputs, transient, context_rows=CONTEXT, proposals=PROPOSALS, mask_validated=True)

        eager = {}

        def audit(output, mode, ordinal, case):
            if [addresses(ttnn, value) for value in inputs] != bindings:
                raise AssertionError('Borrowed full-history input binding changed')
            for chip in range(2):
                actual = host(output, chip)
                golden = expected[case][chip]
                close = torch.isclose(actual.float(), golden, rtol=.01, atol=.01)
                valid = tuple(actual.shape) == (1, 16, 32, 128) and actual.dtype == torch.bfloat16 and bool(torch.isfinite(actual).all())
                checksum = tensor_digest(actual)
                if mode == 'eager':
                    eager[case, chip] = checksum
                exact = mode == 'eager' or checksum == eager[case, chip]
                row = dict(ordinal=ordinal, case=case, chip=chip, passed=valid and bool(close.all()) and exact,
                    numerical_close=bool(close.all()), replay_exact=exact if mode == 'replay' else None,
                    sha256=checksum, expected_sha256=tensor_digest(golden),
                    failed_elements=int((~close).sum()), max_abs=float((actual.float() - golden).abs().max()))
                report[mode + '_checks'].append(row)
                if not row['passed']:
                    raise AssertionError('Full-history attention fails its retained FP32 numerical or exact replay gate')
                for index, value in enumerate(inputs):
                    original = patterns[case][index] if index == 3 else patterns[case][index][chip:chip + 1]
                    exact_input = tensor_digest(host(value, chip)) == tensor_digest(original)
                    report['input_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, chip=chip, index=index, exact=exact_input))
                    if not exact_input:
                        raise AssertionError('Borrowed full-history input changed')

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
        output_binding = addresses(ttnn, output)
        for ordinal, case in enumerate(REPLAYS):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh, trace, blocking=True)
            if addresses(ttnn, output) != output_binding:
                raise AssertionError('Captured full-history output binding changed')
            audit(output, 'replay', ordinal, case)
        for chip in range(2):
            detected = eager[0, chip] != eager[1, chip]
            report['stale_controls'].append(dict(chip=chip, detected=detected))
            if not detected:
                raise AssertionError('Oldest-history changes did not reach the full-context output')
        if {name:len(report[name]) for name in COUNTS} != COUNTS:
            raise AssertionError('Incomplete full-history numerical/replay matrix')
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
            report['sources_after'], report['native_sources_after'] = source_hashes(), NOISE.fingerprints(root)
            if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
                raise ValueError('Full-history source or native runtime changed')
            report['resources_after'] = snapshot(bounded=True)
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
