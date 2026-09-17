"""Simulator-only attention capture prerequisite; not a complete learned draft or timing test."""

import argparse
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_attention import composed_draft_attention, draft_attention_mask, draft_sdpa
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precise-native', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if not options.precise_native and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required; hardware promotion is not enabled')
    from native_draft_sdpa import run_precise_probe

    run(options, run_precise_probe(__file__) if options.precise_native else None)


def run(options, kernel_audit):
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, eager_checks=[], replay_checks=[], negative_controls=[],
        native_kernel=kernel_audit)
    mesh, trace = None, None
    persistent, transient = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def host(tensor, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if options.precise_native else ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        patterns = []
        for seed in (913, 914):
            generator = torch.Generator().manual_seed(seed)
            values = [torch.randn(shape, generator=generator).bfloat16() for shape in
                ((1, 16, 32, 128), (1, 4, 64, 128), (1, 4, 64, 128))]
            mask = draft_attention_mask(31)
            if seed == 914:
                mask[..., :8, :5] = float('-inf')
            patterns.append([*values, mask])
        payloads = [[ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper) for value in values] for values in patterns]
        persistent.extend(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            for value in patterns[0])
        original = [addresses(ttnn, value) for value in persistent]

        def upload(pattern):
            for source, destination in zip(payloads[pattern], persistent, strict=True):
                ttnn.copy_host_to_device_tensor(source, destination)
            ttnn.synchronize_device(mesh)

        def execute():
            if options.precise_native:
                output = draft_sdpa(ttnn, *persistent)
                transient.append(output)
                return output
            return composed_draft_attention(ttnn, mesh, *persistent, explicit_softmax=True,
                fused_row_sum=True, fused_dots=True, cache_dot_tiles=True, trace_owned=transient)

        references = []
        for pattern, values in enumerate(patterns):
            upload(pattern)
            output = execute()
            ttnn.synchronize_device(mesh)
            query, key, value, mask = values
            expected = torch.nn.functional.scaled_dot_product_attention(query.float(),
                key.float().repeat_interleave(4, dim=1), value.float().repeat_interleave(4, dim=1),
                attn_mask=mask.float(), is_causal=False)
            results = []
            for chip in range(2):
                actual = host(output, chip).clone()
                torch.testing.assert_close(actual[..., :8, :].float(), expected[..., :8, :], rtol=.01, atol=.01)
                results.append(actual)
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, reference_close=True))
            references.append(results)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_{pattern}_complete')
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Changing operands must change attention outputs')
        upload(0)
        trace, output = capture_operation(ttnn, mesh, execute)
        output_addresses = addresses(ttnn, output)
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if [addresses(ttnn, value) for value in persistent] != original or addresses(ttnn, output) != output_addresses:
                raise AssertionError('Captured input/output addresses moved')
            for chip in range(2):
                if not torch.equal(host(output, chip), references[pattern][chip]):
                    raise AssertionError('Captured attention differs from validated eager output')
                for device, expected in zip(persistent, patterns[pattern], strict=True):
                    if not torch.equal(host(device, chip), expected):
                        raise AssertionError('Attention mutated caller input')
                report['replay_checks'].append(dict(pattern=pattern, chip=chip, exact=True))
            if repetition == 0:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for chip in range(2):
                    actual = host(output, chip)
                    if not torch.equal(actual, references[0][chip]) or torch.equal(actual, references[1][chip]):
                        raise AssertionError('Missing-input-update negative control failed')
                    report['negative_controls'].append(dict(chip=chip, stale_input_detected=True))
            progress(f'replay_{repetition}_complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, transient)
            release_owned(ttnn, persistent)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = (len(report['eager_checks']) == 4 and len(report['replay_checks']) == 6
        and len(report['negative_controls']) == 2)
    progress('complete')


if __name__ == '__main__':
    main()
