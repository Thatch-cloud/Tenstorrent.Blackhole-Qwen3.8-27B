"""Approximate native proposal attention: mask isolation and replay, not an exact target replacement."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_attention import draft_attention_mask
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from proposal_native_attention import POLICY, attention, numerical_difference, validate_mask
from proposal_native_attention_gate import FIXTURE_SHA256, SOURCES, hashes, native_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(31, 2048), required=True)
    parser.add_argument('--fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if bool(options.fixture) != (options.context == 31):
        parser.error('Short context requires the pinned learned fixture; 2048 uses synthetic operands')
    import torch
    import ttnn

    root = os.environ['TT_METAL_HOME']
    report = dict(passed=False, closed_cleanly=False, backend='simulator', policy=POLICY,
        context=options.context, target_integrated=False, accuracy_qualified=False, scope=__doc__,
        sources=hashes(Path(__file__).parent, SOURCES), native_sources=native_hashes(root),
        fixture_sha256=FIXTURE_SHA256 if options.fixture else None,
        operand_scope='Learned layer0 rank0 replicated on both chips plus changed synthetic inputs' if options.fixture
            else 'Synthetic 2048-row history on both chips; not learned model integration',
        eager_checks=[], replay_checks=[], input_checks=[], negative_controls=[], masked_input_checks=[])
    patterns = []
    for seed in (3827, 3828):
        generator = torch.Generator().manual_seed(seed)
        mask = draft_attention_mask(options.context)
        patterns.append([*(torch.randn(shape, generator=generator).bfloat16() for shape in
            ((1, 16, 32, 128), (1, 4, mask.shape[-1], 128), (1, 4, mask.shape[-1], 128))), mask])
    if options.fixture:
        data = options.fixture.read_bytes()
        if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
            raise ValueError('Pinned learned operand integrity mismatch')
        fixture = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
        patterns[0] = [fixture[name] for name in ('query', 'key', 'value', 'mask')]
    patterns[1][3][..., :8, :5] = float('-inf')
    masked = [value.clone() for value in patterns[0]]
    blocked = torch.isneginf(masked[3][0, 0]).all(dim=0)
    if not blocked.any():
        raise ValueError('The fixture must include keys masked from every live and padded query')
    for tensor in masked[1:3]:
        tensor[:, :, blocked, :] = (tensor[:, :, blocked, :].float() * -3.0 + 17.0).bfloat16()
    patterns.append(masked)
    for values in patterns:
        validate_mask(values[3])
        if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in values[:3]):
            raise ValueError('Finite BF16 query, key and value operands required')
    mesh, trace = None, None
    persistent, transient = [], []

    def progress(stage):
        report['stage'] = stage
        print(json.dumps(dict(stage=stage, context=options.context)), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        payloads = [[ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper) for value in values] for values in patterns]
        persistent.extend(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper) for value in patterns[0])
        bindings = [addresses(ttnn, value) for value in persistent]

        def host(value, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        def upload(pattern):
            validate_mask(patterns[pattern][3])
            for payload, destination in zip(payloads[pattern], persistent, strict=True):
                ttnn.copy_host_to_device_tensor(payload, destination)
            ttnn.synchronize_device(mesh)

        def execute():
            output = attention(ttnn, *persistent, mask_validated=True)
            transient.append(output)
            return output

        def audit_inputs(phase, ordinal, pattern):
            for tensor, (value, expected) in enumerate(zip(persistent, patterns[pattern], strict=True)):
                for chip in range(2):
                    if not torch.equal(host(value, chip), expected):
                        raise AssertionError('Native proposal attention changed borrowed inputs')
                    report['input_checks'].append(dict(phase=phase, pattern=ordinal, tensor=tensor, chip=chip, unchanged=True))

        references = []
        for pattern, values in enumerate(patterns):
            upload(pattern)
            output = execute()
            ttnn.synchronize_device(mesh)
            query, key, value, mask = values
            expected = torch.nn.functional.scaled_dot_product_attention(query.float(),
                key.float().repeat_interleave(4, dim=1), value.float().repeat_interleave(4, dim=1),
                attn_mask=mask.float(), is_causal=False)
            actual = [host(output, chip) for chip in range(2)]
            for chip in range(2):
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, finite_all_rows=True,
                    numerical_difference=numerical_difference(actual[chip], expected)))
                if pattern == 2:
                    if not torch.equal(actual[chip], references[0][chip]):
                        raise AssertionError('Completely masked key/value changes influenced native attention')
                    report['masked_input_checks'].append(dict(phase=0, chip=chip, masked_changes_ignored=True))
            references.append(actual)
            audit_inputs(0, pattern, pattern)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_{pattern}_complete')
        upload(0)
        trace, output = capture_operation(ttnn, mesh, execute)
        output_bindings = addresses(ttnn, output)
        for repetition, pattern in enumerate((0, 1, 2, 0)):
            upload(pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if [addresses(ttnn, value) for value in persistent] != bindings or addresses(ttnn, output) != output_bindings:
                raise AssertionError('Captured proposal bindings changed')
            for chip in range(2):
                actual = host(output, chip)
                if not torch.equal(actual, references[pattern][chip]):
                    raise AssertionError('Native proposal replay differs from its own eager result')
                report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, chip=chip,
                    exact_eager_all_rows=True, bindings_stable=True))
                if pattern == 2:
                    if not torch.equal(actual, references[0][chip]):
                        raise AssertionError('Masked changes influenced captured native attention')
                    report['masked_input_checks'].append(dict(phase=1, chip=chip, masked_changes_ignored=True))
            audit_inputs(1, repetition, pattern)
            if repetition == 0:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for chip in range(2):
                    actual = host(output, chip)
                    if not torch.equal(actual, references[0][chip]) or torch.equal(actual, references[1][chip]):
                        raise AssertionError('Stale-input negative control failed')
                    report['negative_controls'].append(dict(chip=chip, stale_detected=True))
            progress(f'replay_{repetition}_complete')
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
                release_owned(ttnn, persistent)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
            report['native_sources_after'] = native_hashes(root)
            if report['native_sources_after'] != report['native_sources']:
                raise RuntimeError('Native sources changed during proposal simulation')
            progress('complete' if report['passed'] else 'failed')
        finally:
            if not report['closed_cleanly'] or report.get('native_sources_after') != report['native_sources']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
