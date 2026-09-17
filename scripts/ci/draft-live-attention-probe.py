"""Simulator-only BF16 GQA-to-FP32 attention integration; all32 rows, not PP/CTX/TG."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_attention import composed_draft_attention, draft_attention_mask
from draft_live_attention import live_attention
from draft_live_qk import validate_live_qk_mask
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from live_attention_gate import FIXTURE_SHA256, source_hashes, native_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(31, 2048), required=True)
    parser.add_argument('--fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if bool(options.fixture) != (options.context == 31):
        parser.error('Short integration requires the pinned learned fixture; long integration uses synthetic operands')
    import torch
    import ttnn

    report = dict(passed=False, backend='simulator', scope=__doc__, context=options.context,
        sources=source_hashes(), native_sources=native_hashes(os.environ['TT_METAL_HOME']),
        fixture_sha256=FIXTURE_SHA256 if options.fixture else None,
        operand_scope='Saved layer0 rank0 replicated on both chips plus changed synthetic inputs' if options.fixture
            else 'Synthetic 2048-row history, not learned rank1 or model integration',
        eager_checks=[], replay_checks=[], negative_controls=[])
    patterns = []
    for seed in (3827, 3828):
        generator = torch.Generator().manual_seed(seed)
        mask = draft_attention_mask(options.context)
        patterns.append([*(torch.randn(shape, generator=generator).bfloat16() for shape in
            ((1, 16, 32, 128), (1, 4, mask.shape[-1], 128), (1, 4, mask.shape[-1], 128))), mask])
    if options.fixture:
        data = options.fixture.read_bytes()
        if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
            raise ValueError('Learned operand integrity mismatch')
        fixture = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
        patterns[0] = [fixture[name] for name in ('query', 'key', 'value', 'mask')]
    patterns[1][3][..., :8, :5] = float('-inf')
    for values in patterns:
        validate_live_qk_mask(values[3])
        if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in values[:3]):
            raise ValueError('Finite BF16 operands required')
    mesh = None
    persistent, transient, traces = [], [], []
    def progress(stage):
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
            validate_live_qk_mask(patterns[pattern][3])
            for payload, destination in zip(payloads[pattern], persistent, strict=True):
                ttnn.copy_host_to_device_tensor(payload, destination)
            ttnn.synchronize_device(mesh)
        def execute(candidate):
            if candidate:
                return live_attention(ttnn, mesh, *persistent, trace_owned=transient, mask_validated=True)
            return composed_draft_attention(ttnn, mesh, *persistent, explicit_softmax=True,
                fused_row_sum=True, fused_dots=True, cache_dot_tiles=True, trace_owned=transient)
        references = []
        for pattern in range(2):
            upload(pattern)
            outputs = [execute(candidate) for candidate in (False, True)]
            ttnn.synchronize_device(mesh)
            results = [[host(output, chip) for chip in range(2)] for output in outputs]
            for chip in range(2):
                if not torch.equal(results[0][chip], results[1][chip]):
                    raise AssertionError('Integrated attention changed live or padded rows')
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, exact_all_rows=True))
            references.append(results)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_{pattern}_complete')
        upload(0)
        outputs = []
        for candidate in (False, True):
            trace, output = capture_operation(ttnn, mesh, lambda: execute(candidate))
            traces.append(trace)
            outputs.append(output)
        output_bindings = [addresses(ttnn, output) for output in outputs]
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            for arm, trace in enumerate(traces):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if ([addresses(ttnn, value) for value in persistent] != bindings
                        or [addresses(ttnn, value) for value in outputs] != output_bindings):
                    raise AssertionError('Captured bindings changed')
                for chip in range(2):
                    if not torch.equal(host(outputs[arm], chip), references[pattern][arm][chip]):
                        raise AssertionError('Changed-input replay differs from eager reference')
                    if any(not torch.equal(host(value, chip), expected)
                            for value, expected in zip(persistent, patterns[pattern], strict=True)):
                        raise AssertionError('Captured attention changed borrowed inputs')
                    report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, arm=arm,
                        chip=chip, exact_all_rows=True, inputs_unchanged=True, bindings_stable=True))
                if repetition == 0:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip in range(2):
                        actual = host(outputs[arm], chip)
                        if not torch.equal(actual, references[0][arm][chip]) or torch.equal(actual, references[1][arm][chip]):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(arm=arm, chip=chip, stale_detected=True))
            progress(f'replay_{repetition}_complete')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in reversed(traces):
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, persistent)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
