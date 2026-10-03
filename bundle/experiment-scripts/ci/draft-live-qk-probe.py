"""Live-query QK and complete attention gate; optional hardware component timing, not PP/CTX/TG."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time

from attention_batch import capture_operation
from draft_attention import draft_attention_mask
from draft_dot import fused_dot
from draft_live_qk import live_qk, validate_live_qk_mask
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from live_qk_gate import native_hashes, qualify_simulator, source_hashes, timing_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(31, 2048), default=31)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--simulator-report', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if bool(options.fixture) != bool(options.sha256) or (options.fixture and options.context != 31):
        parser.error('Captured operands require a pinned SHA and CTX31')
    if options.hardware and (not options.attention or options.fixture or not options.simulator_report):
        parser.error('Hardware requires complete attention and a simulator report; no learned hardware claim')
    if options.simulator_report and not options.hardware:
        parser.error('Simulator report input is only for hardware qualification')
    if options.hardware:
        qualify_simulator(json.loads(options.simulator_report.read_text()), options.context,
            source_hashes(), native_hashes(os.environ['TT_METAL_HOME']))
    run(options)


def run(options):
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, context=options.context, attention=options.attention,
        fixture_sha256=options.sha256, eager_checks=[], replay_checks=[], negative_controls=[],
        sources=source_hashes(), native_sources=native_hashes(os.environ['TT_METAL_HOME']),
        backend='hardware' if options.hardware else 'simulator', timing_samples=[],
        timing_scope='Captured complete attention on uploaded operands; excludes uploads/readback, not learned layer or PP/CTX/TG')
    if options.simulator_report:
        report['simulator_report_sha256'] = hashlib.sha256(options.simulator_report.read_bytes()).hexdigest()
    patterns = []
    for seed in (3827, 3828):
        generator = torch.Generator().manual_seed(seed)
        mask = draft_attention_mask(options.context)
        values = [torch.randn(shape, generator=generator).bfloat16() for shape in
            ((1, 16, 32, 128), (1, 4, mask.shape[-1], 128), (1, 4, mask.shape[-1], 128))]
        patterns.append([*values, mask])
    if options.fixture:
        data = options.fixture.read_bytes()
        if hashlib.sha256(data).hexdigest() != options.sha256:
            raise ValueError('Captured operand integrity mismatch')
        fixture = torch.load(BytesIO(data), map_location='cpu', weights_only=True)
        values = [fixture[name] for name in ('query', 'key', 'value', 'mask')]
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or value.shape != expected.shape
                for value, expected in zip(values, patterns[0], strict=True)):
            raise ValueError('Captured learned operand dtype or shape mismatch')
        patterns[0] = values
        report['fixture_scope'] = 'Saved layer0 rank0 operands replicated on two simulator chips; not original rank1'
    for query, key, value, mask in patterns:
        validate_live_qk_mask(mask)
        if any(not torch.isfinite(tensor).all() for tensor in (query, key, value)):
            raise ValueError('Finite Q/K/V required')
    patterns[1][3][..., :8, :5] = float('-inf')
    patterns = [[query.float(), key.float().repeat_interleave(4, dim=1),
        value.float().repeat_interleave(4, dim=1), mask.float()] for query, key, value, mask in patterns]
    mesh = None
    persistent, traces, transient = [], [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def host(tensor, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        payloads = [[ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in values] for values in patterns]
        persistent.extend(ttnn.from_torch(value, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper) for value in patterns[0])
        input_addresses = [addresses(ttnn, value) for value in persistent]

        def upload(pattern):
            for source, destination in zip(payloads[pattern], persistent, strict=True):
                ttnn.copy_host_to_device_tensor(source, destination)
            ttnn.synchronize_device(mesh)

        def retain(value):
            transient.append(value)
            return value

        def execute(candidate):
            scores = live_qk(mesh, *persistent[:2], transient) if candidate else fused_dot(
                mesh, *persistent[:2], transient, cache_tiles=True)
            result = dict(scores=scores)
            if options.attention:
                from draft_row_sum import row_sum

                scaled = retain(ttnn.multiply(scores, 128 ** -.5, dtype=ttnn.float32))
                masked = retain(ttnn.add(scaled, persistent[3], dtype=ttnn.float32))
                maximum = retain(ttnn.max(masked, dim=-1, keepdim=True))
                centered = retain(ttnn.subtract(masked, maximum, dtype=ttnn.float32))
                exponentials = retain(ttnn.exp(centered, fast_and_approximate_mode=False))
                total = row_sum(mesh, exponentials, transient)
                inverse = retain(ttnn.reciprocal(total))
                probabilities = retain(ttnn.multiply(exponentials, inverse, dtype=ttnn.float32))
                transposed = retain(ttnn.transpose(persistent[2], -1, -2))
                output = fused_dot(mesh, probabilities, transposed, transient, cache_tiles=True)
                result.update(probabilities=probabilities, output=output)
            return result

        references = []
        for pattern, values in enumerate(patterns):
            upload(pattern)
            outputs = [execute(candidate) for candidate in (False, True)]
            ttnn.synchronize_device(mesh)
            results = [{name: [host(tensor, chip).clone() for chip in range(2)] for name, tensor in output.items()}
                for output in outputs]
            expected = (values[0].double() @ values[1].double().transpose(-1, -2)).float()
            for chip in range(2):
                control, candidate = [result['scores'][chip] for result in results]
                torch.testing.assert_close(control, expected, rtol=1e-5, atol=1e-4)
                if not torch.equal(control[..., :8, :], candidate[..., :8, :]) or candidate[..., 8:, :].count_nonzero():
                    raise AssertionError('Live QK rows changed or padding was not zeroed')
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, component='scores', exact_live=True, padding_zero=True))
                for name in ('probabilities', 'output') if options.attention else ():
                    if not torch.equal(results[0][name][chip], results[1][name][chip]):
                        raise AssertionError(f'Complete {name} changed, including padding')
                    report['eager_checks'].append(dict(pattern=pattern, chip=chip, component=name, exact_all_rows=True))
                if options.attention:
                    expected_attention = torch.nn.functional.scaled_dot_product_attention(*values[:3],
                        attn_mask=values[3], is_causal=False)
                    torch.testing.assert_close(results[1]['output'][chip][..., :8, :],
                        expected_attention[..., :8, :], rtol=.01, atol=.01)
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
        output_addresses = [{name: addresses(ttnn, value) for name, value in output.items()} for output in outputs]
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            for arm, trace in enumerate(traces):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if [addresses(ttnn, value) for value in persistent] != input_addresses:
                    raise AssertionError('Captured input bindings changed')
                for name, value in outputs[arm].items():
                    if addresses(ttnn, value) != output_addresses[arm][name]:
                        raise AssertionError('Captured output bindings changed')
                    for chip in range(2):
                        if not torch.equal(host(value, chip), references[pattern][arm][name][chip]):
                            raise AssertionError('Changed-input replay differs from eager reference')
                        report['replay_checks'].append(dict(repetition=repetition, pattern=pattern,
                            arm=arm, chip=chip, component=name, exact=True))
                for chip in range(2):
                    if any(not torch.equal(host(value, chip), expected)
                            for value, expected in zip(persistent, patterns[pattern], strict=True)):
                        raise AssertionError('Kernel changed a borrowed input')
                if repetition == 0:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip in range(2):
                        actual = host(outputs[arm]['scores'], chip)
                        if not torch.equal(actual, references[0][arm]['scores'][chip]) or torch.equal(actual, references[1][arm]['scores'][chip]):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(arm=arm, chip=chip, stale_detected=True))
            progress(f'replay_{repetition}_complete')
        if options.hardware:
            for pattern in range(2):
                upload(pattern)
                for trace in traces:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for block in range(3):
                    for order, arm in enumerate((0, 1, 1, 0)):
                        ttnn.synchronize_device(mesh)
                        started = time.perf_counter()
                        for replay in range(50):
                            ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        elapsed_ms = (time.perf_counter() - started) * 1000 / 50
                        if ([addresses(ttnn, value) for value in persistent] != input_addresses
                                or any(addresses(ttnn, value) != output_addresses[arm][name]
                                    for name, value in outputs[arm].items())):
                            raise AssertionError('Timed trace bindings changed')
                        for chip in range(2):
                            if any(not torch.equal(host(value, chip), references[pattern][arm][name][chip])
                                    for name, value in outputs[arm].items()):
                                raise AssertionError('Timed output changed')
                            if any(not torch.equal(host(value, chip), expected)
                                    for value, expected in zip(persistent, patterns[pattern], strict=True)):
                                raise AssertionError('Timed trace changed a borrowed input')
                        report['timing_samples'].append(dict(pattern=pattern, block=block, order=order,
                            arm=arm, replays=50, ms=elapsed_ms, outputs_exact=True,
                            inputs_unchanged=True, bindings_stable=True))
                    progress(f'timing_{pattern}_{block}_complete')
            report['timing_summary'] = timing_summary(report['timing_samples'])
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
