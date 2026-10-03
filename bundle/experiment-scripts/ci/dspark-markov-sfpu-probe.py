"""Fused SFPU Markov score accuracy and replay gate; no complete draft trajectory, hardware timing or target claim."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_hardware_gate import digest
from dspark_markov_sfpu import execute
from dspark_projection import tensor_digest
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from sim_memory_budget import require_clean, snapshot


SPEC = importlib.util.spec_from_file_location('markov_sfpu_native', Path(__file__).with_name('dspark-full-attention-probe.py'))
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)
SOURCES = tuple(sorted(set(NATIVE.SOURCES + ('dspark-markov-sfpu-probe.py', 'dspark_markov_sfpu.py',
    'dspark_markov_sfpu_io.cpp', 'dspark_markov_sfpu_compute.cpp', 'dspark_markov_fixture.py'))))
OPERANDS_SHA256 = 'd050d3be174cf85dc3b87bbd0aa0323f6fbd308fa57b3ffd68eadbbd2d049d5c'
STEPS = (0, 6)


def patterns(operands=None, fixture=None):
    import torch

    if (operands is None) == (fixture is None):
        raise ValueError('Exactly one pinned historical operand capture or full learned fixture required')
    if operands is not None:
        if digest(operands) != OPERANDS_SHA256:
            raise ValueError('Unchanged historical failing operands required')
        saved = torch.load(operands, map_location='cpu', weights_only=True)
        latent = saved['latent'].clone().reshape(1, 1, 1, 256)
        weight = saved['weight'].T.contiguous().reshape(1, 1, 64, 256)
        base = saved['base'].clone().expand(1, 1, 7, 64).clone()
        base += torch.arange(7).reshape(1, 1, 7, 1) / 8
        following = -latent
        metadata = dict(scope='Historical failing 64-column subset plus sign-flipped perturbation',
            operands_sha256=OPERANDS_SHA256, vocabulary_offset=1312, complete_learned_vocabulary=False)
    else:
        from dspark_markov_fixture import load_fixture
        manifest, predecessor, successor = load_fixture(fixture)
        latent = predecessor[1596:1597].clone().reshape(1, 1, 1, 256)
        following = predecessor[1597:1598].clone().reshape(1, 1, 1, 256)
        weight = successor.reshape(1, 1, 248320, 256)
        base = torch.randn((1, 1, 7, 248320), generator=torch.Generator().manual_seed(38256)) / 8
        metadata = dict(scope='All learned successor rows and two learned predecessor embeddings',
            fixture=manifest, complete_learned_vocabulary=True)
    return [dict(latent=latent, weight=weight, base=base),
        dict(latent=following, weight=weight, base=base - .25)], metadata


def reference(values, step):
    return values['latent'].float() @ values['weight'].float().transpose(-1, -2) + values['base'][:, :, step:step + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--operands', type=Path)
    selection.add_argument('--fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF') != '1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY') != '1'):
        raise ValueError('Fresh bounded two-chip simulator gate required')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    source_hashes = lambda: {name: digest(Path(__file__).parent / name) for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=source_hashes(), native_sources=NATIVE.NOISE.fingerprints(root), resources_before=snapshot(bounded=True),
        numerical_tolerances=dict(rtol=1e-4, atol=1e-4), score_checks=[], token_checks=[], input_checks=[],
        stale_controls=[], hardware_qualified=False, target_integrated=False, committed_tg=None)
    mesh = trace = None
    owned, temporary = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        values, metadata = patterns(options.operands, options.fixture)
        expected = {(case, step): reference(pattern, step) for case, pattern in enumerate(values) for step in STEPS}
        policies = (1, 64) if options.operands else (110,)
        report.update(fixture=metadata, workers=policies, steps=STEPS, vocabulary=values[0]['base'].shape[-1])
        for step in STEPS:
            detected = not torch.allclose(expected[0, step], expected[1, step], rtol=1e-4, atol=1e-4)
            report['stale_controls'].append(dict(step=step, detected=detected))
            if not detected:
                raise AssertionError('Changed learned inputs must expose stale replay')
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=536870912)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        inputs = {}
        for name, value in values[0].items():
            inputs[name] = ttnn.from_torch(value, dtype=ttnn.float32 if name == 'base' else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(inputs[name])
        payloads = [{name: ttnn.from_torch(value, dtype=ttnn.float32 if name == 'base' else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for name, value in pattern.items() if name != 'weight'} for pattern in values]
        bindings = {name: addresses(ttnn, value) for name, value in inputs.items()}
        eager = {}

        def update(case):
            for name, payload in payloads[case].items():
                ttnn.copy_host_to_device_tensor(payload, inputs[name])
            ttnn.synchronize_device(mesh)

        def run():
            results = {}
            for workers in policies:
                for step in STEPS:
                    scores = execute(ttnn, mesh, inputs['latent'], inputs['weight'], inputs['base'], step, temporary,
                        worker_limit=workers)
                    token = ttnn.argmax(scores, dim=-1, keepdim=False)
                    temporary.append(token)
                    results[workers, step] = (scores, token)
            return results

        def audit(results, mode, ordinal, case):
            if {name: addresses(ttnn, value) for name, value in inputs.items()} != bindings:
                raise AssertionError('Captured Markov input bindings changed')
            for (workers, step), (scores, token) in results.items():
                for chip in range(2):
                    actual = ttnn.to_torch(ttnn.get_device_tensors(scores)[chip]).clone()
                    golden = expected[case, step]
                    close = torch.isclose(actual, golden, rtol=1e-4, atol=1e-4)
                    checksum = tensor_digest(actual)
                    if mode == 'eager':
                        eager[workers, step, case, chip] = checksum
                    exact = mode == 'eager' or checksum == eager[workers, step, case, chip]
                    passed = actual.dtype == torch.float32 and bool(torch.isfinite(actual).all()) and bool(close.all()) and exact
                    report['score_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, step=step, workers=workers, chip=chip,
                        passed=passed, numerical_close=bool(close.all()), replay_exact=exact, sha256=checksum,
                        reference_sha256=tensor_digest(golden), max_abs=float((actual - golden).abs().max()),
                        failed_elements=int((~close).sum())))
                    if not passed:
                        torch.save(dict(actual=actual, expected=golden.clone(), case=case, step=step, chip=chip, workers=workers),
                            options.output.with_suffix('.failure.pt'))
                        raise AssertionError('SFPU Markov score fails the unchanged FP32 tolerance or exact replay')
                    observed = int(ttnn.to_torch(ttnn.get_device_tensors(token)[chip]).reshape(-1).item())
                    wanted = int(golden.reshape(-1).argmax())
                    report['token_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, step=step, workers=workers,
                        chip=chip, token=observed, expected=wanted, exact=observed == wanted))
                    if observed != wanted:
                        raise AssertionError('Fused score/argmax does not reproduce the independent FP32 choice')
            for name, tensor in inputs.items():
                for chip, shard in enumerate(ttnn.get_device_tensors(tensor)):
                    identical = torch.equal(ttnn.to_torch(shard), values[case][name])
                    report['input_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, chip=chip, name=name, exact=identical))
                    if not identical:
                        raise AssertionError('Fused Markov score overwrites an immutable operand')

        for case in range(2):
            progress(f'eager_{case}')
            update(case)
            results = run()
            ttnn.synchronize_device(mesh)
            audit(results, 'eager', case, case)
            release_owned(ttnn, temporary)
            temporary.clear()
        update(0)
        progress('capture')
        trace, results = capture_operation(ttnn, mesh, run)
        output_bindings = {key: tuple(addresses(ttnn, value) for value in pair) for key, pair in results.items()}
        for ordinal, case in enumerate((1, 0)):
            progress(f'replay_{ordinal}')
            update(case)
            ttnn.execute_trace(mesh, trace, blocking=True)
            if {key: tuple(addresses(ttnn, value) for value in pair) for key, pair in results.items()} != output_bindings:
                raise AssertionError('Fused score/reduction output bindings changed')
            audit(results, 'replay', ordinal, case)
        expected_scores = len(policies) * len(STEPS) * 2 * 4
        if (len(report['score_checks']) != expected_scores or len(report['token_checks']) != expected_scores
                or len(report['input_checks']) != 24 or len(report['stale_controls']) != 2):
            raise AssertionError('Incomplete numerical/replay/ownership matrix')
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
                release_owned(ttnn, temporary)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'], report['native_sources_after'] = source_hashes(), NATIVE.NOISE.fingerprints(root)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Fused Markov source or native runtime changed')
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
