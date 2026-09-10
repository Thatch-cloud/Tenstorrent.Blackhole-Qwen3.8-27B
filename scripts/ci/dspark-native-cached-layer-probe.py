"""Learned layer-zero differential on synthetic fixed history; no model quality or TG claim."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_cached_layer import execute as baseline
from dspark_native_cached_layer import execute as candidate
from dspark_native_fixed_gate import qualify
from dspark_layer import SPECIFICATIONS, pack_weight
from dspark_projection import tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from native_draft_sdpa import run_precise_probe
from sim_memory_budget import snapshot, require_clean


SPEC = importlib.util.spec_from_file_location('native_fixed_fixture', Path(__file__).with_name('dspark-native-fixed-attention-probe.py'))
FIXED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXED)
SOURCES = (*FIXED.SOURCES, 'dspark-native-cached-layer-probe.py', 'dspark_native_cached_layer.py',
    'dspark_weights.py', 'dspark_native_fixed_gate.py', '../../optimisation/sim/run-native-layer-dispatch-probe.sh')


def source_hashes():
    return {name: FIXED.digest(Path(__file__).parent / name) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists() or os.environ.get('QWEN_SIM_BOUNDED_MEMORY') != '1':
        raise ValueError('Fresh bounded simulator output required')
    prerequisite = qualify(Path(__file__).parent)
    kernel_audit = run_precise_probe(__file__)
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    native_hashes = lambda: FIXED.NATIVE.fingerprints(root, packer_compat=True, precise_native=True)
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=source_hashes(), native_sources=native_hashes(), prerequisite=prerequisite,
        kernel_audit=kernel_audit, positions=FIXED.POSITIONS, capacity=FIXED.CAPACITY,
        proposals=15, tolerance=dict(rtol=.01, atol=.01), checks=[], storage_checks=[],
        resources_before=snapshot(bounded=True), committed_tg=None)
    mesh = trace = None
    owned, transient = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    def retain(value):
        transient.append(value)
        return value

    try:
        progress('load_learned_weights')
        with VerifiedWeights(options.checkpoint) as reader:
            packed = {name: pack_weight(name, reader.tensor('layers.0.' + name)) for name in SPECIFICATIONS}
            report['parameter_sha256'] = {name: reader.fingerprints()['layers.0.' + name] for name in SPECIFICATIONS}
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=536870912)
            mesh.enable_program_cache()
            collectives = TT_CCL(mesh)

            def upload(value, *, sharded=False, device=True):
                result = ttnn.from_torch(value, dtype=ttnn.float32 if value.dtype == torch.float32 else ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
                    if sharded else ttnn.ReplicateTensorToMesh(mesh),
                    **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
                if device:
                    owned.append(result)
                return result

            weights = {name: upload(value, sharded=sharded) for name, (value, sharded) in packed.items()}
            generator = torch.Generator().manual_seed(383929)
            noise = torch.randn(1, 1, 32, 5120, generator=generator).bfloat16()
            patterns = []
            for index, values in enumerate(FIXED.fixtures()):
                live = torch.zeros(1, 1, 32, 1)
                live[..., :15, :] = 1
                angles = torch.arange(32).float().reshape(1, 1, 32, 1) * .01 + index * .02
                patterns.append(dict(noise=noise + index * .125, key=values['history_key'],
                    value=values['history_value'], mask=values['mask'], live=live,
                    cosine=angles.cos().expand(1, 1, 32, 128).bfloat16().clone(),
                    sine=angles.sin().expand(1, 1, 32, 128).bfloat16().clone()))
            inputs = {name: upload(value, sharded=name in ('key', 'value')) for name, value in patterns[0].items()}
            payloads = [{name: upload(value, sharded=name in ('key', 'value'), device=False)
                for name, value in pattern.items()} for pattern in patterns]

            def storage():
                return [(addresses(ttnn, value), tuple(tensor_digest(ttnn.to_torch(shard))
                    for shard in ttnn.get_device_tensors(value))) for value in [*weights.values(), *inputs.values()]]

            def update(case):
                for name, payload in payloads[case].items():
                    ttnn.copy_host_to_device_tensor(payload, inputs[name])
                ttnn.synchronize_device(mesh)

            def run(implementation):
                return implementation(ttnn, mesh, collectives, inputs['noise'], (inputs['key'], inputs['value']),
                    weights, (inputs['cosine'], inputs['sine']), inputs['mask'], inputs['live'], retain,
                    position=4384, proposals=15, mask_validated=True)

            def read(result):
                values = dict(attention=result['attention'], output=result['finish']['output'],
                    **{'head_' + name: value for name, value in result['heads'].items()},
                    **{'mlp_' + name: value for name, value in result['mlp'].items()})
                return {(name, chip): ttnn.to_torch(shard).clone() for name, value in values.items()
                    for chip, shard in enumerate(ttnn.get_device_tensors(value))}

            references, eager = {}, {}
            for implementation, label in ((baseline, 'baseline'), (candidate, 'candidate')):
                for case in range(2):
                    progress(f'{label}_{case}')
                    update(case)
                    before = storage()
                    result = read(run(implementation))
                    report['storage_checks'].append(dict(mode=label, case=case, exact=before == storage()))
                    if not report['storage_checks'][-1]['exact']:
                        raise AssertionError('Learned layer changed persistent inputs or weights')
                    if label == 'baseline':
                        references[case] = result
                    else:
                        eager[case] = result
                        for key, actual in result.items():
                            expected = references[case][key]
                            exact_required = key[0].startswith('head_')
                            passed = torch.equal(actual, expected) if exact_required else torch.allclose(
                                actual.float(), expected.float(), rtol=.01, atol=.01)
                            report['checks'].append(dict(mode=label, case=case, stage=key[0], chip=key[1],
                                passed=bool(passed), exact_required=exact_required,
                                max_abs=float((actual.float() - expected.float()).abs().max())))
                            if not passed or not torch.isfinite(actual).all():
                                raise AssertionError('Learned native-attention differential failed: ' + str(key))
                    release_owned(ttnn, transient)
                    transient.clear()
            update(0)
            progress('capture')
            trace, output = capture_operation(ttnn, mesh, lambda: run(candidate))
            for case in (1, 0):
                progress(f'replay_{case}')
                update(case)
                before = storage()
                ttnn.execute_trace(mesh, trace, blocking=True)
                for key, actual in read(output).items():
                    exact = torch.equal(actual, eager[case][key])
                    report['checks'].append(dict(mode='replay', case=case, stage=key[0], chip=key[1], passed=exact))
                    if not exact:
                        raise AssertionError('Learned native-attention replay changed')
                report['storage_checks'].append(dict(mode='replay', case=case, exact=before == storage()))
                if not report['storage_checks'][-1]['exact']:
                    raise AssertionError('Learned replay changed persistent storage')
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
            report['sources_after'], report['native_sources_after'] = source_hashes(), native_hashes()
            if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
                raise ValueError('Learned comparison sources changed')
            report['resources_after'] = snapshot(bounded=True)
            require_clean(report['resources_before'], report['resources_after'])
            report['closed_cleanly'] = True
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
