"""Fixed full-history banks across unrelated trace replay; synthetic projected K/V, not learned math or TG."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

from attention_batch import capture_operation
from dspark_hardware_gate import digest
from dspark_history import TensorScope, leaves
from dspark_projection import tensor_digest
from dspark_stable_history import StableHistoryKV
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from sim_memory_budget import require_clean, snapshot


SPEC = importlib.util.spec_from_file_location('bank_wide_layout', Path(__file__).with_name('dspark-wide-layout-probe.py'))
WIDE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WIDE)
SOURCES = tuple(sorted(set(WIDE.SOURCES + ('dspark-history-bank-probe.py', 'dspark_stable_history.py',
    'dspark_device.py', 'full_dspark_request.py', 'dspark_history_audit.py'))))
CAPACITY = 4384
INITIAL = 4093
ACTIONS = ((3, True), (15, True), (1, False), (1, True))
COUNTS = dict(bank_checks=260, view_checks=80, trace_checks=8, binding_checks=4, stale_controls=20)


def source_hashes():
    return {name: digest(Path(__file__).parent / name) for name in SOURCES}


def fixture(start, rows, pattern):
    import torch

    values = ((torch.arange(start, start + rows) % 193).reshape(1, 1, rows, 1)
        + (torch.arange(128) % 7).reshape(1, 1, 1, 128)).expand(2, 4, rows, 128).clone()
    values[1] += 1024
    return tuple(tuple((values + layer * 32 + operand * 256 + pattern * 512).bfloat16()
        for operand in range(2)) for layer in range(5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF') != '1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE', 'QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh bounded two-chip simulator evidence and unchanged native runtime required')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        capacity=CAPACITY, initial_position=INITIAL, actions=ACTIONS, sources=source_hashes(),
        native_sources=WIDE.NOISE.fingerprints(root), resources_before=snapshot(bounded=True),
        learned_model_executed=False, target_integrated=False, physical_fabric_tested=False, committed_tg=None,
        **{name: [] for name in COUNTS})
    owned = []
    mesh = trace = cache = output = None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()

        def upload(value, keep=True):
            tensor = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            if keep:
                owned.append(tensor)
            return tensor

        def compare(actual, expected, field, phase, ordinal):
            for index, (value, golden) in enumerate(zip(leaves(actual), leaves(expected), strict=True)):
                shards = ttnn.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Two physical simulated banks required')
                for chip, shard in enumerate(shards):
                    host = ttnn.to_torch(shard)
                    reference = golden[chip:chip + 1]
                    exact = torch.equal(host, reference)
                    report[field].append(dict(phase=phase, ordinal=ordinal, layer=index // 2,
                        operand=index % 2, chip=chip, shape=list(host.shape), passed=exact,
                        sha256=tensor_digest(host), expected_sha256=tensor_digest(reference)))
                    if not exact:
                        raise AssertionError('Fixed bank or logical history differs from the complete CPU prefix')

        initial = fixture(0, INITIAL, 0)
        projected = tuple(tuple(upload(value, keep=False) for value in pair) for pair in initial)
        progress('allocate_both_banks_before_capture')
        with patch('dspark_history.project_chunks', return_value=projected):
            cache = StableHistoryKV(ttnn, mesh, None, {}, [], (), None, position=INITIAL, capacity=CAPACITY)
        expected = tuple(tuple(torch.nn.functional.pad(value, (0, 0, 0, CAPACITY - INITIAL)) for value in pair) for pair in initial)
        compare(cache.layers, expected, 'bank_checks', 'initial', -1)
        bindings = sorted(addresses(ttnn, value) for value in leaves(cache.layers) + leaves(cache.spare_layers))
        delta_inputs, delta_expected = [], []
        position = INITIAL
        for ordinal, (prefix, commit) in enumerate(ACTIONS):
            delta = fixture(position, prefix, ordinal + 1)
            delta_expected.append(delta)
            delta_inputs.append(tuple(tuple(upload(value) for value in pair) for pair in delta))
            if commit:
                position += prefix
        trace_input = upload(torch.full((2, 4, CAPACITY, 128), 7, dtype=torch.bfloat16))

        def trace_operation():
            temporary = ttnn.clone(trace_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            try:
                return ttnn.add(temporary, temporary, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            finally:
                ttnn.deallocate(temporary)

        warm = trace_operation()
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        progress('capture_unrelated_scratch_writer_after_banks')
        trace, output = capture_operation(ttnn, mesh, trace_operation)
        for ordinal, (prefix, commit) in enumerate(ACTIONS):
            progress(f'publication_{ordinal}')
            publication = cache.prepare_projected(delta_inputs[ordinal], prefix, position=cache.position)
            prepared = tuple(tuple(torch.nn.functional.pad(torch.cat((old[..., :cache.position, :], new), dim=2),
                (0, 0, 0, CAPACITY - cache.position - prefix)) for old, new in zip(previous, delta, strict=True))
                for previous, delta in zip(expected, delta_expected[ordinal], strict=True))
            host_input = ttnn.from_torch(torch.full((2, 4, CAPACITY, 128), 7 + ordinal, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            ttnn.copy_host_to_device_tensor(host_input, trace_input)
            ttnn.execute_trace(mesh, trace, blocking=True)
            for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                host = ttnn.to_torch(shard)
                passed = torch.equal(host, torch.full((1, 4, CAPACITY, 128), 2 * (7 + ordinal), dtype=torch.bfloat16))
                report['trace_checks'].append(dict(ordinal=ordinal, chip=chip, passed=passed))
                if not passed:
                    raise AssertionError('Unrelated trace must actually execute with changing inputs')
            compare(cache.layers, expected, 'bank_checks', 'active_after_replay', ordinal)
            compare(publication.layers, prepared, 'bank_checks', 'prepared_after_replay', ordinal)
            if commit:
                cache.commit_publication(publication)
                expected = prepared
            else:
                cache.discard_publication(publication)
            ttnn.execute_trace(mesh, trace, blocking=True)
            compare(cache.layers, expected, 'bank_checks', 'resolved_after_replay', ordinal)
            scope = TensorScope(ttnn, leaves(cache.layers) + leaves(cache.spare_layers))
            try:
                logical = cache.logical_layers(scope.retain)
                compare(logical, tuple(tuple(value[..., :cache.position, :] for value in pair) for pair in expected),
                    'view_checks', 'complete_logical_prefix', ordinal)
            finally:
                scope.release()
            same = sorted(addresses(ttnn, value) for value in leaves(cache.layers) + leaves(cache.spare_layers)) == bindings
            report['binding_checks'].append(dict(ordinal=ordinal, passed=same))
            if not same:
                raise AssertionError('No persistent cache allocation or address change allowed after capture')
        for index, (before, after) in enumerate(zip(leaves(initial), leaves(expected), strict=True)):
            for chip in range(2):
                detected = not torch.equal(before[chip:chip + 1], after[chip:chip + 1, :, :INITIAL, :])
                if detected:
                    raise AssertionError('Oldest full history must not change')
                changed = bool(torch.count_nonzero(after[chip:chip + 1, :, INITIAL:cache.position, :]))
                report['stale_controls'].append(dict(layer=index // 2, operand=index % 2, chip=chip, detected=changed))
                if not changed:
                    raise AssertionError('Missing publication must be detectable')
        if {name: len(report[name]) for name in COUNTS} != COUNTS:
            raise AssertionError('Incomplete fixed-bank lifetime matrix')
        report['final_position'] = cache.position
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
                if cache is not None:
                    cache.close()
                if output is not None:
                    ttnn.deallocate(output)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'], report['native_sources_after'] = source_hashes(), WIDE.NOISE.fingerprints(root)
            if report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']:
                raise ValueError('Bank source or native runtime changed')
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
