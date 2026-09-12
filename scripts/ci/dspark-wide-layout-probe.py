"""Targeted full-history and fifteen-query layouts only; no learned model, target verification or TG claim."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from attention_batch import capture_operation
from dspark_cached_layer import append_queries
from dspark_hardware_gate import digest
from dspark_history import join_rows
from dspark_prefill import FullHistoryCapture
from dspark_projection import tensor_digest
from dspark_vocabulary_fixture import inputs as vocabulary_fixture
from dspark_wide_target import gather_logits, noise_embeddings, pack_tokens
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import validate
from sim_memory_budget import require_clean, snapshot


SPEC = importlib.util.spec_from_file_location('wide_layout_noise', Path(__file__).with_name('dspark-noise-probe.py'))
NOISE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NOISE)
SOURCES = tuple(sorted(set(NOISE.SOURCES + ('dspark-wide-layout-probe.py', 'dspark_wide_target.py',
    'dspark_history.py', 'dspark_prefill.py', 'dspark_cached_layer.py', 'dspark_full_attention.py',
    'dspark_rotary_device.py', 'target_features.py', 'model_batch.py', 'sim_memory_budget.py'))))
NAMES = ('table', 'identifiers', 'logits', 'features', 'history', 'queries', 'tokens')
OUTPUTS = ('noise', 'logits', 'tokens', 'feature_tail', 'keys_4093', 'keys_4096', 'keys_4108')
REPLAYS = (1, 0)
COUNTS = dict(eager_checks=28, replay_checks=28, input_checks=56, stale_controls=14)


def source_hashes():
    return {name: digest(Path(__file__).parent / name) for name in SOURCES}


def fixtures(pattern):
    import torch

    table, unused_ids, unused_outputs = NOISE.fixtures()
    identifiers = ((torch.arange(15) * 3 + pattern * 7) % 64).reshape(1, 15)
    vocabulary = vocabulary_fixture(pattern)
    features = ((torch.arange(2048)[None, None, :, None] % 127) + torch.arange(2560)[None, None, None, :] % 31).bfloat16()
    features = torch.cat((features, features + 256), dim=0) + pattern
    history = (torch.arange(4096)[None, None, :, None] % 193).expand(2, 4, 4096, 128).bfloat16().clone()
    history[1] += 256
    history += pattern * 2
    queries = torch.full((2, 4, 32, 128), -8192, dtype=torch.bfloat16)
    queries[:, :, :15] = torch.arange(15)[None, None, :, None] + 512 + pattern * 8
    tokens = torch.tensor([[0, 248319, 124159, 124160, 248070, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512]])
    tokens = tokens.roll(pattern, dims=1)
    values = dict(table=table, identifiers=identifiers, logits=vocabulary['packed'],
        features=features, history=history, queries=queries, tokens=tokens)
    complete_table = torch.cat(table.unbind(dim=0), dim=-1).reshape(64, 5120)
    noise = torch.zeros(1, 1, 32, 5120, dtype=torch.bfloat16)
    noise[:, :, :15] = complete_table[identifiers].unsqueeze(1)
    expected = []
    for chip in range(2):
        output = dict(noise=noise, logits=vocabulary['full_logits'][:, :, :15].float(),
            tokens=tokens.reshape(1, 1, 15, 1), feature_tail=features[chip:chip + 1, :, 2016:2048])
        for position in (4093, 4096, 4108):
            cached = history[chip:chip + 1, :, :min(position, 4096)]
            if position == 4108:
                cached = torch.cat((history[chip:chip + 1, :, :4093], queries[chip:chip + 1, :, :15]), dim=2)
            joined = torch.cat((cached, queries[chip:chip + 1, :, :15]), dim=2)
            output[f'keys_{position}'] = torch.nn.functional.pad(joined, (0, 0, 0, 4160 - position - 15))
        expected.append(output)
    return values, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if (options.output.exists() or os.environ.get('QWEN_SIM_SHARED_BDF') != '1'
            or os.environ.get('QWEN_SIM_BOUNDED_MEMORY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED',
                'QWEN_PRECISE_DRAFT_ACTIVE', 'QWEN_SIM_PACKER_ZERO_GRAFT'))):
        raise ValueError('Fresh bounded shared-BDF simulator evidence and original native runtime required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        contexts=[4093, 4096, 4108], proposals=15, vocabulary=248320,
        sources=source_hashes(), native_sources=NOISE.fingerprints(root), resources_before=snapshot(bounded=True),
        link_policy=validate(os.environ), target_integrated=False, learned_model_executed=False,
        physical_fabric_tested=False, committed_tg=None, **{name: [] for name in COUNTS})
    owned, transient = [], []
    mesh = trace = None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        patterns = [fixtures(pattern) for pattern in (0, 1)]
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)

        def upload(value, name, device=True):
            integer = name in ('identifiers', 'tokens')
            tensor = ttnn.from_torch(value, dtype=ttnn.uint32 if integer else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if integer or name == 'table' else ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if integer else ttnn.ShardTensorToMesh(mesh, dim=0),
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
            if device:
                owned.append(tensor)
            return tensor

        inputs = {name: upload(patterns[0][0][name], name) for name in NAMES}
        host_inputs = [{name: upload(values[name], name, False) for name in NAMES} for values, expected in patterns]
        bindings = {name: addresses(ttnn, value) for name, value in inputs.items()}
        target = SimpleNamespace(mesh_device=mesh, num_devices=2, vocab_size=248320,
            embd=lambda values, memory_config: ttnn.embedding(values, inputs['table'], layout=ttnn.TILE_LAYOUT, memory_config=memory_config))
        capture = FullHistoryCapture(ttnn, SimpleNamespace(_forward_prefill_chunk_masked_tp=lambda: None), 4096)

        def retain(value):
            transient.append(value)
            return value

        def update(case):
            for name in NAMES:
                ttnn.copy_host_to_device_tensor(host_inputs[case][name], inputs[name])
            ttnn.synchronize_device(mesh)

        def run():
            output = dict(noise=noise_embeddings(ttnn, target, mesh, collectives, inputs['identifiers'], retain, proposals=15),
                logits=gather_logits(ttnn, mesh, collectives, inputs['logits'], retain, proposals=15))
            records = [dict(token=retain(ttnn.slice(inputs['tokens'], (0, index), (1, index + 1)))) for index in range(15)]
            output['tokens'] = pack_tokens(ttnn, records, retain)
            cloned = retain(capture.snapshot(inputs['features'], 2048))
            output['feature_tail'] = retain(ttnn.slice(cloned, (0, 0, 2016, 0), (1, 1, 2048, 2560)))
            pieces = [retain(ttnn.slice(inputs['history'], (0, 0, start, 0), (1, 4, start + 32, 128))) for start in range(0, 4096, 32)]
            complete = join_rows(ttnn, pieces, retain)
            tail = retain(ttnn.slice(pieces[-1], (0, 0, 0, 0), (1, 4, 29, 128)))
            ragged = join_rows(ttnn, [*pieces[:-1], tail], retain)
            valid = retain(ttnn.slice(inputs['queries'], (0, 0, 0, 0), (1, 4, 15, 128)))
            extended = join_rows(ttnn, [ragged, valid], retain)
            for position, cached in ((4093, ragged), (4096, complete), (4108, extended)):
                output[f'keys_{position}'] = append_queries(ttnn, cached, inputs['queries'], retain, position=position, proposals=15)
            return output

        def host(value, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        eager = {}

        def audit(output, mode, ordinal, case):
            if set(output) != set(OUTPUTS) or {name: addresses(ttnn, value) for name, value in inputs.items()} != bindings:
                raise AssertionError('Complete output set and stable borrowed input bindings required')
            for chip in range(2):
                for name, value in output.items():
                    actual, expected = host(value, chip), patterns[case][1][chip][name]
                    if name == 'tokens':
                        actual = actual.to(torch.int64)
                    exact = actual.dtype == expected.dtype and torch.equal(actual, expected)
                    checksum = tensor_digest(actual)
                    if mode == 'eager':
                        eager[case, chip, name] = checksum
                    replay_exact = mode == 'eager' or checksum == eager[case, chip, name]
                    report[mode + '_checks'].append(dict(ordinal=ordinal, case=case, chip=chip, name=name,
                        passed=bool(exact and replay_exact), sha256=checksum, expected_sha256=tensor_digest(expected)))
                    if not exact or not replay_exact:
                        raise AssertionError('Wider layout changed rows, ranks, full vocabulary or captured output')
                for name, value in inputs.items():
                    actual = host(value, chip)
                    expected = patterns[case][0][name]
                    if name not in ('identifiers', 'tokens'):
                        expected = expected[chip:chip + 1]
                    exact = torch.equal(actual.to(expected.dtype), expected)
                    report['input_checks'].append(dict(mode=mode, ordinal=ordinal, case=case, chip=chip, name=name, exact=bool(exact)))
                    if not exact:
                        raise AssertionError('Wider layout mutated a borrowed input')

        for case in (0, 1):
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
                raise AssertionError('Captured layout output bindings moved')
            audit(output, 'replay', ordinal, case)
        for chip in range(2):
            for name in OUTPUTS:
                detected = eager[0, chip, name] != eager[1, chip, name]
                report['stale_controls'].append(dict(chip=chip, name=name, detected=detected))
                if not detected:
                    raise AssertionError('Every fixture output must detect a missing input update')
        if {name: len(report[name]) for name in COUNTS} != COUNTS:
            raise AssertionError('Incomplete wider layout check matrix')
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
                raise ValueError('Layout source or native runtime changed')
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
