"""Simulator-only full-vocabulary Markov feedback gate; replicated heads, synthetic base logits, no target model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_markov_device import execute
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-markov-probe.py', 'dspark_markov_device.py', 'dspark_markov.py', 'dspark_intake.py',
    'dspark_markov_fixture.py', 'attention_batch.py', 'gdn_multitoken_conv.py')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
BINARY_SHA256 = 'd2652fc01a6836b4d567a788a9c11d8f6cb238bb480bbf68d0e32ee4037c3e24'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprints(root):
    paths = [Path(PACKER), Path('build_Release/lib/_ttnncpp.so'), Path('build_Release/ttnn/_ttnncpp.so')]
    for directory in ('embedding', 'matmul', 'reduction/argmax'):
        paths.extend(path.relative_to(root) for path in (root / 'ttnn/cpp/ttnn/operations' / directory).rglob('*')
            if path.suffix in ('.cpp', '.hpp', '.h') and path.is_file())
    result = {str(path): digest(root / path) for path in sorted(paths)}
    if result[PACKER] != ORIGINAL_PACKER or any(result[str(path)] != BINARY_SHA256 for path in paths[1:3]):
        raise ValueError('Original reviewed native runtime and packer required')
    return result


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    generator = torch.Generator().manual_seed(38256)
    if options.fixture:
        from dspark_markov_fixture import load_fixture
        fixture, predecessor, successor = load_fixture(options.fixture)
        vocabulary, steps = 248320, 7
    else:
        vocabulary, steps = 64, 3
        predecessor, successor = [(torch.randint(-2, 3, (vocabulary, 256), generator=generator) / 16).bfloat16()
            for _ in range(2)]
        predecessor[0].zero_()
        fixture = None
    patterns = [(torch.tensor([[[[anchor]]]], dtype=torch.int64),
        torch.randn((1, 1, steps, vocabulary), generator=generator) / 8)
        for anchor in (1596 if options.fixture else 2, vocabulary - 1)]
    if not options.fixture:
        patterns.append((torch.zeros((1, 1, 1, 1), dtype=torch.int64), torch.zeros((1, 1, steps, vocabulary))))
    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', vocabulary=vocabulary, proposals=steps,
        fixture=fixture, sources=source_hashes(), native_sources=fingerprints(root), scope=__doc__,
        target_integrated=False, eligible_for_hardware=False, accuracy_policy='FP32 bias and sum; no SGLang BF16 bitwise claim',
        eager_checks=[], replay_checks=[], input_checks=[], weight_checks=[], stale_controls=[])
    mesh, trace = None, None
    persistent, transient = [], []

    def progress(stage):
        report['stage'] = stage
        print(json.dumps(dict(stage=stage, vocabulary=vocabulary, proposals=steps)), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload_tensor(value, dtype, layout, device=True):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper,
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))

        weights = [predecessor.reshape(1, 1, vocabulary, 256), successor.T.contiguous().reshape(1, 1, 256, vocabulary)]
        persistent.extend(upload_tensor(value, ttnn.bfloat16, layout) for value, layout in
            zip(weights, (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), strict=True))
        persistent.extend(upload_tensor(value, dtype, layout) for value, dtype, layout in
            zip(patterns[0], (ttnn.uint32, ttnn.float32), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), strict=True))
        payloads = [[upload_tensor(value, dtype, layout, False) for value, dtype, layout in
            zip(pattern, (ttnn.uint32, ttnn.float32), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), strict=True)]
            for pattern in patterns]
        bindings = [addresses(ttnn, tensor) for tensor in persistent]

        def host(tensor, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).clone()

        def update(pattern):
            for payload, tensor in zip(payloads[pattern], persistent[2:], strict=True):
                ttnn.copy_host_to_device_tensor(payload, tensor)
            ttnn.synchronize_device(mesh)

        def run():
            return execute(ttnn, persistent[2], persistent[3], persistent[0], persistent[1], transient)

        def inspect(records, chip):
            return [(host(record['token'], chip).long(), host(record['scores'], chip).float()) for record in records]

        def audit_inputs(phase, ordinal, pattern):
            for index, (tensor, expected) in enumerate(zip(persistent[2:], patterns[pattern], strict=True)):
                for chip in range(2):
                    if not torch.equal(host(tensor, chip).to(expected.dtype), expected):
                        raise AssertionError('Markov prototype changed a borrowed input')
                    report['input_checks'].append(dict(phase=phase, ordinal=ordinal, tensor=index, chip=chip, exact=True))

        def audit_weights(phase):
            for index, (tensor, expected) in enumerate(zip(persistent[:2], weights, strict=True)):
                for chip in range(2):
                    if not torch.equal(host(tensor, chip), expected):
                        raise AssertionError('Learned Markov weights changed')
                    report['weight_checks'].append(dict(phase=phase, tensor=index, chip=chip, exact=True))

        audit_weights('before')
        references = []
        for pattern, (anchor, base) in enumerate(patterns):
            update(pattern)
            records = run()
            ttnn.synchronize_device(mesh)
            expected = []
            previous = anchor.reshape(1).long()
            for step in range(steps):
                scores = base[0, 0, step][None] + predecessor[previous].float() @ successor.float().T
                previous = scores.argmax(-1)
                expected.append((previous.clone(), scores))
            observed = [inspect(records, chip) for chip in range(2)]
            for chip, values in enumerate(observed):
                for step, ((token, scores), (expected_token, expected_scores)) in enumerate(zip(values, expected, strict=True)):
                    scores = scores.reshape(1, vocabulary)
                    torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)
                    if not torch.equal(token.reshape(1), expected_token):
                        raise AssertionError('Full-vocabulary greedy Markov proposal differs from CPU reference')
                    report['eager_checks'].append(dict(pattern=pattern, step=step, chip=chip, token_exact=True,
                        full_vocabulary_close=True, max_abs=float((scores - expected_scores).abs().max()),
                        token=int(expected_token[0])))
            references.append(observed)
            audit_inputs('eager', pattern, pattern)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_{pattern}_complete')
        update(0)
        trace, records = capture_operation(ttnn, mesh, run)
        output_bindings = [[addresses(ttnn, record[name]) for name in ('token', 'scores')] for record in records]
        for repetition, pattern in enumerate((*range(len(patterns)), 0)):
            update(pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if ([addresses(ttnn, tensor) for tensor in persistent] != bindings or output_bindings !=
                    [[addresses(ttnn, record[name]) for name in ('token', 'scores')] for record in records]):
                raise AssertionError('Markov trace bindings changed')
            for chip in range(2):
                observed = inspect(records, chip)
                for step, (actual, expected) in enumerate(zip(observed, references[pattern][chip], strict=True)):
                    if any(not torch.equal(value, reference) for value, reference in zip(actual, expected, strict=True)):
                        raise AssertionError('Markov trace differs from its own changed-input eager trajectory')
                    report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, step=step,
                        chip=chip, token_and_scores_exact=True, bindings_stable=True))
            audit_inputs('replay', repetition, pattern)
            progress(f'replay_{repetition}_complete')
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        for chip in range(2):
            observed = inspect(records, chip)
            if (any(not torch.equal(actual[1], expected[1]) for actual, expected in zip(observed, references[0][chip], strict=True))
                    or all(torch.equal(actual[1], changed[1]) for actual, changed in zip(observed, references[1][chip], strict=True))):
                raise AssertionError('Missing-update negative control is not distinguishable')
            report['stale_controls'].append(dict(chip=chip, missing_update_detected=True))
        audit_weights('after')
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
            report['native_sources_after'] = fingerprints(root)
            report['sources_after'] = source_hashes()
            if report['native_sources_after'] != report['native_sources'] or report['sources_after'] != report['sources']:
                raise ValueError('Native runtime or probe sources changed during execution')
            progress('complete' if report['passed'] else 'failed')
        finally:
            if (not report['closed_cleanly'] or report.get('native_sources_after') != report['native_sources']
                    or report.get('sources_after') != report['sources']):
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
