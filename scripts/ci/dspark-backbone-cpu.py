"""Execute all five learned DSpark layers on synthetic CPU inputs; not a device, model-quality or throughput gate."""

import argparse
import hashlib
import json
from pathlib import Path

from dspark_backbone_reference import CPUBackbone
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import FILES, MODEL, REVISION, TAPS
from dspark_weights import VerifiedWeights


SOURCES = ('dspark-backbone-cpu.py', 'dspark_backbone_reference.py', 'dspark_weights.py',
    'dspark_checkpoint.py', 'dspark_intake.py', 'dspark_rope_tables.py', 'dspark_markov_fixture.py')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def inputs(pattern):
    import torch

    if type(pattern) is not int or pattern not in (0, 1):
        raise ValueError('Declared deterministic CPU input pattern required')
    generator = torch.Generator().manual_seed(382610 + pattern)
    features = {layer: torch.randn(1, 32, 5120, generator=generator).bfloat16() for layer in TAPS}
    noise = torch.randn(1, 7, 5120, generator=generator).bfloat16()
    return features, noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    import torch

    outputs_path = options.output.with_suffix('.outputs.pt')
    if options.output.exists() or outputs_path.exists():
        raise ValueError('Fresh report and output-tensor paths required')
    if options.config.stat().st_size != FILES['config.json'][0] or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Pinned DSpark configuration required')
    sources = {name: digest(Path(__file__).with_name(name)) for name in SOURCES}
    report = dict(cpu_backbone_executed=False, closed_cleanly=False, model=MODEL, revision=REVISION,
        backend='cpu', scope=__doc__, sources=sources, checkpoint_sha256=CHECKPOINT_SHA256,
        config_sha256=FILES['config.json'][1], context_rows=32, proposals=7, stages=[], cases=[],
        upstream_full_model_compared=False, target_integrated=False, eligible_for_hardware=False,
        serving_qualified=False, remote_checkpoint_code_executed=False)
    weights, outputs = None, {}
    try:
        print('Verifying whole checkpoint and per-tensor hashes', flush=True)
        with VerifiedWeights(options.checkpoint) as weights:
            report['tensor_sha256'] = weights.fingerprints()
            backbone = CPUBackbone(weights, json.loads(options.config.read_text()))
            for case, (pattern, start) in enumerate(((0, 0), (0, 8190), (1, 8190), (0, 0))):
                features, noise = inputs(pattern)
                borrowed = [*features.values(), noise]
                before = [tensor_digest(value) for value in borrowed]
                stages = {}

                def inspect(stage, value):
                    stages[stage] = value
                    record = dict(case=case, stage=stage, shape=list(value.shape), dtype=str(value.dtype),
                        finite=bool(torch.isfinite(value).all()), max_abs=float(value.float().abs().max()),
                        sha256=tensor_digest(value))
                    report['stages'].append(record)
                    print(json.dumps(record), flush=True)

                output = backbone.forward(features, noise, context_start=start, inspect=inspect)
                if before != [tensor_digest(value) for value in borrowed]:
                    raise AssertionError('CPU reference mutated a borrowed target feature or noise embedding')
                outputs[case] = dict(pattern=pattern, context_start=start, stages=stages, output=output)
                report['cases'].append(dict(case=case, pattern=pattern, context_start=start,
                    input_sha256=before, inputs_unchanged=True, output_sha256=tensor_digest(output)))
            if (any(tensor_digest(value) != tensor_digest(outputs[3]['stages'][name])
                    for name, value in outputs[0]['stages'].items())
                    or tensor_digest(outputs[1]['output']) == tensor_digest(outputs[2]['output'])):
                raise AssertionError('Repeated inputs are not deterministic or changed inputs are indistinguishable')
            report['repeat_all_stages_exact'] = True
            report['changed_features_and_noise_detected'] = True
            report['position_shift_output_exact'] = tensor_digest(outputs[0]['output']) == tensor_digest(outputs[1]['output'])
            print('Rechecking whole checkpoint after learned execution', flush=True)
        report['closed_cleanly'] = weights.source is None
        report['sources_after'] = {name: digest(Path(__file__).with_name(name)) for name in SOURCES}
        if report['sources_after'] != sources:
            raise ValueError('CPU backbone sources changed during execution')
        torch.save(outputs, outputs_path)
        report['output_tensors_sha256'] = digest(outputs_path)
        report['cpu_backbone_executed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['closed_cleanly'] = weights is None or weights.source is None
        raise
    finally:
        options.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cpu_backbone_executed=True, cases=len(report['cases']), stages=len(report['stages']),
        target_integrated=False, eligible_for_hardware=False)), flush=True)


if __name__ == '__main__':
    main()
