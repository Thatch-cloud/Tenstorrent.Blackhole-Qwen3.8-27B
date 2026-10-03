"""Offline attribution of captured learned FC/norm errors; never a device qualification."""

import argparse
import importlib.util
import json
from pathlib import Path

from dspark_backbone_reference import rms_norm
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import TAPS
from dspark_projection import POLICY, TOLERANCE, difference, digest, load_reference, tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import projection_shards


CAPTURE_SHA256 = 'f5b73a158c94bf1c81c8370aed9e6cfc8c8e31668229f0088c1cff57a5ee80f7'
OPERANDS_SHA256 = '896140c440c4a84552a1ef303ee79d7ae970ba354a4ef14ae036da6a0528b3bd'


def analyse_case(observed, context, gamma, full_projection, split_projection):
    import torch

    required = {'sum','narrowed','unweighted_norm','context'}
    if (set(observed) != required or any(value.shape != context.shape for value in observed.values())
            or context.dtype != torch.bfloat16 or gamma.dtype != torch.bfloat16
            or tuple(gamma.shape) != (context.shape[-1],) or full_projection.shape != context.shape
            or split_projection.shape != context.shape):
        raise ValueError('Complete matching captured normalization stages and learned references required')
    if (tensor_digest(observed['sum'].bfloat16()) != tensor_digest(observed['narrowed'])
            or tensor_digest(observed['unweighted_norm'] * gamma) != tensor_digest(observed['context'])
            or tensor_digest(rms_norm(full_projection,gamma)) != tensor_digest(context)):
        raise ValueError('Captured casts, learned gamma product and frozen CPU backbone must agree exactly')
    own_unweighted = rms_norm(observed['narrowed'],torch.ones_like(gamma))
    own_context = rms_norm(observed['narrowed'],gamma)
    split_context = rms_norm(split_projection,gamma)
    comparisons = dict(
        split_projection_vs_cpu=difference(split_projection,full_projection,exact=False),
        native_projection_vs_cpu=difference(observed['narrowed'],full_projection,exact=False),
        split_context_vs_backbone=difference(split_context,context,exact=False),
        native_projection_cpu_norm_vs_backbone=difference(own_context,context,exact=False),
        native_norm_vs_own_input=difference(observed['unweighted_norm'],own_unweighted,exact=False),
        native_context_vs_own_input=difference(observed['context'],own_context,exact=False),
        native_context_vs_backbone=difference(observed['context'],context,exact=False))
    failures = (observed['context'].float()-context.float()).abs() > TOLERANCE['atol'] + TOLERANCE['rtol']*context.float().abs()
    coordinates = torch.nonzero(failures).tolist()
    values = dict(cpu_projection=full_projection, split_projection=split_projection,
        native_projection=observed['narrowed'], native_sum=observed['sum'],
        cpu_norm_of_native_projection=own_unweighted, native_norm=observed['unweighted_norm'],
        cpu_context_of_native_projection=own_context, native_context=observed['context'], backbone_context=context)
    errors = []
    for coordinate in coordinates:
        position = tuple(coordinate)
        row = dict(coordinate=coordinate, gamma=float(gamma[coordinate[-1]]),
            limit=float(TOLERANCE['atol']+TOLERANCE['rtol']*context[position].float().abs()))
        row.update({name:float(value[position]) for name,value in values.items()})
        row['native_projection_bits'] = int(observed['narrowed'].contiguous().view(torch.int16)[position])
        row['cpu_projection_bits'] = int(full_projection.contiguous().view(torch.int16)[position])
        errors.append(row)
    return dict(comparisons=comparisons, failures=errors)


def validate_capture(report, payload, *, sources, reference):
    if (report.get('mode') not in ('matrix','eager_diagnostic') or report.get('backend') != 'simulator'
            or report.get('stage') not in ('complete','failed') or report.get('closed_cleanly') is not True
            or report.get('checkpoint_closed') is not True or report.get('cleanup_error')
            or report.get('checkpoint_sha256') != CHECKPOINT_SHA256 or payload.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or report.get('policy') != POLICY or payload.get('policy') != POLICY
            or report.get('tolerance') != TOLERANCE or report.get('reference') != reference
            or payload.get('reference') != reference or not sources
            or report.get('sources') != sources or report.get('sources_after') != sources or payload.get('sources') != sources
            or not report.get('native_sources') or report.get('native_sources_after') != report['native_sources']
            or payload.get('native_sources') != report['native_sources']
            or any(report.get(key) is not False for key in ('fabric_tested','full_pipeline_captured',
                'target_integrated','eligible_for_hardware'))):
        raise ValueError('Source-bound, closed simulator operand capture required; failure is not qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','operands','checkpoint','cpu-outputs','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Refusing to replace diagnostic evidence')
    import torch

    source_report = json.loads(options.report.read_text())
    if digest(options.operands) != source_report.get('operands',{}).get('sha256'):
        raise ValueError('Captured operand file does not match its simulator report')
    payload = torch.load(options.operands,weights_only=True,map_location='cpu')
    features,contexts,reference = load_reference(options.cpu_outputs)
    spec = importlib.util.spec_from_file_location('dspark_projection_source',Path(__file__).with_name('dspark-projection-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    frozen_capture = digest(options.report) == CAPTURE_SHA256 and digest(options.operands) == OPERANDS_SHA256
    expected_sources = source_report['sources'] if frozen_capture else probe.source_hashes()
    validate_capture(source_report,payload,sources=expected_sources,reference=reference)
    for pattern in range(2):
        if (any(tensor_digest(features[pattern][layer]) != tensor_digest(payload['features'][pattern][layer]) for layer in TAPS)
                or tensor_digest(contexts[pattern]) != tensor_digest(payload['contexts'][pattern])):
            raise ValueError('Operand capture differs from frozen CPU fixtures')
    output = dict(scope=__doc__,report_sha256=digest(options.report),operands_sha256=digest(options.operands),
        sources={name:digest(Path(__file__).with_name(name)) for name in ('dspark_projection_diagnostic.py',
            'dspark_projection.py','dspark_backbone_reference.py','dspark_weights.py','feature_projection.py')},
        checkpoint_sha256=CHECKPOINT_SHA256,policy=POLICY,tolerance=TOLERANCE,reference=reference,
        eligible_for_hardware=False,target_integrated=False,pinned_capture_snapshot=frozen_capture,cases=[])
    weights = VerifiedWeights(options.checkpoint)
    try:
        weight,gamma = (weights.tensor(name) for name in ('fc.weight','hidden_norm.weight'))
        if tensor_digest(gamma) != tensor_digest(payload['gamma']):
            raise ValueError('Captured gamma differs from pinned learned weights')
        packed = projection_shards(weight)
        for pattern in range(2):
            full = torch.nn.functional.linear(torch.cat([features[pattern][layer] for layer in TAPS],dim=-1),weight).unsqueeze(1)
            local = [torch.cat([features[pattern][layer][...,chip*2560:(chip+1)*2560] for layer in TAPS],dim=-1)
                for chip in range(2)]
            split = sum(value.float() @ shard.float() for value,shard in zip(local,packed,strict=True)).bfloat16().unsqueeze(1)
            observed = payload['eager']['tail',pattern]
            partials = payload['eager']['projection',pattern]['partial']
            for chip in range(2):
                stages = {name:values[chip] for name,values in observed.items()}
                if tensor_digest(stages['sum']) != tensor_digest(partials[0]+partials[1]):
                    raise ValueError('Tail inputs are not the captured device projection partials')
                result = analyse_case(stages,contexts[pattern],gamma,full,split)
                original = next(entry for entry in source_report['eager_checks'] if
                    entry['pattern']==pattern and entry['chip']==chip and entry['stage']=='backbone_context')
                repeated = result['comparisons']['native_context_vs_backbone']
                if any(original.get(key) != value for key,value in repeated.items()):
                    raise ValueError('Captured tensors do not reproduce the original final-output comparison')
                output['cases'].append(dict(pattern=pattern,chip=chip,**result))
    finally:
        weights.__exit__(None,None,None)
    output['checkpoint_closed'] = True
    with options.output.open('x') as stream:
        stream.write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(output=str(options.output),sha256=digest(options.output),
        failures=[len(case['failures']) for case in output['cases']],eligible_for_hardware=False)))


if __name__ == '__main__':
    main()
