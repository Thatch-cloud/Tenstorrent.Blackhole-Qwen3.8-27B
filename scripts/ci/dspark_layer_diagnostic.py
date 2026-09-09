"""Attribute captured layer errors to individual native operations without changing qualification limits."""

import argparse
import json
from pathlib import Path

from dspark_backbone_reference import rms_norm
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import SPECIFICATIONS
from dspark_layer_reference import local_weight, wide_rotate
from dspark_projection import TOLERANCE, difference, digest, tensor_digest
from dspark_weights import VerifiedWeights


REPORT_SHA256 = 'f81fbb9aaab4eaae9ddadaddb4bbd59f8be0ebab6769eb67f038f192232f965c'
OPERANDS_SHA256 = 'e65c254e3197d28a888ddc5be194fdccc321a9afc29dd9a08d1d2f11bff31c2d'


def load_capture(report_path, operands_path):
    import torch

    if digest(report_path) != REPORT_SHA256 or digest(operands_path) != OPERANDS_SHA256:
        raise ValueError('Exact retained complete-layer failure and observed tensors required')
    report = json.loads(report_path.read_text())
    payload = torch.load(operands_path,weights_only=True,map_location='cpu')
    if (report['closed_cleanly'] is not True or report['checkpoint_closed'] is not True
            or report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']
            or payload['sources'] != report['sources'] or payload['native_sources'] != report['native_sources']
            or payload['parameter_sha256'] != report['parameter_sha256'] or payload['policy'] != report['policy']
            or report['checkpoint_sha256'] != CHECKPOINT_SHA256 or report['tolerance'] != TOLERANCE):
        raise ValueError('Closed unchanged layer execution and complete parameter provenance required')
    return report,payload


def error_summary(actual, expected):
    import torch

    result = difference(actual,expected,exact=False)
    if actual.dtype == torch.bfloat16:
        result['different_bits'] = int((actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).sum())
    scale = float(torch.linalg.vector_norm(expected.double()))
    distance = float(torch.linalg.vector_norm(actual.double()-expected.double()))
    result['relative_l2'] = distance/scale if scale else (0. if distance==0 else None)
    return result


def analyse(payload, weights):
    import torch

    results = []
    for case in range(3):
        fixture = payload['fixtures'][case]
        for chip in range(2):
            attention,mlp,finish = ({name:values[chip] for name,values in payload['eager'][phase,case].items()}
                for phase in ('attention','mlp','finish'))
            comparisons = {}
            for name in ('q','k'):
                comparisons[name+'_norm_own_input'] = (attention[name+'_norm'],rms_norm(attention[name+'_heads'],weights[f'self_attn.{name}_norm.weight']))
                comparisons[name+'_rotary_own_input'] = (attention[name+'_rotary'],wide_rotate(attention[name+'_norm'],fixture['tables'][name]))
            expected_attention = torch.nn.functional.scaled_dot_product_attention(attention['q_rotary'].float(),
                attention['k_rotary'].float().repeat_interleave(4,dim=1),attention['v_heads'].float().repeat_interleave(4,dim=1),
                attn_mask=fixture['mask'].float(),is_causal=False).bfloat16()
            comparisons['attention_own_input'] = (attention['attended'],expected_attention)
            comparisons['output_projection_own_input'] = (attention['attention_partial'],
                (attention['merged'].float() @ local_weight(weights,'self_attn.o_proj.weight',chip).float())*fixture['live'])
            comparisons['attention_residual_own_input'] = (mlp['attention_residual'],fixture['noise']+mlp['attention_narrowed'])
            comparisons['post_norm_own_input'] = (mlp['post_norm'],rms_norm(mlp['attention_residual'],weights['post_attention_layernorm.weight']))
            for name in ('gate','up'):
                comparisons[name+'_own_input'] = (mlp[name],
                    (mlp['post_norm'].float() @ local_weight(weights,f'mlp.{name}_proj.weight',chip).float()).bfloat16())
            comparisons['down_own_input'] = (mlp['down_partial'],mlp['product'].float() @ local_weight(weights,'mlp.down_proj.weight',chip).float())
            comparisons['final_residual_own_input'] = (finish['output'],mlp['attention_residual']+finish['down_narrowed'])
            comparisons['upstream_output'] = (finish['output'][:,0,:7],fixture['cpu_output'])
            results.append(dict(case=case,chip=chip,comparisons={name:error_summary(*values) for name,values in comparisons.items()}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','operands','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError('Refusing to replace diagnostic evidence')
    report,payload = load_capture(options.report,options.operands)
    reader = VerifiedWeights(options.checkpoint)
    try:
        weights = {name:reader.tensor('layers.0.'+name) for name in SPECIFICATIONS}
        if any(tensor_digest(value) != report['parameter_sha256'][name] for name,value in weights.items()):
            raise ValueError('Captured layer and diagnostic checkpoint weights differ')
        cases = analyse(payload,weights)
    finally:
        reader.__exit__(None,None,None)
    result = dict(scope=__doc__,report_sha256=REPORT_SHA256,operands_sha256=OPERANDS_SHA256,
        sources={name:digest(Path(__file__).with_name(name)) for name in ('dspark_layer_diagnostic.py',
            'dspark_layer_reference.py','dspark_layer.py','dspark_projection.py','dspark_backbone_reference.py','dspark_weights.py')},
        checkpoint_sha256=CHECKPOINT_SHA256,checkpoint_closed=True,tolerance=TOLERANCE,cases=cases,
        eligible_for_hardware=False,target_integrated=False)
    with options.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(options.output),sha256=digest(options.output),cases=len(cases),eligible_for_hardware=False)))


if __name__ == '__main__':
    main()
