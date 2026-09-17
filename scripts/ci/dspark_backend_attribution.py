"""Isolate CPU eager versus FP32 attention/rotary policy differences; not a replacement qualification reference."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import SPECIFICATIONS
from dspark_layer_diagnostic import COMPOSED_REPORT_SHA256, COMPOSED_OPERANDS_SHA256, error_summary, load_capture
from dspark_layer_reference import cpu_layer, wide_rotate
from dspark_projection import TOLERANCE, digest, tensor_digest
from dspark_weights import VerifiedWeights


SOURCES = ('dspark_backend_attribution.py','dspark_layer_diagnostic.py','dspark_layer_reference.py',
    'dspark_backbone_reference.py','dspark_layer.py','dspark_projection.py','dspark_weights.py',
    'dspark_checkpoint.py','dspark_intake.py','dspark_markov_fixture.py')


def fp32_attention(query, key, value):
    import torch

    if (query.ndim!=4 or key.ndim!=4 or query.shape[0]!=1 or key.shape[0]!=1 or key.shape!=value.shape
            or key.shape[1]<1 or query.shape[1]<1 or query.shape[1]%key.shape[1]
            or query.shape[-1]!=128 or key.shape[-1]!=128 or key.shape[2]<query.shape[2]
            or any(operand.dtype!=torch.bfloat16 or operand.device.type!='cpu'
                or not torch.isfinite(operand).all() for operand in (query,key,value))):
        raise ValueError('Finite complete CPU BF16 GQA inputs with head width 128 required')
    groups = query.shape[1]//key.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(query.float(),
        key.float().repeat_interleave(groups,1),value.float().repeat_interleave(groups,1),is_causal=False).bfloat16()


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','operands','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError('Refusing to overwrite backend attribution evidence')
    sources = source_hashes()
    report,payload = load_capture(options.report,options.operands,composed_layer=True)
    result = dict(scope=__doc__,backend='cpu_attribution',report_sha256=COMPOSED_REPORT_SHA256,
        operands_sha256=COMPOSED_OPERANDS_SHA256,checkpoint_sha256=CHECKPOINT_SHA256,tolerance=TOLERANCE,
        sources=sources,eligible_for_hardware=False,target_integrated=False,reference_requalified=False,
        frozen_cpu_checks=[],cases=[])
    reader = VerifiedWeights(options.checkpoint)
    try:
        weights = {name:reader.tensor('layers.0.'+name) for name in SPECIFICATIONS}
        if any(tensor_digest(value)!=report['parameter_sha256'][name] for name,value in weights.items()):
            raise ValueError('Backend attribution requires the unchanged complete learned layer weights')
        for case in range(3):
            fixture = payload['fixtures'][case]
            noise = fixture['noise'][:,0,:7]
            frozen = cpu_layer(fixture['cpu_context'],noise,weights,fixture['tables'])
            for stage,value in frozen.items():
                if tensor_digest(value)!=tensor_digest(fixture['cpu_'+stage]):
                    raise ValueError('Frozen upstream-checked eager reference changed')
                result['frozen_cpu_checks'].append(dict(case=case,stage=stage,exact=True))
            inputs = (fixture['context'][:,0],noise,weights,fixture['tables'])
            variants = dict(own_context_eager=cpu_layer(*inputs))
            with patch('dspark_layer_reference.full_attention',side_effect=fp32_attention):
                variants['own_context_fp32_attention'] = cpu_layer(*inputs)
                with patch('dspark_layer_reference.rotate',side_effect=lambda value,cosine,sine:wide_rotate(value,(cosine,sine))):
                    variants['own_context_fp32_attention_rotary'] = cpu_layer(*inputs)
            for chip in range(2):
                actual = dict(attention_residual=payload['eager']['mlp',case]['attention_residual'][chip][:,0,:7],
                    output=payload['eager']['finish',case]['output'][chip][:,0,:7])
                result['cases'].append(dict(case=case,chip=chip,
                    device_against_variant={name:{stage:error_summary(actual[stage],value)
                        for stage,value in variant.items()} for name,variant in variants.items()},
                    variant_against_frozen={name:{stage:error_summary(value,frozen[stage])
                        for stage,value in variant.items()} for name,variant in variants.items()}))
            print(json.dumps(dict(case=case,variants=list(variants))),flush=True)
    finally:
        reader.__exit__(None,None,None)
    result['checkpoint_closed'] = True
    result['sources_after'] = source_hashes()
    if result['sources_after']!=sources:
        raise ValueError('Backend attribution sources changed during execution')
    with options.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(options.output),sha256=digest(options.output),reference_requalified=False)))


if __name__ == '__main__':
    main()
