"""Bounded CPU attribution of retained learned-layer arithmetic; not a simulator or hardware qualification."""

import argparse
import json
from pathlib import Path
import struct

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer_diagnostic import OPERANDS_SHA256, REPORT_SHA256, error_summary, load_capture
from dspark_layer_reference import local_weight
from dspark_projection import TOLERANCE, digest, tensor_digest
from dspark_weights import VerifiedWeights
from projection_rounding import grouped_projection_reference


SOURCES = ('dspark_layer_rounding.py','dspark_layer_diagnostic.py','dspark_layer_reference.py',
    'dspark_layer.py','dspark_projection.py','dspark_weights.py','dspark_checkpoint.py',
    'dspark_intake.py','dspark_markov_fixture.py','projection_rounding.py')


def sample_coordinates(actual, expected):
    import torch

    if actual.shape != expected.shape or actual.ndim != 2 or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Finite matching two-dimensional full projection outputs required')
    close = torch.isclose(actual.float(),expected.float(),**TOLERANCE)
    return [(kind,tuple(coordinate)) for kind,mask in (('failure',~close),('control',close))
        for coordinate in mask.nonzero().tolist()[:4]]


def attention_scales(values, mask):
    import torch

    scale = 128**-.5
    bits = struct.unpack('<I',struct.pack('<f',scale))[0]
    truncated = struct.unpack('<f',struct.pack('<I',bits & 0xffff0000))[0]
    result = {}
    for policy,scaling in (('fp32',scale),('bf16_scale',truncated)):
        expected = torch.nn.functional.scaled_dot_product_attention(values['q_rotary'].float(),
            values['k_rotary'].float().repeat_interleave(4,1),values['v_heads'].float().repeat_interleave(4,1),
            attn_mask=mask.float(),scale=scaling).bfloat16()
        result[policy] = dict(scale=scaling,**error_summary(values['attended'],expected))
    return result


def down_samples(values, weight):
    import torch

    inputs = values['product'].reshape(32,8704)
    actual = values['down_partial'].reshape(32,5120)
    expected = inputs.float() @ weight.float()
    coordinates = sample_coordinates(actual,expected)
    rows = sorted({coordinate[0] for kind,coordinate in coordinates})
    columns = sorted({coordinate[1] for kind,coordinate in coordinates})
    sampled_inputs,sampled_weight = inputs[rows].contiguous(),weight[:,columns].contiguous()
    native = {span:grouped_projection_reference(sampled_inputs,sampled_weight,
        destination_rounding=True,fidelity_span=span).float() for span in (16,32)}
    samples = []
    for kind,(row,column) in coordinates:
        value = actual[row:row+1,column:column+1]
        comparisons = {str(span):error_summary(value,result[rows.index(row):rows.index(row)+1,
            columns.index(column):columns.index(column)+1]) for span,result in native.items()}
        samples.append(dict(kind=kind,row=row,column=column,actual=float(value.item()),
            fp32=float(expected[row,column]),native=comparisons))
    return dict(full_reduction_width=8704,full_output_comparison=error_summary(actual,expected),
        selection='First four failing and first four passing row-major coordinates; full K retained',
        sampled_input_sha256=tensor_digest(sampled_inputs),sampled_weight_sha256=tensor_digest(sampled_weight),
        samples=samples)


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','operands','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError('Refusing to overwrite arithmetic attribution evidence')
    report,payload = load_capture(options.report,options.operands)
    sources = source_hashes()
    result = dict(scope=__doc__,backend='cpu_attribution',report_sha256=REPORT_SHA256,
        operands_sha256=OPERANDS_SHA256,checkpoint_sha256=CHECKPOINT_SHA256,sources=sources,
        tolerance=TOLERANCE,eligible_for_hardware=False,target_integrated=False,
        full_pipeline_captured=False,fabric_tested=False,cases=[])
    reader = VerifiedWeights(options.checkpoint)
    try:
        weight = reader.tensor('layers.0.mlp.down_proj.weight')
        if tensor_digest(weight) != report['parameter_sha256']['mlp.down_proj.weight']:
            raise ValueError('Captured and current full down-projection parameter differ')
        for case in range(3):
            for chip in range(2):
                values = {phase:{name:parts[chip] for name,parts in payload['eager'][phase,case].items()}
                    for phase in ('attention','mlp')}
                entry = dict(case=case,chip=chip,
                    attention_scales=attention_scales(values['attention'],payload['fixtures'][case]['mask']),
                    down=down_samples(values['mlp'],local_weight({'mlp.down_proj.weight':weight},'mlp.down_proj.weight',chip)))
                result['cases'].append(entry)
                print(json.dumps(dict(case=case,chip=chip,samples=len(entry['down']['samples']))),flush=True)
    finally:
        reader.__exit__(None,None,None)
    result['checkpoint_closed'] = True
    result['sources_after'] = source_hashes()
    if result['sources_after'] != sources:
        raise ValueError('Arithmetic attribution source changed during execution')
    with options.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(options.output),sha256=digest(options.output),eligible_for_hardware=False)))


if __name__ == '__main__':
    main()
