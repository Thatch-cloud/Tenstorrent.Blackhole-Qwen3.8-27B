"""Compare all five learned CPU FP32-policy layers with reviewed upstream topology and retain the original eager controls."""

import argparse
import importlib.util
import json
from pathlib import Path

from dspark_backend_reference import BACKEND, POLICY, FP32Backbone, reviewed_backend
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import FILES
from dspark_projection import CPU_REPORT, CPU_SHA256, digest, reference_metadata, tensor_digest
from dspark_upstream_backbone import control_forward, reviewed_functions
from dspark_weights import VerifiedWeights


CASES = ((0,0),(0,8190),(1,8190),(0,0))
STAGES = ('projected_context',*[f'layer_{layer}_{stage}' for layer in range(5)
    for stage in ('attention_residual','output')],'final_norm')
SOURCES = ('dspark-backend-reference.py','dspark_backend_reference.py','dspark_upstream_backbone.py',
    'dspark_backend_attribution.py','dspark_layer_diagnostic.py','dspark_layer_reference.py','dspark_layer.py',
    'dspark-backbone-cpu.py','dspark_backbone_reference.py','dspark_projection.py','dspark_rope_reference.py',
    'dspark_rope_tables.py','dspark_weights.py','dspark_checkpoint.py','dspark_intake.py','dspark_markov_fixture.py')


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','config','draft-source','upstream-directory','cpu-outputs','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    import torch

    output_path = options.output.with_suffix('.outputs.pt')
    if options.output.exists() or output_path.exists():
        raise ValueError('Fresh backend reference report and output paths required')
    metadata = reference_metadata()
    if options.cpu_outputs.stat().st_size>10000000 or digest(options.cpu_outputs)!=metadata['outputs_sha256']:
        raise ValueError('Pinned complete eager CPU outputs required')
    if options.config.stat().st_size!=FILES['config.json'][0] or digest(options.config)!=FILES['config.json'][1]:
        raise ValueError('Pinned DSpark configuration required')
    config = json.loads(options.config.read_text())
    baseline = json.loads(Path(__file__).with_name(CPU_REPORT).read_text())
    frozen = torch.load(options.cpu_outputs,weights_only=True,map_location='cpu')
    if set(frozen)!=set(range(4)) or any(set(values['stages'])!=set(STAGES) for values in frozen.values()):
        raise ValueError('Complete four-case twelve-stage eager control required')
    records = {(entry['case'],entry['stage']):entry for entry in baseline['stages']}
    for case,values in frozen.items():
        for stage,value in values['stages'].items():
            if tensor_digest(value)!=records[case,stage]['sha256']:
                raise ValueError('Eager tensor differs from the pinned baseline stage')
    spec = importlib.util.spec_from_file_location('backend_cpu_inputs',Path(__file__).with_name('dspark-backbone-cpu.py'))
    inputs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs_module)
    eager_functions,upstream = reviewed_functions(options.draft_source,options.upstream_directory)
    matched_functions,matched_upstream = reviewed_functions(options.draft_source,options.upstream_directory)
    matched_functions = reviewed_backend(matched_functions)
    if upstream!=matched_upstream:
        raise ValueError('Both upstream controls must use the same pinned source bodies')
    report = dict(passed=False,closed_cleanly=False,backend='cpu',scope=__doc__,policy=POLICY,
        sources=source_hashes(),upstream_sources=upstream,checkpoint_sha256=CHECKPOINT_SHA256,
        config_sha256=FILES['config.json'][1],eager_report_sha256=CPU_SHA256,
        eager_outputs_sha256=metadata['outputs_sha256'],context_rows=32,proposals=7,
        cases=[],eager_checks=[],backend_checks=[],port_stages=[],
        original_eager_reference_retained=True,checkpoint_modules_imported=False,
        custom_attention_backend=BACKEND,reviewed_rotary_executed_with_widened_inputs=True,
        native_arithmetic_emulated=False,target_integrated=False,eligible_for_hardware=False,serving_qualified=False)
    weights = None
    outputs = {}

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    try:
        progress('verify_checkpoint')
        weights = VerifiedWeights(options.checkpoint)
        report['tensor_sha256'] = weights.fingerprints()
        if report['tensor_sha256']!=metadata['tensor_sha256']:
            raise ValueError('Backend reference weights differ from frozen eager weights')
        backbone = FP32Backbone(weights,config)
        for case,(pattern,start) in enumerate(CASES):
            features,noise = inputs_module.inputs(pattern)
            before = [tensor_digest(value) for value in (*features.values(),noise)]
            if before!=baseline['cases'][case]['input_sha256']:
                raise ValueError('Backend reference inputs differ from the frozen eager case')
            for mode,functions,implementation,expected in (
                    ('eager',eager_functions,'eager',frozen[case]['stages']),
                    ('backend',matched_functions,BACKEND,None)):
                progress(f'{mode}_{case}')
                if mode=='backend':
                    expected = {}
                    backbone.forward(features,noise,context_start=start,inspect=lambda stage,value:expected.__setitem__(stage,value.clone()))
                    if set(expected)!=set(STAGES):
                        raise ValueError('Complete backend port stage inventory required')
                    outputs[case] = dict(pattern=pattern,context_start=start,stages=expected,output=expected['final_norm'])
                    report['port_stages'].extend(dict(case=case,stage=stage,sha256=tensor_digest(value)) for stage,value in expected.items())
                seen = set()

                def inspect(stage, value):
                    if stage not in STAGES or stage in seen:
                        raise ValueError('Every upstream stage must appear exactly once')
                    reference = expected[stage]
                    exact = (value.shape==reference.shape and value.dtype==torch.bfloat16
                        and bool(torch.isfinite(value).all()) and tensor_digest(value)==tensor_digest(reference))
                    report[mode+'_checks'].append(dict(case=case,stage=stage,exact=exact,sha256=tensor_digest(value)))
                    seen.add(stage)

                control_forward(functions,weights,config,features,noise,start,inspect,attention_implementation=implementation)
                if seen!=set(STAGES) or before!=[tensor_digest(value) for value in (*features.values(),noise)]:
                    raise ValueError('Complete unchanged CPU input and stage coverage required')
            report['cases'].append(dict(case=case,pattern=pattern,context_start=start,input_sha256=before,inputs_unchanged=True))
        report['repeat_exact'] = all(tensor_digest(value)==tensor_digest(outputs[3]['stages'][stage])
            for stage,value in outputs[0]['stages'].items())
        report['changed_input_detected'] = tensor_digest(outputs[1]['output'])!=tensor_digest(outputs[2]['output'])
        if (len(report['eager_checks'])!=48 or len(report['backend_checks'])!=48
                or not report['repeat_exact'] or not report['changed_input_detected']
                or any(not entry['exact'] for mode in ('eager','backend') for entry in report[mode+'_checks'])):
            raise AssertionError('Complete original eager and backend-matched upstream controls must both pass exactly')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if weights is not None:
                weights.__exit__(None,None,None)
            report['closed_cleanly'] = weights is not None and weights.source is None
            report['sources_after'] = source_hashes()
            if report['sources_after']!=report['sources']:
                raise ValueError('Backend reference source changed during execution')
            with output_path.open('xb') as stream:
                torch.save(outputs,stream)
            report['output_tensors_sha256'] = digest(output_path)
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
