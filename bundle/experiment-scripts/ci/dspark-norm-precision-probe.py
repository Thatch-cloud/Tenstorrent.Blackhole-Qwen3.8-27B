"""Simulator-only RMS attribution on frozen observed FC outputs; no replay or full-pipeline qualification."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from dspark_backbone_reference import rms_norm
from dspark_norm_precision import COMPOSED_POLICY, POLICY as CANDIDATE_POLICY, normalize
from dspark_projection import TOLERANCE, compute_config, difference, digest, reference_metadata, tensor_digest
from dspark_projection_diagnostic import CAPTURE_SHA256, OPERANDS_SHA256, validate_capture
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


SOURCES = ('dspark-norm-precision-probe.py','dspark_norm_precision.py','dspark_projection.py',
    'dspark_projection_diagnostic.py','dspark-projection-probe.py','dspark_projection_gate.py',
    'dspark_backbone_reference.py','dspark_checkpoint.py','dspark_intake.py','dspark_weights.py',
    'feature_projection.py','gdn_multitoken_conv.py','../../optimisation/sim/run-dispatch-probe.sh')


def sources():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def fingerprints(root, base):
    result = dict(base)
    directories = ['ttnn/cpp/ttnn/operations/'+name for name in ('reduction','experimental/reduction','data_movement','core')]
    directories.append('tt_metal/tt-llk/tt_llk_blackhole/common/inc')
    for name in directories:
        directory = root/name
        if not directory.is_dir():
            raise ValueError('Complete native reduction, data movement and SFPU source closure required')
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.cpp','.hpp','.h'):
                result[str(path.relative_to(root))] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture','operands','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--composed',action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ,False)
    if os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1':
        raise RuntimeError('Simulator-only diagnostic refuses hardware')
    if digest(options.capture) != CAPTURE_SHA256 or digest(options.operands) != OPERANDS_SHA256:
        raise ValueError('Exact retained failing projection report and observed operands required')
    import torch
    import ttnn

    original = json.loads(options.capture.read_text())
    payload = torch.load(options.operands,weights_only=True,map_location='cpu')
    validate_capture(original,payload,sources=original['sources'],reference=reference_metadata())
    spec = importlib.util.spec_from_file_location('dspark_projection_runtime',Path(__file__).with_name('dspark-projection-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    root = Path(os.environ['TT_METAL_HOME'])
    native = probe.fingerprints(root,packer_compat=False)
    if native != original['native_sources']:
        raise ValueError('Unmodified native runtime matching the operand capture required')
    native = fingerprints(root,native)
    report = dict(passed=False,closed_cleanly=False,backend='simulator',scope=__doc__,sources=sources(),native_sources=native,
        capture_sha256=CAPTURE_SHA256,operands_sha256=OPERANDS_SHA256,tolerance=TOLERANCE,
        composed=options.composed,candidate_policy=COMPOSED_POLICY if options.composed else CANDIDATE_POLICY,
        eligible_for_hardware=False,full_pipeline_captured=False,fabric_tested=False,target_integrated=False,
        control_checks=[],candidate_checks=[],input_checks=[])
    mesh = None
    owned,transient = [],[]

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage)),flush=True)

    def retain(value):
        transient.append(value)
        return value

    try:
        progress('open')
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2),l1_small_size=24576,trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value):
            result = ttnn.from_torch(value,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        gamma = upload(payload['gamma'].reshape(1,1,1,5120))
        inputs = [upload(payload['eager']['tail',pattern]['narrowed'][0]) for pattern in range(2)]

        def host(value,chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

        for pattern,value in enumerate(inputs):
            progress(f'pattern_{pattern}_control')
            expected_input = payload['eager']['tail',pattern]['narrowed'][0]
            own_norm = rms_norm(expected_input,torch.ones_like(payload['gamma']))
            own_context = rms_norm(expected_input,payload['gamma'])
            control = retain(ttnn.rms_norm(value,epsilon=1e-6,weight=None,compute_kernel_config=compute_config(ttnn),
                memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(mesh)
            for chip in range(2):
                observed = host(control,chip)
                expected = payload['eager']['tail',pattern]['unweighted_norm'][chip]
                report['control_checks'].append(dict(pattern=pattern,chip=chip,**difference(observed,expected,exact=True)))
            release_owned(ttnn,transient)
            transient.clear()
            progress(f'pattern_{pattern}_candidate')
            result = normalize(ttnn,value,gamma,retain,composed=options.composed)
            ttnn.synchronize_device(mesh)
            for chip in range(2):
                observed = {name:host(tensor,chip) for name,tensor in result.items()}
                comparisons = dict(unweighted_norm=(observed['unweighted_norm'],own_norm,False),
                    context=(observed['context'],own_context,False),
                    backbone_context=(observed['context'],payload['contexts'][pattern],False),
                    cast=(observed['unweighted_norm'],observed['wide_norm'].bfloat16(),True),
                    gamma_product=(observed['context'],observed['unweighted_norm']*payload['gamma'],True))
                for stage,(actual,expected,exact) in comparisons.items():
                    report['candidate_checks'].append(dict(pattern=pattern,chip=chip,stage=stage,
                        **difference(actual,expected,exact=exact)))
                for name,tensor,expected in (('projection',value,expected_input),
                        ('gamma',gamma,payload['gamma'].reshape(1,1,1,5120))):
                    identical = tensor_digest(host(tensor,chip)) == tensor_digest(expected)
                    report['input_checks'].append(dict(pattern=pattern,chip=chip,tensor=name,exact=identical))
            release_owned(ttnn,transient)
            transient.clear()
        if (any(not entry['passed'] for key in ('control_checks','candidate_checks') for entry in report[key])
                or any(not entry['exact'] for entry in report['input_checks'])):
            raise AssertionError('RMS diagnostic fails unchanged numerical or exact operand gates')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn,transient)
                release_owned(ttnn,owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = sources()
            report['native_sources_after'] = fingerprints(root,probe.fingerprints(root,packer_compat=False))
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != native:
                raise ValueError('Diagnostic source or native runtime changed')
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
