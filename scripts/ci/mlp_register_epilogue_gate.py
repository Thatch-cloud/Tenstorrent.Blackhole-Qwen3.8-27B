"""Admit only the reviewed nearest-away numerical result, never performance."""

import copy
import hashlib
import json
from pathlib import Path

from mlp_register_epilogue import adapt_projection
from mlp_rounding_policy import runtime
from mlp_weight_pipeline_gate import validate_report


REPORT_SHA256 = '0fa237377c79e0d98d848f9b811ce1d16c81ae63778d33611431a1f5a4e7635f'
CONTROL = '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93'
CANDIDATE = 'd26d67ab4c893de327c8a2c6aa9ec949d904332a9b2a91353c5889f30272b6d4'
COMPUTE = 'd212a495967dd0db31ee447ed7761b7ce6af3306b35fa42611e1296c4a1b1a82'
TYPECAST = '1cfea093d37e36f22dd8c34dbed0342b2dce2b745e19a4996ae5ab1c3c304971'
SIM_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
HARDWARE_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
HELPERS = {
    'mlp_register_epilogue.py': 'a44b8bff866c1c1b4ae07f54800849c2d4a7c94dcb5ad70e6bde433b07120cca',
    'mlp_rounding_policy.py': '39f108b367984a07d960f5188ab690e9475268751b7ac9ec69e8b04495e54b9f',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_candidate(directory):
    directory = Path(directory)
    candidate = directory / 'mlp-register-epilogue-candidate'
    source = directory / 'fused_1d.py'
    if digest(source) != CONTROL:
        raise ValueError('Unchanged winning projection required')
    changed = adapt_projection(source.read_text(encoding='utf-8'), nearest_away=True).encode()
    if hashlib.sha256(changed).hexdigest() != CANDIDATE:
        raise ValueError('Exact simulated projection required')
    for name, expected in HELPERS.items():
        if digest(directory / name) != expected:
            raise ValueError('Simulated epilogue helper changed: ' + name)
    candidate.mkdir()
    (candidate / 'fused_1d.py').write_bytes(changed)
    for name in ('fused_1d_input.cpp', 'fused_1d_weights.cpp'):
        (candidate / name).write_bytes((directory / name).read_bytes())
    return candidate


def expected_kernel(baseline, *, hardware=False):
    expected = copy.deepcopy(baseline)
    expected.update(fused_compute_sha256=COMPUTE, register_epilogue=True,
        activation_diagnostic=False, rounding_diagnostic=False,
        intermediate_rounding='sfpu-bf16-nearest-away', rounding_runtime=dict(
            typecast_header_sha256=TYPECAST,
            packer_header_sha256=HARDWARE_PACKER if hardware else SIM_PACKER,
            rounding_policy='nearest-away hypothesis', hardware_qualified=False))
    return expected


def qualify(directory, evidence, *, runtime_root=None):
    from fused_t16_admission import qualify_simulator

    directory, evidence = Path(directory), Path(evidence)
    path = evidence / 'fused-batch.json'
    if digest(path) != REPORT_SHA256:
        raise ValueError('Exact reviewed nearest-away report required')
    report = json.loads(path.read_text())
    validate_report(report)
    if ((evidence / 'fused-batch.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) !=
                dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned simulator runtime and clean teardown required')
    control = qualify_simulator(directory)
    baseline = next(kernel for kernel in control['kernels'] if kernel['token_rows'] == 16)
    if (report.get('kernels') != [expected_kernel(baseline)]
            or report.get('buffer_candidate_sha256') != CANDIDATE
            or report.get('packer_header_sha256') != SIM_PACKER or report.get('packer_zero_graft') is not True
            or digest(directory / 'fusion_trace.py') != report.get('trace_source_sha256')):
        raise ValueError('Exact executed kernel, known simulator graft and replay helper required')
    candidate = directory / 'mlp-register-epilogue-candidate'
    for path, expected in ((directory / 'fused_1d.py', CONTROL), (candidate / 'fused_1d.py', CANDIDATE),
            *((directory / name, expected) for name, expected in HELPERS.items())):
        if digest(path) != expected:
            raise ValueError('Simulated source changed: ' + str(path))
    for name, expected in baseline['reader_sha256'].items():
        if digest(candidate / name) != expected:
            raise ValueError('Unchanged candidate reader required: ' + name)
    expected = expected_kernel(baseline, hardware=runtime_root is not None)
    if runtime_root is not None and runtime(runtime_root) != expected['rounding_runtime']:
        raise ValueError('Pinned physical packer and identical cast implementation required')
    return dict(report_sha256=REPORT_SHA256, kernels=[expected], passed=True,
        hardware_qualified=False, performance_qualified=False)
