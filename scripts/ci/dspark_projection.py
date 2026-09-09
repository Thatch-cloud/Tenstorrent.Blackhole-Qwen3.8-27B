"""Learned DSpark TP2 projection and explicitly rounded normalization, separate from transport."""

import hashlib
import importlib.util
import json
from pathlib import Path

from dspark_backbone_reference import rms_norm
from dspark_intake import TAPS
from feature_projection import concatenate_local_features


CPU_REPORT = 'dspark-backbone-cpu-reference.json'
CPU_SHA256 = '2dcaf7f4e5d1052da206dcead5ccc54b15b0a5d9d81dc3f8605e2a5fd349b912'
UPSTREAM_REPORT = 'dspark-backbone-upstream-reference.json'
UPSTREAM_SHA256 = 'eb79532aeee5121b184f24121b9d2012aa9905f0a202100c98093b3657f9ed16'
POLICY = 'BF16 learned weights; HiFi4 FP32 partials; FP32 sum; BF16 projection; unweighted BF16 RMSNorm; BF16 gamma product'
TOLERANCE = dict(rtol=0.01, atol=0.01)
HANDOFF = 'Host staging of observed FP32 partials; separate traces; no fabric or complete captured pipeline'
PROJECTION_STAGES = ('joined', 'partial')
TAIL_STAGES = ('sum', 'narrowed', 'unweighted_norm', 'context')
EAGER_STAGES = (*PROJECTION_STAGES, *TAIL_STAGES, 'native_input_norm', 'backbone_context')
EXACT_STAGES = ('joined', 'sum', 'narrowed', 'context')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(value):
    import torch

    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def reference_metadata():
    root = Path(__file__).parent
    if digest(root / CPU_REPORT) != CPU_SHA256 or digest(root / UPSTREAM_REPORT) != UPSTREAM_SHA256:
        raise ValueError('Frozen learned CPU backbone and independent upstream comparison required')
    report = json.loads((root / CPU_REPORT).read_text())
    if any(digest(root / name) != checksum for name, checksum in report['sources'].items()):
        raise ValueError('CPU backbone reference sources changed')
    return dict(cpu_report_sha256=CPU_SHA256, upstream_report_sha256=UPSTREAM_SHA256,
        outputs_sha256=report['output_tensors_sha256'], tensor_sha256=report['tensor_sha256'])


def load_reference(outputs_path):
    import torch

    metadata = reference_metadata()
    root = Path(__file__).parent
    report = json.loads((root / CPU_REPORT).read_text())
    if digest(outputs_path) != report['output_tensors_sha256']:
        raise ValueError('Saved complete CPU backbone tensors do not match the frozen report')
    outputs = torch.load(outputs_path, weights_only=True, map_location='cpu')
    spec = importlib.util.spec_from_file_location('dspark_frozen_cpu_inputs', root / 'dspark-backbone-cpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features, expected = {}, {}
    for pattern, case in ((0, 0), (1, 2)):
        taps, noise = module.inputs(pattern)
        if [tensor_digest(value) for value in (*taps.values(), noise)] != report['cases'][case]['input_sha256']:
            raise ValueError('Projection inputs differ from the verified learned CPU backbone')
        context = outputs[case]['stages']['projected_context']
        record = next(entry for entry in report['stages'] if entry['case'] == case and entry['stage'] == 'projected_context')
        if tensor_digest(context) != record['sha256']:
            raise ValueError('Frozen projected-context stage differs from its audited tensor')
        features[pattern], expected[pattern] = taps, context.unsqueeze(1)
    return features, expected, metadata


def compute_config(operations):
    return operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)


def require_tensor(operations, value, shape, dtype):
    if (tuple(value.shape) != shape or value.dtype != dtype or value.layout != operations.TILE_LAYOUT
            or value.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Exact DSpark shape, dtype and tiled DRAM operand required')


def project(operations, features, weight, retain):
    if not isinstance(features, dict) or set(features) != set(TAPS) or not callable(retain):
        raise ValueError('All five named target taps and explicit tensor ownership required')
    ordered = [features[layer] for layer in TAPS]
    for value in ordered:
        require_tensor(operations, value, (1, 1, 32, 2560), operations.bfloat16)
    require_tensor(operations, weight, (1, 1, 12800, 5120), operations.bfloat16)
    joined = retain(concatenate_local_features(operations, ordered))
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10),
        in0_block_w=8, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2,
        fuse_batch=True, fused_activation=None, mcast_in0=True)
    partial = retain(operations.matmul(joined, weight, dtype=operations.float32, program_config=program,
        compute_kernel_config=compute_config(operations), memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(joined=joined, partial=partial)


def normalize_partials(operations, first, second, gamma, retain):
    if not callable(retain):
        raise ValueError('Explicit normalization tensor ownership required')
    for value in (first, second):
        require_tensor(operations, value, (1, 1, 32, 5120), operations.float32)
    require_tensor(operations, gamma, (1, 1, 1, 5120), operations.bfloat16)
    summed = retain(operations.add(first, second, dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG))
    narrowed = retain(operations.typecast(summed, operations.bfloat16))
    normalized = retain(operations.rms_norm(narrowed, epsilon=1e-6, weight=None,
        compute_kernel_config=compute_config(operations), memory_config=operations.DRAM_MEMORY_CONFIG))
    require_tensor(operations, normalized, (1, 1, 32, 5120), operations.bfloat16)
    context = retain(operations.mul(normalized, gamma, dtype=operations.bfloat16,
        memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(sum=summed, narrowed=narrowed, unweighted_norm=normalized, context=context)


def norm_references(first, second, gamma):
    import torch

    summed = first + second
    narrowed = summed.bfloat16()
    unweighted = rms_norm(narrowed, torch.ones_like(gamma))
    expected = rms_norm(narrowed, gamma)
    wide = narrowed.float()
    fused = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-6) * gamma.float()).bfloat16()
    if torch.equal(expected.view(torch.int16), fused.view(torch.int16)):
        raise ValueError('Fixture does not distinguish DSpark intermediate rounding from fused weighted normalization')
    return dict(sum=summed, narrowed=narrowed, unweighted_norm=unweighted, native_input_norm=expected)


def difference(actual, expected, *, exact):
    import torch

    if (actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all()
            or not torch.isfinite(expected).all() or type(exact) is not bool):
        raise ValueError('Matching finite tensors and explicit arithmetic comparison required')
    distance = (actual.float() - expected.float()).abs()
    close = distance <= TOLERANCE['atol'] + TOLERANCE['rtol'] * expected.float().abs()
    identical = tensor_digest(actual) == tensor_digest(expected)
    return dict(passed=identical if exact else bool(close.all()), exact_required=exact,
        bitwise_exact=identical, max_abs=float(distance.max()), failed_elements=int((~close).sum()))
