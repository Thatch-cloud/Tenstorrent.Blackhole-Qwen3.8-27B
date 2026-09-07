"""Learned attention with optional convolution/residual integration; no MLP or request-history integration."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_attention import draft_attention_mask, composed_draft_attention
from draft_attention_fixture import load_attention
from draft_convolution import grouped_causal_convolution, convolution_reference
from draft_convolution_fixture import load_convolution
from draft_head_preparation import rope_tables, rope_reference, head_norm_reference
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance, rms_reference
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--fp32-rope', action='store_true')
    parser.add_argument('--inspect-attention', action='store_true')
    parser.add_argument('--explicit-softmax', action='store_true')
    parser.add_argument('--wide-attention', action='store_true')
    parser.add_argument('--pairwise-softmax', action='store_true')
    parser.add_argument('--pairwise-dots', action='store_true')
    parser.add_argument('--fused-row-sum', action='store_true')
    parser.add_argument('--fused-dots', action='store_true')
    parser.add_argument('--cache-dot-tiles', action='store_true')
    parser.add_argument('--context', type=int, choices=(31, 2048), default=31)
    parser.add_argument('--wide-dot-placement', action='store_true')
    parser.add_argument('--convolution-fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.hardware and options.convolution_fixture:
        parser.error('Integrated attention branch requires simulator validation')
    if options.hardware and (options.context != 31 or options.wide_dot_placement) and not (
            options.context == 2048 and options.wide_dot_placement and options.cache_dot_tiles
            and options.fused_dots and options.fused_row_sum):
        parser.error('Long-context hardware requires the simulator-validated cached wide fused path')
    if not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required for output reduction')
    if options.hardware and (not all((options.fp32_rope, options.explicit_softmax))
            or options.inspect_attention or options.wide_attention
            or (options.cache_dot_tiles and not options.fused_dots)
            or options.pairwise_softmax == options.fused_row_sum
            or options.pairwise_dots == options.fused_dots
            or (options.fused_dots and not options.fused_row_sum)):
        parser.error('Hardware requires the inspection-free simulator-validated precise path')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_attention(options.fixture)
    conv_manifest, conv_weights = load_convolution(options.convolution_fixture) if options.convolution_fixture else (None, None)
    context = options.context
    valid_keys = context + 8
    key_rows = ((valid_keys + 31) // 32) * 32
    query_start = 4096 + context
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, context=context, block_rows=8,
        convolution_checkpoint=conv_manifest, integrated_branch=bool(options.convolution_fixture),
        projection_fidelity_span=32 if options.convolution_fixture else 16,
        wide_dot_placement=options.wide_dot_placement,
        backend='hardware' if options.hardware else 'simulator',
        fp32_rope=options.fp32_rope,
        explicit_softmax=options.explicit_softmax,
        wide_attention=options.wide_attention,
        pairwise_softmax=options.pairwise_softmax,
        pairwise_dots=options.pairwise_dots,
        fused_row_sum=options.fused_row_sum,
        fused_dots=options.fused_dots,
        cache_dot_tiles=options.cache_dot_tiles,
        attention_diagnostics=[],
        start_position=4096, checks=[], tolerance=dict(projection_rtol=1e-4, projection_atol=1e-4,
            attention_rtol=.01, attention_atol=.01, norm_ulps=2), sources={name:
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('learned-attention-probe.py', 'draft_head_preparation.py', 'draft_attention.py',
                    'draft_attention_fixture.py', 'feature_collective.py', 'projection_rounding.py',
                    'draft_convolution.py', 'draft_convolution_fixture.py',
                    'draft_row_sum.py', 'draft_row_sum_io.cpp', 'draft_row_sum_compute.cpp',
                    'draft_dot.py', 'draft_dot_io.cpp', 'draft_dot_compute.cpp')})
    mesh = None
    tensors = []

    def checkpoint(stage, **details):
        report['progress'] = dict(stage=stage, **details)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['progress']), flush=True)

    def retain(value):
        tensors.append(value)
        return value

    def upload(value, *, sharded=False, layout=None):
        return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT if layout is None else layout, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh)))

    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip])

    try:
        checkpoint('opening_mesh')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        key_hidden = torch.randn((1, 1, key_rows, 5120), generator=torch.Generator().manual_seed(8147)).bfloat16()
        query_hidden = torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)
        query_hidden[..., :8, :] = key_hidden[..., context:valid_keys, :]
        inputs = {'q': upload(query_hidden), 'k': upload(key_hidden)}
        residual = inputs['q']
        if conv_weights is not None:
            branch_norm_weight = conv_weights['layers.0.input_layernorm.weight']
            conv_weight = conv_weights['layers.0.attention_conv.kernel_projection.weight'].T.contiguous()
            base_weight = conv_weights['layers.0.attention_conv.base_kernel']
            device_branch_norm = upload(branch_norm_weight.reshape(1, 1, 160, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
            branch_normalized = retain(ttnn.rms_norm(residual, epsilon=1e-6, weight=device_branch_norm,
                compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            conv_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 5),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
                fuse_batch=True, fused_activation=None, mcast_in0=True)
            conv_projection = retain(ttnn.matmul(branch_normalized, upload(conv_weight), dtype=ttnn.float32,
                program_config=conv_program, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            rounded_dynamic = retain(ttnn.typecast(conv_projection, ttnn.bfloat16))
            dynamic = [retain(ttnn.slice(rounded_dynamic, (0, 0, 0, offset * 320), (1, 1, 32, (offset + 1) * 320)))
                for offset in range(4)]
            bases = [upload(base_weight[phase, offset].reshape(1, 1, 1, 5120))
                for phase in range(2) for offset in range(2)]
            inputs['q'] = retain(grouped_causal_convolution(ttnn, mesh, branch_normalized, dynamic[:2], bases[:2],
                fp32_intermediates=True))
            context_input = retain(ttnn.slice(inputs['k'], (0, 0, 0, 0), (1, 1, context, 5120)))
            proposal_input = retain(ttnn.slice(inputs['q'], (0, 0, 0, 0), (1, 1, 8, 5120)))
            padding_input = upload(torch.zeros((1, 1, key_rows - valid_keys, 5120), dtype=torch.bfloat16))
            inputs['k'] = retain(ttnn.concat((context_input, proposal_input, padding_input), dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG))
            checkpoint('convolution_prepare_complete')
        projected, heads, normalized, rotated, local_weights = {}, {}, {}, {}, {}
        for name, count in (('q', 16), ('k', 4), ('v', 4)):
            weight = weights[f'layers.0.self_attn.{name}_proj.weight']
            local_weights[name] = [part.T.contiguous() for part in weight.chunk(2, dim=0)]
            device_weight = upload(torch.cat(local_weights[name], dim=0), sharded=True)
            rows = 32 if name == 'q' else key_rows
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 8),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32,
                per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
            projected[name] = retain(ttnn.matmul(inputs['q' if name == 'q' else 'k'], device_weight,
                dtype=ttnn.float32, compute_kernel_config=kernel, program_config=program, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            checkpoint('projection_dispatched', projection=name, rows=rows)
            rounded = retain(ttnn.typecast(projected[name], ttnn.bfloat16))
            reshaped = retain(ttnn.reshape(rounded, (1, rows, count, 128)))
            heads[name] = retain(ttnn.transpose(reshaped, 1, 2))
            if name != 'v':
                norm_weight = upload(weights[f'layers.0.self_attn.{name}_norm.weight'].reshape(1, 1, 4, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
                normalized[name] = retain(ttnn.rms_norm(heads[name], epsilon=1e-6, weight=norm_weight,
                    compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                cosine, sine = rope_tables(query_start if name == 'q' else 4096, rows)
                rope_input, device_cosine, device_sine = normalized[name], upload(cosine), upload(sine)
                if options.fp32_rope:
                    rope_input, device_cosine, device_sine = [retain(ttnn.typecast(value, ttnn.float32))
                        for value in (rope_input, device_cosine, device_sine)]
                rotated[name] = retain(ttnn.experimental.rotary_embedding_hf(rope_input, device_cosine, device_sine,
                    is_decode_mode=False, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                if options.fp32_rope:
                    rotated[name] = retain(ttnn.typecast(rotated[name], ttnn.bfloat16))
        mask = draft_attention_mask(context)
        def inspect_attention(query, keys, values, scores, masked, probabilities, result):
            for chip in range(2):
                actual_query, actual_keys, actual_values, actual_scores, actual_masked, actual_probabilities, actual_result = [
                    host(value, chip).float() for value in (query, keys, values, scores, masked, probabilities, result)]
                expected_scores = actual_query @ actual_keys.transpose(-1, -2)
                expected_probabilities = torch.softmax(actual_masked, dim=-1)
                expected_result = actual_probabilities @ actual_values
                report['attention_diagnostics'].append(dict(chip=chip,
                    qk_max_error=float((actual_scores - expected_scores).abs().max()),
                    softmax_max_error=float((actual_probabilities - expected_probabilities).abs().max()),
                    probability_bf16_fraction=float((actual_probabilities == actual_probabilities.bfloat16().float()).float().mean()),
                    probability_row_sum_error=float((actual_probabilities.sum(-1) - 1).abs().max()),
                    pv_max_error=float((actual_result - expected_result).abs().max())))

        attention = retain(composed_draft_attention(ttnn, mesh, rotated['q'], rotated['k'], heads['v'], upload(mask),
            inspect=inspect_attention if options.inspect_attention else None, explicit_softmax=options.explicit_softmax,
            wide_operands=options.wide_attention, pairwise_sum=options.pairwise_softmax, pairwise_dots=options.pairwise_dots,
            fused_row_sum=options.fused_row_sum, fused_dots=options.fused_dots, cache_dot_tiles=options.cache_dot_tiles,
            wide_dot_placement=options.wide_dot_placement))
        rounded_attention = retain(ttnn.typecast(attention, ttnn.bfloat16))
        transposed = retain(ttnn.transpose(rounded_attention, 1, 2))
        merged = retain(ttnn.reshape(transposed, (1, 1, 32, 2048)))
        output_weights = [part.T.contiguous() for part in weights['layers.0.self_attn.o_proj.weight'].chunk(2, dim=1)]
        device_output_weight = upload(torch.cat(output_weights, dim=0), sharded=True)
        output_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10),
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        partial_output = retain(ttnn.matmul(merged, device_output_weight, dtype=ttnn.float32,
            compute_kernel_config=kernel, program_config=output_program, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        output = retain(gather_add_projection(ttnn, mesh, collectives, partial_output))
        if conv_weights is not None:
            rounded_output = retain(ttnn.typecast(output, ttnn.bfloat16))
            finished = retain(grouped_causal_convolution(ttnn, mesh, rounded_output, dynamic[2:], bases[2:],
                fp32_intermediates=True))
            wide_finished = retain(ttnn.typecast(finished, ttnn.float32))
            wide_residual = retain(ttnn.typecast(residual, ttnn.float32))
            branch_sum = retain(ttnn.add(wide_finished, wide_residual, dtype=ttnn.float32))
            branch_output = retain(ttnn.typecast(branch_sum, ttnn.bfloat16))
        ttnn.synchronize_device(mesh)
        checkpoint('device_pipeline_complete')
        partials = [host(partial_output, chip) for chip in range(2)]
        for chip in range(2):
            if conv_weights is not None:
                actual_branch_norm = host(branch_normalized, chip)
                norm_ulps = int(bf16_ulp_distance(actual_branch_norm, rms_reference(query_hidden, branch_norm_weight)).max())
                if norm_ulps > 2:
                    raise AssertionError('Attention branch normalization exceeds two BF16 ULPs')
                actual_conv_projection = host(conv_projection, chip)
                expected_conv_projection = grouped_projection_reference(actual_branch_norm, conv_weight,
                    destination_rounding=True, fidelity_span=32)
                torch.testing.assert_close(actual_conv_projection.double(), expected_conv_projection, rtol=1e-4, atol=1e-4)
                expected_dynamic = actual_conv_projection.bfloat16().split(320, dim=-1)
                if any(not torch.equal(host(value, chip), expected) for value, expected in zip(dynamic, expected_dynamic, strict=True)):
                    raise AssertionError('Attention dynamic kernel slicing must be exact')
                expected_prepared = convolution_reference(actual_branch_norm, expected_dynamic[:2],
                    [base_weight[0, offset].reshape(1, 1, 1, 5120) for offset in range(2)])
                if not torch.equal(host(inputs['q'], chip), expected_prepared):
                    raise AssertionError('Attention prepare convolution must be exact')
                expected_keys = torch.cat((key_hidden[..., :context, :], expected_prepared[..., :8, :],
                    torch.zeros((1, 1, key_rows - valid_keys, 5120), dtype=torch.bfloat16)), dim=2)
                if not torch.equal(host(inputs['k'], chip), expected_keys):
                    raise AssertionError('Attention context/proposal assembly must be exact')
                expected_finished = convolution_reference(host(output, chip).bfloat16(), expected_dynamic[2:],
                    [base_weight[1, offset].reshape(1, 1, 1, 5120) for offset in range(2)])
                if not torch.equal(host(finished, chip), expected_finished):
                    raise AssertionError('Attention finish convolution must be exact')
                if not torch.equal(host(branch_output, chip), (expected_finished.float() + query_hidden.float()).bfloat16()):
                    raise AssertionError('Attention residual must be exact')
                report['checks'].append(dict(chip=chip, stage='convolution/residual', norm_ulps=norm_ulps,
                    prepare_exact=True, context_proposal_exact=True, finish_exact=True, residual_exact=True))
            for name, count in (('q', 16), ('k', 4), ('v', 4)):
                valid = 8 if name == 'q' else valid_keys
                actual_projection = host(projected[name], chip)[..., :valid, :]
                reference_input = query_hidden if name == 'q' else key_hidden
                if conv_weights is not None:
                    reference_input = host(inputs['q' if name == 'q' else 'k'], chip)
                max_projection_error = 0.
                for start in range(0, valid, 32):
                    stop = min(start + 32, valid)
                    expected = grouped_projection_reference(reference_input[..., start:stop, :], local_weights[name][chip],
                        destination_rounding=True, fidelity_span=report['projection_fidelity_span'])
                    actual_chunk = actual_projection[..., start:stop, :].double()
                    torch.testing.assert_close(actual_chunk, expected, rtol=1e-4, atol=1e-4)
                    max_projection_error = max(max_projection_error, float((actual_chunk - expected).abs().max()))
                    checkpoint('projection_reference', chip=chip, projection=name, validated_rows=stop, total_rows=valid)
                actual_heads = host(heads[name], chip)
                expected_heads = host(projected[name], chip).bfloat16().reshape(1, -1, count, 128).transpose(1, 2)
                if not torch.equal(actual_heads, expected_heads):
                    raise AssertionError('Head layout changed learned projection channels')
                check = dict(chip=chip, projection=name, validated_rows=valid, max_projection_error=max_projection_error)
                if name != 'v':
                    actual_norm = host(normalized[name], chip)
                    expected_norm = head_norm_reference(actual_heads, weights[f'layers.0.self_attn.{name}_norm.weight'])
                    distance = bf16_ulp_distance(actual_norm, expected_norm)
                    check['norm_max_ulps'] = int(distance.max())
                    if check['norm_max_ulps'] > 2:
                        raise AssertionError('Head normalization gate failed')
                    tables = rope_tables(query_start if name == 'q' else 4096, actual_heads.shape[2])
                    expected_rope = rope_reference(actual_norm, *tables)
                    actual_rope = host(rotated[name], chip)
                    torch.testing.assert_close(actual_rope.float(), expected_rope.float(), rtol=.01, atol=.01)
                    check['rope_max_error'] = float((actual_rope.float() - expected_rope.float()).abs().max())
                report['checks'].append(check)
            expected_attention = torch.nn.functional.scaled_dot_product_attention(host(rotated['q'], chip).float(),
                host(rotated['k'], chip).float().repeat_interleave(4, dim=1), host(heads['v'], chip).float().repeat_interleave(4, dim=1),
                attn_mask=mask.float(), is_causal=False)
            actual_attention = host(attention, chip)
            torch.testing.assert_close(actual_attention[..., :8, :], expected_attention[..., :8, :], rtol=.01, atol=.01)
            actual_merged = host(merged, chip)
            expected_merged = actual_attention.bfloat16().transpose(1, 2).reshape(1, 1, 32, 2048)
            if not torch.equal(actual_merged, expected_merged):
                raise AssertionError('Output head merge changed channel order')
            expected_output = grouped_projection_reference(actual_merged[..., :8, :], output_weights[chip],
                destination_rounding=True, fidelity_span=report['projection_fidelity_span'])
            torch.testing.assert_close(partials[chip][..., :8, :].double(), expected_output, rtol=1e-4, atol=1e-4)
            if not torch.equal(host(output, chip), partials[0] + partials[1]):
                raise AssertionError('Output projection fabric sum must be exact')
            report['checks'].append(dict(chip=chip, stage='attention/output', sum_exact=True,
                attention_max_error=float((actual_attention[..., :8, :] - expected_attention[..., :8, :]).abs().max())))
            checkpoint('rank_checks_complete', chip=chip)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, tensors)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
