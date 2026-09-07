"""Learned draft attention on synthetic hidden features; no convolution, MLP or request-history integration."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_attention import draft_attention_mask, composed_draft_attention
from draft_attention_fixture import load_attention
from draft_head_preparation import rope_tables, rope_reference, head_norm_reference
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance
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
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required for output reduction')
    if options.hardware and (not all((options.fp32_rope, options.explicit_softmax,
            options.pairwise_softmax, options.pairwise_dots)) or options.inspect_attention or options.wide_attention or options.fused_row_sum):
        parser.error('Hardware requires the inspection-free simulator-validated precise path')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_attention(options.fixture)
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, context=31, block_rows=8,
        backend='hardware' if options.hardware else 'simulator',
        fp32_rope=options.fp32_rope,
        explicit_softmax=options.explicit_softmax,
        wide_attention=options.wide_attention,
        pairwise_softmax=options.pairwise_softmax,
        pairwise_dots=options.pairwise_dots,
        fused_row_sum=options.fused_row_sum,
        attention_diagnostics=[],
        start_position=4096, checks=[], tolerance=dict(projection_rtol=1e-4, projection_atol=1e-4,
            attention_rtol=.01, attention_atol=.01, norm_ulps=2), sources={name:
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('learned-attention-probe.py', 'draft_head_preparation.py', 'draft_attention.py',
                    'draft_attention_fixture.py', 'feature_collective.py', 'projection_rounding.py',
                    'draft_row_sum.py', 'draft_row_sum_io.cpp', 'draft_row_sum_compute.cpp')})
    mesh = None
    tensors = []

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
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        key_hidden = torch.randn((1, 1, 64, 5120), generator=torch.Generator().manual_seed(8147)).bfloat16()
        query_hidden = torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)
        query_hidden[..., :8, :] = key_hidden[..., 31:39, :]
        inputs = {'q': upload(query_hidden), 'k': upload(key_hidden)}
        projected, heads, normalized, rotated, local_weights = {}, {}, {}, {}, {}
        for name, count in (('q', 16), ('k', 4), ('v', 4)):
            weight = weights[f'layers.0.self_attn.{name}_proj.weight']
            local_weights[name] = [part.T.contiguous() for part in weight.chunk(2, dim=0)]
            device_weight = upload(torch.cat(local_weights[name], dim=0), sharded=True)
            rows = 32 if name == 'q' else 64
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 8),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32,
                per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
            projected[name] = retain(ttnn.matmul(inputs['q' if name == 'q' else 'k'], device_weight,
                dtype=ttnn.float32, compute_kernel_config=kernel, program_config=program, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            rounded = retain(ttnn.typecast(projected[name], ttnn.bfloat16))
            reshaped = retain(ttnn.reshape(rounded, (1, rows, count, 128)))
            heads[name] = retain(ttnn.transpose(reshaped, 1, 2))
            if name != 'v':
                norm_weight = upload(weights[f'layers.0.self_attn.{name}_norm.weight'].reshape(1, 1, 4, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
                normalized[name] = retain(ttnn.rms_norm(heads[name], epsilon=1e-6, weight=norm_weight,
                    compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                cosine, sine = rope_tables(4127 if name == 'q' else 4096, rows)
                rope_input, device_cosine, device_sine = normalized[name], upload(cosine), upload(sine)
                if options.fp32_rope:
                    rope_input, device_cosine, device_sine = [retain(ttnn.typecast(value, ttnn.float32))
                        for value in (rope_input, device_cosine, device_sine)]
                rotated[name] = retain(ttnn.experimental.rotary_embedding_hf(rope_input, device_cosine, device_sine,
                    is_decode_mode=False, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                if options.fp32_rope:
                    rotated[name] = retain(ttnn.typecast(rotated[name], ttnn.bfloat16))
        mask = draft_attention_mask(31)
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
            fused_row_sum=options.fused_row_sum))
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
        ttnn.synchronize_device(mesh)
        partials = [host(partial_output, chip) for chip in range(2)]
        for chip in range(2):
            for name, count in (('q', 16), ('k', 4), ('v', 4)):
                valid = 8 if name == 'q' else 39
                actual_projection = host(projected[name], chip)[..., :valid, :]
                reference_input = query_hidden if name == 'q' else key_hidden
                expected = grouped_projection_reference(reference_input[..., :valid, :], local_weights[name][chip], destination_rounding=True)
                torch.testing.assert_close(actual_projection.double(), expected, rtol=1e-4, atol=1e-4)
                actual_heads = host(heads[name], chip)
                expected_heads = host(projected[name], chip).bfloat16().reshape(1, -1, count, 128).transpose(1, 2)
                if not torch.equal(actual_heads, expected_heads):
                    raise AssertionError('Head layout changed learned projection channels')
                check = dict(chip=chip, projection=name, max_projection_error=float((actual_projection.double() - expected).abs().max()))
                if name != 'v':
                    actual_norm = host(normalized[name], chip)
                    expected_norm = head_norm_reference(actual_heads, weights[f'layers.0.self_attn.{name}_norm.weight'])
                    distance = bf16_ulp_distance(actual_norm, expected_norm)
                    check['norm_max_ulps'] = int(distance.max())
                    if check['norm_max_ulps'] > 2:
                        raise AssertionError('Head normalization gate failed')
                    tables = rope_tables(4127 if name == 'q' else 4096, actual_heads.shape[2])
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
            expected_output = grouped_projection_reference(actual_merged[..., :8, :], output_weights[chip], destination_rounding=True)
            torch.testing.assert_close(partials[chip][..., :8, :].double(), expected_output, rtol=1e-4, atol=1e-4)
            if not torch.equal(host(output, chip), partials[0] + partials[1]):
                raise AssertionError('Output projection fabric sum must be exact')
            report['checks'].append(dict(chip=chip, stage='attention/output', sum_exact=True,
                attention_max_error=float((actual_attention[..., :8, :] - expected_attention[..., :8, :]).abs().max())))
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
