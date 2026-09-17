from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from dspark_layer import PHASES, SPECIFICATIONS, attention_partial, linear, mlp_partial, norm, pack_weight, reduce_residual, wide_attention
from dspark_layer_reference import CASES, finish_reference, wide_rotate
from test_dspark_projection import operations, tensor


class DSparkLayerTests(unittest.TestCase):
    @staticmethod
    def local_parameters():
        result = {}
        for name,(shape,axis) in SPECIFICATIONS.items():
            if axis is None:
                packed = (1,1,1,shape[0])
            elif axis == 0:
                packed = (1,1,shape[1],shape[0]//2)
            else:
                packed = (1,1,shape[1]//2,shape[0])
            result[name] = tensor(packed)
        return result

    @staticmethod
    def shape_runtime():
        runtime = operations()
        runtime.typecast.side_effect = lambda value,dtype:tensor(value.shape,dtype)
        runtime.add.side_effect = lambda first,second,**kwargs:tensor(first.shape,kwargs['dtype'])
        runtime.mul.side_effect = lambda first,second,**kwargs:tensor(first.shape,kwargs['dtype'])
        runtime.multiply = MagicMock(side_effect=lambda first,second,**kwargs:tensor(first.shape,kwargs.get('dtype',first.dtype)))
        runtime.reshape = MagicMock(side_effect=lambda value,shape:tensor(shape,value.dtype))
        def transpose(value,first,second):
            shape = list(value.shape)
            shape[first],shape[second] = shape[second],shape[first]
            return tensor(tuple(shape),value.dtype)
        runtime.transpose = MagicMock(side_effect=transpose)
        def concat(values,*,dim,**kwargs):
            shape = list(values[0].shape)
            shape[dim] = sum(value.shape[dim] for value in values)
            return tensor(tuple(shape),values[0].dtype)
        runtime.concat.side_effect = concat
        runtime.silu = MagicMock(side_effect=lambda value,**kwargs:tensor(value.shape,value.dtype))
        return runtime

    def test_attention_uses_all_local_heads_full_history_and_masked_padding(self):
        runtime = self.shape_runtime()
        parameters = self.local_parameters()
        context,noise = tensor((1,1,32,5120)),tensor((1,1,32,5120))
        tables = {name:[tensor((1,1,rows,128)),tensor((1,1,rows,128))] for name,rows in (('q',32),('k',64))}
        mask,live = tensor((1,1,32,64)),tensor((1,1,32,1),'fp32')
        def project(ops,value,weight,retain,*,rounded=True):
            return retain(tensor((1,1,value.shape[2],weight.shape[-1]),'bf16' if rounded else 'fp32'))
        with patch('dspark_layer.linear',side_effect=project) as projected, \
                patch('dspark_layer.norm',side_effect=lambda ops,value,gamma,retain:value), \
                patch('dspark_layer.rotate',side_effect=lambda ops,value,cosine,sine,owned,**kwargs:value), \
                patch('dspark_layer.attend',return_value=tensor((1,16,32,128))) as attended:
            result = attention_partial(runtime,context,noise,parameters,tables,mask,live,lambda value:value,mask_validated=True)
        self.assertEqual(set(result),set(PHASES['attention']))
        self.assertEqual(projected.call_count,6)
        self.assertEqual(tuple(result['q_heads'].shape),(1,16,32,128))
        self.assertEqual(tuple(result['k_heads'].shape),(1,4,64,128))
        self.assertEqual(tuple(result['v_heads'].shape),(1,4,64,128))
        self.assertEqual(attended.call_args.kwargs,dict(context_rows=32,mask_validated=True,key_chunk_size=64))
        self.assertIs(runtime.multiply.call_args.args[1],live)
        self.assertEqual(result['attention_partial'].dtype,'fp32')

    def test_mlp_executes_gate_up_silu_product_and_full_down(self):
        runtime = self.shape_runtime()
        parameters = self.local_parameters()
        def project(ops,value,weight,retain,*,rounded=True):
            return retain(tensor((1,1,value.shape[2],weight.shape[-1]),'bf16' if rounded else 'fp32'))
        with patch('dspark_layer.linear',side_effect=project) as projected, \
                patch('dspark_layer.norm',side_effect=lambda ops,value,gamma,retain:value):
            result = mlp_partial(runtime,tensor((1,1,32,5120),'fp32'),tensor((1,1,32,5120),'fp32'),
                tensor((1,1,32,5120)),parameters,lambda value:value)
        self.assertEqual(set(result),set(PHASES['mlp']))
        self.assertEqual(projected.call_count,3)
        self.assertEqual(result['gate'].shape,(1,1,32,8704))
        self.assertEqual(result['up'].shape,(1,1,32,8704))
        self.assertEqual(runtime.silu.call_args.args[0].dtype,'fp32')
        self.assertEqual(result['activation'].dtype,'bf16')
        self.assertEqual(result['down_partial'].shape,(1,1,32,5120))
        self.assertEqual(result['down_partial'].dtype,'fp32')

    def test_wide_attention_retains_every_capture_allocation_and_narrows_once(self):
        runtime = self.shape_runtime()
        mesh = object()
        values = [tensor(shape) for shape in ((1,16,32,128),(1,4,64,128),(1,4,64,128),(1,1,32,64))]
        allocations = [tensor((1,16,32,64),'fp32'),tensor((1,16,32,128),'fp32')]
        owned = []
        def compose(*args,**kwargs):
            kwargs['trace_owned'].extend(allocations)
            return allocations[-1]
        with patch('dspark_layer.composed_draft_attention',side_effect=compose) as composed:
            result = wide_attention(runtime,mesh,*values,lambda value:owned.append(value) or value)
        self.assertEqual(owned,[*allocations,result])
        self.assertEqual(composed.call_args.args,(runtime,mesh,*values))
        self.assertEqual({name:value for name,value in composed.call_args.kwargs.items() if name!='trace_owned'},
            dict(explicit_softmax=True,fused_dots=True,cache_dot_tiles=True,fused_row_sum=True))
        runtime.typecast.assert_called_once_with(allocations[-1],'bf16')
        self.assertEqual(result.dtype,'bf16')

    def test_wide_attention_returns_partial_ownership_on_error(self):
        runtime = self.shape_runtime()
        allocated = object()
        owned = []
        def fail(*args,**kwargs):
            kwargs['trace_owned'].append(allocated)
            raise RuntimeError('injected kernel failure')
        with patch('dspark_layer.composed_draft_attention',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                wide_attention(runtime,object(),None,None,None,None,owned.append)
        self.assertEqual(owned,[allocated])
        runtime.typecast.assert_not_called()

    def test_complete_attention_selects_composed_path_only_explicitly(self):
        runtime = self.shape_runtime()
        tables = {name:[tensor((1,1,rows,128)),tensor((1,1,rows,128))] for name,rows in (('q',32),('k',64))}
        mesh = object()
        def project(ops,value,weight,retain,*,rounded=True):
            return retain(tensor((1,1,value.shape[2],weight.shape[-1]),'bf16' if rounded else 'fp32'))
        with patch('dspark_layer.linear',side_effect=project), \
                patch('dspark_layer.norm',side_effect=lambda ops,value,gamma,retain:value), \
                patch('dspark_layer.rotate',side_effect=lambda ops,value,*args,**kwargs:value), \
                patch('dspark_layer.wide_attention',return_value=tensor((1,16,32,128))) as wide, \
                patch('dspark_layer.attend') as native:
            result = attention_partial(runtime,tensor((1,1,32,5120)),tensor((1,1,32,5120)),self.local_parameters(),
                tables,tensor((1,1,32,64)),tensor((1,1,32,1),'fp32'),lambda value:value,
                mask_validated=True,composed_attention=True,mesh=mesh)
        self.assertEqual(set(result),set(PHASES['attention']))
        self.assertIs(wide.call_args.args[1],mesh)
        self.assertIs(result['attended'],wide.return_value)
        native.assert_not_called()

    def test_inventory_is_complete_and_cpu_norm_parameters_replicate(self):
        self.assertEqual(len(SPECIFICATIONS),11)
        self.assertEqual(SPECIFICATIONS['mlp.down_proj.weight'],((5120,17408),1))
        gamma = torch.arange(128).bfloat16()
        packed,sharded = pack_weight('self_attn.q_norm.weight',gamma)
        self.assertFalse(sharded)
        self.assertEqual(tuple(packed.shape),(1,1,1,128))
        self.assertTrue(torch.equal(packed.flatten(),gamma))
        self.assertNotEqual(packed.data_ptr(),gamma.data_ptr())
        for name,value in (('unknown',gamma),('self_attn.q_norm.weight',gamma.float()),
                ('self_attn.q_norm.weight',gamma[:64]),('self_attn.q_norm.weight',torch.full_like(gamma,float('nan')))):
            with self.assertRaises(ValueError):
                pack_weight(name,value)

    def test_q_weight_keeps_every_output_and_full_input_dimension(self):
        weight = torch.arange(4096,dtype=torch.float32).bfloat16().unsqueeze(1).expand(4096,5120)
        packed,sharded = pack_weight('self_attn.q_proj.weight',weight)
        self.assertTrue(sharded)
        self.assertEqual(tuple(packed.shape),(2,1,5120,2048))
        for chip in range(2):
            self.assertTrue(torch.equal(packed[chip,0,0],weight[chip*2048:(chip+1)*2048,0]))
            self.assertTrue(torch.equal(packed[chip,0,-1],packed[chip,0,0]))

    def test_full_projection_geometry_and_explicit_precision(self):
        for input_width,output_width in ((5120,2048),(5120,512),(2048,5120),(5120,8704),(8704,5120)):
            runtime = operations()
            linear(runtime,tensor((1,1,32,input_width)),tensor((1,1,input_width,output_width)),lambda value:value)
            self.assertEqual(runtime.matmul.call_args.kwargs['dtype'],'fp32')
            self.assertEqual(runtime.typecast.call_args.args,(runtime.matmul.return_value,'bf16'))
            config = runtime.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs
            self.assertEqual(config['per_core_N'],(output_width//32+79)//80)
            self.assertEqual(config['in0_block_w'],8)
        runtime = operations()
        with self.assertRaises(ValueError):
            linear(runtime,tensor((1,1,32,5120)),tensor((1,1,5120,32)),lambda value:value)
        runtime.matmul.assert_not_called()

    def test_head_norm_scales_by_128_and_rounds_before_gamma(self):
        runtime = operations()
        runtime.multiply,runtime.sum,runtime.rsqrt = MagicMock(),MagicMock(),MagicMock()
        values = [SimpleNamespace(),SimpleNamespace()]
        runtime.typecast.side_effect = values
        gamma = tensor((1,1,1,128))
        norm(runtime,tensor((1,16,32,128)),gamma,lambda value:value)
        self.assertEqual(runtime.multiply.call_args_list[1].args[1],1/128)
        self.assertEqual(runtime.typecast.call_args_list[1].args[1],'bf16')
        self.assertEqual(runtime.mul.call_args.args,(values[1],gamma))
        self.assertNotIn('fast_and_approximate_mode',runtime.sum.call_args.kwargs)

    def test_tp_sum_is_fp32_then_bf16_then_residual(self):
        runtime = operations()
        summed,output = SimpleNamespace(),SimpleNamespace()
        runtime.add.return_value = summed
        residual = tensor((1,1,32,5120))
        retain = lambda value:value
        with patch('dspark_layer.add_residual',return_value=output) as added:
            result = reduce_residual(runtime,tensor((1,1,32,5120),'fp32'),tensor((1,1,32,5120),'fp32'),residual,retain)
        runtime.add.assert_called_once()
        self.assertEqual(runtime.add.call_args_list[0].kwargs['dtype'],'fp32')
        self.assertEqual(runtime.typecast.call_args.args,(summed,'bf16'))
        added.assert_called_once_with(runtime,residual,runtime.typecast.return_value,retain,widen_inputs=True)
        self.assertEqual(result,(summed,runtime.typecast.return_value,output))

    def test_reference_preserves_residual_rounding_and_half_split_rotary(self):
        first = torch.full((1,1,32,5120),1.002)
        second = torch.full_like(first,.001)
        residual = torch.ones_like(first).bfloat16()
        result = finish_reference([first,second],residual)
        self.assertTrue(torch.equal(result['output'],(first+second).bfloat16()+residual))
        value = torch.arange(128).reshape(1,1,1,128).bfloat16()
        rotated = wide_rotate(value,[torch.zeros_like(value),torch.ones_like(value)])
        self.assertTrue(torch.equal(rotated,torch.cat((-value[...,64:],value[...,:64]),dim=-1)))
        self.assertEqual(tuple(PHASES),('attention','mlp','finish'))
        self.assertEqual(len(CASES),3)


if __name__ == '__main__':
    unittest.main()
