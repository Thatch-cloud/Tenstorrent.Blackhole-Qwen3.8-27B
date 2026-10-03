from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from dspark_backbone_reference import CPUBackbone
from dspark_backend_reference import BACKEND, FP32Backbone, fp32_attention, reviewed_backend, rotate
from dspark_upstream_backbone import control_forward


class BackendReferenceTests(unittest.TestCase):
    def functions(self):
        def original_attention():
            return None
        original_rotary = MagicMock(side_effect=lambda query,key,cosine,sine:(query*cosine+query*sine,key*cosine+key*sine))
        namespace = {'apply_rotary_pos_emb':original_rotary}
        attention = FunctionType(original_attention.__code__,namespace)
        functions = dict(attention=attention,apply_rotary_pos_emb=original_rotary,
            repeat_kv=lambda value,groups:value.repeat_interleave(groups,1))
        return functions,namespace,original_rotary

    def test_port_policy_does_not_mutate_the_frozen_eager_namespace(self):
        def original(self,features,noise,*,context_start,inspect):
            return globals()['full_attention'],globals()['rotate'],features,noise,context_start,inspect
        before = dict(original.__globals__)
        with patch.object(CPUBackbone,'forward',original):
            result = FP32Backbone.__new__(FP32Backbone).forward('features','noise',context_start=8190)
        self.assertEqual(result,(fp32_attention,rotate,'features','noise',8190,None))
        self.assertEqual(original.__globals__.get('full_attention'),before.get('full_attention'))
        self.assertIs(original.__globals__['rotate'],before['rotate'])

    def test_reviewed_registry_executes_noncausal_fp32_gqa_with_bf16_output(self):
        functions,namespace,unused = self.functions()
        reviewed_backend(functions)
        generator = torch.Generator().manual_seed(382711)
        query,key,value = [torch.randn(shape,generator=generator).bfloat16() for shape in
            ((1,8,7,128),(1,2,39,128),(1,2,39,128))]
        output,weights = namespace['ALL_ATTENTION_FUNCTIONS'][BACKEND](SimpleNamespace(num_key_value_groups=4),
            query,key,value,None,dropout=0.,scaling=128**-.5,sliding_window=None)
        expected = fp32_attention(query,key,value).transpose(1,2).contiguous()
        self.assertTrue(torch.equal(output,expected))
        self.assertIsNone(weights)

    def test_rotary_calls_the_original_reviewed_body_with_widened_inputs(self):
        functions,namespace,original = self.functions()
        reviewed_backend(functions)
        inputs = [torch.ones(1,1,4,128,dtype=torch.bfloat16) for index in range(4)]
        result = namespace['apply_rotary_pos_emb'](*inputs)
        self.assertTrue(all(value.dtype==torch.float32 for value in original.call_args.args))
        self.assertTrue(all(value.dtype==torch.bfloat16 and torch.equal(value,inputs[0]*2) for value in result))
        self.assertTrue(all(value.dtype==torch.bfloat16 and bool((value==1).all()) for value in inputs))
        with self.assertRaises(ValueError):
            reviewed_backend(functions)

    def test_control_rejects_undeclared_and_unregistered_backends(self):
        with self.assertRaises(ValueError):
            control_forward({},None,{},None,None,0,None,attention_implementation='unreviewed')
        functions,unused,original = self.functions()
        with self.assertRaises(ValueError):
            control_forward(functions,None,{},None,None,0,None,attention_implementation=BACKEND)

    def test_control_rejects_mask_dropout_window_and_causal_changes(self):
        functions,namespace,unused = self.functions()
        reviewed_backend(functions)
        operands = [torch.ones(shape,dtype=torch.bfloat16) for shape in ((1,8,7,128),(1,2,39,128),(1,2,39,128))]
        for changed in (dict(dropout=.1),dict(sliding_window=2048),dict(is_causal=True),dict(scaling=1.)):
            arguments = dict(dropout=0.,sliding_window=None,scaling=128**-.5)
            arguments.update(changed)
            with self.assertRaises(ValueError):
                namespace['ALL_ATTENTION_FUNCTIONS'][BACKEND](SimpleNamespace(num_key_value_groups=4),*operands,None,**arguments)


if __name__ == '__main__':
    unittest.main()
