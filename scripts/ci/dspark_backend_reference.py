"""Declared FP32 attention/rotary CPU policy, retaining BF16 weights and all other backbone boundaries."""

from types import FunctionType

from dspark_backbone_reference import CPUBackbone
from dspark_backend_attribution import fp32_attention
from dspark_layer_reference import wide_rotate


BACKEND = 'qwen_fp32_reference'
POLICY = 'FP32 attention and rotary intermediates; BF16 weights, outputs, linear and RMS boundaries; no native matmul emulation'


def rotate(value, cosine, sine):
    return wide_rotate(value,(cosine,sine))


class FP32Backbone(CPUBackbone):
    def forward(self, features, noise, *, context_start=0, inspect=None):
        original = CPUBackbone.forward
        namespace = {**original.__globals__,'full_attention':fp32_attention,'rotate':rotate}
        forward = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        return forward(self,features,noise,context_start=context_start,inspect=inspect)


def reviewed_backend(functions):
    import torch

    namespace = functions['attention'].__globals__
    if 'ALL_ATTENTION_FUNCTIONS' in namespace or namespace['apply_rotary_pos_emb'] is not functions['apply_rotary_pos_emb']:
        raise ValueError('Fresh reviewed upstream function namespace required')
    original_rotary = functions['apply_rotary_pos_emb']
    repeat_kv = functions['repeat_kv']

    def attention(module, query, key, value, attention_mask, *, dropout, scaling, sliding_window, **kwargs):
        if (attention_mask is not None or dropout!=0. or sliding_window is not None
                or kwargs.get('is_causal',False) is not False or scaling!=128**-.5
                or any(operand.dtype!=torch.bfloat16 or operand.device.type!='cpu' for operand in (query,key,value))
                or query.shape[1]!=key.shape[1]*module.num_key_value_groups):
            raise ValueError('Noncausal full-context CPU reference attention with BF16 boundaries required')
        result = torch.nn.functional.scaled_dot_product_attention(query.float(),
            repeat_kv(key,module.num_key_value_groups).float(),repeat_kv(value,module.num_key_value_groups).float(),
            dropout_p=0.,is_causal=False,scale=scaling)
        return result.bfloat16().transpose(1,2).contiguous(),None

    def rotary(query, key, cosine, sine, *args, **kwargs):
        if any(value.dtype!=torch.bfloat16 for value in (query,key,cosine,sine)):
            raise ValueError('BF16 rotary inputs and tables required before explicit widening')
        values = original_rotary(query.float(),key.float(),cosine.float(),sine.float(),*args,**kwargs)
        return tuple(value.bfloat16() for value in values)

    namespace['ALL_ATTENTION_FUNCTIONS'] = {BACKEND:attention}
    namespace['apply_rotary_pos_emb'] = rotary
    return functions
