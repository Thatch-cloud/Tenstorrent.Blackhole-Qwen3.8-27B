"""Complete learned DSpark layer arithmetic, with explicit caller-owned TP reduction boundaries."""

from dspark_attention import execute as attend
from draft_attention import composed_draft_attention
from dspark_projection import compute_config, require_tensor
from dspark_residual import add as add_residual
from dspark_rotary_device import execute as rotate


POLICY = 'TP2 BF16 learned weights; HiFi4 FP32 projections; explicit BF16 casts; composed FP32 RMS/rotary/SiLU; precise chunk64 attention; FP32 TP sums; explicitly widened residual operands'
WIDE_POLICY = POLICY.replace('precise chunk64 attention','cached FP32 SFPU dot/row-sum attention')
CONTEXT_ROWS = 32
PROPOSAL_ROWS = 7
PADDED_ROWS = 32
KEY_ROWS = 64
SPECIFICATIONS = {
    'input_layernorm.weight': ((5120,),None),
    'self_attn.q_proj.weight': ((4096,5120),0),
    'self_attn.k_proj.weight': ((1024,5120),0),
    'self_attn.v_proj.weight': ((1024,5120),0),
    'self_attn.q_norm.weight': ((128,),None),
    'self_attn.k_norm.weight': ((128,),None),
    'self_attn.o_proj.weight': ((5120,4096),1),
    'post_attention_layernorm.weight': ((5120,),None),
    'mlp.gate_proj.weight': ((17408,5120),0),
    'mlp.up_proj.weight': ((17408,5120),0),
    'mlp.down_proj.weight': ((5120,17408),1),
}
ATTENTION_STAGES = ('input_norm','q_projection','k_projection','v_projection','q_heads','k_heads','v_heads',
    'q_norm','k_norm','q_rotary','k_rotary','attended','merged','attention_partial')
MLP_STAGES = ('attention_sum','attention_narrowed','attention_residual','post_norm','gate','up','activation','product','down_partial')
FINISH_STAGES = ('down_sum','down_narrowed','output')
PHASES = {'attention':ATTENTION_STAGES,'mlp':MLP_STAGES,'finish':FINISH_STAGES}
REPLAY_CASES = (0,2,1,0)
EXACT = {('mlp',name) for name in ('attention_sum','attention_narrowed','attention_residual','native_product')}
EXACT.update(('finish',name) for name in FINISH_STAGES)
ADDITIONAL = {'attention':(), 'mlp':('upstream_attention_residual','own_context_attention_residual','native_activation','native_product'),
    'finish':('upstream_output','own_context_output')}
INPUT_COUNTS = {'attention':8,'mlp':3,'finish':3}


def pack_weight(name, value):
    import torch

    if name not in SPECIFICATIONS:
        raise ValueError('Declared complete DSpark layer parameter required')
    shape,axis = SPECIFICATIONS[name]
    if (not isinstance(value,torch.Tensor) or tuple(value.shape) != shape
            or value.dtype != torch.bfloat16 or value.device.type != 'cpu' or not torch.isfinite(value).all()):
        raise ValueError('Complete finite CPU BF16 learned parameter required')
    if axis is None:
        return value.reshape(1,1,1,shape[0]).clone(),False
    return torch.stack([part.T.contiguous() for part in value.chunk(2,dim=axis)]).unsqueeze(1),True


def norm(operations, value, gamma, retain):
    shape = tuple(value.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] not in (1,4,16) or shape[2] not in (32,64) or shape[3] not in (128,5120):
        raise ValueError('Complete padded hidden states or local DSpark query/key heads required')
    require_tensor(operations,value,shape,operations.bfloat16)
    require_tensor(operations,gamma,(1,1,1,shape[-1]),operations.bfloat16)
    memory = operations.DRAM_MEMORY_CONFIG
    wide = retain(operations.typecast(value,operations.float32))
    squared = retain(operations.multiply(wide,wide,fast_and_approximate_mode=False,memory_config=memory))
    summed = retain(operations.sum(squared,dim=3,keepdim=True,compute_kernel_config=compute_config(operations),memory_config=memory))
    variance = retain(operations.multiply(summed,1/shape[-1],fast_and_approximate_mode=False,memory_config=memory))
    stabilized = retain(operations.add(variance,1e-6,dtype=operations.float32,memory_config=memory))
    reciprocal = retain(operations.rsqrt(stabilized,fast_and_approximate_mode=False,memory_config=memory))
    normalized = retain(operations.multiply(wide,reciprocal,fast_and_approximate_mode=False,memory_config=memory))
    rounded = retain(operations.typecast(normalized,operations.bfloat16))
    return retain(operations.mul(rounded,gamma,dtype=operations.bfloat16,memory_config=memory))


def linear(operations, value, weight, retain, *, rounded=True):
    shape,weight_shape = tuple(value.shape),tuple(weight.shape)
    if (len(shape) != 4 or shape[:2] != (1,1) or shape[2] not in (32,64)
            or len(weight_shape) != 4 or weight_shape[:2] != (1,1) or shape[-1] != weight_shape[-2]
            or (shape[-1],weight_shape[-1]) not in ((5120,2048),(5120,512),(2048,5120),(5120,8704),(8704,5120))
            or type(rounded) is not bool):
        raise ValueError('Complete local DSpark projection dimensions required')
    for value_to_check in (value,weight):
        require_tensor(operations,value_to_check,tuple(value_to_check.shape),operations.bfloat16)
    tiles = weight_shape[-1]//32
    per_core = (tiles+79)//80
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8,10),
        in0_block_w=8,out_subblock_h=1,out_subblock_w=1,per_core_M=shape[2]//32,per_core_N=per_core,
        fuse_batch=True,fused_activation=None,mcast_in0=True)
    output = retain(operations.matmul(value,weight,dtype=operations.float32,program_config=program,
        compute_kernel_config=compute_config(operations),memory_config=operations.DRAM_MEMORY_CONFIG))
    return retain(operations.typecast(output,operations.bfloat16)) if rounded else output


def wide_attention(operations, mesh, query, key, value, mask, retain):
    if mesh is None or not callable(retain):
        raise ValueError('Explicit mesh and caller-owned attention tensors required')
    owned = []
    try:
        output = composed_draft_attention(operations,mesh,query,key,value,mask,explicit_softmax=True,
            fused_dots=True,cache_dot_tiles=True,fused_row_sum=True,trace_owned=owned)
    finally:
        for tensor in owned:
            retain(tensor)
    return retain(operations.typecast(output,operations.bfloat16))


def attention_partial(operations, context, noise, weights, tables, mask, live_rows, retain, *,
        mask_validated=False, composed_attention=False, mesh=None):
    if type(composed_attention) is not bool or (composed_attention and mesh is None):
        raise ValueError('Explicit attention arithmetic and mesh for composed attention required')
    if set(weights) != set(SPECIFICATIONS) or not callable(retain) or mask_validated is not True:
        raise ValueError('All eleven learned parameters, ownership and validated full-context mask required')
    for value in (context,noise):
        require_tensor(operations,value,(1,1,32,5120),operations.bfloat16)
    require_tensor(operations,live_rows,(1,1,32,1),operations.float32)
    if set(tables) != {'q','k'} or any(len(values) != 2 for values in tables.values()):
        raise ValueError('Absolute-position query and key rotary tables required')
    result = dict(input_norm=norm(operations,noise,weights['input_layernorm.weight'],retain))
    for name,heads in (('q',16),('k',4),('v',4)):
        parameter = weights[f'self_attn.{name}_proj.weight']
        projected = linear(operations,result['input_norm'],parameter,retain)
        if name != 'q':
            historical = linear(operations,context,parameter,retain)
            projected = retain(operations.concat([historical,projected],dim=2,memory_config=operations.DRAM_MEMORY_CONFIG))
        result[name+'_projection'] = projected
        rows = 32 if name == 'q' else 64
        reshaped = retain(operations.reshape(projected,(1,rows,heads,128)))
        result[name+'_heads'] = retain(operations.transpose(reshaped,1,2))
        if name != 'v':
            result[name+'_norm'] = norm(operations,result[name+'_heads'],weights[f'self_attn.{name}_norm.weight'],retain)
            rotary_owned = []
            try:
                result[name+'_rotary'] = rotate(operations,result[name+'_norm'],*tables[name],rotary_owned,composed=True)
            finally:
                for value in rotary_owned:
                    retain(value)
    if composed_attention:
        result['attended'] = wide_attention(operations,mesh,result['q_rotary'],result['k_rotary'],result['v_heads'],mask,retain)
    else:
        result['attended'] = retain(attend(operations,result['q_rotary'],result['k_rotary'],result['v_heads'],mask,
            context_rows=32,mask_validated=True,key_chunk_size=64))
    transposed = retain(operations.transpose(result['attended'],1,2))
    result['merged'] = retain(operations.reshape(transposed,(1,1,32,2048)))
    partial = linear(operations,result['merged'],weights['self_attn.o_proj.weight'],retain,rounded=False)
    result['attention_partial'] = retain(operations.multiply(partial,live_rows,dtype=operations.float32,
        fast_and_approximate_mode=False,memory_config=operations.DRAM_MEMORY_CONFIG))
    return result


def reduce_residual(operations, first, second, residual, retain):
    for value in (first,second):
        require_tensor(operations,value,(1,1,32,5120),operations.float32)
    require_tensor(operations,residual,(1,1,32,5120),operations.bfloat16)
    summed = retain(operations.add(first,second,dtype=operations.float32,memory_config=operations.DRAM_MEMORY_CONFIG))
    narrowed = retain(operations.typecast(summed,operations.bfloat16))
    output = add_residual(operations,residual,narrowed,retain,widen_inputs=True)
    return summed,narrowed,output


def mlp_partial(operations, first, second, noise, weights, retain):
    if set(weights) != set(SPECIFICATIONS) or not callable(retain):
        raise ValueError('All eleven learned parameters and explicit tensor ownership required')
    summed,narrowed,residual = reduce_residual(operations,first,second,noise,retain)
    normalized = norm(operations,residual,weights['post_attention_layernorm.weight'],retain)
    gate = linear(operations,normalized,weights['mlp.gate_proj.weight'],retain)
    up = linear(operations,normalized,weights['mlp.up_proj.weight'],retain)
    widened = retain(operations.typecast(gate,operations.float32))
    activated = retain(operations.silu(widened,memory_config=operations.DRAM_MEMORY_CONFIG))
    activated = retain(operations.typecast(activated,operations.bfloat16))
    product = retain(operations.mul(activated,up,dtype=operations.bfloat16,memory_config=operations.DRAM_MEMORY_CONFIG))
    down = linear(operations,product,weights['mlp.down_proj.weight'],retain,rounded=False)
    return dict(attention_sum=summed,attention_narrowed=narrowed,attention_residual=residual,post_norm=normalized,
        gate=gate,up=up,activation=activated,product=product,down_partial=down)


def finish(operations, first, second, residual, retain):
    summed,narrowed,output = reduce_residual(operations,first,second,residual,retain)
    return dict(down_sum=summed,down_narrowed=narrowed,output=output)
