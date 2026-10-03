"""Frozen upstream-compatible CPU layer and separate TP arithmetic references for the complete TT port."""

import importlib.util
import json
from pathlib import Path

from dspark_attention import full_mask
from dspark_backbone_reference import full_attention, rms_norm, rotate
from dspark_intake import FILES, TAPS
from dspark_layer import SPECIFICATIONS
from dspark_projection import CPU_REPORT, digest, reference_metadata, tensor_digest
from dspark_rope_tables import DSparkRotary


PROJECTION_REPORT_SHA256 = '082e2ba464320b7ad91312e196b8c5f75406a6d2494ed7f1ebafd184b55303dc'
PROJECTION_OPERANDS_SHA256 = 'ea13e5e19b8c041fdd8560982c03e5bfcc6e2d0e95747d285e95173b4af709e1'
CASES = ((0,0),(0,8190),(1,8190))


def load_fixtures(*, projection_report, projection_operands, cpu_outputs, config):
    import torch

    if (digest(projection_report) != PROJECTION_REPORT_SHA256 or digest(projection_operands) != PROJECTION_OPERANDS_SHA256
            or config.stat().st_size != FILES['config.json'][0] or digest(config) != FILES['config.json'][1]):
        raise ValueError('Pinned successful complete FC/norm evidence, observed tensors and YaRN config required')
    reference = reference_metadata()
    if digest(cpu_outputs) != reference['outputs_sha256']:
        raise ValueError('Frozen upstream-compared CPU backbone output tensors required')
    report = json.loads(projection_report.read_text())
    if (report['passed'] is not True or report['closed_cleanly'] is not True or report['checkpoint_closed'] is not True
            or report['mode'] != 'matrix' or report['composed_norm'] is not True
            or report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']
            or report['reference'] != reference):
        raise ValueError('Complete unmodified and clean FC/norm matrix required, not an eager diagnostic')
    projection = torch.load(projection_operands,weights_only=True,map_location='cpu')
    cpu = torch.load(cpu_outputs,weights_only=True,map_location='cpu')
    cpu_report = json.loads(Path(__file__).with_name(CPU_REPORT).read_text())
    spec = importlib.util.spec_from_file_location('dspark_layer_frozen_inputs',Path(__file__).with_name('dspark-backbone-cpu.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rotary = DSparkRotary(json.loads(config.read_text()))
    fixtures = []
    for case,(pattern,start) in enumerate(CASES):
        features,noise = module.inputs(pattern)
        if [tensor_digest(value) for value in (*features.values(),noise)] != cpu_report['cases'][case]['input_sha256']:
            raise ValueError('Layer inputs differ from the original full CPU backbone fixtures')
        for name,value in cpu[case]['stages'].items():
            original = next(entry for entry in cpu_report['stages'] if entry['case']==case and entry['stage']==name)
            if tensor_digest(value) != original['sha256']:
                raise ValueError('CPU layer output differs from frozen upstream-qualified evidence')
        contexts = projection['eager']['tail',pattern]['context']
        if len(contexts) != 2 or tensor_digest(contexts[0]) != tensor_digest(contexts[1]):
            raise ValueError('Both observed TP projection contexts must agree bitwise')
        tables = rotary.block_tables(start,32)
        padded = {}
        for name,rows in (('q',32),('k',64)):
            pair = [torch.ones(1,1,rows,128,dtype=torch.bfloat16),torch.zeros(1,1,rows,128,dtype=torch.bfloat16)]
            for destination,source in zip(pair,tables[name],strict=True):
                destination[:,:,:source.shape[2]] = source
            padded[name] = pair
        live = torch.zeros(1,1,32,1,dtype=torch.float32)
        live[:,:,:7] = 1
        fixtures.append(dict(case=case,pattern=pattern,start=start,context=contexts[0].clone(),
            noise=torch.nn.functional.pad(noise.unsqueeze(1),(0,0,0,25)),tables=padded,
            mask=full_mask(32,key_multiple=64),live=live,cpu_context=cpu[case]['stages']['projected_context'],
            cpu_attention_residual=cpu[case]['stages']['layer_0_attention_residual'],cpu_output=cpu[case]['stages']['layer_0_output']))
    return fixtures,reference


def cpu_layer(context, noise, weights, tables):
    import torch
    from torch.nn.functional import linear, silu

    if (tuple(context.shape)!=(1,32,5120) or tuple(noise.shape)!=(1,7,5120)
            or context.dtype != torch.bfloat16 or noise.dtype != torch.bfloat16 or set(weights) != set(SPECIFICATIONS)):
        raise ValueError('Complete unpadded learned CPU layer inputs and eleven parameters required')
    normalized = rms_norm(noise,weights['input_layernorm.weight'])
    projected = {}
    for name,heads in (('q',32),('k',8),('v',8)):
        weight = weights[f'self_attn.{name}_proj.weight']
        value = linear(normalized,weight)
        if name != 'q':
            value = torch.cat((linear(context,weight),value),dim=1)
        value = value.reshape(1,value.shape[1],heads,128)
        if name != 'v':
            value = rms_norm(value,weights[f'self_attn.{name}_norm.weight'])
        projected[name] = value.transpose(1,2)
    for name,rows in (('q',7),('k',39)):
        projected[name] = rotate(projected[name],*[table[:,:,:rows] for table in tables[name]])
    attended = full_attention(projected['q'],projected['k'],projected['v'])
    residual = noise + linear(attended.transpose(1,2).reshape(1,7,4096),weights['self_attn.o_proj.weight'])
    normalized = rms_norm(residual,weights['post_attention_layernorm.weight'])
    gate,up = [linear(normalized,weights[f'mlp.{name}_proj.weight']) for name in ('gate','up')]
    output = residual + linear(silu(gate)*up,weights['mlp.down_proj.weight'])
    return dict(attention_residual=residual,output=output)


def local_weight(weights, name, chip):
    shape,axis = SPECIFICATIONS[name]
    value = weights[name]
    if tuple(value.shape) != shape or chip not in (0,1):
        raise ValueError('Complete learned CPU parameter and TP2 chip required')
    if axis is None:
        return value
    return value.chunk(2,dim=axis)[chip].T.contiguous()


def wide_rotate(value, tables):
    import torch

    half = torch.cat((-value[...,64:].float(),value[...,:64].float()),dim=-1)
    return (value.float()*tables[0].float()+half*tables[1].float()).bfloat16()


def attention_reference(fixture, weights, chip):
    import torch

    noise,context = fixture['noise'],fixture['context']
    result = dict(input_norm=rms_norm(noise,weights['input_layernorm.weight']))
    for name,heads in (('q',16),('k',4),('v',4)):
        weight = local_weight(weights,f'self_attn.{name}_proj.weight',chip)
        projected = (result['input_norm'].float() @ weight.float()).bfloat16()
        if name != 'q':
            projected = torch.cat(((context.float() @ weight.float()).bfloat16(),projected),dim=2)
        result[name+'_projection'] = projected
        rows = projected.shape[2]
        result[name+'_heads'] = projected.reshape(1,rows,heads,128).transpose(1,2).contiguous()
        if name != 'v':
            result[name+'_norm'] = rms_norm(result[name+'_heads'],weights[f'self_attn.{name}_norm.weight'])
            result[name+'_rotary'] = wide_rotate(result[name+'_norm'],fixture['tables'][name])
    result['attended'] = torch.nn.functional.scaled_dot_product_attention(result['q_rotary'].float(),
        result['k_rotary'].float().repeat_interleave(4,dim=1),result['v_heads'].float().repeat_interleave(4,dim=1),
        attn_mask=fixture['mask'].float(),is_causal=False).bfloat16()
    result['merged'] = result['attended'].transpose(1,2).reshape(1,1,32,2048).contiguous()
    result['attention_partial'] = (result['merged'].float() @ local_weight(weights,'self_attn.o_proj.weight',chip).float())*fixture['live']
    return result


def mlp_reference(partials, noise, weights, chip):
    import torch

    summed = partials[0]+partials[1]
    narrowed = summed.bfloat16()
    residual = noise+narrowed
    normalized = rms_norm(residual,weights['post_attention_layernorm.weight'])
    gate,up = [(normalized.float() @ local_weight(weights,f'mlp.{name}_proj.weight',chip).float()).bfloat16()
        for name in ('gate','up')]
    activation = torch.nn.functional.silu(gate.float()).bfloat16()
    product = activation*up
    down = product.float() @ local_weight(weights,'mlp.down_proj.weight',chip).float()
    return dict(attention_sum=summed,attention_narrowed=narrowed,attention_residual=residual,post_norm=normalized,
        gate=gate,up=up,activation=activation,product=product,down_partial=down)


def finish_reference(partials, residual):
    summed = partials[0]+partials[1]
    narrowed = summed.bfloat16()
    return dict(down_sum=summed,down_narrowed=narrowed,output=residual+narrowed)
