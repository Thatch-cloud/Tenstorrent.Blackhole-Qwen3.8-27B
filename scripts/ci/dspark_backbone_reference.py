"""CPU BF16 eager DSpark backbone reference; full attention, no target LM head or serving integration."""

from dspark_intake import TAPS, validate_config
from dspark_rope_tables import DSparkRotary


def rms_norm(value, weight):
    import torch

    if (value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or value.device.type != 'cpu' or weight.device.type != 'cpu' or weight.shape != (value.shape[-1],)):
        raise ValueError('CPU BF16 values and matching direct RMSNorm weight required')
    wide = value.float()
    normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    return normalized * weight


def rotate(value, cosine, sine):
    import torch

    if (value.ndim != 4 or value.shape[-1] != 128 or cosine.shape != sine.shape
            or tuple(cosine.shape) != (1, 1, value.shape[2], 128)
            or any(operand.dtype != torch.bfloat16 or operand.device.type != 'cpu' for operand in (value, cosine, sine))):
        raise ValueError('Matching CPU BF16 half-split rotary heads and tables required')
    half = torch.cat((-value[..., 64:], value[..., :64]), dim=-1)
    return value * cosine + half * sine


def full_attention(query, key, value):
    import torch

    if (query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1
            or key.shape != value.shape or query.shape[-1] != key.shape[-1] or key.shape[1] < 1
            or query.shape[1] < 1 or query.shape[2] < 1 or query.shape[-1] < 1
            or query.shape[1] % key.shape[1] or key.shape[2] < query.shape[2]
            or any(operand.dtype != torch.bfloat16 or operand.device.type != 'cpu' for operand in (query, key, value))):
        raise ValueError('Complete CPU BF16 GQA heads with all historical and proposal keys required')
    groups = query.shape[1] // key.shape[1]
    keys = key.repeat_interleave(groups, dim=1)
    values = value.repeat_interleave(groups, dim=1)
    scores = (query @ keys.transpose(-1, -2)) * (query.shape[-1] ** -.5)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).bfloat16()
    return probabilities @ values


class CPUBackbone:
    def __init__(self, weights, config):
        validate_config(config)
        if config.get('hidden_act') != 'silu':
            raise ValueError('Pinned DSpark SiLU activation required')
        self.weights = weights
        self.rotary = DSparkRotary(config)

    def forward(self, features, noise, *, context_start=0, inspect=None):
        import torch
        from torch.nn.functional import linear, silu

        if (not isinstance(features, dict) or set(features) != set(TAPS)
                or any(not isinstance(value, torch.Tensor) for value in features.values())
                or not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (1, 7, 5120)
                or noise.dtype != torch.bfloat16 or noise.device.type != 'cpu'
                or type(context_start) is not int or context_start < 0
                or (inspect is not None and not callable(inspect))):
            raise ValueError('All five named target feature taps and seven CPU BF16 noise embeddings required')
        shape = tuple(features[TAPS[0]].shape)
        if (len(shape) != 3 or shape[0] != 1 or shape[2] != 5120 or not 1 <= shape[1] <= 4096
                or any(tuple(value.shape) != shape or value.dtype != torch.bfloat16 or value.device.type != 'cpu'
                    for value in features.values()) or context_start + shape[1] + 7 > self.rotary.max_positions
                or any(not torch.isfinite(value).all() for value in (*features.values(), noise))):
            raise ValueError('Finite matching target features and bounded CPU full-attention context required')
        tables = self.rotary.block_tables(context_start, shape[1])
        packed_features = torch.cat([features[layer] for layer in TAPS], dim=-1)
        context = rms_norm(linear(packed_features, self.weights.tensor('fc.weight')), self.weights.tensor('hidden_norm.weight'))
        hidden = noise.clone()

        def observe(stage, value):
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite CPU backbone output at {stage}')
            if inspect is not None:
                inspect(stage, value.clone())

        observe('projected_context', context)
        for layer in range(5):
            prefix = f'layers.{layer}.'
            normalized = rms_norm(hidden, self.weights.tensor(prefix + 'input_layernorm.weight'))
            projections = {}
            for name, heads in (('q', 32), ('k', 8), ('v', 8)):
                weight = self.weights.tensor(prefix + f'self_attn.{name}_proj.weight')
                projected = linear(normalized, weight)
                if name != 'q':
                    projected = torch.cat((linear(context, weight), projected), dim=1)
                projected = projected.reshape(1, projected.shape[1], heads, 128)
                if name != 'v':
                    projected = rms_norm(projected, self.weights.tensor(prefix + f'self_attn.{name}_norm.weight'))
                projections[name] = projected.transpose(1, 2)
            projections['q'] = rotate(projections['q'], *tables['q'])
            projections['k'] = rotate(projections['k'], *tables['k'])
            attended = full_attention(projections['q'], projections['k'], projections['v'])
            hidden = hidden + linear(attended.transpose(1, 2).reshape(1, 7, 4096),
                self.weights.tensor(prefix + 'self_attn.o_proj.weight'))
            observe(f'layer_{layer}_attention_residual', hidden)
            normalized = rms_norm(hidden, self.weights.tensor(prefix + 'post_attention_layernorm.weight'))
            gate = linear(normalized, self.weights.tensor(prefix + 'mlp.gate_proj.weight'))
            up = linear(normalized, self.weights.tensor(prefix + 'mlp.up_proj.weight'))
            hidden = hidden + linear(silu(gate) * up, self.weights.tensor(prefix + 'mlp.down_proj.weight'))
            observe(f'layer_{layer}_output', hidden)
        result = rms_norm(hidden, self.weights.tensor('norm.weight'))
        observe('final_norm', result)
        return result
