"""CPU YaRN tables for the pinned DSpark checkpoint, separate from DFlash2's unscaled RoPE."""

import math

from dspark_intake import validate_config


class DSparkRotary:
    def __init__(self, config):
        import torch

        validate_config(config)
        if type(config.get('max_position_embeddings')) is not int or config['max_position_embeddings'] != 262144:
            raise ValueError('Pinned DSpark maximum absolute position required')
        parameters = config['rope_parameters']
        width, base = config['head_dim'], parameters['rope_theta']
        original, factor = parameters['original_max_position_embeddings'], parameters['factor']

        def correction(rotations):
            return width * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(parameters['beta_fast'])), 0)
        high = min(math.ceil(correction(parameters['beta_slow'])), width - 1)
        ramp = ((torch.arange(width // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        extrapolation = 1 - ramp
        wavelengths = base ** (torch.arange(0, width, 2, dtype=torch.float32) / width)
        self.inverse_frequency = (1 / (factor * wavelengths)) * (1 - extrapolation) + (1 / wavelengths) * extrapolation
        self.attention_scaling = 1 + 0.1 * math.log(factor)
        self.correction_range = low, high
        self.max_positions = config['max_position_embeddings']

    def positions(self, identifiers, *, dtype=None):
        import torch

        dtype = torch.bfloat16 if dtype is None else dtype
        if (not isinstance(identifiers, torch.Tensor) or identifiers.device.type != 'cpu'
                or identifiers.dtype != torch.int64 or identifiers.ndim != 1
                or not 1 <= identifiers.numel() <= self.max_positions
                or bool((identifiers < 0).any()) or bool((identifiers >= self.max_positions).any())
                or dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError('Bounded CPU integer absolute positions and explicit BF16 or FP32 tables required')
        angles = identifiers.float()[:, None] * self.inverse_frequency[None, :]
        doubled = torch.cat((angles, angles), dim=-1).reshape(1, 1, identifiers.numel(), 128)
        return tuple((operation(doubled) * self.attention_scaling).to(dtype)
            for operation in (torch.cos, torch.sin))

    def tables(self, start, rows, *, dtype=None):
        import torch

        if (type(start) is not int or type(rows) is not int or start < 0 or rows < 1
                or start + rows > self.max_positions):
            raise ValueError('Bounded absolute YaRN positions required')
        return self.positions(torch.arange(start, start + rows, dtype=torch.int64), dtype=dtype)

    def block_tables(self, context_start, context_rows):
        if type(context_rows) is not int or context_rows < 1:
            raise ValueError('Nonempty absolute target-feature history required')
        key_tables = self.tables(context_start, context_rows + 7)
        return dict(k=key_tables, q=tuple(table[:, :, -7:].clone() for table in key_tables))
