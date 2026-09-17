from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import torch

from dspark_backbone_reference import CPUBackbone, full_attention, rms_norm, rotate
from dspark_intake import TAPS
from test_dspark_intake import configuration


class DSparkBackboneReferenceTests(unittest.TestCase):
    def test_rms_norm_preserves_bf16_rounding_before_direct_weight_multiply(self):
        value = torch.tensor([[.3, -.7, 1.4, 2.1]], dtype=torch.bfloat16)
        weight = torch.tensor([.2, .9, 1.3, 2.4], dtype=torch.bfloat16)
        normalized = (value.float() / torch.sqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        self.assertTrue(torch.equal(rms_norm(value, weight), normalized * weight))
        self.assertFalse(torch.equal(rms_norm(value, weight), normalized * (1 + weight)))

    def test_rotary_uses_half_split_and_bf16_operator_boundaries(self):
        value = torch.arange(128).bfloat16().reshape(1, 1, 1, 128)
        actual = rotate(value, torch.zeros_like(value), torch.ones_like(value))
        self.assertTrue(torch.equal(actual, torch.cat((-value[..., 64:], value[..., :64]), dim=-1)))
        with self.assertRaises(ValueError):
            rotate(value.float(), torch.zeros_like(value), torch.ones_like(value))

    def test_full_gqa_attention_keeps_distant_history_and_future_noise_keys(self):
        query = torch.zeros(1, 4, 7, 4, dtype=torch.bfloat16)
        key = torch.zeros(1, 2, 2056, 4, dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        value[:, 0, 0] = 1024
        value[:, 1, -1] = 1024
        actual = full_attention(query, key, value)
        self.assertTrue(torch.all(actual > 0))
        self.assertTrue(torch.equal(actual[:, 0], actual[:, 1]))
        self.assertTrue(torch.equal(actual[:, 2], actual[:, 3]))
        self.assertTrue(torch.equal(actual[:, :, 0], actual[:, :, -1]))

    def test_gqa_attention_uses_complete_eager_bf16_score_pipeline(self):
        generator = torch.Generator().manual_seed(382603)
        query = torch.randn(1, 4, 7, 8, generator=generator).bfloat16()
        key, value = [torch.randn(1, 2, 13, 8, generator=generator).bfloat16() for _ in range(2)]
        outputs = []
        for head in range(4):
            scores = (query[:, head] @ key[:, head // 2].transpose(-1, -2)) * (8 ** -.5)
            probabilities = torch.softmax(scores.float(), -1).bfloat16()
            outputs.append(probabilities @ value[:, head // 2])
        self.assertTrue(torch.equal(full_attention(query, key, value), torch.stack(outputs, dim=1)))

    def test_invalid_inputs_fail_before_loading_learned_weights(self):
        config = configuration()
        config.update(max_position_embeddings=262144, hidden_act='silu')
        weights = SimpleNamespace(tensor=MagicMock())
        backbone = CPUBackbone(weights, config)
        features = {layer: torch.zeros(1, 2, 5120, dtype=torch.bfloat16) for layer in TAPS}
        noise = torch.zeros(1, 7, 5120, dtype=torch.bfloat16)
        for bad_features, bad_noise, start in (({}, noise, 0), (features, noise[:, :6], 0),
                (features, noise.float(), 0), (features, noise, 262140), (features, noise, True),
                ({**features, TAPS[0]: torch.full_like(features[TAPS[0]], float('nan'))}, noise, 0)):
            with self.assertRaises(ValueError):
                backbone.forward(bad_features, bad_noise, context_start=start)
        weights.tensor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
