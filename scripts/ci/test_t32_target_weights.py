import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from t32_target_weights import TargetWeights


class TargetWeightsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('t32_target_weights.VOCABULARY', 4))
        stack.enter_context(patch('t32_target_weights.HIDDEN_WIDTH', 2))
        self.value = torch.arange(8).reshape(4, 2).bfloat16()
        payload = self.value.view(torch.uint8).numpy().tobytes()
        header = json.dumps({'model.embed_tokens.weight': dict(dtype='BF16', shape=[4, 2],
            data_offsets=[0, len(payload)])}).encode()
        (self.root / 'weights.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + payload)
        (self.root / 'config.json').write_text(json.dumps(dict(hidden_size=2, vocab_size=4, tie_word_embeddings=True)))
        self.index('weights.safetensors')

    def index(self, shard):
        (self.root / 'model.safetensors.index.json').write_text(json.dumps(
            dict(weight_map={'model.embed_tokens.weight': shard})))

    def test_tied_target_is_loaded_and_fingerprinted(self):
        reader = TargetWeights(self.root)
        self.assertTrue(torch.equal(reader.tensor('embedding'), self.value))
        self.assertTrue(torch.equal(reader.tensor('head'), self.value))
        self.assertEqual(reader.manifest['tensors']['head']['sha256'], reader.manifest['tensors']['embedding']['sha256'])

    def test_external_shard_rejected(self):
        self.index('../weights.safetensors')
        with self.assertRaisesRegex(ValueError, 'basename'):
            TargetWeights(self.root)

    def test_changed_checkpoint_rejected(self):
        reader = TargetWeights(self.root)
        (self.root / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            reader.tensor('head')

    def test_missing_untied_head_rejected(self):
        (self.root / 'config.json').write_text(json.dumps(dict(hidden_size=2, vocab_size=4, tie_word_embeddings=False)))
        with self.assertRaisesRegex(ValueError, 'untied'):
            TargetWeights(self.root)


if __name__ == '__main__':
    unittest.main()
