import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dspark_t32_weights import load_checkpoint


class T32WeightTests(unittest.TestCase):
    def test_exact_selected_bytes_and_corruption_detection(self):
        matrices = {'first': torch.arange(8).reshape(2, 4).bfloat16(),
                    'second': torch.ones(2, 4, dtype=torch.bfloat16)}
        expected = {name: dict(shape=list(value.shape), bytes=16,
            sha256=hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest())
            for name, value in matrices.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            header = json.dumps({name: dict(dtype='BF16', shape=[2, 4], data_offsets=[index * 16, (index + 1) * 16])
                for index, name in enumerate(matrices)}).encode()
            content = len(header).to_bytes(8, 'little') + header + b''.join(value.view(torch.uint8).numpy().tobytes()
                for value in matrices.values())
            path.write_bytes(content)
            with patch.multiple('dspark_t32_weights', TENSORS=expected, HEADER_BYTES=len(header),
                    HEADER_SHA256=hashlib.sha256(header).hexdigest(), CHECKPOINT_BYTES=len(content)):
                _, first, second = load_checkpoint(path)
                self.assertTrue(torch.equal(first, matrices['first']))
                self.assertTrue(torch.equal(second, matrices['second']))
                path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
                with self.assertRaises(ValueError):
                    load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
