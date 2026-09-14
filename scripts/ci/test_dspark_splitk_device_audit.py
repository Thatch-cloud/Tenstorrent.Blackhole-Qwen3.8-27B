from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest

import torch

from dspark_splitk_device_audit import audit_layout
from dspark_splitk_layout import fold_mask, fold_query


class DeviceAuditTests(unittest.TestCase):
    def test_accepts_exact_lanes_and_rejects_mask_row_corruption(self):
        query = torch.arange(65536).reshape(1, 16, 32, 128)
        key = torch.arange(32768).reshape(1, 4, 64, 128)
        mask = torch.arange(2048).reshape(1, 1, 32, 64)
        original = (query, key, key, mask)
        folded = (fold_query(query).reshape(1, 4, 128, 128), key.reshape(4, 1, 64, 128),
            key.reshape(4, 1, 64, 128), fold_mask(mask).reshape(4, 1, 128, 64))
        operations = SimpleNamespace(get_device_tensors=lambda tensor: [tensor, tensor],
            to_torch=lambda tensor: tensor)
        with redirect_stdout(io.StringIO()):
            audit_layout(operations, original, folded)
            corrupted = folded[-1].clone()
            corrupted[2, 0, 4, 0] += 1
            with self.assertRaisesRegex(AssertionError, 'mask'):
                audit_layout(operations, original, folded[:-1] + (corrupted,))


if __name__ == '__main__':
    unittest.main()
