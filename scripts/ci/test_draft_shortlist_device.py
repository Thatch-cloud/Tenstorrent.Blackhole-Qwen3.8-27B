from types import SimpleNamespace
import unittest

import torch

from draft_shortlist_device import prepare_head, select_token


class ShortlistDeviceTests(unittest.TestCase):
    def test_device_selection_maps_global_ids_without_collective_or_host_read(self):
        calls = []

        def tensor(shape, dtype='bf16', layout='tile', data=None):
            return SimpleNamespace(shape=shape, dtype=dtype, layout=layout,
                                   memory_config=lambda: 'dram', data=data)

        scores = torch.zeros(1, 1, 1, 32768)
        scores[..., 123] = scores[..., 231] = 5
        mapping = torch.arange(32768).reshape(1, 1, 1, -1) * 7

        def linear(hidden, head, **kwargs):
            calls.append('linear')
            return tensor(scores.shape, data=scores)

        def to_layout(value, layout):
            calls.append('untilize')
            return tensor(value.shape, layout=layout, data=value.data)

        def argmax(value, **kwargs):
            calls.append('argmax')
            data = value.data.argmax(**kwargs)
            return tensor(data.shape, dtype='u32', layout='rm', data=data)

        def gather(value, dim, index):
            calls.append('gather')
            data = value.data.gather(dim, index.data)
            return tensor(data.shape, dtype='u32', layout='rm', data=data)

        operations = SimpleNamespace(bfloat16='bf16', uint32='u32', TILE_LAYOUT='tile',
                                     ROW_MAJOR_LAYOUT='rm', DRAM_MEMORY_CONFIG='dram',
                                     linear=linear, to_layout=to_layout, argmax=argmax, gather=gather)
        hidden = tensor((1, 1, 1, 5120))
        head = tensor((1, 1, 5120, 32768))
        identifiers = tensor(mapping.shape, dtype='u32', layout='rm', data=mapping)
        owned = []
        result = select_token(operations, hidden, head, identifiers, owned)
        self.assertEqual(result.data.item(), 861)
        self.assertEqual(calls, ['linear', 'untilize', 'argmax', 'gather'])
        self.assertEqual(len(owned), 4)
        self.assertTrue(all(value is not head and value is not identifiers for value in owned))
        hidden.shape = (1, 1, 8, 5120)
        with self.assertRaises(ValueError):
            select_token(operations, hidden, head, identifiers, [])
        self.assertEqual(len(calls), 4)

    def test_prepare_rejects_wrong_mesh_or_unbounded_head_before_allocation(self):
        for devices, width in ((1, 32768), (2, 248320)):
            with self.assertRaises(ValueError):
                prepare_head(SimpleNamespace(), SimpleNamespace(get_num_devices=lambda: devices),
                             SimpleNamespace(shape=(248320, 5120)), range(width), [])


if __name__ == '__main__':
    unittest.main()
