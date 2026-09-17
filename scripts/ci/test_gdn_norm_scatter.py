"""Host layout checks, not device correctness or speed qualification."""

import struct
import unittest
from pathlib import Path

import gdn_norm_scatter as scatter


class ScatterTests(unittest.TestCase):
    def test_every_head_row_and_padding_bit(self):
        for rows in (1, 2, 4, 8, 16, 32):
            source = b''.join(struct.pack('<I', word) for word in range(rows * 96 * 32))
            for head in range(24):
                output = bytearray(4 * 4096)
                touched = set()
                for page, offset, destination, size in scatter.transfers(rows, head):
                    addresses = set(range(destination, destination + size))
                    self.assertFalse(touched & addresses)
                    touched.update(addresses)
                    output[destination:destination + size] = source[page * 128 + offset:page * 128 + offset + size]
                for token in range(32):
                    for column in range(128):
                        address = column // 32 * 4096 + scatter.batch.tile_element(token, column % 32) * 4
                        expected = token * 3072 + head * 128 + column if token < rows else 0
                        self.assertEqual(struct.unpack_from('<I', output, address)[0], expected)

    def test_source_changes_only_transfer_loop(self):
        control = scatter.batch.READER_BODY
        candidate = scatter.replace_reader(control)
        self.assertEqual(candidate[:candidate.index(scatter.START)], control[:control.index(scatter.START)])
        self.assertEqual(candidate[candidate.index(scatter.END):], control[control.index(scatter.END):])
        self.assertEqual(scatter.SCATTER.count('noc_async_read_barrier();'), 1)
        self.assertNotIn('stick.', scatter.SCATTER)

    def test_drift_fails_closed(self):
        for source in ('', scatter.batch.READER_BODY * 2,
                       scatter.batch.READER_BODY.replace('stick.pop_front(1);', 'stick.pop_front(2);')):
            with self.assertRaises(ValueError):
                scatter.replace_reader(source)

    def test_invalid_geometry(self):
        for rows, head in ((True, 0), (3, 0), (16, True), (16, -1), (16, 24)):
            with self.assertRaises(ValueError):
                scatter.transfers(rows, head)

    def test_pinned_compute_writer_and_recurrence_unchanged(self):
        root = Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source'
        if not root.exists():
            self.skipTest('Pinned kernel source unavailable')
        control = scatter.batch.load_kernels(root)
        candidate = scatter.load_kernels(root)
        self.assertEqual(control['recurrence'], candidate['recurrence'])
        for role in ('compute', 'writer'):
            self.assertEqual(control['norm_gate'][role], candidate['norm_gate'][role])
        self.assertNotEqual(control['norm_gate']['reader'], candidate['norm_gate']['reader'])


if __name__ == '__main__':
    unittest.main()
