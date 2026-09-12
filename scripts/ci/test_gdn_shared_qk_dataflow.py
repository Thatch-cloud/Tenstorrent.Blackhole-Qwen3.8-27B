"""Host layout and ownership checks for the proposed normalization dataflow."""

import unittest

import gdn_shared_qk_dataflow as candidate


class DataflowTests(unittest.TestCase):
    def test_tile_layout_is_bijective(self):
        self.assertEqual(sorted(candidate.row_element(row, column)
            for row in range(32) for column in range(32)), list(range(1024)))

    def test_serial_row_round_trip_crosses_face_boundaries(self):
        source = list(range(1024))
        assembled = [-1] * 1024
        for row in range(16):
            start = candidate.row_element(row)
            row_zero = source[start:start + 16] + source[start + 256:start + 272]
            for column, value in enumerate(row_zero):
                assembled[candidate.row_element(row, column)] = value
        for row in range(16):
            for column in range(32):
                offset = candidate.row_element(row, column)
                self.assertEqual(assembled[offset], source[offset])
        self.assertEqual(assembled[512:], [-1] * 512)

    def test_reference_assembly_does_not_alias_compute_scratch(self):
        block_io, block_fp32 = candidate.buffer_plan(serial=False)
        serial_io, serial_fp32 = candidate.buffer_plan(serial=True)
        self.assertEqual(set(serial_io) - set(block_io), {31})
        self.assertEqual(set(serial_fp32) - set(block_fp32), {12, 13})
        self.assertFalse(set(serial_io) & set(serial_fp32))
        self.assertEqual(serial_io[31], 8)

    def test_only_full_page_dma_and_writer_drains_before_release(self):
        sources = candidate.kernels()
        self.assertIn('noc.async_read(input, destination, 2048', sources['reader'])
        self.assertIn('query_output, 4096', sources['writer'])
        self.assertIn('key_output, 4096', sources['writer'])
        self.assertIn('noc.async_write_barrier();\n        result.pop_front(4);', sources['writer'])
        self.assertNotIn('noc_async_write(', sources['writer'])


if __name__ == '__main__':
    unittest.main()
