"""Host source and ownership checks, not numerical-kernel qualification."""

import unittest

import gdn_shared_qk_compute as candidate


class SharedQKTests(unittest.TestCase):
    def source(self):
        return ('helpers\nvoid kernel_main() {\n' + candidate.NORM_START + '\n'
                '        q_math();\n        WAIT(cb_qn, Kt);\n'
                '        k_math();\n        WAIT(cb_kn, Kt);\n' + candidate.NORM_END + '\n}')

    def test_block_and_serial_share_identical_arithmetic(self):
        block = candidate.compute(self.source(), serial=False)
        serial = candidate.compute(self.source(), serial=True)
        self.assertEqual(block, serial.replace('iterations = get_arg_val<uint32_t>(0)', 'iterations = 1'))
        self.assertIn('q_math();', block)
        self.assertIn('k_math();', block)

    def test_writer_exclusively_consumes_normalized_outputs(self):
        block = candidate.compute(self.source(), serial=False)
        for buffer in ('cb_qn', 'cb_kn'):
            self.assertNotIn(f'WAIT({buffer}, Kt)', block)
            self.assertNotIn(f'POP({buffer}, Kt)', block)
        self.assertIn('WAIT(cb_qf, Kt)', block)
        self.assertIn('WAIT(cb_kf, Kt)', block)

    def test_shared_head_mapping_covers_all_workers(self):
        heads = [candidate.worker_head(worker) for worker in range(96)]
        self.assertEqual([heads.count(head) for head in range(8)], [12] * 8)
        for key, offset in ((False, 0), (True, 32)):
            self.assertEqual([candidate.input_page(head, tile, key=key)
                for head in range(8) for tile in range(4)], list(range(offset, offset + 32)))

    def test_distinct_fp32_norm_factor_buffers(self):
        io, fp32 = candidate.buffer_plan()
        self.assertFalse(set(io) & set(fp32))
        self.assertEqual([fp32[index] for index in (9, 28, 29)], [1, 1, 1])
        self.assertEqual([fp32[index] for index in (10, 11)], [4, 4])

    def test_invalid_policy_and_changed_anchors_fail(self):
        for source in (self.source().replace(candidate.NORM_START, 'changed'), self.source() * 2):
            with self.assertRaises(ValueError):
                candidate.compute(source, serial=False)
        with self.assertRaises(ValueError):
            candidate.compute(self.source(), serial=1)
        for worker in (-1, 96, True):
            with self.assertRaises(ValueError):
                candidate.worker_head(worker)


if __name__ == '__main__':
    unittest.main()
