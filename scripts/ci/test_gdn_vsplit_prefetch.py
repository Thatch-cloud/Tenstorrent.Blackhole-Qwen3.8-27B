"""Host mapping and source checks for the experimental L1 input cache."""

import os
from pathlib import Path
import struct
import tempfile
import unittest

import gdn_vsplit as split
import gdn_vsplit_prefetch as prefetch


ROOT = Path(os.environ.get('GDN_VSPLIT_SOURCE_ROOT', str(
    Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359'
    / 'qwen-hardware-inventory-34009341359/gdn-source')))


class LayoutTests(unittest.TestCase):
    def test_all_physical_elements_are_bijective(self):
        self.assertEqual(sorted(prefetch.source_element(row, column)
            for row in range(32) for column in range(32)), list(range(1024)))

    def test_row_copy_preserves_both_faces_and_zero_padding(self):
        for row in range(32):
            source = bytearray(struct.pack('<1024H', *([0x7fc0] * 1024)))
            for column in range(32):
                struct.pack_into('<H', source, prefetch.source_element(row, column) * 2, 0x3f80 + column)
            target = bytearray(2048)
            offset = (512 * (row // 16) + 16 * (row % 16)) * 2
            target[:32] = source[offset:offset + 32]
            target[512:544] = source[offset + 512:offset + 544]
            for column in range(32):
                self.assertEqual(struct.unpack_from('<H', target, prefetch.source_element(0, column) * 2)[0], 0x3f80 + column)
            self.assertEqual(target[32:512], bytes(480))
            self.assertEqual(target[544:], bytes(1504))

    def test_scalar_halfword_selection_masks_neighbour(self):
        for row in range(32):
            for head in range(24):
                address = prefetch.source_element(row, head) * 2
                word = (0x7fc0 << 16) | 0x3f81 if not address & 2 else (0x3f81 << 16) | 0x7f80
                selected = (word >> ((address & 2) * 8)) & 0xffff
                self.assertEqual(selected, 0x3f81)

    def test_only_free_recurrence_cb31_is_added(self):
        for stage in ('recurrence', 'norm_gate'):
            old_io, old_fp32 = split.cb_plan(stage)
            new_io, new_fp32 = split.cb_plan(stage, prefetch_inputs=True)
            self.assertEqual(old_fp32, new_fp32)
            if stage == 'recurrence':
                self.assertNotIn(31, old_io)
                self.assertNotIn(31, old_fp32)
                self.assertEqual(new_io, old_io | {31: 11})
                self.assertEqual(sum(new_io.values()) * 2048 + sum(new_fp32.values()) * 4096, 305152)
            else:
                self.assertEqual(old_io, new_io)

    def test_invalid_coordinates_and_implicit_flags_rejected(self):
        for row, column in ((32, 0), (-1, 0), (0, 32), (True, 0)):
            with self.assertRaises(ValueError):
                prefetch.source_element(row, column)
        for flag in (1, None, 'true'):
            with self.assertRaises(ValueError):
                split.cb_plan('recurrence', prefetch_inputs=flag)

    def test_missing_runtime_headers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prefetch.validate_runtime(Path(directory))


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / split.native.KERNEL_ROOT).exists():
            raise unittest.SkipTest('Pinned source unavailable')
        cls.control = split.load_kernels(ROOT)
        cls.candidate = prefetch.load_kernels(ROOT)

    def test_compute_writer_and_norm_stage_are_unchanged(self):
        self.assertEqual(self.control['norm_gate'], self.candidate['norm_gate'])
        for role in ('compute', 'writer'):
            self.assertEqual(self.control['recurrence'][role], self.candidate['recurrence'][role])

    def test_full_cache_lifetime_wraps_token_loop(self):
        source = self.candidate['recurrence']['reader']
        loop = source.index(split.READER_LOOP)
        self.assertLess(source.index('cache.reserve_back(11);'), loop)
        self.assertLess(source.index('cache.wait_front(11);'), loop)
        self.assertGreater(source.index('cache.pop_front(11);'), loop)
        self.assertEqual(source.count('cache.reserve_back(11);'), 1)
        self.assertTrue(source.endswith('    }\n    cache.pop_front(11);\n}\n'))
        self.assertNotIn('gather_row(', source)
        self.assertNotIn('gather_scalar(', source)
        self.assertIn('if (token == 0)', source)

    def test_no_cache_dma_inside_token_loop(self):
        source = self.candidate['recurrence']['reader']
        loop = source[source.index(split.READER_LOOP):]
        for accessor in ('q_acc', 'k_acc', 'v_acc', 'beta_acc', 'g_acc'):
            self.assertNotIn(f'noc.async_read({accessor}', loop)
        self.assertIn('noc.async_read(s0_acc', loop)

    def test_changed_anchors_fail_closed(self):
        source = self.control['recurrence']['reader']
        with self.assertRaises(ValueError):
            prefetch.transform_reader(source.replace('    auto gather_row =', '    auto renamed ='))
        with self.assertRaises(ValueError):
            prefetch.transform_reader(source + '\n')

    def test_issued_payload_is_not_a_speed_claim(self):
        report = prefetch.audit(ROOT)
        self.assertEqual(report['issued_input_bytes_per_chip']['32'], dict(control=69206016, candidate=2162688))
        self.assertEqual(report['issued_input_bytes_per_chip']['1']['control'], 2162688)


if __name__ == '__main__':
    unittest.main()
