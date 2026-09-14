import unittest

from dspark_splitk_sum_audit import face_offset, transform


class SumAuditTests(unittest.TestCase):
    def test_face_offsets_cover_one_tile(self):
        offsets = {face_offset(row, column) for row in range(32) for column in range(32)}
        self.assertEqual(offsets, set(range(1024)))
        self.assertEqual(face_offset(16, 0), 512)
        self.assertEqual(face_offset(0, 16), 256)

    def test_audit_is_bounded_and_does_not_consume_buffers(self):
        result = transform('                /* OUT_IM = QK @ V_CHUNK */')
        self.assertIn('k_chunk == k_chunk_start', result)
        self.assertIn('const volatile float*', result)
        self.assertNotIn('pop_front', result)
        self.assertNotIn('push_back', result)
        self.assertNotIn('reserve_back', result)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, 'boundary'):
            transform('')


if __name__ == '__main__':
    unittest.main()
