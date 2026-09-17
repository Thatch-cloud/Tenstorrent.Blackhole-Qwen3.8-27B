import unittest

from target_t16_64k_geometry import geometry, geometry_scope, validate_ticket


class GeometryTests(unittest.TestCase):
    def test_explicit_backend_bounds(self):
        for hardware in (True, False):
            shape = geometry(hardware=hardware)
            for start in shape['starts']:
                validate_ticket(start, 16, shape['capacity'], hardware=hardware)
            for start, rows, capacity in ((shape['context'] - 1, 16, shape['capacity']),
                    (shape['capacity'] - 15, 16, shape['capacity']),
                    (shape['context'], 32, shape['capacity']),
                    (shape['context'], 16, shape['capacity'] + 256)):
                with self.assertRaises(ValueError):
                    validate_ticket(start, rows, capacity, hardware=hardware)

    def test_original_policy_restored(self):
        import attention_mask_replay
        import attention_replay
        original = attention_mask_replay.validate_ticket
        alias = attention_replay.validate_ticket
        with geometry_scope(hardware=True):
            attention_replay.validate_ticket(65536, 16, 65792)
            self.assertIs(attention_mask_replay.validate_ticket, attention_replay.validate_ticket)
        self.assertIs(attention_mask_replay.validate_ticket, original)
        self.assertIs(attention_replay.validate_ticket, alias)
        with self.assertRaises(ValueError):
            original(65536, 16, 65792)


if __name__ == '__main__':
    unittest.main()
