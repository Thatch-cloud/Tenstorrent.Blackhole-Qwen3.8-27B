from types import SimpleNamespace
import unittest

from captured_publication_profile import profile_publications


class PublicationProfileTests(unittest.TestCase):
    def arm_class(self):
        class Arm:
            def __init__(self):
                self.projection = SimpleNamespace(project=lambda features: features)
                self.history = SimpleNamespace(prepare_projected=lambda value: value)
            def publication(self, features, prefix, *, position):
                return self.history.prepare_projected(self.projection.project(features))
        return Arm

    def test_bounds_restoration_and_preserved_result(self):
        arm_class, records, emitted = self.arm_class(), [], []
        original = arm_class.publication
        arms = [arm_class() for index in range(3)]
        originals = [(arm.projection.project, arm.history.prepare_projected) for arm in arms]
        ticks = iter(range(100))
        with profile_publications(arm_class, records, emitted.append, clock=lambda: next(ticks)):
            for arm in arms:
                for position in range(4):
                    self.assertEqual(arm.publication('features', 2, position=position), 'features')
        self.assertIs(arm_class.publication, original)
        self.assertEqual(len(records), 6)
        self.assertEqual(emitted, records)
        for arm, before in zip(arms, originals):
            self.assertIs(arm.projection.project, before[0])
            self.assertIs(arm.history.prepare_projected, before[1])
        for record in records:
            self.assertTrue(record['passed'])
            self.assertEqual([stage['stage'] for stage in record['stages']], ['projection', 'bank_assembly'])
            self.assertEqual(record['total_ms'], 5000)
            self.assertEqual(record['other_ms'], 3000)
            self.assertFalse(record['added_device_fences'])

    def test_failure_is_retained_and_callbacks_restore(self):
        arm_class, records = self.arm_class(), []
        arm = arm_class()
        def failed(value):
            raise RuntimeError('projection failed')
        arm.projection.project = failed
        with self.assertRaisesRegex(RuntimeError, 'projection failed'):
            with profile_publications(arm_class, records, lambda record: None):
                arm.publication('features', 2, position=65536)
        self.assertIs(arm.projection.project, failed)
        self.assertFalse(records[0]['passed'])
        self.assertEqual(len(records[0]['stages']), 1)


if __name__ == '__main__':
    unittest.main()
