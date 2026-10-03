import unittest
from unittest.mock import patch

from dspark_device import DSparkDevice
from dspark_prepared_proposal import TracedDSparkDevice
from dspark_native_cached_layer import execute


class NativeSelectionTests(unittest.TestCase):
    def test_native_selection_is_explicit_and_default_stays_unchanged(self):
        with patch.object(DSparkDevice, '__init__') as initialize:
            default = TracedDSparkDevice()
            native = TracedDSparkDevice(native_attention=True)
        self.assertFalse(hasattr(default, 'proposal_layer'))
        self.assertIs(native.proposal_layer, execute)
        self.assertEqual(initialize.call_count, 2)

    def test_non_boolean_selection_fails_before_allocation(self):
        with patch.object(DSparkDevice, '__init__') as initialize:
            with self.assertRaises(ValueError):
                TracedDSparkDevice(native_attention='1')
        initialize.assert_not_called()
