from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mtp_device_step import MTPDeviceStep


class MTPDeviceStepTests(unittest.TestCase):
    def test_native_row_selection_reaches_full_head_only(self):
        step = MTPDeviceStep.__new__(MTPDeviceStep)
        hidden, logits, identifiers = object(), object(), object()
        step.operations = SimpleNamespace(linear=Mock(return_value=logits))
        step.mtp = SimpleNamespace(forward=Mock(return_value=hidden))
        step.model = SimpleNamespace(lm_head_weight=object())
        step.inputs = {name: object() for name in ('embedding', 'hidden', 'positions', 'cosine', 'sine')}
        step.pages, step.sampler, step.shortlist = object(), object(), None
        for enabled in (False, True):
            step.native_sampling_rows, step.owned = enabled, []
            with patch('mtp_device_step.sample_rows', return_value=identifiers) as sample:
                self.assertEqual(step.execute(True), (hidden, identifiers))
                sample.assert_called_once_with(step.sampler, logits, 1, step.operations, native_rows=enabled)
                self.assertEqual(step.owned, [hidden, logits, identifiers])
            step.owned = []
            with patch('mtp_device_step.sample_rows') as sample:
                self.assertEqual(step.execute(False), (hidden, None))
                sample.assert_not_called()
                self.assertEqual(step.owned, [hidden])

    def test_invalid_sampling_selection_fails_before_device_allocation(self):
        for enabled, shortlist in ((1, None), (True, object())):
            with self.assertRaises(ValueError):
                MTPDeviceStep(None, None, None, None, None, None,
                    native_sampling_rows=enabled, shortlist=shortlist)
