import copy
from pathlib import Path
import unittest

from dram_eligibility_inventory import summarize


class EligibilityInventoryTests(unittest.TestCase):
    def test_missing_masks_are_not_zero_or_supported(self):
        result = summarize({'devices': [{'firmware_version': '19.8.1'}]})
        self.assertEqual(result['dram_harvesting_status'], 'not_reported')
        self.assertEqual(result['explicit_dram_harvest_fields'], [])
        self.assertIsNone(result['prefetch_supported'])

    def test_nested_masks_preserve_values_and_board_paths(self):
        snapshot = {'devices': [{'serial': 'first', 'dram_harvesting_mask': 0},
                                {'serial': 'second', 'dram_harvesting_mask': '0x2'}]}
        before = copy.deepcopy(snapshot)
        result = summarize(snapshot)
        self.assertEqual(snapshot, before)
        self.assertEqual(result['explicit_dram_harvest_fields'], [
            dict(path=['devices', 0, 'dram_harvesting_mask'], value=0),
            dict(path=['devices', 1, 'dram_harvesting_mask'], value='0x2')])
        self.assertIsNone(result['prefetch_supported'])
        self.assertFalse(result['performance_qualified'])

    def test_empty_or_unstructured_snapshot_rejected(self):
        for value in ({}, [], None, 'text', False):
            with self.assertRaises(ValueError):
                summarize(value)

    def test_workflow_only_requests_snapshot_not_reset_or_flash(self):
        path = Path(__file__).resolve().parents[2] / '.github/workflows/qwen-dram-eligibility-inventory.yml'
        source = path.read_text()
        self.assertIn('"$smi" -f ', source)
        self.assertNotIn('"$smi" -r', source)
        self.assertNotIn('"$smi" --reset', source)
        self.assertNotIn('tt-flash', source)
        self.assertNotIn('TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1', source)
        self.assertIn('group: qwen-two-p150a-exclusive', source)
