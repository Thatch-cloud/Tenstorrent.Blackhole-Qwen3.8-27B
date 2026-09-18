from pathlib import Path
import unittest

from dram_harvest_probe import classify, collect


class HarvestClassificationTests(unittest.TestCase):
    def test_absent_masks_are_unavailable_not_zero(self):
        for value in (None, {}, False):
            self.assertEqual(classify(value), 'masks_unavailable')

    def test_none_mask_is_unknown_not_clear(self):
        masks = {'0': dict(dram_harvesting_mask=0), '1': dict(dram_harvesting_mask=None)}
        self.assertEqual(classify(masks), 'unknown_mask_reported')

    def test_missing_key_is_unknown(self):
        self.assertEqual(classify({'0': dict(tensix_harvesting_mask=0)}), 'unknown_mask_reported')

    def test_any_harvested_chip_blocks_multi_device_prefetch(self):
        masks = {'0': dict(dram_harvesting_mask=0), '2': dict(dram_harvesting_mask=2)}
        self.assertEqual(classify(masks), 'harvested_dram_blocks_multi_device_prefetch')

    def test_all_zero_masks_report_no_harvesting(self):
        masks = {'0': dict(dram_harvesting_mask=0), '2': dict(dram_harvesting_mask=0)}
        self.assertEqual(classify(masks), 'no_dram_harvesting_observed')

    def test_booleans_are_not_accepted_as_masks(self):
        self.assertEqual(classify({'0': dict(dram_harvesting_mask=True)}), 'unknown_mask_reported')


class ProbeContractTests(unittest.TestCase):
    def test_collect_without_runtime_reports_unavailable_and_claims_nothing(self):
        result = collect(['/nonexistent-probe-root'])
        self.assertEqual(result['dram_harvest_verdict'], 'masks_unavailable')
        self.assertIsNone(result['chip_masks'])
        self.assertIsNone(result['prefetch_supported'])
        self.assertFalse(result['firmware_modified'])
        self.assertFalse(result['devices_reset'])
        self.assertFalse(result['performance_qualified'])

    def test_device_open_is_opt_in(self):
        result = collect(['/nonexistent-probe-root'])
        crosscheck = [probe for probe in result['probes']
                      if probe['probe'] == 'ttnn_dram_channel_crosscheck']
        self.assertEqual(len(crosscheck), 1)
        self.assertFalse(crosscheck[0]['ok'])
        self.assertIn('not authorised', crosscheck[0]['error'])

    def test_every_probe_failure_is_recorded_rather_than_raised(self):
        result = collect(['/nonexistent-probe-root'])
        names = [probe['probe'] for probe in result['probes']]
        self.assertIn('umd_soc_descriptor_masks', names)
        self.assertIn('cluster_descriptor_files', names)


class WorkflowSafetyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / '.github/workflows/qwen-dram-harvest-probe.yml'
        self.source = path.read_text()

    def test_workflow_never_flashes_resets_or_forces_the_override(self):
        # tt-flash --version is inventory and stays allowed; anything that writes does not.
        for forbidden in ('--fw-tar', 'tt-flash flash', '--skip-missing-fw', '--force',
                          '"$smi" -r', '--reset', 'tensix-reset',
                          'TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1',
                          '--allow-device-open', 'reboot', 'ipmitool'):
            self.assertNotIn(forbidden, self.source)

    def test_workflow_only_reads_tt_flash_version(self):
        for line in self.source.splitlines():
            if 'tt-flash' not in line:
                continue
            stripped = line.strip()
            if stripped.startswith(('printf', 'echo', '#')):
                continue
            self.assertTrue(
                any(token in stripped for token in ('--version', 'command -v', 'candidate', 'grep')),
                'unexpected tt-flash invocation: %s' % stripped)

    def test_workflow_keeps_exclusive_card_concurrency_and_snapshot_only_smi(self):
        self.assertIn('group: qwen-two-p150a-exclusive', self.source)
        self.assertIn('cancel-in-progress: false', self.source)
        self.assertIn('"$smi" -f ', self.source)

    def test_workflow_requires_clear_device_ownership_before_probing(self):
        self.assertIn('Device ownership not clear', self.source)
        self.assertIn('fuser', self.source)

    def test_workflow_does_not_assume_a_fixed_card_count(self):
        self.assertNotIn("boards = {'blackhole-", self.source)
        self.assertIn('device_nodes', self.source)


if __name__ == '__main__':
    unittest.main()
