import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from firmware_staging_manifest import (ARTEFACTS, FIRMWARE_VERSION, ROLLBACK_BUNDLE,
                                       TT_FLASH_VERSION, summarize, verify)


class ManifestPinningTests(unittest.TestCase):
    def test_every_artefact_pins_a_full_sha256_and_size(self):
        for name, entry in ARTEFACTS.items():
            self.assertRegex(entry['sha256'], r'^[0-9a-f]{64}$', name)
            self.assertGreater(entry['size'], 0, name)
            self.assertTrue(entry['url'].startswith('https://github.com/tenstorrent/'), name)

    def test_targets_the_version_metal_requires(self):
        self.assertEqual(FIRMWARE_VERSION, '19.12.0')
        for name in ('fw_pack-19.12.0.fwbundle', 'fw-pack-v19.12.0-recovery.tar.gz'):
            self.assertIn(name, ARTEFACTS)

    def test_tt_flash_clears_the_three_six_floor_and_avoids_the_interlock_release(self):
        major, minor, _ = (int(part) for part in TT_FLASH_VERSION.split('.'))
        self.assertEqual(major, 3, 'v4 adds the 19.15 board-variable interlock')
        self.assertGreaterEqual(minor, 6, '19.5.0 requires tt-flash >= 3.6.0 on Blackhole')

    def test_a_recovery_image_is_staged(self):
        roles = [entry['role'] for entry in ARTEFACTS.values()]
        self.assertTrue(any('recovery' in role for role in roles))

    def test_rollback_bundle_points_at_the_existing_host_image(self):
        self.assertEqual(ROLLBACK_BUNDLE, '/home/thatch/fw_pack-19.8.1.fwbundle')


class VerificationTests(unittest.TestCase):
    def test_missing_file_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            results = verify(directory, ['p150a.fwbundle'])
            entry = results['p150a.fwbundle']
            self.assertFalse(entry['present'])
            self.assertFalse(entry['verified'])
            self.assertIsNone(entry['actual_sha256'])

    def test_wrong_content_fails_even_at_the_right_size(self):
        name = 'p150a.fwbundle'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_bytes(b'\0' * ARTEFACTS[name]['size'])
            entry = verify(directory, [name])[name]
            self.assertTrue(entry['size_ok'])
            self.assertFalse(entry['sha256_ok'])
            self.assertFalse(entry['verified'])

    def test_matching_content_verifies(self):
        name = 'p150a.fwbundle'
        payload = b'staged-bytes'
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / name).write_bytes(payload)
            ARTEFACTS[name] = dict(ARTEFACTS[name], size=len(payload),
                                   sha256=hashlib.sha256(payload).hexdigest())
            try:
                entry = verify(directory, [name])[name]
                self.assertTrue(entry['verified'])
            finally:
                ARTEFACTS[name] = dict(
                    ARTEFACTS[name], size=383586,
                    sha256='244c74fe01857dc6ea9d8be064b85f61c1ab9c63b7fbbc929ba97eeb33032faa')

    def test_summary_never_claims_a_flash_happened(self):
        with tempfile.TemporaryDirectory() as directory:
            report = summarize(verify(directory, ['p150a.fwbundle']))
        self.assertFalse(report['flash_performed'])
        self.assertFalse(report['firmware_modified'])
        self.assertFalse(report['devices_reset'])
        self.assertFalse(report['all_verified'])
        json.dumps(report)


class WorkflowSafetyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / '.github/workflows/qwen-firmware-staging.yml'
        self.source = path.read_text()

    def test_staging_workflow_cannot_flash_reset_or_reboot(self):
        for forbidden in ('--fw-tar', 'tt-flash flash', '--skip-missing-fw', '--force',
                          '-r /dev/tenstorrent', '--reset', 'tensix-reset', 'reboot',
                          'ipmitool', 'TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1'):
            self.assertNotIn(forbidden, self.source)

    def test_flasher_is_only_ever_asked_for_its_version(self):
        self.assertIn('--version', self.source)
        self.assertNotIn('tt-flash.run /', self.source)

    def test_keeps_exclusive_card_concurrency(self):
        self.assertIn('group: qwen-two-p150a-exclusive', self.source)
        self.assertIn('cancel-in-progress: false', self.source)

    def test_staging_directory_is_separate_from_path(self):
        self.assertIn('/home/thatch/firmware-staging-19.12.0', self.source)
        self.assertNotIn('/usr/local/bin', self.source)
        self.assertNotIn('.local/bin', self.source)


if __name__ == '__main__':
    unittest.main()
