"""Guard the flash workflow's flag set and gates; these are the bits that brick a board."""

from pathlib import Path
import unittest


class FlashWorkflowSafetyTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[2]
                / '.github/workflows/qwen-firmware-flash-19-12.yml')
        self.source = path.read_text()
        # The workflow documents, in a comment, the flags it deliberately omits.
        # Safety checks must therefore read the executable lines, not the whole file.
        self.executable = '\n'.join(line for line in self.source.splitlines()
                                    if not line.strip().startswith('#'))
        self.flash_lines = [line for line in self.executable.splitlines()
                            if 'tt-flash' in line and 'verify' not in line]

    def test_comment_documents_the_omitted_flags_without_using_them(self):
        for flag in ('--force', '--update-boot-images', '--skip-missing-fw'):
            self.assertIn(flag, self.source, 'the rationale comment should name %s' % flag)
            self.assertNotIn(flag, self.executable)

    def test_never_forces_past_the_version_and_board_checks(self):
        self.assertNotIn('--force', self.executable)

    def test_never_rewrites_the_boot_path(self):
        # Bootloader/recovery images are left alone so a power loss mid-update
        # cannot corrupt the board's ability to boot at all.
        self.assertNotIn('--update-boot-images', self.executable)

    def test_never_skips_a_board_whose_firmware_is_absent(self):
        self.assertNotIn('--skip-missing-fw', self.executable)

    def test_never_permits_a_major_downgrade(self):
        self.assertNotIn('--allow-major-downgrades', self.executable)

    def test_does_not_suppress_the_end_of_flash_reset(self):
        self.assertNotIn('--no-reset', self.executable)

    def test_flashes_only_the_pinned_19_12_bundle(self):
        self.assertIn('fw_pack-19.12.0.fwbundle', self.executable)
        self.assertNotIn('--download', self.executable)
        self.assertNotIn('-d ', ' '.join(self.flash_lines))

    def test_refuses_when_a_process_holds_a_card(self):
        self.assertIn('fuser', self.source)
        self.assertIn('refusing to flash', self.source)

    def test_verifies_the_bundle_digest_before_flashing(self):
        digest_at = self.source.index('firmware_staging_manifest.py')
        flash_at = self.source.index('flash "$staging/fw_pack-19.12.0.fwbundle"')
        self.assertLess(digest_at, flash_at,
                        'the bundle must be digest-checked before it is written to a board')

    def test_holds_the_exclusive_card_lock(self):
        self.assertIn('group: qwen-two-p150a-exclusive', self.source)
        self.assertIn('cancel-in-progress: false', self.source)

    def test_requires_a_dedicated_tag_so_it_cannot_fire_by_accident(self):
        self.assertIn("tags: ['experiment/firmware-flash-19-12-v*']", self.source)
        self.assertNotIn('workflow_dispatch', self.source)
        self.assertNotIn('branches:', self.source)

    def test_confirms_the_result_rather_than_assuming_it(self):
        self.assertIn('verify-after.txt', self.source)
        self.assertIn('all_on_19_12', self.source)

    def test_pins_the_runner(self):
        self.assertIn('test "$RUNNER_NAME" = thatch-build-amd64-02-cp-temp', self.source)


if __name__ == '__main__':
    unittest.main()
