import os
import unittest
import unittest.mock
from types import SimpleNamespace


class FlagTests(unittest.TestCase):
    def test_default_is_off(self):
        from dflash_pipelined_publish import pipelined_publish_enabled

        self.assertFalse(pipelined_publish_enabled({}))

    def test_only_0_or_1_accepted(self):
        from dflash_pipelined_publish import pipelined_publish_enabled

        self.assertFalse(pipelined_publish_enabled({'QWEN_FAST_PIPELINED_PUBLISH': '0'}))
        self.assertTrue(pipelined_publish_enabled({'QWEN_FAST_PIPELINED_PUBLISH': '1'}))
        for bad in ('true', 'yes', '2', ' 1', ''):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    pipelined_publish_enabled({'QWEN_FAST_PIPELINED_PUBLISH': bad})

    def test_reads_the_real_os_environ_by_default(self):
        from dflash_pipelined_publish import pipelined_publish_enabled

        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_PIPELINED_PUBLISH': '1'}):
            self.assertTrue(pipelined_publish_enabled())
        self.assertNotIn('QWEN_FAST_PIPELINED_PUBLISH', os.environ)
        self.assertFalse(pipelined_publish_enabled())


class FakeDrafter:
    """A minimal DFlashDevice-shaped object: prepare_publication as the class defines
    it (a real method taking merge_release), so install_merge_release's rebind can be
    checked without any device machinery."""

    def __init__(self):
        self.calls = []

    def prepare_publication(self, features, prefix, *, position, merge_release=False):
        self.calls.append((features, prefix, position, merge_release))
        return 'publication'


class InstallMergeReleaseTests(unittest.TestCase):
    def test_the_rebound_method_forces_merge_release_true(self):
        from dflash_pipelined_publish import install_merge_release

        drafter = FakeDrafter()
        restore = install_merge_release(drafter)
        result = drafter.prepare_publication('features', 5, position=100)
        self.assertEqual(result, 'publication')
        self.assertEqual(drafter.calls, [('features', 5, 100, True)])
        restore()

    def test_restore_removes_the_override_and_the_class_method_runs_again(self):
        from dflash_pipelined_publish import install_merge_release

        drafter = FakeDrafter()
        restore = install_merge_release(drafter)
        self.assertIn('prepare_publication', drafter.__dict__)
        restore()
        self.assertNotIn('prepare_publication', drafter.__dict__)
        drafter.prepare_publication('features', 5, position=100)
        self.assertEqual(drafter.calls[-1], ('features', 5, 100, False), 'default merge_release after restore')

    def test_double_install_is_refused(self):
        from dflash_pipelined_publish import install_merge_release

        drafter = FakeDrafter()
        restore = install_merge_release(drafter)
        with self.assertRaises(ValueError):
            install_merge_release(drafter)
        restore()

    def test_works_on_any_object_not_just_a_real_drafter(self):
        """The shim only ever sets and later deletes an instance attribute - no
        dependency on DFlashDevice's own class machinery."""
        from dflash_pipelined_publish import install_merge_release

        class Bare:
            def prepare_publication(self, features, prefix, *, position, merge_release=False):
                return (features, prefix, position, merge_release)

        drafter = Bare()
        restore = install_merge_release(drafter)
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, True))
        restore()
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, False))


if __name__ == '__main__':
    unittest.main()
