import os
import unittest
import unittest.mock


class FlagTests(unittest.TestCase):
    def test_default_is_off(self):
        from dflash_traced_publish import traced_publish_enabled

        self.assertFalse(traced_publish_enabled({}))

    def test_only_0_or_1_accepted(self):
        from dflash_traced_publish import traced_publish_enabled

        self.assertFalse(traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': '0'}))
        self.assertTrue(traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': '1'}))
        for bad in ('true', 'yes', '2', ' 1', ''):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': bad})

    def test_reads_the_real_os_environ_by_default(self):
        from dflash_traced_publish import traced_publish_enabled

        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_TRACED_PUBLISH': '1'}):
            self.assertTrue(traced_publish_enabled())
        self.assertNotIn('QWEN_FAST_TRACED_PUBLISH', os.environ)
        self.assertFalse(traced_publish_enabled())


class FakeDrafter:
    """A minimal DFlashDevice-shaped object: prepare_publication as the class
    defines it (a real method taking both options), so install_publish_options'
    rebind can be checked without any device machinery."""

    def __init__(self):
        self.calls = []

    def prepare_publication(self, features, prefix, *, position, merge_release=False, fused_steady_state=False):
        self.calls.append((features, prefix, position, merge_release, fused_steady_state))
        return 'publication'


class InstallPublishOptionsTests(unittest.TestCase):
    def test_the_rebound_method_forces_both_options_together(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        result = drafter.prepare_publication('features', 5, position=100)
        self.assertEqual(result, 'publication')
        self.assertEqual(drafter.calls, [('features', 5, 100, True, True)])
        restore()

    def test_either_option_can_be_forced_independently(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=False, fused_steady_state=True)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, False, True))
        restore()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=False)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, True, False))
        restore()

    def test_restore_removes_the_override_and_the_class_method_runs_again(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        self.assertIn('prepare_publication', drafter.__dict__)
        restore()
        self.assertNotIn('prepare_publication', drafter.__dict__)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, False, False), 'both default off after restore')

    def test_double_install_is_refused(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        with self.assertRaises(ValueError):
            install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        restore()

    def test_non_bool_options_are_refused(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        for kwargs in (dict(merge_release=1, fused_steady_state=True), dict(merge_release=True, fused_steady_state=1)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    install_publish_options(drafter, **kwargs)

    def test_works_on_any_object_not_just_a_real_drafter(self):
        """The shim only ever sets and later deletes an instance attribute - no
        dependency on DFlashDevice's own class machinery."""
        from dflash_traced_publish import install_publish_options

        class Bare:
            def prepare_publication(self, features, prefix, *, position, merge_release=False, fused_steady_state=False):
                return (features, prefix, position, merge_release, fused_steady_state)

        drafter = Bare()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, True, True))
        restore()
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, False, False))


if __name__ == '__main__':
    unittest.main()
