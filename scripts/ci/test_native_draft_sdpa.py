from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import os
import sys
import unittest
from unittest.mock import patch

import native_draft_sdpa as kernel


class NativeDraftSdpaTests(unittest.TestCase):
    def test_experimental_reciprocal_is_scoped_and_restored(self):
        with TemporaryDirectory() as root:
            directory, original, hashes = self.fixture(root)
            original['compute_common.hpp'] += (kernel.ORIGINAL_RECIP_INIT + '\n' + kernel.ORIGINAL_RECIP + '\n').encode()
            (directory / 'compute_common.hpp').write_bytes(original['compute_common.hpp'])
            hashes['compute_common.hpp'] = hashlib.sha256(original['compute_common.hpp']).hexdigest()
            with patch.object(kernel, 'SOURCE_HASHES', hashes), patch.dict(os.environ,
                    {'QWEN_T32_PRECISE_RECIP': '1', 'QWEN_SIM_ONLY': '1',
                     'QWEN_HARDWARE_TESTS': '0', 'QWEN_CARDS_ALLOCATED': '0'}):
                with kernel.precise_draft_kernel(root):
                    source = (directory / 'compute_common.hpp').read_text()
                    self.assertIn('recip_tile_first_column<QWEN_DRAFT_EXP_APPROX>', source)
                    self.assertIn('sfpu_reciprocal_init<false>()', source)
                self.assertEqual((directory / 'compute_common.hpp').read_bytes(), original['compute_common.hpp'])

    def test_experimental_reciprocal_rejects_hardware(self):
        with patch.dict(os.environ, {'QWEN_T32_PRECISE_RECIP': '1', 'QWEN_SIM_ONLY': '1',
                                    'QWEN_HARDWARE_TESTS': '1'}), self.assertRaises(ValueError):
            kernel.replacements()

    def fixture(self, root):
        directory = Path(root) / kernel.KERNEL_DIRECTORY
        directory.mkdir(parents=True)
        original = {
            'compute_common.hpp': f'{kernel.ORIGINAL_INIT}\n{kernel.ORIGINAL_EXP}\n'.encode(),
            'sdpa.cpp': b'#include "compute_common.hpp"\n',
        }
        for name, source in original.items():
            (directory / name).write_bytes(source)
        hashes = {name: hashlib.sha256(source).hexdigest() for name, source in original.items()}
        return directory, original, hashes

    def test_precision_is_bounded_to_masked_draft_geometry(self):
        self.assertEqual(kernel.SIGNATURE, {0: 1, 1: 16, 2: 4, 4: 4, 5: 4, 6: 1, 23: 0, 24: 1, 30: 0})
        source = kernel.replacements()['sdpa.cpp'][0][1]
        self.assertIn('EXP_APPROX_MODE || !(', source)
        self.assertIn('#define QWEN_DRAFT_EXP_APPROX true', kernel.PRECISE_INIT)
        self.assertIn('!QWEN_DRAFT_EXP_APPROX', kernel.PRECISE_EXP)
        self.assertIn('scale_fp32 >> 16', kernel.PRECISE_EXP)

    def test_patch_and_restore_on_success_and_failure(self):
        for fail in (False, True):
            with TemporaryDirectory() as root:
                directory, original, hashes = self.fixture(root)
                with patch.object(kernel, 'SOURCE_HASHES', hashes):
                    try:
                        with kernel.precise_draft_kernel(root) as audit:
                            self.assertEqual(audit['original'], hashes)
                            for name, digest in audit['patched'].items():
                                source = (directory / name).read_bytes()
                                self.assertNotEqual(source, original[name])
                                self.assertEqual(hashlib.sha256(source).hexdigest(), digest)
                            if fail:
                                raise RuntimeError('failed probe')
                    except RuntimeError:
                        self.assertTrue(fail)
                for name, source in original.items():
                    self.assertEqual((directory / name).read_bytes(), source)
                self.assertFalse((directory / '.qwen-precise-draft.lock').exists())

    def test_source_drift_fails_before_any_write(self):
        with TemporaryDirectory() as root:
            directory, original, hashes = self.fixture(root)
            hashes['sdpa.cpp'] = '0' * 64
            with patch.object(kernel, 'SOURCE_HASHES', hashes), self.assertRaisesRegex(ValueError, 'Unaudited'):
                with kernel.precise_draft_kernel(root):
                    self.fail('Drift entered device scope')
            for name, source in original.items():
                self.assertEqual((directory / name).read_bytes(), source)

    def test_concurrent_patch_is_rejected_without_touching_owner(self):
        with TemporaryDirectory() as root:
            directory, original, hashes = self.fixture(root)
            lock = directory / '.qwen-precise-draft.lock'
            lock.write_text('other owner')
            with patch.object(kernel, 'SOURCE_HASHES', hashes), self.assertRaises(FileExistsError):
                with kernel.precise_draft_kernel(root):
                    self.fail('Concurrent patch entered device scope')
            self.assertEqual(lock.read_text(), 'other owner')
            for name, source in original.items():
                self.assertEqual((directory / name).read_bytes(), source)

    def test_child_abort_does_not_leave_kernel_patch_installed(self):
        with TemporaryDirectory() as root:
            directory, original, hashes = self.fixture(root)
            child = Path(root) / 'abort.py'
            child.write_text('import os\nos._exit(19)\n')
            environment = dict(os.environ, TT_METAL_HOME=root)
            environment.pop('QWEN_PRECISE_DRAFT_ACTIVE', None)
            with patch.object(kernel, 'SOURCE_HASHES', hashes), patch.dict(os.environ, environment, clear=True), \
                    patch.object(sys, 'argv', [str(child)]), self.assertRaises(SystemExit) as result:
                kernel.run_precise_probe(child)
            self.assertEqual(result.exception.code, 19)
            self.assertFalse((directory / '.qwen-precise-draft.lock').exists())
            for name, source in original.items():
                self.assertEqual((directory / name).read_bytes(), source)
