from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import dspark_combined_probe_build as candidate


class CombinedProbeBuildTests(unittest.TestCase):
    def test_validated_binaries_and_original_packer_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernel.cpp',
                'tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h',
                'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h',
                'tt_metal/hw/inc/api/compute/reduce.h', 'packer.h',
                'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            evidence = dict(binaries={name: candidate.digest(root / name) for name in names[-2:]})
            audit = Mock()
            options = dict(packer='packer.h', expected_packer=candidate.digest(root / 'packer.h'),
                audit_kernel=audit, precise_native=True)
            with patch.object(candidate, 'validate_build', return_value=evidence) as validate:
                result = candidate.fingerprints(root, **options)
                self.assertEqual(set(result), set(names))
                validate.assert_called_once_with(root,
                    '/experiment/results/dspark-64k-hardware-build.json', Path(candidate.__file__).parent)
                audit.assert_called_once_with(root)
                (root / names[-1]).write_text('changed binary')
                with self.assertRaisesRegex(ValueError, 'binary'):
                    candidate.fingerprints(root, **options)
            with patch.object(candidate, 'validate_build', side_effect=ValueError('factory mismatch')):
                with self.assertRaisesRegex(ValueError, 'factory mismatch'):
                    candidate.fingerprints(root, **options)

    def test_simulator_packer_rejected_before_build_access(self):
        with patch.object(candidate, 'validate_build') as validate:
            with self.assertRaisesRegex(ValueError, 'hardware'):
                candidate.fingerprints('/unused', packer='unused', expected_packer='unused',
                    audit_kernel=Mock(), packer_compat=True, precise_native=True)
            validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
