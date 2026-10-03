import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dspark_attention import validate_mask
from dspark_hardware_gate import digest
from dspark_rope_tables import DSparkRotary
from test_dspark_intake import configuration


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('dspark_target_hardware', ROOT / 'dspark-target-hardware.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class DSparkTargetIntegrationTests(unittest.TestCase):
    def config(self):
        return dict(configuration(), max_position_embeddings=262144)

    def test_metadata_retains_full_32_row_history_and_all_seven_queries(self):
        values = PROBE.metadata(self.config())
        self.assertEqual(set(values), {'q_cos', 'q_sin', 'k_cos', 'k_sin', 'mask', 'live'})
        validate_mask(values['mask'], 32, key_multiple=64)
        self.assertEqual(tuple(values['live'].shape), (1, 1, 32, 1))
        self.assertEqual(values['live'].dtype, torch.float32)
        self.assertEqual(int(values['live'].sum()), 7)
        expected = DSparkRotary(self.config()).block_tables(0, 32)
        for name, pair in expected.items():
            for kind, original, fill in zip(('cos', 'sin'), pair, (1., 0.), strict=True):
                actual = values[name + '_' + kind]
                self.assertTrue(torch.equal(actual[:, :, :original.shape[2]], original))
                self.assertTrue(torch.all(actual[:, :, original.shape[2]:] == fill))

    def test_row_zero_is_a_proposal_not_a_discarded_anchor_output(self):
        golden = list(range(10, 18))
        self.assertEqual(PROBE.accepted_prefix(golden[:7], golden), 7)
        self.assertEqual(PROBE.accepted_prefix([99, *golden[1:7]], golden), 0)
        self.assertEqual(PROBE.accepted_prefix([*golden[:3], 99, *golden[4:7]], golden), 3)

    def test_nonprefix_matches_do_not_inflate_acceptance(self):
        self.assertEqual(PROBE.accepted_prefix([0, 9, 2, 3, 4, 5, 6], list(range(8))), 1)

    def test_complete_global_integer_ids_required(self):
        for proposals, golden in ((list(range(6)), list(range(8))), (list(range(7)), list(range(7))),
                ([True, *range(1, 7)], list(range(8))), ([248320, *range(1, 7)], list(range(8))),
                ([-1, *range(1, 7)], list(range(8)))):
            with self.subTest(proposals=proposals, golden=golden), self.assertRaises(ValueError):
                PROBE.accepted_prefix(proposals, golden)

    def fixture(self, directory):
        root = Path(directory)
        weights = root / '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
        weights.mkdir()
        (weights / 'config.json').write_text('{}')
        (weights / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight':'model.safetensors'})))
        (weights / 'model.safetensors').write_bytes(b'host fixture, never executed')
        config = root / 'dspark-config.json'
        config.write_text(json.dumps(self.config()))
        return root, weights, config

    def check(self, root, weights, config):
        with patch.object(PROBE, 'TARGET_SOURCES', {}), patch.object(PROBE, 'FILES', {'config.json':(0, digest(config))}):
            return PROBE.preflight(root, weights, config)

    def test_current_component_proofs_and_source_closure_are_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.check(*self.fixture(directory))
        self.assertEqual(result['prerequisites'], PROBE.PREREQUISITES)
        for name in ('dspark_target.py', 'dspark_vocabulary.py', 'dspark_markov_device.py', 'dspark_pipeline.py'):
            self.assertEqual(result['sources'][name], digest(ROOT / name))
        self.assertNotIn('../../optimisation/sim/run-dispatch-probe.sh', result['sources'])
        self.assertTrue(any('/embedding/' in name for name in result['native_reference']))
        self.assertFalse(result['simulator_preflight']['retained_cpu_numerical_gate_passed'])

    def test_changed_proposal_source_fails_before_device_open(self):
        def changed(path):
            return '0' * 64 if Path(path).name == 'dspark_target.py' else digest(path)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(PROBE, 'digest', side_effect=changed), self.assertRaisesRegex(ValueError, 'Qualified proposal source changed'):
                self.check(*fixture)

    def test_missing_target_weight_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root, weights, config = self.fixture(directory)
            (weights / 'model.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'Complete pinned local target'):
                self.check(root, weights, config)

    def test_snapshot_mismatch_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root, weights, config = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'Pinned target snapshot'):
                self.check(root, weights.parent, config)


if __name__ == '__main__':
    unittest.main()
