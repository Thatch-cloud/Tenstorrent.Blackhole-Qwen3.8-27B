from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import projection_link_policy as policy


class ProjectionLinkPolicyTests(unittest.TestCase):
    def test_simulator_never_claims_physical_four_links(self):
        self.assertEqual(policy.validate({'TT_METAL_SIMULATOR': 'sim'})['requested_links'], 1)
        for requested in ('2', '4', '0', 'four'):
            with self.assertRaises(ValueError):
                policy.validate(dict(TT_METAL_SIMULATOR='sim', QWEN_PROJECTION_LINKS=requested))

    def test_hardware_requires_allocation_and_pinned_pair_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = root / policy.DESCRIPTOR
            descriptor.parent.mkdir(parents=True)
            descriptor.write_bytes(b'four-channel-fixture')
            environment = dict(QWEN_PROJECTION_LINKS='4', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1',
                               TT_METAL_HOME=str(root), TT_MESH_GRAPH_DESC_PATH=str(descriptor))
            with patch.dict(policy.SOURCES, {policy.DESCRIPTOR: policy.hashlib.sha256(descriptor.read_bytes()).hexdigest()}):
                self.assertEqual(policy.validate(environment)['requested_links'], 4)
                for missing in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'TT_MESH_GRAPH_DESC_PATH'):
                    with self.assertRaises(ValueError):
                        policy.validate({key: value for key, value in environment.items() if key != missing})
            with self.assertRaises(ValueError):
                policy.validate(environment)
