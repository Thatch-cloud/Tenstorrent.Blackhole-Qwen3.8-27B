import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sampling_link_policy as policy


class Collective:
    def get_num_links(self, cluster_axis=None):
        return 2


class SamplingLinkPolicyTests(unittest.TestCase):
    def sampler(self):
        return SimpleNamespace(mesh_device=SimpleNamespace(shape=(1, 2)),
            tt_ccl=Collective(), num_argmax_gather_links=1)

    def test_requested_links_and_restoration(self):
        sampler = self.sampler()
        for links in (1, 2, 4):
            with policy.sampler_links(sampler, links):
                self.assertEqual(sampler.num_argmax_gather_links, links)
                for axis in (None, 0, 1):
                    self.assertEqual(sampler.tt_ccl.get_num_links(axis), 4)
                for axis in (True, 2, '1'):
                    with self.assertRaises(ValueError):
                        sampler.tt_ccl.get_num_links(axis)
            self.assertEqual(sampler.num_argmax_gather_links, 1)
            self.assertEqual(sampler.tt_ccl.get_num_links(), 2)
            self.assertNotIn('get_num_links', vars(sampler.tt_ccl))

    def test_exception_restores_existing_override(self):
        sampler = self.sampler()
        original = lambda axis=None: 1
        sampler.tt_ccl.get_num_links = original
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with policy.sampler_links(sampler, 4):
                raise RuntimeError('failure')
        self.assertIs(sampler.tt_ccl.get_num_links, original)
        self.assertEqual(sampler.num_argmax_gather_links, 1)

    def test_invalid_configuration_never_mutates(self):
        sampler = self.sampler()
        for links in (True, 0, 3, 8, '4'):
            with self.assertRaises(ValueError), policy.sampler_links(sampler, links):
                self.fail('Entered invalid configuration')
        sampler.mesh_device.shape = (2, 1)
        with self.assertRaises(ValueError), policy.sampler_links(sampler, 4):
            self.fail('Entered wrong mesh')
        self.assertEqual(sampler.num_argmax_gather_links, 1)
        self.assertNotIn('get_num_links', vars(sampler.tt_ccl))

    def test_audit_requires_explicit_environment_and_exact_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for name in policy.SOURCES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(name.encode())
                sources[name] = hashlib.sha256(name.encode()).hexdigest()
            environment = dict(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1',
                QWEN_FABRIC_LINK_PROBE='1', TT_MESH_GRAPH_DESC_PATH=str(root / policy.DESCRIPTOR))
            with patch.object(policy, 'SOURCES', sources):
                self.assertEqual(policy.audit(root, environment), sources)
                for key in environment:
                    with self.assertRaises(ValueError):
                        policy.audit(root, {**environment, key: '0'})
                for key in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'):
                    with self.assertRaises(ValueError):
                        policy.audit(root, {**environment, key: '1'})
                (root / policy.DESCRIPTOR).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    policy.audit(root, environment)


if __name__ == '__main__':
    unittest.main()
