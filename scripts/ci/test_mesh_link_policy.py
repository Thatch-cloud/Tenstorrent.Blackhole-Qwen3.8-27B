"""mesh_link_policy: the pinned pair policies generalised to the four-card (1, 4) ring without touching their bytes,
and ccl_link_patch's discovery counting a ring's edges (a two-device mesh counted exactly as before)."""

import hashlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccl_link_patch  # noqa: E402
import mesh_link_policy as policy  # noqa: E402
import projection_link_policy  # noqa: E402
import sampling_link_policy  # noqa: E402
import tp4_mesh  # noqa: E402

HERE = Path(__file__).resolve().parent

# The evidence pins that forbid editing the pair modules (this module's reason to exist): if one of these moves,
# the pair policy was edited after all and the recorded evidence that hashes it no longer holds.
PINNED = {
    'sampling_link_policy.py': 'f9e8d30c8347fc99bed94957dc6aa9b273fcba6f09881fbccefb0395204ec668',
    'projection_link_policy.py': '659c90f163f15b59ecd7a45d11637243d01cafbda81c5c598fda8bf9ace1774d',
    'model_link_policy.py': 'fb75ac4b5ae933dcc6b29ea120c42383e38ee7b6cc8903d6b585f0fc7d2127ae',
}


class Collective(object):
    def __init__(self, links=2):
        self._links = links

    def get_num_links(self, cluster_axis=None):
        return self._links


def sampler(shape):
    return SimpleNamespace(mesh_device=SimpleNamespace(shape=shape), tt_ccl=Collective(), num_argmax_gather_links=1)


def model(shape, layers=64):
    collective = Collective()
    owners = [SimpleNamespace(tt_ccl=collective, attention=SimpleNamespace(tt_ccl=collective),
                              feed_forward=SimpleNamespace(tt_ccl=collective)) for _ in range(layers)]
    return SimpleNamespace(mesh_device=SimpleNamespace(shape=shape), tt_ccl=collective, layers=owners)


class PinnedPairTests(unittest.TestCase):
    def test_the_pair_policies_keep_their_evidence_bytes(self):
        for name, digest in PINNED.items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)

    def test_the_pair_entry_is_sampling_link_policys_own(self):
        pair = policy.entry((1, 2))
        self.assertEqual(pair['descriptor'], sampling_link_policy.DESCRIPTOR)
        self.assertEqual(pair['sha256'], sampling_link_policy.SOURCES[sampling_link_policy.DESCRIPTOR])
        self.assertEqual((pair['links'], pair['capacity'], pair['mesh_device']), ((1, 2, 4), 4, 'P300'))

    def test_the_ring_entry_is_tp4_meshs(self):
        ring = policy.entry((1, 4))
        self.assertEqual((ring['descriptor'], ring['links'], ring['capacity'], ring['mesh_device']),
                         (tp4_mesh.DESCRIPTOR_PATH, (1, 2), 2, 'P150x4'))
        self.assertEqual(policy.expected_sha256((1, 4)),
                         hashlib.sha256((HERE / tp4_mesh.DESCRIPTOR_NAME).read_bytes()).hexdigest())
        for shape in ((2, 2), (1, 3), (1, 8), (1, 1)):
            with self.assertRaises(ValueError):
                policy.entry(shape)


class SamplerTests(unittest.TestCase):
    def test_the_pair_delegates_to_the_pinned_policy(self):
        pair = sampler((1, 2))
        with policy.sampler_links(pair, 4):
            self.assertEqual(pair.num_argmax_gather_links, 4)
            self.assertEqual(pair.tt_ccl.get_num_links(), 4, "the pinned policy's capacity")
        self.assertNotIn('get_num_links', vars(pair.tt_ccl))
        with self.assertRaises(ValueError), policy.sampler_links(sampler((1, 2)), 3):
            self.fail('entered')

    def test_the_ring_takes_one_or_two_links_and_restores(self):
        ring = sampler((1, 4))
        for links in (1, 2):
            with policy.sampler_links(ring, links):
                self.assertEqual(ring.num_argmax_gather_links, links)
                self.assertEqual([ring.tt_ccl.get_num_links(axis) for axis in (None, 0, 1)], [2, 2, 2])
                with self.assertRaises(ValueError):
                    ring.tt_ccl.get_num_links(2)
            self.assertEqual(ring.num_argmax_gather_links, 1)
            self.assertNotIn('get_num_links', vars(ring.tt_ccl))
        for links in (4, 0, True, '2'):
            with self.assertRaises(ValueError), policy.sampler_links(sampler((1, 4)), links):
                self.fail('entered')
        with self.assertRaises(ValueError), policy.sampler_links(sampler((2, 2)), 2):
            self.fail('entered')

    def test_a_ring_exception_restores_an_existing_override(self):
        ring = sampler((1, 4))
        original = lambda axis=None: 1  # noqa: E731
        ring.tt_ccl.get_num_links = original
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            with policy.sampler_links(ring, 2):
                raise RuntimeError('boom')
        self.assertIs(ring.tt_ccl.get_num_links, original)


class ProjectionTests(unittest.TestCase):
    def test_the_pair_is_the_pinned_validate(self):
        environment = {'TT_METAL_SIMULATOR': 'sim'}
        self.assertEqual(policy.projection_validate(environment, (1, 2)), projection_link_policy.validate(environment))

    def test_the_ring_takes_explicit_links_against_its_descriptor(self):
        self.assertEqual(policy.projection_validate({'TT_METAL_SIMULATOR': 'sim'}, (1, 4))['requested_links'], 1)
        with self.assertRaises(ValueError):
            policy.projection_validate(dict(TT_METAL_SIMULATOR='sim', QWEN_PROJECTION_LINKS='2'), (1, 4))
        with self.assertRaises(ValueError):
            policy.projection_validate(dict(QWEN_PROJECTION_LINKS='4'), (1, 4))
        environment = dict(QWEN_PROJECTION_LINKS='2', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1',
                           TT_MESH_GRAPH_DESC_PATH=tp4_mesh.DESCRIPTOR_PATH)
        with self.assertRaises((ValueError, OSError)):
            policy.projection_validate(environment, (1, 4))   # the image path does not exist here
        original = policy.audit_descriptor
        try:
            policy.audit_descriptor = lambda environ, shape, read=None: 'ok'
            report = policy.projection_validate(environment, (1, 4))
            self.assertEqual((report['requested_links'], report['backend'], report['mesh']), (2, 'hardware', [1, 4]))
            for missing in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'):
                with self.assertRaises(ValueError):
                    policy.projection_validate({k: v for k, v in environment.items() if k != missing}, (1, 4))
        finally:
            policy.audit_descriptor = original


class DescriptorAuditTests(unittest.TestCase):
    def test_each_shape_holds_its_descriptor_path_and_bytes(self):
        ring_bytes = (HERE / tp4_mesh.DESCRIPTOR_NAME).read_bytes()
        environment = dict(TT_MESH_GRAPH_DESC_PATH=tp4_mesh.DESCRIPTOR_PATH)
        self.assertEqual(policy.audit_descriptor(environment, (1, 4), read=lambda path: ring_bytes),
                         hashlib.sha256(ring_bytes).hexdigest())
        with self.assertRaisesRegex(ValueError, 'audited bytes'):
            policy.audit_descriptor(environment, (1, 4), read=lambda path: ring_bytes + b' ')
        with self.assertRaisesRegex(ValueError, 'requires TT_MESH_GRAPH_DESC_PATH'):
            policy.audit_descriptor(dict(TT_MESH_GRAPH_DESC_PATH='/elsewhere'), (1, 4), read=lambda path: ring_bytes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = root / sampling_link_policy.DESCRIPTOR
            pair.parent.mkdir(parents=True)
            pair.write_bytes(b'not the audited pair descriptor')
            with self.assertRaisesRegex(ValueError, 'audited bytes'):
                policy.audit_descriptor(dict(TT_METAL_HOME=str(root), TT_MESH_GRAPH_DESC_PATH=str(pair)), (1, 2))

    def test_the_environment_names_its_shape(self):
        self.assertEqual(policy.environment_shape(dict(TT_MESH_GRAPH_DESC_PATH=tp4_mesh.DESCRIPTOR_PATH)), (1, 4))
        pair = str(Path('/opt/tt-metal') / sampling_link_policy.DESCRIPTOR)
        self.assertEqual(policy.environment_shape(dict(TT_METAL_HOME='/opt/tt-metal', TT_MESH_GRAPH_DESC_PATH=pair)),
                         (1, 2))
        self.assertIsNone(policy.environment_shape(dict(TT_MESH_GRAPH_DESC_PATH='/opt/other.textproto')))


class TargetTests(unittest.TestCase):
    def test_the_ring_scope_covers_every_owner_and_restores(self):
        target, other = model((1, 4)), Collective()
        with policy.target_links(target, 1) as report:
            for layer in target.layers:
                for owner in (layer, layer.attention, layer.feed_forward):
                    self.assertEqual(owner.tt_ccl.get_num_links(0), 1)
            self.assertEqual(target.tt_ccl.get_num_links(), 1)
            self.assertEqual(other.get_num_links(), 2)
        self.assertTrue(report['restored'])
        self.assertEqual(report['owners_validated'], 193)
        self.assertEqual(report['mesh'], [1, 4])
        self.assertNotIn('get_num_links', vars(target.tt_ccl))
        self.assertNotIn('_qwen_target_link_scope', vars(target))

    def test_the_ring_refuses_what_the_pair_policy_refuses(self):
        for mutation in ('links', 'layers', 'owner', 'nested'):
            target = model((1, 4))
            links = 2
            if mutation == 'links':
                links = 4
            elif mutation == 'layers':
                target.layers.pop()
            elif mutation == 'owner':
                target.layers[3].feed_forward.tt_ccl = Collective()
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                with policy.target_links(target, links):
                    if mutation == 'nested':
                        with policy.target_links(target, links):
                            self.fail('nested')
            self.assertNotIn('get_num_links', vars(target.tt_ccl))

    def test_the_pair_delegates_to_model_link_policy(self):
        target = model((1, 2))
        with policy.target_links(target, 4) as report:
            self.assertEqual(target.tt_ccl.get_num_links(1), 4)
        self.assertTrue(report['restored'])
        self.assertNotIn('mesh', report, "the pinned policy's own report")


FAKE_TT_CCL = '''import ttnn


def get_num_links(mesh_device, cluster_axis=None):
    device_name = "P150x4" if mesh_device.get_num_devices() == 4 else "P300"
    link_dict = {"P300": (2, 2), "P150x4": (2, 2)}
    device_links = link_dict[device_name]
    if cluster_axis is None:
        return min(device_links)
    return device_links[cluster_axis]
'''


def descriptor_links(pairs):
    """A block-style cluster descriptor with the given (a, b) links, one record per link."""
    lines = ['ethernet_connections:']
    channel = {}
    for a, b in pairs:
        channel[a], channel[b] = channel.get(a, 3) + 1, channel.get(b, 3) + 1
        lines += ['  -', '    - chip: %d' % a, '      chan: %d' % channel[a], '    - chip: %d' % b, '      chan: %d' % channel[b]]
    lines.append('ethernet_connections_to_remote_devices:')
    return '\n'.join(lines) + '\n'


class CclLinkPatchTests(unittest.TestCase):
    def patched(self, descriptor_text):
        source = ccl_link_patch.patch_ccl(FAKE_TT_CCL)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'cluster.yaml')
        with open(path, 'w') as handle:
            handle.write(descriptor_text)
        fake_ttnn = types.ModuleType('ttnn')
        fake_ttnn.cluster = SimpleNamespace(serialize_cluster_descriptor=lambda: path)
        logged = []
        fake_loguru = types.ModuleType('loguru')
        fake_loguru.logger = SimpleNamespace(info=lambda *a: logged.append(a), warning=lambda *a: logged.append(a))
        saved = {name: sys.modules.get(name) for name in ('ttnn', 'loguru')}
        sys.modules['ttnn'], sys.modules['loguru'] = fake_ttnn, fake_loguru
        try:
            namespace = {}
            exec(compile(source, 'tt_ccl.py', 'exec'), namespace)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return namespace, logged

    def test_a_pair_is_counted_between_chip_0_and_1_as_before(self):
        namespace, logged = self.patched(descriptor_links([(0, 1)] * 4))
        mesh = SimpleNamespace(get_num_devices=lambda: 2, get_device_ids=lambda: [5, 6])
        self.assertEqual(namespace['get_num_links'](mesh), 4)
        self.assertTrue(any('overriding' in str(entry) for entry in logged))

    def test_a_ring_takes_its_smallest_edge_and_never_goes_below_the_table(self):
        four = [(0, 1)] * 4 + [(1, 2)] * 4 + [(2, 3)] * 4 + [(3, 0)] * 4
        namespace, _ = self.patched(descriptor_links(four))
        ring = SimpleNamespace(get_num_devices=lambda: 4, get_device_ids=lambda: [0, 1, 2, 3])
        self.assertEqual(namespace['get_num_links'](ring), 4, 'four links on every ring edge')
        namespace, _ = self.patched(descriptor_links(four[:-3]))           # the closing edge has one link
        ring = SimpleNamespace(get_num_devices=lambda: 4, get_device_ids=lambda: [0, 1, 2, 3])
        self.assertEqual(namespace['get_num_links'](ring), 2, 'the table, never fewer')
        namespace, _ = self.patched(descriptor_links([(a, b) for a in range(4) for b in range(a + 1, 4)] * 2))
        ring = SimpleNamespace(get_num_devices=lambda: 4, get_device_ids=lambda: [2, 0, 3, 1])
        self.assertEqual(namespace['get_num_links'](ring), 2, 'the full mesh at two links: the table already')

    def test_the_patch_script_still_refuses_a_second_application(self):
        once = ccl_link_patch.patch_ccl(FAKE_TT_CCL)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            ccl_link_patch.patch_ccl(once)


if __name__ == '__main__':
    unittest.main()
