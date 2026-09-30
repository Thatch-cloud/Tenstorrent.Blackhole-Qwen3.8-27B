"""tp4_mesh: the four-card (1, 4) ring's descriptor, the cluster-descriptor link parse and the ring check."""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tp4_mesh as mesh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NL = chr(10)

# The six pairs of a fully connected four-card fabric, two links each (the 2026-09-30 measurement's shape;
# UMD chip ids are arbitrary here, as they are on the rig after every reset).
K4 = {(0, 1): 2, (0, 2): 2, (0, 3): 2, (1, 2): 2, (1, 3): 2, (2, 3): 2}


def block_yaml(connections, remote=()):
    """A UMD cluster descriptor in block style: one '-' line per connection, then its two endpoints."""
    lines = ['arch: {0: blackhole, 1: blackhole, 2: blackhole, 3: blackhole}', 'chips: {}', 'ethernet_connections:']
    for (a, x), (b, y) in connections:
        lines += ['  -', '    - chip: %d' % a, '      chan: %d' % x, '    - chip: %d' % b, '      chan: %d' % y]
    lines.append('ethernet_connections_to_remote_devices:')
    for (a, x), (b, y) in remote:
        lines += ['  -', '    - chip: %d' % a, '      chan: %d' % x, '    - remote_chip_id: %d' % b, '      chan: %d' % y]
    lines.append('harvesting: {}')
    return NL.join(lines) + NL


def flow_yaml(connections):
    lines = ['ethernet_connections: [']
    lines += ['  [{chip: %d, chan: %d}, {chip: %d, chan: %d}],' % (a, x, b, y) for (a, x), (b, y) in connections]
    lines += [']', 'chips_with_mmio: [{0: 0}]']
    return NL.join(lines) + NL


def connections(links):
    """Two endpoints per link, distinct channels per chip."""
    used, result = {}, []
    for (a, b), count in sorted(links.items()):
        for _ in range(count):
            x, y = used.get(a, 3) + 1, used.get(b, 3) + 1
            used[a], used[b] = x, y
            result.append(((a, x), (b, y)))
    return result


class DescriptorTests(unittest.TestCase):
    def test_the_checked_in_descriptor_is_the_generated_one(self):
        with open(os.path.join(HERE, mesh.DESCRIPTOR_NAME), 'rb') as handle:
            data = handle.read()
        self.assertEqual(data.decode('utf-8'), mesh.descriptor_text())
        self.assertNotIn(b'\r', data)
        self.assertEqual(mesh.descriptor_problems(data.decode('utf-8')), [])

    def test_it_is_a_2x2_two_channel_relaxed_single_mesh(self):
        found = mesh.parse_descriptor(mesh.descriptor_text())
        self.assertEqual((found['dims'], found['host'], found['channels'], found['policy'], found['arch']),
                         ((2, 2), (1, 1), 2, 'RELAXED', 'BLACKHOLE'))
        self.assertFalse(found['ring_axes'], 'no dim_types: a non-Galaxy fabric builds MESH connectivity anyway')

    def test_upstream_shapes_that_are_not_the_ring_are_named(self):
        four_channel = mesh.descriptor_text(channels=4)
        self.assertEqual(mesh.descriptor_problems(four_channel), ['4 channels, not the 2 each card pair trains'])
        line = mesh.descriptor_text(dims=(1, 4))
        self.assertTrue(any('needs every 4-cycle edge' in problem for problem in mesh.descriptor_problems(line)))
        ring_axes = mesh.descriptor_text().replace('dims: [ 2, 2 ] }', 'dims: [ 2, 2 ] dim_types: [ RING, RING ] }')
        self.assertTrue(any('dim_types' in problem for problem in mesh.descriptor_problems(ring_axes)))
        self.assertTrue(mesh.descriptor_problems('nonsense')[0].startswith('unreadable'))
        twice = mesh.descriptor_text() + mesh.descriptor_text()
        self.assertTrue(mesh.descriptor_problems(twice)[0].startswith('unreadable'))

    def test_the_descriptor_names_no_card(self):
        text = mesh.descriptor_text()
        for word in ('blackhole-', 'pinnings', 'asic', 'tray_id', '0000:'):
            self.assertNotIn(word, text)

    def test_the_image_lays_it_where_the_profiles_look(self):
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-overlay.txt'), encoding='utf-8') as handle:
            manifest = handle.read()
        line = [raw for raw in manifest.splitlines() if raw.startswith(mesh.DESCRIPTOR_SOURCE)]
        self.assertEqual(len(line), 1)
        self.assertIn(mesh.DESCRIPTOR_PATH, line[0].split())


class EthernetParseTests(unittest.TestCase):
    def test_block_and_flow_styles_give_the_same_links(self):
        wires = connections(K4)
        self.assertEqual(mesh.parse_ethernet_links(block_yaml(wires)), K4)
        self.assertEqual(mesh.parse_ethernet_links(flow_yaml(wires)), K4)

    def test_a_connection_listed_from_both_ends_counts_once(self):
        wires = connections({(0, 1): 2})
        both = wires + [(second, first) for first, second in wires]
        self.assertEqual(mesh.parse_ethernet_links(block_yaml(both)), {(0, 1): 2})

    def test_links_to_remote_devices_are_not_counted(self):
        text = block_yaml(connections({(0, 1): 1}), remote=[((0, 9), (7, 9))])
        self.assertEqual(mesh.parse_ethernet_links(text), {(0, 1): 1})

    def test_no_section_is_none_and_an_odd_section_refuses(self):
        self.assertIsNone(mesh.parse_ethernet_links('arch: {}' + NL))
        odd = 'ethernet_connections:' + NL + '  -' + NL + '    - chip: 0' + NL + '      chan: 4' + NL
        with self.assertRaises(ValueError):
            mesh.parse_ethernet_links(odd)


class RingTests(unittest.TestCase):
    def test_a_full_mesh_is_a_ring_in_any_order_with_the_diagonals_unused(self):
        for order in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 1, 0, 2]):
            report = mesh.check_ring(order, K4)
            self.assertTrue(report['ok'] and not report['degraded'], report)
            self.assertEqual([edge['links'] for edge in report['edges']], [2, 2, 2, 2])
            self.assertEqual((report['edges'][-1]['a'], report['edges'][-1]['b']), (order[3], order[0]),
                             'the closing edge is checked')
            self.assertEqual(report['unused'], 4, 'two diagonals, two links each')
        self.assertIn('ring OK', mesh.describe(mesh.check_ring([0, 1, 2, 3], K4)))

    def test_a_missing_closing_edge_breaks_the_ring(self):
        chain = {(0, 1): 2, (1, 2): 2, (2, 3): 2}
        report = mesh.check_ring([0, 1, 2, 3], chain)
        self.assertFalse(report['ok'])
        self.assertEqual(report['problems'], ['ring edge 3-0 has 0 trained links: a Ring collective has no path there'])
        self.assertIn('ring BROKEN', mesh.describe(report))

    def test_one_link_on_an_edge_is_degraded_not_broken(self):
        links = dict(K4)
        links[(1, 2)] = 1
        report = mesh.check_ring([0, 1, 2, 3], links)
        self.assertTrue(report['ok'])
        self.assertTrue(report['degraded'])
        self.assertIn('ring DEGRADED', mesh.describe(report))

    def test_ids_the_descriptor_does_not_list_are_unknown_never_broken(self):
        report = mesh.check_ring([10, 11, 12, 13], K4)
        self.assertIsNone(report['ok'])
        self.assertIn('do not share a numbering', report['problems'][0])
        self.assertIn('ring UNKNOWN', mesh.describe(report))
        module = types.SimpleNamespace(open_mesh_device=lambda: 'mesh')
        logged = []
        mesh.install_ring_check(module, logged.append, check=lambda opened: report)
        self.assertEqual(module.open_mesh_device(), 'mesh', 'an unrelated numbering is logged, never refused')
        self.assertIn('ring UNKNOWN', logged[0])

    def test_the_order_must_be_four_distinct_devices(self):
        self.assertFalse(mesh.check_ring([0, 1, 2], K4)['ok'])
        self.assertFalse(mesh.check_ring([0, 1, 1, 2], K4)['ok'])

    def test_hamiltonian_rings(self):
        self.assertEqual(mesh.hamiltonian_rings([3, 2, 1, 0], K4), [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)])
        square = {(0, 1): 2, (1, 3): 2, (3, 2): 2, (0, 2): 2}
        self.assertEqual(mesh.hamiltonian_rings([0, 1, 2, 3], {tuple(sorted(k)): v for k, v in square.items()}),
                         [(0, 1, 3, 2)])
        self.assertEqual(mesh.hamiltonian_rings([0, 1, 2, 3], {(0, 1): 2, (1, 2): 2, (2, 3): 2}), [])

    def test_the_sampler_fits_at_four_devices_only(self):
        self.assertFalse(mesh.sampler_fits(2))
        self.assertTrue(mesh.sampler_fits(4))
        self.assertEqual(-(-mesh.VOCABULARY // 4), 62080)


class FakeMesh(object):
    def __init__(self, shape, ids):
        self.shape, self._ids = shape, ids

    def get_device_ids(self):
        return list(self._ids)


def fake_ttnn(path='/tmp/cluster.yaml'):
    return types.SimpleNamespace(cluster=types.SimpleNamespace(serialize_cluster_descriptor=lambda: path))


class OpenMeshCheckTests(unittest.TestCase):
    def test_the_opened_mesh_is_checked_against_the_cluster_descriptor(self):
        text = block_yaml(connections(K4))
        report = mesh.check_open_mesh(FakeMesh((1, 4), [2, 0, 3, 1]), fake_ttnn(), read=lambda path: text)
        self.assertTrue(report['ok'])
        self.assertEqual(report['order'], [2, 0, 3, 1])
        self.assertIsNone(mesh.check_open_mesh(FakeMesh((1, 2), [0, 1]), fake_ttnn(), read=lambda path: text))
        with self.assertRaises(ValueError):
            mesh.check_open_mesh(FakeMesh((1, 4), [0, 1, 2, 3]), fake_ttnn(), read=lambda path: 'arch: {}')

    def test_the_wrapper_refuses_a_broken_ring_logs_a_degraded_one_and_survives_its_own_failure(self):
        logged = []
        opened = FakeMesh((1, 4), [0, 1, 2, 3])
        module = types.SimpleNamespace(open_mesh_device=lambda *args, **kwargs: opened)
        reports = iter([mesh.check_ring([0, 1, 2, 3], K4),
                        mesh.check_ring([0, 1, 2, 3], {(0, 1): 2, (1, 2): 1, (2, 3): 2, (0, 3): 2}),
                        mesh.check_ring([0, 1, 2, 3], {(0, 1): 2, (1, 2): 2, (2, 3): 2})])

        def check(mesh_device):
            report = next(reports, None)
            if report is None:
                raise OSError('descriptor unreadable')
            return report

        self.assertTrue(mesh.install_ring_check(module, logged.append, check=check))
        self.assertFalse(mesh.install_ring_check(module, logged.append, check=check), 'installed once')
        self.assertIs(module.open_mesh_device('tt', 'decode_only'), opened)
        self.assertIs(module.open_mesh_device(), opened)
        with self.assertRaises(mesh.RingError):
            module.open_mesh_device()
        self.assertIs(module.open_mesh_device(), opened, 'a check that cannot run never stops serving')
        self.assertIn('ring OK', logged[0])
        self.assertIn('ring DEGRADED', logged[1])
        self.assertIn('ring BROKEN', logged[2])
        self.assertIn('ring check could not run: OSError', logged[3])

    def test_the_module_imports_nothing_but_the_stdlib(self):
        with open(mesh.__file__, encoding='utf-8') as handle:
            source = handle.read()
        imports = [line for line in source.splitlines() if line.startswith(('import ', 'from '))]
        self.assertEqual(imports, ['import re'])


if __name__ == '__main__':
    unittest.main()
