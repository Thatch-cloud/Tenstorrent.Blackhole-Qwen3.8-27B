"""The registry's shard map (mesh coordinate -> the PCI address of its card): B1 of the integration review, amendment A3 of INTERFACE.md.

The plugin builds the registry with the map filled, from ``qwen_p2p_mesh_devices``, an extension op that allocates nothing on a card. These tests cover the pure
part: the records the op returns, what is refused, that nothing is imported when the module is, and that a registry is completed from a COPY. The boot of a
registry that arrives without a map, through the production factory, is in test_startup_sequence.py (stand-ins) and test_runtime_p2pio_seam.py (the real
runtime, CardLink and P2PIO). The module needs neither vLLM nor numpy.
"""

import dataclasses
import os
import subprocess
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from ttkvp2p import registry as registry_module  # noqa: E402
from ttkvp2p.registry import ShardMapError, TTKVRegistry, resolve_shard_bdf, shard_bdf_from_devices, with_shard_bdf  # noqa: E402

BDFS = ['0000:01:00.0', '0000:02:00.0', '0000:03:00.0', '0000:04:00.0']        # synthetic: no real card has these


def record(position, bdf=None, number=None, **more):
    """One device record as qwen_p2p_mesh_devices / qwen_p2p_staging_info()['devices'] carry it. The logical device id and the tt-kmd number differ, as on the rig."""
    entry = {'mesh_coord': [position, 0], 'mesh_index': position, 'device_id': position, 'mmio_chip_id': position,
             'pcie_device': (position + 1) % 4 if number is None else number, 'pci_bus_id': BDFS[position] if bdf is None else bdf,
             'pci_bus_id_source': 'cluster_descriptor', 'pci_bus_id_sysfs': BDFS[position], 'pci_bus_id_agrees': True}
    entry.update(more)
    return entry


def records(**by_position):
    return [record(position, **by_position.get(str(position), {})) for position in range(4)]


def make_registry(shard_bdf=()):
    return TTKVRegistry(mesh_device=object(), tensors=tuple(range(10)), layer_to_tensor_idx={}, group_layers=(), num_blocks=19305, pages_per_block=64,
                        row_bytes_per_chip=69632, shard_bdf=tuple(shard_bdf))


class ShardMapFromDevices(unittest.TestCase):
    def test_the_map_is_one_lower_case_address_per_mesh_coordinate_in_mesh_order(self):
        self.assertEqual(shard_bdf_from_devices(records()), tuple(BDFS))
        shouted = [record(p, bdf=BDFS[p].upper()) for p in range(4)]
        self.assertEqual(shard_bdf_from_devices(shouted), tuple(BDFS))
        self.assertEqual(shard_bdf_from_devices([{'pci_bus_id': BDFS[0], 'pcie_device': 0}]), (BDFS[0],), 'a record without mesh_index is taken in list order')

    def test_what_cannot_be_trusted_is_refused_not_guessed(self):
        cases = {
            'an empty mesh': [],
            'no PCI address': records(**{'1': {'bdf': ''}}),
            'a None PCI address': [record(0), dict(record(1), pci_bus_id=None)] + [record(2), record(3)],
            'no tt-kmd device number': [record(0), dict(record(1), pcie_device=None)] + [record(2), record(3)],
            'a negative tt-kmd number': [record(0), dict(record(1), pcie_device=-1)] + [record(2), record(3)],
            'a boolean tt-kmd number': [record(0), dict(record(1), pcie_device=True)] + [record(2), record(3)],
            'sysfs says the address is not that device\'s': [record(0), dict(record(1), pci_bus_id_agrees=False)] + [record(2), record(3)],
            'two coordinates, one card': [record(0), record(1, bdf=BDFS[0])] + [record(2), record(3)],
            'two coordinates, one tt-kmd number': [record(0), record(1, number=record(0)['pcie_device'])] + [record(2), record(3)],
            'records out of mesh order': [record(1), record(0), record(2), record(3)],
            'a record that is not a record': [record(0), 'card-1', record(2), record(3)],
        }
        for what, devices in cases.items():
            with self.subTest(what):
                with self.assertRaises(ShardMapError):
                    shard_bdf_from_devices(devices)

    def test_a_disagreement_with_sysfs_says_which_device_and_both_addresses(self):
        with self.assertRaises(ShardMapError) as raised:
            shard_bdf_from_devices([record(0), dict(record(1), pci_bus_id_agrees=False, pci_bus_id_sysfs='0000:99:00.0')])
        self.assertIn('mesh device 1', str(raised.exception))
        self.assertIn('0000:99:00.0', str(raised.exception))

    def test_an_unknown_agreement_is_not_a_disagreement(self):
        """pci_bus_id_agrees is None when there was nothing to compare (the container has no sysfs): the descriptor's address stands."""
        self.assertEqual(shard_bdf_from_devices([dict(r, pci_bus_id_agrees=None, pci_bus_id_sysfs=None) for r in records()]), tuple(BDFS))


class ResolveShardBdf(unittest.TestCase):
    def ops(self, devices=None, raises=None):
        calls = []
        module = types.ModuleType('qwen_kv_read')

        def qwen_p2p_mesh_devices(mesh):
            calls.append(mesh)
            if raises is not None:
                raise raises
            return records() if devices is None else devices

        module.qwen_p2p_mesh_devices = qwen_p2p_mesh_devices
        module.calls = calls
        return module

    def test_it_asks_the_extension_about_the_mesh_and_nothing_else(self):
        ops, mesh = self.ops(), object()
        self.assertEqual(resolve_shard_bdf(mesh, ops), tuple(BDFS))
        self.assertEqual(ops.calls, [mesh])
        self.assertEqual([name for name in vars(ops) if name.startswith('qwen_p2p_')], ['qwen_p2p_mesh_devices'], 'no staging, no warm-up, no queue: the map needs none')

    def test_an_extension_that_predates_the_op_is_refused_with_the_reason(self):
        with self.assertRaises(ShardMapError) as raised:
            resolve_shard_bdf(object(), types.ModuleType('qwen_kv_read'))
        self.assertIn('qwen_p2p_mesh_devices', str(raised.exception))

    def test_a_refusal_or_a_crash_of_the_op_is_a_shard_map_error(self):
        for error in (ValueError('QwenP2PRefused stand-in: the mesh device is closed'), RuntimeError('boom')):
            with self.subTest(type(error).__name__):
                with self.assertRaises(ShardMapError) as raised:
                    resolve_shard_bdf(object(), self.ops(raises=error))
                self.assertIn(str(error), str(raised.exception))

    def test_an_extension_that_cannot_be_imported_is_a_shard_map_error(self):
        saved = sys.modules.pop('qwen_kv_read', None)
        sys.modules['qwen_kv_read'] = None                    # an import of it raises ImportError
        try:
            with self.assertRaises(ShardMapError) as raised:
                resolve_shard_bdf(object())
            self.assertIn('cannot be imported', str(raised.exception))
        finally:
            if saved is None:
                sys.modules.pop('qwen_kv_read', None)
            else:
                sys.modules['qwen_kv_read'] = saved

    def test_the_extension_is_imported_by_name_when_none_is_given(self):
        ops = self.ops()
        saved = sys.modules.get('qwen_kv_read')
        sys.modules['qwen_kv_read'] = ops
        try:
            self.assertEqual(resolve_shard_bdf(object()), tuple(BDFS))
        finally:
            if saved is None:
                sys.modules.pop('qwen_kv_read', None)
            else:
                sys.modules['qwen_kv_read'] = saved


class WithShardBdf(unittest.TestCase):
    def test_a_registry_without_a_map_is_completed_as_a_copy(self):
        bare = make_registry()
        self.assertEqual((bare.shard_bdf, bare.num_shards, bare.object_bytes), ((), 0, 0))
        ops = ResolveShardBdf().ops()
        filled = with_shard_bdf(bare, ops)
        self.assertEqual(filled.shard_bdf, tuple(BDFS))
        self.assertEqual((filled.num_shards, filled.object_bytes), (4, 4 * 10 * 69632))
        self.assertEqual(bare.shard_bdf, (), 'the registry is frozen: the original is untouched')
        for name in ('mesh_device', 'tensors', 'layer_to_tensor_idx', 'group_layers', 'num_blocks', 'pages_per_block', 'row_bytes_per_chip'):
            self.assertIs(getattr(filled, name), getattr(bare, name), name + ' is carried over, not copied')
        self.assertEqual(ops.calls, [bare.mesh_device])

    def test_a_registry_that_has_a_map_is_returned_as_it_is_and_the_extension_is_not_asked(self):
        have = make_registry(BDFS)
        ops = ResolveShardBdf().ops()
        self.assertIs(with_shard_bdf(have, ops), have)
        self.assertEqual(ops.calls, [])

    def test_the_filled_registry_fits_the_gemma_layout(self):
        from_ops = with_shard_bdf(make_registry(), ResolveShardBdf().ops())
        layout = types.SimpleNamespace(name='gemma-chipmajor-v1', tensors=10, chips=4, row_bytes=69632, pages_per_block=64, object_bytes=2785280)
        self.assertEqual(from_ops.problems(layout), [])
        self.assertTrue(make_registry().problems(layout), 'the registry as the plugin built it before the fix does not')


class ImportRules(unittest.TestCase):
    def test_importing_the_registry_imports_neither_ttnn_nor_the_extension(self):
        # I14, in a fresh interpreter: the API-server process imports ttkvp2p for the HMA check and has neither
        code = ('import sys; sys.path.insert(0, %r); import ttkvp2p.registry as r; '
                'bad = [m for m in ("ttnn", "qwen_kv_read") if m in sys.modules]; '
                'sys.exit("imported " + ",".join(bad) if bad else 0)') % PACKAGE_ROOT
        done = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(dataclasses.is_dataclass(registry_module.TTKVRegistry))


if __name__ == '__main__':
    unittest.main()
