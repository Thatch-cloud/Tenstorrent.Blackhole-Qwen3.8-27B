"""The fabric-upload pack on the CPU (docs/tp4-fabric-upload.md): fabric_upload_plan's arithmetic, pairing and decision
rule; the two probes (tp4_h2d_probe, tp4_p2p_probe) run end to end against a fake ttnn whose transfers can be made to
broadcast, refuse or corrupt; the job parser and the workflow know both probes; and the pack's templates parse.

Run at py 3.11 with torch: `python -B -m unittest test_fabric_upload` from scripts/ci."""

import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import c2_serving_job as job  # noqa: E402
import fabric_upload_plan as plan  # noqa: E402
import tp4_h2d_probe as h2d  # noqa: E402
import tp4_p2p_probe as p2p  # noqa: E402
import tp4_upload_probe_common as common  # noqa: E402

PACK = os.path.join(HERE, 'references', 'fabric-upload-jobs')
ORDER = [1, 0, 2, 3]            # chip ids in (1, 4) order
WIDTHS = {0: 16, 1: 16, 2: 4, 3: 4}  # by chip: positions 0 and 1 are x16, 2 and 3 are x4


# ---------------------------------------------------------------------------------------------------------------------
# A fake ttnn: tensors are lists of torch shards; devices are meshes of n chips; views write one shard of a parent.
# ---------------------------------------------------------------------------------------------------------------------
class FakeTensor(object):
    def __init__(self, shards, device=None, dtype=None, layout=None, parent=None, index=None, populated=None):
        self.shards, self.device, self.dtype, self.layout, self.parent, self.index = shards, device, dtype, layout, parent, index
        self.populated = populated
        first = next(shard for shard in shards if shard is not None)
        self.shape = tuple(first.shape)
        self.spec = ('spec', self.shape, dtype, layout)


class FakeMesh(object):
    def __init__(self, fake, chips):
        self.fake, self.chips = fake, list(chips)

    def get_device_ids(self):
        return list(self.chips)

    def get_num_devices(self):
        return len(self.chips)

    def create_submeshes(self, shape):
        if self.fake.refuse_submesh:
            raise RuntimeError('submeshes refused by the fake')
        return [FakeMesh(self.fake, [chip]) for chip in self.chips]

    def create_submesh(self, shape, offset):
        if self.fake.refuse_submesh:
            raise RuntimeError('submeshes refused by the fake')
        return FakeMesh(self.fake, [self.chips[offset[1]]])

    def quiesce_devices(self):
        self.fake.quiesced = True


class Mapper(object):
    def __init__(self, kind, mesh, dim=0, pos=None):
        self.kind, self.mesh, self.dim, self.pos = kind, mesh, dim, pos


class FakeTtnn(object):
    """view: what a single-device host tensor written into a get_device_tensors view does ('broadcast', as the pinned
    runtime reads: a 1x1 host buffer is replicated to the mesh; or exact/missing/refuse). mapper: what a one-coordinate
    host tensor does (exact as the pinned runtime reads; or broadcast/corrupt/refuse)."""
    bfloat16, bfloat8_b = 'bf16', 'bf8'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT = 'rm', 'tile'
    DRAM_MEMORY_CONFIG = 'dram'

    class FabricConfig(object):
        FABRIC_1D, FABRIC_1D_RING = 'FABRIC_1D', 'FABRIC_1D_RING'

    class Topology(object):
        Linear, Ring = 'Linear', 'Ring'

    class BufferType(object):
        DRAM, L1 = 'DRAM', 'L1'

    def __init__(self, descriptor_path, view='broadcast', mapper='exact', refuse_submesh=False, corrupt_p2p=False,
                 fail_open=False, refuse_socket=False):
        self.view, self.mapper, self.refuse_submesh, self.corrupt_p2p = view, mapper, refuse_submesh, corrupt_p2p
        self.fail_open, self.refuse_socket = fail_open, refuse_socket
        self.descriptor_path = descriptor_path
        self.cluster = type('cluster', (), {'serialize_cluster_descriptor': staticmethod(lambda: descriptor_path)})
        self.calls, self.closed, self.quiesced, self.fabric = [], False, False, None
        fake = self

        class Experimental(object):
            @staticmethod
            def send_async(tensor, socket):
                fake.calls.append('send')
                socket['channel'].append(tensor.shards[0].clone())

            @staticmethod
            def recv_async(tensor, socket):
                tensor.shards[0].copy_(socket['channel'].pop(0))
        self.experimental = Experimental

    # mesh lifecycle
    def set_fabric_config(self, value):
        self.fabric = value

    def MeshShape(self, *dims):
        return tuple(dims)

    def MeshCoordinate(self, row, col):
        return (row, col)

    def open_mesh_device(self, shape, **kwargs):
        if self.fail_open:
            raise RuntimeError('No core coordinate found at (1, 2)')
        self.mesh = FakeMesh(self, ORDER)
        return self.mesh

    def close_mesh_device(self, mesh):
        self.closed = True

    def synchronize_device(self, device):
        self.calls.append('sync')

    def Shape(self, dims):
        return tuple(dims)

    # mappers
    def ShardTensorToMesh(self, mesh, dim=0):
        return Mapper('shard', mesh, dim)

    def ReplicateTensorToMesh(self, mesh):
        return Mapper('replicate', mesh)

    def ConcatMeshToTensor(self, mesh, dim=0):
        return Mapper('concat', mesh, dim)

    def PlacementReplicate(self):
        return 'R'

    def MeshMapperConfig(self, placements, shape=None, offset=None):
        return dict(placements=placements, shape=shape, offset=offset)

    def create_mesh_mapper(self, mesh, config):
        assert config['shape'] == (1, 1) and config['placements'] == ['R', 'R']
        return Mapper('one', mesh, pos=config['offset'][1])

    # sockets
    def CoreCoord(self, x, y):
        return (x, y)

    def MeshCoreCoord(self, coord, core):
        return (coord, core)

    def SocketConnection(self, sender, receiver):
        return (sender, receiver)

    def SocketMemoryConfig(self, storage, fifo):
        return dict(storage=storage, fifo=fifo)

    def SocketConfig(self, connections, memory):
        return dict(connections=connections, memory=memory)

    def create_socket_pair(self, sender, receiver, config):
        if self.refuse_socket:
            raise RuntimeError('socket refused by the fake')
        channel = []
        return dict(channel=channel, config=config), dict(channel=channel, config=config)

    # host <-> device
    def from_torch(self, tensor, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        populated = None
        if mesh_mapper is None:
            shards = [tensor.clone()]
        elif mesh_mapper.kind == 'shard':
            shards = [part.clone() for part in torch.chunk(tensor, len(mesh_mapper.mesh.chips), dim=mesh_mapper.dim)]
        elif mesh_mapper.kind == 'one':
            shards = [None] * len(mesh_mapper.mesh.chips)
            shards[mesh_mapper.pos] = tensor.clone()
            populated = [mesh_mapper.pos]
        else:
            shards = [tensor.clone() for _ in mesh_mapper.mesh.chips]
        return FakeTensor(shards, device=None, dtype=dtype, layout=layout, populated=populated)

    def to_torch(self, tensor, mesh_composer=None):
        if mesh_composer is not None:
            return torch.cat([shard.clone() for shard in tensor.shards], dim=mesh_composer.dim)
        assert len(tensor.shards) == 1, 'to_torch of a multi-shard tensor needs a composer'
        return tensor.shards[0].clone()

    def allocate_tensor_on_device(self, shape_or_spec, *args):
        if isinstance(shape_or_spec, tuple) and shape_or_spec and shape_or_spec[0] == 'spec':
            _, shape, dtype, layout = shape_or_spec
            device = args[0]
        else:
            shape, dtype, layout, device = shape_or_spec, args[0], args[1], args[2]
        return FakeTensor([torch.zeros(shape, dtype=torch.bfloat16) for _ in device.chips], device=device, dtype=dtype,
                          layout=layout)

    def deallocate(self, tensor):
        self.calls.append('deallocate')

    def get_device_tensors(self, tensor):
        if tensor.device is None:
            return [FakeTensor([shard], device=None, dtype=tensor.dtype, layout=tensor.layout) for shard in tensor.shards]
        return [FakeTensor([shard], device=tensor.device, dtype=tensor.dtype, layout=tensor.layout, parent=tensor, index=index)
                for index, shard in enumerate(tensor.shards)]

    def from_device(self, tensor):
        if tensor.parent is not None:
            return FakeTensor([tensor.parent.shards[tensor.index].clone()], device=None, dtype=tensor.dtype, layout=tensor.layout)
        return FakeTensor([shard.clone() for shard in tensor.shards], device=None, dtype=tensor.dtype, layout=tensor.layout)

    def copy_host_to_device_tensor(self, host, device_tensor):
        self.calls.append('copy')
        parent = device_tensor.parent if device_tensor.parent is not None else device_tensor
        if host.populated is not None:
            pos = host.populated[0]
            mode = self.mapper
            source = host.shards[pos]
        elif len(host.shards) == 1 and len(parent.shards) > 1:
            mode = self.view if device_tensor.parent is not None else 'broadcast'
            pos = device_tensor.index
            source = host.shards[0]
        else:
            assert len(host.shards) == len(device_tensor.shards), (len(host.shards), len(device_tensor.shards))
            for target, part in zip(device_tensor.shards, host.shards):
                target.copy_(part.reshape(target.shape))
            return
        if mode == 'refuse':
            raise RuntimeError('this write is refused by the fake')
        if mode == 'missing':
            return
        if mode == 'broadcast':
            for index in range(len(parent.shards)):
                parent.shards[index].copy_(source.reshape(parent.shards[index].shape))
            return
        if mode == 'corrupt':
            parent.shards[(pos + 1) % len(parent.shards)].fill_(99)
        parent.shards[pos].copy_(source.reshape(parent.shards[pos].shape))

    # fabric
    def p2p_compute_intermediate_tensor_spec(self, source, sender, receiver, topology):
        return ('spec', source.shape, source.dtype, source.layout)

    def point_to_point(self, source, sender, receiver, output_tensor=None, intermediate_tensor=None, topology=None):
        self.calls.append(('p2p', sender[1], receiver[1]))
        data = source.shards[sender[1]].clone()
        if self.corrupt_p2p and sender != receiver:
            data.view(-1)[0] = data.view(-1)[0] + 1
        output_tensor.shards[receiver[1]].copy_(data)
        return output_tensor


def descriptor_text():
    return ('arch: {}\nchips_with_mmio:\n' + ''.join('  - %d: %d\n' % (chip, chip) for chip in range(4)) +
            'ethernet_connections:\n' +
            ''.join('  - - chip: %d\n      chan: %d\n    - chip: %d\n      chan: %d\n' % (a, c, b, c) for a in range(4)
                    for b in range(a + 1, 4) for c in (0, 1)) + 'harvesting: {}\n')


def fake_sysfs(root):
    """A sysfs tree: chips 0 and 1 on root ports at x16, chips 2 and 3 behind one switch at x4 (by PCI index == chip)."""
    links = {}
    layout = {0: ('pci0000:00/0000:00:01.1', '0000:01:00.0', 16),
              1: ('pci0000:00/0000:00:03.1', '0000:02:00.0', 16),
              2: ('pci0000:40/0000:40:01.1/0000:41:00.0/0000:42:01.0', '0000:43:00.0', 4),
              3: ('pci0000:40/0000:40:01.1/0000:41:00.0/0000:42:02.0', '0000:44:00.0', 4)}
    for index, (chain, bdf, width) in layout.items():
        device = os.path.join(root, 'sys', 'devices', chain, bdf)
        os.makedirs(device)
        for name, value in (('current_link_speed', '32.0 GT/s PCIe'), ('current_link_width', str(width)),
                            ('max_link_speed', '32.0 GT/s PCIe'), ('max_link_width', '16')):
            with open(os.path.join(device, name), 'w') as handle:
                handle.write(value + '\n')
        os.makedirs(os.path.join(root, 'sys', 'bus', 'pci', 'devices'), exist_ok=True)
        os.symlink(device, os.path.join(root, 'sys', 'bus', 'pci', 'devices', bdf))
        char = os.path.join(root, 'sys', 'dev', 'char', '%d:%d' % (240, index))
        os.makedirs(char)
        os.symlink(device, os.path.join(char, 'device'))
        links[index] = bdf
    return dict(dev_root=os.path.join(root, 'dev'), sys_root=os.path.join(root, 'sys'),
                rdev_of=lambda node: os.makedev(240, int(os.path.basename(node))))


class Workspace(object):
    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        root = self.directory.name
        self.descriptor = os.path.join(root, 'cluster.yaml')
        with open(self.descriptor, 'w') as handle:
            handle.write(descriptor_text())
        self.links = fake_sysfs(root)
        self.output = os.path.join(root, 'report.json')
        return self

    def __exit__(self, *exc):
        self.directory.cleanup()


# ---------------------------------------------------------------------------------------------------------------------
class LinkTests(unittest.TestCase):
    def test_speeds_ceilings_and_generations(self):
        self.assertEqual(plan.parse_link_speed('32.0 GT/s PCIe'), 32.0)
        self.assertEqual(plan.parse_link_speed('16 GT/s'), 16.0)
        self.assertIsNone(plan.parse_link_speed('Unknown'))
        self.assertIsNone(plan.parse_link_speed(None))
        self.assertEqual(plan.pcie_ceiling_gbps(32.0, 16), 63.015)
        self.assertEqual(plan.pcie_ceiling_gbps(32.0, 4), 15.754)
        self.assertEqual(plan.pcie_ceiling_gbps(5.0, 1), 0.5, 'gen2 is 8b/10b')
        self.assertIsNone(plan.pcie_ceiling_gbps(33.0, 16))
        self.assertIsNone(plan.pcie_ceiling_gbps(32.0, None))

    def test_mmio_map_block_flow_and_absent(self):
        self.assertEqual(plan.parse_mmio_chips(descriptor_text()), {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(plan.parse_mmio_chips('chips_with_mmio: [{0: 2}, {1: 0}]\nharvesting: {}\n'), {0: 2, 1: 0})
        self.assertIsNone(plan.parse_mmio_chips('arch: {}\n'))
        self.assertIsNone(plan.parse_mmio_chips(None))

    def test_card_links_join_the_descriptor_to_sysfs(self):
        with Workspace() as space:
            rows = plan.card_links(ORDER, plan.parse_mmio_chips(descriptor_text()), **space.links)
        self.assertEqual([row['chip'] for row in rows], ORDER)
        self.assertEqual([row['width'] for row in rows], [16, 16, 4, 4])
        self.assertEqual(rows[0]['pci'], '0000:02:00.0', 'position 0 is chip 1')
        self.assertEqual(rows[2]['ports'], ['0000:42:01.0', '0000:41:00.0', '0000:40:01.1'])
        self.assertEqual(rows[2]['ports'][1:], rows[3]['ports'][1:], 'the two x4 cards share the switch')
        self.assertEqual(rows[0]['ports'], ['0000:00:03.1'], 'an x16 card sits on a root port')
        self.assertEqual(rows[3]['ceiling_gbps'], 15.754)
        self.assertFalse(rows[3]['iommu_group'], 'no iommu_group link in the fake tree')

    def test_unknown_steps_stay_none(self):
        rows = plan.card_links([0, 1], None)
        self.assertTrue(all(row['width'] is None and row['pci'] is None for row in rows))
        with Workspace() as space:
            rows = plan.card_links([0], {0: 9}, **space.links)
        self.assertIsNone(rows[0]['pci'], 'no sysfs entry for the node')


class PairingTests(unittest.TestCase):
    def test_every_arrangement_pairs_each_x4_with_its_own_x16_by_fewest_hops(self):
        # x16 cards side by side: every route to the far x4 card crosses the middle edge, so the tie on total hops goes to
        # the smaller worst hop (two 2-hop routes, not a 1-hop and a 3-hop one)
        cases = {(16, 16, 4, 4): [(0, 2, 2), (1, 3, 2)],
                 (4, 4, 16, 16): [(2, 0, 2), (3, 1, 2)],
                 (16, 4, 4, 16): [(0, 1, 1), (3, 2, 1)],
                 (4, 16, 16, 4): [(1, 0, 1), (2, 3, 1)],
                 (16, 4, 16, 4): [(0, 1, 1), (2, 3, 1)],
                 (4, 16, 4, 16): [(1, 0, 1), (3, 2, 1)]}
        for widths, want in cases.items():
            with self.subTest(widths=widths):
                got = plan.relay_pairs(widths)
                self.assertEqual(sorted(got, key=lambda pair: pair[1]), sorted(want, key=lambda pair: pair[1]))
                self.assertEqual(len(set(sender for sender, _, _ in got)), 2, 'one receiver per sender')

    def test_no_pairs_without_known_narrow_and_wide_links(self):
        self.assertEqual(plan.relay_pairs([16, 16, 16, 16]), [])
        self.assertEqual(plan.relay_pairs([4, 4, 4, 4]), [])
        self.assertEqual(plan.relay_pairs([16, None, 4, 4]), [])
        self.assertEqual(plan.relay_pairs([]), [])

    def test_one_wide_card_serves_both_narrow_ones(self):
        self.assertEqual(plan.relay_pairs([4, 16, 4, 4]), [(1, 0, 1), (1, 2, 1), (1, 3, 2)])

    def test_hops_are_line_distances(self):
        self.assertEqual(plan.line_hops(0, 3), 3, 'the closing edge is not a route')
        self.assertEqual(plan.line_hops(2, 1), 1)


class ModelTests(unittest.TestCase):
    GB = 1e9

    def test_summary(self):
        record = plan.summarize([0.5, 0.25, 1.0], 1e9)
        self.assertEqual((record['median_s'], record['min_s'], record['gbps_median'], record['gbps_best']), (0.5, 0.25, 2.0, 4.0))
        self.assertEqual(plan.summarize([], 5), dict(n=0, nbytes=5))
        self.assertIsNone(plan.gbps(1, 0))

    def test_direct_is_the_slowest_card_or_the_sum(self):
        cards = [1 * self.GB] * 4
        self.assertAlmostEqual(plan.direct_seconds(cards, [40, 40, 10, 10]), 0.1)
        self.assertAlmostEqual(plan.direct_seconds(cards, [40, 40, 10, 10], overlap=False), 0.25)
        with self.assertRaises(plan.PlanError):
            plan.direct_seconds(cards, [40, 40, 0, 10])

    def test_relay_moves_the_x4_bytes_to_the_senders_and_pipelines_the_fabric(self):
        cards = [1 * self.GB] * 4
        pairs = [(0, 2, 1), (1, 3, 1)]
        # senders carry 2 GB at 40 GB/s = 50 ms; the fabric 1 GB at 50 GB/s = 20 ms plus a 64 MiB fill: host-bound
        relay = plan.relay_seconds(cards, [40, 40, 10, 10], pairs, 50.0)
        self.assertAlmostEqual(relay, 0.05)
        # a slow fabric sets the time: 1 GB at 5 GB/s plus the fill
        slow = plan.relay_seconds(cards, [40, 40, 10, 10], pairs, 5.0)
        self.assertAlmostEqual(slow, 0.2 + (64 << 20) / 5e9)
        per_pair = plan.relay_seconds(cards, [40, 40, 10, 10], pairs, {(0, 2): 50.0, (1, 3): 10.0})
        self.assertAlmostEqual(per_pair, 0.1 + (64 << 20) / 10e9)
        with self.assertRaises(plan.PlanError):
            plan.relay_seconds(cards, [40, 40, 10, 10], pairs, {(0, 2): 50.0})
        with self.assertRaises(plan.PlanError):
            plan.relay_seconds(cards, [40, 40, 10, 10], [(0, 2, 1), (2, 3, 1)], 50.0)

    def test_replicated_dedup(self):
        direct, dedup = plan.replicated_seconds(1 * self.GB, [40, 40, 10, 10], 50.0)
        self.assertAlmostEqual(direct, 0.1)
        self.assertAlmostEqual(dedup, 0.025 + 0.02)

    def test_restore_bytes(self):
        self.assertEqual(plan.KV_BYTES_PER_TOKEN_PER_CHIP, 8704)
        self.assertEqual(plan.restore_bytes_per_card(0), 78446592 // 4)
        self.assertEqual(plan.restore_bytes_per_card(131072), 131072 * 8704 + 19611648)


class DecisionTests(unittest.TestCase):
    def h2d(self, alone, together_s, widths=(16, 16, 4, 4)):
        return dict(widths=list(widths), alone_gbps=list(alone), together_s=together_s, together_bytes=256 << 20)

    def test_link_bound_with_a_fast_fabric_is_relay(self):
        cards = [7e9] * 4
        out = plan.decide(self.h2d([40, 40, 10, 10], 0.03), {(0, 2): 40.0, (1, 3): 40.0}, cards)
        self.assertEqual(out['verdict'], 'RELAY', out)
        self.assertTrue(out['overlap'])
        self.assertLess(out['relay_s'], out['direct_s'])

    def test_x4_as_fast_as_x16_is_host_bound(self):
        out = plan.decide(self.h2d([1.2, 1.2, 1.15, 1.1], 0.9), {(0, 2): 40.0, (1, 3): 40.0}, [7e9] * 4)
        self.assertEqual(out['verdict'], 'HOST-BOUND', out)
        self.assertFalse(out['overlap'], 'four cards together took four times one card')

    def test_a_slow_fabric_is_direct(self):
        out = plan.decide(self.h2d([40, 40, 10, 10], 0.03), {(0, 2): 6.0, (1, 3): 6.0}, [7e9] * 4)
        self.assertEqual(out['verdict'], 'DIRECT', out)

    def test_the_absolute_floor(self):
        out = plan.decide(self.h2d([40, 40, 10, 10], 0.03), {(0, 2): 40.0, (1, 3): 40.0}, [7e6] * 4, min_saving_s=1.0)
        self.assertEqual(out['verdict'], 'DIRECT')

    def test_missing_inputs_are_not_measured(self):
        self.assertEqual(plan.decide(self.h2d([40, None, 10, 10], 0.03), {}, [1] * 4)['verdict'], 'NOT-MEASURED')
        self.assertEqual(plan.decide(self.h2d([40, 40, 10, 10], None), {}, [1] * 4)['verdict'], 'NOT-MEASURED')
        out = plan.decide(self.h2d([40, 40, 10, 10], 0.03), {(0, 2): 40.0}, [1e9] * 4)
        self.assertEqual(out['verdict'], 'NOT-MEASURED')
        self.assertIn('1->3', out['reasons'][0])
        self.assertEqual(plan.decide(self.h2d([40] * 4, 0.03, widths=(16,) * 4), {}, [1] * 4)['verdict'], 'NOT-MEASURED')


# ---------------------------------------------------------------------------------------------------------------------
def h2d_args(output, *extra):
    return ['--output', output, '--sizes-mib', '1', '--tile-sizes-mib', '1', '--together-mib', '1', '--repeats', '1',
            '--warm', '0', '--subset-mib', '1', '--restore-layers', '2'] + list(extra)


def p2p_args(output, *extra):
    return ['--output', output, '--sizes-mib', '1', '--edge-mib', '1', '--repeats', '1', '--warm', '0'] + list(extra)


class H2dProbeTests(unittest.TestCase):
    def run_probe(self, **fake_options):
        with Workspace() as space:
            fake = FakeTtnn(space.descriptor, **fake_options)
            lines = []

            def runner(options, log=print, report=None):
                return h2d.run(options, ttnn=fake, torch=torch, environ={}, log=log, report=report, link_kwargs=space.links)

            status = h2d.main(h2d_args(space.output), runner=runner, log=lines.append)
            with open(space.output) as handle:
                written = json.load(handle)
        return status, written, lines, fake

    def test_a_clean_run_measures_every_arm_exactly(self):
        status, report, lines, fake = self.run_probe()
        self.assertEqual(status, 0, report.get('problems'))
        self.assertEqual(report['verdict'], 'MEASURED')
        arms = report['arms']
        subset = arms['subset']
        self.assertEqual(subset['methods']['view']['status'], 'broadcast', 'the pinned runtime replicates a 1x1 host buffer')
        self.assertEqual(subset['methods']['mapper']['status'], 'exact')
        self.assertEqual((subset['qualified'], subset['method']), (True, 'mapper'))
        self.assertNotIn('exact', subset, 'a broadcast is an answer about the API, not a corrupted transfer')
        self.assertEqual(report['alone_mode'], 'mapper')
        alone = [name for name in arms if name.startswith('alone/')]
        self.assertEqual(len(alone), 3 * 4, 'rm16k, tile_bf16 and tile_bf8 at one size, four positions')
        self.assertTrue(all(arms[name]['exact'] for name in alone))
        self.assertTrue(all(arms[name]['first_s'] is not None for name in alone))
        self.assertEqual(arms['alone/rm16k/1MiB/pos2']['width'], 4)
        self.assertIn('together/rm16k/1MiB', arms)
        self.assertIn('replicated/rm16k/1MiB', arms)
        self.assertTrue(arms['restore']['exact'])
        self.assertEqual(arms['restore']['logical_bytes'], h2d.gdn_checkpoint_bytes(4, 2))
        self.assertEqual(h2d.gdn_checkpoint_bytes(), 78446592, 'the production checkpoint')
        self.assertEqual(report['decide_inputs']['widths'], [16, 16, 4, 4])
        self.assertTrue(all(report['decide_inputs']['alone_gbps']))
        self.assertEqual(report['positions']['pairs'], [[0, 2, 2], [1, 3, 2]])
        self.assertTrue(fake.closed)
        self.assertTrue(lines[-1].startswith('H2D_PROBE verdict=MEASURED'))

    def test_a_mapper_write_that_broadcasts_sends_alone_to_submeshes(self):
        status, report, _, fake = self.run_probe(mapper='broadcast')
        self.assertFalse(report['arms']['subset']['qualified'])
        self.assertEqual(report['alone_mode'], 'submesh')
        self.assertTrue(fake.quiesced, 'the queues are drained around the submesh phase')
        self.assertEqual((status, report['verdict']), (0, 'MEASURED'))

    def test_a_corrupting_one_coordinate_write_is_inexact(self):
        status, report, _, _ = self.run_probe(mapper='corrupt')
        self.assertIn('corrupt', report['arms']['subset']['methods']['mapper']['status'])
        self.assertFalse(report['arms']['subset']['exact'])
        self.assertEqual((status, report['verdict']), (1, 'INEXACT'))

    def test_a_refused_mapper_write_is_not_qualified_but_not_inexact(self):
        status, report, _, _ = self.run_probe(mapper='refuse')
        self.assertEqual(report['arms']['subset']['methods']['mapper']['status'], 'refused')
        self.assertNotIn('exact', report['arms']['subset'])
        self.assertEqual(report['alone_mode'], 'submesh')

    def test_an_exact_view_write_qualifies_when_the_mapper_does_not(self):
        _, report, _, _ = self.run_probe(view='exact', mapper='refuse')
        self.assertEqual(report['arms']['subset']['method'], 'view')
        self.assertEqual(report['alone_mode'], 'submesh', 'alone takes the mapper write only')

    def test_no_per_card_path_at_all(self):
        status, report, _, _ = self.run_probe(refuse_submesh=True, mapper='refuse')
        self.assertIsNone(report['alone_mode'])
        self.assertFalse([name for name in report['arms'] if name.startswith('alone/')])
        self.assertEqual((status, report['verdict']), (2, 'PARTIAL'))

    def test_a_mesh_that_does_not_open_is_not_measured(self):
        status, report, lines, _ = self.run_probe(fail_open=True)
        self.assertEqual((status, report['verdict'], report['opened']), (2, 'NOT-MEASURED', False))
        self.assertIn('No core coordinate', report['error'])

    def test_the_descriptor_is_refused_before_ttnn_is_imported(self):
        options = h2d.build_parser().parse_args(['--output', 'x'])
        data = h2d.run(options, environ={'TT_MESH_GRAPH_DESC_PATH': '/x/p150_x2_mesh_graph_descriptor.textproto'}, log=lambda *a: None)
        self.assertFalse(data['opened'])
        self.assertIn('refused to open', data['error'])
        self.assertEqual(h2d.verdict(data), 'NOT-MEASURED')

    def test_bad_arguments_are_refused(self):
        for extra in (['--sizes-mib', '0'], ['--repeats', '0'], ['--arms', 'alone,bogus'], ['--restore-layers', '49']):
            with self.subTest(extra=extra):
                self.assertEqual(h2d.main(['--output', 'x'] + extra, runner=None, log=lambda *a: None), 2)

    def test_shapes_are_whole_tiles_and_bytes_follow_the_format(self):
        self.assertEqual(h2d.shape_for('rm16k', 1), (64, 8192, 64 * 8192, 1 << 20))
        rows, width, elements, nbytes = h2d.shape_for('tile_bf8', 256)
        self.assertEqual((rows % 32, width), (0, 1024))
        self.assertEqual(nbytes, elements * 1088 // 1024)
        self.assertEqual(h2d.shape_for('tile_bf16', 64)[3], 64 << 20)


class P2pProbeTests(unittest.TestCase):
    def run_probe(self, *extra, **fake_options):
        with Workspace() as space:
            fake = FakeTtnn(space.descriptor, **fake_options)
            lines = []

            def runner(options, log=print, report=None):
                return p2p.run(options, ttnn=fake, torch=torch, environ={}, log=log, report=report, link_kwargs=space.links)

            status = p2p.main(p2p_args(space.output, *extra), runner=runner, log=lines.append)
            with open(space.output) as handle:
                written = json.load(handle)
        return status, written, lines, fake

    def test_a_clean_run_times_every_edge_the_pairs_and_both_at_once(self):
        status, report, lines, fake = self.run_probe()
        self.assertEqual((status, report['verdict']), (0, 'MEASURED'), report.get('problems'))
        arms = report['arms']
        self.assertEqual(report['pairs_used'], [[0, 2], [1, 3]])
        for sender, receiver in p2p.LINE_EDGES:
            self.assertTrue(arms['edge/rm16k/1MiB/%d->%d' % (sender, receiver)]['exact'])
        self.assertEqual(arms['far/rm16k/1MiB/0->3']['hops'], [3])
        self.assertEqual(arms['local/rm16k/1MiB/0->0']['hops'], [0])
        self.assertEqual(arms['both/rm16k/1MiB']['transfers'], [[0, 2], [1, 3]])
        self.assertEqual(arms['pair/tile_bf8/1MiB/0->2']['widths'], [[16, 4]])
        self.assertEqual(arms['pair/rm16k/1MiB/1->3']['hops'], [2])
        self.assertIn('reverse/rm16k/1MiB/2->0', arms)
        self.assertEqual(sorted(report['decide_inputs']), ['0->2', '1->3'])
        self.assertIn(('p2p', 0, 2), fake.calls)
        relay = arms['relay/rm16k/1MiB/all']
        self.assertTrue(relay['exact'])
        self.assertIn('direct_sharded', relay, 'all four positions: the production sharded write is the baseline too')
        self.assertTrue(relay['relay_over_direct'])
        self.assertTrue(arms['relay/tile_bf8/1MiB/0->2']['exact'])
        self.assertNotIn('direct_sharded', arms['relay/rm16k/1MiB/0->2'])
        self.assertEqual(sorted(report['relay_ab']), sorted(name for name in arms if name.startswith('relay/')))
        for connections in (1, 2):
            self.assertTrue(arms['socket/rm16k/1MiB/1->3/%dconn' % connections]['exact'])
        self.assertIn('send', fake.calls)
        self.assertTrue(lines[-1].startswith('P2P_PROBE verdict=MEASURED'))

    def test_a_refused_socket_is_partial_and_keeps_the_fabric_arms(self):
        status, report, _, _ = self.run_probe(refuse_socket=True)
        self.assertEqual((status, report['verdict']), (2, 'PARTIAL'))
        self.assertIn('socket refused', report['arms']['socket/rm16k/1MiB/0->2/1conn']['error'])
        self.assertTrue(report['arms']['relay/rm16k/1MiB/all']['exact'])

    def test_the_relay_arms_wait_for_a_one_coordinate_host_write(self):
        # a 'one-coordinate' write that lands on every card would read as a corrupted relay: the arms are skipped instead
        status, report, _, _ = self.run_probe(mapper='broadcast')
        self.assertEqual(report['mapper_write']['status'], 'broadcast')
        self.assertFalse([name for name in report['arms'] if name.startswith('relay/')])
        self.assertTrue(any('relay arms skipped' in problem for problem in report['problems']))
        self.assertEqual((status, report['verdict']), (2, 'PARTIAL'))
        self.assertTrue(report['arms']['edge/rm16k/1MiB/0->1']['exact'], 'the fabric arms do not depend on it')
        status, report, _, _ = self.run_probe(mapper='corrupt')
        self.assertEqual((status, report['verdict']), (1, 'INEXACT'))

    def test_a_corrupting_transfer_is_inexact(self):
        status, report, _, _ = self.run_probe(corrupt_p2p=True)
        self.assertEqual((status, report['verdict']), (1, 'INEXACT'))
        self.assertTrue(report['arms']['local/rm16k/1MiB/0->0']['exact'], 'the local copy is not a fabric transfer')
        self.assertFalse(report['arms']['edge/rm16k/1MiB/0->1']['exact'])

    def test_unknown_widths_fall_back_to_one_hop_pairs(self):
        self.assertEqual(p2p.fallback_pairs({'pairs': []}), ([(1, 0), (2, 3)], True))
        self.assertEqual(p2p.fallback_pairs({'pairs': [(1, 2, 1)]}), ([(1, 2)], False))

    def test_decide_inputs_prefer_the_concurrent_rate(self):
        data = {'arms': {'both/rm16k/256MiB': {'transfers': [[1, 2], [0, 3]], 'per_transfer_gbps_median': 30.0},
                         'pair/rm16k/256MiB/1->2': {'mib': 256, 'per_transfer_gbps_median': 45.0}}}
        self.assertEqual(p2p.decide_inputs(data), {'1->2': 30.0, '0->3': 30.0})
        del data['arms']['both/rm16k/256MiB']
        data['arms']['pair/rm16k/4MiB/1->2'] = {'mib': 4, 'per_transfer_gbps_median': 5.0}
        self.assertEqual(p2p.decide_inputs(data), {'1->2': 45.0})
        data['arms']['socket/rm16k/256MiB/1->2/2conn'] = {'exact': True, 'gbps_median': 60.0}
        data['arms']['socket/rm16k/256MiB/1->2/1conn'] = {'exact': False, 'gbps_median': 90.0}
        self.assertEqual(p2p.decide_inputs(data), {'1->2': 60.0}, 'a faster exact socket replaces the op; an inexact one never')


class CommonTests(unittest.TestCase):
    def test_the_pattern_is_exact_small_integers_with_a_prime_period(self):
        values = common.pattern(torch, 3 * common.PATTERN_PERIOD // 2, offset=5)
        self.assertEqual(values.dtype, torch.bfloat16)
        self.assertLess(float(values.float().abs().max()), 128)
        self.assertTrue(torch.equal(values.float(), values.float().round()))
        self.assertFalse(torch.equal(common.pattern(torch, 4096, 0), common.pattern(torch, 4096, 1)))

    def test_the_report_is_rewritten_whole_after_each_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'r.json')
            report = common.Report(path, 'k')
            report.arm('a', {'x': 1})
            with open(path) as handle:
                self.assertEqual(json.load(handle)['arms'], {'a': {'x': 1}})
            report.problem('p')
            with open(path) as handle:
                self.assertEqual(json.load(handle)['problems'], ['p'])
            self.assertFalse(os.path.exists(path + '.part'))

    def test_the_modules_import_no_device_library_at_import(self):
        for module in (h2d, p2p, common, plan):
            with open(module.__file__, encoding='utf-8') as handle:
                head = handle.read().split('\ndef run(')[0]
            for banned in ('\nimport ttnn', '\nimport torch'):
                self.assertNotIn(banned, head, module.__name__)


class WiringTests(unittest.TestCase):
    def test_the_job_parser_and_the_workflow_know_both_probes(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            workflow = handle.read()
        for probe, script, report in (('h2d', 'tp4_h2d_probe.py', 'h2d-probe.json'), ('p2p', 'tp4_p2p_probe.py', 'p2p-probe.json')):
            self.assertIn(probe, job.FABRIC_PROBES)
            self.assertIn('%s) script=%s; report=%s' % (probe, script, report), workflow)
            self.assertTrue(os.path.isfile(os.path.join(HERE, script)))
        self.assertIn('H2D_PROBE|P2P_PROBE', workflow)

    def test_the_cpu_suite_runs_this_module(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            workflow = handle.read()
        self.assertIn(' test_fabric_upload', workflow)


class PackTests(unittest.TestCase):
    def profiles(self):
        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            return sorted(json.load(handle)['profiles'])

    def order(self):
        with open(os.path.join(PACK, 'ORDER.txt'), encoding='utf-8') as handle:
            return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]

    def parsed(self, name):
        with open(os.path.join(PACK, name + '.env'), encoding='utf-8') as handle:
            return job.read_job(job.parse_env(handle.read()), self.profiles(), root=ROOT)

    def test_every_template_has_an_order_row_parses_and_runs_one_probe_alone(self):
        names = sorted(name[:-4] for name in os.listdir(PACK) if name.endswith('.env'))
        self.assertEqual(names, sorted(row[0] for row in self.order()))
        probes = {}
        for name in names:
            with self.subTest(name=name):
                outputs = self.parsed(name)
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['actions'], 'reset fabric')
                probes[name] = outputs['fabric_probe']
        self.assertEqual(sorted(probes.values()), ['h2d', 'p2p'])

    def test_the_templates_name_no_rig_card_host_or_registry(self):
        for name in os.listdir(PACK):
            with open(os.path.join(PACK, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(re.search(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', text), '%s carries an IP address' % name)
            for banned in (':5000', 'blackhole-', '/home/', '.local'):
                self.assertNotIn(banned, text, '%s names %s' % (name, banned))


if __name__ == '__main__':
    unittest.main()
