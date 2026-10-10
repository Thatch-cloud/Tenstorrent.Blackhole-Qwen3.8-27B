"""Fabric unicast bandwidth x16 -> x4 on the served (1, 4) mesh: job F2 of the fabric-upload pack (docs/tp4-fabric-upload.md,
scripts/ci/references/fabric-upload-jobs). qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=p2p, C2_CARDS=quad:

    python3 -B /c2/scripts/ci/tp4_p2p_probe.py --fabric FABRIC_1D --output /probe-results/p2p-probe.json

WHY. The relay moves an x4 card's bytes from an x16 card's DRAM to the x4 card's DRAM over the ethernet fabric. The pinned
runtime offers two ops for it:
  ttnn.point_to_point (ttnn/cpp/ttnn/operations/point_to_point): interleaved DRAM or L1, TILE or ROW_MAJOR, the sender's
      shard of a mesh tensor written into the receiver's shard of a same-spec output, in place, through an input-sized
      intermediate. ONE worker core and link 0 only (send/receive program factories), so one link's rate at best.
  ttnn.experimental.send_async / recv_async over a MeshSocket (ttnn.create_socket_pair between two submeshes): one core
      per socket connection, the connections spread over the available links.
The only fabric numbers on record are collectives (the TP2 all-gather, 83.74-90.43 GB/s counting both directions of
each card on two links); nobody has timed a unicast, its first call (programs, a global semaphore, a mesh-wide
barrier), a multi-hop route, two pairs at once, or the relay end to end against a direct write. This job does, and
checks every transfer byte for byte.

ARMS (the report is rewritten after each; a failed arm is a problem line and the next arm still runs):
  links     as the h2d probe: positions, chip ids, PCIe widths, the trained ethernet links and the relay pairs
            (fabric_upload_plan.relay_pairs: every x4 position gets an x16 sender, fewest line hops first)
  local     point_to_point with sender == receiver (an on-card DRAM -> DRAM copy, no fabric): the op's own ceiling
  edges     point_to_point on every directed edge of the line (0->1, 1->0, 1->2, 2->1, 2->3, 3->2) at --edge-mib, rm16k
  far       0 -> 3, three hops (the ring's closing edge is not a route: FABRIC_1D and the op route along the line)
  pairs     each relay pair alone at --sizes-mib in rm16k (16 KiB pages), tile_bf8 (1,088 B pages) and tile_bf16
  both      the relay pairs at once (both ops issued, one synchronize) at the largest size, rm16k and tile_bf8
  reverse   each pair reversed (x4 -> x16, the spill direction) at --edge-mib
  relay     THE A/B: the relay end to end against a direct write of the same bytes, at --edge-mib, per pair and for all
            pairs at once. relay = each sender's own shard written to the destination at its coordinate and its
            receiver's shard written to a staging tensor at the sender's coordinate (one-coordinate mapper writes:
            no byte crosses a receiver's PCIe), then point_to_point(staging -> destination) per pair, one synchronize.
            direct = the same bytes written to every coordinate over its own PCIe (one-coordinate writes; for all
            pairs also the production path, one ShardTensorToMesh write). exact = every destination shard holds its
            bytes and the others their sentinel
  socket    send_async/recv_async between (1, 1) submeshes for each pair at --edge-mib, one and two socket connections
            (DRAM FIFO): the multi-link alternative to point_to_point. Last, because it switches to submeshes (their
            own allocators) and a socket hang ends at the watchdog
Every transfer: the first call timed apart (first_call_s: programs, semaphore, cross-device barrier), then --warm and
--repeats calls; exact = the receiver's shard equals the sender's source and every other shard keeps its sentinel.

VERDICT (last line 'P2P_PROBE verdict=...'): MEASURED (exit 0), INEXACT (exit 1), PARTIAL or NOT-MEASURED (exit 2),
watchdog 3; as the h2d probe.
"""

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fabric_upload_plan as plan  # noqa: E402
import tp4_h2d_probe as h2d  # noqa: E402
import tp4_mesh  # noqa: E402
import tp4_upload_probe_common as common  # noqa: E402

KIND = 'p2p-probe-quad'
TAG = 'P2P_PROBE'
SIZES_MIB = (4, 64, 256)
EDGE_MIB = 256
REPEATS, WARM = 5, 1
LINE_EDGES = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2))
FAR = (0, 3)
STATUS = {'MEASURED': 0, 'INEXACT': 1}
ARMS = ('local', 'edges', 'far', 'pairs', 'both', 'reverse', 'relay', 'socket')
SOCKET_FIFO_KIB = 64
SOCKET_CONNECTIONS = (1, 2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes-mib', default=','.join(str(size) for size in SIZES_MIB))
    parser.add_argument('--edge-mib', type=int, default=EDGE_MIB)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--warm', type=int, default=WARM)
    parser.add_argument('--arms', default='local,edges,far,pairs,both,reverse,relay,socket')
    parser.add_argument('--socket-fifo-kib', type=int, default=SOCKET_FIFO_KIB)
    return parser


verdict = h2d.verdict


def fallback_pairs(found):
    """The relay pairs when the widths are known; otherwise (1 -> 0, 2 -> 3), labelled UNKNOWN-WIDTH, so the arms still
    measure a 1-hop unicast in each half of the line."""
    pairs = found.get('pairs') or []
    if pairs:
        return [tuple(pair[:2]) for pair in pairs], False
    return [(1, 0), (2, 3)], True


class Probe(h2d.Probe):
    def prepared(self, name, mib):
        """(rows, width, nbytes, source, sources): a mesh tensor whose shard p holds pattern p, uploaded (untimed)."""
        ttnn = self.ttnn
        rows, width, elements, nbytes = h2d.shape_for(name, mib)
        host = self.host(name, rows, width, 0, mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), count=self.positions)
        source = self.allocate(name, rows, width, self.mesh)
        ttnn.copy_host_to_device_tensor(host, source)
        ttnn.synchronize_device(self.mesh)
        return rows, width, nbytes, source, self.shards(host)

    def output(self, name, rows, width):
        """An output tensor prefilled with a sentinel per shard (offset 50 + p), and those sentinels as torch tensors."""
        ttnn = self.ttnn
        host = self.host(name, rows, width, 50, mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), count=self.positions)
        out = self.allocate(name, rows, width, self.mesh)
        ttnn.copy_host_to_device_tensor(host, out)
        ttnn.synchronize_device(self.mesh)
        return out, self.shards(host)

    def intermediate(self, source, sender, receiver):
        ttnn = self.ttnn
        try:
            spec = ttnn.p2p_compute_intermediate_tensor_spec(source, self.coord(sender), self.coord(receiver),
                                                             ttnn.Topology.Linear)
            return ttnn.allocate_tensor_on_device(spec, self.mesh)
        except Exception as error:  # noqa: BLE001
            self.report.problem('intermediate %d->%d not preallocated (%s): the op allocates its own each call'
                                % (sender, receiver, common.error_text(error)))
            return None

    def call(self, source, sender, receiver, out, inter):
        ttnn = self.ttnn
        kwargs = dict(output_tensor=out, topology=ttnn.Topology.Linear)
        if inter is not None:
            kwargs['intermediate_tensor'] = inter
        return ttnn.point_to_point(source, self.coord(sender), self.coord(receiver), **kwargs)

    def exact(self, out, transfers, sentinels, sources, rows, width):
        """Every receiver's shard holds its sender's source and every other shard its sentinel."""
        got = self.shards(self.ttnn.from_device(out))
        want = list(sentinels)
        for sender, receiver in transfers:
            want[receiver] = sources[sender]
        return len(got) == len(want) and all(self.torch.equal(a.reshape(rows, width), b.reshape(rows, width))
                                             for a, b in zip(got, want))

    def transfer(self, arm, name, mib, transfers, hops=None):
        """One timed transfer set (one pair, or several issued together and synchronized once)."""
        ttnn = self.ttnn
        entry = dict(format=name, mib=mib, transfers=[list(pair) for pair in transfers],
                     hops=[plan.line_hops(s, r) for s, r in transfers])
        cards = (self.report.data.get('positions') or {}).get('cards') or []
        if cards:
            entry['widths'] = [[cards[s].get('width'), cards[r].get('width')] for s, r in transfers]
        held = []
        try:
            rows, width, nbytes, source, sources = self.prepared(name, mib)
            held.append(source)
            outs = []
            for sender, receiver in transfers:
                out, sentinels = self.output(name, rows, width)
                inter = self.intermediate(source, sender, receiver) if sender != receiver else None
                held.extend([out] + ([inter] if inter is not None else []))
                outs.append((sender, receiver, out, inter, sentinels))

            def issue():
                for sender, receiver, out, inter, _ in outs:
                    self.call(source, sender, receiver, out, inter)

            entry['first_call_s'] = round(common.timed(issue, lambda: ttnn.synchronize_device(self.mesh)), 6)
            samples = []
            for index in range(self.options.warm + self.options.repeats):
                seconds = common.timed(issue, lambda: ttnn.synchronize_device(self.mesh))
                if index >= self.options.warm:
                    samples.append(seconds)
            entry.update(plan.summarize(samples, nbytes * len(transfers)))
            entry['nbytes_per_transfer'] = nbytes
            entry['per_transfer_gbps_median'] = plan.gbps(nbytes, entry.get('median_s'))
            entry['exact'] = all(self.exact(out, [(sender, receiver)], sentinels, sources, rows, width)
                                 for sender, receiver, out, _, sentinels in outs)
        except Exception as error:  # noqa: BLE001
            entry['error'] = common.error_text(error)
        finally:
            for tensor in held:
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # noqa: BLE001
                    pass
        self.report.arm(arm, entry)
        self.log('%s arm=%s transfers=%s hops=%s first_call_s=%s gbps_median=%s per_transfer_gbps=%s exact=%s%s' % (
            TAG, arm, entry['transfers'], entry['hops'], entry.get('first_call_s'), entry.get('gbps_median'),
            entry.get('per_transfer_gbps_median'), entry.get('exact'), (' error=' + entry['error']) if entry.get('error') else ''))
        return entry

    def relay(self, arm, name, mib, pairs, sharded_direct=False):
        """The relay end to end against a direct write of the same bytes (module docstring, arm relay)."""
        ttnn, torch = self.ttnn, self.torch
        rows, width, elements, nbytes = h2d.shape_for(name, mib)
        senders = [sender for sender, _ in pairs]
        receivers = [receiver for _, receiver in pairs]
        entry = dict(format=name, mib=mib, pairs=[list(pair) for pair in pairs], nbytes_per_card=nbytes,
                     hops=[plan.line_hops(s, r) for s, r in pairs])
        held = []
        try:
            dest, sentinels = self.output(name, rows, width)
            staging = self.allocate(name, rows, width, self.mesh)
            held += [dest, staging]
            inters = {}
            for sender, receiver in pairs:
                inters[(sender, receiver)] = self.intermediate(staging, sender, receiver)
                if inters[(sender, receiver)] is not None:
                    held.append(inters[(sender, receiver)])
            own = dict((pos, self.one_coordinate_host(name, rows, width, 400 + pos, pos)) for pos in senders + receivers)
            share = dict((receiver, self.one_coordinate_host(name, rows, width, 400 + receiver, sender)) for sender, receiver in pairs)

            def relay_once():
                for sender, receiver in pairs:
                    ttnn.copy_host_to_device_tensor(own[sender], ttnn.get_device_tensors(dest)[sender])
                    ttnn.copy_host_to_device_tensor(share[receiver], ttnn.get_device_tensors(staging)[sender])
                for sender, receiver in pairs:
                    self.call(staging, sender, receiver, dest, inters[(sender, receiver)])

            def direct_once():
                for pos in senders + receivers:
                    ttnn.copy_host_to_device_tensor(own[pos], ttnn.get_device_tensors(dest)[pos])

            sync = lambda: ttnn.synchronize_device(self.mesh)  # noqa: E731
            entry['relay_first_s'] = round(common.timed(relay_once, sync), 6)
            want = list(sentinels)
            for pos in senders + receivers:
                want[pos] = common.pattern(torch, elements, 400 + pos).reshape(rows, width)
            got = [ttnn.to_torch(ttnn.from_device(view)).reshape(rows, width) for view in ttnn.get_device_tensors(dest)]
            entry['exact'] = len(got) == len(want) and all(torch.equal(a, b.reshape(rows, width)) for a, b in zip(got, want))
            for label, once in (('relay', relay_once), ('direct', direct_once)):
                samples = []
                for index in range(self.options.warm + self.options.repeats):
                    seconds = common.timed(once, sync)
                    if index >= self.options.warm:
                        samples.append(seconds)
                entry[label] = plan.summarize(samples, nbytes * len(set(senders + receivers)))
            if sharded_direct and len(set(senders + receivers)) == self.positions:
                host = self.host(name, rows, width, 400, mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), count=self.positions)
                samples = []
                for index in range(self.options.warm + self.options.repeats):
                    seconds = common.timed(lambda: ttnn.copy_host_to_device_tensor(host, dest), sync)
                    if index >= self.options.warm:
                        samples.append(seconds)
                entry['direct_sharded'] = plan.summarize(samples, nbytes * self.positions)
            relay_s = (entry.get('relay') or {}).get('median_s')
            direct_s = min(value for value in ((entry.get('direct') or {}).get('median_s'),
                                               (entry.get('direct_sharded') or {}).get('median_s')) if value) \
                if (entry.get('direct') or {}).get('median_s') else None
            entry['relay_over_direct'] = round(relay_s / direct_s, 4) if relay_s and direct_s else None
        except Exception as error:  # noqa: BLE001
            entry['error'] = common.error_text(error)
        finally:
            for tensor in held:
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # noqa: BLE001
                    pass
        self.report.arm(arm, entry)
        self.log('%s arm=%s pairs=%s relay_s=%s direct_s=%s direct_sharded_s=%s relay_over_direct=%s exact=%s%s' % (
            TAG, arm, entry['pairs'], (entry.get('relay') or {}).get('median_s'), (entry.get('direct') or {}).get('median_s'),
            (entry.get('direct_sharded') or {}).get('median_s'), entry.get('relay_over_direct'), entry.get('exact'),
            (' error=' + entry['error']) if entry.get('error') else ''))
        return entry

    def socket(self, arm, name, mib, sender, receiver, connections):
        """send_async/recv_async between (1, 1) submeshes over `connections` socket connections (DRAM FIFO)."""
        ttnn, torch = self.ttnn, self.torch
        rows, width, elements, nbytes = h2d.shape_for(name, mib)
        entry = dict(format=name, mib=mib, transfers=[[sender, receiver]], hops=[plan.line_hops(sender, receiver)],
                     connections=connections, fifo_kib=self.options.socket_fifo_kib)
        held = []
        try:
            if hasattr(self.mesh, 'quiesce_devices'):
                self.mesh.quiesce_devices()
            send_dev = self.mesh.create_submesh(ttnn.MeshShape(1, 1), self.coord(sender))
            recv_dev = self.mesh.create_submesh(ttnn.MeshShape(1, 1), self.coord(receiver))
            self.submeshes = list(getattr(self, 'submeshes', ()) or ()) + [send_dev, recv_dev]
            origin = ttnn.MeshCoordinate(0, 0)
            links = [ttnn.SocketConnection(ttnn.MeshCoreCoord(origin, ttnn.CoreCoord(0, index)),
                                           ttnn.MeshCoreCoord(origin, ttnn.CoreCoord(0, index))) for index in range(connections)]
            config = ttnn.SocketConfig(links, ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, self.options.socket_fifo_kib * 1024))
            send_socket, recv_socket = ttnn.create_socket_pair(send_dev, recv_dev, config)
            host = self.host(name, rows, width, 500 + sender)
            source = self.allocate(name, rows, width, send_dev)
            out = self.allocate(name, rows, width, recv_dev)
            held += [source, out]
            ttnn.copy_host_to_device_tensor(host, source)
            ttnn.synchronize_device(send_dev)

            def once():
                ttnn.experimental.send_async(source, send_socket)
                ttnn.experimental.recv_async(out, recv_socket)

            def sync():
                ttnn.synchronize_device(send_dev)
                ttnn.synchronize_device(recv_dev)

            entry['first_call_s'] = round(common.timed(once, sync), 6)
            back = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(recv_dev, dim=0)).reshape(rows, width)
            expect = ttnn.to_torch(host).reshape(rows, width)
            entry['exact'] = bool(torch.equal(back, expect))
            samples = []
            for index in range(self.options.warm + self.options.repeats):
                seconds = common.timed(once, sync)
                if index >= self.options.warm:
                    samples.append(seconds)
            entry.update(plan.summarize(samples, nbytes))
            entry['per_transfer_gbps_median'] = entry.get('gbps_median')
        except Exception as error:  # noqa: BLE001
            entry['error'] = common.error_text(error)
        finally:
            for tensor in held:
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(self.mesh, 'quiesce_devices'):
                try:
                    self.mesh.quiesce_devices()
                except Exception:  # noqa: BLE001
                    pass
        self.report.arm(arm, entry)
        self.log('%s arm=%s connections=%d first_call_s=%s gbps_median=%s exact=%s%s' % (
            TAG, arm, connections, entry.get('first_call_s'), entry.get('gbps_median'), entry.get('exact'),
            (' error=' + entry['error']) if entry.get('error') else ''))
        return entry

    def fabric_facts(self):
        ttnn = self.ttnn
        facts = {}
        for name in ('get_tt_fabric_max_payload_size_bytes', 'get_tt_fabric_channel_buffer_size_bytes'):
            for owner in (ttnn, getattr(ttnn, 'fabric', None)):
                function = getattr(owner, name, None) if owner is not None else None
                if callable(function):
                    try:
                        facts[name] = int(function())
                    except Exception as error:  # noqa: BLE001
                        facts[name] = common.error_text(error)
                    break
        self.report.data['fabric_facts'] = facts
        self.report.save()
        return facts


def run(options, ttnn=None, torch=None, environ=None, log=print, report=None, link_kwargs=None):
    environ = os.environ if environ is None else environ
    report = report or common.Report(None, KIND)
    report.data.update(fabric=options.fabric, opened=False, serving_contract=environ.get('QWEN_C2_SERVING'),
                       environment=h2d.environment(environ))
    if ttnn is None:
        refusal, descriptor, problems = common.descriptor_refusal(environ)
        report.data.update(descriptor=descriptor, descriptor_problems=problems)
        if refusal:
            report.data['error'] = 'refused to open: ' + refusal
            report.save()
            return report.data
        import torch  # noqa: F811
        import ttnn  # noqa: F811
    arms = [arm for arm in options.arms.split(',') if arm]
    mesh = None
    probe = None
    try:
        mesh = common.open_mesh(ttnn, options.fabric)
        report.data['opened'] = True
        report.save()
        probe = Probe(ttnn, torch, mesh, report, options, log)
        probe.submeshes = ()
        found = probe.links(**(link_kwargs or {}))
        probe.fabric_facts()
        pairs, unknown = fallback_pairs(found)
        report.data['pairs_used'] = [list(pair) for pair in pairs]
        report.data['pairs_unknown_width'] = unknown
        largest = max(h2d.sizes(options.sizes_mib))
        if 'local' in arms:
            probe.transfer('local/rm16k/%dMiB/0->0' % options.edge_mib, 'rm16k', options.edge_mib, [(0, 0)])
        if 'edges' in arms:
            for sender, receiver in LINE_EDGES:
                probe.transfer('edge/rm16k/%dMiB/%d->%d' % (options.edge_mib, sender, receiver), 'rm16k', options.edge_mib,
                               [(sender, receiver)])
        if 'far' in arms:
            probe.transfer('far/rm16k/%dMiB/%d->%d' % ((options.edge_mib,) + FAR), 'rm16k', options.edge_mib, [FAR])
        if 'pairs' in arms:
            for name in ('rm16k', 'tile_bf8', 'tile_bf16'):
                for mib in h2d.sizes(options.sizes_mib):
                    for sender, receiver in pairs:
                        probe.transfer('pair/%s/%dMiB/%d->%d' % (name, mib, sender, receiver), name, mib, [(sender, receiver)])
        if 'both' in arms and len(pairs) > 1:
            for name in ('rm16k', 'tile_bf8'):
                probe.transfer('both/%s/%dMiB' % (name, largest), name, largest, pairs)
        if 'reverse' in arms:
            for sender, receiver in pairs:
                probe.transfer('reverse/rm16k/%dMiB/%d->%d' % (options.edge_mib, receiver, sender), 'rm16k', options.edge_mib,
                               [(receiver, sender)])
        relay_ok = False
        if 'relay' in arms:
            # the relay's host leg must write one coordinate only (the h2d probe's subset arm, mapper method, at 1 MiB):
            # a write that lands elsewhere would make every relay arm read as a corrupted transfer
            rows, width, elements, nbytes = h2d.shape_for('rm16k', 1)
            check = common.guarded(report, 'mapper-write', lambda: probe.subset_method('mapper', rows, width, elements, nbytes), {})
            report.data['mapper_write'] = check
            relay_ok = bool(check.get('qualified'))
            if 'corrupt' in str(check.get('status')):
                report.arm('mapper-write', dict(check, exact=False))
            elif not relay_ok:
                report.problem('relay arms skipped: the one-coordinate host write is %s, not exact' % check.get('status'))
        if relay_ok:
            for name in ('rm16k', 'tile_bf8'):
                for sender, receiver in pairs:
                    probe.relay('relay/%s/%dMiB/%d->%d' % (name, options.edge_mib, sender, receiver), name, options.edge_mib,
                                [(sender, receiver)])
                if len(pairs) > 1:
                    probe.relay('relay/%s/%dMiB/all' % (name, options.edge_mib), name, options.edge_mib, pairs,
                                sharded_direct=True)
        if 'socket' in arms:
            for sender, receiver in pairs:
                for connections in SOCKET_CONNECTIONS:
                    probe.socket('socket/rm16k/%dMiB/%d->%d/%dconn' % (options.edge_mib, sender, receiver, connections), 'rm16k',
                                 options.edge_mib, sender, receiver, connections)
    except Exception as error:  # noqa: BLE001
        report.data.update(error=common.error_text(error), traceback=traceback.format_exc()[-3000:])
        report.problem('stopped: ' + report.data['error'])
    finally:
        if mesh is not None:
            report.data['closed'] = common.close_mesh(ttnn, mesh, getattr(probe, 'submeshes', ()) if probe else ())
    report.data['decide_inputs'] = decide_inputs(report.data)
    report.data['relay_ab'] = relay_ab(report.data)
    report.save()
    return report.data


def decide_inputs(data):
    """{'S->R': GB/s} per relay pair for fabric_upload_plan.decide: the both-pairs-at-once point_to_point rate at the largest
    size when it ran (each pair's share of the aggregate), else the pair alone at its largest size; then a socket arm's rate
    for the pair when it ran exact and is higher (the multi-link path the relay would use instead)."""
    arms = data.get('arms') or {}
    rates = {}
    both = [record for name, record in arms.items() if name.startswith('both/rm16k/') and record.get('per_transfer_gbps_median')]
    if both:
        for sender, receiver in both[0].get('transfers') or []:
            rates['%d->%d' % (sender, receiver)] = both[0]['per_transfer_gbps_median']
    else:
        sized = {}
        for name, record in arms.items():
            if name.startswith('pair/rm16k/') and record.get('per_transfer_gbps_median'):
                key = name.rsplit('/', 1)[1]
                if key not in sized or record.get('mib', 0) >= sized[key][0]:
                    sized[key] = (record.get('mib', 0), record['per_transfer_gbps_median'])
        rates = dict((key, value[1]) for key, value in sized.items())
    for name, record in arms.items():
        if name.startswith('socket/') and record.get('exact') and record.get('gbps_median'):
            key = name.split('/')[3]
            if record['gbps_median'] > rates.get(key, 0):
                rates[key] = record['gbps_median']
    return rates


def relay_ab(data):
    """{arm: relay_over_direct} for the relay arms that ran exact: under 1.0 the relay was faster than writing the same bytes
    over every card's own PCIe (the all-pairs arm against the better of the per-coordinate and the sharded direct write)."""
    return dict((name, record.get('relay_over_direct')) for name, record in (data.get('arms') or {}).items()
                if name.startswith('relay/') and record.get('exact') and record.get('relay_over_direct'))


def main(argv=None, runner=run, log=print):
    options = build_parser().parse_args(argv)
    try:
        h2d.sizes(options.sizes_mib)
        if options.repeats < 1 or options.warm < 0 or not 1 <= options.edge_mib <= 2048:
            raise ValueError('repeats >= 1, warm >= 0, edge-mib in 1..2048')
        if not 1 <= options.socket_fifo_kib <= 1024:
            raise ValueError('socket-fifo-kib in 1..1024')
        unknown = sorted(set(arm for arm in options.arms.split(',') if arm) - set(ARMS))
        if unknown:
            raise ValueError('unknown arms %s (known: %s)' % (', '.join(unknown), ', '.join(ARMS)))
    except ValueError as error:
        print('refusing: %s' % error, file=sys.stderr)
        return 2
    timer = common.start_watchdog(TAG, log=log)
    report = common.Report(options.output, KIND)
    data = runner(options, log=log, report=report)
    data['verdict'] = verdict(data)
    report.data = data
    report.save()
    log('%s verdict=%s opened=%s pairs=%s unknown_width=%s rates=%s relay_over_direct=%s problems=%d%s' % (
        TAG, data['verdict'], data.get('opened'), data.get('pairs_used'), data.get('pairs_unknown_width'),
        data.get('decide_inputs'), data.get('relay_ab'), len(data.get('problems') or []),
        (' error=' + data['error']) if data.get('error') else ''))
    timer.cancel()
    return STATUS.get(data['verdict'], 2)


if __name__ == '__main__':
    sys.exit(main())
