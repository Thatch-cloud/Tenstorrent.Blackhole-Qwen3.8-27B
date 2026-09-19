"""How many ethernet links actually connect the two cards, and what caps them.

The bandwidth probe (35424379930) measured 83.74 GB/s, which is 84% of one
800 Gb/s port, and sweeping num_links across 1/2/4 changed nothing. Two very
different explanations fit that equally well:

  one cable is connected      -> both its links are already in use; enabling more
                                 means cabling another port, not a config change
  more cables are connected   -> a software cap is why only one pair is used, and
                                 the rest are available now

One QSFP-DD800 cable carries TWO ethernet links, 400 Gb/s each, so a cable is
100 GB/s. The measured 83.74 GB/s is 84% of exactly that, which is what one
fully-used cable looks like - but a half-used pair of cables would look similar,
so the bandwidth number alone cannot separate the two.

An earlier reading of `intra-mesh degree histograms mesh0 {1:2}` claimed the
first. That was wrong: degree counts neighbours, not edges, and with two chips
in the mesh each has exactly one neighbour however many cables run between them.
The log line is uninformative and the question was never settled.

This probe settles it by asking the runtime which ethernet cores are actually
linked, rather than inferring from a bandwidth number. Everything is guarded and
introspected, because a wrong API guess should cost a line of output rather than
the hardware slot: unknown attributes are reported, not raised.
"""

import argparse
import io
import json
import traceback


def describe(value):
    """Render an API result without assuming its type."""
    if value is None:
        return None
    try:
        return sorted(str(item) for item in value)
    except TypeError:
        return str(value)


def probe_object(target, names):
    """Call each zero-argument accessor that exists, recording what happened."""
    found = {}
    for name in names:
        accessor = getattr(target, name, None)
        if accessor is None:
            found[name] = 'ABSENT'
            continue
        try:
            found[name] = describe(accessor())
        except BaseException as error:
            found[name] = '%s: %s' % (type(error).__name__, str(error)[:160])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'scope': __doc__}
    try:
        import ttnn

        # Surface the vocabulary before using it, so that if the accessor names
        # below are wrong the run still says what the right ones are.
        report['ttnn_eth_symbols'] = sorted(
            name for name in dir(ttnn)
            if 'eth' in name.lower() or 'fabric' in name.lower() or 'cluster' in name.lower())

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
        try:
            report['mesh_shape'] = str(mesh.shape)
            devices = list(mesh.get_devices())
            report['device_count'] = len(devices)

            report['device_symbols'] = sorted(
                name for name in dir(devices[0])
                if 'eth' in name.lower() or 'core' in name.lower())

            per_device = []
            for device in devices:
                entry = {'id': device.id()}
                entry.update(probe_object(device, (
                    'get_active_ethernet_cores',
                    'get_inactive_ethernet_cores',
                    'get_ethernet_cores_grouped_by_connected_chip_ids')))
                # Links to the peer specifically. On a two-card mesh this is the
                # number that decides the question.
                for peer in devices:
                    if peer.id() == device.id():
                        continue
                    sockets = getattr(device, 'get_ethernet_sockets', None)
                    if sockets is None:
                        entry['sockets_to_%d' % peer.id()] = 'ABSENT'
                        continue
                    try:
                        cores = describe(sockets(peer.id()))
                        entry['sockets_to_%d' % peer.id()] = cores
                        entry['socket_count_to_%d' % peer.id()] = (
                            len(cores) if isinstance(cores, list) else None)
                    except BaseException as error:
                        entry['sockets_to_%d' % peer.id()] = (
                            '%s: %s' % (type(error).__name__, str(error)[:160]))
                per_device.append(entry)
            report['devices'] = per_device

            active = [d.get('get_active_ethernet_cores') for d in per_device]
            counts = [len(a) for a in active if isinstance(a, list)]
            if counts:
                report['active_ethernet_cores_per_device'] = counts

            # The per-peer socket count is the load-bearing number: active cores
            # can include links to a third board, and this rig has a spare.
            # Device ids are read back rather than assumed to be 0 and 1.
            peer = []
            for entry in per_device:
                for key, value in entry.items():
                    if key.startswith('socket_count_to_') and isinstance(value, int):
                        peer.append(value)
            if peer:
                links = max(peer)
                report['links_to_peer'] = links
                # Two links per QSFP-DD800 cable, 400 Gb/s each.
                report['cables_to_peer'] = links / 2.0
                report['theoretical_gb_s'] = links * 50.0
                report['measured_gb_s'] = 83.74
                report['percent_of_theoretical'] = round(100 * 83.74 / (links * 50.0), 1) if links else None
                report['reading'] = (
                    'one cable, both links already in use at 84% of its 100 GB/s: '
                    'more bandwidth needs another cable, not a config change'
                    if links <= 2 else
                    '%d links (%.0f cables) are up but only ~2 links of bandwidth is '
                    'being delivered, so a software cap is leaving %.0f GB/s unused'
                    % (links, links / 2.0, links * 50.0 - 83.74))
        finally:
            ttnn.close_mesh_device(mesh)
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['traceback'] = traceback.format_exc()[-1500:]

    print('<<<FABRIC_TOPO_JSON_BEGIN>>>', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print('<<<FABRIC_TOPO_JSON_END>>>', flush=True)
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
