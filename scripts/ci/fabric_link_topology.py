"""How many ethernet links actually connect the two cards, and what caps them.

The bandwidth probe (35424379930) measured 83.74 GB/s. One QSFP-DD800 cable
carries TWO ethernet links of 400 Gb/s, so a link is 50 GB/s and a cable is
100 GB/s. The measurement therefore exceeds one link - at least two are carrying
traffic - and sits at 84% of one cable. Two readings still fit:

  one cable, both links used   -> more bandwidth needs another cable
  two cables, half used        -> a software cap, and the rest are free now

The mesh graph descriptor already rules out one candidate cap. It says
`channels { count: 4 policy: RELAXED }`: the runtime is asking for four channels
and RELAXED lets it proceed with fewer. So the descriptor is not the limit, and
the question is what the hardware reports.

Run 35425107580 failed here on `MeshDevice.get_devices`, which does not exist in
this build. It did return the symbol tables, so this version drives the API the
build actually has - `ttnn.cluster` - and falls back to introspection rather than
guessing. Every accessor is called under a guard and unknown names are reported,
because a wrong guess should cost a line of output, not the hardware slot.
"""

import argparse
import io
import json
import traceback

# Zero-argument accessors worth trying on whatever object exposes them. Names
# come from the ttnn.cluster symbol table returned by run 35425107580.
CLUSTER_NULLARY = (
    'get_cluster_desc', 'number_of_devices', 'number_of_pci_devices',
    'get_num_active_ethernet_cores', 'get_ethernet_connections',
    'get_ethernet_connections_to_remote_devices', 'serialize_cluster_descriptor',
    'get_cluster_type', 'get_board_type',
)
DEVICE_NULLARY = (
    'get_active_ethernet_cores', 'get_inactive_ethernet_cores',
    'get_ethernet_cores_grouped_by_connected_chip_ids',
)


def describe(value, limit=64):
    """Render an API result without assuming its type."""
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, dict):
        return dict((str(k), describe(v)) for k, v in list(value.items())[:limit])
    try:
        items = list(value)
    except TypeError:
        return str(value)[:600]
    return [str(item)[:200] for item in items[:limit]]


def call_all(target, names, args=()):
    """Call each accessor that exists, recording outcome rather than raising."""
    found = {}
    for name in names:
        accessor = getattr(target, name, None)
        if accessor is None:
            found[name] = 'ABSENT'
            continue
        try:
            found[name] = describe(accessor(*args))
        except BaseException as error:
            found[name] = '%s: %s' % (type(error).__name__, str(error)[:200])
    return found



def count_links_from_yaml(text, a=0, b=1):
    """Count ethernet links between two chips in a UMD cluster descriptor.

    The descriptor lists each link as a pair of (chip, chan) entries under
    ethernet_connections, so every pair naming both chips is one link. Parsed
    by hand because the container is not guaranteed a yaml module, and the
    shape is fixed and simple:

        ethernet_connections:
          -
            - chip: 0
              chan: 6
            - chip: 1
              chan: 6

    Links to a third board, and to remote devices, must not be counted: this
    rig has a spare p150a.
    """
    import re
    parts = text.split("ethernet_connections:", 1)
    if len(parts) < 2:
        return None
    block = parts[1].split("\nethernet_connections_to_remote_devices:", 1)[0]
    links = 0
    for record in re.split(r"\n\s*-\s*\n", block):
        chips = [int(m) for m in re.findall(r"chip:\s*(\d+)", record)]
        if len(chips) == 2 and set(chips) == set([a, b]):
            links += 1
    return links


def count_links(connections, a=0, b=1):
    """Count ethernet channels joining chip a to chip b.

    The cluster descriptor maps chip -> channel -> (peer chip, peer channel).
    Each entry is one ethernet link, so the number of entries naming the peer is
    the link count, and two links make one QSFP-DD800 cable.
    """
    if not isinstance(connections, dict):
        return None
    channels = connections.get(str(a), connections.get(a))
    if not isinstance(channels, dict):
        return None
    return sum(1 for target in channels.values() if str(b) in str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'measured_gb_s': 83.74, 'gb_s_per_link': 50.0,
              'descriptor_channels': 4, 'descriptor_policy': 'RELAXED'}
    try:
        import ttnn

        cluster = getattr(ttnn, 'cluster', None)
        if cluster is None:
            report['cluster'] = 'ABSENT'
        else:
            report['cluster_symbols'] = sorted(
                name for name in dir(cluster) if not name.startswith('_'))
            report['cluster'] = call_all(cluster, CLUSTER_NULLARY)
            # Run 35425234093: every direct accessor is ABSENT in this build,
            # but serialize_cluster_descriptor returns a PATH to a yaml holding
            # the ethernet_connections. The container is --rm, so it has to be
            # read here rather than collected afterwards.
            path = report['cluster'].get('serialize_cluster_descriptor')
            if isinstance(path, str) and path.endswith('.yaml'):
                try:
                    text = io.open(path, encoding='utf-8', errors='replace').read()
                    report['cluster_descriptor_yaml'] = text[:20000]
                    links = count_links_from_yaml(text)
                    if links is not None:
                        report['links_to_peer'] = links
                        report['links_source'] = 'cluster_descriptor.yaml'
                except BaseException as error:
                    report['cluster_descriptor_error'] = (
                        '%s: %s' % (type(error).__name__, str(error)[:200]))

        # Opening the mesh is the fallback, and also confirms the runtime agrees.
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
        try:
            report['mesh_shape'] = str(mesh.shape)
            report['mesh_symbols'] = sorted(
                name for name in dir(mesh)
                if 'device' in name.lower() or 'eth' in name.lower())
            for name in ('get_device_ids', 'get_devices', 'devices'):
                accessor = getattr(mesh, name, None)
                if accessor is None:
                    continue
                try:
                    value = accessor() if callable(accessor) else accessor
                    report['mesh_%s' % name] = describe(value)
                    devices = list(value)
                except BaseException as error:
                    report['mesh_%s' % name] = '%s: %s' % (type(error).__name__,
                                                           str(error)[:160])
                    continue
                if devices and not isinstance(devices[0], int):
                    report['devices'] = [call_all(d, DEVICE_NULLARY) for d in devices[:2]]
                break
        finally:
            ttnn.close_mesh_device(mesh)

        links = report.get('links_to_peer')
        if links:
            report['cables_to_peer'] = links / 2.0
            report['theoretical_gb_s'] = links * 50.0
            report['percent_of_theoretical'] = round(100 * 83.74 / (links * 50.0), 1)
            report['reading'] = (
                'one cable, both links in use at 84% of its 100 GB/s: more bandwidth '
                'needs another cable, not a config change'
                if links <= 2 else
                '%d links (%g cables) are up and the descriptor asks for 4, but only '
                'about two links of bandwidth is delivered, so %.0f GB/s is unused'
                % (links, links / 2.0, links * 50.0 - 83.74))
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['traceback'] = traceback.format_exc()[-2000:]

    print('<<<FABRIC_TOPO_JSON_BEGIN>>>', flush=True)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print('<<<FABRIC_TOPO_JSON_END>>>', flush=True)
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
