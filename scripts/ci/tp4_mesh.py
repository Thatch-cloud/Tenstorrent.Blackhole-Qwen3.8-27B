"""Four p150a cards as one (1, 4) mesh: the TP4 fabric table, its mesh graph descriptor and the ring check.

WHY A 2x2 DESCRIPTOR FOR A 1x4 RING (tt-metal v0.77.0-rc1, read at the cited lines):
  - The TT plugin sets FABRIC_1D on every non-Galaxy box (vllm_tt_plugin worker.get_fabric_config), and on a
    non-Galaxy box even FABRIC_1D_RING builds MESH connectivity (fabric_host_utils.cpp get_fabric_type): a
    declared [1, 4] RING descriptor would lose its wrap edge (mesh_graph.cpp: the FabricConfig may only
    restrict what the descriptor declares, and MESH declares no wrap). The ring's closing edge would not be a
    fabric edge.
  - The model picks Ring collectives whenever the cluster reports P150_X4 (tt_transformers model_config.py
    ccl_topology), which four visible p150 cards always do (tt_cluster.cpp: 4 x P150 -> P150_X4). Ring needs
    the closing edge.
  - A [2, 2] mesh makes every edge of a 4-cycle a mesh edge, and opening MeshShape(1, 4) on it maps the line
    ring-first (mesh_device_view.cpp get_line_coordinates: a closed path is tried before a plain line), so
    device 3's neighbour is device 0 over a real link. That is exactly how upstream runs this model family at
    P150x4 (its p150_x4 descriptor is [2, 2]; qwen36 demo: MESH_DEVICE=P150x4 -> (1, 4) with FABRIC_1D), and
    it is the shape of upstream's own QuietBox 2x2 test descriptor (2 channels).
  - Channels: our fabric trains 2 links per card pair (a full mesh: every pair of the four cards is cabled
    once). The descriptor says 2 with policy RELAXED, as upstream's p300_x2 descriptor does; the check below
    (not the descriptor) is what refuses a broken ring. The two diagonals of the 4-cycle are cabled too; the
    fabric routes the declared mesh only, so they carry nothing.
  - num_links 2 needs no patch: tt_ccl's link table gives P150x4 (2, 2) (models/common/modules/tt_ccl.py), the
    trained count per ring edge; tp_common's prefill collectives hard-code 2 as well.

THE CHECK. After the mesh opens, its device ids in (1, 4) order are the ring the collectives use; the UMD
cluster descriptor (ttnn.cluster.serialize_cluster_descriptor) lists every trained ethernet link. check_ring
counts the links of each ring edge, the closing edge included: an edge with none refuses (a Ring collective
there has no path), an edge with fewer than LINKS is DEGRADED (it serves, at that edge's links), and a mesh
whose device ids the descriptor does not list at all is UNKNOWN (logged, never refused: the check cannot
relate the two numberings, which J0's fabric probe settles on hardware). Nothing here names a card: the ids
are UMD's, read at run time, and they renumber on every reset.

Stdlib only, Python 3.7 syntax: the gate scripts on the rig host import it, and so does the serving contract
inside the image (ttnn only inside the functions that are handed it).
"""

import re

MESH_DEVICE = 'P150x4'
MESH_SHAPE = (1, 4)
DEVICES = 4
# Trained links per card pair on the four-card fabric (measured 2026-09-30: 2 on every one of the six pairs),
# and the count tt_ccl's table gives P150x4 collectives.
LINKS = 2
# Below this an edge cannot carry a Ring collective at all.
MIN_LINKS = 1
FABRIC_CONFIG = 'FABRIC_1D'
FABRIC_CONFIGS = ('FABRIC_1D', 'FABRIC_1D_RING')
DESCRIPTOR_NAME = 'qwen_p150x4_ring_mesh_graph_descriptor.textproto'
DESCRIPTOR_SOURCE = 'scripts/ci/' + DESCRIPTOR_NAME
# Where the serving image lays it (docker/qwen-c2-overlay.txt); the TP4 profiles name this path.
DESCRIPTOR_PATH = '/opt/qwen-c2/mesh/' + DESCRIPTOR_NAME
DESCRIPTOR_DIMS = (2, 2)
# The pair (a (1, 2) mesh) at the links this cabling trains: 2 per pair, where the p150_x2 descriptor declares 4.
PAIR_DESCRIPTOR_NAME = 'qwen_p150x2_2link_mesh_graph_descriptor.textproto'
PAIR_DESCRIPTOR_SOURCE = 'scripts/ci/' + PAIR_DESCRIPTOR_NAME
PAIR_DESCRIPTOR_PATH = '/opt/qwen-c2/mesh/' + PAIR_DESCRIPTOR_NAME
DESCRIPTOR_POLICY = 'RELAXED'
# Vocabulary shards the on-device sampler takes (qwen36 model.py: <= 65,536 logits per device).
SAMPLER_MAX_LOGITS = 65536
VOCABULARY = 248320
LOG_PREFIX = '[QWEN-TP4]'


class RingError(RuntimeError):
    """The opened mesh is not a ring of trained links: a Ring collective would have no path."""


def descriptor_text(dims=DESCRIPTOR_DIMS, channels=LINKS, policy=DESCRIPTOR_POLICY):
    """The mesh graph descriptor, byte for byte as scripts/ci carries it (test_tp4_mesh holds the file to it)."""
    rows, cols = dims
    return ''.join((
        '# Four p150a cards opened as one (1, 4) mesh (scripts/ci/tp4_mesh.py says why the mesh is 2x2).\n',
        '# A 2x2 mesh makes every edge of the 4-cycle a mesh edge; MeshShape(1, 4) maps onto it ring-first.\n',
        '# %d channels per edge: the links each card pair trains. No card is named: the fabric maps\n' % channels,
        '# logical chips onto whatever the cluster discovers, and tp4_mesh.check_ring verifies the ring.\n',
        '\n',
        'mesh_descriptors {\n',
        '  name: "M0"\n',
        '  arch: BLACKHOLE\n',
        '  device_topology { dims: [ %d, %d ] }\n' % (rows, cols),
        '  host_topology   { dims: [ 1, 1 ] }\n',
        '  channels { count: %d policy: %s }\n' % (channels, policy),
        '}\n',
        '\n',
        'top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }\n',
    ))


def pair_descriptor_text(channels=LINKS, policy=DESCRIPTOR_POLICY):
    """The two-card mesh graph descriptor at `channels` links, byte for byte as scripts/ci carries it. Under
    STRICT_INIT (the TT plugin's default) fabric init needs as many trained links per edge as the descriptor
    declares (control_plane.cpp), and upstream's p150_x2 declares 4 where M-A now trains 2: the pair's TP2
    reference and baseline profiles (general-2link) open under this one instead."""
    return ''.join((
        '# Two p150a cards opened as one (1, 2) mesh (the TP2 pair) on a cabling that trains %d links per pair.\n' % channels,
        '# Upstream p150_x2 declares 4 channels; STRICT_INIT refuses an edge that trains fewer than declared.\n',
        '\n',
        'mesh_descriptors {\n',
        '  name: "M0"\n',
        '  arch: BLACKHOLE\n',
        '  device_topology { dims: [ 1, 2 ] }\n',
        '  host_topology   { dims: [ 1, 1 ] }\n',
        '  channels { count: %d policy: %s }\n' % (channels, policy),
        '}\n',
        '\n',
        'top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }\n',
    ))


def parse_descriptor(text):
    """The fields this project's descriptors set: {name, arch, dims, host, channels, policy, instances}.
    Refuses (ValueError) anything with more than one mesh or instance, or a field it cannot read."""
    body = re.sub(r'#[^\n]*', '', text)
    meshes = re.findall(r'mesh_descriptors\s*\{', body)
    instances = re.findall(r'top_level_instance\s*\{', body)
    if len(meshes) != 1 or len(instances) != 1:
        raise ValueError('expected one mesh_descriptors and one top_level_instance, found %d and %d'
                         % (len(meshes), len(instances)))

    def one(pattern, what):
        found = re.findall(pattern, body)
        if len(found) != 1:
            raise ValueError('expected one %s, found %d' % (what, len(found)))
        return found[0]

    dims = one(r'device_topology\s*\{\s*dims:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', 'device_topology dims')
    host = one(r'host_topology\s*\{\s*dims:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', 'host_topology dims')
    count, policy = one(r'channels\s*\{\s*count:\s*(\d+)\s*policy:\s*([A-Z_]+)\s*\}', 'channels')
    return dict(name=one(r'name:\s*"([^"]+)"', 'name'), arch=one(r'arch:\s*([A-Z_0-9]+)', 'arch'),
                dims=(int(dims[0]), int(dims[1])), host=(int(host[0]), int(host[1])), channels=int(count),
                policy=policy, ring_axes='dim_types' in body)


def descriptor_problems(text, channels=LINKS):
    """Why a descriptor is not the four-card ring's, [] when it is."""
    try:
        found = parse_descriptor(text)
    except ValueError as error:
        return ['unreadable: %s' % error]
    problems = []
    if found['arch'] != 'BLACKHOLE':
        problems.append('arch %s, not BLACKHOLE' % found['arch'])
    if found['dims'] != DESCRIPTOR_DIMS:
        problems.append('device_topology %s, not %s: a (1, 4) ring needs every 4-cycle edge to be a mesh edge'
                        % (list(found['dims']), list(DESCRIPTOR_DIMS)))
    if found['host'] != (1, 1):
        problems.append('host_topology %s: one host' % list(found['host']))
    if found['channels'] != channels:
        problems.append('%d channels, not the %d each card pair trains' % (found['channels'], channels))
    if found['policy'] != DESCRIPTOR_POLICY:
        problems.append('policy %s, not %s' % (found['policy'], DESCRIPTOR_POLICY))
    if found['ring_axes']:
        problems.append('dim_types declared: on this box the fabric builds MESH connectivity whatever they say')
    return problems


def ethernet_section(text):
    """The ethernet_connections block of a UMD cluster descriptor (YAML), or None."""
    match = re.search(r'^ethernet_connections:[^\n]*\n', text, flags=re.M)
    if match is None:
        return None
    rest = text[match.end():]
    end = re.search(r'^\S', rest, flags=re.M)
    return rest[:end.start()] if end else rest


def parse_ethernet_links(text):
    """{(a, b): links} with a < b, one count per trained ethernet link between chips a and b, from a UMD
    cluster descriptor. Each connection is two endpoints (chip, chan), in block or flow YAML; a connection
    listed from both ends counts once. None when the descriptor has no ethernet_connections section;
    ValueError when the section cannot be paired into connections."""
    section = ethernet_section(text)
    if section is None:
        return None
    endpoints = [(int(chip), int(chan)) for chip, chan in
                 re.findall(r'chip:\s*(\d+)[\s,]*chan:\s*(\d+)', section)]
    if len(endpoints) % 2:
        raise ValueError('ethernet_connections holds %d endpoints, not pairs' % len(endpoints))
    seen, links = set(), {}
    for index in range(0, len(endpoints), 2):
        first, second = endpoints[index], endpoints[index + 1]
        key = frozenset((first, second))
        if first == second or key in seen:
            continue
        seen.add(key)
        if first[0] == second[0]:
            continue
        pair = tuple(sorted((first[0], second[0])))
        links[pair] = links.get(pair, 0) + 1
    return links


def links_between(links, a, b):
    return links.get(tuple(sorted((a, b))), 0)


def ring_edges(order):
    """Consecutive device pairs of a ring, the closing edge (last, first) included."""
    order = list(order)
    return [(order[index], order[(index + 1) % len(order)]) for index in range(len(order))] if len(order) > 2 \
        else ([(order[0], order[1])] if len(order) == 2 else [])


def check_ring(order, links, want=LINKS, need=MIN_LINKS):
    """The ring `order` (device ids in mesh order) over `links`: {ok, degraded, order, edges, unused, problems}.
    ok is False when an edge has fewer than `need` links (or the order is not DEVICES distinct ids);
    degraded when every edge has at least `need` but one has fewer than `want`. unused counts the trained
    links between cards that are not ring neighbours (the full mesh's diagonals). ok is None - unknown, never a
    refusal - when a device id of the order has no link at all in `links`: the mesh's ids and the descriptor's
    chip ids are then not one numbering, and no edge can be judged."""
    order = [int(device) for device in order]
    problems = []
    if len(order) != DEVICES or len(set(order)) != DEVICES:
        problems.append('mesh order %s is not %d distinct devices' % (order, DEVICES))
    linked = set(chip for pair, count in links.items() if count for chip in pair)
    strangers = [device for device in order if device not in linked]
    if strangers and not problems:
        edges = [dict(a=a, b=b, links=links_between(links, a, b)) for a, b in ring_edges(order)]
        return dict(ok=None, degraded=False, order=order, want=want, need=need, edges=edges, unused=0,
                    problems=['device ids %s have no link in the cluster descriptor (chips %s): the mesh and the '
                              'descriptor do not share a numbering' % (strangers, sorted(linked))])
    edges = [dict(a=a, b=b, links=links_between(links, a, b)) for a, b in ring_edges(order)]
    short = [edge for edge in edges if edge['links'] < need]
    for edge in short:
        problems.append('ring edge %d-%d has %d trained links: a Ring collective has no path there'
                        % (edge['a'], edge['b'], edge['links']))
    degraded = not short and any(edge['links'] < want for edge in edges)
    neighbours = set(frozenset((edge['a'], edge['b'])) for edge in edges)
    unused = sum(count for pair, count in links.items()
                 if pair[0] in order and pair[1] in order and frozenset(pair) not in neighbours)
    return dict(ok=not problems, degraded=degraded, order=order, want=want, need=need, edges=edges, unused=unused,
                problems=problems)


def hamiltonian_rings(chips, links, need=MIN_LINKS):
    """Every 4-cycle (in general: Hamiltonian cycle) over `chips` whose edges each have >= `need` links, each once:
    rotated to start at the smallest id, its direction the one with the smaller second id."""
    chips = sorted(set(int(chip) for chip in chips))
    if len(chips) < 3:
        return []
    start, rings = chips[0], []

    def extend(path, left):
        if not left:
            if links_between(links, path[-1], start) >= need and path[1] < path[-1]:
                rings.append(tuple(path))
            return
        for chip in sorted(left):
            if links_between(links, path[-1], chip) >= need:
                extend(path + [chip], left - {chip})

    extend([start], set(chips[1:]))
    return rings


def describe(report):
    """One log line for a check_ring report."""
    edges = ' '.join('%d-%d:%d' % (edge['a'], edge['b'], edge['links']) for edge in report['edges'])
    if report['ok'] is None:
        verdict = 'UNKNOWN'
    else:
        verdict = 'OK' if report['ok'] and not report['degraded'] else ('DEGRADED' if report['ok'] else 'BROKEN')
    return '%s ring %s order=%s edges=[%s] unused_links=%d%s' % (
        LOG_PREFIX, verdict, report['order'], edges, report['unused'],
        (' problems: ' + '; '.join(report['problems'])) if report['problems'] else '')


def sampler_fits(devices, vocabulary=VOCABULARY, limit=SAMPLER_MAX_LOGITS):
    """Whether the on-device sampler's per-device logits fit (qwen36 model.py builds it only then)."""
    return devices > 0 and -(-vocabulary // devices) <= limit


def check_open_mesh(mesh_device, ttnn, want=LINKS, need=MIN_LINKS, read=None):
    """check_ring on an opened (1, 4) mesh: its device ids in mesh order against the cluster descriptor's links.
    None for any other shape. `read` (path -> text) is injectable."""
    if tuple(int(dim) for dim in mesh_device.shape) != MESH_SHAPE:
        return None
    order = [int(device) for device in mesh_device.get_device_ids()]
    path = ttnn.cluster.serialize_cluster_descriptor()
    if read is None:
        def read(name):
            with open(str(name), 'r') as handle:
                return handle.read()
    links = parse_ethernet_links(read(path))
    if links is None:
        raise ValueError('the cluster descriptor %s has no ethernet_connections section' % path)
    return check_ring(order, links, want=want, need=need)


def install_ring_check(module, log, want=LINKS, need=MIN_LINKS, check=None):
    """Wrap `module`.open_mesh_device (the TT plugin's worker module) so a (1, 4) mesh is held to check_ring as
    soon as it opens: a broken ring raises RingError (the engine does not start), a degraded one is logged, and a
    check that cannot run is logged and serving goes on. Returns False if already installed."""
    original = getattr(module, 'open_mesh_device')
    if getattr(original, '_qwen_tp4_ring_check', False):
        return False

    def open_mesh_device(*args, **kwargs):
        mesh = original(*args, **kwargs)
        try:
            if check is not None:
                report = check(mesh)
            else:
                import ttnn

                report = check_open_mesh(mesh, ttnn, want=want, need=need)
        except Exception as error:
            log('%s ring check could not run: %s: %s' % (LOG_PREFIX, type(error).__name__, error))
            return mesh
        if report is None:
            return mesh
        log(describe(report))
        if report['ok'] is False:
            raise RingError('; '.join(report['problems']))
        return mesh

    open_mesh_device._qwen_tp4_ring_check = True
    module.open_mesh_device = open_mesh_device
    return True
