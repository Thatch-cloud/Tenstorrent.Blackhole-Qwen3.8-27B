"""The four-card fabric probe (qwen-c2-serving.yml 'fabric', C2_CARDS=quad): open all four cards as the (1, 4)
mesh the TP4 profiles open, prove the ring, and time the model's own collectives on it.

    python3 -B /probe/tp4_fabric_probe.py --fabric FABRIC_1D --output /probe-results/fabric-probe.json

Runs INSIDE a serving image (ttnn, tt-metal's models tree) with every Blackhole board mapped and the four-card
descriptor at TT_MESH_GRAPH_DESC_PATH; the workflow mounts this file, tp4_mesh.py and the descriptor from the
checkout, so it needs no image built from this commit. One process, one fabric config, one mesh open: a
second open in the same job is what the ethernet-core teardown wedge punishes.

WHAT IT REPORTS (one JSON file, and FABRIC_PROBE lines on stdout):
  mesh     the cluster type tt-metal sees (P150_X4 expected: four visible p150), the mesh shape, the device ids in
           (1, 4) order, every trained link of the cluster descriptor per chip pair, and tp4_mesh.check_ring on that
           order (OK / DEGRADED / BROKEN / UNKNOWN), plus the ring-first mapping's closing edge
  links    tt_ccl.get_num_links for the mesh (the count every model collective takes; 2 expected for P150x4)
  ops      the model's own wrappers (tt_transformers ccl.tt_all_gather and tt_all_reduce, the latter a
           reduce_scatter_minimal_async on a 1xN mesh), each at decode size (32 rows x 5,120) and prefill size
           (2,048 rows x 5,120), Ring and Linear topology, 1 and 2 links: exact against a host reference (small
           integers, exact in bfloat16), then N timed calls after W warm ones, each bracketed by
           synchronize_device: the median, and the bus bandwidth the median implies - (n - 1) / n of the
           gathered tensor's bytes per device per call for an all-gather, the same of the reduced input for a
           reduce-scatter (the standard ring-algorithm figure, so 1 and 2 links compare directly)
  verdict  FAIL, without touching a device, unless TT_MESH_GRAPH_DESC_PATH names the ring descriptor and it has no
           problems (the image's serving contract, QWEN_C2_SERVING=1, would otherwise replace it with the pair's
           before ttnn loads: the workflow runs the probe with QWEN_C2_SERVING=0); PASS when the mesh opened, the ring is OK (not merely DEGRADED), the link count is 2 and every op
           was exact; the timings are reported, never judged (the upstream p150_x4 goldens are a different box)

Stdlib at import; ttnn, torch and the models tree only inside run(), so the CPU suite can test the arithmetic.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tp4_mesh  # noqa: E402

# (rows, width): the decode residual (one tile of rows) and a prefill chunk, both at the model's hidden size.
SIZES = (('decode', 32, 5120), ('prefill', 2048, 5120))
TOPOLOGIES = ('Ring', 'Linear')
LINKS = (1, 2)
OPS = ('all_gather', 'reduce_scatter')
WARM, TIMED = 3, 20
# Upstream issue 49701: opening a 2x2 on four p150a died in the L1 allocator on a harvested ethernet core.
KNOWN_SIGNATURES = (('No core coordinate found at', 'tenstorrent/tt-metal#49701 (harvested ETH core in the L1 '
                                                    'banking allocator on a 4 x p150a open)'),
                    ('Timed out while waiting for active ethernet core', 'the ethernet-core wedge: reset all four '
                                                                         'cards together'),
                    ('eth links', 'a cabled pair trained fewer links than the descriptor declares: STRICT_INIT '
                                  'refuses the edge at fabric init (TT_FATAL "Expected N eth links"); reset all '
                                  'four cards together and read the trained links before blaming the model'))


def bus_bytes(op, rows, width, devices, element_bytes=2):
    """Bytes each device moves in one call by the ring-algorithm count: (n - 1) / n of the full tensor."""
    full = rows * width * element_bytes
    return full * (devices - 1) // devices


def bandwidth(op, rows, width, devices, seconds):
    """GB/s (decimal) the median call implies; None for a zero time."""
    if not seconds:
        return None
    return round(bus_bytes(op, rows, width, devices) / seconds / 1e9, 3)


def known_signature(text):
    for needle, meaning in KNOWN_SIGNATURES:
        if needle in (text or ''):
            return meaning
    return None


def descriptor_refusal(path, problems):
    """Why the mesh must not be opened under this descriptor, None when it is the ring's: the path must name the
    ring descriptor file (an image whose serving contract booted in this process would have replaced it with the
    pair's) and the file must have no problems."""
    if not path or os.path.basename(path) != tp4_mesh.DESCRIPTOR_NAME:
        return 'TT_MESH_GRAPH_DESC_PATH is %r, not the ring descriptor %s' % (path, tp4_mesh.DESCRIPTOR_NAME)
    if problems is None:
        return 'the ring descriptor %s is not readable' % path
    if problems:
        return 'the ring descriptor has problems: %s' % '; '.join(problems)
    return None


def verdict(report):
    """(passed, reasons) for a finished report."""
    reasons = []
    if report.get('descriptor_refusal') or report.get('descriptor_problems'):
        reasons.append('descriptor: %s' % (report.get('descriptor_refusal')
                                           or '; '.join(report['descriptor_problems'])))
        return False, reasons
    if not report.get('opened'):
        reasons.append('the (1, 4) mesh did not open: %s' % (report.get('error') or 'unknown'))
        return False, reasons
    ring = report.get('ring') or {}
    if ring.get('ok') is not True:
        reasons.append('ring %s: %s' % ('UNKNOWN' if ring.get('ok') is None else 'BROKEN',
                                          '; '.join(ring.get('problems') or ['no report'])))
    elif ring.get('degraded'):
        reasons.append('ring DEGRADED: an edge trained fewer than %d links' % tp4_mesh.LINKS)
    if report.get('num_links') != tp4_mesh.LINKS:
        reasons.append('tt_ccl gives %r links, not %d' % (report.get('num_links'), tp4_mesh.LINKS))
    inexact = [record['name'] for record in report.get('ops') or () if record.get('exact') is not True]
    if inexact:
        reasons.append('not exact: %s' % ', '.join(inexact))
    if not report.get('ops'):
        reasons.append('no collective ran')
    return not reasons, reasons


def op_name(op, size, topology, links):
    return '%s/%s/%s/%dlink' % (op, size, topology.lower(), links)


def run(options, log=print):
    import torch
    import ttnn
    from models.common.modules.tt_ccl import get_num_links
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather, tt_all_reduce

    report = dict(fabric=options.fabric, descriptor=os.environ.get('TT_MESH_GRAPH_DESC_PATH'), opened=False, ops=[])
    descriptor = report['descriptor']
    problems = None
    if descriptor and os.path.isfile(descriptor):
        with open(descriptor, encoding='utf-8') as handle:
            problems = tp4_mesh.descriptor_problems(handle.read())
        report['descriptor_problems'] = problems
    report['serving_contract'] = os.environ.get('QWEN_C2_SERVING')
    refusal = descriptor_refusal(descriptor, problems)
    if refusal:
        # Refuse before any device is touched: a (1, 4) open under another descriptor fails, and would read as a
        # TP4 blocker when it is only the wrong descriptor.
        report['descriptor_refusal'] = refusal
        report['error'] = 'refused to open: ' + refusal
        return report
    mesh = None
    try:
        report['cluster_type'] = str(ttnn.cluster.get_cluster_type())
        report['devices_visible'] = int(ttnn.get_num_devices())
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576,
                                     trace_region_size=options.trace_region)
        report['opened'] = True
        report['mesh_shape'] = list(mesh.shape)
        report['order'] = [int(device) for device in mesh.get_device_ids()]
        with open(ttnn.cluster.serialize_cluster_descriptor(), 'r') as handle:
            links = tp4_mesh.parse_ethernet_links(handle.read()) or {}
        report['links'] = dict(('%d-%d' % pair, count) for pair, count in sorted(links.items()))
        report['ring'] = tp4_mesh.check_ring(report['order'], links)
        report['rings_available'] = [list(ring) for ring in tp4_mesh.hamiltonian_rings(report['order'], links,
                                                                                       tp4_mesh.LINKS)]
        log(tp4_mesh.describe(report['ring']))
        report['num_links'] = int(get_num_links(mesh))
        collective = TT_CCL(mesh)
        devices = mesh.get_num_devices()
        for size, rows, width in SIZES:
            shard = width // devices
            # device d holds columns [d * shard, (d + 1) * shard) of a full [1, 1, rows, width] tensor
            full = (torch.arange(rows * width, dtype=torch.float32).reshape(1, 1, rows, width) % 17)
            for topology in TOPOLOGIES:
                for links_count in LINKS:
                    for op in OPS:
                        name = op_name(op, size, topology, links_count)
                        record = dict(name=name, op=op, size=size, rows=rows, width=width, topology=topology,
                                      links=links_count)
                        try:
                            record.update(time_op(ttnn, torch, mesh, collective, op, full, shard, devices, topology,
                                                  links_count, tt_all_gather, tt_all_reduce, options))
                            record['gbps'] = bandwidth(op, rows, width, devices, record['median_ms'] / 1000.0)
                        except Exception as error:
                            record.update(exact=False, error='%s: %s' % (type(error).__name__, error))
                        report['ops'].append(record)
                        log('FABRIC_PROBE op %s exact=%s median_us=%s gbps=%s%s' % (
                            name, record.get('exact'), round(record['median_ms'] * 1000, 1)
                            if record.get('median_ms') is not None else None, record.get('gbps'),
                            (' error=' + record['error']) if record.get('error') else ''))
    except Exception as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()[-4000:]
        report['known_failure'] = known_signature(report['traceback'])
    finally:
        if mesh is not None:
            # The image's engines skip tt-metal's teardown for the ethernet-core wedge; a probe that closes the mesh
            # would leave the ring for the next job's reset to heal anyway, so it closes and says so.
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    return report


def time_op(ttnn, torch, mesh, collective, op, full, shard, devices, topology, links, tt_all_gather, tt_all_reduce,
            options):
    """One op: exactness on a fresh input, then WARM + TIMED calls, each synchronized; returns the record fields."""
    topology_value = getattr(ttnn.Topology, topology)
    if op == 'all_gather':
        source = torch.cat([full[..., d * shard:(d + 1) * shard] for d in range(devices)], dim=0)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        expected = full
    else:
        # every device holds the whole row block with its own offset: the reduce-scatter sums them and scatters
        # the sum's columns, device d receiving [d * shard, (d + 1) * shard)
        source = torch.cat([full + d for d in range(devices)], dim=0)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        expected = sum(full + d for d in range(devices))

    def upload():
        return ttnn.from_torch(source, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)

    def call(tensor):
        if op == 'all_gather':
            return tt_all_gather(tensor, mesh, collective, cluster_axis=None, dim=3, num_links=links,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=topology_value)
        return tt_all_reduce(tensor, mesh, collective, cluster_axis=0, dim=3, num_reduce_scatter_links=links,
                             num_all_gather_links=links, topology=topology_value,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output = call(upload())
    parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(output)]
    if op == 'all_gather':
        exact = all(torch.equal(part.reshape(expected.shape), expected) for part in parts)
    else:
        exact = all(torch.equal(part.reshape(1, 1, full.shape[2], shard), expected[..., d * shard:(d + 1) * shard])
                    for d, part in enumerate(parts))
    ttnn.deallocate(output)
    times = []
    for index in range(options.warm + options.timed):
        tensor = upload()
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        result = call(tensor)
        ttnn.synchronize_device(mesh)
        elapsed = time.perf_counter() - started
        ttnn.deallocate(result)
        if index >= options.warm:
            times.append(elapsed * 1000.0)
    return dict(exact=exact, median_ms=round(statistics.median(times), 4), min_ms=round(min(times), 4),
                calls=len(times))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warm', type=int, default=WARM)
    parser.add_argument('--timed', type=int, default=TIMED)
    parser.add_argument('--trace-region', type=int, default=0)
    return parser


def main(argv=None, runner=run, log=print):
    options = build_parser().parse_args(argv)
    report = runner(options, log=log)
    passed, reasons = verdict(report)
    report.update(passed=passed, reasons=reasons)
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    ring = report.get('ring') or {}
    log('FABRIC_PROBE fabric=%s opened=%s cluster=%s order=%s ring=%s num_links=%s passed=%s%s' % (
        options.fabric, report.get('opened'), report.get('cluster_type'), report.get('order'),
        tp4_mesh.describe(ring).split(' ')[2] if ring else None, report.get('num_links'), passed,
        (' reasons: ' + '; '.join(reasons)) if reasons else ''))
    if report.get('known_failure'):
        log('FABRIC_PROBE known failure: %s' % report['known_failure'])
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
