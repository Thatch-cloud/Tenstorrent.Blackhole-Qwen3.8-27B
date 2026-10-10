"""What the two fabric-upload probes share (tp4_h2d_probe.py, tp4_p2p_probe.py; docs/tp4-fabric-upload.md section 3).

Both run INSIDE a serving image on the four-card (1, 4) mesh from the fabric step of qwen-c2-serving.yml (C2_CARDS=quad,
C2_ACTIONS=... fabric, C2_FABRIC_PROBE=h2d|p2p), which mounts this checkout read-only at /c2, maps every board, sets the
ring descriptor and runs with the serving contract off. Each opens the mesh ONCE and closes it (a second open in one job
is what the ethernet-core teardown wedge punishes), refuses before any device is touched when the descriptor is not the
ring's, and rewrites its JSON report after every arm, so a watchdog or a step timeout still leaves what was measured.

Stdlib at import; ttnn and torch arrive as arguments (the CPU suite hands in fakes).
"""

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fabric_upload_plan as plan  # noqa: E402
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

WATCHDOG_S = 1680
# The pattern's period (elements): a prime, so no page size (16 KiB, 2 KiB, 1,088 B) or shard repeats it, and its values,
# small integers below 128 in magnitude, are exact in bfloat16 and in bfloat8_b whatever a block's shared exponent.
PATTERN_PERIOD = 1000003
PATTERN_MODULUS = 241
PATTERN_SHIFT = 120


def descriptor_refusal(environ):
    """(refusal or None, descriptor path, problems): the probes open the mesh only under the ring descriptor."""
    descriptor = environ.get('TT_MESH_GRAPH_DESC_PATH')
    problems = None
    if descriptor and os.path.isfile(descriptor):
        with open(descriptor, encoding='utf-8') as handle:
            problems = tp4_mesh.descriptor_problems(handle.read())
    return tp4_fabric_probe.descriptor_refusal(descriptor, problems), descriptor, problems


class Report(object):
    """The probe's JSON report, rewritten whole after every arm (write, then rename: never a torn file)."""

    def __init__(self, path, kind, **fields):
        self.path = path
        self.data = dict(kind=kind, arms={}, problems=[], started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        self.data.update(fields)

    def arm(self, name, record):
        self.data['arms'][name] = record
        self.save()
        return record

    def problem(self, text):
        self.data['problems'].append(text)
        self.save()

    def save(self):
        if not self.path:
            return
        temporary = self.path + '.part'
        with open(temporary, 'w') as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True, default=str)
        os.replace(temporary, self.path)


def start_watchdog(tag, seconds=WATCHDOG_S, log=print):
    """A daemon timer that ends the process with status 3 (the report on disk is what was measured until then)."""
    def fire():
        log('%s watchdog after %d s: exiting 3; the report holds the arms that finished' % (tag, seconds))
        sys.stdout.flush()
        os._exit(3)
    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def open_mesh(ttnn, fabric):
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
    return ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)


def close_mesh(ttnn, mesh, submeshes=()):
    """Close the mesh (draining the submeshes' queues first when the runtime offers it); returns True or the error."""
    try:
        if submeshes and hasattr(mesh, 'quiesce_devices'):
            mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)
        return True
    except Exception as error:  # noqa: BLE001
        return '%s: %s' % (type(error).__name__, error)


def positions(ttnn, mesh, dev_root='/dev/tenstorrent', sys_root='/sys', rdev_of=None):
    """The mesh's chip ids in (1, 4) order and each one's PCIe link (fabric_upload_plan.card_links), plus the trained
    ethernet links per chip pair: what the relay pairing and every per-card number are labelled with."""
    order = [int(chip) for chip in mesh.get_device_ids()]
    text = ''
    try:
        with open(ttnn.cluster.serialize_cluster_descriptor(), 'r') as handle:
            text = handle.read()
    except Exception as error:  # noqa: BLE001
        text = ''
        mmio_error = '%s: %s' % (type(error).__name__, error)
    else:
        mmio_error = None
    mmio = plan.parse_mmio_chips(text) if text else None
    rows = plan.card_links(order, mmio, dev_root=dev_root, sys_root=sys_root, rdev_of=rdev_of)
    links = tp4_mesh.parse_ethernet_links(text) if text else None
    return dict(order=order, cards=rows, mmio_known=bool(mmio), descriptor_error=mmio_error,
                eth_links=dict(('%d-%d' % pair, count) for pair, count in sorted((links or {}).items())),
                widths=[row['width'] for row in rows], pairs=plan.relay_pairs([row['width'] for row in rows]))


def pattern(torch, elements, offset=0):
    """A deterministic bfloat16 vector of `elements` small integers (period PATTERN_PERIOD, shifted by offset)."""
    period = min(int(elements), PATTERN_PERIOD)
    base = ((torch.arange(period, dtype=torch.int32) + int(offset)) % PATTERN_MODULUS - PATTERN_SHIFT).to(torch.bfloat16)
    reps = -(-int(elements) // period)
    return base.repeat(reps)[:int(elements)]


def timed(call, sync):
    """Seconds for call() followed by sync() (the device has finished when this returns)."""
    started = time.perf_counter()
    call()
    sync()
    return time.perf_counter() - started


def error_text(error):
    return '%s: %s' % (type(error).__name__, str(error)[:400])


def guarded(report, name, call, default=None):
    """Run one arm; an exception becomes a problem line in the report (the next arm still runs) and `default` is returned."""
    try:
        return call()
    except Exception as error:  # noqa: BLE001
        report.problem('arm %s stopped: %s' % (name, error_text(error)))
        return default
