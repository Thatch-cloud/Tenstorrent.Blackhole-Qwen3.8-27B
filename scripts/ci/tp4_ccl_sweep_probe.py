"""The four-card collective-options sweep (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=ccl-sweep, C2_CARDS=quad): F-C1 of the fusion plan.

    python3 -B /c2/scripts/ci/tp4_ccl_sweep_probe.py --fabric FABRIC_1D --output /probe-results/ccl-sweep.json [--payload 8192]

The arms, the exactness rule, the in-trace timing and the verdict are optimisation/ttnn-op/ccl_sweep/ccl_sweep.py's; this wrapper opens the served (1, 4)
mesh the way the fabric probe, the reduction-order spike and the mesh-read probe do (the ring descriptor, ONE fabric config, one open, one close) and hands
the sweep the model's own TT_CCL and tt_all_reduce. ONE FABRIC CONFIG PER RUN: --fabric (FABRIC_1D is what the TT plugin sets; FABRIC_1D_RING the
alternative) and --payload (the fabric router's max packet payload in bytes; absent keeps the runtime's default, 4352; 8192 is four bfloat16 tile pages)
are process-level and fixed before the mesh opens. With --payload the runtime's own readback (get_tt_fabric_max_payload_size_bytes) is the control: when
it does not report the requested size the run measures nothing (an inert lever reads like a physical result), and a payload below 4096 is refused (it
shrinks the ring reduce-scatter's tile_granularity from 8 to 4 and moves its chunk parity).

Last stdout lines: 'CCL_SWEEP verdict=...' then one JSON object (kind ccl-sweep-quad). Exit 0 any measured verdict, 1 the served config is not bit-exact
(BASE-INEXACT), 2 the mesh did not open / refused to open / the payload lever did not move, 3 a watchdog (the partial report is written; reset all four
cards before the next job).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'ccl_sweep'))

import ccl_sweep  # noqa: E402
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

KIND = ccl_sweep.KIND
MIN_PAYLOAD = 4096


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--payload', type=int, default=None, help='fabric max packet payload in bytes (default: the runtime\'s)')
    parser.add_argument('--only', choices=('all', 'rs', 'ag'), default='all')
    parser.add_argument('--quick', action='store_true', help='a short grid (a smoke of the harness, not a sweep)')
    parser.add_argument('--skip-probe-only', action='store_true', help='skip the barrier-removal arms (the ones that can hang)')
    parser.add_argument('--no-exotic', action='store_true', help='skip the exotic stage (uneven worker splits, the gather\'s line route and via-broadcast program)')
    parser.add_argument('--seeds', type=int, default=ccl_sweep.SEEDS)
    parser.add_argument('--rounds', type=int, default=ccl_sweep.ROUNDS)
    parser.add_argument('--replays', type=int, default=ccl_sweep.REPLAYS)
    parser.add_argument('--trace-region', type=int, default=256 * 1024 * 1024)
    return parser


def check_options(options):
    """Why these options cannot run, or None."""
    if options.payload is not None and not MIN_PAYLOAD <= options.payload <= 15232:
        return '--payload %d: from %d (below it tile_granularity falls to 4 and the ring order moves) to 15232 (the Blackhole maximum)' % (
            options.payload, MIN_PAYLOAD)
    if options.seeds < 1 or options.rounds < 3 or options.replays < 3:
        return 'at least 1 seed, 3 rounds and 3 replays'
    return None


def set_fabric(ttnn, options, report):
    """The one fabric config of this run, with the router config when a payload is asked for."""
    config = getattr(ttnn.FabricConfig, options.fabric)
    if options.payload is None:
        ttnn.set_fabric_config(config)
        return
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = options.payload
    ttnn.set_fabric_config(config, router_config=router)
    report['payload_requested'] = options.payload


def run(options, ttnn=None, torch=None, environ=None, log=print, modules=None, deadlines=True):
    """The measurement; returns the report dict. `ttnn`, `torch` and `modules` (TT_CCL, tt_all_reduce, get_num_links) are the image's, or fakes in the tests."""
    environ = os.environ if environ is None else environ
    report = dict(kind=KIND, fabric=options.fabric, payload_requested=options.payload, opened=False, descriptor=environ.get('TT_MESH_GRAPH_DESC_PATH'),
                  serving_contract=environ.get('QWEN_C2_SERVING'), seeds=options.seeds, rounds=options.rounds, replays=options.replays, quick=options.quick,
                  n_high=ccl_sweep.N_HIGH, n_low=ccl_sweep.N_LOW, scenarios={})
    refusal = check_options(options)
    if refusal:
        report.update(error='refused: ' + refusal)
        return report
    if ttnn is None:
        descriptor = report['descriptor']
        problems = None
        if descriptor and os.path.isfile(descriptor):
            with open(descriptor, encoding='utf-8') as handle:
                problems = tp4_mesh.descriptor_problems(handle.read())
            report['descriptor_problems'] = problems
        refusal = tp4_fabric_probe.descriptor_refusal(descriptor, problems)
        if refusal:
            report.update(error='refused to open: ' + refusal)
            return report
        import torch
        import ttnn
        from models.common.modules.tt_ccl import get_num_links
        from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
        modules = dict(TT_CCL=TT_CCL, tt_all_reduce=tt_all_reduce, get_num_links=get_num_links)
    mesh = None
    deadline = ccl_sweep.Deadline(report, options.output, seconds=ccl_sweep.CONFIG_DEADLINE_S if deadlines else None, log=log)
    try:
        set_fabric(ttnn, options, report)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=options.trace_region)
        report['opened'] = True
        report['chips'] = int(mesh.get_num_devices())
        report['order'] = [int(device) for device in mesh.get_device_ids()]
        try:
            report['cluster_type'] = str(ttnn.cluster.get_cluster_type())
        except Exception as error:  # noqa: BLE001
            report['cluster_type'] = 'unknown (%s: %s)' % (type(error).__name__, error)
        report['num_links'] = int(modules['get_num_links'](mesh))
        actual = getattr(ttnn, 'get_tt_fabric_max_payload_size_bytes', None)
        report['payload_actual'] = int(actual()) if callable(actual) else None
        if options.payload is not None and report['payload_actual'] != options.payload:
            report.update(error='the payload lever did not move: asked for %d, the runtime reports %s; nothing is measured' % (options.payload, report['payload_actual']),
                          lever_moved=False)
            return report
        report['lever_moved'] = True if options.payload is not None else None
        if report['chips'] != 4 or report['num_links'] != 2:
            report.update(error='the stack\'s census is four chips at two links; this mesh has %s chips and %s links' % (report['chips'], report['num_links']))
            return report
        collective = modules['TT_CCL'](mesh)

        def save(current):
            with open(options.output, 'w') as handle:
                json.dump(current, handle, indent=2, sort_keys=True, default=str)

        ccl_sweep.run(options, ttnn, torch, mesh, collective, modules['tt_all_reduce'], report, save, deadline=deadline, log=log)
    except Exception as error:  # noqa: BLE001
        report.update(error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:],
                      known_failure=tp4_fabric_probe.known_signature(traceback.format_exc()))
    finally:
        deadline.disarm()
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    return report


def main(argv=None, ttnn=None, torch=None, environ=None, log=print, modules=None, deadlines=True):
    options = build_parser().parse_args(argv)
    timer = None
    if deadlines:
        timer = threading.Timer(ccl_sweep.JOB_DEADLINE_S, lambda: (print('CCL_SWEEP job watchdog', flush=True), os._exit(3)))
        timer.daemon = True
        timer.start()
    report = run(options, ttnn=ttnn, torch=torch, environ=environ, log=log, modules=modules, deadlines=deadlines)
    report['summary'] = ccl_sweep.summarise(report)
    text, status = ccl_sweep.verdict(report)
    if report.get('error') and not report.get('opened'):
        text, status = 'NOT-MEASURED', 2
    elif report.get('error') and 'payload lever' in report['error']:
        text, status = 'NOT-MEASURED', 2
    report['verdict'] = text
    line = ccl_sweep.verdict_line(report, text)
    report['verdict_line'] = line
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    log(line)
    if report.get('error'):
        log('CCL_SWEEP error: %s' % report['error'])
    if report.get('known_failure'):
        log('CCL_SWEEP known failure: %s' % report['known_failure'])
    log(json.dumps(report, sort_keys=True, default=str))
    if timer is not None:
        timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
