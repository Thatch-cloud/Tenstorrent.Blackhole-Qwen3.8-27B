"""The four-card mesh-read probe (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=mr, C2_CARDS=quad): MR and 2a of the fusion plan, decided at four chips.

    python3 -B /c2/scripts/ci/tp4_mr_probe.py --fabric FABRIC_1D --output /probe-results/mr-probe.json

The arms and the verdict rule are optimisation/ttnn-op/mr_probe/mr_probe.py's (the card-M harness runs the same arms on one chip and can only read
INCONCLUSIVE-SINGLE-CHIP). This wrapper opens the served (1, 4) mesh the way the fabric probe and the reduction-order spike do (the ring descriptor,
the fabric config, one open, one close) and runs them over four chips, where `serial` is the served verify readback (2 x 4 blocking reads), `mesh`
is two mesh-level reads, `word` is one joined word tensor read once and `overlap` is the non-blocking per-chip read: GO, MESH-ONLY or NO-GO.
Host-side reads only: nothing is computed on the cards and no byte of the model's output can change.

Last stdout lines: 'MR_PROBE verdict=...' then one JSON object (kind mr-probe-quad). Exit 0 any measured verdict, 2 the mesh did not open or the
descriptor is not the ring's, 3 the watchdog.
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
sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'mr_probe'))

import mr_probe  # noqa: E402
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

KIND = 'mr-probe-quad'
WATCHDOG_S = 1500


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--rows', type=int, default=64)
    return parser


def run(options, ttnn=None, environ=None, log=print):
    """The measurement; returns the report dict (verdict NOT-MEASURED with an error when the mesh could not be opened). `ttnn` is the module (a fake in the tests)."""
    environ = os.environ if environ is None else environ
    report = dict(kind=KIND, fabric=options.fabric, iterations=options.iterations, warmup=options.warmup, rows=options.rows, opened=False,
                  descriptor=environ.get('TT_MESH_GRAPH_DESC_PATH'), serving_contract=environ.get('QWEN_C2_SERVING'))
    if ttnn is None:
        descriptor = report['descriptor']
        problems = None
        if descriptor and os.path.isfile(descriptor):
            with open(descriptor, encoding='utf-8') as handle:
                problems = tp4_mesh.descriptor_problems(handle.read())
            report['descriptor_problems'] = problems
        refusal = tp4_fabric_probe.descriptor_refusal(descriptor, problems)
        if refusal:
            report.update(verdict='NOT-MEASURED', error='refused to open: ' + refusal)
            log('MR_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
            return report
        import ttnn
    mesh = None
    try:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
        report['opened'] = True
        chips = int(mesh.get_num_devices())
        report['chips'] = chips
        results, errors = mr_probe.run(ttnn, mesh, chips, options.iterations, options.warmup, options.rows)
        text, evidence = mr_probe.verdict(chips, results)
        report.update(arms=results, errors=errors, verdict=text, evidence=evidence)
        log('MR_PROBE verdict=%s chips=%d floor_us=%s serial_us=%s mesh_us=%s word_us=%s overlap_us=%s' % (
            text, chips, evidence['floor_us'], evidence['serial_us'], evidence['mesh_us'], evidence['word_us'], evidence['overlap_us']))
    except Exception as error:  # noqa: BLE001
        report.update(verdict='NOT-MEASURED', error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:])
        log('MR_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    return report


def main(argv=None, ttnn=None, environ=None, log=print):
    options = build_parser().parse_args(argv)
    if not 1 <= options.rows <= 64 or options.iterations < 20 or options.warmup < 0:
        print('refusing: rows 1..64, iterations >= 20, warmup >= 0', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('MR_PROBE watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    report = run(options, ttnn=ttnn, environ=environ, log=log)
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    log(json.dumps(report, sort_keys=True))
    timer.cancel()
    return 0 if report.get('verdict') != 'NOT-MEASURED' else 2


if __name__ == '__main__':
    sys.exit(main())
