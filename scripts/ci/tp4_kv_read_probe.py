"""The four-card region-read qualification (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=kvread, C2_CARDS=quad).

    python3 -B /c2/scripts/ci/tp4_kv_read_probe.py --fabric FABRIC_1D --output /probe-results/kv-read-probe.json

QUALIFY, part one: ttnn.qwen_read_blocks (the qwen_kv_read extension, optimisation/ttnn-op/kv_region_read) against the
whole-cache read, byte for byte, on the SERVED (1, 4) mesh. The checks are optimisation/ttnn-op/kv_region_read/kv_region_read_card.py's
run(): a cache of the production pool's size allocated as production allocates its KV, five block sets (one block, a run, scattered ids,
a shuffled order, two runs) read through the region read and through the whole-cache read and compared as bytes, no program-cache
growth, the read volume following the row, and the timing split (device read against host unpack) that the audit's cost model lacks.
This wrapper opens the mesh the way the fabric probe does (the ring descriptor, the fabric config, one open, one close).
Reads only: nothing is computed on the cards and no byte of the model's output can change.

The real-prompt half of the qualification is the prefix gate's read-qualify arm (QWEN_PREFIX_AUDIT_READ=cross): see docs/prefix-audit-cost.md.

Last stdout lines: 'KV_READ_PROBE verdict=PASS|FAIL|NOT-MEASURED ...' then one JSON object (kind kv-read-probe-quad). Exit 0 PASS, 1 FAIL
(any mismatch: the audit falls back to the old read), 2 the mesh did not open, the descriptor is not the ring's, or the extension is
absent, 3 the watchdog.
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
sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'kv_region_read'))

import kv_region_read_card as card  # noqa: E402
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

KIND = 'kv-read-probe-quad'
WATCHDOG_S = 1500
# The production pool at eight seats x 262k (docs/prefix-audit-cost.md): 19,968 blocks of 64 tokens, one KV head per chip.
PRODUCTION_BLOCKS = 19968


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--blocks', type=int, default=PRODUCTION_BLOCKS)
    parser.add_argument('--heads-per-chip', type=int, default=1)
    return parser


def run(options, ttnn=None, environ=None, log=print):
    """The measurement; returns the report dict (verdict NOT-MEASURED with an error when the mesh or the extension is missing)."""
    environ = os.environ if environ is None else environ
    report = dict(kind=KIND, fabric=options.fabric, blocks=options.blocks, heads_per_chip=options.heads_per_chip, opened=False,
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
            log('KV_READ_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
            return report
        import ttnn
    missing, source = card.ensure_extension(ttnn, environ.get('QWEN_KVREAD_DIR'))
    report['extension_source'] = source
    if missing:
        report.update(verdict='NOT-MEASURED', error=missing)
        log('KV_READ_PROBE verdict=NOT-MEASURED error=%s' % missing)
        return report
    mesh = None
    try:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
        report['opened'] = True
        chips = int(mesh.get_num_devices())
        report['chips'] = chips
        checks = card.run(ttnn, mesh, chips, options.blocks, options.heads_per_chip)
        log('KV_READ_PROBE extension source: %s' % source)
        report.update(checks=checks, verdict='PASS' if checks['ok'] else 'FAIL', problems=checks['problems'])
        log('KV_READ_PROBE verdict=%s chips=%d blocks=%d whole_device_read_s=%s whole_unpack_s=%s one_ms=%s run_ms=%s ms_per_block=%s growth=%s problems=%d' % (
            report['verdict'], chips, options.blocks, checks.get('whole_device_read_s'), checks.get('whole_unpack_s'),
            (checks.get('one') or {}).get('read_ms'), (checks.get('run') or {}).get('read_ms'), checks.get('read_ms_per_block_largest'),
            checks.get('program_cache_growth'), len(checks['problems'])))
        for problem in checks['problems']:
            log('KV_READ_PROBE problem: %s' % problem)
    except Exception as error:  # noqa: BLE001
        report.update(verdict='NOT-MEASURED', error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:])
        log('KV_READ_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
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
    if options.blocks < 64 or options.heads_per_chip < 1:
        print('refusing: blocks >= 64, heads-per-chip >= 1', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('KV_READ_PROBE watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    report = run(options, ttnn=ttnn, environ=environ, log=log)
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    log(json.dumps(report, sort_keys=True))
    timer.cancel()
    return {'PASS': 0, 'FAIL': 1}.get(report.get('verdict'), 2)


if __name__ == '__main__':
    sys.exit(main())
