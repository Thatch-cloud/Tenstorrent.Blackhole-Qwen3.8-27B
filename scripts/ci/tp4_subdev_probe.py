"""The four-card sub-device collective probe H2 (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=subdev, C2_CARDS=quad).

    python3 -B /c2/scripts/ci/tp4_subdev_probe.py --fabric FABRIC_1D --output /probe-results/subdev-probe.json [--no-timing] [--watcher]

Can an all_gather on the target sub-device and an all_gather on the drafter sub-device be in flight together on the served (1, 4) ring? The arms,
the exactness rule and the verdict are optimisation/ttnn-op/subdev_h/subdev_h2.py's (its docstring has them); this wrapper opens the served mesh
the way the fabric probe and the mr probe do (the ring descriptor, the fabric config, ONE open, ONE close), with TWO command queues, under a per-call
watchdog and a heartbeat. The probe holds all four cards: run it LAST in a window and follow it with an all-board reset.

--watcher: TT_METAL_WATCHER=5 is set before ttnn is imported (the watcher pass: bytes and hangs only, so it implies --no-timing) and the watcher
log is copied beside the report with a count of its error lines.

Last stdout lines: 'SUBDEV_H2 verdict=...' then one JSON object (kind subdev-h2). Exit: 0 PASS, PASS-SEPARATE-LINKS or UNTIMED-PASS, 1 FAIL, 2 NOT-MEASURED
(the mesh did not open, the descriptor is not the ring's), 3 the watchdog.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'subdev_h'))

import subdev_h2  # noqa: E402
import subdev_plan as plan  # noqa: E402
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

KIND = subdev_h2.KIND
WATCHER_LEVEL = '5'
WATCHER_ERRORS = re.compile(r'error|assert|tripped|sanitiz|hang|stuck', re.I)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--watcher', action='store_true')
    subdev_h2.add_arguments(parser)
    return parser


def watcher_summary(output, environ, home=None):
    """Copy the watcher log beside `output` and count its error lines: {'log': path, 'lines': n, 'errors': n} or {'log': None}."""
    root = home or environ.get('TT_METAL_HOME', '/opt/tt-metal')
    source = Path(root) / 'generated' / 'watcher' / 'watcher.log'
    if not source.is_file():
        return dict(log=None)
    target = Path(output).with_name('subdev-watcher.log')
    try:
        shutil.copyfile(str(source), str(target))
    except OSError as error:
        return dict(log=None, error=str(error))
    lines = source.read_text(errors='replace').splitlines()
    bad = [line for line in lines if WATCHER_ERRORS.search(line)]
    return dict(log=target.name, lines=len(lines), errors=len(bad), first=bad[:5])


def run(options, ttnn=None, torch=None, environ=None, log=plan.say, watchdog=None, heartbeat=None, clock=None):
    """The measurement; returns (report, exit status). `ttnn` and `torch` are the modules (fakes in the tests)."""
    import time
    environ = os.environ if environ is None else environ
    clock = clock or time.perf_counter_ns
    report = dict(kind=KIND, fabric=options.fabric, opened=False, descriptor=environ.get('TT_MESH_GRAPH_DESC_PATH'),
                  serving_contract=environ.get('QWEN_C2_SERVING'), watcher=bool(options.watcher),
                  options={key: value for key, value in sorted(vars(options).items())})

    def persist():
        try:
            with open(options.output, 'w') as handle:
                json.dump(report, handle, indent=2, sort_keys=True, default=str)
        except OSError:
            pass

    def on_fire(label):
        report.update(verdict='HANG', error='watchdog: %s exceeded its budget' % label)
        persist()

    if watchdog is None:
        watchdog = plan.Watchdog(tag=subdev_h2.TAG, on_fire=on_fire).start()
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
            log('%s verdict=NOT-MEASURED error="%s"' % (subdev_h2.TAG, report['error']))
            return report, 2
        if options.watcher:
            environ['TT_METAL_WATCHER'] = WATCHER_LEVEL    # before ttnn is imported: the runtime reads it once, at initialisation
    harness, mesh, error = None, None, None
    try:
        if ttnn is None:
            import torch  # noqa: F811
            import ttnn  # noqa: F811
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        with watchdog.span('open-mesh', options.compile_watchdog_s):
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576,
                                         trace_region_size=options.trace_region_bytes, num_command_queues=2,
                                         dispatch_core_config=ttnn.DispatchCoreConfig())
        report['opened'] = True
        mesh.enable_program_cache()      # a trace can only capture programs that already ran: they come from the program cache
        harness = subdev_h2.Harness2(ttnn, torch, mesh, options, watchdog, heartbeat=heartbeat, log=log, clock=clock, report=report,
                                     persist=persist, environ=environ)
        harness.run()
    except BaseException as caught:  # noqa: BLE001
        error = '%s: %s' % (type(caught).__name__, ' '.join(str(caught).split())[:500])
        report['error'] = error
        report['traceback'] = traceback.format_exc()[-3000:]
        report['known_failure'] = tp4_fabric_probe.known_signature(report['traceback'])
    finally:
        if harness is not None:
            harness.teardown()
        if mesh is not None:
            try:
                with watchdog.span('close-mesh', options.compile_watchdog_s):
                    ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except BaseException as caught:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(caught).__name__, ' '.join(str(caught).split())[:300])
    text, evidence, line = subdev_h2.finish(harness, report, options, error)
    if options.watcher:
        report['watcher_log'] = watcher_summary(options.output, environ)
        line += ' watcher_errors=%s' % report['watcher_log'].get('errors', '-')
    log(line)
    persist()
    log(json.dumps(report, sort_keys=True, default=str))
    return report, plan.exit_status(text)


def main(argv=None, ttnn=None, torch=None, environ=None, log=plan.say, watchdog=None, heartbeat=None, clock=None):
    options = build_parser().parse_args(argv)
    if options.watcher:
        options.no_timing = True
    problems = subdev_h2.problems_of(options)
    if problems:
        print('refusing: ' + '; '.join(problems), file=sys.stderr)
        return 2
    beat = heartbeat
    if beat is None and options.heartbeat_s:
        beat = plan.Heartbeat(subdev_h2.TAG, period=options.heartbeat_s).start()
    try:
        _, status = run(options, ttnn=ttnn, torch=torch, environ=environ, log=log, watchdog=watchdog, heartbeat=beat, clock=clock)
    finally:
        if beat is not None:
            beat.stop()
    return status


if __name__ == '__main__':
    sys.exit(main())
