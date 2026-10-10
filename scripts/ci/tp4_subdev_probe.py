"""The four-card sub-device collective probe H2 (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=subdev, C2_CARDS=quad).

    python3 -B /c2/scripts/ci/tp4_subdev_probe.py --fabric FABRIC_1D --output /probe-results/subdev-probe.json [--no-timing] [--watcher]

Can an all_gather on the target sub-device and an all_gather on the drafter sub-device be in flight together on the served (1, 4) ring? The arms,
the exactness rule and the verdict are optimisation/ttnn-op/subdev_h/subdev_h2.py's (its docstring has them); this wrapper opens the served mesh
the way the fabric probe and the mr probe do (the ring descriptor, the fabric config, ONE open, ONE close), with TWO command queues, under a per-call
watchdog and a heartbeat. The probe holds all four cards: run it LAST in a window and follow it with an all-board reset.

--watcher: TT_METAL_WATCHER=5 and TT_METAL_WATCHER_DISABLE_ETH=1 are set before ttnn is imported (the watcher pass: bytes and hangs only, so it implies
--no-timing) and the watcher log is copied beside the report with a count of its error lines. The second variable is not optional: with the watcher on the
ethernet cores the fabric router does not fit the ACTIVE_ETH kernel config buffer and the mesh does not open (see WATCHER_ENV).

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
# The watcher instruments every kernel it builds, the fabric router's ethernet kernels included, and with it on the router no longer fits the
# ACTIVE_ETH kernel config buffer (a fixed 25 KiB of the HAL, whatever sub-device manager is loaded): open_mesh_device dies in the fabric init with
# 'Program size (28256) too large for kernel config buffer (25600) on ACTIVE_ETH' before the first sub-device exists (run 38044011709). The ethernet
# switch compiles the instrumentation out of ethernet kernels only (CreateEthernetKernel adds FORCE_WATCHER_OFF); worker cores, where the gathers' workers
# and muxes run, stay watched.
WATCHER_ENV = (('TT_METAL_WATCHER', WATCHER_LEVEL), ('TT_METAL_WATCHER_DISABLE_ETH', '1'))
ETH_OVERFLOW = 'too large for kernel config buffer'
WATCHER_ERRORS = re.compile(r'error|assert|tripped|sanitiz|hang|stuck', re.I)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--watcher', action='store_true')
    subdev_h2.add_arguments(parser)
    return parser


def apply_watcher_env(options, environ):
    """Put the watcher variables in `environ` for a --watcher run (before ttnn is imported: the runtime reads them once, at initialisation) and
    return what the run will see: {name: value or None} for each of WATCHER_ENV."""
    if options.watcher:
        for name, value in WATCHER_ENV:
            environ[name] = value
    return {name: environ.get(name) for name, _ in WATCHER_ENV}


def known_failure(text):
    """What a traceback means when this project has seen it before, else None."""
    if ETH_OVERFLOW in (text or '') and 'ACTIVE_ETH' in (text or ''):
        return ('the fabric router (an ACTIVE_ETH kernel) does not fit its kernel config buffer: the watcher is on the ethernet cores; '
                'TT_METAL_WATCHER_DISABLE_ETH=1 compiles it out of ethernet kernels')
    return tp4_fabric_probe.known_signature(text)


GRAFT_ENV = 'QWEN_AG_LINK_GRAFT_SHA256'
GRAFT_FILES = ('/opt/tt-metal/build_Release/ttnn/_ttnncpp.so', '/opt/tt-metal/build_Release/lib/_ttnncpp.so')
GRAFT_MARKER = subdev_h2.LINK_ENV.encode()


def file_sha256(path, chunk=1 << 22):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def file_contains(path, needle, chunk=1 << 24):
    """Whether the file holds `needle`, read in blocks that overlap by the needle's length."""
    keep = b''
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            window = keep + block
            if needle in window:
                return True
            keep = window[-(len(needle) - 1):]
    return False


def verify_graft(environ, files=GRAFT_FILES):
    """Is the link-offset graft what this container will load? Returns dict(ok, problems, expected, binaries={path: sha}, marker={path: bool}).
    The workflow mounts the graft's _ttnncpp.so over the image's two copies and exports its sha256 as QWEN_AG_LINK_GRAFT_SHA256; here each copy must hash to
    exactly that and carry the literal the patched factory reads. A graft mounted is not a graft executed: loaded_graft() checks the mapped libraries too."""
    expected = environ.get(GRAFT_ENV, '')
    found = dict(expected=expected, binaries={}, marker={}, problems=[])
    if not expected:
        found['problems'].append('%s is not set: the fabric step mounts the graft and exports its sha256 (probe subdev-links)' % GRAFT_ENV)
    for path in files:
        try:
            found['binaries'][path] = file_sha256(path)
            found['marker'][path] = file_contains(path, GRAFT_MARKER)
        except OSError as error:
            found['problems'].append('%s: %s' % (path, error))
            continue
        if expected and found['binaries'][path] != expected:
            found['problems'].append('%s is %s, not the graft %s' % (path, found['binaries'][path][:16], expected[:16]))
        if not found['marker'][path]:
            found['problems'].append('%s does not contain %s: it is not the link-offset graft' % (path, subdev_h2.LINK_ENV))
    found['ok'] = not found['problems']
    return found


def loaded_graft(found, maps_text):
    """After `import ttnn`: every _ttnncpp.so the process has mapped (from /proc/self/maps) is one of the verified copies or hashes to the graft's sha."""
    mapped = sorted({line.split()[-1] for line in maps_text.splitlines() if line.split() and line.split()[-1].endswith('_ttnncpp.so')})
    found['mapped'] = mapped
    for path in mapped:
        if path in found['binaries']:
            continue
        try:
            digest = file_sha256(path)
        except OSError as error:
            found['problems'].append('mapped %s: %s' % (path, error))
            continue
        found['binaries'][path] = digest
        if digest != found['expected']:
            found['problems'].append('the process mapped %s (%s), which is not the graft %s' % (path, digest[:16], found['expected'][:16]))
    if not mapped:
        found['problems'].append('no _ttnncpp.so is mapped after the import: the graft check cannot see what was loaded')
    found['ok'] = not found['problems']
    return found


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


def run(options, ttnn=None, torch=None, environ=None, log=plan.say, watchdog=None, heartbeat=None, clock=None, graft_files=GRAFT_FILES):
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

    held = {}

    def on_fire(label):
        report.update(verdict='HANG', error='watchdog: %s exceeded its budget' % label)
        if held.get('harness') is not None:
            held['harness'].snapshot()      # every arm measured before the hang goes into the report
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
    report['watcher_env'] = apply_watcher_env(options, environ)
    harness, mesh, error = None, None, None
    try:
        graft = None
        if options.link_offset > 0:
            # the separate-links arm means nothing on a binary without the graft: check the files BEFORE anything is imported or opened
            graft = report['graft'] = verify_graft(environ, graft_files)
            if not graft['ok']:
                raise RuntimeError('the link-offset graft is not what is mounted: ' + '; '.join(graft['problems']))
        if ttnn is None:
            import torch  # noqa: F811
            import ttnn  # noqa: F811
            if graft is not None:
                with open('/proc/self/maps') as handle:
                    loaded_graft(graft, handle.read())
                if not graft['ok']:
                    raise RuntimeError('the link-offset graft is not what is loaded: ' + '; '.join(graft['problems']))
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        with watchdog.span('open-mesh', options.compile_watchdog_s):
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576,
                                         trace_region_size=options.trace_region_bytes, num_command_queues=2,
                                         dispatch_core_config=ttnn.DispatchCoreConfig())
        report['opened'] = True
        mesh.enable_program_cache()      # a trace can only capture programs that already ran: they come from the program cache
        harness = subdev_h2.Harness2(ttnn, torch, mesh, options, watchdog, heartbeat=heartbeat, log=log, clock=clock, report=report,
                                     persist=persist, environ=environ)
        held['harness'] = harness
        harness.run()
    except BaseException as caught:  # noqa: BLE001
        error = '%s: %s' % (type(caught).__name__, ' '.join(str(caught).split())[:500])
        report['error'] = error
        report['traceback'] = traceback.format_exc()[-3000:]
        report['known_failure'] = known_failure(report['traceback'] + ' ' + error)
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


def main(argv=None, ttnn=None, torch=None, environ=None, log=plan.say, watchdog=None, heartbeat=None, clock=None, graft_files=GRAFT_FILES):
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
        _, status = run(options, ttnn=ttnn, torch=torch, environ=environ, log=log, watchdog=watchdog, heartbeat=beat, clock=clock,
                        graft_files=graft_files)
    finally:
        if beat is not None:
            beat.stop()
    return status


if __name__ == '__main__':
    sys.exit(main())
