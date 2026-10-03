"""Stage the simulator-qualified assembly probe for full-size physical measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import math
import statistics

from frozen_draft_tail_gate import qualify, SOURCES
from frozen_recipe_context import replace_once


SIMULATOR_SHA256 = '57ef37a3f2fbc44943e33c0e636e96afedd453bf992c60b9e5b3c030535af279'


def validate_hardware(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'hardware' or report.get('performance_qualified') is not False
            or report.get('model_integrated') is not False
            or report.get('simulator_report_sha256') != SIMULATOR_SHA256):
        raise ValueError('Closed component-only hardware report with pinned simulator required')
    expected = set()
    for position in (33024, 33056):
        for proposals in (7, 15):
            for chip in (0, 1):
                for name in ('reference_eager', 'candidate_eager', 'reference_replay',
                        'candidate_replay', 'history_unchanged', 'queries_unchanged', 'reference_timed', 'candidate_timed'):
                    seeds = (0,) if name.endswith('_eager') else (2,) if name.endswith('_timed') else (1, 0, 2)
                    expected.update((position, proposals, chip, name, seed) for seed in seeds)
    actual = []
    for check in report.get('checks', []):
        if check.get('exact') is not True or any(type(check.get(name)) is not int
                for name in ('position', 'proposals', 'chip', 'seed')):
            raise ValueError('Exact typed hardware check required')
        actual.append(tuple(check.get(name) for name in ('position', 'proposals', 'chip', 'name', 'seed')))
    if len(actual) != 128 or set(actual) != expected:
        raise ValueError('Complete unique 128-check hardware matrix required')
    expected_order = [(position, proposals, repeat, arm) for position in (33024, 33056)
        for proposals in (7, 15) for repeat in range(4)
        for arm in ('reference', 'candidate', 'candidate', 'reference')]
    timings = report.get('component_timings', [])
    if [tuple(row.get(name) for name in ('position', 'proposals', 'repeat', 'arm'))
            for row in timings] != expected_order:
        raise ValueError('Complete ordered ABBA component timing required')
    if any(type(row.get(name)) is not int for row in timings for name in ('position', 'proposals', 'repeat')):
        raise ValueError('Integer component timing identities required')
    if any(type(row.get('elapsed_ms')) not in (int, float) or not math.isfinite(row['elapsed_ms'])
            or row['elapsed_ms'] <= 0 for row in timings):
        raise ValueError('Positive finite blocking replay measurements required')
    summary = []
    for position in (33024, 33056):
        for proposals in (7, 15):
            medians = {arm: statistics.median(row['elapsed_ms'] for row in timings
                if row['position'] == position and row['proposals'] == proposals and row['arm'] == arm)
                for arm in ('reference', 'candidate')}
            summary.append(dict(position=position, proposals=proposals, median_ms=medians))
    return dict(passed=True, component_only=True, committed_tg=None, cases=summary)


def adapt(source):
    source = replace_once(source, 'import os\n', 'import os\nimport time\n')
    source = replace_once(source,
        "    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')\n"
        "            or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1'\n"
        "            or Path('/dev/tenstorrent').exists() or options.output.exists()):\n"
        "        raise ValueError('Fresh device-free simulator with pinned packer compatibility required')",
        "    if (os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'\n"
        "            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('QWEN_SIM_ONLY') == '1'\n"
        "            or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT')\n"
        "            or not Path('/dev/tenstorrent').exists() or options.output.exists()):\n"
        "        raise ValueError('Fresh allocated physical cards without simulator grafts required')\n"
        "    from frozen_draft_tail_gate import qualify\n"
        f"    qualify(Path(__file__).parent, '/experiment/results/simulator', '{SIMULATOR_SHA256}')\n"
        "    import dspark_full_attention\n"
        "    dspark_full_attention.MAX_CONTEXT = 33056")
    source = replace_once(source, '    checks, owned, traces = [], [], []\n',
        '    checks, owned, traces = [], [], []\n    timings = []\n')
    source = replace_once(source, "backend='simulator', checks=checks,",
        "backend='hardware', checks=checks, component_timings=timings,\n"
        f"        simulator_report_sha256='{SIMULATOR_SHA256}',\n"
        "        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),")
    source = replace_once(source, '        for position in (64, 96):',
        '        for position in (33024, 33056):')
    source = replace_once(source,
        "                    compare(device_queries, queries, 'queries_unchanged', position, proposals, seed)\n"
        '                ttnn.synchronize_device(mesh)\n                for trace in traces:',
        "                    compare(device_queries, queries, 'queries_unchanged', position, proposals, seed)\n"
        "                by_name = {name: (trace, output) for name, trace, output in outputs}\n"
        "                for repeat in range(4):\n"
        "                    for name in ('reference', 'candidate', 'candidate', 'reference'):\n"
        "                        ttnn.synchronize_device(mesh)\n"
        "                        started = time.perf_counter()\n"
        "                        ttnn.execute_trace(mesh, by_name[name][0], blocking=True)\n"
        "                        elapsed_ms = (time.perf_counter() - started) * 1000\n"
        "                        timings.append(dict(position=position, proposals=proposals, arm=name,\n"
        "                            repeat=repeat, elapsed_ms=elapsed_ms))\n"
        "                for name, trace, output in outputs:\n"
        "                    compare(output, expected, name + '_timed', position, proposals, 2)\n"
        '                ttnn.synchronize_device(mesh)\n                for trace in traces:')
    source = replace_once(source, "len(checks) == 112", "len(checks) == 128 and len(timings) == 64")
    compile(source, 'draft-tail-hardware-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    options = parser.parse_args()
    qualify(options.directory, options.evidence, SIMULATOR_SHA256)
    output = options.directory / 'draft-tail-hardware-probe.py'
    if output.exists():
        raise ValueError('Fresh staging path required')
    output.write_bytes(adapt((options.directory / 'draft-tail-probe.py').read_text()).encode())
    sources = {name: hashlib.sha256((options.directory / name).read_bytes()).hexdigest()
        for name in (*SOURCES, output.name)}
    (options.directory / 'draft-tail-hardware-sources.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
