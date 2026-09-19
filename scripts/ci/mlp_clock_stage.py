"""Stage a bounded T16 simulator replay diagnostic without full model loading."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_mlp_buffer_trial import adapt_probe
from frozen_recipe_context import REVISION, replace_once
from mlp_clock_projection import instrument_projection


def trace_source(source):
    source = replace_once(source, '*, timing=False):', '*, timing=False, sample_capture=None):')
    lines = source.splitlines(keepends=True)
    count = 0
    result = []
    for line in lines:
        if line.strip() == 'operations.execute_trace(mesh, trace, cq_id=0, blocking=True)':
            indent = line[:len(line) - len(line.lstrip())]
            result.extend([indent + "if name == 'fused':\n",
                indent + '    sample_capture.prepare()\n', line,
                indent + "if name == 'fused':\n",
                indent + "    sample_capture.collect(f'replay-{repetition}')\n"])
            count += 1
        else:
            result.append(line)
    if count != 2:
        raise ValueError('Exact two untimed replay sites required')
    result = ''.join(result)
    return replace_once(result, '    import torch\n',
        '    import torch\n'
        '    if timing or sample_capture is None:\n'
        "        raise ValueError('Untimed diagnostic with persistent sample capture required')\n")


def probe_source(source):
    source = adapt_probe(source)
    changes = (
        ('        mesh.enable_program_cache()\n',
         '        mesh.enable_program_cache()\n'
         '        from mlp_clock_capture import ClockCapture\n'
         '        sample_capture = ClockCapture(ttnn, mesh, owned)\n'
         "        report['missing_execution_rejected'] = sample_capture.reject_missing_execution()\n"),
        ("source_root=os.environ['TT_METAL_HOME'], math_approx_mode=options.target_math)",
         "source_root=os.environ['TT_METAL_HOME'], math_approx_mode=options.target_math,\n"
         '                sample_buffers=sample_capture.buffers)'),
        ('            actual = operation(inputs)\n',
         '            sample_capture.prepare()\n'
         '            actual = operation(inputs)\n'),
        ('            owned.append(actual)\n',
         '            owned.append(actual)\n'
         "            sample_capture.collect('eager')\n"),
        ('(device_gate, device_up, device_packed), timing=options.timing)',
         '(device_gate, device_up, device_packed), timing=options.timing, sample_capture=sample_capture)'),
        ("    report['passed'] = True",
         "    if len(sample_capture.records) != 5 or report.get('missing_execution_rejected') is not True:\n"
         "        raise AssertionError('Fresh eager and all four replay sample sets required')\n"
         "    report['clock_samples'] = sample_capture.records\n"
         "    report['committed_tg'] = None\n"
         "    report['passed'] = True"),
        ("'T16 buffering only; other row widths and performance unqualified'",
         "'T16 bounded clock diagnostic; numerical replay only, no performance qualification'"),
    )
    for before, after in changes:
        source = replace_once(source, before, after)
    compile(source, 'fused-batch-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh manifest required')
    scripts = options.checkout / 'scripts/ci'
    transforms = {'fused_1d.py': instrument_projection,
        'fusion_trace.py': trace_source, 'fused-batch-probe.py': probe_source}
    candidates, hashes = {}, {}
    for name, transform in transforms.items():
        original = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != original:
            raise ValueError('Exact frozen source required: ' + name)
        candidates[name] = transform(original)
        hashes[name] = dict(before=hashlib.sha256(original.encode()).hexdigest(),
            after=hashlib.sha256(candidates[name].encode()).hexdigest())
    candidates['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name in ('mlp_clock_capture.py', 'mlp_clock_projection.py', 'mlp_clock_samples.py',
                 'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py'):
        candidates[name] = Path(__file__).with_name(name).read_text()
    for name, source in candidates.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(base=REVISION, transformations=hashes,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in candidates.items()},
        simulator_qualified=False, hardware_qualified=False, committed_tg=None), indent=2) + '\n')


if __name__ == '__main__':
    main()
