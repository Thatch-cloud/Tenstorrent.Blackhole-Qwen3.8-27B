"""Stage bounded synthetic compute-clock checks, without target weights."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from mlp_clock_stage import trace_source, probe_source as reader_probe_source
from mlp_compute_clock_projection import instrument_projection


def probe_source(source):
    source = reader_probe_source(source)
    source = replace_once(source, 'from mlp_clock_capture import ClockCapture',
        'from mlp_compute_clock_capture import ComputeClockCapture as ClockCapture')
    source = replace_once(source, 'sample_buffers=sample_capture.buffers',
        'compute_samples=sample_capture.buffer')
    source = replace_once(source, "report['clock_samples'] = sample_capture.records",
        "report['compute_clock_samples'] = sample_capture.records")
    source = replace_once(source,
        'T16 bounded clock diagnostic; numerical replay only, no performance qualification',
        'T16 compute processor clock diagnostic; no hardware or throughput qualification')
    compile(source, 'compute-clock-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh compute-clock staging required')
    scripts = options.checkout / 'scripts/ci'
    result, transforms = {}, {}
    for name, transform in (('fused_1d.py', instrument_projection),
            ('fusion_trace.py', trace_source), ('fused-batch-probe.py', probe_source)):
        original = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != original:
            raise ValueError('Exact frozen source required: ' + name)
        result[name] = transform(original)
        transforms[name] = dict(before=hashlib.sha256(original.encode()).hexdigest(),
            after=hashlib.sha256(result[name].encode()).hexdigest())
    result['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name in ('mlp_compute_clock.py', 'mlp_compute_clock_projection.py',
            'mlp_compute_clock_capture.py', 'mlp_clock_samples.py',
            'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py'):
        result[name] = Path(__file__).with_name(name).read_text()
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(base=REVISION, transformations=transforms,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        simulator_qualified=False, hardware_qualified=False, committed_tg=None), indent=2) + '\n')


if __name__ == '__main__':
    main()
