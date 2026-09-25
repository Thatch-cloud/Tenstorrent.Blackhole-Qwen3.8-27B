"""Stage synthetic clock qualification with the complete prefix-state replay matrix."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_multitoken import replace_once
from gdn_recurrence_clock_program import instrument_pipeline


HELPERS = ('gdn_recurrence_clock.py', 'gdn_recurrence_clock_capture.py',
    'mlp_compute_clock_projection.py', 'mlp_clock_samples.py',
    'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py')


def adapt(source):
    source = replace_once(source, 'from gdn_shared_qk_pipeline import build as build_pipeline',
        'from gdn_recurrence_clock_pipeline import build as build_pipeline, BUILD_RECORDS\n'
        '    from gdn_recurrence_clock_capture import RecurrenceClockCapture')
    source = replace_once(source,
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)',
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True,\n'
        '        recurrence_clock=True, generated_kernels=BUILD_RECORDS, committed_tg=None)')
    source = replace_once(source, '                try:\n                    def allocate(',
        '                try:\n'
        '                    self.samples = RecurrenceClockCapture(ttnn, mesh, self.owned)\n'
        '                    report["missing_execution_rejected"] = self.samples.reject_missing_execution()\n'
        '                    def allocate(')
    source = replace_once(source, 'self.programs = build_pipeline(ttnn, mesh, tensors, root=root)',
        'self.programs = build_pipeline(ttnn, mesh, tensors, root=root, samples=self.samples.buffer)')
    source = replace_once(source, "        save('eager')\n        reference, actual = control.run(), candidate.run()",
        "        save('eager')\n        candidate.samples.prepare()\n"
        '        reference, actual = control.run(), candidate.run()\n'
        "        candidate.samples.collect('eager')")
    source = replace_once(source, '            for trace in traces:\n',
        '            candidate.samples.prepare()\n            for trace in traces:\n')
    source = replace_once(source, "            check(reference, actual, host, 'replay_' + str(seed))",
        "            candidate.samples.collect('replay_' + str(seed))\n"
        "            check(reference, actual, host, 'replay_' + str(seed))")
    source = replace_once(source, "        report['passed'] = True",
        "        report['recurrence_clock_samples'] = candidate.samples.records\n"
        "        report['missing_execution_rejected_after_replay'] = candidate.samples.reject_missing_execution()\n"
        '        if len(BUILD_RECORDS) != 1 or len(candidate.samples.records) != 4:\n'
        "            raise AssertionError('One instrumented recurrence and four fresh samples required')\n"
        "        report['passed'] = True")
    compile(source, 'gdn-shared-recurrence-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh recurrence clock manifest required')
    scripts = options.checkout / 'scripts/ci'
    result = {}
    transforms = {}
    for before_name, after_name, transform in (
            ('gdn_shared_qk_pipeline.py', 'gdn_recurrence_clock_pipeline.py', instrument_pipeline),
            ('gdn-shared-recurrence-probe.py', 'gdn-shared-recurrence-probe.py', adapt)):
        original = (scripts / before_name).read_bytes()
        result[after_name] = transform(original.decode().replace('\r\n', '\n')).encode()
        transforms[after_name] = dict(before=hashlib.sha256(original).hexdigest(),
            after=hashlib.sha256(result[after_name]).hexdigest())
    for name in HELPERS:
        result[name] = Path(__file__).with_name(name).read_bytes()
    for name, data in result.items():
        (scripts / name).write_bytes(data)
    options.manifest.write_text(json.dumps(dict(transformations=transforms,
        sources={name: hashlib.sha256(data).hexdigest() for name, data in result.items()},
        simulator_qualified=False, hardware_qualified=False, committed_tg=None), indent=2) + '\n')


if __name__ == '__main__':
    main()
