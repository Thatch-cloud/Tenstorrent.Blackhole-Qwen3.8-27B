"""Stage the admitted recurrence diagnostic without changing the winning recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from gdn_multitoken import replace_once
from gdn_recurrence_clock_experiment import FILES
from gdn_recurrence_clock_gate import retained
from gdn_recurrence_clock_program import instrument_pipeline


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True),)")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from gdn_recurrence_clock_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_GDN_RECURRENCE_CLOCK_COMBINED=${QWEN_GDN_RECURRENCE_CLOCK_COMBINED:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    if options.manifest.exists() or (scripts / 'recurrence-clock-evidence').exists():
        raise ValueError('Fresh combined recurrence clock staging required')
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: data.decode() for name, data in original.items()})
    result['gdn_recurrence_clock_pipeline.py'] = instrument_pipeline(
        (scripts / 'gdn_shared_qk_pipeline.py').read_text(encoding='utf-8'))
    for name in FILES:
        if name not in result:
            result[name] = Path(__file__).with_name(name).read_text(encoding='utf-8')
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(options.evidence, scripts / 'recurrence-clock-evidence')
    retained(scripts / 'recurrence-clock-evidence', scripts)
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(data).hexdigest() for name, data in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        combined_qualified=False, performance_qualified=False, diagnostic_only=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
