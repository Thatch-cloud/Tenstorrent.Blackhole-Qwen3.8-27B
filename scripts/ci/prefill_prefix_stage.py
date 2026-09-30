"""Apply only the tested cache hook to a separately staged frozen T16 ladder."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from prefill_prefix_experiment import SOURCES


def function_source(source, name):
    matches = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError('Exactly one function required: ' + name)
    node = matches[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def adapt_full_request(source, tested):
    if 't32' in source or 'cached_prefill_factory' in source:
        raise ValueError('Unmodified frozen T16 request required')
    source = replace_once(source, 'combined_profile=False, gdn_shared_qk=False):',
        'combined_profile=False, gdn_shared_qk=False, cached_prefill_factory=None):')
    source = replace_once(source, '    from full_request import measure_request\n',
        '    from full_request import measure_request\n'
        '    if cached_prefill_factory is not None and not callable(cached_prefill_factory):\n'
        "        raise ValueError('Explicit T16 cached-prefill factory required')\n")
    source = replace_once(source,
        '    prefill_records, feature_checks, history_checks, proposal_checks = [], [], [], []\n',
        '    prefill_records, feature_checks, history_checks, proposal_checks = [], [], [], []\n'
        '    cache_context, cache_records = ExitStack(), []\n')
    source = replace_once(source, function_source(source, 'captured_prefill'),
        function_source(tested, 'captured_prefill'))
    source = replace_once(source, "        result['instrumented_timing'] = audit_features\n",
        '        if cached_prefill_factory is not None:\n'
        '            if len(cache_records) != 2:\n'
        "                raise ValueError('Both cold-control and cached-candidate prefill records required')\n"
        "            result['dspark']['prefix_cache'] = cache_records\n"
        "        result['instrumented_timing'] = audit_features\n")
    source = replace_once(source, '                if capture is not None:\n                    capture.close()\n',
        '                if cached_prefill_factory is not None:\n'
        '                    import sys\n'
        '                    cache_context.__exit__(*sys.exc_info())\n'
        '                elif capture is not None:\n                    capture.close()\n')
    compile(source, 'full_dspark_request.py', 'exec')
    return source


def stage(checkout, manifest):
    scripts, manifest = Path(checkout) / 'scripts/ci', Path(manifest)
    if manifest.exists() or (scripts / 'prefill_prefix_experiment.py').exists():
        raise ValueError('Fresh frozen prefix-cache staging required')
    names = ('full_dspark_request.py', 'dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    originals = {name: (scripts / name).read_bytes() for name in names}
    sources = {name: value.decode().replace('\r\n', '\n') for name, value in originals.items()}
    if '    from frozen_ladder_requests import finish\n    finish(report, summarize)' not in sources['dspark_request_experiment.py']:
        raise ValueError('Fresh complete-context ladder staging required')
    sources['full_dspark_request.py'] = adapt_full_request(sources['full_dspark_request.py'],
        Path(__file__).with_name('full_dspark_request.py').read_text())
    sources['dspark_request_experiment.py'] = replace_once(sources['dspark_request_experiment.py'],
        function_source(sources['dspark_request_experiment.py'], 'summarize'),
        function_source(Path(__file__).with_name('dspark_request_experiment.py').read_text(), 'summarize'))
    sources['dspark-target-hardware.py'] = replace_once(sources['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from prefill_prefix_experiment import run_loaded_requests')
    sources['run-dspark-hardware.sh'] = replace_once(sources['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_PREFIX_CACHE_EXPERIMENT=${QWEN_PREFIX_CACHE_EXPERIMENT:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in sources.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    payloads = {name: value.encode() for name, value in sources.items()}
    for name in SOURCES:
        if name not in payloads:
            payloads[name] = Path(__file__).with_name(name).read_bytes()
    for name, payload in payloads.items():
        (scripts / name).write_bytes(payload)
    manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(value).hexdigest() for name, value in originals.items()},
        after={name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        context=4096, prefix_tokens=2048, hardware_qualified=False, serving_enabled=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    arguments = parser.parse_args()
    stage(arguments.checkout, arguments.manifest)
