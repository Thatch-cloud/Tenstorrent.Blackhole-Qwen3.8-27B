"""Check serving startup dependencies without opening devices or loading weights."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from sampling_link_policy import SOURCES


DEPENDENCIES = {
    'serving_startup': ('start', 'stop', 'warmup'),
    'serving_request_factory': ('device_components',),
    'dflash_combined_request': ('combined_runtime',),
    'dflash_prefill_window': ('PrefillWindowCapture',),
    'gdn_snapshot': ('ActiveSnapshot',),
    'full_dflash_request': ('load_dflash_fixtures',),
    'mlp_block_stream_pool': ('owned_streams',),
    'models.common.sampling.generator': ('SamplingGenerator',),
    'models.tt_transformers.tt.ccl': ('TT_CCL', 'tt_all_reduce'),
    'vllm.v1.worker.worker_base': ('CompilationTimes',),
}


def audit(root):
    checks, failures = {}, {}
    for name, expected in SOURCES.items():
        try:
            actual = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Startup source fingerprint mismatch: ' + actual)
            checks[name] = actual
        except Exception as error:
            failures[name] = f'{type(error).__name__}: {error}'
    for name, attributes in DEPENDENCIES.items():
        try:
            module = importlib.import_module(name)
            for attribute in attributes:
                if not callable(getattr(module, attribute)):
                    raise ValueError('Expected callable: ' + attribute)
            checks[name] = list(attributes)
        except Exception as error:
            failures[name] = f'{type(error).__name__}: {error}'
    try:
        from serving_request_factory import device_components

        components = device_components()
        if not all(callable(value) for value in vars(components).values()):
            raise ValueError('Request components must be callable')
        checks['request_components'] = sorted(vars(components))
    except Exception as error:
        failures['request_components'] = f'{type(error).__name__}: {error}'
    return dict(checks=checks, failures=failures, passed=not failures,
        devices_accessed=False, weights_loaded=False, serving_qualified=False,
        performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(arguments.root)
    arguments.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['passed'] else 1)
