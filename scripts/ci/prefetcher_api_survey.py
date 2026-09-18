"""Enumerate the prefetcher surface the pinned runtime exposes, without running one.

The repo has only ever called is_tensor_prefetcher_supported. Qualification needs
to know what else exists: the op that consumes a prefetched tensor, the global
circular buffer it feeds, and the signatures of both. Introspection only — this
opens no device and launches no kernel.
"""

import argparse
import inspect
import json
from pathlib import Path

TERMS = ('prefetch', 'global_cb', 'globalcircular', 'global_circular', 'dram_prefetch', 'drisc')


def describe(value):
    entry = dict(kind=type(value).__name__)
    doc = inspect.getdoc(value)
    if doc:
        entry['doc'] = doc[:600]
    try:
        entry['signature'] = str(inspect.signature(value))
    except (TypeError, ValueError):
        pass
    return entry


def survey(module, prefix, seen, depth=0):
    found = {}
    for name in sorted(dir(module)):
        if name.startswith('_'):
            continue
        path = '%s.%s' % (prefix, name)
        try:
            value = getattr(module, name)
        except BaseException:
            continue
        if any(term in name.lower() for term in TERMS):
            found[path] = describe(value)
        if depth < 1 and inspect.ismodule(value) and id(value) not in seen:
            seen.add(id(value))
            found.update(survey(value, path, seen, depth + 1))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = dict(scope=__doc__, device_opened=False, kernel_launched=False)
    try:
        import ttnn
        report['ttnn_version'] = getattr(ttnn, '__version__', 'unknown')
        seen = set()
        matches = {}
        for prefix, module in (('ttnn', ttnn), ('ttnn.experimental', getattr(ttnn, 'experimental', None))):
            if module is not None:
                matches.update(survey(module, prefix, seen))
        report['matches'] = matches
        report['match_count'] = len(matches)
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
    options.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:12000])


if __name__ == '__main__':
    main()
