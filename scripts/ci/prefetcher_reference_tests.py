"""Find upstream's own tests for the tensor prefetcher and dump a working config.

The design doc gives the contract but not a runnable geometry. tt-metal ships
tests that already drive start_tensor_prefetcher / test_dram_prefetcher_consumer;
copying a known-good configuration is worth more than guessing shard specs.
Read-only: greps the image and prints matching sources.
"""

import argparse
import json
from pathlib import Path

BEGIN = '<<<PREFETCH_TESTS_JSON_BEGIN>>>'
END = '<<<PREFETCH_TESTS_JSON_END>>>'
NEEDLES = ('start_tensor_prefetcher', 'test_dram_prefetcher_consumer',
           'test_dram_prefetcher_validator',
           'create_global_circular_buffer_for_tensor_prefetcher',
           'queue_tensor_prefetcher_request', 'tensor_prefetcher_matmul')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/opt/tt-metal')
    parser.add_argument('--max-bytes', type=int, default=60000)
    options = parser.parse_args()
    report = dict(scope=__doc__, hits=[], files={})
    root = Path(options.root)
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in ('.py', '.cpp', '.hpp', '.md'):
            continue
        try:
            text = path.read_text(errors='replace')
        except OSError:
            continue
        found = [n for n in NEEDLES if n in text]
        if not found:
            continue
        rel = str(path.relative_to(root))
        report['hits'].append(dict(path=rel, needles=found, size=len(text),
                                   is_test='test' in rel.lower()))
    # Prefer python tests: they carry the runnable geometry.
    ranked = sorted(report['hits'],
                    key=lambda h: (not (h['is_test'] and h['path'].endswith('.py')),
                                   -len(h['needles']), h['size']))
    budget = options.max_bytes
    for hit in ranked:
        if budget <= 0:
            break
        text = (root / hit['path']).read_text(errors='replace')
        take = text[:min(len(text), budget)]
        report['files'][hit['path']] = take
        budget -= len(take)
    report['hit_count'] = len(report['hits'])
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)


if __name__ == '__main__':
    main()
