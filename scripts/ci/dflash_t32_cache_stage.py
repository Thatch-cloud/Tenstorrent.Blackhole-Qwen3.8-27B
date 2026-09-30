"""Stage a bounded, weight-free cache and native proposal replay integration."""

import argparse
import hashlib
import json
from pathlib import Path

from dflash_t32_cached_adapter import adapt_sources


def payloads(directory):
    directory = Path(directory)
    sources = adapt_sources({name: (directory / name).read_text() for name in
        ('dflash_device.py', 'draft_attention_branch.py', 'dflash_proposal_trace.py')})
    suite = (directory / 'simulator-suite.sh').read_text()
    start = suite.index('    for context in 31 2048; do')
    end = suite.index('    exit 0\nfi', start)
    suite = suite[:start] + '''    status=0
    timeout -k 15 360 python3 -B -u /experiment-scripts/ci/dflash-t32-cache-replay-probe.py \\
        --output /experiment/results/dflash-t32-cache.json || status=$?
    printf '%s\\n' "$status" > /experiment/results/dflash-t32-cache.exit-status
    test "$status" = 0
    python3 -B /experiment-scripts/ci/dflash_t32_cache_gate.py \\
        --report /experiment/results/dflash-t32-cache.json
''' + suite[end:]
    sources['simulator-suite.sh'] = suite
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh staging required')
    directory = Path(__file__).parent
    sources = payloads(directory)
    for name, source in sources.items():
        (directory / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(learned_weights=False, hardware_qualified=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}), indent=2) + '\n')


if __name__ == '__main__':
    main()
