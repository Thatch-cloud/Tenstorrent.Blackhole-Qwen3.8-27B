"""Retain all MLP marker rows and provenance without copying unrelated raw events."""

import hashlib
import json
from pathlib import Path
import sys


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Raw source must remain unchanged')
    before = source.stat()
    source_digest, selected_digest = hashlib.sha256(), hashlib.sha256()
    rows = selected = 0
    with source.open('rb') as incoming, destination.open('wb') as outgoing:
        architecture, columns = incoming.readline(), incoming.readline()
        if not architecture.startswith(b'ARCH: blackhole,') or b'zone name' not in columns:
            raise ValueError('Native Blackhole profiler header required')
        for header in (architecture, columns):
            source_digest.update(header)
            selected_digest.update(header)
            outgoing.write(header)
        for line in incoming:
            source_digest.update(line)
            rows += 1
            if b'QWEN_MLP_' in line:
                outgoing.write(line)
                selected_digest.update(line)
                selected += 1
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Raw capture changed during export')
    if not selected:
        raise ValueError('No MLP marker rows in raw capture')
    report = dict(source_sha256=source_digest.hexdigest(), selected_sha256=selected_digest.hexdigest(),
        source_bytes=before.st_size, source_rows=rows, selected_rows=selected,
        scope='All raw rows containing QWEN_MLP_; no pairing, replay or completeness filtering',
        diagnostic_only=True, committed_tg=None)
    destination.with_suffix('.selection.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    export(*sys.argv[1:])
