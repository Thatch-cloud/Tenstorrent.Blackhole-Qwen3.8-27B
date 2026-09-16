"""Reject heavily contended benchmark starts; this is not performance acceptance."""

import argparse
import json
import math
from pathlib import Path
import time


def full_total(text):
    rows = [row.split() for row in text.splitlines() if row.startswith('full ')]
    if len(rows) != 1:
        raise ValueError('One full I/O pressure counter required')
    fields = dict(field.split('=', 1) for field in rows[0][1:])
    total = int(fields['total'])
    if total < 0:
        raise ValueError('Nonnegative pressure counter required')
    return total


def evaluate(before, after, elapsed_seconds, maximum_fraction=0.01):
    if (not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or not math.isfinite(maximum_fraction) or not 0 <= maximum_fraction <= 1):
        raise ValueError('Finite duration and pressure fraction required')
    delta = full_total(after) - full_total(before)
    if delta < 0:
        raise ValueError('Pressure counter reset during observation')
    fraction = delta / (elapsed_seconds * 1_000_000)
    return dict(passed=fraction <= maximum_fraction, full_io_stall_fraction=fraction,
        elapsed_seconds=elapsed_seconds, maximum_fraction=maximum_fraction,
        full_io_stall_microseconds=delta, before=before, after=after,
        scope='Pre-load host pressure only; no proof of subsequent isolation or benchmark quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh pressure report required')
    try:
        pressure = Path('/proc/pressure/io')
        before = pressure.read_text()
        started = time.monotonic()
        time.sleep(15)
        after = pressure.read_text()
        report = evaluate(before, after, time.monotonic() - started)
    except Exception as error:
        report = dict(passed=False, error=f'{type(error).__name__}: {error}')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['passed'] else 75)


if __name__ == '__main__':
    main()
