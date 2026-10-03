"""Independent simulator phase budgets, with durable start and completion evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def run_phase(phase, seconds, output, command):
    if phase not in ('prepare', 'probe') or type(seconds) is not int or seconds <= 0 or not command:
        raise ValueError('Explicit bounded simulator phase required')
    output = Path(output)
    if output.exists():
        raise ValueError('Fresh phase timing evidence required')
    report = dict(phase=phase, budget_seconds=seconds, status='running',
        elapsed_seconds=None, exit_code=None)
    output.write_text(json.dumps(report) + '\n')
    print(json.dumps(report), flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(['timeout', '-k', '15', str(seconds), *command], check=False)
        report.update(status='completed', exit_code=completed.returncode,
            timed_out=completed.returncode == 124,
            killed_or_timeout=completed.returncode == 137)
        return completed.returncode
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ['--'] else options.command
    raise SystemExit(run_phase(options.phase, options.seconds, options.output, command))


if __name__ == '__main__':
    main()
