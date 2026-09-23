"""Run qwen-integration-cpu.yml's whole unittest list locally, on CI's interpreter.

Why this exists: the workflow installs python 3.11 and torch 2.14, and the default
interpreter on the development machine is 3.7. A large part of this suite cannot even
be imported under 3.7 - zip(strict=), ast end_lineno, f-string = specifiers - so a
local `python -m pytest scripts/ci` silently skips over exactly the modules most
likely to break. Worse, the CPU workflow triggers on branch pushes only, not on the
experiment tags the rig lanes use, so a stretch of tag-driven work can leave it unrun
for days: it last went green at cfd3cfc1 and the next push, 465 commits later,
surfaced five failures at once (see the commits of 2026-09-22).

Usage, from the repository root:

    py -3.11 scripts/ci/run_cpu_suite_locally.py              # everything
    py -3.11 scripts/ci/run_cpu_suite_locally.py test_gdn_user_batch   # just those

Requires torch and pyyaml in the chosen interpreter, the same as the workflow:

    py -3.11 -m pip install pyyaml torch==2.14.0 \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

Modules whose only failure is a missing third-party package are reported separately,
so a partial environment still gives a usable answer rather than a wall of noise.
"""

import io
import os
from pathlib import Path
import re
import subprocess
import sys

WORKFLOW = Path(__file__).resolve().parents[2] / '.github/workflows/qwen-integration-cpu.yml'
SCRIPTS = Path(__file__).resolve().parent
MISSING = re.compile(r"No module named '([A-Za-z0-9_.]+)'")


DISCOVER = re.compile(r"\s*python -B -m unittest discover -s scripts/ci -p '([A-Za-z0-9_*.?\[\]-]+)'$")


def groups():
    """Every unittest group the workflow runs, in order. A `discover -s scripts/ci -p PATTERN`
    line becomes the modules its pattern matches here. Any other unittest line this cannot
    read is an error, never a silent skip: the discover lines were skipped for weeks, and a
    regression they cover stayed red in CI while this runner reported green."""
    text = io.open(WORKFLOW, encoding='utf-8').read().replace('\r\n', '\n')
    found, unread = [], []
    for line in text.split('\n'):
        match = re.match(r'\s*python -B -m unittest ([A-Za-z0-9_. -]+)$', line)
        if match and not match.group(1).startswith('discover'):
            found.append(match.group(1).split())
            continue
        discover = DISCOVER.match(line)
        if discover:
            modules = sorted(path.stem for path in SCRIPTS.glob(discover.group(1)))
            if not modules:
                unread.append(line.strip() + '  (pattern matches no module)')
            else:
                found.append(modules)
            continue
        if re.match(r'\s*python -B -m unittest', line):
            unread.append(line.strip())
    if unread:
        raise SystemExit('Unittest lines this runner cannot reproduce:\n  ' + '\n  '.join(unread))
    if not found:
        raise SystemExit('No unittest invocations found in %s' % WORKFLOW)
    return found


def main():
    only = set(sys.argv[1:])
    real, absent, modules = [], {}, 0
    for group in groups():
        if only and not only.intersection(group):
            continue
        modules += len(group)
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest'] + group,
            cwd=str(SCRIPTS), capture_output=True, text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode == 0:
            print('ok    ' + ' '.join(group[:4]) + (' ...' if len(group) > 4 else ''))
            continue
        lines = [line.strip() for line in result.stderr.split('\n')
                 if line.startswith(('FAIL:', 'ERROR:')) or MISSING.search(line)]
        names = {match.group(1) for line in lines for match in [MISSING.search(line)] if match}
        # A group that fails only because a package is not installed says nothing about
        # the code; the workflow installs those, this machine may not.
        if names and not any(line.startswith('FAIL:') for line in lines):
            absent.setdefault(tuple(sorted(names)), []).append(group[0])
            print('skip  %-44s (no %s)' % (group[0], ', '.join(sorted(names))))
            continue
        real.append((group, lines[:10]))
        print('FAIL  ' + ' '.join(group))
        for line in lines[:10]:
            print('        ' + line[:160])

    print()
    for names, owners in sorted(absent.items()):
        print('%d group(s) unrun for want of %s: %s' % (
            len(owners), ', '.join(names), ' '.join(owners[:6]) + (' ...' if len(owners) > 6 else '')))
    print('%d modules across %d groups; %d genuinely failing' % (modules, len(groups()), len(real)))
    return 1 if real else 0


if __name__ == '__main__':
    sys.exit(main())
