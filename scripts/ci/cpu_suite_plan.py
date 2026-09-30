"""Plan and run the CPU regression suite of any ref, in parallel shards, reporting every failure.

The suite is whatever the ref's own qwen-integration-cpu.yml runs: its dependency step and its
regression step. `plan` splits the regression step into commands (one per unittest module, so a
multi-module line becomes several independent commands; discover and shell lines stay whole) and
deals them round-robin into N shards. `run` executes one shard, never stopping at a failure, and
exits non-zero if any command failed, listing them all.

Used by .github/workflows/qwen-cpu-suite.yml (workflow_dispatch on the default branch), which checks
out the requested ref first, so this file only needs to exist on the default branch.
"""
import json
import os
import shlex
import subprocess
import sys
import time

WORKFLOW = '.github/workflows/qwen-integration-cpu.yml'
INSTALL_STEP = 'Install CPU test dependencies'
SUITE_STEP = 'T32 and unchanged T16 regression gates'


def steps(path=WORKFLOW):
    import yaml
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    found = {}
    for job in data.get('jobs', {}).values():
        for step in job.get('steps', []):
            if step.get('name') in (INSTALL_STEP, SUITE_STEP):
                found[step['name']] = [line.strip() for line in step.get('run', '').splitlines() if line.strip()]
    if SUITE_STEP not in found:
        raise SystemExit('%s has no step named %r' % (path, SUITE_STEP))
    return found.get(INSTALL_STEP, []), found[SUITE_STEP]


def split(lines):
    """One command per unittest module; everything else (discover, bash -n, git diff) as written."""
    commands = []
    for line in lines:
        words = shlex.split(line)
        if words[:4] == ['python', '-B', '-m', 'unittest'] and len(words) > 4 and not any(
                word.startswith('-') or word == 'discover' for word in words[4:]):
            commands.extend('python -B -m unittest %s' % module for module in words[4:])
        else:
            commands.append(line)
    return commands


def plan(shards, only=''):
    install, suite = steps()
    commands = ['python -B -m unittest %s' % module for module in only.split()] if only.strip() else split(suite)
    shards = max(1, min(int(shards), len(commands)))
    buckets = [[] for _ in range(shards)]
    for index, command in enumerate(commands):
        buckets[index % shards].append(command)
    return {'install': install, 'shards': buckets}


def run(commands):
    failed, lines = [], []
    for command in commands:
        began = time.time()
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        took = time.time() - began
        output = result.stdout.decode('utf-8', 'replace')
        tail = [line for line in output.splitlines() if line.startswith(('Ran ', 'OK', 'FAILED'))][-2:]
        status = 'ok' if result.returncode == 0 else 'FAIL rc=%d' % result.returncode
        print('::group::%s %s (%.1fs)' % (status, command, took))
        print(output)
        print('::endgroup::')
        lines.append('| %s | `%s` | %.1f s | %s |' % (status, command, took, ' '.join(tail)))
        if result.returncode != 0:
            failed.append(command)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as handle:
            handle.write('| result | command | time | tail |\n|---|---|---|---|\n' + '\n'.join(lines) + '\n')
            if failed:
                handle.write('\n**Failed (%d):**\n\n' % len(failed) + '\n'.join('- `%s`' % c for c in failed) + '\n')
    print('\n%d commands, %d failed' % (len(commands), len(failed)))
    for command in failed:
        print('FAILED: %s' % command)
    return 1 if failed else 0


def main(argv):
    if argv[:1] == ['plan']:
        result = plan(argv[1] if len(argv) > 1 else '6', argv[2] if len(argv) > 2 else '')
        print(json.dumps(result))
        return 0
    if argv[:1] == ['install']:
        install, _ = steps()
        for command in install:
            print('+ ' + command, flush=True)
            subprocess.run(command, shell=True, check=True)
        return 0
    if argv[:1] == ['run']:
        return run(json.loads(argv[1]))
    raise SystemExit('usage: cpu_suite_plan.py plan [shards] [modules] | install | run <json list>')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
