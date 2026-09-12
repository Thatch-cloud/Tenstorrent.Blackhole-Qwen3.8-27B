"""Run generated functions only inside the dedicated local systemd isolation root."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

from coding_holdout_tasks import EXPECTED, TASKS, messages


ROOT = Path('/opt/ttsim/coding-eval-root')
WORKER = '''import contextlib, copy, json, os, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
payload = json.load(sys.stdin)
results = []
with open(os.devnull, 'w') as discard, contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
    namespace = {}
    try:
        exec(compile(payload['source'], '<generated>', 'exec'), namespace)
        function = namespace[payload['function']]
        for arguments, expected in payload['cases']:
            supplied = copy.deepcopy(arguments)
            try:
                actual = function(*supplied)
                exact = json.loads(json.dumps(actual)) == expected
                results.append(dict(passed=exact and supplied == arguments,
                    value_exact=exact, input_unchanged=supplied == arguments))
            except BaseException as error:
                results.append(dict(passed=False, error=type(error).__name__))
    except BaseException as error:
        results = [dict(passed=False, error=type(error).__name__)]
print(json.dumps(dict(checks=results, passed=len(results) == len(payload['cases'])
    and all(value['passed'] for value in results))))
'''


def extract_source(text, function):
    if not isinstance(text, str) or len(text) > 32768:
        raise ValueError('Bounded generated text required')
    match = re.fullmatch(r'\s*```python\s*\n(.*?)\n```\s*', text, re.DOTALL)
    if match is None:
        raise ValueError('One complete Python code block required')
    source = match.group(1)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != function:
        raise ValueError('Exactly the requested function required')
    return source


def command(unit):
    return ['systemd-run', '--quiet', '--wait', '--pipe', '--collect', '--unit=' + unit,
        '-p', 'RootDirectory=' + str(ROOT), '-p', 'BindReadOnlyPaths=/usr /lib /lib64',
        '-p', 'DynamicUser=yes', '-p', 'PrivateNetwork=yes', '-p', 'PrivateDevices=yes',
        '-p', 'ProtectSystem=strict', '-p', 'ProtectHome=yes', '-p', 'NoNewPrivileges=yes',
        '-p', 'MemoryMax=134217728', '-p', 'TasksMax=16', '-p', 'RuntimeMaxSec=10',
        '-p', 'KillMode=control-group', '-p', 'RestrictSUIDSGID=yes',
        '-p', 'ProtectProc=invisible', '-p', 'ProcSubset=pid',
        '-p', 'RestrictNamespaces=yes', '-p', 'CapabilityBoundingSet=',
        '/usr/bin/python3', '-I', '-c', WORKER]


def evaluate(name, text):
    messages(name)
    task = next(task for task in TASKS if task['name'] == name)
    source = extract_source(text, task['function'])
    if os.name != 'posix' or os.geteuid() != 0 or not ROOT.is_dir() or ROOT.is_symlink():
        raise ValueError('Dedicated local Linux isolation root and system manager required')
    unit = 'qwen-coding-eval-' + uuid.uuid4().hex
    payload = json.dumps(dict(source=source, function=task['function'], cases=task['cases'])).encode()
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            result = subprocess.run(command(unit), input=payload, stdout=output, stderr=errors, timeout=15)
            output.seek(0)
            data = output.read(65537)
            if result.returncode != 0 or len(data) > 65536:
                return dict(passed=False, task=name, task_sha256=EXPECTED[name], failure='sandbox execution failed')
            value = json.loads(data)
            if type(value.get('passed')) is not bool or not isinstance(value.get('checks'), list):
                raise ValueError('Structured functional result required')
            value.update(task=name, task_sha256=EXPECTED[name], source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                scope='Local functional cases only; not comprehensive coding-quality certification')
            return value
        except subprocess.TimeoutExpired:
            return dict(passed=False, task=name, task_sha256=EXPECTED[name], failure='sandbox timeout')
        finally:
            subprocess.run(['systemctl', 'stop', unit], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=15)
