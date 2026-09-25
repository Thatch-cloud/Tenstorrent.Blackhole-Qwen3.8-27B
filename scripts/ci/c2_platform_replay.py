"""Replay the Thatch node agent's serving sequence on the rig, with full logs.

Smokes v2-v5 ran vLLM directly in the agent's container shape and passed; the agent's own
placement (2026-09-25, job 01M3BK9NQ1WQM3JTJSX2V03D1M) then wedged card A, because the Thatch
runtime restarts the engine inside one container and only the smokes' docker rm -f never did.
This replays what the agent does, step by step, in a copy of the agent's own container:

  1. start: the agent's container config (docker inspect of --source), the runtime entrypoint
     (serving.server), a new name and port, cards M then A re-resolved by board id;
  2. load:  `serving.manage --model <model> --release-first` (the placed job; release-first is
     forced on Tenstorrent), then the agent's warmup chat (max_tokens 1, no temperature);
  3. serve: a coding request, four concurrent requests, a tool call;
  4. reload: a second release-first load in the same container - the in-place restart;
  5. restart: docker stop (SIGTERM, graceful) and docker start, load again - a redeploy.
A chat after 4 and 5 must succeed. Everything the runtime and vLLM wrote under /tmp is copied
out before the container is removed. Exit 0 only if every step passed.

Usage: c2_platform_replay.py --source thatch-inference-Qwen-Qwen3.8-27B --image <ref> --results <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'


def run(command, timeout=None, check=True):
    print('$ ' + ' '.join(command[:12]) + (' ...' if len(command) > 12 else ''), flush=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('%s exited %d: %s' % (command[0], result.returncode, (result.stderr or result.stdout)[-800:]))
    return result


def run_arguments(source, image, name, port):
    info = json.loads(run(['docker', 'inspect', source]).stdout)[0]
    config, host = info['Config'], info['HostConfig']
    arguments = ['docker', 'run', '-d', '--name', name, '-p', '127.0.0.1:%d:8000' % port]
    if host.get('ReadonlyRootfs'):
        arguments.append('--read-only')
    for path, options in (host.get('Tmpfs') or {}).items():
        arguments += ['--tmpfs', path + (':' + options if options else '')]
    for bind in host.get('Binds') or ():
        arguments += ['-v', bind]
    for mount in info.get('Mounts') or ():
        if mount.get('Type') == 'bind' and not any(b.startswith(mount['Source'] + ':') for b in host.get('Binds') or ()):
            arguments += ['-v', '%s:%s%s' % (mount['Source'], mount['Destination'], '' if mount.get('RW', True) else ':ro')]
    if host.get('ShmSize'):
        arguments += ['--shm-size', str(host['ShmSize'])]
    if host.get('Memory'):
        arguments += ['--memory', str(host['Memory'])]
    if host.get('NanoCpus'):
        arguments += ['--cpus', '%g' % (host['NanoCpus'] / 1e9)]
    for capability in host.get('CapAdd') or ():
        arguments += ['--cap-add', capability]
    for device in host.get('Devices') or ():
        if 'tenstorrent' not in device['PathOnHost']:
            arguments += ['--device', device['PathOnHost']]
    for board in (CARD_M, CARD_A):
        arguments += ['--device', os.path.realpath('/dev/tenstorrent/by-id/' + board)]
    for variable in config.get('Env') or ():
        arguments += ['-e', variable]
    return arguments + [image]


def http(port, path, body=None, timeout=60):
    request = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path), method='POST' if body else 'GET',
                                     data=json.dumps(body).encode() if body else None,
                                     headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b'null')
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors='replace')[:600]
    except Exception as error:
        return None, repr(error)[:300]


def chat(port, model, content, max_tokens, timeout=900, **extra):
    started = time.time()
    status, body = http(port, '/v1/chat/completions', dict(model=model, max_tokens=max_tokens,
                        messages=[{'role': 'user', 'content': content}], **extra), timeout=timeout)
    wall = time.time() - started
    if status != 200:
        return dict(ok=False, status=status, body=body, wall_s=round(wall, 1))
    usage = body.get('usage') or {}
    message = body['choices'][0]['message']
    tokens = usage.get('completion_tokens') or 0
    return dict(ok=True, status=status, wall_s=round(wall, 1), completion_tokens=tokens,
                tok_s_e2e=round(tokens / wall, 2) if wall else None,
                finish=body['choices'][0].get('finish_reason'),
                tool_calls=message.get('tool_calls'), text=(message.get('content') or '')[:200])


def wait_http(port, container, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = run(['docker', 'inspect', '-f', '{{.State.Running}}', container], check=False).stdout.strip()
        if state != 'true':
            return 'container exited'
        status, _ = http(port, '/v1/models', timeout=5)
        if status is not None:
            return None
        time.sleep(5)
    return 'no HTTP response in %d s' % seconds


def load(container, model):
    result = run(['docker', 'exec', container, 'python3', '-m', 'serving.manage', '--url', '127.0.0.1:8000',
                  '--model', model, '--release-first'], timeout=1500, check=False)
    return dict(ok=result.returncode == 0, exit=result.returncode, out=(result.stdout + result.stderr)[-1500:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--name', default='qwen-c2-platform')
    parser.add_argument('--port', type=int, default=8011)
    options = parser.parse_args()
    name, port, model = options.name, options.port, options.model
    os.makedirs(options.results, exist_ok=True)
    steps = {}

    def record(step, value):
        steps[step] = value
        print('STEP %s %s' % (step, json.dumps(value)[:1500]), flush=True)
        return value.get('ok', True)

    def capture(tag):
        # /tmp is the agent's tmpfs, which docker cp cannot read, and a restart empties it: tar it
        # out while the container runs.
        with open(os.path.join(options.results, 'container-tmp-%s.tar' % tag), 'wb') as handle:
            subprocess.run(['docker', 'exec', name, 'tar', '-C', '/tmp', '-cf', '-', '.'], stdout=handle,
                           stderr=subprocess.DEVNULL, timeout=120)

    run(['docker', 'rm', '-f', name], check=False)
    arguments = run_arguments(options.source, options.image, name, port)
    with open(os.path.join(options.results, 'docker-run.json'), 'w') as handle:
        json.dump(arguments, handle, indent=1)
    passed = False
    try:
        run(arguments)
        problem = wait_http(port, name, 600)
        if record('start', dict(ok=problem is None, problem=problem)):
            if record('load', load(name, model)):
                record('warmup', chat(port, model, 'warmup', 1))
                ok = record('coding', chat(port, model, 'Write a Python function that merges two sorted lists, '
                                                        'with doctests. Code only.', 600))
                outs = [None] * 4

                def one(index):
                    outs[index] = chat(port, model, 'Write a unit test for a %s parser in Python.'
                                       % ('CSV', 'JSON', 'INI', 'TOML')[index], 300)

                threads = [threading.Thread(target=one, args=(index,)) for index in range(4)]
                [thread.start() for thread in threads]
                [thread.join() for thread in threads]
                ok = record('concurrent4', dict(ok=all(o and o['ok'] for o in outs), users=outs)) and ok
                tool = chat(port, model, 'Read src/main.rs using the tool.', 300, tool_choice='auto',
                            tools=[{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
                                    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                                                   'required': ['path']}}}])
                tool['ok'] = tool['ok'] and bool(tool.get('tool_calls'))
                ok = record('tool_call', tool) and ok
                ok = record('reload', load(name, model)) and ok
                ok = record('after_reload', chat(port, model, 'Say OK.', 8)) and ok
                capture('before-restart')
                run(['docker', 'stop', '-t', '60', name], timeout=120, check=False)
                run(['docker', 'start', name])
                problem = wait_http(port, name, 600)
                ok = record('restart', dict(ok=problem is None, problem=problem)) and ok
                if problem is None:
                    ok = record('load_after_restart', load(name, model)) and ok
                    ok = record('after_restart', chat(port, model, 'Say OK.', 8)) and ok
                passed = ok
    finally:
        capture('final')
        logs = run(['docker', 'logs', name], check=False)
        with open(os.path.join(options.results, 'platform-container.log'), 'w') as handle:
            handle.write(logs.stdout + '\n----- stderr -----\n' + logs.stderr)
        run(['docker', 'rm', '-f', name], check=False)
        with open(os.path.join(options.results, 'platform-replay.json'), 'w') as handle:
            json.dump(dict(passed=passed, steps=steps), handle, indent=1)
    print('PLATFORM_REPLAY passed=%s' % passed, flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
