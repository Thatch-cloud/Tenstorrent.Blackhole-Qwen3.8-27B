"""Replay the Thatch node agent's serving sequence on the rig, with full logs.

Smokes v2-v5 ran vLLM directly in the agent's container shape and passed; the agent's own
placement (2026-09-25, job 01M3BK9NQ1WQM3JTJSX2V03D1M) then wedged card A, because the Thatch
runtime restarts the engine inside one container and only the smokes' docker rm -f never did.
This replays what the agent does, step by step, in a copy of the agent's own container:

  1. start: the agent's container config (docker inspect of --source, or a saved inspect JSON),
     the runtime entrypoint (serving.server), a new name and port, cards M then A re-resolved by
     board id, and --profile (QWEN_C2_PROFILE) when given. The recorded Env merges the source
     image's ENV with the agent's; replaying it onto another --image needs the source image here to
     subtract its ENV, else the replay refuses (source_problem) rather than serve the old image's
     kernel cache key and defaults;
  2. load:  `serving.manage --model <model> --release-first` (the placed job; release-first is
     forced on Tenstorrent), then the agent's warmup chat (max_tokens 1, no temperature);
  3. serve: a coding request, four concurrent requests, a tool call;
  4. traffic (G7, c2-serve-for-real-plan 2.2 item 5): a long prompt (~60% of the served context),
     a streamed answer of at least 1000 tokens (not exercised, and said so, when the profile's
     ceiling is lower), four arrivals at seeded random times, n=2 refused AT THE EDGE under a
     contract profile (a 400 with the contract's message; it never reaches the engine) and the
     engine still answering after it, max_tokens=1 (exactly one token) and the engine still
     answering after it;
  5. reload: a second release-first load in the same container - the in-place restart;
  6. restart: docker stop (SIGTERM, graceful) and docker start, load again - a redeploy. The
     restart time (docker start to HTTP, and to the load's end) is recorded.
A chat after 5 and 6 must succeed. Everything the runtime and vLLM wrote under /tmp is copied
out before the container is removed. Exit 0 only if every step passed.

Usage: c2_platform_replay.py --source thatch-inference-Qwen-Qwen3.8-27B|<inspect.json> --image <ref> --results <dir>
       [--profile NAME] [--seed N]
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'
ARRIVAL_WINDOW_S = 30.0
LONG_ANSWER_TOKENS = 3000
LONG_PROMPT_SHARE = 0.6       # of the served max_model_len
CHARS_PER_TOKEN = 3.0         # conservative for Python source: the prompt stays under its share


def run(command, timeout=None, check=True):
    print('$ ' + ' '.join(command[:12]) + (' ...' if len(command) > 12 else ''), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                            timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('%s exited %d: %s' % (command[0], result.returncode, (result.stderr or result.stdout)[-800:]))
    return result


def argv_to_inspect(argv):
    """A recorded `docker run` argv (this script's own docker-run.json) as the inspect fields
    run_arguments reads: the flags run_arguments writes, read back."""
    info = dict(Config=dict(Env=[]), HostConfig=dict(Tmpfs={}, Binds=[], CapAdd=[], Devices=[]), Mounts=[])
    host, index = info['HostConfig'], 0
    while index < len(argv):
        token = argv[index]
        value = argv[index + 1] if index + 1 < len(argv) else None
        if token == '--read-only':
            host['ReadonlyRootfs'] = True
            index += 1
            continue
        if token == '--tmpfs':
            path, _, options = value.partition(':')
            host['Tmpfs'][path] = options
        elif token == '-v':
            host['Binds'].append(value)
        elif token == '--shm-size':
            host['ShmSize'] = int(value)
        elif token == '--memory':
            host['Memory'] = int(value)
        elif token == '--cpus':
            host['NanoCpus'] = int(round(float(value) * 1e9))
        elif token == '--cap-add':
            host['CapAdd'].append(value)
        elif token == '--device':
            host['Devices'].append(dict(PathOnHost=value))
        elif token == '-e':
            info['Config']['Env'].append(value)
        elif token not in ('--name', '-p'):
            index += 1
            continue
        index += 2
    return info


def inspect_source(source):
    """(the agent's container config, the image it ran): `docker inspect` of a live container, or a
    saved file - an inspect JSON, a recorded docker-run.json argv, or a tracked record
    {'source': {'image': ...}, 'argv': [...]} (references/c2-serving/agent-container-*.json): the
    placement's container is gone once the pair is taken for development."""
    if not os.path.isfile(source):
        info = json.loads(run(['docker', 'inspect', source]).stdout)
        info = info[0] if isinstance(info, list) else info
        return info, info.get('Image')
    with open(source, encoding='utf-8') as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'argv' in data:
        return argv_to_inspect(data['argv']), (data.get('source') or {}).get('image')
    if isinstance(data, list) and data and isinstance(data[0], str):
        return argv_to_inspect(data), data[-1]
    info = data[0] if isinstance(data, list) else data
    return info, info.get('Image')


def image_environment(reference):
    """The image's own ENV ('K=V' entries), or None when docker cannot inspect it."""
    result = run(['docker', 'image', 'inspect', reference], check=False)
    if result.returncode:
        return None
    try:
        return list(json.loads(result.stdout)[0]['Config'].get('Env') or ())
    except (ValueError, KeyError, IndexError):
        return None


def source_problem(source_image, image, inherited):
    """Why the recorded container cannot be replayed onto `image`, or None. Its Env merges the source
    image's own ENV with what the agent passed; without the source image's ENV to subtract, copying it
    whole onto ANOTHER image would serve the old image's kernel cache key (TT_METAL_CACHE), weight
    cache, PATH, venv and QWEN_* defaults over the new image's - a replay that passes on the wrong
    configuration (read-the-launched-argv). Onto the source image itself the copy is exact."""
    if inherited is not None or (source_image and image == source_image):
        return None
    if not source_image:
        return ('the source names no image, so its own ENV cannot be told apart from what the agent passed; '
                'replaying its whole Env onto %s could override that image\'s ENV' % image)
    return ('the source image %s is not on this host, so its own ENV cannot be told apart from what the agent '
            'passed; replaying its whole Env onto %s would override that image\'s ENV (kernel cache key, PATH, '
            'QWEN_* defaults). Pull %s first, or replay that image itself' % (source_image, image, source_image))


def serving_devices(root='/dev/tenstorrent/by-id'):
    return [os.path.realpath(os.path.join(root, board)) for board in (CARD_M, CARD_A)]


def run_arguments(info, image, name, port, profile=None, devices=None, image_env=()):
    """`docker run -d` of a copy of the agent's container (config `info`), with `image`, a new name
    and port, cards M then A, and QWEN_C2_PROFILE=`profile` in place of the agent's when given.
    `image_env` is the SOURCE image's own ENV: a container's Env merges it with what the agent passed,
    and copied whole it would override the replayed image's ENV with the old image's (its kernel
    cache key, a changed QWEN_* default) - so only what the agent added is copied."""
    config, host = info['Config'], info['HostConfig']
    inherited = set(image_env or ())
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
    for device in (devices if devices is not None else serving_devices()):
        arguments += ['--device', device]
    for variable in config.get('Env') or ():
        if (profile and variable.startswith('QWEN_C2_PROFILE=')) or variable in inherited:
            continue
        arguments += ['-e', variable]
    if profile:
        arguments += ['-e', 'QWEN_C2_PROFILE=%s' % profile]
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
                prompt_tokens=usage.get('prompt_tokens'), tok_s_e2e=round(tokens / wall, 2) if wall else None,
                finish=body['choices'][0].get('finish_reason'),
                tool_calls=message.get('tool_calls'), text=(message.get('content') or '')[:200])


def parse_sse(lines, on_first=None):
    """A streamed chat completion's lines -> (text pieces, usage, finish reason, error); `on_first` is
    called once, as the first non-empty piece arrives (a role-only first chunk carries no text)."""
    pieces, usage, finish, error = [], None, None, None
    for raw in lines:
        line = raw.decode(errors='replace').strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith('data:'):
            continue
        body = line[5:].strip()
        if body == '[DONE]':
            break
        try:
            chunk = json.loads(body)
        except ValueError:
            continue
        if chunk.get('error') is not None:
            error = str(chunk['error'])[:300]
            break
        if chunk.get('usage'):
            usage = chunk['usage']
        for choice in chunk.get('choices') or ():
            delta = choice.get('delta') or {}
            text = (delta.get('content') or '') + (delta.get('reasoning_content') or delta.get('reasoning') or '')
            if text:
                if not pieces and on_first is not None:
                    on_first()
                pieces.append(text)
            finish = choice.get('finish_reason') or finish
    return pieces, usage, finish, error


def stream_chat(port, model, content, max_tokens, timeout=1800, clock=time.time, opener=None, **extra):
    """A streamed chat: time to first token, tokens, decode rate after the first, finish reason."""
    body = dict(model=model, messages=[{'role': 'user', 'content': content}], max_tokens=max_tokens, stream=True,
                stream_options={'include_usage': True}, **extra)
    request = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % port, data=json.dumps(body).encode(),
                                     method='POST', headers={'content-type': 'application/json'})
    started, first = clock(), [None]

    def on_first():
        first[0] = clock()

    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            pieces, usage, finish, error = parse_sse(response, on_first)
    except urllib.error.HTTPError as http_error:
        return dict(ok=False, status=http_error.code, body=http_error.read().decode(errors='replace')[:600])
    except Exception as failure:
        return dict(ok=False, status=None, body=repr(failure)[:300])
    ended = clock()
    tokens = (usage or {}).get('completion_tokens') or 0
    decode = (ended - first[0]) if first[0] else None
    return dict(ok=error is None and finish in ('stop', 'length'), status=200, error=error, finish=finish,
                completion_tokens=tokens, prompt_tokens=(usage or {}).get('prompt_tokens'),
                ttft_s=round(first[0] - started, 2) if first[0] else None, wall_s=round(ended - started, 1),
                decode_tok_s=round((tokens - 1) / decode, 2) if decode and tokens > 1 else None,
                text=''.join(pieces)[:200])


def arrival_offsets(seed, count=4, window=ARRIVAL_WINDOW_S):
    """`count` sorted arrival times in [0, window) seconds, seeded so a failing run can be replayed."""
    generator = random.Random(seed)
    return sorted(round(generator.uniform(0.0, window), 2) for _ in range(count))


def source_corpus(limit, root=None):
    """At least `limit` characters of Python source (the host's standard library, sorted, each file
    headed by its path), or all of it."""
    root = root or sysconfig.get_paths()['stdlib']
    parts, total = [], 0
    for path in sorted(glob.glob(os.path.join(root, '*.py'))):
        try:
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        parts.append('# File: %s\n%s' % (os.path.basename(path), text))
        total += len(parts[-1])
        if total >= limit:
            break
    return '\n\n'.join(parts)[:limit]


def long_prompt(max_model_len, corpus=None, share=LONG_PROMPT_SHARE, chars_per_token=CHARS_PER_TOKEN):
    """A coding request whose context is ~`share` of the served context (in characters at a
    conservative characters-per-token), so it is long without meeting the prompt cap."""
    chars = int(max_model_len * share * chars_per_token)
    corpus = corpus if corpus is not None else source_corpus(chars)
    return ('Here is part of a Python codebase:\n\n' + corpus[:chars] +
            '\n\nName the last complete function above and explain what it does in two sentences.')


LONG_ANSWER_PROMPT = ('Write a complete, production-quality Python module implementing an in-memory key-value store '
                      'with TTL expiry, LRU eviction, snapshots to disk and a small command-line interface. Include '
                      'full docstrings, type hints and a thorough pytest suite, then explain every design decision '
                      'at length. Do not abbreviate anything.')
ARRIVAL_PROMPTS = ('Write a Python function that validates an IPv6 address, with tests.',
                   'Explain the difference between a process and a thread, with examples in Python.',
                   'Write a Rust function that merges overlapping intervals, with unit tests.',
                   'Write a SQL query that finds the second-highest salary per department, and explain it.')


def served_context(port):
    status, body = http(port, '/v1/models', timeout=30)
    if status == 200 and isinstance(body, dict):
        for model in body.get('data') or ():
            if model.get('max_model_len'):
                return int(model['max_model_len'])
    return None


def arrivals(port, model, offsets, max_tokens=400, sleep=time.sleep, stream=stream_chat):
    """Four requests arriving at `offsets` seconds from now, each streamed to its end."""
    outs = [None] * len(offsets)

    def one(index):
        sleep(offsets[index])
        outs[index] = dict(stream(port, model, ARRIVAL_PROMPTS[index % len(ARRIVAL_PROMPTS)], max_tokens),
                           arrived_s=offsets[index])

    threads = [threading.Thread(target=one, args=(index,)) for index in range(len(offsets))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    return dict(ok=all(o and o['ok'] for o in outs), offsets=list(offsets), users=outs)


LONG_ANSWER_MIN_TOKENS = 1000
CONTRACT_INSTALLED = '[QWEN-C2] request contract installed'
N_REFUSAL = 'n must be 1 on this model'   # serving_c2_contract.enforce_request's own words


def long_answer_verdict(result, minimum=LONG_ANSWER_MIN_TOKENS):
    """The streamed-long-answer step: at least `minimum` tokens streamed. An answer the profile's own
    ceiling cut first ('length' short of it) did not exercise the step and says so; one that stopped at
    EOS short of it fails - a 50-token answer is not thousands."""
    tokens = result.get('completion_tokens') or 0
    result['thousands'] = tokens >= minimum
    if result.get('ok') and not result['thousands']:
        if result.get('finish') == 'length':
            result['exercised'] = False
            result['note'] = 'the profile\'s output ceiling (%d tokens) is below %d: not exercised' % (tokens, minimum)
        else:
            result['ok'] = False
            result['note'] = 'ended at %r after %d tokens, short of %d' % (result.get('finish'), tokens, minimum)
    return result


def edge_refusal_verdict(result, contract):
    """n=2: under a contract profile the EDGE refuses it (serving_c2_contract.enforce_request raises
    before the engine sees it: a 400 carrying the contract's message); without a contract (general,
    the stock path) it is served, or refused by the stock stack's own 400 - never a server error.
    Nothing here reaches an in-engine refusal."""
    if contract:
        result['ok'] = result.get('status') == 400 and N_REFUSAL in str(result.get('body') or '')
        result['expected'] = '400 with %r (the contract refuses at the edge)' % N_REFUSAL
    else:
        result['ok'] = result.get('status') in (200, 400)
        result['expected'] = '200 or 400 (no request contract: the stock path answers for itself)'
    return result


def traffic_steps(port, model, record, seed, stream=stream_chat, ask=chat, sleep=time.sleep, contract=True):
    """Step 4, the G7 additions; returns whether every one passed. `record` logs a step and returns
    its ok. `contract`: whether the served profile installed the request contract (its log line). The
    survival checks are what matter: the engine must answer after an edge refusal and after a
    one-token request (serving_lifecycle builds a bridge even for max_tokens=1)."""
    ok = True
    context = served_context(port) or 65536
    prompt = long_prompt(context)
    result = stream(port, model, prompt, 256)
    result.update(prompt_characters=len(prompt), served_context=context)
    ok = record('long_prompt', result) and ok
    result = long_answer_verdict(stream(port, model, LONG_ANSWER_PROMPT, LONG_ANSWER_TOKENS))
    ok = record('streamed_long_answer', result) and ok
    ok = record('arrivals4', arrivals(port, model, arrival_offsets(seed), sleep=sleep, stream=stream)) and ok
    refused = edge_refusal_verdict(ask(port, model, 'hi', 8, n=2), contract)
    ok = record('edge_refusal_n2', refused) and ok
    ok = record('alive_after_refusal', ask(port, model, 'Say OK.', 8)) and ok
    one = ask(port, model, 'Write a haiku about compilers.', 1)
    one['ok'] = one['ok'] and one.get('completion_tokens') == 1
    ok = record('max_tokens_1', one) and ok
    ok = record('alive_after_max_tokens_1', ask(port, model, 'Say OK.', 8)) and ok
    return ok


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
    started = time.time()
    result = run(['docker', 'exec', container, 'python3', '-m', 'serving.manage', '--url', '127.0.0.1:8000',
                  '--model', model, '--release-first'], timeout=1500, check=False)
    return dict(ok=result.returncode == 0, exit=result.returncode, out=(result.stdout + result.stderr)[-1500:],
                seconds=round(time.time() - started, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='the agent\'s container, or a saved `docker inspect` JSON')
    parser.add_argument('--image', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--name', default='qwen-c2-platform')
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--profile', default=None, help='QWEN_C2_PROFILE for the copy (default: the source\'s)')
    parser.add_argument('--seed', type=int, default=None, help='the arrivals\' seed (default: the clock)')
    options = parser.parse_args()
    name, port, model = options.name, options.port, options.model
    seed = options.seed if options.seed is not None else int(time.time())
    os.makedirs(options.results, exist_ok=True)
    steps = dict(seed=dict(ok=True, seed=seed))

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
    info, source_image = inspect_source(options.source)
    with open(os.path.join(options.results, 'source-inspect.json'), 'w') as handle:
        json.dump(info, handle, indent=1)
    inherited = image_environment(source_image) if source_image else None
    problem = source_problem(source_image, options.image, inherited)
    record('source', dict(ok=problem is None, source=options.source, source_image=source_image,
                          image_env_subtracted=inherited is not None, problem=problem))
    if problem is not None:
        with open(os.path.join(options.results, 'platform-replay.json'), 'w') as handle:
            json.dump(dict(passed=False, steps=steps), handle, indent=1)
        print('PLATFORM_REPLAY passed=False refused: %s' % problem, flush=True)
        return 1
    arguments = run_arguments(info, options.image, name, port, options.profile, image_env=inherited or ())
    with open(os.path.join(options.results, 'docker-run.json'), 'w') as handle:
        json.dump(arguments, handle, indent=1)
    passed = False
    try:
        started = time.time()
        run(arguments)
        problem = wait_http(port, name, 600)
        if record('start', dict(ok=problem is None, problem=problem, http_s=round(time.time() - started, 1))):
            if record('load', load(name, model)):
                ok = record('warmup', chat(port, model, 'warmup', 1))
                ok = record('coding', chat(port, model, 'Write a Python function that merges two sorted lists, '
                                                        'with doctests. Code only.', 600)) and ok
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
                logs = run(['docker', 'logs', name], check=False)
                contract = CONTRACT_INSTALLED in (logs.stdout or '') + (logs.stderr or '')
                ok = traffic_steps(port, model, record, seed, contract=contract) and ok
                ok = record('reload', load(name, model)) and ok
                ok = record('after_reload', chat(port, model, 'Say OK.', 8)) and ok
                capture('before-restart')
                run(['docker', 'stop', '-t', '60', name], timeout=120, check=False)
                restarted = time.time()
                run(['docker', 'start', name])
                problem = wait_http(port, name, 600)
                ok = record('restart', dict(ok=problem is None, problem=problem,
                                            http_s=round(time.time() - restarted, 1))) and ok
                if problem is None:
                    loaded = load(name, model)
                    loaded['restart_to_loaded_s'] = round(time.time() - restarted, 1)
                    ok = record('load_after_restart', loaded) and ok
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
    restart = steps.get('load_after_restart') or {}
    print('PLATFORM_REPLAY passed=%s restart_http_s=%s restart_to_loaded_s=%s seed=%s' % (
        passed, (steps.get('restart') or {}).get('http_s'), restart.get('restart_to_loaded_s'), seed), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
