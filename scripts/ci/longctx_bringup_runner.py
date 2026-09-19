"""Bring-up only: can the stack allocate KV for 4 users at 161k and reach readiness?

This does not measure throughput. It is the cheapest thing that can falsify the
4-user plan: the canary runs 128 KV blocks (8192 tokens) and the target needs about
10240 (655360 tokens, ~21 GB), alongside 19.92 GB of weights in 64 GB. If the
allocator will not take that, nothing downstream matters.

Reports whether the server reached /health, how long it took, and whatever the log
says about the KV allocation. A single short completion afterwards proves it serves;
long-context behaviour and rate are explicitly out of scope here.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BEGIN = '<<<LONGCTX_JSON_BEGIN>>>'
END = '<<<LONGCTX_JSON_END>>>'
LOG_BEGIN = '<<<LONGCTX_SERVERLOG_BEGIN>>>'
LOG_END = '<<<LONGCTX_SERVERLOG_END>>>'

BLOCK_SIZE = 64
TOKENS_PER_BLOCK_KB = 32  # 16 attention layers x 2 x 8 kv heads x 128 dim at bf8


def kv_plan(users, context):
    tokens = users * context
    blocks = -(-tokens // BLOCK_SIZE)
    return dict(users=users, context=context, tokens=tokens, block_size=BLOCK_SIZE,
                blocks=blocks,
                kv_bytes=blocks * BLOCK_SIZE * TOKENS_PER_BLOCK_KB * 1024,
                kv_gib=round(blocks * BLOCK_SIZE * TOKENS_PER_BLOCK_KB * 1024
                             / (1024 ** 3), 2))


def scrape(log_path):
    """Pull whatever the server said about memory and KV, without assuming a format."""
    if not log_path.is_file():
        return {}
    text = log_path.read_text(errors='replace')
    patterns = {
        'kv_cache_lines': r'(?i)^.*kv[ _-]?cache.*$',
        'gpu_blocks_lines': r'(?i)^.*(?:gpu|num).{0,12}blocks.*$',
        'oom_lines': r'(?i)^.*(?:out of memory|oom|allocation failed|not enough|kv cache memory).*$',
        'firmware_line': r'^.*firmware bundle version.*$',
    }
    found = {}
    for name, pattern in patterns.items():
        hits = re.findall(pattern, text, re.M)
        if hits:
            found[name] = [h.strip()[:240] for h in hits[-6:]]
    found['log_bytes'] = len(text)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--context', type=int, default=163840)
    parser.add_argument('--results', type=Path, default=Path('/tmp/longctx-results'))
    parser.add_argument('--model', default=os.environ.get(
        'QWEN_TARGET_MODEL',
        '/models/hub/models--Qwen--Qwen3.8-27B/snapshots/'
        '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'))
    parser.add_argument('--draft-config', default=os.environ.get('QWEN_DRAFT_CONFIG'),
                        help='enable speculation against this draft config path')
    parser.add_argument('--chunked-prefill', action='store_true')
    parser.add_argument('--batched-tokens', type=int, default=None)
    parser.add_argument('--readiness-seconds', type=int, default=900)
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    plan = kv_plan(options.users, options.context)
    report = dict(scope=__doc__, plan=plan, throughput_measured=False,
                  stage='starting', ready=False)
    # Best-effort only: a host-owned bind mount is not writable from inside the
    # container, and the delimited stdout below is the real channel.
    try:
        (options.results / 'bringup.json').write_text(json.dumps(report, indent=2) + '\n')
    except OSError:
        pass

    # Speculation is off by default here: this branch has no draft-config, and KV
    # allocation - the thing being falsified - does not depend on it. The draft model
    # does consume some memory, so a passing result is mildly optimistic; the headroom
    # measured below is what tells us by how much.
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', options.model, '--served-model-name', 'qwen-longctx',
               '--host', '127.0.0.1', '--port', '8000', '--dtype', 'bfloat16',
               '--max-model-len', str(options.context),
               '--max-num-seqs', str(options.users),
               # Without chunked prefill the scheduler needs to admit a whole prefill.
               '--max-num-batched-tokens', str(options.batched_tokens or options.context),
               '--block-size', str(BLOCK_SIZE),
               '--num-gpu-blocks-override', str(plan['blocks']),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--shutdown-timeout', '30']
    if options.chunked_prefill:
        command += ['--enable-chunked-prefill']
    else:
        command += ['--no-enable-chunked-prefill']
    if options.draft_config:
        command += ['--speculative-config', json.dumps(dict(
            model=options.draft_config, method='dflash', num_speculative_tokens=15,
            draft_sample_method='greedy', rejection_sample_method='standard'))]
    report['speculative'] = bool(options.draft_config)
    report['command'] = command
    log_path = options.results / 'server.log'
    process = None
    started = time.perf_counter()
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + options.readiness_seconds
            while True:
                if process.poll() is not None:
                    report.update(stage='exited_before_ready',
                                  returncode=process.returncode)
                    raise RuntimeError('server exited before readiness: %s'
                                       % process.returncode)
                try:
                    with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    report['stage'] = 'readiness_timeout'
                    raise TimeoutError('readiness exceeded %ds' % options.readiness_seconds)
                time.sleep(2)
        report.update(ready=True, stage='ready',
                      startup_seconds=round(time.perf_counter() - started, 1))

        # One short completion: proves it serves, says nothing about rate.
        try:
            import urllib.request
            payload = json.dumps(dict(model='qwen-longctx', prompt='def add(a, b):',
                                      max_tokens=16, temperature=0.0)).encode()
            request = urllib.request.Request(
                'http://127.0.0.1:8000/v1/completions', data=payload,
                headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
            report['smoke_completion'] = body['choices'][0]['text'][:200]
            report['stage'] = 'served_short_request'
        except BaseException as error:
            report['smoke_error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:500])
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=60)
            except BaseException:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except BaseException:
                    pass
        report['log'] = scrape(log_path)
        try:
            (options.results / 'bringup.json').write_text(
                json.dumps(report, indent=2) + '\n')
        except OSError as error:
            report['results_write_error'] = str(error)[:200]
        # Delimited stdout is the real channel: the container cannot write to a
        # host-owned bind mount, which is how the first two attempts died.
        print(BEGIN)
        print(json.dumps(report, indent=2))
        print(END)
        print(LOG_BEGIN)
        if log_path.is_file():
            for line in log_path.read_text(errors='replace').splitlines()[-400:]:
                print(line[:300])
        print(LOG_END)
        sys.stdout.flush()
    return 0 if report.get('ready') else 1


if __name__ == '__main__':
    sys.exit(main())
