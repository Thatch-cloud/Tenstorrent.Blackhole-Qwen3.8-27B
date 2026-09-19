"""Checkpoint B: inter-token latency at 4 concurrent streams on the fast path.

The target is 200 tok/s per user, which is 5 ms of inter-token latency per stream.
Speculative decoding is what makes that arithmetically possible: the dense projection
payload is 19.92 GB per weight pass against 512 GB/s per card, a 19.5 ms floor, so
unbatched decode at 200 steps/s is 3.9x infeasible. Committing ~12 tokens per pass
turns that into a 60 ms cycle, and 60/12 is the 5 ms figure.

ITL is measured per stream with the first token dropped, matching the rig's existing
bench_itl.py convention - whole-request time was rejected there because controls came
out 41% apart.

Starts the server with the canary's fast-path recipe and speculation, then drives
concurrent streams. Reports measured latency only; it makes no claim about whether
the target is reachable.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BEGIN = '<<<CYCLE_BENCH_JSON_BEGIN>>>'
END = '<<<CYCLE_BENCH_JSON_END>>>'
LOG_BEGIN = '<<<CYCLE_BENCH_LOG_BEGIN>>>'
LOG_END = '<<<CYCLE_BENCH_LOG_END>>>'
BLOCK_SIZE = 64
TARGET_TOKS_PER_USER = 200.0


def stream_once(port, prompt, max_tokens, results, index):
    """One streaming completion; record the gap between successive tokens."""
    payload = json.dumps(dict(model='qwen-longctx', prompt=prompt,
                              max_tokens=max_tokens, temperature=0.0,
                              stream=True)).encode()
    request = Request('http://127.0.0.1:%d/v1/completions' % port, data=payload,
                      headers={'Content-Type': 'application/json'})
    gaps, tokens, started = [], 0, time.perf_counter()
    first_token_at = None
    try:
        with urlopen(request, timeout=900) as response:
            previous = None
            for raw in response:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                body = line[5:].strip()
                if body == '[DONE]':
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                text = (chunk.get('choices') or [{}])[0].get('text', '')
                if not text:
                    continue
                now = time.perf_counter()
                tokens += 1
                if first_token_at is None:
                    first_token_at = now
                elif previous is not None:
                    gaps.append(now - previous)   # first token deliberately dropped
                previous = now
        results[index] = dict(tokens=tokens, gaps_ms=[1000.0 * g for g in gaps],
                              ttft_s=(first_token_at - started) if first_token_at else None,
                              wall_s=time.perf_counter() - started)
    except BaseException as error:
        results[index] = dict(error='%s: %s' % (type(error).__name__, str(error)[:300]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--context', type=int, default=163840)
    parser.add_argument('--prompt-tokens', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--l1-small-size', type=int, default=24576)
    parser.add_argument('--trace-region-size', type=int, default=1073741824)
    parser.add_argument('--readiness-seconds', type=int, default=900)
    parser.add_argument('--results', type=Path, default=Path('/tmp/cycle-results'))
    parser.add_argument('--model', default='/models/hub/models--Qwen--Qwen3.8-27B/'
                                          'snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    blocks = -(-(options.users * options.context) // BLOCK_SIZE)
    report = dict(scope=__doc__, users=options.users, context=options.context,
                  blocks=blocks, target_tokens_per_user=TARGET_TOKS_PER_USER,
                  target_itl_ms=1000.0 / TARGET_TOKS_PER_USER, ready=False)

    speculative = dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                       draft_sample_method='greedy', rejection_sample_method='standard')
    recipe = dict(qwen_fast_t16=True,
                  tt=dict(trace_mode='decode_only',
                          trace_region_size=options.trace_region_size,
                          l1_small_size=options.l1_small_size),
                  qwen_fast_runtime=dict(directory='/experiment-scripts/ci',
                                         runtime_root='/opt/tt-metal',
                                         fixtures='/experiment-dflash-fixture',
                                         target_snapshot=options.model))
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', options.model, '--served-model-name', 'qwen-longctx',
               '--host', '127.0.0.1', '--port', str(options.port), '--dtype', 'bfloat16',
               '--max-model-len', str(options.context),
               '--max-num-seqs', str(options.users),
               '--max-num-batched-tokens', str(options.context),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--no-enable-chunked-prefill', '--shutdown-timeout', '30',
               '--speculative-config', json.dumps(speculative),
               '--additional-config', json.dumps(recipe)]
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
                    raise RuntimeError('server exited before readiness: %s'
                                       % process.returncode)
                try:
                    with urlopen('http://127.0.0.1:%d/health' % options.port,
                                 timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('readiness exceeded %ds' % options.readiness_seconds)
                time.sleep(2)
        report.update(ready=True, startup_seconds=round(time.perf_counter() - started, 1))

        prompt = 'def solve(n):\n    # ' + ('compute the answer carefully. ' *
                                            max(1, options.prompt_tokens // 6))
        results = [None] * options.users
        threads = [threading.Thread(target=stream_once,
                                    args=(options.port, prompt, options.max_tokens,
                                          results, index))
                   for index in range(options.users)]
        wall = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        report['wall_seconds'] = round(time.perf_counter() - wall, 3)
        report['streams'] = results

        gaps = [gap for entry in results if entry and entry.get('gaps_ms')
                for gap in entry['gaps_ms']]
        total_tokens = sum(entry.get('tokens', 0) for entry in results if entry)
        if gaps:
            report['itl_ms_median'] = round(statistics.median(gaps), 3)
            report['itl_ms_mean'] = round(statistics.fmean(gaps), 3)
            report['itl_ms_p90'] = round(sorted(gaps)[int(0.9 * (len(gaps) - 1))], 3)
            report['tokens_per_user_per_s'] = round(1000.0 / statistics.median(gaps), 1)
            report['aggregate_tokens_per_s'] = round(
                options.users * 1000.0 / statistics.median(gaps), 1)
            report['fraction_of_target'] = round(
                (1000.0 / statistics.median(gaps)) / TARGET_TOKS_PER_USER, 3)
        report['total_tokens'] = total_tokens
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=90)
            except BaseException:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except BaseException:
                    pass
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
