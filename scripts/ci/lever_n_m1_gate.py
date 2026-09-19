"""Lever N M1 gate: is resumable prefill byte-identical to the one-shot path?

Design section 5 asks for equality at three prompt lengths. The patch leaves
prefill_paged_slots untouched and adds prefill_paged_slots_range alongside it, so a
single grafted image can serve the same prompts both ways and the outputs compared
directly - no cross-image or cross-day comparison.

Baseline arm: chunked prefill off, so _prefill_forward_tp_batched takes the
start_pos-is-None branch and calls prefill_paged_slots exactly as today.
Resumable arm: chunked prefill on with max_num_batched_tokens equal to the model
chunk size, so the runner supplies a window per step and the new range path runs.

Sampling is greedy with n=1 and no penalties - the determinism constraints the fast
policy still pins - so identical prompts must give identical completions. Anything
else means the resumable path diverges, and the failure mode this guards against is
silent GDN state corruption at long context rather than a crash.

Prompt lengths cover the three shapes: no full chunk, one full chunk plus a tail, and
several full chunks plus a tail. The mid-prefill short-prompt case from section 5 is
NOT covered here - v1 allows one in-flight prefill per lane, so it needs the M2
scheduler and is gated there.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BEGIN = '<<<M1_GATE_JSON_BEGIN>>>'
END = '<<<M1_GATE_JSON_END>>>'
LOG_BEGIN = '<<<M1_GATE_LOG_BEGIN>>>'
LOG_END = '<<<M1_GATE_LOG_END>>>'
CHUNK_SIZE = 2048
BLOCK_SIZE = 64
MODEL = ('/models/hub/models--Qwen--Qwen3.8-27B/snapshots/'
         '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')


def build_prompt(target_tokens):
    """A deterministic prompt of roughly target_tokens tokens.

    Exactness does not matter: both arms receive the identical string, and the gate is
    equality between arms, not a particular length.
    """
    unit = 'def step_%d(value):\n    return value + %d\n\n'
    text, index = [], 0
    approx = 0
    while approx < target_tokens:
        chunk = unit % (index, index)
        text.append(chunk)
        approx += len(chunk) // 3          # ~3 chars per token, deliberately rough
        index += 1
    return ''.join(text)


def start_server(port, context, chunked, results, log_name):
    recipe = dict(qwen_fast_t16=True,
                  tt=dict(trace_mode='decode_only', trace_region_size=1073741824,
                          l1_small_size=24576),
                  qwen_fast_runtime=dict(directory='/experiment-scripts/ci',
                                         runtime_root='/opt/tt-metal',
                                         fixtures='/experiment-dflash-fixture',
                                         target_snapshot=MODEL))
    speculative = dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                       draft_sample_method='greedy', rejection_sample_method='standard')
    blocks = -(-context // BLOCK_SIZE)
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', 'qwen-m1',
               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
               '--max-model-len', str(context), '--max-num-seqs', '1',
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--shutdown-timeout', '30',
               '--speculative-config', json.dumps(speculative),
               '--additional-config', json.dumps(recipe)]
    if chunked:
        command += ['--enable-chunked-prefill',
                    '--max-num-batched-tokens', str(CHUNK_SIZE)]
    else:
        command += ['--no-enable-chunked-prefill',
                    '--max-num-batched-tokens', str(context)]
    log_path = results / log_name
    handle = log_path.open('w')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + 900
    while True:
        if process.poll() is not None:
            raise RuntimeError('server exited before readiness (chunked=%s): %s'
                               % (chunked, process.returncode))
        try:
            with urlopen('http://127.0.0.1:%d/health' % port, timeout=2) as response:
                if response.status == 200:
                    return process, handle, log_path, command
        except (URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('readiness exceeded 900s (chunked=%s)' % chunked)
        time.sleep(2)


def stop_server(process, handle):
    if process is not None and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=120)
        except BaseException:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except BaseException:
                pass
    if handle is not None:
        try:
            handle.close()
        except BaseException:
            pass


def complete(port, prompt, max_tokens):
    payload = json.dumps(dict(model='qwen-m1', prompt=prompt, max_tokens=max_tokens,
                              temperature=0.0, n=1)).encode()
    request = Request('http://127.0.0.1:%d/v1/completions' % port, data=payload,
                      headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urlopen(request, timeout=1800) as response:
        body = json.loads(response.read())
    choice = body['choices'][0]
    return dict(text=choice['text'], finish_reason=choice.get('finish_reason'),
                prompt_tokens=body.get('usage', {}).get('prompt_tokens'),
                completion_tokens=body.get('usage', {}).get('completion_tokens'),
                seconds=round(time.perf_counter() - started, 3))


def run_arm(port, context, chunked, prompts, max_tokens, results, log_name):
    process = handle = None
    arm = dict(chunked=chunked, completions=[])
    try:
        process, handle, log_path, command = start_server(port, context, chunked,
                                                          results, log_name)
        arm['command'] = command
        arm['ready'] = True
        for name, prompt in prompts:
            arm['completions'].append(dict(name=name, **complete(port, prompt, max_tokens)))
    except BaseException as error:
        arm['error'] = '%s: %s' % (type(error).__name__, str(error)[:500])
    finally:
        stop_server(process, handle)
    return arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--results', type=Path, default=Path('/tmp/m1-gate'))
    parser.add_argument('--lengths', default='400,3000,5000')
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    targets = [int(v) for v in options.lengths.split(',')]
    prompts = [('approx_%d' % n, build_prompt(n)) for n in targets]
    report = dict(scope=__doc__, context=options.context, chunk_size=CHUNK_SIZE,
                  targets=targets, max_tokens=options.max_tokens)
    try:
        report['baseline'] = run_arm(8000, options.context, False, prompts,
                                     options.max_tokens, options.results, 'baseline.log')
        report['resumable'] = run_arm(8001, options.context, True, prompts,
                                      options.max_tokens, options.results, 'resumable.log')
        comparisons = []
        base = {c['name']: c for c in report['baseline'].get('completions', [])}
        test = {c['name']: c for c in report['resumable'].get('completions', [])}
        for name, _ in prompts:
            b, t = base.get(name), test.get(name)
            entry = dict(name=name, both_present=bool(b and t))
            if b and t:
                entry.update(prompt_tokens=b.get('prompt_tokens'),
                             identical=b['text'] == t['text'],
                             baseline_len=len(b['text']), resumable_len=len(t['text']),
                             baseline_seconds=b['seconds'], resumable_seconds=t['seconds'])
                if b['text'] != t['text']:
                    for index, (x, y) in enumerate(zip(b['text'], t['text'])):
                        if x != y:
                            entry['first_divergence'] = index
                            entry['baseline_at'] = b['text'][index:index + 40]
                            entry['resumable_at'] = t['text'][index:index + 40]
                            break
                    else:
                        entry['first_divergence'] = min(len(b['text']), len(t['text']))
            comparisons.append(entry)
        report['comparisons'] = comparisons
        checked = [c for c in comparisons if c.get('both_present')]
        report['gate_passed'] = bool(checked) and all(c['identical'] for c in checked)
        report['lengths_checked'] = len(checked)
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    print(LOG_BEGIN)
    for name in ('baseline.log', 'resumable.log'):
        path = options.results / name
        if path.is_file():
            print('--- %s ---' % name)
            for line in path.read_text(errors='replace').splitlines()[-120:]:
                print(line[:260])
    print(LOG_END)
    sys.stdout.flush()
    return 0 if report.get('gate_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
