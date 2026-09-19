"""Lever N M1 gate: is resumable prefill byte-identical to the one-shot path?

Design section 5 asks for equality at three prompt lengths. The patch leaves
prefill_paged_slots untouched and adds prefill_paged_slots_range alongside it, so a
single grafted image can serve the same prompts both ways and the outputs compared
directly - no cross-image or cross-day comparison.

Baseline arm: chunked prefill off, so _prefill_forward_tp_batched takes the
start_pos-is-None branch and calls prefill_paged_slots exactly as today.
Resumable arm: chunked prefill on with max_num_batched_tokens equal to the model
chunk size, so the runner supplies a window per step and the new range path runs.

Runs on the PLAIN path by default. Run 35413668471 sent both arms through the T16 fast
runtime and every request died in dflash_device.__init__ with 'bounded prefill
required': that path pins a 4096-token prompt in two places, so it rejects every prompt
this gate needs. M1 is a prefill contract and owes nothing to speculation. Plain greedy
decode is also the cleaner probe, being a direct function of the state prefill left
behind, with no acceptance dynamics on top. --fast-path restores the old behaviour.

Sampling is greedy with n=1 and no penalties, so identical prompts must give identical
completions. Anything else means the resumable path diverges, and the failure mode this
guards against is silent GDN state corruption at long context rather than a crash.

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


def start_server(port, context, chunked, results, log_name, fast=False, seqs=4):
    # Without the tt block the plugin opens the mesh with l1_small_size=0 and the
    # first L1_SMALL allocation dies with 'bank size is 0 B'.
    recipe = dict(tt=dict(trace_mode='decode_only', trace_region_size=1073741824,
                          l1_small_size=24576))
    if fast:
        recipe['qwen_fast_t16'] = True
        recipe['qwen_fast_runtime'] = dict(directory='/experiment-scripts/ci',
                                           runtime_root='/opt/tt-metal',
                                           fixtures='/experiment-dflash-fixture',
                                           target_snapshot=MODEL)
    # prefill_paged_slots and prefill_paged_slots_range are the BATCHED path, and
    # prefill_forward only reaches it when model.args.max_batch_size > 1. At
    # max_num_seqs=1 run 35415167291 served both arms through _prefill_forward_tp
    # instead, so the methods under test never ran and the control caught it.
    blocks = seqs * (-(-context // BLOCK_SIZE))
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', 'qwen-m1',
               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
               '--max-model-len', str(context), '--max-num-seqs', str(seqs),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--shutdown-timeout', '30',
               '--additional-config', json.dumps(recipe)]
    if fast:
        command += ['--speculative-config', json.dumps(
            dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                 draft_sample_method='greedy', rejection_sample_method='standard'))]
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


def run_arm(port, context, chunked, prompts, max_tokens, results, log_name, fast=False,
            seqs=4):
    process = handle = None
    arm = dict(chunked=chunked, completions=[])
    try:
        process, handle, log_path, command = start_server(port, context, chunked,
                                                          results, log_name, fast, seqs)
        arm['command'] = command
        arm['ready'] = True
        for name, prompt in prompts:
            arm['completions'].append(dict(name=name, **complete(port, prompt, max_tokens)))
    except BaseException as error:
        arm['error'] = '%s: %s' % (type(error).__name__, str(error)[:500])
    finally:
        stop_server(process, handle)
    # Positive control. The patched entry logs which prefill path it took. If the
    # plugin never supplies start_pos the resumable arm quietly serves the one-shot
    # path and an equality comparison between the arms passes having tested nothing.
    path = results / log_name
    if path.is_file():
        text = path.read_text(errors='replace')
        arm['one_shot_path_seen'] = '[M1] prefill path: one-shot' in text
        arm['range_path_seen'] = '[M1] prefill path: resumable' in text
    return arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--seqs', type=int, default=4,
                        help='max_num_seqs; must exceed 1 or prefill_forward never '
                             'reaches the batched path these methods live on')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--results', type=Path, default=Path('/tmp/m1-gate'))
    parser.add_argument('--lengths', default='400,3000,5000')
    parser.add_argument('--arm', choices=('baseline', 'resumable', 'both'),
                        default='both',
                        help='which arm to serve; one per CI job, because each server '
                             'start opens the mesh and two opens in one job wedges a card')
    parser.add_argument('--peer', type=Path,
                        help='the other arm report to compare against')
    parser.add_argument('--fast-path', action='store_true',
                        help='route through the T16 fast runtime; its 4096-token prompt '
                             'pins apply and will reject shorter prompts')
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if options.seqs < 2:
        raise SystemExit('--seqs must be at least 2; at 1 the batched prefill path '
                         'this gate exists to test is never reached')
    targets = [int(v) for v in options.lengths.split(',')]
    prompts = [('approx_%d' % n, build_prompt(n)) for n in targets]
    report = dict(scope=__doc__, context=options.context, chunk_size=CHUNK_SIZE,
                  seqs=options.seqs,
                  targets=targets, max_tokens=options.max_tokens)
    try:
        report['fast_path'] = options.fast_path
        report['arm'] = options.arm
        if options.peer and options.peer.is_file():
            peer = json.loads(options.peer.read_text())
            for name in ('baseline', 'resumable'):
                if peer.get(name):
                    report[name] = peer[name]
        if options.arm in ('baseline', 'both'):
            report['baseline'] = run_arm(8000, options.context, False, prompts,
                                         options.max_tokens, options.results,
                                         'baseline.log', options.fast_path, options.seqs)
        if options.arm in ('resumable', 'both'):
            report['resumable'] = run_arm(8001, options.context, True, prompts,
                                          options.max_tokens, options.results,
                                          'resumable.log', options.fast_path, options.seqs)
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
        # Every length must actually run. Scoring only the lengths that happened to
        # succeed is how a gate passes while silently testing less than it claims.
        base_arm, test_arm = report.get('baseline') or {}, report.get('resumable') or {}
        # The resumable arm legitimately uses BOTH paths: a prompt's first chunk starts
        # at 0 and stays one-shot, continuations go to the range method. So the control
        # is that the baseline arm never reaches the range method, not that the
        # resumable arm never reaches the one-shot one.
        controls = dict(baseline_took_one_shot=bool(base_arm.get('one_shot_path_seen')),
                        baseline_avoided_range=not base_arm.get('range_path_seen'),
                        resumable_took_range=bool(test_arm.get('range_path_seen')))
        report['controls'] = controls
        report['gate_passed'] = (len(checked) == len(prompts)
                                 and all(c['identical'] for c in checked)
                                 and all(controls.values()))
        report['lengths_checked'] = len(checked)
        report['lengths_required'] = len(prompts)
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    print(LOG_BEGIN)
    for name in ('baseline.log', 'resumable.log'):
        path = options.results / name
        if not path.is_file():
            continue
        lines = path.read_text(errors='replace').splitlines()
        first = next((i for i, l in enumerate(lines)
                      if 'ERROR' in l or 'Traceback' in l), None)
        print('--- %s (%d lines) ---' % (name, len(lines)))
        if first is not None and first < len(lines) - 200:
            print('--- first error at line %d ---' % first)
            for line in lines[first:first + 80]:
                print(line[:260])
            print('--- tail ---')
        for line in lines[-200:]:
            print(line[:260])
    print(LOG_END)
    sys.stdout.flush()
    return 0 if report.get('gate_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
