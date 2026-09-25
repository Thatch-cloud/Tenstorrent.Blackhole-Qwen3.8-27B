"""Lever N M2 gate: does alternation keep a concurrent decode from stalling
across a chunked prefill?

M1 proved resumable prefill byte-identical to the one-shot path (run 35416319586)
but the design doc is explicit that chunked prefill ALONE does not fix decode
starvation: LaneScheduler._negotiate_forced_mode votes prefill-only for the whole
step whenever any lane has a partial prefill in flight, so a long prompt still
monopolises every step until it finishes, just in smaller increments. M2 is the
two scheduler edits (lever_n_scheduler_patch.py) that make the coordinator yield
decode steps between chunks instead. This gate is the first one that can actually
observe the failure mode M2 targets, because M1's gate never runs a decode and a
prefill at the same time - it compares two sequential single-request servers.

One server per arm, one request each, run concurrently against it:

  - D: a short prompt, streamed, sampled token by token so its inter-token
    arrival gaps are directly observable.
  - P: a long prompt (multiple 2048-token chunks), submitted once D has settled
    into a steady decode cadence, and awaited synchronously.

"m2" arm: the image grafted with M1's three files plus both M2 scheduler edits.
"baseline" arm: M1's three files only - stock scheduler.py/lane_scheduler.py, so
chunked prefill runs but nothing yields decode between chunks.

The concrete claim under test: with M2, D's worst inter-token gap while P is in
flight stays within one forced prefill-chunk step's duration of D's own steady
cadence (GAP_BUDGET_SECONDS, sized off the M1 gate's own timing); without it, the
same measurement blows well past that bound because D gets nothing until P's
prefill finishes entirely. Positive controls (mirroring M1's) rule out both
arms happening to produce a comparison that tests nothing: P must actually take
multiple chunk steps in both arms, the m2 arm's server log must show the
alternation patch actually firing, and the baseline's must show it did not
(confirming the baseline graft is genuinely unpatched, not just quiet by luck).

Runs on the plain path, like the M1 gate, for the same reason: M2 is a scheduler
contract and owes nothing to speculation.
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

BEGIN = '<<<M2_GATE_JSON_BEGIN>>>'
END = '<<<M2_GATE_JSON_END>>>'
LOG_BEGIN = '<<<M2_GATE_LOG_BEGIN>>>'
LOG_END = '<<<M2_GATE_LOG_END>>>'
CHUNK_SIZE = 2048
BLOCK_SIZE = 64
CONTEXT = 16384
MODEL = ('/models/hub/models--Qwen--Qwen3.8-27B/snapshots/'
         '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
# Generous allowance for one extra forced prefill-chunk step, sized off the M1
# gate's own measurement: chunking a 5,918-token prompt took 4.43 s across three
# 2048-token steps, ~1.5 s/step. 3.0 s covers one step plus scheduling and HTTP
# overhead with margin, while still being far short of a multi-step stall (P
# below needs at least 3 chunk steps, so an unbounded baseline should exceed
# this by a wide margin, not a narrow one).
GAP_BUDGET_SECONDS = 3.0


def build_prompt(target_tokens):
    """A deterministic prompt of roughly target_tokens tokens (see lever_n_m1_gate)."""
    unit = 'def step_%d(value):\n    return value + %d\n\n'
    text, index = [], 0
    approx = 0
    while approx < target_tokens:
        chunk = unit % (index, index)
        text.append(chunk)
        approx += len(chunk) // 3
        index += 1
    return ''.join(text)


def start_server(port, results, log_name, seqs):
    recipe = dict(tt=dict(trace_mode='decode_only', trace_region_size=1073741824,
                          l1_small_size=24576))
    blocks = seqs * (-(-CONTEXT // BLOCK_SIZE))
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', 'qwen-m2',
               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
               '--max-model-len', str(CONTEXT), '--max-num-seqs', str(seqs),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--shutdown-timeout', '30',
               '--limit-mm-per-prompt', json.dumps(dict(image=0, video=0)),
               '--additional-config', json.dumps(recipe),
               '--enable-chunked-prefill', '--max-num-batched-tokens', str(CHUNK_SIZE)]
    log_path = results / log_name
    handle = log_path.open('w')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + 900
    while True:
        if process.poll() is not None:
            raise RuntimeError('server exited before readiness: %s' % process.returncode)
        try:
            with urlopen('http://127.0.0.1:%d/health' % port, timeout=2) as response:
                if response.status == 200:
                    return process, handle, log_path, command
        except (URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('readiness exceeded 900s')
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


def stream_decode(port, prompt, max_tokens, arrivals, started_event, errors):
    """Stream D's completion, appending (perf_counter, text) per token to arrivals.

    Runs in its own thread; the caller only reads arrivals/errors after joining
    it, so no lock is needed. started_event lets the caller wait for the first
    token before submitting P, so the "steady" baseline gap is D's cadence, not
    its TTFT.
    """
    try:
        payload = json.dumps(dict(model='qwen-m2', prompt=prompt, max_tokens=max_tokens,
                                  temperature=0.0, n=1, stream=True)).encode()
        request = Request('http://127.0.0.1:%d/v1/completions' % port, data=payload,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=1800) as response:
            for raw in response:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                body = line[len('data:'):].strip()
                if body == '[DONE]':
                    break
                try:
                    event = json.loads(body)
                except ValueError:
                    continue
                choices = event.get('choices') or [{}]
                text = choices[0].get('text', '')
                if not text:
                    continue
                arrivals.append((time.perf_counter(), text))
                started_event.set()
    except BaseException as error:
        errors.append('%s: %s' % (type(error).__name__, str(error)[:300]))
    finally:
        started_event.set()  # never let the caller hang if D fails before any token


def complete(port, prompt, max_tokens):
    payload = json.dumps(dict(model='qwen-m2', prompt=prompt, max_tokens=max_tokens,
                              temperature=0.0, n=1)).encode()
    request = Request('http://127.0.0.1:%d/v1/completions' % port, data=payload,
                      headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urlopen(request, timeout=1800) as response:
        body = json.loads(response.read())
    choice = body['choices'][0]
    return dict(text=choice['text'], finish_reason=choice.get('finish_reason'),
                seconds=round(time.perf_counter() - started, 3))


def gaps(arrivals):
    return [b[0] - a[0] for a, b in zip(arrivals, arrivals[1:])]


def run_arm(arm, port, results, log_name, seqs, decode_tokens, decode_warmup_tokens,
            prefill_target_tokens, prefill_max_tokens):
    process = handle = None
    report = dict(arm=arm)
    try:
        process, handle, log_path, command = start_server(port, results, log_name, seqs)
        report['command'] = command
        report['ready'] = True

        arrivals = []
        decode_errors = []
        started = threading.Event()
        decode_thread = threading.Thread(
            target=stream_decode,
            args=(port, build_prompt(64), decode_tokens, arrivals, started, decode_errors),
            daemon=True)
        decode_thread.start()
        if not started.wait(timeout=120):
            raise TimeoutError('decode request never produced a first token or failed')
        deadline = time.monotonic() + 60
        while len(arrivals) < decode_warmup_tokens and decode_thread.is_alive():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        prefill_submitted_at = time.perf_counter()
        prefill_result = complete(port, build_prompt(prefill_target_tokens),
                                  prefill_max_tokens)
        report['prefill'] = dict(seconds=prefill_result['seconds'],
                                 finish_reason=prefill_result.get('finish_reason'))

        decode_thread.join(timeout=300)
        report['decode_thread_alive'] = decode_thread.is_alive()
        report['decode_tokens_received'] = len(arrivals)
        report['decode_errors'] = decode_errors

        steady = [pair for pair in arrivals if pair[0] < prefill_submitted_at]
        overlap = [pair for pair in arrivals if pair[0] >= prefill_submitted_at]
        steady_gaps = gaps(steady)
        # The gap that matters is the one SPANNING the overlap window (last
        # steady token -> each overlap token), not just gaps between overlap
        # tokens: a single huge stall right after P is submitted, followed by
        # fast catch-up tokens once it clears, must not be averaged away.
        overlap_gaps = gaps(steady[-1:] + overlap)
        report['steady_gap_median'] = (round(statistics.median(steady_gaps), 3)
                                       if steady_gaps else None)
        report['overlap_gap_max'] = round(max(overlap_gaps), 3) if overlap_gaps else None
        report['prefill_submitted_at_offset'] = (
            round(prefill_submitted_at - arrivals[0][0], 3) if arrivals else None)
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:500])
    finally:
        stop_server(process, handle)

    path = results / log_name
    if path.is_file():
        text = path.read_text(errors='replace')
        # Positive controls, same role as M1's "[M1] prefill path" markers: the
        # gate must confirm the mechanism it is testing actually engaged, not
        # infer it from timing alone.
        report['alternation_fired'] = text.count('[M2] alternation:')
        report['prefill_chunk_steps_seen'] = text.count('[M1] prefill path: resumable')
    return report


def bounded(arm):
    steady, overlap = arm.get('steady_gap_median'), arm.get('overlap_gap_max')
    if steady is None or overlap is None:
        return None
    return overlap <= steady + GAP_BUDGET_SECONDS


def decode_completed(arm):
    return (bool(arm) and not arm.get('decode_thread_alive', True)
            and not arm.get('decode_errors') and (arm.get('decode_tokens_received') or 0) > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqs', type=int, default=4)
    parser.add_argument('--results', type=Path, default=Path('/tmp/m2-gate'))
    parser.add_argument('--arm', choices=('m2', 'baseline', 'both'), default='both',
                        help='which arm to serve; CI runs one per job, mirroring the '
                             'M1 gate, because each server start opens the mesh')
    parser.add_argument('--peer', type=Path, help='the other arm report to merge in')
    parser.add_argument('--decode-tokens', type=int, default=96)
    parser.add_argument('--decode-warmup-tokens', type=int, default=8)
    parser.add_argument('--prefill-target-tokens', type=int, default=6000,
                        help='>2*chunk_size so P needs at least 3 chunk steps')
    parser.add_argument('--prefill-max-tokens', type=int, default=8)
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    report = dict(scope=__doc__, seqs=options.seqs, gap_budget_seconds=GAP_BUDGET_SECONDS,
                  decode_tokens=options.decode_tokens,
                  prefill_target_tokens=options.prefill_target_tokens)
    try:
        report['arm'] = options.arm
        if options.peer and options.peer.is_file():
            peer = json.loads(options.peer.read_text())
            for name in ('m2', 'baseline'):
                if peer.get(name):
                    report[name] = peer[name]
        if options.arm in ('m2', 'both'):
            report['m2'] = run_arm('m2', 8010, options.results, 'm2.log', options.seqs,
                                   options.decode_tokens, options.decode_warmup_tokens,
                                   options.prefill_target_tokens, options.prefill_max_tokens)
        if options.arm in ('baseline', 'both'):
            report['baseline'] = run_arm('baseline', 8011, options.results, 'baseline.log',
                                         options.seqs, options.decode_tokens,
                                         options.decode_warmup_tokens,
                                         options.prefill_target_tokens,
                                         options.prefill_max_tokens)

        m2, baseline = report.get('m2') or {}, report.get('baseline') or {}
        m2_bounded, baseline_bounded = bounded(m2), bounded(baseline)
        report['m2_bounded'] = m2_bounded
        report['baseline_bounded'] = baseline_bounded
        controls = dict(
            m2_ready=bool(m2.get('ready')),
            baseline_ready=bool(baseline.get('ready')),
            m2_decode_completed=decode_completed(m2),
            baseline_decode_completed=decode_completed(baseline),
            # P must actually need multiple chunk steps in both arms, or the
            # comparison tests a one-shot prefill that finishes too fast to
            # starve anything either way.
            m2_prefill_chunked=(m2.get('prefill_chunk_steps_seen') or 0) >= 2,
            baseline_prefill_chunked=(baseline.get('prefill_chunk_steps_seen') or 0) >= 2,
            # The mechanism under test must be observed firing, not inferred.
            m2_alternation_fired=(m2.get('alternation_fired') or 0) > 0,
            baseline_alternation_absent=(baseline.get('alternation_fired') or 0) == 0,
            m2_gap_bounded=(m2_bounded is True),
            baseline_gap_unbounded=(baseline_bounded is False),
        )
        report['controls'] = controls
        report['gate_passed'] = bool(m2) and bool(baseline) and all(controls.values())
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])

    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    print(LOG_BEGIN)
    for name in ('m2.log', 'baseline.log'):
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
