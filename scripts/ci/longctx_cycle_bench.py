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
import re
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


def stream_once(port, prompt, max_tokens, results, index, stream_timeout=180, *, ignore_eos=True, detail=False,
                model='qwen-longctx', drop_after=None, drop_after_s=None):
    """One streaming completion; record the gap between successive tokens.

    The defaults send exactly the payload every existing caller always sent (ignore_eos=True,
    usage in the final chunk only) and record exactly the same fields. ignore_eos=False lets a
    stream end at the snapshot's EOS (real text reaches it; the fast path's GreedySession
    finishes there whatever vLLM is told). detail=True also asks vLLM for continuous usage stats
    and records, per chunk that carries tokens, its completion-token count (chunk_tokens) and its
    arrival offset from started_s (chunk_s, one time.perf_counter clock shared by the gate's
    threads), plus the stream's id (request_id: the engine id cmpl-<hex>-0-<sfx> extends it, which
    is how a '[PACKED] request=' line is tied to its user) and its finish_reason. `model` is the served
    name the request names: 'qwen-longctx' (every gate's --served-model-name) unless the caller serves
    another (the C2 serving gate's platform argv keeps the platform's, Qwen/Qwen3.8-27B).

    A client that goes away (the C2 serving gate's lifecycle arms; both off by default): drop_after=N
    closes the stream once N text chunks have arrived (N=1: during the engine build that follows the
    prefill's token); drop_after_s=S closes it if no byte has arrived S seconds after the request (a
    cancel during prefill - S is then also the inactivity limit, so a stream whose first byte beat S
    keeps going and records that it was not dropped). A drop records `dropped` (why) and what had
    arrived, and is not an error."""
    stream_options = dict(include_usage=True)
    if detail:
        stream_options['continuous_usage_stats'] = True
    payload = json.dumps(dict(model=model, prompt=prompt,
                              max_tokens=max_tokens, temperature=0.0,
                              stream=True,
                              stream_options=stream_options,
                              ignore_eos=ignore_eos)).encode()
    request = Request('http://127.0.0.1:%d/v1/completions' % port, data=payload,
                      headers={'Content-Type': 'application/json'})
    gaps, tokens, started = [], 0, time.perf_counter()
    # The whole generated text, so two runs on the same prompts can be compared
    # token for token offline: the packed step's gate is equality with the
    # sequential step's output, and greedy decoding makes that exact.
    pieces = []
    first_token_at = None
    entry = {}
    details = dict(request_id=None, finish_reason=None, started_s=started, chunk_s=[], chunk_tokens=[]) if detail else None
    completed = 0
    try:
        # The socket timeout is the inactivity limit between chunks. At 900 s a hung
        # decode sat for fifteen minutes (run 35481903377) and produced no log; 180 s
        # is nine times the 32k-token TTFT and hundreds of times the inter-token gap.
        # It is configurable because four users admit one prefill at a time (one-in-
        # flight): the fourth user's first byte lands only after three ~78 s prefills,
        # ~312 s, so measuring a four-user packed round needs the limit raised above
        # that ramp until prefill/decode alternation removes it.
        with urlopen(request, timeout=drop_after_s if drop_after_s is not None else stream_timeout) as response:
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
                if chunk.get('error') is not None:
                    # vLLM reports a rejected request inside the stream. Without this
                    # the loop simply saw no 'text' and reported zero tokens with no
                    # reason, which is what run 35418622804 did.
                    entry['error'] = str(chunk['error'])[:300]
                    break
                usage = chunk.get('usage')
                if usage:
                    entry['prompt_tokens'] = usage.get('prompt_tokens')
                    entry['completion_tokens'] = usage.get('completion_tokens')
                text = (chunk.get('choices') or [{}])[0].get('text', '')
                if details is not None:
                    if details['request_id'] is None and chunk.get('id'):
                        details['request_id'] = chunk['id']
                    choices = chunk.get('choices') or []
                    if choices:
                        if choices[0].get('finish_reason'):
                            details['finish_reason'] = choices[0]['finish_reason']
                        count = (usage or {}).get('completion_tokens')
                        delta = None if count is None else count - completed
                        if count is not None:
                            completed = count
                        if text or delta:
                            details['chunk_s'].append(round(time.perf_counter() - started, 6))
                            details['chunk_tokens'].append(delta)
                if not text:
                    continue
                now = time.perf_counter()
                tokens += 1
                pieces.append(text)
                if first_token_at is None:
                    first_token_at = now
                elif previous is not None:
                    gaps.append(now - previous)   # first token deliberately dropped
                previous = now
                if drop_after is not None and tokens >= drop_after:
                    entry['dropped'] = 'after %d chunks' % tokens
                    break
        entry.update(tokens=tokens, gaps_ms=[1000.0 * g for g in gaps], text=''.join(pieces),
                     ttft_s=(first_token_at - started) if first_token_at else None,
                     wall_s=time.perf_counter() - started)
        if details is not None:
            entry.update(details)
        if drop_after_s is not None and 'dropped' not in entry:
            entry['dropped'] = None   # the first byte beat drop_after_s: the stream ran to its end
        results[index] = entry
    except BaseException as error:
        timed_out = isinstance(error, (TimeoutError, OSError, URLError)) and 'timed out' in str(error)
        if drop_after_s is not None and not pieces and timed_out:
            results[index] = dict(dropped='no byte within %s s' % drop_after_s, tokens=0, gaps_ms=[], text='',
                                  ttft_s=None, wall_s=time.perf_counter() - started)
            if details is not None:
                results[index].update(details)
            return
        # Keep what arrived before the failure: a hang after N tokens and a refusal
        # at admission are different findings.
        results[index] = dict(error='%s: %s' % (type(error).__name__, str(error)[:300]),
                              tokens=tokens, gaps_ms=[1000.0 * g for g in gaps], text=''.join(pieces),
                              ttft_s=(first_token_at - started) if first_token_at else None,
                              wall_s=time.perf_counter() - started)
        if details is not None:
            results[index].update(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    # Probe 35436384975: TTScheduler batches SIMULTANEOUS prefills into one step
    # (new=['A','B']) but serialises a LATER arrival (new=['B'] cached=[]). Starting
    # every thread at once therefore exercises the batched case, which the fast path
    # cannot serve at all - one capture and one prompt per prefill - while the
    # staggered case is the one the current code is built for.
    parser.add_argument('--stagger', type=float, default=0.0,
                        help='seconds between user starts; 0 starts them together')
    parser.add_argument('--plain', action='store_true',
                        help='serve without the T16 fast runtime or speculation. The fast '
                             'path admits one request, so this is the only way to measure '
                             'how decode cost scales with concurrent users on this stack')
    parser.add_argument('--context', type=int, default=163840)
    parser.add_argument('--prompt-tokens', type=int, default=512)
    # Exact token ids [base + i % 64]; a per-user offset gives each user its own
    # prompt, which is how a corrupted user is told apart from a coincidence: two
    # users on ONE prompt (run 35492676194) agreed for a chunk before differing.
    parser.add_argument('--prompt-base', type=int, default=1000)
    parser.add_argument('--prompt-user-offset', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--stream-timeout', type=int, default=180,
                        help='per-stream socket inactivity limit in seconds; raise above the '
                             'one-in-flight prefill ramp (~(users-1) x 78 s) to measure a '
                             'four-user packed round before alternation removes the ramp')
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
    report = dict(scope=__doc__, users=options.users, stagger=options.stagger,
                  context=options.context,
                  plain=options.plain,
                  blocks=blocks, target_tokens_per_user=TARGET_TOKS_PER_USER,
                  target_itl_ms=1000.0 / TARGET_TOKS_PER_USER, ready=False)

    # The tt block is load-bearing on both paths: without it the plugin opens the
    # mesh with l1_small_size=0 and the first L1_SMALL allocation dies at 'bank size
    # is 0 B'. Only the fast-path keys are conditional.
    recipe = dict(tt=dict(trace_mode='decode_only',
                          trace_region_size=options.trace_region_size,
                          l1_small_size=options.l1_small_size))
    if not options.plain:
        recipe['qwen_fast_t16'] = True
        recipe['qwen_fast_runtime'] = dict(directory='/experiment-scripts/ci',
                                           runtime_root='/opt/tt-metal',
                                           fixtures='/experiment-dflash-fixture',
                                           target_snapshot=options.model)
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', options.model, '--served-model-name', 'qwen-longctx',
               '--host', '127.0.0.1', '--port', str(options.port), '--dtype', 'bfloat16',
               '--max-model-len', str(options.context),
               '--max-num-seqs', str(options.users),
               '--max-num-batched-tokens', str(options.context),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--no-enable-chunked-prefill', '--shutdown-timeout', '30',
               '--additional-config', json.dumps(recipe)]
    if not options.plain:
        command += ['--speculative-config', json.dumps(
            dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                 draft_sample_method='greedy', rejection_sample_method='standard'))]
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

        # Eight, not six. The phrase runs about six BPE tokens, so dividing by six
        # targeted the whole budget and overshot max_model_len when the real count
        # ran high; run 35418622804 then had every request rejected in-band and
        # reported zero tokens with no reason. Undershooting costs a little KV
        # occupancy and keeps the request servable, and the usage field now reports
        # what the prompt actually came to.
        # An EXACT token count, as a token-id array rather than text. The text
        # form was an approximation: run 35473721637 asked for 32768 and the
        # server reported position=20488. Every qualification gate on this path
        # compares position for EQUALITY, so an approximate prompt can never
        # satisfy one, at any user count.
        def prompt_for(user):
            base = options.prompt_base + user * options.prompt_user_offset
            return [base + (index % 64) for index in range(options.prompt_tokens)]
        report['prompt_base'] = options.prompt_base
        report['prompt_user_offset'] = options.prompt_user_offset
        # ignore_eos is what makes this a latency measurement rather than a content
        # one. Run 35418922350 accepted a 79,368-token prompt, spent 33.4 s on it and
        # returned zero tokens with no error: the stream completed normally because
        # the model emitted EOS as its first token. A prompt that repeats one phrase
        # ten thousand times invites exactly that, and a fixed decode count is what
        # the inter-token latency needs regardless.
        results = [None] * options.users
        threads = [threading.Thread(target=stream_once,
                                    args=(options.port, prompt_for(index), options.max_tokens,
                                          results, index, options.stream_timeout))
                   for index in range(options.users)]
        wall = time.perf_counter()
        for index, thread in enumerate(threads):
            if index and options.stagger:
                time.sleep(options.stagger)
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
        # Chunks are not tokens: one verify commits several accepted tokens in one
        # SSE chunk (run 35490298652: 64 tokens in 10 chunks), so the gap-based rate
        # above is chunks per second. The rate that matters is completion tokens
        # over each user's decode window, from the server's own usage report.
        rates = []
        for entry in results:
            if (not entry or not entry.get('completion_tokens') or entry.get('ttft_s') is None
                    or not entry.get('wall_s')):
                continue
            window = entry['wall_s'] - entry['ttft_s']
            if window > 0 and entry['completion_tokens'] > 1:
                rates.append((entry['completion_tokens'] - 1) / window)
        if rates:
            report['decode_tokens_per_user_per_s'] = [round(rate, 2) for rate in rates]
            report['decode_tokens_per_user_per_s_median'] = round(statistics.median(rates), 2)
            report['decode_tokens_aggregate_per_s'] = round(sum(rates), 2)
            report['fraction_of_target_tokens'] = round(statistics.median(rates) / TARGET_TOKS_PER_USER, 4)
        # Positive control for any experiment that retunes the prefill chunk. The model
        # does chunk_size = self._chunked_chunk_size or 2048, so a constant that never
        # reaches the warmup falls back silently and two arms measure the same thing.
        # Run 35423170257 compared 2048 against 4096 and they agreed to 0.1%, which is
        # what a lever that did not move looks like.
        try:
            text = log_path.read_text(errors='replace')
            found = re.findall(r'\[CHUNK\] prefill chunk_size=(\d+)', text)
            report['prefill_chunk_observed'] = sorted({int(v) for v in found}) or None
        except BaseException as error:
            report['prefill_chunk_observed'] = '%s' % type(error).__name__
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
            lines = log_path.read_text(errors='replace').splitlines()
            # Diagnostic lines first, from the WHOLE log: the attach-time stage line
            # with the pooled and shared addresses prints thousands of lines before
            # the tail below, and the container it lives in is discarded.
            # [PHASE] and [CARRY] too: run 35483704438 had none in the tail, and
            # whether they were never written or scrolled off is the finding.
            # Bounded from both ends: the first lines hold the attach-time
            # addresses, the last ones hold the step the run died in.
            # The request lifecycle lines too: whether the second prompt reached
            # the engine, and when, was unreadable in run 35484349353 because
            # both arrivals had scrolled out of the tail.
            diagnostic = [line[:600] for line in lines
                          if '[PINDIAG]' in line or '"stage"' in line
                          or '[PHASE]' in line or '[CARRY]' in line
                          or 'Received request' in line or 'Added request' in line
                          or 'bort' in line or '/v1/completions' in line
                          or 'Running: ' in line]
            if len(diagnostic) > 800:
                omitted = len(diagnostic) - 800
                diagnostic = (diagnostic[:400] + ['... %d diagnostic lines omitted' % omitted]
                              + diagnostic[-400:])
            for line in diagnostic:
                print(line)
            for line in lines[-400:]:
                print(line[:300])
        print(LOG_END)
        # TT-Metal's watcher, when enabled, records each RISC's last waypoint: a
        # stalled core names the class of hang. It lives inside the container.
        watcher = Path('/opt/tt-metal/generated/watcher/watcher.log')
        print('<<<CYCLE_BENCH_WATCHER_BEGIN>>>')
        if watcher.is_file():
            # Run 35483320919 tripped a kernel assert, and a blind 200-line tail held
            # only idle cores and the kernel table. Keep the lines that carry the
            # finding: the assert text, any core not parked at a wait, the kernel id
            # table, and the dump headers - from the whole file, bounded.
            idle = re.compile(r'^Device \d+ worker core.*:\s+GW,\s+W,\s+W,\s+W,\s+W\s')
            keep = re.compile(r'assert|tripped|halt|exception|Last waypoint|k_id\[|Dump #|^Legend|noc|sanit|stalled', re.I)
            finding = re.compile(r'assert|tripped|halt|exception|Last waypoint|sanit|stalled|While running', re.I)
            text = watcher.read_text(errors='replace')
            # Every dump repeats the kernel table, so 600 lines from the FRONT of a
            # long run end before the dump that matters: in run 35483704438 one
            # mid-prefill dump alone was 270 lines. Findings from anywhere, then
            # the last two dumps.
            dumps = re.split(r'(?m)^(?=Dump #\d+ at )', text)
            printed = 0
            for line in text.splitlines():
                if finding.search(line):
                    print(line[:300])
                    printed += 1
                    if printed >= 80:
                        print('... watcher finding lines truncated')
                        break
            print('--- last %d of %d watcher dumps ---' % (min(2, len(dumps)), len(dumps)))
            kept = 0
            for dump in dumps[-2:]:
                for line in dump.splitlines():
                    if keep.search(line) or (line.startswith('Device') and 'worker core' in line and not idle.match(line)):
                        print(line[:300])
                        kept += 1
                        if kept >= 600:
                            break
                if kept >= 600:
                    print('... watcher lines truncated')
                    break
        print('<<<CYCLE_BENCH_WATCHER_END>>>')
        sys.stdout.flush()
    return 0 if report.get('ready') else 1


if __name__ == '__main__':
    sys.exit(main())
