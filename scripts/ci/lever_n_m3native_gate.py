"""Lever N M3native gate: is the native 64-row decode graft token-exact against each
user's own single-stream reference, at four packed users, while retiring the two-call
MLP and GDN-output wrappers?

Serves the SAME four-user packed round the qwen-fp2u-image.yml lane measures (the
base-1000..1003 prompt scheme, longctx_cycle_bench.py's own streaming/parsing logic
reused here via import) with the M3native graft mounted over the model sources
(lever_n_m3native_run_arm.sh), then asserts three things design section 5 (this
graft's own scope note) requires:

  1. Each stream's text is a byte-exact PREFIX match against its single-user
     reference (runner-evidence.local/packed-gate/single-user-*.json), compared up
     to the reference's own length - the packed round may run more decode tokens
     than the reference did, but everything the reference covers must agree exactly.
     Anything else means the native path diverged from the two-call one it replaces.
  2. The round's [PACKED-PHASE] trace_ms is reported (min/mean/max over every packed
     round in the log) - not asserted against a threshold here (that is a benchmark
     question, not a correctness one), just surfaced for the run to be read against
     the two-call baseline (run 35544598063: 1453 ms/round).
  3. The [PINDIAG] native_m3 marker is present (the positive control that the graft
     actually engaged, not that the model happened to produce the right bytes some
     other way) and every "[PINDIAG] native_m3 binder calls this round" line
     (model_batch.ModelBatch.run) reports zero calls for the retired binders - a
     silent fallback to the two-call MLP/GDN-output wrappers is exactly the failure
     mode this control exists to catch instead of measuring a graft that is not
     actually native.

Runs on the fast T16 path with speculation, like the four-user cycle bench: this is
what the target 200 tok/s/user work measures, and the packed-step/audit env vars
that produce [PACKED-PHASE] lines are set by lever_n_m3native_run_arm.sh at
`docker run`, not here - they are inherited by this process and by the vLLM server
subprocess it starts.

K64 KERNEL GRAFT (KOPGRAFT64, run_arm's optional mount block). A second, independent
graft: the batch-64 attn_decode_prep and nlp_concat_heads_decode C++ kernels
themselves, mounted only when KOPGRAFT64 is set (QWEN_FAST_NATIVE_ATTN=1 then reaches
the container). Not part of the pass/fail criteria above - this gate is run once
without it (the plain graft) and once with it, so `native_attn_engaged`
(NATIVE_ATTN_MARKER, the "[PINDIAG] native_attn engaged" line two_tile_bindings logs
once) is recorded in the JSON for the run to be read against, not asserted here.
"""

import argparse
import ast
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longctx_cycle_bench import stream_once  # reuse the exact streaming/parsing logic

BEGIN = '<<<M3NATIVE_GATE_JSON_BEGIN>>>'
END = '<<<M3NATIVE_GATE_JSON_END>>>'
LOG_BEGIN = '<<<M3NATIVE_GATE_LOG_BEGIN>>>'
LOG_END = '<<<M3NATIVE_GATE_LOG_END>>>'

BLOCK_SIZE = 64
MODEL = ('/models/hub/models--Qwen--Qwen3.8-27B/snapshots/'
         '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
NATIVE_M3_MARKER = '[PINDIAG] native_m3'
NATIVE_ATTN_MARKER = '[PINDIAG] native_attn engaged'
PACKED_PHASE_TRACE_MS = re.compile(r'\[PACKED-PHASE\][^\n]*\btrace_ms=([0-9.]+)')
BINDER_CALLS_LINE = re.compile(r'\[PINDIAG\] native_m3 binder calls this round: (\{.*\})')
# The binder-calls diagnostic (model_batch.ModelBatch.run) lists EVERY two-tile binder,
# retired or not: 'decode norm' (129 = two per layer plus the final norm) and
# 'full-attention forward' (16) are expected to be non-zero in every arm, and run
# 35559199392 was reported NOT PASSED on exactly those two. Only the binders the
# native path retires must be zero: the MLP and GDN-output wrappers under native_m3,
# and the sliced prep / two-tile concat guards once QWEN_FAST_NATIVE_ATTN retires them.
RETIRED_LABELS = ('MLP forward', 'GDN output projection', 'sliced attn_decode_prep', 'two-tile head concat')
REFERENCE_NAME = re.compile(r'^single-user-(?:(\d{4})-)?\d+\.json$')


def load_references(directory):
    """Each user's single-stream reference text, keyed by its prompt base.

    single-user-35492921706.json (base 1000, no offset) carries no digit group; the
    others are named single-user-<base>-<run>.json.
    """
    references = {}
    directory = Path(directory)
    if not directory.is_dir():
        return references
    for path in sorted(directory.glob('single-user*.json')):
        match = REFERENCE_NAME.match(path.name)
        if not match:
            continue
        base = int(match.group(1)) if match.group(1) else 1000
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        streams = data.get('streams') or []
        if not streams:
            continue
        references[base] = dict(path=str(path), text=streams[0].get('text', ''),
                                text_sha256=streams[0].get('text_sha256'))
    return references


def start_server(port, users, context, results, log_name, readiness_seconds=900,
                 trace_region_bytes=1073741824):
    """The fast T16 + speculation recipe the four-user cycle bench serves
    (qwen-fp2u-image.yml), so the packed round under test is the one the 200
    tok/s/user work actually measures.

    --served-model-name is 'qwen-longctx', not this gate's own name: stream_once
    (longctx_cycle_bench.py, reused here exactly) hard-codes model='qwen-longctx' in
    its request payload, so any other served name 404s every stream in milliseconds
    (gate 1, run 35556533480 - a false negative that looked like readiness with zero
    decode rounds actually run)."""
    # trace_region_bytes: the recipe's 1 GiB unless the profile arm shrinks it - the device
    # profiler reserves ~0.72 GB of DRAM per chip (run 35563019626: 32.38 GB allocatable
    # instead of 33.10, and the fourth user's first proposal hit Out of Memory), while
    # decode traces are command streams that this repo's own probes run at 256 MiB.
    recipe = dict(tt=dict(trace_mode='decode_only', trace_region_size=int(trace_region_bytes),
                          l1_small_size=24576),
                 qwen_fast_t16=True,
                 qwen_fast_runtime=dict(directory='/experiment-scripts/ci', runtime_root='/opt/tt-metal',
                                        fixtures='/experiment-dflash-fixture', target_snapshot=MODEL))
    blocks = -(-(users * context) // BLOCK_SIZE)
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', 'qwen-longctx',
               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
               '--max-model-len', str(context), '--max-num-seqs', str(users),
               '--max-num-batched-tokens', str(context),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               '--no-enable-chunked-prefill', '--shutdown-timeout', '30',
               '--additional-config', json.dumps(recipe),
               '--speculative-config', json.dumps(
                   dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                        draft_sample_method='greedy', rejection_sample_method='standard'))]
    log_path = results / log_name
    handle = log_path.open('w')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + readiness_seconds
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
            raise TimeoutError('readiness exceeded %ds' % readiness_seconds)
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


def prompt_for(base, offset, user, tokens):
    """Exact token ids [base' + i % 64], matching longctx_cycle_bench's own scheme
    (a per-user offset, so a corrupted user is told apart from a coincidence)."""
    start = base + user * offset
    return [start + (index % 64) for index in range(tokens)]


def packed_phase_stats(text):
    values = [float(v) for v in PACKED_PHASE_TRACE_MS.findall(text)]
    if not values:
        return None
    return dict(rounds=len(values), trace_ms_min=round(min(values), 3),
               trace_ms_mean=round(statistics.fmean(values), 3), trace_ms_max=round(max(values), 3))


def compare_prefix(actual, reference_text):
    """Whether `actual` (this run's stream) is consistent with `reference_text` (the
    single-user reference).

    A full-length or longer stream must match the reference exactly over the
    reference's own length - unchanged from the original all-length semantics. A
    stream shorter than the reference (as a profiling run capped at a small
    --max-tokens produces) cannot be checked that way, since `actual` never reaches
    the reference's length; instead it must itself be an exact prefix of the
    reference. Returns (identical_prefix, partial)."""
    if len(actual) < len(reference_text):
        return reference_text.startswith(actual), True
    return actual[:len(reference_text)] == reference_text, False


def retired_binder_leaks(rounds):
    """The rounds in which a RETIRED binder (RETIRED_LABELS) saw a call; every other
    label in the payload is a binder that is meant to run and is ignored here."""
    return [{label: calls for label, calls in payload.items() if label in RETIRED_LABELS and calls}
            for payload in rounds
            if any(payload.get(label) for label in RETIRED_LABELS)]


def retired_binder_rounds(text):
    """Every per-round '[PINDIAG] native_m3 binder calls this round: {...}' payload
    (model_batch.ModelBatch.run), each mapping a retired binder's label to how many
    calls leaked through it this round - every value here must be zero."""
    rounds = []
    for match in BINDER_CALLS_LINE.finditer(text):
        try:
            payload = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, dict):
            rounds.append(payload)
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--context', type=int, default=33024)
    parser.add_argument('--prompt-tokens', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--stream-timeout', type=int, default=600)
    parser.add_argument('--prompt-base', type=int, default=1000)
    parser.add_argument('--prompt-user-offset', type=int, default=1)
    parser.add_argument('--stagger', type=float, default=0.0)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--results', type=Path, default=Path('/tmp/m3native-gate'))
    parser.add_argument('--trace-region-bytes', type=int, default=1073741824,
                        help='device trace region per chip (the recipe uses 1 GiB; the profile arm passes less)')
    parser.add_argument('--references', type=Path,
                        default=Path('runner-evidence.local/packed-gate'),
                        help='directory holding single-user-*.json single-stream references')
    options = parser.parse_args()
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    report = dict(scope=__doc__, users=options.users, context=options.context,
                 prompt_tokens=options.prompt_tokens, prompt_base=options.prompt_base,
                 prompt_user_offset=options.prompt_user_offset, ready=False)
    process = handle = None
    try:
        references = load_references(options.references)
        report['references_loaded'] = sorted(references)

        process, handle, log_path, command = start_server(
            options.port, options.users, options.context, options.results, 'server.log',
            trace_region_bytes=options.trace_region_bytes)
        report['command'] = command
        report['ready'] = True

        results = [None] * options.users
        threads = [threading.Thread(
            target=stream_once,
            args=(options.port, prompt_for(options.prompt_base, options.prompt_user_offset, index,
                                           options.prompt_tokens),
                  options.max_tokens, results, index, options.stream_timeout))
            for index in range(options.users)]
        for index, thread in enumerate(threads):
            if index and options.stagger:
                time.sleep(options.stagger)
            thread.start()
        for thread in threads:
            thread.join()
        report['streams'] = results

        comparisons = []
        for index, entry in enumerate(results):
            base = options.prompt_base + index * options.prompt_user_offset
            reference = references.get(base)
            comparison = dict(user=index, prompt_base=base, reference_present=bool(reference))
            if reference:
                comparison['reference_path'] = reference['path']
                comparison['reference_len'] = len(reference['text'])
                actual = (entry or {}).get('text') or ''
                comparison['actual_len'] = len(actual)
                identical_prefix, partial = compare_prefix(actual, reference['text'])
                comparison['identical_prefix'] = identical_prefix
                if partial:
                    comparison['partial'] = True
                if entry and entry.get('error'):
                    comparison['error'] = entry['error']
            comparisons.append(comparison)
        report['comparisons'] = comparisons

        log_text = log_path.read_text(errors='replace') if log_path.is_file() else ''
        report['native_m3_marker_present'] = NATIVE_M3_MARKER in log_text
        report['native_attn_engaged'] = NATIVE_ATTN_MARKER in log_text
        report['packed_phase'] = packed_phase_stats(log_text)
        binder_rounds = retired_binder_rounds(log_text)
        report['retired_binder_rounds_observed'] = len(binder_rounds)
        report['retired_binder_calls_nonzero'] = retired_binder_leaks(binder_rounds)

        checked = [c for c in comparisons if c.get('reference_present')]
        report['users_checked'] = len(checked)
        report['gate_passed'] = bool(
            report['ready']
            and len(checked) == options.users
            and all(c.get('identical_prefix') for c in checked)
            and report['native_m3_marker_present']
            and report['packed_phase'] is not None
            and bool(binder_rounds)
            and not report['retired_binder_calls_nonzero']
        )
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    finally:
        stop_server(process, handle)
        print(BEGIN)
        print(json.dumps(report, indent=2))
        print(END)
        print(LOG_BEGIN)
        log_path = options.results / 'server.log'
        if log_path.is_file():
            lines = log_path.read_text(errors='replace').splitlines()
            diagnostic = [line[:300] for line in lines
                         if '[PINDIAG]' in line or '[PACKED-PHASE]' in line or '[PACKED-COMMIT]' in line or '[PHASE]' in line
                         or 'ERROR' in line or 'Traceback' in line]
            if len(diagnostic) > 800:
                omitted = len(diagnostic) - 800
                diagnostic = diagnostic[:400] + ['... %d diagnostic lines omitted' % omitted] + diagnostic[-400:]
            for line in diagnostic:
                print(line)
            for line in lines[-200:]:
                print(line[:300])
        print(LOG_END)
        sys.stdout.flush()
    return 0 if report.get('gate_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
