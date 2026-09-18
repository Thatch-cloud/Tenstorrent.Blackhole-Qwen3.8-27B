"""Launch a disposable loopback-only fast vLLM canary, never the production service."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def main():
    if (os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit hardware canary allocation required')
    target = '/models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    results = Path('/experiment/results')
    results.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '/experiment-scripts/ci/device-owners.py'], check=True, timeout=20)
    recipe = dict(qwen_fast_t16=True, qwen_fast_runtime=dict(directory='/experiment-scripts/ci',
        runtime_root='/opt/tt-metal', fixtures='/experiment-dflash-fixture', target_snapshot=target))
    speculative = dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
        draft_sample_method='greedy', rejection_sample_method='standard')
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', target,
        '--served-model-name', 'qwen-fast-canary', '--host', '127.0.0.1', '--port', '8000',
        '--dtype', 'bfloat16', '--max-model-len', '4352', '--max-num-seqs', '1',
        '--max-num-batched-tokens', '4352', '--block-size', '64', '--num-gpu-blocks-override', '128',
        '--no-enable-prefix-caching', '--no-async-scheduling', '--no-enable-chunked-prefill',
        '--speculative-config', json.dumps(speculative), '--additional-config', json.dumps(recipe)]
    report = dict(passed=False, stage='starting', serving_qualified=False, performance_qualified=False)
    started = time.perf_counter()
    process = None
    try:
        with (results / 'server.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 480
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Canary exited before readiness: {process.returncode}; see server.log')
                try:
                    with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Canary startup exceeded eight-minute bound')
                time.sleep(1)
            report.update(stage='http_reference_checks', startup_seconds=time.perf_counter() - started)
            print(json.dumps(report), flush=True)
            subprocess.run([sys.executable, '/canary/serving_canary_client.py',
                '--reference', '/canary/reference.json', '--output', str(results / 'http-reference.json')],
                check=True, timeout=360)
            report.update(passed=True, stage='http_checks_complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                report['forced_shutdown'] = True
                report['passed'] = False
            report['server_exit_code'] = process.returncode
            if process.returncode not in (0, -15):
                report['passed'] = False
        report['elapsed_seconds'] = time.perf_counter() - started
        (results / 'canary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('Canary did not complete normally')


if __name__ == '__main__':
    main()
