"""Unqualified diagnostic scopes; never a throughput candidate or serving default."""

from frozen_recipe_context import replace_once


HEADER = '#include "tools/profiler/kernel_profiler.hpp"\n'
PROFILER_ENV = 'TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1'
ZONES = {
    'input': (
        ('cb_reserve_back(0, 8);', 'QWEN_MLP_INPUT_FREE',
            'block == 10 && worker <= 1'),
        ('noc_async_read_barrier();', 'QWEN_MLP_INPUT_READ', 'block == 10'),
        ('noc_semaphore_wait(ready, receivers);', 'QWEN_MLP_RECEIVERS_READY', 'block == 10'),
        ('noc_async_write_barrier();', 'QWEN_MLP_INPUT_MCAST', 'block == 10'),
        ('noc_semaphore_wait(received, 1);', 'QWEN_MLP_INPUT_RECEIVED',
            'block == 10 && worker == 1'),
    ),
    'weights': (
        ('cb_reserve_back(1, 16 * pairs_per_worker);', 'QWEN_MLP_WEIGHT_FREE',
            'block == 10 && first_pair == 0'),
        ('noc_async_read_barrier();', 'QWEN_MLP_WEIGHT_READ',
            'block == 10 && first_pair == 0'),
        ('cb_wait_front(4, output_tiles);', 'QWEN_MLP_OUTPUT_READY',
            'pair == 0 && first_pair == 0'),
        ('noc_async_write_barrier();', 'QWEN_MLP_OUTPUT_WRITE',
            'pair == 0 && first_pair == 0'),
    ),
}


def _before_exit_drain(source):
    """Split off the exit drain commit 4d890d6a added after the block loop.

    Every zone predicate here is written in terms of `block`, which only exists
    inside that loop, so the drain can never be a valid instrumentation target -
    wrapping its barrier would not even compile. Its own
    noc_async_write_barrier() nonetheless made the bare statement anchor match
    twice, which is what replace_once refused. Splitting first keeps every zone
    anchor and every emitted string byte-identical to what was reviewed.

    The tail is not a blind spot: it must be exactly that drain and the closing
    brace, comments aside. Anything else after the loop - including source drift
    appended to the end - is refused here rather than quietly skipped.
    """
    marker = source.find('    // Drain before exit:')
    if marker < 0:
        return source, ''
    body, drain = source[:marker], source[marker:]
    code = [line.strip() for line in drain.split('\n')
            if line.strip() and not line.strip().startswith('//')]
    if code != ['noc_async_write_barrier();', 'noc_async_atomic_barrier();', '}']:
        raise ValueError('Unexpected source after the exit drain: %r' % (code,))
    return body, drain


def scoped_statement(statement, name, predicate):
    return ('{\n'
        f'            if ({predicate}) {{\n'
        f'                DeviceZoneScopedN("{name}");\n'
        f'                {statement}\n'
        '            } else {\n'
        f'                {statement}\n'
        '            }\n'
        '        }')


def instrument(source, role):
    if role not in ZONES:
        raise ValueError('Unknown MLP dataflow role')
    if 'DeviceZoneScopedN' in source or HEADER in source:
        raise ValueError('Already instrumented source is not admitted')
    result = replace_once(source, '#include "api/dataflow/dataflow_api.h"\n',
        '#include "api/dataflow/dataflow_api.h"\n' + HEADER)
    result, drain = _before_exit_drain(result)
    for statement, name, predicate in ZONES[role]:
        result = replace_once(result, statement, scoped_statement(statement, name, predicate))
    result += drain
    if remove_scopes(result, role) != source:
        raise ValueError('Instrumentation changed operations outside sampled scopes')
    return result


def remove_scopes(source, role):
    result = replace_once(source, HEADER, '')
    for statement, name, predicate in ZONES[role]:
        result = replace_once(result, scoped_statement(statement, name, predicate), statement)
    return result


def profile_trace_source(source):
    statements = {
        'operations.execute_trace(mesh, trace, cq_id=0, blocking=True)': 2,
        'operations.execute_trace(mesh, traces[name], cq_id=0, blocking=True)': 1,
    }
    lines = source.splitlines(keepends=True)
    if any(sum(line.strip() == statement for line in lines) != count
            for statement, count in statements.items()):
        raise ValueError('Exact replay call sites required')
    result = ''.join(line + (line[:len(line) - len(line.lstrip())]
        + 'operations.ReadDeviceProfiler(mesh)\n' if line.strip() in statements else '')
        for line in lines)
    return replace_once(result, '    finally:\n        for trace in traces.values():',
        '    finally:\n        operations.synchronize_device(mesh)\n'
        '        operations.ReadDeviceProfiler(mesh)\n        for trace in traces.values():')


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    from frozen_recipe_context import REVISION
    from frozen_mlp_buffer_trial import adapt_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--unmodified-control', action='store_true')
    parser.add_argument('--mid-run-dump', action='store_true')
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh diagnostic manifest required')
    scripts = options.checkout / 'scripts/ci'
    sources = {}
    for name in ('fused_1d_input.cpp', 'fused_1d_weights.cpp', 'fused-batch-probe.py', 'fusion_trace.py'):
        source = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != source:
            raise ValueError('Historical source required: ' + name)
        sources[name] = source
    adapted = {f'fused_1d_{role}.cpp': instrument(sources[f'fused_1d_{role}.cpp'], role)
        for role in ZONES}
    if options.unmodified_control:
        adapted = {name: sources[name] for name in adapted}
    adapted['fused-batch-probe.py'] = replace_once(adapt_probe(sources['fused-batch-probe.py']),
        "T16 buffering only; other row widths and performance unqualified",
        ("T16 unmodified profiler control; no diagnostic kernel qualification"
            if options.unmodified_control else
            "T16 sampled diagnostic scopes only; profiler export and performance unqualified"))
    profiler_env = PROFILER_ENV
    if options.mid_run_dump:
        profiler_env += ' TT_METAL_PROFILER_MID_RUN_DUMP=1'
        adapted['fusion_trace.py'] = profile_trace_source(sources['fusion_trace.py'])
    adapted['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        profiler_env + ' timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    if options.mid_run_dump:
        adapted['simulator-suite.sh'] = replace_once(adapted['simulator-suite.sh'],
            'cd /opt/tt-metal\n',
            'cd /opt/tt-metal\n'
            "preserve_profiler() {\npython3 - <<'PY'\n"
            'from pathlib import Path\nimport shutil\n'
            'from tracy.common import PROFILER_LOGS_DIR\n'
            'source = Path(PROFILER_LOGS_DIR)\n'
            "if source.is_dir():\n    shutil.copytree(source, '/experiment/results/raw-profiler', dirs_exist_ok=True)\n"
            'PY\n}\ntrap preserve_profiler EXIT\n')
    for name, source in adapted.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in adapted.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in adapted.items()},
        diagnostic_only=True, unmodified_control=options.unmodified_control,
        mid_run_dump=options.mid_run_dump,
        profiler_requested=True, simulator_qualified=False,
        hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
