"""Geometry-only source adapter for the historical native draft attention probe."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_context_geometry import CONTEXTS, geometry


REVISION = '8c102b20df22329106955b4006bf4d650bb94e40'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Historical geometry anchor missing or ambiguous: ' + before)
    return source.replace(before, after, 1)


def adapt_probe_sources(sources, context):
    geometry(context)
    result = dict(sources)
    changes = {
        'dspark_fp32_intermediates.py': (
            ("REPLACEMENT = '''", "from frozen_context_geometry import factory_selector\nREPLACEMENT = f'''"),
            ('Skt == 272', '{factory_selector()}')),
        'run-simulator.sh': ((
            '    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}"',
            '    -e "QWEN_DSPARK_REQUEST_CONTEXT=${QWEN_DSPARK_REQUEST_CONTEXT:-8192}" \\\n    -e "QWEN_FROZEN_BUILD_ONLY=${QWEN_FROZEN_BUILD_ONLY:-0}" \\\n    -e "QWEN_FROZEN_PROBE_PART=${QWEN_FROZEN_PROBE_PART:-full}" \\\n    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}"'),),
        'dspark_attention_chunk_trial.py': (
            ('PADDED_KEYS = 8704', "from frozen_context_geometry import selected_geometry\nSHAPE = selected_geometry()\nPADDED_KEYS = SHAPE['padded_keys']"),
            ('context_rows != 8448', "context_rows != SHAPE['capacity']"),
            ('key.shape[2] != 8512', "key.shape[2] != SHAPE['storage_keys']"),
            ('PADDED_KEYS - 8512), (0, 0)', "PADDED_KEYS - SHAPE['storage_keys']), (0, 0)"),
            ('PADDED_KEYS - 8512)]', "PADDED_KEYS - SHAPE['storage_keys'])]")),
        'dspark-native-8k-attention-probe.py': (
            ("        for kind in KINDS:",
                "        part = os.environ.get('QWEN_FROZEN_PROBE_PART', 'full')\n"
                "        if part not in ('full', 'diagnostics', 'numerical'):\n"
                "            raise ValueError('Explicit frozen probe part required')\n"
                "        report['probe_part'] = part\n"
                "        report['complete_probe_coverage'] = part == 'full'\n"
                "        for kind in (() if part == 'numerical' else KINDS):"),
            ('        for case in range(2):\n            progress',
                "        if part == 'diagnostics':\n"
                "            from frozen_probe_evidence import validate_diagnostics\n"
                "            validate_diagnostics(report['value_diagnostics'])\n"
                "            if len(report['value_diagnostics']) != len(KINDS) * 2:\n"
                "                raise AssertionError('Incomplete value diagnostics')\n"
                "            report['diagnostics_complete'] = True\n"
                "            return\n\n"
                '        for case in range(2):\n            progress'),
            ("progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')",
                "progress('complete' if report['passed'] and report['closed_cleanly'] else "
                "'diagnostics_complete' if report.get('diagnostics_complete') and report['closed_cleanly'] else 'failed')"),
            ('CAPACITY, PROPOSALS = 8448, 15', "from frozen_context_geometry import selected_geometry\nSHAPE = selected_geometry()\nCAPACITY, PROPOSALS = SHAPE['capacity'], 15\nSOURCES = tuple(sorted(set(SOURCES + ('frozen_context_geometry.py', 'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py'))))"),
            ('POSITIONS = (8192, 8433)', "POSITIONS = SHAPE['positions']"),
            ('native_padded_keys=8704', "native_padded_keys=SHAPE['padded_keys']")),
        'dspark_stats_pack.py': (
            ("SELECTOR_ASSERT = '''", "from frozen_context_geometry import selected_geometry\nSELECTOR_ASSERT = f'''"),
            ('get_compile_time_arg_val(3) == 272',
                "get_compile_time_arg_val(3) == {selected_geometry()['padded_keys'] // 32}")),
    }
    for name, replacements in changes.items():
        for before, after in replacements:
            result[name] = replace_once(result[name], before, after)
    return result


def adapt_cache_launcher(sources):
    result = dict(sources)
    from frozen_sim_assets import ASSETS
    downloads = '\n'.join(f'curl --fail --location --max-time 180 {url} -o "$assets/{name}"'
        for name, url, checksum in ASSETS)
    result['run-simulator.sh'] = replace_once(result['run-simulator.sh'], downloads,
        'timeout -k 5 60 python3 -B scripts/ci/frozen_sim_assets.py '
        '--cache "$cache/frozen-simulator-assets" --destination "$assets"')
    result['run-simulator.sh'] = replace_once(result['run-simulator.sh'],
        '''        docker logs "$container" > experiment-results/simulator-container.log 2>&1 || true
        docker cp "$container:/experiment/results/." experiment-results/ || true
        docker rm -f "$container" >/dev/null || true''',
        '''        stop_status=0; logs_status=0; copy_status=0; remove_status=0
        timeout -k 1 5 docker stop -t 2 "$container" >/dev/null 2>&1 || stop_status=$?
        timeout -k 1 5 docker logs "$container" > experiment-results/simulator-container.log 2>&1 || logs_status=$?
        timeout -k 1 15 docker cp "$container:/experiment/results/." experiment-results/ || copy_status=$?
        timeout -k 1 5 docker rm -f "$container" >/dev/null || remove_status=$?
        printf '{"stop_exit":%s,"logs_exit":%s,"copy_exit":%s,"remove_exit":%s}\\n' \\
            "$stop_status" "$logs_status" "$copy_status" "$remove_status" > experiment-results/container-cleanup.json
        if [ "$status" = 0 ] && [ "$stop_status:$logs_status:$copy_status:$remove_status" != 0:0:0:0 ]; then
            status=70
        fi''')
    result['simulator-suite.sh'] = replace_once(result['simulator-suite.sh'],
        'timeout -k 30 1900 python3 -u /experiment-scripts/ci/dspark_fp32_build.py',
        'prepare_seconds=120\n'
        '            if [ "${QWEN_FROZEN_BUILD_ONLY:-0}" = 1 ]; then prepare_seconds=510; fi\n'
        '            python3 -u /experiment-scripts/ci/frozen_sim_phase.py --phase prepare --seconds "$prepare_seconds" '
        '--output /experiment/results/prepare-timing.json -- '
        'python3 -u /experiment-scripts/ci/frozen_sim_build_cache.py\n'
        '            if [ "${QWEN_FROZEN_BUILD_ONLY:-0}" = 1 ]; then exit 0; fi')
    result['simulator-suite.sh'] = replace_once(result['simulator-suite.sh'],
        'timeout -k 15 "$limit" python3 -u "/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"',
        'python3 -u /experiment-scripts/ci/frozen_sim_phase.py --phase probe --seconds 510 '
        '--output /experiment/results/probe-timing.json -- '
        'python3 -u "/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"')
    result['run-simulator.sh'] = replace_once(result['run-simulator.sh'],
        'container=$(docker create',
        'volume=qwen-frozen-simulator-f1e9b1a64b4f\n'
        'if ! docker volume inspect "$volume" >/dev/null 2>&1; then\n'
        '    docker volume create --label thatch.qwen.frozen-simulator-cache=true "$volume" >/dev/null\n'
        'fi\n'
        'test "$(docker volume inspect --format \'{{index .Labels "thatch.qwen.frozen-simulator-cache"}}\' "$volume")" = true\n'
        'container=$(docker create --mount "type=volume,src=$volume,dst=/frozen-simulator-cache"')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=CONTEXTS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    checkout = options.checkout.resolve(strict=True)

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(checkout), *arguments])

    if git('rev-parse', 'HEAD').decode().strip() != REVISION:
        raise ValueError('Exact historical recipe checkout required')
    if git('status', '--porcelain', '--untracked-files=no').strip() or options.manifest.exists():
        raise ValueError('Clean tracked checkout and fresh manifest required')
    from frozen_runtime_context import FILES, adapt_runtime_sources
    names = tuple(dict.fromkeys(('dspark_attention_chunk_trial.py', 'dspark-native-8k-attention-probe.py',
        'dspark_stats_pack.py', 'dspark_fp32_intermediates.py', 'run-simulator.sh', 'simulator-suite.sh') + FILES))
    sources = {}
    for name in names:
        original = git('show', f'{REVISION}:scripts/ci/{name}').decode()
        actual = (checkout / 'scripts/ci' / name).read_text()
        if original.replace('\r\n', '\n') != actual:
            raise ValueError('Historical source differs: ' + name)
        sources[name] = actual
    adapted = adapt_runtime_sources(adapt_cache_launcher(adapt_probe_sources(sources, options.context)))
    for name in ('frozen_context_geometry.py', 'frozen_sim_build_cache.py', 'frozen_binary_cache.py',
            'frozen_sim_phase.py', 'frozen_sim_assets.py', 'frozen_probe_evidence.py'):
        adapted[name] = Path(__file__).with_name(name).read_text()
    for name, source in adapted.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in adapted.items():
        (checkout / 'scripts/ci' / name).write_text(source)
    checksum = lambda source: hashlib.sha256(source.encode()).hexdigest()
    options.manifest.write_text(json.dumps(dict(revision=REVISION,
        geometry=geometry(options.context), before={name: checksum(source) for name, source in sources.items()},
        after={name: checksum(source) for name, source in adapted.items()},
        scope='Shared probe/runtime geometry adaptation; no numerical or runtime admission',
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
