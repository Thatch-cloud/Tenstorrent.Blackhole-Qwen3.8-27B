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


def adapt_cache_launcher(sources, probe_seconds=510):
    if type(probe_seconds) is not int or probe_seconds not in (510, 1020):
        raise ValueError('Explicit supported probe budget required')
    result = dict(sources)
    result['run-simulator.sh'] = replace_once(result['run-simulator.sh'],
        '    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}"',
        '    -e "QWEN_FROZEN_TARGET_SCRATCH=${QWEN_FROZEN_TARGET_SCRATCH:-0}" \\\n'
        '    -e "QWEN_SDPA_TREE_SCRATCH_ROUNDS=${QWEN_FROZEN_TARGET_SCRATCH:-0}" \\\n'
        '    -e "QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}"')
    result['simulator-suite.sh'] = replace_once(result['simulator-suite.sh'],
        'if [[ "$QWEN_SIM_CASE" = dspark-native-8k-attention ]]; then',
        'if [[ "$QWEN_SIM_CASE" = dspark-native-8k-attention || "${QWEN_FROZEN_TARGET_SCRATCH:-0}" = 1 ]]; then')
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
        'env QWEN_SIM_CASE=dspark-native-8k-attention python3 -u /experiment-scripts/ci/frozen_sim_build_cache.py\n'
        '            if [ "${QWEN_FROZEN_BUILD_ONLY:-0}" = 1 ]; then exit 0; fi')
    result['simulator-suite.sh'] = replace_once(result['simulator-suite.sh'],
        'timeout -k 15 "$limit" python3 -u "/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"',
        f'python3 -u /experiment-scripts/ci/frozen_sim_phase.py --phase probe --seconds {probe_seconds} '
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


def adapt_scalar_reciprocal(sources):
    result = dict(sources)
    name = 'dspark-native-8k-attention-probe.py'
    result[name] = replace_once(result[name], '    with scoped_stats_pack(),',
        '    from dspark_ladder_scalar_reciprocal import scalar_reciprocal\n'
        '    with scalar_reciprocal(), scoped_stats_pack(),')
    result[name] = replace_once(result[name], "backend='simulator', scope=__doc__,",
        "backend='simulator', scope=__doc__, reciprocal_variant='scalar-fp32',")
    result[name] = replace_once(result[name], "'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py'",
        "'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py', 'dspark_ladder_scalar_reciprocal.py'")
    return result


def adapt_eager_only(sources):
    result = dict(sources)
    name = 'dspark-native-8k-attention-probe.py'
    result[name] = replace_once(result[name], "        update(0)\n        progress('capture')",
        "        report['eager_complete'] = True\n"
        "        report['complete_probe_coverage'] = False\n"
        "        return\n        update(0)\n        progress('capture')")
    result[name] = replace_once(result[name],
        "'diagnostics_complete' if report.get('diagnostics_complete')",
        "'eager_complete' if report.get('eager_complete') and report['closed_cleanly'] else "
        "'diagnostics_complete' if report.get('diagnostics_complete')")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=CONTEXTS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--probe-seconds', type=int, choices=(510, 1020), default=510)
    parser.add_argument('--scalar-reciprocal', action='store_true',
        help='Explicit changed-math diagnostic candidate, not the unchanged winning recipe')
    parser.add_argument('--eager-only', action='store_true',
        help='Stop after both eager fixtures; never qualifies replay or full probe coverage')
    parser.add_argument('--target-replay', action='store_true',
        help='Prepare context-selected target replay with runtime BF16 KV; does not grant admission')
    parser.add_argument('--combined-runtime', action='store_true',
        help='Prepare guarded 32K offline candidate entry; requires retained component evidence')
    parser.add_argument('--verifier-profile', action='store_true',
        help='Instrument exact 32K shared-Q/K verifier; never a throughput benchmark')
    parser.add_argument('--mlp-buffer', action='store_true',
        help='Matched four-block MLP comparison; retained simulator report required')
    parser.add_argument('--gdn-input-cache', action='store_true',
        help='Matched simulator-qualified GDN V/beta/gate input-cache comparison')
    parser.add_argument('--incremental-history', action='store_true',
        help='Matched qualified incremental history publication comparison')
    parser.add_argument('--norm-prefetch', action='store_true',
        help='Matched norm bridge prefetch with incremental publication in both arms')
    options = parser.parse_args()
    if options.norm_prefetch and (not options.combined_runtime or options.incremental_history
            or options.verifier_profile or options.mlp_buffer or options.gdn_input_cache):
        parser.error('Norm prefetch requires isolated combined runtime')
    if options.incremental_history and (not options.combined_runtime or options.verifier_profile
            or options.mlp_buffer or options.gdn_input_cache):
        parser.error('Incremental history requires isolated uninstrumented combined runtime')
    if options.gdn_input_cache and (not options.combined_runtime or options.verifier_profile or options.mlp_buffer):
        parser.error('GDN cache requires isolated uninstrumented combined runtime')
    if options.mlp_buffer and (not options.combined_runtime or options.verifier_profile):
        parser.error('Buffer comparison requires uninstrumented combined runtime')
    if options.verifier_profile and not options.combined_runtime:
        parser.error('Verifier profiling requires the admitted combined runtime')
    if options.combined_runtime and (options.context != 32768 or not options.scalar_reciprocal
            or not options.target_replay or options.eager_only):
        parser.error('Combined candidate requires 32768, scalar reciprocal and target replay, not eager-only')
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
    if options.target_replay:
        names += ('target-t16-attention-8k-probe.py', 'attention_mask_replay.py')
    if options.combined_runtime:
        from frozen_combined_adapters import FILES as COMBINED_FILES
        names += COMBINED_FILES
    if options.verifier_profile:
        from frozen_verifier_profile import FILES as PROFILE_FILES
        names += PROFILE_FILES
    sources = {}
    if options.mlp_buffer:
        names += ('fused_1d.py', 'dspark_request_experiment.py')
    if options.gdn_input_cache or options.incremental_history or options.norm_prefetch:
        names += ('dspark_request_experiment.py',)
    for name in names:
        original = git('show', f'{REVISION}:scripts/ci/{name}').decode()
        actual = (checkout / 'scripts/ci' / name).read_text()
        if original.replace('\r\n', '\n') != actual:
            raise ValueError('Historical source differs: ' + name)
        sources[name] = actual
    adapted = adapt_runtime_sources(adapt_cache_launcher(adapt_probe_sources(sources, options.context), options.probe_seconds))
    if options.target_replay:
        from frozen_target_replay import adapt_target_probe, adapt_target_mask
        name = 'target-t16-attention-8k-probe.py'
        adapted[name] = adapt_target_probe(adapted[name])
        adapted['attention_mask_replay.py'] = adapt_target_mask(adapted['attention_mask_replay.py'])
    if options.scalar_reciprocal:
        adapted = adapt_scalar_reciprocal(adapted)
        adapted['dspark_ladder_scalar_reciprocal.py'] = Path(__file__).with_name(
            'dspark_ladder_scalar_reciprocal.py').read_text()
    if options.eager_only:
        adapted = adapt_eager_only(adapted)
    if options.combined_runtime:
        from frozen_combined_adapters import adapt_combined_sources
        adapted = adapt_combined_sources(adapted)
        for name in ('frozen_combined_runtime.py', 'frozen_combined_gate.py', 'frozen_target_replay.py',
                'frozen_combined_history.py', 'frozen_reciprocal_isolation.py'):
            adapted[name] = Path(__file__).with_name(name).read_text()
    if options.verifier_profile:
        from frozen_verifier_profile import adapt_sources
        adapted = adapt_sources(adapted)
    if options.mlp_buffer:
        from frozen_mlp_buffer_trial import transform
        adapted['frozen_mlp_buffer_candidate.py'] = transform(sources['fused_1d.py'])
        adapted['dspark_request_experiment.py'] = replace_once(adapted['dspark_request_experiment.py'],
            'Native versus shared Q/K preparation and recurrence; complete combined runtime',
            'Two versus four MLP buffer blocks; shared Q/K enabled in both complete runtime arms')
        adapted['dspark_8k_scope.py'] = replace_once(adapted['dspark_8k_scope.py'],
            '        yield evidence',
            '        from frozen_mlp_buffer_scope import runtime_scope as buffer_scope\n'
            '        stack.enter_context(buffer_scope(directory))\n'
            '        yield evidence')
        for name in ('frozen_mlp_buffer_gate.py', 'frozen_mlp_buffer_scope.py'):
            adapted[name] = Path(__file__).with_name(name).read_text()
    if options.gdn_input_cache:
        adapted['dspark_request_experiment.py'] = replace_once(adapted['dspark_request_experiment.py'],
            'Native versus shared Q/K preparation and recurrence; complete combined runtime',
            'Uncached versus cached V/beta/gate; shared Q/K enabled in both complete runtime arms')
        adapted['dspark_8k_scope.py'] = replace_once(adapted['dspark_8k_scope.py'],
            '        yield evidence',
            '        from frozen_gdn_cache_scope import runtime_scope as cache_scope\n'
            '        stack.enter_context(cache_scope(directory))\n'
            '        yield evidence')
        for name in ('frozen_gdn_input_cache.py', 'frozen_gdn_cache_gate.py', 'frozen_gdn_cache_scope.py'):
            adapted[name] = Path(__file__).with_name(name).read_text()
    if options.incremental_history or options.norm_prefetch:
        scope_module = 'frozen_gdn_norm_scope' if options.norm_prefetch else 'frozen_incremental_scope'
        comparison = ('Original versus prefetched norm bridge; incremental publication and shared Q/K in both arms'
            if options.norm_prefetch else
            'Full-bank versus incremental publication; shared Q/K enabled in both complete runtime arms')
        adapted['dspark_request_experiment.py'] = replace_once(adapted['dspark_request_experiment.py'],
            'Native versus shared Q/K preparation and recurrence; complete combined runtime',
            comparison)
        adapted['dspark_8k_scope.py'] = replace_once(adapted['dspark_8k_scope.py'],
            '        yield evidence',
            f'        from {scope_module} import runtime_scope as incremental_scope\n'
            '        stack.enter_context(incremental_scope(directory))\n'
            '        yield evidence')
        for name in ('frozen_incremental_scope.py', 'incremental_history_scope.py',
                'history_append_hardware_gate.py', 'history_append_plan.py',
                'history_append_dma.py', 'history_append_dma.cpp', 'history-append-probe.py'):
            adapted[name] = Path(__file__).with_name(name).read_text()
        if options.norm_prefetch:
            for name in ('frozen_gdn_norm_scope.py', 'frozen_gdn_norm_gate.py', 'frozen_gdn_norm_prefetch.py'):
                adapted[name] = Path(__file__).with_name(name).read_text()
    for name in ('frozen_context_geometry.py', 'frozen_sim_build_cache.py', 'frozen_binary_cache.py',
            'frozen_sim_phase.py', 'frozen_sim_assets.py', 'frozen_probe_evidence.py'):
        adapted[name] = Path(__file__).with_name(name).read_text()
    for name, source in adapted.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in adapted.items():
        (checkout / 'scripts/ci' / name).write_bytes(source.encode('utf-8'))
    checksum = lambda source: hashlib.sha256(source.encode()).hexdigest()
    options.manifest.write_text(json.dumps(dict(revision=REVISION,
        geometry=geometry(options.context), before={name: checksum(source) for name, source in sources.items()},
        after={name: checksum(source) for name, source in adapted.items()},
        scope='Shared probe/runtime geometry adaptation; no numerical or runtime admission',
        reciprocal_variant='scalar-fp32' if options.scalar_reciprocal else 'native',
        eager_only=options.eager_only,
        target_replay=options.target_replay,
        combined_runtime=options.combined_runtime,
        verifier_profile=options.verifier_profile,
        mlp_buffer=options.mlp_buffer,
        gdn_input_cache=options.gdn_input_cache,
        incremental_history=options.incremental_history,
        norm_prefetch=options.norm_prefetch,
        probe_seconds=options.probe_seconds,
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
