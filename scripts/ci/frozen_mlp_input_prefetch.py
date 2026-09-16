"""Unqualified full-K activation staging for the fused T16 MLP."""

from frozen_recipe_context import replace_once
from frozen_recipe_context import REVISION


def reader(source):
    source = replace_once(source, '    for (uint32_t block = 0; block < 20; ++block) {',
        '    {')
    source = replace_once(source, '            for (uint32_t tile = 0; tile < 8; ++tile) {',
        '            for (uint32_t tile = 0; tile < 160; ++tile) {')
    source = replace_once(source, 'noc_async_read_tile(block * 8 + tile, input,',
        'noc_async_read_tile(tile, input,')
    for operation in ('cb_reserve_back', 'cb_push_back', 'cb_wait_front', 'cb_pop_front'):
        source = replace_once(source, f'{operation}(0, 8);', f'{operation}(0, 160);')
    source = replace_once(source,
        '            const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, destination);\n'
        '            noc_async_write_multicast(destination, target, 8 * 2048, receivers);',
        '            for (uint32_t block = 0; block < 20; ++block) {\n'
        '                const uint32_t chunk = destination + block * 8 * 2048;\n'
        '                const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, chunk);\n'
        '                noc_async_write_multicast(chunk, target, 8 * 2048, receivers);\n'
        '            }')
    return source


def projection(source):
    source = replace_once(source, 'cb(0, ttnn.bfloat16, 2048, 16, all_cores)',
        'cb(0, ttnn.bfloat16, 2048, 160, all_cores)')
    source = replace_once(source,
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        '                             full_k_input_prefetch=True, input_buffer_tiles=160,')
    compile(source, 'fused_1d.py', 'exec')
    return source


def memory_budget(pairs_per_worker=3):
    if pairs_per_worker != 3:
        raise ValueError('Only current three-pair T16 worker mapping is considered')
    return dict(input_bytes_per_core=160 * 2048,
        extra_input_bytes_per_core=(160 - 16) * 2048,
        worker_cb_bytes=160 * 2048 + 32 * 3 * 576 + 3 * 2048 + 6 * 4096 + 6 * 2048,
        scope='Declared CB bytes only; not device L1 admission or a performance prediction')


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    from frozen_mlp_buffer_trial import adapt_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh candidate manifest required')
    scripts = options.checkout / 'scripts/ci'
    sources = {}
    for name in ('fused_1d.py', 'fused_1d_input.cpp', 'fused-batch-probe.py'):
        source = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != source:
            raise ValueError('Historical source required: ' + name)
        sources[name] = source
    adapted = {'fused_1d.py': projection(sources['fused_1d.py']),
        'fused_1d_input.cpp': reader(sources['fused_1d_input.cpp']),
        'fused-batch-probe.py': adapt_probe(sources['fused-batch-probe.py'])}
    adapted['fused-batch-probe.py'] = replace_once(adapted['fused-batch-probe.py'],
        "    report['qualification_scope'] = 'T16 buffering only; other row widths and performance unqualified'",
        "    report['input_reader_candidate_sha256'] = hashlib.sha256(\n"
        "        Path(__file__).with_name('fused_1d_input.cpp').read_bytes()).hexdigest()\n"
        "    report['qualification_scope'] = 'T16 full-K input prefetch; other widths and performance unqualified'")
    adapted['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name, source in adapted.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in adapted.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(before={name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in sources.items()}, after={name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in adapted.items()}, memory=memory_budget(),
        simulator_qualified=False, hardware_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
