"""Two in-flight weight blocks in the existing two-block circular buffer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once


START = '    for (uint32_t block = 0; block < 20; ++block) {\n'
END = '    for (uint32_t pair = 0; pair < pairs_per_worker; ++pair) {\n'
RESERVE = '''        cb_reserve_back(1, 16 * pairs_per_worker);
        const uint32_t destination = get_write_ptr(1);
'''
FINISH = '''        noc_async_read_barrier();
        cb_push_back(1, 16 * pairs_per_worker);
    }
'''
SETUP = '''    static_assert(pairs_per_worker == 3);
    constexpr uint32_t block_tiles = 16 * pairs_per_worker;
    constexpr uint32_t block_bytes = block_tiles * 576;
    cb_reserve_back(1, 2 * block_tiles);
    const uint32_t first_slot = get_write_ptr(1);
    auto issue_block = [&](uint32_t block) {
        const uint32_t destination = first_slot + (block % 2) * block_bytes;
        const uint32_t transaction = 1 + block % 2;
'''
READ = '''                    const uint64_t source = weights.get_noc_addr(page);
                    noc_async_read_one_packet_set_state(source, 576);
                    noc_async_read_set_trid(transaction);
                    noc_async_read_one_packet_with_state_with_trid(
                        0, static_cast<uint32_t>(source), tile_address, transaction);
'''
PIPELINE = '''    };
    issue_block(0);
    for (uint32_t block = 0; block < 20; ++block) {
        if (block + 1 < 20) {
            cb_reserve_back(1, 2 * block_tiles);
            issue_block(block + 1);
        }
        noc_async_read_barrier_with_trid(1 + block % 2);
        cb_push_back(1, block_tiles);
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
'''


def transform(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Exact frozen weight reader required')
    start, end = source.index(START), source.index(END)
    body = source[start + len(START):end]
    body = replace_once(body, RESERVE, '')
    body = replace_once(body, FINISH, '')
    body = replace_once(body, '                    noc_async_read_tile(page, weights, tile_address);\n', READ)
    return source[:start] + SETUP + body + PIPELINE + source[end:]


def schedule():
    slots, events = [None, None], []
    consumed = -1
    for block in range(20):
        if block == 0:
            slots[0] = 0
            events.append(('issue', 0, 0, 1))
        if block + 1 < 20:
            if block:
                consumed = block - 1
                slots[consumed % 2] = None
                events.append(('consume', consumed, consumed % 2, 1 + consumed % 2))
            next_block = block + 1
            slot = next_block % 2
            if slots[slot] is not None:
                raise ValueError('Overwrite of unconsumed circular-buffer block')
            slots[slot] = next_block
            events.append(('issue', next_block, slot, 1 + slot))
        events.append(('barrier', block, block % 2, 1 + block % 2))
        events.append(('publish', block, block % 2, 1 + block % 2))
    for block in range(consumed + 1, 20):
        events.append(('consume', block, block % 2, 1 + block % 2))
    return events


def main():
    from frozen_mlp_buffer_trial import adapt_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh pipeline staging required')
    scripts = options.checkout / 'scripts/ci'
    originals = {}
    for name in ('fused_1d_weights.cpp', 'fused-batch-probe.py', 'fused_1d.py'):
        original = subprocess.check_output(['git', '-C', str(options.checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != original:
            raise ValueError('Exact frozen source required: ' + name)
        originals[name] = original
    payloads = dict(originals)
    payloads['fused_1d_weights.cpp'] = transform(originals['fused_1d_weights.cpp'])
    payloads['fused-batch-probe.py'] = replace_once(adapt_probe(originals['fused-batch-probe.py']),
        'T16 buffering only; other row widths and performance unqualified',
        'T16 two-transaction weight pipeline; no performance qualification')
    payloads['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        compute_changed=False, buffers_changed=False, weight_layout_changed=False,
        inflight_blocks=2, buffer_blocks=2, transaction_ids=[1, 2],
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
