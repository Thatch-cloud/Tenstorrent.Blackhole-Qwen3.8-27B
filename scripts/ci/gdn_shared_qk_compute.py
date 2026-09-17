"""Generate shared-Q/K compute from the exact pinned native normalization chain."""

import hashlib
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT, replace_once


MAIN = 'void kernel_main() {'
NORM_START = '        // ---- L2 norm q (scale folded):'
NORM_END = '        // ---- decay: h = state * exp(g) ----'


def section(source):
    if source.count(NORM_START) != 1 or source.count(NORM_END) != 1:
        raise ValueError('Expected unique native normalization boundaries')
    start, end = source.index(NORM_START), source.index(NORM_END)
    if end <= start:
        raise ValueError('Native normalization boundary order changed')
    return source[start:end]


def compute(source, *, serial):
    if type(serial) is not bool:
        raise ValueError('Explicit serial versus block normalization required')
    if source.count(MAIN) != 1:
        raise ValueError('Expected one pinned compute entry point')
    chain = section(source)
    chain = replace_once(chain, '        WAIT(cb_qn, Kt);\n', '')
    chain = replace_once(chain, '        WAIT(cb_kn, Kt);\n', '')
    iterations = 'get_arg_val<uint32_t>(0)' if serial else '1'
    return source[:source.index(MAIN)] + '''void kernel_main() {
    constexpr uint32_t Kt = 4;
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(0);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(cb_q, cb_k, cb_qn);
    WAIT(cb_ones, 1);
    const uint32_t iterations = ''' + iterations + ''';
    for (uint32_t token = 0; token < iterations; ++token) {
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        copy_tiles(cb_q, cb_qf, Kt);
        copy_tiles(cb_k, cb_kf, Kt);
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        WAIT(cb_qf, Kt);
        WAIT(cb_kf, Kt);
''' + chain + '''    }
}
'''


def load_compute(root, *, serial=False):
    relative = 'compute/decode_gated_delta_rule.cpp'
    data = (Path(root) / KERNEL_ROOT / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[relative]:
        raise ValueError('Native normalization source hash changed')
    return compute(data.decode(), serial=serial)


def buffer_plan():
    return {0: 4, 1: 4}, {6: 1, 7: 4, 8: 4, 9: 1, 10: 4, 11: 4,
                          20: 4, 21: 4, 28: 1, 29: 1}


def worker_head(worker):
    if type(worker) is not int or not 0 <= worker < 96:
        raise ValueError('Expected one of 96 recurrence workers')
    return worker // 12


def input_page(head, tile, *, key=False):
    if (type(head) is not int or not 0 <= head < 8 or type(tile) is not int
            or not 0 <= tile < 4 or type(key) is not bool):
        raise ValueError('Expected an eight-head, four-tile Q/K coordinate')
    return (32 if key else 0) + head * 4 + tile
