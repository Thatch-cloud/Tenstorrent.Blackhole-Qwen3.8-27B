"""Unqualified recurrence consumer of eight shared FP32 normalized Q/K heads."""

from gdn_multitoken import replace_once, replace_section
from gdn_shared_qk_compute import section
from gdn_vsplit_norm_batch import load_kernels as baseline_kernels


CACHE = '''    static_assert(Kt == 4 && Vt == 1 && PACKED && !FNG);
    CircularBuffer query_cache(20), key_cache(21);
    query_cache.reserve_back(4);
    key_cache.reserve_back(4);
    for (uint32_t tile = 0; tile < 4; ++tile) {
        const uint32_t page = (bh_start / 12) * 4 + tile;
        noc.async_read(q_acc, query_cache, 4096, {.page_id = page}, {.offset_bytes = tile * 4096});
        noc.async_read(k_acc, key_cache, 4096, {.page_id = page}, {.offset_bytes = tile * 4096});
    }
    noc.async_read_barrier();
    query_cache.push_back(4);
    key_cache.push_back(4);
    query_cache.wait_front(4);
    key_cache.wait_front(4);
    auto gather_normalized = [&](uint32_t cache_id, uint32_t destination_id, uint32_t token) {
        CircularBuffer destination(destination_id);
        destination.reserve_back(4);
        const uint32_t target_base = destination.get_write_ptr();
        const uint32_t source_base = CircularBuffer(cache_id).get_read_ptr();
        if (token == 0) { zero(target_base, 4 * 1024); }
        const uint32_t offset = (512 * (token / 16) + 16 * (token % 16)) * 4;
        for (uint32_t tile = 0; tile < 4; ++tile) {
            asm volatile("" ::: "memory");
            auto source = CoreLocalMem<volatile uint32_t>(source_base + tile * 4096 + offset);
            auto target = CoreLocalMem<volatile uint32_t>(target_base + tile * 4096);
            for (uint32_t word = 0; word < 16; ++word) {
                target[word] = source[word];
                target[256 + word] = source[256 + word];
            }
            asm volatile("" ::: "memory");
        }
        destination.push_back(4);
    };

'''

INPUTS = '''        gather_normalized(20, 10, token);
        gather_normalized(21, 11, token);
        const uint32_t row_page0 = (b / 32) * Ct;
        const uint32_t r = b % 32;
        gather_row(v_acc, cb_v, Vt, row_page0 + VOT + h * Vt, r);
        gather_scalar(beta_acc, cb_beta, (b / 32) * NVT + (h / 4) / 32, r, (h / 4) % 32);
        gather_scalar(g_acc, cb_g, (b / 32) * NVT + (h / 4) / 32, r, (h / 4) % 32);

'''


def compute(source):
    source = replace_once(source, section(source), '')
    for original, replacement in (('        WAIT(cb_q, Kt);', '        WAIT(cb_qn, Kt);'),
            ('        WAIT(cb_k, Kt);', '        WAIT(cb_kn, Kt);')):
        source = replace_once(source, original, replacement)
    for line in ('copy_tiles(cb_q, cb_qf, Kt);', 'copy_tiles(cb_k, cb_kf, Kt);',
                 'POP(cb_q, Kt);', 'POP(cb_k, Kt);', 'WAIT(cb_qf, Kt);', 'WAIT(cb_kf, Kt);'):
        source = replace_once(source, '        ' + line + '\n', '')
    return source


def reader(source):
    for name in ('q', 'k'):
        source = replace_once(source,
            f'const auto {name}_acc = TensorAccessor({name}_a, {name}_addr, tb_io);',
            f'const auto {name}_acc = TensorAccessor({name}_a, {name}_addr, 4096);')
    loop = '    for (uint32_t token = 0; token < n_inst; ++token) {'
    source = replace_once(source, loop, CACHE + loop)
    source = replace_section(source, '        if constexpr (PACKED) {', '        // State [B,H,K,V]', INPUTS)
    if not source.endswith('    }\n}\n'):
        raise ValueError('Pinned recurrence reader tail changed')
    return source[:-2] + '    query_cache.pop_front(4);\n    key_cache.pop_front(4);\n}\n'


def load_kernels(root):
    kernels = baseline_kernels(root)
    recurrence = kernels['recurrence']
    recurrence['compute'] = compute(recurrence['compute'])
    recurrence['reader'] = reader(recurrence['reader'])
    return kernels


def recurrence_spec(native):
    if (native.get('workers') != 96 or native.get('reader_addresses') != [0, 0, 0, 1, 2, 3]
            or native.get('reader_accessors') != [0, 0, 0, 1, 2, 3, 3, 3]):
        raise ValueError('Pinned recurrence operand layout required')
    return dict(native, reader_addresses=[9, 10, 0, 1, 2, 3],
                reader_accessors=[9, 10, 0, 1, 2, 3, 3, 3])
