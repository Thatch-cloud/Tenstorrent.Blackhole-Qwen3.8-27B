"""Opt-in T16 phase clocks; intervals include waits and are not active-cycle counts."""

from gdn_multitoken import replace_once
from mlp_clock_samples import HELPER


WORDS = 64
MAGIC = 0x5147444E
ZONES = (
    ('input_wait', '        WAIT(cb_qn, Kt);'),
    ('input_conversion', '\n        copy_tiles(cb_v, cb_vf, Vt);'),
    ('state_decay', '        expc(cb_gf, cb_gexp, 1);'),
    ('value_read_delta', '        mm(cb_kn, cb_sdec, cb_vread, 1, Kt, Vt, false);'),
    ('rank_update', '        transpose_row(cb_kn, cb_kcol, Kt);'),
    ('output_projection', '        if constexpr (FNG) {'),
    ('state_publication', '        copy_tiles(cb_snew, cb_sout, kv);'),
)
TAIL = '        POP(cb_snew, kv);'
SETUP = '''
#if defined(TRISC_UNPACK)
    constexpr uint32_t qwen_processor = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t qwen_processor = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t qwen_processor = 2;
#else
#error "GDN clocks require an explicit TRISC role"
#endif
    const bool qwen_enabled = get_arg_val<uint32_t>(2) == 1;
    const uint32_t qwen_token = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* qwen_samples = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_arg_val<uint32_t>(1) + qwen_processor * 256);
'''


def marks(index):
    offset = index * 6
    begin = f'''
        if (qwen_enabled && it == qwen_token) {{
            const uint64_t stamp = qwen_sample_clock();
            qwen_samples[{offset}] = static_cast<uint32_t>(stamp);
            qwen_samples[{offset + 1}] = static_cast<uint32_t>(stamp >> 32);
        }}
'''
    end = f'''
        if (qwen_enabled && it == qwen_token) {{
            const uint64_t stamp = qwen_sample_clock();
            qwen_samples[{offset + 2}] = static_cast<uint32_t>(stamp);
            qwen_samples[{offset + 3}] = static_cast<uint32_t>(stamp >> 32);
            qwen_samples[{offset + 4}] = {index};
            qwen_samples[{offset + 5}] = {MAGIC ^ index} ^ (qwen_processor << 16) ^ (qwen_token << 8);
        }}
'''
    return begin, end


def instrument(source):
    if 'qwen_sample_clock' in source or 'DeviceZoneScopedN' in source:
        raise ValueError('Uninstrumented shared-Q/K recurrence required')
    anchors = [anchor for _, anchor in ZONES] + [TAIL]
    if any(source.count(anchor) != 1 for anchor in anchors):
        raise ValueError('Unique recurrence phase boundaries required')
    offsets = [source.index(anchor) for anchor in anchors]
    if offsets != sorted(offsets):
        raise ValueError('Native recurrence phase order changed')
    if source.count('if (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }') != 1:
        raise ValueError('Native rounded state feedback required')
    result = replace_once(source, '#include <cstdint>\n', '#include <cstdint>\n' + HELPER)
    result = replace_once(result, 'void kernel_main() {', 'void kernel_main() {' + SETUP)
    for index, (_, anchor) in enumerate(ZONES):
        boundary = (marks(index - 1)[1] if index else '') + marks(index)[0]
        result = replace_once(result, anchor, boundary + anchor)
    result = replace_once(result, TAIL, TAIL + marks(len(ZONES) - 1)[1])
    if remove(result) != source:
        raise ValueError('Clock instrumentation changed numerical source')
    return result


def remove(source):
    for index in reversed(range(len(ZONES))):
        begin, end = marks(index)
        source = replace_once(replace_once(source, begin, ''), end, '')
    return replace_once(replace_once(source, SETUP, ''), HELPER, '')


def decode(words, processor, token, *, max_cycles=100_000_000):
    if (type(processor) is not int or processor not in (0, 1, 2)
            or type(token) is not int or not 0 <= token < 16
            or len(words) != WORDS or type(max_cycles) is not int or max_cycles <= 0
            or any(type(value) is not int or not 0 <= value <= 0xffffffff for value in words)):
        raise ValueError('Complete uint32 page, TRISC and T16 token required')
    result = []
    for index, (name, _) in enumerate(ZONES):
        low, high, end_low, end_high, identifier, magic = words[index * 6:index * 6 + 6]
        start, end = high << 32 | low, end_high << 32 | end_low
        if (identifier != index or magic != MAGIC ^ index ^ (processor << 16) ^ (token << 8)
                or not 0 <= end - start <= max_cycles
                or result and start < result[-1]['end_cycle']):
            raise ValueError('Missing, out-of-order or invalid recurrence sample')
        result.append(dict(zone=name, processor=processor, token=token,
            start_cycle=start, end_cycle=end, duration_cycles=end - start))
    return result
