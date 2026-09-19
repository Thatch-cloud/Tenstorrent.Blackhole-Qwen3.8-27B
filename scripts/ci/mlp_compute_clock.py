"""Bounded per-TRISC intervals; not active-cycle counters or performance qualification."""

from frozen_recipe_context import replace_once
from mlp_clock_samples import HELPER


WORDS = 64
MAGIC = 0x51434C4B
ZONES = (
    ('input_weight_wait', 'block == 10', 10, None),
    ('matmul_issue', 'block == 10 && in0_subblock == 0 && in1_subblock == 0', 10, 0),
    ('partial_handoff_pack', 'block == 10 && in0_subblock == 0 && in1_subblock == 0', 10, 0),
    ('final_reload', 'block == 19 && in0_subblock == 0 && in1_subblock == 0', 19, 0),
    ('gate_round_handoff_pack', 'block == 19 && in0_subblock == 0 && in1_subblock == 0', 19, 0),
    ('rounded_product_epilogue', 'true', None, None),
)
SETUP = '''
#if defined(TRISC_UNPACK)
    constexpr uint32_t qwen_processor = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t qwen_processor = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t qwen_processor = 2;
#else
#error "Compute clock sampling requires an explicit TRISC role"
#endif
    const bool qwen_enabled = get_arg_val<uint32_t>(1) == 1;
    volatile tt_l1_ptr uint32_t* qwen_samples = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_arg_val<uint32_t>(0) + qwen_processor * 256);
'''


def marks(index):
    predicate = ZONES[index][1]
    offset = index * 6
    begin = f'''\n    if (qwen_enabled && ({predicate})) {{
        const uint64_t stamp = qwen_sample_clock();
        qwen_samples[{offset}] = static_cast<uint32_t>(stamp);
        qwen_samples[{offset + 1}] = static_cast<uint32_t>(stamp >> 32);
    }}
'''
    end = f'''\n    if (qwen_enabled && ({predicate})) {{
        const uint64_t stamp = qwen_sample_clock();
        qwen_samples[{offset + 2}] = static_cast<uint32_t>(stamp);
        qwen_samples[{offset + 3}] = static_cast<uint32_t>(stamp >> 32);
        qwen_samples[{offset + 4}] = {index};
        qwen_samples[{offset + 5}] = {MAGIC ^ index} ^ qwen_processor;
    }}
'''
    return begin, end


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Exact fused compute timing boundaries required')
    first, last = source.index(start), source.index(end)
    if last <= first:
        raise ValueError('Ordered compute timing boundaries required')
    return source[first:last]


def segment(source, index):
    if index == 0:
        return '                    in0_dfb.wait_front(in0_block_num_tiles);\n                    in1_dfb.wait_front(in1_block_num_tiles);'
    if index == 1:
        return section(source, '                            tile_regs_acquire();\n                            if (enable_reload)',
            '                            if (last_out) {')
    if index == 2:
        start = '                            } else {\n                                tile_regs_commit();'
        end = '\n                            }\n\n                            in1_index_subblock_offset'
        body = section(source, start, end)
        return body[len('                            } else {\n'):]
    if index == 3:
        return section(source, '                                reload_from_cb_to_dst(',
            '\n                            }\n\n#ifndef SKIP_COMPUTE')
    if index == 4:
        return section(source, '                                static_assert(out_subblock_num_tiles == 2);',
            '                            } else {\n                                tile_regs_commit();')
    if index == 5:
        start = '    constexpr uint32_t rounded_cb = 30;\n    cb_wait_front(rounded_cb, 6);'
        if source.count(start) != 1 or source[source.rfind('}') + 1:].strip():
            raise ValueError('Unchanged three-pair rounded epilogue required')
        return source[source.index(start):source.rfind('}')]
    raise ValueError('Unknown compute timing interval')


def instrument(source):
    if 'qwen_sample_clock' in source or 'DeviceZoneScopedN' in source:
        raise ValueError('Uninstrumented compute source required')
    result = replace_once(source, '#include <cstdint>\n', '#include <cstdint>\n' + HELPER)
    result = replace_once(result, 'void kernel_main() {', 'void kernel_main() {' + SETUP)
    for index in (0, 1, 3, 4, 2, 5):
        body = segment(result, index)
        begin, end = marks(index)
        result = replace_once(result, body, begin + body + end)
    if remove(result) != source:
        raise ValueError('Compute timing changed numerical source or order')
    return result


def remove(source):
    for index in reversed(range(len(ZONES))):
        begin, end = marks(index)
        source = replace_once(replace_once(source, begin, ''), end, '')
    return replace_once(replace_once(source, SETUP, ''), HELPER, '')


def decode(words, processor, *, max_cycles=100_000_000):
    if (type(processor) is not int or processor not in (0, 1, 2) or len(words) != WORDS
            or type(max_cycles) is not int or max_cycles <= 0
            or any(type(value) is not int or not 0 <= value <= 0xffffffff for value in words)):
        raise ValueError('Complete uint32 page and explicit TRISC required')
    result = []
    for index, (name, predicate, block, subblock) in enumerate(ZONES):
        low, high, end_low, end_high, identifier, magic = words[index * 6:index * 6 + 6]
        start, end = high << 32 | low, end_high << 32 | end_low
        if identifier != index or magic != MAGIC ^ index ^ processor or not 0 <= end - start <= max_cycles:
            raise ValueError('Missing, malformed or out-of-bound compute sample')
        if result and start < result[-1]['end_cycle']:
            raise ValueError('Compute samples must preserve processor execution order')
        result.append(dict(zone=name, processor=processor, block=block, subblock=subblock,
            start_cycle=start, end_cycle=end, duration_cycles=end - start))
    return result
