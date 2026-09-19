"""Unqualified bounded MLP wait samples without global profiler buffers."""

from frozen_mlp_wait_zones import ZONES
from frozen_recipe_context import replace_once


MAGIC = 0x514D4C50
WORDS = 32
SCRATCH = {'input': 31, 'weights': 6}
HELPER = '''#include "risc_common.h"

static inline uint64_t qwen_sample_clock() {
    volatile tt_reg_ptr uint32_t* clock_registers =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t high_before, low, high_after;
    do {
        high_before = clock_registers[1];
        low = clock_registers[0];
        high_after = clock_registers[1];
    } while (high_before != high_after);
    return (static_cast<uint64_t>(high_after) << 32) | low;
}

'''


def wrapper(statement, predicate, index):
    return f'''{{
            if ({predicate}) {{
                const uint64_t before = qwen_sample_clock();
                {statement}
                const uint64_t after = qwen_sample_clock();
                samples[{index * 6}] = static_cast<uint32_t>(before);
                samples[{index * 6 + 1}] = static_cast<uint32_t>(before >> 32);
                samples[{index * 6 + 2}] = static_cast<uint32_t>(after);
                samples[{index * 6 + 3}] = static_cast<uint32_t>(after >> 32);
                samples[{index * 6 + 4}] = {index};
                samples[{index * 6 + 5}] = {MAGIC ^ index};
            }} else {{
                {statement}
            }}
        }}'''


def additions(role):
    if role not in ZONES:
        raise ValueError('Explicit input or weights reader role required')
    accessor = 'tensor_args' if role == 'input' else 'output_args'
    argument = 8 if role == 'input' else 5
    selector = 'worker <= 1' if role == 'input' else 'first_pair == 0'
    page = 'worker' if role == 'input' else '0'
    scratch = SCRATCH[role]
    setup = f'''    constexpr auto sample_args = TensorAccessorArgs<{accessor}.next_compile_time_args_offset()>();
    const auto sample_output = TensorAccessor(sample_args, get_arg_val<uint32_t>({argument}), 128);
    volatile tt_l1_ptr uint32_t* samples = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr({scratch}));
    if ({selector}) {{
        for (uint32_t word = 0; word < 32; ++word) samples[word] = 0xffffffff;
    }}
'''
    finish = f'''    if ({selector}) {{
        noc_async_write_tile({page}, sample_output, get_write_ptr({scratch}));
        noc_async_write_barrier();
    }}
'''
    return setup, finish


def instrument(source, role):
    setup, finish = additions(role)
    if 'qwen_sample_clock' in source or 'DeviceZoneScopedN' in source:
        raise ValueError('Only uninstrumented native reader operations admitted')
    result = replace_once(source, '#include "api/dataflow/dataflow_api.h"\n',
        '#include "api/dataflow/dataflow_api.h"\n' + HELPER)
    for index, (statement, name, predicate) in enumerate(ZONES[role]):
        result = replace_once(result, statement, wrapper(statement, predicate, index))
    anchor = '    for (uint32_t block = 0; block < 20; ++block) {\n'
    result = replace_once(result, anchor, setup + anchor)
    end = result.rfind('}')
    if end < 0 or result[end + 1:].strip():
        raise ValueError('Explicit kernel end required')
    result = result[:end] + finish + result[end:]
    if remove(result, role) != source:
        raise ValueError('Sampling changed original operation ordering')
    return result


def remove(source, role):
    setup, finish = additions(role)
    result = replace_once(replace_once(replace_once(source, HELPER, ''), setup, ''), finish, '')
    for index, (statement, name, predicate) in enumerate(ZONES[role]):
        result = replace_once(result, wrapper(statement, predicate, index), statement)
    return result


def decode(words, role, worker, *, max_cycles=100_000_000):
    expected = {('input', 0): (0, 1, 2, 3), ('input', 1): (0, 4), ('weights', 0): (0, 1, 2, 3)}
    if ((role, worker) not in expected or len(words) != WORDS
            or type(max_cycles) is not int or max_cycles <= 0
            or any(type(value) is not int or not 0 <= value <= 0xffffffff for value in words)):
        raise ValueError('One complete bounded uint32 sample page required')
    records = []
    for index in expected[role, worker]:
        low, high, end_low, end_high, identifier, magic = words[index * 6:index * 6 + 6]
        start, end = (high << 32) | low, (end_high << 32) | end_low
        if identifier != index or magic != MAGIC ^ index or not 0 <= end - start <= max_cycles:
            raise ValueError('Missing, malformed or out-of-bound clock sample')
        records.append(dict(zone=ZONES[role][index][1], worker=worker, start_cycle=start,
            end_cycle=end, duration_cycles=end - start))
    if any(current['start_cycle'] < previous['end_cycle']
            for previous, current in zip(records, records[1:])):
        raise ValueError('Sampled operations must preserve reader execution order')
    return records
