"""Simulator-only register-resident rounded SwiGLU candidate."""

from gdn_multitoken import replace_once


START = '                            if (last_out) {\n'
END = '                            } else {\n                                tile_regs_commit();'
TAIL = '    constexpr uint32_t rounded_cb = 30;\n    cb_wait_front(rounded_cb, 6);'
CAST = 'static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)'
BF16_PRODUCT = """MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, false), 0, 1, 0, VectorMode::RC)));"""
FINAL = '''                            if (last_out) {
                                static_assert(out_subblock_num_tiles == 2);
                                silu_tile_init();
                                silu_tile(0);
                                typecast_tile_init<''' + CAST + '''>();
                                typecast_tile<''' + CAST + '''>(0);
                                typecast_tile<''' + CAST + '''>(1);
                                mul_binary_tile_init();
                                ''' + BF16_PRODUCT + '''
                                tile_regs_commit();
                                cb_reserve_back(out_dfb_id, 1);
                                tile_regs_wait();
                                PACK((pack_reconfig_data_format(out_dfb_id)));
                                PACK((llk_pack_reconfig_l1_acc(0)));
                                pack_tile(0, out_dfb_id);
                                tile_regs_release();
                                cb_push_back(out_dfb_id, 1);
'''


def transform(source):
    for anchor in (START, END, TAIL):
        if source.count(anchor) != 1:
            raise ValueError('Exact three-pair fused compute required')
    start, end, tail = source.index(START), source.index(END), source.index(TAIL)
    finish = source.rfind('}')
    if not start < end < tail < finish or source[finish + 1:].strip():
        raise ValueError('Ordered final block and epilogue required')
    final = source[start:end]
    expected = '''                            if (last_out) {
                                static_assert(out_subblock_num_tiles == 2);
                                constexpr uint32_t rounded_cb = 30;
                                tile_regs_commit();
                                cb_reserve_back(rounded_cb, 2);
                                apply_activation_from_pack<KernelActivation::SILU>(1);
                                PACK((pack_reconfig_data_format(rounded_cb)));
                                PACK((llk_pack_reconfig_l1_acc(0)));
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, rounded_cb, 2);
                                tile_regs_release();
                                cb_push_back(rounded_cb, 2);
'''
    if final != expected or source[tail:finish].count(BF16_PRODUCT) != 1:
        raise ValueError('Rounded native product or gate/up handoff changed')
    result = source[:start] + FINAL + source[end:tail] + source[finish:]
    return '#include "api/compute/eltwise_unary/typecast.h"\n' + result


def activation_control(source):
    transform(source)
    result = replace_once(source,
        '                                constexpr uint32_t rounded_cb = 30;\n                                tile_regs_commit();',
        '                                constexpr uint32_t rounded_cb = 30;\n'
        '                                silu_tile_init();\n'
        '                                silu_tile(0);\n'
        '                                tile_regs_commit();')
    return replace_once(result, '                                apply_activation_from_pack<KernelActivation::SILU>(1);',
        '                                tile_regs_wait();')


def adapt_projection(source, *, diagnose_activation=False):
    if type(diagnose_activation) is not bool:
        raise ValueError('Explicit activation diagnostic policy required')
    if 'mlp_register_epilogue' in source:
        raise ValueError('Projection already contains the register epilogue')
    function = 'activation_control' if diagnose_activation else 'transform'
    result = replace_once(source, 'import hashlib\n',
        f'import hashlib\nfrom mlp_register_epilogue import {function} as register_epilogue\n')
    anchor = '        self.compute = fused_compute(original, intermediates=intermediates, pairs_per_worker=pairs_per_worker)'
    result = replace_once(result, anchor, anchor + '''
        if intermediates or pairs_per_worker != 3 or token_rows != 16 or math_approx_mode is not True:
            raise ValueError('Register epilogue is restricted to the T16 target-math simulator candidate')
        self.compute = register_epilogue(self.compute)''')
    result = replace_once(result, 'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,',
        'intermediates=intermediates, input_noc=1, weight_noc=0, token_rows=token_rows,\n'
        f'                             register_epilogue={not diagnose_activation}, activation_diagnostic={diagnose_activation},\n'
        f'                             intermediate_rounding="{"native-pack" if diagnose_activation else "sfpu-fp32-to-bf16"}",')
    compile(result, 'fused_1d.py', 'exec')
    return result
