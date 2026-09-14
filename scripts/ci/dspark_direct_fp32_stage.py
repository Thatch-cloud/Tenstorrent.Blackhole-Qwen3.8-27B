"""Unqualified direct FP32 staging; preserve normal source-CB matmul formats."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_score_sfpu


START = 'void qwen_stage_score_tile(uint32_t source_cb, uint32_t scratch_cb) {'
END = 'void qwen_prepare_center_scratch(uint32_t maxima_cb, uint32_t scratch_cb) {'
REPLACEMENT = r'''void qwen_unpack_fp32_at(uint32_t source_cb, uint32_t format_cb) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    const uint32_t source_operand = get_operand_id(source_cb);
    const uint32_t format_operand = get_operand_id(format_cb);
    LLK_ASSERT(cb_access_within_bounds(source_operand, 0, 1), "FP32 staging source is outside its CB");
    LLK_ASSERT(unpack_src_format[source_operand] == unpack_src_format[format_operand], "FP32 staging formats differ");
    LLK_ASSERT(get_operand_num_faces(source_operand) == get_operand_num_faces(format_operand), "FP32 staging faces differ");
    LLK_ASSERT(get_operand_face_r_dim(source_operand) == get_operand_face_r_dim(format_operand), "FP32 staging face dimensions differ");
    const uint32_t address = get_local_cb_interface(source_operand).fifo_rd_ptr - 1;
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
        address, unpack_src_format[format_operand], unpack_dst_format[format_operand]);
#endif
}

void qwen_stage_score_tile(uint32_t source_cb, uint32_t scratch_cb) {
    CircularBuffer(source_cb).wait_front(1);
    CircularBuffer(scratch_cb).reserve_back(1);
    qwen_copy_fp32_init(scratch_cb);
    pack_reconfig_data_format(scratch_cb);
    tile_regs_acquire();
    UNPACK((qwen_unpack_fp32_at(source_cb, scratch_cb)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, true>(0, scratch_cb)));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch_cb);
    tile_regs_release();
    CircularBuffer(scratch_cb).push_back(1);
    CircularBuffer(scratch_cb).wait_front(1);
}

'''


def transform(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('Unique original FP32 staging helper required')
    first, last = source.index(START), source.index(END)
    if first >= last or 'scratch[index] = source[index];' not in source[first:last]:
        raise ValueError('Original scalar staging copy required')
    return source[:first] + REPLACEMENT + source[last:]


DIAGNOSTIC = r'''
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    static uint32_t diagnostic_calls = 0;
    if (diagnostic_calls++ < 3) {
        const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(
            get_local_cb_interface(source_cb).fifo_rd_ptr << cb_addr_shift);
        const volatile uint32_t* destination = reinterpret_cast<const volatile uint32_t*>(
            get_local_cb_interface(scratch_cb).fifo_rd_ptr << cb_addr_shift);
        DEVICE_PRINT("QWEN_STAGE_BITS call={} source_cb={} scratch_cb={} srcfmt={} dstfmt={} source={},{},{},{},{} staged={},{},{},{},{}\n",
            diagnostic_calls, source_cb, scratch_cb, unpack_src_format[get_operand_id(scratch_cb)],
            unpack_dst_format[get_operand_id(scratch_cb)], source[0], source[1], source[16], source[256], source[512],
            destination[0], destination[1], destination[16], destination[256], destination[512]);
    }
#endif
'''


@contextmanager
def staging_scope(*, diagnostic=False):
    if type(diagnostic) is not bool:
        raise ValueError('Explicit staging diagnostic policy required')
    candidate = transform(dspark_score_sfpu.HELPER)
    if diagnostic:
        anchor = '    CircularBuffer(scratch_cb).wait_front(1);\n}\n\nvoid qwen_prepare_center_scratch'
        if candidate.count(anchor) != 1:
            raise ValueError('Unique completed staging boundary required')
        candidate = candidate.replace(anchor,
            '    CircularBuffer(scratch_cb).wait_front(1);\n' + DIAGNOSTIC + '}\n\nvoid qwen_prepare_center_scratch')
    with patch.object(dspark_score_sfpu, 'HELPER', candidate):
        yield
