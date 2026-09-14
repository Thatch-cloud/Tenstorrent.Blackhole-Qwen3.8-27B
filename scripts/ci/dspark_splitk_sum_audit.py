"""Read-only device probability-sum audit for the small simulator fixture."""


def face_offset(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


def transform(source):
    anchor = '                /* OUT_IM = QK @ V_CHUNK */'
    if source.count(anchor) != 1:
        raise ValueError('Exact sum audit boundary required')
    audit = '''
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
                if (k_chunk == k_chunk_start) {
                    CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);
                    CircularBuffer(cb_cur_sum).wait_front(Sq_chunk_t);
                    constexpr uint32_t audit_rows[] = {0, 1, 2, 3, 12, 55};
                    for (uint32_t query_row : audit_rows) {
                        if (query_row >= Sq_chunk_t * 32) continue;
                        const uint32_t tile_row = query_row / 32;
                        const uint32_t local_row = query_row % 32;
                        float probability_sum = 0.0f;
                        for (uint32_t key_column = 0; key_column < Sk_chunk_t_dynamic * 32; ++key_column) {
                            const uint32_t tile = tile_row * Sk_chunk_t_dynamic + key_column / 32;
                            const uint32_t column = key_column % 32;
                            const uint32_t address = (get_local_cb_interface(cb_qk_im).fifo_rd_ptr +
                                tile * get_local_cb_interface(cb_qk_im).fifo_page_size) << cb_addr_shift;
                            const volatile float* values = reinterpret_cast<const volatile float*>(address);
                            const uint32_t offset = (local_row / 16) * 512 + (column / 16) * 256 +
                                (local_row % 16) * 16 + column % 16;
                            probability_sum += values[offset];
                        }
                        const uint32_t sum_address = (get_local_cb_interface(cb_cur_sum).fifo_rd_ptr +
                            tile_row * get_local_cb_interface(cb_cur_sum).fifo_page_size) << cb_addr_shift;
                        const volatile uint16_t* sums = reinterpret_cast<const volatile uint16_t*>(sum_address);
                        union { uint32_t bits; float value; } reduced;
                        reduced.bits = static_cast<uint32_t>(sums[(local_row / 16) * 512 + (local_row % 16) * 16]) << 16;
                        DEVICE_PRINT("QWEN_SPLITK_SUM_AUDIT row={} probability_sum={:.9f} reduced_sum={:.9f}\\n",
                            query_row, probability_sum, reduced.value);
                    }
                }
#endif
'''
    return source.replace(anchor, audit + anchor)
