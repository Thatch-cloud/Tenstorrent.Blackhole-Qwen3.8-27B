"""Unqualified TR0 scalar reciprocal diagnostic; no shared CB format changes."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = 'void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {'
BODY = '''
    if constexpr (!QWEN_DRAFT_EXP_APPROX) {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
        CircularBuffer(in_cb).wait_front(num_tiles);
        for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
            const uint32_t address = (get_local_cb_interface(in_cb).fifo_rd_ptr +
                tile_index * get_local_cb_interface(in_cb).fifo_page_size) << cb_addr_shift;
            volatile float* values = reinterpret_cast<volatile float*>(address);
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t offset = row < 16 ? row * 16 : 512 + (row - 16) * 16;
                values[offset] = 1.0f / values[offset];
            }
        }
#endif
        return;
    }
'''


@contextmanager
def scalar_reciprocal():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((BEFORE, BEFORE + BODY),)
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
