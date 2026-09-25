"""Unqualified direct causal-window reader; native convolution math and writer stay unchanged."""


START = '        // window = [st1, st2, st3, x]  (page `inst` in every [1,Bmax,C] tensor)\n'
END = '        // taps: page cc of each [1,1,C] tap tensor\n'


def causal_source(token, slot):
    if type(token) is not int or type(slot) is not int or not 0 <= token < 16 or not 0 <= slot < 4:
        raise ValueError('T16 causal convolution coordinates required')
    position = token + slot - 3
    return ('history', position + 4) if position < 0 else ('projected', position)


def reader(source):
    if source.count(START) != 1 or source.count(END) != 1 or source.index(START) >= source.index(END):
        raise ValueError('Exact native convolution reader window boundary required')
    body = '''        {
            static_assert(K == 4 && B == 16 && xBt == 1 && Ct == 160 && Wt == 258 && NV == 24);
            const uint32_t scratch = get_write_ptr(14);
            noc_async_read_tile(cc, x_acc, scratch);
            noc_async_read_tile(cc, s1_acc, scratch + 2048);
            noc_async_read_tile(cc, s2_acc, scratch + 4096);
            noc_async_read_tile(cc, s3_acc, scratch + 6144);
            noc_async_read_barrier();
            CircularBuffer cb(cb_win);
            cb.reserve_back(K);
            const uint32_t base = cb.get_write_ptr();
            zero_words(base, 4 * 512);
            for (uint32_t slot = 0; slot < 4; ++slot) {
                for (uint32_t token = 0; token < 16; ++token) {
                    const uint32_t position = token + slot;
                    const uint32_t source_tile = position < 3 ? position + 1 : 0;
                    const uint32_t source_row = position < 3 ? 0 : position - 3;
                    for (uint32_t face = 0; face < 2; ++face) {
                        const auto input = reinterpret_cast<volatile const uint32_t*>(
                            scratch + source_tile * 2048 + source_row * 32 + face * 512);
                        auto output = reinterpret_cast<volatile uint32_t*>(
                            base + slot * 2048 + token * 32 + face * 512);
                        for (uint32_t word = 0; word < 8; ++word) { output[word] = input[word]; }
                    }
                }
            }
            asm volatile("" ::: "memory");
            cb.push_back(K);
        }
'''
    return source[:source.index(START)] + body + source[source.index(END):]
