import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import native_draft_sdpa
from dspark_ladder_sum_update import BODY, scalar_sum_update
from dspark_ladder_factory import scoped_stats_pack


STUB = '''
#include <cstdint>
constexpr uint32_t cb_addr_shift=4;
struct Interface { uint32_t fifo_rd_ptr, fifo_page_size; };
Interface buffers[3];
uint32_t popped[3]={0,0,0};
Interface& get_local_cb_interface(uint32_t index) { return buffers[index]; }
struct CircularBuffer {
    uint32_t index;
    explicit CircularBuffer(uint32_t value):index(value) {}
    void wait_front(uint32_t) {}
    void pop_front(uint32_t count) { popped[index]+=count; }
};
void update() {
    uint32_t alias_prev_sum=0, alias_cur_sum=1, cb_exp_max_diff=2, Sq_chunk_t=2;
''' + BODY + '\n}\n'


class SumUpdateTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux compiler required')
    def test_two_tiles_all_faces_and_buffer_ownership(self):
        source = STUB + '''
#include <sys/mman.h>
int main() {
    void* memory=mmap(nullptr,24576,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_32BIT,-1,0);
    if(memory==MAP_FAILED) return 2;
    float* values=static_cast<float*>(memory);
    for(uint32_t buffer=0;buffer<3;++buffer) {
        buffers[buffer]={static_cast<uint32_t>(reinterpret_cast<uintptr_t>(values+buffer*2048)>>4),256};
        for(uint32_t index=0;index<2048;++index) values[buffer*2048+index]=buffer==2 ? -999.0f : 1.0f+index/2048.0f;
    }
    for(uint32_t tile=0;tile<2;++tile)
        for(uint32_t row=0;row<32;++row)
            values[4096+tile*1024+(row<16?row*16:512+(row-16)*16)]=0.5f;
    update();
    if(popped[0]!=2 || popped[1]!=0 || popped[2]!=0) return 3;
    for(uint32_t index=0;index<2048;++index) {
        float original=1.0f+index/2048.0f;
        if(values[index]!=original || values[2048+index]!=original+original*0.5f) return 4;
        uint32_t local=index%1024;
        bool first=(local<256 || (local>=512 && local<768)) && local%16==0;
        if(values[4096+index]!=(first?0.5f:-999.0f)) return 5;
    }
    return munmap(memory,24576)==0?0:6;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sum.cpp'
            path.write_text(source)
            binary = path.with_suffix('')
            subprocess.run(['g++', '-std=c++17', '-O2', str(path), '-o', str(binary)], check=True, timeout=30)
            subprocess.run([str(binary)], check=True, timeout=10)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_native_composition_and_blackhole_compiler(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        directory = root / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        original = native_draft_sdpa.replacements
        with scalar_sum_update(), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(sources)
            self.assertEqual(patched['compute_common.hpp'].count(BODY.encode()), 1)
        self.assertIs(native_draft_sdpa.replacements, original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sum.cpp'
            path.write_text(STUB)
            compiler = root / 'runtime/sfpi/compiler/bin/riscv-tt-elf-g++'
            subprocess.run([str(compiler), '-std=c++17', '-O2', '-mcpu=tt-bh', '-c', str(path),
                '-o', str(path.with_suffix('.o'))], check=True, timeout=30)
