import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_ladder_scalar_reciprocal import BODY, scalar_reciprocal


STUB = '''
#include <cstdint>
#define QWEN_DRAFT_EXP_APPROX false
constexpr uint32_t cb_addr_shift=4;
struct Interface { uint32_t fifo_rd_ptr, fifo_page_size; };
Interface state;
Interface& get_local_cb_interface(uint32_t) { return state; }
struct CircularBuffer {
    explicit CircularBuffer(uint32_t) {}
    void wait_front(uint32_t) {}
};
void reciprocal(uint32_t in_cb, uint32_t num_tiles) {
''' + BODY + '\n}\n'


class ScalarReciprocalTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux host compiler required')
    def test_only_first_column_changes_on_unpack_thread(self):
        source = STUB + '''
#include <sys/mman.h>
#include <cmath>
int main() {
    void* allocation=mmap(nullptr,8192,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_32BIT,-1,0);
    if(allocation==MAP_FAILED) return 2;
    auto* values=static_cast<float*>(allocation);
    state={static_cast<uint32_t>(reinterpret_cast<uintptr_t>(allocation)>>4),256};
    for(uint32_t index=0;index<2048;++index) values[index]=1.247028351f;
    reciprocal(0,2);
    for(uint32_t index=0;index<2048;++index) {
        uint32_t local=index%1024;
        bool column=(local<256 || (local>=512 && local<768)) && local%16==0;
        float expected=(COMPILE_FOR_TRISC==0 && column) ? 1.0f/1.247028351f : 1.247028351f;
        if(values[index]!=expected) return 3;
    }
    return munmap(allocation,8192)==0 ? 0 : 4;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reciprocal.cpp'
            path.write_text(source)
            for processor in (0, 1, 2):
                binary = Path(directory) / ('reciprocal-' + str(processor))
                result = subprocess.run(['g++', '-std=c++17', '-O2', '-DCOMPILE_FOR_TRISC=' + str(processor),
                    str(path), '-o', str(binary)], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                subprocess.run([str(binary)], check=True, timeout=10)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_native_source_and_blackhole_object_compilation(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        directory = root / native_draft_sdpa.KERNEL_DIRECTORY
        original = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        replacements = native_draft_sdpa.replacements
        with scalar_reciprocal(), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(original)
            self.assertEqual(patched['compute_common.hpp'].count(BODY.encode()), 1)
        self.assertIs(native_draft_sdpa.replacements, replacements)
        compiler = root / 'runtime/sfpi/compiler/bin/riscv-tt-elf-g++'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reciprocal.cpp'
            path.write_text(STUB)
            result = subprocess.run([str(compiler), '-std=c++17', '-O2', '-mcpu=tt-bh',
                '-DCOMPILE_FOR_TRISC=0', '-c', str(path), '-o', str(path.with_suffix('.o'))],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
