import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import native_draft_sdpa
from dspark_ladder_factory import scoped_stats_pack
from dspark_ladder_score_center import HELPER, scalar_score_center
from dspark_ladder_scalar_reciprocal import scalar_reciprocal
from dspark_ladder_sum_update import scalar_sum_update
from dspark_ladder_stage_print import stage_snapshots


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
''' + HELPER


class ScoreCenterTests(unittest.TestCase):
    def test_smoke_selector_is_explicit_and_bounded(self):
        with scalar_score_center(key_tiles=40):
            replacements = native_draft_sdpa.replacements()['compute_common.hpp']
            selected = [after for before, after in replacements if before in (
                '                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);',
                '    sub_bcast_cols_init(in0_cb, in1_cb);',
                '                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);')]
            self.assertEqual(len(selected), 3)
            self.assertTrue(all('== 40' in source and '== 2112' not in source for source in selected))
        with self.assertRaises(ValueError):
            with scalar_score_center(key_tiles=296):
                pass

    @unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux compiler required')
    def test_mask_and_center_preserve_precision_faces_and_ownership(self):
        source = STUB + '''
#include <sys/mman.h>
#include <cmath>
int main() {
    void* memory=mmap(nullptr,32768,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_32BIT,-1,0);
    if(memory==MAP_FAILED) return 2;
    auto* scores=static_cast<float*>(memory);
    auto* mask=reinterpret_cast<uint16_t*>(scores+4096);
    auto* maxima=scores+6144;
    buffers[0]={static_cast<uint32_t>(reinterpret_cast<uintptr_t>(scores)>>4),256};
    buffers[1]={static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mask)>>4),128};
    buffers[2]={static_cast<uint32_t>(reinterpret_cast<uintptr_t>(maxima)>>4),256};
    for(uint32_t index=0;index<4096;++index) { scores[index]=162.189453125f; mask[index]=index%3==0?0xff80:0; }
    for(uint32_t index=0;index<2048;++index) maxima[index]=index<1024?162.0f:160.0f;
    qwen_scalar_score_transform(0,1,4,2,true);
    for(uint32_t index=0;index<4096;++index) {
        if(COMPILE_FOR_TRISC==0 && index%3==0) { if(!std::isinf(scores[index]) || scores[index]>0) return 3; }
        else if(scores[index]!=162.189453125f) return 4;
    }
    qwen_scalar_score_transform(0,2,4,2,false);
    for(uint32_t index=0;index<4096;++index) {
        if(COMPILE_FOR_TRISC==0 && index%3==0) { if(!std::isinf(scores[index]) || scores[index]>0) return 5; }
        else {
            float expected=COMPILE_FOR_TRISC==0 ? (index<2048?0.189453125f:2.189453125f) : 162.189453125f;
            if(scores[index]!=expected) return 6;
        }
        if(mask[index]!=(index%3==0?0xff80:0)) return 7;
    }
    if(popped[0]!=0 || popped[2]!=0 || popped[1]!=(COMPILE_FOR_TRISC==0?4u:0u)) return 8;
    return munmap(memory,32768)==0?0:9;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'score.cpp'
            path.write_text(source)
            for processor in (0, 1, 2):
                binary = Path(directory) / f'score-{processor}'
                subprocess.run(['g++', '-std=c++17', '-O2', f'-DCOMPILE_FOR_TRISC={processor}',
                    str(path), '-o', str(binary)], check=True, timeout=30)
                subprocess.run([str(binary)], check=True, timeout=10)

    @unittest.skipUnless(os.environ.get('TT_NATIVE_TEST_ROOT'), 'Pinned native source required')
    def test_composition_and_blackhole_compile(self):
        root = Path(os.environ['TT_NATIVE_TEST_ROOT'])
        directory = root / native_draft_sdpa.KERNEL_DIRECTORY
        sources = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with scalar_reciprocal(), scalar_sum_update(), stage_snapshots(row=8, column=116), scalar_score_center(), scoped_stats_pack():
            patched = native_draft_sdpa.patched_sources(sources)
            self.assertEqual(patched['compute_common.hpp'].count(HELPER.encode()), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'score.cpp'
            path.write_text(STUB)
            compiler = root / 'runtime/sfpi/compiler/bin/riscv-tt-elf-g++'
            subprocess.run([str(compiler), '-std=c++17', '-O2', '-mcpu=tt-bh', '-DCOMPILE_FOR_TRISC=0',
                '-c', str(path), '-o', str(path.with_suffix('.o'))], check=True, timeout=30)
