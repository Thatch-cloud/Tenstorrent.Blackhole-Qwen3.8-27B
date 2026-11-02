"""CPU tests of the P2P device ops (ttnn.qwen_p2p_staging_alloc / _info, qwen_blocks_to_staging, qwen_write_blocks_from_staging, qwen_blocks_digest_async, qwen_p2p_warm /
_freeze, qwen_staging_checksum; docs/tp4-p2p-restore.md), everything that can be checked without a card:

  * the checksum algorithm: numpy, plain Python and the three shared test vectors (INTERFACE section 4) agree;
  * the block-row digest: the checksum over a block's shard (its rows in the tensors, in canonical order), against numpy; linearity; k continuing across tensors;
  * qwen_p2p_plan.hpp (the pure arithmetic of the extension) is compiled and every function is compared with its Python mirror (p2p_staging_ref.py), with the Gemma
    geometry (69,632-byte rows of 64 pages of 1,088 bytes, 696,320-byte shards, 96 objects per staging slot) among the cases;
  * the arithmetic block of the checksum kernel is compiled as it stands in the kernel and reproduces the vectors, and the digest kernel carries the same block;
  * ALL FOUR DEVICE KERNELS (scatter, gather, checksum, digest) are compiled against a host stand-in of the dataflow API (emu/) and RUN, driven by the same argument builders
    the extension uses: on random block sets, page sizes, offsets (including the end of the ring), banks and launch splits. The stand-in performs NOC transfers late (at the
    barrier) and refuses a sub-page access, an unaligned DRAM address, an access outside the allowed pages and a kernel that ends with a transfer pending;
  * the emulation has teeth: deliberately broken copies of each kernel are caught;
  * the extension source, the build script and the card test agree with the contract (names, version, argument names and order, validation before enqueue, no private strings).

Run from this directory:  python3 -B -m unittest discover -p 'test_p2p_staging.py'   (numpy and a C++ compiler; python 3.10 or later)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p2p_staging_ref as ref  # noqa: E402

PLAN = os.path.join(HERE, 'qwen_p2p_plan.hpp')
SCATTER = os.path.join(HERE, 'qwen_p2p_scatter.cpp')
CHECKSUM = os.path.join(HERE, 'qwen_p2p_checksum.cpp')
GATHER = os.path.join(HERE, 'qwen_p2p_gather.cpp')
DIGEST = os.path.join(HERE, 'qwen_p2p_digest_rows.cpp')
EXTENSION = os.path.join(HERE, 'qwen_kv_read.cpp')
BUILD = os.path.join(HERE, 'build_kv_read.sh')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def have_compiler():
    return ref.compiler() is not None


class ChecksumReferenceTests(unittest.TestCase):
    def test_the_three_shared_vectors(self):
        for name, data, want in (('zero', ref.vector_zero(), ref.VECTOR_ZERO_4096), ('mod251', ref.vector_mod251(), ref.VECTOR_MOD251_4096),
                                 ('prng', ref.vector_prng(), ref.VECTOR_PRNG_1MIB)):
            with self.subTest(vector=name):
                self.assertEqual(ref.checksum_numpy(data), want)
                self.assertEqual(ref.checksum_python(data), want)

    def test_the_vectors_are_what_the_contract_names(self):
        self.assertEqual((len(ref.vector_zero()), len(ref.vector_mod251()), len(ref.vector_prng())), (4096, 4096, 1 << 20))
        self.assertEqual(ref.vector_mod251()[:5], bytes([0, 1, 2, 3, 4]))
        self.assertEqual(ref.vector_mod251()[251], 0)
        self.assertEqual(ref.PRNG_SEED, 20261030)
        self.assertEqual(ref.vector_prng(), np.random.default_rng(20261030).integers(0, 256, 1 << 20, np.uint8).tobytes())

    def test_the_ranged_vectors(self):
        """4 ranges, each one of the single-range vectors: the ranged reference gives each range's own value (numpy and plain Python agree)."""
        for name, data, length, want in (('small', ref.vector_ranged_small(), 4096, ref.RANGED_SMALL_EXPECTED),
                                         ('mib', ref.vector_ranged_prng(), 1 << 20, ref.RANGED_PRNG_EXPECTED)):
            with self.subTest(vector=name):
                self.assertEqual(len(data), 4 * length)
                self.assertEqual(ref.checksum_ranges(data, length, 4), want)
                if length == 4096:
                    self.assertEqual(ref.checksum_ranges(data, length, 4, ref.checksum_python), want)
        self.assertEqual(ref.RANGED_SMALL_EXPECTED, [0, 17453227578024978262, 17453227578024978262, 0])
        self.assertEqual(ref.RANGED_PRNG_EXPECTED, [4942361704597259676, 0, 4942361704597259676, 0])

    def test_every_range_restarts_k(self):
        """The same bytes in every range give the same sum in every range (k is counted from each range's own start, not from the first range)."""
        block = ref.vector_prng()[:8192]
        sums = ref.checksum_ranges(block * 5, 8192, 5)
        self.assertEqual(sums, [ref.checksum_numpy(block)] * 5)
        self.assertNotEqual(ref.checksum_numpy(block * 5), sums[0] * 5 & ref.MASK64)

    def test_k_counts_from_the_start_of_the_range(self):
        """The weights restart at every object: a range embedded at a byte offset has the checksum of the range alone."""
        data = ref.vector_prng()[:65536]
        padded = bytes(4096) + data
        self.assertEqual(ref.checksum_numpy(padded[4096:]), ref.checksum_numpy(data))
        self.assertNotEqual(ref.checksum_numpy(padded), ref.checksum_numpy(data))

    def test_one_flipped_bit_changes_the_sum(self):
        data = bytearray(ref.vector_prng()[:8192])
        want = ref.checksum_numpy(bytes(data))
        for position in (0, 3, 4096, 8191):
            changed = bytearray(data)
            changed[position] ^= 0x10
            self.assertNotEqual(ref.checksum_numpy(bytes(changed)), want, position)

    def test_word_order_matters(self):
        """A permutation of the words is not a collision (the weights make the sum position dependent)."""
        words = np.arange(1, 2049, dtype='<u4')
        self.assertNotEqual(ref.checksum_numpy(words.tobytes()), ref.checksum_numpy(words[::-1].copy().tobytes()))

    def test_the_sum_wraps_modulo_2_to_the_64(self):
        data = bytes([0xFF]) * 4096
        want = sum(((k * ref.STEP + 1) & ref.MASK32) * 0xFFFFFFFF for k in range(1024)) & ref.MASK64
        self.assertEqual(ref.checksum_python(data), want)
        self.assertEqual(ref.checksum_numpy(data), want)
        self.assertGreater(sum(((k * ref.STEP + 1) & ref.MASK32) * 0xFFFFFFFF for k in range(1024)), 1 << 64, 'this input does wrap')

    def test_partial_sums_add_to_the_whole(self):
        """What the cores do: sums of disjoint unit runs, each with its own start weight, add up to the whole sum."""
        data = ref.vector_prng()[:4096 * 37]
        units = len(data) // 4096
        for cores in (1, 8, 64, 80):
            total = 0
            for first, last in ref.split_units(units, cores):
                part = data[first * 4096:last * 4096]
                words = np.frombuffer(part, dtype='<u4').astype(np.uint64)
                k = np.arange(words.size, dtype=np.uint64) + np.uint64(first * 1024)
                weights = (k * np.uint64(ref.STEP) + np.uint64(1)) & np.uint64(ref.MASK32)
                total = (total + int((weights * words).sum(dtype=np.uint64))) & ref.MASK64
            self.assertEqual(total, ref.checksum_numpy(data), cores)


MIB = 1 << 20
# check_scatter(byte_offset, slice_bytes, page_bytes, blocks, staging_bytes, row_stride): the contiguous cases (row_stride == slice_bytes) are the ones the
# op had before row_stride; the rest are strided rows.
SCATTER_CASES = [
    (64, 17408, 1088, 3, MIB, 17408),                      # 0: fine
    (0, 17408, 1088, 0, MIB, 17408),                       # no blocks
    (0, 17000, 1088, 1, MIB, 17000),                       # a block that is not whole pages
    (32, 17408, 1088, 1, MIB, 17408),                      # byte_offset not a multiple of 64
    (0, 17408, 1000, 1, MIB, 17408),                       # a page that is not a multiple of 64
    (MIB, 17408, 1088, 1, MIB, 17408),                     # starts at the end
    (MIB - 3 * 17408, 17408, 1088, 3, MIB, 17408),         # 6: ends exactly at the end of the ring
    (MIB - 3 * 17408, 17408, 1088, 4, MIB, 17408),         # one block past the end
    (0, 65536, 8192, 1, MIB, 65536),                       # the largest page
    (0, 65536, 16384, 1, MIB, 65536),                      # a page above 8192
    (0, 17408, 1088, 3, MIB, 17408 + 64),                  # 10: strided, fine
    (0, 17408, 1088, 3, 4 * MIB, 557056),                  # strided: the chipmajor-v1 rows (32 caches x 17408 apart)
    (0, 17408, 1088, 3, MIB, 17408 + 32),                  # 12: a stride that is not a multiple of 64
    (0, 17408, 1088, 3, MIB, 17408 - 64),                  # a stride smaller than the block (overlapping rows)
    (0, 17408, 1088, 3, MIB, 0),                           # a stride of zero
    (0, 17408, 1088, 2, 17408 + 557056, 557056),           # 15: strided and ending exactly at the end (the last row's last byte is the last byte)
    (0, 17408, 1088, 2, 17408 + 557056 - 64, 557056),      # strided, one row byte past the end... 64 bytes short
    (0, 17408, 1088, 1, 17408, 557056),                    # one row needs no room for a stride
    (64, 17408, 1088, 1, 17408, 557056),                   # 18: one row, shifted by 64, 64 bytes past the end
    (0, 17408, 1088, 1000, MIB, 1 << 40),                  # a stride so large that blocks * stride would wrap nothing but is far outside
    (0, 17408, 1088, (1 << 33), (4 << 30) - 1, 17408),     # 20: more blocks than 32 bits can count, contiguous: outside
    (0, 17408, 1088, 2, MIB, (1 << 63) + 64),              # a huge stride (blocks * stride would wrap 64 bits)
]
# check_checksum(byte_offset, length, staging_bytes, cores, ranges)
CHECKSUM_CASES = [
    (0, 4096, 8192, 64, 1),                                # 0: fine
    (0, 0, 8192, 64, 1),                                   # empty
    (0, 4000, 8192, 64, 1),                                # not a multiple of 4096
    (8, 4096, 8192, 64, 1),                                # byte_offset not a multiple of 64
    (4096, 8192, 8192, 64, 1),                             # past the end
    (0, 4096, 8192, 7, 1),                                 # cores not a multiple of 8
    (0, 4096, 8192, 88, 1),                                # more than 80 cores
    (0, 4096, 8192, 0, 1),                                 # no cores
    (0, 4096, 8192, 64, 2),                                # 8: two ranges, exactly the staging
    (0, 4096, 8192, 64, 3),                                # a third range is outside
    (0, 4096, 8192, 64, 0),                                # count 0
    (4096, 4096, 8192, 64, 1),                             # 11: the last range
    (4096, 4096, 8192, 64, 2),                             # the last range and one more
    (64, 4096, 8192, 64, 2),                               # two ranges shifted by 64: the second ends 64 past the end
    (64, 4096 * 3, 1 << 20, 80, 85),                       # 14: fine
    (64, 4096 * 3, 1 << 20, 80, 86),                       # the 86th range ends past the end
    (0, 4096, 1 << 30, 64, 1 << 18),                       # 16: 2^18 ranges (several launches): fine
    (0, 4096, 1 << 30, 64, (1 << 18) + 1),                 # one more
    (0, 4096, 8192, 64, (1 << 32) - 1),                    # 18: a count that wraps 32 bits when multiplied: outside
]
# scatter_core_args(destination 111, staging 222, bank 5, byte_offset, row_stride, blocks, launch_first, run): (byte_offset, row_stride, launch_first, run_first, run_count)
SCATTER_ARG_CASES = [
    (4160, 17408, 0, 3, 4),                                # contiguous rows
    (4160, 557056, 0, 3, 4),                               # strided rows
    (64, 17408 + 128, 16, 5, 3),                           # a second launch with a stride that is not the block size
    (0, 557056, 0, 0, 0),                                  # an idle core
]
# checksum_core_args(staging 111, partials 222, bank 5, byte_offset, units (first, last), slot, ranges, range_bytes, slot_stride)
CHECKSUM_ARG_CASES = [
    (4160, 7, 12, 3, 1, 64 * 4096, 64),                    # one range
    (4160, 7, 12, 3, 60, 557056, 64),                      # sixty ranges of 136 units
    (0, 0, 0, 79, 512, 4096, 80),                          # an idle core of the largest launch
]


GEMMA_PAGE = 1088                      # one bfp8 tile of a Gemma KV block
GEMMA_PAGES_PER_BLOCK = 64             # a block is 64 pages
GEMMA_ROW = GEMMA_PAGE * GEMMA_PAGES_PER_BLOCK   # 69,632 bytes: one block's row of ONE tensor on one chip
GEMMA_TENSORS = 10                     # K0, V0 .. K4, V4
GEMMA_SHARD = GEMMA_TENSORS * GEMMA_ROW  # 696,320 bytes: one block (object) on one chip, 170 x 4096
GEMMA_RING = 64 << 20                  # staging bytes per bank
GEMMA_CANARY = 4096
GEMMA_SLOT_OBJECTS = 96
# check_gather(byte_offset, slice_bytes, page_bytes, pages_per_block, blocks, staging_bytes, row_stride): Gemma's geometry, the slot of 96 objects of bank 0.
GATHER_CASES = [
    (8192, GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING, GEMMA_SHARD),                    # 0: fine: the slot after the canary page
    (8192, GEMMA_ROW, GEMMA_PAGE, 64, 0, GEMMA_RING, GEMMA_SHARD),                     # no blocks
    (8192, GEMMA_ROW, GEMMA_PAGE, 63, 1, GEMMA_RING, GEMMA_SHARD),                     # pages_per_block one short of the tensor's block
    (8192, GEMMA_ROW, GEMMA_PAGE, 65, 1, GEMMA_RING, GEMMA_SHARD),                     # one over
    (0, GEMMA_ROW, GEMMA_PAGE, 0, 1, GEMMA_RING, GEMMA_SHARD),                         # pages_per_block 0
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 513, GEMMA_RING, GEMMA_ROW),                        # 5: more than one launch
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 512, GEMMA_RING, GEMMA_ROW),                        # 512 blocks, contiguous: fine
    (GEMMA_RING - (95 * GEMMA_SHARD + GEMMA_ROW), GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING, GEMMA_SHARD),   # 7: ends on the last byte of the bank's range: fine
    (GEMMA_RING - (95 * GEMMA_SHARD + GEMMA_ROW) + 64, GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING, GEMMA_SHARD),   # 8: 64 bytes later: past the end
    (8192, GEMMA_ROW, GEMMA_PAGE, 64, 1, GEMMA_RING, GEMMA_ROW - 64),                  # a stride below the block
    (8192, GEMMA_ROW, GEMMA_PAGE, 64, 2, GEMMA_RING, GEMMA_SHARD + 32),                # 10: a stride that is not a multiple of 64
    (8200, GEMMA_ROW, GEMMA_PAGE, 64, 1, GEMMA_RING, GEMMA_SHARD),                     # byte_offset not a multiple of 64
    (0, 64000, 1000, 64, 1, GEMMA_RING, 64000),                                        # 12: a page that is not a multiple of 64
    (0, 131072, 16384, 8, 1, GEMMA_RING, 131072),                                      # a page above 8192
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 1, 0, GEMMA_ROW),                                   # 14: an empty staging range
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 2, GEMMA_RING, 1 << 63),                            # a stride so large that blocks * stride would wrap 64 bits
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 1, GEMMA_ROW, 1 << 40),                             # 16: one row needs no room for a stride: fine
    (0, GEMMA_ROW + 64, GEMMA_PAGE, 64, 1, GEMMA_RING, GEMMA_ROW + 64),                # a block that is not whole pages
    (0, GEMMA_ROW, GEMMA_PAGE, 64, 1 << 33, GEMMA_RING, GEMMA_ROW),                    # 18: more blocks than 32 bits can count
]
GATHER_OK = [0, 6, 7, 16]
# check_digest_rows(tensors, blocks, pages_per_block, page_bytes, slices)
DIGEST_CASES = [
    (10, 96, 64, 1088, 19305),                             # 0: Gemma's slot of 96 objects, 10 tensors, the production pool
    (0, 96, 64, 1088, 19305),                              # no tensors
    (17, 96, 64, 1088, 19305),                             # 17 tensors
    (16, 128, 64, 1088, 19305),                            # 3: the largest call
    (1, 1, 64, 1088, 19305),                               # the smallest
    (10, 0, 64, 1088, 19305),                              # no blocks
    (10, 129, 64, 1088, 19305),                            # 6: 129 blocks
    (10, 96, 64, 1000, 19305),                             # a page that is not a multiple of 64
    (10, 96, 64, 16384, 19305),                            # a page above 8192
    (10, 96, 0, 1088, 19305),                              # pages_per_block 0
    (10, 96, 64, 1088, 0),                                 # 10: no slices
    (10, 96, 64, 1088, 1 << 27),                           # page ids past 32 bits
    (10, 96, 64, 1088, (1 << 26) - 1),                     # 12: the last pool whose page ids fit
    (10, 96, 64, 1088, 1 << 26),                           # one more block than that
    (10, 96, 64, 8192, 100),                               # 14: the largest page
]
DIGEST_OK = [0, 3, 4, 12, 14]
# check_staging(size_bytes, page_bytes, num_banks, bank_bytes)
STAGING_CASES = [
    (1 << 20, 4096, 8, 1 << 30),                           # 0: fine
    (1 << 20, 4096, 0, 1 << 30),                           # no banks
    (1000, 4096, 8, 1 << 30),                              # not a multiple of the page
    (1 << 20, 100, 8, 1 << 30),                            # a page that is not a multiple of 64
    (0, 4096, 8, 1 << 30),                                 # empty
    (2 << 30, 4096, 8, 1 << 30),                           # more than the bank can give
    (1 << 32, 4096, 8, 1 << 40),                           # 32-bit offsets
    (1 << 30, 4096, 8, 1 << 30),                           # 7: exactly what the bank can give: fine
    (6400, 64, 8, 1 << 30),                                # a multiple of the page but not of 4096
    (GEMMA_RING, 65536, 8, 4 << 30),                       # 9: Gemma's 64 MiB per bank
]
STAGING_OK = [0, 7, 9]
# digest_core_args(partials 5000, tensor addresses 1000 + 64 t, block ids 7 i + 3, run): (tensors, blocks in the call, run first, run count)
DIGEST_ARG_CASES = [
    (10, 96, 0, 2),                                        # a core with two blocks
    (10, 96, 95, 1),                                       # the last block
    (10, 96, 96, 0),                                       # a core past the end
    (16, 128, 126, 2),                                     # the largest call, the last core
    (1, 1, 0, 1),
    (3, 5, 4, 1),
]
# check_block_ids(blocks, 130, require_distinct): the blocks of each case
BLOCK_ID_CASES = [[1, 2, 3], [1, 2, 2], [5, 130, 7], [], [0], [129, 3, 129], [131]]


def cpp_table(name, rows):
    """A C++ table of uint64 rows -> the source of `static const unsigned long long name[][n] = {...}` (as std::vector rows for range-for)."""
    return 'static const std::vector<std::vector<unsigned long long>> %s = {%s};' % (name, ', '.join('{%s}' % ', '.join('%dull' % v for v in row) for row in rows))


class PlanMirrorTests(unittest.TestCase):
    """qwen_p2p_plan.hpp compiled and run against the Python mirrors."""

    @classmethod
    def setUpClass(cls):
        if not have_compiler():
            raise unittest.SkipTest('no C++ compiler')
        cls.directory = tempfile.mkdtemp(prefix='p2p_plan_')
        source = os.path.join(cls.directory, 'dump.cpp')
        with open(source, 'w') as handle:
            handle.write('''#include <cstdio>
#include <vector>
#include "%s"
using namespace qwen_p2p;
static void pr(const char* s) { std::printf("%%s\\n", s ? s : "OK"); }
%s
%s
%s
%s
%s
%s
%s
%s
int main() {
  for (uint32_t n : {0u, 1u, 5u, 16u, 17u, 100u, 512u, 513u, 1030u}) {
    std::printf("L %%u:", n);
    for (auto l : scatter_launches(n)) std::printf(" %%u,%%u", l.first, l.second);
    std::printf("\\n");
  }
  for (uint32_t n : {0u, 1u, 5u, 511u, 512u, 513u, 1024u, 1030u}) {
    std::printf("Q %%u:", n);
    for (auto l : checksum_launches(n)) std::printf(" %%u,%%u", l.first, l.second);
    std::printf("\\n");
  }
  for (uint32_t n : {1u, 5u, 16u, 17u, 100u, 511u, 512u}) {
    std::printf("B %%u:", n);
    for (auto r : split_blocks(n, kScatterCores)) std::printf(" %%u,%%u", r.first, r.second);
    std::printf("\\n");
  }
  for (uint64_t u : {1ull, 3ull, 33ull, 64ull, 65ull, 256ull, 1000ull}) for (uint32_t c : {8u, 64u, 80u}) {
    std::printf("U %%llu %%u:", (unsigned long long)u, c);
    for (auto r : split_units(u, c)) std::printf(" %%llu,%%llu", (unsigned long long)r.first, (unsigned long long)r.second);
    std::printf("\\n");
  }
  for (uint64_t k : {0ull, 1ull, 1024ull, 123456789ull, 1ull << 33, (1ull << 40) + 7}) std::printf("W %%llu %%u\\n", (unsigned long long)k, start_weight(k));
  for (uint32_t p : {64u, 1088u, 4096u}) std::printf("C %%u %%u %%u\\n", p, scatter_cb_bytes(p), checksum_cb_bytes());
  for (const auto& c : scatter_cases) { std::printf("S "); pr(check_scatter(c[0], c[1], c[2], c[3], c[4], c[5])); }
  for (const auto& c : checksum_cases) { std::printf("K "); pr(check_checksum(c[0], c[1], c[2], (uint32_t)c[3], c[4])); }
  std::vector<uint32_t> blocks; for (uint32_t i = 0; i < 40; ++i) blocks.push_back(1000 - 7 * i);
  for (auto a : scatter_arg_cases) {
    auto args = scatter_core_args(111, 222, 5, a[0], a[1], blocks, (uint32_t)a[2], {(uint32_t)a[3], (uint32_t)a[4]});
    std::printf("A"); for (auto v : args) std::printf(" %%u", v); std::printf("\\n");
  }
  for (auto a : checksum_arg_cases) {
    auto cargs = checksum_core_args(111, 222, 5, a[0], {a[1], a[2]}, (uint32_t)a[3], (uint32_t)a[4], a[5], (uint32_t)a[6]);
    std::printf("T"); for (auto v : cargs) std::printf(" %%u", v); std::printf("\\n");
  }
  for (const auto& c : gather_cases) { std::printf("H "); pr(check_gather(c[0], c[1], c[2], c[3], c[4], c[5], c[6])); }
  for (const auto& c : digest_cases) { std::printf("D "); pr(check_digest_rows(c[0], c[1], c[2], c[3], c[4])); }
  for (const auto& c : staging_cases) { std::printf("G "); pr(check_staging(c[0], c[1], (uint32_t)c[2], c[3])); }
  for (uint64_t bank : {0ull, 7ull, 8ull, 1000ull}) { std::printf("N "); pr(check_bank(bank, 8)); }
  {
    std::vector<std::vector<uint32_t>> ids = {{1, 2, 3}, {1, 2, 2}, {5, 130, 7}, {}, {0}, {129, 3, 129}, {131}};
    for (const auto& v : ids) for (int distinct = 0; distinct < 2; ++distinct) {
      size_t bad = 99; const char* r = check_block_ids(v, 130, distinct != 0, &bad);
      std::printf("I %%s %%zu\\n", r ? r : "OK", r ? bad : (size_t)0);
    }
  }
  for (uint32_t p : {64u, 1088u, 4096u, 8192u}) std::printf("E %%u %%u %%u\\n", p, gather_cb_bytes(p), digest_cb_bytes(p));
  for (auto a : scatter_arg_cases) {
    auto gargs = gather_core_args(111, 222, 5, a[0], a[1], blocks, (uint32_t)a[2], {(uint32_t)a[3], (uint32_t)a[4]});
    std::printf("A2"); for (auto v : gargs) std::printf(" %%u", v); std::printf("\\n");
  }
  for (auto a : digest_arg_cases) {
    std::vector<uint32_t> addresses; for (uint32_t t = 0; t < a[0]; ++t) addresses.push_back(1000 + 64 * t);
    std::vector<uint32_t> ids; for (uint32_t i = 0; i < a[1]; ++i) ids.push_back(7 * i + 3);
    auto dargs = digest_core_args(5000, addresses, ids, {(uint32_t)a[2], (uint32_t)a[3]});
    std::printf("Z"); for (auto v : dargs) std::printf(" %%u", v); std::printf("\\n");
  }
  for (uint32_t n : {96u, 128u, 1u, 80u, 81u}) {
    std::printf("Y"); for (auto r : split_blocks(n, kDigestMaxCores)) std::printf(" %%u,%%u", r.first, r.second); std::printf("\\n");
  }
}
''' % (PLAN.replace('\\', '/'), cpp_table('scatter_cases', SCATTER_CASES), cpp_table('checksum_cases', CHECKSUM_CASES),
       cpp_table('scatter_arg_cases', SCATTER_ARG_CASES), cpp_table('checksum_arg_cases', CHECKSUM_ARG_CASES), cpp_table('gather_cases', GATHER_CASES),
       cpp_table('digest_cases', DIGEST_CASES), cpp_table('digest_arg_cases', DIGEST_ARG_CASES), cpp_table('staging_cases', STAGING_CASES)))
        exe = os.path.join(cls.directory, 'dump')
        done = subprocess.run([ref.compiler(), '-std=c++17', '-Wall', '-Wextra', '-Werror', source, '-o', exe], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        cls.lines = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.splitlines()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def of(self, tag):
        return [line[len(tag) + 1:] for line in self.lines if line.startswith(tag + ' ')]

    def test_constants_match_the_python_mirror(self):
        text = read(PLAN)
        for name, value in (('kScatterColumns', ref.SCATTER_COLUMNS), ('kScatterRows', ref.SCATTER_ROWS), ('kScatterMaxBlocksPerCore', ref.SCATTER_MAX_BLOCKS_PER_CORE),
                            ('kScatterChunkPages', ref.SCATTER_CHUNK_PAGES), ('kScatterMaxPageBytes', ref.SCATTER_MAX_PAGE_BYTES), ('kScatterFixedArgs', ref.SCATTER_FIXED_ARGS), ('kChecksumUnit', ref.CHECKSUM_UNIT),
                            ('kChecksumChunkBytes', ref.CHECKSUM_CHUNK_BYTES), ('kChecksumPartialBytes', ref.CHECKSUM_PARTIAL_BYTES),
                            ('kChecksumColumns', ref.CHECKSUM_COLUMNS), ('kChecksumDefaultCores', ref.CHECKSUM_DEFAULT_CORES),
                            ('kChecksumMaxCores', ref.CHECKSUM_MAX_CORES), ('kChecksumMaxRanges', ref.CHECKSUM_MAX_RANGES), ('kChecksumArgs', ref.CHECKSUM_ARGS),
                            ('kDramAlignment', ref.DRAM_ALIGNMENT), ('kP2PMaxBlocksPerCall', ref.P2P_MAX_BLOCKS_PER_CALL), ('kDigestColumns', ref.DIGEST_COLUMNS),
                            ('kDigestMaxCores', ref.DIGEST_MAX_CORES), ('kDigestMaxTensors', ref.DIGEST_MAX_TENSORS), ('kDigestMaxBlocks', ref.DIGEST_MAX_BLOCKS),
                            ('kDigestMaxBlocksPerCore', ref.DIGEST_MAX_BLOCKS_PER_CORE), ('kDigestChunkPages', ref.DIGEST_CHUNK_PAGES),
                            ('kDigestMaxPageBytes', ref.DIGEST_MAX_PAGE_BYTES), ('kDigestPartialBytes', ref.DIGEST_PARTIAL_BYTES),
                            ('kDigestRingBuffers', ref.DIGEST_RING_BUFFERS), ('kDigestFixedArgs', ref.DIGEST_FIXED_ARGS), ('kDigestArgs', ref.DIGEST_ARGS)):
            match = re.search(r'constexpr uint32_t %s = (\d+);' % name, text)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), value, name)
        self.assertIn('constexpr uint64_t kChecksumStep = %dull;' % ref.STEP, text)

    def test_launches_and_runs(self):
        for line in self.of('L'):
            n, rest = line.split(':')
            got = [tuple(int(x) for x in item.split(',')) for item in rest.split()]
            self.assertEqual(got, ref.scatter_launches(int(n)), line)
        for line in self.of('Q'):
            n, rest = line.split(':')
            got = [tuple(int(x) for x in item.split(',')) for item in rest.split()]
            self.assertEqual(got, ref.checksum_launches(int(n)), line)
            self.assertEqual(sum(count for _, count in got), int(n))
            self.assertTrue(all(count <= ref.CHECKSUM_MAX_RANGES for _, count in got))
            self.assertTrue(all(a[0] + a[1] == b[0] for a, b in zip(got, got[1:])), 'launches are contiguous')
        for line in self.of('B'):
            n, rest = line.split(':')
            got = [tuple(int(x) for x in item.split(',')) for item in rest.split()]
            self.assertEqual(got, ref.split_blocks(int(n), ref.SCATTER_CORES), line)
            self.assertEqual(sum(count for _, count in got), int(n))
            self.assertLessEqual(max(count for _, count in got), ref.SCATTER_MAX_BLOCKS_PER_CORE)
        for line in self.of('U'):
            head, rest = line.split(':')
            units, cores = (int(x) for x in head.split())
            got = [tuple(int(x) for x in item.split(',')) for item in rest.split()]
            self.assertEqual(got, ref.split_units(units, cores), line)
            self.assertEqual(got[0][0], 0)
            self.assertEqual(got[-1][1], units)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(got, got[1:])), 'runs are contiguous')

    def test_weights_and_buffer_sizes(self):
        for line in self.of('W'):
            k, w = (int(x) for x in line.split())
            self.assertEqual(w, ref.start_weight(k), line)
            self.assertEqual(w, (k * ref.STEP + 1) % (1 << 32))
        for line in self.of('C'):
            page, scatter, checksum = (int(x) for x in line.split())
            self.assertEqual((scatter, checksum), (ref.scatter_cb_bytes(page), ref.checksum_cb_bytes()), line)

    def test_argument_checks(self):
        want = [ref.check_scatter(*case) for case in SCATTER_CASES]
        self.assertEqual([None if x == 'OK' else x for x in self.of('S')], want)
        self.assertEqual([index for index, problem in enumerate(want) if problem is None], [0, 6, 8, 10, 11, 15, 17])
        self.assertIsNotNone(want[7], 'one block past the end is refused')
        self.assertIsNone(want[6], 'a run that ends exactly at the end of the ring is fine')
        self.assertIsNone(want[15], 'and so does a strided one: the last row ends on the last byte')
        for index in (12, 13, 14):
            self.assertIn('row_stride', want[index], index)
        for index in (16, 18, 19, 20, 21):
            self.assertEqual(want[index], 'the byte range lies outside the staging range', index)
        want = [ref.check_checksum(*case) for case in CHECKSUM_CASES]
        self.assertEqual([None if x == 'OK' else x for x in self.of('K')], want)
        self.assertEqual([index for index, problem in enumerate(want) if problem is None], [0, 8, 11, 14, 16])
        self.assertEqual(want[10], 'count must be at least 1')
        for index in (9, 12, 13, 15, 17, 18):
            self.assertEqual(want[index], 'the byte range lies outside the staging range', index)

    def test_gather_checks_with_gemma_geometry(self):
        want = [ref.check_gather(*case) for case in GATHER_CASES]
        self.assertEqual([None if x == 'OK' else x for x in self.of('H')], want)
        self.assertEqual([index for index, problem in enumerate(want) if problem is None], GATHER_OK)
        self.assertEqual(want[1], 'no blocks')
        for index in (2, 3, 4, 17):
            self.assertIn('pages_per_block', want[index], index)
        for index in (5, 18):
            self.assertEqual(want[index], 'at most 512 blocks per call (one launch)', index)
        self.assertIn('row_stride', want[9])
        self.assertIn('row_stride', want[10])
        self.assertIn('multiple of 64', want[11])
        self.assertIn('page size must be a multiple of 64', want[12])
        self.assertIn('above 8192', want[13])
        for index in (8, 14, 15):
            self.assertEqual(want[index], 'the byte range lies outside the staging range', index)
        # the slot of the design: the canary page (4 KiB), then 96 objects of 696,320 bytes, fits one 64 MiB bank with room to spare
        self.assertEqual(GEMMA_ROW, 69632)
        self.assertEqual(GEMMA_SHARD, 696320)
        self.assertEqual(GEMMA_SHARD % 4096, 0)
        self.assertLessEqual(GEMMA_CANARY + GEMMA_SLOT_OBJECTS * GEMMA_SHARD, GEMMA_RING)
        self.assertEqual(GEMMA_CANARY + GEMMA_SLOT_OBJECTS * GEMMA_SHARD, 66850816)
        # every tensor's rows of the slot: the tensor t of object i sits at slot base + t * 69,632 + i * 696,320 and the last of them ends inside the slot
        last = GEMMA_CANARY + (GEMMA_TENSORS - 1) * GEMMA_ROW + (GEMMA_SLOT_OBJECTS - 1) * GEMMA_SHARD + GEMMA_ROW
        self.assertEqual(last, GEMMA_CANARY + GEMMA_SLOT_OBJECTS * GEMMA_SHARD)
        for tensor in range(GEMMA_TENSORS):
            self.assertIsNone(ref.check_gather(GEMMA_CANARY + tensor * GEMMA_ROW, GEMMA_ROW, GEMMA_PAGE, 64, GEMMA_SLOT_OBJECTS, GEMMA_RING, GEMMA_SHARD), tensor)

    def test_the_edge_of_the_bank_is_exact_for_the_gather(self):
        """The last byte of the bank's range is writable; one 64-byte unit further is not (an off-by-one in either direction fails here)."""
        edge = GEMMA_RING - (95 * GEMMA_SHARD + GEMMA_ROW)
        self.assertEqual(edge % 64, 0)
        self.assertIsNone(ref.check_gather(edge, GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING, GEMMA_SHARD))
        self.assertIsNotNone(ref.check_gather(edge + 64, GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING, GEMMA_SHARD))
        self.assertIsNotNone(ref.check_gather(edge, GEMMA_ROW, GEMMA_PAGE, 64, 97, GEMMA_RING, GEMMA_SHARD))
        self.assertIsNone(ref.check_gather(edge, GEMMA_ROW, GEMMA_PAGE, 64, 96, GEMMA_RING + 1, GEMMA_SHARD))

    def test_digest_checks_with_gemma_geometry(self):
        want = [ref.check_digest_rows(*case) for case in DIGEST_CASES]
        self.assertEqual([None if x == 'OK' else x for x in self.of('D')], want)
        self.assertEqual([index for index, problem in enumerate(want) if problem is None], DIGEST_OK)
        self.assertEqual(want[1], 'tensors must be 1 to 16')
        self.assertEqual(want[2], 'tensors must be 1 to 16')
        self.assertEqual(want[5], 'blocks must be 1 to 128')
        self.assertEqual(want[6], 'blocks must be 1 to 128')
        self.assertIn('multiple of 64', want[7])
        self.assertIn('above 8192', want[8])
        self.assertIn('pages_per_block', want[9])
        for index in (10, 11, 13):
            self.assertIn('32 bits', want[index], index)

    def test_staging_and_bank_checks(self):
        want = [ref.check_staging(*case) for case in STAGING_CASES]
        self.assertEqual([None if x == 'OK' else x for x in self.of('G')], want)
        self.assertEqual([index for index, problem in enumerate(want) if problem is None], STAGING_OK)
        self.assertIn('multiple of 4096', want[8])
        self.assertEqual(want[1], 'the device has no DRAM banks')
        banks = [None if x == 'OK' else x for x in self.of('N')]
        self.assertEqual(banks, [ref.check_bank(b, 8) for b in (0, 7, 8, 1000)])
        self.assertEqual([b is None for b in banks], [True, True, False, False])

    def test_block_id_checks(self):
        lines = self.of('I')
        self.assertEqual(len(lines), 2 * len(BLOCK_ID_CASES))
        index = 0
        for blocks in BLOCK_ID_CASES:
            for distinct in (False, True):
                reason, bad = ref.check_block_ids(blocks, 130, distinct)
                words = lines[index].rsplit(' ', 1)
                got_reason, got_bad = words[0], int(words[1])
                self.assertEqual((None if got_reason == 'OK' else got_reason, got_bad if got_reason != 'OK' else None), (reason, bad if reason else None), (blocks, distinct))
                index += 1
        self.assertEqual(ref.check_block_ids([1, 2, 2], 130, False), (None, None), 'a gather and a digest only read: a repeated id is fine')
        self.assertEqual(ref.check_block_ids([1, 2, 2], 130, True)[0][:17], 'the block ids mus', 'a scatter needs distinct ids')
        self.assertEqual(ref.check_block_ids([5, 130, 7], 130, False), ('a block id is outside the tensor', 1), 'the first offender is named')
        self.assertEqual(ref.check_block_ids([129, 3, 129], 130, True)[1], 0)

    def test_gather_and_digest_arguments_and_buffers(self):
        scatter_lines, gather_lines = self.of('A'), self.of('A2')
        self.assertEqual(scatter_lines, gather_lines, 'the gather\'s runtime args are the scatter\'s, value for value')
        blocks = [1000 - 7 * i for i in range(40)]
        for line, (byte_offset, row_stride, launch_first, first, count) in zip(gather_lines, SCATTER_ARG_CASES):
            args = ref.gather_core_args(111, 222, 5, byte_offset, row_stride, blocks, launch_first, (first, count))
            self.assertEqual([int(x) for x in line.split()], args)
            self.assertEqual(len(args), ref.SCATTER_ARGS, 'the gather has the scatter\'s constant argument count')
        for line, (tensors, total, first, count) in zip(self.of('Z'), DIGEST_ARG_CASES):
            addresses = [1000 + 64 * t for t in range(tensors)]
            ids = [7 * i + 3 for i in range(total)]
            args = ref.digest_core_args(5000, addresses, ids, (first, count))
            self.assertEqual([int(x) for x in line.split()], args)
            self.assertEqual(len(args), ref.DIGEST_ARGS)
            self.assertEqual(args[:4], [5000, tensors, count, first])
            self.assertEqual(args[4:4 + tensors], addresses)
            self.assertTrue(all(v == 0 for v in args[4 + tensors:4 + ref.DIGEST_MAX_TENSORS]), 'unused tensor slots are zero')
            self.assertEqual(args[20:20 + count], ids[first:first + count])
            self.assertTrue(all(v == 0 for v in args[20 + count:]), 'unused block slots are zero')
        for line, page in zip(self.of('E'), (64, 1088, 4096, 8192)):
            words = [int(x) for x in line.split()]
            self.assertEqual(words, [page, ref.gather_cb_bytes(page), ref.digest_cb_bytes(page)])
        self.assertEqual(ref.gather_cb_bytes(1088), 34880, 'the gather\'s scratch is the design\'s 34,880 bytes')
        self.assertEqual(ref.digest_cb_bytes(1088), 64 + 2 * 8 * 1088 + 64)

    def test_the_digest_covers_every_block_of_a_call_with_at_most_two_per_core(self):
        for line, n in zip(self.of('Y'), (96, 128, 1, 80, 81)):
            runs = [tuple(int(x) for x in item.split(',')) for item in line.split()]
            self.assertEqual(runs, ref.split_blocks(n, ref.DIGEST_MAX_CORES))
            self.assertEqual(len(runs), ref.DIGEST_MAX_CORES)
            self.assertEqual(sum(count for _, count in runs), n)
            self.assertLessEqual(max(count for _, count in runs), ref.DIGEST_MAX_BLOCKS_PER_CORE)
            self.assertTrue(all(a[0] + a[1] == b[0] for a, b in zip(runs, runs[1:])), 'runs are contiguous')
        self.assertEqual(ref.DIGEST_MAX_CORES % ref.DIGEST_COLUMNS, 0)
        self.assertEqual((ref.DIGEST_COLUMNS, ref.DIGEST_MAX_CORES // ref.DIGEST_COLUMNS), (10, 8), 'ten columns by eight rows fits the 10 x 9 grid the serving image pins as well as 13 x 10')

    def test_the_contiguous_default_is_todays_behaviour(self):
        """row_stride == slice_bytes (what None means) is accepted and refused exactly where the contiguous op was: the same reasons, the same boundaries."""
        for byte_offset in (0, 64, 4096, MIB - 17408 * 5, MIB - 17408 * 5 + 64, MIB - 17408):
            for blocks in (0, 1, 4, 5, 6, 61, 62):
                old = None
                if blocks == 0:
                    old = 'no blocks'
                elif byte_offset > MIB or blocks > (MIB - byte_offset) // 17408:
                    old = 'the byte range lies outside the staging range'
                self.assertEqual(ref.check_scatter(byte_offset, 17408, 1088, blocks, MIB), old, (byte_offset, blocks))
                self.assertEqual(ref.check_scatter(byte_offset, 17408, 1088, blocks, MIB, 17408), old, (byte_offset, blocks))

    def test_runtime_argument_vectors_are_constant_length_and_match(self):
        blocks = [1000 - 7 * i for i in range(40)]
        lines = self.of('A')
        self.assertEqual(len(lines), len(SCATTER_ARG_CASES))
        for line, (byte_offset, row_stride, launch_first, first, count) in zip(lines, SCATTER_ARG_CASES):
            args = ref.scatter_core_args(111, 222, 5, byte_offset, row_stride, blocks, launch_first, (first, count))
            self.assertEqual(len(args), ref.SCATTER_ARGS)
            self.assertEqual([int(x) for x in line.split()], args)
            self.assertEqual(args[:6], [111, 222, 5, byte_offset + (launch_first + first) * row_stride, count, row_stride])
            self.assertEqual(args[6:6 + count], blocks[launch_first + first:launch_first + first + count])
            self.assertTrue(all(v == 0 for v in args[6 + count:]), 'unused block slots are zero')
        args = ref.scatter_core_args(111, 222, 5, 4160, 557056, blocks, 0, (3, 4))
        self.assertEqual(args[:6], [111, 222, 5, 4160 + 3 * 557056, 4, 557056], 'a chipmajor-v1 shard apart: the core starts at its first row')
        lines = self.of('T')
        self.assertEqual(len(lines), len(CHECKSUM_ARG_CASES))
        for line, (byte_offset, first, last, slot, ranges, range_bytes, slot_stride) in zip(lines, CHECKSUM_ARG_CASES):
            cargs = ref.checksum_core_args(111, 222, 5, byte_offset, (first, last), slot, ranges, range_bytes, slot_stride)
            self.assertEqual(len(cargs), ref.CHECKSUM_ARGS)
            self.assertEqual([int(x) for x in line.split()], cargs)
            self.assertEqual(cargs, [111, 222, 5, byte_offset + first * 4096, (last - first) * 4096, ref.start_weight(first * 1024), slot, ranges, range_bytes,
                                     slot_stride])
        self.assertEqual(ref.checksum_core_args(111, 222, 5, 4160, (7, 12), 3, 1, 64 * 4096, 64),
                         [111, 222, 5, 4160 + 7 * 4096, 5 * 4096, ref.start_weight(7 * 1024), 3, 1, 64 * 4096, 64])
        self.assertEqual(ref.start_weight(7 * 1024), ref.checksum_core_args(111, 222, 5, 4160, (7, 12), 3, 60, 557056, 64)[5],
                         'the weight a core starts a range with is relative to the range start, the same in every range')


class StagingGeometryTests(unittest.TestCase):
    """The claim the staging placement rests on: in an interleaved buffer of banks x S bytes the pages of ONE bank are one contiguous range of S bytes."""

    def test_the_pages_of_one_bank_are_one_contiguous_range(self):
        for page in (64, 1088, 4096):
            for bank in (0, 3, 7):
                size = page * 40
                buffer = np.zeros(ref.BANKS * size, dtype=np.uint8)
                # what the allocator's interleaving does: page p on bank p % 8, at (p // 8) * page of that bank
                per_bank = [bytearray(size) for _ in range(ref.BANKS)]
                for p in range(buffer.size // page):
                    per_bank[p % ref.BANKS][(p // ref.BANKS) * page:(p // ref.BANKS + 1) * page] = bytes([p % 251]) * page
                for byte in range(0, size, page // 2 or 1):
                    logical = ref.staging_bank_position(byte, page, bank)
                    self.assertEqual(logical // page % ref.BANKS, bank)
                    self.assertEqual(per_bank[bank][byte], (logical // page) % 251, (page, bank, byte))

    def test_the_cost_is_banks_times_the_range(self):
        for size in (16 << 20, 128 << 20):
            self.assertEqual(ref.BANKS * size, size * 8)


class ChecksumArithmeticTests(unittest.TestCase):
    """The arithmetic block between the markers of the kernel, compiled on its own."""

    @classmethod
    def setUpClass(cls):
        if not have_compiler():
            raise unittest.SkipTest('no C++ compiler')
        text = read(CHECKSUM)
        begin = text.index('// BEGIN-CHECKSUM-ARITHMETIC')
        end = text.index('// END-CHECKSUM-ARITHMETIC')
        cls.block = text[begin:end]
        cls.directory = tempfile.mkdtemp(prefix='p2p_ck_')
        source = os.path.join(cls.directory, 'arith.cpp')
        with open(source, 'w') as handle:
            handle.write('''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define tt_l1_ptr
%s
// argv[1]: file of bytes, argv[2]: words per call (the kernel calls it once per chunk and carries w)
int main(int argc, char** argv) {
  (void)argc;
  FILE* f = std::fopen(argv[1], "rb"); std::vector<uint32_t> words(1 << 20);
  size_t n = std::fread(words.data(), 4, words.size(), f); std::fclose(f);
  const uint32_t step = (uint32_t)std::strtoul(argv[2], nullptr, 10);
  uint32_t w = 1; uint64_t sum = 0;
  for (size_t at = 0; at < n; at += step) sum = qwen_p2p_accumulate(words.data() + at, (uint32_t)(n - at < step ? n - at : step), w, sum);
  std::printf("%%llu %%u\\n", (unsigned long long)sum, w);
}
''' % cls.block)
        cls.exe = os.path.join(cls.directory, 'arith')
        done = subprocess.run([ref.compiler(), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', source, '-o', cls.exe], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def run_on(self, data, step):
        path = os.path.join(self.directory, 'in.bin')
        with open(path, 'wb') as handle:
            handle.write(data)
        out = subprocess.run([self.exe, path, str(step)], capture_output=True, text=True, check=True).stdout.split()
        return int(out[0]), int(out[1])

    def test_vectors_and_chunking_do_not_matter(self):
        for name, data, want in (('zero', ref.vector_zero(), ref.VECTOR_ZERO_4096), ('mod251', ref.vector_mod251(), ref.VECTOR_MOD251_4096),
                                 ('prng', ref.vector_prng(), ref.VECTOR_PRNG_1MIB)):
            for step in (1, 3, 4, 7, 1024, 8192, 1 << 20):
                with self.subTest(vector=name, step=step):
                    total, weight = self.run_on(data, step)
                    self.assertEqual(total, want)
                    self.assertEqual(weight, (len(data) // 4 * ref.STEP + 1) & ref.MASK32, 'the weight is carried to the next chunk')

    def test_the_kernel_text_has_the_contract_constants(self):
        self.assertIn('constexpr uint32_t kQwenP2PStep = 2654435761u;', self.block)
        self.assertIn('static_cast<uint64_t>(weight) * x0', self.block)


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class ScatterKernelEmulationTests(unittest.TestCase):
    """qwen_p2p_scatter.cpp, compiled and run on the CPU, against numpy. Transfers land at their barrier (late); the subclass below issues them at once."""

    eager = False

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_scatter_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def scenario(self, rng, page_bytes, pages_per_block, bank, byte_offset, blocks, slices, staging_address=4096, dst_address=None, ring=None, row_stride=None):
        slice_bytes = page_bytes * pages_per_block
        stride = slice_bytes if row_stride is None else row_stride
        needed = byte_offset + (len(blocks) - 1) * stride + slice_bytes
        ring = needed if ring is None else ring
        staging = rng.integers(0, 256, ring, dtype=np.uint8)       # (the bytes between two rows are random too: a read of a gap shows in the result)
        dst_pages = slices * pages_per_block
        dst_in = rng.integers(0, 256, (dst_pages, page_bytes), dtype=np.uint8)
        out, line = ref.run_scatter_emulation(self.directory, page_bytes, pages_per_block, bank, staging_address, byte_offset, staging.tobytes(), dst_pages,
                                              dst_address, blocks, dst_in, eager=self.eager, row_stride=row_stride)
        want = dst_in.copy()
        for i, block in enumerate(blocks):
            start = byte_offset + i * stride
            want[block * pages_per_block:(block + 1) * pages_per_block] = staging[start:start + slice_bytes].reshape(pages_per_block, page_bytes)
        return out, want, line, dst_in

    def check(self, *args, **keywords):
        rng = np.random.default_rng(keywords.pop('seed', 1))
        out, want, line, dst_in = self.scenario(rng, *args, **keywords)
        self.assertTrue(np.array_equal(out, want), 'the destination differs from the expected image: ' + line)
        self.assertIn('stray=0', line, 'nothing outside the destination was written')
        return line, dst_in

    def test_one_block_of_bf8_pages(self):
        line, _ = self.check(1088, 16, 0, 0, [5], 8)
        self.assertIn('reads=16 writes=16', line)

    def test_a_run_of_consecutive_blocks_crosses_pages_and_chunks(self):
        line, _ = self.check(1088, 16, 3, 4096, list(range(10, 30)), 40)
        self.assertIn('reads=320 writes=320', line)

    def test_scattered_and_shuffled_blocks(self):
        rng = np.random.default_rng(5)
        blocks = [int(b) for b in rng.permutation(200)[:57]]
        self.check(1088, 16, 7, 64, blocks, 200, seed=2)

    def test_byte_offset_at_the_end_of_the_ring(self):
        page, ppb, n = 1088, 16, 3
        slice_bytes = page * ppb
        ring = 1 << 20
        self.check(page, ppb, 1, ring - n * slice_bytes, [0, 7, 3], 8, ring=ring)

    def test_byte_offsets_that_are_64_but_not_1088_aligned(self):
        """A second tensor's data starts at the object start plus a multiple of 4096-aligned sizes: the offset is a multiple of 64 only."""
        for offset in (64, 192, 4096 + 64, 4160 + 3 * 17408):
            with self.subTest(offset=offset):
                self.check(1088, 16, 2, offset, [3, 2, 9], 12)

    def test_every_bank(self):
        for bank in range(8):
            with self.subTest(bank=bank):
                self.check(1088, 16, bank, 128, [1, 4, 5], 8, seed=bank)

    def test_other_page_sizes_and_block_shapes(self):
        for page, ppb in ((64, 1), (576, 3), (2048, 32), (1088, 1), (1088, 48), (4096, 5), (8192, 2)):
            with self.subTest(page=page, pages_per_block=ppb):
                self.check(page, ppb, 6, 0, [4, 1, 7, 6], 10, seed=page)

    def test_more_blocks_than_one_launch_and_than_one_core_takes(self):
        """17 blocks over 16 cores, 100 over 16, 513 and 600 take a second launch."""
        for n in (17, 100, 513, 600):
            with self.subTest(blocks=n):
                rng = np.random.default_rng(n)
                blocks = [int(b) for b in rng.permutation(n + 30)[:n]]
                line, _ = self.check(64, 2, 5, 0, blocks, n + 30, seed=n)
                self.assertIn('reads=%d writes=%d' % (2 * n, 2 * n), line)

    def test_blocks_not_named_are_not_touched(self):
        _, dst_in = self.check(1088, 16, 0, 0, [2], 6)
        self.assertEqual(dst_in.shape[0], 96)

    def test_a_destination_that_is_not_at_a_round_address(self):
        self.check(1088, 16, 2, 0, [0, 1], 4, dst_address=(1 << 20) + 64 * 3, staging_address=4096 + 64 * 5)

    def test_rows_apart_in_the_staging_bytes(self):
        """row_stride: row i of the call is read from byte_offset + i * row_stride; the bytes between two rows are never read."""
        for stride in (17408 + 64, 17408 + 4096, 17408 * 2, 17408 * 16 + 192):
            with self.subTest(row_stride=stride):
                line, _ = self.check(1088, 16, 3, 128, [9, 2, 30, 17, 4, 11], 40, row_stride=stride, seed=stride)
                self.assertIn('reads=96 writes=96', line, 'only the rows are read')

    def test_the_chipmajor_shard_stride_of_one_cache_tensor(self):
        """The host's chipmajor-v1 object: [cache][slice] per block, so one cache tensor's rows of consecutive blocks are 32 x 17408 = 557,056 bytes apart."""
        self.assertEqual(32 * 17408, 557056)
        line, _ = self.check(1088, 16, 5, 64 + 7 * 17408, [100, 101, 102, 7, 8, 55, 54, 53], 120, row_stride=557056, seed=3)
        self.assertIn('reads=128 writes=128', line)

    def test_strided_rows_of_many_blocks_cross_cores_and_launches(self):
        """17, 100, 513 and 600 blocks with a stride (a run on each of the 16 cores, then a second launch): the offsets of the later launches carry the stride."""
        for n in (17, 100, 513, 600):
            with self.subTest(blocks=n):
                rng = np.random.default_rng(n + 7)
                blocks = [int(b) for b in rng.permutation(n + 30)[:n]]
                line, _ = self.check(64, 2, 5, 192, blocks, n + 30, row_stride=128 + 64 * 3, seed=n)
                self.assertIn('reads=%d writes=%d' % (2 * n, 2 * n), line)

    def test_a_strided_range_that_ends_on_the_last_byte_of_the_ring(self):
        page, ppb, stride, n = 1088, 16, 557056, 3
        ring = 1 << 22
        offset = ring - ((n - 1) * stride + page * ppb)
        self.assertEqual(offset % 64, 0)
        self.check(page, ppb, 1, offset, [3, 0, 7], 8, ring=ring, row_stride=stride)

    def test_strided_rows_for_other_page_sizes_and_block_shapes(self):
        for page, ppb in ((64, 1), (576, 3), (2048, 32), (1088, 1), (1088, 48), (4096, 5), (8192, 2)):
            with self.subTest(page=page, pages_per_block=ppb):
                self.check(page, ppb, 6, 64, [4, 1, 7, 6], 10, row_stride=page * ppb + 192, seed=page)

    def test_a_stride_equal_to_the_block_is_the_contiguous_layout(self):
        rng = np.random.default_rng(21)
        out_a, want_a, _, _ = self.scenario(rng, 1088, 16, 2, 4160, [3, 5, 1], 8)
        rng = np.random.default_rng(21)
        out_b, want_b, _, _ = self.scenario(rng, 1088, 16, 2, 4160, [3, 5, 1], 8, row_stride=17408)
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertTrue(np.array_equal(want_a, want_b))

    def test_the_gemma_object_major_layout(self):
        """Gemma's staged object: ten rows of 69,632 bytes (64 pages of 1,088) per block, 696,320 bytes apart. One tensor's rows are read with row_stride 696,320 from
        slot base + tensor x 69,632; the other nine tensors' rows, in the gaps, are never read."""
        for tensor in (0, 4, 9):
            with self.subTest(tensor=tensor):
                line, _ = self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 3, GEMMA_CANARY + tensor * GEMMA_ROW, [9, 2, 30, 17, 4, 11, 20, 21, 22, 23, 1, 0], 40,
                                     row_stride=GEMMA_SHARD, seed=tensor)
                self.assertIn('reads=768 writes=768', line, 'only the rows are read')

    def test_a_full_gemma_slot_of_96_objects(self):
        rng = np.random.default_rng(91)
        blocks = [int(b) for b in rng.permutation(100)[:GEMMA_SLOT_OBJECTS]]
        line, _ = self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 6, GEMMA_CANARY + 7 * GEMMA_ROW, blocks, 100, row_stride=GEMMA_SHARD, seed=92)
        self.assertIn('reads=%d writes=%d' % (96 * 64, 96 * 64), line)

    def test_the_stand_in_refuses_a_misaligned_destination_page(self):
        """The emulation has a rule for the sub-page no-op: a page size that is not a multiple of 64 cannot be addressed whole."""
        rng = np.random.default_rng(3)
        with self.assertRaises(RuntimeError) as caught:
            self.scenario(rng, 100, 2, 0, 0, [1], 4)
        self.assertIn('EMU-VIOLATION', str(caught.exception))


class ScatterKernelEmulationEagerTests(ScatterKernelEmulationTests):
    """The same scenarios with every transfer performed the moment it is issued (the other extreme of the NOC's timing)."""

    eager = True


def mutate(source_path, directory, old, new, name):
    text = read(source_path)
    assert old in text, old
    path = os.path.join(directory, name)
    with open(path, 'w') as handle:
        handle.write(text.replace(old, new, 1))
    return path


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class ChecksumKernelEmulationTests(unittest.TestCase):
    eager = False

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_cksum_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_the_vectors_on_the_kernel(self):
        for name, data, want in (('zero', ref.vector_zero(), ref.VECTOR_ZERO_4096), ('mod251', ref.vector_mod251(), ref.VECTOR_MOD251_4096),
                                 ('prng', ref.vector_prng(), ref.VECTOR_PRNG_1MIB)):
            for cores in (8, 64, 80):
                with self.subTest(vector=name, cores=cores):
                    total, line = ref.run_checksum_emulation(self.directory, 4096, None, 0, 0, data, cores=cores, eager=self.eager)
                    self.assertEqual(total, want)

    def test_offsets_banks_and_lengths(self):
        rng = np.random.default_rng(11)
        for offset, bank, units, cores in ((0, 0, 1, 64), (64, 3, 2, 8), (4096, 7, 3, 64), (4160, 5, 33, 16), (0, 2, 64, 64), (192, 4, 65, 80),
                                           (8192, 1, 130, 24), (0, 6, 257, 64)):
            with self.subTest(offset=offset, bank=bank, units=units, cores=cores):
                data = rng.integers(0, 256, units * 4096, dtype=np.uint8).tobytes()
                total, _ = ref.run_checksum_emulation(self.directory, 8192, None, bank, offset, data, cores=cores, eager=self.eager)
                self.assertEqual(total, ref.checksum_numpy(data))

    def test_a_range_longer_than_a_chunk_per_core(self):
        data = ref.vector_prng()
        total, line = ref.run_checksum_emulation(self.directory, 4096, None, 0, 0, data, cores=8, eager=self.eager)
        self.assertEqual(total, ref.VECTOR_PRNG_1MIB)
        self.assertIn('reads=%d' % (8 * 4), line, '1 MiB over 8 cores is 128 KiB each: four 32 KiB chunks per core')

    def ranged(self, offset, bank, data, count, cores, partials_address=None):
        sums, line = ref.run_checksum_ranges_emulation(self.directory, 8192, partials_address, bank, offset, data, count, cores=cores, eager=self.eager)
        return sums, line

    def test_many_ranges_in_one_launch_each_with_k_from_its_own_start(self):
        """count ranges of `length` bytes, adjacent, one launch: one sum per range, equal to the reference of that range alone."""
        rng = np.random.default_rng(31)
        # (offset, bank, units per range, ranges, cores): idle cores (more cores than units), a core's run crossing a chunk, ranges not a multiple of anything
        for offset, bank, units, count, cores in ((0, 0, 1, 1, 64), (64, 3, 2, 5, 8), (4096, 7, 3, 17, 64), (4160, 5, 33, 4, 16), (0, 2, 136, 7, 64),
                                                  (192, 4, 65, 3, 80), (8192, 1, 9, 60, 24), (0, 6, 257, 2, 64), (64, 0, 1, 40, 80), (0, 5, 8, 9, 8)):
            with self.subTest(offset=offset, bank=bank, units=units, count=count, cores=cores):
                data = rng.integers(0, 256, units * 4096 * count, dtype=np.uint8).tobytes()
                sums, _ = self.ranged(offset, bank, data, count, cores)
                self.assertEqual(sums, ref.checksum_ranges(data, units * 4096, count))
                self.assertEqual(sums, ref.checksum_ranges(data, units * 4096, count, ref.checksum_python))

    def test_the_ranged_vectors(self):
        """4 ranges, each the same as one of the single-range vectors: 4 x 4096 bytes, and 4 x 1 MiB."""
        for name, data, count, want in (('small', ref.vector_ranged_small(), 4, ref.RANGED_SMALL_EXPECTED), ('mib', ref.vector_ranged_prng(), 4, ref.RANGED_PRNG_EXPECTED)):
            for cores in (8, 64, 80):
                for offset in (0, 4160):
                    with self.subTest(vector=name, cores=cores, offset=offset):
                        sums, _ = self.ranged(offset, 3, data, count, cores)
                        self.assertEqual(sums, want)

    def test_one_range_through_the_ranged_launch_is_the_single_range_result(self):
        for name, data, want in (('zero', ref.vector_zero(), ref.VECTOR_ZERO_4096), ('mod251', ref.vector_mod251(), ref.VECTOR_MOD251_4096),
                                 ('prng', ref.vector_prng(), ref.VECTOR_PRNG_1MIB)):
            with self.subTest(vector=name):
                sums, _ = self.ranged(0, 0, data, 1, 64)
                self.assertEqual(sums, [want])

    def test_ranges_are_read_in_one_stream(self):
        """4 ranges of 1 MiB on 8 cores: each core takes 128 KiB of every range, four 32 KiB chunks, so 16 reads and 4 partial pages a core."""
        sums, line = self.ranged(0, 0, ref.vector_ranged_prng(), 4, 8)
        self.assertEqual(sums, ref.RANGED_PRNG_EXPECTED)
        self.assertIn('reads=%d writes=%d' % (8 * 16, 8 * 4), line)

    def test_a_changed_byte_changes_only_its_range(self):
        data = bytearray(ref.vector_ranged_prng())
        want = ref.RANGED_PRNG_EXPECTED
        data[(1 << 20) * 1 + 12345] ^= 0x40                  # a byte of range 1 (zeros): its sum, and only its sum, changes
        sums, _ = self.ranged(0, 2, bytes(data), 4, 64)
        self.assertEqual([a != b for a, b in zip(sums, want)], [False, True, False, False])
        self.assertEqual(sums[1], ref.checksum_numpy(bytes(data[1 << 20:2 << 20])))

    def test_the_sum_is_over_the_range_not_the_ring(self):
        data = ref.vector_prng()[:16384]
        a, _ = ref.run_checksum_emulation(self.directory, 4096, None, 0, 4096, data, cores=8, eager=self.eager)
        b, _ = ref.run_checksum_emulation(self.directory, 4096, None, 0, 12288, data, cores=8, eager=self.eager)
        self.assertEqual(a, b)
        self.assertEqual(a, ref.checksum_numpy(data))


class ChecksumKernelEmulationEagerTests(ChecksumKernelEmulationTests):
    eager = True


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class EmulationHasTeethTests(unittest.TestCase):
    """The emulation is only evidence if a broken kernel fails it. Each mutant below is a plausible slip; each must fail under at least one of the two NOC
    schedules (late: transfers land at their barrier; eager: at once), and the unbroken kernel must pass under both."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_teeth_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def build_mutant(self, kind, source_path, old, new, eager):
        path = mutate(source_path, self.directory, old, new, 'mutant_%s_%d.cpp' % (kind, abs(hash((old, new)))))
        compile_args = ([1088, 16, 16, 32, 0, 0, 0, 0] if kind == 'scatter' else [32768, 64, 0, 0, 0, 0])
        return ref.build_emulation(kind, compile_args, self.directory, eager=eager, kernel_file=path, strict=False)

    def run_scatter(self, exe, blocks=(3, 9, 1, 12, 6, 0, 20, 17, 8, 5, 30, 31, 2, 4, 10, 11, 13, 14, 15, 16, 18, 19, 21, 22, 23), row_stride=17408 + 192):
        rng = np.random.default_rng(9)
        page, ppb, slices = 1088, 16, 40
        slice_bytes = page * ppb
        stride = slice_bytes if row_stride is None else row_stride
        staging = rng.integers(0, 256, (len(blocks) - 1) * stride + slice_bytes, dtype=np.uint8)
        dst_in = rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8)
        d = self.directory
        scenario = os.path.join(d, 'm.txt')
        ref._write_scenario(scenario, [('page_bytes', page), ('pages_per_block', ppb), ('bank', 2), ('staging_address', 4096), ('byte_offset', 0), ('dst_address', 1 << 20),
                                       ('dst_pages', slices * ppb), ('bank_bytes', 1 << 22), ('blocks', list(blocks))] + ([] if row_stride is None else [('row_stride', row_stride)]))
        for name, data in (('s.bin', staging), ('i.bin', dst_in)):
            with open(os.path.join(d, name), 'wb') as handle:
                handle.write(data.tobytes())
        done = subprocess.run([exe, scenario, os.path.join(d, 's.bin'), os.path.join(d, 'i.bin'), os.path.join(d, 'o.bin')], capture_output=True, text=True)
        want = dst_in.copy()
        for i, block in enumerate(blocks):
            want[block * ppb:(block + 1) * ppb] = staging[i * stride:i * stride + slice_bytes].reshape(ppb, page)
        if done.returncode:
            return 'violation', done.stdout
        with open(os.path.join(d, 'o.bin'), 'rb') as handle:
            out = np.frombuffer(handle.read(), dtype=np.uint8).reshape(slices * ppb, page)
        return ('ok' if np.array_equal(out, want) and 'stray=0' in done.stdout else 'wrong'), done.stdout

    def run_checksum(self, exe, cores=8, units=130, ranges=1):
        data = ref.vector_prng()[:units * 4096 * ranges]
        d = self.directory
        scenario = os.path.join(d, 'c.txt')
        ref._write_scenario(scenario, [('staging_address', 4096), ('partials_address', 1 << 20), ('bank', 1), ('byte_offset', 64), ('length', units * 4096), ('cores', cores),
                                       ('bank_bytes', 1 << 21), ('ranges', ranges)])
        with open(os.path.join(d, 'c.bin'), 'wb') as handle:
            handle.write(data)
        done = subprocess.run([exe, scenario, os.path.join(d, 'c.bin')], capture_output=True, text=True)
        if done.returncode:
            return 'violation', done.stdout
        totals = [int(word) for word in [line for line in done.stdout.splitlines() if line.startswith('SUMS ')][0].split()[1:]]
        return ('ok' if totals == ref.checksum_ranges(data, units * 4096, ranges) else 'wrong'), done.stdout

    def verdicts(self, kind, source, old, new, runner):
        return [runner(self.build_mutant(kind, source, old, new, eager))[0] for eager in (False, True)]

    def test_the_unbroken_kernels_pass_the_same_checks_under_both_schedules(self):
        self.assertEqual(self.verdicts('scatter', SCATTER, 'half ^= 1u;', 'half ^= 1u;', self.run_scatter), ['ok', 'ok'])
        self.assertEqual(self.verdicts('scatter', SCATTER, 'half ^= 1u;', 'half ^= 1u;', lambda exe: self.run_scatter(exe, row_stride=None)), ['ok', 'ok'])
        self.assertEqual(self.verdicts('checksum', CHECKSUM, 'const uint32_t half = i & 1u;', 'const uint32_t half = i & 1u;', self.run_checksum), ['ok', 'ok'])
        self.assertEqual(self.verdicts('checksum', CHECKSUM, 'const uint32_t half = i & 1u;', 'const uint32_t half = i & 1u;', lambda exe: self.run_checksum(exe, cores=8, units=3, ranges=5)), ['ok', 'ok'])
        self.assertEqual(self.verdicts('checksum', CHECKSUM, 'const uint32_t half = i & 1u;', 'const uint32_t half = i & 1u;', lambda exe: self.run_checksum(exe, cores=64, units=2, ranges=7)), ['ok', 'ok'])

    def test_scatter_mutants_are_caught(self):
        mutants = {
            'reads into the half the pending writes still use': ('const uint32_t other = scratch + (half ^ 1u) * chunk_pages * page_bytes;',
                                                                  'const uint32_t other = scratch + half * chunk_pages * page_bytes;'),
            'the write barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                             '        noc_async_read_barrier();\n        if (next >= total) {'),
            'the read barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                            '        noc_async_write_barrier();\n        if (next >= total) {'),
            'a destination page off by one': ('block * pages_per_block + page % pages_per_block', 'block * pages_per_block + page % pages_per_block + 1'),
            'the staging offset ignores the core start': ('qwen_p2p_source(source_offset, row_stride, k,', 'qwen_p2p_source(0, row_stride, k,'),
            'a half page is written': ('destination.get_noc_addr(destination_page), page_bytes);', 'destination.get_noc_addr(destination_page), page_bytes / 2);'),
            'the second chunk reads from the wrong source': ('qwen_p2p_source(source_offset, row_stride, next + k,', 'qwen_p2p_source(source_offset, row_stride, next + k + 1,'),
            'the block table is indexed by the page, not the block': ('get_arg_val<uint32_t>(block_ids + page / pages_per_block)', 'get_arg_val<uint32_t>(block_ids + page % 32)'),
            'the block table starts one word early (the stride word is read as a block id)': ('constexpr uint32_t block_ids = 6;', 'constexpr uint32_t block_ids = 5;'),
            # the row stride
            'the stride is ignored (rows are read back to back)': ('(page / pages_per_block) * row_stride', '(page / pages_per_block) * pages_per_block * page_bytes'),
            'the stride is applied to every page, not every block': ('(page / pages_per_block) * row_stride', 'page * row_stride'),
            'the stride is one alignment unit short': ('(page / pages_per_block) * row_stride', '(page / pages_per_block) * (row_stride - 64)'),
            'the page inside the row is dropped (every page of a block reads the first)': ('+ (page % pages_per_block) * page_bytes;', '+ 0 * page_bytes;'),
            'the stride is read from the wrong argument': ('const uint32_t row_stride = get_arg_val<uint32_t>(5);', 'const uint32_t row_stride = get_arg_val<uint32_t>(3);'),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                verdicts = self.verdicts('scatter', SCATTER, old, new, self.run_scatter)
                self.assertNotEqual(set(verdicts), {'ok'}, 'the emulation accepted: ' + name)

    def test_checksum_mutants_are_caught(self):
        mutants = {
            'the step constant is wrong': ('constexpr uint32_t kQwenP2PStep = 2654435761u;', 'constexpr uint32_t kQwenP2PStep = 2654435769u;'),
            'the weight restarts for every chunk': ('weight, sum);', '(weight = get_arg_val<uint32_t>(5)), sum);'),
            'the product is truncated to 32 bits': ('sum += static_cast<uint64_t>(weight) * x0;', 'sum += static_cast<uint32_t>(weight * x0);'),
            'the last word of a chunk is dropped': ('count >> 2, weight, sum);', '(count >> 2) - 1, weight, sum);'),
            'the next chunk is read into the half being summed': ('next_range, next_within, scratch + (half ^ 1u) * chunk_bytes);',
                                                                   'next_range, next_within, scratch + half * chunk_bytes);'),
            'the partial is added at the wrong slot': ('slot + range * slot_stride, out, out_page, out_page_bytes, sum);',
                                                       'slot + range * slot_stride + 1, out, out_page, out_page_bytes, sum);'),
            'the high word is not stored': ('out[1] = static_cast<uint32_t>(sum >> 32);', 'out[1] = 0;'),
            'the weight of the first word is wrong': ('const uint32_t weight_start = get_arg_val<uint32_t>(5);', 'const uint32_t weight_start = get_arg_val<uint32_t>(5) + 1;'),
            'a chunk is not awaited': ('        const uint32_t half = i & 1u;\n        noc_async_read_barrier();', '        const uint32_t half = i & 1u;'),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                verdicts = self.verdicts('checksum', CHECKSUM, old, new, self.run_checksum)
                self.assertNotEqual(set(verdicts), {'ok'}, 'the emulation accepted: ' + name)

    def test_many_ranges_mutants_are_caught(self):
        """The slips that only a launch of several ranges can make; each runs the ranged scenarios (and an idle-core one: more cores than units)."""
        mutants = {
            'the weight is not restarted for the next range': ('            weight = weight_start;\n', ''),
            'the sum is not restarted for the next range': ('            sum = 0;\n            weight = weight_start;', '            weight = weight_start;'),
            'the next range is read from the wrong place': ('source_offset + range * range_stride + done', 'source_offset + range * bytes + done'),
            'a range is read from the previous range': ('source_offset + range * range_stride + done', 'source_offset + (range ? range - 1 : 0) * range_stride + done'),
            'a range writes the previous range\'s slot': ('slot + range * slot_stride, out, out_page, out_page_bytes, sum);',
                                                          'slot + (range ? range - 1 : 0) * slot_stride, out, out_page, out_page_bytes, sum);'),
            'the slot stride is ignored': ('slot + range * slot_stride, out, out_page, out_page_bytes, sum);', 'slot + range, out, out_page, out_page_bytes, sum);'),
            'the last range is dropped': ('const uint32_t total = ranges * chunks;', 'const uint32_t total = (ranges - 1) * chunks;'),
            'an idle core writes no partials': ('if (bytes == 0) {\n        for (uint32_t range = 0; range < ranges; ++range) {',
                                                'if (bytes == 0) {\n        for (uint32_t range = 0; range < 0; ++range) {'),
            'an idle core writes only the first range': ('qwen_p2p_put_partial(partials, slot + range * slot_stride, out, out_page, out_page_bytes, 0);',
                                                         'qwen_p2p_put_partial(partials, slot, out, out_page, out_page_bytes, 0);'),
            'a partial is not awaited before the page is reused': ('    noc_async_write(out_page, partials.get_noc_addr(page), out_page_bytes);\n    noc_async_write_barrier();',
                                                                   '    noc_async_write(out_page, partials.get_noc_addr(page), out_page_bytes);'),
            'the range stride is read from the wrong argument': ('const uint32_t range_stride = get_arg_val<uint32_t>(8);', 'const uint32_t range_stride = get_arg_val<uint32_t>(4);'),
            'the number of ranges is read from the wrong argument': ('const uint32_t ranges = get_arg_val<uint32_t>(7);', 'const uint32_t ranges = get_arg_val<uint32_t>(9);'),
        }
        runners = (lambda exe: self.run_checksum(exe, cores=8, units=5, ranges=4), lambda exe: self.run_checksum(exe, cores=64, units=2, ranges=7),
                   lambda exe: self.run_checksum(exe, cores=8, units=130, ranges=3))
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                caught = False
                for eager in (False, True):
                    exe = self.build_mutant('checksum', CHECKSUM, old, new, eager)
                    if any(runner(exe)[0] != 'ok' for runner in runners):
                        caught = True
                self.assertTrue(caught, 'the emulation accepted: ' + name)


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class GatherKernelEmulationTests(unittest.TestCase):
    """qwen_p2p_gather.cpp, compiled and run on the CPU, against numpy. Transfers land at their barrier (late); the subclass below issues them at once.
    The staging range holds NOISE before the call: every byte outside the rows (the gaps between two rows, the rest of the ring) must come back unchanged, and the
    harness lets the kernel read only the pages of the blocks it was asked for and write only the pages of the rows."""

    eager = False

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_gather_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def gather(self, rng, page, ppb, bank, byte_offset, blocks, slices, staging_address=4096, ring=None, row_stride=None, tensor=None, staging_in=None):
        slice_bytes = page * ppb
        stride = slice_bytes if row_stride is None else row_stride
        if ring is None:
            ring = byte_offset + (len(blocks) - 1) * stride + slice_bytes
        if tensor is None:
            tensor = rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8)
        if staging_in is None:
            staging_in = rng.integers(0, 256, ring, dtype=np.uint8)
        out, line = ref.run_gather_emulation(self.directory, page, ppb, bank, staging_address, byte_offset, staging_in.tobytes(), slices * ppb, None, blocks, tensor,
                                             eager=self.eager, row_stride=row_stride)
        want = staging_in.copy()
        for i, block in enumerate(blocks):
            want[byte_offset + i * stride:byte_offset + i * stride + slice_bytes] = tensor[block * ppb:(block + 1) * ppb].reshape(-1)
        return np.frombuffer(out, dtype=np.uint8), want, line, tensor

    def check(self, *args, seed=1, **keywords):
        rng = np.random.default_rng(seed)
        out, want, line, _ = self.gather(rng, *args, **keywords)
        self.assertTrue(np.array_equal(out, want), 'the staging differs from the expected image: ' + line)
        self.assertIn('stray=0', line, 'nothing outside the rows was written')
        return line

    def test_one_gemma_row(self):
        line = self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 0, 0, [5], 8)
        self.assertIn('reads=64 writes=64', line)

    def test_a_run_of_consecutive_blocks_crosses_pages_and_chunks(self):
        line = self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 3, 4096, list(range(10, 30)), 40)
        self.assertIn('reads=1280 writes=1280', line)

    def test_scattered_and_shuffled_blocks(self):
        rng = np.random.default_rng(5)
        blocks = [int(b) for b in rng.permutation(60)[:37]]
        self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 7, 64, blocks, 60, seed=2)

    def test_repeated_block_ids_are_fine(self):
        """A gather only reads: the same block twice gives two equal rows (the scatter refuses repeats, this does not)."""
        self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 1, 0, [4, 4, 2, 4], 6)

    def test_every_bank(self):
        for bank in range(8):
            with self.subTest(bank=bank):
                self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, bank, 128, [1, 4, 5], 8, seed=bank)

    def test_the_gemma_object_major_layout_one_tensor_row_at_a_time(self):
        """The slot of the design: object i of the slot holds, for each tensor t, the block's row at slot base + i x 696,320 + t x 69,632. One call per tensor
        (row_stride 696,320, byte_offset = slot base + t x 69,632); the bytes of the other tensors' rows, in the gaps, are never touched."""
        rng = np.random.default_rng(31)
        objects, slices = 12, 40
        base = GEMMA_CANARY
        blocks = [int(b) for b in rng.permutation(slices)[:objects]]
        ring = base + objects * GEMMA_SHARD
        staging = rng.integers(0, 256, ring, dtype=np.uint8)
        tensors = [rng.integers(0, 256, (slices * GEMMA_PAGES_PER_BLOCK, GEMMA_PAGE), dtype=np.uint8) for _ in range(GEMMA_TENSORS)]
        want = staging.copy()
        for t in range(GEMMA_TENSORS):
            out, want_t, line, _ = self.gather(rng, GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 3, base + t * GEMMA_ROW, blocks, slices, ring=ring, row_stride=GEMMA_SHARD,
                                               tensor=tensors[t], staging_in=staging)
            self.assertTrue(np.array_equal(out, want_t), 'tensor %d: %s' % (t, line))
            self.assertIn('stray=0', line)
            self.assertIn('reads=%d writes=%d' % (objects * 64, objects * 64), line)
            staging = out.copy()
        for i, block in enumerate(blocks):
            rows = [tensors[t][block * 64:(block + 1) * 64] for t in range(GEMMA_TENSORS)]
            shard = bytes(staging[base + i * GEMMA_SHARD:base + (i + 1) * GEMMA_SHARD])
            self.assertEqual(shard, b''.join(r.tobytes() for r in rows), 'object %d is [tensor 0 row | tensor 1 row | ...]' % i)
            # the layout is the digest's shard: the checksum of the staged object IS the digest of the rows it came from
            self.assertEqual(ref.checksum_numpy(shard), ref.digest_shard(rows))
        self.assertTrue(np.array_equal(staging[:base], want[:base]), 'the canary page before the slot is untouched')

    def test_a_full_slot_of_96_objects_of_one_tensor(self):
        rng = np.random.default_rng(32)
        slices = 100
        blocks = [int(b) for b in rng.permutation(slices)[:GEMMA_SLOT_OBJECTS]]
        ring = GEMMA_CANARY + GEMMA_SLOT_OBJECTS * GEMMA_SHARD
        line = self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 5, GEMMA_CANARY + 4 * GEMMA_ROW, blocks, slices, ring=ring, row_stride=GEMMA_SHARD, seed=33)
        self.assertIn('reads=%d writes=%d' % (96 * 64, 96 * 64), line)

    def test_byte_offset_at_the_end_of_the_bank_range(self):
        page, ppb, n = GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 3
        ring = 4 << 20
        offset = ring - ((n - 1) * GEMMA_SHARD + GEMMA_ROW)
        self.assertEqual(offset % 64, 0)
        self.check(page, ppb, 1, offset, [0, 7, 3], 8, ring=ring, row_stride=GEMMA_SHARD)

    def test_byte_offsets_that_are_64_but_not_1088_aligned(self):
        for offset in (64, 192, 4096 + 64, 4160 + 3 * 17408):
            with self.subTest(offset=offset):
                self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, 2, offset, [3, 2, 9], 12)

    def test_other_page_sizes_and_block_shapes(self):
        for page, ppb in ((64, 1), (576, 3), (2048, 32), (1088, 1), (1088, 48), (4096, 5), (8192, 2), (1088, 20), (1088, 33)):
            with self.subTest(page=page, pages_per_block=ppb):
                self.check(page, ppb, 6, 0, [4, 1, 7, 6], 10, seed=page + ppb)

    def test_more_blocks_than_one_core_takes_up_to_a_full_launch(self):
        """17 blocks over 16 cores, 100 over 16, 512 = 16 cores x 32 blocks (the most one call may carry)."""
        for n in (17, 100, 512):
            with self.subTest(blocks=n):
                rng = np.random.default_rng(n)
                blocks = [int(b) for b in rng.permutation(n + 30)[:n]]
                line = self.check(64, 2, 5, 0, blocks, n + 30, seed=n)
                self.assertIn('reads=%d writes=%d' % (2 * n, 2 * n), line)

    def test_strided_rows_for_other_page_sizes_and_block_shapes(self):
        for page, ppb in ((64, 1), (576, 3), (2048, 32), (1088, 1), (1088, 48), (4096, 5), (8192, 2), (1088, 20)):
            with self.subTest(page=page, pages_per_block=ppb):
                self.check(page, ppb, 6, 64, [4, 1, 7, 6], 10, row_stride=page * ppb + 192, seed=page)

    def test_a_stride_equal_to_the_block_is_the_contiguous_layout(self):
        rng = np.random.default_rng(21)
        out_a, want_a, _, _ = self.gather(rng, 1088, 16, 2, 4160, [3, 5, 1], 8)
        rng = np.random.default_rng(21)
        out_b, want_b, _, _ = self.gather(rng, 1088, 16, 2, 4160, [3, 5, 1], 8, row_stride=17408)
        self.assertTrue(np.array_equal(out_a, out_b) and np.array_equal(want_a, want_b))

    def test_the_stand_in_refuses_a_misaligned_page(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(RuntimeError) as caught:
            self.gather(rng, 100, 2, 0, 0, [1], 4)
        self.assertIn('EMU-VIOLATION', str(caught.exception))

    def test_the_scatter_undoes_the_gather(self):
        """gather (tensor A, blocks) -> staging -> scatter (into tensor B at other blocks): B's rows are A's, the rest of B untouched. Gemma geometry, 6 objects."""
        rng = np.random.default_rng(41)
        slices, n = 40, 6
        src = [int(b) for b in rng.permutation(slices)[:n]]
        dst = [int(b) for b in rng.permutation(slices)[:n]]
        ring = GEMMA_CANARY + n * GEMMA_SHARD
        tensor_a = rng.integers(0, 256, (slices * 64, GEMMA_PAGE), dtype=np.uint8)
        tensor_b = rng.integers(0, 256, (slices * 64, GEMMA_PAGE), dtype=np.uint8)
        staging = rng.integers(0, 256, ring, dtype=np.uint8)
        out, _, line, _ = self.gather(rng, GEMMA_PAGE, 64, 4, GEMMA_CANARY + 2 * GEMMA_ROW, src, slices, ring=ring, row_stride=GEMMA_SHARD, tensor=tensor_a, staging_in=staging)
        back, line = ref.run_scatter_emulation(self.directory, GEMMA_PAGE, 64, 4, 4096, GEMMA_CANARY + 2 * GEMMA_ROW, out.tobytes(), slices * 64, None, dst, tensor_b,
                                               eager=self.eager, row_stride=GEMMA_SHARD)
        want = tensor_b.copy()
        for s, d in zip(src, dst):
            want[d * 64:(d + 1) * 64] = tensor_a[s * 64:(s + 1) * 64]
        self.assertTrue(np.array_equal(back, want), line)
        self.assertIn('stray=0', line)


class GatherKernelEmulationEagerTests(GatherKernelEmulationTests):
    """The same scenarios with every transfer performed the moment it is issued (the other extreme of the NOC's timing)."""

    eager = True


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class DigestKernelEmulationTests(unittest.TestCase):
    """qwen_p2p_digest_rows.cpp, compiled and run on the CPU, against the numpy reference (the checksum over each block's shard). The harness lets the kernel read only
    the pages of the blocks it was asked for, in every tensor, and write only the partial pages of those blocks."""

    eager = False

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_digest_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def tensors(self, rng, page, ppb, count, slices):
        return [rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8) for _ in range(count)]

    def digest(self, tensors, page, ppb, blocks):
        return ref.run_digest_emulation(self.directory, page, ppb, tensors, blocks, eager=self.eager)

    def check(self, page, ppb, count, blocks, slices, seed=1, tensors=None):
        rng = np.random.default_rng(seed)
        tensors = self.tensors(rng, page, ppb, count, slices) if tensors is None else tensors
        got, line = self.digest(tensors, page, ppb, blocks)
        self.assertEqual(got, ref.digest_blocks(tensors, blocks, ppb), line)
        self.assertIn('reads=%d writes=%d stray=0' % (count * len(blocks) * ppb, len(blocks)), line, 'every page of every row is read once, one partial page per block is written')
        return got

    def test_one_gemma_block_ten_tensors(self):
        self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, GEMMA_TENSORS, [3], 8)

    def test_a_gemma_slot_one_block_per_core_and_two_on_some(self):
        """80 blocks (one per core), 81 (the first core takes two), 96 (the first 16 cores take two): the split of the plan header."""
        for n in (80, 81, 96):
            with self.subTest(blocks=n):
                rng = np.random.default_rng(n)
                blocks = [int(b) for b in rng.permutation(n + 4)[:n]]
                self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, GEMMA_TENSORS, blocks, n + 4, seed=n)

    def test_the_largest_call_128_blocks_16_tensors_of_small_rows(self):
        rng = np.random.default_rng(7)
        blocks = [int(b) for b in rng.permutation(130)[:128]]
        self.check(1088, 4, 16, blocks, 130, seed=7)

    def test_128_gemma_blocks_ten_tensors(self):
        rng = np.random.default_rng(8)
        blocks = [int(b) for b in rng.permutation(128)]
        self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, GEMMA_TENSORS, blocks, 128, seed=8)

    def test_any_number_of_tensors(self):
        for count in (1, 2, 3, 16):
            with self.subTest(tensors=count):
                self.check(GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, count, [5, 2, 0, 7, 6], 8, seed=count)

    def test_other_page_sizes_and_block_shapes(self):
        """Rows that are not a whole number of the 8-page chunks (20 = 8 + 8 + 4, 9, 3, 1) and other pages."""
        for page, ppb in ((64, 1), (576, 3), (1088, 20), (1088, 9), (2048, 32), (4096, 5), (8192, 2), (1088, 8), (1088, 16), (1088, 17)):
            with self.subTest(page=page, pages_per_block=ppb):
                self.check(page, ppb, 3, [4, 1, 7, 6, 0], 10, seed=page + ppb)

    def test_a_block_repeated_in_a_call_is_digested_at_every_position(self):
        got = self.check(GEMMA_PAGE, 8, 3, [4, 4, 2, 4], 6, seed=11)
        self.assertEqual(got[0], got[1])
        self.assertEqual(got[0], got[3])
        self.assertNotEqual(got[0], got[2])

    def test_a_digest_does_not_depend_on_the_other_blocks_of_the_call(self):
        rng = np.random.default_rng(12)
        tensors = self.tensors(rng, GEMMA_PAGE, 8, 3, 20)
        alone, _ = self.digest(tensors, GEMMA_PAGE, 8, [9])
        among, _ = self.digest(tensors, GEMMA_PAGE, 8, [1, 2, 3, 9, 4])
        self.assertEqual(alone[0], among[3])

    def test_all_zero_rows_digest_to_zero(self):
        zeros = [np.zeros((8 * 4, GEMMA_PAGE), dtype=np.uint8) for _ in range(3)]
        got, _ = self.digest(zeros, GEMMA_PAGE, 4, [0, 3])
        self.assertEqual(got, [0, 0])

    def test_the_tensor_order_is_part_of_the_digest(self):
        rng = np.random.default_rng(13)
        tensors = self.tensors(rng, GEMMA_PAGE, 8, 3, 6)
        forward, _ = self.digest(tensors, GEMMA_PAGE, 8, [2, 5])
        swapped, _ = self.digest([tensors[1], tensors[0], tensors[2]], GEMMA_PAGE, 8, [2, 5])
        self.assertNotEqual(forward[0], swapped[0])
        self.assertEqual(swapped, ref.digest_blocks([tensors[1], tensors[0], tensors[2]], [2, 5], 8))

    def test_k_continues_from_one_tensor_to_the_next(self):
        """The digest of T tensors is the sum of the checksum of tensor 0's row and the partial sums of the other rows, each starting at its own word of the shard:
        sum_t checksum_with_offset(row_t, t x row_words). A kernel that restarted k at each tensor fails this."""
        rng = np.random.default_rng(14)
        ppb, count = 4, 3
        tensors = self.tensors(rng, GEMMA_PAGE, ppb, count, 6)
        got, _ = self.digest(tensors, GEMMA_PAGE, ppb, [1, 4])
        row_words = ppb * GEMMA_PAGE // 4
        for position, block in enumerate([1, 4]):
            parts = [ref.checksum_with_offset(tensors[t][block * ppb:(block + 1) * ppb].tobytes(), ref.digest_words_offset(row_words, t)) for t in range(count)]
            self.assertEqual(got[position], sum(parts) & ref.MASK64)
        # and it is not the sum of independent per-tensor checksums
        independent = sum(ref.checksum_numpy(tensors[t][1 * ppb:2 * ppb].tobytes()) for t in range(count)) & ref.MASK64
        self.assertNotEqual(got[0], independent)

    def test_the_digest_is_linear_in_the_data(self):
        """S(a + b) = S(a) + S(b) mod 2^64 for words that add without wrapping: three runs of the kernel, A, B and A + B."""
        rng = np.random.default_rng(15)
        pages, page, ppb, count = 6 * 8, GEMMA_PAGE, 8, 2
        words_a = [rng.integers(0, 1 << 31, (pages, page // 4), dtype=np.uint32) for _ in range(count)]
        words_b = [rng.integers(0, 1 << 31, (pages, page // 4), dtype=np.uint32) for _ in range(count)]
        as_bytes = lambda arrays: [a.view(np.uint8).reshape(pages, page) for a in arrays]
        blocks = [0, 3, 5]
        da, _ = self.digest(as_bytes(words_a), page, ppb, blocks)
        db, _ = self.digest(as_bytes(words_b), page, ppb, blocks)
        dab, line = self.digest(as_bytes([a + b for a, b in zip(words_a, words_b)]), page, ppb, blocks)
        self.assertEqual(dab, [(x + y) & ref.MASK64 for x, y in zip(da, db)], line)
        self.assertEqual(da, ref.digest_blocks(as_bytes(words_a), blocks, ppb))
        # and scaling by 3 (no wrap)
        d3, _ = self.digest(as_bytes([w // 3 * 3 for w in words_a]), page, ppb, blocks)
        d1, _ = self.digest(as_bytes([w // 3 for w in words_a]), page, ppb, blocks)
        self.assertEqual(d3, [(3 * x) & ref.MASK64 for x in d1])

    def test_one_flipped_bit_changes_only_its_block_and_swapped_rows_change_both(self):
        rng = np.random.default_rng(16)
        tensors = self.tensors(rng, GEMMA_PAGE, 8, 3, 8)
        blocks = [1, 2, 3, 4]
        before, _ = self.digest(tensors, GEMMA_PAGE, 8, blocks)
        flipped = [t.copy() for t in tensors]
        flipped[2][3 * 8 + 5, 700] ^= 0x04                  # tensor 2, block 3, page 5, byte 700
        after, _ = self.digest(flipped, GEMMA_PAGE, 8, blocks)
        self.assertEqual([a != b for a, b in zip(before, after)], [False, False, True, False])
        swapped = [t.copy() for t in tensors]
        swapped[0][1 * 8:2 * 8], swapped[0][4 * 8:5 * 8] = tensors[0][4 * 8:5 * 8].copy(), tensors[0][1 * 8:2 * 8].copy()   # tensor 0: blocks 1 and 4 trade rows
        moved, _ = self.digest(swapped, GEMMA_PAGE, 8, blocks)
        self.assertEqual([a != b for a, b in zip(before, moved)], [True, False, False, True])

    def test_the_digest_of_a_gathered_object_is_the_checksum_of_its_staged_bytes(self):
        """The spill's invariant: the digest of the SOURCE rows (this kernel) equals the checksum of the object the gather laid out in staging (the oracle kernel)."""
        rng = np.random.default_rng(17)
        slices, objects, count = 16, 5, GEMMA_TENSORS
        tensors = self.tensors(rng, GEMMA_PAGE, GEMMA_PAGES_PER_BLOCK, count, slices)
        blocks = [int(b) for b in rng.permutation(slices)[:objects]]
        ring = GEMMA_CANARY + objects * GEMMA_SHARD
        staging = rng.integers(0, 256, ring, dtype=np.uint8).tobytes()
        for t in range(count):
            staging, line = ref.run_gather_emulation(self.directory, GEMMA_PAGE, 64, 1, 4096, GEMMA_CANARY + t * GEMMA_ROW, staging, slices * 64, None, blocks, tensors[t],
                                                     eager=self.eager, row_stride=GEMMA_SHARD)
        digests, _ = self.digest(tensors, GEMMA_PAGE, 64, blocks)
        for i in range(objects):
            obj = staging[GEMMA_CANARY + i * GEMMA_SHARD:GEMMA_CANARY + (i + 1) * GEMMA_SHARD]
            self.assertEqual(ref.checksum_numpy(obj), digests[i], i)
        # the oracle KERNEL (not just the numpy reference) agrees: one launch over the objects with count ranges
        sums, _ = ref.run_checksum_ranges_emulation(self.directory, 4096, None, 1, GEMMA_CANARY, staging[GEMMA_CANARY:], objects, cores=64, eager=self.eager)
        self.assertEqual(sums, digests)


class DigestKernelEmulationEagerTests(DigestKernelEmulationTests):
    eager = True


@unittest.skipUnless(have_compiler(), 'no C++ compiler')
class GatherDigestTeethTests(unittest.TestCase):
    """The emulation of the gather and of the digest is only evidence if a broken kernel fails it. Each mutant below is a plausible slip; each must fail under at
    least one of the two NOC schedules (late: transfers land at their barrier; eager: at once), and the unbroken kernels must pass under both."""

    GATHER_ARGS = [1088, 16, 16, 32, 0, 0, 0, 0]
    DIGEST_ARGS = [1088, 20, 8, 2, 16, 0, 0, 0, 0]      # rows of 20 pages: chunks of 8 + 8 + 4, so the last chunk of a row is a partial one

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p_teeth2_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def mutant(self, kind, source_path, compile_args, old, new, eager):
        path = mutate(source_path, self.directory, old, new, 'mutant_%s_%d.cpp' % (kind, abs(hash((old, new)))))
        return ref.build_emulation(kind, compile_args, self.directory, eager=eager, kernel_file=path, strict=False)

    def gather_verdict(self, exe):
        """25 blocks over 16 cores (nine cores take two), a stride with a gap, an offset that is 64 but not 1088 aligned, noise in the staging."""
        rng = np.random.default_rng(9)
        page, ppb, slices, stride, offset = 1088, 16, 40, 17408 + 192, 192
        blocks = [int(b) for b in rng.permutation(slices)[:25]]
        slice_bytes = page * ppb
        tensor = rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8)
        staging_in = rng.integers(0, 256, offset + 24 * stride + slice_bytes + 256, dtype=np.uint8)
        try:
            out, line = ref.run_gather_emulation(self.directory, page, ppb, 2, 4096, offset, staging_in.tobytes(), slices * ppb, None, blocks, tensor, row_stride=stride, exe=exe)
        except RuntimeError:
            return 'violation'
        want = staging_in.copy()
        for i, block in enumerate(blocks):
            want[offset + i * stride:offset + i * stride + slice_bytes] = tensor[block * ppb:(block + 1) * ppb].reshape(-1)
        return 'ok' if np.array_equal(np.frombuffer(out, dtype=np.uint8), want) and 'stray=0' in line else 'wrong'

    def digest_verdict(self, exe):
        """100 blocks over 80 cores (twenty cores take two), 3 tensors, rows of 20 pages."""
        rng = np.random.default_rng(10)
        page, ppb, count, slices = 1088, 20, 3, 100
        tensors = [rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8) for _ in range(count)]
        blocks = [int(b) for b in rng.permutation(slices)]
        try:
            got, line = ref.run_digest_emulation(self.directory, page, ppb, tensors, blocks, exe=exe)
        except RuntimeError:
            return 'violation'
        return 'ok' if got == ref.digest_blocks(tensors, blocks, ppb) and 'stray=0' in line else 'wrong'

    def verdicts(self, kind, source, args, old, new, runner):
        return [runner(self.mutant(kind, source, args, old, new, eager)) for eager in (False, True)]

    def test_the_unbroken_kernels_pass_under_both_schedules(self):
        self.assertEqual(self.verdicts('gather', GATHER, self.GATHER_ARGS, 'half ^= 1u;', 'half ^= 1u;', self.gather_verdict), ['ok', 'ok'])
        self.assertEqual(self.verdicts('digest_rows', DIGEST, self.DIGEST_ARGS, 'const uint32_t half = i & 1u;', 'const uint32_t half = i & 1u;', self.digest_verdict), ['ok', 'ok'])

    def test_gather_mutants_are_caught(self):
        mutants = {
            'reads into the half the pending writes still use': ('const uint32_t other = scratch + (half ^ 1u) * chunk_pages * page_bytes;',
                                                                  'const uint32_t other = scratch + half * chunk_pages * page_bytes;'),
            'the write barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                             '        noc_async_read_barrier();\n        if (next >= total) {'),
            'the read barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                            '        noc_async_write_barrier();\n        if (next >= total) {'),
            'the first chunk reads a source page off by one': ('noc_async_read(source.get_noc_addr(block * pages_per_block + k % pages_per_block)',
                                                               'noc_async_read(source.get_noc_addr(block * pages_per_block + k % pages_per_block + 1)'),
            'the next chunk reads a source page off by one': ('noc_async_read(source.get_noc_addr(block * pages_per_block + page % pages_per_block), other',
                                                              'noc_async_read(source.get_noc_addr(block * pages_per_block + page % pages_per_block + 1), other'),
            'the first chunk reads the row of the previous block': ('noc_async_read(source.get_noc_addr(block * pages_per_block + k % pages_per_block)',
                                                                    'noc_async_read(source.get_noc_addr((block ? block - 1 : 0) * pages_per_block + k % pages_per_block)'),
            'the staging offset ignores the core start': ('qwen_p2p_destination(destination_offset, row_stride, page, pages_per_block, page_bytes)',
                                                          'qwen_p2p_destination(0, row_stride, page, pages_per_block, page_bytes)'),
            'a half page is written': ('pages_per_block, page_bytes)),\n                page_bytes);', 'pages_per_block, page_bytes)),\n                page_bytes / 2);'),
            'the stride is ignored (rows are written back to back)': ('(page / pages_per_block) * row_stride', '(page / pages_per_block) * pages_per_block * page_bytes'),
            'the stride is applied to every page, not every block': ('(page / pages_per_block) * row_stride', 'page * row_stride'),
            'the stride is one alignment unit short': ('(page / pages_per_block) * row_stride', '(page / pages_per_block) * (row_stride - 64)'),
            'the page inside the row is dropped (every page of a block lands on the first)': ('+ (page % pages_per_block) * page_bytes;', '+ 0 * page_bytes;'),
            'the stride is read from the wrong argument': ('const uint32_t row_stride = get_arg_val<uint32_t>(5);', 'const uint32_t row_stride = get_arg_val<uint32_t>(3);'),
            'the block table is indexed by the page, not the block': ('const uint32_t block = get_arg_val<uint32_t>(block_ids + k / pages_per_block);',
                                                                      'const uint32_t block = get_arg_val<uint32_t>(block_ids + k % 32);'),
            'the block table starts one word early (the stride word is read as a block id)': ('constexpr uint32_t block_ids = 6;', 'constexpr uint32_t block_ids = 5;'),
            'the next chunk reads from the wrong source page': ('const uint32_t page = next + k;', 'const uint32_t page = next + k + 1;'),
            'the rows go to the wrong bank': ('staging.get_noc_addr(bank, qwen_p2p_destination', 'staging.get_noc_addr((bank + 1) % 8, qwen_p2p_destination'),
            'the source and staging addresses are swapped': ('const uint32_t source_address = get_arg_val<uint32_t>(0);\n    const uint32_t staging_address = get_arg_val<uint32_t>(1);',
                                                             'const uint32_t source_address = get_arg_val<uint32_t>(1);\n    const uint32_t staging_address = get_arg_val<uint32_t>(0);'),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                verdicts = self.verdicts('gather', GATHER, self.GATHER_ARGS, old, new, self.gather_verdict)
                self.assertNotEqual(set(verdicts), {'ok'}, 'the emulation accepted: ' + name)

    def test_digest_mutants_are_caught(self):
        mutants = {
            'k restarts at each tensor (the weight)': ('            sum = qwen_p2p_accumulate(\n', '            if (i % chunks_per_row == 0) { weight = 1; }\n            sum = qwen_p2p_accumulate(\n'),
            'the sum restarts at each tensor': ('            sum = qwen_p2p_accumulate(\n', '            if (i % chunks_per_row == 0) { sum = 0; }\n            sum = qwen_p2p_accumulate(\n'),
            'the weight restarts at each chunk': ('            sum = qwen_p2p_accumulate(\n', '            weight = 1;\n            sum = qwen_p2p_accumulate(\n'),
            'the step constant is wrong': ('constexpr uint32_t kQwenP2PStep = 2654435761u;', 'constexpr uint32_t kQwenP2PStep = 2654435769u;'),
            'the first weight is 0, not 1': ('uint32_t weight = 1;', 'uint32_t weight = 0;'),
            'the last tensor is dropped': ('const uint32_t chunks = tensors * chunks_per_row;', 'const uint32_t chunks = (tensors - 1) * chunks_per_row;'),
            'the partial goes one page too far': ('partials.get_noc_addr(first_slot + b)', 'partials.get_noc_addr(first_slot + b + 1)'),
            'the first partial page of the core is ignored': ('partials.get_noc_addr(first_slot + b)', 'partials.get_noc_addr(b)'),
            'the high word is not stored': ('out[1] = static_cast<uint32_t>(sum >> 32);', 'out[1] = 0;'),
            'the high word is shifted by 31': ('out[1] = static_cast<uint32_t>(sum >> 32);', 'out[1] = static_cast<uint32_t>(sum >> 31);'),
            'the low word is not stored': ('out[0] = static_cast<uint32_t>(sum);', 'out[0] = 0;'),
            'the tensor address index is off by one': ('get_arg_val<uint32_t>(tensor_addresses + next / chunks_per_row)', 'get_arg_val<uint32_t>(tensor_addresses + next / chunks_per_row + 1)'),
            'the first chunk reads tensor 1': ('TensorAccessor(tensor_args, get_arg_val<uint32_t>(tensor_addresses), page_bytes), block, 0,',
                                               'TensorAccessor(tensor_args, get_arg_val<uint32_t>(tensor_addresses + 1), page_bytes), block, 0,'),
            'every block uses the first block id of the core': ('const uint32_t block = get_arg_val<uint32_t>(block_ids + b);', 'const uint32_t block = get_arg_val<uint32_t>(block_ids);'),
            'a chunk is not awaited': ('const uint32_t half = i & 1u;\n            noc_async_read_barrier();', 'const uint32_t half = i & 1u;'),
            'the next chunk is read into the half being summed': ('scratch + (half ^ 1u) * half_bytes);', 'scratch + half * half_bytes);'),
            'the sum reads the other half': ('reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + half * half_bytes)', 'reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (half ^ 1u) * half_bytes)'),
            'the last word of a chunk is dropped': ('(pages * page_bytes) >> 2, weight, sum);', '((pages * page_bytes) >> 2) - 1, weight, sum);'),
            'the word count is a byte count': ('(pages * page_bytes) >> 2, weight, sum);', '(pages * page_bytes), weight, sum);'),
            'a partial chunk is read as a full one': ('const uint32_t count = pages_per_block - first_page < chunk_pages ? pages_per_block - first_page : chunk_pages;',
                                                      'const uint32_t count = chunk_pages;'),
            'the partial page is not awaited before it is reused': ('        noc_async_write_barrier();  // the page is reused by the next block\n', ''),
            'the weight and the sum carry over from one block to the next': ('uint32_t weight = 1;  // w_0: the weight of word 0 of the shard\n        uint64_t sum = 0;',
                                                                              'static uint32_t weight_keep = 1;\n        static uint64_t sum_keep = 0;\n        uint32_t& weight = weight_keep;\n        uint64_t& sum = sum_keep;'),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                verdicts = self.verdicts('digest_rows', DIGEST, self.DIGEST_ARGS, old, new, self.digest_verdict)
                self.assertNotEqual(set(verdicts), {'ok'}, 'the emulation accepted: ' + name)

    def test_the_two_digest_arithmetic_blocks_are_the_checksums(self):
        """The digest kernel carries the checksum kernel's arithmetic between the markers, character for character: what the oracle computes, the digest computes."""
        def block(path):
            text = read(path)
            return text[text.index('// BEGIN-CHECKSUM-ARITHMETIC'):text.index('// END-CHECKSUM-ARITHMETIC')]
        self.assertEqual(block(DIGEST), block(CHECKSUM))
        self.assertIn('constexpr uint32_t kQwenP2PStep = 2654435761u;', block(DIGEST))


class SourceAgreementTests(unittest.TestCase):
    OPS = ('qwen_p2p_staging_alloc', 'qwen_p2p_staging_info', 'qwen_blocks_to_staging', 'qwen_write_blocks_from_staging', 'qwen_blocks_digest_async', 'qwen_p2p_record_event',
           'qwen_p2p_warm', 'qwen_p2p_freeze', 'qwen_staging_checksum', 'qwen_p2p_staging_write', 'qwen_p2p_staging_read')

    def parameters(self, op):
        text = read(EXTENSION)
        match = re.search(r'm\.def\(\s*"%s",' % op, text)
        self.assertIsNotNone(match, op)
        end = min(position for position in (text.find('m.def(', match.end()), text.find('m.attr(', match.end())) if position >= 0)
        return re.findall(r'nb::arg\("(\w+)"\)(?: = ([^,]+),)?', text[match.end():end])

    def names(self, op):
        return [name for name, _ in self.parameters(op)]

    def function_body(self, text, header, until):
        start = text.index(header)
        return text[start:text.index(until, start)]

    def test_the_extension_registers_the_contract_names_and_arguments(self):
        text = read(EXTENSION)
        for name in self.OPS:
            self.assertEqual(len(re.findall(r'm\.def\(\s*"%s"' % name, text)), 1, name)
            self.assertIn('"%s"' % name, text.split('for (const char* name : {')[1], name + ' is also exported on ttnn')
        self.assertIn('m.attr("QWEN_KV_READ_VERSION") = "4";', text)
        self.assertIn('#include "qwen_p2p_plan.hpp"', text)

    def test_the_signatures_are_the_frozen_interface(self):
        """INTERFACE.md 3.3: the order of the parameters is the contract (the worker passes them positionally)."""
        self.assertEqual(self.names('qwen_p2p_staging_alloc')[:3], ['mesh', 'bytes_per_bank', 'bottom_up'])
        self.assertEqual(self.names('qwen_blocks_to_staging'), ['tensor', 'staging', 'bank', 'byte_offset', 'blocks', 'row_stride', 'pages_per_block', 'cq_id'])
        self.assertEqual(self.names('qwen_write_blocks_from_staging'), ['tensor', 'staging', 'bank', 'byte_offset', 'blocks', 'row_stride', 'pages_per_block', 'cq_id'])
        self.assertEqual(self.names('qwen_blocks_digest_async'), ['tensors', 'blocks', 'pages_per_block', 'cq_id'])
        self.assertEqual(self.names('qwen_p2p_record_event'), ['mesh', 'cq_id'])
        self.assertEqual(self.names('qwen_p2p_warm')[:3], ['staging', 'tensors', 'null_block'])
        self.assertEqual(self.names('qwen_p2p_freeze'), ['staging'])
        self.assertEqual(self.names('qwen_p2p_staging_info'), ['staging'])
        # the oracles keep the v3 signatures
        self.assertEqual(self.names('qwen_staging_checksum'), ['staging', 'byte_offset', 'length', 'bank', 'cores', 'count'])
        defaults = dict(self.parameters('qwen_blocks_to_staging'))
        self.assertEqual(defaults['cq_id'], '0')
        self.assertEqual(dict(self.parameters('qwen_p2p_staging_alloc'))['bottom_up'], 'false')

    def test_every_queue_is_named_and_nothing_uses_the_removed_device_list(self):
        text = read(EXTENSION)
        self.assertNotIn('mesh_command_queue()', text, 'an explicit command queue everywhere: the default is the calling thread\'s, not CQ0')
        self.assertNotIn('get_devices()', text, 'MeshDevice::get_devices() is removed after 2026-11-19: use MeshCoordinateRange + the view')
        self.assertIn('get_view().get_devices(MeshCoordinateRange(', text)
        self.assertIn('CreateProgram(mesh)', text, 'the program is bound to the mesh\'s own MetalEnv')
        self.assertNotIn('CreateProgram()', text)
        self.assertNotIn('MeshDeviceConfig', text, 'never CQ1: the extension opens no device and no second queue')
        self.assertNotIn('open_mesh_device', text)

    def test_every_op_validates_before_it_enqueues_anything(self):
        text = read(EXTENSION)
        rows = self.function_body(text, 'void rows_op(', '// qwen_blocks_digest_async. No GIL needed.')
        before = rows[:rows.index('enqueue_program(')]
        for call in ('view_tensor(', 'check_queue_usable(', 'check_bank(', 'check_block_rows(', 'check_block_ids(', 'program_for('):
            self.assertIn(call, before, call)
        digest = self.function_body(text, 'DigestHandle digest_blocks(', '// qwen_p2p_record_event:')
        before = digest[:digest.index('enqueue_program(')]
        for call in ('view_tensor(', 'check_queue_usable(', 'check_digest_rows(', 'check_block_ids(', 'program_for(', 'set_args('):
            self.assertIn(call, before, call)
        self.assertEqual(digest.count('enqueue_program('), 1)
        self.assertEqual(rows.count('enqueue_program('), 1, 'one launch per call: the only way to leave half an op in the queue is that enqueue throwing')

    def test_the_two_exception_classes_say_whether_anything_was_enqueued(self):
        text = read(EXTENSION)
        self.assertIn('nb::exception<qwen_p2p_host::P2PRefused>(m, "QwenP2PRefused", PyExc_ValueError);', text)
        self.assertIn('nb::exception<qwen_p2p_host::P2PEnqueueError>(m, "QwenP2PEnqueueError", PyExc_RuntimeError);', text)
        enqueue = self.function_body(text, 'void enqueue_program(', '// What an op needs to know about one KV tensor')
        self.assertIn('catch (const std::exception& error)', enqueue)
        self.assertIn('throw P2PEnqueueError(', enqueue)
        self.assertNotIn('P2PRefused', enqueue, 'a throw from the enqueue is never reported as a refusal')
        self.assertIn('QwenP2PRefused', text.split('for (const char* name : {')[1])
        self.assertIn('QwenP2PEnqueueError', text.split('for (const char* name : {')[1])
        digest = self.function_body(text, 'DigestHandle digest_blocks(', '// qwen_p2p_record_event:')
        self.assertIn('the sequence is half-enqueued', digest, 'a throw after the digest program was enqueued says so')

    def test_a_trace_capture_in_progress_is_refused_by_every_op(self):
        text = read(EXTENSION)
        body = self.function_body(text, 'void check_queue_usable(', 'std::vector<MeshCoordinate> mesh_coordinates(')
        self.assertIn('trace_id().has_value()', self.function_body(text, 'bool trace_active(', 'void check_queue_usable('))
        self.assertIn('refused while a trace is being captured', body)
        self.assertIn('if (cq_id != 0) {', body, 'only CQ0: any other queue is refused (never CQ1)')
        self.assertIn('trace_active(mesh, 0)', body, 'the capture check is on CQ0')
        for header, until in (('void rows_op(', '// qwen_blocks_digest_async. No GIL needed.'), ('DigestHandle digest_blocks(', '// qwen_p2p_record_event:'),
                              ('P2PEvent record_event(', '// Blocks until every command'), ('void warm(', '// qwen_p2p_freeze:'),
                              ('staging_checksum(\n', '// Diagnostics (card tests only)'), ('void staging_transfer(', 'struct BankPoint')):
            self.assertIn('check_queue_usable(', self.function_body(text, header, until), header)
        self.assertIn('qwen_p2p_host::check_queue_usable(mesh, 0, op);', text, 'the allocation too')

    def test_the_digest_reads_back_without_blocking_and_nothing_spins_a_core(self):
        text = read(EXTENSION)
        digest = self.function_body(text, 'DigestHandle digest_blocks(', '// qwen_p2p_record_event:')
        self.assertIn('queue.enqueue_read_shards(transfers, partials, /*blocking=*/false);', digest)
        self.assertIn('enqueue_record_event_to_host()', digest)
        self.assertNotIn('/*blocking=*/true', digest)
        self.assertNotIn('EventSynchronize', text, 'event_synchronize busy-spins a core (distributed.cpp): the host polls EventQuery')
        self.assertNotIn('event_synchronize', text)
        self.assertIn('tt::tt_metal::distributed::EventQuery(', text)
        self.assertIn('qwen_p2p::kDigestRingBuffers', text, 'two partials buffers take turns')
        self.assertIn('state.inflight.push_back(pending);', digest, 'the host buffer outlives its handle until the event completes')

    def test_release_waits_for_the_queue_within_a_bound_and_leaks_rather_than_free_what_is_in_use(self):
        text = read(EXTENSION)
        release = self.function_body(text, 'void release() {', 'bool released() const')
        self.assertIn('enqueue_record_event_to_host()', release)
        self.assertIn('std::chrono::seconds(10)', release, 'a hung device must not hang the shutdown')
        self.assertIn('leaked_buffers', release)
        self.assertIn('leaked_pending', release)
        self.assertIn('(void)entry.second.release();', release, 'a program whose queue did not drain is leaked, never destroyed under a running dispatch')

    def test_warm_runs_every_program_to_completion_and_freeze_checks_it(self):
        text = read(EXTENSION)
        warm = self.function_body(text, 'void warm(', '// qwen_p2p_freeze:')
        for stage in ('kGatherOp', 'digest_blocks(', 'kScatterOp'):
            self.assertIn(stage, warm, stage)
        self.assertLess(warm.index('kGatherOp'), warm.index('digest_blocks('))
        self.assertLess(warm.index('digest_blocks('), warm.index('kScatterOp'), 'gather -> digest -> scatter-back')
        self.assertGreaterEqual(warm.count('finish_queue('), 3, 'a blocking finish after each stage')
        self.assertIn('program->completed = true', warm)
        freeze = self.function_body(text, 'void freeze(', '// ttnn.qwen_staging_checksum (ORACLE')
        self.assertIn('!entry.second->completed', freeze)
        self.assertIn('state.frozen = true', freeze)
        program_for = self.function_body(text, 'CachedProgram& program_for(', '// Enqueues the program non-blocking')
        self.assertIn('if (state.frozen)', program_for)
        self.assertIn('refuse(op,', program_for)

    def test_the_checksum_makes_several_launches_for_more_ranges_than_the_partials_hold(self):
        text = read(EXTENSION)
        body = self.function_body(text, 'staging_checksum(\n', '// Diagnostics (card tests only)')
        self.assertIn('qwen_p2p::checksum_launches(count)', body)
        self.assertIn('enqueue_read_shards(transfers, partials, /*blocking=*/true)', body, 'each launch is read back before the next reuses the partials buffer')
        self.assertIn('(uint64_t(range) * cores + core) * qwen_p2p::kChecksumPartialBytes', body, 'the partial of range r, core c is partials page r * cores + c')
        self.assertIn('uint64_t(qwen_p2p::kChecksumMaxRanges) * qwen_p2p::kChecksumMaxCores * qwen_p2p::kChecksumPartialBytes', text,
                      'the partials buffer holds one page per core of the largest launch')

    def test_the_default_staging_page_is_64_kib(self):
        self.assertIn('nb::arg("page_bytes") = uint64_t(64) << 10', read(EXTENSION))

    def test_runtime_args_are_rewritten_from_the_live_buffers_on_every_call(self):
        text = read(EXTENSION)
        for header, until in (('void rows_op(', '// qwen_blocks_digest_async. No GIL needed.'), ('DigestHandle digest_blocks(', '// qwen_p2p_record_event:')):
            body = self.function_body(text, header, until)
            self.assertIn('set_args(', body)
            self.assertNotIn('static ', body.replace('static_cast', '').replace('static std::vector<std::shared_ptr<DigestPending>> leaked;', ''), 'no remembered address: ' + header)
        rows = self.function_body(text, 'void rows_op(', '// qwen_blocks_digest_async. No GIL needed.')
        self.assertIn('view.address', rows)
        self.assertIn('static_cast<uint32_t>(state.address)', rows)

    def test_the_kernels_are_whole_page_only_and_nothing_else_touches_dram(self):
        for path in (SCATTER, GATHER, CHECKSUM, DIGEST):
            text = read(path)
            self.assertNotIn('noc_async_read_tile', text)
        for path in (SCATTER, GATHER):
            calls = re.findall(r'noc_async_(read|write)\((.*?)\);', read(path), re.S)
            self.assertEqual(sorted(kind for kind, _ in calls), ['read', 'read', 'write'], path)
            for kind, arguments in calls:
                self.assertTrue(arguments.rstrip().endswith('page_bytes'), 'every transfer moves exactly one page: ' + arguments)
        digest = read(DIGEST)
        calls = re.findall(r'noc_async_(read|write)\((.*?)\);', digest, re.S)
        self.assertEqual(sorted(kind for kind, _ in calls), ['read', 'write'])
        for kind, arguments in calls:
            self.assertTrue(arguments.rstrip().endswith('page_bytes' if kind == 'read' else 'partial_bytes'), arguments)

    def test_the_build_script_ships_the_kernels_and_the_header(self):
        text = read(BUILD)
        for name in ('qwen_p2p_scatter.cpp', 'qwen_p2p_gather.cpp', 'qwen_p2p_checksum.cpp', 'qwen_p2p_digest_rows.cpp', 'qwen_p2p_plan.hpp', 'rv_kernel_check.sh'):
            self.assertIn(name, text, name)
        # the RISC-V front-end check runs on all four kernels, with compile-time args of the right count
        for kernel in ('scatter', 'gather', 'checksum', 'digest_rows'):
            self.assertIn('rv_kernel_check.sh /tmp/kvr %s' % kernel, text, kernel)
        for op in self.OPS:
            self.assertIn(op, text, op)
        self.assertIn('QWEN_KV_READ_VERSION == "4"', text)

    def test_the_riscv_check_uses_the_jit_flags_that_matter(self):
        text = read(os.path.join(HERE, 'rv_kernel_check.sh'))
        for flag in ('-Wall -Werror', '-ftt-no-dyninit', '-mcpu=tt-bh', '-std=c++17', '-DCOMPILE_FOR_BRISC', '-DKERNEL_BUILD', 'riscv-tt-elf-g++', '-flto=1 -flto-partition=one',
                     '--param=min-pagesize=0', '-Os', 'TT_METAL_ROOT', 'SFPI_ROOT'):
            self.assertIn(flag, text, flag)
        self.assertTrue(os.access(os.path.join(HERE, 'rv_kernel_check.sh'), os.X_OK))
        if shutil.which('bash'):
            self.assertEqual(subprocess.run(['bash', '-n', os.path.join(HERE, 'rv_kernel_check.sh')]).returncode, 0)
            self.assertEqual(subprocess.run(['bash', '-n', BUILD]).returncode, 0)

    def test_no_private_string_in_the_new_files(self):
        """The repository is public: no hostnames, addresses, serials, registries, bus addresses or tokens in what this change adds."""
        forbidden = re.compile(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|thatch\.local|zot\.|/home/[a-z]|\bspark-|blackhole-[0-9A-F]{8,}|ghp_|ssh-rsa|BEGIN [A-Z ]*PRIVATE'
                               r'|\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b|/dev/disk/by-id|nvme-[A-Za-z0-9_]+_[A-Z0-9]{8,})')
        names = ['qwen_p2p_scatter.cpp', 'qwen_p2p_gather.cpp', 'qwen_p2p_checksum.cpp', 'qwen_p2p_digest_rows.cpp', 'qwen_p2p_plan.hpp', 'qwen_kv_read.cpp', 'rv_kernel_check.sh', 'build_kv_read.sh',
                 'p2p_staging_ref.py', 'p2p_staging_card.py', 'p2p_staging_mock.py', 'test_p2p_staging_card.py',
                 os.path.join('emu', 'api', 'dataflow', 'dataflow_api.h'), os.path.join('emu', 'emu_common.hpp'), os.path.join('emu', 'emu_scatter.cpp'),
                 os.path.join('emu', 'emu_gather.cpp'), os.path.join('emu', 'emu_checksum.cpp'), os.path.join('emu', 'emu_digest_rows.cpp'),
                 os.path.join('..', '..', '..', 'docs', 'tp4-p2p-restore.md')]
        for name in names:
            path = os.path.join(HERE, name)
            if not os.path.exists(path):
                continue
            for number, line in enumerate(read(path).splitlines(), 1):
                self.assertIsNone(forbidden.search(line), '%s:%d: %s' % (name, number, line))
        # this file holds the pattern itself: check every other line of it
        for number, line in enumerate(read(__file__).splitlines(), 1):
            if 'forbidden = re.compile' not in line and not line.lstrip().startswith("r'|"):
                self.assertIsNone(forbidden.search(line), 'test_p2p_staging.py:%d: %s' % (number, line))


if __name__ == '__main__':
    unittest.main()
