"""CPU tests of the P2P restore's device ops (ttnn.qwen_p2p_staging_alloc / _info, qwen_write_blocks_from_staging, qwen_staging_checksum;
docs/tp4-p2p-restore.md), everything that can be checked without a card:

  * the checksum algorithm: numpy, plain Python and the three shared test vectors (INTERFACE section 4) agree;
  * qwen_p2p_plan.hpp (the pure arithmetic of the extension) is compiled and every function is compared with its Python mirror;
  * the arithmetic block of the checksum kernel is compiled as it stands in the kernel and reproduces the vectors;
  * BOTH DEVICE KERNELS are compiled against a host stand-in of the dataflow API (emu/) and RUN, driven by the same argument builders the
    extension uses: the scatter kernel on random block sets, page sizes, offsets (including the end of the ring), banks and launch splits; the
    checksum kernel on the vectors, offsets and core counts. The stand-in performs NOC transfers late (at the barrier) and refuses a
    sub-page access, an unaligned DRAM address, an access outside the allowed range and a kernel that ends with a transfer pending;
  * the emulation has teeth: deliberately broken copies of each kernel are caught;
  * the extension source, the build script and the card test agree with the contract (names, version, argument names, no private strings).

Run from this directory:  python3 -B -m unittest discover -p 'test_p2p_staging.py'   (numpy and a C++ compiler; python 3.11 or later in CI)
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
#include "%s"
using namespace qwen_p2p;
static void pr(const char* s) { std::printf("%%s\\n", s ? s : "OK"); }
int main() {
  for (uint32_t n : {0u, 1u, 5u, 16u, 17u, 100u, 512u, 513u, 1030u}) {
    std::printf("L %%u:", n);
    for (auto l : scatter_launches(n)) std::printf(" %%u,%%u", l.first, l.second);
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
  std::printf("S "); pr(check_scatter(64, 17408, 1088, 3, 1 << 20));
  std::printf("S "); pr(check_scatter(0, 17408, 1088, 0, 1 << 20));
  std::printf("S "); pr(check_scatter(0, 17000, 1088, 1, 1 << 20));
  std::printf("S "); pr(check_scatter(32, 17408, 1088, 1, 1 << 20));
  std::printf("S "); pr(check_scatter(0, 17408, 1000, 1, 1 << 20));
  std::printf("S "); pr(check_scatter(1 << 20, 17408, 1088, 1, 1 << 20));
  std::printf("S "); pr(check_scatter((1 << 20) - 3 * 17408, 17408, 1088, 3, 1 << 20));
  std::printf("S "); pr(check_scatter((1 << 20) - 3 * 17408, 17408, 1088, 4, 1 << 20));
  std::printf("K "); pr(check_checksum(0, 4096, 8192, 64));
  std::printf("K "); pr(check_checksum(0, 0, 8192, 64));
  std::printf("K "); pr(check_checksum(0, 4000, 8192, 64));
  std::printf("K "); pr(check_checksum(8, 4096, 8192, 64));
  std::printf("K "); pr(check_checksum(4096, 8192, 8192, 64));
  std::printf("K "); pr(check_checksum(0, 4096, 8192, 7));
  std::printf("K "); pr(check_checksum(0, 4096, 8192, 88));
  std::printf("K "); pr(check_checksum(0, 4096, 8192, 0));
  std::printf("G "); pr(check_staging(1 << 20, 4096, 0, 8));
  std::printf("G "); pr(check_staging(1 << 20, 4096, 8, 8));
  std::printf("G "); pr(check_staging(1000, 4096, 0, 8));
  std::printf("G "); pr(check_staging(1 << 20, 100, 0, 8));
  std::printf("G "); pr(check_staging(0, 4096, 0, 8));
  std::vector<uint32_t> blocks; for (uint32_t i = 0; i < 40; ++i) blocks.push_back(1000 - 7 * i);
  auto args = scatter_core_args(111, 222, 5, 4160, 17408, blocks, 0, {3, 4});
  std::printf("A"); for (auto v : args) std::printf(" %%u", v); std::printf("\\n");
  auto cargs = checksum_core_args(111, 222, 5, 4160, {7, 12}, 3);
  std::printf("T"); for (auto v : cargs) std::printf(" %%u", v); std::printf("\\n");
}
''' % PLAN.replace('\\', '/'))
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
                            ('kScatterChunkPages', ref.SCATTER_CHUNK_PAGES), ('kScatterFixedArgs', ref.SCATTER_FIXED_ARGS), ('kChecksumUnit', ref.CHECKSUM_UNIT),
                            ('kChecksumChunkBytes', ref.CHECKSUM_CHUNK_BYTES), ('kChecksumPartialBytes', ref.CHECKSUM_PARTIAL_BYTES),
                            ('kChecksumColumns', ref.CHECKSUM_COLUMNS), ('kChecksumDefaultCores', ref.CHECKSUM_DEFAULT_CORES),
                            ('kChecksumMaxCores', ref.CHECKSUM_MAX_CORES), ('kChecksumArgs', ref.CHECKSUM_ARGS), ('kDramAlignment', ref.DRAM_ALIGNMENT)):
            match = re.search(r'constexpr uint32_t %s = (\d+);' % name, text)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), value, name)
        self.assertIn('constexpr uint64_t kChecksumStep = %dull;' % ref.STEP, text)

    def test_launches_and_runs(self):
        for line in self.of('L'):
            n, rest = line.split(':')
            got = [tuple(int(x) for x in item.split(',')) for item in rest.split()]
            self.assertEqual(got, ref.scatter_launches(int(n)), line)
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
        want = [ref.check_scatter(64, 17408, 1088, 3, 1 << 20), ref.check_scatter(0, 17408, 1088, 0, 1 << 20), ref.check_scatter(0, 17000, 1088, 1, 1 << 20),
                ref.check_scatter(32, 17408, 1088, 1, 1 << 20), ref.check_scatter(0, 17408, 1000, 1, 1 << 20), ref.check_scatter(1 << 20, 17408, 1088, 1, 1 << 20),
                ref.check_scatter((1 << 20) - 3 * 17408, 17408, 1088, 3, 1 << 20), ref.check_scatter((1 << 20) - 3 * 17408, 17408, 1088, 4, 1 << 20)]
        self.assertEqual([line or None for line in (x if x != 'OK' else None for x in self.of('S'))], want)
        self.assertEqual(want[0], None)
        self.assertEqual(want[6], None, 'a run that ends exactly at the end of the ring is fine')
        self.assertIsNotNone(want[7], 'one block past the end is refused')
        want = [ref.check_checksum(0, 4096, 8192, 64), ref.check_checksum(0, 0, 8192, 64), ref.check_checksum(0, 4000, 8192, 64), ref.check_checksum(8, 4096, 8192, 64),
                ref.check_checksum(4096, 8192, 8192, 64), ref.check_checksum(0, 4096, 8192, 7), ref.check_checksum(0, 4096, 8192, 88), ref.check_checksum(0, 4096, 8192, 0)]
        self.assertEqual([None if x == 'OK' else x for x in self.of('K')], want)
        self.assertEqual([want[0]] + [bool(x) for x in want[1:]], [None] + [True] * 7)
        want = [ref.check_staging(1 << 20, 4096, 0, 8), ref.check_staging(1 << 20, 4096, 8, 8), ref.check_staging(1000, 4096, 0, 8),
                ref.check_staging(1 << 20, 100, 0, 8), ref.check_staging(0, 4096, 0, 8)]
        self.assertEqual([None if x == 'OK' else x for x in self.of('G')], want)

    def test_runtime_argument_vectors_are_constant_length_and_match(self):
        blocks = [1000 - 7 * i for i in range(40)]
        args = ref.scatter_core_args(111, 222, 5, 4160, 17408, blocks, 0, (3, 4))
        self.assertEqual(len(args), ref.SCATTER_ARGS)
        self.assertEqual([int(x) for x in self.of('A')[0].split()], args)
        self.assertEqual(args[:5], [111, 222, 5, 4160 + 3 * 17408, 4])
        self.assertEqual(args[5:9], blocks[3:7])
        self.assertTrue(all(v == 0 for v in args[9:]), 'unused block slots are zero')
        cargs = ref.checksum_core_args(111, 222, 5, 4160, (7, 12), 3)
        self.assertEqual(len(cargs), ref.CHECKSUM_ARGS)
        self.assertEqual([int(x) for x in self.of('T')[0].split()], cargs)
        self.assertEqual(cargs, [111, 222, 5, 4160 + 7 * 4096, 5 * 4096, ref.start_weight(7 * 1024), 3])


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

    def scenario(self, rng, page_bytes, pages_per_block, bank, byte_offset, blocks, slices, staging_address=4096, dst_address=None, ring=None):
        slice_bytes = page_bytes * pages_per_block
        needed = byte_offset + len(blocks) * slice_bytes
        ring = needed if ring is None else ring
        staging = rng.integers(0, 256, ring, dtype=np.uint8)
        dst_pages = slices * pages_per_block
        dst_in = rng.integers(0, 256, (dst_pages, page_bytes), dtype=np.uint8)
        out, line = ref.run_scatter_emulation(self.directory, page_bytes, pages_per_block, bank, staging_address, byte_offset, staging.tobytes(), dst_pages,
                                              dst_address, blocks, dst_in, eager=self.eager)
        want = dst_in.copy()
        for i, block in enumerate(blocks):
            start = byte_offset + i * slice_bytes
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
        for page, ppb in ((64, 1), (576, 3), (2048, 32), (1088, 1), (1088, 48), (4096, 5), (32768, 2)):
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

    def run_scatter(self, exe, blocks=(3, 9, 1, 12, 6, 0, 20, 17, 8, 5, 30, 31, 2, 4, 10, 11, 13, 14, 15, 16, 18, 19, 21, 22, 23)):
        rng = np.random.default_rng(9)
        page, ppb, slices = 1088, 16, 40
        staging = rng.integers(0, 256, len(blocks) * page * ppb, dtype=np.uint8)
        dst_in = rng.integers(0, 256, (slices * ppb, page), dtype=np.uint8)
        d = self.directory
        scenario = os.path.join(d, 'm.txt')
        ref._write_scenario(scenario, [('page_bytes', page), ('pages_per_block', ppb), ('bank', 2), ('staging_address', 4096), ('byte_offset', 0), ('dst_address', 1 << 20),
                                       ('dst_pages', slices * ppb), ('bank_bytes', 1 << 22), ('blocks', list(blocks))])
        for name, data in (('s.bin', staging), ('i.bin', dst_in)):
            with open(os.path.join(d, name), 'wb') as handle:
                handle.write(data.tobytes())
        done = subprocess.run([exe, scenario, os.path.join(d, 's.bin'), os.path.join(d, 'i.bin'), os.path.join(d, 'o.bin')], capture_output=True, text=True)
        want = dst_in.copy()
        for i, block in enumerate(blocks):
            want[block * ppb:(block + 1) * ppb] = staging[i * page * ppb:(i + 1) * page * ppb].reshape(ppb, page)
        if done.returncode:
            return 'violation', done.stdout
        with open(os.path.join(d, 'o.bin'), 'rb') as handle:
            out = np.frombuffer(handle.read(), dtype=np.uint8).reshape(slices * ppb, page)
        return ('ok' if np.array_equal(out, want) and 'stray=0' in done.stdout else 'wrong'), done.stdout

    def run_checksum(self, exe, cores=8, units=130):
        data = ref.vector_prng()[:units * 4096]
        d = self.directory
        scenario = os.path.join(d, 'c.txt')
        ref._write_scenario(scenario, [('staging_address', 4096), ('partials_address', 1 << 20), ('bank', 1), ('byte_offset', 64), ('length', len(data)), ('cores', cores),
                                       ('bank_bytes', 1 << 21)])
        with open(os.path.join(d, 'c.bin'), 'wb') as handle:
            handle.write(data)
        done = subprocess.run([exe, scenario, os.path.join(d, 'c.bin')], capture_output=True, text=True)
        if done.returncode:
            return 'violation', done.stdout
        total = int([line for line in done.stdout.splitlines() if line.startswith('SUM')][0].split()[1])
        return ('ok' if total == ref.checksum_numpy(data) else 'wrong'), done.stdout

    def verdicts(self, kind, source, old, new, runner):
        return [runner(self.build_mutant(kind, source, old, new, eager))[0] for eager in (False, True)]

    def test_the_unbroken_kernels_pass_the_same_checks_under_both_schedules(self):
        self.assertEqual(self.verdicts('scatter', SCATTER, 'half ^= 1u;', 'half ^= 1u;', self.run_scatter), ['ok', 'ok'])
        self.assertEqual(self.verdicts('checksum', CHECKSUM, 'half ^= 1u;', 'half ^= 1u;', self.run_checksum), ['ok', 'ok'])

    def test_scatter_mutants_are_caught(self):
        mutants = {
            'reads into the half the pending writes still use': ('const uint32_t other = scratch + (half ^ 1u) * chunk_pages * page_bytes;',
                                                                  'const uint32_t other = scratch + half * chunk_pages * page_bytes;'),
            'the write barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                             '        noc_async_read_barrier();\n        if (next >= total) {'),
            'the read barrier is dropped': ('        noc_async_read_barrier();\n        noc_async_write_barrier();\n        if (next >= total) {',
                                            '        noc_async_write_barrier();\n        if (next >= total) {'),
            'a destination page off by one': ('block * pages_per_block + page % pages_per_block', 'block * pages_per_block + page % pages_per_block + 1'),
            'the staging offset ignores the core start': ('source_offset + k * page_bytes), scratch + k * page_bytes', '0 + k * page_bytes), scratch + k * page_bytes'),
            'a half page is written': ('destination.get_noc_addr(destination_page), page_bytes);', 'destination.get_noc_addr(destination_page), page_bytes / 2);'),
            'the second chunk reads from the wrong source': ('staging.get_noc_addr(bank, source_offset + (next + k) * page_bytes)',
                                                           'staging.get_noc_addr(bank, source_offset + (next + k) * page_bytes + 64)'),
            'the block table is indexed by the page, not the block': ('get_arg_val<uint32_t>(5 + page / pages_per_block)', 'get_arg_val<uint32_t>(5 + page % 32)'),
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
            'the next chunk is read into the half being summed': ('scratch + (half ^ 1u) * chunk_bytes,\n                    next_count);',
                                                                   'scratch + half * chunk_bytes,\n                    next_count);'),
            'the partial is added at the wrong slot': ('partials.get_noc_addr(slot)', 'partials.get_noc_addr(slot + 1)'),
            'the high word is not stored': ('out[1] = static_cast<uint32_t>(sum >> 32);', 'out[1] = 0;'),
            'the weight of the first word is wrong': ('uint32_t weight = get_arg_val<uint32_t>(5);', 'uint32_t weight = get_arg_val<uint32_t>(5) + 1;'),
            'the first chunk is not awaited': ('        while (true) {\n            noc_async_read_barrier();', '        while (true) {'),
        }
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                verdicts = self.verdicts('checksum', CHECKSUM, old, new, self.run_checksum)
                self.assertNotEqual(set(verdicts), {'ok'}, 'the emulation accepted: ' + name)


class SourceAgreementTests(unittest.TestCase):
    def test_the_extension_registers_the_contract_names_and_arguments(self):
        text = read(EXTENSION)
        for name in ('qwen_p2p_staging_alloc', 'qwen_p2p_staging_info', 'qwen_write_blocks_from_staging', 'qwen_staging_checksum', 'qwen_p2p_staging_write',
                     'qwen_p2p_staging_read'):
            self.assertEqual(len(re.findall(r'm\.def\(\s*"%s"' % name, text)), 1, name)
            self.assertIn('"%s"' % name, text.split('for (const char* name : {')[1], name + ' is also exported on ttnn')
        # INTERFACE.md: ttnn.qwen_p2p_staging_alloc(mesh_device, size_bytes); qwen_write_blocks_from_staging(device_tensor, staging, byte_offset, blocks);
        # qwen_staging_checksum(staging, byte_offset, length)
        self.assertRegex(text, r'nb::arg\("mesh_device"\),\s*nb::arg\("size_bytes"\) = uint64_t\(128\) << 20')
        self.assertRegex(text, r'nb::arg\("device_tensor"\),\s*nb::arg\("staging"\),\s*nb::arg\("byte_offset"\),\s*nb::arg\("blocks"\)')
        self.assertRegex(text, r'nb::arg\("staging"\),\s*nb::arg\("byte_offset"\),\s*nb::arg\("length"\)')
        self.assertIn('m.attr("QWEN_KV_READ_VERSION") = "3";', text)
        self.assertIn('#include "qwen_p2p_plan.hpp"', text)

    def test_the_default_staging_is_two_times_64_mib(self):
        self.assertIn('uint64_t(128) << 20', read(EXTENSION))

    def test_runtime_args_are_rewritten_from_the_live_buffers_on_every_call(self):
        text = read(EXTENSION)
        body = text[text.index('void write_blocks_from_staging('):text.index('// ttnn.qwen_staging_checksum')]
        self.assertIn('const uint32_t destination_address = reference->address();', body)
        self.assertIn('set_args(', body)
        self.assertNotIn('static', body.replace('static_cast', ''), 'no remembered address')

    def test_the_kernels_are_whole_page_only_and_nothing_else_touches_dram(self):
        for path in (SCATTER, CHECKSUM):
            text = read(path)
            self.assertNotIn('noc_async_read_tile', text)
            for call in re.findall(r'noc_async_(?:read|write)\(', text):
                self.assertTrue(call)
        scatter = read(SCATTER)
        calls = re.findall(r'noc_async_(read|write)\((.*?)\);', scatter, re.S)
        self.assertEqual(sorted(kind for kind, _ in calls), ['read', 'read', 'write'])
        for kind, arguments in calls:
            self.assertTrue(arguments.rstrip().endswith('page_bytes'), 'every scatter transfer moves exactly one page: ' + arguments)

    def test_the_build_script_ships_the_kernels_and_the_header(self):
        text = read(BUILD)
        for name in ('qwen_p2p_scatter.cpp', 'qwen_p2p_checksum.cpp', 'qwen_p2p_plan.hpp'):
            self.assertIn(name, text, name)
        for op in ('qwen_p2p_staging_alloc', 'qwen_write_blocks_from_staging', 'qwen_staging_checksum', 'qwen_p2p_staging_info'):
            self.assertIn(op, text, op)

    def test_no_private_string_in_the_new_files(self):
        """The repository is public: no hostnames, addresses, serials, registries or tokens in what this change adds."""
        forbidden = re.compile(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|thatch\.local|zot\.|/home/[a-z]|\bspark-|blackhole-[0-9A-F]{8,}|ghp_|ssh-rsa|BEGIN [A-Z ]*PRIVATE)')
        names = ['qwen_p2p_scatter.cpp', 'qwen_p2p_checksum.cpp', 'qwen_p2p_plan.hpp', 'p2p_staging_ref.py', 'p2p_staging_card.py', 'test_p2p_staging_card.py',
                 os.path.join('emu', 'api', 'dataflow', 'dataflow_api.h'), os.path.join('emu', 'emu_common.hpp'), os.path.join('emu', 'emu_scatter.cpp'),
                 os.path.join('emu', 'emu_checksum.cpp'), os.path.join('..', '..', '..', 'docs', 'tp4-p2p-restore.md')]
        for name in names:
            for number, line in enumerate(read(os.path.join(HERE, name)).splitlines(), 1):
                self.assertIsNone(forbidden.search(line), '%s:%d: %s' % (name, number, line))
        # this file holds the pattern itself: check every other line of it
        for number, line in enumerate(read(__file__).splitlines(), 1):
            if 'forbidden = re.compile' not in line:
                self.assertIsNone(forbidden.search(line), 'test_p2p_staging.py:%d: %s' % (number, line))


if __name__ == '__main__':
    unittest.main()
