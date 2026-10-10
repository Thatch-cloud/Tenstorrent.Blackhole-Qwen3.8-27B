"""The REAL kernel source (scripts/ci/draft_permute_tp.cpp) executed on the CPU against a simulated L1 / NoC / DRAM (sim/), over the very runtime arguments the host planner writes.

test_draft_permute_tp runs a Python transliteration of the kernel's loops; this test runs the C++ itself (g++ -shared, then ctypes), so a transliteration that drifted from the source, an operator
precedence slip, a signed shift, an off-by-one at the capacity bound or a missing barrier cannot hide. The simulation obeys the alignment the hardware asks (DRAM side 32 B, L1 side 16 B), queues
NoC transfers until their barrier (a read the kernel uses before its barrier sees junk), and starts every run with junk in L1. The outputs are compared with the served composition's model
(test_draft_permute_tp.CanonOps) bit for bit for the K/V assembly (quad, pair, mixed pair, octo), the fold and unfold (pair, quad, octo) and the fused q|k|v head split, and with the Python
transliteration for the scalar rule (CANON_DENORM=0). Evidence, not proof: the stub is a declaration written from the kernel's call sites; the card-M probe is the proof.

Skipped without g++. Run: python -B -m unittest discover -s optimisation/ttnn-op/draft_permute -p 'test_*.py'
"""

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(CI))
import draft_permute_tp as perm  # noqa: E402
import draft_qkv_tp as qkv  # noqa: E402
import test_draft_permute_tp as support  # noqa: E402
from test_draft_permute_tp import CanonOps, bits, from_raw, tensor_pages, to_raw  # noqa: E402

GXX = shutil.which('g++')
KERNEL = CI / 'draft_permute_tp.cpp'
SIM = HERE / 'sim'
KV, HEADS, DIM = 2, 8, 128
DRAM_BYTES = 192 * 1024 * 1024
_LIBRARIES = {}
_FOLDER = []


def library(capacity, nsrc, ndst, processor=0, denorm=True, kernel=None):
    """The kernel compiled with these compile-time args (capacity, NSRC, NDST, scratch CB) and CANON_DENORM; cached."""
    kernel = kernel or KERNEL
    key = (capacity, nsrc, ndst, processor, denorm, str(kernel))
    if key in _LIBRARIES:
        return _LIBRARIES[key]
    if not _FOLDER:
        _FOLDER.append(tempfile.mkdtemp(prefix='permute-sim-'))
    name = hashlib.sha1(repr(key).encode()).hexdigest()[:12]
    output = os.path.join(_FOLDER[0], 'sim-%s.so' % name)
    command = [GXX, '-std=c++20', '-O1', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', '-I', str(SIM),
               '-DSIM_KERNEL="%s"' % kernel, '-DSIM_CT_ARGS=1,1,%d,%d,%d,%d' % (capacity, nsrc, ndst, processor), '-DCANON_DENORM=%d' % (1 if denorm else 0),
               str(SIM / 'sim.cpp'), '-o', output]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise RuntimeError('the kernel does not compile against the simulation:\n' + done.stderr[-3000:])
    handle = ctypes.CDLL(output)
    handle.sim_run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    handle.sim_run.restype = ctypes.c_int
    _LIBRARIES[key] = handle
    return handle


class Memory(object):
    """The simulated DRAM: tensors at multiples of 4 KiB, a page of 2,048 bytes at base + 2,048 page."""

    def __init__(self):
        self.buffer = ctypes.create_string_buffer(DRAM_BYTES)
        self.next = 4096

    def put(self, pages, count):
        base = self.next
        self.next += (count * 2048 + 8191) // 4096 * 4096
        for page, raw in pages.items():
            data = raw.contiguous().numpy().tobytes()
            self.buffer[base + page * 2048:base + (page + 1) * 2048] = data
        return base

    def reserve(self, count):
        base = self.next
        self.next += (count * 2048 + 8191) // 4096 * 4096
        return base

    def get(self, base, count):
        out = {}
        for page in range(count):
            data = self.buffer[base + page * 2048:base + (page + 1) * 2048]
            out[page] = torch.frombuffer(bytearray(data), dtype=torch.int16).clone()
        return out


def run_native(records, sources, destinations, *, grid=(13, 10), processors=2, denorm=True, junk=1, kernel=None):
    """Plan `records` as the launch builder does and run every lane through the compiled kernel. `sources` is a list of (heads, rows, width) int16 tensors; `destinations` a list of
    (heads, rows, width) shapes. Returns the destination tensors."""
    per_lane, size, rows = perm.plan_lanes(records, grid, len(sources), len(destinations), processors)
    memory = Memory()
    source_bases = [memory.put(tensor_pages(tensor), tensor_pages_count(tensor)) for tensor in sources]
    destination_bases = [memory.reserve(heads * (-(-rows_ // 32)) * (width // 32)) for heads, rows_, width in destinations]
    arguments = perm.runtime_arguments(per_lane, source_bases, destination_bases)
    for lane, words in enumerate(arguments):
        lib = library(size, len(sources), len(destinations), lane % processors, denorm, kernel)
        array = (ctypes.c_uint32 * len(words))(*words)
        errors = lib.sim_run(ctypes.addressof(memory.buffer), DRAM_BYTES, ctypes.addressof(array), len(words), junk + lane)
        if errors:
            raise AssertionError('the simulation reported %d errors on lane %d' % (errors, lane))
    out = []
    for base, (heads, rows_, width) in zip(destination_bases, destinations):
        pages = memory.get(base, heads * (-(-rows_ // 32)) * (width // 32))
        out.append(support.pages_tensor(pages, heads, rows_, width))
    return out


def tensor_pages_count(tensor):
    heads, rows, width = tensor.shape
    return heads * (-(-rows // 32)) * (width // 32)


def same(left, right):
    return tuple(left.shape) == tuple(right.shape) and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


@unittest.skipUnless(GXX, 'no g++')
class NativeKernelTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if _FOLDER:
            shutil.rmtree(_FOLDER.pop(), ignore_errors=True)
            _LIBRARIES.clear()

    def test_the_kernel_compiles_with_werror_against_the_simulation(self):
        self.assertIsNotNone(library(45, 10, 2))
        self.assertIsNotNone(library(45, 10, 2, 1, False))

    def test_the_kv_assembly_runs_as_the_served_composition_for_the_quad_pair_mixed_pair_and_octo_shapes(self):
        for case in support.CASES:
            for grid, processors in (((13, 10), 2), ((11, 10), 1)):
                with self.subTest(case=case, grid=grid):
                    plan, caches, live = support.kv_operands(case, seed=len(case) + 3)
                    users = len(caches)
                    served = support.served_kv(plan, caches, live)
                    rows = dict((user, cache['k'].shape[2]) for user, cache in enumerate(caches))
                    records = []
                    for index in range(2):
                        part, total = perm.kv_records(plan, rows, live['k'].shape[2], KV, destination=index, cached_source=lambda user, index=index: index * (users + 1) + user,
                                                      live_source=index * (users + 1) + users)
                        records.extend(part)
                    sources = []
                    for name in 'kv':
                        sources.extend(cache[name][0] for cache in caches)
                        sources.append(live[name][0])
                    made = run_native(records, sources, [(KV, total, DIM)] * 2, grid=grid, processors=processors)
                    for name, value in zip('kv', made):
                        self.assertTrue(same(value.reshape(1, KV, total, DIM), served[name]), name)

    def test_the_fold_and_the_unfold_run_as_the_served_composition_for_the_pair_quad_and_octo_shapes(self):
        for case in ('pair', 'quad', 'octo'):
            geometry = support.GEOMETRY[case]
            group = HEADS // KV
            folded = KV * geometry['halves'] * geometry['users'] * group
            with self.subTest(case=case, site='fold'):
                query = bits(torch.Generator().manual_seed(3 + len(case)), (1, HEADS, 32 * geometry['halves'], DIM))
                (made,) = run_native(perm.fold_records(KV, group, **geometry), [query[0]], [(folded, 32, DIM)])
                self.assertTrue(same(made.reshape(1, folded, 32, DIM), support.fold_served(case, query)))
            with self.subTest(case=case, site='unfold'):
                output = bits(torch.Generator().manual_seed(5 + len(case)), (1, folded, 32, DIM))
                (made,) = run_native(perm.unfold_records(KV, group, **geometry), [output[0]], [(HEADS, 32 * geometry['halves'], DIM)])
                self.assertTrue(same(made.reshape(1, HEADS, 32 * geometry['halves'], DIM), support.unfold_served(case, output)))

    def test_the_fused_projection_split_places_the_heads_as_nlp_create_qkv_heads_does(self):
        for rows in (32, 64):
            projection = bits(torch.Generator().manual_seed(rows), (1, rows, 1536))
            q, k, v = run_native(qkv.split_records(HEADS, KV, rows), [projection], [(HEADS, rows, DIM), (KV, rows, DIM), (KV, rows, DIM)], grid=(13, 10))
            flat = projection[0]
            for got, low, high, count in ((q, 0, 1024, HEADS), (k, 1024, 1280, KV), (v, 1280, 1536, KV)):
                self.assertTrue(torch.equal(got, flat[:, low:high].reshape(rows, count, DIM).permute(1, 0, 2)), (rows, low))

    def test_the_scalar_rule_of_the_define_off_build_is_the_python_transliteration_and_only_negative_zero_changes(self):
        plan, caches, live = support.kv_operands('pair', seed=7)
        rows = dict((user, cache['k'].shape[2]) for user, cache in enumerate(caches))
        users = len(caches)
        records, total = perm.kv_records(plan, rows, live['k'].shape[2], KV, destination=0)
        sources = [cache['k'][0] for cache in caches] + [live['k'][0]]
        (native,) = run_native(records, sources, [(KV, total, DIM)], denorm=False)
        per_lane, size, _ = perm.plan_lanes(records, (13, 10), len(sources), 1, 2)
        addresses = [100 + index for index in range(len(sources))]
        arguments = perm.runtime_arguments(per_lane, addresses, [900])
        destination = {}
        support.run_kernel(arguments, size, len(sources), 1, dict(zip(addresses, [tensor_pages(source) for source in sources])), {900: destination}, canon_denorm=False)
        self.assertTrue(torch.equal(native, support.pages_tensor(destination, KV, total)))
        full = run_native(records, sources, [(KV, total, DIM)], denorm=True)[0]
        self.assertFalse(torch.equal(native, full), 'the two rules differ on denormals, which the edge data carries')
        both = (native != full)
        self.assertTrue(bool((full[both] == 0).all()), 'the full rule turns the denormals into +0')

    def test_a_list_longer_than_the_capacity_or_a_malformed_record_stops_the_loop_without_a_stray_access(self):
        records = [('run', 0, 0, 0, 0, 3, True)]
        source = bits(torch.Generator().manual_seed(2), (1, 32, 96))
        per_lane, size, _ = perm.plan_lanes(records, (13, 10), 1, 1, 2)
        memory = Memory()
        source_base = memory.put(tensor_pages(source), 3)
        destination_base = memory.reserve(3)
        arguments = perm.runtime_arguments(per_lane, [source_base], [destination_base])[0]
        lib = library(size, 1, 1)
        for name, words in (('claims more words than the capacity', [arguments[0] + 50] + arguments[1:]),
                            ('a record type that is not one', arguments[:3] + [0xF0000000] + arguments[4:]),
                            ('a run cut by the capacity', arguments[:3] + arguments[3:4] + [3] * (size - 4))):
            with self.subTest(name):
                memory.buffer[destination_base:destination_base + 3 * 2048] = b'\xAA' * (3 * 2048)
                array = (ctypes.c_uint32 * len(words))(*words)
                errors = lib.sim_run(ctypes.addressof(memory.buffer), DRAM_BYTES, ctypes.addressof(array), len(words), 3)
                self.assertEqual(errors, 0, 'no out-of-range read, no queued transfer left behind')
        # and the well-formed list works
        array = (ctypes.c_uint32 * len(arguments))(*arguments)
        self.assertEqual(lib.sim_run(ctypes.addressof(memory.buffer), DRAM_BYTES, ctypes.addressof(array), len(arguments), 9), 0)
        got = support.pages_tensor(memory.get(destination_base, 3), 1, 32, 96)
        self.assertTrue(torch.equal(got, support.canon(source)))

    def test_a_kernel_that_used_a_tile_before_its_read_barrier_would_be_caught(self):
        # the simulation queues reads until the barrier: junk fills L1 first, so the output of a correct run is independent of the junk seed
        records = [('run', 0, 0, 0, 0, 3, False), ('mix', 0, 3, ((0, 0, 1), (0, 1, 6), (0, 2, 3), (0, 0, 8)))]
        source = bits(torch.Generator().manual_seed(4), (1, 32, 96))
        outputs = [run_native(records, [source], [(1, 64, 96)], junk=junk)[0] for junk in (1, 2, 77)]
        self.assertTrue(torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[1], outputs[2]))

    def test_the_simulation_has_teeth_a_kernel_without_its_read_barrier_or_with_a_wrong_quarter_offset_is_caught(self):
        text = KERNEL.read_text()
        self.assertEqual(text.count('noc_async_read_barrier();'), 2)
        variants = {'no read barrier in the run loop': text.replace('noc_async_read_barrier();', '', 1),
                    'a quarter offset off by one chunk': text.replace('(quarter & 1) * CHUNK_BYTES', '(quarter & 1) * (CHUNK_BYTES + 16)', 1),
                    'a canonical rule that spares the sign bit': text.replace('0x80008000u', '0x00008000u', 1)}
        source = bits(torch.Generator().manual_seed(11), (1, 64, 96))
        records = [('run', 0, 0, 0, 0, 6, True), ('mix', 0, 6, ((0, 0, 1), (0, 1, 6), (0, 2, 3), (0, 3, 8)))]
        good = run_native(records, [source], [(1, 64, 96)])[0]
        for name, variant in variants.items():
            with self.subTest(name):
                path = Path(_FOLDER[0] if _FOLDER else tempfile.mkdtemp()) / ('variant-%s.cpp' % hashlib.sha1(name.encode()).hexdigest()[:8])
                path.write_text(variant)
                try:
                    bad = run_native(records, [source], [(1, 64, 96)], kernel=path)[0]
                except AssertionError:
                    continue          # the simulation itself flagged it
                self.assertFalse(torch.equal(good, bad), name)


if __name__ == '__main__':
    unittest.main()
