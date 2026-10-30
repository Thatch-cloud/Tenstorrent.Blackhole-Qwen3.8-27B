"""Host-side reference for the P2P restore's device ops (ttnn.qwen_p2p_staging_*, qwen_write_blocks_from_staging, qwen_staging_checksum;
docs/tp4-p2p-restore.md): the checksum algorithm in numpy and in plain Python, the three shared test vectors with their expected values, Python
mirrors of the pure arithmetic in qwen_p2p_plan.hpp, and the runners that compile and run the two device kernels on the CPU against the stand-in
dataflow API (emu/). No device, no ttnn, no torch: numpy only (and a C++ compiler for the emulation).

The checksum (INTERFACE section 4): words x_k are the little-endian uint32 at byte offset 4k of the range, k counted from the START OF THE RANGE
(the object start), w_k = (k * 2654435761 + 1) mod 2^32, S = sum_k w_k * x_k mod 2^64.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 2654435761
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DRAM_ALIGNMENT = 64
BANKS = 8

# The three shared vectors. The expected values were computed with the numpy reference below AND the plain-Python one (two implementations,
# equal), numpy 2.4.6; the third vector depends on numpy's Generator stream for that seed.
VECTOR_ZERO_4096 = 0
VECTOR_MOD251_4096 = 17453227578024978262
VECTOR_PRNG_1MIB = 4942361704597259676
PRNG_SEED = 20261030


def vector_zero():
    return bytes(4096)


def vector_mod251():
    return bytes(i % 251 for i in range(4096))


def vector_prng():
    return np.random.default_rng(PRNG_SEED).integers(0, 256, 1 << 20, np.uint8).tobytes()


def checksum_numpy(data):
    """S of `data` (a multiple of 4 bytes) with numpy; uint64 arithmetic wraps mod 2^64."""
    if len(data) % 4:
        raise ValueError('the checksum covers whole 32-bit words')
    words = np.frombuffer(bytes(data), dtype='<u4').astype(np.uint64)
    k = np.arange(words.size, dtype=np.uint64)
    weights = (k * np.uint64(STEP) + np.uint64(1)) & np.uint64(MASK32)
    return int((weights * words).sum(dtype=np.uint64))


def checksum_python(data):
    """The same S with plain integers (an independent implementation: no numpy wrap-around to trust)."""
    if len(data) % 4:
        raise ValueError('the checksum covers whole 32-bit words')
    total = 0
    for k in range(len(data) // 4):
        word = int.from_bytes(data[4 * k:4 * k + 4], 'little')
        total = (total + ((k * STEP + 1) & MASK32) * word) & MASK64
    return total


# -- Python mirrors of qwen_p2p_plan.hpp (the CPU test compiles the header and checks every function against these) ------------------------

SCATTER_COLUMNS = 8
SCATTER_ROWS = 2
SCATTER_CORES = SCATTER_COLUMNS * SCATTER_ROWS
SCATTER_MAX_BLOCKS_PER_CORE = 32
SCATTER_CHUNK_PAGES = 16
SCATTER_LAUNCH_BLOCKS = SCATTER_CORES * SCATTER_MAX_BLOCKS_PER_CORE
SCATTER_FIXED_ARGS = 5
SCATTER_ARGS = SCATTER_FIXED_ARGS + SCATTER_MAX_BLOCKS_PER_CORE
CHECKSUM_UNIT = 4096
CHECKSUM_CHUNK_BYTES = 32768
CHECKSUM_PARTIAL_BYTES = 64
CHECKSUM_COLUMNS = 8
CHECKSUM_DEFAULT_CORES = 64
CHECKSUM_MAX_CORES = 80
CHECKSUM_ARGS = 7


def scatter_launches(n):
    return [(first, min(n - first, SCATTER_LAUNCH_BLOCKS)) for first in range(0, n, SCATTER_LAUNCH_BLOCKS)]


def split_blocks(n, cores):
    base, extra = divmod(n, cores)
    runs, first = [], 0
    for core in range(cores):
        count = base + (1 if core < extra else 0)
        runs.append((first, count))
        first += count
    return runs


def split_units(units, cores):
    return [(units * core // cores, units * (core + 1) // cores) for core in range(cores)]


def start_weight(k):
    return (k * STEP + 1) & MASK32


def scatter_cb_bytes(page_bytes):
    return 2 * SCATTER_CHUNK_PAGES * page_bytes + 64


def checksum_cb_bytes():
    return 64 + 2 * CHECKSUM_CHUNK_BYTES + CHECKSUM_PARTIAL_BYTES


def scatter_core_args(destination_address, staging_address, bank, byte_offset, slice_bytes, blocks, launch_first, run):
    values = [0] * SCATTER_ARGS
    values[0], values[1], values[2] = destination_address, staging_address, bank
    values[3] = (byte_offset + (launch_first + run[0]) * slice_bytes) & MASK32
    values[4] = run[1]
    for i in range(run[1]):
        values[SCATTER_FIXED_ARGS + i] = blocks[launch_first + run[0] + i]
    return values


def checksum_core_args(staging_address, partials_address, bank, byte_offset, units, slot):
    values = [0] * CHECKSUM_ARGS
    values[0], values[1], values[2] = staging_address, partials_address, bank
    values[3] = (byte_offset + units[0] * CHECKSUM_UNIT) & MASK32
    values[4] = (units[1] - units[0]) * CHECKSUM_UNIT
    values[5] = start_weight(units[0] * CHECKSUM_UNIT // 4)
    values[6] = slot
    return values


def check_scatter(byte_offset, slice_bytes, page_bytes, blocks, staging_bytes):
    """-> None, or the reason the arguments are refused (the C++ check_scatter)."""
    if page_bytes == 0 or page_bytes % DRAM_ALIGNMENT:
        return "the tensor's page size must be a multiple of 64 bytes (a whole DRAM-aligned page)"
    if slice_bytes == 0 or slice_bytes % page_bytes:
        return 'a block must be a whole number of tensor pages'
    if byte_offset % DRAM_ALIGNMENT:
        return 'byte_offset must be a multiple of 64'
    if blocks == 0:
        return 'no blocks'
    if byte_offset > staging_bytes or blocks > (staging_bytes - byte_offset) // slice_bytes:
        return 'the byte range lies outside the staging range'
    return None


def check_checksum(byte_offset, length, staging_bytes, cores):
    if length == 0 or length % CHECKSUM_UNIT:
        return 'length must be a positive multiple of 4096'
    if byte_offset % DRAM_ALIGNMENT:
        return 'byte_offset must be a multiple of 64'
    if byte_offset > staging_bytes or length > staging_bytes - byte_offset:
        return 'the byte range lies outside the staging range'
    if cores == 0 or cores > CHECKSUM_MAX_CORES or cores % CHECKSUM_COLUMNS:
        return 'cores must be a multiple of 8, at most 80'
    return None


def check_staging(size_bytes, page_bytes, bank, num_banks):
    if page_bytes == 0 or page_bytes % DRAM_ALIGNMENT:
        return 'the staging page size must be a multiple of 64'
    if size_bytes == 0 or size_bytes % page_bytes:
        return 'the staging size must be a positive multiple of the staging page size'
    if num_banks == 0 or bank >= num_banks:
        return "the staging bank is outside the device's DRAM banks"
    return None


# -- the object layout (INTERFACE section 1) -------------------------------------------------------------------------------------------------

def object_bytes(per_tensor_arrays):
    """One card's P2P object: for each cache tensor t, data_t[d] (len(blocks) x slice_bytes, C order), concatenated, zero padded to 4096.
    per_tensor_arrays: a list of uint8 arrays [len(blocks), slice_bytes]."""
    flat = b''.join(np.ascontiguousarray(a, dtype=np.uint8).tobytes() for a in per_tensor_arrays)
    return flat + bytes(-len(flat) % 4096)


def tensor_offsets(blocks, slice_bytes, tensors):
    """byte_offset of each cache tensor inside the object: it advances by len(blocks) * slice_bytes per tensor."""
    return [t * blocks * slice_bytes for t in range(tensors)]


def staging_bank_position(byte, page_bytes, bank, banks=BANKS):
    """Where byte `byte` of the bank's staging range sits in the interleaved staging buffer: (logical byte offset in the buffer). The buffer's page p
    is on bank p mod banks, so bank byte x is in page (x // page) * banks + bank."""
    return ((byte // page_bytes) * banks + bank) * page_bytes + byte % page_bytes


# -- running the device kernels on the CPU (emu/) --------------------------------------------------------------------------------------------

def compiler():
    for name in ('g++', 'c++', 'clang++'):
        found = shutil.which(name)
        if found:
            return found
    return None


_BUILT = {}


def build_emulation(kind, compile_args, directory, eager=False, kernel_file=None, strict=True):
    """Compiles emu/emu_<kind>.cpp (which includes the REAL kernel, or kernel_file: a broken copy for the teeth tests) with the given compile-time args ->
    the executable's path. kind: scatter | checksum. eager: NOC transfers happen when issued, else at their barrier (see emu/api/dataflow/dataflow_api.h)."""
    key = (kind, tuple(compile_args), directory, eager, kernel_file)
    if key in _BUILT:
        return _BUILT[key]
    cxx = compiler()
    if cxx is None:
        raise RuntimeError('no C++ compiler')
    name = 'emu_%s_%s%s%s' % (kind, '_'.join(str(a) for a in compile_args), '_eager' if eager else '', '_k%d' % abs(hash(kernel_file)) if kernel_file else '')
    out = os.path.join(directory, name)
    command = [cxx, '-std=c++17', '-O1', '-Wall', '-Wextra'] + (['-Werror'] if strict else []) + ['-Wno-int-to-pointer-cast', '-Wno-unused-variable', '-Wno-unused-function',
               '-Wno-unused-parameter', '-I', os.path.join(HERE, 'emu'), '-DEMU_CT_ARGS=' + ','.join(str(a) for a in compile_args),
               os.path.join(HERE, 'emu', 'emu_%s.cpp' % kind), '-o', out]
    if eager:
        command.insert(-3, '-DEMU_EAGER=1')
    if kernel_file:
        command.insert(-3, '-DEMU_KERNEL_FILE="%s"' % kernel_file)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError('emulation build failed:\n' + done.stderr[-3000:])
    _BUILT[key] = out
    return out


def _write_scenario(path, items):
    with open(path, 'w') as handle:
        for key, values in items:
            handle.write('%s %s\n' % (key, ' '.join(str(int(v)) for v in (values if isinstance(values, (list, tuple)) else [values]))))


def _after(end, gap=4096):
    """The first 64-byte aligned address `gap` bytes past `end`: scenario buffers never overlap (the allocator never lets them)."""
    return -(-(end + gap) // 64) * 64


def run_scatter_emulation(directory, page_bytes, pages_per_block, bank, staging_address, byte_offset, staging_range, dst_pages, dst_address,
                          blocks, dst_in, bank_bytes=None, eager=False):
    """Runs qwen_p2p_scatter.cpp for the blocks. staging_range: the bytes of the staging range from byte 0 (they sit at staging_address in the bank).
    dst_in: uint8 [dst_pages, page_bytes]. -> (dst_out uint8 [dst_pages, page_bytes], stdout line)."""
    exe = build_emulation('scatter', [page_bytes, pages_per_block, SCATTER_CHUNK_PAGES, SCATTER_MAX_BLOCKS_PER_CORE, 0, 0, 0, 0], directory, eager)
    aligned = -(-page_bytes // 64) * 64
    if dst_address is None:
        dst_address = _after(staging_address + len(staging_range))
    elif dst_address < staging_address + len(staging_range):
        raise ValueError('the destination overlaps the staging range')
    if bank_bytes is None:
        bank_bytes = max(dst_address + -(-dst_pages // BANKS) * aligned, staging_address + len(staging_range)) + 4096
    scenario = os.path.join(directory, 'scatter.txt')
    _write_scenario(scenario, [('page_bytes', page_bytes), ('pages_per_block', pages_per_block), ('bank', bank), ('staging_address', staging_address),
                               ('byte_offset', byte_offset), ('dst_address', dst_address), ('dst_pages', dst_pages), ('bank_bytes', bank_bytes),
                               ('blocks', list(blocks))])
    staging_file, in_file, out_file = (os.path.join(directory, n) for n in ('staging.bin', 'dst_in.bin', 'dst_out.bin'))
    with open(staging_file, 'wb') as handle:
        handle.write(bytes(staging_range))
    with open(in_file, 'wb') as handle:
        handle.write(np.ascontiguousarray(dst_in, dtype=np.uint8).tobytes())
    done = subprocess.run([exe, scenario, staging_file, in_file, out_file], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError('emulation exit %d: %s' % (done.returncode, (done.stdout + done.stderr).strip()[-2000:]))
    with open(out_file, 'rb') as handle:
        out = np.frombuffer(handle.read(), dtype=np.uint8).reshape(dst_pages, page_bytes).copy()
    return out, done.stdout.strip()


def run_checksum_emulation(directory, staging_address, partials_address, bank, byte_offset, data, cores=CHECKSUM_DEFAULT_CORES, bank_bytes=None, eager=False):
    """Runs qwen_p2p_checksum.cpp over `data` (placed at byte_offset of the staging range) on `cores` cores -> (S, stdout)."""
    exe = build_emulation('checksum', [CHECKSUM_CHUNK_BYTES, CHECKSUM_PARTIAL_BYTES, 0, 0, 0, 0], directory, eager)
    if partials_address is None:
        partials_address = _after(staging_address + byte_offset + len(data))
    elif partials_address < staging_address + byte_offset + len(data):
        raise ValueError('the partials overlap the staging range')
    if bank_bytes is None:
        bank_bytes = max(staging_address + byte_offset + len(data), partials_address + 64 * cores) + 4096
    scenario = os.path.join(directory, 'checksum.txt')
    _write_scenario(scenario, [('staging_address', staging_address), ('partials_address', partials_address), ('bank', bank),
                               ('byte_offset', byte_offset), ('length', len(data)), ('cores', cores), ('bank_bytes', bank_bytes)])
    staging_file = os.path.join(directory, 'staging_ck.bin')
    with open(staging_file, 'wb') as handle:
        handle.write(bytes(data))
    done = subprocess.run([exe, scenario, staging_file], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError('emulation exit %d: %s' % (done.returncode, (done.stdout + done.stderr).strip()[-2000:]))
    for line in done.stdout.splitlines():
        if line.startswith('SUM '):
            return int(line.split()[1]), done.stdout.strip()
    raise RuntimeError('no SUM line: ' + done.stdout)


def scratch_directory():
    return tempfile.mkdtemp(prefix='p2p_emu_')
